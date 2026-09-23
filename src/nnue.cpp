// -----------------------------------------------------------------------------
// nnue.cpp — NNUE inference: threat-augmented HalfKA network, int8 SIMD play path.
//
// Architecture (matches train_bullet/src/train_threats.rs + threat_inputs.rs):
//   inputs = [ king-bucketed mirrored HalfKA-768 base (768*buckets)
//            | FullThreats threat features (59808)
//            | PP_3Wide pawn-pair features  (4560) ]
//   FT 512, crelu + pairwise-multiply; 8 material output buckets; body 512->L2->32->1.
//
// The threat / pawn-pair feature indexing here is byte-identical to the Rust
// trainer's threat_inputs.rs (validated by train_bullet's check_threats), so a
// net trained there evaluates identically here.
//
// Two paths: the float path (SCN4 nets, used for self-play labelling) and the
// quantised int8 SIMD path (SCN5 nets, the playing format), both driven by the
// incremental accumulator maintained around make/unmake.
//
// SCN4 float net (train_threats exporter):
//   "SCN4", u32{version=4, hl, input_buckets, l2, out_buckets},
//   then u32 len + f32[len] for each of l0w l0b l1w l1b l2w l2b l3w l3b.
//   total_inputs is derived as 768*input_buckets + 59808 + 4560.
//   (SCN4 float -> SCN5 int8 playing net via the net quantiser in the dev tooling.)
// -----------------------------------------------------------------------------

#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <istream>
#include <streambuf>
#include <new>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>   // x86 AVX2 kernels (Haswell+; the CCRL ship target)
#endif

#if defined(_WIN32)
// D2: back the ~35 MB feature-transformer weights (l0w_i8) with 2 MB large pages to cut
// TLB misses on the scattered 512 B feature-row gathers. Only allocations >= 2 MB take
// this path, so the small L1/L2/L3 weight vectors keep operator new. A tiny registry
// records which pointers are large-page-backed so deallocate() frees them correctly.
// The stored weights are byte-identical either way -> bit-identical.
#include <mutex>
#include <unordered_set>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace {
constexpr std::size_t kLargePageThresh = 2u * 1024u * 1024u;
std::mutex             g_lp_mu;
std::unordered_set<void*>& g_lp_set() { static std::unordered_set<void*> s; return s; }
bool sc_enable_lock_pages_nnue() {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok))
        return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount           = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) &&
              GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}
void* sc_nnue_large_alloc(std::size_t bytes) {
    static const bool priv = sc_enable_lock_pages_nnue();
    if (!priv) return nullptr;
    const std::size_t gran = GetLargePageMinimum();
    if (gran == 0) return nullptr;
    const std::size_t rounded = ((bytes + gran - 1) / gran) * gran;
    void* p = VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                           PAGE_READWRITE);
    if (p) { std::lock_guard<std::mutex> lk(g_lp_mu); g_lp_set().insert(p); }
    return p;
}
bool sc_nnue_large_free(void* p) {
    std::lock_guard<std::mutex> lk(g_lp_mu);
    auto& s = g_lp_set();
    auto it = s.find(p);
    if (it == s.end()) return false;
    s.erase(it);
    VirtualFree(p, 0, MEM_RELEASE);
    return true;
}
}  // namespace
#endif

namespace engine::nnue {

namespace {

// ---- Threat feature-set constants ------------------------------------------
constexpr int THREAT_DIMS = 59808;   // FullThreats feature count
constexpr int PP_PAWN_IDS = 2 * 48;  // colours * 48 pawn squares
constexpr int PP_DIMS = PP_PAWN_IDS * (PP_PAWN_IDS - 1) / 2;  // 4560
constexpr int PP_INDEX_BASE = THREAT_DIMS;

// Piece code used by the threat index math: color bit = 8, type = code & 7 in 1..6.
constexpr int SF_W_PAWN = 1;
constexpr int SF_B_PAWN = 9;

constexpr std::array<int, 16> NUM_VALID_TARGETS = {0, 4, 10, 8, 8, 10, 0, 0,
                                                   0, 4, 10, 8, 8, 10, 0, 0};
// map[attackerType-1][attackedType-1], types PAWN..KING = 1..6.
constexpr std::array<std::array<int, 6>, 6> MAP = {{
    {{-1, 0, -1, 1, -1, -1}},
    {{0, 1, 2, 3, 4, -1}},
    {{0, 1, 2, 3, -1, -1}},
    {{0, 1, 2, 3, -1, -1}},
    {{0, 1, 2, 3, 4, -1}},
    {{-1, -1, -1, -1, -1, -1}},
}};

// ---- king-bucket layout (MUST match train_threats.rs BUCKET_LAYOUT) ---------
constexpr std::array<int, 32> BUCKET_LAYOUT = {
    0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
};
constexpr std::array<int, 8> MIRROR = {0, 1, 2, 3, 3, 2, 1, 0};
inline int king_bucket(int sq) { return BUCKET_LAYOUT[(sq / 8) * 4 + MIRROR[sq % 8]]; }

inline int file_of(int sq) { return sq & 7; }
inline int rank_of(int sq) { return sq >> 3; }
inline int orient_tbl(int ksq) { return (file_of(ksq) > 3) ? 7 : 0; }

// ---- geometric attack helpers (raw u64, mirror threat_inputs.rs) ------------
std::uint64_t ray_attacks(int sq, std::uint64_t occ, const int (*deltas)[2], int nd) {
    std::uint64_t bb = 0;
    const int f0 = file_of(sq), r0 = rank_of(sq);
    for (int d = 0; d < nd; ++d) {
        int f = f0 + deltas[d][0], r = r0 + deltas[d][1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            const int s = r * 8 + f;
            bb |= 1ull << s;
            if (occ & (1ull << s)) break;
            f += deltas[d][0];
            r += deltas[d][1];
        }
    }
    return bb;
}
constexpr int ROOK_DIRS[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
constexpr int BISHOP_DIRS[4][2] = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

std::uint64_t knight_attacks(int sq) {
    const int f0 = file_of(sq), r0 = rank_of(sq);
    static const int d[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    std::uint64_t bb = 0;
    for (auto& k : d) {
        const int f = f0 + k[0], r = r0 + k[1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
    }
    return bb;
}
std::uint64_t king_attacks(int sq) {
    const int f0 = file_of(sq), r0 = rank_of(sq);
    std::uint64_t bb = 0;
    for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr) {
            if (!df && !dr) continue;
            const int f = f0 + df, r = r0 + dr;
            if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
        }
    return bb;
}
std::uint64_t pawn_attacks(int color, int sq) {  // 0 = white (north), 1 = black
    const int f0 = file_of(sq), r0 = rank_of(sq), dr = (color == 0) ? 1 : -1;
    std::uint64_t bb = 0;
    for (int df = -1; df <= 1; df += 2) {
        const int f = f0 + df, r = r0 + dr;
        if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
    }
    return bb;
}
std::uint64_t pseudo_attacks_type(int pt, int sq) {  // piece type 2..6 = N,B,R,Q,K
    switch (pt) {
        case 2: return knight_attacks(sq);
        case 3: return ray_attacks(sq, 0, BISHOP_DIRS, 4);
        case 4: return ray_attacks(sq, 0, ROOK_DIRS, 4);
        case 5: return ray_attacks(sq, 0, BISHOP_DIRS, 4) | ray_attacks(sq, 0, ROOK_DIRS, 4);
        case 6: return king_attacks(sq);
        default: return 0;
    }
}
std::uint64_t attacks_bb(int pt, int sq, std::uint64_t occ) {
    // Hot path: magic-bitboard slider attacks (same sets as the scalar rays,
    // parity preserved). Knight/king are table lookups.
    switch (pt) {
        case 2: return knight_attacks(sq);
        case 3: return chess::attacks::bishop(chess::Square(sq), chess::Bitboard(occ)).getBits();
        case 4: return chess::attacks::rook(chess::Square(sq), chess::Bitboard(occ)).getBits();
        case 5: return chess::attacks::queen(chess::Square(sq), chess::Bitboard(occ)).getBits();
        case 6: return king_attacks(sq);
        default: return 0;
    }
}
std::uint64_t pseudo_for_index(int sf_piece, int sq) {
    const int pt = sf_piece & 7;
    if (pt == SF_W_PAWN) return pawn_attacks(sf_piece < 8 ? 0 : 1, sq);
    return pseudo_attacks_type(pt, sq);
}
inline int popcount64(std::uint64_t x) { return __builtin_popcountll(x); }

// ---- precomputed threat-index lookup tables (built once at load) ------------
struct Tables {
    std::array<std::array<std::uint32_t, 64>, 16> offsets{};
    std::array<std::array<std::array<std::uint32_t, 2>, 16>, 16> lut1{};
    std::array<std::array<std::array<std::uint8_t, 64>, 64>, 16> lut2{};
    std::array<std::uint64_t, 64> pawn_pair{};
    bool built = false;
};
Tables g_tab;

void build_tables() {
    if (g_tab.built) return;
    Tables& t = g_tab;
    const int all_pieces[12] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};

    std::array<std::uint32_t, 16> piece_off{};
    std::array<std::uint32_t, 16> cum_off{};
    std::uint32_t cumulative_offset = 0;
    for (int piece : all_pieces) {
        const int pt = piece & 7;
        std::uint32_t cpo = 0;
        for (int from = 0; from < 64; ++from) {
            t.offsets[piece][from] = cpo;
            if (pt != SF_W_PAWN)
                cpo += popcount64(pseudo_attacks_type(pt, from));
            else if (from >= 8 && from <= 55)
                cpo += popcount64(pseudo_for_index(piece, from));
        }
        piece_off[piece] = cpo;
        cum_off[piece] = cumulative_offset;
        cumulative_offset += static_cast<std::uint32_t>(NUM_VALID_TARGETS[piece]) * cpo;
    }

    for (int attacker : all_pieces)
        for (int attacked : all_pieces) {
            const bool enemy = (attacker ^ attacked) == 8;
            const int at = attacker & 7, dt = attacked & 7;
            const int map_val = MAP[at - 1][dt - 1];
            const bool semi_excluded = at == dt && (enemy || at != SF_W_PAWN);
            const int color_attacked = (attacked >> 3) & 1;
            const long feature = static_cast<long>(cum_off[attacker])
                + (static_cast<long>(color_attacked) * (NUM_VALID_TARGETS[attacker] / 2) + map_val)
                    * static_cast<long>(piece_off[attacker]);
            const bool excluded = map_val < 0;
            t.lut1[attacker][attacked][0] = excluded ? THREAT_DIMS : static_cast<std::uint32_t>(feature);
            t.lut1[attacker][attacked][1] =
                (excluded || semi_excluded) ? THREAT_DIMS : static_cast<std::uint32_t>(feature);
        }

    for (int piece : all_pieces)
        for (int from = 0; from < 64; ++from) {
            const std::uint64_t attacks = pseudo_for_index(piece, from);
            for (int to = 0; to < 64; ++to) {
                const std::uint64_t mask = (to == 0) ? 0 : ((1ull << to) - 1);
                t.lut2[piece][from][to] = static_cast<std::uint8_t>(popcount64(mask & attacks));
            }
        }

    for (int s = 0; s < 64; ++s) {
        const int f = file_of(s);
        std::uint64_t files = 0;
        for (int ff = f - 1; ff <= f + 1; ++ff)
            if (ff >= 0 && ff < 8)
                for (int r = 0; r < 8; ++r) files |= 1ull << (r * 8 + ff);
        std::uint64_t rank27 = 0;
        for (int r = 1; r < 7; ++r) rank27 |= 0xFFull << (r * 8);
        t.pawn_pair[s] = files & rank27 & ~(1ull << s);
    }

    t.built = true;
}

// make_index on already-oriented squares / already-color-swapped pieces
// (caller folds perspective). Returns THREAT_DIMS for excluded features.
inline std::uint32_t make_threat_index(int attacker, int from, int to, int attacked) {
    return g_tab.lut1[attacker][attacked][from < to ? 1 : 0]
         + g_tab.offsets[attacker][from]
         + g_tab.lut2[attacker][from][to];
}

// ---- net --------------------------------------------------------------------
// Quantisation scales (must match the net quantiser and the trainer's
// save scheme): feature transformer x QA (int8 weights), L1 x QB (int8). The
// int16 accumulator holds sums of int8 weights (activation scale 127).
constexpr int QA = 127;
constexpr int QB = 64;

// Stack-buffer caps for the quant path (no per-eval heap allocation).
constexpr int MAX_HL = 512;    // tight accumulator slots (2 KB, not 4 KB) for a 32 KB L1d; x86 campaign A10.
                               // load() rejects nets with hl > MAX_HL, so this is a hard cap, not a hint.
constexpr int MAX_L2 = 64;
// Fixed-point scale for the float labeler's incremental accumulator. Integer add is
// associative, so incremental == from-scratch bit-exactly; S is large enough that the
// float eval it feeds is byte-identical to the from-scratch float path, and small
// enough that the int32 accumulator never overflows (<=~250 active feats * 2^22 < 2^31).
constexpr int    FX_SHIFT = 22;
constexpr double FX_S     = double(1u << FX_SHIFT);

// 64-byte-aligned storage for the hot weight tensors (x86 campaign G3): std::vector only
// guarantees 16 B, so a 512 B FT row could straddle 9 cache lines instead of 8 and every
// AVX2 row load is unaligned. Same bytes, same indexing -- only the address changes.
template <class T> struct Align64Alloc {
    using value_type = T;
    Align64Alloc() = default;
    template <class U> Align64Alloc(const Align64Alloc<U>&) noexcept {}
    T* allocate(std::size_t n) {
#if defined(_WIN32)
        if (n * sizeof(T) >= kLargePageThresh) {
            if (void* p = sc_nnue_large_alloc(n * sizeof(T))) return static_cast<T*>(p);
        }
#endif
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64}));
    }
    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32)
        if (sc_nnue_large_free(p)) return;
#endif
        ::operator delete(p, std::align_val_t{64});
    }
    template <class U> bool operator==(const Align64Alloc<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const Align64Alloc<U>&) const noexcept { return false; }
};
#if defined(__ARM_NEON)
template <class T> using WVec = std::vector<T>;                    // pre-G3: default (16-B) alignment
#else
template <class T> using WVec = std::vector<T, Align64Alloc<T>>;   // x86 campaign G3: +1.06% (all phases)
#endif

struct Net {
    bool loaded = false;
    bool quant = false;  // false = SCN4 float, true = SCN5 quantised (int8 FT)
    int hl = 0, input_buckets = 0, l2 = 0, out_buckets = 0;
    std::size_t base_dims = 0, total_inputs = 0;
    std::vector<float> l0w, l0b, l1w, l1b, l2b, l3w, l3b;
    WVec<float> l2w;               // 64-B aligned (G3): finish_body reads 32-float rows
    WVec<float> l2w_bm;            // G2: bucket-major [(b*L2 + i)*32 + o] -- 4 KB contiguous per bucket
    // quant path: int8 feature transformer + int8 L1 (body stays float);
    // int16 accumulator + int16 FT bias.
    WVec<std::int8_t> l0w_i8;      // 64-B aligned (G3): 512 B feature rows
    std::vector<std::int16_t> l0b_i;
    // float labeler path: fixed-point FT weights/bias (lround(w*FX_S)) for the
    // incremental int32 accumulator. Built at load for float (SCN4) nets.
    std::vector<std::int32_t> l0w_fx, l0b_fx;
    std::vector<std::int8_t> l1w_i;
    // L1 weights transposed to [bucket*L2 + o][HL] (HL contiguous) for the sdot
    // dot-product path; built from l1w_i on load.
    WVec<std::int8_t> l1w_dot;     // 64-B aligned (G3): 512 B rows per output
    std::vector<float> l3w_t;      // A6: [b*32 + i] = l3w[i*OB + b] -- contiguous per output bucket
    // int8 body: L2 weights as [bucket*32 + o][L2] (input contiguous) and L3 as
    // [bucket*32 + i]; per-layer global scale. Built from l2w/l3w on load (quant net).
    std::vector<std::int8_t> l2w_dot, l3w_dot;
    float l2_scale = 0.0f, l3_scale = 0.0f;
};
Net g_net;
bool g_base_only = false;  // SCNNUE_BASEONLY profiling probe (skip threats+pp)
bool g_no_accum = false;   // SCNNUE_NOACCUM profiling probe (enumerate, skip column adds)
bool g_verify = false;     // SCNNUE_VERIFY: cross-check incremental base vs recompute
int g_squeeze = 0;         // SCNNUE_SQUEEZE=N: fold threat/pp columns into N (cache probe)
bool g_no_update = false;  // SCNNUE_NOUPDATE: skip the threat/pp delta in acc_make (cost probe)
bool g_no_finny = false;   // SCNNUE_NOFINNY: force king moves to full refresh (A/B the finny path)
bool g_prefetch = true;    // SCNNUE_NOPREFETCH: disable FT-row prefetch in acc_make (A/B)
bool g_no_cols = false;    // SCNNUE_NOCOLS: skip apply_diff column ops (isolate enum vs column cost)
bool g_direct = true;      // SCNNUE_NODIRECT: use the old collect+sort+merge-diff delta instead (A/B + fallback)
bool g_l1dense = false;    // SCNNUE_L1DENSE: force the dense float L1 matvec (A/B vs NNZ sparse)
bool g_i8body = false;     // SCNNUE_I8BODY=1: opt into the (lossy, post-hoc) int8 body; default is
                           // the bit-identical float body. Post-hoc int8 flips bestmoves (~3cp noise)
                           // for only ~3% nps; a real int8 body needs quantization-aware retrain (4.4).

template <class Vec>   // std::vector<float> or the 64-B-aligned WVec<float> (G3)
bool read_arr(std::istream& f, Vec& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()),
           static_cast<std::streamsize>(n) * static_cast<std::streamsize>(sizeof(typename Vec::value_type)));
    return static_cast<bool>(f);
}
bool read_i16(std::istream& f, std::vector<std::int16_t>& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n) * 2);
    return static_cast<bool>(f);
}
template <class Vec>   // std::vector<int8_t> or the 64-B-aligned WVec<int8_t> (G3)
bool read_i8(std::istream& f, Vec& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

[[maybe_unused]] inline float crelu(float x) { return std::clamp(x, 0.0f, 1.0f); }
inline float screlu(float x) {
    const float c = std::clamp(x, 0.0f, 1.0f);
    return c * c;
}

}  // namespace

static bool load_stream(std::istream& f) {

    char magic[4];
    f.read(magic, 4);
    if (!f) return false;
    const bool is_scn4 = std::memcmp(magic, "SCN4", 4) == 0;  // float
    const bool is_scn5 = std::memcmp(magic, "SCN5", 4) == 0;  // quantised (int8 FT)
    if (!is_scn4 && !is_scn5) return false;

    std::uint32_t hdr[5];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) return false;
    if ((is_scn4 && hdr[0] != 4) || (is_scn5 && hdr[0] != 5)) return false;  // version

    Net n;
    n.quant = is_scn5;
    n.hl = int(hdr[1]);
    n.input_buckets = int(hdr[2]);
    n.l2 = int(hdr[3]);
    n.out_buckets = int(hdr[4]);
    if (n.hl > MAX_HL || n.l2 > MAX_L2) return false;   // the eval's stack buffers are sized by these caps
    const std::uint32_t HL = hdr[1], IB = hdr[2], L2 = hdr[3], OB = hdr[4];
    n.base_dims = 768ull * IB;
    n.total_inputs = n.base_dims + THREAT_DIMS + PP_DIMS;
    const std::uint32_t TI = static_cast<std::uint32_t>(n.total_inputs);

    bool ok;
    if (is_scn4) {
        ok = read_arr(f, n.l0w, TI * HL)
          && read_arr(f, n.l0b, HL)
          && read_arr(f, n.l1w, OB * L2 * HL)
          && read_arr(f, n.l1b, OB * L2)
          && read_arr(f, n.l2w, OB * 32u * L2)
          && read_arr(f, n.l2b, OB * 32u)
          && read_arr(f, n.l3w, OB * 32u)
          && read_arr(f, n.l3b, OB);
    } else {
        ok = read_i8(f, n.l0w_i8, TI * HL)
          && read_i16(f, n.l0b_i, HL)
          && read_i8(f, n.l1w_i, OB * L2 * HL)
          && read_arr(f, n.l1b, OB * L2)
          && read_arr(f, n.l2w, OB * 32u * L2)
          && read_arr(f, n.l2b, OB * 32u)
          && read_arr(f, n.l3w, OB * 32u)
          && read_arr(f, n.l3b, OB);
    }
    if (!ok) return false;

    if (n.quant) {  // transpose L1 to [bucket*L2 + o][HL] (HL contiguous) for sdot
        const std::size_t OBL2 = static_cast<std::size_t>(OB) * L2;
        n.l1w_dot.resize(OBL2 * HL);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t o = 0; o < L2; ++o)
                for (std::uint32_t i = 0; i < HL; ++i)
                    n.l1w_dot[(bkt * L2 + o) * HL + i] = n.l1w_i[static_cast<std::size_t>(i) * OBL2 + bkt * L2 + o];
        // int8 body layouts + per-layer global scales (see finish_body int8 path).
        float m2 = 0.0f, m3 = 0.0f;
        for (float w : n.l2w) m2 = std::max(m2, std::fabs(w));
        for (float w : n.l3w) m3 = std::max(m3, std::fabs(w));
        n.l2_scale = m2 > 0.0f ? 127.0f / m2 : 1.0f;
        n.l3_scale = m3 > 0.0f ? 127.0f / m3 : 1.0f;
        auto q8 = [](float v) -> std::int8_t {
            return static_cast<std::int8_t>(std::clamp<long>(std::lround(v), -127, 127));
        };
        n.l2w_dot.resize(static_cast<std::size_t>(OB) * 32 * L2);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t o = 0; o < 32; ++o)
                for (std::uint32_t i = 0; i < L2; ++i)
                    n.l2w_dot[(static_cast<std::size_t>(bkt) * 32 + o) * L2 + i] =
                        q8(n.l2w[static_cast<std::size_t>(i) * (OB * 32) + bkt * 32 + o] * n.l2_scale);
        n.l3w_dot.resize(static_cast<std::size_t>(OB) * 32);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t i = 0; i < 32; ++i)
                n.l3w_dot[static_cast<std::size_t>(bkt) * 32 + i] = q8(n.l3w[static_cast<std::size_t>(i) * OB + bkt] * n.l3_scale);
    } else {  // float labeler: fixed-point FT weights/bias for the incremental accumulator
        const std::size_t NW = static_cast<std::size_t>(TI) * HL;
        n.l0w_fx.resize(NW);
        for (std::size_t i = 0; i < NW; ++i)
            n.l0w_fx[i] = static_cast<std::int32_t>(std::lround(double(n.l0w[i]) * FX_S));
        n.l0b_fx.resize(HL);
        for (std::uint32_t j = 0; j < HL; ++j)
            n.l0b_fx[j] = static_cast<std::int32_t>(std::lround(double(n.l0b[j]) * FX_S));
        // eval_float now reads the fixed-point accumulator; the float FT weights/bias
        // are no longer needed -> free them (l0w_fx int32 replaces l0w float: net-neutral).
        std::vector<float>().swap(n.l0w);
        std::vector<float>().swap(n.l0b);
    }

    // L3 weights transposed to [bucket][32] so finish_body reads a contiguous row instead of a
    // stride-OB gather (both net formats; the sequential fma chain is unchanged). x86 campaign A6.
    n.l3w_t.resize(static_cast<std::size_t>(OB) * 32);
    for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
        for (std::uint32_t i = 0; i < 32; ++i)
            n.l3w_t[static_cast<std::size_t>(bkt) * 32 + i] = n.l3w[static_cast<std::size_t>(i) * OB + bkt];
    // L2 weights bucket-major [(b*L2 + i)*32 + o] so the AVX2 matvec walks 4 KB contiguous
    // per bucket instead of 32 rows at a 1 KB stride (same fma per lane, same i order). x86 campaign G2.
    n.l2w_bm.resize(static_cast<std::size_t>(OB) * L2 * 32);
    for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
        for (std::uint32_t i = 0; i < L2; ++i)
            for (std::uint32_t o = 0; o < 32; ++o)
                n.l2w_bm[(static_cast<std::size_t>(bkt) * L2 + i) * 32 + o] =
                    n.l2w[static_cast<std::size_t>(i) * (OB * 32) + bkt * 32 + o];
    build_tables();
    g_base_only = std::getenv("SCNNUE_BASEONLY") != nullptr;
    g_no_accum = std::getenv("SCNNUE_NOACCUM") != nullptr;
    g_verify = std::getenv("SCNNUE_VERIFY") != nullptr;
    g_no_finny = std::getenv("SCNNUE_NOFINNY") != nullptr;
    g_prefetch = std::getenv("SCNNUE_NOPREFETCH") == nullptr;
    g_no_cols = std::getenv("SCNNUE_NOCOLS") != nullptr;
    { const char* q = std::getenv("SCNNUE_SQUEEZE"); g_squeeze = q ? std::atoi(q) : 0; }
    g_no_update = std::getenv("SCNNUE_NOUPDATE") != nullptr;
    g_direct = std::getenv("SCNNUE_NODIRECT") == nullptr;
    g_l1dense = std::getenv("SCNNUE_L1DENSE") != nullptr;
    g_i8body = std::getenv("SCNNUE_I8BODY") != nullptr;
    n.loaded = true;
    g_net = std::move(n);
    return true;
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    return load_stream(f);
}

// Same loader over an in-memory image of a net file (the embedded net, see embed_net.hpp): a
// read-only streambuf over the bytes, so the parser is shared and the two paths cannot diverge.
bool load_memory(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) return false;
    struct membuf : std::streambuf {
        membuf(const char* b, std::size_t n) {
            char* c = const_cast<char*>(b);
            setg(c, c, c + n);
        }
    } buf(static_cast<const char*>(data), size);
    std::istream f(&buf);
    return load_stream(f);
}

bool loaded() noexcept { return g_net.loaded; }

namespace {

// Enumerate the active features (base + threats + pp), invoking add_stm(feat) /
// add_ntm(feat) for each. Shared verbatim by the float and quantised paths so
// the two can never drift. `base` = net.base_dims (offset of the threat block).
template <class AddStm, class AddNtm>
void gather(const Board& board, std::size_t base, AddStm add_stm, AddNtm add_ntm,
            bool with_base = true) {
    const Color stm = board.sideToMove();
    const bool white = (stm == Color::WHITE);
    auto rel = [&](int sq) { return white ? sq : (sq ^ 56); };

    // ---- canonical board (mover = white), mirroring threat_inputs.rs ----
    std::uint64_t occ = 0, by_piece[16] = {0};
    int piece_on[64];
    for (int& p : piece_on) p = 255;
    static constexpr std::array<PieceType::underlying, 6> kTypes = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
    static constexpr std::array<Color, 2> kColors = {Color::WHITE, Color::BLACK};
    for (int t = 0; t < 6; ++t)
        for (Color col : kColors) {
            const int ccolor = (col == stm) ? 0 : 1;   // 0 = mover = white
            const int sf = (ccolor << 3) | (t + 1);
            Bitboard bb = board.pieces(PieceType(kTypes[t]), col);
            while (bb) {
                const int csq = rel(bb.pop());
                occ |= 1ull << csq;
                by_piece[sf] |= 1ull << csq;
                piece_on[csq] = sf;
            }
        }
    // bulletformat stores our_ksq = mover king (canonical), opp_ksq = opponent
    // king canonical square ^ 56 (see ChessBoard::from_str). The trainer's
    // ChessBucketsMirrored + threat orientation both read pos.opp_ksq(), so the
    // engine must apply the same ^56 to opp_ksq to match the ntm king bucket.
    const int our_ksq = rel(board.kingSq(stm).index());
    const int opp_ksq = rel(board.kingSq(white ? Color::BLACK : Color::WHITE).index()) ^ 56;
    const int stm_orient = orient_tbl(our_ksq);
    const int ntm_orient = orient_tbl(opp_ksq) ^ 56;

    // ---- base : king-bucketed mirrored HalfKA-768 (ChessBucketsMirrored) ----
    // Skipped when the incremental accumulator already holds the base sum.
    if (with_base) {
        const int stm_flip = (file_of(our_ksq) > 3) ? 7 : 0;
        const int ntm_flip = (file_of(opp_ksq) > 3) ? 7 : 0;
        const int stm_bucket = 768 * king_bucket(our_ksq);
        const int ntm_bucket = 768 * king_bucket(opp_ksq);
        for (int sf : {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14}) {
            const int c = (sf >> 3) & 1;
            const int t0 = (sf & 7) - 1;
            std::uint64_t bb = by_piece[sf];
            while (bb) {
                const int csq = __builtin_ctzll(bb);
                bb &= bb - 1;
                const int stm = (c == 0 ? 0 : 384) + 64 * t0 + csq;
                const int ntm = (c == 0 ? 384 : 0) + 64 * t0 + (csq ^ 56);
                add_stm(stm_bucket + (stm ^ stm_flip));
                add_ntm(ntm_bucket + (ntm ^ ntm_flip));
            }
        }
    }

    if (g_base_only) return;  // profiling probe

    // ---- threats : FullThreats (accumulate each perspective independently) ----
    auto push_threat = [&](int attacker_abs, int from, int to, int attacked_abs) {
        const std::uint32_t s =
            make_threat_index(attacker_abs, from ^ stm_orient, to ^ stm_orient, attacked_abs);
        if (s < static_cast<std::uint32_t>(THREAT_DIMS)) add_stm(base + (g_squeeze ? (s % g_squeeze) : s));
        const std::uint32_t nn =
            make_threat_index(attacker_abs ^ 8, from ^ ntm_orient, to ^ ntm_orient, attacked_abs ^ 8);
        if (nn < static_cast<std::uint32_t>(THREAT_DIMS)) add_ntm(base + (g_squeeze ? (nn % g_squeeze) : nn));
    };

    const std::uint64_t pawn_targets = by_piece[2] | by_piece[10] | by_piece[4] | by_piece[12];
    for (int color = 0; color < 2; ++color) {
        const int attacker_abs = (color == 0) ? SF_W_PAWN : SF_B_PAWN;
        std::uint64_t bb = by_piece[attacker_abs];
        while (bb) {
            const int from = __builtin_ctzll(bb);
            bb &= bb - 1;
            std::uint64_t a = pawn_attacks(color, from) & pawn_targets;
            while (a) {
                const int to = __builtin_ctzll(a);
                a &= a - 1;
                push_threat(attacker_abs, from, to, piece_on[to]);
            }
        }
    }
    const std::uint64_t nn = by_piece[2] | by_piece[10];
    const std::uint64_t bsh = by_piece[3] | by_piece[11];
    const std::uint64_t rk = by_piece[4] | by_piece[12];
    const std::uint64_t qn = by_piece[5] | by_piece[13];
    const std::uint64_t pn = by_piece[SF_W_PAWN] | by_piece[SF_B_PAWN];
    const std::uint64_t minor_slider_targets = pn | nn | bsh | rk;
    const std::uint64_t queen_targets = pn | nn | bsh | rk | qn;
    for (int color = 0; color < 2; ++color)
        for (int pt = 2; pt <= 5; ++pt) {
            const int sf = (color << 3) | pt;
            std::uint64_t bb = by_piece[sf];
            const std::uint64_t targets = (pt == 2 || pt == 5) ? queen_targets : minor_slider_targets;
            while (bb) {
                const int from = __builtin_ctzll(bb);
                bb &= bb - 1;
                std::uint64_t a = attacks_bb(pt, from, occ) & targets;
                while (a) {
                    const int to = __builtin_ctzll(a);
                    a &= a - 1;
                    push_threat(sf, from, to, piece_on[to]);
                }
            }
        }

    // ---- pp : PP_3Wide (pawn pairs in the 3-wide band) ----
    auto emit_pp = [&](int color_abs, int from, int to, int paired_abs) {
        auto pp_one = [&](int orient, int persp) -> std::size_t {
            const int fo = (from ^ orient), to_o = (to ^ orient);
            const int ca = color_abs ^ persp, cb = paired_abs ^ persp;
            const int id_a = 48 * ca + (fo - 8);
            const int id_b = 48 * cb + (to_o - 8);
            const int hi = std::max(id_a, id_b), lo = std::min(id_a, id_b);
            return static_cast<std::size_t>(hi * (hi - 1) / 2 + lo + PP_INDEX_BASE);
        };
        add_stm(base + (g_squeeze ? (pp_one(stm_orient, 0) % g_squeeze) : pp_one(stm_orient, 0)));
        add_ntm(base + (g_squeeze ? (pp_one(ntm_orient, 1) % g_squeeze) : pp_one(ntm_orient, 1)));
    };
    const std::uint64_t white_pawns = by_piece[SF_W_PAWN];
    const std::uint64_t black_pawns = by_piece[SF_B_PAWN];
    {
        std::uint64_t bb = white_pawns;
        while (bb) {
            const int from = __builtin_ctzll(bb);
            bb &= bb - 1;
            const std::uint64_t band = g_tab.pawn_pair[from];
            std::uint64_t ww = band & bb;
            while (ww) { const int to = __builtin_ctzll(ww); ww &= ww - 1; emit_pp(0, from, to, 0); }
            std::uint64_t wb = band & black_pawns;
            while (wb) { const int to = __builtin_ctzll(wb); wb &= wb - 1; emit_pp(0, from, to, 1); }
        }
        bb = black_pawns;
        while (bb) {
            const int from = __builtin_ctzll(bb);
            bb &= bb - 1;
            const std::uint64_t band = g_tab.pawn_pair[from];
            std::uint64_t bk = band & bb;
            while (bk) { const int to = __builtin_ctzll(bk); bk &= bk - 1; emit_pp(1, from, to, 1); }
        }
    }

}  // gather

// ---- incremental base accumulator (thread-local) ----------------------------
// Persistent per-perspective BASE accumulator (int16), maintained across
// make/unmake so the base sum isn't rebuilt every eval. Only the base block is
// incremental (Stage A); threats+pp are still recomputed in eval_quant.
constexpr int ACC_STACK = MAX_PLY + 8;
struct Acc {
    alignas(64) std::int16_t v[2][MAX_HL];  // v[0]=WHITE perspective, v[1]=BLACK perspective
};
thread_local std::vector<Acc> g_stack;
thread_local int g_ply = -1;  // -1 = uninitialised -> evaluate() recomputes the base

// Fixed-point (int32) accumulator stack for the float labeler path. Shares g_ply and
// the g_bb bitboard stack with the quant path; only one of g_stack / g_stack_fx is
// live per run (chosen by the loaded net format).
struct AccFx {
    alignas(64) std::int32_t v[2][MAX_HL];
};
thread_local std::vector<AccFx> g_stack_fx;

// Absolute-perspective base feature index (persp 0=WHITE, 1=BLACK). color 0/1,
// t0 0..5 (pawn..king), king_abs = absolute king square of persp's side.
inline int base_feat(int persp, int color, int t0, int sq, int king_abs) {
    const int kp = (persp == 0) ? king_abs : (king_abs ^ 56);
    const int bucket = 768 * king_bucket(kp);
    const int flip = (file_of(kp) > 3) ? 7 : 0;
    const int rsq = (persp == 0) ? sq : (sq ^ 56);
    const int idx = ((color == persp) ? 0 : 384) + 64 * t0 + rsq;
    return bucket + (idx ^ flip);
}
// int16 += int8 with explicit NEON widening (the scalar/auto-vec form does not
// vectorize the mixed-width add). 16 int8/iter. hl must be a multiple of 16.
inline void acc_add(std::int16_t* a, const std::int8_t* w, int hl) {
#if defined(__ARM_NEON)
    for (int j = 0; j < hl; j += 16) {
        const int8x16_t wv = vld1q_s8(w + j);
        vst1q_s16(a + j, vaddq_s16(vld1q_s16(a + j), vmovl_s8(vget_low_s8(wv))));
        vst1q_s16(a + j + 8, vaddq_s16(vld1q_s16(a + j + 8), vmovl_s8(vget_high_s8(wv))));
    }
#else
    for (int j = 0; j < hl; ++j) a[j] += w[j];
#endif
}
[[maybe_unused]] inline void acc_sub(std::int16_t* a, const std::int8_t* w, int hl) {
#if defined(__ARM_NEON)
    for (int j = 0; j < hl; j += 16) {
        const int8x16_t wv = vld1q_s8(w + j);
        vst1q_s16(a + j, vsubq_s16(vld1q_s16(a + j), vmovl_s8(vget_low_s8(wv))));
        vst1q_s16(a + j + 8, vsubq_s16(vld1q_s16(a + j + 8), vmovl_s8(vget_high_s8(wv))));
    }
#else
    for (int j = 0; j < hl; ++j) a[j] -= w[j];
#endif
}

// Defined with the other fused kernels below; the refresh uses it for its one-pass form.
inline void fused_apply_src(std::int16_t* dst, const std::int16_t* src, const std::int8_t* const* add, int na,
                            const std::int8_t* const* sub, int ns, int hl);

// Full refresh (base + threats + pp) for a position, both perspectives. Uses
// gather routed to the absolute WHITE/BLACK-perspective accumulators.
void refresh_into(Acc& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int8_t* l0 = n.l0w_i8.data();
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    // Collect every active feature column, then ONE src->dst fused pass per perspective
    // straight from the bias vector -- instead of a bias seed loop plus a 1 KB accumulator
    // read+write per feature. int16 sums are order-independent -> bit-identical.
    // (x86 campaign G1: +1.76%, all phases.)
    static thread_local const std::int8_t* cols[2][1024];
    int nc[2] = {0, 0};
    gather(
        board, n.base_dims,
        [&](std::size_t f) { cols[stmp][nc[stmp]++] = l0 + f * hl; },
        [&](std::size_t f) { cols[1 - stmp][nc[1 - stmp]++] = l0 + f * hl; });
    for (int p = 0; p < 2; ++p)
        fused_apply_src(a.v[p], n.l0b_i.data(), cols[p], nc[p], nullptr, 0, hl);
}

inline void fused_apply_src_fx(std::int32_t* dst, const std::int32_t* src, const std::int32_t* const* add, int na,
                               const std::int32_t* const* sub, int ns, int hl);

// Fixed-point (int32) full refresh for the float labeler path. Mirrors refresh_into:
// collect every active column, then ONE src->dst fused pass per perspective straight
// from the bias vector -- no bias seed loop, no 2 KB accumulator read+write per feature.
// int32 sums are order-independent -> bit-identical. (Self-play campaign FX1: +12.0%.)
void refresh_into_fx(AccFx& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int32_t* l0 = n.l0w_fx.data();
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    static thread_local const std::int32_t* cols[2][1024];
    int nc[2] = {0, 0};
    gather(
        board, n.base_dims,
        [&](std::size_t f) { cols[stmp][nc[stmp]++] = l0 + f * hl; },
        [&](std::size_t f) { cols[1 - stmp][nc[1 - stmp]++] = l0 + f * hl; });
    for (int p = 0; p < 2; ++p)
        fused_apply_src_fx(a.v[p], n.l0b_fx.data(), cols[p], nc[p], nullptr, 0, hl);
}

// ---- incremental THREAT delta helpers ---------------------------------------
// Absolute board state in threat-index piece coding (sf = color<<3 | (type+1)); by[sf],
// occupancy, per-square piece, and king squares. Built from a chess::Board and
// mutated in place to form the after-move state.
struct BB {
    std::uint64_t occ = 0;
    std::uint64_t by[16] = {0};
    std::int8_t piece_on[64];
    int wk = 0, bk = 0;
    BB() = default;  // for the thread_local stack (slots are filled before use)
    explicit BB(const Board& b) {
        for (auto& p : piece_on) p = -1;
        std::uint64_t o = b.occ().getBits();
        while (o) {
            const int sq = __builtin_ctzll(o);
            o &= o - 1;
            const int pi = static_cast<int>(b.at(Square(sq)).internal());  // 0..11
            const int sf = ((pi / 6) << 3) | ((pi % 6) + 1);
            by[sf] |= 1ull << sq;
            piece_on[sq] = static_cast<std::int8_t>(sf);
        }
        occ = b.occ().getBits();
        wk = b.kingSq(Color::WHITE).index();
        bk = b.kingSq(Color::BLACK).index();
    }
};

// Per-ply incremental board bitboards, maintained in lockstep with g_stack so a
// normal move reads its before-state from g_bb[g_ply] instead of rebuilding a BB
// from board.at() every make. Same g_ply / reset / make / unmake / null hooks.
thread_local std::vector<BB> g_bb;

// Item #9 (novelty ledger): how many NEW attacks on a HIGHER-valued enemy piece the last acc_make's
// move created, counted for the mover's side only. The threat-delta emitter already has the attacker
// and the victim in hand for every column it adds, so this is two comparisons per emission rather
// than any new enumeration -- the same "read what the net already computed" shape as item #8, but a
// semantically sharp count (this move makes a threat) instead of a magnitude of feature change.
// Zero on the king-cross refresh path, where the threat set is rebuilt rather than emitted as a
// delta: a false negative on a few percent of moves, never a false positive.
thread_local int g_threats_made = 0;

// Squares of all pieces attacking `s` (both colors).
inline std::uint64_t attackers_of(const BB& bb, int s) {
    const std::uint64_t N = bb.by[2] | bb.by[10], B = bb.by[3] | bb.by[11],
                        R = bb.by[4] | bb.by[12], Q = bb.by[5] | bb.by[13],
                        K = bb.by[6] | bb.by[14];
    std::uint64_t a = 0;
    a |= knight_attacks(s) & N;
    a |= king_attacks(s) & K;
    a |= attacks_bb(3, s, bb.occ) & (B | Q);
    a |= attacks_bb(4, s, bb.occ) & (R | Q);
    a |= bb.by[SF_W_PAWN] & pawn_attacks(1, s);  // white pawns that attack s
    a |= bb.by[SF_B_PAWN] & pawn_attacks(0, s);  // black pawns that attack s
    return a;
}

// A thread-local scratch that collects threat/pp feature indices (in
// [0, THREAT_DIMS+PP_DIMS)) per perspective, so acc_make can apply only the
// exact multiset difference between the before- and after-move feature sets.
struct FeatList {
    std::uint32_t v[2][1024];
    int n[2] = {0, 0};
    inline void push(int p, std::uint32_t idx) { v[p][n[p]++] = idx; }
    inline void clear() { n[0] = n[1] = 0; }
};

// Collect all threat feature indices of the piece `sf` on `sq` (board `bb`) into
// `out`, for both perspectives.
inline void collect_piece_threats(FeatList& out, const BB& bb, int sq, int sf, int wo, int bo) {
    const int pt = sf & 7;
    if (pt == 6 || pt == 0) return;
    const std::uint64_t N = bb.by[2] | bb.by[10], Bp = bb.by[3] | bb.by[11],
                        R = bb.by[4] | bb.by[12], Q = bb.by[5] | bb.by[13],
                        P = bb.by[SF_W_PAWN] | bb.by[SF_B_PAWN];
    std::uint64_t attacks;
    if (pt == 1) {
        attacks = pawn_attacks(sf < 8 ? 0 : 1, sq) & (N | R);
    } else {
        const std::uint64_t tgt = (pt == 2 || pt == 5) ? (P | N | Bp | R | Q) : (P | N | Bp | R);
        attacks = attacks_bb(pt, sq, bb.occ) & tgt;
    }
    while (attacks) {
        const int to = __builtin_ctzll(attacks);
        attacks &= attacks - 1;
        const int attacked = bb.piece_on[to];
        const std::uint32_t s = make_threat_index(sf, sq ^ wo, to ^ wo, attacked);
        if (s < static_cast<std::uint32_t>(THREAT_DIMS)) out.push(0, s);
        const std::uint32_t nn = make_threat_index(sf ^ 8, sq ^ bo, to ^ bo, attacked ^ 8);
        if (nn < static_cast<std::uint32_t>(THREAT_DIMS)) out.push(1, nn);
    }
}

// Reduced collect for a NON-SLIDER attacker (knight or pawn; kings are excluded
// as threat attackers). Their attack set is occupancy-independent, so the only
// threat of theirs that can change when a piece appears/leaves square `tgt` is
// their threat TO `tgt`. Push just that one (both perspectives) if they attack
// `tgt` and the piece there is a valid target. Sliders are NOT routed here — a
// ray extension/truncation changes threats to squares beyond from/to, so they
// still need a full recompute.
inline void collect_ns_threat(FeatList& out, const BB& bb, int a_sq, int a_sf, int tgt,
                              int wo, int bo) {
    const int pt = a_sf & 7;
    const std::uint64_t atk = (pt == 1) ? pawn_attacks(a_sf < 8 ? 0 : 1, a_sq) : knight_attacks(a_sq);
    if (!(atk & (1ull << tgt))) return;
    const int attacked = bb.piece_on[tgt];
    if (attacked < 0) return;
    const std::uint32_t s = make_threat_index(a_sf, a_sq ^ wo, tgt ^ wo, attacked);
    if (s < static_cast<std::uint32_t>(THREAT_DIMS)) out.push(0, s);
    const std::uint32_t nn = make_threat_index(a_sf ^ 8, a_sq ^ bo, tgt ^ bo, attacked ^ 8);
    if (nn < static_cast<std::uint32_t>(THREAT_DIMS)) out.push(1, nn);
}

// Collect all pawn-pair (PP_3Wide) feature indices for the pawn on `sq`.
inline void collect_pp_for_pawn(FeatList& out, const BB& bb, int sq, int color, int wo, int bo) {
    const std::uint64_t wp = bb.by[SF_W_PAWN], bp = bb.by[SF_B_PAWN];
    std::uint64_t band = g_tab.pawn_pair[sq] & (wp | bp);
    auto pp_index = [&](int orient, int persp, int cA, int to, int cB) -> std::uint32_t {
        const int fo = sq ^ orient, to_o = to ^ orient;
        const int ca = cA ^ persp, cb = cB ^ persp;
        const int id_a = 48 * ca + (fo - 8), id_b = 48 * cb + (to_o - 8);
        const int hi = std::max(id_a, id_b), lo = std::min(id_a, id_b);
        return static_cast<std::uint32_t>(hi * (hi - 1) / 2 + lo + PP_INDEX_BASE);
    };
    while (band) {
        const int to = __builtin_ctzll(band);
        band &= band - 1;
        const int cB = (bp & (1ull << to)) ? 1 : 0;
        out.push(0, pp_index(wo, 0, color, to, cB));
        out.push(1, pp_index(bo, 1, color, to, cB));
    }
}

// Apply many int8 weight columns to the int16 accumulator in ONE pass: read each
// 16-wide accumulator chunk once, fold in every add/sub column, write once. This
// amortizes the accumulator load/store traffic over all changed features instead
// of paying it per column (the per-column acc_add/acc_sub did read+write per
// feature). Correctness identical to sequential add/sub (order-independent sums).
inline void fused_apply(std::int16_t* acc, const std::int8_t* const* add, int na,
                        const std::int8_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    // Two independent accumulator lanes per chunk break the serial add-chain
    // dependency (better ILP across the M-series NEON units). int16 add is
    // associative mod 2^16, so any grouping is bit-identical.
    for (int c = 0; c < hl; c += 16) {
        int16x8_t lo = vld1q_s16(acc + c), hi = vld1q_s16(acc + c + 8);
        int16x8_t lo2 = vdupq_n_s16(0), hi2 = vdupq_n_s16(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t w0 = vld1q_s8(add[k] + c), w1 = vld1q_s8(add[k + 1] + c);
            lo  = vaddq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vaddq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vaddq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vaddq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < na; ++k) {
            const int8x16_t w = vld1q_s8(add[k] + c);
            lo = vaddq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vaddq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t w0 = vld1q_s8(sub[k] + c), w1 = vld1q_s8(sub[k + 1] + c);
            lo  = vsubq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vsubq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vsubq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vsubq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t w = vld1q_s8(sub[k] + c);
            lo = vsubq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vsubq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        vst1q_s16(acc + c,     vaddq_s16(lo, lo2));
        vst1q_s16(acc + c + 8, vaddq_s16(hi, hi2));
    }
#elif defined(__AVX2__)
    // AVX2 port of the NEON structure at 32 int16 per chunk (2 x 256-bit), two independent
    // accumulator lanes to break the add chain. int16 add is associative mod 2^16 -> any
    // grouping is bit-identical. hl must be a multiple of 32 (it is 512).
    // Few columns (quiet endgame makes): the fused chunk loop pays 16x the branchy tiny
    // k-loops (0-3 trips each, mispredicted); a column-outer pass with long predictable
    // 32-lane loops is cheaper. Same int16 sums -> bit-identical either way.
    // (x86 campaign A2c: +6.5% on <=6-piece endgames, -3.5% on a few middlegames, +0.44% mean.)
    // K=2: only 1-add/1-sub makes take the column-outer path (x86 campaign A2d; K=4 traded
    // -2.2% on every middlegame for +4.5% on endgames and failed the phase guard).
    constexpr int kSmallCols = 2;
    if (na + ns <= kSmallCols) {
        for (int k = 0; k < na; ++k) {
            const std::int8_t* w = add[k];
            for (int c = 0; c < hl; c += 32) {
                __m256i* p0 = reinterpret_cast<__m256i*>(acc + c);
                __m256i* p1 = reinterpret_cast<__m256i*>(acc + c + 16);
                _mm256_storeu_si256(p0, _mm256_add_epi16(_mm256_loadu_si256(p0),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c)))));
                _mm256_storeu_si256(p1, _mm256_add_epi16(_mm256_loadu_si256(p1),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16)))));
            }
        }
        for (int k = 0; k < ns; ++k) {
            const std::int8_t* w = sub[k];
            for (int c = 0; c < hl; c += 32) {
                __m256i* p0 = reinterpret_cast<__m256i*>(acc + c);
                __m256i* p1 = reinterpret_cast<__m256i*>(acc + c + 16);
                _mm256_storeu_si256(p0, _mm256_sub_epi16(_mm256_loadu_si256(p0),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c)))));
                _mm256_storeu_si256(p1, _mm256_sub_epi16(_mm256_loadu_si256(p1),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16)))));
            }
        }
        return;
    }
    {
        auto w16 = [](const std::int8_t* p) {
            return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        };
        int c = 0;
        for (; c + 64 <= hl; c += 64) {   // 64 int16 per chunk (4 vectors x 2 lanes): fewer iterations, more ILP (A9 +1.2%)
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 16));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 32));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 48));
            __m256i u0 = _mm256_setzero_si256(), u1 = u0, u2 = u0, u3 = u0;
            int k = 0;
            for (; k + 1 < na; k += 2) {
                const std::int8_t* p = add[k] + c; const std::int8_t* q = add[k + 1] + c;
                v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
                u0 = _mm256_add_epi16(u0, w16(q)); u1 = _mm256_add_epi16(u1, w16(q + 16)); u2 = _mm256_add_epi16(u2, w16(q + 32)); u3 = _mm256_add_epi16(u3, w16(q + 48));
            }
            for (; k < na; ++k) {
                const std::int8_t* p = add[k] + c;
                v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
            }
            for (k = 0; k + 1 < ns; k += 2) {
                const std::int8_t* p = sub[k] + c; const std::int8_t* q = sub[k + 1] + c;
                v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
                u0 = _mm256_sub_epi16(u0, w16(q)); u1 = _mm256_sub_epi16(u1, w16(q + 16)); u2 = _mm256_sub_epi16(u2, w16(q + 32)); u3 = _mm256_sub_epi16(u3, w16(q + 48));
            }
            for (; k < ns; ++k) {
                const std::int8_t* p = sub[k] + c;
                v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),      _mm256_add_epi16(v0, u0));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 16), _mm256_add_epi16(v1, u1));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 32), _mm256_add_epi16(v2, u2));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 48), _mm256_add_epi16(v3, u3));
        }
        for (; c < hl; c += 32) {   // hl % 64 tail (empty for hl = 512)
            __m256i lo  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            __m256i hi  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 16));
            __m256i lo2 = _mm256_setzero_si256(), hi2 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 1 < na; k += 2) {
                lo  = _mm256_add_epi16(lo,  w16(add[k] + c));      hi  = _mm256_add_epi16(hi,  w16(add[k] + c + 16));
                lo2 = _mm256_add_epi16(lo2, w16(add[k + 1] + c));  hi2 = _mm256_add_epi16(hi2, w16(add[k + 1] + c + 16));
            }
            for (; k < na; ++k) {
                lo = _mm256_add_epi16(lo, w16(add[k] + c));  hi = _mm256_add_epi16(hi, w16(add[k] + c + 16));
            }
            for (k = 0; k + 1 < ns; k += 2) {
                lo  = _mm256_sub_epi16(lo,  w16(sub[k] + c));      hi  = _mm256_sub_epi16(hi,  w16(sub[k] + c + 16));
                lo2 = _mm256_sub_epi16(lo2, w16(sub[k + 1] + c));  hi2 = _mm256_sub_epi16(hi2, w16(sub[k + 1] + c + 16));
            }
            for (; k < ns; ++k) {
                lo = _mm256_sub_epi16(lo, w16(sub[k] + c));  hi = _mm256_sub_epi16(hi, w16(sub[k] + c + 16));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),      _mm256_add_epi16(lo, lo2));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 16), _mm256_add_epi16(hi, hi2));
        }
    }
#else
    for (int k = 0; k < na; ++k) acc_add(acc, add[k], hl);
    for (int k = 0; k < ns; ++k) acc_sub(acc, sub[k], hl);
#endif
}

// dst = src (+ add columns) (- sub columns) in ONE pass: reads src, writes dst. Replaces
// memcpy(dst, src) + fused_apply(dst, ...): the accumulator is read once and written once
// per chunk instead of copied, re-read and re-written. int16 sums are order-independent
// -> bit-identical (x86 campaign G4).
inline void fused_apply_src(std::int16_t* dst, const std::int16_t* src, const std::int8_t* const* add, int na,
                            const std::int8_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    // NEON src->dst form of fused_apply (NEON campaign N1): each 16-int16 chunk is read from
    // src once, every add/sub column is folded in over two independent lanes, and the chunk is
    // written to dst once -- no memcpy and no per-column accumulator pass (which is what the
    // #else fallback below costs on ARM). int16 add is associative mod 2^16 -> bit-identical.
    int c = 0;
    // N2b: 64 int16 per chunk -- 8 vectors x 2 lanes = 16 accumulators (a full FT row of a column
    // per two chunk iterations at hl = 512). Same wrapping int16 sums -> bit-identical.
#define SCN_COL64(V0, V1, V2, V3, V4, V5, V6, V7, P, OP)                                              \
    {                                                                                                  \
        const int8x16_t p0 = vld1q_s8((P) + c),      p1 = vld1q_s8((P) + c + 16),                      \
                        p2 = vld1q_s8((P) + c + 32), p3 = vld1q_s8((P) + c + 48);                      \
        V0 = OP(V0, vmovl_s8(vget_low_s8(p0)));  V1 = OP(V1, vmovl_s8(vget_high_s8(p0)));             \
        V2 = OP(V2, vmovl_s8(vget_low_s8(p1)));  V3 = OP(V3, vmovl_s8(vget_high_s8(p1)));             \
        V4 = OP(V4, vmovl_s8(vget_low_s8(p2)));  V5 = OP(V5, vmovl_s8(vget_high_s8(p2)));             \
        V6 = OP(V6, vmovl_s8(vget_low_s8(p3)));  V7 = OP(V7, vmovl_s8(vget_high_s8(p3)));             \
    }
    for (; c + 64 <= hl; c += 64) {
        int16x8_t v0 = vld1q_s16(src + c),      v1 = vld1q_s16(src + c + 8),
                  v2 = vld1q_s16(src + c + 16), v3 = vld1q_s16(src + c + 24),
                  v4 = vld1q_s16(src + c + 32), v5 = vld1q_s16(src + c + 40),
                  v6 = vld1q_s16(src + c + 48), v7 = vld1q_s16(src + c + 56);
        int16x8_t u0 = vdupq_n_s16(0), u1 = u0, u2 = u0, u3 = u0, u4 = u0, u5 = u0, u6 = u0, u7 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, add[k],     vaddq_s16)
            SCN_COL64(u0, u1, u2, u3, u4, u5, u6, u7, add[k + 1], vaddq_s16)
        }
        for (; k < na; ++k) SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, add[k], vaddq_s16)
        for (k = 0; k + 1 < ns; k += 2) {
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, sub[k],     vsubq_s16)
            SCN_COL64(u0, u1, u2, u3, u4, u5, u6, u7, sub[k + 1], vsubq_s16)
        }
        for (; k < ns; ++k) SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, sub[k], vsubq_s16)
        vst1q_s16(dst + c,      vaddq_s16(v0, u0));  vst1q_s16(dst + c + 8,  vaddq_s16(v1, u1));
        vst1q_s16(dst + c + 16, vaddq_s16(v2, u2));  vst1q_s16(dst + c + 24, vaddq_s16(v3, u3));
        vst1q_s16(dst + c + 32, vaddq_s16(v4, u4));  vst1q_s16(dst + c + 40, vaddq_s16(v5, u5));
        vst1q_s16(dst + c + 48, vaddq_s16(v6, u6));  vst1q_s16(dst + c + 56, vaddq_s16(v7, u7));
    }
#undef SCN_COL64
    // N2 (A9-analog): 32 int16 per chunk -- 4 vectors x 2 lanes = 8 accumulators, half the
    // chunk iterations and twice the independent adds in flight. Same wrapping int16 sums.
    for (; c + 32 <= hl; c += 32) {
        int16x8_t v0 = vld1q_s16(src + c),      v1 = vld1q_s16(src + c + 8),
                  v2 = vld1q_s16(src + c + 16), v3 = vld1q_s16(src + c + 24);
        int16x8_t u0 = vdupq_n_s16(0), u1 = u0, u2 = u0, u3 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t p0 = vld1q_s8(add[k] + c),     p1 = vld1q_s8(add[k] + c + 16);
            const int8x16_t q0 = vld1q_s8(add[k + 1] + c), q1 = vld1q_s8(add[k + 1] + c + 16);
            v0 = vaddq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vaddq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vaddq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vaddq_s16(v3, vmovl_s8(vget_high_s8(p1)));
            u0 = vaddq_s16(u0, vmovl_s8(vget_low_s8(q0)));  u1 = vaddq_s16(u1, vmovl_s8(vget_high_s8(q0)));
            u2 = vaddq_s16(u2, vmovl_s8(vget_low_s8(q1)));  u3 = vaddq_s16(u3, vmovl_s8(vget_high_s8(q1)));
        }
        for (; k < na; ++k) {
            const int8x16_t p0 = vld1q_s8(add[k] + c), p1 = vld1q_s8(add[k] + c + 16);
            v0 = vaddq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vaddq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vaddq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vaddq_s16(v3, vmovl_s8(vget_high_s8(p1)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t p0 = vld1q_s8(sub[k] + c),     p1 = vld1q_s8(sub[k] + c + 16);
            const int8x16_t q0 = vld1q_s8(sub[k + 1] + c), q1 = vld1q_s8(sub[k + 1] + c + 16);
            v0 = vsubq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vsubq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vsubq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vsubq_s16(v3, vmovl_s8(vget_high_s8(p1)));
            u0 = vsubq_s16(u0, vmovl_s8(vget_low_s8(q0)));  u1 = vsubq_s16(u1, vmovl_s8(vget_high_s8(q0)));
            u2 = vsubq_s16(u2, vmovl_s8(vget_low_s8(q1)));  u3 = vsubq_s16(u3, vmovl_s8(vget_high_s8(q1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t p0 = vld1q_s8(sub[k] + c), p1 = vld1q_s8(sub[k] + c + 16);
            v0 = vsubq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vsubq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vsubq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vsubq_s16(v3, vmovl_s8(vget_high_s8(p1)));
        }
        vst1q_s16(dst + c,      vaddq_s16(v0, u0));  vst1q_s16(dst + c + 8,  vaddq_s16(v1, u1));
        vst1q_s16(dst + c + 16, vaddq_s16(v2, u2));  vst1q_s16(dst + c + 24, vaddq_s16(v3, u3));
    }
    for (; c < hl; c += 16) {   // 16-int16 chunks (the whole vector without N2; the hl % 32 tail with it)
        int16x8_t lo = vld1q_s16(src + c), hi = vld1q_s16(src + c + 8);
        int16x8_t lo2 = vdupq_n_s16(0), hi2 = vdupq_n_s16(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t w0 = vld1q_s8(add[k] + c), w1 = vld1q_s8(add[k + 1] + c);
            lo  = vaddq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vaddq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vaddq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vaddq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < na; ++k) {
            const int8x16_t w = vld1q_s8(add[k] + c);
            lo = vaddq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vaddq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t w0 = vld1q_s8(sub[k] + c), w1 = vld1q_s8(sub[k + 1] + c);
            lo  = vsubq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vsubq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vsubq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vsubq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t w = vld1q_s8(sub[k] + c);
            lo = vsubq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vsubq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        vst1q_s16(dst + c,     vaddq_s16(lo, lo2));
        vst1q_s16(dst + c + 8, vaddq_s16(hi, hi2));
    }
#elif defined(__AVX2__)
    auto w16 = [](const std::int8_t* p) {
        return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    };
    if (na + ns <= 2) {   // A2d: 1-add/1-sub makes go column-outer (long predictable loops)
        const std::int16_t* s = src;
        for (int k = 0; k < na; ++k, s = dst) {
            const std::int8_t* w = add[k];
            for (int c = 0; c < hl; c += 32) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),
                    _mm256_add_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c)), w16(w + c)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16),
                    _mm256_add_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c + 16)), w16(w + c + 16)));
            }
        }
        for (int k = 0; k < ns; ++k, s = dst) {
            const std::int8_t* w = sub[k];
            for (int c = 0; c < hl; c += 32) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),
                    _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c)), w16(w + c)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16),
                    _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c + 16)), w16(w + c + 16)));
            }
        }
        if (s == src) std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int16_t));  // no columns at all
        return;
    }
    int c = 0;
    for (; c + 64 <= hl; c += 64) {   // 64 int16 per chunk (4 vectors x 2 lanes), src -> dst (A9)
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 16));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 32));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 48));
        __m256i u0 = _mm256_setzero_si256(), u1 = u0, u2 = u0, u3 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const std::int8_t* p = add[k] + c; const std::int8_t* q = add[k + 1] + c;
            v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
            u0 = _mm256_add_epi16(u0, w16(q)); u1 = _mm256_add_epi16(u1, w16(q + 16)); u2 = _mm256_add_epi16(u2, w16(q + 32)); u3 = _mm256_add_epi16(u3, w16(q + 48));
        }
        for (; k < na; ++k) {
            const std::int8_t* p = add[k] + c;
            v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const std::int8_t* p = sub[k] + c; const std::int8_t* q = sub[k + 1] + c;
            v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
            u0 = _mm256_sub_epi16(u0, w16(q)); u1 = _mm256_sub_epi16(u1, w16(q + 16)); u2 = _mm256_sub_epi16(u2, w16(q + 32)); u3 = _mm256_sub_epi16(u3, w16(q + 48));
        }
        for (; k < ns; ++k) {
            const std::int8_t* p = sub[k] + c;
            v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),      _mm256_add_epi16(v0, u0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16), _mm256_add_epi16(v1, u1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 32), _mm256_add_epi16(v2, u2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 48), _mm256_add_epi16(v3, u3));
    }
    for (; c < hl; c += 32) {   // hl % 64 tail (empty for hl = 512)
        __m256i lo  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c));
        __m256i hi  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 16));
        __m256i lo2 = _mm256_setzero_si256(), hi2 = _mm256_setzero_si256();
        int k = 0;
        for (; k + 1 < na; k += 2) {
            lo  = _mm256_add_epi16(lo,  w16(add[k] + c));      hi  = _mm256_add_epi16(hi,  w16(add[k] + c + 16));
            lo2 = _mm256_add_epi16(lo2, w16(add[k + 1] + c));  hi2 = _mm256_add_epi16(hi2, w16(add[k + 1] + c + 16));
        }
        for (; k < na; ++k) {
            lo = _mm256_add_epi16(lo, w16(add[k] + c));  hi = _mm256_add_epi16(hi, w16(add[k] + c + 16));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            lo  = _mm256_sub_epi16(lo,  w16(sub[k] + c));      hi  = _mm256_sub_epi16(hi,  w16(sub[k] + c + 16));
            lo2 = _mm256_sub_epi16(lo2, w16(sub[k + 1] + c));  hi2 = _mm256_sub_epi16(hi2, w16(sub[k + 1] + c + 16));
        }
        for (; k < ns; ++k) {
            lo = _mm256_sub_epi16(lo, w16(sub[k] + c));  hi = _mm256_sub_epi16(hi, w16(sub[k] + c + 16));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),      _mm256_add_epi16(lo, lo2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16), _mm256_add_epi16(hi, hi2));
    }
#else
    std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int16_t));
    for (int k = 0; k < na; ++k) acc_add(dst, add[k], hl);
    for (int k = 0; k < ns; ++k) acc_sub(dst, sub[k], hl);
#endif
}

// Fixed-point in-place fused apply for the float labeler (int32 accumulator, int32 weight
// columns): the src == dst form of fused_apply_src_fx below (aliasing is safe: each chunk is
// fully loaded before it is stored). Integer add is associative -> bit-exact.
inline void fused_apply_fx(std::int32_t* acc, const std::int32_t* const* add, int na,
                           const std::int32_t* const* sub, int ns, int hl) {
    fused_apply_src_fx(acc, acc, add, na, sub, ns, hl);
}

// dst = src (+ add columns) (- sub columns) in ONE pass for the int32 fixed-point
// accumulator (float labeler): the FX analog of fused_apply_src (x86 G4 / NEON N1).
// Replaces memcpy(dst, src) + a separate column pass. int32 sums are order-independent
// -> bit-identical labels. (Self-play campaign FX2 single pass +6.5%, FX3 blocking +6.4%.)
inline void fused_apply_src_fx(std::int32_t* dst, const std::int32_t* src, const std::int32_t* const* add, int na,
                               const std::int32_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    int c = 0;
    // N2b analog: 32 int32 per chunk -- 8 vectors x 2 lanes = 16 accumulators, a quarter of
    // the chunk iterations of the 8-wide form and 4x the adds in flight.
#define SCFX_COL32(V0, V1, V2, V3, V4, V5, V6, V7, P, OP)                                        \
    {                                                                                            \
        V0 = OP(V0, vld1q_s32((P) + c));      V1 = OP(V1, vld1q_s32((P) + c + 4));               \
        V2 = OP(V2, vld1q_s32((P) + c + 8));  V3 = OP(V3, vld1q_s32((P) + c + 12));              \
        V4 = OP(V4, vld1q_s32((P) + c + 16)); V5 = OP(V5, vld1q_s32((P) + c + 20));              \
        V6 = OP(V6, vld1q_s32((P) + c + 24)); V7 = OP(V7, vld1q_s32((P) + c + 28));              \
    }
    for (; c + 32 <= hl; c += 32) {
        int32x4_t v0 = vld1q_s32(src + c),      v1 = vld1q_s32(src + c + 4),
                  v2 = vld1q_s32(src + c + 8),  v3 = vld1q_s32(src + c + 12),
                  v4 = vld1q_s32(src + c + 16), v5 = vld1q_s32(src + c + 20),
                  v6 = vld1q_s32(src + c + 24), v7 = vld1q_s32(src + c + 28);
        int32x4_t u0 = vdupq_n_s32(0), u1 = u0, u2 = u0, u3 = u0, u4 = u0, u5 = u0, u6 = u0, u7 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, add[k],     vaddq_s32)
            SCFX_COL32(u0, u1, u2, u3, u4, u5, u6, u7, add[k + 1], vaddq_s32)
        }
        for (; k < na; ++k) SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, add[k], vaddq_s32)
        for (k = 0; k + 1 < ns; k += 2) {
            SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, sub[k],     vsubq_s32)
            SCFX_COL32(u0, u1, u2, u3, u4, u5, u6, u7, sub[k + 1], vsubq_s32)
        }
        for (; k < ns; ++k) SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, sub[k], vsubq_s32)
        vst1q_s32(dst + c,      vaddq_s32(v0, u0)); vst1q_s32(dst + c + 4,  vaddq_s32(v1, u1));
        vst1q_s32(dst + c + 8,  vaddq_s32(v2, u2)); vst1q_s32(dst + c + 12, vaddq_s32(v3, u3));
        vst1q_s32(dst + c + 16, vaddq_s32(v4, u4)); vst1q_s32(dst + c + 20, vaddq_s32(v5, u5));
        vst1q_s32(dst + c + 24, vaddq_s32(v6, u6)); vst1q_s32(dst + c + 28, vaddq_s32(v7, u7));
    }
#undef SCFX_COL32
    for (; c < hl; c += 8) {   // hl % 32 tail (empty for hl = 512)
        int32x4_t a0 = vld1q_s32(src + c),  a1 = vld1q_s32(src + c + 4);
        int32x4_t b0 = vdupq_n_s32(0),      b1 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            a0 = vaddq_s32(a0, vld1q_s32(add[k]     + c));  a1 = vaddq_s32(a1, vld1q_s32(add[k]     + c + 4));
            b0 = vaddq_s32(b0, vld1q_s32(add[k + 1] + c));  b1 = vaddq_s32(b1, vld1q_s32(add[k + 1] + c + 4));
        }
        for (; k < na; ++k) {
            a0 = vaddq_s32(a0, vld1q_s32(add[k] + c));      a1 = vaddq_s32(a1, vld1q_s32(add[k] + c + 4));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            a0 = vsubq_s32(a0, vld1q_s32(sub[k]     + c));  a1 = vsubq_s32(a1, vld1q_s32(sub[k]     + c + 4));
            b0 = vsubq_s32(b0, vld1q_s32(sub[k + 1] + c));  b1 = vsubq_s32(b1, vld1q_s32(sub[k + 1] + c + 4));
        }
        for (; k < ns; ++k) {
            a0 = vsubq_s32(a0, vld1q_s32(sub[k] + c));      a1 = vsubq_s32(a1, vld1q_s32(sub[k] + c + 4));
        }
        vst1q_s32(dst + c,     vaddq_s32(a0, b0));
        vst1q_s32(dst + c + 4, vaddq_s32(a1, b1));
    }
#else
    if (dst != src) std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int32_t));
    for (int k = 0; k < na; ++k) { const std::int32_t* w = add[k]; for (int j = 0; j < hl; ++j) dst[j] += w[j]; }
    for (int k = 0; k < ns; ++k) { const std::int32_t* w = sub[k]; for (int j = 0; j < hl; ++j) dst[j] -= w[j]; }
#endif
}

// Apply the exact multiset difference (bef -> aft) of feature indices for one
// perspective: columns only in bef are subtracted, only in aft are added,
// matched pairs cancel (unchanged features cost no column op).
inline void apply_diff(std::int16_t* acc, std::uint32_t* bef, int nb, std::uint32_t* aft, int na) {
    const std::int8_t* l0 = g_net.l0w_i8.data();
    const int hl = g_net.hl;
    const std::size_t base = g_net.base_dims;
    // Features are binary (each present 0/1x in the true feature set), so dedup
    // each side to a SET before the difference. Without this, a pawn-takes-pawn
    // collects the (mover,captured) pawn-pair into `bef` twice (once from each
    // pawn's band -> same symmetric PP index), over-subtracting that column.
    std::sort(bef, bef + nb);
    nb = static_cast<int>(std::unique(bef, bef + nb) - bef);
    std::sort(aft, aft + na);
    na = static_cast<int>(std::unique(aft, aft + na) - aft);
    if (g_no_cols) return;  // probe: keep enumeration+sort, skip the column adds
    static thread_local const std::int8_t* addp[1200];
    static thread_local const std::int8_t* subp[1200];
    int nadd = 0, nsub = 0;
    int i = 0, j = 0;
    while (i < nb && j < na) {
        if (bef[i] == aft[j]) { ++i; ++j; }
        else if (bef[i] < aft[j]) { subp[nsub++] = l0 + (base + bef[i]) * hl; ++i; }
        else { addp[nadd++] = l0 + (base + aft[j]) * hl; ++j; }
    }
    for (; i < nb; ++i) subp[nsub++] = l0 + (base + bef[i]) * hl;
    for (; j < na; ++j) addp[nadd++] = l0 + (base + aft[j]) * hl;
    fused_apply(acc, addp, nadd, subp, nsub, hl);
}

inline int output_bucket(const Board& board, int OB) {
    const int divisor = (32 + OB - 1) / OB;
    return std::clamp((int(board.occ().count()) - 2) / divisor, 0, OB - 1);
}

// L2/L3 tail. Two numerically-close paths: a float matvec (bit-identical to the
// trainer/stock body) and an int8 sdot path (weights quantized at load; screlu
// activations quantized to int8 [0,127]). x1 is the float L1 output in [0,1].
Value finish_body(const Net& n, const float* x1, int b) {
    const int L2 = n.l2, OB = n.out_buckets;
    float y;
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_i8body && !n.l2w_dot.empty()) {
        // Quantize x1 to int8 [0,127]; L2 = 32 int8 dots (vdotq), dequant, screlu;
        // quantize x2 to int8; L3 = one 32-wide int8 dot. Biases stay float.
        const float* l2b = n.l2b.data() + b * 32;
        alignas(16) std::int8_t xi[MAX_L2];
        for (int i = 0; i < L2; ++i) xi[i] = static_cast<std::int8_t>(x1[i] * 127.0f + 0.5f);
        const std::int8_t* w2 = n.l2w_dot.data() + static_cast<std::size_t>(b) * 32 * L2;
        const float deq2 = 127.0f * n.l2_scale;
        alignas(16) std::int8_t x2i[MAX_L2];
        for (int o = 0; o < 32; ++o) {
            const std::int8_t* wo = w2 + static_cast<std::size_t>(o) * L2;
            int32x4_t acc = vdupq_n_s32(0);
            for (int i = 0; i < L2; i += 16) acc = vdotq_s32(acc, vld1q_s8(xi + i), vld1q_s8(wo + i));
            const float a = screlu(static_cast<float>(vaddvq_s32(acc)) / deq2 + l2b[o]);
            x2i[o] = static_cast<std::int8_t>(a * 127.0f + 0.5f);
        }
        const std::int8_t* w3 = n.l3w_dot.data() + static_cast<std::size_t>(b) * 32;
        int32x4_t acc3 = vdupq_n_s32(0);
        for (int i = 0; i < 32; i += 16) acc3 = vdotq_s32(acc3, vld1q_s8(x2i + i), vld1q_s8(w3 + i));
        y = static_cast<float>(vaddvq_s32(acc3)) / (127.0f * n.l3_scale) + n.l3b[b];
        const float cp = std::clamp(400.0f * y, -15000.0f, 15000.0f);
        return static_cast<Value>(std::lround(cp));
    }
#endif
    std::array<float, 32> x2{};
    const float* l2w = n.l2w.data();
    const float* l2b = n.l2b.data() + b * 32;
    const std::size_t ostride = static_cast<std::size_t>(OB) * 32;
    (void)ostride;   // used by the scalar fallback only (NEON reads l2w_bm, AVX2 reads l2w_bm)
#if defined(__ARM_NEON)
    const float* w2base = l2w + static_cast<std::size_t>(b) * 32;  // row i at + i*ostride + o
    // N6 (G2 port): the bucket-major copy [(b*L2 + i)*32 + o] -- 4 KB contiguous per bucket, row i
    // at + i*32 -- instead of 32 rows at a 1 KB stride. Same values, same fma order -> bit-identical.
    const float* w2bm = n.l2w_bm.data() + static_cast<std::size_t>(b) * L2 * 32;
    (void)w2base;
    if (!g_l1dense && L2 == 32) {
        // §L3: NNZ sparse L2 (input-major over nonzero x1). screlu(x1) >= 0, ~half zero;
        // a skipped x1[i]==0 adds w*0.0f == 0 exactly -> per output the ascending-i fmla
        // chain is unchanged -> BYTE-IDENTICAL to the dense path. 8 f32x4 hold 32 outputs.
        int nz2[32]; int nnz2 = 0;
        for (int i = 0; i < 32; ++i) { nz2[nnz2] = i; nnz2 += (x1[i] != 0.0f); }
        float32x4_t c0=vld1q_f32(l2b),    c1=vld1q_f32(l2b+4),  c2=vld1q_f32(l2b+8),  c3=vld1q_f32(l2b+12),
                    c4=vld1q_f32(l2b+16), c5=vld1q_f32(l2b+20), c6=vld1q_f32(l2b+24), c7=vld1q_f32(l2b+28);
        for (int k = 0; k < nnz2; ++k) {
            const int i = nz2[k]; const float xi = x1[i];
            const float* w = w2bm + static_cast<std::size_t>(i) * 32;
            c0=vfmaq_n_f32(c0, vld1q_f32(w),    xi); c1=vfmaq_n_f32(c1, vld1q_f32(w+4),  xi);
            c2=vfmaq_n_f32(c2, vld1q_f32(w+8),  xi); c3=vfmaq_n_f32(c3, vld1q_f32(w+12), xi);
            c4=vfmaq_n_f32(c4, vld1q_f32(w+16), xi); c5=vfmaq_n_f32(c5, vld1q_f32(w+20), xi);
            c6=vfmaq_n_f32(c6, vld1q_f32(w+24), xi); c7=vfmaq_n_f32(c7, vld1q_f32(w+28), xi);
        }
        alignas(16) float tmp[32];
        vst1q_f32(tmp,c0);    vst1q_f32(tmp+4,c1);  vst1q_f32(tmp+8,c2);  vst1q_f32(tmp+12,c3);
        vst1q_f32(tmp+16,c4); vst1q_f32(tmp+20,c5); vst1q_f32(tmp+24,c6); vst1q_f32(tmp+28,c7);
        for (int o = 0; o < 32; ++o) x2[o] = screlu(tmp[o]);
    } else
    // 32x32 L2 matvec, 4 outputs at a time; per-output i-accumulation fmla chain,
    // matching the fp-contracted scalar -> bit-identical.
    for (int o = 0; o < 32; o += 4) {
        float32x4_t acc = vld1q_f32(l2b + o);
        for (int i = 0; i < L2; ++i)
            acc = vfmaq_n_f32(acc, vld1q_f32(w2bm + static_cast<std::size_t>(i) * 32 + o), x1[i]);
        float tmp[4];
        vst1q_f32(tmp, acc);
        for (int k = 0; k < 4; ++k) x2[o + k] = screlu(tmp[k]);
    }
#elif defined(__AVX2__)
    // AVX2: 4 x 8 outputs, i-outer. Per output the ascending-i FMA chain is exactly the
    // fp-contracted scalar chain (s = fma(x1[i], w, s)) -> bit-identical. Rows are 32
    // contiguous floats at l2w + i*ostride + b*32 (the scalar loop walks them with a
    // 1 KB stride per output instead).
    {
        const float* w2base = n.l2w_bm.data() + static_cast<std::size_t>(b) * L2 * 32;   // contiguous rows of 32 (G2)
        constexpr std::size_t wstride = 32;
        (void)l2w;
        __m256 c0 = _mm256_loadu_ps(l2b), c1 = _mm256_loadu_ps(l2b + 8),
               c2 = _mm256_loadu_ps(l2b + 16), c3 = _mm256_loadu_ps(l2b + 24);
        for (int i = 0; i < L2; ++i) {
            const __m256 xi = _mm256_set1_ps(x1[i]);
            const float* w = w2base + static_cast<std::size_t>(i) * wstride;
            c0 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w),      c0);
            c1 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 8),  c1);
            c2 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 16), c2);
            c3 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 24), c3);
        }
        alignas(32) float tmp[32];
        _mm256_store_ps(tmp, c0); _mm256_store_ps(tmp + 8, c1);
        _mm256_store_ps(tmp + 16, c2); _mm256_store_ps(tmp + 24, c3);
        for (int o = 0; o < 32; o += 8) {   // 8-wide screlu, same ops per lane (clamp, square); A7
            __m256 v = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(tmp + o), _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(x2.data() + o, _mm256_mul_ps(v, v));
        }
    }
#else
    for (int o = 0; o < 32; ++o) {
        float s = l2b[o];
        for (int i = 0; i < L2; ++i)
            s += x1[i] * l2w[static_cast<std::size_t>(i) * ostride + b * 32 + o];
        x2[o] = screlu(s);
    }
#endif
    y = n.l3b[b];
    {   // contiguous per-bucket L3 row (A6); the sequential fma chain is unchanged -> bit-identical
        const float* w3 = n.l3w_t.data() + static_cast<std::size_t>(b) * 32;
        for (int i = 0; i < 32; ++i) y += x2[i] * w3[i];
    }
    const float cp = std::clamp(400.0f * y, -15000.0f, 15000.0f);
    return static_cast<Value>(std::lround(cp));
}

// Float reference (matches the trainer's forward pass; parity-verified).
// §P1: build the ascending nonzero-index list of h[0..n) via NEON left-packing.
// Per 4 lanes, vcgtq_f32>0 -> 4-bit code -> a LUT of set-lane offsets (ascending),
// advance by popcount. Returns the SAME list as the branchless scalar build (so the
// sparse matvec is byte-identical); this just builds it faster.
static const std::uint8_t kNnzLut[16][4] = {
    {0,0,0,0},{0,0,0,0},{1,0,0,0},{0,1,0,0},{2,0,0,0},{0,2,0,0},{1,2,0,0},{0,1,2,0},
    {3,0,0,0},{0,3,0,0},{1,3,0,0},{0,1,3,0},{2,3,0,0},{0,2,3,0},{1,2,3,0},{0,1,2,3}};
static const std::uint8_t kNnzPop[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};

inline int build_nnz(const float* h, int n, int* nz) {
#if defined(__ARM_NEON)
    int nnz = 0;
    const float32x4_t z = vdupq_n_f32(0.0f);
    const uint32x4_t  bit = {1u, 2u, 4u, 8u};
    for (int i = 0; i < n; i += 4) {
        const uint32x4_t m = vcgtq_f32(vld1q_f32(h + i), z);
        const unsigned code = vaddvq_u32(vandq_u32(m, bit));  // 0..15
        const std::uint8_t* off = kNnzLut[code];
        nz[nnz + 0] = i + off[0];  nz[nnz + 1] = i + off[1];
        nz[nnz + 2] = i + off[2];  nz[nnz + 3] = i + off[3];
        nnz += kNnzPop[code];
    }
    return nnz;
#else
    int nnz = 0;
    for (int i = 0; i < n; ++i) { nz[nnz] = i; nnz += (h[i] != 0.0f); }
    return nnz;
#endif
}

// SWAP=false: the position as it is. SWAP=true: the same accumulator read with the two
// perspectives exchanged (the position after a pass), negated so the result is from the
// ORIGINAL side to move's point of view. See evaluate_pass() in nnue.hpp.
template <bool SWAP>
Value eval_float_t(const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    // §3.1: read the incremental fixed-point accumulator (int32, scale FX_S) and
    // convert to float. Integer add is associative, so the incremental value is
    // bit-exact to a from-scratch refresh_into_fx; both produce the same integer-cp
    // eval as the old float gather (verified). Mirrors eval_quant's structure.
    alignas(64) float acc_stm[MAX_HL], acc_ntm[MAX_HL];
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    auto fill = [&](const AccFx& a) {
#if defined(__ARM_NEON)
        // §L4: vectorised int32->float. Byte-identical after CReLU: a value lands in
        // (0,1) only for v in [0, FX_S=2^22), where (float)v is exact (v < 2^24) and
        // x2^-FX_SHIFT is an exact power-of-two scale; all other v are CReLU-clamped
        // to 0/1 identically regardless of the int->float rounding point.
        const float32x4_t inv = vdupq_n_f32(float(1.0 / FX_S));
        for (int j = 0; j < hl; j += 4) {
            vst1q_f32(acc_stm + j, vmulq_f32(vcvtq_f32_s32(vld1q_s32(&a.v[stmp][j])),     inv));
            vst1q_f32(acc_ntm + j, vmulq_f32(vcvtq_f32_s32(vld1q_s32(&a.v[1 - stmp][j])), inv));
        }
#else
        for (int j = 0; j < hl; ++j) {
            acc_stm[j] = float(double(a.v[stmp][j])     / FX_S);
            acc_ntm[j] = float(double(a.v[1 - stmp][j]) / FX_S);
        }
#endif
    };
    if (g_ply >= 0) {  // search has the incremental accumulator in sync
        const AccFx& a = g_stack_fx[g_ply];
        if (g_verify) {  // incremental must equal a fresh refresh, bit-exact
            AccFx chk; refresh_into_fx(chk, board);
            for (int j = 0; j < hl; ++j)
                if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
                    std::fprintf(stderr, "[nnue] FX ACC MISMATCH ply=%d j=%d inc=%d,%d chk=%d,%d\n",
                                 g_ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
                    break;
                }
        }
        fill(a);
    } else {  // outside search (bare `eval`): from-scratch fixed-point
        AccFx a; refresh_into_fx(a, board); fill(a);
    }

    alignas(64) float h[MAX_HL];
    // Pairwise destinations (see eval_quant_t): SWAP exchanges the two perspectives, i.e. the
    // eval of the position with the side to move flipped (the position after a pass).
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
#if defined(__ARM_NEON)
    // §2.3: elementwise crelu(a)*crelu(b), 4 lanes/iter. clamp = vmin(vmax(x,0),1),
    // bit-identical to std::clamp for finite accumulator values. half % 4 == 0.
    const float32x4_t vz = vdupq_n_f32(0.0f), vo = vdupq_n_f32(1.0f);
    for (int i = 0; i < half; i += 4) {
        const float32x4_t a = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i),        vz), vo);
        const float32x4_t b = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i + half), vz), vo);
        vst1q_f32(h + off_stm + i, vmulq_f32(a, b));
        const float32x4_t c = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i),        vz), vo);
        const float32x4_t d = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i + half), vz), vo);
        vst1q_f32(h + off_ntm + i, vmulq_f32(c, d));
    }
#else
    for (int i = 0; i < half; ++i) {
        h[off_stm + i] = crelu(acc_stm[i]) * crelu(acc_stm[i + half]);
        h[off_ntm + i] = crelu(acc_ntm[i]) * crelu(acc_ntm[i + half]);
    }
#endif
    const int b = output_bucket(board, OB);
    float x1[MAX_L2];
#if defined(__ARM_NEON)
    const std::size_t istride = static_cast<std::size_t>(OB) * L2;
    const float* wbase = &n.l1w[static_cast<std::size_t>(b) * L2];   // row i at + i*istride + o
    const float* l1bb  = &n.l1b[b * L2];
    if (!g_l1dense && L2 == 32) {
        // §L1 NNZ sparse (input-major): iterate only nonzero h[i]. A skipped h[i]==0 would
        // add w*0.0f == 0 exactly, so per output the ascending-i fmla chain is unchanged ->
        // BYTE-IDENTICAL to the dense §2.4 path. ~76% of h is zero (round-1 h8 measurement).
        // 8 f32x4 accumulators hold all 32 outputs; the 32 o-weights at a fixed i are contiguous.
        int nz[MAX_HL];
        const int nnz = build_nnz(h, hl, nz);  // §P1: NEON left-packing (same list, faster)
        float32x4_t a0=vld1q_f32(l1bb),    a1=vld1q_f32(l1bb+4),  a2=vld1q_f32(l1bb+8),
                    a3=vld1q_f32(l1bb+12), a4=vld1q_f32(l1bb+16), a5=vld1q_f32(l1bb+20),
                    a6=vld1q_f32(l1bb+24), a7=vld1q_f32(l1bb+28);
        for (int k = 0; k < nnz; ++k) {
            const int i = nz[k];
            const float hi = h[i];
            const float* w = wbase + static_cast<std::size_t>(i) * istride;
            a0=vfmaq_n_f32(a0, vld1q_f32(w),    hi); a1=vfmaq_n_f32(a1, vld1q_f32(w+4),  hi);
            a2=vfmaq_n_f32(a2, vld1q_f32(w+8),  hi); a3=vfmaq_n_f32(a3, vld1q_f32(w+12), hi);
            a4=vfmaq_n_f32(a4, vld1q_f32(w+16), hi); a5=vfmaq_n_f32(a5, vld1q_f32(w+20), hi);
            a6=vfmaq_n_f32(a6, vld1q_f32(w+24), hi); a7=vfmaq_n_f32(a7, vld1q_f32(w+28), hi);
        }
        alignas(16) float tmp[32];
        vst1q_f32(tmp,a0);    vst1q_f32(tmp+4,a1);  vst1q_f32(tmp+8,a2);  vst1q_f32(tmp+12,a3);
        vst1q_f32(tmp+16,a4); vst1q_f32(tmp+20,a5); vst1q_f32(tmp+24,a6); vst1q_f32(tmp+28,a7);
        for (int o = 0; o < 32; ++o) x1[o] = screlu(tmp[o]);
    } else {
        // dense §2.4 (SCNNUE_L1DENSE A/B baseline, or L2 != 32)
        for (int o = 0; o < L2; o += 4) {
            float32x4_t acc = vld1q_f32(l1bb + o);
            for (int i = 0; i < hl; ++i)
                acc = vfmaq_n_f32(acc, vld1q_f32(wbase + static_cast<std::size_t>(i) * istride + o), h[i]);
            float tmp[4];
            vst1q_f32(tmp, acc);
            for (int k = 0; k < 4; ++k) x1[o + k] = screlu(tmp[k]);
        }
    }
#else
    for (int o = 0; o < L2; ++o) {
        float s = n.l1b[b * L2 + o];
        for (int i = 0; i < hl; ++i)
            s += h[i] * n.l1w[static_cast<std::size_t>(i) * (OB * L2) + b * L2 + o];
        x1[o] = screlu(s);
    }
#endif
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}

// Quantised (SCN5): int8 feature transformer (x QA=127) with int16 bias and int16
// accumulator; crelu clamps to [0,QA], the pairwise product of the two halves is
// scaled back to int8 [0,QA] (h8), which feeds the int8 L1 sdot; the L1 output is
// dequantised (by QA*QB) to float; body kept float.
//
// Speed path: fixed stack buffers (no per-eval heap alloc) and an int16 accumulator
// so the hot add loop widens the int8 weights 16-wide on NEON (vmovl_s8 + vaddq_s16).
// int16 headroom: |l0w_i8| <= QA=127 (l0w is clipped to +/-0.99 in training, then
// x127), so the accumulator holds ~250 active features before nearing 32767; real
// positions peak ~112 (verified over 200k positions), so it does not overflow.
// SWAP as in eval_float_t: false = the position as it is, true = the perspectives exchanged
// (the position after a pass), negated to the original side to move's point of view.
template <bool SWAP>
Value eval_quant_t(const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;

    // Buffers used ONLY by the gather fallback; the incremental path reads the
    // accumulator in place through these pointers (no 2KB memcpy — that copy was
    // profiled at ~11% of eval on the play path; byte-identical, same int16 values).
    alignas(64) std::int16_t acc_buf_stm[MAX_HL], acc_buf_ntm[MAX_HL];
    const std::int16_t* acc_stm;
    const std::int16_t* acc_ntm;
    const std::int8_t* l0 = n.l0w_i8.data();

    // Full accumulator (base + threats + pp) from the incremental state if the
    // search has it in sync — no gather at all. Else full recompute (fallback).
    const bool have_acc = (g_ply >= 0);
    if (have_acc) {
        const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
        const Acc& a = g_stack[g_ply];
        acc_stm = a.v[stmp];       // read the accumulator in place — no copy
        acc_ntm = a.v[1 - stmp];
        if (g_verify) {  // cross-check the incremental accumulator vs a fresh recompute
            Acc chk;
            refresh_into(chk, board);
            for (int j = 0; j < hl; ++j)
                if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
                    std::fprintf(stderr, "[nnue] ACC MISMATCH ply=%d j=%d  inc=%d,%d chk=%d,%d\n",
                                 g_ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
                    break;
                }
        }
    } else {
        for (int j = 0; j < hl; ++j) { acc_buf_stm[j] = n.l0b_i[j]; acc_buf_ntm[j] = n.l0b_i[j]; }
        gather(
            board, n.base_dims,
            [&](std::size_t f) { acc_add(acc_buf_stm, l0 + f * hl, hl); },
            [&](std::size_t f) { acc_add(acc_buf_ntm, l0 + f * hl, hl); });
        acc_stm = acc_buf_stm;
        acc_ntm = acc_buf_ntm;
    }

    // Int8 pairwise activation: crelu-clamp to [0,QA], multiply the
    // two halves, and scale down by QA so the result fits int8 [0,QA]. This is
    // the L1 input for the sdot dot-products (h8 = pairwise * QA).
    alignas(16) std::int8_t h8[MAX_HL];
    // Where each perspective's pairwise products land in h8. SWAP=false: side to move first
    // (the position as it is). SWAP=true: the two perspectives exchanged, i.e. the net sees the
    // position with the side to move flipped -- the position after a pass. Nothing above this
    // line depends on the side to move (Acc::v is per absolute colour, the output bucket is
    // occupancy-only), so the swapped query is exactly the eval of the null-moved position.
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
#if defined(__ARM_NEON)
    // NEON, 8 lanes/iter. Bit-identical to the scalar form: clamped products are
    // <= QA*QA = 16129 (fit int16), and floor(p/127) == (p*16514) >> 21 exactly
    // for p in [0,16129] (verified exhaustively). This dominated the eval body.
    static_assert(QA == 127, "NEON pairwise /QA magic assumes QA==127");
    const int16x8_t vz = vdupq_n_s16(0), vqa = vdupq_n_s16(QA);
    const int32x4_t vM = vdupq_n_s32(16514);
    auto pairwise8 = [&](const std::int16_t* src, std::int8_t* dst) {
        const int16x8_t a = vminq_s16(vmaxq_s16(vld1q_s16(src), vz), vqa);
        const int16x8_t b = vminq_s16(vmaxq_s16(vld1q_s16(src + half), vz), vqa);
        const int16x8_t p = vmulq_s16(a, b);
        const int32x4_t lo = vshrq_n_s32(vmulq_s32(vmovl_s16(vget_low_s16(p)), vM), 21);
        const int32x4_t hi = vshrq_n_s32(vmulq_s32(vmovl_s16(vget_high_s16(p)), vM), 21);
        vst1_s8(dst, vmovn_s16(vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))));
    };
    for (int i = 0; i < half; i += 8) {
        pairwise8(acc_stm + i, h8 + off_stm + i);
        pairwise8(acc_ntm + i, h8 + off_ntm + i);
    }
#else
    auto cr = [](int x) -> int { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int i = 0; i < half; ++i) {
        h8[off_stm + i] = static_cast<std::int8_t>((cr(acc_stm[i]) * cr(acc_stm[i + half])) / QA);
        h8[off_ntm + i] = static_cast<std::int8_t>((cr(acc_ntm[i]) * cr(acc_ntm[i + half])) / QA);
    }
#endif
    const int b = output_bucket(board, OB);
    const float DEQ = float(QA) * QB;  // h8 already carries the 1/QA scale
    const std::int8_t* wbucket = n.l1w_dot.data() + static_cast<std::size_t>(b) * L2 * hl;
    float x1[MAX_L2];
#if defined(__ARM_FEATURE_DOTPROD)
    // NEON campaign N5 (A1-analog): 4 outputs per pass, each on TWO independent vdotq chains --
    // 8 dot chains in flight instead of one serial 32-op chain per output (latency-bound), and
    // every h8 vector is loaded once for 4 outputs. Integer sums are order-independent and
    // cannot overflow (|sum| <= 512*127*127) -> bit-identical to the one-chain form.
    {
        int o = 0;
        for (; o + 4 <= L2; o += 4) {
            const std::int8_t* w0 = wbucket + static_cast<std::size_t>(o) * hl;
            const std::int8_t* w1 = w0 + hl;
            const std::int8_t* w2 = w1 + hl;
            const std::int8_t* w3 = w2 + hl;
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            int32x4_t c0 = a0, c1 = a0, c2 = a0, c3 = a0;
            int i = 0;
            for (; i + 32 <= hl; i += 32) {
                const int8x16_t h = vld1q_s8(h8 + i), h2 = vld1q_s8(h8 + i + 16);
                a0 = vdotq_s32(a0, h, vld1q_s8(w0 + i));  c0 = vdotq_s32(c0, h2, vld1q_s8(w0 + i + 16));
                a1 = vdotq_s32(a1, h, vld1q_s8(w1 + i));  c1 = vdotq_s32(c1, h2, vld1q_s8(w1 + i + 16));
                a2 = vdotq_s32(a2, h, vld1q_s8(w2 + i));  c2 = vdotq_s32(c2, h2, vld1q_s8(w2 + i + 16));
                a3 = vdotq_s32(a3, h, vld1q_s8(w3 + i));  c3 = vdotq_s32(c3, h2, vld1q_s8(w3 + i + 16));
            }
            for (; i < hl; i += 16) {   // hl % 32 tail (empty for hl = 512)
                const int8x16_t h = vld1q_s8(h8 + i);
                a0 = vdotq_s32(a0, h, vld1q_s8(w0 + i));  a1 = vdotq_s32(a1, h, vld1q_s8(w1 + i));
                a2 = vdotq_s32(a2, h, vld1q_s8(w2 + i));  a3 = vdotq_s32(a3, h, vld1q_s8(w3 + i));
            }
            const float* bias = n.l1b.data() + b * L2 + o;
            x1[o]     = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a0, c0))) / DEQ + bias[0]);
            x1[o + 1] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a1, c1))) / DEQ + bias[1]);
            x1[o + 2] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a2, c2))) / DEQ + bias[2]);
            x1[o + 3] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a3, c3))) / DEQ + bias[3]);
        }
        for (; o < L2; ++o) {   // L2 % 4 tail (empty for L2 = 32)
            const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
            int32x4_t acc4 = vdupq_n_s32(0);
            for (int i = 0; i < hl; i += 16) acc4 = vdotq_s32(acc4, vld1q_s8(h8 + i), vld1q_s8(w + i));
            x1[o] = screlu(static_cast<float>(vaddvq_s32(acc4)) / DEQ + n.l1b[b * L2 + o]);
        }
    }
#elif defined(__ARM_FEATURE_DOTPROD)
    // Dense int8 sdot. NNZ sparse (input-major over nonzero h8) was tried and lost
    // hard on NEON (-20%): vdotq packs 16 MACs + reduction per op, so the ~76% h8
    // sparsity can't beat it (see NNUE_SPEED_TODO_2.md item 6).
    for (int o = 0; o < L2; ++o) {
        const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
        int32x4_t acc4 = vdupq_n_s32(0);
        for (int i = 0; i < hl; i += 16)
            acc4 = vdotq_s32(acc4, vld1q_s8(h8 + i), vld1q_s8(w + i));
        x1[o] = screlu(static_cast<float>(vaddvq_s32(acc4)) / DEQ + n.l1b[b * L2 + o]);
    }
#elif defined(__AVX2__)
    // AVX2 counterpart of the sdot path. h8 is in [0,QA]=[0,127], i.e. a valid UNSIGNED
    // byte, and l1w_dot is signed int8: VPMADDUBSW (u8*s8, adjacent pairs summed into
    // int16) then VPMADDWD (int16 pairs -> int32). Saturation is impossible: a pair sum
    // is at most 2*127*127 = 32258 < 32767. Integer sums are order-independent, so the
    // result equals the scalar int32 accumulation exactly (bit-identical). Four outputs
    // share every 32-byte h8 load; the epilogue is the scalar one, verbatim.
    {
        const __m256i ones16 = _mm256_set1_epi16(1);
        auto hsum32 = [](__m256i v) -> std::int32_t {
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            return _mm_cvtsi128_si32(s);
        };
        auto ld = [](const std::int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        int o = 0;
        for (; o + 4 <= L2; o += 4) {
            const std::int8_t* w0 = wbucket + static_cast<std::size_t>(o) * hl;
            const std::int8_t* w1 = w0 + hl;
            const std::int8_t* w2 = w1 + hl;
            const std::int8_t* w3 = w2 + hl;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            int i = 0;
            for (; i + 32 <= hl; i += 32) {
                const __m256i h = ld(h8 + i);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w0 + i)), ones16));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w1 + i)), ones16));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w2 + i)), ones16));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w3 + i)), ones16));
            }
            std::int32_t s0 = hsum32(a0), s1 = hsum32(a1), s2 = hsum32(a2), s3 = hsum32(a3);
            for (; i < hl; ++i) {  // hl % 32 tail (empty for hl = 512)
                s0 += static_cast<int>(h8[i]) * w0[i]; s1 += static_cast<int>(h8[i]) * w1[i];
                s2 += static_cast<int>(h8[i]) * w2[i]; s3 += static_cast<int>(h8[i]) * w3[i];
            }
            {   // 4-wide epilogue -- the same IEEE ops per lane as the scalar form
                // (int->float, divide by DEQ, add bias, clamp to [0,1], square). x86 campaign A7: +1.04%.
                const __m128 sv = _mm_cvtepi32_ps(_mm_setr_epi32(s0, s1, s2, s3));
                __m128 v = _mm_add_ps(_mm_div_ps(sv, _mm_set1_ps(DEQ)), _mm_loadu_ps(n.l1b.data() + b * L2 + o));
                v = _mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
                _mm_storeu_ps(x1 + o, _mm_mul_ps(v, v));
            }
        }
        for (; o < L2; ++o) {  // L2 % 4 tail (empty for L2 = 32)
            const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
            std::int32_t s = 0;
            for (int i = 0; i < hl; ++i) s += static_cast<int>(h8[i]) * w[i];
            x1[o] = screlu(static_cast<float>(s) / DEQ + n.l1b[b * L2 + o]);
        }
    }
#else
    for (int o = 0; o < L2; ++o) {
        const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
        std::int32_t s = 0;
        for (int i = 0; i < hl; ++i) s += static_cast<int>(h8[i]) * w[i];
        x1[o] = screlu(static_cast<float>(s) / DEQ + n.l1b[b * L2 + o]);
    }
#endif
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}

}  // namespace

Value evaluate(const Board& board) {
    return g_net.quant ? eval_quant_t<false>(board) : eval_float_t<false>(board);
}

Value evaluate_pass(const Board& board) {
    return g_net.quant ? eval_quant_t<true>(board) : eval_float_t<true>(board);
}

// ---- incremental accumulator hooks (called by search) -----------------------
void acc_reset(const Board& board) {
    if (!g_net.loaded) { g_ply = -1; return; }
    if (g_bb.empty()) g_bb.resize(ACC_STACK);
    g_ply = 0;
    if (g_net.quant) {
        if (g_stack.empty()) g_stack.resize(ACC_STACK);
        refresh_into(g_stack[0], board);
    } else {  // float labeler: fixed-point incremental accumulator
        if (g_stack_fx.empty()) g_stack_fx.resize(ACC_STACK);
        refresh_into_fx(g_stack_fx[0], board);
    }
    g_bb[0] = BB(board);
}

// Incremental delta, templated on accumulator/weight type so the intricate delta
// logic lives once. Quant instantiates <Acc,int8> (byte-identical to before, guarded
// by verify_acc); the float labeler instantiates <AccFx,int32> (fixed-point).
template <class AccT, class WT>
void acc_make_t(const Board& before, Move m, std::vector<AccT>& g_stk,
                const WT* l0, void (*refresh_fn)(AccT&, const Board&)) {
    constexpr bool FX = (sizeof(WT) != 1);  // int32 fixed-point float vs int8 quant
    g_threats_made = 0;                     // item #9: per-move, so reset on every entry
    if (g_ply < 0 || g_ply + 1 >= ACC_STACK) { g_ply = -1; return; }
    const int hl = g_net.hl;
    AccT& cur = g_stk[g_ply];
    AccT& nxt = g_stk[g_ply + 1];
    // fused-apply dispatch: int8/int16 (quant) vs int32/int32 (fixed-point float).
    auto fused = [&](auto* acc, const WT* const* add, int na, const WT* const* sub, int ns) {
        if constexpr (FX) fused_apply_fx(acc, add, na, sub, ns, hl);
        else              fused_apply(acc, add, na, sub, ns, hl);
    };
    // G4 single pass: dst = src +/- columns (quant path; the float labeler joins with FX2).
    [[maybe_unused]] auto fused_src = [&](auto* dst, const auto* src, const WT* const* add, int na,
                                          const WT* const* sub, int ns) {
        if constexpr (FX) fused_apply_src_fx(dst, src, add, na, sub, ns, hl);
        else              fused_apply_src(dst, src, add, na, sub, ns, hl);
    };
    const int from = m.from().index(), to = m.to().index();
    const int moved_pi = static_cast<int>(before.at(m.from()).internal());
    const int mcolor = moved_pi / 6, mt0 = moved_pi % 6;
    // Refresh is only needed when the moving side's king changes BUCKET or MIRROR
    // (e-file) side: base features are king-bucketed+mirrored, threats/pp are only
    // mirrored, and the king is never a threat attacker/target -- so a king move
    // that stays in the same bucket and mirror side is a plain incremental move
    // (the normal delta path handles the king-as-piece + slider blocking exactly).
    // Castle/promo/ep always refresh (multi-piece / index-space changes).
    bool special;
    bool king_rebucket = false;  // king crossed BUCKET but not mirror -> incremental + base fix
    if (m.typeOf() != Move::NORMAL) {
        special = true;
    } else if (mt0 == 5) {  // king normal move
        const int kp_f = (mcolor == 0) ? from : (from ^ 56);
        const int kp_t = (mcolor == 0) ? to : (to ^ 56);
        // Only a MIRROR (e-file) cross needs a full refresh: it flips the threat/pp
        // orientation for the mover's perspective. A pure bucket cross keeps the same
        // orientation, so it stays on the incremental path and only the mover's-own
        // base block is rebucketed afterwards (see king_rebucket below).
        const bool mirror_change = (file_of(kp_f) > 3) != (file_of(kp_t) > 3);
        special = g_no_finny || mirror_change;
        king_rebucket = !special && (king_bucket(kp_f) != king_bucket(kp_t));
    } else {
        special = false;
    }
    if (special) {
        // G8: make the move on the caller's own board and unmake it after the refresh,
        // instead of copying the Board (which deep-copies its undo stack) per refresh.
        // Every caller passes its mutable search board; the refresh reads exactly the
        // state a copy would have had, and unmakeMove restores the board in full.
        Board& b = const_cast<Board&>(before);
        b.makeMove(m);
        refresh_fn(nxt, b);
        g_bb[g_ply + 1] = BB(b);  // keep the incremental BB chain in sync
        b.unmakeMove(m);
    } else {
        // Single pass (x86 campaign G4, +3.7% all phases): the base columns are deferred into
        // the threat add/sub lists and ONE src->dst fused pass per perspective reads cur and
        // writes nxt -- no memcpy, no separate base pass. Integer sums are order-independent
        // -> bit-identical. The probes (NOUPDATE/NOCOLS/NODIRECT) keep the two-pass form.
        // The float labeler takes the same single pass (int32 sums -> bit-identical; self-play
        // campaign FX2). The NODIRECT probe is int16-only, so FX ignores g_direct here.
        const bool single_pass = (FX || g_direct) && !g_no_update && !g_no_cols;
        if (!single_pass) {
            std::memcpy(nxt.v[0], cur.v[0], hl * sizeof(nxt.v[0][0]));
            std::memcpy(nxt.v[1], cur.v[1], hl * sizeof(nxt.v[0][0]));
        }
        const int kabs[2] = {before.kingSq(Color::WHITE).index(), before.kingSq(Color::BLACK).index()};
        const chess::Piece capp = before.at(m.to());
        const bool cap = (capp != chess::Piece::NONE);
        const int cpi = cap ? static_cast<int>(capp.internal()) : 0;
        // ---- base delta (piece-square): one fused pass per persp, or deferred into the
        // threat lists when single_pass (G4) ----
        const WT* base_add[2];
        const WT* base_sub[2][2];
        const int nbase_sub = cap ? 2 : 1;
        for (int p = 0; p < 2; ++p) {
            base_add[p]    = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, to, kabs[p])) * hl;
            base_sub[p][0] = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, from, kabs[p])) * hl;
            if (cap) base_sub[p][1] = l0 + static_cast<std::size_t>(base_feat(p, cpi / 6, cpi % 6, to, kabs[p])) * hl;
            if (!single_pass) fused(nxt.v[p], &base_add[p], 1, base_sub[p], nbase_sub);
        }
        // ---- threat + pp delta (exact: collect before/after feature sets over
        // the affected pieces, then apply only the multiset difference) ----
        // Incremental board bitboards: the before-state is already on the stack
        // (g_bb[g_ply]), so bb0 is a reference -- no BB rebuild from board.at().
        // Form the after-move child directly in the next slot.
        const BB& bb0 = g_bb[g_ply];
        BB& bb1 = g_bb[g_ply + 1];
        if (g_verify) {  // the stack BB must equal a fresh recompute of `before`
            BB chk(before);
            if (bb0.occ != chk.occ || bb0.wk != chk.wk || bb0.bk != chk.bk
                || std::memcmp(bb0.by, chk.by, sizeof(chk.by)) != 0
                || std::memcmp(bb0.piece_on, chk.piece_on, sizeof(chk.piece_on)) != 0)
                std::fprintf(stderr, "[nnue] BB MISMATCH ply=%d\n", g_ply);
        }
        bb1 = bb0;  // copy parent state, then apply the move in place
        const int sf_moved = (mcolor << 3) | (mt0 + 1);
        bb1.by[sf_moved] ^= (1ull << from) | (1ull << to);
        bb1.piece_on[from] = -1;
        if (cap) bb1.by[((cpi / 6) << 3) | ((cpi % 6) + 1)] &= ~(1ull << to);
        bb1.piece_on[to] = static_cast<std::int8_t>(sf_moved);
        bb1.occ = (bb0.occ & ~(1ull << from)) | (1ull << to);
        if (mt0 == 5) { if (mcolor == 0) bb1.wk = to; else bb1.bk = to; }  // king moved
        if (g_no_update) { g_ply++; return; }  // cost probe: skip the delta (BB kept)
        const int wo = orient_tbl(bb0.wk);       // white perspective (swap 0)
        const int bo = orient_tbl(bb0.bk) ^ 56;  // black perspective (swap 8)

        const std::uint64_t kings = bb0.by[6] | bb0.by[14];  // king move is `special`, so kings0==kings1
        const std::uint64_t excl = (1ull << from) | (1ull << to);
      if (g_direct || FX) {  // float always uses the direct path (no int16-only fallback)
        // ===== Direct delta: emit only genuinely-changed columns straight to the
        // fused add/sub lists -- no full-set collect, no sort, no merge-diff.
        // Moved/captured/non-slider threat indices never cancel between before and
        // after (attacker square or attacked-type always differs), so they are
        // emitted directly; each slider emits its exact a0^a1 ray change (plus the
        // from/to identity flip). Every emitted column is a real change.
        const std::size_t bd = g_net.base_dims;
        static thread_local const WT* addp[2][1024];
        static thread_local const WT* subp[2][1024];
        int na[2] = {0, 0}, ns[2] = {0, 0};
        if (single_pass)   // G4: the base columns ride in the same lists as the threat/pp columns
            for (int p = 0; p < 2; ++p) {
                addp[p][na[p]++] = base_add[p];
                subp[p][ns[p]++] = base_sub[p][0];
                if (cap) subp[p][ns[p]++] = base_sub[p][1];
            }
        auto push = [&](int p, std::uint32_t idx, bool add) {
            const WT* col = l0 + (bd + idx) * hl;
            if (add) addp[p][na[p]++] = col; else subp[p][ns[p]++] = col;
        };
        auto emit_threat = [&](int a_sf, int a_sq, int t_sq, int attacked, bool add) {
            // Item #9: count only threats the MOVER gains on a higher-valued victim. piece-type bits
            // (1=pawn..5=queen) are ordered by value, so the comparison needs no table; the colour
            // bit (8) selects the mover's own attackers, so a threat created AGAINST the mover does
            // not inflate the count.
            if (add && (a_sf & 8) == (sf_moved & 8) && (attacked & 7) > (a_sf & 7)) ++g_threats_made;
            const std::uint32_t s = make_threat_index(a_sf, a_sq ^ wo, t_sq ^ wo, attacked);
            if (s < (std::uint32_t)THREAT_DIMS) push(0, s, add);
            const std::uint32_t nn = make_threat_index(a_sf ^ 8, a_sq ^ bo, t_sq ^ bo, attacked ^ 8);
            if (nn < (std::uint32_t)THREAT_DIMS) push(1, nn, add);
        };
        // full threat set of piece `sf` on `sq` in board `b` (mover: sub@bb0, add@bb1;
        // captured: sub@bb0). Masks match collect_piece_threats exactly.
        auto emit_full = [&](const BB& b, int sq, int sf, bool add) {
            const int pt = sf & 7;
            if (pt == 6 || pt == 0) return;
            const std::uint64_t N = b.by[2] | b.by[10], Bp = b.by[3] | b.by[11],
                                R = b.by[4] | b.by[12], Q = b.by[5] | b.by[13],
                                P = b.by[SF_W_PAWN] | b.by[SF_B_PAWN];
            std::uint64_t attacks;
            if (pt == 1) attacks = pawn_attacks(sf < 8 ? 0 : 1, sq) & (N | R);
            else { const std::uint64_t tgt = (pt == 2 || pt == 5) ? (P | N | Bp | R | Q) : (P | N | Bp | R);
                   attacks = attacks_bb(pt, sq, b.occ) & tgt; }
            while (attacks) { const int t = __builtin_ctzll(attacks); attacks &= attacks - 1;
                emit_threat(sf, sq, t, b.piece_on[t], add); }
        };
        // one non-slider attacker's threat to a single target square on a board
        auto emit_ns = [&](int a_sq, int a_sf, int tgt, const BB& b, bool add) {
            const int pt = a_sf & 7;
            const std::uint64_t atk = (pt == 1) ? pawn_attacks(a_sf < 8 ? 0 : 1, a_sq) : knight_attacks(a_sq);
            if (!(atk & (1ull << tgt))) return;
            const int attacked = b.piece_on[tgt];
            if (attacked < 0) return;
            emit_threat(a_sf, a_sq, tgt, attacked, add);
        };
        auto pp_index = [&](int orient, int persp, int cA, int fsq, int tsq, int cBc) -> std::uint32_t {
            const int fo = fsq ^ orient, to_o = tsq ^ orient;
            const int ca = cA ^ persp, cb = cBc ^ persp;
            const int id_a = 48 * ca + (fo - 8), id_b = 48 * cb + (to_o - 8);
            const int hi = std::max(id_a, id_b), lo = std::min(id_a, id_b);
            return (std::uint32_t)(hi * (hi - 1) / 2 + lo + PP_INDEX_BASE);
        };
        auto emit_pp = [&](const BB& b, int sq, int color, bool add, int exclude_sq) {
            const std::uint64_t wp = b.by[SF_W_PAWN], bp = b.by[SF_B_PAWN];
            std::uint64_t band = g_tab.pawn_pair[sq] & (wp | bp);
            while (band) {
                const int t = __builtin_ctzll(band); band &= band - 1;
                if (t == exclude_sq) continue;
                const int cB = (bp >> t) & 1;
                push(0, pp_index(wo, 0, color, sq, t, cB), add);
                push(1, pp_index(bo, 1, color, sq, t, cB), add);
            }
        };
        // 1) moved piece: old threats (bb0@from) subtracted, new (bb1@to) added
        emit_full(bb0, from, sf_moved, false);
        emit_full(bb1, to,   sf_moved, true);
        // 2) captured piece: its threats vanish
        if (cap) emit_full(bb0, to, ((cpi / 6) << 3) | ((cpi % 6) + 1), false);
        // 3) attackers of from/to. Non-sliders (knights/pawns) are occupancy-
        //    independent -> computed ONCE (identical in bb0/bb1, off the moved
        //    squares); kings never attack -> skipped. Sliders are relevant if they
        //    see from/to in EITHER board (occupancy changed only there), via the
        //    super-piece rays from from/to under each occupancy.
        const std::uint64_t Nb = bb0.by[2] | bb0.by[10];
        const std::uint64_t Wp = bb0.by[SF_W_PAWN], Bp = bb0.by[SF_B_PAWN];
        std::uint64_t nsatt = (knight_attacks(from) | knight_attacks(to)) & Nb;
        nsatt |= Wp & (pawn_attacks(1, from) | pawn_attacks(1, to));  // white pawns hitting from/to
        nsatt |= Bp & (pawn_attacks(0, from) | pawn_attacks(0, to));  // black pawns hitting from/to
        nsatt &= ~excl;
        const std::uint64_t BQ = bb0.by[3] | bb0.by[11] | bb0.by[5] | bb0.by[13];
        const std::uint64_t RQ = bb0.by[4] | bb0.by[12] | bb0.by[5] | bb0.by[13];
        std::uint64_t sliders =
            (((attacks_bb(3, from, bb0.occ) | attacks_bb(3, from, bb1.occ)
             | attacks_bb(3, to, bb0.occ)   | attacks_bb(3, to, bb1.occ)) & BQ)
           | ((attacks_bb(4, from, bb0.occ) | attacks_bb(4, from, bb1.occ)
             | attacks_bb(4, to, bb0.occ)   | attacks_bb(4, to, bb1.occ)) & RQ)) & ~excl;
        while (sliders) {
            const int s = __builtin_ctzll(sliders); sliders &= sliders - 1;
            const int sf = bb0.piece_on[s];      // slider exists in both (s not in excl)
            const int pt = sf & 7;
            const std::uint64_t a0 = attacks_bb(pt, s, bb0.occ);
            const std::uint64_t a1 = attacks_bb(pt, s, bb1.occ);
            std::uint64_t rel = (a0 ^ a1) | ((a0 & a1) & excl);  // gained/lost squares + from/to identity flip
            while (rel) {
                const int q = __builtin_ctzll(rel); rel &= rel - 1;
                if ((a0 >> q) & 1) { const int p0 = bb0.piece_on[q]; if (p0 >= 0) emit_threat(sf, s, q, p0, false); }
                if ((a1 >> q) & 1) { const int p1 = bb1.piece_on[q]; if (p1 >= 0) emit_threat(sf, s, q, p1, true); }
            }
        }
        while (nsatt) {
            const int s = __builtin_ctzll(nsatt); nsatt &= nsatt - 1;
            const int sf = bb0.piece_on[s];      // non-slider: same square/type in both
            emit_ns(s, sf, from, bb0, false);    // it threatened the mover at `from`
            emit_ns(s, sf, to,   bb0, false);    // and whatever was on `to` (capture/empty)
            emit_ns(s, sf, to,   bb1, true);     // now threatens the mover at `to`
            // (`from` is empty in bb1 -> nothing to add there)
        }
        // 4) pawn pairs: moved pawn's pairs move; captured pawn's vanish (dedup the
        //    shared P x P pair by excluding the mover's `from` square once).
        if (mt0 == 0) {
            emit_pp(bb0, from, mcolor, false, -1);
            emit_pp(bb1, to,   mcolor, true,  -1);
        }
        if (cap && (cpi % 6) == 0)
            emit_pp(bb0, to, cpi / 6, false, mt0 == 0 ? from : -1);

        if (g_prefetch)
            for (int p = 0; p < 2; ++p) {
                for (int k = 0; k < na[p]; ++k) __builtin_prefetch(addp[p][k]);
                for (int k = 0; k < ns[p]; ++k) __builtin_prefetch(subp[p][k]);
            }
        if (!g_no_cols) {  // NOCOLS probe: keep enumeration, skip the column adds
            if (single_pass) {   // G4: cur -> nxt in one pass (base + threat + pp columns)
                fused_src(nxt.v[0], cur.v[0], addp[0], na[0], subp[0], ns[0]);
                fused_src(nxt.v[1], cur.v[1], addp[1], na[1], subp[1], ns[1]);
            } else {
                fused(nxt.v[0], addp[0], na[0], subp[0], ns[0]);
                fused(nxt.v[1], addp[1], na[1], subp[1], ns[1]);
            }
        }
      } else if constexpr (!FX) {  // int16-only collect+diff fallback (SCNNUE_NODIRECT)
        static thread_local FeatList bef, aft;
        bef.clear();
        aft.clear();
        // BEF: moved piece (full) + captured piece (full) + attackers of from/to
        // (sliders full for ray effects, knights/pawns reduced to the from/to target).
        collect_piece_threats(bef, bb0, from, bb0.piece_on[from], wo, bo);
        if (cap) collect_piece_threats(bef, bb0, to, bb0.piece_on[to], wo, bo);
        {
            const std::uint64_t att = (attackers_of(bb0, from) | attackers_of(bb0, to)) & ~excl;
            const std::uint64_t sl = bb0.by[3] | bb0.by[11] | bb0.by[4] | bb0.by[12] | bb0.by[5] | bb0.by[13];
            std::uint64_t full = att & sl;
            while (full) { const int s = __builtin_ctzll(full); full &= full - 1;
                collect_piece_threats(bef, bb0, s, bb0.piece_on[s], wo, bo); }
            std::uint64_t ns = att & ~sl & ~kings;
            while (ns) { const int s = __builtin_ctzll(ns); ns &= ns - 1;
                collect_ns_threat(bef, bb0, s, bb0.piece_on[s], from, wo, bo);
                collect_ns_threat(bef, bb0, s, bb0.piece_on[s], to, wo, bo); }
        }
        // AFT: moved piece now at `to` (full) + attackers on bb1 (same split).
        collect_piece_threats(aft, bb1, to, bb1.piece_on[to], wo, bo);
        {
            const std::uint64_t att = (attackers_of(bb1, from) | attackers_of(bb1, to)) & ~excl;
            const std::uint64_t sl = bb1.by[3] | bb1.by[11] | bb1.by[4] | bb1.by[12] | bb1.by[5] | bb1.by[13];
            std::uint64_t full = att & sl;
            while (full) { const int s = __builtin_ctzll(full); full &= full - 1;
                collect_piece_threats(aft, bb1, s, bb1.piece_on[s], wo, bo); }
            std::uint64_t ns = att & ~sl & ~kings;
            while (ns) { const int s = __builtin_ctzll(ns); ns &= ns - 1;
                collect_ns_threat(aft, bb1, s, bb1.piece_on[s], from, wo, bo);
                collect_ns_threat(aft, bb1, s, bb1.piece_on[s], to, wo, bo); }
        }
        if (mt0 == 0) {  // moved pawn: its pairs move
            collect_pp_for_pawn(bef, bb0, from, mcolor, wo, bo);
            collect_pp_for_pawn(aft, bb1, to, mcolor, wo, bo);
        }
        if (cap && (cpi % 6) == 0)  // captured pawn: its pairs vanish
            collect_pp_for_pawn(bef, bb0, to, cpi / 6, wo, bo);
        if (g_prefetch) {
            const std::size_t bd = g_net.base_dims;
            for (int p = 0; p < 2; ++p) {
                for (int k = 0; k < bef.n[p]; ++k)
                    __builtin_prefetch(l0 + (bd + bef.v[p][k]) * hl);
                for (int k = 0; k < aft.n[p]; ++k)
                    __builtin_prefetch(l0 + (bd + aft.v[p][k]) * hl);
            }
        }
        apply_diff(nxt.v[0], bef.v[0], bef.n[0], aft.v[0], aft.n[0]);
        apply_diff(nxt.v[1], bef.v[1], bef.n[1], aft.v[1], aft.n[1]);
      }
        // King crossed a bucket (same mirror): the mover's-own perspective still has
        // every piece indexed under the OLD king bucket. Rebucket v[mcolor] in place
        // -- sub each piece's old-bucket base column, add its new-bucket column. The
        // opponent perspective and all threats/pp were already correct on the
        // incremental path (same mirror -> same orientation; king isn't a threat).
        if (king_rebucket) {
            static thread_local const WT* ra[40];
            static thread_local const WT* rs[40];
            int nra = 0, nrs = 0;
            std::uint64_t occ = bb1.occ;
            while (occ) {
                const int sq = __builtin_ctzll(occ); occ &= occ - 1;
                const int sf = bb1.piece_on[sq];
                const int c = sf >> 3, t0 = (sf & 7) - 1;
                rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, from) * hl;  // old bucket (king@from)
                ra[nra++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, to)   * hl;  // new bucket (king@to)
            }
            fused(nxt.v[mcolor], ra, nra, rs, nrs);
        }
    }
    g_ply++;
}

// Dispatch acc_make to the active accumulator (quant int16 / float fixed-point int32).
void acc_make(const Board& before, Move m) {
    if (g_net.quant)
        acc_make_t<Acc, std::int8_t>(before, m, g_stack, g_net.l0w_i8.data(), refresh_into);
    else
        acc_make_t<AccFx, std::int32_t>(before, m, g_stack_fx, g_net.l0w_fx.data(), refresh_into_fx);
}

void acc_unmake() { if (g_ply > 0) g_ply--; }

// Accumulator units read by acc_delta_l1: a CONTIGUOUS PREFIX of each perspective, not all of them.
// The full 1024-unit norm measured -3.395% nps with the tree held identical
// (logs/accsig/cost-pure.json), which a term firing on a quarter of quiet moves cannot pay for. The
// plug only needs to rank a move against a quartile threshold, not to know the norm exactly, and the
// hidden units carry no ordering, so a fixed prefix is as good a sample as any.
//
// It must be a PREFIX and not a stride: SC_ACCSIG_STRIDE=4 measured -5.763%, WORSE than reading
// everything, because the contiguous loop auto-vectorises to 16-wide SIMD and a strided one does not
// -- 64 vector operations become 256 scalar ones with strided loads (logs/accsig/cost-pure4.json).
#ifndef SC_ACCSIG_UNITS
#define SC_ACCSIG_UNITS 128
#endif

int acc_threats_made() noexcept { return g_net.quant ? g_threats_made : 0; }

int acc_delta_l1() noexcept {
    // Item #8 (novelty ledger): the magnitude of the feature-column change a move makes.
    // A quiet move swaps two columns per perspective, so this is ||W[to] - W[from]||_1 read
    // off the accumulator stack rather than recomputed -- the values are already resident,
    // which is what makes the signal cheap. Both perspectives are summed because the mover's
    // piece changes square in both halves' feature sets.
    if (g_ply < 1 || !g_net.quant) return 0;
    const int hl = g_net.hl;
    const std::int16_t* c0 = g_stack[g_ply].v[0];
    const std::int16_t* p0 = g_stack[g_ply - 1].v[0];
    const std::int16_t* c1 = g_stack[g_ply].v[1];
    const std::int16_t* p1 = g_stack[g_ply - 1].v[1];
    const int n = hl < SC_ACCSIG_UNITS ? hl : SC_ACCSIG_UNITS;
    int sum = 0;
    for (int j = 0; j < n; ++j)
        sum += std::abs(int(c0[j]) - int(p0[j])) + std::abs(int(c1[j]) - int(p1[j]));
    return sum;
}

void acc_make_null() {
    if (g_ply < 0 || g_ply + 1 >= ACC_STACK) { g_ply = -1; return; }
    const int hl = g_net.hl;
    if (g_net.quant) {
        std::memcpy(g_stack[g_ply + 1].v[0], g_stack[g_ply].v[0], hl * sizeof(std::int16_t));
        std::memcpy(g_stack[g_ply + 1].v[1], g_stack[g_ply].v[1], hl * sizeof(std::int16_t));
    } else {
        std::memcpy(g_stack_fx[g_ply + 1].v[0], g_stack_fx[g_ply].v[0], hl * sizeof(std::int32_t));
        std::memcpy(g_stack_fx[g_ply + 1].v[1], g_stack_fx[g_ply].v[1], hl * sizeof(std::int32_t));
    }
    g_bb[g_ply + 1] = g_bb[g_ply];  // piece placement unchanged by a null move
    g_ply++;
}
void acc_unmake_null() { if (g_ply > 0) g_ply--; }

}  // namespace engine::nnue
