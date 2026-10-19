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
#include <memory>
#include <new>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
constexpr int kKingBuckets = *std::max_element(BUCKET_LAYOUT.begin(), BUCKET_LAYOUT.end()) + 1;

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

// Knight / king / pawn attack sets. The loop forms are constexpr and build lookup tables at
// compile time; the hot paths read the tables (NPS 3.6 item T1). Same sets by construction.
constexpr std::uint64_t knight_attacks_loop(int sq) {
    const int f0 = sq & 7, r0 = sq >> 3;
    constexpr int d[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    std::uint64_t bb = 0;
    for (const auto& k : d) {
        const int f = f0 + k[0], r = r0 + k[1];
        if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
    }
    return bb;
}
constexpr std::uint64_t king_attacks_loop(int sq) {
    const int f0 = sq & 7, r0 = sq >> 3;
    std::uint64_t bb = 0;
    for (int df = -1; df <= 1; ++df)
        for (int dr = -1; dr <= 1; ++dr) {
            if (!df && !dr) continue;
            const int f = f0 + df, r = r0 + dr;
            if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
        }
    return bb;
}
constexpr std::uint64_t pawn_attacks_loop(int color, int sq) {  // 0 = white (north), 1 = black
    const int f0 = sq & 7, r0 = sq >> 3, dr = (color == 0) ? 1 : -1;
    std::uint64_t bb = 0;
    for (int df = -1; df <= 1; df += 2) {
        const int f = f0 + df, r = r0 + dr;
        if (f >= 0 && f < 8 && r >= 0 && r < 8) bb |= 1ull << (r * 8 + f);
    }
    return bb;
}
// NPS 3.6 item S1: at load, reorder the hidden neurons so the ones that are usually zero after the
// pairwise activation sit together, then run L1 as a sparse product over the nonzero 4-byte input
// chunks (int8 dot-products of 4 inputs). Any neuron order gives the same integer L1 sums -> the
// evaluation is bit-identical; the order only decides how many all-zero chunks can be skipped.
#ifndef SC_NPS_S1
#define SC_NPS_S1 1
#endif
// NPS 3.6 item P2: pairwise division by 127 without 32-bit multiplies. For x = a*b with a, b in
// [0,127] (x <= 16129), floor(x/127) == (x + 1 + ((x + 1) >> 7)) >> 7 exactly (checked for every
// x in [0, 16129]), so: vmla (1 + a*b, u16), vsra (+ >>7), vshrn (>>7, narrowed to u8). Same h8.
#ifndef SC_NPS_P2
#define SC_NPS_P2 1
#endif
// NPS 3.6 item K1: king-bucket crossings from cached piece-square sums. When the moving king
// changes bucket (same mirror side), its own perspective needs every piece's base column under the
// new bucket instead of the old one. Each (perspective, mirror, bucket) keeps the int16 sum of the
// base columns of the pieces it last saw plus those pieces' bitboards; bringing a cache up to date
// costs only the pieces that changed since. Then acc -= old bucket's sum, += new bucket's sum.
// int16 sums wrap identically however they are grouped -> bit-identical.
#ifndef SC_NPS_K1
#define SC_NPS_K1 1
#endif
#ifndef SC_NPS_K1_MIN   // men on the board from which K1 is used: 16 on both machines (i7-9700 and, re-tuned
#define SC_NPS_K1_MIN 16  // on net 10-15, the M1 Max: +0.43% over 20, 12 below 16)
#endif
#ifndef SC_NPS_A2
#define SC_NPS_A2 1
#endif
// x86 FX: the float labeler path (SCN4 net: int32 fixed-point accumulator, float body) on AVX2 --
// it ran the scalar fallbacks (the int8 play path had all the x86 work). Same integers in the
// accumulator, the same per-element float ops, and the same per-output fma chain in L1.
#ifndef SC_FX_AVX2
#if defined(__AVX2__)
#define SC_FX_AVX2 1
#else
#define SC_FX_AVX2 0
#endif
#endif
// Mac FX (float labeler on NEON), both layout only -> bit-identical:
//  FXB  bucket-major float L1 copy (l1w_bm, as x86 47303d3) for the NEON input-major L1: each bucket's
//       64 KB contiguous instead of 128-byte rows at a 1 KB stride.
//  FXS  one physical copy of the 147 MB fixed-point FT (l0w_fx) for all engine processes: the first
//       process to load a float net writes it to a cache file in $TMPDIR (under an flock, renamed into
//       place) and every process maps that file read-only and shared. Any failure keeps a private copy.
//       Windows (default 1): a named pagefile-backed section instead (see fx_share_map). i7-9700,
//       8-worker self-play at depth 9: +26% (private copies 505/498 pos/s, shared 634/627).
// Mac, measured with tools/speed/fx8_bench.py (8 workers, depth 9, 240 superbook-p1 games per run, M1 Max):
// FXB +2.37% / +4.53% (two sessions, 4 and 6 rounds), FXS on top of FXB +2.72% / +1.63%, both +5.16% / +6.24%.
#ifndef SC_FX_L1BM_NEON
#if defined(__ARM_NEON)
#define SC_FX_L1BM_NEON 1
#else
#define SC_FX_L1BM_NEON 0
#endif
#endif
// x86 FXZ: lazy float accumulator. acc_make records a ply's add/sub column lists instead of applying
// them, and the first read of that ply (eval, a child that must apply its own delta, a null move)
// applies them from the newest computed ancestor, one ply at a time -- the same src->dst int32 pass,
// so the accumulator holds the same integers. A ply popped unread costs no column traffic (~18% of
// float updates are never read; SC_LZSTAT on 12 depth-9 games).
#ifndef SC_FX_LAZY
#if defined(__x86_64__) || defined(_M_X64)
#define SC_FX_LAZY 1
#else
#define SC_FX_LAZY 0
#endif
#endif
constexpr int kFxPendCap = 96;
// x86 FXP: the float eval's int32->float fill, pairwise product and the L1's nonzero masks in one
// pass straight from the accumulator (the same per-element ops; no 4 KB float copy written and read
// back), and an eval frame under one page (no Windows stack probe per eval).
#ifndef SC_FX_FUSE
#define SC_FX_FUSE 1
#endif
#if defined(__AVX2__) && SC_FX_AVX2 && SC_FX_FUSE
#define SC_FXP 1
#else
#define SC_FXP 0
#endif   // FXZ: columns per perspective a deferred ply can hold (more -> applied at once)
#ifndef SC_FX_SHARE   // the Windows section and the macOS file mapping; not measured elsewhere
#if defined(_WIN32) || defined(__APPLE__)
#define SC_FX_SHARE 1
#else
#define SC_FX_SHARE 0
#endif
#endif
#ifndef SC_NPS_SMALLCOLS   // fused_apply_src (AVX2): column-outer up to this many columns
#define SC_NPS_SMALLCOLS 2
#endif
#ifndef SC_NPS_L3S   // L3 over the nonzero screlu outputs only: always on AVX2; NEON behind this switch
#define SC_NPS_L3S 0
#endif
#ifndef SC_NPS_T1
#define SC_NPS_T1 1
#endif
constexpr auto kKnightAtt = [] { std::array<std::uint64_t, 64> t{}; for (int s = 0; s < 64; ++s) t[s] = knight_attacks_loop(s); return t; }();
constexpr auto kKingAtt   = [] { std::array<std::uint64_t, 64> t{}; for (int s = 0; s < 64; ++s) t[s] = king_attacks_loop(s); return t; }();
constexpr auto kPawnAtt   = [] {
    std::array<std::array<std::uint64_t, 64>, 2> t{};
    for (int c = 0; c < 2; ++c) for (int s = 0; s < 64; ++s) t[c][s] = pawn_attacks_loop(c, s);
    return t;
}();
inline std::uint64_t knight_attacks(int sq) { return SC_NPS_T1 ? kKnightAtt[sq] : knight_attacks_loop(sq); }
inline std::uint64_t king_attacks(int sq) { return SC_NPS_T1 ? kKingAtt[sq] : king_attacks_loop(sq); }
inline std::uint64_t pawn_attacks(int color, int sq) { return SC_NPS_T1 ? kPawnAtt[color][sq] : pawn_attacks_loop(color, sq); }
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
    // TI: per [piece][from] the attack set lut2 counts in, next to that square's offset (16 KB
    // in all, against lut2's 64 KB): lut2[p][f][t] == popcount(att & ((1 << t) - 1)).
    struct AttOff { std::uint64_t att; std::uint32_t off; std::uint32_t pad; };
    std::array<std::array<AttOff, 64>, 16> ao{};
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
            t.ao[piece][from] = {attacks, t.offsets[piece][from], 0};
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
// x86 TI: the lut2 term computed (bzhi + popcnt on BMI2) from the 16 KB attack/offset table
// instead of read from the 64 KB lut2, which does not fit the 32 KB L1d. Same integer.
#ifndef SC_NPS_TI
#if defined(__x86_64__) || defined(_M_X64)
#define SC_NPS_TI 1
#else
#define SC_NPS_TI 0
#endif
#endif
inline std::uint32_t make_threat_index(int attacker, int from, int to, int attacked) {
    if constexpr (SC_NPS_TI) {
        const Tables::AttOff& e = g_tab.ao[attacker][from];
        return g_tab.lut1[attacker][attacked][from < to ? 1 : 0] + e.off
             + static_cast<std::uint32_t>(__builtin_popcountll(e.att & ((1ull << to) - 1)));
    }
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
    // Output bucket per piece count, set at load: its two divisions (the divisor comes from the
    // header, so they are runtime idivs) once per count instead of on every eval.
    std::array<std::uint8_t, 65> obucket{};
    std::size_t base_dims = 0, total_inputs = 0;
    std::vector<float> l0w, l0b, l1w, l1b, l2b, l3w, l3b;
    WVec<float> l2w;               // 64-B aligned (G3): finish_body reads 32-float rows
    WVec<float> l2w_bm;            // G2: bucket-major [(b*L2 + i)*32 + o] -- 4 KB contiguous per bucket
    WVec<float> l1w_bm;            // FXB (float net, L2 = 32): bucket-major [(b*HL + i)*32 + o], 64 KB per bucket
    // quant path: int8 feature transformer + int8 L1 (body stays float);
    // int16 accumulator + int16 FT bias.
    WVec<std::int8_t> l0w_i8;      // 64-B aligned (G3): 512 B feature rows
    std::vector<std::int16_t> l0b_i;
    // float labeler path: fixed-point FT weights/bias (lround(w*FX_S)) for the
    // incremental int32 accumulator. Built at load for float (SCN4) nets.
    WVec<std::int32_t> l0w_fx;     // 147 MB int32 FT columns: 64-B aligned, large pages on Windows (as l0w_i8)
    std::shared_ptr<const std::int32_t> l0w_fx_shared;   // FXS: the same columns in a shared read-only mapping
    const std::int32_t* l0w_fx_p = nullptr;              // what the labeler reads: the mapping or l0w_fx
    std::vector<std::int32_t> l0b_fx;
    std::vector<std::int8_t> l1w_i;
    // L1 weights transposed to [bucket*L2 + o][HL] (HL contiguous) for the sdot
    // dot-product path; built from l1w_i on load.
    WVec<std::int8_t> l1w_dot;     // 64-B aligned (G3): 512 B rows per output
    std::vector<float> l3w_t;      // A6: [b*32 + i] = l3w[i*OB + b] -- contiguous per output bucket
    // int8 body: L2 weights as [bucket*32 + o][L2] (input contiguous) and L3 as
    // [bucket*32 + i]; per-layer global scale. Built from l2w/l3w on load (quant net).
    std::vector<std::int8_t> l2w_dot, l3w_dot;
    float l2_scale = 0.0f, l3_scale = 0.0f;
    // S1: L1 weights by input chunk, [bucket][hl/4 chunks][L2 outputs][4 inputs], for the sparse
    // L1 (one 128-byte block per nonzero chunk at L2 = 32). Built after the neuron permutation.
    WVec<std::int8_t> l1w_sp;
};
Net g_net;

// FXS: the fixed-point FT columns of a float net, from (or into) a cache file shared by every engine
// process: $TMPDIR/simplechess-fx-<key>.i32, key = FNV-1a over the dimensions, the fixed-point shift and
// the float weights' bits. Under an exclusive flock the first process converts the weights exactly as
// the private path does, writes them to a temporary file and renames it into place; every process then
// maps the file read-only and shared, so the page cache holds one copy for all of them. Returns null on
// any failure (no TMPDIR, disk full, a short file): the caller builds its private copy instead.
//
// Windows: the first process publishes the columns in a named pagefile-backed section
// (SEC_LARGE_PAGES when the lock-pages privilege allows), named by a hash of the int32 columns
// themselves; the others open it under a named mutex and map it read-only only after a full memcmp
// against their own conversion (a section from another build or layout is never used). Nothing
// outlives the processes. SCNNUE_NOFXSHARE=1 disables it at run time on both (A/B on one binary);
// SCNNUE_FXSHARE_LOG=1 prints which path was taken to stderr.
std::shared_ptr<const std::int32_t> fx_share_map([[maybe_unused]] const std::vector<float>& l0w,
                                                 [[maybe_unused]] std::size_t ti, [[maybe_unused]] std::size_t hl) {
#if defined(_WIN32)
    const std::size_t nw = ti * hl, bytes = nw * sizeof(std::int32_t);
    if (l0w.size() != nw || nw == 0 || std::getenv("SCNNUE_NOFXSHARE")) return nullptr;
    std::vector<std::int32_t> fx(nw);
    for (std::size_t i = 0; i < nw; ++i)
        fx[i] = static_cast<std::int32_t>(std::lround(double(l0w[i]) * FX_S));
    std::uint64_t h = 1469598103934665603ull ^ bytes;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(fx.data());
    for (std::size_t i = 0; i + 8 <= bytes; i += 8) {
        std::uint64_t x;
        std::memcpy(&x, b + i, 8);
        h = (h ^ x) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    char name[96], mname[104];
    std::snprintf(name, sizeof name, "Local\\SimpleChessFX_%016llx_%llu",
                  static_cast<unsigned long long>(h), static_cast<unsigned long long>(bytes));
    std::snprintf(mname, sizeof mname, "%s_m", name);
    HANDLE mtx = CreateMutexA(nullptr, FALSE, mname);
    if (!mtx) return nullptr;
    const DWORD wr = WaitForSingleObject(mtx, 120000);
    if (wr != WAIT_OBJECT_0 && wr != WAIT_ABANDONED) { CloseHandle(mtx); return nullptr; }
    void* view = nullptr;
    bool created = false, large = false;
    HANDLE sec = OpenFileMappingA(FILE_MAP_READ, FALSE, name);
    if (sec) {
        view = MapViewOfFile(sec, FILE_MAP_READ | FILE_MAP_LARGE_PAGES, 0, 0, 0);
        large = view != nullptr;
        if (!view) view = MapViewOfFile(sec, FILE_MAP_READ, 0, 0, 0);
        if (view && std::memcmp(view, fx.data(), bytes) != 0) { UnmapViewOfFile(view); view = nullptr; }
    } else {
        static const bool priv = sc_enable_lock_pages_nnue();
        const std::size_t gran = GetLargePageMinimum();
        if (priv && gran) {
            const std::size_t r = (bytes + gran - 1) / gran * gran;
            sec = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT | SEC_LARGE_PAGES,
                                     static_cast<DWORD>(r >> 32), static_cast<DWORD>(r), name);
            if (sec) {
                view = MapViewOfFile(sec, FILE_MAP_WRITE | FILE_MAP_LARGE_PAGES, 0, 0, 0);
                if (!view) view = MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, 0);
                if (!view) { CloseHandle(sec); sec = nullptr; }
                large = view != nullptr;
            }
        }
        if (!sec) {
            sec = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                     static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), name);
            if (sec) view = MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, 0);
        }
        if (view) { std::memcpy(view, fx.data(), bytes); created = true; }
    }
    ReleaseMutex(mtx);
    CloseHandle(mtx);
    if (std::getenv("SCNNUE_FXSHARE_LOG"))
        std::fprintf(stderr, "FXS: %s %s pages\n", view ? (created ? "created" : "opened") : "private",
                     large ? "large" : "small");
    if (!view) { if (sec) CloseHandle(sec); return nullptr; }
    return std::shared_ptr<const std::int32_t>(static_cast<const std::int32_t*>(view),
                                               [sec](const std::int32_t* v) { UnmapViewOfFile(v); CloseHandle(sec); });
#else
    const std::size_t nw = ti * hl, bytes = nw * sizeof(std::int32_t);
    if (l0w.size() != nw || nw == 0 || std::getenv("SCNNUE_NOFXSHARE")) return nullptr;
    std::uint64_t key = 1469598103934665603ull;
    auto mix = [&key](std::uint64_t v) { key = (key ^ v) * 1099511628211ull; };
    mix(ti); mix(hl); mix(FX_SHIFT);
    for (float w : l0w) { std::uint32_t u; std::memcpy(&u, &w, 4); mix(u); }
    const char* dir = std::getenv("TMPDIR");
    std::string path = std::string(dir && *dir ? dir : "/tmp");
    if (path.back() != '/') path += '/';
    char name[48];
    std::snprintf(name, sizeof(name), "simplechess-fx-%016llx.i32", static_cast<unsigned long long>(key));
    path += name;

    const int lfd = ::open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
    if (lfd < 0) return nullptr;
    void* map = MAP_FAILED;
    if (::flock(lfd, LOCK_EX) == 0) {
        int fd = ::open(path.c_str(), O_RDONLY);
        struct stat st{};
        if (fd >= 0 && (::fstat(fd, &st) != 0 || static_cast<std::size_t>(st.st_size) != bytes)) {
            ::close(fd);
            fd = -1;
        }
        if (fd < 0) {   // first process for this net: convert and publish
            std::vector<std::int32_t> fx(nw);
            for (std::size_t i = 0; i < nw; ++i)
                fx[i] = static_cast<std::int32_t>(std::lround(double(l0w[i]) * FX_S));
            const std::string tmp = path + "." + std::to_string(::getpid()) + ".tmp";
            const int wfd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
            bool ok = wfd >= 0;
            const char* src = reinterpret_cast<const char*>(fx.data());
            for (std::size_t done = 0; ok && done < bytes;) {
                const ssize_t w = ::write(wfd, src + done, bytes - done);
                if (w <= 0) ok = false;
                else done += static_cast<std::size_t>(w);
            }
            if (wfd >= 0 && ::close(wfd) != 0) ok = false;
            if (ok && ::rename(tmp.c_str(), path.c_str()) == 0) fd = ::open(path.c_str(), O_RDONLY);
            else ::unlink(tmp.c_str());
        }
        if (fd >= 0) {
            map = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
        }
        ::flock(lfd, LOCK_UN);
    }
    ::close(lfd);
    if (map == MAP_FAILED) return nullptr;
    return std::shared_ptr<const std::int32_t>(static_cast<const std::int32_t*>(map),
                                               [bytes](const std::int32_t* p) { ::munmap(const_cast<std::int32_t*>(p), bytes); });
#endif
}
bool g_base_only = false;  // SCNNUE_BASEONLY profiling probe (skip threats+pp)
bool g_no_accum = false;   // SCNNUE_NOACCUM profiling probe (enumerate, skip column adds)
bool g_verify = false;     // SCNNUE_VERIFY: cross-check incremental base vs recompute
unsigned g_net_gen = 0;    // bumped on every successful load (K1: Finny caches hold column sums)
int g_squeeze = 0;         // SCNNUE_SQUEEZE=N: fold threat/pp columns into N (cache probe)
bool g_no_update = false;  // SCNNUE_NOUPDATE: skip the threat/pp delta in acc_make (cost probe)
bool g_no_finny = false;   // SCNNUE_NOFINNY: force king moves to full refresh (A/B the finny path)
bool g_prefetch = true;    // SCNNUE_NOPREFETCH: disable FT-row prefetch in acc_make (A/B)
// NPS 3.6 campaign item P1: prefetch EVERY cache line of a feature column the moment the
// delta enumeration emits it (a 512-byte int8 column is four 128-byte lines on Apple
// Silicon), instead of only the first line of each column after the enumeration ends.
// A prefetch cannot change any value -> bit-identical.
// x86 keeps the first-line form: on the i7-9700 it measured +2.15% faster than every line.
#ifndef SC_NPS_P1
#if defined(__x86_64__) || defined(_M_X64)
#define SC_NPS_P1 0
#else
#define SC_NPS_P1 1
#endif
#endif
// The prefetch stride: the cache line (128 bytes on Apple Silicon, 64 on x86).
#if defined(__APPLE__) && defined(__aarch64__)
constexpr std::size_t kCacheLine = 128;
#else
constexpr std::size_t kCacheLine = 64;
#endif
// Prefetch shape knobs (x86 tuning): the stride of the P1 form (0 = kCacheLine), and how many
// lines per column the P1=0 form fetches after the enumeration (at that stride).
#ifndef SC_NPS_PF_STRIDE
#define SC_NPS_PF_STRIDE 0
#endif
#ifndef SC_NPS_PF_N   // x86: none at all (i7-9700: +0.40% over the first line, which beat every line)
#if defined(__x86_64__) || defined(_M_X64)
#define SC_NPS_PF_N 0
#else
#define SC_NPS_PF_N 1
#endif
#endif
constexpr std::size_t kPfStride = SC_NPS_PF_STRIDE ? SC_NPS_PF_STRIDE : kCacheLine;
// Locality hint of those column prefetches (3 = all levels / prefetcht0, 1 = L2 and out / prefetcht2).
#ifndef SC_NPS_PF_LOC
#define SC_NPS_PF_LOC 3
#endif
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

namespace {
void permute_neurons(Net& n);   // S1; defined after gather()
}

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
    // Only the shape the eval code is written for loads (a self-consistent file of another shape
    // would otherwise load "successfully" and then index out of range or divide by zero):
    //   input buckets == the king-bucket count of BUCKET_LAYOUT (king_bucket() returns 0..9),
    //   hl a multiple of 64 up to MAX_HL (the accumulator and pairwise SIMD loops; stack buffers),
    //   l2 a multiple of 16 up to MAX_L2 (the int8 dot loops step 16; stack buffers),
    //   1..32 output buckets (the output bucket divides the 32 men by the count).
    // The bounds also keep every array size below in 32 bits. Header fields are unsigned, so a
    // huge value turns negative here and fails the lower bounds.
    if (n.input_buckets != kKingBuckets || n.hl < 64 || n.hl > MAX_HL || n.hl % 64 != 0 ||
        n.l2 < 16 || n.l2 > MAX_L2 || n.l2 % 16 != 0 || n.out_buckets < 1 || n.out_buckets > 32)
        return false;
    const std::uint32_t HL = hdr[1], IB = hdr[2], L2 = hdr[3], OB = hdr[4];
    {   // output bucket of each piece count: (men - 2) / ceil(32 / OB), clamped to [0, OB)
        const int ob = n.out_buckets, divisor = (32 + ob - 1) / ob;
        for (int men = 0; men < 65; ++men)
            n.obucket[men] = static_cast<std::uint8_t>(std::clamp((men - 2) / divisor, 0, ob - 1));
    }
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
        if constexpr (SC_FX_SHARE) n.l0w_fx_shared = fx_share_map(n.l0w, TI, HL);
        if (!n.l0w_fx_shared) {
            n.l0w_fx.resize(NW);
            for (std::size_t i = 0; i < NW; ++i)
                n.l0w_fx[i] = static_cast<std::int32_t>(std::lround(double(n.l0w[i]) * FX_S));
        }
        n.l0w_fx_p = n.l0w_fx_shared ? n.l0w_fx_shared.get() : n.l0w_fx.data();
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
    // Float-net L1 the same way (the AVX2 input-major L1 reads the 32 weights of one input: contiguous
    // per bucket instead of rows at a 1 KB stride; same values, same fma order).
    if ((SC_FX_AVX2 || SC_FX_L1BM_NEON) && !n.quant && L2 == 32) {
        n.l1w_bm.resize(static_cast<std::size_t>(OB) * HL * 32);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t i = 0; i < HL; ++i)
                for (std::uint32_t o = 0; o < 32; ++o)
                    n.l1w_bm[(static_cast<std::size_t>(bkt) * HL + i) * 32 + o] =
                        n.l1w[static_cast<std::size_t>(i) * (OB * L2) + bkt * L2 + o];
    }
    build_tables();
    if constexpr (SC_NPS_S1)
        if (n.quant) permute_neurons(n);
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
    ++g_net_gen;
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

// ---- S1: neuron permutation + sparse-L1 weight layout ---------------------------
// Reorders the hidden neurons by how often their pairwise activation is zero (most often first),
// measured over deterministic random games from the start position (160 plies, so later phases
// count too), then lays L1 out by 4-input
// chunk. A neuron j moves as a unit: its two accumulator halves (j, j + hl/2), its FT bias, its FT
// weight in every feature row, and its L1 weight in both the side-to-move and the other block.
// Every L1 sum is the same integer in any neuron order, so the evaluation is unchanged.
void permute_neurons(Net& n) {
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    const std::size_t TI = n.total_inputs;
    const std::int8_t* l0 = n.l0w_i8.data();

    std::vector<std::uint32_t> zeros(static_cast<std::size_t>(half), 0);
    std::vector<std::int16_t> a_stm(static_cast<std::size_t>(hl)), a_ntm(static_cast<std::size_t>(hl));
    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto cr = [](int x) { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int game = 0; game < 128; ++game) {
        Board b;
        for (int ply = 0; ply < 160; ++ply) {
            if (ply >= 6) {
                for (int j = 0; j < hl; ++j) a_stm[j] = a_ntm[j] = n.l0b_i[j];
                gather(
                    b, n.base_dims,
                    [&](std::size_t f) { const std::int8_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a_stm[j] += w[j]; },
                    [&](std::size_t f) { const std::int8_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a_ntm[j] += w[j]; });
                for (int j = 0; j < half; ++j) {
                    zeros[j] += (cr(a_stm[j]) * cr(a_stm[j + half]) / QA == 0);
                    zeros[j] += (cr(a_ntm[j]) * cr(a_ntm[j + half]) / QA == 0);
                }
            }
            Movelist ml;
            chess::movegen::legalmoves(ml, b);
            if (ml.empty()) break;
            b.makeMove(ml[static_cast<int>(next() % static_cast<std::uint64_t>(ml.size()))]);
        }
    }
    std::vector<int> perm(static_cast<std::size_t>(half));
    for (int j = 0; j < half; ++j) perm[j] = j;
    std::stable_sort(perm.begin(), perm.end(), [&](int x, int y) { return zeros[x] > zeros[y]; });

    // FT weights (every feature row) and bias.
    std::vector<std::int8_t> row8(static_cast<std::size_t>(hl));
    for (std::size_t r = 0; r < TI; ++r) {
        std::int8_t* row = n.l0w_i8.data() + r * hl;
        for (int j = 0; j < half; ++j) { row8[j] = row[perm[j]]; row8[half + j] = row[half + perm[j]]; }
        std::memcpy(row, row8.data(), static_cast<std::size_t>(hl));
    }
    std::vector<std::int16_t> b16(n.l0b_i);
    for (int j = 0; j < half; ++j) { n.l0b_i[j] = b16[perm[j]]; n.l0b_i[half + j] = b16[half + perm[j]]; }
    // L1 inputs: [0, half) the side to move's neurons, [half, hl) the other side's.
    for (std::size_t r = 0; r < static_cast<std::size_t>(OB) * L2; ++r) {
        std::int8_t* row = n.l1w_dot.data() + r * hl;
        for (int j = 0; j < half; ++j) { row8[j] = row[perm[j]]; row8[half + j] = row[half + perm[j]]; }
        std::memcpy(row, row8.data(), static_cast<std::size_t>(hl));
    }
    // Sparse layout: [bucket][chunk k][output o][4 inputs 4k..4k+3].
    const int nch = hl / 4;
    n.l1w_sp.resize(static_cast<std::size_t>(OB) * nch * L2 * 4);
    for (int bkt = 0; bkt < OB; ++bkt)
        for (int k = 0; k < nch; ++k)
            for (int o = 0; o < L2; ++o)
                for (int t = 0; t < 4; ++t)
                    n.l1w_sp[((static_cast<std::size_t>(bkt) * nch + k) * L2 + o) * 4 + t] =
                        n.l1w_dot[(static_cast<std::size_t>(bkt) * L2 + o) * hl + 4 * k + t];
}

// ---- incremental base accumulator (thread-local) ----------------------------
// Persistent per-perspective BASE accumulator (int16), maintained across
// make/unmake so the base sum isn't rebuilt every eval. Only the base block is
// incremental (Stage A); threats+pp are still recomputed in eval_quant.
constexpr int ACC_STACK = MAX_PLY + 8;
// The search never makes a move from a node at ply >= MAX_PLY (both negamax and qsearch return
// first), so the stack index peaks at MAX_PLY and the overflow branch in acc_make (which drops
// to full recompute for the rest of the iteration) is unreachable. Keep it that way.
static_assert(ACC_STACK > MAX_PLY + 1, "accumulator stack must cover every search ply");
struct Acc {
    alignas(64) std::int16_t v[2][MAX_HL];  // v[0]=WHITE perspective, v[1]=BLACK perspective
};

// Fixed-point (int32) accumulator stack for the float labeler path. Shares g_ply and
// the g_bb bitboard stack with the quant path; only one of g_stack / g_stack_fx is
// live per run (chosen by the loaded net format).
struct AccFx {
    alignas(64) std::int32_t v[2][MAX_HL];
};

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
    const std::int32_t* l0 = n.l0w_fx_p;
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

// Per-ply incremental board bitboards, maintained in lockstep with the accumulator stack so a
// normal move reads its before-state from bb[ply] instead of rebuilding a BB from board.at()
// every make. Same ply / reset / make / unmake / null hooks.

// Item #9 (novelty ledger): how many NEW attacks on a HIGHER-valued enemy piece the last acc_make's
// move created, counted for the mover's side only. The threat-delta emitter already has the attacker
// and the victim in hand for every column it adds, so this is two comparisons per emission rather
// than any new enumeration -- the same "read what the net already computed" shape as item #8, but a
// semantically sharp count (this move makes a threat) instead of a magnitude of feature change.
// Zero on the king-cross refresh path, where the threat set is rebuilt rather than emitted as a
// delta: a false negative on a few percent of moves, never a false positive.

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

}  // namespace

// Per-search-thread incremental state (accumulator stacks, bitboard stack, delta scratch). The
// search owns one per worker and passes it to every call, so the hot path makes no thread-local
// lookups (NPS 3.6 item C1: on macOS every thread_local access is a call through the TLV thunk).
struct Context {
    std::vector<Acc>   stack;     // int16 accumulators (quant net), one per ply
    std::vector<AccFx> stack_fx;  // int32 fixed-point accumulators (float labeler net)
    std::vector<BB>    bb;        // per-ply board bitboards, in lockstep with the stack
    int ply = -1;                 // -1 = uninitialised -> evaluate() recomputes from the board
    int threats_made = 0;         // item #9: see acc_threats_made()
    const void* addp[2][1024];    // delta column scratch (acc_make)
    const void* subp[2][1024];
    const void* ra[40];           // king-rebucket scratch
    const void* rs[40];
    // K1: per [perspective][mirror][bucket], the base-column sum of the pieces in `by` (as of the
    // last time this bucket was used).
    struct Finny {
        alignas(64) std::int16_t acc[MAX_HL];
        std::uint64_t by[16];
        bool valid = false;
    };
    Finny finny[2][2][kKingBuckets];
    unsigned finny_gen = 0;      // g_net_gen the caches were built for
    // evaluate()'s full-recompute fallback (no incremental accumulator yet). Kept here, not in
    // evaluate()'s frame: a frame over one page makes every call run the Windows stack probe.
    alignas(64) std::int16_t fb[2][MAX_HL];
    // FXZ: stack_fx[p] holds its values only when fx_ok[p]; otherwise fx_pend[p] has the column lists
    // acc_make recorded for it (adds in col[s][0, na), subs in col[s][na, na + ns)).
    struct FxPend { int na[2], ns[2]; const std::int32_t* col[2][kFxPendCap]; };
    std::vector<std::uint8_t> fx_ok;
    std::vector<FxPend> fx_pend;
#if SC_FXP
    AccFx fx_fb;   // FXP: the float eval's from-scratch fallback (kept out of its frame)
#endif
#ifdef SC_LZSTAT
    std::uint8_t lz_needed[ACC_STACK] = {};   // measurement only: was this ply's accumulator read?
#endif
};
// FXZ: bring stack_fx[p] up to date from the newest computed ancestor (fx_ok[0] is always set),
// one ply at a time, each the src->dst fused pass acc_make would have made.
inline void fx_ensure(Context& cx, int p) {
    if (cx.fx_ok[p]) return;
    int q = p - 1;
    while (!cx.fx_ok[q]) --q;
    const int hl = g_net.hl;
    for (int k = q + 1; k <= p; ++k) {
        const Context::FxPend& e = cx.fx_pend[k];
        for (int sd = 0; sd < 2; ++sd)
            fused_apply_src_fx(cx.stack_fx[k].v[sd], cx.stack_fx[k - 1].v[sd], e.col[sd], e.na[sd],
                               e.col[sd] + e.na[sd], e.ns[sd], hl);
        cx.fx_ok[k] = 1;
    }
}
#ifdef SC_LZSTAT
// Measurement build: how many accumulator updates are popped without that ply or any descendant
// ever being evaluated (what lazy updates could skip). Printed to stderr at exit.
std::atomic<long> g_lz_upd{0}, g_lz_skip{0}, g_lz_null{0}, g_lz_null_skip{0};
// Move kinds acc_make saw: castle, promotion, en passant, king move across the mirror (full refresh),
// king move across a bucket only (rebucket), other king move.
std::atomic<long> g_lz_kind[6] = {};
struct LzPrinter {
    ~LzPrinter() {
        std::fprintf(stderr, "LZSTAT updates %ld skipped %ld (%.2f%%) | null %ld skipped %ld\n", g_lz_upd.load(),
                     g_lz_skip.load(), 100.0 * g_lz_skip.load() / std::max(1L, g_lz_upd.load()), g_lz_null.load(),
                     g_lz_null_skip.load());
        std::fprintf(stderr, "LZSTAT kinds castle %ld promo %ld ep %ld kmirror %ld krebucket %ld kother %ld\n",
                     g_lz_kind[0].load(), g_lz_kind[1].load(), g_lz_kind[2].load(), g_lz_kind[3].load(),
                     g_lz_kind[4].load(), g_lz_kind[5].load());
    }
} g_lz_printer;
#endif

Context* context_new() { return new Context(); }
void context_free(Context* c) { delete c; }

namespace {
// The context of callers that do not own one (UCI debug commands): one per thread.
Context& tl_context() {
    thread_local Context c;
    return c;
}


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
    // A2 (NPS 3.6): 128 int16 per chunk, one set of 16 accumulators -- half the chunk iterations
    // and column-pointer reloads of the 64-lane form. Same wrapping int16 sums -> bit-identical.
    if constexpr (SC_NPS_A2) {
        for (; c + 128 <= hl; c += 128) {
            int16x8_t v0 = vld1q_s16(src + c),       v1 = vld1q_s16(src + c + 8),
                      v2 = vld1q_s16(src + c + 16),  v3 = vld1q_s16(src + c + 24),
                      v4 = vld1q_s16(src + c + 32),  v5 = vld1q_s16(src + c + 40),
                      v6 = vld1q_s16(src + c + 48),  v7 = vld1q_s16(src + c + 56),
                      v8 = vld1q_s16(src + c + 64),  v9 = vld1q_s16(src + c + 72),
                      v10 = vld1q_s16(src + c + 80), v11 = vld1q_s16(src + c + 88),
                      v12 = vld1q_s16(src + c + 96), v13 = vld1q_s16(src + c + 104),
                      v14 = vld1q_s16(src + c + 112), v15 = vld1q_s16(src + c + 120);
            for (int k = 0; k < na; ++k) {
                const std::int8_t* P = add[k];   // SCN_COL64 adds the chunk offset c
                SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, P, vaddq_s16)
                SCN_COL64(v8, v9, v10, v11, v12, v13, v14, v15, P + 64, vaddq_s16)
            }
            for (int k = 0; k < ns; ++k) {
                const std::int8_t* P = sub[k];   // SCN_COL64 adds the chunk offset c
                SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, P, vsubq_s16)
                SCN_COL64(v8, v9, v10, v11, v12, v13, v14, v15, P + 64, vsubq_s16)
            }
            vst1q_s16(dst + c,       v0);  vst1q_s16(dst + c + 8,   v1);  vst1q_s16(dst + c + 16,  v2);
            vst1q_s16(dst + c + 24,  v3);  vst1q_s16(dst + c + 32,  v4);  vst1q_s16(dst + c + 40,  v5);
            vst1q_s16(dst + c + 48,  v6);  vst1q_s16(dst + c + 56,  v7);  vst1q_s16(dst + c + 64,  v8);
            vst1q_s16(dst + c + 72,  v9);  vst1q_s16(dst + c + 80,  v10); vst1q_s16(dst + c + 88,  v11);
            vst1q_s16(dst + c + 96,  v12); vst1q_s16(dst + c + 104, v13); vst1q_s16(dst + c + 112, v14);
            vst1q_s16(dst + c + 120, v15);
        }
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
    // A2d: makes with at most SC_NPS_SMALLCOLS columns go column-outer (long predictable loops).
    if (na + ns <= SC_NPS_SMALLCOLS) {
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
    // A2 (NPS 3.6) on AVX2: 128 int16 per chunk, one set of 8 ymm accumulators -- half the chunk
    // iterations and column-pointer reloads of the 64-lane form. int16 adds wrap -> bit-identical.
    if constexpr (SC_NPS_A2) {
        for (; c + 128 <= hl; c += 128) {
            auto ldv = [&](int o) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + o)); };
            __m256i v0 = ldv(0), v1 = ldv(16), v2 = ldv(32), v3 = ldv(48),
                    v4 = ldv(64), v5 = ldv(80), v6 = ldv(96), v7 = ldv(112);
            for (int k = 0; k < na; ++k) {
                const std::int8_t* p = add[k] + c;
                v0 = _mm256_add_epi16(v0, w16(p));      v1 = _mm256_add_epi16(v1, w16(p + 16));
                v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
                v4 = _mm256_add_epi16(v4, w16(p + 64)); v5 = _mm256_add_epi16(v5, w16(p + 80));
                v6 = _mm256_add_epi16(v6, w16(p + 96)); v7 = _mm256_add_epi16(v7, w16(p + 112));
            }
            for (int k = 0; k < ns; ++k) {
                const std::int8_t* p = sub[k] + c;
                v0 = _mm256_sub_epi16(v0, w16(p));      v1 = _mm256_sub_epi16(v1, w16(p + 16));
                v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
                v4 = _mm256_sub_epi16(v4, w16(p + 64)); v5 = _mm256_sub_epi16(v5, w16(p + 80));
                v6 = _mm256_sub_epi16(v6, w16(p + 96)); v7 = _mm256_sub_epi16(v7, w16(p + 112));
            }
            auto stv = [&](int o, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + o), v); };
            stv(0, v0); stv(16, v1); stv(32, v2); stv(48, v3); stv(64, v4); stv(80, v5); stv(96, v6); stv(112, v7);
        }
    }
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
#if defined(__AVX2__)
    if constexpr (SC_FX_AVX2) {   // one src -> dst pass, 64 int32 (8 ymm) per chunk; int32 sums -> bit-exact
        auto ld = [](const std::int32_t* q) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)); };
        auto st = [](std::int32_t* q, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(q), v); };
        int c = 0;
        for (; c + 64 <= hl; c += 64) {
            __m256i v0 = ld(src + c),      v1 = ld(src + c + 8),  v2 = ld(src + c + 16), v3 = ld(src + c + 24),
                    v4 = ld(src + c + 32), v5 = ld(src + c + 40), v6 = ld(src + c + 48), v7 = ld(src + c + 56);
            for (int k = 0; k < na; ++k) {
                const std::int32_t* q = add[k] + c;
                v0 = _mm256_add_epi32(v0, ld(q));      v1 = _mm256_add_epi32(v1, ld(q + 8));
                v2 = _mm256_add_epi32(v2, ld(q + 16)); v3 = _mm256_add_epi32(v3, ld(q + 24));
                v4 = _mm256_add_epi32(v4, ld(q + 32)); v5 = _mm256_add_epi32(v5, ld(q + 40));
                v6 = _mm256_add_epi32(v6, ld(q + 48)); v7 = _mm256_add_epi32(v7, ld(q + 56));
            }
            for (int k = 0; k < ns; ++k) {
                const std::int32_t* q = sub[k] + c;
                v0 = _mm256_sub_epi32(v0, ld(q));      v1 = _mm256_sub_epi32(v1, ld(q + 8));
                v2 = _mm256_sub_epi32(v2, ld(q + 16)); v3 = _mm256_sub_epi32(v3, ld(q + 24));
                v4 = _mm256_sub_epi32(v4, ld(q + 32)); v5 = _mm256_sub_epi32(v5, ld(q + 40));
                v6 = _mm256_sub_epi32(v6, ld(q + 48)); v7 = _mm256_sub_epi32(v7, ld(q + 56));
            }
            st(dst + c, v0);      st(dst + c + 8, v1);  st(dst + c + 16, v2); st(dst + c + 24, v3);
            st(dst + c + 32, v4); st(dst + c + 40, v5); st(dst + c + 48, v6); st(dst + c + 56, v7);
        }
        for (; c < hl; c += 8) {   // hl % 64 tail (empty for hl = 512)
            __m256i v = ld(src + c);
            for (int k = 0; k < na; ++k) v = _mm256_add_epi32(v, ld(add[k] + c));
            for (int k = 0; k < ns; ++k) v = _mm256_sub_epi32(v, ld(sub[k] + c));
            st(dst + c, v);
        }
        return;
    }
#endif
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
    [[maybe_unused]] unsigned nz3 = 0xFFFFFFFFu;   // AVX2: the nonzero x2 (L3S)
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
        if constexpr (SC_NPS_L3S) {
            nz3 = 0;
            for (int o = 0; o < 32; ++o) { x2[o] = screlu(tmp[o]); nz3 |= static_cast<unsigned>(x2[o] != 0.0f) << o; }
        } else {
            for (int o = 0; o < 32; ++o) x2[o] = screlu(tmp[o]);
        }
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
        auto row = [&](int i) {
            const __m256 xi = _mm256_set1_ps(x1[i]);
            const float* w = w2base + static_cast<std::size_t>(i) * wstride;
            c0 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w),      c0);
            c1 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 8),  c1);
            c2 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 16), c2);
            c3 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 24), c3);
        };
        if (!g_l1dense && L2 == 32) {
            // NNZ-sparse L2 (the NEON §L3 form): only the nonzero x1 (screlu >= 0, about half are
            // zero), in ascending i, so every output's fma chain is the dense one minus terms that
            // add +0 -- the same value (a +-0 sum can only flip a zero's sign, which the screlu below
            // erases). Shorter chains: the dense loop is bound by fma latency, 32 deep.
            const __m256 z = _mm256_setzero_ps();
            unsigned m = 0;
            for (int q = 0; q < 4; ++q)
                m |= static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x1 + 8 * q), z, _CMP_NEQ_OQ))) << (8 * q);
            while (m) { row(__builtin_ctz(m)); m &= m - 1; }
        } else {
            for (int i = 0; i < L2; ++i) row(i);
        }
        alignas(32) float tmp[32];
        _mm256_store_ps(tmp, c0); _mm256_store_ps(tmp + 8, c1);
        _mm256_store_ps(tmp + 16, c2); _mm256_store_ps(tmp + 24, c3);
        nz3 = 0;
        for (int o = 0; o < 32; o += 8) {   // 8-wide screlu, same ops per lane (clamp, square); A7
            __m256 v = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(tmp + o), _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
            const __m256 sq = _mm256_mul_ps(v, v);
            _mm256_storeu_ps(x2.data() + o, sq);
            nz3 |= static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(sq, _mm256_setzero_ps(), _CMP_NEQ_OQ))) << o;
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
#if defined(__AVX2__) || (defined(__ARM_NEON) && SC_NPS_L3S)
        // L3S: the same chain over the nonzero x2 only, ascending (dropped terms add +0: at most a
        // zero's sign changes, and lround maps +-0 alike). The dense chain is 32 dependent fma.
        if (!g_l1dense) {
            while (nz3) { const int i = __builtin_ctz(nz3); nz3 &= nz3 - 1; y += x2[i] * w3[i]; }
        } else
#endif
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
// S1: kNnzLut as uint16 offsets, for the sparse L1's chunk-index list.
alignas(8) static const std::uint16_t kNnzLut16[16][4] = {
    {0,0,0,0},{0,0,0,0},{1,0,0,0},{0,1,0,0},{2,0,0,0},{0,2,0,0},{1,2,0,0},{0,1,2,0},
    {3,0,0,0},{0,3,0,0},{1,3,0,0},{0,1,3,0},{2,3,0,0},{0,2,3,0},{1,2,3,0},{0,1,2,3}};
#if defined(__AVX2__)
// S1 on AVX2: for an 8-bit nonzero-chunk mask, the ascending set-bit offsets (one 128-bit store
// per 8 chunks, no data-dependent branch).
alignas(16) static constexpr auto kNnzLut8 = [] {
    std::array<std::array<std::uint16_t, 8>, 256> t{};
    for (int m = 0; m < 256; ++m) {
        int c = 0;
        for (int i = 0; i < 8; ++i)
            if ((m >> i) & 1) t[m][c++] = static_cast<std::uint16_t>(i);
    }
    return t;
}();
#endif

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
#if SC_FXP
template <bool SWAP>
[[gnu::noinline]] Value eval_float_t0(Context& cx, const Board& board);

[[gnu::noinline, gnu::cold]] void fx_verify_now(const AccFx& a, int stmp, const Board& board, int ply) {
    AccFx chk;
    refresh_into_fx(chk, board);
    for (int j = 0; j < g_net.hl; ++j)
        if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
            std::fprintf(stderr, "[nnue] FX ACC MISMATCH ply=%d j=%d inc=%d,%d chk=%d,%d\n",
                         ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
            break;
        }
}

template <bool SWAP>
[[gnu::noinline]] Value eval_float_t(Context& cx, const Board& board) {   // its buffers stay out of evaluate()'s frame
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    if (g_l1dense || L2 != 32 || (hl & 63) != 0) return eval_float_t0<SWAP>(cx, board);
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    const AccFx* ap;
    if (cx.ply >= 0) {  // search has the incremental accumulator in sync
        if constexpr (SC_FX_LAZY) fx_ensure(cx, cx.ply);
        ap = &cx.stack_fx[cx.ply];
#ifdef SC_LZSTAT
        for (int p = cx.ply; p >= 0 && !cx.lz_needed[p]; --p) cx.lz_needed[p] = 1;
#endif
        if (g_verify) fx_verify_now(*ap, stmp, board, cx.ply);   // incremental must equal a fresh refresh
    } else {  // outside search (bare `eval`): from-scratch fixed-point
        refresh_into_fx(cx.fx_fb, board);
        ap = &cx.fx_fb;
    }
    // h = crelu(a) * crelu(b) of (float)v * 2^-22 (the FXA fill and pairwise ops, lane for lane), and
    // per 64 inputs the mask of nonzero h for the L1. SWAP exchanges the two perspectives' halves.
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
    alignas(64) float h[MAX_HL];
    std::uint64_t hm[MAX_HL / 64] = {};
    {
        const __m256 inv = _mm256_set1_ps(float(1.0 / FX_S)), vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        auto cl = [&](const std::int32_t* q) {
            return _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(
                       _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q))), inv), vz), vo);
        };
        const std::int32_t* vs = ap->v[stmp];
        const std::int32_t* vn = ap->v[1 - stmp];
        for (int i = 0; i < half; i += 8) {
            const __m256 ps = _mm256_mul_ps(cl(vs + i), cl(vs + i + half));
            const __m256 pn = _mm256_mul_ps(cl(vn + i), cl(vn + i + half));
            _mm256_store_ps(h + off_stm + i, ps);
            _mm256_store_ps(h + off_ntm + i, pn);
            hm[(off_stm + i) >> 6] |= static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_cmp_ps(ps, vz, _CMP_NEQ_OQ)))) << ((off_stm + i) & 63);
            hm[(off_ntm + i) >> 6] |= static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_cmp_ps(pn, vz, _CMP_NEQ_OQ)))) << ((off_ntm + i) & 63);
        }
    }
    const int b = n.obucket[board.occ().count()];
    float x1[MAX_L2];
    {   // the FXA/FXB L1: input-major over the nonzero h[i] ascending, all 32 outputs in 4 ymm
        const bool bm = !n.l1w_bm.empty();
        const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
        const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                                : &n.l1w[static_cast<std::size_t>(b) * L2];
        const float* l1bb  = &n.l1b[b * L2];
        __m256 a0 = _mm256_loadu_ps(l1bb), a1 = _mm256_loadu_ps(l1bb + 8),
               a2 = _mm256_loadu_ps(l1bb + 16), a3 = _mm256_loadu_ps(l1bb + 24);
        for (int g = 0; g < hl; g += 64) {
            std::uint64_t m = hm[g >> 6];
            while (m) {
                const int i = g + __builtin_ctzll(m);
                m &= m - 1;
                const __m256 hv = _mm256_set1_ps(h[i]);
                const float* w = wbase + static_cast<std::size_t>(i) * istride;
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w),      a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 8),  a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 24), a3);
            }
        }
        const __m256 vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        const __m256 acc[4] = {a0, a1, a2, a3};
        for (int q = 0; q < 4; ++q) {   // 8-wide screlu, same ops per lane (clamp, square)
            const __m256 v = _mm256_min_ps(_mm256_max_ps(acc[q], vz), vo);
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    }
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}
// The float eval before FXP: on x86 its fallback (dense-L1 probe, L2 != 32, HL % 64 != 0).
#define SC_FX_EVAL0 eval_float_t0
#else
#define SC_FX_EVAL0 eval_float_t
#endif

template <bool SWAP>
[[gnu::noinline]] Value SC_FX_EVAL0(Context& cx, const Board& board) {
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
#if defined(__AVX2__)
        if constexpr (SC_FX_AVX2) {   // the NEON §L4 form: (float)v * 2^-22, equal to the scalar one after CReLU
            const __m256 inv = _mm256_set1_ps(float(1.0 / FX_S));
            for (int j = 0; j < hl; j += 8) {
                _mm256_store_ps(acc_stm + j, _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a.v[stmp][j]))), inv));
                _mm256_store_ps(acc_ntm + j, _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a.v[1 - stmp][j]))), inv));
            }
            return;
        }
#endif
        for (int j = 0; j < hl; ++j) {
            acc_stm[j] = float(double(a.v[stmp][j])     / FX_S);
            acc_ntm[j] = float(double(a.v[1 - stmp][j]) / FX_S);
        }
#endif
    };
    if (cx.ply >= 0) {  // search has the incremental accumulator in sync
        if constexpr (SC_FX_LAZY) fx_ensure(cx, cx.ply);
        const AccFx& a = cx.stack_fx[cx.ply];
#ifdef SC_LZSTAT
        for (int p = cx.ply; p >= 0 && !cx.lz_needed[p]; --p) cx.lz_needed[p] = 1;
#endif
        if (g_verify) {  // incremental must equal a fresh refresh, bit-exact
            AccFx chk; refresh_into_fx(chk, board);
            for (int j = 0; j < hl; ++j)
                if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
                    std::fprintf(stderr, "[nnue] FX ACC MISMATCH ply=%d j=%d inc=%d,%d chk=%d,%d\n",
                                 cx.ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
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
    int pw0 = 0;
#if defined(__AVX2__)
    if constexpr (SC_FX_AVX2) {   // the NEON §2.3 form, 8 lanes (no input is -0.0: an int 0 converts to +0)
        const __m256 vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        auto cl = [&](const float* q) { return _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(q), vz), vo); };
        for (; pw0 + 8 <= half; pw0 += 8) {
            _mm256_storeu_ps(h + off_stm + pw0, _mm256_mul_ps(cl(acc_stm + pw0), cl(acc_stm + pw0 + half)));
            _mm256_storeu_ps(h + off_ntm + pw0, _mm256_mul_ps(cl(acc_ntm + pw0), cl(acc_ntm + pw0 + half)));
        }
    }
#endif
    for (int i = pw0; i < half; ++i) {
        h[off_stm + i] = crelu(acc_stm[i]) * crelu(acc_stm[i + half]);
        h[off_ntm + i] = crelu(acc_ntm[i]) * crelu(acc_ntm[i + half]);
    }
#endif
    const int b = n.obucket[board.occ().count()];
    float x1[MAX_L2];
#if defined(__ARM_NEON)
    // Row i at wbase + i*istride + o: the bucket-major copy when built (FXB), else the [i][bucket][o] rows.
    const bool bm = !n.l1w_bm.empty();
    const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
    const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                            : &n.l1w[static_cast<std::size_t>(b) * L2];
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
#if defined(__AVX2__)
    if (SC_FX_AVX2 && !g_l1dense && L2 == 32 && (hl & 63) == 0) {
        // The NEON §L1 form: input-major over the nonzero h[i] (ascending: a 64-bit mask per 64 inputs,
        // walked with tzcnt), all 32 outputs in 4 ymm. Per output that is the scalar loop's fma chain
        // (s = fma(h[i], w, s), ascending i, from the bias) minus terms that add +0 -- the same value,
        // a +-0 sum only flipping a zero's sign, which screlu erases.
        const bool bm = !n.l1w_bm.empty();
        const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
        const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                                : &n.l1w[static_cast<std::size_t>(b) * L2];
        const float* l1bb  = &n.l1b[b * L2];
        __m256 a0 = _mm256_loadu_ps(l1bb), a1 = _mm256_loadu_ps(l1bb + 8),
               a2 = _mm256_loadu_ps(l1bb + 16), a3 = _mm256_loadu_ps(l1bb + 24);
        const __m256 vz = _mm256_setzero_ps();
        for (int g = 0; g < hl; g += 64) {
            std::uint64_t m = 0;
            for (int t = 0; t < 8; ++t)
                m |= static_cast<std::uint64_t>(static_cast<unsigned>(_mm256_movemask_ps(
                         _mm256_cmp_ps(_mm256_load_ps(h + g + 8 * t), vz, _CMP_NEQ_OQ)))) << (8 * t);
            while (m) {
                const int i = g + __builtin_ctzll(m);
                m &= m - 1;
                const __m256 hv = _mm256_set1_ps(h[i]);
                const float* w = wbase + static_cast<std::size_t>(i) * istride;
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w),      a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 8),  a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 24), a3);
            }
        }
        const __m256 acc[4] = {a0, a1, a2, a3};
        for (int q = 0; q < 4; ++q) {   // 8-wide screlu, same ops per lane (clamp, square)
            const __m256 v = _mm256_min_ps(_mm256_max_ps(acc[q], vz), _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    } else
#endif
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
// SCNNUE_VERIFY: the incremental accumulator against a fresh recompute (its 2 KB Acc stays out
// of evaluate()'s frame).
[[gnu::noinline, gnu::cold]] void verify_acc_now(const Acc& a, int stmp, const Board& board, int ply) {
    const int hl = g_net.hl;
    Acc chk;
    refresh_into(chk, board);
    for (int j = 0; j < hl; ++j)
        if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
            std::fprintf(stderr, "[nnue] ACC MISMATCH ply=%d j=%d  inc=%d,%d chk=%d,%d\n",
                         ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
            break;
        }
}
template <bool SWAP>
Value eval_quant_t(Context& cx, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2;

    // The incremental path reads the accumulator in place through these pointers (no 2KB
    // memcpy — that copy was profiled at ~11% of eval on the play path; byte-identical, same
    // int16 values). The gather fallback fills the Context's buffers.
    const std::int16_t* acc_stm;
    const std::int16_t* acc_ntm;
    const std::int8_t* l0 = n.l0w_i8.data();

    // Full accumulator (base + threats + pp) from the incremental state if the
    // search has it in sync — no gather at all. Else full recompute (fallback).
    const bool have_acc = (cx.ply >= 0);
    if (have_acc) {
        const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
        const Acc& a = cx.stack[cx.ply];
        acc_stm = a.v[stmp];       // read the accumulator in place — no copy
        acc_ntm = a.v[1 - stmp];
#ifdef SC_LZSTAT
        for (int p = cx.ply; p >= 0 && !cx.lz_needed[p]; --p) cx.lz_needed[p] = 1;
#endif
        if (g_verify) verify_acc_now(a, stmp, board, cx.ply);   // vs a fresh recompute
    } else {
        std::int16_t* acc_buf_stm = cx.fb[0];
        std::int16_t* acc_buf_ntm = cx.fb[1];
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
    if constexpr (SC_NPS_P2) {
        (void)vM; (void)pairwise8;
        const uint16x8_t one = vdupq_n_u16(1);
        auto pw16 = [&](const std::int16_t* src, std::int8_t* dst) {
            const uint16x8_t a0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src), vz), vqa));
            const uint16x8_t a1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + 8), vz), vqa));
            const uint16x8_t b0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + half), vz), vqa));
            const uint16x8_t b1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + half + 8), vz), vqa));
            uint16x8_t x0 = vmlaq_u16(one, a0, b0), x1 = vmlaq_u16(one, a1, b1);   // 1 + a*b
            x0 = vsraq_n_u16(x0, x0, 7);  x1 = vsraq_n_u16(x1, x1, 7);             // + (1 + a*b) >> 7
            vst1q_u8(reinterpret_cast<std::uint8_t*>(dst), vshrn_high_n_u16(vshrn_n_u16(x0, 7), x1, 7));
        };
        for (int i = 0; i < half; i += 16) {
            pw16(acc_stm + i, h8 + off_stm + i);
            pw16(acc_ntm + i, h8 + off_ntm + i);
        }
    } else {
        for (int i = 0; i < half; i += 8) {
            pairwise8(acc_stm + i, h8 + off_stm + i);
            pairwise8(acc_ntm + i, h8 + off_ntm + i);
        }
    }
#else
    auto cr = [](int x) -> int { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int i = 0; i < half; ++i) {
        h8[off_stm + i] = static_cast<std::int8_t>((cr(acc_stm[i]) * cr(acc_stm[i + half])) / QA);
        h8[off_ntm + i] = static_cast<std::int8_t>((cr(acc_ntm[i]) * cr(acc_ntm[i + half])) / QA);
    }
#endif
    const int b = n.obucket[board.occ().count()];
    const float DEQ = float(QA) * QB;  // h8 already carries the 1/QA scale
    const std::int8_t* wbucket = n.l1w_dot.data() + static_cast<std::size_t>(b) * L2 * hl;
    float x1[MAX_L2];
#if defined(__ARM_FEATURE_DOTPROD)
    if (SC_NPS_S1 && L2 == 32 && !n.l1w_sp.empty()) {
        // S1 sparse L1: list the nonzero 4-byte chunks of h8 (ascending), then for each one add its
        // 4 inputs x all 32 outputs (eight vdotq over one 128-byte weight block). Two accumulator
        // sets alternate chunks to halve the dependency chains. Zero chunks add exactly 0, and
        // integer sums are order-independent -> the same int32 per output as the dense form.
        alignas(16) std::uint16_t nz[MAX_HL / 4 + 8];
        int nnz = 0;
        const std::uint32_t* h32 = reinterpret_cast<const std::uint32_t*>(h8);
        const uint32x4_t bit = {1u, 2u, 4u, 8u};
        for (int k = 0; k < hl / 4; k += 4) {
            const uint32x4_t v = vld1q_u32(h32 + k);
            const unsigned m = vaddvq_u32(vandq_u32(vtstq_u32(v, v), bit));
            vst1_u16(nz + nnz, vadd_u16(vdup_n_u16(static_cast<std::uint16_t>(k)), vld1_u16(kNnzLut16[m])));
            nnz += kNnzPop[m];
        }
        const std::int8_t* W = n.l1w_sp.data() + static_cast<std::size_t>(b) * (hl / 4) * 128;
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        int32x4_t c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0, c6 = a0, c7 = a0;
        int j = 0;
        for (; j + 2 <= nnz; j += 2) {
            const int k0 = nz[j], k1 = nz[j + 1];
            const int8x16_t i0 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k0]));
            const int8x16_t i1 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k1]));
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            const std::int8_t* w1 = W + static_cast<std::size_t>(k1) * 128;
            a0 = vdotq_s32(a0, vld1q_s8(w0),      i0);  c0 = vdotq_s32(c0, vld1q_s8(w1),      i1);
            a1 = vdotq_s32(a1, vld1q_s8(w0 + 16), i0);  c1 = vdotq_s32(c1, vld1q_s8(w1 + 16), i1);
            a2 = vdotq_s32(a2, vld1q_s8(w0 + 32), i0);  c2 = vdotq_s32(c2, vld1q_s8(w1 + 32), i1);
            a3 = vdotq_s32(a3, vld1q_s8(w0 + 48), i0);  c3 = vdotq_s32(c3, vld1q_s8(w1 + 48), i1);
            a4 = vdotq_s32(a4, vld1q_s8(w0 + 64), i0);  c4 = vdotq_s32(c4, vld1q_s8(w1 + 64), i1);
            a5 = vdotq_s32(a5, vld1q_s8(w0 + 80), i0);  c5 = vdotq_s32(c5, vld1q_s8(w1 + 80), i1);
            a6 = vdotq_s32(a6, vld1q_s8(w0 + 96), i0);  c6 = vdotq_s32(c6, vld1q_s8(w1 + 96), i1);
            a7 = vdotq_s32(a7, vld1q_s8(w0 + 112), i0); c7 = vdotq_s32(c7, vld1q_s8(w1 + 112), i1);
        }
        if (j < nnz) {
            const int k0 = nz[j];
            const int8x16_t i0 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k0]));
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            a0 = vdotq_s32(a0, vld1q_s8(w0),      i0);  a1 = vdotq_s32(a1, vld1q_s8(w0 + 16), i0);
            a2 = vdotq_s32(a2, vld1q_s8(w0 + 32), i0);  a3 = vdotq_s32(a3, vld1q_s8(w0 + 48), i0);
            a4 = vdotq_s32(a4, vld1q_s8(w0 + 64), i0);  a5 = vdotq_s32(a5, vld1q_s8(w0 + 80), i0);
            a6 = vdotq_s32(a6, vld1q_s8(w0 + 96), i0);  a7 = vdotq_s32(a7, vld1q_s8(w0 + 112), i0);
        }
        alignas(16) std::int32_t sum[32];
        vst1q_s32(sum,      vaddq_s32(a0, c0));  vst1q_s32(sum + 4,  vaddq_s32(a1, c1));
        vst1q_s32(sum + 8,  vaddq_s32(a2, c2));  vst1q_s32(sum + 12, vaddq_s32(a3, c3));
        vst1q_s32(sum + 16, vaddq_s32(a4, c4));  vst1q_s32(sum + 20, vaddq_s32(a5, c5));
        vst1q_s32(sum + 24, vaddq_s32(a6, c6));  vst1q_s32(sum + 28, vaddq_s32(a7, c7));
        const float* bias = n.l1b.data() + b * L2;
        for (int o = 0; o < 32; ++o) x1[o] = screlu(static_cast<float>(sum[o]) / DEQ + bias[o]);
    } else
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
#elif defined(__AVX2__)
    if (SC_NPS_S1 && L2 == 32 && (hl & 31) == 0 && !n.l1w_sp.empty()) {
        // S1 sparse L1 on AVX2: list the nonzero 4-byte chunks of h8 (8 per compare, one LUT store,
        // no branch), then per chunk broadcast its 4 inputs and add them x all 32 outputs: maddubs
        // (u8 h8 <= 127 x s8 weight; a pair sum is at most 2*127*128 = 32512, no int16 saturation)
        // + madd(., 1) -> 8 int32 outputs per vector, 4 vectors. Two accumulator sets alternate
        // chunks. Zero chunks add exactly 0 and int32 sums are order-independent, so every output is
        // the dense kernel's integer; the epilogue is the same IEEE ops per lane.
        alignas(32) std::uint16_t nz[MAX_HL / 4 + 8];
        int nnz = 0;
        const __m256i zero = _mm256_setzero_si256();
        __m128i base = _mm_setzero_si128();
        const __m128i eight = _mm_set1_epi16(8);
        for (int i = 0; i < hl; i += 32) {
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h8 + i));
            const unsigned m = ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(h, zero)))) & 0xFFu;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(nz + nnz),
                             _mm_add_epi16(base, _mm_load_si128(reinterpret_cast<const __m128i*>(kNnzLut8[m].data()))));
            nnz += __builtin_popcount(m);
            base = _mm_add_epi16(base, eight);
        }
        const std::int8_t* W = n.l1w_sp.data() + static_cast<std::size_t>(b) * (hl / 4) * 128;
        const __m256i ones16 = _mm256_set1_epi16(1);
        auto in4 = [&](int k) { std::int32_t v; std::memcpy(&v, h8 + 4 * k, 4); return _mm256_set1_epi32(v); };
        auto wv = [](const std::int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        auto dp = [&](__m256i acc, __m256i in, const std::int8_t* w) {
            return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(in, wv(w)), ones16));
        };
        __m256i a0 = zero, a1 = zero, a2 = zero, a3 = zero, c0 = zero, c1 = zero, c2 = zero, c3 = zero;
        int j = 0;
        for (; j + 2 <= nnz; j += 2) {
            const int k0 = nz[j], k1 = nz[j + 1];
            const __m256i i0 = in4(k0), i1 = in4(k1);
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            const std::int8_t* w1 = W + static_cast<std::size_t>(k1) * 128;
            a0 = dp(a0, i0, w0);      c0 = dp(c0, i1, w1);
            a1 = dp(a1, i0, w0 + 32); c1 = dp(c1, i1, w1 + 32);
            a2 = dp(a2, i0, w0 + 64); c2 = dp(c2, i1, w1 + 64);
            a3 = dp(a3, i0, w0 + 96); c3 = dp(c3, i1, w1 + 96);
        }
        if (j < nnz) {
            const int k0 = nz[j];
            const __m256i i0 = in4(k0);
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            a0 = dp(a0, i0, w0);      a1 = dp(a1, i0, w0 + 32);
            a2 = dp(a2, i0, w0 + 64); a3 = dp(a3, i0, w0 + 96);
        }
        const __m256i s[4] = {_mm256_add_epi32(a0, c0), _mm256_add_epi32(a1, c1),
                              _mm256_add_epi32(a2, c2), _mm256_add_epi32(a3, c3)};
        const float* bias = n.l1b.data() + b * L2;
        for (int q = 0; q < 4; ++q) {   // the dense kernel's epilogue (A7), 8 lanes at a time
            __m256 v = _mm256_add_ps(_mm256_div_ps(_mm256_cvtepi32_ps(s[q]), _mm256_set1_ps(DEQ)),
                                     _mm256_loadu_ps(bias + 8 * q));
            v = _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    } else
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

Value evaluate(Context& cx, const Board& board) {
    return g_net.quant ? eval_quant_t<false>(cx, board) : eval_float_t<false>(cx, board);
}

Value evaluate_pass(Context& cx, const Board& board) {
    return g_net.quant ? eval_quant_t<true>(cx, board) : eval_float_t<true>(cx, board);
}

// ---- incremental accumulator hooks (called by search) -----------------------
void acc_reset(Context& cx, const Board& board) {
    if (!g_net.loaded) { cx.ply = -1; return; }
    if (cx.finny_gen != g_net_gen) {   // K1: cached sums belong to the previous net
        for (auto& p : cx.finny) for (auto& f : p) for (auto& e : f) e.valid = false;
        cx.finny_gen = g_net_gen;
    }
    if (cx.bb.empty()) cx.bb.resize(ACC_STACK);
    cx.ply = 0;
    if (g_net.quant) {
        if (cx.stack.empty()) cx.stack.resize(ACC_STACK);
        refresh_into(cx.stack[0], board);
    } else {  // float labeler: fixed-point incremental accumulator
        if (cx.stack_fx.empty()) cx.stack_fx.resize(ACC_STACK);
        refresh_into_fx(cx.stack_fx[0], board);
        if constexpr (SC_FX_LAZY) {
            if (cx.fx_ok.empty()) { cx.fx_ok.resize(ACC_STACK); cx.fx_pend.resize(ACC_STACK); }
            cx.fx_ok[0] = 1;
        }
    }
    cx.bb[0] = BB(board);
}

// Incremental delta, templated on accumulator/weight type so the intricate delta
// logic lives once. Quant instantiates <Acc,int8> (byte-identical to before, guarded
// by verify_acc); the float labeler instantiates <AccFx,int32> (fixed-point).
template <class AccT, class WT>
void acc_make_t(Context& cx, const Board& before, Move m, std::vector<AccT>& g_stk,
                const WT* l0, void (*refresh_fn)(AccT&, const Board&)) {
    constexpr bool FX = (sizeof(WT) != 1);  // int32 fixed-point float vs int8 quant
    constexpr bool LZ = FX && SC_FX_LAZY;   // FXZ: the float path defers its column passes
    cx.threats_made = 0;                     // item #9: per-move, so reset on every entry
    if (cx.ply < 0 || cx.ply + 1 >= ACC_STACK) { cx.ply = -1; return; }
    const int hl = g_net.hl;
    AccT& cur = g_stk[cx.ply];
    AccT& nxt = g_stk[cx.ply + 1];
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
#ifdef SC_LZSTAT
    if (m.typeOf() == Move::CASTLING) ++g_lz_kind[0];
    else if (m.typeOf() == Move::PROMOTION) ++g_lz_kind[1];
    else if (m.typeOf() == Move::ENPASSANT) ++g_lz_kind[2];
    else if (mt0 == 5) ++g_lz_kind[special ? 3 : king_rebucket ? 4 : 5];
#endif
    if (special) {
        // G8: make the move on the caller's own board and unmake it after the refresh,
        // instead of copying the Board (which deep-copies its undo stack) per refresh.
        // Every caller passes its mutable search board; the refresh reads exactly the
        // state a copy would have had, and unmakeMove restores the board in full.
        Board& b = const_cast<Board&>(before);
        b.makeMove(m);
        refresh_fn(nxt, b);
        cx.bb[cx.ply + 1] = BB(b);  // keep the incremental BB chain in sync
        b.unmakeMove(m);
        if constexpr (LZ) cx.fx_ok[cx.ply + 1] = 1;
    } else {
        // Single pass (x86 campaign G4, +3.7% all phases): the base columns are deferred into
        // the threat add/sub lists and ONE src->dst fused pass per perspective reads cur and
        // writes nxt -- no memcpy, no separate base pass. Integer sums are order-independent
        // -> bit-identical. The probes (NOUPDATE/NOCOLS/NODIRECT) keep the two-pass form.
        // The float labeler takes the same single pass (int32 sums -> bit-identical; self-play
        // campaign FX2). The NODIRECT probe is int16-only, so FX ignores g_direct here.
        const bool single_pass = (FX || g_direct) && !g_no_update && !g_no_cols;
        if constexpr (LZ) {
            cx.fx_ok[cx.ply + 1] = 1;               // computed unless the column pass below is deferred
            if (!single_pass) fx_ensure(cx, cx.ply);   // the probes read cur up front
        }
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
        // (cx.bb[cx.ply]), so bb0 is a reference -- no BB rebuild from board.at().
        // Form the after-move child directly in the next slot.
        const BB& bb0 = cx.bb[cx.ply];
        BB& bb1 = cx.bb[cx.ply + 1];
        if (g_verify) {  // the stack BB must equal a fresh recompute of `before`
            BB chk(before);
            if (bb0.occ != chk.occ || bb0.wk != chk.wk || bb0.bk != chk.bk
                || std::memcmp(bb0.by, chk.by, sizeof(chk.by)) != 0
                || std::memcmp(bb0.piece_on, chk.piece_on, sizeof(chk.piece_on)) != 0)
                std::fprintf(stderr, "[nnue] BB MISMATCH ply=%d\n", cx.ply);
        }
        bb1 = bb0;  // copy parent state, then apply the move in place
        const int sf_moved = (mcolor << 3) | (mt0 + 1);
        bb1.by[sf_moved] ^= (1ull << from) | (1ull << to);
        bb1.piece_on[from] = -1;
        if (cap) bb1.by[((cpi / 6) << 3) | ((cpi % 6) + 1)] &= ~(1ull << to);
        bb1.piece_on[to] = static_cast<std::int8_t>(sf_moved);
        bb1.occ = (bb0.occ & ~(1ull << from)) | (1ull << to);
        if (mt0 == 5) { if (mcolor == 0) bb1.wk = to; else bb1.bk = to; }  // king moved
        if (g_no_update) { cx.ply++; return; }  // cost probe: skip the delta (BB kept)
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
        const WT* (&addp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.addp);
        const WT* (&subp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.subp);
        int na[2] = {0, 0}, ns[2] = {0, 0};
        // P1: every line of a column, issued when the column is known.
        [[maybe_unused]] auto prefetch_col = [&](const WT* col) {
            const char* c = reinterpret_cast<const char*>(col);
            const std::size_t bytes = static_cast<std::size_t>(hl) * sizeof(WT);
            for (std::size_t off = 0; off < bytes; off += kPfStride) __builtin_prefetch(c + off, 0, SC_NPS_PF_LOC);
        };
        if (single_pass)   // G4: the base columns ride in the same lists as the threat/pp columns
            for (int p = 0; p < 2; ++p) {
                addp[p][na[p]++] = base_add[p];
                subp[p][ns[p]++] = base_sub[p][0];
                if (cap) subp[p][ns[p]++] = base_sub[p][1];
                if constexpr (SC_NPS_P1) {
                    prefetch_col(base_add[p]); prefetch_col(base_sub[p][0]);
                    if (cap) prefetch_col(base_sub[p][1]);
                }
            }
        auto push = [&](int p, std::uint32_t idx, bool add) {
            const WT* col = l0 + (bd + idx) * hl;
            if constexpr (SC_NPS_P1) prefetch_col(col);
            if (add) addp[p][na[p]++] = col; else subp[p][ns[p]++] = col;
        };
        auto emit_threat = [&](int a_sf, int a_sq, int t_sq, int attacked, bool add) {
            // Item #9: count only threats the MOVER gains on a higher-valued victim. piece-type bits
            // (1=pawn..5=queen) are ordered by value, so the comparison needs no table; the colour
            // bit (8) selects the mover's own attackers, so a threat created AGAINST the mover does
            // not inflate the count. Only pairs that are threat features count (attacks on a king
            // and the other excluded pairs map past THREAT_DIMS).
            const std::uint32_t s = make_threat_index(a_sf, a_sq ^ wo, t_sq ^ wo, attacked);
            if (s < (std::uint32_t)THREAT_DIMS) {
                if (add && (a_sf & 8) == (sf_moved & 8) && (attacked & 7) > (a_sf & 7)) ++cx.threats_made;
                push(0, s, add);
            }
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

        if (g_prefetch && !SC_NPS_P1)
            for (int p = 0; p < 2; ++p) {
                for (int k = 0; k < na[p]; ++k)
                    for (int t = 0; t < SC_NPS_PF_N; ++t)
                        __builtin_prefetch(reinterpret_cast<const char*>(addp[p][k]) + t * kPfStride, 0, SC_NPS_PF_LOC);
                for (int k = 0; k < ns[p]; ++k)
                    for (int t = 0; t < SC_NPS_PF_N; ++t)
                        __builtin_prefetch(reinterpret_cast<const char*>(subp[p][k]) + t * kPfStride, 0, SC_NPS_PF_LOC);
            }
        if (!g_no_cols) {  // NOCOLS probe: keep enumeration, skip the column adds
            bool deferred = false;
            if constexpr (LZ) {   // FXZ: record the lists; a rebucket pass needs nxt now
                if (single_pass && !king_rebucket && na[0] + ns[0] <= kFxPendCap && na[1] + ns[1] <= kFxPendCap) {
                    Context::FxPend& e = cx.fx_pend[cx.ply + 1];
                    for (int p = 0; p < 2; ++p) {
                        e.na[p] = na[p];
                        e.ns[p] = ns[p];
                        std::memcpy(e.col[p], addp[p], static_cast<std::size_t>(na[p]) * sizeof(e.col[p][0]));
                        std::memcpy(e.col[p] + na[p], subp[p], static_cast<std::size_t>(ns[p]) * sizeof(e.col[p][0]));
                    }
                    cx.fx_ok[cx.ply + 1] = 0;
                    deferred = true;
                } else if (single_pass) {
                    fx_ensure(cx, cx.ply);
                }
            }
            if (deferred) {
                // FXZ: applied on the first read of this ply (fx_ensure)
            } else if (single_pass) {   // G4: cur -> nxt in one pass (base + threat + pp columns)
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
          bool rebucketed = false;
          // With few men the direct form (2 columns per man) is already cheaper than two cache
          // passes; the caches pay where the board is full.
          if constexpr (!FX && SC_NPS_K1) if (__builtin_popcountll(bb1.occ) >= SC_NPS_K1_MIN) {
            rebucketed = true;
            // Bring the old and the new bucket's cache to the current pieces, then swap them in.
            const int flip = (file_of(mcolor == 0 ? to : (to ^ 56)) > 3) ? 1 : 0;
            const int kb_old = king_bucket(mcolor == 0 ? from : (from ^ 56));
            const int kb_new = king_bucket(mcolor == 0 ? to : (to ^ 56));
            auto refresh_finny = [&](Context::Finny& e, int ksq) {
                const WT** ra = reinterpret_cast<const WT**>(cx.ra);
                const WT** rs = reinterpret_cast<const WT**>(cx.rs);
                int nra = 0, nrs = 0;
                if (!e.valid) {
                    std::memset(e.acc, 0, sizeof(e.acc));
                    std::memset(e.by, 0, sizeof(e.by));
                    e.valid = true;
                }
                for (int sf = 1; sf < 15; ++sf) {
                    if ((sf & 7) == 0 || (sf & 7) == 7) continue;
                    std::uint64_t gone = e.by[sf] & ~bb1.by[sf], added = bb1.by[sf] & ~e.by[sf];
                    const int c = sf >> 3, t0 = (sf & 7) - 1;
                    while (gone)  { const int sq = __builtin_ctzll(gone);  gone &= gone - 1;
                        rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, ksq) * hl; }
                    while (added) { const int sq = __builtin_ctzll(added); added &= added - 1;
                        ra[nra++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, ksq) * hl; }
                    e.by[sf] = bb1.by[sf];
                    if (nra > 32 || nrs > 32) {   // flush (the scratch holds 40)
                        fused(e.acc, ra, nra, rs, nrs);
                        nra = nrs = 0;
                    }
                }
                if (nra + nrs) fused(e.acc, ra, nra, rs, nrs);
            };
            Context::Finny& eo = cx.finny[mcolor][flip][kb_old];
            Context::Finny& en = cx.finny[mcolor][flip][kb_new];
            refresh_finny(eo, from);
            refresh_finny(en, to);
            std::int16_t* a = nxt.v[mcolor];
            for (int j = 0; j < hl; ++j)
                a[j] = static_cast<std::int16_t>(a[j] - eo.acc[j] + en.acc[j]);
          }
          if (!rebucketed) {
            const WT** ra = reinterpret_cast<const WT**>(cx.ra);
            const WT** rs = reinterpret_cast<const WT**>(cx.rs);
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
    }
    cx.ply++;
}

// Dispatch acc_make to the active accumulator (quant int16 / float fixed-point int32).
void acc_make(Context& cx, const Board& before, Move m) {
#ifdef SC_LZSTAT
    struct Mark { Context& c; ~Mark() { if (c.ply > 0) { c.lz_needed[c.ply] = 0; ++g_lz_upd; } } } mark{cx};
#endif
    if (g_net.quant)
        acc_make_t<Acc, std::int8_t>(cx, before, m, cx.stack, g_net.l0w_i8.data(), refresh_into);
    else
        acc_make_t<AccFx, std::int32_t>(cx, before, m, cx.stack_fx, g_net.l0w_fx_p, refresh_into_fx);
}

void acc_unmake(Context& cx) {
#ifdef SC_LZSTAT
    if (cx.ply > 0 && !cx.lz_needed[cx.ply]) ++g_lz_skip;
#endif
    if (cx.ply > 0) cx.ply--;
}

void acc_clear(Context& cx) { cx.ply = -1; }

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

int acc_threats_made(const Context& cx) noexcept { return g_net.quant ? cx.threats_made : 0; }

int acc_delta_l1(const Context& cx) noexcept {
    // Item #8 (novelty ledger): the magnitude of the feature-column change a move makes.
    // A quiet move swaps two columns per perspective, so this is ||W[to] - W[from]||_1 read
    // off the accumulator stack rather than recomputed -- the values are already resident,
    // which is what makes the signal cheap. Both perspectives are summed because the mover's
    // piece changes square in both halves' feature sets.
    if (cx.ply < 1 || !g_net.quant) return 0;
    const int hl = g_net.hl;
    const std::int16_t* c0 = cx.stack[cx.ply].v[0];
    const std::int16_t* p0 = cx.stack[cx.ply - 1].v[0];
    const std::int16_t* c1 = cx.stack[cx.ply].v[1];
    const std::int16_t* p1 = cx.stack[cx.ply - 1].v[1];
    const int n = hl < SC_ACCSIG_UNITS ? hl : SC_ACCSIG_UNITS;
    int sum = 0;
    for (int j = 0; j < n; ++j)
        sum += std::abs(int(c0[j]) - int(p0[j])) + std::abs(int(c1[j]) - int(p1[j]));
    return sum;
}

void acc_make_null(Context& cx) {
    if (cx.ply < 0 || cx.ply + 1 >= ACC_STACK) { cx.ply = -1; return; }
    const int hl = g_net.hl;
    if (g_net.quant) {
        std::memcpy(cx.stack[cx.ply + 1].v[0], cx.stack[cx.ply].v[0], hl * sizeof(std::int16_t));
        std::memcpy(cx.stack[cx.ply + 1].v[1], cx.stack[cx.ply].v[1], hl * sizeof(std::int16_t));
    } else if constexpr (SC_FX_LAZY) {   // FXZ: an empty column list = the copy, made on first read
        Context::FxPend& e = cx.fx_pend[cx.ply + 1];
        e.na[0] = e.na[1] = e.ns[0] = e.ns[1] = 0;
        cx.fx_ok[cx.ply + 1] = 0;
    } else {
        std::memcpy(cx.stack_fx[cx.ply + 1].v[0], cx.stack_fx[cx.ply].v[0], hl * sizeof(std::int32_t));
        std::memcpy(cx.stack_fx[cx.ply + 1].v[1], cx.stack_fx[cx.ply].v[1], hl * sizeof(std::int32_t));
    }
    cx.bb[cx.ply + 1] = cx.bb[cx.ply];  // piece placement unchanged by a null move
    cx.ply++;
#ifdef SC_LZSTAT
    cx.lz_needed[cx.ply] = 0; ++g_lz_null;
#endif
}
void acc_unmake_null(Context& cx) {
#ifdef SC_LZSTAT
    if (cx.ply > 0 && !cx.lz_needed[cx.ply]) ++g_lz_null_skip;
#endif
    if (cx.ply > 0) cx.ply--;
}

// Context-free forms: the calling thread's own context (UCI debug commands, tools).
Value evaluate(const Board& board) { return evaluate(tl_context(), board); }
Value evaluate_pass(const Board& board) { return evaluate_pass(tl_context(), board); }
void acc_reset(const Board& board) { acc_reset(tl_context(), board); }
void acc_make(const Board& before, Move m) { acc_make(tl_context(), before, m); }
void acc_unmake() { acc_unmake(tl_context()); }
void acc_clear() { acc_clear(tl_context()); }
void acc_make_null() { acc_make_null(tl_context()); }
void acc_unmake_null() { acc_unmake_null(tl_context()); }
int acc_threats_made() noexcept { return acc_threats_made(tl_context()); }
int acc_delta_l1() noexcept { return acc_delta_l1(tl_context()); }

}  // namespace engine::nnue
