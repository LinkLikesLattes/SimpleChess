#pragma once
// -----------------------------------------------------------------------------
// nnue_internal.hpp -- what the NNUE translation units share: the feature-set constants and threat
// index, the build switches, the Net layout, the feature enumerator, and the globals defined in
// nnue_load.cpp (g_net, g_tab, g_net_gen, the probes). nnue_load.cpp parses a net file into g_net and
// lays its weights out; nnue.cpp runs the incremental accumulator and both inference paths.
// Internal to the engine's NNUE code: nothing outside src/nnue*.cpp includes it.
// -----------------------------------------------------------------------------
#include "nnue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>   // x86 AVX2 kernels (Haswell+; the CCRL ship target)
#endif

namespace engine::nnue {


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
inline std::uint64_t ray_attacks(int sq, std::uint64_t occ, const int (*deltas)[2], int nd) {
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
// NPS 3.6 item K1: king-bucket crossings from cached piece-square sums. When the moving king
// changes bucket (same mirror side), its own perspective needs every piece's base column under the
// new bucket instead of the old one. Each (perspective, mirror, bucket) keeps the int16 sum of the
// base columns of the pieces it last saw plus those pieces' bitboards; bringing a cache up to date
// costs only the pieces that changed since. Then acc -= old bucket's sum, += new bucket's sum.
// int16 sums wrap identically however they are grouped -> bit-identical.
#ifndef SC_NPS_K1_MIN   // men on the board from which K1 is used: 16 on both machines (i7-9700 and, re-tuned
#define SC_NPS_K1_MIN 16  // on net 10-15, the M1 Max: +0.43% over 20, 12 below 16)
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
// FXZ: lazy float accumulator. acc_make records a ply's add/sub column lists instead of applying
// them, and the first read of that ply (eval, a child that must apply its own delta, a null move)
// applies them from the newest computed ancestor, one ply at a time -- the same src->dst int32 pass,
// so the accumulator holds the same integers. A ply popped unread costs no column traffic (~18% of
// float updates are never read; SC_LZSTAT on 12 depth-9 games).
#ifndef SC_FX_LAZY
#define SC_FX_LAZY 1
#endif
// x86 FXP: the float eval's int32->float fill, pairwise product and the L1's nonzero masks in one
// pass straight from the accumulator (the same per-element ops; no 4 KB float copy written and read
// back), and an eval frame under one page (no Windows stack probe per eval).
#if defined(__AVX2__) && SC_FX_AVX2
#define SC_FXP 1
#else
#define SC_FXP 0
#endif
// NEON FXN: the float eval in one pass from the accumulator, with no float copies, h array or nonzero
// list. h[i] is nonzero exactly when both int32 inputs are positive, so most lanes are settled by an
// integer compare; the rest get the same float ops and the same ascending-i fmla chain.
#ifndef SC_FXN
#if defined(__ARM_NEON)
#define SC_FXN 1
#else
#define SC_FXN 0
#endif
#endif
// x86 lazy int8 updates: acc_make records the move; the board, the column lists and the pass wait for the first read
// of that ply (eval, or a descendant's read), so a ply popped unread and an eval-cache hit cost no column traffic. A
// refresh (castling, promotion, en passant, a king across the mirror) stays eager: it needs the board.
#ifndef SC_Q_LAZY        // i7-9700: +15.9% at fixed depth, +16 to +28% in games; the Mac keeps the eager form
#if defined(__x86_64__) || defined(_M_X64)
#define SC_Q_LAZY 1
#else
#define SC_Q_LAZY 0
#endif
#endif
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
constexpr auto kKnightAtt = [] { std::array<std::uint64_t, 64> t{}; for (int s = 0; s < 64; ++s) t[s] = knight_attacks_loop(s); return t; }();
constexpr auto kKingAtt   = [] { std::array<std::uint64_t, 64> t{}; for (int s = 0; s < 64; ++s) t[s] = king_attacks_loop(s); return t; }();
constexpr auto kPawnAtt   = [] {
    std::array<std::array<std::uint64_t, 64>, 2> t{};
    for (int c = 0; c < 2; ++c) for (int s = 0; s < 64; ++s) t[c][s] = pawn_attacks_loop(c, s);
    return t;
}();
inline std::uint64_t knight_attacks(int sq) { return kKnightAtt[sq]; }
inline std::uint64_t king_attacks(int sq) { return kKingAtt[sq]; }
inline std::uint64_t pawn_attacks(int color, int sq) { return kPawnAtt[color][sq]; }
inline std::uint64_t pseudo_attacks_type(int pt, int sq) {  // piece type 2..6 = N,B,R,Q,K
    switch (pt) {
        case 2: return knight_attacks(sq);
        case 3: return ray_attacks(sq, 0, BISHOP_DIRS, 4);
        case 4: return ray_attacks(sq, 0, ROOK_DIRS, 4);
        case 5: return ray_attacks(sq, 0, BISHOP_DIRS, 4) | ray_attacks(sq, 0, ROOK_DIRS, 4);
        case 6: return king_attacks(sq);
        default: return 0;
    }
}
inline std::uint64_t attacks_bb(int pt, int sq, std::uint64_t occ) {
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
inline std::uint64_t pseudo_for_index(int sf_piece, int sq) {
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
extern Tables g_tab;   // nnue_load.cpp
void build_tables();   // nnue_load.cpp: once, on the first load


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

#if defined(_WIN32)
// D2 (nnue_load.cpp): allocations >= kLargePageThresh are backed by 2 MB large pages when the
// lock-pages privilege allows; sc_nnue_large_free returns false for a pointer it did not allocate.
inline constexpr std::size_t kLargePageThresh = 2u * 1024u * 1024u;
void* sc_nnue_large_alloc(std::size_t bytes);
bool sc_nnue_large_free(void* p);
#endif
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
    // S1: L1 weights by input chunk, [bucket][hl/4 chunks][L2 outputs][4 inputs], for the sparse
    // L1 (one 128-byte block per nonzero chunk at L2 = 32). Built after the neuron permutation.
    WVec<std::int8_t> l1w_sp;
};
extern Net g_net;      // nnue_load.cpp
// FXS (nnue_load.cpp): the fixed-point FT columns of a float net in a mapping shared by every engine
// process; null on any failure (the caller then builds its private copy).
std::shared_ptr<const std::int32_t> fx_share_map(const std::vector<float>& l0w, std::size_t ti, std::size_t hl);

extern unsigned g_net_gen;   // bumped on every successful load (K1: Finny caches hold column sums)
// Profiling / verification probes, read from SCNNUE_* environment variables at net load. Compiled in
// only with -DSC_NNUE_PROBES=1 (speed and correctness campaigns: cost decomposition, A/B of a path,
// tools/speed/verify_acc.py); in any other build each probe is a constant and its branches vanish.
#ifndef SC_NNUE_PROBES
#define SC_NNUE_PROBES 0
#endif
#define SC_NNUE_PROBE_LIST(X) \
    X(bool, g_base_only, false)  /* SCNNUE_BASEONLY: skip threats + pp (cost probe) */ \
    X(bool, g_verify, false)  /* SCNNUE_VERIFY: cross-check the incremental accumulator vs a recompute */ \
    X(int, g_squeeze, 0)  /* SCNNUE_SQUEEZE=N: fold threat/pp columns into N (cache probe) */ \
    X(bool, g_no_update, false)  /* SCNNUE_NOUPDATE: skip the threat/pp delta in acc_make (cost probe) */ \
    X(bool, g_no_finny, false)  /* SCNNUE_NOFINNY: force king moves to a full refresh (A/B the finny path) */ \
    X(bool, g_prefetch, true)  /* SCNNUE_NOPREFETCH: no FT-column prefetch in acc_make (A/B) */ \
    X(bool, g_no_cols, false)  /* SCNNUE_NOCOLS: skip the column ops (isolate enumeration vs column cost) */ \
    X(bool, g_l1dense, false)  /* SCNNUE_L1DENSE: force the dense float L1 matvec (A/B vs NNZ sparse) */
#if SC_NNUE_PROBES
#define SC_PROBE_DECL(type, name, init) extern type name;
#else
#define SC_PROBE_DECL(type, name, init) inline constexpr type name = init;
#endif
SC_NNUE_PROBE_LIST(SC_PROBE_DECL)
#undef SC_PROBE_DECL
// The SQUEEZE probe's fold of a threat / pp index (the identity in a build without probes).
inline std::uint32_t squeezed(std::uint32_t idx) {
#if SC_NNUE_PROBES
    return g_squeeze ? idx % static_cast<std::uint32_t>(g_squeeze) : idx;
#else
    return idx;
#endif
}
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
// How many lines per column the P1=0 form prefetches after the enumeration (one cache line apart).
#ifndef SC_NPS_PF_N   // x86: none at all (i7-9700: +0.40% over the first line, which beat every line)
#if defined(__x86_64__) || defined(_M_X64)
#define SC_NPS_PF_N 0
#else
#define SC_NPS_PF_N 1
#endif
#endif
constexpr std::size_t kPfStride = kCacheLine;
// Locality hint of those column prefetches (3 = all levels / prefetcht0, 1 = L2 and out / prefetcht2).
#ifndef SC_EVC_BITS     // the eval cache: 2^N entries of 8 B, per worker (N >= 16). i7-9700: 23 beat 21 and 24 at long searches
#if defined(__x86_64__) || defined(_M_X64)
#define SC_EVC_BITS 23
#else
#define SC_EVC_BITS 21
#endif
#endif
#ifndef SC_EVC_LARGE    // Windows: the eval cache in 2 MB pages when the lock-pages privilege allows
#define SC_EVC_LARGE 1
#endif
#ifndef SC_FX_K1_MIN   // men on the board from which the float path uses its K1 caches (i7-9700: 8 beat 4, 12, 16 and 20)
#if defined(__x86_64__) || defined(_M_X64)
#define SC_FX_K1_MIN 8
#else
#define SC_FX_K1_MIN 16
#endif
#endif
#ifndef SC_FX_PF_X86      // x86 takes the column prefetch below on the float path too (i7-9700: +1.8 to +2.7%)
#if defined(__x86_64__) || defined(_M_X64)
#define SC_FX_PF_X86 1
#else
#define SC_FX_PF_X86 0
#endif
#endif
#ifndef SC_FX_SCRATCH     // the pending float lists in one scratch instead of per ply (i7-9700: about +0.6%)
#if defined(__x86_64__) || defined(_M_X64)
#define SC_FX_SCRATCH 1
#else
#define SC_FX_SCRATCH 0
#endif
#endif
#ifndef SC_FX_PF_STRIDE   // the float path's column prefetch: bytes between prefetches, 0 = none
#define SC_FX_PF_STRIDE 2048
#endif
#ifndef SC_NPS_PF_LOC
#define SC_NPS_PF_LOC 3
#endif


[[maybe_unused]] inline float crelu(float x) { return std::clamp(x, 0.0f, 1.0f); }
inline float screlu(float x) {
    const float c = std::clamp(x, 0.0f, 1.0f);
    return c * c;
}

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
        if (s < static_cast<std::uint32_t>(THREAT_DIMS)) add_stm(base + squeezed(s));
        const std::uint32_t nn =
            make_threat_index(attacker_abs ^ 8, from ^ ntm_orient, to ^ ntm_orient, attacked_abs ^ 8);
        if (nn < static_cast<std::uint32_t>(THREAT_DIMS)) add_ntm(base + squeezed(nn));
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
        add_stm(base + squeezed(pp_one(stm_orient, 0)));
        add_ntm(base + squeezed(pp_one(ntm_orient, 1)));
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

// S1 (nnue_load.cpp): reorder the hidden neurons of a quantised net so the usually-zero ones sit together.
void permute_neurons(Net& n);

}  // namespace engine::nnue
