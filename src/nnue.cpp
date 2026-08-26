// -----------------------------------------------------------------------------
// nnue.cpp — EXPERIMENTAL Stockfish-style threats inference (engine FORK).
//
// This is the src-threats/ fork. The stock engine (src/nnue.cpp) is untouched;
// revert by building from src/ instead of src-threats/.
//
// Architecture (matches train_bullet/src/train_threats.rs + threat_inputs.rs):
//   inputs = [ king-bucketed mirrored HalfKA-768 base (768*buckets)
//            | Stockfish FullThreats (59808, exact)
//            | Stockfish PP_3Wide    (4560,  exact) ]
//   FT 512, crelu + pairwise-multiply; 8 material output buckets; body 512->L2->32->1.
//
// The FullThreats/PP_3Wide index math is a faithful port of Stockfish's
// src/nnue/features/{full_threats,pp_3wide}.cpp, and is byte-identical to the
// Rust trainer's threat_inputs.rs (validated by train_bullet's check_threats).
// Correctness-first: a single float full-recompute per eval (no incremental
// accumulator, no quantisation yet).
//
// Loads a self-describing "SCN4" float net (train_threats exporter):
//   "SCN4", u32{version=4, hl, input_buckets, l2, out_buckets},
//   then u32 len + f32[len] for each of l0w l0b l1w l1b l2w l2b l3w l3b.
//   total_inputs is derived as 768*input_buckets + 59808 + 4560.
//   (SCN4 float -> SCN5 int8 playing net via tools/quantize_threats.py.)
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
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace engine::nnue {

namespace {

// ---- Stockfish feature-set constants (verbatim) -----------------------------
constexpr int THREAT_DIMS = 59808;   // FullThreats::Dimensions
constexpr int PP_PAWN_IDS = 2 * 48;  // COLOR_NB * 48
constexpr int PP_DIMS = PP_PAWN_IDS * (PP_PAWN_IDS - 1) / 2;  // 4560
constexpr int PP_INDEX_BASE = THREAT_DIMS;

// SF piece encoding: color bit = 8, type = piece & 7 in 1..6.
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
std::uint64_t pseudo_attacks_type(int pt, int sq) {  // SF type 2..6
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

// ---- precomputed Stockfish LUTs (built once at load) ------------------------
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
// Quantisation scales (must match tools/quantize_threats.py and the trainer's
// save scheme): feature transformer x QA (int8 weights), L1 x QB (int8). The
// int16 accumulator holds sums of int8 weights (activation scale 127).
constexpr int QA = 127;
constexpr int QB = 64;

// Stack-buffer caps for the quant path (no per-eval heap allocation).
constexpr int MAX_HL = 1024;
constexpr int MAX_L2 = 64;
// Fixed-point scale for the float labeler's incremental accumulator. Integer add is
// associative, so incremental == from-scratch bit-exactly; S is large enough that the
// float eval it feeds is byte-identical to the from-scratch float path, and small
// enough that the int32 accumulator never overflows (<=~250 active feats * 2^22 < 2^31).
constexpr int    FX_SHIFT = 22;
constexpr double FX_S     = double(1u << FX_SHIFT);

struct Net {
    bool loaded = false;
    bool quant = false;  // false = SCN4 float, true = SCN5 quantised (int8 FT)
    int hl = 0, input_buckets = 0, l2 = 0, out_buckets = 0;
    std::size_t base_dims = 0, total_inputs = 0;
    std::vector<float> l0w, l0b, l1w, l1b, l2w, l2b, l3w, l3b;
    // quant path: int8 feature transformer + int8 L1 (body stays float);
    // int16 accumulator + int16 FT bias.
    std::vector<std::int8_t> l0w_i8;
    std::vector<std::int16_t> l0b_i;
    // float labeler path: fixed-point FT weights/bias (lround(w*FX_S)) for the
    // incremental int32 accumulator. Built at load for float (SCN4) nets.
    std::vector<std::int32_t> l0w_fx, l0b_fx;
    std::vector<std::int8_t> l1w_i;
    // L1 weights transposed to [bucket*L2 + o][HL] (HL contiguous) for the sdot
    // dot-product path; built from l1w_i on load.
    std::vector<std::int8_t> l1w_dot;
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

bool read_arr(std::ifstream& f, std::vector<float>& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n) * 4);
    return static_cast<bool>(f);
}
bool read_i16(std::ifstream& f, std::vector<std::int16_t>& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n) * 2);
    return static_cast<bool>(f);
}
bool read_i8(std::ifstream& f, std::vector<std::int8_t>& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

inline float crelu(float x) { return std::clamp(x, 0.0f, 1.0f); }
inline float screlu(float x) {
    const float c = std::clamp(x, 0.0f, 1.0f);
    return c * c;
}

}  // namespace

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

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
inline void acc_sub(std::int16_t* a, const std::int8_t* w, int hl) {
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

// Full refresh (base + threats + pp) for a position, both perspectives. Uses
// gather routed to the absolute WHITE/BLACK-perspective accumulators.
void refresh_into(Acc& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int8_t* l0 = n.l0w_i8.data();
    for (int p = 0; p < 2; ++p)
        for (int j = 0; j < hl; ++j) a.v[p][j] = n.l0b_i[j];
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    gather(
        board, n.base_dims,
        [&](std::size_t f) { acc_add(a.v[stmp], l0 + f * hl, hl); },
        [&](std::size_t f) { acc_add(a.v[1 - stmp], l0 + f * hl, hl); });
}

// Fixed-point (int32) full refresh for the float labeler path: seed from l0b_fx, then
// gather adds l0w_fx rows. Mirrors refresh_into; the scalar int32 add auto-vectorises.
void refresh_into_fx(AccFx& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int32_t* l0 = n.l0w_fx.data();
    for (int p = 0; p < 2; ++p)
        for (int j = 0; j < hl; ++j) a.v[p][j] = n.l0b_fx[j];
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    gather(
        board, n.base_dims,
        [&](std::size_t f) { const std::int32_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a.v[stmp][j] += w[j]; },
        [&](std::size_t f) { const std::int32_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a.v[1 - stmp][j] += w[j]; });
}

// ---- incremental THREAT delta helpers ---------------------------------------
// Absolute board state in SF piece indexing (sf = color<<3 | (type+1)); by[sf],
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
#else
    for (int k = 0; k < na; ++k) acc_add(acc, add[k], hl);
    for (int k = 0; k < ns; ++k) acc_sub(acc, sub[k], hl);
#endif
}

// Fixed-point fused apply for the float labeler: int32 accumulator, int32 weight
// columns, 8 lanes/iter (two int32x4). Integer add is associative -> bit-exact.
inline void fused_apply_fx(std::int32_t* acc, const std::int32_t* const* add, int na,
                           const std::int32_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    // §L2: two independent accumulators per 4-lane group break the serial add-chain
    // (better NEON ILP). int32 add is associative -> any grouping is bit-identical.
    for (int c = 0; c < hl; c += 8) {
        int32x4_t a0 = vld1q_s32(acc + c),  a1 = vld1q_s32(acc + c + 4);
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
        vst1q_s32(acc + c,     vaddq_s32(a0, b0));
        vst1q_s32(acc + c + 4, vaddq_s32(a1, b1));
    }
#else
    for (int k = 0; k < na; ++k) { const std::int32_t* w = add[k]; for (int j = 0; j < hl; ++j) acc[j] += w[j]; }
    for (int k = 0; k < ns; ++k) { const std::int32_t* w = sub[k]; for (int j = 0; j < hl; ++j) acc[j] -= w[j]; }
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
#if defined(__ARM_NEON)
    const float* w2base = l2w + static_cast<std::size_t>(b) * 32;  // row i at + i*ostride + o
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
            const float* w = w2base + static_cast<std::size_t>(i) * ostride;
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
            acc = vfmaq_n_f32(acc, vld1q_f32(l2w + static_cast<std::size_t>(i) * ostride + b * 32 + o), x1[i]);
        float tmp[4];
        vst1q_f32(tmp, acc);
        for (int k = 0; k < 4; ++k) x2[o + k] = screlu(tmp[k]);
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
    for (int i = 0; i < 32; ++i)
        y += x2[i] * n.l3w[static_cast<std::size_t>(i) * OB + b];
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

Value eval_float(const Board& board) {
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
#if defined(__ARM_NEON)
    // §2.3: elementwise crelu(a)*crelu(b), 4 lanes/iter. clamp = vmin(vmax(x,0),1),
    // bit-identical to std::clamp for finite accumulator values. half % 4 == 0.
    const float32x4_t vz = vdupq_n_f32(0.0f), vo = vdupq_n_f32(1.0f);
    for (int i = 0; i < half; i += 4) {
        const float32x4_t a = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i),        vz), vo);
        const float32x4_t b = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i + half), vz), vo);
        vst1q_f32(h + i, vmulq_f32(a, b));
        const float32x4_t c = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i),        vz), vo);
        const float32x4_t d = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i + half), vz), vo);
        vst1q_f32(h + half + i, vmulq_f32(c, d));
    }
#else
    for (int i = 0; i < half; ++i) {
        h[i]        = crelu(acc_stm[i]) * crelu(acc_stm[i + half]);
        h[half + i] = crelu(acc_ntm[i]) * crelu(acc_ntm[i + half]);
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
    return finish_body(n, x1, b);
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
Value eval_quant(const Board& board) {
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

    // Int8 pairwise activation (SF-style): crelu-clamp to [0,QA], multiply the
    // two halves, and scale down by QA so the result fits int8 [0,QA]. This is
    // the L1 input for the sdot dot-products (h8 = pairwise * QA).
    alignas(16) std::int8_t h8[MAX_HL];
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
        pairwise8(acc_stm + i, h8 + i);
        pairwise8(acc_ntm + i, h8 + half + i);
    }
#else
    auto cr = [](int x) -> int { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int i = 0; i < half; ++i) {
        h8[i]        = static_cast<std::int8_t>((cr(acc_stm[i]) * cr(acc_stm[i + half])) / QA);
        h8[half + i] = static_cast<std::int8_t>((cr(acc_ntm[i]) * cr(acc_ntm[i + half])) / QA);
    }
#endif
    const int b = output_bucket(board, OB);
    const float DEQ = float(QA) * QB;  // h8 already carries the 1/QA scale
    const std::int8_t* wbucket = n.l1w_dot.data() + static_cast<std::size_t>(b) * L2 * hl;
    float x1[MAX_L2];
#if defined(__ARM_FEATURE_DOTPROD)
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
#else
    for (int o = 0; o < L2; ++o) {
        const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
        std::int32_t s = 0;
        for (int i = 0; i < hl; ++i) s += static_cast<int>(h8[i]) * w[i];
        x1[o] = screlu(static_cast<float>(s) / DEQ + n.l1b[b * L2 + o]);
    }
#endif
    return finish_body(n, x1, b);
}

}  // namespace

Value evaluate(const Board& board) {
    return g_net.quant ? eval_quant(board) : eval_float(board);
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
    if (g_ply < 0 || g_ply + 1 >= ACC_STACK) { g_ply = -1; return; }
    const int hl = g_net.hl;
    AccT& cur = g_stk[g_ply];
    AccT& nxt = g_stk[g_ply + 1];
    // fused-apply dispatch: int8/int16 (quant) vs int32/int32 (fixed-point float).
    auto fused = [&](auto* acc, const WT* const* add, int na, const WT* const* sub, int ns) {
        if constexpr (FX) fused_apply_fx(acc, add, na, sub, ns, hl);
        else              fused_apply(acc, add, na, sub, ns, hl);
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
        Board tmp = before;
        tmp.makeMove(m);
        refresh_fn(nxt, tmp);
        g_bb[g_ply + 1] = BB(tmp);  // keep the incremental BB chain in sync
    } else {
        std::memcpy(nxt.v[0], cur.v[0], hl * sizeof(nxt.v[0][0]));
        std::memcpy(nxt.v[1], cur.v[1], hl * sizeof(nxt.v[0][0]));
        const int kabs[2] = {before.kingSq(Color::WHITE).index(), before.kingSq(Color::BLACK).index()};
        const chess::Piece capp = before.at(m.to());
        const bool cap = (capp != chess::Piece::NONE);
        const int cpi = cap ? static_cast<int>(capp.internal()) : 0;
        // ---- base delta (piece-square), one fused accumulator pass per persp ----
        for (int p = 0; p < 2; ++p) {
            const WT* ba[1];
            const WT* bs[2];
            ba[0] = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, to, kabs[p])) * hl;
            bs[0] = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, from, kabs[p])) * hl;
            int nbs = 1;
            if (cap) bs[nbs++] = l0 + static_cast<std::size_t>(base_feat(p, cpi / 6, cpi % 6, to, kabs[p])) * hl;
            fused(nxt.v[p], ba, 1, bs, nbs);
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
        auto push = [&](int p, std::uint32_t idx, bool add) {
            const WT* col = l0 + (bd + idx) * hl;
            if (add) addp[p][na[p]++] = col; else subp[p][ns[p]++] = col;
        };
        auto emit_threat = [&](int a_sf, int a_sq, int t_sq, int attacked, bool add) {
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
            fused(nxt.v[0], addp[0], na[0], subp[0], ns[0]);
            fused(nxt.v[1], addp[1], na[1], subp[1], ns[1]);
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
