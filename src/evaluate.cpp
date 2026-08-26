// -----------------------------------------------------------------------------
// evaluate.cpp
//
// Tapered HCE: material + piece-square tables (v0.1), plus small safe-mobility
// and square-control terms (v0.2), blended between a middlegame and endgame
// score by the amount of material on the board. Future HCE terms (pawn
// structure, king safety, ...) slot in alongside.
//
// All tables are written from White's point of view with A1 = index 0. Black's
// squares are mirrored vertically (sq ^ 56) and its contributions are subtracted
// from White's, giving a white-relative score that we flip for Black to move.
// -----------------------------------------------------------------------------

#include "evaluate.hpp"

#include <algorithm>
#include <array>
#include <bit>

#include "nnue.hpp"

namespace engine::eval {

namespace {

// Piece values in centipawns, indexed [PieceType] for the middlegame (MG) and
// endgame (EG) phases. Order matches PieceType: PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING.
constexpr std::array<Value, 6> kMaterialMG = {82, 337, 365, 477, 1025, 0};
constexpr std::array<Value, 6> kMaterialEG = {94, 281, 297, 512, 936, 0};

// Phase weights per piece type; summed over the board they give a 0..24 phase
// where 24 == full middlegame and 0 == bare king endgame.
constexpr std::array<int, 6> kPhaseInc = {0, 1, 1, 2, 4, 0};
constexpr int                kPhaseMax = 24;

// Simplified piece-square tables (centipawns). These are placeholders good
// enough to steer development toward the centre; tuning them is future work.
// clang-format off
constexpr std::array<Value, 64> kPstPawn = {
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr std::array<Value, 64> kPstKnight = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
};
constexpr std::array<Value, 64> kPstBishop = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
};
constexpr std::array<Value, 64> kPstRook = {
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
};
constexpr std::array<Value, 64> kPstQueen = {
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
};
constexpr std::array<Value, 64> kPstKingMG = {
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
};
constexpr std::array<Value, 64> kPstKingEG = {
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
};
// clang-format on

// Accumulate one side's material + PST into (mg, eg). White uses squares as-is;
// Black mirrors vertically. Returns nothing; adds into the referenced accumulators.
void add_side(const Board& board, Color color, int& mg, int& eg, int& phase) {
    const int sign = (color == Color::WHITE) ? 1 : -1;

    auto scan = [&](PieceType::underlying pt, const std::array<Value, 64>& pst_mg,
                    const std::array<Value, 64>& pst_eg) {
        Bitboard bb = board.pieces(PieceType(pt), color);
        while (bb) {
            const int sq  = bb.pop();
            const int idx = (color == Color::WHITE) ? sq : (sq ^ 56);
            const int p   = static_cast<int>(pt);
            mg += sign * (kMaterialMG[p] + pst_mg[idx]);
            eg += sign * (kMaterialEG[p] + pst_eg[idx]);
            phase += kPhaseInc[p];
        }
    };

    scan(PieceType::PAWN, kPstPawn, kPstPawn);
    scan(PieceType::KNIGHT, kPstKnight, kPstKnight);
    scan(PieceType::BISHOP, kPstBishop, kPstBishop);
    scan(PieceType::ROOK, kPstRook, kPstRook);
    scan(PieceType::QUEEN, kPstQueen, kPstQueen);
    scan(PieceType::KING, kPstKingMG, kPstKingEG);
}

// ---- Mobility & square control (v0.2) -----------------------------------------
//
// Two deliberately small terms, sized so they nudge rather than steer (a big
// control edge is worth a few tens of centipawns, never a piece):
//
//   * Mobility: per piece, the number of *safe* squares it can go to — squares
//     not occupied by our own men and not covered by an enemy pawn. The tables
//     are asymmetric on purpose: near-zero mobility is punished harder than
//     high mobility is rewarded, so cramped/trapped pieces hurt without the
//     engine chasing empty-board activity.
//
//   * Control: squares where we out-man the opponent. Exact per-square
//     attacker counting is too slow for an eval that runs at every node, so we
//     track "attacked at least once" and "attacked at least twice" bitboards
//     and call a square controlled if we attack it and they don't, or we
//     attack it twice and they only once. Central squares (c3-f6) count a
//     little extra.
//
// A lazy-eval gate skips both terms when material+PST alone is already a
// runaway (the fine-grained terms can't change the verdict there), which keeps
// the added cost negligible in lopsided subtrees.

// Safe-mobility tables, indexed by number of safe squares. {MG, EG} pairs.
// clang-format off
// The low-end penalties are deliberately moderate: home-square pieces start at
// 0-2 mobility, and harsh floors would turn every developing move into a huge
// eval swing. Truly trapped pieces still pay these floors *plus* the PST edge/
// corner penalties, which is plenty of signal.
constexpr int kMobKnightMG[9]  = {-18, -9, -3, 2, 6, 9, 12, 14, 15};
constexpr int kMobKnightEG[9]  = {-16, -8, -3, 2, 6, 9, 11, 13, 14};
constexpr int kMobBishopMG[14] = {-14, -7, -2, 2, 6, 10, 13, 15, 17, 18, 19, 20, 21, 21};
constexpr int kMobBishopEG[14] = {-12, -6, -2, 2, 6,  9, 12, 14, 16, 17, 18, 19, 20, 20};
constexpr int kMobRookMG[15]   = {-10, -6, -3, -1, 2, 4, 6, 8, 10, 11, 12, 13, 14, 14, 14};
constexpr int kMobRookEG[15]   = {-14, -8, -3,  0, 4, 8, 11, 14, 16, 18, 19, 20, 21, 22, 22};
constexpr int kMobQueenMG[28]  = {-8, -5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5, 6, 6,
                                   7,  7,  8,  8,  9, 9, 9, 10, 10, 10, 11, 11, 11, 11};
constexpr int kMobQueenEG[28]  = {-10, -7, -4, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                    9, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14};
// clang-format on

// Control weights, per square of (our controlled − their controlled).
constexpr int kControlMG = 2;
constexpr int kControlEG = 1;
constexpr int kControlCenterMG = 2;  // extra for controlled central squares

// The 16 central squares c3-f6.
constexpr Bitboard kCenterMask{0x00003C3C3C3C0000ULL};

// Skip mobility/control when |material+PST| already exceeds this.
constexpr int kLazyMargin = 600;

// ---- King attack / safety (v0.3) -----------------------------------------------
//
// Pieces are scored by pressure on the enemy king's zone (the king plus its
// ring of 8 squares): each piece whose attacks reach the zone adds weighted
// "attack units", the defender subtracts about half a unit per piece guarding
// its own zone (plus its pawn shield), and the net feeds a saturating table.
// Design constraints, in order:
//   * 0 attackers (or just 1 — a lone piece is a sortie, not an attack) = 0.
//   * Defenders only *soften* the attack (half weight), so 2-vs-2 still scores
//     above 0-vs-0 — pressure with contact beats no pressure.
//   * The table caps well under a minor piece (~90cp MG), and without the
//     queen participating the value is halved; this is an attacking-chances
//     nudge for equal positions, never a reason to sac material by itself.
//   * Middlegame-heavy: in the endgame kings want to be active, so only a
//     quarter of the value survives the taper on the EG side.

// Attack units contributed by each piece type reaching the enemy king zone.
// Sizing note: this term is inherently volatile — whether a slider reaches the
// zone flips with every blocking move — and the search's pruning heuristics
// (improving flag, RFP, aspiration) assume a fairly stable static eval.
// Empirically the tree-size overhead scaled with the term's amplitude (a ~90cp
// cap cost ~36% more nodes to equal depth; a graded, larger variant cost ~97%),
// so the table is kept deliberately small: enough to steer piece placement in
// ordering, cheap enough not to burn search depth.
constexpr int kAttackUnits[6] = {0, 2, 2, 3, 5, 0};  // P,N,B,R,Q,K
constexpr int kZoneContactCap = 0;                   // graded contact off (see above)

// Saturating attack table, indexed by net attack units. The amplitude is a
// build-time knob (percent of the base shape below) so weight experiments can
// produce variant binaries without touching source:
//   make EXTRA="-DSC_KATT_SCALE=50" ...   -> half-amplitude table (~45cp cap)
#ifndef SC_KATT_SCALE
#define SC_KATT_SCALE 67  // default ≈ 60cp cap, the tuned value
#endif
constexpr std::array<int, 22> kKingAttackTable = [] {
    constexpr int base[22] = {0,  1,  3,  6,  10, 14, 19, 24, 30, 36, 42,
                              48, 54, 60, 65, 70, 74, 78, 81, 84, 87, 90};
    std::array<int, 22> t{};
    for (int i = 0; i < 22; ++i) t[i] = base[i] * SC_KATT_SCALE / 100;
    return t;
}();

// Per-side tallies filled during the mobility sweep (attack maps + king pressure).
struct SideData {
    Bitboard atk;             // squares attacked at least once
    Bitboard atk2;            // squares attacked at least twice
    Bitboard atk_pawn{0};     // by pawns           (per-attacker maps for threats)
    Bitboard atk_minor{0};    // by knights/bishops
    Bitboard atk_rook{0};     // by rooks
    Bitboard atk_king{0};     // by the king
    int      att_units = 0;   // weighted units into the *enemy* king zone
    int      att_count = 0;   // number of non-pawn pieces attacking that zone
    bool     queen_att = false;  // queen among the attackers
    int      def_units = 0;   // pieces guarding our *own* king zone
};

// Computes mobility score and fills this side's attack maps and king-pressure
// tallies. `enemy_pawn_atk` must already be computed. Adds white-relative cp.
void add_mobility(const Board& board, Color color, Bitboard enemy_pawn_atk,
                  Bitboard pawn_atk, Bitboard pawn_atk2, Bitboard own_zone,
                  Bitboard enemy_zone, SideData& sd, int& mg, int& eg) {
    const int      sign     = (color == Color::WHITE) ? 1 : -1;
    const Bitboard occ      = board.occ();
    const Bitboard mob_area = ~board.us(color) & ~enemy_pawn_atk;

    sd.atk      = pawn_atk;
    sd.atk2     = pawn_atk2;
    sd.atk_pawn = pawn_atk;

    auto merge = [&](Bitboard a, PieceType::underlying pt) {
        sd.atk2 = sd.atk2 | (sd.atk & a);
        sd.atk  = sd.atk | a;
        // Per-attacker-class maps, consumed by the threats term.
        if (pt == PieceType::KNIGHT || pt == PieceType::BISHOP)
            sd.atk_minor = sd.atk_minor | a;
        else if (pt == PieceType::ROOK)
            sd.atk_rook = sd.atk_rook | a;
        if (const int z = (a & enemy_zone).count(); z > 0) {
            // Base weight plus a graded bonus for deeper contact: smooth, not
            // a cliff, as the piece works its way toward the king.
            sd.att_units += kAttackUnits[static_cast<int>(pt)] +
                            (z - 1 > kZoneContactCap ? kZoneContactCap : z - 1);
            ++sd.att_count;
            if (pt == PieceType::QUEEN) sd.queen_att = true;
        }
        if (!(a & own_zone).empty()) ++sd.def_units;
    };

    // King contributes to control but has no mobility or attack term.
    {
        const Bitboard a = chess::attacks::king(board.kingSq(color));
        sd.atk2          = sd.atk2 | (sd.atk & a);
        sd.atk           = sd.atk | a;
        sd.atk_king      = a;
    }

    Bitboard bb = board.pieces(PieceType::KNIGHT, color);
    while (bb) {
        const Bitboard a = chess::attacks::knight(Square(bb.pop()));
        merge(a, PieceType::KNIGHT);
        const int n = (a & mob_area).count();
        mg += sign * kMobKnightMG[n > 8 ? 8 : n];
        eg += sign * kMobKnightEG[n > 8 ? 8 : n];
    }
    bb = board.pieces(PieceType::BISHOP, color);
    while (bb) {
        const Bitboard a = chess::attacks::bishop(Square(bb.pop()), occ);
        merge(a, PieceType::BISHOP);
        const int n = (a & mob_area).count();
        mg += sign * kMobBishopMG[n > 13 ? 13 : n];
        eg += sign * kMobBishopEG[n > 13 ? 13 : n];
    }
    bb = board.pieces(PieceType::ROOK, color);
    while (bb) {
        const Bitboard a = chess::attacks::rook(Square(bb.pop()), occ);
        merge(a, PieceType::ROOK);
        const int n = (a & mob_area).count();
        mg += sign * kMobRookMG[n > 14 ? 14 : n];
        eg += sign * kMobRookEG[n > 14 ? 14 : n];
    }
    bb = board.pieces(PieceType::QUEEN, color);
    while (bb) {
        const Bitboard a = chess::attacks::queen(Square(bb.pop()), occ);
        merge(a, PieceType::QUEEN);
        const int n = (a & mob_area).count();
        mg += sign * kMobQueenMG[n > 27 ? 27 : n];
        eg += sign * kMobQueenEG[n > 27 ? 27 : n];
    }
}

// ---- Pawn structure (v0.4) ------------------------------------------------------
//
// Three classic terms plus an endgame push, all low-volatility (pawn structure
// changes slowly, so unlike the king-attack term this shouldn't disturb the
// search tree much):
//
//   * Doubled pawns  — penalty per extra pawn stacked on a file.
//   * Isolated pawns — penalty per pawn with no friendly pawn on an adjacent
//     file (it can never be defended by a pawn).
//   * Passed pawns   — no enemy pawn ahead on its own or adjacent files.
//     Bonus grows sharply with rank and is much larger in the endgame; halved
//     when the square in front is blockaded; +50% when our own attack map
//     covers the advance square ("supported by their pieces").
//   * Advancement    — small EG-only per-pawn rank bonus: with pieces off,
//     every pawn wants to march.
//
// All pawn terms share one amplitude knob for weight sweeps:
//   make EXTRA="-DSC_PAWN_SCALE=125" ...   -> 1.25x all pawn terms
#ifndef SC_PAWN_SCALE
#define SC_PAWN_SCALE 175  // sweep winner: 50..200 tested vs v0.3, peak at 175
#endif

constexpr int kDoubledMG  = 10, kDoubledEG  = 18;  // per extra pawn on a file
constexpr int kIsolatedMG = 12, kIsolatedEG = 10;  // per isolated pawn

// Passed-pawn bonus by relative rank (index = rank from own side, 1..6).
constexpr int kPassedMG[8] = {0, 2, 5, 10, 18, 32, 50, 0};
constexpr int kPassedEG[8] = {0, 8, 14, 24, 40, 65, 95, 0};

// EG-only advancement bonus for every pawn, by relative rank.
constexpr int kAdvanceEG[8] = {0, 0, 2, 4, 7, 12, 20, 0};

constexpr std::uint64_t kFileABits = 0x0101010101010101ULL;

constexpr std::array<std::uint64_t, 8> kFileMask = [] {
    std::array<std::uint64_t, 8> m{};
    for (int f = 0; f < 8; ++f) m[f] = kFileABits << f;
    return m;
}();

constexpr std::array<std::uint64_t, 8> kAdjFilesMask = [] {
    std::array<std::uint64_t, 8> m{};
    for (int f = 0; f < 8; ++f)
        m[f] = (f > 0 ? kFileABits << (f - 1) : 0) | (f < 7 ? kFileABits << (f + 1) : 0);
    return m;
}();

// kPassedMask[color][sq]: squares an enemy pawn must occupy to stop the pawn —
// everything strictly ahead of `sq` on its own and adjacent files.
constexpr std::array<std::array<std::uint64_t, 64>, 2> kPassedMask = [] {
    std::array<std::array<std::uint64_t, 64>, 2> m{};
    for (int sq = 0; sq < 64; ++sq) {
        const int f = sq & 7, r = sq >> 3;
        const std::uint64_t files = kFileABits << f |
                                    (f > 0 ? kFileABits << (f - 1) : 0) |
                                    (f < 7 ? kFileABits << (f + 1) : 0);
        std::uint64_t ahead_w = 0, ahead_b = 0;
        for (int rr = r + 1; rr < 8; ++rr) ahead_w |= 0xFFULL << (8 * rr);
        for (int rr = 0; rr < r; ++rr) ahead_b |= 0xFFULL << (8 * rr);
        m[0][sq] = files & ahead_w;  // WHITE
        m[1][sq] = files & ahead_b;  // BLACK
    }
    return m;
}();

// Doubled/isolated/passed/advancement for one side. `own_atk` is this side's
// full attack map from the mobility pass. Adds white-relative cp to mg/eg.
void add_pawn_structure(Color color, std::uint64_t own_pawns, std::uint64_t enemy_pawns,
                        std::uint64_t own_atk, std::uint64_t occ_bits, int& mg, int& eg) {
    const int sign  = (color == Color::WHITE) ? 1 : -1;
    const int white = (color == Color::WHITE);

    // Per-file terms: doubled and isolated.
    for (int f = 0; f < 8; ++f) {
        const int cnt = std::popcount(own_pawns & kFileMask[f]);
        if (cnt == 0) continue;
        if (cnt > 1) {
            mg -= sign * kDoubledMG * (cnt - 1) * SC_PAWN_SCALE / 100;
            eg -= sign * kDoubledEG * (cnt - 1) * SC_PAWN_SCALE / 100;
        }
        if ((own_pawns & kAdjFilesMask[f]) == 0) {
            mg -= sign * kIsolatedMG * cnt * SC_PAWN_SCALE / 100;
            eg -= sign * kIsolatedEG * cnt * SC_PAWN_SCALE / 100;
        }
    }

    // Per-pawn terms: passers and EG advancement.
    std::uint64_t bb = own_pawns;
    while (bb) {
        const int sq = std::countr_zero(bb);
        bb &= bb - 1;
        const int rel_rank = white ? (sq >> 3) : 7 - (sq >> 3);

        eg += sign * kAdvanceEG[rel_rank] * SC_PAWN_SCALE / 100;

        if ((kPassedMask[white ? 0 : 1][sq] & enemy_pawns) == 0) {
            int pmg = kPassedMG[rel_rank];
            int peg = kPassedEG[rel_rank];

            const int stop = white ? sq + 8 : sq - 8;  // rel_rank <= 6, always on-board
            if (occ_bits & (1ULL << stop)) {  // blockaded: much less dangerous
                pmg /= 2;
                peg /= 2;
            }
            if (own_atk & (1ULL << stop)) {  // advance square covered by our men
                pmg += pmg / 2;
                peg += peg / 2;
            }
            mg += sign * pmg * SC_PAWN_SCALE / 100;
            eg += sign * peg * SC_PAWN_SCALE / 100;
        }
    }
}

// Net king-attack value for one side (attacker's perspective, MG cp).
// `att` is the attacking side's tally, `def` the defender's; `att_pawn_units` /
// `def_shield` are the pawn contributions computed from aggregate bitboards.
int king_attack_value(const SideData& att, const SideData& def, int att_pawn_units,
                      int def_shield) {
    if (att.att_count < 2) return 0;  // a single piece is not an attack

    int idx = att.att_units + att_pawn_units - (def.def_units + def_shield) / 2;
    if (idx <= 0) return 0;
    if (idx > 21) idx = 21;

    int v = kKingAttackTable[idx];
    if (!att.queen_att) v /= 2;  // attacks rarely land without the queen
    return v;
}

// ---- King safety: the defender's side of the ledger (v0.8) ---------------------
//
// The king-attack term above scores the *attacker's* pressure; these terms
// score the defender's own arrangement, sharing the same zones and attack
// maps. Four factors, one amplitude knob:
//
//   * Shelter    — missing pawns in the 3-file x 2-rank box in front of the
//     king (plus extra for a fully open king file). Only charged while the
//     *enemy queen* is on the board — that's when exposure is fatal — and
//     MG-weighted, so it fades naturally toward the endgame. "Hide within
//     pawns" is this same term's bonus side: a full box costs nothing.
//   * Own-piece crowding — ring squares occupied by own NON-pawn pieces.
//     Pawns next to the king are shelter (good, see above); pieces are
//     clutter that steals escape squares.
//   * Enemy denial — ring squares the opponent attacks weigh heavier, and a
//     king with *zero* free ring squares (not own-occupied, not attacked)
//     pays a large extra penalty: that's back-rank / mating-net territory.
//   * Inactivity — with BOTH queens off, a king parked far from the centre is
//     doing nothing; penalty per distance step. Phase alone can't see this
//     (queens can trade early at high material), hence the explicit gate.
#ifndef SC_KSAFE
#define SC_KSAFE 200  // sweep winner: +89 Elo @ depth10/200g vs v0.7 (150/200/250 tested)
#endif

constexpr int kShelterMissMG    = 12;  // per missing shelter-box pawn (of 3)
constexpr int kShelterOpenMG    = 10;  // extra: no own pawn anywhere on king's file
constexpr int kOwnCrowdMG       = 3;   // per ring square blocked by own non-pawn
constexpr int kEnemyDenyMG      = 5, kEnemyDenyEG = 3;  // per ring square enemy-attacked
constexpr int kNoEscapeMG       = 20, kNoEscapeEG = 15; // zero free ring squares
constexpr int kInactiveKingStep = 5;   // per centre-distance step > 1, queens off

// Chebyshev distance to the nearest of d4/e4/d5/e5.
constexpr std::array<int, 64> kCenterDist = [] {
    std::array<int, 64> t{};
    constexpr int ctr[4] = {27, 28, 35, 36};  // d4 e4 d5 e5
    for (int sq = 0; sq < 64; ++sq) {
        int best = 8;
        for (int c : ctr) {
            int df = (sq & 7) - (c & 7);
            int dr = (sq >> 3) - (c >> 3);
            if (df < 0) df = -df;
            if (dr < 0) dr = -dr;
            best = std::min(best, std::max(df, dr));
        }
        t[sq] = best;
    }
    return t;
}();

// Defender-perspective king-safety penalty for one side. Returns {mg, eg}
// penalties (positive = bad for this king). `enemy_atk` is the opponent's full
// attack map; queen gates are passed in from the caller's piece counts.
struct KingSafety {
    int mg = 0, eg = 0;
};
KingSafety king_safety(const Board& board, Color color, Bitboard enemy_atk,
                       bool enemy_queen_alive, bool queens_off) {
    KingSafety ks;
    const Square   ksq     = board.kingSq(color);
    const int      sq      = ksq.index();
    const int      f       = sq & 7;
    const int      r       = sq >> 3;
    const bool     white   = (color == Color::WHITE);
    const Bitboard ring    = chess::attacks::king(ksq);
    const std::uint64_t own_occ   = board.us(color).getBits();
    const std::uint64_t own_pawns = board.pieces(PieceType::PAWN, color).getBits();

    // ---- Shelter (enemy queen alive; MG via the taper) ----
    if (enemy_queen_alive) {
        std::uint64_t front = 0;
        for (int step = 1; step <= 2; ++step) {
            const int rr = white ? r + step : r - step;
            if (0 <= rr && rr < 8) front |= 0xFFULL << (8 * rr);
        }
        const std::uint64_t box     = (kFileMask[f] | kAdjFilesMask[f]) & front;
        const int           have    = std::popcount(box & own_pawns);
        const int           missing = 3 - std::min(3, have);
        ks.mg += kShelterMissMG * missing;
        if ((own_pawns & kFileMask[f]) == 0) ks.mg += kShelterOpenMG;
    }

    // ---- Ring restriction ----
    const std::uint64_t ring_bits  = ring.getBits();
    const std::uint64_t own_block  = ring_bits & own_occ & ~own_pawns;
    const std::uint64_t enemy_deny = ring_bits & ~own_occ & enemy_atk.getBits();
    const std::uint64_t free_sqs   = ring_bits & ~own_occ & ~enemy_atk.getBits();

    ks.mg += kOwnCrowdMG * std::min(4, std::popcount(own_block));
    const int denied = std::min(4, std::popcount(enemy_deny));
    ks.mg += kEnemyDenyMG * denied;
    ks.eg += kEnemyDenyEG * denied;
    if (free_sqs == 0) {
        ks.mg += kNoEscapeMG;
        ks.eg += kNoEscapeEG;
    }

    // ---- Inactivity with queens off ----
    if (queens_off) {
        const int d = std::max(0, kCenterDist[sq] - 1);
        ks.mg += kInactiveKingStep * d;
        ks.eg += kInactiveKingStep * d;
    }

    return ks;
}

// ---- Threats (v1.0) -------------------------------------------------------------
//
// "Loose pieces drop off": pressure on *pieces*, as distinct from the control
// term's pressure on *squares*. Classical top-engine design, structured around
// a three-way classification of enemy men:
//
//   * strongly protected — defended by a pawn, or defended more times than we
//     attack it (their double-cover beats ours);
//   * defended — non-pawn enemies that are strongly protected (still worth a
//     nibble for a minor: attacking a defended rook with a knight forces it
//     to move or trade down);
//   * weak — attacked by us and not strongly protected: real targets.
//
// Scored, per side: minor attacks graded by victim value (on defended and
// weak alike), rook attacks on weak men, king attacks on weak men, a large
// bonus per *hanging* target (weak and either undefended entirely or a
// non-pawn we attack twice), safe pawns attacking pieces (a pawn never loses
// that trade), and pawns that can *push* next move to safely attack a piece.
// Values taper MG->EG per victim class. One amplitude knob for sweeps.
#ifndef SC_THREAT
#define SC_THREAT 50  // sweep winner: sequential 60ms/200g pass, 50 (+70 Elo) beat 75 (+49)
#endif

// Victim-indexed tables (P, N, B, R, Q; kings can't be "won" so index unused).
constexpr int kThreatMinorMG[6] = {5, 46, 62, 70, 62, 0};
constexpr int kThreatMinorEG[6] = {15, 19, 26, 56, 76, 0};
constexpr int kThreatRookMG[6]  = {2, 30, 30, 0, 40, 0};
constexpr int kThreatRookEG[6]  = {21, 33, 29, 18, 18, 0};
constexpr int kHangingMG   = 54, kHangingEG   = 17;  // per hanging target
constexpr int kThreatKingMG = 19, kThreatKingEG = 42;  // king attacks something weak
constexpr int kSafePawnMG  = 135, kSafePawnEG  = 44;  // pawn attacks a piece, safely
constexpr int kPawnPushMG  = 37, kPawnPushEG   = 18;  // ...or can do so next move

constexpr std::uint64_t kFileHBits = kFileABits << 7;
constexpr std::uint64_t kRank3Bits = 0x0000000000FF0000ULL;
constexpr std::uint64_t kRank6Bits = 0x0000FF0000000000ULL;

[[nodiscard]] constexpr std::uint64_t pawn_attacks_of(bool white, std::uint64_t pawns) {
    return white ? (((pawns & ~kFileHBits) << 9) | ((pawns & ~kFileABits) << 7))
                 : (((pawns & ~kFileABits) >> 9) | ((pawns & ~kFileHBits) >> 7));
}

// Threat score for `color` against the other side. Adds white-relative cp.
void add_threats(const Board& board, Color color, const SideData& us, const SideData& them,
                 std::uint64_t own_pawns, std::uint64_t enemy_occ, std::uint64_t enemy_pawns,
                 std::uint64_t occ, int& mg, int& eg) {
    const int  sign  = (color == Color::WHITE) ? 1 : -1;
    const bool white = (color == Color::WHITE);

    const std::uint64_t non_pawn_enemies = enemy_occ & ~enemy_pawns;
    const std::uint64_t strongly =
        them.atk_pawn.getBits() | (them.atk2.getBits() & ~us.atk2.getBits());
    const std::uint64_t defended = non_pawn_enemies & strongly;
    const std::uint64_t weak     = enemy_occ & ~strongly & us.atk.getBits();

    int tmg = 0, teg = 0;

    if (defended | weak) {
        // Minor pieces nibbling at anything valuable, defended or not.
        std::uint64_t b = (defended | weak) & us.atk_minor.getBits();
        while (b) {
            const int sq = std::countr_zero(b);
            b &= b - 1;
            const int pt = static_cast<int>(board.at<PieceType>(Square(sq)));
            tmg += kThreatMinorMG[pt];
            teg += kThreatMinorEG[pt];
        }
        // Rooks only profit from genuinely weak targets.
        b = weak & us.atk_rook.getBits();
        while (b) {
            const int sq = std::countr_zero(b);
            b &= b - 1;
            const int pt = static_cast<int>(board.at<PieceType>(Square(sq)));
            tmg += kThreatRookMG[pt];
            teg += kThreatRookEG[pt];
        }
        // The king joining in against something weak (mostly an endgame skill).
        if (weak & us.atk_king.getBits()) {
            tmg += kThreatKingMG;
            teg += kThreatKingEG;
        }
        // Hanging: weak and either totally undefended, or a non-pawn we hit twice.
        const std::uint64_t near_hanging =
            ~them.atk.getBits() | (non_pawn_enemies & us.atk2.getBits());
        const int hanging = std::popcount(weak & near_hanging);
        tmg += kHangingMG * hanging;
        teg += kHangingEG * hanging;
    }

    // Squares where our pawns stand (or land) safely: not attacked, or covered.
    const std::uint64_t safe = ~them.atk.getBits() | us.atk.getBits();

    // Safe pawns attacking real pieces.
    const int spa = std::popcount(pawn_attacks_of(white, own_pawns & safe) & non_pawn_enemies);
    tmg += kSafePawnMG * spa;
    teg += kSafePawnEG * spa;

    // Pawn pushes (single, plus double from the third rank) landing on safe
    // squares from which they would attack a piece next move.
    std::uint64_t push = white ? ((own_pawns << 8) & ~occ) : ((own_pawns >> 8) & ~occ);
    push |= white ? (((push & kRank3Bits) << 8) & ~occ) : (((push & kRank6Bits) >> 8) & ~occ);
    push &= ~them.atk_pawn.getBits() & safe;
    const int ppa = std::popcount(pawn_attacks_of(white, push) & non_pawn_enemies);
    tmg += kPawnPushMG * ppa;
    teg += kPawnPushEG * ppa;

    mg += sign * tmg * SC_THREAT / 100;
    eg += sign * teg * SC_THREAT / 100;
}

// ---- Progress scaling (v1.2) ----------------------------------------------------
//
// The halfmove clock counts plies since the last pawn move or capture — the
// only *irreversible* actions in chess, and so the game's own measure of
// "actual progress". When it climbs, moves are passing with no conversion of
// the position: an advantage that can't be turned into anything (the classic
// up-a-pawn opposite-coloured-bishop draw) or a lost position the opponent
// can't finish. We damp the score toward zero as the clock rises, so a static
// "winning"/"losing" number decays toward the draw it actually is.
//
// The beauty of doing this at the leaf: it propagates through the search for
// free. A line that only shuffles drives the clock up at its leaves, whose
// evals decay, so the search sees shuffling as drawish and prefers a move that
// makes real progress (a pawn break/capture *resets* the clock and restores
// the full eval). No effect below kProgOnset, so ordinary play — where
// something captures or pushes every few moves — is untouched; the threshold
// self-selects genuinely stuck positions. Mate scores are never produced by
// evaluate() (they come from the search's mate path), so this can't hide a
// mate. One amplitude knob for sweeps; SC_PROGRESS=0 leaves the score exact.
#ifndef SC_PROGRESS
#define SC_PROGRESS 100  // progress-decay amplitude; 0 disables (control builds)
#endif

constexpr int kProgOnset    = 16;  // plies without progress before decay starts
constexpr int kProgFloorPct = 20;  // score retained at the 50-move (hmc=100) edge

// ---- Space (v1.3, disabled) ------------------------------------------------------
//
// SPACE — room to manoeuvre behind your own pawns. Counted as safe squares in
// the four centre files on our own half: not occupied by our pawns, not
// attacked by enemy pawns, with squares tucked behind our own pawn chain
// counting double. Scaled *quadratically* by piece count, because space is
// only worth something if you have pieces to put in it — a space edge with
// four minors is crushing, the same edge in a bare endgame is nothing. This is
// a genuinely different claim from v0.2's square control (which counts
// contested squares anywhere on the board); this is specifically "my pawns
// have given my pieces somewhere to live". Middlegame only.
//
// (The pawn-storm term that shipped alongside this in v1.3 was removed in v1.5:
// a 13-version, 11.5k-game round-robin at 60ms measured it at roughly -36 Elo,
// the largest regression in the version history. See CHANGELOG.)
#ifndef SC_SPACE
#define SC_SPACE 0  // DISABLED: depth-10 attribution showed space inert on its
                   // own (-2 Elo) and actively harmful combined with storm
                   // (-16 vs +26 storm-only) — it re-measures v0.2 square
                   // control. Code kept for a future retest; 100 re-enables.
#endif

constexpr std::uint64_t kCenterFiles = (kFileABits << 2) | (kFileABits << 3) |
                                       (kFileABits << 4) | (kFileABits << 5);

// Middlegame space bonus for `color`.
// [[maybe_unused]]: compiled out while SC_SPACE=0 (kept for a future retest).
[[maybe_unused]] [[nodiscard]] int space_bonus(const Board& board, Color color,
                              std::uint64_t own_pawns,
                              std::uint64_t enemy_pawn_atk, std::uint64_t enemy_atk) {
    const bool white = (color == Color::WHITE);
    // Our half, centre files: ranks 2-4 for White, 5-7 for Black.
    const std::uint64_t half = white ? (0xFFULL << 8) | (0xFFULL << 16) | (0xFFULL << 24)
                                     : (0xFFULL << 32) | (0xFFULL << 40) | (0xFFULL << 48);
    const std::uint64_t area = kCenterFiles & half;

    const std::uint64_t safe = area & ~own_pawns & ~enemy_pawn_atk;

    // Squares tucked behind our own pawns count twice — that's the sheltered
    // manoeuvring room a pawn chain actually buys.
    std::uint64_t behind = own_pawns;
    behind |= white ? (behind >> 8) : (behind << 8);
    behind |= white ? (behind >> 16) : (behind << 16);

    const int bonus = std::popcount(safe) + std::popcount(behind & safe & ~enemy_atk);
    const int weight = std::max(0, board.us(color).count() - 3);
    return bonus * weight * weight / 16;
}

// ---- Bishop pair (v1.6) ----------------------------------------------------------
//
// "The two bishops are a long-term advantage." Material is otherwise strictly
// per-piece, so without this the engine trades B+B for B+N without complaint.
// The bonus applies only when a side's bishops cover BOTH square colors — that
// complementary coverage is the actual asset (two same-colored bishops from a
// promotion are just doubled coverage, not the pair). Endgame-heavy: the pair
// grows as the board opens and pawns come off. Values follow classical
// practice (~half a pawn combined at full amplitude); one knob for amplitude.
//
// The endgame value is larger than the middlegame value — the pair's scope
// grows as the board empties — but NOT by the 3.5-4x that jointly-tuned open
// engines use (Ethereal S(22,88), Weiss S(33,110)). A three-way bake-off in
// this engine's own eval (Opus {30,55}, mid {30,75}, consensus {27,100}, each
// vs the pair-blind v1.5 base, 1000 games x 2 seeds at depth 10) put the
// benefit at +23 / +7 / ~0 Elo respectively: the "consensus" EG value
// over-values the pair here and washes the gain out, while EG=75 peaks. Another
// case of imported constants not transferring across differently-tuned evals.
#ifndef SC_BPAIR
#define SC_BPAIR 100  // amplitude knob; 0 disables (control builds)
#endif
#ifndef SC_BPAIR_MG
#define SC_BPAIR_MG 30   // MG value; overridable for MG/EG-profile experiments
#endif
#ifndef SC_BPAIR_EG
#define SC_BPAIR_EG 75   // EG value (tuned here, below the open-engine consensus)
#endif
constexpr int kBPairMG = SC_BPAIR_MG, kBPairEG = SC_BPAIR_EG;

constexpr std::uint64_t kLightSquares = 0x55AA55AA55AA55AAULL;

// ---- Tempo (v1.7 candidate) -------------------------------------------------
// Flat bonus for the side to move, added after the perspective flip at every
// return path (lazy exit included). The modern top-HCE consensus is exactly
// this shape: a flat post-taper constant — SF classical 28, Ethereal 20,
// Weiss 18 — none taper it by phase or gate it on material/zugzwang. Besides
// its eval value (having the move is worth something), it damps score
// oscillation with search-depth parity. 0 == v1.6.4 exactly.
#ifndef SC_TEMPO
#define SC_TEMPO 16  // centipawns for the side to move (v1.7); 0 disables (control builds)
#endif

// ---- Outposts (v1.8 candidate) ----------------------------------------------
// A knight or bishop on a square that is (a) advanced (relative ranks 4-6),
// (b) defended by one of our own pawns, and (c) unreachable by any enemy pawn
// now or by advancing (outside the enemy pawn-attack span). Condition (c) is
// the non-redundant signal: PST already prices central/advanced squares and
// mobility already prices safe activity, but neither knows the square can never
// be challenged by a pawn lever. Requiring (b) ties the bonus to a pawn without
// re-scoring pawn structure (the pawn's own doubled/isolated/passed status is a
// separate term). SF classical scores knight:bishop 2:1, MG-weighted; we keep
// that shape but start well below SF's magnitude because our PST+mobility
// already front-load part of the value. One amplitude knob = the knight MG
// bonus; 0 == v1.7 exactly.
#ifndef SC_OUTPOST
#define SC_OUTPOST 0   // knight-on-outpost MG bonus in cp; 0 disables (control builds)
#endif
constexpr int kOutKnightMG = SC_OUTPOST;
constexpr int kOutKnightEG = SC_OUTPOST * 2 / 5;  // MG-weighted
constexpr int kOutBishopMG = SC_OUTPOST / 2;      // bishop = half knight (SF 2:1)
constexpr int kOutBishopEG = SC_OUTPOST / 5;

constexpr std::uint64_t kOutpostRanksW = 0x0000FFFFFF000000ULL;  // ranks 4,5,6
constexpr std::uint64_t kOutpostRanksB = 0x000000FFFFFF0000ULL;  // ranks 3,4,5 (black's 4-6)

// Forward fill: smear a bitboard toward higher (north) / lower (south) ranks.
[[nodiscard]] constexpr std::uint64_t north_fill(std::uint64_t b) {
    b |= b << 8; b |= b << 16; b |= b << 32; return b;
}
[[nodiscard]] constexpr std::uint64_t south_fill(std::uint64_t b) {
    b |= b >> 8; b |= b >> 16; b |= b >> 32; return b;
}

// ---- Stockfish PSQT + material anchor (NNUE mediation) -------------------------
//
// The NNUE is trained on pure game results, which leaves it materially and
// positionally under-grounded (a queen scored ~183cp at epoch 1). The old anchor
// — eval::material() — fixed the material half but is positionally blind: a
// knight on a1 and a knight on e5 score identically. These are Stockfish 14's
// own tables (src/psqt.cpp + the PieceValue constants in src/types.h), verbatim,
// so the anchor carries piece values AND placement.
//
// Layout follows SF exactly: Bonus is indexed [piece][rank][edge_distance(file)]
// — only files A-D are stored, and E-H mirror onto them via min(f, 7-f). Pawns
// get their own asymmetric full-width table (PBonus), with ranks 1 and 8 zero.
// Ranks are written RANK_1 first, from White's point of view; Black mirrors
// vertically (sq ^ 56), which is this engine's existing PST convention.
//
// Scale: SF's internal unit is ~1.5x ours (its MG pawn is 126, ours is 82), so
// the whole table is rescaled by one factor that maps SF's pawn onto ours. A
// single factor preserves SF's internally-tuned MG:EG ratios rather than
// re-deriving them; overall amplitude is the MaterialBlend knob's job.
#ifndef SC_PSQT_SCALE
#define SC_PSQT_SCALE 65  // SF units -> our cp (126 * 65% ~= 82, our MG pawn)
#endif

struct SfScore {
    int mg, eg;
};

// SF PieceValue[MG|EG], indexed by PieceType (P,N,B,R,Q,K).
constexpr std::array<int, 6> kSfPieceMG = {126, 781, 825, 1276, 2538, 0};
constexpr std::array<int, 6> kSfPieceEG = {208, 854, 915, 1380, 2682, 0};

// clang-format off
// Bonus[KNIGHT..KING][rank][min(file, 7-file)] — SF14 psqt.cpp verbatim.
constexpr SfScore kSfBonus[5][8][4] = {
    {   // Knight
        {{-175, -96}, { -92, -65}, { -74, -49}, { -73, -21}},
        {{ -77, -67}, { -41, -54}, { -27, -18}, { -15,   8}},
        {{ -61, -40}, { -17, -27}, {   6,  -8}, {  12,  29}},
        {{ -35, -35}, {   8,  -2}, {  40,  13}, {  49,  28}},
        {{ -34, -45}, {  13, -16}, {  44,   9}, {  51,  39}},
        {{  -9, -51}, {  22, -44}, {  58, -16}, {  53,  17}},
        {{ -67, -69}, { -27, -50}, {   4, -51}, {  37,  12}},
        {{-201,-100}, { -83, -88}, { -56, -56}, { -26, -17}},
    },
    {   // Bishop
        {{ -37, -40}, {  -4, -21}, {  -6, -26}, { -16,  -8}},
        {{ -11, -26}, {   6,  -9}, {  13, -12}, {   3,   1}},
        {{  -5, -11}, {  15,  -1}, {  -4,  -1}, {  12,   7}},
        {{  -4, -14}, {   8,  -4}, {  18,   0}, {  27,  12}},
        {{  -8, -12}, {  20,  -1}, {  15, -10}, {  22,  11}},
        {{ -11, -21}, {   4,   4}, {   1,   3}, {   8,   4}},
        {{ -12, -22}, { -10, -14}, {   4,  -1}, {   0,   1}},
        {{ -34, -32}, {   1, -29}, { -10, -26}, { -16, -17}},
    },
    {   // Rook
        {{ -31,  -9}, { -20, -13}, { -14, -10}, {  -5,  -9}},
        {{ -21, -12}, { -13,  -9}, {  -8,  -1}, {   6,  -2}},
        {{ -25,   6}, { -11,  -8}, {  -1,  -2}, {   3,  -6}},
        {{ -13,  -6}, {  -5,   1}, {  -4,  -9}, {  -6,   7}},
        {{ -27,  -5}, { -15,   8}, {  -4,   7}, {   3,  -6}},
        {{ -22,   6}, {  -2,   1}, {   6,  -7}, {  12,  10}},
        {{  -2,   4}, {  12,   5}, {  16,  20}, {  18,  -5}},
        {{ -17,  18}, { -19,   0}, {  -1,  19}, {   9,  13}},
    },
    {   // Queen
        {{   3, -69}, {  -5, -57}, {  -5, -47}, {   4, -26}},
        {{  -3, -54}, {   5, -31}, {   8, -22}, {  12,  -4}},
        {{  -3, -39}, {   6, -18}, {  13,  -9}, {   7,   3}},
        {{   4, -23}, {   5,  -3}, {   9,  13}, {   8,  24}},
        {{   0, -29}, {  14,  -6}, {  12,   9}, {   5,  21}},
        {{  -4, -38}, {  10, -18}, {   6, -11}, {   8,   1}},
        {{  -5, -50}, {   6, -27}, {  10, -24}, {   8,  -8}},
        {{  -2, -74}, {  -2, -52}, {   1, -43}, {  -2, -34}},
    },
    {   // King
        {{ 271,   1}, { 327,  45}, { 271,  85}, { 198,  76}},
        {{ 278,  53}, { 303, 100}, { 234, 133}, { 179, 135}},
        {{ 195,  88}, { 258, 130}, { 169, 169}, { 120, 175}},
        {{ 164, 103}, { 190, 156}, { 138, 172}, {  98, 172}},
        {{ 154,  96}, { 179, 166}, { 105, 199}, {  70, 199}},
        {{ 123,  92}, { 145, 172}, {  81, 184}, {  31, 191}},
        {{  88,  47}, { 120, 121}, {  65, 116}, {  33, 131}},
        {{  59,  11}, {  89,  59}, {  45,  73}, {  -1,  78}},
    },
};

// PBonus[rank][file] — pawns are asymmetric, so all eight files are stored.
// Ranks 1 and 8 are zero: a pawn is never on either.
constexpr SfScore kSfPawnBonus[8][8] = {
    {{  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}},
    {{  2, -8}, {  4, -6}, { 11,  9}, { 18,  5}, { 16, 16}, { 21,  6}, {  9, -6}, { -3,-18}},
    {{ -9, -9}, {-15, -7}, { 11,-10}, { 15,  5}, { 31,  2}, { 23,  3}, {  6, -8}, {-20, -5}},
    {{ -3,  7}, {-20,  1}, {  8, -8}, { 19, -2}, { 39,-14}, { 17,-13}, {  2,-11}, { -5, -6}},
    {{ 11, 12}, { -4,  6}, {-11,  2}, {  2, -6}, { 11, -5}, {  0, -4}, {-12, 14}, {  5,  9}},
    {{  3, 27}, {-11, 18}, { -6, 19}, { 22, 29}, { -8, 30}, { -5,  9}, {-14,  8}, {-11, 14}},
    {{ -7, -1}, {  6,-14}, { -2, 13}, {-11, 22}, {  4, 24}, {-14, 17}, { 10,  7}, { -9,  7}},
    {{  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}, {  0,  0}},
};
// clang-format on

// Damp a white-relative score toward zero by the halfmove clock.
[[nodiscard]] int scale_by_progress(int score, int hmc) {
    const int over = hmc - kProgOnset;
    if (over <= 0) return score;
    const int span = 100 - kProgOnset;                        // plies from onset to draw
    int       drop = (100 - kProgFloorPct) * over / span;     // percent removed, full strength
    drop           = drop * SC_PROGRESS / 100;                // scaled by the sweep knob
    return score * (100 - drop) / 100;
}

}  // namespace

void init() {
    // Nothing to precompute yet — the tables are compile-time constants. This
    // hook exists so future HCE terms (e.g. distance tables) have a place to
    // initialise from without touching startup code elsewhere.
}

// Material-blend weight in percent. Baked in at 90% (NNUE + 0.90 * material);
// runtime-tunable via the "MaterialBlend" UCI option, so pure NNUE (0) and the
// original hand-crafted eval remain reachable from the same binary.
int g_material_blend = 0;

void set_material_blend(int pct) { g_material_blend = pct; }

Value material(const Board& board) {
    static constexpr std::array<PieceType::underlying, 5> kMatTypes = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN};
    int w = 0, b = 0;
    for (int i = 0; i < 5; ++i) {
        w += kMaterialMG[i] *
             std::popcount(board.pieces(PieceType(kMatTypes[i]), Color::WHITE).getBits());
        b += kMaterialMG[i] *
             std::popcount(board.pieces(PieceType(kMatTypes[i]), Color::BLACK).getBits());
    }
    const Value s = w - b;
    return (board.sideToMove() == Color::WHITE) ? s : -s;
}

// Stockfish PSQT + material, tapered MG->EG on this engine's existing phase, and
// rescaled from SF's unit to ours. Side-to-move-relative centipawns. This is the
// anchor that grounds a materially/positionally naive NNUE; it supersedes the
// flat material() above, which knows piece values but not where they stand.
Value psqt_material(const Board& board) {
    int mg = 0, eg = 0, phase = 0;

    for (const Color color : {Color::WHITE, Color::BLACK}) {
        const int sign = (color == Color::WHITE) ? 1 : -1;
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard bb = board.pieces(PieceType(static_cast<PieceType::underlying>(pt)), color);
            while (bb) {
                const int sq = bb.pop();
                // Mirror Black vertically so every table read is White-relative.
                const int idx  = (color == Color::WHITE) ? sq : (sq ^ 56);
                const int rank = idx >> 3;
                const int file = idx & 7;
                // SF stores files A-D only; E-H mirror onto them.
                const SfScore b = (pt == static_cast<int>(PieceType::PAWN))
                                      ? kSfPawnBonus[rank][file]
                                      : kSfBonus[pt - 1][rank][std::min(file, 7 - file)];
                mg += sign * (kSfPieceMG[pt] + b.mg);
                eg += sign * (kSfPieceEG[pt] + b.eg);
                phase += kPhaseInc[pt];
            }
        }
    }

    if (phase > kPhaseMax) phase = kPhaseMax;
    int score = (mg * phase + eg * (kPhaseMax - phase)) / kPhaseMax;
    score = score * SC_PSQT_SCALE / 100;  // SF units -> our centipawns

    return static_cast<Value>((board.sideToMove() == Color::WHITE) ? score : -score);
}

// Hand-crafted evaluation (material + PST + mobility + king safety + ...), split
// out from evaluate() so the latter can fuse it with the NNUE. Returns a
// side-to-move-relative centipawn score.
Value hce_eval(const Board& board) {
    int mg = 0, eg = 0, phase = 0;

    add_side(board, Color::WHITE, mg, eg, phase);
    add_side(board, Color::BLACK, mg, eg, phase);

    if (phase > kPhaseMax) phase = kPhaseMax;

    // Lazy eval: if material+PST is already decisive, mobility/control noise
    // can't flip the verdict — skip the attack computation entirely.
    int score = (mg * phase + eg * (kPhaseMax - phase)) / kPhaseMax;
    if (score < -kLazyMargin || score > kLazyMargin)
        return ((board.sideToMove() == Color::WHITE) ? score : -score) + SC_TEMPO;

    // ---- Bishop pair ----
    if constexpr (SC_BPAIR > 0) {
        auto has_pair = [&](Color c) {
            const std::uint64_t b = board.pieces(PieceType::BISHOP, c).getBits();
            return (b & kLightSquares) && (b & ~kLightSquares) ? 1 : 0;
        };
        const int pair = has_pair(Color::WHITE) - has_pair(Color::BLACK);
        mg += pair * kBPairMG * SC_BPAIR / 100;
        eg += pair * kBPairEG * SC_BPAIR / 100;
    }

    // ---- Mobility + control ----
    using A = chess::attacks;  // static-member "namespace" class in the library

    const Bitboard wp = board.pieces(PieceType::PAWN, Color::WHITE);
    const Bitboard bp = board.pieces(PieceType::PAWN, Color::BLACK);

    const Bitboard wp_l = A::pawnLeftAttacks<Color::WHITE>(wp);
    const Bitboard wp_r = A::pawnRightAttacks<Color::WHITE>(wp);
    const Bitboard bp_l = A::pawnLeftAttacks<Color::BLACK>(bp);
    const Bitboard bp_r = A::pawnRightAttacks<Color::BLACK>(bp);

    // King zones: the king plus its ring of squares.
    const Square   wk     = board.kingSq(Color::WHITE);
    const Square   bk     = board.kingSq(Color::BLACK);
    const Bitboard zone_w = A::king(wk) | Bitboard::fromSquare(wk);
    const Bitboard zone_b = A::king(bk) | Bitboard::fromSquare(bk);

    SideData sd_w, sd_b;
    add_mobility(board, Color::WHITE, bp_l | bp_r, wp_l | wp_r, wp_l & wp_r, zone_w, zone_b,
                 sd_w, mg, eg);
    add_mobility(board, Color::BLACK, wp_l | wp_r, bp_l | bp_r, bp_l & bp_r, zone_b, zone_w,
                 sd_b, mg, eg);

    // Controlled squares: we cover them strictly harder than the opponent
    // (their-side-only coverage, or our double vs their single).
    const Bitboard ctrl_w = (sd_w.atk & ~sd_b.atk) | (sd_w.atk2 & sd_b.atk & ~sd_b.atk2);
    const Bitboard ctrl_b = (sd_b.atk & ~sd_w.atk) | (sd_b.atk2 & sd_w.atk & ~sd_w.atk2);

    const int d_all    = ctrl_w.count() - ctrl_b.count();
    const int d_center = (ctrl_w & kCenterMask).count() - (ctrl_b & kCenterMask).count();

    mg += kControlMG * d_all + kControlCenterMG * d_center;
    eg += kControlEG * d_all;

    // ---- King attacks ----
    // Pawn pressure on the enemy zone (capped) and pawn shield in our own zone
    // (capped) come from the aggregate bitboards; pieces were tallied above.
    const int wp_press  = std::min(2, ((wp_l | wp_r) & zone_b).count());
    const int bp_press  = std::min(2, ((bp_l | bp_r) & zone_w).count());
    const int w_shield  = std::min(3, (wp & zone_w).count());
    const int b_shield  = std::min(3, (bp & zone_b).count());

    const int katt = king_attack_value(sd_w, sd_b, wp_press, b_shield) -
                     king_attack_value(sd_b, sd_w, bp_press, w_shield);
    mg += katt;
    eg += katt / 4;

    // ---- King safety (defender's ledger) ----
    if constexpr (SC_KSAFE > 0) {
        const bool wq_alive  = !board.pieces(PieceType::QUEEN, Color::WHITE).empty();
        const bool bq_alive  = !board.pieces(PieceType::QUEEN, Color::BLACK).empty();
        const bool queens_off = !wq_alive && !bq_alive;

        const KingSafety ks_w = king_safety(board, Color::WHITE, sd_b.atk, bq_alive, queens_off);
        const KingSafety ks_b = king_safety(board, Color::BLACK, sd_w.atk, wq_alive, queens_off);

        mg -= (ks_w.mg - ks_b.mg) * SC_KSAFE / 100;
        eg -= (ks_w.eg - ks_b.eg) * SC_KSAFE / 100;
    }

    // ---- Pawn structure ----
    // constexpr-guarded so a scale-0 build compiles the whole term out — that
    // build is then eval- and speed-identical to the pre-pawn version (used to
    // reconstruct baselines for testing).
    if constexpr (SC_PAWN_SCALE > 0) {
        const std::uint64_t occ_bits = board.occ().getBits();
        add_pawn_structure(Color::WHITE, wp.getBits(), bp.getBits(), sd_w.atk.getBits(),
                           occ_bits, mg, eg);
        add_pawn_structure(Color::BLACK, bp.getBits(), wp.getBits(), sd_b.atk.getBits(),
                           occ_bits, mg, eg);
    }

    // ---- Threats ----
    if constexpr (SC_THREAT > 0) {
        const std::uint64_t occ_bits = board.occ().getBits();
        add_threats(board, Color::WHITE, sd_w, sd_b, wp.getBits(),
                    board.us(Color::BLACK).getBits(), bp.getBits(), occ_bits, mg, eg);
        add_threats(board, Color::BLACK, sd_b, sd_w, bp.getBits(),
                    board.us(Color::WHITE).getBits(), wp.getBits(), occ_bits, mg, eg);
    }

    // ---- Space ----
    if constexpr (SC_SPACE > 0) {
        const std::uint64_t w_pawn_atk = (wp_l | wp_r).getBits();
        const std::uint64_t b_pawn_atk = (bp_l | bp_r).getBits();
        const int spw = space_bonus(board, Color::WHITE, wp.getBits(), b_pawn_atk,
                                    sd_b.atk.getBits());
        const int spb = space_bonus(board, Color::BLACK, bp.getBits(), w_pawn_atk,
                                    sd_w.atk.getBits());
        mg += (spw - spb) * SC_SPACE / 100;
    }

    // ---- Outposts ----
    // Minor on an advanced, pawn-defended square outside the enemy pawn-attack
    // span. Own pawn attacks (wp_l|wp_r) supply the defense predicate; the
    // enemy's forward-filled pawn attacks give every square a pawn could ever
    // reach, so ~span is "no enemy pawn lever, now or later".
    if constexpr (SC_OUTPOST > 0) {
        const std::uint64_t w_pawn_atk = (wp_l | wp_r).getBits();
        const std::uint64_t b_pawn_atk = (bp_l | bp_r).getBits();
        const std::uint64_t w_out = kOutpostRanksW & w_pawn_atk & ~south_fill(b_pawn_atk);
        const std::uint64_t b_out = kOutpostRanksB & b_pawn_atk & ~north_fill(w_pawn_atk);

        const std::uint64_t wkn = board.pieces(PieceType::KNIGHT, Color::WHITE).getBits() & w_out;
        const std::uint64_t wbi = board.pieces(PieceType::BISHOP, Color::WHITE).getBits() & w_out;
        const std::uint64_t bkn = board.pieces(PieceType::KNIGHT, Color::BLACK).getBits() & b_out;
        const std::uint64_t bbi = board.pieces(PieceType::BISHOP, Color::BLACK).getBits() & b_out;

        const int dkn = std::popcount(wkn) - std::popcount(bkn);
        const int dbi = std::popcount(wbi) - std::popcount(bbi);
        mg += dkn * kOutKnightMG + dbi * kOutBishopMG;
        eg += dkn * kOutKnightEG + dbi * kOutBishopEG;
    }

    score = (mg * phase + eg * (kPhaseMax - phase)) / kPhaseMax;

    // ---- Progress scaling ----
    // Decay the (white-relative) score toward a draw the longer the game has
    // gone without an irreversible move. Applied last, before the perspective
    // flip, so every consumer sees one consistent value.
    if constexpr (SC_PROGRESS > 0)
        score = scale_by_progress(score, static_cast<int>(board.halfMoveClock()));

    // Return from the side-to-move's perspective, plus the tempo bonus.
    return ((board.sideToMove() == Color::WHITE) ? score : -score) + SC_TEMPO;
}

// ---- HCE / NNUE fusion -------------------------------------------------------
//
// When a net is loaded, evaluate() fuses HCE and NNUE. Two percent knobs:
//   g_nnue_weight (w): NNUE's share of the fused score; HCE gets (100 - w).
//   g_nnue_scale  (s): pre-scales the raw NNUE first, to correct its "hot" cp
//                      scale relative to real centipawns.
//   fused = ((100 - w) * hce + w * (s * nnue / 100)) / 100
// Defaults w=100, s=100 reproduce the prior pure-NNUE behavior (plus the
// material anchor). The self-play data-gen uses w=20, s=30 -> 0.80*HCE + 0.06*NNUE.
int g_nnue_weight = 100;
int g_nnue_scale  = 100;

void set_nnue_weight(int pct) { g_nnue_weight = pct; }
void set_nnue_scale(int pct)  { g_nnue_scale = pct; }

Value evaluate(const Board& board) {
    // HCE is only used with no net loaded, or in the fusion path (w<100). At pure
    // NNUE (w>=100) it is never used, so it is never computed there — no wasted work.
    if (!nnue::loaded())
        return hce_eval(board);

    const Value nn = nnue::evaluate(board);

    // Pure NNUE (w>=100): the net stands alone, so anchor it on Stockfish's
    // PSQT+material. Deliberately NOT applied in the fusion path below — HCE
    // already carries material and PST there, and adding it again would
    // double-count (and would move the self-play data-gen's labels).
    if (g_nnue_weight >= 100) {
        Value v = nn;
        if (g_material_blend != 0)
            v += static_cast<Value>(
                static_cast<long>(g_material_blend) * psqt_material(board) / 100);
        return v;
    }

    // Fusion: HCE already carries material, so the material anchor is not applied.
    const Value hce = hce_eval(board);
    const long scaled = static_cast<long>(g_nnue_scale) * nn / 100;
    return static_cast<Value>(
        (static_cast<long>(100 - g_nnue_weight) * hce +
         static_cast<long>(g_nnue_weight) * scaled) / 100);
}

}  // namespace engine::eval
