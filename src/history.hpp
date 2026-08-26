#pragma once

// -----------------------------------------------------------------------------
// history.hpp
//
// Statistics tables that power move ordering and LMR:
//
//   * main (butterfly) history  — [stm][from][to], quiet moves that caused
//     cutoffs anywhere in the tree
//   * capture history           — [moving piece][to][captured type]
//   * countermove               — [prev piece][prev to] -> the quiet refutation
//   * continuation history      — [prev piece][prev to][piece][to], the quiet
//     follow-ups that work after a given move; consulted at 1 and 2 plies back
//
// All numeric tables use the "gravity" update h += bonus - h*|bonus|/MAX so
// values saturate smoothly at ±HISTORY_MAX and recent results outweigh stale
// ones. Piece indices are the library's Piece encoding 0..11; index 12 is the
// "no previous move" (root / null move) sentinel and is simply never consulted.
// -----------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>
#include <memory>

#include "types.hpp"

namespace engine {

constexpr int HISTORY_MAX = 16384;

// Static-eval correction table sizing. The stored value is a learned centipawn
// bias (times CORR_GRAIN) between the raw static eval and what search actually
// returns, bucketed by pawn structure; a power-of-two bucket count keeps the
// index a cheap mask. Same saturation ceiling as the move-history tables so the
// one gravity update below serves both.
constexpr int CORR_SIZE  = 16384;      // pawn-structure buckets (2^14)
constexpr int CORR_GRAIN = 128;        // stored units per centipawn of correction

// Saturating history update. `bonus` should already be clamped to ±HISTORY_MAX.
inline void history_update(std::int16_t& h, int bonus) noexcept {
    const int v = h + bonus - h * std::abs(bonus) / HISTORY_MAX;
    h           = static_cast<std::int16_t>(v);
}

struct History {
    // Quiet-move butterfly history: [side to move][from][to].
    std::int16_t main[2][64][64];

    // Capture history: [moving piece (0..11)][to][captured piece type (0..5)].
    std::int16_t capture[12][64][6];

    // Countermove: the quiet move that refuted [prev piece (0..12)][prev to].
    Move counter[13][64];

    // Continuation history, indexed by the previous move's (piece, to) and then
    // the candidate move's (piece, to). ~1.2 MB, so it lives on the heap.
    struct ContTable {
        std::int16_t v[13][64][12][64];
    };
    std::unique_ptr<ContTable> cont;

    // Static-eval correction, bucketed by [side to move][pawn-structure hash].
    // Learns the running gap between the raw static eval and the searched score
    // for positions sharing a pawn skeleton, so the eval that feeds pruning can
    // be nudged toward what search keeps discovering.
    std::int16_t corr_pawn[2][CORR_SIZE];

    History() : cont(std::make_unique<ContTable>()) { clear(); }

    void clear() {
        std::memset(main, 0, sizeof(main));
        std::memset(capture, 0, sizeof(capture));
        std::memset(cont->v, 0, sizeof(cont->v));
        std::memset(corr_pawn, 0, sizeof(corr_pawn));
        for (auto& row : counter)
            for (auto& m : row) m = Move(Move::NO_MOVE);
    }

    // Continuation entry for (previous move piece/to, candidate piece/to).
    [[nodiscard]] std::int16_t& cont_entry(int prev_piece, int prev_to, int piece,
                                           int to) noexcept {
        return cont->v[prev_piece][prev_to][piece][to];
    }

    // Combined quiet score used for ordering and LMR adjustment: butterfly plus
    // the continuation entries for 1 and 2 plies back (sentinel-safe).
    [[nodiscard]] int quiet_score(int stm, int from, int to, int piece, int prev1_piece,
                                  int prev1_to, int prev2_piece, int prev2_to) const noexcept {
        int s = main[stm][from][to];
        if (prev1_piece < 12) s += cont->v[prev1_piece][prev1_to][piece][to];
        if (prev2_piece < 12) s += cont->v[prev2_piece][prev2_to][piece][to];
        return s;
    }
};

}  // namespace engine
