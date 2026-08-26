#pragma once

// -----------------------------------------------------------------------------
// see.hpp
//
// Static Exchange Evaluation (SEE). Answers "does this move win at least
// `threshold` centipawns of material after the best sequence of recaptures on
// the target square?" without doing any search. Used for:
//   * move ordering  — good captures before killers, bad captures last
//   * search pruning — skip moves that lose material at shallow depth
//   * qsearch        — don't bother searching losing captures
//
// The implementation is the standard "swap algorithm": alternately apply the
// least valuable attacker of each side, adding x-ray attackers as sliders are
// removed, until one side runs out or refuses the exchange.
// -----------------------------------------------------------------------------

#include <array>

#include "types.hpp"

namespace engine::see {

// Piece values used *only* for exchange evaluation and pruning margins. Indexed
// by PieceType; index 6 (NONE, i.e. quiet moves) is 0 by construction.
constexpr std::array<int, 7> kPieceValue = {100, 320, 330, 500, 950, 0, 0};

[[nodiscard]] inline int value(PieceType pt) noexcept { return kPieceValue[static_cast<int>(pt)]; }

// True if the exchange sequence started by `move` nets >= `threshold` cp.
// Castling/en-passant/promotions are approximated conservatively (see .cpp).
[[nodiscard]] bool see_ge(const Board& board, Move move, int threshold) noexcept;

}  // namespace engine::see
