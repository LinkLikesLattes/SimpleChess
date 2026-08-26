#pragma once

// -----------------------------------------------------------------------------
// movepick.hpp
//
// Move ordering. Generates the legal moves once, scores them into tiers, and
// hands them out best-first via incremental selection (so an early beta cutoff
// never pays for sorting the tail). The tier order:
//
//   TT move > good captures (SEE >= 0, by MVV + capture history) > queen promos
//   > killers > countermove > quiets (by combined history) > bad captures
//   > underpromotions
//
// A staged generator (captures first, quiets only if needed) is the natural
// future optimization; the interface (`next()`) already supports swapping the
// internals for one.
// -----------------------------------------------------------------------------

#include "history.hpp"
#include "types.hpp"

namespace engine {

// Everything the picker needs to know about "where we are" in the tree, kept
// explicit so this file has no dependency on the search stack layout.
struct OrderingContext {
    Move tt_move  = Move(Move::NO_MOVE);
    Move killer0  = Move(Move::NO_MOVE);
    Move killer1  = Move(Move::NO_MOVE);
    Move counter  = Move(Move::NO_MOVE);
    int  stm      = 0;   // side to move (0/1)
    int  prev1_piece = 12, prev1_to = 0;  // move made 1 ply ago (12 == none)
    int  prev2_piece = 12, prev2_to = 0;  // move made 2 plies ago
};

class MovePicker {
   public:
    // `captures_only` selects quiescence generation (ignored when the side to
    // move is in check — evasions need the full move list).
    MovePicker(const Board& board, const History& hist, const OrderingContext& ctx,
               bool captures_only);

    // Next best move, or NO_MOVE when exhausted. When `skip_quiets` is set,
    // non-captures/non-promotions are silently discarded (late move pruning).
    [[nodiscard]] Move next(bool skip_quiets);

    // Ordering score of the move most recently returned by next(). For quiets
    // this is the combined history score, which LMR uses to tune reductions.
    [[nodiscard]] int last_score() const noexcept { return last_score_; }

   private:
    void score(const History& hist, const OrderingContext& ctx);

    const Board& board_;
    Movelist     moves_;
    int          scores_[MAX_MOVES];
    int          cur_        = 0;
    int          last_score_ = 0;
};

}  // namespace engine
