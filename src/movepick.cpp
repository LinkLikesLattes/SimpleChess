// -----------------------------------------------------------------------------
// movepick.cpp
//
// Score tiers (see header). The absolute numbers only need to keep the tiers
// disjoint: quiet history is bounded by ±3*HISTORY_MAX ≈ ±49k, well inside the
// ±800k gap between the killer tier and the bad-capture tier.
// -----------------------------------------------------------------------------

#include "movepick.hpp"

#include <utility>

#include "see.hpp"

namespace engine {

namespace {
constexpr int kTTScore         = 4'000'000;
constexpr int kGoodCaptureBase = 2'000'000;
constexpr int kQueenPromoScore = 1'900'000;
constexpr int kKiller0Score    = 900'000;
constexpr int kKiller1Score    = 850'000;
constexpr int kCounterScore    = 800'000;
constexpr int kBadCaptureBase  = -2'000'000;
constexpr int kUnderPromoScore = -3'000'000;
}  // namespace

MovePicker::MovePicker(const Board& board, const History& hist, const OrderingContext& ctx,
                       bool captures_only)
    : board_(board) {
    if (captures_only && !board.inCheck())
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves_, board);
    else
        chess::movegen::legalmoves(moves_, board);

    score(hist, ctx);
}

void MovePicker::score(const History& hist, const OrderingContext& ctx) {
    for (int i = 0; i < moves_.size(); ++i) {
        const Move m = moves_[i];

        if (m == ctx.tt_move) {
            scores_[i] = kTTScore;
            continue;
        }

        if (m.typeOf() == Move::PROMOTION) {
            // Queen promotions are near-captures; underpromotions are almost
            // never best and get searched dead last.
            scores_[i] = (m.promotionType() == PieceType(PieceType::QUEEN)) ? kQueenPromoScore
                                                                            : kUnderPromoScore;
            continue;
        }

        if (board_.isCapture(m)) {
            const int moved  = static_cast<int>(board_.at(m.from()));
            const int victim = static_cast<int>(board_.getCapturing<PieceType>(m));
            const int base   = see::see_ge(board_, m, 0) ? kGoodCaptureBase : kBadCaptureBase;
            // MVV dominates; capture history breaks ties within a victim class.
            scores_[i] = base + 16 * see::kPieceValue[victim] + hist.capture[moved][m.to().index()][victim];
            continue;
        }

        // Quiets: killers, countermove, then history.
        if (m == ctx.killer0) {
            scores_[i] = kKiller0Score;
        } else if (m == ctx.killer1) {
            scores_[i] = kKiller1Score;
        } else if (m == ctx.counter) {
            scores_[i] = kCounterScore;
        } else {
            const int piece = static_cast<int>(board_.at(m.from()));
            scores_[i]      = hist.quiet_score(ctx.stm, m.from().index(), m.to().index(), piece,
                                               ctx.prev1_piece, ctx.prev1_to, ctx.prev2_piece,
                                               ctx.prev2_to);
        }
    }
}

Move MovePicker::next(bool skip_quiets) {
    while (cur_ < moves_.size()) {
        // Selection step: float the best remaining move (and its score) to cur_.
        int best = cur_;
        for (int j = cur_ + 1; j < moves_.size(); ++j)
            if (scores_[j] > scores_[best]) best = j;
        if (best != cur_) {
            std::swap(moves_[cur_], moves_[best]);
            std::swap(scores_[cur_], scores_[best]);
        }

        const Move m = moves_[cur_];
        last_score_  = scores_[cur_];
        ++cur_;

        if (skip_quiets && !board_.isCapture(m) && m.typeOf() != Move::PROMOTION) continue;

        return m;
    }
    return Move(Move::NO_MOVE);
}

}  // namespace engine
