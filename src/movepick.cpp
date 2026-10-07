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
    if (captures_only && !board.inCheck()) {
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves_, board);
        // The capture generator leaves out non-capturing promotions; a queening move belongs at
        // the horizon as much as a capture does. Only when a pawn stands on its 7th rank.
        const Color    us    = board.sideToMove();
        const Bitboard rank7 = Bitboard(us == Color::WHITE ? 0x00FF000000000000ULL : 0x000000000000FF00ULL);
        if (SC_QS_PROMO && !(board.pieces(PieceType::PAWN, us) & rank7).empty()) {
            Movelist quiet;
            chess::movegen::legalmoves<chess::movegen::MoveGenType::QUIET>(quiet, board,
                                                                         chess::PieceGenType::PAWN);
            for (const Move m : quiet)
                if (m.typeOf() == Move::PROMOTION && m.promotionType() == PieceType::QUEEN) moves_.add(m);
        }
    } else {
        chess::movegen::legalmoves(moves_, board);
    }

    score(hist, ctx);
}

void MovePicker::score(const History& hist, const OrderingContext& ctx) {
    // E6: the continuation rows depend only on the context; resolve them once.
    const History::ContRow cont1 = hist.cont_row(ctx.prev1_piece, ctx.prev1_to);
    const History::ContRow cont2 = hist.cont_row(ctx.prev2_piece, ctx.prev2_to);
    for (int i = 0; i < moves_.size(); ++i) {
        const Move m = moves_[i];
        see_[i]      = 0;   // unknown until the capture branch classifies it

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
            const bool good  = see::see_ge(board_, m, 0);
            see_[i]          = good ? 1 : -1;   // remembered for the search's SEE gates (last_see())
            const int base   = good ? kGoodCaptureBase : kBadCaptureBase;
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
            // Same three terms in the same order as History::quiet_score.
            const int to = m.to().index();
            int       s  = hist.main[ctx.stm][m.from().index()][to];
            if (cont1) s += cont1[piece][to];
            if (cont2) s += cont2[piece][to];
            scores_[i] = s;
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
            std::swap(see_[cur_], see_[best]);
        }

        const Move m = moves_[cur_];
        last_score_  = scores_[cur_];
        last_see_    = see_[cur_];
        ++cur_;

        if (skip_quiets && !board_.isCapture(m) && m.typeOf() != Move::PROMOTION) continue;

        return m;
    }
    return Move(Move::NO_MOVE);
}

}  // namespace engine
