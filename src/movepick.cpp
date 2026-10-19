// -----------------------------------------------------------------------------
// movepick.cpp
//
// Score tiers (see header). The absolute numbers only need to keep the tiers
// disjoint: quiet history is bounded by ±3*HISTORY_MAX ≈ ±49k, well inside the
// ±800k gap between the killer tier and the bad-capture tier.
// -----------------------------------------------------------------------------

#include "movepick.hpp"

#include <climits>
#include <utility>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

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
    if (captures_only) {   // the caller passes false in check (it already knows: no second inCheck())
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
    // M3x: the AVX2 selection scan reads 8 scores at a time past the last move; INT_MIN there
    // never wins (real scores are within +-4M).
    for (int k = 0; k < 8; ++k) scores_[moves_.size() + k] = INT_MIN;
}

// NPS 3.6 item M4: once quiets are skipped (late move pruning), every remaining quiet still wins
// the selection scan before any capture or promotion is reached, and is then thrown away -- one
// full rescan per discarded quiet. The non-quiets come out in score order, with array position
// deciding only between EQUAL scores. So on the first skip call: if no non-quiet remains, the
// picker is done; if the remaining non-quiets have pairwise distinct scores, take them by score
// directly (their order cannot depend on where the quiets would have moved them); if two share a
// score, keep the exact scan for the rest of the node. Same moves, same order -> bit-identical.
#ifndef SC_NPS_M4
#define SC_NPS_M4 1
#endif

// NPS 3.6 item M1: the selection scan keeps the best score in a register instead of reloading
// scores_[best] each step (a dependent load chain). Same comparisons, same pick -> bit-identical.
#ifndef SC_NPS_M1
#define SC_NPS_M1 1
#endif

// x86 M3x: the selection scan as an AVX2 max over the remaining scores, then the first index
// holding that max (cmpeq + movemask + tzcnt) -- the same move the strict-greater scalar scan
// keeps, which compiles to a chain of mispredicting jg on x86.
#ifndef SC_NPS_M3X
#if defined(__AVX2__)
#define SC_NPS_M3X 1
#else
#define SC_NPS_M3X 0
#endif
#endif

Move MovePicker::next(bool skip_quiets) {
    if (SC_NPS_M4 && skip_quiets && skip_mode_ != 2) {
        auto non_quiet = [&](int i) {
            return board_.isCapture(moves_[i]) || moves_[i].typeOf() == Move::PROMOTION;
        };
        if (skip_mode_ == 0) {   // decide once per node; the remaining non-quiets only shrink
            skip_mode_ = 1;
            for (int i = cur_; i < moves_.size() && skip_mode_ == 1; ++i)
                if (non_quiet(i))
                    for (int j = i + 1; j < moves_.size(); ++j)
                        if (scores_[j] == scores_[i] && non_quiet(j)) { skip_mode_ = 2; break; }
        }
        if (skip_mode_ == 1) {
            int best = -1;
            for (int i = cur_; i < moves_.size(); ++i)
                if (non_quiet(i) && (best < 0 || scores_[i] > scores_[best])) best = i;
            if (best < 0) { cur_ = moves_.size(); return Move(Move::NO_MOVE); }
            std::swap(moves_[cur_], moves_[best]);
            std::swap(scores_[cur_], scores_[best]);
            std::swap(see_[cur_], see_[best]);
            last_score_ = scores_[cur_];
            last_see_   = see_[cur_];
            return moves_[cur_++];
        }
    }
    while (cur_ < moves_.size()) {
        // Selection step: float the best remaining move (and its score) to cur_.
        int best = cur_;
#if defined(__AVX2__)
        if constexpr (SC_NPS_M3X) {
            const int n = moves_.size();
            __m256i mx = _mm256_set1_epi32(INT_MIN);
            for (int j = cur_; j < n; j += 8)
                mx = _mm256_max_epi32(mx, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scores_ + j)));
            __m128i m4 = _mm_max_epi32(_mm256_castsi256_si128(mx), _mm256_extracti128_si256(mx, 1));
            m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0x4E));
            m4 = _mm_max_epi32(m4, _mm_shuffle_epi32(m4, 0xB1));
            const __m256i target = _mm256_broadcastd_epi32(m4);
            for (int j = cur_;; j += 8) {
                const unsigned mask = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scores_ + j)), target))));
                if (mask) { best = j + __builtin_ctz(mask); break; }
            }
        } else
#endif
        if constexpr (SC_NPS_M1) {
            int best_score = scores_[cur_];
            for (int j = cur_ + 1; j < moves_.size(); ++j)
                if (scores_[j] > best_score) { best = j; best_score = scores_[j]; }
        } else {
            for (int j = cur_ + 1; j < moves_.size(); ++j)
                if (scores_[j] > scores_[best]) best = j;
        }
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
