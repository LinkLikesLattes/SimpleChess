// -----------------------------------------------------------------------------
// search.cpp
//
// Lazy SMP pool + the per-worker alpha-beta search. Layout of Worker::negamax(),
// top to bottom:
//
//   1. horizon dispatch to qsearch, draw checks, mate-distance pruning
//   2. TT probe (+ cutoff at non-PV nodes)
//   3. static evaluation, `improving` flag
//   4. whole-node pruning: IIR, razoring, reverse futility, null move, ProbCut
//   5. move loop: per-move pruning (LMP / futility / SEE), singular extensions,
//      PVS with late move reductions, alpha/beta bookkeeping, PV tracking
//   6. history/killer updates on a cutoff, TT store
//
// Every technique is a self-contained block with its margins defined in the
// "tunables" section below. All pruning is gated on
// `best > VALUE_MATED_IN_MAX_PLY`, which guarantees the first move of every
// node is searched in full — no node can "prune itself to death".
//
// Threading contract: the shared stop flag lives in the pool; after every
// recursive call, check it before using the returned score — a stopped search
// returns meaningless values that must never reach the TT, history tables, or
// best_move_. The TT itself is shared and unlocked (see search.hpp banner).
// -----------------------------------------------------------------------------

#include "search.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#include "evaluate.hpp"
#include "movepick.hpp"
#include "nnue.hpp"
#include "see.hpp"

namespace engine {

// Root-move noise for self-play data generation. When > 0, each root move gets a
// small random +-cp bonus (reseeded implicitly as the per-thread RNG advances),
// so the engine picks among near-equal moves and repeated games diverge. Mate
// scores are never perturbed. Off (0) by default — normal play is unaffected.
namespace {
std::atomic<int>              g_root_noise{0};
bool                          g_gen_silent = false;  // set before gengame's searches (single UCI thread)
thread_local std::mt19937_64  t_noise_rng{std::random_device{}()};
thread_local std::uint64_t    t_noise_seed{0};   // fixed per search (set in think())

// Deterministic per (search seed, move) offset in [-noise, +noise]. Fixed across
// a search's deepening iterations, so it does not drift through the aspiration
// window; varies per search so repeated games diverge.
[[nodiscard]] int root_noise_offset(Move m, int noise) {
    std::uint64_t h = t_noise_seed ^ (0x9E3779B97F4A7C15ULL * (m.move() + 1ULL));
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 27;
    return static_cast<int>(h % (2ULL * noise + 1ULL)) - noise;
}
}  // namespace

void set_root_noise(int cp) { g_root_noise.store(cp < 0 ? 0 : cp, std::memory_order_relaxed); }

// Set before any gengame search on the UCI thread; worker threads (created in
// start()) observe it via the thread-creation happens-before. Not changed mid-search.
void set_gen_silent(bool silent) { g_gen_silent = silent; }

namespace {

// Static-eval correction history: learn the running gap between the raw static
// eval and the score search returns, bucketed by pawn structure, and fold it
// into the eval that drives pruning. Build-time switch so a control build (0)
// reproduces the prior search byte-for-byte for A/B measurement.
#ifndef SC_CORRHIST
#define SC_CORRHIST 1
#endif

// Bucket a position by its pawn skeleton; the [side to move] split is applied by
// the caller. splitmix-finalized so near-identical skeletons still scatter.
[[nodiscard]] inline int pawn_corr_index(const Board& board) {
    std::uint64_t h = board.pieces(PieceType::PAWN, Color::WHITE).getBits();
    h ^= 0x9E3779B97F4A7C15ULL * (board.pieces(PieceType::PAWN, Color::BLACK).getBits() + 1ULL);
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 27;
    h *= 0x94D049BB133111EBULL; h ^= h >> 31;
    return static_cast<int>(h & (CORR_SIZE - 1));
}

// ---- Tunables ----------------------------------------------------------------
// Margins are in centipawns unless noted. These are sane starting points, not
// tuned values — SPSA them once the evaluation stabilizes.

constexpr Depth kIIRMinDepth       = 4;    // reduce when no TT move at/above this depth
constexpr Depth kRazorMaxDepth     = 3;    // razoring applies at depth <= this
constexpr int   kRazorMargin       = 300;  // per-depth margin below alpha
constexpr Depth kRFPMaxDepth       = 8;    // reverse futility max depth
constexpr int   kRFPMargin         = 80;   // per-depth margin above beta
constexpr Depth kNMPMinDepth       = 3;    // null move minimum depth
constexpr Depth kProbCutMinDepth   = 5;
constexpr int   kProbCutMargin     = 200;  // beta + margin must be beaten by a capture
constexpr Depth kSingularMinDepth  = 8;
constexpr Depth kFutilityMaxDepth  = 10;   // futility pruning of quiets
constexpr int   kFutilityBase      = 100;
constexpr int   kFutilityPerDepth  = 120;
constexpr Depth kSeeGateMaxDepth   = 8;    // SEE pruning applies at depth <= this
constexpr int   kSeeQuietMargin    = 70;   // quiets must not lose more than this * depth
constexpr int   kSeeCaptureMargin  = 180;  // captures likewise
constexpr int   kQsFutilityMargin  = 200;  // qsearch delta-pruning cushion
constexpr int   kAspirationDelta   = 20;   // initial aspiration half-window

// ---- Effort-based time scaling (v1.4) ----
// In a clock game, an obvious move does not deserve the whole budget. The
// measure of "obvious" is where the search spent its nodes: `root_cost_`
// records what each root move consumed, and when the best move's share of that
// total is high the alternatives refuted themselves cheaply.
//
// This replaces a score-gap test (v0.6-v1.3), which could not work: non-best
// root moves are searched with a null window, so their fail-soft value is a
// bound that hugs alpha rather than a real score. The measured gap sat at 0cp
// for the median iteration and cleared 100cp about once in 2500, and then only
// where the alternatives hung a piece or more.
//
// Above kEffortOnsetPct the soft budget is trimmed linearly, reaching
// kEffortMaxCutPct at a share of 100%. Stability is still required, so eval
// noise on one iteration cannot cut the think short.
// Equal-position depth cap (v1.6.1): stop deepening a real-game search past this
// depth once the score is within kGameCapMargin of 0 and the best move has held
// for kGameCapStable iterations. Clock games only (use_clock gate).
constexpr Depth kGameDepthCap   = 30;
constexpr Value kGameCapMargin  = 75;   // "roughly equal" band (cp)
constexpr int   kGameCapStable  = 3;

constexpr Depth kEffortMinDepth  = 12;
constexpr int   kEffortStable    = 4;
constexpr int   kEffortOnsetPct  = 95;  // no trim at or below this share
constexpr int   kEffortMaxCutPct = 60;  // strongest trim, at a 100% share

// ---- Adaptive search width (v0.7) ----
// Rather than a binary mode switch, width adapts *continuously* to position
// character: Leela's PUCT spreads visits wide when many moves have similar
// Q-values and goes deep when one dominates; the equivalent here is to shrink
// LMR reductions (widening the tree) in unclear positions and grow them on
// decisive mainlines. The main worker recomputes a width in [0, SC_BREADTH]
// after each iteration as the product of three fading signals —
//     gap closeness   (1 at best==2nd best root move .. 0 at >= 80cp apart)
//   x score closeness (1 at 0.00 .. 0 at |score| >= 200cp: decisive = narrow)
//   x middlegame-ness (1 at full material .. 0 in the endgame: endings = deep)
// — and every worker scales its pruning by it: LMR reductions shrink by
// width/256, LMP move budgets and futility margins grow by width/256. A width
// of 128 is "one full ply" of widening; 0 prunes bit-identically to v0.6.
// SC_BREADTH (the ceiling, i.e. the aggressiveness) is the sweepable knob.
#ifndef SC_BREADTH
#define SC_BREADTH 64  // max width; 0 disables. Sweep winner: 32..160 tested, peak at 64
#endif
constexpr int   kBreadthMax        = SC_BREADTH;
constexpr Value kBreadthGapRange   = 80;   // gap signal fades to 0 here (cp)
constexpr Value kBreadthScoreRange = 200;  // score signal fades to 0 here (cp)
constexpr int   kBreadthPhaseMin   = 12;   // phase signal is 0 at/below this material
constexpr int   kBreadthPhaseSpan  = 12;   // ...and saturates 12 phase points above it
constexpr Depth kBreadthMinDepth   = 6;    // trust the root gap only from this iteration on

// Late-move-reduction table, indexed [depth][move number]; log-shaped.
// Built once at startup.
const auto kLmr = [] {
    std::array<std::array<std::uint8_t, 64>, 64> t{};
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            t[d][m] = static_cast<std::uint8_t>(0.77 + std::log(d) * std::log(m) / 2.36);
    return t;
}();

// Convert a TT-stored (root-relative) mate score back to node-relative at `ply`.
[[nodiscard]] Value tt_value_from(std::int16_t stored, int ply) noexcept {
    Value v = stored;
    if (v >= VALUE_MATE_IN_MAX_PLY) return v - ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v + ply;
    return v;
}

[[nodiscard]] bool bound_covers(Bound b, Value v, Value threshold) noexcept {
    // Does bound `b` on value `v` prove v >= / <= threshold as appropriate?
    return static_cast<int>(b) &
           static_cast<int>(v >= threshold ? Bound::LOWER : Bound::UPPER);
}

[[nodiscard]] bool has_non_pawn_material(const Board& board, Color stm) noexcept {
    return !(board.pieces(PieceType::KNIGHT, stm) | board.pieces(PieceType::BISHOP, stm) |
             board.pieces(PieceType::ROOK, stm) | board.pieces(PieceType::QUEEN, stm))
                .empty();
}

[[nodiscard]] bool is_quiet(const Board& board, Move m) noexcept {
    return !board.isCapture(m) && m.typeOf() != Move::PROMOTION;
}

}  // namespace

// ---- Pool lifecycle -------------------------------------------------------------

Search::~Search() {
    stop();
    wait();
}

void Search::set_threads(int n) {
    stop();
    wait();

    n = std::clamp(n, 1, kMaxThreads);
    workers_.clear();
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) workers_.push_back(std::make_unique<Worker>(*this, i));
}

void Search::start(const Board& root, const SearchLimits& limits) {
    stop();
    wait();

    root_       = root;
    limits_     = limits;
    // Game ply (0-based plies from the start), so the time manager can spend
    // less early and more in the middlegame.
    const int game_ply = 2 * (static_cast<int>(root.fullMoveNumber()) - 1) +
                         (root.sideToMove() == Color::BLACK ? 1 : 0);
    budget_     = compute_budget(limits, root.sideToMove(), game_ply);
    start_time_ = now();

    for (auto& w : workers_) w->new_search();
    tt_.new_search();

    width_.store(0, std::memory_order_relaxed);  // every search opens at full depth-focus
    pondering_.store(limits.ponder, std::memory_order_release);  // search free until ponderhit
    stop_.store(false, std::memory_order_release);
    searching_.store(true, std::memory_order_release);

    threads_.reserve(workers_.size());
    for (auto& w : workers_) threads_.emplace_back([worker = w.get()] { worker->think(); });
}

void Search::stop() { stop_.store(true, std::memory_order_release); }

void Search::ponderhit() {
    // Pondering ran on the opponent's clock (free). The engine's own clock only
    // starts now, so reset the reference point: the move budget is measured from
    // ponderhit, giving a full allocation of clean, completed iterations (the
    // warm TT from pondering carries over, so it starts deep). Set the time
    // before clearing the flag so a racing should_stop() never sees the old one.
    start_time_ = now();
    pondering_.store(false, std::memory_order_release);
}

void Search::wait() {
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

void Search::new_game() {
    stop();
    wait();
    for (auto& w : workers_) w->history_.clear();
}

std::uint64_t Search::total_nodes() const {
    std::uint64_t n = 0;
    for (const auto& w : workers_) n += w->nodes_.load(std::memory_order_relaxed);
    return n;
}

// ---- Worker lifecycle -------------------------------------------------------------

void Worker::new_search() {
    nodes_.store(0, std::memory_order_relaxed);
    seldepth_    = 0;
    completed_   = 0;
    best_move_   = Move(Move::NO_MOVE);
    ponder_move_ = Move(Move::NO_MOVE);
    root_second_ = -VALUE_INFINITE;
    root_n_      = 0;
    pv_len_.fill(0);
}

bool Worker::should_stop() {
    if (pool_.stop_.load(std::memory_order_relaxed)) return true;

    // Always let the first iteration finish so we can return a real move.
    if (completed_ < 1) return false;

    if ((nodes_.load(std::memory_order_relaxed) & 1023) == 0) {
        // Only the main worker enforces the shared limits; helpers run until
        // it raises the stop flag.
        if (id_ != 0) return false;

        if (pool_.limits_.nodes && pool_.total_nodes() >= pool_.limits_.nodes) {
            pool_.stop_.store(true, std::memory_order_relaxed);
            return true;
        }
        if (pool_.budget_.use_clock && !pool_.pondering_.load(std::memory_order_relaxed) &&
            elapsed_ms(pool_.start_time_) >= pool_.budget_.hard_ms) {
            pool_.stop_.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

// ---- Quiescence ----------------------------------------------------------------

Value Worker::qsearch(Board& board, Stack* ss, Value alpha, Value beta) {
    const bool pv_node = beta - alpha > 1;

    pv_len_[ss->ply] = 0;

    if (should_stop()) return VALUE_ZERO;

    // Require a true threefold (isRepetition(2) == two prior occurrences); see
    // the note at the equivalent check in search().
    if (board.isRepetition(2) || board.isHalfMoveDraw() || board.isInsufficientMaterial())
        return VALUE_DRAW;

    const bool in_check = board.inCheck();
    if (ss->ply >= MAX_PLY) return in_check ? VALUE_DRAW : eval::evaluate(board);

    nodes_.fetch_add(1, std::memory_order_relaxed);
    seldepth_ = std::max(seldepth_, ss->ply);

    (ss + 1)->ply = ss->ply + 1;

    // TT probe: qsearch entries are stored at depth 0, so any hit qualifies.
    const Key     key   = board.hash();
    const TTProbe probe = pool_.tt_.probe(key);
    Move          tt_move  = Move(Move::NO_MOVE);
    Value         tt_value = VALUE_NONE;
    Value         tt_eval  = VALUE_NONE;
    Bound         tt_bound = Bound::NONE;
    if (probe.hit) {
        tt_move  = Move(probe.entry->move16);
        tt_value = tt_value_from(probe.entry->value, ss->ply);
        tt_eval  = probe.entry->eval;
        tt_bound = probe.entry->bound();

        if (!pv_node && bound_covers(tt_bound, tt_value, beta)) return tt_value;
    }

    // Stand pat: when not in check the side to move may simply decline to
    // continue the tactical sequence.
    Value best, raw_eval = VALUE_NONE, futility_base = -VALUE_INFINITE;
    if (in_check) {
        best = -VALUE_INFINITE;
    } else {
        raw_eval = (probe.hit && tt_eval != VALUE_NONE) ? tt_eval : eval::evaluate(board);
        best     = raw_eval;
        // A TT value with the right bound is a tighter estimate than raw eval.
        if (probe.hit && tt_value != VALUE_NONE && bound_covers(tt_bound, tt_value, best))
            best = tt_value;

        if (best >= beta) return best;
        if (best > alpha) alpha = best;

        futility_base = best + kQsFutilityMargin;
    }
    ss->static_eval = raw_eval;

    OrderingContext ctx;
    ctx.tt_move     = tt_move;
    ctx.stm         = static_cast<int>(board.sideToMove());
    ctx.prev1_piece = (ss - 1)->moved_piece;
    ctx.prev1_to    = (ss - 1)->moved_to;
    ctx.prev2_piece = (ss - 2)->moved_piece;
    ctx.prev2_to    = (ss - 2)->moved_to;

    MovePicker picker(board, history_, ctx, /*captures_only=*/true);

    Move best_move  = Move(Move::NO_MOVE);
    int  move_count = 0;
    Move m;
    while ((m = picker.next(false)) != Move(Move::NO_MOVE)) {
        ++move_count;

        if (best > VALUE_MATED_IN_MAX_PLY) {
            if (!in_check) {
                // Delta pruning: even capturing this victim for free can't
                // reach alpha, so don't bother searching it.
                if (m.typeOf() != Move::PROMOTION &&
                    futility_base + see::value(board.getCapturing<PieceType>(m)) <= alpha) {
                    best = std::max(best,
                                    futility_base + see::value(board.getCapturing<PieceType>(m)));
                    continue;
                }
                // Losing captures are almost never the refutation at the horizon.
                if (!see::see_ge(board, m, 0)) continue;
            } else if (move_count > 2 && is_quiet(board, m)) {
                // Cap the quiet-evasion explosion once a non-mating line exists.
                break;
            }
        }

        ss->current_move = m;
        ss->moved_piece  = static_cast<int>(board.at(m.from()));
        ss->moved_to     = m.to().index();

        nnue::acc_make(board, m);
        board.makeMove(m);
        pool_.tt_.prefetch(board.hash());
        const Value score = -qsearch(board, ss + 1, -beta, -alpha);
        board.unmakeMove(m);
        nnue::acc_unmake();

        if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

        if (score > best) {
            best = score;
            if (score > alpha) {
                best_move = m;
                if (score >= beta) break;
                alpha = score;
            }
        }
    }

    if (in_check && move_count == 0) return mated_in(ss->ply);

    pool_.tt_.store(key, best, raw_eval, best >= beta ? Bound::LOWER : Bound::UPPER, 0,
                    best_move, ss->ply, pv_node);
    return best;
}

// ---- Main search ----------------------------------------------------------------

Value Worker::negamax(Board& board, Stack* ss, Depth depth, Value alpha, Value beta,
                      bool cut_node) {
    const bool pv_node = beta - alpha > 1;
    const bool root    = ss->ply == 0;

    pv_len_[ss->ply] = 0;

    if (depth <= 0) return qsearch(board, ss, alpha, beta);

    if (should_stop()) return VALUE_ZERO;

    if (!root) {
        // Threefold, not a single repeat. isRepetition(1) treated ANY 2-fold as
        // a draw, including one whose earlier occurrence is in pre-root GAME
        // history — which is not a forced draw: the winning side just declines
        // it. That made a losing engine score such a line 0.00 and shuffle
        // toward a "forced draw" it couldn't hold; the moment the opponent
        // stepped out of the repetition the eval fell back to losing. Requiring
        // a genuine threefold (two prior occurrences) removes the phantom draw.
        if (board.isRepetition(2) || board.isHalfMoveDraw() || board.isInsufficientMaterial())
            return VALUE_DRAW;
        if (ss->ply >= MAX_PLY) return board.inCheck() ? VALUE_DRAW : eval::evaluate(board);

        // Mate distance pruning: the window can't contain mates longer than
        // one we've already proven from the root.
        alpha = std::max(alpha, mated_in(ss->ply));
        beta  = std::min(beta, mate_in(ss->ply + 1));
        if (alpha >= beta) return alpha;
    }

    nodes_.fetch_add(1, std::memory_order_relaxed);
    seldepth_ = std::max(seldepth_, ss->ply);

    const bool in_check = board.inCheck();
    const Move excluded = ss->excluded;

    (ss + 1)->ply        = ss->ply + 1;
    (ss + 1)->excluded   = Move(Move::NO_MOVE);
    (ss + 2)->killers[0] = (ss + 2)->killers[1] = Move(Move::NO_MOVE);

    // ---- Transposition table ----
    // In a singular verification search the stored entry describes the node
    // *with* the excluded move available, so it must not be consulted at all.
    const Key     key   = board.hash();
    const TTProbe probe = (excluded != Move(Move::NO_MOVE)) ? TTProbe{} : pool_.tt_.probe(key);

    Move  tt_move  = Move(Move::NO_MOVE);
    Value tt_value = VALUE_NONE;
    Value tt_eval  = VALUE_NONE;
    Depth tt_depth = 0;
    Bound tt_bound = Bound::NONE;
    if (probe.hit) {
        tt_move  = Move(probe.entry->move16);
        tt_value = tt_value_from(probe.entry->value, ss->ply);
        tt_eval  = probe.entry->eval;
        tt_depth = probe.entry->depth;
        tt_bound = probe.entry->bound();

        if (!pv_node && tt_depth >= depth && bound_covers(tt_bound, tt_value, beta))
            return tt_value;
    }

    // ---- Static evaluation ----
    const int stm      = static_cast<int>(board.sideToMove());
    const int corr_idx = pawn_corr_index(board);
    Value raw_eval = VALUE_NONE;  // pure static eval — this is what goes into the TT
    Value eval     = VALUE_NONE;  // corrected + possibly TT-sharpened — drives pruning
    if (!in_check) {
        raw_eval = (probe.hit && tt_eval != VALUE_NONE) ? tt_eval : eval::evaluate(board);
        // Fold in the learned per-pawn-structure correction. The corrected value
        // is what `improving` and the pruning heuristics see, and what the update
        // at the node's tail measures its residual against — a feedback loop that
        // self-limits as the bias is absorbed. The raw eval still goes to the TT,
        // so a later probe re-derives the correction from the current tables.
        const int corr = SC_CORRHIST ? history_.corr_pawn[stm][corr_idx] / CORR_GRAIN : 0;
        eval = static_cast<Value>(std::clamp(raw_eval + corr, -(VALUE_MATE_IN_MAX_PLY - 1),
                                             VALUE_MATE_IN_MAX_PLY - 1));
        ss->static_eval = eval;
        if (probe.hit && tt_value != VALUE_NONE && bound_covers(tt_bound, tt_value, eval))
            eval = tt_value;
    } else {
        ss->static_eval = VALUE_NONE;
    }

    // Is the static eval better than two plies ago? Loosens pruning when our
    // position is trending up, tightens it when trending down.
    const bool improving = !in_check && (ss - 2)->static_eval != VALUE_NONE &&
                           ss->static_eval > (ss - 2)->static_eval;

    // ---- Internal iterative reduction ----
    // No TT move at a node that matters means the previous search of this node
    // was shallow or absent; a reduced search will populate one cheaply.
    if (!in_check && depth >= kIIRMinDepth && tt_move == Move(Move::NO_MOVE) &&
        (pv_node || cut_node))
        --depth;

    if (!pv_node && !in_check && excluded == Move(Move::NO_MOVE)) {
        // ---- Razoring ----
        // Hopelessly below alpha at low depth: verify with qsearch and give up.
        if (depth <= kRazorMaxDepth && eval + kRazorMargin * depth < alpha) {
            const Value v = qsearch(board, ss, alpha - 1, alpha);
            if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            if (v < alpha && !is_mate_score(v)) return v;
        }

        // ---- Reverse futility pruning (static null move) ----
        // So far above beta that a real search is a formality.
        if (depth <= kRFPMaxDepth && !is_mate_score(eval) &&
            eval - kRFPMargin * (depth - improving) >= beta)
            return (eval + beta) / 2;

        // ---- Null move pruning ----
        // Hand the opponent a free move; if we still beat beta the position is
        // almost certainly a cutoff. Disabled without non-pawn material
        // (zugzwang) and never twice in a row.
        if (depth >= kNMPMinDepth && eval >= beta &&
            (ss - 1)->current_move != Move(Move::NULL_MOVE) &&
            has_non_pawn_material(board, board.sideToMove())) {
            const Depth R = 4 + depth / 4 + std::min(3, (eval - beta) / 200);

            ss->current_move = Move(Move::NULL_MOVE);
            ss->moved_piece  = 12;
            ss->moved_to     = 0;

            nnue::acc_make_null();
            board.makeNullMove();
            const Value v = -negamax(board, ss + 1, depth - R, -beta, -beta + 1, !cut_node);
            board.unmakeNullMove();
            nnue::acc_unmake_null();

            if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            // Never return unproven mate scores from a null search.
            if (v >= beta) return is_mate_score(v) ? beta : v;
        }

        // ---- ProbCut ----
        // If a good capture beats beta by a wide margin at reduced depth, the
        // full-depth search will almost certainly beat beta too.
        const Value probcut_beta = beta + kProbCutMargin;
        if (depth >= kProbCutMinDepth && !is_mate_score(beta) &&
            !(probe.hit && tt_depth >= depth - 3 && tt_value < probcut_beta)) {
            OrderingContext pc_ctx;
            pc_ctx.tt_move = tt_move;
            pc_ctx.stm     = static_cast<int>(board.sideToMove());

            MovePicker pc_picker(board, history_, pc_ctx, /*captures_only=*/true);
            Move       pm;
            while ((pm = pc_picker.next(true)) != Move(Move::NO_MOVE)) {
                // Only captures that win enough material to plausibly clear the bar.
                if (!see::see_ge(board, pm, probcut_beta - ss->static_eval)) continue;

                ss->current_move = pm;
                ss->moved_piece  = static_cast<int>(board.at(pm.from()));
                ss->moved_to     = pm.to().index();

                nnue::acc_make(board, pm);
                board.makeMove(pm);
                pool_.tt_.prefetch(board.hash());
                Value v = -qsearch(board, ss + 1, -probcut_beta, -probcut_beta + 1);
                if (v >= probcut_beta && !pool_.stop_.load(std::memory_order_relaxed))
                    v = -negamax(board, ss + 1, depth - 4, -probcut_beta, -probcut_beta + 1,
                                 !cut_node);
                board.unmakeMove(pm);
                nnue::acc_unmake();

                if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
                if (v >= probcut_beta) {
                    pool_.tt_.store(key, v, raw_eval, Bound::LOWER, depth - 3, pm, ss->ply, pv_node);
                    return v;
                }
            }
        }
    }

    // ---- Move loop ----
    OrderingContext ctx;
    ctx.tt_move     = tt_move;
    ctx.killer0     = ss->killers[0];
    ctx.killer1     = ss->killers[1];
    ctx.stm         = static_cast<int>(board.sideToMove());
    ctx.prev1_piece = (ss - 1)->moved_piece;
    ctx.prev1_to    = (ss - 1)->moved_to;
    ctx.prev2_piece = (ss - 2)->moved_piece;
    ctx.prev2_to    = (ss - 2)->moved_to;
    if (ctx.prev1_piece < 12) ctx.counter = history_.counter[ctx.prev1_piece][ctx.prev1_to];

    MovePicker picker(board, history_, ctx, /*captures_only=*/false);

    Value best        = -VALUE_INFINITE;
    Move  best_move   = Move(Move::NO_MOVE);
    int   move_count  = 0;
    bool  skip_quiets = false;

    if (root) root_second_ = -VALUE_INFINITE;  // reset per-iteration 2nd-best tracker
    if (root) root_n_      = 0;                // reset per-iteration effort tracker

    // Moves actually searched, for history maluses after a cutoff.
    Move tried_quiets[64];
    Move tried_caps[32];
    int  n_quiets = 0, n_caps = 0;

    Move m;
    while ((m = picker.next(skip_quiets)) != Move(Move::NO_MOVE)) {
        if (m == excluded) continue;

        const bool quiet       = is_quiet(board, m);
        const bool capture     = board.isCapture(m);
        const bool gives_check = board.givesCheck(m) != chess::CheckType::NO_CHECK;

        ++move_count;

        // ---- Per-move pruning (only once a real best exists) ----
        if (!root && best > VALUE_MATED_IN_MAX_PLY) {
            if (quiet) {
                // Late move pruning: past this move count, quiets are noise.
                // Width admits proportionally more quiets before cutting off.
                int lmp_limit = (3 + depth * depth) / (2 - improving);
                lmp_limit += lmp_limit * width_ / 256;
                if (move_count >= lmp_limit) skip_quiets = true;

                // Futility: static eval so far below alpha that a quiet move
                // has no realistic chance of raising it. Width demands a
                // proportionally deeper deficit before giving up on quiets.
                int fut_margin = kFutilityBase + kFutilityPerDepth * depth;
                fut_margin += fut_margin * width_ / 256;
                if (!in_check && !gives_check && depth <= kFutilityMaxDepth &&
                    ss->static_eval + fut_margin <= alpha)
                    skip_quiets = true;

                if (skip_quiets) continue;

                // SEE: skip quiets that lose material badly (walking into a pawn).
                if (depth <= kSeeGateMaxDepth &&
                    !see::see_ge(board, m, -kSeeQuietMargin * depth))
                    continue;
            } else {
                // SEE: skip clearly losing captures at shallow depth.
                if (depth <= kSeeGateMaxDepth &&
                    !see::see_ge(board, m, -kSeeCaptureMargin * depth))
                    continue;
            }
        }

        // ---- Singular extension ----
        // If the TT move is far better than everything else (proved by a
        // reduced search that excludes it), extend it; if even the rest of the
        // moves beat beta, the node is a multicut fail-high.
        Depth extension = 0;
        if (!root && depth >= kSingularMinDepth && m == tt_move &&
            excluded == Move(Move::NO_MOVE) && probe.hit && !is_mate_score(tt_value) &&
            (static_cast<int>(tt_bound) & static_cast<int>(Bound::LOWER)) &&
            tt_depth >= depth - 3) {
            const Value sing_beta  = tt_value - 2 * depth;
            const Depth sing_depth = (depth - 1) / 2;

            ss->excluded  = m;
            const Value v = negamax(board, ss, sing_depth, sing_beta - 1, sing_beta, cut_node);
            ss->excluded  = Move(Move::NO_MOVE);

            if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

            if (v < sing_beta) extension = 1;         // truly singular: look deeper
            else if (sing_beta >= beta) return sing_beta;  // multicut
            else if (tt_value >= beta) extension = -2;     // negative extension
            else if (cut_node) extension = -1;
        }

        const Depth new_depth = depth - 1 + extension;

        ss->current_move = m;
        ss->moved_piece  = static_cast<int>(board.at(m.from()));
        ss->moved_to     = m.to().index();

        const std::uint64_t nodes_before =
            root ? nodes_.load(std::memory_order_relaxed) : 0;

        nnue::acc_make(board, m);
        board.makeMove(m);
        pool_.tt_.prefetch(board.hash());

        // ---- PVS + late move reductions ----
        Value score = -VALUE_INFINITE;

        if (depth >= 2 && move_count > 1 + (root ? 1 : 0)) {
            int r = kLmr[std::min<int>(depth, 63)][std::min(move_count, 63)];
            r += !improving;
            r += 2 * cut_node;
            r -= pv_node;
            r -= gives_check;
            if (!quiet)
                r -= 1;  // reduce tactical moves less
            else
                r -= std::clamp(picker.last_score() / 8192, -2, 2);  // history-informed

            // Adaptive width: shrink positive reductions in proportion to the
            // current width (continuous widening; never scales extensions,
            // only reductions).
            if (r > 0) r = r * (256 - width_) / 256;

            const Depth d = std::clamp<Depth>(new_depth - r, 1, new_depth);

            score = -negamax(board, ss + 1, d, -alpha - 1, -alpha, true);
            if (score > alpha && d < new_depth)
                score = -negamax(board, ss + 1, new_depth, -alpha - 1, -alpha, !cut_node);
        } else if (!pv_node || move_count > 1) {
            score = -negamax(board, ss + 1, new_depth, -alpha - 1, -alpha, !cut_node);
        }

        // Full-window search: first move of a PV node, or a scout fail-high
        // that needs an exact score.
        if (pv_node && (move_count == 1 || (score > alpha && (root || score < beta))))
            score = -negamax(board, ss + 1, new_depth, -beta, -alpha, false);

        board.unmakeMove(m);
        nnue::acc_unmake();

        if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;

        // Self-play data-gen: perturb this root move's score by a small per-move
        // bonus (fixed for this search) so near-equal moves get chosen sometimes.
        if (root) {
            const int rn = g_root_noise.load(std::memory_order_relaxed);
            if (rn > 0 && std::abs(score) < VALUE_MATE_IN_MAX_PLY)
                score += root_noise_offset(m, rn);
        }

        if (quiet && n_quiets < 64)
            tried_quiets[n_quiets++] = m;
        else if (capture && n_caps < 32)
            tried_caps[n_caps++] = m;

        // Charge this move's subtree to it, for the time manager's effort share.
        if (root && root_n_ < MAX_MOVES) {
            root_mv_[root_n_]   = m;
            root_cost_[root_n_] = nodes_.load(std::memory_order_relaxed) - nodes_before;
            ++root_n_;
        }

        // Track the best score among non-best root moves (fail-soft, so it's a
        // true upper bound on each): when a new best appears the old best is
        // demoted to second; otherwise this move itself is a second candidate.
        if (root) {
            if (score > best) {
                if (best > root_second_) root_second_ = best;
            } else if (score > root_second_) {
                root_second_ = score;
            }
        }

        if (score > best) {
            best = score;

            if (score > alpha) {
                best_move = m;

                if (pv_node) {
                    // Record the PV: this move followed by the child's PV.
                    pv_[ss->ply][0] = m;
                    std::copy(pv_[ss->ply + 1].begin(),
                              pv_[ss->ply + 1].begin() + pv_len_[ss->ply + 1],
                              pv_[ss->ply].begin() + 1);
                    pv_len_[ss->ply] = pv_len_[ss->ply + 1] + 1;
                }

                if (root) best_move_ = m;

                if (score >= beta) break;  // fail-high
                alpha = score;
            }
        }
    }

    // ---- Terminal positions ----
    if (move_count == 0) {
        // With an excluded move this means the TT move was the only legal move:
        // report a fail-low so the exclusion search sees it as singular.
        best = (excluded != Move(Move::NO_MOVE)) ? alpha
               : in_check                        ? mated_in(ss->ply)
                                                 : VALUE_DRAW;
    } else if (best >= beta) {
        update_stats(board, ss, best_move, depth, tried_quiets, n_quiets, tried_caps, n_caps);
    }

    if (excluded == Move(Move::NO_MOVE)) {
        const Bound bound = best >= beta                             ? Bound::LOWER
                            : (pv_node && best_move != Move(Move::NO_MOVE)) ? Bound::EXACT
                                                                            : Bound::UPPER;
        pool_.tt_.store(key, best, raw_eval, bound, depth, best_move, ss->ply, pv_node);

        if constexpr (SC_CORRHIST) {
            // Teach the pawn-structure bucket the gap between the corrected static
            // eval and the score search actually returned — but only from quiet,
            // non-mate outcomes whose direction the node's bound agrees with, so
            // tactics and material swings can't pollute a positional correction.
            const bool have_best = best_move != Move(Move::NO_MOVE);
            const bool best_cap  = have_best && board.isCapture(best_move);
            if (!in_check && !best_cap && ss->static_eval != VALUE_NONE &&
                !is_mate_score(best) && (best > ss->static_eval) == have_best) {
                const int bonus = std::clamp((best - ss->static_eval) * depth *
                                                 (have_best ? 12 : 18) / 128,
                                             -HISTORY_MAX / 4, HISTORY_MAX / 4);
                history_update(history_.corr_pawn[stm][corr_idx], bonus);
            }
        }
    }

    return best;
}

// ---- History bookkeeping ----------------------------------------------------------

void Worker::update_stats(const Board& board, Stack* ss, Move best_move, Depth depth,
                          const Move* quiets, int quiet_count, const Move* captures,
                          int capture_count) {
    const int bonus = std::min(160 * depth - 90, 1700);
    const int stm   = static_cast<int>(board.sideToMove());

    // Bump one quiet move's butterfly + continuation entries by `b`.
    auto bump_quiet = [&](Move m, int b) {
        history_update(history_.main[stm][m.from().index()][m.to().index()], b);
        const int piece = static_cast<int>(board.at(m.from()));
        for (int off = 1; off <= 2; ++off) {
            const Stack* prev = ss - off;
            if (prev->moved_piece < 12)
                history_update(history_.cont_entry(prev->moved_piece, prev->moved_to, piece,
                                                   m.to().index()),
                               b);
        }
    };

    auto bump_capture = [&](Move m, int b) {
        const int moved  = static_cast<int>(board.at(m.from()));
        const int victim = static_cast<int>(board.getCapturing<PieceType>(m));
        history_update(history_.capture[moved][m.to().index()][victim], b);
    };

    if (is_quiet(board, best_move)) {
        // Killers: promote to first slot, shifting the previous killer down.
        if (ss->killers[0] != best_move) {
            ss->killers[1] = ss->killers[0];
            ss->killers[0] = best_move;
        }
        // Countermove: refutation of whatever the opponent just played.
        if ((ss - 1)->moved_piece < 12)
            history_.counter[(ss - 1)->moved_piece][(ss - 1)->moved_to] = best_move;

        bump_quiet(best_move, bonus);
        for (int i = 0; i < quiet_count; ++i)
            if (quiets[i] != best_move) bump_quiet(quiets[i], -bonus);
    } else if (board.isCapture(best_move)) {
        bump_capture(best_move, bonus);
    }

    // Captures that were tried before the cutoff move get a malus either way.
    for (int i = 0; i < capture_count; ++i)
        if (captures[i] != best_move) bump_capture(captures[i], -bonus);
}

// ---- Iterative deepening ----------------------------------------------------------

void Worker::think() {
    const bool main_worker = (id_ == 0);

    Board board = pool_.root_;  // search mutates via make/unmake, so work on a copy

    Movelist root_moves;
    chess::movegen::legalmoves(root_moves, board);

    if (root_moves.empty()) {
        // Mated or stalemated root: the protocol still expects a bestmove line.
        if (main_worker) {
            if (!g_gen_silent) std::cout << "bestmove 0000" << std::endl;
            best_move_ = Move(Move::NO_MOVE);  // signal terminal position to gengame
            pool_.stop_.store(true, std::memory_order_release);
            pool_.searching_.store(false, std::memory_order_release);
        }
        return;
    }
    best_move_ = root_moves[0];  // guaranteed fallback

    // Fresh root-noise seed per search: fixed offsets within this search's
    // deepening, different across searches so repeated self-play games diverge.
    t_noise_seed = t_noise_rng();

    // The +8 pads (ss-2) history probes at the root and (ss+2) killer clears at
    // the tips; slot 4 is ply 0.
    std::vector<Stack> stack(MAX_PLY + 8);
    Stack*             ss = stack.data() + 4;

    const Depth max_depth =
        (pool_.limits_.depth > 0) ? std::min<Depth>(pool_.limits_.depth, MAX_PLY - 1)
                                  : MAX_PLY - 1;

    // Root material phase (24 == full board), for the breadth-mode endgame guard.
    const int root_phase =
        (board.pieces(PieceType::KNIGHT).count() + board.pieces(PieceType::BISHOP).count()) +
        2 * board.pieces(PieceType::ROOK).count() + 4 * board.pieces(PieceType::QUEEN).count();

    Value prev_score = VALUE_NONE;
    Move  prev_best  = Move(Move::NO_MOVE);
    int   stable     = 0;  // consecutive completed iterations with the same best move

    for (Depth d = 1; d <= max_depth; ++d) {
        // Lazy SMP depth staggering: odd-numbered helpers skip even depths, so
        // half the pool runs ahead and seeds the shared TT from above while
        // the rest (and the main worker) fill it in order.
        if (!main_worker && (id_ & 1) && (d & 1) == 0 && d < max_depth) continue;

        nnue::acc_reset(board);  // fresh incremental base accumulator at the root

        width_    = pool_.width_.load(std::memory_order_relaxed);  // stable per iteration
        seldepth_ = 0;

        // ---- Aspiration windows ----
        // Search a narrow window around the previous score, widening
        // geometrically on failure. Small windows produce far more cutoffs.
        Value delta = kAspirationDelta;
        Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
        if (d >= 4 && prev_score != VALUE_NONE) {
            alpha = std::max<Value>(prev_score - delta, -VALUE_INFINITE);
            beta  = std::min<Value>(prev_score + delta, VALUE_INFINITE);
        }

        Value score;
        while (true) {
            score = negamax(board, ss, d, alpha, beta, false);
            if (pool_.stop_.load(std::memory_order_relaxed)) break;

            if (score <= alpha) {  // fail-low: drop alpha, pull beta toward it
                beta  = (alpha + beta) / 2;
                alpha = std::max<Value>(score - delta, -VALUE_INFINITE);
            } else if (score >= beta) {  // fail-high: raise beta
                beta = std::min<Value>(score + delta, VALUE_INFINITE);
            } else {
                break;
            }
            delta += delta / 2;
        }

        if (pool_.stop_.load(std::memory_order_relaxed)) break;

        completed_ = d;
        prev_score = score;

        if (!main_worker) continue;  // helpers never report or manage time

        report(d, score);

        // The 2nd move of the root PV is the reply we expect; offer it as the
        // ponder move so the GUI can search it on the opponent's clock.
        ponder_move_ = (pv_len_[0] >= 2) ? pv_[0][1] : Move(Move::NO_MOVE);

        // Best-move stability across completed iterations (for the only-move exit).
        stable    = (best_move_ == prev_best) ? stable + 1 : 0;
        prev_best = best_move_;

        // ---- Search width for the next iteration ----
        if constexpr (kBreadthMax > 0) {
            int width = 0;
            if (d >= kBreadthMinDepth && !is_mate_score(score) &&
                root_second_ > -VALUE_INFINITE) {
                const int wg = std::max<Value>(0, kBreadthGapRange -
                                                      std::max<Value>(0, score - root_second_));
                const int ws = std::max<Value>(0, kBreadthScoreRange - std::abs(score));
                const int wp = std::clamp(root_phase - kBreadthPhaseMin, 0, kBreadthPhaseSpan);
                width        = kBreadthMax * wg * ws * wp /
                        (kBreadthGapRange * kBreadthScoreRange * kBreadthPhaseSpan);
            }
            const int old = pool_.width_.load(std::memory_order_relaxed);
            if (width != old) {
                pool_.width_.store(width, std::memory_order_relaxed);
                // Log only meaningful shifts, not every wobble.
                if (!g_gen_silent && ((width == 0) != (old == 0) || std::abs(width - old) >= 32))
                    std::cout << "info string search width " << width << "/" << kBreadthMax
                              << std::endl;
            }
        }

        // ---- Between-iteration stop conditions (main worker only) ----
        if (pool_.limits_.nodes && pool_.total_nodes() >= pool_.limits_.nodes) break;

        // Convergence stops — active even while pondering. Once a mate is proven
        // or only one move is legal, deeper search is pointless. Firing these
        // during a ponder search is essential: otherwise a mate found on the
        // opponent's clock spins the iteration counter to absurd depths (each
        // iteration a trivial TT hit), and the post-ponderhit search then can't
        // finish a single (now enormous) iteration within budget — so it runs
        // the full clock and reports no PV. Gated on use_clock, so fixed-depth
        // and analysis (go infinite / go depth) searches are unaffected.
        if (pool_.budget_.use_clock) {
            if (root_moves.size() == 1 && d >= 6) {
                break;                                       // forced move: nothing to choose
            } else if (is_mate_score(score)) {
                // Stop only once the mate is proven SHORTEST at this depth. A mate
                // whose distance still exceeds the searched depth is a TT-injected long
                // mate from an earlier search: breaking on it plays a "mates eventually"
                // move and abandons the rest of the budget instead of deepening to the
                // quickest mate. Requiring d >= distance still caps ponder spins (it
                // fires the moment the mate is genuinely proven, not on a stale TT hit).
                const int mate_plies = (score > 0) ? (VALUE_MATE - score) : (VALUE_MATE + score);
                if (d >= mate_plies) break;                  // shortest mate proven: play it
            }
        }

        // Time-based stops: only when the clock is actually ours (not pondering).
        if (pool_.budget_.use_clock && !pool_.pondering_.load(std::memory_order_relaxed)) {
            // Trim the soft budget when the best move is obvious. Only for a
            // real game clock, not a fixed `movetime` (there is no clock to
            // save, so we honor the full think).
            std::int64_t soft = pool_.budget_.soft_ms;
            if (pool_.limits_.movetime == 0 && d >= kEffortMinDepth &&
                stable >= kEffortStable && !is_mate_score(score) && root_n_ > 1) {
                std::uint64_t total = 0, best_cost = 0;
                for (int i = 0; i < root_n_; ++i) {
                    total += root_cost_[i];
                    if (root_mv_[i] == best_move_) best_cost = root_cost_[i];
                }
                if (total > 0) {
                    const int share = static_cast<int>(100 * best_cost / total);
                    if (share > kEffortOnsetPct) {
                        const int cut = kEffortMaxCutPct * (share - kEffortOnsetPct) /
                                        (100 - kEffortOnsetPct);
                        soft = soft * (100 - cut) / 100;
                    }
                }
            }

            if (elapsed_ms(pool_.start_time_) >= soft) break;

            // Equal-position depth cap (v1.6.1): in a real game, don't spend
            // clock searching past kGameDepthCap once the position is roughly
            // balanced and the best move has settled — deeper search there
            // essentially never changes the move. Never applies to fixed
            // movetime/depth or analysis (use_clock is false for those), so it
            // can't affect depth-limited tests.
            if (d >= kGameDepthCap && stable >= kGameCapStable &&
                std::abs(score) <= kGameCapMargin)
                break;
        }
    }

    if (main_worker) {
        root_score_ = prev_score;  // final completed-iteration score (stm-relative), for gengame
        // If the ID loop ended on its own while still pondering (e.g. a proven
        // mate), hold the bestmove: the GUI expects it only after ponderhit or
        // stop, never while we're searching on its clock.
        while (pool_.pondering_.load(std::memory_order_acquire) &&
               !pool_.stop_.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // A time-cut search can leave the ponder move (pv[0][1]) out of sync with
        // best_move_ — a stale 2nd move from a line we no longer play, which is
        // then illegal after best_move_. Only announce it if it is genuinely a
        // legal reply. `board` is back at the root here (make/unmake is balanced).
        if (ponder_move_ != Move(Move::NO_MOVE) && best_move_ != Move(Move::NO_MOVE)) {
            Board after = board;
            after.makeMove(best_move_);
            Movelist replies;
            chess::movegen::legalmoves(replies, after);
            bool legal = false;
            for (const auto& r : replies)
                if (r == ponder_move_) { legal = true; break; }
            if (!legal) ponder_move_ = Move(Move::NO_MOVE);
        }

        if (!g_gen_silent) {
            std::cout << "bestmove " << chess::uci::moveToUci(best_move_);
            if (ponder_move_ != Move(Move::NO_MOVE))
                std::cout << " ponder " << chess::uci::moveToUci(ponder_move_);
            std::cout << std::endl;
        }

        pool_.stop_.store(true, std::memory_order_release);
        pool_.searching_.store(false, std::memory_order_release);
    }
}

// ---- Reporting ----------------------------------------------------------------

std::string Worker::pv_string() const {
    std::ostringstream ss;
    for (int i = 0; i < pv_len_[0]; ++i) {
        if (i) ss << ' ';
        ss << chess::uci::moveToUci(pv_[0][i]);
    }
    return ss.str();
}

void Worker::report(Depth depth, Value score) {
    if (g_gen_silent) return;  // gengame: suppress per-iteration info lines
    const std::int64_t  ms    = std::max<std::int64_t>(1, elapsed_ms(pool_.start_time_));
    const std::uint64_t nodes = pool_.total_nodes();
    const std::uint64_t nps   = nodes * 1000ULL / static_cast<std::uint64_t>(ms);

    std::ostringstream ss;
    ss << "info depth " << depth << " seldepth " << seldepth_ << " score ";

    if (is_mate_score(score)) {
        // UCI wants distance in moves; positive when we deliver the mate.
        const int plies      = (score > 0) ? (VALUE_MATE - score) : (VALUE_MATE + score);
        const int mate_moves = (score > 0) ? (plies + 1) / 2 : -((plies + 1) / 2);
        ss << "mate " << mate_moves;
    } else {
        // Remove the played move's root-noise bonus so the reported score (the
        // self-play training label) is the move's true eval, not the noisy one.
        const int rn = g_root_noise.load(std::memory_order_relaxed);
        const Value clean = (rn > 0) ? score - root_noise_offset(best_move_, rn) : score;
        ss << "cp " << clean;
    }

    ss << " nodes " << nodes << " nps " << nps << " time " << ms << " hashfull "
       << pool_.tt_.hashfull();

    const std::string pv = pv_string();
    if (!pv.empty()) ss << " pv " << pv;

    std::cout << ss.str() << std::endl;
}

}  // namespace engine
