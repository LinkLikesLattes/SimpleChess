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
#include <cctype>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#include "movepick.hpp"
#include "nnue.hpp"
#include "see.hpp"
#include "syzygy.hpp"
#include "io.hpp"
#include "wdl.hpp"

namespace engine {

// Root-move noise for self-play data generation. When > 0, each root move gets a
// small random +-cp bonus (reseeded implicitly as the per-thread RNG advances),
// so the engine picks among near-equal moves and repeated games diverge. Mate
// scores are never perturbed. Off (0) by default — normal play is unaffected.
namespace {
std::atomic<int>              g_root_noise{0};
std::atomic<bool>             g_show_wdl{false};     // UCI_ShowWDL: display only (setoption may arrive mid-search)
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

// ---- Pass-eval tension (SC_PASSEVAL) ----
// P = our static score if we had to pass (nnue::evaluate_pass: the same accumulator read with the
// perspectives swapped), T = raw eval - P: ~tempo in quiet positions, large when either side has a
// threat, negative when passing beats moving (zugzwang-like). tn = the excess over the quiet
// baseline PassTau, capped at PassCap, is what the plug points scale by. Every scale at 0 == that
// plug point off, so a build with all scales 0 searches identically to SC_PASSEVAL=0 and differs
// only by the cost of computing T (that is how the cost is measured). Runtime-settable through
// UCI (masters below); each worker snapshots them once per iteration into Worker::pass_.
struct PassParamInfo { const char* name; int def, lo, hi; };
constexpr PassParamInfo kPassParams[] = {
    {"PassMinDepth",  SC_PASS_MINDEPTH,  1,    20},   // compute T at main-search nodes with depth >= this (1 = every node: cost cliff)
    {"PassTau",       SC_PASS_TAU,       -200, 500},  // quiet baseline subtracted from T (diagnostic: P50 of T)
    {"PassCap",       SC_PASS_CAP,       1,    3000}, // cap on tn (diagnostic: P99 of T); >= 1 keeps std::clamp well-formed
    {"PassRazor",     SC_PASS_RAZOR,     0,    64},   // razor margin += tn * PassRazor / 16  (F1: DROPPED, AUC 0.540)
    {"PassRfp",       SC_PASS_RFP,       0,    64},   // RFP margin   += tu * PassRfp   / 16  (tu = two-sided)
    {"PassLmrDiv",    SC_PASS_LMRDIV,    0,    2000}, // r += min(2, tn / PassLmrDiv) on QUIETS only   (0 = off)
    {"PassNmpMargin", SC_PASS_NMPMARGIN, 0,    2000}, // skip the null search when P_est < beta - margin (0 = off)
    {"PassZug",       SC_PASS_ZUG,       0,    500},  // skip the null search when T < -PassZug          (0 = off)
    {"PassQsLambda",  SC_PASS_QSLAMBDA,  0,    64},   // qsearch stand-pat -= tq * PassQsLambda / 64
    {"PassTmScale",   SC_PASS_TMSCALE,   0,    64},   // soft budget: +ts/256 at tn 0 .. -ts/256 saturated, 1.0 at kPassTmMid
    {"PassCorrW",     SC_PASS_CORRW,     0,    64},   // corrhist update bonus *= 1 + tn * PassCorrW / 4096  (0 = off)
};
enum PassIdx { PASS_MIN_DEPTH, PASS_TAU, PASS_CAP, PASS_RAZOR, PASS_RFP, PASS_LMR_DIV,
               PASS_NMP_MARGIN, PASS_ZUG, PASS_QS_LAMBDA, PASS_TM_SCALE, PASS_CORR_W, PASS_N };
static_assert(sizeof(kPassParams) / sizeof(kPassParams[0]) == PASS_N, "kPassParams/PassIdx out of sync");
static_assert(PASS_N == 11, "Worker::pass_ is sized for 11 tunables");
std::atomic<int> g_pass[PASS_N];   // UCI-settable masters; workers snapshot them per iteration
const bool g_pass_init = [] { for (int i = 0; i < PASS_N; ++i) g_pass[i].store(kPassParams[i].def); return true; }();
}  // namespace

bool set_pass_param(std::string_view name, int value) {
    for (int i = 0; i < PASS_N; ++i) {
        const std::string_view n = kPassParams[i].name;
        if (n.size() == name.size() &&
            std::equal(n.begin(), n.end(), name.begin(), [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
            })) {
            g_pass[i].store(std::clamp(value, kPassParams[i].lo, kPassParams[i].hi), std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

void pass_param_options(std::ostream& out) {
    // Development knobs, not user options: advertised only in SC_PASS_UCI_OPTIONS tuning builds.
    // setoption accepts them regardless (see set_pass_param), so the match tooling can drive them.
    if constexpr (SC_PASSEVAL && SC_PASS_UCI_OPTIONS)
        for (const auto& p : kPassParams)
            out << "option name " << p.name << " type spin default " << p.def
                << " min " << p.lo << " max " << p.hi << "\n";
    else
        (void)out;
}

void set_root_noise(int cp) { g_root_noise.store(cp < 0 ? 0 : cp, std::memory_order_relaxed); }

void set_show_wdl(bool on) { g_show_wdl.store(on, std::memory_order_relaxed); }

// Set on the UCI thread before gengame/revgame's searches; the workers observe it through
// start()'s hand-off (thread creation, or the persistent pool's mutex). Those callers stop and
// join any running search first, so the flag never changes while a search is live.
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

// The tn at which the PassTmScale time apportionment leaves the budget unchanged. Started as the
// median of tn on F1's measured pool (P50(T) 117 - PassTau 45 = 72), but the tn distribution is
// skewed, so that value left the candidate using +1.89% more mean time than the control -- worth
// about 1.6 Elo of "thinking longer" at 10+0.1, which against [0,2] bounds could manufacture a
// false PASS. CALIBRATED instead: tools/verify/passeval_tmcheck.py measures mean time consumed over
// 300 phase-diverse positions under a real clock, and 88 is the value that makes the ratio 1.000.
constexpr int   kPassTmMid         = 88;
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
// character: when several root moves score close together the tree should be
// wide, and when one clearly dominates it should be deep. Concretely, shrink
// LMR reductions (widening the tree) in unclear positions and grow them on
// decisive mainlines. The main worker recomputes a width in [0, SC_BREADTH]
// after each iteration as the product of two fading signals —
//     gap closeness   (1 at best==2nd best root move .. 0 at >= 80cp apart)
//   x score closeness (1 at 0.00 .. 0 at |score| >= 200cp: decisive = narrow)
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
// The old third factor (middlegame-ness, "endings = deep") was DROPPED in 3.2:
// it forced width to 0 from ~Q+R each side on, so every endgame ran the
// narrowest tree — the strongest engines carry no material term in their
// reductions at all. See dev-notes/campaigns/search-3.2.
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

// Convert a TT-stored mate or tablebase score (distance counted from the stored node) back to the
// root-relative form the search uses at `ply`. `r50c` is the position's halfmove clock; with
// SC_R50MATE it downgrades a result the 50-move rule voids before it can be reached (see the switch
// comment in search.hpp): the stored value alone gives the distance from this node -- to the mate,
// or to the zeroing position whose probe gave the TB result -- so the result is unreachable when
// that distance exceeds 100 - r50c. r50c is unused when the switch is off.
[[nodiscard]] Value tt_value_from(std::int16_t stored, int ply, [[maybe_unused]] int r50c) noexcept {
    Value v = stored;
    if (v >= VALUE_TB_WIN_IN_MAX_PLY) {
        if constexpr (SC_R50MATE) {
            const int dist = v >= VALUE_MATE_IN_MAX_PLY ? VALUE_MATE - v : VALUE_TB - v;
            if (dist > 100 - r50c) return VALUE_TB_WIN_IN_MAX_PLY - 1;
        }
        return v - ply;
    }
    if (v <= VALUE_TB_LOSS_IN_MAX_PLY) {
        if constexpr (SC_R50MATE) {
            const int dist = v <= VALUE_MATED_IN_MAX_PLY ? VALUE_MATE + v : VALUE_TB + v;
            if (dist > 100 - r50c) return VALUE_TB_LOSS_IN_MAX_PLY + 1;
        }
        return v + ply;
    }
    return v;
}

// Repetition draw. With SC_TREE_REP off: a genuine threefold (two prior occurrences) anywhere.
// With it on: a position that already occurred inside the search tree (at or after the root) is
// a draw on its first recurrence: the side that could have deviated chose to repeat, so
// the cycle can be forced. A position whose earlier occurrences are only in the game history
// before the root still needs a genuine threefold (two prior occurrences): a single pre-root
// repeat is not a forced draw, the winning side just declines it, and scoring it 0.00 made a
// losing engine shuffle toward a "draw" it couldn't hold. The in-tree scan stops at a null move:
// a cycle through a pass is not a repetition, and no position before the pass can count.
[[nodiscard]] bool is_repetition_draw(const Board& board, [[maybe_unused]] const Stack* ss) noexcept {
    if constexpr (!SC_TREE_REP) return board.isRepetition(2);  // switch off: threefold everywhere
    const Key key   = board.hash();
    const int limit = std::min(ss->ply, static_cast<int>(board.halfMoveClock()));
    for (int k = 1; k <= limit; ++k) {
        if ((ss - k)->current_move == Move(Move::NULL_MOVE)) return false;
        if (k % 2 == 0 && (ss - k)->key == key) return true;
    }
    return board.isRepetition(2);
}

// A node whose halfmove clock has reached 100 is a 50-move draw, unless the move that reached it
// gave checkmate: mate takes precedence over the 50-move rule, so that node is mated.
[[nodiscard]] Value fifty_move_value(const Board& board, int ply) noexcept {
    if (board.inCheck() && !chess::movegen::anylegalmoves(board)) return mated_in(ply);
    return VALUE_DRAW;
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
#if !defined(__ARM_NEON)
    pool_shutdown();
#endif
}

void Search::set_threads(int n) {
    stop();
    wait();
#if !defined(__ARM_NEON)
    pool_shutdown();  // retire the old pool before resizing
#endif

    n = std::clamp(n, 1, kMaxThreads);
    workers_.clear();
    workers_.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) workers_.push_back(std::make_unique<Worker>(*this, i));
#if !defined(__ARM_NEON)
    pool_spawn();     // one long-lived thread per worker, parked on the start CV
#endif
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
    start_time_.store(now(), std::memory_order_relaxed);  // published by the release stores below

    for (auto& w : workers_) w->new_search();
    tt_.new_search();
    syzygy::reset_hits();

    width_.store(0, std::memory_order_relaxed);  // every search opens at full depth-focus
    pondering_.store(limits.ponder, std::memory_order_release);  // search free until ponderhit
    stop_.store(false, std::memory_order_release);
    searching_.store(true, std::memory_order_release);

#if defined(__ARM_NEON)
    threads_.reserve(workers_.size());   // pre-F1: one fresh std::thread per worker per `go`
    for (auto& w : workers_) threads_.emplace_back([worker = w.get()] { worker->think(); });
#else
    {
        std::lock_guard<std::mutex> lk(pool_mu_);
        pool_active_ = static_cast<int>(workers_.size());
        std::fill(pool_go_.begin(), pool_go_.end(), static_cast<char>(1));
    }
    pool_start_cv_.notify_all();   // release the parked pool; wait() blocks on completion
#endif
}

void Search::stop() { stop_.store(true, std::memory_order_release); }

void Search::ponderhit() {
    // Pondering ran on the opponent's clock (free). The engine's own clock only
    // starts now, so reset the reference point: the move budget is measured from
    // ponderhit, giving a full allocation of clean, completed iterations (the
    // warm TT from pondering carries over, so it starts deep). Set the time
    // before clearing the flag (release; readers load the flag with acquire) so
    // a racing should_stop() never sees the old one. A ponderhit that arrives
    // when no ponder search is running changes nothing: resetting a normal
    // search's clock would make it overspend.
    if (!pondering_.load(std::memory_order_acquire)) return;
    start_time_.store(now(), std::memory_order_relaxed);
    pondering_.store(false, std::memory_order_release);
}

void Search::wait() {
#if defined(__ARM_NEON)
    for (auto& t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
#else
    std::unique_lock<std::mutex> lk(pool_mu_);
    pool_done_cv_.wait(lk, [this] { return pool_active_ == 0; });
#endif
}

void Search::pool_spawn() {
    pool_quit_   = false;
    pool_active_ = 0;
    pool_go_.assign(workers_.size(), static_cast<char>(0));
    threads_.reserve(workers_.size());
    for (std::size_t i = 0; i < workers_.size(); ++i)
        threads_.emplace_back([this, i] { pool_loop(static_cast<int>(i)); });
}

void Search::pool_shutdown() noexcept {
    {
        std::lock_guard<std::mutex> lk(pool_mu_);
        pool_quit_ = true;
    }
    pool_start_cv_.notify_all();
    for (auto& th : threads_)
        if (th.joinable()) th.join();
    threads_.clear();
}

void Search::pool_loop(int id) {
    const std::size_t k = static_cast<std::size_t>(id);
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(pool_mu_);
            pool_start_cv_.wait(lk, [this, k] { return pool_go_[k] != 0 || pool_quit_; });
            if (pool_quit_) return;
            pool_go_[k] = 0;
        }
        workers_[k]->think();
        {
            std::lock_guard<std::mutex> lk(pool_mu_);
            if (--pool_active_ == 0) pool_done_cv_.notify_all();
        }
    }
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
    root_tension_ = VALUE_NONE;
    pv_idx_       = 0;
    multipv_      = 1;
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
        if (pool_.budget_.use_clock && !pool_.pondering_.load(std::memory_order_acquire) &&
            elapsed_ms(pool_.start_time_.load(std::memory_order_relaxed)) >= pool_.budget_.hard_ms) {
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

    ss->key = board.hash();
    if (is_repetition_draw(board, ss) || board.isInsufficientMaterial()) return VALUE_DRAW;
    if (board.isHalfMoveDraw()) return fifty_move_value(board, ss->ply);

    const bool in_check = board.inCheck();
    if (ss->ply >= MAX_PLY) return in_check ? VALUE_DRAW : nnue::evaluate(board);

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
        tt_value = tt_value_from(probe.entry->value, ss->ply, static_cast<int>(board.halfMoveClock()));
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
        if (probe.hit && tt_eval != VALUE_NONE) {
            raw_eval = tt_eval;  // reuse the stored static eval as-is (already damped when written)
        } else {
            raw_eval = nnue::evaluate(board);
            if constexpr (SC_R50DAMP)
                raw_eval -= raw_eval * static_cast<int>(board.halfMoveClock()) / 199;
        }
        best     = raw_eval;
        if constexpr (SC_PASSEVAL) {
            // Null-move-consistent stand-pat: P (our eval if we passed) is the conservative bound
            // on this node, raw_eval the optimistic one; lambda interpolates (0 = today). The net
            // is queried only where the penalty can change a decision: above alpha, and not so far
            // above beta that even the maximal penalty (cap * lambda / 64) leaves the cutoff intact
            // -- at a null window that band is ~cap*lambda/64 cp wide, so the cost stays bounded.
            const int lam = pass_[PASS_QS_LAMBDA];
            if (lam > 0 && raw_eval > alpha && raw_eval - pass_[PASS_CAP] * lam / 64 < beta) {
                const int t  = raw_eval - nnue::evaluate_pass(board);
                const int tq = std::clamp<int>(t - pass_[PASS_TAU], 0, pass_[PASS_CAP]);
                best = static_cast<Value>(raw_eval - tq * lam / 64);
            }
        }
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
                // The picker already evaluated see_ge(m, 0) for the captures it scored; only the
                // TT move / promotions (verdict 0) still pay for the SEE here.
#if defined(__ARM_NEON)
                if (!see::see_ge(board, m, 0)) continue;   // pre-E3: SEE recomputed here
#else
                {
                    const int ks = picker.last_see();
                    if (ks < 0 || (ks == 0 && !see::see_ge(board, m, 0))) continue;
                }
#endif
            } else if (move_count > 2 && is_quiet(board, m)) {
                // Cap the quiet-evasion explosion once a non-mating line exists.
                break;
            }
        }

        ss->current_move = m;
        ss->moved_piece  = static_cast<int>(board.at(m.from()));
        ss->moved_to     = m.to().index();

        // D3: prefetch the child's TT cluster BEFORE the NNUE update so the accumulator
        // work hides the miss (issued after makeMove it led the probe by only a few ns).
        // zobristAfter() is the library's exact post-move key for the makeMove<false>
        // the search uses (same en-passant rule); a prefetch cannot change any result.
        pool_.tt_.prefetch(board.zobristAfter(m));
        nnue::acc_make(board, m);
        board.makeMove(m);
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

    ss->key = board.hash();
    if (!root) {
        if (is_repetition_draw(board, ss) || board.isInsufficientMaterial()) return VALUE_DRAW;
        if (board.isHalfMoveDraw()) return fifty_move_value(board, ss->ply);
        if (ss->ply >= MAX_PLY) return board.inCheck() ? VALUE_DRAW : nnue::evaluate(board);

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
        tt_value = tt_value_from(probe.entry->value, ss->ply, static_cast<int>(board.halfMoveClock()));
        tt_eval  = probe.entry->eval;
        tt_depth = probe.entry->depth;
        tt_bound = probe.entry->bound();

        if (!pv_node && tt_depth >= depth && bound_covers(tt_bound, tt_value, beta))
            return tt_value;
    }

    // ---- Syzygy tablebase probe ----
    // A small, quiet node (few enough men, no castling rights, halfmove clock 0)
    // has an exact WDL verdict; a hit is a cutoff. Entirely inert unless tables
    // are loaded (probe_limit() == 0 disables the gate). Skipped in check and in a
    // singular verification search. The score is not cached in the TT: it is
    // ply-encoded for this node, and re-probing on a transposition is cheap.
    if constexpr (SC_SYZYGY) {
        // E5: with no tables probe_limit() is 0, so test it FIRST and skip the popcount
        // and the probe-depth read on every node (pure operands, identical outcome).
        const int tb_limit = syzygy::probe_limit();
        if (tb_limit > 0 && !root && !in_check && excluded == Move(Move::NO_MOVE) &&
            depth >= syzygy::probe_depth() &&
            static_cast<int>(board.occ().count()) <= tb_limit) {
            if (const std::optional<Value> tb = syzygy::probe_wdl(board, ss->ply); tb)
                return *tb;
        }
    }

    // ---- Static evaluation ----
    const int stm      = static_cast<int>(board.sideToMove());
    // E9a: both readers of corr_idx (the eval correction below and the tail update)
    // sit behind !in_check, so skip the splitmix hash on in-check nodes.
#if defined(__ARM_NEON)
    const int corr_idx = pawn_corr_index(board);   // pre-E9a: hashed on in-check nodes too
#else
    const int corr_idx = in_check ? 0 : pawn_corr_index(board);
#endif
    Value raw_eval = VALUE_NONE;  // pure static eval — this is what goes into the TT
    Value eval     = VALUE_NONE;  // corrected + possibly TT-sharpened — drives pruning
    if (!in_check) {
        if (probe.hit && tt_eval != VALUE_NONE) {
            raw_eval = tt_eval;  // reuse the stored static eval as-is (already damped when written)
        } else {
            raw_eval = nnue::evaluate(board);
            if constexpr (SC_R50DAMP)
                raw_eval -= raw_eval * static_cast<int>(board.halfMoveClock()) / 199;
        }
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
    // Razoring's qsearch and the singular verification search run on this same stack slot;
    // both put this value back afterwards.
    [[maybe_unused]] const Value eval_slot = ss->static_eval;

    // Is the static eval better than two plies ago? Loosens pruning when our
    // position is trending up, tightens it when trending down.
    const bool improving = !in_check && (ss - 2)->static_eval != VALUE_NONE &&
                           ss->static_eval > (ss - 2)->static_eval;

    // ---- Pass-eval tension (SC_PASSEVAL) ----
    // T = raw static eval minus our static eval if we had to pass. Net-internal on purpose:
    // raw_eval (fresh, or the TT's stored raw eval of this same position), not the corrected
    // eval; never in check (the passed position would be illegal). tn = excess over the quiet
    // baseline, capped: the quantity the plug points scale by. Costs pairwise + L1 + body (no
    // accumulator work), so it is computed LAZILY, at most once per node, and only when a
    // decision actually hinges on it: razoring / RFP / NMP can only be vetoed by tension, so they
    // ask for it once their untensioned condition already says "prune"; LMR asks before the
    // move loop; the time manager asks at the root. With every scale at 0 nothing is ever
    // computed and the search is byte-identical to SC_PASSEVAL=0. Gate: the node's nominal depth
    // >= PassMinDepth (IIR fires only at depth >= 4 and reduces by one).
    Value tension = VALUE_NONE;
    int   tn      = 0;
    // tu = "how over-optimistic is this static eval", the TWO-SIDED asymmetric form RFP needs.
    // F1 measured frac(R < -100) -- precisely the error reverse futility is exposed to -- by T
    // decile: it bottoms at 10.0% in the decile T in [15,45], climbs to 32.0% at T >= 418, and ALSO
    // rises to 15.2% in the T <= 15 tail, with the left slope about 2x the right per centipawn
    // (+5.2pp over ~80cp going down vs +22pp over ~670cp going up). The one-sided
    // clamp(T - tau, 0, cap) that tn uses scores that whole left tail as zero.
    int   tu      = 0;
    [[maybe_unused]] bool tension_done = false;
    auto need_tension = [&]() {
        if constexpr (SC_PASSEVAL) {
            if (tension_done) return;
            tension_done = true;
            if (in_check || depth < pass_[PASS_MIN_DEPTH]) return;
            const Value p = nnue::evaluate_pass(board);
            tension = raw_eval - p;
            tn      = std::clamp<int>(tension - pass_[PASS_TAU], 0, pass_[PASS_CAP]);
            tu      = std::clamp<int>(tension > pass_[PASS_TAU]
                                          ? tension - pass_[PASS_TAU]
                                          : 2 * (pass_[PASS_TAU] - tension),
                                      0, pass_[PASS_CAP]);
            if (root) root_tension_ = tension;
#if SC_PASSEVAL_VERIFY
            // In-search oracle (dev builds only): P must equal the eval of the actual null-moved
            // position through the incremental path, and a TT-supplied raw_eval must equal a fresh
            // query. Any mismatch is a bug, never noise (both are deterministic integer paths).
            nnue::acc_make_null();
            board.makeNullMove();
            const Value p_ref = -nnue::evaluate(board);
            board.unmakeNullMove();
            nnue::acc_unmake_null();
            if (p_ref != p)
                std::fprintf(stderr, "[passeval] P MISMATCH ply=%d p=%d ref=%d\n", ss->ply, int(p), int(p_ref));
            if (probe.hit && tt_eval != VALUE_NONE) {
                const Value fresh = nnue::evaluate(board);
                if (tt_eval != fresh)
                    std::fprintf(stderr, "[passeval] TT EVAL MISMATCH ply=%d tt=%d fresh=%d\n", ss->ply, int(tt_eval), int(fresh));
            }
#endif
        }
    };
    if constexpr (SC_PASSEVAL)
        if (root && pass_[PASS_TM_SCALE] > 0) need_tension();   // the time manager reads root_tension_

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
            // Pass-eval veto (SC_PASSEVAL): under tension the deficit must also cover
            // tn * PassRazor / 16 before the node is given up (T computed here, lazily).
            if constexpr (SC_PASSEVAL)
                if (pass_[PASS_RAZOR] > 0) need_tension();
            if (eval + kRazorMargin * depth + (SC_PASSEVAL ? tn * pass_[PASS_RAZOR] / 16 : 0) < alpha) {
                // qsearch runs on this node's own stack slot and overwrites static_eval
                // with its raw eval; restore the corrected one the rest of the node reads.
                const Value v = qsearch(board, ss, alpha - 1, alpha);
                if constexpr (SC_EVAL_SLOT) ss->static_eval = eval_slot;
                if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
                if (v < alpha && !is_mate_score(v)) return v;
            }
        }

        // ---- Reverse futility pruning (static null move) ----
        // So far above beta that a real search is a formality.
        if (depth <= kRFPMaxDepth && !is_mate_score(eval) &&
            eval - kRFPMargin * (depth - improving) >= beta) {
            // Pass-eval veto (SC_PASSEVAL): the static surplus must also cover tu * PassRfp / 16.
            // RFP is the ONE plug point F1's signed decomposition supports: it trusts that a high
            // eval means a cutoff, so it is exposed to OVER-optimism, and tension predicts exactly
            // that (AUC 0.619, frac(R < -100) climbing 10.0% -> 32.0% across T deciles). Razoring is
            // exposed to over-PESSIMISM instead, which tension barely predicts (AUC 0.540), so
            // PassRazor stays 0: E1 scaled both at 8/8 and about half of what it paid bought noise.
            // Uses tu, the two-sided asymmetric measure, not tn (T computed lazily).
            if constexpr (SC_PASSEVAL)
                if (pass_[PASS_RFP] > 0) need_tension();
            if (eval - kRFPMargin * (depth - improving) - (SC_PASSEVAL ? tu * pass_[PASS_RFP] / 16 : 0) >= beta)
                return (eval + beta) / 2;
        }

        // ---- Null move pruning ----
        // Hand the opponent a free move; if we still beat beta the position is
        // almost certainly a cutoff. Disabled without non-pawn material
        // (zugzwang) and never twice in a row.
        if (depth >= kNMPMinDepth && eval >= beta &&
            (ss - 1)->current_move != Move(Move::NULL_MOVE) &&
            has_non_pawn_material(board, board.sideToMove())) {
            bool nmp_ok = true;
            if constexpr (SC_PASSEVAL) {
                // Static prediction of the null search (T computed lazily, only here if no
                // earlier plug asked). P_est = eval - T: the (TT-sharpened) eval as the node's
                // best value estimate minus the static tempo. Skip the null search when passing
                // looks better than moving (the zugzwang signature NMP is blind to) or when the
                // passed position is already far below beta (the reduced search would fail low
                // and its cost is wasted). Both thresholds individually guarded: 0 = off.
                const int zug = pass_[PASS_ZUG], nm = pass_[PASS_NMP_MARGIN];
                if ((zug > 0 || nm > 0) && !is_mate_score(beta)) {
                    need_tension();
                    if (tension != VALUE_NONE) {
                        if (zug > 0 && tension < -zug)             nmp_ok = false;
                        if (nm > 0 && eval - tension < beta - nm)  nmp_ok = false;
                    }
                }
            }
            if (nmp_ok) {
            const Depth R = 4 + depth / 4 + std::min(3, (eval - beta) / 200);

            ss->current_move = Move(Move::NULL_MOVE);
            ss->moved_piece  = 12;
            ss->moved_to     = 0;

            pool_.tt_.prefetch(board.zobristAfter(Move(Move::NULL_MOVE)));  // D3b: null child
            nnue::acc_make_null();
            board.makeNullMove();
            const Value v = -negamax(board, ss + 1, depth - R, -beta, -beta + 1, !cut_node);
            board.unmakeNullMove();
            nnue::acc_unmake_null();

            if (pool_.stop_.load(std::memory_order_relaxed)) return VALUE_ZERO;
            // Never return unproven mate scores from a null search.
            if (v >= beta) return is_mate_score(v) ? beta : v;
            }
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

                pool_.tt_.prefetch(board.zobristAfter(pm));
                nnue::acc_make(board, pm);
                board.makeMove(pm);
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
    if constexpr (SC_PASSEVAL)   // LMR reads tn for every reduced move: compute T now, at THIS node
        if (pass_[PASS_LMR_DIV] > 0 && depth >= 2) need_tension();
    OrderingContext ctx;
    ctx.tt_move     = tt_move;
    // MultiPV: search the current PV line's move first. Line 0 keeps the TT-probe
    // move, so the default search orders exactly as before.
    if (root && pv_idx_ > 0) ctx.tt_move = root_moves_[static_cast<std::size_t>(pv_idx_)].move;
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

    // MultiPV: only the first line resets these -- a later line's root search must not
    // wipe line 0's second-best and effort shares, which the time manager reads afterwards.
    if (root && pv_idx_ == 0) root_second_ = -VALUE_INFINITE;  // reset per-iteration 2nd-best tracker
    if (root && pv_idx_ == 0) root_n_      = 0;                // reset per-iteration effort tracker

    // Moves actually searched, for history maluses after a cutoff.
    Move tried_quiets[64];
    Move tried_caps[32];
    int  n_quiets = 0, n_caps = 0;

    Move m;
    while ((m = picker.next(skip_quiets)) != Move(Move::NO_MOVE)) {
        if (m == excluded) continue;

        // MultiPV: at the root skip the PV lines already found this iteration, before
        // they can count toward move_count.
        if (root && pv_idx_ > 0) {
            bool searched = false;
            for (int k = 0; k < pv_idx_; ++k)
                if (root_moves_[static_cast<std::size_t>(k)].move == m) { searched = true; break; }
            if (searched) continue;
        }
        // `go searchmoves`: the root searches only the listed moves.
        if (root && root_restricted_ &&
            std::find(root_moves_.begin(), root_moves_.end(), m) == root_moves_.end())
            continue;

        const bool quiet       = is_quiet(board, m);
        const bool capture     = board.isCapture(m);
        // E4: givesCheck is a magic-lookup predicate computed for EVERY move here, but only
        // quiets read it before pruning (futility) and only surviving moves read it at LMR.
        // Memoise it and evaluate it lazily, so moves pruned by LMP/SEE never pay for it.
        int  gives_check_memo = -1;
        auto gives_check      = [&]() -> bool {
            if (gives_check_memo < 0)
                gives_check_memo = (board.givesCheck(m) != chess::CheckType::NO_CHECK) ? 1 : 0;
            return gives_check_memo != 0;
        };

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
                if (!in_check && depth <= kFutilityMaxDepth &&
                    ss->static_eval + fut_margin <= alpha && !gives_check())
                    skip_quiets = true;

                if (skip_quiets) continue;

                // SEE: skip quiets that lose material badly (walking into a pawn).
                if (depth <= kSeeGateMaxDepth &&
                    !see::see_ge(board, m, -kSeeQuietMargin * depth))
                    continue;
            } else {
                // SEE: skip clearly losing captures at shallow depth.
                // A capture the picker classified good (see_ge(m, 0) held) trivially passes any
                // threshold <= 0 -- see_ge is monotone in the threshold -- so only bad or
                // unclassified captures pay for this SEE.
#if defined(__ARM_NEON)
                if (depth <= kSeeGateMaxDepth &&
#else
                if (depth <= kSeeGateMaxDepth && picker.last_see() <= 0 &&
#endif
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
            if constexpr (SC_EVAL_SLOT) ss->static_eval = eval_slot;  // the verification search reuses this slot

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

        // Resolve givesCheck on the PARENT position for every move that is actually
        // searched (the LMR term below runs after makeMove and must not re-query).
        const bool gives_check_now = gives_check();
        // D3: prefetch the child's TT cluster BEFORE the NNUE update so the accumulator
        // work hides the miss (issued after makeMove it led the probe by only a few ns).
        // zobristAfter() is the library's exact post-move key for the makeMove<false>
        // the search uses (same en-passant rule); a prefetch cannot change any result.
        pool_.tt_.prefetch(board.zobristAfter(m));
        nnue::acc_make(board, m);
        board.makeMove(m);

        // ---- PVS + late move reductions ----
        Value score = -VALUE_INFINITE;

        if (depth >= 2 && move_count > 1 + (root ? 1 : 0)) {
            int r = kLmr[std::min<int>(depth, 63)][std::min(move_count, 63)];
            r += !improving;
            r += 2 * cut_node;
            r -= pv_node;
            r -= gives_check_now;
            if (!quiet)
                r -= 1;  // reduce tactical moves less
            else
                r -= std::clamp(picker.last_score() / 8192, -2, 2);  // history-informed

            if constexpr (SC_PASSEVAL) {
                // Reduce MORE under tension, and only quiets. The measured signal (F1, phase
                // controlled): best-move change d3->d7 falls 66.9% -> 39.9% from the lowest to the
                // highest T decile, partial Spearman(T, unstable | |E|, occ) = -0.152 and negative
                // within every phase. High tension means the position is FORCED, so a late QUIET is
                // LESS likely than usual to be the refutation -- the opposite of what the first
                // attempt assumed, which lost 4 Elo. (|R| also rises with T, but that is error in
                // the eval NUMBER, which LMR never returns; it belongs to the RFP margin.)
                // Captures and check-givers are exempt: in a forced position they ARE the candidate
                // refutation, and they already get r -= 1 / r -= gives_check_now above. The first
                // attempt applied its term outside that branch, so it hit them too. Kept before the
                // width scaling on purpose: a positive term is damped at an unstable root, where a
                // negative one used to escape damping entirely by driving r <= 0.
                const int div = pass_[PASS_LMR_DIV];
                if (div > 0 && quiet && !gives_check_now) r += std::min(2, tn / div);
            }

            if constexpr (SC_THRSIG > 0) {
                // Item #9: do not reduce a quiet move that creates a new attack on a higher-valued
                // piece. Unlike item #8's feature-delta magnitude, which failed in both directions,
                // this is a semantic condition -- "this move makes a threat" -- and it is the class of
                // quiet move a reduction is most likely to mis-handle, because history scores where a
                // move worked before, not what it threatens here.
                if (quiet && !gives_check_now && nnue::acc_threats_made() >= SC_THRSIG_MIN)
                    r -= SC_THRSIG;
            }

            if constexpr (SC_ACCSIG != 0 && SC_ACCSIG_LOW > 0) {
                // Reduce a quiet MORE when it barely changes what the net sees: a move whose feature
                // delta is in the bottom quartile is a do-nothing move (a shuffle behind the lines),
                // and is correspondingly unlikely to be the refutation. Structurally cheaper than the
                // relief form below, because this one PRUNES harder and so shrinks the tree, where
                // reducing less grows it -- the measured difference between starting ~3 Elo behind and
                // starting ahead.
                if (quiet && !gives_check_now && nnue::acc_delta_l1() < SC_ACCSIG_LOW)
                    r += 1;
            }

            if constexpr (SC_ACCSIG != 0 && SC_ACCSIG_DIV > 0) {
                // Item #8: reduce a quiet move LESS when it changes what the net sees a lot.
                // sig = ||accumulator(child) - accumulator(parent)||_1, i.e. the L1 norm of the
                // feature columns this move swaps -- a per-MOVE measure of how consequential the
                // move is, which history cannot supply because history scores a move's past
                // success, not how much it alters this position's features.
                //
                // Why this is not E2 repeated. E2's tension was a property of the NODE, identical
                // for every late quiet there, so it could only shift the average reduction and had
                // nothing to discriminate with; it failed in both directions. This signal differs
                // per move at the same node, which is the shape a reduction decision actually
                // wants. Captures and check-givers stay exempt for the same reason as in E2: they
                // already get relief above.
                if (quiet && !gives_check_now && depth >= SC_ACCSIG_MINDEPTH) {
                    const int sig = nnue::acc_delta_l1();
                    r -= std::min(SC_ACCSIG_MAX, sig / SC_ACCSIG_DIV);
                }
            }

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
        // MultiPV: line 0 only (the time manager reads line 0's shares).
        if (root && pv_idx_ == 0 && root_n_ < MAX_MOVES) {
            root_mv_[root_n_]   = m;
            root_cost_[root_n_] = nodes_.load(std::memory_order_relaxed) - nodes_before;
            ++root_n_;
        }

        // Track the best score among non-best root moves (fail-soft, so it's a
        // true upper bound on each): when a new best appears the old best is
        // demoted to second; otherwise this move itself is a second candidate.
        // MultiPV: line 0 only -- the time manager and the width read it, and a
        // later line's "second best" is meaningless to them.
        if (root && pv_idx_ == 0) {
            if (score > best) {
                if (best > root_second_) root_second_ = best;
            } else if (score > root_second_) {
                root_second_ = score;
            }
        }

        // MultiPV bookkeeping, separate from the search's own
        // best/alpha/pv_ state: the new best of this line records its score (clipped to
        // the bound it hit, flagged inexact) and PV; every other root move drops to
        // -VALUE_INFINITE so the stable sort in think() moves only the PV to the front.
        if (root) {
            auto it = std::find(root_moves_.begin(), root_moves_.end(), m);
            if (it != root_moves_.end()) {
                RootMove& rm = *it;
                if (move_count == 1 || score > alpha) {
                    rm.score = rm.uci_score = score;
                    rm.unset_inexact();
                    if (score >= beta) {
                        rm.inexact_lower = true;
                        rm.uci_score     = beta;
                    } else if (score <= alpha) {
                        rm.inexact_upper = true;
                        rm.uci_score     = alpha;
                    }
                    rm.pv[0] = m;
                    std::copy(pv_[ss->ply + 1].begin(),
                              pv_[ss->ply + 1].begin() + pv_len_[ss->ply + 1],
                              rm.pv.begin() + 1);
                    rm.pv_len = pv_len_[ss->ply + 1] + 1;
                } else {
                    rm.score = -VALUE_INFINITE;
                }
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

                if (root && pv_idx_ == 0) best_move_ = m;

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
        // MultiPV: a non-first PV line must not overwrite the root entry, which holds the
        // true best.
        if (!(root && pv_idx_ > 0))
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
                int bonus = std::clamp((best - ss->static_eval) * depth *
                                           (have_best ? 12 : 18) / 128,
                                       -HISTORY_MAX / 4, HISTORY_MAX / 4);
                if constexpr (SC_PASSEVAL) {
                    // Weight the update by how LEARNABLE this node's residual is, not by how large
                    // it is. Correction history can only absorb the part of R = search - static that
                    // is consistent among positions sharing the key; the rest is noise it averages
                    // away. F2 (tools/verify/passeval_learnability.py) measured exactly that on F1's
                    // 19,529 rows, as the intraclass correlation of R within pawn-structure keys:
                    // it RISES with tension, 0.5456 -> 0.6004 from the lowest to the highest T
                    // quintile, while mean |R| rises 117 -> 222. So high-tension residuals are both
                    // bigger AND more systematically tied to the key, and deserve a larger step.
                    //
                    // This is the only plug point in the campaign whose direction comes from a
                    // measurement of its OWN mechanism rather than from a general tension statistic;
                    // E5 failed twice by borrowing the move-stability table for a question about
                    // scores, so the distinction is load-bearing.
                    //
                    // T is needed only at nodes that actually reach this update, which is why the
                    // query sits here rather than at the node head, and only at depth >=
                    // PassMinDepth -- deep nodes carry the most reliable labels anyway.
                    const int w = pass_[PASS_CORR_W];
                    if (w > 0) {
                        need_tension();
                        if (tension != VALUE_NONE)
                            bonus = std::clamp(static_cast<int>(
                                        static_cast<std::int64_t>(bonus) * (256 + tn * w / 16) / 256),
                                    -HISTORY_MAX / 4, HISTORY_MAX / 4);
                    }
                }
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

    // E6: the two continuation rows are fixed for this node; resolve them once instead
    // of per bumped quiet (same entries updated in the same order).
    std::int16_t (*cont_rows[2])[64] = {nullptr, nullptr};
    for (int off = 1; off <= 2; ++off) {
        const Stack* prev = ss - off;
        if (prev->moved_piece < 12)
            cont_rows[off - 1] = history_.cont->v[prev->moved_piece][prev->moved_to];
    }

    // Bump one quiet move's butterfly + continuation entries by `b`.
    auto bump_quiet = [&](Move m, int b) {
        history_update(history_.main[stm][m.from().index()][m.to().index()], b);
        const int piece = static_cast<int>(board.at(m.from()));
        for (int k = 0; k < 2; ++k)
            if (cont_rows[k]) history_update(cont_rows[k][piece][m.to().index()], b);
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

    // While pondering, the bestmove is held until ponderhit or stop: the GUI expects it only
    // then, never while we're searching on its clock. `go infinite` is held the same way until
    // `stop`: a trivial position can run the loop out to MAX_PLY in milliseconds, and infinite
    // analysis must never answer unasked.
    const auto hold_bestmove = [this] {
        while ((pool_.pondering_.load(std::memory_order_acquire) || pool_.limits_.infinite) &&
               !pool_.stop_.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };

    Movelist root_moves;
    chess::movegen::legalmoves(root_moves, board);
    // `go searchmoves`: search only the listed moves (the UCI parser keeps legal ones only).
    root_restricted_ = false;
    if (const auto& only = pool_.limits_.searchmoves; !only.empty()) {
        Movelist kept;
        for (const auto& m : root_moves)
            if (std::find(only.begin(), only.end(), m) != only.end()) kept.add(m);
        if (!kept.empty() && kept.size() < root_moves.size()) {
            root_moves       = kept;
            root_restricted_ = true;
        }
    }

    if (root_moves.empty()) {
        // Mated or stalemated root: the protocol still expects a bestmove line.
        if (main_worker) {
            hold_bestmove();
            if (!g_gen_silent) sync_cout << "bestmove 0000" << sync_endl;
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

    // MultiPV (3.3): one RootMove per legal move, in
    // generation order; the effective line count is clamped to the legal-move count.
    root_moves_.clear();
    root_moves_.reserve(root_moves.size());
    for (const auto& rm : root_moves) root_moves_.emplace_back(rm);
    multipv_ = std::clamp<int>(pool_.limits_.multipv, 1, static_cast<int>(root_moves_.size()));

    Value prev_score = VALUE_NONE;  // line 0's score at the last completed iteration
    Move  prev_best  = Move(Move::NO_MOVE);
    int   stable     = 0;  // consecutive completed iterations with the same best move

    for (Depth d = 1; d <= max_depth; ++d) {
        // Lazy SMP depth staggering: odd-numbered helpers skip even depths, so
        // half the pool runs ahead and seeds the shared TT from above while
        // the rest (and the main worker) fill it in order.
        if (!main_worker && (id_ & 1) && (d & 1) == 0 && d < max_depth) continue;

        nnue::acc_reset(board);  // fresh incremental base accumulator at the root

        width_    = pool_.width_.load(std::memory_order_relaxed);  // stable per iteration
        if constexpr (SC_PASSEVAL)   // one consistent tunable set per iteration (see kPassParams)
            for (int i = 0; i < PASS_N; ++i) pass_[i] = g_pass[i].load(std::memory_order_relaxed);

        // Save the last iteration's scores and PVs before the first line is searched and
        // every non-PV score is reset to -VALUE_INFINITE.
        for (std::size_t i = 0; i < root_moves_.size(); ++i) {
            RootMove& rm        = root_moves_[i];
            rm.prev_score       = rm.score;
            rm.prev_pv          = rm.pv;
            rm.prev_pv_len      = rm.pv_len;
            rm.prev_score_exact = static_cast<int>(i) < multipv_;
        }

        // ---- MultiPV loop: one full root search per PV line ----
        bool stopped = false;
        for (pv_idx_ = 0; pv_idx_ < multipv_; ++pv_idx_) {
            seldepth_ = 0;  // per line

            // ---- Aspiration windows ----
            // Search a narrow window around THIS line's previous score, widening
            // geometrically on failure. Small windows produce far more cutoffs. At
            // MultiPV 1 the centre is exactly the old prev_score: line 0's score after
            // the sort is the returned fail-soft best.
            const Value centre = root_moves_[static_cast<std::size_t>(pv_idx_)].prev_score;
            Value delta = kAspirationDelta;
            Value alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
            if (d >= 4 && centre != -VALUE_INFINITE) {
                alpha = std::max<Value>(centre - delta, -VALUE_INFINITE);
                beta  = std::min<Value>(centre + delta, VALUE_INFINITE);
            }

            Value v;
            while (true) {
                v = negamax(board, ss, d, alpha, beta, false);
                // Bring this line's best to the front of the unsearched tail. Stable, so
                // every other root move keeps its order.
                std::stable_sort(root_moves_.begin() + pv_idx_, root_moves_.end());
                if (pool_.stop_.load(std::memory_order_relaxed)) break;

                if (v <= alpha) {  // fail-low: drop alpha, pull beta toward it
                    beta  = (alpha + beta) / 2;
                    alpha = std::max<Value>(v - delta, -VALUE_INFINITE);
                } else if (v >= beta) {  // fail-high: raise beta
                    beta = std::min<Value>(v + delta, VALUE_INFINITE);
                } else {
                    break;
                }
                delta += delta / 2;
            }
            // This line's selective depth: its maximum, taken at the end of its search
            // (freezing it when the PV move is recorded would under-report).
            root_moves_[static_cast<std::size_t>(pv_idx_)].seldepth = seldepth_;

            if (pool_.stop_.load(std::memory_order_relaxed)) {
                stopped = true;
                if (pv_idx_ > 0) {
                    // An aborted search must not spoil a completed earlier line: never let it
                    // overtake a proven-loss pv_idx_-1, and never trust an exact loss from an
                    // aborted search.
                    RootMove& cur  = root_moves_[static_cast<std::size_t>(pv_idx_)];
                    RootMove& prev = root_moves_[static_cast<std::size_t>(pv_idx_ - 1)];
                    auto is_loss = [](Value s) noexcept {
                        return s != -VALUE_INFINITE && s <= VALUE_TB_LOSS_IN_MAX_PLY;
                    };
                    auto exact_loss = [&](const RootMove& r) { return is_loss(r.score) && !r.is_inexact(); };
                    if ((is_loss(prev.score) && cur < prev) || exact_loss(cur)) {
                        if (cur.prev_score != -VALUE_INFINITE && cur.prev_score_exact &&
                            cur.prev_score <= prev.score) {
                            // The exact previous score is safe to show; it cannot overtake prev.
                            cur.score = cur.uci_score = cur.prev_score;
                            cur.prev_score = -VALUE_INFINITE;
                            cur.pv         = cur.prev_pv;
                            cur.pv_len     = cur.prev_pv_len;
                            cur.unset_inexact();
                        } else {
                            // Cap to the best possible and mark the score inexact.
                            if (is_loss(prev.score)) {
                                cur.score = cur.uci_score = prev.score;
                                cur.prev_score    = -VALUE_INFINITE;
                                cur.pv_len        = 1;
                                cur.inexact_upper = true;
                            } else {
                                cur.inexact_upper = false;
                            }
                            cur.inexact_lower = !cur.inexact_upper;
                        }
                    }
                    for (int i = pv_idx_ + 1; i < multipv_; ++i) {
                        RootMove& r = root_moves_[static_cast<std::size_t>(i)];
                        if (exact_loss(r)) r.inexact_lower = true;
                    }
                }
                break;
            }

            // Sort the lines searched so far.
            std::stable_sort(root_moves_.begin(), root_moves_.begin() + pv_idx_ + 1);
        }
        if (stopped) break;

        completed_ = d;
        // The iteration's score is line 0's: what time management, the width, gengame and
        // the effort trim read (at MultiPV 1 exactly the returned fail-soft best). The mate
        // stop is judged on the LAST line so a proven mate on line 0 does not cut the other
        // lines short; at MultiPV 1 that is line 0.
        const Value score      = root_moves_[0].score;
        const Value stop_score = root_moves_[static_cast<std::size_t>(multipv_ - 1)].score;
        prev_score             = score;
        // The move to play is the sorted top line.
        // At MultiPV 1 this is the move line 0 just found, so nothing changes; at MultiPV > 1
        // a later line, searched with its own full window, can out-score line 0 (line 0 may
        // have dropped that move on a reduced null-window fail-low), and the sort has it.
        best_move_ = root_moves_[0].move;

        if (!main_worker) continue;  // helpers never report or manage time

        report_multipv(d);

        // The 2nd move of line 0's PV is the reply we expect; offer it as the ponder
        // move so the GUI can search it on the opponent's clock. (pv_[0] holds the LAST
        // line's PV once the loop has run, so it can no longer serve here.)
        ponder_move_ = (root_moves_[0].pv_len >= 2) ? root_moves_[0].pv[1] : Move(Move::NO_MOVE);

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
                width        = kBreadthMax * wg * ws /
                        (kBreadthGapRange * kBreadthScoreRange);
            }
            const int old = pool_.width_.load(std::memory_order_relaxed);
            if (width != old) {
                pool_.width_.store(width, std::memory_order_relaxed);
                // Log only meaningful shifts, not every wobble.
                if (!g_gen_silent && ((width == 0) != (old == 0) || std::abs(width - old) >= 32))
                    sync_cout << "info string search width " << width << "/" << kBreadthMax
                              << sync_endl;
            }
        }

        // ---- Between-iteration stop conditions (main worker only) ----
        if (pool_.limits_.nodes && pool_.total_nodes() >= pool_.limits_.nodes) break;
        // `go mate N`: done once the best line mates within N moves.
        if (pool_.limits_.mate > 0 && score >= VALUE_MATE_IN_MAX_PLY &&
            VALUE_MATE - score <= 2 * pool_.limits_.mate - 1)
            break;

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
            } else if (is_mate_score(stop_score)) {
                // Stop only once the mate is proven SHORTEST at this depth. A mate
                // whose distance still exceeds the searched depth is a TT-injected long
                // mate from an earlier search: breaking on it plays a "mates eventually"
                // move and abandons the rest of the budget instead of deepening to the
                // quickest mate. Requiring d >= distance still caps ponder spins (it
                // fires the moment the mate is genuinely proven, not on a stale TT hit).
                // MultiPV: judged on the LAST line (see stop_score above).
                const int mate_plies = (stop_score > 0) ? (VALUE_MATE - stop_score)
                                                        : (VALUE_MATE + stop_score);
                if (d >= mate_plies) break;                  // shortest mate proven: play it
            }
        }

        // Time-based stops: only when the clock is actually ours (not pondering).
        if (pool_.budget_.use_clock && !pool_.pondering_.load(std::memory_order_acquire)) {
            // Trim the soft budget when the best move is obvious. Only for a
            // real game clock, not a fixed `movetime` (there is no clock to
            // save, so we honor the full think).
            std::int64_t soft = pool_.budget_.soft_ms;
            if constexpr (SC_PASSEVAL) {
                // Root tension: spend MORE time where tension is high, LESS where it is low, at
                // constant mean. Measured, not assumed: the opposite apportionment (E5-prime) lost
                // 18 Elo [-36, -1], LOS 2%, over 538 games at the same mean time, so the direction
                // is established by experiment rather than by which correlation one picks.
                //
                // Why the instability table misleads here. Tension predicts that the best MOVE is
                // settled (change d3->d7 falls 66.9% -> 39.9%), which argues for less time; but it
                // also predicts that the static SCORE is badly wrong (mean |R| 96.8 -> 235.2 across
                // the same deciles, and that survives phase control). Time allocation should track
                // the second, not the first: a node whose move is already settled still hands its
                // parent a score, and a score that is 235 cp off loses games no matter how stable
                // the move was. Extra search buys score accuracy where there is most of it to buy.
                //
                // Mean-preserving by construction: the factor is -PassTmScale/256 at tn = 0, crosses
                // exactly 1.0 at kPassTmMid (the measured median of tn) and saturates at
                // +PassTmScale/256. The ORIGINAL design had this direction but scaled upward only,
                // so it raised mean time per move and tested "think longer" as much as "think where
                // it matters"; a PASS could not have been attributed to either. Clock games only;
                // never past the hard limit (a soft above hard would only ever end in a truncated
                // iteration).
                const int ts = pass_[PASS_TM_SCALE];
                if (ts > 0 && root_tension_ != VALUE_NONE && pool_.limits_.movetime == 0) {
                    const int tn  = std::clamp<int>(root_tension_ - pass_[PASS_TAU], 0, pass_[PASS_CAP]);
                    const int adj = std::clamp<int>(tn * ts / kPassTmMid, 0, 2 * ts) - ts;
                    soft = std::min<std::int64_t>(pool_.budget_.hard_ms, soft * (256 + adj) / 256);
                }
            }
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

            if (elapsed_ms(pool_.start_time_.load(std::memory_order_relaxed)) >= soft) break;

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
        // The ID loop can end on its own (a proven mate, MAX_PLY) while pondering or in
        // `go infinite`: hold the bestmove until ponderhit / stop.
        hold_bestmove();

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
            const bool c960 = pool_.root_.chess960();  // castling spelling follows the root's mode
            std::string line = "bestmove " + chess::uci::moveToUci(best_move_, c960);
            if (ponder_move_ != Move(Move::NO_MOVE))
                line += " ponder " + chess::uci::moveToUci(ponder_move_, c960);
            sync_cout << line << sync_endl;
        }

        pool_.stop_.store(true, std::memory_order_release);
        pool_.searching_.store(false, std::memory_order_release);
    }
}

// ---- Reporting ----------------------------------------------------------------

// One `info` line per PV line. A line not reached this iteration
// (score == -VALUE_INFINITE) is shown at depth-1 from its previous score and PV; a line
// whose search stopped on an aspiration bound carries lowerbound/upperbound.
void Worker::report_multipv(Depth depth) {
    if (g_gen_silent) return;  // gengame: suppress per-iteration info lines
    const std::int64_t  ms    = std::max<std::int64_t>(1, elapsed_ms(pool_.start_time_.load(std::memory_order_relaxed)));
    const std::uint64_t nodes = pool_.total_nodes();
    const std::uint64_t nps   = nodes * 1000ULL / static_cast<std::uint64_t>(ms);
    const bool          c960  = pool_.root_.chess960();
    const int           rn    = g_root_noise.load(std::memory_order_relaxed);
    // Read once per report so every line of it agrees; the chances use the root's material.
    const bool          show_wdl = g_show_wdl.load(std::memory_order_relaxed);
    const int           root_mat = show_wdl ? wdl_material(pool_.root_) : 0;

    for (int i = 0; i < multipv_; ++i) {
        const RootMove& rm       = root_moves_[static_cast<std::size_t>(i)];
        const bool      use_prev = (rm.score == -VALUE_INFINITE);
        if (depth == 1 && use_prev && i > 0) continue;

        const Depth d = use_prev ? std::max<Depth>(1, depth - 1) : depth;
        Value       v = use_prev ? rm.prev_score : rm.uci_score;
        if (v == -VALUE_INFINITE) v = VALUE_ZERO;

        std::ostringstream ss;
        ss << "info depth " << d << " seldepth " << rm.seldepth << " multipv " << (i + 1) << " score ";

        Value shown = v;   // the value the score token reports (win/draw/loss is computed from it)
        if (is_mate_score(v)) {
            // UCI wants distance in moves; positive when we deliver the mate.
            const int plies      = (v > 0) ? (VALUE_MATE - v) : (VALUE_MATE + v);
            const int mate_moves = (v > 0) ? (plies + 1) / 2 : -((plies + 1) / 2);
            ss << "mate " << mate_moves;
        } else {
            // Remove this line's root-noise bonus so the reported score (the self-play
            // training label) is the move's true eval, not the noisy one.
            const Value clean = (rn > 0) ? v - root_noise_offset(rm.move, rn) : v;
            ss << "cp " << clean;
            shown = clean;
        }

        // Previous-iteration scores are exact whatever their flags say.
        if (!use_prev) {
            if (rm.inexact_lower)      ss << " lowerbound";
            else if (rm.inexact_upper) ss << " upperbound";
        }

        // After the bound tag: a GUI parser may read a bound only directly after the score value.
        if (show_wdl) {
            const WdlPermille p = wdl_from_value(shown, root_mat);
            ss << " wdl " << p.win << ' ' << p.draw << ' ' << p.loss;
        }

        ss << " nodes " << nodes << " nps " << nps << " time " << ms << " tbhits "
           << syzygy::hits() << " hashfull " << pool_.tt_.hashfull();

        const auto& pv  = use_prev ? rm.prev_pv : rm.pv;
        const int   len = use_prev ? rm.prev_pv_len : rm.pv_len;
        if (len > 0) {
            ss << " pv";
            for (int k = 0; k < len; ++k) ss << ' ' << chess::uci::moveToUci(pv[k], c960);
        }

        sync_cout << ss.str() << sync_endl;
    }
}

}  // namespace engine
