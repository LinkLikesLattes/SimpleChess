#pragma once

// -----------------------------------------------------------------------------
// search.hpp
//
// Alpha-beta searcher with Lazy SMP multithreading (v0.5).
//
// Parallelism model (the modern top-engine approach): all threads search the
// same root position and communicate only through the shared transposition
// table. There is no explicit work splitting — helpers desynchronize through
// TT hits and (for odd-numbered helpers) by skipping even depths so they run
// ahead and seed the table from above. Per-thread state (history heuristics,
// killers, PV, node counts) lives in a Worker; the main worker (id 0) owns
// time management, `info` output, and the final `bestmove`. Helpers are
// silent.
//
// TT races are deliberately tolerated (no locks): entries may tear under
// concurrent writes, but every TT move is only ever used after matching it
// against generated legal moves, so a torn entry can at worst cost a bad
// ordering hint or a wrong-depth cutoff — noise, not crashes.
//
// Search techniques (per worker) are unchanged from v0.3/v0.4:
//   iterative deepening + aspiration windows, PVS, TT cutoffs, IIR, razoring,
//   reverse futility, null move, ProbCut, singular extensions, LMR/LMP,
//   futility + SEE pruning, quiescence with TT/SEE/delta, full history stack.
// -----------------------------------------------------------------------------

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <iosfwd>
#include <thread>
#include <vector>
#include <condition_variable>
#include <mutex>

#include "history.hpp"
#include "timeman.hpp"
#include "tt.hpp"

// Build-time overridable UCI defaults. The native build keeps the originals; the
// The Windows/CCRL release build sets Hash 256 / Threads 1 via these defines.
// SC_DEFAULT_HASH is in MB.
#ifndef SC_DEFAULT_HASH
#define SC_DEFAULT_HASH 4096
#endif
#ifndef SC_DEFAULT_THREADS
#define SC_DEFAULT_THREADS 8
#endif
// Pass-eval tension (dev-notes/NOVELTY_CANDIDATES.md #1; campaign record dev-notes/PASSEVAL.md).
// 0 = no tension machinery at all, bit-for-bit the pre-campaign engine; 1 = the machinery compiled in
// and the Pass* UCI tunables exposed. ON BY DEFAULT since 3.2.0 because it carries E6, the one plug
// point kept: the correction-history update weighted by tension (PassCorrW 16, PassMinDepth 6, PassTau
// 45, PassCap 1151 -- constants from the phase-controlled F1 diagnostic, +3.5 Elo [-1.7,+8.7] LOS 90.4%
// over 6000 games at 10+0.1, deployed to lichess-bot 2026-09-11). Every OTHER tunable defaults to 0 and
// is inert: those plug points were tested and stripped (see PASSEVAL.md), the code stays for the record.
// SC_PASSEVAL_VERIFY adds the in-search oracle (dev builds only: every T is cross-checked against a real
// null move).
#ifndef SC_PASSEVAL
#define SC_PASSEVAL 1
#endif
#ifndef SC_PASSEVAL_VERIFY
#define SC_PASSEVAL_VERIFY 0
#endif
// The Pass* tunables are development knobs, not user options: they are never advertised in
// the `uci` option list unless a tuning build sets SC_PASS_UCI_OPTIONS=1. setoption still
// accepts them silently either way, so the match tooling can drive them (--set PassX=...).
#ifndef SC_PASS_UCI_OPTIONS
#define SC_PASS_UCI_OPTIONS 0
#endif
// Compile-time defaults of the tunables (each experiment builds with ITS values as defaults so the
// PGO profile and the played configuration agree; UCI setoption only serves alternates).
#ifndef SC_PASS_MINDEPTH
#define SC_PASS_MINDEPTH 6
#endif
#ifndef SC_PASS_TAU
#define SC_PASS_TAU 45
#endif
#ifndef SC_PASS_CAP
#define SC_PASS_CAP 1151
#endif
#ifndef SC_PASS_RAZOR
#define SC_PASS_RAZOR 0
#endif
#ifndef SC_PASS_RFP
#define SC_PASS_RFP 0
#endif
#ifndef SC_PASS_LMRDIV
#define SC_PASS_LMRDIV 0
#endif
#ifndef SC_PASS_NMPMARGIN
#define SC_PASS_NMPMARGIN 0
#endif
#ifndef SC_PASS_ZUG
#define SC_PASS_ZUG 0
#endif
#ifndef SC_PASS_QSLAMBDA
#define SC_PASS_QSLAMBDA 0
#endif
#ifndef SC_PASS_TMSCALE
#define SC_PASS_TMSCALE 0
#endif
#ifndef SC_PASS_CORRW
#define SC_PASS_CORRW 16
#endif
// Item #8: per-move accumulator-delta significance into LMR. SC_ACCSIG compiles the plug in;
// SC_ACCSIG_DIV 0 keeps it inert, so a candidate build with DIV 0 is identical to the control.
#ifndef SC_ACCSIG
#define SC_ACCSIG 0
#endif
#ifndef SC_ACCSIG_DIV
#define SC_ACCSIG_DIV 0
#endif
#ifndef SC_ACCSIG_MAX
#define SC_ACCSIG_MAX 2
#endif
// The opposite half of the same signal: reduce a quiet MORE when it barely changes what the net sees.
// 0 = off. Cheaper by construction than SC_ACCSIG_DIV, because reducing more shrinks the tree where
// reducing less grows it.
#ifndef SC_ACCSIG_LOW
#define SC_ACCSIG_LOW 0
#endif
// Minimum node depth at which the accsig plug fires (0 = every LMR node). The relief form's whole
// problem is the tree it grows, and a depth gate cuts the number of firings far faster than it cuts
// the effect -- the same trade that took E2-prime from +7.12% to -0.10% time-to-depth.
#ifndef SC_ACCSIG_MINDEPTH
#define SC_ACCSIG_MINDEPTH 0
#endif
// Item #9: plies of LMR relief for a quiet move that creates at least SC_THRSIG_MIN new attacks on a
// higher-valued enemy piece. 0 = off.
#ifndef SC_THRSIG
#define SC_THRSIG 0
#endif
#ifndef SC_THRSIG_MIN
#define SC_THRSIG_MIN 1
#endif
// 50-move-rule static-eval damping: scale the fresh static eval toward zero in proportion to the
// halfmove clock, so a position shuffling toward the 50-move draw stops reporting a full advantage
// it can no longer force. `eval -= eval * halfmove_clock / 199` (roughly halved at clock 100, the
// draw threshold; 0 only at 199, which real play never reaches). Applied ONLY to a freshly computed
// net eval, never to a value reused from the TT (that value was already damped when written -- damping
// it again would double-damp on transpositions). Shipped on by default (+4 Elo [-2,+10], LOS 89% over
// 1881 pairs at 10+0.1 -- a small, correctness-motivated gain); set to 0 to restore the old behaviour.
#ifndef SC_R50DAMP
#define SC_R50DAMP 1
#endif
// 50-move-rule mate reclassification: a mate score read back from the transposition table is only
// real if the mating side can actually deliver it before the 50-move draw resets the game. When the
// reported distance to mate exceeds the plies the halfmove clock still allows (100 - clock), the mate
// is unreachable, so downgrade it to a high but non-mate score (VALUE_TB_WIN_IN_MAX_PLY - 1) -- the
// search keeps treating the position as winning but stops trusting, extending, and reporting a forced
// mate it cannot force. Scoped strictly to the mate band (>= VALUE_MATE_IN_MAX_PLY): tablebase-band
// scores are never cached in the TT here (the WDL probe returns before the store), so they cannot
// reach this path and are intentionally left untouched. Shipped on by default as a correctness fix:
// SPRT was neutral at fast TC (+1 [-3,+5], LOS 69%, no regression over 3000 pairs) because the node
// cost of surrendering false-mate cutoffs cancels the benefit there, but the engine no longer reports
// or chases a mate the 50-move rule voids, and the payoff is a long-TC/endgame property the fast test
// cannot see. Set to 0 to restore the plain ply de-shift.
#ifndef SC_R50MATE
#define SC_R50MATE 1
#endif
#include "types.hpp"

namespace engine {

class Search;

// Set the root-move noise magnitude in centipawns (0 = off, normal play). Used
// by the self-play data generator to diversify games; see search.cpp.
void set_root_noise(int cp);

// UCI_ShowWDL: add "wdl W D L" (win/draw/loss per mille for the side to move, from the model in
// wdl.hpp) to every scored info line. Display only; takes effect from the next report.
void set_show_wdl(bool on);

// Silence all UCI output (info/bestmove) from the search. Used by the in-engine
// game generator (`gengame`) so its many per-ply searches don't spam stdout.
void set_gen_silent(bool silent);

// Pass-eval tunables (inert unless SC_PASSEVAL). set_pass_param returns false for an unknown
// name (case-insensitive) and clamps to the tunable's range; pass_param_options prints the
// `option name ...` lines only in SC_PASS_UCI_OPTIONS tuning builds (they are not user options).
bool set_pass_param(std::string_view name, int value);
void pass_param_options(std::ostream& out);

// Per-ply search state, addressed relative to the current node (ss-1 == parent).
// The array lives in Worker::think(); a few slots of margin on both sides make
// (ss-2) and (ss+2) accesses safe at the extremes.
struct Stack {
    Move  killers[2]   = {Move(Move::NO_MOVE), Move(Move::NO_MOVE)};
    Move  current_move = Move(Move::NO_MOVE);  // move that was made at this ply
    Move  excluded     = Move(Move::NO_MOVE);  // move excluded by singular search
    int   moved_piece  = 12;                   // Piece index of current_move (12 == none/null)
    int   moved_to     = 0;                    // destination square of current_move
    Value static_eval  = VALUE_NONE;           // static eval at this node (NONE in check)
    int   ply          = 0;
};

// One legal root move and what the current search knows about it (MultiPV, 3.3).
// `score` is -VALUE_INFINITE for every root move except the PV
// line(s) found so far this iteration, so a STABLE descending sort moves only the new PV to
// the front and keeps every other move's order. `uci_score` is what the info line prints:
// the fail-soft score clipped to the aspiration bound it hit, with the matching inexact_*
// flag set so the line can be tagged lowerbound/upperbound.
struct RootMove {
    explicit RootMove(Move m) : move(m) { pv[0] = m; pv_len = 1; }
    bool operator==(Move m) const noexcept { return move == m; }
    // Sort descending: better score first, ties by the previous iteration's score.
    bool operator<(const RootMove& o) const noexcept {
        return o.score != score ? o.score < score : o.prev_score < prev_score;
    }
    [[nodiscard]] bool is_inexact() const noexcept { return inexact_lower || inexact_upper; }
    void unset_inexact() noexcept { inexact_lower = inexact_upper = false; }

    Move  move;
    Value score            = -VALUE_INFINITE;
    Value prev_score       = -VALUE_INFINITE;
    Value uci_score        = -VALUE_INFINITE;
    bool  inexact_lower    = false;
    bool  inexact_upper    = false;
    bool  prev_score_exact = false;
    int   seldepth         = 0;
    std::array<Move, MAX_PLY + 1> pv{};
    int                            pv_len = 0;
    std::array<Move, MAX_PLY + 1> prev_pv{};
    int                            prev_pv_len = 0;
};

// One search thread's private world. Everything mutable during a search lives
// here except the shared TT and the pool's control flags.
class Worker {
   public:
    Worker(Search& pool, int id) : pool_(pool), id_(id) {}

    // Thread entry point: iterative deepening with aspiration windows.
    void think();

    // Reset per-search state; called by the pool before launching threads.
    void new_search();

   private:
    friend class Search;

    Value negamax(Board& board, Stack* ss, Depth depth, Value alpha, Value beta, bool cut_node);
    Value qsearch(Board& board, Stack* ss, Value alpha, Value beta);
    void  update_stats(const Board& board, Stack* ss, Move best_move, Depth depth,
                       const Move* quiets, int quiet_count, const Move* captures,
                       int capture_count);

    // Poll the shared stop flag; the main worker additionally enforces the
    // node and hard-time limits for everyone.
    [[nodiscard]] bool should_stop();

    // Emit the UCI `info` lines for a completed iteration, one per PV line (main worker only).
    void report_multipv(Depth depth);

    Search& pool_;
    int     id_;

    History history_;

    // Relaxed atomic: each worker only writes its own counter (its own cache
    // line); readers (limit checks, info lines) sum across workers.
    std::atomic<std::uint64_t> nodes_{0};

    int   seldepth_  = 0;
    Depth completed_ = 0;  // deepest fully-completed iteration
    Move  best_move_ = Move(Move::NO_MOVE);
    Value root_score_ = VALUE_NONE;            // final ID score (stm-relative); for gengame
    Move  ponder_move_ = Move(Move::NO_MOVE);  // 2nd PV move (expected reply), for `bestmove ... ponder`

    // Best fail-soft score among all *non-best* root moves of the current
    // iteration (VALUE_NONE == not yet set / only one root move). The gap
    // best - second drives the "only move" early exit and the breadth-mode
    // trigger in think().
    Value root_second_ = -VALUE_INFINITE;

    // Nodes each root move consumed during the current iteration. The best
    // move's share of the total measures how *obvious* it is: when the
    // alternatives refute themselves cheaply the share approaches 1. This is
    // the signal the time manager uses; a score gap cannot serve, because
    // non-best root moves are searched with a null window and return a bound
    // that hugs alpha, carrying no information about how far behind they are.
    std::array<Move, MAX_MOVES>          root_mv_{};
    std::array<std::uint64_t, MAX_MOVES> root_cost_{};
    int                                  root_n_ = 0;

    // MultiPV (3.3): one RootMove per legal root move, kept
    // sorted so the PV lines found this iteration lead; pv_idx_ is the line being
    // searched and multipv_ the effective line count, min(MultiPV, #legal). At MultiPV 1
    // every pv_idx_ gate in the search is inert and the node sequence is unchanged.
    std::vector<RootMove> root_moves_;
    int                   pv_idx_  = 0;
    int                   multipv_ = 1;

    // Snapshot of the pool's search width, taken once per ID iteration so the
    // widening is stable within an iteration.
    int width_ = 0;

    // Pass-eval: per-iteration snapshot of the tunables (plain ints in the hot path, one
    // consistent set per iteration, like width_), and the root's tension for the time manager
    // (VALUE_NONE until the root node computed it; helpers keep their own copy).
    std::array<int, 11> pass_{};
    Value               root_tension_ = VALUE_NONE;

    // Triangular PV table: pv_[ply] holds the PV starting at that ply.
    std::array<std::array<Move, MAX_PLY + 1>, MAX_PLY + 1> pv_{};
    std::array<int, MAX_PLY + 1>                           pv_len_{};
};

// The search pool / public facade. UCI talks to this; it fans a `go` out to
// `Threads` workers and joins them again on stop/quit.
class Search {
   public:
    explicit Search(TranspositionTable& tt) : tt_(tt) { set_threads(kDefaultThreads); }
    ~Search();

    Search(const Search&)            = delete;
    Search& operator=(const Search&) = delete;

    static constexpr int kDefaultThreads = SC_DEFAULT_THREADS;
    static constexpr int kMaxThreads     = 64;

    // Resize the worker pool (joins any running search first).
    void set_threads(int n);

    // Launch an asynchronous search of `root` under `limits` on all workers.
    void start(const Board& root, const SearchLimits& limits);

    // Ask the running search to stop as soon as possible (non-blocking).
    void stop();

    // The pondered move was actually played: stop searching on "free" time and
    // begin enforcing the clock budget (measured from the ponder search's start,
    // so the time already spent counts toward this move). Non-blocking.
    void ponderhit();

    // Block until every worker thread has finished.
    void wait();

    // Reset game-specific state (all workers' histories). Called on `ucinewgame`.
    void new_game();

    [[nodiscard]] bool searching() const noexcept { return searching_.load(std::memory_order_acquire); }

    // In-engine game generator accessors: the main worker's result after wait().
    // best_move == NO_MOVE means the searched position was checkmate/stalemate.
    [[nodiscard]] Move  gen_best_move()  const { return workers_[0]->best_move_; }
    [[nodiscard]] Value gen_root_score() const { return workers_[0]->root_score_; }

   private:
    friend class Worker;

    // Sum of all workers' node counters (approximate while searching).
    [[nodiscard]] std::uint64_t total_nodes() const;

    TranspositionTable& tt_;

    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<std::thread>             threads_;
    // F1: threads_ holds a PERSISTENT pool (spawned in set_threads, joined on shutdown)
    // instead of per-go threads. start() signals the pool; wait() blocks on completion.
    // Avoids creating threads and first-touching each worker's ~1 MB thread_local
    // accumulator stack on every `go`. Search results are unchanged (acc_reset +
    // new_search re-initialise all per-search state at the root), so it is bit-identical.
    std::mutex               pool_mu_;
    std::condition_variable  pool_start_cv_;
    std::condition_variable  pool_done_cv_;
    std::vector<char>        pool_go_;        // per-worker start flag (char, not vector<bool>)
    int                      pool_active_ = 0;// workers still in think()
    bool                     pool_quit_   = false;
    void pool_loop(int id);
    void pool_spawn();
    void pool_shutdown() noexcept;

    std::atomic<bool> stop_{true};        // true == search should unwind now
    std::atomic<bool> searching_{false};  // true between start() and bestmove
    std::atomic<bool> pondering_{false};  // true while searching on the opponent's clock
                                          // (go ponder); time limits are not enforced
                                          // until ponderhit() clears it

    // Adaptive search width (v0.7): a continuous 0..~160 fixed-point value
    // (128 ~= one full ply of LMR widening) set by the main worker between
    // iterations from root-position character (see search.cpp banner); read by
    // every worker at its next iteration. 0 == prune exactly like v0.6.
    std::atomic<int> width_{0};

    Board        root_;
    SearchLimits limits_;
    TimeBudget   budget_;
    TimePoint    start_time_;
};

}  // namespace engine
