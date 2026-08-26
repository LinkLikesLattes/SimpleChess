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
#include <thread>
#include <vector>

#include "history.hpp"
#include "timeman.hpp"
#include "tt.hpp"

// Build-time overridable UCI defaults. The native build keeps the originals; the
// Windows/CCRL cross-build (tools/build_windows.sh) sets Hash 256 / Threads 1.
// SC_DEFAULT_HASH is in MB.
#ifndef SC_DEFAULT_HASH
#define SC_DEFAULT_HASH 4096
#endif
#ifndef SC_DEFAULT_THREADS
#define SC_DEFAULT_THREADS 8
#endif
#include "types.hpp"

namespace engine {

class Search;

// Set the root-move noise magnitude in centipawns (0 = off, normal play). Used
// by the self-play data generator to diversify games; see search.cpp.
void set_root_noise(int cp);

// Silence all UCI output (info/bestmove) from the search. Used by the in-engine
// game generator (`gengame`) so its many per-ply searches don't spam stdout.
void set_gen_silent(bool silent);

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

    // Emit one UCI `info` line for a completed iteration (main worker only).
    void report(Depth depth, Value score);

    [[nodiscard]] std::string pv_string() const;

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

    // Snapshot of the pool's search width, taken once per ID iteration so the
    // widening is stable within an iteration.
    int width_ = 0;

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
