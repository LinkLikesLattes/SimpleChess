// -----------------------------------------------------------------------------
// timeman.cpp
//
// Time allocation (v0.6). Two thresholds are produced:
//   * soft_ms  — the *optimum*: once exceeded, don't begin another
//     iterative-deepening pass.
//   * hard_ms  — the *maximum*: a single move may borrow up to here mid-search,
//     but never past it.
//
// The key idea vs. the old flat "remaining / 30" policy: the fraction of the
// clock spent scales with `game_ply`. Early moves (a mostly-booked, low-
// branching opening the engine already blitzes to high depth) get a *small*
// slice via `pow(ply + c, e)`; the slice grows into the middlegame where the
// thinking matters. That directly fixes "burns 30s reaching depth 30 on move 2".
//
// On top of that formula sits a hard project cap: a move never spends more
// than 60 s, and in sudden death / increment play never more than 10% of the
// remaining clock — see kMaxMoveMs / kMaxMoveFraction. (A moves-to-go control is
// already rationed by its move count: with `movestogo 1` the move may use most of
// the clock, because the rest would be unused when the control is reached.)
// -----------------------------------------------------------------------------

#include "timeman.hpp"

#include <algorithm>
#include <cmath>

namespace engine {

namespace {
// Safety margin (ms) subtracted from any allotment to cover GUI/network lag and
// the cost of actually emitting the move: the UCI `Move Overhead` option, carried in
// SearchLimits::move_overhead_ms (default 30). Read per call below, never hard-coded.

// Project-level hard ceiling on a single move: never more than 60 s, and in sudden
// death / increment play never more than this fraction of the remaining clock.
constexpr double kMaxMoveMs       = 60'000.0;
constexpr double kMaxMoveFraction = 0.10;

// ---- No-increment safety (v0.9) ----
// With no increment, your whole clock has to last the entire game, and bullet
// games routinely run 50-80 moves. The generous sudden-death allocation (~2 s a
// move at 60 s) burns through it, and a single worst-case move could otherwise
// borrow up to 10% of the clock. For inc == 0 we therefore (1) spend a smaller
// slice per move and (2) clamp both the maximum multiplier and the per-move
// ceiling hard, so no single move can gut the flag.
constexpr double kNoIncOptFactor  = 0.70;  // spend ~30% less per move
constexpr double kNoIncMaxScale   = 2.00;  // maximum <= 2x optimum
constexpr double kNoIncMaxFraction = 0.05; // ...and <= 5% of the remaining clock
}  // namespace

TimeBudget compute_budget(const SearchLimits& limits, Color stm, int game_ply) {
    TimeBudget budget;

    // Infinite analysis ignores any clock. Otherwise a given clock always applies (depth, nodes
    // and mate limits are enforced by the search on top of it); a search bounded only by depth,
    // nodes or mate is not clock-bound.
    const bool bounded = limits.depth > 0 || limits.nodes > 0 || limits.mate > 0;
    const bool clock   = limits.movetime > 0 || limits.time[static_cast<int>(stm)] > 0;
    if (limits.infinite || (bounded && !clock)) {
        budget.use_clock = false;
        return budget;
    }

    budget.use_clock = true;

    // Fixed move time: spend (almost) all of it, both thresholds equal.
    if (limits.movetime > 0) {
        const std::int64_t t =
            std::max<std::int64_t>(1, limits.movetime - static_cast<std::int64_t>(limits.move_overhead_ms));
        budget.soft_ms = t;
        budget.hard_ms = t;
        return budget;
    }

    const int    us        = static_cast<int>(stm);
    const double time_left = static_cast<double>(limits.time[us]);
    const double inc       = static_cast<double>(limits.inc[us]);
    const double overhead  = static_cast<double>(limits.move_overhead_ms);

    // Degenerate/absent clock: fall back to a tiny fixed think so we still move.
    if (time_left <= 0.0) {
        budget.soft_ms = budget.hard_ms = 50;
        return budget;
    }

    // moves-to-go: cap at 50 so a distant time control doesn't make us hoard;
    // 0 (from the caller) means sudden death / increment only.
    int mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 50) : 50;
    // Sudden death: 50 is only a guess at the moves left, and when the overhead exceeds the
    // increment the reserve below, (2 + mtg) x overhead - (mtg - 1) x inc, swallows the whole
    // clock at low time (with no increment once time_left < 52 x overhead: 1.56 s at 30 ms,
    // 5.2 s at 100 ms), leaving ~1 ms a move. Shorten the horizon just enough that the reserve
    // never takes more than half the remaining clock. A real moves-to-go count is never
    // shortened, and with inc >= overhead the reserve never grows, so nothing changes there.
    if (limits.movestogo == 0 && overhead > inc) {
        const double fit = (0.5 * time_left - inc - 2.0 * overhead) / (overhead - inc);
        mtg = std::clamp(static_cast<int>(fit), 1, mtg);
    }
    const double ply = static_cast<double>(game_ply);

    // Effective time we may plan to consume before the next control, keeping a
    // per-move overhead in reserve for every move until then.
    const double time_for_control =
        std::max(1.0, time_left + inc * (mtg - 1) - overhead * (2 + mtg));

    double opt_scale, max_scale;

    if (limits.movestogo == 0) {
        // Sudden death / increment.
        const double log_time_sec = std::log10(std::max(1.0, time_left) / 1000.0);
        const double opt_constant = std::min(0.0029869 + 0.00033554 * log_time_sec, 0.004905);
        const double max_constant = std::max(3.3744 + 3.0608 * log_time_sec, 3.1441);

        opt_scale = std::min(0.012112 + std::pow(ply + 3.22713, 0.46866) * opt_constant,
                             0.19404 * time_left / time_for_control);
        max_scale = std::min(6.873, max_constant + ply / 12.352);
    } else {
        // "x moves in y seconds": spend an even slice, gently ply-weighted.
        opt_scale = std::min((0.88 + ply / 116.4) / mtg, 0.88 * time_left / time_for_control);
        max_scale = 1.3 + 0.11 * mtg;
    }

    // No-increment: ration harder and forbid a single move from spiking (a
    // fail-high re-search must not borrow 6 s at 60 s no-inc — that flags).
    const bool no_increment = (limits.movestogo == 0 && inc <= 0.0);
    if (no_increment) {
        opt_scale *= kNoIncOptFactor;
        max_scale = std::min(max_scale, kNoIncMaxScale);
    }

    double optimum = std::max(1.0, opt_scale * time_for_control);
    double maximum =
        std::max(optimum, std::min(0.8097 * time_left - overhead, max_scale * optimum));

    // ---- Project hard cap: <= 60 s, and in sudden death <= a fraction of remaining ----
    const double cap = limits.movestogo > 0
                           ? kMaxMoveMs
                           : std::min(kMaxMoveMs, (no_increment ? kNoIncMaxFraction : kMaxMoveFraction) * time_left);
    optimum          = std::min(optimum, cap);
    maximum          = std::min(maximum, cap);

    budget.soft_ms = static_cast<std::int64_t>(optimum);
    budget.hard_ms = static_cast<std::int64_t>(std::max(optimum, maximum));
    return budget;
}

}  // namespace engine
