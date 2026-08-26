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
// than min(60s, 10% of the remaining clock) — see kMaxMoveMs / kMaxMoveFraction.
// -----------------------------------------------------------------------------

#include "timeman.hpp"

#include <algorithm>
#include <cmath>

namespace engine {

namespace {
// Safety margin (ms) subtracted from any allotment to cover GUI/network lag and
// the cost of actually emitting the move.
constexpr double kMoveOverhead = 30.0;

// Project-level hard ceiling on a single move: never more than 60 s, and never
// more than this fraction of the remaining clock — whichever is smaller.
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

    // Depth-, node-, mate-limited or infinite searches are not clock-bound.
    if (!limits.uses_time_control() && limits.movetime == 0) {
        budget.use_clock = false;
        return budget;
    }

    budget.use_clock = true;

    // Fixed move time: spend (almost) all of it, both thresholds equal.
    if (limits.movetime > 0) {
        const std::int64_t t =
            std::max<std::int64_t>(1, limits.movetime - static_cast<std::int64_t>(kMoveOverhead));
        budget.soft_ms = t;
        budget.hard_ms = t;
        return budget;
    }

    const int    us        = static_cast<int>(stm);
    const double time_left = static_cast<double>(limits.time[us]);
    const double inc       = static_cast<double>(limits.inc[us]);

    // Degenerate/absent clock: fall back to a tiny fixed think so we still move.
    if (time_left <= 0.0) {
        budget.soft_ms = budget.hard_ms = 50;
        return budget;
    }

    // moves-to-go: cap at 50 so a distant time control doesn't make us hoard;
    // 0 (from the caller) means sudden death / increment only.
    const int    mtg = limits.movestogo > 0 ? std::min(limits.movestogo, 50) : 50;
    const double ply = static_cast<double>(game_ply);

    // Effective time we may plan to consume before the next control, keeping a
    // per-move overhead in reserve for every move until then.
    const double time_for_control =
        std::max(1.0, time_left + inc * (mtg - 1) - kMoveOverhead * (2 + mtg));

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
        std::max(optimum, std::min(0.8097 * time_left - kMoveOverhead, max_scale * optimum));

    // ---- Project hard cap: <= min(60s, fraction of remaining) ----
    const double cap = std::min(kMaxMoveMs,
                                (no_increment ? kNoIncMaxFraction : kMaxMoveFraction) * time_left);
    optimum          = std::min(optimum, cap);
    maximum          = std::min(maximum, cap);

    budget.soft_ms = static_cast<std::int64_t>(optimum);
    budget.hard_ms = static_cast<std::int64_t>(std::max(optimum, maximum));
    return budget;
}

}  // namespace engine
