#pragma once

// -----------------------------------------------------------------------------
// timeman.hpp
//
// Time management. Turns the raw `go` clock parameters into a soft/hard budget
// the search polls against. The policy here is a simple, safe baseline; it is
// isolated so it can be made smarter (stability-based extensions, etc.) without
// disturbing the search loop.
// -----------------------------------------------------------------------------

#include <chrono>
#include <cstdint>

#include "types.hpp"

namespace engine {

using TimePoint = std::chrono::steady_clock::time_point;

[[nodiscard]] inline TimePoint now() noexcept { return std::chrono::steady_clock::now(); }

[[nodiscard]] inline std::int64_t elapsed_ms(TimePoint since) noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(now() - since).count();
}

// The parsed contents of a UCI `go` command. Milliseconds throughout; a value of
// 0 for a clock field means "not provided". `movetime`/`depth`/`nodes` of 0 and
// `infinite == false` means "use the clock-based budget".
struct SearchLimits {
    std::int64_t time[2]    = {0, 0};  // remaining time for [WHITE], [BLACK]
    std::int64_t inc[2]     = {0, 0};  // increment for [WHITE], [BLACK]
    std::int64_t movetime   = 0;       // fixed time for this move
    int          movestogo  = 0;       // moves until next time control (0 == sudden death)
    Depth        depth      = 0;       // fixed depth
    std::uint64_t nodes     = 0;       // node cap
    int          mate       = 0;       // search for a mate in `mate` moves
    bool         infinite   = false;   // search until `stop`
    bool         ponder     = false;   // pondering (search on opponent's time)

    [[nodiscard]] bool uses_time_control() const noexcept {
        return !infinite && depth == 0 && nodes == 0 && movetime == 0 && mate == 0;
    }
};

// A concrete budget for one search, derived from SearchLimits at `go` time.
struct TimeBudget {
    bool         use_clock = false;  // false == depth/nodes/infinite bound, ignore times
    std::int64_t soft_ms   = 0;      // stop starting new iterations past this
    std::int64_t hard_ms   = 0;      // abort the search immediately past this
};

// Compute the time budget for the side to move. `game_ply` is the number of
// plies played from the start of the game (0 on move 1) — top engines spend a
// smaller fraction of the clock early and ramp up into the middlegame, so the
// allocation scales with it.
[[nodiscard]] TimeBudget compute_budget(const SearchLimits& limits, Color stm, int game_ply);

}  // namespace engine
