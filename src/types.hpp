#pragma once

// -----------------------------------------------------------------------------
// types.hpp
//
// Engine-wide type aliases, score conventions, and tunable limits. Everything
// downstream (search, evaluation, transposition table, time management) speaks
// in terms of the definitions here, so this is the first place to look when
// tuning the engine.
//
// The engine is built on top of Disservin's chess-library, which provides the
// board representation, (bit)board utilities, and legal move generation. We pull
// it in here once and re-export the handful of names the rest of the engine
// uses so translation units don't each reach into the `chess` namespace.
// -----------------------------------------------------------------------------

#include <cstdint>

#include "chess.hpp"

namespace engine {

// Re-export the library primitives we lean on the most. Keeping these aliases in
// one spot means the rest of the codebase is insulated from library renames.
using Board     = chess::Board;
using Move      = chess::Move;
using Movelist  = chess::Movelist;
using Color     = chess::Color;
using PieceType = chess::PieceType;
using Square    = chess::Square;
using Bitboard  = chess::Bitboard;

// A search score, in centipawns from the side-to-move's point of view.
// 32-bit keeps the mate arithmetic below comfortably away from overflow.
using Value = std::int32_t;

// A search depth in plies. Signed so that reductions/extensions can transiently
// push a local depth below zero without underflow surprises.
using Depth = std::int32_t;

// A zobrist position key, as produced by Board::hash().
using Key = std::uint64_t;

// ---- Score conventions -------------------------------------------------------
//
// Scores live in a symmetric window around zero. `VALUE_INFINITE` is the alpha/
// beta sentinel and must stay strictly larger than any real evaluation or mate
// score so that fail-high/fail-low comparisons behave.
//
// Mate scores are encoded as `VALUE_MATE - ply`, so a shorter mate is preferred
// over a longer one and the distance-to-mate can be recovered for UCI output.
constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_INFINITE = 32000;
constexpr Value VALUE_NONE     = 32001;  // "no score yet" sentinel

constexpr Value VALUE_MATE             = 31000;
constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - 1000;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

// ---- Search limits -----------------------------------------------------------

// Hard ceiling on search depth / recursion. Sizes stack-allocated per-ply arrays.
constexpr int MAX_PLY   = 246;
// Upper bound on legal moves in any position (library uses 256 internally).
constexpr int MAX_MOVES = 256;

// Convenience: is `v` a mate-or-mated score (as opposed to a normal eval)?
[[nodiscard]] constexpr bool is_mate_score(Value v) noexcept {
    return v >= VALUE_MATE_IN_MAX_PLY || v <= VALUE_MATED_IN_MAX_PLY;
}

// Adjust a mate score for the current distance from the root. Search stores
// scores relative to the current node; the TT and UCI output want them relative
// to the root, hence these two helpers.
[[nodiscard]] constexpr Value mate_in(int ply) noexcept { return VALUE_MATE - ply; }
[[nodiscard]] constexpr Value mated_in(int ply) noexcept { return -VALUE_MATE + ply; }

}  // namespace engine
