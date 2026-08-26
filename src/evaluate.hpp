#pragma once

// -----------------------------------------------------------------------------
// evaluate.hpp
//
// Hand-crafted evaluation (HCE) entry point. This is the surface a solid HCE is
// meant to grow behind: today it returns a tapered material + piece-square-table
// score so the search plays sensible, legal chess, but the intent is for terms
// like mobility, king safety, pawn structure, and threats to be layered in here
// without any change to the search that consumes it.
//
// Contract: evaluate() returns a score in centipawns from the perspective of the
// side to move (positive == good for the mover). Search negates across plies, so
// keeping this side-relative is essential.
// -----------------------------------------------------------------------------

#include "types.hpp"

namespace engine::eval {

// Static evaluation of a quiet-ish position. Must be symmetric: evaluating a
// mirrored position for the other side should return the negated score.
[[nodiscard]] Value evaluate(const Board& board);

// Called once at startup so the evaluation can precompute tables (PSTs, etc.).
// Safe to call more than once.
void init();

// Material-blend knob (see nnue.hpp). When an NNUE net is loaded and stands
// alone (NNUEWeight == 100), the final eval becomes
//   nnue_cp + pct% * psqt_material_cp.
// Default is 90 (the pure-result net under-weights material, so it is anchored
// at 0.90x); pct == 0 recovers pure NNUE. Tune via the "MaterialBlend" UCI
// option. Not applied in the fusion path (weight < 100), where HCE already
// supplies material and PST.
void set_material_blend(int pct);

// Simple side-to-move-relative material score in centipawns (piece values only).
// Positionally blind; kept for comparison against the PSQT anchor below.
[[nodiscard]] Value material(const Board& board);

// Stockfish's PSQT + piece values (SF14 psqt.cpp / types.h), tapered and
// rescaled to this engine's centipawn unit. Side-to-move-relative. This is the
// anchor used to mediate a standalone NNUE.
[[nodiscard]] Value psqt_material(const Board& board);

// HCE/NNUE fusion knobs (percent). With a net loaded, the eval becomes
//   ((100 - weight) * HCE + weight * (scale * NNUE / 100)) / 100.
// weight defaults to 100 (pure NNUE + material anchor); scale defaults to 100.
// The self-play data-gen sets weight=20, scale=30 -> 0.80*HCE + 0.06*NNUE.
void set_nnue_weight(int pct);
void set_nnue_scale(int pct);

}  // namespace engine::eval
