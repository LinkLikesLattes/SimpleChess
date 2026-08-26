#pragma once

// -----------------------------------------------------------------------------
// nnue.hpp
//
// HalfKP NNUE evaluation for SimpleChessNNUE v0.1. Sits behind the same
// side-to-move-relative centipawn contract as the hand-crafted eval, so search
// consumes it unchanged. The network file (.scn) is produced by
// train/export.py; the feature indexing and cp = 400 * logit output scaling
// must match train/data.py and train/model.py exactly (see memory:
// nnue-encoding-conventions).
//
// v0.1 does a full accumulator recompute per call in float — simple and exactly
// matches the Python trainer. Incremental updates and int8/int16 quantization
// are later speed passes.
// -----------------------------------------------------------------------------

#include <string>

#include "types.hpp"

namespace engine::nnue {

// Load a .scn network from disk. Returns true on success; on failure the engine
// keeps whatever eval it had (HCE) and this returns false.
bool load(const std::string& path);

// True once a network is loaded and evaluate() will use it.
[[nodiscard]] bool loaded() noexcept;

// Static evaluation in centipawns from the side-to-move's POV. Precondition:
// loaded() is true.
[[nodiscard]] Value evaluate(const Board& board);

// -----------------------------------------------------------------------------
// Incremental accumulator (thread-local). The search maintains it around
// make/unmake so evaluate() reads a persisted BASE accumulator instead of
// rebuilding it from scratch every call. If these are never called, evaluate()
// transparently falls back to full recompute (correctness preserved), so the
// hooks can be added incrementally. Each search thread owns its own stack.
// -----------------------------------------------------------------------------

// Full refresh at `board`, resetting this thread's stack to ply 0. Call at the
// top of each search (root) before descending.
void acc_reset(const Board& board);

// Push an incremental update for move `m` about to be made on `before`. Call
// immediately BEFORE board.makeMove(m).
void acc_make(const Board& before, Move m);

// Pop the last pushed accumulator. Call immediately AFTER board.unmakeMove(m).
void acc_unmake();

// Null move: position unchanged, only side-to-move flips. Call around
// makeNullMove / unmakeNullMove.
void acc_make_null();
void acc_unmake_null();

}  // namespace engine::nnue
