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

#include <cstddef>
#include <string>

#include "types.hpp"

namespace engine::nnue {

// Load a .scn5 network from disk. Returns true on success and false on failure.
// On failure the engine keeps the previously loaded net, if any. There is no
// hand-crafted-eval fallback in 3.0.0: startup aborts if no net can be loaded at
// all, so once running, a net is always present.
bool load(const std::string& path);

// Same loader over an in-memory image of a net file (the embedded net, embed_net.hpp).
bool load_memory(const void* data, std::size_t size);

// True once a network is loaded and evaluate() will use it.
[[nodiscard]] bool loaded() noexcept;

// Static evaluation in centipawns from the side-to-move's POV. Precondition:
// loaded() is true.
[[nodiscard]] Value evaluate(const Board& board);

// Static eval of the position with the side to move flipped (the position after a pass),
// returned from the ORIGINAL side to move's point of view: "our score if we had to pass".
// Reads the same accumulator (and the same output bucket) as evaluate(); only the L1 input
// order and the sign differ, so it costs pairwise + L1 + body and no accumulator work.
// Precondition: loaded(), and the side to move is NOT in check (the flipped position would
// be illegal and the value meaningless). See dev-notes/NOVELTY_CANDIDATES.md #1.
[[nodiscard]] Value evaluate_pass(const Board& board);

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

// L1 norm of the accumulator change made by the LAST acc_make(), summed over both
// perspectives: "how much did this move change what the net sees". Valid only between
// acc_make() and the matching acc_unmake(); returns 0 if the incremental stack is not in
// sync or the float (labeler) net is loaded. Free of search state, so it is safe to call
// from move-loop code. Raw sum, not normalised: the caller divides by a tuned constant.
[[nodiscard]] int acc_delta_l1() noexcept;

// Item #9: number of NEW attacks on a higher-valued enemy piece that the last acc_make()'s move
// created, for the mover's side. Valid between acc_make() and its acc_unmake(); 0 on the king-cross
// refresh path and for the float labeler net.
[[nodiscard]] int acc_threats_made() noexcept;

}  // namespace engine::nnue
