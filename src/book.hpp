#pragma once

// -----------------------------------------------------------------------------
// book.hpp
//
// Polyglot (.bin) opening book support. A book maps a position hash to one or
// more candidate moves with relative weights; probing is a lookup, not search,
// so book moves cost effectively zero time. The engine consults the book on
// every `go` and falls back to normal search the moment a position isn't
// covered — "stays in book until it runs out" needs no explicit mode switch,
// since a probe miss is itself the signal to start searching.
//
// The position hash used here is intentionally a standalone implementation
// (see book.cpp) rather than a reuse of Board::hash(): it doesn't depend on
// which make/unmake variant built the board, so book lookups are correct
// regardless of how the current position was reached (FEN, UCI `moves`, or
// mid-search). See polyglot_random.hpp for the constants it's built from.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

#include "types.hpp"

namespace engine {

class Book {
   public:
    // Load a Polyglot book from `path`. Replaces any previously loaded book.
    // Returns false (and leaves the book unloaded) if the file is missing,
    // empty, or not a well-formed sequence of 16-byte Polyglot entries —
    // callers are expected to treat that as "no book" and play normally.
    bool load(const std::filesystem::path& path);

    // Discard the loaded book, if any.
    void unload();

    [[nodiscard]] bool loaded() const noexcept { return !entries_.empty(); }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    // Weighted-random choice among the book's legal moves for `board`, or
    // NO_MOVE if unloaded / the position has no entry (book exhausted).
    [[nodiscard]] Move probe(const Board& board);

   private:
    struct Entry {
        std::uint64_t key;
        std::uint16_t move;
        std::uint16_t weight;
    };

    std::vector<Entry>   entries_;  // sorted ascending by key for binary search
    std::filesystem::path path_;
    std::mt19937_64       rng_{std::random_device{}()};
};

}  // namespace engine
