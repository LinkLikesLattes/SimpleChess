#pragma once

// -----------------------------------------------------------------------------
// uci.hpp
//
// The UCI (Universal Chess Interface) front end. It owns the engine's persistent
// state — the transposition table, the current board, and the search worker —
// and translates the text protocol a GUI speaks into calls on those objects.
//
// The design keeps the main (reader) thread free of search work: `go` hands off
// to the Search worker thread, so `stop`, `isready`, and `quit` remain responsive
// while a search is in flight.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <filesystem>
#include <sstream>
#include <string>

#include "book.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "types.hpp"
#include "version.hpp"  // kEngineName / kEngineAuthor / kEngineVersion

namespace engine {

class UCI {
   public:
    // `exe_dir` anchors the default opening-book search (see try_load_default_book).
    explicit UCI(std::filesystem::path exe_dir);

    // Read commands from stdin until `quit` (or EOF). Blocks the calling thread.
    void loop();

   private:
    void handle_uci() const;
    void handle_isready() const;
    void handle_newgame();
    void handle_setoption(std::istringstream& is);
    void handle_position(std::istringstream& is);
    void handle_go(std::istringstream& is);
    void handle_gengame(std::istringstream& is);  // in-engine self-play game generation
    void handle_print() const;

    // Look for a book at, in order: <exe_dir>/books/book.bin, <exe_dir>/book.bin,
    // ./books/book.bin, ./book.bin. Leaves the book unloaded (silently — no
    // book is a normal, supported state) if none of those exist.
    void try_load_default_book();
    void try_load_default_net();

    TranspositionTable tt_;
    Search             search_;
    Board              board_;
    Book               book_;
    std::filesystem::path exe_dir_;

    // Options (mirrors the `option` lines emitted on `uci`).
    std::size_t hash_mb_       = SC_DEFAULT_HASH;     // "Hash", in MB
    int         threads_       = SC_DEFAULT_THREADS;  // "Threads" — Lazy SMP worker count
    int         move_overhead_ = 30;    // "Move Overhead", in ms
    bool        ponder_        = false; // "Ponder"
    bool        own_book_      = true;  // "OwnBook" — use the opening book when loaded
};

}  // namespace engine
