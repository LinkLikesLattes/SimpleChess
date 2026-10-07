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
    void handle_revgame(std::istringstream& is);  // in-engine reverse-analysis labelling of one game
    void handle_perft(std::istringstream& is);    // debug: bulk-counted perft of the current position
    void handle_print() const;

    // Parse a FEN into `out` in the current mode: the plain FEN parser when UCI_Chess960 is
    // off (K/Q/k/q only, as before 3.2), the X-FEN parser when it is on (K/Q = the outermost
    // rook, A-H/a-h Shredder file letters, sets the board's chess960 flag). `out` is only
    // assigned on success; a FEN without both kings is rejected.
    bool set_fen(Board& out, std::string_view fen) const;

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
    bool        own_book_      = false; // "OwnBook" — opt-in; no book ships or loads by default
    bool        chess960_      = false; // "UCI_Chess960" — read by the NEXT position/gengame
                                        // (lazy, as GUIs expect); never re-sets the current board
    int         multipv_       = 1;     // "MultiPV" — principal variations to report (analysis)
};

}  // namespace engine
