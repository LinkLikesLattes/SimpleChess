#pragma once

// -----------------------------------------------------------------------------
// io.hpp
//
// stdout is shared by two threads: the UCI thread (uciok, readyok, info strings)
// and the search's main worker (info lines, bestmove). main() turns off the
// stdio sync for speed, and with it the stream does no locking of its own, so
// two concurrent writes can interleave bytes inside a line. Every write goes
// through one lock:
//
//     sync_cout << "readyok" << sync_endl;      // one statement = one locked write
//
// `sync_cout` takes the lock and `sync_endl` ends the line, flushes and releases
// it, so the pair must always appear in the same statement. A block of several
// lines holds `io_mutex()` with a lock_guard instead.
// -----------------------------------------------------------------------------

#include <iostream>
#include <mutex>

namespace engine {

[[nodiscard]] inline std::mutex& io_mutex() {
    static std::mutex m;
    return m;
}

enum class IoSync { Lock, Unlock };

inline std::ostream& operator<<(std::ostream& os, IoSync s) {
    if (s == IoSync::Lock) io_mutex().lock();
    else io_mutex().unlock();
    return os;
}

}  // namespace engine

#define sync_cout std::cout << engine::IoSync::Lock
#define sync_endl std::endl << engine::IoSync::Unlock
