// -----------------------------------------------------------------------------
// main.cpp
//
// Entry point. The engine is a UCI process: it reads commands from stdin and
// writes responses to stdout, so we keep std::cout unbuffered-ish (flushed on
// each protocol line by the handlers) and hand control straight to the UCI loop.
// -----------------------------------------------------------------------------

#include <filesystem>
#include <iostream>

#include "uci.hpp"

int main(int /*argc*/, char** argv) {
    // A GUI expects timely, line-buffered replies; untie cin/cout so a pending
    // input read never delays our output.
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Anchor the default opening-book search to the executable's own
    // directory, not the (GUI-controlled) working directory, so a book placed
    // alongside the binary is found no matter where the engine is launched from.
    std::filesystem::path exe_dir;
    try {
        exe_dir = std::filesystem::canonical(std::filesystem::path(argv[0])).parent_path();
    } catch (const std::filesystem::filesystem_error&) {
        exe_dir = std::filesystem::current_path();
    }

    engine::UCI uci(exe_dir);
    uci.loop();
    return 0;
}
