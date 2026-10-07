// -----------------------------------------------------------------------------
// main.cpp
//
// Entry point. The engine is a UCI process: it reads commands from stdin and
// writes responses to stdout, so we keep std::cout unbuffered-ish (flushed on
// each protocol line by the handlers) and hand control straight to the UCI loop.
// -----------------------------------------------------------------------------

#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "uci.hpp"

namespace {
// The running executable's own path, from the OS. argv[0] is only what the launcher typed:
// started through PATH it is a bare name that resolves against the working directory, so
// the directory beside the binary (its nets/ and book) would be missed. argv[0] stays the
// fallback where the OS query is unavailable.
std::filesystem::path executable_path(const char* argv0) {
#if defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) == 0) return std::filesystem::path(buf.c_str());
#elif defined(__linux__)
    std::error_code ec;
    const std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) return p;
#elif defined(_WIN32)
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n > 0 && n < buf.size()) return std::filesystem::path(buf.substr(0, n));
#endif
    return std::filesystem::path(argv0);
}
}  // namespace

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
        exe_dir = std::filesystem::canonical(executable_path(argv[0])).parent_path();
    } catch (const std::filesystem::filesystem_error&) {
        exe_dir = std::filesystem::current_path();
    }

    engine::UCI uci(exe_dir);
    uci.loop();
    return 0;
}
