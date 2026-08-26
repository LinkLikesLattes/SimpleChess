#pragma once

// -----------------------------------------------------------------------------
// version.hpp
//
// Single source of truth for the engine's identity. The version string itself
// lives in the top-level `VERSION` file; the Makefile reads it and injects it
// here as `SC_VERSION` (and the build date as `SC_BUILD_DATE`) at compile time.
// That means every built binary self-reports exactly which version it is via
// the UCI `id name` line — which is what makes automated version testing (see
// tools/version.py and tools/match.py) able to tell builds apart.
//
// The fallbacks below only apply when a translation unit is compiled without
// the Makefile's -D flags (e.g. an ad-hoc single-file compile); a normal
// `make` always defines both.
// -----------------------------------------------------------------------------

#ifndef SC_VERSION
#define SC_VERSION "dev"
#endif

#ifndef SC_BUILD_DATE
#define SC_BUILD_DATE "unknown"
#endif

namespace engine {

// The bare version, e.g. "0.1". Matches the top-level VERSION file.
inline constexpr char kEngineVersion[] = SC_VERSION;

// The date this binary was compiled, "YYYY-MM-DD".
inline constexpr char kEngineBuildDate[] = SC_BUILD_DATE;

// Reported in the UCI `id author` line.
inline constexpr char kEngineAuthor[] = "Sam Moore";

// Reported in the UCI `id name` line — includes the version so that two
// different builds are distinguishable by a GUI or a match runner. The "NNUE"
// tag also keeps this fork distinct from the hand-crafted-eval SimpleChess in
// gauntlets that run both side by side.
inline constexpr char kEngineName[] = "SimpleChess " SC_VERSION;

}  // namespace engine
