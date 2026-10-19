// -----------------------------------------------------------------------------
// tune.hpp
//
// Search constants that SPSA tunes. A normal build (SC_TUNE=0) makes each one a
// constexpr int, so the generated code is what the literal would give. A tuning
// build (-DSC_TUNE=1) makes each a plain int registered under its name minus the
// leading 'k' (kRazorMargin -> RazorMargin): `uci` advertises it as a spin option
// and setoption changes it between searches, never during one. Both engines of a
// tuning match run the same binary, so the cost of reading the values from memory
// cancels out.
// -----------------------------------------------------------------------------
#pragma once

#include <ostream>
#include <string_view>

#ifndef SC_TUNE
#define SC_TUNE 0
#endif

namespace engine::tune {

// Register a tunable (tuning builds call this at static initialisation).
bool add(const char* name, int* value, int lo, int hi);

// Called after every successful set(), e.g. to rebuild a table derived from tunables.
void on_change(void (*fn)());

// Set a tunable by name (case-insensitive), clamped to its range. False for an
// unknown name, which in a normal build is every name.
bool set(std::string_view name, int value);

// The `option name ...` lines of the registered tunables (none in a normal build).
void options(std::ostream& out);

}  // namespace engine::tune

#if SC_TUNE
#define SC_PARAM(var, def, lo, hi)                                                   \
    int        var = def;                                                            \
    const bool var##_registered = ::engine::tune::add(&#var[1], &var, lo, hi);
#else
#define SC_PARAM(var, def, lo, hi) constexpr int var = def;
#endif
