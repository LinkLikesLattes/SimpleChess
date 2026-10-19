// -----------------------------------------------------------------------------
// tune.cpp
//
// Registry of the SC_PARAM tunables (see tune.hpp). Empty in a normal build.
// -----------------------------------------------------------------------------

#include "tune.hpp"

#include <algorithm>
#include <cctype>
#include <vector>

namespace engine::tune {

namespace {

struct Param {
    const char* name;
    int*        value;
    int         lo, hi;
};

// Function-local, so registration from another translation unit's static
// initialisers never runs before the vector exists.
std::vector<Param>& registry() {
    static std::vector<Param> r;
    return r;
}

void (*g_on_change)() = nullptr;

[[nodiscard]] bool iequals(std::string_view a, std::string_view b) noexcept {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::tolower(static_cast<unsigned char>(x)) ==
                      std::tolower(static_cast<unsigned char>(y));
           });
}

}  // namespace

bool add(const char* name, int* value, int lo, int hi) {
    registry().push_back({name, value, lo, hi});
    return true;
}

void on_change(void (*fn)()) { g_on_change = fn; }

bool set(std::string_view name, int value) {
    for (const Param& p : registry()) {
        if (iequals(p.name, name)) {
            *p.value = std::clamp(value, p.lo, p.hi);
            if (g_on_change) g_on_change();
            return true;
        }
    }
    return false;
}

void options(std::ostream& out) {
    for (const Param& p : registry())
        out << "option name " << p.name << " type spin default " << *p.value << " min " << p.lo
            << " max " << p.hi << "\n";
}

}  // namespace engine::tune
