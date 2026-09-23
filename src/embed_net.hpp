#pragma once
// Optional network embedded in the executable (Makefile EMBED_NET=<path>): a single file that
// runs with no net beside it, for phone front-ends and tournament harnesses that copy the
// binary alone. Without EMBED_NET the accessors report "nothing embedded" and discovery
// (uci.cpp) behaves exactly as before.
#include <cstddef>

namespace engine::embed {

const unsigned char* net_data() noexcept;  // nullptr when nothing is embedded
std::size_t          net_size() noexcept;  // 0 when nothing is embedded
const char*          net_name() noexcept;  // the embedded file's name (dated, so discovery can rank it)

}  // namespace engine::embed
