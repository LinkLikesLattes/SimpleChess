#include "embed_net.hpp"

#ifdef SC_EMBED_NET_PATH
// The net file is placed in read-only data at assembly time (.incbin) behind one global label:
// no source-level array, so the tens of megabytes never pass through the compiler. The size is
// a build-time constant (Makefile: wc -c), not an end label -- a Mach-O linker may reorder a
// second label's atom, and a size derived from layout must not exist. Section and symbol
// spelling differ between Mach-O and ELF; COFF (MinGW) is not covered -- no Windows embed.
#ifndef SC_EMBED_NET_SIZE
#error "SC_EMBED_NET_PATH needs SC_EMBED_NET_SIZE (the Makefile sets both from EMBED_NET)"
#endif
#if defined(__APPLE__)
#define SC_EMBED_SECTION ".const_data\n"
#define SC_EMBED_SYM(name) "_" name
#else
#define SC_EMBED_SECTION ".section .rodata\n"
#define SC_EMBED_SYM(name) name
#endif
__asm__(SC_EMBED_SECTION
        ".balign 64\n"
        ".globl " SC_EMBED_SYM("sc_embed_net_data") "\n"
        SC_EMBED_SYM("sc_embed_net_data") ":\n"
        ".incbin \"" SC_EMBED_NET_PATH "\"\n"
        ".text\n");
extern "C" const unsigned char sc_embed_net_data[];

namespace engine::embed {

const unsigned char* net_data() noexcept { return sc_embed_net_data; }
std::size_t          net_size() noexcept { return static_cast<std::size_t>(SC_EMBED_NET_SIZE); }
const char* net_name() noexcept { return SC_EMBED_NET_NAME; }

}  // namespace engine::embed
#else
namespace engine::embed {

const unsigned char* net_data() noexcept { return nullptr; }
std::size_t          net_size() noexcept { return 0; }
const char*          net_name() noexcept { return nullptr; }

}  // namespace engine::embed
#endif
