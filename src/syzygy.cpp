// -----------------------------------------------------------------------------
// syzygy.cpp — Fathom bridge (see syzygy.hpp).
// -----------------------------------------------------------------------------
#include "syzygy.hpp"

#include <algorithm>
#include <atomic>

// Vendored Fathom is compiled as a SINGLE translation unit right here: its
// tbprobe.h defines tb_probe_wdl/tb_probe_root in the header body, so exactly one
// object may include it — this one. tbprobe.c pulls in tbchess.c + tbprobe.h.
// Warnings are silenced for the third-party code only; the bridge below keeps the
// normal -Wall -Wextra. The header's own `extern "C"` guards give the symbols C
// linkage, matching callers that include just tbprobe.h.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif
#include "tbprobe.c"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
// tbchess.c defines function-like min/max macros that would clobber std::min/max
// (the preprocessor expands `min` even in `std::min`). Drop them before our code.
#undef min
#undef max

namespace engine::syzygy {

namespace {
std::atomic<int>           g_probe_depth{1};
std::atomic<int>           g_probe_limit{7};  // user cap; effective = min(this, TB_LARGEST)
std::atomic<bool>          g_fifty_rule{true};
std::atomic<std::uint64_t> g_hits{0};
}  // namespace

int init(const std::string& path) {
    tb_free();
    if (path.empty() || path == "<empty>") return 0;
    if (!tb_init(path.c_str())) return 0;
    return static_cast<int>(TB_LARGEST);
}

void teardown() { tb_free(); }

int max_men() { return static_cast<int>(TB_LARGEST); }

int probe_limit() {
    return std::min(g_probe_limit.load(std::memory_order_relaxed), static_cast<int>(TB_LARGEST));
}

int probe_depth() { return g_probe_depth.load(std::memory_order_relaxed); }

void set_probe_depth(int d) { g_probe_depth.store(d < 0 ? 0 : d, std::memory_order_relaxed); }
void set_probe_limit(int men) { g_probe_limit.store(men < 0 ? 0 : men, std::memory_order_relaxed); }
void set_fifty_rule(bool on) { g_fifty_rule.store(on, std::memory_order_relaxed); }

std::uint64_t hits() { return g_hits.load(std::memory_order_relaxed); }
void reset_hits() { g_hits.store(0, std::memory_order_relaxed); }

std::optional<Value> probe_wdl(const Board& board, int ply) {
    if (TB_LARGEST == 0) return std::nullopt;
    // Fathom's WDL helper requires no castling rights and a zero halfmove clock
    // (a WDL win with rule50 > 0 could be a 50-move draw). Both hold exactly at
    // the quiet, capture/pawn-move nodes we want to trust.
    const auto cr = board.castlingRights();
    if (cr.has(Color::WHITE) || cr.has(Color::BLACK)) return std::nullopt;
    if (board.halfMoveClock() != 0) return std::nullopt;

    const Square   ep  = board.enpassantSq();
    const unsigned res = tb_probe_wdl(
        board.us(Color::WHITE).getBits(), board.us(Color::BLACK).getBits(),
        board.pieces(PieceType::KING).getBits(), board.pieces(PieceType::QUEEN).getBits(),
        board.pieces(PieceType::ROOK).getBits(), board.pieces(PieceType::BISHOP).getBits(),
        board.pieces(PieceType::KNIGHT).getBits(), board.pieces(PieceType::PAWN).getBits(),
        0u /* rule50 (checked above) */, 0u /* castling (checked above) */,
        ep != Square::NO_SQ ? static_cast<unsigned>(ep.index()) : 0u,
        board.sideToMove() == Color::WHITE);

    if (res == TB_RESULT_FAILED) return std::nullopt;

    g_hits.fetch_add(1, std::memory_order_relaxed);
    const bool  fifty = g_fifty_rule.load(std::memory_order_relaxed);
    // Inside the TB band [VALUE_TB_WIN_IN_MAX_PLY, VALUE_TB] (types.hpp), ply-encoded like a mate.
    const Value win   = static_cast<Value>(VALUE_TB - ply);
    const Value loss  = static_cast<Value>(-VALUE_TB + ply);
    switch (TB_GET_WDL(res)) {
        case TB_WIN:          return win;
        case TB_LOSS:         return loss;
        case TB_CURSED_WIN:   return fifty ? VALUE_DRAW : win;   // win but drawn by the 50-move rule
        case TB_BLESSED_LOSS: return fifty ? VALUE_DRAW : loss;  // loss but drawn by the 50-move rule
        default:              return VALUE_DRAW;                 // TB_DRAW
    }
}

}  // namespace engine::syzygy
