// -----------------------------------------------------------------------------
// see.cpp
//
// Swap-algorithm SEE (static exchange evaluation). Deliberate simplifications,
// all on the conservative/optimistic side:
//   * non-NORMAL moves (castling, en passant, promotions) are not simulated —
//     castling can't lose material, ep is pawn-takes-pawn, and promotions are
//     rare enough that treating them optimistically costs little;
//   * pinned pieces are allowed to recapture (no pin legality filtering).
// -----------------------------------------------------------------------------

#include "see.hpp"

namespace engine::see {

namespace {

// All pieces of either color attacking `sq`, given custom occupancy `occ`
// (so callers can "lift" pieces off the board to expose x-ray attackers).
[[nodiscard]] Bitboard attackers_to(const Board& board, Square sq, Bitboard occ) noexcept {
    using A = chess::attacks;  // static-member "namespace" class in the library

    // A white pawn attacking `sq` sits on a square attacked by a *black* pawn
    // standing on `sq`, hence the flipped colors.
    return (A::pawn(Color::BLACK, sq) & board.pieces(PieceType::PAWN, Color::WHITE)) |
           (A::pawn(Color::WHITE, sq) & board.pieces(PieceType::PAWN, Color::BLACK)) |
           (A::knight(sq) & board.pieces(PieceType::KNIGHT)) |
           (A::bishop(sq, occ) & (board.pieces(PieceType::BISHOP) | board.pieces(PieceType::QUEEN))) |
           (A::rook(sq, occ) & (board.pieces(PieceType::ROOK) | board.pieces(PieceType::QUEEN))) |
           (A::king(sq) & board.pieces(PieceType::KING));
}

}  // namespace

bool see_ge(const Board& board, Move move, int threshold) noexcept {
    // Only plain moves are simulated; see file comment for the rationale.
    if (move.typeOf() != Move::NORMAL) return 0 >= threshold;

    const Square from = move.from();
    const Square to   = move.to();

    // Best case: we take the target and the opponent never recaptures.
    int swap = value(board.at<PieceType>(to)) - threshold;
    if (swap < 0) return false;

    // Worst case: we take, they recapture our piece, exchange ends there.
    swap = value(board.at<PieceType>(from)) - swap;
    if (swap <= 0) return true;

    Bitboard occ = board.occ() ^ Bitboard::fromSquare(from) ^ Bitboard::fromSquare(to);
    Color    stm = board.sideToMove();
    Bitboard attackers = attackers_to(board, to, occ);
    int      result    = 1;  // 1 == side to move wins the exchange so far

    while (true) {
        stm = ~stm;
        attackers = attackers & occ;

        Bitboard stm_attackers = attackers & board.us(stm);
        if (stm_attackers.empty()) break;  // no more recaptures: current result stands

        result ^= 1;

        // Apply the least valuable attacker; removing a slider may reveal an
        // x-ray attacker behind it, so refresh the relevant slider attacks.
        Bitboard bb;
        if (!(bb = stm_attackers & board.pieces(PieceType::PAWN)).empty()) {
            if ((swap = kPieceValue[0] - swap) < result) break;
            occ ^= Bitboard::fromSquare(bb.lsb());
            attackers = attackers | (chess::attacks::bishop(to, occ) &
                                     (board.pieces(PieceType::BISHOP) | board.pieces(PieceType::QUEEN)));
        } else if (!(bb = stm_attackers & board.pieces(PieceType::KNIGHT)).empty()) {
            if ((swap = kPieceValue[1] - swap) < result) break;
            occ ^= Bitboard::fromSquare(bb.lsb());
        } else if (!(bb = stm_attackers & board.pieces(PieceType::BISHOP)).empty()) {
            if ((swap = kPieceValue[2] - swap) < result) break;
            occ ^= Bitboard::fromSquare(bb.lsb());
            attackers = attackers | (chess::attacks::bishop(to, occ) &
                                     (board.pieces(PieceType::BISHOP) | board.pieces(PieceType::QUEEN)));
        } else if (!(bb = stm_attackers & board.pieces(PieceType::ROOK)).empty()) {
            if ((swap = kPieceValue[3] - swap) < result) break;
            occ ^= Bitboard::fromSquare(bb.lsb());
            attackers = attackers | (chess::attacks::rook(to, occ) &
                                     (board.pieces(PieceType::ROOK) | board.pieces(PieceType::QUEEN)));
        } else if (!(bb = stm_attackers & board.pieces(PieceType::QUEEN)).empty()) {
            if ((swap = kPieceValue[4] - swap) < result) break;
            occ ^= Bitboard::fromSquare(bb.lsb());
            attackers = attackers |
                        (chess::attacks::bishop(to, occ) &
                         (board.pieces(PieceType::BISHOP) | board.pieces(PieceType::QUEEN))) |
                        (chess::attacks::rook(to, occ) &
                         (board.pieces(PieceType::ROOK) | board.pieces(PieceType::QUEEN)));
        } else {
            // King capture: only stands if the opponent has no attackers left,
            // otherwise the "capture" would be illegal and the result flips back.
            return !(attackers & ~board.us(stm)).empty() ? bool(result ^ 1) : bool(result);
        }
    }

    return bool(result);
}

}  // namespace engine::see
