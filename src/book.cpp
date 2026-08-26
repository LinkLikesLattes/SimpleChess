// -----------------------------------------------------------------------------
// book.cpp
//
// Polyglot book loading and probing. Two independent pieces:
//   * polyglot_hash() — recomputes the standard Polyglot zobrist key from
//     board state, from scratch, every probe. Probing isn't hot enough
//     (once per `go`, not once per node) to justify incremental maintenance.
//   * Book::probe() — binary search for the key, then match each candidate
//     book move against the position's actual legal moves so a decoded move
//     is guaranteed playable even in the face of a hash collision.
// -----------------------------------------------------------------------------

#include "book.hpp"

#include <algorithm>
#include <fstream>

#include "polyglot_random.hpp"

namespace engine {

namespace {

using CastlingSide = chess::Board::CastlingRights::Side;

// Big-endian field reads: Polyglot books are a fixed-width binary format
// written network-byte-order, independent of the host's endianness.
[[nodiscard]] std::uint64_t read_u64be(const unsigned char* p) noexcept {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

[[nodiscard]] std::uint16_t read_u16be(const unsigned char* p) noexcept {
    return static_cast<std::uint16_t>((p[0] << 8) | p[1]);
}

// Standard Polyglot position hash. Deliberately independent of the engine's
// own Board::hash() — see book.hpp for why.
[[nodiscard]] std::uint64_t polyglot_hash(const Board& board) noexcept {
    using polyglot::RANDOM_ARRAY;

    std::uint64_t h = 0;

    for (int c = 0; c < 2; ++c) {
        const Color color = static_cast<Color::underlying>(c);
        for (int pt = 0; pt < 6; ++pt) {
            const auto piece_type  = static_cast<PieceType::underlying>(pt);
            const int  piece_index = pt * 2 + (color == Color::WHITE ? 1 : 0);

            Bitboard bb = board.pieces(PieceType(piece_type), color);
            while (bb) {
                const int sq = bb.pop();
                h ^= RANDOM_ARRAY[64 * piece_index + sq];
            }
        }
    }

    const auto cr = board.castlingRights();
    if (cr.has(Color::WHITE, CastlingSide::KING_SIDE)) h ^= RANDOM_ARRAY[768];
    if (cr.has(Color::WHITE, CastlingSide::QUEEN_SIDE)) h ^= RANDOM_ARRAY[769];
    if (cr.has(Color::BLACK, CastlingSide::KING_SIDE)) h ^= RANDOM_ARRAY[770];
    if (cr.has(Color::BLACK, CastlingSide::QUEEN_SIDE)) h ^= RANDOM_ARRAY[771];

    const Square ep = board.enpassantSq();
    if (ep != Square::NO_SQ) {
        // Polyglot only hashes the ep file when a pawn of the side to move is
        // actually positioned to make the capture (not merely when the FEN ep
        // field is set). Compute the square of the pawn that would be taken,
        // then check its two diagonal neighbours on that rank.
        const Color stm         = board.sideToMove();
        const int   ep_idx      = ep.index();
        const int   captured    = (stm == Color::WHITE) ? ep_idx - 8 : ep_idx + 8;
        const int   rank        = captured / 8;
        const int   file        = captured % 8;
        const Bitboard stm_pawns = board.pieces(PieceType::PAWN, stm);

        bool can_capture = false;
        if (file > 0 && (stm_pawns & Bitboard::fromSquare(rank * 8 + file - 1))) can_capture = true;
        if (file < 7 && (stm_pawns & Bitboard::fromSquare(rank * 8 + file + 1))) can_capture = true;

        if (can_capture) h ^= RANDOM_ARRAY[772 + static_cast<int>(ep.file())];
    }

    if (board.sideToMove() == Color::WHITE) h ^= RANDOM_ARRAY[780];

    return h;
}

}  // namespace

bool Book::load(const std::filesystem::path& path) {
    entries_.clear();
    path_.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size <= 0 || size % 16 != 0) return false;
    f.seekg(0);

    std::vector<unsigned char> raw(static_cast<std::size_t>(size));
    if (!f.read(reinterpret_cast<char*>(raw.data()), size)) return false;

    entries_.reserve(raw.size() / 16);
    for (std::size_t i = 0; i < raw.size(); i += 16) {
        entries_.push_back(
            Entry{read_u64be(&raw[i]), read_u16be(&raw[i + 8]), read_u16be(&raw[i + 10])});
    }

    // Sort defensively rather than trust the file's ordering: binary search
    // below requires it, and re-sorting a few thousand entries at load time
    // is unmeasurable next to the cost of reading the file.
    std::sort(entries_.begin(), entries_.end(),
              [](const Entry& a, const Entry& b) { return a.key < b.key; });

    path_ = path;
    return true;
}

void Book::unload() {
    entries_.clear();
    path_.clear();
}

Move Book::probe(const Board& board) {
    if (entries_.empty()) return Move(Move::NO_MOVE);

    const std::uint64_t key = polyglot_hash(board);

    const auto lo = std::lower_bound(entries_.begin(), entries_.end(), key,
                                     [](const Entry& e, std::uint64_t k) { return e.key < k; });
    if (lo == entries_.end() || lo->key != key) return Move(Move::NO_MOVE);

    auto hi = lo;
    while (hi != entries_.end() && hi->key == key) ++hi;

    Movelist legal;
    chess::movegen::legalmoves(legal, board);

    // Polyglot promotion codes: 0=none,1=knight,2=bishop,3=rook,4=queen.
    static constexpr PieceType::underlying kPromoMap[5] = {
        PieceType::NONE, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN,
    };

    struct Candidate {
        Move          move;
        std::uint16_t weight;
    };
    std::vector<Candidate> candidates;
    std::uint64_t          total_weight = 0;

    for (auto it = lo; it != hi; ++it) {
        // Decode the 16-bit Polyglot move (see polyglot_random.hpp banner /
        // the format spec): bits 0-2 to-file, 3-5 to-rank, 6-8 from-file,
        // 9-11 from-rank, 12-14 promotion. Castling is encoded as the king
        // "capturing" its own rook — which happens to be exactly how this
        // library represents castling internally, so no special-casing is
        // needed: it falls out of the from/to match below.
        const int to_sq   = ((it->move >> 3) & 0x7) * 8 + (it->move & 0x7);
        const int from_sq = ((it->move >> 9) & 0x7) * 8 + ((it->move >> 6) & 0x7);
        const int promo   = (it->move >> 12) & 0x7;
        const PieceType::underlying promo_pt = (promo <= 4) ? kPromoMap[promo] : PieceType::NONE;

        for (int m = 0; m < legal.size(); ++m) {
            const Move cand = legal[m];
            if (cand.from().index() != from_sq || cand.to().index() != to_sq) continue;

            const bool is_promo = cand.typeOf() == Move::PROMOTION;
            if (is_promo != (promo_pt != PieceType::NONE)) continue;
            if (is_promo && cand.promotionType() != promo_pt) continue;

            candidates.push_back({cand, it->weight});
            total_weight += it->weight;
            break;
        }
    }

    if (candidates.empty()) return Move(Move::NO_MOVE);

    // Weighted-random selection, as the Polyglot format intends; falls back
    // to a uniform pick if every matching entry has weight zero.
    if (total_weight == 0) {
        std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
        return candidates[pick(rng_)].move;
    }

    std::uniform_int_distribution<std::uint64_t> pick(0, total_weight - 1);
    std::uint64_t                                r = pick(rng_);
    for (const auto& c : candidates) {
        if (r < c.weight) return c.move;
        r -= c.weight;
    }
    return candidates.back().move;  // unreachable
}

}  // namespace engine
