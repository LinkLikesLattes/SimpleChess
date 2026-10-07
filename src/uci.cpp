// -----------------------------------------------------------------------------
// uci.cpp
//
// UCI command dispatch. Each recognised command maps to a small handler; unknown
// tokens are ignored, as the protocol requires. Only the commands a typical GUI
// needs are implemented, plus a couple of debug conveniences (`d`).
// -----------------------------------------------------------------------------

#include "uci.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <optional>
#include <limits>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "embed_net.hpp"
#include "io.hpp"
#include "nnue.hpp"
#include "syzygy.hpp"

namespace engine {

// Networks live next to the binary / in nets/ and are named
//   SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5
// (engine MAJOR + export/quant date, year-month-day so names sort
// chronologically). The playing net is the quantised int8 SIMD format ".scn5"
// (magic "SCN5"); the pre-3.0 ".scn3" format (magic "SCN3", no threat features)
// is rejected by nnue::load().
//
// MAJOR encodes format compatibility (a MAJOR bump is an architecture/format
// change); retraining a net does NOT bump the version, so within a MAJOR nets
// are distinguished by DATE, not a minor. Discovery loads the newest net of
// THIS engine's MAJOR; if none exists yet it falls back to the newest net of a
// LOWER major (e.g. the pre-3.0 "SCNNUEv2-5.scn5", which is the same SCN5
// architecture), and only if NOTHING loads it aborts -- there is no
// hand-crafted-eval fallback (removed in 3.0.0): a netless engine is not this
// engine. Point EvalFile at a file to use any other net explicitly (e.g. a
// float ".scn4").
static constexpr int kNetMajor = SC_NET_MAJOR;
static constexpr int kMaxHashMb = 4096;  // the advertised Hash maximum, in MB

UCI::UCI(std::filesystem::path exe_dir) : search_(tt_), exe_dir_(std::move(exe_dir)) {
    hash_mb_ = tt_.resize(hash_mb_);
    board_ = Board(chess::constants::STARTPOS);
    try_load_default_net();
}

// Parse the {major, ...} version vector out of a net filename. Accepts the dated
// form "SCNNUEv3-2026-08-30.scn5" -> {3, 2026, 8, 30} and the legacy form
// "SCNNUEv2-5.scn5" -> {2, 5}. Returns {} if the name isn't our convention.
// The date is YEAR-MONTH-DAY, so the fields land in the vector already in
// chronological significance order and "newest" is a straight std::vector<int>
// max — no reordering. A dated net's year (>=2026) always outranks a legacy
// minor within the same major, so a dated net supersedes a same-major legacy net.
static std::vector<int> net_version_of(const std::string& filename) {
    static constexpr const char* kPrefix = "SCNNUEv";
    static constexpr const char* kSuffix = ".scn5";  // threats playing format (int8 SCN5; see header note)
    const std::size_t plen = std::char_traits<char>::length(kPrefix);
    const std::size_t slen = std::char_traits<char>::length(kSuffix);
    if (filename.size() <= plen + slen) return {};
    if (filename.compare(0, plen, kPrefix) != 0) return {};
    if (filename.compare(filename.size() - slen, slen, kSuffix) != 0) return {};

    std::vector<int> parts;
    const std::string body = filename.substr(plen, filename.size() - plen - slen);
    std::size_t pos = 0;
    while (pos <= body.size()) {
        const std::size_t dash = body.find('-', pos);
        const std::string tok =
            body.substr(pos, dash == std::string::npos ? std::string::npos : dash - pos);
        if (tok.empty() || tok.size() > 9 || tok.find_first_not_of("0123456789") != std::string::npos)
            return {};  // non-numeric (or longer than any version field): not our convention
        parts.push_back(std::stoi(tok));
        if (dash == std::string::npos) break;
        pos = dash + 1;
    }
    return parts;
}

void UCI::try_load_default_net() {
    // The directories a net may live in, most specific first.
    const std::filesystem::path dirs[] = {
        exe_dir_ / "nets", exe_dir_, std::filesystem::path("nets"), std::filesystem::path("."),
    };

    // Scan every recognised net once into two candidate lists:
    //   own -- nets whose MAJOR == this engine's MAJOR (preferred)
    //   low -- nets whose MAJOR <  this engine's MAJOR (transition fallback)
    // ranked newest first by the numeric version vector (see net_version_of). A HIGHER major is
    // never chosen: it may be a future, format-incompatible net this build predates. A net
    // embedded in the executable (Makefile EMBED_NET=..., src/embed_net.cpp) is one more
    // candidate ranked by the same dated name: a newer net on disk still wins, and at an equal
    // version the file comes first. Candidates are tried in order until one loads, so an
    // unreadable newest file falls back to the next-newest net rather than skipping the major.
    struct Candidate {
        std::vector<int>      ver;
        std::filesystem::path path;  // empty = the embedded net
    };
    std::vector<Candidate> own, low;
    std::vector<std::string> seen;  // the same directory can be listed twice (nets/ vs exe/nets)
    for (const auto& dir : dirs) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            std::vector<int> ver = net_version_of(entry.path().filename().string());
            if (ver.empty() || ver[0] > kNetMajor) continue;
            std::error_code cec;
            const std::string key = std::filesystem::weakly_canonical(entry.path(), cec).string();
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
            seen.push_back(key);
            (ver[0] == kNetMajor ? own : low).push_back({std::move(ver), entry.path()});
        }
    }
    if (embed::net_size() > 0) {
        std::vector<int> ver = net_version_of(std::string(embed::net_name()));
        if (!ver.empty() && ver[0] <= kNetMajor)
            (ver[0] == kNetMajor ? own : low).push_back({std::move(ver), {}});
    }
    const auto newest_first = [](const Candidate& a, const Candidate& b) {
        if (a.ver != b.ver) return a.ver > b.ver;
        return !a.path.empty() && b.path.empty();  // equal version: the file before the embedded net
    };
    std::stable_sort(own.begin(), own.end(), newest_first);
    std::stable_sort(low.begin(), low.end(), newest_first);

    const auto try_list = [](const std::vector<Candidate>& list, const std::string& note) {
        for (const Candidate& c : list) {
            if (c.path.empty()) {
                if (nnue::load_memory(embed::net_data(), embed::net_size())) {
                    sync_cout << "info string NNUE loaded: " << embed::net_name() << " (embedded)" << note
                              << sync_endl;
                    return true;
                }
            } else if (nnue::load(c.path.string())) {
                sync_cout << "info string NNUE loaded: " << c.path.string() << note << sync_endl;
                return true;
            }
            sync_cout << "info string NNUE load FAILED: "
                      << (c.path.empty() ? std::string(embed::net_name()) : c.path.string())
                      << " -- trying the next-newest net" << sync_endl;
        }
        return false;
    };

    // 1. Preferred: the newest loadable net of this engine's own major.
    if (try_list(own, "")) return;

    // 2. Transition fallback: no own-major net loads -> the newest loadable LOWER-major
    //    net (e.g. the pre-3.0 "SCNNUEv2-5.scn5", same SCN5 architecture), and SAY SO.
    const std::string low_note = "  (no SCNNUEv" + std::to_string(kNetMajor) +
                                 "-<date>.scn5 found for version " SC_VERSION "; using newest available)";
    if (try_list(low, low_note)) return;

    // 3. Nothing loaded. There is NO hand-crafted-eval fallback in 3.0.0: a netless
    //    engine is a different, weaker program and every tool here depends on the
    //    net. Fail loudly instead of silently playing without it.
    std::cerr << "FATAL: no NNUE net could be loaded (looked for SCNNUEv" << kNetMajor
              << "-<YYYY-MM-DD>.scn5 in nets/ and beside the binary). This engine does "
                 "not run without its network -- place a net there or point EvalFile at one."
              << std::endl;
    std::exit(1);
}


void UCI::handle_uci() const {
    const std::lock_guard<std::mutex> lock(io_mutex());  // one block: id, options, uciok
    std::cout << "id name " << kEngineName << '\n';
    std::cout << "id author " << kEngineAuthor << '\n';

    // Advertise supported options with their type/range so GUIs can render them.
    std::cout << "option name Hash type spin default " << SC_DEFAULT_HASH
              << " min 1 max " << kMaxHashMb << "\n";
    std::cout << "option name Threads type spin default " << Search::kDefaultThreads
              << " min 1 max " << Search::kMaxThreads << "\n";
    std::cout << "option name Move Overhead type spin default 30 min 0 max 5000\n";
    std::cout << "option name Ponder type check default false\n";
    std::cout << "option name MultiPV type spin default 1 min 1 max " << MAX_MOVES << "\n";
    std::cout << "option name UCI_Chess960 type check default false\n";
    std::cout << "option name UCI_ShowWDL type check default false\n";
    std::cout << "option name Clear Hash type button\n";
    std::cout << "option name OwnBook type check default false\n";
    std::cout << "option name Book File type string default <empty>\n";
    std::cout << "option name EvalFile type string default SCNNUEv"
              << kNetMajor << "-<YYYY-MM-DD>.scn5\n";
    std::cout << "option name RootNoise type spin default 0 min 0 max 200\n";
    std::cout << "option name SyzygyPath type string default <empty>\n";
    std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 100\n";
    std::cout << "option name SyzygyProbeLimit type spin default 7 min 0 max 7\n";
    std::cout << "option name Syzygy50MoveRule type check default true\n";
#ifdef SC_BUILD_TAG
    // Dev/campaign builds only (-DSC_BUILD_TAG=<tag>): lets a benchmark harness
    // prove which -D set produced this binary. Absent from release builds.
#define SC_STR_(x) #x
#define SC_STR(x) SC_STR_(x)
    std::cout << "option name BuildTag type string default " << SC_STR(SC_BUILD_TAG) << "\n";
#endif
    pass_param_options(std::cout);  // Pass* tunables: dev knobs, advertised only in SC_PASS_UCI_OPTIONS builds
    std::cout << "uciok" << std::endl;
}

void UCI::handle_isready() const { sync_cout << "readyok" << sync_endl; }

void UCI::handle_newgame() {
    search_.new_game();  // joins any running search, clears history heuristics
    tt_.clear();
    set_fen(board_, chess::constants::STARTPOS);  // cannot fail
}

bool UCI::set_fen(Board& out, std::string_view fen) const {
    Board b;
    const bool ok = chess960_ ? b.setXfen(fen) : b.setFen(fen);
    if (!ok) return false;
    // Refuse positions the engine is not built for, instead of crashing later:
    //  - exactly one king a side (the search and the NNUE index by king square);
    //  - at most 16 men and 8 pawns a side, no pawn on rank 1 or 8 (the NNUE's per-move update
    //    buffers are sized for a legal man count; 48 men overflowed them).
    // A position with the side not to move in check is accepted (it searches without fault, and
    // test positions use it).
    for (const Color c : {Color::WHITE, Color::BLACK}) {
        if (b.pieces(chess::PieceType::KING, c).count() != 1) return false;
        if (b.us(c).count() > 16 || b.pieces(chess::PieceType::PAWN, c).count() > 8) return false;
    }
    if (!(b.pieces(chess::PieceType::PAWN) & Bitboard(0xFF000000000000FFULL)).empty()) return false;
    out = b;
    return true;
}

namespace {
// Move parsing: a token is accepted iff it equals the spelling of a LEGAL move in the
// board's mode (castling is king-to-rook in 960 mode). In 960 mode a token that matches no
// legal spelling is given a second chance as the standard g/c-file castling spelling, so a
// GUI that sends "e1g1" for a castle still castles; a token that already names a legal king
// step is taken as that step (first pass).
Move parse_move(const Board& b, std::string tok) {
    std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char c) { return std::tolower(c); });
    Movelist ml;
    chess::movegen::legalmoves(ml, b);
    for (const Move& m : ml)
        if (chess::uci::moveToUci(m, b.chess960()) == tok) return m;
    if (b.chess960())
        for (const Move& m : ml)
            if (m.typeOf() == Move::CASTLING && chess::uci::moveToUci(m, false) == tok) return m;
    return Move(Move::NO_MOVE);
}

// Bulk-counted perft: the leaves are the legal-move count one ply above the horizon.
std::uint64_t perft_count(Board& b, int depth) {
    Movelist ml;
    chess::movegen::legalmoves(ml, b);
    if (depth <= 1) return static_cast<std::uint64_t>(ml.size());
    std::uint64_t n = 0;
    for (const Move& m : ml) {
        b.makeMove(m);
        n += perft_count(b, depth - 1);
        b.unmakeMove(m);
    }
    return n;
}
}  // namespace

void UCI::handle_setoption(std::istringstream& is) {
    // Grammar: setoption name <name...> [value <value...>]
    std::string token, name, value;
    is >> token;  // expect "name"

    // Collect the (possibly multi-word) option name up to "value".
    while (is >> token && token != "value") {
        if (!name.empty()) name += ' ';
        name += token;
    }
    // Collect the (possibly multi-word) value.
    while (is >> token) {
        if (!value.empty()) value += ' ';
        value += token;
    }

    auto iequals = [](std::string a, std::string b) {
        auto lower = [](std::string& s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        };
        lower(a);
        lower(b);
        return a == b;
    };
    // A numeric option value clamped to its advertised range. A value that is not a number is
    // reported and ignored (the option keeps its setting); one beyond the integer range clamps.
    auto spin = [&](int lo, int hi) -> std::optional<int> {
        try {
            std::size_t used = 0;
            const long long v = std::stoll(value, &used);
            if (used != value.size()) throw std::invalid_argument("trailing characters");
            return static_cast<int>(std::clamp<long long>(v, lo, hi));
        } catch (const std::out_of_range&) {
            return value.find('-') == 0 ? lo : hi;
        } catch (...) {
            sync_cout << "info string " << name << ": invalid value '" << value << "' ignored" << sync_endl;
            return std::nullopt;
        }
    };

    if (iequals(name, "Hash")) {
        // Never reallocate the table under a running search.
        search_.stop();
        search_.wait();
        if (const auto mb = spin(1, kMaxHashMb)) {
            hash_mb_ = tt_.resize(static_cast<std::size_t>(*mb));
            if (hash_mb_ != static_cast<std::size_t>(*mb))
                sync_cout << "info string Hash: only " << hash_mb_ << " MB could be allocated" << sync_endl;
        }
    } else if (iequals(name, "Threads")) {
        if (const auto n = spin(1, Search::kMaxThreads)) {
            threads_ = *n;
            search_.set_threads(threads_);  // joins any running search first
        }
    } else if (iequals(name, "Move Overhead")) {
        if (const auto ms = spin(0, 5000)) move_overhead_ = *ms;
    } else if (iequals(name, "Ponder")) {
        ponder_ = iequals(value, "true");
    } else if (iequals(name, "MultiPV")) {
        // Principal variations to report; clamped to the legal-move count at search time.
        if (const auto n = spin(1, MAX_MOVES)) multipv_ = *n;
    } else if (iequals(name, "UCI_Chess960")) {
        // Takes effect at the next `position` / `gengame` (lazy, as GUIs expect). In 960 mode
        // castling is spoken king-to-rook ("e1h1") and FENs may carry X-FEN or Shredder
        // castling fields; DFRC needs nothing extra (rights are per colour).
        chess960_ = iequals(value, "true") || value == "1";
    } else if (iequals(name, "UCI_ShowWDL")) {
        // Win/draw/loss per mille in the info lines (display only; see wdl.hpp).
        set_show_wdl(iequals(value, "true") || value == "1");
    } else if (iequals(name, "Clear Hash")) {
        search_.stop();
        search_.wait();
        tt_.clear();
    } else if (iequals(name, "OwnBook")) {
        own_book_ = iequals(value, "true");
    } else if (iequals(name, "Book File")) {
        // An explicit path that fails to load just leaves the book unloaded
        // (own_book_ then has nothing to serve) — no book is a normal state.
        book_.load(value);
    } else if (iequals(name, "EvalFile")) {
        // Load an NNUE network on demand. Success switches to the new net; failure
        // leaves the previously loaded net untouched. The engine always has a net
        // (startup aborts if none can be loaded) and there is no HCE fallback.
        // Never swap the net under a running search, and drop the TT with it: its stored
        // static evals are the OLD net's and would feed improving / pass-eval as this net's.
        search_.stop();
        search_.wait();
        if (nnue::load(value)) {
            tt_.clear();
            sync_cout << "info string NNUE loaded: " << value << sync_endl;
        } else
            sync_cout << "info string NNUE load FAILED: " << value << sync_endl;
    } else if (iequals(name, "RootNoise")) {
        if (const auto cp = spin(0, 200)) {
            set_root_noise(*cp);
            sync_cout << "info string RootNoise = " << *cp << " cp" << sync_endl;
        }
    } else if (iequals(name, "SyzygyPath")) {
        // Load Syzygy tablebases on demand (like EvalFile/Book File). An empty or
        // failing path leaves probing disabled — a normal state. init() frees the old
        // tables, so never under a running search (its threads may be probing them).
        search_.stop();
        search_.wait();
        const int men = syzygy::init(value);
        if (men > 0)
            sync_cout << "info string Syzygy: " << men << "-man tablebases loaded" << sync_endl;
        else
            sync_cout << "info string Syzygy: no tablebases loaded" << sync_endl;
    } else if (iequals(name, "SyzygyProbeDepth")) {
        if (const auto d = spin(1, 100)) syzygy::set_probe_depth(*d);
    } else if (iequals(name, "SyzygyProbeLimit")) {
        if (const auto men = spin(0, 7)) syzygy::set_probe_limit(*men);
    } else if (iequals(name, "Syzygy50MoveRule")) {
        syzygy::set_fifty_rule(iequals(value, "true") || value == "1");
    } else if (name.size() > 4 && iequals(name.substr(0, 4), "Pass")) {
        // Pass-eval tunables (SC_PASSEVAL builds; see kPassParams in search.cpp).
        if (const auto v = spin(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
            v && set_pass_param(name, *v))
            sync_cout << "info string " << name << " = " << *v << sync_endl;
    }
    // Unknown options are silently ignored per the protocol.
}

void UCI::handle_position(std::istringstream& is) {
    std::string token;
    is >> token;

    // Build into a local board and install it at the end: an unparseable FEN leaves the
    // current position untouched, and a bad move keeps every move applied before it.
    Board board;
    if (token == "startpos") {
        set_fen(board, chess::constants::STARTPOS);  // cannot fail
        is >> token;  // consume optional "moves"
    } else if (token == "fen") {
        // Reassemble the six-field FEN that follows.
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token;
            fen += ' ';
        }
        if (!set_fen(board, fen)) {
            sync_cout << "info string position: unparseable or illegal FEN '" << fen << "' -- position unchanged"
                      << sync_endl;
            return;
        }
        // token now holds "moves" (or is exhausted).
    } else {
        return;  // malformed
    }

    if (token == "moves") {
        int applied = 0;
        while (is >> token) {
            const Move m = parse_move(board, token);
            if (m == Move(Move::NO_MOVE)) {
                sync_cout << "info string position: illegal move '" << token << "' at ply " << applied
                          << " -- ignoring it and the rest" << sync_endl;
                break;
            }
            board.makeMove(m);
            ++applied;
        }
    }
    board_ = board;
}

void UCI::handle_go(std::istringstream& is) {
    SearchLimits limits;
    std::string  token;
    bool         any_limit      = false;  // a clock, movetime, depth, nodes, mate or infinite
    bool         in_searchmoves = false;  // reading the move list after `searchmoves`

    // A numeric field: an unparseable value leaves the field unset instead of corrupting the stream.
    const auto read_int = [&is](std::int64_t& v) {
        std::string t;
        if (!(is >> t)) return false;
        try { v = std::stoll(t); } catch (...) { return false; }
        return true;
    };
    Movelist legal;
    chess::movegen::legalmoves(legal, board_);

    while (is >> token) {
        std::int64_t v = 0;
        if (token == "searchmoves") {
            in_searchmoves = true;
            continue;
        }
        if (token == "wtime" || token == "btime" || token == "winc" || token == "binc" ||
            token == "movestogo" || token == "depth" || token == "nodes" || token == "movetime" ||
            token == "mate" || token == "infinite" || token == "ponder")
            in_searchmoves = false;

        if (token == "wtime" || token == "btime") {
            if (read_int(v)) {
                limits.time[static_cast<int>(token == "wtime" ? Color::WHITE : Color::BLACK)] = std::max<std::int64_t>(0, v);
                any_limit = true;
            }
        } else if (token == "winc" || token == "binc") {
            if (read_int(v))
                limits.inc[static_cast<int>(token == "winc" ? Color::WHITE : Color::BLACK)] = std::max<std::int64_t>(0, v);
        } else if (token == "movestogo") {
            if (read_int(v)) limits.movestogo = static_cast<int>(std::clamp<std::int64_t>(v, 0, 10'000));
        } else if (token == "depth") {
            // An explicit depth is at least 1 (0 or below used to mean "no limit": unbounded).
            if (read_int(v)) { limits.depth = static_cast<Depth>(std::clamp<std::int64_t>(v, 1, MAX_PLY)); any_limit = true; }
        } else if (token == "nodes") {
            if (read_int(v)) { limits.nodes = static_cast<std::uint64_t>(std::max<std::int64_t>(1, v)); any_limit = true; }
        } else if (token == "movetime") {
            if (read_int(v)) { limits.movetime = std::max<std::int64_t>(1, v); any_limit = true; }
        } else if (token == "mate") {
            if (read_int(v) && v > 0) { limits.mate = static_cast<int>(std::min<std::int64_t>(v, MAX_PLY)); any_limit = true; }
        } else if (token == "infinite") {
            limits.infinite = true;
            any_limit       = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        } else if (in_searchmoves) {
            Move m = Move(Move::NO_MOVE);
            try { m = chess::uci::uciToMove(board_, token); } catch (...) {}
            for (const auto& lm : legal)
                if (lm == m) { limits.searchmoves.push_back(m); break; }  // illegal entries are dropped
        }
    }
    // A bare `go` (no clock, depth, nodes, mate or movetime) is analysis: search until `stop`.
    if (!any_limit) limits.infinite = true;

    // A previous search's worker thread must be fully stopped before we can
    // either answer from the book or start a new one — otherwise a late
    // bestmove from that thread could land after ours and violate the
    // protocol's one-bestmove-per-go contract.
    search_.stop();
    search_.wait();

    // `go infinite` is analysis mode: always search, never answer from the
    // book — an analyst wants the engine's own evaluation of the position, not
    // a canned opening reply. A `searchmoves` restriction also skips the book, and so does
    // `go ponder`: a book reply would be a bestmove before ponderhit/stop.
    if (own_book_ && book_.loaded() && !limits.infinite && !limits.ponder && limits.searchmoves.empty()) {
        const Move book_move = book_.probe(board_);
        if (book_move != Move(Move::NO_MOVE)) {
            sync_cout << "info string book move" << sync_endl;
            sync_cout << "bestmove " << chess::uci::moveToUci(book_move, board_.chess960()) << sync_endl;
            return;
        }
    }

    limits.move_overhead_ms = move_overhead_;  // the parsed `Move Overhead` option, now honoured
    limits.multipv          = multipv_;        // principal variations to report
    search_.start(board_, limits);
}

namespace {
// A gengame/revgame root score in the same cp/mate form Worker::report() emits.
void write_gen_score(std::ostream& out, Value sc) {
    if (is_mate_score(sc)) {
        const int p  = (sc > 0) ? (VALUE_MATE - sc) : (VALUE_MATE + sc);
        const int mm = (sc > 0) ? (p + 1) / 2 : -((p + 1) / 2);
        out << "mate " << mm;
    } else {
        out << "cp " << sc;
    }
}
}  // namespace

void UCI::handle_gengame(std::istringstream& is) {
    // gengame <depth> <maxplies> <fen...>
    // Plays a full game internally from <fen>, greedy (bestmove) each ply, and
    // streams "genply <uci> <cp X|mate Y>" per non-terminal ply, then "genend".
    // Each ply is a fixed-depth search identical to the self-play generator's per-ply analyse
    // (single thread, TT carried across plies, RootNoise 0) -> same moves+scores,
    // so Python rebuilds byte-identical training records without per-ply UCI cost.
    int depth = 8, maxplies = 300;
    is >> depth >> maxplies;
    std::string fen;
    std::getline(is, fen);
    const std::size_t start = fen.find_first_not_of(" \t");
    if (start == std::string::npos) { sync_cout << "genend" << sync_endl; return; }
    fen = fen.substr(start);

    Board board;
    if (!set_fen(board, fen)) { sync_cout << "genend" << sync_endl; return; }
    SearchLimits limits;
    limits.depth = depth;

    std::ostringstream out;
    search_.stop();  // a still-running `go` must finish and print its bestmove before
    search_.wait();  // the silent flag goes up
    set_gen_silent(true);
    for (int ply = 0; ply < maxplies; ++ply) {
        // Stop at exactly the draws python-chess's is_game_over(claim_draw=True) claims
        // (threefold / fifty-move / insufficient material) -- same predicates the search
        // uses (search.cpp) -- so gengame performs the SAME number of searches as the old
        // per-ply loop and leaves the shared TT in the same state (byte-identical labels).
        if (board.isRepetition(2) || board.isHalfMoveDraw() || board.isInsufficientMaterial())
            break;
        search_.start(board, limits);
        search_.wait();
        const Move bm = search_.gen_best_move();
        if (bm == Move(Move::NO_MOVE)) break;  // checkmate / stalemate at this position
        const Value sc = search_.gen_root_score();
        out << "genply " << chess::uci::moveToUci(bm, board.chess960()) << ' ';
        write_gen_score(out, sc);
        out << '\n';
        board.makeMove(bm);
    }
    set_gen_silent(false);
    out << "genend\n";
    const std::lock_guard<std::mutex> lock(io_mutex());
    std::cout << out.str() << std::flush;
}

void UCI::handle_revgame(std::istringstream& is) {
    // revgame <depth> <fen 0>;<fen 1>;...;<fen n-1>
    // Reverse-analysis labelling (train/reverse_analyze.py): searches the given positions from
    // the LAST to the FIRST at fixed depth, the TT carried from each search into the next (the
    // caller sends ucinewgame once per game), each position set from its FEN alone with no game
    // history -- the same searches as one `position fen <fen i>` + `go depth <depth>` per
    // position -- and streams "revply <i> <cp X|mate Y>" per position (in search order, i.e.
    // descending i), then "revend". A FEN that does not parse, or a position with no legal
    // move, gives "revply <i> none".
    int depth = 7;
    is >> depth;
    std::string rest;
    std::getline(is, rest);
    std::vector<std::string> fens;
    for (std::size_t a = 0; a < rest.size();) {
        std::size_t b = rest.find(';', a);
        if (b == std::string::npos) b = rest.size();
        const std::string f = rest.substr(a, b - a);
        const std::size_t s = f.find_first_not_of(" \t");
        fens.push_back(s == std::string::npos ? std::string() : f.substr(s, f.find_last_not_of(" \t\r") - s + 1));
        a = b + 1;
    }
    SearchLimits limits;
    limits.depth = depth;

    std::ostringstream out;
    search_.stop();  // a still-running `go` must finish and print its bestmove before
    search_.wait();  // the silent flag goes up
    set_gen_silent(true);
    for (int i = static_cast<int>(fens.size()) - 1; i >= 0; --i) {
        Board board;
        out << "revply " << i << ' ';
        if (fens[i].empty() || !set_fen(board, fens[i])) {
            out << "none\n";
            continue;
        }
        search_.start(board, limits);
        search_.wait();
        if (search_.gen_best_move() == Move(Move::NO_MOVE)) {
            out << "none\n";
            continue;
        }
        write_gen_score(out, search_.gen_root_score());
        out << '\n';
    }
    set_gen_silent(false);
    out << "revend\n";
    const std::lock_guard<std::mutex> lock(io_mutex());
    std::cout << out.str() << std::flush;
}

void UCI::handle_print() const {
    // Human-readable board dump for debugging (non-standard convenience).
    sync_cout << board_ << "\nFen: " << board_.getFen() << "\nKey: " << std::hex << board_.hash()
              << std::dec << "\nChess960: " << (board_.chess960() ? "true" : "false") << sync_endl;
}

void UCI::handle_perft(std::istringstream& is) {
    // perft <depth>: per-root-move leaf counts and the total, in the shape perft tools expect.
    // Debug command (blocks the UCI thread); the current position is not mutated.
    int depth = 1;
    is >> depth;
    depth = std::max(1, depth);
    Board b = board_;
    Movelist ml;
    chess::movegen::legalmoves(ml, b);
    const auto t0 = std::chrono::steady_clock::now();
    std::uint64_t total = 0;
    std::ostringstream lines;
    for (const Move& m : ml) {
        b.makeMove(m);
        const std::uint64_t n = depth > 1 ? perft_count(b, depth - 1) : 1;
        b.unmakeMove(m);
        lines << chess::uci::moveToUci(m, b.chess960()) << ": " << n << '\n';
        total += n;
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    sync_cout << lines.str() << "\ninfo string perft depth " << depth << " time " << ms << " ms\nNodes searched: " << total
              << sync_endl;
}

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string        command;
        is >> command;

        if (command.empty()) {
            continue;
        } else if (command == "uci") {
            handle_uci();
        } else if (command == "isready") {
            handle_isready();
        } else if (command == "ucinewgame") {
            handle_newgame();
        } else if (command == "setoption") {
            handle_setoption(is);
        } else if (command == "position") {
            handle_position(is);
        } else if (command == "go") {
            handle_go(is);
        } else if (command == "gengame") {
            handle_gengame(is);
        } else if (command == "revgame") {
            handle_revgame(is);
        } else if (command == "perft") {
            handle_perft(is);
        } else if (command == "stop") {
            search_.stop();
        } else if (command == "ponderhit") {
            // The pondered move was played: begin enforcing the clock budget on
            // the search already running (time spent pondering counts toward it).
            search_.ponderhit();
        } else if (command == "d" || command == "print") {
            handle_print();
        } else if (command == "eval") {
            // Debug convenience: static eval of the current position,
            // from the side to move's perspective.
            sync_cout << "static eval (stm pov): " << nnue::evaluate(board_) << " cp"
                      << sync_endl;
        } else if (command == "accsig") {
            // Debug: for every legal move of the current position, print the L1 norm of the
            // accumulator change it makes (item #8's signal) plus whether it is a capture, so the
            // divisor can be set from the measured distribution of QUIET moves rather than guessed.
            Movelist ml;
            chess::movegen::legalmoves(ml, board_);
            nnue::acc_reset(board_);
            for (const auto& m : ml) {
                nnue::acc_make(board_, m);
                const int sig = nnue::acc_delta_l1();
                const int thr = nnue::acc_threats_made();
                nnue::acc_unmake();
                sync_cout << "accsig " << chess::uci::moveToUci(m, board_.chess960())
                          << ' ' << sig << ' ' << (board_.isCapture(m) ? "cap" : "quiet")
                          << ' ' << thr << sync_endl;
            }
            nnue::acc_clear();  // later eval/passeval must not read this position's accumulator
            sync_cout << "accsigend" << sync_endl;
        } else if (command == "passeval") {
            // Debug: E = static eval, P = our eval if we passed, T = E - P (tension); see
            // nnue::evaluate_pass. Undefined in check (the passed position would be illegal).
            const Value e = nnue::evaluate(board_);
            if (board_.inCheck())
                sync_cout << "passeval E " << e << " P n/a T n/a (side to move in check)" << sync_endl;
            else {
                const Value p = nnue::evaluate_pass(board_);
                sync_cout << "passeval E " << e << " P " << p << " T " << (e - p) << sync_endl;
            }
        } else if (command == "quit" || command == "exit") {
            search_.stop();
            search_.wait();
            syzygy::teardown();
            break;
        }
        // Unknown commands are ignored, as required by the protocol.
    }
}

}  // namespace engine
