// -----------------------------------------------------------------------------
// uci.cpp
//
// UCI command dispatch. Each recognised command maps to a small handler; unknown
// tokens are ignored, as the protocol requires. Only the commands a typical GUI
// needs are implemented, plus a couple of debug conveniences (`d`).
// -----------------------------------------------------------------------------

#include "uci.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "nnue.hpp"

namespace engine {

// Networks live next to the binary / in nets/ and are named
//   SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5
// (engine MAJOR + export/quant date, year-month-day so names sort
// chronologically). This engine plays a *threats* net (FullThreats+PP_3Wide)
// whose quantised SIMD format is ".scn5" (int8 feature transformer, magic
// "SCN5") -- NOT the stock ".scn3" (magic "SCN3", which nnue::load() rejects).
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

UCI::UCI(std::filesystem::path exe_dir) : search_(tt_), exe_dir_(std::move(exe_dir)) {
    tt_.resize(hash_mb_);
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
        if (tok.empty() || tok.find_first_not_of("0123456789") != std::string::npos)
            return {};  // non-numeric component: not our convention
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

    // Scan every recognised net once, tracking two candidates:
    //   best_own -- newest net whose MAJOR == this engine's MAJOR (the preferred net)
    //   best_low -- newest net whose MAJOR <  this engine's MAJOR (transition fallback)
    // "Newest" is the numeric version vector (see net_version_of). A HIGHER major is
    // never chosen: it may be a future, format-incompatible net this build predates.
    std::filesystem::path best_own, best_low;
    std::vector<int>      ver_own, ver_low;
    for (const auto& dir : dirs) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            const std::vector<int> ver = net_version_of(entry.path().filename().string());
            if (ver.empty()) continue;
            const int major = ver[0];
            if (major == kNetMajor) {
                if (ver_own.empty() || ver > ver_own) { ver_own = ver; best_own = entry.path(); }
            } else if (major < kNetMajor) {
                if (ver_low.empty() || ver > ver_low) { ver_low = ver; best_low = entry.path(); }
            }
        }
    }

    // 1. Preferred: the newest net of this engine's own major.
    if (!best_own.empty() && nnue::load(best_own.string())) {
        std::cout << "info string NNUE loaded: " << best_own.string() << std::endl;
        return;
    }

    // 2. Transition fallback: no own-major net yet -> load the newest LOWER-major
    //    net (e.g. the pre-3.0 "SCNNUEv2-5.scn5", same SCN5 architecture) and SAY SO.
    if (!best_low.empty() && nnue::load(best_low.string())) {
        std::cout << "info string NNUE loaded: " << best_low.string()
                  << "  (no SCNNUEv" << kNetMajor << "-<date>.scn5 found for version "
                  << SC_VERSION << "; using newest available)" << std::endl;
        return;
    }

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
    std::cout << "id name " << kEngineName << '\n';
    std::cout << "id author " << kEngineAuthor << '\n';

    // Advertise supported options with their type/range so GUIs can render them.
    std::cout << "option name Hash type spin default " << SC_DEFAULT_HASH
              << " min 1 max 4096\n";
    std::cout << "option name Threads type spin default " << Search::kDefaultThreads
              << " min 1 max " << Search::kMaxThreads << "\n";
    std::cout << "option name Move Overhead type spin default 30 min 0 max 5000\n";
    std::cout << "option name Ponder type check default false\n";
    std::cout << "option name Clear Hash type button\n";
    std::cout << "option name OwnBook type check default false\n";
    std::cout << "option name Book File type string default <empty>\n";
    std::cout << "option name EvalFile type string default SCNNUEv"
              << kNetMajor << "-<YYYY-MM-DD>.scn5\n";
    std::cout << "option name RootNoise type spin default 0 min 0 max 200\n";
    std::cout << "uciok" << std::endl;
}

void UCI::handle_isready() const { std::cout << "readyok" << std::endl; }

void UCI::handle_newgame() {
    search_.new_game();  // joins any running search, clears history heuristics
    tt_.clear();
    board_ = Board(chess::constants::STARTPOS);
}

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

    if (iequals(name, "Hash")) {
        // Never reallocate the table under a running search.
        search_.stop();
        search_.wait();
        hash_mb_ = static_cast<std::size_t>(std::max(1, std::stoi(value)));
        tt_.resize(hash_mb_);
    } else if (iequals(name, "Threads")) {
        threads_ = std::clamp(std::stoi(value), 1, Search::kMaxThreads);
        search_.set_threads(threads_);  // joins any running search first
    } else if (iequals(name, "Move Overhead")) {
        move_overhead_ = std::clamp(std::stoi(value), 0, 5000);
    } else if (iequals(name, "Ponder")) {
        ponder_ = iequals(value, "true");
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
        if (nnue::load(value))
            std::cout << "info string NNUE loaded: " << value << std::endl;
        else
            std::cout << "info string NNUE load FAILED: " << value << std::endl;
    } else if (iequals(name, "RootNoise")) {
        const int cp = std::clamp(std::stoi(value), 0, 200);
        set_root_noise(cp);
        std::cout << "info string RootNoise = " << cp << " cp" << std::endl;
    }
    // Unknown options are silently ignored per the protocol.
}

void UCI::handle_position(std::istringstream& is) {
    std::string token;
    is >> token;

    if (token == "startpos") {
        board_ = Board(chess::constants::STARTPOS);
        is >> token;  // consume optional "moves"
    } else if (token == "fen") {
        // Reassemble the six-field FEN that follows.
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token;
            fen += ' ';
        }
        board_ = Board(fen);
        // token now holds "moves" (or is exhausted).
    } else {
        return;  // malformed
    }

    if (token == "moves") {
        while (is >> token) {
            const Move m = chess::uci::uciToMove(board_, token);
            if (m == Move(Move::NO_MOVE)) break;  // stop on an unparseable move
            board_.makeMove(m);
        }
    }
}

void UCI::handle_go(std::istringstream& is) {
    SearchLimits limits;
    std::string  token;

    while (is >> token) {
        if (token == "wtime") {
            is >> limits.time[static_cast<int>(Color::WHITE)];
        } else if (token == "btime") {
            is >> limits.time[static_cast<int>(Color::BLACK)];
        } else if (token == "winc") {
            is >> limits.inc[static_cast<int>(Color::WHITE)];
        } else if (token == "binc") {
            is >> limits.inc[static_cast<int>(Color::BLACK)];
        } else if (token == "movestogo") {
            is >> limits.movestogo;
        } else if (token == "depth") {
            is >> limits.depth;
        } else if (token == "nodes") {
            is >> limits.nodes;
        } else if (token == "movetime") {
            is >> limits.movetime;
        } else if (token == "mate") {
            is >> limits.mate;
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        }
        // `searchmoves` is not yet supported and is intentionally ignored.
    }

    // A previous search's worker thread must be fully stopped before we can
    // either answer from the book or start a new one — otherwise a late
    // bestmove from that thread could land after ours and violate the
    // protocol's one-bestmove-per-go contract.
    search_.stop();
    search_.wait();

    // `go infinite` is analysis mode: always search, never answer from the
    // book — an analyst wants the engine's own evaluation of the position, not
    // a canned opening reply.
    if (own_book_ && book_.loaded() && !limits.infinite) {
        const Move book_move = book_.probe(board_);
        if (book_move != Move(Move::NO_MOVE)) {
            std::cout << "info string book move" << std::endl;
            std::cout << "bestmove " << chess::uci::moveToUci(book_move) << std::endl;
            return;
        }
    }

    search_.start(board_, limits);
}

void UCI::handle_gengame(std::istringstream& is) {
    // gengame <depth> <maxplies> <fen...>
    // Plays a full game internally from <fen>, greedy (bestmove) each ply, and
    // streams "genply <uci> <cp X|mate Y>" per non-terminal ply, then "genend".
    // Each ply is a fixed-depth search identical to selfplay.py's per-ply analyse
    // (single thread, TT carried across plies, RootNoise 0) -> same moves+scores,
    // so Python rebuilds byte-identical training records without per-ply UCI cost.
    int depth = 8, maxplies = 300;
    is >> depth >> maxplies;
    std::string fen;
    std::getline(is, fen);
    const std::size_t start = fen.find_first_not_of(" \t");
    if (start == std::string::npos) { std::cout << "genend" << std::endl; return; }
    fen = fen.substr(start);

    Board        board(fen);
    SearchLimits limits;
    limits.depth = depth;

    std::ostringstream out;
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
        out << "genply " << chess::uci::moveToUci(bm) << ' ';
        if (is_mate_score(sc)) {  // same cp/mate form Worker::report() emits
            const int p  = (sc > 0) ? (VALUE_MATE - sc) : (VALUE_MATE + sc);
            const int mm = (sc > 0) ? (p + 1) / 2 : -((p + 1) / 2);
            out << "mate " << mm;
        } else {
            out << "cp " << sc;
        }
        out << '\n';
        board.makeMove(bm);
    }
    set_gen_silent(false);
    out << "genend\n";
    std::cout << out.str() << std::flush;
}

void UCI::handle_print() const {
    // Human-readable board dump for debugging (non-standard convenience).
    std::cout << board_ << "\nFen: " << board_.getFen() << "\nKey: " << std::hex << board_.hash()
              << std::dec << std::endl;
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
            std::cout << "static eval (stm pov): " << nnue::evaluate(board_) << " cp"
                      << std::endl;
        } else if (command == "quit" || command == "exit") {
            search_.stop();
            search_.wait();
            break;
        }
        // Unknown commands are ignored, as required by the protocol.
    }
}

}  // namespace engine
