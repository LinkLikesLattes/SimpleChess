<p align="center">
  <img src="assets/simplechess.png" width="200" alt="SimpleChess">
</p>

<h1 align="center">SimpleChess</h1>

<p align="center">A UCI chess engine in C++20 with a custom, self-play-trained neural-network evaluation.</p>

<p align="center">
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess bullet rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.bullet.rating&label=bullet&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess blitz rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.blitz.rating&label=blitz&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess rapid rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.rapid.rating&label=rapid&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess Chess960 rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.chess960.rating&label=chess960&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
</p>
<p align="center"><sub>Live ratings of the <a href="https://lichess.org/@/SimpleChessNNUE">SimpleChessNNUE</a> bot in Lichess's bot pool.</sub></p>

---

SimpleChess pairs a modern, heavily-tuned alpha-beta search with a from-scratch
NNUE evaluation trained entirely on the engine's own self-play. Board representation
and legal move generation come from a vendored, header-only move-generation
library. The engine has hand-tuned SIMD paths for both Apple Silicon (NEON) and
x86-64 (AVX2), and builds and runs anywhere with a C++20 compiler.

## Features

- **Evaluation** — a custom NNUE: a HalfKAv2_hm input (king-bucketed and
  mirrored) extended with threat and pawn-pair features, a 512-wide accumulator,
  and int8 SIMD inference (NEON on Apple Silicon, AVX2 on x86-64).
- **Endgame tablebases** — Syzygy WDL probing (via Fathom): point `SyzygyPath` at
  your 3-4-5 or 6-man tables.
- **Search** — iterative-deepening alpha-beta with aspiration windows, null-move
  pruning, late-move reductions, SEE-based pruning, a transposition table,
  killer / history / continuation move ordering, correction history (pawn-structure,
  weighted by an evaluation-tension signal), and lazy-SMP multithreading.
- **Chess960 / FRC and DFRC** — set `UCI_Chess960` (GUIs and lichess-bot do it for
  you): castling is spoken king-to-rook (`e1h1`), X-FEN and Shredder-FEN castling
  fields are both accepted, and asymmetric (DFRC) back ranks need nothing extra.
- **UCI** — speaks the UCI protocol; drop it into any UCI GUI (Cute Chess,
  BanksiaGUI, En Croissant, Arena). Debug conveniences: `d` (board, FEN, key) and
  `perft <depth>` (bulk-counted, Stockfish-style output).
- Optional Polyglot opening book support — off by default; no book ships, bring your own.

## Strength

SimpleChess 3.2.0 is estimated at **3529 Elo on the CCRL Blitz 2'+1" 1-CPU scale** (95% interval 3501 to 3557).
It is an estimate, not a CCRL rating: CCRL has not yet tested 3.1 or 3.2. The number comes from a 464-game
gauntlet against 29 engines with published CCRL ratings, from Stockfish 19 (3784) down to c4ke 3.0 (3303), each
the exact CCRL-tested version and each held at its CCRL 1-CPU rating while 3.2's rating was fitted with BayesElo's
model at its default settings, on CCRL's printed scale. SimpleChess scored +162 =133 -169 (49.2%) against an
average opponent of 3533.

Conditions: 120000 ms + 1000 ms per move on a wall clock; every engine at Threads 1 with 256 MB hash, fresh
processes each game; one CPU core per game on an Intel i7-9700; ponder, opening books and tablebases off; no
adjudication (games end only by mate, stalemate, repetition, the fifty-move rule, insufficient material, time or an
illegal move); 16 games per opponent from the same 8 UHO 2024 (Unbalanced Human Openings) 8-move lines, each played
with both colours. Full method, opponent versions, openings and per-opponent results are in the
[changelog](CHANGELOG.md) under v3.2.0.

## Build

The Makefile is the primary build path — no CMake required. Requires a C++20
compiler (Apple clang works out of the box).

```sh
make            # optimized native build   ->  ./simplechess
make debug      # -O0 -g with ASan/UBSan and assertions
make clean
```

For the fastest (shipping) binary, use the profile-guided build:

```sh
make profile-build PGO_NET=nets/SCNNUEv3-2026-09-12.scn5
```

The release build uses `-O3 -flto -mcpu=native`; retarget the architecture with,
e.g., `make ARCH="-mcpu=apple-m2"` or `make ARCH="-march=x86-64-v3"`. On x86-64 hosts
the Makefile (and CMake) select `-march=native` automatically; for a binary that runs on
any AVX2 machine (Haswell or newer) use `make ARCH="-march=x86-64-v3"`.
`profile-build` additionally needs Python 3 with
[python-chess](https://pypi.org/project/chess/) (`pip install chess`) for its
training workload. A `CMakeLists.txt` is provided for IDE / CMake users.
`make EMBED_NET=nets/SCNNUEv3-2026-09-12.scn5` bakes the network into the executable
for a single-file deployment; a newer net beside the binary still takes precedence.

**Android (arm64).** The repository's GitHub Actions workflow
(`.github/workflows/android.yml`) cross-compiles a fully static build for 64-bit
Android phones — for DroidFish or any other UCI front-end — on every release tag and
on demand. Open the *Android build* run under the Actions tab and download
`simplechess-<version>-android-arm64.tar.gz` (or `-arm64-generic` for SoCs older
than about 2018, which lack the Arm dot-product extension). The network is embedded
in that binary — nothing else needs copying; the archive adds only install notes.

## Run

```sh
./simplechess
```

```
uci
position startpos
go depth 12
```

The network is discovered automatically at startup: the engine loads the newest
`SCNNUEv3-<YYYY-MM-DD>.scn5` found next to the binary or under `nets/` — the
current net ships in the repo, so the engine runs out of the box. There is no
fallback evaluation: if no network can be loaded the engine exits with an error.
Point the `EvalFile` option at another file to use a different network.

## UCI options

| Option | Default | Meaning |
|---|---|---|
| `Hash` | 4096 | transposition-table size, in MB |
| `Threads` | 8 | search threads (lazy SMP) |
| `Move Overhead` | 30 | ms reserved for GUI / network lag |
| `Ponder` | false | think on the opponent's clock |
| `MultiPV` | 1 | principal variations to report, best first (analysis; 1 = normal play) |
| `UCI_Chess960` | false | Chess960 / FRC / DFRC: castling moves are sent and received king-to-rook (`e1h1`); FENs may carry X-FEN (`KQkq`) or Shredder (`HAha`) castling fields |
| `OwnBook` | false | opt in to a Polyglot book (none ships) |
| `Book File` | *(none)* | path to a Polyglot book, if you supply one |
| `EvalFile` | *(newest `SCNNUEv3-<date>.scn5`)* | network file to load |
| `RootNoise` | 0 | root-move score jitter (cp) for varied play |
| `SyzygyPath` | *(none)* | directory of Syzygy WDL tables (`*.rtbw`); probing is off until set |
| `SyzygyProbeDepth` | 1 | minimum remaining depth for an interior probe |
| `SyzygyProbeLimit` | 7 | probe only with this many pieces or fewer |
| `Syzygy50MoveRule` | true | treat cursed wins / blessed losses as draws |

## The network

`nets/SCNNUEv3-2026-09-12.scn5` is the pre-quantized int8 playing network (file
magic `SCN5`), committed to the repo so the engine works immediately after a
clone or "Download ZIP." Nets are named `SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5`
(engine major + export date) and the engine loads the newest one it finds; the
repo always ships only the latest network — older nets are available from older
releases. It is trained from the engine's own self-play; the trainer and
training data are developed separately and are not part of this repository.

## License

Licensed under the **GNU General Public License v3.0** — see
[LICENSE](LICENSE).

Author: **Sam Moore**
