<p align="center">
  <img src="assets/simplechess.png" width="200" alt="SimpleChess">
</p>

<h1 align="center">SimpleChess</h1>

<p align="center">A UCI chess engine in C++20 with a custom, self-play-trained neural-network evaluation.</p>

<p align="center">
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess bullet rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.bullet.rating&label=bullet&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess blitz rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.blitz.rating&label=blitz&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
  <a href="https://lichess.org/@/SimpleChessNNUE"><img alt="Lichess rapid rating" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2FSimpleChessNNUE&query=%24.perfs.rapid.rating&label=rapid&logo=lichess&logoColor=white&color=6f42c1&style=for-the-badge"></a>
</p>
<p align="center"><sub>Live ratings of the <a href="https://lichess.org/@/SimpleChessNNUE">SimpleChessNNUE</a> bot in Lichess's bot pool.</sub></p>

---

SimpleChess pairs a modern, heavily-tuned alpha-beta search with a from-scratch
NNUE evaluation trained entirely on the engine's own self-play. Board representation
and legal move generation come from a vendored, header-only move-generation
library. The engine is tuned for Apple Silicon but builds and runs anywhere with
a C++20 compiler.

## Features

- **Evaluation** — a custom NNUE: a HalfKAv2_hm input (king-bucketed and
  mirrored) extended with threat and pawn-pair features, a 512-wide accumulator,
  and int8 SIMD inference (NEON on Apple Silicon).
- **Search** — iterative-deepening alpha-beta with aspiration windows, null-move
  pruning, late-move reductions, SEE-based pruning, a transposition table,
  killer / history / continuation move ordering, and lazy-SMP multithreading.
- **UCI** — speaks the UCI protocol; drop it into any UCI GUI (Cute Chess,
  BanksiaGUI, En Croissant, Arena).
- Optional Polyglot opening book support — off by default; no book ships, bring your own.

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
make profile-build PGO_NET=nets/SCNNUEv3-2026-08-30.scn5
```

The release build uses `-O3 -flto -mcpu=native`; retarget the architecture with,
e.g., `make ARCH="-mcpu=apple-m2"` or `make ARCH="-march=x86-64-v3"`.
`profile-build` additionally needs Python 3 with
[python-chess](https://pypi.org/project/chess/) (`pip install chess`) for its
training workload. A `CMakeLists.txt` is provided for IDE / CMake users.

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
| `OwnBook` | false | opt in to a Polyglot book (none ships) |
| `Book File` | *(none)* | path to a Polyglot book, if you supply one |
| `EvalFile` | *(newest `SCNNUEv3-<date>.scn5`)* | network file to load |
| `RootNoise` | 0 | root-move score jitter (cp) for varied play |

## The network

`nets/SCNNUEv3-2026-08-30.scn5` is the pre-quantized int8 playing network (file
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
