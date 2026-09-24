# Changelog

Release history for SimpleChess. An entry is added here whenever a tested candidate is promoted to the live `simplechess` binary.

<!-- new entries inserted below this line by `version.py promote` -->

## v3.3.2 — 2026-09-24

- Self-reports: `SimpleChess 3.3.2`
- Source VERSION at save time: `3.3.2`

### Android build: static executable (fixes the 3.3.1 startup crash)

The 3.3.1 Android binary segfaulted the moment it started, on real devices (DroidFish "Failed to
start engine"; a bare run in Termux: `Segmentation fault`). It was linked as a static-PIE, and
Bionic's static startup code reads `main` and the init arrays through the GOT before anything applies
the executable's own `R_AARCH64_RELATIVE` relocations: in the shipped file the GOT slot holding
`main` is zero with its relocation still pending, so startup jumps through null before `main` runs.
The 3.3.0 notes put the same crash under qemu down to qemu; it was this, and the CI never caught it
because it run-tested a second, plain-static link of the objects instead of the shipped file. The
workflow now ships that plain static executable (Android's position-independence rule lives in its
dynamic linker, which a static executable never meets; the kernel loads it directly), asserts it is
`ET_EXEC` with no `PT_INTERP` and no `PT_DYNAMIC` (nothing to relocate at startup), and run-tests
the shipped file itself — the stripped binary unpacked from the uploaded archive, `cmp`-checked,
alone in an empty directory. Engine source unchanged; play is identical to 3.3.1. Reported by
Arzam18 (issue #1).

## v3.3.1 — 2026-09-23

- Self-reports: `SimpleChess 3.3.1`
- Source VERSION at save time: `3.3.1`

### Embedded network (Android build)

`make EMBED_NET=nets/SCNNUEv3-<date>.scn5` bakes the network into the executable (`src/embed_net.cpp`:
an `.incbin` into read-only data, so the file never passes through the compiler) and a single file
then runs with nothing beside it. Startup ranks the embedded net with the on-disk candidates by the
same dated version — a newer net beside the binary or in `nets/` still wins and `EvalFile` still
overrides — and loads it through the same parser as a file (`nnue::load_memory` over an in-memory
stream), so the two paths cannot diverge. The Android workflow now builds with it, run-tests the
binary with no net file present, and ships `simplechess` + `README-ANDROID.txt` only. Reported by
Arzam18 (issue #1): a phone tournament harness copies the executable alone, and the 3.3.0 artifact
exited for want of its net.

## v3.3.0 — 2026-09-22

- Self-reports: `SimpleChess 3.3.0`
- Source VERSION at save time: `3.3.0`

**At a glance** (full detail in the sections below):

- `MultiPV` analysis mode.
- 50-move rule: the static evaluation is damped toward the draw as the halfmove clock climbs, and a
  cached mate the clock voids is no longer trusted.
- Android arm64 build (fully static, for DroidFish and other phone front-ends) from the repository's
  GitHub Actions workflow.
- `Pass*` tuning knobs no longer advertised; `make clean` and header dependency tracking fixed for
  source checkouts.

### MultiPV

`MultiPV` (spin, default 1, max 256): the engine reports the N best root moves, each with its own
principal variation, as `info ... multipv I ...` lines. Each iteration runs one full root search
per line: the line's move is searched first, the lines already found are excluded, and only the first
line writes the root transposition-table entry, counts the time manager's effort shares and drives
the adaptive width. Lines are kept in a stable descending sort; a line not reached in an aborted
iteration is shown at depth-1 from its previous score, and a line cut off on an aspiration bound
carries `lowerbound`/`upperbound`. The move played is the top line. The default is untouched: at
`MultiPV 1` the search is node-identical and bestmove-identical to 3.2.0, and the only change to the
info line is the standard ` multipv 1` token.

### 50-move-rule eval damping

The static evaluation is now scaled toward zero in proportion to the halfmove clock: a freshly
computed net eval is damped by `eval -= eval * halfmove_clock / 199` before it drives pruning and
before it enters the transposition table. A position shuffling toward the 50-move draw — no capture
or pawn move for dozens of plies — therefore stops reporting a full advantage it can no longer force,
and the search stops chasing a win that the draw counter is about to erase. The factor is roughly a
half at clock 100 (the draw threshold) and reaches zero only at 199, which real play never sees. The
damping is applied only to a freshly evaluated position, never to a value reused from the table (that
value was already damped when it was written, so re-damping would compound on transpositions). Both
static-eval sites — the main search and the quiescence stand-pat — are covered. A correctness change
whose effect lives in the high-clock endgame tail.

### 50-move-rule mate reclassification

A mate score read back from the transposition table is now trusted only if the mating side can
actually deliver it before the 50-move draw resets the game. When the reported distance to mate
exceeds the plies the halfmove clock still allows, the mate is unreachable, so it is downgraded to a
high but non-mate score: the search keeps treating the position as winning, but stops trusting,
extending and reporting a forced mate the draw counter voids — the case where a mate found at a low
clock is reused, via a transposition-table cutoff, at a higher one. A mate that is reachable within
the clock budget is unaffected and still reported as a mate. The rule is confined to true mate scores;
tablebase scores are never cached in the table, so they cannot reach this path. A correctness fix:
the engine no longer chases a mate it cannot force; giving up a false-mate cutoff costs a few nodes,
and the payoff is a long-time-control property.

### Pass* tunables no longer advertised

The eleven `Pass*` pass-eval tunables were development knobs, not user options; they are no longer
listed in the `uci` option set (a tuning build sets `SC_PASS_UCI_OPTIONS=1` to re-expose them).
`setoption` still accepts them, so match tooling that sets `PassX=...` keeps working. E6's defaults
are compiled in and unchanged.

### Android arm64 build (GitHub Actions)

`.github/workflows/android.yml` cross-compiles the engine for 64-bit Android with the NDK, as a
fully static position-independent executable (the form Android will exec and a phone front-end
such as DroidFish can run), in two variants: `arm64` (`-march=armv8.2-a+dotprod`, the build for any
phone SoC from about 2018 on) and `arm64-generic` (`-march=armv8-a`, for older SoCs; the int8 network
layers have no NEON path without the dot-product extension and fall back to scalar code, so it is
much slower but plays identically). Each run checks the ELF (AArch64, PIE, no dynamic loader, no
shared-library dependencies), then links the same objects once more as a plain static executable and
runs that under qemu-user — qemu cannot load a static-PIE built against Bionic, and the plain static
link differs from the shipped file in nothing but the link mode — with the shipped net beside it,
through `tools/ci/uci_smoke.py`: net discovery, then four positions searched to a fixed depth at
1 thread, emulating a Cortex-A76 for the dotprod build and a Cortex-A53 for the generic one so an
instruction the target lacks would fault in CI rather than on a phone. Both variants reproduce the
native build's node counts exactly. Only then is `simplechess-<version>-android-<variant>.tar.gz`
(stripped binary, network, `README-ANDROID.txt` install notes) uploaded as a run artifact. The
workflow runs on every release tag and on demand, so the build for this release is under the
repository's Actions tab. The phone build lowers the compiled-in Hash default from 4 GB to 64 MB (the
table is zero-filled at startup; the front-end sets Hash anyway) and requires NDK 28 or newer (NDK
27's static link is rejected by Bionic at startup). The idea, the DroidFish PIE/static requirement
and the NDK build recipe come from Arzam18's fork (github.com/Arzam18/SimpleChess), whose workflow
this replaces.

The Makefile's `clean` and `version` targets and its header-dependency tracking were gated by
accident on the dev-only `tools/release/version.py`, so a source-only checkout had no `make clean`
and did not rebuild objects after a header edit. Both are now unconditional.

## v3.2.0 — 2026-09-16

- Self-reports: `SimpleChess 3.2.0`
- Source VERSION at save time: `3.2.0`

**At a glance** (numbers only; full method in the sections below):

- 3.1.0 -> 3.2.0: **+34 Elo** [+26, +41] at 10+0.1 -- the gain is the net retrain; the code is at parity at fast TC.
- E6 correction-history tension weighting **+3.5 Elo**; adaptive-width endgame fix **+3 Elo**.
- Estimated **CCRL Blitz 2'+1" (1-CPU): 3529** (95% interval 3501-3557) -- our estimate, not a CCRL rating.

### Correction history weighted by pass-eval tension (E6)

The net is queried a second time with the two
perspectives swapped, giving P, our static score if we had to pass; T = E - P is the static value of
having the move. A phase-controlled diagnostic over 19,529 positions showed that the residual
`search - static` is not only larger where T is high but also MORE consistent within a pawn-structure
key (intraclass correlation 0.55 -> 0.60 across tension quintiles), so the correction table now takes a
larger step there: the update bonus is scaled by `1 + tn * 16/4096` at nodes of depth >= 6, where
`tn = clamp(T - 45, 0, 1151)`. Measured +3.5 Elo [-1.7, +8.7], likelihood of superiority 90.4%, over
6,000 games at 10+0.1 against the bit-identical control; the SPRT did not certify it (LLR +0.70 at the
3,000-pair backstop); it ships on the positive point estimate. The other six pass-eval plug
points (reverse futility / razoring margins, LMR, null move, quiescence stand-pat, time management) were
tested and stripped; their code stays compiled in behind `Pass*` tunables that default to 0 and are
inert.

### Adaptive search width no longer narrows with material

The continuous width that scales the late-move
reductions, late-move pruning and futility margins carried a phase factor, `clamp(root_phase - 12, 0, 12)`,
that fell to zero from a queen-and-rook ending onward, so the engine ran its narrowest tree exactly where
lines are most forcing: it would reach depth 40+ in seconds while its move stopped changing before depth 30,
and sit on a mate-in-3 for half a minute. Neither Stockfish nor Reckless scales width by material at all.
The factor is removed; the width is now the root-gap x score signal only. Measured +3 Elo [-1, +8],
likelihood of superiority 92.3% over 6,000 games at 10+0.1 (the GSPRT did not certify it, LLR +0.86); kept
because it is neutral-to-positive and fixes the endgame pathology the campaign was about. Five other search
changes tried in the same campaign - the pawn-only-endgame prune-off, TB/mate-band prune guards,
never-LMP-a-check, a seek-mate mode, and an LMR clamp with re-search - each lost its SPRT and were dropped.

### Net vs. code isolation

The 3.1.0 -> 3.2.0 strength gain is the net retrain; the two code changes above are at parity at fast
time control (short-TC parity, below). Two SPRT
tests at 10+0.1, `tools/match/pmatch_tc.py`, Threads 1, Hash 256, book `8moves_v3.suite`, seed 42, paired
openings, each engine's own net swapped in via `--set-a`/`--set-b EvalFile=`:

- **Code only** (net held at the September net, `SCNNUEv3-2026-09-12.scn5`, on both sides): 3.2.0 code
  vs 3.1.0 code, +0 Elo (+0 [-8,+9], LLR -0.08, 1332 games).
- **Combined** (3.2.0 code + September net vs 3.1.0 code + August net, the actual shipped gap): +34 Elo
  [+26,+41], LLR +4.89, LOS 100%, 5526 games.

### Short-TC parity: depth vs. width

The code-only +0 is expected, not a null result. The width change trades a little search depth for width,
and that extra breadth in low-material positions needs time to convert; at 10+0.1 the depth given up and the
width gained roughly cancel, so the code reads as parity. The gain is expected to show at the longer controls
the search is tuned for (2'+1" and up), where there is time for the wider endgame tree to pay off, rather than
costing anything at fast TC.

### Move Overhead honoured

Time management now honours the `Move Overhead` option. It was parsed since 1.x but the budget used a
hard-coded 30 ms reserve, so a GUI asking for more (lichess-bot sets 100 ms; phone GUIs need more)
was silently ignored. The reserve now flows through `SearchLimits::move_overhead_ms` into the
fixed-movetime path, the per-move clock reserve and the maximum clamp. Default 30 ms: every search
at the default is unchanged (fixed-depth identity 28/28); at 500 ms a 2000 ms movetime spends
1500 ms and a 60 s clock budget shrinks from 1272 to 820 ms on the same position.

### Chess960 / FRC / DFRC

`UCI_Chess960` (check, default false, applied by the next `position`, the
Stockfish contract). In 960 mode castling is spoken king-to-rook (`e1h1`, `bestmove`, `info pv` and
`position … moves` alike; the standard `e1g1` spelling is still accepted when unambiguous), FEN castling
fields may be X-FEN (`KQkq`, outermost rook) or Shredder (`HAha`), and `d` reports Shredder-FEN.
Asymmetric back ranks (DFRC) need nothing extra. The NNUE and the search are unchanged: castling was
already encoded king-takes-rook internally, the evaluation reads no castling rights, and standard-chess
play is bit-identical to 3.1.0 (fixed-depth nodes/score/PV 28/28 on the int8 table and 29/29 on the
float table, 40/40 self-play games byte-identical in both modes, the `position … moves` path identical).

### Move and FEN validation

`position` now validates every move against the legal-move list (Stockfish's `to_move`): an illegal or
misspelled token is reported with `info string` and the rest of the list is ignored; an unparseable FEN
(including a king-less one) leaves the position unchanged instead of crashing later.

### Move generation: double-check counting fix

The vendored chess-library mis-counted double checks given by two diagonal sliders (and by
two pawns or two knights, which only random positions produce), generating interpositions that leave a check
standing; the engine could then play an illegal move or crash. Fixed in the vendored header (count checkers);
verified against python-chess over 1.7M positions.

### `perft` debug command

New debug command `perft <depth>` (bulk-counted, per-root-move counts, `Nodes searched: N`). Verified
on Stockfish 19's 13 FRC/DFRC perft positions at full depth (Shredder and X-FEN spellings), the classic
standard set, and all 960 positions of Ethereal's `fischer.epd` (depth 5; 60 at depth 6); engine-vs-
itself play from FRC and DFRC starts checked move by move with python-chess (every move legal, castling
king-to-rook, `d` round-trips).

### Self-play is start-position agnostic

The labeler always runs in `UCI_Chess960` mode, so the seed book
may mix standard and FRC/DFRC positions; labels for standard seeds are byte-identical to 3.1.0's. No
960 strength claim: the current net was trained on standard data only.

### Strength estimate: CCRL-anchored gauntlet

**SimpleChess 3.2.0 is estimated at 3529 Elo on the CCRL Blitz 2'+1" 1-CPU scale, 95% interval 3501 to 3557.**
This is our own estimate, not a CCRL rating. CCRL has not tested 3.1.0, the first official x86 build, or 3.2.0, so
3.2.0 played a gauntlet against 29 engines with published CCRL ratings, and its rating was fitted with BayesElo's
model while every opponent was held at its CCRL rating. Result: +162 =133 -169 in 464 games (228.5/464, 49.2%)
against a game-weighted average opponent of 3533; the plain logistic performance over the same games is 3528.

**Anchors.** CCRL Blitz 2'+1" rating list, "complete list" (all engines), computed September 12, 2026 with Bayeselo
from 2,115,612 games; CCRL conditions: ponder off, general book up to 12 moves, up to 6-piece EGTB, time control
equivalent to 2'+1" on an Intel i7-4770K. Every anchor is the engine's 1-CPU entry, so every engine, SimpleChess
included, played at Threads 1. Seven engines whose best-version entry is an 8CPU one were anchored at their 1-CPU
entry (marked 8CPU* below, with the 8CPU rating for reference). Stockfish 19's rating is from the complete list.
Each opponent is the exact CCRL-tested version, taken from that engine's official GitHub release as a Windows
x86-64 build that runs on AVX2 (Stockfish 19's universal build picks its code path by CPU), with the asset size
checked against the server and the file sha256'd.

**Hardware and build.** Intel Core i7-9700 (8 cores, 8 threads, AVX2 and BMI2, no AVX-512), Windows 11.
`simplechess-3.2.0-avx2.exe` sha256 `981b2b92650d9f6716c250d73f389b2fc2eb580749379a418e156cb61456f25f`
(1,227,264 bytes) with `SCNNUEv3-2026-09-12.scn5` sha256
`1bdd72bc999b4fcdc5b204fa0a0b667b38070c4e5a1cd8c5aa876a59459fff87`.

**Conditions.**
- Time control 120000 ms + 1000 ms per move (2'+1"), charged by wall clock: each engine's clock loses the real time
  from `go` to `bestmove`, and a clock below -100 ms after a move loses on time. `ucinewgame` and the
  `isready`/`readyok` handshake happen before the clock starts. Not scaled to CCRL's i7-4770K reference.
- One core per game: 8 games at a time on 8 cores, each game's two engines locked to one logical CPU by a Windows
  job object with a one-CPU affinity limit an engine cannot override, and interleaved there; ponder off, so only
  one engine thinks at a time. All 464 games ran from one fixed work queue: a core starts the next game the moment
  its current game ends, so no core waits for another.
- Threads 1 for every engine (Petrel 4.0, bitbit 1.7 and Maelstrom 3.3.0 are single-threaded and have no Threads
  option). Hash 256 MB for every engine. Both engines are fresh processes every game, so the hash starts empty.
- Opening books off: `OwnBook` false for SimpleChess, Koivisto and Texel, the only engines exposing a book option.
  No tablebase path was set for any engine (every engine's default is empty).
- No adjudication: no resignation, no draw by agreement, no move limit. A game ends only by checkmate, stalemate,
  threefold repetition, the fifty-move rule, insufficient material, loss on time, an illegal move, or a crash.
  Every game was recorded to PGN.

**Openings: UHO.** 16 games per opponent: the same 8 openings for every opponent, each played once with each
colour. The lines are 8-move (16-ply) UHO openings (Unbalanced Human Openings) from SP-CC's UHO 2024 set, White
evaluated +1.10 to +1.39: a 14-line subset (5 e4, 5 d4, 2 c4, 2 Nf3) from which 3 e4, 3 d4, 1 c4 and 1 Nf3 lines
were drawn with a fixed seed (`random.Random(42).sample` within each first-move group). Evaluations are UHO's, in
centipawns for White.

| # | Line (White to move after 8 moves each) | UHO eval |
|---|---|---|
| 1 | 1. e4 e6 2. d4 d5 3. e5 c5 4. c3 Ne7 5. Nf3 Nec6 6. h4 b6 7. h5 h6 8. Rh3 a5 | +1.24 |
| 2 | 1. e4 e6 2. Qe2 d5 3. exd5 Qxd5 4. Nc3 Qd8 5. b3 Nf6 6. Bb2 Be7 7. O-O-O O-O 8. g4 c5 | +1.17 |
| 3 | 1. e4 c6 2. d4 d5 3. f3 dxe4 4. fxe4 e5 5. Nf3 Bg4 6. c3 Nf6 7. Bc4 Qc7 8. dxe5 Bxf3 | +1.32 |
| 4 | 1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 dxc4 5. a4 a5 6. Ne5 Na6 7. e4 Be6 8. Bxc4 Bxc4 | +1.35 |
| 5 | 1. d4 Nf6 2. Bf4 e6 3. e3 c5 4. Nf3 b6 5. Nc3 a6 6. d5 d6 7. dxe6 Bxe6 8. Ng5 d5 | +1.26 |
| 6 | 1. d4 Nf6 2. c4 c5 3. d5 e6 4. Nc3 exd5 5. cxd5 Bd6 6. Bg5 O-O 7. e3 Re8 8. Bd3 Bf8 | +1.33 |
| 7 | 1. c4 c6 2. e4 d5 3. e5 dxc4 4. Bxc4 Qd4 5. Qe2 Bg4 6. f3 Bf5 7. g4 Be6 8. Bxe6 fxe6 | +1.34 |
| 8 | 1. Nf3 d5 2. g3 g6 3. c4 dxc4 4. Na3 e5 5. Nxe5 Bxa3 6. Qa4+ b5 7. Qxa3 Bb7 8. Nf3 Nc6 | +1.25 |

The unbalanced starts show in the colour split: White won 301 games, Black 30, and 133 were drawn (28.7% draws,
against 46.3% across the CCRL list). SimpleChess scored 78.4% as White (+146 =72 -14) and 20.0% as Black
(+16 =61 -155).

**Rating method.** BayesElo's model (Remi Coulom, https://www.remi-coulom.fr/Bayesian-Elo/) with its default
settings, reproduced in a small fitter because BayesElo itself cannot hold 29 players at fixed ratings:

    f(D) = 1 / (1 + 10^(D/400))
    P(White wins) = f(eloBlack - eloWhite - eloAdvantage + eloDraw)
    P(Black wins) = f(eloWhite - eloBlack + eloAdvantage + eloDraw)
    P(draw)       = 1 - P(White wins) - P(Black wins)
    eloAdvantage = 32.8, eloDraw = 97.3                          (BayesElo defaults)

- Printed scale. BayesElo fits internal ratings and, after `mm`, prints `EloScale * internal + offset`, with
  `x = 10^(-eloDraw/400)` and `EloScale = 4x / (1+x)^2`, which is 0.9255 at the default eloDraw. CCRL's ratings are
  printed values, so the internal difference between SimpleChess and opponent j is `(R - anchor_j) / 0.9255`, and R
  comes out on CCRL's printed scale.
- Prior. BayesElo's default `prior 2` adds, for each player, `2 * 0.25 / games * games_vs_j` virtual draws per
  colour against each opponent: one virtual draw in total for SimpleChess, spread over its opponents by games
  played. The opponents' own share is negligible on CCRL, where each has thousands of games, and is not added.
- Only R is fitted, by maximum likelihood; the 95% interval is where the log-likelihood is within 1.92 of its
  maximum. The anchors' own CCRL error bars (+-7 to +-19) are not propagated into it.
- Fitting eloAdvantage and eloDraw from these games (BayesElo's `mm 1 1`) was tried and rejected: with the anchors
  on the printed scale, EloScale depends on the fitted eloDraw and the two run away together (16 losses to
  Stockfish 19 came out at 3741). The fitter was checked by recovering a synthetic engine's rating exactly.

**Results by opponent.** Score and performance are per 16 games; "Elo vs it" is the pentanomial estimate over the 8
opening pairs, with its 95% interval.

| # | Opponent (CCRL name) | CCRL 1-CPU | W | D | L | Score | Elo vs it (95%) | Perf |
|---|---|---|---|---|---|---|---|---|
| 1 | Stockfish 19 | 3784 +-13 | 0 | 6 | 10 | 3/16 | -255 [-359, -179] | 3529 |
| 2 | PlentyChess 8.0.0 | 3774 +-13 | 2 | 5 | 9 | 4.5/16 | -163 [-266, -81] | 3611 |
| 3 | Reckless 0.9.0 | 3766 +-11 | 1 | 6 | 9 | 4/16 | -191 [-284, -118] | 3575 |
| 4 | Viridithas 20.0.0 | 3747 +-12 | 0 | 6 | 10 | 3/16 | -255 [-359, -179] | 3492 |
| 5 | Alexandria 8.0.0 (8CPU* 3755) | 3722 +-10 | 2 | 5 | 9 | 4.5/16 | -163 [-266, -81] | 3559 |
| 6 | Halogen 16.0.0 | 3721 +-8 | 1 | 6 | 9 | 4/16 | -191 [-284, -118] | 3530 |
| 7 | Horsie 1.1.0 (8CPU* 3735) | 3709 +-10 | 0 | 7 | 9 | 3.5/16 | -221 [-286, -167] | 3488 |
| 8 | Motor 0.9.0 | 3695 +-9 | 3 | 6 | 7 | 6/16 | -89 [-189, -2] | 3606 |
| 9 | Koivisto 9.0 (8CPU* 3682) | 3632 +-8 | 2 | 6 | 8 | 5/16 | -137 [-203, -80] | 3495 |
| 10 | Avalanche 4.0.0 | 3597 +-16 | 4 | 4 | 8 | 6/16 | -89 [-215, +17] | 3508 |
| 11 | Black Marlin 9.0 (8CPU* 3622) | 3590 +-10 | 4 | 4 | 8 | 6/16 | -89 [-157, -27] | 3501 |
| 12 | Altair 7.0.0 (8CPU* 3632) | 3579 +-10 | 7 | 2 | 7 | 8/16 | 0 [-142, +142] | 3579 |
| 13 | Elixir 3.0 | 3567 +-10 | 5 | 2 | 9 | 6/16 | -89 [-189, -2] | 3478 |
| 14 | pawn 4.0 | 3553 +-13 | 6 | 3 | 7 | 7.5/16 | -22 [-96, +51] | 3531 |
| 15 | Petrel 4.0 | 3537 +-16 | 7 | 2 | 7 | 8/16 | 0 [-61, +61] | 3537 |
| 16 | Carp 3.0.1 | 3525 +-9 | 6 | 3 | 7 | 7.5/16 | -22 [-96, +51] | 3503 |
| 17 | Lambergar 1.5 | 3509 +-14 | 8 | 2 | 6 | 9/16 | +44 [-9, +98] | 3553 |
| 18 | Texel 1.11 (8CPU* 3584) | 3506 +-11 | 8 | 3 | 5 | 9.5/16 | +66 [-119, +305] | 3572 |
| 19 | Molybdenum 4.1 | 3432 +-10 | 7 | 6 | 3 | 10/16 | +89 [+2, +189] | 3521 |
| 20 | bitbit 1.7 | 3405 +-15 | 7 | 7 | 2 | 10.5/16 | +112 [+51, +182] | 3517 |
| 21 | Perseus 1.1 | 3397 +-10 | 8 | 6 | 2 | 11/16 | +137 [+80, +203] | 3534 |
| 22 | Pea 9.1 | 3383 +-16 | 9 | 6 | 1 | 12/16 | +191 [+118, +284] | 3574 |
| 23 | Frozenight 6.0.0 | 3362 +-9 | 9 | 5 | 2 | 11.5/16 | +163 [+81, +266] | 3525 |
| 24 | Xiphos 0.6 (8CPU* 3462) | 3354 +-7 | 8 | 4 | 4 | 10/16 | +89 [-48, +262] | 3443 |
| 25 | Tunguska 2.1 | 3349 +-19 | 10 | 3 | 3 | 11.5/16 | +163 [+59, +307] | 3512 |
| 26 | Catalyst 3.1.0 | 3335 +-16 | 10 | 4 | 2 | 12/16 | +191 [+91, +334] | 3526 |
| 27 | Maelstrom 3.3.0 | 3314 +-12 | 9 | 6 | 1 | 12/16 | +191 [+118, +284] | 3505 |
| 28 | Mantissa 3.7.2 | 3311 +-9 | 11 | 3 | 2 | 12.5/16 | +221 [+105, +416] | 3532 |
| 29 | c4ke 3.0 | 3303 +-14 | 8 | 5 | 3 | 10.5/16 | +112 [+51, +182] | 3415 |
| | **Total** | avg 3533 | **162** | **133** | **169** | **228.5/464** | | **3529 (BayesElo)** |

**How the games ended.** Checkmate 330, threefold repetition 100, fifty-move rule 32, insufficient material 1,
loss on time 1. SimpleChess never lost on time (its lowest clock in any game was 9.0 s); no engine crashed. Games
averaged 5.7 minutes; the longest, against bitbit, ran 737 plies after the opening in 14.9 minutes.

**Differences from CCRL testing.** UHO unbalanced openings instead of a general book up to 12 moves (hence far
fewer draws); no tablebases instead of up to 6-piece EGTB; 2'+1" on an i7-9700 without scaling to CCRL's i7-4770K reference,
and the i7-9700 is the faster CPU, so every engine searched somewhat more per move than CCRL's scaled time control
intends; 16 games per opponent, so the
per-opponent intervals are wide and the estimate's precision comes from all 464 games together.

## v3.1.0 — 2026-09-07

- Self-reports: `SimpleChess 3.1.0`
- Source VERSION at save time: `3.1.0`

Syzygy endgame tablebases (Fathom, MIT): interior WDL probing behind `SyzygyPath` /
`SyzygyProbeDepth` / `SyzygyProbeLimit` / `Syzygy50MoveRule`, `tbhits` in the info line;
verified against python-chess (correctness-only — no strength claim).

Two speed campaigns, every item bit-identical to 3.0.0's search (same nodes/score/PV at
fixed depth on 28 positions) — the strength gain below is node rate turned into depth:
- x86/AVX2 (Windows, i7-9700; 52 items tried, 28 kept): AVX2
  kernels for the int8 L1 dot (maddubs), fused accumulator updates, the L2 matvec and the
  epilogues; single-pass make, fused refresh, child-key TT prefetch, 64-byte weight
  alignment, Windows large pages for the TT and FT weights, PEXT magics, thin LTO, PGO.
  +131% nps @1T; 4T/1GB ≈ 90% of Stockfish 18's node rate on the 30-position suite (from 45%).
- NEON (macOS, M1 Max; 24 items tried, 5 kept): the AVX2 ideas
  ported — src→dst fused pass, 4-output×2-chain `vdotq` L1 dot (+23.7% alone), 64-wide
  fused chunks; the x86 items re-measured on ARM (four now x86-only behind `#if defined(__ARM_NEON)`:
  the 64-B allocator, the off-check pawn hash, the SEE tri-state, the persistent thread pool).
  +59% nps @1T over 3.0.0's Mac build; 4T/1GB ≈ 97% of Stockfish 18's node rate on the same
  suite (from 61%); midgame 1.00M nps @1T.
Build: Makefile/CMake pick `-march=native` on x86 hosts (`-mcpu=native` on Apple Silicon);
PGO builds keep `EXTRA` defines; PGO workload runs in a sandbox for deterministic profiles.

BEATS 3.0.0 by +77 Elo [+63, +92] LOS 100%: +173 =400 -39 (61.0%) over 612 games
@10+0.1s/1T/256MB, paired openings, pentanomial [0/22/146/120/18] (stopped at a ±15 Elo
95% CI; macOS M1 Max, both engines on SCNNUEv3-2026-08-30).

## v3.0.0 — 2026-08-30

- Self-reports: `SimpleChess 3.0.0`
- Source VERSION at save time: `3.0.0`

NNUE-only relabel and rewiring. The 2.5 architecture (FullThreats+PP_3Wide, HL 512,
SCN5 int8) was a major-class, format-incompatible change; 3.0.0 makes the line honest
and finishes the transition. Hand-crafted eval fully removed (evaluate.cpp/.hpp gone,
with the MaterialBlend/NNUEWeight/NNUEScale blend knobs): the engine requires its
network and exits with an error if none can be loaded. Nets are now date-keyed —
`SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5` (year-month-day so names sort chronologically);
discovery loads the newest own-major net, falling back to the newest lower-major net.
New net SCNNUEv3-2026-08-30 trained on an 879,778,971-position pool: the v2-5 + v2-6
self-play runs combined through the new dedup-combine (one row per unique position,
newest generation's label wins), shuffled on disk. Search is behavior-identical to
2.5 at shipping defaults (verified: same bestmove/score/nodes on a fixed-depth suite
with the same net). PGO build.

BEATS 2.5 by +127 Elo [+118, +136] LOS 100%: +1322 =1408 -270 (67.5%) over 3000 games
@300ms/1T/256MB, paired openings.

## v2.5 — 2026-08-23

- Self-reports: `SimpleChess NNUE 2.5`
- Source VERSION at save time: `2.5`

Threats graduation to mainline: FullThreats(59808)+PP_3Wide(4560) 512HL; +139 Elo [128,150] LOS 100% vs 2.4 over 2549g @100ms/4T/1GB. Net SCNNUEv2-5.scn5 (int8 SCN5). PGO build.

## v0.6 — 2026-07-29

- Self-reports: `Simple Chess NNUE 0.6`
- Source VERSION at save time: `0.6`

Net SCNNUEv0-6.scn: trained on 50M pool (all five self-play batches), lambda=0.8, patience-2 early-stop @ epoch 20. Data labeled by v0.5 fusion at NNUEWeight=50/NNUEScale=100 (first equal HCE/NNUE vote). CONVINCINGLY BEATS HCE: beat champion v1.7 +76 [+30,+126] LOS99.9, v1.0 +71 [+24,+121] LOS99.8, v1.2 +35; ~+61 avg, ~2709 on family BayesElo scale (anchor v1.6=2661.7) -> #1 in the entire SimpleChess family, above every hand-crafted version. Beat v0.5 +152. 50M chosen over 40M alternate (drops oldest v0.1 batch) by Sam's judgment. Carries threefold-rep fix; MaterialBlend default 0.

## v0.5 — 2026-07-28

- Self-reports: `Simple Chess NNUE 0.5`
- Source VERSION at save time: `0.5`

Net SCNNUEv0-5.scn: trained on 40M pool (all four self-play batches), lambda=0.8, patience-2 early-stop @ epoch 21. Beat v0.4 +152 Elo [+108,+202] LOS100 (148g d10 NNUE-only). REACHED HCE PARITY: even with champion v1.7 (-0), beat v1.6 +57 (sig), pooled +16 avg across HCE v1.0/1.2/1.6/1.7 field vs v0.4's -90 (592g, tournament suite, d10). Data labeled by v0.4 fusion (NNUEWeight=40, NNUEScale=100). Carries threefold-repetition fix; MaterialBlend default 0.

## v0.4 — 2026-07-28

- Self-reports: `Simple Chess NNUE 0.4`
- Source VERSION at save time: `0.4`

Net SCNNUEv0-4.scn: trained externally on ~10M self-play labeled by v0.3 fusion (NNUEWeight=40, NNUEScale=90, PSQT off). Tops the internal NNUE ladder: beat v0.3 +71, v0.2 +94, v0.1 +685 (depth10, NNUE-only). vs HCE gauntlet avg -90 Elo (v1.0 -76, v1.2 -117, v1.6 -47, v1.7 -122; 148g each). Includes threefold-repetition search fix (isRepetition 1->2): a 2-fold reaching into pre-root game history no longer scored as a forced draw. MaterialBlend default now 0 (pure net eval).

## v0.3 — 2026-07-25

- Self-reports: `Simple Chess NNUE 0.3`
- Source VERSION at save time: `0.3`

Net trained on pooled 20M search-labeled self-play (v0.2's 10M + fresh 10M generated at NNUEWeight=40 fusion labels), lambda=0.7, patience-2 early-stop @ epoch 18. A/B winner: 'pool' (20M) beat 'new' (10M-only), and pool beat v0.2 +53 Elo [95% CI +26,+79] LOS 100%, 500g depth 10 4-thread. Engine change vs v0.2: standalone Hash default raised to 4096MB.

## v0.2 — 2026-07-25

- Self-reports: `Simple Chess NNUE 0.2`
- Source VERSION at save time: `0.2`

PSQT+material anchor (classical piece-square tables, rescaled) replaces flat-material anchor on the pure-NNUE path; net trained on 10M search-labeled self-play (2x5M, fused-eval depth-8 labels) with lambda=0.7 blend, early-stopped epoch 10. Beat v0.1 net +285 Elo [95% CI +251,+325], 500g depth 10, 83.8%.

## v1.7 — 2026-07-24

- Self-reports: `Simple Chess 1.7`
- Source VERSION at save time: `1.7`

Tempo bonus: flat +16 cp for the side to move, added post-taper on every return path (lazy exit included). Standard hand-crafted-eval term (typical values 18-28 cp); 16 tuned in-engine via a 5-point sweep (8/12/16/20/28) against v1.6.4. Result: +20 Elo [+1, +40], LOS 97.8% at 300ms/500g (the deeper, more realistic control); ~neutral (+6) at 60ms. Zero NPS cost — a single add. Gauntlet placed it #2 at 2656, a statistical tie with v1.6 (2662, gap 6 +-17); promoted on the significant longer-TC evidence per Sam's call that the 300ms result is the more trustworthy signal.

## v1.6.4 — 2026-07-23

- Self-reports: `Simple Chess 1.6.4`
- Source VERSION at save time: `1.6.4`

Hotfix over v1.6.3: mate-proven and forced-move breaks fire during ponder too (gated on use_clock). A mate found on the opponent's clock kept re-deepening and poisoned the post-ponderhit search (full clock, no PV). Now: mate found -> stop deepening, play instantly on ponderhit with PV. Node-identical to v1.6.1 at fixed depth incl. mate lines.

## v1.6.4 — 2026-07-23

- Self-reports: `Simple Chess 1.6.4`
- Source VERSION at save time: `1.6.4`

Hotfix over v1.6.3: mate-proven and forced-move breaks fire during ponder too (gated on use_clock). A mate found on the opponent's clock kept re-deepening and poisoned the post-ponderhit search (full clock, no PV). Now: mate found -> stop deepening, play instantly on ponderhit with PV. Node-identical to v1.6.1 at fixed depth incl. mate lines.

## v1.6.3 — 2026-07-22

- Self-reports: `Simple Chess 1.6.3`
- Source VERSION at save time: `1.6.3`

Hotfix over v1.6.2 ponder: ponderhit now resets the clock so the budget is measured from ponderhit (pondering is the opponent's free time), not from go-ponder. v1.6.2 stopped instantly when the opponent out-thought the budget -> half-baked move from an aborted iteration, no PV. Node-identical to v1.6.1 at fixed depth.

## v1.6.2 — 2026-07-22

- Self-reports: `Simple Chess 1.6.2`
- Source VERSION at save time: `1.6.2`

Ponder support: search on opponent's clock (go ponder, no time enforcement until ponderhit); emit 'bestmove X ponder Y' from the root PV; ponderhit resumes budget enforcement. Node-identical to v1.6.1 at fixed depth/movetime.

## v1.6.1 — 2026-07-22

- Self-reports: `Simple Chess 1.6.1`
- Source VERSION at save time: `1.6.1`

Equal-position depth cap (clock games): stop deepening past depth 30 when |score|<=75cp and best move stable 3 iters. Node-identical to v1.6 at fixed depth/movetime. Validated: caps d43->30 in deep-equal positions, same move, ~all time saved.

## v1.6 — 2026-07-22

- Self-reports: `Simple Chess 1.6`
- Source VERSION at save time: `1.6`

Bishop pair {30,75} MG/EG when bishops cover both colors. Won a 3-way EG bake-off vs base v1.5 (1000g x2 seeds, depth 10): mid{30,75}=+23 (LOS 99%), opus{30,55}=+7, consensus{27,100}=~0. Consensus EG over-values the pair in this eval.

## v1.5 — 2026-07-22

- Self-reports: `Simple Chess 1.5`
- Source VERSION at save time: `1.5`

Remove v1.3 pawn-storm eval; keep v1.4 effort-time mgmt. Storm cost ~36 Elo in the 13-version round-robin (largest regression). Eval now == v1.2, search == v1.4.

## v1.4 — 2026-07-21

- Self-reports: `Simple Chess 1.4`
- Source VERSION at save time: `1.4`

Effort-based time scaling: replace the structurally-dead score-gap 'only move' exit with a root-node effort share signal

## v1.3 — 2026-07-20

- Self-reports: `Simple Chess 1.3`
- Source VERSION at save time: `1.3`

pawn storms: enemy pawns advancing on the king's file + neighbours, charged by proximity, with a separate cheap table for blocked (stalled) storms; MG-weighted. SC_STORM=100. Space term implemented but DISABLED (SC_SPACE=0): depth-10 attribution showed it inert alone (-2) and harmful with storm (-16 vs +26), redundant with v0.2 square control.

## v1.3 — 2026-07-20 — REVERTED

Rook knowledge (connected rooks, conditional 7th-rank infiltration, open/semi-open
file, rook behind passed pawn). Promoted on a 60ms sweep showing +49 Elo vs v1.2
(LOS 99.7%), then **reverted**: a depth-10 gauntlet had it at −14 vs v1.2 and −9 vs
v1.0. With search speed neutralised by fixed depth, the terms added no evaluation
quality — the 60ms gain was speed-derived, not knowledge. Live version rolled back
to v1.2; the code remains archived in the dev repo (renamed
to free the `v1.3` name for the pawn-storm release above).

## v1.2 — 2026-07-19

- Self-reports: `Simple Chess 1.2`
- Source VERSION at save time: `1.2`

progress scaling: halfmove-clock eval decay toward draw (no progress = no conversion); onset 16 plies, 20% floor at hmc=100, mate-safe. SC_PROGRESS=100. Sweep vs v1.1 neutral (+7 Elo, non-regressing).

## v1.1 — 2026-07-19

- Self-reports: `Simple Chess 1.1`
- Source VERSION at save time: `1.1`

go infinite disables opening book (analysis mode always searches). One-line UCI fix; eval/search identical to v1.0.

## v1.0 — 2026-07-18

- Self-reports: `Simple Chess 1.0`
- Source VERSION at save time: `1.0`

threats eval (classical design): strongly-protected/weak classification, minor/rook/king attacks by victim, hanging pieces (LPDO), safe pawn attacks + push threats; SC_THREAT=50 sequential-sweep winner (+70 Elo @ 60ms/200g vs v0.9, LOS 100%). Tree 0.48x.

## v0.9 — 2026-07-18

- Self-reports: `Simple Chess 0.9`
- Source VERSION at save time: `0.9`

no-increment time mgmt fix: for inc==0, spend 30% less/move + cap max at 2x optimum and 5% of clock (was 10%). Worst-case move at 60s no-inc 4.3s->2.1s. Increment/depth play unchanged (search identical to v0.8).

## v0.8 — 2026-07-17

- Self-reports: `Simple Chess 0.8`
- Source VERSION at save time: `0.8`

integrated king safety (extends v0.3 king-attack): shelter/exposure (queen-gated, hide-in-pawns), own-piece crowding, enemy denial + no-escape penalty, inactive-king-when-queens-off; SC_KSAFE=200 sweep winner (+89 Elo @ depth10/200g vs v0.7, LOS 100%)

## v0.7 — 2026-07-17

- Self-reports: `Simple Chess 0.7`
- Source VERSION at save time: `0.7`

adaptive continuous search width: root-gap x score x phase signal scales LMR/LMP/futility; endgames+decisive stay full-depth; SC_BREADTH=64 sweep winner (+38 Elo @ 60ms/100g vs v0.6, curve peak of 32-160)

## v0.6 — 2026-07-17

- Self-reports: `Simple Chess 0.6`
- Source VERSION at save time: `0.6`

clock: ply-scaled time allocation (less in opening), hard cap min(60s,10% remaining); only-move early exit (>=100cp over 2nd best, stable, depth>=16, real clock only)

## v0.5 — 2026-07-17

- Self-reports: `Simple Chess 0.5`
- Source VERSION at save time: `0.5`

Lazy SMP multithreading: shared TT, per-thread history/PV, main-thread TM, helper depth-stagger; Threads option default 8 (7.4x nps @ 8T)

## v0.4 — 2026-07-17

- Self-reports: `Simple Chess 0.4`
- Source VERSION at save time: `0.4`

pawn structure: doubled/isolated penalties, rank-scaled passers (blockade/support aware), EG advancement; scale-175 sweep winner (+106 Elo @ depth8/216g vs v0.3). Gauntlet 100g@60ms: +235 vs v0.1, +123 vs v0.2, +78 vs v0.3.

## v0.3 — 2026-07-16

- Self-reports: `Simple Chess 0.3`
- Source VERSION at save time: `0.3`

king-attack eval, amplitude-swept (katt67 best of 5 @ depth8/216g): +31 Elo @ depth 8, +-0 @ 60ms (358g pooled each)
