# Changelog

Release history for Simple Chess. An entry is added here whenever a tested candidate is promoted to the live `simplechess` binary via `tools/version.py promote`. Every version — accepted or not — is also kept as a runnable binary in `versions/`.

<!-- new entries inserted below this line by `version.py promote` -->

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

PSQT+material anchor (Stockfish 14 psqt.cpp/types.h, rescaled) replaces flat-material anchor on the pure-NNUE path; net trained on 10M search-labeled self-play (2x5M, fused-eval depth-8 labels) with lambda=0.7 blend, early-stopped epoch 10. Beat v0.1 net +285 Elo [95% CI +251,+325], 500g depth 10, 83.8%.

## v1.7 — 2026-07-24

- Self-reports: `Simple Chess 1.7`
- Source VERSION at save time: `1.7`

Tempo bonus: flat +16 cp for the side to move, added post-taper on every return path (lazy exit included). Standard top-HCE term (SF classical 28, Ethereal 20, Weiss 18); 16 tuned in-engine via a 5-point sweep (8/12/16/20/28) against v1.6.4. Result: +20 Elo [+1, +40], LOS 97.8% at 300ms/500g (the deeper, more realistic control); ~neutral (+6) at 60ms. Zero NPS cost — a single add. Gauntlet placed it #2 at 2656, a statistical tie with v1.6 (2662, gap 6 +-17); promoted on the significant longer-TC evidence per Sam's call that the 300ms result is the more trustworthy signal.

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
to v1.2; the code remains archived under `versions/v1.3-rooks-reverted/` (renamed
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

adaptive continuous search width (SF-style): root-gap x score x phase signal scales LMR/LMP/futility; endgames+decisive stay full-depth; SC_BREADTH=64 sweep winner (+38 Elo @ 60ms/100g vs v0.6, curve peak of 32-160)

## v0.6 — 2026-07-17

- Self-reports: `Simple Chess 0.6`
- Source VERSION at save time: `0.6`

clock: SF-style ply-scaled time allocation (less in opening), hard cap min(60s,10% remaining); only-move early exit (>=100cp over 2nd best, stable, depth>=16, real clock only)

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
