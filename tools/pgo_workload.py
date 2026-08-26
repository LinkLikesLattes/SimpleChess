#!/usr/bin/env python3
"""
pgo_workload.py — PGO training workload for the engine (used by `make profile-build`).

Drives the (instrumented) engine over a spread of positions at a fixed depth so
the profiler sees representative search + NNUE-eval + accumulator code paths. The
build's -fprofile-generate=<dir> (and/or the LLVM_PROFILE_FILE env the Makefile
sets) decides where the .profraw lands; this script just does the work.

  LLVM_PROFILE_FILE=.../prof-%p-%m.profraw python3 tools/pgo_workload.py <engine> <net> [depth]

<net> must match the engine's expected format (threats engine -> SCN4 float / SCN5 int8).
Weights are irrelevant to profiling (only the executed code paths matter), so a
throwaway correctly-shaped net is fine when no trained net exists yet.
"""
import sys, time
import chess, chess.engine

engine_path = sys.argv[1]
net = sys.argv[2]
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 11

# Opening / kiwipete / quiet middlegame / R+P endgame / heavy-piece / tactical.
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2r3k1/1p3pp1/p2p3p/4n3/1PP1P3/P2r1PP1/3R2KP/3R4 b - - 0 1",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 1",
]

import os

def run_pass(net_path, tag, d):
    eng = chess.engine.SimpleEngine.popen_uci(engine_path)
    eng.configure({"OwnBook": False, "Threads": 1, "EvalFile": net_path})
    total = 0
    t0 = time.perf_counter()
    for fen in FENS:
        info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=d),
                           info=chess.engine.INFO_ALL)
        total += info.get("nodes") or 0
    dt = time.perf_counter() - t0
    eng.quit()
    print(f"pgo_workload[{tag}]: {total:,} nodes in {dt:.1f}s ({total/dt:,.0f} nps), depth {d}")

# Primary pass: the passed net (quant SCN5 for the shipping play path).
run_pass(net, "quant" if net.endswith(".scn5") else "primary", depth)

# §2.5: also profile the FLOAT path (eval_float) — the path self-play labels with
# (.scn4). WITHOUT this pass, a quant-only profile marks eval_float as dead code and
# the -fprofile-use rebuild pessimizes it badly (~2.5x slower than LTO, measured).
# Weights are irrelevant to profiling (only executed code paths matter), so any
# correctly-shaped .scn4 works: use PGO_FLOAT if set, else auto-discover one. A
# shallower depth keeps the slow float path from dominating workload wall-time.
import glob
float_net = os.environ.get("PGO_FLOAT")
if not float_net:
    cands = sorted(glob.glob("data/*.scn4") + glob.glob("nets/*.scn4") + glob.glob("*.scn4"),
                   key=lambda p: os.path.getmtime(p), reverse=True)
    float_net = cands[0] if cands else None
if float_net and os.path.exists(float_net):
    run_pass(float_net, "float", max(6, depth - 2))
else:
    print("pgo_workload: no .scn4 float net found -> eval_float NOT profiled "
          "(set PGO_FLOAT or place a .scn4 in data/) — float/self-play path will be slow")
