#!/usr/bin/env python3
"""
pgo_workload.py — PGO training workload for the engine (used by `make profile-build`).

Drives the (instrumented) engine over a spread of positions at a fixed depth so
the profiler sees representative search + NNUE-eval + accumulator code paths. The
build's -fprofile-generate=<dir> (and/or the LLVM_PROFILE_FILE env the Makefile
sets) decides where the .profraw lands; this script just does the work.

  LLVM_PROFILE_FILE=.../prof-%p-%m.profraw python3 <path-to>/pgo_workload.py <engine> <net> [depth]

<net> must match the engine's expected format (threats engine -> SCN4 float / SCN5 int8).
Weights are irrelevant to profiling (only the executed code paths matter), so a
throwaway correctly-shaped net is fine when no trained net exists yet. The engine
cannot START without a discoverable .scn5, so with a .scn4 <net> the newest
nets/*.scn5 also goes into the sandbox (startup only; the passes load <net> by
EvalFile); with no .scn5 anywhere the workload stops with an error.
"""
import sys, time, os, shutil, tempfile
import chess, chess.engine

engine_path = sys.argv[1]
net = sys.argv[2]
# depth: env PGO_DEPTH wins, then argv[3], else 11 (unset -> unchanged default).
depth = int(os.environ.get("PGO_DEPTH") or (sys.argv[3] if len(sys.argv) > 3 else 11))

# Deterministic profile: the engine's startup net discovery iterates exe_dir and
# cwd (src/uci.cpp try_load_default_net), so its branch counts -- and therefore the
# PGO layout -- change whenever the repo root gains a file (measured: two builds of
# identical source differed in 15 cold functions and 928 bytes of .text). Run the
# instrumented engine from a private sandbox holding ONLY a copy of the exe and
# nets/<net>, so every profile sees the same directory contents.
# PGO_NO_SANDBOX=1 restores running in place.
SANDBOX = None
if not os.environ.get("PGO_NO_SANDBOX"):
    SANDBOX = tempfile.mkdtemp(prefix="pgo-sandbox-")
    sb_exe = os.path.join(SANDBOX, os.path.basename(engine_path))
    shutil.copy2(engine_path, sb_exe)
    os.makedirs(os.path.join(SANDBOX, "nets"))
    sb_net = os.path.join(SANDBOX, "nets", os.path.basename(net))
    shutil.copy2(net, sb_net)
    if not net.endswith(".scn5"):
        import glob as _glob
        boot = sorted(_glob.glob("nets/*.scn5"))
        if not boot:
            sys.exit("pgo_workload: a .scn4 PGO_NET needs a .scn5 in nets/ for the engine to start")
        shutil.copy2(boot[-1], os.path.join(SANDBOX, "nets", os.path.basename(boot[-1])))
    engine_path, net = sb_exe, sb_net

# Opening / kiwipete / quiet middlegame / R+P endgame / heavy-piece / tactical.
_DEFAULT_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "2r3k1/1p3pp1/p2p3p/4n3/1PP1P3/P2r1PP1/3R2KP/3R4 b - - 0 1",
    "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - - 0 1",
]
# env PGO_FENS = path to a JSON list of FENs (a JSON list of FEN strings); unset -> the 6 above.
import json as _json
_fens_file = os.environ.get("PGO_FENS")
FENS = _json.load(open(_fens_file, encoding="utf-8")) if _fens_file else _DEFAULT_FENS

import os

def run_pass(net_path, tag, d):
    if SANDBOX and os.path.dirname(os.path.abspath(net_path)) != os.path.join(SANDBOX, "nets"):
        # a net discovered outside the sandbox (the float pass): bring it in too
        dst = os.path.join(SANDBOX, "nets", os.path.basename(net_path))
        shutil.copy2(net_path, dst)
        net_path = dst
    popen_kw = {"cwd": SANDBOX} if SANDBOX else {}
    eng = chess.engine.SimpleEngine.popen_uci(engine_path, **popen_kw)
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

if SANDBOX:
    shutil.rmtree(SANDBOX, ignore_errors=True)  # the .profraw went to LLVM_PROFILE_FILE, not here
