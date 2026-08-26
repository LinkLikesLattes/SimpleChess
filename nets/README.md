# nets/ — networks, checkpoints, and the versioning convention

This directory holds the NNUE evaluation networks and their training runs. The
layout and naming are deliberately regular so a file always identifies itself.

## Layout

```
nets/
├── SCNNUEv0-1.scn      deployed net for engine v0.1
├── SCNNUEv0-2.scn      deployed net for engine v0.2
├── v0-1/               v0.1's training run   (best.pt, last.pt, metrics.jsonl)
├── v0-2/               v0.2's training run
└── README.md           this file
```

- **Top level = deployed nets only**, one per engine version, named
  `SCNNUEv<VERSION>.scn`. These are the float32 files the C++ engine loads.
- **`v<VERSION>/` subdirs = training checkpoints** for that version: the PyTorch
  `best.pt` / `last.pt` and the per-epoch `metrics.jsonl`. Never deployed
  directly; `train/export.py` turns a `.pt` into the deployed `.scn`.

## Naming convention: `SCNNUEv<VERSION>.scn`

The version's dot is written as a **hyphen** (`0.1` → `SCNNUEv0-1.scn`) so the
filename contains exactly one dot and `.scn` is unambiguously the extension, no
matter how a tool splits the name.

This name is **derived from the `VERSION` file, never hand-typed**:
- The engine binary is compiled to look for `SCNNUEv<VERSION>.scn` beside it
  (`src/uci.cpp` `kDefaultNetName`, built from the `SC_VERSION_FS` macro, which
  the Makefile computes as `$(subst .,-,$(VERSION))`).
- `train/train.py` defaults its checkpoint dir to `nets/v<VERSION>/` and prints
  the exact `export.py` command with the right output name.
- `tools/version.py` archives the net matching the current `VERSION` and records
  its name + size + SHA-256 in the version's `meta.json`.

Bumping `VERSION` therefore retargets everything at once. A bump made before its
net is trained is safe: the engine loudly falls back to the newest net it can
find (`info string NNUE loaded: ... (expected SCNNUEv0-3.scn ...)`) and warns if
there is none — it never silently drops to HCE.

## A/B variants (testing two candidates for one version)

To compare two candidate nets for the same version — e.g. a net trained on a
pooled dataset vs. one trained on only the newest batch — give each a **tag**:

```
python3 train/train.py data/pool20m.bin  --lam 0.7 --tag pool   # -> nets/v0-3-pool/
python3 train/train.py data/new10m.bin   --lam 0.7 --tag new    # -> nets/v0-3-new/
python3 train/export.py nets/v0-3-pool/best.pt nets/SCNNUEv0-3-pool.scn
python3 train/export.py nets/v0-3-new/best.pt  nets/SCNNUEv0-3-new.scn
```

Tagged nets are `SCNNUEv<VERSION>-<TAG>.scn` and coexist without touching the
plain `SCNNUEv<VERSION>.scn`. To gauntlet them, each candidate is a self-contained
(binary + net) pair in its own dir, with its tagged net linked under the base name
the binary loads (`SCNNUEv<VERSION>.scn`) — see `test/` build dirs. The gauntlet
winner is copied to the plain `SCNNUEv<VERSION>.scn`; that is the one that gets
built into root and archived.

## Full cycle for a new version N

```
echo N > VERSION                                     # e.g. 0.3
# ... train self-play data (see train/selfplay.py) ...
python3 train/train.py data/selfplay_10m.bin --lam 0.7   # -> nets/vN/best.pt
python3 train/export.py nets/vN/best.pt nets/SCNNUEvN.scn
make                                                 # root binary now loads it
# ... gauntlet; if accepted: ...
python3 tools/version.py save vN -m "..."            # archive binary+source+net
python3 tools/version.py promote vN                  # install as live + CHANGELOG
```

## v0.1 is a legacy exception

v0.1 was built before this convention. Its **archived** binary
(`versions/v0.1/`) looks for `simplechessnnue.scn`, so that snapshot keeps its
net under the old name (beside the binary, for the archive to run). The
deployed copy in this directory follows the current convention
(`SCNNUEv0-1.scn`). Everything from v0.2 on is uniform.
