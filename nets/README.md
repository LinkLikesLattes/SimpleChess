# nets/ — the evaluation network

This directory holds the NNUE network the engine plays with. The repo always
ships exactly **one** net — the current one. Older networks are available from
older [releases](https://github.com/LinkLikesLattes/SimpleChess/releases); each
release's source archive contains the net that version shipped with.

## Naming: `SCNNUEv<MAJOR>-<YYYY-MM-DD>.scn5`

Example: `SCNNUEv3-2026-08-30.scn5` — the network for the 3.x engine line,
exported/quantized on 30 Aug 2026.

- **`<MAJOR>`** is the engine's major version and encodes **format
  compatibility**: a major bump means the network architecture or file format
  changed and older/newer nets are not interchangeable across it.
- **`<YYYY-MM-DD>`** is the export date, year-month-day **specifically so
  filenames sort chronologically**. Retraining the network does *not* bump the
  engine version — within a major, nets are distinguished by date alone.

## Discovery

At startup the engine scans `nets/` (and the directory beside the binary) and
loads, in order of preference:

1. the **newest net of its own major** (by date),
2. else the newest net of a **lower** major (a transition fallback — it will say
   so on stdout),
3. else it **exits with an error**. There is no fallback evaluation; the engine
   does not run without a network.

A net of a *higher* major is never loaded — it may be format-incompatible with
this build. To use a specific file regardless of discovery, set the UCI option
`EvalFile` to its path.

## Format

`.scn5` (file magic `SCN5`) is the pre-quantized int8 playing format: an int8
feature transformer plus per-bucket layer weights, quantized once from the
trainer's float export. The float intermediate (`.scn4`, magic `SCN4`) is a
training artifact and is not shipped in the repo.

The network is trained entirely on the engine's own self-play; the trainer and
training data live outside this repository.
