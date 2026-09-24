SimpleChess -- Android arm64 build
==================================

Files
  simplechess            the engine (UCI protocol), 64-bit Arm, fully static (no loader, no libraries).
                         The neural network is EMBEDDED in this file: nothing else needs copying.
  README-ANDROID.txt     this file

Which archive
  ...-android-arm64          any phone SoC from about 2018 on (Cortex-A75/A55 and newer, i.e. anything
                             with the Arm dot-product extension). This is the build to use.
  ...-android-arm64-generic  older 64-bit SoCs without dot-product (2016-2017 era). Same engine, but its
                             int8 network layers run as scalar code, so it is much slower. Use it only
                             if the arm64 build does not start on your device.

Install in DroidFish
  1. Copy `simplechess` into DroidFish's engine folder
     (by default DroidFish/uci on internal storage, i.e. /storage/emulated/0/DroidFish/uci).
  2. In DroidFish, Settings > Engine settings > pick `simplechess`.
  Any other UCI front-end or tournament harness: the single file is all it needs.

Other networks
  A different SCNNUEv<MAJOR>-<date>.scn5 placed beside the binary (or in a `nets/` subfolder) is
  used instead when it is newer than the embedded one; the UCI option EvalFile overrides either.

Notes
  Built by the repository's GitHub Actions workflow (.github/workflows/android.yml) from the
  tagged source: a plain LTO build, no profile-guided optimisation and no per-SoC tuning, so it
  runs somewhat below a native tuned build. The front-end sets Hash and Threads; the engine's
  compiled-in defaults assume a desktop.
  Source and license (GPL-3.0): https://github.com/LinkLikesLattes/SimpleChess
