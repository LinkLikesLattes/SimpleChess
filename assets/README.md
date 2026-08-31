# assets/ — logos

One logo, several packagings. UCI has no mechanism for an engine to send a GUI
its logo — every GUI loads images itself, and each has its own expected format,
size, and filename convention. These files exist so every common convention is
covered from one place:

| File | What it's for |
|---|---|
| `simplechess.png` | The master logo (1024×1024). Used by this README, and by GUIs that accept an arbitrary image file for an engine. |
| `simplechess-100x50.bmp` | The classic engine-logo convention: 100×50, 24-bit BMP. Arena picks these up, and ChessBase/Fritz auto-load one placed **next to the engine binary and named exactly like it** (i.e. copy it to `simplechess.bmp` beside `simplechess.exe`). |
| `simplechess-130x65.bmp` | The same logo at 130×65 — the larger BMP size some GUIs and engine listings use. |
| `simplechess.ico` | Multi-resolution Windows icon (16–256 px) for the executable and shortcuts on Windows builds. |

The BMPs are pre-sized rather than scaled at load because the GUIs that want
them do no scaling of their own — a wrong-sized logo is simply not shown.
