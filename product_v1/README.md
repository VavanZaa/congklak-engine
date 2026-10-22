# Congklak local app

Start from the project folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File tools/start-product-v1.ps1
```

The local app opens at **http://127.0.0.1:18767/**. Keep its engine window running; Ctrl+C stops it. For an existing Python environment or an in-app browser, start the server without opening another browser:

```powershell
python -B -m congklak.product_v1
```

Python 3.12 and the existing `build/congklak_native.dll` are required. If the DLL has not been built, run `tools/build.ps1` once with Visual Studio C++ Build Tools installed. No Torch, CUDA, Node packages or internet connection are required to use the app.

Choose **Analyze / two players** to explore both sides, or **Play H_v1 · you are A** to play the CPU engine. Pit labels in the app are **A1–A7 and B1–B7**. The underlying protocol indexes pits 0–6. Filled legal pits can be selected directly. A mandatory empty-row pass has an explicit button; the CPU handles its own passes. Stop also returns the table to two-player mode so the CPU will not immediately restart.

Analyze shows only complete search depths, tied best moves, the score's player perspective, a conditional continuation and replayed sowing/capture facts. The displayed estimate is the store margin at that search depth. It is not a win probability or a guarantee of the game's final outcome. H_v1 is the fixed CPU evaluator; the app does not load experimental learned models.

**Edit position** accepts 14 pit counts, two stores and the player to move, totaling exactly 98 seeds. Its JSON reader accepts a position object or this app's exported table envelope. The declared Betawi seven-pit single-round profile is fixed; refill, simultaneous opening and closed-pit variants are unsupported.

Save boards by name. Saved analyses retain completed depths automatically. After restarting, expand **Your saved table**, load the saved analysis and choose **Resume** with a depth at least as high as the saved depth. Interrupted work after the latest complete depth is recomputed. Changed rules or engine builds require a new analysis. A result from another board is labeled as previous and cannot be applied to the current board.

User data resides in `data/product-v1/`, separate from research campaigns, and is marked ineligible for research training. The first version retains up to 100 saved boards and 1,000 analysis jobs; no automatic deletion is performed. Export downloads position, exact rules/build identity, limits and the latest report as JSON. This app binds only to loopback and has no publishing or remote-access feature.

Acceptance:

```powershell
python -B -m unittest discover -s tests -p test_product_v1.py -v
```

Tests use only `data/product-v1-fixtures/`. They include an actual server termination and restart, retaining the durable completed report and resuming the saved job. Their receipt is `build/product-v1/recovery.json`.
