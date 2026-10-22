# Phase 6 pilot: run and resume

This guide uses the existing Betawi `id-betawi-relay7-v0` profile and the frozen Phase 6 plan. Values measure the final single-round seed margin in player 0's direction. Phase 6 models and PUCT are experimental; H_v1 remains the default. A learned estimate or arena win is not a game-solving proof.

The production policy warm start has completed 1,000 steps and 20 checkpoints. Its selected step is 1,000 and its six value/trunk tensors are unchanged from the selected H8 relay model. Production self-play has completed 64 games, 2,154 committed choices and 623 admitted training rows. The production self-play update has completed its fixed 500 steps and 10 checkpoints. Inference timing, all 80 position cases, all 192 terminal strength games and the independent final audit are complete. Engineering checks passed, while both CUDA/PUCT challengers failed the strength gates. H_v1 remains the default; inspect the dashboard and acceptance report for the recorded evidence.

## Runtime and progress

Run the commands from PowerShell. CUDA uses the existing environment; this flow does not install PyTorch or replace the environment.

```powershell
Set-Location -LiteralPath 'F:\My Engine\Congklak Stockfish'
$phase6Python = 'D:\3D Project\Ministral-3-14B-Reasoning-2512\.venv\Scripts\python.exe'
py -3.12 -m congklak.phase6_progress snapshot
```

The read-only observer is [http://127.0.0.1:18766/](http://127.0.0.1:18766/). If it is already running, reuse it. Otherwise, start it in a separate terminal:

```powershell
py -3.12 -m congklak.phase6_progress serve --port 18766
```

The wrapper below records tool output and status in `build/phase6-progress/run-*.json` and its matching log. The dashboard reads actual checkpoint and game catalogs. Counters describe committed work, not engine strength. Other existing observer ports belong to separate views.

## Policy training, self-play and the update

Resume a stage by repeating its original command with the same directory, settings, backend, model and frozen sources. Do not launch two writers for the same stage. These commands are sequential: finish or stop the current CUDA worker before starting the next CUDA stage.

```powershell
py -3.12 -m congklak.phase6_progress run --tool phase6-warm -- $phase6Python tools/phase6-training.py warm
py -3.12 -m congklak.phase6_progress run --tool phase6-verify-warm -- $phase6Python tools/phase6-training.py verify-warm
py -3.12 -m congklak.phase6_progress run --tool phase6-selfplay -- $phase6Python -m congklak.phase6_selfplay generate --path data/phase6-selfplay-v1 --model data/training-phase6-warm-v1/model.json --backend cuda
py -3.12 -m congklak.phase6_progress run --tool phase6-verify-selfplay -- py -3.12 -m congklak.phase6_selfplay verify --path data/phase6-selfplay-v1
py -3.12 -m congklak.phase6_progress run --tool phase6-update -- $phase6Python tools/phase6-training-v2.py update
py -3.12 -m congklak.phase6_progress run --tool phase6-verify-update -- $phase6Python tools/phase6-training-v2.py verify-update
```

Training stores immutable checkpoint files and commits their checksums in `catalog.sqlite3` with FULL/WAL durability. A hard stop restores the last committed optimizer, model, sampler, draw counters and RNG states. Policy warm selection uses only scheduled development policy cross-entropy checkpoints. The resource V2 update validates CPU teacher/replay tensors before initializing CUDA; it preserves the declared mathematical training protocol. It uses its fixed final step 500, with 10 scheduled checkpoints and no development-best self-play selection. Existing completed exports and reports are preserved by matching completed reruns.

Self-play uses eight independent roots, 64 completed edge-backup simulations after root expansion, a 2,048-node limit per root and per-game RNG state. A transaction commits the whole completed decision batch and resulting game positions together. Interrupted trees are discarded and recomputed. A completed prefix survives either Ctrl+C or a process kill. CUDA future visits are reproducible only under the same batch schedule and numerical runtime; changing schedules is not a promise of identical floating-point search allocation.

Both games in a family must finish and avoid protected active-state bridges for that family's rows to enter training. Incomplete/capped games receive no invented terminal score. Rows with four or fewer active seeds are excluded. Deterministic thinning retains at most four observations of a raw state, with unit weights and individual game outcomes preserved. Full game provenance remains in the journal.

If a CUDA stage fails with a resource error, its committed catalog remains the recovery boundary. Rerun the unchanged command once that runtime can execute successfully. Changing the backend or source identity requires a new experiment directory; a CUDA allocation error does not become a draw or a successful strength result.

## Native position analysis

This command analyzes the ordinary opening with the warm policy/value export using the CPU backend. Its journal path is separate from production self-play.

```powershell
py -3.12 -m congklak.phase6_progress run --tool phase6-analysis -- py -3.12 -m congklak.phase6_analysis --path data/phase6-analysis-user --model data/training-phase6-warm-v1/model.json --backend native --simulations 64
```

Repeat the command to load a completed recommendation or recompute an unfinished tree. To request more work, repeat it with `--simulations 128`; the prior completed recommendation and its `analysis-64.json` remain available until a complete larger result can be published. Changing the position, model, backend or frozen source contract requires a new journal path. Targets that cannot complete within the node bound return interrupted status and retain the previous recommendation.

For a custom stable board, add `--position position.json`. The JSON contains exactly `pits`, `stores`, and `controller`; the two seven-pit rows plus two stores must total 98 seeds. Example:

```json
{"pits":[[7,7,7,7,7,7,7],[7,7,7,7,7,7,7]],"stores":[0,0],"controller":0}
```

Rankings display pits 1–7, visits, visit fractions and estimated player-0 margins. The underlying `search.chosen` and move APIs use indices 0–6. Visit fractions describe search allocation, not calibrated win probabilities. A positive player-0 margin favors row 0; a negative margin favors row 1. The immediate explanation contains the exact native move reason, deposited-seed count, normalized passes and resulting board. Bonus moves and forced passes may keep the same controller: PUCT backs up fixed player-0 values and changes perspective only while selecting at each normalized node.

Exact terminal values are labeled `terminal_outcome`. Ordinary PUCT recommendations are labeled `monte_carlo_estimate`; this analyzer does not retrieve the K=4 oracle or claim minimax proof. A prior-only result with no completed simulation is not a usable recommendation.

## Timing and strength evidence

After warm/update training and GPU acceptance finish, run these tools in an otherwise quiet project CPU/GPU window. Batching measurements cover sizes 1, 2, 4, 8, 16 and 32; the self-play queue remains capped at eight.

```powershell
py -3.12 -m congklak.phase6_progress run --tool phase6-inference-speed -- $phase6Python tools/phase6-resource-run.py speed
py -3.12 -m congklak.phase6_progress run --tool phase6-strength -- $phase6Python tools/phase6-resource-run.py benchmark
```

Inference measurements include feature extraction, queue waiting, transfers, forward evaluation, synchronization and validation. Startup and fixed warmups are separate observations. Native telemetry currently requires a separate feature-counter pass in addition to the frozen forward's feature pass; both actual passes are charged and this comparison concerns these implemented adapters.

The resource wrapper uses the frozen producer and output paths in `build/phase6-implementation-v2.json`. It records process working-set/private-commit peaks, CUDA allocator peaks, elapsed time and already committed units for that invocation. These measurements exclude earlier interrupted invocations and do not measure electricity. Reusing an existing resource report checks its identity and output checksum without claiming a new semantic test run.

The V2 benchmark first checks `build/phase6-implementation-v2.json`, mandatory rule/plan/registry/runtime/native-library/dependency pins, completed warm/update result and catalog witnesses, selected checkpoint checksums, and the exact self-play model/replay lineage using CPU-only admission. Torch payload and optimizer checks are a separate independent audit. An incomplete or incompatible export is rejected before CUDA initialization.

Strength uses 80 predeclared positions and 192 games over 24 development families, with paired seats. PUCT searches to the declared time budget using a large simulation ceiling, rather than stopping at self-play's 64 simulations. The native engines use the same declared search milliseconds; whole-wrapper wall time and overshoot are recorded for every engine. Startup and one batch-eight warmup per PUCT model are outside decision timing. Missing completed depths/simulations cause an explicitly counted legal-first-pit fallback.

The benchmark commits complete position cases and complete games, not in-progress arena trees or games. A killed case or unfinished arena game is recomputed; committed cases and games are validated and retained. Repeating the same command resumes the catalog. A completed rerun checks the canonical report against the journal and preserves published bytes and timestamps.

The primary strength metric is paired family mean final seed margin with the predeclared bootstrap uncertainty. A challenger needs a positive lower bound against both H_v1 and the H8 relay CPU baseline; WDL, incomplete pairs, fallbacks and latency are reported separately. These are reused development families, not fresh release validation. Sealed targets remain untouched, power is unmeasured, and no result automatically promotes a default engine.

Keep the frozen plan, source hashes, warm/update exports, checkpoints, replay manifest, journals, audit reports and failed-resource logs. Any later producer fix requires a declared new version before generating comparable evidence. See [scope and remaining product work](PHASE6_SCOPE_AND_ENGINE_READINESS.md).
