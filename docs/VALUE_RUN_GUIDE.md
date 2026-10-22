# Run and resume the fixed-policy value experiment

Run from `F:\My Engine\Congklak Stockfish`. The predeclared experiment is in `docs/VALUE_SCOPE.md`, `build/value-plan.json` and `build/value-implementation.json`. All 1,647 preceding accepted inputs stay immutable. Use a new experiment version for changed sources, settings or data ownership; do not edit a producer and reuse its journal.

`policy_control` uses the accepted joint-refined model at `data/quality-train-v1/model.json`. `value_refit` uses `data/value-train-v1/model.json`. Both use unchanged `quality_search`, c_puct=0.25 and the uncached structured CPU evaluator. Only the 65 scalar parameters in `layers.4.weight/bias` train; six shared-trunk/policy tensors, totaling 12,807 parameters, must remain bit-exact. H_v1 remains the default while this experimental comparison runs.

## Runtime and progress

Training and actual tensor verification use the CUDA Python already recorded in `build/training-runtime.json`. In PowerShell:

```powershell
$valuePython = (Get-Content -LiteralPath build/training-runtime.json -Raw | ConvertFrom-Json).python
```

Use the desktop `python` for CPU benchmark, analysis and acceptance. They admit the immutable model witness without importing Torch. The independent full audit launches its tensor worker in the pinned CUDA interpreter.

The observer is `http://127.0.0.1:18766/`. Start it only if this task's observer is not already serving the port:

```powershell
python -m congklak.value_progress serve --port 18766
```

Wrap invocations so their events, logs and real child exit codes appear on the dashboard. A completed invocation does not establish a complete campaign: check semantic status, durable counts and final acceptance. In particular, cooperative training interruption may return `interrupted` without a final model.

## Training and model verification

The fixed contract is 1,500 steps, batch 256, learning rate 0.001, seed 20261018, checkpoint every 50 and AdamW/fp32 with zero weight decay. The loss is Huber(delta 4) on fixed-player-0 **seed margins** against original-root H8 finite-choice teacher targets. Policy loss is zero. Policy action targets enter descriptive metrics only. No new self-play returns, oracle labels or larger network enter this treatment. Development metrics cannot choose a checkpoint or settings.

The admitted corpus contains 7,592 original training rows after 408 ownership exclusions, plus 1,000 descriptive development rows. Protected ownership now includes all 384 previous quality games. A protected quotient hit in a selected training root excludes its entire selected family. Original data and reasons remain intact. These exclusions restrict new optimizer draws. The warm ancestor trained on 8,000 roots and the joint-refined ancestor on 7,669; this refit does not erase their historical training/development exposure.

Run or resume the identical production command:

```powershell
python -m congklak.value_progress run --tool value-training -- "$valuePython" tools/value-train.py train --device cuda
```

The catalog is `data/value-train-v1/catalog.sqlite3`. It commits checkpoint metadata only after durable tensor publication. Checkpoints retain all model tensors, only the two value-head optimizer states, draw counts, sampler/Python/Torch/CUDA RNG and the frozen numeric/data identity. A hard kill may leave an uncommitted file; the catalog remains authoritative, that file remains diagnostic evidence and unfinished work is recomputed. A cooperative stop can add an intermediate checkpoint. Resume from the same compatible directory; do not select a checkpoint manually.

Only fixed-final step 1,500 publishes `model.json` and `result.json`. Verify every actual tensor payload, frozen tensor block and final export before CPU use:

```powershell
python -m congklak.value_progress run --tool value-model-verification -- "$valuePython" tools/value-train.py verify
```

This publishes `build/value-model-witness.json`. CPU admission checks its current dependencies, actual PT/catalog hashes, final model mapping and fixed-policy ownership. Benchmark admission also directly compares the six frozen exported tensor mappings against the source. The independent audit separately deserializes every production checkpoint and verifies optimizer ownership, RNG and frozen bytes; a checksum claim is not a substitute for that audit.

Training clocks and allocator peaks describe their documented invocation scope, excluding prior invocations. They are not whole-machine or electricity observations. Completed training reruns validate the same immutable final result rather than adding steps.

## Equal-time comparison and recovery

After genuine model verification:

```powershell
python tools/benchmark-value.py admit
python -m congklak.value_progress run --tool value-strength -- python tools/benchmark-value.py run
```

`data/value-benchmark-v1/work.sqlite3` retains 80 cases and 640 measurements: four engines at 25/100 ms. Each candidate faces H_v1 and H8 relay CPU on 24 paired-seat families at 25 ms, totaling 192 games. Candidates share c_puct=0.25, parent neural FPU, fixed-player-0 backup,2,048-node / 100,000-backup bounds and zero root noise. Both baselines use depth 16 / TT 16 MiB/PVS on the same declared performance core. Comparison order rotates by family and alternates seats.

The 48 H8 case families and 24 arena families exclude all prior arena family IDs and the preceding quality experiment's 48 H8 case families, and are disjoint from each other. Older diagnostic case families can recur. All remain development data with ancestral exposure. The 32 known-exact diagnostics form a separate cohort; this is not unseen or sealed release validation.

Cases and entire games commit individually with FULL/WAL durability. A kill preserves committed units and recomputes the unfinished case/game. Mutable search trees are not checkpointed. Resume with the identical command. Source/model/plan drift, concurrent writers, holes, unexpected game ordering and rehashed semantic corruption are refused. For a small retained fixture only:

```powershell
python tools/benchmark-value.py run --run build/my-value-fixture/run --output build/my-value-fixture/result.json --case-limit 2 --no-arena
```

The whole decision clock includes reset, search and validation. Only floored positive remaining milliseconds enter search; an exhausted budget skips search and becomes an explicit unusable-decision fallback. Zero never turns into an unlimited call. Physical forwards, relay feature work, wall overshoot and fallback counts remain retained. Nominal budgets are not hard real-time guarantees.

`build/value-benchmark.json` holds all comparisons. `build/value-benchmark-resources.json` records the original producing invocation after module import: admission, model setup, remaining units, validation, journal and report publication. Prior committed counts identify resumed work. The scope excludes module/Python startup and previous invocations; the progress wrapper has its separate surrounding clock. Resources publish before completion metadata. A completed rerun validates those original resources, preserves equal bytes/timestamps and repairs a missing semantic report from committed state. Contradictory publications or a missing committed resource witness are refused.

The primary gate requires 24 terminal pairs, zero fallbacks and a positive lower 95% family-bootstrap seed-margin bound against both baselines. All four intervals are exploratory and unadjusted. The refit-minus-control mean is descriptive over all 24 fully matched families, with no extra confidence interval or gate; missing matched pairs produce a null mean and their completed-family count. All variants/outcomes remain visible; neither value loss nor improved MAE automatically promotes the engine.

## Recoverable position analysis

```powershell
python -m congklak.value_progress run --tool value-analysis -- python -m congklak.value_analysis --path data/value-analysis-demo --profile value_refit --simulations 128
```

For another position, pass a JSON file with `--position`: two seven-pit rows, two stores and a controller, totaling 98 seeds. Another position/profile/model identity requires another analysis directory. Increasing requested backups retains the previous completed recommendation while a new bounded tree is recomputed. An interrupted tree publishes no partial recommendation. Postcommit missing JSON exports repair from the durable journal. Completed reruns preserve SQLite and export bytes/mtimes. Margins and moves are estimates; visit fractions are allocation, not calibrated probabilities or proofs.

## Acceptance and retained audit

Author suites and independent group/full commands are **prepublication operations**. They create fresh fixtures and may rewrite durations, reports and retained-directory bindings. After immutable acceptance, use read-only `python tools/record-value.py --check-only`, or a separately versioned verification report for new live fixtures.

```powershell
& $valuePython tests/value_learning_acceptance.py
python tools/benchmark-value.py acceptance
```

`tools/value-audit.py` coordinates independent admission, frozen-policy/literal checks, CUDA state and recovery tests. Its analyzer/case recovery groups require the genuine model; its arena group requires all 192 production games. Raw killed states, before/after commit markers, backups, refusal logs and optimizer/RNG comparisons remain in `build/value-audit`. After every group completes:

The plan, targets and CUDA recovery groups were executed with the exact coordinator retained as `tools/value-audit-executed-v1.py`. Their original report bytes remain in `build/value-audit/executed-v1`. Derived reports relocate that old source binding to its immutable snapshot; original outcomes, methods, durations and other inputs remain unchanged. They were not rerun. Later groups and the full production audit use the corrected coordinator, which tightens packed-byte checks in two full-audit routines and adds collector provenance. The collector and recorder expose both versions and count the original four methods once.

```powershell
python tools/value-audit.py collect
python tools/value-audit.py full
```

Once the full campaign, independent proof, demo/rerun witness and final documents are complete:

```powershell
python tools/record-value.py --check-only
python tools/record-value.py
```

The new aggregate requires four learning, thirteen benchmark and ten independent unique executed methods. Prior accepted search checks remain protected evidence and are not counted again. The CPU-only recorder checks all 1,647 prior input hashes, genuine value/frozen-policy proof, actual checkpoint files, all 80 / 640 / 192 units, literal game replays, paired statistics, original resources, analysis stability and retained diagnostics. Engineering acceptance is distinct from playing strength.

`build/value-acceptance.json` is immutable. A completed identical recorder rerun preserves its bytes and modification time. Dynamic progress logs, its own acceptance output and its later derivative rerun witness are excluded from the input map. Changed finalized evidence needs a separately recorded version. The initial board remains unsolved; energy use, exact-memory integration and release packaging remain separate work.
