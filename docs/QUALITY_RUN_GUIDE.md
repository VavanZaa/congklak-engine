# Run and resume the compact quality experiment

Run commands from `F:\My Engine\Congklak Stockfish`. The frozen producer manifest is `build/quality-implementation.json`; `docs/QUALITY_SCOPE.md` declares the experiment. Existing accepted sources, DLLs, data and models stay immutable. Use a new experiment version for a changed protocol or producer; do not edit a producer and reuse its production journal.

The four profiles are `warm_control`, `warm_search`, `refined_control` and `refined_search`. The model factor is the existing warm export versus a fixed-final joint refinement; the search factor is c_puct 1.5 versus 0.25. All use the same uncached structured CPU adapter. H_v1 remains the default; these commands run an experimental comparison.

## Runtime and dashboard

Training and tensor verification use the existing CUDA interpreter recorded in `build/training-runtime.json`, currently `D:\3D Project\Ministral-3-14B-Reasoning-2512\.venv\Scripts\python.exe`. Load its path in PowerShell:

```powershell
$qualityPython = (Get-Content -LiteralPath build/training-runtime.json -Raw | ConvertFrom-Json).python
```

Use the desktop `python` interpreter for CPU inference, benchmarking, analysis and acceptance. Those tools refuse to import Torch. The independent full audit launches its tensor worker through the pinned training interpreter.

The dashboard is at `http://127.0.0.1:18766/`. Start its observer only when this task's observer is not already serving that port:

```powershell
python -m congklak.quality_progress serve --port 18766
```

Wrap work so its events and actual child exit code appear on the dashboard:

```powershell
python -m congklak.quality_progress run --tool quality-training -- "$qualityPython" tools/quality-train.py train --device cuda
```

The new progress CLI propagates the child's exit code to the shell. A completed child invocation is not sufficient evidence that a multi-invocation campaign is finished: check its reported semantic status, committed counts and final acceptance. In particular, a cooperatively interrupted training invocation can return a saved `interrupted` result without producing a final model. The dashboard's optimizer counter comes from durable checkpoints; acceptance requires the genuine fixed-final model.

## Training and verification

The production contract is 1,500 optimizer steps, batch 256, checkpoint every 50, seed 20261015, all 12,872 parameters trainable. The original H8 root teacher margins supervise the residual value; actor-relative original-root softQ at two-seed temperature supervises legal policy. The admitted corpus has 7,669 training rows after 331 ownership exclusions, plus 1,000 descriptive development rows. The data and budget are frozen; development metrics do not select another checkpoint.

These exclusions restrict this treatment's new optimizer draws. The warm ancestor previously learned from the original 8,000-row corpus; filtering new draws does not remove that historical exposure from its weights.

Run or resume the same production training command:

```powershell
python -m congklak.quality_progress run --tool quality-training -- "$qualityPython" tools/quality-train.py train --device cuda
```

Its journal is `data/quality-train-v1/catalog.sqlite3`. Committed checkpoints contain model, AdamW moments, draw counters, sampler/Python/Torch/CUDA RNG and the numeric/data identity. A hard interruption may leave a checkpoint file not committed in the catalog; the catalog remains authoritative, the orphan remains diagnostic evidence, and unfinished work is recomputed. A later invocation restores the latest valid committed checkpoint. A cooperative stop may also save an additional intermediate checkpoint. Do not delete the catalog or select a checkpoint manually.

Only the fixed final step exports `data/quality-train-v1/model.json` and `result.json`. Validate every actual checkpoint and the export mapping before CPU benchmarking:

```powershell
python -m congklak.quality_progress run --tool quality-model-verification -- "$qualityPython" tools/quality-train.py verify
```

Verification publishes `build/quality-model-witness.json`. CPU admission checks that proof's current input hashes, catalog, actual checkpoint files, model identity, fixed-final contract and result. The independent auditor separately deserializes every actual checkpoint and verifies model/optimizer/RNG tensors; the CPU proof does not replace that audit.

Training's `elapsed_seconds_this_invocation` and memory peaks describe that invocation. They exclude previous invocations and are not whole-machine or electricity measurements. Completed reruns validate immutable results rather than adding training steps.

## Equal-time comparison and recovery

Run admission first:

```powershell
python tools/benchmark-quality.py admit
```

Run or resume the complete campaign:

```powershell
python -m congklak.quality_progress run --tool quality-strength -- python tools/benchmark-quality.py run
```

The journal is `data/quality-benchmark-v1/work.sqlite3`. Eighty cases contain 960 measurements: six engines at 25/100 ms. Four challengers each face both H_v1 and H8 relay CPU at 25 ms over 24 paired families, totaling 384 games. The same declared performance core, baseline depth/TT/PVS settings and bounded PUCT limits apply throughout. Cases and entire games commit individually using FULL/WAL durability. A killed invocation preserves committed units and recomputes its unfinished case or game; it does not save an active mutable search tree. Resume with the identical command and directory. Concurrent writers, changed identities, case holes, unknown arena order and rehashed semantic corruption are refused.

The decision clock charges cache reset, search and validation. Remaining search time is floored to a positive integer millisecond budget. If no whole millisecond remains, search is skipped and the unusable decision is explicitly counted as a fallback; zero never becomes an unlimited search call. Physical forwards and relay feature work are retained independently from search allocation. Wrapper-wall overshoot is reported rather than hidden; nominal budgets are not hard real-time guarantees.

The primary evidence is `build/quality-benchmark.json`, with original invocation resources in `build/quality-benchmark-resources.json`. That resource clock starts inside the producer after module imports and includes admission, startup, validation, remaining units and publication for that invocation. Its prior committed counts identify resumed work. It excludes Python/module startup and previous invocations; the progress runner has its own surrounding invocation clock. Resources publish before the completion marker. Completed reruns validate their original resource record, preserve matching bytes/timestamps and can repair a missing final semantic report from the committed journal. They refuse contradictory existing publications or a missing committed resource witness.

All eight paired 95% intervals are exploratory and unadjusted. All four profiles and all outcomes remain visible. Secondary factor contrasts are descriptive differences in seed margin over all 24 matched complete families, with no additional confidence intervals or promotion gates; incomplete matching produces a null mean and the complete-family count. New case/arena family IDs remain development data exposed to ancestral checkpoint selection, so these results do not establish unseen release generalization.

The limit flags are for retained fixtures, not substitutes for the full campaign:

```powershell
python tools/benchmark-quality.py run --run build/my-quality-fixture/run --output build/my-quality-fixture/result.json --case-limit 2 --no-arena
```

## Recoverable position analysis

Run the declared refined-search demo or continue its interrupted work:

```powershell
python -m congklak.quality_progress run --tool quality-analysis -- python -m congklak.quality_analysis --path data/quality-analysis-demo --profile refined_search --simulations 128
```

For another position, pass `--position` with a JSON file containing two seven-pit rows, two stores and the controller, totaling 98 seeds. Use a separate analysis directory for another position/profile/model identity. Increasing the requested backup count in the same compatible directory retains the previous completed recommendation while it recomputes the new bounded tree. An interruption publishes no partial recommendation. Once the result commits, a missing JSON export is repaired on restart. A completed rerun preserves the result and SQLite bytes/mtimes. Search estimates are fixed-player-0 final seed margins; visit fractions describe allocation, not calibrated probabilities or proof of the best move.

## Acceptance and retained audit

The author suites and independent group/full commands below are **prepublication operations**. They create fresh fixtures and may rewrite reports, durations and bindings to new retained directories. Running them again after immutable acceptance can invalidate its input hashes even when producer code is unchanged. After acceptance, use `python tools/record-quality.py --check-only` for read-only verification of the accepted evidence. Use a separately versioned verification report and fixture directory if a new experiment or renewed live recovery test is required.

Author checks before publication:

```powershell
& $qualityPython tests/quality_learning_acceptance.py
python tests/quality_search_acceptance.py
python tools/benchmark-quality.py acceptance
```

The independent coordinator is `tools/quality-audit.py`. Its `group` operation accepts `plan`, `search`, `targets`, `training`, `analysis`, `case` and `arena`; the arena group requires the complete production campaign. It retains raw live pre/postcommit CUDA training and analyzer kills, a case-prefix kill, a 383-to-384-game recovery fixture, validated backups, refusal logs and exact optimizer/RNG comparisons in `build/quality-audit`. After all groups complete:

```powershell
python tools/quality-audit.py collect
python tools/quality-audit.py full
```

Once all production evidence, independent proof, the completed demo/rerun witness and final documents exist, check and publish acceptance:

```powershell
python tools/record-quality.py --check-only
python tools/record-quality.py
```

The recorder verifies **36 distinct executed methods**: four learning, nine search, thirteen benchmark and ten independent checks. It checks current source coverage and input hashes, all 1,368 prior accepted input hashes, actual training/model proofs, all 80 cases and 384 literal game replays, paired statistics, original resources, analysis stability and raw retained diagnostics. Earlier checks are preserved evidence, not counted again. Acceptance is separate from strength; H_v1 remains the default even when engineering checks pass.

`build/quality-acceptance.json` is immutable. Repeating the recorder against identical finalized inputs preserves bytes and mtime. Dynamic progress logs, the acceptance output and its own later rerun witness are excluded from its input map to avoid recursive or constantly changing acceptance. Do not edit final documents or evidence after publication; a changed accepted artifact needs a separately recorded version.
