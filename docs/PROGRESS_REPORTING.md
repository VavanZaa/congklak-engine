# Flow progress reporting

The progress observer reports each implemented tool's committed work, latest result, evidence path and next improvement. A local browser dashboard refreshes every two seconds. Training, dataset generation, analysis and benchmarks use their SQLite catalogs as the authority; their raw logs are supplementary. The observer never loads training examples or sealed targets.

From the project directory:

```powershell
py -3.12 -m congklak.progress serve
# Open http://127.0.0.1:18765/
py -3.12 -m congklak.progress status
py -3.12 -m congklak.progress watch
py -3.12 -m congklak.progress report
```

`report` saves `build/progress/latest.json`; `status --json` prints a machine-readable snapshot. `watch` prints on changes and stops with Ctrl+C. The server binds only to `127.0.0.1`. Its only routes are the dashboard and a read-only status endpoint. Stop it with Ctrl+C and restart the same command after a machine restart.

The dashboard separates completed artifacts, running invocations, interrupted processes, stale acceptance evidence and the held strength gate. It shows counts for collection families, labels, optimizer steps, search depths and benchmark games. These are work-unit counters; they do not measure progress toward solving the full game. An observation is not a new correctness verification. Missing or invalid evidence appears as pending, stale or a read error.

The default view shows deeper teacher data and relay evaluation: 9,000 label/feature rows, eight matched model runs, 391 position cases and 288 paired games. Counts come from committed catalogs. The older audits are labeled historical when only reporting/documentation has changed. Each experiment keeps its separate independent acceptance record. “All tools” retains the earlier stages.

## Stream progress from any flow tool

Prefix a tool command with `py -3.12 -m congklak.progress run TOOL`. The wrapper prints each event in readable form, preserves bounded raw output and records start, latest event, finish and the actual exit code. It preserves the underlying tool's recovery semantics. `--raw` before the tool name streams its original output instead.

```powershell
py -3.12 -m congklak.progress run campaign build
py -3.12 -m congklak.progress run training-v2 train --run data/training-stronger2-w256-seed31415 --width 256 --seed 31415
py -3.12 -m congklak.progress run benchmark
py -3.12 -m congklak.progress run audit-neural
py -3.12 -m congklak.progress run neural analyze --model data/training-stronger2-w256-seed31415/model.json --position build/opening-position.json --run data/neural-demo --depth 8 --ms 5000
```

Re-running these completed experiment commands is idempotent. After an interruption, repeat the **same** arguments and run directory. For a new seed, model, dataset or position, use a new run directory. Wrapper logs do not override identity checks, checksum validation or committed checkpoints. A hard-killed wrapper is shown as `interrupted_or_exited` when its child process is absent; its exit outcome is unknown until the durable tool is inspected or resumed.

| Tool name | Flow tool |
| --- | --- |
| `build`, `build-neural` | Original / separate neural C++ build |
| `analyzer`, `neural` | NDJSON H_v1 / neural position analyzer |
| `oracle`, `pilot`, `campaign` | K=4 oracle / original pilot / corrected depth-six corpus |
| `training`, `training-v2` | Frozen pilot / corrected depth-six CUDA trainer |
| `model-sizes`, `benchmark` | Capacity comparison / current equal-time and paired arena experiment |
| `audit` | Original engine acceptance |
| `audit-neural`, `audit-campaign`, `audit-curriculum`, `audit-training`, `audit-progress` | Native neural / corpus / current CUDA / original training / progress authority checks |
| `training-v3`, `stage5-evaluation` | Versioned 15/43-input trainer / frozen eight-run feature experiment |
| `build-neural-fast`, `neural-fast`, `stage5-speed` | Packed AVX2/scalar build / analyzer / parity and speed experiment |
| `stage5-benchmark`, `audit-stage5`, `audit-stage5-evaluation` | Current407-case/72-game comparison / independent audit / feature training recovery |

The CUDA tools reuse the interpreter in `build/training-runtime.json`. The observer and dashboard themselves require only standard-library Python. Operational logs live in `build/progress/`, capped at four MiB per invocation and the newest 100 ordinary finished records. Active records and logs pinned by acceptance manifests remain protected. Dataset archives, checkpoints, quarantined evidence and scientific acceptance artifacts are outside this operational cleanup.

Progress JSON publication uses unique temporary files and retries transient Windows reader/indexer sharing collisions. The production Stage 5 benchmark encountered such a collision after189 committed cases; the fix passed a regression check and the unchanged benchmark resumed to completion. A permanent write failure remains visible as failure; committed SQLite work survives.

Current improvements are visible next to each tool. The deeper/relay candidate now passes its ordinary H8 error and paired seed-margin gates; exact relay simulation remains the main measured inference cost. The first depth-six campaign and its models remain quarantined after a two-row lineage-alias leak; the corrected campaign and retrained models are the current experimental lineage. H_v1 remains the default; these development experiments do not automatically change deployment.

The current action trial has its own dashboard filter and four work cards. It tracks all six objective/seed catalogs, controlled391-position comparisons and144 paired games. Use `stage5-action-training train` or `verify`, `stage5-action-speed`, `stage5-action-benchmark run`, `neural-action`, and `audit-action` through the same wrapper. Runtime modes and model identities are immutable; rerun an identical command to resume committed checkpoints, analysis depths or benchmark cases.

The deeper/relay view adds five cards: teacher data, matched training, native evaluation, strength and independent audit. Use `stage5-deeper-training generate`, `train` or `verify`, `stage5-relay-speed --model MODEL_PATH`, `stage5-deeper-benchmark run`, `neural-relay`, and `audit-deeper` through the wrapper. The teacher generator uses the corrected V2 report-linked timing contract and preserves its migration provenance. Eight trainer catalogs each show up to 8,000 committed optimizer steps. The ordinary error and strength gates remain separate from completed work counters. See [deeper/relay results and recovery](STAGE5_DEEPER_RESULTS.md).
