# CPU PUCT comparison: run and resume

This experiment reuses the user-confirmed Betawi rule profile, both completed compact models and the accepted Phase 6 development cohorts. It performs no training. H_v1 remains the default.

The six fixed challengers are warm and updated models with each of the legacy double-feature adapter, the new single-pass adapter, and the single-pass adapter with a 4,096-entry per-evaluator LRU. All six are tested against H_v1 and H8 relay CPU. The plan contains 80 positions at 25 and 100 ms and 576 games, using the same 24 development families with paired seats. These are reused evaluation families, not new release validation. Separate model-versus-baseline comparisons do not establish a causal training improvement.

The cache key contains the complete sixteen board cells and input controller. Model/evaluator instances own separate caches. Every decision starts its clock before clearing the cache. Frozen PUCT receives the floored remaining positive integer milliseconds. If less than one millisecond remains, the wrapper produces no completed recommendation and counts a legal-first-pit fallback; it never passes an expired deadline as unlimited `time_ms=0`.

Frozen PUCT's neural-evaluation count measures returned logical leaf requests and retains its existing accounting for stopped batches. It is not universally equal to all submitted requests or physical forwards. Additional telemetry records actual native forward executions, cache hits/misses/evictions, feature simulations and deposited seeds. Legacy inference performs two feature passes; the structured adapter performs one per miss and zero on a hit. Partial stopped work is retained. These measurements describe the implemented adapters, not isolated dense-kernel performance or electricity use.

Run from the workspace after the implementation manifest and independent preflight are frozen. Run the timing campaigns one at a time, with other heavy jobs paused. Use the resource wrapper from the first production invocation so its original import, admission, startup and remaining-work costs are retained:

```powershell
Set-Location -LiteralPath 'F:\My Engine\Congklak Stockfish'
python tools/benchmark-puct-cpu.py admit
python -m congklak.puct_cpu_progress run --tool cpu-puct-costs -- python tools/puct-cpu-resource-run.py speed
python -m congklak.puct_cpu_progress run --tool cpu-puct-strength -- python tools/puct-cpu-resource-run.py benchmark
```

The CPU-only admission checks all 1,192 prior accepted input hashes, 42 prior checks, both model exports, all 30 checkpoint catalog/file witnesses and the new sources/DLL/build record. It does not import Torch. The benchmark pins its own declared performance core, records whole-wrapper latency and overshoot, and separates cold startup and batch-eight warmup from decision time. The progress observer is available at `http://127.0.0.1:18766/` while `python -m congklak.puct_cpu_progress serve --port 18766` is running.

The frozen progress CLI records the actual child `exit_code` in `build/puct-cpu-progress/run-*.json`, but its shell exit status does not propagate that code. Read the run record and producer artifacts, then use the acceptance recorder as the pipeline check; a zero progress-command shell exit alone does not establish success.

Repeat the identical wrapped command to resume its journal: `data/puct-cpu-speed-v1` or `data/puct-cpu-benchmark-v1`. Speed measurements commit one scheduled unit at a time. Cases commit atomically; games commit only after the full terminal/capped replay is available. An interrupted unit, case or game is recomputed. The journal uses FULL/WAL durability and validates scheduled prefixes, legal moves, ownership, reference regret, policies, counters and terminal margins before extending them. Incomplete games receive no invented outcome. Resource reports describe the final invocation and record its prior committed counts; they do not reconstruct the resource costs of an interrupted process.

If a process is killed after publishing its completed report but before publishing the outer resource JSON, the wrapper refuses to reconstruct that missing resource observation. The semantic journal is retained, but this experiment cannot claim a complete original outer-resource record for that invocation. Preserve the files and diagnostic; use a separately declared version for a fresh resource campaign.

The speed campaign contains 1,410 units: six startup observations, 36 numerical parity batches, 504 cold/hot microbenchmarks and 864 fixed-64-backup trees. Cold timings use loaded, warmed models and clear the cache inside the call clock. Hot timings prefill outside the clock and measure a retrieval ceiling. The predeclared cost gate uses matched whole-tree ratios for both models and requires exact fixed-tree semantics and half the legacy feature work. Cached comparisons are descriptive. The twelve paired arena intervals are exploratory and unadjusted for multiple comparisons.

To produce the resumable opening analysis demonstration with the updated compact model:

```powershell
python -m congklak.puct_cpu_analysis --path data/puct-cpu-analysis-demo --model data/training-phase6-update-v1/model.json --cache-capacity 4096 --simulations 128
```

Repeat the same command after interruption. The latest completed recommendation remains durable; an unfinished tree is recomputed. A missing JSON export is repaired from committed metadata. Completed reruns retain bytes and timestamps. Search visit fractions are allocation statistics, not calibrated win probabilities. This analyzer is an experimental candidate; H_v1 remains the default.

For bounded audit fixtures, use a separate directory with `--case-limit 1 --no-arena` or `--case-limit 0 --arena-limit 1`. `PUCT_CPU_BLOCK_AFTER_CASE=1` and `PUCT_CPU_BLOCK_AFTER_GAME=1` expose committed-prefix barriers for an external hard-kill test. Never use fixture directories for production evidence.

A direct completed producer rerun checks the canonical report against the journal and preserves completed bytes and timestamps. A completed outer resource-wrapper rerun checks implementation admission and its original resource/report binding, then returns without rerunning producer journal validation. The acceptance recorder independently rederives the reports from their journals. A new source, model or plan requires a new version/directory. Keep all outcomes, including failed strength gates and diagnostics. No profile is selected from these outcomes before every registered candidate has been reported.

After the independent arena recovery test and full audit, and after the results document is finalized, validate and record the aggregate evidence:

```powershell
python tools/record-puct-cpu.py --check-only
python tools/record-puct-cpu.py
```

The recorder requires 34 unique executed checks (nine inference, twelve benchmark and thirteen independent audit methods), all production measurements, the completed analyzer demonstration and the final new documents. Engineering acceptance records recovery and evidence integrity separately from cost and strength gates. Repeat recording with the same inputs preserves the acceptance file's bytes and timestamp.
