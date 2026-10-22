# Pair 7 allocation trace protocol

This study reconstructs the unchanged CPU PUCT rule at the exact saved pair 7 position. It keeps both model files, the shared policy, `c_puct=.25`, the declared CPU core and numerical backend fixed. Eight independent trees cover both models at 128, 256, 512 and 1,024 simulations: 3,840 requested root simulations. Each complete report must match the corresponding previously accepted search report after removing its declared timing fields.

Selection uses player-0 seed margin divided by 98. The actual controller supplies the sign in the selection score; backup retains the same player-0 sign across bonus turns and player changes. Each legal action records its prior, pre-selection visits, sampled mean or separate parent-bootstrap FPU, exploration term and total score. An unvisited action has no sampled mean. The final visit winner and the action with the best sampled mean are reported separately.

The journal retains a startup identity, a root-prediction transaction before simulations, and one atomic record for every completed simulation. Saved raw states, compressed paths, transitions, leaf predictions and physical counter deltas let the reader reconstruct every internal selection and backup without model evaluation. The report identifies the first differing root allocation and the first differing internal choice at a shared raw state. After all root actions have visits, FPU no longer acts directly at the root; unvisited descendant edges can still use it.

Only observed inference and diagnostic elapsed times are removed from semantic comparisons. Predictions, selection values, visits, sums and physical counter deltas remain. Capped cells remain incomplete. Cancellation during root evaluation or after an uncommitted child expansion preserves only the committed prefix; resume reconstructs it and retries uncommitted work. Interrupted inference counts are separate from committed counts. Uncommitted simulation transition/deposit totals are unavailable, so these counters do not claim complete lifetime work.

The run has a one-GiB journal quota, a 128-KiB per-step cap and two-GiB free-space headroom. SQLite uses FULL/WAL commits. Nonempty WAL is durable evidence; shared-memory files and absent or empty WAL are not included in byte/mtime stability claims. Per-simulation persistence and verification add diagnostic overhead, so elapsed time is not an inference speed or energy comparison.

The frozen implementation is `congklak/allocation_trace.py`; the plan and source manifest are `build/allocation-trace-plan.json` and `build/allocation-trace-implementation.json`. Production belongs to the root orchestrator after independent preflight:

```text
python -m congklak.allocation_progress run --tool allocation-trace -- python tools/trace-allocation.py run
python tools/trace-allocation.py verify
python tools/trace-allocation.py witness
```

Running the first command again resumes the committed prefix. Once complete, it validates the full journal, completion marker and resource evidence before reading or repairing exports; it performs no inference or search. `verify` requires all exports and remains read-only. The witness requires all eight genuine production cells, all 3,840 root simulations and all eight old-report matches before the separate H12 weighting study can start.

Four source-bound owner methods use only two already-consumed engineering roots in build-only directories, at 8/16 simulations. They cover arithmetic/controller units, exact prefix recovery and unchanged-engine parity, pre-root and post-expansion cancellation/caps/corruption, and completed API/CLI reopening without inference. These fixture outcomes do not replace independent physical-kill tests or current-position production checks.

No coefficients, policy, weights, training data or defaults change. This consumed single-position trace cannot establish a cause of the earlier timed loss, game-theoretic action Q, stronger play or general convergence. Every declared budget and every existing teacher horizon must remain visible in any interpretation.
