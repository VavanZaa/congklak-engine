# Fixed 400-step value pilot results

**The pilot is not promising under its predeclared gates. The final candidate is quarantined, and H_v1 remains the default.** Training and all requested evaluations completed; the value-fit gate failed and post-game ownership bridges invalidate strength interpretation. No model promotion, new teacher labels, replacement checkpoint, expanded database or additional training followed the result.

## Training and fit

CPU FP32 training completed exactly 400 updates and 51,200 uniform replacement draws. Only the 2,065-parameter value branch learned; all eight original tensors and 12,872 parameters, including policy, stayed fixed. All 20 scheduled checkpoints and step zero were retained. A real stop at step 20 followed by 380 resumed updates preserved the entire committed prefix. Total charged training worker time was 43.25 seconds; CUDA allocation was zero.

| Diagnostic on 128 reserved fit roots | Control | Final candidate | Gate |
|---|---:|---:|---|
| Normalized Huber | 0.01007445 | 0.01025379 | Failed |
| Overall seed MAE | 6.13176 | 6.12314 | Passed |
| Capture seed MAE | 5.61788 | 5.95729 | Failed |

Fit-check data never entered these gradients, but its ancestors trained on the historical corpus. These are training-source diagnostics, not unseen validation. H8 targets remain approximate finite-horizon estimates.

## Position and inference checks

The final learned model passed 733-position native scalar/AVX2/batch and FP32 conformance, with unchanged policy. All 512 timed measurements and 138 fixed-work searches completed. The latter completed all 17,664 requested backups. The candidate chose an optimal move on all 32 reused exact positions at both 25 and 100 ms, with zero fallbacks. Control and candidate matched action, visits, Q values and physical search work exactly on all 34 low-active roots. Sixteen exact positions have only one legal action.

| H8 best-action regret | Control | Candidate | Gate |
|---|---:|---:|---|
| 25 ms | 0.75000 | 0.62500 | Passed |
| 100 ms | 0.71875 | 0.71875 | Passed |
| 128 backups | 0.90625 | 0.65625 | Passed |

These action-quality gains do not override the failed fit gate. Nominal decision budgets include reset, inference and validation; actual overshoots were retained. The largest candidate 100 ms case overshoot was 28.4873 ms.

## Arena and ownership hold

All 192 scheduled games ended terminally, with 6,663 committed decisions, zero fallbacks and zero choice caps. Both models faced H_v1 and H8 relay CPU in both seats on all 24 starting components.

**The following raw averages are descriptive only.** Two trajectory observations in games 152 and 156 reached a frozen training alias. Twelve cross-family quotient bridges and three historical-component alias bridges were found, including a bridge connecting all 24 families. The planned independent 24-cluster analysis is invalid. All six confidence intervals are withheld; games were not filtered, families were not replaced and clusters were not merged after outcomes.

| Opponent | Control raw mean margin | Candidate raw mean margin | Raw candidate minus control |
|---|---:|---:|---:|
| H_v1 | 1.54167 | 1.33333 | -0.20833 |
| H8_relay_CPU | 1.25000 | 1.16667 | -0.08333 |

The nonnegative paired-mean advancement requirement is unmet; it cannot supply an interpretable cluster-based strength claim under this ownership hold. No claim that the candidate beats either baseline follows from these raw margins. Ordinary starts are synthetic custom boards with unproven standard-opening reachability.

## Recovery, resource accounting and audit

Six independent real process kills passed before production evaluation: two case-journal boundaries and four arena decision/game boundaries. Fixed-search results matched fresh runs exactly and committed bytes, sizes and modification times survived. Completed readers rejected missing required artifacts before writers, engines, Torch or searches.

The production arena also recovered from two unexpected infrastructure stops. The first coincided with a failed derived progress-file replacement; its child exception was not captured. An actual Windows experiment showed that legacy replacement failed even with delete-sharing on an open target. Observer v3 now avoids mutable arena files entirely, exposing provisional physical file counts until immutable completion evidence exists. The second failure was captured: a watchdog quota scan encountered a temporary file during its atomic rename. A separately versioned, attested watchdog ignores only vanished pending files and still holds on unrelated I/O errors. All original worker, model, policy, search settings and raw committed prefixes remained unchanged.

Arena worker invocations used 302.171 seconds actual active time and 271.9 MB peak private commit. The durable ledger conservatively charged 785.5 seconds, including both full 300-second failed reservations. No ledger reset, pending lease or unknown overrun remains. The five-minute lease and 90-minute cumulative cap stayed fixed.

Independent evidence is in [VALUE_RUN_AUDIT.md](VALUE_RUN_AUDIT.md); the full gate record is [decision.json](../build/value-run-v1/decision.json). The final acceptance record binds all preserved prior inputs and every required new artifact.

Next, diagnose the existing ordinary/capture errors and the full trajectory ownership bridges before specifying another treatment or larger training dataset. The unchanged improved policy/control model remains the learned reference.
