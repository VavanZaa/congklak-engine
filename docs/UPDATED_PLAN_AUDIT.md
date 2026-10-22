# Full updated Congklak plan audit

Review date: 14 October 2026, Asia/Bangkok. Three independent agents reviewed the updated architecture, database generation, training, weighting, retention, pruning and recovery plans. The primary editor corrected the original documents, and each agent re-read the corrections before issuing its final verdict.

**Accept the revised plan and implement [FIRST_IMPLEMENTATION.md](FIRST_IMPLEMENTATION.md) steps 1 through 5 first.** All 22 numbered findings and the subsequent actor-policy and staging-writer clarifications are addressed in the planning contracts. No architecture, data or recovery design blocker remains for this initial slice. Correctness, playing strength, recovery and performance are still unexecuted implementation gates.

## Reviewed scope and independent reports

- [Architecture audit](audit/ARCHITECTURE_AUDIT_V2.md) reviewed rules, simulation, score/controller semantics, horizon search, tablebase recurrence and first-release dependencies. Findings V2-A1 through V2-A6 are resolved.
- [Data and training audit](audit/DATA_AUDIT_V2.md) reviewed labels, objectives, sampling, leakage, weighting and quality-based pruning. Findings D1 through D8 are addressed.
- [Recovery and pruning audit](audit/RECOVERY_AUDIT_V2.md) reviewed resource admission, checkpoints, reference closure, bulk publication and deletion. Findings R1 through R8 and the live-writer refinement are resolved.

The reviewed governing documents are [architecture](CONGKLAK_ENGINE_PLAN.md), [data and training](DATA_AND_TRAINING_PLAN.md), [first implementation](FIRST_IMPLEMENTATION.md), [compute and recovery](research/compute-and-recovery.md), [rules](research/congklak-rules.md), and the [engine research options](research/engine-comparison.md). The [initial audit](PLAN_AUDIT.md) is retained as a historical record, with its older implementation gates superseded. A [reviewed-file snapshot](audit/reviewed-files-v2.json) records the final governing artifact hashes; these file hashes are not the future formal rule-contract digest.

## Corrections made

| Area | Defect found | Final contract |
| --- | --- | --- |
| Finite search and oracle facts | Full-game bounds or tablebase values could silently change a strict horizon result | Declare terminal/oracle/leaf recurrence and its identity; the first baseline has no oracle or outcome-bound pruning |
| Exact database values | Store omission lacked immediate award accounting | Fixed-player future value backs up deltaD plus child future value; terminal future value is zero and full margin adds existing D |
| Forced passes | Opponent pit masks could appear as original-player moves | Normalize before lookup; retain input actor/pass trace, label next actor and score perspective; terminal and unknown masks stay distinct |
| Training independence | Promotion, pruning or oracle lookups could expose the final holdout | Development selects models and policies; final test is sealed. Guard direct, symmetry and quotient aliases, cached facts and model ancestry |
| Loss and tied moves | Normalized residual loss changed seed-error weighting and ordinary tied-target cross-entropy penalized tied choices | Default value loss uses reconstructed seed units; exact nonterminal policy loss uses optimal-set probability mass |
| Label and sampling weights | Newer/deeper labels, duplicate frequency and overlapping priority routes could create unjustified weights | Explicit compatible label selector, separate reliability/priority, uniform first; later marginal sampling correction and coefficients applied once |
| Pruning quality | Error-based eviction could erase useful rare or older data without a comparison | First pilot validates and deduplicates; later pruning uses matched-cost development/rehearsal ablations with declared tolerances |
| Storage and proof dependencies | Logical quota and a simple dependency check were insufficient for rewriting and deletion | Physical quota plus replacement reserve; serialized roots/pins/readers/writers, transitive evidence handling, tombstones and idempotent recovery |
| Training checkpoints | Prefetch queues and unfinished optimizer/device work could be omitted | Start without prefetch; snapshot completed steps and all sampling state; later queues are consumed, rewound or explicitly captured |
| First implementation scope | Larger tables, learning, other variants or undefined UI choices could delay the first useful product | One complete experimental profile, independent kernels, deterministic CLI search and inline SQLite completed-report persistence |

A source fact previously proved does not become false when a file is removed. The retention contract instead preserves independently verifiable evidence or retires the operational certificate/coverage claim that depends on missing evidence. Likewise, an experimental rules contract is implementable without asserting that it is the user's confirmed physical variant.

## First release to implement

1. Freeze the complete `id-betawi-relay7-v0` experimental rule JSON, coordinate map, digest and validated stable-position encoding.
2. Implement independent reference and production sowing interpreters with exact traces and normalized control.
3. Add single-worker full-width minimax/alpha-beta with iterative choice-depth search and H_v1 equal to store difference. Score each root candidate with a full window under the same horizon.
4. Expose a local NDJSON command-line protocol with validation, transitions, analysis, cancellation, saved reports and resume. Show the actual actor and score perspective.
5. Persist small input/settings/completed-iteration reports transactionally in SQLite, with startup storage admission and fixed-horizon restart checks.

The first scope starts without a TT, PVS, learned network, GUI, complete bulk collector or exact tablebase dependency. This gives the later optimizations an independent reference to compare against.

The proposed initial interactive settings are one second, choice depth 16 and one million decision nodes, with a 512 MiB managed-memory budget. They are configurable starting settings, not benchmark results. Job data defaults to the workspace data directory with an explicit override. The [first-scope contract](FIRST_IMPLEMENTATION.md) specifies validation, resource admission, cancellation and test coverage in detail.

## First release acceptance

Acceptance requires the declared golden traces, exhaustive reduced-profile transition and horizon comparisons, and 1,024 seeded legal histories under the product profile. Recommended pits and trace facts must belong to the displayed actor; interrupted sowing cannot publish a partial successor. Restart at the same horizon/evaluator must reproduce the semantic result, while equal wall-time runs may finish different depths.

Test process termination before and after SQLite commits, cancellation during sowing, full-disk admission and malformed/incompatible saved records. Keep the last complete report, or state that no iteration completed. A completed horizon result is not automatically a full-game proof.

These are acceptance obligations. No engine implementation, database generation, model training, simulator test, fault injection or performance/energy benchmark was executed during this plan audit.

## Follow-on order

After the first release passes, introduce TT and PVS independently. Run a 10,000-position uniform data pilot with validation, deduplication, fixed split ownership and explicit teacher labels; a K=4 exact oracle is a small separately verified aid. Then compare a tiny MLP on a 100,000-position corpus with development-only promotion and a sealed accepted-release evaluation.

Larger exact tables, NNUE, prioritized weighting, quality-based pruning and GPU PUCT are experiments after that baseline. Select their numerical settings from measured accuracy, overhead, latency and energy. Corpus volume and network size alone are not acceptance criteria.

The remaining local-convention question affects whether the selected profile matches the user's physical play. It does not block implementation of the complete experimental contract. Other variants receive separate state/contract namespaces before generating results or training data.
