# Saved best-action regret and capture forensics

`congklak/regret_forensics.py` examines existing H_v1 finite-horizon reports with literal Python sowing and exact arithmetic. It performs no Native calls or construction, model evaluation, inference, teacher search, new labeling, games, training, sampling changes or pruning. The scope is [REGRET_SCOPE.md](REGRET_SCOPE.md); the producer separately protects the prior baseline, reads the closed saved source, selects the exact panel and journals the five detailed cases.

## Panel and source contract

Exactly five detailed cases are prescribed: `midgame-history-pilot/20261023/178` in the low-bonus stratum, then capture families ending91,382,92,390. These are the root178 best-action reversal and **all four** capture roots from the completed Q-stability panel. They are consumed retrospective diagnostics; no new root or weight rule is selected. The producer retains the full original 16-root numeric context and fixed-denominator contributions so five-case detail cannot be mistaken for a newly normalized assessment population.

The pure helper API is:

```text
build_case(row, items_by_stage, attempts_by_stage) -> case
validate_case(row, items_by_stage, attempts_by_stage, case) -> case
pair_trajectory(case, pit_a, pit_b) -> signed_pair_history
```

Rows are unchanged source wrappers with full raw `state`, raw17-byte `state_key`, `actor`, `family`, `stratum`, `legal_mask`, nested original `root` and `source_root_sha256`. The wrapper hash and original-root hash are distinct and retained. No canonical player swap or store-offset Q reuse is performed. All roots must be normalized nonterminal 25–49-seed bonus/capture positions with at least two legal actions. Metadata family/panel ownership remains the producer's responsibility.

The maps are keyed by exactly `h8`, `h10`, `h12`. Every item has the existing `{root_sha256, label, report}` shape. Every attempt must be complete at its requested depth, with the full consecutive history, exact source/study/label hashes and endpoint report equal to its last history entry. Partial or auxiliary reports cannot become requested endpoints. Saved labels and reports must give complete unique integer action Q, full sorted ties and actor-correct best target at the identical full raw state.

## Accepted pure interpreter reuse

The new helper reuses the accepted `capture_forensics.replay_line`, `compare_lines`, `rank_diagnostics`, saved-report signature and immediate-transition validators **without editing them**. Their underlying interpreter is [tests/reference.py](../tests/reference.py). Source hashes for both are pinned before fixtures and retained in each new case. Nested literal lines keep the accepted `capture-saved-literal-forensics-v1-line` schema, while their parent case uses `regret-saved-bonus-capture-literal-forensics-v1`. This schema reuse describes the interpreter; it does not classify a bonus root as capture.

The new generic wrapper derives the root mechanism from all immediate legal actions, using the original pass→capture→bonus precedence. A capture-class root has an available capturing action; it does not imply its best action captures. A bonus root has an available own-store bonus and no higher-priority pass/capture classification. Every literal transition must match the saved immediate transition's full state, actual actor, reason, store gains, capture transfer, relay/deposit counts, passes and terminal status.

Saved PV moves retain actual controllers, bonus choices, forced passes and terminal next-controller nulls. Fixed-player0 values never receive an automatic alternating-ply sign. Every saved line's reported H_v1 score must equal its literal leaf-store margin. Cumulative captures, sowing store gains and total store-margin change are reconciled, with terminal status and active seeds retained. A zero arithmetic residual is not a bound on the unplayed remainder of a nonterminal position.

## Complete histories, pair crossings and line comparisons

Every requested job's full saved history is used: depths1–8,1–10,1–12. For every report, replay the root PV and every legal-action PV, preserving actual-depth/report/source links. Full Q and best-set signatures must agree wherever jobs overlap at the same depth; PV differences at an equal depth may remain explicit. Endpoint Q/best/target must agree with the label. There are 150 report references across the five cases, 75 endpoint action-Q values, and900 root/action PV references before deduplication. These are coverage counts, not independent samples.

Deduplicate only identical raw-root/PV/actual-depth/saved-value lines. Retain every source reference on each deduplicated line; repeated jobs, root/action PV duplicates and repeated depth prefixes are not additional evidence. Counts distinguish reference choices from unique-line literal choices.

For every legal action, retain Q8/Q10/Q12 in player0 and actor orientation, signed/absolute changes, full best membership and actor-relative gaps. For each requested horizon pair, retain the entire earlier best set, later regret of each tied action, minimum/mean/maximum regret and the lowest-pit earlier-best choice separately. With actor1, regret at the later horizon is `Q[action]-min_legal(Q)`.

For every unordered legal-action pair a<b, the new `action_pair_trajectories` records all three requested endpoints and every depth1–12 in the same H12 job. Pair advantage is:

```text
actor_sign = +1 for actor0; -1 for actor1
advantage_a_over_b_actor = actor_sign * (Q[a]-Q[b])
```

Positive means a is preferred, negative means b is preferred, and zero preserves a tie. Adjacent records distinguish a strict sign reversal from entering/leaving a tie and from a numeric gap change that preserves order. The five cases contain55 unordered action pairs and660 pair-depth records, all derived from the same saved reports. This exposes the actor1 human-pit6/pit7 crossing at root178 without choosing a favorable intermediate depth or claiming independent depth samples.

For each action and the root-best line, compare every requested horizon pair. A common prefix requires identical full raw state, actor and action. Stop at the first different choice; do not treat later unmatched states as shared positions. Keep first-choice mechanics, post-prefix capture/store/bonus/pass totals, and both leaves. Captures, relays and changing leaves are line observations, not a controlled causal explanation of the Q or regret change.

## Validation and formal owner coverage

`validate_case` deterministically rebuilds the complete case and compares canonical JSON. Rehashed source aliases, wrong requested depths, incomplete action/history support, altered signs, ties, pair crossings, immediate mechanics, source hashes or leaf facts refuse. Completed producer reopening may repeat this pure arithmetic/literal validation but must add no case commits or engine/model/training work.

Before formal fixtures, pin the helper, four-method owner source, this document, scope/common/baseline, accepted replay helper/interpreter and saved source files in `build/regret-forensics-source-freeze.json`. The source-bound result is `build/regret-forensics-author.json`. Exactly four owner methods cover:

1. Complete history/action/source/dedup accounting and accepted literal-helper reuse, using consumed root178 and capture91.
2. Actor1 best-action reversal, every alternative and explicit pair-depth signs/ties.
3. Generic mechanism validation, raw-state prefix/tail facts, literal terminal/bonus/pass behavior and leaf-store reconciliation.
4. Rehashed-fact/source/alias refusals and a byte/size/mtime-stable rebuild.

These pure tests do not claim new process kills. Real pre/post case-transaction interruption probes belong to the producer/independent audit and are counted separately. Preserve any failed frozen-source report and exact executed source before repair. This stage does not modify active weighting, model policy or search settings; all H12 interpretation remains finite-reference and noncausal. Outcome findings are published later in the separate `REGRET_FINDINGS.md`, after closed production evidence exists.
