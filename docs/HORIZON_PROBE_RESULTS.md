# Six horizon disagreements: H12/H14 investigation

All six previously identified H8/H10 disagreements were investigated with the fixed model and bounded searches. **H12 completed on all six roots; H14 completed on one and hit its node cap on five.** Each rejected H14 attempt retains a valid completed H13 history. Only D2 is eligible for a complete requested-H12/H14 comparison, and its best set agrees. The other five requested comparisons remain incomplete. This does not resolve or solve the six positions.

No model, active weights, data ingestion, training or promotion changed. H_v1 remains the default. The retained model file is `data/quality-train-v1/model.json`, SHA-256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`.

## Exact scope and completion

Selection is exactly the sorted six roots whose best sets were disjoint in the sealed H8/H10 study. There are no new positions or replacement cases. All original wrapped roots, H8/H10 items and ownership remain unchanged: five original training candidates and one validation root; four originated in the fresh natural pool and two were earlier follow-ups. Every new probe result remains investigation-only, including roots whose historical cohort is training_candidate.

Each root receives an H12 attempt capped at 10 million nodes and an H14 attempt capped at 25 million nodes, irrespective of H12 completion. Searches use time limit zero, TT16, PVS, a 512 MiB search limit and complete iteration history. The caps were not expanded after observing a rejection.

| Stage | Requested roots | Committed attempts | Complete requested labels | Budget rejects | Last completed depth on rejected attempts |
|---|---:|---:|---:|---:|---:|
| H12 | 6 | 6 | 6 | 0 | Not applicable |
| H14 | 6 | 6 | 1 | 5 | H13 on all five |

There are seven new complete requested labels, 12 unchanged H8/H10 reference items and 151 retained completed-iteration report entries. The 151 entries include the final requested reports and overlapping depths from both jobs; they are not 151 independent targets or positions. Every report retains its actual completed choice depth and source job. H13 from a capped H14 search is auxiliary H13 evidence, not a complete H14 label.

All 2,197 previously sealed inputs remain unchanged. `build/horizon-probe-baseline.json` additionally binds the natural-study acceptance and its immutable rerun witness, preserving 2,199 prior artifacts. No earlier progress source, HTML, model, database, report or accepted documentation was edited.

## What the six cases show

The following table uses **human pit numbers 1-7**; retained machine moves use 0-6. Braces preserve tied best sets. All six canonical roots have Player 1 to choose, so lower Player-0 seed margins are preferable at these roots. Regrets are actor-signed at the same depth; choosing the largest Player-0 score here would be incorrect.

| Case | Original origin/cohort | H8 best | H10 best | Complete H12 best | Auxiliary H13 best | Complete H14 best |
|---|---|---|---|---|---|---|
| D1 | Follow-up / training candidate | 4 | 5 | 5 | 5 | Unavailable: capped |
| D2 | Follow-up / training candidate | 3 | 1 | 1 | 1 | 1 |
| D3 | Fresh / training candidate | 1 | 4 | 1 | 1 | Unavailable: capped |
| D4 | Fresh / training candidate | 7 | {2,4} | {2,6} | {2,6} | Unavailable: capped |
| D5 | Fresh / validation | 6 | 4 | 2 | 2 | Unavailable: capped |
| D6 | Fresh / training candidate | 1 | 2 | {1,2} | {2,6} | Unavailable: capped |

This is a selected disagreement sample, not an estimate of general teacher reliability. H10 is not uniformly vindicated by looking deeper: D3 returns to the H8 move, D5 prefers a third move, and D4/D6 change tie structure. D2 is the only complete requested pair. Agreement at H12/H14 on that one root does not establish convergence or game-theoretic correctness.

The forensic report replays optimal finite-depth continuations for the union of old and current best root actions. These are concrete facts from particular reported lines, not a causal proof that a capture alone caused a minimax change. Deeper lines often select different replies before the old cutoff, so they cannot be treated as simple extensions of the old principal variations.

- **D1:** pit 5 remains best at actual depths H9-H13. In the H13 line beginning with the old H8 pit 4, Player 0 gains four bonus choices at choices 9-12, then captures 16 seeds at choice 13. Its stores gain 21 seeds after choice 8, and the H13 Player-0 margin is 9 versus 3 for best pit 5: six seeds of actor-signed regret. This line already differs from the H10 continuation at choice 2. It is auxiliary H13 evidence; H14 is unavailable.
- **D2:** pit 1 stays best at H9-H14 and agrees between complete H12 and H14. At H14, old H8 pit 3 has four seeds of regret. The reported best-pit-1 continuation has three capture events transferring 10 seeds after choice 8 and ends with 19 active seeds. Its first changed choice versus the H10 continuation is choice 8. The leaf is nonterminal, so the stable best set does not solve the position.
- **D3:** H10 pit 4 is replaced by old H8 pit 1 at H11-H13. The H13 pit-4 continuation gives Player 0 four bonus choices at choices 9-12 and a seven-seed capture at choice 13, with 14 seeds added to Player 0's store after choice 8. Pit 4 then has four seeds of H13 regret. Its line changes at choice 9 versus H10, rather than merely extending all ten old choices.
- **D4:** old H10 pit 2 remains best, but its tied pit 4 drops out and pit 6 joins at H12/H13. The H13 continuation through dropped pit 4 includes a 24-seed capture at choice 13. That continuation differs from the H10 line at choice 4. The result requires retaining whole best sets: the H10 lexicographic choice can remain best while another H10 tie no longer does.
- **D5:** deeper complete H12 and auxiliary H13 prefer pit 2 over both old choices. On the H13 line through H10 pit 4, Player 0 gains 22 store seeds after choice 8 and captures five at choice 13; the H8 pit-6 line gains 25 and captures 15. Their H13 regrets are four and seven seeds. These lines change from the H10 continuations at choices 6 and 4 respectively. This remains the original held-out development-validation root and is not ingested for training.
- **D6:** available best sets are H8 {1}, H9 {6}, H10 {2}, H11 {1}, H12 {1,2} and H13 {2,6}. The H12 tie admits both old choices, and the auxiliary H13 tie changes again. H14 is capped. This is depth-sensitive evidence, not a resolved choice between the original teachers.

All 87 inspected forensic lines end at nonterminal cutoffs. The report retains full leaf states, remaining active seeds, store changes, captures, bonus choices, forced passes, same-line cutoff snapshots and first changed choices. Store margins at a shorter point along one deeper line are explicitly distinguished from independently optimized shorter-depth targets. Bonus choices also mean odd/even choice depth does not imply alternating players.

The findings show sensitivity to finite choice-depth cutoffs around long same-player store runs and later captures. The replay audit found no actor-sign, bonus/pass or capture-fact mismatch in the checked evidence. It does not prove a universal engine property or isolate a model defect. Before training on these cases, target handling for depth-sensitive ties and incomplete requested horizons must be decided explicitly. This investigation does not authorize automatically increasing budgets, treating H13 as H14 or admitting diagnostic probe targets into training.

## Work and recovery

The uninterrupted production ledger records **173,556,679 new nodes**, within the declared production-attempt sum cap of 210 million: 6 x (10M + 25M). H12 consumed 27,001,092 nodes and H14 consumed 146,555,587. Five H14 attempts each reached their full 25M cap. No larger budget or replacement root was used.

Native per-attempt timers sum to 7,243 ms for H12 and 37,694 ms for H14. The producer's initial-run timer is 47.078 seconds. These values exclude surrounding admission, subsequent verification/audit and other process costs. The 210M cap bounds this declared production attempt set; fixture searches and the independent fresh bounded verification are additional work. An interrupted uncommitted root must be recomputed, and its lost work is additional to the committed ledger. This is not a lifetime compute or energy cap, and no power saving is claimed.

Four workers were actually killed at the H12/H14 precommit and postcommit barriers. Each started from an explicitly recorded prefix seeded from the independently computed fresh fixture. The seed source, removed completion/export metadata, retained target, marker JSON and killed-prefix SQLite backup remain preserved. The resumed workers retained committed prefixes exactly and matched fresh semantics/counters. This tests durable attempt-boundary recovery; it does not checkpoint an in-memory search frontier.

The real H14-capped fixture preserves completed auxiliary H13 without creating H14. A separate synthetic eligibility test confirms that an auxiliary H12 from a complete H14 job does not fabricate a missing requested-H12 label. Corrupt histories, references, identities, TT/PVS settings, prefix order and invalid completion markers are refused. Missing exports can be repaired from valid durable state without teacher searches.

## Acceptance and use

Engineering acceptance contains **12 actual unique new methods: four author methods and eight independent methods**, derived from current source and executed report coverage. The independent audit checks all 151 completed history reports, 956 literal principal-variation paths and 6,506 choices, including same-depth Q/best-set agreement across old references and new jobs. It separately verifies 37 forensic depth rows, 87 lines, 930 choices, 153 capture events, 647 bonus choices and six literal model forwards. These overlapping replay counts are evidence within the methods, not additional independent data or test counts.

The author suite's initial fixture-name failure source is retained as diagnostic evidence; only the corrected current four successful methods are counted. The independent audit required no harness corrections. Its fresh H12 verification is separately capped at 10M nodes and is outside the production node total. See `docs/HORIZON_PROBE_INDEPENDENT_AUDIT.md` and `build/horizon-probe-audit.json`.

Completed-production and forensic reruns preserve the closed SQLite, manifests, reports and modification times while prohibiting `Native.analyze`; they perform zero new teacher searches. All 2,199 baseline pins remain current. The aggregate binds new immutable sources, raw recovery evidence, reports and final documentation, excluding mutable observer state, browser output and its own acceptance-rerun witness.

Run or resume the exact six-root investigation with:

```powershell
python -m congklak.horizon_probe_progress run --tool horizon-probe -- python tools/horizon-probe.py all
```

An interrupted run resumes committed attempts under the same identity; an unfinished root is recomputed within its original cap. To verify completed artifacts and descriptive analysis:

```powershell
python tools/horizon-probe.py verify
python tools/analyze-horizon-probe.py
python tools/record-horizon-probe.py --check-only
```

The recorder is read-only and prohibits teacher search. Author/independent fixture suites are prepublication operations that create fresh evidence and rewrite reports. After sealing, use `--check-only`; another fixture campaign requires a new evidence version. Source or identity mismatches are refused.

The new observer runs at `http://127.0.0.1:18766/` with six cards for exact selection, H12, H14, comparison eligibility, case outcomes/auxiliary depths and audit. It keeps five H14 rejects and the 1/6 requested-pair denominator visible. Compilation and direct backend snapshots were checked. Browser access remained blocked by policy, so no screenshot or visual browser-verification claim is included. No training, ingestion, active weights or promotion follows from engineering acceptance.
