# Completion of the five capped H14 roots

All five previously capped requested-H14 searches completed under the authorized initial **250,000,000-node cap per root**. There were zero new budget rejections and no production fallback searches. Together with the unchanged prior D2 H14 label, all six roots now have complete requested H12/H14 pairs. This finishes the bounded search task; the model, training data, sampling weights and default engine remain unchanged.

The new production ledger contains **196,968,386 nodes and 70 completed iterative reports**. The initial producer timer was 53.734 seconds; this timer is not an all-process resource or energy measurement. D2's earlier complete search remains historical work. The prior probe, recovery fixtures, killed/retried work and independent fresh verification are outside the new production node ledger.

Human pit numbers below are **1-7**; retained machine moves are 0-6. Braces preserve every tied best move. All six canonical roots have controller Player 1, so the teacher minimizes its fixed-Player-0 seed-margin score. The table compares completed finite choice horizons, not exact game solutions.

| Case | H8 best | H10 best | H12 best | Prior auxiliary H13 best | Complete H14 best | New production nodes |
| --- | --- | --- | --- | --- | --- | ---: |
| D1 | 4 | 5 | 5 | 5 | 5 | 26,598,107 |
| D2 | 3 | 1 | 1 | 1 | 1, prior label reused | 0 |
| D3 | 1 | 4 | 1 | 1 | 1 | 41,560,363 |
| D4 | 7 | {2,4} | {2,6} | {2,6} | 2 | 49,319,259 |
| D5 | 6 | 4 | 2 | 2 | 2 | 41,532,605 |
| D6 | 1 | 2 | {1,2} | {2,6} | 1 | 37,958,052 |

Four H12/H14 best sets are exactly equal. Two narrow: D4 drops pit 6, which has an actor-signed H14 regret of 3 seeds; D6 drops pit 2, with regret 5. The lexicographically first H12 choice remains H14-optimal in all six cases, but retaining the complete tied policy target matters. H14 assigns regret 8 to D3's old H10 choice of pit 4, regret 6 to D5's old H10 pit 4, and regret 5 to D6's old H10 pit 2. These are disagreements between approximate teachers, not proven game-theoretic losses.

D6 still changes from H13 best pits {2,6} to H14 best pit 1. Agreement at H12 and H14 therefore does not establish convergence across the intervening horizon. All 14 inspected H14 principal-variation lines end at nonterminal positions. The report records their store gains, captures, bonus choices, forced passes, leaf states and common prefixes, but it does not establish that one tactical event caused a horizon change. These six roots were deliberately selected for earlier H8/H10 disagreement; they cannot estimate general teacher reliability or playing strength.

The model remains `data/quality-train-v1/model.json`, SHA-256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. All six original wrapped roots, ownership fields, H8/H10/H12 reference items and prior H12/H14 histories are unchanged. The five new targets retain four training-candidate and one validation ownership labels, with one original follow-up and four fresh-study roles. These are historical ownership labels; all new outputs are investigation-only and have not been ingested into training.

The frozen teacher uses time limit zero, TT16 MiB, PVS, 512 MiB and complete iterative histories. The unused 1B fallback was authorized only after a genuine 250M rejection. The 6.25B worst-case production-attempt allocation is not a lifetime compute or energy cap: interrupted uncommitted work may require recomputation. No further expansion occurred.

Four author and eight independent methods passed: **12 unique new checks**, zero failures, errors or skips. The auditor replayed all 70 new reports (448 PV paths, 3,360 choices), revalidated all 151 historical reports, and independently rederived the six-root comparison and PV facts. An additional bounded H14 verification of D1 used 26,598,107 nodes and matched every completed iterative action-Q/best signature; its cost is separate from production. Detailed coverage and retained evidence are in the independent audit.

Both tiers were recovery-tested despite zero production fallback. A separately versioned real-target fixture used **25M then 250M**, genuinely rejected after H13, then completed H14; its total fresh cost was 51,598,107 nodes. Four actual worker kills covered precommit and postcommit for both tiers, with zero- and one-attempt SQLite prefixes. Disclosed fixture clones supplied recovery seeds; journal hashes, SQL deletions, markers, backups and the retained initial-rejection digest are preserved. Copied committed rows are not new search work. Production rejects the fixture identity and budgets.

The exclusive WAL/FULL journal commits each attempt and label atomically. Resume retains committed rows and restarts an unfinished root; no in-memory search frontier is restored. Corruption and invalid fallback eligibility are refused; valid missing publications can be repaired without search. The completed producer/comparison rerun prohibited `Native.analyze`, made zero calls and preserved bytes, sizes and mtimes of all five durable artifacts. All **2,314 prior hashes** remain current.

To resume or verify this exact run:

```powershell
python -m congklak.horizon_finish_progress run --tool horizon-finish -- python tools/horizon-finish.py all
python tools/horizon-finish.py verify
python tools/analyze-horizon-finish.py
python tools/record-horizon-finish.py --check-only
```

Completed commands validate existing output without searches or rewriting. Author/independent suites are prepublication operations that create reports. After acceptance, use the read-only recorder check; renewed fixtures need new versioned paths.

The compiled dashboard backend snapshot shows five completions, no fallback and six pairs. Separately labeled historical 25M cards retain five caps and one complete H14 label. Existing browser policy blocked visual verification; no new screenshot is claimed. Mutable activity, process snapshots and logs remain outside acceptance.

Evidence: [comparison](../build/horizon-finish-comparison.json), [production result](../build/horizon-finish-result.json), [independent audit](HORIZON_FINISH_INDEPENDENT_AUDIT.md), [combined audit proof](../build/horizon-finish-audit.json), [completed rerun proof](../build/horizon-finish-root-verification.json), [scope](HORIZON_FINISH_SCOPE.md). Finishing these requested horizons authorizes no training, target ingestion, weighting change or model promotion, and does not solve the positions or Congklak.
