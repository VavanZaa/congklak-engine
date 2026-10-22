# Independent phase-two implementation audit

Reviewed 14 October 2026, Asia/Bangkok.

**Accept TT/PVS, the verified K=4 oracle and the completed 10,000-position pilot for the experimental id-betawi-relay7-v0 profile. No unresolved implementation blocker remains in this scope.** Final evidence records **62 passing tests, zero failures, zero errors and zero skips**: a 61-test discovery run in 173.094 seconds and a subsequent independent packed-data gate in 0.406 seconds. A fresh discovery includes all 62 tests.

The [acceptance manifest](../build/acceptance/acceptance.json) and [test log](../build/acceptance/acceptance.log) record those two runs separately. The auditor re-read both and independently verified that all **28 recorded source hashes** match the final files. This verdict concerns the declared rules, finite-horizon search, exact small-domain coverage, data provenance and process-interruption recovery. It does not establish championship strength, energy savings or a solved full game.

## Independence

The auditor owns [test_phase2_audit.py](../tests/test_phase2_audit.py) and this document and did not edit production code. Expected transitions, full minimax scores, optimal action sets and canonical continuations come from [reference.py](../tests/reference.py), which imports no production transition, search or oracle helpers. Oracle expectations use a separate recursive future-award solver with explicit cycle detection, rather than the production sink-first graph solver. The packed-data gate uses an independent little-endian decoder and never calls the production packer.

Review covered native TT/PVS and the control boundary, report identities and replay validation, oracle construction/verification/read snapshots, pilot collection and unit ledger, canonicalization and selection, label/context validation, immutable publication, sampling and measured development-only stability. Production authors corrected the findings below before acceptance.

## Executed evidence

| Gate | Verified result |
| --- | --- |
| Search semantics | Four combinations of TT/PVS agree with independent full minimax. The complete 8,274-state reduced corpus remains covered, with both actors, stores, bonus choices, forced passes and horizon-zero cases. The auditor's additional corpus checks 4,176 reports, including every candidate value, every optimal tie and the first lexicographic optimal continuation |
| TT safety | Full state, pits, normalized controller, exact remaining depth and evaluator/policy identity are checked. Exact entries and directional bounds are distinct; bounds cut off only when they prove the existing window fails. Interrupted subtrees never publish their own incomplete entry. Tables are bounded, request-local and disposable |
| PVS safety | Windows stay in fixed player-0 coordinates; MAX/MIN follows the actual controller. Every displayed root alternative receives a full window. Separate full-window reconstruction preserves deterministic legal PVs despite cache ordering and null-window probes |
| Oracle coverage | Independent enumeration verifies all 5,462 normalized records, including two terminal controllers, and all 9,520 legal edges. Independent DFS detects no cycle; 5,968 edges preserve active-seed count. Every future-award value and complete optimal mask agrees |
| Store reduction and passes | Builder verification checks 19,040 shifted-store transitions. The auditor additionally checks 12,240 raw-controller/extreme-store lookups, including empty-row aliases, mask ownership and the distinction between terminal zero masks and uncovered positions |
| Pilot completeness | The committed artifact contains exactly 10,000 valid unique canonical states: 8,796 finite-choice-horizon labels and 1,204 exact K=4 benchmark labels. Every family/archive and selected origin replays independently; all exact root/action labels agree with independent solving, and 48 additional horizon positions agree with independent full minimax |
| Full teacher verification | The complete published pilot was re-labeled with the frozen teacher, comparing every root target, optimal set and per-action score. The declared horizon is four; the teacher has no oracle leaves |
| Packed export | All 10,000 52-byte rows independently decode to the indexed state, label kind, horizon, scalar and seven action values, masks, weight, draw probability, stratum and split |
| Weighting and splits | Unit loss weights, priorities, inclusion probabilities and uniform-with-replacement stratum draw probabilities are checked. The retained family/state graph has one split owner per family and one canonical owner per state. Cross-split copies are excluded; every K<=4 record is excluded from train to prevent oracle-dependency leakage |
| Recovery and faults | Real processes are killed before/after collection and label commits, during export, around catalog publication and through oracle construction. Resumption reproduces semantic content; uncommitted files never establish completion. Real SQLITE_FULL, admission refusal, damaged files, incompatible identities and checksum-valid malformed records are exercised |
| Actual 10,000-row campaign | The final campaign was killed after at least 1,000 durable labels, restarted and completed. Its semantic digest and all ten packed shards match the retained content from the preceding valid campaign; separate controlled recovery tests compare interrupted runs with fresh uninterrupted corpora |

The final pilot has **6,851 train**, **1,416 development** and **1,733 sealed-final-test** records. Family-level split proposals do not promise exact row percentages, and the oracle domain is benchmark-only. Sources comprise 3,493 random-history, 2,755 heuristic-history, 3,278 search-history and 474 explicitly counterfactual constructed records. Historical archives provide provenance; they are not additional loader examples or outcomes for unfinished games.

## Findings corrected before approval

| Finding | Correction and recheck |
| --- | --- |
| Exact training-root values can inherit sealed successor facts even when the root quotient differs | All K<=4 states are benchmark-only; training uses the untrained H_v1 lineage with no oracle leaf lookup |
| Checksum-valid malformed oracle manifests and SQLite errors escaped as generic exceptions | Strict metadata/manifest validation and structured persistence errors refuse them, preserving the SQLite cause |
| Oracle verification followed by reopening allowed a snapshot gap | The reader holds the verified SQLite read transaction throughout its lifetime |
| A constructed quotient family merged different store-offset archives | Archive IDs now include the full canonical state while quotient aliases retain consistent benchmark ownership; every selected origin replays |
| Checksummed labels could carry invalid teacher, horizon or proof identities | Shared strict validators run at commit, resume, before publication and during verification; exact labels and all teacher action values are rechecked |
| A valid record body could be copied into another database key or name an unrelated selected origin | Pending validation binds body/key to the selected position, family, step, swap, split and probabilities, and replays the originating archive before publication |
| A checksum-valid cursor change from 11 to 32 skipped 21 families and still published a dataset | SQLite work schema 2 transactionally records contiguous, checksummed collection units and archive hashes. The regression refuses the forged cursor before publication. The final pilot was regenerated after this correction |
| Campaign identity omitted Python generation code, and archive seeds differed from actual RNG seeds | Implementation source hashes are pinned, and archives record the actual seed + index*104729 |
| Shared opening states made family components ambiguous | The declared eligible graph comprises retained examples and their owning families. Cross-split observation copies are excluded; full historical archives remain provenance evidence |

## Accepted identities

- Native DLL SHA256: d0066bb395db3323c46b372ab32f2c80e0a6845343282ec3cc894634d22faf3d.
- Rules digest: c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e.
- Horizon policy: terminal-first/no-oracle/full-root-window-v2, with H_v1 in integer seed units.
- Oracle manifest SHA256: 76b465742996803525322054c8e862c99af54b65c42c6638f5a20318c85d5cbd.
- Pilot semantic SHA256: 125b7a81b41baeeeb3f44bdf4f7bac227423d564db106e717a88ccbe1ace7bcb.
- Pilot teacher revision: 8470eb12516d1a4e9f71aebae2d5d2cf792e46eb15b713a7e7f9f560e40849e5.

The default pilot catalog is [data/pilot-10000/work.sqlite3](../data/pilot-10000/work.sqlite3), with 2,000 committed collection units and ten packed shards. Committed catalog pointers and verified file closures establish publication; filename existence alone does not. The published index plus shards occupy **21,966,656 bytes**. Work databases, retained earlier generations and recovery files consume additional physical storage and remain included in quota admission.

## Practical limits

The [measured report](../data/pilot-10000/measurements.json) records 92.297 seconds for the actual interrupted campaign, approximately 108.35 completed records per second including its observed recovery/export work. The separate committed collection/label rate excludes resume revalidation, publication and lost in-flight work and is explicitly labeled accordingly. This is one local pilot observation, not an overnight or larger-campaign guarantee.

On the frozen development-only 64-position stability sample, increasing depth four to six changes 55 root values, 20 optimal sets and at least one action value in all 64 positions. This supports treating depth-four targets as a declared baseline rather than final-game truth. No model has been trained or promoted, no quality-based pruning or calibrated reliability weights were introduced, and energy was not measured. Sealed results did not select teacher budgets or retention settings.

The tests establish process-termination recovery, not hardware power-loss behavior. Memory is bounded and monitored rather than reserved through a hard OS cap. Source/build changes require compatible identities or a new campaign. Protected exclusions currently belong to this campaign: future training generations must carry them forward before admission, and independently created pilots must not be combined into an isolated evaluation lineage without reconciling their permanent ownership/exclusions.
