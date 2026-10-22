# TT/PVS, K=4 oracle and 10,000-position pilot

Phase two is implemented and its requested data artifacts are generated. The independent audit has its own full-minimax/seed-by-seed interpreter and packed-row decoder. See [audit verdict](PHASE2_AUDIT.md) and the [acceptance manifest](../build/acceptance/acceptance.json).

**62 checks passed with no failures, errors or skips**: a 61-method full discovery run in 173.094 seconds, followed by the independently added packed-row gate in 0.406 seconds. The [log](../build/acceptance/acceptance.log) records both runs. A fresh `py -3.12 tools/acceptance.py` discovers all 62 methods in one run. Source, compiler, binary, oracle and dataset identities are recorded in the manifest.

## Search

The disposable bounded TT uses full state, normalized controller, exact remaining depth and evaluator/policy identity. It stores exact/lower/upper bounds only after completed work. PVS follows actual MAX/MIN control through bonuses and forced passes. Every root candidate receives a full-window exact horizon value; canonical full-window reconstruction preserves lexicographic optimal continuations. Tables are recomputed after interruption; durable completed reports remain the recovery authority.

The semantic policy is `terminal-first/no-oracle/full-root-window-v2`. Requested TT/PVS settings, table capacity and search coverage are explicit report fields and teacher-budget identity. The default is 16 MiB TT plus PVS; both can be disabled independently.

Opening depth-seven ablation, including PV reconstruction:

| Options | Nodes | Deposits |
| --- | ---: | ---: |
| Alpha-beta, no TT/PVS | 116,854 | 3,204,302 |
| TT only, 16 MiB | 60,453 | 1,648,494 |
| PVS only | 140,139 | 3,651,651 |
| TT 16 MiB + PVS | 56,925 | 1,526,264 |

All four return value 46 for player A and best move A0 under the same horizon. TT+PVS uses about 51% fewer nodes here; PVS alone increases work. This is one fixed-position work comparison, not a strength or energy benchmark. Raw timing and identity are in [benchmark evidence](../build/phase2-search-benchmark.json).

## Exact oracle

[data/oracle-k4.sqlite3](../data/oracle-k4.sqlite3) covers **5,462 normalized states**, **9,520 legal edges** and all **6,120 raw-controller allocations** via explicit forced-pass normalization. Complete graph resolution establishes no cycle in the K<=4 domain; it does not assume every move reduces active seeds.

The stored future-award value F satisfies `F=max/min(deltaD+F(child))`, with terminal F=0; lookup returns existing store difference D plus F. Every value, tied action mask and successor/award edge is independently regenerated. Verification checks 19,040 extreme-store-offset transitions, range and parity. Terminal zero masks are valid; uncovered and incomplete entries are distinct.

The [oracle contract](K4_ORACLE.md) records proof assumptions, dependencies and recovery behavior. Ordinary horizon search keeps oracle lookup disabled. Exact root lookup is a separate operation and a separately labeled benchmark source.

## Generated pilot

The committed catalog is [data/pilot-10000/work.sqlite3](../data/pilot-10000/work.sqlite3). A readable [manifest export](../data/pilot-10000/manifest.json) mirrors that catalog; the JSON export is a convenience view, not publication authority. The catalog selects ten immutable packed row shards (52 bytes per row) and an immutable SQLite provenance/label index. Published bytes total **21,966,656**, including **520,000 packed-row bytes**. SQLite work pages and preserved prior generations are additional managed bytes; deletion of candidate rows does not promise immediate physical file shrinkage.

| Split | Positions |
| --- | ---: |
| Train | 6,851 |
| Development | 1,416 |
| Sealed final test | 1,733 |

There are 8,796 complete depth-four H_v1 targets and 1,204 exact K=4 benchmark targets. Every K<=4 root is excluded from train, and the train teacher has no oracle leaves or learned ancestry. Thus exact benchmark successor facts are not supplied through training oracle labels.

Sources contribute 3,493 random-history, 2,755 heuristic-history, 3,278 search-history and 474 constructed exact positions. All 10,000 full canonical states are unique after normalization and verified player-swap remapping. The pilot processed 22,213 thinned observations, merged 6,055 repeated states and excluded 1,875 train-owned endgame candidates. A fixed 2,000-unit collection ledger commits each family together with progress; its contiguous indices prevent a forged cursor from skipping generation.

Family ownership is fixed before generation. Holdout state/quotient exclusions are frozen before labeling. The eligible isolation graph contains retained examples and their owning families; overlapping observed copies are excluded. Archives are replay evidence, not hidden extra learning rows. Protected exclusions remain available after candidate compaction. Future generations must import these exclusions; arbitrary independent campaigns cannot be merged into one isolated lineage.

The pilot has 18 primary active-seed/mechanism strata, with balanced quotas capped by availability. Seeded uniform-priority selection is without replacement and records inclusion probability `quota/eligible_count`. The learning sampler separately draws strata uniformly, then rows uniformly with replacement, recording P=q. Loss coefficients and sampling priorities are one; importance ratio is one. No model-error weighting, priority mixture, quality eviction or training is performed.

## Recovery and verification

The actual 10,000-row campaign was killed after at least 1,000 committed labels and resumed successfully. Its measured wall time was **92.297 seconds**, including collection, interrupted work, resume checks, labeling, revalidation and publication: approximately **108.35 completed records/second** on this run. Fresh all-action label verification is additional work. The committed collection/label-only rate is about 193.77 records/second and excludes those recovery/publication costs; it must not serve as an overall ETA estimate.

After the cursor-ledger audit fix, the full campaign was regenerated. The earlier untampered generation was preserved at `data/pilot-10000-pre-ledger`, and the final generation has the same semantic content digest and byte-identical packed shards:

`125b7a81b41baeeeb3f44bdf4f7bac227423d564db106e717a88ccbe1ace7bcb`

Test processes were killed before and after chunk, label and manifest commits, including after final-named files existed but before their catalog commit. Resume matches fresh semantic content. Real SQLite-full failures preserve committed work. Checksum-valid malformed kinds, horizons, teacher identities, keys/provenance and collection cursors are rejected before publication. Capacity checks reserve 2 GiB bulk recovery space within the 50 GiB physical quota.

Every retained family and originating state was independently replayed; every exact pilot target and action score was checked with the independent K=4 solver. The independent audit also checks sampled horizon targets by full minimax and independently decodes all 10,000 packed rows. `pilot verify --relabel` freshly recomputed **every** horizon root and action score with the declared native teacher, and checked every exact root/action against the pinned oracle. Process-memory monitoring uses a 512 MiB default and bounded collection/row buffers, not a hard OS reservation.

## Development measurements and limits

[measurements.json](../data/pilot-10000/measurements.json) includes source, store-difference, active-seed, legal-action-count and mechanism coverage; actual storage/loader measurements; and a deterministic development-only depth-four versus depth-six sample. Among 64 non-oracle development positions, 55 changed root value and 20 changed best-action set. These observations identify teacher instability; they do not certify deeper targets or promote a model. Sealed final-test results were not used to choose budgets, weights or retention.

The declarations still apply only to the experimental single-round rules. This phase creates a validated pilot and small exact oracle, not a full-game solution or expert-strength evaluator. GPU/model training, larger oracles, cross-generation exclusion registries, parallel labeling and a general protected-artifact garbage collector remain subsequent work. Power-loss hardware testing and energy measurements have not been performed.
