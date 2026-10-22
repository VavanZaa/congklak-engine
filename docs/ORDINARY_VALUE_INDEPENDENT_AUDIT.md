# Independent ordinary value pilot audit

All eight current independent methods passed. The implementation, fixed-policy ownership and interruption recovery checks pass; the frozen data-allocation rule selects **hold**, so production followup contains **zero games**. H_v1 remains the default and no further optimizer run is authorized.

The combined proof is `build/ordinary-value-audit.json`, with six source-bound groups in `build/ordinary-value-audit/`: preflight (3 methods), labels, model, recovery, benchmark and final (1 each). The current main source is SHA256 `2c5921957e4f6dce402b26634e151e04a96d73e62f59c34eeb1e64762ff67ccc`; the isolated worker is `94ab4b2d118266dc10deea58561367510a8d396657e97ed968fd69b5fbce97dc`. The source-freeze artifact records the retained first execution and corrected current revision. Across the current author and independent suites there are 18 unique passing methods: 4 learning, 6 benchmark and 8 independent. Failed or superseded executions are retained and are not counted again.

## Data, targets and optimization

The initial model is the unchanged accepted control `data/quality-train-v1/model.json`, file SHA256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. Independent admission reconstruction checked all 7,587 optimizer rows: 7,539 earlier H8 rehearsal roots and 48 individual choice0 terminal-outcome records from 24 ordinary starts, twelve in each band. All 62 preceding ordinary observations participate in whole-origin-family bridge detection, while the 14 descendants never become gradient rows. The entire validation ownership pool of 100 families and 38 eligible quotients, previous evaluation roles, latest96 trajectories and new arena reservations remain protected. Historical training metadata is not silently treated as an evaluation ban; filtering new draws does not undo ancestral exposure.

All six reserved validation roots completed all three requested horizons: H8, H10 and H12. Independent checks covered 18 requested labels, 180 complete iteration reports, 930 literal PV paths and 5,239 literal choices, including root action-Q coverage, actor orientation, ties, bonus turns, work and identity links. New teacher work was 12,169,649 nodes against the frozen 102-million-node ceiling. H10 alone is the primary reference; H8/H12 remain separate. These completed finite-horizon estimates are not full-game truth. Only already-consumed diagnostic roots were used for engineering label fixtures before production answers existed.

The independent Torch worker rebuilt the frozen trunk cache and reexecuted every one of the 200 declared CPU updates from the source tensors. It matched all ten actual checkpoint states, including the complete model, two-parameter AdamW state, sampler state, Python RNG and Torch RNG by raw tensor bytes. All six trunk/policy tensors and all cached policy logits remained byte-identical. Exactly the two value-head tensors changed; 65 parameters were trainable and 12,807 frozen. The final exported weights match the actual final PT state (weights SHA256 `efed8c15fd61a1b6239c4f7200e9d55b629102353b9db0ef7849256a38a7a112`). Corrupt frozen tensors, optimizer ownership/shapes/weight decay, draw counts and sampler states were refused.

The checkpoint-bound reconstructed ledger contains 12,800 draws: 3,200 high-band terminal observations, 3,200 low-band terminal observations and 6,400 H8 rehearsal observations. It follows the frozen root-then-outcome sampling order and 90% root/10% stratum rehearsal CDF. This ledger is deterministic reconstruction checked against saved RNG states, not a separately logged independent sample. Each draw has one normalized Huber loss; no second weight or policy loss was applied. The rehearsal targets and on-policy terminal returns have different estimands, so this intervention does not isolate a weighting, target-horizon or data-source effect.

## Actual interruption and completed-run checks

Ten actual child-process kills were executed and recovered:

| Boundary | Physical committed prefix |
|---|---|
| H10 label pre/post commit | 1 / 2 root attempts |
| Checkpoint step4 file-ready/catalog-committed | catalog steps `[2]` / `[2,4]` |
| Benchmark decision pre/post commit | 1 / 2 decisions |
| Benchmark game pre/post commit | 0 / 1 complete games |
| Followup decision pre/post commit | 1 / 2 decisions |

Raw worker logs, independent helper markers, SQLite prefix backups, PT files and state-comparison proofs are retained in the audit directories. Producer marker files can be interrupted before atomic publication; they are checked when present, rather than assumed to exist. Training resumed to the same full model/optimizer/RNG state as a fresh six-step fixture, while committed PT bytes and mtimes stayed unchanged. Timed benchmark recovery preserves committed decisions and literal legal/terminal continuation; it does not claim future timed choices must equal another fresh run. Followup recovery used the predeclared first-original-high-root, two-game, four-simulation engineering fixture. It is distinct from the unused 64-simulation production followup.

Completed APIs were exercised in isolated no-Torch processes with teacher search, PUCT, benchmark decisions and training runtime prohibited. Durable bytes and mtimes stayed unchanged. Valid missing label exports repaired without search; rehashed corrupt label/history/completion markers and a corrupt benchmark completion marker refused before publishing a replacement. All 3,364 preceding baseline hashes remain current. The root's later combined no-work and acceptance checks are additional publication checks, not extra independent method counts here.

## Evaluation and held allocation

The independent benchmark method literally replayed all 96 terminal games and 3,268 choices and checked all 304 measurements across 38 roots. It rederived all four primary paired-family intervals and both contemporaneous candidate-minus-control intervals from raw margins, plus H10 errors using literal exported-weight calculations. Both value profiles met the observed 32/32 known-exact guard at 25ms and100ms, with zero exact regret and no fallbacks. These reused guards are not a universal endgame proof.

The candidate-minus-control paired mean was -0.6667 seeds against H_v1 (exploratory95% interval [-1.4167,0]) and 0 against H8 relay CPU ([-0.6667,0.5833]). Neither profile passed the two-baseline primary strength gate. On all six complete requested-H10 roots, MAE changed from 9.599085 to 9.632052 seeds. This small consumed-development experiment does not establish improved strength or generalization.

The predeclared admitted-pool overlap gate found two arena observations sharing one active-5 store-offset quotient with rehearsal row `rehearsal/000000000000002d010000020100013001` (`random_history/20261015/7686`, `5-24/pass`). The occurrences are game0 and game4, choice31. Read-only comparison independently confirmed that neither is the same full raw state and that this row had **zero draws in this pilot's reconstructed optimizer ledger**. Thus this is a conservative admitted-pool/player-swap/store-offset-pattern overlap: it is not evidence of an actual gradient from that row in this pilot, exact raw-state reuse, absence of ancestral exposure, or a causal explanation of the game outcomes. The fixed gate nevertheless requires hold; it was not relaxed after seeing results, and no rows were retroactively removed or models retrained.

The held followup identity records the new96 trajectories for future protection, but no followup collection or rollout was launched. The unused fresh-start and repeat-high production branches are not claimed as executed production data. No candidate was promoted and no new training was triggered.

## Retained corrections and limits

Before production, review fixed a duplicate sampler identity keyword, finite-horizon parity validation that incorrectly assumed terminal parity, and an original-root versus study-wrapper SHA mismatch in benchmark admission. The first independent preflight also exposed a real public-CLI cancellation boundary bug: a non-callable CancellationToken reached code calling `cancel()`. The repaired producer uses token-aware `stopped()` checks and preserves native-control cancellation registration. Current owner checks include fresh public label and training CLIs, and the independent real-kill preflight was rerun successfully after refreezing.

The same first independent run caught an audit assertion that compared cached feature metadata with only the numerical `{x,d,r}` projection. That assertion was corrected; all numerical inputs were then rederived. Exact executed sources, the failed report, raw logs and incomplete fixture journals are retained under `build/ordinary-value-audit-first-preflight-failure/` and the recorded raw fixture directory. Producer cancellation and benchmark-link repair bundles retain their own original sources/manifests/proofs. None of these repairs changed training targets, budgets, model selection or post-result behavior, and no earlier accepted artifact was edited.

The audit establishes this bounded experiment's identities, calculations, recovery and declared hold. It does not establish calibrated terminal targets from two trajectories, full-game optimal labels, independent sealed evaluation, product readiness, or beneficial larger training.
