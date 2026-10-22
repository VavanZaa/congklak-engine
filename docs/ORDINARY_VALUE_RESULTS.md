# Ordinary validation and value-only pilot

**The candidate is not promoted, and the conditional self-play followup is held with zero games.** H_v1 remains the default analyzer and the existing compact control is retained. The candidate's point estimate was worse against H_v1, unchanged against H8 relay, and slightly worse on the six-root primary H10 validation error. These small consumed-development results do not demonstrate a strength gain.

The [protected projection](../build/ordinary-value-benchmark-projection.json) reserved 12 arena families before learning, excluding all 144 previous arena family IDs, plus six ordinary validation roots and 32 reused exact endgames. New-to-arena IDs still have development and ancestral exposure; they are not an unseen release test. Validation roots and known evaluation trajectories were excluded from optimizer admission through the declared origin-family closure.

All six validation roots completed requested H8, H10 and H12 labels: **18 labels, zero caps and 12,169,649 production teacher nodes**, against the fixed 102-million requested-attempt ceiling. Fixtures and audit searches have separate costs. Best-move sets agree at H8/H10 for all six, but one high-band root changes from display pit5 to pit6 at H12. Mean absolute target shift is 1.67 seeds from H8 to H10 and 3.33 from H10 to H12; another high-band root retains its best move while its target shifts by 11 seeds. The [saved labels](../build/ordinary-value-labels.json) remain finite-horizon estimates. H10 was the predeclared primary reference and was not replaced after seeing deeper results.

Training used 48 initial self-play records from 24 starts, with the 14 ordinary descendants excluded, and 7,539 admitted historical H8 rehearsal roots after 10 origin families/24 rows were filtered out. Each batch contained 16 high-band, 16 low-band and 32 rehearsal draws; rehearsal used the declared 90% root/10% stratum mixture. All losses had unit per-draw weight. Only the **65 value-head parameters** trained; the other 12,807 parameters, including the entire policy subgraph, remained bit-exact. The fixed-final run completed 200 steps, 12,800 draws and ten verified checkpoints, without checkpoint selection from outcomes.

| Training-target MAE, seeds | Control initialization | Final candidate |
|---|---:|---:|
| High ordinary terminal returns, 24 records | 13.2581 | 13.1459 |
| Low ordinary terminal returns, 24 records | 9.6088 | 9.4854 |
| Historical approximate H8 rehearsal, 7,539 roots | 4.43955 | 4.43734 |

These are in-sample fits to different target types, including two dependent control-policy outcomes per ordinary start. They are not strength evidence. On all six protected requested-H10 roots, literal exported-model validation MAE instead changed from **9.59909 to 9.63205 seeds**. This tiny panel does not establish population accuracy.

The [equal-time benchmark](../build/ordinary-value-benchmark.json) completed **96 terminal games, 3,268 decisions, 38 cases and 304 scalar measurements**, with zero fallbacks. Control and candidate used the same c_puct0.25, uncached native CPU inference and pinned core; whole 25ms game decisions included feature extraction, inference and validation. The other engines were H_v1 and the H8 relay CPU baseline. The six approximate-reference cases were kept separate from the 32 exact guards at 25/100ms.

| Opponent | Control mean margin | Candidate mean margin | Paired candidate minus control | Exploratory 95% interval |
|---|---:|---:|---:|---:|
| H_v1 | -0.4167 | -1.0833 | -0.6667 | [-1.4167, 0] |
| H8 relay CPU | -0.5833 | -0.5833 | 0 | [-0.6667, 0.5833] |

Margins are seeds, paired over the same 12 families and both seats. The intervals are unadjusted development bootstrap estimates, not a release or promotion test. Both models chose an exact-optimal move on all 32 reused endgame roots at both clocks; half these roots have only one legal action. This observed guard is useful but not exhaustive endgame certification.

The predeclared **admitted-pool** overlap guard triggered a conservative hold: two saved arena observations share one late K5 player-swapped/store-offset quotient with a historical rehearsal row. There is **no identical full raw state and no family collision**. Independent reconstruction of all optimizer draws found that particular rehearsal row had **zero draws**; this is not evidence of actual gradient leakage or a causal game error. The frozen rule nevertheless protected the entire admitted pool, and was not weakened after outcomes. Separately, the worse validation fit and negative H_v1 paired mean also failed the favorable allocation conditions. No retrospective filtering, retraining or replacement arena occurred. The [held followup](../build/ordinary-value-followup.json) adds all 96 new trajectories to the future protection ledger and authorizes no further training.

All **18 current unique checks passed**: four learning-owner, six benchmark-owner and eight independent methods, including ten actual process kills. The [independent audit](ORDINARY_VALUE_INDEPENDENT_AUDIT.md) verifies labels, all 200 optimizer updates/ten checkpoints, frozen policy, all games and reference denominators, allocation and recovery. Preproduction fixtures did catch finite-target parity, cancellation, feature-cache metadata and original-root/wrapper-link issues; their exact failed sources, reports and logs, including the first benchmark freeze and metadata-stub failure, remain retained and bound. The passing current results do not erase those earlier failures.

The [completed no-work proof](../build/ordinary-value-root-verification.json) preserved 37 durable files with identical bytes, sizes and modification times, and all 3,364 prior inputs. It performed zero new searches or optimizer steps and imported no Torch; deterministic literal CPU verification remains permitted. Recorded invocation costs were 6.875s/681.74MiB peak working set for training and 209.368s/342.59MiB for the benchmark. These are separate process observations, excluding audit/fixture costs; energy was not measured.

After sealing, verify with `python tools/record-ordinary-value.py verify`; rerunning owner suites creates fresh evidence and belongs in a separately versioned run. The progress dashboard remains at `http://127.0.0.1:18766/`. It was checked through the local backend; browser verification was not performed because earlier browser access was denied. Further training or large data expansion is not justified by this pilot's small fit changes.
