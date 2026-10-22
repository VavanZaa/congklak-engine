# Independent Stage 5 audit

The independent auditor owns its acceptance suite and audit recorder, rather than the feature trainer, native evaluator or arena implementation. **Implementation, recovery and experiment integrity pass. Stronger play is not established; H_v1 remains the default.** The completed audit checks all 407 committed root experiments and independently replays all 72 arena games. Both selected new models fail the ordinary no-regression promotion gate.

## Frozen contracts

The count-only control has 15 inputs. The feature challenger adds 14 occupied and 14 singleton indicators in normalized controller-relative pit order, for 43 inputs. Both reconstruct fixed-player-0 value as exact store difference plus signed active seeds multiplied by tanh. Terminal values are exact store differences. There is no runtime fallback or learned gate. Ordinary-stratum development error is a separate promotion gate.

Eight matched CUDA experiments use inputs 15/43, widths 32/64 and seeds 2718/31415, each with 8,000 optimizer steps, batch 512 and selection every 100 steps. Paired mean development macro MAE chooses a width for each input count; then lower macro MAE chooses that width's seed. The saved plan was fixed before these results and before arena measurements. Comparison with the earlier 2,000-step models includes additional training, so feature attribution uses the new matched controls.

The selected width-64 controls contain 5,249 parameters for 15 inputs and 7,041 for 43 inputs. Width and optimization budget are matched, rather than parameter count; the added indicators also add 1,792 first-layer weights. Any improvement applies to that combined change. Both are substantially smaller than the old 70,145-parameter width-256 candidate.

The fast evaluator packs weights by input and vectorizes four double output neurons at a time. It preserves each neuron's input accumulation order and uses separate multiplication/addition with strict floating-point compilation. Explicit scalar and AVX2 modes are tested; the automatic dispatch checks CPU and OS AVX support. Scores round the residual separately to 1/256 seed before adding exact store difference. Full-state, exact-horizon TT and model-local residual caches preserve fixed-player MAX/MIN search, passes and bonus turns.

## Independent implementation evidence

Nine checks pass in `build/stage5-audit/tests.json`:

- Original simulator, search, scalar neural DLL and original trainers retain their accepted hashes.
- 888 independently encoded/evaluated fixtures cover 15/43 inputs, widths 16/17/32/64, terminal/pass states, occupied/singleton discontinuities, antisymmetry, physical bounds and exact store-offset translation. Packed scalar and AVX2 raw double outputs match bitwise; the width-17 tail is exercised.
- The old W256 model matches the frozen scalar DLL bitwise on all 1,000 current development roots.
- Literal seed-by-seed rules and full-width minimax independently check 48 positions at depth four across eight backend/TT/PVS combinations. Every legal root-action score and optimal set agrees.
- A real analyzer child process is killed after committed depth two; resumption to depth five matches a fresh run. A second writer, changed backend and checksum-valid corrupt checkpoint are refused.
- Damaged encoders, shape, rules and oracle ancestry are refused. An interrupted first iteration supplies no recommendation.
- All 10,000 corpus states are checked through split/family metadata, including prior states, quotient aliases and family exclusions. No sealed target bodies are read.
- A real CUDA training child is killed after committed step ten. Resuming to step forty reproduces the complete model, optimizer, sampler, Python/Torch CPU/all CUDA random state and best-selection state bitwise. The loader returns only 8,000 train and 1,000 development rows.
- A real benchmark child is killed after committing its first full case. Resumption commits the second case without changing the first, repeated resumption is idempotent, and checksum-valid forged regret is refused. This verifies case-level benchmark recovery; an arena hard-kill is not claimed.

The independent audit recorder also checks all eight models' selection schedules and identities, independently recomputes selected-model development errors from literal scalar inference, and checks all 1,000 development roots per selected export. Both selected representatives fail the ordinary no-regression promotion gate.

## Equal-time protocol and limitations

The root plan fixes 148 original development cases (100 H6 heuristic proxies and 48 verified K4 exact references), plus all 259 current ordinary development cases. Their results remain separate. Five engines rotate order under equal 25/100 ms native search deadlines, depth ceiling 16 and TT16/PVS, with fresh working caches. Wall time additionally includes setup and result transfer. The two same-weight W256 engines isolate inference changes; matched 15/43 controls isolate the new features at the same training budget.

The feature candidate plays H_v1 from 24 distinct starting families, both seats, with eight ordinary/capture/bonus families each. A separate same-weights fast/scalar arena uses 12 families and both seats. Completed games use terminal seed margins; unfinished games receive no fabricated draw. Paired bootstrap intervals group by starting family. These are development experiments, rather than an untouched final-test estimate.

**Legacy exact split limitation found by this audit:** 16 of the 48 old K4 development roots have store-offset/player-swap quotient aliases among old sealed history states. The old constructed K4 families use quotient-based ownership, while old historical families retained their own family splits. No sealed target body was read to discover this; the audit uses only state/split/family metadata. These roots serve as known exact search diagnostics and must not be called an independent sealed/generalization test. This does not leak into the current models: every old quotient and all K<=4 positions are excluded from corrected training. The corrected corpus additionally isolates all train/development/sealed quotient aliases; its sealed targets remain untouched. A future broad holdout claim should use that corrected lineage, rather than the old K4 diagnostic split.

The direct inference measurements have substantial timing variation on this heterogeneous-core CPU. For example, identical width-64/input-15 operation counts yield different absolute latencies across measurement blocks. Median speed ratios must retain their sample ranges and measurement scope; they do not establish universal latency improvements or lower energy consumption. Core affinity and controlled power/timing conditions are a useful follow-up before changing runtime defaults.

## Final measured evidence

The selected 15-input and 43-input exports are both width 64, seed 31415. Independent inference on all 1,000 development roots per model exactly matches quantized native units; reported stratum errors differ from independent double-reference errors by at most 3.36e-6 seeds. Fixed-cadence histories and predeclared representative selection agree.

| Ordinary development stratum | Selected 15-input MAE | Selected 43-input MAE | H_v1 MAE |
| --- | ---: | ---: | ---: |
| 5–24 active seeds | 2.743 | 2.326 | 2.333 |
| 25–49 active seeds | 8.712 | 9.204 | 5.226 |
| 50–74 active seeds | 24.721 | 21.344 | 11.822 |

Added features improve low/high ordinary errors relative to the matched selected control, but regress in the middle band. The low band improves slightly against H_v1; the other two remain materially worse. The paired aggregate macro MAE also regresses with the extra features. These are errors against H6 heuristic labels, rather than game-theoretic values.

On all 259 ordinary roots, the feature engine's H6-proxy agreement is 74.90% / 73.36% at 25 / 100 ms, against H_v1's 82.24% / 79.15%. Proxy regret is 0.807 / 0.927 seeds versus 0.405 / 0.556. Proxy agreement can fall as search deepens; it must not be treated as an exact measure of eventual playing strength.

All 407 cases have valid, contiguous, complete five-engine measurements. The audit recomputes engine-order rotations, legal selections, agreement/regret, separate cohort/stratum summaries, sources, model/export/DLL identities and final catalog commit. On 273 comparisons where old scalar and same-weight fast engines completed the same horizon, values and optimal root sets agree exactly.

The literal-rule replay of all 72 games agrees with saved final boards, terminal margins, WDL scores and independently recomputed paired bootstraps. Every game reached terminal with zero fallback choices.

| Arena | Games | Candidate wins / draws / losses | Paired score |
| --- | ---: | --- | ---: |
| Features43 versus H_v1 | 48 | 23 / 2 / 23 | 50.00% |
| Same weights, fast versus scalar | 24 | 12 / 1 / 11 | 52.08% |

All 24 feature starts and 12 speed-control starts satisfy `abs(initial_store_margin) < active_seeds`; none has a mathematically fixed winner or win/tie boundary from banked stores alone. The feature arena nevertheless has the same played winner/tie within each swapped-seat pair. Its empirical bootstrap interval [50%, 50%] is therefore degenerate, rather than evidence of precisely equal strength.

A **descriptive secondary metric**, requested after the frozen protocol, averages candidate terminal seed margin across both seats so that initial store advantage cancels. Features43 versus H_v1 averages +1.708 seeds, paired bootstrap interval [-0.542, 4.083]. Same-weight fast versus scalar averages +4.333 seeds, interval [1.000, 8.167]. These small development samples and the post-protocol metric do not alter model selection or promotion gates. Per-family margins and start bounds are saved in `build/stage5-audit/arena-margin-secondary.json`.

## Final decision

The native implementation is eligible as an explicit experimental option. **H_v1 remains the default.** The new occupied/singleton features improve some ordinary strata relative to matched count-only models, but regress in another stratum and have worse paired aggregate macro MAE. No model or inference mode is promoted from these development numbers. Stage 5 remains ongoing: better ordinary evaluation, more controlled CPU timing and a broader predeclared arena are still required before promotion. No lower-power claim is supported because energy was not measured.

Evidence and reproducible commands:

```powershell
py -3.12 tests/stage5_audit_acceptance.py
py -3.12 tools/stage5-audit.py
```
