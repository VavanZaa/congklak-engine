# Engine algorithm and source-reference audit

This is a focused architecture review and CPU regression check, not a strength comparison, independent-agent audit, or proof of complete engine correctness. No engine source, model, sampling distribution, or prior evidence was changed. No training or strength games ran.

External snapshots are in `build/algorithm-reference-audit-20261015/`. Stockfish was pinned at commit `5c4cef0bfe61fb3249178b2bcf21f5673557efe7`; OpenSpiel at `48401890ee9857e611678302371378175a8e4c6b`. Their original copyright/license notices are retained. External code was read, never executed or incorporated into our engine.

The AlphaZero reference is the [published algorithm](https://arxiv.org/abs/1712.01815) plus Google DeepMind's [OpenSpiel implementation](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/alpha_zero/alpha_zero.py). OpenSpiel's [documentation](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/docs/alpha_zero.md) explicitly distinguishes its illustrative implementation from the original paper's production code. Its old `model.py` documentation link returned 404; current `model_nnx.py`, README and replay buffer were obtained at the same pinned commit. The original failed request remains in the source manifest.

## Names and actual algorithms

`Hn` means a complete finite search horizon of n pit choices, or earlier termination. It is not a model generation, training epoch or number of seed drops. Automatic relay sowing and forced-pass normalization are part of the transition; a bonus grants another choice, which consumes another horizon step. `H_v1` names the baseline evaluator. The H8 relay model's name records its historical training horizon, not its current maximum analysis depth. K=4 counts active seeds and is unrelated to H4.

The default/teacher route is exact rules -> increasing finite horizons -> alpha-beta/PVS with a bounded TT -> current banked-seed difference at a nonterminal horizon, or exact final margin at termination. See `native/search.cpp` and `native/rules.cpp`. Full-root targets and canonical continuations are audited separately from playing strength.

The learned alpha-beta route substitutes a small neural evaluation. The experimental PUCT route uses a policy prior to allocate visits, evaluates new leaves with the network, and backs up fixed-player values. Its current tree traversal/selection/backup is in `congklak/quality_search.py`; native C++ implements transitions, relay features and dense inference. `native/neural_puct_cpu.cpp` is an inference boundary, not a whole C++ tree-search implementation.

The shared compact network has 127 inputs, two width-64 hidden layers, seven policy logits and one value output, totaling 12,872 parameters. Its value is a store offset plus an actor-relative residual scaled by remaining seeds. Latest value-head experiments train 65 parameters while freezing the shared trunk/policy. The recent data pilot trained nothing. Earlier Phase 6 evidence includes a real 64-game self-play and one subsequent learning cycle; that is distinct from repeated improvement of the latest candidate.

## Reference comparison findings

| Finding | Evidence and implication |
| --- | --- |
| Baseline leaf evaluation is simple | `native/search.cpp:280` and `native/rules.cpp:102` use current stores at the horizon. Teacher labels can inherit horizon effects; more copies do not remove them. |
| Current PUCT is partly native | Python owns tree operations; C++ owns rules/features/inference. Profile tree overhead before claiming fully native search or choosing a rewrite. |
| Value objectives differ | Our declared payoff is final seed margin. Original AlphaZero's chess target is game outcome. A win-rate objective would need a separately declared experiment. |
| Self-play plumbing exists; sustained learning remains open | Phase 6 has policy visits, completed-game returns, replay and checkpoints. Current teacher diagnosis does not establish that repeated latest-candidate self-play improves strength. |
| Evaluation cost differs from NNUE | Stockfish's [evaluation](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/evaluate.cpp) uses NNUE and [incremental accumulators](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/nnue/nnue_accumulator.h). Our features simulate many legal sowings. Congklak relays may change many pits, so update economics require measurement. |
| Playing search and teacher semantics must stay separate | Stockfish's [search](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/search.cpp) includes quiescence, extensions, reductions and selective pruning. Our strict teacher uses a declared fixed horizon. Adding these features changes target semantics and needs a new version. |
| Deeper-cache reuse cannot be copied blindly | Our TT requires the same remaining horizon. A deeper finite-horizon value need not equal a shallower value; Stockfish's search has a different contract. |
| Exact facts remain separate | K=4 is explicit verified lookup, not a learned estimate or an automatic PUCT proof mechanism. |

The [OpenSpiel MCTS source](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/mcts.py) is useful for checking actor-aware reward handling, legal priors and visit accounting. Its general solver options and implementation details should not be treated as mandatory parts of original AlphaZero.

## Actual checks and limits

`tools/audit-engine-reference.py` ran eight selected existing CPU methods: four TT/PVS/cancellation/budget checks and four PUCT actor, independent-tree/noise, bootstrap/store-offset and cancellation checks. All eight passed with zero failures, errors or skips. Their normal script entry points were not invoked, so no sealed historical test report was overwritten. All 2,820 prior input hashes remained current. The new proof and log are `build/algorithm-reference-audit-20261015/local-checks.json` and `local-checks.log`.

The selected checks found no new correctness failure. They do not establish equivalent behavior or strength to Stockfish/AlphaZero, cover all possible states, audit the unavailable original production implementation, or demonstrate lower energy use. The actionable gaps are evaluator/target quality, evaluation and tree costs, sustained self-play learning, fresh strength evidence, and product integration. Neither a larger model nor copied chess pruning is an automatic remedy.

For discussion, prefer “depth-8 teacher label,” “baseline evaluator v1,” and “relay policy/value model” over using H for several unrelated names. Preserve the existing machine identities and sealed evidence.
