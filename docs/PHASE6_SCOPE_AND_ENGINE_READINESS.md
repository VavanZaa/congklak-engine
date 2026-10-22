# Phase 6 scope and engine completion checklist

Status: the bounded Phase 6 engineering pilot is complete with 42 acceptance checks, including 14 independent checks. Both CUDA/PUCT models failed the strength gate against both CPU baselines, so the roadmap's Phase 6 strength requirement remains open. H_v1 remains the default. The user confirmed Betawi on 2026-10-15. See [measured results](PHASE6_RESULTS.md) for completion, cost, recovery and strength evidence.

The intended first finished product is a strong, reliable position analyzer with move rankings and explanations for the declared `id-betawi-relay7-v0` Betawi profile and final single-round seed-margin objective. A full-game solution is a separate later objective. The existing verified rule contract remains unchanged. User confirmation establishes the project target; it does not establish a universal regional or national rule standard.

## Verified foundation

The simulator, CPU alpha-beta/PVS search, bounded TT, resumable analysis, K=4 exact database and explicit lookup, supervised CUDA training, CPU neural inference, and progress dashboard are implemented. The deeper/relay experiment passed 34 acceptance checks. It produced 9,000 H8 targets, eight training runs and 640 verified checkpoints. Actual analysis, teacher, training, case and arena interruption checks passed.

The selected H8/127-input, width-64 model passes the ordinary H8 error gate and its versus-H_v1 paired seed-margin gate: +1.3056 seeds, family bootstrap interval [+0.2778,+2.4444]. Its WDL score is 50.69%; the games were development games. H_v1 remains the default and the H8 relay model is available as an experimental CPU candidate. See [completed results](STAGE5_DEEPER_RESULTS.md) and [independent audit](STAGE5_DEEPER_INDEPENDENT_AUDIT.md).

CPU teacher estimates and completed game outcomes are different training targets. The selected H8 CPU candidate was trained on finite-depth teacher values. Phase 6 separately warmed its policy head, generated 64 self-play games and trained one updated generation on 623 admitted replay rows plus teacher rehearsal. The measured CUDA/PUCT challengers are weaker than both CPU baselines; neither replaces the existing candidate or default.

## Bounded Phase 6 implementation sequence

Keep the existing compact architecture and existing H8 corpus for the initial policy warm start. Freeze budgets, model selection, dataset ownership, comparisons and admission rules before production outcomes. Measure a bounded pilot before enlarging model capacity or launching a larger corpus campaign.

1. **Train useful legal-move priorities.** Train the policy head from the saved H8 root-action scores, with an explicit actor-relative conversion, legal mask and recorded score temperature. Initially preserve the existing value weights so the policy experiment has a clear control. A soft distribution derived from approximate scores is supervision, not a calibrated probability of optimality. Verify the exported model and confirm that the frozen value function is unchanged.

2. **Implement and independently check PUCT.** Start with a single-worker CPU reference implementation. Use fixed-player values or an equivalent explicitly verified actor conversion. Bonus moves and normalized passes can retain the same player; unconditional sign changes at every edge are invalid. Keep estimated visit/value statistics separate from exact terminal or tablebase facts. Bound tree memory and simulations, honor cancellation, and expose completed root visits and legal recommendations.

3. **Add GPU inference with bounded batching.** First batch independent roots or games. Measure feature extraction, queue delay, transfer, GPU evaluation and total decision latency. Test single-position latency separately from multi-game throughput. Use declared numeric tolerances for identical-model CPU/GPU conformance, and record startup/warmup policy. A queue must not wait indefinitely to fill a batch. CUDA availability alone does not establish a GPU search advantage.

4. **Generate recoverable self-play.** Store starting state, legal moves, rules/model/search identities, seeds and RNG state, decision-boundary snapshots, root visit distributions, and final seed allocation for completed games. After interruption, retain committed decisions and recompute unfinished tree work. Incomplete games do not receive fabricated terminal outcomes or draw labels. Replay every admitted completed game with the independent interpreter.

5. **Perform a small learning iteration.** Train a new model generation from admitted visit-policy targets and completed-game returns, with target sources, sampling and loss coefficients recorded separately from teacher supervision. Preserve train/development/sealed ownership and all prior state, quotient and family exclusions. Deduplicate retained examples and keep compact game provenance. Keep rare mechanisms and protected evidence when pruning; low current loss alone is not a deletion rule. Compare the updated generation with the warm-start generation so game generation is actually tested as learning input.

6. **Measure and audit the challenger.** Compare against both H_v1 and the frozen H8 relay CPU candidate. Predeclare fresh evaluation families, paired seats, whole-decision time budgets, primary seed-margin metric, uncertainty, fallback accounting and incomplete-game treatment. Report WDL separately. Record memory/VRAM, generation and training costs, GPU batching throughput, single-position latency and feature costs. Energy remains unmeasured unless a scoped measurement is performed. Independently test process kills during policy training, game publication and self-play restart.

A recoverable pilot that fails the strength comparison is still useful engineering evidence, but does not satisfy the roadmap's Phase 6 strength gate. Retain the successful CPU engine and the rejected challenger evidence. The preferred final backend must follow measured quality, latency and cost for the intended use case.

## Remaining work for a finished strong analyzer

| Area | Current state | Required work or decision |
| --- | --- | --- |
| Rules and scope | Verified single-round profile; Betawi target confirmed | Clearly display the precise supported profile and payoff; any local variation requires its own versioned contract |
| Policy, PUCT and self-play learning | One recoverable generation-to-training cycle completed; CUDA/PUCT strength gates failed | Measure native PUCT with both actual exports against both baselines in a new frozen experiment before changing capacity; improve ordinary decisions and search efficiency |
| Evaluation cost | Native adapter faster than CUDA at batches 1–32; relay inputs still expensive; CUDA warm PUCT completes about 100 backups versus 869 native at 100 ms on current H8 cases | Test cheaper features, reusable buffers/ABI, distillation or bounded caching with actual whole-call timing and strength. Native telemetry currently performs an extra feature pass; the timings compare implemented adapters |
| Training data and retention | Replay admission, protected exclusions, unit weighting and deterministic cap implemented; 12 admitted families/623 rows, including a two-row rare stratum | Improve diversity and predeclare sampling/rehearsal ablations. Keep original returns and protected evidence. The broader 100,000-position campaign is unfinished; growth must follow pilot evidence |
| Generalization and strength | H8 CPU candidate's prior development gates pass; both Phase 6 CUDA challengers fail | Validate a chosen backend on fresh families, diversify opponents and time budgets, freeze the accepted release, then characterize it once on the sealed release test without using those results to tune that release |
| Exact-memory retrieval in the main product | K=4 database and explicit lookup work; ordinary search intentionally uses no oracle | Add an explicit, versioned analysis mode that routes covered states to verified facts and labels heuristic results correctly. Any interior lookup changes must declare the recurrence and coverage |
| Unified analysis interface | Working CLI analysis, basic trace-derived explanations and separate experimental backends | Consolidate backend/model selection, ranked moves, continuation traces, estimate/proof labels and cancellation/resume behavior. A graphical board editor is an optional client, not a correctness prerequisite |
| Release and rollback | Immutable experimental models and evidence retained | Select the accepted default from evidence, retain rollback, package reproducible setup and document supported hardware/resource limits. Current CUDA setup reuses a machine-specific existing environment |
| Efficiency claims | Timing and work are measured; energy is not | Measure the relevant component or whole-machine scope before claiming lower power cost or quality per joule |

Correctness and existing recovery are foundations already achieved. Fresh validation, integration and product release remain work even if a new search algorithm succeeds. Larger models, NNUE, larger exact tables and prioritized sampling are candidate experiments; none is automatically required just because another engine uses it.

## Later solving goals

Phase 7 adds verified bounds, proof certificates and resumable solver frontiers for selected positions. Phase 8 investigates whether larger-domain solving is feasible within measured resources. Neither learned predictions nor a winning arena result proves the initial position solved or establishes exact play from every legal board.

Those later proof goals are outside the completion requirement for the first strong position analyzer. Larger regional profiles, refill matches, simultaneous openings and additional user interfaces also require their own declared scope.

Governing references: [architecture and stages](CONGKLAK_ENGINE_PLAN.md), [data/training/replay contracts](DATA_AND_TRAINING_PLAN.md), and [first rules/analysis scope](FIRST_IMPLEMENTATION.md).
