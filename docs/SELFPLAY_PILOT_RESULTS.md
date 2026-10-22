# Self-play and joint-training pilot

The complete pilot is recoverable and audited, but **it did not establish a strength improvement**. The challenger gains +0.25 seed margin against H_v1 relative to the contemporaneous control, and 0 against H8 relay; both paired confidence intervals include zero. **Keep H_v1 as default and retain the current control as the reference. No model promotion occurred.**

## Frozen experiment and replay

The existing compact control (`bc1160e...`) generated all self-play; the updated model never generated its own training games. Sixteen fixed training starts covered four each of 50-74 ordinary, 25-49 ordinary, bonus and capture. Two trajectories per family used native CPU PUCT, 64 simulations, c_puct=0.25, a2048-node bound, Dirichlet root noise and the frozen behavior-temperature schedule. Actual production completed **32 terminal trajectories, 1,029 decisions and 65,856 tree simulations**, with no capped games.

Before production, the first eligibility fixture exposed protected-state collisions at 5-11 active seeds. The approved v2 scope restricts optimizer candidates to **at least25 active seeds**, omitting later states before the unchanged evaluation mask and whole-family collision ban. The same16 starts, 32-trajectory limit, protection ledger and minimum-eight-family gate remained; there was no outcome-driven replacement or weakened evaluation protection. Original preflight evidence and the original scope are retained.

Production admitted all 16 families: **276 replay observations representing 231 unique raw states**. It excluded306 observations at active <=4 and 447 at 5-24 from optimizer candidates; full decision histories remain retained. No eligible protected bridge occurred, and the four-observations-per-raw-state cap removed no rows. All prior 2,839 artifact pins remain unchanged. Known evaluation trajectories, the reserved benchmark starts, latest10 midgame validation roots and nine diagnostics were protected before generation/training.

The predeclared sampler chooses a family uniformly, then a retained observation uniformly, with unit loss. Equal starting quotas do not balance later replay mechanisms:

| Exposure | Expected mass | Reconstructed draws /25,600 |
|---|---:|---:|
| All ordinary states | 10.26% | 2,583 |
| All50-74 states | 3.33% | 853 |
| 50-74 ordinary alone | 2.00% | 504 |

The reconstruction uses the accepted seed/sampler and independently verified checkpoint RNG states; draw IDs were not separately logged. Inverse probability concentration is 225.824 over retained observations and 16 over families, not independent games or gradient ESS. [Full sampling review](../build/selfplay-pilot-sampling-review.json). No post-outcome sampling or loss change was applied.

## Training and measured strength

All eight tensors of the `127x64x64` model, 12,872 parameters, trained jointly:200 CPU steps, **batch 128**,25,600 replacement draws, ten checkpoints, one CPU thread. The loss combines signed-active terminal-return residual Huber(delta=0.1) and 0.25 times visit-policy cross entropy. No teacher rehearsal, model expansion or checkpoint selection occurred; final step 200 is the challenger.

On its 276 training observations, value MAE fell 7.8573 to 6.1107 seeds and visit-policy cross entropy 1.1446 to 0.9848. These are in-sample fits to source-policy outcomes and visits, not minimax truth or independent validation. [Training result](../data/selfplay-training-pilot-v1/result.json), [actual checkpoint/export witness](../build/selfplay-training-pilot-model-witness.json).

The frozen benchmark interleaved control and challenger against **both H_v1 and H8 relay CPU** on12 paired families, both seats: **96 terminal games, 2,958 decisions, zero fallbacks**. All engines received25 ms whole-decision budgets on the same declared performance core; physical feature/inference costs and overshoot were retained. Candidate PUCT settings and tree implementation were unchanged.

| Opponent | Control mean margin [95% interval] | Challenger mean margin [95% interval] | Challenger minus control [paired95% interval] |
|---|---:|---:|---:|
| H_v1 | -1.000 [-2.667, 0.750] | -0.750 [-2.333, 0.833] | +0.250 [-1.083, 1.750] |
| H8 relay CPU | -1.667 [-3.667, 0.333] | -1.667 [-3.917, 0.333] | 0.000 [-1.583, 1.833] |

Margins are candidate seed margins, paired across two seats within each of 12 families. The six bootstrap intervals are exploratory and unadjusted; all strength gates are false. Challenger W/D/L is 11/1/12 against each baseline; control is 12/0/12. There was no retuning from these outcomes.

The exact endgame guard retained **32 known-exact K<=4 positions**,25/100 ms, four engines:256 scalar measurements. Control and challenger both selected32/32 exact-optimal moves at each budget, with zero exact regret. Sixteen roots are choice-bearing and 16 forced-move; this small reused guard does not establish general endgame preservation.

Posthoc comparison found **zero actual replay quotient/family collisions** among 2,282 nonterminal>4 arena observations, 1,106 unique quotients. Future arena trajectories could not be known before training. Zero recorded collisions does not erase ancestral/development exposure: the 12 families are new to prior arena IDs but remain consumed development, not a sealed release test. [All benchmark results and overlap ledger](../build/selfplay-pilot-benchmark.json).

## Recovery, cost and readiness

All **20 new methods passed**: four learning-author, eight benchmark-author and eight independent checks. Eight actual process-kill barriers cover self-play decisions, checkpoint file/catalog commits, and benchmark decision/game commits. The audit independently replayed all 200 optimizer updates, all ten checkpoints, terminal returns, visits, raw games, exact measurements, paired intervals and execution-source provenance. [Independent audit](SELFPLAY_PILOT_INDEPENDENT_AUDIT.md).

The [completed no-work proof](../build/selfplay-pilot-root-verification.json) preserves30 durable artifacts byte-for-byte and by modification time, all 2,839 prior pins, and performs zero new searches/optimizer activations with no Torch import. Resume using the identical frozen generation/training/benchmark commands and paths; committed decision prefixes and optimizer/RNG state survive. After acceptance, use `python tools/record-selfplay-pilot.py verify`; renewed author/audit fixtures require separately versioned evidence.

The training result records 2.297 s in its measured invocation scope and about 485 MiB peak working set. The complete benchmark invocation records 173.277 s, about 314 MiB peak working set and 304 MiB peak private commit. Collection/admission outside those measured scopes, separate inspection and audit fixtures add cost; no total energy or GPU-efficiency claim is made. [Benchmark resource scope](../build/selfplay-pilot-benchmark-resources.json).

The dashboard at http://127.0.0.1:18766/ now uses a **new read-only display overlay**, preserving frozen producer/observer/HTML files. It reads nested summary counts, separates initialized slots from terminal games and shows the authoritative batch 128. [Backend proof](../build/selfplay-pilot-observer-proof.json) verifies eight cards and 24 unchanged durable files without search, training, Torch, browser or HTTP inspection; Browser verification was not performed because earlier browser access was denied.

The bounded self-play/training/evaluation pipeline is usable. This one small, capture-heavy replay and inconclusive development test do **not** justify blind large-data generation, longer training, a larger model or a finished-engine release. A further experiment should predeclare ordinary-state coverage and evaluation separation before adding compute.
