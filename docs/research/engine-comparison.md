# Engine architecture research for Indonesian Congklak and Dakon

Research date: **2026-10-14 (Asia/Bangkok)**. Scope: architecture and an experiment plan; no engine implementation. User hardware: **Intel i9-13900K, 32 GB RAM, RTX 4080 with 16 GB VRAM**. First product goal: analyze arbitrary positions and explain good moves. Overnight work must support interruption and resumption.

This is the research options record. The [first implementation scope](../FIRST_IMPLEMENTATION.md) selects the initial experimental profile, no-oracle full-width CPU CLI and inline persistence; options here are follow-ons unless that scope explicitly includes them.

## Recommendation

Build a Congklak-specific rules kernel and an iterative CPU alpha-beta/PVS analyzer first. Add collision-checked exact endgame results and mathematically valid score bounds. Keep a separate interface for learned evaluation and batched policy/value search so that the RTX 4080 can be evaluated later. Begin learning with labels from the baseline and exact solver; compare a tiny dense network against an incremental count-feature network before adopting NNUE. Treat full AlphaZero-style self-play as an optional measured experiment.

This recommendation is an **engineering inference**, not a measured Congklak benchmark. Reasons: the chosen goal needs principal variations, counterexamples to inferior moves, and honest uncertainty; a deterministic rules simulator can be built; pit selection has a small action space; data and proof infrastructure do not yet exist. A GPU is available, so neural experiments are practical, but that does not establish that they consume less energy or give better answers than CPU search.

The D1 opening witness is an optional matched-profile regression; see the [rules report](congklak-rules.md) for its scope. It is not a prerequisite for the initial analyzer or learning pilot.

## Evidence boundaries and reproducibility

Verified facts below come from authors' papers, official engine code, or developers' statements. Proposed adaptations are explicitly recommendations. Proprietary search details are left unknown. Rankings in chess are not Congklak comparisons. Source snapshots:

| Project | Inspected snapshot | What was checked |
|---|---|---|
| Stockfish | [5c4cef0bfe61fb3249178b2bcf21f5673557efe7](https://github.com/official-stockfish/Stockfish/commit/5c4cef0bfe61fb3249178b2bcf21f5673557efe7), commit 2026-10-13 | Search, TT, NNUE architecture, accumulator source |
| Stockfish trainer | [b5f65bfb38ea3e1c14ccf1c0f286db0ad2c96c24](https://github.com/official-stockfish/nnue-pytorch/commit/b5f65bfb38ea3e1c14ccf1c0f286db0ad2c96c24), commit 2026-10-11 | Official NNUE documentation and trainer identity |
| Lc0 engine | [dab9898f33f63fa58c567daf0b33994d45d12135](https://github.com/LeelaChessZero/lc0/commit/dab9898f33f63fa58c567daf0b33994d45d12135), commit 2026-10-09 | Repository identity; architecture facts use official documentation |
| Lc0 website | [a99d59c497bf6a81151d0d97b06a67d7f6c5722c](https://github.com/LeelaChessZero/lczero.org/commit/a99d59c497bf6a81151d0d97b06a67d7f6c5722c), commit 2026-10-09 | Documentation snapshot; overview last updated 2025-10-05 |
| Berserk | [32628515050b83805bab4afa1026dd2bcaa93f55](https://github.com/jhonnold/berserk/commit/32628515050b83805bab4afa1026dd2bcaa93f55) | Developer README's search/evaluation description |
| Ethereal | [0e47e9b67f345c75eb965d9fb3e2493b6a11d09a](https://github.com/AndyGrant/Ethereal/commit/0e47e9b67f345c75eb965d9fb3e2493b6a11d09a) | Developer README's framework/testing description |
| OpenSpiel | [48401890ee9857e611678302371378175a8e4c6b](https://github.com/google-deepmind/open_spiel/commit/48401890ee9857e611678302371378175a8e4c6b) | Environment/interface description |

All web sources were retrieved on 2026-10-14. Article versions and release dates are identified below. Moving documentation pages can change; recheck before implementing their details. Commit IDs above were read from the official GitHub API; exact file links below use pinned commits.

## Architecture comparison

| Family | Verified search/evaluation | Reusable idea for Congklak | Principal limitation | Proposed role |
|---|---|---|---|---|
| Stockfish | Iterative selective alpha-beta/PVS, NNUE, TT, chess tablebases | Fast CPU search; move ordering; bounded-memory caching; incremental evaluation | Chess move generation/features and pruning assumptions do not transfer directly | Main architectural inspiration |
| Original AlphaZero, 2018 | Neural policy/value, PUCT-style MCTS, self-play policy iteration | Discover long-range strategy and train policy priors | Training overhead; finite search is an estimate, not proof | Later comparison experiment |
| Leela Chess Zero (interpreting "lyla" as Leela) | Neural MCTS/PUCT; multiple search/backends; modern transformer networks | Separate search from inference; batch GPU work; outcome heads | Chess input/history/attention biases and large networks need redesign | Inference and self-play design inspiration |
| Komodo Dragon 3.3 | Publicly disclosed NNUE, standard and MCTS modes, CPU operation, tablebases | Compare several search modes using one evaluator | Proprietary internals; no reproducible Congklak implementation to import | Conceptual comparison |
| Torch | Original proprietary engine; developers disclose NNUE tooling and testing tools | Rigorous experimentation and compact neural evaluation as candidates | Public announcement does not document complete search/network architecture | No implementation dependency |
| Berserk/Ethereal | Public alpha-beta-family engines with neural evaluation | Smaller readable references and controlled engine testing | Chess-specific heuristics remain unsafe to transplant | Secondary code references |
| Exact mancala solvers | Kalah: MTD(f), TT and databases; Awari: parallel retrograde analysis | Combinatorial ranking, exact cache checks, endgame reuse | Different capture/relay/termination rules and state spaces | Strong source for solver design |

The rows summarize the evidence detailed and linked in the next sections. No row claims a speed, strength, or energy result for our game.

### Stockfish search architecture and NNUE

At the pinned revision, `search.cpp` implements iterative deepening, aspiration searches, principal-variation/null-window searches, quiescence, history-based ordering, reductions, and selective pruning including null-move-related logic. Thread workers share infrastructure. These form a heavily chess-tuned system, rather than a generic exact minimax implementation. [Pinned search source](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/search.cpp).

The NNUE architecture has chess feature families `K32Q2`, `FullThreats`, and `PP_3Wide`; declared dimensions include 1024, 32, 32 and 32 layer stacks. These are source constants, not a complete equivalent layer diagram. Current master should not be described with an older HalfKP or big/small-network diagram. [Pinned architecture](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/nnue/nnue_architecture.h).

The accumulator code supports incremental updates and refresh paths. Official NNUE documentation explains how sparse feature changes avoid recalculating the expensive initial transform and how quantized arithmetic enables CPU execution. This is the transferable principle; chess's king/piece/threat encodings and trained weights are not Congklak features. [Pinned accumulator source](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/nnue/nnue_accumulator.cpp), [official NNUE explanation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).

Stockfish's TT stores move, score/evaluation, depth, bound and age-related information in compact entries. Its inspected implementation uses a 16-bit stored key fragment and replacement policies. That is useful playing-engine engineering, but a partial-key hit alone cannot certify a mathematical solution. [Pinned TT implementation](https://github.com/official-stockfish/Stockfish/blob/5c4cef0bfe61fb3249178b2bcf21f5673557efe7/src/tt.cpp).

**Adaptation:** retain iterative deepening, complete legal move coverage, PVS with required re-search, ordering and sound alpha-beta bounds. Initially disable speculative null-move, futility, late-move pruning, ProbCut and unverified reductions. Congklak may punish giving up a move; a forced-move state may invalidate chess-style stand-pat evaluation. Captures and extra turns can make an apparently harmless low-ranked move decisive. Later playing-only heuristics must be tested separately and never silently certify proof.

### Original AlphaZero published architecture

The 2018 paper describes a policy/value network trained from self-play; MCTS uses policy priors and averaged values, and root visit distributions train the policy. Its trunk has a convolutional stem and 19 residual blocks with 256-filter convolutions. The reported training searches used 800 simulations. Loss combines value error, policy cross-entropy and regularization. This is the original published method, not current Lc0. [2018 paper and supplementary methods, pp. 2–4, 17–19](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf).

**Adaptation:** a Congklak policy can output one logit per legal pit, with illegal pits masked. Start with a small fully connected model over counts and rule-state features; compare a small topology-aware model only if needed. A 2-D chess convolutional trunk is not justified merely by the existence of two rows. Store distance, opposite-pit relations and sowing direction have more direct meaning than arbitrary visual adjacency.

For position analysis, use a seed-margin head if the task is to maximize margin, and a separate win/draw/loss head if the task is to maximize game outcome. Two moves that both force wins can differ substantially in seed margin. Self-play outcomes also reflect the current opponents and search budgets; they are not labels of perfect play. Do not introduce discounting or arbitrary move caps into exact-value claims: either changes the game's objective or termination semantics.

### Leela Chess Zero search and inference

Official documentation describes an engine split between search algorithms and inference backends. Since v0.32 it supports multiple searches, including classic MCTS, a DAG preview, policy-only and value-only modes, plus `lc3` development. Its typical heads predict outcome, move policy and moves remaining. GPU backends batch positions. Its ecosystem separately manages clients, uploaded games and training servers. [Official overview](https://lczero.org/dev/overview/).

The official primer discusses PUCT, first-play urgency, and converting win/draw/loss probabilities to a scalar utility. Current transformer work differs from the historical residual/SE description still found on older backend pages. The project's transformer article now points to its ICLR 2026 Chessformer publication as the preferred architecture reference; detailed 64-square chess attention biases should not be assumed useful for pits. [PUCT primer](https://lczero.org/dev/lc0/search/alphazero/), [transformer research notice](https://lczero.org/blog/2024/02/transformer-progress/).

**Adaptation:** use a local versioned self-play coordinator, replay store, training worker, model registry and evaluator service. Do not reproduce distributed internet volunteer infrastructure for one machine. GPU batching should combine independent games or roots when necessary. Large batches can increase throughput while delaying a single analysis query or using stale tree selections. Benchmark batch size and maximum wait separately; expose immediate stop to the user. Reusing a cached neural prediction is simpler than merging full MCTS statistics into a transposition graph; graph sharing needs explicit backup, edge-statistic and cycle semantics.

### Dragon public architecture disclosures

Dragon 3.3's developers disclose NNUE trained using their evaluation and generated positions, later reinforcement learning, standard and MCTS modes, multi-core/tablebase support, and AVX2 CPU execution without requiring a GPU. They describe the modes as serving different practical purposes. The site identifies Dragon 3.3, released October 2023, as its final production iteration and reports sales ended May 31, 2026. [Dragon 3.3 developer page](https://komodochess.com/dragon-3.3.htm), [project announcement](https://www.komodochess.com/).

Precise layer sizes, policy generation, internal MCTS/alpha-beta coupling, pruning schedules and training recipes are **not established here**. Its developer descriptions of human-oriented MCTS should not be generalized into a theorem that all MCTS assumes weak opponents. Congklak's perfect-play analyzer should optimize the chosen adversarial objective.

### Torch and other CPU engines

Chess.com's 2023 developer announcement identifies Torch as an original engine with a team drawn from Ethereal, Koivisto, Berserk and Dragon, and names PyTorch NNUE tooling, Cutechess and OpenBench among development tools. It does not publish the complete network or search implementation. Tool use is not enough evidence for exact layer/search claims. [Torch team's announcement](https://www.chess.com/news/view/torch-chess-engine).

Berserk's pinned README documents PVS/negamax, iterative deepening, TT, quiescence, several selective search methods, and a neural evaluator. Ethereal's author describes alpha-beta with neural evaluation and an OpenBench testing workflow. These are useful readable references for the broad CPU design family, with no demonstrated Congklak advantage over a fresh implementation. [Berserk README](https://github.com/jhonnold/berserk/blob/32628515050b83805bab4afa1026dd2bcaa93f55/README.md), [Ethereal README](https://github.com/AndyGrant/Ethereal/blob/0e47e9b67f345c75eb965d9fb3e2493b6a11d09a/README.md).

Before importing code, record each file's license and dependency license; architectural inspiration does not require copying chess source. The recommendation does not depend on buying Dragon, Torch or chess neural weights.

## Algorithms beyond these engines

The proposed order for experiments is below. The detailed sources and constraints follow; priority means usefulness for the current product goal, not a claim of universal algorithm superiority.

| Priority | Candidate | Why test it | Admission requirement |
|---|---|---|---|
| 1 | Small-state forward solving, retrograde/SCC DP | Exact answers, reusable endgames and reference labels | Frozen rules, complete graph semantics, measured state counts |
| 2 | MTD(f) | Reuses the CPU baseline with integer bounds and TT | Same objective/horizon and correctness as PVS |
| 3 | Memory-bounded CPU search/checkpointed frontier | Predictable RAM and overnight recovery | Eviction may lose speed, never establish a result |
| 4 | DFPN threshold proof | Focuses effort on a win or margin proposition | Cycle/history handling and verifiable completed proof |
| 5 | Proof-assisted or score-bounded MCTS | Combines exploratory search with sound solved subtrees | Separate exact bounds from statistical averages |
| 6 | Expert Iteration and efficient self-play | Compact evaluator/policy from existing search | Count total learning energy; hold out exact test states |
| 7 | Gumbel planning | Improves learning when small PUCT budgets omit actions | Evidence that root-action coverage is actually a problem |

### MTD(f) search

Plaat's algorithm repeatedly drives a memory-enhanced alpha-beta search using narrow integer windows. Stored upper/lower bounds prevent each call from repeating all prior work; a good initial value guess matters. [Author's algorithm description](https://people.csail.mit.edu/plaat/mtdf).

**Proposal:** benchmark PVS against MTD(f) with exactly the same simulator, move ordering, evaluator, TT budget and terminal objective. Integer seed scores and substantial transpositions are promising conditions. Test score oscillation, first-guess error and TT eviction. MTD(f) has extra re-searches; its benefit is conditional, not guaranteed. Keep the simpler PVS as a reference until evidence favors switching.

### Exact mancala databases

Irving, Donkers and Uiterwijk's 2000 Kalah solver combines MTD(f), ordering, TT and endgame databases. It checks full encoded state to prevent hash errors and prunes using a provable remaining-seed score range. Its database construction uses forward minimax over ranked configurations, avoiding costly reverse moves. Its reported variants extend to six pits and five starting seeds; these are not relay-sowing Congklak solutions. [Solving Kalah, pp. 139–146](https://graphics.stanford.edu/~irving/papers/irving2000_kalah.pdf).

Romein and Bal's Awari result computes exact outcomes through large parallel retrograde analysis. The ICGA publication also contains a contemporaneous warning about its precise rule variant. This supports graph/database methods and the need for rule identity; it cannot predict the cost or result of Dakon. [Authors' solution and rule discussion](https://icga.org/icga/journal/contents/content25-3.htm), [institutional publication record](https://research.vu.nl/en/publications/solving-the-game-of-awari-using-parallel-retrograde-analysis/).

**Proposal:** rank small active-seed states and create exact, checksummed tables. A layer based on remaining active seeds is only an ordering aid if some moves preserve that number. Resolve same-layer edges with a correct graph/forward-search procedure. Do not assume a strictly decreasing DAG just because seeds eventually reach stores. If cycles are possible, use SCC/attractor analysis under the frozen repetition/termination rules. Avoid attempting a complete initial-state tablebase on 32 GB before state-count measurements.

### Proof-number and depth-first proof-number search

DFPN tracks proof/disproof effort and local thresholds on AND/OR search graphs. Research shows both graph-history correctness problems and completeness problems with naive versions; threshold controlling variants address specific completeness issues. [IJCAI 2022 paper, sections 2–3](https://www.ijcai.org/proceedings/2022/0658.pdf), [Kishimoto and Müller's GHI research](https://www.sciencedirect.com/science/article/pii/S0020025504002749).

**Proposal:** use DFPN experimentally for propositions such as "the root player can force final margin at least T." Root-player decision nodes are OR nodes and opponent decision nodes are AND nodes, including consecutive turns by the same player. A completed threshold proof can establish a bound; repeated thresholds can establish exact integer margin. A timeout establishes no disproof. Require collision-checked persistent states, explicit cycles/history treatment and independent certificate verification. Learned values may order expansions but cannot turn unknown leaves into proved outcomes.

### Proof-assisted MCTS and bounded memory

PN-MCTS incorporates proof/disproof information into MCTS selection, final choices and solving subtrees; its experiments include Awari. Later GPN-MCTS combines proof-number guidance with score bounds, which is closer to a seed-margin objective than WDL alone. These papers report results in their tested domains, without a Congklak performance result. [PN-MCTS authors' paper](https://arxiv.org/abs/2303.09449), [GPN-MCTS, arXiv:2506.13249v1, 2025](https://arxiv.org/html/2506.13249).

**Proposal:** first attach exact tablebase/terminal values and sound intervals to an ordinary PUCT implementation. Statistical Q values guide exploration; interval propagation follows minimax with the actual controller. Prune only when an interval comparison proves dominance. Test proof-number biases later, including their bookkeeping overhead. This gives a simpler baseline than immediately implementing every hybrid feature.

**Memory proposal:** depth-first iterative CPU search with a fixed-size TT has a predictable RAM target; recompute evicted states. For larger proof jobs, spill a checksummed full-state frontier to disk and retain dependency/certificate information. Put explicit limits on an MCTS tree; restarting an estimated tree is permitted, but do not pretend an evicted unknown subtree was disproved. Resource caps preserve an unknown result when insufficient work has completed. Test disk-backed lookup latency before assuming more retrieval always helps.

### Expert Iteration and efficient learning

Expert Iteration separates planning by a search expert from generalization by a learned apprentice, which then guides the expert. [Anthony, Tian and Barber, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html). **Proposal:** train on deeper CPU searches plus exact endgames, recording teacher budget and proof status; test whether a compact policy improves ordering or enables cheap direct-play decisions. This avoids a requirement to discover everything from random self-play.

KataGo's research introduces randomized search caps, pruning exploration artifacts from policy targets, global information, and auxiliary score/ownership targets. Its efficiency results concern Go. [Wu, arXiv:1902.10565v5, 2020](https://arxiv.org/html/1902.10565). **Proposal:** try search-cap randomization and an auxiliary seed-margin target; pit-based credit-assignment targets need a carefully defined meaning because seeds move between pits. Do not transplant the paper's speedup multiplier.

Gumbel AlphaZero addresses weak policy improvement when a small search omits root actions, using sampling without replacement and associated planning methods. DeepMind's `mctx` repository includes a policy-improvement demonstration. [Danihelka et al., ICLR 2022, authors' paper](https://openreview.net/pdf?id=bERaNdoegnO), [pinned official demonstration, 428bfb7e](https://github.com/google-deepmind/mctx/blob/428bfb7e1c931715cd3d74f9c65e9991afd86df8/examples/policy_improvement_demo.py). **Proposal:** consider it only if low-budget PUCT learning misses moves. With at most a handful of pit choices, explicitly visiting every legal root action may already solve that practical problem. OpenReview full-text access failed during this research; its indexed author-paper abstract and inspected official code establish the broad method, not a fully audited Congklak implementation recipe.

MuZero's learned dynamics are unnecessary for the initial architecture because known deterministic rules permit an exact simulator; learned transition errors would obstruct explanations and proof. Pure model-free RL can be a cheap runtime policy experiment, but its training cost and adversarial reliability must beat the teacher/apprentice baseline. CFR/equilibrium solvers become relevant only if the actual rule profile has simultaneous or imperfect-information choices; ordinary sequential minimax must not silently replace such a rule.

OpenSpiel supports sequential/simultaneous and perfect/imperfect-information game interfaces with C++/Python components. [Pinned official README](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/README.md). **Proposal:** use it as an optional research adapter/cross-check, while retaining our faster specialized kernel. Verify any provided mancala environment's exact rules before reuse.

## Congklak simulation and value semantics

The following are proposed correctness requirements, conditional on the frozen rules report.

1. **One search edge means one player choice.** Complete deterministic sowing, forced relay pickup and capture automatically inside `apply_choice`. Branch again only when a player has a real choice. Keep the seed-by-seed implementation as an independent reference and as the explanation trace generator.
2. **Extra turns preserve perspective.** For a current-player value V, back up `+V(child)` when the controller remains the same and `-V(child)` when it changes. Alpha-beta windows invert only in the second case. In PUCT, keep Q in the parent's decision-maker perspective. Alternatively use a fixed-root value and explicit maximizing/minimizing actors throughout. Never blindly negate once per edge.
3. **Forced transitions still need termination correctness.** A simulator must not hang on a relay cycle or label an arbitrary safety cap as an official draw. Represent a detectable cycle according to the actual rule semantics; a guard-triggered run is an error/unknown result until those semantics are defined.
4. **Optimize after differential validation.** A sowing pass can sometimes be compressed with quotient/remainder over its legal circuit, but store skipping, source revisits, landing tests, captures and relay order must match the reference. Benchmark copy/apply of a compact count array versus make/unmake with an undo record. A tiny board does not automatically favor either method.
5. **Depth is an explicit budget.** Count player choices, complete turns, transitions and relay work separately. The terminal game value is independent of that measurement; truncated analysis is not. Bound every search by both time/cancellation and work counters.
6. **Only prove real bounds.** In a single round with permanently banked stores, margin d and R unbanked seeds imply a conservative final-margin interval `[d-R, d+R]`, subject to the selected terminal scoring. Refill/multiple-round rules invalidate that argument. Apply tighter bounds only after a proof and exhaustive small-state check.

## NNUE versus a tiny dense evaluator

**Proposed candidate representations:** pit counts, store counts or justified score offset, current controller, legal-pit mask, active-seed total, round/phase, blocked pits and relevant historical rule state. Optional derived features include opposite-pit vulnerability, distance to store and exact next-choice consequences. The network must not depend on hidden information unavailable to the player.

A count-based incremental model can use `accumulator = bias + sum E(pit, count)`. Changing k pits removes k old embeddings and adds k new ones. With P pit/store features, an incremental refresh performs approximately `2k` vector additions versus P for rebuilding the sum. Relay moves can alter most counts, so incremental updates may do more work than a refresh. Use a refresh fallback and measure the distribution of k; this is an operation-count inference, not a speed benchmark.

Compare a linear evaluator, a tiny dense MLP over normalized counts, and count-embedding/NNUE candidates with widths such as 64/128/256. Full exact count categories preserve discontinuities; coarse buckets require justification near capture/extra-turn boundaries. Benchmark evaluator latency, cache footprint, end-to-end search accuracy and joules, including feature extraction. An inference model with excellent held-out loss can still order the decisive move badly.

Training should mix reachable self-play states, curated tactical/extra-turn examples and exact small states. Split by games or transposition components, rather than allowing identical positions in train and test. Label exact results separately from finite-depth teacher estimates and self-play outcomes. Save rules IDs, target objective, generator revision, model revision and evaluator scale with every dataset/model.

## Memory retrieval and resumption

| Memory | Key and payload | Can affect proof? | Persistence |
|---|---|---|---|
| Search TT | Full or hashed state; depth/horizon, bound, score, best move, evaluator/search generation | Only if equality and validity are verified; heuristic entries stay heuristic | Optional versioned snapshot; eviction is allowed |
| Neural evaluation cache | Exact state/rule encoding + model hash + value perspective; policy/value output | No: prediction remains an estimate | Bounded RAM; optional reusable disk cache |
| Exact tablebase | Collision-free rank or collision-checked full state; objective and rules; exact value/best moves | Yes, after validated construction | Checksummed, memory-mapped files with manifest |
| Proof cache/frontier | Full state, threshold, sound bounds, dependencies/history obligations, proved/unknown status | Yes, after verification | Durable job records and atomic checkpoints |
| Replay dataset | Full state, legal choices, teacher/search visits, target outcome/margin and provenance | No by itself | Append-only compressed batches with index |
| Similarity/strategy retrieval | Approximate embedding and explanatory exemplars | No: ordering/explanation suggestions only | Optional later feature |

A state key includes every item affecting legal actions, transitions or final payoff: all counts, controller, phase, variant/rules version, blocked/refill/match state if relevant, and history information where repetition outcomes depend on it. A 64- or 128-bit hash is still probabilistic equality; proofs require full verification or collision-free ranking. An entry marked "exact" by a depth-limited search means exact for that search horizon, not necessarily exact for the full game.

For a proven single-round model, stores may be factored into an additive score offset if they never affect future legal play or termination. Validate that reduction before omitting them. Multi-round refill makes store holdings strategically active. Only canonicalize a symmetry after demonstrating that it preserves sowing direction, legal moves and payoff, with reversible move mapping.

Checkpoint a complete analysis result at each finished iteration: input state, rules, binary/revision, model hash, limits, scores/bounds, principal variations and work counts. Cancellation returns the last complete iteration. Persist root work queues and proof/tablebase progress for overnight jobs; use atomic replacement and checksums. TT restoration is a speed aid and does not resume the exact scheduling of running threads. Incompatible revisions/rules/models reject or downgrade old cache entries. Test interruption/restart by comparing final values against uninterrupted small exact jobs.

## Hardware and energy experiment gates

The 2018 AlphaZero study used 5,000 first-generation TPUs for self-play and 16 second-generation TPUs for training; its chess training lasted about nine hours. Match hardware used four first-generation TPUs and 44 CPU cores for AlphaZero versus 44 CPU cores for Stockfish. The preliminary 2017 report's 64 training TPUs describe a different version. Historical node counts do not imply lower total energy, and those runs do not forecast Congklak training. [2018 study, pp. 4–5](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf).

**Proposed benchmark sequence on the supplied machine:**

| Gate | Compare | Required measurements | Decision |
|---|---|---|---|
| A: kernel | Reference vs optimized sowing; copy/apply vs undo | Exact state agreement, choices/s, relay steps, worst-case latency | Optimize only while retaining agreement |
| B: CPU search | Plain alpha-beta, PVS, MTD(f); TT sizes/thread settings | Solved tactical/endgame accuracy, elapsed time, nodes, cache hit rate, resident RAM | Retain best time/quality tradeoff |
| C: evaluator | Linear, tiny dense MLP, count NNUE | Eval latency, search regret on exact positions, joules/query, k-changed histogram | Learning survives only if useful end-to-end |
| D: policy search | CPU analyzer vs GPU PUCT; batches 1 through practical sizes | Fixed time and equal-energy strength, exact-move agreement, p50/p95 latency, VRAM | Adopt PUCT only if it wins on target workload |
| E: learning | Teacher/apprentice vs self-play; randomized caps; optional Gumbel | Total training/self-play energy, independent opponents, train/test leakage, tactical misses | Keep smallest reproducible improvement |
| F: durable solve | Exact forward/DP/tablebase and optional DFPN | Verified bounds/certificates, restart equality, disk/RAM use, time-to-proof | Increase solved domain within a measured budget |

Use equal wall-time and equal-energy comparisons, not just equal node counts. A CPU node and a GPU neural evaluation perform different amounts of work. Tune CPU thread count/affinity and GPU batch timeout empirically; using every logical thread or maximum GPU power is not automatically optimal. Large independent self-play batches suit the GPU better than a single narrow tree, but measure this on actual positions.

Report whole-machine energy from a meter when available; otherwise report the actual scope of CPU/GPU telemetry and avoid calling it total system energy. Account for warm-up, idle baseline, CPU simulation, GPU inference and training. Compare total joules for N analyses as `training/data-generation energy + sum query energy`; do not hide learning cost when claiming low power. Check that RAM/VRAM limits leave room for the OS, simulator, replay pipeline and model; grow endgame coverage from measured ranked state counts.

Explanation output should cite the selected action, concrete sow/capture/extra-turn trace, principal line, strongest reply and comparison with alternatives. Label scores **estimated**, **sound lower/upper bound**, or **proven**, and identify the objective and rules. Multiple candidate lines cost more search; use a configurable analysis budget. Fluent generated prose must not invent a tactical reason unsupported by the simulator/search evidence.

## Related Congklak research

Darmawan and Hermawan's 2016 Congklak paper reports faster NegaScout search but higher win percentages for its Minimax comparison. It is a small implementation study, not a universal speedup claim. Correct full-width PVS and alpha-beta with identical horizons/objectives should agree; differences motivate examining implementation, stopping budgets, evaluation and test conditions before importing conclusions. [Author paper](https://ojs.unikom.ac.id/index.php/komputa/article/view/2462), [full PDF](https://ojs.unikom.ac.id/index.php/komputa/article/download/2462/1638/5988).

A 2026 UPI undergraduate thesis implements AlphaZero-based AlphaDDA for adaptive difficulty. The institutional abstract reports an internally measured 1729.31 Elo after 50 round-robin tournaments, competitive performance against its Minimax agent, and dynamic changes to MCTS budgets. This is relevant prior work, not proof, an externally calibrated skill rating, or a benchmark for our rules/hardware. The accessible indexed abstract was read; the complete implementation/evaluation has not been audited. [Institutional record](https://repository.upi.edu/155169/).

## Immediate architectural decision

Choose **a purpose-built CPU analyzer with a shared exact simulator, explicit result bounds, verified endgames and resumable jobs**, plus an interchangeable learned-evaluation/policy-search interface. Use Stockfish for ideas about fast CPU search, Lc0 for search/inference separation, and mancala solving research for exact state storage and databases. Add small neural evaluation and GPU self-play only after the baseline produces trustworthy datasets and the benchmark gates show a practical benefit. This is the most testable path to the user's current arbitrary-position analysis goal without committing to unnecessary training or claiming that strong play has solved the game.
