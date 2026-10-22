# Congklak database generation and engine training plan

Build verified small endgames and a growing corpus of search-labeled positions, then train a compact evaluator on the RTX 4080. A complete seven-by-seven database is not a prerequisite. The CPU analyzer supplies the first teacher; accepted learned models can improve the next teacher generation. This extension specifies experiments for the confirmed i9-13900K, 32 GB RAM and 16 GB VRAM machine. The K=4 oracle, original pilot, corrected depth-six corpus and supervised CUDA models are now implemented; see [phase-two results](PHASE2_RESULTS.md) and [phase-three results](PHASE3_RESULTS.md). The learned candidates have not passed promotion gates. [Progress reporting](PROGRESS_REPORTING.md) exposes current tool status and remaining improvements.

The [main architecture plan](CONGKLAK_ENGINE_PLAN.md), [first implementation scope](FIRST_IMPLEMENTATION.md) and [recovery contract](research/compute-and-recovery.md) govern correctness, resource limits and resumption. Freeze the complete experimental profile and validate its simulator before producing reusable labels. Matching the user's physical tradition is a separate convention check, not a prerequisite for implementing the declared experimental game. A record's rules digest, objective and state encoding must be identifiable throughout its lifetime.

## Three stores with different jobs

| Store | Generation | Use |
| --- | --- | --- |
| Exact endgame tablebase | Enumerate a declared small-state domain and solve every covered state with an independent recurrence verifier | Perfect lookup inside verified coverage; exact training and benchmark labels |
| Training corpus | Sample legal histories, curate tactical states and label them with a versioned search teacher | Train an evaluator and optional policy; preserve label quality and origin |
| Analysis library | Save user analyses and completed background analyses | Retrieve compatible positions, explanations and reusable results |

A transposition table is a bounded working cache, not the durable training corpus. Similar-position retrieval may suggest educational examples; only an exact compatible state lookup can return an exact value. Learned weights generalize from the corpus but do not make a previously unsolved position proven.

## Generate exact endgames first

For the proposed single-round profile with permanent stores, first prove that future-award margin depends only on pit counts and controller, while existing store difference is an additive offset. That reduction requires the margin objective and the selected pass/end policy; it does not apply automatically to refill games or store-dependent rules. Without the proof, retain stores in the exact state and recalculate the sizing.

Under that conditional reduction, enumerate pit-count compositions by active seeds, with both controllers. Normalize forced control without mistaking a depleted row for a terminal board. Zero-removal moves can remain in the same active-seed layer. Use the validated termination ranking or complete dependency-graph treatment specified in the main plan.

Define the quotient value F in fixed player-0 perspective as future store-0 awards minus future store-1 awards, excluding the existing store difference D. At terminal states F=0. For a normalized player-0 choice, `F(s)=max_a(deltaD(s,a)+F(child))`; player-1 choices use min. Here deltaD is the change in store difference during the complete transition, including captures. Lookup returns full margin `D+F`. Verify `abs(F)<=R`, integer parity `F mod 2 = R mod 2`, immediate award accounting and every tied optimum. Simply propagating terminal zeros without deltaD would lose every captured or banked seed.

Tablebase pit masks describe only the normalized actor's legal choices. Lookup of an empty-row input returns a forced-pass event and redirects to the normalized actor's rank; it never presents that actor's pit mask as choices for the original player. Keep original controller and pass events in the report. Rank-space slots representing unnormalized nonterminal inputs can remain unused and are excluded from the covered normalized domain in the manifest. A valid zero mask is terminal, not a selectable pass. The two-controller figures below remain allocation upper bounds.

The following are calculated allocation counts, not observed reachable states or runtime estimates. For 14 pits and at most K active seeds, the count before eligibility/history fields is `2 * C(14+K,14)`.

| Active seed limit K | Allocations and controllers | Two-byte payload sizing |
| --- | --- | --- |
| 4 | 6,120 | 12,240 bytes |
| 6 | 77,520 | 155,040 bytes |
| 8 | 639,540 | 1,279,080 bytes |
| 10 | 3,922,512 | 7,845,024 bytes |
| 12 | 19,315,400 | 38,630,800 bytes |

The illustrative two-byte entry stores a signed future-award margin and a seven-bit optimal-move mask plus a validity bit. Future margin fits in a signed byte for these limits. A terminal state can have a valid zero move mask; incomplete coverage must remain distinguishable. Coverage metadata, checksums, indexing, construction buffers and graph dependencies add storage and memory. If the profile requires more state or another payoff, this encoding changes.

After the resumable analyzer, build K=4 as a small exact oracle and validation aid. K=6 and K=8 follow measured benefit and construction costs; they are not prerequisites for the 10,000-position pilot or first training experiment. Validate every small covered entry against the reference solver and independently regenerate its legal successors. Increase coverage only after measuring construction memory, elapsed time and benefit. The resulting exact labels include tied optimal moves; the policy loss and evaluation must respect that set.

## Generate a diverse position corpus

Use several sources rather than repeatedly sampling the normal opening:

- Legal histories from random, heuristic and accepted search agents, with diversified legal prefixes and controlled exploration.
- User analysis positions and curated capture, empty-opposite, pass, long-relay and bonus-choice examples.
- Exact endgames and intentionally constructed valid states, tagged separately from states known to occur in generated histories.
- Positions where successive search budgets or accepted models disagree, prioritized for deeper relabeling.

Stratify sampling by active seeds, store difference, legal choices and tactical mechanism. Limit repeated visits to the same canonical state and highly similar consecutive decisions. Preserve a bounded game archive with the dependencies needed by retained examples, but sample fewer decisions from redundant stretches. Do not let a known opening construction dominate the learning curriculum.

Keep reachability status explicit. A syntactically valid counterfactual state can be useful for robustness; it must not be described as observed reachable play. Do not generate arbitrary hand/cursor states and treat them as stable player choices.

Start with a 10,000-position pipeline pilot. Measure validation failures, duplicate rate, source coverage, completed labels per second, bytes per record and teacher stability under increased budgets. Grow to 100,000 unique positions for a first learning comparison; try one million only if scaling continues to improve independent analysis results. These are experiment sizes, not strength guarantees.

## Label positions using the CPU teacher

The first teacher is the accepted full-width CPU search with its declared heuristic evaluator. Exact endgame lookup is an optional later leaf policy with its own immutable table/coverage and exclusion-manifest identities. Terminal checking precedes any oracle lookup; the declared oracle precedes the depth-zero heuristic. Freeze that recurrence in the teacher identity. Give each labeling job a state, objective, teacher revision, horizon policy and work budget. Compare several budgets in the pilot; select production budgets from development accuracy and completed-label throughput rather than a depth number alone.

For each legal root move, store its completed value or its bound, search coverage and controller perspective. A cutoff score is not automatically a complete root score. Generate a scalar teacher value only with the declared horizon semantics; generate a policy target from comparable completed action values, or explicitly mark the policy incomplete. Partial labels can support interval-aware training later, but initially exclude them from ordinary scalar/policy supervision.

Use distinct label kinds:

| Label kind | Meaning | Training treatment |
| --- | --- | --- |
| Exact optimal margin | Independently verified terminal/tablebase/proof value | Strong supervision in its declared domain |
| Search estimate or horizon result | Teacher result under a recorded evaluator and budget | Approximate supervision; retain depth, reply coverage and generation |
| Certified lower or upper bound | Rigorous eventual-outcome constraint | Keep separate; never replace it with a fabricated point target |
| Actual game return | Final margin of a completed generated game | Outcome of those agents, not necessarily optimal play |
| Incomplete game or interrupted search | Unresolved work | Resume or discard the pending target; never synthesize a draw |

Every target declares its player perspective. A completed game's return is the recorded player's final store minus the other store, regardless of how many bonus choices occurred. Never alternate label signs merely by decision index. Side normalization must also map policy actions and interval signs correctly.

For an exact nonterminal choice state, store the complete optimal-action set O and use the tie-set policy loss `-log(sum_a_in_O policy(a))`. This permits concentration on any tied optimal action. Terminal and administrative-pass inputs receive value supervision only until normalized to a choice state; an empty O never enters that policy loss. Uniform cross-entropy over O is a different optional distributional objective and must not be described as tie-indifferent. For a fully scored teacher state, an optional soft policy is `pi(a) proportional to exp((Q_actor(a)-max(Q_actor))/temperature)` over legal moves. Q_actor uses the normalized actor's maximizing perspective: negate raw fixed-player-0 scores for actor 1, or equivalently use softmin on those raw scores. Temperature is measured in score units and is experimental; it does not make approximate scores probabilities of optimality.

Check every target's perspective and physical range `[D-R,D+R]` for the declared single-round margin convention. Exact final margins also satisfy conservation parity. Approximate horizon targets need not have final-margin parity, but eligible scalar targets must fit the declared output range. Quarantine an inconsistent target or use a separately declared proxy objective; do not silently clip stored labels. If a heuristic uses clipping, freeze it in the evaluator before search and record that identity.

Revisit disagreement states with larger budgets. Keep label revisions required by active or frozen datasets rather than silently overwriting evidence; compact unreferenced revisions under the retention policy below. Teacher generations must identify their learned model, if any, and their exact-table coverage.

## Admission and redundant data removal

Apply validation and deduplication before spending a large search budget. The active training corpus is a selected working set, with limits on its size and redundant game segments.

| Incoming data | Action |
| --- | --- |
| Invalid counts, illegal actions, damaged records or unresolved rule/encoding identity | Exclude from training; retain bounded diagnostic examples and reason counts, then remove disposable payloads |
| Exact duplicate state with the same compatible label | Merge provenance and observation frequency into one selected example |
| Verified symmetry equivalent | Canonicalize and remap actions, value perspective and interval signs before deduplication |
| Same state with conflicting exact answers | Quarantine and investigate; averaging cannot repair a contradiction |
| Same state with different search budgets | Keep the selected compatible label and referenced revisions; relabel when needed |
| Different generated-game returns for the same state | Preserve distinct returns or distributions grouped by generator, opponent, model and search-policy identities; never mix them into optimal-value labels |
| Highly redundant neighboring game decisions | Thin within a declared sampling stratum, preserving capture, pass and bonus-chain mechanisms |
| Training row overlapping a holdout | Exclude the training copy while keeping holdout ownership fixed |
| Incomplete job or game | Resume from its committed boundary, or remove abandoned temporary data without making a terminal target |

Canonical deduplication includes the rule contract, objective and every semantic state field. A collision-checked hash can index it. Differences in one pit, controller or eligibility can change the best move, so embedding similarity alone is insufficient to merge positions. Low-quality generated moves can expose useful recovery positions; decide using validity, diversity and label usefulness rather than whether the game was won.

A frozen dataset declares a deterministic label selector. Compatible independently verified exact targets take precedence for optimal-margin supervision. Otherwise select only the designated teacher generation, leaf/oracle policy, perspective, target kind and budget/stability policy; record the chosen revision. Larger budgets and newer models do not automatically prove a better label. Conflicting or unstable estimates request relabeling and remain distinct evidence; they are not averaged into an exact answer. Observation counts are provenance, not an implicit second training multiplier.

## Reliability weights and learning priorities

Give each example separate fields for `label_weight`, `sampling_priority`, `relabel_priority` and `retention_reason`. Sampling a record often does not make its label more reliable. Teacher disagreement primarily requests better labeling; it does not turn an uncertain teacher target into high-confidence supervision.

The first pilot uses admission-filtered exact and complete teacher targets with equal unit loss coefficients, while reporting errors separately by source. Incomplete or incompatible targets receive no ordinary scalar supervision. Completed self-play returns retain a separate outcome objective. A teacher coefficient such as 0.5 is a later controlled ablation selected on development results; it is not a calibrated reliability probability.

Within each active-seed and mechanism stratum, maintain a bounded reservoir of ordinary examples plus examples important for rare tactics or previous development failures. Start with uniform stratified sampling under a frozen intended distribution q. Give each row one primary sampling stratum so the initial baseline has no overlapping-route ambiguity. Later test a mixture such as 30 percent q and 70 percent capped priority r. Priorities can combine validated model error, novelty, rare mechanism coverage and development-set regression; refresh them with a recorded model version. Cap noisy outliers and retain baseline support.

For the simple sampler, draw independently with replacement. Record each selected row's marginal probability P over every route that can select it: `P(i)=sum_c mixture_mass(c)*P(i|c)`. For the optional two-route mixture this is `.3*q(i)+.7*r(i)`, not just the probability on the chosen route. If prioritization preserves the q objective, use q(i)/P(i) once and multiply any source loss coefficient once. Fixed quotas or sampling without replacement need their own correct inclusion-probability treatment before implementation. Freeze q, route masses, priority snapshot/model identity and sampler generator state with each run and checkpoint. Any clipping or deliberately different objective is explicit. Observation frequency and rarity never become undocumented extra multipliers.

Prioritized Experience Replay studies nonuniform replay, diversity and importance correction in DQN. Here it motivates a supervised sampling experiment; its Atari results do not establish a Congklak benefit or a rule for deleting data. [Original prioritized replay paper](https://arxiv.org/html/1511.05952)

Use training or development feedback to compute priorities. The final test set remains fixed and does not select which training examples survive. Run a uniform stratified baseline alongside the proposed weighting policy to measure whether it actually improves analysis per unit of compute.

## Bounded retention and safe compaction

Start with a 100,000-example active corpus and grow toward one million only when the promotion gates justify it. Enforce per-source, per-stratum and per-game limits so common openings cannot fill the budget. Reduce representation of frequently mastered, redundant examples while preserving a reservoir for rehearsal; low present error alone is insufficient to erase a rare mechanism.

Retain current useful labels, exact supervision sampled across endgame strata, rare capture/pass/bonus cases, and representative older examples that catch forgetting. An older exact answer remains exact for its unchanged profile. Evaluate old teacher labels against compatibility and measured usefulness, rather than discarding all of them by age. Relabel strategically useful stale examples; remove superseded unreferenced training rows and excess easy duplicates.

Use the 50 GiB total managed job-storage limit, counting physical files once: live shards, temporary/uncommitted files, fallback and retained generations, SQLite/WAL, managed backups, proof/model dependencies and compact split-exclusion records. Reserve space before writes or compaction for the complete replacement and metadata/recovery overhead; existing and replacement files coexist until commit. Initially reserve at least 2 GiB for bulk operations within that quota and check actual filesystem free space too. If the rewrite cannot fit, pause or compact a smaller safe unit; never delete its live source to manufacture space.

As proposed sublimits, keep unpinned raw/debug game artifacts within 2 GiB and ordinary quarantine samples within 10 MiB. Record reason counts after payload removal; unresolved exact contradictions retain sufficient pinned evidence until investigated. Compact game records contain start state, moves, rule version and final allocation. Generate large animation traces on demand, retaining evidence required by game-return labels, pinned explanations or regressions. Bound individual provenance/event records and aggregate redundant counts; keep only dependency IDs needed for retained data and reproducibility. Permanent minimal split-exclusion records are counted and cannot be discarded to refill training with a former holdout.

Model retention includes the current accepted model, a previous accepted model, the baseline and every checkpoint/model dependency pinned by a retained run or frozen dataset. Delete failed candidates and redundant checkpoints only after their useful measurements are recorded and no retained manifest depends on them. Completed datasets and analysis records have explicit pin or expiry metadata.

Exact tablebase coverage and proof dependencies have a different retention contract from replay. Compression can preserve declared coverage; retiring coverage must atomically update direct and aggregate manifests and affected lookups. Keep transitive proof/result evidence pinned or export and verify a self-contained replacement before removing its source. Otherwise retire dependent certified/proven availability, invalidate certified cache entries and quarantine interrupted solver checkpoints that would use missing evidence. Previously proved mathematical facts do not become false, but unavailable evidence cannot support a current auditable certificate. Saved user analyses and their evidence, golden fixtures, pinned runs and frozen development/final-test sets remain protected. If protected artifacts alone exceed quota, pause new generation and report it.

Compaction writes and validates a new immutable generation, then commits it through the single local metadata writer. The stable reference closure includes current/fallback generations, retained or retired-but-retained datasets, models/teachers, proofs/results, backups, active readers and active-job writer/staging pins registered before file creation. Do not reclaim a temp or final-uncommitted file while its owner can still publish it; establish owner inactivity or quiesce bulk writers. Pin acquisition and retirement are serialized. Fallbacks using removed rows keep their files live until replaced by validated reduced fallbacks.

Commit retirement intent and tombstones, then recheck the closure and delete only unreferenced managed files under the resolved data root. Paths/names are not reused for a different generation. Deletion is idempotent: missing files complete an existing deletion intent, failed deletes remain retryable and counted. Startup uses committed catalog pointers, never auto-promotes orphan manifests or final-named files. Record removal reasons, released bytes and coverage. A restart at any step recovers one complete retained generation rather than resurrecting intentionally retired coverage. The [recovery contract](research/compute-and-recovery.md) specifies these later bulk-artifact requirements.

Before accepting quality-based pruning, compare the retained corpus with the larger uniform baseline using matched training budgets, at least two recorded training seeds and the same development/rehearsal evaluations. Bound relabeling and prioritization work by the campaign's measured time/energy budget and report that overhead. Declare per-stratum tolerances before running the comparison. The first pilot performs validation, exact deduplication and storage compaction only; model-error-based eviction and prioritized weighting require this later ablation. A lower byte count is useful only while important analysis quality is preserved.

## Record format and split isolation

Use SQLite for jobs, manifests, provenance and compact indexes. Store bulk training arrays in immutable binary shards, with packed states and columnar labels; choose the concrete array format after a loader throughput benchmark. Keep long PVs, sow traces and proof dependencies in separate files or tables referenced by ID.

Each training example needs:

- Canonical state bytes, rules and encoding identity, controller convention and legal-action mask.
- Source type, game or position-family ID, generation seed and reachability status.
- Label kind, objective, scalar or interval targets, optional per-action scores and policy.
- Teacher/model revision, budget, completed horizon, opponent-reply coverage and terminal status.
- Dataset generation, split ownership and any weighting or relabeling decision.

Assign generation families to `train`, `development` and `sealed_final_test` before sampling positions. Development is the validation pool used for tuning, promotion, priority design and pruning comparisons. Across all generations, reject an exact or verified symmetry-equivalent sealed-test state and excluded family from training. Keep split ownership after pruning with compact permanent exclusion records. If a later family joins overlapping split components, quarantine the overlap or exclude it; do not move it to improve coverage. Label provenance and observation frequencies remain distinct from split ownership.

A proposed starting split is 80/10/10 by eligible family for train/development/sealed-final-test. These percentages do not promise exact row counts. Divide tactical and exact benchmark families into development and sealed portions too. Final-test results are not used for promotion, hyperparameters, retention or relabeling. Evaluate the already accepted model once for the declared release; if results influence a subsequent revision, retire that suite to development and designate a new sealed suite before the next final report.

For strict oracle isolation, direct training admission and teacher lookup exclusions also cover the proved tablebase quotient: profile, objective, normalized pit counts/controller and any eligibility fields, independent of store offset. Guard every leaf lookup and derived certificate/table cache against all excluded aliases. Cache identity includes coverage and exclusion-manifest digests; unguarded entries cannot supply guarded labels. Track precomputed-value dependencies and teacher/model training ancestry. A new lookup mask cannot erase sealed facts already used by an ancestor model: use an unexposed baseline lineage or label that benchmark oracle-assisted. Register sealed exclusions before generation and preserve them across revisions.

At an illustrative 192 bytes per packed example, one million examples occupy 192 MB before compression, indexes, games and traces. Benchmark real records before adopting that estimate. Corpus creation and label quality may dominate the work even when the stored arrays are small.

## Train a small evaluator and compare challengers

Train a tiny dense count-vector MLP first. Once the pipeline and development evaluation work, compare an NNUE-style count-embedding challenger on the same frozen corpus. Start with small models; test hidden widths of 64, 128 and 256 only when needed. The dense baseline can use two hidden layers. The later embedding candidate compares incremental sums with complete refresh after relay moves.

For the fixed initial profile, inputs include all pit/store counts and current controller. Derived active-seed total and store difference may help. Add eligibility, phase and other semantic state if the selected profile requires them. Normalize count scales without dropping exact distinctions; unknown counts are not ordinary exact inputs. Symmetry and side normalization must transform move labels and targets as well as the board.

Predict final margin first. A proposed residual head uses current-controller store difference D and active seeds R: `estimated_margin = D + R * residual_prediction`. Train the value loss in reconstructed seed-margin units, for example Huber with a recorded seed-unit threshold, so errors of one seed have the same meaning across R. Unweighted normalized-residual loss would instead change the objective by scaling squared seed error by R^-2. Normalized-residual training is a separate declared ablation; squared-error equivalence requires the corresponding R^2 factor. Terminal states use the exact rules result. The residual can be constrained to [-1,1] under the applicable store policy, while its prediction remains heuristic.

Keep exact, teacher and game-return targets separate in the loader. Initial training uses exact and complete teacher labels with uniform stratified batches; add completed self-play returns through a separately measured loss or experiment. Sample across active-seed strata so easy endgames do not overwhelm middle-game decisions. Use seed-unit value loss, optional optimal-set loss for exact policies and legal-masked cross-entropy only for teacher distributions. Source weights and any policy temperature are saved settings.

This teacher/search feedback is inspired by Expert Iteration, which separates planning from learning and then uses the learned policy to guide stronger planning. The Congklak choices here are proposed adaptations. [Original Expert Iteration paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html)

Stockfish's public trainer is specifically for chess evaluation. Build a Congklak state loader, feature encoder, targets and export contract; inspect reusable components and licenses before importing code. [Official NNUE trainer](https://github.com/official-stockfish/nnue-pytorch)

## Local resource and recovery settings

For labeling, start with four independent CPU workers and a 256 MiB table per worker; test eight workers separately. The aggregate table allocation is 1 GiB or 2 GiB, not an unbounded per-worker copy of the main analyzer's budget. Count all concurrent processes toward the existing 24 GiB aggregate RAM cap. Interactive analysis has priority over background generation.

For training, test batches such as 512, 1,024 and 4,096 within the initial 12 GiB GPU cap. Pick from measured throughput and validation behavior. Large VRAM does not imply that a large network is necessary. Begin with staged generation followed by training; overlap them only after queue, resource and version consistency work reliably.

Commit completed labeling work in immutable shards and completed games under unique IDs. Pending jobs can be reissued without duplicating committed records. Training checkpoints follow the full recovery contract, including prefetched-batch handling, random generators, dataset manifest and previous valid generation. PyTorch's general checkpoint guidance includes optimizer state for continuing training. [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

Start with a five-minute training checkpoint interval and checkpoint each completed generation shard. Tune those intervals using write overhead and lost-work measurements. An interrupted job resumes its committed generation; a partially written model is never promoted.

Estimate generation time from the pilot's measured aggregate completed labels per second: `remaining_positions / measured_throughput`. Report relabeling and validation costs separately. No overnight completion time or training-energy saving is established before this measurement.

## Promotion and repeated improvement

The first useful engine remains the CPU analyzer, which can work with a handcrafted evaluator before any network is trained. Its learning cycle is:

```text
verified simulator and endgames
    -> diverse positions
    -> CPU teacher labels
    -> train compact evaluator on GPU
    -> export and cross-check CPU inference
    -> independent analysis and arena tests
    -> accept or reject model
    -> accepted search generates the next labels
```

Promotion requires more than lower training loss:

Evaluate raw network predictions and controlled searches without exact lookups on the development oracle suite; otherwise retrieval could conceal an inaccurate network. Evaluate the complete product with tablebases separately and report lookup coverage. Sealed-final-test exclusions apply transitively as specified above. Identify any benchmark that permits oracle knowledge. Promotion uses development results only.

1. Correct export/inference agreement and declared score scaling, including quantization if used.
2. Improvement in development exact-set move regret and tactical behavior, with tied optimal choices accepted.
3. Useful equal-time or equal-energy search performance on diverse unsolved development positions, without a material loss on important development strata.
4. Head-to-head evidence using diversified starts and player swaps, with uncertainty reported.
5. Successful checkpoint interruption/restart and retention of the previous accepted model for rollback.

Never replace an exact tablebase hit with a learned prediction. Learned values and policies can order proof work, but only verified facts establish proof cutoffs. A newer teacher trained by this process is still not automatically an exact solver.

After the compact CPU evaluator earns its place, build a policy/value PUCT challenger using the same rules, replay provenance and evaluation protocol. Store its visit-policy targets separately from score-derived CPU targets. Measure whether self-play policy improvement justifies its generation, training and inference costs on this machine before enlarging the model or corpus.

## First concrete campaign

The sequence is the declared experimental profile, simulator conformance and resumable CPU analyzer, followed by a 10,000-position uniform pilot and a 100,000-position small-MLP comparison. A verified K=4 oracle is a small parallel follow-on aid; K=6/8, NNUE, prioritized weighting and quality-based pruning are separately measured expansions. A million-position campaign and PUCT self-play follow evidence of improvement rather than a fixed data-volume target. The [first implementation scope](FIRST_IMPLEMENTATION.md) is the authoritative initial backlog.
