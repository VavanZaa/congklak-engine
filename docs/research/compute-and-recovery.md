# Congklak compute and recovery design

The first product should analyze Indonesian Congklak or Dakon positions and explain the best available moves. Use the i9-13900K for the initial search engine and the RTX 4080 for optional evaluator training and batched search experiments. Overnight work must resume from committed progress after an interruption. The limits below are proposed starting settings, not measured requirements or performance claims.

## Hardware and resource limits

The user confirmed an i9-13900K, 32 GB RAM and RTX 4080 with 16 GB VRAM on 14 October 2026. Read-only system queries matched these components; NVIDIA's query reported 16,376 MiB GPU memory. Intel specifies 8 performance cores, 16 efficient cores, 32 threads and AVX2 support for this CPU. Discover the actual logical processor mapping before assigning worker affinity. [Intel specifications](https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html)

| Resource | Initial proposal | Experiment or limit |
| --- | --- | --- |
| Interactive search | One deterministic worker first | Compare 1, 2, 4 and 8 workers; later test all cores and SMT separately |
| Search transposition table | None in first release; 2 GiB when introduced later | Compare 1, 2, 4 and 8 GiB inside the total RAM budget |
| Total engine RAM | At most 24 GiB across concurrent jobs | Include workers, tables, queues, replay buffers and resident mapped pages |
| GPU memory | At most 12 GiB initially | Measure actual peak allocation, leaving room for the desktop and temporary tensors |
| Local job storage | Initial 50 GiB physical managed-data quota | Includes temporary/uncommitted, fallback and retained files, SQLite/WAL/backups and dependency metadata; reserve rewrite space before admission |
| Checkpoint interval | Training every 5 minutes; self-play after each completed game | Tune against observed write cost and maximum acceptable lost work |
| Background work | Local overnight jobs | No cloud compute is assumed |

The [first implementation](../FIRST_IMPLEMENTATION.md) uses one analyzer and small inline SQLite reports, without a bulk collector or training worker. Its initial managed-memory budget is 512 MiB, configurable under the aggregate ceiling. Batch independent self-play games and corpus analyses only as follow-ons. A scheduler should reduce background work when interactive analysis needs resources; each job has time, memory and storage limits.

## Measure quality and energy together

For an analysis engine, the important measures are best-move accuracy on verified positions, regret in seeds when exact values exist, stability across increasing budgets, worst-case response time, and correctness of the explanation. Report peak memory, joules per analyzed position and energy to reach a specified accuracy alongside elapsed time. Nodes per second and average watts alone cannot select the strongest or most efficient engine.

Compare search backends under equal wall-clock budgets and equal energy budgets. Use fixed-node runs to isolate evaluator or move-ordering effects within the same search family; an alpha-beta node and a PUCT simulation are different units. For learned challengers, also report the energy and time spent generating training data and training the model. Estimate the number of analyses needed to amortize that cost; do not hide it in inference-only comparisons.

NVIDIA exposes GPU power readings and, on supported devices, cumulative energy in millijoules since the driver reload. Prefer an energy-counter difference if this GPU exposes it; otherwise integrate sampled power and identify that estimate. Counter resets and concurrent GPU use must be recorded. [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

Intel PCM exposes hardware performance and energy monitoring facilities, but support and access on this Windows installation need to be checked during implementation. GPU energy and CPU package energy do not include the whole system. Use a wall meter for a whole-computer claim; otherwise label precisely which components were measured. [Intel PCM](https://github.com/intel/pcm)

Test performance-core-only search, mixed cores, SMT and different supported power settings as experiments. Keep the machine's existing configuration until such an experiment is deliberately run. The GPU's reported power limit is a limit, not an observed workload draw. Lower wattage may increase run time, so compare total joules and achieved analysis quality.

## Durable data and disposable caches

| Data | Persistence contract | Retrieval purpose |
| --- | --- | --- |
| Transposition table | Disposable acceleration data; optional consistent snapshot | Reuse completed search bounds and move ordering inside compatible search contexts |
| Neural evaluation cache | Disposable; keyed by state and model identity | Avoid duplicate inference |
| Analysis library | Durable, versioned input and completed result | Reopen an analysis and identify whether it is an estimate, rigorous bound or proof |
| Self-play replay | Immutable completed game chunks and committed manifest | Resume data generation and reproduce a training dataset |
| Training checkpoint | Committed model, optimizer and progression state | Continue the same training run |
| Verified tablebase | Immutable validated shards with coverage manifests | Retrieve exact outcomes only inside established coverage |
| Solver work queue | Transactional frontier and independently verified completed subproblems | Resume unfinished proof work |

Every durable record should carry a canonical rule contract and its digest, state encoding version, objective, source engine revision, and model digest when applicable. Search results also need their search mode, horizon convention, evaluator identity and terminal policy. A large cache key cannot compensate for omitting rule-dependent history from the state.

Store compact structured states for exact lookup. Vector similarity search is useful only for optional educational examples or strategic motif retrieval. Similar positions must not supply exact values, cutoffs or proof facts. Any retrieved explanation example must be checked against the current move trace and rule profile.

Use a local SQLite metadata and job database with a single coordinated writer, plus immutable bulk files for replay, models and tablebases. WAL with `synchronous=FULL` is the proposed metadata durability setting; verify the resulting settings and test them on the actual filesystem. SQLite documents the different durability guarantees of its synchronization modes. [SQLite synchronization settings](https://www.sqlite.org/pragma.html#synchronous)

Back up a live metadata database through SQLite's backup mechanism, rather than copying only its main file while a WAL is active. The WAL can contain committed changes not yet transferred to that file. For later bulk artifacts, pin the backup's complete referenced file closure while copying and verify it before publishing a complete backup. [SQLite WAL documentation](https://www.sqlite.org/wal.html)

## What an interruption preserves

Interactive search publishes a coherent result only after a completed iteration, or publishes explicitly partial per-move results with their individual coverage. On cancellation, retain the last coherent report. On restart, load the input, settings and committed report, then repeat unfinished work. A transposition-table snapshot can accelerate that work; it does not restore an arbitrary multithreaded search instruction by instruction.

Self-play commits completed games with unique job and game identifiers. Save game state, move list, model identity and generator state at a decision boundary if resuming an unfinished game is worthwhile. Reissuing a completed game must not append it twice. An interrupted game must not receive a fabricated terminal value or enter training as a finished result.

Training checkpoints include model and optimizer state, optimizer step, learning-rate scheduler, mixed-precision scaler if used, dataset manifest and cursor, sampling state, and Python, NumPy, CPU and GPU random-generator states. PyTorch's guidance distinguishes an inference model from a checkpoint that also restores optimizer and training progress. [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

Snapshot training after a completed optimizer step, awaited device updates and no partial gradient accumulation. Start with num_workers=0 and no prefetch. With prefetch later, either consume pending batches before snapshot, reset the sampler to its consumed boundary, or persist pending sample/augmentation identities and their matching state. Discarding queued batches with an advanced cursor is not a valid drain. Capture q, route masses, priority snapshot/model identity and generator state too.

Single-worker search conformance compares values, deterministic moves and traces after restart at the same fixed horizon/evaluator or fully specified logical budget. Equal elapsed-time cutoffs can complete different depths; time and work counters need not match. Test bitwise training continuation only in a supported deterministic configuration with queued work captured. Parallel training can resume validly with different later scheduling. PyTorch's deterministic-algorithm option alone is insufficient for complete application reproducibility. [PyTorch determinism API](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)

Exact solving checkpoints preserve the unresolved frontier and completed proof dependencies. Only verified complete subresults can be reused as solved. Requeue in-flight work after interruption. Tablebase construction checkpoints shard completion and dependency layers; a file's existence alone never marks its coverage complete.

## Checkpoint commit protocol

1. Reach a consistent boundary; capture the components belonging to the same step or completed unit of work.
2. Write new immutable files to temporary names on the same volume, close and flush them through the platform's durability facilities, then verify their sizes and checksums.
3. Publish final filenames and transactionally commit the manifest referring to those files. Do not advance the committed step before all its required components exist.
4. Retain the previous valid generation. On startup, validate the newest committed generation; fall back to the previous valid one if necessary. Uncommitted files can be ignored and cleaned up later.

Committed catalog pointers are the authority. A final filename or an orphan manifest does not establish a committed generation. Before writing, account for simultaneous source, replacement, temporary and fallback bytes, SQLite/WAL and managed backups. Check the physical quota and actual free space, reserving at least 2 GiB for later bulk rewrite/recovery overhead. The inline first analyzer uses a smaller explicit startup reserve. A full quota must pause admission rather than forcing deletion of a live recovery dependency.

For later compaction, use one local writer to compute a stable transitive closure of current/fallback generations, retained/retired-but-retained datasets, teacher/model checkpoints, proofs/results, backup pins, live readers and active-job writer/staging pins. Register staging ownership before creating temporary or final-uncommitted files. Uncommitted files are reclaimable only after their owner is known inactive; alternatively quiesce bulk writers during collection. Serialize pin acquisition with retirement. Commit tombstones, prohibit new references to retiring paths, recheck dependencies and reclaim only unreferenced managed paths within the resolved directory. Delete retries are idempotent and failures remain counted. Do not reuse generation paths. Retain old fallback roots until validated reduced fallbacks replace them; startup cannot resurrect retired coverage.

Tablebase retirement updates direct/aggregate coverage and every dependent lookup, certified cache and solver checkpoint. Preserve and independently verify self-contained evidence before allowing a retained proof to lose external dependencies; otherwise mark its certified evidence unavailable. Removing source bytes does not make a previously true fact false. These rules govern operational availability and auditability.

Atomic publication and checksums solve different problems: publication prevents mixing generations, while checksums detect damaged or incomplete content. A rename alone is not a tested guarantee against power loss. Implement the protocol with Windows-aware filesystem operations and verify process-termination recovery before claiming crash resilience.

## Recovery acceptance tests

Terminate jobs around file writes, manifest commits and publication. Test truncated checkpoints, missing chunks, full disk and incompatible versions. Later compaction tests include a new pin/reader during retirement, a live staging writer, a fallback referencing old files, failed deletion, crash after deletion but before catalog completion, orphan files with inactive versus active owners, backup closure and transitive proof invalidation. Recover a complete committed generation or a clear recoverable failure; never promote incomplete work.

The first analysis release passes when it can stop during a long search, reopen the saved position, retain the last completed explanation, and continue analysis. Training and solver resilience become additional acceptance gates only when those jobs are implemented. None of these tests or energy benchmarks have been run yet; this artifact defines the implementation contract.
