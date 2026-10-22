# Phase 6 independent audit

All **14 independent checks passed**, and the full production audit passed. The recoverable learning and analysis pipeline works. **Both PUCT candidates failed the strength gates against both CPU baselines; H_v1 remains the default.** No initial-board solution, sealed-test result, energy saving or automatic promotion is claimed.

The rule contract is the user-confirmed id-betawi-relay7-v0 profile: one round with 98 seeds, relay sowing, permanent stores and final player0 store difference. Literal checks cover capture, bonus turns and forced passes. Every PUCT backup retains the fixed-player0 value; selection uses the actual normalized controller. Visit fractions are search allocations, not calibrated probabilities. Exact terminal outcomes remain distinct from estimated search means and finite H8 teacher targets.

The 14 accepted methods comprise eight core ownership/search/journal/analyzer checks, one mixed-target tensor check, one explicitly limited author-warm evidence review, one actual CUDA self-play recovery check, one actual CUDA update recovery check, one benchmark case recovery check and one arena recovery check. Exact method names, current source hashes and retained artifacts are bound in [tests.json](../build/phase6-audit/tests.json) and [audit.json](../build/phase6-audit/audit.json). Supporting artifact verification and the dashboard review do not add to this count.

The final audit independently verified:

- All **878** earlier accepted artifacts still match their hashes after Phase 6 production.
- Metadata-only reconstruction matches **21,777 protected quotients and 2,370 families**. Literal replay covers all **720 historical games and 23,081 choices**, without reading sealed target bodies.
- Predeclared cohorts remain 32 train families for self-play, 24 development arena families excluding 60 earlier arena families, 48 H8 development proxies and 32 exact diagnostic positions.
- All **64 self-play games terminated**, with **2,154 literal decisions** and matching sampled moves/RNG states. All **623 admitted rows** have independently rederived features, individual terminal seed-margin targets and legal visit targets. Twenty entire families were excluded for protected-state bridges; twelve were admitted. The four-observation raw-state cap required no further pruning. Targets were not averaged and retained rows have unit weights.
- All **30 production checkpoints** pass file/catalog checks and safe weights-only inspection: 20 warm checkpoints through step 1000 and ten update checkpoints through step 500. Every sampler state and draw counter was replayed. Warm training freezes six value/trunk tensors and trains two policy tensors; the update trains all eight tensors with separate half-batch H8 and self-play objectives. Warm selection is the scheduled development-policy winner at step 1000; update selection is the fixed final step 500.
- Both exports match independently derived features and double-precision forwards on all **1,000 development roots**. Literal seed-MAE calculations reproduce published FP32 overall/per-stratum metrics within **0.0001 seed**, including eligible ordinary preservation gates. Overall H8 MAE is 5.062 seeds for warm and 5.730 for updated, versus 14.796 for the store-difference baseline. Both ordinary gates pass; these development metrics do not establish game strength.
- All **80 cases / 800 scalar measurements** and **192 arena games / 6,304 choices** match durable publications, current producer/CPU identities and the whole-decision timing contract. Every arena game terminates, with zero fallbacks. Family-paired bootstrap intervals were independently recalculated using the fixed 4,096-sample protocol.

At 25 ms per decision, all four primary paired terminal-margin gates fail:

| Candidate | Opponent | Mean margin, seeds | 95% family-bootstrap interval |
|---|---|---:|---:|
| Warm CUDA PUCT | H_v1 | -23.583 | [-29.250, -18.167] |
| Warm CUDA PUCT | H8 relay CPU | -23.333 | [-28.542, -18.458] |
| Updated CUDA PUCT | H_v1 | -21.708 | [-26.417, -17.125] |
| Updated CUDA PUCT | H8 relay CPU | -20.167 | [-24.583, -15.958] |

These are development-family pilot results; the development corpus was already used for model selection. They support keeping the CPU baseline and improving search/inference efficiency before expanding the model or database. They do not establish an unseen-set result or a statistically verified improvement of updated over warm.

Recovery checks kill real processes at self-play decision/publication boundaries, analyzer commit/export boundaries, CUDA update checkpoint/preexport boundaries and benchmark case/completed-game boundaries. The update six-step batch32 fixture uses the production warm model and 623-row replay: precommit recovery restores step 2, final preexport recovery restores step 6, and both exactly match uninterrupted model, optimizer, sampler, Python/Torch CPU/all CUDA RNG and selected state. Completed reruns preserve bytes and mtimes; checksum-valid forged states/exports/outcomes are refused. The arena fixture retains 191 committed games, resumes to 192 and literally replays all games. Raw fresh/resumed checkpoints, the precommit orphan, correct benchmark backups, kill markers and individual worker logs remain hash-bound in group reports. CPU deterministic fixture equality and committed GPU-prefix retention are verified; different GPU batch shapes after interruption are not promised to produce bitwise-identical future play.

The **warm recovery review has a narrower scope**. It verifies the trainer author's source and published six-check record reporting two actual warm CUDA kills, plus a separate complete 20-step fixture's catalog, hashes, scheduled winner and frozen-value export. The author's successful kill fixtures were temporary and removed, so this independent audit cannot recompare their raw fresh/resumed states. It does **not** claim a successful independent live warm kill. Both earlier independent warm attempts failed before kill markers and remain unaccepted diagnostics:

| Retained diagnostic | Scope |
|---|---|
| [warm-resource-failure.json](../build/phase6-audit/warm-resource-failure.json) | First independent live warm attempt failed allocating host-backed memory before its kill marker. |
| [warm-batch32-host-memory-failure.json](../build/phase6-audit/warm-batch32-host-memory-failure.json) | Bounded warm retry failed at the teacher semantic digest before CUDA admission. |
| [phase6-selfplay-training-acceptance-v2](../build/phase6-selfplay-training-acceptance-v2) | Original failed update probe and transient post-exit writer-lock hash diagnostic are preserved separately from later successful retained recovery runs. |

The separately versioned V2 update validates CPU corpora before opening CUDA. Its retained CPU comparison matches V1 model, optimizer, sampler, CPU RNG, selection metrics and nonidentity export fields at steps 5 and 10. Both CPU working sets are about 783 MB; no CPU peak-memory reduction is claimed. Successful CUDA recovery and production updating followed the user's decision to pause other heavy work. No unrelated process, Windows memory/power setting, driver or accepted runtime was changed.

The final dashboard supporting review verifies a real live process identity, JSON-list/tuple compatibility, dead or reused PID detection and read-only log bytes/mtimes. It also checks that completed 80/192 work visibly retains failed strength gates and attention flags. An earlier live-only probe raced with an already completed arena test; that diagnostic is retained separately, not counted as a dashboard defect or passed live check. Current final UI sources are bound in [dashboard-review.json](../build/phase6-audit/dashboard-review.json).

Equal-time decisions include wrapper work and report overshoot; feature extraction and launched CUDA kernels are not fully preemptible. The native telemetry adapter performs two real feature passes while CUDA performs one, so adapter timings are not a pure kernel comparison. The measured native adapter is faster at every tested batch size. Energy remains unmeasured. The pilot is complete as engineering evidence and fails its strength objective; release validation, faster search/evaluation and remaining finalized-analyzer work are recorded separately in [engine readiness](PHASE6_SCOPE_AND_ENGINE_READINESS.md).
