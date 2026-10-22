# Independent audit of root-action preferences and ordinary bias

**Implementation, recovery and experiment integrity pass. H_v1 remains the default.** The completed V2 audit checks all 391 root cases and literally replays all 144 arena games. The selected joint candidate passes the predeclared development seed-margin gate but fails ordinary evaluation gates; isolated policy ordering does not improve play. The auditor owns its tests, recorder and this report separately from the trainer, native engine and arena implementation.

## Targets and objective boundaries

Value targets are fixed-player-0 **root H6** estimates. Action targets are the complete legal root H6 action scores, each incorporating the chosen action and **H5 evaluation of its successor**. They are used only as root policy preferences. No successor is added as an H6 value-training row. Independent literal minimax fixtures demonstrate that successor H5 and H6 values can differ.

Policy targets softmax root-actor-signed action scores at a fixed temperature of four seeds. Forced passes normalize the actual root actor; bonus choices and successor control changes do not introduce an extra sign flip. Illegal actions have zero target probability. Every legal action must have exactly one Q label, ties receive equal preference and impossible physical margins are refused. The training CE masks illegal log probabilities before multiplication, avoiding zero-times-infinity. Native policy scores order legal actions rather than prune them or assert winning probabilities.

Six matched experiments use 43 inputs, width 64, 7,496 parameters, seeds 2718/31415, 8,000 steps and batch 512. A uses root value Huber loss. B adds 0.25 policy CE. C adds 0.01 mean-squared ordinary-stratum minibatch actor-relative bias. Eligibility is determined solely from TRAIN ordinary strata with at least 20 rows; absent groups are omitted, and an all-absent minibatch contributes a graph-connected zero. There is no fitted development calibration or runtime residual knob. A's 455 policy-head parameters receive no preference learning, apart from AdamW decay; matching parameter count and sampling does not imply identical effective gradient computation.

Scheduled value macro MAE selects checkpoints every 100 steps. Lower macro MAE selects a seed per variant; paired mean macro MAE selects B or C as the challenger. The selected A control and joint challenger are fixed before arena results. Policy diagnostics and arena outcomes do not select models.

## Native modes and independent checks

The separate action DLL supports H_v1 leaves with/without policy ordering and learned leaves with/without policy ordering. Ordering starts at remaining depth two, keeps valid TT hints first, sorts actor-relative logits with pit-order tie breaks, and preserves full-window ascending root candidates and lexicographic PV reconstruction. Model, mode, backend, ordering threshold, rules, binding and DLL identities gate analysis resumption.

Nine independent checks pass in 14.595 seconds:

- Prior scientific files, models, simulator DLLs and environment tools remain unchanged.
- Scalar and AVX2 value/policy outputs agree bitwise on width-17 and width-64 adversarial fixtures in all four modes, including very large illegal logits, controller/pass normalization, antisymmetry and exact store-offset translation.
- Literal seed-by-seed minimax checks 32 positions at depth four across 32 mode/backend/TT/PVS combinations, for 1,024 searches. Complete root action values agree; H modes preserve the original H_v1 optimal sets and lexicographic PVs.
- Literal H6 target fixtures independently verify root-actor signs, legal masks, normalization, ties and the H5/H6 successor distinction.
- Synthetic CPU-set records exercise variable sizes, unknown types, parking/allocation flags and permitted masks. Actual affinity is verified and restored after normal and exceptional context exits.
- A real analyzer process is killed after committed depth two; depth-five resumption equals fresh analysis. Changed mode/threshold and checksum-valid corrupt reports are refused.
- A real CUDA process is killed after writing an uncommitted step-20 checkpoint. It restores committed step ten and reaches bitwise-identical full heads, optimizer, sampler and all RNG states at step forty. Changed objective is refused; masked CE and missing bias groups remain finite.
- A real pinned benchmark process is killed after its first complete case. Resumption preserves that case, commits the next, retains identical affinity identity, is idempotent and refuses forged checksum-valid regret.
- A read-only SQLite backup of completed V2 production is copied into an isolated test directory. Only the copy's games/completion marker are cleared. Two real child kills after successive full games verify 1→2 committed games, an unchanged first-game body, idempotent validation and literal replay. All 391 original cases and all 144 production games remain untouched.

Evidence: `build/stage5-action-audit/tests.json`, with the earlier first-eight V2 evidence preserved separately. Arena instrumentation observes the actual constructor identity in a test sidecar; it does not change frozen benchmark code, model, CPU declaration, search or results.

## Recovery defect found and versioned correction

An additional real arena-kill test caught a genuine V1 timing-admission problem. After committing a full game and killing its child, the next process refused the copied ledger: V1 had selected CPU-set id258/logical2, while its automatic selector next chose id262/logical6. V1 selects the second currently unparked performance core, so ordinary OS parking changed its scientific identity even though sources, models, topology and permitted masks were unchanged. The completed V1 results remain historical evidence; no identity or data was rewritten to bypass the refusal.

A separately versioned V2 protocol declares the exact seven structural CPU fields before measurement. It locates that CPU, permits transient parking, validates actual permission/performance class/reservation, pins and wakes the real CPU, and verifies execution. It never falls back to another core. Stable topology and selected CPU form scientific identity; initial parking and original affinity belong to truthful invocation observations. The exact prior process mask is restored after success or exception. The new V2 benchmark starts from an empty ledger and reruns all 391 cases and 144 games.

The first nine-check attempt also had a CUDA allocation/unknown-error failure before its requested checkpoint barrier. The exact cause is unconfirmed; GPU VRAM was available and system RAM was under pressure. Sequential fresh reruns passed full-state recovery without changing the trainer. The original failures and confirmed identity difference are preserved in `first-nine-check-failure.log`, `additional-check-failure.json` and `arena-identity-drift.json` under the audit directory.

## Pinned equal-time protocol

The primary API definition confirms that higher Windows CPU-set efficiency classes indicate faster, less efficient cores: [Microsoft CPU-set documentation](https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-system_cpu_set_information). V2 pins only its own process to the predeclared allowed OS-reported performance core and restores its original affinity mask. It refuses multi-group topology. Pinning controls migration; it does not reserve an exclusive core, control frequency/power or establish lower energy consumption.

The frozen root plan contains 100 old H6 proxies, 32 known exact K4 diagnostics and all 259 current ordinary H6 development roots. The 16 legacy exact roots with old sealed quotient aliases are excluded using metadata. Sealed target bodies are not read. Six engines rotate under equal 25/100 ms native deadlines, TT16/PVS and depth ceiling 16, with fresh caches. Original H_v1 and action H/no-order controls both remain present because finer fixed-point PVS windows can change work counts without changing fixed-horizon values.

Three arenas each use the same 24 predeclared distinct, contested starting families, both seat assignments, 20 ms per choice and a 512-choice cap. **Paired terminal seed margin is the predeclared primary outcome**, with 4,096 family bootstrap resamples and a strictly positive lower 95% bound required. WDL is secondary. Learned leaves additionally require the ordinary no-regression gate. Completed games are independently replayed with literal rules; unfinished games receive no invented draw.

## Final V2 evidence and decision

The selected control is A seed 31415; the joint challenger is B seed 31415. C reduces some high-ordinary bias but has worse macro value error and was not selected. All ordinary promotion gates fail. The auditor independently verifies all six fixed-cadence histories, train-only eligibility, selected policy/value exports, representative selection and full-head native parity on all 1,000 development roots per selected model.

All 391 root cases have complete, contiguous six-engine measurements, with immutable source/model/DLL identities, legal selections, separate reference cohorts and regenerated summaries. Every case and game records actual group0/logical2/mask4 verification. Dynamic startup observations are checksum-checked outside scientific identity. On 1,360 matching-horizon H comparisons and 676 matching-horizon learned-value comparisons, root values and optimal sets agree exactly across ordering ablations.

All 144 games independently replay to the same terminal boards, margins and WDL scores, with zero fallback choices. Primary seed-margin bootstrap resampling was predeclared, groups by the same 24 distinct starting families and is independently regenerated.

| V2 arena | Paired mean candidate seed margin | Family bootstrap 95% interval | Primary gate |
| --- | ---: | --- | --- |
| Joint value + policy versus original H_v1 | +2.500 | [1.083, 3.917] | Pass |
| H_v1 policy ordering versus H_v1 without ordering | −0.792 | [−1.583, −0.125] | Fail |
| Joint policy ordering versus the same learned value without ordering | −0.333 | [−0.917, 0.167] | Fail |

The first result is positive evidence within this repeated development cohort. It does not establish broad generalization, and it does not establish a benefit from policy ordering: the two isolated ordering comparisons fail. V1 results are historical evidence of the previous timing protocol and are not pooled with V2. No parameter, target, selection or source identity was adjusted using these outcomes.

Both selected models remain worse than H_v1 in the middle/high ordinary regression strata. For B, ordinary MAE is 2.192 / 10.562 / 27.021 seeds in the 5–24 / 25–49 / 50–74 bands, versus H_v1's 2.333 / 5.226 / 11.822. Therefore **H_v1 remains the default and learned-model promotion is held**, despite the positive development seed-margin result. No full-game solution or lower-energy claim is supported.

```powershell
py -3.12 tests/stage5_action_audit_acceptance.py
py -3.12 tools/stage5-action-audit.py
```
