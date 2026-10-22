# Small fixed-policy value pilot

**The pilot is complete and is not promoted.** It slightly improves aggregate target fit but worsens the small ordinary-development cohort. Against contemporaneous control, its paired mean margin improves by 0.75 seeds versus H_v1 and declines by 0.9167 versus H8 relay. Neither baseline's exploratory interval clears zero.

The candidate starts from the retained improved-policy model at `data/quality-train-v1/model.json`, SHA-256 `bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`. It does not start from the earlier unpromoted value refit. Only the final value head's 65 parameters may train; six trunk/policy tensors remain byte-identical. Raw policy invariance does not require identical PUCT choices, because value backups affect search.

The fixed treatment completed 200 deterministic single-thread CPU steps, batch 256, seed 20261022, ten checkpoints and 51,200 draws. Step 200 remains the candidate regardless of metrics. Its 7,624 optimizer rows comprise 7,563 replay roots admitted after protection filtering and 61 ordinary roots across 59 families; 29 replay rows were excluded. Original artifacts remain intact; filtering cannot undo the source model's ancestral exposure.

The sampler assigns 95% mass to replay (90% uniform-root and 10% uniform-stratum within replay) and 5% to ordinary roots. Ordinary targets use the deepest complete requested label: 32 H8, 24 H10, 5 H14; H14 validation D5 is excluded from training. Confidence uses the last three complete requested depths among 8/10/12/14: `1/(1+target_span/4)`, a 0.5 single-depth factor and another 0.5 for disjoint final two best sets. These are uncalibrated exposure heuristics, not accuracy probabilities. The nine historically overexposed roots have expected mass 1.4589% rather than historical uniform-stratum 14.29%; roots with 5-24 active seeds retain 33.667% mass. The [per-root review](../build/value-pilot-sampling-review.json) records exact exposure. Unit loss weights and no importance correction mean this sampler changes the objective. The bundled data, sampler and refit result cannot isolate one component's effect.

The schedule was frozen before outcomes or training: 12 consumed development arena families (4 each ordinary/capture/bonus), paired seats, 48 pretraining control games and 96 posttraining interleaved control/pilot games against both baselines. Forty-four retained diagnostics (12 H8 and 32 exact), at 25/100 ms, yield 264 pretraining and 352 posttraining measurements. These are reused development cohorts with ancestral exposure; no sealed validation was consumed.

Search remains uncached CPU PUCT, c_puct 0.25, no noise, 100,000 simulations and 2,048 nodes maximum; arenas use the same pinned core and 25 ms whole-decision budgets. Baselines retain TT16 MiB, PVS and depth 16. Clocks include reset, features, inference and validation, with no unlimited time-zero search. Raw physical counters, overshoots and traces are retained.

Pretraining control finished all 44 cases and 48 terminal games with zero fallbacks:

| Opponent | Paired mean seed margin | Exploratory 95% paired bootstrap interval | Wins/draws/losses |
| --- | ---: | --- | --- |
| H_v1 | +0.8333 | [-2.9167, 4.7500] | 10/1/13 |
| H8 relay CPU | -0.9167 | [-5.4167, 2.5833] | 11/1/12 |

Neither pretraining interval clears zero. These controls remain a descriptive timing reference; between-invocation differences are not attributed to training.

The final model file SHA-256 is `7d19e6685000406349cf1a7aa862f4a13665ff59de41145a8f29de10eb8c0f1d`. All six frozen tensors and admitted-dataset policy logits remain bit-identical. Target fit is mixed:

| Descriptive cohort | Roots | Source MAE, seeds | Pilot MAE, seeds |
| --- | ---: | ---: | ---: |
| Retained development | 1,000 | 5.3391 | 5.2683 |
| Ordinary development | 14 | 9.1190 | 9.1550 |
| Protected diagnostic | 8 | 4.0433 | 3.9843 |

The 372 reused H8 development roots with 5-24 active seeds are a subset of the 1,000-row cohort: their row-weighted MAE is essentially unchanged, 1.9244 to 1.9230. The ordinary cohort worsens slightly despite the aggregate improvement. These reused cohorts contain approximate finite-horizon targets; their errors do not establish playing strength or generalization. The final checkpoint remains the predeclared step 200 without additional optimization or model selection.

All 96 posttraining games terminated with zero fallbacks. Each row contains twelve paired families and 24 games:

| Profile | Opponent | Paired mean margin, seeds | Exploratory 95% paired bootstrap interval |
| --- | --- | ---: | --- |
| Unchanged control | H_v1 | +1.2500 | [-2.5000, 5.0000] |
| Value pilot | H_v1 | +2.0000 | [-0.4167, 4.5000] |
| Unchanged control | H8 relay CPU | -0.7500 | [-5.3333, 2.7500] |
| Value pilot | H8 relay CPU | -1.6667 | [-5.3333, 1.3333] |

The predeclared matched candidate-minus-contemporaneous-control differences are **+0.7500 seeds against H_v1 and -0.9167 against H8 relay**. These descriptive means have no additional intervals or promotion gates. All four baseline intervals include zero, and both profiles fail the both-baseline gate. This small, reused development sample gives mixed evidence rather than a demonstrated strength improvement. No checkpoint or search setting was changed in response.

Both PUCT profiles match all 32 exact diagnostic references at both budgets. On the twelve H8 diagnostics both match 10/12 references with mean reference regret 0.9167 seeds at both 25 and 100 ms. Reference agreement does not change the PUCT report's Monte Carlo proof status into an exact proof.

All **24 unique current checks passed** (4 training-owner, 12 benchmark-owner, 8 independent). The auditor replayed all 200 optimizer steps, inspected all 10 checkpoints and optimizer/RNG/cache/policy states, literally replayed 144 production games and 4,852 choices, validated 616 measurements and rederived all paired statistics. Prefreeze author fixture errors are retained; no postoutcome source change or retuning occurred.

Four real kills covered checkpoint file-ready/committed at fixture step 4 and game pre/postcommit at 0/1 committed games. Resumed training matched uninterrupted model, optimizer and RNG bytes; game prefixes stayed exact and resumed traces replayed legally, without requiring fresh timed choices to match. Completed reruns made zero optimizer/runtime activations, decisions or searches and preserved 26 durable files' bytes, sizes and mtimes. All **2,426 prior hashes** remain current. Missing-export repair and corrupt marker, margin and physical-work refusal passed.

Benchmark `run()` spans are 49.6554 s pretraining and 100.2656 s posttraining; peak working sets are 247,177,216 and 300,388,352 bytes. They include admission, engine startup, work and publication, excluding wrapper/imports, training, fixtures, audits and earlier invocations; they measure neither total pipeline cost nor electricity.

Original run/resume commands preserve the registered configuration and durable committed prefixes; tensor inspection is a prepublication step:

```powershell
python -m congklak.value_pilot_progress run --tool value-pilot-pretraining -- python tools/benchmark-value-pilot.py run --stage pretraining
python tools/benchmark-value-pilot.py verify --stage pretraining
$pilotPython = (Get-Content build/training-runtime.json -Raw | ConvertFrom-Json).python
python -m congklak.value_pilot_progress run --tool value-pilot-training -- $pilotPython tools/value-pilot-train.py train
& $pilotPython tools/value-pilot-train.py inspect
python -m congklak.value_pilot_progress run --tool value-pilot-posttraining -- python tools/benchmark-value-pilot.py run --stage posttraining
python tools/benchmark-value-pilot.py verify --stage posttraining
python tools/record-value-pilot.py --check-only
```

Use the pinned Torch interpreter for training/actual tensor inspection; desktop Python handles benchmarks and completed CPU-only checks. Training requires the complete pretraining witness. Uncommitted steps/games restart from durable prefixes. Fixture suites write evidence before publication; after acceptance use the recorder read-only check or new versioned fixtures.

The dashboard composes immutable historical observers. Browser policy blocked visual verification; no new screenshot is claimed. Mutable activity/runtime state is outside acceptance. H_v1 remains default; no sealed final test, promotion or energy claim.

Evidence: [training result](../data/value-pilot-v1/result.json), [actual checkpoint witness](../build/value-pilot-model-witness.json), [pretraining control](../build/value-pilot-benchmark-pretraining.json), [posttraining comparisons](../build/value-pilot-benchmark-posttraining.json), [independent audit](VALUE_PILOT_INDEPENDENT_AUDIT.md), [combined no-work proof](../build/value-pilot-root-verification.json), [scope](VALUE_PILOT_SCOPE.md). Keep the retained control and H_v1 default; this small bundled pilot does not justify larger training or a strength claim.
