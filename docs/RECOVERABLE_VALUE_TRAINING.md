# Recoverable value-training implementation

This stage implements the already audited, fixed 400-step pilot. It accepts a trainer and bounded engineering recovery evidence. The production run, final fit checks and both-baseline strength tests are separate stages; H_v1 remains the default engine.

The original 12,872 parameters, including both heads and the shared trunk, stay frozen. Only the separate 2,065-parameter value branch learns. One complete original H8 finite-horizon root target is normalized as `(margin - store_difference) / signed_active_seeds`. These labels are approximate teacher targets. Training uses 512 reserved roots; 128 distinct fit-check roots never enter gradients. Existing cached features are packed to FP32, without new positions, labels, searches or feature simulations.

Production uses the recorded Torch environment on CPU FP32 with one thread: 400 AdamW updates, batch 128, uniform replacement sampling, Huber delta 0.1 and gradient norm clipping at 1. The final step is the only candidate; there is no checkpoint selection. Full settings and source/runtime/model/data identities are recorded by the trainer.

Checkpoints retain branch and frozen tensors, every optimizer moment and parameter group, Python and Torch RNG states, the dedicated sampler RNG, draw cursor and complete sample-prefix digest. Scheduled checkpoints occur every 20 steps, with an initial step-zero checkpoint. A cooperative interruption may create an additional checkpoint. Recovery validates the committed prefix and independent commit receipts before restoring state. Uncommitted files are retained and replay-checked; missing or inconsistent committed evidence stops the run.

The resource supervisor durably reserves two minutes before each worker launch, with a cumulative 20-minute budget. An independent Windows watchdog verifies the actual worker process, contains it in a kill-on-close Job Object, and enforces the deadline, 4 GiB private-commit limit and 5 GiB new-stage namespace quota. Preflight requires 4 GiB free commit and RAM and 2 GiB free disk. Clean work charges elapsed time; unclean work charges its entire reservation. Unverifiable overruns stop automatic continuation.

## Commands

From the workspace root, verify the accepted implementation and packed cache:

```powershell
python tools/verify-value-training.py verify
python tools/prepare-value-training.py verify
```

When beginning the production pilot, use the supervisor. It selects the recorded Torch Python executable, so the system Python does not need Torch:

```powershell
python tools/run-value-training.py run
```

Repeat that command to resume a cleanly paused run or a verified recoverable interruption. Do not delete the output, catalogs, checkpoints or lease ledger to force a restart. A HOLD requires inspecting the recorded reason; it must not be reset by removing accounting evidence. A completed run is verified before creating workers or importing Torch.

```powershell
python tools/train-value-sidecar.py status
python tools/train-value-sidecar.py verify
python tools/run-value-training.py status
```

The implementation dashboard listens on `http://127.0.0.1:18766/` and separates engineering optimization from production steps. Once production initializes, this implementation snapshot explicitly becomes historical; publish a run observer for live production progress. Full acceptance verification hashes every bound file, while the live observer checks sources and small certificates.

## Evidence and remaining gates

See `build/value-training-v1/audit.json` and `docs/VALUE_TRAINING_IMPLEMENTATION_AUDIT.md` for actual independent checks, failed attempts retained as history, process-kill results and engineering execution counts. The six-step fixtures use only the first 32 frozen training roots and export ineligible engineering models. They demonstrate recovery and numerical conformance, without demonstrating playing strength.

After production step 400, run the prescribed 128-root fit diagnostics, exact endgame guards, fixed-simulation diagnostics, equal-time position tests and 192 games against H_v1 and H8 relay CPU. Recheck full game trajectories for ownership overlap before interpreting the arena. Fit-check roots have historical training ancestry; synthetic ordinary arena starts have no established standard-opening reachability. These limits constrain the conclusions even when the checks pass.
