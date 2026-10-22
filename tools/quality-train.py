"""Admit, run, and resume the fixed compact joint policy/value treatment."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, quality_learning as learning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "train", "verify"))
    parser.add_argument("--registry", default=str(learning.REGISTRY))
    parser.add_argument("--run", default=str(learning.DEFAULT))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    def event(value):
        print(config.canonical(value), flush=True)
        if value["event"] == os.environ.get("QUALITY_TRAIN_BLOCK_EVENT") and str(value.get("step")) == os.environ.get("QUALITY_TRAIN_BLOCK_STEP"):
            marker = Path(os.environ["QUALITY_TRAIN_MARKER"])
            learning.teacher.atomic_json(marker, value)
            while True:
                time.sleep(.05)
    try:
        if args.operation == "prepare":
            manifest, _ = learning.prepare(args.registry, ROOT / "build/quality-data.json")
            event({"event": "quality_corpus_admitted", "train_rows": manifest["train_rows"],
                   "excluded_rows": len(manifest["excluded_rows"]), "development_rows": manifest["development_rows"],
                   "sha256": manifest["sha256"], "path": "build/quality-data.json"})
        elif args.operation == "verify":
            proof = learning.verify_run(args.run, None if args.fixture else ROOT / "build/quality-model-witness.json", args.fixture)
            event({"event": "quality_training_verified", "selected_step": proof["selected_step"],
                   "checkpoints_verified": proof["checkpoints_verified"], "model_weights_sha256": proof["model_weights_sha256"]})
        else:
            result = learning.train(args.run, args.device, event, lambda: stopped, args.fixture,
                                    args.steps, args.checkpoint_every, args.batch_size, args.registry)
            event({"event": "quality_training_done", "status": result["status"], "steps": result["steps_completed"],
                   "weights_sha256": result.get("weights_sha256")})
        return 0
    except Exception as exc:
        event({"event": "quality_training_error", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
