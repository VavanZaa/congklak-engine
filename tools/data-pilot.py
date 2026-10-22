"""Plan, freeze, collect and label the bounded ordinary-root data pilot."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, data_pilot as pilot
from congklak.puct import CancellationToken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "freeze", "collect", "label", "all", "verify"))
    parser.add_argument("--run", default=str(pilot.DEFAULT))
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--families", type=int)
    parser.add_argument("--diagnostic-limit", type=int)
    args = parser.parse_args(); cancel = CancellationToken()
    def stop(*_):
        cancel.cancel()
    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    def event(value):
        print(config.canonical(value), flush=True)
        step = value.get("index") if "family_" in value["event"] else value.get("completed")
        if value["event"] == os.environ.get("DATA_PILOT_BLOCK_EVENT") and str(step) == os.environ.get("DATA_PILOT_BLOCK_STEP"):
            pilot.teacher.atomic_json(Path(os.environ["DATA_PILOT_MARKER"]), value)
            while True:
                time.sleep(.05)
    try:
        if args.operation in ("plan", "freeze"):
            value = pilot.plan() if args.operation == "plan" else pilot.freeze()
            event({"event": "data_pilot_" + args.operation, "sha256": config.digest(value),
                   "diagnostic_pool": len(pilot.read(pilot.REGISTRY)["diagnostic_pool"]), "fixed_model_sha256": pilot.MODEL_SHA})
        else:
            if args.operation == "verify":
                pilot.require((Path(args.run) / "manifest.json").exists(), "Pilot is incomplete")
            value = pilot.run(args.run, "all" if args.operation == "verify" else args.operation, args.fixture,
                              args.families, args.diagnostic_limit, event, cancel)
            event({"event": "data_pilot_done", "status": value["status"], "families": value["families"],
                   "roots": value.get("selected_roots", value.get("roots")), "labels": value["labels"]})
        return 0
    except Exception as exc:
        event({"event": "data_pilot_error", "error": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
