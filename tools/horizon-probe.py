"""Plan, freeze and resume six investigation-only H12/H14 horizon probes."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,horizon_probe as probe
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","freeze","h12","h14","all","verify"));parser.add_argument("--run",default=str(probe.DEFAULT))
    parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int)
    args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("HORIZON_PROBE_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("HORIZON_PROBE_BLOCK_STEP"):
            probe.teacher.atomic_json(Path(os.environ["HORIZON_PROBE_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            value=probe.plan() if args.operation=="plan" else probe.freeze();event({"event":"horizon_probe_"+args.operation,"sha256":config.digest(value),"fixed_model_sha256":probe.MODEL_SHA})
        else:
            if args.operation=="verify":probe.require((Path(args.run)/"manifest.json").exists(),"Probe incomplete; use all to resume or repair publication")
            value=probe.run(args.run,"all" if args.operation=="verify" else args.operation,args.fixture,args.roots,event,cancel)
            event({"event":"horizon_probe_done","status":value["status"],"roots":value["roots"],"h12_labels":value["h12_labels"],"h14_labels":value["h14_labels"]})
        return 0
    except Exception as exc:event({"event":"horizon_probe_error","error":str(exc)});return 2


if __name__=="__main__":raise SystemExit(main())
