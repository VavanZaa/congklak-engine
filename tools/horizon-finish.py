"""Plan, freeze and resume exactly five capped H14 roots with bounded tiers."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,horizon_finish as finish
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","freeze","tier1","tier2","all","verify"));parser.add_argument("--run",default=str(finish.DEFAULT))
    parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int)
    args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("HORIZON_FINISH_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("HORIZON_FINISH_BLOCK_STEP"):
            finish.teacher.atomic_json(Path(os.environ["HORIZON_FINISH_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            value=finish.plan() if args.operation=="plan" else finish.freeze();event({"event":"horizon_finish_"+args.operation,"sha256":config.digest(value),"fixed_model_sha256":finish.MODEL_SHA})
        else:
            if args.operation=="verify":finish.require((Path(args.run)/"manifest.json").exists(),"Finish incomplete; use all to resume or repair publication")
            value=finish.run(args.run,"all" if args.operation=="verify" else args.operation,args.fixture,args.roots,event,cancel)
            event({"event":"horizon_finish_done","status":value["status"],"target_roots":value["target_roots"],"new_h14_labels":value["new_h14_labels"],"unresolved_target_roots":value["unresolved_target_roots"]})
            if value["status"]=="budget_exhausted":return 3
        return 0
    except Exception as exc:event({"event":"horizon_finish_error","error":str(exc)});return 2


if __name__=="__main__":raise SystemExit(main())
