"""Collect and label the bounded natural ordinary H8/H10 stability study."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config, natural_study as study
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","freeze","collect","h8","h10","all","verify"))
    parser.add_argument("--run",default=str(study.DEFAULT));parser.add_argument("--fixture",action="store_true")
    parser.add_argument("--families",type=int);parser.add_argument("--followups",type=int)
    args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        step=value.get("index") if "family_" in value["event"] else value.get("completed")
        if value["event"]==os.environ.get("NATURAL_STUDY_BLOCK_EVENT") and str(step)==os.environ.get("NATURAL_STUDY_BLOCK_STEP"):
            study.teacher.atomic_json(Path(os.environ["NATURAL_STUDY_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            value=study.plan() if args.operation=="plan" else study.freeze()
            event({"event":"natural_study_"+args.operation,"sha256":config.digest(value),"fixed_model_sha256":study.MODEL_SHA})
        else:
            if args.operation=="verify":study.require((Path(args.run)/"manifest.json").exists(),"Study incomplete; use all to resume or repair publication")
            result=study.run(args.run,"all" if args.operation=="verify" else args.operation,args.fixture,args.families,args.followups,event,cancel)
            event({"event":"natural_study_done","status":result["status"],"families":result["families"],"selected_fresh":result["selected_fresh"],"h10_labels":result["h10_labels"],"h10_rejected":result["h10_rejected"]})
        return 0
    except Exception as exc:
        event({"event":"natural_study_error","error":str(exc)});return 2


if __name__=="__main__":raise SystemExit(main())
