"""Freeze a separate weighting study, resume H12 attempts and assess saved references."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,allocation_weighting as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("prepare","plan","freeze","run","all","h12","verify-labels","assess","verify"));parser.add_argument("--run",default=str(p.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int);args=parser.parse_args();cancel=CancellationToken();signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("ALLOCATION_WEIGHTING_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("ALLOCATION_WEIGHTING_BLOCK_STEP"):
            p.history.teacher.atomic_json(Path(os.environ["ALLOCATION_WEIGHTING_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation=="prepare":r,w=p.prepare();event(dict(event="allocation_weighting_prepared",roots=len(r["rows"]),registry_sha256=p.sha(p.REGISTRY),rule_sha256=p.sha(p.RULE)))
        elif args.operation in ("plan","freeze"):
            body=p.plan() if args.operation=="plan" else p.freeze();event(dict(event="allocation_weighting_"+args.operation,sha256=config.digest(body)))
        elif args.operation in ("run","all","h12"):
            body=p.run(args.run,args.fixture,args.roots,event,cancel);event(dict(event="allocation_weighting_done",status=body["status"]))
        elif args.operation=="verify-labels":_,_,body=p.load_rows(args.run);event(dict(event="allocation_weighting_labels_verified",summary=body["summary"]))
        else:
            body=p.assess() if args.operation=="assess" else p.verify();event(dict(event="allocation_weighting_assessed" if args.operation=="assess" else "allocation_weighting_verified",requested=body["requested_roots"],complete=body["complete_roots"],screen=body["screen"]))
        return 0
    except Exception as exc:event(dict(event="allocation_weighting_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
