"""Freeze metadata, resume finite-reference stages and inspect fixed-cell metrics."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,stratum_weighting as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("prepare","plan","freeze","run","all","h8","h10","h12","realize","freeze-rule","verify-labels","assess","verify"))
    parser.add_argument("--stage",choices=("all","h8","h10","h12"),default="all");parser.add_argument("--run",default=str(p.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int);parser.add_argument("--allow-partial",action="store_true");args=parser.parse_args()
    cancel=CancellationToken();signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("STRATUM_WEIGHTING_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("STRATUM_WEIGHTING_BLOCK_STEP"):
            p.history.teacher.atomic_json(Path(os.environ["STRATUM_WEIGHTING_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation=="prepare":body=p.prepare();event(dict(event="stratum_weighting_prepared",roots=len(body["rows"]),registry_sha256=p.sha(p.REGISTRY)))
        elif args.operation in ("plan","freeze"):
            body=p.plan() if args.operation=="plan" else p.freeze();event(dict(event="stratum_weighting_"+args.operation,sha256=config.digest(body)))
        elif args.operation in ("run","all","h8","h10","h12"):
            body=p.run(args.run,args.stage if args.operation=="run" else args.operation, args.fixture,args.roots,event,cancel);event(dict(event="stratum_weighting_done",status=body["status"]))
        elif args.operation in ("realize","freeze-rule"):
            body=p.realize(args.run);event(dict(event="stratum_weighting_rule_frozen",fixture=body["fixture"],rule_sha256=p.sha(p.rule_path(args.run,body["fixture"]))))
        elif args.operation=="verify-labels":
            _,_,body=p.load_rows(args.run,args.allow_partial);event(dict(event="stratum_weighting_labels_verified",status=body["status"],summary=body["summary"]))
        else:
            body=p.assess() if args.operation=="assess" else p.verify();event(dict(event="stratum_weighting_assessed" if args.operation=="assess" else "stratum_weighting_verified",requested=body["requested_roots"],screen=body["screen"]))
        return 0
    except Exception as exc:event(dict(event="stratum_weighting_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
