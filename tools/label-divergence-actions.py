"""Plan, freeze, resume and verify requested five-root diagnostic teacher labels."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,divergence_teacher as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","freeze","h8","h10","h12","all","verify"));parser.add_argument("--run",default=str(p.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int);args=parser.parse_args();cancel=CancellationToken();signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("DIVERGENCE_TEACHER_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("DIVERGENCE_TEACHER_BLOCK_STEP"):
            p.history.teacher.atomic_json(Path(os.environ["DIVERGENCE_TEACHER_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            body=p.plan() if args.operation=="plan" else p.freeze();event(dict(event="divergence_teacher_"+args.operation,sha256=config.digest(body)))
        elif args.operation=="verify":
            _,_,body=p.load_rows(args.run);event(dict(event="divergence_teacher_verified",summary=body["summary"]))
        else:p.run(args.run,args.operation,args.fixture,args.roots,event,cancel)
        return 0
    except Exception as exc:event(dict(event="divergence_teacher_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
