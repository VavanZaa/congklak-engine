"""Freeze, collect and generate one data-only ordinary coverage pilot."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_selfplay_pilot as p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","freeze","collect","generate","all","verify"));parser.add_argument("--run",default=str(p.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--families",type=int);parser.add_argument("--decision-limit",type=int)
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,stop)
    def event(value):
        print(config.canonical(value),flush=True);step=value.get("completed",value.get("committed_decisions"))
        if value["event"]==os.environ.get("ORDINARY_SELFPLAY_BLOCK_EVENT") and str(step)==os.environ.get("ORDINARY_SELFPLAY_BLOCK_STEP"):
            p.histories.teacher.atomic_json(Path(os.environ["ORDINARY_SELFPLAY_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation=="plan":value=p.plan();event(dict(event="ordinary_selfplay_plan",sha256=config.digest(value)))
        elif args.operation=="freeze":value=p.freeze();event(dict(event="ordinary_selfplay_freeze",sha256=config.digest(value)))
        elif args.operation=="verify":
            value,rows=p.load_replay(args.run);event(dict(event="ordinary_selfplay_verified",summary=value["summary"],rows=len(rows),torch_imported="torch" in sys.modules))
        elif args.operation=="collect":p.collect(args.run,args.fixture,args.families,event,lambda:stopped)
        elif args.operation=="generate":p.generate(args.run,args.fixture,args.families,event,lambda:stopped,args.decision_limit)
        else:p.run(args.run,args.fixture,args.families,"all",event,lambda:stopped)
        return 0
    except Exception as exc:event(dict(event="ordinary_selfplay_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
