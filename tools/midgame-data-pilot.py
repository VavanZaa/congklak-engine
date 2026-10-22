"""Plan, freeze, collect and resume the bounded targeted midgame pool."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,midgame_data_pilot as pilot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","freeze","collect","all","verify"));parser.add_argument("--run",default=str(pilot.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--families",type=int)
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,stop)
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("MIDGAME_COLLECTION_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("MIDGAME_COLLECTION_BLOCK_STEP"):
            pilot.teacher.atomic_json(Path(os.environ["MIDGAME_COLLECTION_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            value=pilot.plan() if args.operation=="plan" else pilot.freeze();event(dict(event="midgame_collection_"+args.operation,sha256=config.digest(value)))
        elif args.operation=="verify":
            rows,value=pilot.load_rows(args.run);event(dict(event="midgame_collection_verified",families=value["families"],selected_roots=len(rows),torch_imported="torch" in sys.modules))
        else:
            value=pilot.collect(args.run,args.fixture,args.families,event,lambda:stopped);event(dict(event="midgame_collection_done",status=value["status"],families=value["families"],selected_roots=value["selected_roots"]))
        return 0
    except Exception as exc:event(dict(event="midgame_collection_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
