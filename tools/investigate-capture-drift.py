"""Freeze and resume an existing-data capture drift investigation; no engine calls."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,capture_drift as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("prepare","plan","freeze","run","all","verify-cases","assess","verify"));parser.add_argument("--run",default=str(p.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--cases",type=int);args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("CAPTURE_DRIFT_BLOCK_EVENT") and str(value.get("completed"))==os.environ.get("CAPTURE_DRIFT_BLOCK_STEP"):
            p.publish(Path(os.environ["CAPTURE_DRIFT_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("prepare","plan","freeze"):
            body=getattr(p,args.operation)();event(dict(event="capture_drift_"+args.operation,sha256=config.digest(body)))
        elif args.operation in ("run","all"):
            body=p.run(args.run,args.fixture,args.cases,event,cancel);event(dict(event="capture_drift_done",status=body["status"],cases=body["cases"]))
            if args.operation=="all" and not args.fixture and body["status"]=="complete":
                result=p.assess();event(dict(event="capture_drift_assessed",cases=result["cases"],summary=result["summary"]))
        elif args.operation=="verify-cases":
            cases,manifest=p.load_rows(args.run);event(dict(event="capture_drift_cases_verified",cases=len(cases),status=manifest["status"]))
        else:
            body=p.assess() if args.operation=="assess" else p.verify();event(dict(event="capture_drift_"+args.operation,cases=body["cases"],summary=body["summary"]))
        return 0
    except Exception as exc:event(dict(event="capture_drift_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
