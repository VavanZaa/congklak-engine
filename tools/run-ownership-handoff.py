"""Launch contained traced-handoff engineering fixtures or verify saved evidence."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ownership_handoff_worker as worker,ownership_handoff_resources as resources

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','worker','verify','resources','watchdog'))
    parser.add_argument('--output',type=Path,default=worker.DEFAULT);parser.add_argument('--spec',type=Path,default=worker.SPEC)
    parser.add_argument('--ledger',type=Path,default=resources.DEFAULT);parser.add_argument('--lease',type=Path)
    parser.add_argument('--pause-at');parser.add_argument('--pause-marker',type=Path)
    args=parser.parse_args()
    try:
        if args.operation=='verify' or (args.output/'catalog.json').exists():result=worker.completed(args.output)
        elif args.operation=='resources':result=resources.status(args.ledger)
        elif args.operation=='watchdog':result=resources.watchdog(args.lease)
        elif args.operation=='prepare':result=worker.prepare(args.output,args.spec)
        elif args.operation=='worker':
            resources.bind_worker(args.lease)
            result=worker.run(args.output,args.spec,args.lease,args.pause_at,args.pause_marker)
            resources.worker_complete(args.lease,dict(completed=True,engineering_games=result['report']['games']))
        else:
            worker.prepare(args.output,args.spec)
            result=resources.launch(args.output,args.spec,ledger=args.ledger,pause_at=args.pause_at,pause_marker=args.pause_marker)
        print(config.canonical(result),flush=True);return 0
    except Exception as exc:
        print(config.canonical(dict(state='hold',error=type(exc).__name__+': '+str(exc))),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
