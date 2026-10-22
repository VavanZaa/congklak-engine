"""Supervise the frozen192-game value pilot or inspect persisted evidence."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,value_run_arena as arena,value_run_resources as resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','worker','verify','status','resources','watchdog'))
    parser.add_argument('--output',default=str(arena.DEFAULT));parser.add_argument('--candidate',default=str(arena.CANDIDATE))
    parser.add_argument('--ledger',default=str(resources.DEFAULT));parser.add_argument('--lease')
    parser.add_argument('--engineering-fixture',action='store_true');parser.add_argument('--fixed-simulations',type=int)
    parser.add_argument('--stop-after-decisions',type=int);parser.add_argument('--pause-at');parser.add_argument('--pause-marker')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    try:
        if args.operation=='verify':answer=arena.verify(args.output)
        elif args.operation=='status':answer=arena.status(args.output)
        elif args.operation=='prepare':answer=arena.prepare(args.output,args.candidate,args.engineering_fixture)
        elif args.operation=='resources':answer=resources.status(args.ledger)
        elif args.operation=='watchdog':answer=resources.watchdog(args.lease)
        elif (Path(args.output)/'certificate.json').exists() or (Path(args.output)/'COMPLETED.json').exists():answer=arena.verify(args.output)
        elif args.operation=='worker':
            if args.lease:resources.bind_worker(args.lease)
            answer=arena.run(args.output,args.candidate,args.engineering_fixture,args.fixed_simulations,
                args.stop_after_decisions,args.pause_at,args.pause_marker,args.lease)
            if args.lease:resources.worker_complete(args.lease,dict(games=answer.get('games',answer.get('games_committed')),completed=answer.get('completed',False)))
        else:
            arena.prepare(args.output,args.candidate,args.engineering_fixture)
            while True:
                launch=resources.launch(args.output,args.candidate,fixture=args.engineering_fixture,
                    stop_after=args.stop_after_decisions,ledger=args.ledger,pause_at=args.pause_at,pause_marker=args.pause_marker)
                print(config.canonical(dict(event='arena_lease_settled',**launch)),flush=True)
                need=arena.need;need(launch['worker_returncode']==0 and launch['watchdog_returncode']==0,'Arena worker/watchdog failed; retained prefix remains available')
                if (Path(args.output)/'certificate.json').exists():answer=arena.verify(args.output);break
                answer=arena.status(args.output)
                if args.once or args.stop_after_decisions is not None:break
        print(config.canonical(answer),flush=True);return 0
    except Exception as error:
        print(config.canonical(dict(state='hold',error=type(error).__name__+': '+str(error))),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
