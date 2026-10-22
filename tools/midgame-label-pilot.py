"""Freeze and resume bounded full-root midgame diagnostic/data targets."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,midgame_label_pilot as labels
from congklak.puct import CancellationToken


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('plan','freeze','h8','h10','h12','all','verify'));p.add_argument('--run',default=str(labels.DEFAULT));p.add_argument('--collection');p.add_argument('--fixture',action='store_true');p.add_argument('--diagnostics',type=int);p.add_argument('--fresh-roots',type=int);args=p.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True)
        if value['event']==os.environ.get('MIDGAME_LABEL_BLOCK_EVENT') and str(value.get('completed'))==os.environ.get('MIDGAME_LABEL_BLOCK_STEP'):
            labels.teacher.atomic_json(Path(os.environ['MIDGAME_LABEL_MARKER']),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ('plan','freeze'):
            value=labels.plan() if args.operation=='plan' else labels.freeze();event(dict(event='midgame_label_'+args.operation,sha256=config.digest(value)))
        elif args.operation=='verify':
            rows,stores,value=labels.load_rows(args.run);event(dict(event='midgame_label_verified',**labels.counts(rows,stores)))
        else:
            value=labels.run(args.run,args.operation,args.fixture,args.collection,args.diagnostics,args.fresh_roots,event,cancel);event(dict(event='midgame_label_done',status=value['status']))
        return 0
    except Exception as exc:event(dict(event='midgame_label_refused',error=str(exc)));return 2


if __name__=='__main__':raise SystemExit(main())
