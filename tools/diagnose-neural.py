"""Prepare, collect and certify frozen neural diagnostics; no training or search."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','plan','freeze','prepare-inputs','collect','raw','repair-raw','assess','verify'))
    parser.add_argument('--run',default=str(p.DEFAULT));parser.add_argument('--fixture',action='store_true');parser.add_argument('--output',default=str(p.OUTPUT))
    args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(body):
        print(config.canonical(body),flush=True)
        if body['event']==os.environ.get('NEURODIAG_BLOCK_EVENT') and str(body.get('completed'))==os.environ.get('NEURODIAG_BLOCK_STEP'):
            p.publish(Path(os.environ['NEURODIAG_MARKER']),body)
            while True:time.sleep(.05)
    try:
        if args.operation in ('prepare','plan','freeze','prepare-inputs'):
            value=getattr(p,args.operation.replace('-','_'))();event(dict(event='neurodiag_'+args.operation.replace('-','_'),sha256=config.digest(value)))
        elif args.operation=='collect':
            value=p.run(args.run,args.fixture,event,cancel);event(dict(event='neurodiag_collect_done',status=value['status'],units=value.get('unit_count',0),batches=value.get('batch_count',0)))
        elif args.operation in ('raw','repair-raw'):
            value=p.raw_complete(args.run)['manifest'] if args.operation=='raw' else p.repair_raw_exports(args.run)
            event(dict(event='neurodiag_'+args.operation.replace('-','_'),certified=False,units=value['unit_count'],batches=value['batch_count']))
        else:
            value=p.assess(args.output) if args.operation=='assess' else p.verify(args.output)
            event(dict(event='neurodiag_'+args.operation,certified=True,units=value['unit_count'],targets=value['target_records']))
        return 0
    except Exception as exc:event(dict(event='neurodiag_error',error=str(exc)));return 2


if __name__=='__main__':raise SystemExit(main())
