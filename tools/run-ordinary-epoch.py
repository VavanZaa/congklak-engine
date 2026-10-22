"""Bounded supervisor for the ordinary-only value experiment."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import ordinary_epoch_resources as r
p=argparse.ArgumentParser();p.add_argument('command',choices=('run','watchdog','status','execute'))
p.add_argument('--output',type=Path)
p.add_argument('--cache',type=Path,default=r.ROOT/'data/ordinary-epoch-pilot-v1/cache-v2')
p.add_argument('--ledger',type=Path,default=r.DEFAULT);p.add_argument('--lease',type=Path)
p.add_argument('--engineering-fixture',action='store_true');p.add_argument('--stop-after',type=int)
p.add_argument('--pause-at');p.add_argument('--pause-marker',type=Path)
p.add_argument('--operation',choices=('labels','arena','conformance','guards'))
a=p.parse_args()
if a.output is None:
    a.output=(r.ROOT/'build/ordinary-epoch-pilot-v1/jobs'/a.operation if a.command=='execute' and a.operation
              else r.ROOT/'data/ordinary-epoch-pilot-v1/training-v2')
if a.command=='watchdog':result=r.watchdog(a.lease)
elif a.command=='status':result=r.status(a.ledger)
elif a.command=='execute':
    if a.operation is None:p.error('--operation required')
    result=r.launch(a.output,a.cache,a.engineering_fixture,a.stop_after,a.ledger,
                    pause_at=a.pause_at,pause_marker=a.pause_marker,
                    entry=r.ROOT/'tools/ordinary-epoch-execute.py',extra_args=['--operation',a.operation])
else:
    from congklak import ordinary_epoch_training as t
    result=t.completed(a.output) if t._has_completion(a.output) else r.launch(
        a.output,a.cache,a.engineering_fixture,a.stop_after,a.ledger,
        pause_at=a.pause_at,pause_marker=a.pause_marker)
print(json.dumps(result,sort_keys=True))
