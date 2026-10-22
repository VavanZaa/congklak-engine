"""Bounded supervisor for the ordinary-only value experiment."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import guarded_value_pilot_v2_resources as r
p=argparse.ArgumentParser();p.add_argument('command',choices=('run','watchdog','status','execute'))
p.add_argument('--output',type=Path)
p.add_argument('--cache',type=Path,default=r.ROOT/'data/guarded-value-pilot-v2/cache')
p.add_argument('--ledger',type=Path,default=r.DEFAULT);p.add_argument('--lease',type=Path)
p.add_argument('--engineering-fixture',action='store_true');p.add_argument('--stop-after',type=int)
p.add_argument('--pause-at');p.add_argument('--pause-marker',type=Path)
p.add_argument('--operation',choices=('arena','conformance','guards'))
a=p.parse_args()
if a.output is None:
    a.output=(r.ROOT/'build/guarded-value-pilot-v2/jobs'/a.operation if a.command=='execute' and a.operation
              else r.ROOT/'data/guarded-value-pilot-v2/training')
if a.command=='watchdog':result=r.watchdog(a.lease)
elif a.command=='status':result=r.status(a.ledger)
elif a.command=='execute':
    if a.operation is None:p.error('--operation required')
    result=r.launch(a.output,a.cache,a.engineering_fixture,a.stop_after,a.ledger,
                    pause_at=a.pause_at,pause_marker=a.pause_marker,
                    entry=r.ROOT/'tools/guarded-value-pilot-v2-execute.py',extra_args=['--operation',a.operation])
else:
    from congklak import guarded_value_pilot_v2_train as t
    result=t.completed(a.output) if t._has_completion(a.output) else r.launch(
        a.output,a.cache,a.engineering_fixture,a.stop_after,a.ledger,
        pause_at=a.pause_at,pause_marker=a.pause_marker)
print(json.dumps(result,sort_keys=True))
