"""Independent watchdog / resource supervisor for value-only training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import value_training_resources as resources
p=argparse.ArgumentParser();p.add_argument('command',choices=('run','watchdog','status'))
p.add_argument('--output',type=Path,default=resources.ROOT/'data/value-training-v1/training')
p.add_argument('--cache',type=Path,default=resources.ROOT/'data/value-training-v1/cache')
p.add_argument('--ledger',type=Path,default=resources.DEFAULT);p.add_argument('--lease',type=Path)
p.add_argument('--engineering-fixture',action='store_true');p.add_argument('--stop-after',type=int)
p.add_argument('--pause-at',choices=('before_checkpoint_catalog','after_checkpoint_catalog'));p.add_argument('--pause-marker',type=Path)
a=p.parse_args()
if a.command=='watchdog':
    if a.lease is None:p.error('--lease required')
    result=resources.watchdog(a.lease)
elif a.command=='status':
    result=resources.status(a.ledger)
else:
    # All completed checks occur before resource writers, watchdogs or Torch.
    from congklak import value_training as training
    if training._has_completion(a.output):
        result=training.completed(a.output)
    else:
        result=resources.launch(a.output,a.cache,a.engineering_fixture,a.stop_after,a.ledger,pause_at=a.pause_at,pause_marker=a.pause_marker)
print(json.dumps(result,sort_keys=True))
