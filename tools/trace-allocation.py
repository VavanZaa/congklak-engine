"""Frozen pair7 trace; completed verification activates no inference/search."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import allocation_trace as trace

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('prepare','freeze','admit','run','verify','witness'));p.add_argument('--path',type=Path,default=trace.DEFAULT);p.add_argument('--fixture',action='store_true');p.add_argument('--variant',choices=('normal','node_cap'),default='normal');p.add_argument('--simulation-limit',type=int);a=p.parse_args()
    if a.operation=='prepare':body=trace.prepare()
    elif a.operation=='freeze':body=trace.freeze()
    elif a.operation=='admit':body=trace.admit(a.fixture,a.variant)[-1]
    elif a.operation=='witness':body=trace.primary_witness()
    elif a.operation=='verify':body=trace.verify(a.path,a.fixture,a.variant)
    else:body=trace.run(a.path,a.fixture,a.variant,a.simulation_limit)
    print(json.dumps({k:body[k] for k in ('schema','status','scope_finished','counts','sha256','passed','complete_cells','completed_root_simulations') if k in body},sort_keys=True))
