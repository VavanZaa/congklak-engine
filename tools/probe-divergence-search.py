"""Frozen five-position fixed-simulation probe; no training or strength games."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import divergence_search as search

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('prepare','freeze','admit','run','verify','witness'));parser.add_argument('--path',type=Path,default=search.DEFAULT);parser.add_argument('--fixture',action='store_true');parser.add_argument('--variant',choices=('normal','node_cap'),default='normal');parser.add_argument('--cell-limit',type=int);args=parser.parse_args()
    if args.operation=='prepare':body=search.prepare()
    elif args.operation=='freeze':body=search.freeze()
    elif args.operation=='admit':body=search.admit(args.fixture,args.variant)[-1]
    elif args.operation=='witness':body=search.primary_witness()
    elif args.operation=='verify':body=search.verify(args.path,args.fixture,args.variant)
    else:body=search.run(args.path,args.fixture,args.variant,args.cell_limit)
    print(json.dumps({key:body[key] for key in ('schema','status','scope_finished','counts','sha256','passed','complete_cells','completed_simulations') if key in body},sort_keys=True))
