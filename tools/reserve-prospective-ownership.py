import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import prospective_reservations as reservations
from congklak import neurodiag_common as common

parser=argparse.ArgumentParser(description='Source-reviewed metadata-only prospective ownership reservations')
parser.add_argument('operation',choices=('binding','run','verify'))
parser.add_argument('--path',type=Path,default=reservations.DEFAULT)
parser.add_argument('--review',type=Path)
parser.add_argument('--out',type=Path)
parser.add_argument('--fixture-draws',type=int)
parser.add_argument('--pause-at',choices=('before_chunk_commit','after_chunk_commit','before_certificate'))
parser.add_argument('--pause-marker',type=Path)
parser.add_argument('--pause-index',type=int,default=0)
args=parser.parse_args()
if args.operation=='binding':
    body=reservations.binding()
    if args.out is not None:common.publish(args.out,body)
    result=dict(schema=body['schema'],binding_sha256=reservations.config.digest(body),candidate_constructor_calls=0)
else:
    body=(reservations.completed(args.path) if args.operation=='verify' else
          reservations.run(args.path,args.review,args.fixture_draws,args.pause_at,args.pause_marker,args.pause_index))
    selection=body['selection']
    result=dict(passed=True,counter_draws=selection['counter_draws'],metadata_reservations_complete=selection['metadata_reservations_complete'],
                training_admission=False,policy_fixed=True,models_fixed=True)
print(json.dumps(result,sort_keys=True),flush=True)
