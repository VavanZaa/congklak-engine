"""Read-only reconstruction of the retained ordinary value pilot games."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import ordinary_value_loss_review as review

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('build','verify'));parser.add_argument('--output',type=Path,default=review.OUTPUT);args=parser.parse_args()
    body=review.run(args.output) if args.operation=='build' else review.verify(args.output)
    print(json.dumps(dict(schema=body['schema'],passed=body['passed'],counts=body['counts'],summary={name:{key:value for key,value in item.items() if key not in ('leave_one_family_out_means','inherited_exploratory_family_interval')} for name,item in body['summary'].items()}),sort_keys=True))
