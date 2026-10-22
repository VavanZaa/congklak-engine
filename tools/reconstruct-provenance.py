"""Metadata-only provenance API driver; recoverable publication is separate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak import provenance_reconstruction as provenance
from congklak import neurodiag_common as common


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('plan','unit','summarize'))
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--index',type=int)
    parser.add_argument('--units',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    common.require(out.is_relative_to(provenance.DEFAULT.resolve()),
                   'Provenance API output must use new implementation namespace')
    if args.operation=='plan':body=provenance.plan()
    elif args.operation=='unit':
        common.require(args.plan is not None and args.index is not None,'Unit requires plan and index')
        plan=common.read(args.plan);common.current(plan['inputs'])
        body=provenance.unit(plan['tasks'][args.index])
    else:
        common.require(args.units is not None,'Summary requires immutable unit directory')
        units=[common.read(p) for p in sorted(args.units.glob('*.json'))]
        body=provenance.summarize(units)
    common.publish(out,body)
    print(json.dumps(dict(operation=args.operation,path=str(out),sha256=common.sha(out),schema=body['schema'])))


if __name__=='__main__':main()
