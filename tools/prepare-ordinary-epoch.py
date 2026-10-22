"""Prepare only the frozen192-root existing-target ordinary cache."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import ordinary_epoch_common as s,ordinary_epoch_data as data,ordinary_epoch_sidecar as side

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('metadata','prepare','verify'))
    parser.add_argument('--cache',default=str(data.DEFAULT));args=parser.parse_args()
    if args.operation=='metadata':
        rows=data.membership();body=dict(count=len(rows),cells=data.COUNTS,source_inputs=data.input_pins(),
            targets_read=0,model_forwards=0,optimizer_steps=0)
    elif args.operation=='prepare':
        manifest=data.prepare(args.cache);initial=side.initialize(Path(args.cache)/'manifest.json')
        body=dict(state='complete',cache_manifest_sha256=s.sha(Path(args.cache)/'manifest.json'),
            train_roots=manifest['role_files']['train']['count'],initial=initial,optimizer_steps=0)
    else:
        manifest=data.admit(args.cache);body=dict(state='complete',train_roots=manifest['role_files']['train']['count'],
            initial=side.verify_initial(manifest=Path(args.cache)/'manifest.json'),optimizer_steps=0)
    print(json.dumps(body,sort_keys=True));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(state='hold',error=type(error).__name__+': '+str(error))));raise SystemExit(2)
