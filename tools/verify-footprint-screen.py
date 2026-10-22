"""Derive, seal or explicitly verify the complete acquisition evidence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import footprint_screen_finish as finish

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('report','seal','verify'));a=p.parse_args()
    result=getattr(finish,a.operation)()
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','compact_inputs','archival_inputs','live_inputs')},
        sort_keys=True,allow_nan=False))
