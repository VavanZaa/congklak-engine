"""Run the frozen acquisition with recoverable bounded native workers."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import footprint_screen_common as s,footprint_screen_resources as r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('run','review','reserve','watchdog','status'))
    p.add_argument('--output',type=Path)
    p.add_argument('--lease',type=Path)
    p.add_argument('--engineering-fixture',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--pause-at');p.add_argument('--pause-marker',type=Path)
    a=p.parse_args()
    if a.output is None:a.output=s.FOLDER/({'review':'audit/raw-review','reserve':'reservations'}.get(a.operation,'acquisition'))
    if a.operation=='watchdog':
        if a.lease is None:p.error('--lease required')
        result=r.watchdog(a.lease)
    elif a.operation=='status':result=r.status()
    else:result=r.launch(a.output,fixture=a.engineering_fixture,stop_after=a.stop_after,
        pause_at=a.pause_at,pause_marker=a.pause_marker,
        entry={'review':r.AUDIT_ENTRY,'reserve':r.RESERVATION_ENTRY}.get(a.operation,r.ENTRY))
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
