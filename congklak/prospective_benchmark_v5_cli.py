"""Explicit fixed-purpose supervisor for the prospective benchmark."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def main(argv=None):
 from congklak import prospective_benchmark_v5_common as s,prospective_benchmark_v5_runtime as r,prospective_benchmark_v5_execute as e
 p=argparse.ArgumentParser();p.add_argument('command',choices=('execute','completed','status','watchdog'))
 p.add_argument('--operation',choices=e.OPERATIONS,default='engineering');p.add_argument('--output',type=Path)
 p.add_argument('--cache',type=Path,default=s.DATA/'cache');p.add_argument('--lease',type=Path)
 p.add_argument('--engineering-fixture',action='store_true');p.add_argument('--pause-at');p.add_argument('--pause-marker',type=Path)
 a=p.parse_args(argv)
 if a.command=='watchdog':answer=r.watchdog(a.lease)
 elif a.command=='status':answer=r.status()
 elif a.command=='completed':answer=e.completed_job(a.output or s.FOLDER/'jobs'/a.operation,a.operation)
 else:
  answer=r.launch(a.output or s.FOLDER/'jobs'/a.operation,a.cache,fixture=a.engineering_fixture,
   pause_at=a.pause_at,pause_marker=a.pause_marker,extra_args=['--operation',a.operation])
 print(json.dumps(answer,sort_keys=True),flush=True);return answer
if __name__=='__main__':main()
