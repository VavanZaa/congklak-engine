"""Direct supervisor for the prospectively extended remaining arena capacity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
def main(argv=None):
 from congklak import prospective_eval_common as s, prospective_eval_runtime_v3 as r, prospective_eval_execute_v3 as e
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('execute','completed','status','watchdog'))
 parser.add_argument('--operation',choices=('arena','endgame'),default='arena');parser.add_argument('--output',type=Path)
 parser.add_argument('--cache',type=Path,default=s.DATA/'cache');parser.add_argument('--lease',type=Path)
 args=parser.parse_args(argv)
 if args.command=='watchdog':answer=r.watchdog(args.lease)
 elif args.command=='status':answer=r.status()
 elif args.command=='completed':answer=e.completed_job(args.output or s.FOLDER/'jobs'/args.operation,args.operation)
 else:
  s.need(args.operation=='arena','Completed endgames cannot be repeated')
  answer=r.launch(args.output or s.FOLDER/'jobs/arena',args.cache,extra_args=['--operation','arena'])
 print(json.dumps(answer,sort_keys=True),flush=True);return answer
if __name__=='__main__':main()
