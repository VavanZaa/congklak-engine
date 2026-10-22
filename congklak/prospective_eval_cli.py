"""Bounded prospective evaluation supervisor; no training entry point."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))

def main(argv=None):
 from congklak import prospective_eval_runtime as resources, prospective_eval_common as s
 parser=argparse.ArgumentParser()
 parser.add_argument('command',choices=('execute','completed','status','watchdog','recovery'))
 parser.add_argument('--operation',choices=('endgame','arena'))
 parser.add_argument('--output',type=Path);parser.add_argument('--cache',type=Path,default=s.DATA/'cache')
 parser.add_argument('--lease',type=Path);parser.add_argument('--engineering-fixture',action='store_true')
 parser.add_argument('--pause-at');parser.add_argument('--pause-marker',type=Path)
 args=parser.parse_args(argv)
 if args.command=='watchdog':answer=resources.watchdog(args.lease)
 elif args.command=='status':answer=resources.status()
 elif args.command=='recovery':
  from congklak import prospective_eval_recovery
  answer=prospective_eval_recovery.main()
 else:
  if args.operation is None:parser.error('--operation required')
  output=args.output or s.FOLDER/'jobs'/args.operation
  if args.command=='completed':
   from congklak import prospective_eval_execute as execute
   answer=execute.completed_job(output,args.operation)
  else:answer=resources.launch(output,args.cache,fixture=args.engineering_fixture,
    pause_at=args.pause_at,pause_marker=args.pause_marker,extra_args=['--operation',args.operation])
 print(json.dumps(answer,sort_keys=True),flush=True);return answer
if __name__=='__main__':main()
