"""Freeze, run and verify the one small CPU value-head pilot."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,value_pilot_learning as learning


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("prepare","plan","freeze","train","inspect","verify"));parser.add_argument("--run",default=str(learning.DEFAULT))
    parser.add_argument("--fixture",action="store_true");parser.add_argument("--steps",type=int);parser.add_argument("--checkpoint-every",type=int);parser.add_argument("--batch-size",type=int)
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,stop)
    def event(value):
        print(config.canonical(value),flush=True)
        if value["event"]==os.environ.get("VALUE_PILOT_BLOCK_EVENT") and str(value.get("step"))==os.environ.get("VALUE_PILOT_BLOCK_STEP"):
            learning.old.teacher.atomic_json(Path(os.environ["VALUE_PILOT_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation=="prepare":
            manifest,groups=learning.prepare();event({"event":"value_pilot_data_admitted","train_rows":len(groups["train"]),"group_counts":{key:len(rows) for key,rows in groups.items()},"manifest_sha256":config.digest(manifest),"torch_imported":"torch" in sys.modules})
        elif args.operation in ("plan","freeze"):
            value=learning.plan() if args.operation=="plan" else learning.freeze();event({"event":"value_pilot_training_"+args.operation,"sha256":config.digest(value)})
        elif args.operation in ("verify","inspect"):
            value=learning.inspect_run(args.run,None if args.fixture else learning.WITNESS,args.fixture) if args.operation=="inspect" else learning.verify_run(args.run,fixture=args.fixture)
            event({"event":"value_pilot_model_"+args.operation+"_verified","selected_step":value["selected_step"],"checkpoints_verified":value["checkpoints_verified"],"weights_sha256":value["model_weights_sha256"],"torch_imported":"torch" in sys.modules})
        else:
            value=learning.train(args.run,event,lambda:stopped,args.fixture,args.steps,args.checkpoint_every,args.batch_size)
            event({"event":"value_pilot_training_done","status":value["status"],"step":value["steps_completed"],"weights_sha256":value.get("weights_sha256")})
        return 0
    except Exception as exc:event({"event":"value_pilot_training_error","error":str(exc)});return 2


if __name__=="__main__":raise SystemExit(main())
