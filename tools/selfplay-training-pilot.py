"""Freeze, generate, train and verify one bounded CPU self-play iteration."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,selfplay_training_pilot as pilot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","freeze","generate","replay","train","inspect","verify"));parser.add_argument("--run",default=str(pilot.DEFAULT));parser.add_argument("--replay",default=str(pilot.DEFAULT));parser.add_argument("--fixture",action="store_true");parser.add_argument("--families",type=int);parser.add_argument("--steps",type=int);parser.add_argument("--checkpoint-every",type=int);parser.add_argument("--batch-size",type=int);parser.add_argument("--decision-limit",type=int)
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,stop)
    def event(value):
        print(config.canonical(value),flush=True);position=value.get("step",value.get("committed_decisions"))
        if value["event"]==os.environ.get("SELFPLAY_PILOT_BLOCK_EVENT") and str(position)==os.environ.get("SELFPLAY_PILOT_BLOCK_STEP"):
            pilot.quality_learning.teacher.atomic_json(Path(os.environ["SELFPLAY_PILOT_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation in ("plan","freeze"):
            value=pilot.plan() if args.operation=="plan" else pilot.freeze();event(dict(event="selfplay_pilot_"+args.operation,sha256=config.digest(value)))
        elif args.operation=="generate":
            value=pilot.generate(args.run,args.fixture,args.families,event,lambda:stopped,args.decision_limit);event(dict(event="selfplay_pilot_generation_done",status=value.get("status"),summary=value.get("summary"),committed_decisions=value.get("committed_decisions")))
        elif args.operation=="replay":
            manifest,rows=pilot.load_replay(args.replay);event(dict(event="selfplay_pilot_replay_verified",summary=manifest["summary"],rows=len(rows),torch_imported="torch" in sys.modules))
        elif args.operation=="train":
            value=pilot.train(args.run,args.replay,args.fixture,args.steps,args.checkpoint_every,args.batch_size,event,lambda:stopped);event(dict(event="selfplay_pilot_training_done",status=value["status"],step=value["steps_completed"],weights_sha256=value.get("weights_sha256")))
        elif args.operation=="inspect":
            value=pilot.inspect_run(args.run,None if args.fixture else pilot.WITNESS,args.fixture);event(dict(event="selfplay_pilot_tensor_inspected",selected_step=value["selected_step"],checkpoints_verified=value["checkpoints_verified"],model_weights_sha256=value["model_weights_sha256"]))
        else:
            value=pilot.verify_run(args.run,args.fixture);event(dict(event="selfplay_pilot_cpu_verified",selected_step=value["selected_step"],checkpoints_verified=value["checkpoints_verified"],torch_imported="torch" in sys.modules))
        return 0
    except Exception as exc:event(dict(event="selfplay_pilot_error",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
