"""Freeze/label/train/inspect one bounded ordinary value-only pilot."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_value_pilot as p
from congklak.puct import CancellationToken


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("prepare","plan","freeze","h8","h10","h12","labels","verify-labels","train","inspect","verify"));parser.add_argument("--run");parser.add_argument("--fixture",action="store_true");parser.add_argument("--roots",type=int);parser.add_argument("--steps",type=int);parser.add_argument("--checkpoint-every",type=int);parser.add_argument("--batch-size",type=int);args=parser.parse_args();cancel=CancellationToken()
    signal.signal(signal.SIGINT,lambda *_:cancel.cancel())
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,lambda *_:cancel.cancel())
    def event(value):
        print(config.canonical(value),flush=True);step=value.get("step",value.get("completed"))
        if value["event"]==os.environ.get("ORDINARY_VALUE_BLOCK_EVENT") and str(step)==os.environ.get("ORDINARY_VALUE_BLOCK_STEP"):
            p.teacher.teacher.atomic_json(Path(os.environ["ORDINARY_VALUE_MARKER"]),value)
            while True:time.sleep(.05)
    try:
        if args.operation=="prepare":
            manifest,groups=p.prepare();event(dict(event="ordinary_value_data",groups=manifest["groups"],paired_starts=manifest["paired_starts"],training_admission=manifest["training_admission"]))
        elif args.operation in ("plan","freeze"):
            value=p.plan() if args.operation=="plan" else p.freeze();event(dict(event="ordinary_value_"+args.operation,sha256=config.digest(value)))
        elif args.operation in ("h8","h10","h12","labels"):
            value=p.run_labels(args.run or p.LABEL_DEFAULT,"all" if args.operation=="labels" else args.operation,args.fixture,args.roots,event,cancel);event(dict(event="ordinary_value_label_done",status=value["status"]))
        elif args.operation=="verify-labels":
            roots,stores,body=p.load_labels(args.run or p.LABEL_DEFAULT);event(dict(event="ordinary_value_labels_verified",summary=body["summary"],torch_imported="torch" in sys.modules))
        elif args.operation=="train":
            value=p.train(args.run or p.DEFAULT,args.fixture,args.steps,args.checkpoint_every,args.batch_size,event,cancel);event(dict(event="ordinary_value_train_done",status=value["status"],steps=value["steps_completed"]))
        elif args.operation=="inspect":
            value=p.inspect_run(args.run or p.DEFAULT,None if args.fixture else p.WITNESS,args.fixture);event(dict(event="ordinary_value_inspected",selected_step=value["selected_step"],checkpoints_verified=value["checkpoints_verified"]))
        else:
            value=p.verify_run(args.run or p.DEFAULT,args.fixture);event(dict(event="ordinary_value_cpu_verified",selected_step=value["selected_step"],checkpoints_verified=value["checkpoints_verified"],torch_imported="torch" in sys.modules))
        return 0
    except Exception as exc:event(dict(event="ordinary_value_refused",error=str(exc)));return 2


if __name__=="__main__":raise SystemExit(main())
