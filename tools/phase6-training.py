"""Fixed-budget policy warmstart and separately labeled selfplay training."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,deeper_teacher_v2 as teacher

PLAN={"schema":"phase6-training-plan-v1","rules_profile":config.PROFILE,"rules_digest":config.RULES_DIGEST,
      "architecture":{"input_count":127,"width":64,"parameters":12872,"export_schema":"congklak-mlp-export-v4"},
      "warm":{"steps":1000,"batch_size":256,"lr":.001,"seed":20261015,"checkpoint_every":50,"trainable_parameters":455,
              "objective":"Masked root H8 actor-relative softQ cross entropy; tau4 SEEDS","value_and_trunk":"All six tensors frozen, excluded from optimizer and decay",
              "selection":"Minimum scheduled uniform-development-stratum teacher cross entropy"},
      "selfplay":{"steps":500,"batch_size":128,"lr":.0003,"seed":20261015,"checkpoint_every":50,"trainable_parameters":12872,
                  "teacher_rows_per_batch":64,"selfplay_rows_per_batch":64,"objective":".5*Huber_teacher(SEEDS)+.5*Huber_SP(SEEDS)+.25*(.5*CE_teacher+.5*CE_visits)",
                  "selection":"Fixed final optimizer step500; development metrics descriptive only","labels":"Terminal-game player0 seed margins; normalized nonterminal active>4 roots; legal visit policies; exact rederived127 features"},
      "teacher_dataset":"data/deeper-9000-v2","teacher_horizon":8,"teacher_rows":{"train":8000,"development":1000},
      "source_model":"data/training-stage5-deeper-h8-f127-seed2718/model.json","sampling":"Uniform strata with replacement; unit row weights; exact halfbatch in update",
      "sealed_final_test_consumed":False,"oracle_training":False,"amp":False,"loader_workers":0}


def publish(path,value):
    data=config.canonical(value)+"\n"
    if path.exists() and path.read_text(encoding="utf-8")==data:return
    teacher.atomic_json(path,value)


def freeze_plan():
    model=json.loads((ROOT/PLAN["source_model"]).read_text(encoding="utf-8"));manifest=json.loads((ROOT/PLAN["teacher_dataset"] / "manifest.json").read_text(encoding="utf-8"))
    teacher.require(config.digest(model["model"])==model["sha256"] and model["model"]["weights_sha256"]=="9399891301cc6a98caa220dbb5d57efbd068b0014505d0a036ff670f371382dc","Fixed H8 source changed")
    teacher.require(manifest["semantic_sha256"]=="f2878aac7ef0bee1a1358e36d981656dfb242683c31e245205790beae393b6c4" and manifest["rows"]==9000,"Teacher dataset changed")
    value={**PLAN,"source_sha256":teacher.sha(__file__),"trainer_sha256":teacher.sha(ROOT/"congklak/phase6_training.py"),
           "source_model_file_sha256":teacher.sha(ROOT/PLAN["source_model"]),"teacher_manifest_sha256":config.digest(manifest),
           "training_runtime_sha256":teacher.sha(ROOT/"build/training-runtime.json")}
    path=ROOT/"build/phase6-training-plan.json"
    if path.exists():teacher.require(json.loads(path.read_text(encoding="utf-8"))==value,"Phase6 training protocol/source changed; use a new version")
    else:teacher.atomic_json(path,value)
    return value


def train(stage,selfplay=None):
    from congklak import phase6_training as tr
    freeze_plan()
    options=PLAN[stage]
    run=tr.DEFAULT_WARM if stage=="warm" else tr.DEFAULT_UPDATE
    initial=tr.SOURCE_MODEL if stage=="warm" else tr.DEFAULT_WARM/"model.json"
    result=tr.train(run,stage,initial_model=initial,selfplay_path=selfplay,device="cuda",on_event=lambda v:print(config.canonical(v),flush=True),
                    **{k:options[k] for k in ("steps","batch_size","lr","seed","checkpoint_every")})
    teacher.require(result["status"]=="complete","Training interrupted; resume identical protocol")
    verify(stage)
    print(config.canonical({"event":"phase6_training_complete","stage":stage,"run":str(run),"steps":result["steps_completed"],"best_step":result["best_step"],"weights_sha256":result["weights_sha256"]}),flush=True)
    return result


def verify(stage):
    import torch
    from congklak import phase6_training as tr
    plan=freeze_plan();run=tr.DEFAULT_WARM if stage=="warm" else tr.DEFAULT_UPDATE
    result=json.loads((run/"result.json").read_text(encoding="utf-8"));identity=result["identity"];expected=plan[stage]
    teacher.require(result["status"]=="complete" and result["steps_completed"]==expected["steps"] and identity["source_sha256"]==plan["trainer_sha256"],"Training completion/source differs")
    with closing(sqlite3.connect((run/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");body,digest=db.execute("SELECT body,hash FROM config").fetchone()
        teacher.require(config.digest(json.loads(body))==digest and json.loads(body)==identity,"Damaged training config")
        entries=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        teacher.require([row[0] for row in entries]==list(range(expected["checkpoint_every"],expected["steps"]+1,expected["checkpoint_every"])),"Checkpoint schedule changed")
        history=[]
        for step,name,checksum,body,digest in entries:
            path=(run/name).resolve();meta=json.loads(body)
            teacher.require(path.parent==run and teacher.sha(path)==checksum and config.digest(meta)==digest,"Damaged catalog/checkpoint")
            state=torch.load(path,map_location="cpu",weights_only=True)
            teacher.require(state["identity"]==identity and state["step"]==step and state["draws"]==step*expected["batch_size"] and tr.weight_digest(state["model"])==meta["weights_sha256"],"Checkpoint state differs")
            count=2 if stage=="warm" else 8
            teacher.require(len(state["optimizer"]["state"])==count and len(state["optimizer"]["param_groups"][0]["params"])==count,"Wrong optimizer parameter ownership")
            teacher.require(state["teacher_draws"]==step*expected["batch_size"]*(1 if stage=="warm" else .5) and state["selfplay_draws"]==(0 if stage=="warm" else step*expected["batch_size"]//2),"Mixed draw counts differ")
            if stage=="warm":teacher.require(tr.weight_digest(tr.value_state(state["model"]))==tr.SOURCE_VALUE_WEIGHTS and tr.weight_digest(tr.value_state(state["best_model"]))==tr.SOURCE_VALUE_WEIGHTS,"Frozen value/trunk changed")
            else:teacher.require(state["best_model"] is None if step<expected["steps"] else state["best_step"]==expected["steps"],"Selfplay introduced checkpoint selection")
            history.append((step,meta["metrics"]["policy"]["uniform_stratum_teacher_cross_entropy"]))
        latest=state
    selected=min(history,key=lambda r:r[1])[0] if stage=="warm" else expected["steps"]
    teacher.require(result["best_step"]==latest["best_step"]==selected,"Selected checkpoint differs from protocol")
    export=json.loads((run/"model.json").read_text(encoding="utf-8"))
    teacher.require(config.digest(export["model"])==export["sha256"] and export["model"]==tr.export_payload(latest["best_model"],identity,selected),"Selected export differs")
    value={"schema":"phase6-training-verification-v1","stage":stage,"passed":True,"checkpoints_verified":len(entries),"latest_step":latest["step"],"selected_step":selected,
           "teacher_draws":latest["teacher_draws"],"selfplay_draws":latest["selfplay_draws"],"source_sha256":teacher.sha(__file__),"trainer_sha256":teacher.sha(tr.__file__),"sealed_final_test_consumed":False}
    publish(ROOT/f"build/phase6-training-{stage}-verification.json",value)
    print(config.canonical({"event":"phase6_training_verified",**value}),flush=True);return value


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","warm","update","verify-warm","verify-update"));parser.add_argument("--selfplay")
    args=parser.parse_args()
    if args.operation=="plan":print(config.canonical(freeze_plan()),flush=True)
    elif args.operation in ("warm","update"):train("warm" if args.operation=="warm" else "selfplay",args.selfplay)
    else:verify("warm" if args.operation=="verify-warm" else "selfplay")
