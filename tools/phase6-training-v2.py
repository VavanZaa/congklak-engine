"""Fixed Phase6 selfplay update with validation before CUDA admission."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,deeper_teacher_v2 as teacher

WARM_V1_SHA="1982e9a3c6fc8402c2075c3f202709f303fabc6ed51237563717469090b39491"
PLAN={"schema":"phase6-update-training-plan-v2","rules_profile":config.PROFILE,"rules_digest":config.RULES_DIGEST,
      "architecture":{"input_count":127,"width":64,"parameters":12872,"export_schema":"congklak-mlp-export-v4"},
      "selfplay":{"steps":500,"batch_size":128,"lr":.0003,"seed":20261015,"checkpoint_every":50,"trainable_parameters":12872,
                  "teacher_rows_per_batch":64,"selfplay_rows_per_batch":64,
                  "objective":".5*Huber_teacher(SEEDS)+.5*Huber_SP(SEEDS)+.25*(.5*CE_teacher+.5*CE_visits)",
                  "selection":"Fixed final optimizer step500; development metrics descriptive only",
                  "labels":"Terminal-game player0 seed margins; normalized nonterminal active>4 roots; legal visit policies; exact rederived127 features"},
      "teacher_dataset":"data/deeper-9000-v2","teacher_horizon":8,"teacher_rows":{"train":8000,"development":1000},
      "source_model":"data/training-phase6-warm-v1/model.json","selfplay_dataset":"data/phase6-selfplay-v1",
      "output_run":"data/training-phase6-update-v1","warm_producer":"congklak/phase6_training.py","warm_producer_sha256":WARM_V1_SHA,
      "sampling":"Uniform strata with replacement; unit row weights; exact halfbatch in update",
      "resource_admission":"CPU deterministic flags and full teacher/SP/target/feature validation before actual CUDA context",
      "sealed_final_test_consumed":False,"oracle_training":False,"amp":False,"loader_workers":0}

def publish(path,value):
    if path.exists() and path.read_text(encoding="utf-8")==config.canonical(value)+"\n":return
    teacher.atomic_json(path,value)

def freeze_plan():
    model=json.loads((ROOT/PLAN["source_model"]).read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/PLAN["teacher_dataset"]/"manifest.json").read_text(encoding="utf-8"))
    sp_manifest=json.loads((ROOT/PLAN["selfplay_dataset"]/"manifest.json").read_text(encoding="utf-8"))
    from congklak.neural_relay import Model
    parsed=Model(ROOT/PLAN["source_model"])
    teacher.require(config.digest(model["model"])==model["sha256"] and parsed.input_count==127 and parsed.width==64,"Warm model changed")
    teacher.require(teacher.sha(ROOT/PLAN["warm_producer"])==WARM_V1_SHA,"Frozen warm producer changed")
    teacher.require(manifest["semantic_sha256"]=="f2878aac7ef0bee1a1358e36d981656dfb242683c31e245205790beae393b6c4" and manifest["rows"]==9000,"Teacher dataset changed")
    teacher.require(sp_manifest["identity"]["model_file_sha256"]==teacher.sha(ROOT/PLAN["source_model"]) and sp_manifest["identity"]["model_weights_sha256"]==model["model"]["weights_sha256"],"Selfplay warm lineage changed")
    value={**PLAN,"source_sha256":teacher.sha(__file__),"trainer_sha256":teacher.sha(ROOT/"congklak/phase6_training_v2.py"),
           "source_model_file_sha256":teacher.sha(ROOT/PLAN["source_model"]),"source_model_weights_sha256":model["model"]["weights_sha256"],
           "teacher_manifest_sha256":config.digest(manifest),"selfplay_manifest_sha256":config.digest(sp_manifest),
           "selfplay_loader_sha256":teacher.sha(ROOT/"congklak/phase6_selfplay.py"),"training_runtime_sha256":teacher.sha(ROOT/"build/training-runtime.json")}
    path=ROOT/"build/phase6-update-training-plan-v2.json"
    if path.exists():teacher.require(json.loads(path.read_text(encoding="utf-8"))==value,"Frozen update producer/source changed; use a new version")
    else:teacher.atomic_json(path,value)
    return value

def train():
    from congklak import phase6_training_v2 as tr
    plan=freeze_plan();options=plan["selfplay"]
    result=tr.train(ROOT/plan["output_run"],initial_model=ROOT/plan["source_model"],selfplay_path=ROOT/plan["selfplay_dataset"],device="cuda",
                    on_event=lambda v:print(config.canonical(v),flush=True),**{k:options[k] for k in ("steps","batch_size","lr","seed","checkpoint_every")})
    teacher.require(result["status"]=="complete","Update interrupted; resume the identical V2 producer")
    verify();print(config.canonical({"event":"phase6_training_complete","stage":"selfplay","run":plan["output_run"],"steps":result["steps_completed"],"best_step":result["best_step"],"weights_sha256":result["weights_sha256"]}),flush=True)
    return result

def verify():
    import torch
    from congklak import phase6_training_v2 as tr
    plan=freeze_plan();run=(ROOT/plan["output_run"]).resolve();expected=plan["selfplay"]
    result=json.loads((run/"result.json").read_text(encoding="utf-8"));identity=result["identity"]
    teacher.require(result["status"]=="complete" and result["steps_completed"]==expected["steps"] and identity["schema"]==tr.SCHEMA and identity["stage"]=="selfplay" and identity["source_sha256"]==plan["trainer_sha256"],"Completion/source differs")
    for key in ("steps","batch_size","lr","seed","checkpoint_every"):
        teacher.require(identity[key]==expected[key],"Fixed update budget differs")
    teacher.require(identity["initial_model_file_sha256"]==plan["source_model_file_sha256"] and identity["initial_weights_sha256"]==plan["source_model_weights_sha256"] and identity["initial_value_weights_sha256"]==tr.SOURCE_VALUE_WEIGHTS,"Warm model ancestry differs")
    teacher.require(identity["selfplay_manifest_sha256"]==plan["selfplay_manifest_sha256"] and identity["selfplay_loader_sha256"]==plan["selfplay_loader_sha256"] and identity["warm_witness_implementation_sha256"]==WARM_V1_SHA,"Loader/warm witness lineage differs")
    tr.warm_completion(ROOT/plan["source_model"],tr.Model(ROOT/plan["source_model"]))
    with closing(sqlite3.connect((run/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");body,digest=db.execute("SELECT body,hash FROM config").fetchone()
        teacher.require(config.digest(json.loads(body))==digest and json.loads(body)==identity,"Damaged update config")
        entries=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        teacher.require([row[0] for row in entries]==list(range(50,501,50)),"Checkpoint schedule changed")
        for step,name,checksum,body,digest in entries:
            path=(run/name).resolve();meta=json.loads(body)
            teacher.require(path.parent==run and teacher.sha(path)==checksum and config.digest(meta)==digest,"Damaged checkpoint/catalog")
            state=torch.load(path,map_location="cpu",weights_only=True)
            teacher.require(state["identity"]==identity and state["step"]==step and state["draws"]==step*128 and tr.weight_digest(state["model"])==meta["weights_sha256"],"Checkpoint state differs")
            teacher.require(len(state["optimizer"]["state"])==len(state["optimizer"]["param_groups"][0]["params"])==8,"Wrong optimizer ownership")
            teacher.require(state["teacher_draws"]==state["selfplay_draws"]==step*64,"Mixed draw counts differ")
            teacher.require(state["best_model"] is None if step<500 else state["best_step"]==500,"Selfplay introduced checkpoint selection")
    teacher.require(result["best_step"]==state["best_step"]==500,"Fixed final selection differs")
    export=json.loads((run/"model.json").read_text(encoding="utf-8"))
    teacher.require(config.digest(export["model"])==export["sha256"] and export["model"]==tr.export_payload(state["best_model"],identity,500),"Selected export differs")
    teacher.require(result["weights_sha256"]==tr.weight_digest(state["best_model"]) and result["value_weights_sha256"]==tr.weight_digest(tr.value_state(state["best_model"])),"Result digests differ")
    proof={"schema":"phase6-update-training-verification-v2","stage":"selfplay","passed":True,"checkpoints_verified":len(entries),"latest_step":500,"selected_step":500,
           "teacher_draws":state["teacher_draws"],"selfplay_draws":state["selfplay_draws"],"source_sha256":teacher.sha(__file__),"trainer_sha256":teacher.sha(tr.__file__),"warm_producer_sha256":WARM_V1_SHA,"sealed_final_test_consumed":False}
    publish(ROOT/"build/phase6-training-selfplay-verification-v2.json",proof)
    print(config.canonical({"event":"phase6_training_verified",**proof}),flush=True);return proof

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("plan","update","verify-update"));args=parser.parse_args()
    if args.operation=="plan":print(config.canonical(freeze_plan()),flush=True)
    elif args.operation=="update":train()
    else:verify()
