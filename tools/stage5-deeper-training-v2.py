"""Predeclared H6/H8 x count43/relay127 VALUE-only factorial experiment."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from congklak import config,deeper_teacher as dt,features_relay

PLAN={"schema":"stage5-deeper-relay-factorial-plan-v2","teacher_depths":[6,8],"input_counts":[43,127],"seeds":[2718,31415],"width":64,
      "parameters":{"43":7496,"127":12872},"trainable_parameters":{"43":7041,"127":12417},"dummy_policy_parameters":455,
      "steps":8000,"batch_size":512,"lr":.001,"checkpoint_every":100,"dataset":"data/deeper-9000-v1","parent_dataset":"data/stronger-10000-v2",
      "objective":"Huber(delta1) reconstructed player0 seed value ONLY; no policy/bias losses; dummy head frozen without decay or gradients",
      "value_targets":"Distinct root H6 or root H8 finite-choice margins, never mixed; H8 root-action Q successorH7 provenance retained but not used for training",
      "sampler":"uniform-stratum/P=q/unitweight; identical draws byseed acrossfourconditions",
      "selection":"Own-teacher scheduled100-step uniform-development-stratum VALUE MAE; bestseed percell; no cross-teacher/arena-driven checkpointselection",
      "cross_horizon_metrics":"BothH6/H8 evaluated only after own-teacher checkpointselection; descriptive factorial comparisons",
      "promotion_gate":"Eachordinarydevelopmentstratumwith>=20 must notexceedH_v1MAE againstH8; independent equal-time paired-family seedmargin95%lower>0 also required",
      "sealed_final_test_consumed":False,"new_roots":0,"successor_augmentation":False,"equal_optimizer_steps_not_equal_flops":True}

def emit(value): print(config.canonical(value),flush=True)
def publish(path,value):
    data=config.canonical(value)+"\n"
    if Path(path).is_file() and Path(path).read_text(encoding="utf-8")==data: return
    dt.atomic_json(path,value)
def cell(h,f): return f"h{h}-f{f}"
def run_path(h,f,seed): return ROOT/f"data/training-stage5-deeper-h{h}-f{f}-seed{seed}"

def freeze_plan():
    ident,_=dt.identity(ROOT/PLAN["parent_dataset"])
    value={**PLAN,"source_sha256":dt.sha(__file__),"trainer_sha256":dt.sha(ROOT/"congklak/training_v6.py"),"deeper_teacher_sha256":dt.sha(dt.__file__),
           "features_sha256":dt.sha(features_relay.__file__),"dataset_identity":ident}
    path=ROOT/"build/stage5-deeper-training-plan-v2.json"
    if path.exists(): dt.require(json.loads(path.read_text(encoding="utf-8"))==value,"Deeper experiment/sourcechanged; use a newversion")
    else: dt.atomic_json(path,value)
    return value

def train_all():
    from congklak import training_v6 as tr
    freeze_plan()
    for h in PLAN["teacher_depths"]:
        for f in PLAN["input_counts"]:
            for seed in PLAN["seeds"]:
                emit({"event":"stage5_deeper_run_started","teacher_depth":h,"input_count":f,"seed":seed,"target":PLAN["steps"]})
                result=tr.train(run_path(h,f,seed),ROOT/PLAN["dataset"],steps=PLAN["steps"],seed=seed,width=PLAN["width"],batch_size=PLAN["batch_size"],lr=PLAN["lr"],device="cuda",checkpoint_every=PLAN["checkpoint_every"],input_count=f,teacher_depth=h,on_event=emit)
                emit({"event":"stage5_deeper_run_done","teacher_depth":h,"input_count":f,"seed":seed,"status":result["status"],"step":result["steps_completed"],"weights_sha256":result.get("weights_sha256")})
                tr.require(result["status"]=="complete","Training interrupted; resume experiment")
    return summarize()

def summarize():
    tr=dt
    plan=freeze_plan(); rows=[]; paired=[]; selected={}
    for h in PLAN["teacher_depths"]:
        for f in PLAN["input_counts"]:
            group=[]
            for seed in PLAN["seeds"]:
                path=run_path(h,f,seed); result=json.loads((path/"result.json").read_text(encoding="utf-8")); identity=result["identity"]
                tr.require(result["status"]=="complete" and result["steps_completed"]==8000 and result["parameters"]==PLAN["parameters"][str(f)] and result["trainable_parameters"]==PLAN["trainable_parameters"][str(f)],"Candidateincomplete/shapechanged")
                tr.require(identity["teacher_depth"]==h and identity["input_count"]==f and identity["seed"]==seed and identity["source_sha256"]==plan["trainer_sha256"] and identity["source_dependencies_sha256"]["congklak.deeper_teacher"]==plan["deeper_teacher_sha256"],"Candidatelineagediffers")
                metric=result["selected_development"]["finite_choice_horizon"]; cross=result["selected_cross_teacher_development"]
                ordinary={name:value for name,value in cross["H8"]["finite_choice_horizon"]["strata"].items() if name.endswith("/ordinary") and value["count"]>=20}
                row={"condition":cell(h,f),"teacher_depth":h,"input_count":f,"seed":seed,"width":64,"model":str((path/"model.json").relative_to(ROOT)),"weights_sha256":result["weights_sha256"],
                     "value_weights_sha256":result["value_weights_sha256"],"parameters":result["parameters"],"trainable_parameters":result["trainable_parameters"],"best_step":result["best_step"],"draws":result["draws"],
                     "macro_mae":metric["uniform_stratum_mae_seeds"],"mae":metric["mae_seeds"],"metrics":metric,"cross_teacher_metrics":cross,"ordinary_H8":ordinary,
                     "ordinary_gate_passed":all(v["mae_seeds"]<=v["H_v1_mae_seeds"] for v in ordinary.values()),"elapsed_training_seconds":result["elapsed_seconds_this_invocation"]}
                rows.append(row); group.append(row)
            paired.append({"condition":cell(h,f),"teacher_depth":h,"input_count":f,"mean_own_teacher_macro_mae":statistics.mean(r["macro_mae"] for r in group),
                           "mean_cross_teacher_macro_mae":{f"H{horizon}":statistics.mean(r["cross_teacher_metrics"][f"H{horizon}"]["finite_choice_horizon"]["uniform_stratum_mae_seeds"] for r in group) for horizon in (6,8)}})
            selected[cell(h,f)]=min(group,key=lambda r:(r["macro_mae"],r["seed"]))
    result={"plan":plan,"rows":rows,"paired":paired,"selected":selected,"sealed_final_test_consumed":False,"promotion":"notpromoted; ordinaryH8 and equal-timegatesrequired"}
    publish(ROOT/"build/stage5-deeper-training-results.json",result); emit({"event":"stage5_deeper_training_complete","selected":selected}); return result

def verify():
    import torch
    from congklak import training_v6 as tr
    summary=summarize(); checked=0; samplers={}; evidence=[]; initial_heads={}
    for row in summary["rows"]:
        path=(ROOT/row["model"]).parent
        with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            body,digest=db.execute("SELECT body,hash FROM config").fetchone(); identity=json.loads(body); tr.require(config.digest(identity)==digest,"Corruptidentity")
            entries=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall(); tr.require([e[0] for e in entries]==list(range(100,8001,100)),"Checkpoint schedulechanged")
            history=[]
            for step,name,checksum,body,digest in entries:
                file=(path/name).resolve(); meta=json.loads(body); tr.require(file.parent==path and tr.sha(file)==checksum and config.digest(meta)==digest,"Damagedcheckpointfile/catalog")
                state=torch.load(file,map_location="cpu",weights_only=True); tr.require(state["identity"]==identity and state["step"]==step and state["draws"]==step*512 and tr.weight_digest(state["model"])==meta["weights_sha256"],"Damagedcheckpointstate")
                tr.require(len(state["optimizer"]["state"])==6 and len(state["optimizer"]["param_groups"][0]["params"])==6,"Dummyheadenteredoptimizer")
                head=tr.weight_digest({k:v for k,v in state["model"].items() if k.startswith("policy.")})
                key=(row["input_count"],row["seed"])
                if key in initial_heads: tr.require(initial_heads[key]==head,"Dummyheadchanged")
                else: initial_heads[key]=head
                history.append((step,meta["metrics"]["finite_choice_horizon"]["uniform_stratum_mae_seeds"])); checked+=1
            latest=state; tr.require(min(history,key=lambda x:x[1])[0]==latest["best_step"]==row["best_step"],"Extra checkpointselectionopportunity")
            if row["seed"] in samplers: tr.require(samplers[row["seed"]]==latest["sampler_rng"],"Matched samplerchanged")
            else: samplers[row["seed"]]=latest["sampler_rng"]
            export=json.loads((path/"model.json").read_text(encoding="utf-8")); model=export["model"]
            tr.require(config.digest(model)==export["sha256"] and model["identity"]==identity and model["weights_sha256"]==tr.weight_digest(latest["best_model"]),"Selectedexportchanged")
            evidence.append({"condition":row["condition"],"seed":row["seed"],"checkpoints":len(entries),"latest_step":latest["step"],"best_step":row["best_step"],"draws":latest["draws"]})
    value={"checkpoint_files_verified":checked,"all_checks_passed":True,"matched_sampler_sequences_equal_by_seed":True,"dummy_policy_head_frozen":True,"runs":evidence,"sealed_final_test_consumed":False,"source_sha256":tr.sha(__file__)}
    publish(ROOT/"build/stage5-deeper-training-verification.json",value); emit({"event":"stage5_deeper_training_verified",**value}); return value

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("plan","generate","train","summarize","verify")); args=parser.parse_args()
    if args.operation=="plan": emit(freeze_plan())
    elif args.operation=="generate":
        freeze_plan(); emit(dt.generate(ROOT/PLAN["dataset"],on_event=emit))
    elif args.operation=="train": train_all()
    elif args.operation=="summarize": summarize()
    else: verify()
