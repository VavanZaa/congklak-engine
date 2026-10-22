"""Predeclared, matched stage-five feature ablation; train/development only."""
from pathlib import Path
import argparse
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,training_v3 as tr

PLAN={"schema":"stage5-feature-ablation-v1","input_counts":[15,43],"widths":[32,64],
      "seeds":[2718,31415],"steps":8000,"batch_size":512,"lr":.001,"checkpoint_every":100,
      "dataset":"data/stronger-10000-v2","selection":"minimum mean finite-development uniform-stratum MAE across two seeds selects width per input count; minimum seed MAE selects exported representative",
      "promotion_gate":"selected representative must have MAE <= H_v1 in every ordinary stratum with >=20 development rows; separate equal-time/arena gates required",
      "recovery_snapshots_add_selection_opportunities":False,"sealed_final_test_consumed":False,
      "runtime_gate":None,"features":"15 count/active inputs, optionally plus14 occupied and14 singleton flags"}

def emit(value): print(config.canonical(value),flush=True)
def run_path(features,width,seed): return ROOT/f"data/training-stage5-f{features}-w{width}-seed{seed}"

def freeze_plan():
    path=ROOT/"build/stage5-evaluation-plan.json"
    value={**PLAN,"source_sha256":tr.sha(__file__),"trainer_sha256":tr.sha(tr.__file__),"features_sha256":tr.sha(tr.features_v3.__file__)}
    if path.exists(): tr.require(json.loads(path.read_text(encoding="utf-8"))==value,"Experiment changed; use a versioned plan")
    else: tr.atomic_json(path,value)
    return value

def train_all():
    freeze_plan()
    # Feature32 first makes its export available for independent native parity.
    runs=[(43,32),(15,32),(43,64),(15,64)]
    for features,width in runs:
        for seed in PLAN["seeds"]:
            emit({"event":"stage5_run_started","input_count":features,"width":width,"seed":seed,"target":PLAN["steps"]})
            result=tr.train(run_path(features,width,seed),ROOT/PLAN["dataset"],steps=PLAN["steps"],seed=seed,width=width,
                            batch_size=PLAN["batch_size"],lr=PLAN["lr"],device="cuda",checkpoint_every=PLAN["checkpoint_every"],on_event=emit,input_count=features)
            emit({"event":"stage5_run_done","input_count":features,"width":width,"seed":seed,"status":result["status"],"step":result["steps_completed"],"weights_sha256":result.get("weights_sha256")})
    return summarize()

def summarize():
    plan=freeze_plan(); rows=[]; paired=[]
    for features in PLAN["input_counts"]:
        for width in PLAN["widths"]:
            group=[]
            for seed in PLAN["seeds"]:
                path=run_path(features,width,seed); result=json.loads((path/"result.json").read_text(encoding="utf-8"))
                identity=result["identity"]
                tr.require(result["status"]=="complete" and result["steps_completed"]==PLAN["steps"] and identity["source_sha256"]==plan["trainer_sha256"] and identity["feature_source_sha256"]==plan["features_sha256"],"Incomplete or incompatible candidate")
                tr.require(identity["input_count"]==features and identity["width"]==width and identity["seed"]==seed,"Candidate ownership differs")
                metrics=result["selected_development"]["finite_choice_horizon"]
                ordinary={name:value for name,value in metrics["strata"].items() if name.endswith("/ordinary") and value["count"]>=20}
                gate=all(value["mae_seeds"]<=value["H_v1_mae_seeds"] for value in ordinary.values())
                row={"input_count":features,"width":width,"seed":seed,"model":str((path/"model.json").relative_to(ROOT)),"weights_sha256":result["weights_sha256"],
                     "best_step":result["best_step"],"parameters":result["parameters"],"draws":result["draws"],"macro_mae":metrics["uniform_stratum_mae_seeds"],"mae":metrics["mae_seeds"],
                     "ordinary":ordinary,"ordinary_gate_passed":gate,"metrics":metrics,"elapsed_training_seconds":result["elapsed_seconds_this_invocation"]}
                rows.append(row); group.append(row)
            paired.append({"input_count":features,"width":width,"mean_macro_mae":statistics.mean(row["macro_mae"] for row in group),
                           "mean_mae":statistics.mean(row["mae"] for row in group),"mean_ordinary_mae":{name:statistics.mean(row["ordinary"][name]["mae_seeds"] for row in group) for name in group[0]["ordinary"]}})
    selected={}
    for features in PLAN["input_counts"]:
        width=min((row for row in paired if row["input_count"]==features),key=lambda row:row["mean_macro_mae"])["width"]
        row=min((row for row in rows if row["input_count"]==features and row["width"]==width),key=lambda row:row["macro_mae"])
        selected[str(features)]=row
    result={"plan":plan,"rows":rows,"paired":paired,"selected":selected,"sealed_final_test_consumed":False,"promotion":"not promoted; development gate and independent equal-time arena required"}
    tr.atomic_json(ROOT/"build/stage5-evaluation-results.json",result)
    emit({"event":"stage5_evaluation_complete","selected":{key:{k:r[k] for k in ("input_count","width","seed","model","macro_mae","ordinary_gate_passed")} for key,r in selected.items()}})
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("plan","train","summarize")); args=parser.parse_args()
    if args.operation=="plan": emit(freeze_plan())
    elif args.operation=="train": train_all()
    else: summarize()
