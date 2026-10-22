"""Six predeclared root-policy/ordinary-bias runs, selected before arena testing."""
from contextlib import closing
import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
from congklak import config,training_v4 as tr

PLAN={"schema":"stage5-root-action-ablation-v1","variants":["A","B","C"],"seeds":[2718,31415],"width":64,"input_count":43,
      "parameters":7496,"value_parameters":7041,"policy_parameters":455,"steps":8000,"batch_size":512,"lr":.001,"checkpoint_every":100,
      "dataset":"data/stronger-10000-v2","objectives":tr.OBJECTIVES,"policy_temperature_seeds":4.0,
      "value_target":"Root H6 player0 value only","policy_target":"Legal root H6 action-Q preferences; root chosen choice plus successor H5; no child H6 value augmentation",
      "calibration":"Ordinary TRAIN strata with >=20 rows only; .01 batch-conditional actor-error mean squared; missing groups omitted, all absent graph-zero",
      "selection":"Every100 steps select minimum finite-development uniform-stratum value MAE; then best seed pervariant; B/C challenger via lowest paired mean macro MAE before arena",
      "policy_selection":None,"runtime_gate":None,"development_calibration":False,"sealed_final_test_consumed":False,
      "promotion_gate":"Ordinary strata with >=20 development rows must not exceed H_v1 MAE; independent equal-time strength gates also required",
      "equal_sampling_not_equal_optimizer_flops":True,"A_policy_head":"Zero policy objective; untrained except AdamW decay"}

def emit(body): print(config.canonical(body),flush=True)
def run_path(variant,seed): return ROOT/f"data/training-stage5-action-{variant}-seed{seed}"

def freeze_plan():
    value={**PLAN,"source_sha256":tr.sha(__file__),"trainer_sha256":tr.sha(tr.__file__),"action_targets_sha256":tr.sha(tr.action_targets.__file__)}
    path=ROOT/"build/stage5-action-training-plan.json"
    if path.exists(): tr.require(json.loads(path.read_text(encoding="utf-8"))==value,"Action experiment changed; create a versioned lineage")
    else: tr.atomic_json(path,value)
    return value

def train_all():
    freeze_plan()
    for variant in PLAN["variants"]:
        for seed in PLAN["seeds"]:
            emit({"event":"stage5_action_run_started","variant":variant,"seed":seed,"target":PLAN["steps"]})
            result=tr.train(run_path(variant,seed),ROOT/PLAN["dataset"],steps=PLAN["steps"],seed=seed,width=PLAN["width"],batch_size=PLAN["batch_size"],
                            lr=PLAN["lr"],device="cuda",checkpoint_every=PLAN["checkpoint_every"],variant=variant,on_event=emit)
            emit({"event":"stage5_action_run_done","variant":variant,"seed":seed,"status":result["status"],"step":result["steps_completed"],"weights_sha256":result.get("weights_sha256")})
    return summarize()

def summarize():
    plan=freeze_plan(); rows=[]; paired=[]; selected={}
    for variant in PLAN["variants"]:
        group=[]
        for seed in PLAN["seeds"]:
            path=run_path(variant,seed); result=json.loads((path/"result.json").read_text(encoding="utf-8")); identity=result["identity"]
            tr.require(result["status"]=="complete" and result["steps_completed"]==PLAN["steps"] and result["parameters"]==PLAN["parameters"],"Candidate incomplete/shape changed")
            tr.require(identity["variant"]==variant and identity["seed"]==seed and identity["source_sha256"]==plan["trainer_sha256"] and identity["source_dependencies_sha256"]["congklak.action_targets"]==plan["action_targets_sha256"],"Candidate lineage differs")
            metrics=result["selected_development"]["finite_choice_horizon"]; policy=result["selected_development"]["policy"]
            ordinary={name:value for name,value in metrics["strata"].items() if name.endswith("/ordinary") and value["count"]>=20}
            row={"variant":variant,"seed":seed,"width":identity["width"],"input_count":43,"model":str((path/"policy-model.json").relative_to(ROOT)),
                 "value_model":str((path/"model.json").relative_to(ROOT)),"weights_sha256":result["weights_sha256"],"value_weights_sha256":result["value_weights_sha256"],
                 "parameters":result["parameters"],"best_step":result["best_step"],"draws":result["draws"],"macro_mae":metrics["uniform_stratum_mae_seeds"],"mae":metrics["mae_seeds"],
                 "ordinary":ordinary,"ordinary_gate_passed":all(value["mae_seeds"]<=value["H_v1_mae_seeds"] for value in ordinary.values()),"metrics":metrics,"policy":policy,
                 "elapsed_training_seconds":result["elapsed_seconds_this_invocation"]}
            rows.append(row); group.append(row)
        paired.append({"variant":variant,"mean_macro_mae":statistics.mean(row["macro_mae"] for row in group),"mean_mae":statistics.mean(row["mae"] for row in group),
                       "mean_ordinary_mae":{name:statistics.mean(row["ordinary"][name]["mae_seeds"] for row in group) for name in group[0]["ordinary"]},
                       "mean_policy_accuracy":statistics.mean(row["policy"]["root_H6_optimal_set_accuracy"] for row in group),
                       "mean_policy_root_H6_regret":statistics.mean(row["policy"]["mean_root_H6_Q_regret_seeds"] for row in group)})
        selected[variant]=min(group,key=lambda row:row["macro_mae"])
    winner=min((row for row in paired if row["variant"] in ("B","C")),key=lambda row:row["mean_macro_mae"])["variant"]
    result={"plan":plan,"rows":rows,"paired":paired,"selected":selected,"selected_control":selected["A"],"selected_challenger":selected[winner],
            "sealed_final_test_consumed":False,"promotion":"not promoted; development ordinary and independent equal-time gates required"}
    tr.atomic_json(ROOT/"build/stage5-action-training-results.json",result)
    emit({"event":"stage5_action_training_complete","control":{key:selected["A"][key] for key in ("variant","seed","model","macro_mae","ordinary_gate_passed")},
          "challenger":{key:selected[winner][key] for key in ("variant","seed","model","macro_mae","ordinary_gate_passed")}})
    return result

def verify():
    summary=summarize(); checked=0; samplers={}; evidence=[]
    for row in summary["rows"]:
        path=(ROOT/row["model"]).parent
        with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            body,digest=db.execute("SELECT body,hash FROM config").fetchone(); identity=json.loads(body); tr.require(config.digest(identity)==digest,"Corrupt identity")
            entries=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
            tr.require([entry[0] for entry in entries]==list(range(100,8001,100)),"Checkpoint schedule differs")
            for step,name,checksum,body,digest in entries:
                file=(path/name).resolve(); meta=json.loads(body)
                tr.require(file.parent==path and tr.sha(file)==checksum and config.digest(meta)==digest,"Damaged checkpoint file/catalog")
                state=torch.load(file,map_location="cpu",weights_only=True)
                tr.require(state["identity"]==identity and state["step"]==step and state["draws"]==step*512 and tr.weight_digest(state["model"])==meta["weights_sha256"],"Damaged state")
                checked+=1
            latest=state
            history=[(step,json.loads(body)["finite_choice_horizon"]["uniform_stratum_mae_seeds"]) for step,body in db.execute("SELECT step,body FROM history ORDER BY step")]
            tr.require(min(history,key=lambda value:value[1])[0]==latest["best_step"]==row["best_step"],"Selection added extra opportunity")
            if row["seed"] in samplers: tr.require(samplers[row["seed"]]==latest["sampler_rng"],"Matched sampler differs")
            else: samplers[row["seed"]]=latest["sampler_rng"]
            for filename,weights in (("policy-model.json",latest["best_model"]),("model.json",tr.value_state(latest["best_model"]))):
                export=json.loads((path/filename).read_text(encoding="utf-8")); model=export["model"]
                tr.require(config.digest(model)==export["sha256"] and model["identity"]==identity and model["weights_sha256"]==tr.weight_digest(weights),"Selected export differs")
            evidence.append({"variant":row["variant"],"seed":row["seed"],"checkpoints":len(entries),"latest_step":latest["step"],"draws":latest["draws"],"best_step":row["best_step"]})
    value={"checkpoint_files_verified":checked,"all_checks_passed":True,"matched_sampler_sequences_equal_by_seed":True,"runs":evidence,"sealed_final_test_consumed":False,"source_sha256":tr.sha(__file__)}
    tr.atomic_json(ROOT/"build/stage5-action-training-verification.json",value); emit({"event":"stage5_action_training_verified",**value}); return value

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("plan","train","summarize","verify")); args=parser.parse_args()
    if args.operation=="plan": emit(freeze_plan())
    elif args.operation=="train": train_all()
    elif args.operation=="summarize": summarize()
    else: verify()
