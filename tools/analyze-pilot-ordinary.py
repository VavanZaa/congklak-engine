"""Read-only ordinary-position error and accepted-sampler review; no searches."""
from collections import Counter,defaultdict
from contextlib import closing
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,value_pilot_data as data,features_relay,value_pilot_learning as learning
from congklak.native import Native
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator

BASELINE=ROOT/"build/pilot-loss-review-baseline.json"
OUTPUT=ROOT/"build/pilot-loss-review-ordinary.json"
SCHEMA="pilot-ordinary-error-review-v1"
SOURCE_FILES=("tools/analyze-pilot-ordinary.py","tests/pilot_loss_review_ordinary_acceptance.py","docs/PILOT_ORDINARY_ERRORS.md")
CONTROL=ROOT/"data/quality-train-v1/model.json"
PILOT=ROOT/"data/value-pilot-v1/model.json"
require,sha=learning.require,learning.sha


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline():
    value=read(BASELINE)
    require(value["schema"]=="pilot-loss-review-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==2600 and value["prior_accepted_inputs"]==2598,"Ordinary review baseline differs")
    require(sha(CONTROL)==value["fixed_model_sha256"] and sha(PILOT)==value["fixed_candidate_model_sha256"],"Fixed review models differ")
    for name,checksum in value["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==checksum,"Prior accepted artifact changed: "+name)
    return value


def replay_exposure(rows,seed=20261022,draws=51200):
    """Exact reconstruction of the accepted sampler, not a historical draw log."""
    cumulative=[];total=0.
    for row in rows:total+=row["sampling_probability"];cumulative.append(total)
    require(abs(total-1)<1e-12 and all(a<b for a,b in zip([0.]+cumulative[:-1],cumulative)),"Invalid sampler CDF")
    rng=random.Random(seed);observed=[0]*len(rows)
    for _ in range(draws):observed[bisect.bisect_right(cumulative,rng.random()*total)]+=1
    roots=[];families=defaultdict(lambda:dict(roots=0,draws=0,probability=0.));pools=defaultdict(lambda:dict(roots=0,draws=0,probability=0.))
    for row,count in zip(rows,observed):
        probability=row["sampling_probability"]
        roots.append(dict(state_key=row["state_key"],family=row["family"],pool=row["pool"],stratum=row["stratum"],probability=probability,expected_draws=draws*probability,observed_draws=count))
        for entry in (families[row["family"]],pools[row["pool"]]):
            entry["roots"]+=1;entry["draws"]+=count;entry["probability"]+=probability
    family_rows=[dict(family=name,**entry,expected_draws=draws*entry["probability"]) for name,entry in sorted(families.items())]
    require(sum(observed)==draws and sum(entry["draws"] for entry in family_rows)==draws,"Reconstructed draw conservation differs")
    return dict(seed=seed,draws=draws,unique_admitted_roots=len(rows),unique_roots_drawn=sum(count>0 for count in observed),
                unique_admitted_families=len(families),unique_families_drawn=sum(entry["draws"]>0 for entry in family_rows),
                replacement_draws_are_not_unique_samples=True,provenance="Exact reconstruction from accepted seed/CDF/51200 draws; prior independent200-step replay verified saved RNG states; not a separately recorded draw log",
                probability_row_order_sha256=config.digest([[row["state_key"],row["sampling_probability"]] for row in rows]),
                root_draw_ledger_sha256=config.digest([[row["state_key"],count] for row,count in zip(rows,observed)]),
                family_draw_ledger_sha256=config.digest([[entry["family"],entry["draws"]] for entry in family_rows]),
                pools=dict(sorted(pools.items())),roots=roots,families=family_rows)


def literal(model,row):
    x=list(row["cached_features"]["x"])
    for prefix in ("layers.0","layers.2"):
        x=[max(0.,sum(weight*value for weight,value in zip(weights,x))+bias) for weights,bias in zip(model.weights[prefix+".weight"],model.weights[prefix+".bias"])]
    z=math.tanh(sum(weight*value for weight,value in zip(model.weights["layers.4.weight"][0],x))+model.weights["layers.4.bias"][0])
    logits=[sum(weight*value for weight,value in zip(weights,x))+bias for weights,bias in zip(model.weights["policy.weight"],model.weights["policy.bias"])]
    feature=row["cached_features"]
    return dict(value_player0=feature["d"]+feature["r"]*z,residual_actor=z,policy_logits=logits)


def target_sources():
    """Read only sealed completed label tables, never regenerate histories."""
    index=defaultdict(dict)
    locations=(("data/ordinary-data-pilot-v1/work.sqlite3","labels",8),
               ("data/natural-study-v1/work.sqlite3","h8_labels",8),("data/natural-study-v1/work.sqlite3","h10_labels",10),
               ("data/horizon-finish-v1/work.sqlite3","h12_references",12),
               ("data/horizon-finish-v1/work.sqlite3","prior_h14_labels",14),
               ("data/horizon-finish-v1/work.sqlite3","tier1_labels",14),("data/horizon-finish-v1/work.sqlite3","tier2_labels",14))
    for filename,table,depth in locations:
        with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:
            db.execute("BEGIN")
            for key,body,digest in db.execute("SELECT state,body,hash FROM "+table+" ORDER BY state"):
                item=json.loads(body);require(config.digest(item)==digest and item["label"]["choice_depth"]==depth,"Corrupt requested-depth target item")
                entry=index[key].setdefault(depth,dict(item=item,links=[]));require(entry["item"]==item,"Same-depth source item differs")
                entry["links"].append(dict(path=filename,table=table,state_key=key,item_sha256=digest))
    return index


def errors(value,target,d,r,actor):
    signed=value-target;sign=1 if actor==0 else -1
    return dict(prediction_player0=value,prediction_actor=sign*value,residual_actor=(value-d)/r,
                error_player0=signed,error_actor=sign*signed,error_residual_actor=signed/r,absolute_error_seeds=abs(signed))


def stats(rows):
    n=len(rows);require(n>0,"Empty ordinary summary")
    absolute=lambda model:math.fsum(row[model]["absolute_error_seeds"] for row in rows)
    delta=[row["absolute_error_delta_seeds"] for row in rows];worse=sorted((row for row in rows if row["absolute_error_delta_seeds"]>0),key=lambda row:(-row["absolute_error_delta_seeds"],row["state_key"]))
    improved=sorted((row for row in rows if row["absolute_error_delta_seeds"]<0),key=lambda row:(row["absolute_error_delta_seeds"],row["state_key"]))
    gross_worse=math.fsum(row["absolute_error_delta_seeds"] for row in worse);gross_improve=-math.fsum(row["absolute_error_delta_seeds"] for row in improved)
    contribution=lambda row:dict(state_key=row["state_key"],family=row["family"],band=row["band"],teacher_depth=row["teacher_depth"],delta_absolute_error_seeds=row["absolute_error_delta_seeds"],control_absolute_error_seeds=row["control"]["absolute_error_seeds"],pilot_absolute_error_seeds=row["pilot"]["absolute_error_seeds"])
    return dict(rows=n,families=len({row["family"] for row in rows}),control_mae_seeds=absolute("control")/n,pilot_mae_seeds=absolute("pilot")/n,
                delta_mae_seeds=math.fsum(delta)/n,total_absolute_error_delta_seeds=math.fsum(delta),
                improved=sum(value<0 for value in delta),worse=sum(value>0 for value in delta),tied=sum(value==0 for value in delta),
                gross_worsening_seeds=gross_worse,gross_improvement_seeds=gross_improve,
                top_worsening=[contribution(row) for row in worse[:5]],top_improvement=[contribution(row) for row in improved[:5]],
                top1_worsening_fraction=(worse[0]["absolute_error_delta_seeds"]/gross_worse if worse else 0.),
                top3_worsening_fraction=(math.fsum(row["absolute_error_delta_seeds"] for row in worse[:3])/gross_worse if worse else 0.),
                top3_pilot_absolute_error_fraction=math.fsum(sorted((row["pilot"]["absolute_error_seeds"] for row in rows),reverse=True)[:3])/absolute("pilot") if absolute("pilot") else 0.,
                control_mean_error_player0=math.fsum(row["control"]["error_player0"] for row in rows)/n,pilot_mean_error_player0=math.fsum(row["pilot"]["error_player0"] for row in rows)/n,
                control_mean_error_actor=math.fsum(row["control"]["error_actor"] for row in rows)/n,pilot_mean_error_actor=math.fsum(row["pilot"]["error_actor"] for row in rows)/n,
                pilot_actor_overestimates=sum(row["pilot"]["error_actor"]>0 for row in rows),pilot_actor_underestimates=sum(row["pilot"]["error_actor"]<0 for row in rows),
                leave_one_out_delta_mae=[dict(state_key=row["state_key"],delta_mae_seeds=(math.fsum(delta)-row["absolute_error_delta_seeds"])/(n-1)) for row in rows] if n>1 else [])


def summaries(rows):
    by_band=defaultdict(list);by_depth=defaultdict(list);by_actor=defaultdict(list);by_legal=defaultdict(list)
    for row in rows:
        by_band[row["band"]].append(row);by_depth[str(row["teacher_depth"])].append(row);by_actor[str(row["actor"])].append(row);by_legal[str(row["legal_count"])].append(row)
    return dict(all=stats(rows),by_band={name:stats(values) for name,values in sorted(by_band.items())},by_teacher_depth={name:stats(values) for name,values in sorted(by_depth.items())},by_actor={name:stats(values) for name,values in sorted(by_actor.items())},by_legal_count={name:stats(values) for name,values in sorted(by_legal.items())})


def nearest(row,pool):
    eligible=[candidate for candidate in pool if candidate["family"]!=row["family"]]
    require(eligible,"Nearest feature pool empty after family exclusion")
    x=row["cached_features"]["x"]
    score=lambda candidate:math.fsum(abs(a-b) for a,b in zip(x,candidate["cached_features"]["x"]))/127
    distance,candidate=min(((score(candidate),candidate) for candidate in eligible),key=lambda pair:(pair[0],pair[1]["state_key"]))
    return dict(state_key=candidate["state_key"],family=candidate["family"],pool=candidate["pool"],stratum=candidate["stratum"],teacher_depth=candidate["teacher_depth"],
                actor=candidate["state"]["controller"],active=sum(map(sum,candidate["state"]["pits"])),store_difference=candidate["cached_features"]["d"],
                mean_l1_raw127=distance,eligible_rows=len(eligible),source_row_sha256=config.digest(candidate))


def build():
    prior=baseline();manifest,groups=data.prepare();source=Model(CONTROL);pilot=Model(PILOT);native=Native()
    require(source.id==learning.SOURCE_WEIGHTS and learning.model_frozen_digest(source)==learning.model_frozen_digest(pilot)==learning.FROZEN_WEIGHTS,"Frozen trunk/policy tensors differ")
    require(set(source.weights)==set(pilot.weights) and all(source.weights[key]==pilot.weights[key] for key in source.weights if key not in learning.TRAINABLE),"Policy/trunk invariant differs")
    train_ordinary=[row for row in groups["train"] if row["pool"]=="ordinary"]
    selected={"ordinary_validation":groups["ordinary_development"],"ordinary_training":train_ordinary,"protected_diagnostic":groups["diagnostic"],"base_ordinary_development":[row for row in groups["development"] if row["stratum"].endswith("/ordinary")]}
    require({name:len(rows) for name,rows in selected.items()}==dict(ordinary_validation=14,ordinary_training=61,protected_diagnostic=8,base_ordinary_development=259),"Review cohort denominators differ")
    exposure=replay_exposure(groups["train"]);draws={entry["state_key"]:entry["observed_draws"] for entry in exposure["roots"]};family_draws={entry["family"]:entry["draws"] for entry in exposure["families"]}
    result=read(learning.DEFAULT/"result.json");require(result["draws"]==51200 and result["selected_step"]==200 and result["identity"]["probability_row_order_sha256"]==exposure["probability_row_order_sha256"],"Accepted sampler budget/order differs")
    evidence=target_sources();output={};maximum_parity=0.;forward_count=0
    with CpuBatchEvaluator(source,max_batch=32,backend="scalar") as control_eval,CpuBatchEvaluator(pilot,max_batch=32,backend="scalar") as pilot_eval:
        identities=dict(control=control_eval.identity,pilot=pilot_eval.identity)
        for name,rows in selected.items():
            records=[]
            for row in rows:
                learning.target(row);p=row["state"];actor=p["controller"];active=sum(map(sum,p["pits"]));d,r=row["cached_features"]["d"],row["cached_features"]["r"]
                before=control_eval.evaluate_many([p])[0];after=pilot_eval.evaluate_many([p])[0];forward_count+=2
                require(before["policy_logits"]==after["policy_logits"],"Actual native policy logits differ")
                for model,prediction in ((source,before),(pilot,after)):
                    literal_prediction=literal(model,row);difference=abs(literal_prediction["value_player0"]-prediction["value_player0"])
                    maximum_parity=max(maximum_parity,difference);require(difference<1e-10 and all(abs(a-b)<1e-10 for a,b in zip(literal_prediction["policy_logits"],prediction["policy_logits"])),"Cached literal/native forward differs")
                target=row["label"]["target_player0"];control_error=errors(before["value_player0"],target,d,r,actor);pilot_error=errors(after["value_player0"],target,d,r,actor)
                targets=[]
                if row["pool"]=="ordinary":
                    require(set(evidence[row["state_key"]])=={item["depth"] for item in row["target_evidence"]},"Requested depth availability differs")
                    for item in row["target_evidence"]:
                        original=evidence[row["state_key"]][item["depth"]];label=original["item"]["label"]
                        require(config.digest(original["item"])==item["item_sha256"] and label["target_player0"]==item["target_player0"] and label["best_moves"]==item["best_moves"] and original["item"]["root_sha256"]==row["original_root_sha256"],"Requested target provenance differs")
                        if item["depth"]==row["teacher_depth"]:require(row["label"]==label,"Deepest selected target differs")
                        targets.append(dict(depth=item["depth"],target_player0=label["target_player0"],item_sha256=item["item_sha256"],sources=original["links"],
                            control_absolute_error_seeds=abs(before["value_player0"]-label["target_player0"]),pilot_absolute_error_seeds=abs(after["value_player0"]-label["target_player0"]),
                            absolute_error_delta_seeds=abs(after["value_player0"]-label["target_player0"])-abs(before["value_player0"]-label["target_player0"])))
                else:
                    targets=[dict(depth=8,target_player0=target,item_sha256=None,sources=[dict(path="build/value-pilot-rows.json",group="development",state_key=row["state_key"],row_sha256=config.digest(row))],control_absolute_error_seeds=control_error["absolute_error_seeds"],pilot_absolute_error_seeds=pilot_error["absolute_error_seeds"],absolute_error_delta_seeds=pilot_error["absolute_error_seeds"]-control_error["absolute_error_seeds"])]
                record=dict(state_key=row["state_key"],family=row["family"],cohort=row["cohort"],pool=row["pool"],stratum=row["stratum"],actor=actor,active=active,band=row["stratum"].split("/")[0],legal_count=row["legal_mask"].bit_count(),legal_mask=row["legal_mask"],
                            target_kind=row["label"]["kind"],teacher_depth=row["teacher_depth"],target_player0=target,target_actor=(1 if actor==0 else -1)*target,target_residual_actor=(target-d)/r,d=d,signed_active=r,
                            source=dict(path="build/value-pilot-rows.json",row_sha256=config.digest(row),label_sha256=config.digest(row["label"]),original_root_sha256=row.get("original_root_sha256"),label_item_sha256=row.get("label_item_sha256"),label_source=row["label_source"]),
                            control=control_error,pilot=pilot_error,absolute_error_delta_seeds=pilot_error["absolute_error_seeds"]-control_error["absolute_error_seeds"],
                            sampler_observed_root_draws=draws.get(row["state_key"],0),sampler_observed_family_draws=family_draws.get(row["family"],0),sampling_probability=row.get("sampling_probability",0.),
                            requested_depth_sensitivity=targets,
                            nearest_training=(dict(all_optimizer=nearest(row,groups["train"]),ordinary_optimizer=nearest(row,train_ordinary)) if name in ("ordinary_validation","protected_diagnostic") else dict(ordinary_optimizer=nearest(row,train_ordinary)) if name=="ordinary_training" else None))
                require(8 in [item["depth"] for item in targets],"Original H8 sensitivity missing")
                records.append(record)
            output[name]=dict(rows=records,summary=summaries(records))
    h8_summary={}
    for name,value in output.items():
        values=[]
        for row in value["rows"]:
            target=next(item for item in row["requested_depth_sensitivity"] if item["depth"]==8)["target_player0"]
            values.append(dict(row,control=errors(row["control"]["prediction_player0"],target,row["d"],row["signed_active"],row["actor"]),pilot=errors(row["pilot"]["prediction_player0"],target,row["d"],row["signed_active"],row["actor"]),absolute_error_delta_seeds=abs(row["pilot"]["prediction_player0"]-target)-abs(row["control"]["prediction_player0"]-target)))
        h8_summary[name]=summaries(values)
    inputs=dict(prior["inputs"]);inputs.update({name:sha(ROOT/name) for name in SOURCE_FILES+("build/pilot-loss-review-baseline.json",)})
    body=dict(schema=SCHEMA,passed=True,rules_digest=config.RULES_DIGEST,source_sha256=sha(__file__),baseline_sha256=sha(BASELINE),
              models=dict(control=dict(path=CONTROL.relative_to(ROOT).as_posix(),file_sha256=source.file_sha256,weights_sha256=source.id),pilot=dict(path=PILOT.relative_to(ROOT).as_posix(),file_sha256=pilot.file_sha256,weights_sha256=pilot.id)),
              frozen_policy=dict(all_six_tensors_equal=True,frozen_weights_sha256=learning.FROZEN_WEIGHTS,frozen_parameters=12807,trainable_value_parameters=65,all342_native_policy_logits_equal=True),
              inference=dict(backend="scalar",identities=identities,model_dense_forwards=forward_count,maximum_literal_native_value_difference=maximum_parity,precision="Strict double native forward versus literal cached-feature forward; distinct from saved FP32 training metrics"),
              sampling=exposure,groups=output,original_H8_same_prediction_sensitivity=h8_summary,
              nearest_feature_scope=dict(metric="Mean absolute difference across the127 raw cached encoder coordinates; no rescaling/clipping",family_exclusion="Exclude every optimizer row with the query family; tie break by state_key",query_groups="14validation and8diagnostics against all7624 and ordinary61;61train roots against other-family ordinary61; base development has no nearest-neighbor claim",limitations="Mixed normalized counts and unclipped relay/drop and binary feature scales; uncalibrated similarity, not a coverage/generalization/capacity proof"),
              interpretation=dict(targets="Approximate completed finite-choice-horizon labels, not game truth; deepest requested target retained unchanged",sampling="Exact accepted seed/CDF reconstruction;51200 replacement draws do not mean51200unique roots or independent games",attribution="One bundled dataset/sampler/head-only pilot; descriptive errors cannot identify capacity failure or isolate causes",ancestral_exposure="Frozen trunk/policy retain ancestral training and consumed-development exposure; current optimizer exclusions are not unlearning"),
              no_teacher_searches=True,no_new_games=True,no_training=True,no_torch=True,no_weight_or_target_changes=True,sealed_final_test_consumed=False,inputs=dict(sorted(inputs.items())))
    require(forward_count==684 and "torch" not in sys.modules,"Read-only CPU review scope differs")
    baseline();return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",nargs="?",choices=("build","verify"),default="build");args=parser.parse_args()
    value=build()
    if args.operation=="verify":require(read(OUTPUT)==value,"Ordinary review changed")
    else:learning.publish(OUTPUT,value)
    summary=value["groups"]["ordinary_validation"]["summary"]["all"]
    print(config.canonical(dict(event="pilot_ordinary_review_"+args.operation,passed=True,rows=342,validation_rows=14,delta_mae_seeds=summary["delta_mae_seeds"],improved=summary["improved"],worse=summary["worse"],output_sha256=sha(OUTPUT))),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
