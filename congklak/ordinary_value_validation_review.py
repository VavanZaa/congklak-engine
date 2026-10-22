"""Deterministic saved-data review of six ordinary validation errors; no fitting."""
from collections import Counter,defaultdict
from contextlib import closing
import bisect
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
from . import config,features_relay,ordinary_value_review_common as common
from .native import ROOT,Native,flatten
from .neural_relay import Model
from .puct_cpu_inference import CpuBatchEvaluator
from .value_learning import model_frozen_digest

SCHEMA="ordinary-value-six-root-saved-validation-review-v1"
OUTPUT=ROOT/"build/ordinary-value-validation-review.json"
CONTROL=ROOT/"data/quality-train-v1/model.json"
CANDIDATE=ROOT/"data/ordinary-value-pilot-v1/model.json"
LAMBDAS=(0.,.25,.5,.75,1.)
SOURCES=("congklak/ordinary_value_validation_review.py","tools/review-ordinary-value-validation.py","tests/ordinary_value_validation_review_author.py","docs/ORDINARY_VALUE_VALIDATION_REVIEW.md","congklak/ordinary_value_review_common.py","docs/ORDINARY_VALUE_REVIEW_SCOPE.md")
PRIOR=("data/quality-train-v1/model.json","data/ordinary-value-pilot-v1/model.json","data/ordinary-value-pilot-v1/result.json","data/ordinary-value-pilot-v1/catalog.sqlite3","data/ordinary-value-labels-v1/work.sqlite3","data/ordinary-value-labels-v1/manifest.json","build/ordinary-value-registry.json","build/ordinary-value-data.json","build/ordinary-value-rows.json","build/ordinary-value-plan.json","build/ordinary-value-model-witness.json","build/quality-data.json","build/sampling-review.json","congklak/features_relay.py","congklak/native.py","congklak/neural_relay.py","congklak/puct_cpu_inference.py","congklak/value_learning.py","build/congklak_native.dll","build/congklak_puct_cpu.dll")
require,read,sha=common.require,common.read,common.sha


def table(db,name,key="state"):
    rows={}
    for key,body,digest in db.execute("SELECT "+key+",body,hash FROM "+name+" ORDER BY "+key):
        value=__import__("json").loads(body);require(config.digest(value)==digest,"Corrupt saved "+name);rows[key]=value
    return rows


def rawkey(p):return bytes(flatten(p)+[p["controller"]]).hex()


def relative_pits(p):
    actor=p["controller"];return p["pits"][actor]+p["pits"][1-actor]


def distance(a,b):return sum(abs(x-y) for x,y in zip(relative_pits(a),relative_pits(b)))


def frozen_forward(model,x,d,r,hidden=None):
    w=model.weights;h=list(x) if hidden is None else hidden
    if hidden is None:
        for prefix in ("layers.0","layers.2"):h=[max(0.,sum(a*b for a,b in zip(row,h))+bias) for row,bias in zip(w[prefix+".weight"],w[prefix+".bias"])]
    t=sum(a*b for a,b in zip(w["layers.4.weight"][0],h))+w["layers.4.bias"][0];z=math.tanh(t);logits=[sum(a*b for a,b in zip(row,h))+bias for row,bias in zip(w["policy.weight"],w["policy.bias"])]
    return dict(hidden=h,preactivation=t,residual=z,value_player0=d+r*z,policy_logits=logits)


def errors(prediction,y,d,r,sign):
    z=(y-d)/r;player0=prediction-y
    return dict(target_player0=y,target_actor=sign*y,target_residual=z,prediction_player0=prediction,prediction_actor=sign*prediction,prediction_residual=(prediction-d)/r,signed_error_player0=player0,signed_error_actor=sign*player0,signed_error_residual=player0/r,absolute_error_seeds=abs(player0),absolute_error_residual=abs(player0/r))


def aggregate(roots,horizon):
    def metric(values):
        n=len(values);control=math.fsum(row["horizons"][horizon]["control"]["absolute_error_seeds"] for row in values)/n;candidate=math.fsum(row["horizons"][horizon]["candidate"]["absolute_error_seeds"] for row in values)/n
        return dict(roots=n,control_mae_seeds=control,candidate_mae_seeds=candidate,mae_delta_seeds=candidate-control,control_signed_player0_bias=math.fsum(row["horizons"][horizon]["control"]["signed_error_player0"] for row in values)/n,candidate_signed_player0_bias=math.fsum(row["horizons"][horizon]["candidate"]["signed_error_player0"] for row in values)/n,control_signed_actor_bias=math.fsum(row["horizons"][horizon]["control"]["signed_error_actor"] for row in values)/n,candidate_signed_actor_bias=math.fsum(row["horizons"][horizon]["candidate"]["signed_error_actor"] for row in values)/n,improved=sum(row["horizons"][horizon]["absolute_error_delta"]<0 for row in values),worsened=sum(row["horizons"][horizon]["absolute_error_delta"]>0 for row in values),tied=sum(row["horizons"][horizon]["absolute_error_delta"]==0 for row in values))
    values=list(roots.values());return dict(all=metric(values),by_band={band:metric([row for row in values if row["stratum"].split("/")[0]==band]) for band in ("25-49","50-74")})


def reconstruct_exposure(groups,identity):
    require(identity["steps"]==200 and identity["batch_size"]==64 and identity["seed"]==20261028,"Saved sampler schedule differs");rootmap={};cdf=[];total=0.;counts=Counter(row["stratum"] for row in groups["rehearsal"])
    for name in ("high","low"):
        byroot=defaultdict(list)
        for index,row in enumerate(groups[name]):byroot[row["state_key"]].append(index)
        require(len(byroot)==12 and all(len(ids)==2 for ids in byroot.values()),"Saved paired roots differ");rootmap[name]={key:byroot[key] for key in sorted(byroot)}
    for row in groups["rehearsal"]:
        p=.9/len(groups["rehearsal"])+.1/(len(counts)*counts[row["stratum"]]);require(row["sampling_probability"]==p,"Saved rehearsal weights differ");total+=p;cdf.append(total)
    require(config.canonical(identity["sampler"])==config.canonical({**rootmap,"rehearsal_cdf":cdf}),"Reconstructed sampler differs from accepted training identity")
    rng=random.Random(identity["seed"]);draws={name:Counter() for name in groups}
    for _ in range(200):
        for name in ("high","low"):
            keys=sorted(rootmap[name])
            for _ in range(16):draws[name][rng.choice(rootmap[name][rng.choice(keys)])]+=1
        for _ in range(32):draws["rehearsal"][bisect.bisect_right(cdf,rng.random()*cdf[-1])]+=1
    ledger={};summary={}
    for name,rows in groups.items():
        state_counts=Counter();family_counts=Counter();n=sum(draws[name].values())
        for index,row in enumerate(rows):
            actual=draws[name][index];p=(.25/12/2 if name!="rehearsal" else .5*row["sampling_probability"]);state_counts[row["state_key"]]+=actual;family_counts[row["family"]]+=actual
            ledger[row["id"]]=dict(group=name,state_key=row["state_key"],family=row["family"],stratum=row["stratum"],target_kind=row["target_kind"],target_player0=row["target_player0"],source_row_sha256=row["source_row_sha256"],actual_draws=actual,expected_draws=12800*p,overall_sampling_probability=p)
        summary[name]=dict(pool_records=len(rows),draws=n,distinct_records_drawn=sum(v>0 for v in draws[name].values()),distinct_states_drawn=sum(v>0 for v in state_counts.values()),distinct_families_drawn=sum(v>0 for v in family_counts.values()),state_effective_support=n*n/sum(v*v for v in state_counts.values()),scope="Exact reconstruction from accepted seed/CDF/order; not a separately recorded historical draw log or independent-sample count")
    require({name:summary[name]["draws"] for name in groups}==dict(high=3200,low=3200,rehearsal=6400),"Draw conservation differs")
    return dict(total_draws=12800,by_group=summary,ledger=ledger,final_sampler_rng_sha256=config.digest(rng.getstate()),copy_of_prior_audit_ledger=False),draws


def checkpoint_metrics():
    result=read(ROOT/"data/ordinary-value-pilot-v1/result.json");items=[];files=[]
    with closing(sqlite3.connect((ROOT/"data/ordinary-value-pilot-v1/catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for step,name,digest in db.execute("SELECT step,name,sha FROM checkpoints ORDER BY step"):
            path=ROOT/"data/ordinary-value-pilot-v1"/name;require(path.resolve().parent==(ROOT/"data/ordinary-value-pilot-v1").resolve() and sha(path)==digest,"Accepted checkpoint physical hash changed");files.append(dict(step=step,path=path.relative_to(ROOT).as_posix(),sha256=digest))
        for step,body,digest in db.execute("SELECT step,body,hash FROM history ORDER BY step"):
            metric=__import__("json").loads(body);require(config.digest(metric)==digest,"Saved metric history corrupt");items.append(dict(step=step,metrics=metric,metrics_sha256=digest))
    require([v["step"] for v in items]==[v["step"] for v in files]==list(range(20,201,20)),"Saved checkpoint cadence differs")
    return dict(initial=result["initial_metrics"],checkpoints=items,checkpoint_files=files,selected_final_step=200,scope="Previously saved in-training metrics plus physical PT hashes only; no new validation per checkpoint, tensor loading, selection or trajectory inference")


def ancestral_exposure(groups,exposure):
    admitted=read(ROOT/"build/quality-data.json");review=read(ROOT/"build/sampling-review.json");observed=review["observed_accepted_run"];keys=set(admitted["train_state_keys"]);rehearsalkeys={row["state_key"] for row in groups["rehearsal"]}
    require(review["passed"] is True and review["inputs"]["build/quality-data.json"]==sha(ROOT/"build/quality-data.json")==review["corpus"]["admitted_manifest_sha256"],"Ancestral draw proof corpus differs")
    require(len(keys)==admitted["train_rows"]==review["corpus"]["rows"]==observed["unique_roots_drawn"]==7669 and rehearsalkeys<=keys,"Positive ancestral draw cardinality proof differs")
    currentzero=sum(exposure["ledger"][row["id"]]["actual_draws"]==0 for row in groups["rehearsal"])
    return dict(admitted_quality_roots=len(keys),observed_unique_quality_roots_drawn=observed["unique_roots_drawn"],observed_quality_draws=observed["total_draws"],rehearsal_unique_roots=len(rehearsalkeys),rehearsal_roots_with_proven_positive_ancestral_draws=len(rehearsalkeys),current_zero_draw_rehearsal_rows_with_positive_ancestral_draws=currentzero,root_draw_ledger_sha256=observed["root_draw_ledger_sha256"],exact_ancestral_per_root_counts_reconstructed=False,nonzero_gradient_or_causal_effect_proven=False,scope="The accepted run drew every root in its admitted corpus; current rehearsal membership proves positive ancestral draws, not their exact individual counts or causal/nonzero-gradient effects")


def build():
    require("torch" not in sys.modules,"Review forbids Torch imports");baseline=common.baseline();checkpoint_history=checkpoint_metrics();priorpins=common.pins(PRIOR+tuple(item["path"] for item in checkpoint_history["checkpoint_files"]))
    require(all(baseline["inputs"].get(name)==digest for name,digest in priorpins.items()),"Saved source differs from accepted review baseline");source=Model(CONTROL);candidate=Model(CANDIDATE);require(model_frozen_digest(source)==model_frozen_digest(candidate),"Policy/trunk FP32 tensors changed")
    training=read(ROOT/"data/ordinary-value-pilot-v1/result.json");groups=read(ROOT/"build/ordinary-value-rows.json")["groups"];require({name:len(v) for name,v in groups.items()}==dict(high=24,low=24,rehearsal=7539),"Saved optimizer cohorts differ");exposure,draws=reconstruct_exposure(groups,training["identity"]);exposure["ancestral_quality_draw_exposure"]=ancestral_exposure(groups,exposure)
    with closing(sqlite3.connect((ROOT/"data/ordinary-value-labels-v1/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");roots=table(db,"roots");labels={s:table(db,s+"_labels") for s in ("h8","h10","h12")};attempts={s:table(db,s+"_attempts") for s in labels}
    expected=read(ROOT/"build/ordinary-value-registry.json")["rows"];require(config.canonical(roots)==config.canonical(expected) and len(roots)==6 and all(set(values)==set(roots) for values in labels.values()),"Exact six requested-reference roots differ")
    native=Native();neighbors={};rehearsal={row["id"]:row for row in groups["rehearsal"]};forward_rows={};descriptors={}
    trainfamilies={row["family"] for values in groups.values() for row in values}
    for key,row in roots.items():
        require(row["family"] not in trainfamilies,"Validation family leaked into admitted training pool");ordered=sorted(rehearsal.values(),key=lambda other:(distance(row["state"],other["state"]),other["id"]))[:5];neighbors[key]=ordered
        for other in ordered:forward_rows[rawkey(other["state"])]=other
    ordinary={}
    for name in ("high","low"):
        for row in groups[name]:ordinary.setdefault(row["state_key"],[]).append(row);forward_rows[rawkey(row["state"])]=row
    for key,row in roots.items():
        x,d,r=features_relay.encode(row["state"],127,native);forward_rows[rawkey(row["state"])]=dict(state=row["state"],cached_features=dict(x=x,d=d,r=r))
    require(len(forward_rows)<=60,"Literal inference unique-state budget exceeded")
    for key,row in sorted(forward_rows.items()):
        f=row["cached_features"];a=frozen_forward(source,f["x"],f["d"],f["r"]);b=frozen_forward(candidate,f["x"],f["d"],f["r"],a["hidden"]);require(a["policy_logits"]==b["policy_logits"],"Literal policy changed")
        descriptors[key]=dict(control=a,candidate=b,d=f["d"],r=f["r"])
    parity=0.
    with CpuBatchEvaluator(source,cache_capacity=0,max_batch=1) as control_eval,CpuBatchEvaluator(candidate,cache_capacity=0,max_batch=1) as candidate_eval:
        for row in roots.values():
            value=descriptors[rawkey(row["state"])]
            for name,evaluator in (("control",control_eval),("candidate",candidate_eval)):
                actual=evaluator.evaluate_many([row["state"]])[0];reference=value[name];parity=max(parity,abs(actual["value_player0"]-reference["value_player0"]),max(abs(a-b) for a,b in zip(actual["policy_logits"],reference["policy_logits"])));require(parity<1e-10,"Literal/native forward mismatch")
    delta_bias=candidate.weights["layers.4.bias"][0]-source.weights["layers.4.bias"][0];delta_weights=[b-a for a,b in zip(source.weights["layers.4.weight"][0],candidate.weights["layers.4.weight"][0])];review={};coverage={}
    for key,row in roots.items():
        p=row["state"];desc=descriptors[rawkey(p)];a,b=desc["control"],desc["candidate"];d,r=desc["d"],desc["r"];sign=1 if p["controller"]==0 else -1;normal=native.normalize(flatten(p),actor=p["controller"],total=98);legal=normal["legal_actions"];horizons={}
        for stage,values in labels.items():
            item=values[key];label=item["label"];attempt=attempts[stage][key];require(item["root_sha256"]==row["source_root_sha256"]==config.digest(row["root"]) and label["choice_depth"]==int(stage[1:]) and attempt["status"]=="complete" and attempt["requested_depth"]==int(stage[1:]) and attempt["label_sha256"]==config.digest(item),"Requested-horizon label provenance differs")
            y=label["target_player0"];ea=errors(a["value_player0"],y,d,r,sign);eb=errors(b["value_player0"],y,d,r,sign);z=(y-d)/r;require(-1<=z<=1,"Finite target outside residual interval")
            horizons[stage]=dict(target_kind="approximate_finite_choice_horizon",requested_depth=int(stage[1:]),label_sha256=config.digest(label),report_sha256=config.digest(item["report"]),source_root_sha256=item["root_sha256"],target_player0=y,control=ea,candidate=eb,absolute_error_delta=eb["absolute_error_seeds"]-ea["absolute_error_seeds"],H10_aggregate_contribution=(eb["absolute_error_seeds"]-ea["absolute_error_seeds"])/6 if stage=="h10" else None,physically_bounded_residual_target=True,exact_finite_head_representable=abs(z)<1,saturation_endpoint=abs(z)==1,target_preactivation=math.atanh(z) if abs(z)<1 else None)
        weight_part=sum(v*h for v,h in zip(delta_weights,a["hidden"]));actual=b["preactivation"]-a["preactivation"];require(abs(actual-(delta_bias+weight_part))<1e-12,"Head algebra decomposition mismatch")
        target_t=horizons["h10"]["target_preactivation"];needed=None if target_t is None else target_t-a["preactivation"]
        review[key]=dict(state=p,state_key=key,family=row["family"],cohort=row["cohort"],stratum=row["stratum"],source_wrapper_sha256=config.digest(row),original_root_sha256=row["source_root_sha256"],actor=p["controller"],actor_sign=sign,D_player0=d,signed_active=r,active_seeds=abs(r),legal_mask=row["legal_mask"],legal_actions_count=len(legal),control_prediction_player0=a["value_player0"],candidate_prediction_player0=b["value_player0"],prediction_change_player0=b["value_player0"]-a["value_player0"],prediction_change_actor=sign*(b["value_player0"]-a["value_player0"]),control_policy_logits=a["policy_logits"],candidate_policy_logits=b["policy_logits"],control_legal_argmax=max(legal,key=lambda action:(a["policy_logits"][action],-action)),candidate_legal_argmax=max(legal,key=lambda action:(b["policy_logits"][action],-action)),horizons=horizons,
            head_shift=dict(control_preactivation=a["preactivation"],candidate_preactivation=b["preactivation"],actual_preactivation_change=actual,bias_contribution=delta_bias,hidden_weight_contribution=weight_part,fixed_store_offset_change=0,H10_needed_preactivation_change=needed,H10_direction_aligned=None if needed is None else actual*needed>=0,control_seed_gain_per_preactivation=r*(1-a["residual"]**2),candidate_seed_gain_per_preactivation=r*(1-b["residual"]**2),scope="Algebraic final-weight decomposition; not optimizer-gradient or causal attribution"),H12_minus_H10_target=horizons["h12"]["target_player0"]-horizons["h10"]["target_player0"])
        coverage[key]=dict(metric="Raw14pit L1 after controller's pits then opponent's pits; stores omitted",tie_break="Admitted rehearsal rowID lexical; no targets/errors used",candidate_pool_records=7539,query_family_excluded=True,neighbors=[dict(row_id=other["id"],state_key=other["state_key"],family=other["family"],stratum=other["stratum"],actor=other["state"]["controller"],distance_seeds=distance(p,other["state"]),target_kind=other["target_kind"],target_player0=other["target_player0"],source_row_sha256=other["source_row_sha256"],actual_draws=exposure["ledger"][other["id"]]["actual_draws"],control_prediction_player0=descriptors[rawkey(other["state"])]["control"]["value_player0"],candidate_prediction_player0=descriptors[rawkey(other["state"])]["candidate"]["value_player0"]) for other in neighbors[key]],scope="Feature-only board similarity, not calibrated uncertainty, support sufficiency, transferable truth or a causal coverage explanation")
    pairs=[]
    for key,rows in sorted(ordinary.items()):
        require(len(rows)==2,"Ordinary target pair changed");ys=[r["target_player0"] for r in rows];ns=[exposure["ledger"][r["id"]]["actual_draws"] for r in rows];desc=descriptors[rawkey(rows[0]["state"])];pairs.append(dict(state_key=key,family=rows[0]["family"],stratum=rows[0]["stratum"],actor=rows[0]["state"]["controller"],record_ids=[r["id"] for r in rows],source_row_sha256=[r["source_row_sha256"] for r in rows],terminal_returns_player0=ys,absolute_pair_gap_seeds=abs(ys[1]-ys[0]),actual_record_draws=ns,actual_root_draws=sum(ns),raw_pair_mean_player0=sum(ys)/2,draw_weighted_return_mean_player0=sum(y*n for y,n in zip(ys,ns))/sum(ns),control_prediction_player0=desc["control"]["value_player0"],candidate_prediction_player0=desc["candidate"]["value_player0"],scope="Two noisy observed fixed-control policy returns; descriptive means do not replace individual training targets or estimate H10 truth"))
    interpolation={}
    for coefficient in LAMBDAS:
        predictions={key:row["D_player0"]+row["signed_active"]*math.tanh(row["head_shift"]["control_preactivation"]+coefficient*row["head_shift"]["actual_preactivation_change"]) for key,row in review.items()}
        interpolation[str(coefficient)]=dict(coefficient=coefficient,predictions_player0=predictions,mae_seeds={stage:math.fsum(abs(predictions[key]-row["horizons"][stage]["target_player0"]) for key,row in review.items())/6 for stage in labels})
    horizons={stage:aggregate(review,stage) for stage in labels};h10=horizons["h10"]["all"];gaps=[p["absolute_pair_gap_seeds"] for p in pairs]
    summary=dict(roots=6,H10_control_mae=h10["control_mae_seeds"],H10_candidate_mae=h10["candidate_mae_seeds"],H10_mae_delta=h10["mae_delta_seeds"],improved_roots=h10["improved"],worsened_roots=h10["worsened"],tied_roots=h10["tied"],policy_logits_identical=True,high_roots_all_actor1=all(r["actor"]==1 for r in review.values() if r["stratum"].startswith("50-74/")),max_H10_H12_target_shift=max(abs(r["H12_minus_H10_target"]) for r in review.values()),finite_head_H10_representable_roots=sum(r["horizons"]["h10"]["exact_finite_head_representable"] for r in review.values()),H10_direction_aligned_roots=sum(r["head_shift"]["H10_direction_aligned"] for r in review.values()),H10_prediction_crossings=sum(r["horizons"]["h10"]["control"]["signed_error_player0"]*r["horizons"]["h10"]["candidate"]["signed_error_player0"]<0 for r in review.values()),ordinary_target_pairs=24,pairs_with_different_returns=sum(v>0 for v in gaps),median_pair_gap_seeds=statistics.median(gaps),max_pair_gap_seeds=max(gaps),total_draws=12800)
    body=dict(schema=SCHEMA,passed=True,summary=summary,validation_roots=review,horizon_summary=horizons,policy_invariance=dict(frozen_fp32_weights_sha256=model_frozen_digest(source),six_frozen_tensors_identical=True,frozen_parameters=12807,per_root_logits_identical=True,per_root_legal_argmax_identical=all(r["control_legal_argmax"]==r["candidate_legal_argmax"] for r in review.values()),all_inputs_policy_identity_from_frozen_parameters=True),head_change=dict(bias_delta=delta_bias,weight_delta_L1=math.fsum(abs(v) for v in delta_weights),weight_delta_L2=math.sqrt(math.fsum(v*v for v in delta_weights)),scope="Final65parameter difference only; no per-sample gradient attribution"),head_interpolation=dict(definition="Mathematical final-head preactivation interpolation in double arithmetic, not FP32quantized hypothetical weights or actual optimizer checkpoints",coefficients=list(LAMBDAS),diagnostic_only=True,model_or_checkpoint_selected=False,values=interpolation),exposure=exposure,ordinary_training_pairs=pairs,coverage=coverage,saved_checkpoint_metrics=checkpoint_history,inference_budget=dict(unique_fullstates_with_literal_trunk_forwards=len(forward_rows),maximum_unique_fullstates=60,validation_states=6,ordinary_start_states=24,nearest_rehearsal_union=len({rawkey(v["state"]) for values in neighbors.values() for v in values}),native_parity_states=6,maximum_literal_native_difference=parity,new_teacher_searches=0,new_PUCT_searches=0,new_optimizer_steps=0,torch_imported=False),interpretation=["Six selected dependent finite-reference roots are not a release or strength test.","All three high-band roots have actor1, confounding band and actor in this panel.","Requested H10 remains primary; rescoring fixed predictions against H8/H12 is sensitivity, not deeper-target-training ablation or game truth.","Representability here checks residual intervals; shared-head capacity or optimization cause is not established.","Draw reconstruction is accepted-seed bookkeeping; repeated observations do not add independent states or outcomes.","All current rehearsal roots have positive ancestral quality-run draw exposure, including current-zero rows; exact individual ancestral counts, nonzero gradients and causal effects remain unclaimed.","Nearest board features do not transfer terminal/H8 labels as validation H10 truth.","Head interpolation and algebraic bias/weight contributions select no model, checkpoint, weighting or policy."],baseline_sha256=sha(common.BASELINE),control_model_sha256=source.file_sha256,candidate_model_sha256=candidate.file_sha256,inputs={**priorpins,**common.pins(SOURCES+(str(common.BASELINE),))})
    body["sha256"]=config.digest(body);return body


def validate_report(report):
    require(config.canonical(report)==config.canonical(build()),"Saved validation review metric/source identities differ from full rederivation");return True


def run(path=OUTPUT):
    body=build();common.publish(path,body);return body


def verify(path=OUTPUT):
    body=read(path);validate_report(body);return body
