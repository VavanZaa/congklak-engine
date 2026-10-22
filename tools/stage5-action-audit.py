"""Independently recompute action-experiment selection and committed arenas."""
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from congklak import config,pilot,campaign,action_targets
from congklak.neural_action import Model,NeuralNative
from stage5_action_audit_acceptance import value_and_policy,literal_position,pilot_rows
import reference as literal
WORK=ROOT/"build/stage5-action-audit";WORK.mkdir(parents=True,exist_ok=True)

def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def require(test,message):
    if not test:raise AssertionError(message)
def interval(values):
    rng=random.Random(20261015);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4096)) if values else []
    return [samples[102],samples[3993]] if samples else None
def frozen_prior():
    old=load(ROOT/"build/stage5-acceptance.json");selected={}
    for name,digest in old["inputs"].items():
        if name.startswith(("native/","data/training-stage5-")) or name in ("congklak/training.py","congklak/training_v2.py","congklak/training_v3.py","congklak/neural.py","congklak/neural_fast.py","congklak/features_v3.py","tools/benchmark-stage5.py","build/stage5-benchmark.json","build/stage5-evaluation-results.json","build/congklak_native.dll","build/congklak_neural.dll","build/congklak_neural_fast.dll"):
            require(sha(ROOT/name)==digest,"Prior scientific input changed: "+name);selected[name]=digest
    for name,digest in load(ROOT/"build/training-acceptance.json")["inputs"].items():
        require(sha(ROOT/name)==digest,"Prior environment/tool changed");selected[name]=digest
    return selected

def evaluation():
    plan=load(ROOT/"build/stage5-action-training-plan.json");r=load(ROOT/"build/stage5-action-training-results.json")
    require(plan==r["plan"],"Selection plan changed")
    require(plan["variants"]==["A","B","C"] and plan["seeds"]==[2718,31415] and plan["parameters"]==7496 and plan["width"]==64,"Matched experiment coverage changed")
    require(plan["steps"]==8000 and plan["batch_size"]==512 and plan["checkpoint_every"]==100,"Training budget changed")
    for name,key in (("congklak/training_v4.py","trainer_sha256"),("congklak/action_targets.py","action_targets_sha256"),("tools/stage5-action-training.py","source_sha256")):
        require(sha(ROOT/name)==plan[key],"Frozen training source changed")
    train=list(pilot_rows("train"));dev=list(pilot_rows("development"));calibration=action_targets.ordinary_calibration(train)
    require(len(train)==8000 and len(dev)==1000,"Unexpected or augmented rows")
    rows={(row["variant"],row["seed"]):row for row in r["rows"]};require(len(rows)==6,"Missing/duplicated candidates")
    schedules={}
    for (variant,seed),row in rows.items():
        folder=(ROOT/row["model"]).parent;result=load(folder/"result.json");identity=result["identity"]
        require(result["status"]=="complete" and result["steps_completed"]==8000 and result["draws"]==4096000 and result["parameters"]==7496,"Incomplete/mismatched candidate")
        require(identity["variant"]==variant and identity["seed"]==seed and identity["ordinary_bias_calibration"]==calibration,"Objective/calibration ownership changed")
        require(identity["policy_objective"]["temperature_seeds"]==4 and identity["policy_objective"]["coefficient"]==plan["objectives"][variant]["policy_coefficient"] and identity["bias_coefficient"]==plan["objectives"][variant]["bias_coefficient"],"Loss contract changed")
        require(identity["source_sha256"]==plan["trainer_sha256"] and identity["oracle_training"] is False and identity["sealed_final_test_consumed"] is False,"Source/ancestry differs")
        for module,digest in identity["source_dependencies_sha256"].items():require(sha(ROOT/(module.replace(".","/")+".py"))==digest,"Loader/target dependency changed")
        with closing(sqlite3.connect((folder/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            history=[]
            for step,body,digest in db.execute("SELECT step,body,hash FROM history ORDER BY step"):
                m=json.loads(body);require(config.digest(m)==digest,"Corrupt model history");history.append((step,m["finite_choice_horizon"]["uniform_stratum_mae_seeds"]))
            require([s for s,m in history]==list(range(100,8001,100)),"Unequal selection opportunities")
            best,score=min(history,key=lambda pair:pair[1]);require(best==row["best_step"]==result["best_step"],"Extra/post-hoc checkpoint selection")
            body,digest=db.execute("SELECT body,hash FROM checkpoints WHERE step=? ORDER BY id DESC LIMIT 1",(best,)).fetchone();meta=json.loads(body)
            require(config.digest(meta)==digest and meta["weights_sha256"]==row["weights_sha256"],"Selected checkpoint/export differs")
            schedules[f"{variant}/{seed}"]={"scheduled_opportunities":80,"best_step":best,"best_macro_mae":score}
        model=Model(ROOT/row["model"]);value=load(ROOT/row["value_model"])
        require(model.id==row["weights_sha256"] and model.envelope["model"]["identity"]==identity,"Policy export identity differs")
        require(config.digest(value["model"])==value["sha256"] and value["model"]["identity"]==identity,"Value-only export envelope differs")
        require(value["model"]["weights"]=={k:v for k,v in model.envelope["model"]["weights"].items() if k.startswith("layers.")},"Value and policy exports use different learned leaves")
    for variant in ("A","B","C"):
        best=min((rows[(variant,seed)] for seed in (2718,31415)),key=lambda row:row["macro_mae"])
        require(best==r["selected"][variant],"Representative selected using another metric")
    winner=min(("B","C"),key=lambda variant:statistics.mean(rows[(variant,seed)]["macro_mae"] for seed in (2718,31415)))
    require(r["selected_control"]==r["selected"]["A"] and r["selected_challenger"]==r["selected"][winner],"Arena changed candidate selection")
    verified={}
    for name,row in (("control",r["selected_control"]),("challenger",r["selected_challenger"])):
        model=Model(ROOT/row["model"]);buckets=defaultdict(list);policy=[];max_raw=0;max_logit=0
        with NeuralNative(model,mode="value-policy",backend="avx2") as native:
            for example in dev:
                p=literal_position(example["state"]);raw,units,logits=value_and_policy(p,model.envelope["model"]);actual=native.evaluate(example["state"])
                require(actual["value_units"]==units,"Selected model units differ");max_raw=max(max_raw,abs(actual["value_player0"]-raw))
                max_logit=max(max_logit,max(abs(a-b) for a,b in zip(actual["policy_logits"],logits)))
                buckets[example["stratum"]].append(abs(raw-example["label"]["target_player0"]))
                legal=list(literal.legal_actions(p));choice=max(legal,key=lambda a:(logits[a],-a));sign=1 if literal.normalize(p).controller==0 else -1
                q={a["pit"]:a["value_player0"] for a in example["label"]["action_scores"]}
                policy.append(sign*(example["label"]["target_player0"]-q[choice]))
        require(max_raw<=1e-10 and max_logit<=1e-10,"Independent full head parity exceeded")
        discrepancy=max(abs(statistics.mean(v)-row["metrics"]["strata"][k]["mae_seeds"]) for k,v in buckets.items());require(discrepancy<=1e-4,"Reported value metrics differ")
        require(abs(statistics.mean(policy)-row["policy"]["mean_root_H6_Q_regret_seeds"])<1e-10,"Reported policy preferences differ")
        ordinary={k:statistics.mean(v) for k,v in buckets.items() if k.endswith("/ordinary") and len(v)>=20}
        require(all(value<=row["metrics"]["strata"][k]["H_v1_mae_seeds"] for k,value in ordinary.items())==row["ordinary_gate_passed"],"Ordinary promotion gate differs")
        verified[name]=dict(variant=row["variant"],seed=row["seed"],model=row["model"],weights_sha256=row["weights_sha256"],development_roots=len(dev),max_raw_reference_error=max_raw,max_policy_reference_error=max_logit,max_reported_mae_error=discrepancy,ordinary_gate_passed=row["ordinary_gate_passed"],ordinary_mae_seeds=ordinary,policy_mean_root_H6_Q_regret_seeds=statistics.mean(policy))
    return dict(plan_sha256=sha(ROOT/"build/stage5-action-training-plan.json"),fixed_parameters=7496,selection=schedules,selected=verified,train_calibration=calibration,train_roots=8000,dev_roots=1000,successor_value_rows=0,sealed_target_bodies_read=False)

def benchmark():
    result=load(ROOT/"build/stage5-action-benchmark-v2.json");plan=load(ROOT/"build/stage5-action-benchmark-plan-v2.json");identity=result["identity"]
    require(result["sealed_final_test_consumed"] is False and result["energy_measured"] is False,"Undeclared sealed/energy claim")
    require(identity["source_sha256"]==sha(ROOT/"tools/benchmark-stage5-action-v2.py") and identity["plan_sha256"]==config.digest(plan),"Benchmark protocol changed")
    require(identity["selection_sha256"]==sha(ROOT/"build/stage5-action-training-results.json"),"Selection changed after arena")
    for name,digest in {**identity["bindings"],**identity["cohort_sources"]}.items():require(sha(ROOT/name)==digest,"Benchmark dependency changed")
    require(plan["budgets_ms"]==[25,100] and plan["policy_min_depth"]==2 and plan["tt_mb"]==16 and plan["use_pvs"] is True and plan["max_depth"]==16 and plan["arena_ms"]==20,"Budgets differ")
    require(identity["cpu"]["protocol"]==plan["cpu_protocol"]=="own-process/declared-os-performance-core/v2" and identity["cpu"]["selected"]==plan["cpu_declared"] and identity["cpu"]["mask"]==1<<identity["cpu"]["selected"]["logical"] and not identity["cpu"]["power_settings_changed"] and not identity["cpu"]["priority_changed"],"CPU protocol differs")
    require("original_mask" not in identity["cpu"] and not {"parked","allocated","allocated_to_process"}&set(identity["cpu"]["selected"]),"Dynamic CPU observations included in scientific identity")
    require(len(plan["cases"])==391 and len(plan["arena_starts"])==24 and len({r["family"] for r in plan["arena_starts"]})==24,"Root/family coverage differs")
    selection=load(ROOT/"build/stage5-action-training-results.json")
    require(identity["baseline_dll_sha256"]==sha(ROOT/"build/congklak_native.dll"),"Baseline DLL changed")
    expected={"h_none":("selected_challenger","h-none"),"h_policy":("selected_challenger","h-policy"),"control_value":("selected_control","value-none"),"joint_value":("selected_challenger","value-none"),"joint_policy":("selected_challenger","value-policy")}
    require(set(identity["models"])==set(expected),"Unexpected engine identities")
    for name,(owner,mode) in expected.items():
        selected=selection[owner];m=identity["models"][name]
        require(m["weights_sha256"]==selected["weights_sha256"] and m["model_file_sha256"]==sha(ROOT/selected["model"]) and m["dll_sha256"]==sha(ROOT/"build/congklak_neural_action.dll"),"Engine uses another model/DLL")
        require(m["mode"]==mode and m["policy_min_depth"]==2 and m["backend"]=="avx2-double-packed","Engine mode/threshold/backend differs")
    excluded={r["state_key"] for r in load(ROOT/"build/stage5-audit/legacy-exact-aliases.json")["cases"]};require(not excluded&{r["state_key"] for r in plan["cases"]},"Legacy sealed exact alias reintroduced")
    cohorts=Counter((r["cohort"],r["reference"]["kind"]) for r in plan["cases"]);require(cohorts==Counter({("original_development","finite_choice_horizon"):100,("original_development","exact_optimal_margin"):32,("ordinary_current_development","finite_choice_horizon"):259}),"Reference cohorts changed")
    # Only state/split/family metadata from sealed partitions; never targets.
    protected={}
    for name,folder in (("original",pilot.DEFAULT_PATH),("current",campaign.DEFAULT)):
        manifest=pilot.load(folder);require(identity["datasets"][name]==manifest["semantic_sha256"],"Dataset changed")
        with closing(sqlite3.connect((folder/manifest["directory"]/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            protected[name]=[(pilot.quotient(pilot.from_key(bytes(k))),s,f) for k,s,f in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples")]
    for source in plan["cases"]:
        q=pilot.quotient(source["state"]);name="original" if source["cohort"]=="original_development" else "current"
        require({s for other,s,f in protected[name] if other==q}=={"development"},"Evaluation aliases sealed/train state")
        require(all(not any(q==other and s=="train" for other,s,f in items) for items in protected.values()),"Cross-generation training alias")
    ledger=ROOT/"data/stage5-action-benchmark-v2/work.sqlite3"
    with closing(sqlite3.connect(ledger.as_uri()+"?mode=ro",uri=True)) as db:
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();require(json.loads(body)==identity and config.digest(identity)==digest,"Catalog identity differs")
        cases=[]
        for idx,body,digest in db.execute("SELECT idx,body,hash FROM cases ORDER BY idx"):
            c=json.loads(body);require(idx==len(cases) and config.digest(c)==digest,"Corrupt/noncontiguous case ledger");cases.append(c)
        games=[]
        for comparison,pair,seat,body,digest in db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat"):
            g=json.loads(body);require(g["comparison"]==comparison and g["pair"]==pair and g["candidate_seat"]==seat and config.digest(g)==digest,"Corrupt game ledger");games.append(g)
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='action_complete'").fetchone();require(config.digest(json.loads(body))==digest and json.loads(body)==dict(cases=391,games=144,report_sha256=config.digest(result)),"Final result lacks matching catalog commit")
        invocations=[]
        for body,digest in db.execute("SELECT body,hash FROM timing_invocations ORDER BY id"):
            observation=json.loads(body);require(config.digest(observation)==digest,"Corrupt timing observations")
            require(observation["verified"]==dict(group=0,logical=2,mask=4),"Invocation did not execute on declared CPU")
            require(type(observation["cpu"]["original_mask"]) is int and observation["cpu"]["original_mask"]&4,"Untruthful original process permission mask")
            invocations.append(observation)
        require(bool(invocations),"Missing truthful invocation observations")
    require(len(cases)==391 and len(games)==144,"Experiments incomplete")
    buckets=defaultdict(list);strata=defaultdict(list);equal_h=0;equal_v=0
    for idx,c in enumerate(cases):
        source=plan["cases"][idx]
        require(c["cpu_verified"]==dict(group=0,logical=2,mask=4),"Case CPU verification differs")
        for key in ("state_key","cohort","stratum","reference"):require(c[key]==source[key],"Case metadata differs")
        q={r["pit"]:r["value_player0"] for r in source["reference"]["action_scores"]};sign=1 if source["state"]["controller"]==0 else -1
        for budgetindex,ms in enumerate((25,100)):
            order=plan["engines"][idx%6:]+plan["engines"][:idx%6]
            if budgetindex:order.reverse()
            require(c["orders"][str(ms)]==order,"Timing order changed")
            values=c["measurements"][str(ms)];require(set(values)==set(plan["engines"]),"Missing engine")
            for name,v in values.items():
                choice=v["chosen"];require(choice is None or choice in q,"Illegal choice")
                require(v["stop_reason"] in ("max_depth","time_budget"),"Resource-failed/canceled case admitted")
                require(v["reference_optimal"]==(choice in source["reference"]["best_moves"] if choice is not None else None) and v["reference_regret"]==(sign*(source["reference"]["target_player0"]-q[choice]) if choice is not None else None),"Corrupt choice metric")
                buckets[(c["cohort"],str(ms),source["reference"]["kind"],name)].append(v);strata[(c["cohort"],str(ms),c["stratum"],name)].append(v)
            for names in (("H_v1","h_none"),("h_none","h_policy"),("joint_value","joint_policy")):
                a,b=[values[name]["report"] for name in names]
                if a and b and a["depth"]==b["depth"]:
                    require(a["value_player0"]==b["value_player0"] and a["best_moves"]==b["best_moves"],"Ordering alters finite-horizon values")
                    if names[0].startswith("joint"):equal_v+=1
                    else:equal_h+=1
    def aggregate(values):
        done=[v for v in values if v["chosen"] is not None]
        return dict(positions=len(values),no_completed_recommendation=len(values)-len(done),reference_optimal_accuracy=statistics.mean(v["reference_optimal"] for v in done) if done else None,mean_reference_regret_seeds=statistics.mean(v["reference_regret"] for v in done) if done else None,mean_completed_depth=statistics.mean(v["report"]["depth"] for v in done) if done else None,median_wall_ms=statistics.median(v["wall_ms"] for v in values),max_wall_ms=max(v["wall_ms"] for v in values))
    require(result["budget_results"]=={"/".join(k):aggregate(v) for k,v in buckets.items()} and result["strata"]=={"/".join(k):aggregate(v) for k,v in strata.items()},"Root summaries differ")
    arenas={};families={};replayed=0
    for comparison in plan["arena_comparisons"]:
        group=[g for g in games if g["comparison"]==comparison["id"]];require({(g["pair"],g["candidate_seat"]) for g in group}=={(pair,seat) for pair in range(24) for seat in (0,1)},"Missing/duplicated seats")
        for game in group:
            require(game["cpu_verified"]==dict(group=0,logical=2,mask=4),"Arena game CPU verification differs")
            start=plan["arena_starts"][game["pair"]];require(game["start"]==start["state"] and game["family"]==start["family"],"Arena start differs")
            p=literal_position(game["start"]);require(len(game["moves"])<=512,"Choice cap exceeded")
            for move in game["moves"]:require(not literal.normalize(p).terminal,"Move after terminal");p=literal.apply(p,move).position
            final={"pits":[list(p.counts[:7]),list(p.counts[8:15])],"stores":[p.counts[7],p.counts[15]],"controller":p.controller};require(game["final"]==final,"Literal final board differs")
            terminal=literal.normalize(p).terminal;margin=(p.counts[7]-p.counts[15])*(1 if game["candidate_seat"]==0 else -1);score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
            require(game["status"]==("terminal" if terminal else "unfinished") and game["candidate_final_margin"]==(margin if terminal else None) and game["candidate_match_score"]==score,"Literal outcome differs");replayed+=1
        paired=[];margins=[];details=[]
        for pair in range(24):
            two=[g for g in group if g["pair"]==pair];start=plan["arena_starts"][pair]
            p=start["state"];active=sum(map(sum,p["pits"]));d=p["stores"][0]-p["stores"][1];require(abs(d)<active,"Starting WDL already fixed")
            if all(g["status"]=="terminal" for g in two):paired.append(statistics.mean(g["candidate_match_score"] for g in two));margins.append(statistics.mean(g["candidate_final_margin"] for g in two));details.append(dict(pair=pair,family=start["family"],primary_paired_margin=margins[-1],secondary_wdl_score=paired[-1]))
        fallbacks={name:sum(g["fallbacks"].get(name,0) for g in group) for name in (comparison["candidate"],comparison["opponent"])};ci=interval(margins)
        gate=len(margins)==24 and not sum(fallbacks.values()) and ci[0]>0
        arenas[comparison["id"]]=dict(pairs=24,terminal_games=sum(g["status"]=="terminal" for g in group),unfinished_games=sum(g["status"]!="terminal" for g in group),complete_pairs=len(paired),candidate_wins=sum(g["candidate_match_score"]==1 for g in group),candidate_draws=sum(g["candidate_match_score"]==.5 for g in group),candidate_losses=sum(g["candidate_match_score"]==0 for g in group),paired_mean_score=statistics.mean(paired) if paired else None,paired_bootstrap_95_interval=interval(paired),primary_paired_mean_final_margin_seeds=statistics.mean(margins) if margins else None,primary_margin_bootstrap_95_interval=ci,primary_strength_gate_passed=gate,fallback_counts=fallbacks)
        families[comparison["id"]]=details
    require(result["arenas"]==arenas,"Primary seed-margin/bootstrap/gates differ")
    (WORK/"paired-margins.json").write_text(json.dumps(families,indent=2)+"\n")
    return dict(cases_recomputed=391,literal_games_replayed=replayed,known_exact_diagnostics=32,legacy_sealed_aliases_excluded=16,training_or_sealed_root_aliases=0,original_h_action_h_equaldepth_checks=equal_h,value_ordering_equaldepth_checks=equal_v,cpu=identity["cpu"],truthful_timing_invocations=len(invocations),primary_seed_margin_predeclared=True,arenas=arenas,sealed_target_bodies_read=False,energy_measured=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--without-benchmark",action="store_true");args=parser.parse_args()
    r=dict(schema="stage5-action-independent-audit-v1",recorded_at_utc=datetime.now(timezone.utc).isoformat(),passed=False,benchmark_audit_complete=False)
    try:
        tests=load(WORK/"tests.json");require(tests["passed"] and tests["tests_run"]==9 and not tests["failures"]+tests["errors"]+tests["skips"],"Independent acceptance incomplete")
        r["tests_run"]=9;r["independent_acceptance"]=tests;r["frozen_prior_inputs"]=frozen_prior();r["evaluation"]=evaluation()
        if not args.without_benchmark:r["benchmark"]=benchmark();r["benchmark_audit_complete"]=True
        r["passed"]=True
    except Exception as exc:r["error"]={"type":type(exc).__name__,"message":str(exc)}
    names=("tests/stage5_action_audit_acceptance.py","tools/stage5-action-audit.py","congklak/neural_action.py","native/neural_action.cpp","build/congklak_neural_action.dll","congklak/training_v4.py","congklak/action_targets.py","congklak/timing.py","congklak/timing_v2.py","tools/stage5-action-training.py","tools/benchmark-stage5-action.py","tools/benchmark-stage5-action-v2.py","build/stage5-action-training-plan.json","build/stage5-action-training-results.json","build/stage5-action-training-verification.json","build/stage5-action-benchmark-plan.json","build/stage5-action-benchmark-plan-v2.json","build/stage5-action-benchmark.json","build/stage5-action-benchmark-v2.json","build/stage5-action-audit/tests.json","build/stage5-action-audit/first-nine-check-failure.log","build/stage5-action-audit/additional-check-failure.json","build/stage5-action-audit/arena-identity-drift.json","build/stage5-action-audit/first-eight-v2.json")
    r["inputs"]={name:sha(ROOT/name) for name in names if (ROOT/name).exists()}
    if (ROOT/"build/stage5-action-training-results.json").exists():
        for row in load(ROOT/"build/stage5-action-training-results.json")["rows"]:
            for name in (row["model"],row["value_model"],str(Path(row["model"]).parent/"result.json")):r["inputs"][name.replace("\\","/")]=sha(ROOT/name)
    (WORK/"audit.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(dict(event="action_independent_audit_complete",passed=r["passed"],benchmark_audit_complete=r["benchmark_audit_complete"],error=r.get("error"))),flush=True)
    return 0 if r["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
