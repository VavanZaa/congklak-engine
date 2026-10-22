"""Independent final audit of Stage 5 immutable inputs and committed results."""
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
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from congklak import config,pilot,campaign
from congklak.neural_fast import Model,NeuralNative
from congklak.native import Native,flatten
from stage5_audit_acceptance import literal_position,reference_value,pilot_rows
import reference as literal

WORK=ROOT/"build/stage5-audit"
WORK.mkdir(parents=True,exist_ok=True)


def require(test,message):
    if not test: raise AssertionError(message)


def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def path(name):
    resolved=(ROOT/name).resolve()
    require(resolved.is_relative_to(ROOT),"Artifact path outside workspace")
    return resolved


def check_evaluation():
    plan=load(ROOT/"build/stage5-evaluation-plan.json")
    results=load(ROOT/"build/stage5-evaluation-results.json")
    require(results["plan"]==plan,"Training selection plan differs")
    require(plan["input_counts"]==[15,43] and plan["widths"]==[32,64] and plan["seeds"]==[2718,31415],"Experiment cohort changed")
    require(plan["steps"]==8000 and plan["batch_size"]==512 and plan["checkpoint_every"]==100,"Matched training settings changed")
    require(plan["runtime_gate"] is None and not plan["sealed_final_test_consumed"],"Undeclared gate or sealed consumption")
    for name,key in (("tools/stage5-evaluation.py","source_sha256"),("congklak/training_v3.py","trainer_sha256"),("congklak/features_v3.py","features_sha256")):
        require(sha(ROOT/name)==plan[key],"Frozen evaluation source changed: "+name)
    require(len(results["rows"])==8,"Missing matched runs")
    rows={};selected={};cadences={};export_checks={};dev=list(pilot_rows("development"))
    for row in results["rows"]:
        modelpath=path(row["model"]);result=load(modelpath.parent/"result.json");identity=result["identity"]
        key=(row["input_count"],row["width"],row["seed"]);require(key not in rows,"Duplicate experiment row");rows[key]=row
        require(key in {(f,w,s) for f in (15,43) for w in (32,64) for s in (2718,31415)},"Unplanned model")
        require(result["status"]=="complete" and result["steps_completed"]==8000 and result["draws"]==4096000,"Incomplete run")
        require(identity["source_sha256"]==plan["trainer_sha256"] and identity["feature_source_sha256"]==plan["features_sha256"],"Training source identity differs")
        require(identity["input_count"]==key[0] and identity["width"]==key[1] and identity["seed"]==key[2],"Wrong model owner")
        require(identity["sealed_final_test_consumed"] is False and identity["oracle_training"] is False,"Ancestry/sealed contract differs")
        require(identity["dataset_semantic_sha256"]==pilot.load(campaign.DEFAULT)["semantic_sha256"],"Dataset changed")
        for module,digest in identity["loader_dependencies_sha256"].items():
            require(sha(ROOT/(module.replace(".","/")+".py"))==digest,"Loader dependency changed")
        with closing(sqlite3.connect((modelpath.parent/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            histories=[]
            for step,body,digest in db.execute("SELECT step,body,hash FROM history ORDER BY step"):
                metrics=json.loads(body);require(config.digest(metrics)==digest,"Corrupt training history")
                histories.append((step,metrics["finite_choice_horizon"]["uniform_stratum_mae_seeds"]))
            require([step for step,score in histories]==list(range(100,8001,100)),"Unequal scheduled model selection opportunities")
            best_step,best_score=min(histories,key=lambda pair:pair[1])
            require(best_step==result["best_step"] and best_step==row["best_step"],"Export is not fixed-cadence macro optimum")
            meta=db.execute("SELECT body,hash FROM checkpoints WHERE step=? ORDER BY id DESC LIMIT 1",(best_step,)).fetchone()
            body=json.loads(meta[0]);require(config.digest(body)==meta[1],"Corrupt selected checkpoint metadata")
            require(body["weights_sha256"]==result["weights_sha256"]==row["weights_sha256"],"Selected weights differ")
            cadences[str(key)]={"checkpoints":len(histories),"best_step":best_step,"best_macro_mae":best_score}
        model=Model(modelpath);require(model.id==row["weights_sha256"],"Export digest differs")
        require(model.envelope["model"]["step"]==row["best_step"],"Export step differs")
        params=sum(len(value) if not isinstance(value[0],list) else sum(map(len,value)) for value in model.weights.values())
        require(params==row["parameters"]==result["parameters"],"Parameter count differs")
    paired=[]
    for f in (15,43):
        for w in (32,64):
            group=[rows[(f,w,s)] for s in (2718,31415)]
            paired.append({"input_count":f,"width":w,"mean_macro_mae":statistics.mean(r["macro_mae"] for r in group)})
        width=min((r for r in paired if r["input_count"]==f),key=lambda r:r["mean_macro_mae"])["width"]
        candidate=min((r for key,r in rows.items() if key[:2]==(f,width)),key=lambda r:r["macro_mae"])
        require(candidate==results["selected"][str(f)],"Selected representative violates predeclared paired-width criterion")
        model=Model(path(candidate["model"]));buckets=defaultdict(list);max_native_error=0.0;max_metric_error=0.0
        with NeuralNative(model,backend="avx2") as n:
            for row in dev:
                raw,units=reference_value(literal_position(row["state"]),model.envelope["model"])
                native=n.evaluate(row["state"]);require(native["value_units"]==units,"Trained feature export quantization differs")
                max_native_error=max(max_native_error,abs(native["value_player0"]-raw))
                buckets[row["stratum"]].append(abs(raw-row["label"]["target_player0"]))
        require(max_native_error<=1e-10,"Native/independent trained model parity exceeded")
        metric=candidate["metrics"]
        for name,error in buckets.items():
            value=statistics.mean(error);max_metric_error=max(max_metric_error,abs(value-metric["strata"][name]["mae_seeds"]))
        require(max_metric_error<=1e-4,"Reported model metrics differ from independent development targets")
        ordinary={name:statistics.mean(error) for name,error in buckets.items() if name.endswith("/ordinary") and len(error)>=20}
        gate=all(value<=metric["strata"][name]["H_v1_mae_seeds"] for name,value in ordinary.items())
        require(gate==candidate["ordinary_gate_passed"],"Ordinary promotion gate differs")
        selected[str(f)]={k:candidate[k] for k in ("input_count","width","seed","model","weights_sha256","macro_mae","parameters","ordinary_gate_passed")}
        export_checks[str(f)]={"development_rows":len(dev),"max_native_reference_error_seeds":max_native_error,"max_reported_metric_error_seeds":max_metric_error,"quantized_units_exact":True,"ordinary_mae_seeds":ordinary}
    return {"predeclared_plan_sha256":sha(ROOT/"build/stage5-evaluation-plan.json"),"selected":selected,"selection_opportunities":cadences,"export_checks":export_checks,"sealed_labels_read":False}


def check_benchmark():
    file=ROOT/"build/stage5-benchmark.json"
    require(file.exists(),"Final equal-time benchmark is not yet complete")
    result=load(file);plan=load(ROOT/"build/stage5-benchmark-plan.json");identity=result["identity"]
    require(result.get("sealed_final_test_consumed") is False,"Benchmark consumed sealed labels")
    require(result.get("energy_measured") is False,"Unverified energy assertion")
    require(identity["source_sha256"]==sha(ROOT/"tools/benchmark-stage5.py"),"Benchmark source changed")
    require(identity["plan_sha256"]==config.digest(plan),"Frozen root plan differs")
    require(identity["selection_sha256"]==sha(ROOT/"build/stage5-evaluation-results.json"),"Selection changed after benchmark")
    require(plan["budgets_ms"]==[25,100] and plan["max_depth"]==16 and plan["tt_mb"]==16 and plan["use_pvs"] is True,"Unequal or undeclared search budgets")
    require(plan["arena_ms"]==20 and plan["arena_max_choices"]==512,"Arena protocol changed")
    require(len(plan["cases"])==407 and len(plan["arena_starts"])==24,"Benchmark size changed")
    require(len({r["family"] for r in plan["arena_starts"]})==24,"Arena families overlap")
    require(Counter(r["stratum"].split("/")[-1] for r in plan["arena_starts"])==Counter(ordinary=8,capture=8,bonus=8),"Arena mechanism coverage changed")
    engines=["H_v1","old256_scalar","old256_fast","control15_fast","features43_fast"]
    require(plan["engines"]==engines,"Challenger cohort changed")
    for name,digest in identity["bindings"].items():require(sha(ROOT/name)==digest,"Benchmark binding changed")
    selection=load(ROOT/"build/stage5-evaluation-results.json")
    modelpaths={"old256_scalar":ROOT/"data/training-stronger2-w256-seed31415/model.json","old256_fast":ROOT/"data/training-stronger2-w256-seed31415/model.json",
                "control15_fast":path(selection["selected"]["15"]["model"]),"features43_fast":path(selection["selected"]["43"]["model"])}
    require(identity["baseline_dll_sha256"]==sha(ROOT/"build/congklak_native.dll"),"Baseline DLL differs")
    for name,modelpath in modelpaths.items():
        model=Model(modelpath);saved=identity["models"][name]
        require(saved["weights_sha256"]==model.id and saved["model_file_sha256"]==sha(modelpath),"Arena model differs")
        require(saved["dll_sha256"]==sha(ROOT/("build/congklak_neural.dll" if name=="old256_scalar" else "build/congklak_neural_fast.dll")),"Arena neural DLL differs")
    require(identity["models"]["old256_fast"]["weights_sha256"]==identity["models"]["old256_scalar"]["weights_sha256"],"Speed-only control changed weights")
    datasets={name:pilot.load(folder) for name,folder in (("original",pilot.DEFAULT_PATH),("current",campaign.DEFAULT))}
    protected={}
    for name,manifest in datasets.items():
        require(identity["datasets"][name]==manifest["semantic_sha256"],"Benchmark dataset differs")
        folder=pilot.DEFAULT_PATH if name=="original" else campaign.DEFAULT
        with closing(sqlite3.connect((folder/manifest["directory"]/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            protected[name]=[(pilot.quotient(pilot.from_key(bytes(key))),split,family) for key,split,family in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples")]
        family_ownership=defaultdict(set)
        for q,split,family in protected[name]:family_ownership[family].add(split)
        require(all(len(owner)==1 for owner in family_ownership.values()),"Family assigned to multiple splits")
    currentdev={r["state_key"]:r for r in pilot_rows("development")};ordinary={k for k,r in currentdev.items() if r["stratum"].endswith("/ordinary")}
    currentcases={r["state_key"] for r in plan["cases"] if r["cohort"]=="ordinary_current_development"}
    require(currentcases==ordinary and len(currentcases)==259,"Current ordinary cohort has been filtered after scores")
    cohorts=Counter((r["cohort"],r["reference"]["kind"]) for r in plan["cases"])
    require(cohorts==Counter({("original_development","finite_choice_horizon"):100,("original_development","exact_optimal_margin"):48,("ordinary_current_development","finite_choice_horizon"):259}),"Reference cohorts pooled or changed")
    legacy_exact_aliases=[]
    for source in plan["cases"]:
        q=pilot.quotient(source["state"])
        relevant="original" if source["cohort"]=="original_development" else "current"
        ownership={split for other,split,family in protected[relevant] if q==other}
        require("train" not in ownership,"Root aliases a training state")
        require(all(not any(q==other and split=="train" for other,split,family in protected[name]) for name in protected),"Root aliases another generation's training state")
        if "sealed_final_test" in ownership:
            require(relevant=="original" and source["reference"]["kind"]=="exact_optimal_margin" and sum(map(sum,source["state"]["pits"]))<=4,"Finite/current development root aliases sealed holdout")
            legacy_exact_aliases.append({"state_key":source["state_key"],"quotient_key":q.hex(),"stratum":source["stratum"],"ownership":sorted(ownership)})
        if relevant=="current":require(source["reference"]==currentdev[source["state_key"]]["label"],"Current H6 target differs")
    require(len(legacy_exact_aliases)==16,"Legacy exact diagnostic alias count changed")
    (WORK/"legacy-exact-aliases.json").write_text(json.dumps({"metadata_only":True,"sealed_target_bodies_read":False,"count":16,"cases":legacy_exact_aliases,"interpretation":"Old K4 development roots share quotient aliases with old sealed history states. They remain known exact diagnostics, not an independent final-test cohort. Corrected corpus/train/dev/sealed quotient isolation is separately checked."},indent=2)+"\n",encoding="utf-8")
    ledger=ROOT/"data/stage5-benchmark/work.sqlite3"
    with closing(sqlite3.connect(ledger.as_uri()+"?mode=ro",uri=True)) as db:
        meta=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        require(config.digest(json.loads(meta[0]))==meta[1] and json.loads(meta[0])==identity,"Benchmark catalog identity differs")
        cases=[]
        for idx,body,digest in db.execute("SELECT idx,body,hash FROM cases ORDER BY idx"):
            c=json.loads(body);require(idx==len(cases) and config.digest(c)==digest,"Noncontiguous/corrupt case ledger");cases.append(c)
        require(len(cases)==407,"Not all root experiments committed")
        games=[]
        for comparison,pair,seat,body,digest in db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat"):
            game=json.loads(body);require(config.digest(game)==digest,"Corrupt arena ledger")
            require(game["comparison"]==comparison and game["pair"]==pair and game["candidate_seat"]==seat,"Arena catalog key differs");games.append(game)
        completion=db.execute("SELECT body,hash FROM meta WHERE key='stage5_complete'").fetchone()
        require(completion is not None and config.digest(json.loads(completion[0]))==completion[1],"Missing final catalog commit")
        require(json.loads(completion[0])==dict(cases=407,games=72,report_sha256=config.digest(result)),"Output is not final committed report")
    groups=defaultdict(list);strata=defaultdict(list);equaldepth=0
    for idx,case in enumerate(cases):
        source=plan["cases"][idx]
        for key in ("state_key","cohort","stratum","reference"):require(case[key]==source[key],"Saved case metadata differs")
        reference=source["reference"];q={r["pit"]:r["value_player0"] for r in reference["action_scores"]};sign=1 if source["state"]["controller"]==0 else -1
        for budgetindex,ms in enumerate((25,100)):
            rotate=idx%5;order=engines[rotate:]+engines[:rotate]
            if budgetindex:order.reverse()
            require(case["orders"][str(ms)]==order,"Engine timing order differs")
            values=case["measurements"][str(ms)];require(set(values)==set(engines),"Missing measurement")
            for name,value in values.items():
                chosen=value["chosen"];report=value["report"]
                require(chosen in q or chosen is None,"Illegal root choice")
                require(value["reference_optimal"]==(chosen in reference["best_moves"] if chosen is not None else None),"Corrupt agreement metric")
                require(value["reference_regret"]==(sign*(reference["target_player0"]-q[chosen]) if chosen is not None else None),"Corrupt regret metric")
                require((report is not None)==(chosen is not None),"Missing recommendation/result relation")
                if report:require(report["best_moves"][0]==chosen and 1<=report["depth"]<=16,"Saved report/ranking differs")
                groups[(case["cohort"],str(ms),reference["kind"],name)].append(value)
                strata[(case["cohort"],str(ms),case["stratum"],name)].append(value)
            a=values["old256_scalar"]["report"];b=values["old256_fast"]["report"]
            if a and b and a["depth"]==b["depth"]:
                require(a["value_player0"]==b["value_player0"] and a["best_moves"]==b["best_moves"],"Same weights differ at same completed horizon");equaldepth+=1
    def aggregate(values):
        completed=[v for v in values if v["chosen"] is not None]
        return dict(positions=len(values),no_completed_recommendation=len(values)-len(completed),reference_optimal_accuracy=statistics.mean(v["reference_optimal"] for v in completed) if completed else None,
                    mean_reference_regret_seeds=statistics.mean(v["reference_regret"] for v in completed) if completed else None,mean_completed_depth=statistics.mean(v["report"]["depth"] for v in completed) if completed else None,
                    median_wall_ms=statistics.median(v["wall_ms"] for v in values),max_wall_ms=max(v["wall_ms"] for v in values))
    require(result["budget_results"]=={"/".join(key):aggregate(values) for key,values in groups.items()},"Budget summary differs")
    require(result["strata"]=={"/".join(key):aggregate(values) for key,values in strata.items()},"Stratum summary differs")
    require(len(games)==72,"Not all swapped-seat games committed")
    replays=0;arenas={};margin_secondary={}
    for comparison in plan["arena_comparisons"]:
        group=[g for g in games if g["comparison"]==comparison["id"]]
        require({(g["pair"],g["candidate_seat"]) for g in group}=={(pair,seat) for pair in range(comparison["pairs"]) for seat in (0,1)},"Missing/duplicated arena seats")
        for game in group:
            source=plan["arena_starts"][game["pair"]]
            require(game["start"]==source["state"] and game["family"]==source["family"],"Arena start differs")
            p=literal_position(game["start"])
            require(len(game["moves"])<=512,"Arena choice cap exceeded")
            for action in game["moves"]:
                require(not literal.normalize(p).terminal,"Moves after terminal")
                p=literal.apply(p,action).position
            final={"pits":[list(p.counts[:7]),list(p.counts[8:15])],"stores":[p.counts[7],p.counts[15]],"controller":p.controller}
            require(final==game["final"],"Independent literal arena replay differs")
            terminal=literal.normalize(p).terminal;margin=(p.counts[7]-p.counts[15])*(1 if game["candidate_seat"]==0 else -1)
            score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
            require(game["status"]==("terminal" if terminal else "unfinished") and game["candidate_match_score"]==score and game["candidate_final_margin"]==(margin if terminal else None),"Independent arena outcome differs")
            replays+=1
        paired=[statistics.mean(g["candidate_match_score"] for g in group if g["pair"]==pair) for pair in range(comparison["pairs"]) if all(g["status"]=="terminal" for g in group if g["pair"]==pair)]
        rng=random.Random(20261015);bootstrap=sorted(statistics.mean(rng.choices(paired,k=len(paired))) for _ in range(4096)) if paired else []
        arenas[comparison["id"]]=dict(pairs=comparison["pairs"],terminal_games=sum(g["status"]=="terminal" for g in group),unfinished_games=sum(g["status"]!="terminal" for g in group),complete_pairs=len(paired),candidate_wins=sum(g["candidate_match_score"]==1 for g in group),candidate_draws=sum(g["candidate_match_score"]==.5 for g in group),candidate_losses=sum(g["candidate_match_score"]==0 for g in group),paired_mean_score=statistics.mean(paired) if paired else None,paired_bootstrap_95_interval=[bootstrap[102],bootstrap[3993]] if bootstrap else None,fallback_counts=dict(Counter({name:sum(g["fallbacks"].get(name,0) for g in group) for name in (comparison["candidate"],comparison["opponent"])})))
        bounds=Counter();family_margins=[]
        for pair in range(comparison["pairs"]):
            source=plan["arena_starts"][pair];position=source["state"]
            active=sum(map(sum,position["pits"]));offset=position["stores"][0]-position["stores"][1]
            category="fixed_winner" if abs(offset)>active else "win_or_tie_boundary" if abs(offset)==active else "contested_wdl"
            bounds[category]+=1
            two=sorted((g for g in group if g["pair"]==pair),key=lambda g:g["candidate_seat"])
            margins=[g["candidate_final_margin"] for g in two]
            advantage=statistics.mean(margins) if all(g["status"]=="terminal" for g in two) else None
            family_margins.append(dict(pair=pair,family=source["family"],state_key=source["state_key"],active_seeds=active,initial_store_margin_player0=offset,wdl_scope=category,candidate_final_margin_by_seat=margins,paired_candidate_mean_final_margin_seeds=advantage))
        values=[v["paired_candidate_mean_final_margin_seeds"] for v in family_margins if v["paired_candidate_mean_final_margin_seeds"] is not None]
        rng=random.Random(20261015);interval=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4096)) if values else []
        margin_secondary[comparison["id"]]=dict(scope="Descriptive secondary metric added after the frozen protocol: seat-swapped initial store offset cancels. No late model/start selection or promotion gate uses this metric.",start_wdl_bounds=dict(bounds),complete_pairs=len(values),mean_paired_candidate_final_margin_seeds=statistics.mean(values) if values else None,paired_bootstrap_95_interval_seeds=[interval[102],interval[3993]] if interval else None,families=family_margins)
    require(result["arenas"]==arenas,"Arena summaries or family bootstrap differs")
    require("held" in result["promotion"],"Development-only experiment promoted automatically")
    (WORK/"arena-margin-secondary.json").write_text(json.dumps(margin_secondary,indent=2)+"\n",encoding="utf-8")
    return {"report_sha256":sha(file),"cases":len(cases),"cohorts":[[list(key),count] for key,count in cohorts.items()],"legacy_exact_diagnostic_sealed_quotient_aliases":len(legacy_exact_aliases),"legacy_exact_is_independent_final_test":False,"current_train_dev_sealed_aliases":0,"sameweights_equaldepth_comparisons":equaldepth,"literal_games_replayed":replays,"arenas":arenas,"arena_descriptive_margin_secondary":margin_secondary,"references":"H6 heuristic proxies and verified K4 exact diagnostics kept separate","sealed_labels_read":False,"energy_measured":False,"promotion":"held"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-tests",action="store_true");parser.add_argument("--without-benchmark",action="store_true");args=parser.parse_args()
    output={"schema":"stage5-independent-audit-v1","recorded_at_utc":datetime.now(timezone.utc).isoformat(),"passed":False,"scope":"independent implementation correctness and experiment integrity; strength is a separate gate"}
    try:
        if args.run_tests:
            completed=subprocess.run([sys.executable,str(ROOT/"tests/stage5_audit_acceptance.py")],cwd=ROOT,text=True,capture_output=True)
            (WORK/"acceptance.log").write_text(completed.stdout+completed.stderr,encoding="utf-8")
            require(completed.returncode==0,"Independent acceptance failed")
        tests=load(WORK/"tests.json");require(tests["passed"] and tests["tests_run"]==9 and tests["skips"]==0,"Independent tests incomplete")
        output["independent_acceptance"]=tests
        output["evaluation"]=check_evaluation()
        if not args.without_benchmark:output["benchmark"]=check_benchmark()
        output["benchmark_audit_complete"]=not args.without_benchmark
        output["passed"]=True
    except Exception as exc:
        output["error"]={"type":type(exc).__name__,"message":str(exc)}
    inputs=("tests/stage5_audit_acceptance.py","tools/stage5-audit.py","native/neural_fast.cpp","congklak/neural_fast.py","congklak/features_v3.py","congklak/training_v3.py","tools/stage5-evaluation.py","build/congklak_neural_fast.dll","build/stage5-evaluation-plan.json","build/stage5-evaluation-results.json","tools/benchmark-stage5.py","build/stage5-benchmark-plan.json","build/stage5-benchmark.json")
    output["inputs"]={name:sha(ROOT/name) for name in inputs if (ROOT/name).exists()}
    (WORK/"audit.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"stage5_independent_audit_complete","passed":output["passed"],"benchmark_audit_complete":output.get("benchmark_audit_complete",False),"error":output.get("error"),"report":"build/stage5-audit/audit.json"}),flush=True)
    return 0 if output["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
