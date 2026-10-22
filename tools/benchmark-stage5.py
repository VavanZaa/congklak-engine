"""Frozen development cohorts, five equal-time engines and paired arenas."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack,closing
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot
from congklak.native import Native,flatten
from congklak.neural import NeuralNative as ScalarNative,development_records,sha
from congklak.progress import atomic_json

PLAN=ROOT/"build/stage5-benchmark-plan.json"
DEFAULT_RUN=ROOT/"data/stage5-benchmark"
DEFAULT_OUTPUT=ROOT/"build/stage5-benchmark.json"


def emit(message):
    print(config.canonical(message),flush=True)


def require(value,message):
    if not value: raise ValueError(message)


def frozen_roots():
    spec=importlib.util.spec_from_file_location("stage5_frozen_selection",ROOT/"tools/benchmark-neural-v3.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original,_=module.selected(list(development_records()))
    references={}
    with closing(sqlite3.connect((ROOT/"data/neural-benchmark-v3/work.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as db:
        for body,digest in db.execute("SELECT body,hash FROM cases ORDER BY idx"):
            row=json.loads(body); require(config.digest(row)==digest,"Prior benchmark checksum failed")
            references[row["state_key"]]=row["reference"]
    require(len(original)==148 and len(references)==148,"Original root cohort changed")
    current=list(development_records(ROOT/"data/stronger-10000-v2"))
    ordinary=[r for r in current if r["stratum"].endswith("/ordinary")]
    require(len(current)==1000 and len(ordinary)==259,"Ordinary cohort changed")
    cases=[]
    for cohort,rows in (("original_development",original),("ordinary_current_development",ordinary)):
        for row in sorted(rows,key=lambda r:r["state_key"]):
            cases.append(dict(cohort=cohort,state_key=row["state_key"],state=row["state"],family=row["family"],
                              stratum=row["stratum"],reference=references[row["state_key"]] if cohort=="original_development" else row["label"]))
    # Select starts solely by metadata; no new model prediction is consulted.
    eligible=sorted((r for r in current if sum(map(sum,r["state"]["pits"]))>=25),
                    key=lambda r:hashlib.sha256(("stage5-arena-v1/"+r["state_key"]).encode()).digest())
    groups={kind:[] for kind in ("ordinary","capture","bonus")}; used=set()
    for kind in groups:
        for row in eligible:
            if row["family"] not in used and row["stratum"].endswith("/"+kind):
                groups[kind].append(row); used.add(row["family"])
                if len(groups[kind])==8: break
        require(len(groups[kind])==8,"Insufficient distinct arena families for "+kind)
    starts=[groups[kind][i] for i in range(8) for kind in groups]
    starts=[{key:r[key] for key in ("state_key","state","family","stratum")} for r in starts]
    return cases,starts


def make_plan():
    cases,starts=frozen_roots()
    plan=dict(schema="stage5-benchmark-plan-v1",cases=cases,arena_starts=starts,
              budgets_ms=[25,100],max_depth=16,tt_mb=16,use_pvs=True,arena_ms=20,arena_max_choices=512,
              engines=["H_v1","old256_scalar","old256_fast","control15_fast","features43_fast"],
              arena_comparisons=[dict(id="features43_vs_H",candidate="features43_fast",opponent="H_v1",pairs=24),
                                 dict(id="sameweights_fast_vs_scalar",candidate="old256_fast",opponent="old256_scalar",pairs=12)],
              order="rotate engine order by case index; reverse rotations for second budget; fresh caches each call",
              reference_scope="separate original H6 proxy/exact roots and current ordinary H6 development proxy roots",
              promotion="must pass ordinary MAE gate and independent strength/latency gates; no automatic promotion",
              sealed_final_test_consumed=False)
    if PLAN.exists(): require(json.loads(PLAN.read_text())==plan,"Frozen benchmark plan changed")
    else: atomic_json(PLAN,plan)
    emit(dict(event="stage5_benchmark_plan",cases=len(cases),ordinary_cases=259,arena_games=72,plan_sha256=config.digest(plan)))
    return plan


def read_plan():
    plan=json.loads(PLAN.read_text())
    require(plan["schema"]=="stage5-benchmark-plan-v1","Unknown benchmark plan")
    require(len(plan["cases"])==407 and len(plan["arena_starts"])==24,"Plan coverage changed")
    require(len({r["family"] for r in plan["arena_starts"]})==24,"Arena families overlap")
    return plan


def search(native,position,ms,neural,plan):
    started=time.perf_counter()
    if neural:
        result=native.neural_analyze(position,max_depth=plan["max_depth"],time_ms=ms,
                                     tt_mb=plan["tt_mb"],use_pvs=plan["use_pvs"])
    else:
        result=native.analyze(flatten(position),actor=position["controller"],total=98,max_depth=plan["max_depth"],
                              time_ms=ms,tt_mb=plan["tt_mb"],use_pvs=plan["use_pvs"],keep_history=False)
    elapsed=(time.perf_counter()-started)*1000
    report=result["report"]
    if report:
        report=dict(depth=report["choice_depth"],best_moves=report["best_moves"],
                    value_player0=report["value_units"]/256 if neural else report["depth_result"]["value_player0"],
                    nn_evaluations=report.get("nn_evaluations",0),nn_cache_hits=report.get("nn_cache_hits",0))
    return dict(report=report,stop_reason=result["stop_reason"],nodes=result["nodes"],
                wall_ms=elapsed,native_elapsed_ms=result["elapsed_ms"])


def barrier(event):
    emit(event)
    requested=os.environ.get("STAGE5_BLOCK_AFTER_CASE")
    if event["event"]=="stage5_case_committed" and requested and int(requested)==event["completed"]:
        emit(dict(event="stage5_case_commit_barrier",completed=event["completed"]))
        # Test-only barrier after full transaction commit. The auditor kills
        # this process; no partial result has been admitted.
        while True: time.sleep(.1)


def validate_case(row,index,plan):
    source=plan["cases"][index]
    require(row["index"]==index and row["state_key"]==source["state_key"] and row["cohort"]==source["cohort"]
            and row["stratum"]==source["stratum"] and row["reference"]==source["reference"],"Case metadata differs")
    require(set(row["measurements"])=={str(ms) for ms in plan["budgets_ms"]},"Missing budget")
    require(set(row["orders"])==set(row["measurements"]),"Missing execution order")
    q={a["pit"]:a["value_player0"] for a in source["reference"]["action_scores"]}
    sign=1 if source["state"]["controller"]==0 else -1
    def integer(value,lo=0,hi=2**63-1):
        return type(value) is int and lo<=value<=hi
    def numeric(value,lo=0,hi=float("inf")):
        return type(value) in (int,float) and math.isfinite(value) and lo<=value<=hi
    for budget_index,ms in enumerate(plan["budgets_ms"]):
        values=row["measurements"][str(ms)]
        order=list(plan["engines"]); rotation=index%len(order); order=order[rotation:]+order[:rotation]
        if budget_index: order.reverse()
        require(row["orders"][str(ms)]==order,"Saved execution order differs")
        require(set(values)==set(plan["engines"]),"Missing engine measurement")
        for value in values.values():
            require(integer(value["nodes"]) and numeric(value["wall_ms"]) and numeric(value["native_elapsed_ms"]),"Invalid timing/work measurement")
            report=value["report"]; chosen=value["chosen"]
            if report:
                require(integer(report["depth"],0,plan["max_depth"]) and numeric(report["value_player0"],-98,98),"Invalid saved report depth/value")
                best=report["best_moves"]
                require(isinstance(best,list) and best==sorted(set(best)) and all(integer(a,0,6) and a in q for a in best),"Invalid saved best moves")
                require(integer(report["nn_evaluations"]) and integer(report["nn_cache_hits"]),"Invalid saved evaluator counters")
                require(chosen==(best[0] if best else None),"Saved choice differs from report")
            else:
                require(chosen is None,"Choice without completed report")
            if chosen is None:
                require(value["reference_optimal"] is None and value["reference_regret"] is None,"Fabricated incomplete recommendation")
            else:
                require(integer(chosen,0,6) and chosen in q,"Invalid saved choice")
                require(type(value["reference_optimal"]) is bool and value["reference_optimal"]==(chosen in source["reference"]["best_moves"]),"Saved reference agreement differs")
                require(numeric(value["reference_regret"]) and value["reference_regret"]==sign*(source["reference"]["target_player0"]-q[chosen]),"Saved reference regret differs")


def replay_game(game,comparison,pair,seat,plan,native):
    row=plan["arena_starts"][pair]
    require(game["comparison"]==comparison["id"] and game["pair"]==pair and game["candidate_seat"]==seat
            and game["start"]==row["state"] and game["family"]==row["family"],"Arena identity differs")
    require(isinstance(game["moves"],list) and len(game["moves"])<=plan["arena_max_choices"] and all(type(a) is int and 0<=a<=6 for a in game["moves"]),"Invalid arena move count/encoding")
    require(isinstance(game["fallbacks"],dict) and set(game["fallbacks"]).issubset({comparison["candidate"],comparison["opponent"]})
            and all(type(v) is int and 0<=v<=len(game["moves"]) for v in game["fallbacks"].values())
            and sum(game["fallbacks"].values())<=len(game["moves"]),"Invalid fallback counts")
    position=row["state"]
    for action in game["moves"]:
        normal=native.normalize(flatten(position),actor=position["controller"],total=98)
        require(not normal["terminal"] and action in normal["legal_actions"],"Illegal saved arena action")
        position=native.apply(flatten(normal["position"]),actor=normal["position"]["controller"],action=action,total=98,trace=False)["position"]
    require(position==game["final"],"Arena final replay differs")
    terminal=not any(map(sum,position["pits"]))
    margin=(position["stores"][0]-position["stores"][1])*(1 if seat==0 else -1)
    score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
    require(game["status"]==("terminal" if terminal else "unfinished") and game["candidate_match_score"]==score
            and game["candidate_final_margin"]==(margin if terminal else None),"Arena score/margin differs")


def summarize(db,identity,plan):
    buckets=defaultdict(list); strata=defaultdict(list)
    for body,digest in db.execute("SELECT body,hash FROM cases ORDER BY idx"):
        case=json.loads(body); require(config.digest(case)==digest,"Damaged case")
        for ms,values in case["measurements"].items():
            for name,value in values.items():
                buckets[(case["cohort"],ms,case["reference"]["kind"],name)].append(value)
                strata[(case["cohort"],ms,case["stratum"],name)].append(value)
    def aggregate(values):
        done=[v for v in values if v["chosen"] is not None]
        return dict(positions=len(values),no_completed_recommendation=len(values)-len(done),
                    reference_optimal_accuracy=statistics.mean(v["reference_optimal"] for v in done) if done else None,
                    mean_reference_regret_seeds=statistics.mean(v["reference_regret"] for v in done) if done else None,
                    mean_completed_depth=statistics.mean(v["report"]["depth"] for v in done) if done else None,
                    median_wall_ms=statistics.median(v["wall_ms"] for v in values),
                    max_wall_ms=max(v["wall_ms"] for v in values))
    arenas={}
    for comparison in plan["arena_comparisons"]:
        games=[json.loads(body) for body, in db.execute("SELECT body FROM games WHERE comparison=? ORDER BY pair,seat",(comparison["id"],))]
        paired=[]
        for pair in range(comparison["pairs"]):
            group=[g for g in games if g["pair"]==pair]
            if len(group)==2 and all(g["status"]=="terminal" for g in group):
                paired.append(statistics.mean(g["candidate_match_score"] for g in group))
        rng=random.Random(20261015)
        bootstrap=sorted(statistics.mean(rng.choices(paired,k=len(paired))) for _ in range(4096)) if paired else []
        arenas[comparison["id"]]=dict(pairs=comparison["pairs"],terminal_games=sum(g["status"]=="terminal" for g in games),
             unfinished_games=sum(g["status"]!="terminal" for g in games),complete_pairs=len(paired),
             candidate_wins=sum(g["candidate_match_score"]==1 for g in games),candidate_draws=sum(g["candidate_match_score"]==.5 for g in games),
             candidate_losses=sum(g["candidate_match_score"]==0 for g in games),paired_mean_score=statistics.mean(paired) if paired else None,
             paired_bootstrap_95_interval=[bootstrap[102],bootstrap[3993]] if bootstrap else None,
             fallback_counts=dict(Counter({name:sum(g["fallbacks"].get(name,0) for g in games)
                                          for name in (comparison["candidate"],comparison["opponent"])})))
    return dict(identity=identity,cases=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
                budget_results={"/".join(k):aggregate(v) for k,v in buckets.items()},
                strata={"/".join(k):aggregate(v) for k,v in strata.items()},arenas=arenas,
                sealed_final_test_consumed=False,energy_measured=False,promotion="held; independent acceptance and strength gates required",
                scope=plan["reference_scope"],deadline_scope="Native search deadline; wall measurements additionally include Python/native setup and report transfer.")


def run(path=DEFAULT_RUN,output=DEFAULT_OUTPUT,case_limit=None,no_arena=False,on_event=barrier):
    from congklak.neural_fast import NeuralNative as FastNative
    plan=read_plan(); selection=json.loads((ROOT/"build/stage5-evaluation-results.json").read_text())
    old=ROOT/"data/training-stronger2-w256-seed31415/model.json"
    with ExitStack() as stack:
        models={"H_v1":(Native(),False),
                "old256_scalar":(stack.enter_context(ScalarNative(old)),True),
                "old256_fast":(stack.enter_context(FastNative(old)),True),
                "control15_fast":(stack.enter_context(FastNative(ROOT/selection["selected"]["15"]["model"])),True),
                "features43_fast":(stack.enter_context(FastNative(ROOT/selection["selected"]["43"]["model"])),True)}
        for inputs,name in ((15,"control15_fast"),(43,"features43_fast")):
            model=models[name][0].model
            require(model.id==selection["selected"][str(inputs)]["weights_sha256"] and model.input_count==inputs,
                    "Selected export weights/features differ")
        require(models["old256_scalar"][0].model.id==models["old256_fast"][0].model.id,"Speed-only control weights differ")
        identity=dict(schema="stage5-benchmark-v1",source_sha256=sha(__file__),plan_sha256=config.digest(plan),
                      selection_sha256=sha(ROOT/"build/stage5-evaluation-results.json"),
                      bindings={name:sha(ROOT/name) for name in ("congklak/neural.py","congklak/neural_fast.py","congklak/features_v3.py")},
                      models={name:dict(dll_sha256=n.build_id,weights_sha256=n.model.id,model_file_sha256=n.model.file_sha256,
                                       backend=n.backend if hasattr(n,"backend") else "frozen_scalar")
                              for name,(n,is_nn) in models.items() if is_nn},
                      baseline_dll_sha256=models["H_v1"][0].build_id,rules_digest=config.RULES_DIGEST,
                      datasets={"original":pilot.load()["semantic_sha256"],"current":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"]})
        work=stack.enter_context(_work(path,identity)); db=work.db
        db.executescript("CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT,PRIMARY KEY(comparison,pair,seat));")
        for expected,(index,body,digest) in enumerate(db.execute("SELECT idx,body,hash FROM cases ORDER BY idx")):
            require(index==expected and config.digest(json.loads(body))==digest,"Corrupt/noncontiguous case ledger")
            validate_case(json.loads(body),index,plan)
        done=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        on_event(dict(event="stage5_benchmark_started",completed=done,total=len(plan["cases"])))
        stop=min(len(plan["cases"]),case_limit) if case_limit is not None else len(plan["cases"])
        for index in range(done,stop):
            row=plan["cases"][index]; q={a["pit"]:a["value_player0"] for a in row["reference"]["action_scores"]}
            sign=1 if row["state"]["controller"]==0 else -1
            measured={}; orders={}
            for budget_index,ms in enumerate(plan["budgets_ms"]):
                order=list(plan["engines"]); rotation=index%len(order); order=order[rotation:]+order[:rotation]
                if budget_index: order.reverse()
                orders[str(ms)]=order; measured[str(ms)]={}
                for name in order:
                    native,is_nn=models[name]; value=search(native,row["state"],ms,is_nn,plan); report=value["report"]
                    chosen=report["best_moves"][0] if report and report["best_moves"] else None
                    value.update(chosen=chosen,reference_optimal=chosen in row["reference"]["best_moves"] if chosen is not None else None,
                                 reference_regret=sign*(row["reference"]["target_player0"]-q[chosen]) if chosen is not None else None)
                    measured[str(ms)][name]=value
            body=dict(index=index,state_key=row["state_key"],cohort=row["cohort"],stratum=row["stratum"],reference=row["reference"],orders=orders,measurements=measured)
            validate_case(body,index,plan); work.admit(1024*1024)
            with db: db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
            on_event(dict(event="stage5_case_committed",completed=index+1,total=len(plan["cases"]),cohort=row["cohort"]))
        if no_arena or stop<len(plan["cases"]):
            on_event(dict(event="stage5_benchmark_partial",completed=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],total=len(plan["cases"])))
            return dict(status="partial",identity=identity)
        for comparison in plan["arena_comparisons"]:
            for pair in range(comparison["pairs"]):
                row=plan["arena_starts"][pair]
                for seat in (0,1):
                    existing=db.execute("SELECT body,hash FROM games WHERE comparison=? AND pair=? AND seat=?",(comparison["id"],pair,seat)).fetchone()
                    if existing:
                        game=json.loads(existing[0]); require(config.digest(game)==existing[1],"Arena checksum differs")
                        replay_game(game,comparison,pair,seat,plan,models["H_v1"][0]); continue
                    position=row["state"]; choices=[]; fallbacks=Counter()
                    for _ in range(plan["arena_max_choices"]):
                        normal=models["H_v1"][0].normalize(flatten(position),actor=position["controller"],total=98)
                        if normal["terminal"]: break
                        position=normal["position"]
                        name=comparison["candidate"] if position["controller"]==seat else comparison["opponent"]
                        native,is_nn=models[name]; answer=search(native,position,plan["arena_ms"],is_nn,plan)
                        report=answer["report"]
                        if report and report["best_moves"]: action=report["best_moves"][0]
                        else: action=normal["legal_actions"][0]; fallbacks[name]+=1
                        choices.append(action)
                        position=models["H_v1"][0].apply(flatten(position),actor=position["controller"],action=action,total=98,trace=False)["position"]
                    terminal=not any(map(sum,position["pits"]))
                    margin=(position["stores"][0]-position["stores"][1])*(1 if seat==0 else -1)
                    score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
                    game=dict(comparison=comparison["id"],pair=pair,candidate_seat=seat,start=row["state"],family=row["family"],moves=choices,final=position,
                              status="terminal" if terminal else "unfinished",candidate_final_margin=margin if terminal else None,
                              candidate_match_score=score,fallbacks=dict(fallbacks))
                    replay_game(game,comparison,pair,seat,plan,models["H_v1"][0]); work.admit(1024*1024)
                    with db: db.execute("INSERT INTO games VALUES(?,?,?,?,?)",(comparison["id"],pair,seat,config.canonical(game),config.digest(game)))
                    on_event(dict(event="stage5_arena_game_committed",comparison=comparison["id"],pair=pair,seat=seat,status=game["status"],candidate_score=score,
                                  completed=db.execute("SELECT COUNT(*) FROM games").fetchone()[0],total=72))
        result=summarize(db,identity,plan); atomic_json(Path(output),result)
        with db:
            work.put("stage5_complete",dict(cases=407,games=72,report_sha256=config.digest(result)))
        on_event(dict(event="stage5_benchmark_complete",completed=407,total=407,arenas=result["arenas"]))
        return result


class _work:
    def __init__(self,path,identity): self.path,self.identity=path,identity
    def __enter__(self): self.work=pilot.Work(self.path,self.identity); return self.work
    def __exit__(self,*_): self.work.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("plan","run"))
    parser.add_argument("--run",default=str(DEFAULT_RUN)); parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    parser.add_argument("--case-limit",type=int); parser.add_argument("--no-arena",action="store_true")
    args=parser.parse_args()
    try:
        if args.operation=="plan": make_plan()
        else:
            require(args.case_limit is None or 0<=args.case_limit<=407,"Invalid case limit")
            run(args.run,args.output,args.case_limit,args.no_arena)
        return 0
    except (ValueError,OSError,KeyboardInterrupt) as exc:
        emit(dict(event="stage5_benchmark_error",error=str(exc))); return 2


if __name__=="__main__": raise SystemExit(main())
