"""Predeclared root-action ablations, pinned equal-time engines and seed-margin arenas."""
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
from congklak.timing_v2 import PerformanceCore

PLAN=ROOT/"build/stage5-action-benchmark-plan-v2.json"
DEFAULT_RUN=ROOT/"data/stage5-action-benchmark-v2"
DEFAULT_OUTPUT=ROOT/"build/stage5-action-benchmark-v2.json"


def emit(message):
    print(config.canonical(message),flush=True)


def require(value,message):
    if not value: raise ValueError(message)


def frozen_roots():
    previous=json.loads((ROOT/"build/stage5-benchmark-plan.json").read_text())
    exclusions=json.loads((ROOT/"build/stage5-audit/legacy-exact-aliases.json").read_text())
    require(exclusions["metadata_only"] is True and exclusions["sealed_target_bodies_read"] is False,
            "Exclusion provenance differs")
    keys={r["state_key"] for r in exclusions["cases"]}
    require(len(keys)==16 and len(previous["cases"])==407,"Frozen prior coverage differs")
    cases=[r for r in previous["cases"] if r["state_key"] not in keys]
    require(len(cases)==391 and len({r["state_key"] for r in cases})==391,"Current coverage differs")
    starts=previous["arena_starts"]
    require(len(starts)==24 and len({r["family"] for r in starts})==24,"Arena families overlap")
    require(all(abs(r["state"]["stores"][0]-r["state"]["stores"][1])<sum(map(sum,r["state"]["pits"])) for r in starts),
            "Arena start already uncontested")
    return cases,starts


def make_plan():
    cases,starts=frozen_roots()
    plan=dict(schema="stage5-action-benchmark-plan-v2",cases=cases,arena_starts=starts,
              budgets_ms=[25,100],max_depth=16,tt_mb=16,use_pvs=True,arena_ms=20,arena_max_choices=512,
              engines=["H_v1","h_none","h_policy","control_value","joint_value","joint_policy"],
              arena_comparisons=[dict(id="joint_policy_vs_H",candidate="joint_policy",opponent="H_v1",pairs=24),
                                 dict(id="h_policy_vs_h_none",candidate="h_policy",opponent="h_none",pairs=24),
                                 dict(id="joint_policy_vs_joint_value",candidate="joint_policy",opponent="joint_value",pairs=24)],
              order="rotate by case index; reverse rotations for second budget; fresh caches each call; no timed-search warmup",
              cpu_protocol="own-process/declared-os-performance-core/v2",
              cpu_declared={"id":258,"group":0,"logical":2,"core":2,"cache":0,"numa":0,"efficiency_class":1},
              revision="Same frozen cases/models/budgets/gates; declare exact prior CPU to eliminate parking-based resume drift; v1 historical results retained",
              policy_min_depth=2,
              primary_arena="Paired mean final seed margin over the 24 families, 4096 family bootstrap resamples, seed20261015; 95% lower bound >0 required",
              secondary_arena="Paired WDL score is descriptive; no arena-guided model selection",
              reference_scope="Separate 100 original H6 proxy, 32 known exact diagnostics and 259 current ordinary H6 development proxy roots; 16 legacy sealed aliases excluded by metadata",
              promotion="Learned leaves require ordinary MAE gate plus primary margin gate; ordering-only requires its isolated primary margin gate and exact fixed-horizon H equivalence; no automatic promotion",
              sealed_final_test_consumed=False)
    if PLAN.exists(): require(json.loads(PLAN.read_text())==plan,"Frozen benchmark plan changed")
    else: atomic_json(PLAN,plan)
    emit(dict(event="action_benchmark_plan",cases=391,ordinary_cases=259,arena_games=144,plan_sha256=config.digest(plan)))
    return plan


def read_plan():
    plan=json.loads(PLAN.read_text())
    require(plan["schema"]=="stage5-action-benchmark-plan-v2","Unknown benchmark plan")
    cases,starts=frozen_roots()
    require(plan["cases"]==cases and plan["arena_starts"]==starts,"Frozen cohorts differ")
    require(plan["engines"]==["H_v1","h_none","h_policy","control_value","joint_value","joint_policy"],"Engine order differs")
    require(plan["budgets_ms"]==[25,100] and plan["policy_min_depth"]==2,"Budgets/policy threshold differ")
    require(plan["cpu_declared"]==dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1),"Declared CPU differs")
    require(plan["max_depth"]==16 and plan["tt_mb"]==16 and plan["use_pvs"] is True
            and plan["arena_ms"]==20 and plan["arena_max_choices"]==512,"Search/arena settings differ")
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
                    nn_evaluations=report.get("nn_evaluations",0),nn_cache_hits=report.get("nn_cache_hits",0),
                    policy_ordered_nodes=report.get("policy_ordered_nodes",0))
    return dict(report=report,stop_reason=result["stop_reason"],nodes=result["nodes"],
                wall_ms=elapsed,native_elapsed_ms=result["elapsed_ms"])


def barrier(event):
    emit(event)
    requested=os.environ.get("STAGE5_ACTION_BLOCK_AFTER_CASE")
    if event["event"]=="action_case_committed" and requested and int(requested)==event["completed"]:
        emit(dict(event="action_case_commit_barrier",completed=event["completed"]))
        # Test-only barrier after full transaction commit. The auditor kills
        # this process; no partial result has been admitted.
        while True: time.sleep(.1)


def validate_case(row,index,plan):
    require(type(index) is int and 0<=index<len(plan["cases"]),"Unknown case index")
    source=plan["cases"][index]
    require(row["cpu_verified"]==dict(group=plan["cpu_declared"]["group"],logical=plan["cpu_declared"]["logical"],mask=1<<plan["cpu_declared"]["logical"]),"Case CPU verification differs")
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
            require(value["stop_reason"] in ("max_depth","time_budget"),"Interrupted/resource-failed case cannot enter timed strength results")
            report=value["report"]; chosen=value["chosen"]
            if report:
                require(integer(report["depth"],0,plan["max_depth"]) and numeric(report["value_player0"],-98,98),"Invalid saved report depth/value")
                best=report["best_moves"]
                require(isinstance(best,list) and best==sorted(set(best)) and all(integer(a,0,6) and a in q for a in best),"Invalid saved best moves")
                require(integer(report["nn_evaluations"]) and integer(report["nn_cache_hits"]),"Invalid saved evaluator counters")
                require(integer(report["policy_ordered_nodes"]),"Invalid saved policy counters")
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
    require(type(pair) is int and 0<=pair<comparison["pairs"] and type(seat) is int and seat in (0,1),"Unknown game identity")
    row=plan["arena_starts"][pair]
    require(game["cpu_verified"]==dict(group=plan["cpu_declared"]["group"],logical=plan["cpu_declared"]["logical"],mask=1<<plan["cpu_declared"]["logical"]),"Game CPU verification differs")
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
        paired=[]; margins=[]
        for pair in range(comparison["pairs"]):
            group=[g for g in games if g["pair"]==pair]
            if len(group)==2 and all(g["status"]=="terminal" for g in group):
                paired.append(statistics.mean(g["candidate_match_score"] for g in group))
                margins.append(statistics.mean(g["candidate_final_margin"] for g in group))
        def interval(values):
            rng=random.Random(20261015)
            samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4096)) if values else []
            return [samples[102],samples[3993]] if samples else None
        margin_ci=interval(margins)
        fallbacks={name:sum(g["fallbacks"].get(name,0) for g in games) for name in (comparison["candidate"],comparison["opponent"])}
        full=len(margins)==comparison["pairs"]
        gate=full and not sum(fallbacks.values()) and margin_ci[0]>0
        arenas[comparison["id"]]=dict(pairs=comparison["pairs"],terminal_games=sum(g["status"]=="terminal" for g in games),
             unfinished_games=sum(g["status"]!="terminal" for g in games),complete_pairs=len(paired),
             candidate_wins=sum(g["candidate_match_score"]==1 for g in games),candidate_draws=sum(g["candidate_match_score"]==.5 for g in games),
             candidate_losses=sum(g["candidate_match_score"]==0 for g in games),paired_mean_score=statistics.mean(paired) if paired else None,
             paired_bootstrap_95_interval=interval(paired),
             primary_paired_mean_final_margin_seeds=statistics.mean(margins) if margins else None,
             primary_margin_bootstrap_95_interval=margin_ci,primary_strength_gate_passed=gate,
             fallback_counts=fallbacks)
    return dict(identity=identity,cases=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],
                budget_results={"/".join(k):aggregate(v) for k,v in buckets.items()},
                strata={"/".join(k):aggregate(v) for k,v in strata.items()},arenas=arenas,
                sealed_final_test_consumed=False,energy_measured=False,promotion="held pending independent audit; gates are development evidence, not game solution",
                scope=plan["reference_scope"],deadline_scope="Native search deadline; wall measurements additionally include Python/native setup and report transfer.")


def run(path=DEFAULT_RUN,output=DEFAULT_OUTPUT,case_limit=None,no_arena=False,on_event=barrier):
    from congklak.neural_action import NeuralNative as ActionNative
    plan=read_plan(); selection_path=ROOT/"build/stage5-action-training-results.json"
    selection=json.loads(selection_path.read_text())
    control=ROOT/selection["selected_control"]["model"]
    joint=ROOT/selection["selected_challenger"]["model"]
    with ExitStack() as stack:
        cpu=stack.enter_context(PerformanceCore(preferred=plan["cpu_declared"]))
        require(cpu.metadata["protocol"]==plan["cpu_protocol"],"Timing protocol differs from frozen plan")
        models={"H_v1":(Native(),False)}
        for name,model,mode in (("h_none",joint,"h-none"),("h_policy",joint,"h-policy"),
                                ("control_value",control,"value-none"),("joint_value",joint,"value-none"),
                                ("joint_policy",joint,"value-policy")):
            models[name]=(stack.enter_context(ActionNative(model,mode=mode,policy_min_depth=2)),True)
        for name,selected in (("control_value",selection["selected_control"]),("joint_policy",selection["selected_challenger"])):
            require(models[name][0].model.id==selected["weights_sha256"],"Selected export weights differ")
        identity=dict(schema="stage5-action-benchmark-v2",source_sha256=sha(__file__),plan_sha256=config.digest(plan),
                      selection_sha256=sha(selection_path),cpu=cpu.metadata,
                      bindings={name:sha(ROOT/name) for name in ("congklak/neural_action.py","congklak/features_v3.py","congklak/timing_v2.py","congklak/progress.py","congklak/native.py")},
                      models={name:dict(dll_sha256=n.build_id,weights_sha256=n.model.id,model_file_sha256=n.model.file_sha256,
                                       backend=n.backend,mode=n.mode,policy_min_depth=n.policy_min_depth)
                              for name,(n,is_nn) in models.items() if is_nn},
                      baseline_dll_sha256=models["H_v1"][0].build_id,rules_digest=config.RULES_DIGEST,
                      datasets={"original":pilot.load()["semantic_sha256"],"current":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"]},
                      cohort_sources={name:sha(ROOT/name) for name in ("build/stage5-benchmark-plan.json","build/stage5-audit/legacy-exact-aliases.json")})
        work=stack.enter_context(_work(path,identity)); db=work.db
        db.executescript("CREATE TABLE IF NOT EXISTS timing_invocations(id INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT,PRIMARY KEY(comparison,pair,seat));")
        for expected,(index,body,digest) in enumerate(db.execute("SELECT idx,body,hash FROM cases ORDER BY idx")):
            require(index==expected and config.digest(json.loads(body))==digest,"Corrupt/noncontiguous case ledger")
            validate_case(json.loads(body),index,plan)
        observations=dict(observed_at=time.time(),cpu=cpu.observations,verified=cpu.verify())
        with db: db.execute("INSERT INTO timing_invocations(body,hash) VALUES(?,?)",(config.canonical(observations),config.digest(observations)))
        comparisons={c["id"]:c for c in plan["arena_comparisons"]}
        for comparison,pair,seat,body,digest in db.execute("SELECT comparison,pair,seat,body,hash FROM games"):
            require(comparison in comparisons and config.digest(json.loads(body))==digest,"Unknown/damaged game")
            replay_game(json.loads(body),comparisons[comparison],pair,seat,plan,models["H_v1"][0])
        done=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        on_event(dict(event="action_benchmark_started",completed=done,total=len(plan["cases"])))
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
                    cpu.verify()
                    native,is_nn=models[name]; value=search(native,row["state"],ms,is_nn,plan); report=value["report"]
                    chosen=report["best_moves"][0] if report and report["best_moves"] else None
                    value.update(chosen=chosen,reference_optimal=chosen in row["reference"]["best_moves"] if chosen is not None else None,
                                 reference_regret=sign*(row["reference"]["target_player0"]-q[chosen]) if chosen is not None else None)
                    measured[str(ms)][name]=value
            body=dict(cpu_verified=cpu.verify(),index=index,state_key=row["state_key"],cohort=row["cohort"],stratum=row["stratum"],reference=row["reference"],orders=orders,measurements=measured)
            validate_case(body,index,plan); work.admit(1024*1024)
            with db: db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
            on_event(dict(event="action_case_committed",completed=index+1,total=len(plan["cases"]),cohort=row["cohort"]))
        if no_arena or stop<len(plan["cases"]):
            on_event(dict(event="action_benchmark_partial",completed=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],total=len(plan["cases"])))
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
                        cpu.verify()
                        native,is_nn=models[name]; answer=search(native,position,plan["arena_ms"],is_nn,plan)
                        report=answer["report"]
                        if report and report["best_moves"]: action=report["best_moves"][0]
                        else: action=normal["legal_actions"][0]; fallbacks[name]+=1
                        choices.append(action)
                        position=models["H_v1"][0].apply(flatten(position),actor=position["controller"],action=action,total=98,trace=False)["position"]
                    terminal=not any(map(sum,position["pits"]))
                    margin=(position["stores"][0]-position["stores"][1])*(1 if seat==0 else -1)
                    score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
                    game=dict(cpu_verified=cpu.verify(),comparison=comparison["id"],pair=pair,candidate_seat=seat,start=row["state"],family=row["family"],moves=choices,final=position,
                              status="terminal" if terminal else "unfinished",candidate_final_margin=margin if terminal else None,
                              candidate_match_score=score,fallbacks=dict(fallbacks))
                    replay_game(game,comparison,pair,seat,plan,models["H_v1"][0]); work.admit(1024*1024)
                    with db: db.execute("INSERT INTO games VALUES(?,?,?,?,?)",(comparison["id"],pair,seat,config.canonical(game),config.digest(game)))
                    on_event(dict(event="action_arena_game_committed",comparison=comparison["id"],pair=pair,seat=seat,status=game["status"],candidate_score=score,
                                  completed=db.execute("SELECT COUNT(*) FROM games").fetchone()[0],total=144))
        result=summarize(db,identity,plan); atomic_json(Path(output),result)
        with db:
            work.put("action_complete",dict(cases=391,games=144,report_sha256=config.digest(result)))
        on_event(dict(event="action_benchmark_complete",completed=391,total=391,arenas=result["arenas"]))
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
            require(args.case_limit is None or 0<=args.case_limit<=391,"Invalid case limit")
            run(args.run,args.output,args.case_limit,args.no_arena)
        return 0
    except (ValueError,OSError,KeyboardInterrupt) as exc:
        emit(dict(event="action_benchmark_error",error=str(exc))); return 2


if __name__=="__main__": raise SystemExit(main())
