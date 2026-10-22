"""Resumable development-only equal-budget searches and paired-start arena."""
from __future__ import annotations
from collections import defaultdict
from contextlib import closing
import hashlib
import json
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
from congklak.neural import NeuralNative,development_records,sha

PATH=ROOT/"data/neural-benchmark-v2"

def event(e): print(config.canonical(e),flush=True)
def selected(rows):
    # The old lineage also has an exact store-offset input reduction. Exclude
    # old training quotients from this comparison's development roots.
    m=pilot.load(); root=pilot.DEFAULT_PATH/m["directory"]
    db=sqlite3.connect((root/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)
    try: old_train={pilot.quotient(pilot.from_key(bytes(k))) for k, in db.execute("SELECT state FROM examples WHERE json_extract(body,'$.split')='train'")}
    finally: db.close()
    eligible=[r for r in rows if pilot.quotient(r["state"]) not in old_train]
    ordered=sorted(eligible,key=lambda r:hashlib.sha256(("benchmark-v1/"+r["state_key"]).encode()).digest())
    finite=[r for r in ordered if r["label"]["kind"]=="finite_choice_horizon"]
    rare=[r for r in finite if r["stratum"] in ("25-49/ordinary","50-74/ordinary")]
    small_ordinary=[r for r in finite if r["stratum"]=="5-24/ordinary"][:20]
    chosen={r["state_key"]:r for r in rare+small_ordinary+finite[:80]}
    exact=[r for r in ordered if r["label"]["kind"]=="exact_optimal_margin" and r["label"]["action_scores"]][:48]
    chosen.update({r["state_key"]:r for r in exact})
    return sorted(chosen.values(),key=lambda r:r["state_key"]),len(rows)-len(eligible)

def compact(report,neural):
    if report is None: return None
    return {"depth":report["choice_depth"],"best_moves":report["best_moves"],"value_player0":report["value_units"]/256 if neural else report["depth_result"]["value_player0"],
            "nn_evaluations":report.get("nn_evaluations",0),"nn_cache_hits":report.get("nn_cache_hits",0)}

def search(native,position,ms,neural):
    before=time.perf_counter()
    if neural: result=native.neural_analyze(position,max_depth=16,time_ms=ms)
    else: result=native.analyze(flatten(position),actor=position["controller"],total=98,max_depth=16,time_ms=ms,keep_history=False)
    return {"report":compact(result["report"],neural),"stop_reason":result["stop_reason"],"nodes":result["nodes"],
            "wall_ms":(time.perf_counter()-before)*1000,"native_elapsed_ms":result["elapsed_ms"]}

def run():
    baseline=Native(); nn64=NeuralNative(ROOT/"data/training-stronger-w64-seed2718/model.json"); nn256=NeuralNative(ROOT/"data/training-stronger-w256-seed31415/model.json")
    work=None
    try:
        rows,withheld=selected(list(development_records()))
        models={"H_v1":(baseline,False),"NN64":(nn64,True),"NN256":(nn256,True)}
        identity={"schema":"neural-benchmark-v1","source_sha256":sha(__file__),"binding_sha256":sha(ROOT/"congklak/neural.py"),
                  "native_h_sha256":baseline.build_id,"native_nn_sha256":nn256.build_id,
                  "models":{k:n.model.id for k,(n,b) in models.items() if b},"dataset_sha256":pilot.load()["semantic_sha256"],
                  "cases":config.digest([r["state_key"] for r in rows]),"budgets_ms":[25,100],"arena_ms":20,"arena_pairs":12,"arena_max_choices":512}
        work=pilot.Work(PATH,identity); db=work.db
        db.executescript("CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS games(pair INTEGER,seat INTEGER,body TEXT,hash TEXT,PRIMARY KEY(pair,seat));")
        # A checksum-valid cursor cannot skip a unit: the ledger is contiguous.
        for expected,r in enumerate(db.execute("SELECT idx,body,hash FROM cases ORDER BY idx")):
            pilot._need(expected==r["idx"] and config.digest(json.loads(r["body"]))==r["hash"],"Corrupt benchmark ledger")
        done=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        event({"event":"benchmark_started","cases":len(rows),"completed":done,"ancestral_train_quotients_withheld":withheld})
        for index,row in enumerate(rows):
            if index<done: continue
            p=row["state"]; exact=row["label"]["kind"]=="exact_optimal_margin"
            if exact: reference=row["label"]
            else:
                r=baseline.analyze(flatten(p),actor=p["controller"],total=98,max_depth=6,time_ms=0,keep_history=False)["report"]
                reference={"kind":"finite_choice_horizon","choice_depth":6,"target_player0":r["depth_result"]["value_player0"],"best_moves":r["best_moves"],
                           "action_scores":[{"pit":c["pit"],"value_player0":c["depth_result"]["value_player0"]} for c in r["candidates"]]}
            q={a["pit"]:a["value_player0"] for a in reference["action_scores"]}; sign=1 if p["controller"]==0 else -1
            measured={}
            order=list(models); order=order[index%3:]+order[:index%3]
            for ms in identity["budgets_ms"]:
                measured[str(ms)]={}
                for name in order:
                    n,is_nn=models[name]; value=search(n,p,ms,is_nn); report=value["report"]
                    if report and report["best_moves"]:
                        chosen=report["best_moves"][0]
                        value.update(chosen=chosen,reference_optimal=chosen in reference["best_moves"],reference_regret=sign*(reference["target_player0"]-q[chosen]))
                    else: value.update(chosen=None,reference_optimal=None,reference_regret=None)
                    measured[str(ms)][name]=value
            body={"state_key":row["state_key"],"stratum":row["stratum"],"reference_kind":reference["kind"],"reference":reference,"measurements":measured}
            work.admit(1024*1024)
            with db: db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
            if index%10==0: event({"event":"benchmark_case_committed","completed":index+1,"total":len(rows)})
        starts=[r for r in rows if r["label"]["kind"]=="finite_choice_horizon" and sum(map(sum,r["state"]["pits"]))>=50][:12]
        pilot._need(len(starts)==12,"Insufficient declared arena starts")
        for pair,row in enumerate(starts):
            for seat in (0,1):
                existing=db.execute("SELECT body,hash FROM games WHERE pair=? AND seat=?",(pair,seat)).fetchone()
                if existing:
                    pilot._need(config.digest(json.loads(existing[0]))==existing[1],"Corrupt arena game"); continue
                p=row["state"]; choices=[]; fallbacks=Counter()
                for _ in range(identity["arena_max_choices"]):
                    normal=baseline.normalize(flatten(p),actor=p["controller"],total=98)
                    if normal["terminal"]: break
                    p=normal["position"]; candidate=p["controller"]==seat
                    answer=search(nn256 if candidate else baseline,p,identity["arena_ms"],candidate)
                    report=answer["report"]
                    if report and report["best_moves"]: action=report["best_moves"][0]
                    else: action=normal["legal_actions"][0]; fallbacks["NN256" if candidate else "H_v1"]+=1
                    choices.append(action); p=baseline.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False)["position"]
                terminal=not any(map(sum,p["pits"])); margin=(p["stores"][0]-p["stores"][1])*(1 if seat==0 else -1)
                score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
                body={"pair":pair,"candidate_seat":seat,"start":row["state"],"start_key":row["state_key"],"moves":choices,"final":p,
                      "status":"terminal" if terminal else "unfinished","candidate_final_margin":margin if terminal else None,"candidate_match_score":score,"fallbacks":dict(fallbacks)}
                work.admit(1024*1024)
                with db: db.execute("INSERT INTO games VALUES(?,?,?,?)",(pair,seat,config.canonical(body),config.digest(body)))
                event({"event":"arena_game_committed","pair":pair,"seat":seat,"status":body["status"],"candidate_score":score})
        buckets=defaultdict(list); strata=defaultdict(list)
        for row in db.execute("SELECT body,hash FROM cases"):
            c=json.loads(row[0]); pilot._need(config.digest(c)==row[1],"Case checksum differs")
            for ms,values in c["measurements"].items():
                for name,result in values.items(): buckets[(ms,c["reference_kind"],name)].append(result); strata[(ms,c["stratum"],name)].append(result)
        def summarize(values):
            completed=[v for v in values if v["chosen"] is not None]
            return {"positions":len(values),"no_completed_recommendation":len(values)-len(completed),
                    "reference_optimal_accuracy":statistics.mean(v["reference_optimal"] for v in completed) if completed else None,
                    "mean_reference_regret_seeds":statistics.mean(v["reference_regret"] for v in completed) if completed else None,
                    "mean_completed_depth":statistics.mean(v["report"]["depth"] for v in values if v["report"]) if completed else None,
                    "median_wall_ms":statistics.median(v["wall_ms"] for v in values),"max_wall_ms":max(v["wall_ms"] for v in values)}
        games=[json.loads(row[0]) for row in db.execute("SELECT body FROM games ORDER BY pair,seat")]; paired=[]
        for pair in range(12):
            g=[g for g in games if g["pair"]==pair]
            if len(g)==2 and all(g["status"]=="terminal" for g in g): paired.append(statistics.mean(g["candidate_match_score"] for g in g))
        rng=random.Random(20261014); bootstrap=sorted(statistics.mean(rng.choices(paired,k=len(paired))) for _ in range(4096)) if paired else []
        out={"identity":identity,"cases":len(rows),"ancestral_train_quotients_withheld":withheld,"sealed_final_test_consumed":False,
             "scope":"development; exact references separate from H_v1 depth-six horizon estimates; no oracle neural prediction",
             "budget_results":{"/".join(k):summarize(v) for k,v in buckets.items()},"strata":{"/".join(k):summarize(v) for k,v in strata.items()},
             "arena":{"pairs":12,"terminal_games":sum(g["status"]=="terminal" for g in games),"unfinished_games":sum(g["status"]!="terminal" for g in games),
                      "candidate_wins":sum(g["candidate_match_score"]==1 for g in games),"candidate_draws":sum(g["candidate_match_score"]==.5 for g in games),
                      "candidate_losses":sum(g["candidate_match_score"]==0 for g in games),"complete_pairs":len(paired),
                      "paired_mean_score":statistics.mean(paired) if paired else None,"paired_bootstrap_95_interval":[bootstrap[102],bootstrap[3993]] if bootstrap else None,
                      "fallback_counts":dict(Counter({name:sum(g["fallbacks"].get(name,0) for g in games) for name in ("NN256","H_v1")}))},
             "energy_measured":False,"promotion":"not_promoted"}
        file=ROOT/"build/neural-benchmark-v2.json"; file.write_text(json.dumps(out,indent=2)+"\n")
        event({"event":"benchmark_done","arena":out["arena"],"cases":len(rows)})
    finally:
        if work: work.close()
        nn64.close(); nn256.close()
if __name__=="__main__":
    from collections import Counter
    run()
