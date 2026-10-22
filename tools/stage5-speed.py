"""Resumable uncached inference and fresh-cache equal-depth measurements."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack,closing
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot
from congklak.native import flatten
from congklak.neural import NeuralNative as FrozenNative,development_records,sha
from congklak.neural_fast import Model,NeuralNative,validate_report

def event(kind,**body):print(config.canonical({"event":kind,**body}),flush=True)
def atomic(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(config.canonical(body)+"\n",encoding="utf-8");temporary.replace(path)
def summary(values):
    return {"median":statistics.median(values),"min":min(values),"max":max(values),"samples":len(values)}
def compact(report):
    return {key:report[key] for key in ("choice_depth","value_units","best_moves","pv","nodes","deposits","nn_evaluations","nn_cache_hits","tt_hits","pvs_probes","pvs_researches")}

def run(args):
    models=[ROOT/"data/training-stronger2-w64-seed2718/model.json",ROOT/"data/training-stronger2-w256-seed31415/model.json"]
    models.extend(Path(p).resolve() for p in args.model)
    models=list(dict.fromkeys(models))
    rows=list(development_records(ROOT/"data/stronger-10000-v2"));rows.sort(key=lambda r:hashlib.sha256(("stage5-speed-v1/"+r["state_key"]).encode()).digest())
    roots=[r["state"] for r in rows[:64]]
    strata=defaultdict(list)
    for row in rows:strata[row["stratum"]].append(row)
    choices=[]
    for stratum,items in sorted(strata.items()):
        if stratum.endswith("/ordinary"):choices.extend(items[:3])
    selected={r["state_key"]:r for r in choices+rows[:args.cases]}
    cases=sorted(selected.values(),key=lambda r:hashlib.sha256(("stage5-depth-v1/"+r["state_key"]).encode()).digest())[:args.cases]
    with ExitStack() as stack:
        engines={}
        for path in models:
            model=Model(path);items={"packed_scalar":stack.enter_context(NeuralNative(model,backend="scalar")),"packed_auto":stack.enter_context(NeuralNative(model))}
            if model.input_count==15:items["frozen"]=stack.enter_context(FrozenNative(path))
            engines[str(path.relative_to(ROOT))]=(model,items)
        identity={"schema":"stage5-speed-v1","source_sha256":sha(__file__),"binding_sha256":sha(ROOT/"congklak/neural_fast.py"),
                  "frozen_binding_sha256":sha(ROOT/"congklak/neural.py"),"native_fast_sha256":next(iter(engines.values()))[1]["packed_auto"].build_id,
                  "native_frozen_sha256":sha(ROOT/"build/congklak_neural.dll"),"dataset_semantic_sha256":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"],
                  "models":{name:{"weights_sha256":model.id,"file_sha256":model.file_sha256,"inputs":model.input_count,"width":model.width} for name,(model,_) in engines.items()},
                  "micro_roots":config.digest(roots),"depth_cases":config.digest([r["state_key"] for r in cases]),"depth":args.depth,"repetitions":args.repetitions,"micro_blocks":7,"micro_evaluations":8192}
        path=ROOT/"data/stage5-speed"/config.digest(identity)[:16]
        with closing(pilot.Work(path,identity)) as work:
            db=work.db;db.execute("CREATE TABLE IF NOT EXISTS units(kind TEXT,model TEXT,unit INTEGER,body TEXT,hash TEXT,PRIMARY KEY(kind,model,unit))");db.commit()
            for row in db.execute("SELECT body,hash FROM units"):
                pilot._need(config.digest(json.loads(row[0]))==row[1],"Corrupt speed benchmark ledger")
            def get(kind,model,unit):
                row=db.execute("SELECT body FROM units WHERE kind=? AND model=? AND unit=?",(kind,model,unit)).fetchone()
                return json.loads(row[0]) if row else None
            def put(kind,model,unit,body):
                work.admit(1024*1024)
                with db:db.execute("INSERT INTO units VALUES(?,?,?,?,?)",(kind,model,unit,config.canonical(body),config.digest(body)))
            total=len(engines)*(1+7+len(cases)*args.repetitions)
            event("stage5_speed_started",ledger=str(path.relative_to(ROOT)),completed=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],total=total)
            for name,(model,items) in engines.items():
                if not get("parity",name,0):
                    maximum=0
                    for row in rows:
                        p=row["state"];a=items["packed_auto"].evaluate(p);b=items["packed_scalar"].evaluate(p)
                        pilot._need(a==b,"AVX/scalar parity failed")
                        if "frozen" in items:pilot._need(a==items["frozen"].evaluate(p),"Frozen evaluator parity failed")
                        else:
                            reference=model.reference(p);maximum=max(maximum,abs(reference[0]-a["value_player0"]))
                            pilot._need(a["value_units"]==reference[1] and maximum<=1e-10,"Feature reference parity failed")
                    put("parity",name,0,{"roots":len(rows),"raw_bitwise_scalar_auto":True,"units_exact":True,"max_feature_reference_abs":maximum,"backend":items["packed_auto"].backend})
                    event("stage5_speed_parity_committed",model=name,roots=len(rows))
                prepared=[((C.c_int*16)(*flatten(p)),p["controller"]) for p in roots]
                for block in range(7):
                    if get("micro",name,block):continue
                    measured={};order=list(items);order=order[block%len(order):]+order[:block%len(order)]
                    for key in order:
                        n=items[key]
                        for arr,actor in prepared:n.lib.cc_nn_evaluate(arr,actor,n.handle)
                        before=time.perf_counter_ns()
                        for _ in range(128):
                            for arr,actor in prepared:n.lib.cc_nn_evaluate(arr,actor,n.handle)
                        api_ns=(time.perf_counter_ns()-before)/8192
                        kernel=n.measure_inference(roots,128) if key!="frozen" else None
                        measured[key]={"api_ns_per_evaluation":api_ns,"single_forward_ns":None if kernel is None else kernel["elapsed_ns"]/8192,"checksum":None if kernel is None else kernel["checksum"]}
                    if "packed_scalar" in measured:pilot._need(measured["packed_scalar"]["checksum"]==measured["packed_auto"]["checksum"],"Kernel benchmark numeric parity failed")
                    put("micro",name,block,measured);event("stage5_speed_micro_committed",model=name,block=block+1,total=7)
                for idx,row in enumerate(cases):
                    for repetition in range(args.repetitions):
                        unit=idx*args.repetitions+repetition
                        if get("depth",name,unit):continue
                        measured={};expected=None;order=list(items);rotation=(idx+repetition)%len(order);order=order[rotation:]+order[:rotation]
                        for key in order:
                            n=items[key];before=time.perf_counter_ns()
                            result=n.neural_analyze(row["state"],max_depth=args.depth,time_ms=0,tt_mb=16,use_pvs=True)
                            wall=(time.perf_counter_ns()-before)/1e6;report=result["report"]
                            pilot._need(result["stop_reason"]=="max_depth" and report["choice_depth"]==args.depth,"Incomplete equal-depth case")
                            if key!="frozen":validate_report(report,row["state"],n)
                            value=compact(report)
                            if expected is None:expected=value
                            else:pilot._need(value==expected,"Fresh-cache equal-depth search semantics differ")
                            measured[key]={"wall_ms":wall,"native_elapsed_ms":result["elapsed_ms"],"report":value}
                        put("depth",name,unit,{"state_key":row["state_key"],"stratum":row["stratum"],"repetition":repetition,"measurements":measured})
                        event("stage5_speed_depth_committed",model=name,completed=unit+1,total=len(cases)*args.repetitions)
            output={"schema":"stage5-speed-results-v1","identity":identity,"ledger":str(path.relative_to(ROOT)),"complete":True,"models":{},
                    "interpretation":"No inference cache in direct evaluation or single-forward loop. Equal-depth searches allocate a fresh TT/evaluation cache each invocation. API timing includes C boundary and JSON; frozen API computes the residual twice, fast API once. Single-forward compares packed scalar with packed AVX only; do not call API timing a pure kernel speedup. Wall ratios include search, simulation and output costs. No strength or energy conclusion follows from these measurements."}
            for name,(model,items) in engines.items():
                micro=[get("micro",name,i) for i in range(7)];depth=[get("depth",name,i) for i in range(len(cases)*args.repetitions)]
                api={k:summary([r[k]["api_ns_per_evaluation"] for r in micro]) for k in items}
                kernel={k:summary([r[k]["single_forward_ns"] for r in micro]) for k in items if k!="frozen"}
                speedups={k:summary([r["measurements"][k]["wall_ms"]/r["measurements"]["packed_auto"]["wall_ms"] for r in depth]) for k in items if k!="packed_auto"}
                output["models"][name]={"parity":get("parity",name,0),"api_ns":api,"single_forward_ns":kernel,"fresh_cache_equal_depth_wall_speedup":speedups,
                                        "cases":len(cases),"depth":args.depth,"repetitions":args.repetitions,"all_search_scores_moves_pvs_and_counters_equal":True}
            atomic(ROOT/"build/stage5-speed-results.json",output)
            event("stage5_speed_done",ok=True,models=len(engines),units=total,artifact="build/stage5-speed-results.json")

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--model",action="append",default=[])
    parser.add_argument("--depth",type=int,default=6);parser.add_argument("--cases",type=int,default=12);parser.add_argument("--repetitions",type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.depth<=16 or not 1<=args.cases<=100 or not 1<=args.repetitions<=10:parser.error("Depth1..16,cases1..100,repetitions1..10")
    try:run(args);return 0
    except KeyboardInterrupt:event("stage5_speed_interrupted",ok=False);return 130
    except Exception as exc:event("stage5_speed_error",ok=False,error=str(exc));return 2
if __name__=="__main__":raise SystemExit(main())
