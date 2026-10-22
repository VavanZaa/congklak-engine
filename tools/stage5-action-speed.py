"""Pinned-core action-head inference and fixed-horizon ordering ablations."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack,closing
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot
from congklak.native import Native,flatten
from congklak.neural import development_records,sha
from congklak.neural_action import Model,NeuralNative,MODES,validate_report
from congklak.timing import PerformanceCore

def event(name,**fields):print(config.canonical({"event":name,**fields}),flush=True)
def atomic(path,body):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(config.canonical(body)+"\n",encoding="utf-8");temp.replace(path)
def summary(values):return {"median":statistics.median(values),"min":min(values),"max":max(values),"samples":len(values)}
def semantic(report,legacy=False):
    return {"value_units":report["depth_result"]["value_player0"]*256 if legacy else report["value_units"],"best_moves":report["best_moves"],"pv":report["pv"],
            "action_scores":[[c["pit"],c["depth_result"]["value_player0"]*256 if legacy else c["value_units"]] for c in report["candidates"]]}

def run(args):
    rows=list(development_records(ROOT/"data/stronger-10000-v2"));rows.sort(key=lambda r:hashlib.sha256(("action-speed-v1/"+r["state_key"]).encode()).digest())
    micro=[r["state"] for r in rows[:64]];strata=defaultdict(list)
    for row in rows:strata[row["stratum"]].append(row)
    ordinary=[]
    for name,items in sorted(strata.items()):
        if name.endswith("/ordinary"):ordinary.extend(items[:3])
    chosen={r["state_key"]:r for r in ordinary+rows[:args.cases]}
    cases=sorted(chosen.values(),key=lambda r:hashlib.sha256(("action-depth-v1/"+r["state_key"]).encode()).digest())[:args.cases]
    with PerformanceCore() as cpu,ExitStack() as stack:
        baseline=Native();models={}
        for file in dict.fromkeys(args.model):
            path=Path(file).resolve();model=Model(path)
            modes={mode:stack.enter_context(NeuralNative(model,mode=mode)) for mode in MODES}
            scalar=stack.enter_context(NeuralNative(model,mode="value-policy",backend="scalar"));models[str(path.relative_to(ROOT))]=(model,modes,scalar)
        identity={"schema":"stage5-action-speed-v1","source_sha256":sha(__file__),"binding_sha256":sha(ROOT/"congklak/neural_action.py"),"timing_sha256":sha(ROOT/"congklak/timing.py"),
                  "native_h_sha256":baseline.build_id,"native_action_sha256":next(iter(models.values()))[1]["h-none"].build_id,"cpu":cpu.metadata,
                  "models":{name:{"weights_sha256":m.id,"file_sha256":m.file_sha256,"width":m.width,"input_count":43} for name,(m,_,_) in models.items()},
                  "corpus_sha256":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"],"roots":config.digest(micro),"cases":config.digest([r["state_key"] for r in cases]),
                  "depth":args.depth,"repetitions":args.repetitions,"micro_blocks":7,"micro_evaluations":8192,"policy_min_depth":2}
        path=ROOT/"data/stage5-action-speed"/config.digest(identity)[:16]
        with closing(pilot.Work(path,identity)) as work:
            db=work.db;db.execute("CREATE TABLE IF NOT EXISTS units(kind TEXT,model TEXT,unit INTEGER,body TEXT,hash TEXT,PRIMARY KEY(kind,model,unit))");db.commit()
            for body,digest in db.execute("SELECT body,hash FROM units"):pilot._need(config.digest(json.loads(body))==digest,"Corrupt action-speed ledger")
            def get(kind,model,unit):
                row=db.execute("SELECT body FROM units WHERE kind=? AND model=? AND unit=?",(kind,model,unit)).fetchone();return json.loads(row[0]) if row else None
            def put(kind,model,unit,body):
                work.admit(1024*1024)
                with db:db.execute("INSERT INTO units VALUES(?,?,?,?,?)",(kind,model,unit,config.canonical(body),config.digest(body)))
            total=len(models)*(8+len(cases)*args.repetitions);event("action_speed_started",ledger=str(path.relative_to(ROOT)),completed=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],total=total)
            for name,(model,modes,scalar) in models.items():
                if not get("parity",name,0):
                    value_error=logit_error=0
                    for row in rows:
                        p=row["state"];a=modes["value-policy"].evaluate(p);b=scalar.evaluate(p);reference=model.reference(p)
                        pilot._need(a==b,"Action scalar/AVX parity failed")
                        value_error=max(value_error,abs(a["model_value_player0"]-reference[0]));logit_error=max(logit_error,max(abs(x-y) for x,y in zip(a["policy_logits"],reference[2])))
                        pilot._need(a["model_value_units"]==reference[1] and max(value_error,logit_error)<=1e-10,"Action reference parity failed")
                    put("parity",name,0,{"roots":len(rows),"units_exact":True,"raw_heads_bitwise_scalar_avx":True,"reference_max_value_abs":value_error,"reference_max_logit_abs":logit_error})
                    event("action_speed_parity_committed",model=name,roots=len(rows))
                kernels={"scalar":scalar,"auto":modes["value-policy"]}
                for block in range(7):
                    if get("micro",name,block):continue
                    cpu.verify();measured={};order=list(kernels);order=order[block%2:]+order[:block%2]
                    for key in order:
                        n=kernels[key];n.measure_inference(micro,16);result=n.measure_inference(micro,128)
                        measured[key]={"ns_per_joint_forward":result["elapsed_ns"]/8192,"checksum":result["checksum"]}
                    cpu.verify();pilot._need(measured["scalar"]["checksum"]==measured["auto"]["checksum"],"Action kernel parity failed")
                    put("micro",name,block,measured);event("action_speed_micro_committed",model=name,completed=block+1,total=7)
                for index,row in enumerate(cases):
                    for repetition in range(args.repetitions):
                        unit=index*args.repetitions+repetition
                        if get("depth",name,unit):continue
                        cpu.verify();measurements={};order=["H_v1"]+list(MODES);rotation=(index+repetition)%len(order);order=order[rotation:]+order[:rotation]
                        for mode in order:
                            before=time.perf_counter_ns()
                            if mode=="H_v1":
                                result=baseline.analyze(flatten(row["state"]),actor=row["state"]["controller"],total=98,max_depth=args.depth,time_ms=0,keep_history=False)
                            else:result=modes[mode].neural_analyze(row["state"],max_depth=args.depth,time_ms=0)
                            wall=(time.perf_counter_ns()-before)/1e6;report=result["report"];pilot._need(result["stop_reason"]=="max_depth" and report["choice_depth"]==args.depth,"Incomplete action depth benchmark")
                            if mode!="H_v1":validate_report(report,row["state"],modes[mode])
                            measurements[mode]={"wall_ms":wall,"native_ms":result["elapsed_ms"],"nodes":result["nodes"],"semantic":semantic(report,mode=="H_v1"),
                                                "nn_evaluations":report.get("nn_evaluations",0),"nn_cache_hits":report.get("nn_cache_hits",0),"policy_ordered_nodes":report.get("policy_ordered_nodes",0)}
                        pilot._need(measurements["H_v1"]["semantic"]==measurements["h-none"]["semantic"]==measurements["h-policy"]["semantic"],"Ordering changes H_v1 fixed-horizon semantics")
                        pilot._need(measurements["value-none"]["semantic"]==measurements["value-policy"]["semantic"],"Ordering changes learned fixed-horizon semantics")
                        cpu.verify();put("depth",name,unit,{"state_key":row["state_key"],"stratum":row["stratum"],"repetition":repetition,"measurements":measurements})
                        event("action_speed_depth_committed",model=name,completed=unit+1,total=len(cases)*args.repetitions)
            output={"schema":"stage5-action-speed-results-v1","identity":identity,"ledger":str(path.relative_to(ROOT)),"complete":True,"models":{},
                    "scope":"Single OS-reported performance core, same verified affinity for all blocks, unmodified power/priority. Joint forward includes both heads without caching/Python/JSON. Fixed-horizon searches use fresh caches and identical full-root/lexPV results within each evaluator. Timing and node reduction do not prove improved equal-time playing strength or lower power."}
            for name,(model,_,_) in models.items():
                micro_rows=[get("micro",name,i) for i in range(7)];depth_rows=[get("depth",name,i)["measurements"] for i in range(len(cases)*args.repetitions)]
                output["models"][name]={"parity":get("parity",name,0),"joint_forward_ns":{key:summary([r[key]["ns_per_joint_forward"] for r in micro_rows]) for key in kernels},
                                        "scalar_to_avx_ratio":summary([r["scalar"]["ns_per_joint_forward"]/r["auto"]["ns_per_joint_forward"] for r in micro_rows]),
                                        "ordering_wall_speedup":{e:summary([r[e+"-none"]["wall_ms"]/r[e+"-policy"]["wall_ms"] for r in depth_rows]) for e in ("h","value")},
                                        "ordering_node_reduction_fraction":{e:summary([1-r[e+"-policy"]["nodes"]/r[e+"-none"]["nodes"] for r in depth_rows]) for e in ("h","value")},
                                        "cases":len(cases),"repetitions":args.repetitions,"all_fixed_horizon_actions_values_pvs_equal":True}
            atomic(ROOT/"build/stage5-action-speed-results.json",output);event("action_speed_done",ok=True,models=len(models),units=total,artifact="build/stage5-action-speed-results.json")

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",action="append",required=True);p.add_argument("--depth",type=int,default=6);p.add_argument("--cases",type=int,default=12);p.add_argument("--repetitions",type=int,default=3)
    args=p.parse_args()
    if not 1<=args.depth<=16 or not 1<=args.cases<=100 or not 1<=args.repetitions<=10:p.error("Depth1..16,cases1..100,repetitions1..10")
    try:run(args);return 0
    except KeyboardInterrupt:event("action_speed_interrupted",ok=False);return 130
    except Exception as exc:event("action_speed_error",ok=False,error=str(exc));return 2
if __name__=="__main__":raise SystemExit(main())
