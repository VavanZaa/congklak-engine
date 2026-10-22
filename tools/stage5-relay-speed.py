"""Durable pinned-core relay-feature extraction and inference measurements.

Every fresh forward includes the actual rule simulations. Training's precomputed
feature catalog is never read by the runtime or this measurement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack,closing
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot,features_relay
from congklak.native import Native,flatten
from congklak.neural import development_records,sha
from congklak.neural_relay import Model,NeuralNative,validate_report
from congklak.timing_v2 import PerformanceCore

CPU={"id":258,"group":0,"logical":2,"core":2,"cache":0,"numa":0,"efficiency_class":1}
def event(name,**fields):print(config.canonical({"event":name,**fields}),flush=True)
def atomic(path,body):
    path=Path(path);temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temp.open("x",encoding="utf-8") as f:
            f.write(config.canonical(body)+"\n");f.flush();os.fsync(f.fileno())
        for attempt in range(12):
            try:os.replace(temp,path);break
            except PermissionError:
                if attempt==11:raise
                time.sleep(.025*(attempt+1))
    finally:
        if temp.exists():temp.unlink()
def summary(values):return {"median":statistics.median(values),"min":min(values),"max":max(values),"samples":len(values)}
def semantic(report,legacy=False):
    return {"value_units":report["depth_result"]["value_player0"]*256 if legacy else report["value_units"],"best_moves":report["best_moves"],"pv":report["pv"],
            "action_scores":[[c["pit"],c["depth_result"]["value_player0"]*256 if legacy else c["value_units"]] for c in report["candidates"]]}
def nonnegative(x):return type(x) in (int,float) and math.isfinite(x) and x>=0

def run(args):
    rows=list(development_records(ROOT/"data/stronger-10000-v2"));rows.sort(key=lambda r:hashlib.sha256(("relay-speed-v1/"+r["state_key"]).encode()).digest())
    micro=[r["state"] for r in rows[:64]];strata=defaultdict(list)
    for row in rows:strata[row["stratum"]].append(row)
    ordinary=[]
    for name,items in sorted(strata.items()):
        if name.endswith("/ordinary"):ordinary.extend(items[:3])
    chosen={r["state_key"]:r for r in ordinary+rows[:args.cases]}
    cases=sorted(chosen.values(),key=lambda r:hashlib.sha256(("relay-depth-v1/"+r["state_key"]).encode()).digest())[:args.cases]
    with PerformanceCore(CPU) as cpu,ExitStack() as stack:
        baseline=Native();models={}
        for file in dict.fromkeys(args.model):
            path=Path(file).resolve();model=Model(path)
            modes={mode:stack.enter_context(NeuralNative(model,mode=mode)) for mode in ("h-none","value-none")}
            scalar=stack.enter_context(NeuralNative(model,mode="value-none",backend="scalar"));models[str(path.relative_to(ROOT))]=(model,modes,scalar)
        identity={"schema":"stage5-relay-speed-v1","source_sha256":sha(__file__),"binding_sha256":sha(ROOT/"congklak/neural_relay.py"),"features_sha256":sha(features_relay.__file__),
                  "base_features_sha256":sha(features_relay.features_v3.__file__),"timing_sha256":sha(ROOT/"congklak/timing_v2.py"),
                  "native_h_sha256":baseline.build_id,"native_relay_sha256":next(iter(models.values()))[1]["h-none"].build_id,"cpu":cpu.metadata,
                  "models":{name:{"weights_sha256":m.id,"file_sha256":m.file_sha256,"width":m.width,"input_count":m.input_count,"encoder":m.encoder} for name,(m,_,_) in models.items()},
                  "corpus_sha256":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"],"roots":config.digest(micro),"cases":config.digest([r["state_key"] for r in cases]),
                  "depth":args.depth,"repetitions":args.repetitions,"micro_blocks":7,"micro_evaluations":8192,"feature_policy":features_relay.FEATURE_POLICY,"policy_ordering":False}
        path=ROOT/"data/stage5-relay-speed"/config.digest(identity)[:16]
        with closing(pilot.Work(path,identity)) as work:
            db=work.db;db.execute("CREATE TABLE IF NOT EXISTS units(kind TEXT,model TEXT,unit INTEGER,body TEXT,hash TEXT,PRIMARY KEY(kind,model,unit))");db.commit()
            def check(kind,name,unit,body):
                pilot._need(name in models,"Unknown relay-speed model")
                if kind=="parity":
                    pilot._need(unit==0 and body["roots"]==len(rows) and body["units_exact"] is True and body["raw_heads_bitwise_scalar_avx"] is True,"Invalid saved parity")
                    pilot._need(0<=body["reference_max_value_abs"]<=1e-10 and 0<=body["reference_max_logit_abs"]<=1e-10,"Saved reference parity failed")
                elif kind=="micro":
                    pilot._need(0<=unit<7 and set(body)=={"scalar","auto"},"Invalid saved micro unit")
                    for row in body.values():pilot._need(set(row)=={"ns_per_complete_forward","checksum"} and nonnegative(row["ns_per_complete_forward"]) and row["ns_per_complete_forward"]>0 and math.isfinite(row["checksum"]),"Invalid saved micro measurement")
                    pilot._need(body["scalar"]["checksum"]==body["auto"]["checksum"],"Saved micro parity failed")
                elif kind=="depth":
                    pilot._need(0<=unit<len(cases)*args.repetitions,"Invalid saved depth index")
                    row=cases[unit//args.repetitions]
                    pilot._need(body["state_key"]==row["state_key"] and body["stratum"]==row["stratum"] and body["repetition"]==unit%args.repetitions,"Saved depth root differs")
                    m=body["measurements"];pilot._need(set(m)=={"H_v1","h-none","value-none","scalar-value"},"Saved engine modes differ")
                    for mode,v in m.items():
                        for field in ("wall_ms","native_ms","nodes","nn_evaluations","nn_cache_hits","feature_simulations","feature_deposits","deposits"):
                            pilot._need(nonnegative(v[field]),"Invalid saved work/time counter")
                        pilot._need(v["feature_deposits"]<=v["deposits"],"Saved feature deposits exceed work")
                        if mode in ("H_v1","h-none") or models[name][0].input_count==43:pilot._need(v["feature_simulations"]==v["feature_deposits"]==0,"Unexpected saved feature work")
                    pilot._need(m["H_v1"]["semantic"]==m["h-none"]["semantic"] and m["value-none"]["semantic"]==m["scalar-value"]["semantic"],"Saved fixed-horizon semantics differ")
                else:raise pilot.PilotError("Unknown relay-speed unit")
            for row in db.execute("SELECT * FROM units"):
                body=json.loads(row["body"]);pilot._need(config.digest(body)==row["hash"],"Corrupt relay-speed ledger");check(row["kind"],row["model"],row["unit"],body)
            def get(kind,model,unit):
                row=db.execute("SELECT body FROM units WHERE kind=? AND model=? AND unit=?",(kind,model,unit)).fetchone();return json.loads(row[0]) if row else None
            def put(kind,model,unit,body):
                check(kind,model,unit,body);work.admit(1024*1024)
                with db:db.execute("INSERT INTO units VALUES(?,?,?,?,?)",(kind,model,unit,config.canonical(body),config.digest(body)))
            total=len(models)*(8+len(cases)*args.repetitions);event("relay_speed_started",ledger=str(path.relative_to(ROOT)),completed=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],total=total)
            for name,(model,modes,scalar) in models.items():
                if not get("parity",name,0):
                    value_error=logit_error=0
                    for row in rows:
                        p=row["state"];a=modes["value-none"].evaluate(p);b=scalar.evaluate(p);reference=model.reference(p)
                        pilot._need(a==b,"Relay scalar/AVX parity failed")
                        value_error=max(value_error,abs(a["model_value_player0"]-reference[0]));logit_error=max(logit_error,max(abs(x-y) for x,y in zip(a["policy_logits"],reference[2])))
                        pilot._need(a["model_value_units"]==reference[1] and max(value_error,logit_error)<=1e-10,"Relay reference parity failed")
                    put("parity",name,0,{"roots":len(rows),"units_exact":True,"raw_heads_bitwise_scalar_avx":True,"reference_max_value_abs":value_error,"reference_max_logit_abs":logit_error})
                    event("relay_speed_parity_committed",model=name,roots=len(rows))
                kernels={"scalar":scalar,"auto":modes["value-none"]}
                for block in range(7):
                    if get("micro",name,block):continue
                    cpu.verify();measured={};order=list(kernels);order=order[block%2:]+order[:block%2]
                    for key in order:
                        n=kernels[key];n.measure_inference(micro,16);result=n.measure_inference(micro,128)
                        measured[key]={"ns_per_complete_forward":result["elapsed_ns"]/8192,"checksum":result["checksum"]}
                    cpu.verify();pilot._need(measured["scalar"]["checksum"]==measured["auto"]["checksum"],"Relay kernel parity failed")
                    put("micro",name,block,measured);event("relay_speed_micro_committed",model=name,completed=block+1,total=7)
                for index,row in enumerate(cases):
                    for repetition in range(args.repetitions):
                        unit=index*args.repetitions+repetition
                        if get("depth",name,unit):continue
                        cpu.verify();measurements={};order=["H_v1","h-none","value-none","scalar-value"];rotation=(index+repetition)%4;order=order[rotation:]+order[:rotation]
                        for mode in order:
                            before=time.perf_counter_ns()
                            if mode=="H_v1":result=baseline.analyze(flatten(row["state"]),actor=row["state"]["controller"],total=98,max_depth=args.depth,time_ms=0,keep_history=False)
                            else:result=(scalar if mode=="scalar-value" else modes[mode]).neural_analyze(row["state"],max_depth=args.depth,time_ms=0)
                            wall=(time.perf_counter_ns()-before)/1e6;report=result["report"];pilot._need(result["stop_reason"]=="max_depth" and report["choice_depth"]==args.depth,"Incomplete relay depth benchmark")
                            if mode!="H_v1":validate_report(report,row["state"],scalar if mode=="scalar-value" else modes[mode])
                            measurements[mode]={"wall_ms":wall,"native_ms":result["elapsed_ms"],"nodes":result["nodes"],"semantic":semantic(report,mode=="H_v1"),
                                                "nn_evaluations":report.get("nn_evaluations",0),"nn_cache_hits":report.get("nn_cache_hits",0),"feature_simulations":report.get("feature_simulations",0),
                                                "feature_deposits":report.get("feature_deposits",0),"deposits":report["deposits"]}
                        cpu.verify();put("depth",name,unit,{"state_key":row["state_key"],"stratum":row["stratum"],"repetition":repetition,"measurements":measurements})
                        event("relay_speed_depth_committed",model=name,completed=unit+1,total=len(cases)*args.repetitions)
            output={"schema":"stage5-relay-speed-results-v1","identity":identity,"ledger":str(path.relative_to(ROOT)),"complete":True,"models":{},
                    "scope":"Declared OS performance core; same verified affinity. Complete uncached forward includes exact hypothetical transitions and both heads. Searches use fresh caches. Timing does not prove equal-time strength or lower power; no policy ordering is enabled."}
            for name,(model,_,_) in models.items():
                micro_rows=[get("micro",name,i) for i in range(7)];depth_rows=[get("depth",name,i)["measurements"] for i in range(len(cases)*args.repetitions)]
                output["models"][name]={"parity":get("parity",name,0),"complete_forward_ns":{key:summary([r[key]["ns_per_complete_forward"] for r in micro_rows]) for key in kernels},
                                        "scalar_to_avx_ratio":summary([r["scalar"]["ns_per_complete_forward"]/r["auto"]["ns_per_complete_forward"] for r in micro_rows]),
                                        "value_scalar_to_avx_search_ratio":summary([r["scalar-value"]["wall_ms"]/r["value-none"]["wall_ms"] for r in depth_rows]),
                                        "value_to_H_search_time_ratio":summary([r["value-none"]["wall_ms"]/r["H_v1"]["wall_ms"] for r in depth_rows]),
                                        "feature_deposit_fraction":summary([r["value-none"]["feature_deposits"]/max(1,r["value-none"]["deposits"]) for r in depth_rows]),
                                        "cases":len(cases),"repetitions":args.repetitions,"all_fixed_horizon_actions_values_pvs_equal_within_evaluator":True}
            atomic(args.output,output);event("relay_speed_done",ok=True,models=len(models),units=total,artifact=str(Path(args.output).resolve().relative_to(ROOT)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",action="append");p.add_argument("--output",default=str(ROOT/"build/stage5-relay-speed-results.json"));p.add_argument("--acceptance-only",action="store_true");p.add_argument("--depth",type=int,default=6);p.add_argument("--cases",type=int,default=12);p.add_argument("--repetitions",type=int,default=3)
    args=p.parse_args()
    if not 1<=args.depth<=16 or not 1<=args.cases<=100 or not 1<=args.repetitions<=10:p.error("Depth1..16,cases1..100,repetitions1..10")
    try:
        if args.acceptance_only:
            import unittest
            sys.path.insert(0,str(ROOT/"tests"))
            suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="stage5_relay_native_acceptance.py")
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            inputs={file:sha(ROOT/file) for file in ("native/engine.hpp","native/rules.cpp","native/search.cpp","native/neural_relay.cpp","congklak/config.py","congklak/native.py","congklak/features_v3.py","congklak/features_relay.py","congklak/neural_relay.py","tools/stage5-relay-speed.py","tools/build-neural-relay.cmd","tools/build-neural-relay.ps1","tests/reference.py","tests/stage5_relay_native_acceptance.py","build/congklak_neural_relay.dll","build/neural-relay-build.json")}
            output={"schema":"stage5-relay-native-acceptance-v1","passed":result.wasSuccessful(),"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"inputs":inputs,
                    "checks":[{"name":name,"passed":result.wasSuccessful()} for name in ("literal_features_scalar_avx_tails_and_invariants","literal_minimax_all_modes_tt_pvs","H_v1_equivalence_and_charged_features","shape_admission_and_cancellation","actual_depth_kill_resume_identity","43_control_and_unclipped_deposits")]}
            atomic(ROOT/"build/stage5-relay-native-acceptance.json",output);event("relay_native_acceptance",ok=result.wasSuccessful(),tests=result.testsRun,artifact="build/stage5-relay-native-acceptance.json")
            return 0 if result.wasSuccessful() else 1
        if not args.model:p.error("--model required for runtime measurements")
        run(args);return 0
    except KeyboardInterrupt:event("relay_speed_interrupted",ok=False);return 130
    except Exception as exc:event("relay_speed_error",ok=False,error=str(exc));return 2
if __name__=="__main__":raise SystemExit(main())
