"""Durable complete-call native/CUDA batching sweep; features and sync are timed."""
from __future__ import annotations
import argparse
from contextlib import ExitStack,closing
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import unittest
import uuid
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot,phase6_store
from congklak.neural import development_records
from congklak.neural_relay import sha
from congklak.phase6_inference import BatchEvaluator,VALUE_TOLERANCE_SEEDS,LOGIT_TOLERANCE
from congklak.timing_v2 import PerformanceCore
CPU={"id":258,"group":0,"logical":2,"core":2,"cache":0,"numa":0,"efficiency_class":1}
SIZES=(1,2,4,8,16,32)

def event(name,**fields):print(config.canonical({"event":name,**fields}),flush=True)
def atomic(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    serialized=config.canonical(body)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")==serialized:return
        raise ValueError("Immutable Phase6 output differs; select a new output path")
    try:
        with temp.open("x",encoding="utf-8") as stream:stream.write(serialized);stream.flush();os.fsync(stream.fileno())
        for attempt in range(12):
            try:os.replace(temp,path);break
            except PermissionError:
                if attempt==11:raise
                time.sleep(.025*(attempt+1))
    finally:
        if temp.exists():temp.unlink()

def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
def summary(rows):return {"median":statistics.median(rows),"min":min(rows),"max":max(rows),"samples":len(rows)}

def acceptance(args):
    os.environ["PHASE6_PUCT_MODEL"]=str(Path(args.model).resolve())
    if args.gpu_tests:os.environ["PHASE6_GPU_ACCEPTANCE"]="1"
    tests=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="phase6_puct_acceptance.py")
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    paths=("congklak/puct.py","congklak/phase6_inference.py","tests/phase6_puct_acceptance.py","tools/phase6-inference-speed.py","congklak/phase6_store.py","congklak/neural_relay.py","congklak/features_relay.py","build/congklak_neural_relay.dll","build/congklak_native.dll","congklak/timing_v2.py","tests/reference.py")
    body={"schema":"phase6-puct-acceptance-v1","passed":result.wasSuccessful(),"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"gpu_test_scheduled":args.gpu_tests,
          "tested_model":str(Path(args.model).resolve().relative_to(ROOT)),"tested_model_sha256":sha(args.model),
          "inputs":{p:sha(ROOT/p) for p in paths},"checks":[{"name":name,"passed":result.wasSuccessful()} for name in ("literal_fixed_player0_actor_bonus_pass_and_seed_conservation","independent8tree_forest_and_noise_rng","per_game_rng_serialized_midgame_recovery","parent_bootstrap_fpu_store_offset_invariance","legal_terminal_and_resource_bounds","deadline_overshoot_and_actual_native_cancel","native_exact_terminal_work_and_repeat")]+([{"name":"cuda_all_batchshape_parity_and_repeat","passed":result.wasSuccessful()}] if args.gpu_tests else [])}
    # Acceptance evidence is versioned by caller; the CPU preflight output is
    # intentionally mutable until this new experiment's sources are frozen.
    output=Path(args.output);temp=output.with_name(output.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temp.open("x",encoding="utf-8") as stream:stream.write(config.canonical(body)+"\n");stream.flush();os.fsync(stream.fileno())
        os.replace(temp,output)
    finally:
        if temp.exists():temp.unlink()
    event("phase6_puct_acceptance",ok=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),artifact=str(Path(args.output).resolve()))
    return 0 if result.wasSuccessful() else 1

def run(args):
    rows=list(development_records(ROOT/"data/stronger-10000-v2"));rows.sort(key=lambda r:hashlib.sha256(("phase6-inference-v1/"+r["state_key"]).encode()).digest());roots=[r["state"] for r in rows[:64]]
    with PerformanceCore(CPU) as cpu,ExitStack() as stack:
        names=("native","cuda") if args.backend=="both" else (args.backend,)
        engines={name:stack.enter_context(BatchEvaluator(args.model,name,max_batch=32)) for name in names}
        identity={"schema":"phase6-complete-inference-sweep-v1","engines":{key:value.identity for key,value in engines.items()},"cpu":cpu.metadata,
                  "source_sha256":sha(__file__),"store_sha256":sha(ROOT/"congklak/phase6_store.py"),"timing_sha256":sha(ROOT/"congklak/timing_v2.py"),"roots":config.digest(roots),"corpus_sha256":pilot.load(ROOT/"data/stronger-10000-v2")["semantic_sha256"],
                  "batch_sizes":list(SIZES),"blocks":args.blocks,"repetitions":args.repetitions,"warmup_calls_per_unit":4,
                  "clock":"complete synchronous call/features/queue/transfer/forward/sync/validation","value_tolerance_seeds":VALUE_TOLERANCE_SEEDS,"logit_tolerance":LOGIT_TOLERANCE,
                  "native_counter_policy":"actual separate frozen feature-counter pass plus frozen forward feature pass; both charged"}
        ledger=ROOT/"data/phase6-inference-speed"/config.digest(identity)[:16]
        with closing(phase6_store.Work(ledger,identity)) as work:
            db=work.db;db.execute("CREATE TABLE IF NOT EXISTS units(kind TEXT,backend TEXT,batch INTEGER,block INTEGER,body TEXT,hash TEXT,PRIMARY KEY(kind,backend,batch,block))");db.commit()
            def check(kind,backend,batch,block,body):
                pilot._need(backend in names and type(body) is dict,"Invalid inference backend/unit")
                if kind=="startup":pilot._need(batch==0 and block==0 and set(body)=={"startup_ms"} and positive(body["startup_ms"]),"Invalid startup unit")
                elif kind=="parity":
                    pilot._need(batch in SIZES and block==0 and set(body)=={"roots","max_value_error_seeds","max_logit_error","reference","passed"},"Invalid parity unit")
                    pilot._need(body["roots"]==64 and body["passed"] is True and body["reference"]=="frozen-native-double-head","Invalid parity scope")
                    pilot._need(type(body["max_value_error_seeds"]) in (int,float) and 0<=body["max_value_error_seeds"]<=VALUE_TOLERANCE_SEEDS and type(body["max_logit_error"]) in (int,float) and 0<=body["max_logit_error"]<=LOGIT_TOLERANCE,"Saved parity failed")
                elif kind=="micro":
                    pilot._need(batch in SIZES and 0<=block<args.blocks and set(body)=={"whole_call_ns","position_ns","feature_simulations","feature_deposits","timing_mean_ms","repetitions","checksum"},"Invalid micro unit")
                    pilot._need(positive(body["whole_call_ns"]) and body["position_ns"]==body["whole_call_ns"]/batch and body["repetitions"]==args.repetitions,"Invalid micro time or repetitions")
                    for key in ("feature_simulations","feature_deposits"):pilot._need(type(body[key]) is int and body[key]>=0,"Invalid feature counters")
                    pilot._need(math.isfinite(body["checksum"]),"Invalid inference checksum")
                    pilot._need(set(body["timing_mean_ms"])=={"cpu_features_ms","queue_wait_ms","host_to_device_ms","gpu_ms","device_to_host_ms","native_forward_ms","whole_call_ms"},"Invalid timing decomposition")
                    pilot._need(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in body["timing_mean_ms"].values()),"Invalid component time")
                else:raise pilot.PilotError("Unknown inference-speed unit")
            for row in db.execute("SELECT * FROM units"):
                body=json.loads(row["body"]);pilot._need(config.digest(body)==row["hash"],"Corrupt inference-speed ledger");check(row["kind"],row["backend"],row["batch"],row["block"],body)
            def get(kind,name,batch,block=0):
                row=db.execute("SELECT body FROM units WHERE kind=? AND backend=? AND batch=? AND block=?",(kind,name,batch,block)).fetchone();return json.loads(row[0]) if row else None
            def put(kind,name,batch,block,body):
                check(kind,name,batch,block,body);work.admit(1024*1024)
                with db:db.execute("INSERT INTO units VALUES(?,?,?,?,?,?)",(kind,name,batch,block,config.canonical(body),config.digest(body)))
                event("phase6_inference_unit_committed",kind=kind,backend=name,batch=batch,block=block,completed=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],total=len(names)*(1+len(SIZES)*(1+args.blocks)))
            event("phase6_inference_started",ledger=str(ledger.relative_to(ROOT)),completed=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],total=len(names)*(1+len(SIZES)*(1+args.blocks)))
            for name,e in engines.items():
                if get("startup",name,0) is None:put("startup",name,0,0,{"startup_ms":e.startup_ms})
            for batch in SIZES:
                for name,e in engines.items():
                    if get("parity",name,batch) is not None:continue
                    value_error=logit_error=0.0
                    for offset in range(0,64,batch):
                        cpu.verify();states=roots[offset:offset+batch];actual=e.evaluate_many(states)
                        for p,row in zip(states,actual):
                            expected=e.native.evaluate(p);value_error=max(value_error,abs(row["value_player0"]-expected["model_value_player0"]));logit_error=max(logit_error,max(abs(x-y) for x,y in zip(row["policy_logits"],expected["policy_logits"])))
                    cpu.verify();pilot._need(value_error<=VALUE_TOLERANCE_SEEDS and logit_error<=LOGIT_TOLERANCE,"Inference parity tolerance exceeded")
                    put("parity",name,batch,0,{"roots":64,"max_value_error_seeds":value_error,"max_logit_error":logit_error,"reference":"frozen-native-double-head","passed":True})
            for block in range(args.blocks):
                order=list(SIZES);rotation=block%len(order);order=order[rotation:]+order[:rotation]
                for batch in order:
                    backend_order=list(names)
                    if block%2:backend_order.reverse()
                    for name in backend_order:
                        if get("micro",name,batch,block) is not None:continue
                        e=engines[name];cpu.verify()
                        for _ in range(4):e.evaluate_many(roots[:batch])
                        times=[];timing={key:0.0 for key in ("cpu_features_ms","queue_wait_ms","host_to_device_ms","gpu_ms","device_to_host_ms","native_forward_ms","whole_call_ms")};simulations=deposits=0;checksum=0.0
                        for index in range(args.repetitions):
                            states=[roots[(block*7+index+j)%64] for j in range(batch)];started=time.perf_counter_ns();values=e.evaluate_many(states);times.append(time.perf_counter_ns()-started)
                            for key in timing:timing[key]+=e.last_metrics[key]
                            simulations+=sum(row["feature_simulations"] for row in values);deposits+=sum(row["feature_deposits"] for row in values)
                            checksum+=sum(row["value_player0"]+sum(row["policy_logits"])/7 for row in values)
                        cpu.verify();mean=sum(times)/len(times)
                        put("micro",name,batch,block,{"whole_call_ns":mean,"position_ns":mean/batch,"feature_simulations":simulations,"feature_deposits":deposits,"timing_mean_ms":{key:value/args.repetitions for key,value in timing.items()},"repetitions":args.repetitions,"checksum":checksum})
            output={"schema":"phase6-inference-speed-results-v1","complete":True,"identity":identity,"ledger":str(ledger.relative_to(ROOT)),"backends":{},
                    "scope":"Complete synchronous inference, no tree or strength claim. Features, queue wait, transfers and synchronization are charged. Native does two actual feature passes for counter provenance; CUDA does one. Startup is separate from four-call warmed units. No power saving claim."}
            for name in names:
                output["backends"][name]={"startup":get("startup",name,0),"batches":{}}
                for batch in SIZES:
                    units=[get("micro",name,batch,block) for block in range(args.blocks)]
                    output["backends"][name]["batches"][str(batch)]={"parity":get("parity",name,batch),"whole_call_ns":summary([u["whole_call_ns"] for u in units]),"position_ns":summary([u["position_ns"] for u in units]),"positions_per_second":summary([1e9/u["position_ns"] for u in units]),"timing_mean_ms":{key:summary([u["timing_mean_ms"][key] for u in units]) for key in units[0]["timing_mean_ms"]}}
            if len(names)==2:output["native_to_cuda_whole_call_ratio"]={str(batch):summary([get("micro","native",batch,block)["whole_call_ns"]/get("micro","cuda",batch,block)["whole_call_ns"] for block in range(args.blocks)]) for batch in SIZES}
            complete=work.get("phase6_inference_complete")
            if complete:pilot._need(complete=={"units":db.execute("SELECT COUNT(*) FROM units").fetchone()[0],"report_sha256":config.digest(output)},"Completed inference report differs from ledger")
            atomic(args.output,output)
            if complete is None:
                with db:work.put("phase6_inference_complete",{"units":db.execute("SELECT COUNT(*) FROM units").fetchone()[0],"report_sha256":config.digest(output)})
            event("phase6_inference_done",ok=True,units=db.execute("SELECT COUNT(*) FROM units").fetchone()[0],artifact=str(Path(args.output).resolve()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",default=str(ROOT/"data/training-stage5-deeper-h8-f127-seed2718/model.json"));p.add_argument("--backend",choices=("native","cuda","both"),default="both")
    p.add_argument("--blocks",type=int,default=7);p.add_argument("--repetitions",type=int,default=128);p.add_argument("--output",default=str(ROOT/"build/phase6-inference-speed-results.json"));p.add_argument("--acceptance-only",action="store_true");p.add_argument("--gpu-tests",action="store_true")
    args=p.parse_args();config.integer(args.blocks,"blocks",1,20);config.integer(args.repetitions,"repetitions",1,10000)
    try:return acceptance(args) if args.acceptance_only else (run(args) or 0)
    except KeyboardInterrupt:event("phase6_inference_interrupted",ok=False);return 130
    except Exception as exc:event("phase6_inference_error",ok=False,error=str(exc));return 2
if __name__=="__main__":raise SystemExit(main())
