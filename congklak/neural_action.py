"""Action-head search ablations with immutable modes and durable completed depths."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import struct
import threading
from . import config
from .native import ROOT,Native,NativeError,Control,flatten

ENCODER="current-controller-pit14-and-active15/store-offset-exact-v1"
FEATURE_ENCODER="current-controller-pit14-active15-occupied14-singleton14/store-offset-exact-v3"
POLICY="terminal-first/action-double-fixed256/full-root/pvs-action-v1"
MODES={"h-none":0,"h-policy":1,"value-none":2,"value-policy":3}
SCALE=256

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value: raise NativeError(message)
def flatten_weights(value):
    if isinstance(value,list):
        for x in value: yield from flatten_weights(x)
    else:
        need(type(value) in (int,float) and math.isfinite(value),"Nonfinite model weight")
        yield struct.unpack("<f",struct.pack("<f",value))[0]
def encoded(position,input_count=15):
    p=config.position(position); active=sum(map(sum,p["pits"])); actor=p["controller"]
    if active and not sum(p["pits"][actor]): actor=1-actor
    counts=p["pits"][actor]+p["pits"][1-actor]
    x=[v/98 for v in counts]+[active/98]
    if input_count==43:x+=[int(v>0) for v in counts]+[int(v==1) for v in counts]
    return x,p["stores"][0]-p["stores"][1],active*(1 if actor==0 else -1)
def rounded(value): return math.floor(value+.5) if value>=0 else math.ceil(value-.5)

def development_records(path=None):
    from . import pilot
    path=Path(path or pilot.DEFAULT_PATH); manifest=pilot.load(path)
    db=sqlite3.connect(((path/manifest["directory"])/"dataset.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)
    try:
        query="SELECT body,hash FROM examples WHERE json_extract(body,'$.split')='development' ORDER BY state"
        for body,digest in db.execute(query):
            row=json.loads(body); need(config.digest(row)==digest,"Corrupt development row"); yield row
    finally: db.close()

class Model:
    def __init__(self,path):
        self.path=Path(path).resolve()
        need(not (self.path.parent/"quarantine.json").exists(),"This model is quarantined; consult its lineage audit")
        self.envelope=json.loads(self.path.read_text(encoding="utf-8"))
        need(isinstance(self.envelope,dict) and set(self.envelope)=={"model","sha256"},"Unknown export envelope fields")
        m=self.envelope["model"]; self.identity=m["identity"]; self.width=config.integer(self.identity["width"],"width",16,256)
        need(config.digest(m)==self.envelope["sha256"] and m["schema"]=="congklak-mlp-export-v3" and m["encoder"]==FEATURE_ENCODER,"Unknown/damaged action model export")
        self.input_count=config.integer(self.identity["input_count"],"input_count",43,43)
        need(self.identity["rules_digest"]==config.RULES_DIGEST and m["uses_oracle"] is False,"Model rules/ancestry mismatch")
        self.weights={name:([[next(flatten_weights([number])) for number in row] for row in value] if value and isinstance(value[0],list)
                            else list(flatten_weights(value))) for name,value in m["weights"].items()}; w=self.width
        shapes={"layers.0.weight":(w,self.input_count),"layers.0.bias":(w,),"layers.2.weight":(w,w),"layers.2.bias":(w,),"layers.4.weight":(1,w),"layers.4.bias":(1,),"policy.weight":(7,w),"policy.bias":(7,)}
        need(self.weights.keys()==shapes.keys(),"Unknown model weight fields")
        digest=hashlib.sha256(); data=[]
        for name in sorted(shapes):
            v=self.weights[name]; dims=shapes[name]
            need(isinstance(v,list) and len(v)==dims[0],"Wrong weight dimensions")
            if len(dims)==2: need(all(isinstance(row,list) and len(row)==dims[1] for row in v),"Wrong matrix dimensions")
            numbers=list(flatten_weights(v)); digest.update(name.encode()); digest.update(struct.pack("<"+"f"*len(numbers),*numbers))
        self.id=digest.hexdigest(); need(self.id==m["weights_sha256"],"Model weight digest differs")
        for prefix in ("layers.0","layers.2","layers.4","policy"):
            for suffix in (".weight",".bias"): data.extend(flatten_weights(self.weights[prefix+suffix]))
        self.array=(C.c_float*len(data))(*data); self.file_sha256=sha(self.path)

    def reference(self,p):
        x,d,r=encoded(p,self.input_count)
        if r==0:return d,d*SCALE,[0.0]*7
        for prefix in ("layers.0","layers.2"):
            x=[sum(a*v for a,v in zip(row,x))+b for row,b in zip(self.weights[prefix+".weight"],self.weights[prefix+".bias"])]
            if prefix!="layers.4": x=[max(0,v) for v in x]
        value=sum(a*v for a,v in zip(self.weights["layers.4.weight"][0],x))+self.weights["layers.4.bias"][0]
        logits=[sum(a*v for a,v in zip(row,x))+b for row,b in zip(self.weights["policy.weight"],self.weights["policy.bias"])]
        residual=r*math.tanh(value)
        return d+residual,d*SCALE+rounded(residual*SCALE),logits

class NeuralNative(Native):
    def __init__(self,model,path=None,backend="auto",mode="h-none",policy_min_depth=2):
        super().__init__(path or ROOT/"build/congklak_neural_action.dll")
        self.model=model if isinstance(model,Model) else Model(model)
        need(backend in ("auto","scalar","avx2"),"Unknown inference backend")
        need(mode in MODES,"Unknown action ablation mode");config.integer(policy_min_depth,"policy_min_depth",2,128)
        self.mode=mode;self.policy_min_depth=policy_min_depth
        self.requested_backend=backend
        self.lib.cc_nn_destroy_model.argtypes=[C.c_void_p]; self.lib.cc_nn_destroy_model.restype=None
        self.lib.cc_nn_evaluate.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p]; self.lib.cc_nn_evaluate.restype=C.c_char_p
        self.lib.cc_nn_analyze.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p,C.c_int,self.callback_type,C.c_void_p]; self.lib.cc_nn_analyze.restype=C.c_char_p
        self.lib.cc_nn_create_model_action.argtypes=[C.POINTER(C.c_float),C.c_int,C.c_int,C.c_int,C.c_char_p,C.c_int,C.c_int,C.c_int]; self.lib.cc_nn_create_model_action.restype=C.c_void_p
        self.lib.cc_nn_backend.argtypes=[C.c_void_p]; self.lib.cc_nn_backend.restype=C.c_char_p
        self.lib.cc_nn_avx2_available.argtypes=[]; self.lib.cc_nn_avx2_available.restype=C.c_int
        self.lib.cc_nn_measure_inference.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_int,C.c_int,C.c_void_p]; self.lib.cc_nn_measure_inference.restype=C.c_char_p
        self.handle=self.lib.cc_nn_create_model_action(self.model.array,len(self.model.array),self.model.width,self.model.input_count,self.model.id.encode("ascii"),{"auto":0,"scalar":1,"avx2":2}[backend],MODES[mode],policy_min_depth)
        need(self.handle,"Native model admission failed"); self.lock=threading.RLock(); self.running=0
        self.backend=self.lib.cc_nn_backend(self.handle).decode("ascii"); self.avx2_available=bool(self.lib.cc_nn_avx2_available())

    def measure_inference(self,positions,repeats=1):
        rows=[config.position(p) for p in positions]
        config.integer(len(rows),"position_count",1,10000); config.integer(repeats,"repeats",1,10000)
        need(len(rows)*repeats<=1000000,"Inference benchmark limit")
        cells=(C.c_int*(len(rows)*16))(*(v for p in rows for v in flatten(p)))
        actors=(C.c_int*len(rows))(*(p["controller"] for p in rows))
        with self.lock:
            need(self.handle,"Model is closed")
            result=self._decode(self.lib.cc_nn_measure_inference(cells,actors,len(rows),repeats,self.handle))
        need(result.get("ok"),str(result)); return result

    def close(self):
        with self.lock:
            need(not self.running,"Cannot close a model with active analysis")
            if self.handle: self.lib.cc_nn_destroy_model(self.handle); self.handle=None
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    def make_control(self,max_depth=16,time_ms=1000,node_budget=0,memory_mb=512,tt_mb=16,use_pvs=True):
        config.integer(max_depth,"max_depth",0,128); config.integer(time_ms,"time_ms",0,43200000)
        config.integer(node_budget,"node_budget",0,2**63-1); config.integer(memory_mb,"memory_mb",16,24576)
        config.integer(tt_mb,"tt_mb",0,min(1024,memory_mb-16)); need(type(use_pvs) is bool,"PVS must be boolean")
        control=Control(self,max_depth,time_ms,node_budget,memory_mb,tt_mb,use_pvs)
        control.neural_limits=(max_depth,time_ms,node_budget,memory_mb,tt_mb,use_pvs)
        return control
    def evaluate(self,position):
        p=config.position(position); arr,_=self._array(flatten(p),7,98)
        with self.lock:
            need(self.handle,"Model is closed")
            result=self._decode(self.lib.cc_nn_evaluate(arr,p["controller"],self.handle))
        need(result.get("ok"),str(result)); return result

    def neural_analyze(self,position,max_depth=16,time_ms=1000,node_budget=0,start_depth=1,tt_mb=16,use_pvs=True,memory_mb=512,on_report=None,control=None):
        p=config.position(position); arr,_=self._array(flatten(p),7,98); owned=control is None
        control=control or self.make_control(max_depth,time_ms,node_budget,memory_mb,tt_mb,use_pvs)
        need(control.native is self,"Control belongs to another library")
        need(getattr(control,"neural_limits",None)==(max_depth,time_ms,node_budget,memory_mb,tt_mb,use_pvs),"Control budgets/options differ; use make_control with matching limits")
        errors=[]; reports=[]
        def progress(raw,_):
            try:
                r=json.loads(raw.decode("utf-8"))
                if on_report and on_report(r) is False: return 0
                reports[:]=[r]; return 1
            except Exception as exc: errors.append(str(exc)); return 0
        callback=self.callback_type(progress)
        leased=False; busy=False
        try:
            with self.lock:
                need(self.handle,"Model is closed"); self.running+=1; leased=True
            with control._lock:
                need(control.handle and not control._running,"Control is closed or busy"); control._running=True; busy=True
            result=self._decode(self.lib.cc_nn_analyze(arr,p["controller"],control.handle,self.handle,start_depth,callback,None))
            if errors: raise NativeError("Report callback failed: "+errors[0])
            need(result.get("ok"),str(result)); return result
        finally:
            if busy:
                with control._lock: control._running=False
            if leased:
                with self.lock: self.running-=1
            if owned: control.close()

def validate_report(report,position,native):
    fields={"schema","completed_iteration","choice_depth","position","input_controller","forced_passes","terminal","model_sha256","policy","score_scale",
            "value_units","best_moves","candidates","pv","proof_status","nodes","deposits","elapsed_ms","tt_allocated_bytes","nn_evaluations","nn_cache_hits","tt_hits","pvs_probes","pvs_researches","mode","policy_min_depth","policy_ordered_nodes"}
    need(isinstance(report,dict) and set(report)==fields,"Unknown neural report fields")
    need(report["schema"]=="neural-analysis-v1" and report["completed_iteration"] is True,"Incomplete neural report")
    depth=config.integer(report["choice_depth"],"depth",0,128)
    need(report["model_sha256"]==native.model.id and report["policy"]==POLICY and report["score_scale"]==SCALE,"Report identity differs")
    need(report["mode"]==native.mode and report["policy_min_depth"]==native.policy_min_depth,"Action mode/ordering threshold differs")
    normal=native.normalize(flatten(position),actor=position["controller"],total=98)
    need(report["input_controller"]==position["controller"] and type(report["terminal"]) is bool,"Wrong neural report perspective")
    need(report["position"]==normal["position"] and report["terminal"]==normal["terminal"] and report["forced_passes"]==normal["passes"],"Report root differs")
    actions=normal["legal_actions"]; root=normal["position"]; actor=root["controller"]
    need([r["pit"] for r in report["candidates"]]==actions,"Incomplete candidate set")
    def check_pv(items,value,first=None):
        current=root
        for i,move in enumerate(items):
            n=native.normalize(flatten(current),actor=current["controller"],total=98)
            need(move["actor"]==n["normalized_controller"] and move["pit"] in n["legal_actions"],"Illegal PV")
            if i==0 and first is not None: need(move["pit"]==first,"Wrong candidate PV origin")
            t=native.apply(flatten(current),actor=current["controller"],action=move["pit"],total=98,trace=False)
            need(move["next_controller"]==(None if t["terminal"] else t["position"]["controller"]) and move["terminal_after"]==t["terminal"]
                 and move["forced_passes_after"]==t["passes"] and move["bonus_choice"]==(not t["terminal"] and t["reason"]=="own_store"),"PV transition differs")
            current=t["position"]
        need(len(items)<=depth and (len(items)==depth or not any(map(sum,current["pits"]))),"Short neural PV")
        need(native.evaluate(current)["value_units"]==value,"PV leaf score differs")
    values=[]
    for row in report["candidates"]:
        t=native.apply(flatten(root),actor=actor,action=row["pit"],total=98,trace=True)
        need(row["transition"]==t,"Candidate transition differs")
        if depth:
            config.integer(row["value_units"],"score",-98*SCALE,98*SCALE)
            check_pv(row["pv"],row["value_units"],row["pit"]); values.append(row["value_units"])
        else: need(row["value_units"] is None and row["pv"]==[],"Depth-zero recommendation")
    config.integer(report["value_units"],"score",-98*SCALE,98*SCALE)
    d=root["stores"][0]-root["stores"][1]; active=sum(map(sum,root["pits"]))
    need((d-active)*SCALE<=report["value_units"]<=(d+active)*SCALE,"Report exceeds physical margin interval")
    for key in ("nodes","deposits","elapsed_ms","tt_allocated_bytes","nn_evaluations","nn_cache_hits","tt_hits","pvs_probes","pvs_researches","policy_ordered_nodes"):
        config.integer(report[key],key,0,2**64-1)
    if values:
        best=(max if actor==0 else min)(values)
        need(best==report["value_units"] and report["best_moves"]==[a for a,v in zip(actions,values) if v==best],"Root rankings differ")
    else: need(report["best_moves"]==[],"Unranked root has recommendations")
    check_pv(report["pv"],report["value_units"])
    expected="terminal_outcome" if normal["terminal"] else ("finite_horizon_result" if native.mode.startswith("h-") else "finite_horizon_neural_estimate")
    need(report["proof_status"]==expected,"Neural result claims an invalid proof")
    return report

class Analysis:
    def __init__(self,path,position,native,tt_mb=16,use_pvs=True):
        from .pilot import Work
        self.root=config.position(position); self.native=native
        self.identity={"schema":"neural-action-job-v1","root":self.root,"rules_digest":config.RULES_DIGEST,"native_build_sha256":native.build_id,
                       "model_sha256":native.model.id,"model_file_sha256":native.model.file_sha256,"source_sha256":sha(__file__),
                       "input_count":native.model.input_count,"backend":native.backend,"mode":native.mode,"policy_min_depth":native.policy_min_depth,
                       "policy":POLICY,"scale":SCALE,"tt_mb":tt_mb,"use_pvs":use_pvs}
        self.store=Work(path,self.identity)
        self.db=self.store.db
        self.db.execute("CREATE TABLE IF NOT EXISTS iterations(depth INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS invocations(id INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)"); self.db.commit()
        self.last=None
        try:
            for row in self.db.execute("SELECT depth,body,hash FROM iterations ORDER BY depth"):
                r=json.loads(row["body"]); need(config.digest(r)==row["hash"] and r["choice_depth"]==row["depth"],"Corrupt neural checkpoint")
                validate_report(r,self.root,native); self.last=r
        except Exception: self.close(); raise
    def close(self): self.store.close()
    def run(self,max_depth=16,time_ms=1000,node_budget=0,on_event=None,control=None):
        config.integer(max_depth,"max_depth",0,128)
        config.integer(time_ms,"time_ms",0,43200000); config.integer(node_budget,"node_budget",0,2**63-1)
        start=self.last["choice_depth"]+1 if self.last else (0 if max_depth==0 else 1)
        normal=self.native.normalize(flatten(self.root),actor=self.root["controller"],total=98)
        if self.last and (normal["terminal"] or start>max_depth): return {"ok":True,"stop_reason":"already_complete","report":self.last}
        def progress(report):
            validate_report(report,self.root,self.native); self.store.admit(1024*1024)
            with self.db: self.db.execute("INSERT INTO iterations VALUES(?,?,?)",(report["choice_depth"],config.canonical(report),config.digest(report)))
            self.last=report
            if on_event: on_event({"event":"neural_iteration_committed","choice_depth":report["choice_depth"],"best_moves":report["best_moves"]})
            return True
        options={"max_depth":max_depth,"time_ms":time_ms,"node_budget":node_budget,"start_depth":start,
                 "tt_mb":self.identity["tt_mb"],"use_pvs":self.identity["use_pvs"]}
        result=self.native.neural_analyze(self.root,**options,on_report=progress,control=control)
        result["report"]=self.last
        body={"limits":options,"stop_reason":result["stop_reason"],"last_depth":None if self.last is None else self.last["choice_depth"]}
        with self.db: self.db.execute("INSERT INTO invocations(body,hash) VALUES(?,?)",(config.canonical(body),config.digest(body)))
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("operation",choices=("evaluate","analyze"))
    p.add_argument("--model",required=True); p.add_argument("--position",help="JSON file; otherwise read stdin")
    p.add_argument("--backend",choices=("auto","scalar","avx2"),default="auto")
    p.add_argument("--mode",choices=tuple(MODES),default="h-none");p.add_argument("--policy-min-depth",type=int,default=2)
    p.add_argument("--run",default=str(ROOT/"data/neural-action-analysis")); p.add_argument("--depth",type=int,default=16)
    p.add_argument("--ms",type=int,default=1000); p.add_argument("--nodes",type=int,default=0)
    args=p.parse_args(); job=None
    def event(e): print(config.canonical(e),flush=True)
    try:
        root=json.loads(Path(args.position).read_text() if args.position else input())
        with NeuralNative(args.model,backend=args.backend,mode=args.mode,policy_min_depth=args.policy_min_depth) as native:
            if args.operation=="evaluate": result=native.evaluate(root)
            else:
                job=Analysis(args.run,root,native)
                with native.make_control(args.depth,args.ms,args.nodes,512,16,True) as control:
                    signal.signal(signal.SIGINT,lambda *_:control.cancel())
                    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,lambda *_:control.cancel())
                    result=job.run(args.depth,args.ms,args.nodes,event,control)
                job.close(); job=None
                if result.get("report"):
                    r=result["report"]
                    result["summary"]={"best_pits_one_based":[a+1 for a in r["best_moves"]],"estimated_margin_player0":r["value_units"]/SCALE,
                                       "completed_choice_depth":r["choice_depth"],"scope":r["proof_status"]}
        event({"event":"neural_done","ok":True,"result":result}); return 0
    except Exception as exc:
        event({"event":"neural_error","ok":False,"error":str(exc)}); return 2
    finally:
        if job: job.close()
if __name__=="__main__": raise SystemExit(main())
