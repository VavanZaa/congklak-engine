"""Versioned depth-six trainer. Frozen pilot trainer remains available for old runs."""
from __future__ import annotations
import argparse
import ctypes as C
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import sys
import time
import uuid

# Must precede CUDA initialization for deterministic matrix operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn
from . import config,pilot
from .native import Native,ROOT,flatten
from . import campaign

SCHEMA="curriculum-mlp-experiment-v2"
ENCODER="current-controller-pit14-and-active15/store-offset-exact-v1"
DEFAULT_RUN=ROOT/"data"/"training-stronger-seed2718"
GIB=1024**3

class TrainingError(RuntimeError):
    def __init__(self,message,code="training_error"):
        super().__init__(message); self.code=code

def host_memory():
    if os.name!="nt": return 0
    class Counters(C.Structure):
        _fields_=[("cb",C.c_ulong),("faults",C.c_ulong)]+[(name,C.c_size_t) for name in
                  ("peak","working","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile")]
    counters=Counters(); counters.cb=C.sizeof(counters)
    kernel=C.WinDLL("kernel32",use_last_error=True); kernel.GetCurrentProcess.restype=C.c_void_p
    psapi=C.WinDLL("psapi",use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(Counters),C.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),C.byref(counters),counters.cb):
        raise TrainingError("Cannot monitor training process memory","memory_monitor_failed")
    if counters.working>=8*GIB: raise TrainingError("Training process exceeds 8 GiB RAM monitor limit","memory_budget")
    return counters.peak

def require(condition,message):
    if not condition: raise TrainingError(message,"corrupt_checkpoint")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def encode(position):
    p=config.position(position)
    active=sum(sum(r) for r in p["pits"])
    actor=p["controller"]
    if active and not sum(p["pits"][actor]): actor=1-actor
    pits=p["pits"][actor]+p["pits"][1-actor]
    return [v/98 for v in pits]+[active/98],p["stores"][0]-p["stores"][1],active*(1 if actor==0 else -1)

class Evaluator(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(15,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1),nn.Tanh())

    def forward(self,x,offset,scale):
        return offset+scale*self.layers(x).squeeze(-1)

def deterministic(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device=="cuda":
        if not torch.cuda.is_available(): raise TrainingError("CUDA is unavailable; choose --device cpu")
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.75,12*GIB/total),0)
        torch.cuda.reset_peak_memory_stats()

def load_data(path=campaign.DEFAULT):
    manifest=pilot.load(path)
    require(not Path(path,"quarantine.json").exists() and manifest["identity"]["schema"]=="stronger-corpus-v2","Quarantined or wrong curriculum dataset version")
    _,reserved,old_families,registry=campaign.imports(manifest["identity"]["parent_path"])
    require(registry==manifest["identity"]["imported_registry_sha256"],"Imported lineage exclusions changed")
    native=Native()
    require(manifest["identity"]["native_build_sha256"]==native.build_id,"Dataset/engine build differs")
    root=Path(path).resolve()/manifest["directory"]
    db=sqlite3.connect((root/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)
    db.execute("BEGIN")
    groups={}
    try:
        # SQL returns only train/development bodies. Sealed targets never enter Python.
        for split in ("train","development"):
            rows=[]
            query="SELECT state,body,hash FROM examples WHERE json_extract(body,'$.split')=? ORDER BY state"
            for key,body,checksum in db.execute(query,(split,)):
                row=json.loads(body)
                require(config.digest(row)==checksum,"Damaged training row")
                require(row["split"]==split and row["state_key"]==key.hex(),"Dataset row ownership/key mismatch")
                require(row["rules_digest"]==config.RULES_DIGEST and row["teacher_revision"]==manifest["identity"]["teacher_revision"],"Row teacher/rules identity mismatch")
                require(row["label_weight"]==1 and row["importance_ratio"]==1,"Unexpected pilot weighting")
                position=config.position(row["state"])
                canonical_key,canonical,_=pilot.canonicalize(position,native)
                require(canonical_key==key and canonical==position,"Noncanonical dataset state")
                active=sum(sum(r) for r in position["pits"])
                require(("state",key) not in reserved and ("quotient",pilot.quotient(position)) not in reserved and row["family"] not in old_families,"Prior lineage leaked into curriculum")
                label=row["label"]
                if split=="train":
                    require(active>4 and label["kind"]=="finite_choice_horizon" and label["oracle_manifest_sha256"] is None,"Oracle ancestry in training")
                    require(label["choice_depth"]==6,"Unexpected teacher horizon")
                    require(db.execute("SELECT 1 FROM exclusions WHERE kind='state' AND state=?",(key,)).fetchone() is None,"Holdout state leaked")
                    require(db.execute("SELECT 1 FROM exclusions WHERE kind='quotient' AND state=?",(pilot.quotient(position),)).fetchone() is None,"Holdout quotient leaked")
                rows.append(row)
            require(bool(rows),"Dataset split is empty")
            groups[split]=rows
    finally: db.close()
    counts=Counter(r["stratum"] for r in groups["train"])
    for row in groups["train"]:
        require(row["draw_probability"]==1/(len(counts)*counts[row["stratum"]]),"P != uniform-stratum q")
    fingerprints={s:config.digest([r["state_key"] for r in rows]) for s,rows in groups.items()}
    return manifest,groups,fingerprints

def tensors(rows,device):
    encoded=[encode(r["state"]) for r in rows]
    return (torch.tensor([e[0] for e in encoded],dtype=torch.float32,device=device),
            torch.tensor([e[1] for e in encoded],dtype=torch.float32,device=device),
            torch.tensor([e[2] for e in encoded],dtype=torch.float32,device=device),
            torch.tensor([r["label"]["target_player0"] for r in rows],dtype=torch.float32,device=device))

def regression_metrics(prediction,target,offset,rows):
    error=(prediction-target).abs().cpu().tolist(); baseline=(offset-target).abs().cpu().tolist()
    buckets=defaultdict(list)
    for i,row in enumerate(rows): buckets[row["stratum"]].append(i)
    strata={s:{"count":len(indices),"mae_seeds":sum(error[i] for i in indices)/len(indices),
               "H_v1_mae_seeds":sum(baseline[i] for i in indices)/len(indices)} for s,indices in buckets.items()}
    return {"count":len(rows),"mae_seeds":sum(error)/len(error),"H_v1_mae_seeds":sum(baseline)/len(baseline),
            "uniform_stratum_mae_seeds":sum(v["mae_seeds"] for v in strata.values())/len(strata),"strata":strata}

def evaluate(model,rows,device):
    model.eval()
    result={}
    for kind in sorted({r["label"]["kind"] for r in rows}):
        subset=[r for r in rows if r["label"]["kind"]==kind]
        x,d,r,y=tensors(subset,device)
        with torch.no_grad(): predicted=model(x,d,r)
        require(bool(torch.isfinite(predicted).all()),"Nonfinite model evaluation")
        result[kind]=regression_metrics(predicted,y,d,subset)
    return result

def weight_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        tensor=tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(bytes(tensor.view(torch.uint8).flatten().tolist()))
    return h.hexdigest()

class Checkpoints:
    def __init__(self,path,identity):
        self.path=Path(path).resolve(); self.path.mkdir(parents=True,exist_ok=True)
        self.identity=identity; self.db=None; self.lock=None
        try:
            self.lock=(self.path/"writer.lock").open("a+b")
            if self.lock.seek(0,2)==0: self.lock.write(b"0"); self.lock.flush()
            self.lock.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.admit(8*1024**2)
            self.db=sqlite3.connect(self.path/"catalog.sqlite3")
            self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("CREATE TABLE IF NOT EXISTS config(body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS checkpoints(id INTEGER PRIMARY KEY,step INTEGER,name TEXT,sha TEXT,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS history(step INTEGER PRIMARY KEY,body TEXT,hash TEXT);")
            old=self.db.execute("SELECT body,hash FROM config").fetchone()
            if old:
                require(config.digest(json.loads(old[0]))==old[1],"Corrupt training identity")
                if json.loads(old[0])!=identity: raise TrainingError("Training dataset/options/device/framework/source changed; use a new run directory","incompatible_run")
            else:
                with self.db: self.db.execute("INSERT INTO config VALUES(?,?)",(config.canonical(identity),config.digest(identity)))
        except Exception:
            self.close(); raise

    def admit(self,projected):
        total=sum(p.stat().st_size for p in self.path.parent.rglob("*") if p.is_file())
        if total+projected+2*GIB>50*GIB or shutil.disk_usage(self.path).free<projected+2*GIB:
            raise TrainingError("Insufficient physical quota/free space plus 2 GiB checkpoint recovery reserve","storage_error")

    def save(self,state,metrics,on_event=None):
        if self.identity["device"]=="cuda": torch.cuda.synchronize()
        self.admit(8*1024**2)
        name="checkpoint-"+uuid.uuid4().hex+".pt"
        file=self.path/name
        with file.open("xb") as stream:
            torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
        checksum=sha(file)
        restored=torch.load(file,map_location="cpu",weights_only=True)
        require(restored["step"]==state["step"] and restored["identity"]==self.identity,"Checkpoint verification failed")
        require(weight_digest(restored["model"])==weight_digest(state["model"]),"Checkpoint weight verification failed")
        if on_event: on_event({"event":"checkpoint_file_ready","step":state["step"],"name":name})
        body={"step":state["step"],"identity_sha256":config.digest(self.identity),"weights_sha256":weight_digest(state["model"]),"metrics":metrics}
        with self.db:
            self.db.execute("INSERT INTO checkpoints(step,name,sha,body,hash) VALUES(?,?,?,?,?)",(state["step"],name,checksum,config.canonical(body),config.digest(body)))
            self.db.execute("INSERT OR REPLACE INTO history VALUES(?,?,?)",(state["step"],config.canonical(metrics),config.digest(metrics)))
        if on_event: on_event({"event":"checkpoint_committed",**body,"name":name})
        return body

    def restore(self):
        invalid=[]
        for step,name,checksum,body,bodyhash in self.db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id DESC"):
            try:
                file=(self.path/name).resolve()
                require(file.parent==self.path and file.is_file() and sha(file)==checksum,"Missing/damaged checkpoint")
                meta=json.loads(body); require(config.digest(meta)==bodyhash,"Corrupt checkpoint catalog")
                state=torch.load(file,map_location="cpu",weights_only=True)
                require(type(state["step"]) is int and state["step"]==step and state["identity"]==self.identity,"Checkpoint identity/step mismatch")
                require(meta["weights_sha256"]==weight_digest(state["model"]),"Checkpoint model digest mismatch")
                return state,{"invalid_generations":invalid,"restored_step":step}
            except (OSError,ValueError,KeyError,RuntimeError,EOFError) as exc:
                invalid.append({"step":step,"reason":str(exc)})
        if invalid: raise TrainingError("No valid committed checkpoint remains","corrupt_checkpoint")
        return None,{"invalid_generations":[],"restored_step":0}

    def close(self):
        if self.db: self.db.close(); self.db=None
        if self.lock: self.lock.close(); self.lock=None

def export_model(model,path,identity,step):
    weights={k:v.detach().cpu().tolist() for k,v in model.state_dict().items()}
    value={"schema":"congklak-mlp-export-v1","encoder":ENCODER,"identity":identity,"step":step,
           "status":"experimental_pilot_candidate","weights":weights,"weights_sha256":weight_digest(model.state_dict()),
           "terminal":"store0-minus-store1","output":"D_player0 + signed_active*tanh(current-actor-residual)","uses_oracle":False}
    out={"model":value,"sha256":config.digest(value)}
    atomic_json(path,out)
    return out

def predict_export(export,position):
    value=export["model"]
    require(config.digest(value)==export["sha256"] and value["encoder"]==ENCODER,"Damaged/unknown model export")
    x,d,r=encode(position); w=value["weights"]
    for prefix in ("layers.0","layers.2","layers.4"):
        x=[sum(v*a for v,a in zip(row,x))+b for row,b in zip(w[prefix+".weight"],w[prefix+".bias"])]
        if prefix!="layers.4": x=[max(v,0) for v in x]
    return d+r*math.tanh(x[0])

def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)

def train(path=DEFAULT_RUN,dataset=campaign.DEFAULT,steps=2000,seed=2718,width=64,batch_size=512,lr=.001,
          device="cuda",checkpoint_every=100,on_event=None,cancel=None):
    config.integer(steps,"steps",1,100000); config.integer(seed,"seed",0,2**31-1)
    config.integer(width,"width",16,256); config.integer(batch_size,"batch_size",1,4096); config.integer(checkpoint_every,"checkpoint_every",1,10000)
    require(device in ("cuda","cpu") and type(lr) in (float,int) and 0<lr<=.1,"Invalid training configuration")
    deterministic(device)
    manifest,data,fingerprints=load_data(dataset)
    identity={"schema":SCHEMA,"rules_digest":config.RULES_DIGEST,"dataset_semantic_sha256":manifest["semantic_sha256"],
              "dataset_manifest_sha256":config.digest(manifest),"teacher_revision":manifest["identity"]["teacher_revision"],
              "split_fingerprints":fingerprints,"seed":seed,"width":width,"batch_size":batch_size,"steps":steps,"lr":lr,
              "checkpoint_every":checkpoint_every,"device":device,"torch_version":str(torch.__version__),"cuda_runtime":torch.version.cuda,
              "python_version":platform.python_version(),"device_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu",
              "source_sha256":sha(__file__),"encoder":ENCODER,"objective":"Huber(delta=1)/reconstructed-player0-seed-units",
              "sampler":"uniform-primary-stratum/with-replacement/P=q/unit-weight/no-prefetch-v1","optimizer":"AdamW/weight_decay0.0001/grad_clip1/fp32",
              "scheduler":"none","amp":False,"oracle_training":False,"sealed_final_test_consumed":False,"loader_workers":0}
    store=Checkpoints(path,identity)
    started=time.monotonic(); rng=random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model=Evaluator(width).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001,foreach=False)
    best=None; best_score=math.inf; best_step=0; step=0; draws=0
    groups=defaultdict(list)
    for i,row in enumerate(data["train"]): groups[row["stratum"]].append(i)
    strata=sorted(groups); x,d,r,y=tensors(data["train"],device)
    try:
        saved,recovery=store.restore()
        if saved:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value): state[key]=value.cpu() if key=="step" else value.to(device)
            rng.setstate(saved["sampler_rng"]); random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            if device=="cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
            step=saved["step"]; draws=saved["draws"]; best=saved["best_model"]
            best_score=saved["best_score"]; best_step=saved["best_step"]
        if on_event: on_event({"event":"training_started","identity":identity,"train_rows":len(data["train"]),"development_rows":len(data["development"]),"recovery":recovery})
        final_metrics=None
        last_saved=step
        def commit():
            nonlocal final_metrics,best_score,best_step,best,last_saved
            if device=="cuda": torch.cuda.synchronize()
            host_memory()
            final_metrics=evaluate(model,data["development"],device)
            score=final_metrics["finite_choice_horizon"]["uniform_stratum_mae_seeds"]
            # Recovery snapshots between scheduled evaluations must not add
            # extra model-selection opportunities to interrupted runs.
            if (step%checkpoint_every==0 or step==steps) and score<best_score:
                best_score=score; best_step=step
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            state={"identity":identity,"step":step,"draws":draws,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   "optimizer":optimizer.state_dict(),"sampler_rng":rng.getstate(),"python_rng":random.getstate(),
                   "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else [],
                   "best_model":best,"best_score":best_score,"best_step":best_step,"scheduler":None,"amp_scaler":None,"pending_batches":[]}
            store.save(state,final_metrics,on_event); last_saved=step
        while step<steps:
            if cancel and cancel(): break
            indices=[rng.choice(groups[rng.choice(strata)]) for _ in range(batch_size)]
            index=torch.tensor(indices,dtype=torch.int64,device=device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction=model(x[index],d[index],r[index])
            loss=nn.functional.huber_loss(prediction,y[index],delta=1.0)
            require(bool(torch.isfinite(loss)),"Nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            step+=1; draws+=batch_size
            if device=="cuda":
                require(torch.cuda.max_memory_reserved()<12*GIB,"Training exceeded 12 GiB allocated GPU reserve")
            if step%checkpoint_every==0 or step==steps or (cancel and cancel()):
                commit()
        if step>last_saved: commit()
        # A cancellation before the first step has no fictitious checkpoint.
        if best is None: return {"status":"interrupted","steps_completed":step,"model_available":False,"recovery":recovery,"sealed_final_test_consumed":False}
        model.load_state_dict(best)
        selected_metrics=evaluate(model,data["development"],device)
        export=export_model(model,Path(path)/"model.json",identity,best_step)
        status="complete" if step==steps else "interrupted"
        result={"status":status,"steps_completed":step,"best_step":best_step,"optimizer_steps_target":steps,"draws":draws,
                "identity":identity,"selected_development":selected_metrics,"elapsed_seconds_this_invocation":time.monotonic()-started,
                "weights_sha256":export["model"]["weights_sha256"],"parameters":sum(p.numel() for p in model.parameters()),
                "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else 0,
                "peak_host_working_set_bytes":host_memory(),
                "sealed_final_test_consumed":False,"promotion":"not_promoted; requires independent engine/arena gates", "recovery":recovery}
        atomic_json(Path(path,"result.json"),result)
        return result
    finally: store.close()

def assess(run=DEFAULT_RUN,dataset=campaign.DEFAULT):
    export=json.loads(Path(run,"model.json").read_text(encoding="utf-8"))
    manifest,data,_=load_data(dataset)
    require(export["model"]["identity"]["dataset_semantic_sha256"]==manifest["semantic_sha256"],"Evaluation dataset changed")
    model=Evaluator(export["model"]["identity"]["width"])
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in export["model"]["weights"].items()}); model.eval()
    native=Native(); rows=data["development"]
    prediction_diff=0; batches=[]; observed=defaultdict(list)
    for row in rows:
        p=row["state"]; e=encode(p)
        with torch.no_grad(): v=model(torch.tensor([e[0]]),torch.tensor([e[1]],dtype=torch.float32),torch.tensor([e[2]],dtype=torch.float32)).item()
        prediction_diff=max(prediction_diff,abs(v-predict_export(export,p)))
        legal=[a["pit"] for a in row["label"]["action_scores"]]
        if not legal: continue
        scored=[]; baseline=[]
        for action in legal:
            child=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False)["position"]
            scored.append((predict_export(export,child),action))
            baseline.append((child["stores"][0]-child["stores"][1],action))
        best=(max if p["controller"]==0 else min)
        choice=best(scored,key=lambda x:x[0])[1]; baseline_choice=best(baseline,key=lambda x:x[0])[1]
        target=row["label"]; q={a["pit"]:a["value_player0"] for a in target["action_scores"]}
        sign=1 if p["controller"]==0 else -1
        regret=sign*(target["target_player0"]-q[choice]); baseline_regret=sign*(target["target_player0"]-q[baseline_choice])
        observed[target["kind"]].append((choice in target["best_moves"],baseline_choice in target["best_moves"],regret,baseline_regret))
    require(prediction_diff<1e-4,"CPU export inference differs beyond 0.0001 seeds")
    metrics={kind:{"positions":len(values),"candidate_optimal_set_accuracy":sum(v[0] for v in values)/len(values),
                   "H_v1_oneply_optimal_set_accuracy":sum(v[1] for v in values)/len(values),
                   "candidate_mean_move_regret_seeds":sum(v[2] for v in values)/len(values),"H_v1_oneply_mean_move_regret_seeds":sum(v[3] for v in values)/len(values)} for kind,values in observed.items()}
    result={"dataset_semantic_sha256":manifest["semantic_sha256"],"weights_sha256":export["model"]["weights_sha256"],"development_only":True,
            "sealed_final_test_consumed":False,"oracle_lookups_used_for_prediction":False,"search":"one legal choice then raw heuristic; reference exact/horizon targets kept separate",
            "export_cpu_max_difference_seeds":prediction_diff,"oneply_choice_metrics":metrics,
            "regression_metrics":evaluate(model,rows,"cpu"),"promotion":"not_promoted; one-ply diagnostic does not establish equal-time/arena improvement"}
    atomic_json(Path(run,"assessment.json"),result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=("train","assess","predict")); p.add_argument("--run",default=str(DEFAULT_RUN)); p.add_argument("--dataset",default=str(campaign.DEFAULT))
    p.add_argument("--steps",type=int,default=2000); p.add_argument("--seed",type=int,default=2718); p.add_argument("--width",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=512); p.add_argument("--lr",type=float,default=.001); p.add_argument("--checkpoint-every",type=int,default=100)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda"); args=p.parse_args(); stopped=False
    import signal
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def event(message): print(config.canonical(message),flush=True)
    try:
        if args.operation=="train": result=train(args.run,args.dataset,args.steps,args.seed,args.width,args.batch_size,args.lr,args.device,args.checkpoint_every,event,lambda:stopped)
        elif args.operation=="assess": result=assess(args.run,args.dataset)
        else:
            position=json.loads(sys.stdin.readline()); export=json.loads(Path(args.run,"model.json").read_text(encoding="utf-8"))
            result={"value_player0":predict_export(export,position),"scope":"experimental_neural_heuristic","model_sha256":export["model"]["weights_sha256"]}
        event({"event":"training_done","ok":True,"result":result}); return 0
    except Exception as exc:
        event({"event":"training_error","ok":False,"error":{"code":getattr(exc,"code","training_error"),"message":str(exc)}}); return 2

if __name__=="__main__": raise SystemExit(main())
