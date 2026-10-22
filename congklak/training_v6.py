"""Matched H6/H8 x count/relay feature VALUE-only factorial training."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn
from . import config,deeper_teacher,features_relay,training_v3 as prior
from .native import ROOT

SCHEMA="stage5-deeper-relay-value-factorial-v6"
DEFAULT_RUN=ROOT/"data/training-stage5-deeper-h8-f127-seed2718"
TrainingError=prior.TrainingError
require=prior.require
sha=prior.sha
atomic_json=deeper_teacher.atomic_json
Checkpoints=prior.Checkpoints
weight_digest=prior.weight_digest
deterministic=prior.deterministic

class Evaluator(nn.Module):
    def __init__(self,width=64,input_count=127):
        super().__init__(); features_relay.encoder(input_count); self.input_count=input_count; self.width=width
        self.layers=nn.Sequential(nn.Linear(input_count,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1),nn.Tanh())
        self.policy=nn.Linear(width,7)
        # Architecture compatibility only. No policy objective, gradient, decay,
        # or selection; zero-grad leaves these parameters outside the optimizer.
        for value in self.policy.parameters(): value.requires_grad_(False)
    def forward(self,x,offset,scale): return offset+scale*self.layers(x).squeeze(-1)

def value_state(state): return {key:value for key,value in state.items() if key.startswith("layers.")}

def tensors(rows,device,input_count,teacher_depth=None):
    features_relay.encoder(input_count)
    return (torch.tensor([r["cached_features"]["x"][:input_count] for r in rows],dtype=torch.float32,device=device),
            torch.tensor([r["cached_features"]["d"] for r in rows],dtype=torch.float32,device=device),
            torch.tensor([r["cached_features"]["r"] for r in rows],dtype=torch.float32,device=device),
            torch.tensor([r["label" if teacher_depth is None else f"H{teacher_depth}_reference"]["target_player0"] for r in rows],dtype=torch.float32,device=device))

def evaluate(model,rows,device,teacher_depth=None):
    model.eval(); x,d,r,y=tensors(rows,device,model.input_count,teacher_depth)
    with torch.no_grad(): prediction=model(x,d,r)
    require(bool(torch.isfinite(prediction).all()),"Nonfinite value prediction")
    result=prior.regression_metrics(prediction,y,d,rows)
    sign=torch.where(r>=0,1.,-1.); error=((prediction-y)*sign).cpu().tolist(); buckets=defaultdict(list)
    for index,row in enumerate(rows): buckets[row["stratum"]].append(index)
    for name,indices in buckets.items(): result["strata"][name]["actor_relative_value_bias_seeds"]=sum(error[i] for i in indices)/len(indices)
    return {"finite_choice_horizon":result}

def export_payload(state,identity,step):
    weights={key:value.detach().cpu().tolist() for key,value in state.items()}
    return {"schema":"congklak-mlp-export-v4","encoder":features_relay.encoder(identity["input_count"]),"identity":identity,"step":step,
           "status":"experimental_stage5_deeper_relay_candidate","weights":weights,"weights_sha256":weight_digest(state),"terminal":"store0-minus-store1",
           "output":"D_player0 + signed_active*tanh(current-actor-residual)","uses_oracle":False,
           "policy_output":"7 dummy untrained current-actor logits; never used or selected","policy_training":"Frozen untrained architecture-compatible head; no objective, gradients, optimizer state or weight decay"}
def export_model(model,path,identity,step):
    value=export_payload(model.state_dict(),identity,step)
    envelope={"model":value,"sha256":config.digest(value)}; atomic_json(path,envelope); return envelope

def predict_export(export,position):
    value=export["model"]; require(config.digest(value)==export["sha256"] and value["schema"]=="congklak-mlp-export-v4" and value["encoder"]==features_relay.encoder(value["identity"]["input_count"]),"Damaged/unknown relay export")
    x,d,r=features_relay.encode(position,value["identity"]["input_count"]); weights=value["weights"]
    for prefix in ("layers.0","layers.2"):
        x=[max(0.,sum(a*b for a,b in zip(row,x))+bias) for row,bias in zip(weights[prefix+".weight"],weights[prefix+".bias"])]
    raw=d+r*math.tanh(sum(a*b for a,b in zip(weights["layers.4.weight"][0],x))+weights["layers.4.bias"][0])
    logits=[sum(a*b for a,b in zip(row,x))+bias for row,bias in zip(weights["policy.weight"],weights["policy.bias"])]
    return raw,logits

def train(path=DEFAULT_RUN,dataset=deeper_teacher.DEFAULT,steps=8000,seed=2718,width=64,batch_size=512,lr=.001,
          device="cuda",checkpoint_every=100,on_event=None,cancel=None,input_count=127,teacher_depth=8):
    config.integer(steps,"steps",1,100000); config.integer(seed,"seed",0,2**31-1); config.integer(width,"width",16,64)
    config.integer(batch_size,"batch_size",1,4096); config.integer(checkpoint_every,"checkpoint_every",1,10000)
    require(type(teacher_depth) is int and teacher_depth in (6,8) and device in ("cuda","cpu") and type(lr) in (float,int) and 0<lr<=.1,"Invalid deeper training configuration")
    features_relay.encoder(input_count); deterministic(device)
    manifest,data,fingerprints=deeper_teacher.load(dataset,teacher_depth=teacher_depth)
    identity={"schema":SCHEMA,"rules_digest":config.RULES_DIGEST,"dataset_semantic_sha256":manifest["semantic_sha256"],"dataset_manifest_sha256":config.digest(manifest),
              "teacher_revision":f"H_v1/root-H{teacher_depth}/no-oracle","teacher_depth":teacher_depth,"split_fingerprints":fingerprints,
              "seed":seed,"width":width,"input_count":input_count,"batch_size":batch_size,"steps":steps,"lr":lr,"checkpoint_every":checkpoint_every,
              "device":device,"torch_version":str(torch.__version__),"cuda_runtime":torch.version.cuda,"python_version":platform.python_version(),
              "device_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu","source_sha256":sha(__file__),
              "source_dependencies_sha256":{m.__name__:sha(m.__file__) for m in (deeper_teacher,features_relay,prior,config)},
              "encoder":features_relay.encoder(input_count),"value_objective":f"Huber(delta1) root H{teacher_depth} player0 seed margin ONLY",
              "policy_objective":None,"bias_objective":None,"policy_head":"Frozen dummy; no gradients, decay, optimizer or selection",
              "selection":f"Minimum finite-development uniform-stratum H{teacher_depth} value MAE at scheduled checkpoints only; cross-horizon metrics after selection",
              "sampler":"uniform-primary-stratum/with-replacement/P=q/unit-weight/no-prefetch-v1","optimizer":"AdamW/weight_decay0.0001/grad_clip1/fp32",
              "scheduler":"none","amp":False,"oracle_training":False,"sealed_final_test_consumed":False,"loader_workers":0}
    store=Checkpoints(path,identity); started=time.monotonic(); rng=random.Random(seed); random.seed(seed); torch.manual_seed(seed)
    model=Evaluator(width,input_count).to(device); optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=lr,weight_decay=.0001,foreach=False)
    best=None; best_score=math.inf; best_step=0; step=0; draws=0; last_loss=None
    groups=defaultdict(list)
    for index,row in enumerate(data["train"]): groups[row["stratum"]].append(index)
    strata=sorted(groups); x,d,r,y=tensors(data["train"],device,input_count)
    try:
        saved,recovery=store.restore()
        if saved:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value): state[key]=value.cpu() if key=="step" else value.to(device)
            rng.setstate(saved["sampler_rng"]); random.setstate(saved["python_rng"]); torch.set_rng_state(saved["torch_rng"])
            if device=="cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
            step=saved["step"]; draws=saved["draws"]; best=saved["best_model"]; best_score=saved["best_score"]; best_step=saved["best_step"]
        completed_path=Path(path)/"result.json"
        completed=json.loads(completed_path.read_text(encoding="utf-8")) if saved is not None and step==steps and completed_path.is_file() else None
        if completed is not None:
            require(completed["identity"]==identity and completed["status"] in ("complete","interrupted"),"Existing result identity/status differs")
            if completed["status"]=="interrupted":
                require(type(completed["steps_completed"]) is int and 0<=completed["steps_completed"]<steps,"Invalid earlier interrupted result")
        if completed is not None and completed["status"]=="complete":
            require(completed["identity"]==identity and completed["status"]=="complete" and completed["steps_completed"]==steps and completed["optimizer_steps_target"]==steps and completed["draws"]==draws and completed["best_step"]==best_step,"Completed result differs from committed checkpoint")
            require(completed["sealed_final_test_consumed"] is False,"Completed result changed sealed-consumption contract")
            require(completed["parameters"]==sum(v.numel() for v in model.parameters()) and completed["trainable_parameters"]==sum(v.numel() for v in model.parameters() if v.requires_grad) and completed["dummy_policy_parameters"]==sum(v.numel() for v in model.policy.parameters()),"Completed parameter counts differ")
            require(best is not None and completed["weights_sha256"]==weight_digest(best) and completed["value_weights_sha256"]==weight_digest(value_state(best)),"Completed selected model digest differs")
            envelope=json.loads((Path(path)/"model.json").read_text(encoding="utf-8")); exported=envelope["model"]
            require(set(envelope)=={"model","sha256"},"Completed export envelope differs")
            require(config.digest(exported)==envelope["sha256"] and exported["schema"]=="congklak-mlp-export-v4" and exported["identity"]==identity and exported["step"]==best_step and exported["encoder"]==features_relay.encoder(input_count) and exported["weights_sha256"]==weight_digest(best),"Completed export identity differs")
            require(exported["weights"]=={k:v.detach().cpu().tolist() for k,v in best.items()},"Completed export weights differ")
            require(config.canonical(exported)==config.canonical(export_payload(best,identity,best_step)),"Completed export contract differs")
            model.load_state_dict(best)
            require(completed["selected_development"]==evaluate(model,data["development"],device) and completed["selected_cross_teacher_development"]=={f"H{h}":evaluate(model,data["development"],device,h) for h in (6,8)},"Completed metrics differ from selected checkpoint")
            if on_event: on_event({"event":"training_completed_verified","identity":identity,"step":step,"best_step":best_step,"weights_sha256":completed["weights_sha256"],"files_rewritten":False})
            return completed
        if on_event: on_event({"event":"training_started","identity":identity,"train_rows":len(data["train"]),"development_rows":len(data["development"]),"recovery":recovery})
        last_saved=step
        def commit():
            nonlocal best,best_score,best_step,last_saved
            if device=="cuda": torch.cuda.synchronize()
            prior.host_memory(); metrics=evaluate(model,data["development"],device); metrics["last_training_batch_loss"]=last_loss
            score=metrics["finite_choice_horizon"]["uniform_stratum_mae_seeds"]
            if (step%checkpoint_every==0 or step==steps) and score<best_score:
                best_score=score; best_step=step; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            state={"identity":identity,"step":step,"draws":draws,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),
                   "sampler_rng":rng.getstate(),"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else [],
                   "best_model":best,"best_score":best_score,"best_step":best_step,"scheduler":None,"amp_scaler":None,"pending_batches":[]}
            store.save(state,metrics,on_event); last_saved=step
        while step<steps:
            if cancel and cancel(): break
            indices=[rng.choice(groups[rng.choice(strata)]) for _ in range(batch_size)]; index=torch.tensor(indices,dtype=torch.int64,device=device)
            model.train(); optimizer.zero_grad(set_to_none=True); prediction=model(x[index],d[index],r[index]); loss=nn.functional.huber_loss(prediction,y[index],delta=1.)
            require(bool(torch.isfinite(loss)),"Nonfinite value loss"); loss.backward(); nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.); optimizer.step()
            step+=1; draws+=batch_size; last_loss=loss.item()
            if device=="cuda": require(torch.cuda.max_memory_reserved()<12*prior.GIB,"12GiB GPU reserve exceeded")
            if step%checkpoint_every==0 or step==steps or (cancel and cancel()): commit()
        if step>last_saved: commit()
        if best is None: return {"status":"interrupted","steps_completed":step,"model_available":False,"recovery":recovery,"sealed_final_test_consumed":False}
        model.load_state_dict(best); selected=evaluate(model,data["development"],device)
        # Never used to choose the checkpoint: describe both horizons afterward.
        cross={f"H{h}":evaluate(model,data["development"],device,h) for h in (6,8)}
        export=export_model(model,Path(path)/"model.json",identity,best_step)
        result={"status":"complete" if step==steps else "interrupted","steps_completed":step,"best_step":best_step,"optimizer_steps_target":steps,"draws":draws,"identity":identity,
                "selected_development":selected,"selected_cross_teacher_development":cross,"weights_sha256":export["model"]["weights_sha256"],"value_weights_sha256":weight_digest(value_state(best)),
                "parameters":sum(v.numel() for v in model.parameters()),"trainable_parameters":sum(v.numel() for v in model.parameters() if v.requires_grad),"dummy_policy_parameters":sum(v.numel() for v in model.policy.parameters()),
                "elapsed_seconds_this_invocation":time.monotonic()-started,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else 0,"peak_host_working_set_bytes":prior.host_memory(),"recovery":recovery,
                "sealed_final_test_consumed":False,"promotion":"not promoted; ordinary and independent equal-time gates required"}
        atomic_json(Path(path)/"result.json",result); return result
    finally: store.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("train",)); parser.add_argument("--run",default=str(DEFAULT_RUN)); parser.add_argument("--dataset",default=str(deeper_teacher.DEFAULT))
    parser.add_argument("--teacher-depth",type=int,choices=(6,8),default=8); parser.add_argument("--input-count",type=int,choices=(43,127),default=127)
    parser.add_argument("--steps",type=int,default=8000); parser.add_argument("--seed",type=int,default=2718); parser.add_argument("--width",type=int,default=64); parser.add_argument("--batch-size",type=int,default=512); parser.add_argument("--checkpoint-every",type=int,default=100); parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args(); stopped=False
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def event(value): print(config.canonical(value),flush=True)
    try:
        result=train(args.run,args.dataset,args.steps,args.seed,args.width,args.batch_size,device=args.device,checkpoint_every=args.checkpoint_every,on_event=event,cancel=lambda:stopped,input_count=args.input_count,teacher_depth=args.teacher_depth)
        event({"event":"training_done","ok":True,"result":result}); return 0
    except Exception as exc: event({"event":"training_error","ok":False,"error":{"message":str(exc)}}); return 2

if __name__=="__main__": raise SystemExit(main())
