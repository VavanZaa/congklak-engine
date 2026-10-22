"""Stage-five root action preferences and train-only ordinary bias ablations."""
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
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn
from . import action_targets,config,pilot,campaign,features_v3,training_v3 as prior
from . import native as native_binding
from .native import ROOT

SCHEMA="stage5-root-action-ordinary-bias-experiment-v4"
DEFAULT_RUN=ROOT/"data/training-stage5-action-C-seed2718"
OBJECTIVES={"A":{"policy_coefficient":0.0,"bias_coefficient":0.0},
            "B":{"policy_coefficient":.25,"bias_coefficient":0.0},
            "C":{"policy_coefficient":.25,"bias_coefficient":.01}}
TrainingError=prior.TrainingError
require=prior.require
sha=prior.sha
atomic_json=prior.atomic_json
Checkpoints=prior.Checkpoints
weight_digest=prior.weight_digest
load_data=prior.load_data
deterministic=prior.deterministic
encode=features_v3.encode

class Evaluator(nn.Module):
    def __init__(self,width=64):
        super().__init__(); self.input_count=43; self.width=width
        self.layers=nn.Sequential(nn.Linear(43,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1),nn.Tanh())
        self.policy=nn.Linear(width,7)

    def both(self,x,offset,scale):
        h=self.layers[1](self.layers[0](x)); h=self.layers[3](self.layers[2](h))
        residual=self.layers[5](self.layers[4](h)).squeeze(-1)
        return offset+scale*residual,self.policy(h)

    def forward(self,x,offset,scale): return self.both(x,offset,scale)[0]

def value_state(state): return {key:value for key,value in state.items() if key.startswith("layers.")}

def tensors(rows,device,calibration):
    x,d,r,y=prior.tensors(rows,device,43)
    targets=[action_targets.target(row) for row in rows]
    eligible={name:index for index,name in enumerate(calibration["eligible_strata"])}
    return {"x":x,"d":d,"r":r,"y":y,
            "probability":torch.tensor([value[0] for value in targets],dtype=torch.float32,device=device),
            "legal":torch.tensor([value[1] for value in targets],dtype=torch.bool,device=device),
            "q":torch.tensor([value[2] for value in targets],dtype=torch.float32,device=device),
            "actor_sign":torch.tensor([value[3] for value in targets],dtype=torch.float32,device=device),
            "ordinary_group":torch.tensor([eligible.get(row["stratum"],-1) for row in rows],dtype=torch.int64,device=device)}

def policy_loss(logits,probability,legal):
    # Explicitly zero illegal log-probabilities, so all illegal contributions
    # remain zero even if a future backend represents its mask as -infinity.
    require(bool(legal.any(dim=1).all()),"Training policy has no legal action")
    masked=logits.masked_fill(~legal,-1e9)
    log_probability=nn.functional.log_softmax(masked,dim=1).masked_fill(~legal,0.0)
    return -(probability*log_probability).sum(dim=1).mean()

def ordinary_bias_loss(error,actor_sign,group,group_count):
    terms=[]
    for index in range(group_count):
        selected=group==index
        if bool(selected.any()): terms.append((error[selected]*actor_sign[selected]).mean().square())
    return torch.stack(terms).mean() if terms else error.sum()*0.0

def evaluate(model,rows,device,calibration):
    model.eval(); t=tensors(rows,device,calibration)
    with torch.no_grad(): prediction,logits=model.both(t["x"],t["d"],t["r"])
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(logits).all()),"Nonfinite model evaluation")
    regression=prior.regression_metrics(prediction,t["y"],t["d"],rows)
    selected=logits.masked_fill(~t["legal"],-1e9).argmax(dim=1)
    signed_q=t["q"].masked_fill(~t["legal"],-1e9)
    optimum=signed_q.max(dim=1).values
    chosen_q=t["q"].gather(1,selected[:,None]).squeeze(1)
    regret=(optimum-chosen_q).cpu().tolist(); correct=[value==0 for value in regret]
    biases=((prediction-t["y"])*t["actor_sign"]).cpu().tolist()
    buckets=defaultdict(list)
    for index,row in enumerate(rows): buckets[row["stratum"]].append(index)
    strata={name:{"count":len(indices),"mean_root_H6_Q_regret_seeds":sum(regret[i] for i in indices)/len(indices),
                  "root_H6_optimal_set_accuracy":sum(correct[i] for i in indices)/len(indices),
                  "actor_relative_value_bias_seeds":sum(biases[i] for i in indices)/len(indices)} for name,indices in buckets.items()}
    return {"finite_choice_horizon":regression,
            "policy":{"count":len(rows),"root_H6_optimal_set_accuracy":sum(correct)/len(correct),"mean_root_H6_Q_regret_seeds":sum(regret)/len(regret),
                      "uniform_stratum_root_H6_Q_regret_seeds":sum(value["mean_root_H6_Q_regret_seeds"] for value in strata.values())/len(strata),"strata":strata,
                      "target_scope":"Root H6 chosen-action + successor H5 values are policy preferences, not successor H6 value labels"}}

def export_model(model,path,identity,step,include_policy=False):
    state=model.state_dict() if include_policy else value_state(model.state_dict())
    weights={key:value.detach().cpu().tolist() for key,value in state.items()}
    value={"schema":"congklak-mlp-export-v3" if include_policy else "congklak-mlp-export-v2",
           "encoder":features_v3.ENCODER43,"identity":identity,"step":step,"status":"experimental_stage5_action_candidate",
           "weights":weights,"weights_sha256":weight_digest(state),"terminal":"store0-minus-store1",
           "output":"D_player0 + signed_active*tanh(current-actor-residual)","uses_oracle":False}
    if include_policy:
        value.update({"policy_output":"7 unbounded current-root-actor preference logits; legal actions masked by rules; terminal ordering empty",
                      "policy_target_schema":action_targets.SCHEMA,"policy_training":"untrained control head; zero objective except AdamW decay" if identity["variant"]=="A" else "root H6 legal soft preferences, tau4 seeds"})
    envelope={"model":value,"sha256":config.digest(value)}; atomic_json(path,envelope); return envelope

def predict_export(export,position):
    value=export["model"]; require(config.digest(value)==export["sha256"] and value["encoder"]==features_v3.ENCODER43,"Unknown/damaged action model export")
    x,d,r=encode(position,43); weights=value["weights"]
    for prefix in ("layers.0","layers.2"):
        x=[max(0.0,sum(a*b for a,b in zip(row,x))+bias) for row,bias in zip(weights[prefix+".weight"],weights[prefix+".bias"])]
    raw=d+r*math.tanh(sum(a*b for a,b in zip(weights["layers.4.weight"][0],x))+weights["layers.4.bias"][0])
    policy=[sum(a*b for a,b in zip(row,x))+bias for row,bias in zip(weights.get("policy.weight",[]),weights.get("policy.bias",[]))]
    return raw,policy

def train(path=DEFAULT_RUN,dataset=campaign.DEFAULT,steps=8000,seed=2718,width=64,batch_size=512,lr=.001,
          device="cuda",checkpoint_every=100,on_event=None,cancel=None,variant="C"):
    config.integer(steps,"steps",1,100000); config.integer(seed,"seed",0,2**31-1); config.integer(width,"width",16,64)
    config.integer(batch_size,"batch_size",1,4096); config.integer(checkpoint_every,"checkpoint_every",1,10000)
    require(variant in OBJECTIVES and device in ("cuda","cpu") and type(lr) in (float,int) and 0<lr<=.1,"Invalid action training configuration")
    deterministic(device); manifest,data,fingerprints=load_data(dataset)
    calibration=action_targets.ordinary_calibration(data["train"])
    identity={"schema":SCHEMA,"rules_digest":config.RULES_DIGEST,"dataset_semantic_sha256":manifest["semantic_sha256"],
              "dataset_manifest_sha256":config.digest(manifest),"teacher_revision":manifest["identity"]["teacher_revision"],"split_fingerprints":fingerprints,
              "seed":seed,"width":width,"input_count":43,"variant":variant,"batch_size":batch_size,"steps":steps,"lr":lr,"checkpoint_every":checkpoint_every,
              "device":device,"torch_version":str(torch.__version__),"cuda_runtime":torch.version.cuda,"python_version":platform.python_version(),
              "device_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu","source_sha256":sha(__file__),
              "source_dependencies_sha256":{module.__name__:sha(module.__file__) for module in (action_targets,features_v3,prior,config,pilot,campaign,native_binding)},
              "encoder":features_v3.ENCODER43,"value_objective":"Huber(delta1) root H6 player0 seed margin only",
              "policy_objective":{"schema":action_targets.SCHEMA,"temperature_seeds":action_targets.TEMPERATURE_SEEDS,"coefficient":OBJECTIVES[variant]["policy_coefficient"],
                                  "perspective":"normalized ROOT current actor for every action Q, irrespective of successor bonus/pass/controller",
                                  "semantics":"Action Q = root chosen choice + H5 successor; policy preference only, not successor H6 value target"},
              "ordinary_bias_calibration":calibration,"bias_coefficient":OBJECTIVES[variant]["bias_coefficient"],
              "selection":"Minimum finite-development uniform-stratum value MAE at scheduled checkpoints only; policy metrics separate",
              "sampler":"uniform-primary-stratum/with-replacement/P=q/unit-weight/no-prefetch-v1","optimizer":"AdamW/weight_decay0.0001/grad_clip1/fp32",
              "scheduler":"none","amp":False,"oracle_training":False,"sealed_final_test_consumed":False,"loader_workers":0}
    store=Checkpoints(path,identity); started=time.monotonic(); rng=random.Random(seed); random.seed(seed); torch.manual_seed(seed)
    model=Evaluator(width).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001,foreach=False)
    best=None; best_score=math.inf; best_step=0; step=0; draws=0; last_loss=None
    groups=defaultdict(list)
    for index,row in enumerate(data["train"]): groups[row["stratum"]].append(index)
    strata=sorted(groups); tensor=tensors(data["train"],device,calibration)
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
        if on_event: on_event({"event":"training_started","identity":identity,"train_rows":len(data["train"]),"development_rows":len(data["development"]),"recovery":recovery})
        last_saved=step
        def commit():
            nonlocal best,best_score,best_step,last_saved
            if device=="cuda": torch.cuda.synchronize()
            prior.host_memory(); metrics=evaluate(model,data["development"],device,calibration)
            metrics["last_training_batch_loss"]=last_loss
            score=metrics["finite_choice_horizon"]["uniform_stratum_mae_seeds"]
            if (step%checkpoint_every==0 or step==steps) and score<best_score:
                best_score=score; best_step=step; best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            state={"identity":identity,"step":step,"draws":draws,"model":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                   "optimizer":optimizer.state_dict(),"sampler_rng":rng.getstate(),"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                   "cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else [],"best_model":best,"best_score":best_score,"best_step":best_step,
                   "scheduler":None,"amp_scaler":None,"pending_batches":[]}
            store.save(state,metrics,on_event); last_saved=step
        while step<steps:
            if cancel and cancel(): break
            indices=[rng.choice(groups[rng.choice(strata)]) for _ in range(batch_size)]
            index=torch.tensor(indices,dtype=torch.int64,device=device); batch={key:value[index] for key,value in tensor.items()}
            model.train(); optimizer.zero_grad(set_to_none=True)
            prediction,logits=model.both(batch["x"],batch["d"],batch["r"])
            value_loss=nn.functional.huber_loss(prediction,batch["y"],delta=1.0)
            action_loss=policy_loss(logits,batch["probability"],batch["legal"])
            bias_loss=ordinary_bias_loss(prediction-batch["y"],batch["actor_sign"],batch["ordinary_group"],len(calibration["eligible_strata"]))
            loss=value_loss+OBJECTIVES[variant]["policy_coefficient"]*action_loss+OBJECTIVES[variant]["bias_coefficient"]*bias_loss
            require(bool(torch.isfinite(loss)),"Nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); step+=1; draws+=batch_size
            last_loss={"value_huber_seeds":value_loss.item(),"policy_CE":action_loss.item(),"ordinary_mean_error_squared_seeds":bias_loss.item(),"total":loss.item()}
            if device=="cuda": require(torch.cuda.max_memory_reserved()<12*prior.GIB,"Training exceeded 12 GiB GPU reserve")
            if step%checkpoint_every==0 or step==steps or (cancel and cancel()): commit()
        if step>last_saved: commit()
        if best is None: return {"status":"interrupted","steps_completed":step,"model_available":False,"recovery":recovery,"sealed_final_test_consumed":False}
        model.load_state_dict(best); selected_metrics=evaluate(model,data["development"],device,calibration)
        value_export=export_model(model,Path(path)/"model.json",identity,best_step)
        policy_export=export_model(model,Path(path)/"policy-model.json",identity,best_step,True)
        result={"status":"complete" if step==steps else "interrupted","steps_completed":step,"best_step":best_step,"optimizer_steps_target":steps,"draws":draws,
                "identity":identity,"selected_development":selected_metrics,"weights_sha256":policy_export["model"]["weights_sha256"],"value_weights_sha256":value_export["model"]["weights_sha256"],
                "parameters":sum(value.numel() for value in model.parameters()),"value_inference_parameters":sum(value.numel() for value in value_state(model.state_dict()).values()),
                "elapsed_seconds_this_invocation":time.monotonic()-started,"peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else 0,"peak_host_working_set_bytes":prior.host_memory(),
                "recovery":recovery,"sealed_final_test_consumed":False,"promotion":"not promoted; independent ordinary and equal-time gates required"}
        atomic_json(Path(path)/"result.json",result); return result
    finally: store.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("operation",choices=("train","predict")); parser.add_argument("--run",default=str(DEFAULT_RUN))
    parser.add_argument("--dataset",default=str(campaign.DEFAULT)); parser.add_argument("--variant",choices=tuple(OBJECTIVES),default="C")
    parser.add_argument("--steps",type=int,default=8000); parser.add_argument("--seed",type=int,default=2718); parser.add_argument("--width",type=int,default=64)
    parser.add_argument("--batch-size",type=int,default=512); parser.add_argument("--checkpoint-every",type=int,default=100); parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args(); stopped=False
    def stop(*_):
        nonlocal stopped; stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,stop)
    def event(body): print(config.canonical(body),flush=True)
    try:
        if args.operation=="train": result=train(args.run,args.dataset,args.steps,args.seed,args.width,args.batch_size,device=args.device,checkpoint_every=args.checkpoint_every,on_event=event,cancel=lambda:stopped,variant=args.variant)
        else:
            value,policy=predict_export(json.loads(Path(args.run,"policy-model.json").read_text(encoding="utf-8")),json.loads(sys.stdin.readline()))
            result={"value_player0":value,"current_actor_policy_logits":policy,"scope":"experimental root H6 teacher preferences and value heuristic"}
        event({"event":"training_done","ok":True,"result":result}); return 0
    except Exception as exc:
        event({"event":"training_error","ok":False,"error":{"code":getattr(exc,"code","training_error"),"message":str(exc)}}); return 2

if __name__=="__main__": raise SystemExit(main())
