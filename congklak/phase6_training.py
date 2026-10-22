"""Frozen-value H8 policy warmstart and separately labeled selfplay update."""
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
from . import config,deeper_teacher_v2 as teacher,features_relay,training_v3 as prior
from . import neural_relay as native_model_binding
from .neural_relay import Model
from .native import ROOT,flatten

SCHEMA="phase6-policy-warmstart-and-selfplay-training-v1"
SOURCE_MODEL=ROOT/"data/training-stage5-deeper-h8-f127-seed2718/model.json"
SOURCE_WEIGHTS="9399891301cc6a98caa220dbb5d57efbd068b0014505d0a036ff670f371382dc"
SOURCE_VALUE_WEIGHTS="29f90512e59ebaeb93eedf74d247684161e6b211a9b21640dc4f9333d213d69f"
DEFAULT_WARM=ROOT/"data/training-phase6-warm-v1"
DEFAULT_UPDATE=ROOT/"data/training-phase6-update-v1"
PROTOCOL={"warm":{"steps":1000,"batch_size":256,"lr":.001,"seed":20261015,"checkpoint_every":50},
          "selfplay":{"steps":500,"batch_size":128,"lr":.0003,"seed":20261015,"checkpoint_every":50}}
TrainingError=prior.TrainingError
require=prior.require
sha=prior.sha
atomic_json=teacher.atomic_json
Checkpoints=prior.Checkpoints
weight_digest=prior.weight_digest
deterministic=prior.deterministic


class Evaluator(nn.Module):
    def __init__(self,stage="warm"):
        super().__init__(); require(stage in PROTOCOL,"Unknown training stage")
        self.input_count=127; self.width=64
        self.layers=nn.Sequential(nn.Linear(127,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh())
        self.policy=nn.Linear(64,7)
        for value in self.layers.parameters(): value.requires_grad_(stage=="selfplay")

    def both(self,x,d,r):
        h=self.layers[1](self.layers[0](x)); h=self.layers[3](self.layers[2](h))
        return d+r*self.layers[5](self.layers[4](h)).squeeze(-1),self.policy(h)

    def forward(self,x,d,r): return self.both(x,d,r)[0]


def value_state(state): return {k:v for k,v in state.items() if k.startswith("layers.")}


def policy_target(probability,legal):
    require(isinstance(probability,list) and len(probability)==7 and len(legal)==7 and all(type(x) is bool for x in legal) and any(legal),"Policy target has wrong dimensions/no legal action")
    require(all(type(p) in (int,float) and math.isfinite(p) and p>=0 for p in probability),"Policy target is nonfinite/negative")
    require(all(p==0 for p,valid in zip(probability,legal) if not valid) and abs(sum(probability)-1.)<=1e-12,"Policy mass is illegal or not normalized")
    return probability


def teacher_target(row,temperature=4.):
    require(type(temperature) in (int,float) and math.isfinite(temperature) and temperature>0,"Invalid policy temperature")
    p=config.position(row["state"]); n=features_relay._native()
    teacher.validate_label(row["label"],p,row["legal_mask"],8,n)
    normal=n.normalize(flatten(p),actor=p["controller"],total=98)
    require(not normal["terminal"],"Terminal policy training row")
    actor=normal["normalized_controller"]; sign=1 if actor==0 else -1
    actions=row["label"]["action_scores"]; best=max(sign*a["value_player0"] for a in actions)
    weights=[math.exp((sign*a["value_player0"]-best)/temperature) for a in actions]; total=sum(weights)
    probability=[0.]*7; legal=[False]*7; q=[0.]*7
    for action,weight in zip(actions,weights):
        pit=action["pit"]; probability[pit]=weight/total; legal[pit]=True; q[pit]=sign*action["value_player0"]
    return policy_target(probability,legal),legal,q,sign


def teacher_tensors(rows,device):
    targets=[teacher_target(row) for row in rows]
    return {"x":torch.tensor([r["cached_features"]["x"] for r in rows],dtype=torch.float32,device=device),
            "d":torch.tensor([r["cached_features"]["d"] for r in rows],dtype=torch.float32,device=device),
            "r":torch.tensor([r["cached_features"]["r"] for r in rows],dtype=torch.float32,device=device),
            "y":torch.tensor([r["label"]["target_player0"] for r in rows],dtype=torch.float32,device=device),
            "probability":torch.tensor([t[0] for t in targets],dtype=torch.float32,device=device),
            "legal":torch.tensor([t[1] for t in targets],dtype=torch.bool,device=device),
            "q":torch.tensor([t[2] for t in targets],dtype=torch.float32,device=device)}


def selfplay_tensors(rows,device):
    require(bool(rows),"No admitted selfplay training rows")
    features=[]; probabilities=[]; masks=[]; targets=[]
    for row in rows:
        require(row["labelkind"]=="selfplay_terminal_seed_margin","Unexpected selfplay value-label kind")
        require(isinstance(row["stratum"],str) and row["stratum"] and isinstance(row["gameid"],str) and row["gameid"],"Missing selfplay sampling provenance")
        p=config.position(row["state"]); x,d,r=features_relay.encode(p,127)
        normal=features_relay._native().normalize(flatten(p),actor=p["controller"],total=98)
        require(not normal["terminal"] and abs(r)>4 and normal["normalized_controller"]==p["controller"],"Selfplay row must be normalized/nonterminal/active>4")
        observed=row.get("features",row.get("cached_features"))
        require(isinstance(observed,dict) and isinstance(observed.get("x"),list) and len(observed["x"])==127 and all(type(v) in (int,float) and math.isfinite(v) for v in observed["x"]) and observed.get("x")==x and type(observed.get("d")) is int and observed["d"]==d and type(observed.get("r")) is int and observed["r"]==r,"Selfplay features differ from exact derivation")
        y=row["value_target_player0"]
        require(type(y) is int and y%2==0 and -98<=y<=98 and d-abs(r)<=y<=d+abs(r),"Selfplay target must be bounded terminal player0 SEEDS")
        legal=[i in normal["legal_actions"] for i in range(7)]
        probabilities.append(policy_target(row["policy_target"],legal)); masks.append(legal); targets.append(y); features.append((x,d,r))
    return {"x":torch.tensor([f[0] for f in features],dtype=torch.float32,device=device),
            "d":torch.tensor([f[1] for f in features],dtype=torch.float32,device=device),
            "r":torch.tensor([f[2] for f in features],dtype=torch.float32,device=device),
            "y":torch.tensor(targets,dtype=torch.float32,device=device),
            "probability":torch.tensor(probabilities,dtype=torch.float32,device=device),
            "legal":torch.tensor(masks,dtype=torch.bool,device=device)}


def policy_losses(logits,probability,legal):
    require(bool(legal.any(dim=1).all()),"All-masked policy row")
    logp=nn.functional.log_softmax(logits.masked_fill(~legal,-math.inf),dim=1).masked_fill(~legal,0.)
    return -(probability*logp).sum(dim=1)


def evaluate(model,rows,t):
    model.eval()
    with torch.no_grad():
        prediction,logits=model.both(t["x"],t["d"],t["r"])
        ce=policy_losses(logits,t["probability"],t["legal"]).cpu().tolist()
        chosen=logits.masked_fill(~t["legal"],-math.inf).argmax(dim=1)
        optimum=t["q"].masked_fill(~t["legal"],-math.inf).max(dim=1).values
        regret=(optimum-t["q"].gather(1,chosen[:,None]).squeeze(1)).cpu().tolist()
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(logits).all()),"Nonfinite development predictions")
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[row["stratum"]].append(i)
    strata={name:{"count":len(ids),"teacher_cross_entropy":sum(ce[i] for i in ids)/len(ids),
                  "teacher_H8_Q_regret_seeds":sum(regret[i] for i in ids)/len(ids),"teacher_H8_optimal_set_accuracy":sum(regret[i]==0 for i in ids)/len(ids)} for name,ids in groups.items()}
    return {"finite_choice_horizon":prior.regression_metrics(prediction,t["y"],t["d"],rows),
            "policy":{"count":len(rows),"teacher_cross_entropy":sum(ce)/len(ce),"uniform_stratum_teacher_cross_entropy":sum(s["teacher_cross_entropy"] for s in strata.values())/len(strata),
                      "mean_teacher_H8_Q_regret_seeds":sum(regret)/len(regret),"teacher_H8_optimal_set_accuracy":sum(v==0 for v in regret)/len(regret),"strata":strata,
                      "scope":"Root H8 choice + successor H7 soft preference, not a successor value target"}}


def export_payload(state,identity,step):
    return {"schema":"congklak-mlp-export-v4","encoder":features_relay.ENCODER127,"identity":identity,"step":step,
            "status":"experimental_phase6_candidate","weights":{k:v.detach().cpu().tolist() for k,v in state.items()},"weights_sha256":weight_digest(state),
            "terminal":"store0-minus-store1","output":"D_player0 + signed_active*tanh(current-actor-residual)","uses_oracle":False,
            "policy_output":"7 current-root-actor logits; legal actions masked by rules; terminal policy empty",
            "policy_training":"Root H8 softQ tau4 seeds" if identity["stage"]=="warm" else "Separate root H8 softQ and selfplay legal visit distributions",
            "value_training":"Frozen selected H8 relay value/trunk" if identity["stage"]=="warm" else "Separate H8 root finite-choice margins and terminal selfplay player0 seed margins"}


def export_model(model,path,identity,step):
    value=export_payload(model.state_dict(),identity,step); envelope={"model":value,"sha256":config.digest(value)}
    atomic_json(path,envelope); return envelope



def warm_completion(path,model):
    """Admit a complete warm export only with its committed catalog witness."""
    import sqlite3
    from contextlib import closing
    run=Path(path).resolve().parent; result=json.loads((run/"result.json").read_text(encoding="utf-8")); identity=model.identity
    require(identity.get("schema")==SCHEMA and identity.get("stage")=="warm" and identity.get("source_sha256")==sha(__file__),"Warm source/stage differs")
    require(result["identity"]==identity and result["status"]=="complete" and type(result["steps_completed"]) is int and result["steps_completed"]==result["optimizer_steps_target"]==identity["steps"],"Warm run is incomplete")
    require(result["weights_sha256"]==model.id and result["value_weights_sha256"]==SOURCE_VALUE_WEIGHTS and result["best_step"]==model.envelope["model"]["step"] and result["sealed_final_test_consumed"] is False,"Warm selected export differs")
    with closing(sqlite3.connect((run/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN"); body,digest=db.execute("SELECT body,hash FROM config").fetchone()
        require(config.digest(json.loads(body))==digest and json.loads(body)==identity,"Warm catalog identity differs")
        step,name,checksum,body,digest=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
        file=(run/name).resolve(); meta=json.loads(body)
        require(file.parent==run and file.is_file() and sha(file)==checksum and config.digest(meta)==digest,"Warm final catalog/file differs")
        state=torch.load(file,map_location="cpu",weights_only=True)
        require(state["identity"]==identity and type(state["step"]) is int and state["step"]==step==identity["steps"] and meta["weights_sha256"]==weight_digest(state["model"]),"Warm final checkpoint differs")
        require(state["best_step"]==result["best_step"] and state["best_model"] is not None and weight_digest(state["best_model"])==model.id and model.envelope["model"]==export_payload(state["best_model"],identity,state["best_step"]),"Warm selected checkpoint/export differs")
        require(weight_digest(value_state(state["model"]))==SOURCE_VALUE_WEIGHTS and weight_digest(value_state(state["best_model"]))==SOURCE_VALUE_WEIGHTS,"Warm frozen value/trunk differs")
        require(len(state["optimizer"]["state"])==2 and len(state["optimizer"]["param_groups"][0]["params"])==2,"Warm optimizer ownership differs")
        history=[]
        for checkpoint_step,body,digest in db.execute("SELECT step,body,hash FROM checkpoints ORDER BY id"):
            entry=json.loads(body); require(config.digest(entry)==digest,"Warm checkpoint metadata corrupt")
            if checkpoint_step%identity["checkpoint_every"]==0 or checkpoint_step==identity["steps"]:history.append((checkpoint_step,entry["metrics"]["policy"]["uniform_stratum_teacher_cross_entropy"]))
        require(history and min(history,key=lambda item:item[1])[0]==state["best_step"],"Warm selection differs from scheduled CE")
    return result


def draw(rng,groups,count):
    strata=sorted(groups)
    return [rng.choice(groups[rng.choice(strata)]) for _ in range(count)]


def train(path=DEFAULT_WARM,stage="warm",initial_model=SOURCE_MODEL,dataset=teacher.DEFAULT,selfplay_path=None,
          steps=None,batch_size=None,lr=None,seed=None,checkpoint_every=None,device="cuda",on_event=None,cancel=None):
    require(stage in PROTOCOL,"Unknown stage"); settings=PROTOCOL[stage]
    steps=settings["steps"] if steps is None else steps; batch_size=settings["batch_size"] if batch_size is None else batch_size
    lr=settings["lr"] if lr is None else lr; seed=settings["seed"] if seed is None else seed; checkpoint_every=settings["checkpoint_every"] if checkpoint_every is None else checkpoint_every
    config.integer(steps,"steps",1,100000); config.integer(batch_size,"batch_size",2,4096); config.integer(seed,"seed",0,2**31-1); config.integer(checkpoint_every,"checkpoint_every",1,10000)
    require(device in ("cuda","cpu") and type(lr) in (int,float) and math.isfinite(lr) and 0<lr<=.1,"Invalid training runtime/lr")
    require(stage!="selfplay" or batch_size%2==0,"Selfplay mixed batch must split equally")
    deterministic(device); initial=Model(initial_model)
    require(initial.width==64 and initial.input_count==127,"Phase6 architecture must be127/W64")
    require(stage!="warm" or initial.id==SOURCE_WEIGHTS,"Warmstart requires the fixed selected H8 relay model")
    warm_result=warm_completion(initial_model,initial) if stage=="selfplay" else None
    manifest,data,fingerprints=teacher.load(dataset,teacher_depth=8)
    sp_manifest=None; sp_rows=None; sp_module=None
    if stage=="selfplay":
        require(selfplay_path is not None,"Selfplay source is required")
        from . import phase6_selfplay as sp_module
        sp_manifest,sp_rows=sp_module.load_training_rows(selfplay_path)
        require(bool(sp_rows),"No admitted selfplay training rows")
        require(sp_manifest["identity"]["model_file_sha256"]==initial.file_sha256 and sp_manifest["identity"]["model_weights_sha256"]==initial.id,"Selfplay generator model differs from initial warm export")
    dependencies=(config,teacher,features_relay,prior,native_model_binding)
    identity={"schema":SCHEMA,"stage":stage,"rules_digest":config.RULES_DIGEST,"width":64,"input_count":127,"encoder":features_relay.ENCODER127,
              "initial_model_file_sha256":initial.file_sha256,"initial_weights_sha256":initial.id,"initial_value_weights_sha256":weight_digest({k:torch.tensor(v,dtype=torch.float32) for k,v in initial.weights.items() if k.startswith("layers.")}),
              "dataset_semantic_sha256":manifest["semantic_sha256"],"dataset_manifest_sha256":config.digest(manifest),"split_fingerprints":fingerprints,
              "selfplay_manifest_sha256":config.digest(sp_manifest) if sp_manifest is not None else None,"selfplay_rows_sha256":config.digest(sp_rows) if sp_rows is not None else None,
              "steps":steps,"batch_size":batch_size,"lr":lr,"seed":seed,"checkpoint_every":checkpoint_every,"device":device,
              "source_sha256":sha(__file__),"source_dependencies_sha256":{m.__name__:sha(m.__file__) for m in dependencies},
              "selfplay_loader_sha256":sha(sp_module.__file__) if sp_module else None,"torch_version":str(torch.__version__),"cuda_runtime":torch.version.cuda,"python_version":platform.python_version(),
              "device_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu","policy_temperature_seeds":4.,"policy_perspective":"Normalized root actor irrespective of successor controller",
              "trainable_parameters":455 if stage=="warm" else 12872,"value_targets":"H8 root finite-choice margins only" if stage=="warm" else "Separate H8 finite-choice root margins and terminal SP player0 SEEDS",
              "objective":"Legal H8 softQ cross entropy only; all six value/trunk tensors frozen" if stage=="warm" else ".5 teacherHuber + .5 selfplayHuber + .25*(.5 teacherCE + .5 visitCE)",
              "selection":"Scheduled checkpoint minimum uniform-development-stratum teacher CE" if stage=="warm" else "Fixed final optimizer step; development metrics descriptive only",
              "sampler":"Uniform teacher strata/P=q/unitweight" if stage=="warm" else "Exactly halfbatch uniform teacher strata, halfbatch uniform selfplay strata; independent root targets",
              "optimizer":"AdamW/fp32/weight_decay0.0001/grad_clip1/foreachFalse","scheduler":"none","amp":False,"loader_workers":0,"sealed_final_test_consumed":False,"oracle_training":False}
    started=time.monotonic(); rng=random.Random(seed); random.seed(seed); torch.manual_seed(seed)
    model=Evaluator(stage).to(device); model.load_state_dict({k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in initial.weights.items()})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.0001,foreach=False)
    groups=defaultdict(list)
    for i,row in enumerate(data["train"]):groups[row["stratum"]].append(i)
    sp_groups=defaultdict(list)
    if sp_rows:
        for i,row in enumerate(sp_rows): sp_groups[row["stratum"]].append(i)
    t=teacher_tensors(data["train"],device); dev=teacher_tensors(data["development"],device)
    s=selfplay_tensors(sp_rows,device) if sp_rows else None
    if warm_result is not None: require(warm_result["selected_development"]==evaluate(model,data["development"],dev),"Warm completion metrics differ")
    require(identity["initial_value_weights_sha256"]==SOURCE_VALUE_WEIGHTS,"Initial value/trunk differs from fixed H8 relay source")
    store=Checkpoints(path,identity)
    step=0; draws=0; teacher_draws=0; selfplay_draws=0; best=None; best_score=math.inf; best_step=0; last_loss=None
    try:
        saved,recovery=store.restore()
        if saved:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value): state[key]=value.cpu() if key=="step" else value.to(device)
            rng.setstate(saved["sampler_rng"]); random.setstate(saved["python_rng"]); torch.set_rng_state(saved["torch_rng"])
            if device=="cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
            step=saved["step"]; draws=saved["draws"]; teacher_draws=saved["teacher_draws"]; selfplay_draws=saved["selfplay_draws"]
            best=saved["best_model"]; best_score=saved["best_score"]; best_step=saved["best_step"]
        require(type(step) is int and 0<=step<=steps and type(draws) is int and draws==step*batch_size,"Committed step/draw counters differ")
        require(type(teacher_draws) is int and type(selfplay_draws) is int and teacher_draws==(draws if stage=="warm" else draws//2) and selfplay_draws==(0 if stage=="warm" else draws//2),"Committed mixed sampler counts differ")
        require(type(best_step) is int and 0<=best_step<=step and (best is None or best_step>0),"Committed selection step differs")
        if stage=="warm":
            require(weight_digest(value_state(model.state_dict()))==SOURCE_VALUE_WEIGHTS and (best is None or weight_digest(value_state(best))==SOURCE_VALUE_WEIGHTS),"Committed frozen value/trunk differs")
            require(best is None or best_step%checkpoint_every==0 or best_step==steps,"Off-cadence warm selection")
        else: require((best is None and step<steps) or (best is not None and step==best_step==steps),"Selfplay selection must be final step only")
        result_path=Path(path)/"result.json"
        completed=json.loads(result_path.read_text(encoding="utf-8")) if saved is not None and step==steps and result_path.is_file() else None
        if completed is not None and completed.get("status")=="complete":
            require(completed["identity"]==identity and completed["steps_completed"]==steps and completed["optimizer_steps_target"]==steps and completed["draws"]==draws and completed["teacher_draws"]==teacher_draws and completed["selfplay_draws"]==selfplay_draws and completed["best_step"]==best_step,"Completed state differs")
            require(best is not None and completed["weights_sha256"]==weight_digest(best) and completed["value_weights_sha256"]==weight_digest(value_state(best)),"Completed digests differ")
            require(completed["parameters"]==12872 and completed["trainable_parameters"]==identity["trainable_parameters"] and completed["sealed_final_test_consumed"] is False,"Completed architecture/ancestry differs")
            envelope=json.loads((Path(path)/"model.json").read_text(encoding="utf-8"))
            require(set(envelope)=={"model","sha256"} and config.digest(envelope["model"])==envelope["sha256"] and envelope["model"]==export_payload(best,identity,best_step),"Completed export differs")
            model.load_state_dict(best); require(completed["selected_development"]==evaluate(model,data["development"],dev),"Completed metrics differ")
            if on_event: on_event({"event":"training_completed_verified","identity":identity,"step":step,"best_step":best_step,"files_rewritten":False})
            return completed
        if completed is not None: require(completed.get("status")=="interrupted" and completed["identity"]==identity and 0<=completed["steps_completed"]<steps,"Corrupt earlier completion artifact")
        if on_event:on_event({"event":"training_started","identity":identity,"train_rows":len(data["train"]),"development_rows":len(data["development"]),"selfplay_rows":len(sp_rows) if sp_rows else 0,"recovery":recovery})
        last_saved=step
        def commit():
            nonlocal best,best_score,best_step,last_saved
            if device=="cuda":torch.cuda.synchronize()
            prior.host_memory(); metrics=evaluate(model,data["development"],dev); metrics["last_training_batch_loss"]=last_loss
            score=metrics["policy"]["uniform_stratum_teacher_cross_entropy"]
            if stage=="warm" and (step%checkpoint_every==0 or step==steps) and score<best_score:
                best_score=score; best_step=step; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if stage=="selfplay" and step==steps:
                best_score=0.; best_step=step; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            state={"identity":identity,"step":step,"draws":draws,"teacher_draws":teacher_draws,"selfplay_draws":selfplay_draws,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   "optimizer":optimizer.state_dict(),"sampler_rng":rng.getstate(),"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else [],
                   "best_model":best,"best_score":best_score,"best_step":best_step,"scheduler":None,"amp_scaler":None,"pending_batches":[]}
            store.save(state,metrics,on_event); last_saved=step
        while step<steps:
            if cancel and cancel():break
            count=batch_size if stage=="warm" else batch_size//2
            ids=torch.tensor(draw(rng,groups,count),dtype=torch.int64,device=device)
            model.train(); optimizer.zero_grad(set_to_none=True); prediction,logits=model.both(t["x"][ids],t["d"][ids],t["r"][ids])
            ce=policy_losses(logits,t["probability"][ids],t["legal"][ids]).mean()
            if stage=="warm":loss=ce
            else:
                sid=torch.tensor(draw(rng,sp_groups,count),dtype=torch.int64,device=device)
                sp_value,sp_logits=model.both(s["x"][sid],s["d"][sid],s["r"][sid])
                loss=.5*nn.functional.huber_loss(prediction,t["y"][ids],delta=1.)+.5*nn.functional.huber_loss(sp_value,s["y"][sid],delta=1.)+.25*(.5*ce+.5*policy_losses(sp_logits,s["probability"][sid],s["legal"][sid]).mean())
                selfplay_draws+=count
            require(bool(torch.isfinite(loss)),"Nonfinite training loss"); loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.); optimizer.step()
            step+=1; draws+=batch_size; teacher_draws+=count; last_loss=loss.item()
            if stage=="warm": require(weight_digest(value_state(model.state_dict()))==identity["initial_value_weights_sha256"],"Frozen warm value/trunk changed")
            if device=="cuda":require(torch.cuda.max_memory_reserved()<12*prior.GIB,"12GiB GPU reserve exceeded")
            if step%checkpoint_every==0 or step==steps or (cancel and cancel()):commit()
        if step>last_saved:commit()
        if best is None:return {"status":"interrupted","steps_completed":step,"model_available":False,"recovery":recovery,"sealed_final_test_consumed":False}
        model.load_state_dict(best); selected=evaluate(model,data["development"],dev)
        if stage=="warm": require(weight_digest(value_state(best))==identity["initial_value_weights_sha256"],"Warm selected value changed")
        if on_event:on_event({"event":"training_final_checkpoint_ready","stage":stage,"step":step,"best_step":best_step})
        export=export_model(model,Path(path)/"model.json",identity,best_step)
        result={"status":"complete" if step==steps else "interrupted","identity":identity,"steps_completed":step,"optimizer_steps_target":steps,"best_step":best_step,
                "draws":draws,"teacher_draws":teacher_draws,"selfplay_draws":selfplay_draws,"weights_sha256":export["model"]["weights_sha256"],"value_weights_sha256":weight_digest(value_state(best)),
                "parameters":12872,"trainable_parameters":identity["trainable_parameters"],"selected_development":selected,"elapsed_seconds_this_invocation":time.monotonic()-started,
                "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,"peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else 0,
                "peak_host_working_set_bytes":prior.host_memory(),"recovery":recovery,"sealed_final_test_consumed":False,"promotion":"Experimental; independent equal-time gates required"}
        atomic_json(result_path,result); return result
    finally:store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=tuple(PROTOCOL));parser.add_argument("--run");parser.add_argument("--model",default=str(SOURCE_MODEL));parser.add_argument("--dataset",default=str(teacher.DEFAULT));parser.add_argument("--selfplay")
    for name in ("steps","batch-size","seed","checkpoint-every"):parser.add_argument("--"+name,type=int)
    parser.add_argument("--lr",type=float);parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,"SIGTERM"):signal.signal(signal.SIGTERM,stop)
    def event(value):print(config.canonical(value),flush=True)
    try:
        result=train(args.run or (DEFAULT_WARM if args.stage=="warm" else DEFAULT_UPDATE),args.stage,args.model,args.dataset,args.selfplay,args.steps,args.batch_size,args.lr,args.seed,args.checkpoint_every,args.device,event,lambda:stopped)
        event({"event":"training_done","ok":True,"result":result});return 0
    except Exception as exc:event({"event":"training_error","ok":False,"error":{"message":str(exc)}});return 2


if __name__=="__main__":raise SystemExit(main())
