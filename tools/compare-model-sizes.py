"""Development-only comparison of frozen 64/128/256-wide pilot models."""
from __future__ import annotations
from collections import defaultdict
from contextlib import closing
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import queue
import sqlite3
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,training as tr
from congklak.native import Native,flatten
import torch

def event(value): print(config.canonical(value),flush=True)

def path_for(width,seed):
    name=f"training-pilot-seed{seed}" if width==64 else f"training-pilot-w{width}-seed{seed}"
    return ROOT/"data"/name

class Reference:
    """Validate once, then decode each board with scalar Python arithmetic."""
    def __init__(self,envelope):
        value=envelope["model"]
        tr.require(config.digest(value)==envelope["sha256"],"Export checksum differs")
        tr.require(value["encoder"]==tr.ENCODER and not value["uses_oracle"],"Unsupported export")
        self.weights=value["weights"]

    def predict(self,encoded):
        x,d,r=encoded
        for prefix in ("layers.0","layers.2","layers.4"):
            x=[sum(v*a for v,a in zip(row,x))+b for row,b in zip(self.weights[prefix+".weight"],self.weights[prefix+".bias"])]
            if prefix!="layers.4": x=[max(v,0) for v in x]
        return d+r*math.tanh(x[0])

def choice_metrics(rows,edges,predicted):
    buckets=defaultdict(list)
    strata=defaultdict(list)
    for row,children in zip(rows,edges):
        if not children: continue
        p=row["state"]; target=row["label"]; sign=1 if p["controller"]==0 else -1
        best=max if sign==1 else min
        choice=best(children,key=lambda e:predicted[e[1]])[0]
        baseline=best(children,key=lambda e:e[2])[0]
        q={a["pit"]:a["value_player0"] for a in target["action_scores"]}
        record=(choice in target["best_moves"],baseline in target["best_moves"],
                sign*(target["target_player0"]-q[choice]),sign*(target["target_player0"]-q[baseline]))
        tr.require(record[2]>=0 and record[3]>=0,"Negative reference move regret")
        buckets[target["kind"]].append(record); strata[row["stratum"]].append(record)
    def summarize(values):
        return {"positions":len(values),"candidate_optimal_set_accuracy":statistics.mean(v[0] for v in values),
                "H_v1_oneply_optimal_set_accuracy":statistics.mean(v[1] for v in values),
                "candidate_mean_move_regret_seeds":statistics.mean(v[2] for v in values),
                "H_v1_oneply_mean_move_regret_seeds":statistics.mean(v[3] for v in values)}
    return {k:summarize(v) for k,v in buckets.items()},{k:summarize(v) for k,v in strata.items()}

def cpu_latency(model,rows):
    x,d,r,_=tr.tensors(rows[:256],"cpu")
    trials=[]
    with torch.inference_mode():
        for i in range(256): model(x[i:i+1],d[i:i+1],r[i:i+1])
        for trial in range(5):
            started=time.perf_counter()
            for repeat in range(4):
                for i in range(256): model(x[i:i+1],d[i:i+1],r[i:i+1])
            trials.append((time.perf_counter()-started)*1e6/1024)
    return {"median_microseconds_per_position":statistics.median(trials),"trials_microseconds_per_position":trials,
            "cpu_threads":1,"batch_size":1,"positions_per_trial":1024,"scope":"PyTorch forward plus tensor slicing; preencoded inputs; no simulator/search/encoder"}

def latest(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:
        name,checksum=db.execute("SELECT name,sha FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()
    tr.require(tr.sha(path/name)==checksum,"Damaged recovery checkpoint")
    return torch.load(path/name,map_location="cpu",weights_only=True)

def equal_tree(a,b):
    if torch.is_tensor(a): tr.require(torch.is_tensor(b) and torch.equal(a,b),"Recovery tensor differs")
    elif isinstance(a,dict):
        tr.require(isinstance(b,dict) and a.keys()==b.keys(),"Recovery dictionary differs")
        for key in a: equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        tr.require(type(a)==type(b) and len(a)==len(b),"Recovery sequence differs")
        for x,y in zip(a,b): equal_tree(x,y)
    else: tr.require(a==b,"Recovery scalar differs")

def recovery_check():
    path=ROOT/"build"/"large-model-recovery"
    proof=path/"recovery.json"
    fresh=path_for(256,2718)
    if proof.is_file():
        evidence=json.loads(proof.read_text())
        tr.require(evidence["trainer_sha256"]==tr.sha(tr.__file__),"Recovery evidence source differs")
        equal_tree(latest(path),latest(fresh))
        return evidence
    tr.require(not (path/"catalog.sqlite3").exists(),"Incomplete recovery experiment already exists; resume it separately before repeating")
    worker='''import json,sys
from congklak import training as tr
def event(e):
    print(json.dumps(e),flush=True)
    if e.get("event")=="checkpoint_file_ready" and e["step"]==700:
        sys.stdin.readline()
tr.train(sys.argv[1],steps=2000,seed=2718,width=256,batch_size=512,device="cuda",checkpoint_every=100,on_event=event)
'''
    child=subprocess.Popen([sys.executable,"-c",worker,str(path)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8")
    messages=queue.Queue()
    def reader():
        for line in child.stdout: messages.put(json.loads(line))
    thread=threading.Thread(target=reader,daemon=True); thread.start()
    try:
        while True:
            e=messages.get(timeout=60)
            if e.get("event")=="checkpoint_file_ready" and e["step"]==700: break
            if e.get("event")=="training_error": raise RuntimeError(str(e))
        child.kill(); child.wait(timeout=20)
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=20)
        thread.join(timeout=10); child.stdin.close(); child.stdout.close(); child.stderr.close()
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:
        committed=db.execute("SELECT max(step) FROM checkpoints").fetchone()[0]
    tr.require(committed==600,"Kill did not precede step-700 catalog commit")
    resumed=tr.train(path,steps=2000,seed=2718,width=256,batch_size=512,device="cuda",checkpoint_every=100)
    tr.require(resumed["recovery"]["restored_step"]==600,"Did not resume the previous committed checkpoint")
    equal_tree(latest(path),latest(fresh))
    expected=json.loads((fresh/"result.json").read_text())
    tr.require(resumed["weights_sha256"]==expected["weights_sha256"],"Selected recovery model differs")
    evidence={"passed":True,"width":256,"parameters":resumed["parameters"],"kill_step":700,"restored_step":600,"completed_step":2000,
              "bitwise_model_optimizer_rng_and_selection_equal":True,"weights_sha256":resumed["weights_sha256"],
              "trainer_sha256":tr.sha(tr.__file__),"power_loss_tested":False}
    tr.atomic_json(proof,evidence); event({"event":"large_model_recovery_checked",**evidence})
    return evidence

def main():
    tr.deterministic("cpu")
    manifest,data,fingerprints=tr.load_data()
    rows=data["development"]; native=Native()
    children=[]; edges=[]
    for row in rows:
        actions=[]; p=row["state"]
        for action in row["label"]["action_scores"]:
            pit=action["pit"]
            child=native.apply(flatten(p),actor=p["controller"],action=pit,total=98,trace=False)["position"]
            actions.append((pit,len(children),child["stores"][0]-child["stores"][1]))
            children.append(child)
        edges.append(actions)
    root_encoded=[tr.encode(r["state"]) for r in rows]
    child_encoded=[tr.encode(p) for p in children]
    event({"event":"comparison_started","development_rows":len(rows),"successors":len(children),"sealed_final_test_consumed":False})
    results=[]; shared=None
    for width in (64,128,256):
        for seed in (2718,31415):
            path=path_for(width,seed)
            trained=json.loads((path/"result.json").read_text())
            export=json.loads((path/"model.json").read_text())
            identity=trained["identity"]
            tr.require(trained["status"]=="complete" and trained["steps_completed"]==2000,"Incomplete comparison model")
            tr.require(identity==export["model"]["identity"] and identity["width"]==width and identity["seed"]==seed,"Comparison model identity differs")
            tr.require(identity["source_sha256"]==tr.sha(tr.__file__) and identity["dataset_semantic_sha256"]==manifest["semantic_sha256"],"Trainer or dataset changed")
            tr.require(identity["dataset_manifest_sha256"]==config.digest(manifest) and identity["split_fingerprints"]==fingerprints,"Dataset generation or split changed")
            tr.require(not trained["sealed_final_test_consumed"],"Sealed test exposure")
            common={k:v for k,v in identity.items() if k not in ("width","seed")}
            if shared is None: shared=common
            tr.require(common==shared,"Unmatched training configuration")
            reference=Reference(export)
            model=tr.Evaluator(width)
            model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in reference.weights.items()}); model.eval()
            digest=tr.weight_digest(model.state_dict())
            tr.require(digest==trained["weights_sha256"]==export["model"]["weights_sha256"],"Export weight digest differs")
            predicted=[reference.predict(e) for e in root_encoded]
            with torch.inference_mode():
                x,d,r,_=tr.tensors(rows,"cpu"); torch_values=model(x,d,r).tolist()
            difference=max(abs(a-b) for a,b in zip(predicted,torch_values))
            tr.require(difference<1e-4,"Large model CPU export parity failed")
            for value,(_,d,r) in zip(predicted,root_encoded):
                tr.require(math.isfinite(value) and d-abs(r)<=value<=d+abs(r),"Export score outside physical bounds")
            successor_values=[reference.predict(e) for e in child_encoded]
            choices,choice_strata=choice_metrics(rows,edges,successor_values)
            if width==64:
                previous=json.loads((path/"assessment.json").read_text())
                tr.require(choices==previous["oneply_choice_metrics"],"Reference decoder disagrees with existing baseline")
            development=tr.evaluate(model,rows,"cpu")
            train_metrics=tr.evaluate(model,data["train"],"cpu")
            result={"width":width,"seed":seed,"parameters":trained["parameters"],"directory":str(path.relative_to(ROOT)),
                    "best_step":trained["best_step"],"weights_sha256":digest,"train":train_metrics,"development":development,
                    "oneply_choice_metrics":choices,"oneply_strata":choice_strata,"export_cpu_max_difference_seeds":difference,
                    "training_seconds_excluding_data_loading":trained["elapsed_seconds_this_invocation"],
                    "peak_gpu_reserved_bytes":trained["peak_gpu_reserved_bytes"],"cpu_latency":cpu_latency(model,rows),
                    "physical_bytes":sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
                    "artifact_sha256":{name:tr.sha(path/name) for name in ("model.json","result.json")}}
            tr.atomic_json(path/"size-assessment.json",result); results.append(result)
            event({"event":"model_compared","width":width,"seed":seed,"parameters":result["parameters"],
                   "mae":development["finite_choice_horizon"]["mae_seeds"],"macro_mae":development["finite_choice_horizon"]["uniform_stratum_mae_seeds"],
                   "choice_agreement":choices["finite_choice_horizon"]["candidate_optimal_set_accuracy"],"cpu_us":result["cpu_latency"]["median_microseconds_per_position"]})
    recovery=recovery_check()
    report={"schema":"congklak-model-size-experiment-v1","executed_at_utc":datetime.now(timezone.utc).isoformat(),
            "dataset_semantic_sha256":manifest["semantic_sha256"],"native_build_sha256":native.build_id,
            "trainer_sha256":tr.sha(tr.__file__),"comparison_source_sha256":tr.sha(__file__),"matched_configuration":shared,
            "widths":[64,128,256],"seeds":[2718,31415],"train_rows":len(data["train"]),"development_rows":len(rows),
            "development_successors":len(children),"sealed_final_test_consumed":False,"promotion":"not_promoted",
            "budget":"matched optimizer steps and sampled examples, not matched GPU time or energy",
            "search":"one legal choice then raw heuristic; exact/horizon references separate; no oracle prediction",
            "energy_measured":False,"results":results,"large_model_recovery":recovery}
    tr.atomic_json(ROOT/"build"/"model-size-results.json",report)
    event({"event":"comparison_complete","models":len(results),"recovery_passed":recovery["passed"],"sealed_final_test_consumed":False})

if __name__=="__main__": main()
