"""Bounded synchronous policy/value inference for independent Phase 6 trees.

Features are produced by the frozen native relay encoder. No dataset, oracle,
evaluation cache, or persistent GPU queue is consulted. Calls are serialized;
the whole-call clock includes lock waiting, features, transfers and synchronization.
Frozen feature extraction and an already launched CUDA kernel cannot be preempted;
cancellation is checked between positions and before/after each launched batch.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import threading
import time
from . import config
from .native import NativeError,flatten
from .neural_relay import Model,NeuralNative,sha

VALUE_TOLERANCE_SEEDS=1e-4
LOGIT_TOLERANCE=1e-5
PROTOCOL="phase6-frozen-relay127/raw-player0/bounded-synchronous-batch-v1"

class InferenceStopped(RuntimeError):
    def __init__(self,reason,work=None):
        super().__init__(reason);self.reason=reason;self.work=work or {}

def _cancelled(cancel):
    if cancel is None:return False
    return bool(cancel() if callable(cancel) else cancel.is_set())

def _poll(cancel,deadline):
    if _cancelled(cancel):raise InferenceStopped("cancelled")
    if deadline is not None and time.perf_counter()>=deadline:raise InferenceStopped("time_budget")

def validate_prediction(row,position):
    if not isinstance(row,dict):raise NativeError("Inference row must be a dictionary")
    value=row.get("value_player0");logits=row.get("policy_logits")
    if type(value) not in (int,float) or not math.isfinite(value):raise NativeError("Nonfinite inference value")
    if not isinstance(logits,list) or len(logits)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits):raise NativeError("Seven finite policy logits required")
    d=position["stores"][0]-position["stores"][1];r=sum(map(sum,position["pits"]))
    if not d-r-VALUE_TOLERANCE_SEEDS<=value<=d+r+VALUE_TOLERANCE_SEEDS:raise NativeError("Inference value outside physical seed bounds")
    if not r and (value!=d or any(logits)):raise NativeError("Terminal inference must be exact and skip policy")
    for key in ("feature_simulations","feature_deposits"):
        if type(row.get(key)) is not int or row[key]<0:raise NativeError("Invalid feature-work counter")
    return row

class BatchEvaluator:
    supports_cancel=True
    def __init__(self,modelpath,backend="native",max_batch=8):
        if backend not in ("native","cuda"):raise ValueError("Backend must be native or cuda")
        config.integer(max_batch,"max_batch",1,32)
        started=time.perf_counter();self.model=modelpath if isinstance(modelpath,Model) else Model(Path(modelpath))
        if self.model.input_count!=127 or self.model.width!=64:raise NativeError("Phase6 pilot requires full v4 relay127/width64 export")
        self.backend=backend;self.max_batch=max_batch;self.native=NeuralNative(self.model,mode="value-none")
        self.lock=threading.RLock();self.closed=False;self.last_metrics={};self.torch=None
        self.identity={"protocol":PROTOCOL,"backend":backend,"max_batch":max_batch,"model_file_sha256":self.model.file_sha256,"model_weights_sha256":self.model.id,
                       "encoder":self.model.encoder,"rules_digest":config.RULES_DIGEST,"native_build_sha256":self.native.build_id,
                       "source_sha256":sha(__file__),"binding_sha256":sha(Path(__file__).with_name("neural_relay.py")),
                       "feature_sha256":sha(Path(__file__).with_name("features_relay.py")),"value_tolerance_seeds":VALUE_TOLERANCE_SEEDS,"logit_tolerance":LOGIT_TOLERANCE,
                       "evaluation_cache":False,"numeric_policy":"native-double/cuda-FP32-no-TF32"}
        try:
            if backend=="cuda":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
                if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8",":16:8"):raise RuntimeError("Deterministic CUDA requires a declared CUBLAS workspace configuration")
                import torch
                if not torch.cuda.is_available():raise RuntimeError("CUDA PyTorch is unavailable in this interpreter")
                self.torch=torch
                torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                torch.use_deterministic_algorithms(True)
                self.tensors={k:torch.tensor(v,dtype=torch.float32,device="cuda") for k,v in self.model.weights.items()}
                torch.cuda.synchronize();self.identity.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),deterministic_algorithms=True,cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"])
                if torch.cuda.memory_allocated()>12*1024**3:raise RuntimeError("Phase6 process CUDA allocation exceeds12GiB")
            self.startup_ms=(time.perf_counter()-started)*1000
        except BaseException:
            self.native.close();raise

    def close(self):
        with self.lock:
            if not self.closed:
                self.native.close();self.closed=True
                if self.torch is not None:self.tensors.clear()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def evaluate_many(self,positions,cancel=None,deadline=None):
        started=time.perf_counter();config.integer(len(positions),"batch_size",1,self.max_batch)
        rows=[config.position(p) for p in positions];_poll(cancel,deadline)
        metrics={"batch_size":len(rows),"cpu_features_ms":0.0,"queue_wait_ms":0.0,"host_to_device_ms":0.0,"gpu_ms":0.0,"device_to_host_ms":0.0,
                 "native_forward_ms":0.0,"feature_simulations":0,"feature_deposits":0,"evaluated_positions":0,"whole_call_ms":0.0}
        metrics["per_position"]=[{"feature_simulations":0,"feature_deposits":0,"evaluated":False} for _ in rows]
        acquired=False
        try:
            # Bounded polling also permits cancellation while another call owns
            # this evaluator. There is no background batch-fill wait.
            wait=time.perf_counter()
            while not self.lock.acquire(timeout=.005):_poll(cancel,deadline)
            acquired=True;metrics["queue_wait_ms"]=(time.perf_counter()-wait)*1000
            if self.closed:raise NativeError("Evaluator is closed")
            _poll(cancel,deadline);outputs=[None]*len(rows)
            for i,p in enumerate(rows):
                if not sum(map(sum,p["pits"])):
                    outputs[i]={"value_player0":p["stores"][0]-p["stores"][1],"policy_logits":[0.0]*7,"feature_simulations":0,"feature_deposits":0}
            active=[i for i,x in enumerate(outputs) if x is None]
            if self.backend=="native":
                for i in active:
                    _poll(cancel,deadline);tick=time.perf_counter();features=self.native.feature_vector(rows[i]);metrics["cpu_features_ms"]+=(time.perf_counter()-tick)*1000
                    # Frozen evaluate also extracts features. Account both
                    # passes honestly; its model value is the unquantized head.
                    for key in ("feature_simulations","feature_deposits"):
                        metrics[key]+=features[key];metrics["per_position"][i][key]+=features[key]
                    _poll(cancel,deadline);tick=time.perf_counter();value=self.native.evaluate(rows[i]);metrics["native_forward_ms"]+=(time.perf_counter()-tick)*1000
                    for key in ("feature_simulations","feature_deposits"):
                        metrics[key]+=features[key];metrics["per_position"][i][key]+=features[key]
                    outputs[i]={"value_player0":value["model_value_player0"],"policy_logits":value["policy_logits"],
                                "feature_simulations":2*features["feature_simulations"],"feature_deposits":2*features["feature_deposits"]}
                    metrics["evaluated_positions"]+=1;metrics["per_position"][i]["evaluated"]=True
            elif active:
                torch=self.torch;vectors=[];offsets=[];signed=[];features_by_row=[]
                for i in active:
                    _poll(cancel,deadline);tick=time.perf_counter();feature=self.native.feature_vector(rows[i]);metrics["cpu_features_ms"]+=(time.perf_counter()-tick)*1000
                    vectors.append(feature["features"]);features_by_row.append(feature)
                    for key in ("feature_simulations","feature_deposits"):
                        metrics[key]+=feature[key];metrics["per_position"][i][key]+=feature[key]
                    normal=self.native.normalize(flatten(rows[i]),actor=rows[i]["controller"],total=98)["position"]
                    offsets.append(normal["stores"][0]-normal["stores"][1]);signed.append(sum(map(sum,normal["pits"]))* (1 if normal["controller"]==0 else -1))
                _poll(cancel,deadline)
                with torch.inference_mode():
                    tick=time.perf_counter();x=torch.tensor(vectors,dtype=torch.float32,device="cuda");d=torch.tensor(offsets,dtype=torch.float32,device="cuda");r=torch.tensor(signed,dtype=torch.float32,device="cuda")
                    torch.cuda.synchronize();metrics["host_to_device_ms"]=(time.perf_counter()-tick)*1000;_poll(cancel,deadline)
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                    for prefix in ("layers.0","layers.2"):
                        x=torch.relu(torch.nn.functional.linear(x,self.tensors[prefix+".weight"],self.tensors[prefix+".bias"]))
                    value=d+r*torch.tanh(torch.nn.functional.linear(x,self.tensors["layers.4.weight"],self.tensors["layers.4.bias"]).squeeze(1))
                    logits=torch.nn.functional.linear(x,self.tensors["policy.weight"],self.tensors["policy.bias"]);end.record();end.synchronize();metrics["gpu_ms"]=begin.elapsed_time(end)
                    metrics["evaluated_positions"]+=len(active)
                    for i in active:metrics["per_position"][i]["evaluated"]=True
                    _poll(cancel,deadline);tick=time.perf_counter();values=value.cpu().tolist();policy=logits.cpu().tolist();torch.cuda.synchronize();metrics["device_to_host_ms"]=(time.perf_counter()-tick)*1000
                for j,i in enumerate(active):
                    feature=features_by_row[j];outputs[i]={"value_player0":values[j],"policy_logits":policy[j],"feature_simulations":feature["feature_simulations"],"feature_deposits":feature["feature_deposits"]}
            for row,p in zip(outputs,rows):validate_prediction(row,p)
            _poll(cancel,deadline);metrics["whole_call_ms"]=(time.perf_counter()-started)*1000
            # Apportion shared synchronous cost to leaves for independent work
            # accounting; whole-call metrics remain authoritative for latency.
            for row in outputs:row["inference_ms"]=metrics["whole_call_ms"]/len(rows)
            self.last_metrics=metrics;return outputs
        except InferenceStopped as exc:
            metrics["whole_call_ms"]=(time.perf_counter()-started)*1000;self.last_metrics=metrics;exc.work=metrics;raise
        finally:
            if acquired:self.lock.release()
