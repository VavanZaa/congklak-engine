"""Small CPU value-head pilot with one sampling layer and immutable policy."""
from collections import defaultdict
from contextlib import closing
import bisect
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import struct
import sys
import time
from . import config, features_relay, value_learning as old
from .native import ROOT
from .neural_relay import Model,flatten_weights

SCHEMA="reviewed-weighted-small-fixed-policy-value-pilot-v1"
SOURCE,SOURCE_SHA,SOURCE_WEIGHTS=old.SOURCE,old.SOURCE_SHA,old.SOURCE_WEIGHTS
FROZEN_WEIGHTS,TRAINABLE=old.FROZEN_WEIGHTS,old.TRAINABLE
DEFAULT=ROOT/"data/value-pilot-v1"
BASELINE=ROOT/"build/value-pilot-baseline.json"
DATA=ROOT/"build/value-pilot-data.json"
ROWS=ROOT/"build/value-pilot-rows.json"
PLAN=ROOT/"build/value-pilot-training-plan.json"
IMPLEMENTATION=ROOT/"build/value-pilot-training-implementation.json"
WITNESS=ROOT/"build/value-pilot-model-witness.json"
SOURCES=("congklak/value_pilot_learning.py","tools/value-pilot-train.py","tests/value_pilot_learning_acceptance.py","docs/VALUE_PILOT_SCOPE.md")
DEPENDENCIES=("congklak/value_pilot_data.py","congklak/value_pilot_benchmark.py","tools/benchmark-value-pilot.py",
              "congklak/value_learning.py","congklak/training_v3.py","congklak/features_relay.py","congklak/neural_relay.py","congklak/config.py",
              "build/training-runtime.json","build/value-pilot-baseline.json","build/value-pilot-data.json","build/value-pilot-rows.json",
              "build/value-pilot-benchmark-plan.json","build/value-pilot-benchmark-implementation.json")
CONTRACT={"steps":200,"batch_size":256,"checkpoint_every":20,"seed":20261022,"lr":.0003,
          "input_count":127,"width":64,"parameters":12872,"trainable_parameters":65,"frozen_parameters":12807,
          "value_huber_delta":4.,"policy_loss_weight":0.,"device":"cpu","cpu_threads":1,"amp":False,"loader_workers":0,
          "optimizer":"AdamW/train65only/fp32/weight_decay0/grad_clip1/foreachFalse",
          "sampler":"Sequential Python-float CDF in frozen row order; bisect_right(cdf,rng.random()*cdf[-1]); one random per draw",
          "loss_weighting":"Unit per-draw seed-space Huber; sampling_probability is the only weighting layer",
          "hidden_cache_policy":"Pinned source frozen trunk forward once, FP32 CPU; raw hidden tensors and input/row identities hashed",
          "selection":"Fixed final optimizer step; descriptive metrics never select a checkpoint",
          "oracle_training":False,"sealed_final_test_consumed":False,"promotion":False}
IDENTITY_FIELDS=set(CONTRACT)|{"schema","fixture","rules_digest","encoder","source_sha256","tool_sha256","source_dependencies",
                "initial_model_file_sha256","initial_weights_sha256","initial_frozen_weights_sha256","data_manifest_sha256","rows_file_sha256",
                "admitted_manifest_sha256","probability_row_order_sha256","hidden_cache_sha256","hidden_cache","pretraining_witness",
                "plan_file_sha256","implementation_file_sha256","torch_version","python_version"}
RESULT_FIELDS={"schema","status","identity","steps_completed","selected_step","draws","weights_sha256","train_rows","development_rows",
               "initial_development","development","initial_descriptive","descriptive","frozen_weights_sha256","frozen_policy_unchanged","hidden_cache_sha256",
               "elapsed_seconds_this_invocation","peak_host_working_set_bytes","recovery","training_performed","sealed_final_test_consumed","promotion"}
require,sha,publish=old.require,old.sha,old.publish
runtime,evaluator,frozen_digest,model_frozen_digest=old.runtime,old.evaluator,old.frozen_digest,old.model_frozen_digest


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline():
    value=read(BASELINE)
    require(value["schema"]=="value-pilot-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==2426 and value["prior_accepted_inputs"]==2424 and value["fixed_model_sha256"]==sha(SOURCE)==SOURCE_SHA,"Pilot prior baseline/model differs")
    for name,checksum in value["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==checksum,"Prior accepted input changed: "+name)
    return value


def prepare():
    """Root-owned, CPU-only reviewed target and probability admission."""
    from . import value_pilot_data
    manifest,groups=value_pilot_data.prepare()
    require(set(groups)=={"train","development","ordinary_development","diagnostic"} and all(groups.values()),"Pilot splits differ")
    for split,rows in groups.items():
        require(len({row["state_key"] for row in rows})==len(rows),"Duplicate pilot row")
        for row in rows:target(row)
    cdf(groups["train"])
    require(sha(SOURCE)==SOURCE_SHA and Model(SOURCE).id==SOURCE_WEIGHTS and model_frozen_digest(Model(SOURCE))==FROZEN_WEIGHTS,"Pilot initial model differs")
    return manifest,groups


def target(row):
    p=config.position(row["state"]);depth=row["teacher_depth"]
    require(type(depth) is int and depth in (8,10,12,14),"Unknown pilot teacher horizon")
    require(type(row["loss_weight"]) in (int,float) and row["loss_weight"]==1.,"Pilot must use unit per-draw loss weights")
    old.teacher.validate_label(row["label"],p,row["legal_mask"],depth,features_relay._native())
    feature=row["cached_features"];d,r=feature["d"],feature["r"];active=sum(map(sum,p["pits"]));sign=1 if p["controller"]==0 else -1
    require(type(d) is int and d==p["stores"][0]-p["stores"][1] and type(r) is int and r==sign*active and active>4,"Pilot signed-active feature scaling differs")
    require(len(feature["x"])==127 and all(type(v) in (int,float) and math.isfinite(v) for v in feature["x"]),"Pilot cached feature shape/values differ")
    y=row["label"]["target_player0"];z=(y-d)/r
    require(math.isfinite(z) and -1<=z<=1 and abs(d+r*z-y)<1e-12,"Pilot root target residual differs")
    return {"d":d,"r":r,"y":y,"z":z}


def cdf(rows):
    values=[];total=0.
    for row in rows:
        probability=row["sampling_probability"]
        require(type(probability) in (int,float) and math.isfinite(probability) and probability>0,"Invalid pilot sampling probability")
        total+=probability;values.append(total)
    require(values and abs(total-1.)<1e-12 and all(a<b for a,b in zip([0.]+values[:-1],values)),"Pilot CDF loses mass/support")
    return values


def sample_ids(rng,cumulative,count):
    return [bisect.bisect_right(cumulative,rng.random()*cumulative[-1]) for _ in range(count)]


def plan():
    baseline();manifest,_=prepare()
    value={"schema":SCHEMA,"contract":CONTRACT,"rules_digest":config.RULES_DIGEST,"source_model_sha256":SOURCE_SHA,
           "data_manifest_sha256":sha(DATA),"rows_file_sha256":sha(ROWS),"reviewed_manifest_sha256":config.digest(manifest),"baseline_sha256":sha(BASELINE),
           "pretraining_requirement":"Strict completed48-game control benchmark witness before Torch import or optimizer work"}
    publish(PLAN,value);return value


def freeze():
    plan();value={"schema":"value-pilot-training-implementation-v1","plan_sha256":sha(PLAN),
                  "sources":{name:sha(ROOT/name) for name in SOURCES},"dependencies":{name:sha(ROOT/name) for name in DEPENDENCIES}}
    publish(IMPLEMENTATION,value);return value


def admit():
    baseline();value,implementation=read(PLAN),read(IMPLEMENTATION)
    require(value["schema"]==SCHEMA and value["contract"]==CONTRACT and value["rules_digest"]==config.RULES_DIGEST and value["source_model_sha256"]==SOURCE_SHA and value["data_manifest_sha256"]==sha(DATA) and value["rows_file_sha256"]==sha(ROWS) and value["baseline_sha256"]==sha(BASELINE),"Pilot training plan differs")
    require(implementation=={"schema":"value-pilot-training-implementation-v1","plan_sha256":sha(PLAN),"sources":{name:sha(ROOT/name) for name in SOURCES},"dependencies":{name:sha(ROOT/name) for name in DEPENDENCIES}},"Pilot implementation/dependencies changed")
    return value,implementation


def settings_for(fixture=False,steps=None,checkpoint_every=None,batch_size=None):
    value=dict(CONTRACT)
    if fixture:
        value.update(steps=6 if steps is None else config.integer(steps,"fixture_steps",1,12),checkpoint_every=2 if checkpoint_every is None else config.integer(checkpoint_every,"fixture_cadence",1,12),batch_size=32 if batch_size is None else config.integer(batch_size,"fixture_batch",2,256))
    else:require(steps is None and checkpoint_every is None and batch_size is None,"Pilot production budget differs")
    return value


def tensors(rows):
    torch,_=runtime();targets=[target(row) for row in rows]
    return {"x":torch.tensor([row["cached_features"]["x"] for row in rows],dtype=torch.float32),
            **{name:torch.tensor([value[name] for value in targets],dtype=torch.float32) for name in ("d","r","y","z")}}


def hidden_cache(model,groups,data):
    torch,prior=runtime();hidden={};policy={}
    with torch.no_grad():
        for split in groups:
            hidden[split]=model.layers[3](model.layers[2](model.layers[1](model.layers[0](data[split]["x"]))))
            policy[split]=model.policy(hidden[split]).clone()
            require(bool(torch.isfinite(hidden[split]).all()) and bool(torch.isfinite(policy[split]).all()),"Nonfinite frozen cache")
    proof={"source_model_sha256":SOURCE_SHA,"frozen_weights_sha256":FROZEN_WEIGHTS,
           "rows_sha256":{split:config.digest([row["state_key"] for row in rows]) for split,rows in groups.items()},
           "features_sha256":{split:prior.weight_digest({"x":value["x"]}) for split,value in data.items()},
           "hidden_sha256":{split:prior.weight_digest({"h":value}) for split,value in hidden.items()},
           "policy_sha256":{split:prior.weight_digest({"logits":value}) for split,value in policy.items()}}
    return hidden,policy,proof


def policy_equal(model,hidden,policy):
    torch,_=runtime()
    with torch.no_grad():return frozen_digest(model.state_dict())==FROZEN_WEIGHTS and all(torch.equal(model.policy(hidden[split]),policy[split]) for split in hidden)


def metrics(model,rows,data,hidden):
    torch,_=runtime()
    with torch.no_grad():
        z=model.layers[5](model.layers[4](hidden)).squeeze(-1);prediction=data["d"]+data["r"]*z
        error=(prediction-data["y"]).abs().tolist();huber=torch.nn.functional.huber_loss(prediction,data["y"],delta=4.,reduction="none").tolist()
    require(all(math.isfinite(v) for v in error+huber),"Nonfinite pilot descriptive metrics")
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[str(row["teacher_depth"])+"/"+row["stratum"]].append(i)
    mean=lambda ids:{"rows":len(ids),"mae_seeds":math.fsum(error[i] for i in ids)/len(ids),"seed_huber_delta4":math.fsum(huber[i] for i in ids)/len(ids)}
    return {**mean(list(range(len(rows)))),"strata":{name:mean(ids) for name,ids in sorted(groups.items())},"scope":"Descriptive finite-horizon development targets; fixed final step"}


def export_payload(state,identity,step):
    _,prior=runtime()
    return {"schema":"congklak-mlp-export-v4","encoder":features_relay.ENCODER127,"identity":identity,"step":step,
            "status":"experimental_small_value_pilot_candidate","weights":{key:value.detach().cpu().tolist() for key,value in state.items()},"weights_sha256":prior.weight_digest(state),
            "terminal":"store0-minus-store1","output":"D_player0 + signed_active*tanh(current-actor-residual)","uses_oracle":False,
            "policy_output":"7 current-root-actor logits; legal actions masked by rules; terminal policy empty",
            "policy_training":"Source trunk and policy six tensors frozen bit-exact",
            "value_training":"Small reviewed weighted sampler; unit seed-space Huber4; valuehead65 only"}


def expected_torch_rng(identity):
    torch,_=runtime()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(identity["seed"]);evaluator();return torch.get_rng_state().clone()


def validate_state(state,identity,model):
    torch,_=runtime();digest=old.validate_state(state,identity,model)
    require(identity["device"]=="cpu" and not state["cuda_rng"],"Pilot state must be CPU only")
    rng=random.Random(identity["seed"])
    for _ in range(state["draws"]):rng.random()
    require(rng.getstate()==state["sampler_rng"] and random.Random(identity["seed"]).getstate()==state["python_rng"],"Pilot RNG/draw replay differs")
    require(torch.equal(expected_torch_rng(identity),state["torch_rng"]),"Pilot Torch RNG derivation differs")
    return digest


def catalog(path,identity,model,require_final=False):
    torch,_=runtime();path=Path(path).resolve()
    with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");configs=db.execute("SELECT body,hash FROM config").fetchall()
        require(len(configs)==1 and read_json(configs[0][0])==identity and configs[0][1]==config.digest(identity),"Pilot catalog identity differs")
        entries=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall();seen=[];verified=[];last=None
        for step,name,checksum,body,digest in entries:
            require(type(step) is int and (not seen or step>seen[-1]) and 0<step<=identity["steps"],"Pilot checkpoint prefix differs")
            file=(path/name).resolve();require(file.parent==path and file.is_file() and sha(file)==checksum,"Pilot checkpoint file missing/damaged")
            meta=read_json(body);require(set(meta)=={"step","identity_sha256","weights_sha256","metrics"} and type(meta["step"]) is int and meta["step"]==step and meta["identity_sha256"]==config.digest(identity) and config.digest(meta)==digest,"Pilot checkpoint catalog body differs")
            last=torch.load(file,map_location="cpu",weights_only=True)
            require(validate_state(last,identity,model)==meta["weights_sha256"] and last["step"]==step,"Pilot checkpoint tensor state differs")
            history=db.execute("SELECT body,hash FROM history WHERE step=?",(step,)).fetchall()
            require(len(history)==1 and read_json(history[0][0])==meta["metrics"] and config.digest(meta["metrics"])==history[0][1],"Pilot checkpoint history differs")
            verified.append({"step":step,"name":name,"sha256":checksum,"weights_sha256":meta["weights_sha256"]});seen.append(step)
        require(db.execute("SELECT COUNT(*) FROM history").fetchone()[0]==len(entries),"Pilot orphan metric history")
        if seen:require(set(range(identity["checkpoint_every"],seen[-1]+1,identity["checkpoint_every"]))<=set(seen),"Pilot scheduled checkpoint missing")
        if require_final:require(seen and seen[-1]==identity["steps"],"Pilot final checkpoint missing")
        return verified,last


def read_json(value):return json.loads(value)


def validate_result(value,identity,groups):
    require(set(value)==RESULT_FIELDS and value["schema"]==SCHEMA and value["status"]=="complete" and value["identity"]==identity,"Pilot completed fields/identity differ")
    for name,expected in (("steps_completed",identity["steps"]),("selected_step",identity["steps"]),("draws",identity["steps"]*identity["batch_size"]),("train_rows",len(groups["train"])),("development_rows",len(groups["development"]))):
        require(type(value[name]) is int and value[name]==expected,"Pilot completed count differs: "+name)
    require(value["frozen_weights_sha256"]==FROZEN_WEIGHTS and value["frozen_policy_unchanged"] is True and value["hidden_cache_sha256"]==identity["hidden_cache_sha256"] and value["training_performed"] is True and value["sealed_final_test_consumed"] is False and value["promotion"] is False,"Pilot completed scope/ownership differs")
    require(type(value["elapsed_seconds_this_invocation"]) in (int,float) and math.isfinite(value["elapsed_seconds_this_invocation"]) and value["elapsed_seconds_this_invocation"]>0 and type(value["peak_host_working_set_bytes"]) is int and 0<value["peak_host_working_set_bytes"]<12*1024**3,"Pilot resource witness differs")
    require(set(value["recovery"])=={"invalid_generations","restored_step"} and value["recovery"]["invalid_generations"]==[] and type(value["recovery"]["restored_step"]) is int and 0<=value["recovery"]["restored_step"]<=identity["steps"],"Pilot recovery witness differs")
    for name in ("initial_descriptive","descriptive"):
        require(set(value[name])==set(groups)-{"train"},"Pilot descriptive split ownership differs")
        for split,metric in value[name].items():validate_metric(metric,groups[split])
    require(value["initial_development"]==value["initial_descriptive"]["development"] and value["development"]==value["descriptive"]["development"],"Pilot descriptive alias differs")


def validate_metric(metric,rows):
    require(set(metric)=={"rows","mae_seeds","seed_huber_delta4","strata","scope"} and metric["scope"]=="Descriptive finite-horizon development targets; fixed final step","Pilot metric fields/scope differ")
    groups=defaultdict(list)
    for row in rows:groups[str(row["teacher_depth"])+"/"+row["stratum"]].append(row)
    require(set(metric["strata"])==set(groups),"Pilot metric stratum ownership differs")
    for item,count in [(metric,len(rows))]+[(metric["strata"][name],len(values)) for name,values in groups.items()]:
        require(type(item["rows"]) is int and item["rows"]==count and all(type(item[name]) in (int,float) and math.isfinite(item[name]) and 0<=item[name]<=limit for name,limit in (("mae_seeds",196.00001),("seed_huber_delta4",776.0001))),"Pilot metric count/value differs")
    require(all(set(item)=={"rows","mae_seeds","seed_huber_delta4"} for item in metric["strata"].values()),"Pilot stratum metric fields differ")


def train(path=DEFAULT,on_event=None,cancel=None,fixture=False,steps=None,checkpoint_every=None,batch_size=None):
    path=Path(path).resolve();settings=settings_for(fixture,steps,checkpoint_every,batch_size)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Pilot fixture must remain inside build")
    else:require(path==DEFAULT.resolve(),"Pilot production path differs");admit()
    manifest,groups=prepare();gate=None
    if not fixture:
        from .value_pilot_benchmark import pretraining_witness
        gate=pretraining_witness();require(gate["passed"] is True,"Pilot pretraining benchmark incomplete")
    if (path/"result.json").exists():
        checked=verify_run(path,fixture=fixture);result=read(path/"result.json")
        require(result["identity"]["steps"]==settings["steps"] and result["identity"]["checkpoint_every"]==settings["checkpoint_every"] and result["identity"]["batch_size"]==settings["batch_size"] and result["identity"]["pretraining_witness"]==gate,"Pilot completed invocation options differ")
        if on_event:on_event({"event":"value_pilot_completed_verified","step":result["selected_step"],"optimizer_steps":0,"files_rewritten":False,"torch_imported":"torch" in sys.modules})
        return result
    if on_event:on_event({"event":"value_pilot_cpu_admitted","train_rows":len(groups["train"]),"development_rows":len(groups["development"]),"torch_imported":"torch" in sys.modules})
    started=time.monotonic();torch,prior=runtime();require(not torch.cuda.is_initialized(),"Pilot CPU worker must not initialize CUDA")
    require(str(torch.__version__)==read(ROOT/"build/training-runtime.json")["torch"],"Pilot pinned Torch runtime differs")
    prior.deterministic("cpu");random.seed(settings["seed"]);torch.manual_seed(settings["seed"])
    model=evaluator();source=Model(SOURCE);model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in source.weights.items()})
    data={split:tensors(rows) for split,rows in groups.items()};hidden,policy,cache=hidden_cache(model,groups,data)
    identity={"schema":SCHEMA,**settings,"fixture":fixture,"rules_digest":config.RULES_DIGEST,"encoder":features_relay.ENCODER127,
              "source_sha256":sha(__file__),"tool_sha256":sha(ROOT/"tools/value-pilot-train.py"),"source_dependencies":{name:sha(ROOT/name) for name in DEPENDENCIES},
              "initial_model_file_sha256":SOURCE_SHA,"initial_weights_sha256":SOURCE_WEIGHTS,"initial_frozen_weights_sha256":FROZEN_WEIGHTS,
              "data_manifest_sha256":sha(DATA),"rows_file_sha256":sha(ROWS),"admitted_manifest_sha256":config.digest(manifest),
              "probability_row_order_sha256":config.digest([[row["state_key"],row["sampling_probability"]] for row in groups["train"]]),
              "hidden_cache_sha256":config.digest(cache),"hidden_cache":cache,"pretraining_witness":gate,
              "plan_file_sha256":None if fixture else sha(PLAN),"implementation_file_sha256":None if fixture else sha(IMPLEMENTATION),
              "torch_version":str(torch.__version__),"python_version":platform.python_version()}
    trainable=[parameter for parameter in model.parameters() if parameter.requires_grad]
    require(sum(parameter.numel() for parameter in trainable)==65 and policy_equal(model,hidden,policy),"Pilot trainable/frozen ownership differs")
    optimizer=torch.optim.AdamW(trainable,lr=settings["lr"],weight_decay=0.,foreach=False)
    initial_descriptive={split:metrics(model,rows,data[split],hidden[split]) for split,rows in groups.items() if split!="train"}
    initial=initial_descriptive["development"];rng=random.Random(settings["seed"]);cumulative=cdf(groups["train"])
    store=prior.Checkpoints(path,identity);step=draws=last_saved=0
    try:
        entries,saved=catalog(path,identity,model)
        recovery={"invalid_generations":[],"restored_step":0 if saved is None else saved["step"]}
        if saved is not None:
            model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);rng.setstate(saved["sampler_rng"])
            random.setstate(saved["python_rng"]);torch.set_rng_state(saved["torch_rng"]);step,draws=saved["step"],saved["draws"]
        last_saved=step
        if (path/"result.json").exists():
            result=read(path/"result.json");validate_result(result,identity,groups)
            require(saved is not None and step==settings["steps"] and read(path/"model.json")=={"model":export_payload(saved["model"],identity,step),"sha256":config.digest(export_payload(saved["model"],identity,step))},"Pilot completed export differs")
            require(result["weights_sha256"]==prior.weight_digest(saved["model"]) and result["initial_development"]==initial and result["initial_descriptive"]==initial_descriptive and result["development"]==result["descriptive"]["development"] and result["descriptive"]=={split:metrics(model,rows,data[split],hidden[split]) for split,rows in groups.items() if split!="train"} and policy_equal(model,hidden,policy),"Pilot completed metrics/policy differ")
            if on_event:on_event({"event":"value_pilot_completed_verified","step":step,"optimizer_steps":0,"files_rewritten":False})
            return result
        if on_event:on_event({"event":"value_pilot_training_started","step":step,"target":settings["steps"],"train_rows":len(groups["train"]),"recovery":recovery})
        def commit():
            nonlocal last_saved
            require(policy_equal(model,hidden,policy),"Pilot changed policy at checkpoint")
            state={"identity":identity,"step":step,"draws":draws,"model":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                   "optimizer":optimizer.state_dict(),"sampler_rng":rng.getstate(),"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":[],"scheduler":None,"amp_scaler":None,"pending_batches":[]}
            validate_state(state,identity,model);store.save(state,metrics(model,groups["development"],data["development"],hidden["development"]),on_event);last_saved=step
        while step<settings["steps"]:
            if cancel and cancel():break
            ids=torch.tensor(sample_ids(rng,cumulative,settings["batch_size"]),dtype=torch.int64);optimizer.zero_grad(set_to_none=True)
            z=model.layers[5](model.layers[4](hidden["train"][ids])).squeeze(-1);prediction=data["train"]["d"][ids]+data["train"]["r"][ids]*z
            loss=torch.nn.functional.huber_loss(prediction,data["train"]["y"][ids],delta=4.)
            require(bool(torch.isfinite(loss)),"Nonfinite pilot value loss");loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();step+=1;draws+=settings["batch_size"]
            if step%settings["checkpoint_every"]==0 or step==settings["steps"] or (cancel and cancel()):commit()
        if step>last_saved:commit()
        if step<settings["steps"]:return {"status":"interrupted","steps_completed":step,"model_available":False}
        entries,saved=catalog(path,identity,model,require_final=True);require(policy_equal(model,hidden,policy),"Pilot final policy changed")
        payload=export_payload(saved["model"],identity,step)
        result={"schema":SCHEMA,"status":"complete","identity":identity,"steps_completed":step,"selected_step":step,"draws":draws,"weights_sha256":payload["weights_sha256"],
                "train_rows":len(groups["train"]),"development_rows":len(groups["development"]),"initial_development":initial,"development":metrics(model,groups["development"],data["development"],hidden["development"]),
                "initial_descriptive":initial_descriptive,"descriptive":{split:metrics(model,rows,data[split],hidden[split]) for split,rows in groups.items() if split!="train"},
                "frozen_weights_sha256":FROZEN_WEIGHTS,"frozen_policy_unchanged":True,"hidden_cache_sha256":identity["hidden_cache_sha256"],
                "elapsed_seconds_this_invocation":time.monotonic()-started,"peak_host_working_set_bytes":prior.host_memory(),"recovery":recovery,"training_performed":True,"sealed_final_test_consumed":False,"promotion":False}
        validate_result(result,identity,groups)
        publish(path/"model.json",{"model":payload,"sha256":config.digest(payload)});publish(path/"result.json",result)
        if on_event:on_event({"event":"value_pilot_training_complete","step":step,"weights_sha256":payload["weights_sha256"],"frozen_policy_unchanged":True})
        return result
    finally:store.close()


def check_identity(identity,fixture,path):
    require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["fixture"] is fixture and identity["device"]=="cpu" and identity["source_sha256"]==sha(__file__) and identity["tool_sha256"]==sha(ROOT/"tools/value-pilot-train.py"),"Pilot verifier source/runtime differs")
    require(identity["rules_digest"]==config.RULES_DIGEST and identity["encoder"]==features_relay.ENCODER127 and identity["torch_version"]==read(ROOT/"build/training-runtime.json")["torch"] and identity["python_version"]==platform.python_version(),"Pilot rules/encoder/framework differs")
    require(identity["initial_model_file_sha256"]==sha(SOURCE)==SOURCE_SHA and identity["initial_weights_sha256"]==SOURCE_WEIGHTS and identity["initial_frozen_weights_sha256"]==FROZEN_WEIGHTS,"Pilot source lineage differs")
    expected=settings_for(fixture,identity["steps"] if fixture else None,identity["checkpoint_every"] if fixture else None,identity["batch_size"] if fixture else None)
    require(all(type(identity[name]) is type(value) and identity[name]==value for name,value in expected.items()),"Pilot fixed contract differs")
    require(identity["source_dependencies"]=={name:sha(ROOT/name) for name in DEPENDENCIES} and identity["data_manifest_sha256"]==sha(DATA) and identity["rows_file_sha256"]==sha(ROWS),"Pilot dependencies/data differ")
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build") and identity["pretraining_witness"] is None and identity["plan_file_sha256"] is None and identity["implementation_file_sha256"] is None,"Pilot fixture identity differs")
    else:
        admit();from .value_pilot_benchmark import pretraining_witness
        require(Path(path).resolve()==DEFAULT.resolve() and identity["pretraining_witness"]==pretraining_witness() and identity["plan_file_sha256"]==sha(PLAN) and identity["implementation_file_sha256"]==sha(IMPLEMENTATION),"Pilot training gate/plan differs")


def inspect_run(path=DEFAULT,output=None,fixture=False):
    path=Path(path).resolve();verify_run(path,fixture=fixture);model=Model(path/"model.json");identity=model.identity
    manifest,groups=prepare();require(identity["admitted_manifest_sha256"]==config.digest(manifest),"Pilot admitted manifest differs")
    result=read(path/"result.json");validate_result(result,identity,groups)
    torch,prior=runtime();prior.deterministic("cpu");check=evaluator()
    check.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in Model(SOURCE).weights.items()})
    data={split:tensors(rows) for split,rows in groups.items()};hidden,policy,cache=hidden_cache(check,groups,data)
    require(config.digest(cache)==identity["hidden_cache_sha256"] and cache==identity["hidden_cache"],"Pilot hidden cache proof differs")
    require(result["initial_descriptive"]=={split:metrics(check,rows,data[split],hidden[split]) for split,rows in groups.items() if split!="train"} and result["initial_development"]==result["initial_descriptive"]["development"],"Pilot initial metrics differ")
    entries,state=catalog(path,identity,check,require_final=True)
    require(model.envelope=={"model":export_payload(state["model"],identity,identity["steps"]),"sha256":config.digest(export_payload(state["model"],identity,identity["steps"]))} and model.id==result["weights_sha256"],"Pilot export/final checkpoint differs")
    check.load_state_dict(state["model"])
    require(policy_equal(check,hidden,policy) and result["descriptive"]=={split:metrics(check,rows,data[split],hidden[split]) for split,rows in groups.items() if split!="train"} and result["development"]==result["descriptive"]["development"],"Pilot final policy/metrics differ")
    files=expected_files(path,identity,entries,fixture)
    value={"schema":"value-pilot-final-model-witness-v1","passed":True,"identity":identity,"model_file_sha256":model.file_sha256,"model_weights_sha256":model.id,
           "selected_step":identity["steps"],"checkpoints_verified":len(entries),"checkpoints":entries,"all_optimizer_sampler_rng_states_verified":True,
           "final_export_equals_checkpoint_tensors":True,"all_frozen_tensors_bit_exact":True,"all_dataset_policy_logits_bit_exact":True,
           "frozen_weights_sha256":FROZEN_WEIGHTS,"frozen_parameter_count":12807,"trainable_parameter_count":65,"development":result["development"],"initial_development":result["initial_development"],
           "descriptive":result["descriptive"],"initial_descriptive":result["initial_descriptive"],
           "hidden_cache_sha256":identity["hidden_cache_sha256"],"sealed_final_test_consumed":False,"promotion":False,
           "inputs":{str(file.relative_to(ROOT)).replace("\\","/"):sha(file) for file in files}}
    value["sha256"]=config.digest(value)
    if output is not None:publish(output,value)
    return value


def expected_files(path,identity,entries,fixture):
    files=[path/"model.json",path/"result.json",path/"catalog.sqlite3",SOURCE]+[ROOT/name for name in SOURCES+DEPENDENCIES]+[path/entry["name"] for entry in entries]
    if not fixture:files += [PLAN,IMPLEMENTATION]
    return files


def float_digest(name,values):
    values=list(flatten_weights(values));return hashlib.sha256(name.encode()+struct.pack("<"+"f"*len(values),*values)).hexdigest()


def validate_cache_identity(identity,groups):
    proof=identity["hidden_cache"]
    require(set(proof)=={"source_model_sha256","frozen_weights_sha256","rows_sha256","features_sha256","hidden_sha256","policy_sha256"} and proof["source_model_sha256"]==SOURCE_SHA and proof["frozen_weights_sha256"]==FROZEN_WEIGHTS and config.digest(proof)==identity["hidden_cache_sha256"],"Pilot cache identity differs")
    for name in ("rows_sha256","features_sha256","hidden_sha256","policy_sha256"):
        require(set(proof[name])==set(groups) and all(isinstance(value,str) and len(value)==64 and len(bytes.fromhex(value))==32 for value in proof[name].values()),"Pilot cache split/hash ownership differs")
    require(proof["rows_sha256"]=={split:config.digest([row["state_key"] for row in rows]) for split,rows in groups.items()} and proof["features_sha256"]=={split:float_digest("x",[row["cached_features"]["x"] for row in rows]) for split,rows in groups.items()},"Pilot cache rows/features differ")


def verify_run(path=DEFAULT,output=None,fixture=False):
    """Strict JSON/catalog/file admission without importing Torch or tensor payloads."""
    path=Path(path).resolve();model=Model(path/"model.json");identity=model.identity;check_identity(identity,fixture,path)
    manifest,groups=prepare();validate_cache_identity(identity,groups)
    require(identity["admitted_manifest_sha256"]==config.digest(manifest) and identity["probability_row_order_sha256"]==config.digest([[row["state_key"],row["sampling_probability"]] for row in groups["train"]]),"Pilot data/probability identity differs")
    require(model_frozen_digest(model)==FROZEN_WEIGHTS and model.envelope["model"]["step"]==identity["steps"] and model.envelope["model"]["uses_oracle"] is False,"Pilot CPU export ownership differs")
    result=read(path/"result.json");validate_result(result,identity,groups);require(result["weights_sha256"]==model.id,"Pilot model/result weights differ")
    entries=[]
    with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");configs=db.execute("SELECT body,hash FROM config").fetchall()
        require(len(configs)==1 and json.loads(configs[0][0])==identity and configs[0][1]==config.digest(identity),"Pilot CPU catalog config differs")
        seen=[]
        for step,name,checksum,body,digest in db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id"):
            require(type(step) is int and 0<step<=identity["steps"] and (not seen or step>seen[-1]),"Pilot CPU checkpoint geometry differs")
            file=(path/name).resolve();require(file.parent==path and sha(file)==checksum,"Pilot CPU checkpoint file differs")
            meta=json.loads(body);require(set(meta)=={"step","identity_sha256","weights_sha256","metrics"} and type(meta["step"]) is int and meta["step"]==step and meta["identity_sha256"]==config.digest(identity) and config.digest(meta)==digest,"Pilot CPU checkpoint metadata differs")
            validate_metric(meta["metrics"],groups["development"])
            history=db.execute("SELECT body,hash FROM history WHERE step=?",(step,)).fetchall();require(len(history)==1 and json.loads(history[0][0])==meta["metrics"] and history[0][1]==config.digest(meta["metrics"]),"Pilot CPU metric history differs")
            entries.append({"step":step,"name":name,"sha256":checksum,"weights_sha256":meta["weights_sha256"]});seen.append(step)
        scheduled=list(range(identity["checkpoint_every"],identity["steps"]+1,identity["checkpoint_every"]))
        if identity["steps"] not in scheduled:scheduled.append(identity["steps"])
        require(seen and set(scheduled)<=set(seen) and seen[-1]==identity["steps"] and db.execute("SELECT COUNT(*) FROM history").fetchone()[0]==len(entries),"Pilot CPU cadence/final history differs")
        require(entries[-1]["weights_sha256"]==model.id and meta["metrics"]==result["development"],"Pilot CPU final checkpoint/export linkage differs")
    value={"schema":"value-pilot-cpu-completed-verification-v1","passed":True,"identity":identity,"model_file_sha256":model.file_sha256,"model_weights_sha256":model.id,"selected_step":identity["steps"],"checkpoints_verified":len(entries),"checkpoints":entries,"result":result,
           "inputs":{str(file.relative_to(ROOT)).replace("\\","/"):sha(file) for file in expected_files(path,identity,entries,fixture)}}
    if output is not None:publish(output,value)
    return value


def witness(path=DEFAULT,proof=WITNESS):
    """CPU-only admission of the separately tensor-verified candidate."""
    value=read(proof)
    require(value["schema"]=="value-pilot-final-model-witness-v1" and value["passed"] is True and value["sha256"]==config.digest({key:item for key,item in value.items() if key!="sha256"}),"Pilot witness checksum differs")
    for name,checksum in value["inputs"].items():
        file=(ROOT/name).resolve();require(file.is_relative_to(ROOT) and sha(file)==checksum,"Pilot witness input changed: "+name)
    completed=verify_run(path);model=Model(Path(path)/"model.json");identity=model.identity
    require(value["inputs"]==completed["inputs"] and value["checkpoints"]==completed["checkpoints"] and value["checkpoints_verified"]==completed["checkpoints_verified"] and value["development"]==completed["result"]["development"] and value["initial_development"]==completed["result"]["initial_development"] and value["descriptive"]==completed["result"]["descriptive"] and value["initial_descriptive"]==completed["result"]["initial_descriptive"],"Pilot inspected proof current catalog/result differs")
    require(model.id==value["model_weights_sha256"] and model.file_sha256==value["model_file_sha256"] and identity==value["identity"] and model_frozen_digest(model)==FROZEN_WEIGHTS,"Pilot witness model/frozen tensors differ")
    require(value["selected_step"]==CONTRACT["steps"] and value["all_optimizer_sampler_rng_states_verified"] is True and value["final_export_equals_checkpoint_tensors"] is True and value["all_frozen_tensors_bit_exact"] is True and value["all_dataset_policy_logits_bit_exact"] is True and value["trainable_parameter_count"]==65 and value["frozen_parameter_count"]==12807,"Pilot witness verification claims differ")
    require(set(value)=={"schema","passed","identity","model_file_sha256","model_weights_sha256","selected_step","checkpoints_verified","checkpoints","all_optimizer_sampler_rng_states_verified","final_export_equals_checkpoint_tensors","all_frozen_tensors_bit_exact","all_dataset_policy_logits_bit_exact","frozen_weights_sha256","frozen_parameter_count","trainable_parameter_count","development","initial_development","descriptive","initial_descriptive","hidden_cache_sha256","sealed_final_test_consumed","promotion","inputs","sha256"} and value["frozen_weights_sha256"]==FROZEN_WEIGHTS and value["hidden_cache_sha256"]==identity["hidden_cache_sha256"] and value["sealed_final_test_consumed"] is False and value["promotion"] is False,"Pilot witness exact fields/scope differ")
    return {**value,"proof_file_sha256":sha(proof)}
