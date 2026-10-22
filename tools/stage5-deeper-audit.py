"""Independent published-root, selected-model and paired-arena audit."""
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from congklak import config,pilot,campaign
from congklak.neural_relay import Model,NeuralNative
from stage5_deeper_audit_acceptance import literal_features,value_and_policy,literal_position,source_rows,frozen_prior,IndependentDeeperAcceptance
import reference as literal

WORK=ROOT/"build/stage5-deeper-audit";WORK.mkdir(parents=True,exist_ok=True)

TEST_GROUPS={
    "cuda-tests.json":["test_08_cuda_actual_precommit_kill_full_state_bitwise","test_12_cuda_final_checkpoint_kill_before_export_resume"],
    "benchmark-tests.json":["test_10_actual_pinned_benchmark_kill_resume_and_corrupt_regret"],
    "core-final-tests.json":["test_01_prior_science_environment_unchanged","test_02_literal_relay_features_and_store_independence",
        "test_03_complete_h8_root_q_is_successor_h7","test_04_native_scalar_avx_both_heads_and_tails",
        "test_05_literal_native_minimax_and_controlled_feature_work","test_06_actual_analyzer_kill_resume_and_checksum_valid_corruption",
        "test_07_teacher_actual_precommit_kill_resume_and_forged_cache_refusal","test_09_cpu_affinity_stable_admission_and_restore",
        "test_11_actual_completed_arena_game_kill_resume_on_isolated_backup","test_13_real_migration_isolated_elapsed_drift_and_corrupt_source_refusal"]}

def collect_tests():
    expected=unittest.defaultTestLoader.getTestCaseNames(IndependentDeeperAcceptance)
    merged={};groups=[];covered=[];duration=0
    metadata={"source_sha256","tests_run","passed","failures","errors","skips","duration_seconds","covered_methods"}
    digest=sha(ROOT/"tests/stage5_deeper_audit_acceptance.py")
    for file,names in TEST_GROUPS.items():
        path=WORK/file;value=load(path)
        require(value["source_sha256"]==digest and value["passed"] and value["tests_run"]==len(names) and not value["failures"]+value["errors"]+value["skips"],"Independent group incomplete/stale: "+file)
        require(value["covered_methods"]==names,"Independent group method coverage differs: "+file)
        for key,item in value.items():
            if key in metadata:continue
            require(key not in merged,"Duplicate independent evidence group: "+key);merged[key]=item
        duration+=value["duration_seconds"];covered.extend(names)
        groups.append(dict(path=str(path.relative_to(ROOT)).replace("\\","/"),sha256=sha(path),methods=names,tests_run=len(names)))
    require(len(covered)==len(set(covered))==13 and sorted(covered)==expected,"Independent coverage omits/repeats a test")
    result={**merged,"source_sha256":digest,"tests_run":13,"passed":True,"failures":0,"errors":0,"skips":0,
            "duration_seconds":duration,"covered_methods":sorted(covered),"groups":groups,"acceptance_group_collection":True}
    (WORK/"tests.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(event="deeper_independent_checks_collected",passed=True,tests_run=13,source_sha256=digest)),flush=True)
    return result

def run_test_group(name):
    from stage5_deeper_audit_acceptance import EVIDENCE
    file={"cuda":"cuda-tests.json","benchmark":"benchmark-tests.json","core":"core-final-tests.json"}[name]
    names=TEST_GROUPS[file];started=time.monotonic();EVIDENCE.clear()
    suite=unittest.TestSuite(IndependentDeeperAcceptance(method) for method in names)
    checked=unittest.TextTestRunner(verbosity=2).run(suite)
    result={**EVIDENCE,"source_sha256":sha(ROOT/"tests/stage5_deeper_audit_acceptance.py"),"tests_run":checked.testsRun,"passed":checked.wasSuccessful(),
            "failures":len(checked.failures),"errors":len(checked.errors),"skips":len(checked.skipped),"duration_seconds":time.monotonic()-started,"covered_methods":names}
    (WORK/file).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(event="deeper_independent_group_done",group=name,passed=result["passed"],tests_run=result["tests_run"])),flush=True)
    return 0 if checked.wasSuccessful() else 1

def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
def require(test,message):
    if not test:raise AssertionError(message)
def interval(values):
    rng=random.Random(20261016);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4096)) if values else []
    return [samples[102],samples[3993]] if samples else None
def checked(db,name):
    out={}
    for key,body,digest in db.execute("SELECT state,body,hash FROM "+name+" ORDER BY state"):
        value=json.loads(body);require(config.digest(value)==digest,"Corrupt "+name);out[key]=value
    return out
def label_rows(folder):
    # Full root reports contain elementary sowing traces. Stream them so the
    # audit does not turn a bounded teacher run into a multi-gigabyte loader.
    with closing(sqlite3.connect((folder/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        for key,body,digest in db.execute("SELECT state,body,hash FROM labels ORDER BY state"):
            value=json.loads(body);require(config.digest(value)==digest,"Corrupt labels");yield key,value

def migration():
    """Compare preserved V1 bodies with V2 independently, including every delta."""
    old=ROOT/"data/deeper-9000-v1";new=ROOT/"data/deeper-9000-v2"
    plan=load(ROOT/"build/stage5-deeper-training-plan-v3.json");identity=plan["dataset_identity"]
    published=load(new/"migration.json");snapshot=identity["migration_source"]
    require(published["source"]==snapshot and published["destination"]==str(new.resolve()) and published["copied"]==dict(roots=9000,features=3473,labels=3472),"Migration source/counts differ")
    require(snapshot["legacy_source_sha256"]==sha(ROOT/"congklak/deeper_teacher.py") and snapshot["path"]==str(old.resolve()),"Preserved legacy source differs")
    observed=[];deltas=Counter();validated=Counter()
    with closing(sqlite3.connect((old/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as source,closing(sqlite3.connect((new/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as target:
        source.execute("BEGIN");target.execute("BEGIN")
        body,digest=source.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        require(config.digest(json.loads(body))==digest==snapshot["identity_sha256"],"Legacy catalog identity differs")
        for table in ("roots","features","labels"):
            keys=list(source.execute("SELECT state,hash FROM "+table+" ORDER BY state"))
            require(dict(rows=len(keys),key_hash_sha256=config.digest(keys))==snapshot["tables"][table],"Predeclared legacy snapshot changed")
            for key,body,digest in source.execute("SELECT state,body,hash FROM "+table+" ORDER BY state"):
                before=json.loads(body);require(config.digest(before)==digest,"Corrupt preserved source row")
                row=target.execute("SELECT body,hash FROM "+table+" WHERE state=?",(key,)).fetchone();require(row is not None,"Migrated row missing")
                after=json.loads(row[0]);require(config.digest(after)==row[1],"Corrupt migrated row")
                if table!="labels":require((body,digest)==row,"Migration changed root/feature bytes")
                else:
                    require(after["report"]==before["report"] and after["root_sha256"]==before["root_sha256"],"Migration changed report/root provenance")
                    require({k:v for k,v in before["label"].items() if k!="work"}=={k:v for k,v in after["label"].items() if k!="work"},"Migration changed target/action Q")
                    original=before["label"]["work"];replacement=after["label"]["work"]
                    require(replacement=={k:before["report"][k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")},"Migration work not anchored to original completed report")
                    require(all(original[k]==replacement[k] for k in original if k!="elapsed_ms") and original["elapsed_ms"]>=replacement["elapsed_ms"],"Migration changed non-timing work/negative drift")
                    delta=original["elapsed_ms"]-replacement["elapsed_ms"];deltas[delta]+=1
                    if delta:observed.append(dict(state_key=key,original_row_sha256=digest,return_elapsed_ms=original["elapsed_ms"],report_elapsed_ms=replacement["elapsed_ms"],migrated_row_sha256=row[1]))
                validated[table]+=1
        require(published["changes"]==observed and published["adjusted_timing_rows"]==len(observed)==543 and deltas==Counter({0:2929,1:536,2:7}),"Migration omitted/revised original timing evidence")
        body,digest=target.execute("SELECT body,hash FROM meta WHERE key='migration'").fetchone()
        require(json.loads(body)==published and config.digest(published)==digest,"Migration lacks matching FULL catalog commit")
    return dict(original_roots_and_feature_bytes_unchanged=True,all_original_reports_targets_action_q_unchanged=True,
                independently_compared=dict(validated),adjusted_timing_rows=543,elapsed_delta_counts=dict(deltas),original_timestamps_and_digests_preserved=True)

def teacher():
    folder=ROOT/"data/deeper-9000-v2";manifest=load(folder/"manifest.json");ident=manifest["identity"]
    require(manifest["schema"]==ident["schema"]=="root-preserving-H8-relay-corpus-v2" and manifest["rows"]==9000,"Teacher shape differs")
    require(ident["splits"]=={"train":8000,"development":1000} and ident["sealed_final_test_consumed"] is False and ident["new_successor_rows"]==0,"Teacher root ownership differs")
    require(ident["choice_depth"]==8 and ident["action_successor_horizon"]==7 and ident["search"]==dict(max_depth=8,time_ms=0,node_budget=0,memory_mb=512,tt_mb=16,use_pvs=True),"Teacher budget/recurrence differs")
    require(ident["native_build_sha256"]==sha(ROOT/"build/congklak_native.dll") and ident["source_sha256"]==sha(ROOT/"congklak/deeper_teacher_v2.py"),"Teacher source/DLL changed")
    require(ident["work_snapshot"]=="Exact completed-depth report; outer analyze-return timing excluded","Teacher work snapshot contract differs")
    for name,digest in ident["teacher_source_sha256"].items():require(sha(ROOT/name)==digest,"Teacher native source changed")
    for name,digest in ident["source_dependencies_sha256"].items():require(sha(ROOT/(name.replace(".","/")+".py"))==digest,"Teacher loader dependency changed")
    require(ident["teacher"]=="H_v1/untrained/oracle-free/exact-completed-root-depth8/full-root-Q","Learned/oracle teacher ancestry")
    expected={r["state_key"]:r for split in ("train","development") for r in source_rows(split)}
    require(len(expected)==9000 and config.digest({s:list(source_rows(s)) for s in ("train","development")})==ident["root_rows_sha256"],"Source root body/order changed")
    with closing(sqlite3.connect((folder/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");roots=checked(db,"roots");features=checked(db,"features")
        label_keys={row[0] for row in db.execute("SELECT state FROM labels")}
        for key,want in (("identity",ident),("complete",manifest)):
            body,digest=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(json.loads(body)==want and config.digest(want)==digest,"Uncommitted teacher metadata")
    require(roots==expected and label_keys==set(features)==set(roots),"Teacher roots added/removed/mutated")
    semantics=[];labels={};horizon_changes=0;best_changes=0;pv_checks=0
    for key,entry in label_rows(folder):
        row=expected[key];p=literal_position(row["state"]);normal=literal.normalize(p);legal=list(literal.legal_actions(p));label=entry["label"];report=entry["report"]
        require(entry["root_sha256"]==features[key]["root_sha256"]==config.digest(row),"Label/feature root provenance differs")
        q={c["pit"]:c["value_player0"] for c in label["action_scores"]};require(list(q)==legal and all(type(v) is int for v in q.values()),"H8 Q coverage differs")
        optimum=(max if normal.controller==0 else min)(q.values());require(label["target_player0"]==optimum and label["best_moves"]==[a for a,v in q.items() if v==optimum],"H8 ranking differs")
        require(label["kind"]=="finite_choice_horizon" and label["choice_depth"]==report["choice_depth"]==8 and label["oracle_manifest_sha256"] is None,"Teacher incomplete/uses oracle")
        require(report["completed_iteration"] is True and report["evaluator"]=="H_v1" and report["proof_status"]=="finite_horizon_result","Teacher proof scope differs")
        require(report["position"]==row["state"] and report["depth_result"]["value_player0"]==optimum and report["best_moves"]==label["best_moves"],"Report root/target differs")
        require(q=={c["pit"]:c["depth_result"]["value_player0"] for c in report["candidates"]},"Label Q/report differs")
        require(label["coverage"]==report["coverage"] and label["work"]=={k:report[k] for k in ("nodes","deposits","elapsed_ms","peak_process_bytes")},"Label work differs")
        # All published relay tensors are checked by the separate literal kernel.
        x,d,r=literal_features(p);cache=features[key]
        require(tuple(cache["x"])==x and (cache["d"],cache["r"])==(d,r),"Independent relay feature derivation differs")
        require(cache["encoder"]==ident["feature_encoder"] and len(x)==127,"Feature schema differs")
        require(all((d-abs(r))<=v<=(d+abs(r)) for v in q.values()),"Teacher Q outside physical interval")
        # Deterministic modulo32 root-key subset of every full candidate PV,
        # not a random 1/32 sample (root-key bytes encode seed counts). Independent
        # transitions and H_v1 terminal/horizon evaluation, not a chosen PV only.
        if int(key[:8],16)%32==0:
            for candidate in report["candidates"]:
                current=p
                for index,move in enumerate(candidate["pv"]):
                    require(move["actor"]==literal.normalize(current).controller,"PV actor differs")
                    require(index!=0 or move["pit"]==candidate["pit"],"PV root choice differs")
                    current=literal.apply(current,move["pit"]).position
                require(len(candidate["pv"])==8 or literal.normalize(current).terminal,"Teacher short horizon")
                require(current.counts[7]-current.counts[15]==candidate["depth_result"]["value_player0"],"Literal teacher PV leaf differs");pv_checks+=1
        horizon_changes+=label["target_player0"]!=row["label"]["target_player0"];best_changes+=label["best_moves"]!=row["label"]["best_moves"]
        semantics.append({"root":row,"H8":{**label,"work":{}},"features":{k:v for k,v in cache.items() if k!="build_seconds"}})
        labels[key]=label
    require(config.digest(semantics)==manifest["semantic_sha256"],"Published teacher semantic hash differs")
    require(len(labels)==9000,"Teacher snapshot changed")
    groups={s:[{**r,"H6_reference":r["label"],"H8_reference":labels[r["state_key"]]} for r in source_rows(s)] for s in ("train","development")}
    return dict(roots=9000,train=8000,development=1000,independent_relay_tensors=9000,literal_candidate_teacher_pvs=pv_checks,
                H6_H8_root_value_changes=horizon_changes,H6_H8_best_set_changes=best_changes,semantic_sha256=manifest["semantic_sha256"],
                source_root_rows_unchanged=True,sealed_target_bodies_read=False,successor_rows=0),groups

def evaluation(groups):
    plan=load(ROOT/"build/stage5-deeper-training-plan-v3.json");result=load(ROOT/"build/stage5-deeper-training-results.json")
    require(result["plan"]==plan and plan["teacher_depths"]==[6,8] and plan["input_counts"]==[43,127] and plan["seeds"]==[2718,31415],"Factorial plan differs")
    require(plan["width"]==64 and plan["steps"]==8000 and plan["batch_size"]==512 and plan["checkpoint_every"]==100 and plan["lr"]==.001,"Matched training budget differs")
    require(plan["parameters"]=={"43":7496,"127":12872} and plan["trainable_parameters"]=={"43":7041,"127":12417} and plan["dummy_policy_parameters"]==455,"Parameter accounting differs")
    for name,key in (("tools/stage5-deeper-training-v3.py","source_sha256"),("congklak/training_v7.py","trainer_sha256"),("congklak/deeper_teacher_v2.py","deeper_teacher_sha256"),("congklak/features_relay.py","features_sha256")):
        require(sha(ROOT/name)==plan[key],"Frozen training source changed")
    rows={(r["teacher_depth"],r["input_count"],r["seed"]):r for r in result["rows"]};require(len(rows)==8 and set(rows)=={(h,f,s) for h in (6,8) for f in (43,127) for s in (2718,31415)},"Missing/repeated factorial cells")
    verified=[];checkpoint_files=0
    for (h,f,seed),row in rows.items():
        model=Model(ROOT/row["model"]);run=model.path.parent;saved=load(run/"result.json");identity=saved["identity"]
        require(model.id==row["weights_sha256"]==saved["weights_sha256"] and model.identity==identity,"Export/selection differs")
        require(identity["teacher_depth"]==h and identity["input_count"]==f and identity["seed"]==seed and identity["width"]==64,"Training condition differs")
        require(identity["policy_objective"] is None and identity["bias_objective"] is None and identity["oracle_training"] is False and identity["sealed_final_test_consumed"] is False,"Undeclared training objective/ancestry")
        require(identity["source_sha256"]==plan["trainer_sha256"] and identity["dataset_semantic_sha256"]==load(ROOT/"data/deeper-9000-v2/manifest.json")["semantic_sha256"],"Training source/teacher differs")
        require(saved["status"]=="complete" and saved["steps_completed"]==8000 and saved["draws"]==4096000 and saved["parameters"]==plan["parameters"][str(f)] and saved["trainable_parameters"]==plan["trainable_parameters"][str(f)],"Training incomplete/changed budget")
        for name,digest in identity["source_dependencies_sha256"].items():require(sha(ROOT/(name.replace(".","/")+".py"))==digest,"Training loader dependency differs")
        with closing(sqlite3.connect((run/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            history=[]
            for step,body,digest in db.execute("SELECT step,body,hash FROM history ORDER BY step"):
                metrics=json.loads(body);require(config.digest(metrics)==digest,"Corrupt checkpoint metrics");history.append((step,metrics["finite_choice_horizon"]["uniform_stratum_mae_seeds"]))
            require([s for s,v in history]==list(range(100,8001,100)),"Unequal checkpoint opportunities")
            require(min(history,key=lambda pair:pair[1])[0]==row["best_step"]==saved["best_step"],"Crosshorizon/arena checkpoint tuning")
            for step,name,digest,body,body_digest in db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id"):
                file=(run/name).resolve();require(file.parent==run and sha(file)==digest and config.digest(json.loads(body))==body_digest,"Checkpoint/file checksum differs");checkpoint_files+=1
        metrics={};max_error=0
        with NeuralNative(model,mode="value-none",backend="avx2") as native:
            buckets={horizon:defaultdict(list) for horizon in (6,8)}
            for example in groups["development"]:
                p=literal_position(example["state"]);raw,units,logits=value_and_policy(p,model.envelope["model"]);actual=native.evaluate(example["state"])
                require(actual["value_units"]==units and abs(actual["value_player0"]-raw)<1e-10 and max(abs(a-b) for a,b in zip(actual["policy_logits"],logits))<1e-10,"Independent selected model parity differs")
                d=p.counts[7]-p.counts[15]
                for horizon in (6,8):buckets[horizon][example["stratum"]].append((abs(raw-example[f"H{horizon}_reference"]["target_player0"]),abs(d-example[f"H{horizon}_reference"]["target_player0"])))
            for horizon in (6,8):
                computed={name:dict(count=len(values),mae_seeds=statistics.mean(x for x,y in values),H_v1_mae_seeds=statistics.mean(y for x,y in values)) for name,values in buckets[horizon].items()}
                claimed=row["cross_teacher_metrics"][f"H{horizon}"]["finite_choice_horizon"]
                for name,value in computed.items():
                    require(value["count"]==claimed["strata"][name]["count"] and value["H_v1_mae_seeds"]==claimed["strata"][name]["H_v1_mae_seeds"],"Value cohort/H baseline differs")
                    max_error=max(max_error,abs(value["mae_seeds"]-claimed["strata"][name]["mae_seeds"]))
                require(max_error<5e-4,"FP32 reported MAE differs beyond double reference tolerance")
                macro=statistics.mean(v["mae_seeds"] for v in computed.values());require(abs(macro-claimed["uniform_stratum_mae_seeds"])<5e-4,"Reported macro differs")
                ordinary={k:v for k,v in computed.items() if k.endswith("/ordinary") and v["count"]>=20}
                metrics[f"H{horizon}"]=dict(macro_mae=macro,ordinary=ordinary)
            gate=all(v["mae_seeds"]<=v["H_v1_mae_seeds"] for v in metrics["H8"]["ordinary"].values());require(gate==row["ordinary_gate_passed"],"H8 ordinary gate differs")
        require(abs(row["macro_mae"]-row["metrics"]["uniform_stratum_mae_seeds"])<1e-15,"Ownteacher selection score differs")
        verified.append(dict(condition=row["condition"],seed=seed,best_step=row["best_step"],metrics=metrics,max_gpu_fp32_mae_difference=max_error,ordinary_H8_gate_passed=gate))
    for h in (6,8):
        for f in (43,127):
            require(result["selected"][f"h{h}-f{f}"]==min((rows[h,f,s] for s in (2718,31415)),key=lambda r:(r["macro_mae"],r["seed"])),"Representative uses another selection metric")
    verification=load(ROOT/"build/stage5-deeper-training-verification.json")
    require(verification["checkpoint_files_verified"]==checkpoint_files==640 and verification["all_checks_passed"] and verification["matched_sampler_sequences_equal_by_seed"] and verification["dummy_policy_head_frozen"],"Full state/sampler verification incomplete")
    state_audit=checkpoint_states(result,groups)
    return dict(factorial_runs=8,checkpoint_files_independently_hashed=checkpoint_files,own_horizon_only_selection=True,all_selected_models_literal_verified=True,
                selected_development_roots_per_model=1000,matched_sampling_state_verified=True,independent_checkpoint_state_audit=state_audit,
                relay_capacity_confound_disclosed=True,runs=verified,sealed_target_bodies_read=False)

def checkpoint_states(result,groups):
    """Use the existing Torch runtime only to read safe tensor checkpoints."""
    runtime=load(ROOT/"build/training-runtime.json")["python"]
    request={"rows":result["rows"],"stratum_sizes":dict(Counter(r["stratum"] for r in groups["train"]))}
    script="""from collections import defaultdict
from contextlib import closing
import hashlib,json,random,sqlite3,sys
from pathlib import Path
import torch
request=json.load(sys.stdin);root=Path.cwd();samplers={};heads={};count=0
def digest(state):
 h=hashlib.sha256()
 for key,tensor in sorted(state.items()):h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
for row in request['rows']:
 folder=(root/row['model']).parent
 torch.manual_seed(row['seed']);layers=torch.nn.Sequential(torch.nn.Linear(row['input_count'],64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Tanh());policy=torch.nn.Linear(64,7)
 initial={'policy.'+k:v for k,v in policy.state_dict().items()}
 with closing(sqlite3.connect((folder/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
  entries=db.execute('SELECT step,name,body FROM checkpoints ORDER BY id').fetchall()
  for step,name,body in entries:
   state=torch.load(folder/name,map_location='cpu',weights_only=True);meta=json.loads(body)
   assert state['step']==step and state['draws']==step*512 and state['identity']['teacher_depth']==row['teacher_depth'] and state['identity']['input_count']==row['input_count']
   assert state['scheduler'] is None and state['amp_scaler'] is None and state['pending_batches']==[]
   assert digest(state['model'])==meta['weights_sha256']
   assert len(state['optimizer']['state'])==6 and len(state['optimizer']['param_groups'][0]['params'])==6
   assert all(torch.equal(state['model'][k],v) for k,v in initial.items())
   assert state['best_step']<=step and state['best_step']%100==0
   assert torch.is_tensor(state['torch_rng']) and len(state['cuda_rng'])==1 and torch.is_tensor(state['cuda_rng'][0])
   count+=1
  assert state['step']==8000 and state['best_step']==row['best_step'] and digest(state['best_model'])==row['weights_sha256']
  if row['seed'] in samplers:assert state['sampler_rng']==samplers[row['seed']]
  else:samplers[row['seed']]=state['sampler_rng']
strata=sorted(request['stratum_sizes'])
for seed,want in samplers.items():
 rng=random.Random(seed)
 for _ in range(4096000):
  stratum=rng.choice(strata);rng.randrange(request['stratum_sizes'][stratum])
 assert rng.getstate()==want
print(json.dumps(dict(checkpoint_states_read=count,checkpoint_model_digests_recomputed=True,all_dummy_heads_equal_independent_seeded_initialization=True,
 optimizer_state_excludes_dummy_head=True,all_rng_pending_scheduler_amp_fields_present=True,final_best_model_export_digests_equal=True,
 matched_sampler_states_equal_all_four_conditions_per_seed=True,full_sampler_draws_replayed_per_seed=4096000,seeds_replayed=2,weights_only_loading=True)))
"""
    completed=subprocess.run([runtime,"-c",script],input=json.dumps(request),cwd=ROOT,text=True,capture_output=True,timeout=600)
    require(completed.returncode==0,"Independent checkpoint state audit failed: "+completed.stdout+completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])

def benchmark(groups):
    """Recompute published metrics without importing benchmark validators."""
    frozen=load(ROOT/"build/stage5-deeper-benchmark-plan.json");result=load(ROOT/"build/stage5-deeper-benchmark.json");identity=result["identity"]
    manifest=load(ROOT/"data/deeper-9000-v2/manifest.json");selected=load(ROOT/"build/stage5-deeper-training-results.json")["selected"]
    current={r["state_key"]:r for r in groups["development"]};plan={**frozen,"cases":[]}
    for row in frozen["cases"]:
        if row["cohort"]=="ordinary_current_development":
            source=current[row["state_key"]]
            require(row["reference"]==dict(kind="H8_reference_lookup",state_key=row["state_key"],choice_depth=8),"Undeclared H8 reference placeholder")
            require(all(row[k]==source[k] for k in ("state","family","stratum")),"H8 root ownership differs")
            plan["cases"].append({**row,"reference":source["H8_reference"]})
        else:plan["cases"].append(row)
    require(identity["schema"]=="stage5-deeper-benchmark-v1" and identity["source_sha256"]==sha(ROOT/"tools/benchmark-stage5-deeper.py") and identity["plan_sha256"]==config.digest(frozen) and identity["resolved_plan_sha256"]==config.digest(plan),"Benchmark source/plan changed")
    require(identity["selection_sha256"]==sha(ROOT/"build/stage5-deeper-training-results.json") and identity["baseline_dll_sha256"]==sha(ROOT/"build/congklak_native.dll") and identity["rules_digest"]==config.RULES_DIGEST,"Benchmark selection/baseline/rules changed")
    for name,digest in identity["bindings"].items():require(sha(ROOT/name)==digest,"Benchmark binding changed: "+name)
    for name,digest in identity["cohort_sources"].items():require(sha(ROOT/name)==digest,"Benchmark cohort source changed: "+name)
    require(identity["datasets"]==dict(original=pilot.load()["semantic_sha256"],currentH6=pilot.load(campaign.DEFAULT)["semantic_sha256"],deepH8=manifest["semantic_sha256"]),"Benchmark dataset ancestry differs")
    require(frozen["engines"]==["H_v1","prior_B","shallow_counts","shallow_relay","deep_counts","deep_relay","relay_h_none"] and frozen["budgets_ms"]==[25,100] and frozen["max_depth"]==16 and frozen["tt_mb"]==16 and frozen["use_pvs"] is True,"Search allocation changed")
    require(frozen["arena_ms"]==20 and frozen["arena_max_choices"]==512 and frozen["sealed_final_test_consumed"] is False,"Arena allocation/ownership changed")
    expected_comparisons=[dict(id="deep_counts_vs_H",candidate="deep_counts",opponent="H_v1",pairs=36),dict(id="deep_relay_vs_H",candidate="deep_relay",opponent="H_v1",pairs=36),dict(id="deep_relay_vs_deep_counts",candidate="deep_relay",opponent="deep_counts",pairs=36),dict(id="deep_counts_vs_shallow_counts",candidate="deep_counts",opponent="shallow_counts",pairs=36)]
    require(frozen["arena_comparisons"]==expected_comparisons,"Arena comparisons changed")
    cpu=identity["cpu"];require(cpu["protocol"]=="own-process/declared-os-performance-core/v2" and cpu["selected"]==frozen["cpu_declared"] and cpu["mask"]==4 and cpu["selected"]["efficiency_class"]==max(cpu["efficiency_classes"]) and cpu["priority_changed"] is False and cpu["power_settings_changed"] is False,"CPU scope/selection differs")
    require(len(cpu["topology"])==cpu["logical_processors"]==32 and cpu["selected"] in cpu["topology"],"CPU topology declaration differs")
    expected_models={"prior_B":ROOT/"data/training-stage5-action-B-seed31415/policy-model.json"}
    expected_models.update({name:ROOT/selected[key]["model"] for name,key in (("shallow_counts","h6-f43"),("shallow_relay","h6-f127"),("deep_counts","h8-f43"),("deep_relay","h8-f127"),("relay_h_none","h8-f127"))})
    require(set(identity["models"])==set(expected_models),"Unexpected/missing engine model")
    for name,path in expected_models.items():
        model=load(path)["model"];row=identity["models"][name]
        require(row["model_file_sha256"]==sha(path) and row["weights_sha256"]==model["weights_sha256"] and row["input_count"]==model["identity"]["input_count"],"Engine model differs")
        require(row["backend"]=="avx2-double-packed" and row["mode"]==("h-none" if name=="relay_h_none" else "value-none"),"Inference/search mode differs")
        require(row["dll_sha256"]==sha(ROOT/("build/congklak_neural_action.dll" if name=="prior_B" else "build/congklak_neural_relay.dll")),"Engine DLL changed")
    previous=load(ROOT/"build/stage5-benchmark-plan.json");excluded={r["state_key"] for r in load(ROOT/"build/stage5-audit/legacy-exact-aliases.json")["cases"]}
    require(len(excluded)==16 and not excluded&{r["state_key"] for r in frozen["cases"]},"Legacy sealed aliases reintroduced")
    unchanged=[r for r in previous["cases"] if r["state_key"] not in excluded]
    require([(r["state_key"],r["state"],r["family"],r["stratum"],r["cohort"]) for r in unchanged]==[(r["state_key"],r["state"],r["family"],r["stratum"],r["cohort"]) for r in frozen["cases"]],"Position coverage changed after outcomes")
    require(Counter((r["cohort"],r["reference"]["kind"],r["reference"]["choice_depth"]) for r in plan["cases"])==Counter({("original_development","finite_choice_horizon",6):100,("original_development","exact_optimal_margin",None):32,("ordinary_current_development","finite_choice_horizon",8):259}),"Finite/exact/depth reference cohorts mixed")
    protected={}
    for name,folder in (("original",pilot.DEFAULT_PATH),("current",campaign.DEFAULT)):
        data_manifest=pilot.load(folder)
        with closing(sqlite3.connect((folder/data_manifest["directory"]/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
            protected[name]=[(pilot.quotient(pilot.from_key(bytes(k))),s,f) for k,s,f in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples")]
    for row in plan["cases"]:
        q=pilot.quotient(row["state"]);name="original" if row["cohort"]=="original_development" else "current"
        require({s for other,s,f in protected[name] if other==q}=={"development"},"Evaluation aliases its training/sealed partition")
        require(all(not any(q==other and s=="train" for other,s,f in items) for items in protected.values()),"Cross-generation training alias")
    old_families={r["family"] for r in previous["arena_starts"]};starts=plan["arena_starts"]
    require(len(starts)==36 and len({r["family"] for r in starts})==36 and not old_families&{r["family"] for r in starts},"Arena repeats old/within-trial collection families")
    require(Counter(r["stratum"].rsplit("/",1)[1] for r in starts)==Counter(ordinary=12,capture=12,bonus=12),"Arena category allocation differs")
    for row in starts:
        require(all(row[k]==current[row["state_key"]][k] for k in ("state","family","stratum")),"Arena start outside frozen development")
        active=sum(map(sum,row["state"]["pits"]));d=row["state"]["stores"][0]-row["state"]["stores"][1]
        require(active>=25 and abs(d)<active,"Arena starts with settled WDL")
    ledger=ROOT/"data/stage5-deeper-benchmark/work.sqlite3"
    with closing(sqlite3.connect(ledger.as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();require(json.loads(body)==identity and config.digest(identity)==digest,"Benchmark catalog identity differs")
        cases=[]
        for idx,body,digest in db.execute("SELECT idx,body,hash FROM cases ORDER BY idx"):
            value=json.loads(body);require(idx==len(cases) and config.digest(value)==digest,"Corrupt/noncontiguous case ledger");cases.append(value)
        games=[]
        for comparison,pair,seat,body,digest in db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat"):
            value=json.loads(body);require(value["comparison"]==comparison and value["pair"]==pair and value["candidate_seat"]==seat and config.digest(value)==digest,"Corrupt arena ledger");games.append(value)
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='deeper_complete'").fetchone();require(config.digest(json.loads(body))==digest and json.loads(body)==dict(cases=391,games=288,report_sha256=config.digest(result)),"Published result lacks matching catalog completion")
        invocations=[]
        for body,digest in db.execute("SELECT body,hash FROM timing_invocations ORDER BY id"):
            value=json.loads(body);require(config.digest(value)==digest and value["verified"]==dict(group=0,logical=2,mask=4),"Invalid CPU invocation observation")
            require(type(value["cpu"]["original_mask"]) is int and value["cpu"]["original_mask"]&4,"Invalid original CPU permission mask");invocations.append(value)
    require(len(cases)==result["cases"]==391 and len(games)==288 and bool(invocations),"Incomplete experiments")
    buckets=defaultdict(list);strata=defaultdict(list);same_h_depth=0
    def number(x,lo=0,hi=float("inf")):return type(x) in (int,float) and math.isfinite(x) and lo<=x<=hi
    def integer(x,lo=0,hi=2**63-1):return type(x) is int and lo<=x<=hi
    for index,case in enumerate(cases):
        source=plan["cases"][index];p=literal_position(source["state"]);legal=literal.legal_actions(p)
        require(case["index"]==index and case["cpu_verified"]==dict(group=0,logical=2,mask=4) and all(case[k]==source[k] for k in ("state_key","cohort","stratum","reference")),"Case metadata differs")
        q={row["pit"]:row["value_player0"] for row in source["reference"]["action_scores"]};require(list(q)==list(legal),"Reference legal coverage differs")
        sign=1 if literal.normalize(p).controller==0 else -1
        require(set(case["measurements"])==set(case["orders"])=={"25","100"},"Missing timed budget/order")
        for budget_index,ms in enumerate((25,100)):
            names=plan["engines"];order=names[index%7:]+names[:index%7]
            if budget_index:order.reverse()
            require(case["orders"][str(ms)]==order,"Engine timing order changed")
            values=case["measurements"][str(ms)];require(set(values)==set(names),"Missing timed engine")
            for name,value in values.items():
                require(value["stop_reason"] in ("max_depth","time_budget") and integer(value["nodes"]) and number(value["wall_ms"]) and number(value["native_elapsed_ms"]),"Resource failure/invalid work admitted")
                choice=value["chosen"];report=value["report"]
                if report:
                    best=report["best_moves"]
                    require(integer(report["depth"],0,16) and number(report["value_player0"],-98,98) and best==sorted(set(best)) and all(integer(a,0,6) and a in legal for a in best),"Malformed finite report")
                    require(choice==(best[0] if best else None) and all(integer(report[k]) for k in ("nn_evaluations","nn_cache_hits","policy_ordered_nodes","feature_simulations","feature_deposits")),"Choice/report work differs")
                    require(report["policy_ordered_nodes"]==0,"Undeclared policy ordering used")
                    if name in ("H_v1","relay_h_none","prior_B","shallow_counts","deep_counts"):require(report["feature_simulations"]==report["feature_deposits"]==0,"Count/H evaluation incurred relay work")
                else:require(choice is None,"Recommendation without complete report")
                require(choice is None or integer(choice,0,6) and choice in legal,"Illegal saved recommendation")
                require(value["reference_optimal"]==(choice in source["reference"]["best_moves"] if choice is not None else None) and value["reference_regret"]==(sign*(source["reference"]["target_player0"]-q[choice]) if choice is not None else None),"Choice accuracy/regret differs")
                buckets[(case["cohort"],str(ms),source["reference"]["kind"],name)].append(value);strata[(case["cohort"],str(ms),case["stratum"],name)].append(value)
            a=values["H_v1"]["report"];b=values["relay_h_none"]["report"]
            if a and b and a["depth"]==b["depth"]:
                require(a["value_player0"]==b["value_player0"] and a["best_moves"]==b["best_moves"],"H-only adapter changes fixed-horizon minimax");same_h_depth+=1
    def aggregate(values):
        done=[v for v in values if v["chosen"] is not None]
        return dict(positions=len(values),no_completed_recommendation=len(values)-len(done),reference_optimal_accuracy=statistics.mean(v["reference_optimal"] for v in done) if done else None,mean_reference_regret_seeds=statistics.mean(v["reference_regret"] for v in done) if done else None,mean_completed_depth=statistics.mean(v["report"]["depth"] for v in done) if done else None,median_wall_ms=statistics.median(v["wall_ms"] for v in values),max_wall_ms=max(v["wall_ms"] for v in values))
    require(result["budget_results"]=={"/".join(k):aggregate(v) for k,v in buckets.items()} and result["strata"]=={"/".join(k):aggregate(v) for k,v in strata.items()},"Root/stratum summaries differ")
    arenas={};families={};replayed=0
    for comparison in plan["arena_comparisons"]:
        group=[g for g in games if g["comparison"]==comparison["id"]]
        require({(g["pair"],g["candidate_seat"]) for g in group}=={(pair,seat) for pair in range(36) for seat in (0,1)} and len(group)==72,"Missing/duplicated arena seat")
        for game in group:
            start=starts[game["pair"]];require(game["cpu_verified"]==dict(group=0,logical=2,mask=4) and game["start"]==start["state"] and game["family"]==start["family"],"Arena start/CPU differs")
            p=literal_position(game["start"]);require(isinstance(game["moves"],list) and len(game["moves"])<=512,"Choice cap exceeded")
            for move in game["moves"]:
                require(integer(move,0,6) and not literal.normalize(p).terminal and move in literal.legal_actions(p),"Illegal move/choice after terminal");p=literal.apply(p,move).position
            final={"pits":[list(p.counts[:7]),list(p.counts[8:15])],"stores":[p.counts[7],p.counts[15]],"controller":p.controller};require(game["final"]==final,"Literal final board differs")
            terminal=literal.normalize(p).terminal;margin=(p.counts[7]-p.counts[15])*(1 if game["candidate_seat"]==0 else -1);score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
            require(game["status"]==("terminal" if terminal else "unfinished") and game["candidate_final_margin"]==(margin if terminal else None) and game["candidate_match_score"]==score,"Literal arena outcome differs")
            require(isinstance(game["fallbacks"],dict) and set(game["fallbacks"]).issubset({comparison["candidate"],comparison["opponent"]}) and all(integer(n,0,len(game["moves"])) for n in game["fallbacks"].values()) and sum(game["fallbacks"].values())<=len(game["moves"]),"Invalid fallback counts");replayed+=1
        paired=[];margins=[];details=[]
        for pair in range(36):
            two=[g for g in group if g["pair"]==pair]
            if all(g["status"]=="terminal" for g in two):
                paired.append(statistics.mean(g["candidate_match_score"] for g in two));margins.append(statistics.mean(g["candidate_final_margin"] for g in two));details.append(dict(pair=pair,family=starts[pair]["family"],primary_paired_margin=margins[-1],secondary_wdl_score=paired[-1]))
        fallbacks={name:sum(g["fallbacks"].get(name,0) for g in group) for name in (comparison["candidate"],comparison["opponent"])};ci=interval(margins)
        gate=len(margins)==36 and not sum(fallbacks.values()) and ci[0]>0
        arenas[comparison["id"]]=dict(pairs=36,terminal_games=sum(g["status"]=="terminal" for g in group),unfinished_games=sum(g["status"]!="terminal" for g in group),complete_pairs=len(paired),candidate_wins=sum(g["candidate_match_score"]==1 for g in group),candidate_draws=sum(g["candidate_match_score"]==.5 for g in group),candidate_losses=sum(g["candidate_match_score"]==0 for g in group),paired_mean_score=statistics.mean(paired) if paired else None,paired_bootstrap_95_interval=interval(paired),primary_paired_mean_final_margin_seeds=statistics.mean(margins) if margins else None,primary_margin_bootstrap_95_interval=ci,primary_strength_gate_passed=gate,fallback_counts=fallbacks)
        families[comparison["id"]]=details
    require(result["arenas"]==arenas and result["sealed_final_test_consumed"] is False and result["energy_measured"] is False,"Primary margin/bootstrap/gates/scope differ")
    (WORK/"paired-margins.json").write_text(json.dumps(families,indent=2)+"\n")
    return dict(cases_independently_recomputed=391,literal_games_replayed=replayed,known_exact_diagnostics=32,original_H6_proxy_roots=100,current_ordinary_H8_proxy_roots=259,
                legacy_sealed_aliases_excluded=16,training_or_sealed_root_aliases=0,H_only_adapter_equal_depth_checks=same_h_depth,distinct_new_arena_families=36,
                cpu=cpu,truthful_timing_invocations=len(invocations),primary_seed_margin_predeclared=True,arenas=arenas,sealed_target_bodies_read=False,energy_measured=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--without-benchmark",action="store_true");parser.add_argument("--collect-tests",action="store_true");parser.add_argument("--test-group",choices=("cuda","benchmark","core"));args=parser.parse_args()
    if args.test_group:return run_test_group(args.test_group)
    if args.collect_tests:collect_tests();return 0
    result=dict(schema="stage5-deeper-independent-audit-v2",recorded_at_utc=datetime.now(timezone.utc).isoformat(),passed=False,benchmark_audit_complete=False)
    try:
        tests=load(WORK/"tests.json");require(tests["passed"] and tests["tests_run"]==13 and tests["source_sha256"]==sha(ROOT/"tests/stage5_deeper_audit_acceptance.py") and not tests["failures"]+tests["errors"]+tests["skips"],"Independent acceptance failed/incomplete/stale")
        require(tests["covered_methods"]==unittest.defaultTestLoader.getTestCaseNames(IndependentDeeperAcceptance) and len(tests["covered_methods"])==len(set(tests["covered_methods"]))==13,"Independent method coverage differs")
        for group in tests["groups"]:require(sha(ROOT/group["path"])==group["sha256"],"Independent group evidence changed")
        result["independent_acceptance"]=tests;result["tests_run"]=tests["tests_run"];result["frozen_prior_inputs"]=frozen_prior()
        result["migration"]=migration();result["teacher"],groups=teacher();result["evaluation"]=evaluation(groups)
        if not args.without_benchmark:result["benchmark"]=benchmark(groups);result["benchmark_audit_complete"]=True
        result["passed"]=True
    except Exception as exc:result["error"]={"type":type(exc).__name__,"message":str(exc)}
    names=("tests/stage5_deeper_audit_acceptance.py","tools/stage5-deeper-audit.py","congklak/deeper_teacher.py","congklak/deeper_teacher_v2.py","congklak/training_v5.py","congklak/training_v6.py","congklak/training_v7.py","congklak/features_relay.py","congklak/neural_relay.py","native/neural_relay.cpp","build/congklak_neural_relay.dll","congklak/timing_v2.py","tools/stage5-deeper-training.py","tools/stage5-deeper-training-v2.py","tools/stage5-deeper-training-v3.py","tools/benchmark-stage5-deeper.py","build/stage5-deeper-training-plan.json","build/stage5-deeper-training-plan-v2.json","build/stage5-deeper-training-plan-v3.json","build/stage5-deeper-training-results.json","build/stage5-deeper-training-verification.json","build/stage5-deeper-benchmark-plan.json","build/stage5-deeper-benchmark.json","build/stage5-deeper-audit/tests.json","build/stage5-deeper-audit/prior-eight-tests.json","build/stage5-deeper-audit/migration-independent.json","build/stage5-deeper-audit/v1-teacher-recovery-failure.json","data/deeper-9000-v2/manifest.json","data/deeper-9000-v2/migration.json")
    result["inputs"]={name:sha(ROOT/name) for name in names if (ROOT/name).exists()}
    for name in ("build/stage5-deeper-audit/prefreeze-benchmark-tests.json","build/stage5-deeper-audit/teacher-independent.json",
                 "build/progress/run-109255705f4d4cd18ee8c973d5bc1233.json","build/progress/run-109255705f4d4cd18ee8c973d5bc1233.log",
                 "build/stage5-deeper-audit/backend-label-failure.json","build/progress/run-16641a184e244f00a0c54f608c9ee6f4.json",
                 "build/progress/run-16641a184e244f00a0c54f608c9ee6f4.log"):
        if (ROOT/name).exists():result["inputs"][name]=sha(ROOT/name)
    for file in TEST_GROUPS:
        path=WORK/file
        if path.exists():result["inputs"][str(path.relative_to(ROOT)).replace("\\","/")]=sha(path)
    if (ROOT/"build/stage5-deeper-training-results.json").exists():
        for row in load(ROOT/"build/stage5-deeper-training-results.json")["rows"]:
            for path in (ROOT/row["model"],(ROOT/row["model"]).parent/"result.json"):
                result["inputs"][str(path.relative_to(ROOT)).replace("\\","/")]=sha(path)
    (WORK/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(event="deeper_independent_audit_complete",passed=result["passed"],benchmark_audit_complete=result["benchmark_audit_complete"],error=result.get("error"))),flush=True)
    return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
