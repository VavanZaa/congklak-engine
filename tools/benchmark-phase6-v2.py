"""Pinned equal-deadline Phase6 choices and paired terminal-margin arenas."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack,closing
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import struct
import sys
import time
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak import config,pilot,phase6_common as common,phase6_store
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative,Model,sha
from congklak.phase6_inference import BatchEvaluator
from congklak.puct import search as puct_search,PROTOCOL as PUCT_PROTOCOL
from congklak.timing_v2 import PerformanceCore
DEFAULT_RUN=ROOT/"data/phase6-benchmark-v1"
DEFAULT_OUTPUT=ROOT/"build/phase6-benchmark.json"
IMPLEMENTATION=ROOT/"build/phase6-implementation-v2.json"
ROLE_PATHS={"warm_trainer":"congklak/phase6_training.py","warm_tool":"tools/phase6-training.py",
            "update_trainer":"congklak/phase6_training_v2.py","update_tool":"tools/phase6-training-v2.py",
            "benchmark":"tools/benchmark-phase6-v2.py","speed":"tools/phase6-inference-speed.py","selfplay":"congklak/phase6_selfplay.py",
            "store":"congklak/phase6_store.py","puct":"congklak/puct.py","inference":"congklak/phase6_inference.py","common":"congklak/phase6_common.py","analysis":"congklak/phase6_analysis.py"}
SCIENCE_REQUIRED=("build/phase6-plan.json","build/phase6-protected-registry.json","build/training-runtime.json",
                  "build/congklak_native.dll","build/congklak_neural_relay.dll","congklak/config.py","congklak/native.py",
                  "congklak/neural_relay.py","congklak/features_relay.py","congklak/features_v3.py","congklak/timing_v2.py",
                  "congklak/training_v3.py","congklak/deeper_teacher_v2.py")
SOURCE_VALUE="29f90512e59ebaeb93eedf74d247684161e6b211a9b21640dc4f9333d213d69f"

def require(condition,message):
    if not condition:raise ValueError(message)
def integer(value,lo=0,hi=2**63-1):return type(value) is int and lo<=value<=hi
def numeric(value,lo=0,hi=float("inf")):return type(value) in (int,float) and math.isfinite(value) and lo<=value<=hi

def implementation_manifest(path=IMPLEMENTATION):
    """No Torch import or GPU allocation is needed to reject source drift."""
    body=json.loads(Path(path).read_text(encoding="utf-8"))
    require(body.get("schema")=="phase6-implementation-v2" and config.digest({k:v for k,v in body.items() if k!="sha256"})==body.get("sha256"),"Corrupt Phase6 implementation manifest")
    require(isinstance(body.get("roles"),dict) and all(body["roles"].get(k)==p for k,p in ROLE_PATHS.items()),"Phase6 implementation roles differ")
    sources=body.get("sources");require(isinstance(sources,dict) and set(ROLE_PATHS.values())<=set(sources),"Incomplete Phase6 producer source mapping")
    science=body.get("science_inputs");require(isinstance(science,dict) and set(SCIENCE_REQUIRED)<=set(science),"Incomplete mandatory Phase6 science input mapping")
    for role,path in body["roles"].items():require(path in sources,"Unpinned producer role "+role)
    for table in (sources,science):
        require(isinstance(table,dict),"Invalid implementation science input mapping")
        for file,digest in table.items():
            candidate=(ROOT/file).resolve();require(candidate.is_relative_to(ROOT.resolve()) and candidate.is_file() and sha(candidate)==digest,"Pinned Phase6 source/input differs: "+file)
    return body

def value_digest(model):
    checksum=__import__("hashlib").sha256()
    def numbers(item):
        if isinstance(item,list):
            for part in item:yield from numbers(part)
        else:yield item
    for name in sorted(k for k in model.weights if k.startswith("layers.")):
        values=list(numbers(model.weights[name]));checksum.update(name.encode());checksum.update(struct.pack("<"+"f"*len(values),*values))
    return checksum.hexdigest()

def admit_training(modelpath,stage,implementation,plan,initial,sp_witness=None):
    """Bind JSON weights to complete catalog metadata and immutable PT checksums.

    This stdlib gate does not deserialize Torch tensors. The independent audit
    separately checks that the PT tensors, optimizer and RNG payloads agree.
    """
    require(stage in ("warm","selfplay"),"Unknown admitted training stage")
    path=Path(modelpath).resolve();run=path.parent;model=Model(path);result_path=run/"result.json";result=json.loads(result_path.read_text(encoding="utf-8"));ident=model.identity
    settings=plan["warm"] if stage=="warm" else plan["update"];steps=settings["steps"];batch=settings["batch_size"];every=settings["checkpoint_every"]
    role="warm_trainer" if stage=="warm" else "update_trainer";expected_source=implementation["sources"][implementation["roles"][role]]
    schema="phase6-policy-warmstart-and-selfplay-training-v1" if stage=="warm" else "phase6-selfplay-deferred-cuda-training-v2"
    require(ident.get("schema")==schema and ident.get("stage")==stage and ident.get("source_sha256")==expected_source,"Wrong training producer/schema/stage")
    require(model.input_count==127 and model.width==64 and ident.get("rules_digest")==config.RULES_DIGEST,"Wrong candidate architecture/rules")
    require(result.get("identity")==ident and result.get("status")=="complete" and result.get("sealed_final_test_consumed") is False,"Incomplete or incompatible candidate result")
    require(integer(result.get("steps_completed")) and result["steps_completed"]==result.get("optimizer_steps_target")==ident.get("steps")==steps,"Incomplete optimizer target")
    for key in ("batch_size","lr","seed","checkpoint_every"):require(ident.get(key)==settings[key] and (key=="lr" or type(ident[key]) is int),"Training budget differs: "+key)
    require(ident.get("device")=="cuda" and ident.get("amp") is False and ident.get("oracle_training") is False and ident.get("sealed_final_test_consumed") is False and ident.get("loader_workers")==0,"Candidate numeric/data contract differs")
    require(result.get("parameters")==12872 and result.get("trainable_parameters")==ident.get("trainable_parameters")== (455 if stage=="warm" else 12872),"Wrong optimizer parameter ownership")
    require(result.get("weights_sha256")==model.id and result.get("value_weights_sha256")==value_digest(model),"Selected model/value digest differs from result")
    require(ident.get("initial_model_file_sha256")==initial.file_sha256 and ident.get("initial_weights_sha256")==initial.id and ident.get("initial_value_weights_sha256")==value_digest(initial),"Candidate initial model lineage differs")
    teacher=json.loads((ROOT/"data/deeper-9000-v2/manifest.json").read_text(encoding="utf-8"))
    require(ident.get("dataset_manifest_sha256")==config.digest(teacher) and ident.get("dataset_semantic_sha256")==teacher["semantic_sha256"],"Teacher manifest lineage differs")
    require(ident.get("policy_temperature_seeds")==4 and ident.get("policy_perspective")=="Normalized root actor irrespective of successor controller","Policy actor/temperature differs")
    require(all(integer(result.get(k)) for k in ("draws","teacher_draws","selfplay_draws")) and result["draws"]==steps*batch and result.get("teacher_draws")==steps*batch*(1 if stage=="warm" else .5) and result.get("selfplay_draws")== (0 if stage=="warm" else steps*batch//2),"Committed training draws differ")
    require(isinstance(ident.get("source_dependencies_sha256"),dict),"Missing trainer dependency pins")
    dependencies={"congklak.config","congklak.deeper_teacher_v2","congklak.features_relay","congklak.training_v3","congklak.neural_relay"}
    if stage=="selfplay":dependencies.add("congklak.phase6_training")
    require(set(ident["source_dependencies_sha256"])==dependencies,"Trainer dependency-module set differs")
    for name,digest in ident["source_dependencies_sha256"].items():
        candidate=(ROOT/(name.replace(".","/")+".py")).resolve();require(candidate.is_relative_to(ROOT.resolve()) and candidate.is_file() and sha(candidate)==digest,"Trainer dependency differs: "+name)
    if stage=="warm":
        require(value_digest(model)==SOURCE_VALUE and all(ident.get(k) is None for k in ("selfplay_loader_sha256","selfplay_manifest_sha256","selfplay_rows_sha256")),"Warm frozen value/selfplay ancestry differs")
    else:
        require(sp_witness is not None and ident.get("selfplay_manifest_sha256")==sp_witness["manifest_sha256"] and ident.get("selfplay_rows_sha256")==sp_witness["training_rows_sha256"] and ident.get("selfplay_loader_sha256")==implementation["sources"][implementation["roles"]["selfplay"]],"Update selfplay lineage differs")
        require(ident.get("warm_witness_implementation_sha256")==implementation["sources"][implementation["roles"]["warm_trainer"]] and ident.get("resource_admission")=="CPU flags and full teacher/SP/policy/feature validation before CUDA context; mathematical protocol unchanged","Deferred-CUDA update witness/resource contract differs")
    catalog=run/"catalog.sqlite3";entries=[];scores=[];selected=None
    with closing(sqlite3.connect(catalog.as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");require(db.execute("PRAGMA quick_check").fetchone()[0]=="ok","Corrupt training catalog")
        configs=db.execute("SELECT body,hash FROM config").fetchall();require(len(configs)==1,"Multiple/missing training configurations")
        conf=json.loads(configs[0][0]);require(config.digest(conf)==configs[0][1] and conf==ident,"Model/catalog identity differs")
        rows=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        require([row[0] for row in rows]==list(range(every,steps+1,every)),"Incomplete or altered scheduled checkpoint catalog")
        require(len({row[1] for row in rows})==len(rows),"Aliased checkpoint filenames")
        for step,name,digest,body,body_hash in rows:
            checkpoint=(run/name).resolve();meta=json.loads(body)
            require(checkpoint.parent==run and checkpoint.is_file() and sha(checkpoint)==digest,"Missing, escaped or damaged checkpoint file")
            require(config.digest(meta)==body_hash and meta.get("step")==step and meta.get("identity_sha256")==config.digest(ident),"Checkpoint metadata differs")
            entries.append(dict(step=step,name=name,file_sha256=digest,metadata_sha256=body_hash,weights_sha256=meta["weights_sha256"]))
            ce=meta["metrics"]["policy"]["uniform_stratum_teacher_cross_entropy"];require(numeric(ce),"Nonfinite checkpoint selection metric");scores.append((step,ce))
            if step==result.get("best_step"):selected=(entries[-1],meta)
    expected_step=min(scores,key=lambda row:row[1])[0] if stage=="warm" else steps
    require(selected is not None and result.get("best_step")==model.envelope["model"].get("step")==expected_step,"Wrong scheduled/final checkpoint selection")
    require(selected[0]["weights_sha256"]==model.id,"Export does not match selected catalog model digest")
    metrics={k:v for k,v in selected[1]["metrics"].items() if k!="last_training_batch_loss"}
    require(result.get("selected_development")==metrics,"Selected development metrics differ from checkpoint witness")
    return dict(stage=stage,model_file_sha256=model.file_sha256,weights_sha256=model.id,value_weights_sha256=value_digest(model),result_sha256=sha(result_path),identity=ident,
                selected_step=expected_step,checkpoints=entries,catalog_witness_sha256=config.digest(entries),selected_checkpoint=selected[0],stdlib_only=True,torch_payload_verified_by="separate independent audit")

def admit_selfplay(path,warm,implementation,plan):
    from congklak.phase6_selfplay import load_training_rows
    manifest,rows=load_training_rows(path);ident=manifest["identity"]
    require(manifest.get("complete") is True and manifest.get("training_admission")=="ready" and len(rows)>0,"Incomplete/unadmitted selfplay source")
    require(ident["model_file_sha256"]==warm.file_sha256 and ident["model_weights_sha256"]==warm.id,"Selfplay source is not selected warm model")
    require(ident["settings"]==plan["selfplay"] and ident["starts"]==plan["selfplay_starts"] and ident["plan_sha256"]==config.digest(plan),"Selfplay generation plan differs")
    for field,role in (("source_sha256","selfplay"),("puct_sha256","puct"),("common_sha256","common"),("store_sha256","store")):
        require(ident[field]==implementation["sources"][implementation["roles"][role]],"Selfplay producer lineage differs: "+role)
    require(ident["evaluator"]["source_sha256"]==implementation["sources"][implementation["roles"]["inference"]] and ident["evaluator"]["model_file_sha256"]==warm.file_sha256 and ident["evaluator"]["model_weights_sha256"]==warm.id,"Selfplay evaluator lineage differs")
    return dict(manifest_sha256=config.digest(manifest),manifest_file_sha256=sha(Path(path)/"manifest.json"),training_rows_sha256=config.digest(rows),records_semantic_sha256=manifest["records_semantic_sha256"],game_ledger_semantic_sha256=manifest["game_ledger_semantic_sha256"],summary=manifest["summary"],identity=ident)

def model_witnesses(plan,implementation):
    source=Model(ROOT/plan["source_model"]);require(source.file_sha256==plan["source_model_sha256"],"Frozen H8 source model differs")
    warm=Model(common.WARM);warm_witness=admit_training(common.WARM,"warm",implementation,plan,source)
    sp=admit_selfplay(common.SELFPLAY,warm,implementation,plan)
    update_witness=admit_training(common.UPDATE,"selfplay",implementation,plan,warm,sp)
    return dict(warm=warm_witness,updated=update_witness,selfplay=sp)
def emit(body):
    print(config.canonical(body),flush=True)
    barrier=os.environ.get("PHASE6_BLOCK_AFTER_CASE")
    if body.get("event")=="phase6_case_committed" and barrier and int(barrier)==body["completed"]:
        print(config.canonical(dict(event="phase6_case_commit_barrier",completed=body["completed"])),flush=True)
        while True:time.sleep(.05)

def compact_puct(report):
    keys=("chosen","visits","policy","root_priors","estimated_value_player0","action_mean_value_player0","root_expanded","completed_simulations","requested_simulations","nodes","max_nodes","stop_reason","proof_status","work","usable_decision",
          "protocol","rules_digest","value_perspective","first_play_urgency","unvisited_root_q_player0","c_puct")
    return {key:report[key] for key in keys}

def decide(engine,position,ms,plan):
    started=time.perf_counter();kind,obj=engine
    if kind=="puct":
        report=puct_search(position,obj,simulations=plan["puct"]["simulations"],time_ms=ms,max_nodes=plan["puct"]["max_nodes"],c_puct=1.5,root_noise=None,memory_mb=8192)
        result=dict(kind=kind,report=compact_puct(report),stop_reason=report["stop_reason"],engine_elapsed_ms=report["timing"]["elapsed_ms"],nodes=report["nodes"],
                    usable_decision=report["usable_decision"],chosen=report["chosen"] if report["usable_decision"] else None,completed_depth=None,completed_simulations=report["completed_simulations"])
    else:
        if kind=="h":answer=obj.analyze(flatten(position),actor=position["controller"],total=98,max_depth=plan["max_depth"],time_ms=ms,node_budget=0,memory_mb=8192,tt_mb=plan["tt_mb"],use_pvs=plan["use_pvs"],keep_history=False)
        else:answer=obj.neural_analyze(position,max_depth=plan["max_depth"],time_ms=ms,node_budget=0,memory_mb=8192,tt_mb=plan["tt_mb"],use_pvs=plan["use_pvs"])
        raw=answer["report"];report=None;usable=bool(raw and raw["choice_depth"]>=1 and raw["best_moves"])
        if raw:
            report=dict(depth=raw["choice_depth"],best_moves=raw["best_moves"],value_player0=raw["depth_result"]["value_player0"] if kind=="h" else raw["value_units"]/256,
                        proof_status="finite_horizon_result" if kind=="h" else raw["proof_status"],nn_evaluations=raw.get("nn_evaluations",0),
                        feature_simulations=raw.get("feature_simulations",0),feature_deposits=raw.get("feature_deposits",0),deposits=raw["deposits"])
        result=dict(kind=kind,report=report,stop_reason=answer["stop_reason"],engine_elapsed_ms=answer["elapsed_ms"],nodes=answer["nodes"],usable_decision=usable,
                    chosen=raw["best_moves"][0] if usable else None,completed_depth=raw["choice_depth"] if raw else None,completed_simulations=None)
    # Report admission is part of the whole wrapper clock for every engine.
    elapsed=(time.perf_counter()-started)*1000;result.update(wall_ms=elapsed,requested_ms=ms,overshoot_ms=max(0.0,elapsed-ms))
    validate_decision(result,position,ms,plan)
    elapsed=(time.perf_counter()-started)*1000;result.update(wall_ms=elapsed,overshoot_ms=max(0.0,elapsed-ms))
    return result

def validate_decision(value,p,ms,plan):
    fields={"kind","report","stop_reason","engine_elapsed_ms","nodes","usable_decision","chosen","completed_depth","completed_simulations","wall_ms","requested_ms","overshoot_ms"}
    require(isinstance(value,dict) and set(value)==fields,"Unknown decision fields")
    require(value["kind"] in ("h","relay","puct") and value["requested_ms"]==ms and integer(value["nodes"]),"Invalid decision identity/work")
    require(numeric(value["wall_ms"]) and numeric(value["engine_elapsed_ms"]) and numeric(value["overshoot_ms"]),"Invalid decision timing")
    require(value["wall_ms"]>0 and value["engine_elapsed_ms"]<=value["wall_ms"]+1,"Engine elapsed exceeds whole wrapper clock")
    require(value["overshoot_ms"]==max(0.0,value["wall_ms"]-ms),"Incorrect whole-decision overshoot")
    require(type(value["usable_decision"]) is bool,"Invalid recommendation admission")
    legal=[a for a,c in enumerate(p["pits"][p["controller"]]) if c];report=value["report"]
    margin=p["stores"][0]-p["stores"][1];remaining=sum(map(sum,p["pits"]));lower=margin-remaining-1e-4;upper=margin+remaining+1e-4
    require(bool(legal),"Strength decision requires a normalized nonterminal root")
    if value["stop_reason"]=="time_budget":require(value["wall_ms"]>=ms,"Early timeout has an incorrect decision clock")
    if value["kind"]=="puct":
        fields={"chosen","visits","policy","root_priors","estimated_value_player0","action_mean_value_player0","root_expanded","completed_simulations","requested_simulations","nodes","max_nodes","stop_reason","proof_status","work","usable_decision",
                "protocol","rules_digest","value_perspective","first_play_urgency","unvisited_root_q_player0","c_puct"}
        require(isinstance(report,dict) and set(report)==fields,"Invalid saved PUCT report")
        require(report["protocol"]==PUCT_PROTOCOL and report["rules_digest"]==config.RULES_DIGEST and report["value_perspective"]=="fixed_player0_seeds" and report["first_play_urgency"]=="parent_neural_bootstrap" and report["c_puct"]==1.5,"PUCT scientific identity differs")
        require(report["requested_simulations"]==plan["puct"]["simulations"] and numeric(report["unvisited_root_q_player0"],lower,upper),"PUCT target/bootstrap physical interval differs")
        require(integer(report["completed_simulations"],0,plan["puct"]["simulations"]) and report["completed_simulations"]==value["completed_simulations"],"Invalid simulation count")
        require(report["max_nodes"]==plan["puct"]["max_nodes"] and integer(report["nodes"],1,report["max_nodes"]) and report["nodes"]==value["nodes"],"Invalid PUCT node bound")
        require(isinstance(report["visits"],list) and len(report["visits"])==7 and all(integer(n) for n in report["visits"]),"Invalid visit vector")
        visits=report["visits"];total=sum(visits);require(total==report["completed_simulations"],"Visits do not match completed edge backups")
        require(all(visits[a]==0 for a in range(7) if a not in legal),"Illegal action has visits")
        require(type(report["root_expanded"]) is bool and report["proof_status"]=="monte_carlo_estimate" and numeric(report["estimated_value_player0"],lower,upper),"Invalid MCTS claim/value physical interval")
        require(not total or report["root_expanded"],"Backups without root expansion")
        for key in ("policy","root_priors"):
            require(isinstance(report[key],list) and len(report[key])==7 and all(numeric(v,0,1) for v in report[key]),"Invalid policy vector")
            require(all(report[key][a]==0 for a in range(7) if a not in legal),"Illegal action has policy mass")
        priors=report["root_priors"]
        require(math.isclose(sum(priors),1 if report["root_expanded"] else 0,abs_tol=1e-12),"Root prior normalization differs")
        expected=[v/total for v in visits] if total else priors if report["root_expanded"] else [1/len(legal) if a in legal else 0 for a in range(7)]
        require(report["policy"]==expected,"Visit policy differs")
        require(report["chosen"]==max(legal,key=lambda a:(visits[a],priors[a] if not total else 0,-a)),"PUCT choice differs")
        means=report["action_mean_value_player0"];require(isinstance(means,list) and len(means)==7,"Invalid action means")
        for a,mean in enumerate(means):require(numeric(mean,lower,upper) if visits[a] else mean is None,"Invalid visited/unvisited mean physical interval")
        if total:require(math.isclose(sum(visits[a]*means[a] for a in legal if visits[a])/total,report["estimated_value_player0"],abs_tol=1e-10),"FixedP0 backup mean differs")
        require(report["usable_decision"] is (total>0) and value["usable_decision"]==report["usable_decision"],"Prior-only fallback hidden")
        require(value["chosen"]==(report["chosen"] if total else None) and value["completed_depth"] is None,"PUCT external choice differs")
        work=report["work"];require(isinstance(work,dict) and set(work)=={"transitions","deposits","feature_simulations","feature_deposits","neural_evaluations","neural_calls","inference_ms","root_expansions","total_deposits"},"Invalid PUCT work")
        require(all(integer(v) for k,v in work.items() if k!="inference_ms") and numeric(work["inference_ms"]),"Invalid work counters")
        require(work["root_expansions"]==int(report["root_expanded"]) and work["transitions"]==report["nodes"]-1 and work["total_deposits"]==work["deposits"]+work["feature_deposits"],"PUCT work conservation differs")
        require(value["stop_reason"]==report["stop_reason"] and value["stop_reason"] in ("time_budget","node_budget","simulations"),"Failed/interrupted PUCT decision cannot enter strength")
        if value["stop_reason"]=="simulations":require(total==plan["puct"]["simulations"],"Incomplete target labeled complete")
        if value["stop_reason"]=="node_budget":require(report["nodes"]==report["max_nodes"],"Early node-budget stop")
    else:
        require(value["completed_simulations"] is None and value["stop_reason"] in ("max_depth","time_budget"),"Failed/interrupted alphabeta cannot enter strength")
        if report:
            require(set(report)=={"depth","best_moves","value_player0","proof_status","nn_evaluations","feature_simulations","feature_deposits","deposits"},"Unknown alphabeta summary")
            require(integer(report["depth"],0,plan["max_depth"]) and report["depth"]==value["completed_depth"] and numeric(report["value_player0"],lower,upper),"Invalid depth/value physical interval")
            best=report["best_moves"];require(isinstance(best,list) and best==sorted(set(best)) and all(integer(a,0,6) and a in legal for a in best),"Invalid legal best moves")
            require(value["usable_decision"] is bool(report["depth"]>=1 and best) and value["chosen"]==(best[0] if value["usable_decision"] else None),"Depth recommendation differs")
            require(report["proof_status"]==("finite_horizon_result" if value["kind"]=="h" else "finite_horizon_neural_estimate"),"Invalid search proof claim")
            require(all(integer(report[k]) for k in ("nn_evaluations","feature_simulations","feature_deposits","deposits")) and report["feature_deposits"]<=report["deposits"],"Invalid alphabeta work")
        else:require(value["completed_depth"] is None and value["chosen"] is None and value["usable_decision"] is False,"Choice without completed report")

def expected_order(index,budget_index,plan):
    names=list(plan["engines"]);rotation=index%len(names);names=names[rotation:]+names[:rotation]
    return names[::-1] if budget_index else names
def verified(plan):return dict(group=plan["cpu_declared"]["group"],logical=plan["cpu_declared"]["logical"],mask=1<<plan["cpu_declared"]["logical"])

def validate_case(body,index,plan):
    require(integer(index,0,len(plan["cases"])-1),"Unknown case index");row=plan["cases"][index]
    require(set(body)=={"index","state_key","cohort","stratum","reference","orders","measurements","cpu_verified"},"Unknown case fields")
    require(body["index"]==index and all(body[k]==row[k] for k in ("state_key","cohort","stratum","reference")) and body["cpu_verified"]==verified(plan),"Case ownership/reference/CPU differs")
    require(set(body["orders"])==set(body["measurements"])=={str(ms) for ms in plan["budgets_ms"]},"Case budgets differ")
    q={a["pit"]:a["value_player0"] for a in row["reference"]["action_scores"]};sign=1 if row["state"]["controller"]==0 else -1
    for bi,ms in enumerate(plan["budgets_ms"]):
        values=body["measurements"][str(ms)];require(set(values)==set(plan["engines"]) and body["orders"][str(ms)]==expected_order(index,bi,plan),"Engine order/coverage differs")
        for name,value in values.items():
            require(set(value)=={"decision","reference_optimal","reference_regret"},"Unknown case measurement")
            answer=value["decision"];require(answer["kind"]==("h" if name=="H_v1" else "relay" if name=="H8_relay_CPU" else "puct"),"Engine family differs")
            validate_decision(answer,row["state"],ms,plan);choice=answer["chosen"]
            if choice is None:require(value["reference_optimal"] is None and value["reference_regret"] is None,"Fabricated fallback reference")
            else:require(choice in q and type(value["reference_optimal"]) is bool and value["reference_optimal"]==(choice in row["reference"]["best_moves"]) and value["reference_regret"]==sign*(row["reference"]["target_player0"]-q[choice]),"Choice regret/agreement differs")

def replay_game(game,comparison,pair,seat,plan,native):
    require(integer(pair,0,comparison["pairs"]-1) and integer(seat,0,1),"Unknown game key")
    fields={"comparison","pair","candidate_seat","start","family","moves","decisions","final","status","candidate_final_margin","candidate_match_score","fallbacks","cpu_verified"}
    require(set(game)==fields,"Unknown game fields");start=plan["arena_starts"][pair]
    require(game["comparison"]==comparison["id"] and game["pair"]==pair and game["candidate_seat"]==seat and game["start"]==start["state"] and game["family"]==start["family"] and game["cpu_verified"]==verified(plan),"Game ownership/CPU differs")
    require(isinstance(game["moves"],list) and isinstance(game["decisions"],list) and len(game["moves"])==len(game["decisions"])<=plan["arena_max_choices"],"Game choice count differs")
    p=game["start"];fallbacks=Counter()
    for action,saved in zip(game["moves"],game["decisions"]):
        normal=native.normalize(flatten(p),actor=p["controller"],total=98);require(not normal["terminal"],"Move after terminal")
        p=normal["position"];name=comparison["candidate"] if p["controller"]==seat else comparison["opponent"]
        require(set(saved)=={"engine","actor","fallback","decision"} and saved["engine"]==name and saved["actor"]==p["controller"] and type(saved["fallback"]) is bool,"Game decision perspective differs")
        answer=saved["decision"];validate_decision(answer,p,plan["arena_ms"],plan)
        require(answer["kind"]==("h" if name=="H_v1" else "relay" if name=="H8_relay_CPU" else "puct"),"Arena engine family differs")
        fallback=not answer["usable_decision"];require(saved["fallback"]==fallback,"Fallback counter hides unusable prior/depth")
        expected=normal["legal_actions"][0] if fallback else answer["chosen"]
        require(integer(action,0,6) and action in normal["legal_actions"] and action==expected,"Illegal or mismatched saved move")
        if fallback:fallbacks[name]+=1
        t=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(t.get("ok"),"Saved transition failed");p=t["position"]
    require(game["final"]==p and game["fallbacks"]==dict(fallbacks),"Game final/fallback replay differs")
    terminal=not any(map(sum,p["pits"]));margin=(p["stores"][0]-p["stores"][1])*(1 if seat==0 else -1);score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
    require(game["status"]==("terminal" if terminal else "unfinished") and game["candidate_final_margin"]==(margin if terminal else None) and game["candidate_match_score"]==score,"Saved terminal score differs")

def interval(values,plan):
    if not values:return None
    rng=random.Random(plan["bootstrap_seed"]);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(plan["bootstrap_samples"]))
    return [samples[int(.025*len(samples))],samples[int(.975*len(samples))]]

def summarize(db,identity,plan):
    buckets=defaultdict(list)
    for body, in db.execute("SELECT body FROM cases ORDER BY idx"):
        case=json.loads(body)
        for ms,engines in case["measurements"].items():
            for name,value in engines.items():buckets[(case["cohort"],ms,name)].append(value)
    case_results={}
    for key,values in buckets.items():
        usable=[v for v in values if v["decision"]["usable_decision"]];case_results["/".join(key)]={"positions":len(values),"no_completed_recommendation":len(values)-len(usable),
            "reference_optimal_accuracy":statistics.mean(v["reference_optimal"] for v in usable) if usable else None,"mean_reference_regret_seeds":statistics.mean(v["reference_regret"] for v in usable) if usable else None,
            "median_wall_ms":statistics.median(v["decision"]["wall_ms"] for v in values),"max_wall_ms":max(v["decision"]["wall_ms"] for v in values),"max_overshoot_ms":max(v["decision"]["overshoot_ms"] for v in values),
            "mean_completed_depth":statistics.mean(v["decision"]["completed_depth"] for v in usable) if usable and usable[0]["decision"]["completed_depth"] is not None else None,
            "mean_completed_simulations":statistics.mean(v["decision"]["completed_simulations"] for v in usable) if usable and usable[0]["decision"]["completed_simulations"] is not None else None}
    arenas={}
    for comparison in plan["arena_comparisons"]:
        games=[json.loads(body) for body, in db.execute("SELECT body FROM games WHERE comparison=? ORDER BY pair,seat",(comparison["id"],))];margins=[];scores=[]
        for pair in range(comparison["pairs"]):
            group=[g for g in games if g["pair"]==pair]
            if len(group)==2 and all(g["status"]=="terminal" for g in group):margins.append(statistics.mean(g["candidate_final_margin"] for g in group));scores.append(statistics.mean(g["candidate_match_score"] for g in group))
        ci=interval(margins,plan);fallbacks={name:sum(g["fallbacks"].get(name,0) for g in games) for name in (comparison["candidate"],comparison["opponent"])}
        arenas[comparison["id"]]={"pairs":comparison["pairs"],"complete_pairs":len(margins),"incomplete_pairs":comparison["pairs"]-len(margins),"terminal_games":sum(g["status"]=="terminal" for g in games),"unfinished_games":sum(g["status"]!="terminal" for g in games),
             "candidate_wins":sum(g["candidate_match_score"]==1 for g in games),"candidate_draws":sum(g["candidate_match_score"]==.5 for g in games),"candidate_losses":sum(g["candidate_match_score"]==0 for g in games),
             "paired_mean_score":statistics.mean(scores) if scores else None,"primary_paired_mean_final_margin_seeds":statistics.mean(margins) if margins else None,"primary_margin_bootstrap_95_interval":ci,
             "primary_strength_gate_passed":bool(len(margins)==comparison["pairs"] and not sum(fallbacks.values()) and ci and ci[0]>0),"fallback_counts":fallbacks}
    gates={name:all(arenas[f"{name}_cuda_vs_{base}"]["primary_strength_gate_passed"] for base in ("H_v1","H8_relay_CPU")) for name in ("warm","updated")}
    return dict(schema="phase6-benchmark-results-v1",identity=identity,cases=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],games=db.execute("SELECT COUNT(*) FROM games").fetchone()[0],case_results=case_results,arenas=arenas,both_baseline_primary_gates=gates,
                sealed_final_test_consumed=False,energy_measured=False,default_promotion=False,scope="Development reuse only; equal declared whole-decision PUCT deadline and native-search deadline with truthful wrapper-wall overshoot; no power/solution claim")

def run(path=DEFAULT_RUN,output=DEFAULT_OUTPUT,case_limit=None,no_arena=False,on_event=emit):
    plan=common.read_plan();require(plan["engines"]==["H_v1","H8_relay_CPU","warm_native","warm_cuda","updated_cuda"] and len(plan["cases"])==80 and len(plan["arena_starts"])==24,"Frozen Phase6 cohort differs")
    require(plan["puct"]==dict(simulations=100000,max_nodes=2048,c_puct=1.5,root_noise=None,first_play_urgency="parent fixed-player0 neural bootstrap"),"Frozen time-mode PUCT limits differ")
    if case_limit is not None:require(integer(case_limit,0,80),"Invalid case limit")
    implementation=implementation_manifest();witnesses=model_witnesses(plan,implementation)
    with ExitStack() as stack:
        cpu=stack.enter_context(PerformanceCore(plan["cpu_declared"]));baseline=Native();source=stack.enter_context(NeuralNative(ROOT/plan["source_model"],mode="value-none"));require(source.model.file_sha256==plan["source_model_sha256"],"Source model differs")
        warm_native=stack.enter_context(BatchEvaluator(common.WARM,"native"));warm_cuda=stack.enter_context(BatchEvaluator(common.WARM,"cuda"));updated=stack.enter_context(BatchEvaluator(common.UPDATE,"cuda"))
        engines={"H_v1":("h",baseline),"H8_relay_CPU":("relay",source),"warm_native":("puct",warm_native),"warm_cuda":("puct",warm_cuda),"updated_cuda":("puct",updated)}
        paths=("tools/benchmark-phase6-v2.py","congklak/phase6_common.py","congklak/phase6_store.py","congklak/puct.py","congklak/phase6_inference.py","congklak/native.py","congklak/neural_relay.py","congklak/features_relay.py","congklak/features_v3.py","congklak/timing_v2.py","congklak/config.py")
        identity=dict(schema="phase6-benchmark-v2",plan_sha256=config.digest(plan),implementation=implementation,implementation_file_sha256=sha(IMPLEMENTATION),admission_witnesses=witnesses,
                      sources={p:sha(ROOT/p) for p in paths},cpu=cpu.metadata,baseline_dll_sha256=baseline.build_id,
                      source_model=dict(file_sha256=source.model.file_sha256,weights_sha256=source.model.id,dll_sha256=source.build_id),evaluators={name:e.identity for name,(_,e) in engines.items() if name not in ("H_v1","H8_relay_CPU")},memory_mb_all_engines=8192)
        work=stack.enter_context(closing(phase6_store.Work(Path(path),identity)));db=work.db
        db.executescript("CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT,PRIMARY KEY(comparison,pair,seat)); CREATE TABLE IF NOT EXISTS setup(id INTEGER PRIMARY KEY,body TEXT,hash TEXT);")
        for expected,(index,body,digest) in enumerate(db.execute("SELECT idx,body,hash FROM cases ORDER BY idx")):
            require(index==expected and config.digest(json.loads(body))==digest,"Corrupt/noncontiguous case ledger");validate_case(json.loads(body),index,plan)
        comparisons={c["id"]:c for c in plan["arena_comparisons"]}
        for comparison,pair,seat,body,digest in db.execute("SELECT comparison,pair,seat,body,hash FROM games"):
            require(comparison in comparisons and config.digest(json.loads(body))==digest,"Corrupt/unknown game ledger");replay_game(json.loads(body),comparisons[comparison],pair,seat,plan,baseline)
        count=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        complete=work.get("phase6_benchmark_complete")
        if complete:
            require(count==80 and db.execute("SELECT COUNT(*) FROM games").fetchone()[0]==192,"Incomplete ledger marked complete")
            saved=json.loads(Path(output).read_text(encoding="utf-8"));require(config.digest(saved)==complete["report_sha256"] and saved==summarize(db,identity,plan),"Completed report differs from ledger")
            on_event(dict(event="phase6_benchmark_already_complete",cases=80,games=192));return saved
        setup_row=db.execute("SELECT body,hash FROM setup WHERE id=0").fetchone()
        if setup_row:
            setup=json.loads(setup_row[0]);require(config.digest(setup)==setup_row[1],"Corrupt setup observation")
        else:
            warmup={}
            for name,e in (("warm_native",warm_native),("warm_cuda",warm_cuda),("updated_cuda",updated)):
                cpu.verify();e.evaluate_many([r["state"] for r in plan["cases"][:8]]);warmup[name]=dict(startup_ms=e.startup_ms,warmup=e.last_metrics)
            setup=dict(cpu_observations=cpu.observations,verified=cpu.verify(),warmup=warmup)
            with db:db.execute("INSERT INTO setup VALUES(0,?,?)",(config.canonical(setup),config.digest(setup)))
        # Each fresh process needs the same declared batch8 warmup even when
        # first-invocation startup observations were already committed.
        if setup_row:
            for e in (warm_native,warm_cuda,updated):e.evaluate_many([r["state"] for r in plan["cases"][:8]])
        on_event(dict(event="phase6_benchmark_started",completed=count,total=80))
        stop=min(80,case_limit) if case_limit is not None else 80
        for index in range(count,stop):
            row=plan["cases"][index];q={a["pit"]:a["value_player0"] for a in row["reference"]["action_scores"]};sign=1 if row["state"]["controller"]==0 else -1;measurements={};orders={}
            for bi,ms in enumerate(plan["budgets_ms"]):
                order=expected_order(index,bi,plan);orders[str(ms)]=order;measurements[str(ms)]={}
                for name in order:
                    cpu.verify();answer=decide(engines[name],row["state"],ms,plan);choice=answer["chosen"]
                    measurements[str(ms)][name]=dict(decision=answer,reference_optimal=choice in row["reference"]["best_moves"] if choice is not None else None,reference_regret=sign*(row["reference"]["target_player0"]-q[choice]) if choice is not None else None)
            body=dict(index=index,state_key=row["state_key"],cohort=row["cohort"],stratum=row["stratum"],reference=row["reference"],orders=orders,measurements=measurements,cpu_verified=cpu.verify());validate_case(body,index,plan);work.admit(1024*1024)
            with db:db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
            on_event(dict(event="phase6_case_committed",completed=index+1,total=80))
        if no_arena or stop<80:on_event(dict(event="phase6_benchmark_partial",completed=db.execute("SELECT COUNT(*) FROM cases").fetchone()[0],total=80));return dict(status="partial",identity=identity)
        for comparison in plan["arena_comparisons"]:
            for pair in range(comparison["pairs"]):
                row=plan["arena_starts"][pair]
                for seat in (0,1):
                    if db.execute("SELECT 1 FROM games WHERE comparison=? AND pair=? AND seat=?",(comparison["id"],pair,seat)).fetchone():continue
                    p=row["state"];moves=[];decisions=[];fallbacks=Counter()
                    for _ in range(plan["arena_max_choices"]):
                        normal=baseline.normalize(flatten(p),actor=p["controller"],total=98)
                        if normal["terminal"]:break
                        p=normal["position"];name=comparison["candidate"] if p["controller"]==seat else comparison["opponent"]
                        cpu.verify();answer=decide(engines[name],p,plan["arena_ms"],plan);fallback=not answer["usable_decision"];action=normal["legal_actions"][0] if fallback else answer["chosen"]
                        if fallback:fallbacks[name]+=1
                        decisions.append(dict(engine=name,actor=p["controller"],fallback=fallback,decision=answer));moves.append(action)
                        t=baseline.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(t.get("ok"),"Arena transition failed");p=t["position"]
                    terminal=not any(map(sum,p["pits"]));margin=(p["stores"][0]-p["stores"][1])*(1 if seat==0 else -1);score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None
                    game=dict(comparison=comparison["id"],pair=pair,candidate_seat=seat,start=row["state"],family=row["family"],moves=moves,decisions=decisions,final=p,status="terminal" if terminal else "unfinished",candidate_final_margin=margin if terminal else None,candidate_match_score=score,fallbacks=dict(fallbacks),cpu_verified=cpu.verify())
                    replay_game(game,comparison,pair,seat,plan,baseline);work.admit(1024*1024)
                    with db:db.execute("INSERT INTO games VALUES(?,?,?,?,?)",(comparison["id"],pair,seat,config.canonical(game),config.digest(game)))
                    on_event(dict(event="phase6_arena_game_committed",comparison=comparison["id"],pair=pair,seat=seat,completed=db.execute("SELECT COUNT(*) FROM games").fetchone()[0],total=192))
        result=summarize(db,identity,plan);common.publish(Path(output),result)
        with db:work.put("phase6_benchmark_complete",dict(cases=80,games=192,report_sha256=config.digest(result)))
        on_event(dict(event="phase6_benchmark_complete",cases=80,games=192,primary_gates=result["both_baseline_primary_gates"]));return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("operation",choices=("run","acceptance"));parser.add_argument("--run",default=str(DEFAULT_RUN));parser.add_argument("--output",default=str(DEFAULT_OUTPUT));parser.add_argument("--case-limit",type=int);parser.add_argument("--no-arena",action="store_true");args=parser.parse_args()
    try:
        if args.operation=="acceptance":
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="phase6_benchmark_v2_acceptance.py");result=unittest.TextTestRunner(verbosity=2).run(suite)
            files=("tools/benchmark-phase6-v2.py","tests/phase6_benchmark_v2_acceptance.py","congklak/puct.py","congklak/phase6_inference.py","tests/phase6_puct_acceptance.py","congklak/phase6_common.py","congklak/phase6_store.py")
            body=dict(schema="phase6-benchmark-admission-acceptance-v2",passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),inputs={p:sha(ROOT/p) for p in files},
                      checks=[dict(name=name,passed=result.wasSuccessful()) for name in ("actual_complete_warm_and_sp_witnesses_without_torch_or_mutation","implementation_checksum_and_rehashed_source_drift_refusal","missing_mandatory_science_inputs_refusal","incomplete_result_and_wrong_stage_export_refusal","incomplete_dependency_mapping_refusal","selected_catalog_metric_and_pt_checksum_corruption_refusal","sp_generator_model_lineage_refusal","position_physical_interval_refusal","real_h_relay_puct_whole_clock_and_fallback_admission","checksum_valid_regret_visit_and_timing_corruption_refusal","predeclared_family_bootstrap")])
            common.teacher.atomic_json(ROOT/"build/phase6-benchmark-v2-acceptance.json",body);emit(dict(event="phase6_benchmark_acceptance",ok=result.wasSuccessful(),tests=result.testsRun));return 0 if result.wasSuccessful() else 1
        run(args.run,args.output,args.case_limit,args.no_arena);return 0
    except KeyboardInterrupt:emit(dict(event="phase6_benchmark_interrupted"));return 130
    except Exception as exc:emit(dict(event="phase6_benchmark_error",error=str(exc)));return 2
if __name__=="__main__":raise SystemExit(main())
