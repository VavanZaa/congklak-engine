"""Investigation-only H12/H14 probes of exactly six sealed H8/H10 disagreements."""
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from . import config, natural_study as old, deeper_teacher_v2 as teacher, reports
from .native import ROOT, Native, Control, flatten
from .neural_relay import Model

SCHEMA="six-disagreement-complete-history-horizon-probe-v1"
DEFAULT=ROOT/"data/horizon-probe-v1"
BASELINE=ROOT/"build/horizon-probe-baseline.json"
REGISTRY=ROOT/"build/horizon-probe-registry.json"
PLAN=ROOT/"build/horizon-probe-plan.json"
IMPLEMENTATION=ROOT/"build/horizon-probe-implementation.json"
OUTPUT=ROOT/"build/horizon-probe-result.json"
MODEL,MODEL_SHA=old.MODEL,old.MODEL_SHA
SOURCES=("congklak/horizon_probe.py","tools/horizon-probe.py","tests/horizon_probe_author.py","docs/HORIZON_PROBE_SCOPE.md")
CONTRACT={"roots":6,"selection":"Exact sorted old H8/H10 disjoint-best-set roots; no new selection",
          "h12":{"depth":12,"node_budget":10000000},"h14":{"depth":14,"node_budget":25000000},
          "h14_denominator":"Every selected root, even if H12 budget-rejected",
          "keep_history":True,"time_ms":0,"tt_mb":16,"use_pvs":True,"memory_mb":512,
          "maximum_new_production_nodes":210000000,"automatic_expansion":False,
          "investigation_only":True,"training_performed":False,"sealed_final_test_consumed":False}
IDENTITY_FIELDS=old.IDENTITY_FIELDS
TABLES=("h8_references","h10_references","h12_attempts","h12_labels","h14_attempts","h14_labels")
MANIFEST_FIELDS={"schema","identity","status","roots","h8_references","h10_references","h12_labels","h12_rejected",
                 "h14_labels","h14_rejected","h12_h14_complete_pairs","auxiliary_completed_reports","total_new_nodes",
                 "table_sha256","investigation_only","training_performed","torch_imported","sealed_final_test_consumed","elapsed_seconds_initial_run"}
require,sha,read,publish,table,stopped=old.require,old.sha,old.read,old.publish,old.table,old.stopped


def checked_baseline():
    baseline=read(BASELINE)
    require(baseline["passed"] is True and baseline["inputs_count"]==len(baseline["inputs"])==2199 and baseline["prior_accepted_inputs"]==2197 and baseline["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA,"Probe baseline/model differs")
    for name,checksum in baseline["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==checksum,"Sealed artifact changed: "+name)
    return baseline


def make_registry():
    checked_baseline();rows,h8,h10,_,_,_=old.load_rows();stability=read(ROOT/"build/natural-study-stability.json")
    require(stability["passed"] is True and stability["fixed_model_sha256"]==MODEL_SHA,"Old stability evidence differs")
    keys=sorted(item["state_key"] for item in stability["metrics"] if item["best_sets_disjoint"] is True)
    actual=sorted(key for key in rows if key in h8 and key in h10 and not (set(h8[key]["label"]["best_moves"])&set(h10[key]["label"]["best_moves"])))
    require(len(keys)==6 and keys==actual,"Exact six disagreement selection differs")
    selected={key:rows[key] for key in keys}
    require(Counter(row["cohort"] for row in selected.values())=={"training_candidate":5,"validation":1} and Counter(row["study_role"] for row in selected.values())=={"followup":2,"fresh":4},"Old disagreement ownership differs")
    value={"schema":"six-disagreement-immutable-reference-registry-v1","rules_digest":config.RULES_DIGEST,
           "state_keys":keys,"rows":selected,"h8_references":{key:h8[key] for key in keys},"h10_references":{key:h10[key] for key in keys},
           "stability_sha256":sha(ROOT/"build/natural-study-stability.json"),"old_manifest_sha256":sha(old.DEFAULT/"manifest.json"),
           "baseline_sha256":sha(BASELINE),"investigation_only":True,"no_ingest_or_training":True}
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def plan():
    registry=make_registry();value={"schema":SCHEMA,"contract":CONTRACT,"rules_digest":config.RULES_DIGEST,
        "fixed_model_sha256":MODEL_SHA,"state_keys":registry["state_keys"],"registry_sha256":sha(REGISTRY),"baseline_sha256":sha(BASELINE),
        "depth_semantics":"Every auxiliary report is explicitly completed at its own depth; requested labels are H12/H14 only",
        "deeper_is_not_game_truth":True}
    publish(PLAN,value);return value


def freeze():
    plan();baseline=checked_baseline();value={"schema":"horizon-probe-implementation-v1","plan_sha256":sha(PLAN),"registry_sha256":sha(REGISTRY),
        "baseline_sha256":sha(BASELINE),"sources":{name:sha(ROOT/name) for name in SOURCES},"prior_inputs":baseline["inputs"]}
    publish(IMPLEMENTATION,value);return value


def admit():
    plan_value,manifest,baseline=read(PLAN),read(IMPLEMENTATION),checked_baseline()
    require(plan_value["schema"]==SCHEMA and plan_value["contract"]==CONTRACT and plan_value["rules_digest"]==config.RULES_DIGEST and plan_value["fixed_model_sha256"]==MODEL_SHA,"Probe fixed plan differs")
    require(manifest["schema"]=="horizon-probe-implementation-v1" and manifest["plan_sha256"]==sha(PLAN) and manifest["registry_sha256"]==sha(REGISTRY) and manifest["baseline_sha256"]==sha(BASELINE),"Probe source freeze differs")
    require(manifest["sources"]=={name:sha(ROOT/name) for name in SOURCES} and manifest["prior_inputs"]==baseline["inputs"],"Probe producer/prior source changed")
    return plan_value,manifest


def settings_for(fixture=False,roots=None):
    settings=json.loads(config.canonical(CONTRACT))
    if fixture:settings["roots"]=2 if roots is None else config.integer(roots,"fixture_roots",1,6)
    else:require(roots is None,"Production root count must equal six")
    return settings


def item_for(report,row):
    label={"kind":"finite_choice_horizon","choice_depth":report["choice_depth"],"oracle_manifest_sha256":None,
           "target_player0":report["depth_result"]["value_player0"],"best_moves":report["best_moves"],
           "action_scores":[{"pit":candidate["pit"],"value_player0":candidate["depth_result"]["value_player0"]} for candidate in report["candidates"]],
           "coverage":report["coverage"],"work":{name:report[name] for name in ("nodes","deposits","elapsed_ms","peak_process_bytes")}}
    return {"root_sha256":row["source_root_sha256"],"label":label,"report":report}


def q_signature(item):
    label=item["label"]
    return {name:label[name] for name in ("target_player0","best_moves","action_scores")}


def validate_attempt(row,attempt,label,stage,settings,native):
    depth,budget=settings[stage]["depth"],settings[stage]["node_budget"]
    fields={"study_row_sha256","source_root_sha256","stage","requested_depth","status","stop_reason","completed_depth",
            "native_last_completed_depth","work","history","label_sha256"}
    require(set(attempt)==fields and attempt["study_row_sha256"]==config.digest(row) and attempt["source_root_sha256"]==row["source_root_sha256"] and attempt["stage"]==stage and type(attempt["requested_depth"]) is int and attempt["requested_depth"]==depth,"Requested attempt identity differs")
    history=attempt["history"];require(isinstance(history,list) and type(attempt["completed_depth"]) is int and attempt["completed_depth"]==len(history)<=depth,"Completed history depth/count differs")
    require(type(attempt["native_last_completed_depth"]) is int and attempt["native_last_completed_depth"]==(len(history) if history else -1),"Native last completed depth differs")
    work=attempt["work"];require(set(work)=={"nodes","deposits","elapsed_ms","peak_process_bytes"} and all(type(value) is int and value>=0 for value in work.values()) and 0<work["nodes"]<=budget,"Attempt physical work differs")
    previous={name:0 for name in work}
    for expected,report in enumerate(history,1):
        require(report["choice_depth"]==expected,"Iteration history is not complete prefix1..last")
        old.validate_label(item_for(report,row),row,expected,native)
        require(all(previous[name]<=report[name]<=work[name] for name in work),"Iteration history work is nonmonotonic/over total")
        previous={name:report[name] for name in work}
    if attempt["status"]=="complete":
        require(len(history)==depth and attempt["stop_reason"]=="max_depth" and label is not None and label==item_for(history[-1],row) and attempt["label_sha256"]==config.digest(label),"Requested complete-label/history differs")
        old.validate_label(label,row,depth,native)
    else:
        require(attempt["status"]=="budget_rejected" and len(history)<depth and attempt["stop_reason"]=="node_budget" and work["nodes"]==budget and label is None and attempt["label_sha256"] is None,"Partial history mislabeled as requested completion")


def validate_tables(rows,stores,settings,registry,native,closed=False):
    require(list(rows)==registry["state_keys"][:settings["roots"]] and rows=={key:registry["rows"][key] for key in rows},"Selected six roots/ownership changed")
    for stage in ("h8","h10"):
        refs=stores[stage+"_references"]
        require(refs=={key:registry[stage+"_references"][key] for key in rows},"Old reference item changed")
        for key,item in refs.items():old.validate_label(item,rows[key],int(stage[1:]),native)
    signatures={key:{8:q_signature(stores["h8_references"][key]),10:q_signature(stores["h10_references"][key])} for key in rows}
    for stage in ("h12","h14"):
        attempts,labels=stores[stage+"_attempts"],stores[stage+"_labels"]
        require(list(attempts)==list(rows)[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(rows)),"Requested-stage journal prefix/coverage differs")
        for key,attempt in attempts.items():
            validate_attempt(rows[key],attempt,labels.get(key),stage,settings,native)
            for report in attempt["history"]:
                depth=report["choice_depth"];signature=q_signature(item_for(report,rows[key]))
                require(depth not in signatures[key] or signatures[key][depth]==signature,"Overlapping old/new completed-depth Q/best semantics disagree")
                signatures[key][depth]=signature
    require(not stores["h14_attempts"] or len(stores["h12_attempts"])==len(rows),"H14 started before all H12 attempts")


def execute(row,stage,settings,native,cancel):
    depth,budget=settings[stage]["depth"],settings[stage]["node_budget"];p=row["state"]
    with Control(native,max_depth=depth,time_ms=0,node_budget=budget,memory_mb=512,tt_mb=16,use_pvs=True) as control:
        if cancel is not None and hasattr(cancel,"_register"):cancel._register(control)
        try:result=native.analyze(flatten(p),actor=p["controller"],total=98,max_depth=depth,time_ms=0,node_budget=budget,memory_mb=512,tt_mb=16,use_pvs=True,keep_history=True,control=control)
        finally:
            if cancel is not None and hasattr(cancel,"_remove"):cancel._remove(control)
    if stopped(cancel):return None,None
    history=result["reports"];require((result.get("report") is None and not history) or (history and result["report"]==history[-1]),"Native returned inconsistent iteration history")
    complete=result["stop_reason"]=="max_depth" and len(history)==depth
    require(complete or result["stop_reason"]=="node_budget","Unexpected teacher stop is not budget shortfall")
    item=item_for(history[-1],row) if complete else None
    attempt={"study_row_sha256":config.digest(row),"source_root_sha256":row["source_root_sha256"],"stage":stage,"requested_depth":depth,
             "status":"complete" if complete else "budget_rejected","stop_reason":result["stop_reason"],"completed_depth":len(history),
             "native_last_completed_depth":result["last_completed_depth"],"work":{name:result[name] for name in ("nodes","deposits","elapsed_ms","peak_process_bytes")},
             "history":history,"label_sha256":config.digest(item) if item is not None else None}
    validate_attempt(row,attempt,item,stage,settings,native);return item,attempt


def counts(rows,stores):
    return {"roots":len(rows),"h8_references":len(stores["h8_references"]),"h10_references":len(stores["h10_references"]),
            "h12_labels":len(stores["h12_labels"]),"h12_rejected":sum(a["status"]=="budget_rejected" for a in stores["h12_attempts"].values()),
            "h14_labels":len(stores["h14_labels"]),"h14_rejected":sum(a["status"]=="budget_rejected" for a in stores["h14_attempts"].values()),
            "h12_h14_complete_pairs":len(set(stores["h12_labels"])&set(stores["h14_labels"])),
            "auxiliary_completed_reports":sum(len(a["history"]) for stage in ("h12","h14") for a in stores[stage+"_attempts"].values()),
            "total_new_nodes":sum(a["work"]["nodes"] for stage in ("h12","h14") for a in stores[stage+"_attempts"].values())}


def validate_complete_marker(manifest,identity,rows,stores):
    require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["reserve_bytes"]==2*1024**3 and identity["quota_bytes"]==50*1024**3,"Probe identity fields/schema/reserve differs")
    require(set(manifest)==MANIFEST_FIELDS and manifest["schema"]==SCHEMA and manifest["identity"]==identity and manifest["status"]=="complete","Probe complete marker fields/identity differ")
    require(len(rows)==identity["settings"]["roots"] and len(stores["h12_attempts"])==len(stores["h14_attempts"])==len(rows),"Probe complete marker has unfinished targets")
    require(all(type(manifest[name]) is int and manifest[name]==value for name,value in counts(rows,stores).items()),"Probe complete counters differ")
    require(manifest["total_new_nodes"]<=210000000 and manifest["table_sha256"]=={"roots":config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},"Probe complete work/table digests differ")
    require(manifest["investigation_only"] is True and manifest["training_performed"] is False and manifest["torch_imported"] is False and manifest["sealed_final_test_consumed"] is False and type(manifest["elapsed_seconds_initial_run"]) in (int,float) and math.isfinite(manifest["elapsed_seconds_initial_run"]) and manifest["elapsed_seconds_initial_run"]>0,"Probe complete scope/clock differs")


def run(path=DEFAULT,stage="all",fixture=False,roots=None,on_event=None,cancel=None):
    path=Path(path).resolve();require(stage in ("h12","h14","all"),"Unknown probe stage");settings=settings_for(fixture,roots)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Fixture must stay inside build")
    else:require(path==DEFAULT.resolve(),"Production probe path differs");admit()
    registry=read(REGISTRY);require(registry["sha256"]==config.digest({k:v for k,v in registry.items() if k!="sha256"}),"Probe registry checksum differs")
    native=Native();identity={"schema":SCHEMA,"settings":settings,"fixture":fixture,"rules_digest":config.RULES_DIGEST,
        "fixed_model_sha256":sha(MODEL),"fixed_model_weights_sha256":Model(MODEL).id,"source_sha256":sha(__file__),"tool_sha256":sha(ROOT/"tools/horizon-probe.py"),
        "registry_sha256":sha(REGISTRY),"baseline_sha256":sha(BASELINE),"native_build_sha256":native.build_id,
        "plan_sha256":None if fixture else sha(PLAN),"implementation_sha256":None if fixture else sha(IMPLEMENTATION),"reserve_bytes":2*1024**3,"quota_bytes":50*1024**3}
    require(identity["fixed_model_sha256"]==MODEL_SHA and "torch" not in sys.modules,"Fixed CPU-only probe model/runtime differs")
    work=teacher.Work(path,identity);started=time.monotonic()
    def event(name,**value):
        if on_event:on_event({"event":name,**value})
    try:
        work.db.executescript("".join("CREATE TABLE IF NOT EXISTS "+name+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for name in TABLES))
        expected={key:registry["rows"][key] for key in registry["state_keys"][:settings["roots"]]};rows=table(work.db,"roots")
        stores={name:table(work.db,name) for name in TABLES}
        if not rows:
            require(not any(stores.values()),"Orphan probe reference/attempt tables")
            with work.db:
                for key,row in expected.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
                for name in ("h8_references","h10_references"):
                    for key in expected:
                        item=registry[name][key];work.db.execute("INSERT INTO "+name+" VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)));stores[name][key]=item
            rows=expected
        require(rows==expected,"Probe original wrapped roots changed");validate_tables(rows,stores,settings,registry,native)
        completed=work.get("complete")
        if completed is None:
            for target in ("h12","h14"):
                if target=="h14" and stage=="h12":break
                attempts,labels=stores[target+"_attempts"],stores[target+"_labels"]
                for key in sorted(set(rows)-set(attempts)):
                    if stopped(cancel):break
                    item,attempt=execute(rows[key],target,settings,native,cancel)
                    if attempt is None:break
                    tentative={name:dict(value) for name,value in stores.items()};tentative[target+"_attempts"][key]=attempt
                    if item is not None:tentative[target+"_labels"][key]=item
                    validate_tables(rows,tentative,settings,registry,native)
                    event("horizon_probe_"+target+"_precommit",state_key=key,completed=len(attempts),labels=len(labels),target=len(rows),completed_depth=attempt["completed_depth"])
                    work.admit(2*1024**2)
                    with work.db:
                        work.db.execute("INSERT INTO "+target+"_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                        if item is not None:work.db.execute("INSERT INTO "+target+"_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                    attempts[key]=attempt
                    if item is not None:labels[key]=item
                    event("horizon_probe_"+target+"_committed",state_key=key,completed=len(attempts),labels=len(labels),rejected=sum(a["status"]=="budget_rejected" for a in attempts.values()),target=len(rows),completed_depth=attempt["completed_depth"])
                if len(attempts)!=len(rows):break
        if completed is None and len(stores["h12_attempts"])==len(stores["h14_attempts"])==len(rows):
            completed={"schema":SCHEMA,"identity":identity,"status":"complete",**counts(rows,stores),"table_sha256":{"roots":config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},
                       "investigation_only":True,"training_performed":False,"torch_imported":False,"sealed_final_test_consumed":False,"elapsed_seconds_initial_run":time.monotonic()-started}
            validate_complete_marker(completed,identity,rows,stores)
            with work.db:work.put("complete",completed)
        if completed is not None:
            validate_complete_marker(completed,identity,rows,stores);publish(path/"manifest.json",completed)
            if not fixture:publish(OUTPUT,completed)
            load_rows(path);event("horizon_probe_complete",**counts(rows,stores));return completed
        return {"status":"interrupted",**counts(rows,stores)}
    finally:work.close()


def load_rows(path=DEFAULT):
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        def meta(key):
            row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Probe missing committed "+key)
            value=json.loads(row[0]);require(config.digest(value)==row[1],"Corrupt probe metadata "+key);return value
        manifest,identity=meta("complete"),meta("identity")
        require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["reserve_bytes"]==2*1024**3 and identity["quota_bytes"]==50*1024**3,"Closed probe identity fields/schema/reserve differ")
        require(identity["source_sha256"]==sha(__file__) and identity["tool_sha256"]==sha(ROOT/"tools/horizon-probe.py") and identity["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA and identity["fixed_model_weights_sha256"]==Model(MODEL).id and identity["rules_digest"]==config.RULES_DIGEST and identity["registry_sha256"]==sha(REGISTRY) and identity["baseline_sha256"]==sha(BASELINE),"Closed probe source/model/protection differs")
        settings=identity["settings"]
        if identity["fixture"] is False:
            admit();require(settings==CONTRACT and identity["plan_sha256"]==sha(PLAN) and identity["implementation_sha256"]==sha(IMPLEMENTATION),"Closed probe production plan differs")
        else:require(identity["fixture"] is True and path.is_relative_to(ROOT/"build") and settings==settings_for(True,settings["roots"]) and identity["plan_sha256"] is None and identity["implementation_sha256"] is None,"Closed probe fixture differs")
        native=Native();require(native.build_id==identity["native_build_sha256"],"Closed probe native build differs")
        registry=read(REGISTRY);require(registry["sha256"]==config.digest({k:v for k,v in registry.items() if k!="sha256"}),"Closed probe registry checksum differs")
        rows=table(db,"roots");stores={name:table(db,name) for name in TABLES};validate_tables(rows,stores,settings,registry,native,closed=True)
        validate_complete_marker(manifest,identity,rows,stores)
        require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(manifest)+"\n","Closed probe file/commit publication differs")
        return rows,stores["h8_references"],stores["h10_references"],stores["h12_labels"],stores["h14_labels"],stores["h12_attempts"],stores["h14_attempts"],manifest
