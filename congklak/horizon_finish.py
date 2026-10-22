"""Versioned, investigation-only completion of the five capped H14 roots."""
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from . import config, horizon_probe as old
from .native import ROOT, Native
from .neural_relay import Model

SCHEMA="five-capped-h14-two-tier-completion-v1"
DEFAULT=ROOT/"data/horizon-finish-v1"
BASELINE=ROOT/"build/horizon-finish-baseline.json"
REGISTRY=ROOT/"build/horizon-finish-registry.json"
PLAN=ROOT/"build/horizon-finish-plan.json"
IMPLEMENTATION=ROOT/"build/horizon-finish-implementation.json"
OUTPUT=ROOT/"build/horizon-finish-result.json"
MODEL,MODEL_SHA=old.MODEL,old.MODEL_SHA
SOURCES=("congklak/horizon_finish.py","tools/horizon-finish.py","tests/horizon_finish_author.py","docs/HORIZON_FINISH_SCOPE.md")
CONTRACT={"roots":5,"selection":"Exactly the five prior requested-H14 budget_rejected roots",
          "tier1":{"depth":14,"node_budget":250000000},"tier2":{"depth":14,"node_budget":1000000000},
          "tier2_denominator":"Only tier1 budget_rejected roots after all tier1 attempts",
          "keep_history":True,"time_ms":0,"tt_mb":16,"use_pvs":True,"memory_mb":512,
          "maximum_new_production_nodes":6250000000,"automatic_expansion":False,
          "investigation_only":True,"training_performed":False,"sealed_final_test_consumed":False}
PRIOR_TABLES=("h8_references","h10_references","h12_references","prior_h12_attempts","prior_h14_attempts","prior_h14_labels")
NEW_TABLES=("tier1_attempts","tier1_labels","tier2_attempts","tier2_labels")
TABLES=PRIOR_TABLES+NEW_TABLES
IDENTITY_FIELDS=old.IDENTITY_FIELDS
COUNT_FIELDS={"comparison_roots","target_roots","h8_references","h10_references","h12_references","prior_h14_labels",
              "tier1_attempts","tier1_labels","tier1_rejected","tier2_eligible","tier2_attempts","tier2_labels","tier2_rejected",
              "new_h14_labels","combined_h14_labels","unresolved_target_roots","auxiliary_completed_reports","total_new_nodes"}
MANIFEST_FIELDS={"schema","identity","status","scope_finished","unresolved_keys","table_sha256","investigation_only",
                 "training_performed","torch_imported","sealed_final_test_consumed","elapsed_seconds_initial_run"}|COUNT_FIELDS
require,sha,read,publish,table,stopped=old.require,old.sha,old.read,old.publish,old.table,old.stopped
item_for,q_signature=old.item_for,old.q_signature
teacher=old.teacher


def checked_baseline():
    value=read(BASELINE)
    require(value["schema"]=="horizon-finish-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==2314 and value["prior_accepted_inputs"]==2312 and value["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA,"Finish baseline/model differs")
    for name,checksum in value["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==checksum,"Sealed artifact changed: "+name)
    return value


def make_registry():
    checked_baseline();rows,h8,h10,h12,h14,a12,a14,manifest=old.load_rows()
    keys=sorted(key for key,value in a14.items() if value["status"]=="budget_rejected")
    require(len(rows)==6 and len(keys)==5 and len(h14)==1 and all(key not in h14 for key in keys),"Exact five old capped H14 selection differs")
    value={"schema":"five-capped-h14-immutable-reference-registry-v1","rules_digest":config.RULES_DIGEST,
           "state_keys":keys,"comparison_keys":list(rows),"rows":rows,
           "h8_references":h8,"h10_references":h10,"h12_references":h12,
           "prior_h12_attempts":a12,"prior_h14_attempts":a14,"prior_h14_labels":h14,
           "old_manifest_sha256":sha(old.DEFAULT/"manifest.json"),"baseline_sha256":sha(BASELINE),
           "investigation_only":True,"no_ingest_or_training":True}
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def plan():
    registry=make_registry();value={"schema":SCHEMA,"contract":CONTRACT,"rules_digest":config.RULES_DIGEST,
        "fixed_model_sha256":MODEL_SHA,"state_keys":registry["state_keys"],"registry_sha256":sha(REGISTRY),"baseline_sha256":sha(BASELINE),
        "fixture_contract":"First exact capped root, H14 at25m then250m, otherwise identical; production never accepts fixture budgets",
        "depth_semantics":"Complete prefix histories are auxiliary; only a completed requested H14 is a full label",
        "deeper_is_not_game_truth":True}
    publish(PLAN,value);return value


def freeze():
    plan();baseline=checked_baseline();value={"schema":"horizon-finish-implementation-v1","plan_sha256":sha(PLAN),"registry_sha256":sha(REGISTRY),
        "baseline_sha256":sha(BASELINE),"sources":{name:sha(ROOT/name) for name in SOURCES},"prior_inputs":baseline["inputs"]}
    publish(IMPLEMENTATION,value);return value


def admit():
    value,implementation,baseline=read(PLAN),read(IMPLEMENTATION),checked_baseline()
    require(value["schema"]==SCHEMA and value["contract"]==CONTRACT and value["rules_digest"]==config.RULES_DIGEST and value["fixed_model_sha256"]==MODEL_SHA,"Finish fixed plan differs")
    require(implementation["schema"]=="horizon-finish-implementation-v1" and implementation["plan_sha256"]==sha(PLAN) and implementation["registry_sha256"]==sha(REGISTRY) and implementation["baseline_sha256"]==sha(BASELINE),"Finish source freeze differs")
    require(implementation["sources"]=={name:sha(ROOT/name) for name in SOURCES} and implementation["prior_inputs"]==baseline["inputs"],"Finish producer/prior source changed")
    return value,implementation


def settings_for(fixture=False,roots=None):
    value=json.loads(config.canonical(CONTRACT))
    if fixture:
        value["roots"]=1 if roots is None else config.integer(roots,"fixture_roots",1,5)
        value["tier1"]["node_budget"]=25000000;value["tier2"]["node_budget"]=250000000
    else:require(roots is None,"Production must target exactly five roots")
    return value


def registry_checked():
    value=read(REGISTRY)
    require(value["sha256"]==config.digest({key:item for key,item in value.items() if key!="sha256"}),"Finish registry checksum differs")
    return value


def target_keys(settings,registry):return registry["state_keys"][:settings["roots"]]


def eligible_keys(tier,settings,registry,stores):
    if tier=="tier1":return target_keys(settings,registry)
    return [key for key in target_keys(settings,registry) if key in stores["tier1_attempts"] and stores["tier1_attempts"][key]["status"]=="budget_rejected"]


def combined_labels(stores):
    return {**stores["prior_h14_labels"],**stores["tier1_labels"],**stores["tier2_labels"]}


def validate_attempt(row,attempt,label,tier,settings,native):
    extra={"schema","tier","node_budget"}
    require(set(attempt)=={"study_row_sha256","source_root_sha256","stage","requested_depth","status","stop_reason","completed_depth","native_last_completed_depth","work","history","label_sha256"}|extra,"Finish attempt fields differ")
    require(attempt["schema"]=="h14-budget-tier-attempt-v1" and attempt["tier"]==tier and type(attempt["node_budget"]) is int and attempt["node_budget"]==settings[tier]["node_budget"],"Finish attempt tier/cap differs")
    old.validate_attempt(row,{key:value for key,value in attempt.items() if key not in extra},label,"h14",{"h14":settings[tier]},native)


def validate_tables(rows,stores,settings,registry,native,closed=False):
    require(rows==registry["rows"] and list(rows)==registry["comparison_keys"] and len(rows)==6,"Finish original six wrapped roots changed")
    require(set(stores)==set(TABLES) and all(stores[name]==registry[name] for name in PRIOR_TABLES),"Finish prior references or ledger changed")
    oldstores={"h8_references":stores["h8_references"],"h10_references":stores["h10_references"],
               "h12_labels":stores["h12_references"],"h12_attempts":stores["prior_h12_attempts"],
               "h14_labels":stores["prior_h14_labels"],"h14_attempts":stores["prior_h14_attempts"]}
    old.validate_tables(rows,oldstores,old.CONTRACT,read(old.REGISTRY),native,closed=True)
    signatures={key:{} for key in rows}
    for name in ("h8_references","h10_references","h12_references","prior_h14_labels"):
        for key,item in stores[name].items():signatures[key][item["label"]["choice_depth"]]=q_signature(item)
    for name in ("prior_h12_attempts","prior_h14_attempts"):
        for key,attempt in stores[name].items():
            for report in attempt["history"]:signatures[key][report["choice_depth"]]=q_signature(item_for(report,rows[key]))
    for tier in ("tier1","tier2"):
        attempts,labels=stores[tier+"_attempts"],stores[tier+"_labels"];eligible=eligible_keys(tier,settings,registry,stores)
        require(list(attempts)==eligible[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(eligible)),"Finish tier prefix/coverage differs")
        for key,attempt in attempts.items():
            validate_attempt(rows[key],attempt,labels.get(key),tier,settings,native)
            for report in attempt["history"]:
                depth=report["choice_depth"];signature=q_signature(item_for(report,rows[key]))
                require(depth not in signatures[key] or signatures[key][depth]==signature,"Finish overlapping same-depth Q/best semantics disagree")
                signatures[key][depth]=signature
    require(not stores["tier2_attempts"] or len(stores["tier1_attempts"])==settings["roots"],"Fallback started before all tier1 attempts")


def execute(row,tier,settings,native,cancel):
    item,attempt=old.execute(row,"h14",{"h14":settings[tier]},native,cancel)
    if attempt is not None:
        attempt={**attempt,"schema":"h14-budget-tier-attempt-v1","tier":tier,"node_budget":settings[tier]["node_budget"]}
        validate_attempt(row,attempt,item,tier,settings,native)
    return item,attempt


def counts(rows,stores,settings,registry):
    selected=target_keys(settings,registry);new={**stores["tier1_labels"],**stores["tier2_labels"]}
    return {"comparison_roots":len(rows),"target_roots":len(selected),"h8_references":len(stores["h8_references"]),"h10_references":len(stores["h10_references"]),
            "h12_references":len(stores["h12_references"]),"prior_h14_labels":len(stores["prior_h14_labels"]),
            "tier1_attempts":len(stores["tier1_attempts"]),"tier1_labels":len(stores["tier1_labels"]),"tier1_rejected":sum(a["status"]=="budget_rejected" for a in stores["tier1_attempts"].values()),
            "tier2_eligible":len(eligible_keys("tier2",settings,registry,stores)),"tier2_attempts":len(stores["tier2_attempts"]),"tier2_labels":len(stores["tier2_labels"]),"tier2_rejected":sum(a["status"]=="budget_rejected" for a in stores["tier2_attempts"].values()),
            "new_h14_labels":len(new),"combined_h14_labels":len(combined_labels(stores)),"unresolved_target_roots":len(set(selected)-set(new)),
            "auxiliary_completed_reports":sum(len(a["history"]) for tier in ("tier1","tier2") for a in stores[tier+"_attempts"].values()),
            "total_new_nodes":sum(a["work"]["nodes"] for tier in ("tier1","tier2") for a in stores[tier+"_attempts"].values())}


def validate_complete_marker(manifest,identity,rows,stores,registry):
    require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["reserve_bytes"]==2*1024**3 and identity["quota_bytes"]==50*1024**3,"Finish identity fields/schema/reserve differs")
    settings=identity["settings"];expected=counts(rows,stores,settings,registry)
    require(set(manifest)==MANIFEST_FIELDS and manifest["schema"]==SCHEMA and manifest["identity"]==identity,"Finish complete marker fields/identity differ")
    require(len(stores["tier1_attempts"])==settings["roots"] and len(stores["tier2_attempts"])==len(eligible_keys("tier2",settings,registry,stores)),"Finish complete marker has unfinished targets")
    unresolved=sorted(set(target_keys(settings,registry))-set(combined_labels(stores)));finished=not unresolved
    require(manifest["scope_finished"] is finished and manifest["status"]==("complete" if finished else "budget_exhausted") and manifest["unresolved_keys"]==unresolved,"Finish unresolved scope/status differs")
    require(all(type(manifest[name]) is int and manifest[name]==value for name,value in expected.items()),"Finish complete counters differ")
    require(manifest["total_new_nodes"]<=settings["roots"]*(settings["tier1"]["node_budget"]+settings["tier2"]["node_budget"]) and manifest["table_sha256"]=={"roots":config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},"Finish complete work/table digests differ")
    require(manifest["investigation_only"] is True and manifest["training_performed"] is False and manifest["torch_imported"] is False and manifest["sealed_final_test_consumed"] is False and type(manifest["elapsed_seconds_initial_run"]) in (int,float) and math.isfinite(manifest["elapsed_seconds_initial_run"]) and manifest["elapsed_seconds_initial_run"]>0,"Finish complete scope/clock differs")


def run(path=DEFAULT,stage="all",fixture=False,roots=None,on_event=None,cancel=None):
    path=Path(path).resolve();require(stage in ("tier1","tier2","all"),"Unknown finish stage");settings=settings_for(fixture,roots)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Finish fixture must stay inside build")
    else:require(path==DEFAULT.resolve(),"Production finish path differs");admit()
    registry=registry_checked();native=Native();identity={"schema":SCHEMA,"settings":settings,"fixture":fixture,"rules_digest":config.RULES_DIGEST,
        "fixed_model_sha256":sha(MODEL),"fixed_model_weights_sha256":Model(MODEL).id,"source_sha256":sha(__file__),"tool_sha256":sha(ROOT/"tools/horizon-finish.py"),
        "registry_sha256":sha(REGISTRY),"baseline_sha256":sha(BASELINE),"native_build_sha256":native.build_id,
        "plan_sha256":None if fixture else sha(PLAN),"implementation_sha256":None if fixture else sha(IMPLEMENTATION),"reserve_bytes":2*1024**3,"quota_bytes":50*1024**3}
    require(identity["fixed_model_sha256"]==MODEL_SHA and "torch" not in sys.modules,"Fixed CPU-only finish model/runtime differs")
    work=teacher.Work(path,identity);started=time.monotonic()
    def event(name,**value):
        if on_event:on_event({"event":name,**value})
    try:
        work.db.executescript("".join("CREATE TABLE IF NOT EXISTS "+name+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for name in TABLES))
        rows=table(work.db,"roots");stores={name:table(work.db,name) for name in TABLES}
        if not rows:
            require(not any(stores.values()),"Orphan finish references/attempts")
            with work.db:
                for key,row in registry["rows"].items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
                for name in PRIOR_TABLES:
                    for key,item in registry[name].items():work.db.execute("INSERT INTO "+name+" VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)));stores[name][key]=item
            rows=registry["rows"]
        validate_tables(rows,stores,settings,registry,native);completed=work.get("complete")
        if completed is None:
            for tier in ("tier1","tier2"):
                if tier=="tier2" and stage=="tier1":break
                attempts,labels=stores[tier+"_attempts"],stores[tier+"_labels"];eligible=eligible_keys(tier,settings,registry,stores)
                for key in eligible[len(attempts):]:
                    if stopped(cancel):break
                    item,attempt=execute(rows[key],tier,settings,native,cancel)
                    if attempt is None:break
                    tentative={name:dict(value) for name,value in stores.items()};tentative[tier+"_attempts"][key]=attempt
                    if item is not None:tentative[tier+"_labels"][key]=item
                    validate_tables(rows,tentative,settings,registry,native)
                    event("horizon_finish_"+tier+"_precommit",state_key=key,completed=len(attempts),labels=len(labels),target=len(eligible),completed_depth=attempt["completed_depth"],node_budget=settings[tier]["node_budget"])
                    work.admit(2*1024**2)
                    with work.db:
                        work.db.execute("INSERT INTO "+tier+"_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                        if item is not None:work.db.execute("INSERT INTO "+tier+"_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                    attempts[key]=attempt
                    if item is not None:labels[key]=item
                    event("horizon_finish_"+tier+"_committed",state_key=key,completed=len(attempts),labels=len(labels),rejected=sum(a["status"]=="budget_rejected" for a in attempts.values()),target=len(eligible),completed_depth=attempt["completed_depth"],node_budget=settings[tier]["node_budget"])
                if len(attempts)!=len(eligible):break
        if completed is None and len(stores["tier1_attempts"])==settings["roots"] and len(stores["tier2_attempts"])==len(eligible_keys("tier2",settings,registry,stores)):
            unresolved=sorted(set(target_keys(settings,registry))-set(combined_labels(stores)))
            completed={"schema":SCHEMA,"identity":identity,"status":"complete" if not unresolved else "budget_exhausted","scope_finished":not unresolved,"unresolved_keys":unresolved,
                       **counts(rows,stores,settings,registry),"table_sha256":{"roots":config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},
                       "investigation_only":True,"training_performed":False,"torch_imported":False,"sealed_final_test_consumed":False,"elapsed_seconds_initial_run":time.monotonic()-started}
            validate_complete_marker(completed,identity,rows,stores,registry)
            with work.db:work.put("complete",completed)
        if completed is not None:
            validate_complete_marker(completed,identity,rows,stores,registry);publish(path/"manifest.json",completed)
            if not fixture:publish(OUTPUT,completed)
            load_rows(path);event("horizon_finish_complete",status=completed["status"],**counts(rows,stores,settings,registry));return completed
        return {"status":"interrupted",**counts(rows,stores,settings,registry)}
    finally:work.close()


def load_rows(path=DEFAULT):
    """Read only a strictly closed journal; return (six_original_rows, stores, manifest)."""
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        def meta(key):
            row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Finish missing committed "+key)
            value=json.loads(row[0]);require(config.digest(value)==row[1],"Corrupt finish metadata "+key);return value
        manifest,identity=meta("complete"),meta("identity")
        require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["reserve_bytes"]==2*1024**3 and identity["quota_bytes"]==50*1024**3,"Closed finish identity fields/schema/reserve differs")
        require(identity["source_sha256"]==sha(__file__) and identity["tool_sha256"]==sha(ROOT/"tools/horizon-finish.py") and identity["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA and identity["fixed_model_weights_sha256"]==Model(MODEL).id and identity["rules_digest"]==config.RULES_DIGEST and identity["registry_sha256"]==sha(REGISTRY) and identity["baseline_sha256"]==sha(BASELINE),"Closed finish source/model/protection differs")
        settings=identity["settings"]
        if identity["fixture"] is False:
            admit();require(path==DEFAULT.resolve() and settings==CONTRACT and identity["plan_sha256"]==sha(PLAN) and identity["implementation_sha256"]==sha(IMPLEMENTATION),"Closed finish production plan differs")
        else:require(identity["fixture"] is True and path.is_relative_to(ROOT/"build") and settings==settings_for(True,settings["roots"]) and identity["plan_sha256"] is None and identity["implementation_sha256"] is None,"Closed finish fixture differs")
        native=Native();require(native.build_id==identity["native_build_sha256"],"Closed finish native build differs")
        registry=registry_checked();rows=table(db,"roots");stores={name:table(db,name) for name in TABLES}
        validate_tables(rows,stores,settings,registry,native,closed=True);validate_complete_marker(manifest,identity,rows,stores,registry)
        require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(manifest)+"\n","Closed finish file/commit publication differs")
        return rows,stores,manifest
