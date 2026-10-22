"""Requested full-action H8/H10/H12 references for five fixed diagnostic roots."""
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from . import config,horizon_probe as history,ordinary_value_pilot as old,phase6_store,divergence_study_common as common
from .native import ROOT,Native,flatten

SCHEMA="five-H_v1-divergence-full-action-teacher-v1"
DEFAULT=ROOT/"data/divergence-teacher-v1"
BASELINE=ROOT/"build/divergence-baseline.json"
REGISTRY=ROOT/"build/divergence-teacher-registry.json"
PLAN=ROOT/"build/divergence-teacher-plan.json"
IMPLEMENTATION=ROOT/"build/divergence-teacher-implementation.json"
OUTPUT=ROOT/"build/divergence-teacher-result.json"
SOURCE_REPORT=ROOT/"build/ordinary-value-loss-review.json"
MODEL,MODEL_SHA=old.SOURCE,old.SOURCE_SHA
CANDIDATE=ROOT/"data/ordinary-value-pilot-v1/model.json"
STAGES=("h8","h10","h12")
TABLES=tuple(stage+suffix for stage in STAGES for suffix in ("_attempts","_labels"))
SOURCES=("congklak/divergence_teacher.py","tools/label-divergence-actions.py","tests/divergence_teacher_author.py","docs/DIVERGENCE_TEACHER.md","docs/DIVERGENCE_STUDY_SCOPE.md","congklak/divergence_study_common.py")
DEPENDENCIES=("congklak/horizon_probe.py","congklak/natural_study.py","congklak/ordinary_value_pilot.py","congklak/phase6_store.py","congklak/native.py","congklak/reports.py","congklak/deeper_teacher_v2.py","congklak/divergence_search.py")
CONTRACT=dict(roots=5,h8=dict(depth=8,node_budget=2000000),h10=dict(depth=10,node_budget=5000000),h12=dict(depth=12,node_budget=10000000),keep_history=True,time_ms=0,tt_mb=16,use_pvs=True,memory_mb=512,maximum_new_nodes=85000000,all_horizons_attempted_independently=True,automatic_expansion=False,auxiliary_horizon_fallback=False,diagnostic_only=True,training_performed=False,promotion=False)
AUDIT_METHODS=dict(preflight=["test_01_prior3914_exact_panel_models_and_predeclared_budgets","test_02_search_real_pre_post_commit_kills_prefix_semantics_and_refusal","test_03_teacher_real_pre_post_commit_kills_history_prefix_and_refusal"],search=["test_04_all40_fixedwork_cells_literal_tree_and_teacher_gate"])
require,read,sha,table,stopped=history.require,history.read,history.sha,history.table,history.stopped
item_for,q_signature,validate_attempt,execute=history.item_for,history.q_signature,history.validate_attempt,history.execute


def equal(a,b):return config.canonical(a)==config.canonical(b)


def publish(path,body):
    path=Path(path)
    if path.exists():require(path.read_text(encoding="utf-8")==config.canonical(body)+"\n","Immutable divergence teacher publication differs")
    else:history.teacher.atomic_json(path,body)
    return body


def baseline():
    value=common.baseline(verify_all=True);require(sha(MODEL)==MODEL_SHA,"Divergence control model differs");return value


def panel():
    rows={};source=common.load_panel();require(equal(source,common.projection()),"Fixed panel typed fields differ")
    for root in source["roots"]:
        key=root["state_key"];rows[key]=dict(root=root,source_root_sha256=config.digest(root),state=root["state"],state_key=key,family=root["family"],legal_mask=root["legal_mask"],cohort="diagnostic",purpose="fixed_five_H_v1_divergence_investigation",stratum=root["stratum"],source_divergence_sha256=root["source_divergence_sha256"],source_seat_sha256=root["source_seat_sha256"],control_action=root["control_action"],candidate_action=root["candidate_action"])
    require(len(rows)==5,"Divergence exact panel count differs")
    return dict(sorted(rows.items()))


def registry_body():
    rows=panel();body=dict(schema="divergence-teacher-diagnostic-panel-v1",rows=rows,state_keys=list(rows),source_panel_sha256=sha(common.PANEL),source_report_sha256=sha(SOURCE_REPORT),baseline_sha256=sha(BASELINE),rules_digest=config.RULES_DIGEST,diagnostic_only=True,quotient_reuse=False);body["sha256"]=config.digest(body);return body


def make_registry():
    baseline();return publish(REGISTRY,registry_body())


def plan_body():
    return dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),control_model_sha256=sha(MODEL),candidate_model_sha256=sha(CANDIDATE),state_keys=list(panel()),prerequisite="Strict closed production divergence_search.primary_witness before any production teacher call",depth_semantics="Requested full-action finite-choice H8/H10/H12 only; auxiliary completed histories remain explicitly their own depth, never fallback truth")


def plan():
    make_registry();return publish(PLAN,plan_body())


def freeze():
    plan();body=dict(schema="divergence-teacher-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES+DEPENDENCIES});return publish(IMPLEMENTATION,body)


def admit():
    baseline();registry=read(REGISTRY);value=read(PLAN);implementation=read(IMPLEMENTATION)
    require(equal(registry,registry_body()),"Frozen diagnostic panel differs");require(equal(value,plan_body()),"Frozen divergence teacher plan differs")
    require(equal(implementation,dict(schema="divergence-teacher-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES+DEPENDENCIES})),"Frozen divergence teacher source differs")
    return value,implementation


def settings_for(fixture=False,roots=None):
    require(type(fixture) is bool,"Fixture identity must be boolean");settings=json.loads(config.canonical(CONTRACT))
    if fixture:
        settings["roots"]=2 if roots is None else config.integer(roots,"fixture_roots",1,2)
        settings.update(h8=dict(depth=2,node_budget=2000000),h10=dict(depth=3,node_budget=5000000),h12=dict(depth=4,node_budget=10000000),fixture_panel="First already-consumed diagnostic roots with complete old H8/H10/H12; actual fixture depths2/3/4 only, no production substitution")
    else:require(roots is None,"Production requires exact five roots")
    return settings


def expected_rows(settings,fixture):
    if not fixture:return read(REGISTRY)["rows"]
    return old.label_rows(dict(roots=settings["roots"]),True)


def gate():
    from . import divergence_search
    witness=divergence_search.primary_witness();expected_primary=dict(schema=divergence_search.SCHEMA+"-primary-witness",passed=True,fixture=False,complete_cells=40,completed_simulations=19200,torch_imported_by_gate=False);require(equal({name:witness[name] for name in expected_primary},expected_primary),"Production fixed-simulation comparison is incomplete");proofs={}
    for group,methods in AUDIT_METHODS.items():
        path=ROOT/"build/divergence-audit"/(group+".json");proof=read(path);expected=dict(schema="divergence-independent-group-v1",group=group,passed=True,tests_run=len(methods),failures=0,errors=0,skips=0,covered_methods=methods,covered_checks=["__main__.DivergenceIndependentAudit."+name for name in methods])
        require(equal({name:proof[name] for name in expected},expected),"Independent divergence "+group+" gate is incomplete or incompatible");inputs=proof["inputs"];require(isinstance(inputs,dict) and bool(inputs),"Independent gate has no source inputs")
        required={"tests/divergence_independent_audit.py","tools/divergence-audit.py","build/divergence-audit-source-freeze.json","build/divergence-baseline.json","build/divergence-panel.json","docs/DIVERGENCE_STUDY_SCOPE.md","congklak/divergence_search.py","congklak/divergence_teacher.py","build/divergence-search-plan.json","build/divergence-search-implementation.json","build/divergence-teacher-registry.json","build/divergence-teacher-plan.json","build/divergence-teacher-implementation.json"}
        if group=="search":required.update({"data/divergence-search-v1/work.sqlite3","data/divergence-search-v1/manifest.json","build/divergence-search-result.json"})
        require(required<=set(inputs) and all(inputs.get(name)==digest for name,digest in common.baseline()["inputs"].items()),"Independent gate omits frozen producer/prior evidence");common.current(inputs);proofs[group]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
    return dict(primary=witness,independent=proofs)


def identity_for(settings,fixture):
    return dict(schema=SCHEMA,settings=settings,fixture=fixture,rules_digest=config.RULES_DIGEST,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/label-divergence-actions.py"),control_model_sha256=sha(MODEL),candidate_model_sha256=sha(CANDIDATE),native_build_sha256=Native().build_id,baseline_sha256=sha(BASELINE),registry_sha256=sha(REGISTRY),plan_sha256=sha(PLAN),implementation_sha256=sha(IMPLEMENTATION),fixed_simulation_witness=None if fixture else gate(),panel_sha256=config.digest(expected_rows(settings,fixture)),fixture_source_journal_sha256=sha(ROOT/"data/midgame-label-pilot-v1/work.sqlite3") if fixture else None,reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def validate_tables(rows,stores,settings,closed=False):
    native=Native();signatures=defaultdict(dict)
    for index,stage in enumerate(STAGES):
        attempts,labels=stores[stage+"_attempts"],stores[stage+"_labels"]
        require(list(attempts)==list(rows)[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(rows)),"Teacher attempt prefix/labels differ")
        require(not attempts or not index or len(stores[STAGES[index-1]+"_attempts"])==len(rows),"Requested teacher stages out of order")
        for key,attempt in attempts.items():
            validate_attempt(rows[key],attempt,labels.get(key),stage,settings,native)
            for report in attempt["history"]:
                depth=report["choice_depth"];signature=q_signature(item_for(report,rows[key]));require(depth not in signatures[key] or equal(signatures[key][depth],signature),"Cross-attempt same-depth full Q/best mismatch");signatures[key][depth]=signature


def body_for(identity,rows,stores):
    summary={stage:dict(attempts=len(stores[stage+"_attempts"]),complete=len(stores[stage+"_labels"]),rejected=sum(attempt["status"]=="budget_rejected" for attempt in stores[stage+"_attempts"].values()),missing_requested_reference=sorted(set(rows)-set(stores[stage+"_labels"])),requested_depth=identity["settings"][stage]["depth"]) for stage in STAGES}
    nodes=sum(a["work"]["nodes"] for stage in STAGES for a in stores[stage+"_attempts"].values());require(nodes<=identity["settings"]["maximum_new_nodes"],"Requested teacher work exceeds predeclared ceiling")
    return dict(schema=SCHEMA,status="complete",identity=identity,roots=len(rows),summary=summary,requested_attempts=sum(len(stores[stage+"_attempts"]) for stage in STAGES),auxiliary_completed_reports=sum(len(a["history"]) for stage in STAGES for a in stores[stage+"_attempts"].values()),total_new_nodes=nodes,table_sha256=dict(roots=config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}),diagnostic_only=True,quotient_Q_reuse=False,auxiliary_horizon_fallback=False,deeper_is_not_game_truth=True,training_performed=False,torch_imported=False,sealed_final_test_consumed=False)


def metadata(db,key):
    row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Missing teacher metadata: "+key);value=json.loads(row[0]);require(config.digest(value)==row[1],"Corrupt teacher metadata: "+key);return value


def read_core(db,path):
    identity=metadata(db,"identity");fixture=identity["fixture"];settings=settings_for(fixture,identity["settings"]["roots"] if fixture else None)
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build"),"Teacher fixture must stay under build")
    else:require(Path(path).resolve()==DEFAULT.resolve(),"Production teacher path differs")
    admit()
    require(equal(identity,identity_for(settings,fixture)),"Exact teacher source/settings identity differs");rows=table(db,"roots");require(equal(rows,expected_rows(settings,fixture)),"Exact diagnostic roots changed");stores={name:table(db,name) for name in TABLES};validate_tables(rows,stores,settings,True);body=body_for(identity,rows,stores);require(equal(metadata(db,"complete"),body),"Teacher completed marker differs");return rows,stores,body


def run(path=DEFAULT,stage="all",fixture=False,roots=None,on_event=None,cancel=None):
    require("torch" not in sys.modules,"Teacher requires CPU-only process without Torch");path=Path(path).resolve();settings=settings_for(fixture,roots);require(stage in (*STAGES,"all"),"Unknown teacher stage")
    if fixture:require(path.is_relative_to(ROOT/"build"),"Teacher fixture must stay under build")
    else:require(path==DEFAULT.resolve(),"Production teacher path differs")
    admit()
    identity=identity_for(settings,fixture);work=phase6_store.Work(path,identity);event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        require(equal(work.get("identity"),identity),"Saved teacher identity typed fields differ")
        work.db.executescript("CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);"+"".join("CREATE TABLE IF NOT EXISTS "+name+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for name in TABLES));expected=expected_rows(settings,fixture);rows=table(work.db,"roots");stores={name:table(work.db,name) for name in TABLES}
        if not rows:
            require(not any(stores.values()),"Orphan teacher attempt/label tables")
            with work.db:
                for key,row in expected.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
            rows=expected
        require(equal(rows,expected),"Teacher original diagnostic roots differ");validate_tables(rows,stores,settings);complete=work.get("complete")
        if complete is None:
            for target_stage in STAGES:
                if stage!="all" and STAGES.index(target_stage)>STAGES.index(stage):break
                for key in sorted(set(rows)-set(stores[target_stage+"_attempts"])):
                    if stopped(cancel):return dict(status="interrupted",stage=target_stage)
                    item,attempt=execute(rows[key],target_stage,settings,Native(),cancel)
                    if attempt is None:return dict(status="interrupted",stage=target_stage)
                    proposed={name:dict(value) for name,value in stores.items()};proposed[target_stage+"_attempts"][key]=attempt
                    if item is not None:proposed[target_stage+"_labels"][key]=item
                    validate_tables(rows,proposed,settings);completed=sum(len(stores[s+"_attempts"]) for s in STAGES);values=dict(stage=target_stage,state_key=key,completed=completed,stage_completed=len(stores[target_stage+"_attempts"]),target=len(rows)*3,status=attempt["status"],requested_depth=settings[target_stage]["depth"],node_budget=settings[target_stage]["node_budget"],completed_depth=attempt["completed_depth"]);event("divergence_teacher_"+target_stage+"_precommit",**values);work.admit(len(config.canonical(attempt))*3+2*1024**2)
                    with work.db:
                        work.db.execute("INSERT INTO "+target_stage+"_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                        if item is not None:work.db.execute("INSERT INTO "+target_stage+"_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                    stores=proposed;values["completed"]+=1;values["stage_completed"]+=1;event("divergence_teacher_"+target_stage+"_committed",**values)
            if not all(len(stores[s+"_attempts"])==len(rows) for s in STAGES):return dict(status="stage_complete",stage=stage)
            validate_tables(rows,stores,settings,True);complete=body_for(identity,rows,stores)
            with work.db:work.put("complete",complete)
        _,_,body=read_core(work.db,path);publish(path/"manifest.json",body)
        if not fixture:publish(OUTPUT,body)
        event("divergence_teacher_complete",summary=body["summary"],total_new_nodes=body["total_new_nodes"]);return body
    finally:work.close()


def load_rows(path=DEFAULT):
    require("torch" not in sys.modules,"Teacher reader forbids Torch");path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");rows,stores,body=read_core(db,path);require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(body)+"\n","Teacher file/committed manifest differs");return rows,stores,body
