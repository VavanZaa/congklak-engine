"""Prospective fixed-cell weighting assessment; no optimizer or model work."""
from collections import Counter,defaultdict
from contextlib import closing
import ast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from . import config,pilot,horizon_probe as history,ordinary_value_pilot as prior,phase6_store
from . import qstability_common as common,qstability_metrics as metrics
from .native import ROOT,Native

SCHEMA="qstability-fixed-cell-max-action-assessment-v1"
DEFAULT=ROOT/"data/qstability-v1"
INVENTORY=ROOT/"build/qstability-inventory.json"
REGISTRY=ROOT/"build/qstability-registry.json"
RULE=ROOT/"build/qstability-rule.json"
RESERVATIONS=ROOT/"build/qstability-reservations.json"
PLAN=ROOT/"build/qstability-plan.json"
IMPLEMENTATION=ROOT/"build/qstability-implementation.json"
OUTPUT=ROOT/"build/qstability-result.json"
BASELINE=common.BASELINE
STRATA=("50-74/ordinary","25-49/ordinary","25-49/bonus","25-49/capture")
CELLS=tuple(s+"/actor"+str(a) for s in STRATA for a in (0,1))
STAGES=("h8","h10","h12")
TABLES=tuple(s+t for s in STAGES for t in ("_attempts","_labels"))
SELECT_SALT="qstability-saved-metadata-panel/20261101/v1/"
SOURCES=("congklak/qstability_weighting.py","tools/validate-qstability.py","tests/qstability_weighting_author.py","docs/QSTABILITY_WEIGHTING.md","docs/QSTABILITY_SCOPE.md","congklak/qstability_common.py")
DEPENDENCIES=("congklak/config.py","congklak/horizon_probe.py","congklak/deeper_teacher_v2.py","congklak/ordinary_value_pilot.py","congklak/natural_study.py","congklak/phase6_store.py","congklak/pilot.py","congklak/native.py","congklak/reports.py","congklak/qstability_metrics.py","tests/qstability_metrics_author.py","docs/QSTABILITY_METRICS.md","build/qstability-metrics-implementation.json")
CONTRACT=dict(roots=16,group="assessment",per_cell=2,cells=list(CELLS),distinct_origin_components=True,distinct_origin_families=True,selection_salt=SELECT_SALT,
 h8=dict(depth=8,node_budget=10000000),h10=dict(depth=10,node_budget=10000000),h12=dict(depth=12,node_budget=10000000),maximum_new_nodes=480000000,
 time_ms=0,tt_mb=16,use_pvs=True,memory_mb=512,keep_history=True,automatic_expansion=False,
 raw_rule="max(1/4,1/(1+max_legal_abs(Q10-Q8)/4))",weighted_probability="(1/8)*raw_reliability/sum_cell(raw_reliability)",uniform_probability="1/16",
 existing_H8_H10_exposure_allowed=True,H12_input_to_selection_or_rule=False,exact_rational=True,training_performed=False,weights_applied=False,promotion=False)
require,read,sha,publish=common.require,common.read,common.sha,common.publish
equal=common.same


def filehash(body):return hashlib.sha256((config.canonical(body)+os.linesep).encode("utf-8")).hexdigest()


def ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)


def rank(text):return hashlib.sha256((SELECT_SALT+text).encode()).hexdigest()


def inventory():
    """Accepted closed metadata only: no journal or target-body reads."""
    require("torch" not in sys.modules,"Q-stability study forbids Torch");baseline=common.baseline()
    names=("build/stratum-inventory.json","build/stratum-reservations.json","build/stratum-registry.json","build/allocation-weighting-reservations.json")
    require(set(names)<=set(baseline["inputs"]),"Missing accepted metadata ancestry");common.current({p:baseline["inputs"][p] for p in names})
    old=read(ROOT/names[0]);overlay=read(ROOT/names[1]);allocation=read(ROOT/names[3]);rf=set(overlay["origin_families"]);rq=set(overlay["eligible_quotients"]);reserved=set(overlay["components"])
    af={f for role in allocation["roles"].values() for f in role["origin_families"]};aq={q for role in allocation["roles"].values() for q in role["eligible_quotients"]}
    components={}
    for cid,c in old["components"].items():
        reasons=list(c["blocked_reasons"])
        if cid in reserved or set(c["families"])&rf or set(c["quotients"])&rq:reasons.append("prior_stratum_consumed_component")
        components[cid]=dict(c,blocked_reasons=reasons)
    eligible={};excluded={}
    for key,row in old["eligible"].items():
        cid=row["component"];reasons=components[cid]["blocked_reasons"]
        if reasons:excluded[key]=dict(component=cid,reasons=reasons);continue
        require(not(set(components[cid]["families"])&(af|rf)) and not(set(components[cid]["quotients"])&(aq|rq)),"Reserved component bridge survived")
        require(all(o["component"]==cid and o["source_candidate"]["cohort"]=="training_candidate" and o["source_candidate"]["state_key"]==key for o in row["all_origins"]),"All-origin source ownership differs")
        require(row["source_cohort"]=="training_candidate" and row["legal_mask"].bit_count()>=2 and row["stratum"] in STRATA and key==common.rawkey(row["state"]),"Candidate choice/state metadata differs")
        eligible[key]=row
    origins=[o for r in eligible.values() for o in r["all_origins"]]
    counts={c:dict(roots=sum(r["cell"]==c for r in eligible.values()),origin_families=len({r["family"] for r in eligible.values() if r["cell"]==c}),components=len({r["component"] for r in eligible.values() if r["cell"]==c})) for c in CELLS}
    return dict(schema=SCHEMA+"-metadata-inventory",eligible=dict(sorted(eligible.items())),components=components,excluded_roots=dict(sorted(excluded.items())),eligible_roots=len(eligible),representative_origin_families=len({r["family"] for r in eligible.values()}),all_origin_families=len({o["source_candidate"]["family"] for o in origins}),eligible_components=len({r["component"] for r in eligible.values()}),eligible_cells=counts,duplicate_raw_observations=len(origins)-len(eligible),source_families=old["source_families"],inherited_original_eligible_observations=old["eligible_candidate_observations"],inherited_eligible_roots=old["eligible_roots"],prior_stratum_reserved_components=sorted(reserved),prior_stratum_reserved_families=sorted(rf),prior_stratum_reserved_quotients=sorted(rq),ancestral_metadata_inputs=common.pins(names),H8_H10_target_bodies_read=False,H12_H14_target_bodies_read=False,shared_prefix_independence_claim=False)


def select(inv):
    slots=[(cell,i) for cell in CELLS for i in range(2)];eligible=inv["eligible"];ordered={cell:sorted((r for r in eligible.values() if r["cell"]==cell),key=lambda r:(rank(r["state_key"]),r["state_key"])) for cell in CELLS};assigned={};chosen={}
    def match(slot,seen):
        for row in ordered[slot[0]]:
            component=row["component"]
            if component in seen:continue
            seen.add(component)
            if component not in assigned or match(assigned[component],seen):assigned[component]=slot;chosen[slot]=row;return True
        return False
    for slot in slots:require(match(slot,set()),"Metadata matching shortage; no relaxed masks, replacement or new games")
    rows={r["state_key"]:dict(r,group="assessment",future_optimizer_reserved=True) for r in chosen.values()}
    require(len(rows)==len({r["component"] for r in rows.values()})==len({r["family"] for r in rows.values()})==16,"Panel origin/component support differs")
    return dict(sorted(rows.items()))


def projection():
    inv=inventory();rows=select(inv);registry=dict(schema=SCHEMA,rows=rows,inventory_sha256=filehash(inv),baseline_sha256=sha(BASELINE),selection_metadata_only=True,formula_frozen_before_measurements=True)
    reservations=dict(schema=SCHEMA+"-future-optimizer-reservations",registry_sha256=filehash(registry),selected_state_keys=sorted(rows),origin_families=sorted({f for r in rows.values() for f in inv["components"][r["component"]]["families"]}),eligible_quotients=sorted({q for r in rows.values() for q in inv["components"][r["component"]]["quotients"]}),components=sorted(r["component"] for r in rows.values()),future_optimizer_reserved=True,historical_roles_rewritten=False,ancestral_exposure_not_removed=True)
    return {INVENTORY:inv,REGISTRY:registry,RESERVATIONS:reservations}


def prepare():
    bodies=projection()
    for path,body in bodies.items():publish(path,body);require(sha(path)==filehash(body),"Published physical file identity differs")
    return bodies[REGISTRY]


def plan_body():return dict(schema=SCHEMA,contract=CONTRACT,baseline_sha256=sha(BASELINE),artifacts=common.pins([INVENTORY,REGISTRY,RESERVATIONS]),models=common.pins([common.CONTROL,common.CANDIDATE]),rules_digest=config.RULES_DIGEST,
    phase_order="Source/panel/formula -> preflight -> allH8/H10 -> immutable realizedrule+prefix -> independentrule04 -> allH12 -> metrics",incomplete_reference_policy="Retain every attempted root; incompleteH8/H10 holdsH12; missingH12 keeps originalmass and nullcomplete-risk/HOLD",no_training_application=True)


def plan():prepare();return publish(PLAN,plan_body())


def implementation_body():return dict(schema=SCHEMA+"-implementation",plan_sha256=sha(PLAN),sources=common.pins(SOURCES+DEPENDENCIES))


def freeze():plan();return publish(IMPLEMENTATION,implementation_body())


def admit():
    require("torch" not in sys.modules,"Q-stability study forbids Torch")
    for path,body in projection().items():require(equal(read(path),body),"Metadata projection changed: "+path.name)
    require(equal(read(PLAN),plan_body()) and equal(read(IMPLEMENTATION),implementation_body()),"Frozen Q-stability source/contract differs")
    return read(REGISTRY)


def settings_for(fixture=False,roots=None,variant="normal"):
    require(type(fixture) is bool,"Fixture flag must be boolean");require(variant in ("normal","reference-cap"),"Unknown fixture variant");require(fixture or variant=="normal","Production requires normal variant");settings=json.loads(config.canonical(CONTRACT));settings["variant"]=variant
    if fixture:
        settings.update(roots=2 if roots is None else config.integer(roots,"fixture_roots",1,2),fixture_panel="First2 previously consumed diagnostic roots, actual requesteddepth2/3/4, never currentassessment")
        for stage,depth in zip(STAGES,(2,3,4)):settings[stage]=dict(depth=depth,node_budget=10000000)
        if variant=="reference-cap":
            for stage in ("h8","h10"):settings[stage]["node_budget"]=1
    else:require(roots is None,"Production requires the exact16-root panel")
    return settings


def expected_rows(settings,fixture):
    if fixture:
        rows=prior.label_rows(dict(roots=settings["roots"]),True)
        return {k:dict(r,actor=r["state"]["controller"],group="assessment",component="fixture/"+k,cell=r["stratum"]+"/actor"+str(r["state"]["controller"]),source_cohort=r["cohort"]) for k,r in rows.items()}
    return {k:dict(root=r["source"]["source_candidate"],source_root_sha256=r["source"]["source_candidate_sha256"],state=r["state"],state_key=k,actor=r["actor"],family=r["family"],cohort="assessment",group="assessment",source_cohort="training_candidate",purpose="within_cell_weighting_assessment",legal_mask=r["legal_mask"],stratum=r["stratum"],component=r["component"],cell=r["cell"],source_metadata_sha256=config.digest(r),future_optimizer_reserved=True) for k,r in read(REGISTRY)["rows"].items()}


AUDIT_METHODS=dict(preflight=["test_01_prior5050_blind_support_components_and_fixed_rule","test_02_exact_max_action_weights_guards_ties_and_censoring","test_03_real_teacher_pre_post_commit_kills_and_exact_resume"],reference=["test_04_all32_references_PVs_and_pre_H12_realized_freeze"])


def proof(name):
    path=ROOT/"build/qstability-audit"/(name+".json");body=read(path);methods=AUDIT_METHODS[name]
    required=dict(schema="qstability-independent-group-v1",group=name,passed=True,tests_run=len(methods),errors=0,failures=0,skips=0,covered_methods=methods,covered_checks=["__main__.QStabilityIndependentAudit."+m for m in methods])
    require(equal({k:body[k] for k in required},required),"Independent stratum stage proof differs")
    inputs=body["inputs"];needed={"tests/qstability_independent_audit.py","tools/qstability-audit.py","build/qstability-audit-source-freeze.json","build/qstability-baseline.json","congklak/qstability_weighting.py","congklak/qstability_metrics.py","build/qstability-implementation.json","build/qstability-plan.json","build/qstability-registry.json","build/qstability-reservations.json"}
    require(needed<=set(inputs) and all(inputs.get(k)==v for k,v in common.baseline()["inputs"].items()),"Independent stage proof omits source/prior/phase identities");common.current(inputs)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def h12_gate(path,rows,stores,fixture):
    if fixture:return dict(fixture=True)
    require(all(len(stores[s+"_labels"])==16 for s in ("h8","h10")),"H12 HOLD: all32 references must be complete")
    bound=proof("reference");body=read(ROOT/bound["path"]);refs=body["evidence"]["references"]
    expected=dict(H12_ready=True,H8_labels=16,H10_labels=16,endpoint_item_sha256={s:{k:config.digest(v) for k,v in stores[s+"_labels"].items()} for s in ("h8","h10")},realized_rule=dict(path=RULE.relative_to(ROOT).as_posix(),sha256=sha(RULE)),references_prefix=dict(path=(Path(path)/"references-prefix.sqlite3").relative_to(ROOT).as_posix(),sha256=sha(Path(path)/"references-prefix.sqlite3")),references_manifest=dict(path=(Path(path)/"references-manifest.json").relative_to(ROOT).as_posix(),sha256=sha(Path(path)/"references-manifest.json")))
    require(equal({k:refs.get(k) for k in expected},expected),"H12 HOLD: independent complete-reference readiness differs")
    for field in ("realized_rule","references_prefix","references_manifest"):
        evidence=expected[field];require(body["inputs"].get(evidence["path"])==evidence["sha256"],"Reference proof omits immutable phase evidence")
    return dict(reference_audit=bound,rule_sha256=sha(RULE))


def owner_gate():
    out={}
    for path,test,count in ((ROOT/"build/qstability-weighting-author.json","tests/qstability_weighting_author.py",4),(ROOT/"build/qstability-metrics-author.json","tests/qstability_metrics_author.py",4)):
        body=read(path);require(body["passed"] is True and type(body["tests_run"]) is int and body["tests_run"]==count and all(type(body[k]) is int and body[k]==0 for k in ("errors","failures","skips")),"Owner proof incomplete")
        tree=ast.parse((ROOT/test).read_text(encoding="utf-8-sig"));expected={node.name+"."+method.name for node in tree.body if isinstance(node,ast.ClassDef) for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith("test_")};ids=body["covered_methods"]
        require(len(ids)==len(set(ids))==count and {".".join(m.split(".")[-2:]) for m in ids}==expected,"Owner executed coverage differs from source")
        require(body["inputs"].get(test)==sha(ROOT/test),"Owner source is unpinned");common.current(body["inputs"]);out[path.name]=sha(path)
    return out


def initial_gate():return dict(independent_preflight=proof("preflight"),owners=owner_gate())


def identity_for(settings,fixture):return dict(schema=SCHEMA,fixture=fixture,settings=settings,rules_digest=config.RULES_DIGEST,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/validate-qstability.py"),baseline_sha256=sha(BASELINE),artifacts=common.pins([INVENTORY,REGISTRY,RESERVATIONS,PLAN,IMPLEMENTATION]),model_files=common.pins([common.CONTROL,common.CANDIDATE]),native_build_sha256=Native().build_id,production_gate=None if fixture else initial_gate(),panel_sha256=config.digest(expected_rows(settings,fixture)),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def metadata(db,key):
    row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Missing stratum metadata: "+key);body=json.loads(row[0]);require(config.digest(body)==row[1],"Corrupt stratum metadata: "+key);return body


def validate_schema(db):
    require({r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}=={"meta","roots",*TABLES} and db.execute("PRAGMA user_version").fetchone()[0]==6 and db.execute("PRAGMA quick_check").fetchone()[0]=="ok","Unknown/corrupt stratum schema")
    keys={r[0] for r in db.execute("SELECT key FROM meta")};require("identity" in keys and keys<={"identity","references_complete","weights_frozen","h12_gate","complete"},"Unknown stratum metadata");return keys


def validate_tables(rows,stores,settings,fixture):
    native=Native();keys=list(rows);overlap={}
    for index,stage in enumerate(STAGES):
        attempts,labels=stores[stage+"_attempts"],stores[stage+"_labels"]
        require(list(attempts)==keys[:len(attempts)] and set(labels)<=set(attempts),"Attempt prefix/label ownership differs")
        if attempts and index:require(len(stores[STAGES[index-1]+"_attempts"])==len(rows),"Teacher stages are out of order")
        for key,a in attempts.items():
            history.validate_attempt(rows[key],a,labels.get(key),stage,settings,native)
            for report in a["history"]:
                identity=(key,report["choice_depth"]);signature=history.q_signature(history.item_for(report,rows[key]))
                require(identity not in overlap or equal(overlap[identity],signature),"Cross-stage completed-history Q/best differs")
                overlap[identity]=signature
                for earlier in STAGES[:index]:
                    if report["choice_depth"]==settings[earlier]["depth"] and key in stores[earlier+"_labels"]:
                        require(equal(history.q_signature(history.item_for(report,rows[key])),history.q_signature(stores[earlier+"_labels"][key])),"Cross-horizon same-depth Q/best differs")
    if stores["h12_attempts"] and not fixture:require(all(len(stores[s+"_labels"])==len(rows) for s in ("h8","h10")),"H12 cannot follow incomplete H8/H10")


def reference_manifest(identity,rows,stores):return dict(schema=SCHEMA+"-references",status="references_complete",identity=identity,requested_roots=len(rows),summary={s:dict(attempts=len(stores[s+"_attempts"]),complete=len(stores[s+"_labels"]),missing=sorted(set(rows)-set(stores[s+"_labels"]))) for s in ("h8","h10")},table_sha256=dict(roots=config.digest(rows),**{t:config.digest(stores[t]) for t in TABLES if not t.startswith("h12")}),all_requested_complete=all(len(stores[s+"_labels"])==len(rows) for s in ("h8","h10")),H12_started=False)


def progress_manifest(identity,rows,stores,keys):
    refs=all(len(stores[s+"_attempts"])==len(rows) for s in ("h8","h10"));labels=all(len(stores[s+"_labels"])==len(rows) for s in ("h8","h10"));done=len(stores["h12_attempts"])==len(rows)
    status="complete" if done else "references_incomplete_hold" if refs and not labels else "h12_pending" if "h12_gate" in keys else "awaiting_rule_audit" if "weights_frozen" in keys else "awaiting_realized_weights" if refs else "references_pending"
    nodes=sum(a["work"]["nodes"] for s in STAGES for a in stores[s+"_attempts"].values());require(nodes<=CONTRACT["maximum_new_nodes"],"Committed search work exceeds ceiling")
    return dict(schema=SCHEMA,status=status,protocol_closed=done or (refs and not labels),assessment_available=done,H12_held=refs and not labels,identity=identity,requested_roots=len(rows),summary={s:dict(attempts=len(stores[s+"_attempts"]),complete=len(stores[s+"_labels"]),rejected=sum(a["status"]=="budget_rejected" for a in stores[s+"_attempts"].values()),missing=sorted(set(rows)-set(stores[s+"_labels"]))) for s in STAGES},table_sha256=dict(roots=config.digest(rows),**{t:config.digest(v) for t,v in stores.items()}),total_new_nodes=nodes,auxiliary_completed_reports=sum(len(a["history"]) for s in STAGES for a in stores[s+"_attempts"].values()),weights_applied=False,training_performed=False,promotion=False,torch_imported=False,future_optimizer_reserved=not identity["fixture"])


def rule_path(path,fixture):return Path(path)/"rule.json" if fixture else RULE


def realized_body(path,identity,rows,stores):
    prefix=Path(path)/"references-prefix.sqlite3";refs=Path(path)/"references-manifest.json"
    require(prefix.is_file() and refs.is_file(),"Missing immutable H8/H10 prefix publication")
    return dict(schema=SCHEMA+"-realized-rule",fixture=identity["fixture"],registry_sha256=sha(REGISTRY),implementation_sha256=sha(IMPLEMENTATION),references_manifest_sha256=sha(refs),references_prefix_sha256=sha(prefix),endpoint_item_sha256={s:{k:config.digest(v) for k,v in stores[s+"_labels"].items()} for s in ("h8","h10")},
        projection=None if identity["fixture"] else metrics.weight_projection(rows,stores["h8_labels"],stores["h10_labels"]),fixture_note="No production probability projection for old two-root engineering fixture" if identity["fixture"] else None,H12_used=False,training_application_authorized=False)


def validate_prefix(path,identity,rows,stores):
    prefix=Path(path)/"references-prefix.sqlite3"
    with closing(ro(prefix)) as db:
        keys=validate_schema(db);require(keys=={"identity","references_complete"},"Immutable references snapshot phase differs")
        require(equal(metadata(db,"identity"),identity) and equal(history.table(db,"roots"),rows),"Immutable reference identity/roots differ")
        for t in TABLES:
            require(equal(history.table(db,t),{} if t.startswith("h12") else stores[t]),"Immutable references differ from live committed tables")
        require(equal(metadata(db,"references_complete"),reference_manifest(identity,rows,stores)),"Immutable reference marker differs")


def read_core(db,path,require_complete=False,require_exports=True):
    keys=validate_schema(db);identity=metadata(db,"identity");fixture=identity["fixture"];settings=settings_for(fixture,identity["settings"]["roots"] if fixture else None,identity["settings"]["variant"]);admit()
    path=Path(path).resolve();require(path.is_relative_to(ROOT/"build") if fixture else path==DEFAULT.resolve(),"Journal path differs")
    require(equal(identity,identity_for(settings,fixture)),"Exact typed identity differs");rows=history.table(db,"roots");require(equal(rows,expected_rows(settings,fixture)),"Selected roots changed")
    stores={t:history.table(db,t) for t in TABLES};validate_tables(rows,stores,settings,fixture)
    if "references_complete" in keys:
        require(all(len(stores[s+"_attempts"])==len(rows) for s in ("h8","h10")),"Premature references marker")
        ref=reference_manifest(identity,rows,stores);require(equal(metadata(db,"references_complete"),ref),"References marker differs")
        if require_exports:require(equal(read(path/"references-manifest.json"),ref),"Reference file differs from marker")
    if "weights_frozen" in keys:
        require("references_complete" in keys and all(len(stores[s+"_labels"])==len(rows) for s in ("h8","h10")),"Weights exist without complete H8/H10")
        validate_prefix(path,identity,rows,stores);rule=realized_body(path,identity,rows,stores);require(equal(read(rule_path(path,fixture)),rule),"Realized rule differs from exact reference endpoints")
        require(equal(metadata(db,"weights_frozen"),dict(schema=SCHEMA+"-weights-frozen",rule_sha256=sha(rule_path(path,fixture)),references_prefix_sha256=sha(path/"references-prefix.sqlite3"))),"Weights marker differs")
    if "h12_gate" in keys:
        require("weights_frozen" in keys,"H12 gate predates realized weights");require(equal(metadata(db,"h12_gate"),h12_gate(path,rows,stores,fixture)),"H12 audit gate differs")
    require(not stores["h12_attempts"] or "h12_gate" in keys,"H12 work preceded audit admission")
    body=progress_manifest(identity,rows,stores,keys)
    if "complete" in keys:
        require(body["protocol_closed"] and equal(metadata(db,"complete"),body),"Completed marker differs")
        if body["status"]=="references_incomplete_hold":require("references_complete" in keys and not({"weights_frozen","h12_gate"}&keys) and not any(stores[t] for t in TABLES if t.startswith("h12")) and not rule_path(path,fixture).exists(),"Reference HOLD has missing reference marker or forbidden rule/H12 evidence")
        if require_exports:require(equal(read(path/"manifest.json"),body),"Complete manifest differs")
    else:require(not require_complete,"Study has no completed all-horizon marker")
    return rows,stores,body


def load_rows(path=DEFAULT,allow_partial=False):
    require("torch" not in sys.modules,"Reader forbids Torch");require(type(allow_partial) is bool,"Partial flag must be boolean");path=Path(path).resolve()
    with closing(ro(path/"work.sqlite3")) as db:db.execute("BEGIN");return read_core(db,path,not allow_partial)


def realize(path=DEFAULT):
    """Freeze exact H8/H10 weights before any production H12 attempt."""
    path=Path(path).resolve();rows,stores,body=load_rows(path,True);identity=body["identity"];fixture=identity["fixture"]
    require(all(len(stores[s+"_labels"])==len(rows) for s in ("h8","h10")),"Incomplete H8/H10: no usable weights, H12 HOLD")
    work=phase6_store.Work(path,identity)
    try:
        # Completed realization is a strict no-write path.
        if work.get("weights_frozen") is not None:read_core(work.db,path);return read(rule_path(path,fixture))
        require(not stores["h12_attempts"] and work.get("h12_gate") is None,"Late realized weights forbidden")
        snapshot=path/"references-prefix.sqlite3"
        if not snapshot.exists():
            temp=path/"references-prefix.tmp.sqlite3"
            if temp.exists():temp.unlink()
            with closing(sqlite3.connect(temp)) as destination:work.db.backup(destination)
            os.replace(temp,snapshot)
        validate_prefix(path,identity,rows,stores);rule=realized_body(path,identity,rows,stores);publish(rule_path(path,fixture),rule)
        with work.db:work.put("weights_frozen",dict(schema=SCHEMA+"-weights-frozen",rule_sha256=sha(rule_path(path,fixture)),references_prefix_sha256=sha(snapshot)))
        read_core(work.db,path);return rule
    finally:work.close()


freeze_rule=realize


def run(path=DEFAULT,stage="all",fixture=False,roots=None,on_event=None,cancel=None,variant="normal"):
    require("torch" not in sys.modules,"Producer forbids Torch");require(stage in ("all",*STAGES),"Unknown teacher stage");settings=settings_for(fixture,roots,variant);path=Path(path).resolve();require(path.is_relative_to(ROOT/"build") if fixture else path==DEFAULT.resolve(),"Journal path differs");admit();identity=identity_for(settings,fixture);work=phase6_store.Work(path,identity)
    event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        require(equal(work.get("identity"),identity),"Saved typed identity differs");work.db.executescript("CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);"+"".join("CREATE TABLE IF NOT EXISTS "+t+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for t in TABLES));validate_schema(work.db)
        rows=history.table(work.db,"roots");stores={t:history.table(work.db,t) for t in TABLES}
        if not rows:
            require(not any(stores.values()),"Orphan attempts/labels")
            rows=expected_rows(settings,fixture)
            with work.db:
                for key,row in rows.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
        require(equal(rows,expected_rows(settings,fixture)),"Root projection differs");validate_tables(rows,stores,settings,fixture)
        # Validate any existing closed/phase markers BEFORE export repair or search.
        read_core(work.db,path,require_exports=False)
        if work.get("references_complete") is not None:publish(path/"references-manifest.json",reference_manifest(identity,rows,stores))
        if work.get("complete") is not None:
            _,_,body=read_core(work.db,path,True,False);publish(path/"manifest.json",body);return body
        for selected_stage in (STAGES if stage=="all" else (stage,)):
            index=STAGES.index(selected_stage)
            if index:require(len(stores[STAGES[index-1]+"_attempts"])==len(rows),"Run the preceding horizon first")
            if selected_stage=="h12":
                require(work.get("weights_frozen") is not None,"H12 HOLD: realize H8/H10 weights and audit rule first")
                gate=h12_gate(path,rows,stores,fixture)
                require(equal(work.get("h12_gate"),gate) if work.get("h12_gate") is not None else True,"Saved H12 gate changed")
                if work.get("h12_gate") is None:
                    with work.db:work.put("h12_gate",gate)
            for key in sorted(set(rows)-set(stores[selected_stage+"_attempts"])):
                if history.stopped(cancel):return dict(status="interrupted")
                item,attempt=history.execute(rows[key],selected_stage,settings,Native(),cancel)
                if attempt is None:return dict(status="interrupted")
                proposed={t:dict(v) for t,v in stores.items()};proposed[selected_stage+"_attempts"][key]=attempt
                if item is not None:proposed[selected_stage+"_labels"][key]=item
                validate_tables(rows,proposed,settings,fixture);count=sum(len(stores[s+"_attempts"]) for s in STAGES);values=dict(stage=selected_stage,state_key=key,completed=count,stage_completed=len(stores[selected_stage+"_attempts"]),target=len(rows),status=attempt["status"],requested_depth=settings[selected_stage]["depth"],completed_depth=attempt["completed_depth"])
                event("qstability_"+selected_stage+"_precommit",**values);work.admit(len(config.canonical(attempt))*3+2*1024**2)
                with work.db:
                    work.db.execute("INSERT INTO "+selected_stage+"_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                    if item is not None:work.db.execute("INSERT INTO "+selected_stage+"_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                stores=proposed;values["completed"]+=1;values["stage_completed"]+=1;event("qstability_"+selected_stage+"_committed",**values)
            if selected_stage=="h10":
                refs=reference_manifest(identity,rows,stores)
                if work.get("references_complete") is None:
                    with work.db:work.put("references_complete",refs)
                require(equal(work.get("references_complete"),refs),"Reference marker changed");publish(path/"references-manifest.json",refs)
                body=progress_manifest(identity,rows,stores,validate_schema(work.db))
                if body["status"]=="references_incomplete_hold":
                    with work.db:work.put("complete",body)
                    _,_,body=read_core(work.db,path,True,False);publish(path/"manifest.json",body);event("qstability_phase_complete",status=body["status"],summary=body["summary"],total_new_nodes=body["total_new_nodes"]);return body
                # A fixture has no production weights; the public stages are still explicit.
            if selected_stage=="h12":
                body=progress_manifest(identity,rows,stores,validate_schema(work.db))
                with work.db:work.put("complete",body)
        _,_,body=read_core(work.db,path,require_exports=False)
        if body["protocol_closed"]:publish(path/"manifest.json",body)
        event("qstability_phase_complete",status=body["status"],summary=body["summary"],total_new_nodes=body["total_new_nodes"]);return body
    finally:work.close()


def build():
    rows,stores,manifest=load_rows();held=manifest["status"]=="references_incomplete_hold"
    files=[BASELINE,INVENTORY,REGISTRY,RESERVATIONS,PLAN,IMPLEMENTATION,DEFAULT/"work.sqlite3",DEFAULT/"manifest.json",DEFAULT/"references-manifest.json",*SOURCES,*DEPENDENCIES]
    if held:
        report=dict(requested_roots=len(rows),complete_roots=0,metrics=None,screen=dict(available=False,passes=False,status="HOLD",reason="Incomplete requested H8/H10 references; no realized weighted distribution"),uniform_probability=dict(numerator=1,denominator=16,value=1/16),weighted_probability=None,weighted_original_mass=None,H12_used=False)
    else:
        files.extend([RULE,DEFAULT/"references-prefix.sqlite3"]);report=metrics.evaluate(rows,read(RULE)["projection"],stores["h12_labels"])
    return dict(schema=SCHEMA+"-result",passed=True,status=manifest["status"],protocol_closed=True,assessment_available=manifest["assessment_available"],H12_held=held,requested_roots=len(rows),complete_roots=report["complete_roots"],summary=manifest["summary"],total_new_nodes=manifest["total_new_nodes"],auxiliary_completed_reports=manifest["auxiliary_completed_reports"],metrics=report["metrics"],screen=report["screen"],assessment=report,
        realized_rule_sha256=None if held else sha(RULE),manifest_sha256=sha(DEFAULT/"manifest.json"),weights_applied=False,training_performed=False,promotion=False,previous_allocation="hold",future_optimizer_reserved=True,H12_is_game_truth=False,population_claim=False,calibrated_confidence_claim=False,
        inputs=common.pins(files))


def assess():return publish(OUTPUT,build())


def verify():
    body=read(OUTPUT);require(equal(body,build()),"Assessment metrics/source identity do not rederive");return body
