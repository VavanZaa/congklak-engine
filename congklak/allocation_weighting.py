"""Metadata-first, reserved finite-reference assessment of one unfit weighting rule."""
from collections import Counter,defaultdict
from contextlib import closing
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from . import config,pilot,horizon_probe as history,ordinary_value_pilot as prior,phase6_store,allocation_common as common
from .native import ROOT,Native

SCHEMA="allocation-separate-exact-rational-weighting-assessment-v1"
DEFAULT=ROOT/"data/allocation-weighting-v1"
INVENTORY=ROOT/"build/allocation-weighting-inventory.json"
REGISTRY=ROOT/"build/allocation-weighting-registry.json"
RULE=ROOT/"build/allocation-weighting-rule.json"
RESERVATIONS=ROOT/"build/allocation-weighting-reservations.json"
PLAN=ROOT/"build/allocation-weighting-plan.json"
IMPLEMENTATION=ROOT/"build/allocation-weighting-implementation.json"
OUTPUT=ROOT/"build/allocation-weighting-result.json"
BASELINE=common.BASELINE
STRATA=("50-74/ordinary","25-49/ordinary","25-49/bonus","25-49/capture")
CELLS=tuple(s+"/actor"+str(a) for s in STRATA for a in (0,1))
SELECT_SALT="allocation-weighting-panel/20261030/v1/"
SPLIT_SALT="allocation-weighting-development-assessment/20261030/v1/"
SOURCE_JOURNALS=("natural-study-v1","midgame-label-pilot-v1")
FAMILY_JOURNALS=("ordinary-data-pilot-v1","natural-study-v1","midgame-data-pilot-v1")
TABLES=("h8_references","h10_references","h12_attempts","h12_labels")
SOURCES=("congklak/allocation_weighting.py","tools/validate-allocation-weighting.py","tests/allocation_weighting_author.py","docs/ALLOCATION_WEIGHTING.md","docs/ALLOCATION_SCOPE.md","congklak/allocation_common.py")
DEPENDENCIES=("congklak/horizon_probe.py","congklak/natural_study.py","congklak/ordinary_value_pilot.py","congklak/phase6_store.py","congklak/pilot.py","congklak/native.py","congklak/reports.py","congklak/allocation_trace.py")
CONTRACT=dict(roots=16,groups=dict(development=8,assessment=8),cells=list(CELLS),per_cell=2,selected_components_unique=True,selected_families_unique=True,selection_salt=SELECT_SALT,split_salt=SPLIT_SALT,h12=dict(depth=12,node_budget=10000000),maximum_new_nodes=160000000,time_ms=0,tt_mb=16,use_pvs=True,memory_mb=512,keep_history=True,automatic_expansion=False,rule="max(1/4,1/(1+abs(V10-V8)/4+mean_legal_abs(Q10-Q8)/4))",exact_rational=True,primary="assessment:weighted_vs_uniform_abs(V12-V10)",screen="all8complete;rootdriftstrictlylower;actiondriftnohigher;ESS>=4N/5",training_performed=False,weights_applied=False,promotion=False)
require,read,sha,publish=common.require,common.read,common.sha,common.publish
equal=common.same


def ro(name):return sqlite3.connect((ROOT/"data"/name/"work.sqlite3").as_uri()+"?mode=ro",uri=True)


def rows_at(name,table,key="state"):
    with closing(ro(name)) as db:return history.table(db,table,key)


def label_keys(name,stage,attempts=False):
    with closing(ro(name)) as db:
        tables={v[0] for v in db.execute("SELECT name FROM sqlite_master")};keys=set()
        for table in ((stage+"_attempts",) if attempts else (stage+"_labels",stage+"_references")):
            if table in tables:keys.update(v[0] for v in db.execute("SELECT state FROM "+table))
        return keys


def rank(salt,text):return hashlib.sha256((salt+text).encode()).hexdigest()


def rational(value):
    value=F(value);return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def fraction(value):return F(value["numerator"],value["denominator"])


def inventory():
    """Only metadata and requested-label keys; never H12/H14 target bodies."""
    common.baseline(verify_all=True);ledger=read(ROOT/"build/ordinary-value-registry.json");protectedq=set(ledger["quotients"]);protectedf=set(ledger["families"])
    observations=0
    with closing(ro("ordinary-value-benchmark-v1")) as db:
        require(db.execute("SELECT count(*) FROM games").fetchone()[0]==96,"Latest evaluation panel differs")
        for before,after in db.execute("SELECT json_extract(body,'$.state'),json_extract(body,'$.after') FROM decisions"):
            for text in (before,after):protectedq.add(pilot.quotient(json.loads(text)).hex());observations+=1
        for family,final in db.execute("SELECT json_extract(body,'$.family'),json_extract(body,'$.final') FROM games"):
            if family:protectedf.add(family)
            if final:protectedq.add(pilot.quotient(json.loads(final)).hex())
    forensic=read(ROOT/"build/divergence-panel.json")["roots"];protectedf.update(row["family"] for row in forensic);protectedq.update(pilot.quotient(row["state"]).hex() for row in forensic)
    exposed=set();exposedq=set();exposedf=set()
    for name in ("horizon-probe-v1","horizon-finish-v1","midgame-label-pilot-v1","ordinary-value-labels-v1","divergence-teacher-v1"):
        keys=set()
        for stage in ("h12","h14"):keys.update(label_keys(name,stage));keys.update(label_keys(name,stage,True))
        exposed.update(keys)
        for key,row in rows_at(name,"roots").items():
            if key in keys:exposedq.add(pilot.quotient(row["state"]).hex());exposedf.add(row["family"])
    candidates={}
    for name in SOURCE_JOURNALS:
        support=label_keys(name,"h8")&label_keys(name,"h10")
        for key,row in rows_at(name,"roots").items():
            if key in support and row["cohort"]=="training_candidate":
                require(key not in candidates,"Duplicate source root needs explicit ownership")
                candidates[key]=dict(source=name,source_row=row,source_wrapper_sha256=config.digest(row),state_key=key,state=row["state"],family=row["family"],stratum=row["stratum"],actor=row["state"]["controller"],legal_mask=row["legal_mask"],source_cohort=row["cohort"])
    parent={};owners={};candidate_observations=0
    def find(node):
        parent.setdefault(node,node)
        if parent[node]!=node:parent[node]=find(parent[node])
        return parent[node]
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    for name in FAMILY_JOURNALS:
        for family in rows_at(name,"families","idx").values():
            fid=family["family"];require(fid not in owners,"Duplicate source family")
            owners[fid]=family["cohort"];find(("family",fid))
            for row in family["candidates"]:
                if row["stratum"] in STRATA and row["legal_mask"].bit_count()>=2:
                    join(("family",fid),("quotient",pilot.quotient(row["state"]).hex()));candidate_observations+=1
    groups=defaultdict(list)
    for node in list(parent):groups[find(node)].append(node)
    components={};byfamily={}
    for nodes in groups.values():
        families=sorted(v for kind,v in nodes if kind=="family");quotients=sorted(v for kind,v in nodes if kind=="quotient");cohorts=sorted({owners[f] for f in families});reasons=[]
        if set(families)&protectedf:reasons.append("protected_origin_family")
        if set(quotients)&protectedq:reasons.append("protected_eligible_quotient_bridge")
        if set(families)&exposedf or set(quotients)&exposedq:reasons.append("prior_requested_H12_H14_exposed_component")
        if len(cohorts)>1:reasons.append("mixed_source_ownership_component")
        body=dict(families=families,quotients=quotients,source_cohorts=cohorts,blocked_reasons=reasons);cid=config.digest(body);components[cid]=body
        for fid in families:byfamily[fid]=cid
    eligible={};excluded={}
    for key,row in candidates.items():
        require(row["family"] in byfamily,"Candidate source family missing from full metadata pool");cid=byfamily[row["family"]];reasons=list(components[cid]["blocked_reasons"])
        if key in exposed:reasons.append("prior_requested_H12_H14_exposure")
        if row["stratum"] not in STRATA or row["legal_mask"].bit_count()<2:reasons.append("outside_declared_choice_support")
        value=dict(row,component=cid,cell=row["stratum"]+"/actor"+str(row["actor"]),quotient=pilot.quotient(row["state"]).hex())
        if reasons:excluded[key]=dict(value,reasons=reasons)
        else:eligible[key]=value
    return dict(schema="allocation-weighting-metadata-inventory-v1",eligible=dict(sorted(eligible.items())),excluded=dict(sorted(excluded.items())),components=components,protected_quotients=sorted(protectedq),protected_families=sorted(protectedf),prior_H12_H14_exposed_rawkeys=sorted(exposed),prior_H12_H14_exposed_quotients=sorted(exposedq),prior_H12_H14_exposed_families=sorted(exposedf),raw_H8_H10_training_roots=len(candidates),eligible_roots=len(eligible),eligible_families=len({r["family"] for r in eligible.values()}),eligible_cells=dict(sorted(Counter(r["cell"] for r in eligible.values()).items())),source_families=len(owners),eligible_candidate_observations=candidate_observations,blocked_source_families=sum(len(c["families"]) for c in components.values() if c["blocked_reasons"]),latest_evaluation_state_observations=observations,H12_H14_bodies_read_for_selection=False,shared_prefix_independence_claim=False)


def select(inventory_value):
    eligible=inventory_value["eligible"];slots=[(cell,index) for cell in CELLS for index in range(2)];ordered={cell:sorted((r for r in eligible.values() if r["cell"]==cell),key=lambda r:(rank(SELECT_SALT,r["state_key"]),r["state_key"])) for cell in CELLS};assigned={};chosen={}
    def assign(slot,seen):
        for row in ordered[slot[0]]:
            component=row["component"]
            if component in seen:continue
            seen.add(component)
            if component not in assigned or assign(assigned[component],seen):assigned[component]=slot;chosen[slot]=row;return True
        return False
    for slot in slots:require(assign(slot,set()),"Metadata matching shortfall; no replacement or relaxed quotas")
    selected={}
    for cell in CELLS:
        pair=sorted((chosen[slot] for slot in slots if slot[0]==cell),key=lambda r:(rank(SPLIT_SALT,r["state_key"]),r["state_key"]))
        for group,row in zip(("development","assessment"),pair):selected[row["state_key"]]=dict(row,group=group,future_optimizer_reserved=True)
    require(len(selected)==len({r["family"] for r in selected.values()})==len({r["component"] for r in selected.values()})==16,"Selected origin/component support differs")
    return dict(sorted(selected.items()))


def endpoint(name,stage,key):
    with closing(ro(name)) as db:
        row=db.execute("SELECT body,hash FROM "+stage+"_labels WHERE state=?",(key,)).fetchone();require(row is not None,"Requested H8/H10 endpoint missing");item=json.loads(row[0]);require(config.digest(item)==row[1],"Corrupt requested endpoint");return item


def weight_rule(h8,h10,legal):
    a,b=h8["label"],h10["label"];qa={v["pit"]:v["value_player0"] for v in a["action_scores"]};qb={v["pit"]:v["value_player0"] for v in b["action_scores"]};require(set(qa)==set(qb)==set(legal),"Rule legal action support differs");root=abs(b["target_player0"]-a["target_player0"]);action=F(sum(abs(qb[p]-qa[p]) for p in legal),len(legal));weight=max(F(1,4),1/(1+F(root,4)+action/4))
    return dict(root_shift_seeds=root,mean_action_shift_seeds=rational(action),weight=rational(weight),formula=CONTRACT["rule"],requested_inputs=[8,10],H12_input=False)


def filehash(body):return hashlib.sha256((config.canonical(body)+os.linesep).encode("utf-8")).hexdigest()


def direct_model_root_exposure(selected):
    oldkeys=set(read(ROOT/"build/quality-data.json")["train_state_keys"]);oldq={pilot.quotient(pilot.from_key(bytes.fromhex(key))).hex() for key in oldkeys};groups=read(ROOT/"build/ordinary-value-rows.json")["groups"];rows=[r for values in groups.values() for r in values];newkeys={r["state_key"] for r in rows};newq={pilot.quotient(r["state"]).hex() for r in rows}
    return dict(scope="Direct admitted optimizer-root membership only; no inference about internal search states, feature similarity or causal ancestral effects",rows={key:dict(control_ancestral_root_raw=row["state_key"] in oldkeys,control_ancestral_root_quotient=row["quotient"] in oldq,candidate_additional_root_raw=row["state_key"] in newkeys,candidate_additional_root_quotient=row["quotient"] in newq) for key,row in selected.items()},source_inputs=common.pins(["build/quality-data.json","build/ordinary-value-rows.json"]),H8_H10_previously_consumed=True,H12_rule_assessment_not_previously_requested=True,unseen_release_holdout_claim=False)


def projection():
    inv=inventory();selected=select(inv);rule={};native=Native()
    for key,row in selected.items():
        refs={s:endpoint(row["source"],s,key) for s in ("h8","h10")}
        for stage,item in refs.items():history.old.validate_label(item,row["source_row"],int(stage[1:]),native)
        legal=[p for p in range(7) if row["legal_mask"]&(1<<p)];rule[key]=dict(h8=refs["h8"],h10=refs["h10"],h8_item_sha256=config.digest(refs["h8"]),h10_item_sha256=config.digest(refs["h10"]),source_root_sha256=row["source_row"]["source_root_sha256"],source_wrapper_sha256=row["source_wrapper_sha256"],rule=weight_rule(refs["h8"],refs["h10"],legal))
    reservations={group:dict(origin_families=sorted({f for r in selected.values() if r["group"]==group for f in inv["components"][r["component"]]["families"]}),eligible_quotients=sorted({q for r in selected.values() if r["group"]==group for q in inv["components"][r["component"]]["quotients"]}),selected_state_keys=sorted(k for k,r in selected.items() if r["group"]==group)) for group in ("development","assessment")}
    require(not(set(reservations["development"]["eligible_quotients"])&set(reservations["assessment"]["eligible_quotients"])),"New reservation roles share eligible quotient support")
    registry=dict(schema=SCHEMA,rows=selected,inventory_sha256=filehash(inv),baseline_sha256=sha(BASELINE),source_cohort_preserved=True,direct_model_root_exposure=direct_model_root_exposure(selected))
    rulebody=dict(schema=SCHEMA,rows=rule,registry_sha256=filehash(registry),exact_rational=True,H12_used=False)
    reserved=dict(schema="allocation-weighting-future-optimizer-reservations-v1",roles=reservations,future_optimizer_reserved=True,historical_roles_rewritten=False,ancestral_exposure_not_removed=True,registry_sha256=filehash(registry))
    return {INVENTORY:inv,REGISTRY:registry,RULE:rulebody,RESERVATIONS:reserved}


def prepare():
    bodies=projection()
    for path,body in bodies.items():publish(path,body);require(sha(path)==filehash(body),"Published file byte identity differs: "+path.name)
    return bodies[REGISTRY],bodies[RULE]


def plan_body():
    return dict(schema=SCHEMA,contract=CONTRACT,baseline_sha256=sha(BASELINE),artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (INVENTORY,REGISTRY,RULE,RESERVATIONS)},models=common.pins([common.CONTROL,common.CANDIDATE]),rules_digest=config.RULES_DIGEST,production_gate="Frozen preflight3 and trace04 plus genuine8cell/3840simulation trace witness",no_automatic_training_application=True)


def plan():
    prepare();return publish(PLAN,plan_body())


def implementation_body():return dict(schema=SCHEMA+"-implementation",plan_sha256=sha(PLAN),sources=common.pins(SOURCES+DEPENDENCIES))


def freeze():
    plan();return publish(IMPLEMENTATION,implementation_body())


def admit():
    require("torch" not in sys.modules,"Weighting study forbids Torch")
    for path,body in projection().items():require(equal(read(path),body),"Frozen metadata/rule/reservation projection differs: "+path.name)
    require(equal(read(PLAN),plan_body()) and equal(read(IMPLEMENTATION),implementation_body()),"Frozen weighting source/plan differs")
    return read(REGISTRY),read(RULE)


def settings_for(fixture=False,roots=None):
    require(type(fixture) is bool,"Fixture flag must be boolean");settings=json.loads(config.canonical(CONTRACT))
    if fixture:settings.update(roots=2 if roots is None else config.integer(roots,"fixture_roots",1,2),h12=dict(depth=4,node_budget=10000000),fixture_panel="First two already-consumed diagnostic roots; actual requesteddepth4, never production assessment")
    else:require(roots is None,"Production requires all16 preselected roots")
    return settings


def expected_rows(settings,fixture):
    if fixture:return prior.label_rows(dict(roots=settings["roots"]),True)
    return {key:dict(row["source_row"],allocation=dict(group=row["group"],component=row["component"],cell=row["cell"],source_cohort=row["source_cohort"],future_optimizer_reserved=True,source_wrapper_sha256=row["source_wrapper_sha256"])) for key,row in read(REGISTRY)["rows"].items()}


def references(rows,fixture):
    if fixture:return {stage+"_references":{key:endpoint("midgame-label-pilot-v1",stage,key) for key in rows} for stage in ("h8","h10")}
    rule=read(RULE)["rows"];return {stage+"_references":{key:rule[key][stage] for key in rows} for stage in ("h8","h10")}


AUDIT_METHODS=dict(preflight=["test_01_prior4255_panel_roles_and_predeclared_rules","test_02_trace_real_pre_post_simulation_kills_and_semantic_resume","test_03_teacher_real_pre_post_attempt_kills_and_H12_gate"],trace=["test_04_all8_production_traces_literal_selection_backup_and_parity"])


def gate():
    from . import allocation_trace
    require("torch" not in sys.modules,"Weighting gate forbids Torch");witness=allocation_trace.primary_witness();expected=dict(schema=allocation_trace.SCHEMA+"-primary-witness",passed=True,fixture=False,complete_cells=8,completed_root_simulations=3840,all_sealed_reports_match=True,torch_imported_by_gate=False);require(equal({key:witness[key] for key in expected},expected),"Production trace witness incomplete")
    groups={}
    for name,methods in AUDIT_METHODS.items():
        path=ROOT/"build/allocation-audit"/(name+".json");body=read(path);expected=dict(schema="allocation-independent-group-v1",group=name,passed=True,tests_run=len(methods),errors=0,failures=0,skips=0,covered_methods=methods,covered_checks=["__main__.AllocationIndependentAudit."+m for m in methods]);require(equal({key:body[key] for key in expected},expected),"Independent weighting stage gate differs")
        inputs=body["inputs"];required={"tests/allocation_independent_audit.py","tools/allocation-audit.py","build/allocation-audit-source-freeze.json","build/allocation-baseline.json","build/allocation-weighting-registry.json","build/allocation-weighting-rule.json","build/allocation-weighting-reservations.json","build/allocation-weighting-plan.json","build/allocation-weighting-implementation.json","congklak/allocation_weighting.py","congklak/allocation_trace.py","build/allocation-trace-implementation.json"}
        if name=="trace":required.update({"data/allocation-trace-v1/work.sqlite3","data/allocation-trace-v1/manifest.json","build/allocation-trace-result.json"})
        require(required<=set(inputs) and all(inputs.get(p)==digest for p,digest in common.baseline()["inputs"].items()),"Independent weighting proof omits source/prior identity");common.current(inputs);groups[name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
    return dict(trace=witness,independent=groups)


def identity_for(settings,fixture):return dict(schema=SCHEMA,settings=settings,fixture=fixture,rules_digest=config.RULES_DIGEST,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/validate-allocation-weighting.py"),baseline_sha256=sha(BASELINE),artifact_sha256=common.pins([INVENTORY,REGISTRY,RULE,RESERVATIONS,PLAN,IMPLEMENTATION]),model_files=common.pins([common.CONTROL,common.CANDIDATE]),native_build_sha256=Native().build_id,production_gate=None if fixture else gate(),panel_sha256=config.digest(expected_rows(settings,fixture)),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def validate_tables(rows,stores,settings,fixture,closed=False):
    native=Native();expected=references(rows,fixture)
    for table,items in expected.items():
        require(equal(stores[table],items),"Original H8/H10 endpoint references changed")
        for key,item in items.items():history.old.validate_label(item,rows[key],int(table[1:table.index('_')]),native)
    attempts,labels=stores["h12_attempts"],stores["h12_labels"];require(list(attempts)==list(rows)[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(rows)),"H12 attempt prefix/coverage differs")
    for key,attempt in attempts.items():
        history.validate_attempt(rows[key],attempt,labels.get(key),"h12",settings,native)
        for report in attempt["history"]:
            depth=report["choice_depth"]
            if depth in (8,10):require(equal(history.q_signature(history.item_for(report,rows[key])),history.q_signature(stores["h"+str(depth)+"_references"][key])),"Requested H8/H10 versus H12 history Q/best disagrees")


def manifest_for(identity,rows,stores):
    attempts,labels=stores["h12_attempts"],stores["h12_labels"];nodes=sum(a["work"]["nodes"] for a in attempts.values());require(nodes<=CONTRACT["maximum_new_nodes"],"H12 committed work exceeds ceiling")
    return dict(schema=SCHEMA,status="complete",identity=identity,requested_roots=len(rows),summary=dict(attempts=len(attempts),complete=len(labels),rejected=sum(a["status"]=="budget_rejected" for a in attempts.values()),missing_requested_reference=sorted(set(rows)-set(labels))),total_new_nodes=nodes,auxiliary_completed_reports=sum(len(a["history"]) for a in attempts.values()),table_sha256=dict(roots=config.digest(rows),**{name:config.digest(body) for name,body in stores.items()}),future_optimizer_reserved=not identity["fixture"],weights_applied=False,training_performed=False,promotion=False,torch_imported=False)


def metadata(db,key):
    row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Missing weighting metadata: "+key);body=json.loads(row[0]);require(config.digest(body)==row[1],"Corrupt weighting metadata: "+key);return body


def validate_schema(db,closed=False):
    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};keys={row[0] for row in db.execute("SELECT key FROM meta")}
    require(tables=={"meta","roots",*TABLES} and db.execute("PRAGMA user_version").fetchone()[0]==6 and db.execute("PRAGMA quick_check").fetchone()[0]=="ok","Unknown/corrupt weighting journal schema")
    require(keys=={"identity","complete"} if closed else keys<={"identity","complete"},"Unknown weighting journal metadata")


def read_core(db,path):
    validate_schema(db,True);identity=metadata(db,"identity");fixture=identity["fixture"];settings=settings_for(fixture,identity["settings"]["roots"] if fixture else None);admit()
    require(Path(path).resolve().is_relative_to(ROOT/"build") if fixture else Path(path).resolve()==DEFAULT.resolve(),"Weighting journal path differs");require(equal(identity,identity_for(settings,fixture)),"Exact weighting identity differs")
    rows=history.table(db,"roots");require(equal(rows,expected_rows(settings,fixture)),"Fixed reserved roots differ");stores={name:history.table(db,name) for name in TABLES};validate_tables(rows,stores,settings,fixture,True);body=manifest_for(identity,rows,stores);require(equal(metadata(db,"complete"),body),"Weighting completed marker differs");return rows,stores,body


def run(path=DEFAULT,fixture=False,roots=None,on_event=None,cancel=None):
    require("torch" not in sys.modules,"Weighting producer forbids Torch");settings=settings_for(fixture,roots);path=Path(path).resolve();require(path.is_relative_to(ROOT/"build") if fixture else path==DEFAULT.resolve(),"Weighting journal path differs");admit();identity=identity_for(settings,fixture);work=phase6_store.Work(path,identity);event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        require(equal(work.get("identity"),identity),"Saved weighting identity typed fields differ");work.db.executescript("CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);"+"".join("CREATE TABLE IF NOT EXISTS "+t+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for t in TABLES));validate_schema(work.db);rows=history.table(work.db,"roots");stores={t:history.table(work.db,t) for t in TABLES};expected=expected_rows(settings,fixture)
        if not rows:
            require(not any(stores.values()),"Orphan weighting references/attempts")
            with work.db:
                for key,row in expected.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
                for t,values in references(expected,fixture).items():
                    for key,body in values.items():work.db.execute("INSERT INTO "+t+" VALUES(?,?,?)",(key,config.canonical(body),config.digest(body)))
            rows=expected;stores={t:history.table(work.db,t) for t in TABLES}
        require(equal(rows,expected),"Reserved root projection differs");validate_tables(rows,stores,settings,fixture);complete=work.get("complete")
        if complete is None:
            for key in sorted(set(rows)-set(stores["h12_attempts"])):
                if history.stopped(cancel):return dict(status="interrupted")
                item,attempt=history.execute(rows[key],"h12",settings,Native(),cancel)
                if attempt is None:return dict(status="interrupted")
                proposed={t:dict(v) for t,v in stores.items()};proposed["h12_attempts"][key]=attempt
                if item is not None:proposed["h12_labels"][key]=item
                validate_tables(rows,proposed,settings,fixture);values=dict(state_key=key,completed=len(stores["h12_attempts"]),target=len(rows),status=attempt["status"],requested_depth=settings["h12"]["depth"],completed_depth=attempt["completed_depth"]);event("allocation_weighting_h12_precommit",**values);work.admit(len(config.canonical(attempt))*3+2*1024**2)
                with work.db:
                    work.db.execute("INSERT INTO h12_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                    if item is not None:work.db.execute("INSERT INTO h12_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                stores=proposed;values["completed"]+=1;event("allocation_weighting_h12_committed",**values)
            validate_tables(rows,stores,settings,fixture,True);complete=manifest_for(identity,rows,stores)
            with work.db:work.put("complete",complete)
        _,_,body=read_core(work.db,path);publish(path/"manifest.json",body);event("allocation_weighting_complete",summary=body["summary"],total_new_nodes=body["total_new_nodes"]);return body
    finally:work.close()


def load_rows(path=DEFAULT):
    require("torch" not in sys.modules,"Weighting reader forbids Torch");path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");rows,stores,body=read_core(db,path);require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(body)+"\n","Weighting file/committed manifest differs");return rows,stores,body


def metric(values):
    complete=[r for r in values if r["H12_complete"]];total=sum((fraction(r["weight"]) for r in values),F());mass=sum((fraction(r["weight"]) for r in complete),F());n=len(complete)
    def average(field,weighted):
        if not n:return None
        result=sum((fraction(r[field])*(fraction(r["weight"]) if weighted else 1) for r in complete),F())/(mass if weighted else n);return rational(result)
    ess=mass*mass/sum((fraction(r["weight"])**2 for r in complete),F()) if n else None
    return dict(requested_roots=len(values),complete_roots=n,missing_roots=len(values)-n,requested_state_keys=sorted(r["state_key"] for r in values),complete_state_keys=sorted(r["state_key"] for r in complete),missing_state_keys=sorted(r["state_key"] for r in values if not r["H12_complete"]),original_weight_mass=rational(total),complete_weight_mass=rational(mass),missing_original_weight_mass=rational(total-mass),missing_original_weight_fraction=rational((total-mass)/total) if total else None,ESS=rational(ess) if ess is not None else None,relative_ESS=rational(ess/n) if n else None,uniform=dict(root_drift=average("root_drift",False),action_Q_drift=average("action_Q_drift",False),H10_best_H12_regret_min=average("H10_best_H12_regret_min",False),H10_best_H12_regret_max=average("H10_best_H12_regret_max",False),H10_best_H12_regret_mean=average("H10_best_H12_regret_mean",False)),weighted=dict(root_drift=average("root_drift",True),action_Q_drift=average("action_Q_drift",True),H10_best_H12_regret_min=average("H10_best_H12_regret_min",True),H10_best_H12_regret_max=average("H10_best_H12_regret_max",True),H10_best_H12_regret_mean=average("H10_best_H12_regret_mean",True)),support="Identical requested-H12 complete intersection for both averages; missing remains null, not zero")


def descriptive_screen(assessment):
    n=assessment["complete_roots"];a,b=assessment["uniform"],assessment["weighted"]
    conditions=dict(all_assessment_complete=assessment["requested_roots"]==n==8,weighted_root_strictly_lower=bool(n and fraction(b["root_drift"])<fraction(a["root_drift"])),weighted_action_no_higher=bool(n and fraction(b["action_Q_drift"])<=fraction(a["action_Q_drift"])),ESS_at_least_4N_over5=bool(n and fraction(assessment["ESS"])>=F(4*n,5)))
    return dict(**conditions,passes=all(conditions.values()),exact_rational_comparisons=True,engineering_acceptance_condition=False,calibrated_uncertainty_claim=False,training_application_authorized=False)


def mass_distribution(values):
    total=sum((fraction(r["weight"]) for r in values),F());complete=sum((fraction(r["weight"]) for r in values if r["H12_complete"]),F());out={}
    for field in ("group","stratum","actor","cell"):
        out[field]={}
        for key in sorted({str(r[field]) for r in values}):
            subset=[r for r in values if str(r[field])==key];mass=sum((fraction(r["weight"]) for r in subset),F());cmass=sum((fraction(r["weight"]) for r in subset if r["H12_complete"]),F());out[field][key]=dict(requested_roots=len(subset),complete_roots=sum(r["H12_complete"] for r in subset),original_fraction=rational(mass/total),complete_mass_fraction_of_original=rational(cmass/total),complete_intersection_fraction=rational(cmass/complete) if complete else None)
    return out


def build():
    registry,rule=admit();rows,stores,manifest=load_rows();selected=registry["rows"];perroot={}
    for key,row in selected.items():
        refs=rule["rows"][key];attempt=stores["h12_attempts"][key];item=stores["h12_labels"].get(key);weight=refs["rule"]["weight"];record=dict(state_key=key,family=row["family"],component=row["component"],stratum=row["stratum"],actor=row["actor"],cell=row["cell"],group=row["group"],source_cohort=row["source_cohort"],future_optimizer_reserved=True,state=row["state"],legal_actions=[p for p in range(7) if row["legal_mask"]&(1<<p)],weight=weight,H12_complete=item is not None,H12_status=attempt["status"],requested_depth=12,completed_depth=attempt["completed_depth"],source_wrapper_sha256=row["source_wrapper_sha256"],source_root_sha256=refs["source_root_sha256"],allocation_wrapper_sha256=config.digest(rows[key]),H8_item_sha256=refs["h8_item_sha256"],H10_item_sha256=refs["h10_item_sha256"],H12_item_sha256=config.digest(item) if item is not None else None,root_drift=None,action_Q_drift=None,H10_best_H12_regret_min=None,H10_best_H12_regret_max=None,H10_best_H12_regret_mean=None,H10_best_H12_regrets=None,H8_player0=refs["h8"]["label"]["target_player0"],H10_player0=refs["h10"]["label"]["target_player0"],H12_player0=None,target_kind="approximate_requested_finite_H_v1_choice_horizon")
        if item is not None:
            h10,h12=refs["h10"]["label"],item["label"];q10={v["pit"]:v["value_player0"] for v in h10["action_scores"]};q12={v["pit"]:v["value_player0"] for v in h12["action_scores"]};sign=1 if row["actor"]==0 else -1;best=max(sign*v for v in q12.values());regrets={str(p):best-sign*q12[p] for p in h10["best_moves"]}
            record.update(root_drift=rational(abs(h12["target_player0"]-h10["target_player0"])),action_Q_drift=rational(F(sum(abs(q12[p]-q10[p]) for p in q10),len(q10))),H10_best_H12_regret_min=rational(min(regrets.values())),H10_best_H12_regret_max=rational(max(regrets.values())),H10_best_H12_regret_mean=rational(F(sum(regrets.values()),len(regrets))),H10_best_H12_regrets=regrets,H12_player0=h12["target_player0"],H10_Q_player0=q10,H12_Q_player0=q12,H10_best_moves=h10["best_moves"],H12_best_moves=h12["best_moves"])
        perroot[key]=record
    values=list(perroot.values());metrics={group:metric([r for r in values if r["group"]==group]) for group in ("development","assessment")};metrics["all"]=metric(values)
    metrics["by_group_stratum"]={g:{s:metric([r for r in values if r["group"]==g and r["stratum"]==s]) for s in STRATA} for g in ("development","assessment")}
    metrics["by_group_actor"]={g:{str(a):metric([r for r in values if r["group"]==g and r["actor"]==a]) for a in (0,1)} for g in ("development","assessment")}
    metrics["by_group_cell"]={g:{c:metric([r for r in values if r["group"]==g and r["cell"]==c]) for c in CELLS} for g in ("development","assessment")}
    metrics["mass_distribution"]={g:mass_distribution([r for r in values if g=="all" or r["group"]==g]) for g in ("development","assessment","all")}
    inputs=common.pins([BASELINE,INVENTORY,REGISTRY,RULE,RESERVATIONS,PLAN,IMPLEMENTATION,DEFAULT/"work.sqlite3",DEFAULT/"manifest.json",common.CONTROL,common.CANDIDATE,*SOURCES,*DEPENDENCIES])
    return dict(schema=SCHEMA+"-assessment",passed=True,status="complete",requested_roots=len(values),complete_roots=len(stores["h12_labels"]),capped_roots=manifest["summary"]["rejected"],manifest_sha256=sha(DEFAULT/"manifest.json"),total_new_nodes=manifest["total_new_nodes"],auxiliary_completed_reports=manifest["auxiliary_completed_reports"],per_root=perroot,metrics=metrics,screen=descriptive_screen(metrics["assessment"]),weights_applied=False,training_performed=False,promotion=False,previous_allocation="hold",future_optimizer_reserved=True,H12_is_game_truth=False,rule_fitted=False,calibrated_confidence_claim=False,population_claim=False,inputs=inputs,limits=["Selected balanced16-root development/assessment design, not a natural position-distribution estimate.","Lower weighted drift describes concentration toward finite-horizon stability, not uncertainty calibration, predictive improvement or strength.","No current or future optimizer weights are authorized by the screen; source training roles remain historical while selected component support is future-reserved.","Complete-intersection weights renormalize only for reported metrics; missing original mass and all requested roots remain visible.","Exact-root optimizer membership is distinct from human reference exposure and from unobserved internal search states or causal ancestral effects."])


def assess():return publish(OUTPUT,build())


def verify():
    actual=read(OUTPUT);require(equal(actual,build()),"Weighting assessment metric/source identities do not rederive");return actual
