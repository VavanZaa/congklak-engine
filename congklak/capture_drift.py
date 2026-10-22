"""Exact retrospective capture attribution with recoverable offline case commits."""
from contextlib import closing
from fractions import Fraction as F
import json
from pathlib import Path
import sqlite3
import sys
from . import config,phase6_store,stratum_metrics as old_metrics,capture_common as common
from . import capture_forensics as forensics

ROOT=common.ROOT
SCHEMA="capture-existing-finite-reference-drift-investigation-v1"
DEFAULT=ROOT/"data/capture-drift-v1"
PANEL=ROOT/"build/capture-drift-panel.json"
PLAN=ROOT/"build/capture-drift-plan.json"
IMPLEMENTATION=ROOT/"build/capture-drift-implementation.json"
SOURCE_FREEZE=ROOT/"build/capture-drift-source-freeze.json"
OUTPUT=ROOT/"build/capture-drift-result.json"
BASELINE=common.BASELINE
STAGES=("h8","h10","h12")
PAIRS=(("h8","h10"),("h10","h12"),("h8","h12"))
TABLES=tuple(s+t for s in STAGES for t in ("_attempts","_labels"))
SOURCE_DEFAULT=ROOT/"data/stratum-weighting-v1"
SOURCE_FILES=("data/stratum-weighting-v1/work.sqlite3","data/stratum-weighting-v1/manifest.json","data/stratum-weighting-v1/references-prefix.sqlite3","data/stratum-weighting-v1/references-manifest.json","build/stratum-registry.json","build/stratum-rule.json","build/stratum-result.json","build/stratum-reservations.json","build/stratum-plan.json","build/stratum-implementation.json")
PRIOR_VALIDATION_FILES=("congklak/stratum_metrics.py","congklak/config.py","congklak/phase6_store.py","congklak/pilot.py","congklak/native.py","congklak/oracle.py","congklak/reports.py","congklak/divergence_study_common.py","congklak/deeper_teacher_v2.py","tests/reference.py","native/search.cpp")
SOURCES=("congklak/capture_drift.py","tools/investigate-capture-drift.py","tests/capture_drift_author.py","docs/CAPTURE_DRIFT.md","docs/CAPTURE_SCOPE.md","congklak/capture_common.py")
DEPENDENCIES=("congklak/capture_forensics.py","tests/capture_forensics_author.py","docs/CAPTURE_FORENSICS.md","build/capture-forensics-source-freeze.json",*PRIOR_VALIDATION_FILES)
CONTRACT=dict(cases=4,global_context_roots=16,case_order="Sorted rawstate keys of ALL four sealed25-49/capture roots",pairs=[list(p) for p in PAIRS],probabilities="Unchanged sealedU=1/16,W=(1/8)*r/sum_cell(r)",exact_arithmetic=True,
    attribution="Allactionsum((W-U)*abs(Qnew-Qold)/legalcount), preserving16rootdenominator",covariance="Cellconditionaldelta=(r1-r2)*(D1-D2)/(2*(r1+r2)); global=cellmass1/8*conditional",
    decomposition="D=meanabsdelta,C=absmeandelta,S=D-C>=0,E=meanabs(delta-meandelta);D=C+S,not generallyC+E",ranking="All unordered legalaction pairs; strict reversal vs tiechange separated",top_actions="H10top2 including all ties at second action score",best_sets="EntireH8/H10/H12 sets preserved; best/nonbest contribution denominators remainalllegal",
    new_teacher_searches=0,new_NN_forwards=0,new_games=0,training_performed=False,weights_applied=False,promotion=False,new_reservations=False,retrospective=True)
require,read,sha,publish=common.require,common.read,common.sha,common.publish
equal=common.same
rational,fraction=old_metrics.rational,old_metrics.fraction


def ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)


def table(db,name,key="state"):
    result={}
    for identity,text,digest in db.execute("SELECT "+key+",body,hash FROM "+name+" ORDER BY "+key):
        body=json.loads(text);require(config.digest(body)==digest,"Corrupt source/journal row: "+name);result[identity]=body
    return result


def metadata(db,key):
    row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Missing metadata: "+key);body=json.loads(row[0]);require(config.digest(body)==row[1],"Corrupt metadata: "+key);return body


def signature(report):return dict(value=report["depth_result"]["value_player0"],best=report["best_moves"],q=[(c["pit"],c["depth_result"]["value_player0"]) for c in report["candidates"]])


def source():
    """Read sealed tables directly; no old admission graph or runtime activation."""
    require("torch" not in sys.modules,"Capture analysis forbids Torch");baseline=common.baseline()
    relevant=set(SOURCE_FILES+PRIOR_VALIDATION_FILES)|{Path(p).relative_to(ROOT).as_posix() for p in (common.CONTROL,common.CANDIDATE)}
    require(relevant<=set(baseline["inputs"]),"Required existing evidence/source is absent from priorbaseline");common.current({p:baseline["inputs"][p] for p in relevant})
    registry=read(ROOT/"build/stratum-registry.json");rule=read(ROOT/"build/stratum-rule.json");previous=read(ROOT/"build/stratum-result.json");manifest=read(SOURCE_DEFAULT/"manifest.json")
    with closing(ro(SOURCE_DEFAULT/"work.sqlite3")) as db:
        db.execute("BEGIN");require(db.execute("PRAGMA quick_check").fetchone()[0]=="ok" and db.execute("PRAGMA user_version").fetchone()[0]==6,"Source journal invalid")
        rows=table(db,"roots");stores={t:table(db,t) for t in TABLES};identity=metadata(db,"identity");complete=metadata(db,"complete")
    require(identity["fixture"] is False and equal(complete,manifest) and equal(manifest["identity"],identity),"Source final identity/manifest differs")
    require(manifest["status"]=="complete" and len(rows)==16 and set(rows)==set(registry["rows"]),"Incomplete source support")
    require(equal(manifest["table_sha256"],dict(roots=config.digest(rows),**{t:config.digest(v) for t,v in stores.items()})),"Source committed tables differ")
    require(rule["H12_used"] is False and rule["fixture"] is False and rule["references_prefix_sha256"]==sha(SOURCE_DEFAULT/"references-prefix.sqlite3") and rule["references_manifest_sha256"]==sha(SOURCE_DEFAULT/"references-manifest.json"),"Frozen reference prefix linkage differs")
    refs=read(SOURCE_DEFAULT/"references-manifest.json")
    with closing(ro(SOURCE_DEFAULT/"references-prefix.sqlite3")) as db:
        require(equal(metadata(db,"identity"),identity) and equal(metadata(db,"references_complete"),refs) and equal(table(db,"roots"),rows),"Prefix identity/roots changed")
        for t in TABLES:require(equal(table(db,t),{} if t.startswith("h12") else stores[t]),"Prefix differs from live H8/H10 evidence")
    require(rule["registry_sha256"]==sha(ROOT/"build/stratum-registry.json") and previous["realized_rule_sha256"]==sha(ROOT/"build/stratum-rule.json") and previous["manifest_sha256"]==sha(SOURCE_DEFAULT/"manifest.json"),"Sealed result/rule links differ")
    old_metrics.validate_projection(rows,rule["projection"])
    for key,row in rows.items():
        require(key==common.rawkey(row["state"])==row["state_key"] and row["source_root_sha256"]==config.digest(row["root"]) and equal(row["state"],registry["rows"][key]["state"]),"Raw source wrapper/root mismatch")
        overlap={}
        for stage,depth in zip(STAGES,(8,10,12)):
            require(set(stores[stage+"_labels"])==set(stores[stage+"_attempts"])==set(rows),"Missing source endpoint or attempt")
            item=stores[stage+"_labels"][key];attempt=stores[stage+"_attempts"][key];label,_,_=old_metrics._label(item,row,depth)
            require(item["root_sha256"]==row["source_root_sha256"] and attempt["source_root_sha256"]==row["source_root_sha256"] and attempt["status"]=="complete" and type(attempt["requested_depth"]) is int and attempt["requested_depth"]==attempt["completed_depth"]==depth,"Endpoint root/depth/status differs")
            require([r["choice_depth"] for r in attempt["history"]]==list(range(1,depth+1)) and equal(item["report"],attempt["history"][-1]),"Source completed history differs")
            for report in attempt["history"]:
                d=report["choice_depth"];sig=signature(report);require(report["search_options"]["tt_mb"]==16 and report["search_options"]["use_pvs"] is True,"Source TT/PVS protocol differs")
                require(d not in overlap or equal(overlap[d],sig),"Cross-stage completed Q/best differs");overlap[d]=sig
            if stage in ("h8","h10"):require(equal(rule["projection"]["rows"][key]["endpoints"][stage],item),"Rule input is not the exact source endpoint")
    assessment=old_metrics.evaluate(rows,rule["projection"],stores["h12_labels"])
    require(equal(assessment,previous["assessment"]) and previous["weights_applied"] is previous["training_performed"] is previous["promotion"] is False,"Source finite-reference result changed")
    capture=sorted(k for k,r in rows.items() if r["stratum"]=="25-49/capture");require(len(capture)==4 and sum(rows[k]["actor"]==0 for k in capture)==2,"Capture support differs")
    return dict(rows=rows,stores=stores,rule=rule,previous=previous,manifest=manifest,capture_keys=capture,inputs=common.pins(SOURCE_FILES))


def qvalues(item):return {c["pit"]:c["value_player0"] for c in item["label"]["action_scores"]}


def sign(value):return (value>0)-(value<0)


def translation(deltas):
    require(bool(deltas) and all(type(v) is int for v in deltas),"Signed Q changes must be integers")
    mean=F(sum(deltas),len(deltas));d=F(sum(abs(v) for v in deltas),len(deltas));c=abs(mean);s=d-c;e=sum((abs(v-mean) for v in deltas),F())/len(deltas);require(s>=0 and d==c+s,"Cancellation decomposition failed")
    return dict(mean_delta_player0=rational(mean),D_mean_absolute=rational(d),C_absolute_common_mean=rational(c),S_signed_cancellation_remainder=rational(s),E_centered_dispersion=rational(e),D_equals_C_plus_S=True,D_equals_C_plus_E=d==c+e,centered_equals_additive_component_claim=False)


def action_analysis(row,endpoints,probability):
    actor_sign=1 if row["actor"]==0 else -1;qs={s:qvalues(endpoints[s]) for s in STAGES};legal=sorted(qs["h10"]);best={s:endpoints[s]["label"]["best_moves"] for s in STAGES};union=set().union(*(set(v) for v in best.values()));n=len(legal)
    require(n>0 and all(set(q)==set(legal) for q in qs.values()),"Full legal action support must match")
    u,w=fraction(probability["uniform_probability"]),fraction(probability["weighted_probability"]);top_threshold=sorted((actor_sign*v for v in qs["h10"].values()),reverse=True)[min(1,n-1)];top={a for a in legal if actor_sign*qs["h10"][a]>=top_threshold}
    actions={a:dict(pit=a,human_pit=a+1,Q_player0={s:qs[s][a] for s in STAGES},Q_actor={s:actor_sign*qs[s][a] for s in STAGES},best_membership={s:a in best[s] for s in STAGES},H10_top2_including_boundary_ties=a in top,ever_requested_best=a in union,gap_to_best_actor={s:actor_sign*(endpoints[s]["label"]["target_player0"]-qs[s][a]) for s in STAGES},pairs={}) for a in legal};pairs={}
    for before,after in PAIRS:
        key=before+"_to_"+after;deltas={a:qs[after][a]-qs[before][a] for a in legal};decomp=translation(list(deltas.values()));decomp["mean_delta_actor"]=rational(actor_sign*fraction(decomp["mean_delta_player0"]));ranking=[]
        for i,a in enumerate(legal):
            for b in legal[i+1:]:
                old=actor_sign*(qs[before][a]-qs[before][b]);new=actor_sign*(qs[after][a]-qs[after][b]);ranking.append(dict(actions=[a,b],old_actor_difference=old,new_actor_difference=new,strict_reversal=old*new<0,tie_changed=(old==0)!=(new==0),order_changed=sign(old)!=sign(new)))
        groups={}
        for group,subset in (("H10_best",set(best["h10"])),("H10_nonbest",set(legal)-set(best["h10"])),("H10_top2",top),("ever_requested_best",union),("never_requested_best",set(legal)-union)):
            total=sum(abs(deltas[a]) for a in subset);groups[group]=dict(actions=sorted(subset),count=len(subset),absolute_sum=total,within_subset_mean_absolute=rational(F(total,len(subset))) if subset else None,root_mean_absolute_contribution=rational(F(total,n)),uniform_global_contribution=rational(u*F(total,n)),weighted_global_contribution=rational(w*F(total,n)),global_weighted_minus_uniform=rational((w-u)*F(total,n)))
        require(fraction(groups["H10_best"]["root_mean_absolute_contribution"])+fraction(groups["H10_nonbest"]["root_mean_absolute_contribution"])==fraction(decomp["D_mean_absolute"]),"Best/nonbest action partition failed")
        for a in legal:
            delta=deltas[a];actions[a]["pairs"][key]=dict(delta_player0=delta,delta_actor=actor_sign*delta,absolute_drift=abs(delta),centered_delta_player0=rational(delta-fraction(decomp["mean_delta_player0"])),root_mean_absolute_contribution=rational(F(abs(delta),n)),uniform_global_contribution=rational(u*F(abs(delta),n)),weighted_global_contribution=rational(w*F(abs(delta),n)),global_weighted_minus_uniform=rational((w-u)*F(abs(delta),n)),H10_best_anchor_gap_changes=[dict(anchor=b,anchor_still_best=b in best[after],change_actor=actor_sign*(deltas[b]-delta)) for b in best["h10"]] if before=="h10" else None)
        pairs[key]=dict(**decomp,root_value_delta_player0=endpoints[after]["label"]["target_player0"]-endpoints[before]["label"]["target_player0"],best_sets_before=best[before],best_sets_after=best[after],best_sets_same=best[before]==best[after],strict_ranking_reversals=sum(x["strict_reversal"] for x in ranking),tie_changes=sum(x["tie_changed"] for x in ranking),pairwise_ranks=ranking,action_groups=groups)
    return dict(actions=actions,pairs=pairs,H10_best_moves=best["h10"],H12_best_moves=best["h12"],best_H10_H12_same=best["h10"]==best["h12"],H10_best_H12_regrets={b:actor_sign*(endpoints["h12"]["label"]["target_player0"]-qs["h12"][b]) for b in best["h10"]},score_units="Seeds; player0 fixedorientation and actualcontroller signs bothretained")


def covariance_identity(raw,drifts,cell_mass=F(1,8)):
    require(len(raw)==len(drifts)==2 and all(type(v) is F for v in raw+drifts) and all(v>0 for v in raw),"Two exact positive reliability scores required")
    a,b=raw;d,e=drifts;uniform=(d+e)/2;weighted=(a*d+b*e)/(a+b);cov=(a-b)*(d-e)/4;difference=(a-b)*(d-e)/(2*(a+b));require(weighted-uniform==difference==cov/((a+b)/2),"Two-root covariance identity failed")
    return dict(raw_reliabilities=[rational(v) for v in raw],root_drifts=[rational(v) for v in drifts],conditional_uniform=rational(uniform),conditional_weighted=rational(weighted),conditional_delta=rational(difference),uniform_covariance=rational(cov),raw_mean=rational((a+b)/2),cell_mass=rational(cell_mass),global_delta=rational(cell_mass*difference),identity_verified=True,causal_claim=False)


def context(data):
    rows=data["rows"];projection=data["rule"]["projection"];rootrows={};cells={};pairtotals={}
    for key,row in rows.items():
        prob=projection["rows"][key];u,w=fraction(prob["uniform_probability"]),fraction(prob["weighted_probability"]);pairs={}
        for a,b in PAIRS:
            qa,qb=qvalues(data["stores"][a+"_labels"][key]),qvalues(data["stores"][b+"_labels"][key]);d=F(sum(abs(qb[p]-qa[p]) for p in qa),len(qa));pairs[a+"_to_"+b]=dict(drift=rational(d),uniform_contribution=rational(u*d),weighted_contribution=rational(w*d),global_delta=rational((w-u)*d))
        rootrows[key]=dict(state_key=key,cell=row["cell"],stratum=row["stratum"],actor=row["actor"],family=row["family"],legal_actions=sorted(qvalues(data["stores"]["h10_labels"][key])),source_root_sha256=row["source_root_sha256"],source_wrapper_sha256=config.digest(row),raw_reliability=prob["raw_rule"]["raw_weight"],uniform_probability=prob["uniform_probability"],weighted_probability=prob["weighted_probability"],pairs=pairs)
    for cell in old_metrics.CELLS:
        keys=sorted(k for k,r in rows.items() if r["cell"]==cell);require(len(keys)==2,"Source cell no longer contains exactlytwo roots");raw=[fraction(rootrows[k]["raw_reliability"]) for k in keys]
        pairs={p:covariance_identity(raw,[fraction(rootrows[k]["pairs"][p]["drift"]) for k in keys]) for p in (a+"_to_"+b for a,b in PAIRS)}
        for p,value in pairs.items():require(fraction(value["global_delta"])==sum((fraction(rootrows[k]["pairs"][p]["global_delta"]) for k in keys),F()),"Cell and root contribution disagree")
        cells[cell]=dict(state_keys=keys,uniform_mass=rational(sum((fraction(rootrows[k]["uniform_probability"]) for k in keys),F())),weighted_mass=rational(sum((fraction(rootrows[k]["weighted_probability"]) for k in keys),F())),pairs=pairs)
    for p in (a+"_to_"+b for a,b in PAIRS):
        def total(keys,field):return rational(sum((fraction(rootrows[k]["pairs"][p][field]) for k in keys),F()))
        capture=data["capture_keys"];other=sorted(set(rows)-set(capture));allkeys=sorted(rows)
        pairtotals[p]=dict(uniform_global_drift=total(allkeys,"uniform_contribution"),weighted_global_drift=total(allkeys,"weighted_contribution"),global_delta=total(allkeys,"global_delta"),capture_global_delta=total(capture,"global_delta"),non_capture_global_delta=total(other,"global_delta"),by_stratum={s:total([k for k,r in rows.items() if r["stratum"]==s],"global_delta") for s in old_metrics.STRATA})
        require(fraction(pairtotals[p]["global_delta"])==fraction(pairtotals[p]["capture_global_delta"])+fraction(pairtotals[p]["non_capture_global_delta"]),"Global capture/noncapture partition failed")
    expected=data["previous"]["metrics"]["overall"]
    for name in ("uniform","weighted"):require(equal(pairtotals["h10_to_h12"][name+"_global_drift"],expected[name]["action_Q_drift"]),"All16 context does not reproduce sealed drift")
    return dict(roots=rootrows,cells=cells,pairs=pairtotals,global_roots=16,capture_roots=4,uniform_global_denominator=16,root_action_denominator="Each root averagesALLlegalactions before fixed rootprobability",probabilities_changed=False)


def panel_body(data):return dict(schema=SCHEMA+"-panel",all_state_keys=sorted(data["rows"]),capture_state_keys=data["capture_keys"],source_rows_sha256=config.digest(data["rows"]),source_endpoints_sha256={s:config.digest(data["stores"][s+"_labels"]) for s in STAGES},source_histories_sha256={s:config.digest(data["stores"][s+"_attempts"]) for s in STAGES},source_rule_sha256=sha(ROOT/"build/stratum-rule.json"),inputs=data["inputs"],retrospective=True,new_reservations=False)


def prepare():return publish(PANEL,panel_body(source()))


def plan_body():return dict(schema=SCHEMA,contract=CONTRACT,panel_sha256=sha(PANEL),baseline_sha256=sha(BASELINE),source_inputs=common.pins(SOURCE_FILES),models=common.pins([common.CONTROL,common.CANDIDATE]),rules_digest=config.RULES_DIGEST)


def plan():prepare();return publish(PLAN,plan_body())


def implementation_body():return dict(schema=SCHEMA+"-implementation",plan_sha256=sha(PLAN),sources=common.pins(SOURCES+DEPENDENCIES))


def freeze():
    plan();body=implementation_body();publish(IMPLEMENTATION,body);return publish(SOURCE_FREEZE,body)


def admit():
    data=source();require(equal(read(PANEL),panel_body(data)) and equal(read(PLAN),plan_body()) and equal(read(IMPLEMENTATION),implementation_body()) and equal(read(SOURCE_FREEZE),implementation_body()),"Frozen capture inputs/source/protocol differ");return data


AUDIT_METHODS=["test_01_prior4923_exact_support_sources_reservations_and_models","test_02_exact_attribution_signed_ranks_ties_and_cancellation","test_03_real_case_pre_post_commit_kills_and_exact_resume"]


def gate():
    path=ROOT/"build/capture-audit/preflight.json";body=read(path);expected=dict(schema="capture-independent-group-v1",group="preflight",passed=True,tests_run=3,errors=0,failures=0,skips=0,covered_methods=AUDIT_METHODS,covered_checks=["__main__.CaptureIndependentAudit."+m for m in AUDIT_METHODS])
    require(equal({k:body[k] for k in expected},expected),"Independent capture preflight incomplete")
    needed={"build/capture-baseline.json","docs/CAPTURE_SCOPE.md","congklak/capture_common.py","tests/capture_independent_audit.py","tools/capture-audit.py","build/capture-audit-source-freeze.json","congklak/capture_drift.py","congklak/capture_forensics.py","build/capture-drift-plan.json","build/capture-drift-implementation.json","build/capture-drift-source-freeze.json","build/capture-drift-panel.json","build/capture-drift-author.json","build/capture-forensics-author.json"}
    inputs=body["inputs"];require(needed<=set(inputs) and all(inputs.get(k)==v for k,v in common.baseline()["inputs"].items()),"Capture preflight omits currentsource/oldinputs");common.current(inputs)
    owners={}
    for name,test,module in (("capture-drift-author.json","tests/capture_drift_author.py","congklak/capture_drift.py"),("capture-forensics-author.json","tests/capture_forensics_author.py","congklak/capture_forensics.py")):
        owner=read(ROOT/"build"/name);require(owner["passed"] is True and type(owner["tests_run"]) is int and owner["tests_run"]==4 and all(type(owner[k]) is int and owner[k]==0 for k in ("errors","failures","skips")),"Capture owner proof incomplete")
        require(owner["inputs"].get(test)==sha(ROOT/test) and owner["inputs"].get(module)==sha(ROOT/module),"Owner executed source missing");common.current(owner["inputs"]);owners[name]=sha(ROOT/"build"/name)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),owners=owners)


def settings_for(fixture=False,cases=None):
    require(type(fixture) is bool,"Fixture flag must be boolean")
    if fixture:n=2 if cases is None else config.integer(cases,"fixture_cases",1,2)
    else:require(cases is None,"Production requiresALLfour fixedcases");n=4
    return dict(fixture=fixture,cases=n,global_context_roots=16,fixture_panel="Firsttwoalreadyconsumedcapturecases" if fixture else None)


def identity_for(data,settings):return dict(schema=SCHEMA,settings=settings,rules_digest=config.RULES_DIGEST,artifacts=common.pins([PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE]),baseline_sha256=sha(BASELINE),source_inputs=data["inputs"],model_files=common.pins([common.CONTROL,common.CANDIDATE]),accepted_native_build_metadata=data["manifest"]["identity"]["native_build_sha256"],production_gate=None if settings["fixture"] else gate(),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def compute_case(index,data):
    key=data["capture_keys"][index];row=data["rows"][key];endpoints={s:data["stores"][s+"_labels"][key] for s in STAGES};attempts={s:data["stores"][s+"_attempts"][key] for s in STAGES};prob=data["rule"]["projection"]["rows"][key]
    return dict(schema=SCHEMA+"-case",idx=index,state_key=key,state=row["state"],actor=row["actor"],cell=row["cell"],family=row["family"],source_root_sha256=row["source_root_sha256"],source_wrapper_sha256=config.digest(row),endpoint_item_sha256={s:config.digest(v) for s,v in endpoints.items()},source_attempt_sha256={s:config.digest(v) for s,v in attempts.items()},uniform_probability=prob["uniform_probability"],weighted_probability=prob["weighted_probability"],raw_rule=prob["raw_rule"],analysis=action_analysis(row,endpoints,prob),forensics=forensics.build_case(row,endpoints,attempts),finite_H12_not_game_truth=True,weights_changed=False)


def validate_cases(cases,data,settings):
    require(list(cases)==list(range(len(cases))) and len(cases)<=settings["cases"],"Case prefix/count differs")
    for index,body in cases.items():require(equal(body,compute_case(index,data)),"Case analysis/source/forensics do not rederive")


def manifest_for(identity,cases):return dict(schema=SCHEMA,status="complete",identity=identity,cases=len(cases),case_sha256=config.digest(cases),committed_case_records=len(cases),new_teacher_searches=0,new_NN_forwards=0,new_games=0,training_performed=False,weights_applied=False,promotion=False)


def read_core(db,path,complete=True):
    require({r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}=={"meta","cases"} and db.execute("PRAGMA user_version").fetchone()[0]==6 and db.execute("PRAGMA quick_check").fetchone()[0]=="ok","Unknown/corrupt capturejournal schema")
    keys={r[0] for r in db.execute("SELECT key FROM meta")};require("identity" in keys and keys<={"identity","complete"},"Unknown journal metadata")
    identity=metadata(db,"identity");settings=identity["settings"];require(equal(settings,settings_for(settings["fixture"],settings["cases"] if settings["fixture"] else None)),"Typed capture settings differ")
    path=Path(path).resolve();require(path.is_relative_to(ROOT/"build") if settings["fixture"] else path==DEFAULT.resolve(),"Capture journal path differs")
    data=admit();require(equal(identity,identity_for(data,settings)),"Capture journal identity differs");cases=table(db,"cases","idx");validate_cases(cases,data,settings);manifest=manifest_for(identity,cases)
    if "complete" in keys:require(len(cases)==settings["cases"] and equal(metadata(db,"complete"),manifest),"Premature/corrupt completed marker")
    else:require(not complete,"Missing complete marker")
    return cases,manifest,data


def stopped(cancel):return cancel.is_set() if hasattr(cancel,"is_set") else bool(cancel and cancel())


def run(path=DEFAULT,fixture=False,cases=None,on_event=None,cancel=None):
    require("torch" not in sys.modules,"Captureproducer forbidsTorch");settings=settings_for(fixture,cases);path=Path(path).resolve();require(path.is_relative_to(ROOT/"build") if fixture else path==DEFAULT.resolve(),"Capture journal path differs");data=admit();identity=identity_for(data,settings);work=phase6_store.Work(path,identity);event=lambda name,**v:on_event(dict(event=name,**v)) if on_event else None
    try:
        require(equal(work.get("identity"),identity),"Saved typed identity differs");work.db.execute("CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT)");saved,manifest,_=read_core(work.db,path,False)
        if work.get("complete") is None:
            for index in range(len(saved),settings["cases"]):
                if stopped(cancel):return dict(status="interrupted",cases=len(saved))
                body=compute_case(index,data);event("capture_drift_precommit",completed=len(saved),target=settings["cases"],state_key=body["state_key"],case_index=index);work.admit(len(config.canonical(body))*3+2*1024**2)
                with work.db:work.db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
                saved[index]=body;event("capture_drift_committed",completed=len(saved),target=settings["cases"],state_key=body["state_key"],case_index=index)
            validate_cases(saved,data,settings);manifest=manifest_for(identity,saved)
            with work.db:work.put("complete",manifest)
        saved,manifest,_=read_core(work.db,path);publish(path/"manifest.json",manifest);event("capture_drift_complete",completed=len(saved),target=settings["cases"]);return manifest
    finally:work.close()


def load_rows(path=DEFAULT):
    require("torch" not in sys.modules,"Capture reader forbidsTorch");path=Path(path).resolve()
    with closing(ro(path/"work.sqlite3")) as db:
        db.execute("BEGIN");cases,manifest,_=read_core(db,path);require(equal(read(path/"manifest.json"),manifest),"Published manifest differs from validmarker");return cases,manifest


def build(path=DEFAULT):
    cases,manifest=load_rows(path);data=admit();ctx=context(data);pair=ctx["pairs"]["h10_to_h12"]
    return dict(schema=SCHEMA+"-result",passed=True,status="complete",fixture=manifest["identity"]["settings"]["fixture"],cases=len(cases),summary=dict(global_Q_drift_delta=pair["global_delta"],capture_Q_drift_delta=pair["capture_global_delta"],non_capture_Q_drift_delta=pair["non_capture_global_delta"],all16_context=True,capture_best_sets_stable=all(v["analysis"]["best_H10_H12_same"] for v in cases.values())),context=ctx,case_results=cases,manifest_sha256=sha(Path(path)/"manifest.json"),weights_applied=False,training_performed=False,promotion=False,new_teacher_searches=0,new_NN_forwards=0,new_games=0,previous_allocation="hold",new_reservations=False,retrospective=True,
        inputs=common.pins([BASELINE,PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE,Path(path)/"work.sqlite3",Path(path)/"manifest.json",*SOURCE_FILES,*SOURCES,*DEPENDENCIES]))


def assess():return publish(OUTPUT,build())


def validate_report(report,path=DEFAULT):require(equal(report,build(path)),"Capture report metrics/source/evidence do not rederive");return report


def verify(path=DEFAULT):return validate_report(read(OUTPUT),path)
