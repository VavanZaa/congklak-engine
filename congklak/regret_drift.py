"""Exact retrospective Q-drift and regret attribution; no engine activation."""
from contextlib import closing
from fractions import Fraction as F
import ast
import json
from pathlib import Path
import sqlite3
import sys
from . import config,phase6_store,qstability_metrics as old_metrics,regret_common as common
from . import regret_forensics as forensics

ROOT=common.ROOT
SCHEMA="regret-existing-finite-reference-drift-investigation-v1"
DEFAULT=ROOT/"data/regret-drift-v1"
PANEL=ROOT/"build/regret-drift-panel.json"
PLAN=ROOT/"build/regret-drift-plan.json"
IMPLEMENTATION=ROOT/"build/regret-drift-implementation.json"
SOURCE_FREEZE=ROOT/"build/regret-drift-source-freeze.json"
OUTPUT=ROOT/"build/regret-drift-result.json"
BASELINE=common.BASELINE
STAGES=("h8","h10","h12")
CASE_IDS=(178,91,382,92,390)
CASE_KEYS=("0200000001050330020205040102011601","000100030500001800050c050007012300","000203000100060a0203050f0202022d00","0200000a0101030e030100030700062f01","040106000000081b080101010600012201")
PAIRS=(("h8","h10"),("h10","h12"),("h8","h12"))
TABLES=tuple(s+t for s in STAGES for t in ("_attempts","_labels"))
SOURCE_DEFAULT=ROOT/"data/qstability-v1"
SOURCE_FILES=("data/qstability-v1/work.sqlite3","data/qstability-v1/manifest.json","data/qstability-v1/references-prefix.sqlite3","data/qstability-v1/references-manifest.json","build/qstability-registry.json","build/qstability-rule.json","build/qstability-result.json","build/qstability-reservations.json","build/qstability-plan.json","build/qstability-implementation.json")
PRIOR_VALIDATION_FILES=("congklak/qstability_metrics.py","congklak/capture_forensics.py","congklak/config.py","congklak/phase6_store.py","congklak/pilot.py","congklak/native.py","congklak/oracle.py","congklak/reports.py","congklak/divergence_study_common.py","congklak/deeper_teacher_v2.py","tests/reference.py","native/search.cpp")
SOURCES=("congklak/regret_drift.py","tools/investigate-regret.py","tests/regret_drift_author.py","docs/REGRET_DRIFT.md","docs/REGRET_SCOPE.md","congklak/regret_common.py")
DEPENDENCIES=("congklak/regret_forensics.py","tests/regret_forensics_author.py","docs/REGRET_FORENSICS.md","build/regret-forensics-source-freeze.json",*PRIOR_VALIDATION_FILES)
CONTRACT=dict(cases=5,case_ids=list(CASE_IDS),case_keys=list(CASE_KEYS),global_context_roots=16,case_order="Exactly consumedbonus178 then capture91,382,92,390",pairs=[list(p) for p in PAIRS],probabilities="Unchanged sealedU=1/16,W=(1/8)*max-actionreliability/sum_cell(reliability)",exact_arithmetic=True,
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
    registry=read(ROOT/"build/qstability-registry.json");rule=read(ROOT/"build/qstability-rule.json");previous=read(ROOT/"build/qstability-result.json");manifest=read(SOURCE_DEFAULT/"manifest.json")
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
    require(rule["registry_sha256"]==sha(ROOT/"build/qstability-registry.json") and previous["realized_rule_sha256"]==sha(ROOT/"build/qstability-rule.json") and previous["manifest_sha256"]==sha(SOURCE_DEFAULT/"manifest.json"),"Sealed result/rule links differ")
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
    require(set(CASE_KEYS[1:])==set(capture) and all(rows[k]["family"]=="midgame-history-pilot/20261023/"+str(i) for k,i in zip(CASE_KEYS,CASE_IDS)),"Exact five diagnostic cases differ")
    require(rows[CASE_KEYS[0]]["stratum"]=="25-49/bonus" and rows[CASE_KEYS[0]]["actor"]==1,"Bonus178 orientation/role differs")
    return dict(rows=rows,stores=stores,rule=rule,previous=previous,manifest=manifest,capture_keys=capture,case_keys=list(CASE_KEYS),inputs=common.pins(SOURCE_FILES))


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
        regrets={a:actor_sign*(endpoints[after]["label"]["target_player0"]-qs[after][a]) for a in best[before]};require(all(v>=0 for v in regrets.values()),"Before-best finite-reference regret became negative")
        pairs[key]=dict(**decomp,root_value_delta_player0=endpoints[after]["label"]["target_player0"]-endpoints[before]["label"]["target_player0"],best_sets_before=best[before],best_sets_after=best[after],best_sets_same=best[before]==best[after],strict_ranking_reversals=sum(x["strict_reversal"] for x in ranking),tie_changes=sum(x["tie_changed"] for x in ranking),pairwise_ranks=ranking,action_groups=groups,before_best_regrets={str(a):v for a,v in regrets.items()},before_best_regret_mean=rational(F(sum(regrets.values()),len(regrets))),before_best_regret_min=rational(min(regrets.values())),before_best_regret_max=rational(max(regrets.values())),before_best_lexicographic_regret=rational(regrets[min(regrets)]))
    return dict(actions=actions,pairs=pairs,H10_best_moves=best["h10"],H12_best_moves=best["h12"],best_H10_H12_same=best["h10"]==best["h12"],H10_best_H12_regrets={b:actor_sign*(endpoints["h12"]["label"]["target_player0"]-qs["h12"][b]) for b in best["h10"]},score_units="Seeds; player0 fixedorientation and actualcontroller signs bothretained")


def covariance_identity(raw,drifts,cell_mass=F(1,8)):
    require(len(raw)==len(drifts)==2 and all(type(v) is F for v in raw+drifts) and all(v>0 for v in raw),"Two exact positive reliability scores required")
    a,b=raw;d,e=drifts;uniform=(d+e)/2;weighted=(a*d+b*e)/(a+b);cov=(a-b)*(d-e)/4;difference=(a-b)*(d-e)/(2*(a+b));require(weighted-uniform==difference==cov/((a+b)/2),"Two-root covariance identity failed")
    return dict(raw_reliabilities=[rational(v) for v in raw],root_drifts=[rational(v) for v in drifts],conditional_uniform=rational(uniform),conditional_weighted=rational(weighted),conditional_delta=rational(difference),uniform_covariance=rational(cov),raw_mean=rational((a+b)/2),cell_mass=rational(cell_mass),global_delta=rational(cell_mass*difference),identity_verified=True,causal_claim=False)


def context(data):
    """All16 roots/allactions; conditional risks and global contributions stay distinct."""
    rows=data["rows"];projection=data["rule"]["projection"];rootrows={};cells={};pairtotals={}
    for key,row in rows.items():
        prob=projection["rows"][key];u,w=fraction(prob["uniform_probability"]),fraction(prob["weighted_probability"]);endpoints={s:data["stores"][s+"_labels"][key] for s in STAGES};analysis=action_analysis(row,endpoints,prob);pairs={}
        for a,b in PAIRS:
            qa,qb=qvalues(endpoints[a]),qvalues(endpoints[b]);p=a+"_to_"+b;d=F(sum(abs(qb[z]-qa[z]) for z in qa),len(qa));best=endpoints[a]["label"]["best_moves"];r=analysis["pairs"][p]
            values=dict(action_Q_drift=d,max_action_Q_drift=F(max(abs(qb[z]-qa[z]) for z in qa)),root_drift=F(abs(endpoints[b]["label"]["target_player0"]-endpoints[a]["label"]["target_player0"])),before_best_regret_mean=fraction(r["before_best_regret_mean"]),before_best_regret_min=fraction(r["before_best_regret_min"]),before_best_regret_max=fraction(r["before_best_regret_max"]),before_best_lexicographic_regret=fraction(r["before_best_lexicographic_regret"]),best_action_Q_drift_contribution=F(sum(abs(qb[z]-qa[z]) for z in best),len(qa)),nonbest_action_Q_drift_contribution=F(sum(abs(qb[z]-qa[z]) for z in qa if z not in best),len(qa)))
            require(values["best_action_Q_drift_contribution"]+values["nonbest_action_Q_drift_contribution"]==d,"Best/nonbest numeric decomposition differs")
            pairs[p]=dict(drift=rational(d),uniform_contribution=rational(u*d),weighted_contribution=rational(w*d),global_delta=rational((w-u)*d),metrics={m:dict(value=rational(v),uniform_global_contribution=rational(u*v),weighted_global_contribution=rational(w*v),global_weighted_minus_uniform=rational((w-u)*v)) for m,v in values.items()})
        rootrows[key]=dict(state_key=key,cell=row["cell"],stratum=row["stratum"],actor=row["actor"],family=row["family"],legal_actions=sorted(qvalues(endpoints["h10"])),source_root_sha256=row["source_root_sha256"],source_wrapper_sha256=config.digest(row),endpoint_item_sha256={s:config.digest(v) for s,v in endpoints.items()},detailed_case=key in data["case_keys"],raw_reliability=prob["raw_rule"]["raw_weight"],raw_rule=prob["raw_rule"],uniform_probability=prob["uniform_probability"],weighted_probability=prob["weighted_probability"],pairs=pairs,analysis=analysis)
    for cell in old_metrics.CELLS:
        keys=sorted(k for k,r in rows.items() if r["cell"]==cell);require(len(keys)==2,"Source cell no longer contains exactlytwo roots");raw=[fraction(rootrows[k]["raw_reliability"]) for k in keys]
        pairs={}
        for p in (a+"_to_"+b for a,b in PAIRS):
            metrics={m:covariance_identity(raw,[fraction(rootrows[k]["pairs"][p]["metrics"][m]["value"]) for k in keys]) for m in rootrows[keys[0]]["pairs"][p]["metrics"]}
            for m,value in metrics.items():require(fraction(value["global_delta"])==sum((fraction(rootrows[k]["pairs"][p]["metrics"][m]["global_weighted_minus_uniform"]) for k in keys),F()),"Cell and root metric contribution disagree")
            pairs[p]=dict(**metrics["action_Q_drift"],metrics=metrics)
        cells[cell]=dict(state_keys=keys,uniform_mass=rational(sum((fraction(rootrows[k]["uniform_probability"]) for k in keys),F())),weighted_mass=rational(sum((fraction(rootrows[k]["weighted_probability"]) for k in keys),F())),pairs=pairs)
    def scope(keys,p):
        metrics={};counts=len(keys)
        for name in ("uniform","weighted"):
            probabilities={k:fraction(rootrows[k][name+"_probability"]) for k in keys};mass=sum(probabilities.values(),F());totals={m:sum((probabilities[k]*fraction(rootrows[k]["pairs"][p]["metrics"][m]["value"]) for k in keys),F()) for m in next(iter(rootrows.values()))["pairs"][p]["metrics"]}
            metrics[name]=dict(original_selected_mass=rational(mass),selected_roots=counts,ESS=rational(mass*mass/sum((v*v for v in probabilities.values()),F())) if mass else None,global_contributions={m:rational(v) for m,v in totals.items()},conditional_risks={m:rational(v/mass) if mass else None for m,v in totals.items()})
        metrics["weighted_minus_uniform_global"]={m:rational(fraction(metrics["weighted"]["global_contributions"][m])-fraction(metrics["uniform"]["global_contributions"][m])) for m in metrics["uniform"]["global_contributions"]}
        metrics["weighted_minus_uniform_conditional"]={m:rational(fraction(metrics["weighted"]["conditional_risks"][m])-fraction(metrics["uniform"]["conditional_risks"][m])) if counts else None for m in metrics["uniform"]["conditional_risks"]}
        return dict(state_keys=keys,roots=counts,**metrics)
    scopes={}
    for p in (a+"_to_"+b for a,b in PAIRS):
        def total(keys,field):return rational(sum((fraction(rootrows[k]["pairs"][p][field]) for k in keys),F()))
        capture=data["capture_keys"];other=sorted(set(rows)-set(capture));allkeys=sorted(rows)
        pairtotals[p]=dict(uniform_global_drift=total(allkeys,"uniform_contribution"),weighted_global_drift=total(allkeys,"weighted_contribution"),global_delta=total(allkeys,"global_delta"),capture_global_delta=total(capture,"global_delta"),non_capture_global_delta=total(other,"global_delta"),by_stratum={s:total([k for k,r in rows.items() if r["stratum"]==s],"global_delta") for s in old_metrics.STRATA})
        require(fraction(pairtotals[p]["global_delta"])==fraction(pairtotals[p]["capture_global_delta"])+fraction(pairtotals[p]["non_capture_global_delta"]),"Global capture/noncapture partition failed")
        scopes[p]=dict(overall=scope(allkeys,p),capture=scope(capture,p),non_capture=scope(other,p),ordinary=scope([k for k,r in rows.items() if r["stratum"].endswith("/ordinary")],p),detailed_cases=scope(data["case_keys"],p),other_cases=scope(sorted(set(rows)-set(data["case_keys"])),p),by_stratum={s:scope([k for k,r in rows.items() if r["stratum"]==s],p) for s in old_metrics.STRATA},by_cell={c:scope([k for k,r in rows.items() if r["cell"]==c],p) for c in old_metrics.CELLS},by_actor={str(a):scope([k for k,r in rows.items() if r["actor"]==a],p) for a in (0,1)})
    expected=data["previous"]["metrics"]["overall"]
    aliases=dict(action_Q_drift="action_Q_drift",max_action_Q_drift="max_action_Q_drift",root_drift="root_drift",before_best_regret_mean="H10_best_H12_regret_mean",before_best_regret_min="H10_best_H12_regret_min",before_best_regret_max="H10_best_H12_regret_max",before_best_lexicographic_regret="H10_lexicographic_H12_regret",best_action_Q_drift_contribution="H10_best_action_Q_drift_contribution",nonbest_action_Q_drift_contribution="H10_nonbest_action_Q_drift_contribution")
    for name in ("uniform","weighted"):
        for own,old in aliases.items():require(equal(scopes["h10_to_h12"]["overall"][name]["conditional_risks"][own],expected[name][old]),"All16 context does not reproduce sealed metric: "+old)
    return dict(roots=rootrows,cells=cells,pairs=pairtotals,scopes=scopes,global_roots=16,capture_roots=4,detailed_case_roots=5,uniform_global_denominator=16,root_action_denominator="Each root averagesALLlegalactions before fixed rootprobability",conditional_mass_scope="Conditional risks divide by declared subset mass; global contributions retain all16 root probabilities",ESS_interpretation="Inverse concentration of probabilities, not independent samples or uncertainty",probabilities_changed=False)


def panel_body(data):return dict(schema=SCHEMA+"-panel",all_state_keys=sorted(data["rows"]),case_ids=list(CASE_IDS),case_state_keys=data["case_keys"],capture_state_keys=data["capture_keys"],source_rows_sha256=config.digest(data["rows"]),source_endpoints_sha256={s:config.digest(data["stores"][s+"_labels"]) for s in STAGES},source_histories_sha256={s:config.digest(data["stores"][s+"_attempts"]) for s in STAGES},source_rule_sha256=sha(ROOT/"build/qstability-rule.json"),inputs=data["inputs"],retrospective=True,new_reservations=False)


def prepare():return publish(PANEL,panel_body(source()))


def plan_body():return dict(schema=SCHEMA,contract=CONTRACT,panel_sha256=sha(PANEL),baseline_sha256=sha(BASELINE),source_inputs=common.pins(SOURCE_FILES),models=common.pins([common.CONTROL,common.CANDIDATE]),rules_digest=config.RULES_DIGEST)


def plan():prepare();return publish(PLAN,plan_body())


def implementation_body():return dict(schema=SCHEMA+"-implementation",plan_sha256=sha(PLAN),sources=common.pins(SOURCES+DEPENDENCIES))


def freeze():
    plan();body=implementation_body();publish(IMPLEMENTATION,body);return publish(SOURCE_FREEZE,body)


def admit():
    data=source();require(equal(read(PANEL),panel_body(data)) and equal(read(PLAN),plan_body()) and equal(read(IMPLEMENTATION),implementation_body()) and equal(read(SOURCE_FREEZE),implementation_body()),"Frozen capture inputs/source/protocol differ");return data


AUDIT_METHODS=["test_01_prior5252_sources_panel_reservations_and_models","test_02_all16_signed_attribution_fullties_covariance_and_crossing","test_03_real_case_pre_post_commit_kills_and_exact_resume"]


def gate():
    path=ROOT/"build/regret-audit/preflight.json";body=read(path);expected=dict(schema="regret-independent-group-v1",group="preflight",passed=True,tests_run=3,errors=0,failures=0,skips=0,covered_methods=AUDIT_METHODS,covered_checks=["__main__.RegretIndependentAudit."+m for m in AUDIT_METHODS])
    require(equal({k:body[k] for k in expected},expected),"Independent capture preflight incomplete")
    needed={"build/regret-baseline.json","docs/REGRET_SCOPE.md","congklak/regret_common.py","tests/regret_independent_audit.py","tools/regret-audit.py","build/regret-audit-source-freeze.json","congklak/regret_drift.py","congklak/regret_forensics.py","build/regret-drift-plan.json","build/regret-drift-implementation.json","build/regret-drift-source-freeze.json","build/regret-drift-panel.json","build/regret-drift-author.json","build/regret-forensics-author.json"}
    inputs=body["inputs"];require(needed<=set(inputs) and all(inputs.get(k)==v for k,v in common.baseline()["inputs"].items()),"Capture preflight omits currentsource/oldinputs");common.current(inputs)
    owners={}
    for name,test,module in (("regret-drift-author.json","tests/regret_drift_author.py","congklak/regret_drift.py"),("regret-forensics-author.json","tests/regret_forensics_author.py","congklak/regret_forensics.py")):
        owner=read(ROOT/"build"/name);require(owner["passed"] is True and type(owner["tests_run"]) is int and owner["tests_run"]==4 and all(type(owner[k]) is int and owner[k]==0 for k in ("errors","failures","skips")),"Capture owner proof incomplete")
        tree=ast.parse((ROOT/test).read_text(encoding="utf-8-sig"));expected_ids={node.name+"."+m.name for node in tree.body if isinstance(node,ast.ClassDef) for m in node.body if isinstance(m,ast.FunctionDef) and m.name.startswith("test_")};ids=owner["covered_methods"];require(len(ids)==len(set(ids))==4 and {".".join(m.split(".")[-2:]) for m in ids}==expected_ids,"Owner executed methods differ from source")
        require(owner["inputs"].get(test)==sha(ROOT/test) and owner["inputs"].get(module)==sha(ROOT/module),"Owner executed source missing");common.current(owner["inputs"]);owners[name]=sha(ROOT/"build"/name)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),owners=owners)


def settings_for(fixture=False,cases=None):
    require(type(fixture) is bool,"Fixture flag must be boolean")
    if fixture:n=2 if cases is None else config.integer(cases,"fixture_cases",1,2)
    else:require(cases is None,"Production requiresALLfive fixedcases");n=5
    return dict(fixture=fixture,cases=n,global_context_roots=16,fixture_panel="Firsttwoalreadyconsumedcases178and91" if fixture else None)


def identity_for(data,settings):return dict(schema=SCHEMA,settings=settings,rules_digest=config.RULES_DIGEST,artifacts=common.pins([PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE]),baseline_sha256=sha(BASELINE),source_inputs=data["inputs"],model_files=common.pins([common.CONTROL,common.CANDIDATE]),accepted_native_build_metadata=data["manifest"]["identity"]["native_build_sha256"],production_gate=None if settings["fixture"] else gate(),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def compute_case(index,data):
    key=data["case_keys"][index];row=data["rows"][key];endpoints={s:data["stores"][s+"_labels"][key] for s in STAGES};attempts={s:data["stores"][s+"_attempts"][key] for s in STAGES};prob=data["rule"]["projection"]["rows"][key]
    return dict(schema=SCHEMA+"-case",idx=index,case_id=CASE_IDS[index],state_key=key,state=row["state"],actor=row["actor"],cell=row["cell"],family=row["family"],source_root_sha256=row["source_root_sha256"],source_wrapper_sha256=config.digest(row),endpoint_item_sha256={s:config.digest(v) for s,v in endpoints.items()},source_attempt_sha256={s:config.digest(v) for s,v in attempts.items()},uniform_probability=prob["uniform_probability"],weighted_probability=prob["weighted_probability"],raw_rule=prob["raw_rule"],analysis=action_analysis(row,endpoints,prob),forensics=forensics.build_case(row,endpoints,attempts),finite_H12_not_game_truth=True,weights_changed=False)


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
                body=compute_case(index,data);event("regret_precommit",completed=len(saved),target=settings["cases"],state_key=body["state_key"],case_index=index);work.admit(len(config.canonical(body))*3+2*1024**2)
                with work.db:work.db.execute("INSERT INTO cases VALUES(?,?,?)",(index,config.canonical(body),config.digest(body)))
                saved[index]=body;event("regret_committed",completed=len(saved),target=settings["cases"],state_key=body["state_key"],case_index=index)
            validate_cases(saved,data,settings);manifest=manifest_for(identity,saved)
            with work.db:work.put("complete",manifest)
        saved,manifest,_=read_core(work.db,path);publish(path/"manifest.json",manifest);event("regret_drift_complete",completed=len(saved),target=settings["cases"]);return manifest
    finally:work.close()


def load_rows(path=DEFAULT):
    require("torch" not in sys.modules,"Capture reader forbidsTorch");path=Path(path).resolve()
    with closing(ro(path/"work.sqlite3")) as db:
        db.execute("BEGIN");cases,manifest,_=read_core(db,path);require(equal(read(path/"manifest.json"),manifest),"Published manifest differs from validmarker");return cases,manifest


def build(path=DEFAULT):
    cases,manifest=load_rows(path);data=admit();ctx=context(data);pair=ctx["pairs"]["h10_to_h12"]
    regret=ctx["scopes"]["h10_to_h12"]["overall"];capture=ctx["scopes"]["h10_to_h12"]["capture"]
    return dict(schema=SCHEMA+"-result",passed=True,status="complete",fixture=manifest["identity"]["settings"]["fixture"],cases=len(cases),summary=dict(global_Q_drift_delta=pair["global_delta"],capture_Q_drift_delta=pair["capture_global_delta"],non_capture_Q_drift_delta=pair["non_capture_global_delta"],global_best_action_regret_delta=regret["weighted_minus_uniform_global"]["before_best_regret_mean"],capture_best_action_regret_delta=capture["weighted_minus_uniform_global"]["before_best_regret_mean"],all16_context=True,capture_best_sets_stable=all(ctx["roots"][k]["analysis"]["best_H10_H12_same"] for k in data["capture_keys"]),case_ids=[v["case_id"] for v in cases.values()],bonus178_best_set_changed=not ctx["roots"][CASE_KEYS[0]]["analysis"]["best_H10_H12_same"]),context=ctx,case_results=cases,manifest_sha256=sha(Path(path)/"manifest.json"),weights_applied=False,training_performed=False,promotion=False,new_teacher_searches=0,new_NN_forwards=0,new_games=0,previous_allocation="hold",new_reservations=False,retrospective=True,
        inputs=common.pins([BASELINE,PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE,Path(path)/"work.sqlite3",Path(path)/"manifest.json",*SOURCE_FILES,*SOURCES,*DEPENDENCIES]))


def assess():return publish(OUTPUT,build())


def validate_report(report,path=DEFAULT):require(equal(report,build(path)),"Capture report metrics/source/evidence do not rederive");return report


def verify(path=DEFAULT):return validate_report(read(OUTPUT),path)
