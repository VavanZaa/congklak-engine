"""Role-aware bounded legal-history collection; no teacher or optimizer calls."""
from collections import Counter,defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import time
from . import config,pilot,data_pilot as old,deeper_teacher_v2 as teacher
from .native import ROOT,Native,flatten
from .neural_relay import Model
from .puct_cpu_inference import CpuBatchEvaluator

SCHEMA="role-protected-targeted-midgame-collection-v1"
DEFAULT=ROOT/"data/midgame-data-pilot-v1"
BASELINE=ROOT/"build/midgame-pilot-baseline.json"
REGISTRY=ROOT/"build/midgame-data-pilot-registry.json"
PLAN=ROOT/"build/midgame-data-pilot-plan.json"
IMPLEMENTATION=ROOT/"build/midgame-data-pilot-implementation.json"
OUTPUT=ROOT/"build/midgame-data-pilot-result.json"
MODEL,MODEL_SHA=old.MODEL,old.MODEL_SHA
SOURCES=("congklak/midgame_data_pilot.py","tools/midgame-data-pilot.py","tests/midgame_data_pilot_author.py","docs/MIDGAME_DATA_PILOT_SCOPE.md","docs/MIDGAME_PILOT_SCOPE.md")
STRATA=("50-74/ordinary","25-49/ordinary","25-49/bonus","25-49/capture")
CONTRACT={"families":512,"seed":20261023,"max_choices":192,"random_opening_choices":8,"epsilon":.25,
          "generation":"Even-index random; odd-index frozen-policy epsilon0.25; legal standard-start prefixes, stop below25",
          "family_namespace":"midgame-history-pilot/seed/index","ownership":"sha256(midgame-pilot-owner-v1/+family) first8bytes big-endian mod100; train<80 elsevalidation, before generation",
          "eligible_strata":list(STRATA),"minimum_legal_actions":2,
          "caps":{"training_candidate":{"50-74/ordinary":16,"25-49/ordinary":8,"25-49/bonus":8,"25-49/capture":8},
                  "validation":{"50-74/ordinary":4,"25-49/ordinary":2,"25-49/bonus":2,"25-49/capture":2}},
          "closure":"Transitive family-eligiblequotient components; exclude entire components touching protectedquotient/oldfamily or mixedownership",
          "selection":"Whole fixedpool first; deterministic familyindex/step/key order; one root/family/stratum; deduplicate quotient",
          "shared_outside_eligible_prefixes_possible":True,"constructed_filler":False,"automatic_expansion":False,
          "teacher_searches":False,"training_performed":False,"sealed_final_test_consumed":False}
IDENTITY_FIELDS={"schema","settings","fixture","rules_digest","fixed_model_sha256","fixed_model_weights_sha256","source_sha256","tool_sha256","registry_sha256","baseline_sha256","native_build_sha256","plan_sha256","implementation_sha256","reserve_bytes","quota_bytes"}
MANIFEST_FIELDS={"schema","identity","status","families","selected_roots","selection","family_semantic_sha256","roots_sha256","registry_sha256","teacher_searches","training_performed","torch_imported","sealed_final_test_consumed","elapsed_seconds_initial_run"}
require,sha,read,publish,table,stopped=old.require,old.sha,old.read,old.publish,old.table,old.stopped


def baseline():
    value=read(BASELINE)
    require(value["schema"]=="midgame-pilot-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==2639 and value["prior_accepted_inputs"]==2637 and value["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA,"Midgame baseline differs")
    require(value["fixed_candidate_model_sha256"]==sha(ROOT/"data/value-pilot-v1/model.json"),"Frozen value-pilot model differs")
    for name,digest in value["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==digest,"Sealed input changed: "+name)
    return value


def raw_key(p):return bytes(flatten(p)+[p["controller"]]).hex()


def metadata_roots(filename,nested=False):
    with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");values=table(db,"roots")
    return {key:(row["root"] if nested else row) for key,row in values.items()}


def make_registry():
    baseline();native=Native();quotients=set();families=set();sources=[]
    for filename in ("build/value-registry.json","build/data-pilot-registry.json","build/natural-study-registry.json"):
        value=read(ROOT/filename);quotients.update(value["quotients"]);families.update(value["families"]);sources.append(dict(path=filename,sha256=sha(ROOT/filename),projection="protectedquotients/families only"))
    admitted=read(ROOT/"build/value-pilot-data.json");quotients.update(admitted["protected_quotients"]);families.update(admitted["protected_families"])
    historical=metadata_roots("data/ordinary-data-pilot-v1/work.sqlite3")
    for key,row in metadata_roots("data/natural-study-v1/work.sqlite3",True).items():
        require(key not in historical or historical[key]==row,"Historical role root changed");historical[key]=row
    require(Counter(row["cohort"] for row in historical.values())==dict(training_candidate=61,validation=14,diagnostic=8),"Historical83 roles differ")
    prior_pool=[]
    for filename,expected in (("data/ordinary-data-pilot-v1/work.sqlite3",256),("data/natural-study-v1/work.sqlite3",512)):
        with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:
            projection=db.execute("SELECT idx,json_extract(body,'$.family'),hash FROM families ORDER BY idx").fetchall()
        require(len(projection)==expected,"Historical full family pool differs");families.update(row[1] for row in projection);prior_pool.extend([filename,*row] for row in projection)
    for row in historical.values():quotients.add(row["quotient"]);families.add(row["family"])
    ordinary_dev=[]
    with closing(sqlite3.connect((ROOT/"data/deeper-9000-v2/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for key,p,family,stratum,digest in db.execute("SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family'),json_extract(body,'$.stratum'),hash FROM roots WHERE json_extract(body,'$.split')='development' AND json_extract(body,'$.stratum') LIKE '%/ordinary' ORDER BY state"):
            p=json.loads(p);q=pilot.quotient(p).hex();quotients.add(q);families.add(family);ordinary_dev.append(dict(state_key=key,quotient=q,family=family,stratum=stratum,source_root_sha256=digest))
    require(len(ordinary_dev)==259,"Base ordinary development coverage differs")
    plan=read(ROOT/"build/value-pilot-benchmark-plan.json");evaluation=[]
    for role in ("cases","arena_starts"):
        for item in plan[role]:
            p=native.normalize(flatten(item["state"]),actor=item["state"]["controller"],total=98)["position"];q=pilot.quotient(p).hex();quotients.add(q)
            if item.get("family"):families.add(item["family"])
            evaluation.append(dict(role=role,state_key=raw_key(p),quotient=q,family=item.get("family"),source_sha256=config.digest(item)))
    require(len(plan["cases"])==44 and len(plan["arena_starts"])==12,"Latest evaluation roots differ")
    latest_q=set();latest_f=set();games=[]
    for stage,expected in (("pretraining",48),("posttraining",96)):
        filename="data/value-pilot-benchmark-"+stage+"-v1/work.sqlite3";seen=0
        with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:
            db.execute("BEGIN")
            for idx,body,digest in db.execute("SELECT idx,body,hash FROM games ORDER BY idx"):
                game=json.loads(body);require(config.digest(game)==digest,"Latest arena game corrupt");p=game["start"];latest_f.add(game["family"]);trajectory=[]
                for action in [*game["moves"],None]:
                    p=native.normalize(flatten(p),actor=p["controller"],total=98)["position"];q=pilot.quotient(p).hex();latest_q.add(q);trajectory.append(dict(state_key=raw_key(p),quotient=q))
                    if action is not None:
                        transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Latest protected arena move illegal");p=transition["position"]
                require(p==game["final"],"Latest protected arena final differs");games.append(dict(path=filename,index=idx,game_sha256=digest,family=game["family"],states=trajectory));seen+=1
        require(seen==expected,"Latest full game count differs")
    quotients.update(latest_q);families.update(latest_f)
    arena=read(ROOT/"build/pilot-loss-review-arena.json");diagnostics={}
    for seat in arena["seats"]:
        div=seat["first_divergence"]
        if div["identical_action_trajectory"]:continue
        p=div["state"];normal=native.normalize(flatten(p),actor=p["controller"],total=98);require(p==normal["position"] and not normal["terminal"],"Diagnostic literal root differs")
        key=raw_key(p);active,stratum=pilot._mechanism(p,native);q=pilot.quotient(p).hex()
        provenance=dict(source="saved_first_divergence",source_report="build/pilot-loss-review-arena.json",source_report_sha256=sha(ROOT/"build/pilot-loss-review-arena.json"),pair=seat["pair"],candidate_seat=seat["candidate_seat"],first_different_choice=div["first_different_choice"],control_chosen=div["control"]["summary"]["chosen"],pilot_chosen=div["pilot"]["summary"]["chosen"],existing_Q_comparison=div["existing_Q_comparison"],raw_state=p,swapped=False)
        row=dict(state_key=key,state=p,quotient=q,family=seat["family"],cohort="diagnostic",purpose="protected_first_divergence",band=stratum.split("/")[0],stratum=stratum,active=active,legal_mask=sum(1<<a for a in normal["legal_actions"]),terminal_children=sum(native.apply(flatten(p),actor=p["controller"],action=a,total=98,trace=False)["terminal"] for a in normal["legal_actions"]),provenance=provenance)
        require(key not in diagnostics,"Duplicate diagnostic literal root");diagnostics[key]=row;quotients.add(q);families.add(seat["family"])
    require(len(diagnostics)==9 and sum(row["provenance"]["existing_Q_comparison"]["coverage"]=="uncovered" for row in diagnostics.values())==8,"Nine diagnostic/eight uncovered states differ")
    evaluation_q=latest_q|{row["quotient"] for row in evaluation};evaluation_f=latest_f|{row["family"] for row in evaluation if row["family"]}
    roles={key:dict(state_key=key,quotient=row["quotient"],family=row["family"],cohort=row["cohort"],original_root_sha256=config.digest(row),latest_eval_quotient_collision=row["quotient"] in evaluation_q,latest_eval_family_collision=row["family"] in evaluation_f) for key,row in sorted(historical.items())}
    value=dict(schema="midgame-role-aware-protection-v1",rules_digest=config.RULES_DIGEST,quotients=sorted(quotients),families=sorted(families),historical_roles=roles,historical_role_counts=dict(Counter(row["cohort"] for row in roles.values())),prior_family_pool_count=768,prior_family_pool_sha256=config.digest(prior_pool),base_ordinary_development=ordinary_dev,evaluation_roots=evaluation,latest_arena_games=games,latest_arena_game_count=144,latest_arena_quotients=sorted(latest_q),latest_arena_families=sorted(latest_f),diagnostics=dict(sorted(diagnostics.items())),sources=sources,baseline_sha256=sha(BASELINE),sealed_target_bodies_read=False,historical_roles_rewritten=False,future_protection_overrides_eligibility=True,ancestry_cannot_be_untrained=True)
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def plan():
    registry=make_registry();value=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,fixed_model_sha256=MODEL_SHA,baseline_sha256=sha(BASELINE),registry_sha256=sha(REGISTRY),diagnostics_sha256=config.digest(registry["diagnostics"]),shared_prefixes_outside_eligibility_possible=True)
    publish(PLAN,value);return value


def freeze():
    plan();b=baseline();value=dict(schema="midgame-data-pilot-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},prior_inputs=b["inputs"])
    publish(IMPLEMENTATION,value);return value


def admit():
    b=baseline();value,implementation=read(PLAN),read(IMPLEMENTATION)
    require(value["schema"]==SCHEMA and value["contract"]==CONTRACT and value["rules_digest"]==config.RULES_DIGEST and value["fixed_model_sha256"]==MODEL_SHA,"Midgame fixed plan differs")
    require(implementation==dict(schema="midgame-data-pilot-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},prior_inputs=b["inputs"]),"Midgame frozen producer differs")
    return value,implementation


def settings_for(fixture=False,families=None):
    value=json.loads(config.canonical(CONTRACT))
    if fixture:
        value["families"]=8 if families is None else config.integer(families,"fixture_families",1,32);value["caps"]={cohort:{stratum:1 for stratum in STRATA} for cohort in CONTRACT["caps"]}
    else:require(families is None,"Midgame production pool differs")
    return value


def candidate(p,native,family,cohort,provenance,settings):
    key,state,swapped=pilot.canonicalize(p,native);active,stratum=pilot._mechanism(state,native)
    normal=native.normalize(flatten(state),actor=state["controller"],total=98)
    if stratum not in settings["eligible_strata"] or len(normal["legal_actions"])<settings["minimum_legal_actions"]:return None
    return dict(state_key=key.hex(),state=state,quotient=pilot.quotient(state).hex(),family=family,cohort=cohort,purpose="targeted_midgame_pilot",band=stratum.split("/")[0],stratum=stratum,active=active,legal_mask=sum(1<<a for a in normal["legal_actions"]),terminal_children=sum(native.apply(flatten(state),actor=state["controller"],action=a,total=98,trace=False)["terminal"] for a in normal["legal_actions"]),provenance={**provenance,"raw_state":p,"swapped":swapped})


def generate_family(index,settings,native,evaluator):
    family="midgame-history-pilot/"+str(settings["seed"])+"/"+str(index)
    rank=int.from_bytes(hashlib.sha256(("midgame-pilot-owner-v1/"+family).encode()).digest()[:8],"big")%100;cohort="training_candidate" if rank<80 else "validation"
    source="random_history" if index%2==0 else "frozen_policy_epsilon_history";rng=random.Random(settings["seed"]+index*104729);p=config.position();moves=[];candidates=[];forwards=0
    for step in range(settings["max_choices"]):
        normal=native.normalize(flatten(p),actor=p["controller"],total=98);p=normal["position"]
        if normal["terminal"] or sum(map(sum,p["pits"]))<25:break
        item=candidate(p,native,family,cohort,dict(source=source,family_index=index,step=step,start=config.position(),moves_prefix=list(moves),source_game_idx=None,source_comparison=None,saved_puct_chosen=None),settings)
        if item is not None:candidates.append(item)
        legal=normal["legal_actions"]
        if index%2==0 or step<settings["random_opening_choices"] or rng.random()<settings["epsilon"]:action=rng.choice(legal)
        else:
            prediction=evaluator.evaluate_many([p])[0];forwards+=1;action=max(legal,key=lambda a:(prediction["policy_logits"][a],-a))
        transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Fresh midgame history action illegal");moves.append(action);p=transition["position"]
    return dict(index=index,family=family,cohort=cohort,source=source,rng_seed=settings["seed"]+index*104729,start=config.position(),moves=moves,final=p,choices=len(moves),candidates=candidates,policy_evaluations=forwards,stop_reason="terminal" if not sum(map(sum,p["pits"])) else "below25" if sum(map(sum,p["pits"]))<25 else "choice_cap")


def select(families,settings,registry):
    observations=[row for family in families.values() for row in family["candidates"]];owners={family["family"]:family["cohort"] for family in families.values()};parent={name:name for name in owners};qfamilies=defaultdict(set)
    def find(name):
        while parent[name]!=name:parent[name]=parent[parent[name]];name=parent[name]
        return name
    for row in observations:qfamilies[row["quotient"]].add(row["family"])
    for names in qfamilies.values():
        ordered=sorted(names)
        for name in ordered[1:]:parent[find(name)]=find(ordered[0])
    components=defaultdict(lambda:dict(families=set(),quotients=set()))
    for row in observations:
        entry=components[find(row["family"])];entry["families"].add(row["family"]);entry["quotients"].add(row["quotient"])
    protected_q,protected_f=set(registry["quotients"]),set(registry["families"]);blocked=set();details=[]
    for entry in components.values():
        cohorts=sorted({owners[name] for name in entry["families"]});reasons=[]
        if entry["quotients"]&protected_q:reasons.append("protected_eligible_quotient")
        if entry["families"]&protected_f:reasons.append("old_protected_family")
        if len(cohorts)>1:reasons.append("mixed_ownership_component")
        if reasons:blocked.update(entry["families"])
        details.append(dict(families=sorted(entry["families"]),quotients=sorted(entry["quotients"]),cohorts=cohorts,reasons=reasons))
    rows={};used_q=set();used_family_stratum=set();counts=Counter()
    for row in sorted(observations,key=lambda row:(row["provenance"]["family_index"],row["provenance"]["step"],row["state_key"])):
        cohort,stratum=row["cohort"],row["stratum"]
        if row["family"] in blocked or row["quotient"] in used_q or (row["family"],stratum) in used_family_stratum or counts[cohort,stratum]>=settings["caps"][cohort][stratum]:continue
        rows[row["state_key"]]=row;used_q.add(row["quotient"]);used_family_stratum.add((row["family"],stratum));counts[cohort,stratum]+=1
    selection=dict(observations=len(observations),observed_by_stratum=dict(sorted(Counter(row["stratum"] for row in observations).items())),unique_candidate_quotients=len(qfamilies),eligible_family_count=len({row["family"] for row in observations}),components=sorted(details,key=lambda entry:entry["families"]),component_count=len(details),blocked_family_count=len(blocked),blocked_families=sorted(blocked),blocked_candidate_observations=sum(row["family"] in blocked for row in observations),direct_protected_candidate_observations=sum(row["quotient"] in protected_q for row in observations),cross_cohort_candidate_quotients=sum(len({owners[name] for name in names})>1 for names in qfamilies.values()),counts={cohort:{stratum:counts[cohort,stratum] for stratum in STRATA} for cohort in settings["caps"]},shortfall={cohort:{stratum:settings["caps"][cohort][stratum]-counts[cohort,stratum] for stratum in STRATA} for cohort in settings["caps"]},observed_legal_action_counts=dict(sorted(Counter(str(row["legal_mask"].bit_count()) for row in observations).items())),selected_legal_action_counts=dict(sorted(Counter(str(row["legal_mask"].bit_count()) for row in rows.values()).items())),selected_families=len({row["family"] for row in rows.values()}),shared_outside_eligible_prefixes_possible=True)
    return dict(sorted(rows.items())),selection


def checked_registry():
    value=read(REGISTRY);require(value["schema"]=="midgame-role-aware-protection-v1" and value["sha256"]==config.digest({key:item for key,item in value.items() if key!="sha256"}),"Midgame registry differs");return value


def validate_complete(value,identity,families,rows,selection):
    require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["reserve_bytes"]==2*1024**3 and identity["quota_bytes"]==50*1024**3,"Collection identity fields/schema/reserve differ")
    require(set(value)==MANIFEST_FIELDS and value["schema"]==SCHEMA and value["identity"]==identity and value["status"]=="complete","Collection complete marker differs")
    require(list(families)==list(range(identity["settings"]["families"])) and type(value["families"]) is int and value["families"]==len(families) and type(value["selected_roots"]) is int and value["selected_roots"]==len(rows),"Collection complete prefix/count differs")
    require(value["selection"]==selection and value["family_semantic_sha256"]==config.digest(families) and value["roots_sha256"]==config.digest(rows) and value["registry_sha256"]==identity["registry_sha256"],"Collection complete digests/selection differ")
    require(value["teacher_searches"] is False and value["training_performed"] is False and value["torch_imported"] is False and value["sealed_final_test_consumed"] is False and type(value["elapsed_seconds_initial_run"]) in (int,float) and math.isfinite(value["elapsed_seconds_initial_run"]) and value["elapsed_seconds_initial_run"]>0,"Collection complete scope/clock differs")


def collect(path=DEFAULT,fixture=False,families=None,on_event=None,cancel=None):
    path=Path(path).resolve();settings=settings_for(fixture,families)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Collection fixture must remain in build")
    else:require(path==DEFAULT.resolve(),"Collection production path differs");admit()
    registry=checked_registry();native=Native();identity=dict(schema=SCHEMA,settings=settings,fixture=fixture,rules_digest=config.RULES_DIGEST,fixed_model_sha256=sha(MODEL),fixed_model_weights_sha256=Model(MODEL).id,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/midgame-data-pilot.py"),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),native_build_sha256=native.build_id,plan_sha256=None if fixture else sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)
    require(identity["fixed_model_sha256"]==MODEL_SHA and "torch" not in sys.modules,"Collection model/runtime differs");work=teacher.Work(path,identity);started=time.monotonic()
    def event(name,**value):
        if on_event:on_event(dict(event=name,**value))
    try:
        work.db.execute("CREATE TABLE IF NOT EXISTS families(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT)");pool=table(work.db,"families","idx");require(list(pool)==list(range(len(pool))) and len(pool)<=settings["families"],"Collection saved family prefix differs")
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,family in pool.items():require(family==generate_family(index,settings,native,evaluator),"Saved legal-history family differs")
            for index in range(len(pool),settings["families"]):
                if stopped(cancel):break
                family=generate_family(index,settings,native,evaluator);event("midgame_collection_precommit",completed=index,target=settings["families"],family=family["family"],candidates=len(family["candidates"]))
                work.admit(len(config.canonical(family).encode())+2*1024**2)
                with work.db:work.db.execute("INSERT INTO families VALUES(?,?,?)",(index,config.canonical(family),config.digest(family)))
                pool[index]=family;event("midgame_collection_committed",completed=index+1,target=settings["families"],family=family["family"],candidates=len(family["candidates"]))
        if len(pool)<settings["families"]:return dict(status="interrupted",families=len(pool),selected_roots=0)
        rows,selection=select(pool,settings,registry);saved=table(work.db,"roots");complete=work.get("complete")
        if complete is None:
            require(not saved or saved==rows,"Collection saved root selection differs")
            with work.db:
                for key,row in rows.items():work.db.execute("INSERT OR IGNORE INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
                work.put("selection",selection)
            complete=dict(schema=SCHEMA,identity=identity,status="complete",families=len(pool),selected_roots=len(rows),selection=selection,family_semantic_sha256=config.digest(pool),roots_sha256=config.digest(rows),registry_sha256=sha(REGISTRY),teacher_searches=False,training_performed=False,torch_imported=False,sealed_final_test_consumed=False,elapsed_seconds_initial_run=time.monotonic()-started)
            validate_complete(complete,identity,pool,rows,selection)
            with work.db:work.put("complete",complete)
        else:require(saved==rows and work.get("selection")==selection,"Closed collection table/selection differs")
        validate_complete(complete,identity,pool,rows,selection);publish(path/"manifest.json",complete)
        if not fixture:publish(OUTPUT,complete)
        load_rows(path);event("midgame_collection_complete",families=len(pool),selected_roots=len(rows),shortfall=selection["shortfall"]);return complete
    finally:work.close()


def read_families(path=DEFAULT):
    with closing(sqlite3.connect((Path(path).resolve()/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:return table(db,"families","idx")


def load_rows(path=DEFAULT):
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN")
        def meta(key):
            row=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(row is not None,"Collection missing committed "+key);value=json.loads(row[0]);require(config.digest(value)==row[1],"Collection metadata corrupt");return value
        identity,complete=meta("identity"),meta("complete");settings=identity["settings"]
        require(set(identity)==IDENTITY_FIELDS and identity["schema"]==SCHEMA and identity["source_sha256"]==sha(__file__) and identity["tool_sha256"]==sha(ROOT/"tools/midgame-data-pilot.py") and identity["rules_digest"]==config.RULES_DIGEST and identity["fixed_model_sha256"]==sha(MODEL)==MODEL_SHA and identity["fixed_model_weights_sha256"]==Model(MODEL).id and identity["registry_sha256"]==sha(REGISTRY) and identity["baseline_sha256"]==sha(BASELINE),"Closed collection source/model identity differs")
        if identity["fixture"] is False:
            admit();require(path==DEFAULT.resolve() and settings==CONTRACT and identity["plan_sha256"]==sha(PLAN) and identity["implementation_sha256"]==sha(IMPLEMENTATION),"Closed collection production contract differs")
        else:require(identity["fixture"] is True and path.is_relative_to(ROOT/"build") and settings==settings_for(True,settings["families"]) and identity["plan_sha256"] is None and identity["implementation_sha256"] is None,"Closed collection fixture contract differs")
        native=Native();require(native.build_id==identity["native_build_sha256"],"Closed collection native build differs")
        pool=table(db,"families","idx");require(list(pool)==list(range(settings["families"])),"Closed collection full pool missing")
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,family in pool.items():require(family==generate_family(index,settings,native,evaluator),"Closed collection history differs")
        rows=table(db,"roots");expected,selection=select(pool,settings,checked_registry());require(rows==expected and meta("selection")==selection,"Closed collection ownership/selection differs")
        validate_complete(complete,identity,pool,rows,selection)
        require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(complete)+"\n","Closed collection file/commit manifest differs")
        return rows,complete
