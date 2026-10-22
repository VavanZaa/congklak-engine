"""One bounded data-only ordinary-root collection and fixed-control self-play pilot."""
from collections import Counter,defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
from . import config,pilot,features_relay,midgame_data_pilot as histories,selfplay_training_pilot as previous,phase6_store
from .native import ROOT,Native,flatten
from .neural_relay import Model
from .puct_cpu_inference import CpuBatchEvaluator

SCHEMA="ordinary-natural-coverage-fixed-control-data-pilot-v1"
DEFAULT=ROOT/"data/ordinary-selfplay-pilot-v1"
BASELINE=ROOT/"build/ordinary-selfplay-baseline.json"
REGISTRY=ROOT/"build/ordinary-selfplay-registry.json"
PLAN=ROOT/"build/ordinary-selfplay-plan.json"
IMPLEMENTATION=ROOT/"build/ordinary-selfplay-implementation.json"
COLLECTION=ROOT/"build/ordinary-selfplay-collection.json"
PROTECTION=ROOT/"build/ordinary-selfplay-rollout-protection.json"
OUTPUT=ROOT/"build/ordinary-selfplay-replay.json"
MODEL,MODEL_SHA=previous.SOURCE,previous.SOURCE_SHA
SOURCES=("congklak/ordinary_selfplay_pilot.py","tools/ordinary-selfplay-pilot.py","tests/ordinary_selfplay_pilot_author.py","docs/ORDINARY_SELFPLAY_PRODUCER.md","docs/ORDINARY_SELFPLAY_SCOPE.md")
DEPENDENCIES=("congklak/midgame_data_pilot.py","congklak/selfplay_training_pilot.py","congklak/quality_search.py","congklak/puct_cpu_inference.py","congklak/features_relay.py","congklak/phase6_store.py")
STRATA=("50-74/ordinary","25-49/ordinary")
CONTRACT=dict(families=512,seed=20261026,max_choices=192,random_opening_choices=8,epsilon=.25,
    family_namespace="ordinary-selfplay-history/seed/index",ownership="sha256(ordinary-selfplay-owner-v1/+family) first8bytes big-endian modulo100; train<80 elsevalidation beforegeneration",
    generation="256random/256frozen-policy-epsilon legal standard-start prefixes; stopbelow25",eligible_strata=list(STRATA),minimum_legal_actions=2,
    caps={"training_candidate":{kind:12 for kind in STRATA},"validation":{kind:3 for kind in STRATA}},
    selection="Fullpooltransitiveeligiblequotient/familyclosure then deterministic distinctfamily/quotient quota matching; onefamily/root acrossbothbands",
    rollouts=dict(runs_per_family=2,seed=20261027,simulations=64,max_nodes=2048,max_choices=512,c_puct=.25,memory_mb=8192,root_noise=dict(alpha=.3,fraction=.25),temperature_first_choices=8,rawstate_observation_cap=4),
    optimizer_eligibility="Strictordinary>=25and>=2legal; allother fullgame decisions retainedonlyasprovenance",minimum_usable_roots={kind:8 for kind in STRATA},
    usable_start="Both fullterminal choice0 policy/return records retained; distinct selected rawstate/family perband; descendantsseparate",
    duplicate_order="choice0first then sha256(ordinary-selfplay-rawcap-v1/+rowid); max4globalrawstate",automatic_expansion=False,constructed_filler=False,
    teacher_searches=False,training_performed=False,new_strength_games=False,default_promotion=False,sealed_final_test_consumed=False)
require,read,sha,publish,table=previous.require,previous.read,previous.sha,previous.publish,previous.table


def equal(a,b):return config.canonical(a)==config.canonical(b)


def baseline():
    value=read(BASELINE);require(value["schema"]=="ordinary-selfplay-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==3171 and value["prior_accepted_inputs"]==3168 and value["control_model_sha256"]==sha(MODEL)==MODEL_SHA,"Ordinary baseline/model differs")
    require(value["challenger_model_sha256"]==sha(ROOT/value["challenger_model"]),"Sealed challenger changed")
    for name,digest in value["inputs"].items():require((ROOT/name).resolve().is_relative_to(ROOT) and sha(ROOT/name)==digest,"Prior accepted input changed: "+name)
    return value


def db_rows(filename,name,key="state"):
    with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:return table(db,name,key)


def make_registry():
    baseline();old=read(previous.REGISTRY);novelty=read(ROOT/"build/midgame-data-pilot-registry.json")
    tq,tf=set(old["quotients"]),set(old["families"]);nq,nf=set(novelty["quotients"]),set(novelty["families"])
    recent=db_rows("data/midgame-data-pilot-v1/work.sqlite3","roots")
    for row in recent.values():nq.add(row["quotient"]);nf.add(row["family"])
    pool=db_rows("data/midgame-data-pilot-v1/work.sqlite3","families","idx");require(len(pool)==512,"Prior midgame pool differs");nf.update(row["family"] for row in pool.values())
    old_replay=db_rows("data/selfplay-training-pilot-v1/work.sqlite3","sp_rows","id")
    for row in old_replay.values():nq.add(row["quotient"]);nf.add(row["family"])
    metadata=[]
    with closing(sqlite3.connect((ROOT/"data/deeper-9000-v2/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for key,body,digest in db.execute("SELECT state,body,hash FROM roots ORDER BY state"):
            row=json.loads(body);require(config.digest(row)==digest,"Prior metadata corrupt");q=pilot.quotient(row["state"]).hex();nq.add(q);nf.add(row["family"]);metadata.append([key,q,row["family"],row["split"]])
    require(len(metadata)==9000,"Prior metadata count differs")
    from . import selfplay_pilot_benchmark as benchmark
    witness=benchmark.closed_witness();require(witness["games"]==96,"Latest true evaluation game count differs");native=Native();games=[];observations=0
    with closing(sqlite3.connect((ROOT/"data/selfplay-pilot-benchmark-v1/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for index,body,digest in db.execute("SELECT idx,body,hash FROM games ORDER BY idx"):
            game=json.loads(body);require(config.digest(game)==digest,"Latest protected game corrupt");p=game["start"];trajectory=[];tf.add(game["family"])
            for action in [*game["moves"],None]:
                p=previous.normalize(p,native);q=pilot.quotient(p).hex();tq.add(q);trajectory.append(dict(state_key=previous.rawkey(p),quotient=q));observations+=1
                if action is not None:
                    transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Latest protected evaluation move illegal");p=transition["position"]
            require(p==game["final"],"Latest evaluation final differs");games.append(dict(index=index,family=game["family"],game_sha256=digest,states=trajectory))
    require(len(games)==96,"Latest96 ledger incomplete");nq.update(tq);nf.update(tf)
    value=dict(schema="ordinary-selfplay-distinct-roles-registry-v1",rules_digest=config.RULES_DIGEST,true_evaluation_quotients=sorted(tq),true_evaluation_families=sorted(tf),novelty_quotients=sorted(nq),novelty_families=sorted(nf),latest96_games=games,latest96_state_observations=observations,latest96_witness=witness,
        ancestral_roles=old["historical_roles"],recent_roles=old["latest_roles"],diagnostics=old["diagnostics"],prior_selfplay_rows=len(old_replay),prior_selfplay_unique_rawstates=len({r["state_key"] for r in old_replay.values()}),prior_midgame_fullfamilies=512,
        metadata9000=dict(rows=9000,semantic_sha256=config.digest(metadata),blanket_rollout_evaluation_ban=False),baseline_sha256=sha(BASELINE),prior_training_used_only_as_collection_novelty=True)
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def plan():
    make_registry();value=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,model_sha256=MODEL_SHA,baseline_sha256=sha(BASELINE),registry_sha256=sha(REGISTRY));publish(PLAN,value);return value


def freeze():
    plan();value=dict(schema="ordinary-selfplay-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES});publish(IMPLEMENTATION,value);return value


def admit():
    baseline();value,implementation=read(PLAN),read(IMPLEMENTATION)
    require(equal(value,dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,model_sha256=MODEL_SHA,baseline_sha256=sha(BASELINE),registry_sha256=sha(REGISTRY))),"Ordinary fixed plan differs")
    require(equal(implementation,dict(schema="ordinary-selfplay-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES})),"Ordinary producer freeze differs")
    return value,implementation


def settings_for(fixture=False,families=None):
    require(type(fixture) is bool,"Fixture flag must be boolean");settings=json.loads(config.canonical(CONTRACT))
    if fixture:
        settings["families"]=32 if families is None else config.integer(families,"fixture_families",1,512);settings["caps"]={owner:{kind:1 for kind in STRATA} for owner in CONTRACT["caps"]};settings["rollouts"]["simulations"]=4
        settings["minimum_usable_roots"]={kind:1 for kind in STRATA};settings["fixture_rollout_panel"]="Prioraccepted39trainmetadata: lexicographicfirsthighordinary then firstdistinctfamilylowordinary; fixtureonly; no fallback"
    else:require(families is None,"Production pool budget differs")
    return settings


def identity_for(settings,fixture):
    native=Native();return dict(schema=SCHEMA,fixture=fixture,settings=settings,rules_digest=config.RULES_DIGEST,model_sha256=sha(MODEL),weights_sha256=Model(MODEL).id,native_build_sha256=native.build_id,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/ordinary-selfplay-pilot.py"),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),plan_sha256=sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def checked_identity(identity,path):
    require(type(identity.get("fixture")) is bool,"Invalid fixture identity");fixture=identity["fixture"];settings=settings_for(fixture,identity["settings"]["families"] if fixture else None)
    require(equal(identity,identity_for(settings,fixture)),"Exact ordinary source/settings identity differs")
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build"),"Fixture must stay in build")
    else:require(Path(path).resolve()==DEFAULT.resolve(),"Ordinary production path differs");admit()
    return settings


def candidate(p,native,family,cohort,provenance,settings):
    row=histories.candidate(p,native,family,cohort,provenance,settings)
    if row is not None:row["purpose"]="ordinary_selfplay_natural_coverage"
    return row


def generate_family(index,settings,native,evaluator):
    family="ordinary-selfplay-history/"+str(settings["seed"])+"/"+str(index)
    rank=int.from_bytes(hashlib.sha256(("ordinary-selfplay-owner-v1/"+family).encode()).digest()[:8],"big")%100;cohort="training_candidate" if rank<80 else "validation"
    rng=random.Random(settings["seed"]+index*104729);p=config.position();moves=[];candidates=[];forwards=0;source="random_history" if index%2==0 else "frozen_policy_epsilon_history"
    for step in range(settings["max_choices"]):
        normal=native.normalize(flatten(p),actor=p["controller"],total=98);p=normal["position"]
        if normal["terminal"] or sum(map(sum,p["pits"]))<25:break
        row=candidate(p,native,family,cohort,dict(source=source,family_index=index,step=step,start=config.position(),moves_prefix=list(moves),source_game_idx=None,source_comparison=None,saved_puct_chosen=None),settings)
        if row is not None:candidates.append(row)
        legal=normal["legal_actions"]
        if index%2==0 or step<settings["random_opening_choices"] or rng.random()<settings["epsilon"]:action=rng.choice(legal)
        else:forwards+=1;prediction=evaluator.evaluate_many([p])[0];action=max(legal,key=lambda a:(prediction["policy_logits"][a],-a))
        transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Ordinary history move illegal");moves.append(action);p=transition["position"]
    return dict(index=index,family=family,cohort=cohort,source=source,rng_seed=settings["seed"]+index*104729,start=config.position(),moves=moves,final=p,choices=len(moves),candidates=candidates,policy_evaluations=forwards,stop_reason="terminal" if not sum(map(sum,p["pits"])) else "below25" if sum(map(sum,p["pits"]))<25 else "choice_cap")


def select(pool,settings,registry):
    closure_settings={**settings,"caps":{owner:{kind:settings["caps"][owner].get(kind,0) for kind in histories.STRATA} for owner in settings["caps"]}}
    _,closure=histories.select(pool,closure_settings,dict(quotients=registry["novelty_quotients"],families=registry["novelty_families"]))
    blocked=set(closure["blocked_families"]);observations=[row for family in pool.values() for row in family["candidates"]]
    slots=[(owner,kind) for owner in settings["caps"] for kind in STRATA for _ in range(settings["caps"][owner][kind])];assigned={}
    candidates={slot:sorted((row for row in observations if (row["cohort"],row["stratum"])==slot and row["family"] not in blocked),key=lambda row:(row["provenance"]["family_index"],row["provenance"]["step"],row["state_key"])) for slot in set(slots)}
    def place(slot,seen):
        if slot in seen:return False
        seen=seen|{slot}
        for row in candidates[slots[slot]]:
            snapshot=dict(assigned);conflicts=[other for other,value in assigned.items() if other!=slot and (value["family"]==row["family"] or value["quotient"]==row["quotient"])]
            if any(other in seen for other in conflicts):continue
            assigned[slot]=row
            for other in conflicts:assigned.pop(other)
            if all(place(other,seen) for other in conflicts):return True
            assigned.clear();assigned.update(snapshot)
        return False
    for slot in range(len(slots)):place(slot,set())
    roots=dict(sorted((row["state_key"],row) for row in assigned.values()));require(len(roots)==len(assigned)==len({row["family"] for row in roots.values()})==len({row["quotient"] for row in roots.values()}),"Ordinary distinct-root/family matching differs")
    counts=Counter((row["cohort"],row["stratum"]) for row in roots.values());selection={key:closure[key] for key in ("observations","observed_by_stratum","unique_candidate_quotients","eligible_family_count","components","component_count","blocked_family_count","blocked_families","blocked_candidate_observations","direct_protected_candidate_observations","cross_cohort_candidate_quotients","observed_legal_action_counts")}
    selection.update(counts={owner:{kind:counts[owner,kind] for kind in STRATA} for owner in settings["caps"]},shortfall={owner:{kind:settings["caps"][owner][kind]-counts[owner,kind] for kind in STRATA} for owner in settings["caps"]},selected_families=len(roots),selected_legal_action_counts=dict(sorted(Counter(str(row["legal_mask"].bit_count()) for row in roots.values()).items())),shared_outside_eligible_prefixes_possible=True)
    val=[row for family in pool.values() if family["cohort"]=="validation" for row in family["candidates"]]
    protection=dict(quotients=sorted(set(registry["true_evaluation_quotients"])|{row["quotient"] for row in val}),families=sorted(set(registry["true_evaluation_families"])|{family["family"] for family in pool.values() if family["cohort"]=="validation"}),new_validation_candidate_observations=len(val),new_validation_unique_quotients=len({row["quotient"] for row in val}),new_validation_family_ids=sorted(family["family"] for family in pool.values() if family["cohort"]=="validation"),selected_validation_roots=sum(row["cohort"]=="validation" for row in roots.values()))
    return roots,selection,protection


def metadata(db,key):
    item=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(item is not None,"Missing committed "+key);value=json.loads(item[0]);require(config.digest(value)==item[1],"Corrupt metadata "+key);return value


def collection_body(identity,pool,roots,selection,protection):
    return dict(schema=SCHEMA,stage="collection",identity=identity,status="complete",families=len(pool),selected_roots=len(roots),family_semantic_sha256=config.digest(pool),roots_sha256=config.digest(roots),selection=selection,rollout_protection=protection,teacher_searches=False,training_performed=False,new_strength_games=False,sealed_final_test_consumed=False)


def validate_collection(db,identity,path):
    settings=checked_identity(identity,path);pool=table(db,"families","idx");require(list(pool)==list(range(settings["families"])),"Incomplete ordinary family pool")
    with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
        for index,family in pool.items():require(equal(family,generate_family(index,settings,Native(),evaluator)),"Saved ordinary history differs")
    roots,selection,protection=select(pool,settings,read(REGISTRY));body=collection_body(identity,pool,roots,selection,protection)
    require(equal(metadata(db,"collection_complete"),body) and equal(table(db,"roots","state"),roots),"Collection marker/ownership/digests differ")
    return roots,body


def collect(path=DEFAULT,fixture=False,families=None,on_event=None,cancel=None):
    path=Path(path).resolve();settings=settings_for(fixture,families);identity=identity_for(settings,fixture);checked_identity(identity,path);require("torch" not in sys.modules,"Collection must not load Torch");work=phase6_store.Work(path,identity)
    event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS families(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);")
        pool=table(work.db,"families","idx");require(list(pool)==list(range(len(pool))) and len(pool)<=settings["families"],"Invalid saved ordinary family prefix")
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,family in pool.items():require(equal(family,generate_family(index,settings,Native(),evaluator)),"Saved collection prefix differs")
            for index in range(len(pool),settings["families"]):
                if cancel and cancel():return dict(status="interrupted",families=len(pool))
                family=generate_family(index,settings,Native(),evaluator);event("ordinary_selfplay_collection_precommit",completed=index,target=settings["families"])
                work.admit(len(config.canonical(family))*3+1024**2)
                with work.db:work.db.execute("INSERT INTO families VALUES(?,?,?)",(index,config.canonical(family),config.digest(family)))
                pool[index]=family;event("ordinary_selfplay_collection_committed",completed=index+1,target=settings["families"])
        roots,selection,protection=select(pool,settings,read(REGISTRY));body=collection_body(identity,pool,roots,selection,protection);saved=work.get("collection_complete")
        if saved is None:
            require(not table(work.db,"roots","state"),"Orphan ordinary selected roots")
            with work.db:
                for key,row in roots.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
                work.put("collection_complete",body)
        else:require(equal(saved,body) and equal(table(work.db,"roots","state"),roots),"Closed ordinary collection marker differs")
        validate_collection(work.db,identity,path);publish(path/"rollout-protection.json",protection);publish(path/"collection-manifest.json",body)
        if not fixture:publish(PROTECTION,protection);publish(COLLECTION,body)
        event("ordinary_selfplay_collection_complete",families=len(pool),selected_roots=len(roots),selection=selection);return body
    finally:work.close()


def load_rows(path=DEFAULT):
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");identity=metadata(db,"identity");roots,body=validate_collection(db,identity,path)
        require((path/"collection-manifest.json").read_text(encoding="utf-8")==config.canonical(body)+"\n" and (path/"rollout-protection.json").read_text(encoding="utf-8")==config.canonical(body["rollout_protection"])+"\n","Collection publication/protection differs");return roots,body


def starts_for(roots,identity):
    if identity["fixture"]:
        prior=db_rows("data/midgame-data-pilot-v1/work.sqlite3","roots");chosen=[];used=set()
        for kind in STRATA:
            row=next(row for key,row in sorted(prior.items()) if row["cohort"]=="training_candidate" and row["stratum"]==kind and row["family"] not in used);chosen.append(row);used.add(row["family"])
    else:chosen=[row for key,row in sorted(roots.items()) if row["cohort"]=="training_candidate"]
    return [dict(slot=index,root=row,root_sha256=config.digest(row)) for index,row in enumerate(chosen)]


def rollout_identity(path,identity,roots,collection):
    return dict(schema=SCHEMA,stage="rollouts",source_identity_sha256=config.digest(identity),collection_file_sha256=sha(Path(path)/"collection-manifest.json"),collection_semantic_sha256=config.digest(collection),protection_file_sha256=sha(Path(path)/"rollout-protection.json"),protection_semantic_sha256=config.digest(collection["rollout_protection"]),starts=starts_for(roots,identity),fixture_boundary="Twoexplicitprioracceptedtrainingmetadata roots" if identity["fixture"] else "Onlyfreshselectedtrainingroots")


def replay_body(db,identity,roots,collection,closed=False):
    native=Native();settings=identity["settings"];roll=settings["rollouts"];starts=starts_for(roots,identity);games=table(db,"sp_games");expected={}
    start_by_family={start["root"]["family"]:start for start in starts}
    for start in starts:
        for index in range(roll["runs_per_family"]):expected[previous.initial_game(start,index,roll,native)["id"]]=(start,index)
    require(set(games)==set(expected),"Ordinary rollout panel differs")
    decisions={key:previous.validate_game(db,game,*expected[key],roll,native) for key,game in games.items()};require(db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0]==sum(map(len,decisions.values())),"Orphan ordinary rollout decisions")
    if closed:require(all(game["status"]!="active" for game in games.values()),"Ordinary rollout panel unfinished")
    protect=collection["rollout_protection"];pq,pf=set(protect["quotients"]),set(protect["families"]);byfamily=defaultdict(list);excluded={};eligible={};skips=Counter()
    for game in games.values():byfamily[game["family"]].append(game)
    for family,entries in byfamily.items():
        reasons=[]
        if any(game["status"]!="terminal" for game in entries):reasons.append("both_trajectories_not_terminal")
        if family in pf:reasons.append("protected_family")
        for game in entries:
            for decision,digest in decisions[game["id"]]:
                p=decision["state"];active=sum(map(sum,p["pits"]));key=(game["id"],decision["choice"])
                if active<=4:skips["exact_domain_le4"]+=1;continue
                if active<25:skips["late5_24"]+=1;continue
                _,mechanism=pilot._mechanism(p,native);legal=native.normalize(flatten(p),actor=p["controller"],total=98)["legal_actions"]
                if not mechanism.endswith("/ordinary"):skips["nonordinary_ge25"]+=1;continue
                if len(legal)<2:skips["forced_ordinary_ge25"]+=1;continue
                eligible[key]=(decision,digest,mechanism,legal)
                if pilot.quotient(p).hex() in pq:reasons.append("protected_eligible_ordinary_quotient_bridge")
        if reasons:excluded[family]=sorted(set(reasons))
    candidates=[]
    for (gameid,choice),(decision,digest,mechanism,legal) in eligible.items():
        game=games[gameid]
        if game["family"] in excluded:continue
        p=decision["state"];x,d,r=features_relay.encode(p,127,native);y=game["terminal_margin"];require(type(y) is int and y%2==0 and d-abs(r)<=y<=d+abs(r),"Ordinary terminal return interval differs")
        start=start_by_family[game["family"]]
        candidates.append(dict(id=gameid+"/"+f"{choice:03d}",gameid=gameid,choice=choice,is_start_record=choice==0,family=game["family"],selected_start_state_key=start["root"]["state_key"],selected_start_root_sha256=start["root_sha256"],purpose="ordinary_selfplay_training_candidate",state=p,state_key=previous.rawkey(p),quotient=pilot.quotient(p).hex(),stratum=mechanism,actor=p["controller"],legal_mask=sum(1<<a for a in legal),root_visits=decision["search"]["visits"],policy_target=decision["policy_target"],terminal_return_player0=y,labelkind="observed_fixed_control_selfplay_terminal_seed_margin",terminal_game_sha256=config.digest(game),source_decision_sha256=digest,features=dict(x=x,d=d,r=r)))
    grouped=defaultdict(list)
    for row in candidates:grouped[row["state_key"]].append(row)
    rows=[];pruned=[]
    for values in grouped.values():
        values.sort(key=lambda row:(row["choice"]!=0,hashlib.sha256(("ordinary-selfplay-rawcap-v1/"+row["id"]).encode()).digest()));rows.extend(values[:roll["rawstate_observation_cap"]]);pruned.extend(row["id"] for row in values[roll["rawstate_observation_cap"]:])
    rows.sort(key=lambda row:row["id"]);retained={row["id"] for row in rows};usable={};counts=Counter()
    for start in starts:
        ids=[previous.initial_game(start,index,roll,native)["id"]+"/000" for index in range(2)];ok=all(key in retained for key in ids);usable[start["root"]["state_key"]]=dict(family=start["root"]["family"],stratum=start["root"]["stratum"],usable=ok,required_record_ids=ids,root_sha256=start["root_sha256"])
        if ok:counts[start["root"]["stratum"]]+=1
    support={kind:dict(records=sum(r["stratum"]==kind for r in rows),unique_rawstates=len({r["state_key"] for r in rows if r["stratum"]==kind}),families=len({r["family"] for r in rows if r["stratum"]==kind}),start_records=sum(r["stratum"]==kind and r["choice"]==0 for r in rows),descendant_records=sum(r["stratum"]==kind and r["choice"]>0 for r in rows),descendant_unique_rawstates=len({r["state_key"] for r in rows if r["stratum"]==kind and r["choice"]>0})) for kind in sorted({r["stratum"] for r in rows}|set(STRATA))}
    for kind,value in support.items():value["descendant_novel_vs_selected_start_rawstates"]=len({r["state_key"] for r in rows if r["stratum"]==kind and r["choice"]>0}-{previous.rawkey(start["root"]["state"]) for start in starts})
    valcounts=collection["selection"]["counts"]["validation"];valcomplete=all(valcounts[kind]==settings["caps"]["validation"][kind] for kind in STRATA);gate=all(counts[kind]>=settings["minimum_usable_roots"][kind] for kind in STRATA)
    summary=dict(games=len(games),terminal_games=sum(g["status"]=="terminal" for g in games.values()),capped_games=sum(g["status"]=="capped" for g in games.values()),committed_decisions=sum(map(len,decisions.values())),selected_training_starts=len(starts),selected_validation_roots=sum(valcounts.values()),validation_panel_complete=valcomplete,future_use_ready=gate and valcomplete,usable_start_gate_passed=gate,usable_start_roots={kind:counts[kind] for kind in STRATA},usable_starts=usable,retained_rows=len(rows),unique_rawstates=len(grouped),retained_families=len({r["family"] for r in rows}),eligible_ordinary_observations=len(eligible),precap_admitted_observations=len(candidates),pruned_duplicate_rows=len(pruned),pruned_row_ids=sorted(pruned),skipped_rows=dict(skips),family_exclusions=excluded,support=support,new_validation_pool_candidate_observations=protect["new_validation_candidate_observations"],new_validation_pool_unique_quotients=protect["new_validation_unique_quotients"],new_validation_pool_families=len(protect["new_validation_family_ids"]),fixture_rollout_panel=identity["fixture"],training_performed=False,default_promotion=False)
    return dict(schema=SCHEMA,stage="rollouts",identity=identity,rollout_identity=metadata(db,"rollout_identity"),status="complete" if closed else "in_progress",collection_sha256=config.digest(collection),starts=starts,summary=summary,games_sha256=config.digest(games),decision_sha256=config.digest([[key,[[row["choice"],digest] for row,digest in values]] for key,values in decisions.items()]),rows_sha256=config.digest(rows),teacher_searches=False,training_performed=False,new_strength_games=False,sealed_final_test_consumed=False),rows


def generate(path=DEFAULT,fixture=False,families=None,on_event=None,cancel=None,decision_limit=None):
    path=Path(path).resolve();roots,collection=load_rows(path);identity=collection["identity"];require(identity["fixture"] is fixture and equal(identity["settings"],settings_for(fixture,families)),"Rollout invocation contract differs");require("torch" not in sys.modules,"Rollouts must not load Torch");work=phase6_store.Work(path,identity);native=Native();roll=identity["settings"]["rollouts"]
    event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS sp_games(id TEXT PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS sp_decisions(gameid TEXT,choice INTEGER,body TEXT,hash TEXT,PRIMARY KEY(gameid,choice));CREATE TABLE IF NOT EXISTS sp_rows(id TEXT PRIMARY KEY,body TEXT,hash TEXT);")
        rollout=rollout_identity(path,identity,roots,collection);saved_identity=work.get("rollout_identity")
        if saved_identity is None:
            require(not table(work.db,"sp_games") and work.db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0]==0,"Orphan rollout identity")
            with work.db:work.put("rollout_identity",rollout)
        else:require(equal(saved_identity,rollout),"Rollout source/collection/protection identity differs")
        if not table(work.db,"sp_games"):
            require(work.db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0]==0,"Orphan ordinary decisions")
            with work.db:
                for start in starts_for(roots,identity):
                    for index in range(2):
                        game=previous.initial_game(start,index,roll,native);work.db.execute("INSERT INTO sp_games VALUES(?,?,?)",(game["id"],config.canonical(game),config.digest(game)))
        replay_body(work.db,identity,roots,collection);completed=work.get("replay_complete");count=work.db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0];new=0
        require(completed is not None or not table(work.db,"sp_rows"),"Orphan ordinary replay rows")
        if completed is None:
            with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
                while True:
                    active=[g for g in table(work.db,"sp_games").values() if g["status"]=="active"]
                    if not active:break
                    if (cancel and cancel()) or (decision_limit is not None and new>=decision_limit):return dict(status="interrupted",committed_decisions=count)
                    game=active[0];rng=previous.rng_restore(game["rng"]);before=rng.getstate();p=game["position"]
                    report=previous.quality_search.search(p,evaluator,simulations=roll["simulations"],time_ms=0,max_nodes=roll["max_nodes"],rng=rng,root_noise=roll["root_noise"],cancel=cancel,c_puct=roll["c_puct"],native=native,memory_mb=roll["memory_mb"])
                    if report["completed_simulations"]!=roll["simulations"] or report["stop_reason"]!="simulations":return dict(status="interrupted",committed_decisions=count)
                    legal=native.normalize(flatten(p),actor=p["controller"],total=98)["legal_actions"];action=previous.behavior_action(report,legal,rng,game["choices"],roll);transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Ordinary rollout action illegal");after=previous.normalize(transition["position"],native)
                    row=dict(gameid=game["id"],choice=game["choices"],state=p,action=action,after=after,rng_before=before,rng_after=rng.getstate(),policy_target=report["policy"],search=report);terminal=not sum(map(sum,after["pits"]));updated={**game,"position":after,"rng":rng.getstate(),"choices":game["choices"]+1,"status":"terminal" if terminal else "capped" if game["choices"]+1>=roll["max_choices"] else "active","terminal_margin":after["stores"][0]-after["stores"][1] if terminal else None}
                    event("ordinary_selfplay_decision_precommit",committed_decisions=count,gameid=game["id"],choice=game["choices"]);work.admit((len(config.canonical(row))+len(config.canonical(updated)))*3)
                    with work.db:work.db.execute("INSERT INTO sp_decisions VALUES(?,?,?,?)",(game["id"],game["choices"],config.canonical(row),config.digest(row)));work.db.execute("UPDATE sp_games SET body=?,hash=? WHERE id=?",(config.canonical(updated),config.digest(updated),game["id"]))
                    count+=1;new+=1;event("ordinary_selfplay_decision_committed",committed_decisions=count,gameid=game["id"],choice=updated["choices"])
            body,rows=replay_body(work.db,identity,roots,collection,True)
            with work.db:
                for row in rows:work.db.execute("INSERT INTO sp_rows VALUES(?,?,?)",(row["id"],config.canonical(row),config.digest(row)))
                work.put("replay_complete",body)
            completed=body
        body,rows=replay_body(work.db,identity,roots,collection,True);require(equal(completed,body) and equal(list(table(work.db,"sp_rows").values()),rows),"Ordinary completed replay marker/rows differ")
        publish(path/"replay-manifest.json",body)
        if not fixture:publish(OUTPUT,body)
        event("ordinary_selfplay_complete",summary=body["summary"]);return body
    finally:work.close()


def load_replay(path=DEFAULT):
    path=Path(path).resolve();roots,collection=load_rows(path);identity=collection["identity"]
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");require(equal(metadata(db,"rollout_identity"),rollout_identity(path,identity,roots,collection)),"Closed rollout collection/protection identity differs");body,rows=replay_body(db,identity,roots,collection,True);require(equal(metadata(db,"replay_complete"),body) and equal(list(table(db,"sp_rows").values()),rows),"Closed ordinary replay differs")
        require((path/"replay-manifest.json").read_text(encoding="utf-8")==config.canonical(body)+"\n","Ordinary replay publication differs");return body,rows


def run(path=DEFAULT,fixture=False,families=None,stage="all",on_event=None,cancel=None):
    require(stage in ("all","collection","rollouts"),"Unknown ordinary stage")
    if stage in ("all","collection"):
        result=collect(path,fixture,families,on_event,cancel)
        if result["status"]!="complete" or stage=="collection":return result
    return generate(path,fixture,families,on_event,cancel)
