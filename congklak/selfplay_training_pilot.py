"""One source-frozen CPU self-play generation and small joint update."""
from collections import Counter,defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import sys
import time
from . import config,pilot,features_relay,quality_search,quality_learning,value_pilot_learning,phase6_store
from .native import ROOT,Native,flatten
from .neural_relay import Model,flatten_weights
from .puct_cpu_inference import CpuBatchEvaluator

SCHEMA="current-control-native-selfplay-joint-pilot-v2"
DEFAULT=ROOT/"data/selfplay-training-pilot-v1"
BASELINE=ROOT/"build/selfplay-pilot-baseline.json"
REGISTRY=ROOT/"build/selfplay-training-pilot-registry.json"
PLAN=ROOT/"build/selfplay-training-pilot-plan-v2.json"
IMPLEMENTATION=ROOT/"build/selfplay-training-pilot-implementation.json"
OUTPUT=ROOT/"build/selfplay-training-pilot-selfplay.json"
WITNESS=ROOT/"build/selfplay-training-pilot-model-witness.json"
PROJECTION=ROOT/"build/selfplay-pilot-benchmark-projection.json"
SOURCE=ROOT/"data/quality-train-v1/model.json"
SOURCE_SHA="bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558"
SOURCE_WEIGHTS="1efd42cf39e64183752ad3bcbf3ba1b82e7e5fdc4c273426c763d4e20d746649"
SOURCES=("congklak/selfplay_training_pilot.py","tools/selfplay-training-pilot.py","tests/selfplay_training_pilot_author.py","docs/SELFPLAY_TRAINING_PILOT_SCOPE.md","docs/SELFPLAY_PILOT_SCOPE.md")
DEPENDENCIES=("congklak/quality_search.py","congklak/puct_cpu_inference.py","congklak/quality_learning.py","congklak/value_pilot_learning.py","congklak/training_v3.py","congklak/phase6_store.py","congklak/features_relay.py","congklak/config.py","congklak/selfplay_pilot_benchmark.py","build/selfplay-pilot-benchmark-plan.json","build/selfplay-pilot-benchmark-implementation.json","build/selfplay-pilot-benchmark-projection.json","build/training-runtime.json")
SP=dict(families=16,runs_per_family=2,seed=20261025,simulations=64,max_nodes=2048,max_choices=512,c_puct=.25,memory_mb=8192,root_noise=dict(alpha=.3,fraction=.25),temperature_first_choices=8,rawstate_observation_cap=4,minimum_admitted_families=8,min_training_active=25,
        quotas={"50-74/ordinary":4,"25-49/ordinary":4,"25-49/bonus":4,"25-49/capture":4},
        orientation="Literal full-state fixed-player0; actor normalized after each rule transition; no alternating-sign shortcut",
        admission="Both trajectories terminal; skip terminal/active<25 before checking unchanged trueevaluation protected bridges; reject entire family; no replacement",
        policy_target="Normalized raw root visits, independent of behavior temperature",replay_sampling="Uniform admitted family then uniform retained observation, unit loss; max4 rawstate records by deterministic hash; individual returns not averaged")
TRAIN=dict(steps=200,batch_size=128,lr=.0001,seed=20261025,checkpoint_every=20,input_count=127,width=64,parameters=12872,device="cpu",cpu_threads=1,
           optimizer="Fresh AdamW/all8/fp32/wd.0001/foreachFalse/clip1",value_huber_delta=.1,policy_loss_weight=.25,
           objective="Huber(tanh_actor_residual,(terminal_player0-D)/signed_active,delta.1)+.25*CE(raw_visits)",
           selection="Fixed final200; no checkpoint selection",teacher_rehearsal=False,amp=False,loader_workers=0)
CONTRACT=dict(selfplay=SP,training=TRAIN,automatic_expansion=False,sealed_final_test_consumed=False,default_promotion=False)
require,sha,publish=value_pilot_learning.require,value_pilot_learning.sha,value_pilot_learning.publish
runtime,evaluator=quality_learning.runtime,quality_learning.evaluator


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def tuples(value):return tuple(tuples(item) for item in value) if isinstance(value,list) else value


def rng_restore(value):
    rng=random.Random();rng.setstate(tuples(value));return rng


def rawkey(p):return bytes(flatten(p)+[p["controller"]]).hex()


def normalize(p,native):return native.normalize(flatten(p),actor=p["controller"],total=98)["position"]


def checked_baseline():
    value=read(BASELINE);require(value["schema"]=="selfplay-pilot-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==2839 and value["fixed_model_sha256"]==sha(SOURCE)==SOURCE_SHA,"Selfplay baseline/model differs")
    for name,digest in value["inputs"].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==digest,"Sealed source changed: "+name)
    return value


def table(db,name,key="id"):
    values={}
    for index,body,digest in db.execute("SELECT "+key+",body,hash FROM "+name+" ORDER BY "+key):
        value=json.loads(body);require(config.digest(value)==digest,"Corrupt selfplay table: "+name);values[index]=value
    return values


def choose_starts(rows,quotients,families):
    candidates={kind:sorted((row for row in rows if row["stratum"]==kind and row["family"] not in families and row["quotient"] not in quotients),key=lambda row:hashlib.sha256(("selfplay-pilot-start-v1/"+row["state_key"]).encode()).digest()) for kind in SP["quotas"]}
    slots=[kind for kind,count in SP["quotas"].items() for _ in range(count)];assigned={};owned={}
    def place(slot,seen):
        for row in candidates[slots[slot]]:
            family=row["family"]
            if family in seen:continue
            seen.add(family)
            if family not in owned or place(owned[family],seen):
                assigned[slot]=row;owned[family]=slot;return True
        return False
    for slot in range(len(slots)):place(slot,set())
    starts=[dict(slot=slot,root=assigned[slot],root_sha256=config.digest(assigned[slot])) for slot in sorted(assigned)]
    require(len({entry["root"]["family"] for entry in starts})==len(starts),"Start family matching repeated a family")
    counts=Counter(entry["root"]["stratum"] for entry in starts)
    return starts,dict(counts={kind:counts[kind] for kind in SP["quotas"]},shortfall={kind:count-counts[kind] for kind,count in SP["quotas"].items()},eligible_by_stratum={kind:len(values) for kind,values in candidates.items()})


def make_registry():
    checked_baseline();projection=read(PROJECTION);require(projection["schema"]=="selfplay-pilot-benchmark-projection-v1" and projection["sha256"]==config.digest({key:value for key,value in projection.items() if key!="sha256"}),"Benchmark reservation differs")
    true=read(ROOT/"build/value-pilot-data.json");previous=read(ROOT/"build/midgame-data-pilot-registry.json")
    quotients=set(true["protected_quotients"]);families=set(true["protected_families"])
    quotients.update(previous["latest_arena_quotients"]);families.update(previous["latest_arena_families"])
    for row in previous["evaluation_roots"]:
        quotients.add(row["quotient"])
        if row["family"]:families.add(row["family"])
    for row in previous["historical_roles"].values():
        if row["cohort"]!="training_candidate" or row["latest_eval_quotient_collision"] or row["latest_eval_family_collision"]:quotients.add(row["quotient"]);families.add(row["family"])
    with closing(sqlite3.connect((ROOT/"data/midgame-data-pilot-v1/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:recent=table(db,"roots","state")
    require(Counter(row["cohort"] for row in recent.values())==dict(training_candidate=39,validation=10),"Latest fresh roles differ")
    for row in [*recent.values(),*previous["diagnostics"].values()]:
        if row["cohort"]!="training_candidate":quotients.add(row["quotient"]);families.add(row["family"])
    quotients.update(projection["reserved_quotients"]);families.update(projection["reserved_families"])
    # These9000 entries reserve collection novelty, not automatic evaluation ownership.
    metadata=[]
    with closing(sqlite3.connect((ROOT/"data/deeper-9000-v2/work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for key,state,family,split in db.execute("SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family'),json_extract(body,'$.split') FROM roots ORDER BY state"):
            q=pilot.quotient(json.loads(state)).hex();metadata.append([key,q,family,split])
            if split!="train":quotients.add(q);families.add(family)
    require(len(metadata)==9000,"Original metadata reservation count differs")
    starts,selection=choose_starts([row for row in recent.values() if row["cohort"]=="training_candidate"],quotients,families)
    value=dict(schema="selfplay-pilot-role-aware-registry-v1",rules_digest=config.RULES_DIGEST,quotients=sorted(quotients),families=sorted(families),starts=starts,start_selection=selection,
               metadata_only_reservations=dict(rows=9000,semantic_sha256=config.digest(metadata),blanket_evaluation_ban=False,train_rows=sum(row[3]=="train" for row in metadata)),
               latest_roles={key:dict(family=row["family"],cohort=row["cohort"],quotient=row["quotient"],root_sha256=config.digest(row)) for key,row in sorted(recent.items())},
               historical_roles=previous["historical_roles"],diagnostics={key:dict(family=row["family"],cohort="diagnostic",quotient=row["quotient"]) for key,row in previous["diagnostics"].items()},
               projection_sha256=sha(PROJECTION),baseline_sha256=sha(BASELINE),ancestral_exposure_not_removed=True,sealed_target_bodies_read=False)
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def plan():
    registry=make_registry();value=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,source_model_sha256=SOURCE_SHA,source_weights_sha256=SOURCE_WEIGHTS,registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),projection_sha256=sha(PROJECTION),start_selection=registry["start_selection"],starts=registry["starts"])
    publish(PLAN,value);return value


def freeze():
    plan();value=dict(schema="selfplay-training-pilot-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES})
    publish(IMPLEMENTATION,value);return value


def admit():
    checked_baseline();value,implementation=read(PLAN),read(IMPLEMENTATION)
    require(value["schema"]==SCHEMA and config.canonical(value["contract"])==config.canonical(CONTRACT) and value["rules_digest"]==config.RULES_DIGEST and value["source_model_sha256"]==SOURCE_SHA,"Selfplay fixed plan differs")
    require(implementation==dict(schema="selfplay-training-pilot-implementation-v1",plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES}),"Selfplay producer freeze differs")
    return value,implementation


def gate():
    from .selfplay_pilot_benchmark import pretraining_witness
    value=pretraining_witness();require(value["passed"] is True,"Selfplay benchmark source gate incomplete");return value


def settings_for(fixture=False,families=None):
    require(type(fixture) is bool,"Fixture flag must be a boolean")
    settings=json.loads(config.canonical(SP))
    if fixture:
        settings["families"]=16 if families is None else config.integer(families,"fixture_families",1,16);settings["simulations"]=4;settings["minimum_admitted_families"]=1
    else:require(families is None,"Production family budget differs")
    return settings


def sp_identity(settings,fixture):
    registry=read(REGISTRY);native=Native()
    require(len(registry["starts"])==SP["families"] and all(registry["start_selection"]["counts"][kind]==count for kind,count in SP["quotas"].items()),"blocked_start_panel_shortfall: exact16 distinct families required; no replacement")
    require(len(registry["starts"][:settings["families"]])==settings["families"],"Fixture requested more starts than frozen panel")
    return dict(schema=SCHEMA,stage="selfplay",fixture=fixture,settings=settings,starts=registry["starts"][:settings["families"]],registry_sha256=sha(REGISTRY),plan_sha256=sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),baseline_sha256=sha(BASELINE),source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/selfplay-training-pilot.py"),rules_digest=config.RULES_DIGEST,model_file_sha256=sha(SOURCE),model_weights_sha256=Model(SOURCE).id,native_build_sha256=native.build_id,search_protocol=quality_search.PROTOCOL,pretraining_witness=None if fixture else gate())


def initial_game(start,run,settings,native):
    root=start["root"];seed=settings["seed"]+start["slot"]*104729+run*1009;rng=random.Random(seed);position=normalize(root["state"],native)
    return dict(id=f"{start['slot']:03d}/{run}",family=root["family"],start_root_sha256=start["root_sha256"],start=position,position=position,seed=seed,rng=rng.getstate(),choices=0,status="active",terminal_margin=None)


def behavior_action(report,legal,rng,choice,settings):
    if choice<settings["temperature_first_choices"]:
        draw=rng.random();running=0.
        for action in legal:
            running+=report["policy"][action]
            if draw<running:return action
        return legal[-1]
    return max(legal,key=lambda action:(report["visits"][action],-action))


def validate_game(db,game,start,run,settings,native):
    expected=initial_game(start,run,settings,native);position=expected["position"];rng=rng_restore(expected["rng"]);count=0;decisions=[]
    require(set(game)==set(expected) and game["id"]==expected["id"] and game["family"]==expected["family"] and game["seed"]==expected["seed"] and game["start"]==expected["start"] and game["start_root_sha256"]==expected["start_root_sha256"],"Selfplay game identity differs")
    for choice,body,digest in db.execute("SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice",(game["id"],)):
        row=json.loads(body);require(config.digest(row)==digest and set(row)=={"gameid","choice","state","action","after","rng_before","rng_after","policy_target","search"} and choice==count==row["choice"] and row["gameid"]==game["id"] and row["state"]==position and tuples(row["rng_before"])==rng.getstate(),"Selfplay decision prefix/state/RNG differs")
        normal=native.normalize(flatten(position),actor=position["controller"],total=98);require(not normal["terminal"] and type(row["action"]) is int,"Decision after terminal or noninteger action")
        report=row["search"];quality_search.validate_report(report,position,settings["simulations"],settings["max_nodes"],settings["c_puct"],native,allow_interrupted=False)
        require(report["completed_simulations"]==settings["simulations"] and report["stop_reason"]=="simulations" and report["memory_mb"]==settings["memory_mb"] and report["forest_size"]==1 and report["timing"]["requested_time_ms"]==0 and row["policy_target"]==[value/settings["simulations"] for value in report["visits"]],"Selfplay policy target/search budget differs")
        if settings["root_noise"] is not None:
            for _ in normal["legal_actions"]:rng.gammavariate(settings["root_noise"]["alpha"],1)
        require(row["action"]==behavior_action(report,normal["legal_actions"],rng,count,settings) and tuples(row["rng_after"])==rng.getstate(),"Behavior temperature/noise RNG differs")
        transition=native.apply(flatten(position),actor=position["controller"],action=row["action"],total=98,trace=False);require(transition["ok"],"Selfplay committed action illegal");position=normalize(transition["position"],native);require(position==row["after"],"Selfplay rule transition differs");count+=1;decisions.append((row,digest))
    terminal=not sum(map(sum,position["pits"]));status="terminal" if terminal else "capped" if count>=settings["max_choices"] else "active"
    require((type(game["terminal_margin"]) is int) if terminal else game["terminal_margin"] is None,"Terminal-return type differs")
    require(type(game["choices"]) is int and game["choices"]==count<=settings["max_choices"] and game["position"]==position and tuples(game["rng"])==rng.getstate() and game["status"]==status and game["terminal_margin"]==(position["stores"][0]-position["stores"][1] if terminal else None),"Selfplay final state/status/return differs")
    return decisions


def build_replay(db,identity,closed=False):
    native=Native();settings=identity["settings"];registry=read(REGISTRY);games=table(db,"sp_games");expected={}
    for start in identity["starts"]:
        for run in range(settings["runs_per_family"]):expected[initial_game(start,run,settings,native)["id"]]=(start,run)
    require(set(games)==set(expected),"Selfplay game panel differs")
    decisions={key:validate_game(db,game,*expected[key],settings,native) for key,game in games.items()}
    require(db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0]==sum(len(rows) for rows in decisions.values()),"Orphan selfplay decision")
    if closed:require(all(game["status"]!="active" for game in games.values()),"Selfplay panel unfinished")
    protected_q,protected_f=set(registry["quotients"]),set(registry["families"]);byfamily=defaultdict(list);exclusions={};low=0;late=0;observed=0
    for game in games.values():byfamily[game["family"]].append(game)
    for family,entries in byfamily.items():
        reasons=[]
        if any(game["status"]!="terminal" for game in entries):reasons.append("both_trajectories_not_terminal")
        if family in protected_f:reasons.append("protected_family")
        for game in entries:
            for row,_ in decisions[game["id"]]:
                active=sum(map(sum,row["state"]["pits"]));observed+=1
                if active<=4:low+=1;continue
                if active<settings["min_training_active"]:late+=1;continue
                if pilot.quotient(row["state"]).hex() in protected_q:reasons.append("protected_eligible_active_ge25_quotient_bridge")
        if reasons:exclusions[family]=sorted(set(reasons))
    candidates=[]
    for game in games.values():
        if game["family"] in exclusions:continue
        for decision,digest in decisions[game["id"]]:
            p=decision["state"]
            if sum(map(sum,p["pits"]))<settings["min_training_active"]:continue
            x,d,r=features_relay.encode(p,127,native);target=game["terminal_margin"]
            require(type(target) is int and target%2==0 and d-abs(r)<=target<=d+abs(r),"Terminal return violates root seed interval")
            candidates.append(dict(id=game["id"]+"/"+f"{decision['choice']:03d}",gameid=game["id"],choice=decision["choice"],family=game["family"],purpose="selfplay_training",state=p,state_key=rawkey(p),quotient=pilot.quotient(p).hex(),actor=p["controller"],legal_mask=sum(1<<a for a in native.normalize(flatten(p),actor=p["controller"],total=98)["legal_actions"]),root_visits=decision["search"]["visits"],policy_target=decision["policy_target"],terminal_return_player0=target,labelkind="selfplay_terminal_seed_margin",terminal_game_sha256=config.digest(game),source_decision_sha256=digest,features=dict(x=x,d=d,r=r),stratum=pilot._mechanism(p,native)[1]))
    duplicate=defaultdict(list)
    for row in candidates:duplicate[row["state_key"]].append(row)
    rows=[];pruned=[]
    for key,values in sorted(duplicate.items()):
        values.sort(key=lambda row:hashlib.sha256(("selfplay-pilot-rawcap-v1/"+row["id"]).encode()).digest());rows.extend(values[:settings["rawstate_observation_cap"]]);pruned.extend(row["id"] for row in values[settings["rawstate_observation_cap"]:])
    rows.sort(key=lambda row:row["id"]);admitted_families=sorted({row["family"] for row in rows});ready=len(admitted_families)>=settings["minimum_admitted_families"]
    summary=dict(games=len(games),terminal_games=sum(game["status"]=="terminal" for game in games.values()),capped_games=sum(game["status"]=="capped" for game in games.values()),committed_decisions=observed,exact_domain_rows_skipped=low,late5_24_rows_skipped=late,optimizer_ineligible_rows=low+late,min_training_active=settings["min_training_active"],family_exclusions=exclusions,excluded_families=len(exclusions),admitted_families=len(admitted_families),admitted_family_ids=admitted_families,retained_rows=len(rows),eligible_rows_before_cap=len(candidates),pruned_duplicate_rows=len(pruned),pruned_row_ids=sorted(pruned),rawstate_unique=len({row["state_key"] for row in rows}),strata=dict(sorted(Counter(row["stratum"] for row in rows).items())),training_admission="ready" if ready else "blocked_insufficient_admitted_families",minimum_admitted_families=settings["minimum_admitted_families"],terminal_return_labels=True,visit_policy_labels=True,targets_averaged=False)
    manifest=dict(schema=SCHEMA,stage="selfplay",identity=identity,status="complete" if closed else "in_progress",summary=summary,games_sha256=config.digest(games),decisions_sha256=config.digest([[key,[[row["choice"],digest] for row,digest in values]] for key,values in decisions.items()]),rows_sha256=config.digest(rows),teacher_searches=False,training_performed=False,sealed_final_test_consumed=False)
    return manifest,rows


def generate(path=DEFAULT,fixture=False,families=None,on_event=None,cancel=None,decision_limit=None):
    path=Path(path).resolve();settings=settings_for(fixture,families)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Selfplay fixture must remain in build")
    else:require(path==DEFAULT.resolve(),"Selfplay production path differs");admit()
    require("torch" not in sys.modules,"Selfplay must remain CPU/noTorch");identity=sp_identity(settings,fixture);work=phase6_store.Work(path,identity);native=Native()
    def event(name,**value):
        if on_event:on_event(dict(event=name,**value))
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS sp_games(id TEXT PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS sp_decisions(gameid TEXT,choice INTEGER,body TEXT,hash TEXT,PRIMARY KEY(gameid,choice));CREATE TABLE IF NOT EXISTS sp_rows(id TEXT PRIMARY KEY,body TEXT,hash TEXT);")
        if not table(work.db,"sp_games"):
            require(work.db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0]==0,"Orphan selfplay decisions")
            with work.db:
                for start in identity["starts"]:
                    for run in range(settings["runs_per_family"]):
                        game=initial_game(start,run,settings,native);work.db.execute("INSERT INTO sp_games VALUES(?,?,?)",(game["id"],config.canonical(game),config.digest(game)))
        build_replay(work.db,identity);completed=work.get("selfplay_manifest");committed=work.db.execute("SELECT COUNT(*) FROM sp_decisions").fetchone()[0];new=0
        require(completed is not None or not table(work.db,"sp_rows"),"Orphan uncommitted replay rows")
        if completed is None:
            with CpuBatchEvaluator(SOURCE,cache_capacity=0,max_batch=1) as evaluator:
                while True:
                    active=[game for game in table(work.db,"sp_games").values() if game["status"]=="active"]
                    if not active:break
                    if (cancel and cancel()) or (decision_limit is not None and new>=decision_limit):return dict(status="interrupted",committed_decisions=committed)
                    game=active[0];rng=rng_restore(game["rng"]);before=rng.getstate();p=game["position"]
                    report=quality_search.search(p,evaluator,simulations=settings["simulations"],time_ms=0,max_nodes=settings["max_nodes"],rng=rng,root_noise=settings["root_noise"],cancel=cancel,c_puct=settings["c_puct"],native=native,memory_mb=settings["memory_mb"])
                    if report["completed_simulations"]!=settings["simulations"] or report["stop_reason"]!="simulations":return dict(status="interrupted",committed_decisions=committed,reason=report["stop_reason"])
                    legal=native.normalize(flatten(p),actor=p["controller"],total=98)["legal_actions"];action=behavior_action(report,legal,rng,game["choices"],settings);transition=native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False);require(transition["ok"],"Generated selfplay action illegal");after=normalize(transition["position"],native)
                    row=dict(gameid=game["id"],choice=game["choices"],state=p,action=action,after=after,rng_before=before,rng_after=rng.getstate(),policy_target=report["policy"],search=report)
                    terminal=not sum(map(sum,after["pits"]));updated={**game,"position":after,"rng":rng.getstate(),"choices":game["choices"]+1,"status":"terminal" if terminal else "capped" if game["choices"]+1>=settings["max_choices"] else "active","terminal_margin":after["stores"][0]-after["stores"][1] if terminal else None}
                    event("selfplay_pilot_decision_precommit",committed_decisions=committed,gameid=game["id"],choice=game["choices"]);work.admit((len(config.canonical(row))+len(config.canonical(updated)))*3)
                    with work.db:
                        work.db.execute("INSERT INTO sp_decisions VALUES(?,?,?,?)",(row["gameid"],row["choice"],config.canonical(row),config.digest(row)));work.db.execute("UPDATE sp_games SET body=?,hash=? WHERE id=?",(config.canonical(updated),config.digest(updated),updated["id"]))
                    committed+=1;new+=1;event("selfplay_pilot_decision_committed",committed_decisions=committed,gameid=game["id"],choice=updated["choices"],terminal_games=sum(entry["status"]=="terminal" for entry in table(work.db,"sp_games").values()),games=len(identity["starts"])*2)
            manifest,rows=build_replay(work.db,identity,closed=True);event("selfplay_pilot_replay_precommit",committed_decisions=committed,rows=len(rows))
            with work.db:
                require(not table(work.db,"sp_rows"),"Orphan selfplay replay rows")
                for row in rows:work.db.execute("INSERT INTO sp_rows VALUES(?,?,?)",(row["id"],config.canonical(row),config.digest(row)))
                work.put("selfplay_manifest",manifest)
            event("selfplay_pilot_replay_committed",committed_decisions=committed,rows=len(rows));completed=manifest
        manifest,rows=build_replay(work.db,identity,closed=True);require(completed==manifest and list(table(work.db,"sp_rows").values())==rows,"Completed selfplay marker/rows differ")
        publish(path/"selfplay-manifest.json",manifest)
        if not fixture:publish(OUTPUT,manifest)
        event("selfplay_pilot_complete",**manifest["summary"]);return manifest
    finally:work.close()


def load_replay(path=DEFAULT):
    path=Path(path).resolve();manifest=read(path/"selfplay-manifest.json");identity=manifest["identity"];fixture=identity["fixture"];settings=identity["settings"]
    require(type(fixture) is bool and config.canonical(settings)==config.canonical(settings_for(fixture,settings["families"] if fixture else None)) and config.canonical(identity)==config.canonical(sp_identity(settings,fixture)),"Closed selfplay identity differs")
    if fixture:require(path.is_relative_to(ROOT/"build"),"Closed fixture path differs")
    else:require(path==DEFAULT.resolve(),"Closed production path differs");admit()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();require(json.loads(body)==identity and digest==config.digest(identity),"Selfplay catalog identity differs")
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='selfplay_manifest'").fetchone();require(json.loads(body)==manifest and digest==config.digest(manifest),"Selfplay committed manifest differs")
        expected,rows=build_replay(db,identity,closed=True);require(expected==manifest and list(table(db,"sp_rows").values())==rows,"Selfplay completed replay differs")
    return manifest,rows


def train_settings(fixture=False,steps=None,checkpoint_every=None,batch_size=None):
    require(type(fixture) is bool,"Training fixture flag must be a boolean")
    value=dict(TRAIN)
    if fixture:value.update(steps=6 if steps is None else config.integer(steps,"fixture_steps",1,12),checkpoint_every=2 if checkpoint_every is None else config.integer(checkpoint_every,"fixture_cadence",1,12),batch_size=32 if batch_size is None else config.integer(batch_size,"fixture_batch",2,128))
    else:require(steps is None and checkpoint_every is None and batch_size is None,"Production training budget differs")
    return value


def sample_groups(rows):
    groups=defaultdict(list)
    for index,row in enumerate(rows):groups[row["family"]].append(index)
    return dict(sorted(groups.items()))


def sample_ids(rng,groups,count):
    names=sorted(groups);return [rng.choice(groups[rng.choice(names)]) for _ in range(count)]


def train_identity(settings,fixture,replay_path,manifest,rows):
    return dict(schema=SCHEMA,stage="training",**settings,fixture=fixture,rules_digest=config.RULES_DIGEST,encoder=features_relay.ENCODER127,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/selfplay-training-pilot.py"),source_dependencies={name:sha(ROOT/name) for name in DEPENDENCIES},source_model_sha256=SOURCE_SHA,source_weights_sha256=SOURCE_WEIGHTS,replay_path=Path(replay_path).resolve().relative_to(ROOT).as_posix(),replay_journal_sha256=sha(Path(replay_path)/"work.sqlite3"),replay_manifest_sha256=sha(Path(replay_path)/"selfplay-manifest.json"),replay_rows_sha256=manifest["rows_sha256"],sampler_groups=sample_groups(rows),plan_sha256=sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),pretraining_witness=None if fixture else gate(),torch_version=read(ROOT/"build/training-runtime.json")["torch"],python_version=platform.python_version())


def tensors(rows):
    torch,_=runtime();normal=Native();values=[]
    for row in rows:
        require(row["labelkind"]=="selfplay_terminal_seed_margin" and row["purpose"]=="selfplay_training","Wrong replay target kind")
        p=row["state"];feature=row["features"];legal=[bool(row["legal_mask"]&(1<<a)) for a in range(7)];probability=row["policy_target"];z=(row["terminal_return_player0"]-feature["d"])/feature["r"]
        require(any(legal) and all(type(value) in (int,float) and math.isfinite(value) and value>=0 and (legal[a] or value==0) for a,value in enumerate(probability)) and abs(sum(probability)-1)<1e-12 and -1<=z<=1,"Invalid replay probability/residual")
        require(probability==[value/SP["simulations"] for value in row["root_visits"]] or probability==[value/4 for value in row["root_visits"]],"Replay policy not raw root visits")
        values.append((feature,legal,probability,z,row["terminal_return_player0"]))
    return dict(x=torch.tensor([value[0]["x"] for value in values],dtype=torch.float32),d=torch.tensor([value[0]["d"] for value in values],dtype=torch.float32),r=torch.tensor([value[0]["r"] for value in values],dtype=torch.float32),legal=torch.tensor([value[1] for value in values],dtype=torch.bool),probability=torch.tensor([value[2] for value in values],dtype=torch.float32),z=torch.tensor([value[3] for value in values],dtype=torch.float32),y=torch.tensor([value[4] for value in values],dtype=torch.float32))


def training_metrics(model,t):
    torch,_=runtime()
    with torch.no_grad():
        z,logits=model.both(t["x"]);prediction=t["d"]+t["r"]*z;ce=quality_learning.policy_losses(logits,t["probability"],t["legal"])
        huber=torch.nn.functional.huber_loss(z,t["z"],delta=.1)
    values=dict(rows=len(t["x"]),mae_seeds=(prediction-t["y"]).abs().mean().item(),residual_huber_delta_point1=huber.item(),visit_policy_cross_entropy=ce.mean().item(),scope="In-sample admitted selfplay targets; no checkpoint selection")
    require(all(type(values[name]) in (int,float) and math.isfinite(values[name]) and values[name]>=0 for name in ("mae_seeds","residual_huber_delta_point1","visit_policy_cross_entropy")),"Nonfinite training metrics")
    return values


def validate_metrics(value,count):
    require(set(value)=={"rows","mae_seeds","residual_huber_delta_point1","visit_policy_cross_entropy","scope"} and type(value["rows"]) is int and value["rows"]==count and value["scope"]=="In-sample admitted selfplay targets; no checkpoint selection","Training metric fields/count differ")
    require(all(type(value[name]) in (int,float) and math.isfinite(value[name]) and value[name]>=0 for name in ("mae_seeds","residual_huber_delta_point1","visit_policy_cross_entropy")),"Training metric values differ")


def export_payload(state,identity,step):
    _,prior=runtime();return dict(schema="congklak-mlp-export-v4",encoder=features_relay.ENCODER127,identity=identity,step=step,status="experimental_selfplay_joint_candidate",weights={key:value.detach().cpu().tolist() for key,value in state.items()},weights_sha256=prior.weight_digest(state),terminal="store0-minus-store1",output="D_player0 + signed_active*tanh(current-actor-residual)",uses_oracle=False,policy_output="7current-root-actor logits; mask legal actions",policy_training="Source-control selfplay root raw visits/64, behavior temperature separate",value_training="Source-control observed terminal player0 margins, signed-active residual Huber.1; no teacher rehearsal")


def expected_torch_rng(identity):
    torch,_=runtime()
    with torch.random.fork_rng(devices=[]):torch.manual_seed(identity["seed"]);evaluator();return torch.get_rng_state().clone()


def validate_state(state,identity,model):
    torch,_=runtime();digest=quality_learning.validate_state(state,identity,model)
    require(identity["device"]=="cpu" and not state["cuda_rng"],"Training state must be CPU only")
    require(state["optimizer"]["param_groups"][0]["params"]==list(range(8)) and all(state["optimizer"]["state"][i]["step"].dtype==torch.float32 and state["optimizer"]["state"][i]["step"].ndim==0 for i in range(8)),"Optimizer8 ownership/step tensor differs")
    rng=random.Random(identity["seed"]);sample_ids(rng,identity["sampler_groups"],state["draws"])
    require(rng.getstate()==state["sampler_rng"] and random.Random(identity["seed"]).getstate()==state["python_rng"] and torch.equal(expected_torch_rng(identity),state["torch_rng"]),"Training sampler/Python/Torch RNG replay differs")
    return digest


def cpu_catalog(path,identity,row_count,final=False,tensor_model=None):
    entries=[];last=None;path=Path(path).resolve()
    with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");configs=db.execute("SELECT body,hash FROM config").fetchall();require(len(configs)==1 and json.loads(configs[0][0])==identity and configs[0][1]==config.digest(identity),"Training catalog identity differs")
        seen=[]
        for step,name,checksum,body,digest in db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id"):
            require(type(step) is int and 0<step<=identity["steps"] and (not seen or step>seen[-1]),"Checkpoint ordering differs");file=(path/name).resolve();require(file.parent==path and file.is_file() and sha(file)==checksum,"Checkpoint file differs")
            meta=json.loads(body);require(set(meta)=={"step","identity_sha256","weights_sha256","metrics"} and type(meta["step"]) is int and meta["step"]==step and meta["identity_sha256"]==config.digest(identity) and config.digest(meta)==digest,"Checkpoint metadata differs");validate_metrics(meta["metrics"],row_count)
            history=db.execute("SELECT body,hash FROM history WHERE step=?",(step,)).fetchall();require(len(history)==1 and json.loads(history[0][0])==meta["metrics"] and history[0][1]==config.digest(meta["metrics"]),"Checkpoint history differs")
            if tensor_model is not None:
                torch,_=runtime();last=torch.load(file,map_location="cpu",weights_only=True);require(last["step"]==step and validate_state(last,identity,tensor_model)==meta["weights_sha256"],"Checkpoint actual tensor state differs")
            entries.append(dict(step=step,name=name,sha256=checksum,weights_sha256=meta["weights_sha256"],metrics=meta["metrics"]));seen.append(step)
        require(db.execute("SELECT COUNT(*) FROM history").fetchone()[0]==len(entries),"Orphan checkpoint history")
        if seen:require(set(range(identity["checkpoint_every"],seen[-1]+1,identity["checkpoint_every"]))<=set(seen),"Missing scheduled checkpoint")
        if final:require(seen and seen[-1]==identity["steps"],"Missing final checkpoint")
    return entries,last


RESULT_FIELDS={"schema","stage","status","identity","steps_completed","selected_step","draws","weights_sha256","initial_weights_sha256","train_rows","admitted_families","initial_metrics","metrics","warm_init_parity","recovery","elapsed_seconds_this_invocation","peak_host_working_set_bytes","training_performed","teacher_rehearsal","default_promotion","sealed_final_test_consumed"}


def validate_result(value,identity,manifest,rows):
    require(set(value)==RESULT_FIELDS and value["schema"]==SCHEMA and value["stage"]=="training" and value["status"]=="complete" and value["identity"]==identity,"Training completed fields/identity differ")
    for name,expected in (("steps_completed",identity["steps"]),("selected_step",identity["steps"]),("draws",identity["steps"]*identity["batch_size"]),("train_rows",len(rows)),("admitted_families",manifest["summary"]["admitted_families"])):require(type(value[name]) is int and value[name]==expected,"Training result count differs")
    validate_metrics(value["initial_metrics"],len(rows));validate_metrics(value["metrics"],len(rows))
    require(value["initial_weights_sha256"]==SOURCE_WEIGHTS and value["training_performed"] is True and value["teacher_rehearsal"] is False and value["default_promotion"] is False and value["sealed_final_test_consumed"] is False,"Training result scope differs")
    require(type(value["elapsed_seconds_this_invocation"]) in (int,float) and math.isfinite(value["elapsed_seconds_this_invocation"]) and value["elapsed_seconds_this_invocation"]>0 and type(value["peak_host_working_set_bytes"]) is int and 0<value["peak_host_working_set_bytes"]<12*1024**3,"Training resource witness differs")
    require(set(value["recovery"])=={"invalid_generations","restored_step"} and value["recovery"]["invalid_generations"]==[] and type(value["recovery"]["restored_step"]) is int and 0<=value["recovery"]["restored_step"]<=identity["steps"],"Training recovery witness differs")
    parity=value["warm_init_parity"];require(set(parity)=={"rows","weights_exact","maximum_value_difference","maximum_policy_difference"} and type(parity["rows"]) is int and 0<parity["rows"]<=32 and parity["weights_exact"] is True and all(type(parity[name]) in (int,float) and math.isfinite(parity[name]) and 0<=parity[name]<.001 for name in ("maximum_value_difference","maximum_policy_difference")),"Warm source parity differs")


def train(path=DEFAULT,replay_path=DEFAULT,fixture=False,steps=None,checkpoint_every=None,batch_size=None,on_event=None,cancel=None):
    path=Path(path).resolve();settings=train_settings(fixture,steps,checkpoint_every,batch_size)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Training fixture must remain in build")
    else:require(path==DEFAULT.resolve() and Path(replay_path).resolve()==DEFAULT.resolve(),"Training production paths differ");admit();gate()
    manifest,rows=load_replay(replay_path);require(manifest["summary"]["training_admission"]=="ready","blocked_insufficient_admitted_families: no production optimizer work, replacements or expansion permitted")
    require(manifest["identity"]["fixture"] is fixture,"Training/replay fixture ownership differs")
    if (path/"result.json").exists():
        verify_run(path,fixture=fixture);result=read(path/"result.json");require(all(result["identity"][key]==value for key,value in settings.items()),"Completed training invocation settings differ")
        if on_event:on_event(dict(event="selfplay_pilot_training_completed_verified",step=result["selected_step"],optimizer_steps=0,torch_imported="torch" in sys.modules))
        return result
    identity=train_identity(settings,fixture,replay_path,manifest,rows)
    if on_event:on_event(dict(event="selfplay_pilot_training_cpu_admitted",rows=len(rows),admitted_families=manifest["summary"]["admitted_families"],torch_imported="torch" in sys.modules))
    started=time.monotonic();torch,prior=runtime();require(not torch.cuda.is_initialized() and str(torch.__version__)==identity["torch_version"],"CPU Torch runtime differs")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    prior.deterministic("cpu");random.seed(settings["seed"]);torch.manual_seed(settings["seed"]);model=evaluator();source=Model(SOURCE)
    model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in source.weights.items()});require(prior.weight_digest(model.state_dict())==SOURCE_WEIGHTS and all(parameter.requires_grad for parameter in model.parameters()),"Exact source/all8 training initialization differs")
    t=tensors(rows);initial=training_metrics(model,t)
    with torch.no_grad():
        z,logits=model.both(t["x"][:32]);prediction=(t["d"][:32]+t["r"][:32]*z).tolist();logits=logits.tolist()
    value_diff=policy_diff=0.
    with CpuBatchEvaluator(source,cache_capacity=0,max_batch=1) as native_eval:
        for index,row in enumerate(rows[:32]):
            expected=native_eval.evaluate_many([row["state"]])[0];value_diff=max(value_diff,abs(expected["value_player0"]-prediction[index]));policy_diff=max(policy_diff,max(abs(a-b) for a,b in zip(expected["policy_logits"],logits[index])))
    parity=dict(rows=min(32,len(rows)),weights_exact=True,maximum_value_difference=value_diff,maximum_policy_difference=policy_diff);require(value_diff<.001 and policy_diff<.001,"Warm literal/native/Torch forward parity differs")
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings["lr"],weight_decay=.0001,foreach=False);rng=random.Random(settings["seed"]);groups=sample_groups(rows);store=prior.Checkpoints(path,identity);step=draws=last_saved=0
    try:
        entries,saved=cpu_catalog(path,identity,len(rows),tensor_model=model);recovery=dict(invalid_generations=[],restored_step=0 if saved is None else saved["step"])
        if saved is not None:model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);rng.setstate(saved["sampler_rng"]);random.setstate(saved["python_rng"]);torch.set_rng_state(saved["torch_rng"]);step,draws=saved["step"],saved["draws"]
        last_saved=step
        def commit():
            nonlocal last_saved
            state=dict(identity=identity,step=step,draws=draws,model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},optimizer=optimizer.state_dict(),sampler_rng=rng.getstate(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=[],scheduler=None,amp_scaler=None,pending_batches=[])
            validate_state(state,identity,model);store.save(state,training_metrics(model,t),on_event);last_saved=step
        while step<settings["steps"]:
            if cancel and cancel():break
            ids=torch.tensor(sample_ids(rng,groups,settings["batch_size"]),dtype=torch.int64);optimizer.zero_grad(set_to_none=True);z,logits=model.both(t["x"][ids]);value_loss=torch.nn.functional.huber_loss(z,t["z"][ids],delta=.1);policy_loss=quality_learning.policy_losses(logits,t["probability"][ids],t["legal"][ids]).mean();loss=value_loss+.25*policy_loss
            require(bool(torch.isfinite(loss)),"Nonfinite joint loss");loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;draws+=settings["batch_size"]
            if step%settings["checkpoint_every"]==0 or step==settings["steps"] or (cancel and cancel()):commit()
        if step>last_saved:commit()
        if step<settings["steps"]:return dict(status="interrupted",steps_completed=step)
        entries,state=cpu_catalog(path,identity,len(rows),final=True,tensor_model=model);payload=export_payload(state["model"],identity,step)
        result=dict(schema=SCHEMA,stage="training",status="complete",identity=identity,steps_completed=step,selected_step=step,draws=draws,weights_sha256=payload["weights_sha256"],initial_weights_sha256=SOURCE_WEIGHTS,train_rows=len(rows),admitted_families=manifest["summary"]["admitted_families"],initial_metrics=initial,metrics=training_metrics(model,t),warm_init_parity=parity,recovery=recovery,elapsed_seconds_this_invocation=time.monotonic()-started,peak_host_working_set_bytes=prior.host_memory(),training_performed=True,teacher_rehearsal=False,default_promotion=False,sealed_final_test_consumed=False)
        validate_result(result,identity,manifest,rows);publish(path/"model.json",dict(model=payload,sha256=config.digest(payload)));publish(path/"result.json",result)
        if on_event:on_event(dict(event="selfplay_pilot_training_complete",step=step,weights_sha256=payload["weights_sha256"]))
        return result
    finally:store.close()


def check_training_identity(identity,path,fixture):
    require(identity["schema"]==SCHEMA and identity["stage"]=="training" and identity["fixture"] is fixture and type(fixture) is bool,"Training identity stage differs")
    manifest,rows=load_replay(ROOT/identity["replay_path"]);settings=train_settings(fixture,identity["steps"] if fixture else None,identity["checkpoint_every"] if fixture else None,identity["batch_size"] if fixture else None)
    require(config.canonical(identity)==config.canonical(train_identity(settings,fixture,ROOT/identity["replay_path"],manifest,rows)),"Exact training source/data/RNG identity differs")
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build"),"Training fixture path differs")
    else:require(Path(path).resolve()==DEFAULT.resolve() and Path(ROOT/identity["replay_path"]).resolve()==DEFAULT.resolve(),"Training production path differs");admit()
    require(manifest["identity"]["fixture"] is fixture and manifest["summary"]["training_admission"]=="ready","Replay ownership/yield gate differs")
    return manifest,rows


def expected_files(path,identity,entries,fixture):
    replay=ROOT/identity["replay_path"];files=[path/"model.json",path/"result.json",path/"catalog.sqlite3",replay/"work.sqlite3",replay/"selfplay-manifest.json",SOURCE,REGISTRY,PLAN,BASELINE,PROJECTION]+[ROOT/name for name in SOURCES+DEPENDENCIES]+[path/entry["name"] for entry in entries]
    if not fixture:files.append(IMPLEMENTATION)
    return {file.relative_to(ROOT).as_posix():sha(file) for file in files}


def verify_run(path=DEFAULT,fixture=False):
    path=Path(path).resolve();model=Model(path/"model.json");identity=model.identity;manifest,rows=check_training_identity(identity,path,fixture);result=read(path/"result.json");validate_result(result,identity,manifest,rows)
    entries,_=cpu_catalog(path,identity,len(rows),final=True);require(model.id==result["weights_sha256"]==entries[-1]["weights_sha256"] and model.envelope["model"]["step"]==identity["steps"] and model.envelope["model"]["uses_oracle"] is False and result["metrics"]==entries[-1]["metrics"],"CPU final checkpoint/model/result linkage differs")
    return dict(schema="selfplay-pilot-cpu-completed-verification-v1",passed=True,identity=identity,selected_step=identity["steps"],model_file_sha256=model.file_sha256,model_weights_sha256=model.id,checkpoints_verified=len(entries),checkpoints=entries,result=result,selfplay_manifest_sha256=identity["replay_manifest_sha256"],selfplay_journal_sha256=identity["replay_journal_sha256"],replay_rows_sha256=identity["replay_rows_sha256"],inputs=expected_files(path,identity,entries,fixture))


def inspect_run(path=DEFAULT,output=None,fixture=False):
    path=Path(path).resolve();cpu=verify_run(path,fixture);identity=cpu["identity"];manifest,rows=load_replay(ROOT/identity["replay_path"]);torch,prior=runtime();torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    prior.deterministic("cpu");model=evaluator();entries,state=cpu_catalog(path,identity,len(rows),final=True,tensor_model=model);t=tensors(rows)
    model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in Model(SOURCE).weights.items()});require(training_metrics(model,t)==cpu["result"]["initial_metrics"],"Inspected initial source metrics differ")
    model.load_state_dict(state["model"]);payload=export_payload(state["model"],identity,identity["steps"])
    require(Model(path/"model.json").envelope==dict(model=payload,sha256=config.digest(payload)) and training_metrics(model,t)==cpu["result"]["metrics"],"Actual final export/metrics differ")
    changed=[name for name in Model(SOURCE).weights if Model(SOURCE).weights[name]!=Model(path/"model.json").weights[name]]
    value=dict(schema="selfplay-pilot-final-model-witness-v1",passed=True,identity=identity,selected_step=identity["steps"],model_file_sha256=cpu["model_file_sha256"],model_weights_sha256=cpu["model_weights_sha256"],checkpoints_verified=len(entries),checkpoints=entries,all_optimizer_sampler_rng_states_verified=True,final_export_equals_checkpoint_tensors=True,all8_tensors_trainable=True,changed_tensors=changed,source_weights_sha256=SOURCE_WEIGHTS,metrics=cpu["result"]["metrics"],initial_metrics=cpu["result"]["initial_metrics"],selfplay_manifest_sha256=identity["replay_manifest_sha256"],selfplay_journal_sha256=identity["replay_journal_sha256"],replay_rows_sha256=identity["replay_rows_sha256"],sealed_final_test_consumed=False,default_promotion=False,inputs=cpu["inputs"])
    value["sha256"]=config.digest(value)
    if output is not None:publish(output,value)
    return value


def witness(path=DEFAULT,proof=WITNESS):
    value=read(proof);cpu=verify_run(path)
    fields={"schema","passed","identity","selected_step","model_file_sha256","model_weights_sha256","checkpoints_verified","checkpoints","all_optimizer_sampler_rng_states_verified","final_export_equals_checkpoint_tensors","all8_tensors_trainable","changed_tensors","source_weights_sha256","metrics","initial_metrics","selfplay_manifest_sha256","selfplay_journal_sha256","replay_rows_sha256","sealed_final_test_consumed","default_promotion","inputs","sha256"}
    changed=[name for name in Model(SOURCE).weights if Model(SOURCE).weights[name]!=Model(Path(path)/"model.json").weights[name]]
    require(set(value)==fields and type(value["selected_step"]) is int and type(value["checkpoints_verified"]) is int and value["changed_tensors"]==changed,"Model witness field/tensor ownership differs")
    require(value["schema"]=="selfplay-pilot-final-model-witness-v1" and value["passed"] is True and value["sha256"]==config.digest({key:item for key,item in value.items() if key!="sha256"}) and value["inputs"]==cpu["inputs"] and value["identity"]==cpu["identity"] and value["checkpoints"]==cpu["checkpoints"] and value["checkpoints_verified"]==cpu["checkpoints_verified"],"Inspected model witness differs")
    require(value["selected_step"]==TRAIN["steps"] and value["model_file_sha256"]==cpu["model_file_sha256"] and value["model_weights_sha256"]==cpu["model_weights_sha256"] and value["all_optimizer_sampler_rng_states_verified"] is True and value["final_export_equals_checkpoint_tensors"] is True and value["all8_tensors_trainable"] is True and value["metrics"]==cpu["result"]["metrics"] and value["initial_metrics"]==cpu["result"]["initial_metrics"] and value["source_weights_sha256"]==SOURCE_WEIGHTS and value["selfplay_manifest_sha256"]==cpu["selfplay_manifest_sha256"] and value["selfplay_journal_sha256"]==cpu["selfplay_journal_sha256"] and value["replay_rows_sha256"]==cpu["replay_rows_sha256"] and value["sealed_final_test_consumed"] is False and value["default_promotion"] is False,"Model witness verification claims differ")
    for name,digest in value["inputs"].items():require(sha(ROOT/name)==digest,"Model witness input changed")
    return {**value,"proof_file_sha256":sha(proof)}
