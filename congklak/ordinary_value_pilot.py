"""Protected ordinary validation labels and one fixed-policy value-head pilot."""
from collections import Counter,defaultdict
from contextlib import closing
import bisect
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import sys
import time
from . import config,features_relay,pilot,horizon_probe as teacher,value_learning as head,value_pilot_learning as cache,selfplay_training_pilot as joint,phase6_store
from .native import ROOT,Native,flatten
from .neural_relay import Model

SCHEMA="ordinary-root-terminal-rehearsal-fixed-policy-value-pilot-v1"
DEFAULT=ROOT/"data/ordinary-value-pilot-v1"
LABEL_DEFAULT=ROOT/"data/ordinary-value-labels-v1"
BASELINE=ROOT/"build/ordinary-value-baseline.json"
PROJECTION=ROOT/"build/ordinary-value-benchmark-projection.json"
DATA=ROOT/"build/ordinary-value-data.json"
ROWS=ROOT/"build/ordinary-value-rows.json"
REGISTRY=ROOT/"build/ordinary-value-registry.json"
PLAN=ROOT/"build/ordinary-value-plan.json"
IMPLEMENTATION=ROOT/"build/ordinary-value-implementation.json"
LABEL_OUTPUT=ROOT/"build/ordinary-value-labels.json"
WITNESS=ROOT/"build/ordinary-value-model-witness.json"
SOURCE,SOURCE_SHA,SOURCE_WEIGHTS=head.SOURCE,head.SOURCE_SHA,head.SOURCE_WEIGHTS
FROZEN_WEIGHTS,TRAINABLE=head.FROZEN_WEIGHTS,head.TRAINABLE
SOURCES=("congklak/ordinary_value_pilot.py","tools/ordinary-value-pilot.py","tests/ordinary_value_pilot_author.py","docs/ORDINARY_VALUE_PRODUCER.md","docs/ORDINARY_VALUE_SCOPE.md")
DEPENDENCIES=("congklak/value_learning.py","congklak/value_pilot_learning.py","congklak/horizon_probe.py","congklak/natural_study.py","congklak/training_v3.py","congklak/features_relay.py","congklak/ordinary_value_benchmark.py","congklak/ordinary_value_followup.py","tools/ordinary-value-followup.py","build/ordinary-value-benchmark-projection.json","build/ordinary-value-benchmark-plan.json","build/ordinary-value-benchmark-implementation.json","build/training-runtime.json")
STRATA=("50-74/ordinary","25-49/ordinary")
TRAIN=dict(steps=200,batch_size=64,checkpoint_every=20,seed=20261028,lr=.0001,input_count=127,width=64,parameters=12872,trainable_parameters=65,frozen_parameters=12807,device="cpu",cpu_threads=1,optimizer="FreshAdamW/head65/fp32/wd0/foreachFalse/clip1",value_huber_delta=.1,policy_loss_weight=0.,amp=False,loader_workers=0,
    sampler_policy="Perbatch16high then16low: uniformsortedstart thenuniformits2records; then32rehearsal sequentialfloatCDF90uniformroot+10uniformstratum; unitloss",
    hidden_cache_policy="Frozen source FP32 trunk once; row/features/hidden/policy ownership hashes",selection="Fixedfinal200; no checkpoint/outcome selection",promotion=False)
LABELS=dict(roots=6,h8=dict(depth=8,node_budget=2000000),h10=dict(depth=10,node_budget=5000000),h12=dict(depth=12,node_budget=10000000),keep_history=True,time_ms=0,tt_mb=16,use_pvs=True,memory_mb=512,maximum_new_nodes=102000000,primary="RequestedH10completeintersection only; no fallback; incomplete coverage doesnotblock training/benchmark",automatic_expansion=False)
CONTRACT=dict(training=TRAIN,labels=LABELS,source=SOURCE_SHA,new_rows="Only48initialselfplayrecords/24starts; no14descendants/validation",rehearsal="Only7563prioradmittedbaseH8trainingroots before new whole-originfamily protection filter",minimum_sp_starts_per_band=8,teacher_rehearsal_fraction=.5,loss="Unit perdraw normalized actorresidual Huber.1; no policyCE",new_ordinary_teacher_rehearsal=False,default_promotion=False,sealed_final_test_consumed=False)
require,read,sha,publish=cache.require,cache.read,cache.sha,cache.publish
runtime,evaluator=head.runtime,head.evaluator
TABLES=tuple(stage+suffix for stage in ("h8","h10","h12") for suffix in ("_attempts","_labels"))


def equal(a,b):return config.canonical(a)==config.canonical(b)


def baseline():
    value=read(BASELINE);require(value["schema"]=="ordinary-value-prior-immutability-v1" and value["passed"] is True and value["inputs_count"]==len(value["inputs"])==3364 and value["prior_accepted_inputs"]==3361 and value["control_model_sha256"]==sha(SOURCE)==SOURCE_SHA,"Ordinary value baseline differs")
    for name,digest in value["inputs"].items():require((ROOT/name).resolve().is_relative_to(ROOT) and sha(ROOT/name)==digest,"Sealed input changed: "+name)
    return value


def sql_rows(filename,table,key="state"):
    with closing(sqlite3.connect((ROOT/filename).as_uri()+"?mode=ro",uri=True)) as db:return joint.table(db,table,key)


def wrap(root):return dict(root=root,source_root_sha256=config.digest(root),state=root["state"],state_key=root["state_key"],family=root["family"],cohort="validation",purpose="protected_ordinary_validation",legal_mask=root["legal_mask"],stratum=root["stratum"])


def make_registry():
    baseline();projection=read(PROJECTION);roots=sql_rows("data/ordinary-selfplay-pilot-v1/work.sqlite3","roots");rows={key:wrap(row) for key,row in roots.items() if row["cohort"]=="validation"};require(len(rows)==6 and Counter(row["stratum"] for row in rows.values())=={kind:3 for kind in STRATA},"Protected six-root panel differs")
    protection=read(ROOT/"data/ordinary-selfplay-pilot-v1/rollout-protection.json");q=set(protection["quotients"])|set(projection["reserved_quotients"]);f=set(protection["families"])|set(projection["reserved_families"])
    value=dict(schema="ordinary-value-protected-role-registry-v1",rows=rows,quotients=sorted(q),families=sorted(f),validation_pool_families=protection["new_validation_family_ids"],validation_pool_unique_quotients=protection["new_validation_unique_quotients"],projection_sha256=sha(PROJECTION),baseline_sha256=sha(BASELINE),source_collection_sha256=sha(ROOT/"data/ordinary-selfplay-pilot-v1/collection-manifest.json"),source_replay_sha256=sha(ROOT/"data/ordinary-selfplay-pilot-v1/replay-manifest.json"),ancestral_exposure_not_removed=True)
    value["sha256"]=config.digest(value);publish(REGISTRY,value);return value


def target(row):
    p=config.position(row["state"]);f=row["cached_features"];active=sum(map(sum,p["pits"]));r=(1 if p["controller"]==0 else -1)*active;d=p["stores"][0]-p["stores"][1];y=row["target_player0"]
    require(type(f["d"]) is int and f["d"]==d and type(f["r"]) is int and f["r"]==r and active>4 and len(f["x"])==127 and all(type(v) in (int,float) and math.isfinite(v) for v in f["x"]),"Value cached actor features differ")
    require(row["target_kind"] in ("observed_control_selfplay_terminal_margin","approximate_H8_root_margin") and type(y) is int and (row["target_kind"]!="observed_control_selfplay_terminal_margin" or y%2==0) and -1<=(y-d)/r<=1,"Value typed target/interval differs")
    return dict(d=d,r=r,y=y,z=(y-d)/r)


def prepare():
    registry=make_registry();q,f=set(registry["quotients"]),set(registry["families"]);original=read(ROOT/"build/value-pilot-rows.json")["groups"]["train"];base=[row for row in original if row["pool"]=="base"]
    require(len(base)==7563 and all(row["cohort"]=="replay_training" and row["teacher_depth"]==8 for row in base),"Prior admitted7563base H8 rehearsal ownership differs")
    sp=sql_rows("data/ordinary-selfplay-pilot-v1/work.sqlite3","sp_rows","id");initial=[row for row in sp.values() if row["choice"]==0];require(len(initial)==48 and len({r["selected_start_state_key"] for r in initial})==24 and Counter(r["stratum"] for r in initial)=={kind:24 for kind in STRATA},"Initial48target cohort differs")
    families=defaultdict(list)
    for pool,values in (("rehearsal",base),("selfplay",list(sp.values()))):
        for row in values:families[row["family"]].append((pool,row))
    excluded={}
    for family,entries in families.items():
        reasons=[]
        if family in f:reasons.append("protected_origin_family")
        if any(pilot.quotient(row["state"]).hex() in q for _,row in entries):reasons.append("protected_root_quotient_family_bridge")
        if reasons:excluded[family]=reasons
    groups={name:[] for name in ("high","low","rehearsal")}
    for row in initial:
        if row["family"] in excluded:continue
        group="high" if row["stratum"]==STRATA[0] else "low";item=dict(id="selfplay/"+row["id"],state_key=row["state_key"],state=row["state"],family=row["family"],stratum=row["stratum"],cohort="training",target_kind="observed_control_selfplay_terminal_margin",target_player0=row["terminal_return_player0"],cached_features=row["features"],source_row_sha256=config.digest(row),source_pool="initial_selfplay",source_id=row["id"],terminal_game_sha256=row["terminal_game_sha256"],source_decision_sha256=row["source_decision_sha256"]);target(item);groups[group].append(item)
    native=Native()
    for row in base:
        if row["family"] in excluded:continue
        head.teacher.validate_label(row["label"],row["state"],row["legal_mask"],8,native)
        item=dict(id="rehearsal/"+row["state_key"],state_key=row["state_key"],state=row["state"],family=row["family"],stratum=row["stratum"],cohort="training",target_kind="approximate_H8_root_margin",target_player0=row["label"]["target_player0"],cached_features=row["cached_features"],source_row_sha256=config.digest(row),source_pool="prior_admitted_base_H8",source_id=row["state_key"]);target(item);groups["rehearsal"].append(item)
    for values in groups.values():values.sort(key=lambda row:row["id"])
    rehearsal=groups["rehearsal"];counts=Counter(r["stratum"] for r in rehearsal);require(rehearsal,"No eligible H8 rehearsal roots")
    for row in rehearsal:row["sampling_probability"]=(.9/len(rehearsal))+.1/(len(counts)*counts[row["stratum"]])
    paired={}
    for name in ("high","low"):
        byroot=defaultdict(list)
        for row in groups[name]:byroot[row["state_key"]].append(row)
        require(all(len(values)==2 and len({r["family"] for r in values})==1 for values in byroot.values()),"Selfplay pair split/origin differs");paired[name]=len(byroot)
    inputs={name:sha(ROOT/name) for name in ("build/value-pilot-rows.json","build/value-pilot-data.json","data/ordinary-selfplay-pilot-v1/work.sqlite3","data/ordinary-selfplay-pilot-v1/replay-manifest.json","data/ordinary-selfplay-pilot-v1/rollout-protection.json")}
    value=dict(schema="ordinary-value-admitted-dataset-v1",rules_digest=config.RULES_DIGEST,source_model_sha256=SOURCE_SHA,registry_sha256=sha(REGISTRY),projection_sha256=sha(PROJECTION),baseline_sha256=sha(BASELINE),groups={name:len(values) for name,values in groups.items()},paired_starts=paired,training_admission=all(n>=8 for n in paired.values()),source_base_rows=7563,source_initial_selfplay_rows=48,excluded_origin_families=excluded,excluded_base_rows=sum(r["family"] in excluded for r in base),excluded_initial_rows=sum(r["family"] in excluded for r in initial),excluded_descendants=sorted(r["id"] for r in sp.values() if r["choice"]!=0),row_groups_sha256=config.digest(groups),inputs=inputs,no_validation_or_descendant_gradients=True,no_new_teacher_rehearsal=True)
    publish(ROWS,dict(schema=value["schema"],groups=groups));value["rows_file_sha256"]=sha(ROWS);publish(DATA,value);return value,groups


def load_data():
    manifest=read(DATA);groups=read(ROWS)["groups"];require(manifest["schema"]=="ordinary-value-admitted-dataset-v1" and manifest["row_groups_sha256"]==config.digest(groups) and manifest["rows_file_sha256"]==sha(ROWS) and manifest["registry_sha256"]==sha(REGISTRY) and manifest["projection_sha256"]==sha(PROJECTION) and manifest["baseline_sha256"]==sha(BASELINE),"Frozen training dataset differs")
    expected_inputs=("build/value-pilot-rows.json","build/value-pilot-data.json","data/ordinary-selfplay-pilot-v1/work.sqlite3","data/ordinary-selfplay-pilot-v1/replay-manifest.json","data/ordinary-selfplay-pilot-v1/rollout-protection.json")
    require(set(manifest["inputs"])==set(expected_inputs),"Training source input set differs")
    for name,digest in manifest["inputs"].items():require(sha(ROOT/name)==digest,"Accepted training source changed: "+name)
    require(set(groups)=={"high","low","rehearsal"} and manifest["groups"]=={name:len(values) for name,values in groups.items()},"Dataset group ownership differs")
    for values in groups.values():
        for row in values:target(row)
    validate_source_rows(manifest,groups)
    return manifest,groups


def validate_source_rows(manifest,groups):
    registry=read(REGISTRY);pq,pf=set(registry["quotients"]),set(registry["families"]);base={row["state_key"]:row for row in read(ROOT/"build/value-pilot-rows.json")["groups"]["train"] if row["pool"]=="base"};allsp=sql_rows("data/ordinary-selfplay-pilot-v1/work.sqlite3","sp_rows","id");sp={key:row for key,row in allsp.items() if row["choice"]==0};allsource=[*base.values(),*allsp.values()];blocked={row["family"] for row in allsource if row["family"] in pf or pilot.quotient(row["state"]).hex() in pq}
    require(set(manifest["excluded_origin_families"])==blocked,"Whole-origin-family exclusion set differs")
    expected={"high":set(),"low":set(),"rehearsal":set()}
    for key,row in sp.items():
        if row["family"] not in blocked:expected["high" if row["stratum"]==STRATA[0] else "low"].add("selfplay/"+key)
    expected["rehearsal"]={"rehearsal/"+key for key,row in base.items() if row["family"] not in blocked}
    counts=Counter(row["stratum"] for row in groups["rehearsal"]);paired={}
    for name,values in groups.items():
        require([row["id"] for row in values]==sorted(expected[name]) and all(row["cohort"]=="training" for row in values),"Exact admitted row/order/role set differs")
        for row in values:
            origin=base[row["source_id"]] if name=="rehearsal" else sp[row["source_id"]];require(row["source_row_sha256"]==config.digest(origin) and all(equal(row[key],origin[key]) for key in ("state","state_key","family","stratum")) and equal(row["cached_features"],origin["cached_features"] if name=="rehearsal" else origin["features"]),"Exact target source/features binding differs")
            require(row["target_player0"]==(origin["label"]["target_player0"] if name=="rehearsal" else origin["terminal_return_player0"]) and row["target_kind"]==("approximate_H8_root_margin" if name=="rehearsal" else "observed_control_selfplay_terminal_margin"),"Typed source target changed")
            if name=="rehearsal":require(row["sampling_probability"]==.9/len(values)+.1/(len(counts)*counts[row["stratum"]]),"Rehearsal weighting changed")
            else:require(row["terminal_game_sha256"]==origin["terminal_game_sha256"] and row["source_decision_sha256"]==origin["source_decision_sha256"],"Raw terminal provenance changed")
        if name!="rehearsal":
            origins=Counter(row["state_key"] for row in values);require(all(n==2 for n in origins.values()),"Admitted terminal pair lost");paired[name]=len(origins)
    require(manifest["paired_starts"]==paired and manifest["training_admission"] is all(n>=8 for n in paired.values()),"Paired start gate changed")
    cache.cdf(groups["rehearsal"])


def plan():
    manifest,_=prepare();value=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,data_sha256=sha(DATA),rows_sha256=sha(ROWS),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),projection_sha256=sha(PROJECTION),source_sha256=SOURCE_SHA);publish(PLAN,value);return value


def freeze():
    plan();value=dict(schema="ordinary-value-implementation-v1",plan_sha256=sha(PLAN),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES});publish(IMPLEMENTATION,value);return value


def admit():
    baseline();manifest,groups=load_data();value=read(PLAN);expected=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,data_sha256=sha(DATA),rows_sha256=sha(ROWS),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),projection_sha256=sha(PROJECTION),source_sha256=SOURCE_SHA)
    require(equal(value,expected) and equal(read(IMPLEMENTATION),dict(schema="ordinary-value-implementation-v1",plan_sha256=sha(PLAN),sources={name:sha(ROOT/name) for name in SOURCES},dependencies={name:sha(ROOT/name) for name in DEPENDENCIES})),"Ordinary value plan/source freeze differs");return value,manifest,groups


def gate():
    from .ordinary_value_benchmark import pretraining_witness
    value=pretraining_witness();require(value["passed"] is True,"Benchmark source/reservation gate incomplete");return value


def label_settings(fixture=False,roots=None):
    require(type(fixture) is bool,"Label fixture flag differs");value=json.loads(config.canonical(LABELS))
    if fixture:
        value["roots"]=2 if roots is None else config.integer(roots,"fixture_roots",1,6);value["fixture_panel"]="FirstalreadyconsumeddiagnosticrootswithcompleteoldH8/H10/H12 frommidgame-label-pilot-v1; no productionroot substitution"
    else:require(roots is None,"Production six-label panel differs")
    return value


def label_rows(settings,fixture):
    if not fixture:return read(REGISTRY)["rows"]
    prior=sql_rows("data/midgame-label-pilot-v1/work.sqlite3","roots");keys=set(prior)
    for stage in ("h8","h10","h12"):keys &= set(sql_rows("data/midgame-label-pilot-v1/work.sqlite3",stage+"_labels"))
    candidates={key:row for key,row in sorted(prior.items()) if key in keys and row["cohort"]=="diagnostic"};require(len(candidates)>=settings["roots"],"Known-answer fixture panel insufficient")
    return dict(list(candidates.items())[:settings["roots"]])


def label_identity(settings,fixture):return dict(schema=SCHEMA,stage="validation_labels",settings=settings,fixture=fixture,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/ordinary-value-pilot.py"),rules_digest=config.RULES_DIGEST,registry_sha256=sha(REGISTRY),data_sha256=sha(DATA),rows_sha256=sha(ROWS),plan_sha256=sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),baseline_sha256=sha(BASELINE),native_build_sha256=Native().build_id,source_model_sha256=sha(SOURCE),pretraining_witness=None if fixture else gate(),panel_sha256=config.digest(label_rows(settings,fixture)),fixture_source_journal_sha256=sha(ROOT/"data/midgame-label-pilot-v1/work.sqlite3") if fixture else None,reserve_bytes=2*1024**3,quota_bytes=50*1024**3)


def validate_label_tables(rows,stores,settings,closed=False):
    native=Native();signatures=defaultdict(dict)
    for stage in ("h8","h10","h12"):
        attempts,labels=stores[stage+"_attempts"],stores[stage+"_labels"]
        require(list(attempts)==list(rows)[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(rows)),"Validation stage prefix/orphan labels differ")
        if attempts and stage!="h8":require(len(stores["h8_attempts" if stage=="h10" else "h10_attempts"])==len(rows),"Validation stages out of order")
        for key,attempt in attempts.items():
            teacher.validate_attempt(rows[key],attempt,labels.get(key),stage,settings,native)
            for report in attempt["history"]:
                depth=report["choice_depth"];sig=teacher.q_signature(teacher.item_for(report,rows[key]));require(depth not in signatures[key] or equal(signatures[key][depth],sig),"Cross-depth-source completed Q/best mismatch");signatures[key][depth]=sig


def label_body(identity,rows,stores):
    summary={stage:dict(attempts=len(stores[stage+"_attempts"]),complete=len(stores[stage+"_labels"]),rejected=sum(a["status"]=="budget_rejected" for a in stores[stage+"_attempts"].values()),missing_requested_reference=sorted(set(rows)-set(stores[stage+"_labels"]))) for stage in ("h8","h10","h12")}
    return dict(schema=SCHEMA,stage="validation_labels",identity=identity,status="complete",roots=len(rows),summary=summary,all_six_primary_H10_complete=not identity["fixture"] and len(stores["h10_labels"])==6,total_new_nodes=sum(a["work"]["nodes"] for stage in ("h8","h10","h12") for a in stores[stage+"_attempts"].values()),table_sha256=dict(roots=config.digest(rows),**{name:config.digest(values) for name,values in stores.items()}),deeper_is_not_game_truth=True,training_performed=False,sealed_final_test_consumed=False)


def label_read_core(db,path):
    identity=metadata(db,"identity");fixture=identity["fixture"];settings=label_settings(fixture,identity["settings"]["roots"] if fixture else None)
    require(equal(identity,label_identity(settings,fixture)),"Closed validation source/settings identity differs")
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build"),"Validation fixture path differs")
    else:require(Path(path).resolve()==LABEL_DEFAULT.resolve(),"Validation production path differs");admit()
    rows=joint.table(db,"roots","state");expected=label_rows(settings,fixture);require(equal(rows,expected),"Protected validation original roots changed")
    stores={name:joint.table(db,name,"state") for name in TABLES};validate_label_tables(rows,stores,settings,True);body=label_body(identity,rows,stores);require(equal(metadata(db,"complete"),body),"Validation completed marker differs");return rows,stores,body


def metadata(db,key):
    result=db.execute("SELECT body,hash FROM meta WHERE key=?",(key,)).fetchone();require(result is not None,"Missing committed "+key);value=json.loads(result[0]);require(config.digest(value)==result[1],"Corrupt "+key);return value


def run_labels(path=LABEL_DEFAULT,stage="all",fixture=False,roots=None,on_event=None,cancel=None):
    path=Path(path).resolve();require(stage in ("all","h8","h10","h12"),"Unknown validation stage");settings=label_settings(fixture,roots)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Label fixture must stay in build")
    else:require(path==LABEL_DEFAULT.resolve(),"Validation production path differs");admit();gate()
    require("torch" not in sys.modules,"Validation teacher worker must not load Torch");identity=label_identity(settings,fixture);work=phase6_store.Work(path,identity);event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    try:
        work.db.executescript("CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);"+"".join("CREATE TABLE IF NOT EXISTS "+name+"(state TEXT PRIMARY KEY,body TEXT,hash TEXT);" for name in TABLES));expected=label_rows(settings,fixture);rows=joint.table(work.db,"roots","state");stores={name:joint.table(work.db,name,"state") for name in TABLES}
        if not rows:
            require(not any(stores.values()),"Orphan validation targets")
            with work.db:
                for key,row in expected.items():work.db.execute("INSERT INTO roots VALUES(?,?,?)",(key,config.canonical(row),config.digest(row)))
            rows=expected
        require(equal(rows,expected),"Original six validation roots differ");validate_label_tables(rows,stores,settings)
        complete=work.get("complete")
        if complete is None:
            for target_stage in ("h8","h10","h12"):
                if stage!="all" and int(target_stage[1:])>int(stage[1:]):break
                attempts,labels=stores[target_stage+"_attempts"],stores[target_stage+"_labels"]
                for key in sorted(set(rows)-set(attempts)):
                    if teacher.stopped(cancel):return dict(status="interrupted",stage=target_stage,completed=len(attempts))
                    item,attempt=teacher.execute(rows[key],target_stage,settings,Native(),cancel)
                    if attempt is None:return dict(status="interrupted",stage=target_stage,completed=len(attempts))
                    tentative={name:dict(values) for name,values in stores.items()};tentative[target_stage+"_attempts"][key]=attempt
                    if item is not None:tentative[target_stage+"_labels"][key]=item
                    validate_label_tables(rows,tentative,settings);event("ordinary_value_"+target_stage+"_precommit",completed=len(attempts),target=len(rows),state_key=key,status=attempt["status"]);work.admit(len(config.canonical(attempt))*3+2*1024**2)
                    with work.db:
                        work.db.execute("INSERT INTO "+target_stage+"_attempts VALUES(?,?,?)",(key,config.canonical(attempt),config.digest(attempt)))
                        if item is not None:work.db.execute("INSERT INTO "+target_stage+"_labels VALUES(?,?,?)",(key,config.canonical(item),config.digest(item)))
                    stores=tentative;attempts,labels=stores[target_stage+"_attempts"],stores[target_stage+"_labels"];event("ordinary_value_"+target_stage+"_committed",completed=len(attempts),target=len(rows),state_key=key,status=attempt["status"])
            if not all(len(stores[s+"_attempts"])==len(rows) for s in ("h8","h10","h12")):return dict(status="stage_complete",stage=stage)
            validate_label_tables(rows,stores,settings,True);complete=label_body(identity,rows,stores)
            with work.db:work.put("complete",complete)
        _,_,body=label_read_core(work.db,path);publish(path/"manifest.json",body)
        if not fixture:publish(LABEL_OUTPUT,body)
        event("ordinary_value_labels_complete",summary=body["summary"],total_new_nodes=body["total_new_nodes"]);return body
    finally:work.close()


def load_labels(path=LABEL_DEFAULT):
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/"work.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        db.execute("BEGIN");rows,stores,body=label_read_core(db,path);require((path/"manifest.json").read_text(encoding="utf-8")==config.canonical(body)+"\n","Validation manifest publication differs");return rows,stores,body


def settings_for(fixture=False,steps=None,checkpoint_every=None,batch_size=None):
    require(type(fixture) is bool,"Training fixture flag must be boolean");settings=dict(TRAIN)
    if fixture:settings.update(steps=6 if steps is None else config.integer(steps,"fixture_steps",1,12),checkpoint_every=2 if checkpoint_every is None else config.integer(checkpoint_every,"fixture_checkpoint",1,12));require(batch_size in (None,64),"Fixture preserves exact64sample mix")
    else:require(steps is None and checkpoint_every is None and batch_size is None,"Production training schedule differs")
    return settings


def sampler_for(groups):
    sampler={}
    for name in ("high","low"):
        byroot=defaultdict(list)
        for index,row in enumerate(groups[name]):byroot[row["state_key"]].append(index)
        require(all(len(ids)==2 for ids in byroot.values()),"Sampler must preserve both raw outcomes");sampler[name]={key:byroot[key] for key in sorted(byroot)}
    sampler["rehearsal_cdf"]=cache.cdf(groups["rehearsal"]);return sampler


def sample_batch(rng,sampler):
    result={}
    for name in ("high","low"):
        keys=sorted(sampler[name]);result[name]=[rng.choice(sampler[name][rng.choice(keys)]) for _ in range(16)]
    result["rehearsal"]=[bisect.bisect_right(sampler["rehearsal_cdf"],rng.random()*sampler["rehearsal_cdf"][-1]) for _ in range(32)];return result


def tensors(rows):
    torch,_=runtime();targets=[target(row) for row in rows]
    return dict(x=torch.tensor([row["cached_features"]["x"] for row in rows],dtype=torch.float32),**{name:torch.tensor([value[name] for value in targets],dtype=torch.float32) for name in ("d","r","y","z")})


def metrics(model,groups,data,hidden):
    torch,_=runtime();result={}
    with torch.no_grad():
        for name,rows in groups.items():
            z=model.layers[5](model.layers[4](hidden[name])).squeeze(-1);error=(data[name]["d"]+data[name]["r"]*z-data[name]["y"]).abs().mean().item();loss=torch.nn.functional.huber_loss(z,data[name]["z"],delta=.1).item()
            result[name]=dict(rows=len(rows),mae_seeds=error,residual_huber_delta_point1=loss,target_kind=rows[0]["target_kind"])
    validate_metrics(result,groups);return result


def validate_metrics(value,groups):
    require(set(value)==set(groups),"Training metrics split ownership differs")
    for name,metric in value.items():
        require(set(metric)=={"rows","mae_seeds","residual_huber_delta_point1","target_kind"} and type(metric["rows"]) is int and metric["rows"]==len(groups[name]) and metric["target_kind"]==groups[name][0]["target_kind"],"Training metric fields/count/target differ")
        require(all(type(metric[key]) in (int,float) and math.isfinite(metric[key]) and 0<=metric[key]<=limit for key,limit in (("mae_seeds",196.001),("residual_huber_delta_point1",.201))),"Training metric values differ")


def label_witness():
    _,_,body=load_labels();return dict(manifest_sha256=sha(LABEL_DEFAULT/"manifest.json"),journal_sha256=sha(LABEL_DEFAULT/"work.sqlite3"),all_attempts_complete=True,primary_H10_complete_roots=body["summary"]["h10"]["complete"],missing_primary_H10=body["summary"]["h10"]["missing_requested_reference"],coverage_does_not_select_training=True)


def training_identity(settings,fixture,manifest,groups,proof):
    return dict(schema=SCHEMA,stage="training",**settings,fixture=fixture,rules_digest=config.RULES_DIGEST,encoder=features_relay.ENCODER127,source_sha256=sha(__file__),tool_sha256=sha(ROOT/"tools/ordinary-value-pilot.py"),source_dependencies={name:sha(ROOT/name) for name in DEPENDENCIES},initial_model_file_sha256=SOURCE_SHA,initial_weights_sha256=SOURCE_WEIGHTS,initial_frozen_weights_sha256=FROZEN_WEIGHTS,data_manifest_sha256=sha(DATA),rows_file_sha256=sha(ROWS),admitted_manifest_sha256=config.digest(manifest),sampler=sampler_for(groups),hidden_cache=proof,hidden_cache_sha256=config.digest(proof),plan_sha256=sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),pretraining_witness=None if fixture else gate(),validation_reference_witness=None if fixture else label_witness(),torch_version=read(ROOT/"build/training-runtime.json")["torch"],python_version=platform.python_version())


def validate_state(state,identity,model):
    torch,_=runtime();digest=head.validate_state(state,identity,model);require(identity["device"]=="cpu" and not state["cuda_rng"],"Value pilot CPU ownership differs");rng=random.Random(identity["seed"])
    for _ in range(state["step"]):sample_batch(rng,identity["sampler"])
    require(rng.getstate()==state["sampler_rng"] and random.Random(identity["seed"]).getstate()==state["python_rng"] and torch.equal(cache.expected_torch_rng(identity),state["torch_rng"]),"Value sampler/Python/Torch RNG conservation differs");return digest


def catalog(path,identity,groups,final=False,model=None):
    path=Path(path).resolve();entries=[];last=None
    with closing(sqlite3.connect((path/"catalog.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        configs=db.execute("SELECT body,hash FROM config").fetchall();require(len(configs)==1 and equal(json.loads(configs[0][0]),identity) and configs[0][1]==config.digest(identity),"Checkpoint config differs");seen=[]
        for step,name,checksum,body,digest in db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id"):
            require(type(step) is int and 0<step<=identity["steps"] and (not seen or step>seen[-1]),"Checkpoint step prefix differs");file=(path/name).resolve();require(file.parent==path and sha(file)==checksum,"Checkpoint physical file changed")
            item=json.loads(body);require(set(item)=={"step","identity_sha256","weights_sha256","metrics"} and type(item["step"]) is int and item["step"]==step and item["identity_sha256"]==config.digest(identity) and config.digest(item)==digest,"Checkpoint metadata differs");validate_metrics(item["metrics"],groups)
            history=db.execute("SELECT body,hash FROM history WHERE step=?",(step,)).fetchall();require(len(history)==1 and equal(json.loads(history[0][0]),item["metrics"]) and history[0][1]==config.digest(item["metrics"]),"Checkpoint history differs")
            if model is not None:
                torch,_=runtime();last=torch.load(file,map_location="cpu",weights_only=True);require(last["step"]==step and validate_state(last,identity,model)==item["weights_sha256"],"Actual checkpoint/frozen tensors/optimizer differ")
            entries.append(dict(step=step,name=name,sha256=checksum,weights_sha256=item["weights_sha256"],metrics=item["metrics"]));seen.append(step)
        require(db.execute("SELECT COUNT(*) FROM history").fetchone()[0]==len(entries),"Orphan checkpoint history")
        if seen:require(set(range(identity["checkpoint_every"],seen[-1]+1,identity["checkpoint_every"]))<=set(seen),"Scheduled checkpoint missing")
        if final:require(seen and seen[-1]==identity["steps"],"Final checkpoint missing")
    return entries,last


def export_payload(state,identity,step):
    _,prior=runtime();value=head.export_payload(state,identity,step);value.update(status="experimental_ordinary_terminal_rehearsal_value_head_pilot",value_training="48initialrawterminalobservations plus admittedapproximateH8base rehearsal; unitnormalizedHuber.1; head65only");return value


RESULT_FIELDS={"schema","status","identity","steps_completed","selected_step","draws","weights_sha256","train_rows","group_rows","initial_metrics","metrics","warm_init_parity","frozen_weights_sha256","frozen_policy_unchanged","hidden_cache_sha256","elapsed_seconds_this_invocation","peak_host_working_set_bytes","recovery","training_performed","sealed_final_test_consumed","promotion"}


def validate_result(value,identity,groups):
    require(set(value)==RESULT_FIELDS and value["schema"]==SCHEMA and value["status"]=="complete" and equal(value["identity"],identity),"Completed value result fields/identity differ")
    for name,expected in (("steps_completed",identity["steps"]),("selected_step",identity["steps"]),("draws",identity["steps"]*64),("train_rows",sum(map(len,groups.values())))):require(type(value[name]) is int and value[name]==expected,"Completed value count differs")
    require(value["group_rows"]=={name:len(rows) for name,rows in groups.items()} and value["frozen_weights_sha256"]==FROZEN_WEIGHTS and value["frozen_policy_unchanged"] is True and value["hidden_cache_sha256"]==identity["hidden_cache_sha256"] and value["training_performed"] is True and value["sealed_final_test_consumed"] is False and value["promotion"] is False,"Completed value scope/policy differs")
    validate_metrics(value["initial_metrics"],groups);validate_metrics(value["metrics"],groups)
    require(type(value["elapsed_seconds_this_invocation"]) in (int,float) and math.isfinite(value["elapsed_seconds_this_invocation"]) and value["elapsed_seconds_this_invocation"]>0 and type(value["peak_host_working_set_bytes"]) is int and 0<value["peak_host_working_set_bytes"]<12*1024**3,"Completed resource observation differs")
    require(set(value["recovery"])=={"invalid_generations","restored_step"} and value["recovery"]["invalid_generations"]==[] and type(value["recovery"]["restored_step"]) is int and 0<=value["recovery"]["restored_step"]<=identity["steps"],"Completed recovery observation differs")
    parity=value["warm_init_parity"];require(set(parity)=={"rows","maximum_value_difference","maximum_policy_difference","weights_exact"} and type(parity["rows"]) is int and 0<parity["rows"]<=48 and parity["weights_exact"] is True and all(type(parity[name]) in (int,float) and math.isfinite(parity[name]) and 0<=parity[name]<.001 for name in ("maximum_value_difference","maximum_policy_difference")),"Warm init parity differs")


def train(path=DEFAULT,fixture=False,steps=None,checkpoint_every=None,batch_size=None,on_event=None,cancel=None):
    path=Path(path).resolve();settings=settings_for(fixture,steps,checkpoint_every,batch_size)
    if fixture:require(path.is_relative_to(ROOT/"build"),"Training fixture must remain in build")
    else:require(path==DEFAULT.resolve(),"Production training path differs");admit();gate();label_witness()
    manifest,groups=load_data();require(manifest["training_admission"] is True and all(manifest["paired_starts"][name]>=8 for name in ("high","low")),"Insufficient paired ordinary origins; no replacement permitted")
    if (path/"result.json").exists():
        proof=verify_run(path,fixture=fixture);require(all(equal(proof["identity"][key],value) for key,value in settings.items()),"Completed invocation options differ");return read(path/"result.json")
    torch,prior=runtime();require(not torch.cuda.is_initialized() and str(torch.__version__)==read(ROOT/"build/training-runtime.json")["torch"],"Pinned CPU runtime differs");torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    prior.deterministic("cpu");random.seed(settings["seed"]);torch.manual_seed(settings["seed"]);started=time.monotonic();model=evaluator();model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in Model(SOURCE).weights.items()});require(prior.weight_digest(model.state_dict())==SOURCE_WEIGHTS and [name for name,p in model.named_parameters() if p.requires_grad]==list(TRAINABLE),"Exact source/head65 ownership differs")
    data={name:tensors(rows) for name,rows in groups.items()};hidden,policy,proof=cache.hidden_cache(model,groups,data);identity=training_identity(settings,fixture,manifest,groups,proof);initial=metrics(model,groups,data,hidden)
    from .puct_cpu_inference import CpuBatchEvaluator
    parity=dict(rows=min(48,len(groups["high"])+len(groups["low"])),maximum_value_difference=0.,maximum_policy_difference=0.,weights_exact=True)
    with CpuBatchEvaluator(SOURCE,cache_capacity=0,max_batch=1) as native:
        for name in ("high","low"):
            with torch.no_grad():z,logits=model.both(data[name]["x"])
            for index,row in enumerate(groups[name]):
                value=native.evaluate_many([row["state"]])[0];parity["maximum_value_difference"]=max(parity["maximum_value_difference"],abs(value["value_player0"]-(data[name]["d"][index]+data[name]["r"][index]*z[index]).item()));parity["maximum_policy_difference"]=max(parity["maximum_policy_difference"],max(abs(a-b) for a,b in zip(value["policy_logits"],logits[index].tolist())))
    require(parity["maximum_value_difference"]<.001 and parity["maximum_policy_difference"]<.001,"Warm native/Torch source parity differs")
    trainable=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(trainable,lr=settings["lr"],weight_decay=0.,foreach=False);rng=random.Random(settings["seed"]);store=prior.Checkpoints(path,identity);step=draws=last_saved=0
    try:
        _,saved=catalog(path,identity,groups,model=model);recovery=dict(invalid_generations=[],restored_step=0 if saved is None else saved["step"])
        if saved is not None:model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"]);rng.setstate(saved["sampler_rng"]);random.setstate(saved["python_rng"]);torch.set_rng_state(saved["torch_rng"]);step,draws=saved["step"],saved["draws"]
        last_saved=step
        def commit():
            nonlocal last_saved
            require(cache.policy_equal(model,hidden,policy),"Frozen policy/trunk changed")
            state=dict(identity=identity,step=step,draws=draws,model={key:value.detach().cpu().clone() for key,value in model.state_dict().items()},optimizer=optimizer.state_dict(),sampler_rng=rng.getstate(),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=[],scheduler=None,amp_scaler=None,pending_batches=[]);validate_state(state,identity,model);store.save(state,metrics(model,groups,data,hidden),on_event);last_saved=step
        while step<settings["steps"]:
            if teacher.stopped(cancel):break
            batch=sample_batch(rng,identity["sampler"]);h=torch.cat([hidden[name][batch[name]] for name in ("high","low","rehearsal")]);y=torch.cat([data[name]["z"][batch[name]] for name in ("high","low","rehearsal")]);optimizer.zero_grad(set_to_none=True);z=model.layers[5](model.layers[4](h)).squeeze(-1);loss=torch.nn.functional.huber_loss(z,y,delta=.1);require(bool(torch.isfinite(loss)),"Nonfinite value loss");loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();step+=1;draws+=64
            if step%settings["checkpoint_every"]==0 or step==settings["steps"] or teacher.stopped(cancel):commit()
        if step>last_saved:commit()
        if step<settings["steps"]:return dict(status="interrupted",steps_completed=step)
        _,state=catalog(path,identity,groups,True,model);payload=export_payload(state["model"],identity,step);result=dict(schema=SCHEMA,status="complete",identity=identity,steps_completed=step,selected_step=step,draws=draws,weights_sha256=payload["weights_sha256"],train_rows=sum(map(len,groups.values())),group_rows={name:len(values) for name,values in groups.items()},initial_metrics=initial,metrics=metrics(model,groups,data,hidden),warm_init_parity=parity,frozen_weights_sha256=FROZEN_WEIGHTS,frozen_policy_unchanged=cache.policy_equal(model,hidden,policy),hidden_cache_sha256=identity["hidden_cache_sha256"],elapsed_seconds_this_invocation=time.monotonic()-started,peak_host_working_set_bytes=prior.host_memory(),recovery=recovery,training_performed=True,sealed_final_test_consumed=False,promotion=False);validate_result(result,identity,groups);publish(path/"model.json",dict(model=payload,sha256=config.digest(payload)));publish(path/"result.json",result)
        if on_event:on_event(dict(event="ordinary_value_training_complete",step=step,weights_sha256=payload["weights_sha256"]))
        return result
    finally:store.close()


def check_identity(identity,path,fixture):
    require(type(fixture) is bool and identity["fixture"] is fixture,"Completed fixture ownership differs");manifest,groups=load_data();settings=settings_for(fixture,identity["steps"] if fixture else None,identity["checkpoint_every"] if fixture else None);cache.validate_cache_identity(identity,groups);require(equal(identity,training_identity(settings,fixture,manifest,groups,identity["hidden_cache"])),"Exact completed training identity differs")
    if fixture:require(Path(path).resolve().is_relative_to(ROOT/"build"),"Completed fixture path differs")
    else:require(Path(path).resolve()==DEFAULT.resolve(),"Completed production path differs");admit()
    return manifest,groups


def expected_files(path,identity,entries,fixture):
    files=[Path(path)/"model.json",Path(path)/"result.json",Path(path)/"catalog.sqlite3",SOURCE,DATA,ROWS,REGISTRY,PLAN,BASELINE,PROJECTION]+[ROOT/name for name in SOURCES+DEPENDENCIES]+[Path(path)/item["name"] for item in entries]
    if not fixture:files+=[IMPLEMENTATION,LABEL_DEFAULT/"manifest.json",LABEL_DEFAULT/"work.sqlite3"]
    return {file.relative_to(ROOT).as_posix():sha(file) for file in files}


def verify_run(path=DEFAULT,fixture=False):
    path=Path(path).resolve();model=Model(path/"model.json");identity=model.identity;manifest,groups=check_identity(identity,path,fixture);result=read(path/"result.json");validate_result(result,identity,groups);entries,_=catalog(path,identity,groups,True)
    require(head.model_frozen_digest(model)==FROZEN_WEIGHTS and model.id==result["weights_sha256"]==entries[-1]["weights_sha256"] and model.envelope["model"]["step"]==identity["steps"] and model.envelope["model"]["uses_oracle"] is False and equal(result["metrics"],entries[-1]["metrics"]),"CPU final export/catalog/frozen linkage differs")
    return dict(schema="ordinary-value-cpu-verification-v1",passed=True,identity=identity,selected_step=identity["steps"],model_file_sha256=model.file_sha256,model_weights_sha256=model.id,checkpoints_verified=len(entries),checkpoints=entries,result=result,inputs=expected_files(path,identity,entries,fixture))


def inspect_run(path=DEFAULT,output=None,fixture=False):
    path=Path(path).resolve();cpu=verify_run(path,fixture);identity=cpu["identity"];_,groups=load_data();torch,prior=runtime();torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    prior.deterministic("cpu");model=evaluator();model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in Model(SOURCE).weights.items()});data={name:tensors(rows) for name,rows in groups.items()};hidden,policy,proof=cache.hidden_cache(model,groups,data);require(equal(proof,identity["hidden_cache"]) and equal(metrics(model,groups,data,hidden),cpu["result"]["initial_metrics"]),"Actual source cache/initial metrics differ");entries,state=catalog(path,identity,groups,True,model);payload=export_payload(state["model"],identity,identity["steps"]);require(equal(Model(path/"model.json").envelope,dict(model=payload,sha256=config.digest(payload))),"Actual final tensor/export differs");model.load_state_dict(state["model"]);require(cache.policy_equal(model,hidden,policy) and equal(metrics(model,groups,data,hidden),cpu["result"]["metrics"]),"Actual final frozen policy/metrics differ")
    value=dict(schema="ordinary-value-final-model-witness-v1",passed=True,identity=identity,selected_step=identity["steps"],model_file_sha256=cpu["model_file_sha256"],model_weights_sha256=cpu["model_weights_sha256"],checkpoints_verified=len(entries),checkpoints=entries,all_optimizer_sampler_rng_states_verified=True,final_export_equals_checkpoint_tensors=True,all_frozen_tensors_bit_exact=True,all_dataset_policy_logits_bit_exact=True,frozen_weights_sha256=FROZEN_WEIGHTS,trainable_parameter_count=65,frozen_parameter_count=12807,initial_metrics=cpu["result"]["initial_metrics"],metrics=cpu["result"]["metrics"],hidden_cache_sha256=identity["hidden_cache_sha256"],sealed_final_test_consumed=False,promotion=False,inputs=cpu["inputs"]);value["sha256"]=config.digest(value)
    if output is not None:publish(output,value)
    return value


def witness(path=DEFAULT,proof=WITNESS):
    value=read(proof);cpu=verify_run(path);fields={"schema","passed","identity","selected_step","model_file_sha256","model_weights_sha256","checkpoints_verified","checkpoints","all_optimizer_sampler_rng_states_verified","final_export_equals_checkpoint_tensors","all_frozen_tensors_bit_exact","all_dataset_policy_logits_bit_exact","frozen_weights_sha256","trainable_parameter_count","frozen_parameter_count","initial_metrics","metrics","hidden_cache_sha256","sealed_final_test_consumed","promotion","inputs","sha256"}
    require(set(value)==fields and value["schema"]=="ordinary-value-final-model-witness-v1" and value["passed"] is True and value["sha256"]==config.digest({key:item for key,item in value.items() if key!="sha256"}),"Inspected witness fields/hash differ")
    for key in ("identity","selected_step","model_file_sha256","model_weights_sha256","checkpoints_verified","checkpoints","inputs"):require(equal(value[key],cpu[key]),"Inspected witness linkage differs: "+key)
    require(type(value["selected_step"]) is int and value["selected_step"]==200 and type(value["checkpoints_verified"]) is int and value["checkpoints_verified"]>=10 and value["frozen_weights_sha256"]==FROZEN_WEIGHTS and value["trainable_parameter_count"]==65 and value["frozen_parameter_count"]==12807 and value["hidden_cache_sha256"]==cpu["identity"]["hidden_cache_sha256"] and equal(value["initial_metrics"],cpu["result"]["initial_metrics"]) and equal(value["metrics"],cpu["result"]["metrics"]),"Inspected witness tensor/cache/count claims differ")
    require(all(value[key] is True for key in ("all_optimizer_sampler_rng_states_verified","final_export_equals_checkpoint_tensors","all_frozen_tensors_bit_exact","all_dataset_policy_logits_bit_exact")) and value["sealed_final_test_consumed"] is False and value["promotion"] is False,"Inspected witness scope claims differ")
    return {**value,"proof_file_sha256":sha(proof)}
