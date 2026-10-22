"""Predeclared Phase 6 cohorts and metadata-only protected ownership registry."""
from __future__ import annotations
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from . import config,pilot,campaign,deeper_teacher_v2 as teacher
from .native import ROOT,Native,flatten

PLAN=ROOT/'build/phase6-plan.json'
REGISTRY=ROOT/'build/phase6-protected-registry.json'
SELFPLAY=ROOT/'data/phase6-selfplay-v1'
WARM=ROOT/'data/training-phase6-warm-v1/model.json'
UPDATE=ROOT/'data/training-phase6-update-v1/model.json'
CPU=dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1)
HISTORY=['data/neural-benchmark','data/neural-benchmark-v2','data/neural-benchmark-v3','data/stage5-benchmark','data/stage5-action-benchmark','data/stage5-action-benchmark-v2','data/stage5-deeper-benchmark']

def require(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def publish(path,value):
    """Completed reruns never touch equal bytes or timestamps."""
    path=Path(path);body=config.canonical(value)+'\n'
    if path.exists():
        require(path.read_text(encoding='utf-8')==body,'Immutable published content differs: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True);teacher.atomic_json(path,value)

def rawkey(p):return bytes(flatten(p)+[p['controller']]).hex()

def normalize(p,native):
    result=native.normalize(flatten(config.position(p)),actor=p['controller'],total=98)
    require(result.get('ok'),'Invalid native normalization');return result['position']

def protected_registry():
    native=Native();quotients=set();families=set();sources=[]
    old_family_by_key={r['state_key']:r['family'] for r in pilot.records(pilot.DEFAULT_PATH,'development')}
    for folder in (pilot.DEFAULT_PATH,campaign.DEFAULT):
        manifest,reserved,owned,fingerprint=campaign.imports(folder)
        for (kind,key),split in reserved.items():
            if split!='train':quotients.add((key if kind=='quotient' else pilot.quotient(pilot.from_key(key))).hex())
        families.update(f for f,s in owned.items() if s!='train')
        sources.append(dict(path=str(folder.relative_to(ROOT)),metadata_fingerprint=fingerprint,manifest_sha256=config.digest(manifest)))
    # Invalidated generation payloads still own the evaluation metadata they exposed.
    legacy=ROOT/'data/stronger-10000/work.sqlite3';legacy_keys=set()
    with closing(sqlite3.connect(legacy.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');legacy_families=[list(r) for r in db.execute("SELECT id,split FROM families ORDER BY id")]
        families.update(f for f,s in legacy_families if s!='train')
        for state, in db.execute("SELECT DISTINCT o.state FROM origins o JOIN families f ON f.id=o.family WHERE f.split!='train'"):
            legacy_keys.add(bytes(state))
        for state, in db.execute("SELECT state FROM selected WHERE split!='train'"):legacy_keys.add(bytes(state))
        for kind,state,split in db.execute('SELECT kind,state,split FROM exclusions'):
            if split!='train':quotients.add((bytes(state) if kind=='quotient' else pilot.quotient(pilot.from_key(bytes(state)))).hex())
    quotients.update(pilot.quotient(pilot.from_key(k)).hex() for k in legacy_keys)
    sources.append(dict(path='data/stronger-10000',metadata_fingerprint=config.digest(dict(families=legacy_families,heldout_states=sorted(k.hex() for k in legacy_keys))),quarantined_payloads_used=False))
    trajectories=0;observations=0
    for folder in HISTORY:
        path=ROOT/folder/'work.sqlite3';semantic=[]
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            db.execute('BEGIN')
            projection="SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.start_key'),json_extract(body,'$.final'),hash FROM games ORDER BY hash"
            for start,moves,family,startkey,final,digest in db.execute(projection):
                game=dict(start=json.loads(start),moves=json.loads(moves),final=json.loads(final),family=family,start_key=startkey)
                semantic.append(digest);family=game.get('family',old_family_by_key.get(game.get('start_key')))
                if family is None:family=old_family_by_key.get(game.get('start_key'))
                require(family is not None,'Unknown historical arena family');families.add(family);p=normalize(game['start'],native)
                for action in [*game['moves'],None]:
                    if sum(map(sum,p['pits']))>4:quotients.add(pilot.quotient(p).hex());observations+=1
                    if action is not None:
                        t=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)
                        require(t.get('ok'),'Invalid protected arena replay');p=normalize(t['position'],native)
                require(p==game['final'],'Protected arena final mismatch');trajectories+=1
        sources.append(dict(path=folder,database_sha256=sha(path),game_semantic_sha256=config.digest(semantic),games=len(semantic),projection='start/moves/family/final; no outcome targets'))
    body=dict(schema='phase6-protected-ownership-v1',rules_digest=config.RULES_DIGEST,quotients=sorted(quotients),families=sorted(families),sources=sources,
              historical_games=trajectories,historical_active_observations=observations,metadata_only_dataset_access=True,sealed_target_bodies_read=False)
    return {**body,'sha256':config.digest(body)}

def choose(rows,counts,namespace,exclude=()):
    ordered=sorted(rows,key=lambda r:hashlib.sha256((namespace+'/'+r['state_key']).encode()).digest());used=set(exclude);groups={}
    for kind,count in counts.items():
        groups[kind]=[]
        for row in ordered:
            if row['family'] not in used and row['stratum'].endswith('/'+kind):
                groups[kind].append(row);used.add(row['family'])
                if len(groups[kind])==count:break
        require(len(groups[kind])==count,'Insufficient predeclared '+kind+' families')
    return [row for kind in groups for row in groups[kind]]

def make_plan():
    registry=protected_registry();publish(REGISTRY,registry)
    _,groups,_=teacher.load();train=groups['train'];dev=groups['development']
    def contested(row):
        p=row['state'];r=sum(map(sum,p['pits']));return r>=25 and abs(p['stores'][0]-p['stores'][1])<r
    starts=choose([r for r in train if contested(r) and r['family'] not in set(registry['families']) and pilot.quotient(r['state']).hex() not in set(registry['quotients'])],
                  dict(ordinary=12,capture=10,bonus=10),'phase6-selfplay-start-v1')
    old=json.loads((ROOT/'build/stage5-deeper-benchmark-plan.json').read_text(encoding='utf-8'))
    historical_families={r['family'] for name in ('stage5-benchmark-plan.json','stage5-deeper-benchmark-plan.json')
                         for r in json.loads((ROOT/'build'/name).read_text(encoding='utf-8'))['arena_starts']}
    arena=choose([r for r in dev if contested(r)],dict(ordinary=8,capture=8,bonus=8),'phase6-arena-start-v1',historical_families)
    proxies=choose(dev,dict(ordinary=16,capture=16,bonus=16),'phase6-H8-proxy-v1')
    cases=[dict(cohort='current_H8_development',**{k:r[k] for k in ('state_key','state','family','stratum')},reference=r['label']) for r in proxies]
    exact=[r for r in old['cases'] if r['reference']['kind']=='exact_optimal_margin'];require(len(exact)==32,'Exact diagnostic cohort changed');cases+=exact
    slim=lambda r:{k:r[k] for k in ('state_key','state','family','stratum')}
    comparisons=[dict(id=f'{generation}_cuda_vs_{base}',candidate=generation+'_cuda',opponent=base,pairs=24)
                 for generation in ('warm','updated') for base in ('H_v1','H8_relay_CPU')]
    plan=dict(schema='phase6-bounded-pilot-plan-v1',rules_profile=config.PROFILE,rules_digest=config.RULES_DIGEST,local_profile='Betawi confirmed by user 2026-10-15',
              registry_sha256=registry['sha256'],source_model='data/training-stage5-deeper-h8-f127-seed2718/model.json',
              source_model_sha256=sha(ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json'),
              selfplay_starts=[slim(r) for r in starts],selfplay=dict(games=64,runs_per_family=2,seed=20261015,parallel_roots=8,simulations=64,max_nodes=2048,max_choices=512,
                  root_noise=dict(alpha=.3,fraction=.25),action_temperature_first_choices=8,rawstate_observation_cap=4,admission='Both games in family terminal, no protected active>4 bridge in either game; active>4 only; no replacement families',
                  weights='Unit retained row weight; uniform stratum then uniform row with replacement; max4 rawstate observations by deterministic hash; preserve individual game returns; no averaging',split='train'),
              warm=dict(steps=1000,batch_size=256,lr=.001,seed=20261015,checkpoint_every=50,temperature_seeds=4,selection='Scheduled uniform-dev-stratum teacher policy CE; frozen six value tensors'),
              update=dict(steps=500,batch_size=128,lr=.0003,seed=20261015,checkpoint_every=50,selection='Fixed final500',loss='.5*Huber_teacher_seeds+.5*Huber_selfplay_seeds+.25*(.5*CE_teacher+.5*CE_visits)'),
              cases=cases,arena_starts=[slim(r) for r in arena],budgets_ms=[25,100],max_depth=16,tt_mb=16,use_pvs=True,memory_mb=8192,arena_ms=25,arena_max_choices=512,
              engines=['H_v1','H8_relay_CPU','warm_native','warm_cuda','updated_cuda'],arena_comparisons=comparisons,cpu_declared=CPU,
              puct=dict(simulations=100000,max_nodes=2048,c_puct=1.5,root_noise=None,first_play_urgency='parent fixed-player0 neural bootstrap'),bootstrap_samples=4096,bootstrap_seed=20261017,
              primary='Paired family mean candidate terminal seed margin; lower95% bootstrap >0 against BOTH baselines; incomplete pairs excluded and counted',
              recommendation='At least one completed depth/simulation; otherwise legal firstpit fallback counted; whole wrapper walltime and overshoot recorded',
              warmup='Per GPU model one batch8 evaluation before timing; startup and warmup separately recorded; no perposition tree warmup',
              release='No automatic default promotion; dev families previously used for checkpoint selection; sealed test untouched',sealed_final_test_consumed=False,
              common_source_sha256=sha(__file__))
    publish(PLAN,plan);return plan

def read_plan():
    plan=json.loads(PLAN.read_text(encoding='utf-8'));registry=read_registry()
    require(plan['schema']=='phase6-bounded-pilot-plan-v1' and plan['rules_digest']==config.RULES_DIGEST and plan['registry_sha256']==registry['sha256'],'Phase6 protocol differs')
    require(plan['common_source_sha256']==sha(__file__),'Phase6 common source differs; version needed');return plan

def read_registry():
    value=json.loads(REGISTRY.read_text(encoding='utf-8'));require(config.digest({k:v for k,v in value.items() if k!='sha256'})==value['sha256'],'Corrupt protected registry');return value

if __name__=='__main__':print(config.canonical(make_plan()),flush=True)
