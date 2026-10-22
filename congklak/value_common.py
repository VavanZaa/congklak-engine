"""Frozen improved policy with one value-head-only refit; CPU admission."""
from __future__ import annotations
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from . import config, quality_common as previous, phase6_common as prior, pilot, deeper_teacher_v2 as teacher
from .native import ROOT, Native, flatten

PLAN=ROOT/'build/value-plan.json'
IMPLEMENTATION=ROOT/'build/value-implementation.json'
REGISTRY=ROOT/'build/value-registry.json'
TRAIN_DATA=ROOT/'build/value-data.json'
CONTROL=ROOT/'data/quality-train-v1/model.json'
REFIT=ROOT/'data/value-train-v1/model.json'
RUN=ROOT/'data/value-benchmark-v1'
OUTPUT=ROOT/'build/value-benchmark.json'
PROFILES=('policy_control','value_refit')
BASELINES=('H_v1','H8_relay_CPU')
ROLES={'common':'congklak/value_common.py','learning':'congklak/value_learning.py','train_cli':'tools/value-train.py',
       'benchmark':'tools/benchmark-value.py','analysis':'congklak/value_analysis.py',
       'progress':'congklak/value_progress.py','dashboard':'tools/value-dashboard.html'}
require=prior.require
sha=prior.sha
publish=prior.publish
load=previous.load
relative=previous.relative
pins=previous.pins
current=previous.current

def check_prior():
    body=load(ROOT/'build/quality-acceptance.json')
    require(body['passed'] and body['engineering_passed'] and body['tests_run']==36 and len(body['inputs'])==1647,'Prior quality acceptance differs')
    current(body['inputs']);return body

def prior_evaluation_families():
    values=set()
    for filename in ('stage5-benchmark-plan.json','stage5-deeper-benchmark-plan.json','phase6-plan.json','puct-cpu-plan.json','quality-plan.json'):
        body=load(ROOT/'build'/filename)
        values.update(r['family'] for r in body['cases'] if r.get('family'))
        values.update(r['family'] for r in body['arena_starts'])
    return values

def make_registry():
    check_prior();old=previous.read_registry();quotients=set(old['quotients']);families=set(old['families']);arena_families=set(old['arena_families'])
    native=Native();games=observations=0;digests=[];path=ROOT/'data/quality-benchmark-v1/work.sqlite3'
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        query="SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.final'),hash FROM games ORDER BY hash"
        for start,moves,family,final,digest in db.execute(query):
            p=prior.normalize(json.loads(start),native);families.add(family);arena_families.add(family)
            for action in [*json.loads(moves),None]:
                if sum(map(sum,p['pits']))>4:quotients.add(pilot.quotient(p).hex());observations+=1
                if action is not None:
                    transition=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)
                    require(transition.get('ok'),'Invalid protected quality move');p=prior.normalize(transition['position'],native)
            require(p==json.loads(final),'Protected quality final differs');digests.append(digest);games+=1
    require(games==384,'Expected all384 preceding quality games')
    body=dict(schema='value-protected-ownership-v1',rules_digest=config.RULES_DIGEST,quotients=sorted(quotients),families=sorted(families),arena_families=sorted(arena_families),
              evaluation_families=sorted(prior_evaluation_families()),
              sources=[dict(path=relative(previous.REGISTRY),sha256=sha(previous.REGISTRY),inherited=True),dict(path=relative(path),sha256=sha(path),games=games,semantic_sha256=config.digest(digests),projection='start/moves/family/final/hash only')],
              recent_arena_games=games,recent_active_observations=observations,metadata_only_dataset_access=True,sealed_target_bodies_read=False,
              scope='Inherited quality ownership plus all384 preceding quality arena trajectories; selected-root family bridge filter; ancestral exposure is not removed.')
    body={**body,'sha256':config.digest(body)};publish(REGISTRY,body);return body

def read_registry():
    body=load(REGISTRY)
    require(body['schema']=='value-protected-ownership-v1' and body['sha256']==config.digest({k:v for k,v in body.items() if k!='sha256'}),'Invalid value ownership registry')
    return body

def expected_plan():
    from .value_learning import CONTRACT
    registry=read_registry();data=load(TRAIN_DATA);_,groups,_=teacher.load();dev=groups['development'];old=previous.read_plan()
    used=set(registry['arena_families'])|{r['family'] for r in old['cases'][:48]}
    cases=prior.choose(dev,dict(ordinary=16,capture=16,bonus=16),'value-head-cases-v1',used)
    used.update(r['family'] for r in cases)
    def contested(row):
        p=row['state'];active=sum(map(sum,p['pits']));return active>=25 and abs(p['stores'][0]-p['stores'][1])<active
    arena=prior.choose([r for r in dev if contested(r)],dict(ordinary=8,capture=8,bonus=8),'value-head-arena-v1',used)
    slim=lambda r:{k:r[k] for k in ('state_key','state','family','stratum')}
    cases=[dict(cohort='value_H8_development',**slim(r),reference=r['label']) for r in cases]+old['cases'][48:]
    comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in PROFILES for b in BASELINES]
    order=[];names=[c['id'] for c in comparisons]
    for pair in range(24):
        shift=pair%len(names)
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
    return dict(schema='value-head-only-plan-v1',rules_profile=config.PROFILE,rules_digest=config.RULES_DIGEST,
        prior_acceptance_sha256=sha(ROOT/'build/quality-acceptance.json'),prior_input_count=1647,
        registry_sha256=registry['sha256'],data_manifest_sha256=sha(TRAIN_DATA),training=CONTRACT,training_data=data,
        models=dict(control=relative(CONTROL),refit=relative(REFIT)),control_model_sha256=sha(CONTROL),
        profiles={c:dict(model='control' if c=='policy_control' else 'refit',c_puct=.25,cache_capacity=0) for c in PROFILES},
        engines=list(BASELINES+PROFILES),cases=cases,arena_starts=[slim(r) for r in arena],arena_comparisons=comparisons,arena_work_order=order,
        budgets_ms=[25,100],arena_ms=25,arena_max_choices=512,max_depth=16,tt_mb=16,use_pvs=True,memory_mb=8192,cpu_declared=prior.CPU,
        source_model=old['source_model'],source_model_sha256=old['source_model_sha256'],
        puct=dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap'),
        bootstrap_samples=4096,bootstrap_seed=20261019,
        primary='24 terminal paired families, zero fallbacks, lower95% paired seed-margin bootstrap above0 against BOTH baselines',
        novelty='48 H8 cases and24 arena families exclude every prior arena ID and precedingquality48 H8 case families, and are mutually disjoint. OlderStage5 diagnostic families may recur; all development with ancestral exposure.',
        selection='One fixed-final1500 value-only treatment; unchanged improved policy and search; no metric-based checkpoint or outcome retuning',
        clock='Whole decision including reset and validation; floored positive remaining milliseconds, never time_ms0; overshoot retained',
        default_promotion=False,sealed_final_test_consumed=False,energy_measured=False,
        release='Exploratory four unadjusted baseline intervals; matched refit-minus-control mean descriptive only; H_v1 stays default')

def make_plan():
    body=expected_plan();publish(PLAN,body);return body

def read_plan():
    from .value_learning import CONTRACT
    value=load(PLAN);registry=read_registry();data=load(TRAIN_DATA);old=previous.read_plan()
    require(set(value)==set(['arena_comparisons', 'arena_max_choices', 'arena_ms', 'arena_starts', 'arena_work_order', 'bootstrap_samples', 'bootstrap_seed', 'budgets_ms', 'cases', 'clock', 'control_model_sha256', 'cpu_declared', 'data_manifest_sha256', 'default_promotion', 'energy_measured', 'engines', 'max_depth', 'memory_mb', 'models', 'novelty', 'primary', 'prior_acceptance_sha256', 'prior_input_count', 'profiles', 'puct', 'registry_sha256', 'release', 'rules_digest', 'rules_profile', 'schema', 'sealed_final_test_consumed', 'selection', 'source_model', 'source_model_sha256', 'training', 'training_data', 'tt_mb', 'use_pvs']),'Value plan fields differ')
    require(value['schema']=='value-head-only-plan-v1' and value['rules_digest']==config.RULES_DIGEST and value['rules_profile']==config.PROFILE,'Value plan rules differ')
    require(value['registry_sha256']==registry['sha256'] and value['data_manifest_sha256']==sha(TRAIN_DATA) and value['training_data']==data and value['training']==CONTRACT,'Value training metadata differs')
    require(value['prior_acceptance_sha256']==sha(ROOT/'build/quality-acceptance.json') and value['prior_input_count']==1647,'Value prior evidence differs')
    require(value['models']==dict(control=relative(CONTROL),refit=relative(REFIT)) and value['control_model_sha256']==sha(CONTROL),'Value model identity differs')
    profiles={c:dict(model='control' if c=='policy_control' else 'refit',c_puct=.25,cache_capacity=0) for c in PROFILES}
    require(value['profiles']==profiles and value['engines']==list(BASELINES+PROFILES),'Value fixed policy/search profiles differ')
    require(value['budgets_ms']==[25,100] and (value['arena_ms'],value['arena_max_choices'],value['max_depth'],value['tt_mb'],value['memory_mb'])==(25,512,16,16,8192) and value['use_pvs'] is True and value['cpu_declared']==prior.CPU,'Value resources differ')
    require(value['puct']==dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap') and value['bootstrap_samples']==4096 and value['bootstrap_seed']==20261019,'Value search/bootstrap differs')
    require(value['default_promotion'] is False and value['sealed_final_test_consumed'] is False and value['energy_measured'] is False,'Unsupported value release claim')
    require(len(value['cases'])==80 and len(value['arena_starts'])==24 and value['cases'][48:]==old['cases'][48:],'Value case coverage differs')
    cf={r['family'] for r in value['cases'][:48]};af={r['family'] for r in value['arena_starts']}
    require(len(cf)==48 and len(af)==24 and not cf&af and not (cf|af)&(set(registry['arena_families'])|{r['family'] for r in old['cases'][:48]}),'Value families are not fresh')
    require(not (cf|af)&(set(data['train_families'])|{r['family'] for r in data['excluded_rows']}),'Value evaluation contains original train family')
    comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in PROFILES for b in BASELINES]
    names=[c['id'] for c in comparisons];order=[]
    for pair in range(24):
        shift=pair%4
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
    require(value['arena_comparisons']==comparisons and value['arena_work_order']==order,'Value schedule differs')
    require(value['source_model']==old['source_model'] and value['source_model_sha256']==old['source_model_sha256'],'H8 baseline differs')
    return value

def freeze(refresh_fixture=False):
    read_plan();files=[ROOT/p for p in ROLES.values()]
    science=[ROOT/'build/quality-acceptance.json',REGISTRY,TRAIN_DATA,PLAN,ROOT/'docs/VALUE_SCOPE.md']
    body=dict(schema='value-implementation-v1',roles=ROLES,sources=pins(files),science_inputs=pins(science),rules_digest=config.RULES_DIGEST)
    body={**body,'sha256':config.digest(body)}
    if IMPLEMENTATION.exists() and load(IMPLEMENTATION)!=body:
        require(refresh_fixture,'Value producer already frozen')
        require(not (ROOT/'data/value-train-v1/catalog.sqlite3').exists() and not (RUN/'work.sqlite3').exists() and not (ROOT/'data/value-analysis-demo/work.sqlite3').exists(),'Production exists: new version required')
        index=0
        while (ROOT/f'build/value-implementation-prefreeze-{index}.json').exists():index+=1
        (ROOT/f'build/value-implementation-prefreeze-{index}.json').write_bytes(IMPLEMENTATION.read_bytes());IMPLEMENTATION.unlink()
    publish(IMPLEMENTATION,body);return body

def admit():
    check_prior();plan=read_plan();manifest=load(IMPLEMENTATION)
    require(manifest['schema']=='value-implementation-v1' and manifest['sha256']==config.digest({k:v for k,v in manifest.items() if k!='sha256'}),'Invalid value implementation')
    require(manifest['roles']==ROLES and set(manifest['sources'])==set(ROLES.values()),'Missing value producer role')
    require(set(manifest['science_inputs'])=={relative(p) for p in (REGISTRY,TRAIN_DATA,PLAN,ROOT/'docs/VALUE_SCOPE.md',ROOT/'build/quality-acceptance.json')},'Value science pins differ')
    current(manifest['sources']);current(manifest['science_inputs']);return plan,manifest
