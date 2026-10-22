"""Predeclared compact policy/value and exploration ablation; CPU-only admission."""
from __future__ import annotations
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from . import config,phase6_common as prior,pilot,deeper_teacher_v2 as teacher
from .native import ROOT,Native,flatten

PLAN=ROOT/'build/quality-plan.json'
IMPLEMENTATION=ROOT/'build/quality-implementation.json'
REGISTRY=ROOT/'build/quality-registry.json'
TRAIN_DATA=ROOT/'build/quality-data.json'
WARM=prior.WARM
REFINED=ROOT/'data/quality-train-v1/model.json'
RUN=ROOT/'data/quality-benchmark-v1'
OUTPUT=ROOT/'build/quality-benchmark.json'
PROFILES=('warm_control','warm_search','refined_control','refined_search')
BASELINES=('H_v1','H8_relay_CPU')
ROLES={'common':'congklak/quality_common.py','learning':'congklak/quality_learning.py',
       'train_cli':'tools/quality-train.py','search':'congklak/quality_search.py',
       'benchmark':'tools/benchmark-quality.py','analysis':'congklak/quality_analysis.py',
       'progress':'congklak/quality_progress.py','dashboard':'tools/quality-dashboard.html'}
require=prior.require
sha=prior.sha
publish=prior.publish

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def pins(paths):return {relative(p):sha(p) for p in sorted(set(map(Path,paths)))}
def current(mapping):
    require(isinstance(mapping,dict) and bool(mapping),'Missing immutable inputs')
    for name,digest in mapping.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path)==digest,'Changed quality input: '+name)

def check_prior():
    body=load(ROOT/'build/puct-cpu-acceptance.json')
    require(body['passed'] and body['engineering_passed'] and body['tests_run']==34 and len(body['inputs'])==1368,'Prior CPU engineering acceptance differs')
    current(body['inputs']);return body

def make_registry():
    check_prior();old=load(prior.REGISTRY);quotients=set(old['quotients']);families=set(old['families']);arena_families=set()
    sources=[dict(path=relative(prior.REGISTRY),sha256=sha(prior.REGISTRY),inherited=True)]
    # These projections read provenance/positions only, never held-out target bodies.
    for folder in prior.HISTORY:
        path=ROOT/folder/'work.sqlite3'
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            for family, in db.execute("SELECT DISTINCT json_extract(body,'$.family') FROM games"):
                if family is not None:arena_families.add(family)
    for filename in ('stage5-benchmark-plan.json','stage5-deeper-benchmark-plan.json','phase6-plan.json','puct-cpu-plan.json'):
        arena_families.update(row['family'] for row in load(ROOT/'build'/filename)['arena_starts'])
    native=Native();games=observations=0
    for folder in ('data/phase6-benchmark-v1','data/puct-cpu-benchmark-v1'):
        path=ROOT/folder/'work.sqlite3'
        require(path.is_file(),'Missing protected recent arena '+folder)
        semantic=[];count=0
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
            query="SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.final'),hash FROM games ORDER BY hash"
            for start,moves,family,final,digest in db.execute(query):
                p=prior.normalize(json.loads(start),native);actions=json.loads(moves);families.add(family);arena_families.add(family)
                for action in [*actions,None]:
                    if sum(map(sum,p['pits']))>4:quotients.add(pilot.quotient(p).hex());observations+=1
                    if action is not None:
                        transition=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)
                        require(transition.get('ok'),'Invalid recent protected move');p=prior.normalize(transition['position'],native)
                require(p==json.loads(final),'Protected recent final differs');semantic.append(digest);count+=1
        games+=count;sources.append(dict(path=folder,database_sha256=sha(path),games=count,semantic_sha256=config.digest(semantic),projection='start/moves/family/final/hash only; no value/policy target bodies'))
    require(games==768,'Expected192 Phase6 plus576 CPU arena games')
    body=dict(schema='quality-protected-ownership-v1',rules_digest=config.RULES_DIGEST,quotients=sorted(quotients),families=sorted(families),arena_families=sorted(arena_families),
              sources=sources,recent_arena_games=games,recent_active_observations=observations,metadata_only_dataset_access=True,sealed_target_bodies_read=False,
              scope='Inherited ownership plus evaluated active-root quotients. Exclude an entire selected teacher family when any selected root bridges protected ownership; not a claim to enumerate every possible family trajectory.')
    body={**body,'sha256':config.digest(body)};publish(REGISTRY,body);return body

def read_registry():
    body=load(REGISTRY);require(body['schema']=='quality-protected-ownership-v1' and body['sha256']==config.digest({k:v for k,v in body.items() if k!='sha256'}),'Invalid quality ownership registry');return body

def expected_plan():
    registry=read_registry();data=load(TRAIN_DATA)
    from .quality_learning import CONTRACT
    _,groups,_=teacher.load();dev=groups['development'];old=load(ROOT/'build/puct-cpu-plan.json')
    used=set(registry['arena_families'])
    proxy=prior.choose(dev,dict(ordinary=16,capture=16,bonus=16),'quality-H8-cases-v1',used)
    used.update(r['family'] for r in proxy)
    def contested(r):
        p=r['state'];active=sum(map(sum,p['pits']));return active>=25 and abs(p['stores'][0]-p['stores'][1])<active
    arena=prior.choose([r for r in dev if contested(r)],dict(ordinary=8,capture=8,bonus=8),'quality-arena-v1',used)
    slim=lambda r:{k:r[k] for k in ('state_key','state','family','stratum')}
    cases=[dict(cohort='quality_H8_development',**slim(r),reference=r['label']) for r in proxy]
    cases += [r for r in old['cases'] if r['reference']['kind']=='exact_optimal_margin']
    require(len(cases)==80 and len(arena)==24,'Quality cohort coverage differs')
    comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in PROFILES for b in BASELINES]
    order=[];names=[c['id'] for c in comparisons]
    for pair in range(24):
        shift=pair%len(names)
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
    profiles={name:dict(model='warm' if name.startswith('warm_') else 'refined',c_puct=1.5 if name.endswith('control') else .25,cache_capacity=0) for name in PROFILES}
    return dict(schema='quality-policy-value-search-plan-v1',rules_profile=config.PROFILE,rules_digest=config.RULES_DIGEST,
        prior_acceptance_sha256=sha(ROOT/'build/puct-cpu-acceptance.json'),prior_input_count=1368,registry_sha256=registry['sha256'],data_manifest_sha256=sha(TRAIN_DATA),
        training=CONTRACT,training_data=data,models=dict(warm=relative(WARM),refined=relative(REFINED)),warm_model_sha256=sha(WARM),profiles=profiles,
        engines=list(BASELINES+PROFILES),cases=cases,arena_starts=[slim(r) for r in arena],arena_comparisons=comparisons,arena_work_order=order,
        budgets_ms=[25,100],arena_ms=25,arena_max_choices=512,max_depth=16,tt_mb=16,use_pvs=True,memory_mb=8192,cpu_declared=prior.CPU,
        source_model=old['source_model'],source_model_sha256=old['source_model_sha256'],puct=dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap'),
        bootstrap_samples=4096,bootstrap_seed=20261017,primary='24 terminal paired families, zero fallbacks, lower95% paired seed-margin bootstrap above0 against BOTH baselines',
        novelty='Cases and arenas use distinct families absent from prior arena familyIDs; development data already exposed to ancestral checkpoint selection. Not unseen release validation.',
        selection='One bundled joint model treatment and one exploration factor; all four profiles retained; fixedfinal1500 model; no outcome-driven parameter or checkpoint selection',
        clock='Whole decision including cache reset and validation; grant floored positive remaining milliseconds, never expired time_ms0; overshoot retained',
        default_promotion=False,sealed_final_test_consumed=False,energy_measured=False,release='Exploratory eight unadjusted paired95% comparisons; H_v1 stays default')

def make_plan():
    value=expected_plan();publish(PLAN,value);return value
def read_plan():
    # Do not recompute all9000 relay encodings on every CLI admission. Source,
    # corpus, registry and this complete plan are hash-pinned by admit(); the
    # independent plan check rederives cohort selection from the original rows.
    value=load(PLAN);registry=read_registry();data=load(TRAIN_DATA)
    from .quality_learning import CONTRACT
    fields={'arena_comparisons','arena_max_choices','arena_ms','arena_starts','arena_work_order','bootstrap_samples','bootstrap_seed','budgets_ms','cases','clock','cpu_declared','data_manifest_sha256','default_promotion','energy_measured','engines','max_depth','memory_mb','models','novelty','primary','prior_acceptance_sha256','prior_input_count','profiles','puct','registry_sha256','release','rules_digest','rules_profile','schema','sealed_final_test_consumed','selection','source_model','source_model_sha256','training','training_data','tt_mb','use_pvs','warm_model_sha256'}
    require(set(value)==fields and value['schema']=='quality-policy-value-search-plan-v1' and value['rules_digest']==config.RULES_DIGEST and value['rules_profile']==config.PROFILE,'Quality plan fields/rules differ')
    require(value['registry_sha256']==registry['sha256'] and value['data_manifest_sha256']==sha(TRAIN_DATA) and value['training_data']==data and value['training']==CONTRACT,'Quality data/training plan differs')
    require(value['prior_acceptance_sha256']==sha(ROOT/'build/puct-cpu-acceptance.json') and value['prior_input_count']==1368,'Quality prior evidence differs')
    require(value['models']==dict(warm=relative(WARM),refined=relative(REFINED)) and value['warm_model_sha256']==sha(WARM),'Quality model paths differ')
    expected={name:dict(model='warm' if name.startswith('warm_') else 'refined',c_puct=1.5 if name.endswith('control') else .25,cache_capacity=0) for name in PROFILES}
    require(value['profiles']==expected and value['engines']==list(BASELINES+PROFILES),'Quality ablation differs')
    require(value['budgets_ms']==[25,100] and (value['arena_ms'],value['arena_max_choices'],value['max_depth'],value['tt_mb'],value['memory_mb'])==(25,512,16,16,8192) and value['use_pvs'] is True and value['cpu_declared']==prior.CPU,'Quality search resources differ')
    require(value['puct']==dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap') and value['bootstrap_samples']==4096 and value['bootstrap_seed']==20261017,'Quality search/bootstrap limits differ')
    require(value['default_promotion'] is False and value['sealed_final_test_consumed'] is False and value['energy_measured'] is False,'Unsupported quality release claim')
    require(len(value['cases'])==80 and len(value['arena_starts'])==24,'Incomplete quality cohorts')
    proxy=value['cases'][:48];arena=value['arena_starts'];pf={r['family'] for r in proxy};af={r['family'] for r in arena}
    require(len(pf)==48 and len(af)==24 and not pf&af and not (pf|af)&set(registry['arena_families']),'Quality cohorts reuse an arena family')
    original_train=set(data['train_families'])|{r['family'] for r in data['excluded_rows']}
    require(not (pf|af)&original_train,'Quality evaluation contains an original training family')
    old=load(ROOT/'build/puct-cpu-plan.json')
    require(value['cases'][48:]==[r for r in old['cases'] if r['reference']['kind']=='exact_optimal_margin'],'Known-exact cohort differs')
    require(value['source_model']==old['source_model'] and value['source_model_sha256']==old['source_model_sha256'],'H8 baseline differs')
    comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in PROFILES for b in BASELINES]
    names=[c['id'] for c in comparisons];order=[]
    for pair in range(24):
        shift=pair%8
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
    require(value['arena_comparisons']==comparisons and value['arena_work_order']==order,'Quality arena schedule differs')
    return value

def freeze(refresh_fixture=False):
    value=read_plan();files=[ROOT/p for p in ROLES.values()]
    science=[ROOT/'build/puct-cpu-acceptance.json',REGISTRY,TRAIN_DATA,PLAN,ROOT/'docs/QUALITY_SCOPE.md']
    body=dict(schema='quality-implementation-v1',roles=ROLES,sources=pins(files),science_inputs=pins(science),rules_digest=config.RULES_DIGEST)
    body={**body,'sha256':config.digest(body)}
    if IMPLEMENTATION.exists() and load(IMPLEMENTATION)!=body:
        require(refresh_fixture,'Quality producer sources already frozen')
        require(not (ROOT/'data/quality-train-v1/catalog.sqlite3').exists() and not (RUN/'work.sqlite3').exists() and not (ROOT/'data/quality-analysis-demo/work.sqlite3').exists(),'Production exists: new experiment version required')
        index=0
        while (ROOT/f'build/quality-implementation-prefreeze-{index}.json').exists():index+=1
        (ROOT/f'build/quality-implementation-prefreeze-{index}.json').write_bytes(IMPLEMENTATION.read_bytes());IMPLEMENTATION.unlink()
    publish(IMPLEMENTATION,body);return body

def admit():
    check_prior();plan=read_plan();manifest=load(IMPLEMENTATION)
    require(manifest['schema']=='quality-implementation-v1' and manifest['sha256']==config.digest({k:v for k,v in manifest.items() if k!='sha256'}),'Invalid quality implementation')
    require(manifest['roles']==ROLES and set(ROLES.values())<=set(manifest['sources']),'Missing quality producer role')
    require({relative(p) for p in (REGISTRY,TRAIN_DATA,PLAN,ROOT/'docs/QUALITY_SCOPE.md',ROOT/'build/puct-cpu-acceptance.json')}<=set(manifest['science_inputs']),'Missing quality science pins')
    current(manifest['sources']);current(manifest['science_inputs']);return plan,manifest
