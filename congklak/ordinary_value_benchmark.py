"""Frozen equal-time fixed-policy ordinary value-head experiment and allocation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack,closing
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import struct
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as common,phase6_store
from congklak import quality_search
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative,Model,sha
from congklak.quality_search import search as puct_search
from congklak.timing_v2 import PerformanceCore
spec=importlib.util.spec_from_file_location('frozen_phase6_benchmark_value_dependency',ROOT/'tools/benchmark-phase6-v2.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
require=prior.require;integer=prior.integer;numeric=prior.numeric
PLAN=ROOT/'build/ordinary-value-benchmark-plan.json'
IMPLEMENTATION=ROOT/'build/ordinary-value-benchmark-implementation.json'
PROJECTION=ROOT/'build/ordinary-value-benchmark-projection.json'
BASELINE=ROOT/'build/ordinary-value-baseline.json'
CONTROL=ROOT/'data/quality-train-v1/model.json'
PILOT=ROOT/'data/ordinary-value-pilot-v1/model.json'
BASELINES=('H_v1','H8_relay_CPU')
CANDIDATES=('ordinary_control','ordinary_value')
ENGINES=list(BASELINES+CANDIDATES)
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
STAGES=('evaluation',)
COUNTERS=('positions_submitted','evaluations_requested','terminal_returns','native_actual_evals','predict_successes','cache_hits','cache_misses','cache_evictions','feature_simulations','feature_deposits','native_calls','cache_resets')


def counter_snapshot(obj):
    value=obj.snapshot_counters()
    if 'cache_current_entries' in value:value={**value,'cache_occupancy':value['cache_current_entries']};del value['cache_current_entries']
    require(set(value)==set(COUNTERS)|{'cache_capacity','cache_occupancy'},'Evaluator counter fields differ')
    require(all(integer(x) for x in value.values()),'Invalid evaluator counters')
    require(value['cache_capacity']==0 and value['cache_occupancy']<=value['cache_capacity'],'Unbounded cache snapshot')
    return value

def validate_telemetry(value,decision,profile):
    if profile in ('H_v1','H8_relay_CPU'):
        require(value is None,'Baseline must not fabricate neural-cache telemetry');return
    fields={'protocol','profile','counts','cache_capacity','cache_occupancy_before','cache_occupancy_after_reset','cache_occupancy_after','reset_ms','granted_search_ms','requested_ms','deadline_policy','logical_row_evaluations','physical_forward_executions','feature_passes_per_native_eval'}
    require(isinstance(value,dict) and set(value)==fields,'Unknown evaluator telemetry fields')
    require(value['protocol']=='ordinary-value-puct-counter-delta-v1' and value['profile']==profile and value['requested_ms']==decision['requested_ms'],'Evaluator profile/budget differs')
    counts=value['counts'];require(isinstance(counts,dict) and set(counts)==set(COUNTERS) and all(integer(x) for x in counts.values()),'Invalid physical counter delta')
    cap=0
    require(value['cache_capacity']==cap and all(integer(value[k],0,cap) for k in ('cache_occupancy_before','cache_occupancy_after_reset','cache_occupancy_after')) and value['cache_occupancy_after_reset']==0,'Cache carries prior decision state')
    require(counts['cache_resets']==1 and counts['evaluations_requested']==counts['cache_hits']+counts['cache_misses'],'Cache request conservation differs')
    passes=1
    require(counts['native_actual_evals']<=counts['native_calls']<=passes*counts['cache_misses'] and counts['predict_successes']<=counts['native_actual_evals'],'Actual native-forward accounting differs')
    require(counts['evaluations_requested']+counts['terminal_returns']<=counts['positions_submitted'],'Admitted counter prefix exceeds submitted rows')
    require(counts['terminal_returns']==0,'PUCT must not evaluate terminal leaves')
    if not cap:require(not counts['cache_hits'] and not counts['cache_evictions'] and not value['cache_occupancy_after'],'Disabled cache has hit/eviction')
    require(counts['cache_evictions']<=counts['native_actual_evals'] and value['cache_occupancy_after']<=counts['predict_successes'],'Cache occupancy/eviction exceeds actual computation')
    require(value['feature_passes_per_native_eval']==passes and counts['feature_simulations']<=14*passes*counts['native_calls'],'Feature work does not match physical passes')
    require(counts['feature_simulations']>=counts['native_actual_evals'] and counts['feature_deposits']>=counts['feature_simulations'],'Completed native forwards lack real feature work')
    require(numeric(value['reset_ms']) and integer(value['granted_search_ms'],0,value['requested_ms']) and value['granted_search_ms']<=max(0,math.floor(value['requested_ms']-value['reset_ms'])),'Cache reset grants extra deadline')
    require(value['deadline_policy']=='Floor remaining integer milliseconds after reset; no new budget' and value['physical_forward_executions']==counts['native_actual_evals'],'Incorrect deadline/physical label')
    report=decision['report'];require(value['logical_row_evaluations']==report['work']['neural_evaluations'],'Logical PUCT evaluations relabeled physical')
    require(report['work']['neural_evaluations']==counts['native_actual_evals'],'Uncached logical rows differ from physical forwards')
    require(report['work']['feature_simulations']==counts['feature_simulations'] and report['work']['feature_deposits']==counts['feature_deposits'],'PUCT and evaluator physical feature work disagree')

def empty_report(p,plan,name):
    legal=[a for a,c in enumerate(p['pits'][p['controller']]) if c];d=p['stores'][0]-p['stores'][1]
    return dict(schema=quality_search.SCHEMA,position=p,chosen=legal[0],visits=[0]*7,policy=[1/len(legal) if a in legal else 0 for a in range(7)],root_priors=[0.]*7,estimated_value_player0=d,action_mean_value_player0=[None]*7,root_expanded=False,
                completed_simulations=0,requested_simulations=plan['puct']['simulations'],nodes=1,max_nodes=plan['puct']['max_nodes'],stop_reason='time_budget',proof_status='monte_carlo_estimate',usable_decision=False,
                protocol=quality_search.PROTOCOL,rules_digest=config.RULES_DIGEST,value_perspective='fixed_player0_seeds',first_play_urgency='parent_neural_bootstrap',unvisited_root_q_player0=d,c_puct=plan['profiles'][name]['c_puct'],memory_mb=8192,
                work=dict(transitions=0,deposits=0,feature_simulations=0,feature_deposits=0,neural_evaluations=0,neural_calls=0,inference_ms=0.,root_expansions=0,total_deposits=0),
                timing=dict(elapsed_ms=0.,requested_time_ms=0,overshoot_ms=0.,whole_decision_includes_inference=True),forest_batch_calls=0,forest_size=1)


def legacy_view(decision):
    """Copied compatibility view only; original protocol/coefficient stay saved."""
    if decision['kind']!='puct':return decision
    report=prior.compact_puct(decision['report'])
    return {**decision,'report':{**report,'protocol':prior.PUCT_PROTOCOL,'c_puct':1.5}}

def decide(name,engine,p,ms,plan):
    started=time.perf_counter();kind,obj=engine
    if kind!='puct':return dict(decision=prior.decide(engine,p,ms,plan),evaluator=None)
    before=counter_snapshot(obj);tick=time.perf_counter();obj.reset_cache();reset_ms=(time.perf_counter()-tick)*1000;reset=counter_snapshot(obj)
    grant=max(0,math.floor(ms-(time.perf_counter()-started)*1000))
    if grant:
        raw=puct_search(p,obj,simulations=plan['puct']['simulations'],time_ms=grant,max_nodes=plan['puct']['max_nodes'],c_puct=plan['profiles'][name]['c_puct'],root_noise=None,memory_mb=8192)
        report=raw;elapsed=raw['timing']['elapsed_ms']
    else:
        remaining=ms/1000-(time.perf_counter()-started)
        if remaining>0:time.sleep(remaining)
        report=empty_report(p,plan,name);elapsed=0.
    after=counter_snapshot(obj);wall=(time.perf_counter()-started)*1000
    # A native search can correctly stop at the floored sub-budget before the
    # original fractional deadline. Retain that time, label budget rounding.
    if report['stop_reason']=='time_budget' and wall<ms:time.sleep((ms-wall)/1000)
    wall=(time.perf_counter()-started)*1000
    answer=dict(kind='puct',report=report,stop_reason=report['stop_reason'],engine_elapsed_ms=elapsed,nodes=report['nodes'],usable_decision=report['usable_decision'],chosen=report['chosen'] if report['usable_decision'] else None,
                completed_depth=None,completed_simulations=report['completed_simulations'],wall_ms=wall,requested_ms=ms,overshoot_ms=max(0.,wall-ms))
    telemetry=dict(protocol='ordinary-value-puct-counter-delta-v1',profile=name,counts={k:after[k]-before[k] for k in COUNTERS},cache_capacity=after['cache_capacity'],cache_occupancy_before=before['cache_occupancy'],cache_occupancy_after_reset=reset['cache_occupancy'],cache_occupancy_after=after['cache_occupancy'],reset_ms=reset_ms,granted_search_ms=grant,requested_ms=ms,
                   deadline_policy='Floor remaining integer milliseconds after reset; no new budget',logical_row_evaluations=report['work']['neural_evaluations'],physical_forward_executions=after['native_actual_evals']-before['native_actual_evals'],feature_passes_per_native_eval=1)
    result=dict(decision=answer,evaluator=telemetry);validate_envelope(result,p,ms,plan,name)
    wall=(time.perf_counter()-started)*1000;answer.update(wall_ms=wall,overshoot_ms=max(0.,wall-ms));return result

def validate_envelope(value,p,ms,plan,name):
    require(isinstance(value,dict) and set(value)=={'decision','evaluator'},'Unknown decision envelope')
    require(name in ENGINES,'Unknown value engine')
    decision=value['decision']
    if name in CANDIDATES:
        require(isinstance(decision,dict) and decision.get('kind')=='puct','Candidate search kind differs')
        quality_search.validate_report(decision['report'],p,plan['puct']['simulations'],plan['puct']['max_nodes'],c_puct=plan['profiles'][name]['c_puct'])
        require(decision['engine_elapsed_ms']==decision['report']['timing']['elapsed_ms'],'Value inner clock differs')
        require(decision['report']['timing']['requested_time_ms']==value['evaluator']['granted_search_ms'],'Value granted clock differs')
    prior.validate_decision(legacy_view(decision),p,ms,plan)
    require(value['decision']['kind']==('h' if name=='H_v1' else 'relay' if name=='H8_relay_CPU' else 'puct'),'Evaluator/search engine family differs')
    validate_telemetry(value['evaluator'],value['decision'],name)

def validate_case(body,index,plan):
    require(type(index) is int and 0<=index<len(plan['cases']),'Unknown case index');row=plan['cases'][index]
    fields={'index','state_key','cohort','stratum','reference','reference_status','reference_provenance','orders','measurements','cpu_verified'}
    require(isinstance(body,dict) and set(body)==fields and body['index']==index and all(body[k]==row[k] for k in ('state_key','cohort','stratum','reference','reference_status','reference_provenance')) and body['cpu_verified']==prior.verified(plan),'Case ownership/reference/CPU differs')
    require(set(body['orders'])==set(body['measurements'])=={str(ms) for ms in plan['budgets_ms']},'Case budgets differ')
    reference=row['reference'];q={entry['pit']:entry['value_player0'] for entry in reference['action_scores']} if reference else {};sign=1 if row['state']['controller']==0 else -1
    for bi,ms in enumerate(plan['budgets_ms']):
        values=body['measurements'][str(ms)];require(set(values)==set(plan['engines']) and body['orders'][str(ms)]==prior.expected_order(index,bi,plan),'Engine order/coverage differs')
        for name,value in values.items():
            require(set(value)=={'decision','reference_optimal','reference_regret'},'Unknown measurement envelope');validate_envelope(value['decision'],row['state'],ms,plan,name);choice=value['decision']['decision']['chosen']
            if choice is None or reference is None:require(value['reference_optimal'] is None and value['reference_regret'] is None,'Missing reference/fallback fabricated target')
            else:require(choice in q and type(value['reference_optimal']) is bool and value['reference_optimal']==(choice in reference['best_moves']) and value['reference_regret']==sign*(reference['target_player0']-q[choice]),'Choice regret/agreement differs')

def replay_game(body,comparison,pair,seat,plan,native):
    p=body['start']
    for action,row in zip(body['moves'],body['decisions']):
        p=native.normalize(flatten(p),actor=p['controller'],total=98)['position'];validate_envelope(row['decision'],p,plan['arena_ms'],plan,row['engine'])
        p=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position']
    stripped={**body,'decisions':[{**row,'decision':legacy_view(row['decision']['decision'])} for row in body['decisions']]}
    prior.replay_game(stripped,comparison,pair,seat,plan,native)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def pins_current(mapping):
    require(isinstance(mapping,dict) and mapping,'Missing input pins')
    for name,digest in mapping.items():
        require((ROOT/name).is_file() and sha(ROOT/name)==digest,'Changed value pilot input: '+name)


def baseline():
    value=load(BASELINE)
    require(value['schema']=='ordinary-value-prior-immutability-v1' and value['passed'] is True and value['inputs_count']==len(value['inputs'])==3364 and value['control_model_sha256']==MODEL_SHA,'Wrong ordinary value prior coverage')
    pins_current(value['inputs']);require(sha(CONTROL)==MODEL_SHA,'Fixed control model changed')
    return value



def prepare_projection():
    """Reserve true benchmark starts before self-play; no model update/outcome access."""
    baseline()
    import hashlib
    from . import pilot
    dataset=load(ROOT/'build/value-pilot-rows.json');require(set(dataset['groups'])=={'train','development','ordinary_development','diagnostic'},'Unexpected consumed dataset groups');dev=dataset['groups']['development']
    old_projection=load(ROOT/'build/selfplay-pilot-benchmark-projection.json');used=set(old_projection['prior_arena_families'])|{row['family'] for row in old_projection['arena_starts']}
    source_files=('build/value-pilot-rows.json','build/selfplay-pilot-benchmark-projection.json','data/ordinary-selfplay-pilot-v1/work.sqlite3','data/ordinary-selfplay-pilot-v1/collection-manifest.json')
    # Prefer contested positions so the small arena does not consist of decided games.
    candidates=[row for row in dev if row['family'] not in used and sum(map(sum,row['state']['pits']))>=25 and abs(row['state']['stores'][0]-row['state']['stores'][1])<sum(map(sum,row['state']['pits']))]
    arena=[];selected=set()
    for mechanism in ('ordinary','capture','bonus'):
        group=sorted((row for row in candidates if row['stratum'].endswith('/'+mechanism)),key=lambda row:hashlib.sha256(('ordinary-value-arena-v1|'+row['family']+'|'+row['state_key']).encode()).hexdigest())
        for row in group:
            if row['family'] in selected:continue
            arena.append({key:row[key] for key in ('state_key','state','family','stratum')});selected.add(row['family'])
            if sum(x['stratum'].endswith('/'+mechanism) for x in arena)==4:break
        require(sum(x['stratum'].endswith('/'+mechanism) for x in arena)==4,'Not enough unused development arena families')
    cases=[{**row,'cohort':'exact_guard','reference_status':'known_exact'} for row in old_projection['cases']]
    validation=[]
    with closing(sqlite3.connect((ROOT/source_files[2]).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        for key,text,digest in db.execute('SELECT state,body,hash FROM roots ORDER BY state'):
            row=json.loads(text);require(config.digest(row)==digest,'Corrupt reserved validation metadata')
            if row['cohort']=='validation':validation.append(row)
    require(len(validation)==6 and Counter(row['stratum'] for row in validation)=={'25-49/ordinary':3,'50-74/ordinary':3},'Reserved validation six/bands changed')
    fixture_row=dataset['groups']['ordinary_development'][0]
    fixture_case=dict(state_key=fixture_row['state_key'],state=fixture_row['state'],family=fixture_row['family'],stratum=fixture_row['stratum'],cohort='ordinary_validation_H10',reference=fixture_row['label'],reference_status='fixture_consumed_finite_reference',fixture_source=dict(path=source_files[0],group='ordinary_development',row=0,row_sha256=config.digest(fixture_row)))
    require(len(arena)==12 and len(selected)==12 and not selected&used and len(cases)==32,'Benchmark reservation geometry differs')
    roots=arena+cases+validation
    body=dict(schema='ordinary-value-benchmark-projection-v1',arena_starts=arena,cases=cases,validation_roots=validation,fixture_case=fixture_case,
        reserved_families=sorted({row['family'] for row in roots}),reserved_quotients=sorted({pilot.quotient(row['state']).hex() for row in roots}),
        prior_arena_families=sorted(used),baseline_sha256=sha(BASELINE),source_inputs={name:sha(ROOT/name) for name in source_files},
        selection='Deterministic SHA256 ranking, ordinary/capture/bonus four each, unique family, active>=25 and abs(store difference)<active; exclude all known previous arena IDs',
        novelty='New to previous arena family IDs only. All chosen roots are consumed development and may have ancestral training/model-selection exposure. Exact diagnostics are reused, not sealed release tests.',
        sealed_final_test_consumed=False,training_or_games_performed=False)
    body['sha256']=config.digest(body);common.publish(PROJECTION,body);return body

def arena_order(plan):
    names=[row['id'] for row in plan['arena_comparisons']];order=[]
    for pair in range(len(plan['arena_starts'])):
        shift=pair%len(names)
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):
                order.append(dict(comparison=name,pair=pair,seat=seat))
    return order


def expected_plan():
    projection=load(PROJECTION);require(projection['schema']=='ordinary-value-benchmark-projection-v1' and projection['sha256']==config.digest({k:v for k,v in projection.items() if k!='sha256'}),'Projection checksum differs')
    pins_current(projection['source_inputs']);require(projection['baseline_sha256']==sha(BASELINE),'Projection baseline differs')
    arena=projection['arena_starts'];cases=projection['cases'];used=set(projection['prior_arena_families']);families={row['family'] for row in arena}
    require(len(arena)==len(families)==12 and not families&used and len(cases)==32 and all(row['reference']['kind']=='exact_optimal_margin' for row in cases),'Reservation geometry changed')
    old=load(ROOT/'build/value-plan.json')
    return dict(schema='ordinary-value-benchmark-plan-v1',rules_digest=config.RULES_DIGEST,baseline_sha256=sha(BASELINE),prior_inputs=3364,projection_sha256=sha(PROJECTION),
        cases=cases,validation_roots=projection['validation_roots'],fixture_case=projection['fixture_case'],arena_starts=arena,models=dict(control='data/quality-train-v1/model.json',challenger='data/ordinary-value-pilot-v1/model.json'),control_model_sha256=MODEL_SHA,
        source_model=old['source_model'],source_model_sha256=old['source_model_sha256'],profiles={name:dict(model='control' if name=='ordinary_control' else 'challenger',c_puct=.25,cache_capacity=0) for name in CANDIDATES},
        engines=ENGINES,budgets_ms=[25,100],arena_ms=25,arena_max_choices=512,max_depth=16,tt_mb=16,use_pvs=True,memory_mb=8192,cpu_declared=old['cpu_declared'],
        puct=dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap'),bootstrap_samples=4096,bootstrap_seed=20261029,
        declared_geometry=dict(cases=38,exact_cases=32,validation_cases=6,scalar_measurements=304,exact_scalar_measurements=256,validation_scalar_measurements=48,arena_families=12,games=96),
        validation_reference=dict(primary_requested_depth=10,roots=6,attempted_depths=[8,10,12],requested_node_caps=[2000000,5000000,10000000],missing='Keep attempted case and allengine recommendations; reference=None and accuracy/regret=None. Score only complete requestedH10 intersection, no auxiliary fallback.',accuracy_scope='Six consumed protected development roots, finite-horizon estimates, not exact or release truth'),
        allocation_contract=dict(engineering_failure='hold',engineering_requires=['source_and_model_policy_data_audit','all96terminal_zero_fallback','exact_nonregression_both_clocks','zero_recorded_posthoc_active_gt4_train_arena_quotient_or_family_collision'],
            favorable_requires=['all6_requested_H10_complete','validation_candidate_MAE_le_control','both_paired_mean_candidate_minus_control_ge0'],
            favorable_branch='new_starts',favorable_budget=dict(families=512,training_starts_high=12,training_starts_low=12,runs_per_start=2,maximum_games=48),
            other_engineering_valid_branch='repeat_high',other_budget=dict(original_high_starts=12,additional_runs_per_start=2,maximum_games=24),
            interpretation='Collection allocation only, not model promotion or statistical proof. Fixed controls, no automatic further training or extra arena/model selection.'),
        primary='12 terminal paired families with zero fallback; lower95% paired seed-margin bootstrap above0 against BOTH baselines; exploratory only',
        attribution='Contemporaneous challenger-minus-control matched paired-family margins against each baseline, including descriptive paired bootstrap intervals; six unadjusted intervals total',
        selection='One fixed final value-head checkpoint; six trunk/policy tensors bit-exact. No model, family, search, time or checkpoint selection from benchmark outcomes',
        novelty=projection['novelty'],clock='Whole decision including reset, features, inference and validation; floor positive remaining milliseconds; no time_ms zero; retain overshoot',
        recovery='Each decision commits FULL/WAL with exact before/after state; resume from committed next-game prefix; complete game is separate atomic commit',
        exact_guard='32 reused known-exact K<=4 roots at25/100ms, every engine; candidate/control completed choices, exact-optimal counts and actor-signed regret; no release-proof claim',
        release='Small consumed-development pilot. No automatic promotion, sealed final access or energy claim; H_v1 remains default.',
        default_promotion=False,sealed_final_test_consumed=False,energy_measured=False)


def make_plan():
    baseline();value=expected_plan();common.publish(PLAN,value);return value


def read_plan(stage='evaluation',fixture=False):
    require(stage=='evaluation' and type(fixture) is bool,'Unknown benchmark stage/fixture');full=load(PLAN);require(full==expected_plan(),'Frozen ordinary value evaluation schedule changed')
    p={**full,'stage':stage,'full_plan_sha256':config.digest(full),'fixture':bool(fixture)}
    exact=[{**row,'reference_provenance':dict(source='sealed_known_exact_reference',projection_path=str(PROJECTION.relative_to(ROOT)),case=index,reference_sha256=config.digest(row['reference']))} for index,row in enumerate(full['cases'])]
    if fixture:
        row=full['fixture_case'];finite={**row,'reference_provenance':dict(source='explicit_consumed_finite_fixture_only',**row['fixture_source'])}
        p.update(cases=exact[:1]+[finite],arena_starts=full['arena_starts'][:1],validation_labels_witness=dict(fixture=True,production_labels_read=False))
    else:
        from . import ordinary_value_pilot as learning
        roots,stores,manifest=learning.load_labels();expected={row['state_key']:row for row in full['validation_roots']};require(set(roots)==set(expected),'Validation label panel changed')
        validation=[]
        for key,original in sorted(expected.items()):
            wrapper=roots[key];require(wrapper['state']==original['state'] and wrapper['root']==original and wrapper['source_root_sha256']==config.digest(original),'Validation raw orientation/ownership changed')
            item=stores['h10_labels'].get(key);attempt=stores['h10_attempts'][key]
            if item:
                require(item['label']['choice_depth']==item['report']['choice_depth']==10 and item['root_sha256']==wrapper['source_root_sha256'],'RequestedH10 original source-root link differs')
            validation.append(dict(state_key=key,state=original['state'],family=original['family'],stratum=original['stratum'],cohort='ordinary_validation_H10',reference=item['label'] if item else None,reference_status='complete_requested_H10' if item else 'missing_requested_H10',
                reference_provenance=dict(source='closed_requested_H10_only',path=(learning.LABEL_DEFAULT/'work.sqlite3').relative_to(ROOT).as_posix(),table='h10_labels',state_key=key,source_root_sha256=wrapper['source_root_sha256'],study_wrapper_sha256=config.digest(wrapper),label_root_sha256=item['root_sha256'] if item else None,attempt_sha256=config.digest(attempt),label_item_sha256=config.digest(item) if item else None,manifest_sha256=config.digest(manifest),requested_depth=10,auxiliary_fallback=False)))
        p.update(cases=exact+validation,validation_labels_witness=dict(fixture=False,manifest_sha256=config.digest(manifest),journal_sha256=sha(learning.LABEL_DEFAULT/'work.sqlite3'),requested_H10_complete=sum(row['reference'] is not None for row in validation),requested_H10_missing=sum(row['reference'] is None for row in validation)))
    p['arena_comparisons']=[dict(id=name+'_vs_'+base,candidate=name,opponent=base,pairs=len(p['arena_starts'])) for name in CANDIDATES for base in BASELINES]
    p['arena_work_order']=arena_order(p);return p


def implementation_body():
    sources=('congklak/ordinary_value_benchmark.py','tools/benchmark-ordinary-value.py','tests/ordinary_value_benchmark_author.py','congklak/ordinary_value_progress.py','tools/ordinary-value-dashboard.html')
    science=('build/ordinary-value-baseline.json','build/ordinary-value-benchmark-projection.json','build/ordinary-value-benchmark-plan.json','docs/ORDINARY_VALUE_SCOPE.md')
    dependencies=('congklak/quality_search.py','congklak/puct_cpu_inference.py','congklak/ordinary_value_followup.py','tools/ordinary-value-followup.py')
    body=dict(schema='ordinary-value-benchmark-implementation-v1',sources={name:sha(ROOT/name) for name in sources},science_inputs={name:sha(ROOT/name) for name in science},dependencies={name:sha(ROOT/name) for name in dependencies},rules_digest=config.RULES_DIGEST)
    return {**body,'sha256':config.digest(body)}


def freeze():
    make_plan();body=implementation_body();common.publish(IMPLEMENTATION,body);return body


def verify_candidate(control,candidate):
    require(control.width==candidate.width==64 and control.input_count==candidate.input_count==127 and control.encoder==candidate.encoder,'Candidate architecture/encoder differs')
    names=('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','layers.4.weight','layers.4.bias','policy.weight','policy.bias')
    require(set(candidate.weights)==set(control.weights)==set(names),'Joint model tensor set differs')
    from .neural_relay import flatten_weights
    def packed(model,name):
        values=list(flatten_weights(model.weights[name]));return struct.pack('<'+'f'*len(values),*values)
    frozen=names[:4]+names[6:];head=names[4:6]
    require(all(packed(control,key)==packed(candidate,key) for key in frozen),'Frozen trunk or policy changed')
    changed=[key for key in head if packed(control,key)!=packed(candidate,key)];require(changed,'Candidate value head has no changed tensor')
    import hashlib
    frozen_sha=hashlib.sha256(b''.join(key.encode()+packed(control,key) for key in frozen)).hexdigest()
    return dict(architecture='127x64x64 shared trunk; value and7policy outputs',parameters=12872,trainable_parameters=65,frozen_parameters=12807,frozen_tensor_names=list(frozen),trainable_tensor_names=list(head),frozen_tensors_fp32_sha256=frozen_sha,changed_tensors=changed,global_policy_unchanged_by_identical_subgraph=True)


def pretraining_witness():
    """Deterministic source/schedule reservation gate, without games or Torch imports."""
    baseline();plan=load(PLAN);require(plan==expected_plan(),'Frozen blind evaluation plan differs');body=implementation_body();require(load(IMPLEMENTATION)==body,'Benchmark source freeze differs')
    return dict(schema='ordinary-value-pretraining-source-witness-v1',passed=True,plan_file_sha256=sha(PLAN),projection_file_sha256=sha(PROJECTION),
        implementation_file_sha256=sha(IMPLEMENTATION),implementation_sha256=body['sha256'],control_model_sha256=MODEL_SHA,control_weights_sha256=Model(CONTROL).id,
        reserved_families=load(PROJECTION)['reserved_families'],reserved_quotients=load(PROJECTION)['reserved_quotients'],
        scheduled_games=96,scheduled_exact_measurements=256,scheduled_validation_measurements=48,pretraining_games_required=False,games_performed_by_gate=False,label_outcomes_consumed_by_gate=False,torch_imported_by_gate=False)


def admit(plan,fixture_control=False):
    baseline();body=implementation_body()
    if IMPLEMENTATION.exists():require(load(IMPLEMENTATION)==body,'Benchmark frozen source/science differs')
    else:require(plan['fixture'] and fixture_control,'Benchmark source freeze is missing')
    require(plan==read_plan(plan['stage'],plan['fixture']),'Unregistered benchmark configuration')
    control=Model(CONTROL);require(control.file_sha256==MODEL_SHA,'Control model changed')
    proof=dict(schema='ordinary-value-benchmark-admission-v1',fixture=plan['fixture'],prior_accepted_inputs=3364,control_model_sha256=control.file_sha256,control_weights_sha256=control.id,
        source_freeze_admitted=IMPLEMENTATION.exists(),fixture_model_substitution=bool(fixture_control))
    if fixture_control:require(plan['fixture'],'Model substitution is forbidden in production');proof['candidate_model_sha256']=control.file_sha256
    else:
        from .ordinary_value_pilot import witness as model_witness
        checked=model_witness(PILOT.parent);candidate=Model(PILOT)
        proof.update(model_witness=checked,model_contract=verify_candidate(control,candidate),pretraining=pretraining_witness(),candidate_model_sha256=candidate.file_sha256)
    return body,proof


def trajectory_overlap(db,identity,plan):
    """Posthoc actual-replay collision check; never a future holdout guarantee."""
    if plan['fixture']:
        return dict(fixture=True,checked=False,reason='Explicit control-substitution fixture; no actual training-replay comparison')
    from .ordinary_value_pilot import prepare
    from . import pilot
    manifest,groups=prepare();require(set(groups)=={'high','low','rehearsal'},'Unexpected admitted optimizer pools');rows=[row for name in ('high','low','rehearsal') for row in groups[name]]
    train_q={pilot.quotient(row['state']).hex() for row in rows};train_f={row['family'] for row in rows};hits=[];observed=[];families=set()
    for game_index,text in db.execute('SELECT idx,body FROM games ORDER BY idx'):
        game=json.loads(text);families.add(game['family'])
        for choice,record in enumerate(records_for(db,game_index)):
            p=record['state']
            if sum(map(sum,p['pits']))<=4:continue
            q=pilot.quotient(p).hex();observed.append(q)
            if q in train_q or game['family'] in train_f:hits.append(dict(game=game_index,choice=choice,comparison=game['comparison'],pair=game['pair'],candidate_seat=game['candidate_seat'],family=game['family'],quotient=q,quotient_collision=q in train_q,family_collision=game['family'] in train_f))
    return dict(fixture=False,checked=True,actual_admitted_rows=len(rows),actual_admitted_families=len(train_f),actual_admitted_quotients=len(train_q),
        training_manifest_sha256=config.digest(manifest),training_rows_file_sha256=sha(ROOT/'build/ordinary-value-rows.json'),
        arena_nonterminal_gt4_observations=len(observed),arena_unique_quotients=len(set(observed)),arena_families=len(families),
        colliding_observations=len(hits),colliding_quotients=len({row['quotient'] for row in hits if row['quotient_collision']}),colliding_family_ids=sorted(families&train_f),collisions=hits,
        contaminated_evaluation=bool(hits),descriptive_only=True,
        interpretation='Known evaluation trajectories were protected before training; future arena trajectories were unknowable. Any actual replay collision flags contaminated evaluation. Absence of these collisions does not erase ancestral/development exposure.')


def validation_model_fit(plan):
    """Literal exported-weight CPU verification; no inference worker or search."""
    models={'control':Model(CONTROL),'candidate':Model(CONTROL if plan['fixture'] else PILOT)};rows=[]
    targets=[row for row in plan['cases'] if row['cohort']=='ordinary_validation_H10']
    for row in targets:
        reference=row['reference'];predictions={name:model.reference(row['state']) for name,model in models.items()}
        require(predictions['control'][2]==predictions['candidate'][2],'Literal frozen policy changed')
        rows.append(dict(state_key=row['state_key'],stratum=row['stratum'],reference_status=row['reference_status'],reference_sha256=config.digest(reference) if reference else None,requested_reference_depth=10,reference_target_player0=reference['target_player0'] if reference else None,
            predictions={name:dict(value_player0=value[0],absolute_seed_error=abs(value[0]-reference['target_player0']) if reference else None) for name,value in predictions.items()}))
    complete=[row for row in rows if row['reference_target_player0'] is not None]
    return dict(requested_roots=len(targets),complete_reference_roots=len(complete),missing_reference_roots=len(targets)-len(complete),
        control_mae=statistics.mean(row['predictions']['control']['absolute_seed_error'] for row in complete) if complete else None,
        candidate_mae=statistics.mean(row['predictions']['candidate']['absolute_seed_error'] for row in complete) if complete else None,rows=rows,
        method='Literal CPU Model.reference with exported FP32 weights and exact relay features; deterministic validation recheck, no CpuBatchEvaluator, Torch, timed inference or search. RequestedH10 only, missing references excluded explicitly.',
        interpretation='Consumed protected finite-H10 development estimates, not exact truth or population/generalization proof; a complete panel is six roots, not a confidence claim.')


def summarize(db,identity,plan):
    buckets=defaultdict(list)
    for text, in db.execute('SELECT body FROM cases ORDER BY idx'):
        body=json.loads(text)
        for ms,engines in body['measurements'].items():
            for name,value in engines.items():buckets[(body['cohort'],ms,name)].append(value)
    cases={}
    for key,rows in buckets.items():
        decisions=[row['decision']['decision'] for row in rows];usable=[row for row in rows if row['decision']['decision']['usable_decision']];scored=[row for row in usable if row['reference_regret'] is not None]
        counts={name:sum((row['decision']['evaluator'] or {}).get('counts',{}).get(name,0) for row in rows) for name in COUNTERS} if key[2] in CANDIDATES else None
        cases['/'.join(key)]=dict(positions=len(rows),no_completed_recommendation=len(rows)-len(usable),scored_complete_reference_choices=len(scored),unscored_complete_choices=len(usable)-len(scored),reference_optimal_accuracy=statistics.mean(row['reference_optimal'] for row in scored) if scored else None,
            mean_reference_regret_seeds=statistics.mean(row['reference_regret'] for row in scored) if scored else None,median_wall_ms=statistics.median(row['wall_ms'] for row in decisions),max_wall_ms=max(row['wall_ms'] for row in decisions),max_overshoot_ms=max(row['overshoot_ms'] for row in decisions),
            mean_completed_simulations=statistics.mean(row['completed_simulations'] for row in decisions) if decisions[0]['kind']=='puct' else None,evaluator_counts=counts)
    arenas={};paired={}
    for comp in plan['arena_comparisons']:
        games=[json.loads(text) for text, in db.execute('SELECT body FROM games WHERE comparison=? ORDER BY pair,seat',(comp['id'],))];values=[];scores=[];paired[comp['id']]={}
        for pair in range(comp['pairs']):
            group=[game for game in games if game['pair']==pair]
            if len(group)==2 and all(game['status']=='terminal' for game in group):
                mean=statistics.mean(game['candidate_final_margin'] for game in group);values.append(mean);paired[comp['id']][pair]=mean;scores.append(statistics.mean(game['candidate_match_score'] for game in group))
        ci=prior.interval(values,plan);fallbacks={name:sum(game['fallbacks'].get(name,0) for game in games) for name in (comp['candidate'],comp['opponent'])}
        arenas[comp['id']]=dict(pairs=comp['pairs'],complete_pairs=len(values),incomplete_pairs=comp['pairs']-len(values),terminal_games=sum(game['status']=='terminal' for game in games),unfinished_games=sum(game['status']!='terminal' for game in games),
            candidate_wins=sum(game['candidate_match_score']==1 for game in games),candidate_draws=sum(game['candidate_match_score']==.5 for game in games),candidate_losses=sum(game['candidate_match_score']==0 for game in games),paired_mean_score=statistics.mean(scores) if scores else None,
            primary_paired_mean_final_margin_seeds=statistics.mean(values) if values else None,primary_margin_bootstrap_95_interval=ci,primary_strength_gate_passed=bool(len(values)==comp['pairs'] and not sum(fallbacks.values()) and ci and ci[0]>0),fallback_counts=fallbacks)
    gates={name:all(arenas[name+'_vs_'+base]['primary_strength_gate_passed'] for base in BASELINES) for name in plan['profiles']};contrasts={}
    if plan['stage']=='evaluation':
        for base in BASELINES:
            treatment=paired['ordinary_value_vs_'+base];control=paired['ordinary_control_vs_'+base];matched=sorted(set(treatment)&set(control));expected=len(plan['arena_starts'])
            contrasts[base]=dict(treatment='ordinary_value',control='ordinary_control',expected_families=expected,complete_matched_families=len(matched),paired_mean_seed_margin_difference=statistics.mean(treatment[pair]-control[pair] for pair in matched) if len(matched)==expected else None,confidence_interval=prior.interval([treatment[pair]-control[pair] for pair in matched],plan) if len(matched)==expected else None,primary_gate=False)
    guard={}
    for ms in plan['budgets_ms']:
        group={}
        for name in CANDIDATES:
            values=[row for (cohort,budget,profile),rows in buckets.items() if cohort=='exact_guard' and budget==str(ms) and profile==name for row in rows];usable=[row for row in values if row['decision']['decision']['usable_decision']]
            group[name]=dict(roots=len(values),complete_choices=len(usable),optimal_choices=sum(bool(row['reference_optimal']) for row in usable),mean_exact_regret=statistics.mean(row['reference_regret'] for row in usable) if usable else None)
        a,b=group['ordinary_control'],group['ordinary_value']
        expected=sum(row['cohort']=='exact_guard' for row in plan['cases'])
        group['non_regression_observed']=bool(expected and a['complete_choices']==b['complete_choices']==expected and b['optimal_choices']>=a['optimal_choices'] and b['mean_exact_regret']<=a['mean_exact_regret'])
        guard[str(ms)]=group
    protocol_valid=all(row['terminal_games']==2*row['pairs'] and not row['unfinished_games'] and not sum(row['fallback_counts'].values()) for row in arenas.values())
    return dict(schema='ordinary-value-benchmark-results-v1',identity=identity,stage=plan['stage'],fixture=plan['fixture'],cases=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],games=db.execute('SELECT COUNT(*) FROM games').fetchone()[0],decisions=db.execute('SELECT count(*) FROM decisions').fetchone()[0],scalar_measurements=sum(len(rows) for rows in buckets.values()),case_results=cases,arenas=arenas,both_baseline_primary_gates=gates,secondary_factor_contrasts=contrasts,endgame_guard=guard,validation_model_fit=validation_model_fit(plan),terminal_zero_fallback_gate=protocol_valid,trajectory_replay_overlap=trajectory_overlap(db,identity,plan),default_promotion=False,sealed_final_test_consumed=False,energy_measured=False,development_reuse=True,scope=plan['release'])


def allocation_from_report(report,proofs_valid):
    """Frozen data-allocation rule, deliberately separate from promotion gates."""
    require(type(proofs_valid) is bool,'Engineering proof flag must be Boolean')
    fit=report['validation_model_fit'];gains={name:report['secondary_factor_contrasts'][name]['paired_mean_seed_margin_difference'] for name in BASELINES}
    exact=all(report['endgame_guard'][str(ms)]['non_regression_observed'] for ms in (25,100));fallbacks=sum(sum(row['fallback_counts'].values()) for row in report['arenas'].values())
    overlap=report['trajectory_replay_overlap'];engineering=bool(proofs_valid and report['terminal_zero_fallback_gate'] and report['games']==96 and exact and not fallbacks and overlap.get('checked') and not overlap['contaminated_evaluation'])
    complete=fit['requested_roots']==fit['complete_reference_roots']==6
    favorable=bool(engineering and complete and fit['candidate_mae'] is not None and fit['candidate_mae']<=fit['control_mae'] and all(value is not None and value>=0 for value in gains.values()))
    branch='hold' if not engineering else 'new_starts' if favorable else 'repeat_high'
    return dict(engineering_valid=engineering,primary_h10_complete=complete,validation_mae_control=fit['control_mae'],validation_mae_candidate=fit['candidate_mae'],baseline_mean_gains=gains,exact_nonregression=exact,fallbacks=fallbacks,branch=branch,
        interpretation='Bounded collection allocation only. Nonnegative means and six finite-H10 roots do not establish strength, release readiness or model promotion; no subsequent training is authorized.')


def allocation_decision():
    witness=closed_witness();report=load(paths()[1]);proof_inputs={};proofs_valid=True
    for group in ('preflight','labels','model','recovery'):
        path=ROOT/f'build/ordinary-value-audit/{group}.json';body=load(path)
        require(body['schema']=='ordinary-value-independent-group-v1' and body['passed'] is True and not body.get('failures',0) and not body.get('errors',0) and not body.get('skips',0),'Required independent engineering group failed: '+group)
        pins_current(body['inputs']);proof_inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    decision=allocation_from_report(report,proofs_valid)
    return dict(schema='ordinary-value-bounded-allocation-v1',**decision,closed_benchmark_witness=witness,benchmark_report_sha256=sha(paths()[1]),plan_sha256=sha(PLAN),projection_sha256=sha(PROJECTION),implementation_sha256=sha(IMPLEMENTATION),engineering_group_inputs=proof_inputs)


def emit(body):
    print(config.canonical(body),flush=True)
    event=os.environ.get('ORDINARY_VALUE_BLOCK_EVENT');step=os.environ.get('ORDINARY_VALUE_BLOCK_STEP');marker=os.environ.get('ORDINARY_VALUE_BLOCK_MARKER')
    if event and body.get('event')==event and (step is None or int(step)==body.get('completed')):
        if marker:
            path=Path(marker);path.parent.mkdir(parents=True,exist_ok=True);common.teacher.atomic_json(path,body)
        while True:time.sleep(.05)


def validate_record(body,game_index,choice_index,p,unit,comp,plan,native):
    fields={'game_index','choice_index','comparison','pair','candidate_seat','state','action','row','after'}
    require(isinstance(body,dict) and set(body)==fields and all(type(body[name]) is int for name in ('game_index','choice_index','pair','candidate_seat','action')) and body['game_index']==game_index and body['choice_index']==choice_index and body['comparison']==comp['id'] and body['pair']==unit['pair'] and body['candidate_seat']==unit['seat'],'Decision journal identity differs')
    normal=native.normalize(flatten(p),actor=p['controller'],total=98);require(not normal['terminal'],'Decision after terminal state');p=normal['position'];row=body['row']
    expected_engine=comp['candidate'] if p['controller']==unit['seat'] else comp['opponent']
    require(config.canonical(body['state'])==config.canonical(p) and set(row)=={'engine','actor','fallback','decision'} and row['engine']==expected_engine and type(row['actor']) is int and row['actor']==p['controller'] and type(row['fallback']) is bool,'Decision actor/engine/state differs')
    validate_envelope(row['decision'],p,plan['arena_ms'],plan,expected_engine)
    fallback=not row['decision']['decision']['usable_decision'];action=normal['legal_actions'][0] if fallback else row['decision']['decision']['chosen']
    require(row['fallback']==fallback and body['action']==action,'Decision action/fallback differs')
    after=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position'];require(config.canonical(body['after'])==config.canonical(after),'Decision journal transition differs')
    return after


def records_for(db,index):
    records=[]
    for expected,(choice,text,digest) in enumerate(db.execute('SELECT choice,body,hash FROM decisions WHERE game=? ORDER BY choice',(index,))):
        body=json.loads(text);require(choice==expected and config.digest(body)==digest,'Corrupt/noncontiguous decision prefix');records.append(body)
    return records


def validate_saved_units(db,plan,native):
    comparisons={row['id']:row for row in plan['arena_comparisons']};order=plan['arena_work_order']
    for expected,(index,text,digest) in enumerate(db.execute('SELECT idx,body,hash FROM cases ORDER BY idx')):
        require(index==expected and index<len(plan['cases']) and config.digest(json.loads(text))==digest,'Corrupt/noncontiguous pilot case prefix');validate_case(json.loads(text),index,plan)
    games=[]
    for expected,(index,name,pair,seat,text,digest) in enumerate(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx')):
        require(index==expected and expected<len(order) and order[expected]==dict(comparison=name,pair=pair,seat=seat) and config.digest(json.loads(text))==digest,'Corrupt/orphan pilot game prefix')
        body=json.loads(text);replay_game(body,comparisons[name],pair,seat,plan,native);games.append(body)
    ids=[index for index, in db.execute('SELECT DISTINCT game FROM decisions ORDER BY game')]
    require(all(integer(index,0,min(len(games),len(order)-1)) for index in ids) and all(index in ids for index in range(len(games))),'Decision game prefix/orphan differs')
    for index in ids:
        unit=order[index];comp=comparisons[unit['comparison']];p=plan['arena_starts'][unit['pair']]['state'];records=records_for(db,index)
        require(0<len(records)<=plan['arena_max_choices'],'Decision prefix bounds differ')
        for choice,body in enumerate(records):p=validate_record(body,index,choice,p,unit,comp,plan,native)
        if index<len(games):
            game=games[index];require(game['moves']==[body['action'] for body in records] and game['decisions']==[body['row'] for body in records] and game['final']==p,'Game differs from its durable decision ledger')
    return db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],len(games)


def paths(stage='evaluation'):
    require(stage=='evaluation','Unknown stage path');return ROOT/'data/ordinary-value-benchmark-v1',ROOT/'build/ordinary-value-benchmark.json'


def validate_resources(body,report_path,implementation,plan,expected_report_sha256=None):
    fields={'schema','implementation_sha256','report_file_sha256','stage','fixture','elapsed_seconds_this_invocation','prior_committed_cases','prior_committed_games','prior_committed_decisions','host','torch_imported','energy_measured','scope'}
    require(isinstance(body,dict) and set(body)==fields and body['schema']=='ordinary-value-benchmark-resources-v1','Unknown resource observation')
    require(body['implementation_sha256']==implementation['sha256'] and body['report_file_sha256']==(expected_report_sha256 or sha(report_path)) and body['stage']==plan['stage'] and body['fixture']==plan['fixture'],'Resource identity differs')
    require(numeric(body['elapsed_seconds_this_invocation']) and integer(body['prior_committed_cases'],0,len(plan['cases'])) and integer(body['prior_committed_games'],0,len(plan['arena_work_order'])) and integer(body['prior_committed_decisions'],0,len(plan['arena_work_order'])*plan['arena_max_choices']),'Invalid resource clock/prefix')
    require(body['torch_imported'] is False and body['energy_measured'] is False and body['scope']=='This CPU-only invocation including admission/startup/remaining units/publication; earlier invocations excluded; no energy claim','Invalid resource claim')
    host=body['host'];require((host is None and os.name!='nt') or (isinstance(host,dict) and set(host)=={'peak_working_set_bytes','peak_private_commit_bytes'} and all(integer(value,1) for value in host.values())),'Invalid process memory observation');return body


def host_peak():
    if os.name!='nt':return None
    class Counters(C.Structure):
        _fields_=[('cb',C.c_ulong),('faults',C.c_ulong)]+[(key,C.c_size_t) for key in ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    kernel=C.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=C.c_void_p;psapi=C.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(Counters),C.c_ulong]
    value=Counters();value.cb=C.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),C.byref(value),value.cb):raise C.WinError(C.get_last_error())
    return dict(peak_working_set_bytes=value.peak,peak_private_commit_bytes=value.peak_pagefile)


def validate_complete(db,identity,plan,implementation,output,marker):
    cases,games=validate_saved_units(db,plan,Native());decisions=db.execute('SELECT count(*) FROM decisions').fetchone()[0]
    fields={'cases','games','decisions','report_sha256','resource_file_sha256'}
    require(isinstance(marker,dict) and set(marker)==fields and all(type(marker[name]) is int for name in ('cases','games','decisions')) and marker['cases']==cases==len(plan['cases']) and marker['games']==games==len(plan['arena_work_order']) and marker['decisions']==decisions,'Incomplete or altered benchmark completion marker')
    result=summarize(db,identity,plan);resource_path=Path(output).with_name(Path(output).stem+'-resources.json')
    require(config.digest(result)==marker['report_sha256'] and resource_path.is_file() and sha(resource_path)==marker['resource_file_sha256'],'Closed results/resource checksum differs')
    if Path(output).exists():require(load(output)==result,'Closed publication differs')
    import hashlib
    expected_file_sha=hashlib.sha256((config.canonical(result)+os.linesep).encode('utf-8')).hexdigest()
    validate_resources(load(resource_path),output,implementation,plan,expected_file_sha);return result


def _metadata(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone()
    require(row is not None,'Missing committed metadata: '+key);body=json.loads(row[0]);require(config.digest(body)==row[1],'Corrupt metadata: '+key);return body


def closed_witness(stage='evaluation'):
    plan=read_plan(stage);implementation,admission=admit(plan);path,output=paths(stage)
    require((path/'work.sqlite3').is_file() and output.is_file(),'Benchmark is not published complete')
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');identity=_metadata(db,'identity');marker=_metadata(db,'ordinary_value_benchmark_complete')
        fields={'schema','stage','fixture','plan_sha256','admission','implementation','implementation_file_sha256','cpu','baseline_dll_sha256','source_model_sha256','model_files','memory_mb_all_engines'}
        expected_identity={**identity,'schema':'ordinary-value-benchmark-v1','stage':'evaluation','fixture':False,'plan_sha256':config.digest(plan),'admission':admission,'implementation':implementation,'implementation_file_sha256':sha(IMPLEMENTATION)}
        require(set(identity)==fields and identity==expected_identity and identity['memory_mb_all_engines']==8192 and identity['model_files']=={name:sha(ROOT/plan['models'][profile['model']]) for name,profile in plan['profiles'].items()} and identity['baseline_dll_sha256']==Native().build_id and identity['source_model_sha256']==plan['source_model_sha256'],'Closed benchmark identity differs')
        result=validate_complete(db,identity,plan,implementation,output,marker)
    return dict(schema='ordinary-value-benchmark-closed-witness-v1',passed=True,journal_sha256=sha(path/'work.sqlite3'),report_file_sha256=sha(output),cases=result['cases'],games=result['games'],decisions=result['decisions'],scalar_measurements=result['scalar_measurements'])


def run(stage='evaluation',path=None,output=None,case_limit=None,no_arena=False,arena_limit=None,on_event=emit,fixture=False,fixture_control=False,decision_limit=None):
    invocation_started=time.perf_counter();require('torch' not in sys.modules,'CPU benchmark invocation has Torch loaded');plan=read_plan(stage,fixture);implementation,witness=admit(plan,fixture_control);default_path,default_output=paths(stage)
    path=Path(path) if path is not None else default_path;output=Path(output) if output is not None else default_output
    if fixture:require(path.resolve().is_relative_to((ROOT/'build').resolve()) and output.resolve().is_relative_to((ROOT/'build').resolve()),'Fixture paths must stay under build')
    else:require(path.resolve()==default_path.resolve() and output.resolve()==default_output.resolve() and not fixture_control,'Unregistered production paths/model substitution')
    cases_total=len(plan['cases']);games_total=len(plan['arena_work_order'])
    if case_limit is not None:require(integer(case_limit,0,cases_total),'Invalid case limit')
    if arena_limit is not None:require(integer(arena_limit,0,games_total),'Invalid arena limit')
    if decision_limit is not None:require(fixture and integer(decision_limit,1,games_total*plan['arena_max_choices']),'Decision limit is fixture-only')
    with ExitStack() as stack:
        cpu=stack.enter_context(PerformanceCore(plan['cpu_declared']));native=Native()
        model_files={name:MODEL_SHA if fixture_control else sha(ROOT/plan['models'][profile['model']]) for name,profile in plan['profiles'].items()}
        identity=dict(schema='ordinary-value-benchmark-v1',stage=stage,fixture=fixture,plan_sha256=config.digest(plan),implementation=implementation,implementation_file_sha256=sha(IMPLEMENTATION) if IMPLEMENTATION.exists() else None,admission=witness,cpu=cpu.metadata,
            baseline_dll_sha256=native.build_id,source_model_sha256=plan['source_model_sha256'],model_files=model_files,memory_mb_all_engines=8192)
        work=stack.enter_context(closing(phase6_store.Work(path,identity)));db=work.db
        db.executescript('CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS games(idx INTEGER PRIMARY KEY,comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT,UNIQUE(comparison,pair,seat));CREATE TABLE IF NOT EXISTS decisions(game INTEGER,choice INTEGER,body TEXT,hash TEXT,PRIMARY KEY(game,choice));CREATE TABLE IF NOT EXISTS setup(id INTEGER PRIMARY KEY,body TEXT,hash TEXT);')
        count,game_count=validate_saved_units(db,plan,native);initial_cases=count;initial_games=game_count;initial_decisions=db.execute('SELECT count(*) FROM decisions').fetchone()[0]
        marker=work.get('ordinary_value_benchmark_complete')
        if marker is not None:
            result=validate_complete(db,identity,plan,implementation,output,marker);common.publish(output,result);on_event(dict(event='ordinary_value_benchmark_already_complete',cases=count,games=game_count,decisions=initial_decisions));return result
        from .puct_cpu_inference import CpuBatchEvaluator
        source=stack.enter_context(NeuralNative(ROOT/plan['source_model'],mode='value-none'));engines={'H_v1':('h',native),'H8_relay_CPU':('relay',source)}
        require(source.model.file_sha256==plan['source_model_sha256'],'H8 relay baseline changed')
        setup_row=db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone();warmup={}
        require(db.execute('SELECT count(*) FROM setup').fetchone()[0] in ((1,) if setup_row else (0,)) and (setup_row is not None or not (count or game_count or initial_decisions)),'Invalid startup witness')
        for name,profile in plan['profiles'].items():
            model=CONTROL if fixture_control else ROOT/plan['models'][profile['model']]
            obj=stack.enter_context(CpuBatchEvaluator(model,cache_capacity=0,max_batch=32,backend='auto'));engines[name]=('puct',obj)
            cpu.verify();obj.reset_cache();tick=time.perf_counter();obj.evaluate_many([row['state'] for row in plan['cases'][:8]]);obj.reset_cache();warmup[name]=dict(startup_ms=obj.startup_ms,warmup_ms=(time.perf_counter()-tick)*1000,counters=counter_snapshot(obj),identity=obj.identity)
        if setup_row:require(config.digest(json.loads(setup_row[0]))==setup_row[1],'Corrupt startup observation')
        else:
            setup=dict(cpu_observations=cpu.observations,verified=cpu.verify(),warmup=warmup)
            with db:db.execute('INSERT INTO setup VALUES(0,?,?)',(config.canonical(setup),config.digest(setup)))
        on_event(dict(event='ordinary_value_benchmark_started',cases=count,total_cases=cases_total,games=game_count,total_games=games_total,committed_decisions=initial_decisions))
        for index in range(count,cases_total if case_limit is None else case_limit):
            row=plan['cases'][index];q={entry['pit']:entry['value_player0'] for entry in row['reference']['action_scores']} if row['reference'] else {};sign=1 if row['state']['controller']==0 else -1;measurements={};orders={}
            for bi,ms in enumerate(plan['budgets_ms']):
                names=prior.expected_order(index,bi,plan);orders[str(ms)]=names;measurements[str(ms)]={}
                for name in names:
                    cpu.verify();answer=decide(name,engines[name],row['state'],ms,plan);choice=answer['decision']['chosen']
                    measurements[str(ms)][name]=dict(decision=answer,reference_optimal=choice in row['reference']['best_moves'] if choice is not None and row['reference'] is not None else None,reference_regret=sign*(row['reference']['target_player0']-q[choice]) if choice is not None and row['reference'] is not None else None)
            body=dict(index=index,state_key=row['state_key'],cohort=row['cohort'],stratum=row['stratum'],reference=row['reference'],reference_status=row['reference_status'],reference_provenance=row['reference_provenance'],orders=orders,measurements=measurements,cpu_verified=cpu.verify());validate_case(body,index,plan);work.admit(1024*1024)
            on_event(dict(event='ordinary_value_case_precommit',completed=index,total=cases_total))
            with db:db.execute('INSERT INTO cases VALUES(?,?,?)',(index,config.canonical(body),config.digest(body)))
            on_event(dict(event='ordinary_value_case_committed',completed=index+1,total=cases_total))
        comparisons={row['id']:row for row in plan['arena_comparisons']};order=plan['arena_work_order'];new_decisions=0;paused=False
        if not no_arena and db.execute('SELECT count(*) FROM cases').fetchone()[0]==cases_total:
            for index in range(game_count,games_total if arena_limit is None else arena_limit):
                unit=order[index];comp=comparisons[unit['comparison']];pair=unit['pair'];seat=unit['seat'];row=plan['arena_starts'][pair];p=row['state'];records=records_for(db,index)
                for choice,record in enumerate(records):p=validate_record(record,index,choice,p,unit,comp,plan,native)
                for choice in range(len(records),plan['arena_max_choices']):
                    normal=native.normalize(flatten(p),actor=p['controller'],total=98)
                    if normal['terminal']:break
                    p=normal['position'];name=comp['candidate'] if p['controller']==seat else comp['opponent'];cpu.verify();answer=decide(name,engines[name],p,plan['arena_ms'],plan);fallback=not answer['decision']['usable_decision'];action=normal['legal_actions'][0] if fallback else answer['decision']['chosen']
                    saved=dict(engine=name,actor=p['controller'],fallback=fallback,decision=answer);after=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position']
                    record=dict(game_index=index,choice_index=choice,comparison=comp['id'],pair=pair,candidate_seat=seat,state=p,action=action,row=saved,after=after)
                    validate_record(record,index,choice,p,unit,comp,plan,native);work.admit(1024*1024)
                    on_event(dict(event='ordinary_value_benchmark_decision_precommit',completed=db.execute('SELECT count(*) FROM decisions').fetchone()[0],game=index,choice=choice))
                    with db:db.execute('INSERT INTO decisions VALUES(?,?,?,?)',(index,choice,config.canonical(record),config.digest(record)))
                    records.append(record);p=after;new_decisions+=1
                    on_event(dict(event='ordinary_value_benchmark_decision_committed',completed=db.execute('SELECT count(*) FROM decisions').fetchone()[0],game=index,choice=choice))
                    if decision_limit is not None and new_decisions>=decision_limit:paused=True;break
                if paused:break
                terminal=not any(map(sum,p['pits']));margin=(p['stores'][0]-p['stores'][1])*(1 if seat==0 else -1);fallbacks=Counter(record['row']['engine'] for record in records if record['row']['fallback'])
                game=dict(comparison=comp['id'],pair=pair,candidate_seat=seat,start=row['state'],family=row['family'],moves=[record['action'] for record in records],decisions=[record['row'] for record in records],final=p,status='terminal' if terminal else 'unfinished',candidate_final_margin=margin if terminal else None,candidate_match_score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None,fallbacks=dict(fallbacks),cpu_verified=cpu.verify())
                replay_game(game,comp,pair,seat,plan,native);work.admit(1024*1024)
                on_event(dict(event='ordinary_value_game_precommit',completed=index,total=games_total,comparison=comp['id'],pair=pair,seat=seat,decisions=len(records)))
                with db:db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(index,comp['id'],pair,seat,config.canonical(game),config.digest(game)))
                on_event(dict(event='ordinary_value_game_committed',completed=index+1,total=games_total,comparison=comp['id'],pair=pair,seat=seat,decisions=len(records)))
        count,game_count=validate_saved_units(db,plan,native);decision_count=db.execute('SELECT count(*) FROM decisions').fetchone()[0]
        if count!=cases_total or game_count!=games_total:on_event(dict(event='ordinary_value_benchmark_partial',cases=count,games=game_count,decisions=decision_count));return dict(status='partial',identity=identity)
        result=summarize(db,identity,plan);common.publish(output,result);resource_path=output.with_name(output.stem+'-resources.json')
        if resource_path.exists():validate_resources(load(resource_path),output,implementation,plan)
        else:
            resources=dict(schema='ordinary-value-benchmark-resources-v1',implementation_sha256=implementation['sha256'],report_file_sha256=sha(output),stage=stage,fixture=fixture,elapsed_seconds_this_invocation=time.perf_counter()-invocation_started,prior_committed_cases=initial_cases,prior_committed_games=initial_games,prior_committed_decisions=initial_decisions,host=host_peak(),torch_imported=False,energy_measured=False,scope='This CPU-only invocation including admission/startup/remaining units/publication; earlier invocations excluded; no energy claim')
            validate_resources(resources,output,implementation,plan);common.publish(resource_path,resources)
        marker=dict(cases=cases_total,games=games_total,decisions=decision_count,report_sha256=config.digest(result),resource_file_sha256=sha(resource_path))
        validate_complete(db,identity,plan,implementation,output,marker)
        with db:work.put('ordinary_value_benchmark_complete',marker)
        on_event(dict(event='ordinary_value_benchmark_complete',cases=count,games=game_count,decisions=decision_count,primary_gates=result['both_baseline_primary_gates']));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('projection','plan','freeze','admit','run','verify','allocation','acceptance'));p.add_argument('--run');p.add_argument('--output');p.add_argument('--fixture',action='store_true');p.add_argument('--fixture-control',action='store_true');p.add_argument('--case-limit',type=int);p.add_argument('--no-arena',action='store_true');p.add_argument('--arena-limit',type=int);p.add_argument('--decision-limit',type=int);args=p.parse_args()
    try:
        if args.operation=='projection':value=prepare_projection();emit(dict(event='ordinary_value_benchmark_reserved',sha256=value['sha256']));return 0
        if args.operation=='plan':make_plan();emit(dict(event='ordinary_value_benchmark_planned'));return 0
        if args.operation=='freeze':value=freeze();emit(dict(event='ordinary_value_benchmark_frozen',sha256=value['sha256']));return 0
        if args.operation=='admit':emit(pretraining_witness());return 0
        if args.operation=='allocation':emit(allocation_decision());return 0
        if args.operation=='verify':
            require(not args.fixture and args.run is None and args.output is None and not args.fixture_control,'Verify accepts registered production only');emit(closed_witness());return 0
        if args.operation=='acceptance':
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='ordinary_value_benchmark_author.py')
            def names(items):
                for item in items:
                    if isinstance(item,unittest.TestSuite):yield from names(item)
                    else:yield item.id().split('.')[-1]
            covered=sorted(names(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
            files=('congklak/ordinary_value_benchmark.py','tools/benchmark-ordinary-value.py','tests/ordinary_value_benchmark_author.py','congklak/ordinary_value_progress.py','tools/ordinary-value-dashboard.html','build/ordinary-value-benchmark-plan.json','build/ordinary-value-benchmark-projection.json','build/ordinary-value-baseline.json')
            body=dict(schema='ordinary-value-benchmark-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),methods=covered,inputs={name:sha(ROOT/name) for name in files})
            common.teacher.atomic_json(ROOT/'build/ordinary-value-benchmark-author.json',body);return 0 if result.wasSuccessful() else 1
        run(path=args.run,output=args.output,case_limit=args.case_limit,no_arena=args.no_arena,arena_limit=args.arena_limit,fixture=args.fixture,fixture_control=args.fixture_control,decision_limit=args.decision_limit);return 0
    except KeyboardInterrupt:emit(dict(event='ordinary_value_benchmark_interrupted'));return 130
    except Exception as exc:emit(dict(event='ordinary_value_benchmark_refused',error=str(exc)));return 2

