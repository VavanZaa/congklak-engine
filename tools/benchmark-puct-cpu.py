"""Recoverable CPU PUCT comparison with cold per-decision evaluation caches."""
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
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as common,phase6_store
from congklak import puct_cpu_common as cpu_common
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative,Model,sha
from congklak.phase6_inference import BatchEvaluator
from congklak.puct import search as puct_search
from congklak.timing_v2 import PerformanceCore
spec=importlib.util.spec_from_file_location('frozen_phase6_benchmark_cpu_dependency',ROOT/'tools/benchmark-phase6-v2.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
require=prior.require;integer=prior.integer;numeric=prior.numeric
PLAN=ROOT/'build/puct-cpu-plan.json';IMPLEMENTATION=ROOT/'build/puct-cpu-implementation.json'
DEFAULT_RUN=ROOT/'data/puct-cpu-benchmark-v1';DEFAULT_OUTPUT=ROOT/'build/puct-cpu-benchmark.json'
CANDIDATES=('warm_legacy','updated_legacy','warm_single','updated_single','warm_cache','updated_cache')
ENGINES=['H_v1','H8_relay_CPU',*CANDIDATES]
COUNTERS=('positions_submitted','evaluations_requested','terminal_returns','native_actual_evals','predict_successes','cache_hits','cache_misses','cache_evictions','feature_simulations','feature_deposits','native_calls','cache_resets')
ROLES=cpu_common.ROLES
SCIENCE=cpu_common.REQUIRED_SCIENCE

def read_plan(path=PLAN):
    plan=json.loads(Path(path).read_text(encoding='utf-8'));old=common.read_plan()
    require(plan==cpu_common.expected_plan(),'CPU plan/current accepted evidence differs')
    require(plan.get('schema')=='puct-cpu-plan-v1' and plan.get('rules_digest')==config.RULES_DIGEST,'Wrong CPU PUCT plan/rules')
    for key in ('cases','arena_starts','budgets_ms','arena_ms','arena_max_choices','max_depth','tt_mb','use_pvs','puct','cpu_declared','bootstrap_seed','bootstrap_samples','source_model','source_model_sha256'):
        require(plan.get(key)==old[key],'CPU phase changed inherited geometry: '+key)
    require(plan.get('engines')==ENGINES and len(plan['cases'])==80 and len(plan['arena_starts'])==24,'Wrong CPU experiment coverage')
    expected=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in CANDIDATES for b in ('H_v1','H8_relay_CPU')]
    require(plan.get('arena_comparisons')==expected,'Unregistered CPU comparisons')
    require(plan.get('cache_capacity')==4096,'Wrong bounded cache capacity')
    require(plan.get('cache_reset')=='Before every decision inside whole-decision clock','Cache reset not declared')
    require(plan.get('arena_work_order')==arena_order(plan),'Arena ordering differs from balanced predeclaration')
    return plan

def arena_order(plan):
    names=[c['id'] for c in plan['arena_comparisons']];order=[]
    for pair in range(len(plan['arena_starts'])):
        shift=pair%len(names)
        for name in names[shift:]+names[:shift]:
            for seat in ((0,1) if pair%2==0 else (1,0)):
                order.append(dict(comparison=name,pair=pair,seat=seat))
    return order

def pinned(mapping):
    require(isinstance(mapping,dict) and bool(mapping),'Missing immutable pins')
    for name,digest in mapping.items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path)==digest,'Pinned CPU experiment input differs: '+name)

def admit(plan,manifest_path=IMPLEMENTATION):
    body=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    require(body.get('schema')=='puct-cpu-implementation-v1' and body.get('sha256')==config.digest({k:v for k,v in body.items() if k!='sha256'}),'Corrupt CPU implementation manifest')
    require(isinstance(body.get('roles'),dict) and all(body['roles'].get(k)==v for k,v in ROLES.items()),'CPU producer role mapping differs')
    require(isinstance(body.get('sources'),dict) and all(path in body['sources'] for path in body['roles'].values()),'Unpinned CPU producer role')
    require(isinstance(body.get('science_inputs'),dict) and set(SCIENCE)<=set(body['science_inputs']),'Missing mandatory CPU science inputs')
    pinned(body['sources']);pinned(body['science_inputs'])
    acceptance=json.loads((ROOT/'build/phase6-acceptance.json').read_text(encoding='utf-8'))
    require(acceptance.get('schema')=='phase6-acceptance-v1' and acceptance.get('passed') is True and acceptance.get('engineering_passed') is True and acceptance.get('tests_run')==42 and all(acceptance.get(k)==0 for k in ('failures','errors','skips')),'Prior42 engineering acceptance incomplete')
    require(len(acceptance.get('inputs',{}))==1192 and acceptance.get('checkpoint_files_verified')==30 and acceptance.get('independent_audit_complete') is True,'Prior exhaustive science coverage differs')
    pinned(acceptance['inputs'])
    old_impl=prior.implementation_manifest();witness=prior.model_witnesses(common.read_plan(),old_impl)
    require('torch' not in sys.modules,'CPU benchmark must not import Torch')
    return body,dict(prior_acceptance_file_sha256=sha(ROOT/'build/phase6-acceptance.json'),prior_accepted_inputs=1192,prior_checks=42,model_witnesses=witness,stdlib_only=True)

class LegacyTelemetry:
    """Count physical frozen double-pass work without modifying its evaluator."""
    supports_cancel=True
    def __init__(self,evaluator):
        self.evaluator=evaluator;self.native=evaluator.native;self.model=evaluator.model;self.backend=evaluator.backend
        self.identity={**evaluator.identity,'telemetry_protocol':'cpu-puct-legacy-cumulative-v1','cache_capacity':0,'feature_passes_per_native_eval':2}
        self.startup_ms=evaluator.startup_ms;self.counters=dict.fromkeys(COUNTERS,0)
        feature_vector=evaluator.native.feature_vector;evaluate=evaluator.native.evaluate;self.uninstrumented_reference=evaluate
        def counted_features(position):
            self.counters['evaluations_requested']+=1;self.counters['cache_misses']+=1;self.counters['native_calls']+=1
            return feature_vector(position)
        def counted_forward(position):
            self.counters['native_calls']+=1;result=evaluate(position)
            self.counters['native_actual_evals']+=1;self.counters['predict_successes']+=1
            return result
        evaluator.native.feature_vector=counted_features;evaluator.native.evaluate=counted_forward
    def reset_cache(self):self.counters['cache_resets']+=1
    def snapshot_counters(self):return {**self.counters,'cache_occupancy':0,'cache_capacity':0}
    def evaluate_many(self,positions,cancel=None,deadline=None):
        positions=[config.position(p) for p in positions];self.counters['positions_submitted']+=len(positions);returned=False
        self.evaluator.last_metrics={}
        try:
            result=self.evaluator.evaluate_many(positions,cancel=cancel,deadline=deadline);returned=True;return result
        finally:
            metrics=self.evaluator.last_metrics
            # The frozen adapter records a prefix of actual feature/forward
            # work even when a deadline discards the enclosing batch.
            if returned:self.counters['terminal_returns']+=sum(not sum(map(sum,p['pits'])) for p in positions)
            for key in ('feature_simulations','feature_deposits'):self.counters[key]+=metrics.get(key,0)

def counter_snapshot(obj):
    value=obj.snapshot_counters()
    if 'cache_current_entries' in value:value={**value,'cache_occupancy':value['cache_current_entries']};del value['cache_current_entries']
    require(set(value)==set(COUNTERS)|{'cache_capacity','cache_occupancy'},'Evaluator counter fields differ')
    require(all(integer(x) for x in value.values()),'Invalid evaluator counters')
    require(value['cache_capacity'] in (0,4096) and value['cache_occupancy']<=value['cache_capacity'],'Unbounded cache snapshot')
    return value

def validate_telemetry(value,decision,profile):
    if profile in ('H_v1','H8_relay_CPU'):
        require(value is None,'Baseline must not fabricate neural-cache telemetry');return
    fields={'protocol','profile','counts','cache_capacity','cache_occupancy_before','cache_occupancy_after_reset','cache_occupancy_after','reset_ms','granted_search_ms','requested_ms','deadline_policy','logical_row_evaluations','physical_forward_executions','feature_passes_per_native_eval'}
    require(isinstance(value,dict) and set(value)==fields,'Unknown evaluator telemetry fields')
    require(value['protocol']=='cpu-puct-counter-delta-v1' and value['profile']==profile and value['requested_ms']==decision['requested_ms'],'Evaluator profile/budget differs')
    counts=value['counts'];require(isinstance(counts,dict) and set(counts)==set(COUNTERS) and all(integer(x) for x in counts.values()),'Invalid physical counter delta')
    cap=4096 if profile.endswith('_cache') else 0
    require(value['cache_capacity']==cap and all(integer(value[k],0,cap) for k in ('cache_occupancy_before','cache_occupancy_after_reset','cache_occupancy_after')) and value['cache_occupancy_after_reset']==0,'Cache carries prior decision state')
    require(counts['cache_resets']==1 and counts['evaluations_requested']==counts['cache_hits']+counts['cache_misses'],'Cache request conservation differs')
    passes=2 if profile.endswith('_legacy') else 1
    require(counts['native_actual_evals']<=counts['native_calls']<=passes*counts['cache_misses'] and counts['predict_successes']<=counts['native_actual_evals'],'Actual native-forward accounting differs')
    require(counts['evaluations_requested']+counts['terminal_returns']<=counts['positions_submitted'],'Admitted counter prefix exceeds submitted rows')
    if not cap:require(not counts['cache_hits'] and not counts['cache_evictions'] and not value['cache_occupancy_after'],'Disabled cache has hit/eviction')
    require(counts['cache_evictions']<=counts['native_actual_evals'] and value['cache_occupancy_after']<=counts['predict_successes'],'Cache occupancy/eviction exceeds actual computation')
    require(value['feature_passes_per_native_eval']==passes and counts['feature_simulations']<=14*passes*counts['native_calls'],'Feature work does not match physical passes')
    require(numeric(value['reset_ms']) and integer(value['granted_search_ms'],0,value['requested_ms']) and value['granted_search_ms']<=max(0,math.floor(value['requested_ms']-value['reset_ms'])),'Cache reset grants extra deadline')
    require(value['deadline_policy']=='Floor remaining integer milliseconds after reset; no new budget' and value['physical_forward_executions']==counts['native_actual_evals'],'Incorrect deadline/physical label')
    report=decision['report'];require(value['logical_row_evaluations']==report['work']['neural_evaluations'],'Logical PUCT evaluations relabeled physical')
    require(report['work']['feature_simulations']==counts['feature_simulations'] and report['work']['feature_deposits']==counts['feature_deposits'],'PUCT and evaluator physical feature work disagree')

def empty_report(p,plan):
    legal=[a for a,c in enumerate(p['pits'][p['controller']]) if c];d=p['stores'][0]-p['stores'][1]
    return dict(chosen=legal[0],visits=[0]*7,policy=[1/len(legal) if a in legal else 0 for a in range(7)],root_priors=[0.]*7,estimated_value_player0=d,action_mean_value_player0=[None]*7,root_expanded=False,
                completed_simulations=0,requested_simulations=plan['puct']['simulations'],nodes=1,max_nodes=plan['puct']['max_nodes'],stop_reason='time_budget',proof_status='monte_carlo_estimate',usable_decision=False,
                protocol=prior.PUCT_PROTOCOL,rules_digest=config.RULES_DIGEST,value_perspective='fixed_player0_seeds',first_play_urgency='parent_neural_bootstrap',unvisited_root_q_player0=d,c_puct=1.5,
                work=dict(transitions=0,deposits=0,feature_simulations=0,feature_deposits=0,neural_evaluations=0,neural_calls=0,inference_ms=0.,root_expansions=0,total_deposits=0))

def decide(name,engine,p,ms,plan):
    started=time.perf_counter();kind,obj=engine
    if kind!='puct':return dict(decision=prior.decide(engine,p,ms,plan),evaluator=None)
    before=counter_snapshot(obj);tick=time.perf_counter();obj.reset_cache();reset_ms=(time.perf_counter()-tick)*1000;reset=counter_snapshot(obj)
    grant=max(0,math.floor(ms-(time.perf_counter()-started)*1000))
    if grant:
        raw=puct_search(p,obj,simulations=plan['puct']['simulations'],time_ms=grant,max_nodes=plan['puct']['max_nodes'],c_puct=1.5,root_noise=None,memory_mb=8192)
        report=prior.compact_puct(raw);elapsed=raw['timing']['elapsed_ms']
    else:
        remaining=ms/1000-(time.perf_counter()-started)
        if remaining>0:time.sleep(remaining)
        report=empty_report(p,plan);elapsed=0.
    after=counter_snapshot(obj);wall=(time.perf_counter()-started)*1000
    # A native search can correctly stop at the floored sub-budget before the
    # original fractional deadline. Retain that time, label budget rounding.
    if report['stop_reason']=='time_budget' and wall<ms:time.sleep((ms-wall)/1000)
    wall=(time.perf_counter()-started)*1000
    answer=dict(kind='puct',report=report,stop_reason=report['stop_reason'],engine_elapsed_ms=elapsed,nodes=report['nodes'],usable_decision=report['usable_decision'],chosen=report['chosen'] if report['usable_decision'] else None,
                completed_depth=None,completed_simulations=report['completed_simulations'],wall_ms=wall,requested_ms=ms,overshoot_ms=max(0.,wall-ms))
    telemetry=dict(protocol='cpu-puct-counter-delta-v1',profile=name,counts={k:after[k]-before[k] for k in COUNTERS},cache_capacity=after['cache_capacity'],cache_occupancy_before=before['cache_occupancy'],cache_occupancy_after_reset=reset['cache_occupancy'],cache_occupancy_after=after['cache_occupancy'],reset_ms=reset_ms,granted_search_ms=grant,requested_ms=ms,
                   deadline_policy='Floor remaining integer milliseconds after reset; no new budget',logical_row_evaluations=report['work']['neural_evaluations'],physical_forward_executions=after['native_actual_evals']-before['native_actual_evals'],feature_passes_per_native_eval=2 if name.endswith('_legacy') else 1)
    result=dict(decision=answer,evaluator=telemetry);validate_envelope(result,p,ms,plan,name)
    wall=(time.perf_counter()-started)*1000;answer.update(wall_ms=wall,overshoot_ms=max(0.,wall-ms));return result

def validate_envelope(value,p,ms,plan,name):
    require(isinstance(value,dict) and set(value)=={'decision','evaluator'},'Unknown decision envelope')
    prior.validate_decision(value['decision'],p,ms,plan)
    require(value['decision']['kind']==('h' if name=='H_v1' else 'relay' if name=='H8_relay_CPU' else 'puct'),'Evaluator/search engine family differs')
    validate_telemetry(value['evaluator'],value['decision'],name)

def validate_case(body,index,plan):
    stripped={**body,'measurements':{ms:{name:{'decision':value['decision']['decision'],'reference_optimal':value['reference_optimal'],'reference_regret':value['reference_regret']} for name,value in engines.items()} for ms,engines in body['measurements'].items()}}
    prior.validate_case(stripped,index,plan)
    for ms,engines in body['measurements'].items():
        for name,value in engines.items():
            require(set(value)=={'decision','reference_optimal','reference_regret'},'Unknown measurement envelope');validate_envelope(value['decision'],plan['cases'][index]['state'],int(ms),plan,name)

def replay_game(body,comparison,pair,seat,plan,native):
    stripped={**body,'decisions':[{**row,'decision':row['decision']['decision']} for row in body['decisions']]}
    prior.replay_game(stripped,comparison,pair,seat,plan,native)
    p=body['start']
    for action,row in zip(body['moves'],body['decisions']):
        p=native.normalize(flatten(p),actor=p['controller'],total=98)['position'];validate_envelope(row['decision'],p,plan['arena_ms'],plan,row['engine'])
        p=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position']

def summarize(db,identity,plan):
    buckets=defaultdict(list)
    for text, in db.execute('SELECT body FROM cases ORDER BY idx'):
        body=json.loads(text)
        for ms,engines in body['measurements'].items():
            for name,value in engines.items():buckets[(body['cohort'],ms,name)].append(value)
    cases={}
    for key,rows in buckets.items():
        decisions=[r['decision']['decision'] for r in rows];usable=[r for r in rows if r['decision']['decision']['usable_decision']]
        counts={k:sum((r['decision']['evaluator'] or {}).get('counts',{}).get(k,0) for r in rows) for k in COUNTERS} if key[2] in CANDIDATES else None
        cases['/'.join(key)]=dict(positions=len(rows),no_completed_recommendation=len(rows)-len(usable),reference_optimal_accuracy=statistics.mean(r['reference_optimal'] for r in usable) if usable else None,
            mean_reference_regret_seeds=statistics.mean(r['reference_regret'] for r in usable) if usable else None,median_wall_ms=statistics.median(r['wall_ms'] for r in decisions),max_wall_ms=max(r['wall_ms'] for r in decisions),max_overshoot_ms=max(r['overshoot_ms'] for r in decisions),
            mean_completed_simulations=statistics.mean(r['completed_simulations'] for r in decisions) if decisions[0]['kind']=='puct' else None,evaluator_counts=counts,cache_hit_fraction=counts['cache_hits']/counts['evaluations_requested'] if counts and counts['evaluations_requested'] else None)
    arenas={}
    for comp in plan['arena_comparisons']:
        games=[json.loads(text) for text, in db.execute('SELECT body FROM games WHERE comparison=? ORDER BY pair,seat',(comp['id'],))];values=[];scores=[]
        for pair in range(comp['pairs']):
            group=[g for g in games if g['pair']==pair]
            if len(group)==2 and all(g['status']=='terminal' for g in group):values.append(statistics.mean(g['candidate_final_margin'] for g in group));scores.append(statistics.mean(g['candidate_match_score'] for g in group))
        ci=prior.interval(values,plan);fallbacks={name:sum(g['fallbacks'].get(name,0) for g in games) for name in (comp['candidate'],comp['opponent'])}
        arenas[comp['id']]=dict(pairs=comp['pairs'],complete_pairs=len(values),incomplete_pairs=comp['pairs']-len(values),terminal_games=sum(g['status']=='terminal' for g in games),unfinished_games=sum(g['status']!='terminal' for g in games),
            candidate_wins=sum(g['candidate_match_score']==1 for g in games),candidate_draws=sum(g['candidate_match_score']==.5 for g in games),candidate_losses=sum(g['candidate_match_score']==0 for g in games),paired_mean_score=statistics.mean(scores) if scores else None,
            primary_paired_mean_final_margin_seeds=statistics.mean(values) if values else None,primary_margin_bootstrap_95_interval=ci,primary_strength_gate_passed=bool(len(values)==comp['pairs'] and not sum(fallbacks.values()) and ci and ci[0]>0),fallback_counts=fallbacks)
    gates={c:all(arenas[c+'_vs_'+b]['primary_strength_gate_passed'] for b in ('H_v1','H8_relay_CPU')) for c in CANDIDATES}
    return dict(schema='puct-cpu-benchmark-results-v1',identity=identity,cases=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],games=db.execute('SELECT COUNT(*) FROM games').fetchone()[0],case_results=cases,arenas=arenas,both_baseline_primary_gates=gates,
                default_promotion=False,sealed_final_test_consumed=False,energy_measured=False,scope='Same reused development families, all six predeclared candidates; cold per-decision caches; fixed-player0 PUCT estimates; no causal training comparison, release validation or power claim')

def emit(body):
    print(config.canonical(body),flush=True)
    for unit in ('case','game'):
        value=os.environ.get('PUCT_CPU_BLOCK_AFTER_'+unit.upper())
        if body.get('event')=='puct_cpu_'+unit+'_committed' and value and int(value)==body['completed']:
            print(config.canonical(dict(event='puct_cpu_'+unit+'_commit_barrier',completed=body['completed'])),flush=True)
            while True:time.sleep(.05)

def validate_saved_units(db,plan,native):
    comparisons={c['id']:c for c in plan['arena_comparisons']};order=plan['arena_work_order']
    for expected,(index,text,digest) in enumerate(db.execute('SELECT idx,body,hash FROM cases ORDER BY idx')):
        require(index==expected and config.digest(json.loads(text))==digest,'Corrupt/noncontiguous CPU case prefix');validate_case(json.loads(text),index,plan)
    for expected,(index,name,pair,seat,text,digest) in enumerate(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx')):
        require(index==expected and expected<len(order) and order[expected]==dict(comparison=name,pair=pair,seat=seat) and config.digest(json.loads(text))==digest,'Corrupt/orphan CPU arena prefix')
        replay_game(json.loads(text),comparisons[name],pair,seat,plan,native)
    return db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],db.execute('SELECT COUNT(*) FROM games').fetchone()[0]

def run(path=DEFAULT_RUN,output=DEFAULT_OUTPUT,case_limit=None,no_arena=False,arena_limit=None,on_event=emit):
    invocation_started=time.perf_counter();plan=read_plan();implementation,witness=admit(plan)
    if case_limit is not None:require(integer(case_limit,0,80),'Invalid case limit')
    if arena_limit is not None:require(integer(arena_limit,0,576),'Invalid arena limit')
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    with ExitStack() as stack:
        cpu=stack.enter_context(PerformanceCore(plan['cpu_declared']));baseline=Native();source=stack.enter_context(NeuralNative(ROOT/plan['source_model'],mode='value-none'))
        engines={'H_v1':('h',baseline),'H8_relay_CPU':('relay',source)}
        for name in CANDIDATES:
            model=common.WARM if name.startswith('warm_') else common.UPDATE
            if name.endswith('_legacy'):obj=LegacyTelemetry(stack.enter_context(BatchEvaluator(model,'native',max_batch=32)))
            else:obj=stack.enter_context(CpuBatchEvaluator(model,cache_capacity=4096 if name.endswith('_cache') else 0,max_batch=32,backend='auto'))
            engines[name]=('puct',obj)
        require('torch' not in sys.modules,'CPU evaluator imported Torch')
        identity=dict(schema='puct-cpu-benchmark-v1',plan_sha256=config.digest(plan),implementation=implementation,implementation_file_sha256=sha(IMPLEMENTATION),admission=witness,cpu=cpu.metadata,
            baseline_dll_sha256=baseline.build_id,source_model=dict(file_sha256=source.model.file_sha256,weights_sha256=source.model.id,dll_sha256=source.build_id),evaluators={k:v.identity for k,(_,v) in engines.items() if k in CANDIDATES},memory_mb_all_engines=8192)
        work=stack.enter_context(closing(phase6_store.Work(Path(path),identity)));db=work.db
        db.executescript('CREATE TABLE IF NOT EXISTS cases(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT); CREATE TABLE IF NOT EXISTS games(idx INTEGER PRIMARY KEY,comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT,UNIQUE(comparison,pair,seat)); CREATE TABLE IF NOT EXISTS setup(id INTEGER PRIMARY KEY,body TEXT,hash TEXT);')
        comparisons={c['id']:c for c in plan['arena_comparisons']};order=plan['arena_work_order']
        count,game_count=validate_saved_units(db,plan,baseline);case_count_initial=count
        complete=work.get('puct_cpu_benchmark_complete')
        if complete:
            saved=json.loads(Path(output).read_text(encoding='utf-8'));resource_path=Path(output).with_name(Path(output).stem+'-resources.json')
            require(resource_path.is_file() and sha(resource_path)==complete['resource_file_sha256'],'Completed CPU resource observation differs')
            require(count==80 and game_count==576 and config.digest(saved)==complete['report_sha256'] and saved==summarize(db,identity,plan),'Completed CPU report/ledger differs')
            on_event(dict(event='puct_cpu_benchmark_already_complete',cases=count,games=game_count));return saved
        setup_row=db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone();warmup={}
        for name in CANDIDATES:
            obj=engines[name][1];cpu.verify();obj.reset_cache();tick=time.perf_counter();obj.evaluate_many([r['state'] for r in plan['cases'][:8]]);obj.reset_cache()
            warmup[name]=dict(startup_ms=obj.startup_ms,warmup_ms=(time.perf_counter()-tick)*1000,counters=counter_snapshot(obj))
        if setup_row:require(config.digest(json.loads(setup_row[0]))==setup_row[1],'Corrupt CPU startup observation')
        else:
            setup=dict(cpu_observations=cpu.observations,verified=cpu.verify(),warmup=warmup)
            with db:db.execute('INSERT INTO setup VALUES(0,?,?)',(config.canonical(setup),config.digest(setup)))
        on_event(dict(event='puct_cpu_benchmark_started',cases=count,total_cases=80,games=game_count,total_games=576))
        stop=80 if case_limit is None else case_limit
        for index in range(count,stop):
            row=plan['cases'][index];q={a['pit']:a['value_player0'] for a in row['reference']['action_scores']};sign=1 if row['state']['controller']==0 else -1;measurements={};orders={}
            for bi,ms in enumerate(plan['budgets_ms']):
                names=prior.expected_order(index,bi,plan);orders[str(ms)]=names;measurements[str(ms)]={}
                for name in names:
                    cpu.verify();answer=decide(name,engines[name],row['state'],ms,plan);choice=answer['decision']['chosen']
                    measurements[str(ms)][name]=dict(decision=answer,reference_optimal=choice in row['reference']['best_moves'] if choice is not None else None,reference_regret=sign*(row['reference']['target_player0']-q[choice]) if choice is not None else None)
            body=dict(index=index,state_key=row['state_key'],cohort=row['cohort'],stratum=row['stratum'],reference=row['reference'],orders=orders,measurements=measurements,cpu_verified=cpu.verify());validate_case(body,index,plan);work.admit(1024*1024)
            with db:db.execute('INSERT INTO cases VALUES(?,?,?)',(index,config.canonical(body),config.digest(body)))
            on_event(dict(event='puct_cpu_case_committed',completed=index+1,total=80))
        if not no_arena:
            stop_game=576 if arena_limit is None else arena_limit
            for index in range(game_count,stop_game):
                unit=order[index];comp=comparisons[unit['comparison']];pair=unit['pair'];seat=unit['seat'];row=plan['arena_starts'][pair];p=row['state'];moves=[];decisions=[];fallbacks=Counter()
                for _ in range(plan['arena_max_choices']):
                    normal=baseline.normalize(flatten(p),actor=p['controller'],total=98)
                    if normal['terminal']:break
                    p=normal['position'];name=comp['candidate'] if p['controller']==seat else comp['opponent'];cpu.verify();answer=decide(name,engines[name],p,plan['arena_ms'],plan);fallback=not answer['decision']['usable_decision'];action=normal['legal_actions'][0] if fallback else answer['decision']['chosen']
                    if fallback:fallbacks[name]+=1
                    decisions.append(dict(engine=name,actor=p['controller'],fallback=fallback,decision=answer));moves.append(action);p=baseline.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position']
                terminal=not any(map(sum,p['pits']));margin=(p['stores'][0]-p['stores'][1])*(1 if seat==0 else -1)
                game=dict(comparison=comp['id'],pair=pair,candidate_seat=seat,start=row['state'],family=row['family'],moves=moves,decisions=decisions,final=p,status='terminal' if terminal else 'unfinished',candidate_final_margin=margin if terminal else None,candidate_match_score=(1 if margin>0 else 0 if margin<0 else .5) if terminal else None,fallbacks=dict(fallbacks),cpu_verified=cpu.verify())
                replay_game(game,comp,pair,seat,plan,baseline);work.admit(1024*1024)
                with db:db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(index,comp['id'],pair,seat,config.canonical(game),config.digest(game)))
                on_event(dict(event='puct_cpu_game_committed',completed=index+1,total=576,comparison=comp['id'],pair=pair,seat=seat))
        count=db.execute('SELECT COUNT(*) FROM cases').fetchone()[0];games=db.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        if count!=80 or games!=576:on_event(dict(event='puct_cpu_benchmark_partial',cases=count,games=games));return dict(status='partial',identity=identity)
        result=summarize(db,identity,plan);common.publish(Path(output),result)
        resource_path=Path(output).with_name(Path(output).stem+'-resources.json')
        if resource_path.exists():
            resources=json.loads(resource_path.read_text(encoding='utf-8'));require(resources['report_file_sha256']==sha(output) and resources['implementation_sha256']==implementation['sha256'],'Original resource observation differs')
        else:
            resources=dict(schema='puct-cpu-benchmark-resources-v1',implementation_sha256=implementation['sha256'],report_file_sha256=sha(output),elapsed_seconds_this_invocation=time.perf_counter()-invocation_started,prior_committed_cases=case_count_initial,prior_committed_games=game_count,host=host_peak(),torch_imported=False,energy_measured=False,scope='This CPU-only invocation including admission/startup/remaining units/publication; previous invocations excluded; no electricity or GPU-use claim')
            common.publish(resource_path,resources)
        with db:work.put('puct_cpu_benchmark_complete',dict(cases=80,games=576,report_sha256=config.digest(result),resource_file_sha256=sha(resource_path)))
        on_event(dict(event='puct_cpu_benchmark_complete',cases=80,games=576,primary_gates=result['both_baseline_primary_gates']));return result

def host_peak():
    if os.name!='nt':return None
    class Counters(C.Structure):
        _fields_=[('cb',C.c_ulong),('faults',C.c_ulong)]+[(key,C.c_size_t) for key in ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    kernel=C.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=C.c_void_p;psapi=C.WinDLL('psapi',use_last_error=True);psapi.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(Counters),C.c_ulong]
    value=Counters();value.cb=C.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),C.byref(value),value.cb):raise C.WinError(C.get_last_error())
    return dict(peak_working_set_bytes=value.peak,peak_private_commit_bytes=value.peak_pagefile)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('run','admit','acceptance'));p.add_argument('--run',default=str(DEFAULT_RUN));p.add_argument('--output',default=str(DEFAULT_OUTPUT));p.add_argument('--case-limit',type=int);p.add_argument('--no-arena',action='store_true');p.add_argument('--arena-limit',type=int);args=p.parse_args()
    try:
        if args.operation=='admit':
            _,witness=admit(read_plan());emit(dict(event='puct_cpu_admitted',prior_inputs=witness['prior_accepted_inputs'],torch_imported=False));return 0
        if args.operation=='acceptance':
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='puct_cpu_benchmark_acceptance.py')
            def names(items):
                for item in items:
                    if isinstance(item,unittest.TestSuite):yield from names(item)
                    else:yield item.id()
            covered=sorted(names(suite));result=unittest.TextTestRunner(verbosity=2).run(suite)
            files=('tools/benchmark-puct-cpu.py','tests/puct_cpu_benchmark_acceptance.py','congklak/puct_cpu_common.py','congklak/puct_cpu_inference.py','build/congklak_puct_cpu.dll','build/puct-cpu-build.json','build/puct-cpu-plan.json','build/phase6-acceptance.json','tools/benchmark-phase6-v2.py','congklak/puct.py','congklak/phase6_store.py','data/training-phase6-warm-v1/model.json','data/training-phase6-update-v1/model.json')
            body=dict(schema='puct-cpu-benchmark-acceptance-v1',passed=result.wasSuccessful(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),covered_methods=covered,inputs={name:sha(ROOT/name) for name in files})
            common.teacher.atomic_json(ROOT/'build/puct-cpu-benchmark-acceptance.json',body);return 0 if result.wasSuccessful() else 1
        run(args.run,args.output,args.case_limit,args.no_arena,args.arena_limit);return 0
    except KeyboardInterrupt:emit(dict(event='puct_cpu_benchmark_interrupted'));return 130
    except Exception as exc:emit(dict(event='puct_cpu_benchmark_error',error=str(exc)));return 2
if __name__=='__main__':raise SystemExit(main())
