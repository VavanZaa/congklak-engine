"""Independent acceptance and literal production verification for CPU PUCT."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import closing,ExitStack
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,puct_cpu_common as common
from puct_cpu_audit_acceptance import PuctCpuIndependentAcceptance,EVIDENCE,WORK,sha,load,module,semantic,normalize_literal,json_position,literal,value_and_policy

GROUPS={'plan':['test_01_prior_acceptance_and_model_artifacts_preserved','test_02_predeclared_cohorts_profiles_clocks_and_game_order'],
        'inference':['test_03_literal_structured_forward_and_actual_single_pass_work','test_04_raw_state_store_actor_model_isolation_and_copy_out','test_05_actual_4096_entry_LRU_bound_and_reset','test_06_fixed_simulation_forest_noise_values_visits_and_RNG','test_07_cancelled_miss_hit_and_lock_wait_have_truthful_prefix_work','test_08_bad_ABI_model_input_and_closed_evaluator_are_refused'],
        'wrapper':['test_09_deadline_floor_legacy_terminal_and_speed_counter_admission'],
        'analysis':['test_10_actual_analysis_pre_postcommit_kills_and_counter_corruption'],
        'case':['test_11_actual_case_kill_prefix_and_rehashed_regret_refusal'],
        'arena':['test_12_actual_arena_kill_576_literal_replays_and_finished_stability'],
        'speed':['test_13_actual_speed_first_unit_kill_resume_1410_and_counter_refusal']}
SOURCES=['tests/puct_cpu_audit_acceptance.py','tools/puct-cpu-audit.py','congklak/puct_cpu_common.py']
def save(path,body):Path(path).write_text(json.dumps(body,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def input_map(paths):return {str(Path(p).relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
def require(value,message):
    if not value:raise AssertionError(message)
def group(name):
    EVIDENCE.clear();started=time.monotonic();methods=GROUPS[name]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(PuctCpuIndependentAcceptance(n) for n in methods))
    paths=[ROOT/p for p in SOURCES]+[common.PLAN,ROOT/'build/phase6-acceptance.json',ROOT/'build/stage5-deeper-acceptance.json']
    if name=='inference':paths += [ROOT/'congklak/puct_cpu_inference.py',ROOT/'native/neural_puct_cpu.cpp',ROOT/'build/puct-cpu-build.json',ROOT/'build/congklak_puct_cpu.dll',common.WARM,common.UPDATE]
    if name in ('wrapper','analysis','case','arena','speed'):paths += [ROOT/path for path in ('congklak/puct_cpu_inference.py','congklak/puct_cpu_analysis.py','tools/benchmark-puct-cpu.py','tools/puct-cpu-speed.py','build/puct-cpu-implementation.json','build/puct-cpu-build.json','build/congklak_puct_cpu.dll')]+[common.WARM,common.UPDATE]
    retained=[p for p in PuctCpuIndependentAcceptance.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    body={**EVIDENCE,'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'duration_seconds':time.monotonic()-started,'covered_methods':methods,'inputs':input_map(paths+retained),'retained_directory':str(PuctCpuIndependentAcceptance.base.relative_to(ROOT)).replace('\\','/'),'failure_details':[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors]}
    save(WORK/(name+'-tests.json'),body)
    print(config.canonical(dict(event='puct_cpu_independent_group',group=name,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1

def collect():
    covered=[];groups=[];evidence={}
    metadata={'passed','tests_run','failures','errors','skips','duration_seconds','covered_methods','inputs','retained_directory','failure_details'}
    for name,methods in GROUPS.items():
        path=WORK/(name+'-tests.json');body=load(path)
        require(body['passed'] and body['tests_run']==len(methods) and body['covered_methods']==methods and not body['failures']+body['errors']+body['skips'],'Incomplete independent group '+name)
        for file,digest in body['inputs'].items():require(sha(ROOT/file)==digest,'Stale independent input '+file)
        for key,value in body.items():
            if key not in metadata:require(key not in evidence,'Duplicate independent evidence '+key);evidence[key]=value
        covered+=methods;groups.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),methods=methods))
    expected=unittest.defaultTestLoader.getTestCaseNames(PuctCpuIndependentAcceptance)
    require(len(covered)==len(set(covered))==13 and sorted(covered)==expected,'Independent unique method coverage differs')
    result={**evidence,'passed':True,'tests_run':13,'failures':0,'errors':0,'skips':0,'covered_methods':sorted(covered),'groups':groups}
    save(WORK/'tests.json',result);return result

def prior_audit():
    for filename,count in [('phase6-acceptance.json',1192),('stage5-deeper-acceptance.json',878)]:
        body=load(ROOT/'build'/filename);require(body['passed'] and len(body['inputs'])==count,'Prior acceptance coverage differs')
        for path,digest in body['inputs'].items():require(sha(ROOT/path)==digest,'Prior artifact changed '+path)
    return dict(phase6_inputs=1192,stage5_inputs=878,all_current_hashes_match=True)

def model_audit():
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    states=[]
    with closing(sqlite3.connect((ROOT/'data/deeper-9000-v2/work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        for body, in db.execute("SELECT json_extract(body,'$.state') FROM roots WHERE json_extract(body,'$.split')='development' ORDER BY state"):states.append(json.loads(body))
    require(len(states)==1000,'Current development model verification scope differs');models=[]
    for name,path in [('warm',common.WARM),('updated',common.UPDATE)]:
        with CpuBatchEvaluator(path,cache_capacity=0,max_batch=32) as evaluator:
            worst=0.
            for start in range(0,1000,32):
                batch=states[start:start+32]
                for p,row in zip(batch,evaluator.evaluate_many(batch)):
                    value,_,logits=value_and_policy(normalize_literal(p),evaluator.model.envelope['model'])
                    error=max(abs(value-row['value_player0']),max(abs(a-b) for a,b in zip(logits,row['policy_logits'])))
                    require(error<=1e-10,'Independent structured full-model forward differs');worst=max(worst,error)
            models.append(dict(model=name,roots=1000,literal_forward=True,max_error=worst,model_file_sha256=sha(path),weights_sha256=evaluator.model.id))
    return dict(models=models,models_unchanged=True,new_training_performed=False)

def benchmark_audit(plan):
    benchmark=module('puct_cpu_benchmark_full_audit','tools/benchmark-puct-cpu.py');implementation,witness=benchmark.admit(plan)
    report=load(benchmark.DEFAULT_OUTPUT);dbpath=benchmark.DEFAULT_RUN/'work.sqlite3';cases=games=choices=0;bycomparison=defaultdict(list)
    with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)) as db:
        text,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();identity=json.loads(text)
        require(config.digest(identity)==digest and identity==report['identity'] and identity['implementation']==implementation and identity['plan_sha256']==config.digest(plan),'Benchmark identity differs')
        for index,body,digest in db.execute('SELECT idx,body,hash FROM cases ORDER BY idx'):
            require(index==cases and config.digest(json.loads(body))==digest,'Case prefix/hash differs');case=json.loads(body);benchmark.validate_case(case,index,plan);cases+=1
        comparisons={c['id']:c for c in plan['arena_comparisons']}
        for index,name,pair,seat,body,digest in db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'):
            require(index==games and plan['arena_work_order'][index]==dict(comparison=name,pair=pair,seat=seat),'Arena declared prefix differs')
            game=json.loads(body);require(config.digest(game)==digest,'Arena checksum differs');benchmark.replay_game(game,comparisons[name],pair,seat,plan,benchmark.Native())
            q=normalize_literal(game['start']);fallbacks=Counter()
            for action,row in zip(game['moves'],game['decisions']):
                require(row['actor']==q.controller and action in literal.legal_actions(q),'Literal actor/action differs')
                if row['fallback']:require(action==literal.legal_actions(q)[0],'Fallback action differs');fallbacks[row['engine']]+=1
                q=literal.apply(q,action).position;choices+=1
            require(json_position(q)==game['final'] and dict(fallbacks)==game['fallbacks'],'Literal arena result differs')
            terminal=literal.normalize(q).terminal;margin=literal.evaluate(q)*(1 if seat==0 else -1)
            require(game['status']==('terminal' if terminal else 'unfinished') and game['candidate_final_margin']==(margin if terminal else None),'Literal arena payoff differs')
            bycomparison[name].append(game);games+=1
        require((cases,games)==(80,576) and report==benchmark.summarize(db,identity,plan),'Benchmark complete publication differs')
    primary={}
    for name,grouplist in bycomparison.items():
        margins=[]
        for pair in range(24):
            group=[g for g in grouplist if g['pair']==pair];require(len(group)==2 and {g['candidate_seat'] for g in group}=={0,1},'Unpaired arena family')
            if all(g['status']=='terminal' for g in group):margins.append(sum(g['candidate_final_margin'] for g in group)/2)
        rng=random.Random(20261017);samples=sorted(sum(rng.choices(margins,k=len(margins)))/len(margins) for _ in range(4096)) if margins else []
        interval=[samples[102],samples[3993]] if samples else None;row=report['arenas'][name];mean=statistics.mean(margins) if margins else None
        gate=bool(len(margins)==24 and not sum(sum(g['fallbacks'].values()) for g in grouplist) and interval and interval[0]>0)
        require(row['primary_paired_mean_final_margin_seeds']==mean and row['primary_margin_bootstrap_95_interval']==interval and row['primary_strength_gate_passed']==gate,'Independent paired family interval/gate differs')
        primary[name]=dict(mean_margin_seeds=mean,interval=interval,gate_passed=gate,complete_pairs=len(margins))
    require(report['default_promotion'] is False and report['sealed_final_test_consumed'] is False and report['energy_measured'] is False,'Unsupported release/energy claim')
    return dict(cases=80,scalar_measurements=1280,games=576,literal_choices=choices,primary=primary,all_source_and_cpu_identity=True,whole_decision_clock_checked=True)

def speed_audit(plan):
    from congklak.phase6_inference import BatchEvaluator
    from congklak.puct import search
    from congklak.native import flatten
    speed=module('puct_cpu_speed_full_audit','tools/puct-cpu-speed.py');benchmark=speed.benchmark_module();report=load(ROOT/'build/puct-cpu-speed.json');path=ROOT/report['journal']
    roots=common.speed_roots(plan);settings=plan['speed'];profiles=list(common.PROFILES);keys=[('startup',p,0,0,-1) for p in profiles]
    keys += [('parity',p,size,0,-1) for p in profiles for size in settings['batch_sizes']]
    for block in range(7):
        ordered=profiles[block%6:]+profiles[:block%6];sizes=settings['batch_sizes'];sizes=sizes[block%6:]+sizes[:block%6]
        for kind in ('cold','hot'):
            for size in sizes:
                for profile in ordered:keys.append((kind,profile,size,block,-1))
    for block in range(3):
        for index in range(48):
            for profile in profiles[block%6:]+profiles[:block%6]:keys.append(('tree',profile,1,block,index))
    saved={}
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        text,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();identity=json.loads(text)
        require(config.digest(identity)==digest and identity==report['identity'],'Speed ledger identity differs')
        for kind,profile,size,block,index,text,digest in db.execute('SELECT kind,profile,batch,block,root,body,hash FROM units'):
            key=(kind,profile,size,block,index);body=json.loads(text);require(key in keys and key not in saved and config.digest(body)==digest,'Speed unit key/hash differs');speed.check(key,body,plan,benchmark);saved[key]=body
    require(len(saved)==1410 and set(saved)==set(keys) and report==speed.summarize(saved,identity,plan,path),'Speed full coverage/publication differs')
    rootdata={};reference_trees={}
    for modelname,modelpath in [('warm',common.WARM),('updated',common.UPDATE)]:
        with BatchEvaluator(modelpath,'native',max_batch=1) as evaluator:
            rootdata[modelname]=[]
            for p in roots:
                row=evaluator.evaluate_many([p])[0];f=evaluator.native.feature_vector(p)
                rootdata[modelname].append(dict(value=row['value_player0'],logits=row['policy_logits'],simulations=f['feature_simulations'],deposits=f['feature_deposits']))
            original=evaluator.evaluate_many;trace=[]
            def traced(positions,**kwargs):
                rows=original(positions,**kwargs)
                for p,r in zip(positions,rows):trace.append((bytes(flatten(p)+[p['controller']]),r['feature_simulations']//2,r['feature_deposits']//2))
                return rows
            evaluator.evaluate_many=traced
            for index,p in enumerate(roots):
                trace.clear();tree=search(p,evaluator,simulations=64,max_nodes=2048,rng=20261015,native=evaluator.native,memory_mb=8192);seen=set();cache_sim=cache_dep=0
                for raw,sims,deps in trace:
                    if raw not in seen:seen.add(raw);cache_sim+=sims;cache_dep+=deps
                reference_trees[(modelname,index)]=dict(search=tree,requests=len(trace),unique=len(seen),single_sim=sum(r[1] for r in trace),single_dep=sum(r[2] for r in trace),cache_sim=cache_sim,cache_dep=cache_dep)
    for key,body in saved.items():
        kind,name,size,block,index=key;modelname=name.split('_')[0];passes=2 if name.endswith('_legacy') else 1
        if kind in ('cold','hot'):
            checksum=0.;simulations=deposits=0
            for repeat in range(64):
                rows=[rootdata[modelname][(block*7+repeat+j)%48] for j in range(size)]
                checksum+=sum(row['value']+sum(row['logits'])/7 for row in rows)
                simulations+=sum(row['simulations'] for row in rows);deposits+=sum(row['deposits'] for row in rows)
            computed=not(kind=='hot' and name.endswith('_cache'));expected_sim=simulations*passes if computed else 0;expected_dep=deposits*passes if computed else 0
            require(body['checksum']==checksum and body['counts']['feature_simulations']==expected_sim and body['counts']['feature_deposits']==expected_dep,'Independent scheduled micro checksum/actual feature work differs')
        elif kind=='tree':
            reference=reference_trees[(modelname,index)];require(semantic(body['search'])==semantic(reference['search']),'Independent fixed-tree semantic differs')
            counts=body['counts'];cached=name.endswith('_cache');miss=reference['unique'] if cached else reference['requests'];hits=reference['requests']-miss
            require(counts['positions_submitted']==counts['evaluations_requested']==reference['requests'] and counts['cache_misses']==miss and counts['cache_hits']==hits and counts['native_actual_evals']==counts['predict_successes']==miss and counts['native_calls']==passes*miss and counts['cache_evictions']==0 and counts['terminal_returns']==0 and body['cache_occupancy']==(miss if cached else 0),'Independent leaf-trace cache/physical execution counts differ')
            require(counts['feature_simulations']==(reference['cache_sim'] if cached else passes*reference['single_sim']) and counts['feature_deposits']==(reference['cache_dep'] if cached else passes*reference['single_dep']),'Independent leaf-trace hypothetical feature work differs')
    ratios={}
    for name in ('warm','updated'):
        values=[]
        for index in range(48):
            values.append(statistics.median(saved[('tree',name+'_single',1,block,index)]['wall_ns']/saved[('tree',name+'_legacy',1,block,index)]['wall_ns'] for block in range(3)))
            for block in range(3):
                single=saved[('tree',name+'_single',1,block,index)];legacy=saved[('tree',name+'_legacy',1,block,index)]
                for counter in ('feature_simulations','feature_deposits'):require(2*single['counts'][counter]==legacy['counts'][counter],'Matched single-pass feature work is not exactly half')
        ratios[name]=dict(median_root_ratio=statistics.median(values),roots=48,blocks_per_root=3,point_gate_passed=statistics.median(values)<1)
    declaration=load(ROOT/'build/puct-cpu-cost-gate.json');manifest=load(common.IMPLEMENTATION)
    require(declaration['schema']=='puct-cpu-cost-gate-v1' and declaration['predeclared_before_production'] is True and declaration['models']==['warm','updated'] and manifest['science_inputs']['build/puct-cpu-cost-gate.json']==sha(ROOT/'build/puct-cpu-cost-gate.json'),'Cost gate was not implementation-pinned')
    return dict(units=1410,startup=6,parity=36,micro=504,trees=864,full_summary_rederived=True,independent_micro_work_and_checksums=True,reference_trees=96,exact_fixed_tree_semantics=True,all_single_feature_work_exactly_half=True,cache_leaf_trace_counts_rederived=True,single_vs_legacy_tree_cost=ratios,point_cost_gate_passed=all(v['point_gate_passed'] for v in ratios.values()),hot_prefill_is_retrieval_ceiling_only=True)

def resource_audit(implementation):
    observed={}
    for producer in ('speed','benchmark'):
        path=ROOT/f'build/puct-cpu-{producer}-process-resources.json';body=load(path)
        expected=dict(producer=producer,producer_sha256=sha(ROOT/implementation['roles'][producer]),wrapper_sha256=sha(ROOT/implementation['roles']['resource_wrapper']),implementation_sha256=implementation['sha256'])
        require(body['schema']=='puct-cpu-process-resources-v1' and body['identity']==expected and body['report_file_sha256']==sha(ROOT/implementation['artifacts'][producer+'_output']),'Outer original process resource identity differs')
        duration=body['elapsed_seconds_including_module_load'];require(type(duration) in (int,float) and math.isfinite(duration) and duration>0,'Invalid original process observation clock')
        require(body['torch_imported'] is False and body['cuda_used'] is False and body['energy_measured'] is False,'Unsupported CPU resource claim')
        require(set(body['prior_committed_units'])<=set(('units','cases','games')) and all(type(v) is int and v>=0 for v in body['prior_committed_units'].values()),'Invalid prior completed unit observation')
        require(set(body['host'])=={'peak_working_set_bytes','working_set_bytes','private_commit_bytes','peak_private_commit_bytes'} and all(type(v) is int and v>0 for v in body['host'].values()),'Invalid peak host memory observation')
        require(body['host']['peak_working_set_bytes']>=body['host']['working_set_bytes'] and body['host']['peak_private_commit_bytes']>=body['host']['private_commit_bytes'],'Peak host observation is smaller than current memory')
        observed[producer]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),elapsed_seconds=duration,prior_committed_units=body['prior_committed_units'],host=body['host'])
    path=ROOT/'build/puct-cpu-benchmark-resources.json';inner=load(path)
    require(inner['schema']=='puct-cpu-benchmark-resources-v1' and inner['implementation_sha256']==implementation['sha256'] and inner['report_file_sha256']==sha(ROOT/implementation['artifacts']['benchmark_output']),'Inner original benchmark resource identity differs')
    require(0<inner['elapsed_seconds_this_invocation']<=observed['benchmark']['elapsed_seconds'] and inner['torch_imported'] is False and inner['energy_measured'] is False,'Inner/outer original observation clock or CPU scope differs')
    for field,table in [('prior_committed_cases','cases'),('prior_committed_games','games')]:require(inner[field]==observed['benchmark']['prior_committed_units'].get(table,0),'Inner/outer previous invocation counts differ')
    observed['benchmark_inner']=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),elapsed_seconds=inner['elapsed_seconds_this_invocation'],host=inner['host'])
    return dict(observations=observed,original_invocations_only=True,python_startup_and_previous_invocations_excluded=True,energy_measured=False)

def full():
    started=time.monotonic();result={'passed':False};paths=[ROOT/p for p in SOURCES]+[common.PLAN,common.IMPLEMENTATION,ROOT/'build/phase6-acceptance.json',ROOT/'build/stage5-deeper-acceptance.json']
    try:
        result['tests']=collect();plan,implementation=common.admit();result['plan']=dict(cases=80,scalar_measurements=1280,games=576,comparisons=12,arena_families=24,development_reuse=True)
        result['models']=model_audit();result['benchmark']=benchmark_audit(plan);result['speed']=speed_audit(plan);result['resources']=resource_audit(implementation);result['prior_artifacts']=prior_audit()
        paths += [WORK/'tests.json']+[WORK/(name+'-tests.json') for name in GROUPS]
        paths += [ROOT/path for name in GROUPS for path in load(WORK/(name+'-tests.json'))['inputs']]
        paths += [ROOT/path for path in implementation['sources']]+[ROOT/path for path in implementation['science_inputs']]
        paths += [ROOT/'build/puct-cpu-speed.json',ROOT/'data/puct-cpu-speed-v1/work.sqlite3',ROOT/'build/puct-cpu-benchmark.json',ROOT/'data/puct-cpu-benchmark-v1/work.sqlite3']
        paths += [ROOT/path for path in ('build/puct-cpu-cost-gate.json','build/puct-cpu-speed-process-resources.json','build/puct-cpu-benchmark-process-resources.json','build/puct-cpu-benchmark-resources.json')]
        result.update(passed=True,inputs=input_map(paths),default_promotion=False,sealed_target_bodies_read=False,energy_measured=False,training_performed=False,independent_live_warm_kill_claimed=False)
    except Exception as exc:result['error']=str(exc)
    result['duration_seconds']=time.monotonic()-started;save(WORK/'audit.json',result);print(config.canonical(dict(event='puct_cpu_independent_full_audit',passed=result['passed'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['group','collect','full']);parser.add_argument('--group',choices=list(GROUPS));args=parser.parse_args()
    if args.operation=='group':require(args.group,'Missing group');return group(args.group)
    if args.operation=='collect':collect();return 0
    return full()
if __name__=='__main__':raise SystemExit(main())
