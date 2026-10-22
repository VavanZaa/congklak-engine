"""Recoverable CPU adapter costs: empty-cache calls, retrieval ceiling and real trees."""
from __future__ import annotations
import argparse
from contextlib import ExitStack,closing
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as prior,puct_cpu_common as common
from congklak.phase6_store import Work
from congklak.phase6_inference import BatchEvaluator
from congklak.puct import search
from congklak.timing_v2 import PerformanceCore

def benchmark_module():
    spec=importlib.util.spec_from_file_location('cpu_costs_benchmark_dependency',ROOT/'tools/benchmark-puct-cpu.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def semantic(report):
    return {k:report[k] for k in ('position','chosen','visits','policy','root_priors','estimated_value_player0','action_mean_value_player0','root_expanded','completed_simulations','nodes','stop_reason','proof_status','unvisited_root_q_player0')}

def summary(values):return dict(median=statistics.median(values),min=min(values),max=max(values),samples=len(values))
def event(value):print(config.canonical(value),flush=True)

def run(path=ROOT/'data/puct-cpu-speed-v1',output=ROOT/'build/puct-cpu-speed.json',on_event=event):
    plan,manifest=common.admit();benchmark=benchmark_module();benchmark.admit(plan)
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    roots=common.speed_roots(plan);settings=plan['speed'];profiles=list(common.PROFILES)
    with PerformanceCore(plan['cpu_declared']) as cpu,ExitStack() as stack:
        engines={}
        for name in profiles:
            info=plan['profiles'][name];model=ROOT/plan['models'][info['model']]
            obj=stack.enter_context(BatchEvaluator(model,'native',max_batch=32)) if info['evaluator']=='legacy' else stack.enter_context(CpuBatchEvaluator(model,cache_capacity=info['cache_capacity'],max_batch=32))
            engines[name]=benchmark.LegacyTelemetry(obj) if info['evaluator']=='legacy' else obj
        identity=dict(schema='puct-cpu-speed-work-v1',implementation_sha256=manifest['sha256'],plan_sha256=config.digest(plan),cpu=cpu.metadata,
                      roots_sha256=config.digest(roots),evaluators={k:v.identity for k,v in engines.items()},source_sha256=prior.sha(__file__))
        keys=[('startup',name,0,0,-1) for name in profiles]
        keys += [('parity',name,size,0,-1) for name in profiles for size in settings['batch_sizes']]
        for block in range(settings['blocks']):
            rotated=profiles[block%6:]+profiles[:block%6];sizes=settings['batch_sizes'];sizes=sizes[block%6:]+sizes[:block%6]
            for workload in ('cold','hot'):
                for size in sizes:
                    for name in rotated:keys.append((workload,name,size,block,-1))
        for block in range(settings['tree_blocks']):
            rotated=profiles[block%6:]+profiles[:block%6]
            for index in range(len(roots)):
                for name in rotated:keys.append(('tree',name,1,block,index))
        with closing(Work(path,identity)) as work:
            db=work.db;db.execute('CREATE TABLE IF NOT EXISTS units(kind TEXT,profile TEXT,batch INTEGER,block INTEGER,root INTEGER,body TEXT,hash TEXT,PRIMARY KEY(kind,profile,batch,block,root))');db.commit()
            saved={}
            for row in db.execute('SELECT * FROM units'):
                key=tuple(row[k] for k in ('kind','profile','batch','block','root'));body=json.loads(row['body'])
                prior.require(key in keys and config.digest(body)==row['hash'],'Corrupt speed unit/key');check(key,body,plan,benchmark);saved[key]=body
            # Only the declared prefix can survive a killed run.
            prior.require(set(saved)==set(keys[:len(saved)]),'Speed journal contains an orphan/nonprefix unit')
            def put(key,body):
                check(key,body,plan,benchmark);work.admit(1024*1024)
                with db:db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?)',(*key,config.canonical(body),config.digest(body)))
                saved[key]=body;on_event(dict(event='puct_cpu_speed_unit',completed=len(saved),total=len(keys),kind=key[0],profile=key[1],batch=key[2]))
            on_event(dict(event='puct_cpu_speed_started',completed=len(saved),total=len(keys)))
            for key in keys:
                if key in saved:continue
                kind,name,size,block,index=key;obj=engines[name];cpu.verify()
                if kind=='startup':body=dict(startup_ms=obj.startup_ms)
                elif kind=='parity':
                    obj.reset_cache();errors=[0.,0.]
                    for offset in range(0,len(roots),size):
                        states=roots[offset:offset+size];actual=obj.evaluate_many(states)
                        for p,row in zip(states,actual):
                            reference=obj.uninstrumented_reference(p) if hasattr(obj,'uninstrumented_reference') else obj.native.evaluate(p)
                            errors[0]=max(errors[0],abs(reference['model_value_player0']-row['value_player0']))
                            errors[1]=max(errors[1],max(abs(a-b) for a,b in zip(reference['policy_logits'],row['policy_logits'])))
                    body=dict(roots=len(roots),max_value_error=errors[0],max_logit_error=errors[1],passed=errors[0]<=1e-10 and errors[1]<=1e-10)
                elif kind in ('cold','hot'):
                    # The model is loaded/warmed in both cases; cold means an
                    # empty evaluation cache, not process startup.
                    obj.reset_cache()
                    for _ in range(4):obj.evaluate_many(roots[:size])
                    if kind=='hot':
                        for offset in range(0,len(roots),size):obj.evaluate_many(roots[offset:offset+size])
                    before=benchmark.counter_snapshot(obj);times=[];checksum=0.
                    for repeat in range(settings['repetitions']):
                        states=[roots[(block*7+repeat+j)%len(roots)] for j in range(size)]
                        tick=time.perf_counter_ns()
                        if kind=='cold':obj.reset_cache()
                        rows=obj.evaluate_many(states);times.append(time.perf_counter_ns()-tick)
                        checksum+=sum(row['value_player0']+sum(row['policy_logits'])/7 for row in rows)
                    after=benchmark.counter_snapshot(obj)
                    body=dict(whole_call_ns=statistics.mean(times),position_ns=statistics.mean(times)/size,repetitions=settings['repetitions'],checksum=checksum,
                              counts={k:after[k]-before[k] for k in benchmark.COUNTERS},cache_capacity=after['cache_capacity'],cache_occupancy=after['cache_occupancy'])
                else:
                    before=benchmark.counter_snapshot(obj);tick=time.perf_counter_ns();obj.reset_cache()
                    report=search(roots[index],obj,simulations=settings['tree_simulations'],max_nodes=settings['tree_max_nodes'],rng=20261015,native=obj.native,memory_mb=plan['memory_mb'])
                    elapsed=time.perf_counter_ns()-tick;after=benchmark.counter_snapshot(obj)
                    body=dict(wall_ns=elapsed,search=report,counts={k:after[k]-before[k] for k in benchmark.COUNTERS},cache_capacity=after['cache_capacity'],cache_occupancy=after['cache_occupancy'])
                cpu.verify();put(key,body)
            result=summarize(saved,identity,plan,path)
            done=work.get('puct_cpu_speed_complete')
            if done:prior.require(done==dict(units=len(keys),report_sha256=config.digest(result)),'Completed speed report differs')
            prior.publish(Path(output),result)
            if not done:
                with db:work.put('puct_cpu_speed_complete',dict(units=len(keys),report_sha256=config.digest(result)))
            on_event(dict(event='puct_cpu_speed_complete',units=len(keys),fixed_tree_semantics_equal=result['fixed_tree_semantics_equal']));return result

def check(key,body,plan,benchmark):
    kind,name,size,block,index=key;require=prior.require
    if kind=='startup':require(set(body)=={'startup_ms'} and benchmark.numeric(body['startup_ms']),'Invalid startup');return
    if kind=='parity':
        require(set(body)=={'roots','max_value_error','max_logit_error','passed'} and body['roots']==48 and body['passed'] is True and benchmark.numeric(body['max_value_error'],0,1e-10) and benchmark.numeric(body['max_logit_error'],0,1e-10),'Speed parity failed');return
    counts=body['counts'];require(set(counts)==set(benchmark.COUNTERS) and all(benchmark.integer(v) for v in counts.values()),'Corrupt speed counters')
    capacity=plan['profiles'][name]['cache_capacity'];require(body['cache_capacity']==capacity and benchmark.integer(body['cache_occupancy'],0,capacity),'Unbounded speed cache')
    require(counts['evaluations_requested']==counts['cache_hits']+counts['cache_misses'] and counts['predict_successes']==counts['native_actual_evals']==counts['cache_misses'],'Completed speed work conservation differs')
    passes=plan['profiles'][name]['physical_feature_passes_per_miss'];require(counts['native_calls']==passes*counts['native_actual_evals'],'Actual speed native call accounting differs')
    require(passes*counts['cache_misses']<=counts['feature_simulations']<=14*passes*counts['cache_misses'] and counts['feature_deposits']>=counts['feature_simulations'],'Completed physical feature work is impossible')
    require(counts['cache_evictions']<=counts['native_actual_evals'],'Too many cache evictions')
    if not capacity:require(counts['cache_hits']==counts['cache_evictions']==body['cache_occupancy']==0,'Disabled speed cache has hits')
    if kind in ('cold','hot'):
        require(set(body)=={'whole_call_ns','position_ns','repetitions','checksum','counts','cache_capacity','cache_occupancy'} and benchmark.numeric(body['whole_call_ns'],1) and body['position_ns']==body['whole_call_ns']/size and body['repetitions']==plan['speed']['repetitions'] and benchmark.numeric(body['checksum'],-float('inf')),'Invalid speed timings')
        require(counts['positions_submitted']==counts['evaluations_requested']==size*body['repetitions'] and not counts['terminal_returns'],'Speed roots/evaluation coverage differs')
        if kind=='cold':require(counts['cache_resets']==body['repetitions'] and not counts['cache_hits'] and body['cache_occupancy']<=size,'Cold workload has cache credit')
        else:
            require(counts['cache_resets']==0,'Hot workload resets inside measurements')
            if capacity:require(counts['cache_hits']==counts['evaluations_requested'] and counts['native_calls']==counts['feature_simulations']==counts['feature_deposits']==0,'Hot retrieval performs hidden computation')
    elif kind=='tree':
        require(set(body)=={'wall_ns','search','counts','cache_capacity','cache_occupancy'} and benchmark.numeric(body['wall_ns'],1) and counts['cache_resets']==1,'Invalid tree clock/reset')
        report=body['search'];from congklak.phase6_selfplay import validate_search
        validate_search(report,common.speed_roots(plan)[index],plan['speed']['tree_simulations'])
        require(report['completed_simulations']==plan['speed']['tree_simulations'] and report['work']['feature_simulations']==counts['feature_simulations'] and report['work']['feature_deposits']==counts['feature_deposits'],'Fixed tree incomplete or work mislabeled')
    else:raise ValueError('Unknown speed workload')

def summarize(saved,identity,plan,path):
    profiles={};equal=True
    for name in common.PROFILES:
        micro={}
        for workload in ('cold','hot'):
            micro[workload]={}
            for size in plan['speed']['batch_sizes']:
                units=[saved[(workload,name,size,block,-1)] for block in range(plan['speed']['blocks'])]
                micro[workload][str(size)]=dict(whole_call_ns=summary([u['whole_call_ns'] for u in units]),position_ns=summary([u['position_ns'] for u in units]),counts={k:sum(u['counts'][k] for u in units) for k in units[0]['counts']})
        trees=[saved[('tree',name,1,block,index)] for block in range(3) for index in range(48)]
        profiles[name]=dict(startup=saved[('startup',name,0,0,-1)],micro=micro,trees=dict(roots=48,blocks=3,wall_ns=summary([t['wall_ns'] for t in trees]),counts={k:sum(t['counts'][k] for t in trees) for k in trees[0]['counts']}))
        for block in range(3):
            for index in range(48):
                reference=saved[('tree',name.split('_')[0]+'_legacy',1,block,index)]
                equal=equal and semantic(reference['search'])==semantic(saved[('tree',name,1,block,index)]['search'])
    prior.require(equal,'Optimization changed fixed-budget tree semantics')
    return dict(schema='puct-cpu-speed-results-v1',complete=True,identity=identity,units=len(saved),journal=str(Path(path).resolve().relative_to(ROOT)).replace('\\','/'),profiles=profiles,
                fixed_tree_semantics_equal=equal,scope='Loaded/warmed models; cold caches reset inside measured calls. Hot prefill outside call clock is retrieval ceiling only. Tree clocks include reset; physical counters separate from logical requests. No energy or unseen/strength claim.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',default=str(ROOT/'data/puct-cpu-speed-v1'));parser.add_argument('--output',default=str(ROOT/'build/puct-cpu-speed.json'));args=parser.parse_args();run(args.run,args.output)

if __name__=='__main__':main()
