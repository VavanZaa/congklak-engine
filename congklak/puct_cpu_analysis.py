"""Recoverable CPU PUCT analysis with exact single-pass inference and optional cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import math
import time
from . import config,phase6_common as prior,puct_cpu_common as common
from .native import Native
from .puct import search,CancellationToken
from .phase6_analysis import describe
from .phase6_selfplay import event,validate_search
from .phase6_store import Work

def _counters(evaluator):return evaluator.metrics_snapshot()

def _delta(before,after):
    result={key:after[key]-before[key] for key in after if key not in ('cache_current_entries','cache_capacity')}
    result.update(cache_current_entries=after['cache_current_entries'],cache_capacity=after['cache_capacity'])
    return result

def validate_counters(value,report,capacity):
    from .puct_cpu_inference import COUNTERS
    prior.require(isinstance(value,dict) and set(value)==set(COUNTERS)|{'cache_capacity','cache_current_entries'},'Unknown saved analysis counters')
    prior.require(all(type(v) is int and v>=0 for v in value.values()),'Invalid saved analysis counters')
    prior.require(value['cache_capacity']==capacity and value['cache_current_entries']<=capacity and value['cache_resets']==1,'Invalid saved cache bound/reset')
    prior.require(value['positions_submitted']==value['evaluations_requested']==value['cache_hits']+value['cache_misses']==report['work']['neural_evaluations'] and value['terminal_returns']==0,'Saved logical request accounting differs')
    prior.require(value['native_calls']==value['native_actual_evals']==value['predict_successes']==value['cache_misses'],'Saved physical forward accounting differs')
    prior.require(value['feature_simulations']==report['work']['feature_simulations'] and value['feature_deposits']==report['work']['feature_deposits'] and value['feature_simulations']<=14*value['cache_misses'],'Saved feature accounting differs')
    prior.require(value['cache_evictions']==0,'Complete bounded analysis cannot evict a4096-entry cache within2048nodes')
    if capacity:prior.require(value['cache_current_entries']==value['cache_misses'],'Saved cache occupancy differs')
    else:prior.require(value['cache_hits']==value['cache_current_entries']==0,'Disabled cache has hits')

def analyze(path,position,modelpath=common.WARM,cache_capacity=4096,simulations=64,cancel=None,on_event=None):
    plan,manifest=common.admit();config.integer(simulations,'simulations',1,100000)
    prior.require(cache_capacity in (0,4096),'Declared analysis cache capacity is0or4096')
    modelpath=Path(modelpath).resolve()
    prior.require(modelpath in {common.ROOT/name for name in plan['models'].values()},'Use a declared warm/update model')
    from .puct_cpu_inference import CpuBatchEvaluator
    native=Native();p=prior.normalize(position,native)
    with CpuBatchEvaluator(modelpath,cache_capacity=cache_capacity,max_batch=1) as evaluator:
        identity=dict(schema='puct-cpu-resumable-analysis-v1',position=p,rules_digest=config.RULES_DIGEST,model=evaluator.identity,
                      implementation_sha256=manifest['sha256'],plan_sha256=config.digest(plan),source_sha256=prior.sha(__file__),
                      max_nodes=2048,seed=20261015,resume='Retain latest completed recommendation; recompute unfinished tree; reset cache each attempt')
        work=Work(path,identity)
        try:
            saved=work.get('analysis_result')
            if saved:
                prior.require(set(saved)=={'status','identity_sha256','search','rankings','explanation','evaluator','logical_request_counter','uncertainty','proof_status','attempt_ms','clock_scope'},'Unknown saved analysis fields')
                q=saved['search'];validate_search(q,p,q['requested_simulations'],native=native)
                ranking,explanation=describe(q,p,native)
                prior.require(saved['status']=='complete' and saved['identity_sha256']==config.digest(identity) and saved['rankings']==ranking and saved['explanation']==explanation and saved['proof_status']==q['proof_status'],'Corrupt CPU analysis')
                validate_counters(saved['evaluator'],q,cache_capacity)
                prior.require(type(saved['attempt_ms']) in (int,float) and math.isfinite(saved['attempt_ms']) and saved['attempt_ms']>=q['timing']['elapsed_ms'],'Saved whole-attempt clock differs')
                prior.require(saved['clock_scope']=='Cache reset, search, validation and explanation; excludes admission/model startup and publication','Unknown saved clock scope')
                prior.publish(Path(path)/f"analysis-{q['requested_simulations']}.json",saved)
                if q['completed_simulations']>=simulations or q['stop_reason']=='terminal':return saved
            event(dict(event='puct_cpu_analysis_started',simulations=simulations),on_event)
            tick=time.perf_counter();before=_counters(evaluator);evaluator.begin_decision()
            report=search(p,evaluator,simulations=simulations,max_nodes=2048,rng=20261015,cancel=cancel,native=evaluator.native)
            counters=_delta(before,_counters(evaluator))
            if (report['stop_reason']!='terminal' and report['completed_simulations']!=simulations) or not report['usable_decision']:
                return dict(status='interrupted',previous_completed=saved,unfinished_search_work=report['work'],evaluator=counters)
            validate_search(report,p,simulations,native=native);ranking,explanation=describe(report,p,native)
            validate_counters(counters,report,cache_capacity)
            result=dict(status='complete',identity_sha256=config.digest(identity),search=report,rankings=ranking,explanation=explanation,evaluator=counters,
                        logical_request_counter='search.work.neural_evaluations; physical forwards are evaluator.native_actual_evals',
                        uncertainty='Visit fractions are search allocation, not calibrated win probabilities',proof_status=report['proof_status'],
                        attempt_ms=(time.perf_counter()-tick)*1000,clock_scope='Cache reset, search, validation and explanation; excludes admission/model startup and publication')
            event(dict(event='puct_cpu_analysis_precommit',simulations=simulations),on_event)
            with work.db:work.put('analysis_result',result)
            event(dict(event='puct_cpu_analysis_committed',simulations=simulations),on_event)
            prior.publish(Path(path)/f'analysis-{simulations}.json',result);return result
        finally:work.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--path',required=True);parser.add_argument('--position');parser.add_argument('--model',default=str(common.WARM));parser.add_argument('--cache-capacity',type=int,choices=(0,4096),default=4096);parser.add_argument('--simulations',type=int,default=64)
    args=parser.parse_args();position=json.loads(Path(args.position).read_text(encoding='utf-8')) if args.position else dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0)
    token=CancellationToken();signal.signal(signal.SIGINT,lambda *_:token.cancel())
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,lambda *_:token.cancel())
    result=analyze(args.path,position,args.model,args.cache_capacity,args.simulations,cancel=token,on_event=lambda row:print(config.canonical(row),flush=True));print(config.canonical(result),flush=True)

if __name__=='__main__':main()
