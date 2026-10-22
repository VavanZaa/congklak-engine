"""Durable value-experiment recommendations; unfinished search is recomputed."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
from . import config,value_common as common,quality_search
from .native import Native
from .puct_cpu_inference import CpuBatchEvaluator
from .puct_cpu_analysis import _delta,validate_counters as previous_validate_counters
from .phase6_analysis import describe
from .phase6_store import Work

CLOCK='Cache reset, search, validation and explanation; excludes admission/model startup and publication'
UNCERTAINTY='Visit fractions are allocation, not calibrated probabilities; values and moves are estimates'

def validate_counters(value,report,capacity):
    previous_validate_counters(value,report,capacity)
    common.require(value['feature_simulations']>=value['native_actual_evals'] and value['feature_deposits']>=value['feature_simulations'],'Impossible completed physical feature work')

def event(value,on_event=None):
    if on_event:on_event(value)
    if os.environ.get('VALUE_ANALYSIS_BLOCK_EVENT')==value['event']:
        marker=os.environ.get('VALUE_ANALYSIS_MARKER')
        if marker:common.teacher.atomic_json(Path(marker),value)
        while True:time.sleep(.05)

def analyze(path,position,profile='value_refit',simulations=128,cancel=None,on_event=None):
    plan,manifest=common.admit();config.integer(simulations,'simulations',1,100000)
    common.require(profile in common.PROFILES,'Unknown value profile');settings=plan['profiles'][profile]
    witness=None
    if settings['model']=='refit':
        from .value_learning import witness as trained_witness
        witness=trained_witness()
    native=Native();p=common.prior.normalize(position,native)
    with CpuBatchEvaluator(common.ROOT/plan['models'][settings['model']],cache_capacity=0,max_batch=1) as evaluator:
        identity=dict(schema='value-resumable-analysis-v1',position=p,profile=profile,settings=settings,rules_digest=config.RULES_DIGEST,
                      model=evaluator.identity,trained_witness=witness,implementation_sha256=manifest['sha256'],plan_sha256=config.digest(plan),
                      max_nodes=2048,seed=20261015,resume='Retain latest completed recommendation; recompute unfinished tree; empty cache each attempt')
        work=Work(path,identity)
        try:
            saved=work.get('analysis_result')
            if saved:
                common.require(set(saved)=={'status','identity_sha256','search','rankings','explanation','evaluator','uncertainty','proof_status','attempt_ms','clock_scope'},'Unknown value analysis fields')
                q=saved['search'];quality_search.validate_report(q,p,q['requested_simulations'],c_puct=settings['c_puct'],native=native,allow_interrupted=False)
                ranking,explanation=describe(q,p,native);validate_counters(saved['evaluator'],q,0)
                common.require(saved['status']=='complete' and saved['identity_sha256']==config.digest(identity) and saved['rankings']==ranking and saved['explanation']==explanation and saved['proof_status']==q['proof_status'] and saved['uncertainty']==UNCERTAINTY,'Corrupt saved value analysis')
                common.require(type(saved['attempt_ms']) in (int,float) and math.isfinite(saved['attempt_ms']) and saved['attempt_ms']>=q['timing']['elapsed_ms'] and saved['clock_scope']==CLOCK,'Invalid analysis clock')
                common.publish(Path(path)/f"analysis-{q['requested_simulations']}.json",saved)
                if q['completed_simulations']>=simulations or q['stop_reason']=='terminal':return saved
            event(dict(event='value_analysis_started',profile=profile,simulations=simulations),on_event)
            started=time.perf_counter();before=evaluator.snapshot_counters();evaluator.begin_decision()
            q=quality_search.search(p,evaluator,simulations=simulations,max_nodes=2048,c_puct=settings['c_puct'],rng=20261015,cancel=cancel,native=evaluator.native)
            counts=_delta(before,evaluator.snapshot_counters())
            quality_search.validate_report(q,p,simulations,c_puct=settings['c_puct'],native=native)
            if (q['stop_reason']!='terminal' and q['completed_simulations']!=simulations) or not q['usable_decision']:
                return dict(status='interrupted',previous_completed=saved,unfinished_search_work=q['work'],evaluator=counts)
            ranking,explanation=describe(q,p,native);validate_counters(counts,q,0)
            result=dict(status='complete',identity_sha256=config.digest(identity),search=q,rankings=ranking,explanation=explanation,evaluator=counts,
                        uncertainty=UNCERTAINTY,proof_status=q['proof_status'],
                        attempt_ms=(time.perf_counter()-started)*1000,clock_scope=CLOCK)
            event(dict(event='value_analysis_precommit',simulations=simulations),on_event)
            with work.db:work.put('analysis_result',result)
            event(dict(event='value_analysis_committed',simulations=simulations),on_event)
            common.publish(Path(path)/f'analysis-{simulations}.json',result);return result
        finally:work.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--path',required=True);parser.add_argument('--profile',choices=common.PROFILES,default='value_refit');parser.add_argument('--position');parser.add_argument('--simulations',type=int,default=128);args=parser.parse_args()
    position=json.loads(Path(args.position).read_text(encoding='utf-8')) if args.position else dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0)
    token=quality_search.CancellationToken();signal.signal(signal.SIGINT,lambda *_:token.cancel())
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,lambda *_:token.cancel())
    result=analyze(args.path,position,args.profile,args.simulations,token,lambda v:print(config.canonical(v),flush=True));print(config.canonical(result),flush=True)
    return 0 if result['status']=='complete' else 130
if __name__=='__main__':raise SystemExit(main())
