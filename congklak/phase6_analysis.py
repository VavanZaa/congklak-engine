"""Recoverable experimental PUCT recommendations; unfinished trees are recomputed."""
from __future__ import annotations
from contextlib import closing
import argparse
import json
from pathlib import Path
import signal
import time
from . import config,pilot
from . import phase6_common as common
from .native import Native,flatten
from .puct import search
from .phase6_inference import BatchEvaluator
from .phase6_selfplay import event,unpack,validate_search
from .phase6_store import Work

def describe(report,p,native):
    ranking=sorted((dict(pit=a+1,visits=report['visits'][a],visit_fraction=report['policy'][a],estimated_margin_player0=report['action_mean_value_player0'][a])
                    for a in range(7) if p['pits'][p['controller']][a]),key=lambda r:(-r['visits'],r['pit']))
    explanation=None
    if report['chosen'] is not None:
        t=native.apply(flatten(p),actor=p['controller'],action=report['chosen'],total=98,trace=True)
        common.require(t.get('ok'),'Invalid recommendation trace');explanation=dict(reason=t['reason'],drops=t['drops'],passes=t['passes'],after=t['position'])
    return ranking,explanation

def analyze(path,position,modelpath=common.WARM,backend='native',simulations=64,cancel=None,on_event=None):
    config.integer(simulations,'simulations',1,100000);native=Native();p=common.normalize(position,native)
    with BatchEvaluator(modelpath,backend,max_batch=1) as evaluator:
        identity=dict(schema='phase6-resumable-puct-analysis-v1',position=p,rules_digest=config.RULES_DIGEST,model=evaluator.identity,
                      source_sha256=common.sha(__file__),puct_sha256=common.sha(Path(__file__).with_name('puct.py')),validation_sha256=common.sha(Path(__file__).with_name('phase6_selfplay.py')),store_sha256=common.sha(Path(__file__).with_name('phase6_store.py')),max_nodes=2048,seed=20261015,
                      resume='Preserve latest completed recommendation; recompute unfinished tree from fixed seed')
        work=Work(path,identity)
        try:
            prior=work.get('analysis_result')
            if prior:
                q=prior['search'];validate_search(q,p,q['requested_simulations'],native=native);ranking,explanation=describe(q,p,native)
                common.require(prior['status']=='complete' and prior['identity_sha256']==config.digest(identity) and prior['rankings']==ranking and prior['explanation']==explanation and prior['proof_status']==q['proof_status'],'Corrupt saved analysis')
                common.publish(Path(path)/f"analysis-{q['requested_simulations']}.json",prior)
                if q['completed_simulations']>=simulations or q['stop_reason']=='terminal':return prior
            event(dict(event='phase6_analysis_started',committed_decisions=0,simulations=simulations),on_event)
            report=search(p,evaluator,simulations=simulations,max_nodes=2048,rng=20261015,cancel=cancel,native=native)
            if (report['stop_reason']!='terminal' and report['completed_simulations']!=simulations) or not report['usable_decision']:
                return dict(status='interrupted',previous_completed=prior,unfinished_search_work=report['work'])
            validate_search(report,p,simulations,native=native);ranking,explanation=describe(report,p,native)
            result=dict(status='complete',identity_sha256=config.digest(identity),search=report,rankings=ranking,explanation=explanation,
                        uncertainty='Visit fractions are search allocation, not calibrated win probabilities',proof_status=report['proof_status'])
            event(dict(event='phase6_analysis_precommit',committed_decisions=1,simulations=simulations),on_event)
            with work.db:work.put('analysis_result',result)
            event(dict(event='phase6_analysis_committed',committed_decisions=1,simulations=simulations),on_event)
            common.publish(Path(path)/f'analysis-{simulations}.json',result);return result
        finally:work.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--path',required=True);p.add_argument('--position');p.add_argument('--model',default=str(common.WARM));p.add_argument('--backend',choices=('native','cuda'),default='native');p.add_argument('--simulations',type=int,default=64);a=p.parse_args()
    position=json.loads(Path(a.position).read_text(encoding='utf-8')) if a.position else dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0)
    stopped=False
    def stop(*_):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,stop)
    result=analyze(a.path,position,a.model,a.backend,a.simulations,cancel=lambda:stopped,on_event=lambda v:print(config.canonical(v),flush=True));print(config.canonical(result),flush=True)

if __name__=='__main__':main()
