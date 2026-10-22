"""Exploration ablation: unchanged control, literal backups and interrupted work."""
from __future__ import annotations
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import reference as ref
from congklak import config,puct as frozen,quality_search as quality
from congklak.native import Native,NativeError,flatten
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.phase6_inference import InferenceStopped

MODELS=('data/training-phase6-warm-v1/model.json','data/training-phase6-update-v1/model.json')
EVIDENCE={}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def position(pits,actor=0,store0=None):
    active=sum(map(sum,pits));store0=(98-active)//2 if store0 is None else store0
    return dict(pits=pits,stores=[store0,98-active-store0],controller=actor)

def prediction(p):
    r=sum(map(sum,p['pits']));actor=p['controller']
    return dict(value_player0=p['stores'][0]-p['stores'][1]+r*.25*(1 if actor==0 else -1),
                policy_logits=[(6-a)/10 for a in range(7)] if r else [0.]*7,
                feature_simulations=0,feature_deposits=0)

class Fixture:
    max_batch=8
    def __init__(self):self.batches=[]
    def evaluate_many(self,positions):
        self.batches.append(len(positions));return [prediction(p) for p in positions]

def literal_search(p,simulations,c):
    """Standalone transition interpreter and locally recomputed selection scores."""
    def make(q):
        norm=ref.normalize(q);q=ref.Position(7,q.counts,norm.controller)
        return dict(p=q,legal=ref.legal_actions(q),terminal=norm.terminal,children={},
                    count=[0]*7,totals=[0.]*7,prior=None,value=(q.counts[7]-q.counts[15])/98)
    def expand(node):
        q=node['p'];row=prediction(dict(pits=[list(q.counts[:7]),list(q.counts[8:15])],stores=[q.counts[7],q.counts[15]],controller=q.controller))
        top=max(row['policy_logits'][a] for a in node['legal'])
        z=sum(math.exp(row['policy_logits'][a]-top) for a in node['legal'])
        node['prior']=[math.exp(row['policy_logits'][a]-top)/z if a in node['legal'] else 0 for a in range(7)]
        node['value']=row['value_player0']/98
    root=make(ref.Position(7,tuple(flatten(p)),p['controller']))
    if root['terminal']:return root['count'],[None]*7
    expand(root)
    for _ in range(simulations):
        node=root;path=[]
        while not node['terminal'] and node['prior'] is not None:
            total=sum(node['count']);scores={}
            for a in node['legal']:
                q=node['totals'][a]/node['count'][a] if node['count'][a] else node['value']
                scores[a]=(q if node['p'].controller==0 else -q)+c*node['prior'][a]*math.sqrt(max(1,total))/(1+node['count'][a])
            action=sorted(scores,key=lambda a:(-scores[a],a))[0];path.append((node,action))
            if action not in node['children']:node['children'][action]=make(ref.apply(node['p'],action).position)
            node=node['children'][action]
        if not node['terminal']:expand(node)
        for ancestor,action in path:ancestor['count'][action]+=1;ancestor['totals'][action]+=node['value']
    return root['count'],[root['totals'][a]/root['count'][a]*98 if root['count'][a] else None for a in range(7)]

def scientific(report):
    excluded={'schema','protocol','c_puct','timing'}
    body={k:v for k,v in report.items() if k not in excluded}
    body['work']={k:v for k,v in body['work'].items() if k!='inference_ms'}
    return body

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots=[position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]),
                   position([[0]*7,[1,2,0,0,0,0,1]],0),
                   position([[2,1,0,0,0,0,0],[0,0,1,0,0,1,0]],1),config.position()]

    def test_01_prior1368_inputs_preserved(self):
        body=json.loads((ROOT/'build/puct-cpu-acceptance.json').read_text())
        self.assertTrue(body['passed']);self.assertEqual(len(body['inputs']),1368)
        for name,digest in body['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['prior_inputs_preserved']=1368

    def test_02_c15_frozen_exact_scientific_parity_both_models(self):
        with closing(sqlite3.connect((ROOT/'data/deeper-9000-v2/work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
            development=[json.loads(text)['state'] for text, in db.execute("SELECT body FROM roots WHERE json_extract(body,'$.split')='development' ORDER BY body LIMIT 4")]
        roots=self.roots+development
        for path in MODELS:
            with CpuBatchEvaluator(ROOT/path,cache_capacity=0,max_batch=8) as e:
                before=e.snapshot_counters()
                a=frozen.forest(roots,e,simulations=64,rng=[random.Random(120+i) for i in range(8)],root_noise=dict(alpha=.3,fraction=.25))
                b=quality.forest(roots,e,simulations=64,c_puct=1.5,rng=[random.Random(120+i) for i in range(8)],root_noise=dict(alpha=.3,fraction=.25))
                self.assertEqual([scientific(x) for x in a],[scientific(x) for x in b])
                for p,row in zip(roots,b):quality.validate_report(row,p,64,c_puct=1.5,allow_interrupted=False)
                after=e.snapshot_counters();self.assertEqual(after['native_actual_evals']-before['native_actual_evals'],sum(r['work']['neural_evaluations'] for r in a+b))
        EVIDENCE['control_parity']=dict(models=2,roots_per_model=8,backups=64,root_noise_and_rng=True,exact_scientific_reports=True)

    def test_03_both_coefficients_literal_bonus_pass_and_player1(self):
        bonus=passed=False;traces=[]
        for p in self.roots:
            q=ref.Position(7,tuple(flatten(p)),p['controller']);passed|=bool(ref.normalize(q).passes)
            for a in ref.legal_actions(q):
                t=ref.apply(q,a);bonus|=t.bonus and not t.terminal;self.assertEqual(sum(t.position.counts),98)
            for c in quality.EXPLORATION:
                row=quality.search(p,Fixture(),simulations=128,c_puct=c);visits,means=literal_search(p,128,c)
                self.assertEqual(row['visits'],visits);self.assertEqual(row['action_mean_value_player0'],means)
                quality.validate_report(row,p,128,c_puct=c,allow_interrupted=False)
                traces.append(dict(controller=p['controller'],c_puct=c,visits=visits,means=means))
        self.assertTrue(bonus);self.assertTrue(passed)
        EVIDENCE['literal_backups']=dict(roots=4,coefficients=list(quality.EXPLORATION),backups=128,bonus=True,initial_forced_pass=True,player1=True,traces=traces)

    def test_04_forest_is_independent_and_noise_deterministic(self):
        roots=self.roots[:3];rngs=[random.Random(217+i) for i in range(3)];noise=dict(alpha=.3,fraction=.25)
        batch=quality.forest(roots,Fixture(),simulations=32,c_puct=.25,rng=rngs,root_noise=noise)
        for i,(p,row) in enumerate(zip(roots,batch)):
            one=quality.search(p,Fixture(),simulations=32,c_puct=.25,rng=random.Random(217+i),root_noise=noise)
            self.assertEqual(scientific({**row,'forest_size':1,'forest_batch_calls':one['forest_batch_calls']}),scientific(one))
        EVIDENCE['forest_independence']=dict(roots=3,backups=32,noise_rng_equal=True)

    def test_05_parent_fpu_store_offset_and_declared_scale(self):
        pits=[[2,1,0,0,0,0,3],[1,0,2,0,1,0,2]]
        p=position(pits,store0=43);shift=position(pits,store0=63)
        for c in quality.EXPLORATION:
            a=quality.search(p,Fixture(),simulations=64,c_puct=c);b=quality.search(shift,Fixture(),simulations=64,c_puct=c)
            self.assertEqual(a['visits'],b['visits']);self.assertEqual(a['chosen'],b['chosen']);self.assertEqual(a['root_priors'],b['root_priors'])
            self.assertAlmostEqual(b['estimated_value_player0']-a['estimated_value_player0'],40)
            self.assertEqual(a['first_play_urgency'],'parent_neural_bootstrap')
        node=quality.Node({},0,(0,1),False,expanded=True,bootstrap=.1,priors=[.1,.9]+[0]*5,visits=[10,0]+[0]*5,sums=[3.,0.]+[0]*5)
        self.assertEqual(quality._select(node,.25),1);node.actor=1;self.assertEqual(quality._select(node,.25),1)
        EVIDENCE['fpu_and_offsets']=dict(offset_seed_margin=40,coefficients=list(quality.EXPLORATION),same_visits=True)

    def test_06_terminal_node_bounds_and_invalid_inputs(self):
        terminal=position([[0]*7,[0]*7],actor=1,store0=73)
        class Bomb(Fixture):
            def evaluate_many(self,p):raise AssertionError('Terminal must skip inference')
        for c in quality.EXPLORATION:
            row=quality.search(terminal,Bomb(),c_puct=c);quality.validate_report(row,terminal,64,c_puct=c)
            self.assertEqual(row['estimated_value_player0'],48);self.assertEqual(row['work']['neural_evaluations'],0)
            row=quality.search(config.position(),Fixture(),max_nodes=1,c_puct=c);quality.validate_report(row,config.position(),64,max_nodes=1,c_puct=c)
            self.assertEqual(row['stop_reason'],'node_budget');self.assertFalse(row['usable_decision']);self.assertEqual(row['nodes'],1)
        for c in (0,.5,True,float('nan'),float('inf')):
            with self.assertRaises(ValueError):quality.search(config.position(),Fixture(),c_puct=c)
        for logits in ([float('nan')]*7,[0]*6):
            with self.assertRaises(NativeError):quality.legal_priors(logits,(0,))
        with self.assertRaises(ValueError):quality.forest(self.roots,Fixture(),rng=[random.Random(1)])
        EVIDENCE['bounds']=dict(terminal_exact=True,max_nodes_one_no_backup=True,undeclared_coefficients_refused=True)

    def test_07_initial_and_post_forward_cancellation_no_backup(self):
        token=quality.CancellationToken();token.cancel();e=Fixture()
        row=quality.search(config.position(),e,cancel=token,c_puct=.25);quality.validate_report(row,config.position(),64,c_puct=.25)
        self.assertEqual(e.batches,[]);self.assertEqual(row['completed_simulations'],0)
        with CpuBatchEvaluator(ROOT/MODELS[0],cache_capacity=0) as native:
            token=quality.CancellationToken()
            class StopAfter:
                max_batch=32;supports_cancel=True
                def __init__(self):self.native=native.native
                def evaluate_many(self,positions,**kwargs):
                    values=native.evaluate_many(positions,**kwargs);token.cancel();return values
            row=quality.search(config.position(),StopAfter(),cancel=token,c_puct=.25)
            quality.validate_report(row,config.position(),64,c_puct=.25)
            self.assertEqual(row['completed_simulations'],0);self.assertFalse(row['root_expanded']);self.assertFalse(row['usable_decision'])
            self.assertEqual(row['work']['neural_evaluations'],1);self.assertGreater(row['work']['feature_deposits'],0)
            self.assertEqual(native.snapshot_counters()['native_actual_evals'],1)
        EVIDENCE['cancelled_forward']=dict(initial_zero_work=True,post_forward_actual_work_charged=True,zero_backup=True)

    def test_08_native_transition_cancel_and_deadline_work(self):
        token=quality.CancellationToken()
        class CancelNative(Native):
            def apply(self,*args,**kwargs):
                token.cancel();self.answer=super().apply(*args,**kwargs);return self.answer
        n=CancelNative();row=quality.search(config.position(),Fixture(),cancel=token,native=n,c_puct=.25)
        self.assertFalse(n.answer['ok']);self.assertEqual(n.answer['error']['code'],'cancelled');self.assertEqual(row['work']['transitions'],0);self.assertEqual(row['completed_simulations'],0)
        quality.validate_report(row,config.position(),64,c_puct=.25)
        class Slow(Fixture):
            def evaluate_many(self,positions):time.sleep(.02);return super().evaluate_many(positions)
        row=quality.search(config.position(),Slow(),time_ms=2,c_puct=.25)
        quality.validate_report(row,config.position(),64,c_puct=.25)
        self.assertEqual(row['stop_reason'],'time_budget');self.assertFalse(row['root_expanded']);self.assertEqual(row['completed_simulations'],0);self.assertGreater(row['timing']['overshoot_ms'],1)
        EVIDENCE['native_transition_deadline']=dict(native_cancel_real=True,unfinished_transition_not_published=True,deadline_root_not_published=True,overrun_counted=True)

    def test_09_rehashed_semantic_corruption_refused(self):
        p=self.roots[0];base=quality.search(p,Fixture(),simulations=64,c_puct=.25)
        changes=[('schema','phase6-puct-report-v1'),('protocol',frozen.PROTOCOL),('c_puct',1.5),('value_perspective','actor'),('nodes',2049),('chosen',5),('estimated_value_player0',999),('stop_reason','terminal')]
        for key,value in changes:
            body=deepcopy(base);body[key]=value
            with self.assertRaises(NativeError,msg=key):quality.validate_report(body,p,64,c_puct=.25)
        for key in ('visits','policy','root_priors','action_mean_value_player0'):
            body=deepcopy(base);body[key][5]=1
            with self.assertRaises(NativeError,msg=key):quality.validate_report(body,p,64,c_puct=.25)
        body=deepcopy(base);body['work']['transitions']+=1
        with self.assertRaises(NativeError):quality.validate_report(body,p,64,c_puct=.25)
        body=deepcopy(base);body['extra']=0
        with self.assertRaises(NativeError):quality.validate_report(body,p,64,c_puct=.25)
        EVIDENCE['semantic_corruption']=dict(rehashed_forgery_refused=True,variants=14)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths=['congklak/quality_search.py','tests/quality_search_acceptance.py','congklak/puct.py','congklak/puct_cpu_inference.py','congklak/phase6_inference.py','congklak/native.py','congklak/config.py','tests/reference.py','build/puct-cpu-acceptance.json','build/puct-cpu-build.json','build/congklak_puct_cpu.dll',*MODELS]
    report=dict(schema='quality-search-acceptance-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=unittest.defaultTestLoader.getTestCaseNames(Tests),evidence=EVIDENCE,inputs={name:sha(ROOT/name) for name in paths},torch_imported='torch' in sys.modules,production_timing=False)
    target=ROOT/'build/quality-search-acceptance.json';target.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],tests=report['tests_run'],failures=report['failures'],errors=report['errors'])))
    raise SystemExit(0 if result.wasSuccessful() else 1)

