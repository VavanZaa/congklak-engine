"""Independent Phase6 literal semantics and actual process-kill acceptance.

No prior accepted source is modified. CUDA workers run in separate processes;
the coordinator has no CUDA context. Production timings own an exclusive slot.
"""
from __future__ import annotations
from collections import Counter
from contextlib import closing
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import reference as literal
from congklak import config,pilot,phase6_common as common,phase6_selfplay as sp
from congklak.native import Native,flatten
from congklak.puct import search,forest,CancellationToken,PROTOCOL
from congklak.phase6_inference import BatchEvaluator,InferenceStopped

MODEL=ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json'
WORK=ROOT/'build/phase6-audit';WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def normalize_literal(p):
    q=literal.Position(7,tuple(flatten(p)),p['controller']);n=literal.normalize(q)
    return literal.Position(7,q.counts,n.controller)
def json_position(p):return dict(pits=[list(p.counts[:7]),list(p.counts[8:15])],stores=[p.counts[7],p.counts[15]],controller=p.controller)
def prediction(p):
    r=sum(map(sum,p['pits']));d=p['stores'][0]-p['stores'][1]
    return dict(value_player0=d+r*.375*(1 if p['controller']==0 else -1),policy_logits=[.13*(a-3) for a in range(7)] if r else [0.]*7,feature_simulations=0,feature_deposits=0)
class Evaluator:
    max_batch=8
    def evaluate_many(self,positions):return [prediction(p) for p in positions]

def literal_tree(p,simulations,noise=None,seed=0):
    """Recursive dictionary search; transitions use only the literal interpreter."""
    rng=random.Random(seed)
    def make(q):
        n=literal.normalize(q);q=literal.Position(7,q.counts,n.controller)
        return {'q':q,'n':[0]*7,'s':[0.]*7,'child':{},'terminal':n.terminal,'prior':None,'bootstrap':literal.evaluate(q)/98}
    def expand(t):
        p=json_position(t['q']);v=prediction(p);legal=literal.legal_actions(t['q']);m=max(v['policy_logits'][a] for a in legal)
        z=sum(math.exp(v['policy_logits'][a]-m) for a in legal)
        t['prior']=[math.exp(v['policy_logits'][a]-m)/z if a in legal else 0. for a in range(7)]
        t['bootstrap']=v['value_player0']/98
    root=make(normalize_literal(p))
    if root['terminal']:return root,rng.getstate()
    expand(root)
    if noise:
        legal=literal.legal_actions(root['q']);draws=[rng.gammavariate(noise['alpha'],1) for a in legal];z=sum(draws)
        for a,v in zip(legal,draws):root['prior'][a]=(1-noise['fraction'])*root['prior'][a]+noise['fraction']*v/z
    def visit(t):
        if t['terminal']:return literal.evaluate(t['q'])/98
        if t['prior'] is None:expand(t);return t['bootstrap']
        scores={}
        for a in literal.legal_actions(t['q']):
            mean=t['s'][a]/t['n'][a] if t['n'][a] else t['bootstrap']
            scores[a]=(1 if t['q'].controller==0 else -1)*mean+1.5*t['prior'][a]*math.sqrt(max(1,sum(t['n'])))/(1+t['n'][a])
        a=min(scores,key=lambda a:(-scores[a],a))
        if a not in t['child']:t['child'][a]=make(literal.apply(t['q'],a).position)
        value=visit(t['child'][a]);t['n'][a]+=1;t['s'][a]+=value;return value
    for _ in range(simulations):visit(root)
    return root,rng.getstate()

def fixture_plan():
    p=copy.deepcopy(common.read_plan());p['selfplay_starts']=p['selfplay_starts'][:2]
    p['selfplay']={**p['selfplay'],'games':4,'simulations':8,'parallel_roots':4}
    return p

def decisions(path):
    with closing(sqlite3.connect(Path(path)/'work.sqlite3')) as db:
        return list(db.execute('SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice'))
def snapshot(path):return {p.name:(sha(p),p.stat().st_mtime_ns) for p in Path(path).iterdir() if p.is_file() and not p.name.endswith(('-wal','-shm'))}
def semantic(rows):
    result=[]
    for gid,choice,body,digest in rows:
        row=json.loads(body);s=row['search'];result.append((gid,choice,row['state'],row['action'],row['after'],row['rng_before'],row['rng_after'],row['policy_target'],s['visits'],s['root_priors'],s['estimated_value_player0'],s['action_mean_value_player0']))
    return result

def kill_worker(code,args,path,event):
    path.mkdir(parents=True,exist_ok=True);marker=path/'marker.json';log=path/'worker.log'
    wrapper="import sys,json,time;from pathlib import Path\ndef callback(v):\n if v['event']==sys.argv[2] and v.get('committed_decisions',1)>=1:\n  Path(sys.argv[1]).write_text(json.dumps(v));\n  while True:time.sleep(.1)\n"+code
    with log.open('w',encoding='utf-8') as output:
        child=subprocess.Popen([sys.executable,'-c',wrapper,str(marker),event,*map(str,args)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while not marker.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.03)
            if not marker.exists():raise AssertionError('Kill barrier unavailable: '+str(log)+'\n'+log.read_text(encoding='utf-8'))
            child.kill();child.wait(timeout=10)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=10)
    return load(marker)

class Phase6IndependentAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=WORK);cls.base=Path(cls.temp.name)
        assert cls.base.resolve().is_relative_to(WORK.resolve())
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self,name):p=self.base/name;p.mkdir();return p
    def test_01_preserved_prior_and_predeclared_cohort_ownership(self):
        accepted=load(ROOT/'build/stage5-deeper-acceptance.json')
        self.assertTrue(accepted['passed']);self.assertEqual(len(accepted['inputs']),878)
        for path,digest in accepted['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        p=common.read_plan();r=common.read_registry();self.assertEqual(p['registry_sha256'],r['sha256'])
        self.assertEqual((len(p['cases']),len(p['arena_starts']),len(p['selfplay_starts'])),(80,24,32))
        self.assertEqual(p['memory_mb'],8192);self.assertEqual(p['budgets_ms'],[25,100]);self.assertEqual(p['arena_ms'],25)
        self.assertEqual((p['bootstrap_samples'],p['bootstrap_seed']),(4096,20261017))
        self.assertEqual(len(p['arena_comparisons'])*24*2,192)
        self.assertEqual(Counter(x['stratum'].split('/')[-1] for x in p['arena_starts']),{'ordinary':8,'capture':8,'bonus':8})
        protected=set(r['quotients']);families=set(r['families'])
        for x in p['selfplay_starts']:
            self.assertNotIn(x['family'],families);self.assertNotIn(pilot.quotient(x['state']).hex(),protected)
            self.assertGreater(sum(map(sum,x['state']['pits'])),4)
        self.assertEqual(len({x['family'] for x in p['selfplay_starts']}),32)
        self.assertEqual(len({x['family'] for x in p['arena_starts']}),24)
        self.assertFalse(p['sealed_final_test_consumed']);self.assertFalse(r['sealed_target_bodies_read'])
        self.assertEqual(r['historical_games'],720)
        self.assertEqual(set(common.HISTORY),{x['path'] for x in r['sources'] if 'games' in x})
        EVIDENCE['prior_and_plan']=dict(prior_artifacts=878,plan_sha256=config.digest(p),registry_sha256=r['sha256'],protected_quotients=len(protected),protected_families=len(families),cases=80,arena_games=192)
    def test_02_literal_fixed_player_backups_bonus_pass_and_fpu(self):
        roots=[config.position(),dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0),dict(pits=[[2,1,0,0,0,0,3],[1,0,2,0,1,0,2]],stores=[43,43],controller=0)]
        bonus=False;passed=False
        for p in roots:
            passed|=bool(literal.normalize(literal.Position(7,tuple(flatten(p)),p['controller'])).passes)
            for a in literal.legal_actions(normalize_literal(p)):bonus|=literal.apply(normalize_literal(p),a).bonus
            for sims in (2,64):
                expect,_=literal_tree(p,sims);actual=search(p,Evaluator(),simulations=sims)
                self.assertEqual(actual['visits'],expect['n']);self.assertEqual(actual['root_priors'],expect['prior'])
                for a,n in enumerate(expect['n']):self.assertEqual(actual['action_mean_value_player0'][a],expect['s'][a]/n*98 if n else None)
                self.assertEqual(actual['protocol'],PROTOCOL);self.assertEqual(actual['first_play_urgency'],'parent_neural_bootstrap')
        self.assertTrue(bonus);self.assertTrue(passed)
        p=roots[-1];q=copy.deepcopy(p);q['stores']=[63,23]
        for sims in (2,64):
            a=search(p,Evaluator(),simulations=sims);b=search(q,Evaluator(),simulations=sims)
            self.assertEqual(a['visits'],b['visits']);self.assertAlmostEqual(b['estimated_value_player0']-a['estimated_value_player0'],40)
        EVIDENCE['literal_puct']=dict(roots=3,budgets=[2,64],bonus=True,normalized_pass=True,store_offset_invariant=True)
    def test_03_independent_forest_noise_rng_bounds_and_incomplete_batch(self):
        p=dict(pits=[[2,1,0,0,0,0,3],[1,0,2,0,1,0,2]],stores=[43,43],controller=0);roots=[p,{**p,'controller':1}]
        noise={'alpha':.3,'fraction':.25};rngs=[random.Random(31),random.Random(32)]
        reports=forest(roots,Evaluator(),simulations=64,rng=rngs,root_noise=noise)
        for i,(root,report) in enumerate(zip(roots,reports)):
            expect,state=literal_tree(root,64,noise,31+i)
            self.assertEqual(report['visits'],expect['n']);self.assertEqual(report['root_priors'],expect['prior']);self.assertEqual(rngs[i].getstate(),state)
        terminal=dict(pits=[[0]*7,[0]*7],stores=[70,28],controller=1)
        class Bomb(Evaluator):
            def evaluate_many(self,p):raise AssertionError('Terminal inference')
        t=search(terminal,Bomb());self.assertEqual(t['estimated_value_player0'],42);self.assertEqual(t['completed_simulations'],0);self.assertEqual(t['proof_status'],'terminal_outcome')
        t=search(p,Evaluator(),max_nodes=1);self.assertFalse(t['usable_decision']);self.assertEqual(t['stop_reason'],'node_budget')
        class Slow(Evaluator):
            def evaluate_many(self,p):time.sleep(.025);return super().evaluate_many(p)
        t=search(p,Slow(),time_ms=5);self.assertEqual(t['completed_simulations'],0);self.assertGreater(t['timing']['overshoot_ms'],10)
        token=CancellationToken();token.cancel();t=search(p,Evaluator(),cancel=token);self.assertEqual(t['completed_simulations'],0)
        class Corrupt(Evaluator):
            def evaluate_many(self,p):v=super().evaluate_many(p);v[-1]['value_player0']=float('nan');return v
        with self.assertRaises(Exception):forest(roots,Corrupt())
        from unittest.mock import patch
        from congklak import phase6_store
        original_connect=sqlite3.connect;path=self.fresh('atomic-initial-store')
        class BrokenConnection:
            def __init__(self,*a,**k):self.db=original_connect(*a,**k)
            def __getattr__(self,k):return getattr(self.db,k)
            def __setattr__(self,k,v):
                if k=='db':object.__setattr__(self,k,v)
                else:setattr(self.db,k,v)
            def __enter__(self):self.db.__enter__();return self
            def __exit__(self,*a):return self.db.__exit__(*a)
            def execute(self,sql,*a):
                result=self.db.execute(sql,*a)
                if sql=='PRAGMA user_version=6':raise RuntimeError('Injected initial identity interruption')
                return result
        with patch('congklak.phase6_store.sqlite3.connect',BrokenConnection):
            with self.assertRaises(RuntimeError):phase6_store.Work(path,{'test':'atomic'})
        with closing(original_connect(path/'work.sqlite3')) as db:
            self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0],0)
            self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone())
        work=phase6_store.Work(path,{'test':'atomic'});self.assertEqual(work.get('identity'),{'test':'atomic'});work.close()
        EVIDENCE['forest_contract']=dict(per_root_rng_literal=True,terminal_neural_calls=0,prior_only_unusable=True,cancelled_backups=0,nonpreemptible_overshoot_reported=True,corrupt_batch_refused=True,initial_schema_identity_atomic=True)
    def fixture(self,name):
        root=self.fresh(name);plan=fixture_plan();(root/'plan.json').write_text(config.canonical(plan));return root,plan
    def sp_kill(self,event,name):
        root,plan=self.fixture(name);run=root/'run'
        code="from congklak import phase6_selfplay as sp\nsp.run(sys.argv[3],sys.argv[4],'native',plan=json.loads(Path(sys.argv[5]).read_text()),on_event=callback)"
        marker=kill_worker(code,[run,MODEL,root/'plan.json'],root/'probe',event)
        prefix=decisions(run);self.assertTrue(prefix)
        result=sp.run(run,MODEL,'native',plan=plan);after=decisions(run)
        indexed={(g,c):(b,h) for g,c,b,h in after}
        for g,c,b,h in prefix:self.assertEqual(indexed[g,c],(b,h))
        fresh=root/'fresh';sp.run(fresh,MODEL,'native',plan=plan)
        self.assertEqual(semantic(after),semantic(decisions(fresh)))
        state=snapshot(run);self.assertEqual(sp.run(run,MODEL,'native',plan=plan),result);self.assertEqual(snapshot(run),state)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            for game in sp.read_games(db):
                q=normalize_literal(game['start'])
                for body, in db.execute('SELECT body FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):q=literal.apply(q,json.loads(body)['action']).position
                self.assertEqual(json_position(q),game['position'])
        return root,plan,result,marker,len(prefix),len(after)
    def test_04_actual_selfplay_precommit_kill_retains_and_replays(self):
        root,p,result,marker,before,after=self.sp_kill('phase6_selfplay_precommit','sp-precommit')
        EVIDENCE['selfplay_precommit_kill']=dict(actual_hard_kill=True,committed_before=before,decisions_after=after,fresh_semantic_equal=True,completed_bytes_mtime_stable=True,manifest_sha256=config.digest(result))
    def test_05_actual_selfplay_postcommit_and_replay_publication_kills(self):
        evidence=[]
        for i,event in enumerate(('phase6_selfplay_committed','phase6_replay_precommit','phase6_replay_committed')):
            root,p,result,marker,before,after=self.sp_kill(event,'sp-publish-'+str(i));evidence.append(dict(event=event,actual_hard_kill=True,committed_before=before,decisions_after=after,manifest_sha256=config.digest(result)))
        EVIDENCE['selfplay_publication_kills']=evidence
    def test_06_selfplay_checksum_valid_corruption_and_family_pruning(self):
        root,plan=self.fixture('sp-corruption');run=root/'run';result=sp.run(run,MODEL,'native',plan=plan)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            original=db.execute('SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice LIMIT 1').fetchone();gid,choice,body,digest=original
            for name,mutate in [('rng',lambda r:r['rng_after'].__setitem__(2,.25)),('protocol',lambda r:r['search'].__setitem__('protocol','forged')),('mean',lambda r:r['search'].__setitem__('estimated_value_player0',1000)),('work',lambda r:r['search']['work'].__setitem__('total_deposits',-1))]:
                row=json.loads(body);mutate(row)
                with db:db.execute('UPDATE sp_decisions SET body=?,hash=? WHERE gameid=? AND choice=?',(config.canonical(row),config.digest(row),gid,choice))
                with self.assertRaises(Exception,msg=name):sp.run(run,MODEL,'native',plan=plan)
                with db:db.execute('UPDATE sp_decisions SET body=?,hash=? WHERE gameid=? AND choice=?',(body,digest,gid,choice))
            row=json.loads(body);row['gameid']='unknown'
            with db:db.execute('INSERT INTO sp_decisions VALUES(?,?,?,?)',('unknown',0,config.canonical(row),config.digest(row)))
            with self.assertRaises(Exception):sp.run(run,MODEL,'native',plan=plan)
            with db:db.execute("DELETE FROM sp_decisions WHERE gameid='unknown'")
            saved=db.execute("SELECT body,hash FROM meta WHERE key='selfplay_manifest'").fetchone()
            with db:db.execute("DELETE FROM meta WHERE key='selfplay_manifest'")
            with self.assertRaises(Exception):sp.load_training_rows(run)
            with db:db.execute("INSERT INTO meta VALUES('selfplay_manifest',?,?)",saved)
            identity=result['identity'];registry=copy.deepcopy(common.read_registry());registry['quotients']=sorted(set(registry['quotients'])|{pilot.quotient(plan['selfplay_starts'][0]['state']).hex()})
            blocked,rows=sp.build_rows(db,identity,registry,Native())
            self.assertEqual(blocked['summary']['family_exclusions'][plan['selfplay_starts'][0]['family']],'protected_active_quotient_bridge')
            self.assertFalse(any(r['family']==plan['selfplay_starts'][0]['family'] for r in rows))
        EVIDENCE['selfplay_admission']=dict(checksum_valid_injections_refused=['rng','protocol','mean','work','orphan_decision','missing_completion'],whole_family_bridge_excluded=True,individual_returns_not_averaged=result['summary']['targets_averaged'] is False)
    def test_07_actual_analysis_pre_postcommit_recovery_and_terminal(self):
        from congklak.phase6_analysis import analyze
        root=self.fresh('analysis');run=root/'run';p=config.position();first=analyze(run,p,MODEL,'native',simulations=16)
        code="from congklak.phase6_analysis import analyze\nanalyze(sys.argv[3],json.loads(sys.argv[4]),sys.argv[5],'native',simulations=int(sys.argv[6]),on_event=callback)"
        kill_worker(code,[run,config.canonical(p),MODEL,32],root/'pre','phase6_analysis_precommit')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:self.assertEqual(json.loads(db.execute("SELECT body FROM meta WHERE key='analysis_result'").fetchone()[0]),first)
        second=analyze(run,p,MODEL,'native',simulations=32);self.assertEqual(second['search']['completed_simulations'],32)
        kill_worker(code,[run,config.canonical(p),MODEL,64],root/'post','phase6_analysis_committed')
        self.assertFalse((run/'analysis-64.json').exists());third=analyze(run,p,MODEL,'native',simulations=64)
        self.assertEqual(load(run/'analysis-64.json'),third);saved=snapshot(run);self.assertEqual(analyze(run,p,MODEL,'native',simulations=64),third);self.assertEqual(snapshot(run),saved)
        terminal=dict(pits=[[0]*7,[0]*7],stores=[72,26],controller=0);done=analyze(root/'terminal',terminal,MODEL,'native',simulations=64)
        self.assertEqual(done['status'],'complete');self.assertEqual(done['search']['estimated_value_player0'],46);self.assertEqual(done['proof_status'],'terminal_outcome')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            forged=copy.deepcopy(third);forged['search']['protocol']='forged'
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='analysis_result'",(config.canonical(forged),config.digest(forged)))
        with self.assertRaises(Exception):analyze(run,p,MODEL,'native',simulations=64)
        EVIDENCE['analysis_kills']=dict(actual_precommit_kill=True,actual_postcommit_kill=True,missing_publication_finalized=True,retained_prior=True,terminal_exact=True,checksum_valid_report_refused=True)
    def test_08_mixed_targets_losses_and_frozen_warm_ownership(self):
        # CPU-only tensor/loss check. CUDA recovery uses separate worker contexts.
        import torch
        from congklak import phase6_training as tr
        torch.set_num_threads(1)
        model=tr.Evaluator('warm');self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),455)
        logits=torch.tensor([[1e10,-1e10,-1e10-1,1e10,1e10,1e10,1e10]],requires_grad=True);mask=torch.tensor([[False,True,True,False,False,False,False]])
        target=torch.tensor([[0.,.75,.25,0.,0.,0.,0.]])
        loss=tr.policy_losses(logits,target,mask).mean();self.assertTrue(torch.isfinite(loss));loss.backward();self.assertEqual(logits.grad[0,0].item(),0)
        plan=common.read_plan();self.assertEqual(plan['update']['loss'],'.5*Huber_teacher_seeds+.5*Huber_selfplay_seeds+.25*(.5*CE_teacher+.5*CE_visits)')
        source=(ROOT/'congklak/phase6_training.py').read_text();self.assertIn('loss=.5*nn.functional.huber_loss(prediction',source);self.assertIn('selfplay_draws+=count',source)
        EVIDENCE['mixed_loss']=dict(warm_trainable_parameters=455,illegal_logit_gradient_zero=True,loss_coefficients=[.5,.5,.125,.125],target_units='fixed_player0_seeds',sources_distinct=True)
    def test_09_cuda_warm_actual_kill_full_state_and_finished_stability(self):
        self.cuda_recovery('warm')
    def test_10_cuda_update_actual_kill_and_final_preexport_recovery(self):
        self.cuda_recovery('selfplay')
    def cuda_recovery(self,stage):
        root=self.fresh('cuda-'+stage)
        args=dict(stage=stage,steps=6,batch_size=16,checkpoint_every=2,seed=187 if stage=='warm' else 197,device='cuda')
        if stage=='selfplay':
            # The generator manifest must describe exactly this completed warm
            # model. Production replay is preferred; fixtures remain scoped.
            source=Path(os.environ.get('PHASE6_AUDIT_SP_PATH',str(common.SELFPLAY)))
            warm=Path(os.environ.get('PHASE6_AUDIT_WARM_MODEL',str(common.WARM)))
            manifest,rows=sp.load_training_rows(source);self.assertTrue(rows)
            args.update(initial_model=str(warm),selfplay_path=str(source))
        code="from congklak import phase6_training as tr;from pathlib import Path;import json,sys;tr.train(Path(sys.argv[1]),**json.loads(sys.argv[2]))"
        def invoke(path,refuse=False):
            result=subprocess.run([sys.executable,'-c',code,str(path),json.dumps(args)],cwd=ROOT,capture_output=True,text=True)
            (root/('last-refusal.log' if refuse else 'last-worker.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            self.assertEqual(result.returncode==0,not refuse,result.stdout+result.stderr)
            return load(path/'result.json') if not refuse else None
        fresh=root/'fresh';reference=invoke(fresh);records=[]
        for event in ('checkpoint_file_ready','training_final_checkpoint_ready'):
            run=root/event
            # Block second checkpoint before catalog commit, or final committed
            # checkpoint before export. A prior complete unit always exists.
            custom="import sys,json,time;from pathlib import Path;from congklak import phase6_training as tr\ndef cb(v):\n if v['event']==sys.argv[2] and (sys.argv[2]!='checkpoint_file_ready' or v['step']==4):\n  Path(sys.argv[1]).write_text(json.dumps(v));\n  while True:time.sleep(.1)\ntr.train(Path(sys.argv[3]),**json.loads(sys.argv[4]),on_event=cb)"
            probe=root/(event+'-probe');probe.mkdir();marker=probe/'marker.json';log=probe/'worker.log'
            with log.open('w',encoding='utf-8') as out:
                child=subprocess.Popen([sys.executable,'-c',custom,str(marker),event,str(run),json.dumps(args)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+180
                    while not marker.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                    self.assertTrue(marker.exists(),log.read_text(encoding='utf-8'));child.kill();child.wait(timeout=10)
                finally:
                    if child.poll() is None:child.kill();child.wait(timeout=10)
            with closing(sqlite3.connect(run/'catalog.sqlite3')) as db:committed=db.execute('SELECT MAX(step) FROM checkpoints').fetchone()[0]
            self.assertEqual(committed,2 if event=='checkpoint_file_ready' else 6)
            self.assertFalse((run/'model.json').exists());restored=invoke(run)
            self.assertEqual(restored['recovery']['restored_step'],committed);self.assertEqual(restored['weights_sha256'],reference['weights_sha256'])
            self.assertEqual((fresh/'model.json').read_bytes(),(run/'model.json').read_bytes())
            compare="""import sys,sqlite3,torch
from pathlib import Path
def last(p):
 db=sqlite3.connect(p/'catalog.sqlite3');name=db.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0];db.close();return torch.load(p/name,map_location='cpu',weights_only=True)
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
a=last(Path(sys.argv[1]));b=last(Path(sys.argv[2]));assert eq(a,b),'Optimizer/sampler/Python/CPU/allCUDA/model/best state differs';assert len(b['optimizer']['state'])==int(sys.argv[3]);assert b['teacher_draws']==(96 if sys.argv[3]=='2' else 48);assert b['selfplay_draws']==(0 if sys.argv[3]=='2' else 48)
"""
            checked=subprocess.run([sys.executable,'-c',compare,str(fresh),str(run),'2' if stage=='warm' else '8'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            stable=snapshot(run);self.assertEqual(invoke(run),restored);self.assertEqual(snapshot(run),stable)
            original=(run/'model.json').read_bytes();forged=json.loads(original);forged['model']['uses_oracle']=True;forged['sha256']=config.digest(forged['model']);(run/'model.json').write_text(config.canonical(forged))
            bad=snapshot(run);invoke(run,True);self.assertEqual(snapshot(run),bad);(run/'model.json').write_bytes(original)
            records.append(dict(event=event,actual_cuda_hard_kill=True,restored_step=committed,whole_state_bitwise_equal=True,finished_bytes_mtime_stable=True,checksum_valid_export_refused=True))
        EVIDENCE['cuda_'+stage]=dict(checks=records,steps=6,batch_size=16,teacher_draws=96 if stage=='warm' else 48,selfplay_draws=0 if stage=='warm' else 48)
    def test_11_actual_benchmark_case_kill_and_forged_reference_refusal(self):
        root=self.fresh('benchmark-case');run=root/'run';output=root/'result.json'
        code="import importlib.util\nspec=importlib.util.spec_from_file_location('b','tools/benchmark-phase6.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\nb.run(sys.argv[3],sys.argv[4],case_limit=2,no_arena=True,on_event=callback)"
        marker=kill_worker(code,[run,output],root/'probe','phase6_case_committed')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:before=db.execute('SELECT idx,body,hash FROM cases').fetchall()
        self.assertEqual(len(before),1)
        args=[sys.executable,str(ROOT/'tools/benchmark-phase6.py'),'run','--run',str(run),'--output',str(output),'--case-limit','2','--no-arena']
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall();self.assertEqual(after[0],before[0]);self.assertEqual(len(after),2)
            idx,body,digest=after[0];forged=json.loads(body);ms=next(iter(forged['measurements']));name=next(iter(forged['measurements'][ms]));forged['measurements'][ms][name]['reference_regret']=999
            with db:db.execute('UPDATE cases SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),idx))
        bad=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0);self.assertIn('regret',bad.stdout.lower())
        EVIDENCE['benchmark_case_kill']=dict(actual_hard_kill=True,committed_cases_retained=1,complete_after_resume=2,forged_reference_refused=True,barrier=marker)
    def test_12_actual_completed_arena_kill_and_literal_replay(self):
        root=self.fresh('benchmark-game');run=root/'run';run.mkdir();output=root/'result.json'
        with closing(sqlite3.connect(ROOT/'data/phase6-benchmark/work.sqlite3')) as source,closing(sqlite3.connect(run/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM games').fetchone()[0],192)
            keys=db.execute('SELECT comparison,pair,seat FROM games ORDER BY comparison,pair,seat DESC LIMIT 2').fetchall()
            with db:
                for key in keys:db.execute('DELETE FROM games WHERE comparison=? AND pair=? AND seat=?',key)
                db.execute("DELETE FROM meta WHERE key='phase6_benchmark_complete'")
        code="import importlib.util\nspec=importlib.util.spec_from_file_location('b','tools/benchmark-phase6.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\nb.run(sys.argv[3],sys.argv[4],on_event=callback)"
        marker=kill_worker(code,[run,output],root/'probe','phase6_arena_game_committed')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:prefix=db.execute('SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat').fetchall()
        self.assertEqual(len(prefix),191)
        args=[sys.executable,str(ROOT/'tools/benchmark-phase6.py'),'run','--run',str(run),'--output',str(output)]
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat').fetchall();self.assertEqual(len(after),192)
            index={(c,p,s):(b,h) for c,p,s,b,h in after}
            for c,p,s,b,h in prefix:self.assertEqual(index[c,p,s],(b,h))
            for c,p,s,b,h in after:
                g=json.loads(b);q=normalize_literal(g['start'])
                for action in g['moves']:q=literal.apply(q,action).position
                self.assertEqual(json_position(q),g['final'])
        state=snapshot(run);saved=load(output);again=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(again.returncode,0,again.stdout+again.stderr);self.assertEqual(snapshot(run),state);self.assertEqual(load(output),saved)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            c,p,s,b,h=after[0];forged=json.loads(b);forged['candidate_final_margin']+=2
            with db:db.execute('UPDATE games SET body=?,hash=? WHERE comparison=? AND pair=? AND seat=?',(config.canonical(forged),config.digest(forged),c,p,s))
        bad=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0)
        EVIDENCE['benchmark_game_kill']=dict(actual_hard_kill=True,committed_games_retained=191,complete_after_resume=192,all_games_literal_replayed=True,finished_bytes_mtime_stable=True,forged_outcome_refused=True,barrier=marker)
    def test_13_declared_whole_clock_statistical_and_proof_contract(self):
        spec=importlib.util.spec_from_file_location('phase6_bench_audit',ROOT/'tools/benchmark-phase6.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        p=common.read_plan();self.assertEqual(m.expected_order(1,0,p),p['engines'][1:]+p['engines'][:1]);self.assertEqual(m.expected_order(1,1,p),list(reversed(m.expected_order(1,0,p))))
        rng=random.Random(p['bootstrap_seed']);values=[-2.,0.,5.,8.];samples=sorted(sum(rng.choices(values,k=4))/4 for _ in range(4096));self.assertEqual(m.interval(values,p),[samples[102],samples[3993]])
        self.assertEqual(p['puct']['first_play_urgency'],'parent fixed-player0 neural bootstrap');self.assertIn('whole',p['recommendation'])
        EVIDENCE['statistical_contract']=dict(primary='paired_family_terminal_seed_margin',bootstrap_samples=4096,bootstrap_seed=20261017,whole_wrapper_clock=True,model_startup_separate=True,zero_fallback_and_full_pairs_required=True,sealed_consumed=False)

if __name__=='__main__':unittest.main(verbosity=2)
