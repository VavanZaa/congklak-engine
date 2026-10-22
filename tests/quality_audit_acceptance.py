"""Independent quality experiment checks and retained live recovery fixtures.

The literal search below uses only tests/reference.py for Congklak transitions;
it does not call either production search's selection or backup helpers.
"""
from __future__ import annotations
from collections import Counter,defaultdict
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
import threading
import time
import unittest
from unittest import mock
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from congklak.puct import CancellationToken
from congklak.phase6_inference import InferenceStopped
import reference as literal

WORK=ROOT/'build/quality-audit';WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def normalize_literal(p):
    q=literal.Position(7,tuple(flatten(p)),p['controller']);n=literal.normalize(q)
    return literal.Position(7,q.counts,n.controller)
def json_position(q):return dict(pits=[list(q.counts[:7]),list(q.counts[8:15])],stores=[q.counts[7],q.counts[15]],controller=q.controller)
def snapshot(path):return {p.relative_to(path).as_posix():(sha(p),p.stat().st_mtime_ns) for p in Path(path).rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))}
def semantic(report):
    return {key:report[key] for key in ('rules_digest','position','chosen','visits','policy','root_priors','estimated_value_player0','action_mean_value_player0','root_expanded','completed_simulations','requested_simulations','nodes','max_nodes','usable_decision','stop_reason','proof_status','unvisited_root_q_player0')}

def prediction(p):
    active=sum(map(sum,p['pits']));d=p['stores'][0]-p['stores'][1]
    return dict(value_player0=d+active*.375*(1 if p['controller']==0 else -1),policy_logits=[.13*(a-3) for a in range(7)] if active else [0.]*7,feature_simulations=0,feature_deposits=0)
class SimpleEvaluator:
    max_batch=8
    def evaluate_many(self,positions):return [prediction(p) for p in positions]

def literal_tree(p,simulations,c,noise=None,seed=0,max_nodes=2048):
    """Recursive dictionary implementation, independent of native rules/search."""
    rng=random.Random(seed);allocated=1;completed=0;stopped=''
    def make(q):
        n=literal.normalize(q);q=literal.Position(7,q.counts,n.controller)
        return dict(q=q,visits=[0]*7,sums=[0.]*7,children={},terminal=n.terminal,priors=None,bootstrap=literal.evaluate(q)/98)
    def expand(t):
        row=prediction(json_position(t['q']));legal=literal.legal_actions(t['q']);top=max(row['policy_logits'][a] for a in legal)
        weights={a:math.exp(row['policy_logits'][a]-top) for a in legal};denom=sum(weights.values())
        t['priors']=[weights.get(a,0.)/denom for a in range(7)];t['bootstrap']=row['value_player0']/98
    root=make(normalize_literal(p))
    if root['terminal']:return root,rng.getstate(),allocated,completed,'terminal'
    expand(root)
    if noise:
        legal=literal.legal_actions(root['q']);draws=[rng.gammavariate(noise['alpha'],1) for _ in legal];total=sum(draws)
        for a,v in zip(legal,draws):root['priors'][a]=(1-noise['fraction'])*root['priors'][a]+noise['fraction']*v/total
    def visit(t):
        nonlocal allocated,stopped
        if t['terminal']:return literal.evaluate(t['q'])/98
        if t['priors'] is None:expand(t);return t['bootstrap']
        scores={}
        for a in literal.legal_actions(t['q']):
            q=t['sums'][a]/t['visits'][a] if t['visits'][a] else t['bootstrap']
            scores[a]=(1 if t['q'].controller==0 else -1)*q+c*t['priors'][a]*math.sqrt(max(1,sum(t['visits'])))/(1+t['visits'][a])
        a=min(scores,key=lambda a:(-scores[a],a))
        if a not in t['children']:
            if allocated>=max_nodes:stopped='node_budget';return None
            t['children'][a]=make(literal.apply(t['q'],a).position);allocated+=1
        value=visit(t['children'][a])
        if value is None:return None
        t['visits'][a]+=1;t['sums'][a]+=value;return value
    for _ in range(simulations):
        if visit(root) is None:break
        completed+=1
    return root,rng.getstate(),allocated,completed,stopped or 'simulations'

def kill_barrier(command,path,event_name,env=None,condition=lambda _:True,timeout=180):
    """Kill a real child only after its declared durable/precommit event."""
    path.mkdir();log=path/'worker.log';marker=path/'marker.json'
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',env={**os.environ,**(env or {})})
        lines=[];done=threading.Event()
        def reader():
            for line in child.stdout:
                stream.write(line);stream.flush();lines.append(line)
                try:body=json.loads(line)
                except ValueError:continue
                if body.get('event')==event_name and condition(body):save(marker,body);done.set();return
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        try:
            deadline=time.monotonic()+timeout
            while not done.wait(.05) and child.poll() is None and time.monotonic()<deadline:pass
            if not marker.exists():raise AssertionError('Unavailable kill barrier '+str(log)+'\n'+''.join(lines)[-6000:])
            child.kill();child.wait(10);thread.join(5)
        finally:
            if child.poll() is None:child.kill();child.wait(10)
    return load(marker)
def run_child(command,log,timeout=180,env=None,success=True):
    with Path(log).open('w',encoding='utf-8') as stream:result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,text=True,env={**os.environ,**(env or {})},timeout=timeout)
    if success and result.returncode:raise AssertionError('Worker failed '+str(log)+'\n'+Path(log).read_text(encoding='utf-8')[-6000:])
    return result.returncode

class QualityIndependentAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p

    def test_01_prior_1368_inputs_unchanged(self):
        accepted=load(ROOT/'build/puct-cpu-acceptance.json');self.assertTrue(accepted['passed']);self.assertEqual(accepted['tests_run'],34);self.assertEqual(len(accepted['inputs']),1368)
        for name,digest in accepted['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['preserved_prior_inputs']=dict(cpu=1368,all_hashes_current=True)

    def test_02_predeclared_two_by_two_experiment(self):
        from congklak import quality_common as common
        plan=common.read_plan();self.assertEqual(plan,common.expected_plan())
        self.assertFalse(plan['sealed_final_test_consumed']);self.assertFalse(plan['default_promotion'])
        self.assertEqual(set(plan['profiles']),{'warm_control','warm_search','refined_control','refined_search'})
        self.assertEqual(len(plan['cases']),80);self.assertEqual(len(plan['arena_starts']),24)
        self.assertEqual(plan['budgets_ms'],[25,100]);self.assertEqual(plan['arena_ms'],25)
        expected_profiles={name:dict(model='warm' if name.startswith('warm_') else 'refined',c_puct=1.5 if name.endswith('control') else .25,cache_capacity=0) for name in ('warm_control','warm_search','refined_control','refined_search')}
        self.assertEqual(plan['profiles'],expected_profiles);self.assertEqual(plan['engines'],['H_v1','H8_relay_CPU',*expected_profiles])
        _,data,_=common.teacher.load();registry=common.read_registry();dev=data['development'];excluded=set(registry['arena_families'])
        def choose(rows,quota,namespace,used):
            ordered=sorted(rows,key=lambda r:hashlib.sha256((namespace+'/'+r['state_key']).encode()).digest());result=[]
            for kind,count in quota:
                chosen=[]
                for row in ordered:
                    if row['family'] not in used and row['stratum'].endswith('/'+kind):chosen.append(row);used.add(row['family'])
                    if len(chosen)==count:break
                self.assertEqual(len(chosen),count);result+=chosen
            return result
        proxy=choose(dev,[('ordinary',16),('capture',16),('bonus',16)],'quality-H8-cases-v1',excluded)
        contested=[r for r in dev if sum(map(sum,r['state']['pits']))>=25 and abs(r['state']['stores'][0]-r['state']['stores'][1])<sum(map(sum,r['state']['pits']))]
        arena=choose(contested,[('ordinary',8),('capture',8),('bonus',8)],'quality-arena-v1',excluded)
        self.assertEqual([r['state_key'] for r in plan['cases'][:48]],[r['state_key'] for r in proxy]);self.assertEqual([r['state_key'] for r in plan['arena_starts']],[r['state_key'] for r in arena])
        self.assertFalse({r['family'] for r in proxy}&{r['family'] for r in arena});self.assertFalse(({r['family'] for r in proxy}|{r['family'] for r in arena})&set(registry['arena_families']))
        for actual,source in zip(plan['cases'][:48],proxy):self.assertEqual(actual['reference'],source['label'])
        self.assertTrue(all(r['reference']['kind']=='exact_optimal_margin' for r in plan['cases'][48:]))
        comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in expected_profiles for b in ('H_v1','H8_relay_CPU')]
        self.assertEqual(plan['arena_comparisons'],comparisons);names=[r['id'] for r in comparisons];order=[]
        for pair in range(24):
            shift=pair%8
            for name in names[shift:]+names[:shift]:
                for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
        self.assertEqual(plan['arena_work_order'],order);self.assertEqual(len(order),384)
        self.assertEqual((plan['bootstrap_seed'],plan['bootstrap_samples']),(20261017,4096))
        self.assertEqual(plan['puct'],dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap'))
        EVIDENCE['predeclared_two_by_two']=dict(profiles=4,cases=80,scalar_measurements=960,arena_families=24,games=384,independently_rederived_family_order=True,new_arena_family_IDs=True,ancestral_development_exposure=True,development_reuse=True,sealed_read=False,automatic_promotion=False)

    def test_03_literal_fixed_player0_search_and_node_limit(self):
        from congklak import quality_search as search
        plan=load(ROOT/'build/puct-cpu-plan.json');roots=[row['state'] for row in plan['cases'][:12]]
        roots += [dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0),dict(pits=[[0]*7,[0]*7],stores=[40,58],controller=1)]
        checked=0
        for c in (1.5,.25):
            for index,p in enumerate(roots):
                for bound in (1,2048):
                    noise={'alpha':.3,'fraction':.25} if index%2 else None;seed=301+index;rng=random.Random(seed)
                    actual=search.search(p,SimpleEvaluator(),simulations=64,max_nodes=bound,rng=rng,root_noise=noise,c_puct=c)
                    expected,state,nodes,completed,stop=literal_tree(p,64,c,noise,seed,bound)
                    self.assertEqual(actual['position'],json_position(expected['q']));self.assertEqual(actual['visits'],expected['visits']);self.assertEqual(actual['completed_simulations'],completed)
                    self.assertEqual(actual['nodes'],nodes);self.assertEqual(actual['stop_reason'],stop);self.assertEqual(rng.getstate(),state)
                    self.assertEqual(actual['root_priors'],expected['priors'] or [0.]*7)
                    for a in range(7):self.assertEqual(actual['action_mean_value_player0'][a],expected['sums'][a]/expected['visits'][a]*98 if expected['visits'][a] else None)
                    checked+=1
        EVIDENCE['literal_fixed_player0_search']=dict(trees=checked,independent_literal_transitions=True,exploration_constants=[1.5,.25],node_limits=[1,2048],noise_and_rng=True)

    def test_04_control_native_fixed_work_equivalence(self):
        from congklak import quality_common as common,quality_search
        from congklak.puct import search as frozen_search
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        plan=common.read_plan();roots=[row['state'] for row in plan['cases'][:12]];models=[ROOT/plan['models'][name] for name in ('warm','refined')];checked=0
        for model in models:
            with CpuBatchEvaluator(model,cache_capacity=0,max_batch=8) as evaluator:
                for index,p in enumerate(roots):
                    original=frozen_search(p,evaluator,simulations=128,max_nodes=2048,rng=724+index)
                    actual=quality_search.search(p,evaluator,simulations=128,max_nodes=2048,rng=724+index,c_puct=1.5)
                    self.assertEqual(semantic(original),semantic(actual));checked+=1
        EVIDENCE['frozen_control_equivalence']=dict(models=2,trees=checked,backups=128,exact_visits_values_actions=True,same_uncached_adapter=True)

    def test_05_cancellation_and_invalid_search_contract(self):
        from congklak import quality_search
        p=load(ROOT/'build/puct-cpu-plan.json')['cases'][0]['state'];token=CancellationToken();token.cancel()
        report=quality_search.search(p,SimpleEvaluator(),simulations=64,cancel=token,c_puct=.25)
        self.assertEqual(report['completed_simulations'],0);self.assertEqual(sum(report['visits']),0);self.assertFalse(report['usable_decision'])
        self.assertEqual(report['stop_reason'],'cancelled')
        for c in (0,-1,float('nan'),float('inf'),True,.3):
            with self.assertRaises((ValueError,TypeError)):quality_search.search(p,SimpleEvaluator(),c_puct=c)
        class Bad(SimpleEvaluator):
            def evaluate_many(self,positions):return [{**prediction(p),'policy_logits':[float('nan')]*7} for p in positions]
        with self.assertRaises(Exception):quality_search.search(p,Bad(),simulations=64,c_puct=.25)
        token=CancellationToken()
        class CancelAfterForward(SimpleEvaluator):
            def evaluate_many(self,positions):
                rows=super().evaluate_many(positions);token.cancel();return rows
        after=quality_search.search(p,CancelAfterForward(),simulations=64,cancel=token,c_puct=.25)
        self.assertEqual(after['completed_simulations'],0);self.assertEqual(after['work']['neural_evaluations'],1);self.assertFalse(after['root_expanded'])
        for changed in ({'requested_simulations':True},{'nodes':2},{'completed_simulations':False},{'root_expanded':True}):
            corrupt=copy.deepcopy(report);corrupt.update(changed)
            with self.assertRaises(Exception):quality_search.validate_report(corrupt,p,64,c_puct=.25)
        EVIDENCE['cancellation_contract']=dict(pre_cancel_zero_backups=True,post_forward_zero_backups_with_charged_work=True,invalid_constants_refused=True,corrupt_predictions_and_saved_reports_refused=True)

    def test_06_training_filter_and_root_target_derivation(self):
        from congklak import quality_learning as learning,deeper_teacher_v2 as teacher,pilot
        admitted,data=learning.prepare();registry=learning.read_registry();source,original,_=teacher.load(teacher_depth=8)
        forbidden_q=set(registry['quotients']);forbidden_f=set(registry['families'])
        bridged={row['family'] for row in original['train'] if pilot.quotient(row['state']).hex() in forbidden_q}
        expected=[row for row in original['train'] if row['family'] not in forbidden_f|bridged and pilot.quotient(row['state']).hex() not in forbidden_q]
        self.assertEqual([r['state_key'] for r in data['train']],[r['state_key'] for r in expected])
        self.assertEqual(data['development'],original['development']);self.assertEqual(len(data['development']),1000)
        self.assertEqual(admitted['train_rows'],len(expected));self.assertEqual(admitted['train_families'],sorted({r['family'] for r in expected}))
        self.assertEqual(admitted['train_strata'],dict(Counter(r['stratum'] for r in expected)))
        self.assertFalse(admitted['sealed_target_bodies_read']);self.assertFalse(admitted['oracle_training']);self.assertEqual(admitted['training_replay_rows'],0)
        picked={}
        for split in ('train','development'):
            for row in data[split]:picked.setdefault((split,row['stratum'],row['state']['controller']),row)
        actor_counts=Counter();checked=0
        for row in picked.values():
            p=row['state'];actual=learning.target(row);sign=1 if p['controller']==0 else -1;d=p['stores'][0]-p['stores'][1];active=sum(map(sum,p['pits']));q={item['pit']:sign*item['value_player0'] for item in row['label']['action_scores']}
            best=max(q.values());weights={a:math.exp((v-best)/2.) for a,v in q.items()};denom=sum(weights.values());expected_p=[weights.get(a,0.)/denom for a in range(7)]
            self.assertEqual(actual['probability'],expected_p);self.assertEqual(actual['legal'],[a in q for a in range(7)])
            self.assertEqual(actual['r'],sign*active);self.assertEqual(actual['d'],d);self.assertEqual(actual['y_player0'],row['label']['target_player0'])
            self.assertEqual(actual['residual'],(row['label']['target_player0']-d)/(sign*active));self.assertLessEqual(abs(actual['residual']),1)
            actor_counts[p['controller']]+=1;checked+=1
        self.assertEqual(set(actor_counts),{0,1});self.assertEqual(learning.CONTRACT['selection'],'Fixed final optimizer step; development metrics descriptive only')
        EVIDENCE['independent_training_contract']=dict(train_rows=len(expected),excluded_rows=8000-len(expected),root_bridge_train_families=len(bridged),development_rows=1000,target_examples=checked,both_actor_perspectives=True,value_targets='H8 original root finite-choice estimates',policy_targets='Actor-relative H8 root softQ tau2; H7 successor provenance',all_parameter_joint_bundle=True,fixed_final_selection=True,sealed_bodies_read=False)

    def test_07_actual_training_pre_postcommit_kills_and_optimizer_rng_recovery(self):
        runtime=load(ROOT/'build/training-runtime.json')['python'];base=self.fresh('training-kills')
        def command(path):return [runtime,'tools/quality-train.py','train','--fixture','--run',str(path),'--device','cuda']
        def catalog(path):
            with closing(sqlite3.connect(path/'catalog.sqlite3')) as db:return list(db.execute('SELECT id,step,name,sha,body,hash FROM checkpoints ORDER BY id'))
        fresh=base/'fresh';run_child(command(fresh),base/'fresh.log')
        fresh_rows=catalog(fresh);self.assertEqual([r[1] for r in fresh_rows],[2,4,6]);runs=[]
        for event in ('checkpoint_file_ready','checkpoint_committed'):
            path=base/event;probe=base/(event+'-probe')
            kill_barrier(command(path),probe,event,{'QUALITY_TRAIN_BLOCK_EVENT':event,'QUALITY_TRAIN_BLOCK_STEP':'4','QUALITY_TRAIN_MARKER':str(probe/'child-marker.json')},lambda row:row.get('step')==4)
            prefix=catalog(path);self.assertEqual([r[1] for r in prefix],[2] if event=='checkpoint_file_ready' else [2,4]);prefix_files={r[2]:(sha(path/r[2]),(path/r[2]).stat().st_mtime_ns) for r in prefix}
            run_child(command(path),base/(event+'-resume.log'));resumed=catalog(path);self.assertEqual(resumed[:len(prefix)],prefix)
            for name,value in prefix_files.items():self.assertEqual((sha(path/name),(path/name).stat().st_mtime_ns),value)
            self.assertEqual(load(path/'model.json'),load(fresh/'model.json'))
            spec=base/(event+'-comparison-spec.json');out=base/(event+'-comparison.json')
            save(spec,dict(left=str(fresh/fresh_rows[-1][2]),right=str(path/resumed[-1][2]),output=str(out)))
            run_child([runtime,'tools/quality-audit.py','training-state-worker','--spec',str(spec)],base/(event+'-comparison.log'))
            self.assertTrue(load(out)['passed']);before=snapshot(path);run_child(command(path),base/(event+'-complete-rerun.log'));self.assertEqual(snapshot(path),before)
            runs.append(dict(kill_event=event,committed_prefix_steps=[r[1] for r in prefix],resumed_step=6,exact_model_optimizer_and_rng=True,completed_bytes_mtimes_unchanged=True,orphan_file_retained=event=='checkpoint_file_ready'))
        EVIDENCE['live_training_recovery']=dict(device='cuda',steps=6,checkpoint_every=2,runtime=runtime,tests=runs,raw_fixtures_retained=True,actual_joint_cuda_training_killed=True,unrelated_prior_warm_kill_claimed=False)

    def test_08_actual_analysis_pre_postcommit_kills_and_rehashed_corruption(self):
        base=self.fresh('analysis-kills')
        def command(path,simulations=128):return [sys.executable,'-m','congklak.quality_analysis','--path',str(path),'--profile','refined_search','--simulations',str(simulations)]
        fresh=base/'fresh';run_child(command(fresh),base/'fresh.log');reference=load(fresh/'analysis-128.json');records=[]
        for event in ('quality_analysis_precommit','quality_analysis_committed'):
            path=base/event;probe=base/(event+'-probe');run_child(command(path,64),base/(event+'-initial.log'));prefix=((path/'analysis-64.json').read_bytes(),(path/'analysis-64.json').stat().st_mtime_ns)
            kill_barrier(command(path),probe,event,{'QUALITY_ANALYSIS_BLOCK_EVENT':event,'QUALITY_ANALYSIS_MARKER':str(probe/'child-marker.json')})
            with closing(sqlite3.connect(path/'work.sqlite3')) as db:
                text,digest=db.execute("SELECT body,hash FROM meta WHERE key='analysis_result'").fetchone();saved=json.loads(text);self.assertEqual(config.digest(saved),digest)
                self.assertEqual(saved['search']['requested_simulations'],64 if event.endswith('precommit') else 128)
            self.assertEqual(((path/'analysis-64.json').read_bytes(),(path/'analysis-64.json').stat().st_mtime_ns),prefix);self.assertFalse((path/'analysis-128.json').exists())
            run_child(command(path),base/(event+'-resume.log'));restored=load(path/'analysis-128.json')
            self.assertEqual(semantic(restored['search']),semantic(reference['search']));self.assertEqual(restored['rankings'],reference['rankings']);self.assertEqual(restored['explanation'],reference['explanation']);self.assertEqual(restored['evaluator'],reference['evaluator'])
            before=snapshot(path);run_child(command(path),base/(event+'-completed-rerun.log'));self.assertEqual(snapshot(path),before)
            with closing(sqlite3.connect(path/'work.sqlite3')) as db:
                with closing(sqlite3.connect(base/(event+'-validated.sqlite3'))) as backup:db.backup(backup)
                forged=copy.deepcopy(restored)
                if event.endswith('precommit'):forged['evaluator']['cache_hits']+=1
                else:
                    forged['evaluator']['feature_simulations']=forged['evaluator']['feature_deposits']=0;forged['search']['work']['feature_simulations']=forged['search']['work']['feature_deposits']=0;forged['search']['work']['total_deposits']=forged['search']['work']['deposits']
                with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='analysis_result'",(config.canonical(forged),config.digest(forged)))
            bad=snapshot(path);self.assertNotEqual(run_child(command(path),base/(event+'-refusal.log'),success=False),0);self.assertEqual(snapshot(path),bad)
            records.append(dict(event=event,actual_process_kill=True,previous64_export_bytes_mtime_retained=True,fresh128_search_and_physical_counters_equal=True,missing_export_repaired=True,completed_bytes_mtimes_unchanged=True,rehashed_corruption_refused=True))
        EVIDENCE['live_analysis_recovery']=dict(profile='refined_search',cache_capacity=0,records=records,raw_fixtures_retained=True)

    def test_09_actual_case_kill_prefix_and_rehashed_regret_refusal(self):
        base=self.fresh('case-kill');path=base/'run';output=base/'result.json';command=[sys.executable,'tools/benchmark-quality.py','run','--run',str(path),'--output',str(output),'--case-limit','2','--no-arena']
        marker=kill_barrier(command,base/'probe','quality_case_commit_barrier',{'QUALITY_BLOCK_AFTER_CASE':'1'})
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall()
        self.assertEqual(len(prefix),1);run_child(command,base/'resume.log')
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            rows=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall();self.assertEqual(len(rows),2);self.assertEqual(rows[:1],prefix)
            with closing(sqlite3.connect(base/'validated-work.sqlite3')) as backup:db.backup(backup)
            index,text,digest=rows[0];forged=json.loads(text);ms=next(iter(forged['measurements']));name=next(iter(forged['measurements'][ms]));forged['measurements'][ms][name]['reference_regret']=999
            with db:db.execute('UPDATE cases SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),index))
        self.assertNotEqual(run_child(command,base/'refusal.log',success=False),0)
        EVIDENCE['live_case_recovery']=dict(actual_process_kill=True,committed_prefix_retained=1,cases_after_resume=2,rehashed_reference_regret_refused=True,raw_journal_backup_marker_retained=True)

    def test_10_actual_arena_kill_384_literal_replays_and_completed_stability(self):
        from congklak import quality_common as common
        base=self.fresh('arena-kill');path=base/'run';path.mkdir();output=base/'result.json'
        with closing(sqlite3.connect(common.RUN/'work.sqlite3')) as source,closing(sqlite3.connect(path/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM games').fetchone()[0],384)
            with db:db.execute('DELETE FROM games WHERE idx>=382');db.execute("DELETE FROM meta WHERE key='quality_benchmark_complete'")
        command=[sys.executable,'tools/benchmark-quality.py','run','--run',str(path),'--output',str(output)]
        marker=kill_barrier(command,base/'probe','quality_game_commit_barrier',{'QUALITY_BLOCK_AFTER_GAME':'383'})
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall()
        self.assertEqual(len(prefix),383);run_child(command,base/'resume.log')
        choices=0
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            rows=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall();self.assertEqual(len(rows),384);self.assertEqual(rows[:383],prefix)
            for index,text,digest in rows:
                game=json.loads(text);self.assertEqual(config.digest(game),digest);q=normalize_literal(game['start'])
                for action in game['moves']:q=literal.apply(q,action).position;choices+=1
                self.assertEqual(json_position(q),game['final']);self.assertEqual(game['status'],'terminal' if literal.normalize(q).terminal else 'unfinished')
                payoff=literal.evaluate(q)*(1 if game['candidate_seat']==0 else -1)
                self.assertEqual(game['candidate_final_margin'],payoff if game['status']=='terminal' else None)
            with closing(sqlite3.connect(base/'validated-work.sqlite3')) as backup:db.backup(backup)
        stable=snapshot(path);published=(output.read_bytes(),output.stat().st_mtime_ns);run_child(command,base/'completed-rerun.log');self.assertEqual(snapshot(path),stable);self.assertEqual((output.read_bytes(),output.stat().st_mtime_ns),published)
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            index,text,digest=rows[0];forged=json.loads(text);forged['candidate_final_margin']=(forged['candidate_final_margin'] or 0)+2
            with db:db.execute('UPDATE games SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),index))
        self.assertNotEqual(run_child(command,base/'refusal.log',success=False),0)
        EVIDENCE['live_arena_recovery']=dict(actual_process_kill=True,committed_prefix_retained=383,games_after_resume=384,literal_choices=choices,all_games_literal_replayed=True,completed_bytes_mtimes_unchanged=True,rehashed_final_margin_refused=True,raw_journal_backup_marker_retained=True)

