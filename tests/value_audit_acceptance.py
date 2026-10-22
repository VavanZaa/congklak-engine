"""Independent frozen-policy value-head audit and retained recovery fixtures."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from quality_audit_acceptance import load,sha,save,module,normalize_literal,json_position,snapshot,semantic,kill_barrier,run_child,literal
WORK=ROOT/'build/value-audit';WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={}
TRAINABLE=('layers.4.weight','layers.4.bias')

class ValueIndependentAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p

    def test_01_prior_1647_inputs_unchanged(self):
        accepted=load(ROOT/'build/quality-acceptance.json');self.assertTrue(accepted['passed']);self.assertEqual(accepted['tests_run'],36);self.assertEqual(len(accepted['inputs']),1647)
        for name,digest in accepted['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['preserved_prior_inputs']=dict(quality=1647,all_hashes_current=True)

    def test_02_predeclared_value_only_plan(self):
        from congklak import value_common as common
        plan=common.read_plan();self.assertEqual(plan,common.expected_plan());registry=common.read_registry()
        self.assertFalse(plan['sealed_final_test_consumed']);self.assertFalse(plan['default_promotion'])
        self.assertEqual(plan['profiles'],dict(policy_control=dict(model='control',c_puct=.25,cache_capacity=0),value_refit=dict(model='refit',c_puct=.25,cache_capacity=0)))
        self.assertEqual(plan['engines'],['H_v1','H8_relay_CPU','policy_control','value_refit']);self.assertEqual(plan['budgets_ms'],[25,100]);self.assertEqual(plan['arena_ms'],25)
        _,data,_=common.teacher.load();dev=data['development'];prior=load(ROOT/'build/quality-plan.json');used=set(registry['arena_families'])|{r['family'] for r in prior['cases'][:48]}
        def choose(rows,quota,namespace):
            ordered=sorted(rows,key=lambda r:hashlib.sha256((namespace+'/'+r['state_key']).encode()).digest());result=[]
            for kind,count in quota:
                chosen=[]
                for row in ordered:
                    if row['family'] not in used and row['stratum'].endswith('/'+kind):chosen.append(row);used.add(row['family'])
                    if len(chosen)==count:break
                self.assertEqual(len(chosen),count);result+=chosen
            return result
        cases=choose(dev,[('ordinary',16),('capture',16),('bonus',16)],'value-head-cases-v1')
        contested=[r for r in dev if sum(map(sum,r['state']['pits']))>=25 and abs(r['state']['stores'][0]-r['state']['stores'][1])<sum(map(sum,r['state']['pits']))]
        arena=choose(contested,[('ordinary',8),('capture',8),('bonus',8)],'value-head-arena-v1')
        self.assertEqual([r['state_key'] for r in plan['cases'][:48]],[r['state_key'] for r in cases]);self.assertEqual([r['state_key'] for r in plan['arena_starts']],[r['state_key'] for r in arena]);self.assertEqual(plan['cases'][48:],prior['cases'][48:])
        for a,b in zip(plan['cases'][:48],cases):self.assertEqual(a['reference'],b['label'])
        comparisons=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=24) for c in ('policy_control','value_refit') for b in ('H_v1','H8_relay_CPU')];self.assertEqual(plan['arena_comparisons'],comparisons);names=[r['id'] for r in comparisons];order=[]
        for pair in range(24):
            for name in names[pair%4:]+names[:pair%4]:
                for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
        self.assertEqual(plan['arena_work_order'],order);self.assertEqual(len(order),192);self.assertEqual((plan['bootstrap_seed'],plan['bootstrap_samples']),(20261019,4096))
        self.assertEqual(plan['puct'],dict(simulations=100000,max_nodes=2048,root_noise=None,first_play_urgency='parent_neural_bootstrap'))
        EVIDENCE['predeclared_value_only']=dict(profiles=2,cases=80,scalar_measurements=640,arena_families=24,games=192,independent_namespaces_labels_order=True,earlier_diagnostic_family_reuse_disclosed=True,ancestral_exposure=True,sealed_read=False,automatic_promotion=False)

    def test_04_global_frozen_policy_structure_and_native_literal_outputs(self):
        from congklak import value_common as common
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        plan=common.read_plan();a=load(ROOT/plan['models']['control'])['model'];b=load(ROOT/plan['models']['refit'])['model'];names=set(a['weights'])-set(TRAINABLE)
        self.assertEqual(names,{'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'});self.assertEqual(a['encoder'],b['encoder'])
        def raw(value):
            if isinstance(value,list):return b''.join(raw(x) for x in value)
            return struct.pack('<f',value)
        self.assertEqual(sum(len(raw(a['weights'][n]))//4 for n in names),12807)
        for name in names:self.assertEqual(raw(a['weights'][name]),raw(b['weights'][name]),name)
        roots=[r['state'] for r in plan['cases']]+[dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0),dict(pits=[[0]*7,[0]*7],stores=[40,58],controller=1)]
        count=0
        with CpuBatchEvaluator(ROOT/plan['models']['control'],max_batch=32) as left,CpuBatchEvaluator(ROOT/plan['models']['refit'],max_batch=32) as right:
            for offset in range(0,len(roots),32):
                batch=roots[offset:offset+32]
                for p,x,y in zip(batch,left.evaluate_many(batch),right.evaluate_many(batch)):
                    self.assertEqual(x['policy_logits'],y['policy_logits']);self.assertEqual(x['feature_simulations'],y['feature_simulations']);self.assertEqual(x['feature_deposits'],y['feature_deposits'])
                    for row,model in ((x,a),(y,b)):
                        value,_,logits=value_and_policy(normalize_literal(p),model);self.assertAlmostEqual(row['value_player0'],value,delta=1e-10);self.assertLessEqual(max(abs(z-w) for z,w in zip(logits,row['policy_logits'])),1e-10)
                    count+=1
        EVIDENCE['frozen_policy_proof']=dict(frozen_tensors=sorted(names),frozen_parameters=12807,trainable_parameters=65,native_positions=count,policy_logits_bit_exact=True,literal_model_tolerance=1e-10,global_policy_unchanged_by_identical_subgraph=True,scope='Identical encoder plus identical two hidden layers and policy head; value head is outside policy dataflow')

    def test_05_frozen_checkpoint_and_two_parameter_optimizer_refusals(self):
        runtime=load(ROOT/'build/training-runtime.json')['python'];base=self.fresh('state-refusals');output=base/'result.json'
        run_child([runtime,'tools/value-audit.py','state-refusal-worker','--spec',str(output)],base/'worker.log');body=load(output);self.assertTrue(body['passed']);self.assertEqual(body['frozen_tensors_mutated_and_refused'],6);self.assertEqual(body['allowed_value_head_mutations'],2)
        EVIDENCE['checkpoint_contract_refusals']=body

    def test_10_expired_deadline_and_physical_counter_admission(self):
        from congklak import value_common as common
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        bench=module('value_independent_deadline','tools/benchmark-value.py');plan=common.read_plan();p=plan['cases'][0]['state'];checked=0
        for name in common.PROFILES:
            with CpuBatchEvaluator(ROOT/plan['models'][plan['profiles'][name]['model']],cache_capacity=0,max_batch=32) as evaluator:
                ticks=iter(i*.002 for i in range(100))
                with mock.patch.object(bench.time,'perf_counter',side_effect=lambda:next(ticks)),mock.patch.object(bench.time,'sleep'),mock.patch.object(bench,'puct_search',side_effect=AssertionError('Expired integer budget became unlimited')):
                    empty=bench.decide(name,('puct',evaluator),p,1,plan)
                self.assertFalse(empty['decision']['usable_decision']);self.assertEqual(empty['evaluator']['granted_search_ms'],0);self.assertEqual(empty['decision']['completed_simulations'],0)
                answer=bench.decide(name,('puct',evaluator),p,25,plan);forged=copy.deepcopy(answer);counts=forged['evaluator']['counts'];counts['feature_simulations']=counts['feature_deposits']=0;work=forged['decision']['report']['work'];work['feature_simulations']=work['feature_deposits']=0;work['total_deposits']=work['deposits']
                self.assertGreater(counts['native_actual_evals'],0)
                with self.assertRaises(Exception):bench.validate_envelope(forged,p,25,plan,name)
                checked+=1
        EVIDENCE['deadline_and_physical_work']=dict(profiles=checked,expired_budget_no_search=True,zero_never_unlimited=True,rehashed_zero_feature_work_refused=True)


    def test_03_training_filter_and_seed_target_derivation(self):
        from congklak import value_learning as learning,deeper_teacher_v2 as teacher,pilot
        admitted,data=learning.prepare();registry=learning.read_registry();source,original,_=teacher.load(teacher_depth=8)
        forbidden_q=set(registry['quotients']);forbidden_f=set(registry['families'])
        bridged={row['family'] for row in original['train'] if pilot.quotient(row['state']).hex() in forbidden_q}
        expected=[row for row in original['train'] if row['family'] not in forbidden_f|bridged and pilot.quotient(row['state']).hex() not in forbidden_q]
        self.assertEqual([r['state_key'] for r in data['train']],[r['state_key'] for r in expected])
        self.assertEqual(data['development'],original['development']);self.assertEqual(len(data['development']),1000)
        self.assertEqual(admitted['train_rows'],len(expected));self.assertEqual(admitted['train_families'],sorted({r['family'] for r in expected}))
        self.assertEqual(admitted['train_strata'],dict(Counter(r['stratum'] for r in expected)))
        self.assertTrue({r['state_key'] for r in expected}<=set(load(ROOT/'build/quality-data.json')['train_state_keys']))
        self.assertEqual((learning.CONTRACT['trainable_parameters'],learning.CONTRACT['frozen_parameters'],learning.CONTRACT['value_huber_delta'],learning.CONTRACT['policy_loss_weight']),(65,12807,4.,0.))
        self.assertEqual(learning.CONTRACT['objective'],'Huber(D_player0+signed_active*tanh_head,H8_root_player0_SEEDS,delta4)')
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
        EVIDENCE['independent_training_contract']=dict(train_rows=len(expected),excluded_rows=8000-len(expected),root_bridge_train_families=len(bridged),development_rows=1000,target_examples=checked,both_actor_perspectives=True,value_targets='H8 original root finite-choice estimates in reconstructed player0 SEEDS',policy_targets='Descriptive only; no policy gradient',trainable_parameters=65,frozen_parameters=12807,seed_space_huber_delta=4.,fixed_final_selection=True,sealed_bodies_read=False)

    def test_06_actual_training_pre_postcommit_kills_and_optimizer_rng_recovery(self):
        runtime=load(ROOT/'build/training-runtime.json')['python'];base=self.fresh('training-kills')
        def command(path):return [runtime,'tools/value-train.py','train','--fixture','--run',str(path),'--device','cuda']
        def catalog(path):
            with closing(sqlite3.connect(path/'catalog.sqlite3')) as db:return list(db.execute('SELECT id,step,name,sha,body,hash FROM checkpoints ORDER BY id'))
        fresh=base/'fresh';run_child(command(fresh),base/'fresh.log')
        fresh_rows=catalog(fresh);self.assertEqual([r[1] for r in fresh_rows],[2,4,6]);runs=[]
        for event in ('checkpoint_file_ready','checkpoint_committed'):
            path=base/event;probe=base/(event+'-probe')
            kill_barrier(command(path),probe,event,{'VALUE_TRAIN_BLOCK_EVENT':event,'VALUE_TRAIN_BLOCK_STEP':'4','VALUE_TRAIN_MARKER':str(probe/'child-marker.json')},lambda row:row.get('step')==4)
            prefix=catalog(path);self.assertEqual([r[1] for r in prefix],[2] if event=='checkpoint_file_ready' else [2,4]);prefix_files={r[2]:(sha(path/r[2]),(path/r[2]).stat().st_mtime_ns) for r in prefix}
            run_child(command(path),base/(event+'-resume.log'));resumed=catalog(path);self.assertEqual(resumed[:len(prefix)],prefix)
            for name,value in prefix_files.items():self.assertEqual((sha(path/name),(path/name).stat().st_mtime_ns),value)
            self.assertEqual(load(path/'model.json'),load(fresh/'model.json'))
            spec=base/(event+'-comparison-spec.json');out=base/(event+'-comparison.json')
            save(spec,dict(left=str(fresh/fresh_rows[-1][2]),right=str(path/resumed[-1][2]),output=str(out)))
            run_child([runtime,'tools/value-audit.py','training-state-worker','--spec',str(spec)],base/(event+'-comparison.log'))
            self.assertTrue(load(out)['passed']);before=snapshot(path);run_child(command(path),base/(event+'-complete-rerun.log'));self.assertEqual(snapshot(path),before)
            runs.append(dict(kill_event=event,committed_prefix_steps=[r[1] for r in prefix],resumed_step=6,exact_model_optimizer_and_rng=True,completed_bytes_mtimes_unchanged=True,orphan_file_retained=event=='checkpoint_file_ready'))
        EVIDENCE['live_training_recovery']=dict(device='cuda',steps=6,checkpoint_every=2,runtime=runtime,tests=runs,raw_fixtures_retained=True,actual_value_cuda_training_killed=True,unrelated_prior_warm_kill_claimed=False)

    def test_07_actual_analysis_pre_postcommit_kills_and_rehashed_corruption(self):
        base=self.fresh('analysis-kills')
        def command(path,simulations=128):return [sys.executable,'-m','congklak.value_analysis','--path',str(path),'--profile','value_refit','--simulations',str(simulations)]
        fresh=base/'fresh';run_child(command(fresh),base/'fresh.log');reference=load(fresh/'analysis-128.json');records=[]
        for event in ('value_analysis_precommit','value_analysis_committed'):
            path=base/event;probe=base/(event+'-probe');run_child(command(path,64),base/(event+'-initial.log'));prefix=((path/'analysis-64.json').read_bytes(),(path/'analysis-64.json').stat().st_mtime_ns)
            kill_barrier(command(path),probe,event,{'VALUE_ANALYSIS_BLOCK_EVENT':event,'VALUE_ANALYSIS_MARKER':str(probe/'child-marker.json')})
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
        EVIDENCE['live_analysis_recovery']=dict(profile='value_refit',cache_capacity=0,records=records,raw_fixtures_retained=True)

    def test_08_actual_case_kill_prefix_and_rehashed_regret_refusal(self):
        base=self.fresh('case-kill');path=base/'run';output=base/'result.json';command=[sys.executable,'tools/benchmark-value.py','run','--run',str(path),'--output',str(output),'--case-limit','2','--no-arena']
        marker=kill_barrier(command,base/'probe','value_case_commit_barrier',{'VALUE_BLOCK_AFTER_CASE':'1'})
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall()
        self.assertEqual(len(prefix),1);run_child(command,base/'resume.log')
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            rows=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall();self.assertEqual(len(rows),2);self.assertEqual(rows[:1],prefix)
            with closing(sqlite3.connect(base/'validated-work.sqlite3')) as backup:db.backup(backup)
            index,text,digest=rows[0];forged=json.loads(text);ms=next(iter(forged['measurements']));name=next(iter(forged['measurements'][ms]));forged['measurements'][ms][name]['reference_regret']=999
            with db:db.execute('UPDATE cases SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),index))
        self.assertNotEqual(run_child(command,base/'refusal.log',success=False),0)
        EVIDENCE['live_case_recovery']=dict(actual_process_kill=True,committed_prefix_retained=1,cases_after_resume=2,rehashed_reference_regret_refused=True,raw_journal_backup_marker_retained=True)

    def test_09_actual_arena_kill_192_literal_replays_and_completed_stability(self):
        from congklak import value_common as common
        base=self.fresh('arena-kill');path=base/'run';path.mkdir();output=base/'result.json'
        with closing(sqlite3.connect(common.RUN/'work.sqlite3')) as source,closing(sqlite3.connect(path/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM games').fetchone()[0],192)
            with db:db.execute('DELETE FROM games WHERE idx>=190');db.execute("DELETE FROM meta WHERE key='value_benchmark_complete'")
        command=[sys.executable,'tools/benchmark-value.py','run','--run',str(path),'--output',str(output)]
        marker=kill_barrier(command,base/'probe','value_game_commit_barrier',{'VALUE_BLOCK_AFTER_GAME':'191'})
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall()
        self.assertEqual(len(prefix),191);run_child(command,base/'resume.log')
        choices=0
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            rows=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall();self.assertEqual(len(rows),192);self.assertEqual(rows[:191],prefix)
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
        EVIDENCE['live_arena_recovery']=dict(actual_process_kill=True,committed_prefix_retained=191,games_after_resume=192,literal_choices=choices,all_games_literal_replayed=True,completed_bytes_mtimes_unchanged=True,rehashed_final_margin_refused=True,raw_journal_backup_marker_retained=True)
