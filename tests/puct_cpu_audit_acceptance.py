"""Independent CPU PUCT efficiency, state isolation and recovery acceptance."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing,ExitStack
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
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
from congklak import config,puct_cpu_common as common
from congklak.native import Native,flatten
from congklak.phase6_inference import BatchEvaluator,InferenceStopped
from congklak.puct import search,forest,CancellationToken
from phase6_audit_acceptance import normalize_literal,json_position
from stage5_deeper_audit_acceptance import literal_features,value_and_policy
import reference as literal

WORK=ROOT/'build/puct-cpu-audit';WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={}

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def semantic(report):
    keys=('protocol','rules_digest','position','chosen','visits','policy','root_priors','estimated_value_player0','action_mean_value_player0','root_expanded','completed_simulations','requested_simulations','nodes','max_nodes','usable_decision','stop_reason','proof_status','unvisited_root_q_player0')
    return {k:report[k] for k in keys}
def snapshot(path):
    return {p.relative_to(path).as_posix():(sha(p),p.stat().st_mtime_ns) for p in Path(path).rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))}
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def kill_barrier(command,path,expected_event,env=None):
    path.mkdir();log=path/'worker.log';marker=path/'marker.json'
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',env={**os.environ,**(env or {})})
        lines=[];done=threading.Event();error=[]
        def reader():
            try:
                for line in child.stdout:
                    stream.write(line);stream.flush();lines.append(line)
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get('event')==expected_event:marker.write_text(json.dumps(event));done.set();return
            except BaseException as exc:error.append(exc);done.set()
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        try:
            deadline=time.monotonic()+180
            while not done.wait(.05) and child.poll() is None and time.monotonic()<deadline:pass
            if not marker.exists():raise AssertionError('Kill barrier unavailable: '+str(log)+'\n'+''.join(lines))
            child.kill();child.wait(timeout=10);thread.join(5)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=10)
    return load(marker)

class PuctCpuIndependentAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
        assert cls.base.resolve().is_relative_to(WORK.resolve())
    @classmethod
    def tearDownClass(cls):pass
    def fresh(self,name):
        path=self.base/name;path.mkdir();return path

    def test_01_prior_acceptance_and_model_artifacts_preserved(self):
        accepted=load(ROOT/'build/phase6-acceptance.json')
        self.assertTrue(accepted['passed'] and accepted['engineering_passed'])
        self.assertEqual(accepted['tests_run'],42);self.assertEqual(len(accepted['inputs']),1192)
        for path,digest in accepted['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        older=load(ROOT/'build/stage5-deeper-acceptance.json')
        self.assertTrue(older['passed']);self.assertEqual(len(older['inputs']),878)
        for path,digest in older['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['preserved_prior_inputs']=dict(phase6=1192,stage5=878,all_hashes_current=True,training_performed=False)

    def test_02_predeclared_cohorts_profiles_clocks_and_game_order(self):
        plan=load(common.PLAN);prior=load(ROOT/'build/phase6-plan.json')
        self.assertEqual(plan,common.expected_plan())
        self.assertEqual(plan['cases'],prior['cases']);self.assertEqual(plan['arena_starts'],prior['arena_starts'])
        self.assertEqual((len(plan['cases']),len(plan['arena_starts'])),(80,24))
        profiles=('warm_legacy','updated_legacy','warm_single','updated_single','warm_cache','updated_cache')
        baselines=('H_v1','H8_relay_CPU')
        self.assertEqual(plan['engines'],list(baselines+profiles));self.assertEqual(set(plan['profiles']),set(profiles))
        for name in profiles:
            self.assertEqual(plan['profiles'][name],dict(model='warm' if name.startswith('warm_') else 'updated',evaluator='legacy' if name.endswith('legacy') else 'single_pass',cache_capacity=4096 if name.endswith('cache') else 0,physical_feature_passes_per_miss=2 if name.endswith('legacy') else 1))
        self.assertEqual(plan['budgets_ms'],[25,100]);self.assertEqual(plan['arena_ms'],25)
        self.assertEqual((plan['arena_max_choices'],plan['puct']['max_nodes'],plan['memory_mb'],plan['tt_mb'],plan['max_depth']),(512,2048,8192,16,16))
        self.assertEqual((plan['puct']['simulations'],plan['puct']['c_puct'],plan['puct']['root_noise']),(100000,1.5,None))
        self.assertEqual((plan['bootstrap_samples'],plan['bootstrap_seed']),(4096,20261017))
        self.assertEqual(plan['cpu_declared'],prior['cpu_declared'])
        self.assertEqual(plan['cache_capacity'],4096)
        self.assertIn('input16cells+inputcontroller',plan['cache_policy']);self.assertIn('inside',plan['cache_reset'])
        self.assertIn('never time_ms=0',plan['clock']);self.assertIn('logical',plan['telemetry'])
        comparisons=[dict(candidate=name,opponent=b,id=name+'_vs_'+b,pairs=24) for name in profiles for b in baselines]
        self.assertEqual(plan['arena_comparisons'],comparisons)
        ids=[c['id'] for c in comparisons];expected=[]
        for pair in range(24):
            offset=pair%12
            for name in ids[offset:]+ids[:offset]:
                for seat in ((0,1) if pair%2==0 else (1,0)):expected.append(dict(comparison=name,pair=pair,seat=seat))
        self.assertEqual(plan['arena_work_order'],expected)
        self.assertEqual(len(expected),len({(x['comparison'],x['pair'],x['seat']) for x in expected}))
        self.assertEqual(len(expected),576)
        self.assertEqual(plan['speed']['batch_sizes'],[1,2,4,8,16,32])
        self.assertEqual((plan['speed']['blocks'],plan['speed']['repetitions'],plan['speed']['tree_blocks'],plan['speed']['tree_simulations']),(7,64,3,64))
        self.assertEqual(len(common.speed_roots(plan)),48)
        self.assertIn('outside',plan['speed']['hot']);self.assertIn('retrieval ceiling',plan['speed']['hot'])
        self.assertFalse(plan['sealed_final_test_consumed']);self.assertIn('No training',plan['training']);self.assertIn('no automatic promotion',plan['release'])
        for name,path in plan['models'].items():self.assertEqual(plan['model_files_sha256'][name],sha(ROOT/path))
        EVIDENCE['predeclared_protocol']=dict(cases=80,scalar_measurements=1280,arena_families=24,comparisons=12,games=576,exact_order_rederived=True,models_frozen=True,cold_cache_and_hot_retrieval_separate=True,development_reuse=True,unadjusted_exploratory_intervals=True)

    def test_03_literal_structured_forward_and_actual_single_pass_work(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        roots=[row['state'] for row in load(common.PLAN)['cases']]
        roots += [dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=actor) for actor in (0,1)]
        roots += [dict(pits=[[0]*7,[0]*7],stores=[98,0],controller=1)]
        total=0
        for modelpath in (common.WARM,common.UPDATE):
            with BatchEvaluator(modelpath,'native',max_batch=32) as old,CpuBatchEvaluator(modelpath,max_batch=32,backend='avx2') as new:
                for start in range(0,len(roots),32):
                    group=roots[start:start+32];original=old.evaluate_many(group);actual=new.evaluate_many(group)
                    for p,left,right in zip(group,original,actual):
                        expected,_,logits=value_and_policy(normalize_literal(p),new.model.envelope['model'])
                        self.assertEqual(left['value_player0'],right['value_player0']);self.assertEqual(left['policy_logits'],right['policy_logits'])
                        self.assertAlmostEqual(right['value_player0'],expected,delta=1e-10)
                        self.assertLessEqual(max(abs(a-b) for a,b in zip(logits,right['policy_logits'])),1e-10)
                        feature=old.native.feature_vector(p)
                        for key in ('feature_simulations','feature_deposits'):
                            self.assertEqual(right[key],feature[key]);self.assertEqual(left[key],2*right[key])
                        self.assertEqual(right['native_actual_evals'],int(bool(sum(map(sum,p['pits'])))));total+=1
                counts=new.snapshot_counters()
                self.assertEqual(counts['native_actual_evals'],counts['cache_misses']);self.assertEqual(counts['native_calls'],counts['predict_successes'])
                self.assertEqual(counts['terminal_returns'],1);self.assertEqual(counts['cache_hits'],0)
        EVIDENCE['literal_structured_forward']=dict(models=2,positions=total,legacy_bitwise_value_logits=True,literal_tolerance=1e-10,actual_single_pass_feature_work=True)

    def test_04_raw_state_store_actor_model_isolation_and_copy_out(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator,_raw_key
        p=dict(pits=[[1]*7,[1]*7],stores=[42,42],controller=0);shift=copy.deepcopy(p);shift['stores']=[43,41]
        forced=dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0);normalized=copy.deepcopy(forced);normalized['controller']=1
        self.assertNotEqual(_raw_key(p),_raw_key(shift));self.assertNotEqual(_raw_key(forced),_raw_key(normalized));self.assertEqual(len(_raw_key(p)),17)
        outputs=[]
        for modelpath in (common.WARM,common.UPDATE):
            with CpuBatchEvaluator(modelpath,cache_capacity=4096) as evaluator:
                left,right,pass0,pass1=evaluator.evaluate_many([p,shift,forced,normalized])
                self.assertEqual(evaluator.snapshot_counters()['cache_misses'],4);self.assertAlmostEqual(right['value_player0']-left['value_player0'],2,delta=1e-12)
                self.assertEqual(pass0['value_player0'],pass1['value_player0']);self.assertEqual(pass0['policy_logits'],pass1['policy_logits'])
                logits=list(left['policy_logits']);left['policy_logits'][0]=1234567
                again=evaluator.evaluate_many([p])[0]
                self.assertEqual(again['policy_logits'],logits);self.assertTrue(again['cache_hit'])
                self.assertEqual((again['feature_simulations'],again['feature_deposits'],again['native_actual_evals']),(0,0,0))
                outputs.append((again['value_player0'],again['policy_logits']))
                evaluator.reset_cache();self.assertEqual(evaluator.snapshot_counters()['cache_current_entries'],0)
                self.assertFalse(evaluator.evaluate_many([p])[0]['cache_hit'])
        self.assertNotEqual(outputs[0],outputs[1])
        EVIDENCE['raw_cache_isolation']=dict(full_input_bytes=17,stores_and_input_actor_in_key=True,forced_pass_aliases_separate_misses=True,models_private=True,returned_logits_do_not_mutate_cache=True,hits_zero_physical_work=True)

    def test_05_actual_4096_entry_LRU_bound_and_reset(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        roots=[]
        for n in range(4097):
            digits=n%2401;flat=[1]*14
            for i in range(4):flat[i]+=digits%7;digits//=7
            remaining=98-sum(flat);roots.append(dict(pits=[flat[:7],flat[7:]],stores=[remaining//2,remaining-remaining//2],controller=n//2401))
        with CpuBatchEvaluator(common.UPDATE,cache_capacity=4096,max_batch=32) as evaluator:
            for start in range(0,4096,32):evaluator.evaluate_many(roots[start:start+32])
            self.assertEqual(evaluator.snapshot_counters()['cache_current_entries'],4096)
            self.assertTrue(evaluator.evaluate_many([roots[0]])[0]['cache_hit'])
            self.assertFalse(evaluator.evaluate_many([roots[4096]])[0]['cache_hit'])
            self.assertTrue(evaluator.evaluate_many([roots[0]])[0]['cache_hit'])
            self.assertFalse(evaluator.evaluate_many([roots[1]])[0]['cache_hit'])
            counts=evaluator.snapshot_counters();self.assertEqual(counts['cache_current_entries'],4096);self.assertEqual(counts['cache_evictions'],2)
            self.assertEqual(counts['evaluations_requested'],counts['cache_hits']+counts['cache_misses']);self.assertEqual(counts['cache_misses'],counts['native_actual_evals'])
            evaluator.reset_cache();self.assertEqual(evaluator.snapshot_counters()['cache_current_entries'],0)
        EVIDENCE['bounded_LRU']=dict(actual_capacity=4096,distinct_states=4097,MRU_touch_preserved=True,evictions=2,reset_clears_occupancy=True)

    def test_06_fixed_simulation_forest_noise_values_visits_and_RNG(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        cases=load(common.PLAN)['cases'];roots=[cases[i]['state'] for i in (0,16,32,1,17,33)]
        roots += [dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0),dict(pits=[[2,1,0,0,0,0,3],[1,0,2,0,1,0,2]],stores=[43,43],controller=0)]
        checked=0
        for modelpath in (common.WARM,common.UPDATE):
            with BatchEvaluator(modelpath,'native',max_batch=8) as old,CpuBatchEvaluator(modelpath,max_batch=8) as single,CpuBatchEvaluator(modelpath,cache_capacity=4096,max_batch=8) as cached:
                expected=None;expected_rng=None
                for evaluator in (old,single,cached):
                    if hasattr(evaluator,'reset_cache'):evaluator.reset_cache()
                    rng=[random.Random(610+n) for n in range(8)]
                    result=forest(roots,evaluator,simulations=128,max_nodes=2048,rng=rng,root_noise=dict(alpha=.3,fraction=.25))
                    self.assertTrue(all(r['completed_simulations']==128 for r in result));values=[semantic(r) for r in result];state=[r.getstate() for r in rng]
                    if expected is None:expected=values;expected_rng=state
                    else:self.assertEqual(values,expected);self.assertEqual(state,expected_rng);checked+=8
        EVIDENCE['fixed_simulation_conformance']=dict(models=2,roots=8,simulations=128,optimized_root_comparisons=checked,exact_visits_priors_values_nodes=True,exact_per_tree_noise_RNG=True,frozen_search_reused=True)

    def test_07_cancelled_miss_hit_and_lock_wait_have_truthful_prefix_work(self):
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        p=config.position()
        with CpuBatchEvaluator(common.WARM,cache_capacity=4096) as evaluator:
            token=CancellationToken();token.cancel();before=evaluator.snapshot_counters()
            with self.assertRaises(InferenceStopped):evaluator.evaluate_many([p],cancel=token)
            self.assertEqual(evaluator.snapshot_counters(),before)
            evaluator.evaluate_many([p]);before=evaluator.snapshot_counters();polls=0
            def cancel_after_hit():
                nonlocal polls
                polls+=1;return polls==4
            with self.assertRaises(InferenceStopped) as context:evaluator.evaluate_many([p],cancel=cancel_after_hit)
            after=evaluator.snapshot_counters();self.assertEqual(after['cache_hits']-before['cache_hits'],1)
            self.assertEqual(after['native_actual_evals'],before['native_actual_evals']);self.assertFalse(context.exception.work['per_position'][0]['evaluated'])
            self.assertEqual(context.exception.work['feature_deposits'],0)
            with self.assertRaises(InferenceStopped):evaluator.evaluate_many([p],deadline=time.perf_counter()-1)
            outcome=[];started=threading.Event();evaluator.lock.acquire()
            def waiting():
                started.set()
                try:evaluator.evaluate_many([p],deadline=time.perf_counter()+.02)
                except InferenceStopped as exc:outcome.append(exc)
            thread=threading.Thread(target=waiting);thread.start();started.wait(1);thread.join(1);evaluator.lock.release()
            self.assertFalse(thread.is_alive());self.assertEqual(len(outcome),1);self.assertEqual(outcome[0].work['native_actual_evals'],0)
        EVIDENCE['cancelled_prefix_accounting']=dict(pre_cancel_zero_work=True,discarded_cache_hit_physical_flag=False,cache_hit_request_distinct_from_physical_forward=True,expired_deadline_refused=True,lock_wait_deadline_bounded=True)

    def test_08_bad_ABI_model_input_and_closed_evaluator_are_refused(self):
        import ctypes as C
        from congklak import puct_cpu_inference as inference
        from congklak.native import NativeError
        out=inference.PredictionResult();out.abi_version=99;out.struct_size=C.sizeof(out)
        with self.assertRaises(NativeError):inference.CpuBatchEvaluator._validate_result(out,0,inference.PredictionResult)
        with mock.patch.object(inference,'DLL',ROOT/'build/congklak_neural_relay.dll'):
            with self.assertRaises(NativeError):inference.CpuBatchEvaluator(common.WARM)
        original_load=inference.json.loads
        def missing_build_inputs(text,*args,**kwargs):
            value=original_load(text,*args,**kwargs)
            if isinstance(value,dict) and value.get('schema')=='puct-cpu-structured-build-v1':value['inputs']={}
            return value
        with mock.patch.object(inference.json,'loads',side_effect=missing_build_inputs):
            with self.assertRaises(NativeError):inference.CpuBatchEvaluator(common.WARM)
        bad=self.fresh('invalid-model')/'model.json';envelope=load(common.WARM);envelope['model']['uses_oracle']=True;envelope['sha256']=config.digest(envelope['model']);bad.write_text(config.canonical(envelope))
        with self.assertRaises(NativeError):inference.CpuBatchEvaluator(bad)
        with inference.CpuBatchEvaluator(common.WARM) as evaluator:
            invalid=config.position();invalid['controller']=2
            with self.assertRaises((ValueError,NativeError)):evaluator.evaluate_many([invalid])
            cells,_=evaluator.native._array(flatten(config.position()),7,98);result=inference.PredictionResult();evaluator._begin(None)
            status=evaluator.native.lib.cc_puct_predict(cells,2,evaluator.native.handle,evaluator._control.handle,C.byref(result))
            self.assertEqual(status,2);self.assertEqual(result.neural_evaluations,0);self.assertEqual(result.feature_deposits,0)
        with self.assertRaises(NativeError):evaluator.evaluate_many([config.position()])
        EVIDENCE['failure_admission']=dict(wrong_build_or_ABI_refused=True,rehashed_oracle_model_refused=True,invalid_controller_structured_error=True,closed_evaluator_refused=True,no_silent_legacy_fallback=True)

    def test_09_deadline_floor_legacy_terminal_and_speed_counter_admission(self):
        bench=module('cpu_benchmark_independent_wrapper','tools/benchmark-puct-cpu.py');plan=load(common.PLAN)
        class SlowReset:
            def __init__(self):self.counts={key:0 for key in bench.COUNTERS}
            def snapshot_counters(self):return {**self.counts,'cache_capacity':4096,'cache_occupancy':0}
            def reset_cache(self):time.sleep(.03);self.counts['cache_resets']+=1
        with mock.patch.object(bench,'puct_search',side_effect=AssertionError('Unbounded time_ms=0 search attempted')):
            answer=bench.decide('warm_cache',('puct',SlowReset()),config.position(),25,plan)
        self.assertFalse(answer['decision']['usable_decision']);self.assertEqual(answer['evaluator']['granted_search_ms'],0)
        self.assertGreaterEqual(answer['decision']['wall_ms'],30);self.assertEqual(answer['evaluator']['counts']['native_actual_evals'],0)
        terminal=dict(pits=[[0]*7,[0]*7],stores=[49,49],controller=0)
        with BatchEvaluator(common.WARM,'native') as raw:
            legacy=bench.LegacyTelemetry(raw);token=CancellationToken();token.cancel()
            with self.assertRaises(InferenceStopped):legacy.evaluate_many([terminal],cancel=token)
            self.assertEqual(legacy.snapshot_counters()['terminal_returns'],0)
            before=legacy.snapshot_counters();legacy.uninstrumented_reference(config.position());self.assertEqual(legacy.snapshot_counters(),before)
        speed=module('cpu_speed_independent_admission','tools/puct-cpu-speed.py')
        counts={key:0 for key in bench.COUNTERS};counts.update(positions_submitted=64,evaluations_requested=64,native_actual_evals=64,predict_successes=64,cache_misses=64,native_calls=64,cache_resets=64)
        forged=dict(whole_call_ns=1e6,position_ns=1e6,repetitions=64,checksum=1.,counts=counts,cache_capacity=0,cache_occupancy=0)
        with self.assertRaises(ValueError):speed.check(('cold','warm_single',1,0,-1),forged,plan,bench)
        EVIDENCE['wrapper_and_speed_failure_admission']=dict(cache_reset_spent_budget=True,zero_search_budget_never_means_unlimited=True,pre_cancel_terminal_no_credit=True,reference_diagnostics_do_not_change_counters=True,rehashed_zero_feature_work_refused=True)

    def test_10_actual_analysis_pre_postcommit_kills_and_counter_corruption(self):
        root=self.fresh('analysis-kills')
        code="import sys,json;from pathlib import Path;from congklak import config,puct_cpu_common as c,puct_cpu_analysis as a;r=a.analyze(Path(sys.argv[1]),config.position(),c.UPDATE,simulations=int(sys.argv[2]));print(json.dumps(r))"
        def invoke(path,simulations=128,fail=False):
            result=subprocess.run([sys.executable,'-c',code,str(path),str(simulations)],cwd=ROOT,capture_output=True,text=True)
            (root/(path.name+('-refusal.log' if fail else '-last.log'))).write_text(result.stdout+result.stderr)
            self.assertEqual(result.returncode==0,not fail,result.stdout+result.stderr)
            return None if fail else load(path/f'analysis-{simulations}.json')
        fresh=root/'fresh';reference=invoke(fresh);records=[]
        for event in ('puct_cpu_analysis_precommit','puct_cpu_analysis_committed'):
            run=root/event;invoke(run,64);prefix=(run/'analysis-64.json').read_bytes()
            custom="import sys,json,time;from pathlib import Path;from congklak import config,puct_cpu_common as c,puct_cpu_analysis as a\ndef callback(v):\n print(json.dumps(v),flush=True)\n if v['event']==sys.argv[2]:\n  print(json.dumps(dict(event='analysis_kill_barrier')),flush=True)\n  while True:time.sleep(.05)\na.analyze(Path(sys.argv[1]),config.position(),c.UPDATE,simulations=128,on_event=callback)"
            marker=kill_barrier([sys.executable,'-c',custom,str(run),event],root/(event+'-probe'),'analysis_kill_barrier')
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                body,digest=db.execute("SELECT body,hash FROM meta WHERE key='analysis_result'").fetchone();saved=json.loads(body);self.assertEqual(config.digest(saved),digest)
                self.assertEqual(saved['search']['requested_simulations'],64 if event.endswith('precommit') else 128)
            self.assertEqual((run/'analysis-64.json').read_bytes(),prefix);self.assertFalse((run/'analysis-128.json').exists())
            restored=invoke(run);self.assertEqual(semantic(restored['search']),semantic(reference['search']));self.assertEqual(restored['rankings'],reference['rankings']);self.assertEqual(restored['explanation'],reference['explanation'])
            self.assertEqual(restored['evaluator'],reference['evaluator']);stable=snapshot(run);self.assertEqual(invoke(run),restored);self.assertEqual(snapshot(run),stable)
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                with closing(sqlite3.connect(root/(event+'-validated.sqlite3'))) as backup:db.backup(backup)
                forged=copy.deepcopy(restored);forged['evaluator']['cache_hits']+=1
                with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='analysis_result'",(config.canonical(forged),config.digest(forged)))
            bad=snapshot(run);invoke(run,fail=True);self.assertEqual(snapshot(run),bad)
            records.append(dict(event=event,actual_process_kill=True,prior_completed_bytes_retained=True,fresh_semantic_and_counters_equal=True,missing_export_repaired=True,finished_bytes_mtimes_stable=True,rehashed_counter_corruption_refused=True,barrier=marker))
        EVIDENCE['analysis_actual_recovery']=dict(checks=records,model='updated',cache_capacity=4096,raw_states_and_logs_retained=True)

    def test_11_actual_case_kill_prefix_and_rehashed_regret_refusal(self):
        root=self.fresh('case-kill');run=root/'run';output=root/'result.json'
        command=[sys.executable,str(ROOT/'tools/benchmark-puct-cpu.py'),'run','--run',str(run),'--output',str(output),'--case-limit','2','--no-arena']
        marker=kill_barrier(command,root/'probe','puct_cpu_case_commit_barrier',env={'PUCT_CPU_BLOCK_AFTER_CASE':'1'})
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:before=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall()
        self.assertEqual(len(before),1)
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'resume.log').write_text(result.stdout+result.stderr);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall();self.assertEqual(len(after),2);self.assertEqual(after[0],before[0])
            with closing(sqlite3.connect(root/'validated-work.sqlite3')) as backup:db.backup(backup)
            idx,body,digest=after[0];forged=json.loads(body);ms=next(iter(forged['measurements']));name=next(iter(forged['measurements'][ms]));forged['measurements'][ms][name]['reference_regret']=999
            with db:db.execute('UPDATE cases SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),idx))
        bad=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'refusal.log').write_text(bad.stdout+bad.stderr);self.assertNotEqual(bad.returncode,0)
        self.assertIn('regret',bad.stdout.lower()+bad.stderr.lower())
        EVIDENCE['case_actual_recovery']=dict(actual_process_kill=True,committed_cases_retained=1,complete_fixture_after_resume=2,rehashed_regret_refused=True,valid_backup_and_marker_retained=True,barrier=marker)

    def test_12_actual_arena_kill_576_literal_replays_and_finished_stability(self):
        root=self.fresh('arena-kill');run=root/'run';run.mkdir();output=root/'result.json';plan=load(common.PLAN)
        with closing(sqlite3.connect(ROOT/'data/puct-cpu-benchmark-v1/work.sqlite3')) as source,closing(sqlite3.connect(run/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM games').fetchone()[0],576)
            with db:db.execute('DELETE FROM games WHERE idx>=574');db.execute("DELETE FROM meta WHERE key='puct_cpu_benchmark_complete'")
        command=[sys.executable,str(ROOT/'tools/benchmark-puct-cpu.py'),'run','--run',str(run),'--output',str(output)]
        marker=kill_barrier(command,root/'probe','puct_cpu_game_commit_barrier',env={'PUCT_CPU_BLOCK_AFTER_GAME':'575'})
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall()
        self.assertEqual(len(prefix),575)
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'resume.log').write_text(result.stdout+result.stderr);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT idx,body,hash FROM games ORDER BY idx').fetchall();self.assertEqual(after[:575],prefix);self.assertEqual(len(after),576)
            for idx,body,digest in after:
                game=json.loads(body);self.assertEqual(config.digest(game),digest);q=normalize_literal(game['start'])
                for action in game['moves']:q=literal.apply(q,action).position
                self.assertEqual(json_position(q),game['final'])
            with closing(sqlite3.connect(root/'validated-work.sqlite3')) as backup:db.backup(backup)
        stable=snapshot(run);published=load(output);again=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);self.assertEqual(again.returncode,0,again.stdout+again.stderr);self.assertEqual(snapshot(run),stable);self.assertEqual(load(output),published)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            idx,body,digest=after[0];forged=json.loads(body);forged['candidate_final_margin']+=2
            with db:db.execute('UPDATE games SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),idx))
        refusal=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'refusal.log').write_text(refusal.stdout+refusal.stderr);self.assertNotEqual(refusal.returncode,0)
        EVIDENCE['arena_actual_recovery']=dict(actual_process_kill=True,committed_games_retained=575,complete_after_resume=576,all_games_literal_replayed=True,finished_bytes_mtimes_stable=True,rehashed_final_margin_refused=True,raw_correct_journal_and_marker_retained=True,barrier=marker)

    def test_13_actual_speed_first_unit_kill_resume_1410_and_counter_refusal(self):
        root=self.fresh('speed-kill');run=root/'run';output=root/'result.json'
        custom="import sys,json,time,importlib.util\nspec=importlib.util.spec_from_file_location('speed','tools/puct-cpu-speed.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)\ndef callback(v):\n s.event(v)\n if v['event']=='puct_cpu_speed_unit' and v['completed']==1:\n  print(json.dumps(dict(event='speed_kill_barrier')),flush=True)\n  while True:time.sleep(.05)\ns.run(sys.argv[1],sys.argv[2],on_event=callback)"
        marker=kill_barrier([sys.executable,'-c',custom,str(run),str(output)],root/'probe','speed_kill_barrier')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:prefix=db.execute('SELECT * FROM units').fetchall()
        self.assertEqual(len(prefix),1)
        command=[sys.executable,str(ROOT/'tools/puct-cpu-speed.py'),'--run',str(run),'--output',str(output)]
        resumed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'resume.log').write_text(resumed.stdout+resumed.stderr);self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
        report=load(output);self.assertTrue(report['complete'] and report['fixed_tree_semantics_equal']);self.assertEqual(report['units'],1410)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM units').fetchone()[0],1410)
            self.assertEqual(db.execute('SELECT * FROM units WHERE kind=? AND profile=? AND batch=? AND block=? AND root=?',prefix[0][:5]).fetchone(),prefix[0])
            with closing(sqlite3.connect(root/'validated-work.sqlite3')) as backup:db.backup(backup)
        stable=snapshot(run);again=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);self.assertEqual(again.returncode,0,again.stdout+again.stderr);self.assertEqual(snapshot(run),stable);self.assertEqual(load(output),report)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            kind,profile,batch,block,index,body,digest=db.execute("SELECT * FROM units WHERE kind='cold' AND profile='warm_single' AND batch=1 AND block=0").fetchone();forged=json.loads(body);forged['counts']['feature_simulations']=0;forged['counts']['feature_deposits']=0
            with db:db.execute('UPDATE units SET body=?,hash=? WHERE kind=? AND profile=? AND batch=? AND block=? AND root=?',(config.canonical(forged),config.digest(forged),kind,profile,batch,block,index))
        bad=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(root/'refusal.log').write_text(bad.stdout+bad.stderr);self.assertNotEqual(bad.returncode,0)
        EVIDENCE['speed_actual_recovery']=dict(actual_process_kill=True,committed_unit_retained=True,complete_units_after_resume=1410,fixed_tree_semantics_equal=True,finished_bytes_mtimes_stable=True,rehashed_impossible_feature_work_refused=True,valid_backup_and_marker_retained=True,fixture_timings_not_production_evidence=True,barrier=marker)

