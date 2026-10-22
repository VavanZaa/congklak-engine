"""Single-pass double inference, bounded raw LRU and interrupted-work checks."""
from contextlib import closing
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,puct_cpu_inference as cpu
from congklak.native import NativeError,flatten
from congklak.neural_relay import NeuralNative
from congklak.phase6_inference import BatchEvaluator,InferenceStopped
from congklak.puct import forest,search,CancellationToken
import reference as literal
from stage5_deeper_audit_acceptance import literal_features,value_and_policy

MODELS=('data/training-phase6-warm-v1/model.json','data/training-phase6-update-v1/model.json')
EVIDENCE={}
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def position(pits,actor=0,store0=0):
    active=sum(map(sum,pits));return dict(pits=pits,stores=[store0,98-active-store0],controller=actor)
def interpreted(p):return literal.Position(7,tuple(flatten(p)),p['controller'])
def scientific(report):
    return {k:report[k] for k in ('chosen','visits','policy','root_priors','estimated_value_player0','action_mean_value_player0','root_expanded','completed_simulations','nodes','stop_reason','proof_status','unvisited_root_q_player0')}
def development():
    with closing(sqlite3.connect((ROOT/'data/deeper-9000-v2/work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        rows=[dict(state=json.loads(p),stratum=s) for p,s in db.execute("SELECT json_extract(body,'$.state'),json_extract(body,'$.stratum') FROM roots WHERE json_extract(body,'$.split')='development' ORDER BY state")]
    if len(rows)!=1000:raise AssertionError('Expected exactly1000 development roots')
    return rows

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=development();cls.opening=position([[7]*7,[7]*7])

    def test_01_all_prior1192_and878_inputs_preserved(self):
        counts={}
        for name,wanted in (('build/phase6-acceptance.json',1192),('build/stage5-deeper-acceptance.json',878)):
            body=json.loads((ROOT/name).read_text());self.assertTrue(body['passed']);self.assertEqual(len(body['inputs']),wanted)
            for path,digest in body['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
            counts[name]=wanted
        EVIDENCE['prior_pins']=counts

    def test_02_structured_abi_layout_and_no_silent_fallback(self):
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],backend='scalar') as e:
            self.assertEqual(e.backend,'scalar-double-packed');self.assertEqual(e.native.lib.cc_puct_cpu_abi().decode(),cpu.ABI)
            self.assertEqual(e.native.lib.cc_puct_result_size(),C.sizeof(cpu.PredictionResult));self.assertEqual(e.native.lib.cc_puct_features_size(),C.sizeof(cpu.FeatureResult))
            self.assertEqual(cpu.PredictionResult.value_player0.offset,40);self.assertEqual(cpu.PredictionResult.policy_logits.offset,48)
        original=cpu.NeuralNative
        def bad_abi(*args,**kwargs):
            instance=original(*args,**kwargs);instance.lib.cc_puct_cpu_abi=lambda:b'unsupported-abi';return instance
        with mock.patch.object(cpu,'NeuralNative',side_effect=bad_abi),self.assertRaisesRegex(NativeError,'ABI/layout'):
            cpu.CpuBatchEvaluator(ROOT/MODELS[0])
        with self.assertRaises(ValueError):cpu.CpuBatchEvaluator(ROOT/MODELS[0],backend='imaginary')
        model=cpu.Model(ROOT/MODELS[0]);original_loads=cpu.json.loads
        for field,value in (('inputs',{}),('flags',['/O2','/fp:fast']),('dispatch','unconditional AVX2')):
            forged=json.loads((ROOT/'build/puct-cpu-build.json').read_text());forged[field]=value
            with mock.patch.object(cpu.json,'loads',return_value=forged),self.assertRaises(NativeError):cpu.CpuBatchEvaluator(model)
        forged=json.loads((ROOT/'build/puct-cpu-build.json').read_text());forged['extra']='unrecognized'
        with mock.patch.object(cpu.json,'loads',return_value=forged),self.assertRaises(NativeError):cpu.CpuBatchEvaluator(model)
        EVIDENCE['abi']=dict(version=cpu.ABI,result_bytes=C.sizeof(cpu.PredictionResult),feature_bytes=C.sizeof(cpu.FeatureResult),wrong_abi_refused=True,scalar_explicit=True,missing_build_source_set_refused=True,strict_build_flags_and_dispatch_required=True,unknown_build_fields_refused=True)

    def test_03_literal127_and_both_models1000_single_pass(self):
        records=[]
        for path in MODELS:
            for backend in ('scalar','avx2'):
                with cpu.CpuBatchEvaluator(ROOT/path,backend=backend) as e,NeuralNative(ROOT/path,backend=backend,mode='value-none') as old:
                    max_error=0.;features_total=deposits_total=0
                    for start in range(0,1000,32):
                        states=[r['state'] for r in self.rows[start:start+32]];outputs=e.evaluate_many(states)
                        for p,value in zip(states,outputs):
                            q=interpreted(p);x,d,r=literal_features(q,127);diagnostic=e.feature_vector(p)
                            self.assertEqual(diagnostic['features'],list(x));self.assertEqual(value['feature_simulations'],sum(bool(n) for row in p['pits'] for n in row))
                            deposits=sum(literal.apply(literal.Position(7,q.counts,actor),pit).deposits for actor in (0,1) for pit in range(7) if p['pits'][actor][pit])
                            self.assertEqual(value['feature_deposits'],deposits);self.assertEqual(diagnostic['feature_deposits'],deposits)
                            exact,_,logits=value_and_policy(q,e.model.envelope['model']);frozen=old.evaluate(p)
                            max_error=max(max_error,abs(value['value_player0']-exact),max(abs(a-b) for a,b in zip(value['policy_logits'],logits)))
                            self.assertLessEqual(max_error,1e-10);self.assertEqual(value['value_player0'],frozen['model_value_player0']);self.assertEqual(value['policy_logits'],frozen['policy_logits'])
                            features_total+=value['feature_simulations'];deposits_total+=deposits
                    stats=e.metrics_snapshot();self.assertEqual(stats['native_actual_evals'],1000);self.assertEqual(stats['native_calls'],1000)
                    self.assertEqual(stats['feature_simulations'],features_total);self.assertEqual(stats['feature_deposits'],deposits_total)
                    records.append(dict(model=path,backend=e.backend,roots=1000,max_literal_error=max_error,single_feature_pass=True,feature_simulations=features_total,feature_deposits=deposits_total))
        EVIDENCE['literal_inference']=records

    def test_04_fixed128_simulation_forest_noise_and_actor_paths(self):
        roots=[next(r['state'] for r in self.rows if r['stratum'].endswith('/'+kind)) for kind in ('ordinary','capture','bonus','pass')]+[self.rows[i]['state'] for i in (127,314,628,999)]
        for path in MODELS:
            with BatchEvaluator(ROOT/path,'native',max_batch=8) as old:
                reference=forest(roots,old,simulations=128,rng=[random.Random(810+i) for i in range(8)],root_noise=dict(alpha=.3,fraction=.25))
                for capacity in (0,4096):
                    with cpu.CpuBatchEvaluator(ROOT/path,cache_capacity=capacity,max_batch=8) as e:
                        e.begin_decision();actual=forest(roots,e,simulations=128,rng=[random.Random(810+i) for i in range(8)],root_noise=dict(alpha=.3,fraction=.25))
                        self.assertEqual([scientific(r) for r in actual],[scientific(r) for r in reference])
                        if capacity==0:self.assertEqual(sum(r['work']['feature_deposits'] for r in actual)*2,sum(r['work']['feature_deposits'] for r in reference))
                        stats=e.metrics_snapshot();self.assertEqual(stats['evaluations_requested'],stats['cache_hits']+stats['cache_misses']);self.assertEqual(stats['native_actual_evals'],stats['cache_misses'])
        EVIDENCE['fixed_simulation']=dict(models=2,capacity_modes=[0,4096],roots=8,simulations=128,noise_rng_and_scientific_reports_equal=True)

    def test_05_full_raw_key_pass_store_model_and_copy_isolation(self):
        p=position([[0]*7,[2,0,0,0,0,0,1]],actor=0,store0=40);alias={**p,'controller':1};offset={**p,'stores':[41,54]}
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],cache_capacity=4096) as e,cpu.CpuBatchEvaluator(ROOT/MODELS[1],cache_capacity=4096) as other:
            a=e.evaluate_many([p,alias,offset]);self.assertEqual(e.metrics_snapshot()['cache_misses'],3)
            self.assertEqual(a[0]['value_player0'],a[1]['value_player0']);self.assertEqual(a[0]['policy_logits'],a[1]['policy_logits']);self.assertEqual(a[2]['value_player0'],a[0]['value_player0']+2)
            a[0]['policy_logits'][0]=1e30;a[0]['value_player0']=1e30
            hit=e.evaluate_many([p])[0];self.assertTrue(hit['cache_hit']);self.assertNotEqual(hit['policy_logits'][0],1e30);self.assertEqual(hit['feature_simulations'],0);self.assertEqual(hit['native_actual_evals'],0)
            other.evaluate_many([p]);self.assertEqual(other.metrics_snapshot()['cache_misses'],1);self.assertEqual(other.metrics_snapshot()['cache_hits'],0)
            terminal=position([[0]*7,[0]*7],store0=51);before=e.metrics_snapshot();v=e.evaluate_many([terminal])[0];after=e.metrics_snapshot()
            self.assertEqual(v['value_player0'],4);self.assertEqual(v['policy_logits'],[0.]*7);self.assertEqual(after['terminal_returns']-before['terminal_returns'],1);self.assertEqual(after['native_calls'],before['native_calls'])
        EVIDENCE['cache_isolation']=dict(full_input_stores_and_actor=True,forced_pass_alias_misses=True,model_private=True,immutable_copy_out=True,terminal_zero_work=True)

    def test_06_lru_bound_collisions_and_every_decision_reset(self):
        roots=[r['state'] for r in self.rows[:3]]
        class CollidingBytes(bytes):
            def __hash__(self):return 1
        original=cpu._raw_key
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],cache_capacity=2) as e,mock.patch.object(cpu,'_raw_key',side_effect=lambda p:CollidingBytes(original(p))):
            e.evaluate_many(roots[:2]);e.evaluate_many(roots[:1]);e.evaluate_many(roots[2:]);s=e.metrics_snapshot();self.assertEqual(s['cache_current_entries'],2);self.assertEqual(s['cache_evictions'],1)
            hit=e.evaluate_many(roots[:1])[0];self.assertTrue(hit['cache_hit']);miss=e.evaluate_many(roots[1:2])[0];self.assertFalse(miss['cache_hit'])
            before=e.metrics_snapshot();e.begin_decision();after=e.metrics_snapshot();self.assertEqual(after['cache_current_entries'],0);self.assertEqual(after['cache_resets'],before['cache_resets']+1)
            for key in cpu.COUNTERS:
                if key!='cache_resets':self.assertEqual(after[key],before[key])
        rng=random.Random(819);states=[];keys=set()
        while len(states)<4097:
            counts=[0]*14
            for _ in range(98):counts[rng.randrange(14)]+=1
            p=position([counts[:7],counts[7:]],actor=rng.randrange(2));key=original(p)
            if key not in keys:keys.add(key);states.append(p)
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],cache_capacity=4096) as e:
            for i in range(0,len(states),32):e.evaluate_many(states[i:i+32])
            self.assertEqual(e.metrics_snapshot()['cache_current_entries'],4096);self.assertEqual(e.metrics_snapshot()['cache_evictions'],1)
        EVIDENCE['lru']=dict(actual_capacity=4096,distinct_insertions=4097,exact_equality_on_hash_collision=True,reset_only_changes_cache_occupancy_and_reset_counter=True)

    def test_07_stopped_forward_and_hit_partial_work_truthful(self):
        def cancellation(at):
            calls=0
            def poll():
                nonlocal calls;calls+=1;return calls>=at
            return poll
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],cache_capacity=4096) as e:
            with self.assertRaises(InferenceStopped) as caught:e.evaluate_many([self.opening],cancel=cancellation(4))
            s=e.metrics_snapshot();self.assertEqual(s['native_actual_evals'],1);self.assertEqual(s['predict_successes'],1);self.assertGreater(s['feature_deposits'],0);self.assertEqual(s['cache_current_entries'],0)
            self.assertTrue(caught.exception.work['per_position'][0]['evaluated'])
            e.evaluate_many([self.opening]);before=e.metrics_snapshot()
            with self.assertRaises(InferenceStopped) as caught:e.evaluate_many([self.opening],cancel=cancellation(4))
            after=e.metrics_snapshot();self.assertEqual(after['cache_hits']-before['cache_hits'],1);self.assertEqual(after['native_actual_evals'],before['native_actual_evals'])
            self.assertFalse(caught.exception.work['per_position'][0]['evaluated']);self.assertEqual(caught.exception.work['feature_deposits'],0)
            token=CancellationToken();token.cancel();before=e.metrics_snapshot()
            with self.assertRaises(InferenceStopped):e.evaluate_many([self.opening],cancel=token)
            self.assertEqual(e.metrics_snapshot(),before)
        EVIDENCE['discarded_calls']=dict(completed_dense_forward_charged_when_post_call_cancelled=True,discarded_hit_has_zero_physical_forward=True,initial_cancel_zero_work=True)

    def test_08_actual_native_partial_deadline_and_no_backup(self):
        long=max(self.rows,key=lambda row:sum(literal_features(interpreted(row['state']))[0][44+i*6] for i in range(14)))['state']
        partial=[]
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0]) as e:
            cells,_=e.native._array(flatten(long),7,98)
            for duration in (3000,5000,8000,12000,20000,40000,80000):
                for _ in range(8):
                    e.native.lib.cc_puct_begin(e._control.handle,duration);out=cpu.PredictionResult()
                    code=e.native.lib.cc_puct_predict(cells,long['controller'],e.native.handle,e._control.handle,C.byref(out))
                    if code==1 and out.feature_simulations and out.feature_deposits and not out.neural_evaluations:
                        partial.append(dict(remaining_ns=duration,feature_simulations=int(out.feature_simulations),feature_deposits=int(out.feature_deposits),physical_forwards=0,reason=out.reason.decode()))
                        break
                if partial:break
            self.assertTrue(partial,'No actual interrupted feature work observed')
            self.assertEqual(partial[0]['reason'],'time_budget')
            token=CancellationToken()
            class StoppingDelegate:
                supports_cancel=True;max_batch=32;native=e.native
                def evaluate_many(self,rows,**kwargs):
                    values=e.evaluate_many(rows,**kwargs);token.cancel();return values
            report=search(long,StoppingDelegate(),simulations=128,cancel=token)
            self.assertEqual(report['completed_simulations'],0);self.assertEqual(report['visits'],[0]*7);self.assertFalse(report['root_expanded']);self.assertFalse(report['usable_decision'])
        EVIDENCE['native_deadline']=dict(actual_partial_feature_stop=partial[0],unfinished_root_has_zero_visits=True)

    def test_09_lock_cancel_closed_and_batch_bounds(self):
        with cpu.CpuBatchEvaluator(ROOT/MODELS[0],max_batch=2) as e:
            with self.assertRaises(config.InputError):e.evaluate_many([self.opening]*3)
            with self.assertRaises(config.InputError):e.evaluate_many([])
            with self.assertRaises(ValueError):e.evaluate_many([self.opening],deadline=float('nan'))
            held=threading.Event();release=threading.Event()
            def own_lock():
                with e.lock:held.set();release.wait(timeout=2)
            worker=threading.Thread(target=own_lock);worker.start();self.assertTrue(held.wait(timeout=1))
            try:
                with self.assertRaises(InferenceStopped):e.evaluate_many([self.opening],deadline=time.perf_counter()+.01)
            finally:release.set();worker.join(timeout=2)
            self.assertGreaterEqual(e.last_metrics['whole_call_ms'],5)
        with self.assertRaises(NativeError):e.evaluate_many([self.opening])
        EVIDENCE['bounds']=dict(max_batch_refused=True,closed_refused=True,queue_wait_cancellable=True)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    files=['native/neural_puct_cpu.cpp','congklak/puct_cpu_inference.py','tools/build-puct-cpu.ps1','tools/build-puct-cpu.cmd','tests/puct_cpu_inference_acceptance.py','build/puct-cpu-build.json','build/congklak_puct_cpu.dll',*MODELS,'tests/reference.py','tests/stage5_deeper_audit_acceptance.py','build/phase6-acceptance.json']
    payload=dict(schema='puct-cpu-inference-acceptance-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=unittest.defaultTestLoader.getTestCaseNames(Tests),evidence=EVIDENCE,
                 inputs={p:sha(ROOT/p) for p in files},torch_imported='torch' in sys.modules,production_timing=False)
    (ROOT/'build/puct-cpu-inference-acceptance.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=payload['passed'],tests=result.testsRun,errors=payload['errors'],failures=payload['failures'],torch_imported=payload['torch_imported'])))
    raise SystemExit(0 if result.wasSuccessful() else 1)
