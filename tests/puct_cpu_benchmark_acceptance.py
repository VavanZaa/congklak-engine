"""CPU-only real-profile deadline, telemetry and durable-prefix contracts."""
from __future__ import annotations
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,phase6_common as common
from congklak.phase6_inference import BatchEvaluator
from congklak.phase6_inference import InferenceStopped
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from phase6_puct_acceptance import position
import phase6_benchmark_acceptance as fixture
spec=importlib.util.spec_from_file_location('puct_cpu_benchmark_test',ROOT/'tools/benchmark-puct-cpu.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

def implementation():
    body=dict(schema='puct-cpu-implementation-v1',roles=bench.ROLES,sources={p:common.sha(ROOT/p) for p in bench.ROLES.values()},science_inputs={p:common.sha(ROOT/p) for p in bench.SCIENCE})
    return {**body,'sha256':config.digest(body)}

class Admission(unittest.TestCase):
    def temp(self):
        parent=ROOT/'build/puct-cpu-benchmark-fixtures';parent.mkdir(exist_ok=True);temporary=tempfile.TemporaryDirectory(prefix='guard-',dir=parent)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(ROOT.resolve()));return temporary
    def test_actual1192_prior_pins_and30_checkpoint_witnesses_no_torch(self):
        self.assertNotIn('torch',sys.modules);plan=bench.read_plan()
        with self.temp() as folder:
            p=Path(folder)/'implementation.json';p.write_text(config.canonical(implementation()),encoding='utf-8');body,witness=bench.admit(plan,p)
            self.assertEqual(witness['prior_accepted_inputs'],1192)
            self.assertEqual([len(witness['model_witnesses'][name]['checkpoints']) for name in ('warm','updated')],[20,10])
            self.assertEqual(witness['model_witnesses']['updated']['selected_step'],500)
        self.assertNotIn('torch',sys.modules)
    def test_checksum_valid_missing_science_and_source_drift_refused(self):
        with self.temp() as folder:
            p=Path(folder)/'implementation.json';plan=bench.read_plan()
            for missing in bench.SCIENCE:
                body=implementation();del body['science_inputs'][missing];body['sha256']=config.digest({k:v for k,v in body.items() if k!='sha256'});p.write_text(config.canonical(body),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'mandatory CPU science'):bench.admit(plan,p)
            body=implementation();body['sources']['congklak/puct_cpu_inference.py']='0'*64;body['sha256']=config.digest({k:v for k,v in body.items() if k!='sha256'});p.write_text(config.canonical(body),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Pinned CPU experiment input'):bench.admit(plan,p)
    def test_balanced576unit_order_and_fixed_geometry(self):
        plan=bench.read_plan();order=bench.arena_order(plan)
        self.assertEqual(len(order),576);self.assertEqual(order,plan['arena_work_order']);self.assertEqual(len({tuple(row.values()) for row in order}),576)
        self.assertEqual([r['seat'] for r in order[:2]],[0,1]);self.assertEqual([r['seat'] for r in order[24:26]],[1,0])
        self.assertEqual(order[24]['comparison'],plan['arena_comparisons'][1]['id'])

class Decisions(unittest.TestCase):
    def setUp(self):
        self.p=position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]);self.plan=fixture.protocol(self.p);self.plan['engines']=bench.ENGINES
    def evaluator(self,name,stack):
        model=common.WARM if name.startswith('warm_') else common.UPDATE
        if name.endswith('_legacy'):return bench.LegacyTelemetry(stack.enter_context(BatchEvaluator(model,'native')))
        return stack.enter_context(CpuBatchEvaluator(model,cache_capacity=4096 if name.endswith('_cache') else 0))
    def test_actual_all6_profiles_whole_clock_and_physical_counters(self):
        with ExitStack() as stack:
            for name in bench.CANDIDATES:
                obj=self.evaluator(name,stack);result=bench.decide(name,('puct',obj),self.p,30,self.plan);bench.validate_envelope(result,self.p,30,self.plan,name)
                decision=result['decision'];c=result['evaluator'];self.assertTrue(decision['usable_decision']);self.assertLess(c['granted_search_ms'],30)
                self.assertEqual(c['counts']['cache_resets'],1);self.assertGreater(c['counts']['native_actual_evals'],0)
                self.assertEqual(c['counts']['native_calls'],c['counts']['native_actual_evals']*(2 if name.endswith('_legacy') else 1))
                self.assertEqual(decision['overshoot_ms'],max(0,decision['wall_ms']-30))
        self.assertNotIn('torch',sys.modules)
    def test_reset_exhausts_deadline_zero_never_unlimited_search(self):
        with BatchEvaluator(common.WARM,'native') as raw:
            class Slow(bench.LegacyTelemetry):
                def reset_cache(self):super().reset_cache();time.sleep(.012)
            obj=Slow(raw)
            with patch.object(bench,'puct_search',side_effect=AssertionError('zero deadline must not call search')):
                result=bench.decide('warm_legacy',('puct',obj),self.p,5,self.plan)
            self.assertFalse(result['decision']['usable_decision']);self.assertEqual(result['evaluator']['granted_search_ms'],0)
            self.assertFalse(result['decision']['report']['root_expanded']);self.assertEqual(result['evaluator']['counts']['native_actual_evals'],0)
    def test_cache_prefill_cleared_inside_decision_and_fullraw_bound(self):
        with CpuBatchEvaluator(common.WARM,cache_capacity=4096) as obj:
            obj.evaluate_many([self.p]);obj.evaluate_many([self.p]);before=obj.snapshot_counters();self.assertEqual(before['cache_hits'],1)
            result=bench.decide('warm_cache',('puct',obj),self.p,15,self.plan);c=result['evaluator']
            self.assertGreater(c['cache_occupancy_before'],0);self.assertEqual(c['cache_occupancy_after_reset'],0)
            self.assertEqual(c['counts']['cache_resets'],1);self.assertGreater(c['counts']['cache_misses'],0)
    def test_checksum_valid_telemetry_and_deadline_corruption_refused(self):
        with CpuBatchEvaluator(common.WARM,cache_capacity=4096) as obj:result=bench.decide('warm_cache',('puct',obj),self.p,20,self.plan)
        for mutate in (lambda t:t['counts'].__setitem__('cache_hits',t['counts']['cache_hits']+1),lambda t:t.__setitem__('cache_occupancy_after_reset',1),lambda t:t.__setitem__('granted_search_ms',21),lambda t:t.__setitem__('physical_forward_executions',t['physical_forward_executions']+1),lambda t:t.__setitem__('logical_row_evaluations',t['logical_row_evaluations']+1)):
            value=copy.deepcopy(result);mutate(value['evaluator']);config.digest(value)
            with self.assertRaises(ValueError):bench.validate_envelope(value,self.p,20,self.plan,'warm_cache')
    def test_baselines_have_no_neural_cache_telemetry(self):
        with NeuralNative(ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json',mode='value-none') as relay:
            for name,kind,obj in (('H_v1','h',Native()),('H8_relay_CPU','relay',relay)):
                value=bench.decide(name,(kind,obj),self.p,30,self.plan);bench.validate_envelope(value,self.p,30,self.plan,name);self.assertIsNone(value['evaluator'])
                altered=copy.deepcopy(value);altered['evaluator']={}
                with self.assertRaises(ValueError):bench.validate_envelope(altered,self.p,30,self.plan,name)
    def test_legacy_cancelled_terminal_and_reference_do_not_invent_work(self):
        terminal=dict(pits=[[0]*7,[0]*7],stores=[50,48],controller=0)
        with BatchEvaluator(common.WARM,'native') as raw:
            obj=bench.LegacyTelemetry(raw)
            with self.assertRaises(InferenceStopped):obj.evaluate_many([terminal],cancel=lambda:True)
            before=obj.snapshot_counters();self.assertEqual(before['terminal_returns'],0)
            obj.uninstrumented_reference(self.p);self.assertEqual(before,obj.snapshot_counters())
            result=obj.evaluate_many([terminal]);self.assertEqual(result[0]['value_player0'],2)
            self.assertEqual(obj.snapshot_counters()['terminal_returns'],1);self.assertEqual(obj.snapshot_counters()['native_actual_evals'],0)
    def test_actual_case_reference_semantics_and_corruption(self):
        h=bench.decide('H_v1',('h',Native()),self.p,5,self.plan)
        with BatchEvaluator(common.WARM,'native') as raw:puct=bench.decide('warm_legacy',('puct',bench.LegacyTelemetry(raw)),self.p,5,self.plan)
        q={r['pit']:r['value_player0'] for r in self.plan['cases'][0]['reference']['action_scores']};body=dict(index=0,state_key='fixture',cohort='fixture',stratum='fixture/ordinary',reference=self.plan['cases'][0]['reference'],orders={},measurements={},cpu_verified=bench.prior.verified(self.plan))
        for bi,ms in enumerate(self.plan['budgets_ms']):
            body['orders'][str(ms)]=bench.prior.expected_order(0,bi,self.plan);body['measurements'][str(ms)]={}
            for name in bench.ENGINES:
                value=copy.deepcopy(h if name in ('H_v1','H8_relay_CPU') else puct);d=value['decision'];d.update(requested_ms=ms,wall_ms=max(ms,d['wall_ms']),overshoot_ms=max(0,d['wall_ms']-ms))
                if name=='H8_relay_CPU':d['kind']='relay';d['report']['proof_status']='finite_horizon_neural_estimate'
                if value['evaluator']:
                    t=value['evaluator'];t.update(profile=name,requested_ms=ms,cache_capacity=4096 if name.endswith('_cache') else 0,feature_passes_per_native_eval=2 if name.endswith('_legacy') else 1)
                    # The fixture carries no features into a synthetic cache
                    # profile; real profiles are independently checked above.
                    if not name.endswith('_legacy'):
                        t['counts']['native_calls']=t['counts']['native_actual_evals'];t['counts']['feature_simulations']=0;t['counts']['feature_deposits']=0;d['report']['work']['feature_simulations']=0;d['report']['work']['feature_deposits']=0;d['report']['work']['total_deposits']=d['report']['work']['deposits']
                choice=d['chosen'];body['measurements'][str(ms)][name]=dict(decision=value,reference_optimal=choice in body['reference']['best_moves'] if choice is not None else None,reference_regret=body['reference']['target_player0']-q[choice] if choice is not None else None)
        bench.validate_case(body,0,self.plan);altered=copy.deepcopy(body);altered['orders']['5'].reverse();config.digest(altered)
        with self.assertRaises(ValueError):bench.validate_case(altered,0,self.plan)
        altered=copy.deepcopy(body);altered['measurements']['5']['H_v1']['reference_regret']+=1
        with self.assertRaises(ValueError):bench.validate_case(altered,0,self.plan)
    def test_family_bootstrap_predeclared_and_unfinished_pair_excluded(self):
        self.assertEqual(bench.prior.interval([1,2,3,4],self.plan),bench.prior.interval([1,2,3,4],self.plan))
        self.assertIsNone(bench.prior.interval([],self.plan))
        plan=bench.read_plan()
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx INTEGER,body TEXT); CREATE TABLE games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT);')
            comparison=plan['arena_comparisons'][0]['id']
            for seat,status,margin,score in ((0,'terminal',10,1),(1,'unfinished',None,None)):
                game=dict(pair=0,status=status,candidate_final_margin=margin,candidate_match_score=score,fallbacks={})
                db.execute('INSERT INTO games VALUES(?,?,?,?)',(comparison,0,seat,config.canonical(game)))
            result=bench.summarize(db,{},plan);arena=result['arenas'][comparison]
            self.assertEqual(arena['complete_pairs'],0);self.assertIsNone(arena['primary_paired_mean_final_margin_seeds']);self.assertFalse(arena['primary_strength_gate_passed']);self.assertEqual(arena['unfinished_games'],1)
    def test_checksum_valid_case_hole_and_arena_orphan_refused(self):
        plan=bench.read_plan();body={'checksum_valid':'unregistered unit'};text=config.canonical(body)
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx INTEGER,body TEXT,hash TEXT); CREATE TABLE games(idx INTEGER,comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT);')
            db.execute('INSERT INTO cases VALUES(?,?,?)',(1,text,config.digest(body)))
            with self.assertRaisesRegex(ValueError,'noncontiguous CPU case prefix'):bench.validate_saved_units(db,plan,Native())
            db.execute('DELETE FROM cases');comparison=plan['arena_comparisons'][0]['id'];db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(0,comparison,1,0,text,config.digest(body)))
            with self.assertRaisesRegex(ValueError,'orphan CPU arena prefix'):bench.validate_saved_units(db,plan,Native())

if __name__=='__main__':unittest.main(verbosity=2)
