"""Owner checks for fixed-stage clocks, physical work and durable admission."""
from __future__ import annotations
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,value_pilot_benchmark as bench
from congklak.native import Native,NativeError
from congklak.neural_relay import Model,NeuralNative
from congklak.puct_cpu_inference import CpuBatchEvaluator
from phase6_puct_acceptance import position
import phase6_benchmark_acceptance as fixture


def protocol(p):
    plan=fixture.protocol(p)
    plan.update(stage='posttraining',fixture=True,engines=bench.ENGINES,profiles={name:dict(model='control',c_puct=.25,cache_capacity=0) for name in bench.CANDIDATES},models=dict(control='data/quality-train-v1/model.json'))
    plan['arena_comparisons']=[dict(id=name+'_vs_'+base,candidate=name,opponent=base,pairs=1) for name in bench.CANDIDATES for base in bench.BASELINES]
    plan['arena_work_order']=bench.arena_order(plan);plan['release']='Synthetic fixture for independent arithmetic; no strength claim'
    return plan


class Tests(unittest.TestCase):
    def setUp(self):
        self.p=position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]])
        self.plan=protocol(self.p)

    def envelope(self,ms=15):
        with CpuBatchEvaluator(bench.CONTROL,cache_capacity=0) as obj:
            return bench.decide('policy_control',('puct',obj),self.p,ms,self.plan)

    def case(self):
        body=dict(index=0,state_key='fixture',cohort='fixture',stratum='fixture/ordinary',reference=self.plan['cases'][0]['reference'],orders={},measurements={},cpu_verified=bench.prior.verified(self.plan))
        q={entry['pit']:entry['value_player0'] for entry in body['reference']['action_scores']}
        with ExitStack() as stack:
            relay=stack.enter_context(NeuralNative(ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json',mode='value-none'))
            obj=stack.enter_context(CpuBatchEvaluator(bench.CONTROL,cache_capacity=0))
            engines={'H_v1':('h',Native()),'H8_relay_CPU':('relay',relay),**{name:('puct',obj) for name in bench.CANDIDATES}}
            for bi,ms in enumerate(self.plan['budgets_ms']):
                names=bench.prior.expected_order(0,bi,self.plan);body['orders'][str(ms)]=names;body['measurements'][str(ms)]={}
                for name in names:
                    answer=bench.decide(name,engines[name],self.p,ms,self.plan);choice=answer['decision']['chosen']
                    body['measurements'][str(ms)][name]=dict(decision=answer,reference_optimal=choice in body['reference']['best_moves'] if choice is not None else None,reference_regret=body['reference']['target_player0']-q[choice] if choice is not None else None)
        return body

    def test_01_actual_prior2426_and_control_model_are_current(self):
        value=bench.baseline();self.assertEqual(len(value['inputs']),2426)
        self.assertEqual(bench.sha(bench.CONTROL),bench.MODEL_SHA);self.assertNotIn('torch',sys.modules)

    def test_02_exact_frozen616_measurements_and144_games(self):
        pre=bench.read_plan('pretraining');post=bench.read_plan('posttraining')
        self.assertEqual(len(pre['cases']),44);self.assertEqual(pre['cases'],post['cases']);self.assertEqual(pre['arena_starts'],post['arena_starts'])
        self.assertEqual(len(pre['arena_starts']),12);self.assertEqual(len(pre['arena_work_order']),48);self.assertEqual(len(post['arena_work_order']),96)
        self.assertEqual(44*2*(len(pre['engines'])+len(post['engines'])),616)
        self.assertEqual(list(pre['profiles']),['policy_control']);self.assertEqual(set(post['profiles']),set(bench.CANDIDATES))
        self.assertEqual(pre['budgets_ms'],[25,100]);self.assertEqual(pre['arena_ms'],25)
        self.assertEqual([row['seat'] for row in post['arena_work_order'][:2]],[0,1]);self.assertEqual([row['seat'] for row in post['arena_work_order'][8:10]],[1,0])
        for kind in ('ordinary','capture','bonus'):
            self.assertEqual(sum(row['stratum'].endswith('/'+kind) for row in pre['arena_starts']),4)
        self.assertTrue(pre['novelty'].startswith('All12'));self.assertFalse(pre['sealed_final_test_consumed'])

    def test_03_frozen_policy_and_changed_head_contract_refuses_mutations(self):
        from types import SimpleNamespace
        source=Model(bench.CONTROL);candidate=SimpleNamespace(weights=copy.deepcopy(source.weights))
        with self.assertRaisesRegex(ValueError,'did not change'):bench.verify_frozen_policy(source,candidate)
        candidate.weights['layers.4.bias'][0]+=.001
        self.assertEqual(bench.verify_frozen_policy(source,candidate)['trainable_parameters'],65)
        candidate.weights['policy.bias'][0]+=.001
        with self.assertRaisesRegex(ValueError,'Frozen trunk or policy'):bench.verify_frozen_policy(source,candidate)

    def test_04_actual_control_clocks_and_physical_work(self):
        envelope=self.envelope();bench.validate_envelope(envelope,self.p,15,self.plan,'policy_control')
        decision=envelope['decision'];counter=envelope['evaluator']['counts']
        self.assertTrue(decision['usable_decision']);self.assertGreater(counter['native_actual_evals'],0)
        self.assertEqual(counter['native_actual_evals'],decision['report']['work']['neural_evaluations']);self.assertGreaterEqual(counter['feature_simulations'],counter['native_actual_evals'])
        self.assertGreaterEqual(counter['feature_deposits'],counter['feature_simulations']);self.assertEqual(counter['cache_hits'],0)
        self.assertEqual(decision['report']['c_puct'],.25);self.assertEqual(decision['overshoot_ms'],max(0,decision['wall_ms']-15))

    def test_05_reset_expiry_never_calls_time_zero_search(self):
        class Slow(CpuBatchEvaluator):
            def reset_cache(self):super().reset_cache();time.sleep(.012)
        with Slow(bench.CONTROL,cache_capacity=0) as obj:
            with patch.object(bench,'puct_search',side_effect=AssertionError('Unlimited time0 call')):
                envelope=bench.decide('policy_control',('puct',obj),self.p,5,self.plan)
        bench.validate_envelope(envelope,self.p,5,self.plan,'policy_control')
        self.assertFalse(envelope['decision']['usable_decision']);self.assertIsNone(envelope['decision']['chosen']);self.assertEqual(envelope['evaluator']['granted_search_ms'],0)
        self.assertEqual(envelope['evaluator']['counts']['native_actual_evals'],0)

    def test_06_both_actual_baseline_families_have_no_neural_counter_fabrication(self):
        with NeuralNative(ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json',mode='value-none') as relay:
            for name,engine in (('H_v1',('h',Native())),('H8_relay_CPU',('relay',relay))):
                answer=bench.decide(name,engine,self.p,5,self.plan);bench.validate_envelope(answer,self.p,5,self.plan,name);self.assertIsNone(answer['evaluator'])

    def test_07_rehashed_zero_feature_work_and_wrong_protocol_are_refused(self):
        value=self.envelope();broken=copy.deepcopy(value)
        for key in ('feature_simulations','feature_deposits'):
            broken['evaluator']['counts'][key]=0;broken['decision']['report']['work'][key]=0
        broken['decision']['report']['work']['total_deposits']=broken['decision']['report']['work']['deposits']
        with self.assertRaises((ValueError,NativeError)):bench.validate_envelope(broken,self.p,15,self.plan,'policy_control')
        broken=copy.deepcopy(value);broken['decision']['report']['protocol']='old-or-forged'
        with self.assertRaises((ValueError,NativeError)):bench.validate_envelope(broken,self.p,15,self.plan,'policy_control')
        broken=copy.deepcopy(value);broken['decision']['report']['c_puct']=1.5
        with self.assertRaises((ValueError,NativeError)):bench.validate_envelope(broken,self.p,15,self.plan,'policy_control')

    def test_08_full_case_raw_reports_and_outer_clock_validation(self):
        body=self.case();bench.validate_case(body,0,self.plan)
        broken=copy.deepcopy(body);entry=next(iter(next(iter(broken['measurements'].values())).values()))
        entry['decision']['decision']['wall_ms']=0
        with self.assertRaises(ValueError):bench.validate_case(broken,0,self.plan)
        broken=copy.deepcopy(body);broken['orders'][str(self.plan['budgets_ms'][0])].reverse()
        with self.assertRaises(ValueError):bench.validate_case(broken,0,self.plan)

    def test_09_noncontiguous_and_rehashed_case_prefixes_refuse(self):
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx,body,hash);CREATE TABLE games(idx,comparison,pair,seat,body,hash);')
            body=self.case();db.execute('INSERT INTO cases VALUES(?,?,?)',(1,config.canonical(body),config.digest(body)))
            with self.assertRaisesRegex(ValueError,'prefix'):bench.validate_saved_units(db,self.plan,Native())
            db.execute('DELETE FROM cases');db.execute('INSERT INTO cases VALUES(?,?,?)',(0,config.canonical(body),'0'*64))
            with self.assertRaisesRegex(ValueError,'prefix'):bench.validate_saved_units(db,self.plan,Native())

    def test_10_matched_margin_contrasts_and_incomplete_denominators(self):
        plan=copy.deepcopy(self.plan)
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx,body,hash);CREATE TABLE games(idx,comparison,pair,seat,body,hash);')
            for index,unit in enumerate(plan['arena_work_order']):
                pilot=unit['comparison'].startswith('value_pilot_');margin=5 if pilot else -3
                game=dict(pair=unit['pair'],status='terminal',candidate_final_margin=margin,candidate_match_score=1 if margin>0 else 0,fallbacks={})
                db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(index,unit['comparison'],unit['pair'],unit['seat'],config.canonical(game),config.digest(game)))
            result=bench.summarize(db,{},plan)
            for value in result['secondary_factor_contrasts'].values():self.assertEqual(value['paired_mean_seed_margin_difference'],8);self.assertEqual(value['complete_matched_families'],1)
            self.assertTrue(result['both_baseline_primary_gates']['value_pilot']);self.assertFalse(result['both_baseline_primary_gates']['policy_control'])
            db.execute("DELETE FROM games WHERE comparison='value_pilot_vs_H_v1' AND seat=1")
            result=bench.summarize(db,{},plan);self.assertIsNone(result['secondary_factor_contrasts']['H_v1']['paired_mean_seed_margin_difference']);self.assertFalse(result['both_baseline_primary_gates']['value_pilot'])

    def test_11_missing_or_forged_completion_refuses_before_publication(self):
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx,body,hash);CREATE TABLE games(idx,comparison,pair,seat,body,hash);')
            with self.assertRaises(ValueError):bench.validate_complete(db,{},self.plan,{'sha256':'x'},Path('nonexistent-value-pilot-report.json'),dict(cases=0,games=0,report_sha256='0'*64,resource_file_sha256='0'*64))
        self.assertFalse((ROOT/'nonexistent-value-pilot-report.json').exists())

    def test_12_resource_lineage_and_stage_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'report.json';path.write_text('{}\n',encoding='utf-8');implementation={'sha256':'a'*64}
            body=dict(schema='value-pilot-benchmark-resources-v1',implementation_sha256=implementation['sha256'],report_file_sha256=bench.sha(path),stage='posttraining',fixture=True,elapsed_seconds_this_invocation=1.,prior_committed_cases=0,prior_committed_games=0,host=bench.host_peak(),torch_imported=False,energy_measured=False,scope='This CPU-only invocation including admission/startup/remaining units/publication; earlier invocations excluded; no energy claim')
            bench.validate_resources(body,path,implementation,self.plan)
            bad={**body,'prior_committed_games':9}
            with self.assertRaises(ValueError):bench.validate_resources(bad,path,implementation,self.plan)
            bad={**body,'report_file_sha256':'0'*64}
            with self.assertRaises(ValueError):bench.validate_resources(bad,path,implementation,self.plan)


if __name__=='__main__':unittest.main(verbosity=2)
