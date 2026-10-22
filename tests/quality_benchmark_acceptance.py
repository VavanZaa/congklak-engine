"""CPU-only 2x2 admission, real decision clocks and durable semantic contracts."""
from __future__ import annotations
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,quality_common as quality,quality_search
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative
from congklak.puct_cpu_inference import CpuBatchEvaluator
from phase6_puct_acceptance import position
import phase6_benchmark_acceptance as fixture
spec=importlib.util.spec_from_file_location('quality_benchmark_under_test',ROOT/'tools/benchmark-quality.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def protocol(p):
    plan=fixture.protocol(p);plan['engines']=bench.ENGINES
    plan['profiles']={name:dict(model='warm' if name.startswith('warm_') else 'refined',c_puct=1.5 if name.endswith('control') else .25,cache_capacity=0) for name in bench.CANDIDATES}
    plan['models']=dict(warm='data/training-phase6-warm-v1/model.json',refined='data/quality-train-v1/model.json')
    plan['arena_comparisons']=[dict(id=c+'_vs_'+b,candidate=c,opponent=b,pairs=1) for c in bench.CANDIDATES for b in quality.BASELINES]
    plan['arena_work_order']=bench.arena_order(plan);return plan


class Tests(unittest.TestCase):
    def setUp(self):
        self.p=position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]);self.plan=protocol(self.p)
    def evaluator(self,name,stack):
        return stack.enter_context(CpuBatchEvaluator(ROOT/self.plan['models'][self.plan['profiles'][name]['model']],cache_capacity=0))
    def engines(self,stack):
        relay=stack.enter_context(NeuralNative(ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json',mode='value-none'))
        return {'H_v1':('h',Native()),'H8_relay_CPU':('relay',relay),**{name:('puct',self.evaluator(name,stack)) for name in bench.CANDIDATES}}
    def case(self):
        body=dict(index=0,state_key='fixture',cohort='fixture',stratum='fixture/ordinary',reference=self.plan['cases'][0]['reference'],orders={},measurements={},cpu_verified=bench.prior.verified(self.plan))
        q={a['pit']:a['value_player0'] for a in body['reference']['action_scores']}
        with ExitStack() as stack:
            engines=self.engines(stack)
            for bi,ms in enumerate(self.plan['budgets_ms']):
                order=bench.prior.expected_order(0,bi,self.plan);body['orders'][str(ms)]=order;body['measurements'][str(ms)]={}
                for name in order:
                    answer=bench.decide(name,engines[name],self.p,ms,self.plan);choice=answer['decision']['chosen']
                    body['measurements'][str(ms)][name]=dict(decision=answer,reference_optimal=choice in body['reference']['best_moves'] if choice is not None else None,reference_regret=body['reference']['target_player0']-q[choice] if choice is not None else None)
        return body
    def test_01_actual_1368_prior_pins_and_final_model_witness_no_torch(self):
        self.assertNotIn('torch',sys.modules);plan=bench.read_plan();manifest,witness=bench.admit(plan)
        self.assertEqual(witness['prior_accepted_inputs'],1368);self.assertEqual(witness['prior_checks'],34)
        self.assertEqual(len(witness['model_witnesses']['warm']['checkpoints']),20)
        self.assertTrue(witness['model_witnesses']['refined']);self.assertEqual(manifest,quality.load(quality.IMPLEMENTATION))
        self.assertNotIn('torch',sys.modules)
    def test_02_balanced384_order_and_fresh_development_geometry(self):
        plan=bench.read_plan();order=bench.arena_order(plan);self.assertEqual(len(order),384);self.assertEqual(order,plan['arena_work_order'])
        self.assertEqual(len({(r['comparison'],r['pair'],r['seat']) for r in order}),384)
        self.assertEqual([r['seat'] for r in order[:2]],[0,1]);self.assertEqual([r['seat'] for r in order[16:18]],[1,0])
        self.assertEqual(order[16]['comparison'],plan['arena_comparisons'][1]['id']);self.assertEqual(plan['budgets_ms'],[25,100]);self.assertEqual(plan['arena_ms'],25)
        self.assertEqual(len(plan['cases']),80);self.assertEqual(len(plan['engines']),6);self.assertEqual(len(plan['arena_starts']),24)
        fresh={r['family'] for r in plan['cases'] if r['cohort']=='quality_H8_development'};arena={r['family'] for r in plan['arena_starts']}
        previous=set(quality.read_registry()['arena_families']);self.assertFalse(fresh&arena);self.assertFalse((fresh|arena)&previous)
    def test_03_unregistered_paths_and_changed_pinned_source_refused(self):
        with self.assertRaises(ValueError):bench.read_plan(ROOT/'build/puct-cpu-plan.json')
        with self.assertRaises(ValueError):bench.admit({},ROOT/'build/puct-cpu-implementation.json')
        with self.assertRaisesRegex(ValueError,'Changed quality input'):quality.current({'congklak/quality_search.py':'0'*64})
    def test_04_all_four_profiles_actual_uncached_clocks_and_counters(self):
        with ExitStack() as stack:
            for name in bench.CANDIDATES:
                obj=self.evaluator(name,stack);result=bench.decide(name,('puct',obj),self.p,15,self.plan);bench.validate_envelope(result,self.p,15,self.plan,name)
                d=result['decision'];t=result['evaluator'];self.assertTrue(d['usable_decision']);self.assertLess(t['granted_search_ms'],15)
                self.assertEqual(d['report']['c_puct'],self.plan['profiles'][name]['c_puct']);self.assertEqual(d['report']['schema'],quality_search.SCHEMA)
                self.assertEqual(t['counts']['cache_resets'],1);self.assertEqual(t['counts']['cache_hits'],0);self.assertGreater(t['counts']['native_actual_evals'],0)
                self.assertEqual(t['counts']['native_actual_evals'],d['report']['work']['neural_evaluations']);self.assertEqual(t['cache_occupancy_after'],0)
                self.assertEqual(d['overshoot_ms'],max(0,d['wall_ms']-15))
        self.assertNotIn('torch',sys.modules)
    def test_05_reset_exhaustion_never_calls_unlimited_time_zero(self):
        class Slow(CpuBatchEvaluator):
            def reset_cache(self):super().reset_cache();time.sleep(.012)
        with Slow(ROOT/self.plan['models']['warm'],cache_capacity=0) as obj:
            with patch.object(bench,'puct_search',side_effect=AssertionError('Expired time must not call search')):
                result=bench.decide('warm_search',('puct',obj),self.p,5,self.plan)
        bench.validate_envelope(result,self.p,5,self.plan,'warm_search')
        self.assertFalse(result['decision']['usable_decision']);self.assertIsNone(result['decision']['chosen']);self.assertEqual(result['evaluator']['granted_search_ms'],0)
        self.assertEqual(result['evaluator']['counts']['native_actual_evals'],0);self.assertFalse(result['decision']['report']['root_expanded'])
    def test_06_actual_baselines_have_no_candidate_telemetry(self):
        with ExitStack() as stack:
            engines=self.engines(stack)
            for name in quality.BASELINES:
                value=bench.decide(name,engines[name],self.p,10,self.plan);bench.validate_envelope(value,self.p,10,self.plan,name);self.assertIsNone(value['evaluator'])
                bad=copy.deepcopy(value);bad['evaluator']={}
                with self.assertRaises(ValueError):bench.validate_envelope(bad,self.p,10,self.plan,name)
    def test_07_actual_raw_protocol_and_coefficient_cannot_be_normalized_away(self):
        with CpuBatchEvaluator(ROOT/self.plan['models']['warm'],cache_capacity=0) as obj:value=bench.decide('warm_search',('puct',obj),self.p,10,self.plan)
        raw=copy.deepcopy(value);view=bench.legacy_view(value['decision']);self.assertEqual(value,raw)
        self.assertEqual(view['report']['protocol'],bench.prior.PUCT_PROTOCOL);self.assertEqual(value['decision']['report']['protocol'],quality_search.PROTOCOL)
        for key,replacement in (('schema','phase6-puct-report-v1'),('protocol',bench.prior.PUCT_PROTOCOL),('c_puct',1.5),('value_perspective','root_actor')):
            bad=copy.deepcopy(value);bad['decision']['report'][key]=replacement;config.digest(bad)
            with self.assertRaises((ValueError,RuntimeError)):bench.validate_envelope(bad,self.p,10,self.plan,'warm_search')
    def test_08_rehashed_physical_work_and_budget_forgery_refused(self):
        with CpuBatchEvaluator(ROOT/self.plan['models']['warm'],cache_capacity=0) as obj:value=bench.decide('warm_control',('puct',obj),self.p,10,self.plan)
        mutations=(lambda t:t['counts'].__setitem__('cache_hits',1),lambda t:t.__setitem__('cache_occupancy_after_reset',1),lambda t:t.__setitem__('granted_search_ms',11),lambda t:t.__setitem__('physical_forward_executions',0),lambda t:t.__setitem__('logical_row_evaluations',0))
        for mutate in mutations:
            bad=copy.deepcopy(value);mutate(bad['evaluator']);config.digest(bad)
            with self.assertRaises(ValueError):bench.validate_envelope(bad,self.p,10,self.plan,'warm_control')
        bad=copy.deepcopy(value)
        for key in ('feature_simulations','feature_deposits'):bad['evaluator']['counts'][key]=0;bad['decision']['report']['work'][key]=0
        bad['decision']['report']['work']['total_deposits']=bad['decision']['report']['work']['deposits']
        with self.assertRaises(ValueError):bench.validate_envelope(bad,self.p,10,self.plan,'warm_control')
    def test_09_actual_case_reference_regret_and_engine_order(self):
        body=self.case();bench.validate_case(body,0,self.plan)
        bad=copy.deepcopy(body);bad['orders']['5'].reverse()
        with self.assertRaises(ValueError):bench.validate_case(bad,0,self.plan)
        bad=copy.deepcopy(body);bad['measurements']['5']['H_v1']['reference_regret']+=1
        with self.assertRaises(ValueError):bench.validate_case(bad,0,self.plan)
    def test_10_checksum_valid_case_holes_and_arena_orphans_refused(self):
        text=config.canonical({'rehashed':'unregistered'});digest=config.digest(json.loads(text))
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx INTEGER,body TEXT,hash TEXT);CREATE TABLE games(idx INTEGER,comparison TEXT,pair INTEGER,seat INTEGER,body TEXT,hash TEXT);')
            db.execute('INSERT INTO cases VALUES(?,?,?)',(1,text,digest))
            with self.assertRaisesRegex(ValueError,'noncontiguous quality case'):bench.validate_saved_units(db,self.plan,Native())
            db.execute('DELETE FROM cases');db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(0,self.plan['arena_comparisons'][0]['id'],1,0,text,digest))
            with self.assertRaisesRegex(ValueError,'orphan quality arena'):bench.validate_saved_units(db,self.plan,Native())
    def test_11_real_terminal_game_replay_and_rehashed_outcome_refusal(self):
        comp=self.plan['arena_comparisons'][0];p=self.p;moves=[];decisions=[];native=Native()
        with ExitStack() as stack:
            engines=self.engines(stack)
            for _ in range(512):
                normal=native.normalize(flatten(p),actor=p['controller'],total=98)
                if normal['terminal']:break
                p=normal['position'];name=comp['candidate'] if p['controller']==0 else comp['opponent'];answer=bench.decide(name,engines[name],p,5,self.plan)
                self.assertTrue(answer['decision']['usable_decision']);action=answer['decision']['chosen'];decisions.append(dict(engine=name,actor=p['controller'],fallback=False,decision=answer));moves.append(action)
                p=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False)['position']
        self.assertFalse(any(map(sum,p['pits'])));margin=p['stores'][0]-p['stores'][1]
        body=dict(comparison=comp['id'],pair=0,candidate_seat=0,start=self.p,family='fixture',moves=moves,decisions=decisions,final=p,status='terminal',candidate_final_margin=margin,candidate_match_score=1 if margin>0 else 0 if margin<0 else .5,fallbacks={},cpu_verified=bench.prior.verified(self.plan))
        bench.replay_game(body,comp,0,0,self.plan,native);bad=copy.deepcopy(body);bad['candidate_final_margin']+=2
        with self.assertRaises(ValueError):bench.replay_game(bad,comp,0,0,self.plan,native)
    def test_12_unfinished_pair_exclusion_and_resource_binding(self):
        self.assertEqual(bench.prior.interval([1,2,3,4],self.plan),bench.prior.interval([1,2,3,4],self.plan));self.assertIsNone(bench.prior.interval([],self.plan))
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx INTEGER,body TEXT);CREATE TABLE games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT);')
            comp=self.plan['arena_comparisons'][0]['id']
            for seat,status,margin,score in ((0,'terminal',10,1),(1,'unfinished',None,None)):
                db.execute('INSERT INTO games VALUES(?,?,?,?)',(comp,0,seat,config.canonical(dict(pair=0,status=status,candidate_final_margin=margin,candidate_match_score=score,fallbacks={}))))
            arena=bench.summarize(db,{},self.plan)['arenas'][comp]
            self.assertEqual(arena['complete_pairs'],0);self.assertIsNone(arena['primary_paired_mean_final_margin_seeds']);self.assertFalse(arena['primary_strength_gate_passed']);self.assertEqual(arena['unfinished_games'],1)
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as folder:
            path=Path(folder)/'report.json';quality.publish(path,{'fixture':True});manifest={'sha256':'1'*64}
            body=dict(schema='quality-benchmark-resources-v1',implementation_sha256=manifest['sha256'],report_file_sha256=quality.sha(path),elapsed_seconds_this_invocation=.1,prior_committed_cases=0,prior_committed_games=0,host=bench.host_peak(),torch_imported=False,energy_measured=False,scope='This CPU-only invocation including admission/startup/remaining units/publication; previous invocations excluded; no electricity or GPU-use claim')
            bench.validate_resources(body,path,manifest)
            for key,replacement in (('report_file_sha256','0'*64),('elapsed_seconds_this_invocation',-1),('prior_committed_games',385),('energy_measured',True)):
                bad=copy.deepcopy(body);bad[key]=replacement
                with self.assertRaises(ValueError):bench.validate_resources(bad,path,manifest)

    def test_13_secondary_factor_contrasts_require_all_matched_pairs(self):
        margins={'warm_control':2,'warm_search':8,'refined_control':12,'refined_search':20}
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx INTEGER,body TEXT);CREATE TABLE games(comparison TEXT,pair INTEGER,seat INTEGER,body TEXT);')
            for comp in self.plan['arena_comparisons']:
                for seat in (0,1):
                    game=dict(pair=0,status='terminal',candidate_final_margin=margins[comp['candidate']]+seat*2,candidate_match_score=1,fallbacks={})
                    db.execute('INSERT INTO games VALUES(?,?,?,?)',(comp['id'],0,seat,config.canonical(game)))
            contrasts=bench.summarize(db,{},self.plan)['secondary_factor_contrasts']
            for baseline in quality.BASELINES:
                for name,wanted in (('search_at_warm',6),('search_at_refined',8),('refinement_at_control',10),('refinement_at_search',12)):
                    row=contrasts[baseline][name];self.assertEqual(row['complete_matched_families'],1);self.assertEqual(row['paired_mean_seed_margin_difference'],wanted);self.assertIsNone(row['confidence_interval']);self.assertFalse(row['primary_gate'])
            db.execute('DELETE FROM games WHERE comparison=? AND seat=1',('refined_search_vs_H_v1',))
            contrasts=bench.summarize(db,{},self.plan)['secondary_factor_contrasts']
            for name in ('search_at_refined','refinement_at_search'):
                row=contrasts['H_v1'][name];self.assertEqual(row['complete_matched_families'],0);self.assertIsNone(row['paired_mean_seed_margin_difference'])
            self.assertEqual(contrasts['H_v1']['search_at_warm']['paired_mean_seed_margin_difference'],6)

if __name__=='__main__':unittest.main(verbosity=2)
