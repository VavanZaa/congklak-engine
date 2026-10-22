"""Author checks for reserved schedules, physical clocks and decision-prefix recovery."""
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,selfplay_pilot_benchmark as b,pilot
from congklak.native import Native,NativeError
from congklak.puct_cpu_inference import CpuBatchEvaluator


class SelfplayBenchmarkAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.make_plan();cls.plan=b.read_plan(fixture=True);cls.native=Native()
        cls.base=ROOT/'build/selfplay-pilot-benchmark-author'/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.path=cls.base/'fixture';cls.output=cls.base/'fixture.json'
        cls.partial=b.run(path=cls.path,output=cls.output,fixture=True,fixture_control=True,decision_limit=1,on_event=lambda _:None)
        with closing(sqlite3.connect(cls.path/'work.sqlite3')) as db:
            cls.prefix=db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice').fetchall()
            cls.case=json.loads(db.execute('SELECT body FROM cases WHERE idx=0').fetchone()[0])

    def copydb(self):
        db=sqlite3.connect(':memory:')
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as source:source.backup(db)
        return db

    def test_01_frozen_reservations_geometry_and_joint_contract(self):
        prior=b.baseline();projection=b.load(b.PROJECTION);plan=b.read_plan()
        self.assertEqual(len(prior['inputs']),2839);self.assertEqual(len(plan['arena_starts']),12)
        self.assertEqual(len({r['family'] for r in plan['arena_starts']}),12)
        self.assertFalse(set(projection['prior_arena_families'])&{r['family'] for r in plan['arena_starts']})
        self.assertEqual({name:sum(r['stratum'].endswith('/'+name) for r in plan['arena_starts']) for name in ('ordinary','capture','bonus')},{'ordinary':4,'capture':4,'bonus':4})
        self.assertEqual(len(plan['arena_work_order']),96);self.assertEqual(len(plan['cases'])*len(plan['budgets_ms'])*len(plan['engines']),256)
        self.assertEqual(set(projection['reserved_quotients']),{pilot.quotient(row['state']).hex() for row in plan['arena_starts']+plan['cases']})
        self.assertTrue(all(row['reference']['kind']=='exact_optimal_margin' for row in plan['cases']))
        self.assertTrue(all(profile['c_puct']==.25 and profile['cache_capacity']==0 for profile in plan['profiles'].values()))
        self.assertFalse(plan['default_promotion']);self.assertFalse(plan['sealed_final_test_consumed']);self.assertNotIn('torch',sys.modules)

    def test_02_real_feature_forward_counters_and_quarter_puct(self):
        position=self.plan['arena_starts'][0]['state']
        with CpuBatchEvaluator(b.CONTROL,cache_capacity=0) as evaluator:value=b.decide('selfplay_control',('puct',evaluator),position,25,self.plan)
        b.validate_envelope(value,position,25,self.plan,'selfplay_control')
        self.assertEqual(value['decision']['report']['c_puct'],.25)
        bad=copy.deepcopy(value);bad['decision']['report']['c_puct']=1.5
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,position,25,self.plan,'selfplay_control')
        bad=copy.deepcopy(value);bad['decision']['report']['work']['feature_simulations']=bad['evaluator']['counts']['feature_simulations']=0
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,position,25,self.plan,'selfplay_control')
        bad=copy.deepcopy(value);bad['evaluator']['granted_search_ms']=26
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,position,25,self.plan,'selfplay_control')

    def test_03_exact_guard_preserves_raw_reports_order_and_true_regret(self):
        b.validate_case(self.case,0,self.plan)
        for ms,values in self.case['measurements'].items():
            for name,entry in values.items():
                report=entry['decision']['decision']['report']
                if name in b.CANDIDATES:self.assertEqual(report['protocol'],b.quality_search.PROTOCOL)
                if entry['decision']['decision']['usable_decision']:self.assertGreaterEqual(entry['reference_regret'],0)
        wrong=copy.deepcopy(self.case);wrong['orders']['25'].reverse()
        with self.assertRaises(ValueError):b.validate_case(wrong,0,self.plan)
        wrong=copy.deepcopy(self.case);entry=next(iter(wrong['measurements']['25'].values()));entry['reference_regret']=999
        with self.assertRaises(ValueError):b.validate_case(wrong,0,self.plan)

    def test_04_corrupt_noncontiguous_and_rehashed_decisions_refused(self):
        with closing(self.copydb()) as db:
            db.execute('UPDATE decisions SET choice=1 WHERE game=0 AND choice=0')
            with self.assertRaisesRegex(ValueError,'prefix'):b.validate_saved_units(db,self.plan,self.native)
        with closing(self.copydb()) as db:
            row=json.loads(db.execute('SELECT body FROM decisions WHERE game=0 AND choice=0').fetchone()[0]);row['after']['stores'][0]+=1
            db.execute('UPDATE decisions SET body=?,hash=? WHERE game=0 AND choice=0',(config.canonical(row),config.digest(row)))
            with self.assertRaisesRegex(ValueError,'transition'):b.validate_saved_units(db,self.plan,self.native)
        with closing(self.copydb()) as db:
            row=json.loads(db.execute('SELECT body FROM decisions WHERE game=0 AND choice=0').fetchone()[0]);row['choice_index']=False
            db.execute('UPDATE decisions SET body=?,hash=? WHERE game=0 AND choice=0',(config.canonical(row),config.digest(row)))
            with self.assertRaisesRegex(ValueError,'identity'):b.validate_saved_units(db,self.plan,self.native)

    def test_05_actual_committed_decision_resume_and_terminal_games(self):
        self.assertEqual(self.partial['status'],'partial');self.assertEqual(len(self.prefix),1)
        result=b.run(path=self.path,output=self.output,fixture=True,fixture_control=True,on_event=lambda _:None)
        self.__class__.result=result
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as db:
            self.assertEqual(self.prefix,db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice LIMIT 1').fetchall())
            self.assertEqual(b.validate_saved_units(db,self.plan,self.native),(1,8))
        self.assertEqual(result['games'],8);self.assertEqual(result['scalar_measurements'],8)
        self.assertTrue(result['terminal_zero_fallback_gate'])
        self.assertFalse(result['default_promotion']);self.assertFalse(result['trajectory_replay_overlap']['checked'])

    def test_06_forged_complete_refuses_before_missing_export_repair(self):
        target=self.base/'bad-marker';target.mkdir()
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as source,closing(sqlite3.connect(target/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(target/'work.sqlite3')) as db:
            marker=b._metadata(db,'selfplay_pilot_benchmark_complete');marker['decisions']+=1
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='selfplay_pilot_benchmark_complete'",(config.canonical(marker),config.digest(marker)))
        output=self.base/'bad-marker.json'
        with self.assertRaisesRegex(ValueError,'completion marker'):b.run(path=target,output=output,fixture=True,fixture_control=True,on_event=lambda _:None)
        self.assertFalse(output.exists())

    def test_07_matched_margin_intervals_and_incomplete_denominators(self):
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE cases(idx,body,hash);CREATE TABLE games(idx,comparison,pair,seat,body,hash);CREATE TABLE decisions(game,choice,body,hash);')
            for index,unit in enumerate(self.plan['arena_work_order']):
                treatment=unit['comparison'].startswith('selfplay_challenger_');margin=3 if treatment else -2
                game=dict(pair=unit['pair'],status='terminal',candidate_final_margin=margin,candidate_match_score=1 if margin>0 else 0,fallbacks={})
                db.execute('INSERT INTO games VALUES(?,?,?,?,?,?)',(index,unit['comparison'],unit['pair'],unit['seat'],config.canonical(game),config.digest(game)))
            result=b.summarize(db,{},self.plan)
            for row in result['secondary_factor_contrasts'].values():
                self.assertEqual(row['paired_mean_seed_margin_difference'],5);self.assertEqual(row['complete_matched_families'],1)
                self.assertEqual(row['confidence_interval'],b.prior.interval([5],self.plan))
            db.execute("DELETE FROM games WHERE comparison='selfplay_challenger_vs_H_v1' AND seat=1")
            result=b.summarize(db,{},self.plan)
            self.assertIsNone(result['secondary_factor_contrasts']['H_v1']['paired_mean_seed_margin_difference']);self.assertFalse(result['both_baseline_primary_gates']['selfplay_challenger'])
            self.assertFalse(result['endgame_guard']['25']['non_regression_observed'])

    def test_08_complete_rerun_has_no_search_evaluator_or_mutation(self):
        def snapshot():return {path.name:(path.read_bytes(),path.stat().st_mtime_ns) for path in list(self.path.iterdir())+[self.output,self.output.with_name(self.output.stem+'-resources.json')] if path.is_file() and not path.name.endswith(('-wal','-shm'))}
        before=snapshot()
        def prohibited(*args,**kwargs):raise AssertionError('Completed benchmark attempted new work')
        with patch.object(Native,'analyze',prohibited),patch.object(b,'decide',prohibited),patch.object(b,'puct_search',prohibited),patch.object(CpuBatchEvaluator,'__init__',prohibited):
            actual=b.run(path=self.path,output=self.output,fixture=True,fixture_control=True,on_event=lambda _:None)
        self.assertEqual(actual,self.result);self.assertEqual(snapshot(),before)


if __name__=='__main__':unittest.main(verbosity=2)
