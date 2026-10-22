"""Six owner checks for value-only clocks, censored labels and bounded allocation."""
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_value_benchmark as b,pilot
from congklak.native import Native,NativeError
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator


class OrdinaryValueBenchmarkAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.make_plan();cls.plan=b.read_plan(fixture=True);cls.native=Native();cls.base=ROOT/'build/ordinary-value-benchmark-author'/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.path=cls.base/'fixture';cls.output=cls.base/'fixture.json'
        cls.partial=b.run(path=cls.path,output=cls.output,fixture=True,fixture_control=True,decision_limit=1,on_event=lambda _:None)
        with closing(sqlite3.connect(cls.path/'work.sqlite3')) as db:
            cls.prefix=db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice').fetchall();cls.cases=[json.loads(text) for text, in db.execute('SELECT body FROM cases ORDER BY idx')]

    def copydb(self):
        db=sqlite3.connect(':memory:')
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as source:source.backup(db)
        return db

    def test_01_blind_reservations_and_fp32_frozen_policy_contract(self):
        projection=b.load(b.PROJECTION);plan=b.load(b.PLAN);self.assertEqual(len(b.baseline()['inputs']),3364)
        self.assertEqual(len(projection['arena_starts']),12);self.assertEqual(len(projection['prior_arena_families']),144);self.assertEqual(len(projection['validation_roots']),6)
        self.assertFalse(set(projection['prior_arena_families'])&{row['family'] for row in projection['arena_starts']})
        self.assertEqual(plan['declared_geometry']['scalar_measurements'],304);self.assertEqual(plan['declared_geometry']['exact_scalar_measurements'],256);self.assertEqual(plan['declared_geometry']['validation_scalar_measurements'],48)
        self.assertTrue(all(profile['c_puct']==.25 and profile['cache_capacity']==0 for profile in plan['profiles'].values()));self.assertEqual(len(plan['cases']),32)
        roots=projection['arena_starts']+projection['cases']+projection['validation_roots'];self.assertEqual(set(projection['reserved_quotients']),{pilot.quotient(row['state']).hex() for row in roots})
        control=Model(b.CONTROL);candidate=copy.copy(control);candidate.weights=copy.deepcopy(control.weights);candidate.weights['layers.4.bias'][0]+=.01
        proof=b.verify_candidate(control,candidate);self.assertEqual(proof['trainable_parameters'],65);self.assertEqual(proof['frozen_parameters'],12807);self.assertTrue(proof['global_policy_unchanged_by_identical_subgraph'])
        candidate.weights['policy.bias'][0]+=.001
        with self.assertRaisesRegex(ValueError,'Frozen'):b.verify_candidate(control,candidate)
        self.assertNotIn('torch',sys.modules)

    def test_02_physical_forward_accounting_and_whole_deadline(self):
        state=self.plan['arena_starts'][0]['state']
        with CpuBatchEvaluator(b.CONTROL,cache_capacity=0) as evaluator:value=b.decide('ordinary_control',('puct',evaluator),state,25,self.plan)
        b.validate_envelope(value,state,25,self.plan,'ordinary_control');self.assertEqual(value['decision']['report']['c_puct'],.25)
        bad=copy.deepcopy(value);bad['decision']['report']['c_puct']=1.5
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,state,25,self.plan,'ordinary_control')
        bad=copy.deepcopy(value);bad['decision']['report']['work']['feature_simulations']=bad['evaluator']['counts']['feature_simulations']=0
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,state,25,self.plan,'ordinary_control')
        bad=copy.deepcopy(value);bad['evaluator']['granted_search_ms']=26
        with self.assertRaises((ValueError,NativeError)):b.validate_envelope(bad,state,25,self.plan,'ordinary_control')

    def test_03_exact_and_finite_missing_reference_denominators(self):
        self.assertEqual({row['cohort'] for row in self.cases},{'exact_guard','ordinary_validation_H10'})
        for index,row in enumerate(self.cases):b.validate_case(row,index,self.plan)
        wrong=copy.deepcopy(self.cases[0]);wrong['orders']['25'].reverse()
        with self.assertRaisesRegex(ValueError,'order'):b.validate_case(wrong,0,self.plan)
        missing=copy.deepcopy(self.cases[1]);plan=copy.deepcopy(self.plan);plan['cases'][1]['reference']=missing['reference']=None;plan['cases'][1]['reference_status']=missing['reference_status']='missing_requested_H10'
        for values in missing['measurements'].values():
            for row in values.values():row['reference_optimal']=row['reference_regret']=None
        b.validate_case(missing,1,plan)
        missing['measurements']['25']['ordinary_control']['reference_regret']=0
        with self.assertRaisesRegex(ValueError,'Missing reference'):b.validate_case(missing,1,plan)
        # Metadata-only closed-loader stub checks original-root versus wrapper
        # linkage. It publishes no teacher answers or production artifacts.
        from congklak import ordinary_value_pilot as learning
        originals=b.load(b.PLAN)['validation_roots'];roots={};stores={'h10_labels':{},'h10_attempts':{}}
        for original in originals:
            key=original['state_key'];wrapper=dict(root=original,state=original['state'],source_root_sha256=config.digest(original),state_key=key)
            roots[key]=wrapper;stores['h10_labels'][key]=dict(root_sha256=config.digest(original),label=dict(choice_depth=10),report=dict(choice_depth=10));stores['h10_attempts'][key]=dict(status='complete',source_root_sha256=config.digest(original),study_row_sha256=config.digest(wrapper))
        # This metadata-link unit stub borrows only the already retained fixture
        # journal's file hash. It does not assert that journal contains labels.
        with patch.object(learning,'load_labels',return_value=(roots,stores,dict(metadata_fixture_only=True))),patch.object(learning,'LABEL_DEFAULT',self.path):
            runtime=b.read_plan();self.assertEqual(len(runtime['cases']),38)
            for row in runtime['cases'][32:]:
                key=row['state_key'];provenance=row['reference_provenance'];self.assertEqual(provenance['source_root_sha256'],config.digest(roots[key]['root']));self.assertEqual(provenance['study_wrapper_sha256'],config.digest(roots[key]));self.assertEqual(provenance['label_root_sha256'],provenance['source_root_sha256'])
            key=next(iter(roots));stores['h10_labels'][key]['root_sha256']=config.digest(roots[key])
            with self.assertRaisesRegex(ValueError,'source-root'):b.read_plan()

    def test_04_committed_decision_resume_and_exact_guard_isolation(self):
        self.assertEqual(self.partial['status'],'partial');self.assertEqual(len(self.prefix),1)
        result=b.run(path=self.path,output=self.output,fixture=True,fixture_control=True,on_event=lambda _:None);self.__class__.result=result
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as db:
            self.assertEqual(self.prefix,db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice LIMIT 1').fetchall());self.assertEqual(b.validate_saved_units(db,self.plan,self.native),(2,8))
        self.assertEqual(result['games'],8);self.assertEqual(result['scalar_measurements'],16);self.assertTrue(result['terminal_zero_fallback_gate'])
        for values in result['endgame_guard'].values():
            for name in b.CANDIDATES:self.assertEqual(values[name]['roots'],1)
        self.assertEqual(result['validation_model_fit']['requested_roots'],1);self.assertFalse(result['trajectory_replay_overlap']['checked']);self.assertFalse(result['default_promotion'])

    def test_05_rehashed_prefix_and_prepublication_marker_refusal(self):
        with closing(self.copydb()) as db:
            row=json.loads(db.execute('SELECT body FROM decisions WHERE game=0 AND choice=0').fetchone()[0]);row['after']['stores'][0]+=1
            db.execute('UPDATE decisions SET body=?,hash=? WHERE game=0 AND choice=0',(config.canonical(row),config.digest(row)))
            with self.assertRaisesRegex(ValueError,'transition'):b.validate_saved_units(db,self.plan,self.native)
        target=self.base/'bad-marker';target.mkdir()
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as source,closing(sqlite3.connect(target/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(target/'work.sqlite3')) as db:
            marker=b._metadata(db,'ordinary_value_benchmark_complete');marker['decisions']+=1
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='ordinary_value_benchmark_complete'",(config.canonical(marker),config.digest(marker)))
        output=self.base/'bad-marker.json'
        with self.assertRaisesRegex(ValueError,'completion marker'):b.run(path=target,output=output,fixture=True,fixture_control=True,on_event=lambda _:None)
        self.assertFalse(output.exists())

    def test_06_no_work_complete_and_predeclared_allocation_branches(self):
        def snapshot():return {path.name:(path.read_bytes(),path.stat().st_mtime_ns) for path in list(self.path.iterdir())+[self.output,self.output.with_name(self.output.stem+'-resources.json')] if path.is_file() and not path.name.endswith(('-wal','-shm'))}
        before=snapshot()
        def prohibited(*_args,**_kwargs):raise AssertionError('Completed benchmark attempted new work')
        with patch.object(Native,'analyze',prohibited),patch.object(b,'decide',prohibited),patch.object(b,'puct_search',prohibited),patch.object(CpuBatchEvaluator,'__init__',prohibited):actual=b.run(path=self.path,output=self.output,fixture=True,fixture_control=True,on_event=lambda _:None)
        self.assertEqual(actual,self.result);self.assertEqual(before,snapshot())
        report=dict(validation_model_fit=dict(requested_roots=6,complete_reference_roots=6,control_mae=10,candidate_mae=9),secondary_factor_contrasts={name:dict(paired_mean_seed_margin_difference=0) for name in b.BASELINES},endgame_guard={str(ms):dict(non_regression_observed=True) for ms in (25,100)},arenas={'fixture':dict(fallback_counts={'profile':0})},terminal_zero_fallback_gate=True,games=96,trajectory_replay_overlap=dict(checked=True,contaminated_evaluation=False))
        self.assertEqual(b.allocation_from_report(report,True)['branch'],'new_starts');self.assertEqual(b.allocation_from_report(report,False)['branch'],'hold')
        for change in ('missing','worsefit','negativegain','exactregression','collision'):
            altered=copy.deepcopy(report)
            if change=='missing':altered['validation_model_fit']['complete_reference_roots']=5
            elif change=='worsefit':altered['validation_model_fit']['candidate_mae']=11
            elif change=='negativegain':altered['secondary_factor_contrasts']['H8_relay_CPU']['paired_mean_seed_margin_difference']=-.01
            elif change=='exactregression':altered['endgame_guard']['100']['non_regression_observed']=False
            else:altered['trajectory_replay_overlap']['contaminated_evaluation']=True
            self.assertEqual(b.allocation_from_report(altered,True)['branch'],'hold' if change in ('exactregression','collision') else 'repeat_high')


if __name__=='__main__':unittest.main(verbosity=2)
