"""Finite evaluation schedule and fail-closed replay tests without any query."""
from collections import Counter
from contextlib import contextmanager
import unittest
from unittest.mock import patch
from congklak import guarded_value_pilot_v2_common as stage
from congklak import guarded_value_pilot_v2_query_contract as contract
from congklak import guarded_value_pilot_v2_execute as execute
from congklak import ordinary_epoch_query_contract as frozen_contract
from congklak import guarded_value_pilot_v2_resources as resources


def metadata_fixture():
    fit = []
    for band, kind in (('25-49','ordinary'),('50-74','ordinary'),('25-49','capture'),('25-49','bonus')):
        for actor in (0,1):
            cell = band + '/' + kind + '/actor' + str(actor)
            for index in range(16):
                key = cell + '/' + str(index)
                fit.append(dict(state_key=key,cell=cell,origin='fit/'+key,family=key))
    cases = {row['state_key'] for row in fit if int(row['state_key'].rsplit('/',1)[1]) < 4}
    arena = [dict(state_key='arena/'+str(index),origin='arena/'+str(index),family='arena/'+str(index),
                  cell=('25-49' if index<12 else '50-74')+'/ordinary/actor'+str((index//6)%2)) for index in range(24)]
    exact = [dict(state_key='exact/'+str(index),origin=contract.SHARED,family='exact/'+str(index),cell='exact')
             for index in range(32)]
    return dict(fit=fit,cases=cases,arena=arena,exact=exact,train=[])


class GuardedEvaluationTests(unittest.TestCase):
    def test_schedule_exactly_matches_frozen_order_and_counterbalancing(self):
        with patch.object(contract,'membership',side_effect=metadata_fixture), \
             patch.object(frozen_contract,'membership',side_effect=metadata_fixture):
            for operation in contract.OPERATIONS:
                for fixture in (False,True):
                    self.assertEqual(contract.schedule(operation,fixture),frozen_contract.schedule(operation,fixture))

    def test_exact_paired_arena_coverage_and_adjacent_profiles(self):
        with patch.object(contract,'membership',side_effect=metadata_fixture):
            rows=contract.schedule('arena')
        self.assertEqual(len(rows),192)
        cells=Counter((row['meta']['origin'],row['baseline'],row['seat'],row['profile']) for row in rows)
        self.assertEqual(set(cells.values()),{1})
        self.assertEqual(len(cells),24*2*2*2)
        self.assertEqual({row['baseline'] for row in rows},{'H_v1','H8_relay_CPU'})
        for first,second in zip(rows[::2],rows[1::2]):
            self.assertEqual((first['meta'],first['baseline'],first['seat']),
                             (second['meta'],second['baseline'],second['seat']))
            self.assertEqual({first['profile'],second['profile']},{'control','candidate'})

    def test_case_endgame_and_initial_equivalence_counts(self):
        with patch.object(contract,'membership',side_effect=metadata_fixture):
            guards=contract.schedule('guards')
            conformance=contract.schedule('conformance')
            initial=contract.schedule('conformance',True)
        self.assertEqual(len(guards),256)
        self.assertEqual(Counter(row['cohort'] for row in guards),dict(cases=128,endgame=128))
        self.assertEqual(Counter(row['mode'] for row in guards),dict(timed25=128,fixed128=128))
        cases={row['meta']['state_key'] for row in guards if row['cohort']=='cases'}
        self.assertEqual(len(cases),32)
        self.assertEqual(len(conformance),256)
        self.assertEqual(Counter(row['engine'] for row in conformance),dict(control=128,candidate=128))
        self.assertEqual([(row['kind'],row['engine'],row.get('mode')) for row in initial],
            [('features','control',None),('prediction','control',None),('prediction','candidate',None),
             ('decision','control','fixed128'),('decision','candidate','fixed128')])
        self.assertTrue(all(row['kind']!='teacher' for row in guards+conformance+initial))
        with self.assertRaises(ValueError):
            contract.schedule('labels')

    def test_global_claim_predecessor_chain_and_registered_scopes(self):
        initial=stage.FOLDER/'jobs/initial-conformance'
        self.assertIsNone(contract.predecessor(initial,'conformance',True))
        expected=[(stage.FOLDER/'jobs/conformance','conformance',False,initial,'conformance'),
            (stage.FOLDER/'jobs/guards','guards',False,stage.FOLDER/'jobs/conformance','conformance'),
            (contract.FIXTURE_BASE/'before_handoff_commit','arena',True,stage.FOLDER/'jobs/guards','guards'),
            (contract.FIXTURE_BASE/'after_handoff_commit','arena',True,contract.FIXTURE_BASE/'before_handoff_commit','arena'),
            (stage.FOLDER/'jobs/arena','arena',False,contract.FIXTURE_BASE/'after_handoff_commit','arena')]
        for output,operation,fixture,prior,prior_operation in expected:
            self.assertEqual(contract.predecessor(output,operation,fixture),
                dict(path=stage.relative(prior),operation=prior_operation))
        for output,operation,fixture in ((stage.FOLDER/'jobs/labels','labels',False),
                (stage.FOLDER/'fixtures/native-recovery-v2/arbitrary','arena',True),
                (stage.FOLDER/'jobs/initial-conformance','arena',True)):
            with self.assertRaises(ValueError):
                contract.classify(output,operation,fixture)

    def test_first_live_claim_prefix_is_empty_and_unknown_query_never_retried(self):
        claims,generation=execute._seed_claims(dict(seed=None),dict(origins=['origin0']))
        self.assertEqual((len(claims),generation),(0,0))
        target=stage.FOLDER/'fixtures/native-recovery-v2/before_handoff_commit'
        with patch.object(execute,'read',return_value={}), \
             patch.object(execute.Path,'is_file',return_value=False):
            with self.assertRaisesRegex(ValueError,'Unknown unreturned query'):
                execute._adopt(target,{},dict(intent=dict(archive='missing.json')))

    def test_completed_native_route_precedes_writer_resources_and_provider(self):
        target=stage.FOLDER/'jobs/arena'
        expected=dict(report=dict(state='complete'))
        with patch.object(execute,'has_completion',return_value=True), \
             patch.object(execute,'completed_job',return_value=expected), \
             patch.object(execute,'Writer',side_effect=AssertionError('Writer entered')):
            self.assertEqual(execute.run(target,operation='arena'),expected)

    def test_leased_main_binds_before_views_and_closes_before_completion(self):
        calls=[]
        @contextmanager
        def scope():
            self.assertEqual(calls,['bind'])
            calls.append('view-enter')
            yield
            calls.append('view-close')
        def run(*args):
            self.assertEqual(calls,['bind','view-enter'])
            calls.append('run')
            return dict(report=dict(state='complete'))
        def complete(*args):
            self.assertEqual(calls,['bind','view-enter','run','view-close'])
            calls.append('complete')
        with patch.object(resources,'bind_worker',side_effect=lambda *a:calls.append('bind')), \
             patch.object(stage,'worker_views',side_effect=scope),patch.object(execute,'run',side_effect=run), \
             patch.object(resources,'worker_complete',side_effect=complete),patch('builtins.print'), \
             patch.object(execute,'has_completion',side_effect=AssertionError('Heavy read before bind')):
            execute.main(['run','--lease','synthetic','--operation','arena'])
        self.assertEqual(calls[-1],'complete')


if __name__=='__main__':
    unittest.main()
