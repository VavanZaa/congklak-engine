"""Acceptance probes for read-only ownership reconstruction on retained data."""
import copy
import sys
import unittest
from unittest.mock import patch
from congklak import value_bridge_diagnostic as diagnostic


class BridgeDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=diagnostic.plan()
        cls.results=[diagnostic.unit(t) for t in cls.plan['tasks']]
        cls.report=diagnostic.summarize(cls.results)

    def test_complete_frozen_reconstruction_and_work(self):
        r=self.report
        self.assertTrue(r['original_ownership_reconstructed_exactly'])
        self.assertEqual((r['units'],r['current_arena_units'],r['literal_replayed_actions']),(207,192,6663))
        self.assertEqual((r['raw_observations'],r['eligible_observations']),(6855,5160))
        self.assertEqual(len(r['direct_quotient_bridges']),12)
        self.assertEqual(len(r['historical_component_bridges']),3)
        self.assertEqual((r['historical_provenance']['sampled_saved_games'],r['historical_literal_replayed_actions']),(120,3970))
        self.assertEqual(len(r['historical_provenance']['sampled_cross_origin_collision_examples']),15)
        self.assertNotIn('torch',sys.modules)
        for key in ('optimizer_steps','model_forwards','searches','new_games','new_teacher_labels'):
            self.assertEqual(r[key],0)

    def test_train_alias_is_store_offset_not_exact_optimizer_root(self):
        witnesses=self.report['training_alias_witnesses']
        self.assertEqual([(w['observation']['game'],w['observation']['choice']) for w in witnesses],[(152,18),(156,18)])
        for w in witnesses:
            observation=w['observation'];owner=w['owners'][0]
            self.assertEqual(observation['active_seeds'],7)
            self.assertEqual(observation['preceding']['engine'],'H8_relay_CPU')
            self.assertEqual(observation['preceding']['action'],0)
            self.assertEqual(observation['preceding']['captured'],0)
            self.assertEqual(len(w['path']),19)
            self.assertEqual(owner['reserved_roots'][0]['family'],'search_history/20261015/2132')
            self.assertEqual(owner['reserved_roots'][0]['cell'],'25-49/bonus/actor1')
            alias=owner['matching_origin_rows'][0]['canonical_relation']
            self.assertFalse(alias['raw_state_equal'])
            self.assertFalse(alias['player_swap'])
            self.assertEqual(alias['store_offsets'],[-3,3])

    def test_late_band_does_not_release_predeclared_hold(self):
        r=self.report
        self.assertEqual(r['direct_bridge_bands']['active_min'],5)
        self.assertEqual(r['direct_bridge_bands']['active_max'],5)
        self.assertEqual(r['historical_bridge_bands']['active_max'],16)
        for key in ('support_hit_bands','direct_bridge_bands','historical_bridge_bands'):
            self.assertEqual(r[key]['in_updated_branch'],0)
        self.assertTrue(r['ownership_hold_remains'])
        self.assertFalse(r['ownership']['ownership_passed'])
        for key in ('strength_interpretation_allowed','posthoc_filtering','posthoc_cluster_merging','bootstrap_performed'):
            self.assertFalse(r[key])

    def test_raw_corruption_refused_before_literal_replay(self):
        task=copy.deepcopy(self.plan['tasks'][0]);task['decisions'][0]['sha256']='0'*64
        with patch.object(diagnostic.arena,'transition',side_effect=AssertionError('forbidden replay after source corruption')) as replay:
            with self.assertRaisesRegex(ValueError,'Bridge raw source changed'):
                diagnostic.unit(task)
            replay.assert_not_called()

    def test_duplicate_or_source_changed_unit_refused(self):
        bad=list(self.results);bad[-1]=bad[-2]
        with self.assertRaisesRegex(ValueError,'duplicate/missing'):
            diagnostic.summarize(bad)
        bad=list(self.results);bad[0]=dict(bad[0],task_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'unit source/task changed'):
            diagnostic.summarize(bad)

    def test_historical_witnesses_are_partial_and_exact_source_bound(self):
        r=self.report;giant=r['historical_component_bridges'][0]
        self.assertEqual((giant['component_families'],giant['component_quotients'],giant['observations']),(369,23976,327))
        self.assertEqual(giant['family_indices'],list(range(24)))
        self.assertEqual(len(giant['sampled_cross_old_origin_collisions']),15)
        self.assertFalse(r['historical_provenance']['exhaustive'])
        for source in r['historical_provenance']['sources']:
            self.assertEqual(source['sampled_saved_games'],8)
            self.assertFalse(source['exhaustive_historical_graph_reconstruction'])
            self.assertEqual(r['inputs'][source['source']],source['source_sha256'])
            for edge in source['edges']:
                self.assertEqual(len(edge['saved_prefix_actions']),edge['choice'])


if __name__=='__main__':unittest.main()
