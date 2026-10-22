"""Bounded source witnesses and graph-cut counterexamples; no model work."""
import copy
import unittest
from unittest.mock import patch
from congklak import provenance_reconstruction as p


def state(active,actor=0):
    cells=[0]*14;cells[0]=active//2;cells[7]=active-active//2
    return dict(pits=[cells[:7],cells[7:]],stores=[0,98-active],controller=actor)


class ProvenanceAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=p.plan()
        cls.origin_task=next(t for t in cls.plan['tasks'] if t['kind']=='dataset_origins')
        cls.origin_unit=p.unit(cls.origin_task)
        cls.opening_task=next(t for t in cls.plan['tasks'] if t['kind']=='dataset_families' and t['dataset']=='pilot-10000' and t['offset']==512)
        cls.opening_unit=p.unit(cls.opening_task)

    def test_01_source_plan_bounds_and_fixed_models(self):
        self.assertEqual(len(self.plan['tasks']),354)
        self.assertEqual([t['index'] for t in self.plan['tasks']],list(range(354)))
        self.assertTrue(set(p.common.pins(p.MODELS)) <= set(self.plan['inputs']))
        self.assertEqual(set(t['kind'] for t in self.plan['tasks']),
                         {'dataset_families','dataset_origins','journal','historical_arena',
                          'recorded_search_trace','current_arena','json_metadata'})
        self.assertFalse(self.plan['no_target_or_model_access'] is False)

    def test_02_actual_sqlite_incidence_has_lossless_raw_witness(self):
        unit=self.origin_unit
        self.assertEqual((unit['source_rows'],len(unit['edges'])),(512,512))
        for edge in unit['edges']:
            witness=edge['witness'];self.assertEqual(witness['path'],self.origin_task['source'])
            self.assertEqual(witness['sha256'],self.origin_task['source_sha256'])
            self.assertEqual(witness['table'],'origins JOIN candidates')
            self.assertEqual(witness['key'][1],edge['family'])
            self.assertEqual(witness['key'][0],edge['raw_state_key'])
            self.assertEqual(edge['active_seeds'],p.domain.quotient_active(edge['quotient']))
            self.assertEqual(len(bytes.fromhex(edge['raw_state_key'])),17)

    def test_03_public_opening_is_preserved_as_high_ancestry(self):
        opening=p.metadata.quotient(p.config.position())
        occurrences=[e for e in self.opening_unit['edges'] if e['quotient']==opening]
        self.assertGreater(len(occurrences),1)
        self.assertGreater(len({e['family'] for e in occurrences}),1)
        for edge in occurrences:
            self.assertEqual(edge['active_seeds'],98)
            self.assertEqual(edge['witness']['trajectory_choice'],0)
            self.assertEqual(edge['domain'],'owned')
        self.assertGreater(self.opening_unit['literal_replayed_actions'],0)

    def test_04_corrupt_source_is_refused_before_literal_load(self):
        task=copy.deepcopy(self.origin_task);task['source_sha256']='0'*64
        with patch.object(p.arena,'_literal',side_effect=AssertionError('forbidden literal after bad source')) as literal:
            with self.assertRaisesRegex(ValueError,'task source changed'):p.unit(task)
            literal.assert_not_called()

    def _fixture(self,missing=False,ancestry=False,low_marker=False):
        low,high,upper=[state(a) for a in (5,25,75)]
        qlow,qhigh,qupper=[p.metadata.quotient(s) for s in (low,high,upper)]
        witness=dict(path='scratch-source',sha256='a'*64,pointer='/fixture')
        edges=[p._edge('a',high,witness),p._edge('a',low,witness),p._edge('b',upper,witness)]
        if not missing:edges.append(p._edge('b',low,witness))
        markers=[]
        if ancestry:markers.append(p._marker('family','a',witness,'ancestor only',ancestral_training=True))
        if low_marker:markers.append(p._marker('family','a',witness,'protected origin observed low',fit=True,arena=True))
        task=dict(index=0,kind='scratch',source='scratch-source')
        unit=dict(schema=p.SCHEMA+'-unit',index=0,kind='scratch',task_sha256=p.config.digest(task),
                  source_rows=1,literal_replayed_actions=0,edges=edges,markers=markers,origin_vertices=[],**p.ZERO)
        inv=dict(components={'old':dict(families=['a','b'],quotients=[qlow,qhigh,qupper],reasons=[])},primary_pool=[],arena_pool=[])
        plan=dict(tasks=[task],inputs={'scratch-source':'a'*64},source_catalog=['scratch-source'])
        with patch.object(p,'plan',return_value=plan),patch.object(p,'read',side_effect=lambda path:inv if path==p.INVENTORY else {'train':[]}):
            return p.summarize([unit])

    def test_05_actual_low_bridge_cut_preserves_upper_nodes(self):
        report=self._fixture()
        self.assertEqual(len(report['components']),2)
        self.assertEqual(report['legacy_components_reconstructed_from_actual_incidence'],1)
        self.assertEqual(report['opaque_legacy_unions'],[])
        self.assertNotEqual(report['family_components']['a'],report['family_components']['b'])
        upper=p.metadata.quotient(state(75))
        self.assertIn(upper,report['quotient_components'])
        self.assertEqual(report['shared_incidence_occurrences'],2)
        self.assertFalse(report['production_admission'])

    def test_06_missing_incidence_retains_conservative_union_not_fake_edges(self):
        report=self._fixture(missing=True)
        self.assertEqual(len(report['components']),1)
        self.assertEqual(len(report['opaque_legacy_unions']),1)
        self.assertFalse(report['opaque_legacy_unions'][0]['actual_incidence'])
        self.assertEqual(report['cartesian_flattened_edges_created'],0)
        self.assertEqual(report['actual_incidence_occurrences'],3)
        self.assertEqual(report['family_components']['a'],report['family_components']['b'])

    def test_07_ancestral_training_separate_and_protected_origin_survives_cut(self):
        report=self._fixture(ancestry=True,low_marker=True)
        a=report['components'][report['family_components']['a']]
        b=report['components'][report['family_components']['b']]
        self.assertTrue(a['ancestral_training'])
        self.assertFalse(a['optimizer_blocked'])
        self.assertTrue(a['fit_blocked']);self.assertTrue(a['arena_blocked'])
        self.assertFalse(b['fit_blocked']);self.assertFalse(b['arena_blocked'])
        self.assertFalse(report['historically_unseen_claim'])


if __name__=='__main__':unittest.main()
