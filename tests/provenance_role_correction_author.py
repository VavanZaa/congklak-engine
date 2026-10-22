"""No-replay acceptance for the fixed512 support training-marker correction."""
import unittest
from congklak import provenance_reconstruction_v2 as correction


class RoleCorrectionAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=correction.read(correction.journal.DEFAULT/'report.json')
        cls.corrected=correction.corrected(cls.original)

    def test_01_only_declared_training_cache_markers_change(self):
        old=self.original['marker_ledger'];new=self.corrected['marker_ledger']
        changes={r['marker_index']:r for r in self.corrected['role_corrections']}
        self.assertEqual(len(changes),7672)
        for index,(before,after) in enumerate(zip(old,new)):
            if index not in changes:self.assertEqual(before,after);continue
            self.assertIsNotNone(correction.training_group(before))
            self.assertFalse(after['optimizer_blocked'])
            self.assertTrue(after['ancestral_training'])
            for key in ('fit_blocked','arena_blocked','acquisition_exposed','witness','kind','value'):
                self.assertEqual(before[key],after[key])

    def test_02_true_eval_protections_remain_and_support_not_silently_broadened(self):
        rows=self.corrected['candidate_pools']['support_reuse']
        self.assertEqual(len(rows),512)
        self.assertEqual(self.corrected['support_roots_with_genuine_exclusion'],320)
        self.assertEqual(sum(r['eligible'] for r in rows),192)
        self.assertFalse(self.corrected['support_training_admission'])
        sample=next(r for r in rows if r['family']=='constructed-ordinary/20261015/4243')
        self.assertTrue(sample['eligible'])
        self.assertFalse(self.corrected['larger_training_pool_authorized'])
        self.assertFalse(self.corrected['training_authorized'])

    def test_03_no_raw_graph_partition_or_literal_work_changes(self):
        for key in ('family_components','quotient_components','actual_incidence_occurrences',
                    'distinct_family_raw_incidences','shared_incidence_occurrences',
                    'owned_incidence_occurrences','literal_replayed_actions','opaque_legacy_unions'):
            self.assertEqual(self.original[key],self.corrected[key])
        self.assertEqual(set(self.original['components']),set(self.corrected['components']))
        self.assertFalse(self.corrected['raw_incidence_and_component_partition_changed'])
        self.assertTrue(self.corrected['prior_known_fit_arena_forensic_protections_preserved'])


if __name__=='__main__':unittest.main()
