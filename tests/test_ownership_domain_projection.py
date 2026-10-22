"""Read-only mass/domain feasibility acceptance for the prospective protocol."""
import copy
import sys
import unittest
from unittest.mock import patch
from congklak import ownership_domain_projection as projection


def state(active, actor=0, stores=None):
    cells = [active // 14] * 14
    for i in range(active % 14): cells[i] += 1
    return dict(pits=[cells[:7], cells[7:]],
                stores=[0, 98-active] if stores is None else stores, controller=actor)


class OwnershipDomainProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = projection.plan()
        cls.units = [projection.unit(task) for task in cls.plan['tasks']]
        cls.report = projection.summarize(cls.units)

    def test_literal_mass_checks_cover_all_current_decisions(self):
        r = self.report
        self.assertEqual((r['units'], r['existing_saved_games_replayed'],
                          r['literal_replayed_actions'], r['observed_complete_states']),
                         (193, 192, 6663, 6855))
        self.assertEqual(r['mass_increases'], 0)
        self.assertEqual(r['shared_to_owned_reentries'], 0)
        self.assertEqual(r['equal_active_mass'] + r['decreasing_active_mass'], 6663)
        self.assertEqual(sum(r['state_band_counts'].values()), 6855)
        self.assertTrue(r['all_complete_transitions_checked'])
        self.assertNotIn('torch', sys.modules)
        for name in projection.ZERO: self.assertEqual(r[name], 0)

    def test_upper_frozen_nodes_still_owned_and_inventory_does_not_grant_admission(self):
        inv = self.report['inventory_projection']
        self.assertEqual(inv['full_inventory_components'], 21073)
        self.assertEqual(inv['unique_quotients']['total'], 50904)
        self.assertEqual(inv['unique_quotients']['owned_ge25'], 25168)
        self.assertEqual(inv['unique_quotients']['bands']['owned_frozen_upper/75-98'], 988)
        self.assertEqual(inv['selected_frozen_role_alias_coverage']['train']['alias_quotients']['owned_ge25'], 751)
        self.assertEqual(inv['selected_frozen_role_alias_coverage']['fit_check']['alias_quotients']['owned_ge25'], 186)
        self.assertFalse(inv['family_to_quotient_incidence_present'])
        self.assertFalse(inv['complete_incidence_rebuilt'])
        self.assertIsNone(inv['fresh_domain_components'])
        self.assertFalse(inv['production_admission'])
        self.assertTrue(inv['old_component_ids_cannot_be_reused_after_domain_cut'])
        self.assertTrue(inv['role_exclusions_cannot_be_dropped_without_source_provenance'])

    def test_collision_domain_counts_preserve_old_hold(self):
        inv = self.report['inventory_projection']
        c = inv['existing_arena_collision_domain_counts']
        self.assertEqual(c['training_alias_hits']['observations'], 2)
        self.assertEqual(c['direct_cross_family']['unique_quotients'], 12)
        self.assertEqual(c['historical_component']['observations'], 337)
        for row in c.values(): self.assertEqual(row['owned_ge25'], 0)
        self.assertTrue(self.report['existing_ownership_hold_remains'])
        for name in ('production_admission', 'strength_interpretation_allowed', 'training_authorized'):
            self.assertFalse(self.report[name])

    def test_quotient_discards_store_offset_and_swaps_player_sides(self):
        p = state(25, 1, [30, 43]); q = projection.metadata.quotient(p)
        shifted = copy.deepcopy(p); shifted['stores'] = [31, 42]
        swapped = dict(pits=list(reversed(p['pits'])), stores=list(reversed(p['stores'])), controller=0)
        self.assertEqual(q, projection.metadata.quotient(shifted))
        self.assertEqual(q, projection.metadata.quotient(swapped))
        self.assertEqual(projection.quotient_active(q), 25)
        for mass, wanted in ((0, 'terminal/0'), (24, 'shared_frozen/1-24'),
                             (25, 'owned_updated/25-74'), (74, 'owned_updated/25-74'),
                             (75, 'owned_frozen_upper/75-98'), (98, 'owned_frozen_upper/75-98')):
            self.assertEqual(projection.band(projection.quotient_active(projection.metadata.quotient(state(mass)))), wanted)

    def test_quotient_and_increasing_mass_counterexamples_fail(self):
        for bad in ('00', 'zz', '00'*14+'01', 'ff'*14+'00'):
            with self.assertRaises(ValueError): projection.quotient_active(bad)
        # Total seeds remain 98 in both states; a hypothetical store withdrawal
        # would make the low-domain partition unsafe, and must be rejected.
        with self.assertRaisesRegex(ValueError, 'mass increased'):
            projection.mass_transition(state(24), state(25))
        self.assertEqual(projection.mass_transition(state(25), state(24)), (25, 24))

    def test_bad_raw_pin_refused_before_literal_transition(self):
        task = copy.deepcopy(self.plan['tasks'][0]); task['decisions'][0]['sha256'] = '0'*64
        with patch.object(projection.arena, 'transition', side_effect=AssertionError('forbidden replay')) as replay:
            with self.assertRaisesRegex(ValueError, 'saved raw source changed'):
                projection.unit(task)
            replay.assert_not_called()

    def test_summary_rejects_duplicate_unit_and_changed_task(self):
        units = list(self.units); units[-1] = units[-2]
        with self.assertRaisesRegex(ValueError, 'duplicate/missing'):
            projection.summarize(units)
        units = list(self.units); units[0] = dict(units[0], task_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'source/task differs'):
            projection.summarize(units)


if __name__ == '__main__': unittest.main()
