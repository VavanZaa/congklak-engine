"""Joint reservations must protect the complete 32-origin development pool."""
import copy
import unittest
from unittest.mock import patch
from congklak import valuepilot_reservations_v2 as module
from congklak import ordinary_development as development


GENERATED = module.ROOT / 'build/ordinary-development-v1/data-owner-fixture-v5'
FIXTURE = module.ROOT / 'build/ordinary-development-v1/data-owner-reservations-v5'


class ReservationsV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        development.prepare(GENERATED); development.run(GENERATED)
        module.prepare(FIXTURE, GENERATED); module.run(FIXTURE)

    def test_joint_quotas_distinct664_and_all32_protected(self):
        manifest = module.admit(FIXTURE)
        self.assertEqual(manifest['counts'], dict(train=512, fit_check=128, fit_cases=32, arena=24, globally_distinct_components=664))
        inv, selected, _ = module.prepared(FIXTURE)
        generated = set(inv['generated_development']['accepted_component_ids'])
        used = {r['component'] for r in selected['arena'] if r['stratum'].endswith('/ordinary')}
        self.assertEqual(len(generated), 32); self.assertEqual(len(used), 8)
        self.assertEqual(len(generated - used), 24)
        for cid in generated:
            self.assertIn('optimizer/generated_development_only', inv['components'][cid]['reasons'])
        self.assertEqual(len(module.resolve_positions(FIXTURE)), 664)

    def test_metadata_matching_is_deterministic_and_has_no_targets(self):
        inv, selection, _ = module.prepared(FIXTURE)
        self.assertEqual(selection, module.select(copy.deepcopy(inv)))
        self.assertEqual(selection['selection_salt'], 'neural-value-pilot/v1/data/')
        for key in ('target_bodies_read', 'model_forwards', 'searches', 'optimizer_steps', 'new_games'):
            self.assertEqual(selection[key], 0)
        self.assertFalse(selection['fit_check_is_unseen'])

    def test_generated_origin_overlap_is_fail_closed(self):
        parent = module.read(GENERATED / 'parent-inventory.json')
        pool = module.read(GENERATED / 'candidates.json')['accepted']
        changed = copy.deepcopy(pool); changed[1] = copy.deepcopy(changed[0])
        with self.assertRaisesRegex(ValueError, 'family duplicates'):
            module._generated(parent, changed, module.old.Graph())

    def test_completed_routes_do_not_rebuild_generate_or_lock(self):
        with patch.object(module, 'Writer', side_effect=AssertionError('writer')), \
             patch.object(module, 'inventory', side_effect=AssertionError('inventory')), \
             patch.object(module.old, 'inventory', side_effect=AssertionError('old inventory')), \
             patch.object(development, '_facts', side_effect=AssertionError('generation')):
            result = module.run(FIXTURE)
            self.assertEqual(result, module.verify(FIXTURE))
            self.assertEqual(module.prepare(FIXTURE), result)
            self.assertTrue(module.status(FIXTURE)['admitted'])

    def test_old_schema_and_mutated_generated_protection_refuse(self):
        inv, selection, _ = module.prepared(FIXTURE)
        bad = copy.deepcopy(selection); bad['schema'] = module.old.SCHEMA + '-selection'
        with self.assertRaisesRegex(ValueError, 'V2 selection schema'): module.validate_selection(bad, inv)
        altered = copy.deepcopy(inv); cid = inv['generated_development']['accepted_component_ids'][0]
        altered['components'][cid]['reasons'] = []
        with self.assertRaises(ValueError): module.validate_selection(selection, altered)


if __name__ == '__main__': unittest.main()
