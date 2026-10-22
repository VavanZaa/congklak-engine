"""Focused mathematical and source-bound tests for fixed value diagnostics."""
import copy
import math
import unittest
from unittest import mock
from congklak import value_capture_diagnostic as d


class CaptureDiagnosticTests(unittest.TestCase):
    def test_literal_forward_and_bias_attribution(self):
        base = {'layers.0.weight': [[2., -1.]], 'layers.0.bias': [-.2],
                'layers.2.weight': [[3.]], 'layers.2.bias': [.4],
                'layers.4.weight': [[.5]], 'layers.4.bias': [-.3],
                'policy.weight': [[2.], [-2.]], 'policy.bias': [1., -1.]}
        branch = {'A': [[1., 2.], [-1., -2.]], 'b': [0., 0.],
                  'v': [[.1, 9.]], 'c': [.2]}
        r = d.forward([.5, .1], base, branch)
        expected_u = .5 * (3 * (.9 - .2) + .4) - .3
        self.assertAlmostEqual(r['base_logit'], expected_u)
        self.assertEqual(r['hidden_activations'], [.7, 0.])
        self.assertAlmostEqual(r['correction_logit'], .27)
        self.assertAlmostEqual(sum(r['hidden_contributions']) + r['branch_bias'], r['correction_logit'])
        self.assertAlmostEqual(r['candidate_z'], math.tanh(expected_u + .27))

    def test_huber_boundary_and_signed_score_regret(self):
        self.assertAlmostEqual(d.huber(.1), .005)
        self.assertAlmostEqual(d.huber(.2), .015)
        self.assertEqual(d.huber(-.2), d.huber(.2))
        actor1q = {0: -12, 1: 13, 3: -6}
        y = min(actor1q.values())
        self.assertEqual(-1 * (y - actor1q[3]), 6)

    def test_actual_plan_membership_frozen_models_and_unknown_task(self):
        p = d.plan()
        self.assertEqual(len(p['tasks']), 21)
        self.assertEqual(p['counters']['cached_rows'], 640)
        self.assertEqual(len(p['tasks'][-1]['records']), 192)
        self.assertEqual(p['inputs'][d.CONTROL], d.CONTROL_SHA)
        self.assertEqual(p['inputs'][d.CANDIDATE], d.CANDIDATE_SHA)
        broken = dict(p['tasks'][0], stop=31)
        with self.assertRaisesRegex(ValueError, 'modified capture diagnostic task'):
            d.unit(broken)

    def test_actual_row_score_perspective_capture_flags_and_no_features(self):
        base, branch = d._models()
        rows = d.read(d.CACHE + 'fit-check.json')['rows']
        training = d.read(d.CACHE + 'train.json')['rows']
        row = next(r for r in rows if r['actor'] == 1 and r['stratum'].endswith('/capture'))
        result = d.describe(row, base, branch, training)
        self.assertEqual(result['predictions']['candidate']['actor_seed_error'],
                         -result['predictions']['candidate']['seed_error'])
        self.assertTrue(result['own_capture_actions'])
        self.assertEqual(result['target_player0'], min(a['value_player0'] for a in row['target_source']['action_q']))
        damaged = copy.deepcopy(row); damaged['target_player0'] += 1
        with self.assertRaisesRegex(ValueError, 'differently oriented H8 target'):
            d.describe(damaged, base, branch, training)
        damaged = copy.deepcopy(row); damaged['r'] *= -1
        with self.assertRaisesRegex(ValueError, 'score perspective'):
            d.describe(damaged, base, branch, training)

    def test_partial_or_duplicate_assembly_refuses(self):
        with self.assertRaisesRegex(ValueError, 'Missing, reordered or duplicated'):
            d.summarize([])


if __name__ == '__main__':
    unittest.main()
