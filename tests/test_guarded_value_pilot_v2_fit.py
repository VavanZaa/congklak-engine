"""Numerical units and completed-reader dispatch; synthetic numbers only."""
import unittest
from pathlib import Path
from unittest.mock import patch
from congklak import guarded_value_pilot_v2_fit as fit


class GuardedValueFitTests(unittest.TestCase):
    def test_native_accumulation_order_and_seed_units(self):
        self.assertEqual(fit._dot([1e16, 1., -1e16], [1., 1., 1.]), 0.)
        rows = [dict(target_z=0., target_player0=0, control=dict(z=.2, value_player0=10.))]
        metrics = fit._metrics(rows, 'control')
        self.assertAlmostEqual(metrics['normalized_huber'], .015)
        self.assertEqual(metrics['seed_mae'], 10.)
        self.assertEqual(metrics['seed_bias_player0'], 10.)

    def test_exact_frozen_numeric_helpers(self):
        self.assertIs(fit._base, fit.arithmetic._base)
        self.assertIs(fit._correction, fit.arithmetic._correction)
        self.assertIs(fit._metrics, fit.arithmetic._metrics)
        branch = dict(A=[[1., 1.]], b=[0.], v=[[0.]], c=[0.])
        self.assertEqual(fit._correction([2., 3.], branch), 0.)

    def test_completed_read_dispatch_does_not_create_a_writer(self):
        with patch.object(fit, 'completed', return_value=dict(passed=False)), \
             patch.object(fit, '_derive', side_effect=AssertionError('Producer entered')), \
             patch.object(fit, 'Writer', side_effect=AssertionError('Writer entered')), \
             patch.object(Path, 'exists', return_value=True):
            self.assertEqual(fit.run(), dict(passed=False))


if __name__ == '__main__':
    unittest.main()
