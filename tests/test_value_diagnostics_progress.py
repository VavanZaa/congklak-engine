import unittest
from unittest.mock import patch
from congklak import value_diagnostics_progress as progress


class DiagnosticProgressTests(unittest.TestCase):
    def test_observer_reads_immutable_events_and_history_only_before_completion(self):
        original = progress.read
        opened = []
        def observed(path):
            opened.append(str(path).replace('\\', '/'))
            return original(path)
        with patch.object(progress, 'read', side_effect=observed):
            result = progress.snapshot()
        self.assertTrue(result['models_fixed'] and result['policy_fixed'])
        self.assertTrue(result['candidate_quarantined'])
        self.assertFalse(result['strength_interpretation_allowed'])
        self.assertTrue(all(v == 0 for v in result['counters'].values()))
        self.assertFalse(any(p.endswith(('/head.json', '/progress.json', '/ledger.json')) for p in opened))

    def test_loaded_source_difference_holds(self):
        with patch.object(progress, 'LOADED_SHA', '0' * 64):
            result = progress.snapshot()
        self.assertFalse(result['accepted'])
        self.assertEqual(result['current_work']['state'], 'hold')

    def test_bound_acceptance_difference_holds(self):
        original = progress.read
        def observed(path):
            if str(path).endswith('acceptance.json'):
                return dict(passed=True, live_inputs={'fake': '0' * 64})
            return original(path)
        real_file = progress.Path.is_file
        def exists(path):
            return True if path == progress.FOLDER / 'acceptance.json' else real_file(path)
        with patch.object(progress, 'read', side_effect=observed), patch.object(progress.Path, 'is_file', exists), patch.object(progress, 'current', return_value=False):
            result = progress.snapshot()
        self.assertFalse(result['accepted'])
        self.assertEqual(result['current_work']['state'], 'hold')


if __name__ == '__main__': unittest.main()
