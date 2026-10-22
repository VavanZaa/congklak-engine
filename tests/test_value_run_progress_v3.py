"""Observer metadata tests; no HTTP, Torch, writer or engine work."""
import copy
import sys
import unittest
from unittest import mock
from congklak import value_run_progress_v3 as progress, value_training as training


class ProgressTests(unittest.TestCase):
    def state(self):
        return dict(schema='value-run-stage-status-v1', state='running', message='Testing pilot',
            cards=[dict(id='training' if n == 3 else str(n), title=str(n), state='running', result='pending') for n in range(9)])

    def reader(self, path):
        if path == progress.FOLDER / 'stage-status.json': return copy.deepcopy(self.state())
        return {'cards': []}

    def test_committed_counters_and_readonly(self):
        with mock.patch.object(progress, 'read', side_effect=self.reader), mock.patch.object(progress, 'optional', return_value={}), mock.patch.object(progress, 'arena_live', return_value={}), mock.patch.object(training, 'status', return_value=dict(state='paused', steps=20, draws=2560)):
            body = progress.snapshot()
        self.assertEqual(body['current_cards'], 9)
        self.assertIn('20 / 400', body['tools'][3]['result'])
        self.assertEqual(body['production']['draws'], 2560)
        self.assertFalse(body['accepted'])
        self.assertNotIn('torch', sys.modules)

    def test_invalid_production_evidence_never_becomes_zero(self):
        with mock.patch.object(progress, 'read', side_effect=self.reader), mock.patch.object(progress, 'optional', return_value={}), mock.patch.object(progress, 'arena_live', return_value={}), mock.patch.object(training, 'status', side_effect=training.TrainingError('bad checkpoint')):
            body = progress.snapshot()
        self.assertEqual(body['current_work']['state'], 'hold')
        self.assertIsNone(body['production']['steps'])
        self.assertFalse(body['pilot_promising'])

    def test_stale_source_holds(self):
        with mock.patch.object(progress, 'read', side_effect=self.reader), mock.patch.object(progress, 'optional', return_value={}), mock.patch.object(progress, 'arena_live', return_value={}), mock.patch.object(training, 'status', return_value=dict(state='complete', steps=400, draws=51200)), mock.patch.object(progress, 'LOADED_SHA', 'stale'):
            body = progress.snapshot()
        self.assertEqual(body['current_work']['state'], 'hold')
        self.assertFalse(body['evidence_accepted'])


if __name__ == '__main__':
    unittest.main()
