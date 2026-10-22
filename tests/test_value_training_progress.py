"""Observer acceptance and refusal paths, with no HTTP or training runtime."""
import copy
import sys
import unittest
from unittest import mock
from congklak import value_training_progress as progress


class ProgressTests(unittest.TestCase):
    def state(self):
        return dict(schema='value-training-implementation-status-v1', state='running',
                    production_counters=progress.ZERO, message='fixture verification',
                    cards=[dict(id=str(n), title=str(n), state='planned', result='pending') for n in range(10)])

    def test_metadata_only_snapshot(self):
        original = self.state()
        def read(path):
            return copy.deepcopy(original) if path.name == 'stage-status.json' and path.parent == progress.FOLDER else {'cards': []}
        with mock.patch.object(progress, 'read', side_effect=read):
            body = progress.snapshot()
        self.assertEqual(body['production_counters'], progress.ZERO)
        self.assertEqual(body['current_cards'], 10)
        self.assertEqual(body['tools'][0]['state'], 'planned')
        self.assertEqual(original['cards'][0]['state'], 'planned')
        self.assertNotIn('torch', sys.modules)

    def test_production_work_cannot_be_reported_as_zero(self):
        body = self.state(); body['production_counters'] = dict(progress.ZERO, training_steps=1)
        with mock.patch.object(progress, 'read', return_value=body):
            with self.assertRaisesRegex(ValueError, 'unexpected production'):
                progress.snapshot()

    def test_changed_evidence_is_hold(self):
        body = self.state(); body['cards'][0].update(state='complete', inputs={'missing': 'bad'})
        def read(path):
            return copy.deepcopy(body) if path.name == 'stage-status.json' and path.parent == progress.FOLDER else {'cards': []}
        with mock.patch.object(progress, 'read', side_effect=read):
            result = progress.snapshot()
        self.assertFalse(result['accepted'])
        self.assertEqual(result['current_work']['state'], 'hold')
        self.assertEqual(result['tools'][0]['state'], 'hold')


if __name__ == '__main__':
    unittest.main()
