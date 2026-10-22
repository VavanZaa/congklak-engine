"""Read-only progress must separate data admission from training execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from congklak import ordinary_development_progress as progress


class OrdinaryDevelopmentObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=progress.FOLDER)
        self.folder = Path(self.temp.name)
        self.state = json.loads((progress.FOLDER / 'stage-status.json').read_text())
        self.write('stage-status.json', self.state)
        self.patcher = patch.object(progress, 'FOLDER', self.folder)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def write(self, name, body):
        (self.folder / name).write_text(json.dumps(body), encoding='utf-8')

    def seal(self):
        path = self.folder / 'stage-status.json'
        self.write('acceptance.json', dict(passed=True,
            live_inputs={path.relative_to(progress.ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()},
            next_step='Data and inference admitted; training remains off.', next_operation='Implement recoverable trainer'))

    def test_completed_snapshot_read_only_and_no_execution_runtime(self):
        self.seal()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        modules = set(sys.modules)
        result = progress.snapshot()
        self.assertTrue(result['accepted'])
        self.assertTrue(result['data_and_inference_ready'])
        self.assertEqual(result['execution_counters'], progress.ZERO)
        self.assertEqual(result['current_work']['state'], 'complete')
        self.assertFalse(result['active_tools'])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()})
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in set(sys.modules) - modules))

    def test_modified_acceptance_binding_holds(self):
        self.seal()
        self.state['message'] = 'Changed after acceptance'
        self.write('stage-status.json', self.state)
        self.assertEqual(progress.snapshot()['current_work']['state'], 'hold')
        self.assertFalse(progress.snapshot()['accepted'])

    def test_missing_complete_card_evidence_holds_entire_work(self):
        self.state['cards'][1].update(state='complete', inputs={'build/ordinary-development-v1/missing-proof.json': '0' * 64})
        self.write('stage-status.json', self.state)
        self.seal()
        result = progress.snapshot()
        self.assertEqual(result['tools'][1]['state'], 'hold')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['current_work']['state'], 'hold')

    def test_loaded_source_drift_requires_observer_restart(self):
        self.seal()
        with patch.object(progress, 'LOADED_SHA', '0' * 64):
            result = progress.snapshot()
        self.assertFalse(result['observer_source_current'])
        self.assertEqual(result['current_work']['state'], 'hold')
        self.assertFalse(result['accepted'])

    def test_nonzero_optimizer_or_game_work_refuses(self):
        for key in progress.ZERO:
            changed = json.loads(json.dumps(self.state))
            changed['execution_counters'][key] = 1
            self.write('stage-status.json', changed)
            with self.assertRaisesRegex(ValueError, 'unexpected training/game work'):
                progress.snapshot()

    def test_card_count_and_unknown_schema_refuse(self):
        self.state['cards'].pop()
        self.write('stage-status.json', self.state)
        with self.assertRaisesRegex(ValueError, 'ten current stages'):
            progress.snapshot()
        self.state['schema'] = 'unknown'
        self.write('stage-status.json', self.state)
        with self.assertRaisesRegex(ValueError, 'Unknown pre-training status'):
            progress.snapshot()


if __name__ == '__main__':
    unittest.main()
