"""Observer acceptance: completion must track bound bytes and do no work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from congklak import valuepilot_interface_progress as progress


class InterfaceObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=progress.FOLDER)
        self.folder = Path(self.temp.name)
        self.status = json.loads((progress.FOLDER / 'stage-status.json').read_text(encoding='utf-8'))
        self.write('stage-status.json', self.status)
        self.patcher = patch.object(progress, 'FOLDER', self.folder)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def write(self, name, body):
        (self.folder / name).write_text(json.dumps(body, sort_keys=True), encoding='utf-8')

    def seal(self, training_ready=True):
        relative = (self.folder / 'stage-status.json').relative_to(progress.ROOT).as_posix()
        checksum = hashlib.sha256((self.folder / 'stage-status.json').read_bytes()).hexdigest()
        self.write('acceptance.json', dict(passed=True, new_inputs={relative: checksum}, training_ready=training_ready,
                                          next_step='Ready' if training_ready else 'HOLD: insufficient ordinary arena coverage',
                                          readiness_blockers=[] if training_ready else ['ordinary arena shortage']))

    def test_completed_snapshot_is_read_only_and_has_no_execution_runtime(self):
        self.seal()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        modules = set(sys.modules)
        result = progress.snapshot()
        self.assertTrue(result['implementation_accepted'])
        self.assertEqual(result['current_work']['state'], 'complete')
        self.assertFalse(result['active_tools'])
        self.assertEqual(result['execution_counters'], dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0))
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        self.assertEqual(before, after)
        self.assertFalse(any(name == 'torch' or name.startswith('torch.') for name in set(sys.modules) - modules))

    def test_changed_accepted_status_holds_completion(self):
        self.seal()
        altered = deepcopy(self.status)
        altered['message'] = 'Changed after acceptance'
        self.write('stage-status.json', altered)
        result = progress.snapshot()
        self.assertFalse(result['implementation_accepted'])
        self.assertEqual(result['current_work']['state'], 'hold')

    def test_engineering_acceptance_does_not_override_data_hold(self):
        self.seal(training_ready=False)
        result = progress.snapshot()
        self.assertTrue(result['implementation_accepted'])
        self.assertFalse(result['pilot_training_ready'])
        self.assertEqual(result['current_work']['state'], 'hold')
        self.assertEqual(result['readiness_blockers'], ['ordinary arena shortage'])

    def test_missing_stage_evidence_holds_that_card(self):
        altered = deepcopy(self.status)
        altered['cards'][1].update(state='complete', inputs={'build/valuepilot-interface-v1/missing-evidence.json': '0' * 64})
        self.write('stage-status.json', altered)
        result = progress.snapshot()
        self.assertEqual(result['tools'][1]['state'], 'hold')

    def test_execution_counter_and_stage_count_fail_closed(self):
        altered = deepcopy(self.status)
        altered['execution_counters']['training_steps'] = 1
        self.write('stage-status.json', altered)
        with self.assertRaisesRegex(ValueError, 'cannot certify'):
            progress.snapshot()
        altered = deepcopy(self.status)
        altered['cards'].pop()
        self.write('stage-status.json', altered)
        with self.assertRaisesRegex(ValueError, 'nine pilot stages'):
            progress.snapshot()


if __name__ == '__main__':
    unittest.main()
