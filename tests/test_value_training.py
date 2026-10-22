"""Standard-library owner checks; real interrupted optimizers are audited separately."""
import builtins
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from congklak import value_training as train

WORK = train.ROOT / 'build/value-training-v1/trainer-owner-unit-fixtures'


class ValueTrainingTests(unittest.TestCase):
    def test_production_settings_are_fixed(self):
        s = train.settings()
        self.assertEqual((s['steps'], s['batch_size'], s['roots'], s['checkpoint_interval']), (400, 128, 512, 20))
        self.assertEqual(s['steps'] * s['batch_size'], 51200)
        self.assertEqual(s['betas'], [.9, .999]); self.assertFalse(s['foreach']); self.assertFalse(s['fused'])
        self.assertEqual(s['gate'], [25, 74]); self.assertEqual(s['huber_delta'], .1)
        with self.assertRaises(train.TrainingError): train.settings(1)

    def test_fixture_is_distinct_fixed_six_step_support(self):
        s = train.settings(True)
        self.assertEqual((s['steps'], s['roots'], s['checkpoint_interval']), (6, 32, 2))
        self.assertEqual(s['batch_size'], 128)
        self.assertEqual(train._path(WORK, True), WORK)
        with self.assertRaises(train.TrainingError): train._path(train.DEFAULT, True)
        with self.assertRaises(train.TrainingError): train._path(WORK, False)

    def test_completed_dispatch_precedes_writer_optimizer_and_lease(self):
        WORK.mkdir(parents=True, exist_ok=True)
        path = WORK / 'reader-dispatch'; path.mkdir(exist_ok=True)
        train.atomic(path / 'COMPLETED.json', {'certificate_sha256': 'fixture'})
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        expected = dict(state='complete', steps=6)
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch' or name.endswith('value_training_resources'):
                raise AssertionError('Forbidden completed import: ' + name)
            return original(name, *args, **kwargs)
        with patch.object(train, 'completed', return_value=expected), patch.object(train, '_identity', return_value={}), \
             patch.object(train, 'Writer', side_effect=AssertionError('writer touched')), patch('builtins.__import__', side_effect=guarded):
            self.assertEqual(train.run(path, fixture=True), expected)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_incomplete_and_corrupt_completion_refuse_without_torch(self):
        path = WORK / 'missing-completion-evidence'; path.mkdir(parents=True, exist_ok=True)
        train.atomic(path / 'certificate.json', {'body': {}, 'sha256': 'bad'})
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()}
        with self.assertRaises((train.TrainingError, FileNotFoundError, KeyError)):
            train.completed(path)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.iterdir()})
        self.assertNotIn('torch', sys.modules)

    def test_self_digest_detects_valid_json_state_mutation(self):
        path = WORK / 'envelope.json'; train.atomic(path, train.envelope({'step': 2, 'draws': 256}))
        self.assertEqual(train.unseal(path)['step'], 2)
        body = train.read(path); body['body']['step'] = 4; train.atomic(path, body)
        with self.assertRaises(train.TrainingError): train.unseal(path)

    def test_input_pins_detect_modified_and_missing_files(self):
        path = WORK / 'source.json'; train.atomic(path, {'state': 2}); pinned = train.pins([path])
        train.current(pinned); train.atomic(path, {'state': 4})
        with self.assertRaises(train.TrainingError): train.current(pinned)
        with self.assertRaises(train.TrainingError): train.current({str(path.parent / 'missing'): '0' * 64})

    def test_no_settings_override_or_gpu_worker_path(self):
        text = Path(train.__file__).read_text()
        self.assertNotIn("device='cuda'", text); self.assertNotIn('.cuda()', text)
        self.assertIn('weights_only=True', text)
        self.assertIn('before_checkpoint_catalog', train.HOOKS)
        self.assertIn('after_checkpoint_catalog', train.HOOKS)
        self.assertEqual(sum(__import__('math').prod(shape) for shape in train.BRANCH_SHAPES.values()), 2065)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ValueTrainingTests))
    root = train.ROOT / 'build/value-training-v1'; root.mkdir(exist_ok=True)
    index = len(list(root.glob('trainer-owner-unit-attempt-*.json'))) + 1
    train.atomic(root / f'trainer-owner-unit-attempt-{index:03d}.json', dict(schema='value-training-owner-unit-v1',
        passed=result.wasSuccessful(), tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        optimizer_executions=0, production_steps=0, inputs=train.pins([train.__file__, __file__])))
    raise SystemExit(0 if result.wasSuccessful() else 1)
