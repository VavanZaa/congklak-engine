"""Pure admission and namespace tests; optimizer recovery is independently run."""
import builtins
import copy
import sys
import subprocess
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from congklak import guarded_value_pilot_v2_common as stage
from congklak import guarded_value_pilot_v2_train as train
from congklak import guarded_value_pilot_v2_sidecar as side
from congklak import guarded_value_pilot_v2_resources as resources


class GuardedValueTrainingTests(unittest.TestCase):
    def test_exact_recipe_and_frozen_helper_reuse(self):
        recipe = train.settings()
        self.assertEqual((recipe['steps'], recipe['roots'], recipe['batch_size']), (400, 192, 128))
        self.assertEqual(recipe['sampling'], 'uniform-four-cells/within-cell-uniform-replacement')
        self.assertEqual(recipe['checkpoint_interval'], 20)
        self.assertEqual(recipe['selected_candidate'], 'final-fixed-step-only')
        self.assertEqual((recipe['device'], recipe['dtype'], recipe['intra_threads']), ('cpu', 'float32', 1))
        self.assertIs(train.settings, train.frozen.settings)
        self.assertIs(train.equal_state, train.frozen.equal_state)
        with self.assertRaises(train.TrainingError):
            train.settings(1)

    def test_registered_namespace_cannot_touch_frozen_runs_or_arbitrary_fixtures(self):
        self.assertEqual(train._path(train.DEFAULT, False), train.DEFAULT.resolve())
        for name in ('uninterrupted', *train.HOOKS):
            target = stage.FOLDER / 'fixtures/independent-training-recovery-v2' / name
            self.assertEqual(train._path(target, True), target.resolve())
        for target in (stage.ROOT / 'data/ordinary-epoch-pilot-v1/training-v2',
                       stage.FOLDER / 'fixtures/unregistered', stage.DATA / 'arbitrary'):
            with self.assertRaises(train.TrainingError):
                train._path(target)

    def test_preparation_gate_is_bound_and_strictly_typed(self):
        checksum = 'c' * 64
        receipt = dict(schema=stage.SCHEMA + '-preparation-admission', passed=True, policy_fixed=True,
            plan_sha256=checksum, reservations_sha256=checksum, cache_manifest_sha256=checksum,
            initial_model_sha256=checksum, initial_conformance_passed=True,
            independent_checkpoint_recovery_passed=True, production_optimizer_steps=0,
            inputs={str(stage.PLAN): checksum})
        with patch.object(stage, 'admit_plan'), patch.object(train, 'read', return_value=receipt), \
             patch.object(train, 'sha', return_value=checksum), patch.object(train, 'current') as check:
            self.assertEqual(train.production_gate(), receipt)
            check.assert_called_once_with(receipt['inputs'])
            for field, value in (('passed', 1), ('policy_fixed', 1), ('initial_conformance_passed', False),
                                 ('independent_checkpoint_recovery_passed', False),
                                 ('production_optimizer_steps', False), ('inputs', {}),
                                 ('reservations_sha256', 'd' * 64)):
                changed = copy.deepcopy(receipt); changed[field] = value
                with patch.object(train, 'read', return_value=changed):
                    with self.assertRaises(train.TrainingError):
                        train.production_gate()

    def test_completed_dispatch_precedes_resources_writer_or_torch(self):
        target = stage.FOLDER / 'fixtures/independent-training-recovery-v2/uninterrupted'
        expected = dict(state='complete', steps=6)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'torch' or name.endswith('guarded_value_pilot_v2_resources'):
                raise AssertionError('Completed route imported a worker dependency')
            return original_import(name, *args, **kwargs)
        with patch.object(train, '_has_completion', return_value=True), \
             patch.object(train, 'completed', return_value=expected), patch.object(train, '_identity'), \
             patch.object(train, 'Writer', side_effect=AssertionError('Writer entered')), \
             patch('builtins.__import__', side_effect=guarded_import):
            self.assertEqual(train.run(target, fixture=True), expected)

    def test_frozen_policy_proof_and_zero_branch_do_not_import_torch(self):
        proof = side.policy_proof()
        self.assertEqual(proof['base_file_sha256'], side.BASE_SHA)
        self.assertEqual(proof['policy_training_steps'], 0)
        self.assertTrue(proof['policy_arithmetic_unchanged'])
        original = side.read(side.data.INITIAL)['model']['branch_weights']
        self.assertEqual(side.tensor_sha(original), side.frozen.INITIAL_TENSOR_SHA)
        self.assertEqual(original['v'], [[0.] * 16])
        self.assertEqual(original['c'], [0.])
        self.assertNotIn('torch', sys.modules)

    def test_leased_trainer_binds_before_finite_worker_views(self):
        calls=[]
        @contextmanager
        def scope():
            self.assertEqual(calls,['bind'])
            calls.append('enter')
            yield
            calls.append('close')
        def run(*args):
            self.assertEqual(calls,['bind','enter'])
            calls.append('run')
            return dict(state='complete')
        with patch.object(sys,'argv',['trainer','run','--lease','synthetic']), \
             patch.object(resources,'bind_worker',side_effect=lambda *a:calls.append('bind')), \
             patch.object(stage,'worker_views',side_effect=scope),patch.object(train,'run',side_effect=run), \
             patch('builtins.print'):
            self.assertEqual(train.main(),0)
        self.assertEqual(calls,['bind','enter','run','close'])

    def test_cli_preserves_nonzero_hold_exit_status(self):
        command=[sys.executable,str(stage.ROOT/'tools/train-guarded-value-pilot-v2.py'),
            'status','--output',str(stage.ROOT/'data/ordinary-epoch-pilot-v1/training-v2')]
        done=subprocess.run(command,cwd=stage.ROOT,capture_output=True,text=True,timeout=15,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(done.returncode,2,done.stdout+done.stderr)
        self.assertEqual(train.frozen.json.loads(done.stdout)['state'],'hold')


if __name__ == '__main__':
    unittest.main()
