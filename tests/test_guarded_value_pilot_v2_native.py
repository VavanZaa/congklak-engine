"""Pure source/binding checks; no DLL load, inference, search, or optimizer."""
import ast
from pathlib import Path
import unittest
from congklak import guarded_value_pilot_v2_native as native


class NativeBindingTests(unittest.TestCase):
    def test_algorithm_copies_are_exact(self):
        for name in ('engine.hpp', 'rules.cpp', 'search.cpp', 'neural_relay.cpp',
                     'neural_puct_cpu.cpp', 'neural_value_sidecar.cpp', 'trace.hpp'):
            self.assertEqual((native.ROOT / 'native/ordinary_epoch_v1' / name).read_bytes(),
                             (native.ROOT / 'native/guarded_value_pilot_v2' / name).read_bytes())

    def test_trace_prefix_has_only_the_declared_static_capacity_change(self):
        old = (native.ROOT / 'native/ordinary_epoch_v1/trace.cpp').read_text()
        new = (native.ROOT / 'native/guarded_value_pilot_v2/trace.cpp').read_text()
        expected = old.replace('count>8000000||owners<1', 'count>16000000||owners<1')
        self.assertNotEqual(old, expected)
        self.assertTrue(new.startswith(expected))
        addition = new[len(expected):]
        self.assertEqual(addition.count('CC_API int cc_fast_guard_project_hash('), 1)
        self.assertEqual(addition.count('CC_API void* cc_fast_guard_create_component('), 1)
        self.assertNotIn('cc_ownership_trace_begin(', addition)

    def test_adapter_namespace_projection_preserves_arithmetic(self):
        expected = (native.ROOT / 'congklak/ordinary_epoch_trace_adapter.py').read_text().replace(
            'from .ordinary_epoch_native import sidecar_model as SidecarModel',
            'from .guarded_value_pilot_v2_native import sidecar_model as SidecarModel')
        actual = (native.ROOT / 'congklak/guarded_value_pilot_v2_adapter.py').read_text()
        self.assertEqual(ast.dump(ast.parse(expected), include_attributes=False),
                         ast.dump(ast.parse(actual), include_attributes=False))

    def test_fixed_candidate_binding_and_equal_time_policy(self):
        binding = native.candidate_binding()
        self.assertEqual(binding['path'], 'data/value-training-v1/training/model.json')
        self.assertEqual(binding['sha256'], native.FROZEN_SHA)
        self.assertEqual(native.POLICY['maximum_outer_ms'], 25)
        self.assertEqual(native.POLICY['reserve_ms'], 12.)
        self.assertEqual(native.POLICY['minimum_search_ms'], 10)
        self.assertIs(native.POLICY['same_all_engines'], True)
        with self.assertRaises(ValueError):
            native.candidate_binding(native.ROOT / 'data/quality-train-v1/model.json')

    def test_native_dependency_closure_is_current(self):
        pins = native.input_pins()
        self.assertTrue(pins)
        self.assertIn('tests/test_guarded_value_pilot_v2_native.py', pins)
        self.assertIn('build/guarded-value-pilot-v2/native/congklak_guarded_value_pilot_v2.dll', pins)
        for name, expected in pins.items():
            self.assertEqual(native.sha(native.ROOT / name), expected)


if __name__ == '__main__':
    unittest.main()
