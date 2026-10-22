"""Metadata generation contracts; no network, target, training or search work."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from congklak import config, ordinary_development as module


FIXTURE = module.ROOT / 'build/ordinary-development-v1/data-owner-fixture-v5'


class OrdinaryDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module.prepare(FIXTURE); module.run(FIXTURE)

    def test_counter_is_exact_stable_legal_sparse_geometry(self):
        for attempt in (0, 1, 2, 3, 1289, 16383):
            cell, state = module.candidate(attempt)
            self.assertEqual((cell, state), module.candidate(attempt))
            self.assertEqual(config.position(state), state)
            self.assertEqual(sum(n > 0 for n in state['pits'][state['controller']]), 2)
            self.assertEqual(sum(n > 0 for n in state['pits'][1 - state['controller']]), 7)
            self.assertEqual(sum(map(sum, state['pits'])) // 25, 1 if cell.startswith('25') else 2)

    def test_fixed_pool_quotas_and_attempt_accounting(self):
        manifest = module.admit(FIXTURE)
        self.assertEqual(manifest['counts']['accepted_components'], 32)
        self.assertEqual(manifest['cells'], {cell: 8 for cell in module.PROTOCOL['cells']})
        self.assertEqual(sum(manifest['rejection_reasons'].values()), manifest['counts']['attempted_counters'])
        self.assertEqual(manifest['counts']['classification_transitions'], 2 * manifest['counts']['generated_roots'])
        self.assertEqual(manifest['counts']['accepted_legal_choice_histogram'], {'2': 32})
        self.assertLess(manifest['counts']['attempted_counters'], 16384)

    def test_all32_full_origin_closures_are_disjoint_old_and_each_other(self):
        inv = module.read(FIXTURE / 'parent-inventory.json')
        seen = {q for c in inv['components'].values() for q in c['quotients']}
        for record in module.read(FIXTURE / 'candidates.json')['accepted']:
            own = module.eligible_quotients(record)
            self.assertFalse(seen & own); seen.update(own)
            self.assertEqual(record['mechanism'], 'ordinary')
            self.assertEqual(len(module.observations(record)), 3)

    def test_completed_paths_are_read_only_with_producers_and_writers_trapped(self):
        before = {p.relative_to(FIXTURE).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, module.sha(p)) for p in FIXTURE.rglob('*') if p.is_file()}
        with patch.object(module.old, 'Writer', side_effect=AssertionError('writer')), \
             patch.object(module.old, 'inventory', side_effect=AssertionError('old inventory')), \
             patch.object(module, '_facts', side_effect=AssertionError('classification')), \
             patch('congklak.native.Native', side_effect=AssertionError('native')):
            self.assertEqual(module.run(FIXTURE), module.verify(FIXTURE))
            self.assertEqual(module.prepare(FIXTURE)['status'], 'complete')
            self.assertEqual(module.status(FIXTURE)['status'], 'complete')
        after = {p.relative_to(FIXTURE).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, module.sha(p)) for p in FIXTURE.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_or_corrupt_completed_certificate_refuses_without_producer(self):
        with tempfile.TemporaryDirectory(prefix='data-owner-corrupt-', dir=module.ROOT / 'build/ordinary-development-v1') as folder:
            out = Path(folder)
            for name in ('identity.json', 'protocol.json', 'parent-inventory.json', 'catalog.json'):
                shutil.copy2(FIXTURE / name, out / name)
            with patch.object(module.old, 'Writer', side_effect=AssertionError('writer')), patch.object(module, '_facts', side_effect=AssertionError('producer')):
                with self.assertRaises(FileNotFoundError): module.run(out)
                module.atomic(out / 'certificate.pending.json', {})
                with self.assertRaises(ValueError): module.run(out)

    def test_counter_budget_and_namespace_guard(self):
        for value in (-1, 16384, True):
            with self.assertRaises(ValueError): module.candidate(value)
        with self.assertRaises(ValueError): module._path(module.ROOT / 'data/unapproved-development')

    def test_missing_committed_chunks_cannot_reset_counter(self):
        with tempfile.TemporaryDirectory(prefix='data-owner-missing-chunks-', dir=module.ROOT / 'build/ordinary-development-v1') as folder:
            out = Path(folder)
            for name in ('identity.json', 'protocol.json', 'parent-inventory.json', 'cursor.json'):
                shutil.copy2(FIXTURE / name, out / name)
            with patch('congklak.native.Native', side_effect=AssertionError('regeneration')):
                with self.assertRaisesRegex(ValueError, 'Missing committed'):
                    module.run(out)

    def test_valid_single_chunk_cursor_gap_recovers_but_two_chunk_gap_refuses(self):
        with tempfile.TemporaryDirectory(prefix='data-owner-cursor-gap-', dir=module.ROOT / 'build/ordinary-development-v1') as folder:
            out = Path(folder)
            for name in ('identity.json', 'protocol.json', 'parent-inventory.json'):
                shutil.copy2(FIXTURE / name, out / name)
            module.atomic(out / 'cursor.json', dict(schema=module.SCHEMA + '-cursor', completed_chunks=0,
                          last_chunk_sha256=None, identity_sha256=module.sha(out / 'identity.json')))
            (out / 'chunks').mkdir()
            shutil.copy2(FIXTURE / 'chunks/000000.json', out / 'chunks/000000.json')
            _, inv = module._identity(out)
            rows = module._load_chunks(out, inv)[3]
            self.assertEqual(len(rows), 64)
            shutil.copy2(FIXTURE / 'chunks/000001.json', out / 'chunks/000001.json')
            with self.assertRaisesRegex(ValueError, 'cursor gap'): module._load_chunks(out, inv)
            (out / 'chunks/000001.json').unlink()
            module._cursor_write(out, sorted((out / 'chunks').glob('*.json')))
            self.assertEqual(module.read(out / 'cursor.json')['completed_chunks'], 1)
            module._load_chunks(out, inv, complete=True)


if __name__ == '__main__': unittest.main()
