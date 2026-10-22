"""Bounded engineering fixtures; no target loading, optimizer or search."""
import copy
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from congklak import config, valuepilot_reservations as module


def feasible_fixture():
    """Manufactured metadata, deliberately not eligible for prepare/admit."""
    primary = []; arena = []; graph = module.Graph()
    sequences = {active: iter(itertools.combinations(range(1, active), 3)) for active in (25, 50, 75)}
    def make(cell, index, active):
        actor = int(cell[-1]); a, b, c = next(sequences[active])
        pits = [[0] * 7, [0] * 7]
        pits[actor][:4] = [a, b - a, c - b, active - c]
        state = dict(pits=pits, stores=[98 - active, 0], controller=actor)
        family = 'synthetic-only/' + cell + '/' + str(index)
        graph.add(family, state)
        return dict(state=state, state_key=module.common.rawkey(state), family=family,
                    actor=actor, cell=cell, stratum=cell.rsplit('/actor', 1)[0], legal_mask=15)
    for cell in module.previous.CELLS:
        for index in range(80): primary.append(make(cell, index, 50 if cell.startswith('50') else 25))
    for mechanism in ('ordinary', 'capture', 'bonus'):
        for index in range(8): arena.append(make('75-98/' + mechanism + '/actor0', index, 75))
    components, lookup = graph.finish()
    for row in primary + arena:
        row['component'] = lookup[('family', row['family'])]
        row['quotient'] = module.previous.quotient(row['state'])
    return dict(components=components, primary_pool=primary, arena_pool=arena, previous_arena_families=[])


class ReservationsTests(unittest.TestCase):
    def test_joint_metadata_matching_and_quota(self):
        inv = feasible_fixture(); result = module.select(inv)
        self.assertTrue(result['admitted']); self.assertEqual(result['counts']['globally_distinct_components'], 664)
        self.assertEqual(result, module.select(copy.deepcopy(inv)))
        module.validate_selection(result, inv)

    def test_shortage_is_explicit_hold_without_partial_reservations(self):
        inv = feasible_fixture(); inv['arena_pool'] = [r for r in inv['arena_pool'] if not r['stratum'].endswith('/ordinary')]
        result = module.select(inv)
        self.assertFalse(result['admitted']); self.assertEqual(result['status'], 'hold')
        self.assertEqual(result['feasible_uncommitted_optimizer_components'], 640)
        self.assertEqual(result['train'], []); self.assertEqual(result['counts']['train'], 0)
        module.validate_selection(result, inv)

    def test_new_alias_tombstone_propagates(self):
        graph = module.Graph(); first = feasible_fixture()['primary_pool'][0]
        graph.add('old-protected', first['state']); graph.add('new-alias', first['state'])
        graph.mark('family', ['old-protected'], 'optimizer/protected_family')
        components, lookup = graph.finish()
        self.assertEqual(lookup[('family', 'new-alias')], lookup[('family', 'old-protected')])
        self.assertIn('optimizer/protected_family', components[lookup[('family', 'new-alias')]]['reasons'])

    def test_tampered_components_or_legal_state_refuse(self):
        inv = feasible_fixture(); result = module.select(inv)
        altered = copy.deepcopy(inv); cid = next(iter(altered['components']))
        altered['components'][cid]['families'].append('forged')
        with self.assertRaisesRegex(ValueError, 'digest'): module.validate_selection(result, altered)
        altered_result = copy.deepcopy(result); altered_result['train'][0]['legal_mask'] = 1
        with self.assertRaises(ValueError): module.validate_selection(altered_result, inv)

    def test_prepare_competitor_writer_and_namespace_guard(self):
        with tempfile.TemporaryDirectory(prefix='data-owner-lock-', dir=module.ROOT / 'build/valuepilot-interface-v1') as folder:
            with module.Writer(folder):
                with self.assertRaises(OSError):
                    with module.Writer(folder): pass
            with self.assertRaises(ValueError): module._path(module.ROOT / 'data/other')

    def test_publication_completed_reader_and_corrupt_cert_refusal(self):
        # Use a complete real metadata fixture prepared by the owner command;
        # trap every producer/writer entrypoint on completed run/status/verify.
        source = module.ROOT / 'build/valuepilot-interface-v1/data-owner-fixture-3'
        if not source.exists(): self.skipTest('Owner metadata fixture not prepared')
        manifest = module.verify(source)
        before = {p.name: (p.read_bytes(), p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
        with patch.object(module, 'Writer', side_effect=AssertionError('completed writer')), \
             patch.object(module, 'inventory', side_effect=AssertionError('completed producer')):
            self.assertEqual(module.run(source), manifest)
            self.assertEqual(module.verify(source), manifest)
            self.assertEqual(module.status(source)['status'], 'hold')
            with self.assertRaisesRegex(ValueError, 'admission held'): module.admit(source)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()})
        with tempfile.TemporaryDirectory(prefix='data-owner-corrupt-', dir=module.ROOT / 'build/valuepilot-interface-v1') as folder:
            out = Path(folder)
            for p in source.iterdir():
                if p.is_file(): shutil.copy2(p, out / p.name)
            (out / 'certificate.pending.json').write_text('{}\n', encoding='utf-8')
            with patch.object(module, 'Writer', side_effect=AssertionError('corrupt completed writer')):
                with self.assertRaises(ValueError): module.run(out)
                (out / 'certificate.pending.json').unlink()
                with self.assertRaises(FileNotFoundError): module.run(out)

    def _real_kill_boundary(self, boundary):
        parent = module.ROOT / 'build/valuepilot-interface-v1'
        out = Path(tempfile.mkdtemp(prefix='data-owner-kill-' + boundary + '-', dir=parent))
        marker = parent / (out.name + '.marker.json')
        module.prepare(out)
        command = [sys.executable, str(module.ROOT / 'tools/reserve-valuepilot.py'), 'run', '--out', str(out), '--pause-at', boundary, '--pause-marker', str(marker)]
        log = parent / (out.name + '.worker.log')
        with log.open('wb') as stream:
            process = subprocess.Popen(command, cwd=module.ROOT, stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 15
            while not marker.exists() and time.monotonic() < deadline and process.poll() is None: time.sleep(.025)
            if not marker.exists():
                if process.poll() is None: process.kill()
                process.wait(); self.fail('Worker did not reach retained pause marker: ' + str(log))
            committed = {p.name: (p.read_bytes(), p.stat().st_size, p.stat().st_mtime_ns) for p in out.iterdir() if p.is_file() and p.name != 'writer.lock'}
            process.kill(); process.wait(timeout=10)
        # Prefix is retained byte-for-byte. No rebuilding inventory or target
        # loading may occur when resuming either publication boundary.
        with patch.object(module, 'inventory', side_effect=AssertionError('Recovery rebuilt metadata')):
            result = module.run(out)
        self.assertEqual(result['status'], 'hold'); self.assertFalse(result['admitted'])
        for name, fingerprint in committed.items():
            p = out / name; self.assertEqual(fingerprint, (p.read_bytes(), p.stat().st_size, p.stat().st_mtime_ns))
        self.assertEqual(result, module.verify(out))
        module._atomic(parent / (out.name + '.result.json'), dict(schema='valuepilot-reservation-owner-kill-v1', boundary=boundary,
                        worker_pid=process.pid, worker_returncode=process.returncode,
                        source_sha256=module.sha(Path(module.__file__)), test_source_sha256=module.sha(__file__),
                        committed_prefix_files=sorted(committed), prefix_preserved=True, passed=True,
                        final_manifest_sha256=module.sha(out / 'manifest.json'), optimizer_steps=0))

    def test_real_kill_before_catalog_commit(self): self._real_kill_boundary('before_catalog_commit')

    def test_real_kill_after_catalog_commit(self): self._real_kill_boundary('after_catalog_commit')


if __name__ == '__main__': unittest.main()
