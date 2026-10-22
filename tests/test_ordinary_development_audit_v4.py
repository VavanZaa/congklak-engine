"""Independent ordinary-development admission/conformance/recovery evidence.

No optimizer, target loader, teacher search, arena game or release claim is
part of this audit. Synthetic board provenance is checked literally.
"""
from __future__ import annotations
from contextlib import ExitStack
import copy
import ctypes
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/ordinary-development-v1'
WORK = None
EVIDENCE = {}
KILLS = []
ATTEMPTS = []


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def save(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Audit evidence is immutable: ' + str(path)
    path.write_text(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8', newline='\n')
def files(path):
    return sorted(p for p in Path(path).rglob('*') if p.is_file() and p.suffix != '.pyc' and not p.name.endswith('-shm'))
def snapshot(path):
    return {str(p.relative_to(path)): (sha(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files(path)}
def forbidden(*args, **kwargs): raise AssertionError('Unexpected computation/publication on completed route')


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import forbidden on metadata/completed route')
        return None


def run_process(argv, directory, timeout=180):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        result = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, timeout=timeout)
    record = dict(argv=[str(v) for v in argv], returncode=result.returncode,
                  elapsed_seconds=time.monotonic() - started, logs=rel(directory))
    ATTEMPTS.append(record); save(directory / 'execution.json', record)
    assert result.returncode == 0, 'Worker failed: ' + str(record) + '\n' + (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:]
    return record


def process_identity(pid):
    assert type(pid) is int and pid > 0
    script = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; if ($p) { [pscustomobject]@{pid=$p.ProcessId;parent=$p.ParentProcessId;command=$p.CommandLine} | ConvertTo-Json -Compress }"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip(), 'Marked worker is no longer running'
    return json.loads(result.stdout)


def kill_worker(argv, directory, marker, boundary, timeout=180):
    """Terminate the actual verified Windows worker, including venv redirectors."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); process = None; worker_handle = None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]; kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]; kernel.TerminateProcess.restype = ctypes.c_int
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]; kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]; kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]; kernel.CloseHandle.restype = ctypes.c_int
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err)
            deadline = time.monotonic() + timeout
            while not marker.exists():
                assert process.poll() is None, 'Worker ended before pause: ' + (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:]
                assert time.monotonic() < deadline, 'Pause marker timeout'
                time.sleep(.05)
            body = read(marker); assert body.get('event', body.get('point')) == boundary
            worker_pid = body['pid']; details = process_identity(worker_pid)
            lineage = [details]; current = details
            while current['pid'] != process.pid and len(lineage) < 5:
                current = process_identity(current['parent']); lineage.append(current)
            assert lineage[-1]['pid'] == process.pid, 'Marked worker is not our launched descendant'
            assert str(Path(argv[1]).resolve()).casefold() in details['command'].casefold(), 'Worker script differs'
            worker_handle = kernel.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, worker_pid)
            assert worker_handle, 'Cannot open verified worker handle'
            assert kernel.TerminateProcess(worker_handle, 1), 'Actual worker termination failed'
            assert kernel.WaitForSingleObject(worker_handle, 30000) == 0, 'Worker termination wait failed'
            worker_code = ctypes.c_ulong(); assert kernel.GetExitCodeProcess(worker_handle, ctypes.byref(worker_code))
            assert worker_code.value != 0
            code = process.wait(timeout=30); assert code != 0
        finally:
            if worker_handle: kernel.CloseHandle(worker_handle)
            if process is not None and process.poll() is None: process.kill(); process.wait(timeout=30)
    record = dict(boundary=boundary, argv=[str(v) for v in argv], launcher_pid=process.pid,
        worker_pid=worker_pid, worker_identity=details, verified_lineage=lineage,
        returncode=code, worker_returncode=worker_code.value, marker=rel(marker), marker_sha256=sha(marker),
        actual_process_kill=True, elapsed_seconds=time.monotonic() - started, logs=rel(directory))
    KILLS.append(record); ATTEMPTS.append(record); save(directory / 'execution.json', record)
    return record


def boundary(active, actor=0, forced=False):
    pits = [[0] * 7, [0] * 7]
    if forced: pits[1 - actor][2] = active
    else:
        for n in range(active): pits[n % 2][(3 * (n // 2) + 1) % 7] += 1
    remainder = 98 - active
    return dict(pits=pits, stores=[remainder // 3, remainder - remainder // 3], controller=actor)


def literal(base, weights, position):
    """Independent scalar algebra, without calling the sidecar reference path."""
    import math
    from congklak import features_relay
    from congklak.neural_relay import rounded
    x, d, r = features_relay.encode(position, 127)
    if not r: return float(d), d * 256, [0.] * 7
    h = list(x)
    for prefix in ('layers.0', 'layers.2'):
        h = [max(0., sum(a * b for a, b in zip(row, h)) + bias)
             for row, bias in zip(base.weights[prefix + '.weight'], base.weights[prefix + '.bias'])]
    u = sum(a * b for a, b in zip(base.weights['layers.4.weight'][0], h)) + base.weights['layers.4.bias'][0]
    logits = [sum(a * b for a, b in zip(row, h)) + bias
              for row, bias in zip(base.weights['policy.weight'], base.weights['policy.bias'])]
    if 25 <= sum(map(sum, position['pits'])) <= 74:
        hidden = [max(0., sum(a * b for a, b in zip(row, x)) + bias)
                  for row, bias in zip(weights['A'], weights['b'])]
        u += sum(a * b for a, b in zip(weights['v'][0], hidden)) + weights['c'][0]
    residual = r * math.tanh(u)
    return d + residual, d * 256 + rounded(residual * 256), logits


def closed_worker(specification):
    from congklak import ordinary_development as generation, valuepilot_reservations_v2 as data, value_sidecar_v2 as side
    from congklak import ordinary_development_v3 as generation_v3
    from congklak import valuepilot_reservations as old, quality_search
    from congklak.native import Native
    from congklak.neural_relay import Model as BaseModel, NeuralNative
    guard = NoTorch(); sys.meta_path.insert(0, guard)
    directories = [Path(p) for p in specification['directories']]
    before = {str(p): snapshot(p) for p in directories}
    with ExitStack() as stack:
        for obj, name in ((generation, '_facts'), (generation_v3, '_facts'), (old, 'inventory'), (old, 'Writer'),
                          (data, 'inventory'), (data, 'Writer'), (side, '_writer'),
                          (Native, '__init__'), (NeuralNative, '__init__'),
                          (side.features_relay, 'encode'), (side.Model, 'reference'),
                          (BaseModel, 'reference'), (side.SidecarEvaluator, '__init__'), (quality_search, 'search')):
            stack.enter_context(mock.patch.object(obj, name, forbidden))
        generation.run(generation.DEFAULT); data.run(data.DEFAULT)
        try: generation_v3.run(generation.DEFAULT)
        except ValueError: pass
        else: raise AssertionError('V2 producer accepted a legacy production namespace')
        generation_v3.run(specification['generation_v3'])
        side.initialize(specification['manifest'], specification['production'])
        side.initialize(specification['manifest'], specification['engineering'], fixture=True)
        import runpy
        for call in specification['clis']:
            old_argv = sys.argv
            try:
                sys.argv = call
                try: runpy.run_path(call[0], run_name='__main__')
                except SystemExit as exc: assert exc.code in (None, 0), str(exc.code)
            finally: sys.argv = old_argv
    assert 'torch' not in sys.modules
    assert before == {str(p): snapshot(p) for p in directories}, 'Completed bytes/sizes/mtimes changed'
    save(specification['proof'], dict(passed=True, durable_unchanged=True, no_writer=True, no_torch=True,
        no_native=True, no_features=True, no_forwards=True, no_search=True, no_generator_facts=True,
        pure_metadata_counter_reconstruction_allowed=True, before=before))


class OrdinaryDevelopmentIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from congklak import value_sidecar_v2 as side, valuepilot_reservations_v2 as data
        from congklak.neural_relay import Model
        cls.manifest = data.DEFAULT / 'manifest.json'
        cls.reservations = data.verify()
        assert cls.reservations['admitted'] is True
        cls.selection = read(data.DEFAULT / 'selection.json')
        cls.inventory = read(data.DEFAULT / 'inventory.json')
        cls.initial = WORK / 'fresh-initialization'
        cls.torch_python = read(ROOT / 'build/training-runtime.json')['python']
        run_process([cls.torch_python, str(ROOT / 'tools/export-value-sidecar-v2.py'), 'initialize',
                     '--manifest', str(cls.manifest), '--output', str(cls.initial), '--engineering-fixture'],
                    WORK / 'initialization-execution')
        cls.model = side.Model(cls.initial / 'model.json'); cls.base = Model(side.BASE)
        weights = copy.deepcopy(cls.model.weights)
        weights['v'] = [[(-1 if i % 2 else 1) * (i + 1) / 256 for i in range(16)]]; weights['c'] = [-.0625]
        cls.synthetic_path = WORK / 'synthetic-nonzero.json'
        save(cls.synthetic_path, side.make_export(weights, cls.manifest,
            dict(engineering_fixture=True, training_eligible=False, optimizer_steps=0,
                 purpose='Independent fixed nonzero branch; not a trained candidate')))
        cls.synthetic = side.Model(cls.synthetic_path)
        cls.positions = [r['state'] for role in ('train', 'fit_check', 'arena') for r in cls.selection[role]]
        assert len(cls.positions) == 664
        exact = data.old._json_metadata('build/ordinary-value-benchmark-projection.json', 'cases')
        forensic = data.old._json_metadata('build/divergence-panel.json', 'roots')
        assert len(exact) == 32 and len(forensic) == 5
        cls.positions += [r['state'] for r in exact + forensic]
        cls.positions += [boundary(a, c) for a in (0, 4, 5, 24, 25, 49, 50, 74, 75, 98) for c in (0, 1)]
        cls.positions += [boundary(a, c, True) for a in (4, 24, 25, 49, 74, 75) for c in (0, 1)]
        EVIDENCE['engineering_initialization'] = dict(model_sha256=cls.model.id,
             model_file_sha256=cls.model.file_sha256, fixture=True, training_eligible=False,
             corpus_rows=len(cls.positions), admitted_rows=664, reused_exact_and_forensic_rows=37, boundary_forced_rows=32)

    def test_01_synthetic_provenance_full_origin_and_filter_accounting(self):
        from collections import Counter
        from congklak import config, neurodiag_common as common, neurodiag_data as old, ordinary_development as generation
        from congklak.native import Native, flatten
        manifest = generation.verify(); candidates = read(generation.DEFAULT / 'candidates.json')['accepted']
        self.assertTrue(manifest['admitted']); self.assertEqual(len(candidates), 32)
        self.assertEqual(Counter(r['cell'] for r in candidates), {c: 8 for c in generation.PROTOCOL['cells']})
        self.assertFalse(read(generation.DEFAULT / 'candidates.json')['opening_reachability_claimed'])
        old_inv = read(generation.DEFAULT / 'parent-inventory.json')
        old_q = {q for c in old_inv['components'].values() for q in c['quotients']}
        records = [r for path in sorted((generation.DEFAULT / 'chunks').glob('*.json')) for r in read(path)['attempts']]
        native = Native(); classified = probes_checked = 0; seen = set(); accepted = []; cells = Counter()
        for attempt, row in enumerate(records):
            self.assertEqual(row['attempt'], attempt)
            cell = generation.PROTOCOL['cells'][attempt % 4]; self.assertEqual(row['cell'], cell)
            if row['reason'] == 'cell_quota_filled':
                self.assertEqual(cells[cell], 8); self.assertFalse(row['accepted']); continue
            # Independent literal construction; no call to generation.candidate/_facts.
            def block(part): return hashlib.sha256(('ordinary-development/v1/20261016/' + str(attempt) + '/' + str(part)).encode('ascii')).digest()
            active = (25 if cell.startswith('25') else 50) + int.from_bytes(block('active')[:4], 'little') % 25
            actor = int(cell[-1]); counts = [0] * 14
            first = int.from_bytes(block('first')[:4], 'little') % 7
            second = (first + 1 + int.from_bytes(block('second')[:4], 'little') % 6) % 7
            own_total = 2 + int.from_bytes(block('own_total')[:4], 'little') % (active - 8)
            first_count = 1 + int.from_bytes(block('first_count')[:4], 'little') % (own_total - 1)
            counts[actor * 7 + first] = first_count; counts[actor * 7 + second] = own_total - first_count
            for index in range(7): counts[(1 - actor) * 7 + index] = 1
            for index in range(active - own_total - 7): counts[(1 - actor) * 7 + int.from_bytes(block(index)[:4], 'little') % 7] += 1
            stored = 98 - active
            state = dict(pits=[counts[:7], counts[7:]], stores=[stored // 2, stored - stored // 2], controller=int(cell[-1]))
            self.assertEqual(config.position(state), row['state']); self.assertEqual(common.rawkey(state), row['state_key'])
            normal = native.normalize(flatten(state), actor=state['controller'], total=98)
            self.assertTrue(normal['ok']); self.assertEqual(normal['position'], state); self.assertFalse(normal['terminal'])
            legal = sorted((first, second)); self.assertEqual(normal['legal_actions'], legal)
            self.assertEqual(row['legal_mask'], sum(1 << a for a in legal)); self.assertEqual(row['legal_mask'].bit_count(), 2)
            actual = []
            for action in normal['legal_actions']:
                result = native.apply(flatten(state), actor=state['controller'], action=action, total=98, trace=False)
                self.assertTrue(result['ok']); child = result['position']
                normalized = native.normalize(flatten(child), actor=child['controller'], total=98); self.assertTrue(normalized['ok'])
                actual.append(dict(action=action, reason=result['reason'], passes=result['passes'], state=normalized['position'], terminal=normalized['terminal']))
                probes_checked += 1
            self.assertEqual(actual, row['probes']); classified += 1
            mechanism = 'pass' if any(p['passes'] for p in actual) else 'capture' if any(p['reason'] == 'own_capture' for p in actual) else 'bonus' if any(p['reason'] == 'own_store' for p in actual) else 'ordinary'
            self.assertEqual(mechanism, row['mechanism'])
            observations = [state] + [p['state'] for p in actual]
            q = {old.quotient(p) for p in observations if sum(map(sum, p['pits'])) > 4}
            reason = 'mechanism_' + mechanism if mechanism != 'ordinary' else 'old_component_overlap' if q & old_q else 'generated_origin_overlap' if q & seen else 'accepted'
            self.assertEqual(reason, row['reason']); self.assertIs(row['accepted'], reason == 'accepted')
            if row['accepted']:
                self.assertEqual(row['family'], 'synthetic-ordinary-development-v1/' + f'{attempt:05d}')
                self.assertFalse(q & old_q); self.assertFalse(q & seen)
                seen.update(q); cells[cell] += 1; accepted.append(row)
        self.assertEqual(accepted, candidates)
        self.assertEqual(manifest['counts']['classification_transitions'], probes_checked)
        self.assertEqual(manifest['counts']['generated_roots'], classified)
        self.assertEqual(manifest['counts']['attempted_counters'], len(records))
        self.assertLessEqual(len(records), 16384)
        self.assertEqual(manifest['counts']['accepted_components'], 32)
        EVIDENCE['synthetic_provenance'] = dict(accepted_components=32, per_band_actor=8,
            attempted_counters=len(records), metadata_classified_roots=classified,
            independently_replayed_classification_transitions=probes_checked,
            generated_classification_transitions=manifest['counts']['classification_transitions'],
            all_accepted_origins_disjoint_old_and_each_other=True,
            rejected_attempts_not_admitted_origins=True, opening_reachability_claimed=False,
            targets_or_predictions_used=False, optimizer_steps=0, new_games=0)

    def test_02_joint_roles_all_origin_protection_and_determinism(self):
        from collections import Counter
        from congklak import config, neurodiag_common as common, neurodiag_data as old, ordinary_development as generation
        from congklak import valuepilot_reservations_v2 as data
        inv = self.inventory; selection = self.selection; components = inv['components']
        by_f = {}; by_q = {}
        for cid, c in components.items():
            self.assertEqual(cid, config.digest(c))
            for f in c['families']: self.assertNotIn(f, by_f); by_f[f] = cid
            for q in c['quotients']: self.assertNotIn(q, by_q); by_q[q] = cid
        prior = read(generation.DEFAULT / 'parent-inventory.json')
        for previous in prior['components'].values():
            destinations = {by_f[f] for f in previous['families']} | {by_q[q] for q in previous['quotients']}
            self.assertEqual(len(destinations), 1, 'Old full component was split')
            self.assertLessEqual(set(previous['reasons']), set(components[next(iter(destinations))]['reasons']))
        candidates = read(generation.DEFAULT / 'candidates.json')['accepted']; candidate_families = set()
        for row in candidates:
            family = row['family']; candidate_families.add(family); cid = by_f[family]
            self.assertTrue(any(r.startswith('optimizer/') for r in components[cid]['reasons']))
            self.assertFalse(any(r.startswith('arena/') for r in components[cid]['reasons']))
            for p in [row['state']] + [p['state'] for p in row['probes']]:
                if sum(map(sum, p['pits'])) > 4: self.assertEqual(cid, by_q[old.quotient(p)])
        rows = [r for role in ('train', 'fit_check', 'arena') for r in selection[role]]
        self.assertEqual(len(rows), 664); self.assertEqual(len({r['component'] for r in rows}), 664)
        self.assertEqual(len({r['family'] for r in rows}), 664); self.assertEqual(len({r['state_key'] for r in rows}), 664)
        for role, quota in (('train', 64), ('fit_check', 16)):
            self.assertEqual(Counter(r['cell'] for r in selection[role]), {c: quota for c in old.CELLS})
            for row in selection[role]:
                c = components[row['component']]
                self.assertFalse(set(c['families']) & candidate_families)
                self.assertFalse(any(r.startswith('optimizer/') for r in c['reasons']))
        self.assertEqual(Counter(r['cell'] for r in selection['fit_cases']), {c: 4 for c in old.CELLS})
        self.assertLessEqual({r['state_key'] for r in selection['fit_cases']}, {r['state_key'] for r in selection['fit_check']})
        self.assertEqual(Counter(r['stratum'].split('/')[-1] for r in selection['arena']), dict(ordinary=8, capture=8, bonus=8))
        ordinary = [r for r in selection['arena'] if r['stratum'].endswith('/ordinary')]
        self.assertEqual(len(ordinary), 8)
        self.assertTrue(all(r['family'] in candidate_families for r in ordinary))
        for role in ('train', 'fit_check', 'arena'):
            pool = {r['state_key']: r for r in inv['arena_pool' if role == 'arena' else 'primary_pool']}
            for row in selection[role]:
                self.assertEqual(common.rawkey(row['state']), row['state_key'])
                self.assertEqual(row['component'], by_f[row['family']]); self.assertEqual(row['component'], by_q[old.quotient(row['state'])])
                self.assertGreaterEqual(row['legal_mask'].bit_count(), 2)
                self.assertFalse(any(r.startswith('arena/' if role == 'arena' else 'optimizer/') for r in components[row['component']]['reasons']))
                self.assertTrue(all(row[k] == pool[row['state_key']][k] for k in row))
        for row in inv['protected_cases']:
            self.assertTrue(any(r.startswith('arena/') for r in components[by_q[old.quotient(row['state'])]]['reasons']))
        with mock.patch.object(old, 'teacher_target', forbidden), mock.patch.object(old, 'source_item', forbidden):
            self.assertEqual(data.select(inv), selection)
        self.assertFalse(selection['fit_check_is_unseen']); self.assertTrue(selection['policy_fixed'])
        EVIDENCE['joint_reservations'] = dict(counts=self.reservations['counts'],
            all_32_generated_origin_components_optimizer_blocked=True, selected_generated_ordinary_components=8,
            unused_generated_components=24, globally_distinct_full_components=664,
            full_old_components_preserved=len(prior['components']), full_current_components=len(components),
            deterministic_joint_selection=True, metadata_before_targets=True, fit_check_unseen=False)

    def test_03_actual_corpus_native_policy_domain_and_batch(self):
        from congklak import value_sidecar_v2 as side
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        maxima = dict(zero_native_literal_seed=0., nonzero_native_literal_seed=0., native_literal_logit=0., score_units=0)
        changed = outside = comparisons = 0; backends = []
        literals = [(literal(self.base, self.model.weights, p), literal(self.base, self.synthetic.weights, p)) for p in self.positions]
        for backend in ('scalar', 'auto'):
            with CpuBatchEvaluator(side.BASE, backend=backend) as base, side.SidecarEvaluator(self.model, backend=backend) as zero, side.SidecarEvaluator(self.synthetic, backend=backend) as candidate:
                backends.append(base.backend)
                for start in range(0, len(self.positions), 32):
                    group = self.positions[start:start + 32]
                    aa = base.evaluate_many(group); zz = zero.evaluate_many(group); bb = candidate.evaluate_many(group)
                    for offset, (p, a, z, b) in enumerate(zip(group, aa, zz, bb)):
                        comparisons += 1
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *z['policy_logits']))
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *b['policy_logits']))
                        self.assertEqual(a['value_player0'], z['value_player0'])
                        for key in ('feature_simulations', 'feature_deposits', 'native_actual_evals'):
                            self.assertEqual(a[key], z[key]); self.assertEqual(a[key], b[key])
                        rz, rb = literals[start + offset]
                        maxima['zero_native_literal_seed'] = max(maxima['zero_native_literal_seed'], abs(z['value_player0'] - rz[0]))
                        maxima['nonzero_native_literal_seed'] = max(maxima['nonzero_native_literal_seed'], abs(b['value_player0'] - rb[0]))
                        self.assertLessEqual(abs(z['value_player0'] - rz[0]), 1e-10)
                        self.assertLessEqual(abs(b['value_player0'] - rb[0]), 1e-10)
                        logit_error = max(abs(t - u) for t, u in zip(b['policy_logits'], rb[2]))
                        maxima['native_literal_logit'] = max(maxima['native_literal_logit'], logit_error)
                        self.assertLessEqual(logit_error, 1e-10)
                        score = candidate.evaluate_units(p)
                        maxima['score_units'] = max(maxima['score_units'], abs(score['value_units'] - rb[1]))
                        self.assertEqual(score['value_units'], rb[1])
                        if not 25 <= sum(map(sum, p['pits'])) <= 74:
                            outside += 1; self.assertEqual(a['value_player0'], b['value_player0'])
                        else: changed += int(a['value_player0'] != b['value_player0'])
                    # Every admitted, guard and boundary occurrence gets a
                    # separate single-row versus batch check.
                    for p, b in zip(group, bb):
                        one = candidate.evaluate_many([p])[0]
                        for key in ('value_player0', 'policy_logits', 'feature_simulations', 'feature_deposits'):
                            self.assertEqual(one[key], b[key])
                counts = candidate.snapshot_counters()
                self.assertGreater(counts['branch_calls'], 0); self.assertGreater(counts['branch_bypasses'], 0)
        self.assertGreater(changed, 0)
        self.assertEqual(backends, ['scalar-double-packed', 'avx2-double-packed'])
        EVIDENCE['native_conformance'] = dict(corpus_rows=len(self.positions), admitted_rows=664,
            policy_comparisons=comparisons, backends=backends, synthetic_in_gate_changed=changed,
            outside_gate_equal=outside, additional_feature_simulations=0, maximum_errors=maxima,
            tolerance=1e-10, single_row_and_batch_all_inputs_equal=True)

    def test_04_low_active_search_identity_fp32_and_stepzero(self):
        from congklak import config, quality_search, value_sidecar_v2 as side
        from congklak.native import NativeError
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        positions = [boundary(a, c) for a in (4, 5, 24) for c in (0, 1)] + [boundary(24, c, True) for c in (0, 1)]
        reports = []
        with CpuBatchEvaluator(side.BASE, backend='scalar') as base, side.SidecarEvaluator(self.synthetic, backend='scalar') as candidate:
            for p in positions:
                base.begin_decision(); candidate.begin_decision()
                a = quality_search.search(p, base, simulations=128, c_puct=.25)
                b = quality_search.search(p, candidate, simulations=128, c_puct=.25)
                quality_search.validate_report(a, p, 128, c_puct=.25); quality_search.validate_report(b, p, 128, c_puct=.25)
                for key in a:
                    if key not in ('timing', 'work'): self.assertEqual(a[key], b[key], key)
                for key in a['work']:
                    if key != 'inference_ms': self.assertEqual(a['work'][key], b['work'][key], key)
                reports.append(dict(position=p, control=a, synthetic=b))
        save(WORK / 'low-active-fixed128.json', reports)
        self.assertEqual(side.tensor_sha(self.model.weights), side.INITIAL_TENSOR_SHA)
        self.assertEqual(self.model.envelope['model']['base_export'], self.base.envelope)
        self.assertEqual(self.model.identity['search']['protocol'], quality_search.PROTOCOL)
        self.assertNotEqual(self.model.id, self.base.id)
        self.assertEqual(side.reservation_bindings(self.manifest)['counts']['globally_distinct_components'], 664)
        rejected = []
        mutations = (
            ('schema', lambda e: e['model'].__setitem__('schema', 'congklak-value-sidecar-v1')),
            ('gate', lambda e: e['model']['bindings']['gate'].__setitem__('minimum_active', 24)),
            ('search', lambda e: e['model']['bindings']['search'].__setitem__('c_puct', 1.5)),
            ('build', lambda e: e['model']['bindings'].__setitem__('dll_sha256', '0' * 64)),
            ('input', lambda e: e['model']['bindings']['input_manifest'].__setitem__('sha256', '0' * 64)),
            ('encoder', lambda e: e['model']['bindings'].__setitem__('encoder', 'wrong')),
            ('rules', lambda e: e['model']['bindings'].__setitem__('rules_digest', '0' * 64)),
            ('extra-tensor', lambda e: e['model']['branch_weights'].update(extra=[0.])),
            ('base-policy', lambda e: e['model']['base_export']['model']['weights']['policy.bias'].__setitem__(0, 0.)),
        )
        for name, mutate in mutations:
            export = copy.deepcopy(self.model.envelope); mutate(export); export['sha256'] = config.digest(export['model'])
            path = WORK / ('rehashed-' + name + '.json'); save(path, export)
            with self.assertRaises((NativeError, ValueError, KeyError)): side.Model(path)
            rejected.append(name)
        with self.assertRaises((NativeError, ValueError)): side._initial_model(self.synthetic)
        from congklak.neural_relay import Model as OldModel
        with self.assertRaises(KeyError) as refusal: OldModel(self.initial / 'model.json')
        self.assertEqual(refusal.exception.args, ('identity',))
        spec = WORK / 'torch-spec.json'
        save(spec, dict(initial=str(self.initial / 'model.json'), synthetic=str(self.synthetic_path),
            positions=self.positions, output=str(WORK / 'torch-proof.json')))
        run_process([self.torch_python, str(ROOT / 'tools/audit-ordinary-development.py'), 'torch-worker', '--spec', str(spec)], WORK / 'torch-execution')
        proof = read(WORK / 'torch-proof.json'); self.assertTrue(proof['passed'])
        production = ROOT / 'data/neural-value-pilot-v2/interface-initial'
        verified = side.verify(production, self.manifest)
        self.assertTrue(verified['training_eligible']); self.assertFalse(verified['engineering_fixture'])
        self.assertEqual(side.Model(production / 'model.json').weights, self.model.weights)
        EVIDENCE['strict_identity_and_fp32'] = dict(rehashed_mutations_rejected=rejected, old_loader_refuses=True,
            synthetic_stepzero_refuses=True, torch=proof, low_active_roots=len(positions),
            fixed128_searches=16, requested_search_backups=2048, production_stepzero=verified,
            training_executed=False, optimizer_steps=0)

    def test_05_real_generation_reservation_export_kills_and_prefixes(self):
        from congklak import ordinary_development_v3 as generation, valuepilot_reservations_v2 as data, value_sidecar_v2 as side
        fresh = WORK / 'generation-fresh'; generation.prepare(fresh)
        run_process([sys.executable, str(ROOT / 'tools/generate-ordinary-development-v3.py'), 'run', '--out', str(fresh)], WORK / 'generation-fresh-execution')
        expected_generation = generation.verify(fresh)
        for point in ('before_chunk_commit', 'before_cursor_commit', 'after_chunk_commit', 'before_catalog_commit', 'after_catalog_commit'):
            route = WORK / ('generation-' + point); route.mkdir()
            for name in ('protocol.json', 'parent-inventory.json', 'identity.json'):
                shutil.copy2(fresh / name, route / name)
            generation._cursor_write(route, [])
            marker = WORK / ('generation-' + point + '-pause.json')
            argv = [sys.executable, str(ROOT / 'tools/generate-ordinary-development-v3.py'), 'run', '--out', str(route),
                    '--pause-at', point, '--pause-marker', str(marker)]
            kill_worker(argv, WORK / ('generation-kill-' + point), marker, point)
            prefix = snapshot(route)
            run_process([sys.executable, str(ROOT / 'tools/generate-ordinary-development-v3.py'), 'run', '--out', str(route)], WORK / ('generation-resume-' + point))
            after = snapshot(route)
            for name, prior in prefix.items():
                if name != 'cursor.json': self.assertEqual(after[name], prior, name)
            if 'cursor.json' in prefix:
                self.assertEqual(read(route / 'cursor.json')['completed_chunks'], len(list((route / 'chunks').glob('*.json'))))
            self.assertEqual(generation.verify(route), expected_generation)
            self.assertEqual((route / 'candidates.json').read_bytes(), (fresh / 'candidates.json').read_bytes())
            self.assertEqual((route / 'manifest.json').read_bytes(), (fresh / 'manifest.json').read_bytes())
            self.assertEqual({p.name: p.read_bytes() for p in (route / 'chunks').glob('*.json')},
                             {p.name: p.read_bytes() for p in (fresh / 'chunks').glob('*.json')})
            if point == 'after_catalog_commit': self.assertEqual(prefix, after)
        for point in ('before_catalog_commit', 'after_catalog_commit'):
            route = WORK / ('reservations-' + point); route.mkdir()
            for name in ('inventory.json', 'selection.json', 'identity.json'): shutil.copy2(data.DEFAULT / name, route / name)
            marker = WORK / ('reservations-' + point + '-pause.json')
            argv = [sys.executable, str(ROOT / 'tools/reserve-valuepilot-v2.py'), 'run', '--out', str(route),
                    '--pause-at', point, '--pause-marker', str(marker)]
            kill_worker(argv, WORK / ('reservations-kill-' + point), marker, point)
            prefix = snapshot(route)
            run_process([sys.executable, str(ROOT / 'tools/reserve-valuepilot-v2.py'), 'run', '--out', str(route)], WORK / ('reservations-resume-' + point))
            after = snapshot(route)
            for name, prior in prefix.items(): self.assertEqual(after[name], prior, name)
            self.assertEqual(data.verify(route), data.verify())
            if point == 'after_catalog_commit': self.assertEqual(prefix, after)
        for point in side.HOOKS:
            route = WORK / ('export-' + point)
            argv = [self.torch_python, str(ROOT / 'tools/export-value-sidecar-v2.py'), 'initialize',
                    '--manifest', str(self.manifest), '--output', str(route), '--engineering-fixture', '--pause-at', point]
            kill_worker(argv, WORK / ('export-kill-' + point), route / 'pause.json', point)
            prefix = snapshot(route)
            executable = self.torch_python if point == 'before_model_commit' else sys.executable
            run_process([executable, str(ROOT / 'tools/export-value-sidecar-v2.py'), 'initialize',
                         '--manifest', str(self.manifest), '--output', str(route), '--engineering-fixture'], WORK / ('export-resume-' + point))
            after = snapshot(route)
            for name, prior in prefix.items(): self.assertEqual(after[name], prior, name)
            self.assertEqual((route / 'model.json').read_bytes(), (self.initial / 'model.json').read_bytes())
            self.assertEqual((route / 'certificate.json').read_bytes(), (self.initial / 'certificate.json').read_bytes())
            self.assertFalse(side.verify(route)['training_eligible'])
            if point == 'after_certificate_commit': self.assertEqual(prefix, after)
        self.assertEqual(len(KILLS), 11)
        EVIDENCE['publication_recovery'] = dict(real_process_kills=11, generation_boundaries=5,
            reservation_boundaries=2, export_boundaries=4, all_committed_prefix_bytes_sizes_mtimes_preserved=True,
            deterministic_generation_and_memberships=True, committed_model_not_regenerated=True,
            generation_counter_budget_not_reset=True, bounded_uncommitted_chunk_replay=True,
            logical_committed_probe_counts_not_total_physical_recovery_attempts=True,
            generation_recovery_schema=generation.SCHEMA, replay_of_existing_pool_not_new_distinct_data=True,
            training_recovery_executed=False, arena_recovery_executed=False, optimizer_steps=0)

    def test_06_completed_failclosed_history_and_dashboard(self):
        from congklak import ordinary_development as generation, valuepilot_reservations_v2 as data, value_sidecar_v2 as side
        from congklak import ordinary_development_v3 as generation_v3
        from congklak import ordinary_development_progress as progress
        from congklak.native import NativeError
        production = ROOT / 'data/neural-value-pilot-v2/interface-initial'
        specification = WORK / 'closed-spec.json'
        save(specification, dict(directories=[str(generation.DEFAULT), str(data.DEFAULT), str(production), str(self.initial), str(WORK / 'generation-fresh')],
            generation_v3=str(WORK / 'generation-fresh'),
            manifest=str(self.manifest), production=str(production), engineering=str(self.initial), proof=str(WORK / 'closed-proof.json'),
            clis=[[str(ROOT / 'tools/generate-ordinary-development.py'), 'run'],
                  [str(ROOT / 'tools/generate-ordinary-development-v3.py'), 'run', '--out', str(WORK / 'generation-fresh')],
                  [str(ROOT / 'tools/reserve-valuepilot-v2.py'), 'run'],
                  [str(ROOT / 'tools/export-value-sidecar-v2.py'), 'initialize', '--manifest', str(self.manifest), '--output', str(production)],
                  [str(ROOT / 'tools/export-value-sidecar-v2.py'), 'initialize', '--manifest', str(self.manifest), '--output', str(self.initial), '--engineering-fixture']]))
        run_process([sys.executable, str(ROOT / 'tools/audit-ordinary-development-v4.py'), 'closed-worker', '--spec', str(specification)], WORK / 'closed-execution')
        self.assertTrue(read(WORK / 'closed-proof.json')['durable_unchanged'])
        rejections = []
        for kind, source, required in (
            ('generation', generation.DEFAULT, ('certificate.pending.json', 'identity.json', 'protocol.json', 'parent-inventory.json', 'candidates.json', 'manifest.json')),
            ('reservations', data.DEFAULT, ('certificate.pending.json', 'identity.json', 'inventory.json', 'selection.json', 'manifest.json')),
            ('export', self.initial, ('certificate.json', 'model.json'))):
            for name in required:
                route = WORK / ('missing-' + kind + '-' + name)
                shutil.copytree(source, route); (route / name).unlink()
                with mock.patch.object(generation, '_facts', forbidden), mock.patch.object(data, 'Writer', forbidden), mock.patch.object(generation.old, 'Writer', forbidden), mock.patch.object(side, '_writer', forbidden):
                    with self.assertRaises((ValueError, NativeError, FileNotFoundError, KeyError)):
                        if kind == 'generation': generation.run(route)
                        elif kind == 'reservations': data.run(route)
                        else: side.initialize(self.manifest, route, fixture=True)
                rejections.append(kind + '/missing/' + name)
        # Unknown/missing journal prefixes must not reset the generation budget.
        route = WORK / 'missing-generation-first-chunk'; shutil.copytree(generation.DEFAULT, route)
        first_chunk = sorted((route / 'chunks').glob('*.json'))[0]; first_chunk.unlink()
        with mock.patch.object(generation, '_facts', forbidden), mock.patch.object(generation.old, 'Writer', forbidden):
            with self.assertRaises((ValueError, FileNotFoundError, KeyError)): generation.run(route)
        rejections.append('generation/missing-first-committed-chunk')
        route = WORK / 'missing-all-incomplete-committed-chunks'; shutil.copytree(WORK / 'generation-fresh', route)
        for name in ('catalog.json', 'certificate.pending.json', 'manifest.json', 'candidates.json'):
            (route / name).unlink()
        for path in (route / 'chunks').glob('*.json'): path.unlink()
        cursor_before = (route / 'cursor.json').read_bytes()
        from congklak.native import Native
        with mock.patch.object(generation_v3, '_facts', forbidden), mock.patch.object(Native, '__init__', forbidden):
            with self.assertRaisesRegex(ValueError, 'cursor|Cursor|chunk|Chunk'): generation_v3.run(route)
        self.assertEqual(cursor_before, (route / 'cursor.json').read_bytes())
        rejections.append('generation/incomplete-positive-cursor-all-chunks-lost-no-reset')
        route = WORK / 'corrupt-earlier-incomplete-committed-chunk'; shutil.copytree(WORK / 'generation-fresh', route)
        for name in ('catalog.json', 'certificate.pending.json', 'manifest.json', 'candidates.json'): (route / name).unlink()
        first_chunk = route / 'chunks/000000.json'; body = read(first_chunk)
        record = next(r for r in body['attempts'] if r.get('accepted'))
        record['probes'][0]['state']['controller'] = 1 - record['probes'][0]['state']['controller']
        first_chunk.write_text(json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
        with mock.patch.object(generation_v3, '_facts', forbidden), mock.patch.object(Native, '__init__', forbidden):
            with self.assertRaisesRegex(ValueError, 'prefix|Prefix|digest|Digest|chunk|Chunk'): generation_v3.run(route)
        rejections.append('generation/incomplete-valid-JSON-earlier-probe-controller-bitflip')
        # A lone orphan is not yet covered by the committed-prefix digest.
        # Verify its literal records before it can be promoted to that prefix.
        def orphan_route(name):
            route = WORK / name; route.mkdir()
            for filename in ('protocol.json', 'parent-inventory.json', 'identity.json'):
                shutil.copy2(WORK / 'generation-fresh' / filename, route / filename)
            generation_v3._cursor_write(route, []); (route / 'chunks').mkdir()
            shutil.copy2(WORK / 'generation-fresh/chunks/000000.json', route / 'chunks/000000.json')
            return route
        route = orphan_route('valid-single-orphan-literal-recovery')
        _, inv = generation_v3._identity(route); cursor_before = (route / 'cursor.json').read_bytes()
        first_before = snapshot(route)['chunks\\000000.json'] if 'chunks\\000000.json' in snapshot(route) else snapshot(route)['chunks/000000.json']
        literal_facts = generation_v3._facts
        with mock.patch.object(generation_v3, '_facts', wraps=literal_facts) as replay:
            recovered = generation_v3._load_chunks(route, inv)
        expected_attempts = [r['attempt'] for r in read(route / 'chunks/000000.json')['attempts'] if r['reason'] != 'cell_quota_filled']
        self.assertEqual([call.args[0] for call in replay.call_args_list], expected_attempts)
        self.assertLessEqual(len(expected_attempts), 64)
        self.assertEqual(cursor_before, (route / 'cursor.json').read_bytes())
        generation_v3._cursor_write(route, [route / 'chunks/000000.json'])
        self.assertEqual(read(route / 'cursor.json')['completed_chunks'], 1)
        with mock.patch.object(generation_v3, '_facts', forbidden), mock.patch.object(Native, '__init__', forbidden):
            aligned = generation_v3._load_chunks(route, inv)
        self.assertEqual(recovered, aligned)
        after = snapshot(route); first_after = after.get('chunks\\000000.json', after.get('chunks/000000.json'))
        self.assertEqual(first_before, first_after)
        route = orphan_route('corrupt-single-orphan-controller-refusal')
        first = route / 'chunks/000000.json'; body = read(first)
        record = next(r for r in body['attempts'] if r.get('accepted'))
        record['probes'][0]['state']['controller'] = 1 - record['probes'][0]['state']['controller']
        first.write_text(json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
        corrupted_before = snapshot(route)
        with mock.patch.object(generation_v3, '_facts', wraps=literal_facts) as corrupted_replay:
            with self.assertRaisesRegex(ValueError, 'orphan|Orphan|literal|Literal|differs'):
                generation_v3.run(route)
        self.assertLessEqual(len(corrupted_replay.call_args_list), 64)
        after = snapshot(route)
        for name, prior in corrupted_before.items(): self.assertEqual(after[name], prior)
        self.assertFalse((route / 'catalog.json').exists())
        rejections.append('generation/uncommitted-valid-JSON-orphan-child-controller-bitflip')
        for kind, source, name in (('generation', generation.DEFAULT, 'certificate.pending.json'), ('reservations', data.DEFAULT, 'certificate.pending.json'), ('export', self.initial, 'certificate.json')):
            route = WORK / ('corrupt-' + kind); shutil.copytree(source, route); (route / name).write_text('{}\n', encoding='utf-8')
            with mock.patch.object(generation, '_facts', forbidden), mock.patch.object(data, 'Writer', forbidden), mock.patch.object(generation.old, 'Writer', forbidden), mock.patch.object(side, '_writer', forbidden):
                with self.assertRaises((ValueError, NativeError, KeyError)):
                    if kind == 'generation': generation.run(route)
                    elif kind == 'reservations': data.run(route)
                    else: side.initialize(self.manifest, route, fixture=True)
            rejections.append(kind + '/corrupt-certificate')
        baseline = read(FOLDER / 'baseline.json')
        self.assertEqual(len(baseline['inputs']), 5950)
        for name, digest in baseline['inputs'].items(): self.assertEqual(sha(ROOT / name), digest, name)
        # Exercise the current observer read-only. Completion acceptance is
        # deliberately published by the root only after this independent audit.
        before = snapshot(FOLDER)
        observed = progress.snapshot(); self.assertEqual(before, snapshot(FOLDER))
        self.assertEqual(observed['execution_counters'], dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0))
        self.assertEqual(observed['current_cards'], 10)
        with mock.patch.object(progress, 'LOADED_SHA', '0' * 64):
            stale = progress.snapshot(); self.assertFalse(stale['data_and_inference_ready']); self.assertEqual(stale['current_work']['state'], 'hold')
        EVIDENCE['completed_failclosed_and_history'] = dict(guard_proof=rel(WORK / 'closed-proof.json'),
            missing_corrupt_rejections=rejections, required_evidence_not_recomputed=True,
            old_5950_inputs_preserved=True, dashboard_readonly=True, stale_source_holds=True,
            dashboard_optimizer_draws=0, browser_or_HTTP_used=False, training_started=False,
            valid_orphan_literal_attempts_rechecked=len(expected_attempts), valid_orphan_transition_probe_bound=2 * len(expected_attempts),
            corrupt_orphan_attempts_before_refusal=len(corrupted_replay.call_args_list),
            no_committed_prefix_redraw=True, invalid_orphan_bytes_and_cursor_preserved=True)


def run(output):
    global WORK
    WORK = Path(output).resolve(); WORK.mkdir(parents=True, exist_ok=False)
    source_folder = WORK / 'execution-sources'; source_folder.mkdir()
    for path in (Path(__file__), ROOT / 'tools/audit-ordinary-development.py'):
        shutil.copy2(path, source_folder / path.name)
    frozen_source = sha(__file__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OrdinaryDevelopmentIndependentAudit)
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert sha(__file__) == frozen_source, 'Executed audit source changed during run'
    methods = unittest.defaultTestLoader.getTestCaseNames(OrdinaryDevelopmentIndependentAudit)
    proof = dict(schema='ordinary-development-independent-methods-v1', passed=result.wasSuccessful(),
        automated_test_methods=methods, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, executed_source_sha256=frozen_source,
        evidence=EVIDENCE, process_kill_cases=KILLS, attempts=ATTEMPTS)
    save(WORK / 'methods.json', proof)
    return proof
