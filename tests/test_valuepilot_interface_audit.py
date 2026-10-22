"""Independent pre-training audit. No optimizer, teacher or game generation."""
from __future__ import annotations
from contextlib import ExitStack, closing
import copy
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'build/valuepilot-interface-v1'
EVIDENCE = {}
KILLS = []
ATTEMPTS = []
WORK = None


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
def forbidden(*args, **kwargs): raise AssertionError('Unexpected computation or publication on completed route')


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import forbidden in metadata/completed audit')
        return None


def run_process(argv, directory, timeout=120):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        result = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, timeout=timeout)
    record = dict(argv=[str(v) for v in argv], returncode=result.returncode,
                  elapsed_seconds=time.monotonic() - started, logs=rel(directory))
    ATTEMPTS.append(record)
    save(directory / 'execution.json', record)
    if result.returncode:
        raise AssertionError('Worker failed: ' + str(record) + '\n' + (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:])
    return record


def kill_process(argv, directory, marker, boundary, timeout=120):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); process = None
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err)
            deadline = time.monotonic() + timeout
            while not marker.exists():
                if process.poll() is not None:
                    raise AssertionError('Worker ended before pause: ' + (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:])
                if time.monotonic() >= deadline:
                    raise AssertionError('Pause marker timeout')
                time.sleep(.05)
            marker_body = read(marker)
            assert marker_body.get('pid') == process.pid, 'Wrong process marker'
            assert marker_body.get('event', marker_body.get('point')) == boundary, 'Wrong durable boundary'
            process.kill(); code = process.wait(timeout=30)
            assert code != 0, 'Worker was not actually killed'
        finally:
            if process is not None and process.poll() is None:
                process.kill(); process.wait(timeout=30)
    record = dict(boundary=boundary, argv=[str(v) for v in argv], pid=process.pid,
                  returncode=code, marker=rel(marker), marker_sha256=sha(marker),
                  actual_process_kill=True, elapsed_seconds=time.monotonic() - started,
                  logs=rel(directory))
    KILLS.append(record); ATTEMPTS.append(record); save(directory / 'execution.json', record)
    return record


def boundary(active, actor=0, forced=False):
    pits = [[0] * 7, [0] * 7]
    if forced:
        pits[1 - actor][2] = active
    else:
        # Independent spread order, with both actors occupied when possible.
        for n in range(active): pits[n % 2][(3 * (n // 2) + 1) % 7] += 1
    remainder = 98 - active
    return dict(pits=pits, stores=[remainder // 3, remainder - remainder // 3], controller=actor)


def literal(base, weights, position):
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


def call_closed(spec):
    """Fresh interpreter guard for completed API/CLI and unchanged durable data."""
    from congklak import value_sidecar as side, valuepilot_reservations as data
    guard = NoTorch(); sys.meta_path.insert(0, guard)
    directories = [Path(p) for p in spec['directories']]
    before = {str(p): snapshot(p) for p in directories}
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(side, '_writer', forbidden))
        stack.enter_context(mock.patch.object(data, 'Writer', forbidden))
        stack.enter_context(mock.patch.object(data, 'inventory', forbidden))
        stack.enter_context(mock.patch.object(side.features_relay, 'encode', forbidden))
        stack.enter_context(mock.patch.object(side.Model, 'reference', forbidden))
        stack.enter_context(mock.patch.object(side.SidecarEvaluator, '__init__', forbidden))
        from congklak import quality_search
        stack.enter_context(mock.patch.object(quality_search, 'search', forbidden))
        for call in spec.get('calls', []):
            if call['kind'] == 'data': data.run(call['path'])
            else: side.initialize(call['manifest'], call['path'], fixture=True)
        for call in spec.get('clis', []):
            import runpy
            old = sys.argv
            try:
                sys.argv = call
                try: runpy.run_path(call[0], run_name='__main__')
                except SystemExit as exc: assert exc.code in (None, 0), str(exc.code)
            finally: sys.argv = old
    assert 'torch' not in sys.modules
    assert before == {str(p): snapshot(p) for p in directories}, 'Completed bytes/sizes/mtimes changed'
    save(spec['proof'], dict(passed=True, durable_unchanged=True, no_writer=True, no_torch=True,
                             no_features=True, no_forwards=True, no_search=True, before=before))


class ValuepilotIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from congklak import value_sidecar as side
        from congklak.neural_relay import Model
        cls.manifest = WORK / 'engineering-input-manifest.json'
        save(cls.manifest, dict(schema='independent-value-interface-fixture-v1', training_eligible=False,
                                targets_read=0, optimizer_steps=0, purpose='Conformance and publication recovery only'))
        cls.initial = WORK / 'fresh-initialization'
        cls.torch_python = read(ROOT / 'build/training-runtime.json')['python']
        argv = [cls.torch_python, str(ROOT / 'tools/export-value-sidecar.py'), 'initialize',
                '--manifest', str(cls.manifest), '--output', str(cls.initial), '--engineering-fixture']
        run_process(argv, WORK / 'initialization-execution')
        cls.model = side.Model(cls.initial / 'model.json'); cls.base = Model(side.BASE)
        weights = copy.deepcopy(cls.model.weights)
        weights['v'] = [[(-1 if i % 2 else 1) * (i + 1) / 256 for i in range(16)]]
        weights['c'] = [-.0625]
        cls.synthetic_path = WORK / 'synthetic-nonzero.json'
        save(cls.synthetic_path, side.make_export(weights, cls.manifest,
             dict(engineering_fixture=True, training_eligible=False, optimizer_steps=0,
                  purpose='Independent fixed nonzero branch fixture; never a trained candidate')))
        cls.synthetic = side.Model(cls.synthetic_path)
        inv = read(ROOT / 'build/neurodiag-inventory.json')
        # Every admitted source row of the eligible existing H8 corpus, not just
        # the 256 roots used by the preceding diagnostic.
        cls.positions = [r['state'] for r in inv['primary_pool']]
        cls.positions += [boundary(a, c) for a in (0, 4, 5, 24, 25, 49, 50, 74, 75, 98) for c in (0, 1)]
        cls.positions += [boundary(a, c, True) for a in (4, 24, 25, 49, 74, 75) for c in (0, 1)]
        EVIDENCE['engineering_initialization'] = dict(model_sha256=cls.model.id,
             model_file_sha256=cls.model.file_sha256, fixture=True, training_eligible=False,
             corpus_rows=len(cls.positions), existing_H8_corpus_rows=len(inv['primary_pool']))

    def test_01_full_graph_and_real_hold(self):
        from congklak import config, neurodiag_data as old, valuepilot_reservations as data
        package = data.DEFAULT
        guard = NoTorch(); sys.meta_path.insert(0, guard)
        try:
            with mock.patch.object(data, 'Writer', forbidden), mock.patch.object(data, 'inventory', forbidden):
                manifest = data.verify(package)
                self.assertEqual(data.run(package), manifest)
                with self.assertRaisesRegex(ValueError, 'held'): data.admit(package)
        finally: sys.meta_path.remove(guard)
        inv = read(package / 'inventory.json'); selection = read(package / 'selection.json')
        self.assertIs(manifest['admitted'], False); self.assertEqual(manifest['status'], 'hold')
        self.assertTrue(all(n == 0 for n in manifest['counts'].values()))
        self.assertEqual(manifest['availability']['arena/ordinary']['components'], 0)
        self.assertTrue(all(selection[k] == [] for k in ('train', 'fit_check', 'fit_cases', 'arena')))
        components = inv['components']; by_f = {}; by_q = {}
        for cid, component in components.items():
            self.assertEqual(cid, config.digest(component))
            for f in component['families']: self.assertNotIn(f, by_f); by_f[f] = cid
            for q in component['quotients']: self.assertNotIn(q, by_q); by_q[q] = cid
        checked = 0
        def check(family, state):
            nonlocal checked
            self.assertEqual(by_f[family], by_q[old.quotient(state)]); checked += 1
        for name in ('deeper-9000-v2', *old.HISTORIES):
            with closing(old.ro(name)) as db:
                for text, family in db.execute("SELECT json_extract(body,'$.state'),json_extract(body,'$.family') FROM roots"):
                    check(family, json.loads(text))
        with closing(old.ro('ordinary-selfplay-pilot-v1')) as db:
            for family, text in db.execute("SELECT json_extract(body,'$.family'),json_extract(body,'$.candidates') FROM families"):
                for row in json.loads(text): check(family, row['state'])
        for name in ('selfplay-training-pilot-v1', 'ordinary-selfplay-pilot-v1'):
            with closing(old.ro(name)) as db:
                for text, family in db.execute("SELECT json_extract(body,'$.state'),json_extract(body,'$.family') FROM sp_rows"):
                    check(family, json.loads(text))
        prior = read(ROOT / 'build/neurodiag-inventory.json')
        for kind, values, lookup in (('family', prior['protected_families'], by_f),
                                     ('quotient', prior['protected_quotients'], by_q)):
            for value in values:
                if value in lookup:
                    self.assertTrue(any(r.startswith('optimizer/') for r in components[lookup[value]]['reasons']))
        panel = read(ROOT / 'build/neurodiag-panel.json')
        for group in ('primary', 'paired', 'MC'):
            for row in panel[group]:
                check(row['family'], row['state'])
                self.assertTrue(any('neurodiag' in r for r in components[by_f[row['family']]]['reasons']))
        for row in inv['protected_cases']:
            q = old.quotient(row['state'])
            self.assertIn(q, by_q)
            self.assertTrue(any(r.startswith('arena/') for r in components[by_q[q]]['reasons']))
        # Independent deterministic selection rerun must preserve the hold;
        # targets, teacher helpers and a neural evaluator remain unavailable.
        with mock.patch.object(old, 'teacher_target', forbidden), mock.patch.object(old, 'source_item', forbidden):
            self.assertEqual(data.select(inv), selection)
        EVIDENCE['full_graph_and_hold'] = dict(origin_observations_checked=checked,
            full_components=len(components), availability=manifest['availability'],
            readiness_blockers=manifest['readiness_blockers'], actual_reservations=manifest['counts'],
            targets_read=0, new_searches=0, new_games=0, optimizer_steps=0)

    def test_02_native_full_corpus_policy_domain_and_batch(self):
        from congklak import value_sidecar as side
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        maxima = {'zero_native_literal_seed': 0., 'nonzero_native_literal_seed': 0., 'native_literal_logit': 0., 'score_units': 0}
        changed = 0; outside = 0; comparisons = 0; backends = []
        for backend in ('scalar', 'auto'):
            with CpuBatchEvaluator(side.BASE, backend=backend) as base, \
                 side.SidecarEvaluator(self.model, backend=backend) as zero, \
                 side.SidecarEvaluator(self.synthetic, backend=backend) as candidate:
                backends.append(base.backend)
                for start in range(0, len(self.positions), 32):
                    group = self.positions[start:start + 32]
                    aa = base.evaluate_many(group); zz = zero.evaluate_many(group); bb = candidate.evaluate_many(group)
                    for p, a, z, b in zip(group, aa, zz, bb):
                        comparisons += 1
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *z['policy_logits']))
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *b['policy_logits']))
                        self.assertEqual(a['value_player0'], z['value_player0'])
                        for key in ('feature_simulations', 'feature_deposits', 'native_actual_evals'):
                            self.assertEqual(a[key], z[key]); self.assertEqual(a[key], b[key])
                        rz = literal(self.base, self.model.weights, p); rb = literal(self.base, self.synthetic.weights, p)
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
                    # Separately compare scalar single-row and batched calls on
                    # the boundaries and a deterministic corpus subsample.
                    if start % (32 * 16) == 0 or start + 32 >= len(self.positions):
                        for p, b in zip(group, bb):
                            one = candidate.evaluate_many([p])[0]
                            for key in ('value_player0', 'policy_logits', 'feature_simulations', 'feature_deposits'):
                                self.assertEqual(one[key], b[key])
                counts = candidate.snapshot_counters()
                self.assertGreater(counts['branch_calls'], 0); self.assertGreater(counts['branch_bypasses'], 0)
        self.assertGreater(changed, 0)
        EVIDENCE['native_conformance'] = dict(corpus_rows=len(self.positions), policy_comparisons=comparisons,
            backends=backends, synthetic_in_gate_changed=changed, outside_gate_equal=outside,
            additional_feature_simulations=0, maximum_errors=maxima, tolerance=1e-10)

    def test_03_low_active_fixed_128_improved_search(self):
        from congklak import quality_search, value_sidecar as side
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        positions = [boundary(a, c) for a in (4, 5, 24) for c in (0, 1)]
        positions += [boundary(24, c, True) for c in (0, 1)]
        reports = []
        with CpuBatchEvaluator(side.BASE, backend='scalar') as base, \
             side.SidecarEvaluator(self.synthetic, backend='scalar') as candidate:
            for p in positions:
                base.begin_decision(); candidate.begin_decision()
                a = quality_search.search(p, base, simulations=128, c_puct=.25)
                b = quality_search.search(p, candidate, simulations=128, c_puct=.25)
                quality_search.validate_report(a, p, 128, c_puct=.25)
                quality_search.validate_report(b, p, 128, c_puct=.25)
                for key in a:
                    if key in ('timing', 'work'): continue
                    self.assertEqual(a[key], b[key], key)
                for key in a['work']:
                    if key != 'inference_ms': self.assertEqual(a['work'][key], b['work'][key], key)
                reports.append(dict(position=p, control=a, synthetic=b))
        save(WORK / 'low-active-fixed128.json', reports)
        EVIDENCE['low_active_fixed128'] = dict(roots=len(positions), searches=2 * len(positions),
             requested_backups=256 * len(positions), exact_actions_visits_Q_and_physical_work=True,
             improved_search_protocol=quality_search.PROTOCOL, ignored=['timing', 'work.inference_ms'])

    def test_04_schema_identity_initialization_and_derivatives(self):
        from congklak import config, value_sidecar as side, quality_search
        from congklak.native import NativeError
        self.assertEqual(side.tensor_sha(self.model.weights), side.INITIAL_TENSOR_SHA)
        self.assertEqual(self.model.weights['v'], [[0.] * 16]); self.assertEqual(self.model.weights['c'], [0.])
        self.assertEqual(self.model.envelope['model']['base_export'], self.base.envelope)
        self.assertEqual(self.model.identity['search']['protocol'], quality_search.PROTOCOL)
        self.assertIn('congklak/quality_search.py', self.model.identity['sources'])
        rejected = []
        mutations = [
            ('schema', lambda e: e['model'].__setitem__('schema', 'congklak-mlp-export-v4')),
            ('gate', lambda e: e['model']['bindings']['gate'].__setitem__('minimum_active', 24)),
            ('search', lambda e: e['model']['bindings']['search'].__setitem__('c_puct', 1.5)),
            ('build', lambda e: e['model']['bindings'].__setitem__('dll_sha256', '0' * 64)),
            ('input', lambda e: e['model']['bindings']['input_manifest'].__setitem__('sha256', '0' * 64)),
            ('encoder', lambda e: e['model']['bindings'].__setitem__('encoder', 'wrong')),
            ('rules', lambda e: e['model']['bindings'].__setitem__('rules_digest', '0' * 64)),
            ('extra-tensor', lambda e: e['model']['branch_weights'].update(extra=[0.])),
            ('base-policy', lambda e: e['model']['base_export']['model']['weights']['policy.bias'].__setitem__(0, 0.)),
        ]
        for name, mutate in mutations:
            export = copy.deepcopy(self.model.envelope); mutate(export)
            export['sha256'] = config.digest(export['model']); path = WORK / ('rehashed-' + name + '.json'); save(path, export)
            with self.assertRaises((NativeError, ValueError, KeyError)): side.Model(path)
            rejected.append(name)
        # A general synthetic model may load, but can never masquerade as the
        # initialized step-zero model even when all public digests are updated.
        with self.assertRaisesRegex(NativeError, 'Initialization|provenance'): side._initial_model(self.synthetic)
        with self.assertRaises((NativeError, ValueError, KeyError)):
            from congklak.neural_relay import Model
            Model(self.initial / 'model.json')
        # The isolated worker independently reconstructs FP32 initialization,
        # performs CPU forward conformance and finite-difference derivatives.
        spec = WORK / 'torch-spec.json'
        save(spec, dict(initial=str(self.initial / 'model.json'), synthetic=str(self.synthetic_path),
             positions=self.positions, output=str(WORK / 'torch-proof.json')))
        run_process([self.torch_python, str(ROOT / 'tools/audit-valuepilot-interface.py'), 'torch-worker', '--spec', str(spec)], WORK / 'torch-execution')
        proof = read(WORK / 'torch-proof.json'); self.assertTrue(proof['passed'])
        self.assertLessEqual(proof['maximum_errors']['seeds'], .001)
        self.assertLessEqual(proof['maximum_errors']['logits'], 1e-5)
        self.assertLessEqual(proof['maximum_errors']['score_units'], 1)
        EVIDENCE['strict_schema_and_numeric'] = dict(rehashed_mutations_rejected=rejected,
             old_v4_loader_refuses=True, stepzero_synthetic_refuses=True, torch=proof)

    def test_05_real_publication_kills_and_prefix_recovery(self):
        from congklak import valuepilot_reservations as data, value_sidecar as side
        # Use the actual frozen HOLD inventory. No engineering fixture can
        # manufacture a complete production split or release training.
        prepared_names = ('inventory.json', 'selection.json', 'identity.json')
        fresh = WORK / 'data-fresh'; fresh.mkdir()
        for name in prepared_names: shutil.copy2(data.DEFAULT / name, fresh / name)
        run_process([sys.executable, str(ROOT / 'tools/reserve-valuepilot.py'), 'run', '--out', str(fresh)], WORK / 'data-fresh-execution')
        expected_manifest = (fresh / 'manifest.json').read_bytes()
        for point in ('before_catalog_commit', 'after_catalog_commit'):
            route = WORK / ('data-' + point); route.mkdir()
            for name in prepared_names: shutil.copy2(data.DEFAULT / name, route / name)
            marker = WORK / ('data-' + point + '-pause.json')
            argv = [sys.executable, str(ROOT / 'tools/reserve-valuepilot.py'), 'run', '--out', str(route),
                    '--pause-at', point, '--pause-marker', str(marker)]
            kill_process(argv, WORK / ('data-kill-' + point), marker, point)
            prefix = snapshot(route)
            run_process([sys.executable, str(ROOT / 'tools/reserve-valuepilot.py'), 'run', '--out', str(route)], WORK / ('data-resume-' + point))
            after = snapshot(route)
            self.assertEqual((route / 'manifest.json').read_bytes(), expected_manifest)
            for name, prior in prefix.items(): self.assertEqual(after[name], prior, name)
            self.assertEqual(data.verify(route), data.verify(fresh))
            if point == 'after_catalog_commit': self.assertEqual(prefix, after)
        expected_model = (self.initial / 'model.json').read_bytes()
        expected_certificate = (self.initial / 'certificate.json').read_bytes()
        for point in side.HOOKS:
            route = WORK / ('export-' + point)
            argv = [self.torch_python, str(ROOT / 'tools/export-value-sidecar.py'), 'initialize',
                    '--manifest', str(self.manifest), '--output', str(route), '--engineering-fixture', '--pause-at', point]
            kill_process(argv, WORK / ('export-kill-' + point), route / 'pause.json', point)
            prefix = snapshot(route)
            # Once a model is committed, the default interpreter (without
            # Torch installed) must finish strictly from persisted bytes.
            executable = self.torch_python if point == 'before_model_commit' else sys.executable
            run_process([executable, str(ROOT / 'tools/export-value-sidecar.py'), 'initialize',
                         '--manifest', str(self.manifest), '--output', str(route), '--engineering-fixture'],
                         WORK / ('export-resume-' + point))
            after = snapshot(route)
            self.assertEqual((route / 'model.json').read_bytes(), expected_model)
            self.assertEqual((route / 'certificate.json').read_bytes(), expected_certificate)
            for name, prior in prefix.items(): self.assertEqual(after[name], prior, name)
            self.assertFalse(side.verify(route)['training_eligible'])
            if point == 'after_certificate_commit': self.assertEqual(prefix, after)
        self.assertEqual(len(KILLS), 6)
        EVIDENCE['publication_recovery'] = dict(real_kills=6, reservation_boundaries=2, export_boundaries=4,
             preserved_all_durable_prefix_bytes_sizes_mtimes=True, committed_initialization_not_regenerated=True,
             fresh_resumed_model_and_certificate_exact=True, held_data_never_admitted=True,
             optimizer_steps=0, training_recovery_executed=False, arena_recovery_executed=False)

    def test_06_completed_corruption_readonly_dashboard_and_history(self):
        from congklak import config, value_sidecar as side, valuepilot_reservations as data
        from congklak import valuepilot_interface_progress as progress
        from congklak.native import NativeError
        # Both API and public CLI execute inside fresh guards that trap writer,
        # inventory, encoder, forward, search and Torch import entrypoints.
        spec = WORK / 'closed-spec.json'
        save(spec, dict(directories=[str(data.DEFAULT), str(self.initial)], proof=str(WORK / 'closed-proof.json'),
            calls=[dict(kind='data', path=str(data.DEFAULT)), dict(kind='export', path=str(self.initial), manifest=str(self.manifest))],
            clis=[[str(ROOT / 'tools/reserve-valuepilot.py'), 'run', '--out', str(data.DEFAULT)],
                  [str(ROOT / 'tools/export-value-sidecar.py'), 'initialize', '--manifest', str(self.manifest),
                   '--output', str(self.initial), '--engineering-fixture']]))
        run_process([sys.executable, str(ROOT / 'tools/audit-valuepilot-interface.py'), 'closed-worker', '--spec', str(spec)], WORK / 'closed-execution')
        self.assertTrue(read(WORK / 'closed-proof.json')['durable_unchanged'])
        rejections = []
        for family, source, required in (
            ('data', data.DEFAULT, ('certificate.pending.json', 'inventory.json', 'identity.json', 'selection.json', 'manifest.json')),
            ('export', self.initial, ('certificate.json', 'model.json'))):
            for name in required:
                folder = WORK / ('missing-' + family + '-' + name); folder.mkdir()
                for path in source.iterdir():
                    if path.is_file() and path.name != name: shutil.copy2(path, folder / path.name)
                with mock.patch.object(data, 'Writer', forbidden), mock.patch.object(side, '_writer', forbidden):
                    with self.assertRaises((ValueError, NativeError, FileNotFoundError, KeyError)):
                        if family == 'data': data.run(folder)
                        else: side.initialize(self.manifest, folder, fixture=True)
                rejections.append(family + '/missing/' + name)
        for family, source, name in (('data', data.DEFAULT, 'certificate.pending.json'), ('export', self.initial, 'certificate.json')):
            folder = WORK / ('corrupt-' + family); folder.mkdir()
            for path in source.iterdir():
                if path.is_file(): shutil.copy2(path, folder / path.name)
            (folder / name).write_text('{}\n', encoding='utf-8')
            with mock.patch.object(data, 'Writer', forbidden), mock.patch.object(side, '_writer', forbidden):
                with self.assertRaises((ValueError, NativeError, KeyError)):
                    if family == 'data': data.run(folder)
                    else: side.initialize(self.manifest, folder, fixture=True)
            rejections.append(family + '/corrupt-certificate')
        # Current production manifest is a certified HOLD. Production exporter
        # refuses it before it can create an output directory or import Torch.
        not_created = ROOT / 'data/neural-value-pilot-v1/interface-initial'
        before_exists = not_created.exists()
        self.assertFalse(before_exists)
        with mock.patch.object(side, '_writer', forbidden):
            with self.assertRaises((NativeError, ValueError)):
                side.initialize(data.DEFAULT / 'manifest.json', not_created)
        self.assertFalse(not_created.exists())
        # Initialization cannot bless a synthetic branch even if its public
        # weight/export hashes have been consistently recalculated.
        mismatched = WORK / 'mismatched-initialization'; mismatched.mkdir()
        export = copy.deepcopy(self.model.envelope)
        export['model']['branch_weights']['v'][0][0] = .125
        export['model']['branch_weights_sha256'] = side.tensor_sha(export['model']['branch_weights'])
        export['sha256'] = config.digest(export['model']); save(mismatched / 'model.json', export)
        with self.assertRaisesRegex(NativeError, 'Initialization tensors'):
            side.initialize(self.manifest, mismatched, fixture=True)
        rejections.append('fully-rehashed-nonzero-stepzero')
        # Observer must make engineering success and training readiness visibly
        # different, remain read-only, and refuse nonzero execution counters.
        status_folder = WORK / 'observer-fixture'; status_folder.mkdir()
        status = read(BASE / 'stage-status.json')
        save(status_folder / 'stage-status.json', status)
        checksum = sha(status_folder / 'stage-status.json')
        save(status_folder / 'acceptance.json', dict(passed=True, training_ready=False,
             new_inputs={rel(status_folder / 'stage-status.json'): checksum}, next_step='HOLD ordinary arena coverage',
             readiness_blockers=['ordinary arena component shortage']))
        with mock.patch.object(progress, 'FOLDER', status_folder):
            before = snapshot(status_folder); observed = progress.snapshot(); self.assertEqual(before, snapshot(status_folder))
            self.assertTrue(observed['implementation_accepted']); self.assertFalse(observed['pilot_training_ready'])
            self.assertEqual(observed['current_work']['state'], 'hold')
            self.assertFalse(observed['active_tools'])
            self.assertEqual(len(observed['tools']), 17)
            with mock.patch.object(progress, 'LOADED_SOURCE_SHA256', '0' * 64):
                stale = progress.snapshot(); self.assertFalse(stale['implementation_accepted'])
                self.assertEqual(stale['current_work']['state'], 'hold')
                self.assertIn('Restart', stale['current_work']['message'])
            bad = copy.deepcopy(status); bad['execution_counters']['training_steps'] = 1
            (status_folder / 'stage-status.json').write_text(config.canonical(bad) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError): progress.snapshot()
        baseline = read(BASE / 'baseline.json')
        self.assertEqual(len(baseline['inputs']), 5627)
        for name, digest in baseline['inputs'].items(): self.assertEqual(sha(ROOT / name), digest, name)
        EVIDENCE['completed_and_corrupt'] = dict(rejections=rejections, required_missing_never_recomputed=True,
             guard_proof=rel(WORK / 'closed-proof.json'), production_hold_cannot_initialize=True,
             original_5627_inputs_current=True, dashboard_engineering_accepted_training_held=True,
             browser_or_HTTP_used=False, no_optimizer=True)


def run(output):
    global WORK
    WORK = Path(output).resolve(); WORK.mkdir(parents=True, exist_ok=False)
    frozen_source = sha(__file__)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValuepilotIndependentAudit)
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert sha(__file__) == frozen_source, 'Executed audit source changed during run'
    methods = [name for name in unittest.defaultTestLoader.getTestCaseNames(ValuepilotIndependentAudit)]
    proof = dict(schema='valuepilot-interface-independent-methods-v1', passed=result.wasSuccessful(),
        automated_test_methods=methods, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, executed_source_sha256=frozen_source,
        evidence=EVIDENCE, process_kill_cases=KILLS, attempts=ATTEMPTS)
    save(WORK / 'methods.json', proof)
    return proof


