"""Independent acceptance of the bounded value trainer (no production optimization)."""
from __future__ import annotations
from contextlib import ExitStack
import copy
import ctypes
import hashlib
import importlib.abc
import importlib.util
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
import traceback
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/value-training-v1'
WORK = None
FIXTURES = None
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
def forbidden(*args, **kwargs): raise AssertionError('Unexpected work on certified completed route')
def fp32(value): return struct.unpack('<f', struct.pack('<f', value))[0]

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import forbidden on metadata/completed route')
        return None

def execute(argv, directory, timeout=180, expected=0):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        result = subprocess.run([str(x) for x in argv], cwd=ROOT, stdout=out, stderr=err, timeout=timeout)
    record = dict(argv=[str(v) for v in argv], returncode=result.returncode,
                  elapsed_seconds=time.monotonic() - started, logs=rel(directory))
    ATTEMPTS.append(record); save(directory / 'execution.json', record)
    assert (result.returncode == expected if expected is not None else result.returncode != 0), record
    return record

def process_identity(pid):
    assert type(pid) is int and pid > 0
    script = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; if ($p) { [pscustomobject]@{pid=$p.ProcessId;parent=$p.ParentProcessId;command=$p.CommandLine;creation=[string]$p.CreationDate} | ConvertTo-Json -Compress }"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip(), 'Marked worker is no longer running'
    return json.loads(result.stdout)

def kill_worker(argv, directory, marker, boundary, timeout=180, worker_script=None):
    """Kill the marked actual Windows descendant with a stable process handle."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); process = None; handle = None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]; kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]; kernel.TerminateProcess.restype = ctypes.c_int
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]; kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]; kernel.CloseHandle.restype = ctypes.c_int
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        try:
            process = subprocess.Popen([str(x) for x in argv], cwd=ROOT, stdout=out, stderr=err)
            deadline = time.monotonic() + timeout
            while not marker.exists():
                assert process.poll() is None, (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:]
                assert time.monotonic() < deadline, 'Pause marker timeout'
                time.sleep(.05)
            body = read(marker); assert body.get('event', body.get('point')) == boundary, body
            worker_pid = body['pid']; details = process_identity(worker_pid)
            lineage = [details]; current = details
            while current['pid'] != process.pid and len(lineage) < 5:
                current = process_identity(current['parent']); lineage.append(current)
            assert lineage[-1]['pid'] == process.pid, 'Marked worker is not our launched descendant'
            expected_script = Path(worker_script or argv[1]).resolve()
            assert str(expected_script).casefold() in details['command'].casefold(), 'Worker script differs'
            handle = kernel.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, worker_pid)
            assert handle
            again = process_identity(worker_pid)
            assert again == details, 'Worker identity changed after opening handle'
            assert kernel.TerminateProcess(handle, 1)
            assert kernel.WaitForSingleObject(handle, 30000) == 0
            code = process.wait(timeout=30)
            if worker_pid == process.pid: assert code != 0
        finally:
            if handle: kernel.CloseHandle(handle)
            if process is not None and process.poll() is None: process.kill(); process.wait(timeout=30)
    record = dict(boundary=boundary, argv=[str(v) for v in argv], launcher_pid=process.pid,
        worker_pid=worker_pid, worker_identity=details, verified_lineage=lineage,
        returncode=code, marker=rel(marker), marker_sha256=sha(marker), actual_process_kill=True,
        elapsed_seconds=time.monotonic() - started, logs=rel(directory))
    KILLS.append(record); ATTEMPTS.append(record); save(directory / 'execution.json', record)
    return record

def pinned_inputs():
    paths = [ROOT / 'build/ordinary-development-v1/acceptance.json', ROOT / 'build/training-runtime.json',
             ROOT / 'docs/NEURAL_VALUE_PILOT_PLAN.md', ROOT / 'tests/test_value_training_audit.py',
             ROOT / 'tools/audit-value-training.py']
    paths += list((ROOT / 'congklak').glob('value_training*.py'))
    paths += list((ROOT / 'tools').glob('*value-training*'))
    paths += list((ROOT / 'tests').glob('test_value_training*.py'))
    for directory in (ROOT / 'data/neural-value-pilot-v2/reservations', ROOT / 'data/neural-value-pilot-v2/interface-initial'):
        paths += files(directory)
    return {rel(p): sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix != '.pyc'}

def literal(base, weights, position):
    """Independent algebra preserving established FP32-weight/f64-dot order."""
    from congklak import features_relay
    from congklak.neural_relay import rounded
    x, d, r = features_relay.encode(position, 127)
    if not r: return float(d), d * 256, [0.] * 7
    hidden = list(x)
    for prefix in ('layers.0', 'layers.2'):
        hidden = [max(0., sum(a * b for a, b in zip(row, hidden)) + bias)
                  for row, bias in zip(base.weights[prefix + '.weight'], base.weights[prefix + '.bias'])]
    value = sum(a * b for a, b in zip(base.weights['layers.4.weight'][0], hidden)) + base.weights['layers.4.bias'][0]
    policy = [sum(a * b for a, b in zip(row, hidden)) + bias
              for row, bias in zip(base.weights['policy.weight'], base.weights['policy.bias'])]
    if 25 <= sum(map(sum, position['pits'])) <= 74:
        branch = [max(0., sum(a * b for a, b in zip(row, x)) + bias)
                  for row, bias in zip(weights['A'], weights['b'])]
        value += sum(a * b for a, b in zip(weights['v'][0], branch)) + weights['c'][0]
    residual = r * math.tanh(value)
    return d + residual, d * 256 + rounded(residual * 256), policy

def closed_worker(specification):
    guard = NoTorch(); sys.meta_path.insert(0, guard)
    from congklak import value_training as trainer, value_training_data as data
    from congklak import value_training_resources as resources, value_sidecar_v2 as side, quality_search
    from congklak.neural_relay import NeuralNative
    from congklak.native import Native
    directories = [Path(p) for p in specification['directories']]
    before = {str(p): snapshot(p) for p in directories}
    with ExitStack() as stack:
        for obj, name in ((trainer, 'Writer'), (trainer, '_objects'), (trainer, '_runtime'),
                          (trainer, '_load_checkpoint'), (trainer, '_finish'),
                          (data, 'Writer'), (data, '_rows'), (data, '_load'),
                          (resources, 'bind_worker'), (resources, 'assert_worker_lease'),
                          (side, '_writer'), (side.Model, 'reference'), (side.features_relay, 'encode'),
                          (Native, '__init__'), (NeuralNative, '__init__')):
            stack.enter_context(mock.patch.object(obj, name, forbidden))
        for name in ('search', 'choose'):
            if hasattr(quality_search, name): stack.enter_context(mock.patch.object(quality_search, name, forbidden))
        output = Path(specification['training'])
        first = trainer.completed(output)
        assert trainer.status(output) == first
        assert trainer.run(output, fixture=True) == first
        assert data.prepare() == data.admit()
        old = sys.argv
        try:
            for operation in ('verify', 'status', 'run'):
                sys.argv = ['train-value-sidecar.py', operation, '--output', str(output), '--engineering-fixture']
                assert trainer.main() == 0
        finally: sys.argv = old
    assert before == {str(p): snapshot(p) for p in directories}, 'Completed bytes/sizes/mtime changed'
    assert 'torch' not in sys.modules
    save(specification['output'], dict(passed=True, APIs=5, CLI_operations=3,
        Torch_imports=0, writers=0, optimizer_steps=0, features=0, forwards=0, searches=0,
        bytes_sizes_mtimes_preserved=True))

def clone_completed(source, destination):
    """Rebase only the copy's certificate paths, then prove it valid first."""
    from congklak import value_training as trainer
    shutil.copytree(source, destination)
    cert = trainer.unseal(destination / 'certificate.json')
    cert['files'] = {str(destination / Path(name).relative_to(source)): digest for name, digest in cert['files'].items()}
    trainer.atomic(destination / 'certificate.json', trainer.envelope(cert))
    trainer.atomic(destination / 'COMPLETED.json', {'certificate_sha256': sha(destination / 'certificate.json')})
    trainer.completed(destination)

def clone_cache(source, destination):
    from congklak import value_training_data as data
    shutil.copytree(source, destination)
    cert = read(destination / 'certificate.json')
    cert['files'] = {rel(destination / Path(name).resolve().relative_to(source)): digest for name, digest in
                     ((str((ROOT / key).resolve()), digest) for key, digest in cert['files'].items())}
    (destination / 'certificate.json').unlink(); data.atomic(destination / 'certificate.json', cert)
    catalog = read(destination / 'catalog.json'); catalog['certificate_sha256'] = sha(destination / 'certificate.json')
    (destination / 'catalog.json').unlink(); data.atomic(destination / 'catalog.json', catalog)
    (destination / 'complete.json').unlink()
    data.atomic(destination / 'complete.json', dict(schema=data.SCHEMA + '-complete', catalog_sha256=sha(destination / 'catalog.json')))
    data.admit(destination)

def run(output, methods=None, fixtures_from=None):
    global WORK, FIXTURES
    WORK = Path(output).resolve(); WORK.mkdir(parents=True, exist_ok=False)
    FIXTURES = Path(fixtures_from).resolve() if fixtures_from else None
    source = pinned_inputs(); save(WORK / 'source-before.json', source)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValueTrainingAudit)
    if methods: suite = unittest.TestSuite(t for t in suite if t._testMethodName in methods)
    stream = (WORK / 'methods.log').open('w', encoding='utf-8')
    started = time.monotonic()
    try: result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    finally: stream.close()
    now = pinned_inputs()
    same = source == now
    proof = dict(schema='value-training-independent-methods-v1', passed=result.wasSuccessful() and same,
        independent_of_implementation=True, tests_run=result.testsRun, failed=len(result.failures), errors=len(result.errors),
        source_unchanged=same, inputs=source, elapsed_seconds=time.monotonic() - started,
        evidence=EVIDENCE, process_kill_cases=KILLS, attempts=ATTEMPTS,
        failures=[dict(method=str(test),traceback=trace) for test,trace in result.failures + result.errors])
    save(WORK / 'methods.json', proof)
    return proof

class ValueTrainingAudit(unittest.TestCase):
    """Independent metadata, numerics, recovery, budget and completed-route checks."""
    def test_01_reserved_H8_target_and_FP32_cache_provenance(self):
        from collections import Counter
        from congklak import config, value_training_data as data, valuepilot_reservations_v2 as reservations, features_relay
        manifest = data.admit(); selection = read(reservations.DEFAULT / 'selection.json')
        self.assertEqual(set(manifest['role_files']), {'train', 'fit_check'})
        source = ROOT / 'data/deeper-9000-v2/work.sqlite3'
        feature_files = {}; counts = {}; rows_checked = 0; endpoint_count = 0
        db = sqlite3.connect('file:' + str(source.resolve()) + '?mode=ro', uri=True)
        try:
            text, digest = db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
            teacher = json.loads(text); self.assertEqual(config.digest(teacher), digest)
            self.assertEqual(teacher['choice_depth'], 8)
            self.assertEqual(teacher['teacher'], 'H_v1/untrained/oracle-free/exact-completed-root-depth8/full-root-Q')
            self.assertEqual(teacher['search'], dict(max_depth=8, memory_mb=512, node_budget=0, time_ms=0, tt_mb=16, use_pvs=True))
            from congklak.config import RULES_DIGEST
            self.assertEqual(teacher['rules_digest'], RULES_DIGEST)
            components = set()
            for role, loader, quota in (('train', data.load_train, 512), ('fit_check', data.load_fit, 128)):
                rows, loaded = loader(); self.assertEqual(loaded, manifest)
                self.assertEqual(len(rows), quota)
                self.assertEqual([row['state_key'] for row in rows], [row['state_key'] for row in selection[role]])
                counts[role] = dict(Counter(row['cell'] for row in rows))
                self.assertEqual(set(counts[role].values()), {64 if role == 'train' else 16})
                for row, meta in zip(rows, selection[role]):
                    self.assertNotIn(row['component'], components); components.add(row['component'])
                    self.assertEqual(row['metadata_sha256'], config.digest(meta))
                    for name in ('state', 'state_key', 'family', 'component', 'stratum', 'actor', 'cell'):
                        self.assertEqual(row[name], meta[name])
                    ref = meta['target_refs'][0]
                    self.assertEqual((ref['journal'], ref['table'], ref['depth']), ('deeper-9000-v2', 'labels', 8))
                    original_text, original_digest = db.execute('SELECT body,hash FROM roots WHERE state=?', (meta['state_key'],)).fetchone()
                    original = json.loads(original_text)
                    self.assertEqual(config.digest(original), original_digest)
                    self.assertEqual(original_digest, meta['source_sha256'])
                    self.assertEqual(original['state'], meta['state']); self.assertEqual(original['family'], meta['family'])
                    self.assertEqual(original['rules_digest'], RULES_DIGEST)
                    item_text, item_digest = db.execute('SELECT body,hash FROM labels WHERE state=?', (meta['state_key'],)).fetchone()
                    item = json.loads(item_text)
                    self.assertEqual(config.digest(item), item_digest); self.assertEqual(item_digest, ref['sha256'])
                    self.assertEqual(item['root_sha256'], original_digest)
                    label, report = item['label'], item['report']; actor = meta['actor']
                    self.assertIs(report['completed_iteration'], True)
                    self.assertEqual(label['choice_depth'], 8); self.assertEqual(report['choice_depth'], 8)
                    self.assertEqual(report['evaluator'], 'H_v1'); self.assertEqual(report['score_player'], actor)
                    self.assertEqual(report['position'], row['state'])
                    self.assertEqual(report['input_controller'], actor); self.assertEqual(report['normalized_controller'], actor)
                    self.assertEqual(report['policy'], 'terminal-first/no-oracle/full-root-window-v2')
                    legal = [pit for pit, n in enumerate(row['state']['pits'][actor]) if n]
                    self.assertGreaterEqual(len(legal), 2)
                    self.assertEqual(report['legal_actions'], legal)
                    q = {a['pit']: a['value_player0'] for a in label['action_scores']}
                    self.assertEqual(len(q), len(label['action_scores'])); self.assertEqual(set(q), set(legal))
                    candidate_q = {}
                    for action in report['candidates']:
                        self.assertIs(action['analysis_complete'], True); self.assertEqual(action['actor'], actor)
                        depth = action['depth_result']
                        self.assertEqual((depth['choice_depth'], depth['bound'], depth['scope'], depth['evaluator'], depth['score_player']),
                                         (8, 'exact', 'finite_choice_horizon', 'H_v1', actor))
                        self.assertEqual(depth['value_actor'], depth['value_player0'] * (1 if actor == 0 else -1))
                        self.assertEqual(depth['policy'], report['policy'])
                        self.assertNotIn(action['pit'], candidate_q); candidate_q[action['pit']] = depth['value_player0']
                    self.assertEqual(q, candidate_q)
                    y = (max if actor == 0 else min)(q.values())
                    self.assertEqual(y, label['target_player0']); self.assertEqual(y, report['depth_result']['value_player0'])
                    self.assertEqual(y, row['target_player0'])
                    self.assertEqual(row['target_source']['action_q'], [{'pit': a, 'value_player0': q[a]} for a in legal])
                    feature = meta['feature_ref']; name = feature['path']
                    if name not in feature_files: feature_files[name] = read(ROOT / name)
                    cached = feature_files[name]['groups'][feature['group']][feature['index']]
                    for key in ('state', 'state_key', 'family'): self.assertEqual(cached[key], row[key])
                    raw_features = cached['cached_features']
                    self.assertEqual(row['x'], [fp32(v) for v in raw_features['x']])
                    literal_x, literal_D, literal_r = features_relay.encode(row['state'], 127)
                    self.assertEqual(row['x'], [fp32(v) for v in literal_x])
                    D = row['state']['stores'][0] - row['state']['stores'][1]
                    r = sum(map(sum, row['state']['pits'])) * (1 if actor == 0 else -1)
                    self.assertEqual((row['D'], row['r']), (D, r)); self.assertNotEqual(r, 0)
                    self.assertEqual((literal_D, literal_r), (D, r))
                    z = (y - D) / r; self.assertTrue(math.isfinite(z) and -1 <= z <= 1)
                    self.assertEqual(row['target_z'], z); endpoint_count += abs(z) == 1
                    rows_checked += 1
            self.assertEqual(len(components), 640)
            self.assertFalse(components & {r['component'] for r in selection['arena']})
        finally: db.close()
        EVIDENCE['reserved_target_cache'] = dict(rows=rows_checked, counts=counts, endpoints_retained=endpoint_count,
            independent_raw_root_label_Q_feature_reconstruction=True, arena_labels_read=0, optimizer_steps=0,
            literal_encoder_checks=640,
            cache_manifest_sha256=sha(data.DEFAULT / 'manifest.json'))

    def test_02_real_step4_checkpoint_kills_exact_optimizer_RNG_and_prefix(self):
        from congklak import value_training as trainer, value_training_data as data, value_training_resources as resources
        if FIXTURES is not None:
            prior = read(FIXTURES / 'methods.json')
            nonexecuted_owner_test_changes = []
            for filename, checksum in prior['inputs'].items():
                if filename in ('tests/test_value_training_audit.py', 'tools/audit-value-training.py'):
                    retained = ROOT / 'build/value-training-v1/audit-source-v1' / filename
                    self.assertEqual(sha(retained), checksum)
                elif filename == 'tests/test_value_training_data.py':
                    if sha(ROOT / filename) != checksum:
                        nonexecuted_owner_test_changes.append(dict(path=filename, before=checksum, after=sha(ROOT / filename),
                            executed_by_independent_campaign=False))
                else: self.assertEqual(sha(ROOT / filename), checksum, filename)
            save(WORK / 'retained-source-reuse.json', dict(production_sources_unchanged=True,
                executed_audit_sources_retained=True, nonexecuted_owner_test_changes=nonexecuted_owner_test_changes,
                prior_global_source_attestation=prior['source_unchanged']))
            self.assertEqual([k['boundary'] for k in prior['process_kill_cases'][:2]],
                             ['before_checkpoint_catalog', 'after_checkpoint_catalog'])
            self.assertTrue(all(k['actual_process_kill'] for k in prior['process_kill_cases'][:2]))
            # The retained v1 method executed all producer/prefix checks before
            # its independent replay assertion used integer instead of float 0.
            stderr = (FIXTURES / 'exact-state-execution/stderr.log').read_text()
            self.assertIn('AssertionError: independent_optimizer/param_groups/0/weight_decay', stderr)
            routes = [FIXTURES / 'fresh', FIXTURES / 'before_checkpoint_catalog', FIXTURES / 'after_checkpoint_catalog']
            for path in routes: self.assertEqual(trainer.completed(path)['steps'], 6)
            executions = {rel(path): len((path / 'fixture-executions.jsonl').read_text().splitlines()) for path in routes}
            self.assertEqual(list(executions.values()), [6, 8, 6])
            spec = dict(checkpoints=[str(p / 'checkpoints/step000006.pt') for p in routes], output=str(WORK / 'exact-state-proof.json'))
            save(WORK / 'exact-state-spec.json', spec)
            execute([read(ROOT / 'build/training-runtime.json')['python'], ROOT / 'tools/audit-value-training.py',
                     '--worker', 'compare', '--spec', WORK / 'exact-state-spec.json'], WORK / 'exact-state-execution')
            rollback = WORK / 'stale-catalog'; clone_completed(routes[0], rollback)
            for name in ('certificate.json', 'COMPLETED.json'): (rollback / name).unlink()
            catalog = trainer.unseal(rollback / 'catalog.json'); records = catalog['checkpoints'][:3]
            catalog.update(checkpoints=records, step=4, draws=512, prefix_digest=trainer.digest(records), complete=False)
            trainer.atomic(rollback / 'catalog.json', trainer.envelope(catalog))
            before = snapshot(rollback)
            with self.assertRaises((trainer.TrainingError, FileNotFoundError, ValueError, KeyError)): trainer.status(rollback)
            self.assertEqual(before, snapshot(rollback))
            EVIDENCE['training_checkpoint_recovery'] = dict(retained_producer_source=rel(ROOT / 'build/value-training-v1/audit-source-v1'),
                prior_attempt=rel(FIXTURES / 'methods.json'), real_process_kills=2, new_trainer_optimizer_executions=0,
                trainer_optimizer_executions=20, independent_algebra_optimizer_executions_this_attempt=6,
                total_audit_optimizer_executions_all_attempts=32, execution_counts=executions,
                exact_checkpoint=read(WORK / 'exact-state-proof.json'), stale_valid_catalog_refused=True,
                production_optimizer_steps=0)
            return
        script = ROOT / 'tools/run-value-training.py'
        def arguments(output, ledger, boundary=None):
            command = [sys.executable, script, 'run', '--output', output, '--cache', data.DEFAULT,
                       '--ledger', ledger, '--engineering-fixture']
            if boundary: command += ['--pause-at', boundary, '--pause-marker', output / 'pause.json']
            return command
        fresh = WORK / 'fresh'
        execute(arguments(fresh, WORK / 'fresh-resources'), WORK / 'fresh-execution')
        self.assertEqual(trainer.completed(fresh)['steps'], 6)
        routes = [fresh]; prefix_proofs = []
        for point in ('before_checkpoint_catalog', 'after_checkpoint_catalog'):
            output = WORK / point; ledger = WORK / (point + '-resources')
            kill_worker(arguments(output, ledger, point), WORK / (point + '-kill'), output / 'pause.json', point,
                        worker_script=ROOT / 'tools/train-value-sidecar.py')
            state = trainer.status(output)
            self.assertEqual(state['steps'], 2 if point.startswith('before') else 4)
            committed = snapshot(output)
            immutable = {name: value for name, value in committed.items() if name not in ('catalog.json', 'fixture-executions.jsonl')}
            before_resource = resources.status(ledger)
            self.assertIsNone(before_resource['pending'])
            last = before_resource['events'][-1]
            self.assertIs(last['clean'], False)
            self.assertEqual(last['charged_ns'], before_resource['events'][-2]['lease']['reserved_ns'])
            execute(arguments(output, ledger), WORK / (point + '-resume'))
            current = snapshot(output)
            self.assertTrue(all(current[name] == value for name, value in immutable.items()), 'A committed/orphan prefix artifact was rewritten')
            self.assertEqual(trainer.completed(output)['steps'], 6)
            prefix_proofs.append(dict(boundary=point, committed_cursor_before=state['steps'],
                                      immutable_files_preserved=len(immutable), unclean_full_allowance_charged=True))
            routes.append(output)
        executions = {rel(path): len((path / 'fixture-executions.jsonl').read_text().splitlines()) for path in routes}
        self.assertEqual(list(executions.values()), [6, 8, 6]); self.assertEqual(sum(executions.values()), 20)
        spec = dict(checkpoints=[str(p / 'checkpoints/step000006.pt') for p in routes], output=str(WORK / 'exact-state-proof.json'))
        save(WORK / 'exact-state-spec.json', spec)
        execute([read(ROOT / 'build/training-runtime.json')['python'], ROOT / 'tools/audit-value-training.py',
                 '--worker', 'compare', '--spec', WORK / 'exact-state-spec.json'], WORK / 'exact-state-execution')
        # A stale valid catalog cannot roll back a committed prefix.
        rollback = WORK / 'stale-catalog'; clone_completed(fresh, rollback)
        for name in ('certificate.json', 'COMPLETED.json'): (rollback / name).unlink()
        catalog = trainer.unseal(rollback / 'catalog.json'); records = catalog['checkpoints'][:3]
        catalog.update(checkpoints=records, step=4, draws=512, prefix_digest=trainer.digest(records), complete=False)
        trainer.atomic(rollback / 'catalog.json', trainer.envelope(catalog))
        before = snapshot(rollback)
        with self.assertRaises(Exception): trainer.status(rollback)
        self.assertEqual(before, snapshot(rollback))
        EVIDENCE['training_checkpoint_recovery'] = dict(real_process_kills=2, prefixes=prefix_proofs,
            trainer_optimizer_executions=20, independent_algebra_optimizer_executions=6,
            total_audit_optimizer_executions=26, execution_counts=executions,
            exact_checkpoint=read(WORK / 'exact-state-proof.json'), stale_valid_catalog_refused=True,
            production_optimizer_steps=0)

    def test_03_learned_branch_native_policy_domain_batch_and_FP32(self):
        from congklak import value_sidecar_v2 as side
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        model = side.Model((FIXTURES or WORK) / 'fresh/model.json')
        initial = side.Model(ROOT / 'data/neural-value-pilot-v2/interface-initial/model.json')
        self.assertNotEqual(model.weights, initial.weights)
        self.assertEqual(model.base.envelope, initial.base.envelope)
        self.assertNotEqual(model.id, initial.id)
        corpus = read(ROOT / 'build/ordinary-development-v1/conformance-owner-corpus.json')
        positions = [row['state'] for role in ('selected', 'guards', 'boundaries') for row in corpus[role]]
        self.assertEqual(len(positions), 733)
        maximum = dict(seeds=0., logits=0., score_units=0); outside = comparisons = 0
        scalar = []
        for backend in ('scalar', 'avx2'):
            with CpuBatchEvaluator(side.BASE, backend=backend) as control, side.SidecarEvaluator(model, backend=backend) as candidate:
                results = []
                for start in range(0, len(positions), 32):
                    group = positions[start:start + 32]
                    aa, bb = control.evaluate_many(group), candidate.evaluate_many(group)
                    for position, a, b in zip(group, aa, bb):
                        comparisons += 1
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *b['policy_logits']))
                        expected = literal(model.base, model.weights, position)
                        native_units = candidate.evaluate_units(position)
                        single = candidate.evaluate_many([position])[0]
                        self.assertEqual((single['value_player0'], single['policy_logits']), (b['value_player0'], b['policy_logits']))
                        maximum['seeds'] = max(maximum['seeds'], abs(b['value_player0'] - expected[0]))
                        maximum['logits'] = max(maximum['logits'], max(abs(x - y) for x, y in zip(b['policy_logits'], expected[2])))
                        maximum['score_units'] = max(maximum['score_units'], abs(native_units['value_units'] - expected[1]))
                        for key in ('feature_simulations', 'feature_deposits', 'native_actual_evals'):
                            self.assertEqual(a[key], b[key])
                        if not 25 <= sum(map(sum, position['pits'])) <= 74:
                            outside += 1; self.assertEqual(a['value_player0'], b['value_player0'])
                        results.append((b['value_player0'], tuple(b['policy_logits'])))
                if backend == 'scalar': scalar = results
                else: self.assertEqual(scalar, results)
        self.assertLessEqual(maximum['seeds'], 1e-10); self.assertLessEqual(maximum['logits'], 1e-10)
        self.assertEqual(maximum['score_units'], 0)
        specification = dict(model=str(model.path), positions=positions, output=str(WORK / 'trained-FP32.json'))
        save(WORK / 'trained-FP32-spec.json', specification)
        execute([read(ROOT / 'build/training-runtime.json')['python'], ROOT / 'tools/audit-value-training.py',
                 '--worker', 'conformance', '--spec', WORK / 'trained-FP32-spec.json'], WORK / 'trained-FP32-execution')
        EVIDENCE['learned_native_conformance'] = dict(rows=733, policy_comparisons=comparisons,
            backends=['scalar', 'avx2'], policy_bit_exact=True, outside_gate_comparisons=outside,
            original_tensors_unchanged=8, branch_nonzero=True, single_batch_and_backend_exact=True,
            maximum_errors=maximum, Torch_FP32=read(WORK / 'trained-FP32.json'))

    def test_04_completed_readonly_and_required_evidence_failclosed(self):
        from congklak import value_training as trainer, value_training_data as data
        fresh = (FIXTURES or WORK) / 'fresh'
        specification = dict(training=str(fresh),
            directories=[str(fresh), str(data.DEFAULT)], output=str(WORK / 'closed-proof.json'))
        save(WORK / 'closed-spec.json', specification)
        execute([sys.executable, ROOT / 'tools/audit-value-training.py', '--worker', 'closed',
                 '--spec', WORK / 'closed-spec.json'], WORK / 'closed-execution')
        refused = []
        for name in ('catalog.json', 'base-cache.json', 'identity.json', 'certificate.json',
                     'checkpoints/step000000.pt', 'checkpoints/step000006.pt', 'checkpoints/step000002.json'):
            clone = WORK / ('required-' + name.replace('/', '-').replace('.', '-'))
            clone_completed(fresh, clone)
            (clone / name).unlink()
            before = snapshot(clone)
            with mock.patch.object(trainer, 'Writer', side_effect=forbidden) as writer:
                with self.assertRaises((trainer.TrainingError, ValueError, FileNotFoundError, KeyError)): trainer.run(clone, fixture=True)
                with self.assertRaises((trainer.TrainingError, ValueError, FileNotFoundError, KeyError)): trainer.completed(clone)
                self.assertEqual(writer.call_count, 0)
            self.assertEqual(before, snapshot(clone)); refused.append(name)
        cache_refusals = []
        for name in ('certificate.json', 'train.json', 'fit-check.json', 'identity.json'):
            clone = WORK / ('cache-required-' + name.replace('.', '-'))
            clone_cache(data.DEFAULT, clone)
            target = clone / name
            if name.endswith('.json') and name in ('train.json', 'fit-check.json'):
                with target.open('ab') as stream: stream.write(b' ')
            else: target.unlink()
            before = snapshot(clone)
            with mock.patch.object(data, 'Writer', side_effect=forbidden) as writer, mock.patch.object(data, '_rows', side_effect=forbidden) as rows:
                with self.assertRaises((ValueError, FileNotFoundError, KeyError)): data.prepare(clone)
                with self.assertRaises((ValueError, FileNotFoundError, KeyError)): data.admit(clone)
                self.assertEqual(writer.call_count, 0); self.assertEqual(rows.call_count, 0)
            self.assertEqual(before, snapshot(clone)); cache_refusals.append(name)
        # Removing the whole closing marker set must not reset the committed run.
        clone = WORK / 'catalog-missing-all-closers'; clone_completed(fresh, clone)
        for name in ('catalog.json', 'certificate.json', 'COMPLETED.json'): (clone / name).unlink()
        before = snapshot(clone)
        with self.assertRaises((trainer.TrainingError, ValueError, FileNotFoundError, KeyError)): trainer.status(clone)
        self.assertEqual(before, snapshot(clone))
        markerless = WORK / 'missing-derived-completion-marker'; clone_completed(fresh, markerless)
        (markerless / 'COMPLETED.json').unlink(); before = snapshot(markerless)
        with mock.patch.object(trainer, 'Writer', forbidden):
            self.assertEqual(trainer.completed(markerless)['steps'], 6)
            self.assertEqual(trainer.run(markerless, fixture=True)['steps'], 6)
        self.assertEqual(before, snapshot(markerless))
        EVIDENCE['completed_failclosed'] = dict(guard=read(WORK / 'closed-proof.json'),
            missing_training_evidence_refused=refused, missing_corrupt_cache_evidence_refused=cache_refusals,
            missing_catalog_all_closers_refused=True, missing_derived_marker_readonly=True, no_recomputation=True)

    def test_05_actual_watchdog_deadline_containment_and_durable_budget(self):
        from congklak import value_training_resources as resources, value_training_data as data
        protocol = dict(resources.PROTOCOL, lease_ns=4_000_000_000, cumulative_ns=4_000_000_000,
                        startup_ns=2_000_000_000, clean_stop_margin_ns=500_000_000)
        ledger = WORK / 'watchdog-resources'
        lease_path = resources.reserve(ledger, WORK / 'watchdog-nooptimizer', data.DEFAULT, True, protocol)
        lease = read(lease_path)
        entry = WORK / 'watchdog-nooptimizer.py'
        entry.write_text("import json,os,pathlib,sys,time\n"
            "sys.path.insert(0," + repr(str(ROOT)) + ")\n"
            "from congklak import value_training_resources as r\n"
            "lease=r.bind_worker(sys.argv[1])\n"
            "assert 'torch' not in sys.modules\n"
            "r._atomic(pathlib.Path(sys.argv[2]),dict(pid=os.getpid(),point='contained-nooptimizer',Torch_imports=0))\n"
            "while True:time.sleep(.01)\n", encoding='utf-8')
        marker = WORK / 'watchdog-nooptimizer-marker.json'
        started = time.monotonic()
        with (WORK / 'watchdog.log').open('wb') as log:
            watchdog = subprocess.Popen([sys.executable, str(ROOT / 'tools/run-value-training.py'), 'watchdog', '--lease', str(lease_path)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            worker = subprocess.Popen([sys.executable, str(entry), str(lease_path), str(marker)], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT)
            launcher = resources.process_identity(worker.pid)
            resources._atomic(lease['launcher_file'], dict(token=lease['token'], process=launcher,
                supervisor=lease['supervisor_identity'], command=[sys.executable, str(entry), str(lease_path), str(marker)]), True)
            try:
                while not marker.exists():
                    self.assertIsNone(worker.poll(), 'Watchdog fixture exited before containment')
                    self.assertLess(time.monotonic() - started, 10)
                    time.sleep(.01)
                actual = resources.process_identity(read(marker)['pid']); permit = read(lease['permit_file'])
                self.assertEqual(permit['worker'], actual); self.assertIs(permit['kill_on_job_close'], True)
                self.assertNotEqual(permit['watchdog']['pid'], os.getpid())
                self.assertNotEqual(worker.wait(timeout=15), 0)
                self.assertEqual(watchdog.wait(timeout=15), 0)
            finally:
                if worker.poll() is None: worker.kill(); worker.wait()
                if watchdog.poll() is None: watchdog.kill(); watchdog.wait()
        report = resources.status(ledger); finish = report['events'][-1]
        self.assertEqual(finish['reason'], 'deadline'); self.assertIs(finish['clean'], False)
        self.assertEqual(report['charged_ns'], 4_000_000_000); self.assertEqual(report['remaining_ns'], 0)
        self.assertGreater(finish['peak_private_commit_bytes'], 0)
        with self.assertRaises(Exception): resources.reserve(ledger, WORK / 'watchdog-restart', data.DEFAULT, True, protocol)
        corruptions = []
        for mutation in ('missing_identity', 'missing_head', 'head_rollback', 'hold'):
            clone = WORK / ('ledger-' + mutation); shutil.copytree(ledger, clone)
            if mutation == 'missing_identity': (clone / 'identity.json').unlink()
            elif mutation == 'missing_head': (clone / 'head.json').unlink()
            elif mutation == 'head_rollback': resources._atomic(clone / 'head.json', dict(schema=resources.SCHEMA + '-head', count=0, last_sha256=None))
            else: resources._atomic(clone / 'hold.json', dict(reason='unknown overrun engineering fixture'))
            before = snapshot(clone)
            with self.assertRaises(Exception): resources.reserve(clone, WORK / 'bad-budget', data.DEFAULT, True, protocol)
            self.assertEqual(before, snapshot(clone)); corruptions.append(mutation)
        record = dict(boundary='independent-watchdog-deadline', actual_process_kill=True,
            worker_identity=actual, permit=permit, finish=finish, no_optimizer=True, Torch_imports=0,
            elapsed_seconds=time.monotonic() - started)
        KILLS.append(record); save(WORK / 'watchdog-proof.json', record)
        EVIDENCE['resources'] = dict(actual_watchdog_deadline_kills=1, job_kill_on_close=True,
            private_commit_measured=True, exhausted_budget_refused=True, unclean_full_allowance_charged=True,
            ledger_reset_corruption_refused=corruptions, optimizer_steps=0,
            production_protocol=resources.PROTOCOL)

    def test_06_accepted_history_zero_production_and_dashboard(self):
        from congklak import value_training_progress as progress
        baseline = read(FOLDER / 'baseline.json')
        pins = baseline['inputs']
        for name, checksum in pins.items(): self.assertEqual(sha(ROOT / name), checksum, name)
        self.assertEqual(len(pins), 7848)
        self.assertFalse((ROOT / 'data/value-training-v1/training').exists())
        before = snapshot(FOLDER)
        with mock.patch.object(progress, 'LOADED_SHA', 'bad-source'):
            observed = progress.snapshot()
            self.assertFalse(observed['implementation_ready']); self.assertEqual(observed['current_work']['state'], 'hold')
        self.assertEqual(before, snapshot(FOLDER))
        observed = progress.snapshot()
        self.assertEqual(observed['production_counters'], dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0))
        self.assertEqual(observed['current_cards'], 10)
        self.assertEqual(len(observed['tools']), 20)
        EVIDENCE['history_and_dashboard'] = dict(prior_accepted_files_preserved=7848,
            production_steps=0, production_draws=0, new_games=0, new_teacher_labels=0,
            readonly_dashboard=True, stale_source_HOLD=True, cards=10, prior_cards=10,
            browser_or_HTTP_inspection=False)
