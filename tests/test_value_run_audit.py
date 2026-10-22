"""Independent audit methods. No method performs production optimization."""
from __future__ import annotations
from contextlib import ExitStack
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / 'data/value-training-v1/training'
WORK = None
EVIDENCE = {}
ATTEMPTS = []

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Audit evidence cannot be overwritten: ' + str(path)
    path.write_text(canonical(value) + '\n', encoding='utf-8', newline='\n')
def files(path):
    return sorted(p for p in Path(path).rglob('*') if p.is_file() and p.suffix != '.pyc' and not p.name.endswith('-shm'))
def snapshot(path):
    return {rel(p): [sha(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files(path)}
def forbidden(*args, **kwargs): raise AssertionError('Completed route performed forbidden work')
def exact(torch, left, right, name='root'):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right), name
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), name
        for key in left: exact(torch, left[key], right[key], name + '/' + str(key))
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), name
        for i, (a, b) in enumerate(zip(left, right)): exact(torch, a, b, name + '/' + str(i))
    else: assert type(left) is type(right) and left == right, name

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import on a completed reader')
        return None

def execute(argv, output, timeout=180):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with (output / 'stdout.log').open('wb') as out, (output / 'stderr.log').open('wb') as err:
        result = subprocess.run([str(v) for v in argv], cwd=ROOT, stdout=out, stderr=err, timeout=timeout)
    evidence = dict(argv=[str(v) for v in argv], returncode=result.returncode,
                    elapsed_seconds=time.monotonic() - started, logs=rel(output))
    ATTEMPTS.append(evidence); save(output / 'execution.json', evidence)
    assert result.returncode == 0, evidence
    return evidence

def recovery_helpers():
    path = ROOT / 'tests/test_value_training_audit.py'
    spec = importlib.util.spec_from_file_location('accepted_independent_process_helpers', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def committed_snapshot(path, kind):
    path = Path(path)
    if kind == 'cases':
        head = read(path / 'catalog.json')
        paths = [path / 'identity.json', path / 'tasks.json', path / 'initialized.json']
        paths += [path / 'records' / row['name'] for row in head['records']]
    else:
        head = read(path / 'head.json')
        paths = [path / 'identity.json', path / 'initialized.json']
        paths += [path / row['name'] for section in ('decisions', 'games') for row in head[section]]
    return {rel(p): [sha(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths}

def same_existing(snapshot_before):
    return all(record == [sha(ROOT / name), (ROOT / name).stat().st_size, (ROOT / name).stat().st_mtime_ns]
               for name, record in snapshot_before.items())

def case_semantics(path):
    from congklak import value_run_cases as cases
    rows = [read(Path(path) / 'records' / item['name']) for item in read(Path(path) / 'catalog.json')['records']]
    return [dict(task=row['task'], score=row['score'], physical=cases.physical_view(row['answer']), cpu=row['cpu_verified']) for row in rows]

def arena_semantics(path):
    from congklak import value_run_cases as cases
    path = Path(path); head = read(path / 'head.json')
    decisions = []
    for item in head['decisions']:
        row = read(path / item['name'])
        decisions.append({**{key: value for key, value in row.items() if key != 'answer'},
                          'physical': cases.physical_view(row['answer'])})
    games = [{key: value for key, value in read(path / item['name']).items() if key != 'decision_sha256'} for item in head['games']]
    return dict(decisions=decisions, games=games)

def training_worker(specification):
    import torch
    from congklak import value_training as trainer
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(specification['training'])
    result = trainer.completed(path)
    identity = trainer.unseal(path / 'identity.json')
    catalog = trainer.unseal(path / 'catalog.json')
    assert result['steps'] == 400 and result['draws'] == 51200 and result['training_eligible'] is True
    assert identity['engineering_fixture'] is False and identity['settings'] == trainer.settings(False)
    assert [row['step'] for row in catalog['checkpoints']] == list(range(0, 401, 20))
    initial = read(ROOT / 'data/neural-value-pilot-v2/interface-initial/model.json')['model']
    expected_frozen = {k: torch.tensor(v, dtype=torch.float32) for k, v in initial['base_export']['model']['weights'].items()}
    expected_python = random.Random(20261016).getstate()
    global_generator = torch.Generator(device='cpu'); global_generator.manual_seed(20261016)
    sampler = torch.Generator(device='cpu'); sampler.manual_seed(20261016)
    prefix = []; states = []; records = []
    for record in catalog['checkpoints']:
        state = torch.load(path / 'checkpoints' / record['name'], map_location='cpu', weights_only=True)
        step = record['step']
        while len(prefix) < step * 128:
            prefix.extend(torch.randint(512, (128,), generator=sampler, device='cpu').tolist())
        assert state['step'] == step and state['draws'] == step * 128
        assert state['samples'] == prefix
        assert state['sample_prefix_digest'] == hashlib.sha256(canonical(prefix).encode()).hexdigest()
        assert torch.equal(state['sampler_rng'], sampler.get_state())
        exact(torch, state['frozen'], expected_frozen, 'all_original_tensors')
        exact(torch, state['python_rng'], expected_python, 'python_rng')
        assert torch.equal(state['torch_rng'], global_generator.get_state())
        assert len(state['metrics']) == step
        assert all(math.isfinite(v) for row in state['metrics'] for v in row.values() if isinstance(v, (int, float)))
        assert len(state['optimizer']['state']) == (4 if step else 0)
        for name, tensor in state['branch'].items():
            assert tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
            if step:
                moment = state['optimizer']['state'][list(state['branch']).index(name)]
                assert set(moment) == {'step', 'exp_avg', 'exp_avg_sq'} and float(moment['step']) == step
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert tensor.shape == moment[key].shape and bool(torch.isfinite(moment[key]).all())
        assert state['optimizer']['param_groups'] == states[0]['optimizer']['param_groups'] if states else True
        if states:
            assert states[-1]['samples'] == state['samples'][:states[-1]['draws']]
            assert states[-1]['metrics'] == state['metrics'][:states[-1]['step']]
        states.append(state)
        records.append(dict(step=step, draws=step * 128, samples_sha256=state['sample_prefix_digest'], checkpoint_sha256=sha(path / 'checkpoints' / record['name'])))
    final = states[-1]
    export = read(path / 'model.json')
    assert export['model']['branch_weights'] == {k: v.tolist() for k, v in final['branch'].items()}
    assert export['model']['base_export'] == initial['base_export']
    assert export['model']['branch_weights'] != initial['branch_weights']
    assert export['model']['provenance']['selection'] == 'final-fixed-step-only'
    assert len(final['frozen']) == 8 and sum(v.numel() for v in final['frozen'].values()) == 12872
    assert sum(v.numel() for v in final['branch'].values()) == 2065
    assert not torch.cuda.is_initialized()
    save(specification['output'], dict(passed=True, checkpoints=records, initialization_plus_scheduled=21,
        production_steps=400, production_draws=51200, independently_executed_optimizer_steps=0,
        exact_sampler_replay=True, exact_python_and_torch_rng=True, all_frozen_tensors_exact=True,
        policy_unchanged=True, final_export_matches_step400=True, CUDA_initialized=False,
        torch=str(torch.__version__), original_parameters=12872, trained_parameters=2065))

def algebra(model, weights, position):
    from congklak import features_relay
    from congklak.neural_relay import rounded
    x, d, r = features_relay.encode(position, 127)
    if not r: return float(d), d * 256, [0.] * 7, 0.
    hidden = x
    for prefix in ('layers.0', 'layers.2'):
        hidden = [max(0., sum(a * b for a, b in zip(row, hidden)) + bias)
                  for row, bias in zip(model.weights[prefix + '.weight'], model.weights[prefix + '.bias'])]
    value = sum(a * b for a, b in zip(model.weights['layers.4.weight'][0], hidden)) + model.weights['layers.4.bias'][0]
    logits = [sum(a * b for a, b in zip(row, hidden)) + bias for row, bias in zip(model.weights['policy.weight'], model.weights['policy.bias'])]
    if weights is not None and 25 <= sum(map(sum, position['pits'])) <= 74:
        units = [max(0., sum(a * b for a, b in zip(row, x)) + bias) for row, bias in zip(weights['A'], weights['b'])]
        value += sum(a * b for a, b in zip(weights['v'][0], units)) + weights['c'][0]
    normalized = math.tanh(value)
    residual = r * normalized
    return d + residual, d * 256 + rounded(residual * 256), logits, normalized

def metrics(rows):
    groups = {'all': rows}
    groups.update({name: [r for r in rows if r['cell'] == name] for name in sorted({r['cell'] for r in rows})})
    groups.update({'actor' + str(actor): [r for r in rows if r['actor'] == actor] for actor in (0, 1)})
    groups.update({name: [r for r in rows if name in r['cell']] for name in ('ordinary', 'capture', 'bonus')})
    result = {}
    for name, group in groups.items():
        result[name] = {}
        for profile in ('control', 'candidate'):
            errors = [r[profile + '_z'] - r['target_z'] for r in group]
            seed_errors = [r[profile + '_seed'] - r['target_player0'] for r in group]
            result[name][profile] = dict(n=len(group), normalized_huber=sum(.5 * e * e if abs(e) <= .1 else .1 * (abs(e) - .05) for e in errors) / len(group),
                normalized_mae=sum(map(abs, errors)) / len(group), seed_mae=sum(map(abs, seed_errors)) / len(group),
                seed_rmse=math.sqrt(sum(e * e for e in seed_errors) / len(group)), seed_bias=sum(seed_errors) / len(group))
    return result

def fp32_worker(specification):
    import torch
    from congklak import features_relay, value_sidecar_v2 as side
    from congklak.neural_relay import rounded
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = side.Model(specification['model'])
    frozen = {k: torch.tensor(v, dtype=torch.float32) for k, v in model.base.weights.items()}
    branch = {k: torch.tensor(v, dtype=torch.float32) for k, v in model.weights.items()}
    maximum = dict(seed=0., logits=0., score_units=0); count = 0
    with torch.no_grad():
        for position in specification['positions']:
            x, d, r = features_relay.encode(position, 127)
            if not r: continue
            h = torch.tensor(x, dtype=torch.float32); vector = h
            for prefix in ('layers.0', 'layers.2'):
                h = torch.relu(frozen[prefix + '.weight'] @ h + frozen[prefix + '.bias'])
            u = (frozen['layers.4.weight'] @ h + frozen['layers.4.bias'])[0]
            policy = (frozen['policy.weight'] @ h + frozen['policy.bias']).tolist()
            if 25 <= sum(map(sum, position['pits'])) <= 74:
                correction = branch['v'] @ torch.relu(branch['A'] @ vector + branch['b']) + branch['c']
                u += correction[0]
            residual = float(r * torch.tanh(u)); expected = algebra(model.base, model.weights, position)
            maximum['seed'] = max(maximum['seed'], abs(d + residual - expected[0]))
            maximum['logits'] = max(maximum['logits'], max(abs(a - b) for a, b in zip(policy, expected[2])))
            maximum['score_units'] = max(maximum['score_units'], abs(d * 256 + rounded(residual * 256) - expected[1])); count += 1
    assert maximum['seed'] <= .001 and maximum['logits'] <= 1e-5 and maximum['score_units'] <= 1, maximum
    assert not torch.cuda.is_initialized()
    save(specification['output'], dict(passed=True, nonterminal=count, maximum=maximum, CUDA_initialized=False, independently_executed_optimizer_steps=0))

def closed_worker(specification):
    from congklak import value_training as trainer, value_training_data as data, value_training_resources as resources
    sys.meta_path.insert(0, NoTorch())
    directory = Path(specification['training']); before = snapshot(directory)
    with ExitStack() as stack:
        for obj, name in ((trainer, 'Writer'), (trainer, '_objects'), (trainer, '_runtime'), (trainer, '_load_checkpoint'),
                          (trainer, '_finish'), (resources, 'bind_worker'), (resources, 'assert_worker_lease'),
                          (data, '_load'), (data, 'Writer')):
            stack.enter_context(mock.patch.object(obj, name, forbidden))
        first = trainer.completed(directory)
        assert trainer.status(directory) == trainer.run(directory) == first
        original = sys.argv
        try:
            for command in ('status', 'verify', 'run'):
                sys.argv = ['train-value-sidecar.py', command, '--output', str(directory)]
                assert trainer.main() == 0
        finally: sys.argv = original
    assert snapshot(directory) == before and 'torch' not in sys.modules
    evaluations = None
    if specification.get('evaluations'):
        from congklak import value_run_cases_v2 as cases, value_run_arena as arena, value_run_resources as arena_resources
        from congklak import value_run_decision as decision, value_run_progress_v3 as progress
        from congklak.native import Native
        from congklak.neural_relay import NeuralNative
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from congklak.value_sidecar_v2 import SidecarEvaluator
        directories = [cases.DEFAULT, arena.DEFAULT]
        before_evaluation = {str(p): snapshot(p) for p in directories}
        with ExitStack() as stack:
            for obj, name in ((cases, '_writer'), (cases, 'open_engines'), (cases, 'timed_decision'), (cases, 'fixed_decision'),
                              (arena, 'Writer'), (arena, 'prepare'), (arena_resources, 'launch'),
                              (Native, '__init__'), (NeuralNative, '__init__'), (CpuBatchEvaluator, '__init__'), (SidecarEvaluator, '__init__')):
                stack.enter_context(mock.patch.object(obj, name, forbidden))
            a = cases.completed(); selfsame = cases.status() == cases.run() == a
            assert selfsame
            b = arena.verify(); assert arena.status() == arena.run() == b
            cases.fit(ROOT / 'build/value-run-v1/cases-fit-v2.json')
            cases.conformance(ROOT / 'build/value-run-v1/cases-conformance-v2.json')
            final = decision.run(); assert final['production_steps'] == 400 and final['promotion'] is False
            observer = progress.snapshot(); assert observer['production']['steps'] == 400
            original = sys.argv
            try:
                with io.StringIO() as sink, __import__('contextlib').redirect_stdout(sink):
                    for operation in ('verify', 'status', 'run'):
                        sys.argv = ['evaluate-value-run-v2.py', operation]
                        assert cases.main() == 0
                    spec = importlib.util.spec_from_file_location('closed_arena_cli', ROOT / 'tools/run-value-arena.py')
                    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
                    for operation in ('verify', 'status', 'run'):
                        sys.argv = ['run-value-arena.py', operation]
                        assert cli.main() == 0
            finally: sys.argv = original
        assert before_evaluation == {str(p): snapshot(p) for p in directories}
        assert 'torch' not in sys.modules
        evaluations = dict(APIs=11, CLI_operations=6, Torch_imports=0, writers=0, forwards=0,
            searches=0, bytes_sizes_mtimes_preserved=True, exact_decision_report=True,
            dashboard_current_cards=observer['current_cards'], dashboard_source_current=observer['observer_source_current'])
    save(specification['output'], dict(passed=True, APIs=3, CLI_operations=3, Torch_imports=0,
        writers=0, optimizer_steps=0, cache_loads=0, checkpoint_loads=0, bytes_sizes_mtimes_preserved=True, evaluations=evaluations))

def pinned_sources():
    # These methods may execute while unrelated evaluation owners are still
    # drafting their modules. Their frozen evidence is verified by later
    # methods; never attest that an unexecuted draft was globally unchanged.
    names = ('tests/test_value_run_audit.py', 'tools/audit-value-run.py', 'congklak/value_training.py',
             'congklak/value_training_data.py', 'congklak/value_training_resources.py',
             'congklak/value_sidecar_v2.py', 'congklak/features_relay.py', 'build/training-runtime.json',
             'data/value-training-v1/training/certificate.json', 'data/value-training-v1/training/model.json')
    return {name: sha(ROOT / name) for name in names if (ROOT / name).is_file()}

def run(output, methods=None):
    global WORK
    WORK = Path(output).resolve(); WORK.mkdir(parents=True, exist_ok=False)
    before = pinned_sources(); save(WORK / 'sources-before.json', before)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValueRunAudit)
    if methods: suite = unittest.TestSuite(test for test in suite if test._testMethodName in methods)
    with (WORK / 'methods.log').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    unchanged = before == pinned_sources()
    report = dict(schema='value-run-independent-methods-v1', passed=result.wasSuccessful() and unchanged,
        independent_of_implementation=True, tests_run=result.testsRun, failed=len(result.failures), errors=len(result.errors),
        source_unchanged=unchanged, inputs=before, evidence=EVIDENCE, attempts=ATTEMPTS,
        failures=[dict(method=str(test), traceback=trace) for test, trace in result.failures + result.errors])
    save(WORK / 'methods.json', report)
    return report

class ValueRunAudit(unittest.TestCase):
    def test_01_actual400_checkpoint_sampler_frozen_and_export(self):
        spec = dict(training=str(TRAINING), output=str(WORK / 'training-contents.json'))
        save(WORK / 'training-spec.json', spec)
        execute([read(ROOT / 'build/training-runtime.json')['python'], ROOT / 'tools/audit-value-run.py', '--worker', 'training', '--spec', WORK / 'training-spec.json'], WORK / 'training-execution')
        EVIDENCE['production_checkpoint_contents'] = read(WORK / 'training-contents.json')
        proof = read(ROOT / 'build/value-run-v1/stop020-proof.json')
        self.assertEqual(proof['status']['steps'], 20)
        checked = []
        for name, record in proof['immutable_prefix'].items():
            path = TRAINING / name
            self.assertEqual(dict(sha256=sha(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns), record)
            checked.append(name)
        from congklak import value_training_resources as resources
        ledger = resources.status(ROOT / 'data/value-training-v1/resources')
        self.assertIsNone(ledger['pending'])
        self.assertEqual(len(ledger['events']), 4)
        self.assertEqual([e['kind'] for e in ledger['events']], ['reserve', 'finish', 'reserve', 'finish'])
        charges = []; peaks = []
        for reserve, finish in zip(ledger['events'][::2], ledger['events'][1::2]):
            lease = reserve['lease']; directory = Path(lease['lease_file']).parent
            worker = read(directory / 'worker.json'); permit = read(directory / 'permit.json')
            done = read(directory / 'completion.json'); launcher = read(directory / 'launcher.json')
            self.assertTrue(finish['clean']); self.assertFalse(finish['unknown_overrun'])
            self.assertEqual(finish['reason'], 'durable completion')
            self.assertEqual(finish['actual_worker'], worker['process'])
            self.assertEqual(permit['worker'], worker['process']); self.assertTrue(permit['kill_on_job_close'])
            self.assertEqual(done['worker'], worker['process'])
            self.assertTrue(worker['process'] == launcher['process'] or worker['process']['parent_pid'] == launcher['process']['pid'])
            self.assertLessEqual(finish['peak_private_commit_bytes'], 4 * 1024 ** 3)
            self.assertEqual(lease['reserved_ns'], 120_000_000_000)
            self.assertLessEqual(done['completed_monotonic_ns'], lease['deadline_monotonic_ns'])
            charges.append(finish['charged_ns']); peaks.append(finish['peak_private_commit_bytes'])
        self.assertEqual(sum(charges), ledger['charged_ns'])
        self.assertLessEqual(ledger['charged_ns'], 1_200_000_000_000)
        EVIDENCE['production_recovery_and_resources'] = dict(actual_cooperative_stop_at_step=20,
            remaining_optimizer_updates=380, exact_preserved_bytes_sizes_mtimes=checked,
            clean_invocations=2, charged_ns=ledger['charged_ns'], invocation_charges_ns=charges,
            actual_peak_private_commit_bytes=max(peaks), actual_worker_containment=True,
            earlier_actual_hardkill_training_audit=rel(ROOT / 'build/value-training-v1/audit.json'),
            earlier_actual_hardkill_audit_sha256=sha(ROOT / 'build/value-training-v1/audit.json'))

    def test_02_independent_fit_and_learned_native_FP32(self):
        from congklak import value_sidecar_v2 as side, value_training_data as data
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        model = side.Model(TRAINING / 'model.json')
        training = data.load_train()[0]; fit = data.load_fit()[0]
        self.assertEqual((len(training), len(fit)), (512, 128))
        self.assertTrue({r['component'] for r in training}.isdisjoint(r['component'] for r in fit))
        self.assertEqual(len({r['component'] for r in training + fit}), 640)
        rows = []
        for row in fit:
            a = algebra(model.base, None, row['state']); b = algebra(model.base, model.weights, row['state'])
            rows.append({**{key: row[key] for key in ('state_key', 'cell', 'actor', 'target_z', 'target_player0')},
                'control_z': a[3], 'candidate_z': b[3], 'control_seed': a[0], 'candidate_seed': b[0]})
        measured = metrics(rows)
        gates = dict(huber=measured['all']['candidate']['normalized_huber'] < measured['all']['control']['normalized_huber'],
            overall_seed_mae=measured['all']['candidate']['seed_mae'] <= measured['all']['control']['seed_mae'],
            capture_seed_mae=measured['capture']['candidate']['seed_mae'] <= measured['capture']['control']['seed_mae'])
        save(WORK / 'independent-fit.json', dict(rows=rows, metrics=measured, gates=gates, fit_passed=all(gates.values()), diagnostic_is_unseen=False))
        corpus = read(ROOT / 'build/ordinary-development-v1/conformance-owner-corpus.json')
        positions = [row['state'] for role in ('selected', 'guards', 'boundaries') for row in corpus[role]]
        maximum = dict(seeds=0., logits=0., score_units=0); comparisons = outside = 0; scalar = []
        for backend in ('scalar', 'avx2'):
            route = []
            with CpuBatchEvaluator(side.BASE, backend=backend) as control, side.SidecarEvaluator(model, backend=backend) as candidate:
                for start in range(0, len(positions), 32):
                    group = positions[start:start + 32]
                    aa, bb = control.evaluate_many(group), candidate.evaluate_many(group)
                    for position, a, b in zip(group, aa, bb):
                        comparisons += 1
                        self.assertEqual(struct.pack('<7d', *a['policy_logits']), struct.pack('<7d', *b['policy_logits']))
                        expected = algebra(model.base, model.weights, position)
                        maximum['seeds'] = max(maximum['seeds'], abs(b['value_player0'] - expected[0]))
                        maximum['logits'] = max(maximum['logits'], max(abs(x - y) for x, y in zip(b['policy_logits'], expected[2])))
                        units = candidate.evaluate_units(position)['value_units']
                        maximum['score_units'] = max(maximum['score_units'], abs(units - expected[1]))
                        for key in ('feature_simulations', 'feature_deposits', 'native_actual_evals'):
                            self.assertEqual(a[key], b[key])
                        if not 25 <= sum(map(sum, position['pits'])) <= 74:
                            outside += 1; self.assertEqual(a['value_player0'], b['value_player0'])
                        route.append((b['value_player0'], b['policy_logits']))
            if backend == 'scalar': scalar = route
            else: self.assertEqual(scalar, route)
        self.assertLessEqual(maximum['seeds'], 1e-10); self.assertLessEqual(maximum['logits'], 1e-10)
        self.assertEqual(maximum['score_units'], 0)
        spec = dict(model=str(model.path), positions=positions, output=str(WORK / 'FP32.json'))
        save(WORK / 'FP32-spec.json', spec)
        execute([read(ROOT / 'build/training-runtime.json')['python'], ROOT / 'tools/audit-value-run.py', '--worker', 'fp32', '--spec', WORK / 'FP32-spec.json'], WORK / 'FP32-execution')
        EVIDENCE['fit_and_conformance'] = dict(independent_fit=rel(WORK / 'independent-fit.json'), fit_gates=gates,
            fit_passed=all(gates.values()), row_occurrences=len(positions), policy_comparisons=comparisons,
            outside_gate_comparisons=outside, policy_bit_exact=True, backends=['scalar', 'avx2'], maximum=maximum, FP32=read(WORK / 'FP32.json'))

    def test_03_all650_search_cases_and_regret(self):
        from congklak import value_run_cases as cases
        path = ROOT / 'build/value-run-v1/cases'; head = read(path / 'catalog.json')
        rows = [read(path / 'records' / r['name']) for r in head['records']]
        self.assertEqual(len(rows), 650)
        groups = defaultdict(list); fixed = defaultdict(dict); combinations = set(); fallbacks = Counter()
        for index, row in enumerate(rows):
            self.assertEqual(row['index'], index)
            task = row['task']; root = task['root']; answer = row['answer']['decision']
            self.assertEqual(row['cpu_verified'], dict(group=0, logical=2, mask=4))
            token = (task['mode'], root['state_key'], task['engine'], task['ms'])
            self.assertNotIn(token, combinations); combinations.add(token)
            self.assertEqual(answer['requested_ms'], task['ms'])
            self.assertGreater(answer['wall_ms'], 0)
            self.assertAlmostEqual(answer['overshoot_ms'], max(0., answer['wall_ms'] - task['ms']) if task['ms'] else 0., places=9)
            if not answer['usable_decision']: fallbacks[task['engine']] += 1
            ref = root.get('reference'); chosen = answer['chosen']; regret = optimal = None
            if ref is not None and chosen is not None:
                q = {a['pit']: a['value_player0'] for a in ref['action_scores']}
                target = (min if root['state']['controller'] else max)(q.values())
                self.assertEqual(target, ref['target_player0'])
                self.assertEqual(set(ref['best_moves']), {a for a, v in q.items() if v == target})
                regret = (1 if root['state']['controller'] == 0 else -1) * (target - q[chosen])
                optimal = q[chosen] == target
                self.assertGreaterEqual(regret, 0)
            self.assertEqual(row['score'], dict(regret=regret, optimal=optimal))
            groups[(task['mode'], root['cohort'], task['ms'], task['engine'])].append(dict(regret=regret, optimal=optimal, answer=answer))
            if task['mode'] == 'fixed128':
                self.assertEqual(answer['report']['requested_simulations'], 128)
                self.assertEqual(sum(answer['report']['visits']), answer['completed_simulations'])
                fixed[root['state_key']][task['engine']] = row
        timed = sum(r['task']['mode'] == 'timed' for r in rows)
        self.assertEqual(timed, 512); self.assertEqual(len(fixed), 69)
        metrics = {}; gates = {}
        for mode, clock in (('timed', 25), ('timed', 100), ('fixed128', 0)):
            values = {}
            for name in ('control', 'candidate'):
                group = groups[(mode, 'fit_H8', clock, name)]
                self.assertEqual(len(group), 32)
                scored = [r['regret'] for r in group if r['regret'] is not None]
                values[name] = sum(scored) / len(scored) if scored else None
                values[name + '_scored'] = len(scored)
            gates[mode + '/' + str(clock)] = values['control_scored'] == values['candidate_scored'] == 32 and values['candidate'] <= values['control']
            metrics[mode + '/' + str(clock)] = values
        guards = {}
        for clock in (25, 100):
            group = groups[('timed', 'exact_guard', clock, 'candidate')]
            self.assertEqual(len(group), 32)
            guards[str(clock)] = dict(optimal=sum(bool(r['optimal']) for r in group), fallbacks=sum(not r['answer']['usable_decision'] for r in group))
            guards[str(clock)]['passed'] = guards[str(clock)]['optimal'] == 32 and guards[str(clock)]['fallbacks'] == 0
        low = 0
        for pair in fixed.values():
            self.assertEqual(set(pair), {'control', 'candidate'})
            row = pair['control']
            if sum(map(sum, row['task']['root']['state']['pits'])) <= 24:
                self.assertEqual(cases.physical_view(row['answer']), cases.physical_view(pair['candidate']['answer']))
                low += 1
        report = read(path / 'report.json')
        self.assertEqual(report['timed_measurements'], 512); self.assertEqual(report['fixed_searches'], 138)
        self.assertEqual(report['fixed_requested_backups'], 17664)
        self.assertEqual(report['low_active']['roots'], low); self.assertTrue(report['low_active']['passed'])
        for name, passed in gates.items(): self.assertEqual(report['h8_regret_gates'][name]['passed'], passed)
        for clock, guard in guards.items(): self.assertEqual(report['exact_guard'][clock]['passed'], guard['passed'])
        # Owner fit must independently agree with literal audit results, including
        # negative gates; no adjustment or checkpoint selection follows them.
        independent = read(ROOT / 'build/value-run-v1/audit-training-v1/independent-fit.json')
        owner = read(ROOT / 'build/value-run-v1/cases-fit-v2.json')
        for name in ('control', 'candidate'):
            for group in ('all', 'capture'):
                for key in ('normalized_huber', 'normalized_mae', 'seed_mae', 'seed_rmse', 'seed_bias'):
                    self.assertAlmostEqual(owner['metrics'][name][group][key], independent['metrics'][group][name][key], places=12)
        self.assertEqual(owner['passed'], independent['fit_passed'])
        EVIDENCE['all_search_cases'] = dict(timed_measurements=512, fixed_searches=138, fixed_requested_backups=17664,
            fixed_actual_backups=sum(r['answer']['decision']['completed_simulations'] for r in rows if r['task']['mode'] == 'fixed128'),
            h8_regret=metrics, h8_regret_gates=gates, exact_guard=guards, low_active_exact_pairs=low,
            fallbacks=dict(fallbacks), policy_fixed=True, optimizer_steps_here=0, fit_matches_independent=True)

    def test_04_all192_arena_ownership_and_joint_cluster_intervals(self):
        from congklak import value_run_arena as arena, value_run_resources as resources, neurodiag_data as metadata
        path = arena.DEFAULT; report = arena.verify(path)
        identity = read(path / 'identity.json'); head = read(path / 'head.json')
        games = [read(path / r['name']) for r in head['games']]
        decisions = [read(path / r['name']) for r in head['decisions']]
        self.assertEqual(len(games), 192); self.assertEqual(len(identity['schedule']), 192)
        self.assertEqual(len({u['family_index'] for u in identity['schedule']}), 24)
        grouped = defaultdict(dict)
        for game in games:
            unit = identity['schedule'][game['index']]
            self.assertEqual(unit['index'], game['index'])
            key = (unit['profile'], unit['baseline'], unit['seat'])
            self.assertNotIn(key, grouped[unit['family_index']]); grouped[unit['family_index']][key] = game
            if game['terminal']:
                self.assertEqual(sum(map(sum, game['final']['pits'])), 0)
                margin = game['final']['stores'][0] - game['final']['stores'][1]
                self.assertEqual(game['model_margin'], margin * (1 if unit['seat'] == 0 else -1))
            else: self.assertTrue(game['capped'] and game['choices'] == 512)
        for values in grouped.values():
            self.assertEqual(set(values), {(p, b, s) for p in ('control', 'candidate') for b in ('H_v1', 'H8_relay_CPU') for s in (0, 1)})
        inventory = read(arena.RESERVATIONS / 'inventory.json')['components']
        selection = read(arena.RESERVATIONS / 'selection.json')
        old_owners = defaultdict(set)
        for cid, component in inventory.items():
            for q in component['quotients']: old_owners[q].add(cid)
        protected = {role: {r['component'] for r in selection[role]} for role in ('train', 'fit_check')}
        families = {role: {f for cid in protected[role] for f in inventory[cid]['families']} for role in protected}
        observed_q = defaultdict(set); observed_components = defaultdict(set); role_hits = []
        for unit in identity['schedule']: observed_components[unit['component']].add(unit['family_index'])
        by_game = defaultdict(list)
        for decision in decisions:
            self.assertEqual(decision['cpu_verified'], dict(group=0, logical=2, mask=4))
            self.assertEqual(decision['answer']['decision']['requested_ms'], 25)
            by_game[decision['game_index']].append(decision)
        observations = 0
        for game in games:
            unit = identity['schedule'][game['index']]
            states = [unit['start']] + [r['transition']['after'] for r in by_game[game['index']]]
            for at, p in enumerate(states):
                if sum(map(sum, p['pits'])) <= 4: continue
                observations += 1; q = metadata.quotient(p); old = old_owners[q]
                observed_q[q].add(unit['family_index'])
                for cid in old: observed_components[cid].add(unit['family_index'])
                roles = [role for role in protected if old & protected[role] or unit['family'] in families[role]]
                if roles: role_hits.append((game['index'], at, roles))
        cross_q = {q: ids for q, ids in observed_q.items() if len(ids) > 1}
        cross_components = {cid: ids for cid, ids in observed_components.items() if len(ids) > 1}
        ownership_passed = not role_hits and not cross_q and not cross_components
        self.assertEqual(report['ownership']['observations'], observations)
        self.assertEqual(len(report['ownership']['support_or_fit_hits']), len(role_hits))
        self.assertEqual(report['ownership']['ownership_passed'], ownership_passed)
        self.assertEqual(len(report['ownership']['cross_arena_alias_component_hits']), len(cross_components))
        valid = ownership_passed and all(g['terminal'] and not g['fallbacks'] for g in games)
        self.assertEqual(report['cluster_analysis_valid'], valid)
        statistics = {}
        if valid:
            keys = [p + '/' + b for p in ('control', 'candidate') for b in ('H_v1', 'H8_relay_CPU')] + ['candidate-minus-control/' + b for b in ('H_v1', 'H8_relay_CPU')]
            family_rows = []
            for index in range(24):
                row = {p + '/' + b: sum(grouped[index][p, b, s]['model_margin'] for s in (0, 1)) / 2
                       for p in ('control', 'candidate') for b in ('H_v1', 'H8_relay_CPU')}
                for b in ('H_v1', 'H8_relay_CPU'): row['candidate-minus-control/' + b] = row['candidate/' + b] - row['control/' + b]
                family_rows.append(row)
            rng = random.Random(20261019); bootstrap = {key: [] for key in keys}
            for _ in range(4096):
                sample = [rng.randrange(24) for _ in range(24)]
                for key in keys: bootstrap[key].append(sum(family_rows[index][key] for index in sample) / 24)
            for key in keys:
                ordered = sorted(bootstrap[key])
                def quantile(probability):
                    at = probability * 4095; lo = math.floor(at); hi = math.ceil(at)
                    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)
                mean = sum(row[key] for row in family_rows) / 24
                interval = [quantile(.025), quantile(.975)]
                self.assertEqual(report['cluster_statistics'][key]['mean'], mean)
                self.assertEqual(report['cluster_statistics'][key]['interval95'], interval)
                self.assertEqual(report['cluster_statistics'][key]['clusters'], 24)
                statistics[key] = dict(mean=mean, interval95=interval)
        else: self.assertEqual(report['cluster_statistics'], {})
        ledger = resources.status()
        self.assertIsNone(ledger['pending']); self.assertLessEqual(ledger['charged_ns'], 5_400_000_000_000)
        for event in ledger['events']:
            if event['kind'] == 'reserve': self.assertLessEqual(event['lease']['reserved_ns'], 300_000_000_000)
            else:
                self.assertFalse(event['unknown_overrun'])
                self.assertLessEqual(event['peak_private_commit_bytes'], 4 * 1024 ** 3)
        prefix_proofs = []
        for version in (1, 2):
            pause = read(ROOT / ('build/value-run-v1/arena-pre-resume-prefix-v' + str(version) + '.json'))
            if version == 1: self.assertEqual((pause['games'], pause['decisions']), (68, 2423))
            for name, expected in pause['raw_committed'].items():
                file = ROOT / name
                self.assertEqual(dict(sha256=sha(file), bytes=file.stat().st_size, mtime_ns=file.stat().st_mtime_ns), expected)
            prefix_proofs.append(dict(version=version, games=pause['games'], decisions=pause['decisions'],
                preserved_files=len(pause['raw_committed']), sha_size_mtime_unchanged=True))
        unclean = [event for event in ledger['events'] if event['kind'] == 'finish' and not event['clean']]
        self.assertGreaterEqual(len(unclean), 2)
        for event in unclean: self.assertEqual(event['charged_ns'], 300_000_000_000)
        EVIDENCE['all_arena_games'] = dict(games=192, whole_family_clusters=24, decisions=len(decisions),
            terminal_games=sum(g['terminal'] for g in games), fallbacks=sum(g['fallbacks'] for g in games),
            capped_games=sum(g['capped'] for g in games), checked_active_gt4_observations=observations,
            protected_support512_and_fit128=True, support_or_fit_hits=len(role_hits),
            cross_quotient_clusters=len(cross_q), cross_historical_component_clusters=len(cross_components),
            ownership_passed=ownership_passed, cluster_analysis_valid=valid, independently_recomputed_statistics=statistics,
            joint_resamples=4096 if valid else 0, cluster_bootstrap_seed=20261019,
            cumulative_active_ns=ledger['charged_ns'], cumulative_cap_ns=5_400_000_000_000,
            unclean_full_reserved_charges=len(unclean), production_resume_prefixes=prefix_proofs,
            unreported_inflight_work_possible_in_unclean_invocations=True,
            retained_physical_decision_files=len(list((path / 'decisions').glob('*.json'))),
            retained_physical_game_files=len(list((path / 'games').glob('*.json'))),
            policy_fixed=True, optimizer_steps_here=0, default_promotion=False)

    def test_05_completed_production_training_no_work(self):
        spec = dict(training=str(TRAINING), output=str(WORK / 'closed-training.json'))
        save(WORK / 'closed-spec.json', spec)
        execute([sys.executable, ROOT / 'tools/audit-value-run.py', '--worker', 'closed', '--spec', WORK / 'closed-spec.json'], WORK / 'closed-execution')
        EVIDENCE['completed_training'] = read(WORK / 'closed-training.json')

    def test_06_required_production_evidence_fail_closed(self):
        from congklak import value_training as trainer
        refused = []
        for name in ('catalog.json', 'certificate.json', 'base-cache.json', 'checkpoints/step000000.pt',
                     'checkpoints/step000020.pt', 'checkpoints/step000400.pt', 'checkpoints/step000380.json'):
            clone = WORK / ('required-' + name.replace('/', '-').replace('.', '-'))
            shutil.copytree(TRAINING, clone)
            cert = trainer.unseal(clone / 'certificate.json')
            cert['files'] = {str(clone / Path(key).relative_to(TRAINING)): value for key, value in cert['files'].items()}
            trainer.atomic(clone / 'certificate.json', trainer.envelope(cert))
            trainer.atomic(clone / 'COMPLETED.json', dict(certificate_sha256=sha(clone / 'certificate.json')))
            with mock.patch.object(trainer, 'DEFAULT', clone):
                trainer.completed(clone)
                (clone / name).unlink()
                before = snapshot(clone)
                with mock.patch.object(trainer, 'Writer', forbidden), mock.patch.object(trainer, '_objects', forbidden):
                    with self.assertRaises((trainer.TrainingError, FileNotFoundError, ValueError, KeyError)):
                        trainer.run(clone)
                    with self.assertRaises((trainer.TrainingError, FileNotFoundError, ValueError, KeyError)):
                        trainer.completed(clone)
                self.assertEqual(before, snapshot(clone))
            refused.append(name)
        EVIDENCE['required_production_evidence'] = dict(missing_required_files=refused, refused_before_writer_optimizer=True,
            production_copies_only=True, actual_production_bytes_untouched=True)

    def test_07_two_actual_case_checkpoint_kills(self):
        from congklak import value_run_cases_v2 as cases
        helper = recovery_helpers(); cli = ROOT / 'tools/evaluate-value-run-v2.py'
        fresh = WORK / 'cases-fresh'
        execute([sys.executable, cli, 'run', '--output', fresh, '--engineering-fixture'], WORK / 'cases-fresh-execution')
        expected = case_semantics(fresh); self.assertEqual(len(expected), 6)
        proofs = []; actual_searches = len(list((fresh / 'records').glob('*.json')))
        for boundary in ('before_decision_commit', 'after_decision_commit'):
            output = WORK / ('cases-' + boundary); marker = output / 'pause.json'
            command = [sys.executable, cli, 'run', '--output', output, '--engineering-fixture', '--pause-at', boundary, '--pause-marker', marker]
            killed = helper.kill_worker(command, WORK / (boundary + '-case-kill'), marker, boundary)
            before = committed_snapshot(output, 'cases')
            cursor = read(output / 'catalog.json')['count']
            self.assertEqual(cursor, 1 if boundary.startswith('before') else 2)
            physical_before = len(list((output / 'records').glob('*.json')))
            execute([sys.executable, cli, 'run', '--output', output, '--engineering-fixture'], WORK / (boundary + '-case-resume'))
            self.assertTrue(same_existing(before)); self.assertEqual(case_semantics(output), expected)
            self.assertEqual(cases.completed(output)['fixed_searches'], 6)
            total = len(list((output / 'records').glob('*.json'))); actual_searches += total
            self.assertEqual(total, 7 if boundary.startswith('before') else 6)
            proofs.append(dict(boundary=boundary, killed=killed, committed_before=cursor,
                physical_before=physical_before, retained_physical_records=total,
                preserved_bytes_sizes_mtimes=before, resumed_searches=6 - cursor,
                exact_fresh_actions_visits_Q_and_physical_work=True))
        self.assertEqual(actual_searches, 19)
        EVIDENCE['actual_case_recovery'] = dict(actual_process_kills=2, fresh_and_resumed_physical_searches=actual_searches,
            requested_fixed_backups=actual_searches * 128, prefix_proofs=proofs,
            process_helper_source=dict(path=rel(ROOT / 'tests/test_value_training_audit.py'), sha256=sha(ROOT / 'tests/test_value_training_audit.py')))

    def test_08_four_actual_arena_checkpoint_kills(self):
        from congklak import value_run_arena as arena
        helper = recovery_helpers(); cli = ROOT / 'tools/run-value-arena.py'
        fresh = WORK / 'arena-fresh'
        def command(path):
            return [sys.executable, cli, 'worker', '--output', path, '--engineering-fixture', '--fixed-simulations', '128']
        execute(command(fresh), WORK / 'arena-fresh-execution', timeout=240)
        expected = arena_semantics(fresh); self.assertEqual(len(expected['games']), 2)
        n = len(expected['decisions']); actual_searches = n; proofs = []
        for boundary in ('before_decision_commit', 'after_decision_commit', 'before_game_commit', 'after_game_commit'):
            output = WORK / ('arena-' + boundary); marker = output / 'pause.json'
            killed = helper.kill_worker(command(output) + ['--pause-at', boundary, '--pause-marker', marker],
                WORK / (boundary + '-arena-kill'), marker, boundary, timeout=240)
            before = committed_snapshot(output, 'arena'); head = read(output / 'head.json')
            cursor = dict(decisions=len(head['decisions']), games=len(head['games']))
            self.assertEqual(cursor['games'], 1 if boundary == 'after_game_commit' else 0)
            if 'decision' in boundary: self.assertEqual(cursor['decisions'], 0 if boundary.startswith('before') else 1)
            execute(command(output), WORK / (boundary + '-arena-resume'), timeout=240)
            self.assertTrue(same_existing(before)); self.assertEqual(arena_semantics(output), expected)
            result = arena.verify(output); self.assertTrue(result['completed']); self.assertEqual(result['terminal_games'], 2)
            total = len(list((output / 'decisions').glob('*.json'))); actual_searches += total
            self.assertEqual(total, n + (1 if boundary == 'before_decision_commit' else 0))
            proofs.append(dict(boundary=boundary, killed=killed, committed_before=cursor,
                retained_physical_decisions=total, exact_fresh_terminal_trajectories=True,
                preserved_bytes_sizes_mtimes=before, no_committed_search_repeated=True))
        self.assertEqual(actual_searches, 5 * n + 1)
        EVIDENCE['actual_arena_recovery'] = dict(actual_process_kills=4, fresh_fixture_decisions=n,
            fresh_and_resumed_physical_searches=actual_searches, requested_fixed_backups=actual_searches * 128,
            terminal_fixture_games=10, prefix_proofs=proofs,
            process_helper_source=dict(path=rel(ROOT / 'tests/test_value_training_audit.py'), sha256=sha(ROOT / 'tests/test_value_training_audit.py')))

    def test_09_completed_evaluation_CLI_and_dashboard_no_work(self):
        spec = dict(training=str(TRAINING), evaluations=True, output=str(WORK / 'all-closed.json'))
        save(WORK / 'all-closed-spec.json', spec)
        execute([sys.executable, ROOT / 'tools/audit-value-run.py', '--worker', 'closed', '--spec', WORK / 'all-closed-spec.json'], WORK / 'all-closed-execution', timeout=240)
        EVIDENCE['completed_evaluation_and_dashboard'] = read(WORK / 'all-closed.json')

    def test_10_classifier_integrity_and_negative_research_boundaries(self):
        from congklak import value_run_decision as decision
        fit = dict(passed=True)
        cases = dict(exact_guard={str(ms): dict(passed=True) for ms in (25, 100)},
            h8_regret_gates={key: dict(passed=True) for key in ('timed/25', 'timed/100', 'fixed128/0')},
            metrics={'example': dict(fallbacks=0)}, low_active=dict(passed=True), timed_measurements=512,
            fixed_searches=138, fixed_completed_backups=17664)
        arena = dict(games=192, terminal_games=192, capped_games=0, fallbacks=0,
            ownership=dict(ownership_passed=True), cluster_analysis_valid=True,
            nonnegative_paired_means_both_baselines=True, positive_paired_lower_bounds_both_baselines=True,
            positive_candidate_absolute_lower_bounds_both_baselines=True)
        conformance = dict(passed=True)
        happy = decision.classify(fit, cases, arena, conformance)
        self.assertTrue(happy['pilot_promising']); self.assertFalse(happy['promotion'])
        negative_fit = decision.classify(dict(passed=False), cases, arena, conformance)
        self.assertFalse(negative_fit['pilot_promising']); self.assertFalse(negative_fit['candidate_quarantined'])
        self.assertTrue(negative_fit['strength_interpretation_allowed'])
        regressed = copy.deepcopy(cases); regressed['h8_regret_gates']['timed/25']['passed'] = False
        research = decision.classify(fit, regressed, arena, conformance)
        self.assertFalse(research['pilot_promising']); self.assertFalse(research['candidate_quarantined'])
        bad_low = copy.deepcopy(cases); bad_low['low_active']['passed'] = False
        bad_coverage = copy.deepcopy(cases); bad_coverage['fixed_completed_backups'] = 17663
        failures = []
        for name, changed in (('low-active mismatch', bad_low), ('missing fixed backup', bad_coverage)):
            result = decision.classify(fit, changed, arena, conformance)
            self.assertFalse(result['strength_interpretation_allowed']); self.assertFalse(result['pilot_promising'])
            self.assertFalse(result['exploratory_paired_improvement']); self.assertFalse(result['candidate_beats_both_baselines_exploratory'])
            if name == 'low-active mismatch': self.assertTrue(result['candidate_quarantined'])
            failures.append(name)
        overlap = copy.deepcopy(arena); overlap['ownership']['ownership_passed'] = False; overlap['cluster_analysis_valid'] = False
        held = decision.classify(fit, cases, overlap, conformance)
        self.assertTrue(held['candidate_quarantined']); self.assertFalse(held['strength_interpretation_allowed'])
        self.assertFalse(held['promotion']); self.assertFalse(held['additional_training']); self.assertFalse(held['expanded_data'])
        EVIDENCE['decision_classifier_boundaries'] = dict(fixtures=6, successful_treatment_never_promoted=True,
            negative_fit_or_regret_preserved=True, integrity_boundaries=failures + ['ownership overlap'],
            future_optimization_or_expansion=False)

    def test_11_completed_evaluation_missing_evidence_refuses_before_work(self):
        from congklak import value_run_cases_v2 as cases, value_run_arena as arena
        records = []
        for kind, module, source, reader in (('cases', cases, cases.DEFAULT, cases.completed),
                                             ('arena', arena, arena.DEFAULT, arena.verify)):
            index_name = 'catalog.json' if kind == 'cases' else 'head.json'
            head = read(source / index_name)
            raw_name = ('records/' + head['records'][0]['name']) if kind == 'cases' else head['decisions'][0]['name']
            for missing in ('certificate.json', index_name, raw_name):
                clone = WORK / ('missing-' + kind + '-' + missing.replace('/', '-'))
                shutil.copytree(source, clone)
                certificate = read(clone / 'certificate.json')
                mapping = {}
                for name, checksum in certificate['files'].items():
                    old = Path(name) if Path(name).is_absolute() else ROOT / name
                    target = clone / old.relative_to(source)
                    mapping[str(target.resolve()) if kind == 'cases' else rel(target)] = checksum
                certificate['files'] = mapping
                module.atomic(clone / 'certificate.json', certificate)
                marker = dict(certificate_sha256=sha(clone / 'certificate.json'))
                if kind == 'cases': marker['schema'] = cases.SCHEMA + '-complete'
                module.atomic(clone / 'COMPLETED.json', marker)
                with mock.patch.object(module, 'DEFAULT', clone):
                    reader(clone)  # A valid copy must pass before corruption.
                    (clone / missing).unlink()
                    before = snapshot(clone)
                    with ExitStack() as stack:
                        stack.enter_context(mock.patch.object(cases, 'open_engines', forbidden))
                        stack.enter_context(mock.patch.object(cases, 'timed_decision', forbidden))
                        stack.enter_context(mock.patch.object(cases, 'fixed_decision', forbidden))
                        stack.enter_context(mock.patch.object(module, '_writer' if kind == 'cases' else 'Writer', forbidden))
                        for call in (lambda: reader(clone), lambda: module.status(clone), lambda: module.run(clone)):
                            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError, KeyError)):
                                call()
                    self.assertEqual(snapshot(clone), before)
                records.append(dict(kind=kind, missing=missing, valid_copy_first=True, forbidden_work_not_caught_as_success=True))
        EVIDENCE['required_evaluation_evidence'] = dict(corruption_cases=records,
            writers=0, searches=0, copies_only=True, production_bytes_unchanged=True)

    def test_12_live_dashboard_OS_listener_matches_reviewed_source(self):
        from congklak import value_run_progress_v3 as progress
        script = "$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); $p=Get-Process -Id $_.OwningProcess; [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine;affinity=[long]$p.ProcessorAffinity} }) | ConvertTo-Json -Compress"
        result = subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True, capture_output=True, text=True, timeout=30)
        rows = json.loads(result.stdout); rows = [rows] if isinstance(rows, dict) else rows
        self.assertEqual(len(rows), 1); row = rows[0]
        self.assertEqual((row['address'], row['port'], row['affinity']), ('127.0.0.1', 18766, 1))
        self.assertIn('-m congklak.value_run_progress_v3 serve --port 18766', row['command'])
        startup = [json.loads(line) for line in (ROOT / 'build/value-run-v1/server-v3.stdout.log').read_text().splitlines() if line.startswith('{')][-1]
        self.assertEqual(startup['event'], 'value_run_observer_v3_ready'); self.assertEqual(startup['pid'], row['pid'])
        self.assertEqual(startup['source_sha256'], sha(ROOT / 'congklak/value_run_progress_v3.py'))
        observer = progress.snapshot()
        self.assertEqual(observer['current_cards'], 9); self.assertEqual(observer['production']['steps'], 400)
        self.assertEqual(observer['production']['draws'], 51200); self.assertTrue(observer['observer_source_current'])
        EVIDENCE['dashboard_OS_listener'] = dict(listener=row, startup=startup,
            direct_backend_snapshot=True, browser_or_HTTP_used=False, production_steps=400, production_draws=51200)

    def test_13_watchdog_repair_errors_limits_and_actual_permit_binding(self):
        from congklak import value_run_watchdog_v2 as adapter, value_run_resources as frozen
        self.assertEqual(adapter._size(WORK / 'not-present.pending'), 0)
        with self.assertRaises(FileNotFoundError): adapter._size(WORK / 'not-present.json')
        with mock.patch.object(Path, 'stat', side_effect=PermissionError('unrelated file error')):
            with self.assertRaises(PermissionError): adapter._size(WORK / 'not-present.pending')
        root = WORK / 'quota-fixture'; root.mkdir()
        (root / 'live.json').write_bytes(b'a' * 113)
        (root / 'atomic.pending').write_bytes(b'b' * 257)
        with mock.patch.object(adapter, 'ROOTS', (root,)):
            self.assertEqual(adapter.namespace_bytes(), 370)
            def denied(directory, onerror):
                onerror(PermissionError('unrelated directory error'))
                return []
            with mock.patch.object(adapter.os, 'walk', side_effect=denied):
                with self.assertRaises(PermissionError): adapter.namespace_bytes()
        witness = adapter.witness()
        owner = read(ROOT / 'build/value-run-v1/arena-watchdog-v2-owner-acceptance-v3.json')
        self.assertTrue(owner['passed']); self.assertEqual(owner['unit_methods'], 5)
        self.assertEqual(owner['source_witness'], witness)
        self.assertEqual(witness['resource_protocol'], frozen.PROTOCOL)
        self.assertEqual(frozen.PROTOCOL['lease_ns'], 300_000_000_000)
        self.assertEqual(frozen.PROTOCOL['cumulative_ns'], 5_400_000_000_000)
        self.assertEqual(frozen.PROTOCOL['private_commit_bytes'], 4 * 1024 ** 3)
        self.assertEqual(frozen.PROTOCOL['quota_bytes'], 5 * 1024 ** 3)
        ledger = frozen.status(); proofs = []
        for event in ledger['events']:
            if event['kind'] != 'reserve': continue
            lease = event['lease']; folder = Path(lease['lease_file']).parent
            permit = read(folder / 'permit.json')
            if 'enforcement_attestation' not in permit: continue
            binding = permit['enforcement_attestation']; file = Path(binding['path'])
            self.assertEqual(sha(file), binding['sha256'])
            attestation = read(file)
            self.assertEqual(permit['enforcement_implementation'], attestation)
            for key, value in witness.items(): self.assertEqual(attestation[key], value)
            self.assertEqual(attestation['token'], lease['token'])
            self.assertEqual(attestation['lease_sha256'], sha(lease['lease_file']))
            self.assertEqual(attestation['actual_watchdog'], permit['watchdog'])
            self.assertEqual(permit['worker'], read(folder / 'worker.json')['process'])
            self.assertTrue(permit['kill_on_job_close'])
            for name, checksum in attestation['sources'].items(): self.assertEqual(sha(ROOT / name), checksum)
            proofs.append(dict(token=lease['token'], attestation_path=rel(file), attestation_sha256=sha(file),
                bound_actual_watchdog=attestation['actual_watchdog'], original_worker_unchanged=True))
        self.assertGreaterEqual(len(proofs), 1)
        EVIDENCE['enforcement_repair'] = dict(pending_only_error_boundary=True,
            nonpending_missing_and_file_or_directory_permission_errors_refused=True,
            extant_pending_bytes_counted=True, unchanged_limits=frozen.PROTOCOL,
            actual_production_permit_attestations=proofs, exact_executed_helper_witness=witness)
