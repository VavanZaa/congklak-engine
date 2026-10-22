"""Ordinary-only CPU value pilot with durable, replay-checked checkpoints.

Import and completed readers deliberately require only the standard library.
Only ``run`` after admission/locking/lease checks can import Torch. Production
settings cannot be overridden; engineering fixtures are six steps on the first
8 already reserved training roots per ordinary cell and never produce eligible candidates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/ordinary-epoch-pilot-v1/training-v2'
CACHE_DEFAULT = ROOT / 'data/ordinary-epoch-pilot-v1/cache-v2'
INITIAL = ROOT / 'data/ordinary-epoch-pilot-v1/initial-v2'
RUNTIME = ROOT / 'build/training-runtime.json'
from . import ordinary_epoch_common as stage
IMPLEMENTATION_ACCEPTANCE = stage.PREFLIGHT
SCHEMA = 'ordinary-epoch-training-v1'
TRAINED_PROTOCOL = 'ordinary-epoch/frozen-policy/value-sidecar-v1/fixed-final-step400'
HOOKS = ('before_checkpoint_catalog', 'after_checkpoint_catalog')
SOURCES = ('congklak/ordinary_epoch_training.py', 'tools/train-ordinary-epoch.py',
           'congklak/ordinary_epoch_data.py', 'congklak/ordinary_epoch_resources.py',
           'tools/prepare-ordinary-epoch.py', 'tools/run-ordinary-epoch.py',
           'congklak/ordinary_epoch_sidecar.py','congklak/ordinary_epoch_common.py',
           'congklak/ordinary_epoch_execute.py','congklak/ordinary_epoch_recovery_gate.py')
BRANCH_SHAPES = {'A': (16, 127), 'b': (16,), 'v': (1, 16), 'c': (1,)}


class TrainingError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise TrainingError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('wb') as stream:
        stream.write((canonical(body) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def envelope(body):
    return {'body': body, 'sha256': digest(body)}


def unseal(path):
    obj = read(path)
    require(isinstance(obj, dict) and set(obj) == {'body', 'sha256'} and digest(obj['body']) == obj['sha256'],
            'Damaged training envelope: ' + str(path))
    return obj['body']


def pins(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def current(files):
    for filename, checksum in files.items():
        require(Path(filename).is_file() and sha(filename) == checksum, 'Required source or evidence changed: ' + filename)


def settings(fixture=False):
    require(type(fixture) is bool, 'Fixture flag must be boolean')
    return dict(steps=6 if fixture else 400, batch_size=128, roots=32 if fixture else 192,
                checkpoint_interval=2 if fixture else 20, sampler_seed=20261016,
                initialization_seed=20261016, learning_rate=0.0003, weight_decay=0.0,
                betas=[0.9, 0.999], eps=1e-8, foreach=False, fused=False, clip_l2=1.0,
                loss='normalized-Huber', huber_delta=0.1, sampling='uniform-four-cells/within-cell-uniform-replacement',
                unit_loss_weight=True, selected_candidate='final-fixed-step-only',
                device='cpu', dtype='float32', intra_threads=1, interop_threads=1,
                deterministic_algorithms=True, optimizer='AdamW', amp=False,
                workers=0, prefetch=False, scheduler=None, gate=[25, 74])


def _path(output, fixture=None):
    path = Path(output).resolve()
    if fixture is None:
        require(path == DEFAULT.resolve() or path.is_relative_to(stage.FOLDER/'fixtures'), 'Training output namespace differs')
    elif fixture:
        require(path.is_relative_to(stage.FOLDER/'fixtures'), 'Engineering training must stay in new stage fixtures')
    else:
        require(path == DEFAULT.resolve(), 'Production training output namespace differs')
    return path


@contextmanager
def Writer(path):
    import msvcrt
    lock = Path(path) / 'writer.lock'
    with lock.open('a+b') as stream:
        if not lock.stat().st_size:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise TrainingError('Another training writer is active') from exc
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _has_completion(path):
    # Any closing artifact signals a completed/closing path. Missing peers refuse
    # instead of silently constructing an optimizer and repeating final work.
    return any((path / name).exists() for name in ('certificate.json', 'COMPLETED.json'))


def _identity(path, cache_path=None, fixture=None):
    identity = unseal(path / 'identity.json')
    require(identity.get('schema') == SCHEMA + '-identity', 'Training identity schema differs')
    require(type(identity.get('engineering_fixture')) is bool, 'Training role missing')
    expected_fixture = identity['engineering_fixture']
    _path(path, expected_fixture)
    require(canonical(identity['settings']) == canonical(settings(expected_fixture)), 'Frozen training settings differ')
    if fixture is not None:
        require(expected_fixture is fixture, 'Training fixture/production role differs')
    if cache_path is not None:
        require(identity['cache_path'] == str(Path(cache_path).resolve()), 'Training cache path differs')
    current(identity['inputs'])
    require(canonical(identity['sources']) == canonical(pins(ROOT / name for name in SOURCES)), 'Training source closure differs')
    require(canonical(identity['runtime_record']) == canonical(read(RUNTIME)), 'Recorded training runtime differs')
    require(identity['initial_export_sha256'] == sha(INITIAL / 'model.json'), 'Initialization export changed')
    from . import ordinary_epoch_data as data,ordinary_epoch_sidecar as side
    side.verify_initial(INITIAL)
    initial=read(INITIAL/'model.json')
    require(identity['cache_path']==str(CACHE_DEFAULT.resolve()) and
            canonical(identity['cache_manifest'])==canonical(data.admit(identity['cache_path'])) and
            identity['initial_model_sha256']==initial['sha256'] and
            canonical(identity['initial_branch_weights'])==canonical(initial['model']['branch_weights']) and
            canonical(identity['base_export'])==canonical(initial['model']['base_export']) and
            canonical(identity['interface_bindings'])==canonical(initial['model']['bindings']) and
            canonical(identity['inputs'])==canonical(_input_files(identity['cache_path'],expected_fixture)),
            'Training identity changed source cache, initialization, policy or interface')
    require(identity.get('trained_protocol') == TRAINED_PROTOCOL, 'Trained export protocol differs')
    if not expected_fixture:
        require(canonical(identity.get('implementation_admission')) == canonical(production_gate()), 'Implementation admission changed')
    return identity


def _catalog(path, identity):
    catalog = unseal(path / 'catalog.json')
    require(catalog.get('schema') == SCHEMA + '-catalog' and catalog.get('identity_sha256') == sha(path / 'identity.json'),
            'Checkpoint catalog identity differs')
    records = catalog.get('checkpoints')
    require(isinstance(records, list) and records and records[0]['step'] == 0, 'Missing initialization checkpoint prefix')
    previous_step = -1
    for index, record in enumerate(records):
        require(set(record) == {'step', 'name', 'sha256', 'record_sha256'}, 'Checkpoint catalog fields differ')
        step = record['step']
        require(type(step) is int and previous_step < step <= identity['settings']['steps'], 'Checkpoint cursor order differs')
        require(record['name'] == f'step{step:06d}.pt', 'Checkpoint filename/cursor differs')
        checkpoint = path / 'checkpoints' / record['name']
        require(checkpoint.is_file() and sha(checkpoint) == record['sha256'], 'Required checkpoint missing/corrupt')
        metadata_path = checkpoint.with_suffix('.json')
        require(metadata_path.is_file() and sha(metadata_path) == record['record_sha256'], 'Required checkpoint record missing/corrupt')
        metadata = unseal(metadata_path)
        expected = dict(schema=SCHEMA + '-checkpoint-record', identity_sha256=sha(path / 'identity.json'),
                        step=step, draws=step * identity['settings']['batch_size'], checkpoint_sha256=record['sha256'],
                        previous_checkpoints=records[:index], checkpoint_prefix_digest=digest(records[:index]),
                        base_cache_sha256=sha(path / 'base-cache.json'))
        require(canonical(metadata) == canonical(expected), 'Committed checkpoint prefix/metadata differs')
        previous_step = step
    require(type(catalog['step']) is int and type(catalog['draws']) is int and
            catalog['step'] == records[-1]['step'] and catalog['draws'] == catalog['step'] * identity['settings']['batch_size'] and
            catalog['prefix_digest'] == digest(records), 'Checkpoint catalog cursor/prefix differs')
    require(type(catalog['complete']) is bool and catalog['complete'] is (catalog['step'] == identity['settings']['steps']),
            'Checkpoint completion differs')
    scheduled = set(range(0, catalog['step'] + 1, identity['settings']['checkpoint_interval']))
    require(scheduled <= {r['step'] for r in records}, 'A scheduled checkpoint was removed')
    require(catalog['base_cache_sha256'] == sha(path / 'base-cache.json'), 'Frozen control-logit cache changed')
    # Independent, append-only commit receipts make a valid older catalog a
    # detectable rollback rather than a plausible pre-commit orphan.
    receipts = sorted((path / 'commits').glob('*.json'))
    require(len(records) - 1 <= len(receipts) <= len(records), 'Committed high-water mark differs from catalog')
    for index, filename in enumerate(receipts):
        require(filename.name == f'{index:06d}.json', 'Commit receipt prefix has a gap')
        expected = dict(schema=SCHEMA + '-commit', identity_sha256=sha(path / 'identity.json'),
                        checkpoint=records[index], committed_prefix_digest=digest(records[:index + 1]),
                        previous_receipt_sha256=sha(receipts[index - 1]) if index else None)
        require(canonical(unseal(filename)) == canonical(expected), 'Checkpoint commit receipt/catalog rollback differs')
    return catalog


def _receipt(path, catalog):
    records = catalog['checkpoints']; index = len(records) - 1
    file = path / 'commits' / f'{index:06d}.json'
    body = dict(schema=SCHEMA + '-commit', identity_sha256=sha(path / 'identity.json'), checkpoint=records[-1],
                committed_prefix_digest=digest(records),
                previous_receipt_sha256=sha(path / 'commits' / f'{index - 1:06d}.json') if index else None)
    if file.exists(): require(canonical(unseal(file)) == canonical(body), 'Checkpoint commit receipt changed')
    else: atomic(file, envelope(body))


def completed(output=DEFAULT):
    """Pure persisted-evidence reader: no Torch, pickle, writer, features or forward."""
    path = _path(output); identity = _identity(path); catalog = _catalog(path, identity)
    require(catalog['complete'] is True, 'Training is not complete')
    certificate = unseal(path / 'certificate.json')
    # The certificate is the durable commit. Its subsequently written marker is
    # derived, so a kill in that final gap is already complete and read-only.
    if (path / 'COMPLETED.json').exists():
        marker = read(path / 'COMPLETED.json')
        require(marker == {'certificate_sha256': sha(path / 'certificate.json')}, 'Training completion marker corrupt')
    expected_files = [path / n for n in ('identity.json', 'catalog.json', 'base-cache.json', 'model.json', 'result.json')]
    expected_files += [p for r in catalog['checkpoints'] for p in (path / 'checkpoints' / r['name'], (path / 'checkpoints' / r['name']).with_suffix('.json'))]
    expected_files += sorted((path / 'commits').glob('*.json')) + [path / 'bootstrap.json']
    if identity['engineering_fixture']:
        expected_files += [path / 'fixture-executions.jsonl']
    require(unseal(path / 'bootstrap.json')['phase'] == 'committed', 'Completed bootstrap differs')
    require(canonical(certificate) == canonical(dict(schema=SCHEMA + '-certificate', files=pins(expected_files),
                                final_step=identity['settings']['steps'], policy_fixed=True, engineering_fixture=identity['engineering_fixture'])),
            'Completed training evidence closure differs')
    from . import ordinary_epoch_sidecar as side
    model = side.verify_export(path / 'model.json'); body = model['model']
    require(set(model) == {'model', 'sha256'} and digest(body) == model['sha256'], 'Trained model envelope differs')
    provenance = body['provenance']
    require(canonical(provenance) == canonical(_provenance(path, identity, catalog)), 'Trained model provenance differs')
    require(canonical(body['bindings']) == canonical(identity['interface_bindings']) and
            canonical(body['base_export']) == canonical(identity['base_export']),
            'Trained model changed base, policy, gate or native interface')
    require(canonical(read(path / 'result.json')) == canonical(_result(path, identity, catalog, model)), 'Completed result differs')
    return read(path / 'result.json')


def status(output=DEFAULT):
    path = _path(output)
    if _has_completion(path):
        return completed(path)
    if not path.exists():
        return dict(schema=SCHEMA + '-status', state='not_started', steps=0, target_steps=400, draws=0, policy_fixed=True)
    require((path / 'identity.json').exists(), 'Unanchored training directory refuses automatic restart')
    identity = _identity(path)
    require((path / 'catalog.json').exists(), 'Missing initialized checkpoint catalog')
    catalog = _catalog(path, identity)
    return dict(schema=SCHEMA + '-status', state='finalizing' if catalog['complete'] else 'paused', steps=catalog['step'],
                target_steps=identity['settings']['steps'], draws=catalog['draws'],
                scheduled_checkpoints=sum(r['step'] > 0 and r['step'] % identity['settings']['checkpoint_interval'] == 0 for r in catalog['checkpoints']),
                policy_fixed=True, engineering_fixture=identity['engineering_fixture'])


def _input_files(cache_path,fixture):
    files = [Path(cache_path) / name for name in ('train.json', 'manifest.json', 'catalog.json')]
    # Keep every cache certificate/identity in the worker input closure, without
    # resolving fit targets or features for optimization.
    files += [p for p in Path(cache_path).iterdir() if p.is_file() and p.suffix == '.json' and p not in files]
    files += [INITIAL / n for n in ('model.json', 'certificate.json')]
    files += [RUNTIME,stage.PRIOR,stage.JOINT,stage.EPOCH,stage.PLAN]
    files += [ROOT / name for name in SOURCES]
    if not fixture:
        files.append(IMPLEMENTATION_ACCEPTANCE)
        admission=production_gate();receipt=admission['labels']
        files.extend(ROOT/receipt[k]['path'] for k in ('catalog_ref','ownership_prefix_ref','fit_manifest_ref'))
        initial=admission['initial_conformance']
        files.extend(ROOT/initial[k]['path'] for k in ('catalog_ref','ownership_prefix_ref','initial_model_ref'))
        files.extend(ROOT/name for name in admission['native_recovery']['inputs'])
    return pins(files)


def _prepare_identity(path, cache_path, fixture):
    from . import ordinary_epoch_data as data, ordinary_epoch_sidecar as side
    require(Path(cache_path).resolve()==CACHE_DEFAULT.resolve(),'Training requires exact192-root source cache')
    manifest = data.admit(cache_path)
    side.verify_initial(INITIAL)
    initial = read(INITIAL / 'model.json')
    implementation_admission=None if fixture else production_gate()
    identity = dict(schema=SCHEMA + '-identity', engineering_fixture=fixture, settings=settings(fixture),
                    cache_path=str(Path(cache_path).resolve()), cache_manifest=manifest, inputs=_input_files(cache_path,fixture),
                    sources=pins(ROOT / name for name in SOURCES), runtime_record=read(RUNTIME),
                    initial_export_sha256=sha(INITIAL / 'model.json'), initial_model_sha256=initial['sha256'],
                    initial_branch_weights=initial['model']['branch_weights'], base_export=initial['model']['base_export'],
                    interface_bindings=initial['model']['bindings'], trained_protocol=TRAINED_PROTOCOL,
                    implementation_admission=implementation_admission)
    atomic(path / 'identity.json', envelope(identity))
    return identity


def production_gate():
    """Named new recipe and independent review precede every optimizer launch."""
    stage.admit_plan()
    from . import ordinary_epoch_execute as execute
    receipt=execute.admit_labels()
    require(receipt['schema']=='ordinary-epoch-pilot-v1-label-admission' and
            type(receipt['required_roots']) is int and receipt['required_roots']==128 and
            type(receipt['completed_roots']) is int and receipt['completed_roots']==128 and
            type(receipt['fit_cases']) is int and receipt['fit_cases']==32 and
            type(receipt['horizon']) is int and receipt['horizon']==8 and
            all(receipt[k] is True for k in ('all_legal_Q_complete','features_complete','ownership_passed','support192_disjoint')) and
            type(receipt['pending_attempts']) is int and receipt['pending_attempts']==0 and receipt['hold'] is False,
            'Required fresh fit labels/ownership are incomplete; no production training')
    for key in ('catalog_ref','ownership_prefix_ref','fit_manifest_ref'):
        ref=receipt[key];require(set(ref)=={'path','sha256'} and
                                sha(stage.path_of(ROOT/ref['path']))==ref['sha256'],
                                'Closed fit-label admission reference changed')
    initial=execute.admit_initial_conformance()
    require(initial['schema']=='ordinary-epoch-pilot-v1-initial-conformance-admission' and
            all(initial[k] is True for k in ('features_exact','control_candidate_predictions_exact',
                'fixed128_Q_exact','fixed128_visits_exact','ownership_passed')) and
            type(initial['pending_attempts']) is int and initial['pending_attempts']==0 and initial['hold'] is False and
            initial['model_sha256']==sha(INITIAL/'model.json'),
            'Initial native feature/prediction/fixed128 equivalence or ownership is not admitted')
    for key in ('catalog_ref','ownership_prefix_ref','initial_model_ref'):
        ref=initial[key];require(set(ref)=={'path','sha256'} and
                                sha(stage.path_of(ROOT/ref['path']))==ref['sha256'],
                                'Closed initial-conformance admission reference changed')
    require((ROOT/initial['initial_model_ref']['path']).resolve()==(INITIAL/'model.json').resolve(),
            'Initial conformance used another model path')
    from . import ordinary_epoch_recovery_gate as recovery_gate
    recovery=recovery_gate.admit()
    require(recovery['schema']=='ordinary-epoch-pilot-v1-recovery-gate-admission' and recovery['passed'] is True and
            type(recovery['actual_kills']) is int and recovery['actual_kills']>=7 and
            type(recovery['accepted_label_boundaries']) is int and recovery['accepted_label_boundaries']==6 and
            recovery['zero_query_hold_verified'] is True and type(recovery['inputs']) is dict and bool(recovery['inputs']),
            'Independent native-label interruption and zero-query HOLD evidence is not admitted')
    for name,checksum in recovery['inputs'].items():
        p=(ROOT/name).resolve();require(p.is_relative_to(ROOT) and sha(p)==checksum,'Recovery admission proof changed')
    return dict(path=str(IMPLEMENTATION_ACCEPTANCE),sha256=sha(IMPLEMENTATION_ACCEPTANCE),
                plan_sha256=sha(stage.PLAN),epoch_id=stage.epoch()['epoch_id'],labels=receipt,
                initial_conformance=initial,native_recovery=recovery)


def _runtime(torch, identity):
    require(Path(sys.executable).resolve() == Path(identity['runtime_record']['python']).resolve(), 'Wrong recorded Torch interpreter')
    require(str(torch.__version__) == identity['runtime_record']['torch'], 'Torch runtime version changed')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1 and torch.are_deterministic_algorithms_enabled(),
            'CPU deterministic runtime settings differ')
    require(not torch.cuda.is_initialized(), 'Training unexpectedly initialized CUDA')
    random.seed(20261016); torch.manual_seed(20261016)


def _objects(torch, identity, rows):
    frozen = {k: torch.tensor(v, dtype=torch.float32, device='cpu', requires_grad=False)
              for k, v in identity['base_export']['model']['weights'].items()}
    require(sum(v.numel() for v in frozen.values()) == 12872 and len(frozen) == 8, 'Frozen parameter contract differs')
    branch = {k: torch.nn.Parameter(torch.tensor(identity['initial_branch_weights'][k], dtype=torch.float32, device='cpu'))
              for k in BRANCH_SHAPES}
    require(sum(v.numel() for v in branch.values()) == 2065 and set(branch) == set(BRANCH_SHAPES), 'Branch parameter contract differs')
    for name, shape in BRANCH_SHAPES.items():
        require(tuple(branch[name].shape) == shape, 'Wrong trainable branch shape')
    optimizer = torch.optim.AdamW(list(branch.values()), lr=0.0003, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.0, foreach=False, fused=False)
    x = torch.tensor([r['x'] for r in rows], dtype=torch.float32, device='cpu')
    target = torch.tensor([r['target_z'] for r in rows], dtype=torch.float32, device='cpu')
    u = []; policy = []
    with torch.no_grad():
        for item in x:
            h = item
            for prefix in ('layers.0', 'layers.2'):
                h = torch.relu(torch.mv(frozen[prefix + '.weight'], h) + frozen[prefix + '.bias'])
            u.append((torch.mv(frozen['layers.4.weight'], h) + frozen['layers.4.bias'])[0])
            policy.append((torch.mv(frozen['policy.weight'], h) + frozen['policy.bias']).tolist())
    base_u = torch.stack(u)
    cache = dict(schema=SCHEMA + '-base-cache', root_keys=[r['state_key'] for r in rows],
                 base_u=base_u.tolist(), policy_logits=policy, backend='torch-FP32/scalar-mv/one-thread',
                 base_export_sha256=identity['interface_bindings']['base_file_sha256'],
                 initial_sidecar_export_sha256=identity['initial_export_sha256'])
    sampler = torch.Generator(device='cpu'); sampler.manual_seed(20261016)
    return frozen, branch, optimizer, x, target, base_u, sampler, cache


def _sample(torch,sampler,rows,batch_size=128):
    """Exact uniform cell, then uniform root; fixed draw order and one RNG."""
    from .ordinary_epoch_data import CELLS
    groups=[[i for i,r in enumerate(rows) if r['cell']==cell] for cell in CELLS]
    require(all(groups),'Sampler cannot renormalize around an empty ordinary cell')
    cells=torch.randint(4,(batch_size,),generator=sampler,device='cpu')
    indices=torch.empty(batch_size,dtype=torch.int64,device='cpu')
    for cell,group in enumerate(groups):
        slots=(cells==cell).nonzero().flatten()
        picked=torch.randint(len(group),(len(slots),),generator=sampler,device='cpu')
        table=torch.tensor(group,dtype=torch.int64,device='cpu');indices[slots]=table[picked]
    return indices


def _before_update(resources,lease):
    """Check the cooperative20s boundary before the strict15s assertion."""
    if resources.remaining_ns(lease)<=20_000_000_000:return False
    resources.assert_worker_lease(lease);return True


def _tensor_digest(torch, tensors):
    h = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        numbers = value.flatten().tolist()
        h.update(struct.pack('<' + 'f' * len(numbers), *numbers))
    return h.hexdigest()


def _state(torch, path, frozen, branch, optimizer, sampler, step, samples, metrics):
    return dict(schema=SCHEMA + '-checkpoint', identity_sha256=sha(path / 'identity.json'),
                step=step, draws=len(samples), samples=list(samples), sample_prefix_digest=digest(samples),
                frozen={k: v.detach().clone() for k, v in frozen.items()},
                branch={k: v.detach().clone() for k, v in branch.items()}, optimizer=optimizer.state_dict(),
                python_rng=random.getstate(), torch_rng=torch.get_rng_state().clone(),
                sampler_rng=sampler.get_state().clone(), metrics=list(metrics),
                base_cache_sha256=sha(path / 'base-cache.json'), frozen_tensor_sha256=_tensor_digest(torch, frozen),
                branch_tensor_sha256=_tensor_digest(torch, branch))


def equal_state(torch, first, second):
    """Exact deep checkpoint equality, including all optimizer and RNG tensors."""
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and first.dtype == second.dtype and first.shape == second.shape and torch.equal(first, second)
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_state(torch, first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal_state(torch, a, b) for a, b in zip(first, second))
    return first == second


def _load_checkpoint(torch, path, identity, record, frozen, branch, optimizer, sampler,rows):
    file = path / 'checkpoints' / record['name']
    require(sha(file) == record['sha256'], 'Checkpoint changed before load')
    # weights_only admits tensor/primitive containers; no arbitrary pickle globals.
    state = torch.load(file, map_location='cpu', weights_only=True)
    require(state['schema'] == SCHEMA + '-checkpoint' and state['identity_sha256'] == sha(path / 'identity.json'), 'Loaded checkpoint identity differs')
    step = state['step']; s = identity['settings']
    require(step == record['step'] and state['draws'] == len(state['samples']) == step * s['batch_size'], 'Loaded checkpoint cursor differs')
    require(state['sample_prefix_digest'] == digest(state['samples']) and len(state['metrics']) == step, 'Loaded draw/metric prefix differs')
    require(state['base_cache_sha256'] == sha(path / 'base-cache.json'), 'Checkpoint frozen logits differ')
    require(equal_state(torch, state['frozen'], frozen), 'A frozen original tensor changed')
    require(state['frozen_tensor_sha256'] == _tensor_digest(torch, frozen) and state['branch_tensor_sha256'] == _tensor_digest(torch, state['branch']), 'Checkpoint tensor digest differs')
    require(set(state['branch']) == set(branch), 'Trainable tensors differ')
    for name, value in state['branch'].items():
        require(tuple(value.shape) == BRANCH_SHAPES[name] and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'Invalid branch checkpoint tensor')
        branch[name].data.copy_(value)
    expected_group = optimizer.state_dict()['param_groups']
    require(state['optimizer']['param_groups'] == expected_group, 'Optimizer settings changed')
    require(len(state['optimizer']['state']) == (4 if step else 0), 'Missing optimizer moments')
    if step:
        for index, value in state['optimizer']['state'].items():
            name = list(branch)[index]
            require(set(value) == {'step', 'exp_avg', 'exp_avg_sq'} and float(value['step']) == step, 'Optimizer moment cursor differs')
            for key in ('exp_avg', 'exp_avg_sq'):
                require(value[key].shape == branch[name].shape and bool(torch.isfinite(value[key]).all()), 'Invalid optimizer moment')
    check = torch.Generator(device='cpu'); check.manual_seed(s['sampler_seed'])
    samples = []
    for _ in range(step):
        samples.extend(_sample(torch,check,rows,s['batch_size']).tolist())
    require(samples == state['samples'] and torch.equal(check.get_state(), state['sampler_rng']), 'Sampler algorithm or committed sample prefix changed')
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); sampler.set_state(state['sampler_rng'])
    return state


def _pause(path, name, requested, marker, step, fixture):
    if requested != name or step != 4:
        return
    require(fixture and marker is not None, 'Checkpoint pause hooks require an engineering fixture marker')
    atomic(marker, dict(event=name, step=step, pid=os.getpid(), output=str(path)))
    print(canonical(dict(event=name, step=step, pid=os.getpid())), flush=True)
    while True:
        time.sleep(0.1)


def _publish_checkpoint(torch, path, identity, state, catalog, pause_at, marker, lease):
    from . import ordinary_epoch_resources as resources
    resources.assert_worker_lease(lease)
    step = state['step']; file = path / 'checkpoints' / f'step{step:06d}.pt'; file.parent.mkdir(exist_ok=True)
    records = catalog['checkpoints'] if catalog else []
    if file.exists():
        # A pre-commit orphan is replayed from the committed cursor and checked
        # against ALL tensors/moments/RNG/prefix fields before adoption.
        previous = torch.load(file, map_location='cpu', weights_only=True)
        require(equal_state(torch, previous, state), 'Uncommitted checkpoint replay differs; retain and hold')
    else:
        pending = file.with_name(file.name + '.pending')
        with pending.open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, file)
    metadata = dict(schema=SCHEMA + '-checkpoint-record', identity_sha256=sha(path / 'identity.json'),
                    step=step, draws=state['draws'], checkpoint_sha256=sha(file),
                    previous_checkpoints=records, checkpoint_prefix_digest=digest(records), base_cache_sha256=sha(path / 'base-cache.json'))
    if file.with_suffix('.json').exists():
        require(canonical(unseal(file.with_suffix('.json'))) == canonical(metadata), 'Orphan checkpoint record differs')
    else:
        atomic(file.with_suffix('.json'), envelope(metadata))
    _pause(path, 'before_checkpoint_catalog', pause_at, marker, step, identity['engineering_fixture'])
    resources.assert_worker_lease(lease)
    record = dict(step=step, name=file.name, sha256=sha(file), record_sha256=sha(file.with_suffix('.json')))
    records = records + [record]
    body = dict(schema=SCHEMA + '-catalog', identity_sha256=sha(path / 'identity.json'), checkpoints=records,
                step=step, draws=state['draws'], prefix_digest=digest(records), base_cache_sha256=sha(path / 'base-cache.json'),
                complete=step == identity['settings']['steps'])
    atomic(path / 'catalog.json', envelope(body))
    _receipt(path, body)
    _pause(path, 'after_checkpoint_catalog', pause_at, marker, step, identity['engineering_fixture'])
    return body


def _provenance(path, identity, catalog):
    return dict(protocol=TRAINED_PROTOCOL,kind='fixed-final-step', engineering_fixture=identity['engineering_fixture'],
                pilot_candidate=not identity['engineering_fixture'], optimizer_steps=identity['settings']['steps'],
                inherited_unknown_background=True,production_evaluation_admitted=False,
                training_identity_sha256=sha(path / 'identity.json'), checkpoint=catalog['checkpoints'][-1],
                checkpoint_catalog_sha256=sha(path / 'catalog.json'), cache_manifest=identity['cache_manifest'],
                settings=identity['settings'], trainer_sources=identity['sources'], runtime=identity['runtime_record'],
                base_cache_sha256=sha(path / 'base-cache.json'), fixed_policy=True, frozen_base_tensors=8,
                cuda_allocations=0, selection='final-fixed-step-only')


def _result(path, identity, catalog, model):
    return dict(schema=SCHEMA + '-result', state='complete', steps=identity['settings']['steps'], draws=catalog['draws'],
                target_steps=identity['settings']['steps'], fixed_policy=True, policy_fixed=True, frozen_tensors=8,
                frozen_parameters=12872, trainable_parameters=2065, model_sha256=model['sha256'],
                model_file_sha256=sha(path / 'model.json'), final_checkpoint=catalog['checkpoints'][-1],
                scheduled_checkpoints=identity['settings']['steps'] // identity['settings']['checkpoint_interval'],
                initialization_preserved=True, engineering_fixture=identity['engineering_fixture'],
                training_eligible=False,pilot_candidate=not identity['engineering_fixture'],
                inherited_unknown_background=True,original_balanced_recipe_available=False,
                production_steps=0 if identity['engineering_fixture'] else identity['settings']['steps'],
                cuda_allocations=0, promotion=False, equal_time_tests_run=False)


def _finish(path, identity, catalog, branch):
    from . import ordinary_epoch_sidecar as side
    weights = {k: v.detach().tolist() for k, v in branch.items()}
    manifest = identity['interface_bindings']['input_manifest']['path']
    export = side.make_export(weights, manifest, _provenance(path, identity, catalog))
    if (path / 'model.json').exists():
        require(canonical(read(path / 'model.json')) == canonical(export), 'Final trained export differs')
    else:
        atomic(path / 'model.json', export)
    side.verify_export(path / 'model.json')
    result=_result(path, identity, catalog, export)
    if (path/'result.json').exists():
        require(canonical(read(path/'result.json'))==canonical(result),'Uncommitted final result differs; retain and hold')
    else:atomic(path/'result.json',result)
    files = [path / n for n in ('identity.json', 'catalog.json', 'base-cache.json', 'model.json', 'result.json')]
    files += [p for r in catalog['checkpoints'] for p in (path / 'checkpoints' / r['name'], (path / 'checkpoints' / r['name']).with_suffix('.json'))]
    files += sorted((path / 'commits').glob('*.json')) + [path / 'bootstrap.json']
    if identity['engineering_fixture']:
        files += [path / 'fixture-executions.jsonl']
    atomic(path / 'certificate.json', envelope(dict(schema=SCHEMA + '-certificate', files=pins(files),
        final_step=identity['settings']['steps'], policy_fixed=True, engineering_fixture=identity['engineering_fixture'])))
    atomic(path / 'COMPLETED.json', {'certificate_sha256': sha(path / 'certificate.json')})


def run(output=DEFAULT, cache_path=CACHE_DEFAULT, fixture=False, stop_after=None, pause_at=None, pause_marker=None, lease=None):
    path = _path(output, fixture)
    # This precedes lease admission, writer creation, Torch and every forward.
    if _has_completion(path):
        result = completed(path)
        identity = _identity(path, cache_path, fixture)
        return result
    s = settings(fixture)
    require(pause_at in (None, *HOOKS) and (not pause_at or fixture), 'Invalid/non-fixture pause boundary')
    require(stop_after is None or type(stop_after) is int and 0 <= stop_after <= s['steps'], 'Invalid cooperative stop cursor')
    require(lease is not None, 'Training requires an independent resource supervisor lease')
    from . import ordinary_epoch_resources as resources
    cache_path = Path(cache_path).resolve()
    admitted_lease = resources.bind_worker(lease)
    require(admitted_lease['worker_output'] == str(path) and admitted_lease['cache_path'] == str(cache_path) and
            admitted_lease['fixture'] is fixture, 'Worker lease output/cache/role differs')
    resources.assert_worker_lease(lease)
    if not fixture:
        # Containment handshake precedes potentially large pure archive/source
        # verification, while every admission still precedes Writer and Torch.
        production_gate()
    bootstrap = dict(schema=SCHEMA + '-bootstrap', cache_path=str(cache_path), engineering_fixture=fixture,
                     sources=pins(ROOT / name for name in SOURCES), phase='prepared')
    if path.exists() and not (path / 'identity.json').exists():
        artifacts = {p.name for p in path.iterdir()}
        require(artifacts <= {'writer.lock', 'bootstrap.json', 'identity.json.pending', 'bootstrap.json.pending'},
                'Unanchored training artifacts refuse automatic restart')
        if (path / 'bootstrap.json').exists():
            require(canonical(unseal(path / 'bootstrap.json')) == canonical(bootstrap), 'Initialization bootstrap binding differs')
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        if _has_completion(path):
            return completed(path)
        existed = (path / 'identity.json').exists()
        if not (path / 'bootstrap.json').exists():
            require(not existed, 'Required training bootstrap missing')
            atomic(path / 'bootstrap.json', envelope(bootstrap))
        identity = _identity(path, cache_path, fixture) if existed else _prepare_identity(path, cache_path, fixture)
        boot = unseal(path / 'bootstrap.json')
        require(canonical(boot) in (canonical(bootstrap),canonical(dict(bootstrap, phase='committed'))), 'Training bootstrap changed')
        require((path / 'catalog.json').exists() or boot['phase'] == 'prepared', 'Missing checkpoint catalog refuses restart')
        from . import ordinary_epoch_data as data
        rows, manifest = data.load_train(cache_path)
        require(canonical(manifest) == canonical(identity['cache_manifest']) and len(rows) == 192, 'Admitted training cache changed')
        rows = data.fixture_rows(rows) if fixture else rows
        require(len(rows)==s['roots'],'Training role root count differs')
        require(all(len(r['x']) == 127 and math.isfinite(r['target_z']) and -1 <= r['target_z'] <= 1 and
                    25 <= sum(map(sum, r['state']['pits'])) <= 74 and
                    r['r'] == sum(map(sum, r['state']['pits'])) * (1 if r['state']['controller'] == 0 else -1) and
                    r['D'] == r['state']['stores'][0] - r['state']['stores'][1] for r in rows), 'Invalid training inputs or gate/controller normalization')
        import torch
        _runtime(torch, identity)
        frozen, branch, optimizer, x, target, base_u, sampler, cache = _objects(torch, identity, rows)
        frozen_digest = _tensor_digest(torch, frozen)
        if (path / 'base-cache.json').exists():
            require(canonical(read(path / 'base-cache.json')) == canonical(cache), 'Frozen FP32 base-logit/policy cache changed')
        else:
            require(boot['phase'] == 'prepared', 'Required frozen FP32 cache missing')
            atomic(path / 'base-cache.json', cache)
        catalog = _catalog(path, identity) if (path / 'catalog.json').exists() else None
        samples = []; metrics = []; start = 0
        if catalog:
            _receipt(path, catalog)
            state = _load_checkpoint(torch, path, identity, catalog['checkpoints'][-1], frozen, branch, optimizer, sampler,rows)
            start, samples, metrics = state['step'], state['samples'], state['metrics']
        else:
            state = _state(torch, path, frozen, branch, optimizer, sampler, 0, samples, metrics)
            catalog = _publish_checkpoint(torch, path, identity, state, None, pause_at, pause_marker, lease)
        if boot['phase'] == 'prepared':
            atomic(path / 'bootstrap.json', envelope(dict(bootstrap, phase='committed')))
        committed_names = {r['name'] for r in catalog['checkpoints']}
        orphans = [p for p in (path / 'checkpoints').glob('*.pt') if p.name not in committed_names]
        require(len(orphans) <= 1, 'Multiple uncommitted checkpoints refuse')
        orphan_step = None
        if orphans:
            next_step = min(s['steps'], ((start // s['checkpoint_interval']) + 1) * s['checkpoint_interval'])
            try: orphan_step = int(orphans[0].stem.removeprefix('step'))
            except ValueError: raise TrainingError('Unexpected uncommitted checkpoint filename')
            require(orphans[0].name == f'step{orphan_step:06d}.pt' and start < orphan_step <= next_step,
                    'Unexpected uncommitted checkpoint cursor')
            require(stop_after is None or stop_after >= orphan_step, 'Cooperative stop must allow pending orphan replay validation')
        executed = 0
        resource_stop = False
        for step in range(start + 1, s['steps'] + 1):
            if stop_after is not None and step > stop_after:
                break
            # Leave a real checkpoint margin under the supervisor's hard
            # deadline. The supervisor still owns the independent kill bound.
            if not _before_update(resources,lease):
                previous_step = step - 1
                if previous_step > catalog['step']:
                    resources.assert_worker_lease(lease)
                    state = _state(torch, path, frozen, branch, optimizer, sampler, previous_step, samples, metrics)
                    catalog = _publish_checkpoint(torch, path, identity, state, catalog, pause_at, pause_marker, lease)
                resource_stop = True
                break
            indices = _sample(torch,sampler,rows,s['batch_size'])
            draws = indices.tolist(); samples.extend(draws)
            optimizer.zero_grad(set_to_none=True)
            correction = torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x[indices], branch['A'], branch['b'])), branch['v'], branch['c']).flatten()
            prediction = torch.tanh(base_u[indices] + correction)
            loss = torch.nn.functional.huber_loss(prediction, target[indices], reduction='mean', delta=0.1)
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(branch.values()), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step(); executed += 1
            require(all(v.grad is None and not v.requires_grad for v in frozen.values()), 'Frozen base acquired gradients')
            require(all(bool(torch.isfinite(v).all()) for v in branch.values()), 'Nonfinite trained branch')
            metrics.append(dict(step=step, normalized_huber=float(loss.detach()), unclipped_gradient_l2=float(norm)))
            if fixture:
                with (path / 'fixture-executions.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(canonical(dict(step=step, pid=os.getpid(), samples_sha256=digest(draws))) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
            checkpoint = step % s['checkpoint_interval'] == 0 or step == s['steps'] or step == stop_after or step == orphan_step
            if checkpoint:
                require(_tensor_digest(torch, frozen) == frozen_digest, 'A frozen original tensor changed')
                resources.assert_worker_lease(lease)
                state = _state(torch, path, frozen, branch, optimizer, sampler, step, samples, metrics)
                catalog = _publish_checkpoint(torch, path, identity, state, catalog, pause_at, pause_marker, lease)
            require(not torch.cuda.is_initialized(), 'CUDA was unexpectedly initialized')
        if catalog['complete']:
            resources.assert_worker_lease(lease)
            _finish(path, identity, catalog, branch)
        else:
            require(resource_stop or catalog['step'] == (stop_after if stop_after is not None and stop_after > start else start), 'Uncommitted cooperative stop')
    result = completed(path) if _has_completion(path) else status(path)
    result = dict(result, invocation_optimizer_executions=executed, resource_stop=resource_stop)
    resources.worker_complete(lease, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'verify', 'status'))
    parser.add_argument('--output', default=str(DEFAULT)); parser.add_argument('--cache', default=str(CACHE_DEFAULT))
    parser.add_argument('--engineering-fixture', action='store_true'); parser.add_argument('--stop-after', type=int)
    parser.add_argument('--pause-at', choices=HOOKS); parser.add_argument('--pause-marker'); parser.add_argument('--lease')
    args = parser.parse_args()
    try:
        if args.operation == 'verify': result = completed(args.output)
        elif args.operation == 'status': result = status(args.output)
        else: result = run(args.output, args.cache, args.engineering_fixture, args.stop_after, args.pause_at, args.pause_marker, args.lease)
        print(canonical(result), flush=True); return 0
    except Exception as exc:
        print(canonical(dict(state='hold', error=str(exc), production_optimization_authorized=False)), flush=True); return 2


if __name__ == '__main__':
    raise SystemExit(main())
