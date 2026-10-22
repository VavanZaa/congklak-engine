"""Screened-reservation value-only pilot using the frozen optimizer arithmetic.

The checkpoint codec and numerical helpers are imported unchanged from the
sealed ordinary trainer.  Admission, namespaces, publication and resources are
explicit new dependencies; no old module or artifact is patched.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import time
from . import config, guarded_value_pilot_v2_common as stage
from . import guarded_value_pilot_v2_data as data
from . import guarded_value_pilot_v2_sidecar as side
from . import ordinary_epoch_training as frozen

ROOT, DEFAULT = stage.ROOT, stage.DATA / 'training'
CACHE_DEFAULT, INITIAL = data.DEFAULT, side.INITIAL
RUNTIME = ROOT / 'build/training-runtime.json'
PREPARATION = stage.FOLDER / 'preparation/admission.json'
SCHEMA, CODEC = stage.SCHEMA + '-training', frozen.SCHEMA
TRAINED_PROTOCOL, HOOKS = side.PROTOCOL, frozen.HOOKS
BRANCH_SHAPES = frozen.BRANCH_SHAPES
SOURCES = ('congklak/guarded_value_pilot_v2_train.py',
    'congklak/guarded_value_pilot_v2_common.py', 'congklak/guarded_value_pilot_v2_data.py',
    'congklak/guarded_value_pilot_v2_sidecar.py', 'congklak/guarded_value_pilot_v2_resources.py',
    'congklak/ordinary_epoch_training.py', 'congklak/ordinary_epoch_data.py')
TrainingError, require = frozen.TrainingError, frozen.require
canonical, digest, sha, read = frozen.canonical, frozen.digest, frozen.sha, frozen.read
atomic, envelope, unseal, pins, current = frozen.atomic, frozen.envelope, frozen.unseal, frozen.pins, frozen.current
Writer, settings = frozen.Writer, frozen.settings
equal_state = frozen.equal_state


def _path(output, fixture=None):
    path = Path(output).resolve()
    fixture_root = (stage.FOLDER / 'fixtures/independent-training-recovery-v2').resolve()
    allowed = {fixture_root / name for name in ('uninterrupted', *HOOKS)}
    if fixture is None:
        require(path == DEFAULT.resolve() or path in allowed, 'Training output is not a registered new scope')
    elif fixture:
        require(path in allowed, 'Engineering training output is not a registered recovery fixture')
    else:
        require(path == DEFAULT.resolve(), 'Production training output differs')
    return path


def production_gate():
    """Require the one bound preparation receipt before a production update."""
    stage.admit_plan()
    receipt = read(PREPARATION)
    require(receipt.get('schema') == stage.SCHEMA + '-preparation-admission' and
        receipt.get('passed') is True and receipt.get('policy_fixed') is True and
        receipt.get('plan_sha256') == sha(stage.PLAN) and
        receipt.get('reservations_sha256') == sha(stage.PRIOR) and
        receipt.get('cache_manifest_sha256') == sha(CACHE_DEFAULT / 'manifest.json') and
        receipt.get('initial_model_sha256') == sha(INITIAL / 'model.json') and
        receipt.get('initial_conformance_passed') is True and
        receipt.get('independent_checkpoint_recovery_passed') is True and
        type(receipt.get('production_optimizer_steps')) is int and receipt['production_optimizer_steps'] == 0 and
        type(receipt.get('inputs')) is dict and bool(receipt['inputs']),
        'Screened reservations, initial conformance or independent recovery are not admitted')
    current({str((ROOT / name).resolve()): checksum for name, checksum in receipt['inputs'].items()})
    return receipt


def _input_files(fixture):
    files = [p for p in CACHE_DEFAULT.iterdir() if p.is_file() and p.suffix == '.json']
    files += [INITIAL / name for name in ('model.json', 'certificate.json')]
    files += [RUNTIME, stage.PLAN, stage.PREFLIGHT, stage.PRIOR]
    files += [ROOT / name for name in SOURCES]
    if not fixture:
        files.append(PREPARATION)
        files += [ROOT / name for name in production_gate()['inputs']]
    return pins(sorted({Path(p).resolve() for p in files}))


def _expected_identity(fixture):
    require(type(fixture) is bool, 'Training fixture role is not Boolean')
    stage.admit_plan()
    manifest = data.admit(CACHE_DEFAULT)
    side.verify_initial(INITIAL)
    initial = read(INITIAL / 'model.json')
    return dict(schema=SCHEMA + '-identity', engineering_fixture=fixture, settings=settings(fixture),
        cache_path=str(CACHE_DEFAULT.resolve()), cache_manifest=manifest, inputs=_input_files(fixture),
        sources=pins(ROOT / name for name in SOURCES), runtime_record=read(RUNTIME),
        initial_export_sha256=sha(INITIAL / 'model.json'), initial_model_sha256=initial['sha256'],
        initial_branch_weights=initial['model']['branch_weights'], base_export=initial['model']['base_export'],
        interface_bindings=initial['model']['bindings'], trained_protocol=TRAINED_PROTOCOL,
        checkpoint_codec=CODEC, numerical_helpers_sha256=sha(ROOT / 'congklak/ordinary_epoch_training.py'),
        implementation_admission=None if fixture else production_gate())


def _identity(path, cache_path=None, fixture=None):
    identity = unseal(path / 'identity.json')
    role = identity.get('engineering_fixture')
    require(type(role) is bool, 'Training identity role is missing')
    _path(path, role)
    require(fixture is None or role is fixture, 'Training fixture/production role differs')
    require(cache_path is None or Path(cache_path).resolve() == CACHE_DEFAULT.resolve(), 'Training cache path differs')
    require(canonical(identity) == canonical(_expected_identity(role)), 'Training source, recipe or reservation identity changed')
    return identity


def _has_completion(path):
    return any((path / name).exists() for name in ('certificate.json', 'COMPLETED.json'))


def _provenance(path, identity, catalog):
    return dict(protocol=TRAINED_PROTOCOL, kind='fixed-final-step',
        engineering_fixture=identity['engineering_fixture'], pilot_candidate=not identity['engineering_fixture'],
        optimizer_steps=identity['settings']['steps'], fixed_policy=True,
        inherited_unknown_background=True, production_evaluation_admitted=False,
        training_identity_sha256=sha(path / 'identity.json'), checkpoint=catalog['checkpoints'][-1],
        checkpoint_catalog_sha256=sha(path / 'catalog.json'), cache_manifest=identity['cache_manifest'],
        settings=identity['settings'], trainer_sources=identity['sources'], runtime=identity['runtime_record'],
        base_cache_sha256=sha(path / 'base-cache.json'), frozen_base_tensors=8,
        cuda_allocations=0, selection='final-fixed-step-only')


def _result(path, identity, catalog, model):
    fixture = identity['engineering_fixture']
    return dict(schema=SCHEMA + '-result', state='complete', steps=identity['settings']['steps'],
        draws=catalog['draws'], target_steps=identity['settings']['steps'], fixed_policy=True,
        policy_fixed=True, base_frozen=True, frozen_tensors=8, frozen_parameters=12872,
        trainable_parameters=2065, model_sha256=model['sha256'], model_file_sha256=sha(path / 'model.json'),
        final_checkpoint=catalog['checkpoints'][-1], scheduled_checkpoints=identity['settings']['steps'] // identity['settings']['checkpoint_interval'],
        initialization_preserved=True, engineering_fixture=fixture, pilot_candidate=not fixture,
        production_steps=0 if fixture else identity['settings']['steps'], cuda_allocations=0,
        inherited_unknown_background=True, historically_unseen=False, promotion=False, equal_time_tests_run=False)


def _closed_files(path, catalog, fixture):
    files = [path / name for name in ('identity.json', 'catalog.json', 'base-cache.json', 'model.json', 'result.json', 'bootstrap.json')]
    files += [p for record in catalog['checkpoints'] for p in
              (path / 'checkpoints' / record['name'], (path / 'checkpoints' / record['name']).with_suffix('.json'))]
    files += sorted((path / 'commits').glob('*.json'))
    if fixture:
        files.append(path / 'fixture-executions.jsonl')
    return files


def completed(output=DEFAULT):
    """Pure persisted reader: no Torch, pickle, writer, feature or model call."""
    path = _path(output)
    identity = _identity(path)
    catalog = frozen._catalog(path, identity)
    require(catalog['complete'] is True, 'Training is not complete')
    certificate = unseal(path / 'certificate.json')
    if (path / 'COMPLETED.json').exists():
        require(read(path / 'COMPLETED.json') == dict(certificate_sha256=sha(path / 'certificate.json')), 'Training marker differs')
    expected = dict(schema=SCHEMA + '-certificate', files=pins(_closed_files(path, catalog, identity['engineering_fixture'])),
        final_step=identity['settings']['steps'], policy_fixed=True, engineering_fixture=identity['engineering_fixture'])
    require(canonical(certificate) == canonical(expected), 'Completed training closure differs')
    require(unseal(path / 'bootstrap.json')['phase'] == 'committed', 'Completed bootstrap differs')
    model = side.verify_export(path / 'model.json')
    require(canonical(model['model']['provenance']) == canonical(_provenance(path, identity, catalog)), 'Trained provenance differs')
    require(canonical(model['model']['bindings']) == canonical(identity['interface_bindings']) and
        canonical(model['model']['base_export']) == canonical(identity['base_export']), 'Trained export changed frozen bindings/policy')
    result = read(path / 'result.json')
    require(canonical(result) == canonical(_result(path, identity, catalog, model)), 'Completed training result differs')
    return result


def status(output=DEFAULT):
    path = _path(output)
    if _has_completion(path):
        return completed(path)
    if not path.exists():
        return dict(schema=SCHEMA + '-status', state='not_started', steps=0, target_steps=400, policy_fixed=True)
    require((path / 'identity.json').exists(), 'Unanchored training output refuses restart')
    identity = _identity(path)
    catalog = frozen._catalog(path, identity)
    return dict(schema=SCHEMA + '-status', state='finalizing' if catalog['complete'] else 'paused',
        steps=catalog['step'], target_steps=identity['settings']['steps'], draws=catalog['draws'],
        policy_fixed=True, engineering_fixture=identity['engineering_fixture'])


def _pause(path, name, requested, marker, step, fixture):
    if requested != name or step != 4:
        return
    require(fixture and marker is not None, 'Checkpoint pause requires registered engineering fixture')
    marker = stage.path_of(marker)
    atomic(marker, dict(event=name, step=step, pid=os.getpid(), output=str(path)))
    while True:
        time.sleep(.05)


def _publish_checkpoint(torch, path, identity, state, catalog, pause_at, marker, lease):
    from . import guarded_value_pilot_v2_resources as resources
    resources.assert_worker_lease(lease)
    step = state['step']
    file = path / 'checkpoints' / f'step{step:06d}.pt'
    file.parent.mkdir(exist_ok=True)
    records = catalog['checkpoints'] if catalog else []
    if file.exists():
        previous = torch.load(file, map_location='cpu', weights_only=True)
        require(equal_state(torch, previous, state), 'Uncommitted checkpoint replay differs; retain and hold')
    else:
        pending = file.with_name(file.name + '.pending')
        with pending.open('wb') as stream:
            torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, file)
    metadata = dict(schema=CODEC + '-checkpoint-record', identity_sha256=sha(path / 'identity.json'),
        step=step, draws=state['draws'], checkpoint_sha256=sha(file), previous_checkpoints=records,
        checkpoint_prefix_digest=digest(records), base_cache_sha256=sha(path / 'base-cache.json'))
    if file.with_suffix('.json').exists():
        require(canonical(unseal(file.with_suffix('.json'))) == canonical(metadata), 'Orphan checkpoint record differs')
    else:
        atomic(file.with_suffix('.json'), envelope(metadata))
    _pause(path, 'before_checkpoint_catalog', pause_at, marker, step, identity['engineering_fixture'])
    resources.assert_worker_lease(lease)
    records = records + [dict(step=step, name=file.name, sha256=sha(file), record_sha256=sha(file.with_suffix('.json')))]
    catalog = dict(schema=CODEC + '-catalog', identity_sha256=sha(path / 'identity.json'), checkpoints=records,
        step=step, draws=state['draws'], prefix_digest=digest(records), base_cache_sha256=sha(path / 'base-cache.json'),
        complete=step == identity['settings']['steps'])
    atomic(path / 'catalog.json', envelope(catalog))
    frozen._receipt(path, catalog)
    _pause(path, 'after_checkpoint_catalog', pause_at, marker, step, identity['engineering_fixture'])
    return catalog


def _finish(path, identity, catalog, branch):
    export = side.make_export({name: value.detach().tolist() for name, value in branch.items()},
        CACHE_DEFAULT / 'manifest.json', _provenance(path, identity, catalog))
    for name, body in (('model.json', export), ('result.json', None)):
        if name == 'result.json':
            side.verify_export(path / 'model.json')
            body = _result(path, identity, catalog, export)
        if (path / name).exists():
            require(canonical(read(path / name)) == canonical(body), 'Uncommitted final ' + name + ' differs')
        else:
            atomic(path / name, body)
    atomic(path / 'certificate.json', envelope(dict(schema=SCHEMA + '-certificate',
        files=pins(_closed_files(path, catalog, identity['engineering_fixture'])),
        final_step=identity['settings']['steps'], policy_fixed=True, engineering_fixture=identity['engineering_fixture'])))
    atomic(path / 'COMPLETED.json', dict(certificate_sha256=sha(path / 'certificate.json')))


def run(output=DEFAULT, cache_path=CACHE_DEFAULT, fixture=False, stop_after=None,
        pause_at=None, pause_marker=None, lease=None):
    path = _path(output, fixture)
    if _has_completion(path):
        result = completed(path)
        _identity(path, cache_path, fixture)
        return result
    settings_ = settings(fixture)
    require(Path(cache_path).resolve() == CACHE_DEFAULT.resolve(), 'Training source cache differs')
    require(pause_at in (None, *HOOKS) and (not pause_at or fixture), 'Invalid checkpoint pause boundary')
    require(stop_after is None or type(stop_after) is int and 0 <= stop_after <= settings_['steps'], 'Invalid cooperative stop')
    require(lease is not None, 'Training requires actual independent containment lease')
    from . import guarded_value_pilot_v2_resources as resources
    admitted = resources.bind_worker(lease)
    require(admitted['worker_output'] == str(path) and admitted['cache_path'] == str(CACHE_DEFAULT.resolve()) and
        admitted['fixture'] is fixture, 'Worker lease scope differs')
    resources.assert_worker_lease(lease)
    stage.admit_plan()
    if not fixture:
        production_gate()
    bootstrap = dict(schema=SCHEMA + '-bootstrap', cache_path=str(CACHE_DEFAULT.resolve()),
        engineering_fixture=fixture, sources=pins(ROOT / name for name in SOURCES), phase='prepared')
    if path.exists() and not (path / 'identity.json').exists():
        require({p.name for p in path.iterdir()} <= {'writer.lock', 'bootstrap.json', 'identity.json.pending', 'bootstrap.json.pending'},
            'Unanchored training artifacts refuse automatic restart')
        if (path / 'bootstrap.json').exists():
            require(canonical(unseal(path / 'bootstrap.json')) == canonical(bootstrap), 'Bootstrap identity changed')
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        if _has_completion(path):
            return completed(path)
        existed = (path / 'identity.json').exists()
        if not (path / 'bootstrap.json').exists():
            require(not existed, 'Training bootstrap is missing')
            atomic(path / 'bootstrap.json', envelope(bootstrap))
        if existed:
            identity = _identity(path, cache_path, fixture)
        else:
            identity = _expected_identity(fixture)
            atomic(path / 'identity.json', envelope(identity))
        boot = unseal(path / 'bootstrap.json')
        require(canonical(boot) in (canonical(bootstrap), canonical(dict(bootstrap, phase='committed'))), 'Bootstrap changed')
        require((path / 'catalog.json').exists() or boot['phase'] == 'prepared', 'Missing committed catalog refuses restart')
        rows, manifest = data.load_train(CACHE_DEFAULT)
        require(canonical(manifest) == canonical(identity['cache_manifest']) and len(rows) == 192, 'Admitted source cache changed')
        rows = data.fixture_rows(rows) if fixture else rows
        require(len(rows) == settings_['roots'] and all(len(row['x']) == 127 and
            math.isfinite(row['target_z']) and -1 <= row['target_z'] <= 1 for row in rows), 'Training numerical inputs differ')
        import torch
        frozen._runtime(torch, identity)
        base, branch, optimizer, x, target, base_u, sampler, cache = frozen._objects(torch, identity, rows)
        base_digest = frozen._tensor_digest(torch, base)
        if (path / 'base-cache.json').exists():
            require(canonical(read(path / 'base-cache.json')) == canonical(cache), 'Frozen policy/logit cache changed')
        else:
            require(boot['phase'] == 'prepared', 'Frozen policy/logit cache is missing')
            atomic(path / 'base-cache.json', cache)
        catalog = frozen._catalog(path, identity) if (path / 'catalog.json').exists() else None
        samples, metrics, start = [], [], 0
        if catalog:
            frozen._receipt(path, catalog)
            state = frozen._load_checkpoint(torch, path, identity, catalog['checkpoints'][-1], base, branch, optimizer, sampler, rows)
            start, samples, metrics = state['step'], state['samples'], state['metrics']
        else:
            state = frozen._state(torch, path, base, branch, optimizer, sampler, 0, samples, metrics)
            catalog = _publish_checkpoint(torch, path, identity, state, None, pause_at, pause_marker, lease)
        if boot['phase'] == 'prepared':
            atomic(path / 'bootstrap.json', envelope(dict(bootstrap, phase='committed')))
        names = {record['name'] for record in catalog['checkpoints']}
        orphans = [p for p in (path / 'checkpoints').glob('*.pt') if p.name not in names]
        require(len(orphans) <= 1, 'Multiple checkpoint orphans refuse restart')
        orphan_step = None
        if orphans:
            next_step = min(settings_['steps'], ((start // settings_['checkpoint_interval']) + 1) * settings_['checkpoint_interval'])
            try:
                orphan_step = int(orphans[0].stem.removeprefix('step'))
            except ValueError:
                raise TrainingError('Unexpected orphan checkpoint name')
            require(orphans[0].name == f'step{orphan_step:06d}.pt' and start < orphan_step <= next_step,
                'Unexpected orphan checkpoint cursor')
            require(stop_after is None or stop_after >= orphan_step, 'Stop cursor must allow orphan replay verification')
        executed, resource_stop = 0, False
        for step in range(start + 1, settings_['steps'] + 1):
            if stop_after is not None and step > stop_after:
                break
            if not frozen._before_update(resources, lease):
                if step - 1 > catalog['step']:
                    state = frozen._state(torch, path, base, branch, optimizer, sampler, step - 1, samples, metrics)
                    catalog = _publish_checkpoint(torch, path, identity, state, catalog, pause_at, pause_marker, lease)
                resource_stop = True
                break
            indices = frozen._sample(torch, sampler, rows, settings_['batch_size'])
            draws = indices.tolist(); samples.extend(draws)
            optimizer.zero_grad(set_to_none=True)
            correction = torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x[indices], branch['A'], branch['b'])), branch['v'], branch['c']).flatten()
            prediction = torch.tanh(base_u[indices] + correction)
            loss = torch.nn.functional.huber_loss(prediction, target[indices], reduction='mean', delta=0.1)
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(branch.values()), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step(); executed += 1
            require(all(value.grad is None and not value.requires_grad for value in base.values()), 'Frozen base acquired gradients')
            require(all(bool(torch.isfinite(value).all()) for value in branch.values()), 'Nonfinite value branch')
            metrics.append(dict(step=step, normalized_huber=float(loss.detach()), unclipped_gradient_l2=float(norm)))
            if fixture:
                with (path / 'fixture-executions.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(canonical(dict(step=step, pid=os.getpid(), samples_sha256=digest(draws))) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
            if step % settings_['checkpoint_interval'] == 0 or step == settings_['steps'] or step == stop_after or step == orphan_step:
                require(frozen._tensor_digest(torch, base) == base_digest, 'Frozen base/policy tensor changed')
                state = frozen._state(torch, path, base, branch, optimizer, sampler, step, samples, metrics)
                catalog = _publish_checkpoint(torch, path, identity, state, catalog, pause_at, pause_marker, lease)
            require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
        if catalog['complete']:
            resources.assert_worker_lease(lease)
            _finish(path, identity, catalog, branch)
        else:
            require(resource_stop or catalog['step'] == (stop_after if stop_after is not None and stop_after > start else start),
                'Uncommitted cooperative stop')
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
        if args.operation == 'run' and args.lease is not None:
            from . import guarded_value_pilot_v2_resources as resources
            resources.bind_worker(args.lease)
            with stage.worker_views():
                result = run(args.output, args.cache, args.engineering_fixture, args.stop_after,
                    args.pause_at, args.pause_marker, args.lease)
        else:
            result = completed(args.output) if args.operation == 'verify' else status(args.output) if args.operation == 'status' else run(
                args.output, args.cache, args.engineering_fixture, args.stop_after, args.pause_at, args.pause_marker, args.lease)
        print(canonical(result), flush=True)
        return 0
    except Exception as exc:
        print(canonical(dict(state='hold', error=str(exc), production_optimization_authorized=False)), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
