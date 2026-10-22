"""Independent literal sampler/optimizer replay under the pilot watchdog.

This verifier reads the admitted training-only cache and already completed
checkpoints. It never publishes a model, a candidate, or optimizer checkpoints.
No training implementation arithmetic/helper is used for the reference replay.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import ordinary_epoch_common as stage, ordinary_epoch_data as admitted_data

SCHEMA = 'ordinary-epoch-independent-optimizer-replay-v1'
FIXTURES = stage.FOLDER / 'fixtures/independent-training-recovery-v1'
OPERATIONS = ('verify_fixture_states', 'verify_production_state')
CELLS = tuple(b + '/ordinary/actor' + str(a) for b in ('25-49', '50-74') for a in (0, 1))
SOURCE_PATHS = (Path(__file__), ROOT / 'tools/audit-ordinary-epoch-training.py')


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def deep_equal(torch, left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(deep_equal(torch, left[k], right[k]) for k in left)
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(deep_equal(torch, a, b) for a, b in zip(left, right))
    return left == right


def tensor_sha(tensors):
    h = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        numbers = value.flatten().tolist()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(struct.pack('<' + 'f' * len(numbers), *numbers))
    return h.hexdigest()


def sample(torch, generator, rows):
    # One cell vector first; then each cell's root draws in cell order. The
    # grouping implementation differs from the producer but RNG order is fixed.
    drawn_cells = torch.randint(low=0, high=4, size=(128,), generator=generator).tolist()
    result = [None] * 128
    for c, name in enumerate(CELLS):
        members = [i for i, row in enumerate(rows) if row['cell'] == name]
        slots = [i for i, cell in enumerate(drawn_cells) if cell == c]
        offsets = torch.randint(low=0, high=len(members), size=(len(slots),), generator=generator).tolist()
        for slot, offset in zip(slots, offsets):
            result[slot] = members[offset]
    need(all(type(i) is int for i in result), 'Independent sampler left an unassigned slot')
    return result


def unseal(path):
    obj = stage.read(path)
    need(set(obj) == {'body', 'sha256'} and digest(obj['body']) == obj['sha256'], 'Invalid saved checkpoint envelope')
    return obj['body']


def paths_for(operation):
    if operation == 'verify_production_state':
        from congklak.ordinary_epoch_training import DEFAULT
        return [DEFAULT]
    return [FIXTURES / name for name in ('uninterrupted', 'before_checkpoint_catalog', 'after_checkpoint_catalog')]


def bindings(operation, cache):
    from congklak import ordinary_epoch_training as producer
    paths = [*SOURCE_PATHS, stage.PLAN, stage.PREFLIGHT,
             ROOT / 'build/training-runtime.json']
    paths += [p for p in cache.iterdir() if p.is_file() and p.suffix == '.json']
    for path in paths_for(operation):
        producer.completed(path)  # Pure closure; no Torch, pickle or writer.
        paths += [p for p in path.rglob('*') if p.is_file() and p.name != 'writer.lock' and not p.name.endswith('.pending')]
    return stage.common.pins(paths)


def completed(output, operation, cache):
    report = stage.read(output / 'result.json')
    fixture = operation == OPERATIONS[0]
    expected_steps, expected_checks = (18, 3) if fixture else (400, 1)
    need(report['schema'] == SCHEMA and report['passed'] is True and report['operation'] == operation and
         type(report['candidate_exports']) is int and report['candidate_exports'] == 0 and
         type(report['native_queries']) is int and report['native_queries'] == 0 and
         type(report['diagnostic_optimizer_steps']) is int and report['diagnostic_optimizer_steps'] == expected_steps and
         type(report['diagnostic_sample_draws']) is int and report['diagnostic_sample_draws'] == 128 * expected_steps and
         report['training_treatment_repeated'] is False and report['cuda_initialized'] is False and
         report['policy_fixed'] is True and len(report['checks']) == expected_checks and
         [r['path'] for r in report['checks']] == [stage.relative(p) for p in paths_for(operation)] and
         canonical(report['source_inputs']) == canonical(bindings(operation, cache)),
         'Independent optimizer replay closure differs')
    for row, path in zip(report['checks'], paths_for(operation)):
        catalog = unseal(path / 'catalog.json')
        expected = 6 if fixture else 400
        need(row['engineering_fixture'] is fixture and type(row['steps']) is int and row['steps'] == expected and
             type(row['draws']) is int and row['draws'] == expected * 128 and
             row['checked_checkpoint_steps'] == [r['step'] for r in catalog['checkpoints']] and
             row['all_optimizer_moments_rng_tensors_exact'] is True,
             'Independent checkpoint proof scope differs')
    return report


def replay(torch, path, rows, fixture, lease):
    from congklak import ordinary_epoch_resources as resources
    identity = unseal(path / 'identity.json')
    catalog = unseal(path / 'catalog.json')
    need(identity['engineering_fixture'] is fixture and catalog['complete'] is True, 'Wrong audit replay role/cursor')
    steps = 6 if fixture else 400
    need(identity['settings']['steps'] == steps and identity['settings']['batch_size'] == 128, 'Wrong fixed replay recipe')
    selected = [r for c in CELLS for r in [q for q in rows if q['cell'] == c][:8]] if fixture else rows
    need(len(selected) == (32 if fixture else 192), 'Independent replay root count differs')
    # Match the declared global-RNG boundary before tensor/optimizer setup,
    # including any deterministic framework initialization during construction.
    random.seed(20261016)
    torch.manual_seed(20261016)
    weights = identity['base_export']['model']['weights']
    base = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    branch = {k: torch.nn.Parameter(torch.tensor(identity['initial_branch_weights'][k], dtype=torch.float32))
              for k in ('A', 'b', 'v', 'c')}
    need(sum(t.numel() for t in base.values()) == 12872 and sum(t.numel() for t in branch.values()) == 2065,
         'Reference replay parameter boundary differs')
    need(all(not t.requires_grad for t in base.values()), 'Reference replay accidentally trains base')
    x = torch.tensor([r['x'] for r in selected], dtype=torch.float32)
    target = torch.tensor([r['target_z'] for r in selected], dtype=torch.float32)
    logits, policy = [], []
    with torch.no_grad():
        for row in x:
            h1 = torch.relu(torch.mv(base['layers.0.weight'], row) + base['layers.0.bias'])
            h2 = torch.relu(torch.mv(base['layers.2.weight'], h1) + base['layers.2.bias'])
            logits.append((torch.mv(base['layers.4.weight'], h2) + base['layers.4.bias'])[0])
            policy.append((torch.mv(base['policy.weight'], h2) + base['policy.bias']).tolist())
    base_u = torch.stack(logits)
    cache = stage.read(path / 'base-cache.json')
    need(canonical(cache['root_keys']) == canonical([r['state_key'] for r in selected]) and
         canonical(cache['base_u']) == canonical(base_u.tolist()) and
         canonical(cache['policy_logits']) == canonical(policy), 'Saved base/policy cache differs from independent arithmetic')
    optimizer = torch.optim.AdamW([branch[k] for k in ('A', 'b', 'v', 'c')], lr=0.0003,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False, fused=False)
    sampler = torch.Generator(device='cpu').manual_seed(20261016)
    by_step = {r['step']: r for r in catalog['checkpoints']}
    need(set(range(0, steps + 1, 2 if fixture else 20)) <= set(by_step), 'Missing scheduled retained checkpoint')
    draws, metrics, checked = [], [], []
    frozen_sha = tensor_sha(base)
    for step in range(steps + 1):
        resources.assert_worker_lease(lease)
        if step:
            indices = sample(torch, sampler, selected)
            draws.extend(indices)
            batch = torch.tensor(indices, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            hidden = torch.relu(torch.nn.functional.linear(x[batch], branch['A'], branch['b']))
            correction = torch.nn.functional.linear(hidden, branch['v'], branch['c']).flatten()
            prediction = torch.tanh(base_u[batch] + correction)
            loss = torch.nn.functional.huber_loss(prediction, target[batch], delta=0.1, reduction='mean')
            need(bool(torch.isfinite(loss)), 'Independent replay produced nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([branch[k] for k in ('A', 'b', 'v', 'c')], 1.0,
                                                 error_if_nonfinite=True, foreach=False)
            optimizer.step()
            metrics.append(dict(step=step, normalized_huber=float(loss.detach()), unclipped_gradient_l2=float(norm)))
        if step not in by_step:
            continue
        record = by_step[step]
        saved = torch.load(path / 'checkpoints' / record['name'], map_location='cpu', weights_only=True)
        need(saved['step'] == step and saved['draws'] == len(draws) and saved['samples'] == draws and
             saved['sample_prefix_digest'] == digest(draws), 'Independent sample/cursor replay differs')
        need(deep_equal(torch, saved['frozen'], base) and saved['frozen_tensor_sha256'] == frozen_sha,
             'Frozen policy/base tensor mismatch')
        need(deep_equal(torch, saved['branch'], {k: v.detach() for k, v in branch.items()}) and
             saved['branch_tensor_sha256'] == tensor_sha(branch), 'Independent branch arithmetic differs')
        need(deep_equal(torch, saved['optimizer'], optimizer.state_dict()), 'Independent AdamW moments/settings differ')
        need(deep_equal(torch, saved['python_rng'], random.getstate()) and
             torch.equal(saved['torch_rng'], torch.get_rng_state()) and
             torch.equal(saved['sampler_rng'], sampler.get_state()), 'Independent global/sampler RNG differs')
        need(canonical(saved['metrics']) == canonical(metrics), 'Independent loss/gradient prefix differs')
        checked.append(step)
    model = stage.read(path / 'model.json')['model']
    need(canonical(model['branch_weights']) == canonical({k: v.detach().tolist() for k, v in branch.items()}),
         'Final export differs from independently replayed final step')
    need(tensor_sha(base) == frozen_sha and all(t.grad is None for t in base.values()), 'Base/policy received gradients')
    need(not torch.cuda.is_initialized(), 'Diagnostic replay initialized CUDA')
    return dict(path=stage.relative(path), engineering_fixture=fixture, steps=steps, draws=len(draws),
                checked_checkpoint_steps=checked, sample_prefix_digest=digest(draws),
                sampled_cell_counts=dict(Counter(selected[i]['cell'] for i in draws)),
                final_branch_tensor_sha256=tensor_sha(branch), frozen_tensor_sha256=frozen_sha,
                all_optimizer_moments_rng_tensors_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'verify'))
    parser.add_argument('--operation', choices=OPERATIONS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--lease')
    args = parser.parse_args()
    output = stage.path_of(args.output)
    need(output == stage.FOLDER / ('audit/training-replay-fixtures' if args.operation == OPERATIONS[0]
                                  else 'audit/training-replay-production'), 'Unexpected diagnostic output path')
    cache = args.cache.resolve()
    need(cache == admitted_data.DEFAULT, 'Diagnostic replay cache differs')
    if args.command == 'verify' or (output / 'result.json').exists():
        print(canonical(completed(output, args.operation, cache)))
        return
    need(args.lease is not None, 'Diagnostic replay requires the same bounded worker ledger')
    stage.admit_plan()
    from congklak import ordinary_epoch_resources as resources, ordinary_epoch_data as data
    lease = resources.bind_worker(args.lease)
    need(lease['worker_output'] == str(output) and lease['cache_path'] == str(cache) and
         Path(lease['worker_entry']).resolve() == ROOT / 'tools/audit-ordinary-epoch-training.py',
         'Diagnostic worker lease differs')
    resources.assert_worker_lease(args.lease)
    sources = bindings(args.operation, cache)
    rows, manifest = data.load_train(cache)
    need(manifest['fit_targets_read'] == manifest['arena_targets_read'] == 0, 'Diagnostic optimizer read evaluation targets')
    runtime = stage.read(ROOT / 'build/training-runtime.json')
    need(Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'Wrong diagnostic Torch interpreter')
    import torch
    need(str(torch.__version__) == runtime['torch'], 'Diagnostic Torch version differs')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = [replay(torch, path, rows, args.operation == OPERATIONS[0], args.lease)
              for path in paths_for(args.operation)]
    if args.operation == OPERATIONS[0]:
        need(len({r['final_branch_tensor_sha256'] for r in checks}) == 1 and
             len({r['sample_prefix_digest'] for r in checks}) == 1, 'Interrupted/uninterrupted replay differs')
    stage.common.current(sources)
    body = dict(schema=SCHEMA, passed=True, operation=args.operation, source_inputs=sources, checks=checks,
                diagnostic_optimizer_steps=sum(r['steps'] for r in checks),
                diagnostic_sample_draws=sum(r['draws'] for r in checks), candidate_exports=0,
                policy_fixed=True, training_treatment_repeated=False, native_queries=0, cuda_initialized=False)
    output.mkdir(parents=True, exist_ok=False)
    stage.atomic(output / 'result.json', body)
    resources.worker_complete(args.lease, body)
    print(canonical(body))


if __name__ == '__main__':
    main()
