"""Independent value-training implementation acceptance; never run production steps."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('independent_value_training', ROOT / 'tests/test_value_training_audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)

def compare_worker(specification):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    states = [torch.load(path, map_location='cpu', weights_only=False) for path in specification['checkpoints']]
    equalities = []
    def exact(left, right, key):
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right), key
        elif isinstance(left, dict):
            assert isinstance(right, dict) and set(left) == set(right), key
            for name in left: exact(left[name], right[name], key + '/' + str(name))
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right), key
            for index, (a, b) in enumerate(zip(left, right)): exact(a, b, key + '/' + str(index))
        else: assert type(left) is type(right) and left == right, key
    fields = ('frozen', 'branch', 'optimizer', 'python_rng', 'torch_rng', 'sampler_rng',
              'step', 'draws', 'sample_prefix_digest', 'samples', 'base_cache_sha256')
    for index, state in enumerate(states[1:], 1):
        for key in fields: exact(states[0][key], state[key], key)
        equalities.append(dict(route=index, exact_fields=list(fields)))
    first = states[0]
    assert first['step'] == 6 and first['draws'] == 768
    assert len(first['frozen']) == 8 and sum(t.numel() for t in first['frozen'].values()) == 12872
    assert set(first['branch']) == {'A', 'b', 'v', 'c'} and sum(t.numel() for t in first['branch'].values()) == 2065
    generator = torch.Generator(device='cpu'); generator.manual_seed(20261016)
    expected = [n for _ in range(6) for n in torch.randint(0, 32, (128,), generator=generator).tolist()]
    assert first['samples'] == expected
    assert torch.equal(first['sampler_rng'], generator.get_state())
    # Independently execute the accepted six-step mathematical contract. This
    # intentionally does not call the trainer's object/step/checkpoint helpers.
    from congklak import value_training_data as data
    initial = audit.read(ROOT / 'data/neural-value-pilot-v2/interface-initial/model.json')['model']
    rows = data.load_train()[0][:32]
    frozen = {k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in initial['base_export']['model']['weights'].items()}
    branch = {k: torch.tensor(initial['branch_weights'][k], dtype=torch.float32, device='cpu', requires_grad=True) for k in ('A', 'b', 'v', 'c')}
    x = torch.tensor([r['x'] for r in rows], dtype=torch.float32, device='cpu')
    z = torch.tensor([r['target_z'] for r in rows], dtype=torch.float32, device='cpu')
    control = []
    with torch.no_grad():
        for vector in x:
            h = vector
            for prefix in ('layers.0', 'layers.2'):
                h = (frozen[prefix + '.weight'] @ h + frozen[prefix + '.bias']).clamp_min(0)
            control.append((frozen['layers.4.weight'] @ h + frozen['layers.4.bias'])[0])
    control = torch.stack(control)
    optimizer = torch.optim.AdamW(list(branch.values()), lr=.0003, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0.0, foreach=False, fused=False)
    sample_generator = torch.Generator(device='cpu'); sample_generator.manual_seed(20261016)
    for _ in range(6):
        indices = torch.randint(32, (128,), generator=sample_generator, device='cpu')
        optimizer.zero_grad(set_to_none=True)
        hidden = torch.nn.functional.linear(x[indices], branch['A'], branch['b']).clamp_min(0)
        correction = torch.nn.functional.linear(hidden, branch['v'], branch['c']).flatten()
        prediction = torch.tanh(control[indices] + correction)
        loss = torch.nn.functional.huber_loss(prediction, z[indices], delta=.1, reduction='mean')
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(branch.values()), 1., error_if_nonfinite=True, foreach=False)
        optimizer.step()
    for name in branch: exact(first['branch'][name], branch[name], 'independent_update/' + name)
    exact(first['optimizer'], optimizer.state_dict(), 'independent_optimizer')
    exact(first['frozen'], frozen, 'independent_frozen')
    assert not torch.cuda.is_initialized()
    audit.save(specification['output'], dict(passed=True, final_step=6, draws=768, exact_routes=equalities,
        independent_sampler_replay=True, independently_executed_optimizer_steps=6,
        frozen_parameters=12872, trainable_parameters=2065,
        torch=torch.__version__, dtype='float32', device='cpu', CUDA_initialized=False))

def conformance_worker(specification):
    import torch
    from congklak import features_relay, value_sidecar_v2 as side
    from congklak.neural_relay import rounded
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = side.Model(specification['model'])
    frozen = {k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in model.base.weights.items()}
    branch = {k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in model.weights.items()}
    maximum = dict(seeds=0., logits=0., score_units=0); nonterminal = 0
    with torch.no_grad():
        for position in specification['positions']:
            raw_x, d, r = features_relay.encode(position, 127)
            if not r: continue
            nonterminal += 1; x = torch.tensor(raw_x, dtype=torch.float32, device='cpu'); hidden = x
            for prefix in ('layers.0', 'layers.2'):
                hidden = torch.relu(torch.mv(frozen[prefix + '.weight'], hidden) + frozen[prefix + '.bias'])
            value = (torch.mv(frozen['layers.4.weight'], hidden) + frozen['layers.4.bias'])[0]
            policy_before = torch.mv(frozen['policy.weight'], hidden) + frozen['policy.bias']
            if 25 <= sum(map(sum, position['pits'])) <= 74:
                value += (torch.mv(branch['v'], torch.relu(torch.mv(branch['A'], x) + branch['b'])) + branch['c'])[0]
            policy_after = torch.mv(frozen['policy.weight'], hidden) + frozen['policy.bias']
            assert torch.equal(policy_before, policy_after)
            residual = float((r * torch.tanh(value)).item())
            expected = audit.literal(model.base, model.weights, position)
            maximum['seeds'] = max(maximum['seeds'], abs(d + residual - expected[0]))
            maximum['logits'] = max(maximum['logits'], max(abs(a - b) for a, b in zip(policy_after.tolist(), expected[2])))
            maximum['score_units'] = max(maximum['score_units'], abs(d * 256 + rounded(residual * 256) - expected[1]))
    assert maximum['seeds'] <= .001 and maximum['logits'] <= 1e-5 and maximum['score_units'] <= 1, maximum
    assert not torch.cuda.is_initialized()
    audit.save(specification['output'], dict(passed=True, torch=torch.__version__, rows=len(specification['positions']),
        nonterminal=nonterminal, maximum_errors=maximum, policy_bit_exact=True, CUDA_initialized=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt')
    parser.add_argument('--methods', nargs='*')
    parser.add_argument('--fixtures-from')
    parser.add_argument('--worker', choices=('compare', 'closed', 'conformance'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.worker:
        body = audit.read(args.spec)
        {'compare': compare_worker, 'closed': audit.closed_worker, 'conformance': conformance_worker}[args.worker](body)
        return 0
    assert args.attempt and all(c.isalnum() or c in '-_' for c in args.attempt)
    report = audit.run(ROOT / 'build/value-training-v1' / ('audit-' + args.attempt), args.methods, args.fixtures_from)
    print(json.dumps(dict(passed=report['passed'], tests_run=report['tests_run'], failed=report['failed'], errors=report['errors']), sort_keys=True))
    return 0 if report['passed'] else 2

if __name__ == '__main__': raise SystemExit(main())
