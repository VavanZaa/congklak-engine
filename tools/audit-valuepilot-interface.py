"""Independent source-bound pre-training audit and isolated fixture workers."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('independent_value_interface', ROOT / 'tests/test_valuepilot_interface_audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
FOLDER = ROOT / 'build/valuepilot-interface-v1'


def torch_worker(specification):
    import torch
    from congklak import features_relay, value_sidecar as side
    from congklak.neural_relay import rounded
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    initial = side.Model(specification['initial']); candidate = side.Model(specification['synthetic'])
    generator = torch.Generator(device='cpu'); generator.manual_seed(20261016)
    bound = 1 / math.sqrt(127)
    assert initial.weights['A'] == torch.empty((16, 127), dtype=torch.float32, device='cpu').uniform_(-bound, bound, generator=generator).tolist()
    assert initial.weights['b'] == torch.empty((16,), dtype=torch.float32, device='cpu').uniform_(-bound, bound, generator=generator).tolist()
    frozen = {k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in initial.base.weights.items()}
    branch = {k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in candidate.weights.items()}
    maximum = dict(seeds=0., logits=0., score_units=0)
    nonterminal = 0
    with torch.no_grad():
        for position in specification['positions']:
            x, d, r = features_relay.encode(position, 127)
            if not r: continue
            nonterminal += 1; x = torch.tensor(x, dtype=torch.float32, device='cpu'); h = x
            for prefix in ('layers.0', 'layers.2'):
                h = torch.relu(torch.mv(frozen[prefix + '.weight'], h) + frozen[prefix + '.bias'])
            u = (torch.mv(frozen['layers.4.weight'], h) + frozen['layers.4.bias'])[0]
            policy_before = torch.mv(frozen['policy.weight'], h) + frozen['policy.bias']
            if 25 <= sum(map(sum, position['pits'])) <= 74:
                u = u + (torch.mv(branch['v'], torch.relu(torch.mv(branch['A'], x) + branch['b'])) + branch['c'])[0]
            policy_after = torch.mv(frozen['policy.weight'], h) + frozen['policy.bias']
            assert torch.equal(policy_before, policy_after)
            residual = float((r * torch.tanh(u)).item())
            literal = audit.literal(initial.base, candidate.weights, position)
            maximum['seeds'] = max(maximum['seeds'], abs(d + residual - literal[0]))
            maximum['logits'] = max(maximum['logits'], max(abs(a - b) for a, b in zip(policy_after.tolist(), literal[2])))
            maximum['score_units'] = max(maximum['score_units'], abs(d * 256 + rounded(residual * 256) - literal[1]))
    assert maximum['seeds'] <= .001 and maximum['logits'] <= 1e-5 and maximum['score_units'] <= 1, maximum
    # No optimizer exists. Autograd and central finite differences are confined
    # to one synthetic mathematical function, with no training-source labels.
    x = torch.linspace(-.2, .5, 127, dtype=torch.float64, device='cpu')
    tensors = {k: torch.tensor(v, dtype=torch.float64, device='cpu', requires_grad=True) for k, v in candidate.weights.items()}
    pre = tensors['A'] @ x + tensors['b']; hidden = torch.relu(pre)
    out = torch.tanh((tensors['v'] @ hidden + tensors['c'])[0] + .123)
    out.backward(); factor = 1 - float(out.detach()) ** 2
    hidden_gradient = factor * tensors['v'].detach()[0] * (pre.detach() > 0)
    assert torch.allclose(tensors['A'].grad, hidden_gradient[:, None] * x[None, :], rtol=0, atol=1e-14)
    assert torch.allclose(tensors['b'].grad, hidden_gradient, rtol=0, atol=1e-14)
    assert torch.allclose(tensors['v'].grad, factor * hidden.detach()[None, :], rtol=0, atol=1e-14)
    assert abs(float(tensors['c'].grad[0]) - factor) <= 1e-14
    active = next(i for i, v in enumerate(pre.detach().tolist()) if v > 1e-4)
    weights = {name: value.detach().tolist() for name, value in tensors.items()}
    def function(w):
        hh = [max(0., sum(a * b for a, b in zip(row, x.tolist())) + bias)
              for row, bias in zip(w['A'], w['b'])]
        return math.tanh(.123 + sum(a * b for a, b in zip(w['v'][0], hh)) + w['c'][0])
    derivatives = {}
    import copy
    for name, index in (('A', (active, 17)), ('b', (active,)), ('v', (0, active)), ('c', (0,))):
        epsilon = 1e-6; lo = copy.deepcopy(weights); hi = copy.deepcopy(weights)
        def add(w, increment):
            target = w[name]
            for coordinate in index[:-1]: target = target[coordinate]
            target[index[-1]] += increment
        add(lo, -epsilon); add(hi, epsilon)
        central = (function(hi) - function(lo)) / (2 * epsilon)
        exact = tensors[name].grad
        for coordinate in index: exact = exact[coordinate]
        error = abs(central - float(exact))
        assert error <= 1e-8, (name, error)
        derivatives[name] = dict(coordinate=list(index), central=central, analytic=float(exact), absolute_error=error)
    assert not torch.cuda.is_initialized(), 'CPU fixture unexpectedly initialized CUDA'
    audit.save(specification['output'], dict(passed=True, torch=torch.__version__, device='cpu',
        dtype='float32', threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        initialization_generator_equal=True, rows=len(specification['positions']), nonterminal=nonterminal,
        maximum_errors=maximum, policy_bit_exact=True, all_2065_analytic_derivatives_checked=True,
        central_difference_coordinates=derivatives, optimizer_steps=0, CUDA_initialized=False))


def inputs(work):
    paths = [ROOT / n for n in (
        'tests/test_valuepilot_interface_audit.py', 'tools/audit-valuepilot-interface.py',
        'congklak/valuepilot_reservations.py', 'tools/reserve-valuepilot.py',
        'congklak/value_sidecar.py', 'tools/export-value-sidecar.py',
        'native/neural_value_sidecar.cpp', 'tools/build-value-sidecar.cmd', 'tools/build-value-sidecar.ps1',
        'build/congklak_value_sidecar.dll', 'build/value-sidecar-build.json',
        'tests/test_value_sidecar.py', 'tests/test_valuepilot_reservations.py',
        'congklak/valuepilot_interface_progress.py', 'tools/valuepilot-interface-dashboard.html',
        'tests/test_valuepilot_interface_progress.py', 'tools/verify-valuepilot-interface.py',
        'build/valuepilot-interface-v1/baseline.json', 'build/valuepilot-plan-v1/acceptance.json',
        'build/training-runtime.json')]
    paths += audit.files(ROOT / 'data/neural-value-pilot-v1/reservations')
    paths += audit.files(work)
    for name in ('data-owner-acceptance.json', 'interface-owner-acceptance.json'):
        path = FOLDER / name
        if path.exists(): paths.append(path)
    return {audit.rel(path): audit.sha(path) for path in sorted(set(paths))}


def run(attempt):
    assert attempt and all(c.isalnum() or c in '-_' for c in attempt)
    work = FOLDER / ('independent-' + attempt)
    before = inputs(work)
    proof = audit.run(work)
    for name, digest in before.items():
        assert audit.sha(ROOT / name) == digest, 'Reviewed source/input changed during independent execution: ' + name
    record = dict(schema='valuepilot-interface-independent-audit-v1', passed=proof['passed'],
        independent_of_implementation=True, automated_test_methods=proof['automated_test_methods'],
        automated_test_methods_count=len(proof['automated_test_methods']), process_kill_cases=proof['process_kill_cases'],
        process_kill_cases_count=len(proof['process_kill_cases']), tests_run=proof['tests_run'],
        unresolved_findings=[] if proof['passed'] else proof['failure_details'],
        readiness_blockers=audit.read(ROOT / 'data/neural-value-pilot-v1/reservations/manifest.json')['readiness_blockers'],
        counters=dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0),
        training_ready=False, production_value_exports=0, inputs=inputs(work),
        evidence=proof['evidence'], attempts=proof['attempts'],
        limits=['Pre-training engineering acceptance with data HOLD.',
                'No production split, training, strength games or promotion.',
                'Training/arena process recovery remains future work; implemented publication recovery was tested.',
                'New ordinary development starts and actual admitted-corpus conformance are required before training.'])
    audit.save(FOLDER / ('audit-' + attempt + '.json'), record)
    if proof['passed']:
        audit.save(FOLDER / 'audit.json', record)
    print(json.dumps(dict(event='valuepilot_independent_audit', passed=proof['passed'],
                         methods=proof['tests_run'], process_kills=len(proof['process_kill_cases'])), sort_keys=True), flush=True)
    return 0 if proof['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'closed-worker', 'torch-worker'))
    parser.add_argument('--spec'); parser.add_argument('--attempt', default='v1')
    args = parser.parse_args()
    if args.operation == 'closed-worker': audit.call_closed(audit.read(args.spec)); return 0
    if args.operation == 'torch-worker': torch_worker(audit.read(args.spec)); return 0
    return run(args.attempt)


if __name__ == '__main__': raise SystemExit(main())
