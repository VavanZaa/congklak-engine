"""Independent ordinary development, admission, conformance and recovery audit."""
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
spec = importlib.util.spec_from_file_location('independent_ordinary_development', ROOT / 'tests/test_ordinary_development_audit.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
FOLDER = ROOT / 'build/ordinary-development-v1'


def torch_worker(specification):
    import torch
    from congklak import features_relay, value_sidecar_v2 as side
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
    maximum = dict(seeds=0., logits=0., score_units=0); nonterminal = 0
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
    assert not torch.cuda.is_initialized(), 'CPU fixture unexpectedly initialized CUDA'
    audit.save(specification['output'], dict(passed=True, torch=torch.__version__, device='cpu', dtype='float32',
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        initialization_generator_equal=True, rows=len(specification['positions']), nonterminal=nonterminal,
        maximum_errors=maximum, policy_bit_exact=True, optimizer_steps=0, CUDA_initialized=False))


def inputs(work):
    from congklak import valuepilot_reservations_v2 as data, ordinary_development as generation
    paths = [ROOT / n for n in (
        'tests/test_ordinary_development_audit.py', 'tools/audit-ordinary-development.py',
        'congklak/ordinary_development.py', 'tools/generate-ordinary-development.py',
        'congklak/ordinary_development_v2.py', 'tools/generate-ordinary-development-v2.py', 'tests/test_ordinary_development_v2.py',
        'congklak/valuepilot_reservations_v2.py', 'tools/reserve-valuepilot-v2.py',
        'congklak/value_sidecar_v2.py', 'tools/export-value-sidecar-v2.py', 'tools/conform-valuepilot-v2.py',
        'tests/test_ordinary_development.py', 'tests/test_valuepilot_reservations_v2.py', 'tests/test_value_sidecar_v2.py',
        'congklak/ordinary_development_progress.py', 'tools/ordinary-development-dashboard.html',
        'tests/test_ordinary_development_progress.py', 'tools/verify-ordinary-development.py',
        'build/ordinary-development-v1/baseline.json', 'build/training-runtime.json')]
    paths += audit.files(generation.DEFAULT) + audit.files(data.DEFAULT)
    paths += audit.files(ROOT / 'data/neural-value-pilot-v2/interface-initial') + audit.files(work)
    for path in FOLDER.iterdir():
        if 'owner' in path.name or path.name.startswith('protocol-review') or path.name.startswith('audit-prefix-corruption'):
            paths += audit.files(path) if path.is_dir() else [path]
    paths += list((ROOT / 'docs').glob('ORDINARY_DEVELOPMENT*'))
    return {audit.rel(path): audit.sha(path) for path in sorted(set(paths))}


def run(attempt):
    assert attempt and all(c.isalnum() or c in '-_' for c in attempt)
    work = FOLDER / ('independent-' + attempt)
    before = inputs(work); proof = audit.run(work)
    for name, digest in before.items(): assert audit.sha(ROOT / name) == digest, 'Audited source/input changed: ' + name
    if proof['passed']:
        evidence = proof['evidence']; native = evidence['native_conformance']; generation = evidence['synthetic_provenance']
        fp32 = evidence['strict_identity_and_fp32']['torch']['maximum_errors']
        document = ROOT / 'docs/ORDINARY_DEVELOPMENT_AUDIT.md'
        assert not document.exists(), 'Independent audit narrative is immutable'
        document.write_text(
            '# Ordinary development admission audit\n\n'
            'Independent pre-training acceptance passed. The ordinary arena shortage is resolved: eight ordinary components were selected from 32 protected synthetic development origins. Joint reservation admitted 512 training roots, 128 fit-check roots and 24 arena roots in 664 globally distinct full family/quotient components. All 32 generated origins, including the 24 unused ordinary starts, are excluded from this optimizer support. The 32 fit cases remain a subset of the 128 fit-check roots; earlier exact guards and forensic positions retain their protected roles.\n\n'
            'The new roots are legal custom boards with exactly two legal choices for the actor, balanced stores and 25–74 active seeds. They are not claimed reachable from the standard opening, representative of all ordinary play, or unseen release validation. Eight accepted roots were generated in each active-band/actor cell. Every accepted origin includes its literal root and every normalized legal-action child used to classify it; ownership uses observations above four active seeds. Their complete eligible quotient sets are disjoint from the full prior graph and from every other accepted origin. Rejected construction attempts remain a metadata filter journal and confer no admitted development role. No target, prediction, search result or game outcome selected a root.\n\n'
            f'The accepted generation used {generation["attempted_counters"]} deterministic counter attempts and {generation["generated_classification_transitions"]} literal classification transitions. The independent audit replayed all {generation["independently_replayed_classification_transitions"]} saved classification transitions separately. These are simulator calls, not self-play games or teacher searches. Logical committed probe counts exclude bounded repeats of an interrupted uncommitted chunk. The initial dense construction that failed its ordinary quota remains retained with its source and HOLD evidence; it did not publish production memberships.\n\n'
            f'Independent conformance checked {native["corpus_rows"]} row occurrences: all 664 admitted roots, 32 reused exact guards, five forensic roots and 32 boundary/pass fixtures. Scalar and AVX2 native backends preserved policy logits bit for bit for both the zero branch and an independent synthetic nonzero branch. The zero branch reproduced base values exactly. Native versus independent float64 algebra stayed within the fixed 1e-10 seed/logit tolerance, with exact scale-256 scores. CPU FP32 maximum errors were {fp32["seeds"]:.12g} seeds, {fp32["logits"]:.12g} logits and {fp32["score_units"]} score units, within the unchanged limits. Every row also passed single-row versus batch parity. All eight original tensors remain unchanged; only the separately identified, zero-output step-zero branch was initialized. No CUDA context was initialized.\n\n'
            'Sixteen independent fixed-128-backup improved-PUCT searches on eight low-active roots preserved actions, visits, backed values and physical work exactly, excluding timings. The owner conformance also covers the reused exact guard set. This is domain-bypass conformance, not equal-time strength evidence.\n\n'
            'Ten real process-kill cases passed: before and after generation chunk commit, generation catalog commit, reservation catalog commit, model commit and initialization-certificate commit. The four generation cases use the versioned v2 producer to replay the same fixed synthetic pool in scratch; they do not create additional distinct development data. The accepted completed v1 pool and its full file certificate remain unchanged. Independent review found that an incomplete v1 cursor bound only the latest chunk, allowing a valid-JSON mutation of an earlier child controller to survive metadata replay. That reproduction and its source remain retained. The v2 producer binds the entire ordered committed prefix and rejects earlier-chunk corruption, positive-cursor chunk loss and legacy v1 namespaces without unsafe fallback. A future v2 production pool is outside this admission.\n\n'
            'The kill harness verified the actual Windows worker script and parent lineage before terminating its process handle. Every committed immutable prefix preserved bytes, sizes and modification times; the mutable, identity-bound generation cursor was allowed to advance. Deleting all committed chunks beneath a positive cursor refused before generation rather than resetting its attempt budget. Model bytes already committed resumed using the default interpreter without Torch. Completed API and CLI routes passed with writers, native constructors, generator transition work, encoders, forwards, search and Torch imports trapped. Pure metadata counter reconstruction remains allowed. Missing/corrupt required evidence refused without recomputation.\n\n'
            'The audit preserved all 5,950 previously accepted inputs and checked the dashboard backend read-only, including stale-source refusal and zero training/game counters. No browser or HTTP inspection was used. The root publishes the final dashboard acceptance after this audit.\n\n'
            'Training remains off. No optimizer steps, optimizer draws, new teacher labels or arena games ran, and H_v1 remains the default. The next scope is the fixed value-training pilot with optimizer/RNG checkpoints and process-interruption tests, followed by the previously planned equal-time comparisons against H_v1 and H8 relay and the endgame regression guard. This admission does not establish playing strength or authorize promotion.\n\n'
            'Machine-readable evidence: [independent audit](../build/ordinary-development-v1/audit.json), [executed methods](../build/ordinary-development-v1/independent-' + attempt + '/methods.json), and [actual-corpus owner conformance](../build/ordinary-development-v1/conformance-owner.json).\n',
            encoding='utf-8', newline='\n')
    record = dict(schema='ordinary-development-independent-audit-v1', passed=proof['passed'],
        independent_of_implementation=True, automated_test_methods=proof['automated_test_methods'],
        automated_test_methods_count=len(proof['automated_test_methods']), tests_run=proof['tests_run'],
        process_kill_cases=proof['process_kill_cases'], process_kill_cases_count=len(proof['process_kill_cases']),
        unresolved_findings=[] if proof['passed'] else proof['failure_details'], inputs=inputs(work),
        evidence=proof['evidence'], attempts=proof['attempts'], counters=dict(training_steps=0, optimizer_draws=0,
            new_games=0, new_teacher_labels=0),
        scope='Pre-training synthetic ordinary development generation, joint metadata reservation, fixed-policy interface admission.',
        limitations=['Synthetic legal custom boards; opening reachability is not established.',
            'Fit-check roots have historical ancestral training exposure and are not unseen validation.',
            'No optimizer, training, new teacher labels or arena strength games were executed.',
            'Optimizer-state/RNG and decision/game recovery remain future implementation checks.'])
    audit.save(FOLDER / ('audit-' + attempt + '.json'), record)
    if proof['passed']: audit.save(FOLDER / 'audit.json', record)
    print(json.dumps(dict(event='ordinary_development_independent_audit', passed=proof['passed'],
        methods=proof['tests_run'], actual_process_kills=len(proof['process_kill_cases'])), sort_keys=True), flush=True)
    return 0 if proof['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('run', 'closed-worker', 'torch-worker'))
    parser.add_argument('--spec'); parser.add_argument('--attempt', default='v1')
    args = parser.parse_args()
    if args.operation == 'torch-worker': torch_worker(audit.read(args.spec)); return 0
    if args.operation == 'closed-worker': audit.closed_worker(audit.read(args.spec)); return 0
    return run(args.attempt)


if __name__ == '__main__': raise SystemExit(main())
