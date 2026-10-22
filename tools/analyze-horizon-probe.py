"""Forensic finite-horizon comparison of six frozen disagreement roots; no search."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, reports, natural_study
from congklak.native import Native, flatten
from congklak.deeper_teacher_v2 import atomic_json
from congklak.puct_cpu_inference import CpuBatchEvaluator

MODEL = ROOT / 'data/quality-train-v1/model.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, body):
    if path.exists():
        assert json.loads(path.read_text()) == body, 'Completed forensic report differs'
    else:
        atomic_json(path, body)


def common_prefix(a, b):
    count = 0
    for x, y in zip(a, b):
        if (x['actor'], x['pit']) != (y['actor'], y['pit']):
            break
        count += 1
    return dict(shared_choices=count,
        first_different_choice=count + 1 if count < min(len(a), len(b)) else None,
        shorter_line_is_prefix=count == min(len(a), len(b)))


def replay_line(root, candidate, depth, native):
    """Record one reported optimal finite-depth continuation for a fixed root pit."""
    p = root; steps = []; snapshots = {}
    for index, choice in enumerate(candidate['pv'], 1):
        normal = native.normalize(flatten(p), actor=p['controller'], total=98)
        p = normal['position']
        assert choice['actor'] == p['controller'] and choice['pit'] in normal['legal_actions']
        before = p
        transition = native.apply(flatten(p), actor=p['controller'], action=choice['pit'], total=98, trace=True)
        assert transition['ok'] and transition['passes'] == choice['forced_passes_after']
        p = transition['position']
        gain = [p['stores'][a] - before['stores'][a] for a in (0, 1)]
        assert gain == transition['store_delta']
        bonus = transition['reason'] == 'own_store' and not transition['terminal']
        assert bonus == choice['bonus_choice'] and transition['terminal'] == choice['terminal_after']
        capture = [e for e in transition['events'] if e['type'] == 'capture']
        assert all(e['after'] - e['before'] == e['count'] for e in capture)
        margin = p['stores'][0] - p['stores'][1]
        steps.append(dict(choice=index, actor=choice['actor'], pit=choice['pit'], display_pit=choice['pit'] + 1,
            reason=transition['reason'], store_gain=gain, stores_after=p['stores'],
            margin_player0_after=margin, active_after=sum(map(sum, p['pits'])),
            capture_events=capture, capture_transfer=sum(e['count'] for e in capture),
            bonus_choice=bonus, forced_passes_after=transition['passes'], terminal_after=transition['terminal'],
            relays=transition['relays'], deposits=transition['drops']))
        if index in (8, 10, 12, 14):
            snapshots[str(index)] = dict(stores=p['stores'], margin_player0=margin, active=sum(map(sum, p['pits'])),
                interpretation='Store margin along THIS reported line; not the independently optimized shorter-depth target')
    value = candidate['depth_result']['value_player0']
    assert p['stores'][0] - p['stores'][1] == value
    terminal = not sum(map(sum, p['pits']))
    assert terminal or len(steps) == depth
    assert candidate['pv_truncated'] is (not terminal)
    tail = [s for s in steps if s['choice'] > 8]
    return dict(pit=candidate['pit'], display_pit=candidate['pit'] + 1,
        target_player0=value, pv=candidate['pv'], steps=steps,
        leaf=dict(state=p, terminal=terminal, remaining_active=sum(map(sum, p['pits'])), margin_player0=value),
        same_line_cutoff_snapshots=snapshots,
        tail_after_choice8=dict(choices=len(tail), capture_events=sum(len(s['capture_events']) for s in tail),
            capture_transfer=sum(s['capture_transfer'] for s in tail), bonus_choices=sum(s['bonus_choice'] for s in tail),
            store_gain=[sum(s['store_gain'][a] for s in tail) for a in (0, 1)]))


def analyze_case(key, row, refs8, refs10, labels12, labels14, a12, a14, prediction, native, budgets):
    p = row['state']; sign = 1 if p['controller'] == 0 else -1
    requested = {8: refs8[key]['report'], 10: refs10[key]['report']}
    if key in labels12:
        requested[12] = labels12[key]['report']
    if key in labels14:
        requested[14] = labels14[key]['report']
    evidence = {}; candidates = {}
    for depth, report in requested.items():
        evidence[depth] = dict(source_kind='reference' if depth <= 10 else 'requested_label',
            source_job='old_h' + str(depth) if depth <= 10 else 'h' + str(depth),
            source_requested_depth=depth, source_node_budget=None if depth <= 10 else budgets[depth], report=report)
    for job, attempt in (('h14', a14[key]), ('h12', a12[key])):
        requested_depth = 14 if job == 'h14' else 12
        for report in attempt['history']:
            depth = report['choice_depth']
            if depth >= 8 and depth not in evidence:
                evidence[depth] = dict(source_kind='auxiliary_completed_iteration', source_job=job,
                    source_requested_depth=requested_depth, source_node_budget=budgets[requested_depth], report=report)
    old_best8, old_best10 = refs8[key]['label']['best_moves'], refs10[key]['label']['best_moves']
    assert not set(old_best8) & set(old_best10)
    legal = refs8[key]['report']['legal_actions']
    policy = max(legal, key=lambda a: (prediction['policy_logits'][a], -a))
    depth_rows = {}
    for depth, source in sorted(evidence.items()):
        report = source['report']
        reports.validate(report, p, config.identity(native.build_id), native)
        assert report['choice_depth'] == depth and report['search_options']['tt_mb'] == 16 and report['search_options']['use_pvs'] is True
        q = {c['pit']: c['depth_result']['value_player0'] for c in report['candidates']}
        assert list(q) == legal
        value = report['depth_result']['value_player0']; best = report['best_moves']
        actions = sorted(set(old_best8) | set(old_best10) | set(best))
        lines = {}
        for action in actions:
            candidate = next(c for c in report['candidates'] if c['pit'] == action)
            line = replay_line(p, candidate, depth, native)
            for old_depth, reference in ((8, refs8[key]), (10, refs10[key])):
                old_line = next(c['pv'] for c in reference['report']['candidates'] if c['pit'] == action)
                line['common_prefix_vs_H' + str(old_depth)] = common_prefix(old_line, candidate['pv'])
            lines[str(action)] = line
        depth_rows[str(depth)] = dict(actual_depth=depth,
            **{k: v for k, v in source.items() if k != 'report'}, source_report_sha256=config.digest(report),
            target_player0=value, best_moves=best, display_best_moves=[a + 1 for a in best],
            action_scores=[dict(pit=a, display_pit=a + 1, value_player0=q[a]) for a in legal],
            H8_lexicographic_choice_regret=sign * (value - q[min(old_best8)]),
            H10_lexicographic_choice_regret=sign * (value - q[min(old_best10)]),
            H8_best_tie_regret_min=min(sign * (value - q[a]) for a in old_best8),
            H8_best_tie_regret_max=max(sign * (value - q[a]) for a in old_best8),
            H10_best_tie_regret_min=min(sign * (value - q[a]) for a in old_best10),
            H10_best_tie_regret_max=max(sign * (value - q[a]) for a in old_best10),
            model_value_absolute_error=abs(prediction['value_player0'] - value),
            raw_policy_regret=sign * (value - q[policy]), lines=lines)
    complete_pair = key in labels12 and key in labels14
    equal = set(labels12[key]['label']['best_moves']) == set(labels14[key]['label']['best_moves']) if complete_pair else None
    deepest_requested = max(requested)
    deepest_actual = max(evidence)
    deepest_best = requested[deepest_requested]['best_moves']
    depths = sorted(evidence)
    return dict(state_key=key, state=p, family=row['family'], source_role=row['study_role'], cohort=row['cohort'],
        investigation_only=True, requested_status=dict(h12=a12[key]['status'], h14=a14[key]['status']),
        requested_depths_available=[d for d in (12, 14) if d in requested],
        auxiliary_completed_depths=[d for d, s in sorted(evidence.items()) if s['source_kind'] == 'auxiliary_completed_iteration'],
        requested_complete_pair=complete_pair, same_best_set_H12_H14=equal,
        classification=('same_best_set_at_H12_H14' if equal else 'different_best_sets_at_H12_H14') if complete_pair else 'requested_comparison_incomplete',
        deepest_requested_depth=deepest_requested, deepest_actual_completed_depth=deepest_actual,
        deepest_requested_best_overlap_H8=bool(set(deepest_best) & set(old_best8)),
        deepest_requested_best_overlap_H10=bool(set(deepest_best) & set(old_best10)),
        available_depth12_and14_best_sets_equal=(set(evidence[12]['report']['best_moves']) == set(evidence[14]['report']['best_moves'])) if 12 in evidence and 14 in evidence else None,
        adjacent_actual_depth_best_set_changes=sum(set(evidence[a]['report']['best_moves']) != set(evidence[b]['report']['best_moves']) for a, b in zip(depths, depths[1:]) if b == a + 1),
        model_value_player0=prediction['value_player0'], raw_policy_chosen=policy,
        reference_H8_best_moves=old_best8, reference_H10_best_moves=old_best10, depth_evidence=depth_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/horizon-probe-forensics.json')
    args = parser.parse_args()
    baseline_path = ROOT / 'build/horizon-probe-baseline.json'
    baseline = json.loads(baseline_path.read_text()); inputs = dict(baseline['inputs'])
    assert sha(MODEL) == MODEL_SHA
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Prior input changed: ' + name
    from congklak import horizon_probe
    rows, h8, h10, h12, h14, a12, a14, manifest = horizon_probe.load_rows()
    assert len(rows) == 6 and set(rows) == set(h8) == set(h10) == set(a12) == set(a14)
    expected = {r['state_key'] for r in json.loads((ROOT / 'build/natural-study-stability.json').read_text())['metrics'] if r.get('best_sets_disjoint')}
    assert set(rows) == expected and manifest is not None
    paths = [Path(__file__), baseline_path, ROOT / 'congklak/horizon_probe.py',
        *[ROOT / ('build/horizon-probe-' + suffix + '.json') for suffix in ('plan', 'registry', 'implementation', 'result')],
        ROOT / 'data/horizon-probe-v1/work.sqlite3', ROOT / 'data/horizon-probe-v1/manifest.json']
    for path in paths:
        inputs[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
    native = Native(); cases = []
    budgets = {12: 10000000, 14: 25000000}
    with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
        for index, key in enumerate(sorted(rows), 1):
            row = rows[key]
            for depth, targets in ((8, h8), (10, h10), (12, h12), (14, h14)):
                if key in targets:
                    natural_study.validate_label(targets[key], row, depth, native)
            case = analyze_case(key, row, h8, h10, h12, h14, a12, a14,
                evaluator.evaluate_many([row['state']])[0], native, budgets)
            case['case_id'] = 'D' + str(index); cases.append(case)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Input changed during analysis: ' + name
    counts = dict(roots=len(rows), h12_attempts=len(a12), h14_attempts=len(a14),
        h12_complete=len(h12), h14_complete=len(h14), requested_complete_pairs=len(set(h12) & set(h14)),
        h12_rejected=sum(a['status'] == 'budget_rejected' for a in a12.values()),
        h14_rejected=sum(a['status'] == 'budget_rejected' for a in a14.values()),
        stable_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] is True for c in cases),
        changed_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] is False for c in cases),
        requested_incomplete_cases=sum(not c['requested_complete_pair'] for c in cases),
        cases_with_deeper_requested_label=sum(c['deepest_requested_depth'] > 10 for c in cases))
    body = dict(schema='fixed-control-horizon-probe-forensics-v1', passed=True,
        fixed_model_sha256=MODEL_SHA, rules_digest=config.RULES_DIGEST, counts=counts,
        case_classifications=dict(sorted(Counter(c['classification'] for c in cases).items())), cases=cases,
        inputs=dict(sorted(inputs.items())), training_performed=False, model_changed=False,
        active_sampling_weights_changed=False, data_ingested=False, teacher_search_performed_by_analysis=False,
        new_PUCT_or_strength_tests=False, sealed_final_test_consumed=False,
        scope='Six roots selected because H8/H10 best sets were disjoint; this selected sample cannot estimate general teacher reliability',
        definitions=dict(pit_numbers='Machine pit0..6; display_pit/display_best_moves use1..7',
            same_best_set='Exact H12/H14 best-set equality among complete requested-label pairs; not a proof of convergence',
            auxiliary='Actual completed report depth with original attempt/budget provenance; never substitutes for requested-label completion',
            PV='One lexicographic optimal finite-depth continuation per fixed root pit; all reported line facts are replayed',
            common_prefix='Same actor/pit sequence; a changed earlier reply means deeper PV is not simply an extension of the shorter PV',
            regret='Actor-signed teacher-best root value minus teacher action value at the SAME depth; ties retained'),
        interpretation='H12/H14 and intermediate depths are bounded finite-horizon H_v1 estimates, not game-theoretic truth. '
            'Replay shows where particular reported lines collect stores, capture, pass or gain extra choices; it does not prove a causal game blunder. '
            'Nonterminal cutoffs retain unresolved play. Equal best sets at two even choice depths do not establish convergence, '
            'and odd/even choice depth does not imply alternating players because bonus choices exist. '
            'Original ownership and consumed development-validation status remain unchanged. No target is admitted to training.')
    publish(args.output, body)
    print(config.canonical(dict(event='horizon_probe_forensics_complete', passed=True, counts=counts,
        cases=[dict(case_id=c['case_id'], classification=c['classification'], deepest_requested_depth=c['deepest_requested_depth'],
                    deepest_actual_completed_depth=c['deepest_actual_completed_depth']) for c in cases])), flush=True)


if __name__ == '__main__':
    main()
