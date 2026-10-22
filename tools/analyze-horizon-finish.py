"""Compare six complete H14 estimates with their frozen shorter horizons; no search."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, horizon_finish as finish, natural_study
from congklak.native import Native
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.deeper_teacher_v2 import atomic_json

OUTPUT = ROOT / 'build/horizon-finish-comparison.json'
MODEL = ROOT / 'data/quality-train-v1/model.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, body):
    if path.exists():
        assert path.read_text(encoding='utf-8') == config.canonical(body) + '\n', 'Immutable comparison differs'
    else:
        atomic_json(path, body)


def main():
    baseline_path = ROOT / 'build/horizon-finish-baseline.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    inputs = dict(baseline['inputs'])
    assert len(inputs) == 2314 and sha(MODEL) == MODEL_SHA
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Prior input changed: ' + name
    rows, stores, manifest = finish.load_rows()
    targets = sorted(key for key, attempt in stores['prior_h14_attempts'].items()
                     if attempt['status'] == 'budget_rejected')
    h8, h10, h12 = (stores[name] for name in ('h8_references', 'h10_references', 'h12_references'))
    h14 = finish.combined_labels(stores)
    assert len(targets) == 5 and len(rows) == len(h12) == len(h14) == 6
    assert set(h14) == set(rows) and len(stores['prior_h14_labels']) == 1
    helper_path = ROOT / 'tools/analyze-horizon-probe.py'
    helpers = runpy.run_path(str(helper_path))
    analyze_case = helpers['analyze_case']
    for path in (Path(__file__), helper_path, baseline_path,
                 ROOT / 'congklak/horizon_finish.py', ROOT / 'tools/horizon-finish.py',
                 ROOT / 'build/horizon-finish-plan.json', ROOT / 'build/horizon-finish-registry.json',
                 ROOT / 'build/horizon-finish-implementation.json', ROOT / 'build/horizon-finish-result.json',
                 ROOT / 'data/horizon-finish-v1/work.sqlite3', ROOT / 'data/horizon-finish-v1/manifest.json'):
        inputs[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
    attempts = dict(stores['prior_h14_attempts'])
    provenance = {}
    for key in rows:
        if key in stores['prior_h14_labels']:
            provenance[key] = dict(source_kind='prior_complete_H14', source_job='prior_h14',
                                   node_budget=25000000, new_teacher_work=False)
            assert h14[key] == stores['prior_h14_labels'][key]
        else:
            tier = 'tier2' if key in stores['tier2_labels'] else 'tier1'
            assert key in stores[tier + '_labels'] and h14[key] == stores[tier + '_labels'][key]
            attempts[key] = stores[tier + '_attempts'][key]
            provenance[key] = dict(source_kind='larger_budget_complete_H14', source_job=tier,
                                   node_budget=attempts[key]['node_budget'], new_teacher_work=True)
    native = Native()
    cases = []
    with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
        for index, key in enumerate(sorted(rows), 1):
            row = rows[key]
            for depth, labels in ((8, h8), (10, h10), (12, h12), (14, h14)):
                natural_study.validate_label(labels[key], row, depth, native)
            case = analyze_case(key, row, h8, h10, h12, h14,
                stores['prior_h12_attempts'], attempts,
                evaluator.evaluate_many([row['state']])[0], native,
                {12: 10000000, 14: provenance[key]['node_budget']})
            case['case_id'] = 'D' + str(index)
            case['H14_provenance'] = provenance[key]
            case['prior_requested_H14_status'] = stores['prior_h14_attempts'][key]['status']
            case['prior_auxiliary_H13_best_moves'] = stores['prior_h14_attempts'][key]['history'][12]['best_moves']
            case['H13_H14_same_best_set'] = set(case['depth_evidence']['13']['best_moves']) == set(h14[key]['label']['best_moves'])
            for evidence in case['depth_evidence'].values():
                if evidence['source_job'] == 'h14':
                    evidence['source_job'] = provenance[key]['source_job']
            best12 = h12[key]['label']['best_moves']
            report14 = h14[key]['report']
            q14 = {c['pit']: c['depth_result']['value_player0'] for c in report14['candidates']}
            sign = 1 if row['state']['controller'] == 0 else -1
            value14 = report14['depth_result']['value_player0']
            case['H14_regret_of_H12_best_ties'] = [dict(pit=pit, display_pit=pit + 1,
                regret=sign * (value14 - q14[pit])) for pit in best12]
            assert all(item['regret'] >= 0 for item in case['H14_regret_of_H12_best_ties'])
            cases.append(case)
    counts = dict(roots=6, new_H14_complete=sum(key in targets for key in h14),
                  reused_H14_complete=1, H12_complete=6, H14_complete=6,
                  requested_complete_pairs=6, requested_incomplete_cases=0,
                  same_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] for c in cases),
                  changed_best_sets_H12_H14=sum(not c['same_best_set_H12_H14'] for c in cases),
                  same_best_sets_H13_H14=sum(c['H13_H14_same_best_set'] for c in cases))
    assert counts['new_H14_complete'] == 5
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Input changed during comparison: ' + name
    body = dict(schema='fixed-control-horizon-finish-comparison-v1', passed=True,
                fixed_model_sha256=MODEL_SHA, rules_digest=config.RULES_DIGEST,
                counts=counts, cases=cases,
                case_classifications=dict(sorted(Counter(c['classification'] for c in cases).items())),
                inputs=dict(sorted(inputs.items())), training_performed=False, model_changed=False,
                active_sampling_weights_changed=False, data_ingested=False,
                teacher_search_performed_by_analysis=False, new_PUCT_or_strength_tests=False,
                sealed_final_test_consumed=False, convergence_proved=False, full_game_solved=False,
                definitions=dict(pit_numbers='Machine 0..6; display fields 1..7',
                    values='Fixed-player0 seed margin; actor-signed regrets. All six canonical controllers are player1.',
                    same_best_set='Exact set equality at completed finite choice depths, not convergence or game proof.',
                    auxiliary='Actual completed depth from its stated job and node budget; never a substitute for a requested completion.',
                    PV='One reported finite-depth optimal continuation per fixed root action. Stores/captures/bonus facts are replayed; earlier replies can change.',
                    H14_provenance='Five new complete H14 labels plus the one unchanged prior complete H14; old H12 and ownership unchanged.'),
                interpretation='This selected disagreement sample cannot estimate general teacher reliability. All comparisons use fully completed requested H12/H14 labels, '
                    'while retaining odd-depth evidence and full best-move ties. Equal sets at two depths do not prove convergence; bonus choices invalidate an alternating-actor '
                    'interpretation of odd/even depth. Nonterminal leaves retain unresolved play. Model forwards and descriptive line replays perform no teacher searches. '
                    'No diagnostic target is admitted to training by this report.')
    publish(OUTPUT, body)
    print(config.canonical(dict(event='horizon_finish_comparison_complete', passed=True, counts=counts,
        cases=[dict(case_id=c['case_id'], H12=c['depth_evidence']['12']['display_best_moves'],
                    H14=c['depth_evidence']['14']['display_best_moves'],
                    classification=c['classification']) for c in cases])), flush=True)


if __name__ == '__main__':
    main()
