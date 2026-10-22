"""Compare complete paired H8/H10 targets with the unchanged control model."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, pilot, reports
from congklak.native import Native, flatten
from congklak.deeper_teacher_v2 import validate_label, atomic_json
from congklak.puct_cpu_inference import CpuBatchEvaluator

MODEL = ROOT / 'data/quality-train-v1/model.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
MEANS = ('value_mae_h8', 'value_mae_h10', 'value_mae_change_h10_minus_h8',
         'policy_regret_h8', 'policy_regret_h10', 'policy_regret_change_h10_minus_h8',
         'absolute_target_shift', 'signed_actor_target_shift', 'H10_regret_H8_chosen',
         'H10_regret_H8_tie_min', 'H10_regret_H8_tie_max', 'action_Q_shift_mae')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, body):
    text = config.canonical(body) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Completed stability report differs'
    else:
        atomic_json(path, body)


def check_label(item, row, depth, native):
    assert item['root_sha256'] == row['source_root_sha256'] == config.digest(row['root'])
    p, label, report = row['state'], item['label'], item['report']
    validate_label(label, p, row['legal_mask'], depth, native)
    reports.validate(report, p, config.identity(native.build_id), native)
    assert report['choice_depth'] == depth
    assert label['target_player0'] == report['depth_result']['value_player0']
    assert label['best_moves'] == report['best_moves']
    assert label['action_scores'] == [dict(pit=q['pit'], value_player0=q['depth_result']['value_player0']) for q in report['candidates']]
    assert label['coverage'] == report['coverage']
    assert label['work'] == {k: report[k] for k in ('nodes', 'deposits', 'elapsed_ms', 'peak_process_bytes')}


def summarize(rows):
    pairs = [r for r in rows if r['complete_pair']]
    choices = [r for r in pairs if r['legal_action_count'] >= 2]
    return dict(selected_roots=len(rows), families=len({r['family'] for r in rows}),
        h8_complete=sum(r['h8_complete'] for r in rows), h10_complete=sum(r['h10_complete'] for r in rows),
        complete_pairs=len(pairs), unpaired_roots=len(rows) - len(pairs),
        choice_bearing_pairs=len(choices), single_legal_action_pairs=len(pairs) - len(choices),
        paired_means={k: statistics.mean(r[k] for r in pairs) if pairs else None for k in MEANS},
        paired_choice_bearing_means={k: statistics.mean(r[k] for r in choices) if choices else None for k in MEANS},
        best_sets_changed=sum(r['best_sets_changed'] for r in pairs),
        best_sets_disjoint=sum(r['best_sets_disjoint'] for r in pairs),
        H8_choice_still_best=sum(r['H8_choice_still_best'] for r in pairs),
        all_H8_ties_still_best=sum(r['all_H8_ties_still_best'] for r in pairs),
        model_MAE_lower_at_H10=sum(r['value_mae_change_h10_minus_h8'] < 0 for r in pairs),
        model_MAE_higher_at_H10=sum(r['value_mae_change_h10_minus_h8'] > 0 for r in pairs),
        model_MAE_equal_at_H10=sum(r['value_mae_change_h10_minus_h8'] == 0 for r in pairs),
        unpaired_statuses=dict(sorted(Counter(r['h8_status'] + '/' + r['h10_status'] for r in rows if not r['complete_pair']).items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'data/natural-study-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/natural-study-stability.json')
    args = parser.parse_args()
    assert args.run.resolve() == (ROOT / 'data/natural-study-v1').resolve(), 'Use the declared production study'
    baseline_path = ROOT / 'build/natural-study-baseline.json'
    baseline = json.loads(baseline_path.read_text())
    inputs = dict(baseline['inputs'])
    assert sha(MODEL) == MODEL_SHA
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Prior input changed: ' + name
    from congklak import natural_study
    rows, h8, h10, a8, a10, manifest = natural_study.load_rows(args.run)
    assert manifest is not None and set(a8) == set(a10) == set(rows), 'Study is incomplete'
    assert set(h8) <= set(rows) and set(h10) <= set(rows)
    for path in (Path(__file__), ROOT / 'congklak/natural_study.py', baseline_path,
                 ROOT / 'build/natural-study-plan.json', ROOT / 'build/natural-study-registry.json',
                 ROOT / 'build/natural-study-implementation.json', ROOT / 'build/natural-study-result.json',
                 args.run / 'work.sqlite3', args.run / 'manifest.json'):
        inputs[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
    metrics = []
    native = Native()
    with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
        for key in sorted(rows):
            row = rows[key]
            p = config.position(row['state'])
            assert row['study_role'] in ('fresh', 'followup')
            assert row['root']['state'] == p and row['root']['cohort'] == row['cohort']
            assert row['root']['family'] == row['family']
            assert pilot.canonicalize(p, native)[0].hex() == key
            active, stratum = pilot._mechanism(p, native)
            assert 50 <= active <= 74 and stratum == row['stratum'] == '50-74/ordinary'
            normal = native.normalize(flatten(p), actor=p['controller'], total=98)
            legal = normal['legal_actions']
            assert row['legal_mask'] == sum(1 << a for a in legal)
            for depth, targets in ((8, h8), (10, h10)):
                if key in targets:
                    check_label(targets[key], row, depth, native)
            prediction = evaluator.evaluate_many([p])[0]
            chosen = max(legal, key=lambda a: (prediction['policy_logits'][a], -a))
            paired = key in h8 and key in h10
            item = dict(state_key=key, state=p, family=row['family'], cohort=row['cohort'],
                study_role=row['study_role'], legal_action_count=len(legal),
                h8_complete=key in h8, h10_complete=key in h10, complete_pair=paired,
                h8_status=a8[key]['status'], h10_status=a10[key]['status'],
                model_value_player0=prediction['value_player0'], policy_chosen=chosen,
                **{k: None for k in MEANS},
                best_sets_changed=None, best_sets_disjoint=None, H8_choice_still_best=None,
                all_H8_ties_still_best=None)
            if paired:
                l8, l10 = h8[key]['label'], h10[key]['label']
                q8 = {q['pit']: q['value_player0'] for q in l8['action_scores']}
                q10 = {q['pit']: q['value_player0'] for q in l10['action_scores']}
                assert list(q8) == list(q10) == legal
                v8, v10 = l8['target_player0'], l10['target_player0']
                b8, b10 = l8['best_moves'], l10['best_moves']
                sign = 1 if p['controller'] == 0 else -1
                error8, error10 = abs(prediction['value_player0'] - v8), abs(prediction['value_player0'] - v10)
                regret8, regret10 = sign * (v8 - q8[chosen]), sign * (v10 - q10[chosen])
                ties = [sign * (v10 - q10[a]) for a in b8]
                assert min(regret8, regret10, *ties) >= 0
                item.update(value_mae_h8=error8, value_mae_h10=error10,
                    value_mae_change_h10_minus_h8=error10 - error8,
                    policy_regret_h8=regret8, policy_regret_h10=regret10,
                    policy_regret_change_h10_minus_h8=regret10 - regret8,
                    absolute_target_shift=abs(v10 - v8), signed_actor_target_shift=sign * (v10 - v8),
                    H10_regret_H8_chosen=sign * (v10 - q10[min(b8)]),
                    H10_regret_H8_tie_min=min(ties), H10_regret_H8_tie_max=max(ties),
                    action_Q_shift_mae=statistics.mean(abs(q10[a] - q8[a]) for a in legal),
                    best_sets_changed=set(b8) != set(b10), best_sets_disjoint=not bool(set(b8) & set(b10)),
                    H8_choice_still_best=min(b8) in b10, all_H8_ties_still_best=set(b8) <= set(b10),
                    H8_target_player0=v8, H10_target_player0=v10,
                    H8_best_moves=b8, H10_best_moves=b10,
                    H8_action_scores=l8['action_scores'], H10_action_scores=l10['action_scores'])
            metrics.append(item)
    grouped = defaultdict(list)
    for item in metrics:
        grouped[item['study_role'] + '/' + item['cohort']].append(item)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Input changed during interpretation: ' + name
    assert sha(MODEL) == MODEL_SHA
    pairs = set(h8) & set(h10)
    counts = dict(selected_roots=len(rows), h8_complete=len(h8), h10_complete=len(h10),
        complete_pairs=len(pairs), unpaired_roots=len(rows) - len(pairs),
        h8_rejected=sum(a['status'] == 'budget_rejected' for a in a8.values()),
        h10_rejected=sum(a['status'] == 'budget_rejected' for a in a10.values()),
        old_h8_references=sum(r['study_role'] == 'followup' for r in rows.values()),
        new_h8_labels=sum(rows[key]['study_role'] == 'fresh' for key in h8), new_h10_labels=len(h10))
    result = dict(schema='fixed-control-H8-H10-stability-v1', passed=True, counts=counts,
        fixed_model_sha256=MODEL_SHA, rules_digest=config.RULES_DIGEST,
        overall=summarize(metrics), groups={k: summarize(v) for k, v in sorted(grouped.items())},
        metrics=metrics, inputs=dict(sorted(inputs.items())),
        comparison_population='Identical complete-H8 AND complete-H10 root intersection only; rejected/unpaired roots have null comparative metrics and explicit status counts',
        definitions=dict(value_mae_change='MAE against H10 minus MAE against H8, with the SAME frozen prediction and SAME roots',
            policy_regret='Actor-signed difference between teacher-best root value and the action selected by the frozen raw policy',
            H10_regret_H8_chosen='H10 actor-signed regret of lexicographically first H8-best pit; tie min/max also reported',
            action_Q_shift_mae='Mean absolute H10-H8 action-score shift within each paired root, then equal-root mean',
            best_sets_changed='Exact best-pit set inequality, including tie changes; disjoint sets reported separately'),
        training_performed=False, model_changed=False, active_sampling_weights_changed=False,
        new_PUCT_or_strength_tests=False, strength_improvement_claimed=False,
        sealed_final_test_consumed=False, independent_unseen_release_test=False,
        interpretation='H8 and H10 are finite-choice H_v1 teacher estimates. H10 is not game-theoretic truth; '
            'deeper-target disagreement and changes in measured model error do not establish a model improvement or causal defect. '
            'A bounded curated pool, common history prefixes, approximate targets and completion censoring limit inference. '
            'Original follow-up ownership is preserved; protected diagnostics are not training candidates. '
            'Previously consumed validation and new validation are development evidence, not an untouched release test.')
    publish(args.output, result)
    print(config.canonical(dict(event='natural_study_stability_complete', passed=True,
        counts=counts, groups=result['groups'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
