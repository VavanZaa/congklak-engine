"""Describe new H8 labels using the unchanged retained control; no training/search."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, pilot
from congklak.native import Native, flatten
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.deeper_teacher_v2 import validate_label, validate_link

MODEL = ROOT / 'data/quality-train-v1/model.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve(path, value):
    """A completed report is immutable; repeated runs must derive identical bytes."""
    path = Path(path)
    text = config.canonical(value) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, 'Completed report differs: ' + str(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        from congklak.deeper_teacher_v2 import atomic_json
        atomic_json(path, value)


def summary(rows):
    saved = [r for r in rows if r['saved_puct_teacher_regret'] is not None]
    return dict(rows=len(rows), families=len({r['family'] for r in rows}),
        single_legal_action_rows=sum(r['legal_action_count'] == 1 for r in rows),
        value_mae_seeds=statistics.mean(r['value_absolute_error'] for r in rows) if rows else None,
        signed_actor_error=statistics.mean(r['signed_actor_error'] for r in rows) if rows else None,
        policy_teacher_regret=statistics.mean(r['policy_teacher_regret'] for r in rows) if rows else None,
        policy_teacher_best_fraction=statistics.mean(r['policy_teacher_regret'] == 0 for r in rows) if rows else None,
        saved_puct_rows=len(saved),
        saved_puct_teacher_regret=statistics.mean(r['saved_puct_teacher_regret'] for r in saved) if saved else None,
        saved_puct_teacher_disagreements=sum(r['saved_puct_teacher_regret'] > 0 for r in saved))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'data/ordinary-data-pilot-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/data-pilot-model-diagnosis.json')
    args = parser.parse_args()
    assert sha(MODEL) == MODEL_SHA
    baseline = json.loads((ROOT / 'build/data-pilot-baseline.json').read_text())
    for name, digest in baseline['inputs'].items():
        assert sha(ROOT / name) == digest, 'Prior input changed: ' + name
    from congklak import data_pilot
    roots, labels, attempts, manifest = data_pilot.load_rows(args.run)
    assert len(attempts) == len(roots), 'Pilot is unfinished'
    assert set(labels) <= set(roots)
    inputs = dict(baseline['inputs'])
    for path in [Path(__file__), ROOT / 'congklak/data_pilot.py',
                 ROOT / 'build/data-pilot-baseline.json', ROOT / 'build/data-pilot-plan.json',
                 ROOT / 'build/data-pilot-implementation.json', ROOT / 'build/data-pilot-registry.json',
                 args.run / 'work.sqlite3', args.run / 'manifest.json']:
        inputs[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
    native = Native()
    metrics = []
    with CpuBatchEvaluator(MODEL, cache_capacity=0, max_batch=1) as evaluator:
        for key in sorted(labels):
            row, entry = roots[key], labels[key]
            p = config.position(row['state'])
            assert entry['root_sha256'] == config.digest(row)
            assert pilot.canonicalize(p, native)[0].hex() == key
            assert pilot._mechanism(p, native)[1] == row['stratum']
            assert row['stratum'].endswith('/ordinary') and sum(map(sum, p['pits'])) >= 25
            label = entry['label']
            validate_label(label, p, row['legal_mask'], 8, native)
            validate_link(entry, row, native)
            legal = [q['pit'] for q in label['action_scores']]
            q = {q['pit']: q['value_player0'] for q in label['action_scores']}
            pred = evaluator.evaluate_many([p])[0]
            chosen = max(legal, key=lambda a: (pred['policy_logits'][a], -a))
            target = label['target_player0']
            sign = 1 if p['controller'] == 0 else -1
            saved = row['provenance'].get('saved_puct_chosen')
            assert saved is None or saved in legal
            regret = sign * (target - q[chosen])
            saved_regret = sign * (target - q[saved]) if saved is not None else None
            assert regret >= 0 and (saved_regret is None or saved_regret >= 0)
            metrics.append(dict(state_key=key, family=row['family'], cohort=row['cohort'],
                provenance_source=row['provenance']['source'],
                legal_action_count=len(legal),
                stratum=row['stratum'], state=p, approximate_H8_target_player0=target,
                control_value_player0=pred['value_player0'],
                value_absolute_error=abs(pred['value_player0'] - target),
                signed_actor_error=sign * (pred['value_player0'] - target),
                policy_chosen=chosen, policy_teacher_regret=regret,
                saved_puct_chosen=saved, saved_puct_teacher_regret=saved_regret,
                teacher_best_moves=label['best_moves'], teacher_action_scores=label['action_scores']))
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row['cohort'] + '/' + row['stratum']].append(row)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Input changed during diagnosis: ' + name
    assert sha(MODEL) == MODEL_SHA
    result = dict(schema='fixed-control-data-pilot-diagnosis-v1', passed=True,
        model_file_sha256=MODEL_SHA, rules_digest=config.RULES_DIGEST,
        selected_roots=len(roots), completed_H8_labels=len(labels),
        rejected_label_attempts=len(attempts) - len(labels),
        cohorts={name: summary([r for r in metrics if r['cohort'] == name])
                 for name in ('diagnostic', 'training_candidate', 'validation')},
        choice_bearing_cohorts={name: summary([r for r in metrics if r['cohort'] == name and r['legal_action_count'] >= 2])
                 for name in ('diagnostic', 'training_candidate', 'validation')},
        strata={name: summary(rows) for name, rows in sorted(grouped.items())},
        by_source={source: summary([r for r in metrics if r['provenance_source'] == source])
                   for source in sorted({r['provenance_source'] for r in metrics})},
        metrics=metrics, inputs=inputs,
        training_performed=False, model_changed=False, new_search_performed=False,
        active_sampling_weights_changed=False, sealed_final_test_consumed=False,
        independent_unseen_release_test=False, strength_improvement_claimed=False,
        interpretation='Descriptive frozen-model errors against approximate full-root H8 H_v1 labels. '
          'Saved PUCT choices use their existing original clocks; no new equal-time games. '
          'Diagnostic trajectories remain excluded from future training. Fresh validation labels '
          'are consumed development evidence, not a sealed release test. Correlated roots and '
          'budget-rejected labels limit inference; teacher disagreement is not proven minimax error.')
    preserve(args.output, result)
    print(config.canonical(dict(event='data_pilot_model_diagnosis', passed=True,
        labels=len(labels), cohorts=result['cohorts'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
