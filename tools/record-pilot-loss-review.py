"""Seal the existing-data diagnosis after author checks and independent review."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as data

OUTPUT = ROOT / 'build/pilot-loss-review-acceptance.json'
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing review input: ' + str(path))
    key = path.relative_to(ROOT).as_posix()
    INPUTS[key] = data.sha(path)
    return INPUTS[key]


def load(name):
    pin(name)
    return data.read(ROOT / name)


def current(mapping):
    for name, digest in mapping.items():
        require(pin(name) == digest, 'Review input changed: ' + name)


def executed(path, source):
    body = load(path)
    require(body['passed'] is True and body['errors'] == body['failures'] == body['skips'] == 0,
            'Executed review group failed: ' + path)
    pin(source)
    tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
    actual = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    require(node.name not in actual, 'Duplicate review test method')
                    actual[node.name] = source + '::' + cls.name + '::' + node.name
    names = body.get('covered_methods', body.get('methods'))
    require(names and len(names) == len(set(names)) == body['tests_run'] and set(names) == set(actual),
            'Executed/source method coverage differs: ' + path)
    current(body['inputs'])
    return body, [actual[name] for name in names]


def build():
    INPUTS.clear()
    pin('tools/record-pilot-loss-review.py')
    pin('tools/prepare-pilot-loss-review-baseline.py')
    baseline = load('build/pilot-loss-review-baseline.json')
    require(baseline['passed'] and len(baseline['inputs']) == baseline['inputs_count'] == 2600,
            'Prior preserved baseline incomplete')
    current(baseline['inputs'])
    ordinary = load('build/pilot-loss-review-ordinary.json')
    arena = load('build/pilot-loss-review-arena.json')
    readiness = load('build/engine-readiness-current.json')
    for report in (ordinary, arena, readiness):
        require(report['passed'] is True, 'Diagnostic report failed')
        current(report['inputs'])
    require(ordinary['schema'] == 'pilot-ordinary-error-review-v1' and
            arena['schema'] == 'pilot-loss-review-arena-v1' and
            readiness['schema'] == 'congklak-current-readiness-v1', 'Review schema differs')
    require(all(ordinary[name] is True for name in ('no_new_games', 'no_teacher_searches', 'no_training',
            'no_torch', 'no_weight_or_target_changes')), 'Ordinary review changed scope')
    require(arena['new_games'] == arena['new_searches'] == arena['new_teacher_labels'] == 0 and
            all(arena[name] is False for name in ('training_performed', 'model_changed', 'policy_changed',
                'data_ingested', 'default_promotion', 'sealed_final_test_consumed', 'torch_imported')),
            'Arena review changed scope')
    require(all(readiness[name] is False for name in ('new_training', 'default_promotion')) and
            readiness['new_games'] == readiness['new_teacher_labels'] == 0 and
            readiness['recommendations_are_unexecuted'] is True, 'Readiness started new work')
    require(ordinary['frozen_policy']['all_six_tensors_equal'] is True and
            ordinary['frozen_policy']['all342_native_policy_logits_equal'] is True,
            'Policy/trunk identity differs')
    require({name: len(group['rows']) for name, group in ordinary['groups'].items()} ==
            dict(ordinary_validation=14, ordinary_training=61, protected_diagnostic=8, base_ordinary_development=259),
            'Ordinary cohort coverage differs')
    require(ordinary['sampling']['draws'] == 51200 and ordinary['sampling']['unique_roots_drawn'] == 7616 and
            ordinary['sampling']['pools']['ordinary']['draws'] == 2512,
            'Existing draw exposure differs')
    require(arena['counts']['families'] == 12 and arena['counts']['matched_seats'] == 24 and
            arena['counts']['replayed_games'] == 72 and arena['counts']['literal_choices'] == 2460,
            'Saved trace coverage differs')
    require(readiness['larger_data']['engineering'] == 'ready_for_bounded_versioned_collection' and
            readiness['larger_data']['learning_evidence'] == 'not_ready_for_blind_100k_or_long_training' and
            readiness['product']['finished_strong_analyzer'] == 'not_release_ready' and
            readiness['product']['default'] == 'H_v1' and readiness['both_baseline_strength_gate_passed'] is False,
            'Readiness recommendation changed')
    author, author_ids = executed('build/pilot-loss-review-ordinary-author.json', 'tests/pilot_loss_review_ordinary_acceptance.py')
    independent, independent_ids = executed('build/pilot-loss-review-audit.json', 'tests/pilot_loss_review_independent_audit.py')
    rerun = load('build/pilot-loss-review-root-verification.json')
    require(rerun['passed'] is True and rerun['prior_inputs_preserved'] == 2600 and
            rerun['before'] == rerun['after'] and rerun['completed_bytes_mtimes_unchanged'] is True and
            all(rerun[name] == 0 for name in ('new_native_searches', 'new_PUCT_searches',
                'new_training_runtime_activations', 'new_training_calls', 'new_benchmark_decisions')) and
            rerun['torch_imported'] is False, 'Completed read-only rerun differs')
    current(rerun['sources'])
    for name, signature in rerun['after'].items():
        path = ROOT / name
        require(pin(name) == signature['sha256'] and path.stat().st_size == signature['size'] and
                path.stat().st_mtime_ns == signature['mtime_ns'], 'Durable review artifact changed: ' + name)
    for name in ('congklak/pilot_loss_review_progress.py', 'tools/pilot-loss-review-dashboard.html',
                 'docs/PILOT_ORDINARY_ERRORS.md', 'docs/PILOT_LOSS_REVIEW_RESULTS.md',
                 'docs/PILOT_LOSS_REVIEW_INDEPENDENT_AUDIT.md', 'docs/ENGINE_READINESS_CURRENT.md',
                 'build/pilot-loss-review-rerun.log'):
        pin(name)
    retained = []
    for pattern in ('pilot-loss-review-*diagnostic*', 'pilot-loss-review-*prefreeze*', 'pilot-loss-review-*failed*',
                    'pilot-loss-review-*failure*', 'engine-readiness-prefreeze*'):
        for path in sorted((ROOT / 'build').glob(pattern)):
            if path.is_file():
                name = path.relative_to(ROOT).as_posix()
                pin(name)
                retained.append(name)
    ids = author_ids + independent_ids
    require(len(ids) == len(set(ids)), 'Duplicate aggregate checks')
    require('torch' not in sys.modules, 'Review acceptance imported Torch')
    current(dict(INPUTS))
    summary = ordinary['groups']['ordinary_validation']['summary']['all']
    return dict(schema='pilot-loss-review-acceptance-v1', passed=True, tests_run=len(ids),
        failures=0, errors=0, skips=0, covered_checks=ids,
        groups=dict(ordinary_author=len(author_ids), independent=len(independent_ids)),
        prior_inputs_preserved=2600, ordinary_rows=342, replayed_H8_games=72, literal_choices=2460,
        control_model_sha256=pin('data/quality-train-v1/model.json'),
        pilot_model_sha256=pin('data/value-pilot-v1/model.json'), policy_and_trunk_fixed=True,
        ordinary_validation_delta_MAE=summary['delta_mae_seeds'],
        H8_pilot_minus_control_margin=arena['aggregate']['pilot_minus_control_paired_mean_margin'],
        larger_data=readiness['larger_data'], product=readiness['product'],
        completed_bytes_mtimes_unchanged=True, new_training=0, new_games=0,
        new_searches=0, new_teacher_labels=0, source_data_pruned=False,
        sealed_final_test_consumed=False, default_promotion=False,
        independent_complete=True, retained_diagnostics=sorted(set(retained)),
        scope='Existing-data diagnostic and readiness assessment; no new experiments or causal blunder certification.',
        inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    body = build()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(body) + '\n', 'Immutable acceptance differs')
    if not args.check_only:
        data.publish(OUTPUT, body)
    print(config.canonical(dict(event='pilot_loss_review_acceptance_checked' if args.check_only else 'pilot_loss_review_accepted',
        passed=True, tests_run=body['tests_run'], inputs=len(INPUTS), acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    main()
