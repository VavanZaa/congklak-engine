"""Seal source-bound, executed saved-data value-review evidence."""
import ast
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.ordinary_value_review_common import baseline, current, pins, publish, read, require

OUTPUT = ROOT / 'build/ordinary-value-review-acceptance.json'


def executed(report_name, source_name):
    body = read(ROOT / report_name)
    require(body['passed'] is True and all(body[key] == 0 for key in ('errors', 'failures', 'skips')),
            'Executed suite failed: ' + report_name)
    ids = body.get('covered_checks', body['covered_methods'])
    require(len(ids) == len(set(ids)) == body['tests_run'] > 0, 'Executed method coverage differs')
    tree = ast.parse((ROOT / source_name).read_text(encoding='utf-8'))
    expected = {node.name + '.' + method.name for node in tree.body if isinstance(node, ast.ClassDef)
                for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name.startswith('test_')}
    actual = {'.'.join(name.split('.')[-2:]) for name in ids}
    require(actual == expected, 'Test source/coverage mismatch: ' + source_name)
    current(body['inputs'])
    require(body['inputs'].get(source_name) == pins([source_name])[source_name], 'Executed source was not pinned')
    return {**body, 'executed_ids': ids}


def build():
    frozen = baseline(verify_all=True)
    inputs = dict(frozen['inputs'])
    groups = {}
    checks = []
    suites = (
        ('arena_owner', 'build/ordinary-value-loss-review-author.json', 'tests/ordinary_value_loss_review_author.py'),
        ('validation_owner', 'build/ordinary-value-validation-review-author.json', 'tests/ordinary_value_validation_review_author.py'),
        ('progress_owner', 'build/ordinary-value-review-progress-author.json', 'tests/ordinary_value_review_progress_author.py'),
        ('independent', 'build/ordinary-value-review-audit.json', 'tests/ordinary_value_review_independent_audit.py'))
    for name, report_name, source_name in suites:
        body = executed(report_name, source_name)
        groups[name] = body['tests_run']
        checks.extend(source_name + '::' + '.'.join(method.split('.')[-2:]) for method in body['executed_ids'])
        inputs.update(body['inputs'])
    require(len(checks) == len(set(checks)), 'Duplicate counted test methods')
    reports = []
    for name in ('build/ordinary-value-overlap-review.json', 'build/ordinary-value-loss-review.json',
                 'build/ordinary-value-validation-review.json', 'build/ordinary-value-review-conclusions.json'):
        body = read(ROOT / name); require(body['passed'] is True, 'Incomplete review: ' + name)
        current(body['inputs']); inputs.update(body['inputs']); reports.append(name)
    result = read(ROOT / reports[-1])
    require(result['decision']['current_allocation'] == 'hold' and
            result['decision']['further_training_authorized'] is result['decision']['new_data_collection_authorized'] is
            result['decision']['new_model_selected'] is result['decision']['candidate_promoted'] is False,
            'Review changed model/data authorization')
    rerun = read(ROOT / 'build/ordinary-value-review-rerun.json')
    require(rerun['passed'] is True and rerun['before'] == rerun['after'] and
            rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved'] == 3779 and
            rerun['reports_rebuilt_and_verified'] == 4 and rerun['torch_imported'] is False and
            all(rerun[key] == 0 for key in ('new_training_updates','new_teacher_searches','new_PUCT_searches','new_games')),
            'Completed no-work proof differs')
    current(rerun['sources'])
    for name, record in rerun['after'].items():
        path = ROOT / name
        require(pins([name])[name] == record['sha256'] and path.stat().st_size == record['size'] and path.stat().st_mtime_ns == record['mtime_ns'],
                'Completed artifact signature changed: ' + name)
    names = ['tools/prepare-ordinary-value-review-baseline.py', 'tools/verify-ordinary-value-review-rerun.py',
             'tools/record-ordinary-value-review.py', 'tools/verify-ordinary-value-review-acceptance.py',
             'tools/verify-ordinary-value-review-dashboard.py', 'congklak/ordinary_value_review_progress.py',
             'tools/ordinary-value-review-dashboard.html', 'docs/ORDINARY_VALUE_REVIEW_SCOPE.md',
             'docs/ORDINARY_VALUE_REVIEW_RESULTS.md', 'docs/ORDINARY_VALUE_REVIEW_INDEPENDENT_AUDIT.md',
             'docs/ORDINARY_VALUE_LOSS_REVIEW.md', 'docs/ORDINARY_VALUE_VALIDATION_REVIEW.md']
    for pattern in ('ordinary-value-review-*.json', 'ordinary-value-overlap-review*.json',
                    'ordinary-value-loss-review*.json', 'ordinary-value-validation-review*.json'):
        names += [path.relative_to(ROOT).as_posix() for path in (ROOT / 'build').glob(pattern)
                  if path.name not in ('ordinary-value-review-acceptance.json', 'ordinary-value-review-acceptance-rerun.json',
                                       'ordinary-value-review-dashboard-final.json')]
    folders = set((ROOT / 'build').glob('ordinary-value-*-review*')) | set((ROOT / 'build').glob('ordinary-value-review*'))
    for path in sorted(folders):
        if path.is_dir() and path.name != 'ordinary-value-review-progress':
            names += [p.relative_to(ROOT).as_posix() for p in path.rglob('*') if p.is_file() and p.suffix not in ('.pyc',)]
        elif path.is_file() and path.name not in ('ordinary-value-review-acceptance.json',
                    'ordinary-value-review-acceptance-rerun.json','ordinary-value-review-dashboard-final.json'):
            names.append(path.relative_to(ROOT).as_posix())
    for path in (ROOT / 'build/ordinary-value-review-progress').glob('run-*.json'):
        row = read(path); require(row['state'] != 'running', 'A review worker is active')
        names.extend([path.relative_to(ROOT).as_posix(), row['log']])
    inputs.update(pins(names))
    current(inputs)
    body = dict(schema='ordinary-value-review-acceptance-v1', passed=True, tests_run=len(checks), covered_checks=checks,
                groups=groups, errors=0, failures=0, skips=0, independent_complete=True,
                old_inputs_preserved=3779, reports=reports, facts=result['facts'], policy_fixed=True, all_model_weights_fixed=True,
                original_allocation='hold', current_allocation='hold', training_performed=False,
                new_teacher_searches=0, new_PUCT_searches=0, new_games=0, dataset_changed=False,
                default_promotion=False, completed_no_work=True, inputs=dict(sorted(inputs.items())))
    publish(OUTPUT, body)
    return body


def verify():
    body = read(OUTPUT)
    require(body['passed'] is body['policy_fixed'] is body['all_model_weights_fixed'] is True and
            body['training_performed'] is body['default_promotion'] is False, 'Invalid review acceptance')
    current(body['inputs'])
    return body


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('operation', choices=('build', 'verify'))
    args = parser.parse_args(); body = build() if args.operation == 'build' else verify()
    print(config.canonical(dict(event='ordinary_value_review_accepted_' + args.operation, passed=True,
                               tests_run=body['tests_run'], inputs=len(body['inputs']))), flush=True)
