"""Seal a fixed-model ordinary-coverage pilot using executed, source-bound evidence."""
from __future__ import annotations
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

# Read-only reuse of already accepted pinning, AST-coverage and physical journal
# helpers. Neither of the previous recorder's build/verify entry points runs.
_spec = importlib.util.spec_from_file_location('sealed_selfplay_evidence_helpers', ROOT / 'tools/record-selfplay-pilot.py')
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)
require, sha, pin, read, current = evidence.require, evidence.sha, evidence.pin, evidence.read, evidence.current
OUTPUT = ROOT / 'build/ordinary-selfplay-acceptance.json'


def physical_recovery(preflight):
    details = preflight['evidence']
    coverage = set()
    for kind, field, table, order in (
        ('collection', 'committed_families', 'families', 'idx'),
        ('selfplay', 'committed_decisions', 'sp_decisions', 'gameid,choice')):
        records = details['live_' + kind + '_recovery']['tests']
        require(len(records) == 2, 'Actual interruption coverage differs')
        for record in records:
            evidence.physical_marker(record)
            expected = 1 if record['event'].endswith('_precommit') else 2
            coverage.add((kind, expected))
            prefix = evidence.physical_rows(Path(record['probe']) / 'validated-prefix.sqlite3', table, order)
            final = evidence.physical_rows(Path(record['run']) / 'work.sqlite3', table, order)
            require(len(prefix) == record[field] == expected and
                    all(final.get(key) == value for key, value in prefix.items()),
                    'A physical committed prefix changed after recovery')
    require(coverage == {(kind, n) for kind in ('collection', 'selfplay') for n in (1, 2)},
            'Missing actual pre/post commit kill')
    stage = read('build/ordinary-selfplay-audit/collection.json')['evidence']['production_collection']
    backup = stage['backup']
    require(pin(backup) == stage['backup_sha256'] and stage['all512_histories_literal_equal'] is True and
            stage['source_journal_sha_is_historical_not_current_after_rollouts'] is True,
            'Closed collection provenance differs')
    live = 'data/ordinary-selfplay-pilot-v1/work.sqlite3'
    for table, order in (('families', 'idx'), ('roots', 'state')):
        require(evidence.physical_rows(backup, table, order) == evidence.physical_rows(live, table, order),
                'Collection ownership changed during rollouts')
    require(pin('data/ordinary-selfplay-pilot-v1/collection-manifest.json') == stage['collection_manifest_sha256'] and
            pin('data/ordinary-selfplay-pilot-v1/rollout-protection.json') == stage['rollout_protection_file_sha256'],
            'Closed collection or complete validation mask changed')


def build():
    from congklak import ordinary_selfplay_pilot as producer
    evidence.INPUTS.clear()
    for name in ('tools/record-ordinary-selfplay.py', 'tools/prepare-ordinary-selfplay-baseline.py',
                 'tools/verify-ordinary-selfplay-rerun.py', 'tools/verify-ordinary-selfplay-acceptance.py',
                 'tools/verify-ordinary-selfplay-dashboard.py',
                 'tools/record-selfplay-pilot.py'):
        pin(name)
    baseline = read('build/ordinary-selfplay-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 3171,
            'Sealed prior baseline differs')
    current(baseline['inputs'])
    checks, groups = [], {}
    for label, report, source in (
        ('producer_owner', 'build/ordinary-selfplay-author.json', 'tests/ordinary_selfplay_pilot_author.py'),
        ('review_owner', 'build/ordinary-selfplay-review-author.json', 'tests/ordinary_selfplay_review_author.py'),
        ('independent', 'build/ordinary-selfplay-audit.json', 'tests/ordinary_selfplay_independent_audit.py')):
        body, methods = evidence.group(report, source)
        checks.extend(methods)
        groups[label] = body['tests_run']
    require(len(checks) == len(set(checks)), 'Duplicate executed check coverage')
    preflight = read('build/ordinary-selfplay-audit/preflight.json')
    physical_recovery(preflight)
    roots, collection = producer.load_rows()
    manifest, rows = producer.load_replay()
    summary = manifest['summary']
    training = [row for row in roots.values() if row['cohort'] == 'training_candidate']
    validation = [row for row in roots.values() if row['cohort'] == 'validation']
    require(collection['status'] == manifest['status'] == 'complete' and collection['families'] == 512 and
            len(training) <= 24 and len(validation) <= 6 and
            len({r['family'] for r in roots.values()}) == len(roots) and
            summary['games'] == 2 * len(training) and
            summary['terminal_games'] + summary['capped_games'] == summary['games'] and
            summary['retained_rows'] == len(rows), 'Bounded collection/rollout completion differs')
    require(all(row['stratum'].endswith('/ordinary') and sum(map(sum, row['state']['pits'])) >= 25 and
                row['legal_mask'].bit_count() >= 2 for row in rows), 'An ineligible position entered replay')
    protection = read('build/ordinary-selfplay-rollout-protection.json')
    require(all(row['quotient'] not in protection['quotients'] and row['family'] not in protection['families']
                for row in rows), 'Evaluation ownership entered the candidate replay')
    require(summary['future_use_ready'] is
            (summary['usable_start_gate_passed'] and summary['validation_panel_complete']), 'Future-use flags differ')
    require(all(value['training_performed'] is False for value in (collection, manifest, summary)),
            'Data collection claims optimization')
    review = read('build/ordinary-selfplay-prior-arena-review.json')
    weights = read('build/ordinary-selfplay-sampling-review.json')
    for result in (review, weights):
        current(result['inputs'])
        require(result['training_performed'] is result['sampling_applied'] is result['default_promotion'] is False,
                'An offline review applied training or promotion')
    require(weights['passed'] is True and weights['actual_new_optimizer_draws'] == 0 and
            weights['model_changed'] is False and weights['new_teacher_searches'] == weights['new_strength_games'] == 0,
            'Sampling review performed unauthorized computation')
    rerun = read('build/ordinary-selfplay-root-verification.json')
    require(rerun['passed'] is True and rerun['before'] == rerun['after'] and
            rerun['completed_bytes_mtimes_unchanged'] is True and rerun['torch_imported'] is False and
            all(rerun[key] == 0 for key in ('new_collection_commits', 'new_selfplay_decisions',
                                          'new_teacher_searches', 'new_optimizer_steps')),
            'Completed no-work proof differs')
    current(rerun['sources'])
    for name, record in rerun['after'].items():
        path = ROOT / name
        require(pin(name) == record['sha256'] and path.stat().st_size == record['size'] and
                path.stat().st_mtime_ns == record['mtime_ns'], 'Completed durable signature changed')
    for name in ('congklak/ordinary_selfplay_pilot.py', 'tools/ordinary-selfplay-pilot.py',
                 'congklak/ordinary_selfplay_review.py', 'tools/review-ordinary-selfplay-pilot.py',
                 'congklak/ordinary_selfplay_sampling.py', 'tools/review-ordinary-selfplay-sampling.py',
                 'congklak/ordinary_selfplay_progress.py', 'tools/ordinary-selfplay-dashboard.html',
                 'tools/ordinary-selfplay-audit.py', 'docs/ORDINARY_SELFPLAY_SCOPE.md',
                 'docs/ORDINARY_SELFPLAY_PRODUCER.md', 'docs/ORDINARY_SELFPLAY_RESULTS.md',
                 'docs/ORDINARY_SELFPLAY_INDEPENDENT_AUDIT.md', 'build/ordinary-selfplay-rerun.log'):
        pin(name)
    for pattern in ('ordinary-selfplay-*.json',):
        for path in sorted((ROOT / 'build').glob(pattern)):
            if path.name in ('ordinary-selfplay-acceptance.json', 'ordinary-selfplay-acceptance-rerun.json',
                             'ordinary-selfplay-dashboard-final.json'):
                continue
            result = read(path.relative_to(ROOT).as_posix())
            if '-prefreeze-' not in path.name:
                for field in ('inputs', 'sources', 'dependencies'):
                    if isinstance(result.get(field), dict):
                        current(result[field])
    for folder in (producer.DEFAULT, ROOT / 'build/ordinary-selfplay-audit'):
        evidence.durable_tree(folder.relative_to(ROOT).as_posix())
    for pattern in ('ordinary-selfplay-*author*', 'ordinary-selfplay-prefreeze-*'):
        for path in sorted((ROOT / 'build').glob(pattern)):
            if path.is_dir():
                evidence.durable_tree(path.relative_to(ROOT).as_posix())
            elif path.is_file():
                pin(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'build/ordinary-selfplay-progress').glob('run-*.json')):
        run = json.loads(path.read_text(encoding='utf-8'))
        require(run['state'] != 'running', 'A producer or audit is still active')
        pin(path.relative_to(ROOT).as_posix())
        pin(run['log'])
    require(pin(baseline['control_model']) == baseline['control_model_sha256'] and
            pin(baseline['challenger_model']) == baseline['challenger_model_sha256'], 'A model changed')
    body = dict(schema='ordinary-selfplay-acceptance-v1', passed=True, tests_run=len(checks), groups=groups,
                covered_checks=checks, errors=0, failures=0, skips=0, independent_complete=True,
                prior_inputs_preserved=len(baseline['inputs']), families=512,
                selected_training_roots=len(training), selected_validation_roots=len(validation),
                replay=summary, actual_process_kills=4, models_fixed=True, training_performed=False,
                new_teacher_searches=0, new_strength_games=0, sampling_applied=False,
                default_promotion=False, source_data_pruned=False, completed_no_work=True,
                inputs=dict(sorted(evidence.INPUTS.items())))
    current(body['inputs'])
    if OUTPUT.exists():
        require(json.loads(OUTPUT.read_text(encoding='utf-8')) == body, 'Immutable acceptance differs')
    else:
        atomic_json(OUTPUT, body)
    print(config.canonical(dict(event='ordinary_selfplay_accepted', passed=True, tests_run=len(checks),
                               inputs=len(body['inputs']), usable_start_roots=summary['usable_start_roots'],
                               future_use_ready=summary['future_use_ready'], training_performed=False)), flush=True)


def verify():
    accepted = read('build/ordinary-selfplay-acceptance.json')
    require(accepted['passed'] is True and accepted['training_performed'] is False and
            accepted['default_promotion'] is False, 'Invalid ordinary data-only acceptance')
    current(accepted['inputs'])
    print(config.canonical(dict(event='ordinary_selfplay_acceptance_verified', passed=True,
                               inputs=len(accepted['inputs']), tests_run=accepted['tests_run'])), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'verify'))
    args = parser.parse_args()
    build() if args.operation == 'build' else verify()
