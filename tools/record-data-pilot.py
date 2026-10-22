"""Seal complete ordinary-data pilot evidence without training or new teacher searches."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, data_pilot as pilot
from congklak.native import Native

INPUTS = {}
OUTPUT = ROOT / 'build/data-pilot-acceptance.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing or external acceptance input: ' + str(path))
    name = path.relative_to(ROOT).as_posix()
    value = sha(path)
    require(name not in INPUTS or INPUTS[name] == value, 'Input changed during acceptance: ' + name)
    INPUTS[name] = value
    return value


def load(path):
    pin(path)
    return json.loads((ROOT / path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def current(mapping):
    require(isinstance(mapping, dict) and mapping, 'Missing immutable evidence inputs')
    for path, digest in mapping.items():
        require(isinstance(digest, str) and len(digest) == 64 and pin(path) == digest, 'Evidence input changed: ' + path)


def method_ids(path):
    pin(path)
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    pairs = [(node.name, item.name) for node in tree.body if isinstance(node, ast.ClassDef)
             for item in node.body if isinstance(item, ast.FunctionDef) and item.name.startswith('test_')]
    require(len(pairs) == len(set(pairs)) == len({name for _, name in pairs}), 'Duplicate source methods: ' + path)
    return {name: path + '::' + cls + '.' + name for cls, name in pairs}


def group(path, source, count, key):
    report = load(path)
    current(report['inputs'])
    require(report['passed'] is True and report['tests_run'] == count and
            report['failures'] == report['errors'] == report['skips'] == 0, 'Failed or incomplete checks: ' + path)
    names = report[key]
    methods = method_ids(source)
    require(len(names) == len(set(names)) == count and set(names) <= set(methods), 'Executed source coverage differs: ' + path)
    require(report['inputs'].get(source) == pin(source), 'Checks do not bind executed source: ' + path)
    return report, [methods[name] for name in names]


def stable_artifacts(mapping):
    require(isinstance(mapping, dict) and mapping, 'No completed artifact stability coverage')
    for path, row in mapping.items():
        require(pin(path) == row['sha256'] and (ROOT / path).stat().st_mtime_ns == row['mtime_ns'],
                'Completed artifact bytes or mtime changed: ' + path)
        if 'size' in row:
            require((ROOT / path).stat().st_size == row['size'], 'Completed artifact size differs: ' + path)


def recovery(report, name, table, events):
    evidence = report['evidence'][name]
    require(evidence['actual_process_kills'] is True and evidence['raw_journal_backup_markers_retained'] is True,
            'Actual retained process kills are missing: ' + name)
    records = evidence['tests']
    require(len(records) == 2 and {row['kill_event'] for row in records} == set(events), 'Recovery barrier coverage differs: ' + name)
    folder = 'collection' if table == 'families' else 'labeling'
    base = Path(report['retained_directory']) / folder
    for row in records:
        event = row['kill_event']
        expected_prefix = 1 if event.endswith('precommit') else 2
        require(row['committed_prefix'] == expected_prefix and row['completed_bytes_mtimes_unchanged'] is True,
                'Recovery prefix or stable completion differs: ' + name)
        marker = base / (event + '-probe') / 'marker.json'
        require(load(marker) == row['marker'] and row['marker']['event'] == event, 'Actual kill marker differs')
        database = base / (event + '-probe') / 'validated-prefix.sqlite3'
        pin(database)
        with closing(sqlite3.connect((ROOT / database).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            require(db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] == expected_prefix,
                    'Retained killed-process committed prefix differs')
        flags = ('exact_fresh_histories_and_selection', 'prefix_unchanged') if table == 'families' else (
            'fresh_complete_H8_semantics_and_counters_equal', 'prefix_exactly_retained')
        require(all(row[flag] is True for flag in flags), 'Recovered work differs from fresh semantics')
    return evidence


def check_summary(summary, values):
    require(summary['rows'] == len(values) and summary['families'] == len({row['family'] for row in values}) and
            summary['single_legal_action_rows'] == sum(row['legal_action_count'] == 1 for row in values), 'Diagnosis group counts differ')
    for output, key in (('value_mae_seeds', 'value_absolute_error'), ('signed_actor_error', 'signed_actor_error'),
                        ('policy_teacher_regret', 'policy_teacher_regret')):
        require(summary[output] == (statistics.mean(row[key] for row in values) if values else None), 'Diagnosis mean differs: ' + output)
    require(summary['policy_teacher_best_fraction'] == (statistics.mean(row['policy_teacher_regret'] == 0 for row in values) if values else None), 'Diagnosis best agreement differs')
    saved = [row for row in values if row['saved_puct_teacher_regret'] is not None]
    require(summary['saved_puct_rows'] == len(saved) and summary['saved_puct_teacher_disagreements'] == sum(row['saved_puct_teacher_regret'] > 0 for row in saved) and
            summary['saved_puct_teacher_regret'] == (statistics.mean(row['saved_puct_teacher_regret'] for row in saved) if saved else None), 'Saved choice diagnosis differs')


def diagnosis(roots, labels, attempts):
    report = load('build/data-pilot-model-diagnosis.json')
    current(report['inputs'])
    require(report['schema'] == 'fixed-control-data-pilot-diagnosis-v1' and report['passed'] is True and report['model_file_sha256'] == pilot.MODEL_SHA,
            'Fixed-model diagnosis identity differs')
    require(report['selected_roots'] == len(roots) and report['completed_H8_labels'] == len(labels) and report['rejected_label_attempts'] == len(attempts)-len(labels), 'Diagnosis target coverage differs')
    actual = {row['state_key']: row for row in report['metrics']}
    require(len(actual) == len(report['metrics']) and set(actual) == set(labels), 'Diagnosis has missing or duplicate rows')
    for key, value in actual.items():
        root, label = roots[key], labels[key]['label']
        scores = {row['pit']: row['value_player0'] for row in label['action_scores']}
        sign = 1 if root['state']['controller'] == 0 else -1
        require(value['family'] == root['family'] and value['cohort'] == root['cohort'] and value['stratum'] == root['stratum'] and
                value['state'] == root['state'] and value['legal_action_count'] == len(scores), 'Diagnosis root metadata differs')
        require(value['approximate_H8_target_player0'] == label['target_player0'] and value['teacher_action_scores'] == label['action_scores'] and
                value['teacher_best_moves'] == label['best_moves'] and value['policy_chosen'] in scores, 'Diagnosis action targets differ')
        require(value['value_absolute_error'] == abs(value['control_value_player0']-label['target_player0']) and
                value['signed_actor_error'] == sign*(value['control_value_player0']-label['target_player0']) and
                value['policy_teacher_regret'] == sign*(label['target_player0']-scores[value['policy_chosen']]), 'Diagnosis sign/error arithmetic differs')
        saved = root['provenance']['saved_puct_chosen']
        require(value['saved_puct_chosen'] == saved and value['saved_puct_teacher_regret'] == (None if saved is None else sign*(label['target_player0']-scores[saved])), 'Saved PUCT target regret differs')
        require(math.isfinite(value['control_value_player0']) and value['policy_teacher_regret'] >= 0, 'Invalid diagnosis numeric output')
    for cohort in ('diagnostic', 'training_candidate', 'validation'):
        values = [row for row in actual.values() if row['cohort'] == cohort]
        check_summary(report['cohorts'][cohort], values)
        check_summary(report['choice_bearing_cohorts'][cohort], [row for row in values if row['legal_action_count'] >= 2])
    require(set(report['strata']) == {row['cohort']+'/'+row['stratum'] for row in actual.values()} and
            set(report['by_source']) == {row['provenance_source'] for row in actual.values()}, 'Diagnosis group keys differ')
    for key, values in report['strata'].items():
        check_summary(values, [row for row in actual.values() if row['cohort']+'/'+row['stratum'] == key])
    for key, values in report['by_source'].items():
        check_summary(values, [row for row in actual.values() if row['provenance_source'] == key])
    for flag in ('training_performed', 'model_changed', 'new_search_performed', 'active_sampling_weights_changed',
                 'sealed_final_test_consumed', 'independent_unseen_release_test', 'strength_improvement_claimed'):
        require(report[flag] is False, 'Diagnosis exceeds authorized scope: ' + flag)
    return report


def retained():
    paths = set()
    for folder in ('build/data-pilot-audit', 'build/data-pilot-author', 'build/data-pilot-smoke-v1'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and not path.name.endswith(('-wal', '-shm')):
                pin(path)
                paths.add(path.relative_to(ROOT).as_posix())
    return sorted(paths)


def build_report():
    INPUTS.clear()
    pin('tools/record-data-pilot.py')
    baseline = load('build/data-pilot-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 1834 and baseline['fixed_model_sha256'] == pilot.MODEL_SHA,
            'Frozen 1834-input baseline differs')
    current(baseline['inputs'])
    prior = load('build/value-acceptance.json')
    require(prior['passed'] is True and len(prior['inputs']) == 1829 and all(baseline['inputs'].get(name) == value for name, value in prior['inputs'].items()), 'Prior accepted 1829-input coverage differs')
    plan, implementation = pilot.admit()
    pin('build/data-pilot-plan.json'); pin('build/data-pilot-implementation.json'); pin('build/data-pilot-registry.json')
    current(implementation['sources']); current(implementation['prior_inputs'])
    require(pin(pilot.MODEL) == pilot.MODEL_SHA, 'Frozen model changed')
    author, author_ids = group('build/data-pilot-author.json', 'tests/data_pilot_author.py', 4, 'methods')
    require(set(author['methods']) == set(method_ids('tests/data_pilot_author.py')), 'Author source execution is incomplete')
    preflight, before_ids = group('build/data-pilot-audit/preflight.json', 'tests/data_pilot_independent_audit.py', 6, 'covered_methods')
    final, after_ids = group('build/data-pilot-audit/final.json', 'tests/data_pilot_independent_audit.py', 2, 'covered_methods')
    independent_ids = before_ids + after_ids
    require(len(independent_ids) == len(set(independent_ids)) == 8 and set(independent_ids) == set(method_ids('tests/data_pilot_independent_audit.py').values()), 'Independent unique execution coverage differs')
    collector = load('build/data-pilot-audit.json'); current(collector['inputs'])
    pin('build/data-pilot-audit/collect.py')
    require(collector['passed'] is True and collector['tests_run'] == 8 and collector['failures'] == collector['errors'] == collector['skips'] == 0 and
            collector['prior_inputs_preserved'] == 1834 and set(collector['covered_methods']) == set(preflight['covered_methods'] + final['covered_methods']) and
            len(collector['covered_methods']) == 8, 'Combined independent proof coverage differs')
    expected_groups = [dict(path=path, sha256=pin(path), tests_run=report['tests_run'], covered_methods=report['covered_methods'])
                       for path, report in (('build/data-pilot-audit/preflight.json', preflight), ('build/data-pilot-audit/final.json', final))]
    require(collector['groups'] == expected_groups and collector['evidence'] == {**preflight['evidence'], **final['evidence']}, 'Combined independent proof changes executed evidence')
    ids = author_ids + independent_ids
    require(len(ids) == len(set(ids)) == 12, 'Aggregate unique executed method count differs')
    require(preflight['evidence']['baseline']['prior_inputs'] == 1834 and preflight['evidence']['baseline']['all_hashes_current'] is True, 'Independent baseline proof differs')
    sampling = load('build/sampling-review.json'); current(sampling['inputs'])
    require(sampling['schema'] == 'data-pilot-offline-sampling-review-v1' and sampling['passed'] is True and
            sampling['observed_accepted_run']['total_draws'] == 384000, 'Offline exposure review differs')
    checked = preflight['evidence']['sampling_review']
    require(checked['distributions'] == 4 and checked['probability_mass_conserved'] is True and checked['positive_support'] is True and checked['root_and_family_concentration_verified'] is True and
            checked['original_draws_literal_rng_replayed'] == 384000 and checked['tiny_nine_root_draws'] == 54873 and checked['training_weights_applied'] is False, 'Independent sampling arithmetic/replay proof incomplete')
    require(set(sampling['distributions']) == set(checked['effective_support']), 'Sampling distribution coverage differs')
    for name, row in sampling['distributions'].items():
        require(row['root_effective_support'] == checked['effective_support'][name]['root'] and row['family_effective_support'] == checked['effective_support'][name]['family'], 'Independently checked sampling concentration differs')
    collection_recovery = recovery(preflight, 'live_collection_recovery', 'families', ('data_pilot_family_precommit', 'data_pilot_family_committed'))
    label_recovery = recovery(preflight, 'live_label_recovery', 'attempts', ('data_pilot_label_precommit', 'data_pilot_label_committed'))
    require(all(value is True for value in preflight['evidence']['refusals'].values()), 'Independent corruption/partial refusals incomplete')
    # A completed recorder must not run fresh teacher work. The closed producer
    # reader checks all tables, histories, identities and report/label links.
    original = Native.analyze
    def prohibited(*_args, **_kwargs):
        raise ValueError('Acceptance must not execute a new teacher search')
    Native.analyze = prohibited
    try:
        roots, labels, attempts, manifest = pilot.load_rows()
    finally:
        Native.analyze = original
    require(manifest == load('build/data-pilot-result.json') == load('data/ordinary-data-pilot-v1/manifest.json') and manifest['identity']['fixture'] is False and manifest['families'] == 256, 'Production durable publication differs')
    pin('data/ordinary-data-pilot-v1/work.sqlite3')
    collected = final['evidence']['production_collection']; labeled = final['evidence']['production_labels']
    require(collected['families'] == 256 and collected['selected_roots'] == len(roots) and collected['selection'] == manifest['selection'] and collected['candidate_observations'] == manifest['selection']['ordinary_observations'] and
            all(collected[name] is True for name in ('all_histories_replayed', 'all_rng_and_policy_choices_rederived', 'full_pool_collision_closure_rederived')), 'Independent exhaustive collection differs')
    require(labeled['selected_roots'] == len(roots) and labeled['attempts'] == len(attempts) and labeled['complete_H8_labels'] == len(labels) and labeled['budget_rejected'] == len(attempts)-len(labels) and labeled['fixed_model_literal_forwards'] == len(labels) and labeled['fresh_H8_reports_recomputed'] == min(2, len(labels)) and
            all(labeled[name] is True for name in ('all_report_label_links_validated', 'actor_signs_best_ties_and_metric_groups_rederived', 'single_action_and_choice_bearing_groups_distinguished', 'completed_journal_and_diagnosis_bytes_mtimes_unchanged')), 'Independent exhaustive labels/model arithmetic incomplete')
    metrics = diagnosis(roots, labels, attempts)
    rerun = load('build/data-pilot-production-rerun.json')
    require(rerun['schema'] == 'ordinary-data-pilot-completed-rerun-v1' and rerun['passed'] is True and rerun['completed_bytes_mtimes_unchanged'] is True and rerun['native_teacher_calls'] == 0 and
            rerun['families'] == 256 and rerun['labels'] == len(labels) and rerun['source_sha256'] == pin('build/data-pilot-production-rerun.py') and rerun['producer_sha256'] == pin('congklak/data_pilot.py') and
            rerun['implementation_sha256'] == pin('build/data-pilot-implementation.json'), 'Completed production rerun proof differs')
    require(set(rerun['artifacts']) == {'data/ordinary-data-pilot-v1/work.sqlite3', 'data/ordinary-data-pilot-v1/manifest.json', 'data/ordinary-data-pilot-v1/writer.lock', 'build/data-pilot-result.json'}, 'Completed production stability coverage differs')
    stable_artifacts(rerun['artifacts'])
    root_proof = load('build/data-pilot-root-verification.json')
    require(root_proof['schema'] == 'fixed-model-data-pilot-root-verification-v1' and root_proof['passed'] is True and root_proof['prior_inputs_preserved'] == 1834 and root_proof['completed_artifacts_bytes_mtimes_unchanged'] is True and root_proof['before'] == root_proof['after'] and
            root_proof['analysis_source_sha256'] == pin('tools/analyze-data-pilot.py'), 'Root completed diagnosis stability differs')
    require(set(root_proof['after']) == {'data/ordinary-data-pilot-v1/work.sqlite3', 'data/ordinary-data-pilot-v1/manifest.json', 'build/data-pilot-result.json', 'build/data-pilot-model-diagnosis.json'}, 'Root stability artifact coverage differs')
    stable_artifacts(root_proof['after'])
    for path in ('congklak/data_pilot_progress.py', 'tools/data-pilot-dashboard.html', 'docs/DATA_PILOT_SCOPE.md', 'docs/DATA_PILOT_RESULTS.md', 'docs/DATA_PILOT_INDEPENDENT_AUDIT.md', 'build/data-pilot-model-diagnosis-rerun.log'):
        pin(path)
    raw = retained()
    current(dict(INPUTS))
    require('torch' not in sys.modules, 'Acceptance imported Torch')
    return dict(schema='data-pilot-acceptance-v1', passed=True, engineering_passed=True,
        tests_run=12, failures=0, errors=0, skips=0, covered_checks=ids,
        groups=dict(author=dict(tests_run=4, methods=author_ids), independent=dict(tests_run=8, methods=independent_ids)),
        prior_inputs_preserved=dict(prior_accepted=1829, prior_plus_completed_diagnosis=1834, all_current_hashes_match=True),
        implementation_sha256=pin('build/data-pilot-implementation.json'), model_file_sha256=pilot.MODEL_SHA,
        families=256, selected_roots=len(roots), complete_H8_labels=len(labels), budget_rejected=len(attempts)-len(labels),
        selected_cohorts=dict(sorted(Counter(row['cohort'] for row in roots.values()).items())),
        labeled_cohorts=dict(sorted(Counter(roots[key]['cohort'] for key in labels).items())), selection=manifest['selection'],
        diagnostic_metrics=metrics['cohorts'], choice_bearing_metrics=metrics['choice_bearing_cohorts'],
        distribution_effective_support=checked['effective_support'], original_optimizer_draws_replayed=384000,
        actual_collection_pre_postcommit_kills=collection_recovery['actual_process_kills'],
        actual_label_pre_postcommit_kills=label_recovery['actual_process_kills'], completed_bytes_mtimes_unchanged=True,
        full_pool_and_H8_links_independently_audited=True, torch_imported=False,
        training_performed=False, model_changed=False, optimizer_weights_applied=False, sampling_weights_applied=False,
        pruning_or_deletion_performed=False, default_promotion=False, full_game_solved=False, sealed_final_test_consumed=False,
        independent_unseen_release_test=False, strength_improvement_claimed=False, energy_measured=False,
        teacher_scope=manifest['teacher_scope'], shared_history_prefixes_possible=True,
        scope='Fixed-model small development data pilot; no new equal-time games, training, data weights or promotion. Shortfalls are retained.',
        retained_audit_owner_and_failed_probe_evidence=raw, inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    report = build_report()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(report)+'\n', 'Previously sealed pilot acceptance differs')
    if not args.check_only:
        pilot.publish(OUTPUT, report)
    print(config.canonical(dict(event='data_pilot_acceptance_checked' if args.check_only else 'data_pilot_accepted',
        passed=True, tests_run=12, inputs=len(INPUTS), families=256, complete_H8_labels=report['complete_H8_labels'],
        budget_rejected=report['budget_rejected'], training_performed=False, acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(config.canonical(dict(event='data_pilot_record_refused', error=str(exc))), flush=True)
        raise SystemExit(2)
