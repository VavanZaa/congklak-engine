"""Seal the bounded natural H8/H10 study from complete, current audited evidence."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import statistics
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, natural_study as study
from congklak.native import Native

OUTPUT = ROOT / 'build/natural-study-acceptance.json'
INPUTS = {}
MEANS = ('value_mae_h8', 'value_mae_h10', 'value_mae_change_h10_minus_h8',
         'policy_regret_h8', 'policy_regret_h10', 'policy_regret_change_h10_minus_h8',
         'absolute_target_shift', 'signed_actor_target_shift', 'H10_regret_H8_chosen',
         'H10_regret_H8_tie_min', 'H10_regret_H8_tie_max', 'action_Q_shift_mae')
FLAGS = ('best_sets_changed', 'best_sets_disjoint', 'H8_choice_still_best', 'all_H8_ties_still_best')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = (ROOT / path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing or external acceptance input: ' + str(path))
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    name = path.relative_to(ROOT).as_posix()
    require(name not in INPUTS or INPUTS[name] == value, 'Acceptance input changed: ' + name)
    INPUTS[name] = value
    return value


def load(path):
    pin(path)
    return json.loads((ROOT / path).read_text(encoding='utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def current(mapping):
    require(isinstance(mapping, dict) and mapping, 'Evidence has no immutable input pins')
    for name, digest in mapping.items():
        require(isinstance(digest, str) and len(digest) == 64 and pin(name) == digest, 'Evidence input changed: ' + name)


def methods(path):
    pin(path)
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    pairs = [(node.name, item.name) for node in tree.body if isinstance(node, ast.ClassDef)
             for item in node.body if isinstance(item, ast.FunctionDef) and item.name.startswith('test_')]
    require(len(pairs) == len(set(pairs)) == len({name for _, name in pairs}), 'Duplicate source test methods')
    return {name: path + '::' + cls + '.' + name for cls, name in pairs}


def group(path, source, count, names_key):
    report = load(path); current(report['inputs'])
    require(report['passed'] is True and report['tests_run'] == count and report['errors'] == report['failures'] == report['skips'] == 0, 'Incomplete executed tests: ' + str(path))
    names, actual = report[names_key], methods(source)
    require(len(names) == len(set(names)) == count and set(names) <= set(actual) and report['inputs'].get(source) == pin(source), 'Executed method/source coverage differs')
    return report, [actual[name] for name in names]


def stable(mapping):
    require(isinstance(mapping, dict) and mapping, 'No closed artifact stability coverage')
    for name, row in mapping.items():
        require(pin(name) == row['sha256'] and (ROOT / name).stat().st_mtime_ns == row['mtime_ns'], 'Completed bytes/mtime changed: ' + name)
        if 'size' in row:
            require((ROOT / name).stat().st_size == row['size'], 'Completed file size differs: ' + name)


def recoveries(report):
    collection = report['evidence']['live_collection_recovery']
    teachers = report['evidence']['live_teacher_recovery']
    require(collection['actual_kills'] is True and collection['raw_backups_retained'] is True and
            teachers['actual_kills'] is True and teachers['raw_backups_retained'] is True and teachers['H8_and_H10_both_barriers'] is True,
            'Actual retained collection/H8/H10 process kills are absent')
    family_events = {'natural_study_family_precommit', 'natural_study_family_committed'}
    teacher_events = {'natural_study_'+stage+'_'+event for stage in ('h8', 'h10') for event in ('precommit', 'committed')}
    require(len(collection['tests']) == 2 and {row['event'] for row in collection['tests']} == family_events and
            len(teachers['tests']) == 4 and {row['event'] for row in teachers['tests']} == teacher_events, 'Six-barrier recovery coverage differs')
    for folder, records in (('collection', collection['tests']), ('teachers', teachers['tests'])):
        for row in records:
            base = Path(report['retained_directory']) / folder / (row['event']+'-probe')
            marker = load(base/'marker.json')
            require(marker == row['marker'] and marker['event'] == row['event'], 'Retained actual-kill marker differs')
            path = base/'validated-prefix.sqlite3'; pin(path)
            with closing(sqlite3.connect((ROOT/path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
                if folder == 'collection':
                    expected = 1 if row['event'].endswith('precommit') else 2
                    require(row['committed_prefix'] == expected == db.execute('SELECT COUNT(*) FROM families').fetchone()[0] and row['families_after_resume'] == 8 and
                            all(row[name] is True for name in ('fresh_histories_and_selection_equal', 'prefix_retained', 'bytes_mtimes_unchanged')), 'Killed collection prefix/recovery differs')
                else:
                    stage = row['stage'].lower(); require(stage in ('h8', 'h10') and '_'+stage+'_' in row['event'], 'Teacher barrier stage differs')
                    expected_new = (0 if row['event'].endswith('precommit') else 1) if stage == 'h8' else (1 if row['event'].endswith('precommit') else 2)
                    inherited = 2 if stage == 'h8' else 0
                    total = db.execute('SELECT COUNT(*) FROM '+stage+'_attempts').fetchone()[0]
                    require(row['committed_new_prefix'] == expected_new and row['committed_attempt_rows'] == total == expected_new+inherited and
                            all(row[name] is True for name in ('fresh_semantics_and_counters_equal', 'prefix_exactly_retained', 'completed_zero_teacher_calls', 'bytes_mtimes_unchanged')),
                            'Killed teacher prefix or exact recovery differs')
    return 6


def summary(values):
    pairs = [row for row in values if row['complete_pair']]
    choices = [row for row in pairs if row['legal_action_count'] >= 2]
    return dict(selected_roots=len(values), families=len({row['family'] for row in values}),
        h8_complete=sum(row['h8_complete'] for row in values), h10_complete=sum(row['h10_complete'] for row in values),
        complete_pairs=len(pairs), unpaired_roots=len(values)-len(pairs), choice_bearing_pairs=len(choices), single_legal_action_pairs=len(pairs)-len(choices),
        paired_means={key: statistics.mean(row[key] for row in pairs) if pairs else None for key in MEANS},
        paired_choice_bearing_means={key: statistics.mean(row[key] for row in choices) if choices else None for key in MEANS},
        **{key: sum(row[key] for row in pairs) for key in FLAGS},
        model_MAE_lower_at_H10=sum(row['value_mae_change_h10_minus_h8'] < 0 for row in pairs),
        model_MAE_higher_at_H10=sum(row['value_mae_change_h10_minus_h8'] > 0 for row in pairs),
        model_MAE_equal_at_H10=sum(row['value_mae_change_h10_minus_h8'] == 0 for row in pairs),
        unpaired_statuses=dict(sorted(Counter(row['h8_status']+'/'+row['h10_status'] for row in values if not row['complete_pair']).items())))


def stability(rows, h8, h10, a8, a10):
    report = load('build/natural-study-stability.json'); current(report['inputs'])
    require(report['schema'] == 'fixed-control-H8-H10-stability-v1' and report['passed'] is True and report['fixed_model_sha256'] == study.MODEL_SHA and report['rules_digest'] == config.RULES_DIGEST, 'Stability identity differs')
    metrics = report['metrics']; actual = {row['state_key']: row for row in metrics}
    require(len(metrics) == len(actual) == len(rows) and set(actual) == set(rows), 'Missing or duplicate stability roots')
    for key, root in rows.items():
        row = actual[key]; paired = key in h8 and key in h10
        require(all(row[name] == root[name] for name in ('state', 'family', 'cohort', 'study_role')) and
                row['legal_action_count'] == root['legal_mask'].bit_count() and row['policy_chosen'] in range(7) and root['legal_mask'] & (1 << row['policy_chosen']), 'Stability root/policy metadata differs')
        require(row['h8_complete'] == (key in h8) and row['h10_complete'] == (key in h10) and row['complete_pair'] == paired and
                row['h8_status'] == a8[key]['status'] and row['h10_status'] == a10[key]['status'], 'Paired inclusion/status differs')
        require(type(row['model_value_player0']) in (int, float) and math.isfinite(row['model_value_player0']), 'Invalid fixed model output')
        if not paired:
            require(all(row[name] is None for name in MEANS + FLAGS), 'Unpaired root has an imputed comparative metric')
            continue
        l8, l10 = h8[key]['label'], h10[key]['label']
        q8 = {entry['pit']: entry['value_player0'] for entry in l8['action_scores']}
        q10 = {entry['pit']: entry['value_player0'] for entry in l10['action_scores']}
        require(list(q8) == list(q10) and sum(1 << action for action in q8) == root['legal_mask'], 'Paired legal action coverage differs')
        sign = 1 if root['state']['controller'] == 0 else -1
        v8, v10, chosen = l8['target_player0'], l10['target_player0'], row['policy_chosen']
        b8, b10 = l8['best_moves'], l10['best_moves']
        e8, e10 = abs(row['model_value_player0']-v8), abs(row['model_value_player0']-v10)
        r8, r10 = sign*(v8-q8[chosen]), sign*(v10-q10[chosen])
        ties = [sign*(v10-q10[action]) for action in b8]
        expected = dict(value_mae_h8=e8, value_mae_h10=e10, value_mae_change_h10_minus_h8=e10-e8,
            policy_regret_h8=r8, policy_regret_h10=r10, policy_regret_change_h10_minus_h8=r10-r8,
            absolute_target_shift=abs(v10-v8), signed_actor_target_shift=sign*(v10-v8), H10_regret_H8_chosen=sign*(v10-q10[min(b8)]),
            H10_regret_H8_tie_min=min(ties), H10_regret_H8_tie_max=max(ties), action_Q_shift_mae=statistics.mean(abs(q10[action]-q8[action]) for action in q8),
            best_sets_changed=set(b8) != set(b10), best_sets_disjoint=not bool(set(b8) & set(b10)), H8_choice_still_best=min(b8) in b10, all_H8_ties_still_best=set(b8) <= set(b10),
            H8_target_player0=v8, H10_target_player0=v10, H8_best_moves=b8, H10_best_moves=b10, H8_action_scores=l8['action_scores'], H10_action_scores=l10['action_scores'])
        require(all(row[name] == value for name, value in expected.items()) and min(r8, r10, *ties) >= 0, 'Teacher shift/model metric arithmetic differs')
    expected_counts = dict(selected_roots=len(rows), h8_complete=len(h8), h10_complete=len(h10), complete_pairs=len(set(h8) & set(h10)),
        unpaired_roots=len(rows)-len(set(h8) & set(h10)), h8_rejected=sum(row['status'] == 'budget_rejected' for row in a8.values()),
        h10_rejected=sum(row['status'] == 'budget_rejected' for row in a10.values()), old_h8_references=sum(row['study_role'] == 'followup' for row in rows.values()),
        new_h8_labels=sum(rows[key]['study_role'] == 'fresh' for key in h8), new_h10_labels=len(h10))
    require(report['counts'] == expected_counts and report['overall'] == summary(metrics), 'Stability overall/intersection denominators differ')
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row['study_role']+'/'+row['cohort']].append(row)
    require(report['groups'] == {name: summary(values) for name, values in sorted(grouped.items())}, 'Origin/cohort paired summaries differ')
    for name in ('training_performed', 'model_changed', 'active_sampling_weights_changed', 'new_PUCT_or_strength_tests', 'strength_improvement_claimed', 'sealed_final_test_consumed', 'independent_unseen_release_test'):
        require(report[name] is False, 'Interpretation exceeds study scope: ' + name)
    return report


def retained():
    result = set()
    for folder in ('build/natural-study-audit', 'build/natural-study-author', 'build/natural-study-smoke-v1'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and not path.name.endswith(('-wal', '-shm')):
                pin(path); result.add(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'build').glob('natural-study-*-feasibility.json')):
        pin(path); result.add(path.relative_to(ROOT).as_posix())
    return sorted(result)


def build_report():
    INPUTS.clear(); pin('tools/record-natural-study.py'); pin('tools/prepare-natural-study-baseline.py')
    baseline = load('build/natural-study-baseline.json'); current(baseline['inputs'])
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2060 and baseline['prior_accepted_inputs'] == 2058, 'Natural study baseline differs')
    prior = load('build/data-pilot-acceptance.json')
    require(prior['passed'] is True and len(prior['inputs']) == 2058 and all(baseline['inputs'].get(name) == digest for name, digest in prior['inputs'].items()), 'Prior sealed input coverage differs')
    plan, implementation = study.admit(); current(implementation['sources']); current(implementation['prior_inputs'])
    for path in ('build/natural-study-plan.json', 'build/natural-study-implementation.json', 'build/natural-study-registry.json'):
        pin(path)
    require(pin(study.MODEL) == study.MODEL_SHA, 'Retained model changed')
    author, author_ids = group('build/natural-study-author.json', 'tests/natural_study_author.py', 4, 'methods')
    require(set(author['methods']) == set(methods('tests/natural_study_author.py')), 'Author execution coverage incomplete')
    before, before_ids = group('build/natural-study-audit/preflight.json', 'tests/natural_study_independent_audit.py', 6, 'covered_methods')
    final, final_ids = group('build/natural-study-audit/final.json', 'tests/natural_study_independent_audit.py', 2, 'covered_methods')
    require(len(before_ids+final_ids) == len(set(before_ids+final_ids)) == 8 and set(before_ids+final_ids) == set(methods('tests/natural_study_independent_audit.py').values()), 'Independent unique source coverage differs')
    audit = load('build/natural-study-audit.json'); current(audit['inputs'])
    require(audit['passed'] is True and audit['tests_run'] == 8 and audit['errors'] == audit['failures'] == audit['skips'] == 0 and
            audit['prior_inputs_preserved'] == 2060 and set(audit['covered_methods']) == set(before['covered_methods']+final['covered_methods']) and len(audit['covered_methods']) == 8, 'Combined audit incomplete')
    pin('build/natural-study-audit/collect.py')
    expected_groups = [dict(path=path, sha256=pin(path), tests_run=report['tests_run'], covered_methods=report['covered_methods'])
                       for path, report in (('build/natural-study-audit/preflight.json', before), ('build/natural-study-audit/final.json', final))]
    require(audit['groups'] == expected_groups and audit['evidence'] == {**before['evidence'], **final['evidence']}, 'Combined audit changes actual group evidence')
    actual_kill_barriers = recoveries(before)
    checked_baseline = before['evidence']['baseline']; protected = before['evidence']['protection_and_followups']
    require(checked_baseline['prior_inputs'] == 2060 and checked_baseline['old_accepted_inputs'] == 2058 and checked_baseline['all_hashes_current'] is True and
            protected['old_pilot_roots'] == 62 and protected['old_pilot_families'] == 256 and protected['exact_old_H8_references'] == 15 and protected['old_H8_hashes_not_rewritten'] is True,
            'Independent prior/follow-up preservation incomplete')
    refused = before['evidence']['refusals_and_censoring']
    require(refused['real_H10_partial_depth'] in (8, 9) and all(value is True for name, value in refused.items() if name != 'real_H10_partial_depth'), 'Actual partial/rehashed corruption refusal proof incomplete')
    # The producer's closed reader replays deterministic collection and validates
    # all complete labels and rejected attempts; fresh teacher searches are forbidden.
    original = Native.analyze
    def prohibited(*_args, **_kwargs):
        raise ValueError('Acceptance must not execute a teacher search')
    Native.analyze = prohibited
    try:
        rows, h8, h10, a8, a10, manifest = study.load_rows()
    finally:
        Native.analyze = original
    require(manifest == load('build/natural-study-result.json') == load('data/natural-study-v1/manifest.json') and manifest['families'] == 512 and manifest['identity']['fixture'] is False, 'Closed production publication differs')
    pin('data/natural-study-v1/work.sqlite3')
    require(manifest['followups'] == manifest['h8_inherited_labels'] == 15 and manifest['selected_fresh'] <= 32, 'Study population differs')
    registry = load('build/natural-study-registry.json')
    collected = final['evidence']['production_collection']
    require(collected['families'] == 512 and collected['selection'] == manifest['selection'] and collected['fresh_selected'] == manifest['selected_fresh'] and collected['unchanged_old_followups'] == 15 and
            all(collected[name] is True for name in ('all_literal_rng_and_policy_choices_match', 'full_pool_closure_and_selection_match', 'exact_old_H8_items_reused', 'all_H10_targets_attempted_independent_of_H8_status')),
            'Independent full-pool collection and ownership verification differs')
    for key, root in rows.items():
        if root['study_role'] == 'followup':
            require(root['root'] == registry['followups'][key]['root'] and h8[key] == registry['followups'][key]['h8'] and
                    root['cohort'] == root['root']['cohort'] and a8[key]['status'] == 'inherited_complete', 'Original follow-up H8/ownership changed')
    interpreted = stability(rows, h8, h10, a8, a10)
    labels_proof = final['evidence']['production_labels_and_metrics']
    require(labels_proof['roots'] == len(rows) and labels_proof['h8_labels'] == len(h8) and labels_proof['h10_labels'] == len(h10) and labels_proof['complete_pairs'] == len(set(h8) & set(h10)) and labels_proof['literal_model_forwards'] == len(rows) and
            all(labels_proof[name] is True for name in ('all_horizon_links_and_ties_verified', 'all_paired_values_regrets_shifts_and_denominators_rederived', 'completed_zero_teacher_calls', 'completed_bytes_mtimes_unchanged', 'stability_bytes_mtimes_unchanged', 'all2060_prior_hashes_current')), 'Independent exhaustive paired verification incomplete')
    production_rerun = load('build/natural-study-production-rerun.json')
    require(production_rerun['passed'] is True and production_rerun['completed_bytes_mtimes_unchanged'] is True and production_rerun['native_teacher_calls'] == 0 and
            production_rerun['source_sha256'] == pin('build/natural-study-production-rerun.py') and production_rerun['producer_sha256'] == pin('congklak/natural_study.py') and
            production_rerun['implementation_sha256'] == pin('build/natural-study-implementation.json'), 'Completed zero-teacher rerun proof differs')
    require(set(production_rerun['artifacts']) == {'data/natural-study-v1/work.sqlite3', 'data/natural-study-v1/manifest.json', 'data/natural-study-v1/writer.lock', 'build/natural-study-result.json'}, 'Completed producer stability coverage differs')
    stable(production_rerun['artifacts'])
    root_rerun = load('build/natural-study-root-verification.json')
    require(root_rerun['schema'] == 'fixed-model-natural-study-root-verification-v1' and root_rerun['passed'] is True and root_rerun['prior_inputs_preserved'] == 2060 and root_rerun['completed_artifacts_bytes_mtimes_unchanged'] is True and
            root_rerun['before'] == root_rerun['after'] and root_rerun['analysis_source_sha256'] == pin('tools/analyze-natural-study.py') and
            root_rerun['verifier_source_sha256'] == pin('tools/verify-natural-study-rerun.py') and root_rerun['baseline_sha256'] == pin('build/natural-study-baseline.json') and
            root_rerun['training_performed'] is False and root_rerun['model_sha256'] == study.MODEL_SHA and root_rerun['analysis_rerun_exit_code'] == 0, 'Completed stability rerun proof differs')
    require(set(root_rerun['after']) == {'data/natural-study-v1/work.sqlite3', 'data/natural-study-v1/manifest.json', 'build/natural-study-result.json', 'build/natural-study-stability.json'}, 'Root stability artifact coverage differs')
    stable(root_rerun['after'])
    for path in ('congklak/natural_study_progress.py', 'tools/natural-study-dashboard.html', 'docs/NATURAL_STUDY_SCOPE.md', 'docs/NATURAL_STUDY_RESULTS.md', 'docs/NATURAL_STUDY_INDEPENDENT_AUDIT.md', 'build/natural-study-stability-rerun.log'):
        pin(path)
    raw = retained(); current(dict(INPUTS)); require('torch' not in sys.modules, 'CPU-only recorder imported Torch')
    ids = author_ids+before_ids+final_ids
    require(len(ids) == len(set(ids)) == 12, 'Aggregate unique executed method count differs')
    return dict(schema='natural-study-acceptance-v1', passed=True, engineering_passed=True, tests_run=12, failures=0, errors=0, skips=0,
        covered_checks=ids, groups=dict(author=dict(tests_run=4, methods=author_ids), independent=dict(tests_run=8, methods=before_ids+final_ids)),
        prior_inputs_preserved=dict(prior_accepted=2058, prior_plus_acceptance_rerun=2060, all_current_hashes_match=True),
        implementation_sha256=pin('build/natural-study-implementation.json'), model_file_sha256=study.MODEL_SHA,
        families=512, selected_fresh=manifest['selected_fresh'], followups=15, counts=interpreted['counts'], selection=manifest['selection'],
        collection_literal_choices=collected['literal_choices'], collection_policy_evaluations=collected['policy_evaluations'],
        overall=interpreted['overall'], paired_groups=interpreted['groups'], independent_complete=True, completed_zero_teacher_calls=True, completed_bytes_mtimes_unchanged=True,
        actual_collection_and_teacher_kill_barriers=actual_kill_barriers,
        original_h8_references_and_ownership_preserved=True, training_performed=False, model_changed=False, sampling_weights_applied=False,
        optimizer_weights_applied=False, pruning_or_deletion_performed=False, default_promotion=False, sealed_final_test_consumed=False,
        independent_unseen_release_test=False, strength_improvement_claimed=False, full_game_solved=False, energy_measured=False, torch_imported=False,
        scope='Bounded natural 50-74 development study. Complete H8/H10 intersection comparisons only; deeper H_v1 targets are estimates, not game truth. No training, active weights or promotion.',
        retained_recovery_and_diagnostic_evidence=raw, inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    report = build_report()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(report)+'\n', 'Previously sealed natural-study acceptance differs')
    if not args.check_only:
        study.publish(OUTPUT, report)
    print(config.canonical(dict(event='natural_study_acceptance_checked' if args.check_only else 'natural_study_accepted', passed=True,
        tests_run=12, inputs=len(INPUTS), selected_fresh=report['selected_fresh'], complete_pairs=report['counts']['complete_pairs'],
        h10_rejected=report['counts']['h10_rejected'], acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(config.canonical(dict(event='natural_study_record_refused', error=str(exc))), flush=True)
        raise SystemExit(2)
