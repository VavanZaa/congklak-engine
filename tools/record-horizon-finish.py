"""Seal the five-root H14 finish only after every requested label completes."""
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
from congklak import config, horizon_finish as finish
from congklak.native import Native

OUTPUT = ROOT / 'build/horizon-finish-acceptance.json'
INPUTS = {}
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


def group(path, source, names_key):
    report = load(path); current(report['inputs'])
    names, actual = report[names_key], methods(source)
    count=len(names)
    require(report['passed'] is True and report['tests_run'] == count and report['errors'] == report['failures'] == report['skips'] == 0, 'Incomplete executed tests: ' + str(path))
    require(len(names) == len(set(names)) == count and set(names) <= set(actual) and report['inputs'].get(source) == pin(source), 'Executed method/source coverage differs')
    return report, [actual[name] for name in names]


def stable(mapping):
    require(isinstance(mapping, dict) and mapping, 'No closed artifact stability coverage')
    for name, row in mapping.items():
        require(pin(name) == row['sha256'] and (ROOT / name).stat().st_mtime_ns == row['mtime_ns'], 'Completed bytes/mtime changed: ' + name)
        if 'size' in row:
            require((ROOT / name).stat().st_size == row['size'], 'Completed file size differs: ' + name)


def recoveries(report):
    barriers = 0
    for tier in ('tier1', 'tier2'):
        evidence = report['evidence']['live_'+tier+'_recovery']; records = evidence['tests']
        require(evidence['actual_kills'] is True and evidence['tier'] == tier and evidence['raw_seed_edits_markers_backups_retained'] is True and
                len(records) == 2 and {row['event'] for row in records} == {'horizon_finish_'+tier+'_'+suffix for suffix in ('precommit', 'committed')}, 'Actual initial/fallback recovery barriers are missing')
        source = Path(report['retained_directory'])/'shared-independent-fresh'
        for row in records:
            expected = 0 if row['event'].endswith('precommit') else 1
            require(row['tier'] == tier and row['committed_prefix'] == expected and row['targets_after_resume'] == 1 and row['initial_tier_retained_exactly'] is (tier == 'tier2') and
                    all(row[name] is True for name in ('prefix_seeded_from_actual_independent_fresh_fixture', 'prefix_exactly_retained', 'fresh_semantics_and_counters_equal', 'completed_zero_teacher_calls', 'bytes_mtimes_unchanged')), 'Recovered tier prefix or semantic proof differs')
            base = Path(report['retained_directory'])/(tier+'-kills')/(row['event']+'-probe')
            marker = load(base/'marker.json'); seed = load(row['seed_record'])
            require(marker == row['marker'] and marker['event'] == row['event'] and marker['completed'] == expected and marker['target'] == 1 and
                    marker['node_budget'] == (25000000 if tier == 'tier1' else 250000000), 'Actual fixture kill marker or tier budget differs')
            require(seed['source'] == source.as_posix() and seed['source_journal_sha256'] == pin(source/'work.sqlite3') and seed['completion_and_export_removed'] is True, 'Recorded seed source or completion edits differ')
            backup = base/'validated-prefix.sqlite3'; pin(backup)
            with closing(sqlite3.connect((ROOT/backup).resolve().as_uri()+'?mode=ro', uri=True)) as db:
                require(db.execute('SELECT COUNT(*) FROM '+tier+'_attempts').fetchone()[0] == expected, 'Physical killed-prefix row count differs')
                initial = {key: json.loads(body) for key, body in db.execute('SELECT state,body FROM tier1_attempts ORDER BY state')}
                initial_raw = dict((key, (body, digest)) for key, body, digest in db.execute('SELECT state,body,hash FROM tier1_attempts ORDER BY state'))
            if tier == 'tier1':
                require(seed['retained_initial_keys'] == [] and seed['retained_initial_digest'] == config.digest({}), 'Initial-stage seed did not remove all new work')
            else:
                require(len(initial) == 1 and list(initial) == seed['retained_initial_keys'] and config.digest(initial) == seed['retained_initial_digest'] and
                        all(value['status'] == 'budget_rejected' and value['node_budget'] == 25000000 for value in initial.values()), 'Fallback seed lost its exact genuine initial rejection')
                with closing(sqlite3.connect((ROOT/source/'work.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
                    original = dict((key, (body, digest)) for key, body, digest in db.execute('SELECT state,body,hash FROM tier1_attempts ORDER BY state'))
                require(initial_raw == original, 'Fallback initial-tier body/hash bytes changed')
            barriers += 1
    return barriers


def comparison(rows, stores, targets):
    report = load('build/horizon-finish-comparison.json'); current(report['inputs'])
    require(report['schema'] == 'fixed-control-horizon-finish-comparison-v1' and report['passed'] is True and
            report['fixed_model_sha256'] == finish.MODEL_SHA and report['rules_digest'] == config.RULES_DIGEST, 'Comparison identity differs')
    labels14 = finish.combined_labels(stores); labels12 = stores['h12_references']
    cases = report['cases']; actual = {case['state_key']: case for case in cases}
    require(len(cases) == len(actual) == 6 and list(actual) == sorted(rows), 'Comparison does not cover the six original roots')
    for index, (key, root) in enumerate(sorted(rows.items()), 1):
        case = actual[key]; old = key in stores['prior_h14_labels']
        if old:
            provenance = dict(source_kind='prior_complete_H14', source_job='prior_h14', node_budget=25000000, new_teacher_work=False)
            require(labels14[key] == stores['prior_h14_labels'][key] and key not in targets, 'Old complete sixth label changed or was searched again')
        else:
            tier = 'tier2' if key in stores['tier2_labels'] else 'tier1'
            attempt = stores[tier+'_attempts'][key]
            provenance = dict(source_kind='larger_budget_complete_H14', source_job=tier, node_budget=attempt['node_budget'], new_teacher_work=True)
            require(key in targets and labels14[key] == stores[tier+'_labels'][key], 'New requested H14 label source differs')
        best12, best14 = labels12[key]['label']['best_moves'], labels14[key]['label']['best_moves']
        same = set(best12) == set(best14)
        require(case['case_id'] == 'D'+str(index) and case['state'] == root['state'] and case['family'] == root['family'] and case['source_role'] == root['study_role'] and
                case['cohort'] == root['cohort'] and case['investigation_only'] is True and case['H14_provenance'] == provenance, 'Case identity/ownership/provenance differs')
        require(case['requested_status'] == dict(h12='complete', h14='complete') and case['requested_depths_available'] == [12, 14] and case['requested_complete_pair'] is True and
                case['same_best_set_H12_H14'] == same and case['classification'] == ('same_best_set_at_H12_H14' if same else 'different_best_sets_at_H12_H14'), 'Complete-pair classification differs')
        prior = stores['prior_h14_attempts'][key]
        prior13 = prior['history'][12]['best_moves']
        require(case['prior_requested_H14_status'] == prior['status'] and case['prior_auxiliary_H13_best_moves'] == prior13, 'Historical cap/H13 facts changed')
        sign = 1 if root['state']['controller'] == 0 else -1
        q14 = {entry['pit']: entry['value_player0'] for entry in labels14[key]['label']['action_scores']}
        value14 = labels14[key]['label']['target_player0']
        expected = [dict(pit=pit, display_pit=pit+1, regret=sign*(value14-q14[pit])) for pit in best12]
        require(case['H14_regret_of_H12_best_ties'] == expected and all(entry['regret'] >= 0 for entry in expected), 'H12 tie regrets against H14 differ')
        for depth, labels in ((8, stores['h8_references']), (10, stores['h10_references']), (12, labels12), (14, labels14)):
            evidence = case['depth_evidence'][str(depth)]; label = labels[key]['label']; original = labels[key]['report']
            require(evidence['actual_depth'] == depth and evidence['target_player0'] == label['target_player0'] and evidence['best_moves'] == label['best_moves'] and
                    evidence['display_best_moves'] == [pit+1 for pit in label['best_moves']] and evidence['source_report_sha256'] == config.digest(original), 'Requested/reference depth Q/best linkage differs')
        evidence14 = case['depth_evidence']['14']
        require(evidence14['source_kind'] == 'requested_label' and evidence14['source_job'] == provenance['source_job'] and evidence14['source_requested_depth'] == 14 and evidence14['source_node_budget'] == provenance['node_budget'], 'Requested H14 source tier/cap differs')
        best13 = case['depth_evidence']['13']['best_moves']
        require(best13 == prior13 and case['H13_H14_same_best_set'] == (set(best13) == set(best14)) and case['deepest_requested_depth'] == case['deepest_actual_completed_depth'] == 14,
                'Same-depth H13 agreement or final completed depth differs')
    expected_counts = dict(roots=6, new_H14_complete=5, reused_H14_complete=1, H12_complete=6, H14_complete=6,
        requested_complete_pairs=6, requested_incomplete_cases=0, same_best_sets_H12_H14=sum(case['same_best_set_H12_H14'] for case in cases),
        changed_best_sets_H12_H14=sum(not case['same_best_set_H12_H14'] for case in cases), same_best_sets_H13_H14=sum(case['H13_H14_same_best_set'] for case in cases))
    require(report['counts'] == expected_counts and report['case_classifications'] == dict(sorted(Counter(case['classification'] for case in cases).items())), 'Comparison completion denominators differ')
    for name in ('training_performed', 'model_changed', 'active_sampling_weights_changed', 'data_ingested', 'teacher_search_performed_by_analysis', 'new_PUCT_or_strength_tests', 'sealed_final_test_consumed', 'convergence_proved', 'full_game_solved'):
        require(report[name] is False, 'Comparison exceeds authorized scope: '+name)
    return report


def retained():
    paths = set()
    for folder in ('build/horizon-finish-audit', 'build/horizon-finish-author', 'build/horizon-finish-smoke-v1'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and not path.name.endswith(('-wal', '-shm')):
                pin(path); paths.add(path.relative_to(ROOT).as_posix())
    for pattern in ('horizon-finish-*-feasibility.json', 'horizon-finish-*failure*', 'horizon-finish-*failed*'):
        for path in sorted((ROOT/'build').glob(pattern)):
            if path.is_file():pin(path); paths.add(path.relative_to(ROOT).as_posix())
    return sorted(paths)


def build_report():
    INPUTS.clear(); pin('tools/record-horizon-finish.py'); pin('tools/prepare-horizon-finish-baseline.py')
    baseline = load('build/horizon-finish-baseline.json'); current(baseline['inputs'])
    require(baseline['schema'] == 'horizon-finish-prior-immutability-v1' and baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2314 and baseline['prior_accepted_inputs'] == 2312, 'Prior finish baseline differs')
    prior = load('build/horizon-probe-acceptance.json')
    require(prior['passed'] is True and len(prior['inputs']) == 2312 and all(baseline['inputs'].get(name) == digest for name, digest in prior['inputs'].items()), 'Prior sealed probe coverage differs')
    plan, implementation = finish.admit(); current(implementation['sources']); current(implementation['prior_inputs'])
    for path in ('build/horizon-finish-plan.json', 'build/horizon-finish-implementation.json', 'build/horizon-finish-registry.json'):
        pin(path)
    require(pin(finish.MODEL) == finish.MODEL_SHA, 'Retained model changed')
    author, author_ids = group('build/horizon-finish-author.json', 'tests/horizon_finish_author.py', 'methods')
    require(set(author['methods']) == set(methods('tests/horizon_finish_author.py')), 'Author source execution coverage incomplete')
    before, before_ids = group('build/horizon-finish-audit/preflight.json', 'tests/horizon_finish_independent_audit.py', 'covered_methods')
    final, final_ids = group('build/horizon-finish-audit/final.json', 'tests/horizon_finish_independent_audit.py', 'covered_methods')
    independent_ids = before_ids+final_ids
    require(len(independent_ids) == len(set(independent_ids)) and set(independent_ids) == set(methods('tests/horizon_finish_independent_audit.py').values()), 'Independent executed source coverage differs')
    audit = load('build/horizon-finish-audit.json'); current(audit['inputs']); pin('build/horizon-finish-audit/collect.py')
    expected_groups = [dict(path=path, sha256=pin(path), tests_run=body['tests_run'], covered_methods=body['covered_methods'])
                       for path, body in (('build/horizon-finish-audit/preflight.json', before), ('build/horizon-finish-audit/final.json', final))]
    require(audit['passed'] is True and audit['tests_run'] == len(independent_ids) and audit['errors'] == audit['failures'] == audit['skips'] == 0 and audit['prior_inputs_preserved'] == 2314 and
            audit['groups'] == expected_groups and audit['evidence'] == {**before['evidence'], **final['evidence']} and set(audit['covered_methods']) == set(before['covered_methods']+final['covered_methods']) and len(audit['covered_methods']) == len(independent_ids),
            'Combined independent proof differs')
    actual_barriers = recoveries(before)
    checked = before['evidence']['baseline']; selected = before['evidence']['exact_selection']
    require(checked['prior_inputs'] == 2314 and checked['old_accepted_inputs'] == 2312 and checked['all_hashes_current'] is True and checked['fixture_caps_separate_from_production'] is True and
            selected['original_rows_and_all_six_prior_tables_exact'] is True and selected['ownership_unchanged'] is True and selected['investigation_only'] is True and
            all(value is True for value in before['evidence']['refusals_and_eligibility'].values()), 'Independent baseline, ownership or fallback refusal proof incomplete')
    original = Native.analyze
    def prohibited(*_args, **_kwargs):
        raise ValueError('Acceptance must not execute a teacher search')
    Native.analyze = prohibited
    try:
        rows, stores, manifest = finish.load_rows()
    finally:
        Native.analyze = original
    require(manifest == load('build/horizon-finish-result.json') == load('data/horizon-finish-v1/manifest.json') and manifest['identity']['fixture'] is False and manifest['status'] == 'complete' and
            manifest['scope_finished'] is True and manifest['unresolved_keys'] == [] and manifest['unresolved_target_roots'] == 0, 'Not all five requested H14 roots are complete; acceptance refused')
    pin('data/horizon-finish-v1/work.sqlite3')
    registry = load('build/horizon-finish-registry.json'); targets = registry['state_keys']
    require(len(rows) == 6 and len(targets) == 5 and targets == plan['state_keys'] == sorted(key for key, value in stores['prior_h14_attempts'].items() if value['status'] == 'budget_rejected') and rows == registry['rows'] and
            all(stores[name] == registry[name] for name in finish.PRIOR_TABLES), 'Exact five targets or six original references/ownership changed')
    labels14 = finish.combined_labels(stores); new_labels = {**stores['tier1_labels'], **stores['tier2_labels']}
    require(set(new_labels) == set(targets) and len(new_labels) == 5 and set(labels14) == set(rows) and len(stores['prior_h14_labels']) == 1 and len(stores['h12_references']) == 6,
            'Five new and one reused requested H14 labels are not fully covered')
    interpreted = comparison(rows, stores, targets)
    audited = final['evidence']['production_teacher_audit']
    require(all(audited[name] == manifest[name] for name in finish.COUNT_FIELDS), 'Independent exhaustive production counters differ')
    history = [report for tier in ('tier1', 'tier2') for attempt in stores[tier+'_attempts'].values() for report in attempt['history']]
    paths = sum(1+len(report['candidates']) for report in history)
    choices = sum(len(report['pv'])+sum(len(candidate['pv']) for candidate in report['candidates']) for report in history)
    require(audited['new_history_reports'] == len(history) and audited['literal_new_PV_paths'] == paths and audited['literal_new_PV_choices'] == choices and
            audited['tier2_eligible_keys'] == [key for key in targets if stores['tier1_attempts'][key]['status'] == 'budget_rejected'] and
            all(audited[name] is True for name in ('all_six_original_rows_and_prior_tables_exact', 'all_five_target_initial_attempts_present', 'all_fallback_eligibility_and_caps_verified', 'all_overlap_Q_best_semantics_match', 'fresh_check_cost_separate_from_production')), 'Independent new histories, literal PVs or conditional eligibility differ')
    fresh = audited['fresh_bounded_check']; winner = 'tier2' if targets[0] in stores['tier2_labels'] else 'tier1'; saved = stores[winner+'_attempts'][targets[0]]
    require(fresh['state_key'] == targets[0] and fresh['requested_depth'] == 14 and fresh['node_cap'] == saved['node_budget'] and 0 < fresh['actual_nodes'] <= saved['node_budget'] and
            fresh['last_completed_depth'] == 14 and fresh['all_history_Q_best_semantics_equal'] is True, 'Independent bounded fresh H14 comparison differs')
    facts = final['evidence']['forensic_audit']; depth_rows = [entry for case in interpreted['cases'] for entry in case['depth_evidence'].values()]
    lines = [line for entry in depth_rows for line in entry['lines'].values()]; steps = [step for line in lines for step in line['steps']]
    require(facts['cases'] == facts['literal_model_forwards'] == 6 and facts['depth_evidence_rows'] == len(depth_rows) and facts['literal_lines'] == len(lines) and facts['literal_PV_choices'] == len(steps) and
            facts['literal_capture_events'] == sum(len(step['capture_events']) for step in steps) and facts['literal_bonus_choices'] == sum(step['bonus_choice'] for step in steps) and facts['counts'] == interpreted['counts'] and
            all(facts[name] is True for name in ('all_regrets_model_errors_classifications_and_denominators_rederived', 'all_source_job_budget_depth_provenance_verified', 'all_literal_store_capture_bonus_pass_terminal_leaf_cutoff_and_common_prefix_facts_verified', 'completed_zero_teacher_calls', 'completed_bytes_mtimes_unchanged', 'forensics_zero_teacher_calls', 'forensics_bytes_mtimes_unchanged', 'all2314_prior_hashes_current')), 'Independent six-root forensic facts or denominator proof differ')
    rerun = load('build/horizon-finish-root-verification.json')
    require(rerun['schema'] == 'fixed-model-horizon-finish-root-verification-v1' and rerun['passed'] is True and rerun['prior_inputs_preserved'] == 2314 and rerun['completed_artifacts_bytes_mtimes_unchanged'] is True and rerun['before'] == rerun['after'] and
            rerun['native_teacher_calls'] == 0 and rerun['training_performed'] is False and rerun['model_sha256'] == finish.MODEL_SHA and rerun['producer_source_sha256'] == pin('congklak/horizon_finish.py') and rerun['implementation_sha256'] == pin('build/horizon-finish-implementation.json') and
            rerun['analysis_source_sha256'] == pin('tools/analyze-horizon-finish.py') and rerun['verifier_source_sha256'] == pin('tools/verify-horizon-finish-rerun.py') and rerun['baseline_sha256'] == pin('build/horizon-finish-baseline.json'), 'Completed zero-search producer/comparison rerun differs')
    require(set(rerun['after']) == {'data/horizon-finish-v1/work.sqlite3', 'data/horizon-finish-v1/manifest.json', 'data/horizon-finish-v1/writer.lock', 'build/horizon-finish-result.json', 'build/horizon-finish-comparison.json'}, 'Completed five-artifact stability coverage differs')
    stable(rerun['after'])
    for path in ('congklak/horizon_finish_progress.py', 'tools/horizon-finish-dashboard.html', 'docs/HORIZON_FINISH_SCOPE.md', 'docs/HORIZON_FINISH_RESULTS.md', 'docs/HORIZON_FINISH_INDEPENDENT_AUDIT.md', 'build/horizon-finish-rerun.log', 'tools/analyze-horizon-probe.py'):
        pin(path)
    raw = retained(); current(dict(INPUTS)); require('torch' not in sys.modules, 'CPU-only finish recorder imported Torch')
    ids = author_ids+independent_ids; require(len(ids) == len(set(ids)), 'Aggregate unique method coverage differs')
    return dict(schema='horizon-finish-acceptance-v1', passed=True, engineering_passed=True, scope_finished=True, tests_run=len(ids), failures=0, errors=0, skips=0, covered_checks=ids,
        groups=dict(author=dict(tests_run=len(author_ids), methods=author_ids), independent=dict(tests_run=len(independent_ids), methods=independent_ids)),
        prior_inputs_preserved=dict(prior_accepted=2312, prior_plus_acceptance_rerun=2314, all_current_hashes_match=True), implementation_sha256=pin('build/horizon-finish-implementation.json'), model_file_sha256=finish.MODEL_SHA,
        target_roots=5, comparison_roots=6, new_H14_complete=5, reused_H14_complete=1, unresolved_target_roots=0, counts=interpreted['counts'], case_classifications=interpreted['case_classifications'],
        tier_counts={name: manifest[name] for name in ('tier1_attempts', 'tier1_labels', 'tier1_rejected', 'tier2_eligible', 'tier2_attempts', 'tier2_labels', 'tier2_rejected')},
        total_new_production_nodes=manifest['total_new_nodes'], retained_completed_iteration_report_entries=manifest['auxiliary_completed_reports'],
        all_original_roots_ownership_and_references_preserved=True, independent_complete=True, completed_zero_teacher_calls=True, completed_bytes_mtimes_unchanged=True,
        actual_initial_and_fallback_process_kill_barriers=actual_barriers, recovery_fixture_caps_distinct_from_production=True,
        investigation_only=True, data_ingested=False, training_performed=False, model_changed=False, active_weights_changed=False, sampling_weights_applied=False,
        pruning_or_deletion_performed=False, unauthorized_budget_expansion=False, default_promotion=False, sealed_final_test_consumed=False, strength_improvement_claimed=False,
        convergence_proved=False, full_game_solved=False, energy_measured=False, torch_imported=False,
        scope='All five formerly capped requested H14 roots completed under the authorized two-tier profile; unchanged old sixth H14 and six H12 labels form the combined comparison. No training or convergence claim.',
        retained_recovery_and_diagnostic_evidence=raw, inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args(); report = build_report()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(report)+'\n', 'Previously sealed horizon-finish acceptance differs')
    if not args.check_only:
        finish.publish(OUTPUT, report)
    print(config.canonical(dict(event='horizon_finish_acceptance_checked' if args.check_only else 'horizon_finish_accepted', passed=True,
        tests_run=report['tests_run'], inputs=len(INPUTS), new_H14_complete=5, combined_H14_complete=6, acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(config.canonical(dict(event='horizon_finish_record_refused', error=str(exc))), flush=True)
        raise SystemExit(2)


