"""Seal the exact-six H12/H14 investigation from current audited evidence."""
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
from congklak import config, horizon_probe as probe
from congklak.native import Native

OUTPUT = ROOT / 'build/horizon-probe-acceptance.json'
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
    for stage in ('h12', 'h14'):
        evidence = report['evidence']['live_'+stage.upper()+'_recovery']
        require(evidence['actual_kills'] is True and evidence['stage'] == stage and evidence['raw_seed_edits_markers_backups_retained'] is True, 'Actual retained horizon process kills are absent')
        records = evidence['tests']
        require(len(records) == 2 and {row['event'] for row in records} == {'horizon_probe_'+stage+'_'+suffix for suffix in ('precommit', 'committed')}, 'Horizon pre/postcommit coverage differs')
        for row in records:
            expected = 1 if row['event'].endswith('precommit') else 2
            require(row['stage'] == stage and row['committed_prefix'] == expected and row['targets_after_resume'] == 2 and
                    all(row[name] is True for name in ('prefix_seeded_from_actual_independent_fresh_fixture', 'prefix_exactly_retained', 'fresh_semantics_and_counters_equal', 'completed_zero_teacher_calls', 'bytes_mtimes_unchanged')),
                    'Killed-prefix seed/recovery semantics differ')
            base = Path(report['retained_directory']) / (stage+'-kills') / (row['event']+'-probe')
            marker = load(base/'marker.json')
            require(marker == row['marker'] and marker['event'] == row['event'], 'Actual horizon kill marker differs')
            seed = load(row['seed_record']); source = Path(report['retained_directory'])/'shared-independent-fresh'
            require(seed['source'] == source.as_posix() and seed['source_journal_sha256'] == pin(source/'work.sqlite3') and seed['completion_and_export_removed'] is True,
                    'Seeded-prefix source/edit provenance differs')
            backup = base/'validated-prefix.sqlite3'; pin(backup)
            with closing(sqlite3.connect((ROOT/backup).resolve().as_uri()+'?mode=ro', uri=True)) as db:
                require(db.execute('SELECT COUNT(*) FROM '+stage+'_attempts').fetchone()[0] == expected, 'Killed-prefix physical backup count differs')
                retained = db.execute('SELECT body,hash FROM '+stage+'_attempts WHERE state=?', (seed['retained_first_target'],)).fetchone()
            with closing(sqlite3.connect((ROOT/source/'work.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)) as db:
                original = db.execute('SELECT body,hash FROM '+stage+'_attempts WHERE state=?', (seed['retained_first_target'],)).fetchone()
            require(retained is not None and retained == original, 'Seeded first target was not retained exactly')
            barriers += 1
    return barriers


def prefix(a, b):
    common = 0
    for x, y in zip(a, b):
        if (x['actor'], x['pit']) != (y['actor'], y['pit']):
            break
        common += 1
    return dict(shared_choices=common, first_different_choice=common+1 if common < min(len(a), len(b)) else None,
                shorter_line_is_prefix=common == min(len(a), len(b)))


def forensics(rows, h8, h10, h12, h14, a12, a14):
    report = load('build/horizon-probe-forensics.json'); current(report['inputs'])
    require(report['schema'] == 'fixed-control-horizon-probe-forensics-v1' and report['passed'] is True and
            report['fixed_model_sha256'] == probe.MODEL_SHA and report['rules_digest'] == config.RULES_DIGEST, 'Forensic report identity differs')
    cases = report['cases']; indexed = {case['state_key']: case for case in cases}
    require(len(cases) == len(indexed) == 6 and list(indexed) == sorted(rows), 'Forensics does not cover the exact sorted six roots')
    for index, (key, root) in enumerate(sorted(rows.items()), 1):
        case = indexed[key]; pair = key in h12 and key in h14
        equal = set(h12[key]['label']['best_moves']) == set(h14[key]['label']['best_moves']) if pair else None
        classification = ('same_best_set_at_H12_H14' if equal else 'different_best_sets_at_H12_H14') if pair else 'requested_comparison_incomplete'
        require(case['case_id'] == 'D'+str(index) and case['state'] == root['state'] and case['family'] == root['family'] and
                case['cohort'] == root['cohort'] and case['source_role'] == root['study_role'] and case['investigation_only'] is True, 'Forensic case identity/ownership differs')
        require(case['requested_status'] == dict(h12=a12[key]['status'], h14=a14[key]['status']) and case['requested_complete_pair'] == pair and
                case['same_best_set_H12_H14'] == equal and case['classification'] == classification and
                case['requested_depths_available'] == [depth for depth, labels in ((12, h12), (14, h14)) if key in labels], 'Requested completion/classification differs')
        require(type(case['model_value_player0']) in (int, float) and math.isfinite(case['model_value_player0']) and
                type(case['raw_policy_chosen']) is int and 0 <= case['raw_policy_chosen'] < 7 and root['legal_mask'] & (1 << case['raw_policy_chosen']), 'Invalid fixed-model case output')
        sources = {8: ('reference', 'old_h8', 8, None, h8[key]['report']), 10: ('reference', 'old_h10', 10, None, h10[key]['report'])}
        for depth, labels in ((12, h12), (14, h14)):
            if key in labels:
                sources[depth] = ('requested_label', 'h'+str(depth), depth, probe.CONTRACT['h'+str(depth)]['node_budget'], labels[key]['report'])
        requested = dict(sources)
        for stage, attempts in (('h14', a14), ('h12', a12)):
            for item in attempts[key]['history']:
                depth = item['choice_depth']
                if depth >= 8 and depth not in sources:
                    sources[depth] = ('auxiliary_completed_iteration', stage, probe.CONTRACT[stage]['depth'], probe.CONTRACT[stage]['node_budget'], item)
        require(set(case['depth_evidence']) == {str(depth) for depth in sources}, 'Auxiliary actual-depth coverage differs')
        require(case['auxiliary_completed_depths'] == sorted(depth for depth, source in sources.items() if source[0] == 'auxiliary_completed_iteration'), 'Auxiliary provenance depth list differs')
        b8, b10 = h8[key]['label']['best_moves'], h10[key]['label']['best_moves']
        require(case['reference_H8_best_moves'] == b8 and case['reference_H10_best_moves'] == b10 and not set(b8) & set(b10), 'Old disjoint best-set references changed')
        sign = 1 if root['state']['controller'] == 0 else -1
        for depth, source in sources.items():
            kind, job, requested_depth, budget, original = source
            item = case['depth_evidence'][str(depth)]; best = original['best_moves']; value = original['depth_result']['value_player0']
            scores = {entry['pit']: entry['depth_result']['value_player0'] for entry in original['candidates']}
            require(item['actual_depth'] == depth and item['source_kind'] == kind and item['source_job'] == job and item['source_requested_depth'] == requested_depth and
                    item['source_node_budget'] == budget and item['source_report_sha256'] == config.digest(original), 'Forensic source job/depth/report identity differs')
            require(item['target_player0'] == value and item['best_moves'] == best and item['display_best_moves'] == [action+1 for action in best] and
                    item['action_scores'] == [dict(pit=action, display_pit=action+1, value_player0=score) for action, score in scores.items()], 'Forensic Q/best/pit display differs')
            expected = dict(H8_lexicographic_choice_regret=sign*(value-scores[min(b8)]), H10_lexicographic_choice_regret=sign*(value-scores[min(b10)]),
                H8_best_tie_regret_min=min(sign*(value-scores[action]) for action in b8), H8_best_tie_regret_max=max(sign*(value-scores[action]) for action in b8),
                H10_best_tie_regret_min=min(sign*(value-scores[action]) for action in b10), H10_best_tie_regret_max=max(sign*(value-scores[action]) for action in b10),
                model_value_absolute_error=abs(case['model_value_player0']-value), raw_policy_regret=sign*(value-scores[case['raw_policy_chosen']]))
            require(all(item[name] == value for name, value in expected.items()) and all(value >= 0 for value in expected.values()), 'Forensic actor-sign/model error arithmetic differs')
            actions = sorted(set(b8) | set(b10) | set(best))
            require(set(item['lines']) == {str(action) for action in actions}, 'Forensic PV root-action coverage differs')
            for action in actions:
                line = item['lines'][str(action)]; candidate = next(entry for entry in original['candidates'] if entry['pit'] == action)
                require(line['pit'] == action and line['display_pit'] == action+1 and line['pv'] == candidate['pv'] and line['target_player0'] == scores[action] and
                        line['leaf']['margin_player0'] == scores[action], 'Forensic reported PV/leaf linkage differs')
                for old_depth, refs in ((8, h8), (10, h10)):
                    old_pv = next(entry['pv'] for entry in refs[key]['report']['candidates'] if entry['pit'] == action)
                    require(line['common_prefix_vs_H'+str(old_depth)] == prefix(old_pv, candidate['pv']), 'Forensic PV common-prefix facts differ')
        deepest_requested, deepest_actual = max(requested), max(sources)
        deepest_best = requested[deepest_requested][4]['best_moves']; depths = sorted(sources)
        require(case['deepest_requested_depth'] == deepest_requested and case['deepest_actual_completed_depth'] == deepest_actual and
                case['deepest_requested_best_overlap_H8'] == bool(set(deepest_best) & set(b8)) and case['deepest_requested_best_overlap_H10'] == bool(set(deepest_best) & set(b10)), 'Deepest requested/auxiliary evidence conflated')
        available_equal = (set(sources[12][4]['best_moves']) == set(sources[14][4]['best_moves'])) if 12 in sources and 14 in sources else None
        adjacent = sum(set(sources[a][4]['best_moves']) != set(sources[b][4]['best_moves']) for a, b in zip(depths, depths[1:]) if b == a+1)
        require(case['available_depth12_and14_best_sets_equal'] == available_equal and case['adjacent_actual_depth_best_set_changes'] == adjacent, 'Actual-depth comparison facts differ')
    expected_counts = dict(roots=6, h12_attempts=len(a12), h14_attempts=len(a14), h12_complete=len(h12), h14_complete=len(h14),
        requested_complete_pairs=len(set(h12) & set(h14)), h12_rejected=sum(row['status'] == 'budget_rejected' for row in a12.values()),
        h14_rejected=sum(row['status'] == 'budget_rejected' for row in a14.values()), stable_best_sets_H12_H14=sum(case['same_best_set_H12_H14'] is True for case in cases),
        changed_best_sets_H12_H14=sum(case['same_best_set_H12_H14'] is False for case in cases), requested_incomplete_cases=sum(not case['requested_complete_pair'] for case in cases),
        cases_with_deeper_requested_label=sum(case['deepest_requested_depth'] > 10 for case in cases))
    require(report['counts'] == expected_counts and report['case_classifications'] == dict(sorted(Counter(case['classification'] for case in cases).items())), 'Forensic case-count denominators differ')
    for name in ('training_performed', 'model_changed', 'active_sampling_weights_changed', 'data_ingested', 'teacher_search_performed_by_analysis', 'new_PUCT_or_strength_tests', 'sealed_final_test_consumed'):
        require(report[name] is False, 'Forensics exceeds authorized scope: ' + name)
    return report


def retained():
    result = set()
    for folder in ('build/horizon-probe-audit', 'build/horizon-probe-author', 'build/horizon-probe-smoke-v1'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and not path.name.endswith(('-wal', '-shm')):
                pin(path); result.add(path.relative_to(ROOT).as_posix())
    for pattern in ('horizon-probe-*-feasibility.json', 'horizon-probe-*failure*', 'horizon-probe-*failed*'):
        for path in sorted((ROOT/'build').glob(pattern)):
            if path.is_file():pin(path); result.add(path.relative_to(ROOT).as_posix())
    return sorted(result)


def build_report():
    INPUTS.clear(); pin('tools/record-horizon-probe.py'); pin('tools/prepare-horizon-probe-baseline.py')
    baseline = load('build/horizon-probe-baseline.json'); current(baseline['inputs'])
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2199 and baseline['prior_accepted_inputs'] == 2197, 'Probe prior baseline differs')
    prior = load('build/natural-study-acceptance.json')
    require(prior['passed'] is True and len(prior['inputs']) == 2197 and all(baseline['inputs'].get(name) == digest for name, digest in prior['inputs'].items()), 'Prior sealed natural-study coverage differs')
    plan, implementation = probe.admit(); current(implementation['sources']); current(implementation['prior_inputs'])
    for path in ('build/horizon-probe-plan.json', 'build/horizon-probe-implementation.json', 'build/horizon-probe-registry.json'):
        pin(path)
    require(pin(probe.MODEL) == probe.MODEL_SHA, 'Fixed model changed')
    author, author_ids = group('build/horizon-probe-author.json', 'tests/horizon_probe_author.py', 'methods')
    require(set(author['methods']) == set(methods('tests/horizon_probe_author.py')), 'Author execution coverage is incomplete')
    before, before_ids = group('build/horizon-probe-audit/preflight.json', 'tests/horizon_probe_independent_audit.py', 'covered_methods')
    final, final_ids = group('build/horizon-probe-audit/final.json', 'tests/horizon_probe_independent_audit.py', 'covered_methods')
    independent_ids = before_ids+final_ids
    require(len(independent_ids) == len(set(independent_ids)) and set(independent_ids) == set(methods('tests/horizon_probe_independent_audit.py').values()), 'Independent executed AST coverage differs')
    audit = load('build/horizon-probe-audit.json'); current(audit['inputs']); pin('build/horizon-probe-audit/collect.py')
    expected_groups = [dict(path=path, sha256=pin(path), tests_run=body['tests_run'], covered_methods=body['covered_methods'])
                       for path, body in (('build/horizon-probe-audit/preflight.json', before), ('build/horizon-probe-audit/final.json', final))]
    require(audit['passed'] is True and audit['tests_run'] == len(independent_ids) and audit['errors'] == audit['failures'] == audit['skips'] == 0 and audit['prior_inputs_preserved'] == 2199 and
            audit['groups'] == expected_groups and audit['evidence'] == {**before['evidence'], **final['evidence']} and set(audit['covered_methods']) == set(before['covered_methods']+final['covered_methods']) and
            len(audit['covered_methods']) == len(independent_ids), 'Combined independent proof differs')
    actual_barriers = recoveries(before)
    checked = before['evidence']['baseline']; selection = before['evidence']['exact_selection']
    require(checked['prior_inputs'] == 2199 and checked['old_accepted_inputs'] == 2197 and checked['all_hashes_current'] is True and
            selection['roots'] == 6 and selection['all_original_wrappers_and_H8_H10_items_exact'] is True and selection['investigation_only'] is True and
            all(value is True for value in before['evidence']['refusals_and_eligibility'].values()), 'Independent baseline/reference/auxiliary refusal proof incomplete')
    original = Native.analyze
    def prohibited(*_args, **_kwargs):
        raise ValueError('Acceptance must not execute a teacher search')
    Native.analyze = prohibited
    try:
        rows, h8, h10, h12, h14, a12, a14, manifest = probe.load_rows()
    finally:
        Native.analyze = original
    require(manifest == load('build/horizon-probe-result.json') == load('data/horizon-probe-v1/manifest.json') and manifest['identity']['fixture'] is False and
            len(rows) == len(h8) == len(h10) == len(a12) == len(a14) == 6 and manifest['total_new_nodes'] <= 210000000, 'Closed six-root production publication differs')
    pin('data/horizon-probe-v1/work.sqlite3')
    registry = load('build/horizon-probe-registry.json')
    expected = sorted(case['state_key'] for case in load('build/natural-study-stability.json')['metrics'] if case['best_sets_disjoint'] is True)
    require(list(rows) == expected == plan['state_keys'] == registry['state_keys'] and rows == registry['rows'] and
            h8 == registry['h8_references'] and h10 == registry['h10_references'], 'Exact six selection, ownership or old references changed')
    forensic = forensics(rows, h8, h10, h12, h14, a12, a14)
    audited = final['evidence']['production_teacher_audit']
    for name in ('roots', 'h8_references', 'h10_references', 'h12_labels', 'h12_rejected', 'h14_labels', 'h14_rejected', 'h12_h14_complete_pairs', 'auxiliary_completed_reports', 'total_new_nodes'):
        require(audited[name] == manifest[name], 'Independent requested/history count differs: '+name)
    history = [report for attempts in (a12, a14) for attempt in attempts.values() for report in attempt['history']]
    paths = sum(1+len(report['candidates']) for report in history)
    choices = sum(len(report['pv'])+sum(len(candidate['pv']) for candidate in report['candidates']) for report in history)
    require(audited['history_reports'] == len(history) and audited['literal_PV_paths'] == paths and audited['literal_PV_choices'] == choices and
            all(audited[name] is True for name in ('all_six_original_rows_and_references_exact', 'all_H14_roots_attempted', 'requested_and_auxiliary_types_verified', 'all_overlap_Q_best_semantics_match', 'fresh_check_cost_separate_from_production')),
            'Independent complete-history/PV coverage differs')
    fresh = audited['fresh_bounded_check']
    require(fresh['state_key'] == next(iter(rows)) and fresh['requested_depth'] == 12 and fresh['node_cap'] == 10000000 and 0 < fresh['actual_nodes'] <= 10000000 and
            fresh['last_completed_depth'] == a12[fresh['state_key']]['completed_depth'] and fresh['all_history_Q_best_semantics_equal'] is True, 'Independent bounded fresh verification differs')
    facts = final['evidence']['forensic_audit']; depth_rows = [entry for case in forensic['cases'] for entry in case['depth_evidence'].values()]
    lines = [line for entry in depth_rows for line in entry['lines'].values()]; steps = [step for line in lines for step in line['steps']]
    require(facts['cases'] == facts['literal_model_forwards'] == 6 and facts['depth_evidence_rows'] == len(depth_rows) and facts['literal_lines'] == len(lines) and facts['literal_PV_choices'] == len(steps) and
            facts['literal_capture_events'] == sum(len(step['capture_events']) for step in steps) and facts['literal_bonus_choices'] == sum(step['bonus_choice'] for step in steps) and facts['requested_counts'] == forensic['counts'] and
            all(facts[name] is True for name in ('all_regrets_model_errors_and_case_classifications_rederived', 'all_source_job_budget_depth_provenance_verified', 'all_literal_store_capture_bonus_pass_terminal_leaf_and_cutoff_facts_verified', 'all_common_prefix_comparisons_verified', 'requested_pairs_distinct_from_auxiliary_availability', 'completed_zero_teacher_calls', 'completed_bytes_mtimes_unchanged', 'forensics_bytes_mtimes_unchanged', 'all2199_prior_hashes_current')),
            'Independent forensic literal facts or eligible denominators differ')
    production = load('build/horizon-probe-production-rerun.json')
    require(production['schema'] == 'horizon-probe-completed-production-rerun-v1' and production['passed'] is True and production['completed_bytes_mtimes_unchanged'] is True and production['native_teacher_calls'] == 0 and
            production['source_sha256'] == pin('build/horizon-probe-production-rerun.py') and production['producer_sha256'] == pin('congklak/horizon_probe.py') and
            production['implementation_sha256'] == pin('build/horizon-probe-implementation.json'), 'Completed zero-search producer rerun differs')
    require(all(production[name] == manifest[name] for name in ('roots', 'h8_references', 'h10_references', 'h12_labels', 'h12_rejected', 'h14_labels', 'h14_rejected', 'h12_h14_complete_pairs', 'auxiliary_completed_reports', 'total_new_nodes')), 'Completed producer witness counts differ')
    require(set(production['artifacts']) == {'data/horizon-probe-v1/work.sqlite3', 'data/horizon-probe-v1/manifest.json', 'data/horizon-probe-v1/writer.lock', 'build/horizon-probe-result.json'}, 'Completed producer stability coverage differs')
    stable(production['artifacts'])
    root_rerun = load('build/horizon-probe-root-verification.json')
    require(root_rerun['schema'] == 'fixed-model-horizon-probe-root-verification-v1' and root_rerun['passed'] is True and root_rerun['prior_inputs_preserved'] == 2199 and root_rerun['completed_artifacts_bytes_mtimes_unchanged'] is True and root_rerun['before'] == root_rerun['after'] and
            root_rerun['analysis_source_sha256'] == pin('tools/analyze-horizon-probe.py') and root_rerun['verifier_source_sha256'] == pin('tools/verify-horizon-probe-rerun.py') and
            root_rerun['baseline_sha256'] == pin('build/horizon-probe-baseline.json') and root_rerun['training_performed'] is False and root_rerun['model_sha256'] == probe.MODEL_SHA and root_rerun['native_teacher_calls'] == 0,
            'Completed forensic rerun proof differs')
    require(set(root_rerun['after']) == {'data/horizon-probe-v1/work.sqlite3', 'data/horizon-probe-v1/manifest.json', 'build/horizon-probe-result.json', 'build/horizon-probe-forensics.json'}, 'Forensic stability artifact coverage differs')
    stable(root_rerun['after'])
    for path in ('congklak/horizon_probe_progress.py', 'tools/horizon-probe-dashboard.html', 'docs/HORIZON_PROBE_SCOPE.md', 'docs/HORIZON_PROBE_RESULTS.md', 'docs/HORIZON_PROBE_INDEPENDENT_AUDIT.md', 'build/horizon-probe-forensics-rerun.log'):
        pin(path)
    raw = retained(); current(dict(INPUTS)); require('torch' not in sys.modules, 'CPU-only probe recorder imported Torch')
    ids = author_ids+independent_ids; require(len(ids) == len(set(ids)), 'Aggregate executed method duplicates')
    return dict(schema='horizon-probe-acceptance-v1', passed=True, engineering_passed=True, tests_run=len(ids), errors=0, failures=0, skips=0, covered_checks=ids,
        groups=dict(author=dict(tests_run=len(author_ids), methods=author_ids), independent=dict(tests_run=len(independent_ids), methods=independent_ids)),
        prior_inputs_preserved=dict(prior_accepted=2197, prior_plus_acceptance_rerun=2199, all_current_hashes_match=True),
        implementation_sha256=pin('build/horizon-probe-implementation.json'), model_file_sha256=probe.MODEL_SHA, roots=6, counts=forensic['counts'],
        case_classifications=forensic['case_classifications'], total_new_production_nodes=manifest['total_new_nodes'], retained_completed_iteration_report_entries=manifest['auxiliary_completed_reports'],
        original_roots_ownership_H8_H10_preserved=True, independent_complete=True, completed_zero_teacher_calls=True, completed_bytes_mtimes_unchanged=True,
        actual_horizon_process_kill_barriers=actual_barriers, recovery_prefix_seeded_from_actual_independent_fixture=True,
        investigation_only=True, data_ingested=False, training_performed=False, model_changed=False, active_weights_changed=False, sampling_weights_applied=False,
        pruning_or_deletion_performed=False, budget_expanded=False, default_promotion=False, sealed_final_test_consumed=False, strength_improvement_claimed=False,
        full_game_solved=False, convergence_proved=False, energy_measured=False, torch_imported=False,
        scope='Only the six prior H8/H10 disjoint-best-set roots. Requested H12/H14 completion remains distinct from actual-depth auxiliary histories. No training ingestion or game-theoretic proof.',
        retained_recovery_and_diagnostic_evidence=raw, inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    report = build_report()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(report)+'\n', 'Previously sealed horizon-probe acceptance differs')
    if not args.check_only:
        probe.publish(OUTPUT, report)
    print(config.canonical(dict(event='horizon_probe_acceptance_checked' if args.check_only else 'horizon_probe_accepted', passed=True,
        tests_run=report['tests_run'], inputs=len(INPUTS), requested_complete_pairs=report['counts']['requested_complete_pairs'],
        h12_rejected=report['counts']['h12_rejected'], h14_rejected=report['counts']['h14_rejected'], acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(config.canonical(dict(event='horizon_probe_record_refused', error=str(exc))), flush=True)
        raise SystemExit(2)


