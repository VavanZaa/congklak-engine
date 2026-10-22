"""Aggregate closed data/label producers, recovery checks and independent review."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as prior

OUTPUT = ROOT / 'build/midgame-pilot-acceptance.json'
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing acceptance input: ' + str(path))
    key = path.relative_to(ROOT).as_posix()
    INPUTS[key] = prior.sha(path)
    return INPUTS[key]


def load(name):
    pin(name)
    return prior.read(ROOT / name)


def current(mapping):
    for name, digest in mapping.items():
        require(pin(name) == digest, 'Acceptance input changed: ' + name)


def group(path, source, partial=False):
    body = load(path)
    require(body['passed'] is True and body['errors'] == body['failures'] == body['skips'] == 0,
            'Executed group failed: ' + path)
    pin(source)
    methods = {}
    for cls in ast.parse((ROOT/source).read_text(encoding='utf-8')).body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    require(node.name not in methods, 'Duplicate audit method')
                    methods[node.name] = source + '::' + cls.name + '::' + node.name
    names = body.get('covered_methods', body.get('methods'))
    require(names and len(names) == len(set(names)) == body['tests_run'] and set(names) <= set(methods),
            'Executed/source method coverage differs: ' + path)
    if not partial:
        require(set(names) == set(methods), 'Incomplete source method coverage: ' + path)
    current(body['inputs'])
    return body, [methods[name] for name in names]


def journal_rows(path, table, key='state'):
    pin(path)
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT '+key+',body,hash FROM '+table+' ORDER BY '+key).fetchall()
    for _, text, digest in rows:
        require(config.digest(json.loads(text)) == digest, 'Retained recovery row corrupt')
    return {row[0]:row for row in rows}


def physical_marker(probe, marker):
    require(load(probe/'marker.json') == marker, 'Helper recovery marker differs')
    committed = probe/'declared-marker.json'
    if committed.exists():
        require(load(committed) == marker, 'Published producer recovery marker differs')
    log = probe/'worker.log'
    pin(log)
    events = []
    for line in log.read_text(encoding='utf-8').splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    require(marker in events, 'Recovery barrier missing from actual worker stdout')
    # A stdout-driven kill can interrupt the subsequent atomic producer marker write.
    # Its retained .tmp file is uncommitted evidence, not a required publication.


def physical_recovery(preflight, labeling):
    folder = (ROOT/preflight['retained_directory']).resolve()
    require(folder.is_relative_to(ROOT/'build'), 'Recovery folder escaped build')
    for record in preflight['evidence']['live_collection_recovery']['tests']:
        event = record['event']
        require(event in ('midgame_collection_precommit', 'midgame_collection_committed'), 'Unknown collection barrier')
        probe = folder/'collection'/(event+'-probe')
        physical_marker(probe, record['marker'])
        prefix = journal_rows(probe/'validated-prefix.sqlite3', 'families', 'idx')
        completed = journal_rows(folder/'collection'/event/'work.sqlite3', 'families', 'idx')
        expected = 1 if event.endswith('_precommit') else 2
        require(list(prefix) == list(range(expected)) and record['committed_prefix'] == expected and
                len(completed) == 8 and {k:completed[k] for k in prefix} == prefix,
                'Retained physical collection prefix differs')
    for record in preflight['evidence']['live_label_recovery']['tests']:
        stage, event = record['stage'], record['event']
        require(stage in ('h8','h10','h12') and event in
                ('midgame_label_'+stage+'_precommit', 'midgame_label_'+stage+'_committed'), 'Unknown label barrier')
        parent = folder/'labels'
        probe = parent/(event+'-probe')
        physical_marker(probe, record['marker'])
        seed = load(record['seed_record'])
        require(pin(seed['source_journal']) == seed['source_journal_sha256'] and
                seed['current_stage'] == stage and seed['seeded_current_prefix'] == seed['new_target_index'] == 1 and
                seed['no_seeded_work_claimed_as_fresh'] is True and
                (ROOT/seed['source_journal']).resolve() == (parent/'fresh/work.sqlite3').resolve() and
                (ROOT/seed['seed_journal']).resolve() == (parent/event/'work.sqlite3').resolve(),
                'Independent same-budget fixture seed provenance differs')
        # seed_journal_sha256 identifies the historical pre-kill seed, not the resumed journal.
        for table in ('roots', *labeling.TABLES):
            prefix = journal_rows(probe/'validated-prefix.sqlite3', table)
            completed = journal_rows(parent/event/'work.sqlite3', table)
            require({k:completed[k] for k in prefix} == prefix, 'Physical committed label prefix changed')
            if table == stage+'_attempts':
                expected = 1 if event.endswith('_precommit') else 2
                key = record['marker']['state_key']
                require(len(prefix) == record['committed_attempt_rows'] == expected and
                        len(completed) == (2 if stage == 'h12' else 3) and
                        ((key in prefix) is event.endswith('_committed')),
                        'Physical pre/postcommit label target differs')


def build():
    from congklak import midgame_data_pilot as collection, midgame_label_pilot as labeling
    INPUTS.clear()
    pin('tools/record-midgame-pilot.py')
    pin('tools/prepare-midgame-pilot-baseline.py')
    baseline = load('build/midgame-pilot-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2639,
            'Prior baseline incomplete')
    current(baseline['inputs'])
    collection.admit()
    labeling.admit()
    fresh, collected = collection.load_rows()
    rows, stores, labeled = labeling.load_rows()
    require(collected['families'] == 512 and len(fresh) <= 50,
            'Bounded collection differs')
    cohort = Counter(r['cohort'] for r in fresh.values())
    require(set(cohort) <= {'training_candidate', 'validation'} and cohort['training_candidate'] <= 40 and
            cohort['validation'] <= 10, 'Fresh ownership/count differs')
    require({k for k, r in rows.items() if r['study_role'] == 'fresh'} == set(fresh),
            'Fresh label projection differs')
    diagnostic = {k for k, r in rows.items() if r['study_role'] == 'diagnostic'}
    require(len(diagnostic) == 9 and len(rows) == len(fresh) + 9 and not diagnostic & set(fresh),
            'Diagnostic label ownership differs')
    for key, root in fresh.items():
        require(rows[key]['root'] == root and rows[key]['cohort'] == root['cohort'] and
                root['legal_mask'].bit_count() >= 2, 'Fresh source/ownership differs')
    require(all(rows[key]['cohort'] == 'diagnostic' for key in diagnostic), 'Diagnostic cohort became trainable')
    for stage in ('h8', 'h10', 'h12'):
        expected = diagnostic if stage == 'h12' else set(rows)
        require(set(stores[stage+'_attempts']) == expected and set(stores[stage+'_labels']) <= expected,
                'Requested attempt/label denominator differs')
    work = sum(a['work']['nodes'] for stage in ('h8', 'h10', 'h12') for a in stores[stage+'_attempts'].values())
    require(work <= 503000000 and len(stores['old_h8_references']) == 1,
            'Teacher cap or old reference differs')
    for module in (collection, labeling):
        for path in (module.REGISTRY, module.PLAN, module.IMPLEMENTATION, module.OUTPUT,
                     module.DEFAULT/'work.sqlite3', module.DEFAULT/'manifest.json'):
            pin(path)
        implementation = load(module.IMPLEMENTATION)
        current(implementation['sources'])
    registry = load(collection.REGISTRY)
    require(registry['latest_arena_game_count'] == 144 and len(registry['base_ordinary_development']) == 259 and
            registry['historical_role_counts'] == dict(training_candidate=61, validation=14, diagnostic=8) and
            registry['historical_roles_rewritten'] is False and registry['future_protection_overrides_eligibility'] is True,
            'Role-aware evaluation protection differs')
    comparison = load('build/midgame-label-pilot-comparison.json')
    review = load('build/midgame-pilot-weight-review.json')
    for report in (comparison, review):
        require(report['passed'] is True, 'Derived review failed')
        current(report['inputs'])
    require(review['sampling_changed'] is review['loss_weights_applied'] is review['training_performed'] is False and
            review['no_auxiliary_fallback'] is True and review['requested_H10_only'] is True,
            'Offline weight review changed training')
    trainable = {k for k, r in fresh.items() if r['cohort'] == 'training_candidate' and k in stores['h10_labels']}
    require({r['state_key'] for r in review['candidate_rows']} == trainable,
            'Offline proposal admitted protected or incomplete roots')
    owner_data, data_ids = group('build/midgame-data-pilot-author.json', 'tests/midgame_data_pilot_author.py')
    owner_labels, label_ids = group('build/midgame-label-pilot-author.json', 'tests/midgame_label_pilot_author.py')
    audit, audit_ids = group('build/midgame-pilot-audit.json', 'tests/midgame_pilot_independent_audit.py')
    independent_parts = []
    if audit.get('groups'):
        executed_ids = []
        for item in audit['groups']:
            require(pin(item['path']) == item['sha256'], 'Independent execution group binding differs')
            part, part_ids = group(item['path'], 'tests/midgame_pilot_independent_audit.py', partial=True)
            require(part['tests_run'] == item['tests_run'], 'Independent group count differs')
            independent_parts.append(part)
            executed_ids += part_ids
        require(len(executed_ids) == len(set(executed_ids)) == len(audit_ids) and set(executed_ids) == set(audit_ids),
                'Independent executed group union differs')
    recovery = audit['evidence']['live_collection_recovery']
    require(recovery['actual_kills'] is True and len(recovery['tests']) == 2 and
            all(t['prefix_retained'] is t['completed_bytes_mtimes_unchanged'] is True for t in recovery['tests']),
            'Actual collection interruption coverage incomplete')
    label_recovery = audit['evidence']['live_label_recovery']
    require(label_recovery['actual_kills'] is True and len(label_recovery['tests']) == 6 and
            {t['event'] for t in label_recovery['tests']} ==
            {'midgame_label_'+s+'_'+b for s in ('h8','h10','h12') for b in ('precommit','committed')} and
            all(t[k] is True for t in label_recovery['tests'] for k in (
                'actual_second_target_executed_and_killed', 'exact_prefix_preserved',
                'fresh_semantics_and_deterministic_work_equal', 'completed_zero_teacher_calls_in_API_and_CLI',
                'completed_bytes_mtimes_unchanged')),
            'Actual three-stage label interruption coverage incomplete')
    preflights = [part for part in independent_parts if 'live_label_recovery' in part['evidence']]
    require(len(preflights) == 1, 'Missing unique source-bound recovery execution group')
    physical_recovery(preflights[0], labeling)
    rerun = load('build/midgame-pilot-root-verification.json')
    require(rerun['passed'] is True and rerun['before'] == rerun['after'] and
            rerun['completed_bytes_mtimes_unchanged'] is True and rerun['prior_inputs_preserved'] == 2639 and
            all(rerun[k] == 0 for k in ('new_native_teacher_searches', 'new_neural_searches', 'new_PUCT_searches',
                'new_training_runtime_activations', 'new_training_calls', 'new_strength_decisions')) and
            rerun['torch_imported'] is False, 'Completed producer/review rerun differs')
    current(rerun['sources'])
    for name, sig in rerun['after'].items():
        path = ROOT/name
        require(pin(name) == sig['sha256'] and path.stat().st_size == sig['size'] and
                path.stat().st_mtime_ns == sig['mtime_ns'], 'Completed durable artifact changed: ' + name)
    for name in ('congklak/midgame_pilot_progress.py', 'tools/midgame-pilot-dashboard.html',
                 'docs/MIDGAME_PILOT_SCOPE.md', 'docs/MIDGAME_DATA_PILOT_SCOPE.md',
                 'docs/MIDGAME_PILOT_RESULTS.md', 'docs/MIDGAME_PILOT_INDEPENDENT_AUDIT.md',
                 'build/midgame-pilot-completed-rerun.log'):
        pin(name)
    retained = []
    for proof in (owner_data, owner_labels, audit, *independent_parts):
        if proof.get('retained_directory'):
            directory = (ROOT / proof['retained_directory']).resolve()
            require(directory.is_relative_to(ROOT/'build'), 'Retained evidence escaped build')
            for path in sorted(directory.rglob('*')):
                if path.is_file() and not path.name.endswith(('-wal', '-shm')):
                    pin(path)
                    retained.append(path.relative_to(ROOT).as_posix())
    smoke = ROOT/'build/midgame-label-comparison-fixture-smoke.json'
    if smoke.exists():
        proof = load(smoke)
        if proof.get('inputs'):
            current(proof['inputs'])
        retained.append(smoke.relative_to(ROOT).as_posix())
    for pattern in ('midgame-*prefreeze*', 'midgame-*failure*', 'midgame-*failed*', 'midgame-*diagnostic*'):
        for path in sorted((ROOT/'build').glob(pattern)):
            if path.is_file():
                pin(path)
                retained.append(path.relative_to(ROOT).as_posix())
    ids = data_ids + label_ids + audit_ids
    require(len(ids) == len(set(ids)), 'Duplicate aggregate checks')
    require('torch' not in sys.modules, 'Acceptance imported Torch')
    current(dict(INPUTS))
    return dict(schema='midgame-pilot-acceptance-v1', passed=True, tests_run=len(ids),
        errors=0, failures=0, skips=0, covered_checks=ids,
        groups=dict(collection_owner=len(data_ids), label_owner=len(label_ids), independent=len(audit_ids)),
        prior_inputs_preserved=2639, families=512, fresh_roots=len(fresh),
        fresh_cohorts=dict(sorted(cohort.items())), protected_diagnostics=9,
        protected_latest_games=144, production_teacher_nodes=work,
        requested_stage_counts={s:dict(attempts=len(stores[s+'_attempts']), labels=len(stores[s+'_labels']),
            capped=len(stores[s+'_attempts'])-len(stores[s+'_labels'])) for s in ('h8','h10','h12')},
        eligible_complete_H10_training_candidates=len(trainable), policy_and_models_fixed=True,
        control_model_sha256=pin('data/quality-train-v1/model.json'),
        pilot_model_sha256=pin('data/value-pilot-v1/model.json'),
        actual_collection_kills=2, actual_label_stage_kills=6,
        completed_bytes_mtimes_unchanged=True, completed_no_new_teacher_search=True,
        new_training=0, new_strength_games=0, sampling_applied=False,
        source_data_pruned=False, sealed_final_test_consumed=False, default_promotion=False,
        independent_complete=True, retained_diagnostics=sorted(set(retained)),
        scope='Bounded fresh midgame data plus diagnostic target study; finite horizons, offline sampling proposals, no training or strength claim.',
        inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    body = build()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(body)+'\n', 'Immutable acceptance differs')
    if not args.check_only:
        prior.publish(OUTPUT, body)
    print(config.canonical(dict(event='midgame_pilot_acceptance_checked' if args.check_only else 'midgame_pilot_accepted',
        passed=True, tests_run=body['tests_run'], inputs=len(INPUTS), acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    main()
