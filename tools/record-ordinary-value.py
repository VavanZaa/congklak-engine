"""Seal executed fixed-policy value, equal-time and conditional data evidence."""
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
from congklak.neural_relay import Model

spec = importlib.util.spec_from_file_location('sealed_selfplay_evidence', ROOT / 'tools/record-selfplay-pilot.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
require, pin, read, current = e.require, e.pin, e.read, e.current
OUTPUT = ROOT / 'build/ordinary-value-acceptance.json'


def kill_records(value):
    if isinstance(value, dict):
        if value.get('actual_process_kill') is True and all(key in value for key in ('probe', 'run', 'marker')):
            yield value
        for child in value.values(): yield from kill_records(child)
    elif isinstance(value, list):
        for child in value: yield from kill_records(child)


def physical_recovery(audit):
    records = list(kill_records(audit['evidence']))
    require(len(records) == len({r['probe'] for r in records}) == 10, 'Actual interruption coverage differs')
    expected_events = {'checkpoint_file_ready', 'checkpoint_committed'}
    expected_events.update('ordinary_value_' + unit + '_' + suffix
            for unit in ('h10', 'benchmark_decision', 'game', 'followup_decision')
            for suffix in ('precommit', 'committed'))
    require({record['event'] for record in records} == expected_events, 'Missing actual kill barrier')
    comparisons = []
    for record in records:
        e.physical_marker(record)
        prefix = Path(record['probe']) / 'validated-prefix.sqlite3'
        pin(prefix)
        run = ROOT / record['run']
        final = run / ('catalog.sqlite3' if (run / 'catalog.sqlite3').is_file() else 'work.sqlite3')
        pin(final.relative_to(ROOT).as_posix())
        with closing(sqlite3.connect((ROOT / prefix).resolve().as_uri() + '?mode=ro', uri=True)) as left, \
             closing(sqlite3.connect(final.resolve().as_uri() + '?mode=ro', uri=True)) as right:
            tables = {r[0] for r in left.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            immutable = {'roots', 'families', 'cases', 'games', 'decisions', 'sp_decisions', 'checkpoints', 'history'}
            immutable.update(stage + suffix for stage in ('h8', 'h10', 'h12') for suffix in ('_attempts', '_labels'))
            checked = {}
            for table in sorted(tables & immutable):
                columns = [r[1] for r in left.execute('PRAGMA table_info(' + table + ')')]
                before = left.execute('SELECT * FROM ' + table).fetchall()
                after = set(right.execute('SELECT * FROM ' + table).fetchall())
                require(set(before) <= after, 'Committed physical prefix changed: ' + table)
                if 'body' in columns and 'hash' in columns:
                    for row in before:
                        require(config.digest(json.loads(row[columns.index('body')])) == row[columns.index('hash')], 'Corrupt retained physical prefix')
                if table == 'checkpoints':
                    for row in before:
                        name, digest = row[columns.index('name')], row[columns.index('sha')]
                        require(pin((run / name).relative_to(ROOT).as_posix()) == digest, 'Committed PT bytes changed')
                checked[table] = len(before)
            require(checked, 'No durable prefix tables verified')
            event = record['event']; marker = record['marker']
            if event.startswith('checkpoint_'):
                expected = [2] if event == 'checkpoint_file_ready' else [2, 4]
                actual = [r[0] for r in left.execute('SELECT step FROM checkpoints ORDER BY step')]
                require(actual == record['committed_prefix_steps'] == expected and marker['step'] == 4,
                        'Physical committed checkpoint steps differ')
                named = run / marker['name']; pin(named.relative_to(ROOT).as_posix())
                catalog_names = {r[0] for r in right.execute('SELECT name FROM checkpoints')}
                require((marker['name'] in catalog_names) is (event == 'checkpoint_committed'),
                        'Uncommitted checkpoint was admitted or committed checkpoint replaced')
                require(record['orphan_checkpoint_retained'] is (event == 'checkpoint_file_ready'),
                        'Orphan evidence label differs')
            else:
                if event.startswith('ordinary_value_h10_'):
                    table = 'h10_attempts'; expected = 1 if event.endswith('_precommit') else 2
                    count = marker['completed']
                elif event.startswith('ordinary_value_followup_'):
                    table = 'sp_decisions'; expected = 1 if event.endswith('_precommit') else 2
                    count = marker['committed_decisions']
                else:
                    table = record['prefix_table']
                    require(table == ('games' if event.startswith('ordinary_value_game_') else 'decisions'),
                            'Unknown benchmark interruption unit')
                    expected = (0 if event.endswith('_precommit') else 1) if table == 'games' else (1 if event.endswith('_precommit') else 2)
                    count = marker['completed']
                require(checked[table] == record['committed_prefix'] == count == expected,
                        'Physical committed prefix differs from its barrier')
            comparisons.append(dict(event=record['event'], probe=record['probe'], committed_prefix_tables=checked))
    return comparisons


def build():
    from congklak import ordinary_value_pilot as learning, ordinary_value_benchmark as benchmark
    from congklak import ordinary_value_followup as followup
    e.INPUTS.clear()
    baseline = read('build/ordinary-value-baseline.json')
    require(baseline['passed'] is True and len(baseline['inputs']) == baseline['inputs_count'] == 3364, 'Wrong prior baseline')
    current(baseline['inputs'])
    checks, groups = [], {}
    reports = (
        ('learning_owner', 'build/ordinary-value-author.json', 'tests/ordinary_value_pilot_author.py'),
        ('benchmark_owner', 'build/ordinary-value-benchmark-author.json', 'tests/ordinary_value_benchmark_author.py'),
        ('independent', 'build/ordinary-value-audit.json', 'tests/ordinary_value_independent_audit.py'))
    audit = None
    for name, report, source in reports:
        body, covered = e.group(report, source); checks.extend(covered); groups[name] = body['tests_run']
        if name == 'independent': audit = body
    require(len(checks) == len(set(checks)), 'Executed checks overlap')
    recovery = physical_recovery(audit)
    roots, stores, labels = learning.load_labels()
    require(len(roots) == 6 and all(len(stores[stage + '_attempts']) == 6 for stage in ('h8', 'h10', 'h12')),
            'All18 requested validation attempts are required')
    label_counts = {stage: len(stores[stage + '_labels']) for stage in ('h8', 'h10', 'h12')}
    manifest, data = learning.load_data()
    require(manifest['training_admission'] is True and manifest['no_validation_or_descendant_gradients'] is True and
            all(manifest['paired_starts'][name] >= 8 for name in ('high', 'low')), 'Training ownership/coverage differs')
    witness = learning.witness(); current(witness['inputs'])
    optimizer = read('build/ordinary-value-audit/model.json')['evidence']['production_optimizer_audit']
    current(optimizer['inputs']); pin(optimizer['path'])
    require(optimizer['passed'] is True and optimizer['checkpoints'] == 10 and optimizer['draws'] == 12800 and
            optimizer['optimizer_parameter_ids'] == [0, 1] and optimizer['trainable_parameters'] == 65 and
            optimizer['frozen_parameters'] == 12807 and
            {item['step'] for item in optimizer['steps']} == set(range(20, 201, 20)) and
            set(optimizer['changed_tensors']) == {'layers.4.weight', 'layers.4.bias'} and
            all(optimizer[key] is True for key in ('all200_optimizer_steps_independently_replayed',
                'all_checkpoint_model_optimizer_sampler_python_torch_rng_raw_bytes_equal',
                'all_frozen_tensors_bit_exact', 'all_dataset_policy_logits_bit_exact',
                'hidden_cache_independently_reconstructed', 'initial_and_final_metrics_independently_rederived',
                'final_export_equals_PT')), 'Independent full optimizer audit differs')
    draw_counts = {name: sum(optimizer['draw_ledger'].get(row['id'], 0) for row in rows)
                   for name, rows in data.items()}
    require(draw_counts == {'high': 3200, 'low': 3200, 'rehearsal': 6400} and
            sum(optimizer['draw_ledger'].values()) == 12800, 'Actual stratified draw allocation differs')
    training = read('data/ordinary-value-pilot-v1/result.json')
    require(training['status'] == 'complete' and training['steps_completed'] == training['selected_step'] == 200 and
            training['draws'] == 12800 and training['frozen_policy_unchanged'] is True and training['promotion'] is False,
            'Fixed-final value training differs')
    contract = benchmark.verify_candidate(Model(learning.SOURCE), Model(learning.DEFAULT / 'model.json'))
    require(contract['trainable_parameters'] == 65 and contract['global_policy_unchanged_by_identical_subgraph'] is True,
            'Frozen policy contract differs')
    closed = benchmark.closed_witness()
    require(closed['games'] == 96 and closed['cases'] == 38 and closed['scalar_measurements'] == 304, 'Equal-time geometry differs')
    result = read('build/ordinary-value-benchmark.json')
    allocation = benchmark.allocation_decision()
    collected, rows = followup.load_run()
    require(collected['training_performed'] is collected['default_promotion'] is collected['future_training_authorized'] is False and
            collected['identity']['branch'] == allocation['branch'], 'Followup deviated from preregistered allocation')
    require(collected['summary']['games'] <= (48 if allocation['branch'] == 'new_starts' else 24 if allocation['branch'] == 'repeat_high' else 0),
            'Conditional collection over budget')
    if allocation['branch'] == 'hold':
        require(collected == dict(schema=followup.SCHEMA, status='held', identity=collected['identity'],
                    summary=dict(games=0, committed_decisions=0), training_performed=False,
                    default_promotion=False, future_training_authorized=False), 'Held followup fields differ')
    rerun = read('build/ordinary-value-root-verification.json')
    require(rerun['passed'] is True and rerun['before'] == rerun['after'] and
            rerun['all_durable_bytes_sizes_mtimes_unchanged'] is True and rerun['torch_imported'] is False and
            all(rerun[key] == 0 for key in ('new_teacher_searches', 'new_optimizer_steps', 'new_benchmark_decisions', 'new_followup_decisions')),
            'Completed no-work proof differs')
    current(rerun['sources'])
    for name, record in rerun['after'].items():
        p = ROOT / name
        require(pin(name) == record['sha256'] and p.stat().st_size == record['size'] and p.stat().st_mtime_ns == record['mtime_ns'], 'Completed durable signature changed')
    for name in ('tools/prepare-ordinary-value-baseline.py', 'tools/record-ordinary-value.py',
                 'tools/verify-ordinary-value-rerun.py', 'tools/verify-ordinary-value-acceptance.py',
                 'tools/verify-ordinary-value-dashboard.py', 'tools/ordinary-value-audit.py',
                 'docs/ORDINARY_VALUE_SCOPE.md', 'docs/ORDINARY_VALUE_PRODUCER.md',
                 'docs/ORDINARY_VALUE_RESULTS.md', 'docs/ORDINARY_VALUE_INDEPENDENT_AUDIT.md',
                 'congklak/ordinary_value_progress.py', 'tools/ordinary-value-dashboard.html',
                 'congklak/ordinary_value_followup.py', 'tools/ordinary-value-followup.py', 'build/ordinary-value-rerun.log'):
        pin(name)
    for path in sorted((ROOT / 'build').glob('ordinary-value-*.json')):
        if path.name in ('ordinary-value-acceptance.json', 'ordinary-value-acceptance-rerun.json', 'ordinary-value-dashboard-final.json'):
            continue
        body = read(path.relative_to(ROOT).as_posix())
        if '-prefreeze-' not in path.name:
            for field in ('inputs', 'sources', 'dependencies', 'science_inputs'):
                if isinstance(body.get(field), dict): current(body[field])
    for folder in (learning.DEFAULT, learning.LABEL_DEFAULT, ROOT / 'data/ordinary-value-benchmark-v1', followup.DEFAULT,
                   ROOT / 'build/ordinary-value-audit'):
        e.durable_tree(folder.relative_to(ROOT).as_posix())
    for pattern in ('ordinary-value-*author*', 'ordinary-value-prefreeze-*', 'ordinary-value-benchmark-prefreeze-*',
                    'ordinary-value-benchmark-frozen-*'):
        for path in sorted((ROOT / 'build').glob(pattern)):
            if path.is_dir(): e.durable_tree(path.relative_to(ROOT).as_posix())
            elif path.is_file(): pin(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'build').glob('ordinary-value-*')):
        if path.is_dir() and path.name != 'ordinary-value-progress':
            e.durable_tree(path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'build/ordinary-value-progress').glob('run-*.json')):
        row = read(path.relative_to(ROOT).as_posix()); require(row['state'] != 'running', 'A worker is still active'); pin(row['log'])
    require(pin(baseline['control_model']) == baseline['control_model_sha256'], 'Control changed')
    body = dict(schema='ordinary-value-acceptance-v1', passed=True, tests_run=len(checks), covered_checks=checks, groups=groups,
                errors=0, failures=0, skips=0, independent_complete=True, actual_process_kills=len(recovery),
                physical_recovery=recovery, prior_inputs_preserved=3364, validation_requested_roots=6,
                validation_complete_requested_labels=label_counts, training_steps=200, optimizer_draws=12800,
                trainable_parameters=65, frozen_policy=True, benchmark=closed,
                matched_contrasts=result['secondary_factor_contrasts'], endgame_guard=result['endgame_guard'],
                allocation=allocation, followup_summary=collected['summary'], completed_no_work=True,
                control_changed=False, default_promotion=False, extra_training_after_followup=False,
                inputs=dict(sorted(e.INPUTS.items())))
    current(body['inputs'])
    if OUTPUT.exists(): require(read(OUTPUT.relative_to(ROOT).as_posix()) == body, 'Immutable acceptance differs')
    else: atomic_json(OUTPUT, body)
    print(config.canonical(dict(event='ordinary_value_accepted', passed=True, tests_run=len(checks), inputs=len(body['inputs']),
                               actual_process_kills=len(recovery), branch=allocation['branch'], default_promotion=False)), flush=True)


def verify():
    body = read('build/ordinary-value-acceptance.json')
    require(body['passed'] is True and body['frozen_policy'] is True and body['default_promotion'] is False, 'Invalid acceptance')
    current(body['inputs'])
    print(config.canonical(dict(event='ordinary_value_acceptance_verified', passed=True,
                               tests_run=body['tests_run'], inputs=len(body['inputs']))), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('operation', choices=('build', 'verify'))
    args = parser.parse_args(); build() if args.operation == 'build' else verify()
