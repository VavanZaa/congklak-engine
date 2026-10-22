"""Read-only aggregate acceptance for the fixed-policy, reviewed-weight value pilot."""
from __future__ import annotations
import argparse
import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as data, value_pilot_learning as learning, value_pilot_benchmark as benchmark

OUTPUT = ROOT / 'build/value-pilot-acceptance.json'
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    require(p.is_relative_to(ROOT) and p.is_file(), 'Missing acceptance input: ' + str(p))
    name = p.relative_to(ROOT).as_posix()
    INPUTS[name] = data.sha(p)
    return INPUTS[name]


def load(path):
    pin(path)
    return data.read(ROOT / path)


def current(mapping):
    for name, digest in mapping.items():
        require(pin(name) == digest, 'Acceptance input changed: ' + name)


def methods(source):
    pin(source)
    tree = ast.parse((ROOT / source).read_text(encoding='utf-8'))
    result = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    require(node.name not in result, 'Duplicate audit method name')
                    result[node.name] = source + '::' + cls.name + '::' + node.name
    require(result, 'No executed methods in ' + source)
    return result


def group(path, source, key='methods', partial=False):
    body = load(path)
    require(body['passed'] is True and body['failures'] == body['errors'] == body['skips'] == 0,
            'Executed group failed: ' + path)
    covered = body[key]
    actual = methods(source)
    require(len(covered) == len(set(covered)) == body['tests_run'] and set(covered) <= set(actual),
            'Executed method coverage differs: ' + path)
    if not partial:
        require(set(covered) == set(actual), 'Incomplete source method coverage: ' + path)
    current(body['inputs'])
    return body, [actual[name] for name in covered]


def readonly(path):
    pin(path)
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def physical_recovery(training, games):
    training_group = load('build/value-pilot-audit/training.json')
    folder = ROOT / training_group['retained_directory'] / 'training-kills'
    for record in training['tests']:
        event = record['event']
        probe = folder / (event + '-probe')
        require(load((probe/'marker.json').relative_to(ROOT).as_posix()) == record['marker'] and
                load((probe/'declared-marker.json').relative_to(ROOT).as_posix()) == record['marker'],
                'Real checkpoint marker differs')
        with closing(readonly(probe/'validated-prefix.sqlite3')) as db:
            entries = db.execute('SELECT step,name,sha FROM checkpoints ORDER BY id').fetchall()
        expected = [2] if event == 'checkpoint_file_ready' else [2, 4]
        require([row[0] for row in entries] == record['committed_prefix_steps'] == expected,
                'Killed checkpoint prefix differs')
        for _, name, digest in entries:
            file = (folder/event/name).resolve()
            require(file.parent == (folder/event).resolve() and pin(file) == digest, 'Retained checkpoint bytes differ')
        if event == 'checkpoint_file_ready':
            orphan = (folder/event/record['marker']['name']).resolve()
            require(orphan.parent == (folder/event).resolve(), 'Orphan checkpoint path escaped its run')
            pin(orphan)
            with closing(readonly(folder/event/'catalog.sqlite3')) as db:
                committed_names = {row[0] for row in db.execute('SELECT name FROM checkpoints')}
            require(orphan.name not in committed_names, 'Uncommitted orphan entered the resumed catalog')
        exact = load((folder/(event+'-exact-state.json')).relative_to(ROOT).as_posix())
        completed = load((folder/(event+'-completed.json')).relative_to(ROOT).as_posix())
        require(exact['passed'] is True and exact['all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal'] is True and
                exact['step'] == 6 and completed['passed'] is True and completed['zero_optimizer_steps'] is True and
                completed['torch_imported'] is False and completed['files_bytes_mtimes_unchanged'] is True,
                'Retained checkpoint comparison/completed proof differs')
    game_group = load('build/value-pilot-audit/recovery.json')
    folder = ROOT / game_group['retained_directory'] / 'game-kills'
    for record in games['tests']:
        suffix = 'precommit' if record['event'].endswith('_precommit') else 'committed'
        probe = folder/(suffix+'-probe')
        require(load((probe/'marker.json').relative_to(ROOT).as_posix()) == record['marker'] and
                load((probe/'declared-marker.json').relative_to(ROOT).as_posix()) == record['marker'],
                'Real game marker differs')
        with closing(readonly(probe/'validated-prefix.sqlite3')) as db:
            prefix = db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx').fetchall()
        with closing(readonly(folder/(suffix+'-valid.sqlite3'))) as db:
            resumed = db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx').fetchall()
        require(len(prefix) == record['committed_prefix'] == (0 if suffix == 'precommit' else 1) and
                len(resumed) == record['games_after_resume'] == 8 and resumed[:len(prefix)] == prefix,
                'Actual killed/resumed game prefixes differ')
        for _, _, _, _, text, digest in resumed:
            require(config.digest(json.loads(text)) == digest, 'Corrupt retained resumed game')


def build_report():
    INPUTS.clear()
    pin('tools/record-value-pilot.py')
    pin('tools/prepare-value-pilot-baseline.py')
    baseline = load('build/value-pilot-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2426
            and baseline['prior_accepted_inputs'] == 2424, 'Incomplete preserved baseline')
    current(baseline['inputs'])
    manifest, groups = data.prepare()
    pin(data.MANIFEST)
    current(manifest['inputs'])
    review = load('build/value-pilot-sampling-review.json')
    require(manifest['groups'] == dict(train=7624, development=1000, ordinary_development=14, diagnostic=8), 'Pilot cohorts differ')
    require(manifest['base_rows'] == 7563 and manifest['new_ordinary_train_rows'] == 61 and
            review['ordinary_families'] == 59 and review['loss_weights_are_all_one'] is True,
            'Reviewed optimizer cohort differs')
    require(all(r['loss_weight'] == 1 for r in groups['train']), 'Double loss weighting')
    source_model = pin(data.MODEL)
    require(source_model == data.MODEL_SHA, 'Source control changed')
    pre = benchmark.pretraining_witness()
    post_closed = benchmark.closed_witness('posttraining')
    trained = learning.witness()
    require(pre['passed'] is post_closed['passed'] is trained['passed'] is True, 'Closed producer admission failed')
    require(pre['cases'] == 44 and pre['games'] == 48 and pre['scalar_measurements'] == 264 and
            post_closed['cases'] == 44 and post_closed['games'] == 96 and post_closed['scalar_measurements'] == 352,
            'Equal-time campaign incomplete')
    require(trained['selected_step'] == 200 and trained['checkpoints_verified'] == 10 and
            trained['trainable_parameter_count'] == 65 and trained['frozen_parameter_count'] == 12807 and
            trained['all_frozen_tensors_bit_exact'] is trained['all_dataset_policy_logits_bit_exact'] is True,
            'Policy/trunk ownership or fixed-final selection differs')
    require(trained['identity']['pretraining_witness'] == pre, 'Pretraining control did not precede training')
    current(trained['inputs'])
    pin('build/value-pilot-model-witness.json')
    result = load('data/value-pilot-v1/result.json')
    require(result['steps_completed'] == result['selected_step'] == 200 and result['draws'] == 51200 and
            result['promotion'] is False and result['sealed_final_test_consumed'] is False,
            'Small training scope changed')
    before = load('build/value-pilot-benchmark-pretraining.json')
    after = load('build/value-pilot-benchmark-posttraining.json')
    require(before['cases'] == after['cases'] == 44 and before['games'] == 48 and after['games'] == 96
            and before['scalar_measurements'] + after['scalar_measurements'] == 616, 'Benchmark totals differ')
    for stage in benchmark.STAGES:
        path, output = benchmark.paths(stage)
        pin(path / 'work.sqlite3')
        pin(output)
        resource = load(output.with_name(output.stem + '-resources.json').relative_to(ROOT).as_posix())
        require(resource['torch_imported'] is False and resource['energy_measured'] is False,
                'Timed workers imported Torch or claimed energy')
    owner_train, train_ids = group('build/value-pilot-learning-author.json', 'tests/value_pilot_learning_acceptance.py')
    owner_bench, bench_ids = group('build/value-pilot-benchmark-author.json', 'tests/value_pilot_benchmark_acceptance.py')
    audit = load('build/value-pilot-audit.json')
    require(audit['passed'] is True and audit['errors'] == audit['failures'] == audit['skips'] == 0,
            'Independent audit failed')
    actual = methods('tests/value_pilot_independent_audit.py')
    names = audit['covered_methods']
    require(len(names) == len(set(names)) == audit['tests_run'] and set(names) == set(actual),
            'Independent source coverage differs')
    current(audit['inputs'])
    audit_ids = []
    for item in audit['groups']:
        require(pin(item['path']) == item['sha256'], 'Independent group binding differs')
        body, ids = group(item['path'], 'tests/value_pilot_independent_audit.py', 'covered_methods', True)
        require(body['tests_run'] == item['tests_run'], 'Independent executed group count differs')
        audit_ids += ids
    require(len(audit_ids) == len(set(audit_ids)) == audit['tests_run'] and set(audit_ids) == {actual[n] for n in names},
            'Independent execution union differs')
    training_recovery = audit['evidence']['live_training_recovery']
    game_recovery = audit['evidence']['live_game_recovery']
    require(training_recovery['actual_new_value_training_killed'] is True and
            game_recovery['actual_initial_and_committed_game_kills'] is True and
            {r['event'] for r in training_recovery['tests']} == {'checkpoint_file_ready', 'checkpoint_committed'} and
            {r['event'] for r in game_recovery['tests']} == {'value_pilot_game_precommit', 'value_pilot_game_committed'} and
            len(training_recovery['tests']) == len(game_recovery['tests']) == 2,
            'Actual pre/postcommit interruption coverage differs')
    require(all(r['actual_process_kill'] is True and r['prefix_bytes_mtimes_exact'] is True and
                r['all_model_optimizer_sampler_python_torch_rng_raw_bytes_equal'] is True and
                r['completed_no_torch_no_optimizer_work'] is True for r in training_recovery['tests']),
            'Deterministic optimizer recovery proof incomplete')
    require(all(r['actual_process_kill'] is True and r['prefix_exactly_retained'] is True and
                r['all_resumed_games_literally_replayed'] is True and
                r['completed_no_search_and_bytes_mtimes_unchanged'] is True for r in game_recovery['tests']),
            'Timed-game recovery proof incomplete')
    physical_recovery(training_recovery, game_recovery)
    optimizer = audit['evidence']['production_optimizer_audit']
    current(optimizer['inputs'])
    require(optimizer['passed'] is True and optimizer['checkpoints'] == 10 and optimizer['draws'] == 51200 and
            optimizer['all200_optimizer_steps_independently_replayed'] is True and
            optimizer['all_checkpoint_model_optimizer_sampler_python_torch_rng_raw_bytes_equal'] is True and
            optimizer['all_frozen_tensors_bit_exact'] is True and optimizer['final_export_equals_PT'] is True and
            optimizer['optimizer_parameter_ids'] == [0, 1] and
            optimizer['changed_tensors'] == ['layers.4.bias', 'layers.4.weight'],
            'Actual optimizer replay/export proof incomplete')
    rerun = load('build/value-pilot-root-verification.json')
    require(rerun['schema'] == 'value-pilot-completed-no-work-rerun-v1' and rerun['passed'] is True and
            rerun['prior_inputs_preserved'] == 2426 and rerun['before'] == rerun['after'] and
            rerun['completed_bytes_mtimes_unchanged'] is True and rerun['new_optimizer_activations'] ==
            rerun['new_decisions'] == rerun['new_native_searches'] == 0 and rerun['torch_imported'] is False,
            'Completed no-work rerun proof differs')
    current(rerun['sources'])
    for name, signature in rerun['after'].items():
        p = ROOT / name
        require(pin(name) == signature['sha256'] and p.stat().st_size == signature['size'] and
                p.stat().st_mtime_ns == signature['mtime_ns'], 'Completed durable artifact changed: ' + name)
    for name in ('congklak/value_pilot_progress.py', 'tools/value-pilot-dashboard.html',
                 'docs/VALUE_PILOT_SCOPE.md', 'docs/VALUE_PILOT_RESULTS.md', 'docs/VALUE_PILOT_INDEPENDENT_AUDIT.md',
                 'build/value-pilot-root-weight-review.json', 'build/value-pilot-rerun.log'):
        pin(name)
    retained = []
    for folder in ('build/value-pilot-audit', 'build/value-pilot-learning-author', 'build/value-pilot-benchmark-author'):
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and not p.name.endswith(('-wal', '-shm')):
                pin(p)
                retained.append(p.relative_to(ROOT).as_posix())
    for pattern in ('value-pilot-*prefreeze*', 'value-pilot-*failed*', 'value-pilot-*failure*'):
        for p in sorted((ROOT / 'build').glob(pattern)):
            if p.is_file():
                pin(p)
                retained.append(p.relative_to(ROOT).as_posix())
    ids = train_ids + bench_ids + audit_ids
    require(len(ids) == len(set(ids)), 'Aggregate test IDs duplicate')
    current(dict(INPUTS))
    require('torch' not in sys.modules, 'Acceptance imported Torch')
    return dict(schema='value-pilot-acceptance-v1', passed=True, engineering_passed=True,
        tests_run=len(ids), errors=0, failures=0, skips=0, covered_checks=ids,
        groups=dict(learning=len(train_ids), benchmark=len(bench_ids), independent=len(audit_ids)),
        prior_inputs_preserved=2426, train_rows=7624, new_ordinary_rows=61, selected_step=200,
        optimizer_draws=51200, trainable_parameters=65, policy_and_trunk_bit_exact=True,
        pretraining_games=48, posttraining_games=96, scalar_measurements=616,
        candidate_model_sha256=trained['model_file_sha256'], source_model_sha256=source_model,
        initial_development=result['initial_descriptive'], final_development=result['descriptive'],
        both_baseline_primary_gates=after['both_baseline_primary_gates'],
        matched_contrasts=after['secondary_factor_contrasts'], independent_complete=True,
        actual_training_process_kills=2, actual_game_process_kills=2,
        completed_no_new_work=True, completed_bytes_mtimes_unchanged=True,
        sealed_final_test_consumed=False, default_promotion=False, source_data_pruned=False,
        scope='One fixed-final value65-only bundled data/sampler pilot; reused development equal-time results, no isolated component attribution or automatic promotion.',
        retained_diagnostics=sorted(set(retained)), inputs=dict(sorted(INPUTS.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    body = build_report()
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == config.canonical(body) + '\n', 'Immutable acceptance differs')
    if not args.check_only:
        data.publish(OUTPUT, body)
    print(config.canonical(dict(event='value_pilot_acceptance_checked' if args.check_only else 'value_pilot_accepted',
        passed=True, tests_run=body['tests_run'], inputs=len(INPUTS), acceptance_written=not args.check_only)), flush=True)


if __name__ == '__main__':
    main()
