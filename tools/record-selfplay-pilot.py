"""Seal the source-bound current-model self-play pilot after independent review."""
from __future__ import annotations
import argparse
import ast
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

OUTPUT = ROOT / 'build/selfplay-pilot-acceptance.json'
INPUTS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Missing input: ' + str(name))
    key = path.relative_to(ROOT).as_posix()
    digest = sha(path)
    require(key not in INPUTS or INPUTS[key] == digest, 'Input changed during seal: ' + key)
    INPUTS[key] = digest
    return digest


def read(name):
    pin(name)
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def current(mapping):
    for name, digest in mapping.items():
        require(pin(name) == digest, 'Input changed: ' + name)


def executed_source_provenance(body):
    p = body.get('execution_provenance')
    if not p:
        return
    original = read(p['original_report'])
    require(pin(p['original_report']) == p['original_report_sha256'] and
            pin(p['executed_main_snapshot']) == p['executed_main_sha256'] and
            pin(p['current_main']) == p['current_main_sha256'] and p['tests_rerun'] is False,
            'Executed-source provenance differs')
    for name, value in original.items():
        if name != 'inputs':
            require(body[name] == value, 'Derived report changed an executed result: ' + name)
    before = ast.parse((ROOT / p['executed_main_snapshot']).read_text(encoding='utf-8'))
    after = ast.parse((ROOT / p['current_main']).read_text(encoding='utf-8'))
    methods_before = {n.name: n for n in ast.walk(before) if isinstance(n, ast.FunctionDef)}
    methods_after = {n.name: n for n in ast.walk(after) if isinstance(n, ast.FunctionDef)}
    for name in body['covered_methods']:
        left, right = (ast.dump(d[name], include_attributes=False) for d in (methods_before, methods_after))
        require(left == right and hashlib.sha256(left.encode()).hexdigest() == p['covered_method_AST_sha256'][name],
                'An executed test body changed: ' + name)
    for tree in (before, after):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == p['only_changed_method']:
                node.body = [ast.Pass()]
    require(ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False),
            'A helper, import, global or executed method changed')
    for name, digest in original['inputs'].items():
        if name == p['current_main']:
            require(digest == p['executed_main_sha256'], 'Original source was not the retained executed snapshot')
        else:
            require(body['inputs'][name] == digest and pin(name) == digest, 'An original execution input changed')


def group(report, source, partial=False):
    body = read(report)
    require(body['passed'] is True and body['errors'] == body['failures'] == body['skips'] == 0,
            'Executed checks failed: ' + report)
    pin(source)
    declared = {}
    for cls in ast.parse((ROOT / source).read_text(encoding='utf-8')).body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                    require(method.name not in declared, 'Duplicate test method')
                    declared[method.name] = source + '::' + cls.name + '::' + method.name
    names = body.get('covered_methods', body.get('methods'))
    require(names and len(names) == len(set(names)) == body['tests_run'] and set(names) <= set(declared) and
            (partial or set(names) == set(declared)),
            'Executed/source coverage differs: ' + report)
    executed_source_provenance(body)
    current(body['inputs'])
    if 'groups' in body:
        children = body['groups'].values() if isinstance(body['groups'], dict) else body['groups']
        union = []
        for record in children:
            require(pin(record['path']) == record['sha256'], 'Independent subgroup changed')
            child, checks = group(record['path'], source, partial=True)
            require(child['tests_run'] == record['tests_run'] and
                    child['covered_methods'] == record['covered_methods'], 'Subgroup execution coverage differs')
            union.extend(checks)
        require(len(union) == len(set(union)) == body['tests_run'] and
                set(union) == {declared[name] for name in names}, 'Independent subgroup union differs')
    return body, [declared[name] for name in names]


def physical_rows(name, table, order):
    pin(name)
    with closing(sqlite3.connect((ROOT / name).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT * FROM ' + table + ' ORDER BY ' + order).fetchall()
    result = {}
    for row in rows:
        require(config.digest(json.loads(row[-2])) == row[-1], 'Retained journal row corrupt')
        key = tuple(row[:-2])
        require(key not in result, 'Duplicate physical evidence key')
        result[key] = row
    return result


def physical_marker(record):
    folder = Path(record['probe'])
    require(read(folder / 'marker.json') == record['marker'], 'Helper marker differs')
    log = folder / 'worker.log'
    pin(log)
    events = []
    for line in (ROOT / log).read_text(encoding='utf-8').splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    require(record['marker'] in events, 'Kill barrier was not in actual stdout')
    declared = folder / 'declared-marker.json'
    if (ROOT / declared).exists():
        require(read(declared) == record['marker'], 'Published producer marker differs')
    require(record['actual_process_kill'] is True, 'Recovery did not kill an actual process')


def physical_recovery():
    preflight = read('build/selfplay-pilot-audit/preflight.json')
    for record in preflight['evidence']['live_selfplay_recovery']['tests']:
        physical_marker(record)
        prefix = physical_rows(Path(record['probe']) / 'validated-prefix.sqlite3', 'sp_decisions', 'gameid,choice')
        final = physical_rows(Path(record['run']) / 'work.sqlite3', 'sp_decisions', 'gameid,choice')
        expected = 1 if record['event'].endswith('_precommit') else 2
        require(len(prefix) == record['committed_decisions'] == expected and
                {k: final[k] for k in prefix} == prefix, 'Physical self-play prefix was not retained')
    for record in preflight['evidence']['live_training_recovery']['tests']:
        physical_marker(record)
        prefix_name = Path(record['probe']) / 'validated-prefix.sqlite3'
        pin(prefix_name)
        with closing(sqlite3.connect((ROOT / prefix_name).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            prefix = db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY step').fetchall()
        final_catalog = Path(record['run']) / 'catalog.sqlite3'
        pin(final_catalog)
        with closing(sqlite3.connect((ROOT / final_catalog).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            final = {row[0]: row for row in db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY step')}
        expected = [2] if record['event'] == 'checkpoint_file_ready' else [2, 4]
        require([r[0] for r in prefix] == record['committed_prefix_steps'] == expected,
                'Physical checkpoint prefix differs')
        for step, name, digest, text, checksum in prefix:
            require(config.digest(json.loads(text)) == checksum and pin(Path(record['run']) / name) == digest,
                    'Committed checkpoint bytes changed after interruption')
            require(final[step] == (step, name, digest, text, checksum), 'Committed checkpoint catalog row changed')
        marker_file = Path(record['run']) / record['marker']['name']
        pin(marker_file)
        require((record['marker']['name'] in {row[1] for row in prefix}) is
                (record['event'] == 'checkpoint_committed'), 'Orphan checkpoint/catalog commit barrier differs')
        if record['event'] == 'checkpoint_file_ready':
            require(record['marker']['name'] not in {row[1] for row in final.values()},
                    'An uncommitted orphan was admitted into the resumed catalog')
        else:
            require(final[record['marker']['step']][1] == record['marker']['name'],
                    'The committed interrupted checkpoint was replaced')
        exact = record['exact_state_proof']
        current(exact['inputs'])
        require(exact['passed'] is exact['all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal'] is True and
                exact['step'] == 6 and exact['draws'] == 192, 'Fresh/resumed full state comparison differs')
        no_work = record['completed_no_work_proof']
        require(no_work['passed'] is no_work['zero_training_runtime_activations'] is no_work['zero_optimizer_steps'] is
                no_work['all_durable_bytes_mtimes_unchanged'] is True and no_work['torch_imported'] is False,
                'Recovered checkpoint completed reader did new work')
    model = read('build/selfplay-pilot-audit/model.json')['evidence']['production_optimizer_audit']
    current(model['inputs'])
    require(model['passed'] is True and model['steps_replayed'] == 200 and model['draws_replayed'] == 25600 and
            len(model['checkpoints']) == 10 and model['optimizer_parameter_ids'] == list(range(8)) and
            model['all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal'] is
            model['all_checkpoint_metrics_independently_equal'] is model['final_export_equals_replayed_tensors'] is True,
            'Independent full production optimizer replay is incomplete')
    for checkpoint in model['checkpoints']:
        require(pin(Path('data/selfplay-training-pilot-v1') / checkpoint['file']) == checkpoint['sha256'],
                'Independently replayed checkpoint bytes changed')
    recovery = read('build/selfplay-pilot-audit/recovery.json')['evidence']['live_benchmark_recovery']
    require(len(preflight['evidence']['live_selfplay_recovery']['tests']) ==
            len(preflight['evidence']['live_training_recovery']['tests']) == 2 and
            len(recovery['tests']) == 4, 'Actual kill coverage differs')
    units = set()
    for record in recovery['tests']:
        physical_marker(record)
        unit = record['unit']
        require(unit in ('decision', 'game'), 'Unknown interrupted benchmark unit')
        units.add((unit, record['event'].endswith('_precommit')))
        table, order = ('decisions', 'game,choice') if unit == 'decision' else ('games', 'idx')
        prefix = physical_rows(Path(record['probe']) / 'validated-prefix.sqlite3', table, order)
        complete = physical_rows(record['validated_complete_journal'], table, order)
        expected = 1 if record['event'].endswith('_precommit') else 2
        require(len(prefix) == record['committed_prefix'] == expected and
                {k: complete[k] for k in prefix} == prefix and record['resumed_terminal_games'] == 8 and
                record['all_committed_decisions_and_games_exact'] is True and
                record['completed_bytes_mtimes_unchanged'] is True,
                'Physical benchmark prefix did not survive recovery')
        if unit == 'game':
            before = physical_rows(Path(record['probe']) / 'validated-prefix.sqlite3', 'decisions', 'game,choice')
            after = physical_rows(record['validated_complete_journal'], 'decisions', 'game,choice')
            require({k: after[k] for k in before} == before, 'Committed game decision prefix changed')
    require(len(units) == 4, 'Pre/post benchmark decision/game coverage differs')
    return preflight


def durable_tree(folder):
    folder = (ROOT / folder).resolve()
    require(folder.is_relative_to(ROOT), 'Evidence folder escaped workspace')
    for path in sorted(folder.rglob('*')):
        if path.is_file() and not path.name.endswith(('-wal', '-shm')) and '__pycache__' not in path.parts:
            pin(path.relative_to(ROOT).as_posix())


def observer_admission():
    value = read('build/selfplay-pilot-observer-proof.json')
    require(value['passed'] is True and value['before'] == value['after'] and
            value['all_bytes_sizes_mtimes_unchanged'] is True and
            value['actual_training_batch'] == 128 and value['actual_training_draws'] == 25600 and
            value['actual_benchmark_games'] == 96 and value['actual_benchmark_measurements'] == 256 and
            len(value['eight_current_card_ids']) == len(set(value['eight_current_card_ids'])) == 8 and
            value['new_optimizer_or_runtime_calls'] == value['new_searches'] == 0 and
            value['torch_imported'] is value['browser_or_HTTP_inspection'] is False,
            'Dashboard overlay proof differs')
    current(value['source_inputs'])
    current(value['frozen_producer_sources'])
    for name, signature in value['after'].items():
        path = ROOT / name
        require(pin(name) == signature['sha256'] and path.stat().st_size == signature['size'] and
                path.stat().st_mtime_ns == signature['mtime_ns'], 'Observer changed a durable artifact')
    history = read('build/selfplay-benchmark-prefreeze-author-diagnostic.json')
    require(history['tests_rerun_for_original_report'] is False and
            pin(history['original_report_path']) == history['original_report_sha256'],
            'Historical owner execution provenance differs')
    original = read(history['original_report_path'])
    for logical, record in history['executed_source_snapshots'].items():
        require(pin(record['path']) == record['sha256'] == original['inputs'][logical],
                'Retained original owner source does not match its execution')


def build():
    from congklak import selfplay_training_pilot as learning, selfplay_pilot_benchmark as benchmark
    INPUTS.clear()
    for name in ('tools/record-selfplay-pilot.py', 'tools/prepare-selfplay-pilot-baseline.py',
                 'tools/verify-selfplay-pilot-rerun.py', 'tools/review-selfplay-pilot-weights.py',
                 'tools/verify-selfplay-pilot-acceptance.py'):
        pin(name)
    baseline = read('build/selfplay-pilot-baseline.json')
    require(baseline['passed'] is True and baseline['inputs_count'] == len(baseline['inputs']) == 2839,
            'Previous acceptance baseline differs')
    current(baseline['inputs'])
    checks, groups = [], {}
    for label, report, source in (
        ('training_owner', 'build/selfplay-training-pilot-author.json', 'tests/selfplay_training_pilot_author.py'),
        ('benchmark_owner', 'build/selfplay-pilot-benchmark-author.json', 'tests/selfplay_pilot_benchmark_author.py'),
        ('independent', 'build/selfplay-pilot-audit.json', 'tests/selfplay_pilot_independent_audit.py')):
        body, names = group(report, source)
        checks.extend(names)
        groups[label] = body['tests_run']
    require(len(checks) == len(set(checks)), 'Acceptance double-counts tests')
    physical_recovery()
    manifest, rows = learning.load_replay()
    cpu, proof = learning.verify_run(), learning.witness()
    current(proof['inputs'])
    result = read('data/selfplay-training-pilot-v1/result.json')
    summary = manifest['summary']
    require(all(sum(map(sum, row['state']['pits'])) >= 25 for row in rows),
            'A late position entered the frozen midgame optimizer domain')
    require(summary['games'] == 32 and summary['terminal_games'] + summary['capped_games'] == 32 and
            summary['admitted_families'] >= 8 and summary['training_admission'] == 'ready' and
            summary['retained_rows'] == len(rows) and manifest['status'] == 'complete',
            'Self-play is incomplete or below the frozen training-yield gate')
    require(result['status'] == 'complete' and result['steps_completed'] == result['selected_step'] == 200 and
            result['draws'] == 25600 and result['teacher_rehearsal'] is False and
            proof['checkpoints_verified'] == cpu['checkpoints_verified'] == 10 and
            proof['all_optimizer_sampler_rng_states_verified'] is True and
            proof['final_export_equals_checkpoint_tensors'] is True and
            proof['all8_tensors_trainable'] is True and
            proof['source_weights_sha256'] == learning.SOURCE_WEIGHTS,
            'Joint update or actual checkpoint inspection is incomplete')
    closed = benchmark.closed_witness('evaluation')
    strength = read('build/selfplay-pilot-benchmark.json')
    require(closed['passed'] is True and closed['games'] == strength['games'] == 96 and
            closed['cases'] == strength['cases'] == 32 and
            closed['scalar_measurements'] == strength['scalar_measurements'] == 256 and
            strength['default_promotion'] is strength['sealed_final_test_consumed'] is strength['energy_measured'] is False,
            'Equal-time comparison is incomplete or makes an unsupported claim')
    for arena in strength['arenas'].values():
        require(arena['complete_pairs'] == arena['pairs'] == 12 and arena['terminal_games'] == 24 and
                arena['unfinished_games'] == 0 and not sum(arena['fallback_counts'].values()),
                'A declared comparison has incomplete games or fallbacks')
    rerun = read('build/selfplay-pilot-root-verification.json')
    require(rerun['passed'] is True and rerun['before'] == rerun['after'] and
            rerun['completed_bytes_mtimes_unchanged'] is True and rerun['torch_imported'] is False and
            all(rerun[field] == 0 for field in ('new_optimizer_activations', 'new_selfplay_searches',
                                              'new_comparison_decisions', 'new_native_searches')),
            'Completed no-work recovery proof differs')
    for name, signature in rerun['after'].items():
        path = ROOT / name
        require(pin(name) == signature['sha256'] and path.stat().st_size == signature['size'] and
                path.stat().st_mtime_ns == signature['mtime_ns'], 'Completed file signature changed: ' + name)
    weights = read('build/selfplay-pilot-sampling-review.json')
    require(weights['passed'] is True and weights['families'] == summary['admitted_families'] and
            weights['train_rows'] == len(rows) and weights['draws'] == 25600 and
            weights['optimization_distribution_changed'] is weights['teacher_rehearsal'] is False,
            'Fixed exposure review differs')
    current(weights['inputs'])
    observer_admission()
    # Preserve all frozen producer dependencies, artifacts and real recovery
    # fixtures. Mutable live observer activity and SQLite sidecars are excluded.
    for name in ('congklak/selfplay_training_pilot.py', 'tools/selfplay-training-pilot.py',
                 'congklak/selfplay_pilot_benchmark.py', 'tools/benchmark-selfplay-pilot.py',
                 'congklak/selfplay_pilot_progress.py', 'tools/selfplay-pilot-dashboard.html',
                 'congklak/selfplay_pilot_live_progress.py', 'tools/verify-selfplay-pilot-observer.py',
                 'tools/selfplay-pilot-audit.py', 'docs/SELFPLAY_PILOT_SCOPE.md',
                 'docs/SELFPLAY_TRAINING_PILOT_SCOPE.md', 'docs/SELFPLAY_PILOT_RESULTS.md',
                 'docs/SELFPLAY_PILOT_INDEPENDENT_AUDIT.md', 'build/selfplay-pilot-rerun.log'):
        pin(name)
    for pattern in ('selfplay-training-pilot-*.json', 'selfplay-pilot-benchmark*.json', 'selfplay-pilot-audit*.json'):
        for path in sorted((ROOT / 'build').glob(pattern)):
            body = read(path.relative_to(ROOT).as_posix())
            if '-prefreeze-' in path.name:
                continue  # Historical inputs identify the retained executed version.
            for mapping_name in ('inputs', 'sources', 'science_inputs'):
                if mapping_name in body and isinstance(body[mapping_name], dict):
                    current(body[mapping_name])
    for folder in ('data/selfplay-training-pilot-v1', 'data/selfplay-pilot-benchmark-v1',
                   'build/selfplay-pilot-audit', 'build/selfplay-pilot-prefreeze-eligibility-v1',
                   'build/selfplay-training-pilot-prefreeze-domain-v1'):
        durable_tree(folder)
    for pattern in ('selfplay-training-pilot-author*', 'selfplay-pilot-benchmark-author*',
                    'selfplay-benchmark-author*'):
        for folder in sorted((ROOT / 'build').glob(pattern)):
            if folder.is_dir():
                durable_tree(folder.relative_to(ROOT).as_posix())
            elif folder.is_file():
                pin(folder.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / 'build/selfplay-pilot-progress').glob('run-*.json')):
        run = json.loads(path.read_text(encoding='utf-8'))
        require(run['state'] != 'running', 'A producer or audit worker is still running')
        pin(path.relative_to(ROOT).as_posix())
        pin(run['log'])
    body = dict(schema='selfplay-pilot-acceptance-v1', passed=True, tests_run=len(checks), groups=groups,
                covered_checks=checks, errors=0, failures=0, skips=0, independent_complete=True,
                prior_inputs_preserved=len(baseline['inputs']), selfplay=summary,
                training_steps=result['steps_completed'], training_draws=result['draws'],
                actual_checkpoints_verified=proof['checkpoints_verified'],
                control_model_sha256=pin('data/quality-train-v1/model.json'),
                challenger_model_sha256=pin('data/selfplay-training-pilot-v1/model.json'),
                strength_games=96, exact_guard_roots=32, exact_guard_measurements=256,
                actual_process_kills=8, independent_optimizer_steps_replayed=200,
                both_baseline_primary_gates=strength['both_baseline_primary_gates'],
                completed_no_work=True, completed_bytes_mtimes_unchanged=True,
                default_promotion=False, sealed_final_test_consumed=False, source_data_pruned=False,
                energy_measured=False, inputs=dict(sorted(INPUTS.items())))
    current(body['inputs'])
    text = config.canonical(body) + '\n'
    if OUTPUT.exists():
        require(OUTPUT.read_text(encoding='utf-8') == text, 'Immutable acceptance differs')
    else:
        atomic_json(OUTPUT, body)
    print(config.canonical(dict(event='selfplay_pilot_accepted', passed=True, tests_run=len(checks),
                               inputs=len(INPUTS), strength_games=96, default_promotion=False)), flush=True)


def verify():
    body = read('build/selfplay-pilot-acceptance.json')
    require(body['passed'] is True and body['default_promotion'] is False, 'Invalid acceptance')
    current(body['inputs'])
    print(config.canonical(dict(event='selfplay_pilot_acceptance_verified', passed=True,
                               inputs=len(body['inputs']), tests_run=body['tests_run'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'verify'))
    args = parser.parse_args()
    build() if args.operation == 'build' else verify()
