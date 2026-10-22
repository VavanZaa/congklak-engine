"""Recoverable existing-data diagnostics; no training or search entry points.

Each immutable commit names exactly one hashed result and the preceding commit.
An interrupted uncommitted result is recomputed. catalog.json is the sole durable
completion marker; completed readers validate before any writer or computation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid
from . import config, neurodiag_common as common
from .valuepilot_reservations import Writer

ROOT = common.ROOT
FOLDER = ROOT / 'build/value-diagnostics-v1'
DEFAULT = FOLDER / 'production'
SCHEMA = 'fixed-value-diagnostics-v1'
SOURCES = ('congklak/value_diagnostics.py', 'congklak/value_capture_diagnostic.py',
           'congklak/value_bridge_diagnostic.py', 'tools/run-value-diagnostics.py')
ZERO = dict(optimizer_steps=0, optimizer_draws=0, new_games=0,
            new_teacher_labels=0, searches=0, feature_simulations=0)
need, read, sha = common.require, common.read, common.sha


def atomic(path, body):
    """Immutable, fsynced publication. Temporary files are never commit records."""
    path = Path(path)
    text = config.canonical(body) + '\n'
    if path.exists():
        need(path.read_text(encoding='utf-8') == text, 'Immutable diagnostic output differs: ' + str(path))
        return body
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with pending.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    return body


def baseline():
    path = FOLDER / 'baseline.json'
    if path.exists():
        body = read(path)
        need(body['schema'] == SCHEMA + '-baseline' and body['passed'] is True and
             body['inputs_count'] == len(body['inputs']) == 40541, 'Diagnostic baseline differs')
        common.current(body['inputs'])
        return body
    accepted = ROOT / 'build/value-run-v1/acceptance.json'
    previous = read(accepted)
    need(previous['passed'] is True and len(previous['inputs']) == 40540, 'Prior pilot acceptance differs')
    inputs = dict(previous['inputs']); inputs.update(common.pins([accepted]))
    common.current(inputs)
    return atomic(path, dict(schema=SCHEMA + '-baseline', passed=True,
                            inputs_count=len(inputs), inputs=inputs))


def path_of(path):
    path = Path(path).resolve()
    need(path == DEFAULT.resolve() or path.is_relative_to(FOLDER / 'fixtures'), 'Diagnostic output namespace differs')
    return path


def modules():
    from . import value_capture_diagnostic as capture, value_bridge_diagnostic as bridges
    return dict(capture=capture, bridges=bridges)


def prepare(path):
    old = baseline()
    plans = {name: module.plan() for name, module in modules().items()}
    tasks = []
    inputs = dict(old['inputs']); inputs.update(common.pins([FOLDER / 'baseline.json', *SOURCES]))
    for name, plan in plans.items():
        for key, digest in plan['inputs'].items():
            need(key not in inputs or inputs[key] == digest, 'Diagnostic input identities conflict')
            inputs[key] = digest
        for task in plan['tasks']:
            tasks.append(dict(index=len(tasks), kind=name, task=task))
    common.current(inputs)
    identity = dict(schema=SCHEMA + '-identity', tasks=tasks, inputs=inputs,
                    counters=ZERO, models_and_policy_fixed=True,
                    candidate_quarantined=True, strength_interpretation_allowed=False)
    return atomic(path / 'identity.json', identity)


def prefix(path, identity):
    files = sorted((path / 'commits').glob('*.json'))
    results = []; previous = None
    for index, filename in enumerate(files):
        need(filename.name == f'{index:06d}.json', 'Diagnostic commit sequence has a gap')
        row = read(filename)
        need(index < len(identity['tasks']) and row['index'] == index and
             row['task_sha256'] == config.digest(identity['tasks'][index]) and
             row['previous_commit_sha256'] == previous and
             row['identity_sha256'] == sha(path / 'identity.json'), 'Diagnostic commit chain differs')
        expected = f'units/{index:06d}-{row["result_sha256"]}.json'
        need(row['result'] == expected and sha(path / expected) == row['result_sha256'], 'Diagnostic result hash differs')
        results.append(read(path / expected)); previous = sha(filename)
    return results


def identity_checks(identity):
    need(identity['schema'] == SCHEMA + '-identity' and identity['counters'] == ZERO and
         identity['models_and_policy_fixed'] is True and identity['candidate_quarantined'] is True and
         identity['strength_interpretation_allowed'] is False, 'Diagnostic scope changed')
    prior = read(FOLDER / 'baseline.json')
    need(prior['schema'] == SCHEMA + '-baseline' and prior['passed'] is True and
         prior['inputs_count'] == len(prior['inputs']) == 40541 and
         all(identity['inputs'].get(k) == v for k, v in prior['inputs'].items()), 'Diagnostic prior coverage differs')
    for name in (*SOURCES, 'build/value-diagnostics-v1/baseline.json'):
        need(name in identity['inputs'], 'Diagnostic required input omitted: ' + name)
    models = {'data/quality-train-v1/model.json': 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558',
              'data/value-training-v1/training/model.json': 'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319'}
    need(all(identity['inputs'].get(k) == v for k, v in models.items()), 'Diagnostic fixed models differ')
    tasks = identity['tasks']
    need(isinstance(tasks, list) and tasks and {t['kind'] for t in tasks} == {'capture', 'bridges'} and
         all(t['index'] == i and isinstance(t['task'], dict) for i, t in enumerate(tasks)), 'Diagnostic task structure differs')


def report_checks(reports, identity):
    need(set(reports) == {'capture', 'bridges'}, 'Diagnostic report coverage differs')
    for report in reports.values():
        need(all(identity['inputs'].get(k) == v for k, v in report['inputs'].items()), 'Diagnostic report input differs')
    capture = reports['capture']; bridges = reports['bridges']
    need(capture['passed'] is True and capture['models_fixed'] is True and capture['policy_fixed'] is True and
         capture['training_authorized'] is False and capture['candidate_quarantined'] is True and
         capture['fit_check_is_unseen'] is False, 'Capture diagnostic scope differs')
    expected = dict(cached_rows=640, saved_case_records=192, frozen_base_forwards=640,
                    sidecar_forwards=640, evaluated_value_predictions=1280, policy_forwards=640,
                    feature_simulations=0, optimizer_steps=0, optimizer_draws=0,
                    new_teacher_labels=0, new_games=0, new_searches=0)
    need(capture['counters'] == expected, 'Capture diagnostic counters differ')
    need(bridges['models_and_policy_fixed'] is True and bridges['ownership_hold_remains'] is True and
         bridges['strength_interpretation_allowed'] is False and bridges['posthoc_filtering'] is False and
         bridges['posthoc_cluster_merging'] is False and bridges['bootstrap_performed'] is False and
         bridges['original_ownership_reconstructed_exactly'] is True and
         bridges['literal_replayed_actions'] == 6663 and
         all(bridges[k] == 0 for k in ('optimizer_steps', 'model_forwards', 'searches', 'new_games', 'new_teacher_labels')),
         'Bridge diagnostic scope or counters differ')


def completed(path=DEFAULT):
    path = path_of(path)
    catalog = read(path / 'catalog.json')
    need(catalog['schema'] == SCHEMA + '-complete' and catalog['passed'] is True and
         catalog['identity_sha256'] == sha(path / 'identity.json'), 'Diagnostic completion identity differs')
    identity = read(path / 'identity.json')
    identity_checks(identity)
    common.current(identity['inputs'])
    results = prefix(path, identity)
    need(len(results) == len(identity['tasks']) == catalog['committed_units'], 'Diagnostic completion prefix differs')
    commits = common.pins(sorted((path / 'commits').glob('*.json')))
    need(commits == catalog['commits'], 'Diagnostic completion commits differ')
    reports = {}
    for name in ('capture', 'bridges'):
        report_path = path / (name + '.json')
        need(catalog['reports'][name] == sha(report_path), 'Diagnostic report hash differs')
        reports[name] = read(report_path)
    report_checks(reports, identity)
    need(catalog['counters'] == ZERO and catalog['models_and_policy_fixed'] is True and
         catalog['candidate_quarantined'] is True and catalog['strength_interpretation_allowed'] is False and
         'torch' not in sys.modules, 'Diagnostic scope counter or runtime differs')
    return dict(catalog=catalog, reports=reports)


def pause(path, point, requested, marker, index):
    if point != requested: return
    need(path != DEFAULT.resolve() and marker is not None, 'Pause hooks require a fixture namespace')
    marker = Path(marker).resolve()
    need(marker.is_relative_to(FOLDER / 'fixtures'), 'Pause marker outside fixture namespace')
    atomic(marker, dict(point=point, index=index, pid=os.getpid()))
    while True: time.sleep(.1)


def run(path=DEFAULT, pause_at=None, pause_marker=None, pause_index=0):
    path = path_of(path)
    # Even corrupt completion anchors take the read-only route and fail closed.
    if (path / 'catalog.json').exists(): return completed(path)
    need(pause_at in (None, 'before_unit_commit', 'after_unit_commit', 'before_catalog'), 'Unknown diagnostic pause')
    need(pause_at is None or path != DEFAULT.resolve(), 'Production pause hook forbidden')
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        if (path / 'catalog.json').exists(): return completed(path)
        # Recreate the metadata-only plan even on unfinished resume. Immutable
        # equality rejects altered uncommitted tasks before any unit executes.
        identity = prepare(path)
        identity_checks(identity)
        results = prefix(path, identity)
        workers = modules()
        for task in identity['tasks'][len(results):]:
            common.current({name: identity['inputs'][name] for name in SOURCES})
            result = workers[task['kind']].unit(task['task'])
            common.current({name: identity['inputs'][name] for name in SOURCES})
            index = task['index']
            # File SHA includes the final newline; name the result by its bytes.
            body_sha = __import__('hashlib').sha256((config.canonical(result) + '\n').encode()).hexdigest()
            relative = f'units/{index:06d}-{body_sha}.json'
            atomic(path / relative, result)
            if index == pause_index: pause(path, 'before_unit_commit', pause_at, pause_marker, index)
            record = dict(index=index, task_sha256=config.digest(task), identity_sha256=sha(path / 'identity.json'),
                          result=relative, result_sha256=body_sha,
                          previous_commit_sha256=sha(path / 'commits' / f'{index-1:06d}.json') if index else None)
            atomic(path / 'commits' / f'{index:06d}.json', record)
            results.append(result)
            if index == pause_index: pause(path, 'after_unit_commit', pause_at, pause_marker, index)
            print(json.dumps(dict(event='diagnostic_unit_committed', index=index, total=len(identity['tasks']),
                                  kind=task['kind'])), flush=True)
        reports = {}; report_bodies = {}
        for name, module in workers.items():
            selected = [body for task, body in zip(identity['tasks'], results) if task['kind'] == name]
            report = module.summarize(selected)
            atomic(path / (name + '.json'), report)
            report_bodies[name] = report
            reports[name] = sha(path / (name + '.json'))
        report_checks(report_bodies, identity)
        common.current(identity['inputs'])
        catalog = dict(schema=SCHEMA + '-complete', passed=True, identity_sha256=sha(path / 'identity.json'),
                       committed_units=len(results), commits=common.pins(sorted((path / 'commits').glob('*.json'))),
                       reports=reports, counters=ZERO, models_and_policy_fixed=True,
                       candidate_quarantined=True, strength_interpretation_allowed=False)
        pause(path, 'before_catalog', pause_at, pause_marker, len(results))
        atomic(path / 'catalog.json', catalog)
    return completed(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('baseline', 'run', 'verify'))
    parser.add_argument('--path', type=Path, default=DEFAULT)
    parser.add_argument('--pause-at', choices=('before_unit_commit', 'after_unit_commit', 'before_catalog'))
    parser.add_argument('--pause-marker', type=Path)
    parser.add_argument('--pause-index', type=int, default=0)
    args = parser.parse_args()
    if args.operation == 'baseline':
        body = baseline(); result = dict(passed=True, prior_inputs=body['inputs_count'])
    else:
        body = completed(args.path) if args.operation == 'verify' else run(args.path, args.pause_at, args.pause_marker, args.pause_index)
        result = dict(passed=True, units=body['catalog']['committed_units'], counters=ZERO)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()
