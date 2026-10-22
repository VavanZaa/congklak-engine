"""Independent audit of the prospective shared-frozen-suffix protocol.

The facts operation imports no production engine, diagnostic or training module.
It reads frozen metadata and literally replays already saved actions only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/ownership-protocol-v1/audit'
CONTROL = 'data/quality-train-v1/model.json'
CANDIDATE = 'data/value-training-v1/training/model.json'
CONTROL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
CANDIDATE_SHA = 'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(name, body):
    path = OUT / name
    content = canonical(body) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding='utf-8') == content, 'Immutable independent evidence differs')
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
    return body


def quotient(state):
    pits = state['pits'][0] + state['pits'][1]
    require(len(pits) == 14 and all(type(x) is int and x >= 0 for x in pits + state['stores']) and
            sum(pits + state['stores']) == 98 and state['controller'] in (0, 1), 'State differs')
    actor = state['controller'] if any(pits) else 0
    return min(bytes(pits + [actor]), bytes(pits[7:] + pits[:7] + [1 - actor if any(pits) else 0])).hex()


def active(state):
    return sum(map(sum, state['pits']))


def prior_closure():
    filename = ROOT / 'build/value-diagnostics-v1/acceptance.json'
    accepted = read(filename)
    require(accepted['passed'] is True and accepted['candidate_quarantined'] is True and
            accepted['strength_interpretation_allowed'] is False and len(accepted['inputs']) == 45201,
            'Prior diagnostic closure differs')
    for name, expected in accepted['inputs'].items():
        require(sha(ROOT / name) == expected, 'Prior accepted file changed: ' + name)
    control, candidate = read(ROOT / CONTROL), read(ROOT / CANDIDATE)
    require(sha(ROOT / CONTROL) == CONTROL_SHA and sha(ROOT / CANDIDATE) == CANDIDATE_SHA,
            'Frozen model differs')
    require(candidate['model']['base_export'] == control, 'Base model or policy changed')
    return dict(previous_inputs_verified=45201, previous_acceptance_sha256=sha(filename),
                preserved_including_acceptance=45202, models_fixed=True, policy_fixed=True)


def reference():
    spec = importlib.util.spec_from_file_location('ownership_protocol_audit_reference', ROOT / 'tests/reference.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def trajectory_facts():
    folder = ROOT / 'data/value-run-v1/arena'
    head, identity = read(folder / 'head.json'), read(folder / 'identity.json')
    require(head['identity_sha256'] == sha(folder / 'identity.json'), 'Arena identity differs')
    decisions = {}
    for entry in head['decisions']:
        require(sha(folder / entry['name']) == entry['sha256'], 'Raw decision changed')
        row = read(folder / entry['name'])
        decisions.setdefault(row['game_index'], []).append(row)
    ref = reference()
    states_by_band, suffix_by_game, upper_observations = Counter(), [], 0
    replayed = decreasing = boundary_crossings = suffix_actions = 0
    maximum_store_monotonic_error = 0
    for entry in head['games']:
        require(sha(folder / entry['name']) == entry['sha256'], 'Raw game changed')
        game = read(folder / entry['name'])
        schedule = identity['schedule'][game['index']]
        state, rows = schedule['start'], decisions[game['index']]
        require(len(rows) == game['choices'], 'Game decision length differs')
        observations = [state]
        first_suffix = 0 if active(state) <= 24 else None
        for choice, row in enumerate(rows):
            require(row['choice_index'] == choice and row['before'] == state, 'Saved decision chain differs')
            before = active(state)
            if first_suffix is not None:
                suffix_actions += 1
            ring = tuple(state['pits'][0] + [state['stores'][0]] + state['pits'][1] + [state['stores'][1]])
            transition = ref.apply(ref.Position(7, ring, state['controller']), row['action'])
            after = dict(pits=[list(transition.position.counts[:7]), list(transition.position.counts[8:15])],
                         stores=[transition.position.counts[7], transition.position.counts[15]],
                         controller=transition.position.controller)
            require(after == row['transition']['after'], 'Literal replay differs')
            a = active(after)
            require(a <= before, 'Frozen suffix not absorbing')
            error = (before - a) - (sum(after['stores']) - sum(state['stores']))
            require(error == 0 and all(v >= u for u, v in zip(state['stores'], after['stores'])),
                    'Store/active conservation differs')
            maximum_store_monotonic_error = max(maximum_store_monotonic_error, abs(error))
            decreasing += a < before
            if before >= 25 and a <= 24:
                boundary_crossings += 1
            if first_suffix is None and a <= 24:
                first_suffix = choice + 1
            state = after
            observations.append(state)
            replayed += 1
        require(state == game['final'] and active(state) == 0 and game['terminal'] and
                not game['capped'] and not game['fallbacks'], 'Raw terminal game differs')
        require(first_suffix is not None, 'Terminal game has no suffix boundary')
        suffix_by_game.append(dict(game=game['index'], first_frozen_observation=first_suffix,
                                   first_frozen_active=active(observations[first_suffix]),
                                   prefix_choices=first_suffix, suffix_choices=len(rows)-first_suffix))
        for item in observations:
            a = active(item)
            states_by_band['shared_frozen_0_24' if a <= 24 else 'learned_25_74' if a <= 74 else 'protected_upper_75_98'] += 1
            upper_observations += a >= 75
    require(replayed == 6663 and len(suffix_by_game) == 192, 'Existing replay coverage differs')
    inventory = read(ROOT / 'data/neural-value-pilot-v2/reservations/inventory.json')
    aliases = {q for component in inventory['components'].values() for q in component['quotients']}
    alias_bands = Counter('shared_frozen_0_24' if sum(bytes.fromhex(q)[:14]) <= 24 else
                         'learned_25_74' if sum(bytes.fromhex(q)[:14]) <= 74 else 'protected_upper_75_98'
                         for q in aliases)
    return dict(existing_games=192, literal_saved_action_replays=replayed,
                saved_state_observations=sum(states_by_band.values()), observations_by_band=dict(states_by_band),
                strictly_decreasing_actions=decreasing, irreversible_boundary_crossings=boundary_crossings,
                frozen_suffix_actions=suffix_actions, absorbing_boundary_proved_on_all_saved_actions=True,
                store_conservation_maximum_error=maximum_store_monotonic_error,
                terminal_games_with_frozen_boundary=len(suffix_by_game), suffix_by_game=suffix_by_game,
                frozen_inventory_quotients=len(aliases), inventory_aliases_by_band=dict(alias_bands),
                fresh_admission_inferred=False, historical_strength_reinterpretation=False)


def facts():
    prior = prior_closure()
    trajectories = trajectory_facts()
    require('torch' not in sys.modules, 'Independent audit imported Torch')
    files = [Path(__file__), ROOT / 'tests/reference.py', ROOT / CONTROL, ROOT / CANDIDATE,
             ROOT / 'data/value-run-v1/arena/head.json', ROOT / 'data/value-run-v1/arena/identity.json',
             ROOT / 'data/neural-value-pilot-v2/reservations/inventory.json']
    result = dict(schema='ownership-protocol-independent-facts-v1', passed=True, prior=prior,
                  trajectories=trajectories, inputs={str(p.relative_to(ROOT)).replace('\\', '/'):sha(p) for p in files},
                  optimizer_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0,
                  searches=0, model_forwards=0, feature_simulations=0, torch_imported=False)
    publish('facts.json', result)
    print(canonical(dict(passed=True, prior_files=45202, literal_replays=trajectories['literal_saved_action_replays'],
                         observations_by_band=trajectories['observations_by_band'],
                         inventory_aliases_by_band=trajectories['inventory_aliases_by_band'])), flush=True)
    return result


def expect_refusal(operation, label):
    try:
        operation()
    except ValueError:
        return
    raise RuntimeError('Expected a specific ValueError refusal: ' + label)


def fixture_state(a, actor=0, stores=None):
    pits = [[0] * 7, [0] * 7]
    if a:
        pits[actor][0] = max(a - 1, 1)
        if a >= 2:
            pits[actor][1] = 1
    return dict(pits=pits, stores=stores if stores is not None else [98-a, 0], controller=actor)


def contract_checks():
    # Imported only after the independent facts operation; this audits the
    # production metadata contract, not a live engine or admission builder.
    sys.path.insert(0, str(ROOT))
    from congklak import ownership_protocol as protocol
    methods = []
    dispatches = 0
    required_reset = {'tree', 'transposition_table', 'evaluator_cache', 'feature_cache',
                      'ordering_history', 'root_policy', 'visits_values', 'rng_state'}
    for a in (0, 4, 5, 24, 25, 49, 50, 74, 75, 98):
        for actor in (0, 1):
            state = fixture_state(a, actor)
            require(protocol.domain(state) == ('shared_frozen' if a <= 24 else
                                               'owned_updated' if a <= 74 else 'owned_upper'),
                    'Boundary classifier differs')
            require(protocol.root_admissible(state) == (25 <= a <= 74), 'Root band/choices differs')
            for profile in ('control', 'candidate'):
                leaf = protocol.leaf_contract(state, profile)
                expected_leaf = 'candidate_sidecar' if profile == 'candidate' and 25 <= a <= 74 else 'frozen_original'
                require(leaf['value_path'] == expected_leaf and leaf['owned'] == (a >= 25) and
                        leaf['quotient'] == quotient(state) and leaf['quotient_is_not_a_value_cache_key'] is True,
                        'Leaf bypass/ownership/key contract differs')
                for baseline in ('H_v1', 'H8_relay_CPU'):
                    for seat in (0, 1):
                        answer = protocol.dispatch(state, profile, baseline, seat)
                        expected = None if a == 0 else baseline if actor != seat else 'control' if a <= 24 else profile
                        require(answer['engine'] == expected and answer['production_execution'] is False and
                                answer['suffix_started'] == (a <= 24), 'Arm-independent seat dispatch differs')
                        if a:
                            require(set(answer['reset_required']) == required_reset and answer['clock_ms'] == 25 and
                                    answer['reset_scope'] == 'both_seats_at_handoff_and_every_decision' and
                                    answer['ownership_tracing_inside_clock'] is True and answer['cache_capacity'] == 0 and
                                    answer['handoff_required'] == (a <= 24), 'Reset/whole-choice clock differs')
                        else:
                            require(answer['decision_required'] is False and answer['boundary_event_required'] is True,
                                    'Terminal crossing incorrectly searches')
                        if a <= 24:
                            resumed = protocol.dispatch(state, profile, baseline, seat, suffix_started=True)
                            require(resumed['engine'] == expected and not resumed['handoff_required'] and
                                    not resumed['boundary_event_required'], 'Suffix resume repeats/forgets handoff')
                        else:
                            expect_refusal(lambda: protocol.dispatch(state, profile, baseline, seat, True),
                                           'Absorbing suffix reentry')
                        dispatches += 1
    methods.append('domain_leaf_dispatch_seat_reset_clock_and_absorbing_cursor')
    state = fixture_state(30)
    offset = dict(state, stores=[0, 68])
    swapped = dict(pits=[state['pits'][1], state['pits'][0]], stores=state['stores'][::-1], controller=1)
    require(quotient(state) == quotient(offset) == quotient(swapped) == protocol.leaf_contract(swapped, 'control')['quotient'],
            'Store/player alias protection differs')
    require(state != offset and state != swapped, 'Alias fixture accidentally identical')
    methods.append('canonical_ownership_aliases_are_not_raw_states_or_targets')

    def edge(family, mass, roles=(), source='scratch-only'):
        return dict(family=family, quotient=quotient(fixture_state(mass)), roles=list(roles), source=source)
    coverage = dict(raw_incidence=True, declared_sources=['scratch-only'], verified_sources=['scratch-only'],
                    missing_sources=[], unresolved_unions=[])
    # A and B were in one flattened whole-trajectory component because of L.
    # Actual incidences A-H25,A-L24,B-L24,B-H26 must split at L, whereas A's
    # H25/H75 incidences stay connected regardless of any intervening low state.
    edges = [edge('A', 25), edge('A', 24), edge('B', 24), edge('B', 26)]
    cut = protocol.graph_cut(edges, coverage)
    parts = {frozenset(c['families']) for c in cut['components']}
    require(parts == {frozenset(['A']), frozenset(['B'])} and cut['owned_edges'] == cut['shared_edges'] == 2 and
            len(cut['shared_ledger']) == 2 and not cut['production_admission'] and not cut['actual_source_certificate'],
            'Low-only old union incorrectly becomes retained cartesian incidence')
    upper = protocol.graph_cut([edge('A', 25), edge('A', 75), edge('C', 75)], coverage)
    require(len(upper['components']) == 1 and set(upper['components'][0]['families']) == {'A', 'C'} and
            upper['owned_edges'] == 3, 'Upper-band or same-origin connectivity dropped')
    methods.append('actual_incidence_low_cut_and_upper_domain_connectivity')

    for left, right in [('optimizer_support', 'arena'), ('fit_check', 'arena'), ('protected', 'arena'),
                        ('optimizer_support', 'fit_check')]:
        collision = protocol.graph_cut([edge('A', 30, [left]), edge('B', 30, [right])], coverage)
        require(len(collision['conflicting_components']) == 1 and not collision['contract_graph_consistent'],
                'High cross-role incidence not held')
    protected = protocol.graph_cut([edge('A', 24, ['protected']), edge('A', 30, ['arena'])], coverage)
    require(bool(protected['conflicting_components']), 'Low edge silently erased protected origin role')
    same_role = protocol.graph_cut([edge('A', 30, ['arena']), edge('B', 30, ['arena'])], coverage)
    require(bool(same_role['conflicting_components']) and not same_role['contract_graph_consistent'],
            'Two distinct arena reservations treated as one independent unit')
    one_reservation = protocol.graph_cut([dict(edge('A', 30, ['arena']), arena_origin='reserved-1'),
                                          dict(edge('B', 30, ['arena']), arena_origin='reserved-1')], coverage)
    require(not one_reservation['conflicting_components'] and
            one_reservation['components'][0]['arena_origins'] == ['reserved-1'],
            'Multiple source families in one reserved unit incorrectly counted as independent units')
    for incomplete in [None, dict(coverage, raw_incidence=False), dict(coverage, verified_sources=[]),
                       dict(coverage, missing_sources=['missing']), dict(coverage, unresolved_unions=['opaque'])]:
        held = protocol.graph_cut([edge('A', 30)], incomplete)
        require(not held['source_coverage_sufficient'] and not held['contract_graph_consistent'] and
                not held['production_admission'], 'Incomplete coverage admitted')
    expect_refusal(lambda: protocol.graph_cut([dict(families=['A', 'B'], quotients=[edge('A',30)['quotient']])], coverage),
                   'Flattened component mistaken for incidence')
    methods.append('role_and_opaque_incidence_fail_closed')

    spec = protocol.contract()
    require(spec['prospective_only'] and not spec['retroactive_old_arena_reclassification'] and
            spec['models_fixed'] and spec['policy_fixed'] and not spec['training_authorized'] and
            not spec['execution_ready'] and spec['suffix']['live_implementation_verified'] is False and
            spec['evaluation']['families'] == 24 and spec['evaluation']['games'] == 192 and
            spec['evaluation']['baselines'] == ['H_v1', 'H8_relay_CPU'] and
            not spec['evaluation']['confidence_intervals_enabled'] and
            not spec['evaluation']['iid_components_assumed'] and
            not spec['evaluation']['posthoc_filtering'] and not spec['evaluation']['posthoc_cluster_merging'] and
            not spec['evaluation']['current_strength_interpretation_allowed'], 'Prospective bounded scope differs')
    require(len(spec['pending']) >= 5, 'Missing implementation gates hidden')
    protocol.validate_spec(read(ROOT/'build/ownership-protocol-v1/design/compiled-section-spec.json'))
    altered = read(ROOT/'build/ownership-protocol-v1/design/compiled-section-spec.json')
    altered['machine_contract']['execution_ready'] = True
    expect_refusal(lambda: protocol.validate_spec(altered), 'Designer-machine mismatch')
    methods.append('prospective_bounded_estimand_no_automatic_admission_or_old_salvage')
    require('torch' not in sys.modules, 'Contract audit imported Torch')
    output = dict(schema='ownership-protocol-independent-contract-checks-v1', passed=True,
                  methods=methods, method_count=len(methods), dispatch_cases=dispatches,
                  graph_source_certificate_claimed=False, live_suffix_executor_tested=False,
                  production_admission=False, torch_imported=False,
                  counters=dict(optimizer_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0,
                                searches=0, model_forwards=0, feature_simulations=0),
                  inputs={name:sha(ROOT/name) for name in ['tests/ownership_protocol_independent_audit.py',
                      'congklak/ownership_protocol.py', 'docs/PROSPECTIVE_OWNERSHIP_PROTOCOL.md',
                      'build/ownership-protocol-v1/design/section-spec.json',
                      'build/ownership-protocol-v1/design/compiled-section-spec.json']})
    publish('contract-checks.json', output)
    print(canonical(dict(passed=True, distinct_methods=len(methods), dispatch_cases=dispatches)), flush=True)
    return output


def file_snapshot(paths):
    result = {}
    for path in paths:
        path = Path(path)
        stat = path.stat()
        result[str(path.relative_to(ROOT)).replace('\\', '/')] = dict(
            sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return result


def prefix_files(folder):
    paths = [folder / 'identity.json']
    for commit in sorted((folder / 'commits').glob('*.json')):
        paths.extend([commit, folder / read(commit)['result']])
    paths.extend(p for p in folder.glob('*.json') if p.name in ('plan.json', 'projection.json', 'contract-checks.json'))
    return sorted(set(paths))


def resume_one(path, calls):
    sys.path.insert(0, str(ROOT))
    from congklak import ownership_protocol_flow_v2 as flow
    original = flow.compute
    def logged(entry):
        with Path(calls).open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(canonical(dict(index=entry['index'], kind=entry['kind'])) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return original(entry)
    with mock.patch.object(flow, 'compute', side_effect=logged):
        result = flow.run(path)
    require('torch' not in sys.modules, 'Recovery imported Torch')
    print(canonical(dict(passed=True, committed=result['catalog']['committed_units'])), flush=True)


def recovery_checks():
    run_id = uuid.uuid4().hex
    base = ROOT / ('build/ownership-protocol-v1/fixtures/audit-recovery-v2-' + run_id)
    base.mkdir(parents=True)
    records = []
    expected_hashes = None
    for hook, expected_prefix in [('before_unit_commit', 1), ('after_unit_commit', 2), ('before_catalog', 195)]:
        folder, marker = base / hook, base / (hook + '.paused.json')
        first_log, resumed_log, calls = base/(hook+'.first.log'), base/(hook+'.resume.log'), base/(hook+'.compute.jsonl')
        command = [sys.executable, str(ROOT/'tools/run-ownership-protocol-v2.py'), 'run', '--path', str(folder),
                   '--pause-at', hook, '--pause-marker', str(marker), '--pause-index', '1']
        with first_log.open('w', encoding='utf-8') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic() + 90
            try:
                while not marker.exists():
                    require(proc.poll() is None, 'Recovery worker failed before pause: ' + hook)
                    require(time.monotonic() < deadline, 'Recovery worker pause timed out: ' + hook)
                    time.sleep(.05)
                event = read(marker)
                require(event['pid'] == proc.pid and event['point'] == hook, 'Pause belongs to another worker')
                prefix = sorted((folder/'commits').glob('*.json'))
                require(len(prefix) == expected_prefix and not (folder/'catalog.json').exists(),
                        'Unexpected committed prefix at kill point')
                before = file_snapshot(prefix_files(folder))
                orphan_units = sorted(p.name for p in (folder/'units').glob('*.json')
                                      if p.name not in {Path(read(c)['result']).name for c in prefix})
                if hook == 'before_unit_commit':
                    require(len(orphan_units) == 1, 'Uncommitted unit was not actually computed')
                proc.kill()
                proc.wait(timeout=10)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait(timeout=10)
        require(proc.returncode != 0, 'Worker was not actually terminated')
        publish('v2/killed-' + hook + '.json', dict(hook=hook, pid=proc.pid, returncode=proc.returncode,
            committed_at_kill=expected_prefix, before_prefix=before, uncommitted_units=orphan_units,
            folder=str(folder.relative_to(ROOT)).replace('\\','/')))
        with resumed_log.open('w', encoding='utf-8') as stream:
            resumed = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'resume-one',
                                      '--path', str(folder), '--calls', str(calls)], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=90,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        require(resumed.returncode == 0, 'Recovery resume failed: ' + hook)
        observed = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        require([r['index'] for r in observed] == list(range(expected_prefix, 195)),
                'Committed units were recomputed or an uncommitted unit was skipped')
        after = file_snapshot(ROOT/name for name in before)
        require(before == after, 'Committed prefix SHA/size/mtime changed after recovery')
        catalog = read(folder/'catalog.json')
        hashes = {name:sha(folder/(name+'.json')) for name in ('plan', 'projection', 'contract-checks')}
        require(catalog['committed_units'] == 195 and catalog['reports'] == hashes, 'Recovered catalog differs')
        if expected_hashes is None: expected_hashes = hashes
        require(hashes == expected_hashes, 'Fresh/recovered logical reports differ across kill points')
        record = dict(hook=hook, killed_pid=proc.pid, actual_terminated_returncode=proc.returncode,
                      folder=str(folder.relative_to(ROOT)).replace('\\','/'), committed_at_kill=expected_prefix,
                      compute_indices_after_resume=[r['index'] for r in observed],
                      uncommitted_units_retained=orphan_units, prefix_unchanged=True,
                      before_prefix=before, after_prefix=after, reports=hashes,
                      invocation=dict(command=command, first_log=str(first_log.relative_to(ROOT)).replace('\\','/'),
                                      resume_log=str(resumed_log.relative_to(ROOT)).replace('\\','/')))
        records.append(record)
        publish('v2/recovery-' + hook + '.json', record)
        print(canonical(dict(hook=hook, passed=True, committed_at_kill=expected_prefix,
                             computed_after_resume=len(observed))), flush=True)
    result = dict(schema='ownership-protocol-independent-recovery-v1', passed=True,
                  actual_process_kills=3, cases=records, reports=expected_hashes,
                  recovery_scope='protocol publication and literal saved-action diagnostics; not a live suffix worker',
                  model_forwards=0, searches=0, optimizer_steps=0, optimizer_draws=0,
                  new_games=0, new_teacher_labels=0, feature_simulations=0,
                  literal_saved_action_replays=20020,
                  inputs={name:sha(ROOT/name) for name in ['tests/ownership_protocol_independent_audit.py',
                      'congklak/ownership_protocol.py','congklak/ownership_protocol_flow_v2.py',
                      'congklak/ownership_domain_projection.py','tools/run-ownership-protocol-v2.py']})
    publish('v2/recovery.json', result)
    return result


def completed_checks():
    sys.path.insert(0, str(ROOT))
    from congklak import ownership_protocol_flow_v2 as flow
    recovery = read(OUT/'v2/recovery.json')
    original = ROOT / recovery['cases'][0]['folder']
    base = ROOT / ('build/ownership-protocol-v1/fixtures/audit-corruption-v2-' + uuid.uuid4().hex)
    base.mkdir(parents=True)
    calls = Counter()
    class ForbiddenUse(RuntimeError): pass
    def forbidden(name):
        def invoke(*args, **kwargs):
            calls[name] += 1
            raise ForbiddenUse('Forbidden completed-reader work: ' + name)
        return invoke
    def snapshot_all(folder):
        return file_snapshot(p for p in folder.rglob('*') if p.is_file())
    patchers = [mock.patch.object(flow, name, side_effect=forbidden(name)) for name in
                ('Writer','compute','projection_module','prepare','atomic','section_spec')]
    original_import = __import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] == 'torch' or name.startswith(('congklak.neural', 'congklak.value_training')):
            return forbidden('forbidden_import_' + name)()
        return original_import(name, *args, **kwargs)
    patchers.append(mock.patch('builtins.__import__', side_effect=guarded_import))
    records = []
    for patcher in patchers: patcher.start()
    try:
        before = snapshot_all(original)
        flow.completed(original); flow.run(original)
        require(before == snapshot_all(original), 'Completed API/run mutated namespace')
        records.append(dict(method='completed_API_and_run_before_writer_or_computation', passed=True))
        for kind in ('catalog_scope','catalog_count','identity_scope','commit_chain','unit_bytes',
                     'missing_unit','plan_scope','projection_admission','contract_live_claim'):
            folder = base/kind
            shutil.copytree(original, folder)
            catalog_path = folder/'catalog.json'
            catalog = read(catalog_path)
            if kind.startswith('catalog'):
                catalog['training_authorized' if kind == 'catalog_scope' else 'committed_units'] = True if kind == 'catalog_scope' else 194
                catalog_path.write_text(canonical(catalog)+'\n', encoding='utf-8')
            elif kind == 'identity_scope':
                path = folder/'identity.json'; body = read(path); body['execution_ready'] = True
                path.write_text(canonical(body)+'\n', encoding='utf-8')
            elif kind == 'commit_chain':
                path=folder/'commits/000001.json'; body=read(path); body['previous_commit_sha256']='0'*64
                path.write_text(canonical(body)+'\n',encoding='utf-8')
            elif kind in ('unit_bytes','missing_unit'):
                path=folder/read(folder/'commits/000001.json')['result']
                if kind == 'missing_unit': path.unlink()
                else: path.write_text(path.read_text()+' ',encoding='utf-8')
            else:
                report, field = {'plan_scope':('plan','training_authorized'),
                                 'projection_admission':('projection','production_admission'),
                                 'contract_live_claim':('contract-checks','live_suffix_worker_tested')}[kind]
                path=folder/(report+'.json'); body=read(path); body[field]=True
                path.write_text(canonical(body)+'\n',encoding='utf-8')
                catalog['reports'][report]=sha(path)
                catalog_path.write_text(canonical(catalog)+'\n',encoding='utf-8')
            before=snapshot_all(folder)
            for operation in (lambda:flow.completed(folder),lambda:flow.run(folder)):
                try: operation()
                except (ValueError,FileNotFoundError): pass
                else: raise RuntimeError('Corrupt completion was accepted: '+kind)
            require(before == snapshot_all(folder), 'Corrupt completion path mutated namespace: '+kind)
            records.append(dict(method=kind, passed=True, no_mutation=True,
                                folder=str(folder.relative_to(ROOT)).replace('\\','/')))
    finally:
        for patcher in reversed(patchers): patcher.stop()
    require(not calls and 'torch' not in sys.modules, 'Completed reader invoked forbidden work')
    # An unfinished forged task plan must immutable-compare with the true
    # metadata plan before it can execute any unit, although acquiring its
    # own writer lock is allowed for this unfinished-publication test.
    folder=base/'forged_unfinished_plan';shutil.copytree(original,folder)
    (folder/'catalog.json').unlink()
    identity=read(folder/'identity.json');identity['tasks'][1]['task']['game_index']=191
    (folder/'identity.json').write_text(canonical(identity)+'\n',encoding='utf-8')
    with mock.patch.object(flow,'compute',side_effect=forbidden('unfinished_compute')):
        expect_refusal(lambda:flow.run(folder),'Forged unfinished metadata plan')
    require(not calls,'Unfinished forged plan performed computation')
    records.append(dict(method='unfinished_plan_recreated_before_unit_work',passed=True))
    result=dict(schema='ownership-protocol-independent-completed-checks-v1',passed=True,
                methods=records,method_count=len(records),forbidden_calls=dict(calls),torch_imported=False,
                optimizer_steps=0,optimizer_draws=0,new_games=0,new_teacher_labels=0,searches=0,
                model_forwards=0,feature_simulations=0,
                inputs={name:sha(ROOT/name) for name in ['tests/ownership_protocol_independent_audit.py',
                    'congklak/ownership_protocol_flow_v2.py','congklak/ownership_protocol.py']})
    publish('v2/completed-checks.json',result)
    print(canonical(dict(passed=True,method_count=len(records),forbidden_calls=dict(calls))),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['facts', 'contracts', 'recovery', 'completed', 'resume-one'])
    parser.add_argument('--path',type=Path)
    parser.add_argument('--calls',type=Path)
    args = parser.parse_args()
    if args.operation == 'facts': facts()
    elif args.operation == 'contracts': contract_checks()
    elif args.operation == 'recovery': recovery_checks()
    elif args.operation == 'completed': completed_checks()
    else: resume_one(args.path,args.calls)
