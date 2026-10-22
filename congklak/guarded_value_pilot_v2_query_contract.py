"""Finite source-derived development schedule over the screened reservations.

The acquisition's labels/features are immutable input data. This contract has
no teacher-label operation and creates no replacement positions or targets.
"""
import hashlib
from . import config, guarded_value_pilot_v2_common as s
from . import guarded_value_pilot_v2_data as data

SCHEMA = s.SCHEMA + '-queries'
OPERATIONS = ('conformance', 'guards', 'arena')
SHARED = data.SHARED
LABEL_KILL_HOOKS = ()
HANDOFF_KILL_HOOKS = ('before_handoff_commit', 'after_handoff_commit')
UNKNOWN_HOOK = 'before_query'
FIXTURE_BASE = s.FOLDER / 'fixtures/native-recovery-v2'


def _salt(prefix, value):
    return hashlib.sha256((prefix + value).encode()).hexdigest()


def membership():
    return data.membership()


def schedule(operation, fixture=False):
    s.need(operation in OPERATIONS and type(fixture) is bool, 'Unknown query schedule scope')
    members = membership()
    rows = []
    if operation == 'conformance':
        if fixture:
            meta = members['fit'][0]
            rows.append(dict(kind='features', engine='control', meta=meta))
            for engine in ('control', 'candidate'):
                rows.append(dict(kind='prediction', engine=engine, meta=meta))
            for engine in ('control', 'candidate'):
                rows.append(dict(kind='decision', engine=engine, mode='fixed128', meta=meta, cohort='initial_equivalence'))
        else:
            roots = sorted(members['fit'], key=lambda row: _salt('ordinary-epoch-conformance-20261016/', row['state_key']))
            for index, meta in enumerate(roots):
                for engine in (('control', 'candidate') if index % 2 == 0 else ('candidate', 'control')):
                    rows.append(dict(kind='prediction', engine=engine, meta=meta))
    elif operation == 'guards':
        cases = [row for row in members['fit'] if row['state_key'] in members['cases']]
        for cohort, roots in (('cases', cases[:1] if fixture else cases),
                              ('endgame', members['exact'][:1] if fixture else members['exact'])):
            for index, meta in enumerate(sorted(roots, key=lambda row: _salt('ordinary-epoch-cases-20261016/', row['state_key']))):
                for mode_index, mode in enumerate(('timed25', 'fixed128') if index % 2 == 0 else ('fixed128', 'timed25')):
                    for engine in (('control', 'candidate') if (index + mode_index) % 2 == 0 else ('candidate', 'control')):
                        rows.append(dict(kind='decision', engine=engine, mode=mode, meta=meta, cohort=cohort))
    else:
        roots = sorted(members['arena'], key=lambda row: _salt('ordinary-epoch-arena-20261016/', row['state_key']))
        if fixture:
            return [dict(index=0, kind='game', meta=roots[0], baseline='H_v1', seat=0, profile='candidate')]
        cycle = [('H_v1', 0), ('H8_relay_CPU', 1), ('H_v1', 1), ('H8_relay_CPU', 0)]
        for index, meta in enumerate(roots):
            offset = index % 4
            matches = cycle[offset:] + cycle[:offset]
            for matchup, (baseline, seat) in enumerate(matches):
                for profile in (('control', 'candidate') if (index + matchup) % 2 == 0 else ('candidate', 'control')):
                    rows.append(dict(kind='game', meta=meta, baseline=baseline, seat=seat, profile=profile))
    return [dict(row, index=index) for index, row in enumerate(rows)]


def classify(output, operation, fixture=False):
    path = s.path_of(output)
    s.need(operation in OPERATIONS and type(fixture) is bool, 'Unknown campaign scope')
    if not fixture:
        s.need(path == s.FOLDER / 'jobs' / operation, 'Unregistered production query path')
        return dict(scope='production', subset_only=False, hook=None)
    if path == s.FOLDER / 'jobs/initial-conformance':
        s.need(operation == 'conformance', 'Initial equivalence operation differs')
        return dict(scope='initial_equivalence', subset_only=False, hook=None)
    s.need(path.parent == FIXTURE_BASE and path.name in HANDOFF_KILL_HOOKS and operation == 'arena',
           'Unregistered native recovery fixture')
    return dict(scope='handoff_recovery', subset_only=False, hook=path.name)


def predecessor(output, operation, fixture=False):
    scope = classify(output, operation, fixture)
    if scope['scope'] == 'initial_equivalence':
        return None  # Complete accepted capture closure is static, live claims start empty.
    if scope['scope'] == 'handoff_recovery':
        if scope['hook'] == 'before_handoff_commit':
            path, previous_operation = s.FOLDER / 'jobs/guards', 'guards'
        else:
            path, previous_operation = FIXTURE_BASE / 'before_handoff_commit', 'arena'
    elif operation == 'conformance':
        path, previous_operation = s.FOLDER / 'jobs/initial-conformance', 'conformance'
    elif operation == 'guards':
        path, previous_operation = s.FOLDER / 'jobs/conformance', 'conformance'
    else:
        path, previous_operation = FIXTURE_BASE / 'after_handoff_commit', 'arena'
    return dict(path=s.relative(path), operation=previous_operation)


def derive_registry(plan_sha256=None):
    from . import guarded_value_pilot_v2_guard as guard
    registry = guard.registry()
    s.need(plan_sha256 is None or registry['plan_sha256'] == plan_sha256, 'Registry plan differs')
    return registry, s.read(s.ROOT / registry['graph']['path'])
