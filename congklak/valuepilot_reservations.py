"""Metadata-only, role-aware reservations for the fixed-policy value pilot.

No target values, predictions or new search/game results enter selection. The
entire current family/quotient graph is rebuilt before jointly matching roles.
Publication uses immutable prepared files, a catalog commit and a certificate.
Completed readers validate before touching the writer lock.
"""
from collections import Counter, defaultdict
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time
from . import config, neurodiag_common as common, neurodiag_data as previous

ROOT = common.ROOT
DEFAULT = ROOT / 'data/neural-value-pilot-v1/reservations'
PLAN = ROOT / 'build/valuepilot-plan-v1/plan.json'
PLAN_ACCEPTANCE = ROOT / 'build/valuepilot-plan-v1/acceptance.json'
SCHEMA = 'congklak-valuepilot-reservations-v1'
SALT = 'neural-value-pilot/v1/data/'
SOURCES = ('congklak/valuepilot_reservations.py', 'tools/reserve-valuepilot.py')
PLANS = tuple('build/' + n for n in (
    'stage5-benchmark-plan.json', 'stage5-action-benchmark-plan.json', 'stage5-action-benchmark-plan-v2.json',
    'stage5-deeper-benchmark-plan.json', 'phase6-plan.json', 'puct-cpu-plan.json',
    'quality-plan.json', 'value-plan.json', 'value-pilot-benchmark-plan.json',
    'selfplay-pilot-benchmark-plan.json', 'ordinary-value-benchmark-plan.json'))
ARENAS = tuple('data/' + n + '/work.sqlite3' for n in (
    'neural-benchmark', 'neural-benchmark-v2', 'neural-benchmark-v3',
    'stage5-benchmark', 'stage5-action-benchmark', 'stage5-action-benchmark-v2',
    'stage5-deeper-benchmark', 'phase6-benchmark-v1', 'puct-cpu-benchmark-v1',
    'quality-benchmark-v1', 'value-benchmark-v1',
    'value-pilot-benchmark-pretraining-v1', 'value-pilot-benchmark-posttraining-v1',
    'selfplay-pilot-benchmark-v1', 'ordinary-value-benchmark-v1'))
EXTRA = (*PLANS, *ARENAS, 'build/neurodiag-panel.json',
         'build/phase6-protected-registry.json',
         'data/pilot-10000/work.sqlite3',
         'build/ordinary-value-benchmark-projection.json', 'build/divergence-panel.json',
         'congklak/neurodiag_data.py', 'congklak/neurodiag_common.py',
         'congklak/native.py', 'build/congklak_native.dll')
require, read, sha = common.require, common.read, common.sha


def _atomic(path, body):
    path = Path(path)
    text = config.canonical(body) + '\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Immutable reservation file differs: ' + str(path))
        return body
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    return body


def _path(path):
    path = Path(path).resolve()
    require(path == DEFAULT.resolve() or path.is_relative_to(ROOT / 'build'),
            'Reservation output must be production namespace or build fixture')
    return path


def _json_metadata(filename, group):
    """Project positions/origins without parsing reference/teacher value bodies."""
    text = (ROOT / filename).read_text(encoding='utf-8')
    import sqlite3
    with closing(sqlite3.connect(':memory:')) as db:
        query = "SELECT key,json_extract(value,'$.state'),json_extract(value,'$.state_key'),json_extract(value,'$.family'),json_extract(value,'$.stratum') FROM json_each(?,?) ORDER BY key"
        return [dict(index=i, state=json.loads(p), state_key=k or common.rawkey(json.loads(p)),
                     family=f, stratum=s) for i, p, k, f, s in db.execute(query, (text, '$.' + group)) if p is not None]


class Graph:
    def __init__(self):
        self.parent = {}; self.reasons = defaultdict(set)

    def find(self, node):
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left

    def add(self, family, state, *reasons):
        require(isinstance(family, str) and family, 'Missing origin family')
        node = ('family', family)
        self.join(node, ('quotient', previous.quotient(state)))
        self.reasons[node].update(reasons)

    def mark(self, kind, values, reason):
        for value in values:
            node = (kind, value); self.find(node); self.reasons[node].add(reason)

    def finish(self):
        members = defaultdict(list); reasons = defaultdict(set)
        for node in self.parent:
            members[self.find(node)].append(node)
        for node, why in self.reasons.items():
            reasons[self.find(node)].update(why)
        components = {}; lookup = {}
        for root, nodes in members.items():
            body = dict(families=sorted(v for k, v in nodes if k == 'family'),
                        quotients=sorted(v for k, v in nodes if k == 'quotient'),
                        reasons=sorted(reasons[root]))
            # Isolated protected quotient tombstones are valid components.
            cid = config.digest(body); components[cid] = body
            for node in nodes: lookup[node] = cid
        return dict(sorted(components.items())), lookup


def _accepted_sources():
    accepted = read(PLAN_ACCEPTANCE)
    require(accepted.get('passed') is True, 'Accepted value pilot plan missing')
    files = set(previous.SOURCE_FILES) | set(EXTRA)
    inherited = read(ROOT / 'build/phase6-protected-registry.json')
    require(sha(ROOT / 'build/phase6-protected-registry.json') == accepted['inputs']['build/phase6-protected-registry.json'], 'Inherited trajectory registry changed')
    older = {r['path'] + '/work.sqlite3': r['database_sha256'] for r in inherited['sources'] if 'database_sha256' in r}
    pins = {}
    for p in sorted(files):
        if p in accepted['inputs']: pins[p] = accepted['inputs'][p]
        elif p in older: pins[p] = older[p]
        else:
            # The initial Stage5 plan is an additional conservative ownership
            # projection. Freeze its current metadata; it cannot grant admission.
            require(p in ('build/stage5-benchmark-plan.json', 'build/stage5-action-benchmark-plan-v2.json', 'data/pilot-10000/work.sqlite3'), 'Uncertified ownership source: ' + p)
            pins[p] = sha(ROOT / p)
    pins.update(common.pins([PLAN, PLAN_ACCEPTANCE]))
    common.current(pins)
    return pins


def inventory():
    require('torch' not in sys.modules, 'Metadata reservation forbids Torch')
    inputs = _accepted_sources()
    prior = previous.metadata_inventory()
    graph = Graph()
    # Preserve the earlier complete closure and all optimizer exclusions. Arena
    # ownership is rebuilt separately: original development alone is allowed.
    for cid, component in prior['components'].items():
        nodes = [('family', f) for f in component['families']] + [('quotient', q) for q in component['quotients']]
        if not nodes: continue
        for node in nodes: graph.join(nodes[0], node)
        if component['blocked_reasons']:
            graph.reasons[nodes[0]].add('optimizer/inherited_protection')
        if any(r.startswith('ancestral/') for r in component['blocked_reasons']):
            graph.reasons[nodes[0]].add('arena/ancestral_reserved_or_evaluated')

    base = previous.metadata_rows('deeper-9000-v2', 'roots', ['state_key', 'state', 'family', 'stratum', 'legal_mask', 'split'])
    origins = len(base)
    for row in base:
        graph.add(row['family'], row['state'], *(['optimizer/original_development'] if row['split'] != 'train' else []))
    for name in previous.HISTORIES:
        for row in previous.metadata_rows(name, 'roots', ['state_key', 'state', 'family', 'stratum', 'legal_mask', 'cohort']):
            graph.add(row['family'], row['state'], *(['optimizer/history_validation', 'arena/history_validation'] if row['cohort'] != 'training_candidate' else []))
            origins += 1
    with closing(previous.ro('ordinary-selfplay-pilot-v1')) as db:
        for family, cohort, text in db.execute("SELECT json_extract(body,'$.family'),json_extract(body,'$.cohort'),json_extract(body,'$.candidates') FROM families ORDER BY idx"):
            for row in json.loads(text):
                graph.add(family, row['state'], *(['optimizer/history_validation', 'arena/history_validation'] if cohort != 'training_candidate' else [])); origins += 1
    for name in ('selfplay-training-pilot-v1', 'ordinary-selfplay-pilot-v1'):
        for row in previous.metadata_rows(name, 'sp_rows', ['id', 'state_key', 'state', 'family'], 'id'):
            graph.add(row['family'], row['state']); origins += 1
    # Reapply the complete inherited optimizer tombstones after adding aliases
    # missing from the old eligible-only graph. Their new bridges are protected.
    graph.mark('family', prior['protected_families'], 'optimizer/protected_family')
    graph.mark('quotient', prior['protected_quotients'], 'optimizer/protected_quotient')

    for filename in ('allocation-weighting-reservations.json', 'stratum-reservations.json', 'qstability-reservations.json'):
        overlay = read(ROOT / 'build' / filename)
        for role in (overlay['roles'].values() if 'roles' in overlay else [overlay]):
            graph.mark('family', role['origin_families'], 'arena/prior_reservation')
            graph.mark('quotient', role['eligible_quotients'], 'arena/prior_reservation')
    older = read(ROOT / 'build/allocation-weighting-inventory.json')
    graph.mark('family', older['prior_H12_H14_exposed_families'], 'arena/prior_H12_H14')
    graph.mark('quotient', older['prior_H12_H14_exposed_quotients'], 'arena/prior_H12_H14')

    protected_cases = []; previous_arena = set()
    for filename in PLANS:
        for group in ('cases', 'arena_starts', 'validation_roots'):
            for row in _json_metadata(filename, group):
                if row['family']:
                    graph.add(row['family'], row['state'], 'optimizer/prior_case_or_arena', 'arena/prior_case_or_arena')
                    if group == 'arena_starts': previous_arena.add(row['family'])
                else:
                    graph.mark('quotient', [previous.quotient(row['state'])], 'arena/prior_case_or_arena')
                protected_cases.append(dict(source=filename, group=group, **row))
    for row in _json_metadata('build/divergence-panel.json', 'roots'):
        graph.add(row['family'], row['state'], 'optimizer/forensic', 'arena/forensic')
        protected_cases.append(dict(source='build/divergence-panel.json', group='roots', **row))

    # Legacy games persist literal actions, not all intermediate states. Replay
    # those existing actions solely to recover ownership positions; no search,
    # features, predictions, target values or new games are generated.
    from .native import Native, flatten
    import sqlite3
    native = Native(); transitions = games = observations = 0
    family_by_key = {r['state_key']: r['family'] for r in base}
    with closing(sqlite3.connect((ROOT / 'data/pilot-10000/work.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for key, family in db.execute("SELECT state,family FROM selected WHERE split='development' ORDER BY state"):
            family_by_key.setdefault(bytes(key).hex(), family)
    for filename in ARENAS:
        with closing(sqlite3.connect((ROOT / filename).resolve().as_uri() + '?mode=ro', uri=True)) as db:
            for start, moves, family, start_key, final in db.execute("SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.start_key'),json_extract(body,'$.final') FROM games ORDER BY hash"):
                family = family or family_by_key.get(start_key)
                require(family is not None, 'Historical arena origin cannot be resolved')
                previous_arena.add(family); p = json.loads(start)
                normalized = native.normalize(flatten(p), actor=p['controller'], total=98)
                require(normalized.get('ok'), 'Historical arena start invalid'); p = normalized['position']
                for action in [*json.loads(moves), None]:
                    if sum(map(sum, p['pits'])) > 4:
                        graph.add(family, p, 'optimizer/prior_trajectory', 'arena/prior_trajectory'); observations += 1
                    if action is not None:
                        result = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                        require(result.get('ok'), 'Historical literal trajectory invalid')
                        p = result['position']; normal = native.normalize(flatten(p), actor=p['controller'], total=98)
                        require(normal.get('ok'), 'Historical literal normalization invalid'); p = normal['position']; transitions += 1
                require(p == json.loads(final), 'Historical arena final state differs'); games += 1

    # Exclude every currently connected alias of each inspected diagnostic root.
    panel = read(ROOT / 'build/neurodiag-panel.json')
    diagnostics = [r for group in ('primary', 'paired', 'MC') for r in panel[group]]
    for row in diagnostics:
        graph.add(row['family'], row['state'], 'optimizer/neurodiag_inspected', 'arena/neurodiag_inspected')
    components, lookup = graph.finish()
    def describe(row):
        cid = lookup[('family', row['family'])]
        require(cid == lookup[('quotient', previous.quotient(row['state']))], 'Origin quotient closure differs')
        require(common.rawkey(row['state']) == row['state_key'], 'Exact raw state metadata differs')
        return dict(row, component=cid, quotient=previous.quotient(row['state']), actor=row['state']['controller'], cell=row['stratum'] + '/actor' + str(row['state']['controller']))
    primary = [describe(r) for r in prior['primary_pool']]
    arena = [describe(r) for r in base if r['split'] == 'development' and r['legal_mask'].bit_count() >= 2
             and sum(map(sum, r['state']['pits'])) >= 25
             and abs(r['state']['stores'][0] - r['state']['stores'][1]) < sum(map(sum, r['state']['pits']))
             and r['stratum'].split('/')[-1] in ('ordinary', 'capture', 'bonus')]
    return dict(schema=SCHEMA + '-inventory', components=components, primary_pool=primary, arena_pool=arena,
                source_inputs=inputs, previous_arena_families=sorted(previous_arena), protected_cases=protected_cases,
                counts=dict(full_components=len(components), metadata_origin_observations=origins,
                            inspected_diagnostic_rawstates=len({r['state_key'] for r in diagnostics}),
                            inspected_diagnostic_prior_components=len({r['component'] for r in diagnostics}),
                            historical_games_replayed=games, literal_transitions_replayed=transitions,
                            historical_active_observations=observations),
                target_bodies_read=0, model_forwards=0, searches=0, optimizer_steps=0, new_games=0,
                role_note='Original-development-only ownership blocks optimizer; arena excludes actual earlier case/arena/forensic/trajectory/reservation closure. All exact raw states retain their own target references.')


def select(inv):
    plan = read(PLAN); data = plan['data']; cells = tuple(data['cells'])
    require(data['hash_salt'] == SALT and data['train_per_cell'] == 64 and data['fit_check_per_cell'] == 16, 'Accepted quota/salt changed')
    components = inv['components']
    clean = lambda row, role: not any(r.startswith(role + '/') for r in components[row['component']]['reasons'])
    options = {}
    for cell in cells:
        options['trainfit/' + cell] = sorted((r for r in inv['primary_pool'] if r['cell'] == cell and r['legal_mask'].bit_count() >= 2 and clean(r, 'optimizer')), key=lambda r: (config.digest(SALT + r['state_key']), r['state_key']))
    for mechanism in ('ordinary', 'capture', 'bonus'):
        options['arena/' + mechanism] = sorted((r for r in inv['arena_pool'] if r['stratum'].endswith('/' + mechanism) and clean(r, 'arena') and r['family'] not in inv['previous_arena_families']), key=lambda r: (config.digest(SALT + 'arena/' + r['state_key']), r['state_key']))
    slots = [(role, n) for role in options for n in range(8 if role.startswith('arena/') else 80)]
    assigned = {}; chosen = {}
    def place(slot, seen):
        for row in options[slot[0]]:
            cid = row['component']
            if cid in seen: continue
            seen.add(cid)
            if cid not in assigned or place(assigned[cid], seen):
                assigned[cid] = slot; chosen[slot] = row; return True
        return False
    availability = {role: dict(roots=len(rows), components=len({r['component'] for r in rows})) for role, rows in options.items()}
    for slot in slots:
        if not place(slot, set()):
            blocker = 'Joint metadata component quota shortage: ' + slot[0] + '; no fillers or replacements'
            return dict(schema=SCHEMA + '-selection-hold', status='hold', admitted=False,
                        selection_salt=SALT, readiness_blockers=[blocker],
                        train=[], fit_check=[], fit_cases=[], arena=[], selected_components={},
                        counts=dict(train=0, fit_check=0, fit_cases=0, arena=0, globally_distinct_components=0),
                        requested_counts=dict(train=512, fit_check=128, fit_cases=32, arena=24, globally_distinct_components=664),
                        availability=availability, feasible_uncommitted_optimizer_components=sum(not k[0].startswith('arena/') for k in chosen),
                        attempted_joint_components=len(assigned), metadata_before_targets=True,
                        fit_check_is_unseen=False, policy_fixed=True, training_started=False,
                        target_bodies_read=0, model_forwards=0, searches=0, optimizer_steps=0, new_games=0)
    train = []; fit = []; cases = []; arena = []
    for cell in cells:
        rows = sorted((chosen[('trainfit/' + cell, n)] for n in range(80)), key=lambda r: (config.digest(SALT + r['state_key']), r['state_key']))
        train.extend(rows[:64]); fit.extend(rows[64:]); cases.extend(rows[64:68])
    for mechanism in ('ordinary', 'capture', 'bonus'):
        arena.extend(chosen[('arena/' + mechanism, n)] for n in range(8))
    slim = lambda row: {k: v for k, v in row.items() if k not in ('historical_role',)}
    body = dict(schema=SCHEMA + '-selection', selection_salt=SALT,
                status='reserved', admitted=True, readiness_blockers=[],
                train=sorted(map(slim, train), key=lambda r: r['state_key']),
                fit_check=sorted(map(slim, fit), key=lambda r: r['state_key']),
                fit_cases=sorted(map(slim, cases), key=lambda r: r['state_key']),
                arena=sorted(map(slim, arena), key=lambda r: r['state_key']),
                selected_components={cid: components[cid] for cid in sorted(assigned)},
                counts=dict(train=512, fit_check=128, fit_cases=32, arena=24, globally_distinct_components=664),
                availability=availability,
                metadata_before_targets=True, fit_check_is_unseen=False, policy_fixed=True, training_started=False,
                target_bodies_read=0, model_forwards=0, searches=0, optimizer_steps=0, new_games=0)
    validate_selection(body, inv)
    return body


def validate_selection(body, inv):
    families = set(); quotients = set()
    for cid, component in inv['components'].items():
        require(config.digest(component) == cid, 'Component content digest differs')
        fs, qs = set(component['families']), set(component['quotients'])
        require(len(fs) == len(component['families']) and len(qs) == len(component['quotients']), 'Component duplicate members')
        require(not fs & families and not qs & quotients, 'Graph components share an origin/quotient')
        families.update(fs); quotients.update(qs)
    if body['schema'] == SCHEMA + '-selection-hold':
        require(body['admitted'] is False and body['status'] == 'hold' and body['readiness_blockers'], 'Invalid hold admission')
        require(all(body[k] == [] for k in ('train', 'fit_check', 'fit_cases', 'arena')) and body['selected_components'] == {}, 'HOLD must not reserve any partial support')
        require(all(n == 0 for n in body['counts'].values()), 'HOLD counts falsely claim reservations')
        return body
    require(body['schema'] == SCHEMA + '-selection', 'Reservation selection schema differs')
    rows = body['train'] + body['fit_check'] + body['arena']
    require(len(rows) == len({r['component'] for r in rows}) == len({r['family'] for r in rows}) == 664, 'Joint role components/families overlap')
    for role, quota in (('train', 64), ('fit_check', 16)):
        require(Counter(r['cell'] for r in body[role]) == {c: quota for c in previous.CELLS}, 'Training/fit cell quota differs')
    require(Counter(r['cell'] for r in body['fit_cases']) == {c: 4 for c in previous.CELLS}, 'Fit cases cell quota differs')
    require({r['state_key'] for r in body['fit_cases']} <= {r['state_key'] for r in body['fit_check']}, 'Fit cases outside reserved fit role')
    require(Counter(r['stratum'].split('/')[-1] for r in body['arena']) == dict(ordinary=8, capture=8, bonus=8), 'Arena mechanism quota differs')
    for role in ('train', 'fit_check', 'arena'):
        original = {r['state_key']: r for r in inv['arena_pool' if role == 'arena' else 'primary_pool']}
        for row in body[role]:
            require(row['state_key'] in original and all(common.same(row[k], original[row['state_key']][k]) for k in row), 'Selected source metadata changed')
            reasons = inv['components'][row['component']]['reasons']
            component = inv['components'][row['component']]
            require(row['family'] in component['families'] and previous.quotient(row['state']) in component['quotients'], 'Raw origin missing from reserved component')
            require(type(row['actor']) is int and row['actor'] == row['state']['controller'] and row['actor'] in (0, 1), 'Raw actor metadata differs')
            require(row['legal_mask'] == sum(1 << a for a, n in enumerate(row['state']['pits'][row['actor']]) if n), 'Raw legal mask differs')
            require(not any(r.startswith(('arena/' if role == 'arena' else 'optimizer/')) for r in reasons), 'Selected component has protected ownership')
            require(row['legal_mask'].bit_count() >= 2 and common.rawkey(row['state']) == row['state_key'], 'Invalid contested raw source')
    require(body['selected_components'] == {r['component']: inv['components'][r['component']] for r in rows}, 'Whole selected component reservation differs')
    return body


def prepare(path=DEFAULT):
    path = _path(path)
    if (path / 'certificate.json').exists() or (path / 'catalog.json').exists():
        return verify(path)
    if (path / 'identity.json').exists():
        return prepared(path)[2]
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        inv = inventory(); selection = select(inv)
        validate_selection(selection, inv)
        _atomic(path / 'inventory.json', inv); _atomic(path / 'selection.json', selection)
        identity = dict(schema=SCHEMA + '-identity', rules_digest=config.RULES_DIGEST,
                        plan_sha256=sha(PLAN), plan_acceptance_sha256=sha(PLAN_ACCEPTANCE),
                        inventory_sha256=sha(path / 'inventory.json'), selection_sha256=sha(path / 'selection.json'),
                        sources=common.pins(SOURCES), input_sources=inv['source_inputs'],
                        training_started=False, optimizer_steps=0, policy_fixed=True)
        _atomic(path / 'identity.json', identity)
    return identity


def prepared(path=DEFAULT):
    path = _path(path)
    identity = read(path / 'identity.json')
    require(identity['schema'] == SCHEMA + '-identity' and identity['rules_digest'] == config.RULES_DIGEST, 'Reservation identity differs')
    require(identity['plan_sha256'] == sha(PLAN) and identity['plan_acceptance_sha256'] == sha(PLAN_ACCEPTANCE), 'Accepted plan changed')
    common.current(identity['sources']); common.current(identity['input_sources'])
    require(identity['inventory_sha256'] == sha(path / 'inventory.json') and identity['selection_sha256'] == sha(path / 'selection.json'), 'Prepared metadata file changed')
    inv = read(path / 'inventory.json'); selection = read(path / 'selection.json')
    require(common.same(inv['source_inputs'], identity['input_sources']), 'Inventory input closure changed')
    validate_selection(selection, inv)
    require(selection == select(inv), 'Deterministic joint matching changed')
    return inv, selection, identity


class Writer:
    def __init__(self, path): self.path = Path(path); self.stream = None
    def __enter__(self):
        import msvcrt
        self.stream = (self.path / 'writer.lock').open('a+b')
        try:
            self.stream.seek(0)
            if not self.stream.read(1): self.stream.write(b'0'); self.stream.flush()
            self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except BaseException:
            self.stream.close(); self.stream = None; raise
        return self
    def __exit__(self, *_):
        import msvcrt
        if self.stream:
            self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1); self.stream.close()


def _pause(path, point, requested, marker):
    if requested != point: return
    require(path != DEFAULT.resolve() and marker is not None, 'Pause hooks are scratch fixtures only')
    marker = Path(marker).resolve()
    require(marker.is_relative_to(ROOT / 'build'), 'Pause marker outside fixture namespace')
    _atomic(marker, dict(schema=SCHEMA + '-pause', point=point, pid=os.getpid()))
    while True: time.sleep(.1)


def run(path=DEFAULT, pause_at=None, pause_marker=None):
    path = _path(path)
    # A catalog is the durable completed marker. Never acquire writer on the
    # completed route, including a missing or corrupt certificate.
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists():
        return verify(path)
    require(pause_at in (None, 'before_catalog_commit', 'after_catalog_commit'), 'Unknown pause boundary')
    if pause_at: require(path != DEFAULT.resolve(), 'Production pause hook forbidden')
    inv, selection, identity = prepared(path)
    with Writer(path):
        manifest = dict(schema=SCHEMA + '-manifest', identity_sha256=sha(path / 'identity.json'),
                        selection_sha256=sha(path / 'selection.json'), inventory_sha256=sha(path / 'inventory.json'),
                        counts=selection['counts'], source_inputs=identity['input_sources'],
                        status=selection['status'], admitted=selection['admitted'], readiness_blockers=selection['readiness_blockers'],
                        availability=selection['availability'],
                        component_roles={r['component']: role for role in ('train', 'fit_check', 'arena') for r in selection[role]},
                        target_bodies_read=0, model_forwards=0, searches=0, optimizer_steps=0, new_games=0,
                        models_and_policy_fixed=True, training_started=False)
        _atomic(path / 'manifest.json', manifest)
        certificate = dict(schema=SCHEMA + '-certificate', identity_sha256=sha(path / 'identity.json'),
                           manifest_sha256=sha(path / 'manifest.json'), files=common.pins([path / n for n in ('identity.json', 'selection.json', 'inventory.json', 'manifest.json')]),
                           passed=True, training_started=False)
        # Stage certification before catalog. The catalog itself remains the
        # commit: an interruption here resumes from immutable prepared bytes.
        _atomic(path / 'certificate.pending.json', certificate)
        _pause(path, 'before_catalog_commit', pause_at, pause_marker)
        _atomic(path / 'catalog.json', dict(schema=SCHEMA + '-catalog', complete=True,
                    manifest_sha256=sha(path / 'manifest.json'), certificate_sha256=sha(path / 'certificate.pending.json')))
        # Commit atomically binds an already fsynced certificate. Recovery may
        # validate its pending filename directly; no writer or repair required.
        _pause(path, 'after_catalog_commit', pause_at, pause_marker)
    return verify(path)


def verify(path=DEFAULT):
    path = _path(path)
    inv, selection, identity = prepared(path)
    catalog = read(path / 'catalog.json')
    require(catalog['schema'] == SCHEMA + '-catalog' and catalog['complete'] is True, 'Reservation catalog incomplete')
    certificate_path = path / 'certificate.pending.json'
    certificate = read(certificate_path)
    require(catalog['certificate_sha256'] == sha(certificate_path) and certificate['passed'] is True, 'Reservation certificate missing/corrupt')
    require(certificate['schema'] == SCHEMA + '-certificate' and certificate['files'] == common.pins([path / n for n in ('identity.json', 'selection.json', 'inventory.json', 'manifest.json')]), 'Reservation certificate file ownership differs')
    common.current(certificate['files'])
    manifest = read(path / 'manifest.json')
    require(catalog['manifest_sha256'] == sha(path / 'manifest.json') == certificate['manifest_sha256'], 'Reservation manifest differs')
    require(manifest['identity_sha256'] == sha(path / 'identity.json') == certificate['identity_sha256'], 'Reservation identity binding differs')
    require(manifest['selection_sha256'] == sha(path / 'selection.json') and manifest['inventory_sha256'] == sha(path / 'inventory.json'), 'Reservation metadata binding differs')
    require(manifest['counts'] == selection['counts'] and manifest['component_roles'] == {r['component']: role for role in ('train', 'fit_check', 'arena') for r in selection[role]}, 'Reserved role catalog differs')
    require(all(common.same(manifest[k], selection[k]) for k in ('status', 'admitted', 'readiness_blockers', 'availability')), 'Admission/hold evidence differs')
    require(all(manifest[k] == 0 for k in ('target_bodies_read', 'model_forwards', 'searches', 'optimizer_steps', 'new_games')) and manifest['training_started'] is False and manifest['models_and_policy_fixed'] is True, 'Unexpected training/inference activity')
    return manifest


def status(path=DEFAULT):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists():
        return dict(verify(path), publication_complete=True)
    if (path / 'identity.json').exists():
        _, selection, _ = prepared(path)
        return dict(status='prepared', admitted=False, selection_status=selection['status'], readiness_blockers=selection['readiness_blockers'], counts=selection['counts'], training_started=False, next_operation='run')
    return dict(status='not_started', training_started=False, next_operation='prepare')


def admit(path=DEFAULT):
    """Training/import admission is separate from successful HOLD publication."""
    manifest = verify(path)
    require(manifest['admitted'] is True and manifest['status'] == 'reserved', 'Value pilot data admission held: ' + '; '.join(manifest['readiness_blockers']))
    return manifest
