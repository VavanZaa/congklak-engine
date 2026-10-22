"""Full-prefix-anchored producer with literal orphan replay, never training data.

These are legal custom positions, not positions claimed reachable from a common
opening. All accepted origin observations are protected; rejected construction
attempts remain a reproducible metadata-filter journal and confer no role.
"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import sys
from . import config, neurodiag_common as common, neurodiag_data as old_data
from . import valuepilot_reservations as old

ROOT = common.ROOT
DEFAULT = ROOT / 'data/ordinary-development-v3'
SCHEMA = 'congklak-ordinary-development-v3'
SOURCES = ('congklak/ordinary_development_v3.py', 'tools/generate-ordinary-development-v3.py')
PARENT = ROOT / 'build/valuepilot-interface-v1/acceptance.json'
PROTOCOL = dict(schema=SCHEMA + '-protocol', candidate_quota=32, per_cell=8,
                cells=['25-49/ordinary/actor0', '25-49/ordinary/actor1',
                       '50-74/ordinary/actor0', '50-74/ordinary/actor1'],
                max_attempts=16384, chunk_attempts=64,
                counter_salt='ordinary-development/v1/20261016/',
                construction='SHA256 counter; exactly 2 positive actor pits and 7 positive opponent pits; balanced stores; exact raw actor',
                origin='synthetic legal custom position; no opening reachability or played trajectory claim',
                mechanism='pilot._mechanism precedence: pass, capture, bonus, ordinary; all legal action probes',
                eligible_observation='active seeds > 4; terminal probe states retained but do not merge components',
                closure='accepted root plus every normalized legal-action child used for classification',
                purity='all accepted eligible quotients disjoint full old graph and every other accepted origin',
                rejected_attempts='retained reproducible metadata filter journal only; not admitted development origins',
                reservation='ALL 32 accepted origins development only, optimizer blocked, including 24 unused arena starts',
                sampling_limit='synthetic starts with exactly two legal choices; not representative opening-play or natural-position coverage',
                counter_semantics='logical committed construction attempts/probes; an interrupted uncommitted chunk may repeat at most 64 attempts',
                progress_anchor='mutable fsynced identity-bound cursor binds ordered filename/SHA256 list and digest of ALL committed chunks; missing or changed committed chunks refuse; at most one fully valid orphan chunk may precede cursor update',
                orphan_verification='before cursor promotion, replay ONLY the one orphan chunk with native rules and compare every complete literal record; at most 64 counter attempts and 128 classification transitions, no committed prefix replay',
                origin_identity='v1 counter salt, recipe and family names preserved: replay of the same origins, never a claim of newly distinct data',
                targets=0, searches=0, model_forwards=0, optimizer_steps=0, new_games=0)
require, read, sha = common.require, common.read, common.sha
atomic = old._atomic


def _path(path):
    path = Path(path).resolve()
    require(path == DEFAULT.resolve() or path.is_relative_to(ROOT / 'build'), 'Development output namespace differs')
    return path


def candidate(attempt):
    """Pure counter construction; identical across process and Python RNG state."""
    require(type(attempt) is int and 0 <= attempt < PROTOCOL['max_attempts'], 'Attempt index differs')
    cell = PROTOCOL['cells'][attempt % 4]
    def block(part):
        return hashlib.sha256((PROTOCOL['counter_salt'] + str(attempt) + '/' + str(part)).encode('ascii')).digest()
    active = (25 if cell.startswith('25') else 50) + int.from_bytes(block('active')[:4], 'little') % 25
    actor = int(cell[-1]); counts = [0] * 14
    first = int.from_bytes(block('first')[:4], 'little') % 7
    second = (first + 1 + int.from_bytes(block('second')[:4], 'little') % 6) % 7
    own_total = 2 + int.from_bytes(block('own_total')[:4], 'little') % (active - 8)
    first_count = 1 + int.from_bytes(block('first_count')[:4], 'little') % (own_total - 1)
    counts[actor * 7 + first] = first_count
    counts[actor * 7 + second] = own_total - first_count
    for index in range(7): counts[(1 - actor) * 7 + index] = 1
    for index in range(active - own_total - 7):
        counts[(1 - actor) * 7 + int.from_bytes(block(index)[:4], 'little') % 7] += 1
    stored = 98 - active
    state = dict(pits=[counts[:7], counts[7:]], stores=[stored // 2, stored - stored // 2], controller=int(cell[-1]))
    return cell, config.position(state)


def _facts(attempt, native):
    from .native import flatten
    cell, state = candidate(attempt)
    normal = native.normalize(flatten(state), actor=state['controller'], total=98)
    require(normal.get('ok') and normal['position'] == state and not normal['terminal'], 'Constructed position not stable')
    probes = []
    for action in normal['legal_actions']:
        result = native.apply(flatten(state), actor=state['controller'], action=action, total=98, trace=False)
        require(result.get('ok'), 'Classification transition invalid')
        child = result['position']
        normalized = native.normalize(flatten(child), actor=child['controller'], total=98)
        require(normalized.get('ok'), 'Classification child normalization invalid')
        probes.append(dict(action=action, reason=result['reason'], passes=result['passes'],
                           state=normalized['position'], terminal=normalized['terminal']))
    mechanism = ('pass' if any(p['passes'] for p in probes) else
                 'capture' if any(p['reason'] == 'own_capture' for p in probes) else
                 'bonus' if any(p['reason'] == 'own_store' for p in probes) else 'ordinary')
    return dict(attempt=attempt, cell=cell, state=state, state_key=common.rawkey(state),
                legal_mask=sum(1 << a for a in normal['legal_actions']), mechanism=mechanism,
                probes=probes, accepted=False)


def observations(record):
    return [record['state']] + [probe['state'] for probe in record['probes']]


def eligible_quotients(record):
    return {old_data.quotient(p) for p in observations(record) if sum(map(sum, p['pits'])) > 4}


def prepare(path=DEFAULT):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return verify(path)
    if (path / 'identity.json').exists():
        identity, inv = _identity(path); _load_chunks(path, inv)
        return identity
    require('torch' not in sys.modules, 'Development preparation forbids Torch')
    path.mkdir(parents=True, exist_ok=True)
    with old.Writer(path):
        parent = read(PARENT)
        require(parent.get('passed') is True and parent.get('data_status') == 'hold', 'Accepted parent HOLD missing')
        # Rebuild every current ownership source, including saved literal games.
        inv = old.inventory()
        accepted = old.verify()
        require(accepted['admitted'] is False, 'Prior reservation must remain the accepted HOLD')
        old_inv = read(old.DEFAULT / 'inventory.json')
        require(inv == old_inv, 'Current full graph differs from accepted parent inventory')
        atomic(path / 'protocol.json', PROTOCOL)
        atomic(path / 'parent-inventory.json', inv)
        identity = dict(schema=SCHEMA + '-identity', protocol_sha256=sha(path / 'protocol.json'),
                        parent_inventory_sha256=sha(path / 'parent-inventory.json'),
                        parent_acceptance_sha256=sha(PARENT), rules_digest=config.RULES_DIGEST,
                        sources=common.pins(SOURCES), input_sources=inv['source_inputs'],
                        accepted_parent_inventory_sha256=sha(old.DEFAULT / 'inventory.json'),
                        training_started=False, models_and_policy_fixed=True)
        # The cursor precedes the identity commit; a prepared identity therefore
        # never authorizes a missing progress anchor, even at zero chunks.
        expected_identity_sha = hashlib.sha256((config.canonical(identity) + '\n').encode('utf-8')).hexdigest()
        _cursor_write(path, [], expected_identity_sha)
        atomic(path / 'identity.json', identity)
    return identity


def _identity(path):
    identity = read(path / 'identity.json')
    require(identity['schema'] == SCHEMA + '-identity' and identity['rules_digest'] == config.RULES_DIGEST, 'Development identity differs')
    require(read(path / 'protocol.json') == PROTOCOL and sha(path / 'protocol.json') == identity['protocol_sha256'], 'Frozen development protocol differs')
    require(sha(PARENT) == identity['parent_acceptance_sha256'], 'Accepted parent changed')
    require(sha(path / 'parent-inventory.json') == identity['parent_inventory_sha256'] == identity['accepted_parent_inventory_sha256'], 'Parent inventory binding differs')
    common.current(identity['sources']); common.current(identity['input_sources'])
    require(sha(old.DEFAULT / 'inventory.json') == identity['accepted_parent_inventory_sha256'], 'Accepted full graph changed')
    require(identity['training_started'] is False and identity['models_and_policy_fixed'] is True, 'Unexpected development activity')
    inv = read(path / 'parent-inventory.json')
    require(inv['source_inputs'] == identity['input_sources'], 'Parent graph source closure differs')
    return identity, inv


def _cursor_write(path, files, identity_sha256=None):
    committed = [dict(name=p.name, sha256=sha(p)) for p in files]
    body = dict(schema=SCHEMA + '-cursor', completed_chunks=len(files),
                last_chunk_sha256=sha(files[-1]) if files else None,
                committed_chunks=committed, prefix_digest=config.digest(committed),
                identity_sha256=identity_sha256 or sha(path / 'identity.json'))
    target = path / 'cursor.json'
    if target.exists() and read(target) == body: return
    temporary = path / 'cursor.json.pending'
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(config.canonical(body) + '\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)


def _load_chunks(path, inv, complete=False):
    old_quotients = {q for component in inv['components'].values() for q in component['quotients']}
    accepted = []; seen = set(); records = []; cell_counts = Counter(); probes = 0
    files = sorted((path / 'chunks').glob('*.json')) if (path / 'chunks').exists() else []
    cursor = read(path / 'cursor.json'); count = cursor['completed_chunks']
    require(cursor['schema'] == SCHEMA + '-cursor' and type(count) is int and count >= 0, 'Development cursor identity differs')
    require(cursor['identity_sha256'] == sha(path / 'identity.json'), 'Development cursor preparation binding differs')
    require(count <= len(files) <= count + (0 if complete else 1), 'Missing committed development chunk or invalid cursor gap')
    prefix = [dict(name=p.name, sha256=sha(p)) for p in files[:count]]
    require(cursor['committed_chunks'] == prefix and cursor['prefix_digest'] == config.digest(prefix), 'Committed development prefix digest differs')
    require(cursor['last_chunk_sha256'] == (sha(files[count - 1]) if count else None), 'Committed development chunk digest differs')
    orphan_native = None
    for index, filename in enumerate(files):
        require(filename.name == f'{index:06d}.json', 'Development chunks have a gap')
        chunk = read(filename)
        require(chunk['schema'] == SCHEMA + '-chunk' and chunk['index'] == index, 'Development chunk identity differs')
        require(chunk['start_attempt'] == len(records), 'Development chunk prefix differs')
        require(0 < len(chunk['attempts']) <= PROTOCOL['chunk_attempts'], 'Development chunk size differs')
        for record in chunk['attempts']:
            attempt = len(records); cell, state = candidate(attempt)
            require(record['attempt'] == attempt and record['cell'] == cell, 'Counter journal ordering differs')
            if record.get('reason') == 'cell_quota_filled':
                require(cell_counts[cell] == PROTOCOL['per_cell'] and set(record) == {'attempt', 'cell', 'reason', 'accepted'}, 'Invalid quota skip')
                require(record['accepted'] is False, 'Quota skip admitted')
            else:
                require(record['state'] == state and record['state_key'] == common.rawkey(state), 'Constructed raw position changed')
                actions = [a for a, n in enumerate(state['pits'][state['controller']]) if n]
                require(record['legal_mask'] == sum(1 << a for a in actions) and len(record['probes']) == len(actions) == 2, 'Classification action coverage differs')
                require([p['action'] for p in record['probes']] == actions, 'Classification legal action order differs')
                for p in record['probes']: config.position(p['state'])
                mechanism = ('pass' if any(p['passes'] for p in record['probes']) else 'capture' if any(p['reason'] == 'own_capture' for p in record['probes']) else 'bonus' if any(p['reason'] == 'own_store' for p in record['probes']) else 'ordinary')
                require(record['mechanism'] == mechanism, 'Classification mechanism differs')
                quotients = eligible_quotients(record)
                reason = 'mechanism_' + mechanism if mechanism != 'ordinary' else 'old_component_overlap' if quotients & old_quotients else 'generated_origin_overlap' if quotients & seen else 'accepted'
                require(record['reason'] == reason and record['accepted'] is (reason == 'accepted'), 'Metadata acceptance decision differs')
                if index >= count:
                    # An orphan was fsynced before its cursor commit. Its bytes
                    # are not yet anchored and must never become trusted merely
                    # because the JSON and metadata are internally consistent.
                    if orphan_native is None:
                        from .native import Native
                        orphan_native = Native()
                    replay = _facts(attempt, orphan_native)
                    replay_q = eligible_quotients(replay)
                    replay_reason = 'mechanism_' + replay['mechanism'] if replay['mechanism'] != 'ordinary' else 'old_component_overlap' if replay_q & old_quotients else 'generated_origin_overlap' if replay_q & seen else 'accepted'
                    replay['reason'] = replay_reason
                    if replay_reason == 'accepted':
                        replay.update(accepted=True, family='synthetic-ordinary-development-v1/' + f'{attempt:05d}')
                    require(replay == record, 'Uncommitted orphan classification replay differs')
                if record['accepted']:
                    require(cell_counts[cell] < PROTOCOL['per_cell'], 'Development cell quota exceeded')
                    family = 'synthetic-ordinary-development-v1/' + f'{attempt:05d}'
                    require(record['family'] == family, 'Synthetic origin provenance differs')
                    accepted.append(record); seen.update(quotients); cell_counts[cell] += 1
                probes += len(record['probes'])
            records.append(record)
        require(chunk['end_attempt'] == len(records), 'Chunk ending counter differs')
    require(len(records) <= PROTOCOL['max_attempts'], 'Generation attempt budget exceeded')
    return accepted, seen, cell_counts, records, probes, old_quotients


def _pause(path, point, pause_at, marker):
    if pause_at != point: return
    require(path != DEFAULT.resolve() and marker is not None, 'Generation pause hooks are fixtures only')
    old._pause(path, point, point, marker)


def run(path=DEFAULT, pause_at=None, pause_marker=None):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return verify(path)
    require(pause_at in (None, 'before_chunk_commit', 'before_cursor_commit', 'after_chunk_commit', 'before_catalog_commit', 'after_catalog_commit'), 'Unknown development pause boundary')
    if pause_at: require(path != DEFAULT.resolve(), 'Production pause hook forbidden')
    identity, inv = _identity(path)
    require('torch' not in sys.modules, 'Development generation forbids Torch')
    with old.Writer(path):
        accepted, seen, counts, records, probes, old_quotients = _load_chunks(path, inv)
        _cursor_write(path, sorted((path / 'chunks').glob('*.json')) if (path / 'chunks').exists() else [])
        native = None
        while len(accepted) < PROTOCOL['candidate_quota'] and len(records) < PROTOCOL['max_attempts']:
            if native is None:
                from .native import Native
                native = Native()
            start = len(records); chunk_records = []
            for attempt in range(start, min(start + PROTOCOL['chunk_attempts'], PROTOCOL['max_attempts'])):
                cell, _ = candidate(attempt)
                if counts[cell] == PROTOCOL['per_cell']:
                    record = dict(attempt=attempt, cell=cell, reason='cell_quota_filled', accepted=False)
                else:
                    record = _facts(attempt, native); probes += len(record['probes'])
                    q = eligible_quotients(record)
                    reason = 'mechanism_' + record['mechanism'] if record['mechanism'] != 'ordinary' else 'old_component_overlap' if q & old_quotients else 'generated_origin_overlap' if q & seen else 'accepted'
                    record['reason'] = reason
                    if reason == 'accepted':
                        record.update(accepted=True, family='synthetic-ordinary-development-v1/' + f'{attempt:05d}')
                        accepted.append(record); counts[cell] += 1; seen.update(q)
                chunk_records.append(record); records.append(record)
                if len(accepted) == PROTOCOL['candidate_quota']: break
            chunk_index = start // PROTOCOL['chunk_attempts']
            chunk = dict(schema=SCHEMA + '-chunk', index=chunk_index, start_attempt=start, end_attempt=len(records), attempts=chunk_records)
            _pause(path, 'before_chunk_commit', pause_at, pause_marker)
            atomic(path / 'chunks' / f'{chunk_index:06d}.json', chunk)
            _pause(path, 'before_cursor_commit', pause_at, pause_marker)
            _cursor_write(path, sorted((path / 'chunks').glob('*.json')))
            _pause(path, 'after_chunk_commit', pause_at, pause_marker)
        admitted = len(accepted) == PROTOCOL['candidate_quota']
        candidates = dict(schema=SCHEMA + '-candidates', accepted=accepted, rejected_attempts_are_reserved=False,
                          all_accepted_are_development_only=True, opening_reachability_claimed=False)
        atomic(path / 'candidates.json', candidates)
        counts_body = dict(accepted_components=len(accepted), accepted_roots=len(accepted), attempted_counters=len(records),
                           generated_roots=sum(r['reason'] != 'cell_quota_filled' for r in records),
                           classification_transitions=probes, eligible_origin_observations=sum(sum(map(sum, p['pits'])) > 4 for r in accepted for p in observations(r)),
                           retained_origin_observations=sum(len(observations(r)) for r in accepted),
                           accepted_legal_choice_histogram=dict(sorted(Counter(str(r['legal_mask'].bit_count()) for r in accepted).items())))
        manifest = dict(schema=SCHEMA + '-manifest', status='complete' if admitted else 'hold', admitted=admitted,
                        identity_sha256=sha(path / 'identity.json'), candidates_sha256=sha(path / 'candidates.json'),
                        counts=counts_body, cells=dict(sorted(counts.items())), rejection_reasons=dict(sorted(Counter(r['reason'] for r in records).items())),
                        readiness_blockers=[] if admitted else ['Fixed counter budget did not fill all four ordinary development cells'],
                        target_bodies_read=0, searches=0, model_forwards=0, optimizer_steps=0, new_games=0,
                        new_teacher_labels=0, training_started=False, models_and_policy_fixed=True)
        atomic(path / 'manifest.json', manifest)
        files = [path / n for n in ('protocol.json', 'identity.json', 'parent-inventory.json', 'cursor.json', 'candidates.json', 'manifest.json')] + sorted((path / 'chunks').glob('*.json'))
        certificate = dict(schema=SCHEMA + '-certificate', files=common.pins(files), passed=True, admitted=admitted)
        atomic(path / 'certificate.pending.json', certificate)
        _pause(path, 'before_catalog_commit', pause_at, pause_marker)
        atomic(path / 'catalog.json', dict(schema=SCHEMA + '-catalog', complete=True, certificate_sha256=sha(path / 'certificate.pending.json')))
        _pause(path, 'after_catalog_commit', pause_at, pause_marker)
    return verify(path)


def verify(path=DEFAULT):
    path = _path(path); _, inv = _identity(path)
    catalog = read(path / 'catalog.json'); certificate = read(path / 'certificate.pending.json')
    require(catalog == dict(schema=SCHEMA + '-catalog', complete=True, certificate_sha256=sha(path / 'certificate.pending.json')), 'Development catalog differs')
    require(certificate['schema'] == SCHEMA + '-certificate' and certificate['passed'] is True, 'Development certificate differs')
    files = [path / n for n in ('protocol.json', 'identity.json', 'parent-inventory.json', 'cursor.json', 'candidates.json', 'manifest.json')] + sorted((path / 'chunks').glob('*.json'))
    require(certificate['files'] == common.pins(files), 'Development file closure differs')
    common.current(certificate['files'])
    accepted, _, counts, records, probes, _ = _load_chunks(path, inv, complete=True)
    require(read(path / 'candidates.json') == dict(schema=SCHEMA + '-candidates', accepted=accepted,
                rejected_attempts_are_reserved=False, all_accepted_are_development_only=True, opening_reachability_claimed=False), 'Development pool differs')
    admitted = len(accepted) == 32
    if admitted: require(counts == {c: 8 for c in PROTOCOL['cells']}, 'Development quota differs')
    else: require(len(records) == PROTOCOL['max_attempts'], 'Premature generation HOLD')
    manifest = read(path / 'manifest.json')
    require(manifest['admitted'] is admitted and certificate['admitted'] is admitted and manifest['status'] == ('complete' if admitted else 'hold'), 'Development admission differs')
    require(manifest['identity_sha256'] == sha(path / 'identity.json') and manifest['candidates_sha256'] == sha(path / 'candidates.json'), 'Development manifest identity differs')
    expected = dict(accepted_components=len(accepted), accepted_roots=len(accepted), attempted_counters=len(records), generated_roots=sum(r['reason'] != 'cell_quota_filled' for r in records), classification_transitions=probes, eligible_origin_observations=sum(sum(map(sum, p['pits'])) > 4 for r in accepted for p in observations(r)), retained_origin_observations=sum(len(observations(r)) for r in accepted), accepted_legal_choice_histogram=dict(sorted(Counter(str(r['legal_mask'].bit_count()) for r in accepted).items())))
    require(manifest['counts'] == expected and manifest['cells'] == dict(sorted(counts.items())), 'Development counters differ')
    require(manifest['rejection_reasons'] == dict(sorted(Counter(r['reason'] for r in records).items())), 'Attempt accounting differs')
    require(all(manifest[k] == 0 for k in ('target_bodies_read', 'searches', 'model_forwards', 'optimizer_steps', 'new_games', 'new_teacher_labels')), 'Unexpected generation activity')
    require(manifest['training_started'] is False and manifest['models_and_policy_fixed'] is True, 'Development policy/training differs')
    return manifest


def admit(path=DEFAULT):
    manifest = verify(path)
    require(manifest['admitted'] is True, 'Ordinary development generation held')
    return manifest


def status(path=DEFAULT):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return verify(path)
    if (path / 'identity.json').exists():
        _, inv = _identity(path); accepted, _, counts, records, _, _ = _load_chunks(path, inv)
        return dict(schema=SCHEMA + '-status', status='prepared' if not records else 'generating',
                    accepted_components=len(accepted), attempted_counters=len(records), cells=dict(counts), training_started=False)
    return dict(schema=SCHEMA + '-status', status='not_started', training_started=False)
