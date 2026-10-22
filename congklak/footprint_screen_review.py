"""Independent, recoverable full raw-state review; never starts a game engine.

Every actual query archive is a unit.  Completed records are reused while a
partially computed pure inspection can be repeated from the same fixed bytes.
The final union must equal the acquisition snapshot before reservations exist.
"""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import struct
import sys
import time
import zlib
from . import config, footprint_screen_common as s
from .valuepilot_reservations import Writer

DEFAULT = s.FOLDER/'audit/raw-review'
FIXTURES = s.FOLDER/'fixtures/raw-review'
SCHEMA = s.SCHEMA+'-independent-raw-review'
HOOKS = ('before_unit_commit', 'after_unit_commit', 'before_catalog', 'after_catalog')
SOURCES = ('congklak/footprint_screen_review.py', 'tools/audit-footprint-screen.py',
           'congklak/footprint_screen_worker.py', 'congklak/footprint_screen_native.py')
PACK = struct.Struct('<16sBBBBhh')
OWNED = struct.Struct('<15sQ16sB')
OP_NAMES = ('decode','normalize_in','normalize_out','legal','terminal_test','margin_eval',
 'move_in','move_out','alpha_node','alpha_tt_probe','alpha_tt_hit','alpha_tt_slot',
 'relay_node','relay_tt_probe','relay_tt_hit','relay_tt_slot','relay_eval',
 'relay_cache_probe','relay_cache_hit','model_eval','model_forward','feature_input',
 'feature_move_in','feature_move_out','sidecar_eval','sidecar_branch','sidecar_bypass',
 'puct_node','puct_select','puct_expand','played_input','played_output','alpha_pv_node',
 'relay_pv_node','model_done','sidecar_done','alpha_root_node')
need, read, sha, atomic = s.need, s.read, s.sha, s.atomic


def same(a, b):
    return config.canonical(a) == config.canonical(b)


def ref(path):
    return dict(path=s.relative(path), sha256=sha(path))


def scope(output, fixture):
    p = s.path_of(output)
    need(type(fixture) is bool, 'Review fixture flag must be boolean')
    need(p == DEFAULT if not fixture else p.parent == FIXTURES and p.name in (*HOOKS, 'reference'),
         'Unregistered full-raw-review output')
    return p


def snapshot():
    from . import footprint_screen_worker as worker
    snap = worker.completed_snapshot()
    need(snap['schema'] == s.SCHEMA+'-completed-snapshot' and len(snap['roots']) == 512,
         'Full finite acquisition snapshot is required before review')
    need([r['index'] for r in snap['roots']] == list(range(512)), 'Snapshot root order differs')
    return snap


def tasks(snap, fixture):
    rows = []
    seen = set()
    roots = snap['roots'][:1] if fixture else snap['roots']
    for root in roots:
        need(type(root['trace_complete']) is bool and root['trace_complete'] is True and root['hold'] is False,
             'A root has unknown acquisition work')
        for source in root['source_refs']:
            path = (s.ROOT/source['path']).resolve()
            need(path.is_relative_to(s.FOLDER) and path.parent.name == 'archives' and
                 snap['inputs'].get(source['path']) == source['sha256'],
                 'Snapshot archive is not an exact captured source')
            need(source['path'] not in seen, 'An actual archive is attributed twice')
            seen.add(source['path'])
            rows.append(dict(index=len(rows), root_index=root['index'], root_id=root['id'], archive_ref=source))
    need(len(rows) >= (2 if fixture else 1024), 'Incomplete actual-query archive inventory')
    return rows


def expected(output, fixture, snap=None):
    snap = snapshot() if snap is None else snap
    work = tasks(snap, fixture)
    return dict(schema=SCHEMA+'-identity', output=s.relative(scope(output, fixture)), fixture=fixture,
                plan_sha256=sha(s.PLAN), pool_sha256=snap['pool_sha256'], snapshot_sha256=config.digest(snap),
                source_inputs=snap['inputs'], review_sources=s.common.pins(SOURCES), tasks=work,
                task_sha256=config.digest(work), models_fixed=True, policy_fixed=True,
                new_queries=0, model_forwards=0, optimizer_steps=0)


def _identity(path, snap=None):
    identity = read(path/'identity.json')
    s.admit_plan()
    need(same(identity, expected(path, identity['fixture'], snap)), 'Full raw review identity/source scope changed')
    return identity


def raw_proof(answer, meta, kind):
    """Audit-owned decode/canonicalization, independent of native compact proof."""
    trace, cert = answer['trace'], answer['fast_certificate']
    count = trace['event_count']
    need(type(count) is int and 0 < count <= 16000000 and trace['event_bytes'] == 24 and
         trace['encoding'] == 'base64/zlib/packed24/little-endian' and trace['capacity_events'] == 16000000,
         'Raw review encoding/capacity differs')
    compressed = base64.b64decode(trace['data'], validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, count*24+1)
    need(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
         len(compressed) == trace['compressed_bytes'] and len(raw) == count*24,
         'Raw review has an unknown compression suffix')
    raw_hash = hashlib.sha256(raw).hexdigest()
    need(raw_hash == trace['event_sha256'] == cert['event_sha256'] and
         trace['overflow'] is False and trace['invalid'] is False and trace['truncated'] is False and
         trace['attempted_events'] == count == cert['event_count'] == cert['attempted_events'],
         'Raw review contains missing physical events')
    counts = Counter()
    first = {}
    low = high = upper = 0
    root = meta['state']
    root_cells = bytes(root['pits'][0]+[root['stores'][0]]+root['pits'][1]+[root['stores'][1]])
    root_actor = root['controller']
    played = []
    for index, (cells, actor, operation, engine_scope, reserved, action, depth) in enumerate(PACK.iter_unpack(raw)):
        need(sum(cells) == 98 and actor in (0, 1) and 1 <= operation <= len(OP_NAMES) and
             engine_scope in (1, 2, 3, 4) and reserved == 0 and action >= -1 and depth >= -1,
             'Invalid complete-state event in independent raw review')
        counts[OP_NAMES[operation-1]] += 1
        if operation == 31:
            played.append((cells, actor))
        pits = cells[:7]+cells[8:15]
        active = sum(pits)
        if active <= 24:
            low += 1
            continue
        high += 1
        upper += active >= 75
        quotient = min(pits+bytes([actor]), pits[7:]+pits[:7]+bytes([1-actor]))
        first.setdefault(quotient, OWNED.pack(quotient, index, cells, actor))
    need(played == [(root_cells, root_actor)], 'Independent raw input anchor/root differs')
    packed = base64.b64decode(answer['fast_owned_high'], validate=True)
    expected_owned = b''.join(value for _, value in sorted(first.items()))
    need(packed == expected_owned and len(first) == cert['owned_count'] and
         hashlib.sha256(packed).hexdigest() == cert['owned_sha256'] and high == cert['owned_occurrences'],
         'Independent complete raw-to-owned map differs')
    operations = {name: counts[name] for name in OP_NAMES}
    need(same(operations, trace['operation_counts']) and
         same([0]+[operations[name] for name in OP_NAMES]+[0]*(63-len(OP_NAMES)), cert['operation_counts']),
         'Independent physical operation partitions differ')
    need(all(operations[name] == 0 for name in ('model_eval','model_forward','model_done','sidecar_eval','sidecar_done')),
         'Forbidden model forward in acquisition')
    if kind == 'features':
        need(operations['feature_input'] == 1 and operations['feature_move_in'] == operations['feature_move_out'],
             'Incomplete independent feature event coverage')
    return dict(raw_sha256=raw_hash, events=count, high_owned_events=high, low_shared_events=low,
                upper_owned_events=upper, owned_count=len(first), owned_sha256=cert['owned_sha256'],
                quotient_sha256=hashlib.sha256(b''.join(sorted(first))).hexdigest(),
                quotients=[q.hex() for q in sorted(first)], operation_counts=operations)


def compute(task):
    from . import footprint_screen_worker as worker
    ap = s.ROOT/task['archive_ref']['path']
    need(sha(ap) == task['archive_ref']['sha256'], 'Review archive bytes changed')
    body = read(ap)
    meta = worker.pool()['roots'][task['root_index']]
    need(meta['id'] == task['root_id'] == body['root_id'] and body['family'] == meta['family'] and
         body['origin'] == meta['origin'] and body['scheduled']['root_index'] == meta['index'],
         'True query family/source attribution differs')
    kind = body['scheduled']['kind']
    need(kind in ('teacher', 'features'), 'Unknown finite acquisition query kind')
    cross = worker.full_raw_review(task['archive_ref'])
    independent = raw_proof(body['answer'], meta, kind)
    need(cross['raw_sha256'] == independent['raw_sha256'] and cross['events'] == independent['events'] and
         cross['canonical_high_quotients'] == independent['quotients'] and
         cross['owned40_sha256'] == independent['owned_sha256'],
         'Independent raw replay disagrees with strict source validator')
    sink = ap.parent.parent/'attempts'/(ap.stem+'.bin')
    need(Path(body['answer']['attempt_sink']).resolve() == sink.resolve(), 'Raw sink belongs to another query')
    raw_hash = hashlib.sha256()
    with sink.open('rb') as stream:
        header = stream.read(64)
        need(len(header) == 64, 'Incomplete durable raw sink header')
        magic, capacity, count, flags, reserved = struct.unpack('<8sQQQ32s', header)
        need(magic == b'OTRACE2\0' and capacity == 16000000 and count == independent['events'] and
             flags == 2 and reserved == bytes(32), 'Durable raw sink was not completely sealed')
        remaining = count*24
        while remaining:
            chunk = stream.read(min(1048576, remaining))
            need(chunk, 'Durable raw sink has a missing suffix')
            raw_hash.update(chunk)
            remaining -= len(chunk)
        need(not stream.read(1), 'Durable raw sink has an unbound suffix')
    need(raw_hash.hexdigest() == independent['raw_sha256'], 'Durable raw sink differs from lossless archive')
    return s.normalized(dict(schema=SCHEMA+'-unit', passed=True, index=task['index'], archive_ref=task['archive_ref'],
                root_id=meta['id'], root_index=meta['index'], family=meta['family'], origin=meta['origin'], kind=kind,
                math_complete=body['capture_proof']['math_complete'], ineligible_reason=body['capture_proof']['ineligible_reason'],
                pool_sha256=sha(s.FOLDER/'pool.json'), sink_ref=ref(sink), **independent,
                new_queries=0, model_forwards=0, optimizer_steps=0))


def _unit_valid(unit, task, identity):
    need(unit['schema'] == SCHEMA+'-unit' and unit['passed'] is True and type(unit['index']) is int and
         unit['index'] == task['index'] and unit['root_id'] == task['root_id'] and
         unit['root_index'] == task['root_index'] and same(unit['archive_ref'], task['archive_ref']) and
         unit['pool_sha256'] == identity['pool_sha256'], 'Raw review checkpoint belongs to another archive')
    need(all(type(unit[k]) is int and unit[k] == 0 for k in ('new_queries','model_forwards','optimizer_steps')),
         'Pure raw review work counters differ')
    need(identity['source_inputs'].get(unit['sink_ref']['path']) == unit['sink_ref']['sha256'],
         'Raw review checkpoint sink lacks its captured source binding')
    need(unit['quotients'] == sorted(set(unit['quotients'])) and len(unit['quotients']) == unit['owned_count'],
         'Raw review quotient inventory differs')
    keys = []
    for text in unit['quotients']:
        q = bytes.fromhex(text)
        need(q.hex() == text and len(q) == 15 and q[14] in (0,1) and 25 <= sum(q[:14]) <= 98 and
             q == min(q, q[7:14]+q[:7]+bytes([1-q[14]])), 'Invalid review high-domain quotient')
        keys.append(q)
    need(hashlib.sha256(b''.join(keys)).hexdigest() == unit['quotient_sha256'] and
         unit['events'] == unit['high_owned_events']+unit['low_shared_events'] == sum(unit['operation_counts'].values()),
         'Raw review checkpoint aggregates differ')


def _thin(unit):
    """Keep one proof digest, not a second whole campaign's quotient strings."""
    result = {key: value for key, value in unit.items() if key != 'quotients'}
    result['proof_sha256'] = config.digest(unit)
    return result


def _prefix(path, identity):
    units = []
    previous = None
    for index, cp in enumerate(sorted((path/'commits').glob('*.json'))):
        need(cp.name == f'{index:06d}.json' and index < len(identity['tasks']), 'Raw review commit cursor gap')
        commit = read(cp)
        up = path/'units'/f'{index:06d}.json'
        need(same(commit, dict(index=index, identity_sha256=sha(path/'identity.json'), unit=f'units/{index:06d}.json',
                              unit_sha256=sha(up), previous_sha256=previous)), 'Raw review commit binding differs')
        unit = read(up)
        _unit_valid(unit, identity['tasks'][index], identity)
        units.append(_thin(unit))
        previous = sha(cp)
    extra = sorted((path/'units').glob('*.json'))[len(units):]
    need(len(extra) <= 1 and (not extra or extra[0].name == f'{len(units):06d}.json'),
         'Unbound raw review suffix differs')
    if extra:
        _unit_valid(read(extra[0]), identity['tasks'][len(units)], identity)
    return units, previous


def _append(path, identity, units, unit):
    index = len(units)
    _unit_valid(unit, identity['tasks'][index], identity)
    up = path/'units'/f'{index:06d}.json'
    atomic(up, unit)
    atomic(path/'commits'/f'{index:06d}.json', dict(index=index, identity_sha256=sha(path/'identity.json'),
           unit=f'units/{index:06d}.json', unit_sha256=sha(up),
           previous_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None))
    units.append(_thin(unit))
    print(config.canonical(dict(event='independent_raw_review_commit', index=index,
                                events=unit['events'], owned_count=unit['owned_count'])), flush=True)


def _report(identity, units, snap, path=None):
    need(len(units) == len(identity['tasks']), 'Full raw review is incomplete')
    by_root = defaultdict(list)
    for unit in units:
        by_root[unit['root_index']].append(unit)
    checked_roots = snap['roots'][:1] if identity['fixture'] else snap['roots']
    for root in checked_roots:
        keys = set()
        for unit in by_root[root['index']]:
            if 'quotients' in unit:
                full = unit
            else:
                need(path is not None, 'Saved raw proof path is required')
                full = read(path/'units'/f'{unit["index"]:06d}.json')
                need(config.digest(full) == unit['proof_sha256'], 'Saved raw proof changed before final union')
            keys.update(full['quotients'])
        need(sorted(keys) == root['quotients'], 'Independent full root footprint union differs')
    return dict(schema=SCHEMA+'-report', passed=True, independent_of_implementation=True,
                fixture=identity['fixture'], pool_sha256=identity['pool_sha256'], snapshot_sha256=identity['snapshot_sha256'],
                plan_sha256=identity['plan_sha256'], source_inputs=identity['source_inputs'], review_sources=identity['review_sources'],
                acquired_roots=len(checked_roots), production_query_count=2*len(checked_roots), query_count=len(units),
                raw_events=sum(u['events'] for u in units), high_owned_events=sum(u['high_owned_events'] for u in units),
                low_shared_events=sum(u['low_shared_events'] for u in units), upper_owned_events=sum(u['upper_owned_events'] for u in units),
                all_actual_query_sources_checked=True, capped_footprints_retained=True, full_high_ownership_min=25,
                historically_unseen=False, inherited_unknown_background=True, models_fixed=True, policy_fixed=True,
                training_admitted=False, new_queries=0, model_forwards=0, optimizer_steps=0,
                units_sha256=config.digest([dict(index=u['index'], archive_ref=u['archive_ref'],
                                                proof_sha256=u.get('proof_sha256', config.digest(u))) for u in units]))


def _catalog(path, identity, report):
    return dict(schema=SCHEMA+'-catalog', identity_ref=ref(path/'identity.json'), report_ref=ref(path/'report.json'),
                commits=s.common.pins(sorted((path/'commits').glob('*.json'))),
                units=s.common.pins(sorted((path/'units').glob('*.json'))),
                query_count=report['query_count'], passed=True)


def _finish(path, identity, units, snap, pause=lambda point: None):
    report = _report(identity, units, snap, path)
    atomic(path/'report.json', report)
    pause('before_catalog')
    atomic(path/'catalog.json', _catalog(path, identity, report))
    pause('after_catalog')
    return dict(report=report, report_ref=ref(path/'report.json'), catalog_ref=ref(path/'catalog.json'), publication_gap=False)


def has_completion(output=DEFAULT):
    path = s.path_of(output)
    if not (path/'identity.json').exists():
        return False
    snap = snapshot()
    identity = _identity(path, snap)
    units, _ = _prefix(path, identity)
    return len(units) == len(identity['tasks']) and (path/'report.json').exists()


def completed(output=DEFAULT):
    path = s.path_of(output)
    snap = snapshot()
    identity = _identity(path, snap)
    units, _ = _prefix(path, identity)
    report = _report(identity, units, snap, path)
    need(same(read(path/'report.json'), report), 'Full raw review report differs')
    catalog = _catalog(path, identity, report)
    gap = not (path/'catalog.json').exists()
    if not gap:
        need(same(read(path/'catalog.json'), catalog), 'Full raw review catalog differs')
    return dict(report=report, report_ref=ref(path/'report.json'),
                catalog_ref=None if gap else ref(path/'catalog.json'), publication_gap=gap)


def publish_completed(output=DEFAULT):
    done = completed(output)
    if not done['publication_gap']:
        return done
    path = s.path_of(output)
    snap = snapshot()
    with Writer(path):
        identity = _identity(path, snap)
        units, _ = _prefix(path, identity)
        _finish(path, identity, units, snap)
    return completed(path)


def preflight(output=DEFAULT, fixture=False):
    s.admit_plan()
    path = scope(output, fixture)
    snap = snapshot()
    if (path/'identity.json').exists():
        identity = _identity(path, snap)
        need(identity['fixture'] is fixture, 'Review fixture treatment changed')
        _prefix(path, identity)
    else:
        need(not path.exists() or not any(path.rglob('*')), 'Unbound review evidence prevents overwrite')
    return dict(state='ready', fixture=fixture, pure_review=True, new_queries=0)


def _pause(path, point, wanted, marker, lease):
    if point != wanted:
        return
    from . import footprint_screen_resources as resources
    atomic(s.path_of(marker), dict(schema=SCHEMA+'-pause', point=point, identity_sha256=sha(path/'identity.json'),
           process=resources.process_identity(), argv=sys.argv, lease=str(Path(lease).resolve()),
           source_sha256=sha(__file__), new_queries=0))
    while True:
        time.sleep(.05)


def run(output=DEFAULT, fixture=False, stop_after=None, pause_at=None, pause_marker=None, lease=None):
    if has_completion(output):
        return publish_completed(output)
    from . import footprint_screen_resources as resources
    need(lease is not None, 'Contained full raw review lease required')
    resources.assert_worker_lease(lease)
    s.admit_plan()
    path = scope(output, fixture)
    need(pause_at is None or fixture and pause_at in HOOKS and pause_marker is not None, 'Unadmitted raw review pause')
    need(stop_after is None or type(stop_after) is int and stop_after > 0, 'Invalid raw review checkpoint cursor')
    snap = snapshot()
    path.mkdir(parents=True, exist_ok=True)
    pause = lambda point: _pause(path, point, pause_at, pause_marker, lease)
    with Writer(path):
        identity = expected(path, fixture, snap)
        atomic(path/'identity.json', identity)
        units, _ = _prefix(path, identity)
        for task in identity['tasks'][len(units):]:
            if resources.remaining_ns(lease) < 30000000000 or stop_after is not None and len(units) >= stop_after:
                return dict(checkpoint=True, completed_units=len(units), required_units=len(identity['tasks']), new_queries=0)
            resources.assert_worker_lease(lease)
            up = path/'units'/f'{task["index"]:06d}.json'
            unit = read(up) if up.exists() else compute(task)
            atomic(up, unit)
            pause('before_unit_commit')
            _append(path, identity, units, unit)
            pause('after_unit_commit')
        return _finish(path, identity, units, snap, pause)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run','completed','publish','status'))
    parser.add_argument('--output', default=str(DEFAULT))
    parser.add_argument('--lease')
    parser.add_argument('--engineering-fixture', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--pause-at')
    parser.add_argument('--pause-marker')
    args = parser.parse_args(argv)
    if args.command == 'run':
        if has_completion(args.output):
            answer = publish_completed(args.output)
        else:
            from . import footprint_screen_resources as resources
            need(args.lease is not None, 'Contained full raw review lease missing')
            resources.bind_worker(args.lease)
            answer = run(args.output, args.engineering_fixture, args.stop_after, args.pause_at, args.pause_marker, args.lease)
            resources.worker_complete(args.lease, dict(state='complete' if 'report' in answer else 'checkpoint'))
    elif args.command == 'completed':
        answer = completed(args.output)
    elif args.command == 'publish':
        answer = publish_completed(args.output)
    else:
        path = s.path_of(args.output)
        if not (path/'identity.json').exists():
            answer = dict(state='not_started')
        else:
            identity = _identity(path)
            units, _ = _prefix(path, identity)
            answer = dict(state='complete' if len(units) == len(identity['tasks']) else 'checkpoint',
                          completed_units=len(units), required_units=len(identity['tasks']))
    print(config.canonical(answer))
    return answer


if __name__ == '__main__':
    main()
