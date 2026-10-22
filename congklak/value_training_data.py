"""Strict, role-separated admission of the already reserved H8 training roots.

Preparation performs no search, model forward, Torch import or optimization.
Completed admission checks metadata and hashes before acquiring any writer.
"""
from __future__ import annotations
from collections import Counter
from contextlib import closing
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import struct
import sys
import time
from . import config, neurodiag_common as common, neurodiag_data as previous
from . import valuepilot_reservations_v2 as reservations, value_sidecar_v2 as side

ROOT = common.ROOT
DEFAULT = ROOT / 'data/value-training-v1/cache'
SCHEMA = 'value-training-data-v1'
SOURCES = ('congklak/value_training_data.py', 'tools/prepare-value-training.py')
ADMISSION = ROOT / 'build/ordinary-development-v1/acceptance.json'
ROLES = {'train': ('train.json', 512), 'fit_check': ('fit-check.json', 128)}
HOOKS = ('before_data_catalog', 'after_data_catalog')
need, read, sha = common.require, common.read, common.sha
atomic, Writer = reservations.atomic, reservations.Writer


def _path(path):
    path = Path(path).resolve()
    need(path == DEFAULT.resolve() or path.is_relative_to(ROOT / 'build'), 'Training cache namespace differs')
    return path


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def identity(preparation_runtime=None):
    """Metadata only: freeze existing membership, target sources and encoder."""
    admitted = reservations.admit()
    selection = read(reservations.DEFAULT / 'selection.json')
    need(ADMISSION.is_file() and read(ADMISSION).get('passed') is True, 'Ordinary development acceptance missing')
    source_files = set(SOURCES) | {'congklak/neurodiag_data.py', 'congklak/neurodiag_common.py',
        'congklak/features_relay.py', 'congklak/features_v3.py', 'congklak/config.py',
        'congklak/value_sidecar_v2.py', 'build/congklak_value_sidecar.dll', 'build/value-sidecar-build.json',
        'data/quality-train-v1/model.json', str(ADMISSION.relative_to(ROOT)).replace('\\', '/')}
    for role in ROLES:
        for row in selection[role]:
            need(row['cohort'] == 'base_H8' and len(row['target_refs']) == 1, 'Mixed target cohort')
            ref = row['target_refs'][0]
            need(ref['journal'] == 'deeper-9000-v2' and ref['table'] == 'labels' and ref['depth'] == 8,
                 'Target is not original H8')
            source_files.add('data/' + ref['journal'] + '/work.sqlite3')
            source_files.add(row['feature_ref']['path'])
    return dict(schema=SCHEMA + '-identity', rules_digest=config.RULES_DIGEST,
        ordinary_admission={'path': str(ADMISSION), 'sha256': sha(ADMISSION)},
        reservations=side.reservation_bindings(reservations.DEFAULT / 'manifest.json'),
        interface=side.bindings(reservations.DEFAULT / 'manifest.json'),
        conformance=side.conformance_admission(reservations.DEFAULT / 'manifest.json'),
        sources=common.pins(sorted(source_files)),
        target=dict(kind='original-complete-H8/full-legal-root-Q/player0-margin', requested_depth=8,
                    completed_depth=8, normalized='(y-D)/signed_active', endpoints='retain',
                    clipping=False, weighting='unit', training_objective='Huber delta=0.1'),
        features=dict(count=127, dtype='IEEE754-binary32', rounding='nearest-even',
                      encoder=side.features_relay.ENCODER127, existing_cache_only=True),
        preparation_runtime=preparation_runtime or dict(python=sys.version, implementation=platform.python_implementation(),
                                 machine=platform.machine(), byteorder=sys.byteorder),
        role_membership={role: [config.digest(row) for row in selection[role]] for role in ROLES},
        counts={role: count for role, (_, count) in ROLES.items()},
        arena_targets_read=0, model_forwards=0, optimizer_steps=0, policy_fixed=True)


def _teacher(meta, db, teacher):
    ref = meta['target_refs'][0]
    need(ref == dict(journal='deeper-9000-v2', table='labels', key=meta['state_key'],
                     sha256=ref['sha256'], depth=8), 'H8 target reference changed')
    raw = db.execute('SELECT body,hash FROM roots WHERE state=?', (ref['key'],)).fetchone()
    need(raw is not None, 'Original source root missing')
    root = json.loads(raw[0]); need(config.digest(root) == raw[1] == meta['source_sha256'], 'Original root hash differs')
    need(root['state'] == meta['state'] and root['state_key'] == meta['state_key'] and
         root['family'] == meta['family'] and root['stratum'] == meta['stratum'] and
         root['split'] == meta['split'] == 'train' and root['rules_digest'] == config.RULES_DIGEST and
         root['legal_mask'] == meta['legal_mask'], 'Original root provenance differs')
    raw = db.execute('SELECT body,hash FROM labels WHERE state=?', (ref['key'],)).fetchone()
    need(raw is not None, 'Reserved H8 label missing')
    item = json.loads(raw[0]); need(config.digest(item) == raw[1] == ref['sha256'], 'Reserved H8 label hash differs')
    need(item['root_sha256'] == meta['source_sha256'], 'H8 label is for another source root')
    label, report = item['label'], item['report']; actor = meta['actor']; state = meta['state']
    legal = [i for i, n in enumerate(state['pits'][actor]) if n]
    need(actor == state['controller'] and len(legal) >= 2 and meta['legal_mask'] == sum(1 << i for i in legal), 'Root legal actions/controller differ')
    need(label['choice_depth'] == report['choice_depth'] == report['depth_result']['choice_depth'] == 8 and
         report['completed_iteration'] is True and report['proof_status'] == 'finite_horizon_result' and
         label['kind'] == 'finite_choice_horizon' and label['oracle_manifest_sha256'] is None and
         report['evaluator'] == report['depth_result']['evaluator'] == 'H_v1' and
         report['depth_result']['bound'] == 'exact' and report['depth_result']['scope'] == 'finite_choice_horizon' and
         report['position'] == state and report['input_controller'] == report['normalized_controller'] == actor and
         report['score_player'] == actor and report['depth_result']['score_player'] == actor and
         report['legal_actions'] == legal and report['terminal'] is False,
         'H8 result is capped, substituted, normalized differently or incomplete')
    need(report['policy'] == report['depth_result']['policy'] == 'terminal-first/no-oracle/full-root-window-v2' and
         report['search_options']['tt_mb'] == teacher['search']['tt_mb'] and
         report['search_options']['use_pvs'] is teacher['search']['use_pvs'], 'Teacher search policy/configuration differs')
    scores = label['action_scores']; q = {a['pit']: a['value_player0'] for a in scores}
    need(len(q) == len(scores) and set(q) == set(legal) and all(type(y) is int for y in q.values()), 'Incomplete or duplicate legal Q values')
    candidates = report['candidates']; cq = {a['pit']: a['depth_result']['value_player0'] for a in candidates}
    need(len(cq) == len(candidates) and cq == q, 'Report and label root Q differ')
    for action in candidates:
        depth = action['depth_result']
        need(action['analysis_complete'] is True and action['actor'] == actor and depth['choice_depth'] == 8 and
             depth['bound'] == 'exact' and depth['evaluator'] == 'H_v1' and depth['score_player'] == actor and
             depth['value_actor'] == depth['value_player0'] * (1 if actor == 0 else -1) and
             depth['scope'] == 'finite_choice_horizon' and depth['policy'] == report['policy'], 'Capped or mixed-depth action Q')
    y = (max if actor == 0 else min)(q.values()); best = [a for a in legal if q[a] == y]
    need(label['target_player0'] == report['depth_result']['value_player0'] == y and
         report['depth_result']['value_actor'] == y * (1 if actor == 0 else -1) and
         label['best_moves'] == report['best_moves'] == best, 'H8 best value or ties differ')
    return y, dict(reference=ref, root_sha256=meta['source_sha256'], item_sha256=raw[1],
        label_sha256=config.digest(label), report_sha256=config.digest(report),
        teacher_identity_sha256=config.digest(teacher), requested_depth=8, completed_depth=8,
        action_q=[{'pit': a, 'value_player0': q[a]} for a in legal], best_moves=best,
        target_is_solved_truth=False)


def _rows(selection):
    """Extract only the reserved 640 labels and their exact cached feature rows."""
    out = {}; cached = {}
    with closing(previous.ro('deeper-9000-v2')) as db:
        raw = db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        # Historical SQLite meta uses name rather than key in some corpora.
        teacher = json.loads(raw[0]); need(config.digest(teacher) == raw[1], 'Teacher identity hash differs')
        need(teacher['rules_digest'] == config.RULES_DIGEST and teacher['choice_depth'] == 8 and
             teacher['teacher'] == 'H_v1/untrained/oracle-free/exact-completed-root-depth8/full-root-Q' and
             teacher['search'] == dict(max_depth=8, memory_mb=512, node_budget=0, time_ms=0, tt_mb=16, use_pvs=True), 'Teacher identity/configuration differs')
        for role in ROLES:
            records = []
            for meta in selection[role]:
                y, evidence = _teacher(meta, db, teacher)
                ref = meta['feature_ref']; filename = ref['path']
                if filename not in cached: cached[filename] = (ROOT / filename).read_text(encoding='utf-8')
                base = '$.groups.' + ref['group'] + '[' + str(ref['index']) + ']'
                values = db.execute('SELECT json_extract(?,?),json_extract(?,?),json_extract(?,?),json_extract(?,?)',
                    (cached[filename], base + '.cached_features', cached[filename], base + '.state',
                     cached[filename], base + '.state_key', cached[filename], base + '.family')).fetchone()
                need(values[0] is not None and json.loads(values[1]) == meta['state'] and values[2] == meta['state_key'] and values[3] == meta['family'], 'Feature cache source/origin differs')
                full = json.loads(values[0]); features = {k: full[k] for k in ('x', 'd', 'r')}
                previous.validate_features(features, meta['state'])
                D, r = features['d'], features['r']; z = (y - D) / r
                need(r != 0 and math.isfinite(z) and -1 <= z <= 1, 'H8 normalized target outside residual interval')
                records.append(dict(x=[fp32(v) for v in features['x']], target_z=z, D=D, r=r,
                    target_player0=y, state=meta['state'], state_key=meta['state_key'], actor=meta['actor'],
                    family=meta['family'], component=meta['component'], cell=meta['cell'],
                    stratum=meta['stratum'], reserved_role=role, metadata_sha256=config.digest(meta),
                    feature_source=dict(reference=ref, full_cached_features_sha256=config.digest(full),
                        extracted_features_sha256=config.digest(features), file_sha256=sha(ROOT / filename)),
                    target_source=evidence))
            out[role] = records
    return out


def _pause(path, point, requested, marker):
    if point != requested: return
    need(path != DEFAULT.resolve(), 'Production data pause hook forbidden')
    need(marker is not None, 'Pause marker required')
    atomic(Path(marker), dict(point=point, pid=os.getpid(), output=str(path)))
    while True: time.sleep(0.05)


def prepare(output=DEFAULT, pause_at=None, pause_marker=None):
    path = _path(output)
    # The completed path runs before Writer and before reading target/cache bodies.
    if (path / 'catalog.json').exists() or (path / 'complete.json').exists(): return admit(path)
    need(pause_at in (None, *HOOKS), 'Unknown cache pause hook')
    frozen = identity(); path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        atomic(path / 'identity.json', frozen)
        selection = read(reservations.DEFAULT / 'selection.json')
        rows = _rows(selection)
        for role, (filename, count) in ROLES.items():
            need(len(rows[role]) == count, 'Reserved role count differs')
            atomic(path / filename, dict(schema=SCHEMA + '-role', role=role, rows=rows[role]))
        manifest = dict(schema=SCHEMA + '-manifest', identity_sha256=sha(path / 'identity.json'),
            role_files={role: dict(filename=filename, sha256=sha(path / filename), count=count,
                row_sha256=[config.digest(row) for row in rows[role]],
                components=[row['component'] for row in rows[role]]) for role, (filename, count) in ROLES.items()},
            labels_read=640, arena_labels_read=0, features_packed=640, model_forwards=0,
            optimizer_steps=0, policy_fixed=True, source_identity=frozen)
        atomic(path / 'manifest.json', manifest)
        certificate = dict(schema=SCHEMA + '-certificate', passed=True,
            files=common.pins([path / name for name in ('identity.json', 'train.json', 'fit-check.json', 'manifest.json')]))
        atomic(path / 'certificate.json', certificate)
        _pause(path, 'before_data_catalog', pause_at, pause_marker)
        atomic(path / 'catalog.json', dict(schema=SCHEMA + '-catalog', complete=True,
            manifest_sha256=sha(path / 'manifest.json'), certificate_sha256=sha(path / 'certificate.json')))
        _pause(path, 'after_data_catalog', pause_at, pause_marker)
        atomic(path / 'complete.json', dict(schema=SCHEMA + '-complete', catalog_sha256=sha(path / 'catalog.json')))
    return admit(path)


def admit(output=DEFAULT):
    """Completed read-only metadata/hash verification; never opens role JSON bodies."""
    path = _path(output); catalog = read(path / 'catalog.json'); frozen = read(path / 'identity.json')
    need(catalog['schema'] == SCHEMA + '-catalog' and catalog['complete'] is True, 'Training data catalog incomplete')
    runtime = frozen['preparation_runtime']
    need(set(runtime) == {'python', 'implementation', 'machine', 'byteorder'} and
         runtime['implementation'] == 'CPython' and runtime['byteorder'] == 'little', 'Preparation runtime provenance differs')
    need(frozen == identity(preparation_runtime=runtime), 'Training data source/runtime identity changed')
    certificate = read(path / 'certificate.json'); manifest = read(path / 'manifest.json')
    need(certificate['schema'] == SCHEMA + '-certificate' and certificate['passed'] is True and
         catalog['certificate_sha256'] == sha(path / 'certificate.json') and
         catalog['manifest_sha256'] == sha(path / 'manifest.json'), 'Training data certificate/manifest differs')
    common.current(certificate['files'])
    need(set(certificate['files']) == set(common.pins([path / name for name in ('identity.json', 'train.json', 'fit-check.json', 'manifest.json')])), 'Cache file closure differs')
    need(manifest['schema'] == SCHEMA + '-manifest' and manifest['identity_sha256'] == sha(path / 'identity.json') and manifest['source_identity'] == frozen, 'Cache identity differs')
    need(set(manifest['role_files']) == set(ROLES), 'Cache roles differ')
    all_components = []
    for role, (filename, count) in ROLES.items():
        item = manifest['role_files'][role]
        need(item['filename'] == filename and item['count'] == len(item['row_sha256']) == len(item['components']) == count and item['sha256'] == sha(path / filename), 'Role file/count/hash differs')
        all_components += item['components']
    need(len(set(all_components)) == len(all_components) == 640, 'Train and fit-check components overlap')
    need(manifest['labels_read'] == manifest['features_packed'] == 640 and manifest['arena_labels_read'] == manifest['model_forwards'] == manifest['optimizer_steps'] == 0 and manifest['policy_fixed'] is True, 'Unexpected cache activity')
    if (path / 'complete.json').exists():
        complete = read(path / 'complete.json'); need(complete == dict(schema=SCHEMA + '-complete', catalog_sha256=sha(path / 'catalog.json')), 'Completed cache marker differs')
    return manifest


def _load(role, output):
    path = _path(output); manifest = admit(path); meta = manifest['role_files'][role]
    body = read(path / meta['filename']); rows = body['rows']
    need(body['schema'] == SCHEMA + '-role' and body['role'] == role and len(rows) == meta['count'] and [config.digest(row) for row in rows] == meta['row_sha256'], 'Role row hashes differ')
    for row in rows:
        active=sum(map(sum,row['state']['pits']))
        need(row['reserved_role'] == role and row['state_key'] == common.rawkey(row['state']) and
             row['actor'] == row['state']['controller'] and row['D'] == row['state']['stores'][0]-row['state']['stores'][1] and
             row['r'] == active*(1 if row['actor']==0 else -1) and 25<=active<=74 and
             len(row['x']) == 127 and all(math.isfinite(v) and fp32(v) == v for v in row['x']) and
             row['r'] != 0 and -1 <= row['target_z'] <= 1 and
             row['target_z'] == (row['target_player0'] - row['D']) / row['r'], 'Malformed role data')
    return rows, manifest


def load_train(output=DEFAULT): return _load('train', output)
def load_fit(output=DEFAULT): return _load('fit_check', output)
