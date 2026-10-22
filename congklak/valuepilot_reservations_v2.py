"""Joint pilot reservations augmented by protected synthetic development origins.

The accepted v1 HOLD, graph builder, quotas and hash matching remain unchanged.
The v2 namespace adds all 32 ordinary origin closures before matching all roles.
"""
from pathlib import Path
from . import config, neurodiag_common as common, neurodiag_data as previous
from . import valuepilot_reservations as old, ordinary_development as development

ROOT = common.ROOT
DEFAULT = ROOT / 'data/neural-value-pilot-v2/reservations'
SCHEMA = 'congklak-valuepilot-reservations-v2'
SOURCES = ('congklak/valuepilot_reservations_v2.py', 'tools/reserve-valuepilot-v2.py',
           'congklak/valuepilot_reservations.py', 'tools/reserve-valuepilot.py',
           *development.SOURCES)
PLAN, PLAN_ACCEPTANCE, SALT = old.PLAN, old.PLAN_ACCEPTANCE, old.SALT
require, read, sha = common.require, common.read, common.sha
atomic, Writer = old._atomic, old.Writer


def _path(path):
    path = Path(path).resolve()
    require(path == DEFAULT.resolve() or path.is_relative_to(ROOT / 'build'), 'V2 reservation output namespace differs')
    return path


def _generated(inv, pool, graph):
    old_families = {f for c in inv['components'].values() for f in c['families']}
    old_quotients = {q for c in inv['components'].values() for q in c['quotients']}
    seen = set(); families = set(); arena_rows = []
    require(len(pool) == 32, 'Ordinary development pool quota differs')
    for record in pool:
        require(record['accepted'] is True and record['reason'] == 'accepted' and record['mechanism'] == 'ordinary', 'Nonordinary/unaccepted generated arena source')
        family = record['family']; quotients = development.eligible_quotients(record)
        require(family not in old_families and family not in families, 'Generated family duplicates existing origin')
        require(not quotients & old_quotients and not quotients & seen, 'Generated full observation closure overlaps prior/other origin')
        seen.update(quotients); families.add(family)
        for state in development.observations(record):
            if sum(map(sum, state['pits'])) > 4:
                graph.add(family, state, 'optimizer/generated_development_only')
        state = record['state']; cell = record['cell']
        arena_rows.append(dict(state=state, state_key=record['state_key'], family=family,
                               actor=state['controller'], stratum=cell.rsplit('/actor', 1)[0],
                               legal_mask=record['legal_mask'], split='development',
                               source_key='synthetic-ordinary-development-v1/' + str(record['attempt']),
                               source_sha256=config.digest(record), generated_origin=True))
    return arena_rows


def inventory(development_path=development.DEFAULT):
    development_path = development._path(development_path)
    development.admit(development_path)
    parent = old.inventory()
    require(parent == read(development_path / 'parent-inventory.json'), 'Rebuilt current graph differs from generated parent binding')
    graph = old.Graph()
    for component in parent['components'].values():
        nodes = [('family', f) for f in component['families']] + [('quotient', q) for q in component['quotients']]
        if not nodes: continue
        for node in nodes: graph.join(nodes[0], node)
        graph.reasons[nodes[0]].update(component['reasons'])
    pool = read(development_path / 'candidates.json')['accepted']
    arena_rows = _generated(parent, pool, graph)
    components, lookup = graph.finish()
    def describe(row):
        cid = lookup[('family', row['family'])]
        require(cid == lookup[('quotient', previous.quotient(row['state']))], 'V2 origin/quotient closure differs')
        return dict(row, component=cid, quotient=previous.quotient(row['state']),
                    actor=row['state']['controller'], cell=row['stratum'] + '/actor' + str(row['state']['controller']))
    out = dict(parent)
    generated_rows = [describe(row) for row in arena_rows]
    out.update(schema=SCHEMA + '-inventory', components=components,
               primary_pool=[describe(r) for r in parent['primary_pool']],
               arena_pool=[describe(r) for r in parent['arena_pool']] + generated_rows,
               generated_development=dict(path=str(development_path), manifest_sha256=sha(development_path / 'manifest.json'),
                                          accepted_components=32, all_components_optimizer_blocked=True,
                                          accepted_root_keys=sorted(r['state_key'] for r in generated_rows),
                                          accepted_component_ids=sorted(r['component'] for r in generated_rows),
                                          rejected_attempts_reserved=False, opening_reachability_claimed=False))
    require(len({r['component'] for r in generated_rows}) == 32, 'Generated origins merged after complete closure')
    files = [p for p in development_path.rglob('*') if p.is_file() and p.name != 'writer.lock' and not p.name.endswith('.pending')]
    out['source_inputs'] = dict(parent['source_inputs'], **common.pins(files))
    out['counts'] = dict(parent['counts'], full_components=len(components), generated_development_components=32,
                         generated_origin_observations=sum(sum(map(sum, p['pits'])) > 4 for r in pool for p in development.observations(r)))
    return out


def select(inv):
    body = old.select(inv)
    body['schema'] = body['schema'].replace(old.SCHEMA, SCHEMA, 1)
    return validate_selection(body, inv)


def validate_selection(body, inv):
    require(body['schema'] in (SCHEMA + '-selection', SCHEMA + '-selection-hold'), 'V2 selection schema differs')
    converted = dict(body, schema=body['schema'].replace(SCHEMA, old.SCHEMA, 1))
    old.validate_selection(converted, inv)
    if 'generated_development' in inv:
        generated = inv['generated_development']; cids = generated['accepted_component_ids']
        require(len(cids) == len(set(cids)) == 32, 'Generated component reservation quota differs')
        for cid in cids:
            reasons = inv['components'][cid]['reasons']
            require('optimizer/generated_development_only' in reasons, 'Unselected generated origin is not optimizer protected')
        if body['admitted']:
            require(not {r['component'] for r in body['train'] + body['fit_check']} & set(cids), 'Development origins entered optimizer support')
            require({r['component'] for r in body['arena'] if r['stratum'].endswith('/ordinary')} <= set(cids), 'Ordinary arena source not from admitted new development pool')
    return body


def prepare(path=DEFAULT, development_path=development.DEFAULT):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return verify(path)
    if (path / 'identity.json').exists(): return prepared(path)[2]
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        inv = inventory(development_path); selection = select(inv)
        atomic(path / 'inventory.json', inv); atomic(path / 'selection.json', selection)
        identity = dict(schema=SCHEMA + '-identity', rules_digest=config.RULES_DIGEST,
                        plan_sha256=sha(PLAN), plan_acceptance_sha256=sha(PLAN_ACCEPTANCE),
                        inventory_sha256=sha(path / 'inventory.json'), selection_sha256=sha(path / 'selection.json'),
                        sources=common.pins(SOURCES), input_sources=inv['source_inputs'],
                        generated_development_manifest=dict(path=str(Path(development_path).resolve() / 'manifest.json'), sha256=sha(Path(development_path) / 'manifest.json')),
                        training_started=False, optimizer_steps=0, policy_fixed=True)
        atomic(path / 'identity.json', identity)
    return identity


def prepared(path=DEFAULT):
    path = _path(path); identity = read(path / 'identity.json')
    require(identity['schema'] == SCHEMA + '-identity' and identity['rules_digest'] == config.RULES_DIGEST, 'V2 reservation identity differs')
    require(identity['plan_sha256'] == sha(PLAN) and identity['plan_acceptance_sha256'] == sha(PLAN_ACCEPTANCE), 'Accepted value pilot plan changed')
    common.current(identity['sources']); common.current(identity['input_sources'])
    require(identity['inventory_sha256'] == sha(path / 'inventory.json') and identity['selection_sha256'] == sha(path / 'selection.json'), 'V2 prepared metadata changed')
    inv = read(path / 'inventory.json'); selection = read(path / 'selection.json')
    require(inv['schema'] == SCHEMA + '-inventory' and inv['source_inputs'] == identity['input_sources'], 'V2 inventory identity differs')
    require(identity['generated_development_manifest'] == dict(path=str(Path(inv['generated_development']['path']) / 'manifest.json'), sha256=inv['generated_development']['manifest_sha256']), 'Generated manifest binding differs')
    validate_selection(selection, inv)
    require(selection == select(inv), 'V2 deterministic joint matching changed')
    require(identity['training_started'] is False and identity['optimizer_steps'] == 0 and identity['policy_fixed'] is True, 'Unexpected v2 training/policy activity')
    return inv, selection, identity


def _pause(path, point, requested, marker):
    if requested != point: return
    require(path != DEFAULT.resolve(), 'Production pause hook forbidden')
    old._pause(path, point, requested, marker)


def run(path=DEFAULT, pause_at=None, pause_marker=None):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return verify(path)
    require(pause_at in (None, 'before_catalog_commit', 'after_catalog_commit'), 'Unknown v2 reservation pause boundary')
    if pause_at: require(path != DEFAULT.resolve(), 'Production pause hook forbidden')
    inv, selection, identity = prepared(path)
    with Writer(path):
        manifest = dict(schema=SCHEMA + '-manifest', identity_sha256=sha(path / 'identity.json'),
                        selection_sha256=sha(path / 'selection.json'), inventory_sha256=sha(path / 'inventory.json'),
                        counts=selection['counts'], source_inputs=identity['input_sources'],
                        status=selection['status'], admitted=selection['admitted'], readiness_blockers=selection['readiness_blockers'],
                        availability=selection['availability'], generated_development=inv['generated_development'],
                        component_roles={r['component']: role for role in ('train', 'fit_check', 'arena') for r in selection[role]},
                        all_generated_components_development_only=inv['generated_development']['accepted_component_ids'],
                        target_bodies_read=0, model_forwards=0, searches=0, optimizer_steps=0, new_games=0,
                        models_and_policy_fixed=True, training_started=False)
        atomic(path / 'manifest.json', manifest)
        certificate = dict(schema=SCHEMA + '-certificate', identity_sha256=sha(path / 'identity.json'),
                           manifest_sha256=sha(path / 'manifest.json'), files=common.pins([path / n for n in ('identity.json', 'selection.json', 'inventory.json', 'manifest.json')]), passed=True, training_started=False)
        atomic(path / 'certificate.pending.json', certificate)
        _pause(path, 'before_catalog_commit', pause_at, pause_marker)
        atomic(path / 'catalog.json', dict(schema=SCHEMA + '-catalog', complete=True, manifest_sha256=sha(path / 'manifest.json'), certificate_sha256=sha(path / 'certificate.pending.json')))
        _pause(path, 'after_catalog_commit', pause_at, pause_marker)
    return verify(path)


def verify(path=DEFAULT):
    path = _path(path); inv, selection, identity = prepared(path)
    catalog = read(path / 'catalog.json'); certificate = read(path / 'certificate.pending.json')
    require(catalog['schema'] == SCHEMA + '-catalog' and catalog['complete'] is True, 'V2 reservation catalog incomplete')
    require(catalog['certificate_sha256'] == sha(path / 'certificate.pending.json') and certificate['passed'] is True, 'V2 reservation certificate missing/corrupt')
    require(certificate['schema'] == SCHEMA + '-certificate' and certificate['files'] == common.pins([path / n for n in ('identity.json', 'selection.json', 'inventory.json', 'manifest.json')]), 'V2 certificate file closure differs')
    common.current(certificate['files'])
    manifest = read(path / 'manifest.json')
    require(catalog['manifest_sha256'] == sha(path / 'manifest.json') == certificate['manifest_sha256'], 'V2 manifest differs')
    require(manifest['identity_sha256'] == sha(path / 'identity.json') == certificate['identity_sha256'], 'V2 identity binding differs')
    require(manifest['selection_sha256'] == sha(path / 'selection.json') and manifest['inventory_sha256'] == sha(path / 'inventory.json'), 'V2 metadata binding differs')
    require(manifest['counts'] == selection['counts'] and manifest['component_roles'] == {r['component']: role for role in ('train', 'fit_check', 'arena') for r in selection[role]}, 'V2 role catalog differs')
    require(all(common.same(manifest[k], selection[k]) for k in ('status', 'admitted', 'readiness_blockers', 'availability')), 'V2 admission evidence differs')
    require(manifest['generated_development'] == inv['generated_development'] and manifest['all_generated_components_development_only'] == inv['generated_development']['accepted_component_ids'], 'V2 unused development reservation differs')
    require(manifest['source_inputs'] == identity['input_sources'], 'V2 manifest source closure differs')
    require(all(manifest[k] == 0 for k in ('target_bodies_read', 'model_forwards', 'searches', 'optimizer_steps', 'new_games')) and manifest['training_started'] is False and manifest['models_and_policy_fixed'] is True, 'Unexpected v2 execution activity')
    return manifest


def admit(path=DEFAULT):
    manifest = verify(path)
    require(manifest['admitted'] is True and manifest['status'] == 'reserved', 'Value pilot v2 data admission held: ' + '; '.join(manifest['readiness_blockers']))
    return manifest


def resolve_positions(path=DEFAULT):
    """Exact admitted raw roots; does not read target bodies or encode features."""
    admit(path); selection = read(_path(path) / 'selection.json')
    return [dict(row, reserved_role=role) for role in ('train', 'fit_check', 'arena') for row in selection[role]]


def status(path=DEFAULT):
    path = _path(path)
    if (path / 'catalog.json').exists() or (path / 'certificate.json').exists(): return dict(verify(path), publication_complete=True)
    if (path / 'identity.json').exists():
        _, selection, _ = prepared(path)
        return dict(status='prepared', admitted=False, selection_status=selection['status'], readiness_blockers=selection['readiness_blockers'], counts=selection['counts'], training_started=False)
    return dict(status='not_started', training_started=False)
