"""Read-only forensic ownership units for the frozen 400-step value arena.

The coordinator owns durability.  This module only reads existing artifacts and
replays saved actions with the independent literal rules; it never searches,
trains, runs a model, starts a game, or changes the predeclared ownership rule.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from contextlib import closing
import hashlib
import json
import sqlite3
from . import config, neurodiag_common as common, neurodiag_data as metadata
from . import value_run_arena as arena
from .valuepilot_reservations import ARENAS as HISTORICAL_ARENAS

ROOT = common.ROOT
ARENA = ROOT / 'data/value-run-v1/arena'
RESERVATIONS = ROOT / 'data/neural-value-pilot-v2/reservations'
SCHEMA = 'value-bridge-diagnostic-v1'
SOURCES = ('congklak/value_bridge_diagnostic.py', 'congklak/value_run_arena.py',
           'congklak/neurodiag_data.py', 'congklak/neurodiag_common.py',
           'congklak/config.py', 'tests/reference.py',
           'congklak/valuepilot_reservations.py', 'congklak/valuepilot_reservations_v2.py')
METADATA_FILES = (ARENA / 'identity.json', ARENA / 'head.json', ARENA / 'result.json',
                  ARENA / 'certificate.json', ARENA / 'COMPLETED.json',
                  RESERVATIONS / 'inventory.json', RESERVATIONS / 'selection.json',
                  ROOT / 'data/quality-train-v1/model.json',
                  ROOT / 'data/value-training-v1/training/model.json')
ORIGIN_FILES = ('data/deeper-9000-v2/work.sqlite3', 'data/pilot-10000/work.sqlite3')
HISTORICAL_GAMES_PER_SOURCE = 8
need, read, sha = common.require, common.read, common.sha


def _rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def _record(entry, kind):
    need(set(entry) == {'name', 'sha256'}, 'Bridge raw entry fields differ')
    relative = Path(entry['name'])
    need(not relative.is_absolute() and len(relative.parts) == 2 and
         relative.parts[0] == kind and relative.suffix == '.json', 'Unsafe bridge raw path')
    path = ARENA / relative
    need(sha(path) == entry['sha256'], 'Bridge raw source changed: ' + entry['name'])
    return read(path)


def plan():
    """Return immutable per-game descriptors and all actually read source pins."""
    identity = read(ARENA / 'identity.json'); head = read(ARENA / 'head.json')
    need(len(identity['schedule']) == len(head['games']) == 192, 'Bridge arena incomplete')
    need(len(head['decisions']) == 6663, 'Bridge decision cardinality differs')
    by_game = defaultdict(list)
    for entry in head['decisions']:
        row = _record(entry, 'decisions')
        by_game[row['game_index']].append(entry)
    metadata_pins = common.pins(METADATA_FILES)
    tasks = []
    for index, game in enumerate(head['games']):
        saved = _record(game, 'games')
        need(saved['index'] == index and len(by_game[index]) == saved['choices'],
             'Bridge game prefix differs')
        tasks.append(dict(index=index, kind='arena', game_index=index, game=game,
                          decisions=by_game[index], metadata_pins=metadata_pins))
    for filename in HISTORICAL_ARENAS:
        tasks.append(dict(index=len(tasks),kind='historical_provenance',source=filename,
            source_sha256=sha(ROOT/filename),metadata_pins=metadata_pins,
            origin_pins=common.pins(ORIGIN_FILES),hash_ordered_game_limit=HISTORICAL_GAMES_PER_SOURCE))
    files = list(METADATA_FILES) + list(SOURCES)
    files += list(HISTORICAL_ARENAS)+list(ORIGIN_FILES)
    files += [ARENA / e['name'] for kind in ('games', 'decisions') for e in head[kind]]
    return dict(schema=SCHEMA + '-plan', tasks=tasks, inputs=common.pins(files),
                raw_games=192, raw_decisions=6663, models_and_policy_fixed=True,
                new_games=0, searches=0, model_forwards=0, optimizer_steps=0,
                new_teacher_labels=0, eligible_predicate='active seeds >4',
                historical_provenance_sources=len(HISTORICAL_ARENAS),historical_game_limit_per_source=HISTORICAL_GAMES_PER_SOURCE,
                units='192 complete current-game replays and 15 bounded historical saved-game provenance probes')


@lru_cache(maxsize=4)
def _context(inventory_sha, selection_sha, identity_sha):
    # Hash keys prevent a stale cached object from granting admission after edits.
    need(sha(RESERVATIONS / 'inventory.json') == inventory_sha and
         sha(RESERVATIONS / 'selection.json') == selection_sha and
         sha(ARENA / 'identity.json') == identity_sha, 'Bridge metadata changed')
    return (read(RESERVATIONS / 'inventory.json'), read(RESERVATIONS / 'selection.json'),
            read(ARENA / 'identity.json'), arena._literal())


def _load_context(pins):
    common.current(pins)
    return _context(pins[_rel(RESERVATIONS / 'inventory.json')],
                    pins[_rel(RESERVATIONS / 'selection.json')],
                    pins[_rel(ARENA / 'identity.json')])


def _observation(state, choice, source, preceding, unit):
    active = sum(map(sum, state['pits']))
    return dict(choice=choice, state=state, state_key=common.rawkey(state),
                quotient=metadata.quotient(state), active_seeds=active,
                eligible=active > 4, updated_value_branch=25 <= active <= 74,
                source=source, preceding=preceding,
                game=unit['index'], family_index=unit['family_index'],
                family=unit['family'], component=unit['component'],
                profile=unit['profile'], baseline=unit['baseline'], seat=unit['seat'])


def unit(task):
    """Compute one recoverable unit.  All actions come from existing records."""
    if task.get('kind')=='historical_provenance':return _historical_unit(task)
    need(task.get('kind')=='arena', 'Unknown bridge diagnostic unit kind')
    inventory, selection, identity, ref = _load_context(task['metadata_pins'])
    index = task['game_index']; need(task['index'] == index, 'Bridge task index differs')
    scheduled = identity['schedule'][index]
    need(scheduled['index'] == index, 'Bridge schedule differs')
    game = _record(task['game'], 'games')
    rows = [_record(e, 'decisions') for e in task['decisions']]
    p = arena.normalize(scheduled['start'], ref)['state']
    observations = [_observation(p, 0, dict(path=_rel(ARENA / 'identity.json'),
        sha256=task['metadata_pins'][_rel(ARENA / 'identity.json')],
        pointer='/schedule/' + str(index) + '/start'), None, scheduled)]
    for choice, (row, entry) in enumerate(zip(rows, task['decisions'])):
        need(row['game_index'] == index and row['choice_index'] == choice and row['before'] == p,
             'Bridge raw trajectory cursor differs')
        need(row['schedule_sha256'] == config.digest(scheduled), 'Bridge schedule hash differs')
        normal = arena.normalize(p, ref)
        need(not normal['terminal'] and row['action'] in normal['legal'], 'Bridge saved action illegal')
        expected = arena.transition(p, row['action'], ref)
        need(row['transition'] == expected, 'Bridge literal replay differs')
        p = expected['after']
        preceding = dict(decision_index=row['index'], choice_index=choice,
            engine=row['engine'], action=row['action'], before=row['before'],
            captured=expected['captured'], relay_pickups=expected['relay_pickups'],
            deposits=expected['deposits'], bonus=expected['bonus'], passes=expected['passes'])
        observations.append(_observation(p, choice + 1,
            dict(path=_rel(ARENA / entry['name']), sha256=entry['sha256'],
                 pointer='/transition/after'), preceding, scheduled))
    arena._validate_game(game, scheduled, rows, p, ref)
    return dict(schema=SCHEMA + '-unit', index=index, kind='arena',game_index=index,
        task_sha256=config.digest(task), game_source=dict(path=_rel(ARENA / task['game']['name']),
        sha256=task['game']['sha256']), decisions=len(rows), literal_replayed_actions=len(rows),
        observations=observations, terminal=game['terminal'], fallbacks=game['fallbacks'],
        optimizer_steps=0, searches=0, model_forwards=0, new_games=0, new_teacher_labels=0)


def _ro(filename):
    return sqlite3.connect((ROOT/filename).resolve().as_uri()+'?mode=ro',uri=True)


@lru_cache(maxsize=2)
def _origin_family_map(deeper_sha,pilot_sha):
    need(sha(ROOT/ORIGIN_FILES[0])==deeper_sha and sha(ROOT/ORIGIN_FILES[1])==pilot_sha,
         'Historical origin map inputs changed')
    with closing(_ro(ORIGIN_FILES[0])) as db:
        families={key:family for key,family in db.execute("SELECT state,json_extract(body,'$.family') FROM roots ORDER BY state")}
    with closing(_ro(ORIGIN_FILES[1])) as db:
        for key,family in db.execute("SELECT state,family FROM selected WHERE split='development' ORDER BY state"):
            families.setdefault(bytes(key).hex(),family)
    return families


def _historical_unit(task):
    """Bounded saved-action witnesses; explicitly not a historical graph rebuild."""
    inv,selection,identity,ref=_load_context(task['metadata_pins'])
    common.current(task['origin_pins'])
    filename=task['source']
    need(filename in HISTORICAL_ARENAS and sha(ROOT/filename)==task['source_sha256'],
         'Historical probe source changed')
    need(task['hash_ordered_game_limit']==HISTORICAL_GAMES_PER_SOURCE,'Historical probe bound differs')
    family_map=_origin_family_map(*(task['origin_pins'][p] for p in ORIGIN_FILES))
    frozen=read(ARENA/'result.json')['ownership']
    targets={r['quotient'] for r in frozen['broader_ancestral_hits']}
    target_components={r['historical_component'] for r in frozen['cross_arena_alias_component_hits']}
    old_qcomponent={q:cid for cid,c in inv['components'].items() if cid in target_components for q in c['quotients']}
    edges={}; sampled=[]; transitions=0;eligible_observations=0
    with closing(_ro(filename)) as db:
        total=db.execute('SELECT count(*) FROM games').fetchone()[0]
        saved=list(db.execute('SELECT hash,body FROM games ORDER BY hash LIMIT ?', (HISTORICAL_GAMES_PER_SOURCE,)))
    for row_hash,text in saved:
        body=json.loads(text);family=body.get('family') or family_map.get(body.get('start_key'))
        need(family is not None,'Historical saved game origin cannot be resolved')
        p=arena.normalize(body['start'],ref)['state'];moves=body['moves']
        source=dict(path=filename,sha256=task['source_sha256'],table='games',row_hash=row_hash,
                    body_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
        sampled.append(dict(source=source,family=family,choices=len(moves),start=body['start']))
        for choice,action in enumerate([*moves,None]):
            active=sum(map(sum,p['pits']))
            if active>4:
                eligible_observations+=1;q=metadata.quotient(p)
                if q in targets or q in old_qcomponent:
                    key=(family,q)
                    if key not in edges:
                        edges[key]=dict(source=source,choice=choice,family=family,quotient=q,state=p,
                            active_seeds=active,historical_component=old_qcomponent.get(q),
                            current_arena_ancestral_hit_quotient=q in targets,
                            start=body['start'],saved_prefix_actions=moves[:choice],
                            edge='existing origin family -> canonical quotient',occurrences=0)
                    edges[key]['occurrences']+=1
            if action is not None:
                need(action in arena.normalize(p,ref)['legal'],'Historical saved action illegal')
                p=arena.transition(p,action,ref)['after'];p=arena.normalize(p,ref)['state'];transitions+=1
        need(p==body['final'],'Historical bounded probe final state differs')
    return dict(schema=SCHEMA+'-unit',kind='historical_provenance',index=task['index'],
        task_sha256=config.digest(task),source=filename,source_sha256=task['source_sha256'],
        total_saved_games=total,hash_ordered_game_limit=HISTORICAL_GAMES_PER_SOURCE,
        sampled_saved_games=len(sampled),sampled_games=sampled,literal_replayed_actions=transitions,
        eligible_observations=eligible_observations,edges=list(edges.values()),
        exhaustive_historical_graph_reconstruction=False,optimizer_steps=0,searches=0,
        model_forwards=0,new_games=0,new_teacher_labels=0)


def _brief(observation):
    return {k: observation[k] for k in ('game', 'choice', 'family_index', 'component', 'quotient')}


def _band_counts(rows):
    active = [r['active_seeds'] for r in rows]
    return dict(observations=len(rows), active_seed_counts=dict(sorted(Counter(active).items())),
                in_updated_branch=sum(r['updated_value_branch'] for r in rows),
                outside_updated_branch=sum(not r['updated_value_branch'] for r in rows),
                active_min=min(active) if active else None, active_max=max(active) if active else None)


def _canonical_relation(observation, origin):
    """Describe quotient equivalence without confusing it with raw-state equality."""
    p = observation['state']; s = origin['state']
    same = p['pits'] == s['pits'] and p['controller'] == s['controller']
    swapped = p['pits'][0] == s['pits'][1] and p['pits'][1] == s['pits'][0] and p['controller'] == 1-s['controller']
    need(same or swapped, 'Bridge quotient alias lacks a direct symmetry witness')
    stores = s['stores'] if same else list(reversed(s['stores']))
    return dict(raw_state_equal=p == s, player_swap=not same,
                store_offsets=[p['stores'][i]-stores[i] for i in (0, 1)],
                pits_and_actor_equal_after_swap=True,
                explanation='The quotient drops both stores and optionally swaps the two player sides and actor; pit order within each side is retained.')


def _origin_rows(inv, quotient=None, component=None):
    out=[]; seen=set()
    for pool in ('primary_pool', 'arena_pool'):
        for i, row in enumerate(inv[pool]):
            if (quotient is None or row['quotient'] == quotient) and (component is None or row['component'] == component):
                key = (row['source_key'], row['family'])
                if key not in seen:
                    out.append(dict(inventory_pointer='/' + pool + '/' + str(i), **row)); seen.add(key)
    return out


def summarize(results):
    """Reconstruct the original ownership result and explain its bridge witnesses."""
    planned = plan(); tasks=planned['tasks']
    need(len(results) == len(tasks), 'Bridge summary requires all committed diagnostic units')
    by_index={r['index']:r for r in results}
    need(set(by_index)==set(range(len(tasks))), 'Bridge duplicate/missing units')
    rows=[]; historical_results=[]
    for task in tasks:
        r=by_index[task['index']]
        need(r['schema']==SCHEMA+'-unit' and r['task_sha256']==config.digest(task), 'Bridge unit source/task changed')
        if task['kind']=='historical_provenance':
            need(r['kind']==task['kind'] and r['source']==task['source'] and
                 r['source_sha256']==task['source_sha256'],'Bridge historical result source differs')
            historical_results.append(r);continue
        need(r['decisions']==len(task['decisions']) and r['terminal'] and not r['fallbacks'], 'Bridge unit game incomplete')
        need(len(r['observations'])==r['decisions']+1, 'Bridge observation count differs')
        rows.extend(r['observations'])
    inv,selection,identity,_ = _load_context(tasks[0]['metadata_pins'])
    ancestral=defaultdict(set); qroles=defaultdict(set); froles=defaultdict(set)
    for cid,c in inv['components'].items():
        for q in c['quotients']: ancestral[q].add(cid)
    for role in ('train','fit_check'):
        for reserved in selection[role]:
            c=inv['components'][reserved['component']]
            for q in c['quotients']: qroles[q].add(role)
            for f in c['families']: froles[f].add(role)
    arena_components={u['family_index']:u['component'] for u in identity['schedule']}
    qowners=defaultdict(set); cowners=defaultdict(set); role_hits=[]; old_hits=[]
    eligible=[r for r in rows if r['eligible']]
    for f,c in arena_components.items():cowners[c].add(f)
    byq=defaultdict(list); byc=defaultdict(list)
    for r in eligible:
        q=r['quotient']; owners=set(ancestral.get(q,()))
        byq[q].append(r); qowners[q].add(r['family_index'])
        for c in owners: cowners[c].add(r['family_index']); byc[c].append(r)
        roles=qroles[q]|froles[r['family']]
        if roles:role_hits.append(dict(_brief(r),roles=sorted(roles)))
        for f,c in arena_components.items():
            if f!=r['family_index'] and c in owners:qowners[q].add(f)
        foreign=owners-{r['component']}
        if foreign:old_hits.append(dict(_brief(r),ancestral_components=sorted(foreign)))
    cross=[dict(quotient=q,family_indices=sorted(v)) for q,v in sorted(qowners.items()) if len(v)>1]
    alias=[dict(historical_component=c,family_indices=sorted(v)) for c,v in sorted(cowners.items()) if len(v)>1]
    reconstructed=dict(checked=True,eligible_predicate='active seeds >4, full origin family/quotient aliases',
        frozen_support_roots=512,frozen_fit_roots=128,all_support_counted_even_if_never_drawn=True,
        observations=len(eligible),distinct_observed_quotients=len(qowners),support_or_fit_hits=role_hits,
        cross_arena_component_hits=cross,cross_arena_alias_component_hits=alias,broader_ancestral_hits=old_hits,
        broader_ancestral_hits_not_automatic_gradient_pool=True,ownership_passed=not role_hits and not cross and not alias,
        posthoc_filtering=False,posthoc_cluster_merging=False)
    frozen=read(ARENA/'result.json')['ownership']
    need(reconstructed==frozen, 'Bridge independent reconstruction differs from frozen ownership result')
    support=[]
    for hit in role_hits:
        observation=next(r for r in byq[hit['quotient']] if r['game']==hit['game'] and r['choice']==hit['choice'])
        owner_details=[]
        for cid in sorted(ancestral[hit['quotient']]):
            reservations=[dict(role=role,**r) for role in ('train','fit_check') for r in selection[role] if r['component']==cid]
            origins=_origin_rows(inv,quotient=hit['quotient'],component=cid)
            owner_details.append(dict(component=cid,component_body=inv['components'][cid],
                reserved_roots=reservations,matching_origin_rows=[dict(origin=r,canonical_relation=_canonical_relation(observation,r)) for r in origins],
                linkage='Full origin family membership, not necessarily the exact optimizer root.'))
        support.append(dict(observation=observation,roles=hit['roles'],owners=owner_details,
            path=[r for r in rows if r['game']==hit['game'] and r['choice']<=hit['choice']]))
    direct=[]
    for hit in cross:
        seen=byq[hit['quotient']]; observed=sorted({r['family_index'] for r in seen})
        first={str(f):next(r for r in seen if r['family_index']==f) for f in observed}
        direct.append(dict(**hit,observed_family_indices=observed,
            inferred_from_preexisting_arena_alias=sorted(set(hit['family_indices'])-set(observed)),
            evidence=seen,earliest_per_family=first,**_band_counts(seen),
            matching_inventory_origin_rows=_origin_rows(inv,quotient=hit['quotient'])))
    historical=[]
    for hit in alias:
        cid=hit['historical_component']; seen=byc[cid]; body=inv['components'][cid]
        first={str(f):next((r for r in seen if r['family_index']==f),None) for f in hit['family_indices']}
        qs={q:[] for q in sorted({r['quotient'] for r in seen})}
        for r in seen:qs[r['quotient']].append(r)
        historical.append(dict(**hit,component_body=body,component_families=len(body['families']),
            component_quotients=len(body['quotients']),earliest_per_family=first,
            observed_quotient_families={q:sorted({r['family_index'] for r in v}) for q,v in qs.items()},
            evidence=seen,**_band_counts(seen),
            interpretation='Different observed quotient nodes already belong to one frozen historical component; this is a transitive historical alias bridge, not a claim that every pair played an identical state.',
            provenance=dict(path=_rel(RESERVATIONS/'inventory.json'),sha256=sha(RESERVATIONS/'inventory.json'),
                pointer='/components/'+cid,graph_builder_sources=common.pins(['congklak/valuepilot_reservations.py','congklak/valuepilot_reservations_v2.py']),
                flattened_component_does_not_retain_each_union_edge=True)))
    old_edges=[e for r in historical_results for e in r['edges']]
    old_byq=defaultdict(list)
    for edge in old_edges:old_byq[edge['quotient']].append(edge)
    examples=[]
    for q,edges in sorted(old_byq.items()):
        families=sorted({e['family'] for e in edges})
        if len(families)>1:
            examples.append(dict(quotient=q,families=families,
                historical_components=sorted({e['historical_component'] for e in edges if e['historical_component']}),
                first_edge_per_family={f:next(e for e in edges if e['family']==f) for f in families}))
    for bridge in historical:
        cid=bridge['historical_component'];bridge['matching_origin_rows']=[r for r in _origin_rows(inv,component=cid) if r['quotient'] in bridge['observed_quotient_families']]
        bridge['sampled_saved_action_edges']=[e for e in old_edges if e['historical_component']==cid]
        bridge['sampled_cross_old_origin_collisions']=[e for e in examples if cid in e['historical_components']]
    return dict(schema=SCHEMA+'-report',inputs=planned['inputs'],units=len(tasks),current_arena_units=192,
        literal_replayed_actions=sum(r['literal_replayed_actions'] for r in results if r['kind']=='arena'),
        historical_literal_replayed_actions=sum(r['literal_replayed_actions'] for r in historical_results),
        historical_provenance=dict(sources=historical_results,sampled_cross_origin_collision_examples=examples,
            source_count=len(historical_results),sampled_saved_games=sum(r['sampled_saved_games'] for r in historical_results),
            total_existing_saved_games=sum(r['total_saved_games'] for r in historical_results),
            exhaustive=False,selection='First8 games ordered by saved row hash per predeclared historical arena DB; bounded illustrative saved-action probe, not statistical sampling.'),
        raw_observations=len(rows),eligible_observations=len(eligible),ineligible_observations=len(rows)-len(eligible),
        original_ownership_reconstructed_exactly=True,ownership=reconstructed,
        training_alias_witnesses=support,direct_quotient_bridges=direct,historical_component_bridges=historical,
        support_hit_bands=_band_counts([w['observation'] for w in support]),
        direct_bridge_bands=_band_counts([r for h in direct for r in h['evidence']]),
        historical_bridge_bands=_band_counts([r for h in historical for r in h['evidence']]),
        models_and_policy_fixed=True,optimizer_steps=0,model_forwards=0,searches=0,new_games=0,new_teacher_labels=0,
        ownership_hold_remains=True,strength_interpretation_allowed=False,posthoc_filtering=False,
        posthoc_cluster_merging=False,bootstrap_performed=False,
        limitations=[
            'Active-band classifications are descriptive diagnostics and do not change the frozen active-seeds>4 eligibility rule.',
            'The flattened reservation inventory retains component membership and reason tags, not every historical union edge; no minimal causal edge chain is claimed for the giant component.',
            'Repeated trajectories and canonical aliases are observations from the fixed arena, not independent validation samples.',
            'Synthetic ordinary start reachability and representativeness are not established by these saved games.',
            'No result here establishes a valid strength confidence interval or permits candidate promotion.'])
