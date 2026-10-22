"""Existing-data feasibility checks for a prospective value-only ownership domain.

This is a diagnostic projection, not an admission builder.  A flattened old
component does not preserve its family/quotient edges.  Its lists cannot be cut
into new components without a complete, independently source-bound incidence
rebuild.  The coordinator owns interruption-safe publication of these units.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from . import config, neurodiag_common as common, neurodiag_data as metadata
from . import value_run_arena as arena

ROOT = common.ROOT
SCHEMA = 'ownership-domain-projection-v1'
ARENA = ROOT / 'data/value-run-v1/arena'
RESERVATIONS = ROOT / 'data/neural-value-pilot-v2/reservations'
BRIDGES = ROOT / 'build/value-diagnostics-v1/production/bridges.json'
SOURCES = ('congklak/ownership_domain_projection.py', 'congklak/neurodiag_data.py',
           'congklak/value_run_arena.py', 'congklak/config.py',
           'congklak/neurodiag_common.py', 'congklak/divergence_study_common.py',
           'tests/reference.py', 'congklak/valuepilot_reservations.py',
           'congklak/valuepilot_reservations_v2.py')
METADATA = (ARENA / 'identity.json', ARENA / 'head.json',
            RESERVATIONS / 'inventory.json', RESERVATIONS / 'selection.json',
            BRIDGES, ROOT / 'data/quality-train-v1/model.json',
            ROOT / 'data/value-training-v1/training/model.json')
need, read, sha = common.require, common.read, common.sha
ZERO = dict(model_forwards=0, feature_simulations=0, searches=0, new_games=0,
            new_teacher_labels=0, optimizer_steps=0, optimizer_draws=0)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def quotient_active(quotient):
    """Validate the established store-offset/player-swap key and read its mass."""
    need(isinstance(quotient, str), 'Quotient must be hexadecimal text')
    try:
        values = bytes.fromhex(quotient)
    except ValueError as error:
        raise ValueError('Invalid quotient hexadecimal text') from error
    need(len(values) == 15 and values.hex() == quotient, 'Quotient width/case differs')
    cells, actor = values[:14], values[14]
    active = sum(cells)
    need(active <= 98 and actor in (0, 1) and (active > 0 or actor == 0),
         'Quotient seed/actor bounds differ')
    swapped = cells[7:] + cells[:7] + bytes([1 - actor if active else 0])
    need(values == min(values, swapped), 'Noncanonical quotient orientation')
    return active


def band(active):
    need(type(active) is int and 0 <= active <= 98, 'Active-seed bounds differ')
    if active == 0: return 'terminal/0'
    if active < 25: return 'shared_frozen/1-24'
    if active <= 74: return 'owned_updated/25-74'
    return 'owned_frozen_upper/75-98'


def mass_transition(before, after):
    """Check complete-decision states; relay-in-hand intermediates are excluded."""
    a = quotient_active(metadata.quotient(before))
    b = quotient_active(metadata.quotient(after))
    need(b <= a, 'Active seed mass increased across a complete decision')
    need(all(after['stores'][p] >= before['stores'][p] for p in (0, 1)),
         'Store seed mass decreased across a complete decision')
    need(a - b == sum(after['stores']) - sum(before['stores']),
         'Active/store mass accounting differs')
    return a, b


def _record(entry, kind):
    need(set(entry) == {'name', 'sha256'}, 'Projection source entry fields differ')
    path = Path(entry['name'])
    need(not path.is_absolute() and len(path.parts) == 2 and
         path.parts[0] == kind and path.suffix == '.json',
         'Unsafe projection source path')
    need(sha(ARENA / path) == entry['sha256'], 'Projection saved raw source changed')
    return read(ARENA / path)


def plan():
    identity = read(ARENA / 'identity.json'); head = read(ARENA / 'head.json')
    need(len(identity['schedule']) == len(head['games']) == 192 and
         len(head['decisions']) == 6663, 'Projection saved arena incomplete')
    by_game = defaultdict(list)
    for entry in head['decisions']:
        row = _record(entry, 'decisions')
        by_game[row['game_index']].append(entry)
    identity_pin = common.pins([ARENA / 'identity.json'])
    tasks = []
    for index, game in enumerate(head['games']):
        saved = _record(game, 'games')
        need(saved['index'] == index and saved['choices'] == len(by_game[index]),
             'Projection saved game cardinality differs')
        tasks.append(dict(index=index, kind='arena_mass', game_index=index,
                          game=game, decisions=by_game[index], identity_pin=identity_pin))
    tasks.append(dict(index=192, kind='inventory_projection',
                      metadata_pins=common.pins(METADATA)))
    files = [*METADATA, *SOURCES]
    files.extend(ARENA / entry['name'] for kind in ('games', 'decisions') for entry in head[kind])
    return dict(schema=SCHEMA + '-plan', tasks=tasks, inputs=common.pins(files),
                current_arena_games=192, current_arena_decisions=6663,
                ownership_domain='complete decision states with active seeds >=25',
                prospective_primary_root_band=[25, 74], shared_frozen_domain=[0, 24],
                admits_new_data=False, existing_ownership_hold_remains=True, **ZERO)


@lru_cache(maxsize=2)
def _identity(identity_sha):
    need(sha(ARENA / 'identity.json') == identity_sha, 'Projection identity changed')
    return read(ARENA / 'identity.json'), arena._literal()


def unit(task):
    if task.get('kind') == 'inventory_projection': return inventory_unit(task)
    need(task.get('kind') == 'arena_mass', 'Unknown projection unit kind')
    common.current(task['identity_pin'])
    identity, ref = _identity(task['identity_pin'][relative(ARENA / 'identity.json')])
    index = task['game_index']
    need(task['index'] == index and 0 <= index < 192, 'Projection unit index differs')
    scheduled = identity['schedule'][index]
    game = _record(task['game'], 'games')
    rows = [_record(entry, 'decisions') for entry in task['decisions']]
    p = arena.normalize(scheduled['start'], ref)['state']
    counts = Counter([band(sum(map(sum, p['pits'])))])
    below = sum(map(sum, p['pits'])) < 25
    boundary = None; equal_mass = decreasing_mass = 0; states = []
    for choice, (row, entry) in enumerate(zip(rows, task['decisions'])):
        need(row['game_index'] == index and row['choice_index'] == choice and
             row['before'] == p and row['schedule_sha256'] == config.digest(scheduled),
             'Projection saved trajectory cursor differs')
        normal = arena.normalize(p, ref)
        need(not normal['terminal'] and row['action'] in normal['legal'],
             'Projection saved action illegal')
        transition = arena.transition(p, row['action'], ref)
        need(row['transition'] == transition, 'Projection literal transition differs')
        a, b = mass_transition(p, transition['after'])
        equal_mass += a == b; decreasing_mass += b < a
        need(not below or b < 25, 'Shared endgame reentered owned domain')
        if a >= 25 and b < 25:
            need(boundary is None, 'Repeated owned-to-shared boundary')
            boundary = dict(choice=choice, source=relative(ARENA / entry['name']),
                            source_sha256=entry['sha256'], active_before=a, active_after=b,
                            engine=row['engine'], action=row['action'])
        p = transition['after']; below |= b < 25; counts[band(b)] += 1
        states.append(dict(choice=choice, active_before=a, active_after=b,
                           quotient_before=metadata.quotient(row['before']),
                           quotient_after=metadata.quotient(p),
                           source_sha256=entry['sha256']))
    arena._validate_game(game, scheduled, rows, p, ref)
    return dict(schema=SCHEMA + '-unit', index=index, kind='arena_mass',
                task_sha256=config.digest(task), game_index=index,
                family_index=scheduled['family_index'], profile=scheduled['profile'],
                baseline=scheduled['baseline'], seat=scheduled['seat'],
                literal_replayed_actions=len(rows), equal_active_mass=equal_mass,
                decreasing_active_mass=decreasing_mass, mass_increases=0,
                observed_states=len(rows)+1, state_band_counts=dict(sorted(counts.items())),
                first_owned_to_shared=boundary, complete_decision_mass_checks=states,
                shared_to_owned_reentries=0, terminal=game['terminal'], **ZERO)


def _quotient_counts(quotients):
    counts = Counter(band(quotient_active(q)) for q in quotients)
    return dict(total=len(quotients), bands=dict(sorted(counts.items())),
                owned_ge25=sum(n for name, n in counts.items() if name.startswith('owned_')),
                shared_lt25=sum(n for name, n in counts.items() if not name.startswith('owned_')))


def inventory_unit(task):
    need(task.get('kind') == 'inventory_projection' and task['index'] == 192,
         'Projection inventory task differs')
    common.current(task['metadata_pins'])
    inventory = read(RESERVATIONS / 'inventory.json')
    selection = read(RESERVATIONS / 'selection.json'); bridges = read(BRIDGES)
    components = inventory['components']; all_q = set(); family_count = 0
    mixed = high_only = low_only = without_q = 0
    for component in components.values():
        need(set(component) == {'families', 'quotients', 'reasons'},
             'Flattened inventory schema unexpectedly changed')
        qs = component['quotients']; all_q.update(qs)
        family_count += len(component['families'])
        counts = _quotient_counts(qs)
        high, low = counts['owned_ge25'], counts['shared_lt25']
        mixed += bool(high and low); high_only += bool(high and not low)
        low_only += bool(low and not high); without_q += not qs
    role_counts = {}
    for role in ('train', 'fit_check', 'arena'):
        rows = selection[role]; cids = {row['component'] for row in rows}
        qs = {q for cid in cids for q in components[cid]['quotients']}
        families = {f for cid in cids for f in components[cid]['families']}
        role_counts[role] = dict(roots=len(rows), frozen_components=len(cids),
                                origin_families=len(families), alias_quotients=_quotient_counts(qs),
                                root_bands=dict(sorted(Counter(band(sum(map(sum, row['state']['pits']))) for row in rows).items())))
    collisions = {}
    for name, values in (('training_alias_hits', [w['observation'] for w in bridges['training_alias_witnesses']]),
                         ('direct_cross_family', [r for w in bridges['direct_quotient_bridges'] for r in w['evidence']]),
                         ('historical_component', [r for w in bridges['historical_component_bridges'] for r in w['evidence']])):
        collisions[name] = dict(observations=len(values),
                               unique_quotients=len({row['quotient'] for row in values}),
                               band_counts=dict(sorted(Counter(band(row['active_seeds']) for row in values).items())),
                               owned_ge25=sum(row['active_seeds'] >= 25 for row in values),
                               active_seed_counts=dict(sorted(Counter(row['active_seeds'] for row in values).items())))
    giant = max(components.items(), key=lambda item: len(item[1]['quotients']))
    return dict(schema=SCHEMA + '-unit', index=192, kind='inventory_projection',
                task_sha256=config.digest(task), full_inventory_components=len(components),
                full_inventory_families=family_count, unique_quotients=_quotient_counts(all_q),
                old_component_domain_counts=dict(mixed_high_and_low=mixed, owned_only=high_only,
                                                shared_only=low_only, no_quotients=without_q),
                largest_old_component=dict(id=giant[0], families=len(giant[1]['families']),
                                            quotient_counts=_quotient_counts(giant[1]['quotients'])),
                selected_frozen_role_alias_coverage=role_counts,
                existing_arena_collision_domain_counts=collisions,
                family_to_quotient_incidence_present=False, complete_incidence_rebuilt=False,
                fresh_domain_components=None, production_admission=False,
                prospective_graph_status='HOLD: complete source-bound family-to-quotient incidence rebuild required',
                retain_all_in_domain_aliases=True, retained_upper_band=[75, 98],
                old_component_ids_cannot_be_reused_after_domain_cut=True,
                role_exclusions_cannot_be_dropped_without_source_provenance=True,
                current_arena_remains_exploratory=True, current_strength_claim_allowed=False,
                existing_ownership_hold_remains=True, **ZERO)


def summarize(results):
    planned = plan(); tasks = planned['tasks']
    need(len(results) == len(tasks), 'Projection summary requires all units')
    by_index = {row['index']: row for row in results}
    need(set(by_index) == set(range(len(tasks))), 'Projection duplicate/missing unit')
    for task in tasks:
        row = by_index[task['index']]
        need(row['schema'] == SCHEMA + '-unit' and row['kind'] == task['kind'] and
             row['task_sha256'] == config.digest(task), 'Projection unit source/task differs')
        need(all(row[key] == 0 for key in ZERO), 'Projection forbidden work counter differs')
    games = [by_index[index] for index in range(192)]
    need(all(row['terminal'] and row['literal_replayed_actions'] == len(tasks[row['index']]['decisions'])
             and row['mass_increases'] == row['shared_to_owned_reentries'] == 0
             and row['equal_active_mass']+row['decreasing_active_mass'] == row['literal_replayed_actions']
             for row in games), 'Projection mass-check completeness differs')
    counts = Counter()
    for row in games: counts.update(row['state_band_counts'])
    boundaries = [row['first_owned_to_shared'] for row in games if row['first_owned_to_shared'] is not None]
    inventory = by_index[192]
    return dict(schema=SCHEMA + '-report', inputs=planned['inputs'], units=193,
                existing_saved_games_replayed=192,
                literal_replayed_actions=sum(row['literal_replayed_actions'] for row in games),
                observed_complete_states=sum(row['observed_states'] for row in games),
                state_band_counts=dict(sorted(counts.items())),
                mass_increases=0, shared_to_owned_reentries=0,
                owned_to_shared_boundaries=len(boundaries),
                equal_active_mass=sum(row['equal_active_mass'] for row in games),
                decreasing_active_mass=sum(row['decreasing_active_mass'] for row in games),
                all_complete_transitions_checked=True, inventory_projection=inventory,
                production_admission=False, fresh_domain_components=None,
                models_and_policy_fixed=True, existing_ownership_hold_remains=True,
                strength_interpretation_allowed=False, training_authorized=False,
                limitation='Current data are exposed diagnostic evidence; flattened historical components cannot establish a fresh prospective split.',
                **ZERO)
