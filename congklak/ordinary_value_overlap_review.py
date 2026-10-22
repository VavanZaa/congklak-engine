"""Classify the prior admitted-pool hold without changing its allocation."""
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import random
import sqlite3
from . import config, features_relay, pilot
from .native import ROOT, flatten
from .ordinary_value_review_common import baseline, current, pins, publish, read, require

OUTPUT = ROOT / 'build/ordinary-value-overlap-review.json'
SCHEMA = 'ordinary-value-overlap-review-v1'
NAMES = ('build/ordinary-value-review-baseline.json', 'docs/ORDINARY_VALUE_REVIEW_SCOPE.md',
         'congklak/ordinary_value_review_common.py', 'congklak/ordinary_value_overlap_review.py',
         'tools/review-ordinary-value-overlap.py', 'congklak/pilot.py', 'congklak/config.py',
         'congklak/features_relay.py', 'congklak/features_v3.py', 'congklak/native.py', 'build/congklak_native.dll',
         'congklak/ordinary_value_benchmark.py',
         'build/ordinary-value-rows.json', 'build/ordinary-value-data.json', 'build/ordinary-value-registry.json',
         'data/ordinary-value-pilot-v1/result.json', 'build/ordinary-value-audit/model.json',
         'data/ordinary-value-benchmark-v1/work.sqlite3', 'build/ordinary-value-benchmark.json',
         'build/ordinary-value-followup.json', 'build/quality-data.json', 'build/sampling-review.json', 'data/quality-train-v1/result.json',
         'data/quality-train-v1/model.json', 'data/ordinary-value-pilot-v1/model.json')


def state_key(p): return tuple(flatten(p) + [p['controller']])


def swap(p):
    return dict(pits=[p['pits'][1], p['pits'][0]], stores=p['stores'][::-1], controller=1 - p['controller'])


def swap_key(p): return min(state_key(p), state_key(swap(p)))


def band(p):
    n = sum(map(sum, p['pits']))
    return '0-4' if n <= 4 else '5-24' if n <= 24 else '25-49' if n <= 49 else '50-74' if n <= 74 else '75-98'


def draw_ledger(groups, result):
    """Reconstruct the declared sampler without calling a training producer."""
    identity = result['identity']
    require(identity['steps'] == result['steps_completed'] == 200 and identity['seed'] == 20261028
            and identity['batch_size'] == 64 and result['draws'] == 12800, 'Pilot sampler settings differ')
    by_root = {}
    for name in ('high', 'low'):
        grouped = defaultdict(list)
        for index, row in enumerate(groups[name]): grouped[row['state_key']].append(index)
        require(len(grouped) == 12 and all(len(ids) == 2 for ids in grouped.values()), 'Paired starts differ')
        by_root[name] = dict(sorted(grouped.items()))
    cumulative, mass = [], 0.
    strata = Counter(row['stratum'] for row in groups['rehearsal'])
    for row in groups['rehearsal']:
        probability = .9 / len(groups['rehearsal']) + .1 / (len(strata) * strata[row['stratum']])
        require(row['sampling_probability'] == probability, 'Rehearsal weighting differs')
        mass += probability; cumulative.append(mass)
    require(abs(mass - 1) < 1e-12 and identity['sampler'] == {**by_root, 'rehearsal_cdf': cumulative}, 'Sampler identity differs')
    rng = random.Random(identity['seed'])
    ledger = Counter()
    for _ in range(identity['steps']):
        for name in ('high', 'low'):
            keys = sorted(by_root[name])
            for _ in range(16):
                index = rng.choice(by_root[name][rng.choice(keys)])
                ledger[groups[name][index]['id']] += 1
        for _ in range(32):
            index = bisect_right(cumulative, rng.random() * cumulative[-1])
            ledger[groups['rehearsal'][index]['id']] += 1
    expected = read(ROOT / 'build/ordinary-value-audit/model.json')['evidence']['production_optimizer_audit']
    require(dict(ledger) == expected['draw_ledger'] and sum(ledger.values()) == 12800, 'Checkpoint-verified draw ledger differs')
    return dict(sorted(ledger.items()))


def indexes(rows):
    values = {name: defaultdict(list) for name in ('raw_full_state', 'swapped_full_state', 'quotient', 'family')}
    for row in rows:
        for name, key in (('raw_full_state', state_key(row['state'])), ('swapped_full_state', swap_key(row['state'])),
                          ('quotient', pilot.quotient(row['state']).hex()), ('family', row['family'])):
            values[name][key].append(row)
    return values


def match_keys(p, family):
    return dict(raw_full_state=state_key(p), swapped_full_state=swap_key(p),
                quotient=pilot.quotient(p).hex(), family=family)


def alignment(training, observation, ledger, old_admission):
    source, arena = training['state'], observation['state']
    aligned = source
    swapped = False
    if source['pits'] != arena['pits'] or source['controller'] != arena['controller']:
        aligned = swap(source); swapped = True
    require(aligned['pits'] == arena['pits'] and aligned['controller'] == arena['controller'], 'Quotient alignment differs')
    offsets = [arena['stores'][i] - aligned['stores'][i] for i in (0, 1)]
    require(sum(offsets) == 0, 'Aligned seed total differs')
    x_source, d_source, r_source = features_relay.encode(source)
    x_arena, d_arena, r_arena = features_relay.encode(arena)
    require(x_source == training['cached_features']['x'] and x_source == x_arena, 'Quotient encoder equality differs')
    return dict(training_id=training['id'], training_family=training['family'], training_stratum=training['stratum'],
                training_source_row_sha256=training['source_row_sha256'], training_state=source,
                original_training_target_player0=training['target_player0'], target_kind=training['target_kind'],
                original_normalized_actor_residual=(training['target_player0'] - d_source) / r_source,
                current_pilot_draws=ledger.get(training['id'], 0), source_control_training_admission=training['source_id'] in old_admission,
                source_control_actual_draw_count=None,
                source_control_positive_draw_exposure_from_sealed_all_roots_schedule=training['source_id'] in old_admission,
                source_control_exact_draw_count_reconstructed_here=False, player_swapped=swapped,
                aligned_training_state=aligned, arena_state=arena, store_offsets_per_player=offsets,
                player0_margin_offset=offsets[0] - offsets[1], exact_raw_state_equal=source == arena,
                swapped_full_state_equal=aligned == arena, actor_relative_encoder_inputs_identical=True,
                encoder_inputs_sha256=config.digest(x_source), source_offset_and_scale=[d_source, r_source],
                arena_offset_and_scale=[d_arena, r_arena], target_transferred_to_arena=False)


def build():
    frozen = baseline()
    ancestral = read(ROOT / 'build/sampling-review.json')
    inputs = pins(NAMES + tuple(ancestral['inputs']))
    for name, digest in inputs.items():
        if name in frozen['inputs']: require(frozen['inputs'][name] == digest, 'Old input changed: ' + name)
    groups = read(ROOT / 'build/ordinary-value-rows.json')['groups']
    rows = [row for name in ('high', 'low', 'rehearsal') for row in groups[name]]
    require(len(rows) == 7587 and len({r['id'] for r in rows}) == len(rows), 'Admitted rows differ')
    source_admission = read(ROOT / 'build/quality-data.json')
    source_result = read(ROOT / 'data/quality-train-v1/result.json')
    ancestral_observed = ancestral['observed_accepted_run']
    require(ancestral['passed'] is True and ancestral_observed['unique_roots_drawn'] == len(source_admission['train_state_keys']) == 7669
            and ancestral_observed['total_draws'] == source_result['draws'] == 384000,
            'Sealed all-roots ancestral draw evidence differs')
    result = read(ROOT / 'data/ordinary-value-pilot-v1/result.json')
    ledger = draw_ledger(groups, result)
    drawn = [row for row in rows if ledger.get(row['id'], 0)]
    pools = {'admitted': indexes(rows), 'drawn': indexes(drawn)}
    counts = {pool: Counter() for pool in pools}
    bands = {pool: defaultdict(Counter) for pool in pools}
    collisions = []
    observations = []
    arena_families = set()
    games = {}
    with closing(sqlite3.connect((ROOT / 'data/ordinary-value-benchmark-v1/work.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for index, body, digest in db.execute('SELECT idx,body,hash FROM games ORDER BY idx'):
            game = json.loads(body); require(config.digest(game) == digest and game['status'] == 'terminal', 'Corrupt retained game')
            games[index] = game; arena_families.add(game['family'])
        require(len(games) == 96, 'Incomplete arena')
        for game_index, choice, text, digest in db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice'):
            record = json.loads(text); game = games[game_index]
            require(config.digest(record) == digest and record['game_index'] == game_index and record['choice_index'] == choice,
                    'Corrupt saved decision')
            p = config.position(record['state']); n = sum(map(sum, p['pits']))
            require(n > 0, 'Terminal state in decision table')
            observation = dict(game=game_index, choice=choice, family=game['family'], comparison=game['comparison'],
                               pair=game['pair'], candidate_seat=game['candidate_seat'], state=p, active=n,
                               quotient=pilot.quotient(p).hex(), source_decision_sha256=digest, after=record['after'])
            observations.append(observation)
            if n <= 4: continue
            keys = match_keys(p, game['family'])
            hit = {}
            for pool, index in pools.items():
                found = {}
                for name, key in keys.items():
                    matching = index[name].get(key, [])
                    counts[pool][name] += bool(matching)
                    bands[pool][band(p)][name] += bool(matching)
                    found[name] = [row['id'] for row in matching]
                hit[pool] = found
            if hit['admitted']['quotient'] or hit['admitted']['family']:
                record = {**observation, 'matches': hit}
                record['alignments'] = [alignment(row, observation, ledger, set(read(ROOT / 'build/quality-data.json')['train_state_keys']))
                                        for row in pools['admitted']['quotient'].get(observation['quotient'], [])]
                collisions.append(record)
    require(len(observations) == 3268, 'Saved choice denominator differs')
    observed = [row for row in observations if row['active'] > 4]
    prior = read(ROOT / 'build/ordinary-value-benchmark.json')
    original = prior['trajectory_replay_overlap']
    require(len(observed) == original['arena_nonterminal_gt4_observations'] == 2584
            and len(collisions) == original['colliding_observations'] == 2, 'Original guard counts differ')
    projected = [dict(game=row['game'], choice=row['choice'], comparison=row['comparison'], pair=row['pair'],
                      candidate_seat=row['candidate_seat'], family=row['family'], quotient=row['quotient'],
                      quotient_collision=bool(row['matches']['admitted']['quotient']), family_collision=bool(row['matches']['admitted']['family']))
                 for row in collisions]
    require(projected == original['collisions'], 'Original guard collision identities differ')
    registry = read(ROOT / 'build/ordinary-value-registry.json')
    roles = read(ROOT / 'build/ordinary-value-followup.json')['identity']['roles']
    admitted_q = set(pools['admitted']['quotient']); admitted_f = set(pools['admitted']['family'])
    require(not admitted_q.intersection(registry['quotients']) and not admitted_f.intersection(registry['families']), 'Predeclared protected role entered admission')
    observed_q = {row['quotient'] for row in observations}
    require(observed_q <= set(roles['true_evaluation_quotients']) and arena_families <= set(roles['true_evaluation_families']),
            'New arena protection is incomplete')
    trajectories = roles['latest96_trajectories']
    require(len(trajectories) == 96, 'Latest trajectory count differs')
    for trace in trajectories:
        game = games[trace['index']]
        states = [row['state'] for row in observations if row['game'] == trace['index']] + [game['final']]
        projected_states = [dict(state_key=bytes(state_key(p)).hex(), quotient=pilot.quotient(p).hex()) for p in states]
        require(trace['family'] == game['family'] and trace['game_sha256'] == config.digest(game)
                and trace['states'] == projected_states,
                'Future-protection literal trajectory differs')
    from .ordinary_value_benchmark import allocation_from_report
    alternate = json.loads(config.canonical(prior))
    alternate['trajectory_replay_overlap']['contaminated_evaluation'] = False
    hypothetical = allocation_from_report(alternate, True)
    output = dict(schema=SCHEMA, passed=True, inputs=inputs, prior_inputs_preserved=frozen['inputs_count'],
                  summary=dict(admitted_rows=len(rows), drawn_unique_rows=len(drawn), current_optimizer_draws=sum(ledger.values()),
                               arena_games=len(games), arena_decisions=len(observations), guard_observations=len(observed),
                               admitted_quotient_match_observations=counts['admitted']['quotient'],
                               sampled_quotient_match_observations=counts['drawn']['quotient'],
                               raw_state_match_observations=counts['admitted']['raw_full_state'],
                               swapped_full_state_match_observations=counts['admitted']['swapped_full_state'],
                               family_match_observations=counts['admitted']['family'], original_allocation='hold',
                               hypothetical_draw_only_allocation=hypothetical['branch'], action_taken='none'),
                  match_counts={pool: {name: counts[pool][name] for name in ('raw_full_state', 'swapped_full_state', 'quotient', 'family')} for pool in pools},
                  by_active_band={pool: {key: dict(value) for key, value in sorted(bands[pool].items())} for pool in pools},
                  observation_bands=dict(Counter(band(row['state']) for row in observations)),
                  original_guard_reproduced=True, original_report_flag=original['contaminated_evaluation'],
                  collisions=collisions, ledger_sha256=config.digest(ledger),
                  draw_totals={name: sum(ledger.get(r['id'], 0) for r in values) for name, values in groups.items()},
                  ancestral_exposure=dict(source_control_admitted_roots=7669, source_control_unique_roots_drawn=7669,
                    source_control_total_draws=384000, already_sealed_schedule_reconstruction=True,
                    matching_row_positive_prior_draw_exposure=True, exact_old_per_row_counts_recomputed=False,
                    nonzero_per_row_gradient_or_game_causality_established=False,
                    evidence='build/sampling-review.json plus source-control admission membership'),
                  protected_roles=dict(pre_admission_quotients=len(registry['quotients']), pre_admission_families=len(registry['families']),
                                       admitted_preprotected_quotient_matches=0, admitted_preprotected_family_matches=0,
                                       latest_full_trajectories=len(trajectories), all_new_observed_quotients_protected=True,
                                       all_new_arena_families_protected=True, old_ledgers_changed=False),
                  hypothetical_rule_change=dict(executed=False, old_guard_preserved=True,
                    alternative='Ignore this admitted-but-undrawn/non-full-state match only for algebraic branch sensitivity.',
                    result=hypothetical, training_authorized=False, collection_authorized=False),
                  interpretation=['The original admitted-pool guard was applied correctly; no historical gate or report was changed.',
                    'This match is player-swapped/store-offset and encoder-identical, not identical full state or current sampled-row exposure.',
                    'The matching row belonged to source-control admission and all7669 admitted roots had positive draws in the sealed old schedule; zero current draws does not erase prior training exposure.',
                    'A hypothetical draw-only exception would still fail favorable validation/gain conditions and does not authorize a next run.'],
                  work=dict(new_training_updates=0, new_teacher_searches=0, new_PUCT_searches=0, new_games=0,
                            torch_imported=False, default_promotion=False, policy_changed=False, dataset_changed=False))
    current(inputs)
    return output


def validate_report(body):
    require(body == build(), 'Overlap review metrics, identities or inference scope differ')
    return body


def run(path=OUTPUT):
    return publish(path, build())


def verify(path=OUTPUT):
    return validate_report(read(path))
