"""Independent, literal verifier for Dakon opening witnesses; standard library only.

This module never imports the searcher, native kernel, or learned evaluators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_N = 64
MAX_BYTES = 16 * 1024 * 1024
RULES = {
    'profile': 'dakon-opening-relay-v1', 'initial_seeds': 'n_per_pit',
    'route': 'A_pits_A_store_B_pits_B_store', 'skip_opponent_store': True,
    'revisit_source': True, 'relay': 'occupied_before_last_deposit_either_row',
    'bonus': 'last_sown_seed_in_own_store',
    'capture': 'own_empty_with_nonempty_opposite_including_landing_seed',
    'empty_opposite': 'retain_landing_seed_end_turn',
    'empty_row': 'forced_pass', 'ending': 'both_rows_empty',
    'stores': 'permanent_single_round', 'opening': 'sequential',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Expected an integer in {low}..{high}')
    return value


def target(n, goal):
    if goal == 'majority':
        return n*n + 1
    if goal == 'refill_threshold':
        return 2*n*n - n + 1
    raise ValueError('Unknown winning objective')


def cells_valid(cells, n):
    if not isinstance(cells, (list, tuple)) or len(cells) != 2*n+2:
        raise ValueError('Invalid board dimensions')
    if any(type(x) is not int or not 0 <= x <= 2*n*n for x in cells):
        raise ValueError('Invalid seed count')
    if sum(cells) != 2*n*n:
        raise ValueError('Seed conservation failed')


def literal(cells, n, actor, action, poll=lambda: None):
    """Sow every individual seed; retain only aggregates, not a huge event list."""
    cells_valid(cells, n)
    integer(actor, 0, 1); integer(action, 0, n-1)
    work = list(cells)
    source = actor*(n+1)+action
    if not work[source]:
        raise ValueError('Selected an empty pit')
    own = actor*(n+1)+n
    forbidden = (1-actor)*(n+1)+n
    cursor, hand = source, work[source]
    work[source] = 0
    deposits = relays = captured = 0
    while True:
        poll()
        while hand:
            cursor = (cursor+1) % len(work)
            if cursor == forbidden:
                cursor = (cursor+1) % len(work)
            before = work[cursor]
            work[cursor] += 1
            hand -= 1
            deposits += 1
            if deposits % 1024 == 0:
                poll()
        if cursor == own:
            reason = 'own_store'
            break
        if before:
            hand = work[cursor]
            work[cursor] = 0
            relays += 1
            continue
        reason = 'opponent_empty'
        if actor*(n+1) <= cursor < actor*(n+1)+n:
            opposite = 2*n-cursor
            reason = 'own_empty'
            if work[opposite]:
                captured = work[cursor]+work[opposite]
                work[own] += captured
                work[cursor] = work[opposite] = 0
                reason = 'own_capture'
        break
    occupied = [any(work[p*(n+1):p*(n+1)+n]) for p in (0, 1)]
    terminal = not any(occupied)
    nominal = actor if reason == 'own_store' else 1-actor
    controller = actor if terminal else nominal
    passes = []
    if not terminal and not occupied[controller]:
        passes.append([controller, 1-controller]); controller = 1-controller
    cells_valid(work, n)
    return dict(cells=work, controller=controller, reason=reason, terminal=terminal,
                passes=passes, deposits=deposits, relays=relays, captured=captured)


def verify(value, poll=lambda: None):
    if not isinstance(value, dict) or set(value) != {'body', 'sha256'}:
        raise ValueError('Invalid certificate envelope')
    body = value['body']
    if not isinstance(body, dict) or digest(body) != value['sha256']:
        raise ValueError('Certificate checksum mismatch')
    if set(body) != {'schema', 'recipe', 'actions', 'steps', 'proof'}:
        raise ValueError('Invalid certificate fields')
    if body['schema'] != 'dakon-opening-certificate-v1':
        raise ValueError('Unknown certificate version')
    recipe = body['recipe']
    if not isinstance(recipe, dict) or set(recipe) != {'schema', 'n', 'goal', 'starter', 'cache_entries', 'rules', 'sources', 'ordering'}:
        raise ValueError('Invalid recipe')
    n = integer(recipe['n'], 1, MAX_N)
    first = integer(recipe['starter'], 0, 1)
    integer(recipe['cache_entries'], 0, 10000)
    if recipe['schema'] != 'dakon-opening-run-v1' or canonical(recipe['rules']) != canonical(RULES) or recipe['ordering'] != 'canonical_pit_ascending':
        raise ValueError('Unsupported rules or search identity')
    sources = recipe['sources']
    if not isinstance(sources, dict) or set(sources) != {'searcher', 'verifier'} or any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in sources.values()):
        raise ValueError('Invalid source identities')
    required = target(n, recipe['goal'])
    actions, steps = body['actions'], body['steps']
    if not isinstance(actions, list) or not 1 <= len(actions) <= 2*n*n+1 or not isinstance(steps, list) or len(steps) != len(actions):
        raise ValueError('Invalid action sequence')
    cells = [n]*n+[0]+[n]*n+[0]
    own, other = first*(n+1)+n, (1-first)*(n+1)+n
    for index, (action, saved) in enumerate(zip(actions, steps)):
        if cells[own] >= required:
            raise ValueError('Sequence continued after its winning target')
        actual = literal(cells, n, first, action, poll)
        if canonical(actual) != canonical(saved):
            raise ValueError(f'Independent replay differs at choice {index+1}')
        if index+1 < len(actions) and (actual['reason'] != 'own_store' or actual['passes'] or actual['terminal'] or actual['controller'] != first):
            raise ValueError('Opponent receives a choice before the target')
        cells = actual['cells']
        if cells[other] != 0:
            raise ValueError('Opponent store changed during the first turn')
    if cells[own] < required:
        raise ValueError('Winning threshold not reached')
    proof = dict(total=2*n*n, required=required, starter_store=cells[own],
                 opponent_max_total=2*n*n-cells[own], opponent_choices_before_goal=0,
                 shortest_proved=False)
    if canonical(body['proof']) != canonical(proof):
        raise ValueError('Incorrect proof summary')
    return dict(passed=True, n=n, goal=recipe['goal'], starter=first,
                choices=len(actions), **proof,
                line_right_store=[a+1 for a in actions], line_left_store=[n-a for a in actions])


def read_certificate(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Certificate exceeds verification size limit')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', type=Path)
    args = parser.parse_args(argv)
    try:
        print(canonical(verify(read_certificate(args.certificate))))
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        parser.exit(2, f'Certificate rejected: {error}\n')


if __name__ == '__main__':
    main()
