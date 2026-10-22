"""Independent, seed-by-seed interpreter for acceptance tests.

No production modules are imported.  The only shared contract is the documented
ring encoding: A pits, A store, B pits, B store.  Values always favour player A.
"""

from dataclasses import dataclass


class WorkGuard(RuntimeError):
    """A test exhausted its work allowance; it must fail, never return a draw."""


@dataclass(frozen=True)
class Position:
    pits: int
    counts: tuple[int, ...]
    controller: int

    def __post_init__(self):
        if self.pits < 1 or len(self.counts) != 2 * self.pits + 2:
            raise ValueError("invalid ring dimensions")
        if self.controller not in (0, 1) or any(n < 0 for n in self.counts):
            raise ValueError("invalid position")


@dataclass(frozen=True)
class Normalized:
    controller: int
    terminal: bool
    passes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Transition:
    before: Position
    position: Position
    actor: int
    nominal_controller: int
    terminal: bool
    events: tuple[dict, ...]
    initial_passes: tuple[tuple[int, int], ...]
    final_passes: tuple[tuple[int, int], ...]
    deposits: int
    relay_pickups: int
    captured: int
    bonus: bool


def row_indices(pits, player):
    start = 0 if player == 0 else pits + 1
    return range(start, start + pits)


def store_index(pits, player):
    return pits if player == 0 else 2 * pits + 1


def normalize(position):
    occupied = [any(position.counts[i] for i in row_indices(position.pits, a))
                for a in (0, 1)]
    if not any(occupied):
        return Normalized(position.controller, True, ())
    if occupied[position.controller]:
        return Normalized(position.controller, False, ())
    other = 1 - position.controller
    return Normalized(other, False, ((position.controller, other),))


def legal_actions(position):
    normal = normalize(position)
    if normal.terminal:
        return ()
    return tuple(i for i in range(position.pits)
                 if position.counts[i + (position.pits + 1) * normal.controller])


def opposite_index(pits, index):
    if index < pits:
        return 2 * pits - index
    if pits < index < 2 * pits + 1:
        return 2 * pits - index
    raise ValueError("stores have no opposite")


def apply(position, action, *, deposit_guard=1_000_000):
    """Execute one choice literally one seed at a time into private scratch."""
    normal = normalize(position)
    if action not in legal_actions(position):
        raise ValueError("illegal action")
    pits, actor = position.pits, normal.controller
    own_store = store_index(pits, actor)
    forbidden_store = store_index(pits, 1 - actor)
    counts = list(position.counts)
    index = action + (pits + 1) * actor
    hand = counts[index]
    counts[index] = 0
    events = [{"kind": "pass", "from": a, "to": b}
              for a, b in normal.passes]
    events.append({"kind": "pickup", "index": index, "seeds": hand})
    deposits = relay_pickups = captured = 0
    bonus = False
    while True:
        occupied_before_last = None
        while hand:
            index = (index + 1) % len(counts)
            if index == forbidden_store:
                events.append({"kind": "skip", "index": index})
                index = (index + 1) % len(counts)
            occupied_before_last = counts[index]
            counts[index] += 1
            hand -= 1
            deposits += 1
            if deposits > deposit_guard:
                raise WorkGuard("reference sowing exceeded deposit guard")
            events.append({"kind": "deposit", "index": index,
                           "before": occupied_before_last, "after": counts[index]})
        if index == own_store:
            bonus = True
            nominal = actor
            events.append({"kind": "bonus", "actor": actor})
            break
        if occupied_before_last:
            hand = counts[index]
            counts[index] = 0
            relay_pickups += 1
            events.append({"kind": "relay", "index": index, "seeds": hand})
            continue
        if index in row_indices(pits, actor):
            opposite = opposite_index(pits, index)
            if counts[opposite]:
                captured = counts[index] + counts[opposite]
                events.append({"kind": "capture", "index": index,
                               "opposite": opposite, "seeds": captured})
                counts[own_store] += captured
                counts[index] = counts[opposite] = 0
        nominal = 1 - actor
        break
    # The rules check terminal before assigning nominal next control.  A
    # terminal encoding retains the mover, although it has no next legal move.
    rows_empty = not any(counts[i] for player in (0, 1)
                         for i in row_indices(pits, player))
    result = Position(pits, tuple(counts), actor if rows_empty else nominal)
    after_normal = normalize(result)
    events.extend({"kind": "pass", "from": a, "to": b}
                  for a, b in after_normal.passes)
    if after_normal.terminal:
        events.append({"kind": "terminal"})
    result = Position(pits, tuple(counts), after_normal.controller)
    if sum(result.counts) != sum(position.counts):
        raise AssertionError("reference violated seed conservation")
    return Transition(position, result, actor, nominal, after_normal.terminal,
                      tuple(events), normal.passes, after_normal.passes,
                      deposits, relay_pickups, captured, bonus)


def evaluate(position):
    return (position.counts[store_index(position.pits, 0)]
            - position.counts[store_index(position.pits, 1)])


def minimax(position, depth):
    """Full-width minimax with no alpha-beta, TT, oracle or physical pruning."""
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if normalize(position).terminal or depth == 0:
        return evaluate(position)
    scores = [minimax(apply(position, move).position, depth - 1)
              for move in legal_actions(position)]
    return (max if normalize(position).controller == 0 else min)(scores)


def root_scores(position, depth):
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if depth == 0 or normalize(position).terminal:
        return {}
    return {move: minimax(apply(position, move).position, depth - 1)
            for move in legal_actions(position)}


def optimal_actions(position, depth):
    scores = root_scores(position, depth)
    if not scores:
        return ()
    optimum = (max if normalize(position).controller == 0 else min)(scores.values())
    return tuple(move for move, value in scores.items() if value == optimum)


def compositions(total, slots):
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, slots - 1):
            yield (first,) + tail


def replay_trace(transition):
    """Reconstruct the final ring using only the emitted elementary facts."""
    counts = list(transition.before.counts)
    hand = 0
    for event in transition.events:
        kind = event["kind"]
        if kind in ("pickup", "relay"):
            if hand or counts[event["index"]] != event["seeds"]:
                raise AssertionError("invalid pickup fact")
            hand = counts[event["index"]]
            counts[event["index"]] = 0
        elif kind == "deposit":
            if not hand or counts[event["index"]] != event["before"]:
                raise AssertionError("invalid deposit fact")
            counts[event["index"]] += 1
            hand -= 1
            if counts[event["index"]] != event["after"]:
                raise AssertionError("invalid deposit result")
        elif kind == "capture":
            if hand or counts[event["index"]] + counts[event["opposite"]] != event["seeds"]:
                raise AssertionError("invalid capture fact")
            counts[store_index(transition.before.pits, transition.actor)] += event["seeds"]
            counts[event["index"]] = counts[event["opposite"]] = 0
    if hand:
        raise AssertionError("trace left seeds in hand")
    return tuple(counts)
