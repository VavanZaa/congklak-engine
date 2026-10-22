"""Acceptance corpus from an interpreter independent of the C++ kernel.

Run with ``python -m unittest discover -s tests -p test_native_acceptance.py -v``.
Missing native binaries are a failure, not a skipped acceptance case.
"""

import hashlib
import json
from pathlib import Path
import random
import threading
import unittest

from congklak.native import Control, Native, NativeError
from congklak.explanations import explain
import reference as ref


def flatten(position):
    return (tuple(position["pits"][0]) + (position["stores"][0],)
            + tuple(position["pits"][1]) + (position["stores"][1],))


def fixture(cells, actor=0):
    return ref.Position(7, tuple(cells), actor)


def ring(**counts):
    cells = [0] * 16
    for name, count in counts.items():
        slot = {"SA": 7, "SB": 15}.get(name)
        if slot is None:
            slot = int(name[1:]) + (0 if name[0] == "A" else 8)
        cells[slot] = count
    return cells


def exchange_players(cells):
    return tuple(cells[8:16]) + tuple(cells[:8])


class NativeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()
        cls.transitions = cls.horizons = cls.histories = 0
        cls.mid_sow_cancel_deposits = None

    @classmethod
    def tearDownClass(cls):
        print(f"ACCEPTANCE_COUNTS transitions={cls.transitions} "
              f"horizons={cls.horizons} histories={cls.histories} "
              f"mid_sow_cancel_deposits={cls.mid_sow_cancel_deposits}")

    def normalize(self, position):
        actual = self.native.normalize(position.counts, position.pits,
                                       position.controller)
        expected = ref.normalize(position)
        self.assertTrue(actual["ok"], (position, actual))
        self.assertEqual(flatten(actual["position"]), position.counts)
        self.assertEqual(actual["position"]["controller"], expected.controller)
        self.assertEqual(actual["terminal"], expected.terminal)
        self.assertEqual([(p["from"], p["to"]) for p in actual["passes"]],
                         list(expected.passes))
        self.assertEqual(tuple(actual["legal_actions"]), ref.legal_actions(position))
        return expected

    def check_transition(self, position, move, *, trace=False):
        normalized = self.normalize(position)
        before = ref.Position(position.pits, position.counts, normalized.controller)
        expected = ref.apply(before, move)
        actual = self.native.apply(before.counts, before.pits, before.controller,
                                   move, trace=trace)
        context = (position, move, actual)
        self.assertTrue(actual["ok"], context)
        self.assertEqual(flatten(actual["position"]), expected.position.counts, context)
        self.assertEqual(actual["position"]["controller"],
                         expected.position.controller, context)
        self.assertEqual(actual["actor"], expected.actor, context)
        self.assertEqual(actual["pit"], move, context)
        self.assertEqual(actual["terminal"], expected.terminal, context)
        self.assertEqual(actual["drops"], expected.deposits, context)
        self.assertEqual(actual["relays"], expected.relay_pickups, context)
        self.assertEqual([(p["from"], p["to"]) for p in actual["passes"]],
                         list(expected.final_passes), context)
        expected_delta = [expected.position.counts[ref.store_index(before.pits, a)]
                          - before.counts[ref.store_index(before.pits, a)]
                          for a in (0, 1)]
        self.assertEqual(actual["store_delta"], expected_delta, context)
        expected_reason = ("own_store" if expected.bonus else
                           "own_capture" if expected.captured else
                           "own_empty" if self.landing(expected) in
                           ref.row_indices(position.pits, expected.actor) else
                           "opponent_empty")
        self.assertEqual(actual["reason"], expected_reason, context)
        self.assertEqual(sum(flatten(actual["position"])), sum(before.counts), context)
        self.assertEqual(ref.replay_trace(expected), expected.position.counts)
        if trace:
            self.replay_native_trace(before, actual)
        type(self).transitions += 1
        return expected, actual

    @staticmethod
    def landing(transition):
        return next(event["index"] for event in reversed(transition.events)
                    if event["kind"] == "deposit")

    def replay_native_trace(self, before, transition):
        cells = list(before.counts)
        hand = 0
        cursor = None
        deposits = relays = 0
        captures = []
        bonuses = passes = terminals = 0
        actor = transition["actor"]
        own_store = ref.store_index(before.pits, actor)
        forbidden_store = ref.store_index(before.pits, 1 - actor)
        for event in transition["events"]:
            kind = event["type"]
            if kind in ("pickup", "relay"):
                slot = event["slot"]
                self.assertEqual(hand, 0)
                self.assertEqual(event["count"], cells[slot])
                self.assertEqual(event["before"], cells[slot])
                self.assertEqual(event["after"], 0)
                self.assertEqual(event["hand"], cells[slot])
                if kind == "pickup":
                    self.assertIn(slot, ref.row_indices(before.pits, actor))
                else:
                    self.assertGreater(cells[slot], 1)
                    self.assertEqual(slot, cursor)
                    relays += 1
                hand = cells[slot]
                cells[slot] = 0
                cursor = slot
            elif kind == "deposit":
                self.assertGreater(hand, 0)
                cursor = (cursor + 1) % len(cells)
                if cursor == forbidden_store:
                    cursor = (cursor + 1) % len(cells)
                self.assertEqual(event["slot"], cursor)
                self.assertEqual(event["count"], 1)
                self.assertEqual(event["before"], cells[cursor])
                cells[cursor] += 1
                hand -= 1
                self.assertEqual(event["after"], cells[cursor])
                self.assertEqual(event["hand"], hand)
                deposits += 1
            elif kind == "capture":
                slot, opposite = event["slot"], event["opposite"]
                self.assertEqual(hand, 0)
                self.assertEqual(slot, cursor)
                self.assertIn(slot, ref.row_indices(before.pits, actor))
                self.assertEqual(opposite, ref.opposite_index(before.pits, slot))
                self.assertGreater(cells[opposite], 0)
                self.assertEqual(event["count"], cells[slot] + cells[opposite])
                self.assertEqual(event["before"], cells[own_store])
                cells[own_store] += cells[slot] + cells[opposite]
                cells[slot] = cells[opposite] = 0
                self.assertEqual(event["after"], cells[own_store])
                captures.append(event["count"])
            elif kind == "bonus":
                self.assertEqual(cursor, own_store)
                self.assertEqual(event["actor"], actor)
                bonuses += 1
            elif kind == "pass":
                self.assertFalse(any(cells[i] for i in
                                     ref.row_indices(before.pits, event["actor"])))
                self.assertTrue(any(cells[i] for i in
                                    ref.row_indices(before.pits, event["to"])))
                passes += 1
            elif kind == "terminal":
                self.assertFalse(any(cells[i] for a in (0, 1)
                                     for i in ref.row_indices(before.pits, a)))
                terminals += 1
            elif kind == "turn_end":
                self.assertEqual(hand, 0)
            else:
                self.fail(f"Unknown native event: {event}")
        self.assertEqual(hand, 0)
        self.assertEqual(tuple(cells), flatten(transition["position"]))
        self.assertEqual(deposits, transition["drops"])
        self.assertEqual(relays, transition["relays"])
        self.assertEqual(bonuses, int(transition["reason"] == "own_store"
                                     and not transition["terminal"]))
        self.assertEqual(bool(captures), transition["reason"] == "own_capture")
        self.assertEqual(passes, len(transition["passes"]))
        self.assertEqual(terminals, int(transition["terminal"]))

    def check_report(self, position, depth, report):
        normal = ref.normalize(position)
        context = (position, depth, report)
        self.assertEqual(report["input_controller"], position.controller, context)
        self.assertEqual(report["normalized_controller"],
                         None if normal.terminal else normal.controller, context)
        score_player = position.controller if normal.terminal else normal.controller
        self.assertEqual(report["score_player"], score_player, context)
        value = ref.minimax(position, depth)
        result = report["depth_result"]
        self.assertEqual(result["value_player0"], value, context)
        self.assertEqual(result["value_actor"], value if score_player == 0 else -value,
                         context)
        self.assertEqual(result["evaluator"], "H_v1", context)
        self.assertEqual(result["scope"], "finite_choice_horizon", context)
        self.assertEqual(result["bound"], "exact", context)
        self.assertEqual(result["choice_depth"], 0 if normal.terminal else depth, context)
        self.assertEqual(report["proof_status"],
                         "terminal_outcome" if normal.terminal else "finite_horizon_result")
        active = sum(position.counts) - position.counts[position.pits] - position.counts[-1]
        lower, upper = ref.evaluate(position) - active, ref.evaluate(position) + active
        interval = report["physical_interval"]
        self.assertEqual((interval["lower_player0"], interval["upper_player0"]),
                         (lower, upper))
        self.assertEqual((interval["lower"], interval["upper"]),
                         (lower, upper) if score_player == 0 else (-upper, -lower))
        self.assertFalse(interval["used_for_pruning"])
        self.assertEqual(tuple(report["legal_actions"]), ref.legal_actions(position))
        self.assertEqual([(p["from"], p["to"]) for p in report["forced_passes"]],
                         list(normal.passes))
        expected_scores = ref.root_scores(position, depth)
        self.assertEqual(tuple(report["best_moves"]), ref.optimal_actions(position, depth),
                         context)
        self.assertEqual([c["pit"] for c in report["candidates"]],
                         list(ref.legal_actions(position)), context)
        for candidate in report["candidates"]:
            self.assertEqual(candidate["actor"], normal.controller, context)
            if depth == 0:
                self.assertIsNone(candidate["depth_result"], context)
                continue
            score = candidate["depth_result"]
            self.assertEqual(score["value_player0"], expected_scores[candidate["pit"]],
                             context)
            self.assertEqual(score["value_actor"],
                             score["value_player0"] if score_player == 0
                             else -score["value_player0"], context)
            self.check_pv(position, depth, candidate["pv"],
                          candidate["pv_truncated"], candidate["pit"],
                          expected_value=score["value_player0"])
            if depth == 1:
                self.replay_native_trace(position, candidate["transition"])
        self.check_pv(position, depth, report["pv"], report["pv_truncated"],
                      expected_value=value)
        type(self).horizons += 1

    def check_pv(self, position, depth, pv, truncated, first_move=None,
                 expected_value=None):
        if first_move is not None:
            self.assertTrue(pv)
            self.assertEqual(pv[0]["pit"], first_move)
        current = position
        for index, choice in enumerate(pv):
            self.assertEqual(choice["actor"], ref.normalize(current).controller)
            self.assertIn(choice["pit"], ref.legal_actions(current))
            if index or first_move is None:
                self.assertEqual(choice["pit"],
                                 ref.optimal_actions(current, depth - index)[0])
            transition = ref.apply(current, choice["pit"])
            self.assertEqual(choice["next_controller"],
                             None if transition.terminal else transition.position.controller)
            self.assertEqual(choice["bonus_choice"],
                             transition.bonus and not transition.terminal)
            self.assertEqual(choice["terminal_after"], transition.terminal)
            self.assertEqual([(p["from"], p["to"])
                              for p in choice["forced_passes_after"]],
                             list(transition.final_passes))
            current = transition.position
        self.assertLessEqual(len(pv), depth)
        ended = ref.normalize(current).terminal
        self.assertEqual(truncated, not ended)
        if not ended and depth:
            self.assertEqual(len(pv), depth)
        if expected_value is not None:
            self.assertEqual(ref.evaluate(current), expected_value)

    def test_01_declared_golden_fixtures_and_traces(self):
        fixtures = [
            ("store_bonus", ring(A6=1, SA=97), 6,
             ring(SA=98), "own_store"),
            ("own_capture", ring(A0=1, B5=3, SA=94), 0,
             ring(SA=98), "own_capture"),
            ("empty_opposite_and_forced_pass", ring(A0=1, SA=97), 0,
             ring(A1=1, SA=97), "own_empty"),
            ("opponent_empty_and_revival", ring(A6=2, SA=96), 6,
             ring(B0=1, SA=97), "opponent_empty"),
            ("own_row_relay", ring(A0=1, A1=1, SA=96), 0,
             ring(A2=1, A3=1, SA=96), "own_empty"),
            ("opponent_row_relay", ring(A6=2, B0=1, SA=95), 6,
             ring(B1=1, B2=1, SA=96), "opponent_empty"),
            ("source_revisit_and_store_skip", ring(A0=15, SA=83), 0,
             ring(A1=1, A2=1, A3=1, A4=1, A5=1, A6=1,
                  B0=1, B1=1, B2=1, B3=1, B4=1, B5=1, SA=86),
             "own_capture"),
        ]
        for name, initial, move, expected_ring, reason in fixtures:
            for actor in (0, 1):
                with self.subTest(fixture=name, actor=actor):
                    counts = initial if actor == 0 else exchange_players(initial)
                    expected_counts = (expected_ring if actor == 0
                                       else exchange_players(expected_ring))
                    transition, native = self.check_transition(fixture(counts, actor),
                                                               move, trace=True)
                    self.assertEqual(transition.position.counts, tuple(expected_counts))
                    self.assertEqual(native["reason"], reason)
        tie = ring(SA=49, SB=49)
        for actor in (0, 1):
            position = fixture(tie, actor)
            self.normalize(position)
            report = self.native.analyze(position.counts, 7, actor, max_depth=4)["report"]
            self.check_report(position, 4, report)
            self.assertEqual(report["depth_result"]["value_player0"], 0)

    def test_02_exhaustive_reduced_transitions_and_horizons(self):
        states = 0
        initial_transitions = type(self).transitions
        for pits in range(1, 4):
            for total in range(7):
                for cells in ref.compositions(total, 2 * pits + 2):
                    for actor in (0, 1):
                        position = ref.Position(pits, cells, actor)
                        self.normalize(position)
                        for move in ref.legal_actions(position):
                            self.check_transition(position, move)
                        zero = self.native.analyze(cells, pits, actor, max_depth=0)["report"]
                        self.check_report(position, 0, zero)
                        reports = {}
                        result = self.native.analyze(
                            cells, pits, actor, max_depth=4,
                            on_report=lambda report: reports.update(
                                {report["choice_depth"]: report}))
                        if ref.normalize(position).terminal:
                            for depth in range(1, 5):
                                self.check_report(position, depth, result["report"])
                        else:
                            reports[result["report"]["choice_depth"]] = result["report"]
                            self.assertEqual(set(reports) - {0}, {1, 2, 3, 4},
                                             (position, result, reports))
                            for depth in range(1, 5):
                                self.check_report(position, depth, reports[depth])
                        states += 1
        self.assertEqual(states, 8274)
        self.assertEqual(type(self).transitions - initial_transitions, 11950)

    def test_03_1024_seeded_product_histories(self):
        corpus = hashlib.sha256()
        choices = 0
        for seed in range(1024):
            rng = random.Random(seed)
            position = fixture([7] * 7 + [0] + [7] * 7 + [0])
            record = {"seed": seed, "states": [list(position.counts)],
                      "controllers": [position.controller], "actions": []}
            for step in range(rng.randrange(1, 65)):
                moves = ref.legal_actions(position)
                if not moves:
                    break
                move = rng.choice(moves)
                transition, _ = self.check_transition(position, move, trace=(step == 0))
                position = transition.position
                choices += 1
                record["actions"].append(move)
                record["states"].append(list(position.counts))
                record["controllers"].append(position.controller)
            self.normalize(position)
            corpus.update(json.dumps(record, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))
            corpus.update(b"\n")
            type(self).histories += 1
        self.assertEqual(choices, 29249)
        self.assertEqual(corpus.hexdigest(),
                         "ab646e915eaa173d84829b6ccf261a0b51629ea6553bae12de41cd9cb9fdbf72")

    def test_04_curated_search_actor_bonus_and_long_relay(self):
        samples = [
            fixture([7] * 7 + [0] + [7] * 7 + [0]),
            fixture(ring(A0=3, A5=2, A6=1, B0=4, B6=1, SA=12, SB=75)),
            fixture(ring(B0=1, B6=2, SA=95), 0),
            fixture(ring(A5=2, A6=1, B4=1, SA=0, SB=94), 0),
            fixture(ring(A0=1, A1=1, B0=1, B1=1, SA=47, SB=47), 1),
        ]
        for position in samples:
            for depth in range(1, 5):
                result = self.native.analyze(position.counts, 7, position.controller,
                                             max_depth=depth)
                self.assertIsNotNone(result["report"], result)
                self.check_report(position, depth, result["report"])
            for move in ref.legal_actions(position):
                self.check_transition(position, move, trace=True)

    def test_05_manual_fixture_corpus(self):
        path = Path(__file__).resolve().parent.parent / "rules" / "golden-fixtures.json"
        fixtures = json.loads(path.read_text(encoding="utf-8"))["fixtures"]

        def subset(expected, actual, context):
            if isinstance(expected, dict):
                for key, value in expected.items():
                    self.assertIn(key, actual, context)
                    subset(value, actual[key], context)
            elif isinstance(expected, list):
                self.assertEqual(len(expected), len(actual), context)
                for left, right in zip(expected, actual):
                    subset(left, right, context)
            else:
                self.assertEqual(expected, actual, context)

        for case in fixtures:
            cells = flatten(case["position"])
            actor = case["position"]["controller"]
            if case["operation"] == "apply":
                result = self.native.apply(cells, 7, actor, case["action"], trace=True)
            else:
                result = getattr(self.native, case["operation"])(cells, 7, actor)
            subset(case["expected"], result, case["id"])
        self.assertEqual(len(fixtures), 25)

    def test_06_limits_and_atomic_iteration_publication(self):
        position = fixture([7] * 7 + [0] + [7] * 7 + [0])
        with Control(self.native, max_depth=4, time_ms=0, node_budget=0) as control:
            control.cancel()
            result = self.native.apply(position.counts, 7, 0, 0, control=control)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "cancelled")
            self.assertNotIn("position", result)
            self.assertNotIn("events", result)
            result = self.native.analyze(position.counts, 7, 0, max_depth=4,
                                         control=control)
            self.assertEqual(result["stop_reason"], "cancelled")
            self.assertIsNone(result["report"])
            self.assertEqual(result["last_completed_depth"], -1)
            self.assertTrue(all(c["depth_result"] is None
                                for c in result["initial"]["candidates"]))
        result = self.native.analyze(position.counts, 7, 0, max_depth=4, node_budget=1)
        self.assertEqual(result["stop_reason"], "node_budget")
        self.assertIsNone(result["report"])
        self.assertLessEqual(result["nodes"], 1)
        result = self.native.analyze(position.counts, 7, 0, max_depth=4,
                                     on_report=lambda report: report["choice_depth"] < 2)
        self.assertEqual(result["stop_reason"], "callback_failed")
        self.assertEqual(result["last_completed_depth"], 1)
        self.check_report(position, 1, result["report"])
        with Control(self.native, max_depth=4, time_ms=0, node_budget=0) as control:
            def cancel_after_one(report):
                if report["choice_depth"] == 1:
                    control.cancel()
            result = self.native.analyze(position.counts, 7, 0, max_depth=4,
                                         control=control, on_report=cancel_after_one)
        self.assertEqual(result["stop_reason"], "cancelled")
        self.assertEqual(result["last_completed_depth"], 1)
        self.check_report(position, 1, result["report"])
        result = self.native.analyze(position.counts, 7, 0, max_depth=128,
                                     time_ms=1)
        self.assertEqual(result["stop_reason"], "time_budget")
        if result["report"] is not None:
            self.check_report(position, result["last_completed_depth"], result["report"])
        result = self.native.analyze(position.counts, 7, 0, max_depth=4,
                                     memory_mb=16)
        self.assertEqual(result["stop_reason"], "memory_budget")
        self.assertIsNone(result["report"])

    def test_07_cancellation_during_sowing_discards_scratch(self):
        # This measured long-relay position has 489 elementary deposits.  A
        # fresh token's diagnostic must show positive work on a failed apply,
        # so cancellation before the first poll cannot satisfy this test.
        position = fixture([10, 5, 9, 6, 3, 9, 5, 0, 9, 3, 7, 6, 8, 8, 10, 0])
        expected = ref.apply(position, 0)
        self.assertEqual(expected.deposits, 489)
        observed = False
        attempts = 0
        for attempts in range(1, 257):
            with Control(self.native, max_depth=4, time_ms=0, node_budget=0) as control:
                ready = threading.Event()
                def cancel_in_other_thread():
                    ready.wait()
                    control.cancel()
                canceller = threading.Thread(target=cancel_in_other_thread)
                canceller.start()
                ready.set()
                actual = self.native.apply(position.counts, 7, 0, 0,
                                           trace=True, control=control)
                canceller.join()
            if actual["ok"]:
                self.assertEqual(flatten(actual["position"]), expected.position.counts)
            else:
                self.assertEqual(actual["error"]["code"], "cancelled")
                self.assertNotIn("position", actual)
                self.assertNotIn("events", actual)
                if 0 < actual.get("deposits", 0) < expected.deposits:
                    type(self).mid_sow_cancel_deposits = actual["deposits"]
                    observed = True
                    break
        self.assertTrue(observed, f"No verified mid-sowing cancellation in {attempts} attempts")
        self.assertEqual(position.counts,
                         (10, 5, 9, 6, 3, 9, 5, 0, 9, 3, 7, 6, 8, 8, 10, 0))
        self.check_transition(position, 0, trace=True)

    def test_08_control_lifetime_and_instance_ownership(self):
        position = fixture([7] * 7 + [0] + [7] * 7 + [0])
        control = Control(self.native, max_depth=1, time_ms=0, node_budget=0)
        try:
            def while_running(report):
                with self.assertRaises(NativeError):
                    control.close()
                with self.assertRaises(NativeError):
                    self.native.apply(position.counts, 7, 0, 0, control=control)
                with self.assertRaises(NativeError):
                    self.native.analyze(position.counts, 7, 0, control=control)
                return True
            result = self.native.analyze(position.counts, 7, 0, max_depth=1,
                                         control=control, on_report=while_running)
            self.assertTrue(result["ok"])
            self.assertNotIn("callback_error", result)
        finally:
            control.close()
        with self.assertRaises(NativeError):
            self.native.apply(position.counts, 7, 0, 0, control=control)
        with self.assertRaises(NativeError):
            self.native.analyze(position.counts, 7, 0, control=control)
        foreign = Native()
        with Control(foreign, max_depth=1, time_ms=0, node_budget=0) as foreign_control:
            with self.assertRaises(NativeError):
                self.native.apply(position.counts, 7, 0, 0, control=foreign_control)
            with self.assertRaises(NativeError):
                self.native.analyze(position.counts, 7, 0, control=foreign_control)

    def test_09_explanations_follow_replayed_actor_and_facts(self):
        positions = [
            fixture(ring(A5=1, A6=1, SA=96)),
            fixture(ring(A6=1, B0=1, SA=96)),
            fixture(ring(A0=1, B5=3, SA=94)),
            fixture(ring(A0=1, SA=97)),
            fixture(ring(B6=1, SA=49, SB=48), 0),
        ]
        for position in positions:
            report = self.native.analyze(position.counts, 7, position.controller,
                                         max_depth=1)["report"]
            text = explain(report)
            actor = ref.normalize(position).controller
            self.assertIn("finite-horizon recommendation", text["summary"])
            for candidate in text["candidates"]:
                expected = ref.apply(position, candidate["pit"])
                player = "A" if actor == 0 else "B"
                self.assertEqual(candidate["actor"], actor)
                self.assertEqual(candidate["move"], f"{player}{candidate['pit']}")
                delta = (expected.position.counts[ref.store_index(7, actor)]
                         - position.counts[ref.store_index(7, actor)])
                self.assertIn(f"Adds {delta} seeds to {player}'s store", candidate["immediate"])
                if expected.captured:
                    self.assertIn(f"Captures {expected.captured} seeds", candidate["immediate"])
                if expected.bonus and not expected.terminal and not expected.final_passes:
                    self.assertIn("grants another choice", candidate["immediate"])
                elif expected.bonus and expected.final_passes:
                    self.assertNotIn("grants another choice", candidate["immediate"])
                if expected.terminal:
                    self.assertIn("the game ends", candidate["immediate"])
                    self.assertIn("reaches a terminal position", candidate["conditional_continuation"])
                else:
                    self.assertIn("later choices are unsearched", candidate["conditional_continuation"])
                for passed, receiver in expected.final_passes:
                    source = "A" if passed == 0 else "B"
                    target = "A" if receiver == 0 else "B"
                    self.assertIn(f"{source} has an empty row and must pass to {target}",
                                  candidate["immediate"])
                if ref.normalize(position).passes:
                    self.assertIn("A must pass to B", text["summary"])
        terminal = self.native.analyze(ring(SA=49, SB=49), 7, 1, max_depth=4)["report"]
        self.assertIn("B draws", explain(terminal)["summary"])
        self.assertEqual(explain(terminal)["candidates"], [])
        zero = self.native.analyze([7] * 7 + [0] + [7] * 7 + [0], 7, 0,
                                   max_depth=0)["report"]
        self.assertIn("no moves have been ranked", explain(zero)["summary"])
        self.assertIn("No complete ranking", explain(None)["summary"])


if __name__ == "__main__":
    unittest.main()
