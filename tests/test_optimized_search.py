"""Independent finite-horizon parity and cache safety checks for TT/PVS."""
from __future__ import annotations

import ctypes as C
import random
import unittest

from congklak.native import Control, Native
from tests import reference as ref


class OptimizedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()
        cls.native.lib.cc_set_search_options.argtypes = [C.c_void_p, C.c_int, C.c_int]
        cls.native.lib.cc_set_search_options.restype = C.c_char_p
        cls.state_horizon_ablation_cases = 0
        cls.exhaustive_optimized_cases = 0

    def search(self, position, depth, tt=1, pvs=True, start_depth=1, on_report=None,
               node_budget=0):
        with Control(self.native, max_depth=depth, time_ms=0,
                     node_budget=node_budget, memory_mb=512) as control:
            options = self.native._decode(self.native.lib.cc_set_search_options(
                control.handle, tt, int(pvs)))
            self.assertTrue(options["ok"], options)
            return self.native.analyze(position.counts, position.pits, position.controller,
                                       max_depth=depth, start_depth=start_depth,
                                       control=control, on_report=on_report)

    def assert_report(self, position, report):
        depth = report["choice_depth"]
        scores = ref.root_scores(position, depth)
        actual = {c["pit"]: c["depth_result"]["value_player0"]
                  for c in report["candidates"] if c["analysis_complete"]}
        self.assertEqual(actual, scores, (position, depth))
        self.assertEqual(report["best_moves"], list(ref.optimal_actions(position, depth)))
        self.assertEqual(report["depth_result"]["value_player0"], ref.minimax(position, depth))
        self.assertEqual(report["policy"], "terminal-first/no-oracle/full-root-window-v2")
        for candidate in report["candidates"]:
            if not candidate["analysis_complete"]:
                continue
            current = position
            for index, move in enumerate(candidate["pv"]):
                normal = ref.normalize(current)
                self.assertEqual(move["actor"], normal.controller)
                self.assertIn(move["pit"], ref.legal_actions(current))
                if index == 0:
                    self.assertEqual(move["pit"], candidate["pit"])
                else:
                    self.assertEqual(move["pit"],
                                     ref.optimal_actions(current, depth - index)[0],
                                     (position, depth, candidate))
                transition = ref.apply(current, move["pit"])
                self.assertEqual(move["terminal_after"], transition.terminal)
                self.assertEqual(move["next_controller"],
                                 None if transition.terminal else transition.position.controller)
                self.assertEqual([(p["from"], p["to"])
                                  for p in move["forced_passes_after"]],
                                 list(transition.final_passes))
                current = transition.position
            ended = ref.normalize(current).terminal
            self.assertEqual(candidate["pv_truncated"], not ended)
            self.assertTrue(ended or len(candidate["pv"]) == depth)
            self.assertEqual(ref.evaluate(current), candidate["depth_result"]["value_player0"])

    @staticmethod
    def semantics(report):
        return (report["depth_result"]["value_player0"], report["best_moves"], report["pv"],
                [(c["pit"], c["depth_result"], c["pv"], c["pv_truncated"])
                 for c in report["candidates"]])

    def test_01_four_way_exhaustive_small_ablation(self):
        states = 0
        for pits in range(1, 4):
            for total in range(5):
                for counts in ref.compositions(total, 2 * pits + 2):
                    for actor in (0, 1):
                        position = ref.Position(pits, counts, actor)
                        baseline = None
                        for tt, pvs in ((0, False), (1, False), (0, True), (1, True)):
                            result = self.search(position, 4, tt, pvs)
                            self.assertIsNotNone(result["report"], result)
                            reports = result["reports"]
                            for report in reports:
                                self.assert_report(position, report)
                                type(self).state_horizon_ablation_cases += 1
                            semantics = [self.semantics(report) for report in reports]
                            if baseline is None:
                                baseline = semantics
                            else:
                                self.assertEqual(semantics, baseline, (position, tt, pvs))
                        states += 1
        self.assertEqual(states, 1550)

    def test_02_full_reduced_corpus_optimized(self):
        states = 0
        for pits in range(1, 4):
            for total in range(7):
                for counts in ref.compositions(total, 2 * pits + 2):
                    for actor in (0, 1):
                        position = ref.Position(pits, counts, actor)
                        result = self.search(position, 4)
                        for report in result["reports"]:
                            self.assert_report(position, report)
                            type(self).exhaustive_optimized_cases += 1
                        states += 1
        self.assertEqual(states, 8274)

    def test_03_product_bonus_pass_and_cache_collision_histories(self):
        for seed in range(64):
            rng = random.Random(seed)
            current = ref.Position(7, tuple([7] * 7 + [0] + [7] * 7 + [0]), seed % 2)
            for _ in range(rng.randrange(0, 48)):
                actions = ref.legal_actions(current)
                if not actions:
                    break
                current = ref.apply(current, rng.choice(actions)).position
            baseline = self.search(current, 4, 0, False)["report"]
            self.assert_report(current, baseline)
            for tt, pvs in ((1, False), (0, True), (1, True)):
                report = self.search(current, 4, tt, pvs)["report"]
                self.assert_report(current, report)
                self.assertEqual(self.semantics(report), self.semantics(baseline))
                self.assertLessEqual(report["search_options"]["tt_allocated_bytes"], tt * 1024**2)

    def test_04_exact_remaining_depth_and_fresh_resume(self):
        position = ref.Position(7, tuple([7] * 7 + [0] + [7] * 7 + [0]), 0)
        result = self.search(position, 7)
        self.assertEqual(result["stop_reason"], "max_depth")
        shallow = result["reports"][2]
        deep = result["report"]
        self.assertNotEqual(shallow["depth_result"]["value_player0"],
                            deep["depth_result"]["value_player0"])
        resumed = self.search(position, 7, start_depth=7)
        self.assertEqual(self.semantics(resumed["report"]), self.semantics(deep))
        independently_shallow = self.search(position, 3, start_depth=3)["report"]
        self.assertEqual(self.semantics(independently_shallow), self.semantics(shallow))
        self.assertGreater(deep["coverage"]["tt_hits"], 0)
        self.assertGreater(deep["coverage"]["pvs_probes"], 0)

    def test_05_cancellation_rejects_partial_iteration(self):
        position = ref.Position(7, tuple([7] * 7 + [0] + [7] * 7 + [0]), 0)
        with Control(self.native, max_depth=7, time_ms=0, node_budget=0) as control:
            self.native.lib.cc_set_search_options(control.handle, 1, 1)
            def cancel_after_complete(report):
                control.cancel()
                return True
            cancelled = self.native.analyze(position.counts, max_depth=7, control=control,
                                           on_report=cancel_after_complete)
        self.assertEqual(cancelled["stop_reason"], "cancelled")
        self.assertEqual(cancelled["last_completed_depth"], 1)
        self.assert_report(position, cancelled["report"])
        fresh = self.search(position, 7)["report"]
        resumed = self.search(position, 7, start_depth=2)["report"]
        self.assertEqual(self.semantics(fresh), self.semantics(resumed))
        aborted = self.search(position, 7, node_budget=10)
        self.assertEqual(aborted["stop_reason"], "node_budget")
        self.assertIsNone(aborted["report"])

    def test_06_options_guards_and_bounded_allocation(self):
        setter = self.native.lib.cc_set_search_options
        self.assertFalse(self.native._decode(setter(None, 0, 0))["ok"])
        with Control(self.native, max_depth=1, time_ms=0, node_budget=0) as control:
            for tt, pvs in ((-1, 0), (1025, 0), (1, -1), (1, 2), (512, 1)):
                self.assertEqual(self.native._decode(setter(control.handle, tt, pvs))
                                 ["error"]["code"], "invalid_search_options")
            self.assertTrue(self.native._decode(setter(control.handle, 1, 1))["ok"])
            observed = []
            def while_running(report):
                observed.append(self.native._decode(setter(control.handle, 0, 0)))
            report = self.native.analyze([7] * 7 + [0] + [7] * 7 + [0],
                                         max_depth=1, control=control,
                                         on_report=while_running)["report"]
            self.assertEqual(observed[0]["error"]["code"], "control_busy")
            self.assertGreater(report["search_options"]["tt_capacity"], 0)
            self.assertLessEqual(report["search_options"]["tt_allocated_bytes"], 1024**2)
            self.assertTrue(self.native._decode(setter(control.handle, 0, 0))["ok"])


if __name__ == "__main__":
    unittest.main()
