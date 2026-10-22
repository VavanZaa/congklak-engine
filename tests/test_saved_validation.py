"""Independent malformed-checkpoint checks, including valid body checksums."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from congklak import config
from congklak.__main__ import Coordinator
from congklak.native import Native, flatten
from congklak.storage import Store, StorageError

TEMP_ROOT = (Path(__file__).resolve().parent.parent / "build" / "acceptance").resolve()


class SavedValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()
        cls.identity = config.identity(cls.native.build_id)
        cls.root = config.position()
        cls.settings = config.limits({"max_depth": 2, "time_ms": 0, "node_budget": 0})
        result = cls.native.analyze(flatten(cls.root), **cls.settings)
        cls.reports = {report["choice_depth"]: report for report in result["reports"]}

    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="saved-validator-", dir=TEMP_ROOT)
        self.path = Path(self.temp.name).resolve()
        self.assertTrue(self.path.is_relative_to(TEMP_ROOT))
        self.store = Store(self.path, self.identity, native=self.native)
        self.job = self.store.create(self.root, self.settings)
        self.store.commit_report(self.job, self.reports[1])
        self.store.commit_report(self.job, self.reports[2])

    def tearDown(self):
        self.store.close()
        self.assertTrue(self.path.resolve().is_relative_to(TEMP_ROOT))
        self.temp.cleanup()

    def rewrite_report(self, report):
        with self.store.db:
            self.store.db.execute(
                "UPDATE reports SET body=?,body_hash=? WHERE job_id=? AND depth=2",
                (config.canonical(report), config.digest(report), self.job))

    def test_checksum_valid_malformed_reports_restore_previous(self):
        def removed_result(report):
            report.pop("depth_result")

        def wrong_ranking(report):
            report["best_moves"] = []

        def fake_pass(report):
            report["forced_passes"] = [{"from": 0, "to": 1}]

        def wrong_interval(report):
            report["physical_interval"]["lower"] += 1

        def wrong_pv_actor(report):
            report["pv"][0]["actor"] = 1

        def wrong_pv_flag(report):
            report["pv"][0]["bonus_choice"] = not report["pv"][0]["bonus_choice"]

        def changed_trace(report):
            report["candidates"][0]["transition"]["events"][0]["count"] += 1

        def bool_trace_integer(report):
            event = next(event for event in report["candidates"][0]["transition"]["events"]
                         if event["type"] == "deposit")
            event["count"] = True

        def bool_pv_next_actor(report):
            choice = next(choice for candidate in report["candidates"]
                          for choice in candidate["pv"]
                          if choice["next_controller"] is not None)
            choice["next_controller"] = bool(choice["next_controller"])

        mutations = [
            ("missing_result", removed_result),
            ("string_terminal", lambda r: r.update(terminal="yes")),
            ("integer_completion", lambda r: r.update(completed_iteration=1)),
            ("boolean_controller", lambda r: r.update(normalized_controller=False)),
            ("wrong_ranking", wrong_ranking),
            ("wrong_pass", fake_pass),
            ("wrong_interval", wrong_interval),
            ("wrong_pv_actor", wrong_pv_actor),
            ("wrong_pv_flag", wrong_pv_flag),
            ("changed_trace", changed_trace),
            ("boolean_trace_count", bool_trace_integer),
            ("boolean_pv_next_actor", bool_pv_next_actor),
            ("invented_proof", lambda r: r.update(proof_status="full_game_proven")),
            ("wrong_evaluator", lambda r: r.update(evaluator="H_v2")),
        ]
        for name, mutation in mutations:
            with self.subTest(mutation=name):
                broken = deepcopy(self.reports[2])
                mutation(broken)
                self.rewrite_report(broken)
                saved = self.store.load(self.job, compatible=True)
                self.assertEqual(saved["last_completed_depth"], 1)
                self.assertEqual(saved["recovery"]["discarded_invalid_depths"], [2])
                self.assertTrue(saved["recovery"]["restored_previous_generation"])
                self.assertEqual(saved["report"], self.reports[1])

    def test_malformed_reports_rejected_before_transaction(self):
        for change in (lambda r: r.pop("depth_result"),
                       lambda r: r.update(terminal="yes")):
            broken = deepcopy(self.reports[2])
            change(broken)
            with self.assertRaises(StorageError):
                self.store.commit_report(self.job, broken)
            self.assertEqual(self.store.load(self.job)["report"], self.reports[2])

    def test_no_valid_generation_explicitly_unranked(self):
        broken = deepcopy(self.reports[2])
        broken.pop("depth_result")
        self.rewrite_report(broken)
        with self.store.db:
            self.store.db.execute("UPDATE reports SET body='null',body_hash=? WHERE depth=1",
                                  (config.digest(None),))
        saved = self.store.load(self.job)
        self.assertIsNone(saved["report"])
        self.assertEqual(saved["last_completed_depth"], -1)
        self.assertEqual(saved["recovery"]["discarded_invalid_depths"], [2, 1])

    def test_null_and_incomplete_saved_payloads_are_errors(self):
        original = self.store.load(self.job)["payload"]
        for name, mutate in (
            ("null_position", lambda p: p.update(position=None)),
            ("null_limits", lambda p: p.update(initial_limits=None)),
            ("incomplete_limits", lambda p: p.update(initial_limits={"max_depth": 2})),
        ):
            with self.subTest(payload=name):
                broken = deepcopy(original)
                mutate(broken)
                with self.store.db:
                    self.store.db.execute("UPDATE jobs SET payload=?,payload_hash=? WHERE id=?",
                                          (config.canonical(broken), config.digest(broken), self.job))
                with self.assertRaises(StorageError):
                    self.store.load(self.job)

    def test_null_and_incomplete_saved_run_limits_are_errors(self):
        run_id = self.store.begin(self.job, self.settings)
        for settings in (None, {"max_depth": 2}):
            with self.subTest(settings=settings):
                with self.store.db:
                    self.store.db.execute("UPDATE runs SET settings=?,settings_hash=? WHERE id=?",
                                          (config.canonical(settings), config.digest(settings), run_id))
                with self.assertRaises(StorageError):
                    self.store.load(self.job)

    def test_lower_resume_horizon_is_rejected_instead_of_relabeling(self):
        coordinator = Coordinator(self.native, self.store)
        with self.assertRaises(config.InputError) as raised:
            coordinator.handle({"op": "resume", "job_id": self.job,
                                "limits": {"max_depth": 1, "time_ms": 0, "node_budget": 0}})
        self.assertEqual(raised.exception.code, "invalid_resume_depth")
        self.assertIsNone(coordinator.active)
        self.assertIsNone(coordinator.worker)
        self.assertEqual(self.store.load(self.job)["report"], self.reports[2])


if __name__ == "__main__":
    unittest.main()
