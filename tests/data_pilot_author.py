"""Author acceptance for strict ordinary sampling, ownership and durable labels."""
from contextlib import closing
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import unittest
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, data_pilot as d
from congklak.native import Native, flatten
from congklak.puct_cpu_inference import CpuBatchEvaluator


class DataPilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d.plan()
        cls.registry = d.read(d.REGISTRY); cls.native = Native()
        cls.base = ROOT / "build/data-pilot-author" / uuid.uuid4().hex
        cls.base.mkdir(parents=True)
        cls.settings = json.loads(config.canonical(d.CONTRACT)); cls.settings["families"] = 8; cls.settings["diagnostic_limit"] = 2
        cls.settings["caps"] = {c: {b: 1 for b in bands} for c, bands in cls.settings["caps"].items()}
        with CpuBatchEvaluator(d.MODEL, cache_capacity=0, max_batch=1) as evaluator:
            cls.families = {i: d.generate_family(i, cls.settings, cls.native, evaluator) for i in range(8)}

    def test_01_frozen_identity_metadata_only_diagnostics(self):
        baseline = d.checked_baseline()
        self.assertEqual(baseline["inputs_count"], 1834)
        self.assertEqual(d.sha(d.MODEL), d.MODEL_SHA)
        self.assertEqual(self.registry["protected_value_games"], 192)
        old = set(self.registry["existing9000_state_keys"])
        self.assertEqual(len(old), 9000)
        self.assertEqual(len({r["quotient"] for r in self.registry["diagnostic_pool"]}), len(self.registry["diagnostic_pool"]))
        for row in self.registry["diagnostic_pool"]:
            self.assertEqual(row["cohort"], "diagnostic")
            self.assertNotIn(row["state_key"], old)
            self.assertIn(row["quotient"], self.registry["quotients"])
        self.assertFalse(self.registry["sealed_target_bodies_read"])

    def test_02_legal_histories_seeded_ownership_and_strict_all_move_ordinary(self):
        with CpuBatchEvaluator(d.MODEL, cache_capacity=0, max_batch=1) as evaluator:
            self.assertEqual(self.families[7], d.generate_family(7, self.settings, self.native, evaluator))
        observed = 0
        for index, family in self.families.items():
            rank = int.from_bytes(d.hashlib.sha256(("data-pilot-owner-v1/" + family["family"]).encode()).digest()[:8], "big") % 100
            self.assertEqual(family["cohort"], "training_candidate" if rank < 80 else "validation")
            p = config.position(); states = []
            for action in family["moves"]:
                p = self.native.normalize(flatten(p), actor=p["controller"], total=98)["position"]; states.append(p)
                p = self.native.apply(flatten(p), actor=p["controller"], action=action, total=98, trace=False)["position"]
            self.assertEqual(p, family["final"])
            self.assertLessEqual(family["choices"], 192)
            for row in family["candidates"]:
                observed += 1
                self.assertEqual(row["provenance"]["raw_state"], states[row["provenance"]["step"]])
                key, canonical, swapped = d.pilot.canonicalize(row["provenance"]["raw_state"], self.native)
                self.assertEqual((key.hex(), canonical, swapped), (row["state_key"], row["state"], row["provenance"]["swapped"]))
                normal = self.native.normalize(flatten(canonical), actor=canonical["controller"], total=98)
                facts = [self.native.apply(flatten(canonical), actor=canonical["controller"], action=a, total=98, trace=False) for a in normal["legal_actions"]]
                self.assertTrue(all(not t["passes"] and t["reason"] not in ("own_capture", "own_store") for t in facts))
                self.assertEqual(row["terminal_children"], sum(t["terminal"] for t in facts))
                self.assertGreaterEqual(row["active"], 25)
        self.assertGreater(observed, 0)

    def test_03_global_cross_cohort_closure_caps_and_no_protected_leak(self):
        roots, stats = d.select(self.families, self.settings, self.registry)
        fresh = [r for r in roots.values() if r["cohort"] != "diagnostic"]
        self.assertEqual(len({r["quotient"] for r in fresh}), len(fresh))
        self.assertEqual(len({(r["family"], r["band"]) for r in fresh}), len(fresh))
        self.assertTrue(all(r["quotient"] not in self.registry["quotients"] for r in fresh))
        self.assertTrue(all(stats["counts"][c][b] + stats["shortfall"][c][b] == 1 for c in self.settings["caps"] for b in self.settings["caps"][c]))
        row = copy.deepcopy(fresh[0]); other = copy.deepcopy(row)
        other["cohort"] = "validation" if row["cohort"] == "training_candidate" else "training_candidate"
        settings = copy.deepcopy(self.settings); settings["diagnostic_limit"] = 0
        blocked, evidence = d.select({0: {"candidates": [row]}, 1: {"candidates": [other]}}, settings, self.registry)
        self.assertFalse(blocked); self.assertEqual(evidence["cross_cohort_candidate_collisions"], 1)
        registry = copy.deepcopy(self.registry); registry["quotients"].append(row["quotient"])
        blocked, _ = d.select({0: {"candidates": [row]}}, settings, registry)
        self.assertFalse(blocked)

    def test_04_commit_resume_full_H8_closed_reader_and_corruption_refusal(self):
        run, fresh = self.base / "resume", self.base / "fresh"
        def interrupt(value):
            if value["event"] == "data_pilot_family_committed" and value["index"] == 2:
                raise RuntimeError("owner interruption after durable family")
        with self.assertRaisesRegex(RuntimeError, "owner interruption"):
            d.run(run, fixture=True, families=8, diagnostic_limit=2, on_event=interrupt)
        with closing(sqlite3.connect(run / "work.sqlite3")) as db:
            before = db.execute("SELECT idx,body,hash FROM families ORDER BY idx").fetchall()
        result = d.run(run, fixture=True, families=8, diagnostic_limit=2)
        d.run(fresh, fixture=True, families=8, diagnostic_limit=2)
        roots, labels, attempts, manifest = d.load_rows(run)
        fresh_roots, fresh_labels, _, _ = d.load_rows(fresh)
        self.assertEqual(roots, fresh_roots)
        self.assertGreaterEqual(len(labels), 2)
        self.assertEqual(manifest, result)
        for key in labels:
            a = {k: v for k, v in labels[key]["label"].items() if k != "work"}
            b = {k: v for k, v in fresh_labels[key]["label"].items() if k != "work"}
            self.assertEqual(a, b)
            self.assertEqual(a["choice_depth"], 8)
            self.assertEqual(len(a["action_scores"]), roots[key]["legal_mask"].bit_count())
        with closing(sqlite3.connect(run / "work.sqlite3")) as db:
            self.assertEqual(before, db.execute("SELECT idx,body,hash FROM families ORDER BY idx LIMIT 3").fetchall())
        def snapshot():
            return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
        stable = snapshot()
        self.assertEqual(result, d.run(run, fixture=True, families=8, diagnostic_limit=2))
        self.assertEqual(stable, snapshot())
        zeroed = copy.deepcopy(attempts)
        zeroed[sorted(zeroed)[0]]["work"] = {name: 0 for name in zeroed[sorted(zeroed)[0]]["work"]}
        with self.assertRaises(ValueError):
            d.validate_attempts(roots, zeroed, labels, self.settings, self.native)
        damaged = self.base / "damaged"; damaged.mkdir()
        with closing(sqlite3.connect(run / "work.sqlite3")) as source, closing(sqlite3.connect(damaged / "work.sqlite3")) as destination:
            source.backup(destination)
        shutil.copy2(run / "manifest.json", damaged / "manifest.json")
        with closing(sqlite3.connect(damaged / "work.sqlite3")) as db:
            key = sorted(attempts)[0]; attempt = copy.deepcopy(attempts[key]); attempt["status"] = "budget_rejected"
            attempt["stop_reason"] = "node_budget"; attempt["label_sha256"] = None; attempt["work"]["nodes"] = 2000000
            with db:
                db.execute("DELETE FROM labels WHERE state=?", (key,))
                db.execute("UPDATE attempts SET body=?,hash=? WHERE state=?", (config.canonical(attempt), config.digest(attempt), key))
        with self.assertRaises(ValueError):
            d.load_rows(damaged)
        with closing(sqlite3.connect(damaged / "work.sqlite3")) as db:
            with db:
                db.execute("DELETE FROM meta WHERE key='complete'")
        with self.assertRaisesRegex(ValueError, "no committed complete"):
            d.load_rows(damaged)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DataPilotAuthor))
    value = {"schema": "ordinary-pilot-author-checks-v1", "passed": result.wasSuccessful(), "tests_run": result.testsRun,
             "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
             "methods": unittest.defaultTestLoader.getTestCaseNames(DataPilotAuthor),
             "inputs": {name: d.sha(ROOT / name) for name in d.SOURCES + ("build/data-pilot-baseline.json", "build/data-pilot-registry.json", "build/data-pilot-plan.json")}}
    d.teacher.atomic_json(ROOT / "build/data-pilot-author.json", value)
    raise SystemExit(not result.wasSuccessful())
