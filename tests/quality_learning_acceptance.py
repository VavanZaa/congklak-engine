"""Author checks for target ownership, joint gradients and checkpoint admission."""
from contextlib import closing
import copy
import json
import math
from pathlib import Path
import sqlite3
import sys
import unittest
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, quality_learning as learning
from congklak.neural_relay import Model


class QualityLearningAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.rows = learning.prepare()
        cls.torch, cls.prior = learning.runtime()
        cls.prior.deterministic("cpu")
        cls.source = Model(learning.SOURCE)
        cls.directory = ROOT / "build/quality-learning-acceptance" / uuid.uuid4().hex
        cls.directory.mkdir(parents=True)

    def test_01_admitted_sets_exclude_quotients_and_whole_families(self):
        _, original, _ = learning.teacher.load(teacher_depth=8)
        registry = learning.read_registry()
        protected_q, protected_f = set(registry["quotients"]), set(registry["families"])
        bridge = {r["family"] for r in original["train"] if learning.pilot.quotient(r["state"]).hex() in protected_q}
        expected = [r["state_key"] for r in original["train"] if r["family"] not in protected_f | bridge and learning.pilot.quotient(r["state"]).hex() not in protected_q]
        self.assertEqual(expected, self.manifest["train_state_keys"])
        self.assertEqual(len(expected), self.manifest["train_rows"])
        self.assertEqual(self.manifest["train_rows"] + len(self.manifest["excluded_rows"]), 8000)
        self.assertFalse(set(self.manifest["train_families"]) & set(self.manifest["development_families"]))
        self.assertFalse(self.manifest["sealed_target_bodies_read"])

    def test_02_actor_softQ_seed_units_residual_and_illegal_mass(self):
        actors = set()
        for row in self.rows["development"][:100]:
            value = learning.target(row)
            sign = 1 if row["state"]["controller"] == 0 else -1
            actors.add(row["state"]["controller"])
            scores = row["label"]["action_scores"]
            best = max(sign * a["value_player0"] for a in scores)
            raw = [math.exp((sign * a["value_player0"] - best) / 2.) for a in scores]
            expected = [0.] * 7
            for action, weight in zip(scores, raw):
                expected[action["pit"]] = weight / sum(raw)
            self.assertEqual(value["probability"], expected)
            self.assertAlmostEqual(value["d"] + value["r"] * value["residual"], row["label"]["target_player0"])
            self.assertTrue(-1 <= value["residual"] <= 1)
            self.assertTrue(all(p == 0 for p, legal in zip(value["probability"], value["legal"]) if not legal))
        self.assertEqual(actors, {0, 1})
        for temperature in (0, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                learning.target(self.rows["train"][0], temperature)
        bad = copy.deepcopy(self.rows["train"][0])
        bad["cached_features"]["r"] *= -1
        with self.assertRaises(ValueError):
            learning.target(bad)

    def test_03_all_eight_tensors_train_and_masked_loss_is_finite(self):
        torch = self.torch
        model = learning.evaluator()
        model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in self.source.weights.items()})
        before = {k: v.clone() for k, v in model.state_dict().items()}
        t = learning.tensors(self.rows["train"][:64])
        z, logits = model.both(t["x"])
        loss = torch.nn.functional.huber_loss(z, t["residual"], delta=.1) + .25 * learning.policy_losses(logits, t["probability"], t["legal"]).mean()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001, foreach=False)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertEqual(len(optimizer.state), 8)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 12872)
        self.assertTrue(all(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        with self.assertRaises(ValueError):
            learning.policy_losses(logits, t["probability"], torch.zeros_like(t["legal"]))

    def test_04_completed_fixture_state_exports_and_no_rewrite(self):
        with self.assertRaisesRegex(ValueError, "fixed CUDA"):
            learning.train(device="cpu")
        with self.assertRaisesRegex(ValueError, "frozen protected registry"):
            learning.train(device="cuda", registry_path=self.directory / "unapproved-registry.json")
        run = self.directory / "run"
        result = learning.train(run, device="cpu", fixture=True, steps=2, checkpoint_every=1)
        self.assertEqual(result["steps_completed"], 2)
        self.assertEqual(result["selected_step"], 2)
        proof = learning.verify_run(run, fixture=True)
        self.assertTrue(proof["final_export_equals_checkpoint_tensors"])
        self.assertEqual(proof["checkpoints_verified"], 2)
        def snapshot():
            return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
        before = snapshot()
        rerun = learning.train(run, device="cpu", fixture=True, steps=2, checkpoint_every=1)
        self.assertEqual(result, rerun)
        self.assertEqual(before, snapshot())
        with closing(sqlite3.connect(run / "catalog.sqlite3")) as db:
            name = db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
        state = self.torch.load(run / name, map_location="cpu", weights_only=True)
        model = learning.evaluator()
        for mutate in (lambda s: s.update(draws=True),
                       lambda s: s["optimizer"]["param_groups"][0].update(capturable=True),
                       lambda s: s["optimizer"]["state"][0]["step"].fill_(1),
                       lambda s: s["model"]["policy.bias"].fill_(float("nan"))):
            damaged = copy.deepcopy(state)
            mutate(damaged)
            with self.assertRaises(ValueError):
                learning.validate_state(damaged, state["identity"], model)
        for name, bad_value in (("train_rows", True), ("sealed_final_test_consumed", True), ("elapsed_seconds_this_invocation", float("nan"))):
            damaged = copy.deepcopy(result)
            damaged[name] = bad_value
            with self.assertRaises(ValueError):
                learning.validate_result(damaged, state["identity"], self.manifest)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QualityLearningAcceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"schema": "quality-learning-author-acceptance-v1", "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
              "methods": unittest.defaultTestLoader.getTestCaseNames(QualityLearningAcceptance),
              "inputs": {name: learning.sha(ROOT / name) for name in ("congklak/quality_learning.py", "tools/quality-train.py", "tests/quality_learning_acceptance.py", "build/quality-registry.json", "build/quality-data.json")}}
    learning.teacher.atomic_json(ROOT / "build/quality-learning-acceptance.json", report)
    raise SystemExit(not result.wasSuccessful())
