"""Author checks for seed-space loss and byte-exact frozen-policy value refit."""
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
from congklak import config, value_learning as learning
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator


class ValueLearningAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.rows = learning.prepare()
        cls.torch, cls.prior = learning.runtime()
        cls.prior.deterministic("cpu")
        cls.source = Model(learning.SOURCE)
        cls.directory = ROOT / "build/value-learning-acceptance" / uuid.uuid4().hex
        cls.directory.mkdir(parents=True)

    def model(self):
        model = learning.evaluator()
        model.load_state_dict({k: self.torch.tensor(v, dtype=self.torch.float32) for k, v in self.source.weights.items()})
        return model

    def test_01_admission_preserves_ancestral_limits_and_root_seed_targets(self):
        ancestral = json.loads((ROOT / "build/quality-data.json").read_text(encoding="utf-8"))
        registry = learning.read_registry()
        protected_q, protected_f = set(registry["quotients"]), set(registry["families"])
        self.assertLessEqual(set(self.manifest["train_state_keys"]), set(ancestral["train_state_keys"]))
        self.assertEqual(self.manifest["train_rows"] + len(self.manifest["excluded_rows"]), 8000)
        self.assertFalse(set(self.manifest["train_families"]) & (protected_f | set(self.manifest["quotient_excluded_families"])))
        self.assertFalse(set(self.manifest["train_families"]) & set(self.manifest["development_families"]))
        self.assertTrue(all(learning.pilot.quotient(r["state"]).hex() not in protected_q for r in self.rows["train"]))
        actors = set()
        for row in self.rows["development"][:100]:
            t = learning.target(row)
            actors.add(row["state"]["controller"])
            self.assertEqual(t["y_player0"], row["label"]["target_player0"])
            self.assertAlmostEqual(t["d"] + t["r"] * t["residual"], t["y_player0"])
            self.assertTrue(-1 <= t["residual"] <= 1)
        self.assertEqual(actors, {0, 1})
        self.assertFalse(self.manifest["sealed_target_bodies_read"])
        self.assertEqual(self.manifest["training_replay_rows"], 0)

    def test_02_direct_seed_huber_and_only_two_value_tensors_train(self):
        torch = self.torch
        model = self.model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        t = learning.tensors(self.rows["train"][:64])
        with torch.no_grad():
            old_z, old_logits = model.both(t["x"])
        z, logits = model.both(t["x"])
        prediction = t["d"] + t["r"] * z
        error = prediction - t["y_player0"]
        loss = torch.nn.functional.huber_loss(prediction, t["y_player0"], delta=4.)
        manual = torch.where(error.abs() < 4., .5 * error.square(), 4. * (error.abs() - 2.)).mean()
        self.assertTrue(torch.equal(loss, manual))
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in trainable), 65)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad], list(learning.TRAINABLE))
        optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=0., foreach=False)
        loss.backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in learning.TRAINABLE))
        torch.nn.utils.clip_grad_norm_(trainable, 1.)
        optimizer.step()
        self.assertEqual(set(optimizer.state_dict()["state"]), {0, 1})
        self.assertEqual(optimizer.state_dict()["param_groups"][0]["params"], [0, 1])
        self.assertEqual(learning.frozen_digest(model.state_dict()), learning.FROZEN_WEIGHTS)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items() if k not in learning.TRAINABLE))
        self.assertTrue(all(not torch.equal(before[k], model.state_dict()[k]) for k in learning.TRAINABLE))
        with torch.no_grad():
            new_z, new_logits = model.both(t["x"])
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertFalse(torch.equal(old_z, new_z))
        self.updated = model

    def test_03_native_policy_logits_bit_exact_after_value_only_export(self):
        torch = self.torch
        model = self.model()
        with torch.no_grad():
            model.layers[4].bias.add_(.01)
        identity = {"input_count": 127, "width": 64, "rules_digest": config.RULES_DIGEST}
        payload = learning.export_payload(model.state_dict(), identity, 1)
        exported = self.directory / "native-policy-proof-model.json"
        learning.publish(exported, {"model": payload, "sha256": config.digest(payload)})
        parsed = Model(exported)
        self.assertEqual(learning.model_frozen_digest(parsed), learning.FROZEN_WEIGHTS)
        values_changed = 0
        with CpuBatchEvaluator(self.source, max_batch=32) as old, CpuBatchEvaluator(parsed, max_batch=32) as new:
            for row in self.rows["development"][:40]:
                before = old.evaluate_many([row["state"]])[0]
                after = new.evaluate_many([row["state"]])[0]
                self.assertEqual(before["policy_logits"], after["policy_logits"])
                values_changed += before["value_player0"] != after["value_player0"]
        self.assertGreater(values_changed, 0)

    def test_04_completed_fixture_recovery_state_and_refusal_guards(self):
        with self.assertRaisesRegex(ValueError, "fixed CUDA"):
            learning.train(device="cpu")
        with self.assertRaisesRegex(ValueError, "frozen protected registry"):
            learning.train(device="cuda", registry_path=self.directory / "other.json")
        run = self.directory / "run"
        result = learning.train(run, device="cpu", fixture=True, steps=2, checkpoint_every=1)
        proof = learning.verify_run(run, fixture=True)
        self.assertEqual(proof["checkpoints_verified"], 2)
        self.assertEqual(proof["trainable_parameter_count"], 65)
        self.assertTrue(proof["all_frozen_tensors_bit_exact"])
        def snapshot():
            return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
        before = snapshot()
        self.assertEqual(result, learning.train(run, device="cpu", fixture=True, steps=2, checkpoint_every=1))
        self.assertEqual(before, snapshot())
        with closing(sqlite3.connect(run / "catalog.sqlite3")) as db:
            name = db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
        state = self.torch.load(run / name, map_location="cpu", weights_only=True)
        for mutation in (lambda s: s.update(draws=True),
                         lambda s: s["model"]["policy.bias"].add_(.01),
                         lambda s: s["optimizer"]["param_groups"][0].update(weight_decay=.0001),
                         lambda s: s["optimizer"]["state"][0]["step"].fill_(1),
                         lambda s: s["optimizer"]["state"][0].update(exp_avg=self.torch.zeros(1))):
            damaged = copy.deepcopy(state)
            mutation(damaged)
            with self.assertRaises(ValueError):
                learning.validate_state(damaged, state["identity"], self.model())
        for name, value in (("frozen_policy_unchanged", False), ("train_rows", True), ("elapsed_seconds_this_invocation", float("nan"))):
            damaged = copy.deepcopy(result)
            damaged[name] = value
            with self.assertRaises(ValueError):
                learning.validate_result(damaged, state["identity"], self.manifest)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValueLearningAcceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"schema": "value-learning-author-acceptance-v1", "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
              "methods": unittest.defaultTestLoader.getTestCaseNames(ValueLearningAcceptance),
              "inputs": {name: learning.sha(ROOT / name) for name in ("congklak/value_learning.py", "tools/value-train.py", "tests/value_learning_acceptance.py", "build/value-registry.json", "build/value-data.json", "data/quality-train-v1/model.json", "build/quality-model-witness.json")}}
    learning.teacher.atomic_json(ROOT / "build/value-learning-acceptance.json", report)
    raise SystemExit(not result.wasSuccessful())
