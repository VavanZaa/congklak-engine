"""Author arithmetic and no-search checks for the ordinary error review."""
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("ordinary_review",ROOT/"tools/analyze-pilot-ordinary.py")
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PilotOrdinaryReviewAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=p.Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError("Ordinary review ran a teacher search")
        p.Native.analyze=prohibited
        try:cls.value=p.build()
        finally:p.Native.analyze=original
        cls.manifest,cls.groups=p.data.prepare()

    def test_01_exact_cohorts_frozen_policy_and_literal_parity(self):
        self.assertEqual({name:len(value["rows"]) for name,value in self.value["groups"].items()},dict(ordinary_validation=14,ordinary_training=61,protected_diagnostic=8,base_ordinary_development=259))
        self.assertTrue(self.value["frozen_policy"]["all_six_tensors_equal"]);self.assertTrue(self.value["frozen_policy"]["all342_native_policy_logits_equal"])
        self.assertLess(self.value["inference"]["maximum_literal_native_value_difference"],1e-10);self.assertEqual(self.value["inference"]["model_dense_forwards"],684)
        self.assertNotIn("torch",sys.modules)
        for flag in ("no_teacher_searches","no_new_games","no_training","no_torch","no_weight_or_target_changes"):self.assertTrue(self.value[flag])

    def test_02_per_root_signed_error_and_cohort_denominators(self):
        for group in self.value["groups"].values():
            rows=group["rows"];summary=group["summary"]["all"]
            self.assertEqual(summary["rows"],len(rows));self.assertEqual(summary["improved"]+summary["worse"]+summary["tied"],len(rows))
            deltas=[]
            for row in rows:
                for model in ("control","pilot"):
                    error=row[model];delta=error["prediction_player0"]-row["target_player0"]
                    self.assertEqual(delta,error["error_player0"]);self.assertEqual((1 if row["actor"]==0 else -1)*delta,error["error_actor"])
                    self.assertEqual(delta/row["signed_active"],error["error_residual_actor"]);self.assertEqual(abs(delta),error["absolute_error_seeds"])
                expected=row["pilot"]["absolute_error_seeds"]-row["control"]["absolute_error_seeds"];self.assertEqual(expected,row["absolute_error_delta_seeds"]);deltas.append(expected)
                self.assertIn(8,[item["depth"] for item in row["requested_depth_sensitivity"]])
            self.assertEqual(math.fsum(deltas)/len(rows),summary["delta_mae_seeds"])
            self.assertEqual(sum(v["rows"] for v in group["summary"]["by_band"].values()),len(rows))

    def test_03_sampler_reconstruction_unique_denominators_and_nontraining_zero(self):
        rows=self.groups["train"];rng=random.Random(20261022);cdf=[];total=0.
        for row in rows:total+=row["sampling_probability"];cdf.append(total)
        import bisect
        observed=[0]*len(rows)
        for _ in range(51200):observed[bisect.bisect_right(cdf,rng.random()*total)]+=1
        ledger=self.value["sampling"];self.assertEqual([entry["observed_draws"] for entry in ledger["roots"]],observed)
        self.assertEqual(sum(observed),51200);self.assertEqual(ledger["unique_roots_drawn"],sum(v>0 for v in observed));self.assertLess(ledger["unique_roots_drawn"],51200)
        for name in ("ordinary_validation","protected_diagnostic","base_ordinary_development"):
            self.assertTrue(all(row["sampler_observed_root_draws"]==row["sampler_observed_family_draws"]==0 for row in self.value["groups"][name]["rows"]))
        self.assertEqual(sum(entry["draws"] for entry in ledger["families"]),51200)

    def test_04_source_provenance_horizon_sensitivity_and_nearest_exclusion(self):
        sources=p.target_sources()
        for name,group in self.value["groups"].items():
            for row in group["rows"]:
                for target in row["requested_depth_sensitivity"]:
                    if row["pool"]=="ordinary":self.assertEqual(target["item_sha256"],p.config.digest(sources[row["state_key"]][target["depth"]]["item"]))
                    self.assertEqual(target["absolute_error_delta_seeds"],abs(row["pilot"]["prediction_player0"]-target["target_player0"])-abs(row["control"]["prediction_player0"]-target["target_player0"]))
                if row["nearest_training"]:
                    for nearest in row["nearest_training"].values():self.assertNotEqual(nearest["family"],row["family"]);self.assertGreaterEqual(nearest["mean_l1_raw127"],0)
        query=self.groups["ordinary_development"][0];allowed=[candidate for candidate in self.groups["train"] if candidate["family"]!=query["family"]]
        expected=min(allowed,key=lambda candidate:(math.fsum(abs(a-b) for a,b in zip(query["cached_features"]["x"],candidate["cached_features"]["x"]))/127,candidate["state_key"]))
        actual=next(row for row in self.value["groups"]["ordinary_validation"]["rows"] if row["state_key"]==query["state_key"])
        self.assertEqual(actual["nearest_training"]["all_optimizer"]["state_key"],expected["state_key"])
        for name,digest in self.value["inputs"].items():self.assertEqual(p.sha(ROOT/name),digest)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PilotOrdinaryReviewAcceptance))
    body=dict(schema="pilot-ordinary-review-author-acceptance-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(PilotOrdinaryReviewAcceptance),inputs={name:p.sha(ROOT/name) for name in p.SOURCE_FILES+("build/pilot-loss-review-baseline.json",)})
    p.learning.old.teacher.atomic_json(ROOT/"build/pilot-loss-review-ordinary-author.json",body)
    raise SystemExit(not result.wasSuccessful())
