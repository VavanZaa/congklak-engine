"""Source-bound owner checks for the saved validation review."""
import copy
from contextlib import ExitStack
from pathlib import Path
import random
import bisect
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_value_validation_review as p


class OrdinaryValueValidationReviewAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.body=p.build();cls.base=ROOT/"build/ordinary-value-validation-review-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True)

    def test_01_six_roots_actor_residual_horizon_arithmetic(self):
        roots=self.body["validation_roots"];self.assertEqual(len(roots),6);self.assertEqual(self.body["summary"]["improved_roots"],3);self.assertEqual(self.body["summary"]["worsened_roots"],3)
        for row in roots.values():
            sign=1 if row["actor"]==0 else -1;self.assertEqual(row["signed_active"],sign*sum(map(sum,row["state"]["pits"])));self.assertEqual(row["D_player0"],row["state"]["stores"][0]-row["state"]["stores"][1])
            for stage,item in row["horizons"].items():
                for profile in ("control","candidate"):
                    value=item[profile];self.assertAlmostEqual(value["signed_error_actor"],sign*value["signed_error_player0"]);self.assertAlmostEqual(value["signed_error_residual"],value["signed_error_player0"]/row["signed_active"]);self.assertAlmostEqual(value["target_player0"],row["D_player0"]+row["signed_active"]*value["target_residual"])
                self.assertAlmostEqual(item["absolute_error_delta"],item["candidate"]["absolute_error_seeds"]-item["control"]["absolute_error_seeds"])
        self.assertAlmostEqual(sum(row["horizons"]["h10"]["H10_aggregate_contribution"] for row in roots.values()),self.body["summary"]["H10_mae_delta"])
        self.assertEqual(self.body["summary"]["max_H10_H12_target_shift"],11)

    def test_02_frozen_policy_head_algebra_fixed_interpolation_and_bounds(self):
        self.assertTrue(self.body["policy_invariance"]["six_frozen_tensors_identical"]);self.assertLessEqual(self.body["inference_budget"]["unique_fullstates_with_literal_trunk_forwards"],60)
        self.assertEqual(self.body["head_interpolation"]["coefficients"],[0.,.25,.5,.75,1.]);self.assertFalse(self.body["head_interpolation"]["model_or_checkpoint_selected"])
        for key,row in self.body["validation_roots"].items():
            self.assertEqual(row["control_policy_logits"],row["candidate_policy_logits"]);self.assertEqual(row["control_legal_argmax"],row["candidate_legal_argmax"]);h=row["head_shift"];self.assertAlmostEqual(h["actual_preactivation_change"],h["bias_contribution"]+h["hidden_weight_contribution"])
            self.assertTrue(row["horizons"]["h10"]["exact_finite_head_representable"]);self.assertFalse(row["horizons"]["h10"]["saturation_endpoint"])
            self.assertAlmostEqual(self.body["head_interpolation"]["values"]["0.0"]["predictions_player0"][key],row["control_prediction_player0"]);self.assertAlmostEqual(self.body["head_interpolation"]["values"]["1.0"]["predictions_player0"][key],row["candidate_prediction_player0"])
        self.assertEqual(self.body["saved_checkpoint_metrics"]["selected_final_step"],200)

    def test_03_independent_draw_ledger_noise_and_targetblind_five_neighbors(self):
        groups=p.read(ROOT/"build/ordinary-value-rows.json")["groups"];sampler=p.read(ROOT/"data/ordinary-value-pilot-v1/result.json")["identity"]["sampler"];rng=random.Random(20261028);actual={row["id"]:0 for values in groups.values() for row in values}
        for _ in range(200):
            for name in ("high","low"):
                names=sorted(sampler[name])
                for _ in range(16):actual[groups[name][rng.choice(sampler[name][rng.choice(names)])]["id"]]+=1
            for _ in range(32):actual[groups["rehearsal"][bisect.bisect_right(sampler["rehearsal_cdf"],rng.random()*sampler["rehearsal_cdf"][-1])]["id"]]+=1
        self.assertEqual(actual,{key:row["actual_draws"] for key,row in self.body["exposure"]["ledger"].items()});self.assertEqual(sum(actual.values()),12800)
        ancestry=self.body["exposure"]["ancestral_quality_draw_exposure"];self.assertEqual(ancestry["rehearsal_roots_with_proven_positive_ancestral_draws"],7539);self.assertEqual(ancestry["current_zero_draw_rehearsal_rows_with_positive_ancestral_draws"],3221);self.assertFalse(ancestry["exact_ancestral_per_root_counts_reconstructed"]);self.assertFalse(ancestry["nonzero_gradient_or_causal_effect_proven"])
        self.assertEqual(len(self.body["ordinary_training_pairs"]),24);self.assertEqual(sum(row["absolute_pair_gap_seeds"]>0 for row in self.body["ordinary_training_pairs"]),22)
        for key,root in self.body["validation_roots"].items():
            pits=lambda state:state["pits"][state["controller"]]+state["pits"][1-state["controller"]]
            expected=sorted(groups["rehearsal"],key=lambda row:(sum(abs(a-b) for a,b in zip(pits(root["state"]),pits(row["state"]))),row["id"]))[:5]
            self.assertEqual([row["id"] for row in expected],[row["row_id"] for row in self.body["coverage"][key]["neighbors"]]);self.assertTrue(all(row["family"]!=root["family"] for row in expected))

    def test_04_rehashed_corruption_refusal_no_search_no_torch_and_stable_API_CLI(self):
        from congklak import quality_search
        with ExitStack() as stack:
            stack.enter_context(patch.object(p.Native,"analyze",side_effect=AssertionError("Review searched teacher")));stack.enter_context(patch.object(quality_search,"search",side_effect=AssertionError("Review searched PUCT")));stack.enter_context(patch.object(quality_search,"forest",side_effect=AssertionError("Review searched PUCT")))
            path=self.base/"ordinary-value-validation-review.json";self.assertEqual(p.run(path),self.body);before=(p.sha(path),path.stat().st_mtime_ns);self.assertEqual(p.verify(path),self.body);self.assertEqual(p.run(path),self.body);self.assertEqual(before,(p.sha(path),path.stat().st_mtime_ns))
            for mutation in (lambda r:r["summary"].update(H10_mae_delta=999.),lambda r:r["inputs"].update({next(iter(r["inputs"])):"0"*64}),lambda r:r["policy_invariance"].update(six_frozen_tensors_identical=1)):
                bad=copy.deepcopy(self.body);mutation(bad);bad["sha256"]=config.digest({key:value for key,value in bad.items() if key!="sha256"})
                with self.assertRaises(ValueError):p.validate_report(bad)
        self.assertNotIn("torch",sys.modules);cli=self.base/"ordinary-value-validation-review-cli.json";run=subprocess.run([sys.executable,str(ROOT/"tools/review-ordinary-value-validation.py"),"--output",str(cli)],cwd=ROOT,text=True,capture_output=True,timeout=120);self.assertEqual(run.returncode,0,run.stdout+run.stderr);self.assertEqual(p.read(cli),self.body)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrdinaryValueValidationReviewAuthor))
    methods=unittest.defaultTestLoader.getTestCaseNames(OrdinaryValueValidationReviewAuthor)
    body=dict(schema="ordinary-value-validation-review-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=methods,covered_methods=[__name__+".OrdinaryValueValidationReviewAuthor."+name for name in methods],inputs=OrdinaryValueValidationReviewAuthor.body["inputs"],fixture_path=OrdinaryValueValidationReviewAuthor.base.relative_to(ROOT).as_posix())
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/"build/ordinary-value-validation-review-author.json",body);raise SystemExit(not result.wasSuccessful())
