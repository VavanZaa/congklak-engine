"""Author checks for the small reviewed sampler and frozen-policy value pilot."""
import copy
from pathlib import Path
import random
import sqlite3
import sys
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,value_pilot_learning as p
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator


class ValuePilotLearningAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.groups=p.prepare();cls.torch,cls.prior=p.runtime();cls.prior.deterministic("cpu")
        cls.source=Model(p.SOURCE);cls.base=ROOT/"build/value-pilot-learning-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)

    def model(self):
        model=p.evaluator();model.load_state_dict({key:self.torch.tensor(value,dtype=self.torch.float32) for key,value in self.source.weights.items()});return model

    def test_01_reviewed_probabilities_unit_loss_and_signed_seed_targets(self):
        rows=self.groups["train"];self.assertEqual(len(rows),self.manifest["train_rows"])
        cumulative=p.cdf(rows);rng=random.Random(p.CONTRACT["seed"]);literal=random.Random(p.CONTRACT["seed"])
        ids=p.sample_ids(rng,cumulative,100)
        import bisect
        self.assertEqual(ids,[bisect.bisect_right(cumulative,literal.random()*cumulative[-1]) for _ in range(100)])
        self.assertEqual(rng.getstate(),literal.getstate());self.assertTrue(all(row["loss_weight"]==1. for row in rows))
        self.assertEqual(self.manifest["new_ordinary_train_rows"],61)
        self.assertFalse({row["family"] for row in rows}&{row["family"] for split,values in self.groups.items() if split!="train" for row in values})
        actors=set()
        for split,values in self.groups.items():
            for row in values[:100]:
                target=p.target(row);actors.add(row["state"]["controller"])
                self.assertEqual(target["y"],row["label"]["target_player0"]);self.assertAlmostEqual(target["d"]+target["r"]*target["z"],target["y"])
        self.assertEqual(actors,{0,1});self.assertEqual(p.CONTRACT["lr"],.0003);self.assertEqual(p.CONTRACT["steps"],200)

    def test_02_cached_forward_seed_huber_and_two_parameter_ownership(self):
        torch=self.torch;model=self.model();rows=self.groups["train"][:64];groups={"train":rows};data={"train":p.tensors(rows)}
        hidden,policy,proof=p.hidden_cache(model,groups,data)
        with torch.no_grad():z,logits=model.both(data["train"]["x"])
        self.assertTrue(torch.equal(logits,policy["train"]))
        self.assertTrue(torch.equal(z,model.layers[5](model.layers[4](hidden["train"])).squeeze(-1)))
        before={key:value.clone() for key,value in model.state_dict().items()};trainable=[value for value in model.parameters() if value.requires_grad]
        self.assertEqual([name for name,value in model.named_parameters() if value.requires_grad],list(p.TRAINABLE));self.assertEqual(sum(value.numel() for value in trainable),65)
        optimizer=torch.optim.AdamW(trainable,lr=p.CONTRACT["lr"],weight_decay=0.,foreach=False)
        prediction=data["train"]["d"]+data["train"]["r"]*model.layers[5](model.layers[4](hidden["train"])).squeeze(-1)
        error=prediction-data["train"]["y"];loss=torch.nn.functional.huber_loss(prediction,data["train"]["y"],delta=4.)
        manual=torch.where(error.abs()<4.,.5*error.square(),4.*(error.abs()-2.)).mean();self.assertTrue(torch.equal(loss,manual))
        loss.backward();self.assertTrue(all(value.grad is None for name,value in model.named_parameters() if name not in p.TRAINABLE))
        torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
        self.assertEqual(optimizer.state_dict()["param_groups"][0]["params"],[0,1]);self.assertEqual(set(optimizer.state_dict()["state"]),{0,1})
        self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in before.items() if key not in p.TRAINABLE))
        self.assertTrue(p.policy_equal(model,hidden,policy));self.assertNotEqual(self.prior.weight_digest(model.state_dict()),self.source.id)

    def test_03_native_policy_logits_identical_with_changed_value_head(self):
        torch=self.torch;model=self.model()
        with torch.no_grad():model.layers[4].bias.add_(.01)
        payload=p.export_payload(model.state_dict(),{"input_count":127,"width":64,"rules_digest":config.RULES_DIGEST},1)
        file=self.base/"native-model.json";p.publish(file,{"model":payload,"sha256":config.digest(payload)});candidate=Model(file)
        self.assertEqual(p.model_frozen_digest(candidate),p.FROZEN_WEIGHTS);changed=0
        rows=self.groups["development"][:20]+self.groups["ordinary_development"]+self.groups["diagnostic"]
        with CpuBatchEvaluator(self.source,max_batch=32) as source,CpuBatchEvaluator(candidate,max_batch=32) as pilot:
            for row in rows:
                before=source.evaluate_many([row["state"]])[0];after=pilot.evaluate_many([row["state"]])[0]
                self.assertEqual(before["policy_logits"],after["policy_logits"]);changed+=before["value_player0"]!=after["value_player0"]
        self.assertGreater(changed,0)

    def test_04_recovery_full_state_refusals_and_no_optimizer_completed_rerun(self):
        torch=self.torch;fresh=self.base/"fresh";resume=self.base/"resume"
        p.train(fresh,fixture=True);p.inspect_run(fresh,fixture=True)
        def interrupt(value):
            if value["event"]=="checkpoint_committed" and value["step"]==2:raise RuntimeError("owner durable interruption")
        with self.assertRaisesRegex(RuntimeError,"owner durable interruption"):p.train(resume,fixture=True,on_event=interrupt)
        with sqlite3.connect(resume/"catalog.sqlite3") as db:prefix=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        result=p.train(resume,fixture=True);proof=p.inspect_run(resume,fixture=True)
        with sqlite3.connect(resume/"catalog.sqlite3") as db:self.assertEqual(prefix,db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id LIMIT 1").fetchall())
        left=torch.load(fresh/p.verify_run(fresh,fixture=True)["checkpoints"][-1]["name"],map_location="cpu",weights_only=True)
        right=torch.load(resume/proof["checkpoints"][-1]["name"],map_location="cpu",weights_only=True)
        def equal(a,b):
            if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a:equal(a[key],b[key])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(type(a),type(b));self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):equal(x,y)
            else:self.assertEqual(a,b)
        equal(left,right)
        for mutation in (lambda s:s.update(draws=True),lambda s:s["model"]["policy.bias"].add_(.01),lambda s:s["optimizer"]["param_groups"][0].update(lr=.001),lambda s:s["optimizer"]["state"][0].update(exp_avg=torch.zeros(1))):
            bad=copy.deepcopy(right);mutation(bad)
            with self.assertRaises(ValueError):p.validate_state(bad,right["identity"],self.model())
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in resume.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot();original=torch.optim.AdamW.step
        def prohibited(*args,**kwargs):raise AssertionError("Completed pilot performed optimizer work")
        torch.optim.AdamW.step=prohibited
        try:self.assertEqual(result,p.train(resume,fixture=True))
        finally:torch.optim.AdamW.step=original
        self.assertEqual(before,snapshot());self.assertEqual(proof["checkpoints_verified"],3)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ValuePilotLearningAcceptance))
    report={"schema":"value-pilot-learning-author-acceptance-v1","passed":result.wasSuccessful(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),
            "methods":unittest.defaultTestLoader.getTestCaseNames(ValuePilotLearningAcceptance),"inputs":{name:p.sha(ROOT/name) for name in p.SOURCES+p.DEPENDENCIES}}
    p.old.teacher.atomic_json(ROOT/"build/value-pilot-learning-author.json",report)
    raise SystemExit(not result.wasSuccessful())
