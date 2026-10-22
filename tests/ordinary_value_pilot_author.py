"""Owner checks for protected labels and the bounded frozen-policy head update."""
import bisect
import copy
from contextlib import closing
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_value_pilot as p


class OrdinaryValuePilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.groups=p.load_data();cls.base=ROOT/"build/ordinary-value-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True)

    def test_01_typed_data_origin_closure_and_exact_mixture_sampler(self):
        self.assertEqual(self.manifest["groups"],dict(high=24,low=24,rehearsal=7539));self.assertEqual(self.manifest["paired_starts"],dict(high=12,low=12));self.assertEqual(len(self.manifest["excluded_descendants"]),14)
        registry=p.read(p.REGISTRY);self.assertEqual(len(registry["rows"]),6);self.assertEqual(len(registry["validation_pool_families"]),100);self.assertEqual(registry["validation_pool_unique_quotients"],38)
        self.assertFalse(set(row["family"] for values in self.groups.values() for row in values)&set(registry["families"]))
        sampler=p.sampler_for(self.groups);rng=random.Random(20261028);literal=random.Random(20261028);batch=p.sample_batch(rng,sampler);expected={}
        for name in ("high","low"):
            keys=sorted(sampler[name]);expected[name]=[literal.choice(sampler[name][literal.choice(keys)]) for _ in range(16)]
        expected["rehearsal"]=[bisect.bisect_right(sampler["rehearsal_cdf"],literal.random()*sampler["rehearsal_cdf"][-1]) for _ in range(32)]
        self.assertEqual(batch,expected);self.assertEqual(rng.getstate(),literal.getstate())
        odd=next(row for row in self.groups["rehearsal"] if row["target_player0"]%2);self.assertEqual(p.target(odd)["y"],odd["target_player0"])
        bad=copy.deepcopy(self.groups["high"][0]);bad["target_player0"]+=1
        with self.assertRaises(ValueError):p.target(bad)
        for values in self.groups.values():
            for row in values[:40]:
                t=p.target(row);self.assertAlmostEqual(t["d"]+t["r"]*t["z"],t["y"])
        self.assertNotIn("torch",sys.modules)

    def test_02_protected_label_prefix_recovery_fullQ_and_completed_refusals(self):
        path=self.base/"labels"
        def interrupt(event):
            if event["event"]=="ordinary_value_h10_committed" and event["completed"]==1:raise RuntimeError("owner label interruption")
        with self.assertRaisesRegex(RuntimeError,"owner label interruption"):p.run_labels(path,fixture=True,on_event=interrupt)
        with closing(sqlite3.connect(path/"work.sqlite3")) as db:prefix=db.execute("SELECT state,body,hash FROM h10_attempts ORDER BY state").fetchall()
        body=p.run_labels(path,fixture=True);rows,stores,closed=p.load_labels(path);self.assertEqual(body,closed)
        with closing(sqlite3.connect(path/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT state,body,hash FROM h10_attempts ORDER BY state LIMIT 1").fetchall())
        self.assertEqual(len(rows),2)
        for stage in ("h8","h10","h12"):
            self.assertEqual(len(stores[stage+"_attempts"]),2)
            for key,item in stores[stage+"_labels"].items():
                self.assertEqual(item["root_sha256"],config.digest(rows[key]["root"]))
                self.assertEqual(len(item["label"]["action_scores"]),rows[key]["legal_mask"].bit_count());self.assertEqual(item["label"]["choice_depth"],int(stage[1:]))
        def snapshot(where):return {f.name:(p.sha(f),f.stat().st_mtime_ns) for f in where.iterdir() if f.is_file() and not f.name.endswith(("-wal","-shm"))}
        before=snapshot(path)
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Completed labels searched")):self.assertEqual(body,p.run_labels(path,fixture=True));self.assertEqual(body,p.load_labels(path)[2])
        self.assertEqual(before,snapshot(path));bad=self.base/"bad-label-marker";shutil.copytree(path,bad);(bad/"manifest.json").unlink()
        with closing(sqlite3.connect(bad/"work.sqlite3")) as db,db:
            value=p.metadata(db,"complete");value["total_new_nodes"]=0;db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(value),config.digest(value)))
        with self.assertRaises(ValueError):p.run_labels(bad,fixture=True)
        self.assertFalse((bad/"manifest.json").exists());self.assertNotIn("torch",sys.modules)
        cli_path=self.base/"cli-labels"
        cli=subprocess.run([sys.executable,str(ROOT/"tools/ordinary-value-pilot.py"),"labels","--run",str(cli_path),"--fixture","--roots","1"],cwd=ROOT,text=True,capture_output=True,timeout=120)
        self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr);self.assertEqual(p.load_labels(cli_path)[2]["roots"],1)
        from congklak.puct import CancellationToken
        token=CancellationToken();token.cancel();self.assertTrue(p.teacher.stopped(token));self.assertTrue(p.teacher.stopped(lambda:True));self.assertFalse(p.teacher.stopped(None))

    def test_03_head65_cache_normalized_huber_and_native_policy_invariance(self):
        torch,prior=p.runtime();torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        prior.deterministic("cpu");model=p.evaluator();model.load_state_dict({key:torch.tensor(value,dtype=torch.float32) for key,value in p.Model(p.SOURCE).weights.items()});groups={name:rows[:16] for name,rows in self.groups.items()};data={name:p.tensors(rows) for name,rows in groups.items()};hidden,policy,proof=p.cache.hidden_cache(model,groups,data)
        before={key:value.clone() for key,value in model.state_dict().items()};trainable=[x for x in model.parameters() if x.requires_grad];self.assertEqual(sum(x.numel() for x in trainable),65);self.assertEqual([name for name,x in model.named_parameters() if x.requires_grad],list(p.TRAINABLE))
        h=torch.cat(list(hidden.values()));y=torch.cat([data[name]["z"] for name in groups]);z=model.layers[5](model.layers[4](h)).squeeze(-1);error=z-y;loss=torch.nn.functional.huber_loss(z,y,delta=.1);manual=torch.where(error.abs()<.1,.5*error.square(),.1*(error.abs()-.05)).mean();self.assertTrue(torch.equal(loss,manual))
        optimizer=torch.optim.AdamW(trainable,lr=.0001,weight_decay=0.,foreach=False);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step();self.assertEqual(optimizer.state_dict()["param_groups"][0]["params"],[0,1]);self.assertTrue(all(torch.equal(value,model.state_dict()[key]) for key,value in before.items() if key not in p.TRAINABLE));self.assertTrue(p.cache.policy_equal(model,hidden,policy))
        identity=dict(input_count=127,width=64,rules_digest=config.RULES_DIGEST);payload=p.export_payload(model.state_dict(),identity,1);file=self.base/"changed-head.json";p.publish(file,dict(model=payload,sha256=config.digest(payload)))
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        with CpuBatchEvaluator(p.SOURCE,max_batch=1) as old,CpuBatchEvaluator(file,max_batch=1) as changed:
            for row in self.groups["high"]+self.groups["low"]:self.assertEqual(old.evaluate_many([row["state"]])[0]["policy_logits"],changed.evaluate_many([row["state"]])[0]["policy_logits"])

    def test_04_exact_checkpoint_recovery_corruption_and_no_runtime_completed_reader(self):
        fresh,resume=self.base/"fresh",self.base/"resume";p.train(fresh,fixture=True);proof=p.inspect_run(fresh,fixture=True);torch,prior=p.runtime()
        def interrupt(event):
            if event["event"]=="checkpoint_committed" and event["step"]==2:raise RuntimeError("owner durable checkpoint interruption")
        with self.assertRaisesRegex(RuntimeError,"owner durable checkpoint interruption"):p.train(resume,fixture=True,on_event=interrupt)
        with closing(sqlite3.connect(resume/"catalog.sqlite3")) as db:prefix=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        result=p.train(resume,fixture=True);after=p.inspect_run(resume,fixture=True)
        with closing(sqlite3.connect(resume/"catalog.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id LIMIT 1").fetchall())
        left=torch.load(fresh/proof["checkpoints"][-1]["name"],map_location="cpu",weights_only=True);right=torch.load(resume/after["checkpoints"][-1]["name"],map_location="cpu",weights_only=True)
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
        for mutation in (lambda s:s.update(draws=True),lambda s:s["model"]["policy.bias"].add_(.1),lambda s:s["optimizer"]["param_groups"][0].update(lr=.001),lambda s:s["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),lambda s:s.update(sampler_rng=random.Random(0).getstate())):
            state=copy.deepcopy(right);mutation(state)
            with self.assertRaises(ValueError):p.validate_state(state,right["identity"],p.evaluator())
        def snapshot():return {f.name:(p.sha(f),f.stat().st_mtime_ns) for f in resume.iterdir() if f.is_file() and not f.name.endswith(("-wal","-shm"))}
        before=snapshot()
        with patch.object(p,"runtime",side_effect=AssertionError("Completed training loaded runtime")),patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Completed run optimized")),patch.object(p.Native,"analyze",side_effect=AssertionError("Completed run searched")):
            self.assertEqual(result,p.train(resume,fixture=True));self.assertTrue(p.verify_run(resume,fixture=True)["passed"])
        self.assertEqual(before,snapshot());self.assertFalse(torch.cuda.is_initialized());self.assertEqual(after["checkpoints_verified"],3)
        cli_path=self.base/"cli-train"
        cli=subprocess.run([sys.executable,str(ROOT/"tools/ordinary-value-pilot.py"),"train","--run",str(cli_path),"--fixture"],cwd=ROOT,text=True,capture_output=True,timeout=180)
        self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr);self.assertTrue(p.verify_run(cli_path,fixture=True)["passed"])


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrdinaryValuePilotAuthor))
    files=p.SOURCES+tuple(name for name in p.DEPENDENCIES if (ROOT/name).exists())+tuple(path.relative_to(ROOT).as_posix() for path in (p.BASELINE,p.REGISTRY,p.PLAN,p.DATA,p.ROWS))
    report=dict(schema="ordinary-value-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(OrdinaryValuePilotAuthor),inputs={name:p.sha(ROOT/name) for name in files},fixture_path=OrdinaryValuePilotAuthor.base.relative_to(ROOT).as_posix())
    p.teacher.teacher.atomic_json(ROOT/"build/ordinary-value-author.json",report)
    raise SystemExit(not result.wasSuccessful())
