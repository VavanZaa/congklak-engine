"""Owner checks for the frozen-control self-play/joint pilot, without promotion."""
import copy
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,selfplay_training_pilot as p


class SelfplayTrainingPilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/"build/selfplay-training-pilot-author-fixtures"/uuid.uuid4().hex
        cls.base.mkdir(parents=True);cls.replay=cls.base/"selfplay"

    def test_01_role_aware_reservations_and_exact_start_matching(self):
        plan=p.plan();registry=p.read(p.REGISTRY)
        self.assertEqual(len(registry["starts"]),16)
        self.assertEqual(len({entry["root"]["family"] for entry in registry["starts"]}),16)
        self.assertEqual(registry["start_selection"]["counts"],p.SP["quotas"])
        self.assertEqual(registry["metadata_only_reservations"]["rows"],9000)
        self.assertEqual(registry["metadata_only_reservations"]["train_rows"],8000)
        self.assertFalse(registry["metadata_only_reservations"]["blanket_evaluation_ban"])
        self.assertTrue(all(entry["root"]["cohort"]=="training_candidate" and entry["root"]["family"] not in registry["families"] and entry["root"]["quotient"] not in registry["quotients"] for entry in registry["starts"]))
        projection=p.read(p.PROJECTION)
        self.assertTrue(set(projection["reserved_quotients"])<=set(registry["quotients"]))
        self.assertTrue(set(projection["reserved_families"])<=set(registry["families"]))
        self.assertEqual(sum(row["cohort"]=="validation" for row in registry["latest_roles"].values()),10)
        self.assertEqual(len(registry["diagnostics"]),9)
        for row in registry["latest_roles"].values():
            if row["cohort"]=="validation":self.assertIn(row["quotient"],registry["quotients"])
        self.assertEqual(p.settings_for(True)["simulations"],4)
        self.assertEqual(p.settings_for()["min_training_active"],25)
        for function in (p.settings_for,p.train_settings):
            with self.assertRaises(ValueError):function(1)
        self.assertEqual(plan["source_model_sha256"],p.SOURCE_SHA)

    def test_02_terminal_replay_visits_rng_and_completed_no_search(self):
        interrupted=p.generate(self.replay,fixture=True,decision_limit=2)
        self.assertEqual(interrupted["committed_decisions"],2)
        with sqlite3.connect(self.replay/"work.sqlite3") as db:prefix=db.execute("SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice").fetchall()
        manifest=p.generate(self.replay,fixture=True);closed,rows=p.load_replay(self.replay)
        self.assertEqual(manifest,closed);self.assertEqual(closed["summary"]["games"],32)
        with sqlite3.connect(self.replay/"work.sqlite3") as db:
            self.assertEqual(prefix,db.execute("SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice LIMIT 2").fetchall())
            games=p.table(db,"sp_games")
        self.assertGreater(closed["summary"]["terminal_games"],0)
        self.assertGreater(len(rows),0);self.assertEqual(closed["summary"]["training_admission"],"ready")
        for row in rows:
            self.assertGreaterEqual(sum(map(sum,row["state"]["pits"])),25)
            self.assertEqual(row["policy_target"],[value/4 for value in row["root_visits"]])
            self.assertEqual(sum(row["root_visits"]),4);self.assertEqual(row["terminal_return_player0"],games[row["gameid"]]["terminal_margin"])
            self.assertEqual(row["terminal_game_sha256"],config.digest(games[row["gameid"]]))
            f=row["features"];z=(row["terminal_return_player0"]-f["d"])/f["r"]
            self.assertGreaterEqual(z,-1);self.assertLessEqual(z,1)
            self.assertEqual(f["r"]>0,row["actor"]==0)
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in self.replay.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot()
        with patch.object(p.quality_search,"search",side_effect=AssertionError("Completed rerun searched")):
            self.assertEqual(manifest,p.generate(self.replay,fixture=True))
        self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)

    def test_03_rehashed_replay_corruption_refused_before_publication(self):
        source=self.replay;manifest,rows=p.load_replay(source)
        for index,mutation in enumerate((lambda row:row.update(action=True),lambda row:row["policy_target"].__setitem__(0,.123),lambda row:row.update(after=row["state"]))):
            path=self.base/("bad-"+str(index));shutil.copytree(source,path)
            with sqlite3.connect(path/"work.sqlite3") as db:
                gameid,choice,body=db.execute("SELECT gameid,choice,body FROM sp_decisions ORDER BY gameid,choice LIMIT 1").fetchone();row=p.json.loads(body);mutation(row)
                db.execute("UPDATE sp_decisions SET body=?,hash=? WHERE gameid=? AND choice=?",(config.canonical(row),config.digest(row),gameid,choice))
            with self.assertRaises(ValueError):p.load_replay(path)
        bad=self.base/"bad-marker";shutil.copytree(source,bad)
        (bad/"selfplay-manifest.json").unlink()
        with sqlite3.connect(bad/"work.sqlite3") as db:
            body=db.execute("SELECT body FROM meta WHERE key='selfplay_manifest'").fetchone()[0];value=p.json.loads(body);value["summary"]["retained_rows"]+=1
            db.execute("UPDATE meta SET body=?,hash=? WHERE key='selfplay_manifest'",(config.canonical(value),config.digest(value)))
        with self.assertRaises(ValueError):p.generate(bad,fixture=True)
        self.assertFalse((bad/"selfplay-manifest.json").exists())
        groups=p.sample_groups(rows);rng=random.Random(777);literal=random.Random(777)
        self.assertEqual(p.sample_ids(rng,groups,100),[literal.choice(groups[literal.choice(sorted(groups))]) for _ in range(100)])
        self.assertEqual(rng.getstate(),literal.getstate())

    def test_04_all8_joint_update_exact_recovery_and_cpu_only_completed_reader(self):
        fresh,resume=self.base/"fresh",self.base/"resume"
        p.train(fresh,replay_path=self.replay,fixture=True)
        proof=p.inspect_run(fresh,fixture=True);torch,prior=p.runtime()
        self.assertEqual(proof["checkpoints_verified"],3)
        self.assertEqual(set(proof["changed_tensors"]),set(p.Model(p.SOURCE).weights))
        def interrupt(event):
            if event["event"]=="checkpoint_committed" and event["step"]==2:raise RuntimeError("durable owner interruption")
        with self.assertRaisesRegex(RuntimeError,"durable owner interruption"):p.train(resume,replay_path=self.replay,fixture=True,on_event=interrupt)
        with sqlite3.connect(resume/"catalog.sqlite3") as db:prefix=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall()
        result=p.train(resume,replay_path=self.replay,fixture=True)
        after=p.inspect_run(resume,fixture=True)
        with sqlite3.connect(resume/"catalog.sqlite3") as db:self.assertEqual(prefix,db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id LIMIT 1").fetchall())
        left=torch.load(fresh/proof["checkpoints"][-1]["name"],map_location="cpu",weights_only=True)
        right=torch.load(resume/after["checkpoints"][-1]["name"],map_location="cpu",weights_only=True)
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
        for mutation in (lambda s:s.update(draws=True),lambda s:s["optimizer"]["param_groups"][0].update(lr=.01),lambda s:s["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),lambda s:s.update(sampler_rng=random.Random(0).getstate())):
            state=copy.deepcopy(right);mutation(state)
            with self.assertRaises(ValueError):p.validate_state(state,right["identity"],p.evaluator())
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in resume.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot()
        with patch.object(p,"runtime",side_effect=AssertionError("Completed run activated Torch")),patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Completed run optimized")),patch.object(p.quality_search,"search",side_effect=AssertionError("Completed run searched")):
            self.assertEqual(result,p.train(resume,replay_path=self.replay,fixture=True));self.assertTrue(p.verify_run(resume,fixture=True)["passed"])
        self.assertEqual(before,snapshot());self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfplayTrainingPilotAuthor))
    files=p.SOURCES+tuple(name for name in p.DEPENDENCIES if (ROOT/name).exists())+tuple(path.relative_to(ROOT).as_posix() for path in (p.BASELINE,p.PLAN,p.REGISTRY,p.PROJECTION) if path.exists())
    report=dict(schema="selfplay-training-pilot-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(SelfplayTrainingPilotAuthor),inputs={name:p.sha(ROOT/name) for name in files},fixture_path=SelfplayTrainingPilotAuthor.base.relative_to(ROOT).as_posix())
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/"build/selfplay-training-pilot-author.json",report)
    raise SystemExit(not result.wasSuccessful())
