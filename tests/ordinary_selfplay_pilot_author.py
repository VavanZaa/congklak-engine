"""Owner checks for data-only ordinary collection, role protection and recovery."""
import copy
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import sys
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ordinary_selfplay_pilot as p


class OrdinarySelfplayPilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/"build/ordinary-selfplay-author-fixtures"/uuid.uuid4().hex;cls.path=cls.base/"run";cls.base.mkdir(parents=True)

    def test_01_roles_latest96_and_no_training_blanket_mask(self):
        registry=p.read(p.REGISTRY)
        self.assertEqual(len(registry["latest96_games"]),96)
        self.assertEqual(registry["latest96_state_observations"],3054)
        self.assertFalse(registry["metadata9000"]["blanket_rollout_evaluation_ban"])
        self.assertEqual(registry["metadata9000"]["rows"],9000)
        self.assertEqual(registry["prior_selfplay_rows"],276);self.assertEqual(registry["prior_selfplay_unique_rawstates"],231)
        self.assertEqual(registry["prior_midgame_fullfamilies"],512)
        self.assertTrue(set(registry["true_evaluation_quotients"])<=set(registry["novelty_quotients"]))
        self.assertTrue(set(registry["true_evaluation_families"])<=set(registry["novelty_families"]))
        self.assertGreater(len(registry["novelty_quotients"]),len(registry["true_evaluation_quotients"]))
        self.assertEqual(p.CONTRACT["minimum_usable_roots"],{kind:8 for kind in p.STRATA})
        with self.assertRaises(ValueError):p.settings_for(1)
        self.assertNotIn("torch",sys.modules)

    def test_02_collection_resume_strict_ordinary_unique_family_and_transitive_closure(self):
        def interrupt(event):
            if event["event"]=="ordinary_selfplay_collection_committed" and event["completed"]==2:raise RuntimeError("owner collection interruption")
        with self.assertRaisesRegex(RuntimeError,"owner collection interruption"):p.collect(self.path,fixture=True,on_event=interrupt)
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as db:prefix=db.execute("SELECT idx,body,hash FROM families ORDER BY idx").fetchall()
        body=p.collect(self.path,fixture=True);roots,closed=p.load_rows(self.path)
        self.assertEqual(body,closed);self.assertEqual(body["families"],32)
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT idx,body,hash FROM families ORDER BY idx LIMIT 2").fetchall())
        self.assertEqual(len(roots),len({row["family"] for row in roots.values()}))
        for row in roots.values():
            self.assertIn(row["stratum"],p.STRATA);self.assertGreaterEqual(row["legal_mask"].bit_count(),2)
            self.assertEqual(p.pilot._mechanism(row["state"],p.Native())[1],row["stratum"])
        self.assertGreaterEqual(body["rollout_protection"]["new_validation_unique_quotients"],sum(row["cohort"]=="validation" for row in roots.values()))
        settings=p.settings_for(True);pool={}
        for index,(family,qs) in enumerate((("A",["q1"]),("B",["q1","q2"]),("C",["q2"]))):
            rows=[dict(family=family,cohort="training_candidate",stratum=p.STRATA[0],quotient=q,state_key=family+q,legal_mask=3,provenance=dict(family_index=index,step=step)) for step,q in enumerate(qs)]
            pool[index]=dict(family=family,cohort="training_candidate",candidates=rows)
        selected,selection,protection=p.select(pool,settings,dict(novelty_quotients=["q1"],novelty_families=[],true_evaluation_quotients=[],true_evaluation_families=[]))
        self.assertFalse(selected);self.assertEqual(set(selection["blocked_families"]),{"A","B","C"})

    def test_03_decision_resume_terminal_pairs_targets_and_priority_cap(self):
        interrupted=p.generate(self.path,fixture=True,decision_limit=2);self.assertEqual(interrupted["committed_decisions"],2)
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as db:prefix=db.execute("SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice").fetchall()
        body=p.generate(self.path,fixture=True);closed,rows=p.load_replay(self.path)
        self.assertEqual(body,closed);self.assertEqual(body["summary"]["games"],4);self.assertEqual(body["summary"]["terminal_games"],4)
        self.assertTrue(body["summary"]["fixture_rollout_panel"]);self.assertGreater(len(rows),0)
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as db:
            self.assertEqual(prefix,db.execute("SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice LIMIT 2").fetchall());games=p.table(db,"sp_games")
        for row in rows:
            self.assertEqual(row["policy_target"],[v/4 for v in row["root_visits"]]);self.assertEqual(sum(row["root_visits"]),4)
            self.assertEqual(row["is_start_record"],row["choice"]==0);self.assertGreaterEqual(row["legal_mask"].bit_count(),2)
            self.assertTrue(row["stratum"].endswith("/ordinary"));self.assertGreaterEqual(sum(map(sum,row["state"]["pits"])),25)
            self.assertEqual(row["terminal_return_player0"],games[row["gameid"]]["terminal_margin"])
            f=row["features"];self.assertGreaterEqual((row["terminal_return_player0"]-f["d"])/f["r"],-1);self.assertLessEqual((row["terminal_return_player0"]-f["d"])/f["r"],1)
        retained={row["id"] for row in rows}
        for root in body["summary"]["usable_starts"].values():self.assertEqual(root["usable"],all(key in retained for key in root["required_record_ids"]))
        self.assertEqual(body["rollout_identity"]["collection_file_sha256"],p.sha(self.path/"collection-manifest.json"))
        self.assertEqual(body["rollout_identity"]["protection_file_sha256"],p.sha(self.path/"rollout-protection.json"))

    def test_04_closed_corruption_refusal_and_no_work_stable_rerun(self):
        def snapshot():return {file.name:(p.sha(file),file.stat().st_mtime_ns) for file in self.path.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot()
        with patch.object(p.previous.quality_search,"search",side_effect=AssertionError("Completed data rerun searched")),patch.object(p.Native,"analyze",side_effect=AssertionError("Data reader searched teacher")):
            body=p.run(self.path,fixture=True);self.assertEqual(body,p.load_replay(self.path)[0])
        self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)
        for stage,filename,key in (("collection","collection-manifest.json","collection_complete"),("rollouts","replay-manifest.json","replay_complete")):
            path=self.base/("bad-"+stage);shutil.copytree(self.path,path);(path/filename).unlink()
            with closing(sqlite3.connect(path/"work.sqlite3")) as db,db:
                row=db.execute("SELECT body FROM meta WHERE key=?",(key,)).fetchone();value=p.json.loads(row[0]);value["status"]="corrupt_complete"
                db.execute("UPDATE meta SET body=?,hash=? WHERE key=?",(config.canonical(value),config.digest(value),key))
            with self.assertRaises(ValueError):p.run(path,fixture=True,stage=stage)
            self.assertFalse((path/filename).exists())
        path=self.base/"bad-action";shutil.copytree(self.path,path)
        with closing(sqlite3.connect(path/"work.sqlite3")) as db,db:
            gameid,choice,body=db.execute("SELECT gameid,choice,body FROM sp_decisions ORDER BY gameid,choice LIMIT 1").fetchone();row=p.json.loads(body);row["action"]=True
            db.execute("UPDATE sp_decisions SET body=?,hash=? WHERE gameid=? AND choice=?",(config.canonical(row),config.digest(row),gameid,choice))
        with self.assertRaises(ValueError):p.load_replay(path)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrdinarySelfplayPilotAuthor))
    report=dict(schema="ordinary-selfplay-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(OrdinarySelfplayPilotAuthor),inputs={name:p.sha(ROOT/name) for name in p.SOURCES+p.DEPENDENCIES+tuple(path.relative_to(ROOT).as_posix() for path in (p.BASELINE,p.REGISTRY,p.PLAN))},fixture_path=OrdinarySelfplayPilotAuthor.base.relative_to(ROOT).as_posix())
    p.histories.teacher.atomic_json(ROOT/"build/ordinary-selfplay-author.json",report)
    raise SystemExit(not result.wasSuccessful())
