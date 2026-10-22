"""Author checks for the exact five capped H14 roots and the two-tier ledger."""
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
import sys
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,horizon_finish as p
from congklak.native import Native


class HorizonFinishAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.plan();cls.registry=p.read(p.REGISTRY);cls.settings=p.settings_for(True,1);cls.native=Native()
        cls.base=ROOT/"build/horizon-finish-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.fresh_run=cls.base/"fresh";p.run(cls.fresh_run,fixture=True,roots=1)
        cls.rows,cls.stores,cls.manifest=p.load_rows(cls.fresh_run)

    def test_01_exact_five_and_all_old_references_unchanged(self):
        self.assertEqual(p.checked_baseline()["inputs_count"],2314)
        rows,h8,h10,h12,h14,a12,a14,_=p.old.load_rows()
        self.assertEqual(self.registry["state_keys"],sorted(k for k,a in a14.items() if a["status"]=="budget_rejected"))
        self.assertEqual(len(self.registry["state_keys"]),5);self.assertEqual(rows,self.rows)
        for name,value in zip(p.PRIOR_TABLES,(h8,h10,h12,a12,a14,h14)):self.assertEqual(self.stores[name],value)
        self.assertEqual(p.CONTRACT["tier1"]["node_budget"],250000000);self.assertEqual(p.CONTRACT["tier2"]["node_budget"],1000000000)
        self.assertEqual(p.settings_for(False),p.CONTRACT);self.assertNotEqual(self.settings,p.CONTRACT)
        self.assertTrue(p.CONTRACT["investigation_only"]);self.assertFalse(p.CONTRACT["training_performed"])

    def test_02_real_initial_rejection_fallback_and_overlap(self):
        p.validate_tables(self.rows,self.stores,self.settings,self.registry,self.native,closed=True)
        key=self.registry["state_keys"][0]
        self.assertEqual(self.stores["tier1_attempts"][key]["status"],"budget_rejected")
        self.assertEqual(self.stores["tier1_attempts"][key]["completed_depth"],13)
        self.assertNotIn(key,self.stores["tier1_labels"])
        self.assertEqual(self.stores["tier2_attempts"][key]["status"],"complete")
        self.assertEqual(self.stores["tier2_attempts"][key]["completed_depth"],14)
        self.assertEqual(self.manifest["new_h14_labels"],1);self.assertEqual(self.manifest["combined_h14_labels"],2)
        for tier in ("tier1","tier2"):
            attempt=self.stores[tier+"_attempts"][key]
            self.assertEqual([r["choice_depth"] for r in attempt["history"]],list(range(1,attempt["completed_depth"]+1)))
            self.assertEqual(attempt["tier"],tier);self.assertEqual(attempt["node_budget"],self.settings[tier]["node_budget"])
        self.assertEqual(self.manifest["status"],"complete");self.assertTrue(self.manifest["scope_finished"])

    def test_03_corrupt_tier_label_history_and_closed_marker_refused(self):
        key=self.registry["state_keys"][0];row=self.rows[key]
        bad=copy.deepcopy(self.stores["tier2_attempts"][key]);bad["tier"]="tier1"
        with self.assertRaises(ValueError):p.validate_attempt(row,bad,self.stores["tier2_labels"][key],"tier2",self.settings,self.native)
        partial=self.stores["tier1_attempts"][key];fake=p.item_for(partial["history"][-1],row)
        with self.assertRaises(ValueError):p.validate_attempt(row,partial,fake,"tier1",self.settings,self.native)
        altered=copy.deepcopy(self.stores);altered["tier2_attempts"][key]["history"][0]["search_options"]["tt_mb"]=0
        with self.assertRaises(ValueError):p.validate_tables(self.rows,altered,self.settings,self.registry,self.native,closed=True)
        missing=copy.deepcopy(self.stores);missing["tier1_attempts"]={}
        with self.assertRaises(ValueError):p.validate_tables(self.rows,missing,self.settings,self.registry,self.native,closed=True)
        broken=self.base/"bad-marker";broken.mkdir()
        with closing(sqlite3.connect(self.fresh_run/"work.sqlite3")) as source,closing(sqlite3.connect(broken/"work.sqlite3")) as destination:source.backup(destination)
        marker=copy.deepcopy(self.manifest);marker["new_h14_labels"]+=1
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):p.run(broken,fixture=True,roots=1)
        self.assertFalse((broken/"manifest.json").exists())
        identity_bad=copy.deepcopy(self.manifest["identity"]);identity_bad["schema"]="wrong"
        marker_bad=copy.deepcopy(self.manifest);marker_bad["identity"]=identity_bad
        with self.assertRaises(ValueError):p.validate_complete_marker(marker_bad,identity_bad,self.rows,self.stores,self.registry)

    def test_04_committed_tier_recovery_and_zero_search_rerun(self):
        run=self.base/"resume"
        def interrupt(value):
            if value["event"]=="horizon_finish_tier1_committed" and value["completed"]==1:raise RuntimeError("owner durable interruption")
        with self.assertRaisesRegex(RuntimeError,"owner durable interruption"):p.run(run,fixture=True,roots=1,on_event=interrupt)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:prefix=db.execute("SELECT state,body,hash FROM tier1_attempts ORDER BY state").fetchall()
        result=p.run(run,fixture=True,roots=1)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT state,body,hash FROM tier1_attempts ORDER BY state").fetchall())
        rows,stores,_=p.load_rows(run);self.assertEqual(rows,self.rows)
        for tier in ("tier1","tier2"):
            for key,attempt in stores[tier+"_attempts"].items():
                reference=self.stores[tier+"_attempts"][key];self.assertEqual(attempt["status"],reference["status"])
                self.assertEqual([p.q_signature(p.item_for(r,rows[key])) for r in attempt["history"]],[p.q_signature(p.item_for(r,rows[key])) for r in reference["history"]])
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in run.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot();original=Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError("Completed finish reran teacher")
        Native.analyze=prohibited
        try:self.assertEqual(result,p.run(run,fixture=True,roots=1))
        finally:Native.analyze=original
        self.assertEqual(before,snapshot())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HorizonFinishAuthor))
    proof={"schema":"horizon-finish-author-acceptance-v1","passed":result.wasSuccessful(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),
           "methods":unittest.defaultTestLoader.getTestCaseNames(HorizonFinishAuthor),"inputs":{name:p.sha(ROOT/name) for name in p.SOURCES+("build/horizon-finish-baseline.json","build/horizon-finish-registry.json","build/horizon-finish-plan.json")}}
    p.teacher.atomic_json(ROOT/"build/horizon-finish-author.json",proof)
    raise SystemExit(not result.wasSuccessful())
