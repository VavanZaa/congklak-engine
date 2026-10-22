"""Author checks for exact disagreement references and complete-prefix histories."""
from contextlib import closing
import copy
from pathlib import Path
import shutil
import sqlite3
import sys
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,horizon_probe as p
from congklak.native import Native


class HorizonProbeAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.plan();cls.registry=p.read(p.REGISTRY);cls.settings=p.settings_for(True,2);cls.native=Native()
        cls.base=ROOT/"build/horizon-probe-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.fresh_run=cls.base/"fresh";p.run(cls.fresh_run,fixture=True,roots=2)
        cls.rows,cls.r8,cls.r10,cls.h12,cls.h14,cls.a12,cls.a14,cls.manifest=p.load_rows(cls.fresh_run)

    def test_01_exact_six_ownership_and_unchanged_references(self):
        baseline=p.checked_baseline();self.assertEqual(baseline["inputs_count"],2199)
        oldrows,h8,h10,_,_,_=p.old.load_rows()
        keys=sorted(k for k in oldrows if not(set(h8[k]["label"]["best_moves"])&set(h10[k]["label"]["best_moves"])))
        self.assertEqual(keys,self.registry["state_keys"]);self.assertEqual(len(keys),6)
        for key in keys:
            self.assertEqual(self.registry["rows"][key],oldrows[key]);self.assertEqual(self.registry["h8_references"][key],h8[key]);self.assertEqual(self.registry["h10_references"][key],h10[key])
        self.assertTrue(p.CONTRACT["investigation_only"]);self.assertFalse(p.CONTRACT["training_performed"])

    def test_02_complete_prefix_full_labels_and_overlap_semantics(self):
        stores={"h8_references":self.r8,"h10_references":self.r10,"h12_labels":self.h12,"h14_labels":self.h14,"h12_attempts":self.a12,"h14_attempts":self.a14}
        p.validate_tables(self.rows,stores,self.settings,self.registry,self.native,closed=True)
        rejected=0
        for stage,attempts,labels in (("h12",self.a12,self.h12),("h14",self.a14,self.h14)):
            depth=self.settings[stage]["depth"]
            for key,attempt in attempts.items():
                self.assertEqual([r["choice_depth"] for r in attempt["history"]],list(range(1,attempt["completed_depth"]+1)))
                for report in attempt["history"]:
                    if report["choice_depth"] in (8,10):
                        reference=self.r8[key] if report["choice_depth"]==8 else self.r10[key]
                        self.assertEqual(p.q_signature(p.item_for(report,self.rows[key])),p.q_signature(reference))
                if attempt["status"]=="complete":self.assertEqual(labels[key],p.item_for(attempt["history"][-1],self.rows[key]));self.assertEqual(attempt["completed_depth"],depth)
                else:
                    rejected+=1;self.assertNotIn(key,labels);self.assertLess(attempt["completed_depth"],depth)
                    self.assertEqual(attempt["work"]["nodes"],self.settings[stage]["node_budget"])
        self.assertGreaterEqual(rejected,1,"Actual first fixture root must retain its bounded H14 rejection")
        self.assertEqual(len(self.a14),len(self.rows))

    def test_03_history_partial_and_rehashed_semantic_corruption_refused(self):
        key=sorted(self.a12)[0];row=self.rows[key];attempt=self.a12[key]
        missing=copy.deepcopy(attempt);missing["history"].pop(3);missing["completed_depth"]-=1;missing["native_last_completed_depth"]-=1
        with self.assertRaises(ValueError):p.validate_attempt(row,missing,self.h12.get(key),"h12",self.settings,self.native)
        zeroed=copy.deepcopy(attempt);zeroed["work"]={name:0 for name in zeroed["work"]}
        with self.assertRaises(ValueError):p.validate_attempt(row,zeroed,self.h12.get(key),"h12",self.settings,self.native)
        wrong=copy.deepcopy(attempt);wrong["history"][0]["search_options"]["tt_mb"]=0
        with self.assertRaises(ValueError):p.validate_attempt(row,wrong,self.h12.get(key),"h12",self.settings,self.native)
        rejected_key=next(k for k,a in self.a14.items() if a["status"]=="budget_rejected")
        rejected=self.a14[rejected_key];fake=p.item_for(rejected["history"][-1],self.rows[rejected_key])
        self.assertEqual(fake["label"]["choice_depth"],13)
        with self.assertRaises(ValueError):p.validate_attempt(self.rows[rejected_key],rejected,fake,"h14",self.settings,self.native)
        broken=self.base/"bad-marker";broken.mkdir()
        with closing(sqlite3.connect(self.fresh_run/"work.sqlite3")) as source,closing(sqlite3.connect(broken/"work.sqlite3")) as destination:source.backup(destination)
        marker=copy.deepcopy(self.manifest);marker["h14_labels"]+=1
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):p.run(broken,fixture=True,roots=2)
        self.assertFalse((broken/"manifest.json").exists())

    def test_04_committed_recovery_and_completed_zero_search_rerun(self):
        run=self.base/"resume"
        def interrupt(value):
            if value["event"]=="horizon_probe_h12_committed" and value["completed"]==1:raise RuntimeError("owner durable interruption")
        with self.assertRaisesRegex(RuntimeError,"owner durable interruption"):p.run(run,fixture=True,roots=2,on_event=interrupt)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:prefix=db.execute("SELECT state,body,hash FROM h12_attempts ORDER BY state").fetchall()
        result=p.run(run,fixture=True,roots=2)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT state,body,hash FROM h12_attempts ORDER BY state LIMIT 1").fetchall())
        rows,r8,r10,h12,h14,a12,a14,_=p.load_rows(run)
        self.assertEqual(rows,self.rows);self.assertEqual(r8,self.r8);self.assertEqual(r10,self.r10)
        for stage,a,b in (("h12",a12,self.a12),("h14",a14,self.a14)):
            for key in a:
                self.assertEqual(a[key]["status"],b[key]["status"])
                self.assertEqual([p.q_signature(p.item_for(r,rows[key])) for r in a[key]["history"]],[p.q_signature(p.item_for(r,rows[key])) for r in b[key]["history"]])
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in run.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot();original=Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError("Completed probe reran teacher")
        Native.analyze=prohibited
        try:self.assertEqual(result,p.run(run,fixture=True,roots=2))
        finally:Native.analyze=original
        self.assertEqual(before,snapshot())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HorizonProbeAuthor))
    proof={"schema":"horizon-probe-author-acceptance-v1","passed":result.wasSuccessful(),"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),
           "methods":unittest.defaultTestLoader.getTestCaseNames(HorizonProbeAuthor),"inputs":{name:p.sha(ROOT/name) for name in p.SOURCES+("build/horizon-probe-baseline.json","build/horizon-probe-registry.json","build/horizon-probe-plan.json")}}
    p.teacher.atomic_json(ROOT/"build/horizon-probe-author.json",proof)
    raise SystemExit(not result.wasSuccessful())
