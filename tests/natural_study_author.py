"""Author checks for natural midgame ownership and stage-aware H8/H10 labels."""
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
from congklak import config,natural_study as s
from congklak.native import Native,flatten
from congklak.puct_cpu_inference import CpuBatchEvaluator


class NaturalStudyAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s.plan();cls.registry=s.read(s.REGISTRY);cls.native=Native();cls.settings=s.settings_for(True,8,2)
        cls.base=ROOT/"build/natural-study-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        with CpuBatchEvaluator(s.MODEL,cache_capacity=0,max_batch=1) as evaluator:
            cls.pool={i:s.prior.generate_family(i,cls.settings,cls.native,evaluator) for i in range(8)}
        cls.rows,cls.references,cls.selection=s.select(cls.pool,cls.settings,cls.registry)

    def test_01_prior_protection_and_unchanged_followup_references(self):
        baseline=s.checked_baseline();self.assertEqual(baseline["inputs_count"],2060)
        self.assertEqual(len(self.registry["followups"]),15)
        old_rows,old_labels,_,_=s.prior.load_rows()
        self.assertTrue({r["quotient"] for r in old_rows.values()}<=set(self.registry["quotients"]))
        with closing(sqlite3.connect(s.prior.DEFAULT/"work.sqlite3")) as db:
            old_pool=s.table(db,"families","idx")
        self.assertTrue({f["family"] for f in old_pool.values()}<=set(self.registry["families"]))
        for key,reference in self.registry["followups"].items():
            self.assertEqual(reference["root"],old_rows[key]);self.assertEqual(reference["h8"],old_labels[key])
            self.assertEqual(reference["h8"]["root_sha256"],config.digest(reference["root"]))
        self.assertEqual(s.sha(s.MODEL),s.MODEL_SHA)

    def test_02_natural_ordinary_pool_ownership_and_collision_closure(self):
        fresh=[row for row in self.rows.values() if row["study_role"]=="fresh"]
        self.assertGreaterEqual(len(fresh),1)
        self.assertEqual(len({r["family"] for r in fresh}),len(fresh));self.assertEqual(len({r["quotient"] for r in fresh}),len(fresh))
        for row in fresh:
            self.assertEqual(row["band"],"50-74");self.assertNotIn(row["quotient"],self.registry["quotients"])
            self.assertNotIn(row["family"],self.registry["families"])
            p=row["root"]["provenance"]["start"]
            for action in row["root"]["provenance"]["moves_prefix"]:
                p=self.native.normalize(flatten(p),actor=p["controller"],total=98)["position"]
                p=self.native.apply(flatten(p),actor=p["controller"],action=action,total=98,trace=False)["position"]
            self.assertEqual(p,row["root"]["provenance"]["raw_state"])
            normal=self.native.normalize(flatten(row["state"]),actor=row["state"]["controller"],total=98)
            facts=[self.native.apply(flatten(row["state"]),actor=row["state"]["controller"],action=a,total=98,trace=False) for a in normal["legal_actions"]]
            self.assertTrue(all(not t["passes"] and t["reason"] not in ("own_capture","own_store") for t in facts))
        root=copy.deepcopy(fresh[0]["root"]);other=copy.deepcopy(root);other["cohort"]="validation" if root["cohort"]=="training_candidate" else "training_candidate"
        blocked,_,evidence=s.select({0:{"candidates":[root]},1:{"candidates":[other]}},self.settings,self.registry)
        self.assertFalse(any(r["study_role"]=="fresh" for r in blocked.values()));self.assertEqual(evidence["cross_cohort_collisions"],1)

    def test_03_full_stage_targets_inheritance_and_unpaired_denominator(self):
        run=self.base/"fresh";result=s.run(run,fixture=True,families=8,followups=2)
        rows,h8,h10,a8,a10,_=s.load_rows(run)
        self.assertEqual(len(a10),len(rows));self.assertEqual(result["paired_complete"],len(rows))
        for key,row in rows.items():
            s.validate_label(h8[key],row,8,self.native);s.validate_label(h10[key],row,10,self.native)
            self.assertEqual(len(h10[key]["label"]["action_scores"]),row["legal_mask"].bit_count())
            if row["study_role"]=="followup":
                self.assertEqual(h8[key],self.registry["followups"][key]["h8"])
                self.assertEqual(a8[key]["status"],"inherited_complete")
            with self.assertRaises(Exception):s.validate_label(h10[key],row,8,self.native)
            bad_options=copy.deepcopy(h10[key]);bad_options["report"]["search_options"]["tt_mb"]=0
            with self.assertRaises(ValueError):s.validate_label(bad_options,row,10,self.native)
        key=next(k for k,r in rows.items() if r["study_role"]=="fresh");row=rows[key]
        result=self.native.analyze(flatten(row["state"]),actor=row["state"]["controller"],total=98,max_depth=8,node_budget=1,time_ms=0,memory_mb=512,tt_mb=16,use_pvs=True,keep_history=False)
        self.assertEqual(result["stop_reason"],"node_budget")
        rejected=s.attempt_for(None,row,"h8",result);changed_a8=copy.deepcopy(a8);changed_a8[key]=rejected
        changed_h8={k:v for k,v in h8.items() if k!=key};settings=copy.deepcopy(self.settings);settings["h8"]["node_budget"]=1
        s.validate_attempts(rows,changed_a8,changed_h8,"h8",settings,self.registry,self.native)
        metrics=s.counts(rows,changed_h8,h10,changed_a8,a10)
        self.assertEqual(metrics["h10_labels"],len(rows));self.assertEqual(metrics["paired_complete"],len(rows)-1)
        self.assertEqual(metrics["h8_rejected"],1)

    def test_04_interrupted_commit_resume_closed_rerun_and_corruption(self):
        run=self.base/"resume"
        def interrupt(value):
            if value["event"]=="natural_study_h10_committed" and value["completed"]==1:raise RuntimeError("owner committed interruption")
        with self.assertRaisesRegex(RuntimeError,"owner committed interruption"):
            s.run(run,fixture=True,families=8,followups=2,on_event=interrupt)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            prefix=db.execute("SELECT state,body,hash FROM h10_attempts ORDER BY state").fetchall()
        result=s.run(run,fixture=True,families=8,followups=2)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            self.assertEqual(prefix,db.execute("SELECT state,body,hash FROM h10_attempts ORDER BY state LIMIT 1").fetchall())
        def snapshot():
            return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in run.iterdir() if p.is_file() and not p.name.endswith(("-wal","-shm"))}
        before=snapshot();original=Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError("Completed rerun searched again")
        Native.analyze=prohibited
        try:self.assertEqual(result,s.run(run,fixture=True,families=8,followups=2))
        finally:Native.analyze=original
        self.assertEqual(before,snapshot())
        rows,h8,h10,a8,a10,_=s.load_rows(run);bad=copy.deepcopy(a10);key=sorted(bad)[0];bad[key]["work"]={k:0 for k in bad[key]["work"]}
        with self.assertRaises(ValueError):s.validate_attempts(rows,bad,h10,"h10",self.settings,self.registry,self.native)
        inherited=next(k for k,r in rows.items() if r["study_role"]=="followup");bad_h8=copy.deepcopy(h8);bad_h8[inherited]["root_sha256"]="0"*64
        with self.assertRaises(ValueError):s.validate_attempts(rows,a8,bad_h8,"h8",self.settings,self.registry,self.native)
        broken=self.base/"incomplete";broken.mkdir()
        with closing(sqlite3.connect(run/"work.sqlite3")) as source,closing(sqlite3.connect(broken/"work.sqlite3")) as destination:source.backup(destination)
        shutil.copy2(run/"manifest.json",broken/"manifest.json")
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            marker=copy.deepcopy(result);marker["selected_fresh"]+=1
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        (broken/"manifest.json").unlink()
        with self.assertRaisesRegex(ValueError,"counters differ"):s.run(broken,fixture=True,families=8,followups=2)
        self.assertFalse((broken/"manifest.json").exists(),"Invalid completed marker published an export")
        bad_identity=copy.deepcopy(result["identity"]);bad_identity["schema"]="wrong-schema"
        marker=copy.deepcopy(result);marker["identity"]=bad_identity
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            with db:
                db.execute("UPDATE meta SET body=?,hash=? WHERE key='identity'",(config.canonical(bad_identity),config.digest(bad_identity)))
                db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        s.publish(broken/"manifest.json",marker)
        with self.assertRaisesRegex(ValueError,"identity fields/schema/reserve"):s.load_rows(broken)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NaturalStudyAuthor))
    report={"schema":"natural-study-author-acceptance-v1","passed":result.wasSuccessful(),"tests_run":result.testsRun,
            "failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),"methods":unittest.defaultTestLoader.getTestCaseNames(NaturalStudyAuthor),
            "inputs":{name:s.sha(ROOT/name) for name in s.SOURCES+("build/natural-study-baseline.json","build/natural-study-registry.json","build/natural-study-plan.json")}}
    s.teacher.atomic_json(ROOT/"build/natural-study-author.json",report)
    raise SystemExit(not result.wasSuccessful())
