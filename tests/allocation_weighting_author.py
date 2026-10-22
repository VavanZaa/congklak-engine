"""Four source-bound owner checks for metadata-first exact weighting assessment."""
from contextlib import closing
import copy
from fractions import Fraction as F
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,allocation_weighting as p
from congklak.puct import CancellationToken


class AllocationWeightingAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry,cls.rule=p.admit();cls.base=ROOT/"build/allocation-weighting-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True);cls.path=cls.base/"fresh";cls.settings=p.settings_for(True,2)
        cls.cli=subprocess.run([sys.executable,str(ROOT/"tools/validate-allocation-weighting.py"),"run","--fixture","--roots","2","--run",str(cls.path)],cwd=ROOT,text=True,capture_output=True,timeout=180);p.require(cls.cli.returncode==0,cls.cli.stdout+cls.cli.stderr);cls.rows,cls.stores,cls.manifest=p.load_rows(cls.path)

    def test_01_metadata_roles_components_exact_matching_and_reservations(self):
        inv=p.inventory();rows=self.registry["rows"];self.assertEqual((inv["eligible_roots"],inv["eligible_families"]),(62,53));self.assertEqual(rows,p.select(inv));self.assertEqual(len(rows),16);self.assertEqual(len({r["component"] for r in rows.values()}),16);self.assertEqual(len({r["family"] for r in rows.values()}),16)
        for cell in p.CELLS:self.assertEqual(sorted(r["group"] for r in rows.values() if r["cell"]==cell),["assessment","development"])
        reservations=p.read(p.RESERVATIONS)["roles"];self.assertFalse(set(reservations["development"]["eligible_quotients"])&set(reservations["assessment"]["eligible_quotients"]))
        for r in rows.values():self.assertEqual(r["source_cohort"],"training_candidate");self.assertTrue(r["future_optimizer_reserved"]);self.assertGreaterEqual(r["legal_mask"].bit_count(),2);self.assertFalse(inv["components"][r["component"]]["blocked_reasons"])
        with self.assertRaises(ValueError):p.settings_for(1,2)
        with patch.object(p,"admit",side_effect=ValueError("unfrozen")),patch.object(p.Native,"analyze",side_effect=AssertionError("Unfrozen fixture searched")):
            with self.assertRaisesRegex(ValueError,"unfrozen"):p.run(self.base/"unfrozen",fixture=True)
        with patch.object(p,"gate",side_effect=ValueError("trace incomplete")),patch.object(p.Native,"analyze",side_effect=AssertionError("Production gate failed open")):
            with self.assertRaisesRegex(ValueError,"trace incomplete"):p.run()
        self.assertFalse(p.DEFAULT.exists())

    def test_02_exact_rule_uniform_constant_drift_ties_and_missing_mass(self):
        for row in self.rule["rows"].values():
            a,b=row["h8"]["label"],row["h10"]["label"];qa={v["pit"]:v["value_player0"] for v in a["action_scores"]};qb={v["pit"]:v["value_player0"] for v in b["action_scores"]};expected=max(F(1,4),1/(1+F(abs(b["target_player0"]-a["target_player0"]),4)+F(sum(abs(qb[k]-qa[k]) for k in qa),4*len(qa))));self.assertEqual(p.fraction(row["rule"]["weight"]),expected);self.assertFalse(row["rule"]["H12_input"])
        values=[dict(state_key=str(i),weight=p.rational(F(1,4) if i%2 else 1),H12_complete=True,root_drift=p.rational(3),action_Q_drift=p.rational(2),H10_best_H12_regret_min=p.rational(0),H10_best_H12_regret_max=p.rational(4),H10_best_H12_regret_mean=p.rational(2)) for i in range(8)]
        m=p.metric(values);self.assertEqual(m["uniform"]["root_drift"],m["weighted"]["root_drift"]);self.assertFalse(p.descriptive_screen(m)["weighted_root_strictly_lower"]);self.assertFalse(p.descriptive_screen(m)["passes"])
        values[0]["H12_complete"]=False;m=p.metric(values);self.assertEqual(p.fraction(m["missing_original_weight_mass"]),1);self.assertEqual(m["missing_roots"],1);self.assertFalse(p.descriptive_screen(m)["all_assessment_complete"])
        for r in values:r["H12_complete"]=False
        self.assertIsNone(p.metric(values)["weighted"]["root_drift"])

    def test_03_real_cli_requested_depth_fullQ_original_links_and_cancel(self):
        self.assertEqual(self.cli.returncode,0);self.assertEqual(self.manifest["summary"]["complete"],2);p.validate_tables(self.rows,self.stores,self.settings,True,True)
        for key,row in self.rows.items():
            a=self.stores["h12_attempts"][key];label=self.stores["h12_labels"][key];self.assertEqual(a["requested_depth"],4);self.assertEqual(label["label"]["choice_depth"],4);self.assertEqual([r["choice_depth"] for r in a["history"]],[1,2,3,4]);self.assertEqual(label["root_sha256"],row["source_root_sha256"]);self.assertEqual(row["source_root_sha256"],config.digest(row["root"]));self.assertEqual({v["pit"] for v in label["label"]["action_scores"]},{i for i in range(7) if row["legal_mask"]&(1<<i)})
        token=CancellationToken();token.cancel()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Canceled token searched")):self.assertEqual(p.run(self.base/"canceled",fixture=True,cancel=token)["status"],"interrupted")

    def test_04_durable_resume_marker_refusal_and_closed_no_work_API_CLI(self):
        target=self.base/"resume"
        def stop(e):
            if e["event"]=="allocation_weighting_h12_committed" and e["completed"]==1:raise RuntimeError("durable stop")
        with self.assertRaisesRegex(RuntimeError,"durable stop"):p.run(target,fixture=True,on_event=stop,cancel=CancellationToken())
        with closing(sqlite3.connect(target/"work.sqlite3")) as d:prefix=d.execute("SELECT state,body,hash FROM h12_attempts ORDER BY state").fetchall()
        p.run(target,fixture=True)
        with closing(sqlite3.connect(target/"work.sqlite3")) as d:self.assertEqual(prefix,d.execute("SELECT state,body,hash FROM h12_attempts ORDER BY state LIMIT 1").fetchall())
        bad=self.base/"bad";bad.mkdir()
        with closing(sqlite3.connect(target/"work.sqlite3")) as a,closing(sqlite3.connect(bad/"work.sqlite3")) as b:a.backup(b)
        wrong=copy.deepcopy(p.load_rows(target)[2]);wrong["summary"]["complete"]+=1
        with closing(sqlite3.connect(bad/"work.sqlite3")) as d:
            with d:d.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(wrong),config.digest(wrong)))
        with self.assertRaises(ValueError):p.run(bad,fixture=True)
        self.assertFalse((bad/"manifest.json").exists())
        snapshot=lambda:{f.name:(p.sha(f),f.stat().st_mtime_ns) for f in target.iterdir() if f.is_file() and not f.name.endswith(("-shm","-wal"))};before=snapshot()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Completed reader searched")):self.assertEqual(p.run(target,fixture=True),p.load_rows(target)[2])
        cli=subprocess.run([sys.executable,str(ROOT/"tools/validate-allocation-weighting.py"),"verify-labels","--run",str(target)],cwd=ROOT,text=True,capture_output=True,timeout=180);self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr);self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AllocationWeightingAuthor));methods=unittest.defaultTestLoader.getTestCaseNames(AllocationWeightingAuthor)
    body=dict(schema="allocation-weighting-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),methods=methods,covered_methods=[__name__+".AllocationWeightingAuthor."+m for m in methods],inputs=p.common.pins([*p.SOURCES,*p.DEPENDENCIES,p.BASELINE,p.INVENTORY,p.REGISTRY,p.RULE,p.RESERVATIONS,p.PLAN,p.IMPLEMENTATION]),fixture_path=AllocationWeightingAuthor.base.relative_to(ROOT).as_posix() if hasattr(AllocationWeightingAuthor,"base") else None)
    p.history.teacher.atomic_json(ROOT/"build/allocation-weighting-author.json",body);raise SystemExit(not result.wasSuccessful())
