"""Four frozen-source owner checks; only old diagnostic teaching fixtures."""
from contextlib import closing
import copy
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,stratum_weighting as p
from congklak.puct import CancellationToken


class StratumWeightingAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry=p.admit();cls.base=ROOT/"build/stratum-weighting-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True);cls.path=cls.base/"fresh"
        for op in ("h8","h10","realize","h12"):
            result=cls.cli(op,cls.path);p.require(result.returncode==0,result.stdout+result.stderr)
        cls.rows,cls.stores,cls.manifest=p.load_rows(cls.path)

    @classmethod
    def cli(cls,operation,path,extra=()):return subprocess.run([sys.executable,str(ROOT/"tools/validate-stratum-weighting.py"),operation,"--fixture","--roots","2","--run",str(path),*extra],cwd=ROOT,text=True,capture_output=True,timeout=240)

    def test_01_blind_full_component_inventory_panel_and_new_reservations(self):
        real=p.ancestral.ro
        def guarded(name):
            db=real(name)
            def auth(action,table,column,*_):
                if action==sqlite3.SQLITE_READ and column=="body" and (table.endswith(("_labels","_references","_attempts")) or table=="labels"):return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(auth);return db
        with patch.object(p.ancestral,"ro",side_effect=guarded):inv=p.inventory()
        self.assertEqual((inv["eligible_roots"],inv["eligible_components"],inv["all_origin_families"],inv["duplicate_raw_observations"]),(3890,547,553,6))
        selected=p.select(inv);self.assertEqual(selected,self.registry["rows"]);self.assertEqual(len(selected),16);self.assertEqual(len({r["family"] for r in selected.values()}),16);self.assertEqual(len({r["component"] for r in selected.values()}),16)
        for cell in p.CELLS:self.assertEqual(sum(r["cell"]==cell for r in selected.values()),2)
        reserved=p.read(p.RESERVATIONS);self.assertTrue(reserved["future_optimizer_reserved"]);self.assertFalse(reserved["historical_roles_rewritten"])
        old=p.read(p.ancestral.RESERVATIONS);oldq={q for role in old["roles"].values() for q in role["eligible_quotients"]};oldf={f for role in old["roles"].values() for f in role["origin_families"]}
        self.assertFalse(set(reserved["eligible_quotients"])&oldq);self.assertFalse(set(reserved["origin_families"])&oldf)
        for r in selected.values():self.assertEqual(r["source_cohort"],"training_candidate");self.assertEqual(r["group"],"assessment");self.assertGreaterEqual(r["legal_mask"].bit_count(),2)
        self.assertFalse(p.RULE.exists());self.assertFalse(p.DEFAULT.exists())

    def test_02_ordered_freeze_missing_rule_gate_and_strict_fixture_identity(self):
        with self.assertRaises(ValueError):p.settings_for(1,2)
        with patch.object(p,"admit",side_effect=ValueError("unfrozen")),patch.object(p.Native,"analyze",side_effect=AssertionError("unfrozen fixture searched")):
            with self.assertRaisesRegex(ValueError,"unfrozen"):p.run(self.base/"unfrozen",fixture=True)
        with patch.object(p,"initial_gate",side_effect=ValueError("preflight missing")),patch.object(p.Native,"analyze",side_effect=AssertionError("production before preflight")):
            with self.assertRaisesRegex(ValueError,"preflight missing"):p.run(stage="h8")
        pending=self.base/"pending";p.run(pending,stage="h8",fixture=True);p.run(pending,stage="h10",fixture=True)
        with patch.object(p.Native,"analyze",side_effect=AssertionError("H12 before realized weights")):
            with self.assertRaisesRegex(ValueError,"H12 HOLD"):p.run(pending,stage="h12",fixture=True)
        self.assertFalse(p.rule_path(pending,True).exists());p.realize(pending)
        with closing(sqlite3.connect(pending/"work.sqlite3")) as db:self.assertEqual(db.execute("SELECT count(*) FROM h12_attempts").fetchone()[0],0)
        with self.assertRaises(ValueError):p.load_rows(pending)
        self.assertEqual(p.load_rows(pending,True)[2]["status"],"awaiting_rule_audit")

    def test_03_public_cli_real_depths_fullQ_cancel_and_resume_prefix(self):
        self.assertEqual(self.manifest["status"],"complete")
        for stage,depth in zip(p.STAGES,(2,3,4)):
            self.assertEqual(self.manifest["summary"][stage]["complete"],2)
            for key,a in self.stores[stage+"_attempts"].items():
                item=self.stores[stage+"_labels"][key];self.assertEqual(a["requested_depth"],depth);self.assertEqual(item["label"]["choice_depth"],depth);self.assertEqual([r["choice_depth"] for r in a["history"]],list(range(1,depth+1)));self.assertEqual(item["root_sha256"],self.rows[key]["source_root_sha256"])
                self.assertEqual({v["pit"] for v in item["label"]["action_scores"]},{i for i in range(7) if self.rows[key]["legal_mask"]&(1<<i)})
        self.assertIsNone(p.read(self.path/"rule.json")["projection"])
        token=CancellationToken();token.cancel()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("cancelled token searched")):self.assertEqual(p.run(self.base/"cancelled",stage="h8",fixture=True,cancel=token)["status"],"interrupted")
        target=self.base/"resume";p.run(target,stage="h8",fixture=True)
        def stop(event):
            if event["event"]=="stratum_weighting_h10_committed" and event["stage_completed"]==1:raise RuntimeError("durable stop")
        with self.assertRaisesRegex(RuntimeError,"durable stop"):p.run(target,stage="h10",fixture=True,on_event=stop,cancel=CancellationToken())
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:prefix={t:db.execute("SELECT state,body,hash FROM "+t+" ORDER BY state").fetchall() for t in ("h8_attempts","h8_labels","h10_attempts","h10_labels")}
        p.run(target,stage="h10",fixture=True)
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:
            for table,values in prefix.items():self.assertEqual(values,db.execute("SELECT state,body,hash FROM "+table+" ORDER BY state").fetchall()[:len(values)])
        p.realize(target);p.run(target,stage="h12",fixture=True);p.load_rows(target)

    def test_04_corrupt_marker_rule_refusal_export_repair_and_closed_no_work(self):
        bad=self.base/"bad";bad.mkdir()
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as source,closing(sqlite3.connect(bad/"work.sqlite3")) as destination:source.backup(destination)
        for name in ("references-prefix.sqlite3","references-manifest.json","rule.json"):shutil.copy2(self.path/name,bad/name)
        wrong=copy.deepcopy(self.manifest);wrong["total_new_nodes"]+=1
        with closing(sqlite3.connect(bad/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(wrong),config.digest(wrong)))
        with self.assertRaises(ValueError):p.run(bad,fixture=True)
        self.assertFalse((bad/"manifest.json").exists())
        # Existing publications are validated before repair; the complete marker remains authoritative.
        manifest=self.path/"manifest.json";manifest.unlink()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("repair searched")):p.run(self.path,fixture=True)
        def snapshot():return {f.name:(p.sha(f),f.stat().st_size,f.stat().st_mtime_ns) for f in self.path.iterdir() if f.is_file() and not f.name.endswith("-shm") and (not f.name.endswith("-wal") or f.stat().st_size)}
        before=snapshot()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("closed reader searched")),patch.object(p.history,"execute",side_effect=AssertionError("closed producer searched")):
            self.assertEqual(p.run(self.path,fixture=True),p.load_rows(self.path)[2]);p.realize(self.path)
        result=self.cli("verify-labels",self.path);self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StratumWeightingAuthor));methods=unittest.defaultTestLoader.getTestCaseNames(StratumWeightingAuthor)
    body=dict(schema="stratum-weighting-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),methods=methods,covered_methods=[__name__+".StratumWeightingAuthor."+m for m in methods],inputs=p.common.pins([*p.SOURCES,*p.DEPENDENCIES,p.BASELINE,p.INVENTORY,p.REGISTRY,p.RESERVATIONS,p.PLAN,p.IMPLEMENTATION]),fixture_path=StratumWeightingAuthor.base.relative_to(ROOT).as_posix() if hasattr(StratumWeightingAuthor,"base") else None)
    p.history.teacher.atomic_json(ROOT/"build/stratum-weighting-author.json",body);raise SystemExit(not result.wasSuccessful())
