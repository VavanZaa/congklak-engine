"""Source-bound owner checks for exact-root full-action diagnostic teacher work."""
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,divergence_teacher as p
from congklak.puct import CancellationToken


class DivergenceTeacherAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.admit();cls.base=ROOT/"build/divergence-teacher-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True);cls.path=cls.base/"fresh";cls.settings=p.settings_for(True,2)
        cls.cli=subprocess.run([sys.executable,str(ROOT/"tools/label-divergence-actions.py"),"all","--fixture","--roots","2","--run",str(cls.path)],cwd=ROOT,text=True,capture_output=True,timeout=120);p.require(cls.cli.returncode==0,cls.cli.stdout+cls.cli.stderr);cls.rows,cls.stores,cls.body=p.load_rows(cls.path)

    def copydb(self,name):
        path=self.base/name;path.mkdir()
        with closing(sqlite3.connect(self.path/"work.sqlite3")) as source,closing(sqlite3.connect(path/"work.sqlite3")) as dest:source.backup(dest)
        return path

    def test_01_exact_panel_linkage_fixed_contract_and_production_gate(self):
        actual=p.panel();roots=p.common.load_panel()["roots"];self.assertEqual(len(actual),5);self.assertEqual(sorted(len(row["legal_actions"]) for row in roots),[4,4,5,5,6])
        for root in roots:
            wrapper=actual[root["state_key"]];self.assertEqual(wrapper["root"],root);self.assertEqual(wrapper["source_root_sha256"],config.digest(root));self.assertEqual(wrapper["state"],root["state"]);self.assertEqual(wrapper["source_divergence_sha256"],root["source_divergence_sha256"]);self.assertEqual(wrapper["cohort"],"diagnostic")
        self.assertEqual(p.CONTRACT["maximum_new_nodes"],85000000);self.assertFalse(p.CONTRACT["automatic_expansion"]);self.assertFalse(p.CONTRACT["training_performed"])
        with self.assertRaises(ValueError):p.settings_for(1,2)
        with self.assertRaises(ValueError):p.settings_for(False,2)
        with patch.object(p,"admit",side_effect=ValueError("implementation not frozen")),patch.object(p.Native,"analyze",side_effect=AssertionError("Unfrozen fixture searched")):
            with self.assertRaisesRegex(ValueError,"implementation not frozen"):p.run(self.base/"unfrozen",fixture=True,roots=2)
        with patch.object(p,"gate",side_effect=ValueError("comparison not closed")),patch.object(p.Native,"analyze",side_effect=AssertionError("gate failed open")):
            with self.assertRaisesRegex(ValueError,"comparison not closed"):p.run()
        self.assertFalse(p.DEFAULT.exists(),"Owner gate probe must not create production teacher data")

    def test_02_real_cli_full_action_histories_and_cross_stage_semantics(self):
        self.assertEqual(self.cli.returncode,0);self.assertEqual(self.body["requested_attempts"],6);p.validate_tables(self.rows,self.stores,self.settings,True)
        for key,row in self.rows.items():
            observed={}
            for stage in p.STAGES:
                attempt=self.stores[stage+"_attempts"][key];item=self.stores[stage+"_labels"][key];depth=self.settings[stage]["depth"];self.assertEqual(attempt["completed_depth"],depth);self.assertEqual(item["label"]["choice_depth"],depth);self.assertEqual([r["choice_depth"] for r in attempt["history"]],list(range(1,depth+1)))
                self.assertEqual({score["pit"] for score in item["label"]["action_scores"]},{i for i in range(7) if row["legal_mask"]&(1<<i)})
                for report in attempt["history"]:
                    signature=p.q_signature(p.item_for(report,row));d=report["choice_depth"]
                    if d in observed:self.assertEqual(signature,observed[d])
                    observed[d]=signature
        self.assertTrue(self.body["identity"]["fixture"]);self.assertEqual(self.body["identity"]["settings"]["h10"]["depth"],3)

    def test_03_partial_history_corruption_and_prepublication_marker_refusal(self):
        key=next(iter(self.rows));row=self.rows[key];attempt=self.stores["h10_attempts"][key];label=self.stores["h10_labels"][key]
        for mutate in (lambda a:a["history"].pop(0),lambda a:a["work"].update(nodes=0),lambda a:a["history"][0]["search_options"].update(use_pvs=False)):
            bad=copy.deepcopy(attempt);mutate(bad)
            with self.assertRaises(ValueError):p.validate_attempt(row,bad,label,"h10",self.settings,p.Native())
        budget=copy.deepcopy(self.settings);budget["h8"]["node_budget"]=1;partial,partial_attempt=p.execute(row,"h8",budget,p.Native(),CancellationToken());self.assertIsNone(partial);self.assertEqual(partial_attempt["status"],"budget_rejected");self.assertEqual(partial_attempt["work"]["nodes"],1)
        p.publish(self.base/"one-node-partial-refusal-fixture.json",dict(fixture=True,refusal_only=True,settings=budget,row=row,attempt=partial_attempt,label=partial))
        with self.assertRaises(ValueError):p.validate_attempt(row,partial_attempt,label,"h8",budget,p.Native())
        broken=self.copydb("corrupt-marker");marker=copy.deepcopy(self.body);marker["requested_attempts"]+=1
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):p.run(broken,fixture=True,roots=2)
        self.assertFalse((broken/"manifest.json").exists())
        typed=self.copydb("fixture-integer");ident=copy.deepcopy(self.body["identity"]);ident["fixture"]=1
        with closing(sqlite3.connect(typed/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='identity'",(config.canonical(ident),config.digest(ident)))
        with self.assertRaises(ValueError):p.load_rows(typed)
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Corrupt identity searched")):
            with self.assertRaises(ValueError):p.run(typed,fixture=True,roots=2)
        self.assertFalse((typed/"manifest.json").exists())

    def test_04_committed_prefix_token_cancel_and_completed_no_search_stability(self):
        target=self.base/"resume"
        def stop(event):
            if event["event"]=="divergence_teacher_h8_committed" and event["completed"]==1:raise RuntimeError("durable owner stop")
        with self.assertRaisesRegex(RuntimeError,"durable owner stop"):p.run(target,fixture=True,roots=2,on_event=stop,cancel=CancellationToken())
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:prefix=db.execute("SELECT state,body,hash FROM h8_attempts ORDER BY state").fetchall()
        p.run(target,fixture=True,roots=2,cancel=CancellationToken())
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT state,body,hash FROM h8_attempts ORDER BY state LIMIT 1").fetchall())
        rows,stores,_=p.load_rows(target)
        for stage in p.STAGES:
            for key in rows:self.assertEqual(p.q_signature(stores[stage+"_labels"][key]),p.q_signature(self.stores[stage+"_labels"][key]))
        canceled=CancellationToken();canceled.cancel()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Canceled token searched")):self.assertEqual(p.run(self.base/"canceled",fixture=True,roots=2,cancel=canceled)["status"],"interrupted")
        snapshot=lambda:{file.name:(p.sha(file),file.stat().st_mtime_ns) for file in target.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot()
        with patch.object(p.Native,"analyze",side_effect=AssertionError("Completed teacher searched")):
            expected=p.run(target,fixture=True,roots=2);self.assertEqual(p.load_rows(target)[2],expected)
        self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)
        cli=subprocess.run([sys.executable,str(ROOT/"tools/label-divergence-actions.py"),"verify","--run",str(target)],cwd=ROOT,text=True,capture_output=True,timeout=120);self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr);self.assertEqual(before,snapshot())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DivergenceTeacherAuthor));methods=unittest.defaultTestLoader.getTestCaseNames(DivergenceTeacherAuthor)
    proof=dict(schema="divergence-teacher-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=methods,covered_methods=[__name__+".DivergenceTeacherAuthor."+name for name in methods],inputs=p.common.pins(p.SOURCES+p.DEPENDENCIES+(str(p.BASELINE),str(p.common.PANEL),str(p.REGISTRY),str(p.PLAN),str(p.IMPLEMENTATION))),fixture_path=DivergenceTeacherAuthor.base.relative_to(ROOT).as_posix() if hasattr(DivergenceTeacherAuthor,"base") else None)
    p.history.teacher.atomic_json(ROOT/"build/divergence-teacher-author.json",proof);raise SystemExit(not result.wasSuccessful())
