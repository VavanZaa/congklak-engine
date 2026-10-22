"""Four owner methods for source-frozen, engine-free retrospective diagnostics."""
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
from congklak import config,capture_drift as p
from congklak.native import Native
from congklak.neural_relay import Model
from congklak.puct import CancellationToken


class CaptureDriftAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(Native,"__init__",side_effect=AssertionError("Native activated")),patch.object(Model,"reference",side_effect=AssertionError("NN evaluated")):cls.source=p.admit()
        cls.base=ROOT/"build/capture-drift-author-fixtures"/uuid.uuid4().hex;cls.base.mkdir(parents=True);cls.path=cls.base/"fresh"
        result=cls.cli("run",cls.path);p.require(result.returncode==0,result.stdout+result.stderr);cls.rows,cls.manifest=p.load_rows(cls.path)

    @classmethod
    def cli(cls,operation,path):return subprocess.run([sys.executable,str(ROOT/"tools/investigate-capture-drift.py"),operation,"--fixture","--cases","2","--run",str(path)],cwd=ROOT,text=True,capture_output=True,timeout=180)

    def test_01_exact_sealed_support_all16_contributions_and_covariance(self):
        data=self.source;ctx=p.context(data);self.assertEqual(len(data["rows"]),16);self.assertEqual(len(data["capture_keys"]),4);self.assertEqual(ctx["global_roots"],16)
        expected=data["previous"]["metrics"]["overall"]
        for name in ("uniform","weighted"):self.assertEqual(ctx["pairs"]["h10_to_h12"][name+"_global_drift"],expected[name]["action_Q_drift"])
        for key,r in ctx["roots"].items():
            for before,after in p.PAIRS:
                qa,qb=p.qvalues(data["stores"][before+"_labels"][key]),p.qvalues(data["stores"][after+"_labels"][key]);d=F(sum(abs(qb[a]-qa[a]) for a in qa),len(qa));self.assertEqual(p.fraction(r["pairs"][before+"_to_"+after]["drift"]),d)
        for cell,r in ctx["cells"].items():
            self.assertEqual(p.fraction(r["uniform_mass"]),F(1,8));self.assertEqual(p.fraction(r["weighted_mass"]),F(1,8))
            for pair,proof in r["pairs"].items():self.assertEqual(p.fraction(proof["global_delta"]),sum((p.fraction(ctx["roots"][k]["pairs"][pair]["global_delta"]) for k in r["state_keys"]),F()))
        pair=ctx["pairs"]["h10_to_h12"];self.assertEqual(p.fraction(pair["global_delta"]),p.fraction(pair["capture_global_delta"])+p.fraction(pair["non_capture_global_delta"]));self.assertGreater(p.fraction(pair["capture_global_delta"]),0)
        self.assertFalse(p.DEFAULT.exists());self.assertEqual([v["state_key"] for v in self.rows.values()],data["capture_keys"][:2])

    def test_02_signed_cancellation_fullties_ranks_and_covariance_synthetics(self):
        d=p.translation([3,-1,-1,-2]);self.assertEqual(p.fraction(d["D_mean_absolute"]),F(7,4));self.assertEqual(p.fraction(d["C_absolute_common_mean"]),F(1,4));self.assertEqual(p.fraction(d["S_signed_cancellation_remainder"]),F(3,2));self.assertEqual(p.fraction(d["E_centered_dispersion"]),F(13,8));self.assertFalse(d["D_equals_C_plus_E"])
        for values in ([2,2],[2,-2],[0,0],[-1,-3]):
            a=p.translation(values);self.assertEqual(p.fraction(a["D_mean_absolute"]),p.fraction(a["C_absolute_common_mean"])+p.fraction(a["S_signed_cancellation_remainder"]))
        cov=p.covariance_identity([F(1,2),F(1,4)],[F(3),F(1)]);self.assertEqual(p.fraction(cov["conditional_delta"]),F(1,3));self.assertEqual(p.fraction(cov["global_delta"]),F(1,24))
        def item(q,best):return dict(label=dict(action_scores=[dict(pit=a,value_player0=v) for a,v in q.items()],target_player0=min(q.values()),best_moves=best))
        endpoints=dict(h8=item({0:2,1:2},[0,1]),h10=item({0:2,1:2},[0,1]),h12=item({0:3,1:1},[1]));prob=dict(uniform_probability=p.rational(F(1,16)),weighted_probability=p.rational(F(1,16)))
        result=p.action_analysis(dict(actor=1),endpoints,prob);self.assertEqual(result["H10_best_H12_regrets"],{0:2,1:0});self.assertEqual(result["pairs"]["h10_to_h12"]["tie_changes"],1);self.assertEqual(result["pairs"]["h10_to_h12"]["strict_ranking_reversals"],0)
        single={s:item({0:2},[0]) for s in p.STAGES};self.assertEqual(p.action_analysis(dict(actor=1),single,prob)["actions"][0]["pairs"]["h10_to_h12"]["delta_actor"],0)
        with self.assertRaises(ValueError):p.settings_for(1,2)

    def test_03_cli_cancel_source_gate_and_durable_resume_prefix(self):
        self.assertEqual(self.manifest["cases"],2)
        token=CancellationToken();token.cancel()
        with patch.object(Native,"__init__",side_effect=AssertionError("Native activated")),patch.object(Model,"reference",side_effect=AssertionError("NN evaluated")):
            self.assertEqual(p.run(self.base/"cancelled",fixture=True,cancel=token)["status"],"interrupted")
        with patch.object(p,"admit",side_effect=ValueError("unfrozen")):
            with self.assertRaisesRegex(ValueError,"unfrozen"):p.run(self.base/"unfrozen",fixture=True)
        with patch.object(p,"gate",side_effect=ValueError("preflight missing")):
            with self.assertRaisesRegex(ValueError,"preflight missing"):p.run()
        target=self.base/"resume"
        def stop(event):
            if event["event"]=="capture_drift_committed" and event["completed"]==1:raise RuntimeError("durable stop")
        with self.assertRaisesRegex(RuntimeError,"durable stop"):p.run(target,fixture=True,on_event=stop)
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:prefix=db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall()
        p.run(target,fixture=True)
        with closing(sqlite3.connect(target/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT idx,body,hash FROM cases ORDER BY idx LIMIT 1").fetchall())
        self.assertEqual(p.load_rows(target)[0],self.rows)

    def test_04_rehashed_corruption_marker_refusal_repair_and_closed_no_work(self):
        for kind in ("case","complete"):
            target=self.base/("bad-"+kind);target.mkdir()
            with closing(sqlite3.connect(self.path/"work.sqlite3")) as source,closing(sqlite3.connect(target/"work.sqlite3")) as destination:source.backup(destination)
            with closing(sqlite3.connect(target/"work.sqlite3")) as db:
                if kind=="case":
                    body=copy.deepcopy(self.rows[0]);body["analysis"]["pairs"]["h10_to_h12"]["root_value_delta_player0"]+=1
                    with db:db.execute("UPDATE cases SET body=?,hash=? WHERE idx=0",(config.canonical(body),config.digest(body)))
                else:
                    body=copy.deepcopy(self.manifest);body["cases"]+=1
                    with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(body),config.digest(body)))
            with self.assertRaises(ValueError):p.run(target,fixture=True)
            self.assertFalse((target/"manifest.json").exists())
        (self.path/"manifest.json").unlink();p.run(self.path,fixture=True)
        def snapshot():return {f.name:(p.sha(f),f.stat().st_size,f.stat().st_mtime_ns) for f in self.path.iterdir() if f.is_file() and not f.name.endswith("-shm") and (not f.name.endswith("-wal") or f.stat().st_size)}
        before=snapshot();events=[]
        with patch.object(Native,"__init__",side_effect=AssertionError("Native activated")),patch.object(Model,"reference",side_effect=AssertionError("NN evaluated")):
            p.run(self.path,fixture=True,on_event=events.append);p.load_rows(self.path)
        self.assertFalse(any(e["event"] in ("capture_drift_precommit","capture_drift_committed") for e in events));self.assertEqual(before,snapshot())
        cli=self.cli("verify-cases",self.path);self.assertEqual(cli.returncode,0,cli.stdout+cli.stderr);self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CaptureDriftAuthor));methods=unittest.defaultTestLoader.getTestCaseNames(CaptureDriftAuthor)
    body=dict(schema="capture-drift-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),methods=methods,covered_methods=[__name__+".CaptureDriftAuthor."+m for m in methods],inputs=p.common.pins([*p.SOURCES,*p.DEPENDENCIES,p.BASELINE,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE]),fixture_path=CaptureDriftAuthor.base.relative_to(ROOT).as_posix() if hasattr(CaptureDriftAuthor,"base") else None)
    p.publish(ROOT/"build/capture-drift-author.json",body);raise SystemExit(not result.wasSuccessful())
