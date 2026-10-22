"""Four source-bound checks for admitted training data and cached recovery."""
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag as stage,neurodiag_data as data
from congklak.native import Native
from congklak.neural_relay import Model


class NeurodiagDataAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stage.admit();cls.pack=data.load_inputs();cls.panel=data.read(data.PANEL);cls.base=ROOT/"build/neurodiag-data-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)

    def test_01_metadata_target_denial_whole_components_and_deterministic_panel(self):
        actual=data.ro
        def guarded(name):
            db=actual(name)
            def authorizer(action,table,column,*_):
                if action==sqlite3.SQLITE_READ and column=="body" and isinstance(table,str) and (table=="labels" or table.endswith(("_labels","_references","_attempts"))):return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorizer);return db
        with patch.object(data,"ro",side_effect=guarded),patch.object(Native,"__init__",side_effect=AssertionError("Native activated")):
            inv=data.metadata_inventory();panel=data.select(inv)
        self.assertTrue(data.equal(inv,data.read(data.INVENTORY)));self.assertTrue(data.equal(panel,self.panel));self.assertEqual((panel["source_records"],panel["target_records"],panel["unique_rawstates"]),(332,360,308))
        self.assertEqual(len({r["component"] for r in panel["primary"]}),256);self.assertEqual(len({r["family"] for r in panel["primary"]}),256)
        for cell in data.CELLS:self.assertEqual(sum(r["cell"]==cell for r in panel["primary"]),32)
        protectedq=set(inv["protected_quotients"]);protectedf=set(inv["protected_families"]);late_components=[]
        for cid,c in inv["components"].items():
            if set(c["families"])&protectedf or set(c["quotients"])&protectedq:self.assertTrue(c["blocked_reasons"])
            if any(q in protectedq and sum(bytes.fromhex(q)[:-1])==5 for q in c["quotients"]):late_components.append(cid);self.assertTrue(c["blocked_reasons"])
        self.assertTrue(late_components)
        for name in ("primary","paired","MC"):
            for row in panel[name]:self.assertFalse(inv["components"][row["component"]]["blocked_reasons"]);self.assertFalse(set(inv["components"][row["component"]]["quotients"])&protectedq)
        self.assertTrue(panel["no_new_optimizer_reservations"]);self.assertFalse(panel["new_assessment_or_holdout_claim"])

    def test_02_exact_typed_target_sources_and_cache_only_input_loading(self):
        helper=stage.metrics()
        with patch.object(helper,"literal_features",side_effect=AssertionError("Input load recomputed features")),patch.object(helper,"observe",side_effect=AssertionError("Input load forwarded")),patch.object(helper,"summarize_cached",side_effect=AssertionError("Input load summarized")):
            pack=data.load_inputs();self.assertTrue(data.equal(pack,self.pack))
        for unit in pack["units"].values():
            for row in unit["records"]:
                ref=row["target_source"]["reference"];body,digest=data.source_item(ref,"id" if row["cohort"]=="ordinary_MC" else "state");self.assertEqual(digest,row["target_source"]["item_sha256"])
                target=body["terminal_return_player0"] if row["cohort"]=="ordinary_MC" else body["label"]["target_player0"];self.assertEqual(row["target_player0"],target)
                if row["cohort"]=="ordinary_MC":self.assertEqual(target%2,0);self.assertTrue(row["target_source"]["individual_policy_return_not_minimax_truth"])
                else:self.assertIn(ref["depth"],(8,10));self.assertEqual(body["label"]["choice_depth"],ref["depth"]);self.assertFalse(row["target_source"]["target_is_solved_truth"])
                self.assertGreaterEqual((target-unit["features"]["d"])/unit["features"]["r"],-1);self.assertLessEqual((target-unit["features"]["d"])/unit["features"]["r"],1)

    def test_03_forward_unit_dedup_individual_returns_and_horizon_estimands(self):
        pack=self.pack;self.assertEqual((len(pack["units"]),pack["counts"]["target_records"]),(308,360));self.assertEqual(pack["counts"]["literal_generated_missing_vectors"],13);self.assertEqual(pack["counts"]["literal_cache_comparisons"],15)
        groups={name:[] for name in ("base_H8","paired_H8_H10","ordinary_MC")};by_state={}
        for uid,unit in pack["units"].items():
            self.assertEqual(uid,config.digest(dict(state_key=unit["state_key"],features=unit["features"])));self.assertNotIn(unit["state_key"],by_state);by_state[unit["state_key"]]=uid
            for row in unit["records"]:groups[row["cohort"]].append(row)
        self.assertEqual({k:len(v) for k,v in groups.items()},dict(base_H8=256,paired_H8_H10=56,ordinary_MC=48))
        for name,n in (("paired_H8_H10",28),("ordinary_MC",24)):
            keys={r["state_key"] for r in groups[name]};self.assertEqual(len(keys),n)
            for key in keys:
                records=[r for r in groups[name] if r["state_key"]==key];self.assertEqual(len(records),2);self.assertEqual(len({r["family"] for r in records}),1)
                if name=="paired_H8_H10":self.assertEqual({r["target_type"] for r in records},{"approximate_H8_root_margin","approximate_H10_root_margin"})
                else:self.assertEqual(len({r["target_source"]["gameid"] for r in records}),2)
        self.assertEqual(len({r["state_key"] for v in groups.values() for r in v}),308)

    def test_04_input_corruption_uncertified_gate_and_no_repeated_prefix_forwards(self):
        wrong=copy.deepcopy(self.pack);key=next(iter(wrong["units"]));wrong["units"][key]["records"][0]["family"]="wrong-origin";wrong["unit_sha256"][key]=config.digest(wrong["units"][key])
        with self.assertRaises(ValueError):data.validate_inputs(wrong)
        with patch.object(stage,"owner_gate",side_effect=ValueError("preflight missing")):
            with self.assertRaisesRegex(ValueError,"preflight missing"):stage.run()
        helper=stage.metrics();fresh=self.base/"fresh";resume=self.base/"resume";calls=[];observe=helper.observe
        def counted(units,models):calls.append(tuple(sorted(units)));return observe(units,models)
        with patch.object(Native,"__init__",side_effect=AssertionError("Native activated")),patch.object(Model,"reference",side_effect=AssertionError("Native model evaluated")):
            stage.run(fresh,fixture=True)
            def stop(event):
                if event["event"]=="neurodiag_committed" and event["completed"]==1:raise RuntimeError("retained batch")
            with patch.object(helper,"observe",side_effect=counted):
                with self.assertRaisesRegex(RuntimeError,"retained batch"):stage.run(resume,fixture=True,on_event=stop)
                stage.run(resume,fixture=True)
        self.assertEqual([len(v) for v in calls],[16,16,16]);self.assertEqual(len(set().union(*(set(v) for v in calls))),48)
        a,b=stage.raw_complete(fresh),stage.raw_complete(resume);self.assertTrue(data.equal(a["observations"],b["observations"]));self.assertTrue(data.equal(a["summary"],b["summary"]))
        def snapshot():return {p.name:(data.sha(p),p.stat().st_size,p.stat().st_mtime_ns) for p in resume.iterdir() if p.is_file() and not p.name.endswith("-shm") and (not p.name.endswith("-wal") or p.stat().st_size)}
        before=snapshot()
        with patch.object(helper,"observe",side_effect=AssertionError("Old batches forwarded")),patch.object(helper,"literal_features",side_effect=AssertionError("Old features recomputed")),patch.object(helper,"summarize_cached",side_effect=AssertionError("Old summary/SVD recomputed")):
            stage.raw_complete(resume)
            with self.assertRaisesRegex(ValueError,"certified"):stage.run(resume,fixture=True)
        self.assertEqual(before,snapshot());self.assertNotIn("torch",sys.modules)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NeurodiagDataAuthor));names=unittest.defaultTestLoader.getTestCaseNames(NeurodiagDataAuthor)
    body=dict(schema="neurodiag-data-author-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),methods=names,covered_methods=[__name__+".NeurodiagDataAuthor."+n for n in names],inputs=data.common.pins([*stage.SOURCES,stage.PLAN,stage.IMPLEMENTATION,stage.SOURCE_FREEZE,data.INVENTORY,data.PANEL,data.INPUTS,data.common.BASELINE]),fixture_path=NeurodiagDataAuthor.base.relative_to(ROOT).as_posix() if hasattr(NeurodiagDataAuthor,"base") else None)
    data.publish(ROOT/"build/neurodiag-data-author.json",body);raise SystemExit(not result.wasSuccessful())
