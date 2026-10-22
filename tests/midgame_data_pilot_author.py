"""Owner role/closure/legal-history/recovery checks for targeted collection."""
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
import sys
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,midgame_data_pilot as p
from congklak.native import Native,flatten


class MidgameDataPilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.plan();cls.registry=p.read(p.REGISTRY);cls.settings=p.settings_for(True,8);cls.native=Native()
        cls.base=ROOT/"build/midgame-data-pilot-author"/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.fresh=cls.base/"fresh";p.collect(cls.fresh,fixture=True,families=8);cls.rows,cls.manifest=p.load_rows(cls.fresh)

    def test_01_role_registry_latest144_and_diagnostic_ownership(self):
        registry=self.registry;self.assertEqual(registry["historical_role_counts"],dict(training_candidate=61,validation=14,diagnostic=8))
        self.assertEqual(registry["prior_family_pool_count"],768);self.assertEqual(len(registry["base_ordinary_development"]),259);self.assertEqual(len(registry["latest_arena_games"]),144)
        self.assertEqual(len(registry["diagnostics"]),9);self.assertEqual(sum(r["provenance"]["existing_Q_comparison"]["coverage"]=="uncovered" for r in registry["diagnostics"].values()),8)
        for game in registry["latest_arena_games"]:
            self.assertIn(game["family"],registry["families"])
            self.assertTrue(all(row["quotient"] in registry["quotients"] for row in game["states"]))
        for key,row in registry["diagnostics"].items():
            self.assertEqual(key,p.raw_key(row["state"]));self.assertEqual(row["state"],row["provenance"]["raw_state"]);self.assertFalse(row["provenance"]["swapped"]);self.assertEqual(row["cohort"],"diagnostic")
            self.assertIn(row["quotient"],registry["quotients"]);self.assertNotIn(key,self.rows)
        self.assertFalse(registry["historical_roles_rewritten"]);self.assertTrue(registry["future_protection_overrides_eligibility"])

    def test_02_transitive_components_protected_and_crossownership(self):
        def row(family,cohort,q,step):return dict(state_key=family+str(step),family=family,cohort=cohort,quotient=q,stratum="25-49/ordinary",legal_mask=3,provenance=dict(family_index=ord(family)-65,step=step))
        families={0:dict(family="A",cohort="training_candidate",candidates=[row("A","training_candidate","q1",0)]),1:dict(family="B",cohort="training_candidate",candidates=[row("B","training_candidate","q1",0),row("B","training_candidate","q2",1)]),2:dict(family="C",cohort="validation",candidates=[row("C","validation","q2",0)])}
        selected,detail=p.select(families,self.settings,dict(quotients=[],families=[]));self.assertFalse(selected);self.assertEqual(detail["blocked_family_count"],3);self.assertIn("mixed_ownership_component",detail["components"][0]["reasons"])
        same=copy.deepcopy(families)
        for family in same.values():
            family["cohort"]="training_candidate"
            for item in family["candidates"]:item["cohort"]="training_candidate"
        selected,detail=p.select(same,self.settings,dict(quotients=["q1"],families=[]));self.assertFalse(selected);self.assertEqual(detail["blocked_family_count"],3);self.assertIn("protected_eligible_quotient",detail["components"][0]["reasons"])
        selected,detail=p.select(same,self.settings,dict(quotients=[],families=[]));self.assertTrue(selected);self.assertEqual(detail["blocked_family_count"],0)

    def test_03_fresh_fullpool_selection_and_literal_history_provenance(self):
        pool=p.read_families(self.fresh);expected,selection=p.select(pool,self.settings,self.registry);self.assertEqual(expected,self.rows);self.assertEqual(selection,self.manifest["selection"])
        seen=set();family_strata=set()
        for key,row in self.rows.items():
            self.assertGreaterEqual(row["legal_mask"].bit_count(),2);self.assertIn(row["stratum"],p.STRATA);self.assertNotIn(row["quotient"],self.registry["quotients"]);self.assertNotIn(row["family"],self.registry["families"])
            self.assertNotIn(row["quotient"],seen);seen.add(row["quotient"]);self.assertNotIn((row["family"],row["stratum"]),family_strata);family_strata.add((row["family"],row["stratum"]))
            state=config.position()
            for action in row["provenance"]["moves_prefix"]:
                state=self.native.normalize(flatten(state),actor=state["controller"],total=98)["position"];transition=self.native.apply(flatten(state),actor=state["controller"],action=action,total=98,trace=False);self.assertTrue(transition["ok"]);state=transition["position"]
            state=self.native.normalize(flatten(state),actor=state["controller"],total=98)["position"];self.assertEqual(state,row["provenance"]["raw_state"])
            canonical,position,swapped=p.pilot.canonicalize(state,self.native);self.assertEqual(key,canonical.hex());self.assertEqual(row["state"],position);self.assertEqual(row["provenance"]["swapped"],swapped);self.assertEqual(p.pilot._mechanism(position,self.native)[1],row["stratum"])
        for cohort,values in selection["counts"].items():
            for stratum,count in values.items():self.assertEqual(count+selection["shortfall"][cohort][stratum],self.settings["caps"][cohort][stratum])

    def test_04_prefix_recovery_closed_stability_and_corrupt_marker_refusal(self):
        run=self.base/"resume"
        def interrupt(value):
            if value["event"]=="midgame_collection_committed" and value["completed"]==2:raise RuntimeError("owner durable interruption")
        with self.assertRaisesRegex(RuntimeError,"owner durable interruption"):p.collect(run,fixture=True,families=8,on_event=interrupt)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:prefix=db.execute("SELECT idx,body,hash FROM families ORDER BY idx").fetchall()
        p.collect(run,fixture=True,families=8)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:self.assertEqual(prefix,db.execute("SELECT idx,body,hash FROM families ORDER BY idx LIMIT 2").fetchall())
        rows,manifest=p.load_rows(run);self.assertEqual(rows,self.rows);self.assertEqual(manifest["family_semantic_sha256"],self.manifest["family_semantic_sha256"])
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in run.iterdir() if file.is_file() and not file.name.endswith(("-wal","-shm"))}
        before=snapshot();original=Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError("Collection executed teacher search")
        Native.analyze=prohibited
        try:p.collect(run,fixture=True,families=8)
        finally:Native.analyze=original
        self.assertEqual(before,snapshot())
        broken=self.base/"bad-marker";broken.mkdir()
        with closing(sqlite3.connect(run/"work.sqlite3")) as source,closing(sqlite3.connect(broken/"work.sqlite3")) as destination:source.backup(destination)
        damaged=copy.deepcopy(manifest);damaged["selected_roots"]+=1
        with closing(sqlite3.connect(broken/"work.sqlite3")) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(damaged),config.digest(damaged)))
        with self.assertRaises(ValueError):p.collect(broken,fixture=True,families=8)
        self.assertFalse((broken/"manifest.json").exists())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MidgameDataPilotAuthor))
    proof=dict(schema="midgame-data-pilot-owner-acceptance-v1",passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),methods=unittest.defaultTestLoader.getTestCaseNames(MidgameDataPilotAuthor),inputs={name:p.sha(ROOT/name) for name in p.SOURCES+("build/midgame-pilot-baseline.json","build/midgame-data-pilot-registry.json","build/midgame-data-pilot-plan.json")})
    p.teacher.atomic_json(ROOT/"build/midgame-data-pilot-author.json",proof)
    raise SystemExit(not result.wasSuccessful())
