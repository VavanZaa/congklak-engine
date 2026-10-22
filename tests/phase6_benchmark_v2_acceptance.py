"""Actual immutable warm/SP witnesses and checksum-valid admission rejections."""
from __future__ import annotations
import copy
from contextlib import closing
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from congklak import config,phase6_common as common
from congklak.neural_relay import Model
import phase6_benchmark_acceptance as original
spec=importlib.util.spec_from_file_location("phase6_benchmark_v2_under_test",ROOT/"tools/benchmark-phase6-v2.py");bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
original.bench=bench

class SharedContracts(original.Tests):
    """Run the actual H/relay/PUCT and semantic rejection fixtures against V2."""
    def test_real_h_relay_puct_timing_and_unusable_fallback(self):
        # This is a completion/admission fixture, not a 5 ms strength gate.
        # Scheduler delay can correctly leave the first iteration incomplete.
        budget=100
        with original.NeuralNative(ROOT/"data/training-stage5-deeper-h8-f127-seed2718/model.json",mode="value-none") as relay:
            for engine in (("h",original.Native()),("relay",relay),("puct",original.Fixture())):
                answer=bench.decide(engine,self.p,budget,self.plan)
                bench.validate_decision(answer,self.p,budget,self.plan)
                self.assertTrue(answer["usable_decision"])
                self.assertGreater(answer["wall_ms"],0)
                self.assertEqual(answer["overshoot_ms"],max(0,answer["wall_ms"]-budget))
        plan=copy.deepcopy(self.plan);plan["puct"]["max_nodes"]=1
        answer=bench.decide(("puct",original.Fixture()),self.p,budget,plan)
        self.assertFalse(answer["usable_decision"])
        self.assertIsNone(answer["chosen"])
        self.assertEqual(answer["report"]["completed_simulations"],0)

    def test_checksum_valid_position_physical_interval_refused(self):
        answer=bench.decide(("puct",original.Fixture()),self.p,5,self.plan)
        for key in ("estimated_value_player0","unvisited_root_q_player0"):
            changed=copy.deepcopy(answer);changed["report"][key]=10;config.digest(changed)
            with self.assertRaisesRegex(ValueError,"physical interval"):bench.validate_decision(changed,self.p,5,self.plan)
        changed=copy.deepcopy(answer)
        for a,visits in enumerate(changed["report"]["visits"]):
            if visits:changed["report"]["action_mean_value_player0"][a]=10;break
        with self.assertRaisesRegex(ValueError,"physical interval"):bench.validate_decision(changed,self.p,5,self.plan)
        answer=bench.decide(("h",original.Native()),self.p,5,self.plan);answer["report"]["value_player0"]=10
        with self.assertRaisesRegex(ValueError,"physical interval"):bench.validate_decision(answer,self.p,5,self.plan)

def implementation():
    sources={path:common.sha(ROOT/path) for path in bench.ROLE_PATHS.values() if (ROOT/path).exists()}
    body=dict(schema="phase6-implementation-v2",roles=dict(bench.ROLE_PATHS),sources=sources,
              science_inputs={path:common.sha(ROOT/path) for path in bench.SCIENCE_REQUIRED})
    return {**body,"sha256":config.digest(body)}

class Admission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=common.read_plan();cls.implementation=implementation();cls.initial=Model(ROOT/cls.plan["source_model"]);cls.warm=Model(common.WARM)
    def clone(self,folder):
        source=common.WARM.parent;path=Path(folder).resolve();self.assertTrue(path.is_relative_to(ROOT.resolve()))
        shutil.copy2(source/"model.json",path/"model.json");shutil.copy2(source/"result.json",path/"result.json")
        with closing(sqlite3.connect((source/"catalog.sqlite3").resolve().as_uri()+"?mode=ro",uri=True)) as original_db,closing(sqlite3.connect(path/"catalog.sqlite3")) as destination:
            original_db.backup(destination)
            for name, in original_db.execute("SELECT name FROM checkpoints"):shutil.copy2(source/name,path/name)
        return path/"model.json"
    def temp(self):
        folder=ROOT/"build/phase6-benchmark-v2-fixtures";folder.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="guard-",dir=folder);self.assertTrue(Path(temporary.name).resolve().is_relative_to(ROOT.resolve()));return temporary
    def test_actual_complete_warm_and_sp_witnesses_without_torch_or_mutation(self):
        paths=[common.WARM,common.WARM.parent/"result.json",common.WARM.parent/"catalog.sqlite3",common.SELFPLAY/"manifest.json",common.SELFPLAY/"work.sqlite3"]
        before={str(p):(common.sha(p),p.stat().st_mtime_ns) for p in paths};had_torch="torch" in sys.modules
        warm=bench.admit_training(common.WARM,"warm",self.implementation,self.plan,self.initial)
        sp=bench.admit_selfplay(common.SELFPLAY,self.warm,self.implementation,self.plan)
        self.assertEqual(warm["selected_step"],1000);self.assertEqual(len(warm["checkpoints"]),20);self.assertEqual(warm["weights_sha256"],self.warm.id)
        self.assertEqual(sp["summary"]["games"],64);self.assertEqual(sp["summary"]["terminal_games"],64);self.assertGreater(sp["summary"]["retained_rows"],0)
        self.assertEqual(had_torch,"torch" in sys.modules);self.assertEqual(before,{str(p):(common.sha(p),p.stat().st_mtime_ns) for p in paths})
    def test_implementation_checksum_and_rehashed_source_drift_refused(self):
        body=copy.deepcopy(self.implementation)
        with self.temp() as folder:
            path=Path(folder)/"implementation.json";path.write_text(config.canonical(body),encoding="utf-8")
            if set(bench.ROLE_PATHS.values())<=set(body["sources"]):self.assertEqual(bench.implementation_manifest(path),body)
            body["sources"]["congklak/puct.py"]="0"*64;body["sha256"]=config.digest({k:v for k,v in body.items() if k!="sha256"});path.write_text(config.canonical(body),encoding="utf-8")
            with self.assertRaises(ValueError):bench.implementation_manifest(path)
    def test_checksum_valid_missing_mandatory_science_inputs_refused(self):
        with self.temp() as folder:
            path=Path(folder)/"implementation.json"
            for missing in bench.SCIENCE_REQUIRED:
                body=copy.deepcopy(self.implementation);del body["science_inputs"][missing]
                body["sha256"]=config.digest({k:v for k,v in body.items() if k!="sha256"})
                path.write_text(config.canonical(body),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"mandatory Phase6 science input mapping"):
                    bench.implementation_manifest(path)
    def test_incomplete_result_and_wrong_stage_exports_refused(self):
        with self.temp() as folder:
            model=self.clone(folder);result_path=Path(folder)/"result.json";result=json.loads(result_path.read_text());result["status"]="interrupted";result_path.write_text(config.canonical(result),encoding="utf-8")
            with self.assertRaises(ValueError):bench.admit_training(model,"warm",self.implementation,self.plan,self.initial)
        with self.assertRaises((ValueError,KeyError)):bench.admit_training(common.WARM,"selfplay",self.implementation,self.plan,self.warm,{})
    def test_selected_catalog_metric_and_pt_checksum_corruption_refused(self):
        with self.temp() as folder:
            model=self.clone(folder)
            with closing(sqlite3.connect(Path(folder)/"catalog.sqlite3")) as db:
                body=json.loads(db.execute("SELECT body FROM checkpoints WHERE step=1000").fetchone()[0]);body["metrics"]["policy"]["uniform_stratum_teacher_cross_entropy"]=1000
                db.execute("UPDATE checkpoints SET body=?,hash=? WHERE step=1000",(config.canonical(body),config.digest(body)))
                db.commit()
            with self.assertRaises(ValueError):bench.admit_training(model,"warm",self.implementation,self.plan,self.initial)
        with self.temp() as folder:
            model=self.clone(folder)
            with closing(sqlite3.connect(Path(folder)/"catalog.sqlite3")) as db:name=db.execute("SELECT name FROM checkpoints WHERE step=1000").fetchone()[0]
            with (Path(folder)/name).open("ab") as stream:stream.write(b"corrupt")
            with self.assertRaises(ValueError):bench.admit_training(model,"warm",self.implementation,self.plan,self.initial)
    def test_checksum_valid_incomplete_dependency_mapping_refused(self):
        with self.temp() as folder:
            model_path=self.clone(folder);envelope=json.loads(model_path.read_text());identity=envelope["model"]["identity"]
            del identity["source_dependencies_sha256"]["congklak.features_relay"];envelope["sha256"]=config.digest(envelope["model"]);model_path.write_text(config.canonical(envelope),encoding="utf-8")
            result_path=Path(folder)/"result.json";result=json.loads(result_path.read_text());result["identity"]=identity;result_path.write_text(config.canonical(result),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"dependency-module set"):bench.admit_training(model_path,"warm",self.implementation,self.plan,self.initial)
    def test_sp_generator_model_lineage_refused(self):
        with self.assertRaises(ValueError):bench.admit_selfplay(common.SELFPLAY,self.initial,self.implementation,self.plan)

if __name__=="__main__":unittest.main(verbosity=2)
