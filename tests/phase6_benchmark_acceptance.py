"""CPU-only real-engine strength-admission and corruption fixtures."""
import copy
import importlib.util
import math
from pathlib import Path
import sqlite3
import sys
import unittest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
from congklak import config
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative
from phase6_puct_acceptance import Fixture,position
spec=importlib.util.spec_from_file_location("phase6_benchmark_under_test",ROOT/"tools/benchmark-phase6.py");bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

def protocol(p):
    native=Native();r=native.analyze(flatten(p),actor=p["controller"],total=98,max_depth=2,time_ms=0,node_budget=0,memory_mb=8192,tt_mb=0,keep_history=False)["report"]
    reference={"kind":"finite_choice_margin","target_player0":r["depth_result"]["value_player0"],"best_moves":r["best_moves"],"action_scores":[dict(pit=a["pit"],value_player0=a["depth_result"]["value_player0"]) for a in r["candidates"]]}
    return dict(max_depth=16,tt_mb=16,use_pvs=True,puct=dict(simulations=100000,max_nodes=2048,c_puct=1.5,root_noise=None),budgets_ms=[5,10],arena_ms=5,arena_max_choices=512,
                engines=["H_v1","H8_relay_CPU","warm_native","warm_cuda","updated_cuda"],cpu_declared=dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1),
                cases=[dict(state=p,state_key="fixture",cohort="fixture",stratum="fixture/ordinary",reference=reference)],arena_starts=[dict(state=p,family="fixture")],bootstrap_samples=4096,bootstrap_seed=20261017)

class Tests(unittest.TestCase):
    def setUp(self):self.p=position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]);self.plan=protocol(self.p)
    def test_real_h_relay_puct_timing_and_unusable_fallback(self):
        h=Native()
        with NeuralNative(ROOT/"data/training-stage5-deeper-h8-f127-seed2718/model.json",mode="value-none") as relay:
            for engine in (("h",h),("relay",relay),("puct",Fixture())):
                answer=bench.decide(engine,self.p,5,self.plan);bench.validate_decision(answer,self.p,5,self.plan)
                self.assertTrue(answer["usable_decision"]);self.assertGreater(answer["wall_ms"],0);self.assertEqual(answer["overshoot_ms"],max(0,answer["wall_ms"]-5))
        plan=copy.deepcopy(self.plan);plan["puct"]["max_nodes"]=1
        answer=bench.decide(("puct",Fixture()),self.p,5,plan)
        self.assertFalse(answer["usable_decision"]);self.assertIsNone(answer["chosen"]);self.assertEqual(answer["report"]["completed_simulations"],0)
    def test_checksum_valid_reference_and_visit_corruption_refused(self):
        h=Native();answer=bench.decide(("h",h),self.p,5,self.plan);puct=bench.decide(("puct",Fixture()),self.p,5,self.plan)
        body=dict(index=0,state_key="fixture",cohort="fixture",stratum="fixture/ordinary",reference=self.plan["cases"][0]["reference"],orders={},measurements={},cpu_verified=bench.verified(self.plan))
        q={r["pit"]:r["value_player0"] for r in body["reference"]["action_scores"]}
        for bi,ms in enumerate(self.plan["budgets_ms"]):
            body["orders"][str(ms)]=bench.expected_order(0,bi,self.plan);body["measurements"][str(ms)]={}
            for name in self.plan["engines"]:
                value=copy.deepcopy(answer if name=="H_v1" else puct)
                value["requested_ms"]=ms;value["wall_ms"]=max(value["wall_ms"],ms);value["overshoot_ms"]=max(0,value["wall_ms"]-ms)
                if name=="H8_relay_CPU":
                    value=copy.deepcopy(answer);value["kind"]="relay";value["report"]["proof_status"]="finite_horizon_neural_estimate";value["requested_ms"]=ms;value["wall_ms"]=max(value["wall_ms"],ms);value["overshoot_ms"]=max(0,value["wall_ms"]-ms)
                choice=value["chosen"];body["measurements"][str(ms)][name]=dict(decision=value,reference_optimal=choice in body["reference"]["best_moves"] if choice is not None else None,reference_regret=body["reference"]["target_player0"]-q[choice] if choice is not None else None)
        bench.validate_case(body,0,self.plan)
        altered=copy.deepcopy(body);altered["measurements"]["5"]["H_v1"]["reference_regret"]+=1;config.digest(altered)
        with self.assertRaises(ValueError):bench.validate_case(altered,0,self.plan)
        altered=copy.deepcopy(puct);altered["report"]["visits"][1]=1;altered["report"]["completed_simulations"]+=1;altered["completed_simulations"]+=1
        with self.assertRaises(ValueError):bench.validate_decision(altered,self.p,5,self.plan)
        altered=copy.deepcopy(answer);altered["wall_ms"]=0;altered["overshoot_ms"]=0
        with self.assertRaises(ValueError):bench.validate_decision(altered,self.p,5,self.plan)
    def test_family_bootstrap_deterministic_and_empty_scope(self):
        first=bench.interval([1,2,3,4],self.plan);self.assertEqual(first,bench.interval([1,2,3,4],self.plan));self.assertEqual(len(first),2);self.assertIsNone(bench.interval([],self.plan))
        self.assertGreater(first[0],0);self.assertLessEqual(first[1],4)

if __name__=="__main__":unittest.main(verbosity=2)
