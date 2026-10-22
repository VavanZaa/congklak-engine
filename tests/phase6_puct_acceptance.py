"""PUCT admission, literal seed transitions, completed backups and backend parity."""
from __future__ import annotations
import math
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import unittest
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
import reference as ref
from congklak import config
from congklak.native import Native,flatten
from congklak.puct import CancellationToken,forest,search,legal_priors
from congklak.phase6_inference import BatchEvaluator,InferenceStopped,VALUE_TOLERANCE_SEEDS,LOGIT_TOLERANCE

MODEL=Path(os.environ.get("PHASE6_PUCT_MODEL",str(ROOT/"data/training-stage5-deeper-h8-f127-seed2718/model.json")))
def position(counts,actor=0):
    total=sum(map(sum,counts));return {"pits":counts,"stores":[(98-total)//2,98-total-(98-total)//2],"controller":actor}

def prediction(p):
    active=sum(map(sum,p["pits"]));actor=p["controller"]
    if active and not sum(p["pits"][actor]):actor=1-actor
    return {"value_player0":p["stores"][0]-p["stores"][1]+active*.25*(1 if actor==0 else -1),
            "policy_logits":[float(6-a)/10 for a in range(7)] if active else [0.0]*7,"feature_simulations":0,"feature_deposits":0}

class Fixture:
    max_batch=8
    def __init__(self):self.batches=[]
    def evaluate_many(self,positions):self.batches.append(len(positions));return [prediction(p) for p in positions]

def literal_puct(p,simulations):
    """Different implementation using the standalone seed-by-seed interpreter."""
    def make(q):
        normal=ref.normalize(q);q=ref.Position(7,q.counts,normal.controller)
        return {"p":q,"actor":normal.controller,"legal":ref.legal_actions(q),"terminal":normal.terminal,"children":{},"counts":[0]*7,"totals":[0.0]*7,"prior":None,"bootstrap":(q.counts[7]-q.counts[15])/98}
    def expand(node):
        q=node["p"];p={"pits":[list(q.counts[:7]),list(q.counts[8:15])],"stores":[q.counts[7],q.counts[15]],"controller":q.controller};row=prediction(p)
        logits=row["policy_logits"];top=max(logits[a] for a in node["legal"]);z=sum(math.exp(logits[a]-top) for a in node["legal"])
        node["prior"]=[math.exp(logits[a]-top)/z if a in node["legal"] else 0 for a in range(7)];node["bootstrap"]=row["value_player0"]/98;return node["bootstrap"]
    root=make(ref.Position(7,tuple(flatten(p)),p["controller"]));expand(root)
    for _ in range(simulations):
        node=root;path=[]
        while not node["terminal"] and node["prior"] is not None:
            count=sum(node["counts"]);scores={}
            for a in node["legal"]:
                mean=node["totals"][a]/node["counts"][a] if node["counts"][a] else node["bootstrap"]
                scores[a]=(mean if node["actor"]==0 else -mean)+1.5*node["prior"][a]*math.sqrt(max(1,count))/(1+node["counts"][a])
            action=sorted(scores,key=lambda a:(-scores[a],a))[0];path.append((node,action))
            if action not in node["children"]:node["children"][action]=make(ref.apply(node["p"],action).position)
            node=node["children"][action]
        value=(node["p"].counts[7]-node["p"].counts[15])/98 if node["terminal"] else expand(node)
        for ancestor,action in path:ancestor["counts"][action]+=1;ancestor["totals"][action]+=value
    return root["counts"],[root["totals"][a]/root["counts"][a]*98 if root["counts"][a] else None for a in range(7)]

class Tests(unittest.TestCase):
    def test_literal_actor_sensitive_backups_and_seed_conservation(self):
        roots=[position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]),position([[0]*7,[1,2,0,0,0,0,1]],0),position([[2,1,0,0,0,0,0],[0,0,1,0,0,1,0]],1),config.position()]
        native=Native();extra=False;passed=False
        for p in roots:
            normal=native.normalize(flatten(p),actor=p["controller"],total=98)
            passed|=bool(normal["passes"])
            for a in normal["legal_actions"]:
                transition=native.apply(flatten(normal["position"]),actor=normal["normalized_controller"],action=a,total=98,trace=False)
                self.assertEqual(sum(flatten(transition["position"])),98)
                extra|=not transition["terminal"] and transition["position"]["controller"]==normal["normalized_controller"]
            actual=search(p,Fixture(),simulations=64);counts,means=literal_puct(p,64)
            self.assertEqual(actual["visits"],counts);self.assertEqual(actual["completed_simulations"],64);self.assertEqual(sum(counts),64)
            self.assertEqual(actual["work"]["root_expansions"],1);self.assertTrue(actual["usable_decision"])
            for a,b in zip(actual["action_mean_value_player0"],means):
                self.assertEqual(a,b)
        self.assertTrue(extra);self.assertTrue(passed)

    def test_forest_independent_fixed_schedule_and_noise_rng(self):
        roots=[position([[a+1,0,0,0,0,0,1],[1,0,0,0,0,0,0]],a%2) for a in range(8)]
        evaluator=Fixture();batched=forest(roots,evaluator,simulations=64)
        self.assertTrue(all(1<=n<=8 for n in evaluator.batches));self.assertTrue(8 in evaluator.batches)
        for p,r in zip(roots,batched):
            alone=search(p,Fixture(),simulations=64)
            for key in ("visits","policy","estimated_value_player0","action_mean_value_player0","completed_simulations"):self.assertEqual(r[key],alone[key])
        a=forest(roots,Fixture(),simulations=12,rng=123,root_noise={"alpha":.3,"fraction":.25})
        b=forest(roots,Fixture(),simulations=12,rng=123,root_noise={"alpha":.3,"fraction":.25})
        for x,y in zip(a,b):self.assertEqual(x["visits"],y["visits"]);self.assertEqual(x["root_priors"],y["root_priors"])

    def test_legal_prior_terminal_and_bounds(self):
        self.assertEqual(legal_priors([100,0,-100,100,100,100,100],(1,2))[0],0)
        self.assertAlmostEqual(sum(legal_priors([0]*7,(1,2))),1)
        for logits in ([float("nan")]*7,[0]*6):
            with self.assertRaises(Exception):legal_priors(logits,(0,))
        terminal={"pits":[[0]*7,[0]*7],"stores":[73,25],"controller":1}
        class Bomb:
            def evaluate_many(self,p):raise AssertionError("Terminal called neural backend")
        r=search(terminal,Bomb());self.assertEqual(r["estimated_value_player0"],48);self.assertIsNone(r["chosen"]);self.assertEqual(r["completed_simulations"],0);self.assertEqual(r["proof_status"],"terminal_outcome")
        r=search(config.position(),Fixture(),max_nodes=1);self.assertEqual(r["nodes"],1);self.assertEqual(r["stop_reason"],"node_budget");self.assertEqual(r["completed_simulations"],0);self.assertFalse(r["usable_decision"])
        with self.assertRaises(Exception):forest([config.position()]*9,Fixture())

    def test_per_game_rng_serialized_midgame_recovery(self):
        roots=[position([[1,0,0,0,0,0,1],[1,0,0,0,0,0,0]]),position([[0,1,0,0,0,0,1],[1,0,0,0,0,1,0]],1)]
        noise={"alpha":.3,"fraction":.25};native=Native()
        def tuple_state(x):return tuple(tuple_state(v) for v in x) if isinstance(x,list) else x
        def advance(states,rngs,plies,evaluator):
            records=[]
            for _ in range(plies):
                active=[i for i,p in enumerate(states) if any(map(sum,p["pits"]))]
                if not active:break
                rows=forest([states[i] for i in active],evaluator,simulations=16,rng=[rngs[i] for i in active],root_noise=noise,native=native)
                for i,row in zip(active,rows):
                    records.append((i,row["visits"],row["root_priors"],row["estimated_value_player0"]))
                    t=native.apply(flatten(row["position"]),actor=row["position"]["controller"],action=row["chosen"],total=98,trace=False)
                    self.assertTrue(t["ok"]);states[i]=t["position"]
            return records
        seeds=[random.Random(111),random.Random(222)];states=config.position(roots[0]),config.position(roots[1]);states=list(states)
        with BatchEvaluator(MODEL) as evaluator:
            prefix=advance(states,seeds,3,evaluator);saved=json.loads(json.dumps({"positions":states,"rng":[r.getstate() for r in seeds]}))
            future=advance(states,seeds,128,evaluator)
        restored=[random.Random(),random.Random()]
        for rng,state in zip(restored,saved["rng"]):rng.setstate(tuple_state(state))
        replay_states=saved["positions"]
        with BatchEvaluator(MODEL) as fresh_evaluator:replay=advance(replay_states,restored,128,fresh_evaluator)
        self.assertTrue(prefix);self.assertTrue(future);self.assertEqual(future,replay);self.assertEqual(states,replay_states);self.assertTrue(all(not any(map(sum,p["pits"])) for p in states))
        alone=search(roots[0],Fixture(),simulations=8,rng=[random.Random(111)],root_noise=noise)
        batch=forest(roots,Fixture(),simulations=8,rng=[random.Random(111),random.Random(222)],root_noise=noise)[0]
        self.assertEqual(alone["visits"],batch["visits"]);self.assertEqual(alone["root_priors"],batch["root_priors"])
        with self.assertRaises(ValueError):forest(roots,Fixture(),rng=[random.Random(1)])

    def test_parent_bootstrap_fpu_store_offset_invariance(self):
        pits=[[2,1,0,0,0,0,3],[1,0,2,0,1,0,2]]
        first={"pits":pits,"stores":[43,43],"controller":0};shifted={"pits":pits,"stores":[63,23],"controller":0}
        for simulations in (2,64):
            a=search(first,Fixture(),simulations=simulations);b=search(shifted,Fixture(),simulations=simulations)
            self.assertEqual(a["visits"],b["visits"]);self.assertEqual(a["root_priors"],b["root_priors"]);self.assertEqual(a["chosen"],b["chosen"])
            self.assertAlmostEqual(b["estimated_value_player0"]-a["estimated_value_player0"],40)
            self.assertEqual(a["first_play_urgency"],"parent_neural_bootstrap")

    def test_deadline_overshoot_incomplete_batch_and_actual_native_cancel(self):
        class Slow(Fixture):
            def evaluate_many(self,p):time.sleep(.03);return super().evaluate_many(p)
        native=Native();r=search(config.position(),Slow(),time_ms=10,native=native)
        self.assertEqual(r["stop_reason"],"time_budget");self.assertEqual(sum(r["visits"]),0);self.assertGreater(r["timing"]["overshoot_ms"],10);self.assertFalse(r["usable_decision"])
        token=CancellationToken()
        class CancelNative(Native):
            def apply(self,*args,**kwargs):
                token.cancel();answer=super().apply(*args,**kwargs);self.answer=answer;return answer
        native=CancelNative();r=search(config.position(),Fixture(),cancel=token,native=native)
        self.assertFalse(native.answer["ok"]);self.assertEqual(native.answer["error"]["code"],"cancelled")
        self.assertNotIn("position",native.answer);self.assertEqual(r["stop_reason"],"cancelled");self.assertEqual(sum(r["visits"]),0)
        ready=threading.Event();release=threading.Event();token=CancellationToken();result=[]
        class Blocking(Fixture):
            def evaluate_many(self,p):ready.set();release.wait(2);return super().evaluate_many(p)
        worker=threading.Thread(target=lambda:result.append(search(config.position(),Blocking(),cancel=token)));worker.start();self.assertTrue(ready.wait(2));token.cancel();release.set();worker.join(2)
        self.assertFalse(worker.is_alive());self.assertEqual(result[0]["stop_reason"],"cancelled");self.assertEqual(result[0]["completed_simulations"],0)

    def test_native_evaluator_exact_terminal_counter_bounds_and_repeat(self):
        with BatchEvaluator(MODEL) as e:
            roots=[config.position(),position([[0]*7,[1,2,0,0,0,0,1]],0),{"pits":[[0]*7,[0]*7],"stores":[73,25],"controller":0}]
            outputs=e.evaluate_many(roots);self.assertEqual(outputs[-1]["value_player0"],48);self.assertEqual(outputs[-1]["feature_deposits"],0)
            for p,row in zip(roots,outputs):
                reference=e.model.reference(p);self.assertEqual(row["value_player0"],reference[0])
                if sum(map(sum,p["pits"])):
                    features=e.native.feature_vector(p)
                    self.assertEqual(row["feature_deposits"],2*features["feature_deposits"]);self.assertEqual(row["feature_simulations"],2*features["feature_simulations"])
            a=search(roots[0],e,simulations=16);b=search(roots[0],e,simulations=16)
            for key in ("visits","policy","estimated_value_player0","work"):
                if key!="work":self.assertEqual(a[key],b[key])
            self.assertGreater(a["work"]["feature_deposits"],0)
            with self.assertRaises(Exception):e.evaluate_many([roots[0]]*9)
            token=CancellationToken();token.cancel()
            with self.assertRaises(InferenceStopped):e.evaluate_many([roots[0]],cancel=token)
        with self.assertRaises(Exception):e.evaluate_many([roots[0]])

    @unittest.skipUnless(os.environ.get("PHASE6_GPU_ACCEPTANCE")=="1","GPU acceptance requires an explicitly scheduled quiet slot")
    def test_gpu_batchshape_parity_actor_passes_and_fixed_schedule(self):
        from congklak.neural import development_records
        roots=[r["state"] for r in list(development_records(ROOT/"data/stronger-10000-v2"))[:32]]
        roots[:2]=[config.position(),position([[0]*7,[1,2,0,0,0,0,1]],0)]
        with BatchEvaluator(MODEL,"native",max_batch=32) as native,BatchEvaluator(MODEL,"cuda",max_batch=32) as gpu:
            for size in (1,2,4,8,16,32):
                a=native.evaluate_many(roots[:size]);b=gpu.evaluate_many(roots[:size])
                for x,y in zip(a,b):
                    self.assertLessEqual(abs(x["value_player0"]-y["value_player0"]),VALUE_TOLERANCE_SEEDS)
                    self.assertLessEqual(max(abs(v-w) for v,w in zip(x["policy_logits"],y["policy_logits"])),LOGIT_TOLERANCE)
            a=forest(roots[:8],gpu,simulations=16);b=forest(roots[:8],gpu,simulations=16)
            for x,y in zip(a,b):
                self.assertEqual(x["visits"],y["visits"]);self.assertEqual(x["estimated_value_player0"],y["estimated_value_player0"])
                self.assertEqual(x["completed_simulations"],16)

if __name__=="__main__":unittest.main(verbosity=2)
