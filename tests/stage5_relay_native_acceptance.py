"""Action inference, literal minimax ordering equivalence, and actual restart checks."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import queue
import random
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from congklak import config,pilot,features_relay
from congklak.native import ROOT,Native,NativeError,flatten
from congklak.neural_relay import Model,NeuralNative,Analysis,validate_report,FEATURE_ENCODER,RELAY_ENCODER,SCALE,rounded,MODES
import reference as ref

def fixture(path,width=17,input_count=127):
    rng=random.Random(width+76);weights={}
    for name,rows,cols in (("layers.0",width,input_count),("layers.2",width,width),("layers.4",1,width),("policy",7,width)):
        weights[name+".weight"]=[[rng.randrange(-8,9)/128 for _ in range(cols)] for _ in range(rows)]
        weights[name+".bias"]=[rng.randrange(-8,9)/128 for _ in range(rows)]
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        flat=[v for row in value for v in row] if isinstance(value[0],list) else value
        digest.update(name.encode());digest.update(struct.pack("<"+"f"*len(flat),*flat))
    model={"schema":"congklak-mlp-export-v4","encoder":FEATURE_ENCODER if input_count==43 else RELAY_ENCODER,"identity":{"width":width,"input_count":input_count,"rules_digest":config.RULES_DIGEST},
           "weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":model,"sha256":config.digest(model)}));return weights

def literal_features(p,input_count=127):
    actor=ref.normalize(p).controller;counts=[p.counts[row*8+i] for row in (actor,1-actor) for i in range(7)]
    x=[v/98 for v in counts]+[sum(counts)/98]+[int(v>0) for v in counts]+[int(v==1) for v in counts]
    if input_count==43:return x
    for side in (actor,1-actor):
        for pit in range(7):
            if not p.counts[side*8+pit]:x.extend([0.0]*6);continue
            hypo=ref.Position(7,p.counts,side);t=ref.apply(hypo,pit)
            x.extend([(t.position.counts[side*8+7]-p.counts[side*8+7])/98,t.deposits/98,t.relay_pickups/98,
                      float(t.captured>0),float(not t.terminal and ref.normalize(t.position).controller==side),float(t.terminal)])
    return x

def independent(p,w):
    actor=ref.normalize(p).controller;active=98-p.counts[7]-p.counts[15];d=p.counts[7]-p.counts[15]
    if not active:return d,d*SCALE,[0.0]*7
    counts=[p.counts[row*8+i] for row in (actor,1-actor) for i in range(7)]
    x=literal_features(p,len(w["layers.0.weight"][0]))
    for name in ("layers.0","layers.2"):
        x=[max(0,sum(a*v for a,v in zip(row,x))+b) for row,b in zip(w[name+".weight"],w[name+".bias"])]
    z=sum(a*v for a,v in zip(w["layers.4.weight"][0],x))+w["layers.4.bias"][0]
    logits=[sum(a*v for a,v in zip(row,x))+b for row,b in zip(w["policy.weight"],w["policy.bias"])]
    import math
    residual=(active if actor==0 else -active)*math.tanh(z)
    return d+residual,d*SCALE+rounded(residual*SCALE),logits

class RelayAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/"build");self.path=Path(self.temp.name);self.model=self.path/"policy-model.json";self.weights=fixture(self.model)
    def tearDown(self):self.temp.cleanup()

    def test_01_scalar_avx_heads_tails_pass_terminal_symmetries(self):
        rng=random.Random(41)
        for width in (17,32,64,256):
            path=self.path/f"w{width}.json";weights=fixture(path,width)
            with NeuralNative(path,mode="value-none") as avx,NeuralNative(path,backend="scalar",mode="value-none") as scalar:
                positions=[{"pits":[[0]*7,[1,0,0,0,0,0,0]],"stores":[48,49],"controller":0},
                           {"pits":[[0]*7,[0]*7],"stores":[40,58],"controller":1}]
                for _ in range(24):
                    active=rng.randrange(1,99);counts=[0]*14
                    for _ in range(active):counts[rng.randrange(14)]+=1
                    first=rng.randrange(99-active);positions.append({"pits":[counts[:7],counts[7:]],"stores":[first,98-active-first],"controller":rng.randrange(2)})
                for p in positions:
                    feature_literal=literal_features(ref.Position(7,tuple(flatten(p)),p["controller"]))
                    self.assertEqual(features_relay.encode(p)[0],feature_literal)
                    self.assertEqual(avx.feature_vector(p)["features"],feature_literal)
                    a=avx.evaluate(p);self.assertEqual(a,scalar.evaluate(p));literal=independent(ref.Position(7,tuple(flatten(p)),p["controller"]),weights)
                    self.assertAlmostEqual(a["value_player0"],literal[0],delta=1e-10);self.assertEqual(a["value_units"],literal[1])
                    for actual,expected in zip(a["policy_logits"],literal[2]):self.assertAlmostEqual(actual,expected,delta=1e-10)
                    swapped={"pits":p["pits"][::-1],"stores":p["stores"][::-1],"controller":1-p["controller"]};b=avx.evaluate(swapped)
                    self.assertEqual(b["value_units"],-a["value_units"]);self.assertEqual(b["policy_logits"],a["policy_logits"])
                    self.assertEqual(avx.feature_vector(swapped)["features"],feature_literal)
                    if p["stores"][1]:
                        shifted={**p,"stores":[p["stores"][0]+1,p["stores"][1]-1]};b=avx.evaluate(shifted)
                        self.assertEqual(b["value_units"],a["value_units"]+512);self.assertEqual(b["policy_logits"],a["policy_logits"])
                        self.assertEqual(avx.feature_vector(shifted)["features"],feature_literal)
                self.assertEqual(avx.measure_inference(positions,2)["checksum"],scalar.measure_inference(positions,2)["checksum"])

    def test_02_full_literal_minimax_all_modes_tt_pvs(self):
        rng=random.Random(51)
        for mode in MODES:
            @lru_cache(None)
            def solve(p,depth):
                if ref.normalize(p).terminal or depth==0:return ref.evaluate(p)*SCALE if mode.startswith("h-") else independent(p,self.weights)[1]
                return (max if ref.normalize(p).controller==0 else min)(solve(ref.apply(p,a).position,depth-1) for a in ref.legal_actions(p))
            with NeuralNative(self.model,mode=mode) as n:
                for _ in range(16):
                    active=rng.randrange(2,13);counts=[0]*16
                    for _ in range(active):counts[rng.choice(list(range(7))+list(range(8,15)))]+=1
                    counts[7]=rng.randrange(99-active);counts[15]=98-active-counts[7];actor=rng.randrange(2)
                    p=ref.Position(7,tuple(counts),actor);root=pilot.from_key(bytes(counts+[actor]));expected={a:solve(ref.apply(p,a).position,3) for a in ref.legal_actions(p)}
                    for tt,pvs in ((0,False),(1,False),(0,True),(1,True)):
                        r=n.neural_analyze(root,max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"];validate_report(r,root,n)
                        self.assertEqual({c["pit"]:c["value_units"] for c in r["candidates"]},expected);self.assertEqual(r["value_units"],solve(p,4))

    def test_03_h_fixed_horizon_original_and_feature_work(self):
        baseline=Native();rng=random.Random(77);root=config.position()
        with NeuralNative(self.model,mode="h-none") as ordered,NeuralNative(self.model,mode="h-none") as plain:
            for _ in range(16):
                a=baseline.analyze(flatten(root),actor=root["controller"],total=98,max_depth=4,time_ms=0)["report"]
                b=ordered.neural_analyze(root,max_depth=4,time_ms=0)["report"];c=plain.neural_analyze(root,max_depth=4,time_ms=0)["report"]
                self.assertEqual((b["value_units"],b["best_moves"],b["pv"]),(c["value_units"],c["best_moves"],c["pv"]))
                self.assertEqual(b["value_units"],a["depth_result"]["value_player0"]*SCALE);self.assertEqual(b["best_moves"],a["best_moves"]);self.assertEqual(b["pv"],a["pv"])
                self.assertEqual(c["nn_evaluations"],0);self.assertEqual(c["proof_status"],"finite_horizon_result")
                normal=baseline.normalize(flatten(root),actor=root["controller"],total=98)
                if normal["terminal"]:break
                root=baseline.apply(flatten(root),actor=normal["normalized_controller"],action=rng.choice(normal["legal_actions"]),total=98,trace=False)["position"]
        with NeuralNative(self.model,mode="value-none") as learned:
            r=learned.neural_analyze(config.position(),max_depth=2,time_ms=0)["report"]
            validate_report(r,config.position(),learned)
            self.assertGreater(r["feature_simulations"],0);self.assertGreater(r["feature_deposits"],0)
            self.assertGreaterEqual(r["deposits"],r["feature_deposits"]);self.assertEqual(r["policy_ordered_nodes"],0)

    def test_04_mode_threshold_shapes_lifetime_and_cancel(self):
        for threshold in (0,1,129,2.0,True):
            with self.assertRaises((NativeError,ValueError)):NeuralNative(self.model,policy_min_depth=threshold)
        with self.assertRaises(NativeError):NeuralNative(self.model,mode="invalid")
        envelope=json.loads(self.model.read_text());envelope["model"]["weights"]["policy.weight"].pop();envelope["sha256"]=config.digest(envelope["model"])
        bad=self.path/"bad.json";bad.write_text(json.dumps(envelope))
        with self.assertRaises(NativeError):Model(bad)
        with NeuralNative(self.model,mode="h-none") as n:
            self.assertIsNone(n.lib.cc_nn_create_model_relay(n.model.array,len(n.model.array),17,15,n.model.id.encode(),0,1,2))
            result=n.neural_analyze(config.position(),max_depth=16,time_ms=0,node_budget=1)
            self.assertIsNone(result["report"]);self.assertEqual(result["stop_reason"],"node_budget")
            n.close()
            with self.assertRaises(NativeError):n.evaluate(config.position())
        with NeuralNative(self.model,mode="value-none") as n:
            result=n.neural_analyze(config.position(),max_depth=128,time_ms=20)
            self.assertEqual(result["stop_reason"],"time_budget")
            self.assertGreater(result["deposits"],0);self.assertLess(result["elapsed_ms"],250)
            with n.make_control(128,0,0,512,16,True) as control:
                control.cancel();result=n.neural_analyze(config.position(),max_depth=128,time_ms=0,control=control)
                self.assertEqual(result["stop_reason"],"cancelled");self.assertIsNone(result["report"])

    def test_05_hard_kill_after_commit_resume_and_mode_identity(self):
        root=config.position();run=self.path/"run"
        worker='''import json,sys
from congklak.neural_relay import NeuralNative,Analysis
with NeuralNative(sys.argv[1],mode='value-none') as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2:sys.stdin.readline()
 j.run(5,0,on_event=event)
 j.close()
'''
        child=subprocess.Popen([sys.executable,"-c",worker,str(self.model),str(run),json.dumps(root)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        messages=queue.Queue()
        def reader():
            for line in child.stdout:messages.put(json.loads(line))
        t=threading.Thread(target=reader,daemon=True);t.start()
        try:
            while messages.get(timeout=60)["choice_depth"]!=2:pass
            child.kill();child.wait(timeout=20)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=20)
            t.join(timeout=5)
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
        with NeuralNative(self.model,mode="value-none") as n:
            job=Analysis(run,root,n)
            try:self.assertEqual(job.last["choice_depth"],2);resumed=job.run(5,0)["report"]
            finally:job.close()
            fresh=n.neural_analyze(root,max_depth=5,time_ms=0)["report"]
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
        for mode,threshold in (("h-none",2),("value-none",3)):
            with NeuralNative(self.model,mode=mode,policy_min_depth=threshold) as other:
                with self.assertRaises(pilot.PilotError):Analysis(run,root,other)

    def test_06_count_control_and_unclipped_relay_work(self):
        path=self.path/"control43.json";weights=fixture(path,17,43)
        root=config.position()
        with NeuralNative(path,mode="value-none") as n,NeuralNative(path,mode="value-none",backend="scalar") as scalar:
            self.assertEqual(n.evaluate(root),scalar.evaluate(root))
            self.assertEqual(n.feature_vector(root)["features"],literal_features(ref.Position(7,tuple(flatten(root)),0),43))
            r=n.neural_analyze(root,max_depth=3,time_ms=0)["report"];validate_report(r,root,n)
            self.assertEqual(r["feature_simulations"],0);self.assertEqual(r["feature_deposits"],0)
            self.assertEqual(n.evaluate(root)["value_units"],independent(ref.Position(7,tuple(flatten(root)),0),weights)[1])
        rng=random.Random(932);found=False
        with NeuralNative(self.model,mode="value-none") as n:
            for _ in range(100):
                pits=[0]*14
                for _ in range(98):pits[rng.randrange(14)]+=1
                root={"pits":[pits[:7],pits[7:]],"stores":[0,0],"controller":rng.randrange(2)}
                expected=literal_features(ref.Position(7,tuple(flatten(root)),root["controller"]))
                if any(expected[44+i*6]>1 for i in range(14)):
                    self.assertEqual(n.feature_vector(root)["features"],expected);found=True;break
            self.assertTrue(found,"Unclipped deposit fixture absent")

if __name__=="__main__":unittest.main()
