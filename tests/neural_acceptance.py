"""Independent neural recurrence, export and durable recovery acceptance."""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from congklak import config,pilot
from congklak.neural import Model,NeuralNative,Analysis,validate_report,rounded,SCALE,development_records
from congklak.native import ROOT,Native,Control,flatten,NativeError
import reference as ref

WORK=ROOT/"build/neural-acceptance"
WORK.mkdir(parents=True,exist_ok=True)

def fixture(path):
    rng=random.Random(9); width=16; weights={}
    for prefix,rows,cols in (("layers.0",width,15),("layers.2",width,width),("layers.4",1,width)):
        weights[prefix+".weight"]=[[rng.randrange(-8,9)/64 for _ in range(cols)] for _ in range(rows)]
        weights[prefix+".bias"]=[rng.randrange(-8,9)/64 for _ in range(rows)]
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        flat=[v for row in value for v in row] if isinstance(value[0],list) else value
        digest.update(name.encode()); digest.update(struct.pack("<"+"f"*len(flat),*flat))
    m={"schema":"congklak-mlp-export-v1","encoder":"current-controller-pit14-and-active15/store-offset-exact-v1",
       "identity":{"width":width,"rules_digest":config.RULES_DIGEST},"weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":m,"sha256":config.digest(m)})); return weights

def independent_value(p,w):
    actor=ref.normalize(p).controller
    active=sum(p.counts)-p.counts[7]-p.counts[15]
    x=[p.counts[row*8+i]/98 for row in (actor,1-actor) for i in range(7)]+[active/98]
    for prefix in ("layers.0","layers.2","layers.4"):
        x=[sum(a*v for a,v in zip(row,x))+b for row,b in zip(w[prefix+".weight"],w[prefix+".bias"])]
        if prefix!="layers.4": x=[max(0,v) for v in x]
    return (p.counts[7]-p.counts[15])*SCALE+rounded((active if actor==0 else -active)*math.tanh(x[0])*SCALE)

class NeuralAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=WORK); self.path=Path(self.temp.name)
        self.model_path=self.path/"model.json"; self.weights=fixture(self.model_path)
    def tearDown(self): self.temp.cleanup()

    def test_01_large_export_parity_every_development_root(self):
        model=Model()
        with NeuralNative(model) as n:
            count=0
            for row in development_records():
                actual=n.evaluate(row["state"]); value,units=model.reference(row["state"])
                self.assertAlmostEqual(actual["value_player0"],value,delta=1e-10)
                self.assertEqual(actual["value_units"],units); count+=1
            self.assertEqual(count,1416)

    def test_02_independent_minimax_and_four_way_tt_pvs(self):
        rng=random.Random(123)
        @lru_cache(None)
        def solve(p,depth):
            if ref.normalize(p).terminal or depth==0: return independent_value(p,self.weights)
            return (max if ref.normalize(p).controller==0 else min)(solve(ref.apply(p,a).position,depth-1) for a in ref.legal_actions(p))
        with NeuralNative(self.model_path) as n:
            for index in range(48):
                active=rng.randrange(2,11); counts=[0]*16
                for _ in range(active): counts[rng.choice(list(range(7))+list(range(8,15)))]+=1
                counts[7]=rng.randrange(99-active); counts[15]=98-active-counts[7]
                p=ref.Position(7,tuple(counts),rng.randrange(2)); root=pilot.from_key(bytes(counts+[p.controller]))
                for tt,pvs in ((0,False),(1,False),(0,True),(1,True)):
                    report=n.neural_analyze(root,max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"]
                    validate_report(report,root,n)
                    expected={a:solve(ref.apply(p,a).position,3) for a in ref.legal_actions(p)}
                    actual={c["pit"]:c["value_units"] for c in report["candidates"]}
                    self.assertEqual(actual,expected)
                    self.assertEqual(report["value_units"],solve(p,4))
                    if expected:
                        best=(max if ref.normalize(p).controller==0 else min)(expected.values())
                        self.assertEqual(report["best_moves"],[a for a,v in expected.items() if v==best])

    def test_03_frozen_h_v1_semantics_and_score_symmetries(self):
        old=Native()
        with NeuralNative(self.model_path) as n:
            rng=random.Random(12); p=config.position()
            for _ in range(24):
                a=old.analyze(flatten(p),actor=p["controller"],total=98,max_depth=3,time_ms=0)["report"]
                b=n.analyze(flatten(p),actor=p["controller"],total=98,max_depth=3,time_ms=0)["report"]
                self.assertEqual((a["best_moves"],a["depth_result"],a["pv"]),(b["best_moves"],b["depth_result"],b["pv"]))
                value=n.evaluate(p)["value_units"]
                swapped={"pits":p["pits"][::-1],"stores":p["stores"][::-1],"controller":1-p["controller"]}
                self.assertEqual(n.evaluate(swapped)["value_units"],-value)
                if p["stores"][1]:
                    shifted={**p,"stores":[p["stores"][0]+1,p["stores"][1]-1]}
                    self.assertEqual(n.evaluate(shifted)["value_units"],value+2*SCALE)
                normal=old.normalize(flatten(p),actor=p["controller"],total=98)
                if normal["terminal"]: break
                p=old.apply(flatten(normal["position"]),actor=normal["normalized_controller"],action=rng.choice(normal["legal_actions"]),total=98,trace=False)["position"]

    def test_04_analysis_hard_kill_resume_and_checkpoint_validation(self):
        root=config.position(); position=self.path/"position.json"; position.write_text(json.dumps(root))
        run=self.path/"run"
        worker='''import json,sys
from congklak.neural import NeuralNative,Analysis
with NeuralNative(sys.argv[1]) as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2: sys.stdin.readline()
 j.run(5,0,on_event=event)
 j.close()
'''
        child=subprocess.Popen([sys.executable,"-c",worker,str(self.model_path),str(run),json.dumps(root)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        messages=queue.Queue()
        def reader():
            for line in child.stdout: messages.put(json.loads(line))
        t=threading.Thread(target=reader,daemon=True); t.start()
        try:
            while messages.get(timeout=60)["choice_depth"]!=2: pass
            child.kill(); child.wait(timeout=20)
        finally:
            if child.poll() is None: child.kill(); child.wait(timeout=20)
            t.join(timeout=5); child.stdin.close(); child.stdout.close(); child.stderr.close()
        with NeuralNative(self.model_path) as n:
            j=Analysis(run,root,n)
            try:
                self.assertEqual(j.last["choice_depth"],2)
                with self.assertRaises((NativeError,OSError)): Analysis(run,root,n)
                resumed=j.run(5,0)["report"]
            finally: j.close()
            fresh=n.neural_analyze(root,max_depth=5,time_ms=0)["report"]
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
            bad={**resumed,"input_controller":1-root["controller"]}
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:
                db.execute("UPDATE iterations SET body=?,hash=? WHERE depth=5",(config.canonical(bad),config.digest(bad))); db.commit()
            with self.assertRaises(NativeError): Analysis(run,root,n)

    def test_05_interruption_before_first_iteration_has_no_move(self):
        with NeuralNative(self.model_path) as n:
            result=n.neural_analyze(config.position(),max_depth=16,time_ms=0,node_budget=1)
            self.assertIsNone(result["report"])
            self.assertEqual(result["stop_reason"],"node_budget")

    def test_06_negative_or_mismatched_control_limits_refused(self):
        with NeuralNative(self.model_path) as n:
            with self.assertRaises(ValueError): n.neural_analyze(config.position(),time_ms=-1)
            with self.assertRaises(ValueError): n.neural_analyze(config.position(),node_budget=-1)
            with n.make_control(max_depth=3,time_ms=0) as control:
                with self.assertRaises(NativeError): n.neural_analyze(config.position(),max_depth=4,time_ms=0,control=control)

if __name__=="__main__": unittest.main()
