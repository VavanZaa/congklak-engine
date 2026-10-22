"""Packed inference numeric parity, feature tails, and durable interruption acceptance."""
from contextlib import closing
import hashlib
import json
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
from congklak.native import ROOT,NativeError
from congklak.neural import NeuralNative as FrozenNative,development_records
from congklak.neural_fast import Model,NeuralNative,Analysis,validate_report,ENCODER,FEATURE_ENCODER

def fixture(path,width=17,inputs=43):
    rng=random.Random(width+inputs); weights={}
    for prefix,rows,cols in (("layers.0",width,inputs),("layers.2",width,width),("layers.4",1,width)):
        weights[prefix+".weight"]=[[rng.randrange(-8,9)/128 for _ in range(cols)] for _ in range(rows)]
        weights[prefix+".bias"]=[rng.randrange(-8,9)/128 for _ in range(rows)]
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        flat=[v for row in value for v in row] if isinstance(value[0],list) else value
        digest.update(name.encode()); digest.update(struct.pack("<"+"f"*len(flat),*flat))
    model={"schema":"congklak-mlp-export-v2" if inputs==43 else "congklak-mlp-export-v1","encoder":FEATURE_ENCODER if inputs==43 else ENCODER,
           "identity":{"width":width,"input_count":inputs,"rules_digest":config.RULES_DIGEST},"weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":model,"sha256":config.digest(model)}));return path

class SpeedAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/"build");self.path=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_01_frozen_raw_and_units_every_development_root(self):
        roots=[r["state"] for r in development_records()]+[r["state"] for r in development_records(ROOT/"data/stronger-10000-v2")]
        self.assertEqual(len(roots),2416)
        for path in ("data/training-stronger2-w64-seed2718/model.json","data/training-stronger2-w256-seed31415/model.json"):
            with FrozenNative(ROOT/path) as old, NeuralNative(ROOT/path) as fast, NeuralNative(ROOT/path,backend="scalar") as scalar:
                for p in roots:
                    expected=old.evaluate(p)
                    self.assertEqual(fast.evaluate(p),expected);self.assertEqual(scalar.evaluate(p),expected)

    def test_02_feature_export_every_development_root(self):
        model=Model(ROOT/"data/training-stage5-f43-w32-seed2718/model.json")
        with NeuralNative(model) as fast, NeuralNative(model,backend="scalar") as scalar:
            for row in development_records(ROOT/"data/stronger-10000-v2"):
                expected=model.reference(row["state"]);a=fast.evaluate(row["state"]);b=scalar.evaluate(row["state"])
                self.assertEqual(a,b);self.assertEqual(a["value_units"],expected[1]);self.assertAlmostEqual(a["value_player0"],expected[0],delta=1e-10)

    def test_03_arbitrary_width_tails_and_invariants(self):
        rng=random.Random(25)
        for inputs,width in ((15,16),(15,17),(43,17),(43,31),(43,256)):
            model=Model(fixture(self.path/f"{inputs}-{width}.json",width,inputs))
            with NeuralNative(model) as fast, NeuralNative(model,backend="scalar") as scalar:
                positions=[{"pits":[[0]*7,[1,0,0,0,0,0,0]],"stores":[49,48],"controller":0},
                           {"pits":[[0]*7,[0]*7],"stores":[98,0],"controller":1}]
                for _ in range(40):
                    active=rng.randrange(1,99);counts=[0]*14
                    for _ in range(active):counts[rng.randrange(14)]+=1
                    first=rng.randrange(99-active)
                    positions.append({"pits":[counts[:7],counts[7:]],"stores":[first,98-active-first],"controller":rng.randrange(2)})
                for p in positions:
                    a=fast.evaluate(p);self.assertEqual(a,scalar.evaluate(p));expected=model.reference(p)
                    self.assertEqual(a["value_units"],expected[1]);self.assertAlmostEqual(a["value_player0"],expected[0],delta=1e-10)
                    swapped={"pits":p["pits"][::-1],"stores":p["stores"][::-1],"controller":1-p["controller"]}
                    self.assertEqual(fast.evaluate(swapped)["value_units"],-a["value_units"])
                    if p["stores"][1]:
                        shifted={**p,"stores":[p["stores"][0]+1,p["stores"][1]-1]}
                        self.assertEqual(fast.evaluate(shifted)["value_units"],a["value_units"]+512)
                measurement=fast.measure_inference(positions,2)
                self.assertEqual(measurement["checksum"],scalar.measure_inference(positions,2)["checksum"])

    def test_04_input_shape_backend_and_lifetime_admission(self):
        path=fixture(self.path/"model.json")
        for bad in (15,42,43.0,True):
            envelope=json.loads(path.read_text());envelope["model"]["identity"]["input_count"]=bad;envelope["sha256"]=config.digest(envelope["model"])
            invalid=self.path/f"bad-{bad}.json";invalid.write_text(json.dumps(envelope))
            with self.assertRaises((NativeError,ValueError)):Model(invalid)
        with self.assertRaises(NativeError):NeuralNative(path,backend="unknown")
        with NeuralNative(path) as n:
            self.assertIsNone(n.lib.cc_nn_create_model_fast(n.model.array,len(n.model.array),17,42,n.model.id.encode(),0))
            result=n.neural_analyze(config.position(),max_depth=16,time_ms=0,node_budget=1)
            self.assertIsNone(result["report"]);self.assertEqual(result["stop_reason"],"node_budget")
            with n.make_control(max_depth=3,time_ms=0) as c:
                with self.assertRaises(NativeError):n.neural_analyze(config.position(),max_depth=4,time_ms=0,control=c)
            n.close()
            with self.assertRaises(NativeError):n.evaluate(config.position())

    def test_05_hard_kill_resume_and_identity_rejection(self):
        model=fixture(self.path/"model.json");root=config.position();run=self.path/"run"
        worker='''import json,sys
from congklak.neural_fast import NeuralNative,Analysis
with NeuralNative(sys.argv[1]) as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2:sys.stdin.readline()
 j.run(5,0,on_event=event)
 j.close()
'''
        child=subprocess.Popen([sys.executable,"-c",worker,str(model),str(run),json.dumps(root)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        messages=queue.Queue()
        def reader():
            for line in child.stdout:messages.put(json.loads(line))
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        try:
            while messages.get(timeout=60)["choice_depth"]!=2:pass
            child.kill();child.wait(timeout=20)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=20)
            thread.join(timeout=5)
            for stream in (child.stdin,child.stdout,child.stderr):stream.close()
        with NeuralNative(model) as n:
            job=Analysis(run,root,n)
            try:
                self.assertEqual(job.last["choice_depth"],2)
                resumed=job.run(5,0)["report"]
            finally:job.close()
            fresh=n.neural_analyze(root,max_depth=5,time_ms=0)["report"]
            validate_report(resumed,root,n)
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
            incompatible=NeuralNative(model,backend="scalar" if n.avx2_available else "auto")
            try:
                if incompatible.backend!=n.backend:
                    with self.assertRaises(pilot.PilotError):Analysis(run,root,incompatible)
            finally:incompatible.close()
            bad={**resumed,"input_controller":1-root["controller"]}
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:
                db.execute("UPDATE iterations SET body=?,hash=? WHERE depth=5",(config.canonical(bad),config.digest(bad)));db.commit()
            with self.assertRaises(NativeError):Analysis(run,root,n)

if __name__=="__main__":unittest.main()
