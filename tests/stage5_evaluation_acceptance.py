"""Feature invariants and actual CUDA checkpoint interruption recovery."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,features_v3 as features,training_v3 as tr
from congklak.native import ROOT

def latest(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:
        name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(path/name,map_location="cpu",weights_only=True)

def exact(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def position(pits=None,stores=None,actor=0):
    return {"pits":pits or [[1,0,3,1,4,0,2],[2,1,0,4,1,2,0]],"stores":stores or [38,39],"controller":actor}

class EvaluationAcceptance(unittest.TestCase):
    def test_01_feature_contract_controller_and_store_invariants(self):
        p=position(); x,d,r=features.encode(p)
        counts=p["pits"][0]+p["pits"][1]
        self.assertEqual(len(x),43); self.assertEqual(x[:15],features.encode(p,15)[0])
        self.assertEqual(x[15:29],[float(v>0) for v in counts]); self.assertEqual(x[29:43],[float(v==1) for v in counts])
        swap={"pits":p["pits"][::-1],"stores":p["stores"][::-1],"controller":1}
        sx,sd,sr=features.encode(swap); self.assertEqual(sx,x); self.assertEqual(sd,-d); self.assertEqual(sr,-r)
        shifted={**p,"stores":[40,37]}; tx,td,ts=features.encode(shifted)
        self.assertEqual(tx,x); self.assertEqual(ts,r); self.assertEqual(td,d+4)
        forced=position([[0]*7,[1,0,0,0,0,0,0]],[49,48],0)
        self.assertEqual(features.encode(forced),features.encode({**forced,"controller":1}))
        terminal=position([[0]*7,[0]*7],[55,43]); self.assertEqual(features.encode(terminal)[1:],(12,0))

    def test_02_exact_store_translation_and_terminal_network_output(self):
        torch.manual_seed(19); model=tr.Evaluator(32,43)
        p=position(); x,d,r=features.encode(p)
        with torch.no_grad():
            base=model(torch.tensor([x]),torch.tensor([d],dtype=torch.float32),torch.tensor([r],dtype=torch.float32)).item()
            moved=model(torch.tensor([x]),torch.tensor([d+4],dtype=torch.float32),torch.tensor([r],dtype=torch.float32)).item()
            end=model(torch.zeros((1,43)),torch.tensor([12.]),torch.tensor([0.])).item()
        self.assertAlmostEqual(moved-base,4,places=5); self.assertEqual(end,12)

    def test_03_training_loader_has_only_isolated_train_development(self):
        manifest,rows,_=tr.load_data()
        self.assertEqual(manifest["identity"]["schema"],"stronger-corpus-v2")
        self.assertEqual({key:len(value) for key,value in rows.items()},{"train":8000,"development":1000})
        self.assertTrue(all(row["split"]==split for split,value in rows.items() for row in value))
        self.assertTrue(all(row["label"]["choice_depth"]==6 and row["label"]["kind"]=="finite_choice_horizon" for value in rows.values() for row in value))

    def test_04_real_cuda_kill_before_commit_bitwise_restore(self):
        self.assertTrue(torch.cuda.is_available())
        base=ROOT/"build/stage5-evaluation-acceptance"; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root=Path(temp); fresh=root/"fresh"; resumed=root/"resumed"; ready=root/"uncommitted-ready"
            args={"steps":40,"seed":19,"width":32,"input_count":43,"batch_size":32,"checkpoint_every":10,"device":"cuda"}
            reference=tr.train(fresh,**args)
            child="""from pathlib import Path
import sys,time
from congklak import training_v3 as tr
path=Path(sys.argv[1]); ready=Path(sys.argv[2])
def event(e):
    if e['event']=='checkpoint_file_ready' and e['step']==20:
        ready.write_text('verified checkpoint file exists before catalog commit')
        while True: time.sleep(.05)
tr.train(path,steps=40,seed=19,width=32,input_count=43,batch_size=32,checkpoint_every=10,device='cuda',on_event=event)
"""
            with (root/"child.log").open("w",encoding="utf-8") as output:
                process=subprocess.Popen([sys.executable,"-c",child,str(resumed),str(ready)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+60
                    while not ready.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
                    self.assertTrue(ready.exists(),(root/"child.log").read_text(encoding="utf-8"))
                    process.kill(); process.wait(timeout=10)
                finally:
                    if process.poll() is None: process.kill(); process.wait(timeout=10)
            with closing(sqlite3.connect(resumed/"catalog.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0],10)
                self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0],"wal")
            restored=tr.train(resumed,**args)
            self.assertEqual(restored["recovery"]["restored_step"],10)
            self.assertEqual(restored["weights_sha256"],reference["weights_sha256"])
            self.assertTrue(exact(latest(fresh),latest(resumed)),"Model, optimizer, sampler and every RNG state must match")
            with self.assertRaises(tr.TrainingError): tr.train(resumed,**{**args,"input_count":15})

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationAcceptance)
    start=time.monotonic(); result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"passed":result.wasSuccessful(),
              "elapsed_seconds":time.monotonic()-start,"source_sha256":tr.sha(__file__),"trainer_sha256":tr.sha(tr.__file__),"features_sha256":tr.sha(features.__file__),
              "real_cuda_hard_kill_before_catalog_commit":result.wasSuccessful(),"sealed_final_test_consumed":False}
    tr.atomic_json(ROOT/"build/stage5-evaluation-acceptance.json",evidence)
    raise SystemExit(0 if result.wasSuccessful() else 1)
