"""Training-environment acceptance; not imported by the stdlib engine suite.

Run with the Python executable recorded in build/training-runtime.json:
python -m unittest discover -s tests -p training_acceptance.py -v.
CUDA is required rather than silently skipped.
"""
from contextlib import closing
import json
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from congklak import training as tr,config
from congklak.native import ROOT

WORK=(ROOT/"build"/"training-acceptance").resolve()
WORK.mkdir(parents=True,exist_ok=True)

def checkpoint(path):
    with closing(sqlite3.connect(Path(path)/"catalog.sqlite3")) as db:
        name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(Path(path)/name,map_location="cpu",weights_only=True)

def equal_tree(test,a,b):
    if torch.is_tensor(a): test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a: equal_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): equal_tree(test,x,y)
    else: test.assertEqual(a,b)

class TrainingAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available(): raise RuntimeError("Required CUDA training verification unavailable")
        cls.manifest,cls.data,cls.fingerprints=tr.load_data()

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix="trainer-",dir=WORK)
        self.path=Path(self.directory.name).resolve()
        self.assertTrue(self.path.is_relative_to(WORK))

    def tearDown(self): self.directory.cleanup()

    def run_model(self,name,device="cpu",event=None,cancel=None):
        return tr.train(self.path/name,steps=40,seed=7,width=16,batch_size=32,checkpoint_every=10,
                        device=device,on_event=event,cancel=cancel)

    def test_01_loader_only_train_and_development_unit_P_equals_q(self):
        self.assertEqual(set(self.data),{"train","development"})
        self.assertEqual(len(self.data["train"]),6851)
        self.assertEqual(len(self.data["development"]),1416)
        for row in self.data["train"]:
            self.assertEqual(row["split"],"train")
            self.assertGreater(sum(sum(r) for r in row["state"]["pits"]),4)
            self.assertIsNone(row["label"]["oracle_manifest_sha256"])
            self.assertEqual(row["label_weight"],1)

    def test_02_actor_symmetry_store_offsets_terminals_and_independent_export(self):
        tr.deterministic("cpu"); torch.manual_seed(9)
        model=tr.Evaluator(16)
        export=tr.export_model(model,self.path/"export.json",{"width":16},0)
        for row in self.data["development"][::17]:
            p=row["state"]; f,d,s=tr.encode(p)
            with torch.no_grad(): value=model(torch.tensor([f]),torch.tensor([d],dtype=torch.float32),torch.tensor([s],dtype=torch.float32)).item()
            prediction=tr.predict_export(export,p)
            self.assertAlmostEqual(prediction,value,delta=1e-4)
            swapped={"pits":p["pits"][::-1],"stores":p["stores"][::-1],"controller":1-p["controller"]}
            self.assertAlmostEqual(tr.predict_export(export,swapped),-prediction,delta=1e-10)
            active=sum(sum(r) for r in p["pits"])
            self.assertGreaterEqual(prediction,d-active); self.assertLessEqual(prediction,d+active)
            if p["stores"][1] and p["stores"][0]<98:
                other={**p,"stores":[p["stores"][0]+1,p["stores"][1]-1]}
                self.assertAlmostEqual(tr.predict_export(export,other)-prediction,2,delta=1e-10)
        terminal={"pits":[[0]*7,[0]*7],"stores":[60,38],"controller":1}
        self.assertEqual(tr.predict_export(export,terminal),22)
        forced={"pits":[[0]*7,[0]*6+[1]],"stores":[49,48],"controller":0}
        actual={**forced,"controller":1}
        self.assertEqual(tr.predict_export(export,forced),tr.predict_export(export,actual))

    def test_03_cpu_and_cuda_resume_match_uninterrupted_optimizer_rng_and_weights(self):
        for device in ("cpu","cuda"):
            fresh=self.run_model(device+"-fresh",device)
            stopped=[False]
            def event(e):
                if e["event"]=="checkpoint_committed" and e["step"]==10: stopped[0]=True
            interrupted=self.run_model(device+"-resumed",device,event,lambda:stopped[0])
            self.assertEqual(interrupted["status"],"interrupted")
            continued=self.run_model(device+"-resumed",device)
            self.assertEqual(continued["weights_sha256"],fresh["weights_sha256"])
            a=checkpoint(self.path/(device+"-fresh")); b=checkpoint(self.path/(device+"-resumed"))
            for key in ("model","optimizer","sampler_rng","python_rng","torch_rng","cuda_rng","step","draws","best_model","best_score","best_step"):
                equal_tree(self,a[key],b[key])
            self.assertFalse(continued["sealed_final_test_consumed"])

    def test_04_real_process_kill_before_and_after_catalog_commits(self):
        worker='''import json,sys
from congklak import training as tr
def event(e):
    print(json.dumps(e),flush=True)
    if e.get("event")==sys.argv[2] and e["step"]==10:
        sys.stdin.readline()
tr.train(sys.argv[1],steps=40,seed=7,width=16,batch_size=32,checkpoint_every=10,device=sys.argv[3],on_event=event)
'''
        for device in ("cpu","cuda"):
            fresh=self.run_model(device+"-kill-fresh",device)
            for boundary in ("checkpoint_file_ready","checkpoint_committed"):
                name=device+"-"+boundary; path=self.path/name
                command=[sys.executable,"-c",worker,str(path),boundary,device]
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8")
                messages=queue.Queue()
                def reader():
                    for line in child.stdout: messages.put(json.loads(line))
                thread=threading.Thread(target=reader,daemon=True); thread.start()
                try:
                    while True:
                        e=messages.get(timeout=60)
                        if e.get("event")==boundary and e["step"]==10: break
                        if e.get("event")=="training_error": self.fail(str(e))
                    child.kill(); child.wait(timeout=20)
                finally:
                    if child.poll() is None: child.kill(); child.wait(timeout=20)
                    thread.join(timeout=10); child.stdin.close(); child.stdout.close(); child.stderr.close()
                with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:
                    committed=db.execute("SELECT step FROM checkpoints ORDER BY id").fetchall()
                self.assertEqual(committed,[] if boundary=="checkpoint_file_ready" else [(10,)])
                self.assertTrue(list(path.glob("checkpoint-*.pt")))
                resumed=self.run_model(name,device)
                self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
                equal_tree(self,checkpoint(path),checkpoint(self.path/(device+"-kill-fresh")))

    def test_05_corrupt_latest_falls_back_incompatible_and_single_writer_refused(self):
        self.run_model("damaged")
        path=self.path/"damaged"
        with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:
            name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
            identity=json.loads(db.execute("SELECT body FROM config").fetchone()[0])
        (path/name).write_bytes(b"truncated checkpoint")
        store=tr.Checkpoints(path,identity)
        try:
            saved,recovery=store.restore(); self.assertEqual(saved["step"],30)
            self.assertEqual(len(recovery["invalid_generations"]),1)
            with self.assertRaises((OSError,tr.TrainingError)): tr.Checkpoints(path,identity)
        finally: store.close()
        with self.assertRaises(tr.TrainingError): tr.Checkpoints(path,{**identity,"seed":100})
        result=self.run_model("damaged")
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["recovery"]["restored_step"],30)

    def test_06_sqlite_full_preserves_last_checkpoint_and_resumes(self):
        stop=[False]
        def event(e):
            if e["event"]=="checkpoint_committed": stop[0]=True
        self.run_model("full",event=event,cancel=lambda:stop[0])
        path=self.path/"full"
        with closing(sqlite3.connect(path/"catalog.sqlite3")) as db: identity=json.loads(db.execute("SELECT body FROM config").fetchone()[0])
        store=tr.Checkpoints(path,identity)
        try:
            state,_=store.restore(); old=state["step"]
            store.db.execute("CREATE TABLE filler(v BLOB)")
            store.db.commit()
            pages=store.db.execute("PRAGMA page_count").fetchone()[0]
            store.db.execute("PRAGMA max_page_count="+str(pages))
            store.db.execute("CREATE TRIGGER full_checkpoint BEFORE INSERT ON checkpoints BEGIN INSERT INTO filler VALUES(zeroblob(1048576)); END")
            with self.assertRaises(sqlite3.Error) as failed: store.save(state,{})
            self.assertEqual(failed.exception.sqlite_errorcode,sqlite3.SQLITE_FULL)
            recovered,_=store.restore(); self.assertEqual(recovered["step"],old)
            store.db.execute("DROP TRIGGER full_checkpoint")
            store.db.execute("DROP TABLE filler")
            store.db.commit()
        finally: store.close()
        resumed=self.run_model("full")
        fresh=self.run_model("full-fresh")
        self.assertEqual(resumed["status"],"complete")
        self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
        equal_tree(self,checkpoint(path),checkpoint(self.path/"full-fresh"))

    def test_07_off_cadence_cancel_preserves_model_selection_and_exact_resume(self):
        fresh=self.run_model("off-cadence-fresh","cuda")
        polls=[0]
        def cancel():
            polls[0]+=1
            return polls[0]>=14
        stopped=self.run_model("off-cadence-resumed","cuda",cancel=cancel)
        self.assertEqual(stopped["steps_completed"],7)
        self.assertFalse(stopped["model_available"])
        saved=checkpoint(self.path/"off-cadence-resumed")
        self.assertIsNone(saved["best_model"])
        resumed=self.run_model("off-cadence-resumed","cuda")
        self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
        equal_tree(self,checkpoint(self.path/"off-cadence-resumed"),checkpoint(self.path/"off-cadence-fresh"))

if __name__=="__main__": unittest.main()
