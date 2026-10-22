"""Root-only teacher lineage, frozen dummy heads and CUDA interruption recovery."""
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
from congklak import config,deeper_teacher_v2 as dt,training_v7 as tr
from congklak.native import ROOT,Native

def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def latest(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db: name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(path/name,map_location="cpu",weights_only=True)

class DeeperTrainingAcceptance(unittest.TestCase):
    def test_01_root_ownership_horizon_and_legal_Q_validation(self):
        _,groups,fingerprints=dt.roots(); self.assertEqual({s:len(r) for s,r in groups.items()},{"train":8000,"development":1000})
        self.assertEqual(set(fingerprints),{"train","development"}); native=Native(); row=groups["train"][0]
        for field,value in (("choice_depth",8),("oracle_manifest_sha256","oracle"),("target_player0",True)):
            damaged=copy.deepcopy(row["label"]); damaged[field]=value
            with self.subTest(field=field),self.assertRaises(dt.DeeperError): dt.validate_label(damaged,row["state"],row["legal_mask"],6,native)
        damaged=copy.deepcopy(row["label"]); damaged["action_scores"].pop()
        with self.assertRaises(dt.DeeperError): dt.validate_label(damaged,row["state"],row["legal_mask"],6,native)
    def test_02_architecture_and_dummy_head_no_gradients_or_decay(self):
        for inputs,total,active in ((43,7496,7041),(127,12872,12417)):
            model=tr.Evaluator(64,inputs); self.assertEqual(sum(v.numel() for v in model.parameters()),total)
            self.assertEqual(sum(v.numel() for v in model.parameters() if v.requires_grad),active)
            original={k:v.clone() for k,v in model.policy.state_dict().items()}; optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001,weight_decay=.0001)
            x=torch.rand(4,inputs); d=torch.tensor([10.,-20.,2.,0.]); r=torch.tensor([20.,-40.,8.,-98.]); y=torch.tensor([20.,-30.,3.,10.])
            optimizer.zero_grad(); prediction=model(x,d,r); torch.nn.functional.huber_loss(prediction,y,delta=1.).backward(); optimizer.step()
            self.assertTrue(all(v.grad is None for v in model.policy.parameters())); self.assertTrue(equal(original,model.policy.state_dict())); self.assertEqual(len(optimizer.state),6)
    def test_03_published_labels_features_and_both_horizons(self):
        manifest,h6,_=dt.load(teacher_depth=6); other,h8,_=dt.load(teacher_depth=8); self.assertEqual(manifest,other)
        for split in ("train","development"):
            self.assertEqual([r["state_key"] for r in h6[split]],[r["state_key"] for r in h8[split]])
            self.assertEqual([r["family"] for r in h6[split]],[r["family"] for r in h8[split]])
            self.assertTrue(all(r["label"]["choice_depth"]==6 for r in h6[split])); self.assertTrue(all(r["label"]["choice_depth"]==8 for r in h8[split]))
            self.assertTrue(all(r["cached_features"]==q["cached_features"] for r,q in zip(h6[split],h8[split])))
    def test_04_real_cuda_precommit_kill_all_state_recovery(self):
        self.assertTrue(torch.cuda.is_available()); base=ROOT/"build/stage5-deeper-training-acceptance"; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary); ready=root/"ready"; fresh=root/"fresh"; resumed=root/"resumed"
            args={"steps":40,"seed":47,"width":17,"batch_size":32,"checkpoint_every":10,"device":"cuda","input_count":127,"teacher_depth":8}
            reference=tr.train(fresh,**args)
            child="""from pathlib import Path
import sys,time
from congklak import training_v7 as tr
def event(value):
 if value['event']=='checkpoint_file_ready' and value['step']==20:
  Path(sys.argv[2]).write_text('checkpoint verified before catalog commit')
  while True: time.sleep(.05)
tr.train(Path(sys.argv[1]),steps=40,seed=47,width=17,batch_size=32,checkpoint_every=10,device='cuda',input_count=127,teacher_depth=8,on_event=event)
"""
            with (root/"child.log").open("w",encoding="utf-8") as output:
                process=subprocess.Popen([sys.executable,"-c",child,str(resumed),str(ready)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+120
                    while not ready.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
                    self.assertTrue(ready.exists(),(root/"child.log").read_text(encoding="utf-8")); process.kill(); process.wait(timeout=10)
                finally:
                    if process.poll() is None: process.kill(); process.wait(timeout=10)
            with closing(sqlite3.connect(resumed/"catalog.sqlite3")) as db: self.assertEqual(db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0],10)
            restored=tr.train(resumed,**args); self.assertEqual(restored["recovery"]["restored_step"],10); self.assertEqual(restored["weights_sha256"],reference["weights_sha256"])
            self.assertTrue(equal(latest(fresh),latest(resumed)),"Model/optimizer/best/sampler/Python/Torch CPU/CUDA state differs after actual kill")
            files=(resumed/"result.json",resumed/"model.json",resumed/"catalog.sqlite3")
            snapshots={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
            repeated=tr.train(resumed,**args); self.assertEqual(repeated,restored)
            self.assertTrue(all(p.read_bytes()==data and p.stat().st_mtime_ns==mtime for p,(data,mtime) in snapshots.items()),"Finished resume rewrote immutable results, export or catalog")
            with self.assertRaises(tr.TrainingError): tr.train(resumed,**{**args,"teacher_depth":6})

    def test_05_final_catalog_commit_pre_export_kill(self):
        self.assertTrue(torch.cuda.is_available()); base=ROOT/"build/stage5-deeper-training-acceptance"; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root=Path(temporary); ready=root/"ready"; fresh=root/"fresh"; resumed=root/"resumed"
            args={"steps":20,"seed":81,"width":17,"batch_size":32,"checkpoint_every":10,"device":"cuda","input_count":43,"teacher_depth":6}
            reference=tr.train(fresh,**args)
            child="""from pathlib import Path
import sys,time
from congklak import training_v7 as tr
original=tr.export_model
def before_export(*args,**kwargs):
 Path(sys.argv[2]).write_text('final catalog committed before export')
 while True: time.sleep(.05)
tr.export_model=before_export
tr.train(Path(sys.argv[1]),steps=20,seed=81,width=17,batch_size=32,checkpoint_every=10,device='cuda',input_count=43,teacher_depth=6)
"""
            with (root/"child.log").open("w",encoding="utf-8") as output:
                process=subprocess.Popen([sys.executable,"-c",child,str(resumed),str(ready)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+180
                    while not ready.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
                    self.assertTrue(ready.exists(),(root/"child.log").read_text(encoding="utf-8")); process.kill(); process.wait(timeout=10)
                finally:
                    if process.poll() is None: process.kill(); process.wait(timeout=10)
            with closing(sqlite3.connect(resumed/"catalog.sqlite3")) as db: self.assertEqual(db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0],20)
            self.assertFalse((resumed/"result.json").exists()); self.assertFalse((resumed/"model.json").exists())
            restored=tr.train(resumed,**args); self.assertEqual(restored["recovery"]["restored_step"],20)
            self.assertEqual(restored["weights_sha256"],reference["weights_sha256"]); self.assertTrue(equal(latest(fresh),latest(resumed)))
            files=[p for p in resumed.iterdir() if p.is_file() and (p.suffix in (".json",".pt") or p.name=="catalog.sqlite3")]
            snapshots={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in files}
            self.assertEqual(tr.train(resumed,**args),restored)
            self.assertTrue(all(p.read_bytes()==data and p.stat().st_mtime_ns==mtime for p,(data,mtime) in snapshots.items()))
            original=(resumed/"model.json").read_bytes(); forged=json.loads(original); forged["model"]["terminal"]="forged terminal contract"; forged["sha256"]=config.digest(forged["model"])
            (resumed/"model.json").write_text(config.canonical(forged)+"\n"); damaged=(resumed/"model.json").read_bytes()
            with self.assertRaises(tr.TrainingError): tr.train(resumed,**args)
            self.assertEqual((resumed/"model.json").read_bytes(),damaged); (resumed/"model.json").write_bytes(original)
            original=(resumed/"result.json").read_bytes(); forged=json.loads(original); forged["selected_development"]["finite_choice_horizon"]["mae_seeds"]+=1
            (resumed/"result.json").write_text(config.canonical(forged)+"\n"); damaged=(resumed/"result.json").read_bytes()
            with self.assertRaises(tr.TrainingError): tr.train(resumed,**args)
            self.assertEqual((resumed/"result.json").read_bytes(),damaged)

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DeeperTrainingAcceptance); start=time.monotonic(); result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"passed":result.wasSuccessful(),"elapsed_seconds":time.monotonic()-start,
              "source_sha256":tr.sha(__file__),"trainer_sha256":tr.sha(tr.__file__),"deeper_teacher_sha256":tr.sha(dt.__file__),"real_cuda_precommit_hard_kill":result.wasSuccessful(),"real_cuda_final_commit_pre_export_hard_kill":result.wasSuccessful(),"completed_files_byte_and_mtime_stable":result.wasSuccessful(),"checksum_valid_forged_export_and_result_refused":result.wasSuccessful(),"sealed_final_test_consumed":False}
    tr.atomic_json(ROOT/"build/stage5-deeper-training-acceptance.json",evidence); raise SystemExit(0 if result.wasSuccessful() else 1)
