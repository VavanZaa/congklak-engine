"""Root action target semantics, masked loss safety and full CUDA recovery."""
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
from congklak import action_targets as targets,training_v4 as tr
from congklak.native import ROOT

def row(actor=0):
    p={"pits":[[1,0,3,1,4,0,2],[2,1,0,4,1,2,0]],"stores":[38,39],"controller":actor}
    legal=[pit for pit,count in enumerate(p["pits"][actor]) if count]
    scores=[8,6,8,3,2] if actor==0 else [-8,-6,-8,-3,-2]
    best=(max if actor==0 else min)(scores)
    return {"state":p,"legal_mask":sum(1<<pit for pit in legal),"stratum":"5-24/ordinary",
            "label":{"kind":"finite_choice_horizon","choice_depth":6,"oracle_manifest_sha256":None,"target_player0":best,
                     "action_scores":[{"pit":pit,"value_player0":score} for pit,score in zip(legal,scores)],"best_moves":[pit for pit,score in zip(legal,scores) if score==best]}}

def latest(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db: name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(path/name,map_location="cpu",weights_only=True)

def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[key],b[key]) for key in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class ActionTrainingAcceptance(unittest.TestCase):
    def test_01_relative_targets_swap_offset_legality_and_horizon(self):
        root=row(); probability,mask,q,sign=targets.target(root)
        self.assertAlmostEqual(sum(probability),1); self.assertEqual(sign,1)
        self.assertEqual(probability[0],probability[3]); self.assertEqual(probability[1],0); self.assertFalse(mask[1])
        swapped=copy.deepcopy(root); swapped["state"]={"pits":root["state"]["pits"][::-1],"stores":root["state"]["stores"][::-1],"controller":1}
        swapped["label"]["target_player0"]*=-1
        for action in swapped["label"]["action_scores"]: action["value_player0"]*=-1
        self.assertEqual(targets.target(swapped)[:3],(probability,mask,q)); self.assertEqual(targets.target(swapped)[3],-1)
        shifted=copy.deepcopy(root); shifted["state"]["stores"]=[40,37]; shifted["label"]["target_player0"]+=4
        for action in shifted["label"]["action_scores"]: action["value_player0"]+=4
        self.assertEqual(targets.target(shifted)[:2],(probability,mask))
        for bad_temperature in (True,None,"4",0,float("nan")):
            with self.assertRaises(ValueError): targets.target(root,bad_temperature)
        for mutation in ("horizon","missing","duplicate","wrong_best","terminal"):
            bad=copy.deepcopy(root)
            if mutation=="horizon": bad["label"]["choice_depth"]=5
            elif mutation=="missing": bad["label"]["action_scores"].pop()
            elif mutation=="duplicate": bad["label"]["action_scores"].append(bad["label"]["action_scores"][0])
            elif mutation=="wrong_best": bad["label"]["target_player0"]+=1
            else: bad["state"]={"pits":[[0]*7,[0]*7],"stores":[55,43],"controller":0}
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): targets.target(bad)

    def test_02_forced_pass_actor_and_child_controller_do_not_change_root_sign(self):
        root=row(1)
        for action in root["label"]["action_scores"]: action.update({"next_controller":0,"bonus_choice":True,"forced_passes_after":[1]})
        target=targets.target(root); self.assertEqual(target[3],-1); self.assertEqual(max(target[2][pit] for pit,legal in enumerate(target[1]) if legal),8)
        forced=copy.deepcopy(root); forced["state"]["pits"][0]=[0]*7; forced["state"]["stores"]=[44,44]; forced["state"]["controller"]=0
        self.assertEqual(targets.target(forced),target)

    def test_03_masked_policy_gradients_and_train_only_bias_groups(self):
        logits=torch.tensor([[2.,1e8,2.,-1e8,3.,0.,2.]],requires_grad=True)
        probability=torch.tensor([[.5,0.,.5,0.,0.,0.,0.]])
        legal=torch.tensor([[True,False,True,False,False,False,False]])
        loss=tr.policy_loss(logits,probability,legal); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(logits.grad).all()); self.assertEqual(logits.grad[0,1].item(),0)
        with self.assertRaises(tr.TrainingError): tr.policy_loss(logits,probability,torch.zeros_like(legal))
        error=torch.tensor([2.,-4.,50.],requires_grad=True); signs=torch.tensor([1.,-1.,1.]); groups=torch.tensor([0,0,-1])
        self.assertEqual(tr.ordinary_bias_loss(error,signs,groups,3).item(),9)
        zero=tr.ordinary_bias_loss(error,signs,torch.full((3,),-1),3); self.assertEqual(zero.item(),0); zero.backward(); self.assertEqual(error.grad.tolist(),[0.,0.,0.])
        calibration=targets.ordinary_calibration([{"stratum":"5-24/ordinary"}]*20+[{"stratum":"75-98/ordinary"}]*5)
        self.assertEqual(calibration["eligible_strata"],["5-24/ordinary"]); self.assertFalse(calibration["development_calibration"])

    def test_04_architecture_export_and_isolated_dataset_targets(self):
        model=tr.Evaluator(64); self.assertEqual(sum(value.numel() for value in model.parameters()),7496)
        self.assertEqual(sum(value.numel() for value in tr.value_state(model.state_dict()).values()),7041)
        manifest,rows,_=tr.load_data(); self.assertEqual({key:len(value) for key,value in rows.items()},{"train":8000,"development":1000})
        for values in rows.values():
            for item in values: targets.target(item)
        identity={"variant":"B","input_count":43}
        base=ROOT/"build/stage5-action-acceptance"; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root=Path(temp); value=tr.export_model(model,root/"value.json",identity,100); full=tr.export_model(model,root/"policy.json",identity,100,True)
            p=row()["state"]; x,d,r=tr.encode(p,43)
            with torch.no_grad(): predicted,policy=model.both(torch.tensor([x]),torch.tensor([d],dtype=torch.float32),torch.tensor([r],dtype=torch.float32))
            raw,logits=tr.predict_export(full,p); self.assertAlmostEqual(raw,predicted.item(),places=5)
            self.assertEqual(tr.predict_export(value,p)[0],raw)
            for expected,actual in zip(policy[0].tolist(),logits): self.assertAlmostEqual(expected,actual,places=5)
            self.assertEqual(value["model"]["schema"],"congklak-mlp-export-v2"); self.assertEqual(full["model"]["schema"],"congklak-mlp-export-v3")

    def test_05_real_cuda_kill_full_policy_optimizer_and_rng_recovery(self):
        self.assertTrue(torch.cuda.is_available()); base=ROOT/"build/stage5-action-acceptance"; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root=Path(temp); ready=root/"verified-file-before-commit"; fresh=root/"fresh"; resumed=root/"resumed"
            args={"steps":40,"seed":23,"width":32,"batch_size":32,"checkpoint_every":10,"device":"cuda","variant":"C"}
            reference=tr.train(fresh,**args)
            child="""from pathlib import Path
import sys,time
from congklak import training_v4 as tr
path=Path(sys.argv[1]);ready=Path(sys.argv[2])
def event(e):
    if e['event']=='checkpoint_file_ready' and e['step']==20:
        ready.write_text('verified immutable full-head checkpoint before catalog commit')
        while True:time.sleep(.05)
tr.train(path,steps=40,seed=23,width=32,batch_size=32,checkpoint_every=10,device='cuda',variant='C',on_event=event)
"""
            with (root/"child.log").open("w",encoding="utf-8") as output:
                process=subprocess.Popen([sys.executable,"-c",child,str(resumed),str(ready)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+60
                    while not ready.exists() and process.poll() is None and time.monotonic()<deadline: time.sleep(.05)
                    self.assertTrue(ready.exists(),(root/"child.log").read_text(encoding="utf-8")); process.kill(); process.wait(timeout=10)
                finally:
                    if process.poll() is None: process.kill(); process.wait(timeout=10)
            with closing(sqlite3.connect(resumed/"catalog.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0],10)
            restored=tr.train(resumed,**args)
            self.assertEqual(restored["recovery"]["restored_step"],10); self.assertEqual(restored["weights_sha256"],reference["weights_sha256"])
            self.assertEqual(restored["value_weights_sha256"],reference["value_weights_sha256"])
            self.assertTrue(equal(latest(fresh),latest(resumed)),"All heads/optimizer/sampler/Python/Torch CPU/CUDA RNG and best-model state must match")
            with self.assertRaises(tr.TrainingError): tr.train(resumed,**{**args,"variant":"B"})

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ActionTrainingAcceptance); start=time.monotonic(); result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"passed":result.wasSuccessful(),
              "elapsed_seconds":time.monotonic()-start,"source_sha256":tr.sha(__file__),"trainer_sha256":tr.sha(tr.__file__),"action_targets_sha256":tr.sha(targets.__file__),
              "real_cuda_hard_kill_before_catalog_commit":result.wasSuccessful(),"sealed_final_test_consumed":False}
    tr.atomic_json(ROOT/"build/stage5-action-training-acceptance.json",evidence); raise SystemExit(0 if result.wasSuccessful() else 1)
