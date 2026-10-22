"""Policy targets, frozen native values, complete warm artifacts and CUDA recovery."""
from contextlib import closing
import copy
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
import hashlib
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ISOLATED_DRIVER=__name__=="__main__" and not os.environ.get("PHASE6_ACCEPTANCE_UNIT_WORKER")
if not ISOLATED_DRIVER:
    import torch
    torch.set_num_threads(1)
    from congklak import config,phase6_training as tr
    from congklak.neural_relay import Model,NeuralNative


def equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def latest(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db:name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(path/name,map_location="cpu",weights_only=True)


def snapshot(path):
    return {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in path.iterdir() if p.is_file() and not p.name.endswith(("-wal","-shm"))}



def run_cuda(path,args,expect_failure=False):
    # The coordinator remains CPU-only. Each CUDA context exits before the next
    # worker starts, including the fresh reference and resume verification.
    code="from pathlib import Path;import sys,json;from congklak import phase6_training as tr;tr.train(Path(sys.argv[1]),**json.loads(sys.argv[2]))"
    result=subprocess.run([sys.executable,"-c",code,str(path),json.dumps(args)],cwd=ROOT,capture_output=True,text=True)
    if (result.returncode==0)==expect_failure:
        failure=ROOT/"build/phase6-training-acceptance"/("worker-failure-"+uuid.uuid4().hex+".log")
        failure.write_text(result.stdout+result.stderr,encoding="utf-8")
        raise AssertionError(str(failure)+"\n"+result.stdout+result.stderr)
    return json.loads((path/"result.json").read_text(encoding="utf-8")) if not expect_failure else result.returncode


def kill_at_marker(child,path,marker):
    with (path/"child.log").open("w",encoding="utf-8") as output:
        process=subprocess.Popen([sys.executable,"-c",child,str(path/"run"),str(marker)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while not marker.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            if not marker.exists():
                failure=ROOT/"build/phase6-training-acceptance"/("kill-probe-failure-"+uuid.uuid4().hex+".log")
                failure.write_bytes((path/"child.log").read_bytes());raise AssertionError(str(failure)+"\n"+failure.read_text(encoding="utf-8"))
            process.kill();process.wait(timeout=10)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)


class Phase6TrainingAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.data,_=tr.teacher.load(teacher_depth=8)
        cls.source=Model(tr.SOURCE_MODEL)
        base=ROOT/"build/phase6-training-acceptance";base.mkdir(exist_ok=True)
        cls.temporary=tempfile.TemporaryDirectory(dir=base);cls.work=Path(cls.temporary.name)
        assert cls.work.resolve().is_relative_to(base.resolve())

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_01_root_H8_actor_softQ_and_strict_coverage(self):
        for row in self.data["development"][:40]:
            probability,legal,q,sign=tr.teacher_target(row)
            raw=[sign*a["value_player0"] for a in row["label"]["action_scores"]];weights=[math.exp((v-max(raw))/4.) for v in raw]
            expected=[0.]*7
            for action,weight in zip(row["label"]["action_scores"],weights):expected[action["pit"]]=weight/sum(weights)
            self.assertEqual(probability,expected);self.assertEqual([i for i,v in enumerate(legal) if v],[a["pit"] for a in row["label"]["action_scores"]])
            for field,value in (("choice_depth",6),("oracle_manifest_sha256","oracle"),("target_player0",True)):
                damaged=copy.deepcopy(row);damaged["label"][field]=value
                with self.assertRaises(tr.teacher.DeeperError):tr.teacher_target(damaged)
        row=copy.deepcopy(self.data["development"][0]);row["state"]={"pits":[[0]*7,[1]*7],"stores":[40,51],"controller":0};row["legal_mask"]=127
        scores=[-10,-15,-11,-14,-10,-12,-13];row["label"].update(target_player0=-15,best_moves=[1],action_scores=[{"pit":i,"value_player0":v} for i,v in enumerate(scores)])
        p,legal,q,sign=tr.teacher_target(row);self.assertEqual(sign,-1);self.assertEqual(max(range(7),key=p.__getitem__),1)
        for temperature in (0,True,float("nan"),float("inf")):
            with self.assertRaises(tr.TrainingError):tr.teacher_target(row,temperature)

    def test_02_policy_mass_mask_and_extreme_finite_logits(self):
        logits=torch.tensor([[-2e9,-3e9,1e20,0.,0.,0.,0.]])
        legal=torch.tensor([[True,True,False,False,False,False,False]])
        target=torch.tensor([[.75,.25,0.,0.,0.,0.,0.]])
        actual=tr.policy_losses(logits,target,legal)
        expected=-(target[:,:2]*torch.nn.functional.log_softmax(logits[:,:2],dim=1)).sum(dim=1)
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.isfinite(actual).all())
        with self.assertRaises(tr.TrainingError):tr.policy_losses(logits,target,torch.zeros_like(legal))
        for damaged in ([1.,0.,0.,0.,0.,0.,.01],[False,1.,0.,0.,0.,0.,0.],[float("nan"),1.,0.,0.,0.,0.,0.]):
            with self.assertRaises(tr.TrainingError):tr.policy_target(damaged,legal[0].tolist())

    def test_03_only455_policy_parameters_and_native_value_exact(self):
        model=tr.Evaluator("warm");model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in self.source.weights.items()})
        original={k:v.clone() for k,v in model.state_dict().items()};t=tr.teacher_tensors(self.data["train"][:20],"cpu")
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,foreach=False)
        value,logits=model.both(t["x"],t["d"],t["r"]);tr.policy_losses(logits,t["probability"],t["legal"]).mean().backward();optimizer.step()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),455)
        self.assertEqual(len(optimizer.state),2);self.assertTrue(equal(tr.value_state(original),tr.value_state(model.state_dict())))
        self.assertTrue(all(p.grad is None for p in model.layers.parameters()));self.assertFalse(equal({k:v for k,v in original.items() if k.startswith("policy.")},{k:v for k,v in model.state_dict().items() if k.startswith("policy.")}))
        path=self.work/"single-update-model.json";identity={"stage":"warm","width":64,"input_count":127,"rules_digest":config.RULES_DIGEST};tr.export_model(model,path,identity,1)
        with NeuralNative(self.source,mode="value-none",backend="auto") as old,NeuralNative(path,mode="value-none",backend="auto") as new:
            for row in self.data["development"][:20]:
                before=old.evaluate(row["state"]);after=new.evaluate(row["state"])
                for key in ("value_player0","value_units","model_value_player0","model_value_units"):self.assertEqual(before[key],after[key])
        self.assertEqual(sum(p.numel() for p in tr.Evaluator("selfplay").parameters() if p.requires_grad),12872)

    def test_04_selfplay_seed_targets_features_and_policy_validation(self):
        source=self.data["train"][0];row={"state":source["state"],"stratum":source["stratum"],"gameid":"fixture-game","labelkind":"selfplay_terminal_seed_margin",
              "value_target_player0":(source["state"]["stores"][0]-source["state"]["stores"][1])//2*2,"policy_target":tr.teacher_target(source)[0],"cached_features":source["cached_features"]}
        tensors=tr.selfplay_tensors([row],"cpu");self.assertEqual(tensors["y"].item(),row["value_target_player0"])
        for field,value in (("labelkind","bootstrap"),("value_target_player0",True),("value_target_player0",99),("gameid",0)):
            damaged=copy.deepcopy(row);damaged[field]=value
            with self.assertRaises(tr.TrainingError):tr.selfplay_tensors([damaged],"cpu")
        damaged=copy.deepcopy(row);damaged["cached_features"]["x"][0]+=.1
        with self.assertRaises(tr.TrainingError):tr.selfplay_tensors([damaged],"cpu")
        with self.assertRaises(tr.TrainingError):tr.selfplay_tensors([],"cpu")

    def test_05_actual_CUDA_precommit_kill_and_all_state_recovery(self):
        path=self.work/"precommit";path.mkdir();ready=path/"ready"
        args={"steps":30,"checkpoint_every":10,"batch_size":32,"seed":147,"device":"cuda"}
        fresh=run_cuda(path/"fresh",args)
        child="""from pathlib import Path
import sys,time
from congklak import phase6_training as tr
def event(value):
 if value['event']=='checkpoint_file_ready' and value['step']==20:
  Path(sys.argv[2]).write_text('before FULL catalog commit')
  while True:time.sleep(.05)
tr.train(Path(sys.argv[1]),steps=30,checkpoint_every=10,batch_size=32,seed=147,device='cuda',on_event=event)
"""
        kill_at_marker(child,path,ready)
        with closing(sqlite3.connect(path/"run/catalog.sqlite3")) as db:self.assertEqual(db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0],10)
        resumed=run_cuda(path/"run",args);self.assertEqual(resumed["recovery"]["restored_step"],10);self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
        self.assertTrue(equal(latest(path/"fresh"),latest(path/"run")),"Model/optimizer/Python/sampler/TorchCPU/CUDA/best state changed")
        state=latest(path/"run");self.assertEqual(len(state["optimizer"]["state"]),2);self.assertEqual(tr.weight_digest(tr.value_state(state["model"])),tr.SOURCE_VALUE_WEIGHTS)

    def test_06_actual_final_commit_pre_export_kill_and_completed_guard(self):
        path=self.work/"preexport";path.mkdir();ready=path/"ready"
        args={"steps":20,"checkpoint_every":10,"batch_size":32,"seed":281,"device":"cuda"}
        fresh=run_cuda(path/"fresh",args)
        child="""from pathlib import Path
import sys,time
from congklak import phase6_training as tr
def event(value):
 if value['event']=='training_final_checkpoint_ready':
  Path(sys.argv[2]).write_text('final FULL commit before export')
  while True:time.sleep(.05)
tr.train(Path(sys.argv[1]),steps=20,checkpoint_every=10,batch_size=32,seed=281,device='cuda',on_event=event)
"""
        kill_at_marker(child,path,ready);self.assertEqual(latest(path/"run")["step"],20);self.assertFalse((path/"run/model.json").exists())
        resumed=run_cuda(path/"run",args);self.assertEqual(resumed["recovery"]["restored_step"],20);self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
        self.assertTrue(equal(latest(path/"fresh"),latest(path/"run")));self.assertEqual((path/"fresh/model.json").read_bytes(),(path/"run/model.json").read_bytes())
        before=snapshot(path/"run");self.assertEqual(run_cuda(path/"run",args),resumed);self.assertEqual(snapshot(path/"run"),before)
        tr.warm_completion(path/"run/model.json",Model(path/"run/model.json"))
        original=(path/"run/model.json").read_bytes();forged=json.loads(original);forged["model"]["uses_oracle"]=True;forged["sha256"]=config.digest(forged["model"])
        (path/"run/model.json").write_text(config.canonical(forged)+"\n");damaged=snapshot(path/"run")
        self.assertNotEqual(run_cuda(path/"run",args,expect_failure=True),0)
        self.assertEqual(snapshot(path/"run"),damaged);(path/"run/model.json").write_bytes(original)
        original=(path/"run/result.json").read_bytes();forged=json.loads(original);forged["selected_development"]["policy"]["uniform_stratum_teacher_cross_entropy"]+=1
        (path/"run/result.json").write_text(config.canonical(forged)+"\n");damaged=snapshot(path/"run")
        self.assertNotEqual(run_cuda(path/"run",args,expect_failure=True),0)
        self.assertEqual(snapshot(path/"run"),damaged)
        forged["status"]="interrupted";(path/"run/result.json").write_text(config.canonical(forged)+"\n")
        with self.assertRaises(tr.TrainingError):tr.warm_completion(path/"run/model.json",Model(path/"run/model.json"))



def isolated_driver():
    """No Torch import/context in the coordinator, even for state comparison."""
    base=ROOT/"build/phase6-training-acceptance";base.mkdir(exist_ok=True);started=time.monotonic()
    env={**os.environ,"PHASE6_ACCEPTANCE_UNIT_WORKER":"1"}
    cpu=subprocess.run([sys.executable,__file__],cwd=ROOT,env=env,capture_output=True,text=True)
    print(cpu.stdout+cpu.stderr,flush=True)
    passed=cpu.returncode==0;errors=[];records=[]
    code="from pathlib import Path;import sys,json;from congklak import phase6_training as tr;tr.train(Path(sys.argv[1]),**json.loads(sys.argv[2]))"
    def invoke(path,args,refusal=False):
        result=subprocess.run([sys.executable,"-c",code,str(path),json.dumps(args)],cwd=ROOT,capture_output=True,text=True)
        if (result.returncode==0)==refusal:
            log=base/("isolated-worker-failure-"+uuid.uuid4().hex+".log");log.write_text(result.stdout+result.stderr,encoding="utf-8")
            raise AssertionError(str(log)+"\n"+result.stdout+result.stderr)
        return json.loads((path/"result.json").read_text(encoding="utf-8")) if not refusal else result.returncode
    if passed:
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            work=Path(temporary);assert work.resolve().is_relative_to(base.resolve())
            for kind,args in (("precommit",dict(steps=30,checkpoint_every=10,batch_size=32,seed=147,device="cuda")),
                              ("preexport",dict(steps=20,checkpoint_every=10,batch_size=32,seed=281,device="cuda"))):
                try:
                    root=work/kind;root.mkdir();fresh=root/"fresh";resumed=root/"run";ready=root/"ready"
                    reference=invoke(fresh,args)
                    blocked="checkpoint_file_ready" if kind=="precommit" else "training_final_checkpoint_ready"
                    child="""from pathlib import Path
import sys,time,json
from congklak import phase6_training as tr
def event(value):
 if value['event']==sys.argv[3] and (sys.argv[3]!='checkpoint_file_ready' or value['step']==20):
  Path(sys.argv[2]).write_text('verified barrier before commit/export')
  while True:time.sleep(.05)
tr.train(Path(sys.argv[1]),**json.loads(sys.argv[4]),on_event=event)
"""
                    with (root/"child.log").open("w",encoding="utf-8") as output:
                        process=subprocess.Popen([sys.executable,"-c",child,str(resumed),str(ready),blocked,json.dumps(args)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                        try:
                            deadline=time.monotonic()+180
                            while not ready.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                            if not ready.exists():
                                log=base/("isolated-kill-failure-"+uuid.uuid4().hex+".log");log.write_bytes((root/"child.log").read_bytes());raise AssertionError(str(log)+"\n"+log.read_text(encoding="utf-8"))
                            process.kill();process.wait(timeout=10)
                        finally:
                            if process.poll() is None:process.kill();process.wait(timeout=10)
                    with closing(sqlite3.connect(resumed/"catalog.sqlite3")) as db:committed=db.execute("SELECT MAX(step) FROM checkpoints").fetchone()[0]
                    assert committed==(10 if kind=="precommit" else 20)
                    if kind=="preexport":assert not (resumed/"model.json").exists() and not (resumed/"result.json").exists()
                    restored=invoke(resumed,args);assert restored["recovery"]["restored_step"]==committed and restored["weights_sha256"]==reference["weights_sha256"]
                    compare="""from pathlib import Path
import sys
sys.path.insert(0,'tests')
import phase6_training_acceptance as m
from congklak import phase6_training as tr
from congklak.neural_relay import Model
a=Path(sys.argv[1]);b=Path(sys.argv[2]);assert m.equal(m.latest(a),m.latest(b));assert (a/'model.json').read_bytes()==(b/'model.json').read_bytes()
state=m.latest(b);assert len(state['optimizer']['state'])==2;assert tr.weight_digest(tr.value_state(state['model']))==tr.SOURCE_VALUE_WEIGHTS
tr.warm_completion(b/'model.json',Model(b/'model.json'))
"""
                    checked=subprocess.run([sys.executable,"-c",compare,str(fresh),str(resumed)],cwd=ROOT,capture_output=True,text=True)
                    assert checked.returncode==0,checked.stdout+checked.stderr
                    before=snapshot(resumed);assert invoke(resumed,args)==restored and snapshot(resumed)==before
                    if kind=="preexport":
                        original=(resumed/"model.json").read_bytes();forged=json.loads(original);forged["model"]["uses_oracle"]=True
                        forged["sha256"]=hashlib.sha256(json.dumps(forged["model"],sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
                        (resumed/"model.json").write_text(json.dumps(forged)+"\n");damaged=snapshot(resumed);invoke(resumed,args,refusal=True);assert snapshot(resumed)==damaged;(resumed/"model.json").write_bytes(original)
                        original=(resumed/"result.json").read_bytes();forged=json.loads(original);forged["selected_development"]["policy"]["uniform_stratum_teacher_cross_entropy"]+=1
                        (resumed/"result.json").write_text(json.dumps(forged)+"\n");damaged=snapshot(resumed);invoke(resumed,args,refusal=True);assert snapshot(resumed)==damaged
                    records.append(dict(kind=kind,passed=True,actual_cuda_kill=True,restored_step=committed,steps=args["steps"],full_state_bitwise=True,completed_bytes_and_mtimes_unchanged=True))
                    print(kind+" CUDA interruption and completed-artifact checks passed",flush=True)
                except Exception as exc:
                    passed=False;errors.append(dict(kind=kind,error=str(exc)));print(kind+": "+str(exc),flush=True);break
    def sha(path):
        with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
    evidence={"schema":"phase6-training-acceptance-v1","tests":4+len(records)+(1 if errors else 0),"passed":passed,"failures":len(errors)+(1 if cpu.returncode else 0),"errors":0,"skipped":0,
              "elapsed_seconds":time.monotonic()-started,"source_sha256":sha(__file__),"trainer_sha256":sha(ROOT/"congklak/phase6_training.py"),"real_CUDA_precommit_kill":passed,
              "real_CUDA_final_commit_pre_export_kill":passed,"all_state_recovery_bitwise":passed,"completed_bytes_and_mtimes_unchanged":passed,
              "native_warm_value_exact":cpu.returncode==0,"selfplay_training_recovery_verified":False,"coordinator":"stdlib-only; isolated serial CUDA workers and later CPU-only comparisons",
              "records":records,"failure_details":errors,"sealed_final_test_consumed":False}
    destination=ROOT/"build/phase6-training-acceptance.json";temporary=destination.with_name(destination.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,destination);print(json.dumps({k:v for k,v in evidence.items() if k not in ("records","failure_details")}),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    if ISOLATED_DRIVER:raise SystemExit(isolated_driver())
    names=[name for name in unittest.defaultTestLoader.getTestCaseNames(Phase6TrainingAcceptance) if name[:7] in ("test_01","test_02","test_03","test_04")]
    suite=unittest.TestSuite(Phase6TrainingAcceptance(name) for name in names);result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
