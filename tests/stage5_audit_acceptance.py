"""Independent adversarial checks for the Stage 5 challenger.

The value and rules references are independent of production encoders/search.
Real child processes are killed only after observable durable commit barriers.
Run with the workspace stdlib Python; GPU checks use the recorded CUDA runtime.
"""
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import reference as literal
from congklak import campaign,config,pilot
from congklak.native import NativeError
from congklak.neural import NeuralNative as FrozenNative
from congklak.neural_fast import Model,NeuralNative,Analysis,validate_report

WORK=ROOT/"build/stage5-audit"
WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={}
SCALE=256


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def fixture(path,inputs=43,width=17,seed=19):
    rng=random.Random(seed); weights={}
    for prefix,rows,cols in (("layers.0",width,inputs),("layers.2",width,width),("layers.4",1,width)):
        weights[prefix+".weight"]=[[rng.randrange(-8,9)/64 for _ in range(cols)] for _ in range(rows)]
        weights[prefix+".bias"]=[rng.randrange(-8,9)/64 for _ in range(rows)]
    digest=hashlib.sha256()
    for name,values in sorted(weights.items()):
        numbers=[number for row in values for number in row] if isinstance(values[0],list) else values
        digest.update(name.encode()); digest.update(struct.pack("<"+"f"*len(numbers),*numbers))
    m={"schema":"congklak-mlp-export-v2" if inputs==43 else "congklak-mlp-export-v1",
       "encoder":"current-controller-pit14-active15-occupied14-singleton14/store-offset-exact-v3" if inputs==43 else "current-controller-pit14-and-active15/store-offset-exact-v1",
       "identity":{"width":width,"input_count":inputs,"rules_digest":config.RULES_DIGEST},
       "weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":m,"sha256":config.digest(m)}),encoding="utf-8")
    return m


def literal_position(p):
    return literal.Position(7,tuple(p["pits"][0]+[p["stores"][0]]+p["pits"][1]+[p["stores"][1]]),p["controller"])


def as_dict(p):
    return {"pits":[list(p.counts[:7]),list(p.counts[8:15])],"stores":[p.counts[7],p.counts[15]],"controller":p.controller}


def reference_value(p,export):
    actor=literal.normalize(p).controller
    counts=[p.counts[row*8+i] for row in (actor,1-actor) for i in range(7)]
    active=sum(counts); offset=p.counts[7]-p.counts[15]
    if not active: return float(offset),offset*SCALE
    x=[number/98 for number in counts]+[active/98]
    if export["identity"]["input_count"]==43:
        x.extend(float(number>0) for number in counts)
        x.extend(float(number==1) for number in counts)
    for prefix in ("layers.0","layers.2","layers.4"):
        output=[]
        for row,bias in zip(export["weights"][prefix+".weight"],export["weights"][prefix+".bias"]):
            value=0.0
            for coefficient,feature in zip(row,x): value+=coefficient*feature
            value+=bias
            output.append(max(0.0,value) if prefix!="layers.4" else value)
        x=output
    residual=active*(1 if actor==0 else -1)*math.tanh(x[0])
    scaled=residual*SCALE
    rounded=math.floor(scaled+.5) if scaled>=0 else math.ceil(scaled-.5)
    return offset+residual,offset*SCALE+rounded


def random_positions(seed,count,active_min=0,active_max=98):
    rng=random.Random(seed)
    for _ in range(count):
        active=rng.randint(active_min,active_max); counts=[0]*16
        for _ in range(active): counts[rng.choice(tuple(range(7))+tuple(range(8,15)))]+=1
        counts[7]=rng.randrange(99-active); counts[15]=98-active-counts[7]
        yield literal.Position(7,tuple(counts),rng.randrange(2))


def killed_at_marker(arguments,marker,env=None,timeout=90):
    child=subprocess.Popen(arguments,cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    messages=queue.Queue(); lines=[]
    def read():
        for line in child.stdout:
            lines.append(line)
            try: messages.put(json.loads(line))
            except json.JSONDecodeError: pass
    reader=threading.Thread(target=read,daemon=True); reader.start(); seen=None
    started=time.monotonic()
    try:
        while time.monotonic()-started<timeout:
            try: e=messages.get(timeout=1)
            except queue.Empty:
                if child.poll() is not None: raise AssertionError("Child exited before durable barrier: "+"".join(lines)+child.stderr.read())
                continue
            if marker(e): seen=e; break
        if seen is None: raise AssertionError("No durable barrier before timeout")
        child.kill(); child.wait(timeout=20)
        return seen
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=20)
        reader.join(timeout=5)
        child.stdin.close(); child.stdout.close(); child.stderr.close()


class Stage5IndependentAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="audit-",dir=WORK)
        self.path=Path(self.temp.name); self.model_path=self.path/"fixture.json"
        self.export=fixture(self.model_path)

    def tearDown(self): self.temp.cleanup()

    def test_01_frozen_sources_and_dlls_unchanged(self):
        prior=json.loads((ROOT/"build/phase3-acceptance.json").read_text())
        frozen=("native/neural.cpp","congklak/neural.py","congklak/campaign.py","congklak/training_v2.py","congklak/training.py","native/engine.hpp","native/rules.cpp","native/search.cpp","congklak/native.py","build/congklak_native.dll","build/congklak_neural.dll")
        checked={}
        for name in frozen:
            expected=prior["inputs"].get(name)
            if expected is None:
                # Original phase-three report keeps earlier binding records separately.
                for section in ("prior_acceptance_inputs","protected_inputs","frozen_inputs"):
                    expected=prior.get(section,{}).get(name)
                    if expected: break
            if expected is None:
                original=json.loads((ROOT/"build/acceptance/acceptance.json").read_text())
                expected=original.get("inputs",{}).get(name)
            if expected is None:
                original=json.loads((ROOT/"build/training-acceptance.json").read_text())
                expected=original.get("inputs",{}).get(name)
            self.assertIsNotNone(expected,"Frozen path missing from acceptance identities: "+name)
            self.assertEqual(sha(ROOT/name),expected,name)
            checked[name]=expected
        EVIDENCE["frozen_inputs"]=checked

    def test_02_literal_encoder_numeric_parity_tail_and_symmetries(self):
        maximum=0.0; checked=0
        roots=list(random_positions(1337,220))
        roots += [literal.Position(7,tuple([0]*7+[49]+[0]*6+[1]+[48]),0),literal.Position(7,tuple([0]*7+[60]+[0]*7+[38]),1)]
        for inputs,width in ((15,16),(43,17),(43,32),(43,64)):
            path=self.path/f"model-{inputs}-{width}.json"; model=fixture(path,inputs,width)
            with NeuralNative(path,backend="scalar") as scalar,NeuralNative(path,backend="avx2") as packed:
                self.assertTrue(packed.avx2_available)
                for p in roots:
                    root=as_dict(p); expected,units=reference_value(p,model)
                    a=scalar.evaluate(root); b=packed.evaluate(root)
                    self.assertEqual(struct.pack("<d",a["value_player0"]),struct.pack("<d",b["value_player0"]))
                    self.assertEqual(a["value_units"],b["value_units"])
                    self.assertEqual(a["value_units"],units)
                    self.assertAlmostEqual(a["value_player0"],expected,delta=1e-10)
                    maximum=max(maximum,abs(a["value_player0"]-expected)); checked+=1
                    swapped={"pits":root["pits"][::-1],"stores":root["stores"][::-1],"controller":1-root["controller"]}
                    self.assertEqual(packed.evaluate(swapped)["value_units"],-units)
                    if root["stores"][1]:
                        shifted={**root,"stores":[root["stores"][0]+1,root["stores"][1]-1]}
                        self.assertEqual(packed.evaluate(shifted)["value_units"],units+2*SCALE)
                    offset=root["stores"][0]-root["stores"][1]; active=sum(map(sum,root["pits"]))
                    self.assertLessEqual((offset-active)*SCALE,units); self.assertLessEqual(units,(offset+active)*SCALE)
        EVIDENCE["literal_evaluation"]={"positions_and_models":checked,"max_error_seeds":maximum,"scalar_avx2_bitwise":True,"width17_tail_checked":True}

    def test_03_frozen_model_bitwise_full_development_parity(self):
        path=ROOT/"data/training-stronger2-w256-seed31415/model.json"; count=0
        with FrozenNative(path) as frozen,NeuralNative(path,backend="scalar") as scalar,NeuralNative(path,backend="avx2") as packed:
            for row in pilot_rows("development"):
                a=frozen.evaluate(row["state"]); b=scalar.evaluate(row["state"]); c=packed.evaluate(row["state"])
                self.assertEqual(struct.pack("<d",a["value_player0"]),struct.pack("<d",b["value_player0"]))
                self.assertEqual(struct.pack("<d",a["value_player0"]),struct.pack("<d",c["value_player0"]))
                self.assertEqual(a["value_units"],c["value_units"]); count+=1
        self.assertEqual(count,1000)
        EVIDENCE["legacy_model_parity"]={"development_roots":count,"bitwise_raw_and_units":True}

    def test_04_literal_minimax_all_tt_pvs_backends(self):
        @lru_cache(None)
        def solve(p,depth):
            if literal.normalize(p).terminal or depth==0: return reference_value(p,self.export)[1]
            scores=[solve(literal.apply(p,a).position,depth-1) for a in literal.legal_actions(p)]
            return (max if literal.normalize(p).controller==0 else min)(scores)
        roots=list(random_positions(734,48,2,10)); checked=0
        for backend in ("scalar","avx2"):
            with NeuralNative(self.model_path,backend=backend) as native:
                for p in roots:
                    expected={a:solve(literal.apply(p,a).position,3) for a in literal.legal_actions(p)}
                    for tt,pvs in ((0,False),(1,False),(0,True),(1,True)):
                        r=native.neural_analyze(as_dict(p),max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"]
                        validate_report(r,as_dict(p),native)
                        self.assertEqual({a["pit"]:a["value_units"] for a in r["candidates"]},expected)
                        self.assertEqual(r["value_units"],solve(p,4))
                        optimum=(max if literal.normalize(p).controller==0 else min)(expected.values())
                        self.assertEqual(r["best_moves"],[a for a,v in expected.items() if v==optimum]); checked+=1
        EVIDENCE["literal_minimax"]={"roots":48,"choice_depth":4,"configurations":8,"searches":checked,"full_root_action_scores_match":True}

    def test_05_analysis_actual_kill_resume_identity_and_corruption(self):
        root=config.position(); run=self.path/"analysis"
        worker="""import json,sys
from congklak.neural_fast import NeuralNative,Analysis
with NeuralNative(sys.argv[1],backend='avx2') as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2:sys.stdin.readline()
 j.run(5,0,on_event=event)
 j.close()
"""
        marker=killed_at_marker([sys.executable,"-c",worker,str(self.model_path),str(run),json.dumps(root)],lambda e:e.get("choice_depth")==2)
        with NeuralNative(self.model_path,backend="avx2") as native:
            job=Analysis(run,root,native)
            try:
                self.assertEqual(job.last["choice_depth"],2)
                with self.assertRaises((NativeError,OSError)): Analysis(run,root,native)
                resumed=job.run(5,0)["report"]
            finally: job.close()
            fresh=native.neural_analyze(root,max_depth=5,time_ms=0)["report"]
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
            # Source-equivalent backend changes are still identity changes for timing jobs.
            with NeuralNative(self.model_path,backend="scalar") as scalar:
                with self.assertRaises(pilot.PilotError): Analysis(run,root,scalar)
            altered={**resumed,"input_controller":1-root["controller"]}
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:
                db.execute("UPDATE iterations SET body=?,hash=? WHERE depth=5",(config.canonical(altered),config.digest(altered))); db.commit()
            with self.assertRaises(NativeError): Analysis(run,root,native)
        EVIDENCE["analysis_recovery"]={"actual_hard_kill":True,"last_committed_depth":marker["choice_depth"],"resumed_matches_fresh":True,"incompatible_backend_refused":True,"checksum_valid_corrupt_checkpoint_refused":True}

    def test_06_admission_and_interrupted_first_iteration(self):
        for change in (lambda m:m["identity"].update(input_count=15),lambda m:m.update(uses_oracle=True),lambda m:m["identity"].update(rules_digest="wrong"),lambda m:m["weights"]["layers.0.weight"][0].pop()):
            damaged=json.loads(json.dumps(self.export)); change(damaged)
            path=self.path/"damaged.json"; path.write_text(json.dumps({"model":damaged,"sha256":config.digest(damaged)}))
            with self.assertRaises((NativeError,ValueError)): Model(path)
        with NeuralNative(self.model_path) as native:
            self.assertIsNone(native.neural_analyze(config.position(),max_depth=16,time_ms=0,node_budget=1)["report"])
            with self.assertRaises(ValueError): native.neural_analyze(config.position(),time_ms=-1)
            with native.make_control(max_depth=3,time_ms=0) as control:
                with self.assertRaises(NativeError): native.neural_analyze(config.position(),max_depth=4,time_ms=0,control=control)
        EVIDENCE["admission"]={"malformed_models_refused":4,"first_iteration_has_no_recommendation":True}

    def test_07_metadata_lineage_all_states_and_aliases(self):
        manifest=pilot.load(campaign.DEFAULT); _,reserved,families,registry=campaign.imports(manifest["identity"]["parent_path"])
        self.assertEqual(registry,manifest["identity"]["imported_registry_sha256"])
        path=campaign.DEFAULT/manifest["directory"]/"dataset.sqlite3"
        metadata=[]; quotients={}; counts={}
        with closing(sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)) as db:
            # Every split's metadata, never a sealed body/label.
            for key,split,family in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples ORDER BY state"):
                key=bytes(key); q=pilot.quotient(pilot.from_key(key)); metadata.append(key)
                self.assertNotIn(("state",key),reserved); self.assertNotIn(("quotient",q),reserved); self.assertNotIn(family,families)
                quotients.setdefault(q,set()).add(split); counts[split]=counts.get(split,0)+1
        self.assertEqual(counts,{"train":8000,"development":1000,"sealed_final_test":1000})
        self.assertTrue(all(not ("train" in ownership and len(ownership)>1) for ownership in quotients.values()))
        self.assertTrue(all(not ("development" in ownership and "sealed_final_test" in ownership) for ownership in quotients.values()))
        EVIDENCE["data_lineage"]={"metadata_rows_checked":len(metadata),"prior_exclusion_keys":len(reserved),"prior_families":len(families),"cross_split_quotient_overlaps":0,"sealed_labels_read":False}

    def test_08_cuda_training_actual_kill_restore_optimizer_and_all_rng(self):
        runtime=json.loads((ROOT/"build/training-runtime.json").read_text())["python"]
        run=self.path/"gpu-resumed"; fresh=self.path/"gpu-fresh"
        worker="""import json,sys
from congklak import training_v3 as tr
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='checkpoint_committed' and e.get('step')==10:sys.stdin.readline()
tr.train(sys.argv[1],steps=40,seed=37,width=17,batch_size=32,checkpoint_every=10,device='cuda',input_count=43,on_event=event)
"""
        killed_at_marker([runtime,"-c",worker,str(run)],lambda e:e.get("event")=="checkpoint_committed" and e.get("step")==10)
        verify="""from contextlib import closing
import json,sqlite3,sys,torch
from pathlib import Path
from congklak import training_v3 as tr
def saved(path):
 with closing(sqlite3.connect(Path(path)/'catalog.sqlite3')) as db:name=db.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0]
 return torch.load(Path(path)/name,map_location='cpu',weights_only=True)
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
args=dict(steps=40,seed=37,width=17,batch_size=32,checkpoint_every=10,device='cuda',input_count=43)
resumed=tr.train(sys.argv[1],**args);fresh=tr.train(sys.argv[2],**args)
assert resumed['recovery']['restored_step']==10
a,b=saved(sys.argv[1]),saved(sys.argv[2]);assert equal(a,b)
_,data,_=tr.load_data();assert set(data)=={'train','development'}
assert len(data['train'])==8000 and len(data['development'])==1000
print(json.dumps({'actual_cuda_hard_kill':True,'restored_step':10,'bitwise_full_state_matches_fresh':True,'weights_sha256':resumed['weights_sha256'],'train_rows':len(data['train']),'development_rows':len(data['development']),'sealed_final_test_consumed':False}))
"""
        result=subprocess.run([runtime,"-c",verify,str(run),str(fresh)],cwd=ROOT,text=True,capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        EVIDENCE["cuda_recovery"]=json.loads(result.stdout.strip().splitlines()[-1])

    def test_09_benchmark_actual_kill_contiguous_resume_and_corrupt_regret(self):
        run=self.path/"benchmark"; output=run/"report.json"
        command=[sys.executable,str(ROOT/"tools/benchmark-stage5.py"),"run","--run",str(run),"--output",str(output),"--case-limit","2","--no-arena"]
        environment=dict(os.environ,STAGE5_BLOCK_AFTER_CASE="1")
        killed_at_marker(command,lambda e:e.get("event")=="stage5_case_commit_barrier" and e.get("completed")==1,env=environment)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            original=db.execute("SELECT idx,body,hash FROM cases").fetchall()
        self.assertEqual(len(original),1); self.assertEqual(original[0][0],0)
        resumed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45)
        self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            saved=db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall()
        self.assertEqual([r[0] for r in saved],[0,1]); self.assertEqual(saved[0],original[0])
        repeated=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45)
        self.assertEqual(repeated.returncode,0,repeated.stdout+repeated.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            self.assertEqual(saved,db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall())
            damaged=json.loads(saved[0][1]);value=next(iter(damaged["measurements"]["25"].values()))
            self.assertIsNotNone(value["reference_regret"])
            value["reference_regret"]+=1
            db.execute("UPDATE cases SET body=?,hash=? WHERE idx=0",(config.canonical(damaged),config.digest(damaged)));db.commit()
        corrupted=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45)
        self.assertNotEqual(corrupted.returncode,0,"Checksum-valid corrupt benchmark regret accepted")
        EVIDENCE["benchmark_recovery"]={"actual_hard_kill":True,"last_committed_cases":1,"resumed_contiguous_cases":2,"prior_committed_case_unchanged":True,"idempotent_resume":True,"checksum_valid_corrupt_regret_refused":True,"arena_hard_kill_tested":False}


def pilot_rows(split):
    manifest=pilot.load(campaign.DEFAULT)
    path=campaign.DEFAULT/manifest["directory"]/"dataset.sqlite3"
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro",uri=True)) as db:
        for body,digest in db.execute("SELECT body,hash FROM examples WHERE json_extract(body,'$.split')=? ORDER BY state",(split,)):
            row=json.loads(body)
            if config.digest(row)!=digest: raise AssertionError("Corrupt row")
            yield row


if __name__=="__main__":
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Stage5IndependentAcceptance))
    EVIDENCE.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),passed=result.wasSuccessful(),duration_seconds=time.monotonic()-started)
    (WORK/"tests.json").write_text(json.dumps(EVIDENCE,indent=2)+"\n",encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
