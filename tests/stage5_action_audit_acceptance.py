"""Independent action-head mathematics, recurrence and interruption checks."""
from contextlib import closing
from functools import lru_cache
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
import reference as literal
from stage5_audit_acceptance import literal_position,as_dict,random_positions,killed_at_marker,pilot_rows,sha
from congklak import config,pilot,action_targets,timing,timing_v2
from congklak.native import Native,NativeError,flatten
from congklak.neural_action import Model,NeuralNative,Analysis,validate_report

WORK=ROOT/"build/stage5-action-audit";WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={};SCALE=256


def fixture(path,width=17,seed=29,tied=False):
    rng=random.Random(seed);weights={}
    for prefix,rows,cols in (("layers.0",width,43),("layers.2",width,width),("layers.4",1,width),("policy",7,width)):
        weights[prefix+".weight"]=[[rng.randrange(-8,9)/64 for _ in range(cols)] for _ in range(rows)]
        weights[prefix+".bias"]=[rng.randrange(-8,9)/64 for _ in range(rows)]
    if tied:weights["policy.weight"]=[[0.0]*width for _ in range(7)];weights["policy.bias"]=[0.0]*7
    else:weights["policy.bias"]=[1048576.0,-1048576.0,0.0,0.0,123.0,-456.0,789.0]
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        flat=[v for row in value for v in row] if isinstance(value[0],list) else value
        digest.update(name.encode());digest.update(struct.pack("<"+"f"*len(flat),*flat))
    model={"schema":"congklak-mlp-export-v3","encoder":"current-controller-pit14-active15-occupied14-singleton14/store-offset-exact-v3",
           "identity":{"width":width,"input_count":43,"rules_digest":config.RULES_DIGEST},"weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":model,"sha256":config.digest(model)}));return model


def value_and_policy(p,m):
    normal=literal.normalize(p);actor=normal.controller
    counts=[p.counts[r*8+i] for r in (actor,1-actor) for i in range(7)];active=sum(counts);d=p.counts[7]-p.counts[15]
    if not active:return float(d),d*SCALE,[0.0]*7
    x=[v/98 for v in counts]+[active/98]+[float(v>0) for v in counts]+[float(v==1) for v in counts]
    def dense(prefix,features,relu=False):
        out=[]
        for row,bias in zip(m["weights"][prefix+".weight"],m["weights"][prefix+".bias"]):
            v=0.0
            for a,b in zip(row,features):v+=a*b
            v+=bias;out.append(max(0.0,v) if relu else v)
        return out
    x=dense("layers.0",x,True);x=dense("layers.2",x,True)
    residual=active*(1 if actor==0 else -1)*math.tanh(dense("layers.4",x)[0]);scaled=residual*SCALE
    units=d*SCALE+(math.floor(scaled+.5) if scaled>=0 else math.ceil(scaled-.5))
    return d+residual,units,dense("policy",x)


@lru_cache(None)
def literal_h(p,depth):
    normal=literal.normalize(p)
    if normal.terminal or not depth:return p.counts[7]-p.counts[15]
    scores=[literal_h(literal.apply(p,a).position,depth-1) for a in literal.legal_actions(p)]
    return (max if normal.controller==0 else min)(scores)


def root_row(p,q):
    legal=list(literal.legal_actions(p));sign=1 if literal.normalize(p).controller==0 else -1;optimum=max(sign*v for v in q.values())
    return {"state":as_dict(p),"legal_mask":sum(1<<a for a in legal),"label":{"kind":"finite_choice_horizon","choice_depth":6,"oracle_manifest_sha256":None,
            "action_scores":[{"pit":a,"value_player0":q[a]} for a in legal],"target_player0":sign*optimum,"best_moves":[a for a in legal if sign*q[a]==optimum]}}


class IndependentActionAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="audit-",dir=WORK);self.path=Path(self.temp.name).resolve();self.model_path=self.path/"model.json"
        self.model=fixture(self.model_path)
    def tearDown(self):
        self.assertTrue(self.path.is_relative_to(WORK.resolve()));self.temp.cleanup()

    def test_01_prior_scientific_sources_and_models_frozen(self):
        accepted=json.loads((ROOT/"build/stage5-acceptance.json").read_text());names=[]
        for name in accepted["inputs"]:
            if name.startswith(("native/","data/training-stage5-")) or name in ("congklak/training.py","congklak/training_v2.py","congklak/training_v3.py","congklak/neural.py","congklak/neural_fast.py","congklak/features_v3.py","tools/benchmark-stage5.py","build/stage5-benchmark.json","build/stage5-evaluation-results.json","build/congklak_native.dll","build/congklak_neural.dll","build/congklak_neural_fast.dll"):
                self.assertEqual(sha(ROOT/name),accepted["inputs"][name],name);names.append(name)
        original=json.loads((ROOT/"build/training-acceptance.json").read_text())
        for name,digest in original["inputs"].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE["frozen_prior_science"]={"files_checked":len(names),"old_environment_tools_unchanged":True}

    def test_02_independent_value_logits_and_mode_symmetries(self):
        checked=0
        for width in (17,64):
            file=self.path/f"w{width}.json";model=fixture(file,width)
            roots=list(random_positions(21,96))+[literal.Position(7,tuple([0]*7+[49]+[0]*6+[1]+[48]),0)]
            for mode in ("h-none","h-policy","value-none","value-policy"):
                with NeuralNative(file,backend="scalar",mode=mode) as scalar,NeuralNative(file,backend="avx2",mode=mode) as vector:
                    for p in roots:
                        raw,units,logits=value_and_policy(p,model);a=scalar.evaluate(as_dict(p));b=vector.evaluate(as_dict(p))
                        expected=(p.counts[7]-p.counts[15])*SCALE if mode.startswith("h-") else units
                        self.assertEqual(a["value_units"],expected);self.assertEqual(b["value_units"],expected)
                        self.assertEqual(struct.pack("<d",a["model_value_player0"]),struct.pack("<d",b["model_value_player0"]))
                        self.assertAlmostEqual(a["model_value_player0"],raw,delta=1e-10)
                        for x,y,z in zip(a["policy_logits"],b["policy_logits"],logits):
                            self.assertEqual(struct.pack("<d",x),struct.pack("<d",y));self.assertAlmostEqual(x,z,delta=1e-10)
                        root=as_dict(p);swapped={"pits":root["pits"][::-1],"stores":root["stores"][::-1],"controller":1-root["controller"]}
                        transformed=vector.evaluate(swapped);self.assertEqual(transformed["value_units"],-expected);self.assertEqual(transformed["policy_logits"],a["policy_logits"])
                        if root["stores"][1]:
                            shifted={**root,"stores":[root["stores"][0]+1,root["stores"][1]-1]};other=vector.evaluate(shifted)
                            self.assertEqual(other["value_units"],expected+2*SCALE);self.assertEqual(other["policy_logits"],a["policy_logits"])
                        checked+=1
        EVIDENCE["numeric_modes"]={"positions_and_modes":checked,"scalar_avx_bitwise":True,"policy_head_tail_and_illegal_huge_logits":True}

    def test_03_literal_minimax_four_modes_four_tt_pvs_two_backends(self):
        @lru_cache(None)
        def solve(p,depth,learned):
            normal=literal.normalize(p)
            if normal.terminal or not depth:return value_and_policy(p,self.model)[1] if learned else (p.counts[7]-p.counts[15])*SCALE
            return (max if normal.controller==0 else min)(solve(literal.apply(p,a).position,depth-1,learned) for a in literal.legal_actions(p))
        checked=0;roots=list(random_positions(432,32,2,10));baseline=Native()
        for mode in ("h-none","h-policy","value-none","value-policy"):
            for backend in ("scalar","avx2"):
                with NeuralNative(self.model_path,mode=mode,backend=backend) as n:
                    for p in roots:
                        expected={a:solve(literal.apply(p,a).position,3,mode.startswith("value")) for a in literal.legal_actions(p)}
                        for tt,pvs in ((0,False),(1,False),(0,True),(1,True)):
                            r=n.neural_analyze(as_dict(p),max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"];validate_report(r,as_dict(p),n)
                            self.assertEqual({a["pit"]:a["value_units"] for a in r["candidates"]},expected)
                            self.assertEqual(r["value_units"],solve(p,4,mode.startswith("value")))
                            optimum=(max if literal.normalize(p).controller==0 else min)(expected.values())
                            self.assertEqual(r["best_moves"],[a for a,v in expected.items() if v==optimum]);checked+=1
                            if mode.startswith("h-"):
                                old=baseline.analyze(flatten(as_dict(p)),actor=p.controller,total=98,max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"]
                                self.assertEqual(r["best_moves"],old["best_moves"]);self.assertEqual(r["pv"],old["pv"])
        EVIDENCE["literal_recurrence"]={"roots":32,"depth":4,"configurations":32,"searches":checked,"full_root_scores_and_h_v1_lex_pv":True}

    def test_04_root_h6_policy_target_mask_sign_ties_and_h5_distinction(self):
        checked=0;different=False
        for p in random_positions(413,48,5,9):
            q={a:literal_h(literal.apply(p,a).position,5) for a in literal.legal_actions(p)};row=root_row(p,q)
            probability,mask,signed,sign=action_targets.target(row);legal=list(literal.legal_actions(p))
            actor_sign=1 if literal.normalize(p).controller==0 else -1;self.assertEqual(sign,actor_sign)
            weights=[math.exp((actor_sign*q[a]-max(actor_sign*v for v in q.values()))/4) for a in legal]
            self.assertAlmostEqual(sum(probability),1,delta=1e-15)
            for a in range(7):
                self.assertEqual(mask[a],a in legal)
                self.assertEqual(probability[a],weights[legal.index(a)]/sum(weights) if a in legal else 0.0)
            different |= any(q[a]!=literal_h(literal.apply(p,a).position,6) for a in legal);checked+=1
        self.assertTrue(different,"Fixture must expose child H5 versus H6 disagreement")
        p=next(random_positions(72,1,6,6));legal=list(literal.legal_actions(p));tied=root_row(p,{a:p.counts[7]-p.counts[15] for a in legal})
        probability,mask,q,sign=action_targets.target(tied)
        self.assertTrue(all(probability[a]==1/len(legal) for a in legal))
        bad=json.loads(json.dumps(tied));bad["label"]["choice_depth"]=5
        with self.assertRaises(ValueError):action_targets.target(bad)
        for temperature in (True,0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):action_targets.target(tied,temperature)
        bad=json.loads(json.dumps(tied));bad["label"]["action_scores"]=bad["label"]["action_scores"][:-1]
        with self.assertRaises(ValueError):action_targets.target(bad)
        EVIDENCE["policy_targets"]={"literal_h6_roots":checked,"child_h5_differs_from_child_h6_observed":True,"legal_tie_normalization":True,"actor_is_root_not_successor":True}

    def test_05_timing_binary_admission_and_actual_restore(self):
        def record(logical,core,efficiency=1,flags=0,group=0,size=32):
            raw=bytearray(size);struct.pack_into("<II",raw,0,size,0);struct.pack_into("<IHBBBBBB",raw,8,100+logical,group,logical,core,0,0,efficiency,flags);return bytes(raw)
        raw=record(0,0)+struct.pack("<II",8,77)+record(2,1,size=40)+record(3,1)+record(4,2,flags=1)+record(5,3,flags=2)+record(6,4,efficiency=0)
        parsed=timing.parse_cpu_sets(raw);self.assertEqual(len(parsed),6)
        selected=timing.select_core(parsed,255);self.assertEqual((selected["core"],selected["logical"]),(1,2))
        self.assertEqual(timing.select_core(timing.parse_cpu_sets(record(5,2,flags=6)),1<<5)["logical"],5)
        for malformed in (b"x",struct.pack("<II",0,0),struct.pack("<II",8,0),struct.pack("<II",33,0)+bytes(24)):
            with self.assertRaises(ValueError):timing.parse_cpu_sets(malformed)
        with self.assertRaises(ValueError):timing.select_core(parsed,0)
        k=C.WinDLL("kernel32",use_last_error=True);k.GetCurrentProcess.restype=C.c_void_p
        k.GetProcessAffinityMask.argtypes=[C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        def actual():
            a=C.c_size_t();s=C.c_size_t();self.assertTrue(k.GetProcessAffinityMask(k.GetCurrentProcess(),C.byref(a),C.byref(s)));return a.value
        original=actual()
        preferred=dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1)
        model_row={**preferred,"parked":True,"allocated":False,"allocated_to_process":False}
        self.assertEqual(timing_v2.structural(timing_v2.select_core([model_row],4,preferred)),preferred)
        self.assertEqual(timing_v2.structural(timing_v2.select_core([{**model_row,"parked":False}],4,preferred)),preferred)
        for rows,mask in (([{**model_row,"allocated":True}],4),([model_row],1),([{**model_row,"logical":6}],64),([model_row,{**model_row,"id":300,"group":1}],4)):
            with self.assertRaises(ValueError):timing_v2.select_core(rows,mask,preferred)
        with timing_v2.PerformanceCore(preferred) as core:
            self.assertEqual(actual(),core.mask);core.verify();metadata=core.metadata
        self.assertEqual(actual(),original)
        with self.assertRaisesRegex(RuntimeError,"synthetic exit"):
            with timing_v2.PerformanceCore(preferred):raise RuntimeError("synthetic exit")
        self.assertEqual(actual(),original)
        EVIDENCE["timing"]={"binary_variable_size_unknown_types_checked":True,"parked_allocated_allowed_mask_checked":True,"explicit_parked_target_keeps_same_identity":True,"no_missing_reserved_or_wrong_target_fallback":True,"actual_normal_and_exception_restore":True,"metadata":metadata}

    def test_06_actual_analyzer_kill_resume_mode_threshold_and_corruption(self):
        root=config.position();run=self.path/"analyzer"
        worker="""import json,sys
from congklak.neural_action import NeuralNative,Analysis
with NeuralNative(sys.argv[1],mode='value-policy',backend='avx2') as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2:sys.stdin.readline()
 j.run(5,0,on_event=event)
 j.close()
"""
        killed_at_marker([sys.executable,"-c",worker,str(self.model_path),str(run),json.dumps(root)],lambda e:e.get("choice_depth")==2)
        with NeuralNative(self.model_path,mode="value-policy",backend="avx2") as n:
            job=Analysis(run,root,n)
            try:self.assertEqual(job.last["choice_depth"],2);resumed=job.run(5,0)["report"]
            finally:job.close()
            fresh=n.neural_analyze(root,max_depth=5,time_ms=0)["report"]
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
            for mode,threshold in (("value-none",2),("value-policy",3)):
                with NeuralNative(self.model_path,mode=mode,policy_min_depth=threshold,backend="avx2") as other:
                    with self.assertRaises(pilot.PilotError):Analysis(run,root,other)
            bad={**resumed,"mode":"h-policy"}
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:
                db.execute("UPDATE iterations SET body=?,hash=? WHERE depth=5",(config.canonical(bad),config.digest(bad)));db.commit()
            with self.assertRaises(NativeError):Analysis(run,root,n)
        EVIDENCE["analysis_recovery"]={"actual_kill_depth":2,"resumed_depth":5,"fresh_equal":True,"mode_threshold_and_hash_valid_corruption_refused":True}

    def test_07_cuda_losses_train_only_eligibility_and_actual_precommit_kill(self):
        runtime=json.loads((ROOT/"build/training-runtime.json").read_text())["python"];run=self.path/"gpu-resumed";fresh=self.path/"gpu-fresh"
        worker="""import json,sys
from congklak import training_v4 as tr
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='checkpoint_file_ready' and e.get('step')==20:sys.stdin.readline()
tr.train(sys.argv[1],steps=40,seed=41,width=17,batch_size=32,checkpoint_every=10,variant='C',device='cuda',on_event=event)
"""
        killed_at_marker([runtime,"-c",worker,str(run)],lambda e:e.get("event")=="checkpoint_file_ready" and e.get("step")==20)
        script="""from contextlib import closing
import json,sqlite3,sys,torch
from pathlib import Path
from congklak import training_v4 as tr,action_targets
def saved(path):
 with closing(sqlite3.connect(Path(path)/'catalog.sqlite3')) as d:name=d.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0]
 return torch.load(Path(path)/name,map_location='cpu',weights_only=True)
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
prob=torch.tensor([[.5,0,.5,0,0,0,0]],dtype=torch.float32);mask=prob>0
logits=torch.tensor([[0.,1000000.,0.,0.,0.,0.,0.]],requires_grad=True)
loss=tr.policy_loss(logits,prob,mask);assert torch.isfinite(loss) and abs(loss.item()-0.69314718056)<1e-6
loss.backward();assert torch.isfinite(logits.grad).all() and logits.grad[0,1]==0
error=torch.tensor([2.,-4.,6.,8.],requires_grad=True);sign=torch.tensor([1.,-1.,1.,-1.]);group=torch.tensor([0,0,1,-1])
assert tr.ordinary_bias_loss(error,sign,group,3).item()==22.5
zero=tr.ordinary_bias_loss(error,sign,torch.tensor([-1]*4),3);assert zero.item()==0;zero.backward();assert torch.isfinite(error.grad).all()
_,data,_=tr.load_data();assert set(data)=={'train','development'} and len(data['train'])==8000 and len(data['development'])==1000
cal=action_targets.ordinary_calibration(data['train']);assert not cal['development_calibration']
args=dict(steps=40,seed=41,width=17,batch_size=32,checkpoint_every=10,variant='C',device='cuda')
a=tr.train(sys.argv[1],**args);b=tr.train(sys.argv[2],**args);assert a['recovery']['restored_step']==10
assert equal(saved(sys.argv[1]),saved(sys.argv[2]))
try:tr.train(sys.argv[1],**{**args,'variant':'B'})
except tr.TrainingError:pass
else:raise AssertionError('Changed objective accepted')
print(json.dumps(dict(actual_cuda_precommit_kill=True,uncommitted_step=20,restored_step=10,full_heads_optimizer_all_rng_bitwise=True,objective_change_refused=True,train_rows=8000,dev_rows=1000,successor_value_rows=0,sealed_labels_read=False,train_only_calibration=cal,masked_ce_and_missing_strata_finite=True)))
"""
        result=subprocess.run([runtime,"-c",script,str(run),str(fresh)],cwd=ROOT,text=True,capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);EVIDENCE["cuda_and_objectives"]=json.loads(result.stdout.strip().splitlines()[-1])

    def test_08_actual_pinned_benchmark_kill_resume_and_corrupt_regret(self):
        run=self.path/"benchmark";command=[sys.executable,str(ROOT/"tools/benchmark-stage5-action-v2.py"),"run","--run",str(run),"--output",str(run/"report.json"),"--case-limit","2","--no-arena"]
        environment=dict(os.environ,STAGE5_ACTION_BLOCK_AFTER_CASE="1")
        killed_at_marker(command,lambda e:e.get("event")=="action_case_commit_barrier" and e.get("completed")==1,env=environment)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            original=db.execute("SELECT idx,body,hash FROM cases").fetchall()
        self.assertEqual(len(original),1);self.assertEqual(original[0][0],0)
        resumed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45);self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:saved=db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall()
        self.assertEqual([r[0] for r in saved],[0,1]);self.assertEqual(saved[0],original[0])
        repeated=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45);self.assertEqual(repeated.returncode,0,repeated.stdout+repeated.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            self.assertEqual(saved,db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall())
            wrong=json.loads(saved[0][1]);next(iter(wrong["measurements"]["25"].values()))["reference_regret"]+=1
            db.execute("UPDATE cases SET body=?,hash=? WHERE idx=0",(config.canonical(wrong),config.digest(wrong)));db.commit()
        refused=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=45);self.assertNotEqual(refused.returncode,0)
        EVIDENCE["benchmark_recovery"]={"actual_process_kill":True,"last_committed_cases":1,"resumed_cases":2,"prior_case_unchanged":True,"idempotent_resume":True,"forged_checksum_valid_regret_refused":True,"same_cpu_affinity_identity_restored":True,"arena_hard_kill_tested":False}

    def test_09_actual_completed_arena_game_kill_resume_on_isolated_backup(self):
        run=self.path/"arena";run.mkdir();production=ROOT/"data/stage5-action-benchmark-v2/work.sqlite3"
        with closing(sqlite3.connect(production.as_uri()+"?mode=ro",uri=True)) as source,closing(sqlite3.connect(run/"work.sqlite3")) as copied:
            source.backup(copied)
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM cases").fetchone()[0],391)
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM games").fetchone()[0],144)
            copied.execute("DELETE FROM games");copied.execute("DELETE FROM meta WHERE key='action_complete'");copied.commit()
        # Observe the constructor identity without changing frozen benchmark
        # code or search behavior; instrumentation only writes a test sidecar.
        worker="""import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('audit_arena_worker',Path.cwd()/'tools/benchmark-stage5-action-v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=m._work.__enter__
def entering(self):
 Path(self.path,'observed-identity.json').write_text(json.dumps(self.identity,indent=2))
 return prior(self)
m._work.__enter__=entering
m.run(Path(sys.argv[1]),Path(sys.argv[1])/'report.json')
"""
        command=[sys.executable,"-c",worker,str(run)]
        try:killed_at_marker(command,lambda e:e.get("event")=="action_arena_game_committed" and e.get("completed")==1)
        except AssertionError:
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:expected=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0])
            observed=json.loads((run/"observed-identity.json").read_text())
            drift={key:{"stored":expected[key],"current":observed.get(key)} for key in expected if expected[key]!=observed.get(key)}
            (WORK/"arena-identity-drift.json").write_text(json.dumps(drift,indent=2)+"\n")
            raise
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:first=db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall()
        self.assertEqual(len(first),1)
        try:killed_at_marker(command,lambda e:e.get("event")=="action_arena_game_committed" and e.get("completed")==2)
        except AssertionError:
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:expected=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0])
            observed=json.loads((run/"observed-identity.json").read_text())
            drift={key:{"stored":expected[key],"current":observed.get(key)} for key in expected if expected[key]!=observed.get(key)}
            (WORK/"arena-identity-drift.json").write_text(json.dumps(drift,indent=2)+"\n")
            raise
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:second=db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall()
        self.assertEqual(len(second),2);self.assertEqual(second[0],first[0])
        validation=[sys.executable,str(ROOT/"tools/benchmark-stage5-action-v2.py"),"run","--run",str(run),"--output",str(run/"report.json"),"--no-arena"]
        result=subprocess.run(validation,cwd=ROOT,text=True,capture_output=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:self.assertEqual(second,db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall())
        for comparison,pair,seat,body,digest in second:
            game=json.loads(body);self.assertEqual(config.digest(game),digest);p=literal_position(game["start"])
            for move in game["moves"]:p=literal.apply(p,move).position
            self.assertEqual(as_dict(p),game["final"]);self.assertTrue(literal.normalize(p).terminal)
        with closing(sqlite3.connect(production.as_uri()+"?mode=ro",uri=True)) as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM games").fetchone()[0],144)
        EVIDENCE["arena_recovery"]={"actual_arena_hard_kill":True,"process_kills":2,"isolated_sqlite_backup":True,"production_cases_reused":391,"completed_games_after_resume":2,"first_completed_game_byte_unchanged":True,"idempotent_saved_game_validation":True,"both_games_literal_replayed":True,"production_games_untouched":144}


if __name__=="__main__":
    import time
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentActionAcceptance))
    EVIDENCE.update(tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),duration_seconds=time.monotonic()-start)
    (WORK/"tests.json").write_text(json.dumps(EVIDENCE,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
