"""Independent literal relay features, horizons, native recurrence and recovery."""
from contextlib import closing
from functools import lru_cache
import ctypes as C
import hashlib
import importlib
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
import time
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))
import reference as literal
from stage5_audit_acceptance import literal_position,as_dict,random_positions,killed_at_marker,sha
from congklak import config,pilot,campaign,timing_v2
from congklak.native import Native,NativeError,flatten

WORK=ROOT/"build/stage5-deeper-audit";WORK.mkdir(parents=True,exist_ok=True)
EVIDENCE={};SCALE=256
ENCODER43="current-controller-pit14-active15-occupied14-singleton14/store-offset-exact-v3"
ENCODER127="current-controller-count43-own-opponent-choice14-storegain-drops-relays-capture-retain-terminal84/store-offset-exact-v5"


def fixture(path,inputs=127,width=17,seed=231):
    rng=random.Random(seed);weights={}
    for prefix,rows,cols in (("layers.0",width,inputs),("layers.2",width,width),("layers.4",1,width),("policy",7,width)):
        weights[prefix+".weight"]=[[rng.randrange(-8,9)/64 for _ in range(cols)] for _ in range(rows)]
        weights[prefix+".bias"]=[rng.randrange(-8,9)/64 for _ in range(rows)]
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        flat=[v for row in value for v in row] if isinstance(value[0],list) else value
        digest.update(name.encode());digest.update(struct.pack("<"+"f"*len(flat),*flat))
    model={"schema":"congklak-mlp-export-v4","encoder":ENCODER43 if inputs==43 else ENCODER127,
           "identity":{"width":width,"input_count":inputs,"rules_digest":config.RULES_DIGEST},
           "weights":weights,"weights_sha256":digest.hexdigest(),"uses_oracle":False}
    path.write_text(json.dumps({"model":model,"sha256":config.digest(model)}));return model


@lru_cache(maxsize=16384)
def literal_features(p,inputs=127):
    """Only the independent seed-by-seed interpreter supplies transitions."""
    actor=literal.normalize(p).controller
    counts=[p.counts[row*8+i] for row in (actor,1-actor) for i in range(7)]
    active=sum(counts);d=p.counts[7]-p.counts[15]
    x=[n/98 for n in counts]+[active/98]+[float(n>0) for n in counts]+[float(n==1) for n in counts]
    if inputs==127:
        for side in (actor,1-actor):
            for pit in range(7):
                if not p.counts[side*8+pit]:x.extend([0.0]*6);continue
                hypothetical=literal.Position(7,p.counts,side)
                t=literal.apply(hypothetical,pit)
                x.extend([(t.position.counts[side*8+7]-p.counts[side*8+7])/98,
                          t.deposits/98,t.relay_pickups/98,float(t.captured>0),
                          float(not t.terminal and t.position.controller==side),float(t.terminal)])
    elif inputs!=43:raise ValueError("Unknown independent input count")
    return tuple(x),d,active*(1 if actor==0 else -1)


def value_and_policy(p,m):
    x,d,r=literal_features(p,m["identity"]["input_count"])
    if not r:return float(d),d*SCALE,[0.0]*7
    def dense(prefix,features,relu=False):
        out=[]
        for row,bias in zip(m["weights"][prefix+".weight"],m["weights"][prefix+".bias"]):
            v=0.0
            for a,b in zip(row,features):v+=a*b
            v+=bias;out.append(max(0.0,v) if relu else v)
        return out
    x=dense("layers.0",x,True);x=dense("layers.2",x,True)
    residual=r*math.tanh(dense("layers.4",x)[0]);scaled=residual*SCALE
    units=d*SCALE+(math.floor(scaled+.5) if scaled>=0 else math.ceil(scaled-.5))
    return d+residual,units,dense("policy",x)


@lru_cache(None)
def literal_h(p,depth):
    normal=literal.normalize(p)
    if normal.terminal or not depth:return p.counts[7]-p.counts[15]
    scores=[literal_h(literal.apply(p,a).position,depth-1) for a in literal.legal_actions(p)]
    return (max if normal.controller==0 else min)(scores)


def source_rows(split):
    manifest=pilot.load(campaign.DEFAULT)
    with closing(sqlite3.connect((campaign.DEFAULT/manifest["directory"]/"dataset.sqlite3").as_uri()+"?mode=ro",uri=True)) as db:
        for body,digest in db.execute("SELECT body,hash FROM examples WHERE json_extract(body,'$.split')=? ORDER BY state",(split,)):
            row=json.loads(body)
            if config.digest(row)!=digest:raise AssertionError("Corrupt corrected teacher source")
            yield row


def frozen_prior():
    checked={}
    accepted=json.loads((ROOT/"build/stage5-action-acceptance.json").read_text())
    for name,digest in accepted["inputs"].items():
        if name in ("congklak/progress.py","tools/progress-dashboard.html","tests/progress_acceptance.py") or name.startswith("docs/") or name in ("README.md","PROGRESS.md"):
            continue
        if name.startswith(("congklak/","native/","data/","build/","tests/","tools/")):
            if sha(ROOT/name)!=digest:raise AssertionError("Prior scientific input changed: "+name)
            checked[name]=digest
    for name,digest in json.loads((ROOT/"build/training-acceptance.json").read_text())["inputs"].items():
        if sha(ROOT/name)!=digest:raise AssertionError("Prior training environment changed: "+name)
        checked[name]=digest
    return checked


class IndependentDeeperAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="audit-",dir=WORK);self.path=Path(self.temp.name).resolve()
    def tearDown(self):
        self.assertTrue(self.path.is_relative_to(WORK.resolve()));self.temp.cleanup()

    def test_01_prior_science_environment_unchanged(self):
        checked=frozen_prior();self.assertGreaterEqual(len(checked),36)
        EVIDENCE["frozen_prior"]={"files_checked":len(checked),"scientific_inputs":checked}

    def test_02_literal_relay_features_and_store_independence(self):
        from congklak import features_relay
        roots=list(random_positions(20261016,192))
        roots.extend([literal.Position(7,tuple([0]*7+[49]+[0]*6+[1]+[48]),0),
                      literal.Position(7,tuple([0]*7+[60]+[0]*7+[38]),1),
                      literal.Position(7,tuple([0]*6+[1]+[48]+[0]*6+[1]+[48]),0)])
        seen={"capture":False,"relay":False,"terminal":False,"retained":False,"passed":False,"unclipped_drops":False}
        checked=0
        for p in roots:
            for count in (43,127):
                expected,d,r=literal_features(p,count);x,actual_d,actual_r=features_relay.encode(as_dict(p),count)
                self.assertEqual(tuple(x),expected);self.assertEqual((actual_d,actual_r),(d,r));self.assertEqual(len(x),count)
            x=literal_features(p)[0];actor=literal.normalize(p).controller
            for index,side in enumerate((actor,1-actor)):
                for pit in range(7):
                    values=x[43+(index*7+pit)*6:43+(index*7+pit+1)*6]
                    if not p.counts[side*8+pit]:self.assertEqual(values,(0.0,)*6);continue
                    t=literal.apply(literal.Position(7,p.counts,side),pit)
                    seen["capture"]|=bool(t.captured);seen["relay"]|=bool(t.relay_pickups);seen["terminal"]|=t.terminal
                    seen["retained"]|=bool(values[4]);seen["passed"]|=bool(t.final_passes);seen["unclipped_drops"]|=values[1]>1
            swapped=literal.Position(7,p.counts[8:]+p.counts[:8],1-p.controller)
            self.assertEqual(literal_features(swapped)[0],x)
            if p.counts[15]:
                cells=list(p.counts);cells[7]+=1;cells[15]-=1;shifted=literal.Position(7,tuple(cells),p.controller)
                self.assertEqual(features_relay.encode(as_dict(shifted))[0],list(x))
                self.assertEqual(literal_features(shifted)[0],x)
            checked+=1
        self.assertTrue(all(seen.values()),seen)
        EVIDENCE["literal_features"]={"roots":checked,"feature_count":127,"hypothetical_own_and_opponent_actions":True,"empty_pits_zero":True,
                                     "normalized_successor_control_terminal_separate":True,"store_offset_and_swap_independent":True,"mechanisms_observed":seen}

    def test_03_complete_h8_root_q_is_successor_h7(self):
        checked=0;h6_differences=0;child8_differences=0
        native=Native()
        for p in random_positions(77,32,5,9):
            legal=list(literal.legal_actions(p));expected={a:literal_h(literal.apply(p,a).position,7) for a in legal}
            result=native.analyze(flatten(as_dict(p)),actor=p.controller,total=98,max_depth=8,time_ms=0,node_budget=0,tt_mb=16,use_pvs=True)
            self.assertEqual(result["stop_reason"],"max_depth");report=result["report"]
            self.assertEqual(report["choice_depth"],8);q={a["pit"]:a["depth_result"]["value_player0"] for a in report["candidates"]}
            self.assertEqual(q,expected);actor=literal.normalize(p).controller
            self.assertEqual(report["depth_result"]["value_player0"],(max if actor==0 else min)(expected.values()))
            h6_differences+=any(q[a]!=literal_h(literal.apply(p,a).position,5) for a in legal)
            child8_differences+=any(q[a]!=literal_h(literal.apply(p,a).position,8) for a in legal);checked+=1
        self.assertGreater(h6_differences,0);self.assertGreater(child8_differences,0)
        EVIDENCE["teacher_recurrence"]={"independent_h8_roots":checked,"successor_h7":True,"h6_differences":h6_differences,"incorrect_successor_h8_differences":child8_differences}

    def test_04_native_scalar_avx_both_heads_and_tails(self):
        from congklak.neural_relay import NeuralNative
        roots=list(random_positions(812,96))
        roots.extend([literal.Position(7,tuple([0]*7+[49]+[0]*6+[1]+[48]),0),literal.Position(7,tuple([0]*7+[60]+[0]*7+[38]),1)])
        checked=0;max_raw=0;max_logits=0
        for inputs,width in ((43,17),(43,64),(127,17),(127,64)):
            path=self.path/f"fixture-{inputs}-{width}.json";model=fixture(path,inputs,width)
            with NeuralNative(path,mode="value-none",backend="scalar") as scalar,NeuralNative(path,mode="value-none",backend="avx2") as vector:
                for p in roots:
                    root=as_dict(p);expected,units,logits=value_and_policy(p,model)
                    feature=vector.feature_vector(root);self.assertEqual(tuple(feature["features"]),literal_features(p,inputs)[0])
                    a=scalar.evaluate(root);b=vector.evaluate(root)
                    for key in ("value_player0","model_value_player0"):
                        self.assertEqual(struct.pack("<d",a[key]),struct.pack("<d",b[key]));self.assertAlmostEqual(a[key],expected,delta=1e-10)
                    self.assertEqual(a["value_units"],b["value_units"]);self.assertEqual(a["value_units"],units)
                    self.assertEqual(a["policy_logits"],b["policy_logits"])
                    max_raw=max(max_raw,abs(expected-a["value_player0"]));max_logits=max(max_logits,max(abs(x-y) for x,y in zip(logits,a["policy_logits"])))
                    self.assertLessEqual(max_logits,1e-10)
                    swapped={"pits":root["pits"][::-1],"stores":root["stores"][::-1],"controller":1-root["controller"]}
                    swap=vector.evaluate(swapped);self.assertEqual(swap["value_units"],-units);self.assertEqual(swap["policy_logits"],b["policy_logits"])
                    if root["stores"][1]:
                        shifted={**root,"stores":[root["stores"][0]+1,root["stores"][1]-1]}
                        s=vector.evaluate(shifted);self.assertEqual(s["value_units"],units+2*SCALE);self.assertEqual(s["policy_logits"],b["policy_logits"])
                    checked+=1
        EVIDENCE["native_forward"]={"evaluations":checked,"scalar_avx_both_heads_bitwise":True,"tail_width17":True,"max_literal_raw_error":max_raw,"max_literal_logit_error":max_logits}

    def test_05_literal_native_minimax_and_controlled_feature_work(self):
        from congklak.neural_relay import NeuralNative,validate_report
        checked=0;feature_searches=0
        roots=list(random_positions(1901,24,5,9));baseline=Native()
        for inputs in (43,127):
            path=self.path/f"minimax-{inputs}.json";model=fixture(path,inputs,17)
            @lru_cache(None)
            def minimax(p,depth,mode):
                normal=literal.normalize(p)
                if normal.terminal or not depth:return (p.counts[7]-p.counts[15])*SCALE if mode=="h-none" else value_and_policy(p,model)[1]
                values=[minimax(literal.apply(p,a).position,depth-1,mode) for a in literal.legal_actions(p)]
                return (max if normal.controller==0 else min)(values)
            for mode in ("h-none","value-none"):
                with NeuralNative(path,mode=mode,backend="avx2") as n:
                    for tt in (0,1):
                        for pvs in (False,True):
                            for p in roots:
                                q={a:minimax(literal.apply(p,a).position,3,mode) for a in literal.legal_actions(p)}
                                result=n.neural_analyze(as_dict(p),max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs);self.assertEqual(result["stop_reason"],"max_depth")
                                r=result["report"];validate_report(r,as_dict(p),n)
                                self.assertEqual({a["pit"]:a["value_units"] for a in r["candidates"]},q)
                                optimum=(max if literal.normalize(p).controller==0 else min)(q.values())
                                self.assertEqual(r["value_units"],optimum);self.assertEqual(r["best_moves"],[a for a,v in q.items() if v==optimum]);self.assertEqual(r["policy_ordered_nodes"],0)
                                self.assertLessEqual(r["feature_deposits"],r["deposits"])
                                if inputs==127 and mode=="value-none" and r["nn_evaluations"]:
                                    self.assertGreater(r["feature_simulations"],0);self.assertGreater(r["feature_deposits"],0);feature_searches+=1
                                else:self.assertEqual((r["feature_simulations"],r["feature_deposits"]),(0,0))
                                if mode=="h-none":
                                    original=baseline.analyze(flatten(as_dict(p)),actor=p.controller,total=98,max_depth=4,time_ms=0,tt_mb=tt,use_pvs=pvs)["report"]
                                    self.assertEqual(r["best_moves"],original["best_moves"]);self.assertEqual(r["pv"],original["pv"])
                                checked+=1
                    with n.make_control(max_depth=16,time_ms=0) as c:
                        c.cancel();stopped=n.neural_analyze(config.position(),max_depth=16,time_ms=0,control=c)
                        self.assertIsNone(stopped["report"]);self.assertEqual(stopped["stop_reason"],"cancelled")
        self.assertGreater(feature_searches,0)
        EVIDENCE["literal_search"]={"roots":24,"searches":checked,"remaining_choice_depth":4,"both_inputs_and_modes":True,"tt_and_pvs_configurations":4,
                                   "actual_feature_searches":feature_searches,"feature_deposits_in_total_control_work":True,"pre_canceled_control_no_partial_report":True,"h_v1_full_root_lex_pv_equal":True}

    def test_06_actual_analyzer_kill_resume_and_checksum_valid_corruption(self):
        from congklak.neural_relay import NeuralNative,Analysis
        file=self.path/"analyzer-model.json";fixture(file,127,17);run=self.path/"analyzer";root=config.position()
        worker="""import json,sys
from congklak.neural_relay import NeuralNative,Analysis
with NeuralNative(sys.argv[1],mode='value-none',backend='avx2') as n:
 j=Analysis(sys.argv[2],json.loads(sys.argv[3]),n)
 def event(e):
  print(json.dumps(e),flush=True)
  if e['choice_depth']==2:sys.stdin.readline()
 try:j.run(4,0,on_event=event)
 finally:j.close()
"""
        killed_at_marker([sys.executable,"-c",worker,str(file),str(run),json.dumps(root)],lambda e:e.get("choice_depth")==2)
        with NeuralNative(file,mode="value-none",backend="avx2") as n:
            job=Analysis(run,root,n)
            try:self.assertEqual(job.last["choice_depth"],2);resumed=job.run(4,0)["report"]
            finally:job.close()
            fresh=n.neural_analyze(root,max_depth=4,time_ms=0)["report"]
            self.assertEqual((resumed["value_units"],resumed["best_moves"],resumed["pv"]),(fresh["value_units"],fresh["best_moves"],fresh["pv"]))
            with NeuralNative(file,mode="h-none",backend="avx2") as other:
                with self.assertRaises(pilot.PilotError):Analysis(run,root,other)
            bad={**resumed,"feature_deposits":resumed["deposits"]+1}
            with closing(sqlite3.connect(run/"work.sqlite3")) as db:
                db.execute("UPDATE iterations SET body=?,hash=? WHERE depth=4",(config.canonical(bad),config.digest(bad)));db.commit()
            with self.assertRaises(NativeError):Analysis(run,root,n)
        EVIDENCE["analyzer_recovery"]={"actual_process_kill":True,"committed_depth":2,"resumed_depth":4,"fresh_value_moves_pv_equal":True,"mode_and_forged_feature_counter_refused":True}

    def test_07_teacher_actual_precommit_kill_resume_and_forged_cache_refusal(self):
        from congklak import deeper_teacher_v2 as teacher
        run=self.path/"teacher"
        worker="""import json,sys
from congklak import deeper_teacher_v2 as tr
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='deeper_label_ready' and e.get('committed_labels')==1:sys.stdin.readline()
tr.generate(sys.argv[1],on_event=event,label_limit=2)
"""
        killed_at_marker([sys.executable,"-c",worker,str(run)],lambda e:e.get("event")=="deeper_label_ready" and e.get("committed_labels")==1,timeout=120)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            original=db.execute("SELECT state,body,hash FROM labels ORDER BY state").fetchall()
            self.assertEqual(len(original),1);self.assertEqual(db.execute("SELECT COUNT(*) FROM roots").fetchone()[0],9000)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM features").fetchone()[0],2)
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0],"wal")
        resumed=teacher.generate(run,label_limit=1);self.assertEqual(resumed["labels"],2);self.assertEqual(resumed["restored_labels"],1)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT state,body,hash FROM labels WHERE state=?",(original[0][0],)).fetchone(),original[0])
            key,body,digest=db.execute("SELECT state,body,hash FROM features ORDER BY state LIMIT 1").fetchone()
            item=json.loads(body);item["x"][43]+=.125
            db.execute("UPDATE features SET body=?,hash=? WHERE state=?",(config.canonical(item),config.digest(item),key));db.commit()
        with self.assertRaisesRegex(teacher.DeeperError,"derivation"):teacher.generate(run,label_limit=1)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            db.execute("UPDATE features SET body=?,hash=? WHERE state=?",(body,digest,key));db.commit()
            key,label_body,label_digest=db.execute("SELECT state,body,hash FROM labels ORDER BY state LIMIT 1").fetchone()
            item=json.loads(label_body);item["label"]["work"]["elapsed_ms"]+=1
            db.execute("UPDATE labels SET body=?,hash=? WHERE state=?",(config.canonical(item),config.digest(item),key));db.commit()
        with self.assertRaisesRegex(teacher.DeeperError,"work/coverage"):teacher.generate(run,label_limit=1)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            db.execute("UPDATE labels SET body=?,hash=? WHERE state=?",(label_body,label_digest,key));db.commit()
        ident,_=teacher.identity();owner=teacher.Work(run,ident)
        try:
            self.assertEqual(owner.db.execute("PRAGMA synchronous").fetchone()[0],2)
            with self.assertRaises(OSError):teacher.Work(run,ident)
        finally:owner.close()
        for limit in (True,0,-1,1.5):
            with self.assertRaises(teacher.DeeperError):teacher.generate(run,label_limit=limit)
        EVIDENCE["teacher_recovery"]={"actual_precommit_process_kill":True,"last_committed_labels":1,"orphan_feature_safe":True,
                                      "resumed_labels":2,"prior_label_byte_unchanged":True,"checksum_valid_suffix_and_work_refused":True,
                                      "exclusive_writer_and_FULL_WAL":True,"only_original_9000_roots":True,"sealed_target_bodies_read":False}

    def test_08_cuda_actual_precommit_kill_full_state_bitwise(self):
        runtime=json.loads((ROOT/"build/training-runtime.json").read_text())["python"];run=self.path/"cuda-resumed";fresh=self.path/"cuda-fresh"
        worker="""import json,sys
from congklak import training_v7 as tr
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='checkpoint_file_ready' and e.get('step')==20:sys.stdin.readline()
tr.train(sys.argv[1],steps=40,seed=41,width=17,batch_size=32,checkpoint_every=10,input_count=127,teacher_depth=8,device='cuda',on_event=event)
"""
        killed_at_marker([runtime,"-c",worker,str(run)],lambda e:e.get("event")=="checkpoint_file_ready" and e.get("step")==20,timeout=240)
        script="""from contextlib import closing
import json,sqlite3,sys,torch
from pathlib import Path
from congklak import training_v7 as tr,config
def saved(path):
 with closing(sqlite3.connect(Path(path)/'catalog.sqlite3')) as d:name=d.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0]
 return torch.load(Path(path)/name,map_location='cpu',weights_only=True)
def equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
args=dict(steps=40,seed=41,width=17,batch_size=32,checkpoint_every=10,input_count=127,teacher_depth=8,device='cuda')
a=tr.train(sys.argv[1],**args);b=tr.train(sys.argv[2],**args);assert a['recovery']['restored_step']==10
left=saved(sys.argv[1]);right=saved(sys.argv[2]);assert equal(left,right)
assert len(left['optimizer']['state'])==6 and len(left['optimizer']['param_groups'][0]['params'])==6
torch.manual_seed(41);initial=tr.Evaluator(17,127)
assert all(torch.equal(left['model'][k],v) for k,v in initial.state_dict().items() if k.startswith('policy.'))
assert all(not p.requires_grad for p in initial.policy.parameters())
paths=[Path(sys.argv[1])/name for name in ('model.json','result.json')]
before={str(path):path.read_bytes() for path in paths}
def unchanged_snapshot():
 return {str(p.relative_to(Path(sys.argv[1]))):(p.read_bytes(),p.stat().st_mtime_ns) for p in Path(sys.argv[1]).iterdir() if p.is_file()}
stable=unchanged_snapshot()
completed=tr.train(sys.argv[1],**args);assert completed==a and unchanged_snapshot()==stable
for field in ('uses_oracle','weight'):
 path=paths[0];envelope=json.loads(before[str(path)]);m=envelope['model']
 if field=='uses_oracle':m['uses_oracle']=True
 else:m['weights']['layers.0.weight'][0][0]+=.125
 envelope['sha256']=config.digest(m);path.write_text(json.dumps(envelope))
 try:tr.train(sys.argv[1],**args)
 except tr.TrainingError:pass
 else:raise AssertionError('Forged completed export accepted')
 finally:path.write_bytes(before[str(path)])
path=paths[1];result=json.loads(before[str(path)]);result['selected_cross_teacher_development']['H8']['finite_choice_horizon']['mae_seeds']+=1;path.write_text(json.dumps(result))
try:tr.train(sys.argv[1],**args)
except tr.TrainingError:pass
else:raise AssertionError('Forged completed metrics accepted')
finally:path.write_bytes(before[str(path)])
for changed in ({'teacher_depth':6},{'input_count':43}):
 try:tr.train(sys.argv[1],**{**args,**changed})
 except tr.TrainingError:pass
 else:raise AssertionError('Changed teacher/feature identity accepted')
print(json.dumps(dict(actual_cuda_precommit_kill=True,uncommitted_step=20,restored_step=10,full_model_optimizer_sampler_python_torch_cuda_best_bitwise=True,
 dummy_policy_frozen_without_optimizer_decay=True,teacher_and_feature_change_refused=True,root_only_H8_value_objective=True,
 completed_model_result_catalog_checkpoint_bytes_and_mtime_unchanged=True,checksum_valid_completed_oracle_flag_weights_metrics_refused=True,sealed_target_bodies_read=False)))
"""
        result=subprocess.run([runtime,"-c",script,str(run),str(fresh)],cwd=ROOT,text=True,capture_output=True,timeout=600)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);EVIDENCE["cuda_recovery"]=json.loads(result.stdout.strip().splitlines()[-1])

    def test_09_cpu_affinity_stable_admission_and_restore(self):
        kernel=C.WinDLL("kernel32",use_last_error=True);kernel.GetCurrentProcess.restype=C.c_void_p
        kernel.GetProcessAffinityMask.argtypes=[C.c_void_p,C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]
        def mask():
            a=C.c_size_t();b=C.c_size_t();self.assertTrue(kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),C.byref(a),C.byref(b)));return a.value
        original=mask();preferred=dict(id=258,group=0,logical=2,core=2,cache=0,numa=0,efficiency_class=1)
        transient={**preferred,"parked":True,"allocated":False,"allocated_to_process":False}
        self.assertEqual(timing_v2.structural(timing_v2.select_core([transient],4,preferred)),preferred)
        self.assertEqual(timing_v2.structural(timing_v2.select_core([{**transient,"parked":False}],4,preferred)),preferred)
        for entries,allowed in (([{**transient,"allocated":True}],4),([transient],1),([{**transient,"logical":6}],64)):
            with self.assertRaises(ValueError):timing_v2.select_core(entries,allowed,preferred)
        with timing_v2.PerformanceCore(preferred) as cpu:
            self.assertEqual(mask(),4);self.assertEqual(cpu.verify(),dict(group=0,logical=2,mask=4));metadata=cpu.metadata
        self.assertEqual(mask(),original)
        with self.assertRaisesRegex(RuntimeError,"synthetic exit"):
            with timing_v2.PerformanceCore(preferred):raise RuntimeError("synthetic exit")
        self.assertEqual(mask(),original)
        EVIDENCE["cpu"]={"parked_flag_does_not_change_scientific_identity":True,"reserved_or_missing_core_refused":True,
                         "actual_normal_and_exception_restore":True,"metadata":metadata}

    def test_10_actual_pinned_benchmark_kill_resume_and_corrupt_regret(self):
        run=self.path/"benchmark"
        command=[sys.executable,str(ROOT/"tools/benchmark-stage5-deeper.py"),"run","--run",str(run),"--output",str(run/"report.json"),"--case-limit","2","--no-arena"]
        environment=dict(os.environ,STAGE5_DEEPER_BLOCK_AFTER_CASE="1")
        killed_at_marker(command,lambda e:e.get("event")=="deeper_case_commit_barrier" and e.get("completed")==1,env=environment,timeout=180)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            original=db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall()
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0],"wal")
        self.assertEqual(len(original),1);self.assertEqual(original[0][0],0)
        resumed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=180)
        self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:saved=db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall()
        self.assertEqual([r[0] for r in saved],[0,1]);self.assertEqual(saved[0],original[0])
        repeated=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=180)
        self.assertEqual(repeated.returncode,0,repeated.stdout+repeated.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            self.assertEqual(saved,db.execute("SELECT idx,body,hash FROM cases ORDER BY idx").fetchall())
            wrong=json.loads(saved[0][1]);next(iter(wrong["measurements"]["25"].values()))["reference_regret"]+=1
            db.execute("UPDATE cases SET body=?,hash=? WHERE idx=0",(config.canonical(wrong),config.digest(wrong)));db.commit()
        refused=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=180)
        self.assertNotEqual(refused.returncode,0)
        EVIDENCE["benchmark_recovery"]={"actual_process_kill":True,"last_committed_cases":1,"resumed_cases":2,
            "prior_case_byte_unchanged":True,"idempotent_resume":True,"forged_checksum_valid_regret_refused":True,
            "same_declared_cpu_identity_restored":True,"FULL_WAL_complete_cases":True}

    def test_11_actual_completed_arena_game_kill_resume_on_isolated_backup(self):
        run=self.path/"arena";run.mkdir();production=ROOT/"data/stage5-deeper-benchmark/work.sqlite3"
        with closing(sqlite3.connect(production.as_uri()+"?mode=ro",uri=True)) as source,closing(sqlite3.connect(run/"work.sqlite3")) as copied:
            source.backup(copied)
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM cases").fetchone()[0],391)
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM games").fetchone()[0],288)
            copied.execute("DELETE FROM games");copied.execute("DELETE FROM meta WHERE key='deeper_complete'");copied.commit()
        worker="""import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('audit_deeper_arena_worker',Path.cwd()/'tools/benchmark-stage5-deeper.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prior=m._work.__enter__
def entering(self):
 Path(self.path,'observed-identity.json').write_text(json.dumps(self.identity,indent=2))
 return prior(self)
m._work.__enter__=entering
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='deeper_arena_game_committed':sys.stdin.readline()
m.run(Path(sys.argv[1]),Path(sys.argv[1])/'report.json',on_event=event)
"""
        command=[sys.executable,"-c",worker,str(run)]
        def kill_after(count):
            try:killed_at_marker(command,lambda e:e.get("event")=="deeper_arena_game_committed" and e.get("completed")==count,timeout=180)
            except AssertionError:
                with closing(sqlite3.connect(run/"work.sqlite3")) as db:expected=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0])
                observed=json.loads((run/"observed-identity.json").read_text())
                drift={key:{"stored":expected[key],"current":observed.get(key)} for key in expected if expected[key]!=observed.get(key)}
                (WORK/"arena-identity-drift.json").write_text(json.dumps(drift,indent=2)+"\n");raise
        kill_after(1)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:first=db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall()
        self.assertEqual(len(first),1)
        kill_after(2)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:second=db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall()
        self.assertEqual(len(second),2);self.assertEqual(second[0],first[0])
        validation=[sys.executable,str(ROOT/"tools/benchmark-stage5-deeper.py"),"run","--run",str(run),"--output",str(run/"report.json"),"--no-arena"]
        result=subprocess.run(validation,cwd=ROOT,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:self.assertEqual(second,db.execute("SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat").fetchall())
        for comparison,pair,seat,body,digest in second:
            game=json.loads(body);self.assertEqual(config.digest(game),digest);p=literal_position(game["start"])
            for move in game["moves"]:p=literal.apply(p,move).position
            self.assertEqual(as_dict(p),game["final"]);self.assertTrue(literal.normalize(p).terminal)
        comparison,pair,seat,body,digest=second[0];wrong=json.loads(body);wrong["candidate_final_margin"]+=2
        with closing(sqlite3.connect(run/"work.sqlite3")) as db:
            db.execute("UPDATE games SET body=?,hash=? WHERE comparison=? AND pair=? AND seat=?",(config.canonical(wrong),config.digest(wrong),comparison,pair,seat));db.commit()
        refused=subprocess.run(validation,cwd=ROOT,text=True,capture_output=True,timeout=180);self.assertNotEqual(refused.returncode,0)
        with closing(sqlite3.connect(production.as_uri()+"?mode=ro",uri=True)) as db:self.assertEqual(db.execute("SELECT COUNT(*) FROM games").fetchone()[0],288)
        EVIDENCE["arena_recovery"]={"actual_arena_hard_kill":True,"process_kills":2,"isolated_sqlite_backup":True,
            "production_cases_reused":391,"completed_games_after_resume":2,"first_completed_game_byte_unchanged":True,
            "idempotent_saved_game_validation":True,"both_games_literal_replayed":True,"forged_checksum_valid_margin_refused":True,"production_games_untouched":288}

    def test_12_cuda_final_checkpoint_kill_before_export_resume(self):
        runtime=json.loads((ROOT/"build/training-runtime.json").read_text())["python"];run=self.path/"final-commit-kill";fresh=self.path/"final-commit-fresh"
        worker="""import json,sys
from congklak import training_v7 as tr
def event(e):
 print(json.dumps(e),flush=True)
 if e.get('event')=='checkpoint_committed' and e.get('step')==20:sys.stdin.readline()
tr.train(sys.argv[1],steps=20,seed=73,width=17,batch_size=32,checkpoint_every=10,input_count=43,teacher_depth=6,device='cuda',on_event=event)
"""
        killed_at_marker([runtime,"-c",worker,str(run)],lambda e:e.get("event")=="checkpoint_committed" and e.get("step")==20,timeout=240)
        self.assertFalse((run/"result.json").exists());self.assertFalse((run/"model.json").exists())
        with closing(sqlite3.connect(run/"catalog.sqlite3")) as db:
            before=db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall();self.assertEqual([r[0] for r in before],[10,20])
        script="""import json,sys
from pathlib import Path
from congklak import training_v7 as tr
args=dict(steps=20,seed=73,width=17,batch_size=32,checkpoint_every=10,input_count=43,teacher_depth=6,device='cuda')
a=tr.train(sys.argv[1],**args);b=tr.train(sys.argv[2],**args)
assert a['recovery']['restored_step']==20 and a['status']=='complete' and a['best_step']==b['best_step']
assert Path(sys.argv[1],'model.json').read_bytes()==Path(sys.argv[2],'model.json').read_bytes()
assert a['selected_development']==b['selected_development'] and a['selected_cross_teacher_development']==b['selected_cross_teacher_development']
print(json.dumps(dict(actual_final_commit_kill=True,committed_step=20,missing_export_and_result_regenerated=True,selected_export_bytes_equal_fresh=True,no_extra_optimizer_or_selection_step=True)))
"""
        result=subprocess.run([runtime,"-c",script,str(run),str(fresh)],cwd=ROOT,text=True,capture_output=True,timeout=600)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/"catalog.sqlite3")) as db:self.assertEqual(before,db.execute("SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id").fetchall())
        EVIDENCE["final_checkpoint_recovery"]=json.loads(result.stdout.strip().splitlines()[-1])

    def test_13_real_migration_isolated_elapsed_drift_and_corrupt_source_refusal(self):
        from congklak import deeper_teacher_v2 as teacher,deeper_teacher as legacy
        old=self.path/"legacy";old.mkdir();destination=self.path/"migrated"
        production=ROOT/"data/deeper-9000-v1/work.sqlite3";production_sha=sha(production)
        with closing(sqlite3.connect(production.as_uri()+"?mode=ro",uri=True)) as source,closing(sqlite3.connect(old/"work.sqlite3")) as copied:source.backup(copied)
        prior_snapshot=teacher.migration_snapshot
        def snapshot(source=None):return prior_snapshot(old if source is None else source)
        with mock.patch.object(legacy,"DEFAULT",old),mock.patch.object(teacher,"migration_snapshot",snapshot):
            migrated=teacher.migrate(old,destination)
            self.assertEqual(migrated["copied"],dict(roots=9000,features=3473,labels=3472));self.assertEqual(migrated["adjusted_timing_rows"],543)
            with closing(sqlite3.connect(old/"work.sqlite3")) as db:
                key,original_body,original_digest=db.execute("SELECT state,body,hash FROM labels ORDER BY state LIMIT 1").fetchone()
                identity_body,identity_digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
            original=json.loads(original_body)
            def reset_destination_metadata():
                with closing(sqlite3.connect(destination/"work.sqlite3")) as db:
                    db.execute("DELETE FROM meta WHERE key IN ('identity','migration')");db.commit()
                publication=destination/"migration.json"
                self.assertTrue(publication.resolve().is_relative_to(self.path));publication.unlink(missing_ok=True)
            def install_label(item):
                with closing(sqlite3.connect(old/"work.sqlite3")) as db:
                    db.execute("UPDATE labels SET body=?,hash=? WHERE state=?",(config.canonical(item),config.digest(item),key));db.commit()
            positive=json.loads(original_body);positive["label"]["work"]["elapsed_ms"]=positive["report"]["elapsed_ms"]+3
            install_label(positive);reset_destination_metadata();accepted=teacher.migrate(old,destination)
            observation=next(c for c in accepted["changes"] if c["state_key"]==key)
            self.assertEqual(observation["return_elapsed_ms"]-observation["report_elapsed_ms"],3)
            with closing(sqlite3.connect(destination/"work.sqlite3")) as db:
                changed=json.loads(db.execute("SELECT body FROM labels WHERE state=?",(key,)).fetchone()[0])
            self.assertEqual(changed["report"],original["report"])
            self.assertEqual({k:v for k,v in changed["label"].items() if k!="work"},{k:v for k,v in original["label"].items() if k!="work"})
            invalid_nodes=json.loads(original_body);invalid_nodes["label"]["work"]["nodes"]+=1
            install_label(invalid_nodes);reset_destination_metadata()
            with self.assertRaisesRegex(teacher.DeeperError,"drift"):teacher.migrate(old,destination)
            invalid_score=json.loads(original_body);invalid_score["label"]["action_scores"][0]["value_player0"]+=1
            install_label(invalid_score);reset_destination_metadata()
            with self.assertRaises(teacher.DeeperError):teacher.migrate(old,destination)
            install_label(original);reset_destination_metadata();forged_identity=json.loads(identity_body);forged_identity["choice_depth"]=7
            with closing(sqlite3.connect(old/"work.sqlite3")) as db:
                db.execute("UPDATE meta SET body=?,hash=? WHERE key='identity'",(config.canonical(forged_identity),config.digest(forged_identity)));db.commit()
            with self.assertRaisesRegex(teacher.DeeperError,"generation identity"):teacher.migrate(old,destination)
        self.assertEqual(sha(production),production_sha)
        EVIDENCE["migration_regression"]={"isolated_full_function_migration":True,"copied_original_roots":9000,"copied_original_labels":3472,
            "copied_original_features":3473,"original_timing_corrections":543,"injected_positive_return_elapsed_delta_preserved":3,
            "roots_reports_scores_unmodified":True,"checksum_valid_non_timing_work_score_source_identity_refused":True,"original_production_ledger_byte_unchanged":True}


if __name__=="__main__":
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentDeeperAcceptance))
    EVIDENCE.update(source_sha256=sha(__file__),tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),duration_seconds=time.monotonic()-start)
    (WORK/"tests.json").write_text(json.dumps(EVIDENCE,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
