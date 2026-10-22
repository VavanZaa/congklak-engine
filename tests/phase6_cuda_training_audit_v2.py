# Versioned private batch32 CUDA audit. Frozen runtime and production settings unchanged.
from phase6_audit_acceptance import *

class Phase6CudaTrainingV2Acceptance(Phase6IndependentAcceptance):
    def cuda_recovery(self,stage):
        root=self.fresh('cuda-'+stage)
        args=dict(stage=stage,steps=6,batch_size=32,checkpoint_every=2,seed=187 if stage=='warm' else 197,device='cuda')
        if stage=='selfplay':
            # The generator manifest must describe exactly this completed warm
            # model. Production replay is preferred; fixtures remain scoped.
            source=Path(os.environ.get('PHASE6_AUDIT_SP_PATH',str(common.SELFPLAY)))
            warm=Path(os.environ.get('PHASE6_AUDIT_WARM_MODEL',str(common.WARM)))
            manifest,rows=sp.load_training_rows(source);self.assertTrue(rows)
            args.update(initial_model=str(warm),selfplay_path=str(source))
        code="from congklak import phase6_training as tr;from pathlib import Path;import json,sys;tr.train(Path(sys.argv[1]),**json.loads(sys.argv[2]))"
        def invoke(path,refuse=False):
            result=subprocess.run([sys.executable,'-c',code,str(path),json.dumps(args)],cwd=ROOT,capture_output=True,text=True)
            (root/('last-refusal.log' if refuse else 'last-worker.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            self.assertEqual(result.returncode==0,not refuse,result.stdout+result.stderr)
            return load(path/'result.json') if not refuse else None
        fresh=root/'fresh';reference=invoke(fresh);records=[]
        for event in ('checkpoint_file_ready','training_final_checkpoint_ready'):
            run=root/event
            # Block second checkpoint before catalog commit, or final committed
            # checkpoint before export. A prior complete unit always exists.
            custom="import sys,json,time;from pathlib import Path;from congklak import phase6_training as tr\ndef cb(v):\n if v['event']==sys.argv[2] and (sys.argv[2]!='checkpoint_file_ready' or v['step']==4):\n  Path(sys.argv[1]).write_text(json.dumps(v));\n  while True:time.sleep(.1)\ntr.train(Path(sys.argv[3]),**json.loads(sys.argv[4]),on_event=cb)"
            probe=root/(event+'-probe');probe.mkdir();marker=probe/'marker.json';log=probe/'worker.log'
            with log.open('w',encoding='utf-8') as out:
                child=subprocess.Popen([sys.executable,'-c',custom,str(marker),event,str(run),json.dumps(args)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
                try:
                    deadline=time.monotonic()+180
                    while not marker.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
                    self.assertTrue(marker.exists(),log.read_text(encoding='utf-8'));child.kill();child.wait(timeout=10)
                finally:
                    if child.poll() is None:child.kill();child.wait(timeout=10)
            with closing(sqlite3.connect(run/'catalog.sqlite3')) as db:committed=db.execute('SELECT MAX(step) FROM checkpoints').fetchone()[0]
            self.assertEqual(committed,2 if event=='checkpoint_file_ready' else 6)
            self.assertFalse((run/'model.json').exists());restored=invoke(run)
            self.assertEqual(restored['recovery']['restored_step'],committed);self.assertEqual(restored['weights_sha256'],reference['weights_sha256'])
            self.assertEqual((fresh/'model.json').read_bytes(),(run/'model.json').read_bytes())
            compare="""import sys,sqlite3,torch
from pathlib import Path
def last(p):
 db=sqlite3.connect(p/'catalog.sqlite3');name=db.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0];db.close();return torch.load(p/name,map_location='cpu',weights_only=True)
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
a=last(Path(sys.argv[1]));b=last(Path(sys.argv[2]));assert eq(a,b),'Optimizer/sampler/Python/CPU/allCUDA/model/best state differs';assert len(b['optimizer']['state'])==int(sys.argv[3]);assert b['teacher_draws']==(192 if sys.argv[3]=='2' else 96);assert b['selfplay_draws']==(0 if sys.argv[3]=='2' else 96)
"""
            checked=subprocess.run([sys.executable,'-c',compare,str(fresh),str(run),'2' if stage=='warm' else '8'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            stable=snapshot(run);self.assertEqual(invoke(run),restored);self.assertEqual(snapshot(run),stable)
            original=(run/'model.json').read_bytes();forged=json.loads(original);forged['model']['uses_oracle']=True;forged['sha256']=config.digest(forged['model']);(run/'model.json').write_text(config.canonical(forged))
            bad=snapshot(run);invoke(run,True);self.assertEqual(snapshot(run),bad);(run/'model.json').write_bytes(original)
            records.append(dict(event=event,actual_cuda_hard_kill=True,restored_step=committed,whole_state_bitwise_equal=True,finished_bytes_mtime_stable=True,checksum_valid_export_refused=True))
        EVIDENCE['cuda_'+stage]=dict(checks=records,steps=6,batch_size=32,teacher_draws=192 if stage=='warm' else 96,selfplay_draws=0 if stage=='warm' else 96)
