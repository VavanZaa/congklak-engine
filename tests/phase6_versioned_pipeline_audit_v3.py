# Versioned pipeline audit for deferred-CUDA update and guarded benchmark.
from phase6_audit_acceptance import *
import uuid

class Phase6VersionedPipelineAuditV3(Phase6IndependentAcceptance):
    @classmethod
    def setUpClass(cls):
        # Keep the actual fresh/resumed states and kill markers for later
        # independent reinspection. These are owned audit fixture directories.
        cls.base=WORK/('versioned-pipeline-'+uuid.uuid4().hex)
        cls.base.mkdir()
        assert cls.base.resolve().is_relative_to(WORK.resolve())
    @classmethod
    def tearDownClass(cls):
        pass
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
        code="from congklak import phase6_training_v2 as tr;from pathlib import Path;import json,sys;tr.train(Path(sys.argv[1]),**json.loads(sys.argv[2]))"
        worker_index=0
        def invoke(path,refuse=False):
            nonlocal worker_index
            worker_index+=1
            result=subprocess.run([sys.executable,'-c',code,str(path),json.dumps(args)],cwd=ROOT,capture_output=True,text=True)
            (root/(f'worker-{worker_index:02d}-'+('refusal' if refuse else 'training')+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            self.assertEqual(result.returncode==0,not refuse,result.stdout+result.stderr)
            return load(path/'result.json') if not refuse else None
        fresh=root/'fresh';reference=invoke(fresh);records=[]
        for event in ('checkpoint_file_ready','training_final_checkpoint_ready'):
            run=root/event
            # Block second checkpoint before catalog commit, or final committed
            # checkpoint before export. A prior complete unit always exists.
            custom="import sys,json,time;from pathlib import Path;from congklak import phase6_training_v2 as tr\ndef cb(v):\n if v['event']==sys.argv[2] and (sys.argv[2]!='checkpoint_file_ready' or v['step']==4):\n  Path(sys.argv[1]).write_text(json.dumps(v));\n  while True:time.sleep(.1)\ntr.train(Path(sys.argv[3]),**json.loads(sys.argv[4]),on_event=cb)"
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
        EVIDENCE['cuda_'+stage]=dict(checks=records,steps=6,batch_size=32,teacher_draws=192 if stage=='warm' else 96,selfplay_draws=0 if stage=='warm' else 96,
                                   raw_fresh_and_resumed_states_retained=True,fixture_directory=str(root.relative_to(ROOT)).replace('\\','/'))
    def test_11_actual_benchmark_case_kill_and_forged_reference_refusal(self):
        root=self.fresh('benchmark-case');run=root/'run';output=root/'result.json'
        code="import importlib.util\nspec=importlib.util.spec_from_file_location('b','tools/benchmark-phase6-v2.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\nb.run(sys.argv[3],sys.argv[4],case_limit=2,no_arena=True,on_event=callback)"
        marker=kill_worker(code,[run,output],root/'probe','phase6_case_committed')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:before=db.execute('SELECT idx,body,hash FROM cases').fetchall()
        self.assertEqual(len(before),1)
        args=[sys.executable,str(ROOT/'tools/benchmark-phase6-v2.py'),'run','--run',str(run),'--output',str(output),'--case-limit','2','--no-arena']
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall();self.assertEqual(after[0],before[0]);self.assertEqual(len(after),2)
            with closing(sqlite3.connect(root/'validated-work.sqlite3')) as validated:db.backup(validated)
            idx,body,digest=after[0];forged=json.loads(body);ms=next(iter(forged['measurements']));name=next(iter(forged['measurements'][ms]));forged['measurements'][ms][name]['reference_regret']=999
            with db:db.execute('UPDATE cases SET body=?,hash=? WHERE idx=?',(config.canonical(forged),config.digest(forged),idx))
        bad=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0);self.assertIn('regret',bad.stdout.lower())
        EVIDENCE['benchmark_case_kill']=dict(actual_hard_kill=True,committed_cases_retained=1,complete_after_resume=2,forged_reference_refused=True,barrier=marker,
                                          valid_journal_backup_retained=True,fixture_directory=str(root.relative_to(ROOT)).replace('\\','/'))
    def test_12_actual_completed_arena_kill_and_literal_replay(self):
        root=self.fresh('benchmark-game');run=root/'run';run.mkdir();output=root/'result.json'
        with closing(sqlite3.connect(ROOT/'data/phase6-benchmark-v1/work.sqlite3')) as source,closing(sqlite3.connect(run/'work.sqlite3')) as dest:source.backup(dest)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM games').fetchone()[0],192)
            keys=db.execute('SELECT comparison,pair,seat FROM games ORDER BY comparison,pair,seat DESC LIMIT 2').fetchall()
            with db:
                for key in keys:db.execute('DELETE FROM games WHERE comparison=? AND pair=? AND seat=?',key)
                db.execute("DELETE FROM meta WHERE key='phase6_benchmark_complete'")
        code="import importlib.util\nspec=importlib.util.spec_from_file_location('b','tools/benchmark-phase6-v2.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\nb.run(sys.argv[3],sys.argv[4],on_event=callback)"
        marker=kill_worker(code,[run,output],root/'probe','phase6_arena_game_committed')
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:prefix=db.execute('SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat').fetchall()
        self.assertEqual(len(prefix),191)
        args=[sys.executable,str(ROOT/'tools/benchmark-phase6-v2.py'),'run','--run',str(run),'--output',str(output)]
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            after=db.execute('SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat').fetchall();self.assertEqual(len(after),192)
            index={(c,p,s):(b,h) for c,p,s,b,h in after}
            for c,p,s,b,h in prefix:self.assertEqual(index[c,p,s],(b,h))
            for c,p,s,b,h in after:
                g=json.loads(b);q=normalize_literal(g['start'])
                for action in g['moves']:q=literal.apply(q,action).position
                self.assertEqual(json_position(q),g['final'])
        state=snapshot(run);saved=load(output);again=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertEqual(again.returncode,0,again.stdout+again.stderr);self.assertEqual(snapshot(run),state);self.assertEqual(load(output),saved)
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            with closing(sqlite3.connect(root/'validated-work.sqlite3')) as validated:db.backup(validated)
            c,p,s,b,h=after[0];forged=json.loads(b);forged['candidate_final_margin']+=2
            with db:db.execute('UPDATE games SET body=?,hash=? WHERE comparison=? AND pair=? AND seat=?',(config.canonical(forged),config.digest(forged),c,p,s))
        bad=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);self.assertNotEqual(bad.returncode,0)
        EVIDENCE['benchmark_game_kill']=dict(actual_hard_kill=True,committed_games_retained=191,complete_after_resume=192,all_games_literal_replayed=True,finished_bytes_mtime_stable=True,forged_outcome_refused=True,barrier=marker,
                                          valid_journal_backup_retained=True,fixture_directory=str(root.relative_to(ROOT)).replace('\\','/'))
