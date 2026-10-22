"""Four current owner methods. All searches use old consumed build fixtures."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,allocation_trace as trace
from congklak.neural_relay import NeuralNative


class AllocationTraceAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trace.common.baseline(verify_all=True);cls.retained=ROOT/'build/allocation-trace-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True);cls.normal=cls.retained/'normal'

    def clone(self,name):
        target=self.retained/name;self.assertTrue(target.resolve().is_relative_to((ROOT/'build').resolve()));self.assertFalse(target.exists());shutil.copytree(self.normal,target);return target

    def test_01_panel_score_units_FPU_ties_and_fixed_player0_backup(self):
        plan,roots,units,identity=trace.admit();self.assertEqual(len(units),8);self.assertEqual(sum(u['simulations'] for u in units),3840);self.assertEqual(roots[0]['pair'],7);self.assertEqual(roots[0]['actor'],1)
        self.assertEqual(len(plan['references']),8);self.assertEqual(identity['policy_contract']['frozen_parameters'],12807);self.assertEqual(plan['run_quota_bytes'],1024**3)
        state=roots[0]['state'];node=trace.Node(0,state,1,(0,1),False,-.5,priors=[.5,.5,0.,0.,0.,0.,0.]);details=trace.selection_details(node)
        self.assertEqual(details['chosen'],0)
        for row in details['rows']:
            self.assertIsNone(row['observed_Q_player0_normalized']);self.assertEqual(row['Q_source'],'parent_bootstrap_FPU');self.assertEqual(row['total_score_normalized'],.625);self.assertEqual(row['total_score_seed_display'],.625*98)
        node.visits[:2]=[1,1];node.sums[:2]=[-.25,-.75];self.assertEqual(trace.selection_details(node)['chosen'],1)
        child=trace.Node(1,state,1,(0,1),False,0.);tree=trace.Tree({0:node,1:child});before=node.sums[0];trace._backup(tree,[[0,1,0],[1,1,1]],-.3)
        self.assertEqual(node.sums[0],before-.3);self.assertEqual(child.sums[1],-.3);self.assertEqual(tree.completed,1)
        child.actor=0;trace._backup(tree,[[0,1,0],[1,0,0]],.2);self.assertEqual(child.sums[0],.2)
        self.assertFalse(trace.same({'rng':False},{'rng':0}));self.assertFalse(trace.same({'simulation':1.0},{'simulation':1}))
        with self.assertRaises(ValueError):trace.run(ROOT/'data/allocation-illegal-fixture',fixture=True)

    def test_02_exact_committed_prefix_restore_and_untouched_engine_parity(self):
        partial=trace.run(self.normal,fixture=True,simulation_limit=3,on_event=lambda event:None);self.assertEqual(partial['committed_simulations'],3)
        with trace.closing(trace._ro(self.normal/'work.sqlite3')) as db:
            prefix=db.execute('SELECT cell,seq,body,hash FROM steps ORDER BY cell,seq').fetchall();setup=db.execute('SELECT cell,body,hash FROM root_setups').fetchall()
        full=trace.run(self.normal,fixture=True,on_event=lambda event:None);self.assertTrue(full['scope_finished']);self.assertEqual(full['counts']['committed_root_simulations'],96)
        with trace.closing(trace._ro(self.normal/'work.sqlite3')) as db:
            self.assertEqual(prefix,db.execute('SELECT cell,seq,body,hash FROM steps ORDER BY cell,seq LIMIT 3').fetchall());self.assertEqual(setup,db.execute('SELECT cell,body,hash FROM root_setups WHERE cell=0').fetchall())
        fresh=self.retained/'fresh';trace.run(fresh,fixture=True,on_event=lambda event:None);roots,stores,manifest=trace.load_trace(self.normal,fixture=True);_,freshstores,_=trace.load_trace(fresh,fixture=True)
        self.assertEqual(trace.semantic_trace(stores),trace.semantic_trace(freshstores))
        plan,roots,units,identity=trace.admit(True)
        with trace.PerformanceCore(plan['cpu_declared']):
            for unit in units:
                model=trace.common.CONTROL if unit['profile']=='control' else trace.common.CANDIDATE
                with trace.CpuBatchEvaluator(model,cache_capacity=0,max_batch=1,backend='auto') as evaluator:
                    evaluator.reset_cache();raw=trace.quality_search.search(roots[unit['root_index']]['state'],evaluator,simulations=unit['simulations'],time_ms=0,max_nodes=unit['max_nodes'],rng=0,root_noise=None,c_puct=.25,memory_mb=8192)
                raw.pop('timing');raw['work'].pop('inference_ms');self.assertEqual(raw,stores['cells'][unit['index']]['semantic_report'])
        self.assertTrue(all(len(row['path'])>0 for rows in stores['steps'].values() for row in rows));self.assertEqual(full['counts']['physical_committed_counts']['cache_resets'],8)

    def test_03_pre_root_and_mutated_simulation_cancel_caps_and_corruption(self):
        original=trace.CpuBatchEvaluator.evaluate_many;event=threading.Event()
        def cancel_root(evaluator,positions,*args,**kwargs):
            rows=original(evaluator,positions,*args,**kwargs);event.set();return rows
        rootpath=self.retained/'root-cancel'
        with patch.object(trace.CpuBatchEvaluator,'evaluate_many',cancel_root):paused=trace.run(rootpath,fixture=True,cancel=event,on_event=lambda e:None)
        self.assertEqual(paused['root_setups'],0);self.assertEqual(paused['committed_simulations'],0)
        with trace.closing(trace._ro(rootpath/'work.sqlite3')) as db:
            row=json.loads(db.execute('SELECT body FROM interruptions').fetchone()[0]);self.assertEqual(row['phase'],'root_setup');self.assertEqual(row['physical_counts']['native_actual_evals'],1)
        resumedroot=trace.run(rootpath,fixture=True,on_event=lambda e:None);self.assertTrue(resumedroot['scope_finished'])
        event.clear();calls=0
        def cancel_simulation(evaluator,positions,*args,**kwargs):
            nonlocal calls
            rows=original(evaluator,positions,*args,**kwargs);calls+=1
            if calls==2:event.set()
            return rows
        simpath=self.retained/'simulation-cancel'
        with patch.object(trace.CpuBatchEvaluator,'evaluate_many',cancel_simulation):paused=trace.run(simpath,fixture=True,cancel=event,on_event=lambda e:None)
        self.assertEqual(paused['root_setups'],1);self.assertEqual(paused['committed_simulations'],0)
        with trace.closing(trace._ro(simpath/'work.sqlite3')) as db:
            row=json.loads(db.execute('SELECT body FROM interruptions').fetchone()[0]);self.assertEqual(row['phase'],'simulation');self.assertIsNone(row['uncommitted_native_transition_work']);self.assertEqual(row['physical_counts']['native_actual_evals'],1)
        seen=[]
        def observe(evaluator,positions,*args,**kwargs):seen.extend(copy.deepcopy(positions));return original(evaluator,positions,*args,**kwargs)
        with patch.object(trace.CpuBatchEvaluator,'evaluate_many',observe):resumedsim=trace.run(simpath,fixture=True,on_event=lambda e:None)
        roots,normalstores,normal=trace.load_trace(self.normal,fixture=True);self.assertEqual(seen[0],normalstores['steps'][0][0]['leaf']['state'])
        self.assertEqual([c['semantic_report'] for c in resumedsim['cells']],[c['semantic_report'] for c in normal['cells']])
        capped=trace.run(self.retained/'node-cap',fixture=True,variant='node_cap',on_event=lambda e:None);self.assertFalse(capped['scope_finished']);self.assertGreater(capped['counts']['incomplete_cells'],0)
        plan,roots,units,identity=trace.admit(True)
        for kind in ('score','actor_alias','zero_feature_work'):
            tree=trace._root_setup(normalstores['root_setups'][0],units[0],roots[0],identity,normalstores['startup'],trace.Native());body=copy.deepcopy(normalstores['steps'][0][0])
            if kind=='score':body['root_selection']['rows'][0]['total_score_normalized']+=.01
            elif kind=='actor_alias':body['path'][0][1]=bool(body['path'][0][1])
            else:body['physical_counts']['feature_simulations']=0;body['prediction']['feature_simulations']=0
            with self.assertRaises(ValueError):trace.replay_step(tree,body,units[0],identity)
        damaged=self.clone('bad-marker');(damaged/'result.json').unlink()
        with trace.sqlite3.connect(damaged/'work.sqlite3') as db:
            marker=json.loads(db.execute("SELECT body FROM meta WHERE key='allocation_trace_complete'").fetchone()[0]);marker['stores_sha256']='0'*64;db.execute("UPDATE meta SET body=?,hash=? WHERE key='allocation_trace_complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):trace.run(damaged,fixture=True)
        self.assertFalse((damaged/'result.json').exists())

    def test_04_closed_API_CLI_no_inference_and_validated_export_repair(self):
        def snapshot():
            result={}
            for file in self.normal.rglob('*'):
                if not file.is_file() or file.name.endswith('-shm') or (file.name.endswith('-wal') and file.stat().st_size==0):continue
                result[str(file.relative_to(self.normal))]=dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),size=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns)
            return result
        before=snapshot()
        def forbidden(*args,**kwargs):raise AssertionError('Completed trace activated inference/search')
        with patch.object(trace,'CpuBatchEvaluator',side_effect=forbidden),patch.object(trace,'PerformanceCore',side_effect=forbidden),patch.object(trace.Model,'reference',side_effect=forbidden),patch.object(trace.quality_search,'search',side_effect=forbidden),patch.object(trace.Native,'analyze',side_effect=forbidden),patch.object(NeuralNative,'analyze',side_effect=forbidden):
            trace.run(self.normal,fixture=True);trace.verify(self.normal,fixture=True);trace.load_trace(self.normal,fixture=True)
            repaired=self.clone('missing-exports')
            for name in ('manifest.json','result.json','resources.json'):(repaired/name).unlink()
            trace.run(repaired,fixture=True)
        self.assertEqual(before,snapshot());(self.retained/'closed-durable-proof.json').write_text(config.canonical(dict(before=before,after=snapshot(),unchanged=True,policy='Include main DB, locks, outputs/resources and nonempty WAL; SHM and absent/empty WAL are ephemeral.'))+'\n',encoding='utf-8')
        worker="import runpy,sys;from congklak import allocation_trace as t;from congklak.neural_relay import NeuralNative\ndef no(*a,**k):raise AssertionError('new inference/search')\nt.CpuBatchEvaluator=no;t.PerformanceCore=no;t.Model.reference=no;t.Native.analyze=no;NeuralNative.analyze=no;t.quality_search.search=no;sys.argv="+repr([str(trace.CLI),'verify','--fixture','--path',str(self.normal)])+";runpy.run_path(str(t.CLI),run_name='__main__')"
        process=subprocess.run([sys.executable,'-c',worker],cwd=ROOT,text=True,capture_output=True);(self.retained/'closed-cli.stdout.log').write_text(process.stdout,encoding='utf-8');(self.retained/'closed-cli.stderr.log').write_text(process.stderr,encoding='utf-8');self.assertEqual(process.returncode,0,process.stderr)
        self.assertNotIn('torch',sys.modules);trace.common.baseline(verify_all=True)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(AllocationTraceAuthor);covered=[test.id() for test in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths=[Path(__file__),Path(trace.__file__),trace.CLI,trace.PLAN,trace.IMPLEMENTATION,trace.common.BASELINE,trace.common.SCOPE,ROOT/'congklak/allocation_common.py']
    paths.extend(AllocationTraceAuthor.retained.glob('closed-*'))
    report=dict(schema='allocation-trace-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=covered,methods=unittest.defaultTestLoader.getTestCaseNames(AllocationTraceAuthor),inputs=trace.common.pins(paths),retained_directory=AllocationTraceAuthor.retained.relative_to(ROOT).as_posix(),duration_seconds=time.perf_counter()-started)
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/'build/allocation-trace-author.json',report)
    raise SystemExit(not result.wasSuccessful())
