"""Four owner checks of frozen fixed-work cells; fixtures are not production."""
from pathlib import Path
import copy
import json
import shutil
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config, divergence_search as search
from congklak.native import NativeError
from congklak.neural_relay import NeuralNative


class DivergenceSearchAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        search.common.baseline(verify_all=True)
        cls.retained=ROOT/'build/divergence-search-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True)
        cls.normal=cls.retained/'normal';cls.events=[]

    def clone(self,name):
        target=self.retained/name;self.assertTrue(target.resolve().is_relative_to((ROOT/'build').resolve()));self.assertFalse(target.exists());shutil.copytree(self.normal,target);return target

    def test_01_five_literal_roots_grid_policy_and_type_identity(self):
        plan,roots,units,identity=search.admit();self.assertEqual([r['pair'] for r in roots],[2,4,6,7,8]);self.assertEqual([r['active_seeds'] for r in roots],[10,16,36,63,27])
        self.assertEqual([r['timed_margin_delta'] for r in roots],[0,-6,-4,-6,0]);self.assertEqual(len(units),40);self.assertEqual(sum(x['simulations'] for x in units),19200)
        self.assertTrue(identity['policy_contract']['global_policy_unchanged_by_identical_subgraph']);self.assertEqual(identity['policy_contract']['frozen_parameters'],12807)
        for row in roots:self.assertEqual(row['state_key'],search.common.rawkey(row['state']));self.assertEqual(row['actor'],1)
        self.assertEqual(plan['max_nodes'],2048);self.assertEqual(plan['cpu_declared']['logical'],2)
        self.assertFalse(search._same({'rng':False},{'rng':0}));self.assertFalse(search._same({'index':1.0},{'index':1}))
        with self.assertRaises(ValueError):search.run(ROOT/'data/divergence-search-invalid-fixture',fixture=True)

    def test_02_real_fresh_work_prefix_conservation_and_repeatability(self):
        partial=search.run(self.normal,fixture=True,cell_limit=3,on_event=self.events.append);self.assertEqual(partial['committed_cells'],3)
        with search.closing(search._ro(self.normal/'work.sqlite3')) as db:prefix=db.execute('SELECT idx,body,hash FROM cells ORDER BY idx').fetchall()
        full=search.run(self.normal,fixture=True,on_event=self.events.append);self.assertEqual(full['counts']['complete_cells'],8);self.assertEqual(full['counts']['completed_simulations'],96)
        with search.closing(search._ro(self.normal/'work.sqlite3')) as db:self.assertEqual(prefix,db.execute('SELECT idx,body,hash FROM cells ORDER BY idx LIMIT 3').fetchall())
        plan,roots,units,identity=search.admit(True)
        with search.PerformanceCore(plan['cpu_declared']):
            for cell in full['cells'][:2]:
                profile=cell['unit']['profile'];model=search.common.CONTROL if profile=='control' else search.common.CANDIDATE
                with search.CpuBatchEvaluator(model,cache_capacity=0,max_batch=1,backend='auto') as evaluator:
                    evaluator.reset_cache();raw=search.quality_search.search(roots[0]['state'],evaluator,simulations=8,time_ms=0,max_nodes=2048,rng=0,root_noise=None,c_puct=.25,memory_mb=8192)
                    self.assertEqual(search.semantic_report(raw),cell['semantic_report'])
        for cell in full['cells']:
            self.assertEqual(sum(cell['report']['visits']),cell['unit']['simulations']);self.assertEqual(cell['counts']['native_actual_evals'],cell['report']['work']['neural_evaluations']);self.assertGreaterEqual(cell['counts']['feature_simulations'],cell['counts']['native_actual_evals'])
        self.assertTrue(all(row['root_priors_identical'] for row in full['comparisons']))
        (self.retained/'events.json').write_text(config.canonical(self.events)+'\n',encoding='utf-8')

    def test_03_typed_cap_cancel_and_rehashed_corruption_refusal(self):
        capped=search.run(self.retained/'node-cap',fixture=True,variant='node_cap',on_event=lambda event:None)
        self.assertGreater(capped['counts']['capped_cells'],0);self.assertFalse(capped['scope_finished'])
        calls=0
        def cancel():
            nonlocal calls
            calls+=1;return calls>=9
        paused=search.run(self.retained/'cancel',fixture=True,cancel=cancel,on_event=lambda event:None);self.assertEqual(paused['status'],'interrupted');self.assertEqual(paused['committed_cells'],0)
        with search.closing(search._ro(self.retained/'cancel/work.sqlite3')) as db:
            attempts=[json.loads(text) for text, in db.execute('SELECT body FROM attempts')];self.assertEqual(len(attempts),1);self.assertEqual(attempts[0]['status'],'cancelled')
        resumed=search.run(self.retained/'cancel',fixture=True,on_event=lambda event:None);normal=search.verify(self.normal,fixture=True);self.assertEqual(resumed['semantic_cells_sha256'],normal['semantic_cells_sha256'])
        plan,roots,units,identity=search.admit(True)
        with search.closing(search._ro(self.normal/'work.sqlite3')) as db:setup=search._hashrow(*db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone())
        for kind in ('unit_float_alias','seq_float_alias','zero_physical_features'):
            body=copy.deepcopy(normal['cells'][0])
            if kind=='unit_float_alias':body['unit']['simulations']=8.0
            elif kind=='seq_float_alias':body['seq']=0.0
            else:
                body['counts']['feature_simulations']=body['counts']['feature_deposits']=0
                for stage in ('evaluator_after',):
                    for key in ('feature_simulations','feature_deposits'):body[stage][key]=body['evaluator_before'][key]
                body['report']['work']['feature_simulations']=body['report']['work']['feature_deposits']=0;body['report']['work']['total_deposits']=body['report']['work']['deposits']
                body['report_sha256']=config.digest(body['report']);body['semantic_report']=search.semantic_report(body['report']);body['semantic_sha256']=config.digest(body['semantic_report'])
            with self.assertRaises((ValueError,NativeError)):search.validate_attempt(body,0,units[0],roots[0],identity,setup)
        damaged=self.clone('bad-complete');(damaged/'result.json').unlink()
        with search.sqlite3.connect(damaged/'work.sqlite3') as db:
            marker=json.loads(db.execute("SELECT body FROM meta WHERE key='divergence_search_complete'").fetchone()[0]);marker['cells_sha256']='0'*64
            db.execute("UPDATE meta SET body=?,hash=? WHERE key='divergence_search_complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):search.run(damaged,fixture=True)
        self.assertFalse((damaged/'result.json').exists())

    def test_04_completed_CPU_API_CLI_and_missing_export_repair_no_search(self):
        def durable_snapshot():
            result={}
            for path in self.normal.rglob('*'):
                if not path.is_file() or path.name.endswith('-shm'):continue
                # An empty/absent WAL has no committed pages. A nonempty WAL is
                # durable evidence and MUST remain in the byte/mtime snapshot.
                if path.name.endswith('-wal') and path.stat().st_size==0:continue
                result[str(path.relative_to(self.normal))]=(path.read_bytes(),path.stat().st_mtime_ns)
            return result
        before=durable_snapshot()
        def forbidden(*args,**kwargs):raise AssertionError('Completed reader attempted new engine work')
        with patch.object(search,'CpuBatchEvaluator',side_effect=forbidden),patch.object(search.quality_search,'search',side_effect=forbidden),patch.object(search.Native,'analyze',side_effect=forbidden),patch.object(NeuralNative,'analyze',side_effect=forbidden):
            search.run(self.normal,fixture=True);search.verify(self.normal,fixture=True);search.load_cells(self.normal,fixture=True)
            repaired=self.clone('missing-exports')
            for name in ('manifest.json','result.json','resources.json'):(repaired/name).unlink()
            search.run(repaired,fixture=True)
        self.assertEqual(before,durable_snapshot())
        wal=self.normal/'work.sqlite3-wal'
        (self.retained/'closed-stability.json').write_text(config.canonical(dict(durable_files=list(before),unchanged_bytes_mtimes=True,wal_present=wal.exists(),wal_size=wal.stat().st_size if wal.exists() else 0,
            policy='SHM is an ephemeral read cache. Only absent/empty completed WAL is excluded; any nonempty WAL remains durable evidence. MainDB, lock, manifest, results and resource records remain exact.'))+'\n',encoding='utf-8')
        worker="import runpy,sys;from congklak import divergence_search as s;from congklak.neural_relay import NeuralNative\ndef no(*a,**k):raise AssertionError('new engine work')\ns.CpuBatchEvaluator=no;s.quality_search.search=no;s.Native.analyze=no;NeuralNative.analyze=no;sys.argv="+repr([str(search.CLI),'verify','--fixture','--path',str(self.normal)])+";runpy.run_path(str(s.CLI),run_name='__main__')"
        result=subprocess.run([sys.executable,'-c',worker],cwd=ROOT,text=True,capture_output=True)
        (self.retained/'closed-cli.stdout.log').write_text(result.stdout,encoding='utf-8');(self.retained/'closed-cli.stderr.log').write_text(result.stderr,encoding='utf-8');self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('torch',sys.modules);search.common.baseline(verify_all=True)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(DivergenceSearchAuthor);methods=[test.id() for test in suite];result=unittest.TextTestRunner(verbosity=2).run(suite)
    paths=(Path(__file__),Path(search.__file__),search.CLI,search.PLAN,search.IMPLEMENTATION,search.common.BASELINE,search.common.PANEL,search.common.SCOPE,ROOT/'congklak/divergence_study_common.py')
    logs=[path for path in DivergenceSearchAuthor.retained.rglob('*.log') if path.is_file()]
    body=dict(schema='divergence-search-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=methods,methods=unittest.defaultTestLoader.getTestCaseNames(DivergenceSearchAuthor),inputs=search.common.pins((*paths,*logs)),retained_directory=DivergenceSearchAuthor.retained.relative_to(ROOT).as_posix(),duration_seconds=time.perf_counter()-started,closed_cli_logs=search.common.pins(logs))
    from congklak.deeper_teacher_v2 import atomic_json
    atomic_json(ROOT/'build/divergence-search-author.json',body)
    raise SystemExit(not result.wasSuccessful())
