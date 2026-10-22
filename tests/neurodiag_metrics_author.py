"""Four source-frozen numerical/contract checks on synthetic engineering units."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import neurodiag_metrics as m

FREEZE=ROOT/'build/neurodiag-metrics-implementation.json'
SOURCES=('congklak/neurodiag_metrics.py','tests/neurodiag_metrics_author.py','docs/NEURODIAG_METRICS.md',
    'congklak/neurodiag_common.py','docs/NEURODIAG_SCOPE.md','congklak/neural_relay.py','congklak/features_relay.py','congklak/features_v3.py','tests/reference.py')
SOURCE_DATA=('data/quality-train-v1/model.json','data/ordinary-value-pilot-v1/model.json')


def freeze_body():
    return dict(schema='neurodiag-metrics-source-freeze-v1',inputs={name:m.sha(ROOT/name) for name in (*SOURCES,*SOURCE_DATA)},settings=m.SETTINGS,
        no_formal_fixture_before_freeze=True,fixture_scope='Three synthetic full states only; no selected production panel forward or labels.',no_training=True,no_native=True)


def accepted_reference(weights,features):
    """Compile just the exact accepted method/rounding AST; do not import Native."""
    source=ROOT/'congklak/neural_relay.py';tree=ast.parse(source.read_text(encoding='utf-8'));rounded=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rounded')
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model');method=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='reference')
    code=ast.fix_missing_locations(ast.Module(body=[rounded,method],type_ignores=[]));namespace=dict(math=math,SCALE=256,encoded=lambda _position,_count:(features['x'],features['d'],features['r']))
    exec(compile(code,str(source),'exec'),namespace)
    return namespace['reference'](SimpleNamespace(weights=weights,input_count=127),None)


def synthetic_units():
    states=[dict(pits=[[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]],stores=[21,21],controller=0),
        dict(pits=[[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]],stores=[21,21],controller=1),
        dict(pits=[[3,0,2,0,1,0,1],[0,2,0,3,0,1,0]],stores=[42,43],controller=0)]
    units={}
    for i,state in enumerate(states):
        features=m.literal_features(state);key=m.rawkey(state);uid=m.digest(dict(state_key=key,features=features));d=features['d'];records=[]
        for j,target in enumerate((d-2,d+2) if i==2 else (d,)):
            records.append(dict(id=f'synthetic/{i}/{j}',state_key=key,actor=state['controller'],family=f'synthetic/family{i}',cohort='ordinary_MC' if i==2 else 'base_H8',
                target_type='observed_control_selfplay_terminal_margin' if i==2 else 'approximate_H8_root_margin',stratum='synthetic',target_player0=target))
        units[uid]=dict(state=state,state_key=key,features=features,records=records,feature_origin=dict(generated=True,sources=['synthetic-literal-owner-only']))
    return units


def synthetic_weights():
    weights={name:([[0.]*shape[1] for _ in range(shape[0])] if len(shape)==2 else [0.]*shape[0]) for name,shape in m.SHAPES.items()}
    weights['layers.0.bias']=[-1.]*64;weights['layers.2.bias']=[-1.]*64
    for j in range(3):
        weights['layers.0.bias'][j]=.2;weights['layers.0.weight'][j][j]=.3
        weights['layers.2.bias'][j]=.15;weights['layers.2.weight'][j][j]=.4
        weights['layers.4.weight'][0][j]=.1*(j+1)
    weights['layers.4.bias'][0]=.05;return weights


class NeurodiagMetricsAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m.require(m.same(json.loads(FREEZE.read_text(encoding='utf-8')),freeze_body()),'Formal owner fixtures require exact current source freeze')
        cls.retained=ROOT/'build/neurodiag-metrics-author'/uuid.uuid4().hex;cls.retained.mkdir(parents=True)
        cls.models=m.load_models();cls.weights,cls.identity=m.prepare_models(cls.models);cls.units=synthetic_units();cls.batches=[m.observe({uid:unit},cls.models) for uid,unit in sorted(cls.units.items())]
        for i,batch in enumerate(cls.batches):(cls.retained/f'batch-{i}.json').write_text(m.canonical(batch)+'\n',encoding='utf-8',newline='\n')

    def test_01_literal_features_fixed_export_scalar_AST_policy_and_terminal(self):
        for batch in self.batches:
            self.assertTrue(batch['models']['shared_trunk_policy_bit_identical']);self.assertEqual(batch['counts']['shared_trunk_forwards'],1);self.assertEqual(batch['counts']['value_head_evaluations'],2)
            for obs in batch['observations'].values():
                features=obs['source_unit']['features'];cheap,d,r,actor=m._cheap_features(obs['source_unit']['state']);self.assertEqual(features['x'][:43],cheap);self.assertEqual((features['d'],features['r']),(d,r));self.assertEqual(len(features['x']),127)
                scalar_logits=[]
                for name in m.NAMES:
                    value,units,logits=accepted_reference(self.weights[name],features);scalar=m.scalar_forward(self.weights[name],features);self.assertEqual((value,units,logits),(scalar['value_player0'],scalar['value_units'],scalar['policy_logits']))
                    head=obs['models'][name];self.assertTrue(math.isclose(value,head['value_player0'],abs_tol=m.SETTINGS['scalar_atol'],rel_tol=m.SETTINGS['scalar_rtol']));self.assertEqual(units,head['value_units']);scalar_logits.append(logits)
                    for a,b in zip(logits,obs['shared']['policy_logits']):self.assertTrue(math.isclose(a,b,abs_tol=m.SETTINGS['scalar_atol'],rel_tol=m.SETTINGS['scalar_rtol']))
                self.assertEqual(scalar_logits[0],scalar_logits[1]);self.assertAlmostEqual(sum(obs['shared']['policy_probabilities']),1.)
                for layer in ('layer0','layer2'):
                    self.assertEqual(obs['shared'][layer+'_relu_active'],[v>0 for v in obs['shared'][layer+'_pre_activation']]);self.assertEqual(obs['shared'][layer+'_kink'],[abs(v)<=m.SETTINGS['relu_kink_abs'] for v in obs['shared'][layer+'_pre_activation']])
        terminal=dict(x=[0.]*127,d=4,r=0)
        for name in m.NAMES:self.assertEqual(accepted_reference(self.weights[name],terminal),(4,1024,[0.]*7));self.assertTrue(m.scalar_forward(self.weights[name],terminal)['terminal_early_return'])
        self.assertTrue(all(self.identity['models']['control']['tensor_sha256'][k]==self.identity['models']['candidate']['tensor_sha256'][k] for k in m.FROZEN))
        malformed=copy.deepcopy(self.models);malformed['candidate']['envelope']['model']['weights']['policy.bias'][0]+=.125
        with self.assertRaises(ValueError):m.prepare_models(malformed)

    def test_02_signed_analytic_sensitivity_Huber_finite_differences_and_kinks(self):
        weights=synthetic_weights();features=dict(x=[.2]*127,d=3,r=-10);trunk=m.forward_arrays(weights,features);head=m.head_diagnostics(weights,features,trunk);step=m.SETTINGS['finite_difference_step']
        self.assertLess(head['seed_output_gain'],0);self.assertGreater(head['normalized_output_gain'],0)
        for j in range(127):
            left=copy.deepcopy(features);right=copy.deepcopy(features);left['x'][j]-=step;right['x'][j]+=step
            a=m.head_diagnostics(weights,left,m.forward_arrays(weights,left))['value_player0'];b=m.head_diagnostics(weights,right,m.forward_arrays(weights,right))['value_player0'];finite=(b-a)/(2*step)
            self.assertTrue(math.isclose(finite,head['input_jacobian_player0'][j],abs_tol=m.SETTINGS['finite_difference_atol'],rel_tol=m.SETTINGS['finite_difference_rtol']))
        for j in range(65):
            left=copy.deepcopy(weights);right=copy.deepcopy(weights)
            if j<64:left['layers.4.weight'][0][j]-=step;right['layers.4.weight'][0][j]+=step
            else:left['layers.4.bias'][0]-=step;right['layers.4.bias'][0]+=step
            a=m.head_diagnostics(left,features,trunk)['tanh'];b=m.head_diagnostics(right,features,trunk)['tanh'];self.assertTrue(math.isclose((b-a)/(2*step),head['head_jacobian_normalized'][j],abs_tol=m.SETTINGS['finite_difference_atol'],rel_tol=m.SETTINGS['finite_difference_rtol']))
        fit=m.target_diagnostics(3,features,head);psi=max(-.1,min(.1,head['tanh']));self.assertEqual(fit['residual_huber_psi'],psi)
        self.assertAlmostEqual(fit['residual_huber_gradient_pre_activation'],psi*head['normalized_output_gain']);self.assertEqual(m.target_diagnostics(13,features,head)['target_status'],'endpoint');self.assertIsNone(m.target_diagnostics(13,features,head)['target_logit'])
        with self.assertRaises(ValueError):m.target_diagnostics(14,features,head)
        kink=synthetic_weights();kink['layers.0.weight'][0]=[0.]*127;kink['layers.0.bias'][0]=0.;kink_trunk=m.forward_arrays(kink,features);self.assertEqual(kink_trunk['a0'][0],0);self.assertEqual(kink_trunk['h0'][0],0)
        cancelled=m._gradient_cancellation([[1.]+[0.]*64,[-1.]+[0.]*64],[m.F(1,2)]*2);self.assertEqual(cancelled['cancellation_ratio'],0)

    def test_03_exact_root_family_mass_endpoint_SVD_and_repeated_outcome_conflict(self):
        records=[dict(id='a1',state_key='a',family='f'),dict(id='a2',state_key='a',family='f'),dict(id='b',state_key='b',family='f'),dict(id='c',state_key='c',family='g')]
        root=m.record_weights(records,'raw_root_balanced');family=m.record_weights(records,'family_balanced');self.assertEqual([root[r['id']] for r in records],[m.F(1,6),m.F(1,6),m.F(1,3),m.F(1,3)]);self.assertEqual([family[r['id']] for r in records],[m.F(1,8),m.F(1,8),m.F(1,4),m.F(1,2)])
        h=[[0.]*64 for _ in range(4)];p=[m.F(1,4)]*4;constant=m.projection_diagnostic(h,[m.F(1,2)]*4,p);self.assertLess(constant['weighted_logit_projection_residual_rmse'],1e-12);self.assertEqual(constant['rank'],1)
        conflict=m.projection_diagnostic(h,[m.F(-1,2),m.F(1,2)]*2,p);self.assertGreater(conflict['weighted_logit_projection_residual_rmse'],.5);self.assertFalse(conflict['coefficients_constructed'])
        endpoint=m.projection_diagnostic(h,[m.F(1),m.F(0),m.F(0),m.F(0)],p);self.assertEqual(endpoint['original_interior_mass'],m.rational(m.F(3,4)));self.assertEqual(endpoint['endpoint_records'],1);self.assertFalse(endpoint['endpoints_clipped'])
        for i in range(4):h[i][i]=1.
        interpolates=m.projection_diagnostic(h,[m.F(-1,2),m.F(1,2)]*2,p);self.assertTrue(interpolates['small_sample_interpolation_warning']);self.assertLess(interpolates['weighted_logit_projection_residual_rmse'],1e-12)
        absent=m.projection_diagnostic(h,[m.F(1)]*4,p);self.assertFalse(absent['available']);self.assertIsNone(absent['rank'])
        geometry=m.feature_geometry([[0.,1.],[0.,2.],[0.,3.],[0.,4.]],p);self.assertIn(0,geometry['exact_zero_variance_columns']);self.assertEqual(geometry['rank'],1)

    def test_04_cached_only_admission_summary_stability_and_source_corruption(self):
        summary=m.summarize_cached(self.batches);path=self.retained/'synthetic-summary.json';path.write_text(m.canonical(summary)+'\n',encoding='utf-8',newline='\n');before=(m.sha(path),path.stat().st_size,path.stat().st_mtime_ns)
        self.assertEqual((m.canonical(m.summarize_cached(self.batches))+'\n').encode(),path.read_bytes());self.assertEqual(before,(m.sha(path),path.stat().st_size,path.stat().st_mtime_ns));self.assertEqual(summary['counts']['distinct_units'],3);self.assertEqual(summary['counts']['target_records'],4)
        self.assertEqual(len(summary['groups']),2);self.assertEqual(summary['ordinary_MC_within_start'][0]['empirical_spread_seeds'],4)
        forbidden=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected numerical/encoder execution in cached admission'))
        names=('observe','forward_arrays','head_diagnostics','target_diagnostics','literal_features','validate_features','prepare_models','numpy_runtime','projection_diagnostic','feature_geometry','scalar_forward')
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:stack.enter_context(patch.object(m,name,forbidden))
            for batch in self.batches:m.validate_cached(batch)
        for kind in ('source','count_alias','target','error','policy'):
            bad=copy.deepcopy(self.batches[0]);item=next(iter(bad['observations'].values()));target=next(iter(item['targets'].values()))
            if kind=='source':item['source_unit_sha256']='0'*64
            elif kind=='count_alias':bad['counts']['shared_trunk_forwards']=True
            elif kind=='target':target['models']['control']['target_player0']=True
            elif kind=='error':target['models']['control']['absolute_error_player0']+=1.
            else:item['shared']['policy_logits'][0]+=.1
            with self.assertRaises(ValueError):m.validate_cached(bad)
        with self.assertRaises(ValueError):m.summarize_cached([self.batches[0],self.batches[0]])
        self.assertNotIn('torch',sys.modules);self.assertNotIn('congklak.native',sys.modules);self.assertNotIn('congklak.neural_relay',sys.modules)


if __name__=='__main__':
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(NeurodiagMetricsAuthor);ids=[test.id() for test in suite];result=unittest.TextTestRunner(verbosity=2).run(suite);retained=getattr(NeurodiagMetricsAuthor,'retained',None)
    inputs={name:m.sha(ROOT/name) for name in (*SOURCES,*SOURCE_DATA)};inputs[FREEZE.relative_to(ROOT).as_posix()]=m.sha(FREEZE)
    if retained is not None:inputs.update({p.relative_to(ROOT).as_posix():m.sha(p) for p in retained.iterdir() if p.is_file()})
    report=dict(schema='neurodiag-metrics-author-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=ids,
        methods=unittest.defaultTestLoader.getTestCaseNames(NeurodiagMetricsAuthor),inputs=inputs,retained_directory=retained.relative_to(ROOT).as_posix() if retained else None,duration_seconds=time.perf_counter()-started)
    (ROOT/'build/neurodiag-metrics-author.json').write_text(m.canonical(report)+'\n',encoding='utf-8',newline='\n');raise SystemExit(not result.wasSuccessful())
