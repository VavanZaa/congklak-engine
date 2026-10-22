"""Engineering acceptance for the separately versioned, frozen-policy sidecar."""
import copy
import json
import math
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
from congklak import config,quality_search as puct,value_sidecar as side
from congklak.native import ROOT,Native,NativeError,flatten
from congklak.neural_relay import Model as BaseModel,rounded
from congklak.puct_cpu_inference import CpuBatchEvaluator

WORK=ROOT/'build/valuepilot-interface-v1/interface-owner-fixtures'
def boundary(active,actor=0,forced=False):
    pits=[[0]*7,[0]*7]
    if forced:pits[1-actor][0]=active
    else:
        for i in range(active):pits[(i%14)//7][i%7]+=1
    return dict(pits=pits,stores=[(98-active)//2,98-active-(98-active)//2],controller=actor)
def cases():
    positions=[boundary(a,c) for a in (0,4,5,24,25,49,50,74,75,98) for c in (0,1)]
    positions += [boundary(a,c,True) for a in (4,24,25,74,75) for c in (0,1)]
    panel=json.loads((ROOT/'build/neurodiag-panel.json').read_text())
    positions += [r['state'] for r in panel['primary']]
    return positions
def synthetic(model,manifest,path):
    w=copy.deepcopy(model.weights)
    w['v']=[[(i-7)/128 for i in range(16)]];w['c']=[0.03125]
    path.write_text(config.canonical(side.make_export(w,manifest,{'engineering_fixture':True,'optimizer_steps':0})),encoding='utf-8')
    return side.Model(path)

class ValueSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.manifest=WORK/'manifest.json'
        if not cls.manifest.exists():cls.manifest.write_text(config.canonical({'schema':'value-sidecar-engineering-fixture-v1','uses_targets':False}),encoding='utf-8')
        cls.initial=WORK/'initial'
        side.initialize(cls.manifest,cls.initial,fixture=True)
        cls.model=side.Model(cls.initial/'model.json');cls.base=BaseModel(side.BASE)
        cls.synthetic=synthetic(cls.model,cls.manifest,WORK/'synthetic.json');cls.positions=cases()
    def test_parameter_shapes_and_initialization(self):
        self.assertEqual(sum(len(list(side.flatten_weights(v))) for v in self.model.weights.values()),2065)
        self.assertEqual(self.model.weights['v'],[[0.]*16]);self.assertEqual(self.model.weights['c'],[0.])
        self.assertEqual(self.model.envelope['model']['base_export'],self.base.envelope)
        self.assertNotEqual(self.model.id,self.base.id)
        self.assertTrue(all(abs(x)<=1/math.sqrt(127) for name in ('A','b') for x in side.flatten_weights(self.model.weights[name])))
    def test_initialization_dedicated_torch_generator(self):
        import torch
        generator=torch.Generator(device='cpu');generator.manual_seed(20261016);limit=1/math.sqrt(127)
        self.assertEqual(self.model.weights['A'],torch.empty((16,127),dtype=torch.float32).uniform_(-limit,limit,generator=generator).tolist())
        self.assertEqual(self.model.weights['b'],torch.empty((16,),dtype=torch.float32).uniform_(-limit,limit,generator=generator).tolist())
    def test_zero_branch_scalar_native_and_reference(self):
        with CpuBatchEvaluator(side.BASE,backend='scalar') as control,side.SidecarEvaluator(self.model,backend='scalar') as candidate:
            for p in self.positions:
                a=control.evaluate_many([p])[0];b=candidate.evaluate_many([p])[0];ref=self.model.reference(p)
                self.assertEqual(a['value_player0'],b['value_player0'])
                self.assertEqual(struct.pack('<7d',*a['policy_logits']),struct.pack('<7d',*b['policy_logits']))
                self.assertLessEqual(abs(b['value_player0']-ref[0]),1e-10)
                self.assertEqual(candidate.evaluate_units(p)['value_units'],ref[1])
                self.assertEqual(a['feature_simulations'],b['feature_simulations']);self.assertEqual(a['feature_deposits'],b['feature_deposits'])
    def test_nonzero_branch_policy_exact_and_domain(self):
        changed=0
        with CpuBatchEvaluator(side.BASE,backend='scalar') as control,side.SidecarEvaluator(self.synthetic,backend='scalar') as candidate:
            for p in self.positions:
                a=control.evaluate_many([p])[0];b=candidate.evaluate_many([p])[0];ref=self.synthetic.reference(p)
                self.assertEqual(struct.pack('<7d',*a['policy_logits']),struct.pack('<7d',*b['policy_logits']))
                self.assertLessEqual(abs(b['value_player0']-ref[0]),1e-10)
                self.assertEqual(candidate.evaluate_units(p)['value_units'],ref[1])
                if not 25<=sum(map(sum,p['pits']))<=74:self.assertEqual(a['value_player0'],b['value_player0'])
                else:changed+=int(a['value_player0']!=b['value_player0'])
            self.assertGreater(candidate.snapshot_counters()['branch_calls'],0)
            self.assertGreater(candidate.snapshot_counters()['branch_bypasses'],0)
        self.assertGreater(changed,0)
    def test_batch_scalar_avx2_parity(self):
        with side.SidecarEvaluator(self.synthetic,backend='scalar') as scalar,side.SidecarEvaluator(self.synthetic,backend='auto') as packed,CpuBatchEvaluator(side.BASE,backend='auto') as control:
            for start in range(0,len(self.positions),32):
                group=self.positions[start:start+32];a=scalar.evaluate_many(group);b=packed.evaluate_many(group);original=control.evaluate_many(group)
                for p,x,y,base in zip(group,a,b,original):
                    z=scalar.evaluate_many([p])[0]
                    self.assertEqual(struct.pack('<7d',*y['policy_logits']),struct.pack('<7d',*base['policy_logits']))
                    for key in ('value_player0','policy_logits','feature_simulations','feature_deposits'):
                        self.assertEqual(x[key],y[key]);self.assertEqual(x[key],z[key])
    def test_torch_fp32_conformance_and_policy_fixed(self):
        import torch
        torch.set_num_threads(1)
        w={k:torch.tensor(v,dtype=torch.float32) for k,v in self.base.weights.items()}
        branch={k:torch.tensor(v,dtype=torch.float32) for k,v in self.synthetic.weights.items()}
        maximum={'seeds':0.,'policy_logits':0.,'score_units':0}
        for p in self.positions:
            x,d,r=side.features_relay.encode(p,127)
            if not r:continue
            x=torch.tensor(x,dtype=torch.float32);h=x
            for prefix in ('layers.0','layers.2'):h=torch.relu(torch.mv(w[prefix+'.weight'],h)+w[prefix+'.bias'])
            u=(torch.mv(w['layers.4.weight'],h)+w['layers.4.bias'])[0]
            before=torch.mv(w['policy.weight'],h)+w['policy.bias']
            active=sum(map(sum,p['pits']))
            if 25<=active<=74:u=u+(torch.mv(branch['v'],torch.relu(torch.mv(branch['A'],x)+branch['b']))+branch['c'])[0]
            after=torch.mv(w['policy.weight'],h)+w['policy.bias'];self.assertTrue(torch.equal(before,after))
            residual=float((r*torch.tanh(u)).item());reference=self.synthetic.reference(p)
            maximum['seeds']=max(maximum['seeds'],abs(d+residual-reference[0]))
            maximum['policy_logits']=max(maximum['policy_logits'],max(abs(a-b) for a,b in zip(after.tolist(),reference[2])))
            maximum['score_units']=max(maximum['score_units'],abs(d*256+rounded(residual*256)-reference[1]))
        self.assertLessEqual(maximum['seeds'],.001);self.assertLessEqual(maximum['policy_logits'],1e-5);self.assertLessEqual(maximum['score_units'],1)
        (WORK/'torch-conformance.json').write_text(config.canonical({'maximum_errors':maximum,'positions':len(self.positions),'optimizer_steps':0}),encoding='utf-8')
    def test_synthetic_analytic_branch_derivatives(self):
        import torch
        x=torch.linspace(-.1,.4,127,dtype=torch.float64)
        A=torch.tensor(self.synthetic.weights['A'],dtype=torch.float64,requires_grad=True)
        b=torch.tensor(self.synthetic.weights['b'],dtype=torch.float64,requires_grad=True)
        v=torch.tensor(self.synthetic.weights['v'],dtype=torch.float64,requires_grad=True)
        c=torch.tensor(self.synthetic.weights['c'],dtype=torch.float64,requires_grad=True)
        pre=A@x+b;h=torch.relu(pre);y=torch.tanh(torch.tensor(.2)+(v@h+c)[0]);y.backward()
        factor=1-float(y.detach())**2;grad_h=factor*v.detach()[0]*(pre.detach()>0)
        self.assertTrue(torch.allclose(A.grad,grad_h[:,None]*x[None,:],atol=1e-14,rtol=0))
        self.assertTrue(torch.allclose(b.grad,grad_h,atol=1e-14,rtol=0))
        self.assertTrue(torch.allclose(v.grad,factor*h.detach()[None,:],atol=1e-14,rtol=0))
        self.assertAlmostEqual(float(c.grad),factor,14)
    def test_low_active_fixed128_search_invariance(self):
        positions=[boundary(a,c) for a in (4,5,24) for c in (0,1)]
        with CpuBatchEvaluator(side.BASE,backend='scalar') as control,side.SidecarEvaluator(self.synthetic,backend='scalar') as candidate:
            for p in positions:
                a=puct.search(p,control,simulations=128,c_puct=.25);b=puct.search(p,candidate,simulations=128,c_puct=.25)
                for key in ('chosen','visits','action_mean_value_player0','estimated_value_player0','completed_simulations','nodes','root_priors','policy'):self.assertEqual(a[key],b[key])
                for key in a['work']:
                    if key!='inference_ms':self.assertEqual(a['work'][key],b['work'][key])
    def test_completed_readonly_before_any_writer_or_torch(self):
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.initial.iterdir() if p.is_file()}
        with patch.object(side,'_writer',side_effect=AssertionError('writer touched')),patch.object(side,'bindings',wraps=side.bindings):
            self.assertEqual(side.initialize(self.manifest,self.initial)['state'],'complete')
        after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.initial.iterdir() if p.is_file()}
        self.assertEqual(before,after)
    def test_malformed_exports_and_identity_refuse(self):
        for tag,mutate in (
            ('extra',lambda e:e['model']['branch_weights'].update(extra=[0.])),
            ('base',lambda e:e['model']['base_export']['model']['weights']['policy.bias'].__setitem__(0,0.)),
            ('gate',lambda e:e['model']['bindings']['gate'].__setitem__('minimum_active',24)),
            ('wrong-schema',lambda e:e['model'].__setitem__('schema','congklak-mlp-export-v4'))):
            e=copy.deepcopy(self.model.envelope);mutate(e);e['sha256']=config.digest(e['model']);path=WORK/(tag+'.json');path.write_text(json.dumps(e),encoding='utf-8')
            with self.assertRaises((NativeError,ValueError)):side.Model(path)
        with self.assertRaises((NativeError,KeyError)):BaseModel(self.initial/'model.json')
    def test_cache_and_cancellation(self):
        from congklak.phase6_inference import InferenceStopped
        with side.SidecarEvaluator(self.synthetic,cache_capacity=2) as candidate:
            p=boundary(49);a=candidate.evaluate_many([p])[0];b=candidate.evaluate_many([p])[0]
            self.assertFalse(a['cache_hit']);self.assertTrue(b['cache_hit']);self.assertEqual(a['value_player0'],b['value_player0'])
            candidate.begin_decision();self.assertFalse(candidate.evaluate_many([p])[0]['cache_hit'])
            with self.assertRaises(InferenceStopped):candidate.evaluate_many([p],cancel=lambda:True)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ValueSidecarTests))
    report={'schema':'value-sidecar-owner-acceptance-v1','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'passed':result.wasSuccessful(),'optimizer_steps':0,'sources':{name:side.sha(ROOT/name) for name in side.DEPENDENCIES},'test_source_sha256':side.sha(__file__)}
    WORK.parent.mkdir(parents=True,exist_ok=True)
    index=len(list(WORK.parent.glob('interface-owner-attempt-*.json')))+1
    (WORK.parent/f'interface-owner-attempt-{index:03}.json').write_text(config.canonical(report),encoding='utf-8')
    (WORK.parent/'interface-owner-acceptance.json').write_text(config.canonical(report),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
