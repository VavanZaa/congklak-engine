"""Bounded adapter acceptance; the actual 664-root check is a separate command."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,value_sidecar_v2 as side
from congklak.native import ROOT,NativeError
from congklak.neural_relay import Model as BaseModel
from congklak.puct_cpu_inference import CpuBatchEvaluator

WORK=ROOT/'build/ordinary-development-v1/conformance-owner-unit-fixtures'

def boundary(active,actor=0):
    pits=[[0]*7,[0]*7]
    for i in range(active):pits[(i%14)//7][i%7]+=1
    return dict(pits=pits,stores=[(98-active)//2,98-active-(98-active)//2],controller=actor)

class ValueSidecarV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True)
        cls.manifest=WORK/'manifest.json'
        side._atomic(cls.manifest,{'schema':'value-sidecar-v2-unit-fixture','training_eligible':False,'uses_targets':False})
        old=json.loads((ROOT/'build/valuepilot-interface-v1/interface-owner-fixtures/initial/model.json').read_text())
        cls.initial=WORK/'initial';cls.initial.mkdir(exist_ok=True)
        cls.export=side.make_export(old['model']['branch_weights'],cls.manifest,old['model']['provenance'])
        side._atomic(cls.initial/'model.json',cls.export)
        cls.model=side.Model(cls.initial/'model.json')
        body=dict(schema='value-sidecar-initialization-certificate-v2',model_sha256=cls.model.id,model_file_sha256=cls.model.file_sha256,bindings=cls.model.identity,optimizer_steps=0)
        certificate={'body':body,'sha256':config.digest(body)}
        side._atomic(cls.initial/'certificate.json',certificate)
        side._atomic(cls.initial/'COMPLETED.json',{'certificate_sha256':certificate['sha256']})
        weights=copy.deepcopy(cls.model.weights);weights['v']=[[(i-7)/128 for i in range(16)]];weights['c']=[.03125]
        side._atomic(WORK/'synthetic.json',side.make_export(weights,cls.manifest,{'engineering_fixture':True,'training_eligible':False,'optimizer_steps':0}))
        cls.synthetic=side.Model(WORK/'synthetic.json')
    def test_versioned_wrapper_reuses_frozen_native_and_base(self):
        from congklak import value_sidecar as old
        self.assertNotEqual(side.SCHEMA,old.SCHEMA)
        self.assertEqual(side.DLL,old.DLL);self.assertEqual(side.ABI,old.ABI)
        self.assertEqual(side.build_record(),old.build_record())
        self.assertEqual(self.model.envelope['model']['base_export'],BaseModel(side.BASE).envelope)
        self.assertNotEqual(self.model.id,self.model.base.id)
        self.assertEqual(side.tensor_sha(self.model.weights),side.INITIAL_TENSOR_SHA)
        self.assertIsNone(self.model.identity['reserved_corpus'])
    def test_zero_and_nonzero_policy_and_direct_bypass(self):
        positions=[boundary(a,c) for a in (0,4,24,25,49,50,74,75,98) for c in (0,1)]
        for backend in ('scalar','avx2'):
            with CpuBatchEvaluator(side.BASE,backend=backend) as base,side.SidecarEvaluator(self.model,backend=backend) as zero,side.SidecarEvaluator(self.synthetic,backend=backend) as candidate:
                for p,a,z,b in zip(positions,base.evaluate_many(positions),zero.evaluate_many(positions),candidate.evaluate_many(positions)):
                    self.assertEqual(a['value_player0'],z['value_player0'])
                    self.assertEqual(struct.pack('<7d',*a['policy_logits']),struct.pack('<7d',*z['policy_logits']))
                    self.assertEqual(struct.pack('<7d',*a['policy_logits']),struct.pack('<7d',*b['policy_logits']))
                    self.assertLessEqual(abs(b['value_player0']-self.synthetic.reference(p)[0]),1e-10)
                    for k in ('feature_simulations','feature_deposits','native_actual_evals'):self.assertEqual(a[k],b[k])
                    if not 25<=sum(map(sum,p['pits']))<=74:self.assertEqual(a['value_player0'],b['value_player0'])
    def test_complete_initialize_is_readonly_before_writer_and_torch(self):
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.initial.iterdir()}
        with patch.object(side,'_writer',side_effect=AssertionError('writer touched')):
            self.assertEqual(side.initialize(self.manifest,self.initial,fixture=True)['state'],'complete')
        after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.initial.iterdir()}
        self.assertEqual(before,after)
    def test_old_schema_and_unknown_tensor_refuse(self):
        for tag,change in (('old',lambda x:x['model'].__setitem__('schema','congklak-value-sidecar-v1')),
                           ('tensor',lambda x:x['model']['branch_weights'].__setitem__('extra',[0.])),
                           ('gate',lambda x:x['model']['bindings']['gate'].__setitem__('minimum_active',24))):
            value=copy.deepcopy(self.export);change(value);value['sha256']=config.digest(value['model'])
            path=WORK/(tag+'.json');side._atomic(path,value)
            with self.assertRaises((NativeError,ValueError)):side.Model(path)
    def test_production_cannot_accept_scratch_or_old_hold_manifest(self):
        old=ROOT/'data/neural-value-pilot-v1/reservations/manifest.json'
        for manifest in (self.manifest,old):
            with self.assertRaises((NativeError,ValueError)):
                side.initialize(manifest,WORK/'forbidden-production',fixture=False)
        self.assertFalse((WORK/'forbidden-production').exists())
    def test_production_loader_requires_conformance_even_with_valid_envelope(self):
        value=copy.deepcopy(self.export)
        value['model']['provenance'].update(engineering_fixture=False,training_eligible=True,conformance_file={'path':'missing','sha256':'0'*64})
        value['sha256']=config.digest(value['model']);path=WORK/'pretend-production.json';side._atomic(path,value)
        with self.assertRaises((NativeError,ValueError)):side.Model(path)
    def test_no_optimizer_or_cuda_in_initializer_sources(self):
        text=Path(side.__file__).read_text()
        self.assertNotIn('torch.optim',text);self.assertNotIn('cuda()',text);self.assertNotIn("device='cuda'",text)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ValueSidecarV2Tests))
    report=dict(schema='value-sidecar-v2-owner-unit-acceptance',passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),optimizer_steps=0,
        inputs={str(Path(side.__file__).relative_to(ROOT)).replace('\\','/'):side.sha(side.__file__),
                'tests/test_value_sidecar_v2.py':side.sha(__file__)})
    destination=ROOT/'build/ordinary-development-v1';destination.mkdir(exist_ok=True,parents=True)
    index=len(list(destination.glob('conformance-owner-unit-attempt-*.json')))+1
    side._atomic(destination/f'conformance-owner-unit-attempt-{index:03}.json',report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
