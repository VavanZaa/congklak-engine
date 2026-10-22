"""Pure owner tests; actual optimizer/recovery tests need independent admission."""
import builtins,copy,sys,types,unittest
import congklak
from unittest.mock import patch
from congklak import ordinary_epoch_common as stage,ordinary_epoch_data as data
from congklak import ordinary_epoch_training as train,ordinary_epoch_sidecar as side

class OrdinaryEpochTests(unittest.TestCase):
    def test_metadata_exactly192_and_all_old_exclusions_remain(self):
        rows=data.membership();self.assertEqual(len(rows),192)
        self.assertEqual({r['metadata']['cell'] for r in rows},set(data.CELLS))
        self.assertEqual(data.COUNTS,dict(zip(data.CELLS,(39,34,58,61))))
        self.assertEqual(len({r['reservation']['joint_component'] for r in rows}),192)
        self.assertNotIn('torch',sys.modules)

    def test_input_pin_inventory_is_resolved_unique_before_hashing(self):
        seen=[]
        def capture(paths):
            seen.extend(paths);return {}
        with patch.object(data.common,'pins',side_effect=capture):data.input_pins()
        self.assertEqual(len(seen),len({p.resolve() for p in seen}))
        self.assertTrue(all(p==p.resolve() for p in seen))
        teacher=(stage.ROOT/'data/deeper-9000-v2/work.sqlite3').resolve()
        self.assertEqual(sum(p==teacher for p in seen),1)

    def test_fixed_new_recipe_and_four_cell_fixture(self):
        s=train.settings();self.assertEqual((s['steps'],s['batch_size'],s['roots']),(400,128,192))
        self.assertEqual(s['sampling'],'uniform-four-cells/within-cell-uniform-replacement')
        self.assertEqual(s['selected_candidate'],'final-fixed-step-only')
        self.assertEqual(s['steps']*s['batch_size'],51200)
        fake=[dict(cell=r['metadata']['cell'],state_key=r['metadata']['state_key']) for r in data.membership()]
        fixture=data.fixture_rows(fake);self.assertEqual(len(fixture),32)
        self.assertEqual([sum(r['cell']==c for r in fixture) for c in data.CELLS],[8]*4)
        for c in data.CELLS:self.assertEqual([r for r in fixture if r['cell']==c],[r for r in fake if r['cell']==c][:8])
        with self.assertRaises(train.TrainingError):train.settings(1)

    def test_output_cannot_touch_prior_stage(self):
        self.assertEqual(train._path(train.DEFAULT,False),train.DEFAULT.resolve())
        for path in (stage.ROOT/'data/value-training-v1/training',stage.ROOT/'build/value-training-v1/fixture'):
            with self.assertRaises(train.TrainingError):train._path(path)

    def test_completed_dispatch_before_writer_lease_forward_and_optimizer(self):
        path=stage.FOLDER/'fixtures/owner-dispatch';expected=dict(state='complete',steps=6)
        with patch.object(train,'_has_completion',return_value=True),patch.object(train,'completed',return_value=expected),\
             patch.object(train,'_identity',return_value={}),patch.object(train,'Writer',side_effect=AssertionError('writer')):
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name=='torch' or name.endswith('ordinary_epoch_resources'):raise AssertionError('completed import')
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guarded):self.assertEqual(train.run(path,fixture=True),expected)

    def test_policy_proof_preserves_original_tensors_without_query(self):
        p=side.policy_proof();self.assertEqual(p['base_file_sha256'],side.BASE_SHA)
        self.assertTrue(p['policy_arithmetic_unchanged']);self.assertEqual(p['policy_training_steps'],0)
        initial=data.read(data.INITIAL)['model']['branch_weights']
        self.assertEqual(side.tensor_sha(initial),side.frozen.INITIAL_TENSOR_SHA)
        self.assertEqual(initial['v'],[[0.]*16]);self.assertEqual(initial['c'],[0.])
        self.assertNotIn('torch',sys.modules)

    def test_cooperative_boundary_precedes_strict_margin_assertion(self):
        class Lease:
            def __init__(self,remaining):self.remaining=remaining;self.calls=[]
            def remaining_ns(self,lease):self.calls.append('remaining');return self.remaining
            def assert_worker_lease(self,lease):self.calls.append('assert')
        for remaining in (14_000_000_000,19_000_000_000,20_000_000_000):
            resource=Lease(remaining);self.assertFalse(train._before_update(resource,'known-lease'))
            self.assertEqual(resource.calls,['remaining'])
        resource=Lease(21_000_000_000);self.assertTrue(train._before_update(resource,'known-lease'))
        self.assertEqual(resource.calls,['remaining','assert'])

    def test_resealed_feature_or_target_digests_cannot_replace_source_rows(self):
        # Synthetic numerical fixture, using an existing root identity only;
        # these values are never a label, feature cache or admitted model input.
        meta=data.membership()[0]['metadata']
        expected=[dict(state_key=meta['state_key'],component='fixture',x=[.25],target_z=.5)]
        role=dict(row_sha256=[data.config.digest(r) for r in expected],components=['fixture'])
        data._validate_source_rows(role,expected,expected)
        for key,new in (('target_z',.75),('x',[.5])):
            forged=copy.deepcopy(expected);forged[0][key]=new
            resealed=dict(role,row_sha256=[data.config.digest(r) for r in forged])
            with self.assertRaisesRegex(ValueError,'original pinned sources'):
                data._validate_source_rows(resealed,expected)
            with self.assertRaisesRegex(ValueError,'Actual train rows'):
                data._validate_source_rows(role,expected,forged)

    def test_production_gate_requires_closed_labels_and_initial_equivalence(self):
        digest='c'*64;ref=dict(path=stage.relative(stage.FOLDER/'fixtures/receipt.json'),sha256=digest)
        labels=dict(schema='ordinary-epoch-pilot-v1-label-admission',required_roots=128,completed_roots=128,
            fit_cases=32,horizon=8,all_legal_Q_complete=True,features_complete=True,ownership_passed=True,
            support192_disjoint=True,pending_attempts=0,hold=False,catalog_ref=ref,ownership_prefix_ref=ref,fit_manifest_ref=ref)
        initial=dict(schema='ordinary-epoch-pilot-v1-initial-conformance-admission',features_exact=True,
            control_candidate_predictions_exact=True,fixed128_Q_exact=True,fixed128_visits_exact=True,
            ownership_passed=True,pending_attempts=0,hold=False,model_sha256=digest,
            initial_model_ref=dict(path=stage.relative(train.INITIAL/'model.json'),sha256=digest),
            catalog_ref=ref,ownership_prefix_ref=ref)
        module=types.ModuleType('congklak.ordinary_epoch_execute')
        module.admit_labels=lambda:labels;module.admit_initial_conformance=lambda:initial
        recovery=dict(schema='ordinary-epoch-pilot-v1-recovery-gate-admission',passed=True,actual_kills=7,
            accepted_label_boundaries=6,zero_query_hold_verified=True,inputs={ref['path']:digest})
        gate=types.ModuleType('congklak.ordinary_epoch_recovery_gate');gate.admit=lambda:recovery
        with patch.dict(sys.modules,{'congklak.ordinary_epoch_execute':module,'congklak.ordinary_epoch_recovery_gate':gate}),\
             patch.object(congklak,'ordinary_epoch_execute',module,create=True),\
             patch.object(congklak,'ordinary_epoch_recovery_gate',gate,create=True),\
             patch.object(stage,'admit_plan',return_value={}),patch.object(train,'sha',return_value=digest):
            self.assertEqual(train.production_gate()['initial_conformance'],initial)
            labels['pending_attempts']=False
            with self.assertRaises(train.TrainingError):train.production_gate()
            labels['pending_attempts']=0;initial['features_exact']=1
            with self.assertRaises(train.TrainingError):train.production_gate()
            initial['features_exact']=True;initial['model_sha256']='d'*64
            with self.assertRaises(train.TrainingError):train.production_gate()
            initial['model_sha256']=digest;recovery['zero_query_hold_verified']=False
            with self.assertRaises(train.TrainingError):train.production_gate()

if __name__=='__main__':unittest.main()
