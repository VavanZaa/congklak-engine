"""Saved-record/synthetic journal checks, never engine queries or new labels."""
import copy,ctypes as C,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from congklak import config,footprint_screen_common as s,footprint_screen_native as n,footprint_screen_worker as w

OLD=s.ROOT/'build/ordinary-epoch-pilot-v1/jobs/labels'


class AcquisitionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.old_identity=s.read(OLD/'identity.json');cls.meta=dict(cls.old_identity['schedule'][0]['meta'])
  cls.meta['id']=cls.meta['family'];cls.meta['index']=0
  cls.answer=s.read(OLD/'archives/query-000000.json')['answer']
  cls.features=s.read(OLD/'archives/query-000001.json')['answer']
 def test_saved_teacher_and_features_no_model_forward(self):
  for kind,answer,cursor in (('teacher',self.answer,'query/000000'),('features',self.features,'query/000001')):
   result=n.capture_proof(answer,kind,self.meta,cursor,self.old_identity['provider_identity'],answer['both_seat_reset']['before'])
   self.assertTrue(result['math_complete']);self.assertEqual(result['model_forwards'],0)
   if kind=='features':self.assertEqual(len(result['features']),127)
 def test_returned_node_cap_is_ineligible_but_full_raw_reviewable(self):
  # Explicit synthetic stop-envelope over immutable saved raw bytes; no search.
  a=copy.deepcopy(self.answer);a['decision']['stop_reason']='node_budget'
  cert=a['fast_certificate'];a['fast_critical_sha256']=config.digest(dict(decision=a['decision'],transition=a['transition'],
   coverage=a['coverage'],lifecycle=a['lifecycle'],trace_sha256=cert['event_sha256'],certificate=cert,
   both_seat_reset=a['both_seat_reset'],binding=a['trace']['binding']))
  proof=n.capture_proof(a,'teacher',self.meta,'query/000000',self.old_identity['provider_identity'],a['both_seat_reset']['before'])
  self.assertFalse(proof['math_complete']);self.assertTrue(proof['trace_complete'])
  self.assertEqual(proof['ineligible_reason'],'predeclared_node_cap')
  with patch.object(C,'CDLL',side_effect=AssertionError('DLL')),patch.object(C,'WinDLL',side_effect=AssertionError('native')):
   self.assertTrue(n.full_raw_review(a,'teacher',self.meta,'query/000000',self.old_identity['provider_identity'])['passed'])
 def test_compact_binding_and_clock_lifecycle_mutations_refuse(self):
  for change in ('root','cursor','rawhash','forward','critical','reset','cap','traceflags','extracounter'):
   a=copy.deepcopy(self.answer)
   if change=='root':a['trace']['binding']['root_state']['controller']=1-a['trace']['binding']['root_state']['controller']
   if change=='cursor':a['trace']['binding']['decision_id']='other'
   if change=='rawhash':a['trace']['event_sha256']='0'*64
   if change=='forward':a['trace']['operation_counts']['model_done']=1
   if change=='critical':a['fast_critical_sha256']='0'*64
   if change=='reset':a['both_seat_reset']['before']=dict(a['both_seat_reset']['before'],created=20)
   if change=='cap':a['fast_certificate']['raw_event_capacity']=2000000
   if change=='traceflags':a['trace']['truncated']=True
   if change=='extracounter':a['fast_certificate']['operation_counts'][-1]=1
   with self.subTest(change=change),self.assertRaises(Exception):
    n.capture_proof(a,'teacher',self.meta,'query/000000',self.old_identity['provider_identity'],self.answer['both_seat_reset']['before'])
 def test_root_local_stage_reconstruction_no_false_family_edges(self):
  a=copy.deepcopy(self.answer);_,keys=n.owned_records(a);after,delta=n.merge_claims(b'',keys)
  cert=a['fast_certificate'];dh=hashlib.sha256(delta).digest()
  cert.update(owner=1,generation=1,previous_generation=0,previous_claim_hash=n.ZERO,stage_delta_hash=dh.hex(),
   staged_claim_hash=hashlib.sha256(bytes.fromhex(n.ZERO)+dh+__import__('struct').pack('<QIQ',1,1,len(keys))).hexdigest(),
   stage_new_count=len(keys),staged_total_claims=len(keys),guard_records_sha256=n.ZERO)
  actual,records,again=n.stage_check(a,b'',n.ZERO,0)
  self.assertEqual(records,after);self.assertEqual(again,keys);self.assertEqual(actual['owner'],1)
  self.assertEqual(n.merge_claims(records,keys)[1],b'')
  cert['previous_generation']=False
  with self.assertRaises(Exception):n.stage_check(a,b'',n.ZERO,0)
 def test_complete_route_before_any_lease_writer_or_provider(self):
  result={'report':{'state':'complete'}}
  with patch.object(w,'has_completion',return_value=True),patch.object(w,'publish_completed',return_value=result),\
   patch.object(w,'Writer',side_effect=AssertionError('Writer')),patch.object(n,'open_session',side_effect=AssertionError('engine')):
   self.assertIs(w.run(lease=None),result)
 def test_unknown_unreturned_intent_refuses_before_new_lease(self):
  from congklak import footprint_screen_resources as resources
  parent=s.FOLDER/'pure-tests';parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as folder:
   path=Path(folder);s.atomic(path/'identity.json',{})
   with patch.object(w,'has_completion',return_value=False),patch.object(w,'scope',return_value=path),\
    patch.object(w,'_identity',return_value={'fixture':True}),\
    patch.object(w,'_read',return_value={'hold':None,'intent':{'archive':'missing.json'}}),\
    patch.object(s,'admit_plan',return_value={}),patch.object(resources,'reserve',side_effect=AssertionError('lease')) as reserve:
    with self.assertRaisesRegex(ValueError,'Unknown unreturned'):resources.launch(path,fixture=True)
    reserve.assert_not_called()
 def test_final_snapshot_requires_independent_fixture_gate(self):
  meta=self.meta;row=dict(root_index=0,kind='teacher');unit=dict(scheduled=row,proof={'math_complete':True},
   archive_ref={'path':'query.json','sha256':'sha'})
  data={'results':[unit,dict(unit,scheduled=dict(root_index=0,kind='features'))]}
  with patch.object(w,'completed',return_value={'report':{'state':'complete'},'publication_gap':False}),\
   patch.object(w,'_identity',return_value={}),patch.object(w,'_read',return_value=data),\
   patch.object(w,'pool',return_value={'roots':[meta]}),patch.object(w,'_unit_quotients',return_value=[]),\
   patch.object(s.common,'pins',return_value={}),\
   patch.object(w,'sha',return_value='sha'),patch.object(w,'_recovery_gate',side_effect=ValueError('independent gate absent')) as gate:
   self.assertEqual(len(w.production_snapshot()['roots']),1);gate.assert_not_called()
   with self.assertRaisesRegex(ValueError,'independent gate absent'):w.completed_snapshot()
 def test_recovery_gate_rejects_bool_interruption_counter(self):
  receipt=dict(schema=s.SCHEMA+'-independent-native-recovery',passed=True,independent_of_implementation=True,
   plan_sha256='plan',process_interruptions=True,subset_only_new_quotients=0,unresolved_findings=[],
   cases={k:{} for k in w.HOOKS if k!='before_query'})
  with patch.object(w,'read',return_value=receipt),patch.object(s,'admit_plan',return_value={}),\
   patch.object(s.common,'pins',return_value={}),patch.object(w,'sha',return_value='plan'):
   with self.assertRaisesRegex(ValueError,'gate is missing or incomplete'):w._recovery_gate()
 def test_fixed_schedule_has_two_queries_per_root(self):
  p={'roots':[dict(self.meta,index=i,id=str(i)) for i in range(512)]};rows=w.schedule(p,False)
  self.assertEqual(len(rows),1024);self.assertEqual([r['index'] for r in rows],list(range(1024)))
  self.assertEqual([r['kind'] for r in rows],['teacher','features']*512)
  self.assertEqual(len(w.schedule(p,True)),2)
 def test_streaming_archive_is_canonical(self):
  from congklak import footprint_screen_resources as resources
  parent=s.FOLDER/'pure-tests';parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as folder:
   p=Path(folder)/'archive.json';body=dict(z=[1,'x'],a=True,finite=.125)
   with patch.object(resources,'check_namespace',return_value=True):w._archive_write(p,body)
   self.assertEqual(p.read_text(),config.canonical(body)+'\n')
   with self.assertRaises(Exception):w._archive_write(p,body)
 def test_recorded_return_adopts_without_queries_at_commit_boundaries(self):
  parent=s.FOLDER/'pure-tests';parent.mkdir(parents=True,exist_ok=True)
  for boundary in ('before_result_commit','after_result_commit','before_trace_commit','after_trace_commit'):
   with self.subTest(boundary=boundary),tempfile.TemporaryDirectory(dir=parent) as folder:
    path=Path(folder);row=dict(index=0,root_index=0,kind='teacher',root_id=self.meta['id'],origin=self.meta['origin'])
    identity=dict(schedule=[row],native_identity={'native':'fixture'},fixture=True);s.atomic(path/'identity.json',identity)
    life=dict(created=2,destroyed=0,retained_inference_controls={'control':1,'candidate':2})
    stage=dict(generation=1,owner=1,previous_generation=0,previous_claim_hash=w.ZERO,stage_delta_hash=w.ZERO,
     staged_claim_hash='chain',stage_new_count=0,staged_total_claims=0,guard_records_sha256=w.ZERO)
    capture=dict(math_complete=True,lifecycle_after=life)
    check=dict(capture=capture,stage=stage,claims=b'',keys=[])
    events=[];w._append(path,events,dict(kind='guard_restore',root_index=0,generation=0,canonical_claims_sha256=w.ZERO))
    intent=dict(kind='intent',cursor='query/000000',scheduled=row,archive=w._archive_path(0),sink=w._sink_path(0));w._append(path,events,intent)
    proof=w._append(path,events,dict(kind='provider_epoch',cursor=intent['cursor'],native_identity=identity['native_identity'],
     lifecycle=life,generation=0,canonical_claims_sha256=w.ZERO,previous_commit_sha256=s.sha(path/'commits/000001.json')))
    epoch=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
    s.atomic(path/intent['archive'],dict(provider_epoch_ref=epoch))
    with patch.object(w,'pool',return_value={'roots':[self.meta]}),patch.object(w,'_check_capture',return_value=check),\
     patch.object(n,'open_session',side_effect=AssertionError('recomputed search')),patch.object(C,'CDLL',side_effect=AssertionError('DLL')):
     data=w._read(path,identity)
     def pause(point):
      if point==boundary:raise InterruptedError(point)
     with self.assertRaises(InterruptedError):w._adopt(path,identity,data,pause)
     settled=w._read(path,identity)
     if settled['intent'] is not None:w._adopt(path,identity,settled)
     complete=w._read(path,identity)
     self.assertEqual(len(complete['results']),1);self.assertIsNone(complete['intent'])
     self.assertEqual(complete['generation'],1)
     self.assertNotIn('quotients',complete['results'][0])


if __name__=='__main__':unittest.main()
