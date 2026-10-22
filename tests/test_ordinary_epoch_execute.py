"""Pure corruption/lifecycle checks: no models, engines, labels or games."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from congklak import ordinary_epoch_execute as e,ordinary_epoch_native as n

CPU=dict(group=0,logical=2,mask=4)
PLAN=dict(resource=dict(native_timed_cpu=dict(group=0,logical=2)))


def fixture(mode='timed25',kind='decision'):
 timing=dict(outer_elapsed_ms=20.,requested_outer_ms=25 if mode=='timed25' else None,
  search_window_ms=12 if mode=='timed25' else 0,archive_outside_clock=True,
  policy=n.TEACHER if kind=='teacher' else n.POLICY)
 answer=dict(fast_timing=timing,decision=dict(requested_ms=timing['search_window_ms'],wall_ms=15.))
 body=dict(query_kind=kind,engine='control',decision_id='query/000000',answer=answer,cpu=CPU,
  clock=dict(critical_ms=26.,requested_ms=25 if mode=='timed25' else None,
   raw_hash_guard_staging_included=True,archive_outside_clock=True))
 identity=dict(operation='guards',schedule=[dict(mode=mode,kind=kind,engine='control')])
 return body,identity


class ExecutePureTests(unittest.TestCase):
 def test_declared_timed_budget_cannot_hide_overshoot(self):
  body,identity=fixture()
  with patch.object(e,'read',return_value=PLAN):
   self.assertTrue(e._clock_check(body,identity))
   for mutate in ('requested','outer','policy','window','nativewall','cpu','boolcpu','nan','archive'):
    bad=copy.deepcopy(body)
    if mutate=='requested':bad['clock']['requested_ms']=None
    if mutate=='outer':bad['answer']['fast_timing']['requested_outer_ms']=None
    if mutate=='policy':bad['answer']['fast_timing']['policy']['reserve_ms']=8.
    if mutate=='window':bad['answer']['decision']['requested_ms']=9
    if mutate=='nativewall':bad['answer']['decision']['wall_ms']=21.
    if mutate=='cpu':bad['cpu']['mask']=8
    if mutate=='boolcpu':bad['cpu']['group']=False
    if mutate=='nan':bad['clock']['critical_ms']=float('nan')
    if mutate=='archive':bad['clock']['archive_outside_clock']=False
    with self.subTest(mutate=mutate),self.assertRaises(Exception):e._clock_check(bad,identity)
 def test_fixed_teacher_and_probe_are_untimed(self):
  for kind,mode in (('decision','fixed128'),('teacher',None),('features',None),('prediction',None)):
   body,identity=fixture(mode,kind)
   with patch.object(e,'read',return_value=PLAN):
    self.assertTrue(e._clock_check(body,identity))
    bad=copy.deepcopy(body);bad['clock']['requested_ms']=25
    with self.assertRaises(Exception):e._clock_check(bad,identity)
    if kind in ('teacher','decision'):
     bad=copy.deepcopy(body);bad['answer']['decision']['requested_ms']=1
     with self.assertRaises(Exception):e._clock_check(bad,identity)
 def test_completed_api_never_requires_lease_writer_or_provider(self):
  result={'report':{'state':'complete'}}
  with patch.object(e,'has_completion',return_value=True),patch.object(e,'completed_job',return_value=result),\
   patch.object(e,'Writer',side_effect=AssertionError('writer')),\
   patch.object(n,'open_providers',side_effect=AssertionError('engine')):
   self.assertIs(e.run(lease=None),result)
 def test_candidate_paths_are_fixed_before_export_loading(self):
  with patch('congklak.ordinary_epoch_sidecar.verify_export',side_effect=AssertionError('export opened')) as verify:
   for operation,fixture_flag in (('conformance',True),('conformance',False),('guards',False),('arena',True),('arena',False)):
    with self.subTest(operation=operation,fixture=fixture_flag),self.assertRaises(ValueError):
     e._candidate(operation,fixture_flag,e.s.DATA/'relocated/model.json')
   verify.assert_not_called()
 def test_probe_lifecycle_continuity_cannot_be_resealed(self):
  body,identity=fixture(None,'features');identity.update(fixture=False,provider_identity={})
  life=dict(created=2,destroyed=0,retained_inference_controls={'control':1,'candidate':2})
  after=dict(created=4,destroyed=2,retained_inference_controls={'control':3,'candidate':4})
  body.update(identity_sha256='file-sha',state={},origin='fit/a')
  body['answer'].update(both_seat_reset=dict(before=life,after=after),lifecycle=dict(before=after,after=after),
   fast_owned_high='',fast_violations='',fast_certificate=dict(owned_count=0,owned_sha256='x',event_sha256='y'))
  with patch.object(e,'read',return_value=PLAN),patch.object(n,'validate_probe',return_value=True),\
   patch('congklak.ordinary_epoch_guard.reference_owned',return_value=[]),\
   patch('congklak.ordinary_epoch_guard.stage_reference',return_value={'delta_records':b''}):
   self.assertTrue(e._archive_check(body,identity,None,None,'claims',0,'file-sha',life)[0]['passed'])
   wrong=copy.deepcopy(life);wrong['created']=6
   with self.assertRaises(Exception):e._archive_check(body,identity,None,None,'claims',0,'file-sha',wrong)
   bad=copy.deepcopy(body);bad['identity_sha256']='body-digest';bad['_file_identity_sha256']='file-sha'
   with self.assertRaises(Exception):e._archive_check(bad,identity,None,None,'claims',0,'file-sha',life)
   bad=copy.deepcopy(body);bad['answer']['both_seat_reset']['after']=dict(after,created=10)
   with self.assertRaises(Exception):e._archive_check(bad,identity,None,None,'claims',0,'file-sha',life)
 def test_completed_prefix_is_exact_canonical_document(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder);(path/'claims.bin').write_bytes(b'')
   identity=dict(registry={'path':'registry','sha256':'registry-sha'})
   data=dict(generation=0,claim_hash='chain',claimed=SimpleNamespace(sha256='claims-sha',records=b''),events=[])
   report=dict(state='complete',ownership_passed=True,hold=None)
   with patch.object(e.s,'path_of',return_value=path),patch.object(e,'_validate_identity',return_value=identity),\
    patch.object(e,'_replay',return_value=data),patch.object(e,'_check_orphans'),\
    patch.object(e,'_phase_summary',return_value=report),patch.object(e,'_complete_data',return_value=True),\
    patch.object(e,'_ref',side_effect=lambda p:dict(path=p.name,sha256='sha')):
    prefix=e._prefix_document(path,identity,data,report)
    def reader(p):return report if p.name=='report.json' else prefix
    with patch.object(e,'read',side_effect=reader):
     self.assertTrue(e.completed_job(path,'guards')['publication_gap'])
     for field,value in (('generation',False),('ownership_passed',1),('hold',False),('registry',{}),('extra',True)):
      original=copy.deepcopy(prefix);prefix[field]=value
      with self.subTest(field=field),self.assertRaises(Exception):e.completed_job(path,'guards')
      prefix.clear();prefix.update(original)
 def test_fit_manifest_exact_flags_paths_and_membership(self):
  path=Path('frozen-labels');rows=[dict(state_key='a',is_case=True),dict(state_key='b',is_case=False)]
  with patch.object(e,'_ref',side_effect=lambda p:dict(path=str(p),sha256='sha')):
   expected=e._fit_manifest_document(path,rows)
   for field,value in (('gradients_from_these_rows',0),('roots',True),('rows_ref',{}),('cases',True),('extra',True)):
    bad=copy.deepcopy(expected);bad[field]=value
    self.assertFalse(e._same(bad,expected))


if __name__=='__main__':unittest.main()
