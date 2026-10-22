"""Pure saved-byte/metadata checks. Native profile calls never query an engine."""
import base64,copy,ctypes as C,hashlib,json,math,struct,unittest,zlib
from contextlib import ExitStack
from unittest.mock import patch
from congklak import ordinary_epoch_native as n,ordinary_epoch_guard as g

EXPECT=n.ROOT/'build/tracing-speed-v1/audit/saved-trace-expectations/expectations.json'

class NativeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.expected=json.loads(EXPECT.read_text());cls.lib=C.CDLL(str(n.DLL));lib=cls.lib
  lib.cc_fast_profile_load.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_profile_load.restype=C.c_int
  lib.cc_epoch_profile_load_limited.argtypes=[C.c_void_p,C.c_uint64,C.c_uint64];lib.cc_epoch_profile_load_limited.restype=C.c_int
  lib.cc_fast_guard_create.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_create.restype=C.c_void_p
  lib.cc_fast_guard_destroy.argtypes=[C.c_void_p]
  lib.cc_fast_guard_discard_stage.argtypes=[C.c_void_p]
  lib.cc_fast_guard_restore.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64];lib.cc_fast_guard_restore.restype=C.c_int
  lib.cc_fast_guard_restore_generation.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_guard_restore_generation.restype=C.c_int
  lib.cc_fast_guard_status.argtypes=[C.c_void_p];lib.cc_fast_guard_status.restype=C.c_char_p
  lib.cc_fast_finalize.argtypes=[C.c_void_p,C.c_uint32];lib.cc_fast_finalize.restype=C.c_char_p
  lib.cc_fast_owned_data.restype=C.c_void_p;lib.cc_ownership_trace_data.restype=C.c_void_p
  lib.cc_ownership_controls_created.restype=C.c_uint64
  lib.cc_ownership_trace_begin.argtypes=[C.c_uint64,C.c_ubyte];lib.cc_ownership_trace_begin.restype=C.c_int
  cls.guard=lib.cc_fast_guard_create(None,0,1);assert cls.guard
  cls.forward_controls_before=int(lib.cc_ownership_controls_created())
 @classmethod
 def tearDownClass(cls):
  assert int(cls.lib.cc_ownership_controls_created())==cls.forward_controls_before==0
  cls.lib.cc_ownership_trace_end();cls.lib.cc_fast_guard_destroy(cls.guard)
 def _raw(self,entry):
  return n.trace_bytes(json.loads((n.ROOT/entry['source']).read_text())['trace'])
 def _profile(self,raw,limit=None):
  self.lib.cc_ownership_trace_end();self.lib.cc_fast_guard_discard_stage(self.guard)
  self.assertEqual(self.lib.cc_fast_guard_restore(self.guard,None,0),1)
  buffer=C.create_string_buffer(raw)
  okay=self.lib.cc_fast_profile_load(buffer,len(raw)//24) if limit is None else self.lib.cc_epoch_profile_load_limited(buffer,len(raw)//24,limit)
  cert=json.loads(self.lib.cc_fast_finalize(self.guard,1));owned=C.string_at(self.lib.cc_fast_owned_data(),cert['owned_count']*40)
  captured=C.string_at(self.lib.cc_ownership_trace_data(),cert['event_count']*24)
  return okay,cert,owned,captured
 def test_all13_saved_bytes_exact(self):
  for entry in self.expected['traces']:
   raw=self._raw(entry);okay,cert,owned,captured=self._profile(raw)
   self.assertEqual(okay,1);self.assertEqual(captured,raw);self.assertEqual(cert['event_sha256'],entry['raw_sha256'])
   self.assertEqual(cert['owned_keys_sha256'],entry['high_key_bytes_sha256'])
   self.assertEqual([q.hex() for q,*_ in g.OWNED.iter_unpack(owned)],entry['high_unique_keys'])
   for op,name in enumerate(n.frozen.OP_NAMES,1):self.assertEqual(cert['operation_counts'][op],entry['operation_counts'][name])
 def test_unique_failure_preserves_raw_and_never_publishes(self):
  entry=next(e for e in self.expected['traces'] if len(e['high_unique_keys'])>8);raw=self._raw(entry)
  okay,cert,owned,captured=self._profile(raw,8)
  self.assertEqual(okay,0);self.assertTrue(cert['unique_owned_overflow']);self.assertFalse(cert['valid']);self.assertFalse(cert['guard_passed'])
  self.assertEqual(cert['unique_owned_capacity'],8);self.assertEqual(cert['owned_count'],8)
  self.assertGreater(len(captured),0);self.assertEqual(captured,raw[:len(captured)])
  status=json.loads(self.lib.cc_fast_guard_status(self.guard));self.assertEqual(status['claims_count'],0);self.assertEqual(status['active_stage'],0)
 def test_owned_omission_and_first_event_refuse(self):
  entry=next(e for e in self.expected['traces'] if len(e['high_unique_keys'])>8)
  body=json.loads((n.ROOT/entry['source']).read_text());raw=n.trace_bytes(body['trace'])
  _,_,owned,_=self._profile(raw)
  self.assertTrue(n._validate_owned(body['trace'],owned))
  with self.assertRaises(Exception):n._validate_owned(body['trace'],b'')
  with self.assertRaises(Exception):n._validate_owned(body['trace'],owned[40:])
  bad=bytearray(owned);struct.pack_into('<Q',bad,15,struct.unpack_from('<Q',owned,15)[0]+1)
  with self.assertRaises(Exception):n._validate_owned(body['trace'],bytes(bad))
 def test_capacity_preflight_is_nonmutating(self):
  self.lib.cc_ownership_trace_end();self.lib.cc_fast_guard_discard_stage(self.guard)
  self.assertEqual(self.lib.cc_fast_guard_restore(self.guard,None,0),1)
  before=self.lib.cc_fast_guard_status(self.guard)
  self.assertEqual(self.lib.cc_fast_guard_restore(self.guard,None,8000001),0)
  self.assertEqual(self.lib.cc_fast_guard_status(self.guard),before)
  self.assertEqual(self.lib.cc_ownership_trace_begin(16000001,1),0)
  self.assertEqual(self.lib.cc_ownership_trace_begin(0,1),0)
 def test_packed_claims_exact_and_atomic(self):
  entry=next(e for e in self.expected['traces'] if len(e['high_unique_keys'])>8)
  keys=[bytes.fromhex(q) for q in entry['high_unique_keys'][:8]];ids={'fit/a':1,'fit/b':2}
  a=g.PackedClaims(b''.join(g.STATIC.pack(q,1) for q in keys[::2]),ids)
  b=a.merge_delta(b''.join(g.STATIC.pack(q,2) for q in keys[1::2]))
  self.assertEqual(len(b),8);self.assertIs(b.merge_delta(b''),b)
  for i,q in enumerate(keys):self.assertEqual(b.get(q),'fit/a' if i%2==0 else 'fit/b')
  before=b.records
  with self.assertRaises(Exception):b.merge_delta(g.STATIC.pack(keys[0],2))
  self.assertEqual(b.records,before)
 def test_exception_and_per_binding_reader_no_engine(self):
  with patch.object(C,'CDLL',side_effect=AssertionError('engine/DLL construction')):
   binding=n.candidate_binding();core=n._configured(binding)
   self.assertIs(core,n._configured(binding));self.assertIs(core.TraceHold,n.TraceHold)
   payload={'raw_trace':b'known-prefix'}
   try:raise core.TraceHold('held',payload)
   except n.TraceHold as exc:self.assertIs(exc.transient,payload)
   with ExitStack() as stack:
    with self.assertRaises(Exception):n.open_providers(stack)
 def test_original_complete_H8_certificate_and_mutants(self):
  from congklak import ordinary_epoch_data as data,neurodiag_data as old
  meta=data.membership()[0]['metadata'];db=old.ro('deeper-9000-v2')
  try:item=json.loads(db.execute('select body from labels where state=?',(meta['state_key'],)).fetchone()[0])
  finally:db.close()
  report=item['report'];answer=dict(decision=dict(report=report,nodes=report['nodes'],stop_reason='max_depth'),trace=dict(event_count=0))
  result=n.teacher_certificate(answer,meta['state']);self.assertEqual(result['completed_depth'],8)
  for change in ('depth','missing_action','partial_action','cap'):
   bad=copy.deepcopy(answer)
   if change=='depth':bad['decision']['report']['choice_depth']=7
   elif change=='missing_action':bad['decision']['report']['candidates'].pop()
   elif change=='partial_action':bad['decision']['report']['candidates'][0]['analysis_complete']=False
   else:bad['decision']['stop_reason']='node_budget'
   with self.assertRaises(Exception):n.teacher_certificate(bad,meta['state'])

if __name__=='__main__':unittest.main()
