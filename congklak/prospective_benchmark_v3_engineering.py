"""Five fixed native guard conformance traces; no model forward or search."""
import base64,ctypes as C,hashlib,json,struct
from . import config,prospective_benchmark_v3_common as s,prospective_benchmark_v2_extension as data
from . import prospective_benchmark_v3_guard as guard
PROTECTED_Q='000100020301010001010d00020201'
def _state(q):
 q=bytes.fromhex(q);active=sum(q[:14]);return dict(pits=[list(q[:7]),list(q[7:14])],stores=[98-active,0],controller=q[14])
def specification():
 records=(data.DEFAULT/'components.bin').read_bytes();allowed=records[:15].hex()
 from .neurodiag_data import quotient
 unknown=quotient(_state(bytes([25]+[0]*13+[0]).hex()))
 lookup=guard.Lookup(records,data.registry()['component_ordinals'])
 s.need(lookup.get(unknown) is None and lookup.get(PROTECTED_Q) is None,'Fixed denied diagnostic unexpectedly listed')
 shared=data.membership()['exact'][0]['state']
 s.need(sum(map(sum,shared['pits']))<=24,'Shared fixture outside shared domain')
 return [dict(name='allowed-high',state=_state(allowed),expected_flags=0,seed_other_owner=False),
  dict(name='different-unlisted-high',state=_state(unknown),expected_flags=2,seed_other_owner=False),
  dict(name='actual-protected-prior-high',state=_state(PROTECTED_Q),expected_flags=2,seed_other_owner=False),
  dict(name='shared-low',state=shared,expected_flags=0,seed_other_owner=False),
  dict(name='allowed-high-other-live-owner',state=_state(allowed),expected_flags=4,seed_other_owner=True)]
def _claims(loaded,row):
 blob=b''
 if row['seed_other_owner']:
  from .neurodiag_data import quotient
  blob=guard.STATIC.pack(bytes.fromhex(quotient(row['state'])),loaded.owner_id(data.SHARED))
 return guard.PackedClaims(blob,loaded.origin_ids)
def validate(receipt,loaded):
 s.need(receipt['schema']==s.SCHEMA+'-guard-conformance' and receipt['specification']==specification() and receipt['model_queries']==receipt['searches']==0,'Diagnostic declaration changed')
 s.need(receipt['lifecycle_before']==receipt['lifecycle_after'],'Guard checks changed inference controls')
 s.need(len(receipt['rows'])==5,'Guard diagnostic count changed')
 for declared,row in zip(receipt['specification'],receipt['rows']):
  s.need(row['case']==declared,'Diagnostic treatment changed')
  raw=base64.b64decode(row['raw'],validate=True);owned=base64.b64decode(row['owned'],validate=True);violations=base64.b64decode(row['violations'],validate=True)
  from .native import flatten
  from .prospective_benchmark_v2_adapter import OPERATIONS
  states=list(guard.EVENT.iter_unpack(raw))
  s.need(len(states)==2 and all(cells==bytes(flatten(declared['state'])) and actor==declared['state']['controller'] and
   op==OPERATIONS[name] and scope==4 and reserved==0 and action==depth==-1 for
   (cells,actor,op,scope,reserved,action,depth),name in zip(states,('played_input','played_output'))),'Diagnostic raw fixture anchors differ')
  proof=guard.reference_raw(raw,loaded,data.OWNER,_claims(loaded,declared));cert=row['certificate']
  s.need(proof['event_count']==2 and proof['owned_records']==owned and proof['violation_records']==violations and
   proof['raw_sha256']==cert['event_sha256'] and proof['owned_sha256']==cert['owned_sha256'] and proof['violation_sha256']==cert['violations_sha256'],'Native/literal diagnostic complete bytes differ')
  flags=[flags for index,flags,other in guard.VIOLATION.iter_unpack(violations)]
  s.need(flags==([] if declared['expected_flags']==0 else [declared['expected_flags']]) and cert['guard_passed'] is (declared['expected_flags']==0) and
   cert['valid'] is True and cert['event_count']==cert['attempted_events']==2 and cert['overflow'] is False and cert['invalid'] is False,'Diagnostic native flags/counts differ')
 return True
def run(providers,loaded,path,milestone):
 path=s.path_of(path);s.need(not path.exists(),'Native diagnostic cannot replace prior evidence')
 rows=[];lib=providers.native.lib;before=providers.lifecycle()
 milestone('diagnostics_before_guard_attach')
 with loaded.attach(providers.native) as check:
  milestone('diagnostics_after_guard_attach')
  for index,declared in enumerate(specification()):
   milestone('diagnostics_before_case_restore',case=declared['name'])
   check.restore(_claims(loaded,declared),0);milestone('diagnostics_after_case_restore',case=declared['name'])
   milestone('diagnostics_before_trace_begin',case=declared['name'])
   s.need(lib.cc_ownership_trace_begin(2000000,4),'Diagnostic trace begin failed')
   milestone('diagnostics_after_trace_begin',case=declared['name'])
   try:
    providers.native.trace_mark(declared['state'],'played_input');milestone('diagnostics_after_played_input',case=declared['name'])
    providers.native.trace_mark(declared['state'],'played_output');milestone('diagnostics_after_played_output',case=declared['name'])
    milestone('diagnostics_before_finalize',case=declared['name'])
    cert=json.loads(lib.cc_fast_finalize(check.handle,loaded.owner_id(data.OWNER)).decode())
    milestone('diagnostics_after_finalize',case=declared['name'],certificate=cert)
    raw=C.string_at(lib.cc_ownership_trace_data(),cert['event_count']*24)
    owned=C.string_at(lib.cc_fast_owned_data(),cert['owned_count']*40) if cert['owned_count'] else b''
    violations=C.string_at(lib.cc_fast_violation_data(),cert['violations_count']*16) if cert['violations_count'] else b''
    rows.append(dict(case=declared,certificate=cert,raw=base64.b64encode(raw).decode(),owned=base64.b64encode(owned).decode(),violations=base64.b64encode(violations).decode()))
    case_path=path.parent/'cases'/f'{index:02d}-{declared["name"]}.json';s.atomic(case_path,rows[-1])
    milestone('diagnostics_after_snapshot_archive',case=declared['name'],archive_ref=s.ref(case_path))
   finally:
    check.discard_stage();lib.cc_ownership_trace_end()
    milestone('diagnostics_after_discard_and_trace_end',case=declared['name'])
 receipt=dict(schema=s.SCHEMA+'-guard-conformance',specification=specification(),rows=rows,model_queries=0,searches=0,
  lifecycle_before=before,lifecycle_after=providers.lifecycle(),source_identity=providers.identity)
 milestone('diagnostics_before_pure_validation')
 validate(receipt,loaded);milestone('diagnostics_after_pure_validation')
 s.atomic(path,receipt);milestone('diagnostics_after_receipt',receipt_ref=s.ref(path));return s.ref(path)
