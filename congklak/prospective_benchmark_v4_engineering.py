"""Five fixed native guard conformance traces; no model forward or search."""
import base64,ctypes as C,hashlib,json,struct
from . import config,prospective_benchmark_v4_common as s,prospective_benchmark_v2_extension as data
from . import prospective_benchmark_v4_guard as guard
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
 evidence=reused_evidence()
 s.need(receipt['reused_evidence']==evidence and receipt['fresh_native_guard_traces']==0 and receipt['reused_native_guard_traces']==5 and
  receipt['source_identity']==s.read(s.ROOT/REUSED_PROVIDER_IDENTITY_REF['path'])['provider_identity'],'Reused native guard evidence/source identity differs')
 s.need(len(receipt['rows'])==5,'Guard diagnostic count changed')
 for declared,row in zip(receipt['specification'],receipt['rows']):
  s.need(row['case']==declared,'Diagnostic treatment changed')
  raw=base64.b64decode(row['raw'],validate=True);owned=base64.b64decode(row['owned'],validate=True);violations=base64.b64decode(row['violations'],validate=True)
  from .native import flatten
  from .prospective_benchmark_v2_adapter import OPERATIONS
  states=list(guard.EVENT.iter_unpack(raw))
  s.need(len(states)==4 and all(cells==bytes(flatten(declared['state'])) and actor==declared['state']['controller'] and
   op==OPERATIONS[name] and scope==4 and reserved==0 and action==depth==-1 for
   (cells,actor,op,scope,reserved,action,depth),name in zip(states,('decode','played_input','decode','played_output'))),'Diagnostic raw fixture anchors differ')
  proof=guard.reference_raw(raw,loaded,data.OWNER,_claims(loaded,declared));cert=row['certificate']
  s.need(cert['guard_records_sha256']==loaded.refs['records_sha256'],'Reused native static projection differs')
  s.need(proof['event_count']==4 and proof['owned_records']==owned and proof['violation_records']==violations and
   proof['raw_sha256']==cert['event_sha256'] and proof['owned_sha256']==cert['owned_sha256'] and proof['violation_sha256']==cert['violations_sha256'],'Native/literal diagnostic complete bytes differ')
  expected_counts=[0]*64
  for cells,actor,op,scope,reserved,action,depth in states:expected_counts[op]+=1
  s.need(cert['operation_counts']==expected_counts and cert['scope_mask']==16 and cert['played_input_index']==1 and cert['played_output_index']==3 and
   proof['operation_counts']=={name:expected_counts[op] for name,op in OPERATIONS.items()},'Diagnostic complete operation partition differs')
  flags=[flags for index,flags,other in guard.VIOLATION.iter_unpack(violations)]
  s.need(flags==([] if declared['expected_flags']==0 else [declared['expected_flags']]) and cert['guard_passed'] is (declared['expected_flags']==0) and
   cert['valid'] is True and cert['event_count']==cert['attempted_events']==4 and cert['overflow'] is False and cert['invalid'] is False,'Diagnostic native flags/counts differ')
 return True
REUSED_CASE_INPUTS={'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases/00-allowed-high.json': 'f200b51c57bdf0882f7da1cbdbff241d278ec8c0213f67e33ed79edcefc484cd', 'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases/01-different-unlisted-high.json': '5f82a588dfe72cc2cece2e42c7130254dc8f0ae77c1fe32de6e86182cab62b64', 'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases/02-actual-protected-prior-high.json': 'd1933a358b18ec81c2ca7ae30cc7dd641967a02e9509cb943dd371819e456de8', 'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases/03-shared-low.json': 'b0041fa4ec06b4bc66bb564bf3566575dd32f905c1452104feb23cbc8edcb526', 'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases/04-allowed-high-other-live-owner.json': '5a349f5a5edf82311b1479f51d412bf157c71ab00d0a0056b3b69338bbde1be9'}
REUSED_PROVIDER_IDENTITY_REF={'path': 'build/prospective-benchmark-v3/jobs/engineering/identity.json', 'sha256': '662974a7d166e656edc82f43ef17a7363b7fd141dca78499fbd0c991ae805cee'}
def reused_evidence():
 s.common.current(REUSED_CASE_INPUTS)
 s.need(s.ref(s.ROOT/REUSED_PROVIDER_IDENTITY_REF['path'])==REUSED_PROVIDER_IDENTITY_REF,'Reused provider identity changed')
 return dict(case_inputs=dict(REUSED_CASE_INPUTS),provider_identity_ref=dict(REUSED_PROVIDER_IDENTITY_REF),
  classification='previous actual native guard-only conformance; no fresh guard traces or engine queries')

def run(providers,loaded,path,milestone):
 path=s.path_of(path);s.need(not path.exists(),'Native diagnostic cannot replace prior evidence')
 before=providers.lifecycle();milestone('diagnostics_before_reused_evidence_validation')
 evidence=reused_evidence();old=s.read(s.ROOT/REUSED_PROVIDER_IDENTITY_REF['path'])
 s.need(providers.identity==old['provider_identity'],'Actual reused native/model source identity differs')
 rows=[s.read(s.ROOT/name) for name in sorted(REUSED_CASE_INPUTS)]
 receipt=dict(schema=s.SCHEMA+'-guard-conformance',specification=specification(),rows=rows,model_queries=0,searches=0,
  lifecycle_before=before,lifecycle_after=providers.lifecycle(),source_identity=providers.identity,
  reused_evidence=evidence,fresh_native_guard_traces=0,reused_native_guard_traces=5)
 validate(receipt,loaded);milestone('diagnostics_after_pure_validation',reused_evidence=evidence)
 s.atomic(path,receipt);milestone('diagnostics_after_receipt',receipt_ref=s.ref(path));return s.ref(path)
