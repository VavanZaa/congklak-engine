"""Acquisition-only use of the immutable native teacher and feature encoder.

The empty static guard is explicitly a recording workspace, never a reservation
permit. True source family/origin remains in the outer recorded query. A fresh
root-local table prevents acquisition order from rejecting later candidates.
"""
from contextlib import ExitStack
import base64,ctypes as C,hashlib,json,math,struct
from . import config,footprint_screen_common as s,ordinary_epoch_native as old
from . import ordinary_epoch_guard as geometry

SCHEMA='footprint-screen-v1-native'
TEACHER=dict(old.TEACHER)
SOURCES=('congklak/footprint_screen_native.py','congklak/footprint_screen_worker.py',
         'tools/footprint-screen-worker.py','tests/test_footprint_screen.py')
TraceHold=old.TraceHold
STAGE_KEYS=('generation','owner','previous_generation','previous_claim_hash','stage_delta_hash',
 'staged_claim_hash','stage_new_count','staged_total_claims','guard_records_sha256')
ZERO=hashlib.sha256(b'').hexdigest()
need=s.need


def same(a,b):return config.canonical(a)==config.canonical(b)
def source_identity():
 return dict(schema=SCHEMA+'-identity',native=old.source_identity(old.candidate_binding()),
  inputs=s.common.pins(SOURCES),teacher=TEACHER,features='captured native binary64/frozen127/control/zero forwards',
  guard_scope='acquisition only; empty static lookup; root-local owner1; no reservation admission',
  native_owner_is_source_family=False,models_fixed=True,policy_fixed=True,training_admission=False)


def input_pins():
 pins=dict(old.input_pins());pins.update(s.common.pins(SOURCES));return pins


class _Compiled:
 def __init__(self,origin):self.origin=origin
 def owner_id(self,origin):need(origin==self.origin,'Acquisition origin differs');return 1
 def attach(self,native):return LocalGuard(native,self.origin)


class LocalGuard:
 def __init__(self,native,origin):
  self.lib=native.lib;self.origin=origin;self.handle=self.lib.cc_fast_guard_create(None,0,1)
  need(self.handle,'Cannot allocate bounded acquisition workspace');self.closed=False
 def close(self):
  if not self.closed:self.lib.cc_fast_guard_destroy(self.handle);self.closed=True
 def restore(self,records,generation):
  need(not self.closed and type(records) is bytes and len(records)%19==0 and type(generation) is int and generation>=0,
   'Invalid root-local restore')
  raw=C.create_string_buffer(records) if records else None
  need(self.lib.cc_fast_guard_restore(self.handle,raw,len(records)//19)==1 and
   self.lib.cc_fast_guard_restore_generation(self.handle,generation)==1,'Acquisition root-local restore refused')
 def publish(self,certificate,proof):
  need(not self.closed,'Acquisition table is closed')
  event_path=s.path_of(proof['event_path']);commit_path=s.path_of(proof['commit_path'])
  need(s.sha(event_path)==proof['event_sha256'] and s.sha(commit_path)==proof['commit_sha256'],
   'Durable acquisition proof changed')
  event=s.read(event_path);commit=s.read(commit_path)
  need(event['kind']=='trace_commit' and same(event['stage'],{k:certificate[k] for k in STAGE_KEYS}) and
   commit['event_sha256']==proof['event_sha256'],'Local publication precedes its exact durable trace commit')
  actual=json.loads(self.lib.cc_fast_guard_stage_certificate(self.handle))
  need(all(same(actual[k],certificate[k]) for k in STAGE_KEYS),'Native stage differs from durable acquisition proof')
  need(self.lib.cc_fast_guard_publish_stage(self.handle,certificate['generation'],1)==1,
   'Acquisition stage publication refused')


class Session:
 def __init__(self,stack,pool):
  need(isinstance(stack,ExitStack),'Use scoped acquisition providers');s.admit_plan()
  actual=s.read(s.FOLDER/'pool.json');need(same(actual,pool),'Acquisition pool snapshot changed')
  self.pool=tuple(config.canonical(row) for row in pool['roots']);self.identity=source_identity()
  self.providers=old.open_providers(stack,capacity=TEACHER['capacity_events'],binding=old.candidate_binding())
  need(same(self.providers.identity,self.identity['native']),'Immutable native source identity differs')
  self.guard=None;self.meta=None
 def lifecycle(self):return self.providers.lifecycle()
 def open_root(self,meta,records=b'',generation=0):
  need(0<=meta['index']<len(self.pool) and config.canonical(meta)==self.pool[meta['index']],
   'Root does not belong to frozen acquisition pool')
  if self.guard is not None:self.guard.close()
  self.meta=dict(meta);self.guard=self.providers.attach_guard(_Compiled(meta['origin']))
  self.guard.restore(records,generation)
 def query(self,kind,meta,cursor,sink):
  need(self.meta is not None and same(meta,self.meta) and kind in ('teacher','features'),'Acquisition query scope differs')
  if kind=='teacher':return self.providers.teacher(meta['state'],cursor,meta['origin'],sink)
  return self.providers.probe('control',meta['state'],cursor,meta['origin'],'features',sink)
 def close(self):
  if self.guard is not None:self.guard.close();self.guard=None


def open_session(stack,pool):return Session(stack,pool)
def archive(transient):return old.archive_decision(transient)
def archive_failure(transient):return old.archive_attempt(transient)


def owned_records(answer):
 cert=answer['fast_certificate'];records=base64.b64decode(answer['fast_owned_high'],validate=True)
 need(len(records)==cert['owned_count']*40 and hashlib.sha256(records).hexdigest()==cert['owned_sha256'],
  'Recorded acquisition owned footprint changed')
 previous=None;keys=[]
 for q,index,cells,actor in geometry.OWNED.iter_unpack(records):
  need((previous is None or previous<q) and index<cert['event_count'] and
   geometry.raw_key(cells,actor)==q and sum(q[:14])>=25,'Malformed acquisition high footprint')
  previous=q;keys.append(q)
 return records,keys


def merge_claims(records,keys):
 """Root-local packed owner1 map, separate from future global reservations."""
 previous={q for q,owner in geometry.STATIC.iter_unpack(records) if owner==1}
 need(len(previous)==len(records)//19,'Malformed root-local claim map')
 added=sorted(set(keys)-previous);after=sorted(previous|set(keys))
 return b''.join(geometry.STATIC.pack(q,1) for q in after),b''.join(geometry.STATIC.pack(q,1) for q in added)


def stage_check(answer,claims,claim_hash,generation):
 _,keys=owned_records(answer);after,delta=merge_claims(claims,keys);cert=answer['fast_certificate']
 delta_hash=hashlib.sha256(delta).digest();next_generation=generation+1
 expected=dict(generation=next_generation,owner=1,previous_generation=generation,previous_claim_hash=claim_hash,
  stage_delta_hash=delta_hash.hex(),staged_claim_hash=hashlib.sha256(bytes.fromhex(claim_hash)+delta_hash+
   struct.pack('<QIQ',next_generation,1,len(delta)//19)).hexdigest(),stage_new_count=len(delta)//19,
  staged_total_claims=len(after)//19,guard_records_sha256=ZERO)
 need(all(same(cert[k],v) for k,v in expected.items()),'Root-local staged acquisition binding differs')
 return expected,after,keys


def capture_proof(answer,kind,meta,cursor,native_identity,previous_lifecycle):
 """Compact runtime certificate; independent full replay remains mandatory."""
 cert=answer['fast_certificate'];trace=answer['trace'];binding=trace['binding'];engine='H_v1' if kind=='teacher' else 'control'
 need(binding['engine']==engine and binding['decision_id']==cursor and same(binding['root_state'],meta['state']) and
  same(binding['provider_identity'],native_identity),'Acquisition raw binding differs')
 need(cert['valid'] is True and cert['guard_passed'] is True and cert['overflow'] is False and cert['invalid'] is False and
  cert['event_count']==cert['attempted_events']==trace['event_count'] and cert['violations_count']==0 and
  cert['raw_event_capacity']==16000000 and cert['unique_owned_capacity']==4000000 and cert['global_claim_capacity']==8000000,
  'Acquisition raw footprint incomplete or resource-capped')
 need(same(cert,trace['fast_certificate']) and trace['overflow'] is False and trace['invalid'] is False and
  trace['truncated'] is False and type(cert['event_count']) is int and cert['event_count']>0 and
  trace['attempted_events']==cert['attempted_events'] and trace['byte_count']==24*cert['event_count'] and
  trace['capacity_events']==16000000,'Acquisition raw metadata/certificate mismatch')
 need(answer.get('fast_error') is None and not answer.get('archival_errors') and
  trace['event_sha256']==cert['event_sha256'],'Acquisition archive invalid')
 ops=trace['operation_counts'];need(sum(ops.values())==cert['event_count'] and
  all(type(v) is int and v>=0 for v in cert['operation_counts']) and sum(cert['operation_counts'])==cert['event_count'] and
  cert['operation_counts'][0]==0 and
  ops=={name:cert['operation_counts'][i+1] for i,name in enumerate(old.frozen.OP_NAMES)},'Native capture partitions differ')
 need(ops['model_done']==ops['sidecar_done']==ops['model_eval']==ops['sidecar_eval']==0,
  'Acquisition performed a forbidden model forward')
 reset=answer['both_seat_reset'];life=answer['lifecycle']
 need(same(reset['before'],previous_lifecycle) and same(reset['after'],life['before']) and
  reset['source_identity_sha256']==config.digest(native_identity) and reset['decision_id']==cursor and
  reset['actual_controls_replaced']==2 and reset['after']['created']-reset['before']['created']==2 and
  reset['after']['destroyed']-reset['before']['destroyed']==2,'Acquisition cold-reset continuity differs')
 complete=False;teacher=None;features=None;reason=None
 if kind=='teacher':
  decision=answer['decision'];need(decision['requested_ms']==0 and decision['kind']=='h','Teacher budget/function differs')
  if decision['stop_reason']=='node_budget':reason='predeclared_node_cap'
  else:
   teacher=old.teacher_certificate(answer,meta['state']);complete=True
  need(life['after']['created']-life['before']['created']==1 and
   life['after']['destroyed']-life['before']['destroyed']==1,'Teacher native context lifetime differs')
  critical=dict(decision=decision,transition=answer['transition'],coverage=answer['coverage'],lifecycle=life,
   trace_sha256=cert['event_sha256'],certificate=cert,both_seat_reset=reset,binding=binding)
 else:
  old._probe_coverage(answer);features=answer['probe']['result']['features']
  need(len(features)==127 and all(type(x) in (int,float) and math.isfinite(x) for x in features),
   'Native acquisition features incomplete')
  complete=True;critical=dict(probe=answer['probe'],lifecycle=life,certificate=cert,binding=binding,both_seat_reset=reset)
 need(answer['fast_critical_sha256']==config.digest(critical),'Native compact critical binding changed')
 _,keys=owned_records(answer)
 return dict(schema=SCHEMA+'-capture-proof',root_id=meta['id'],index=meta['index'],family=meta['family'],origin=meta['origin'],
  cursor=cursor,kind=kind,trace_complete=True,math_complete=complete,ineligible_reason=reason,
  raw_sha256=cert['event_sha256'],raw_events=cert['event_count'],owned_sha256=cert['owned_sha256'],owned_keys=len(keys),
  native_identity_sha256=config.digest(native_identity),lifecycle_after=life['after'],
  teacher=teacher,features=features,model_forwards=0,reservation_admitted=False,full_raw_review_pending=True)


def full_raw_review(answer,kind,meta,cursor,native_identity):
 """Explicit expensive audit, separate from zero-query compact completion."""
 engine='H_v1' if kind=='teacher' else 'control'
 if kind=='teacher':old.validate_decision(answer,engine,meta['state'],cursor,native_identity)
 else:old.validate_probe(answer,engine,meta['state'],cursor,native_identity)
 return dict(passed=True,raw_sha256=answer['fast_certificate']['event_sha256'],full_owned_map_checked=True,
  queries=0,model_forwards=0)
