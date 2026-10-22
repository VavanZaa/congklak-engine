"""Source-bound prospective native guards; predecessor code and files stay frozen.

The native ABI and staged durable ownership protocol are reused verbatim. Only
constructor admission and the immutable complete component map are prospective.
"""
from dataclasses import dataclass
import ctypes as C
import hashlib,json,os,struct,uuid,weakref
from pathlib import Path
from types import MappingProxyType
from . import config, prospective_eval_common as stage
from . import prospective_eval_reservations as data, prospective_eval_annex as annex, ordinary_epoch_guard as old

ROOT=stage.ROOT
DEFAULT=stage.FOLDER/'guard'
SHARED_MANIFEST=data.DEFAULT/'report.json'
SHARED_RECORDS=data.DEFAULT/'components.bin'
SCHEMA='prospective-eval-v1-packed-guard'
STATIC,OWNED,VIOLATION,EVENT=old.STATIC,old.OWNED,old.VIOLATION,old.EVENT
EXPLICIT_FORBIDDEN,UNPERMITTED_COMPONENT,OTHER_LIVE_ORIGIN=1,2,4
PackedClaims=old.PackedClaims
key,raw_key,owned_rows=old.key,old.raw_key,old.owned_rows
SOURCES=('congklak/prospective_eval_guard.py','congklak/prospective_eval_reservations.py','congklak/prospective_eval_annex.py',
         'congklak/prospective_eval_common.py','congklak/ordinary_epoch_guard.py','congklak/config.py')
need,read,sha=stage.need,stage.read,stage.sha
_SEALS=weakref.WeakKeyDictionary()


def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return dict(path=stage.relative(path),sha256=sha(path))
def registry():return data.registry()
def prepare_shared():return registry()

def spec(origin,registry=None,batch=None):
    r=registry or globals()['registry']()
    inputs=dict(batch.inputs) if batch is not None else stage.common.pins(
        [stage.PLAN,data.DEFAULT/'registry.json',SHARED_MANIFEST,SHARED_RECORDS,annex.POLICY])
    need(origin in r['origin_components'],'Unregistered guard origin')
    return dict(guard=dict(graph_path=r['graph']['path'],allowed_components=[r['origin_components'][origin]],
        forbidden_quotients=[],epoch_owned_component_only=True,owner_origin=origin),
        schedule=[dict(origin=o) for o in r['origins']],inputs=inputs)

class Lookup:
    """Literal binary-search q -> CID reference over the one immutable buffer."""
    def __init__(self,records,ordinals):
        self.records=records;self.count=len(records)//19;self.components={v:k for k,v in ordinals.items() if v}
    def get(self,text,default=None):
        q=key(text,True);lo=0;hi=self.count
        while lo<hi:
            mid=(lo+hi)//2;found=self.records[mid*19:mid*19+15]
            if found<q:lo=mid+1
            else:hi=mid
        if lo==self.count or self.records[lo*19:lo*19+15]!=q:return default
        return self.components[struct.unpack_from('<I',self.records,lo*19+15)[0]]

class AdmissionBatch:
    """One freshly verified shared source snapshot per finite caller batch."""
    def __setattr__(self,name,value):
        need(not getattr(self,'_sealed',False),'Guard admission snapshot is immutable');object.__setattr__(self,name,value)
    def __init__(self,spec):
        self.inputs=MappingProxyType(dict(spec['inputs']));self.source_inputs=MappingProxyType(stage.common.pins(SOURCES))
        stage.common.current(self.inputs);self.graph_path=spec['guard']['graph_path']
        need(self.graph_path==stage.relative(SHARED_MANIFEST),'Guard shared manifest path differs')
        manifest=read(SHARED_MANIFEST);self.manifest=MappingProxyType(manifest)
        report=data.verify(full=True)
        self.annex_policy=MappingProxyType(annex.verify())
        self.eligible_components=frozenset(self.annex_policy['eligible_components'])
        need(manifest==report and manifest['schema']==data.SCHEMA+'-report' and
             manifest['policy']['complete_predecessor_component_unions'] is True and
             manifest['policy']['historically_unseen'] is False,'Prospective mapping admission differs')
        ordinals=report['component_ordinals'];records=SHARED_RECORDS.read_bytes()
        need(len(records)==manifest['record_count']*19 and
             hashlib.sha256(records).hexdigest()==manifest['shared_records_ref']['sha256'] and
             manifest['shared_records_ref']['path']==stage.relative(SHARED_RECORDS),
             'Prospective complete guard record bytes changed')
        self.records=records;self.buffer=(C.c_uint8*len(records)).from_buffer_copy(records)
        self.component_ordinals=MappingProxyType(ordinals);self.components=frozenset(ordinals)
        r=registry();self.origin_components=MappingProxyType(r['origin_components']);self.origins=tuple(r['origins'])
        need(tuple(sorted({r['origin'] for r in spec['schedule']}))==self.origins,'Caller owner registry differs')
        self.graph_lookup=Lookup(records,ordinals);self.graph_sha256=sha(SHARED_MANIFEST)
        self.closed=False;self._projections={};self._sealed=True
        _SEALS[self]=(self.records,self.buffer,self.component_ordinals,self.origin_components,self.inputs,self.graph_lookup,False,
            self.annex_policy,self.eligible_components)
    def assert_sealed(self,require_closed=False):
        seal=_SEALS.get(self)
        need(type(self) is AdmissionBatch and seal is not None and
             all(getattr(self,n) is v for n,v in zip(('records','buffer','component_ordinals','origin_components','inputs','graph_lookup'),seal[:6])) and
             self.closed is seal[6] and self.annex_policy is seal[7] and self.eligible_components is seal[8] and
             (not require_closed or self.closed is True),'Guard admission seal differs')
        return True
    def assert_spec(self,spec):
        self.assert_sealed();g=spec['guard']
        need(set(g)=={'graph_path','allowed_components','forbidden_quotients','epoch_owned_component_only','owner_origin'} and
             g['graph_path']==self.graph_path and g['forbidden_quotients']==[] and g['epoch_owned_component_only'] is True and
             self.origin_components.get(g['owner_origin']) in g['allowed_components'] and len(g['allowed_components'])==1 and
             spec['inputs']==dict(self.inputs) and tuple(sorted({r['origin'] for r in spec['schedule']}))==self.origins,
             'Guard spec differs from exact admission snapshot')
    def verify_current(self):
        self.assert_sealed();stage.common.current(self.inputs);stage.common.current(self.source_inputs)
        stage.common.current(self.manifest['source_inputs']);object.__setattr__(self,'closed',True)
        seal=_SEALS[self];_SEALS[self]=(*seal[:6],True,*seal[7:]);return True
    def projection(self,cid):
        self.assert_sealed();need(cid in self.component_ordinals,'Unknown component projection')
        ordinal=self.component_ordinals[cid]
        if ordinal not in self._projections:
            self._projections[ordinal]=hashlib.sha256(self.projected_records(cid)).hexdigest()
        return self._projections[ordinal]
    def projected_records(self,cid):
        allowed={self.component_ordinals[c] for c in self.eligible_components|{cid}}
        return annex.projected_records(self.records,allowed)

def _manifest(spec,batch):
    batch.assert_spec(spec);g=spec['guard'];origins=list(batch.origins);cid=g['allowed_components'][0]
    body=dict(schema=SCHEMA+'-manifest',rules_digest=config.RULES_DIGEST,guard=g,schedule_origins=origins,
        origin_ids={o:i+1 for i,o in enumerate(origins)},spec_guard_and_origins_sha256=config.digest(dict(guard=g,origins=origins)),
        graph_sha256=batch.graph_sha256,inputs=dict(batch.inputs,**dict(batch.source_inputs)),
        records=len(batch.records)//19,record_bytes=19,owned_record_bytes=40,violation_record_bytes=16,
        records_sha256=batch.projection(cid),shared_records_ref=_ref(SHARED_RECORDS),allowed_ordinal=batch.component_ordinals[cid],
        annex_policy_ref=_ref(annex.POLICY),static_projection='own component plus predeclared whole unused development pool',
        projected_records_not_persisted=True,owned_min=25,upper_states_retained=True,raw_stores_retained_in_trace=True,
        ownership_keys_are_not_value_cache_keys=True,claims_publish='atomic native staged-map publication after durable ownership-check commit',
        engineering_exception=False,training_admission=False,evaluation_admission=False,production_admission=False,
        models_fixed=True,policy_fixed=True)
    body['guard_stamp']=config.digest(body);return body

def prepare(spec,path=None,batch=None):
    own=batch is None;batch=AdmissionBatch(spec) if own else batch
    path=Path(path or DEFAULT/config.digest(spec['guard']['owner_origin'])).resolve()
    need(path.is_relative_to(DEFAULT),'Guard manifest outside new namespace')
    manifest=_manifest(spec,batch);stage.atomic(path/'manifest.json',manifest)
    if own:batch.verify_current()
    return dict(manifest_path=stage.relative(path/'manifest.json'),manifest_sha256=sha(path/'manifest.json'),
        records_path=stage.relative(SHARED_RECORDS),records_sha256=manifest['records_sha256'],guard_stamp=manifest['guard_stamp'])

def expected_refs(origin,registry=None,batch=None):
    r=registry or globals()['registry']();p=DEFAULT/config.digest(origin)/'manifest.json';m=read(p)
    refs=dict(manifest_path=stage.relative(p),manifest_sha256=sha(p),records_path=stage.relative(SHARED_RECORDS),
        records_sha256=m['records_sha256'],guard_stamp=m['guard_stamp'])
    load_pure(refs,spec(origin,r,batch),batch);return refs

@dataclass
class LoadedGuard:
    refs:dict
    manifest:dict
    records:bytes
    graph_lookup:object
    allowed:set
    forbidden:set
    expected_spec:dict
    admission_batch:object
    @property
    def origin_ids(self):return self.manifest['origin_ids']
    @property
    def guard_stamp(self):return self.manifest['guard_stamp']
    def owner_id(self,origin):
        need(origin in self.origin_ids,'Unknown live owner');return self.origin_ids[origin]
    def attach(self,native):return NativeGuard(native,self)
    claims_blob=old.LoadedGuard.claims_blob

def load_pure(refs,expected_spec,batch=None):
    need(type(refs) is dict and set(refs)=={'manifest_path','manifest_sha256','records_path','records_sha256','guard_stamp'},
         'Guard reference shape differs')
    path=(ROOT/refs['manifest_path']).resolve()
    need(path.is_relative_to(DEFAULT) and path.name=='manifest.json' and sha(path)==refs['manifest_sha256'] and
         refs['records_path']==stage.relative(SHARED_RECORDS),'Guard manifest/source path changed')
    own=batch is None;batch=AdmissionBatch(expected_spec) if own else batch
    expected=_manifest(expected_spec,batch);manifest=read(path)
    need(manifest==expected and refs['records_sha256']==manifest['records_sha256'] and
         refs['guard_stamp']==manifest['guard_stamp'],'Guard projection no longer follows admitted source')
    if own:batch.verify_current()
    return LoadedGuard(dict(refs),manifest,batch.records,batch.graph_lookup,set(expected_spec['guard']['allowed_components'])|set(batch.eligible_components),
        set(),json.loads(config.canonical(expected_spec)),batch)

reference_owned=old.reference_owned
reference_trace=old.reference_trace
reference_raw=old.reference_raw
derive_annex=annex.derive

def stage_reference(owned,loaded,origin,claimed,previous_claim_hash,generation,previous_generation):
    need(type(generation) is int and type(previous_generation) is int and 0<=previous_generation<generation<2**64,
         'Stage generation must advance')
    try:previous=bytes.fromhex(previous_claim_hash)
    except (ValueError,TypeError) as error:raise ValueError('Malformed previous claim hash') from error
    need(len(previous)==32 and previous.hex()==previous_claim_hash,'Previous claim hash width differs')
    need(not reference_owned(owned,loaded,origin,claimed),'Cannot stage failed ownership check')
    owner=loaded.owner_id(origin);delta=[]
    for row in owned_rows(owned):
        q=bytes.fromhex(row['quotient']);old_origin=claimed.get(q.hex());prior=None if old_origin is None else loaded.owner_id(old_origin)
        need(prior in (None,owner),'Cannot stage another live origin')
        if prior is None:delta.append(STATIC.pack(q,owner))
    blob=b''.join(delta);delta_hash=hashlib.sha256(blob).digest()
    staged=hashlib.sha256(previous+delta_hash+struct.pack('<QIQ',generation,owner,len(delta))).hexdigest()
    return dict(previous_claim_hash=previous_claim_hash,stage_delta_hash=delta_hash.hex(),staged_claim_hash=staged,
        staged_total_claims=len(claimed)+len(delta),stage_new_count=len(delta),generation=generation,
        previous_generation=previous_generation,owner=owner,
        canonical_restored_claim_hash=claimed.sha256 if isinstance(claimed,PackedClaims) else hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),
        delta_records=blob,guard_records_sha256=loaded.refs['records_sha256'])

class NativeGuard:
    def __init__(self,native,loaded):
        need(loaded.expected_spec is not None and type(loaded.admission_batch) is AdmissionBatch,
             'Native guard requires source-bound admission')
        loaded.admission_batch.assert_sealed(require_closed=True)
        verified=load_pure(loaded.refs,loaded.expected_spec,loaded.admission_batch)
        need(loaded.records is verified.records and loaded.manifest==verified.manifest,'Loaded guard mutated before attach')
        self.lib=getattr(native,'lib',native);self.loaded=verified;self.closed=False
        self.run_identity=None;self.last_commit_index=-1;self.published_generation=0
        lib=self.lib;lib.cc_fast_guard_create.argtypes=[C.POINTER(C.c_uint8),C.c_uint64,C.c_uint32]
        lib.cc_fast_guard_create.restype=C.c_void_p
        projected=loaded.admission_batch.projected_records(loaded.manifest['guard']['allowed_components'][0])
        projected_buffer=(C.c_uint8*len(projected)).from_buffer_copy(projected)
        self.handle=lib.cc_fast_guard_create(projected_buffer,len(projected)//19,len(loaded.origin_ids))
        need(self.handle,'Native guard rejected complete role-aware component projection')
        lib.cc_fast_guard_destroy.argtypes=[C.c_void_p];lib.cc_fast_guard_destroy.restype=None
        for name in ('cc_fast_guard_restore','cc_fast_guard_update'):
            fn=getattr(lib,name);fn.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_uint64];fn.restype=C.c_int
        lib.cc_fast_guard_publish_stage.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_publish_stage.restype=C.c_int
        lib.cc_fast_guard_discard_stage.argtypes=[C.c_void_p];lib.cc_fast_guard_discard_stage.restype=None
        lib.cc_fast_guard_restore_generation.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_guard_restore_generation.restype=C.c_int
        for name in ('cc_fast_guard_stage_certificate','cc_fast_guard_status'):
            getattr(lib,name).argtypes=[C.c_void_p];getattr(lib,name).restype=C.c_char_p
        status=json.loads(lib.cc_fast_guard_status(self.handle).decode())
        if status.get('static_hash')!=loaded.refs['records_sha256']:
            self.close();raise ValueError('Native projection differs from manifest hash')
    restore=old.NativeGuard.restore
    discard_stage=old.NativeGuard.discard_stage
    close=old.NativeGuard.close
    __enter__=old.NativeGuard.__enter__
    __exit__=old.NativeGuard.__exit__

    def bind_run(self,identity_ref,last_commit_index=-1):
        need(not self.closed and self.run_identity is None,'Guard run already bound/closed')
        need(type(identity_ref) is dict and set(identity_ref)=={'path','sha256'} and type(last_commit_index) is int and
             last_commit_index>=-1,'Run identity shape/cursor differs')
        path=_proof_path(identity_ref['path']);need(path.is_file() and path.stat().st_size<=64*1024*1024,
            'Guard identity missing/oversized');payload=path.read_bytes()
        need(hashlib.sha256(payload).hexdigest()==identity_ref['sha256'],'Guard run identity hash differs')
        identity=json.loads(payload)
        need(identity.get('guard_refs_by_origin',{}).get(self.loaded.manifest['guard']['owner_origin'])==self.loaded.refs,
             'Run identity does not bind exact projection')
        self.run_identity=dict(path=path,sha256=identity_ref['sha256']);self.last_commit_index=last_commit_index

    def publish_stage(self,generation,origin,durable_ref):
        need(not self.closed and self.run_identity is not None,'Guard lacks durable run')
        need(type(durable_ref) is dict and set(durable_ref)=={'event_path','event_sha256','commit_path','commit_sha256','index'},
             'Durable ownership proof shape differs')
        need(type(generation) is int and self.published_generation<generation<2**64 and
             type(durable_ref['index']) is int and durable_ref['index']>self.last_commit_index,'Durable ownership cursor did not advance')
        ep=_proof_path(durable_ref['event_path']);cp=_proof_path(durable_ref['commit_path']);run=self.run_identity['path'].parent
        index=durable_ref['index'];need(cp==run/'commits'/f'{index:06d}.json','Commit path/cursor differs')
        commit=_proof_json(cp,durable_ref['commit_sha256'])
        need(commit.get('index')==index and commit.get('identity_sha256')==self.run_identity['sha256'] and
             commit.get('event_sha256')==durable_ref['event_sha256'] and
             commit.get('event')==f'events/{index:06d}-{durable_ref["event_sha256"]}.json' and ep==run/commit['event'],
             'Ownership commit binding differs')
        previous=run/'commits'/f'{index-1:06d}.json' if index else None
        need((previous is None and commit.get('previous_sha256') is None) or
             (previous is not None and previous.is_file() and previous.stat().st_size<=65536 and
              commit.get('previous_sha256')==sha(previous)),'Immediate predecessor differs')
        event=_proof_json(ep,durable_ref['event_sha256']);result=event.get('result',{});cert=result.get('fast_guard_certificate',{})
        owner=self.loaded.owner_id(origin)
        need(event.get('kind')=='ownership_check' and result.get('passed') is True and result.get('origin')==origin and
             cert.get('generation')==generation and cert.get('owner')==owner and
             cert.get('previous_generation')==self.published_generation and cert.get('guard_records_sha256')==self.loaded.refs['records_sha256'],
             'Durable check does not authorize native stage')
        active=self.lib.cc_fast_guard_stage_certificate(self.handle)
        need(type(active) is bytes and 0<len(active)<=8192,'Native stage certificate absent/oversized');active=json.loads(active)
        required={'generation','owner','previous_generation','previous_claim_hash','stage_delta_hash','staged_claim_hash',
                  'staged_total_claims','stage_new_count','guard_records_sha256'}
        need(type(active) is dict and required<=set(active) and all(cert.get(k)==v for k,v in active.items()),
             'Durable certificate differs from actual native stage')
        need(self.lib.cc_fast_guard_publish_stage(self.handle,generation,owner)==1,'Native publication rejected generation/origin')
        self.last_commit_index=index;self.published_generation=generation

def _proof_path(value):
    need(type(value) is str and value,'Durable proof requires a path');path=Path(value)
    path=(path if path.is_absolute() else ROOT/path).resolve()
    need(path.is_relative_to(stage.FOLDER),'Durable proof outside new pilot namespace');return path

def _proof_json(path,expected):
    need(path.is_file() and path.stat().st_size<=65536,'Durable proof missing/oversized');raw=path.read_bytes()
    need(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==expected,'Durable proof hash differs')
    value=json.loads(raw);need(type(value) is dict,'Durable proof must be an object');return value
