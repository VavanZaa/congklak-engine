"""Source-bound packed ownership guards and independent off-clock references.

Only canonical15-byte ownership keys enter this lookup.  Raw stores remain in
the complete trace and owned40 records; this is never a value cache.  Native
finalization stages claims inside the whole decision clock.  A caller may
publish that stage only after its ownership-check record is durably committed.
"""
from dataclasses import dataclass
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import uuid
from . import config,neurodiag_common as common

ROOT=common.ROOT
DEFAULT=ROOT/'build/tracing-speed-v1/guard'
SCHEMA='packed-ownership-guard-v1'
EXPLICIT_FORBIDDEN=1
UNPERMITTED_COMPONENT=2
OTHER_LIVE_ORIGIN=4
STATIC=struct.Struct('<15sI')
OWNED=struct.Struct('<15sQ16sB')
VIOLATION=struct.Struct('<QII')
EVENT=struct.Struct('<16sBBBBhh')
SOURCES=('congklak/ownership_fast_guard.py','congklak/config.py',
         'congklak/neurodiag_common.py','congklak/divergence_study_common.py')
need,read,sha=common.require,common.read,common.sha


def key(value,owned_only=False):
    if type(value) is str:
        try:raw=bytes.fromhex(value)
        except ValueError as error:raise ValueError('Malformed ownership key') from error
        need(raw.hex()==value,'Ownership key must use canonical lowercase hexadecimal')
    else:
        need(type(value) is bytes,'Ownership key requires bytes or hex text');raw=value
    need(len(raw)==15 and raw[14] in (0,1) and sum(raw[:14])<=98,
         'Ownership key geometry differs')
    active=sum(raw[:14]);other=raw[7:14]+raw[:7]+bytes([1-raw[14] if active else 0])
    need(raw<=other and (active or raw[14]==0),'Noncanonical ownership key')
    need(not owned_only or active>=25,'Shared low key cannot enter owned lookup or claims')
    return raw


def raw_key(cells,actor):
    need(type(cells) is bytes and len(cells)==16 and sum(cells)==98 and
         type(actor) is int and actor in (0,1),'Ownership event is not a complete98-seed state')
    pits=cells[:7]+cells[8:15];a=pits+bytes([actor]);b=pits[7:]+pits[:7]+bytes([1-actor if any(pits) else 0])
    return min(a,b)


def _publish(path,payload):
    path=Path(path)
    if path.exists():need(path.read_bytes()==payload,'Immutable fast guard artifact differs: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def _static(guard,graph):
    need(set(guard)=={'graph_path','allowed_components','forbidden_quotients','engineering_consumed_only'} and
         guard['engineering_consumed_only'] is True,'Only the frozen consumed-state engineering guard is admitted')
    lookup=graph['quotient_components'];allowed=set(guard['allowed_components'])
    need(len(allowed)==len(guard['allowed_components']) and all(type(v) is str for v in allowed) and
         allowed<=set(graph['components']),'Unknown/duplicate allowed component')
    flags={};shared=set()
    for text,cid in lookup.items():
        q=key(text);need(cid in graph['components'],'Ownership graph component missing')
        if sum(q[:14])<25:shared.add(text);continue
        flags[q]=0 if cid in allowed else UNPERMITTED_COMPONENT
    forbidden=set()
    for text in guard['forbidden_quotients']:
        q=key(text);forbidden.add(q)
        if sum(q[:14])<25:shared.add(text);continue
        flags[q]=flags.get(q,0)|EXPLICIT_FORBIDDEN
    records=b''.join(STATIC.pack(q,flags[q]) for q in sorted(flags))
    return records,flags,forbidden,allowed,sorted(shared)


def prepare(spec,path=DEFAULT):
    """Compile once off-clock; the original graph/spec/source files stay fixed."""
    path=Path(path).resolve();need(path.is_relative_to(DEFAULT),'Fast guard output outside new guard namespace')
    guard=spec['guard'];relative=guard['graph_path'];graph_path=(ROOT/relative).resolve()
    need(graph_path.is_relative_to(ROOT) and relative in spec['inputs'] and
         sha(graph_path)==spec['inputs'][relative],'Fast guard graph lacks exact source binding')
    common.current(spec['inputs'])
    origins=sorted({row['origin'] for row in spec['schedule']})
    need(origins and all(type(v) is str and v for v in origins) and len(origins)<2**32,
         'Frozen schedule origin identities invalid')
    graph=read(graph_path);records,flags,forbidden,allowed,shared=_static(guard,graph)
    origin_ids={origin:index+1 for index,origin in enumerate(origins)}
    inputs=dict(spec['inputs']);inputs.update(common.pins(SOURCES))
    manifest=dict(schema=SCHEMA+'-manifest',rules_digest=config.RULES_DIGEST,
        guard=guard,schedule_origins=origins,origin_ids=origin_ids,
        spec_guard_and_origins_sha256=config.digest(dict(guard=guard,origins=origins)),
        graph_sha256=sha(graph_path),inputs=inputs,record_bytes=19,owned_record_bytes=40,
        violation_record_bytes=16,records=len(flags),records_sha256=hashlib.sha256(records).hexdigest(),
        static_flags=dict(explicit_forbidden=1,unpermitted_component=2),dynamic_flags=dict(other_live_origin=4),
        source_shared_keys_retained_outside_lookup=shared,owned_min=25,upper_states_retained=True,
        raw_stores_retained_in_trace=True,ownership_keys_are_not_value_cache_keys=True,
        claims_publish='Only atomic native staged-map publication after durable ownership-check commit',
        engineering_exception=True,training_admission=False,evaluation_admission=False,
        production_admission=False,models_fixed=True,policy_fixed=True)
    manifest['guard_stamp']=config.digest(manifest)
    _publish(path/'records.bin',records)
    _publish(path/'manifest.json',(config.canonical(manifest)+'\n').encode())
    return dict(manifest_path=(path/'manifest.json').relative_to(ROOT).as_posix(),
                manifest_sha256=sha(path/'manifest.json'),records_path=(path/'records.bin').relative_to(ROOT).as_posix(),
                records_sha256=sha(path/'records.bin'),guard_stamp=manifest['guard_stamp'])


@dataclass
class LoadedGuard:
    refs:dict
    manifest:dict
    records:bytes
    flags:dict
    graph_lookup:dict
    allowed:set
    forbidden:set
    expected_spec:dict|None=None
    @property
    def origin_ids(self):return self.manifest['origin_ids']
    @property
    def guard_stamp(self):return self.manifest['guard_stamp']
    def owner_id(self,origin):
        need(origin in self.origin_ids,'Unregistered live origin');return self.origin_ids[origin]
    def attach(self,native):return NativeGuard(native,self)
    def claims_blob(self,claimed):
        rows={}
        for q,origin in claimed.items():
            packed=key(q,True);owner=self.owner_id(origin)
            need(packed not in rows or rows[packed]==owner,'Canonical live key has conflicting owners')
            rows[packed]=owner
        return b''.join(STATIC.pack(q,rows[q]) for q in sorted(rows))


def load_pure(refs,expected_spec):
    """Validate graph/lookup metadata and bytes before creating any engine."""
    need(type(expected_spec) is dict and type(refs) is dict and set(refs)==
         {'manifest_path','manifest_sha256','records_path','records_sha256','guard_stamp'},
         'Fast guard requires exact frozen caller spec and reference shape')
    path=(ROOT/refs['manifest_path']).resolve();records_path=(ROOT/refs['records_path']).resolve()
    need(path.is_relative_to(DEFAULT) and records_path==path.parent/'records.bin' and
         sha(path)==refs['manifest_sha256'] and sha(records_path)==refs['records_sha256'],
         'Fast guard immutable artifact hash/path differs')
    manifest=read(path);need(manifest['schema']==SCHEMA+'-manifest' and
        manifest['rules_digest']==config.RULES_DIGEST and manifest['record_bytes']==STATIC.size==19 and
        manifest['owned_record_bytes']==OWNED.size==40 and manifest['violation_record_bytes']==VIOLATION.size==16,
        'Fast guard schema/rules/layout differs')
    body=dict(manifest);stamp=body.pop('guard_stamp')
    need(config.digest(body)==stamp==refs['guard_stamp'],'Fast guard content stamp differs')
    need(manifest['owned_min']==25 and manifest['upper_states_retained'] is True and
         manifest['production_admission'] is False and manifest['training_admission'] is False and
         manifest['evaluation_admission'] is False and manifest['models_fixed'] is True and manifest['policy_fixed'] is True,
         'Fast guard scope/holds differ')
    origins=sorted({row['origin'] for row in expected_spec['schedule']})
    need(manifest['guard']==expected_spec['guard'] and manifest['schedule_origins']==origins and
         manifest['spec_guard_and_origins_sha256']==config.digest(dict(guard=expected_spec['guard'],origins=origins)),
         'Fast guard differs from caller frozen guard/origin authorization')
    need(all(manifest['inputs'].get(name)==checksum for name,checksum in expected_spec['inputs'].items()) and
         all(manifest['inputs'].get(name)==sha(ROOT/name) for name in SOURCES),
         'Fast guard lacks required caller/current implementation source pins')
    common.current(manifest['inputs']);graph_path=ROOT/manifest['guard']['graph_path']
    need(manifest['inputs'].get(manifest['guard']['graph_path'])==manifest['graph_sha256']==sha(graph_path),
         'Fast guard source graph changed/unbound')
    graph=read(graph_path);expected,flags,forbidden,allowed,shared=_static(manifest['guard'],graph)
    records=records_path.read_bytes()
    need(records==expected and hashlib.sha256(records).hexdigest()==manifest['records_sha256']==refs['records_sha256'] and
         len(records)==manifest['records']*STATIC.size and shared==manifest['source_shared_keys_retained_outside_lookup'],
         'Fast guard compilation/source relation differs')
    origins=manifest['schedule_origins'];need(origins==sorted(set(origins)) and
        manifest['origin_ids']=={origin:index+1 for index,origin in enumerate(origins)},'Fast guard frozen owner map differs')
    return LoadedGuard(refs,manifest,records,flags,graph['quotient_components'],allowed,forbidden,
                       json.loads(config.canonical(expected_spec)))


def owned_rows(data,event_count=None):
    need(type(data) is bytes and len(data)%OWNED.size==0,'Packed owned footprint width differs')
    out=[];previous=None
    for index,(q,first,cells,actor) in enumerate(OWNED.iter_unpack(data)):
        key(q,True);need(previous is None or previous<q,'Owned footprint keys duplicate/unsorted')
        need(raw_key(cells,actor)==q,'Owned footprint raw state/key differs')
        need(event_count is None or first<event_count,'Owned first-event index outside trace')
        out.append(dict(index=index,quotient=q.hex(),first_event=first,raw_state_key=(cells+bytes([actor])).hex(),
                        active_seeds=sum(q[:14])));previous=q
    return out


def reference_owned(data,loaded,origin,claimed):
    """Independent dictionary guard semantics, deliberately outside live clock."""
    loaded.claims_blob(claimed)  # Validate aliases and every registered origin, including absent rows.
    owner=loaded.owner_id(origin);rows=owned_rows(data);violations=[]
    for row in rows:
        q=bytes.fromhex(row['quotient']);flags=0;other=0
        if q in loaded.forbidden:flags|=EXPLICIT_FORBIDDEN
        cid=loaded.graph_lookup.get(q.hex())
        if cid is not None and cid not in loaded.allowed:flags|=UNPERMITTED_COMPONENT
        prior=claimed.get(q.hex(),claimed.get(q))
        if prior is not None:
            other=loaded.owner_id(prior)
            if other!=owner:flags|=OTHER_LIVE_ORIGIN
            else:other=0
        if flags:violations.append(VIOLATION.pack(row['index'],flags,other))
    return b''.join(violations)


def reference_trace(trace,loaded,origin,claimed):
    """Strict saved-trace reader; malformed/incomplete evidence holds."""
    from . import native_ownership_trace as codec
    need(trace.get('overflow') is False and trace.get('invalid') is False and trace.get('truncated') is False and
         trace.get('attempted_events')==trace.get('event_count'),'Incomplete ownership trace: HOLD')
    result=reference_raw(codec.trace_bytes(trace),loaded,origin,claimed)
    need(result['operation_counts']==trace['operation_counts'],'Saved operation-count partition differs')
    return result


def reference_raw(raw,loaded,origin,claimed):
    """Literal packed-event compiler, suitable for off-clock native conformance."""
    from . import native_ownership_trace as codec
    need(type(raw) is bytes and len(raw)%EVENT.size==0,'Packed complete-state event width differs')
    first={};owned_events=0;operation_counts=[0]*(len(codec.OP_NAMES)+1)
    for index,(cells,actor,operation,scope,reserved,action,depth) in enumerate(EVENT.iter_unpack(raw)):
        need(actor in (0,1) and 1<=operation<=len(codec.OP_NAMES) and scope in codec.SCOPES and
             reserved==0 and action>=-1 and depth>=-1,'Malformed complete-state event metadata')
        q=raw_key(cells,actor);operation_counts[operation]+=1
        if sum(q[:14])<25:continue
        owned_events+=1
        if q not in first:first[q]=(index,cells,actor)
    owned=b''.join(OWNED.pack(q,*first[q]) for q in sorted(first))
    violations=reference_owned(owned,loaded,origin,claimed)
    before=loaded.claims_blob(claimed);after=dict(claimed)
    if not violations:
        for q in first:after[q.hex()]=origin
    return dict(schema=SCHEMA+'-literal-reference',event_count=len(raw)//EVENT.size,
        operation_counts={name:operation_counts[index+1] for index,name in enumerate(codec.OP_NAMES)},
        raw_sha256=hashlib.sha256(raw).hexdigest(),owned_event_count=owned_events,owned_unique_count=len(first),
        owned_records=owned,owned_sha256=hashlib.sha256(owned).hexdigest(),
        violation_records=violations,violation_count=len(violations)//VIOLATION.size,
        violation_sha256=hashlib.sha256(violations).hexdigest(),passed=not violations,
        claims_before_sha256=hashlib.sha256(before).hexdigest(),
        claims_after_sha256=hashlib.sha256(loaded.claims_blob(after)).hexdigest(),
        guard_stamp=loaded.guard_stamp,origin=origin,owner_id=loaded.owner_id(origin),
        low_complete_events_retained=len(raw)//EVENT.size-owned_events,
        model_forwards=0,searches=0,new_games=0,production_admission=False)


def stage_reference(owned,loaded,origin,claimed,previous_claim_hash,generation,previous_generation):
    """Recompute native staged-map commitments without publishing live claims."""
    need(type(generation) is int and type(previous_generation) is int and
         0<=previous_generation<generation<2**64,'Stage generation must advance')
    try:previous=bytes.fromhex(previous_claim_hash)
    except (ValueError,TypeError) as error:raise ValueError('Malformed previous claim hash') from error
    need(len(previous)==32 and previous.hex()==previous_claim_hash,'Previous claim hash width/case differs')
    need(not reference_owned(owned,loaded,origin,claimed),'Cannot stage a failed ownership check')
    owner=loaded.owner_id(origin);rows=owned_rows(owned)
    normalized={key(q,True):loaded.owner_id(value) for q,value in claimed.items()}
    delta=[]
    for row in rows:
        q=bytes.fromhex(row['quotient']);prior=normalized.get(q)
        need(prior in (None,owner),'Cannot stage claims for another live origin')
        if prior is None:delta.append(STATIC.pack(q,owner))
    blob=b''.join(delta);delta_hash=hashlib.sha256(blob).digest()
    staged=hashlib.sha256(previous+delta_hash+struct.pack('<QIQ',generation,owner,len(delta))).hexdigest()
    return dict(previous_claim_hash=previous_claim_hash,stage_delta_hash=delta_hash.hex(),
        staged_claim_hash=staged,staged_total_claims=len(normalized)+len(delta),
        stage_new_count=len(delta),generation=generation,previous_generation=previous_generation,owner=owner,
        canonical_restored_claim_hash=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),
        delta_records=blob,guard_records_sha256=hashlib.sha256(loaded.records).hexdigest())


class NativeGuard:
    """Preloaded native context; live claim publication remains worker-owned."""
    def __init__(self,native,loaded):
        need(loaded.expected_spec is not None,'Native attach requires authorized load_pure result')
        verified=load_pure(loaded.refs,loaded.expected_spec)
        need(loaded.records==verified.records and loaded.manifest==verified.manifest,'Loaded fast guard was mutated before attachment')
        loaded=verified
        self.lib=getattr(native,'lib',native);self.loaded=loaded;self.closed=False
        self.run_identity=None;self.last_commit_index=-1;self.published_generation=0
        lib=self.lib;lib.cc_fast_guard_create.argtypes=[C.POINTER(C.c_uint8),C.c_uint64,C.c_uint32];lib.cc_fast_guard_create.restype=C.c_void_p
        lib.cc_fast_guard_destroy.argtypes=[C.c_void_p];lib.cc_fast_guard_destroy.restype=None
        for name in ('cc_fast_guard_restore','cc_fast_guard_update'):
            function=getattr(lib,name);function.argtypes=[C.c_void_p,C.POINTER(C.c_uint8),C.c_uint64];function.restype=C.c_int
        raw=(C.c_uint8*len(loaded.records)).from_buffer_copy(loaded.records)
        self.handle=lib.cc_fast_guard_create(raw,len(loaded.records)//STATIC.size,len(loaded.origin_ids))
        need(self.handle,'Native fast guard rejected compiled lookup')
        lib.cc_fast_guard_publish_stage.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_publish_stage.restype=C.c_int
        lib.cc_fast_guard_discard_stage.argtypes=[C.c_void_p];lib.cc_fast_guard_discard_stage.restype=None
        lib.cc_fast_guard_restore_generation.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_guard_restore_generation.restype=C.c_int
        lib.cc_fast_guard_stage_certificate.argtypes=[C.c_void_p];lib.cc_fast_guard_stage_certificate.restype=C.c_char_p
    def restore(self,claimed,generation=0):
        need(not self.closed,'Fast guard is closed');payload=self.loaded.claims_blob(claimed)
        need(type(generation) is int and 0<=generation<2**64,'Cold-restored generation differs')
        raw=(C.c_uint8*len(payload)).from_buffer_copy(payload)
        need(self.lib.cc_fast_guard_restore(self.handle,raw,len(payload)//STATIC.size)==1,'Native guard cold-restore rejected live claims')
        need(self.lib.cc_fast_guard_restore_generation(self.handle,generation)==1,'Native guard cold-restore rejected generation')
        self.published_generation=generation
    def bind_run(self,identity_ref,last_commit_index=-1):
        """Bind the durable run before decisions; this does not create a writer."""
        need(not self.closed and self.run_identity is None,'Fast guard run already bound/closed')
        need(type(identity_ref) is dict and set(identity_ref)=={'path','sha256'} and
             type(last_commit_index) is int and last_commit_index>=-1,'Run identity reference/cursor differs')
        path=_proof_path(identity_ref['path'])
        need(path.is_file() and path.stat().st_size<=64*1024*1024,'Fast guard identity missing/oversized')
        payload=path.read_bytes()
        need(len(payload)<=64*1024*1024 and hashlib.sha256(payload).hexdigest()==identity_ref['sha256'],
             'Fast guard run identity hash differs')
        identity=json.loads(payload)
        need(type(identity) is dict and identity.get('guard_refs')==self.loaded.refs,
             'Run identity does not bind the exact compiled guard')
        self.run_identity=dict(path=path,sha256=identity_ref['sha256'])
        self.last_commit_index=last_commit_index
    def publish_stage(self,generation,origin,durable_ref):
        """Publish only the exact staged claim set authorized by a durable check.

        Proof files are compact and bounded.  No key iteration, key allocation,
        canonicalization or map insertion is deferred to this method.
        """
        need(not self.closed and self.run_identity is not None,'Fast guard has no bound durable run')
        need(type(durable_ref) is dict and set(durable_ref)==
             {'event_path','event_sha256','commit_path','commit_sha256','index'},'Durable ownership proof shape differs')
        need(type(generation) is int and self.published_generation<generation<2**64 and
             type(durable_ref['index']) is int and durable_ref['index']>self.last_commit_index,
             'Durable ownership cursor/generation did not advance')
        event_path=_proof_path(durable_ref['event_path']);commit_path=_proof_path(durable_ref['commit_path'])
        run=self.run_identity['path'].parent;index=durable_ref['index']
        need(commit_path==run/'commits'/f'{index:06d}.json','Durable ownership commit path/cursor differs')
        commit=_proof_json(commit_path,durable_ref['commit_sha256'])
        need(commit.get('index')==index and commit.get('identity_sha256')==self.run_identity['sha256'] and
             commit.get('event_sha256')==durable_ref['event_sha256'] and
             commit.get('event')==f'events/{index:06d}-{durable_ref["event_sha256"]}.json' and
             event_path==run/commit['event'],'Durable ownership commit binding differs')
        previous=run/'commits'/f'{index-1:06d}.json' if index else None
        need((previous is None and commit.get('previous_sha256') is None) or
             (previous is not None and previous.is_file() and previous.stat().st_size<=65536 and
              commit.get('previous_sha256')==sha(previous)),
             'Durable ownership immediate predecessor differs')
        event=_proof_json(event_path,durable_ref['event_sha256']);result=event.get('result',{})
        certificate=result.get('fast_guard_certificate',{})
        owner=self.loaded.owner_id(origin)
        need(event.get('kind')=='ownership_check' and result.get('passed') is True and result.get('origin')==origin and
             certificate.get('generation')==generation and certificate.get('owner')==owner and
             certificate.get('previous_generation')==self.published_generation and
             certificate.get('guard_records_sha256')==self.loaded.refs['records_sha256'],
             'Durable ownership check does not authorize this native stage')
        active=self.lib.cc_fast_guard_stage_certificate(self.handle)
        need(type(active) is bytes and 0<len(active)<=8192,'Native stage certificate absent/oversized')
        active=json.loads(active)
        required={'generation','owner','previous_generation','previous_claim_hash','stage_delta_hash',
                  'staged_claim_hash','staged_total_claims','stage_new_count','guard_records_sha256'}
        need(type(active) is dict and required<=set(active) and
             all(certificate.get(name)==value for name,value in active.items()),
             'Durable ownership certificate differs from the actual staged native map')
        need(self.lib.cc_fast_guard_publish_stage(self.handle,generation,owner)==1,'Native staged-map publication rejected generation/origin')
        self.last_commit_index=index;self.published_generation=generation
    def discard_stage(self):
        need(not self.closed,'Fast guard is closed');self.lib.cc_fast_guard_discard_stage(self.handle)
    def close(self):
        if not self.closed:self.lib.cc_fast_guard_destroy(self.handle);self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def _proof_path(value):
    need(type(value) is str and value,'Durable ownership proof requires a path')
    path=Path(value);path=(path if path.is_absolute() else ROOT/path).resolve()
    need(path.is_relative_to(ROOT/'build/tracing-speed-v1'),'Durable ownership proof outside new namespace')
    return path


def _proof_json(path,expected):
    need(path.is_file() and path.stat().st_size<=65536,'Durable ownership proof missing/oversized')
    payload=path.read_bytes();need(len(payload)<=65536 and hashlib.sha256(payload).hexdigest()==expected,
                                 'Durable ownership proof hash differs')
    try:value=json.loads(payload)
    except (ValueError,UnicodeError) as error:raise ValueError('Malformed durable ownership proof JSON') from error
    need(type(value) is dict,'Durable ownership proof must be an object');return value
