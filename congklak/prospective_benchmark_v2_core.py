"""Native trace validity/ownership inside the clock; lossless archival afterward.

The accepted engine adapter is executed in a private versioned namespace. Only
its DLL/trace finalizer/metadata validator bindings change. Search arithmetic,
trace hooks, deterministic ties and model tensors remain the frozen versions.
"""
from __future__ import annotations
from contextlib import ExitStack
import base64
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import uuid
import zlib
from . import config,prospective_benchmark_v2_adapter as old,prospective_benchmark_v2_adapter as reader,trace_puct_witness_v1 as witness

ROOT=old.ROOT
DIRECTORY=ROOT/'build/prospective-benchmark-v2/native'
DLL=DIRECTORY/'congklak_prospective_benchmark_v2.dll'
BUILD=DIRECTORY/'build.json'
ADAPTER=ROOT/'congklak/prospective_benchmark_v2_adapter.py'
PACK,OP_NAMES,OPERATIONS,SCOPES=old.PACK,old.OP_NAMES,old.OPERATIONS,old.SCOPES
MODEL_SHA,ENGINES=old.MODEL_SHA,old.ENGINES
OWNED=struct.Struct('<15sQ16sB')
VIOLATION=struct.Struct('<QII')
# Calibration-only initial policy. Production/timing admission requires the
# separately frozen measured calibration artifact, never a retrospective pass.
DEFAULT_POLICY={'schema':'trace-finalization-debt-v1','reserve_ms':8.0,'minimum_search_ms':10,
    'maximum_outer_ms':25,'same_all_engines':True,'calibrated':False}

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return old.sha(path)
def read(path):return old.read(path)
def digest(body):return old.digest(body)

class TraceHold(RuntimeError):
    def __init__(self,message,transient=None):
        super().__init__(message);self.transient=transient;self.trace=transient

def build_record():
    r=read(BUILD)
    need(r.get('schema')=='fast-complete-trace-build-v1' and r['dll_sha256']==sha(DLL),
         'Fast native build identity differs')
    need(r['flags']==['/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD'] and
         r['event_bytes']==24 and r['owned_bytes']==40 and r['violation_bytes']==16,'Fast native layout differs')
    for name,h in r['inputs'].items():need(sha(ROOT/name)==h,'Fast source changed: '+name)
    # All six algorithm source copies are byte-identical to accepted ownership_v1.
    for name in ('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp'):
        need((ROOT/'native/ownership_v1'/name).read_bytes()==(ROOT/'native/prospective_benchmark_v2'/name).read_bytes(),
             'A frozen algorithm source changed: '+name)
    return r

def source_identity():
    r=build_record();base=old.source_identity()
    return dict(base,module_sha256=sha(__file__),dll_sha256=r['dll_sha256'],build_sha256=sha(BUILD),
        native_sources=r['inputs'],fast_trace_protocol='complete-packed24/native-partition/SHA256/high15/staged-guard-v2-timed-pending',
        puct_witness=witness.build_identity(),
        frozen_adapter_sha256=sha(ROOT/'congklak/native_ownership_trace.py'),
        projected_adapter_sha256=sha(ADAPTER),adapter_projection_sha256=sha(DIRECTORY/'source-projection.json'),
        archive_boundary='owned raw/hash/native full validity+physical checks+guard inside; lossless encoding/fsync outside',
        production_admission=False)

def attach_api(native):
    lib=native.lib
    lib.cc_fast_guard_create.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_create.restype=C.c_void_p
    lib.cc_fast_guard_destroy.argtypes=[C.c_void_p];lib.cc_fast_guard_destroy.restype=None
    for name in ('cc_fast_guard_restore','cc_fast_guard_update'):
        getattr(lib,name).argtypes=[C.c_void_p,C.c_void_p,C.c_uint64];getattr(lib,name).restype=C.c_int
    lib.cc_fast_guard_restore_generation.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_guard_restore_generation.restype=C.c_int
    lib.cc_fast_guard_publish_stage.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_publish_stage.restype=C.c_int
    lib.cc_fast_guard_discard_stage.argtypes=[C.c_void_p];lib.cc_fast_guard_discard_stage.restype=None
    for name in ('cc_fast_guard_status','cc_fast_guard_stage_certificate'):
        getattr(lib,name).argtypes=[C.c_void_p];getattr(lib,name).restype=C.c_char_p
    lib.cc_fast_finalize.argtypes=[C.c_void_p,C.c_uint32];lib.cc_fast_finalize.restype=C.c_char_p
    for name in ('cc_fast_owned_data','cc_fast_violation_data'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_void_p
    lib.cc_fast_profile_load.argtypes=[C.c_void_p,C.c_uint64];lib.cc_fast_profile_load.restype=C.c_int
    lib.cc_fast_profile_expected.argtypes=[C.c_uint64,C.c_int,C.c_int,C.POINTER(C.c_uint64)];lib.cc_fast_profile_expected.restype=C.c_int
    lib.cc_fast_sink_open.argtypes=[C.c_wchar_p];lib.cc_fast_sink_open.restype=C.c_int
    lib.cc_fast_sink_close.argtypes=[];lib.cc_fast_sink_close.restype=C.c_int

def _state_event(raw,index):
    need(0<=index<len(raw)//24,'Missing critical played-state event')
    cells,actor,operation,scope,reserved,action,depth=PACK.unpack_from(raw,index*24)
    return dict(state=dict(pits=[list(cells[:7]),list(cells[8:15])],stores=[cells[7],cells[15]],controller=actor),
                action=action,index=index)

def _native_summary(trace,identity=None,decision_id=None):
    cert=trace['fast_certificate'];raw=trace['_raw_trace']
    need(cert['valid'] and cert['guard_passed'] and not cert['overflow'] and not cert['invalid'] and
         cert['event_count']==cert['attempted_events'] and cert['event_count']*24==len(raw),
         'Native complete validity/ownership failed: '+cert.get('error',''))
    need(sum(trace['operation_counts'].values())==cert['event_count'] and
         trace['operation_counts']=={name:cert['operation_counts'][k] for name,k in OPERATIONS.items()},
         'Native physical operation partition differs')
    if identity is not None:need(trace['binding']['provider_identity']==identity,'Native identity differs')
    if decision_id is not None:need(trace['binding']['decision_id']==decision_id,'Native decision cursor differs')
    ins=[] if not trace['operation_counts']['played_input'] else [_state_event(raw,cert['played_input_index'])]
    outs=[] if not trace['operation_counts']['played_output'] else [_state_event(raw,cert['played_output_index'])]
    need(trace['operation_counts']['played_input']==len(ins) and trace['operation_counts']['played_output']==len(outs),
         'Multiple/missing native played anchors')
    return dict(scopes=[k for k in SCOPES if cert['scope_mask']&(1<<k)],played_inputs=ins,played_outputs=outs)

def _private_adapter(owner):
    name='congklak._fast_frozen_adapter_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,ADAPTER)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.DIRECTORY=DIRECTORY;module.DLL=DLL;module.BUILD=BUILD
    module.build_record=build_record;module.source_identity=source_identity
    attach_old=module._attach
    def attached(native):attach_old(native);attach_api(native)
    module._attach=attached
    module.trace_snapshot=owner._snapshot
    module.validate_trace=_native_summary
    return module

class Providers:
    def __init__(self,stack,capacity=200000,policy=None):
        need(isinstance(stack,ExitStack),'Use ExitStack for resource lifetime')
        from . import quality_ownership_v1  # Import only: no inference or search.
        self._witness=witness.NativeWitness()
        self.policy=dict(DEFAULT_POLICY if policy is None else policy)
        need(self.policy.get('same_all_engines') is True and 0<float(self.policy['reserve_ms'])<25 and
             1<=self.policy['minimum_search_ms']<25,'Invalid common finalization-debt policy')
        self.capacity=capacity;self.guard=None;self.owner_id=0;self._sink_path=None;self._last_trace=None;self._attempt_begun=False
        self._module=_private_adapter(self);self._providers=self._module.Providers(stack,capacity)
        self.native=self._providers.native;self.identity=self._providers.identity
        begin=self.native.lib.cc_ownership_trace_begin
        # Preallocate raw/high-map capacity outside decisions; no model forward.
        allocation_started=time.perf_counter_ns();allocation_before=self.lifecycle()
        need(begin(capacity,1),'Cannot preallocate bounded native trace workspace')
        self.native.lib.cc_ownership_trace_end()
        allocation_after=self.lifecycle()
        need(allocation_before==allocation_after,'Trace preallocation changed engine lifetime')
        self.allocation_receipt=dict(schema='prospective-benchmark-v2-trace-preallocation',capacity_events=capacity,
            elapsed_ns=time.perf_counter_ns()-allocation_started,before=allocation_before,after=allocation_after,
            model_queries=0,searches=0,empty_finalize_calls=0,guard_generation_changes=0)
        def begin_sink(cap,scope):
            okay=begin(cap,scope)
            if okay:self._attempt_begun=True
            if okay and self._sink_path is not None:okay=self.native.lib.cc_fast_sink_open(str(self._sink_path))
            return okay
        self.native.lib.cc_ownership_trace_begin=begin_sink
    def attach_guard(self,compiled):
        self.guard=compiled.attach(self.native);self.compiled_guard=compiled;return self.guard
    def reset_all(self,reason='handoff',decision_id=None):return self._providers.reset_all(reason,decision_id)
    def lifecycle(self):return self._providers.lifecycle()
    def _snapshot(self,native,identity,name,decision_id,state,capacity):
        lib=native.lib;tick=time.perf_counter()
        handle=self.guard.handle if self.guard is not None else None
        cert=json.loads(lib.cc_fast_finalize(handle,self.owner_id).decode())
        raw=C.string_at(lib.cc_ownership_trace_data(),cert['event_count']*24) if cert['event_count'] else b''
        owned=C.string_at(lib.cc_fast_owned_data(),cert['owned_count']*40) if cert['owned_count'] else b''
        violations=C.string_at(lib.cc_fast_violation_data(),cert['violations_count']*16) if cert['violations_count'] else b''
        trace=dict(schema='complete-state-ownership-trace-v1',encoding='base64/zlib/packed24/little-endian',event_bytes=24,
            event_count=cert['event_count'],byte_count=len(raw),capacity_events=capacity,
            attempted_events=cert['attempted_events'],overflow=cert['overflow'],invalid=cert['invalid'],truncated=False,
            operation_counts={name:cert['operation_counts'][k] for name,k in OPERATIONS.items()},scope_names=dict(SCOPES),
            event_sha256=cert['event_sha256'],binding=dict(provider_identity=identity,engine=name,decision_id=decision_id,root_state=state),
            ownership_domain='complete98seed states, full high domain including75-98',production_admission=False,
            fast_certificate=cert,_raw_trace=raw,_owned_high=owned,_violations=violations,
            _finalization_ms=(time.perf_counter()-tick)*1000)
        self._last_trace=trace
        need(hashlib.sha256(raw).hexdigest()==cert['event_sha256'] and hashlib.sha256(owned).hexdigest()==cert['owned_sha256'] and
             hashlib.sha256(violations).hexdigest()==cert['violations_sha256'],'Owned Python/native snapshot hash differs')
        gap=cert['operation_counts'][35]+cert['operation_counts'][36]-cert['operation_counts'][30]
        cert['puct_order_witness']=self._witness.compute(raw) if name in ('control','candidate') and gap==1 and not getattr(self,'direct_probe',False) else None
        trace['_finalization_ms']=(time.perf_counter()-tick)*1000
        return trace
    def decide(self,name,state,decision_id,guard=None,origin=None,outer_started=None,deadline=None,
               fixed_work=None,attempt_sink=None,reset_receipt=None):
        started=time.perf_counter() if outer_started is None else outer_started
        self._sink_path=Path(attempt_sink).resolve() if attempt_sink is not None else None
        self._last_trace=None;self._attempt_begun=False;self._providers.last_failed_envelope=None
        if guard is not None:self.guard=guard
        if deadline is None and fixed_work is None:deadline=started+.025
        context=None
        try:
            need(self.guard is not None,'Attach the frozen ownership guard before a decision')
            self.owner_id=self.compiled_guard.owner_id(origin) if hasattr(self,'compiled_guard') else int(origin)
            need(self.owner_id>0,'Missing live origin identity')
            status=json.loads(self.native.lib.cc_fast_guard_status(self.guard.handle).decode())
            need(not status['active_stage'],'Previous staged claims require durable publication before new search')
            reset=reset_receipt or self.reset_all(reason='every_decision',decision_id=decision_id)
            before_search=time.perf_counter()
            if fixed_work is None:
                window=math_floor((deadline-before_search)*1000-float(self.policy['reserve_ms']))
                need(window>=self.policy['minimum_search_ms'],'Insufficient positive useful search window: HOLD')
                answer=self._providers._decision(name,state,window,128,decision_id)
            else:
                need(fixed_work==128,'Only fixed128 engineering work admitted')
                answer=self._providers.fixed_decision(name,state,128,decision_id)
            context=self._transient(answer,name,state,decision_id,None)
            context['both_seat_reset']=reset
            context['timing']=dict(outer_elapsed_ms=(time.perf_counter()-started)*1000,
                requested_outer_ms=25 if fixed_work is None else None,
                search_window_ms=answer['decision']['requested_ms'],
                finalization_ms=self._last_trace['_finalization_ms'],policy=dict(self.policy),
                archive_outside_clock=True)
            critical=dict(decision=answer['decision'],transition=answer['transition'],coverage=answer['coverage'],
                lifecycle=answer['lifecycle'],trace_sha256=context['certificate']['event_sha256'],
                certificate=context['certificate'],both_seat_reset=reset,
                binding=answer['trace']['binding'])
            context['critical_sha256']=digest(critical)
            context['timing']['outer_elapsed_ms']=(time.perf_counter()-started)*1000
            return context
        except BaseException as exc:
            if self._last_trace is None and self._attempt_begun:
                try:self._snapshot(self.native,self.identity,name,decision_id,state,self.capacity)
                except BaseException:pass
            context=self._transient(self._providers.last_failed_envelope,name,state,decision_id,str(exc))
            context['timing']={'outer_elapsed_ms':(time.perf_counter()-started)*1000,'failure':True,'policy':dict(self.policy)}
            raise TraceHold(str(exc),context) from exc
        finally:self.native.lib.cc_ownership_trace_end()
    def _transient(self,answer,name,state,decision_id,error):
        t=self._last_trace
        return dict(answer=answer,raw_trace=t['_raw_trace'] if t else b'',owned_high=t['_owned_high'] if t else b'',
            violations=t['_violations'] if t else b'',certificate=t['fast_certificate'] if t else {'valid':False,'guard_passed':False,'error':'finalization unavailable'},
            trace_metadata={k:v for k,v in t.items() if not k.startswith('_')} if t else None,
            attempt_sink=str(self._sink_path) if self._sink_path else None,error=error,
            engine_attempt_started=self._attempt_begun,
            engine=name,state=state,decision_id=decision_id,_close_sink=self.native.lib.cc_fast_sink_close)

def math_floor(value):return __import__('math').floor(value)
def open_providers(stack,capacity=200000,policy=None):return Providers(stack,capacity,policy)

def archive_decision(transient):
    """Outside clock: encode ONLY captured immutable bytes, flush mapped prefix."""
    tick=time.perf_counter();raw=transient['raw_trace'];cert=transient['certificate']
    archival_errors=[]
    if cert.get('event_sha256') and hashlib.sha256(raw).hexdigest()!=cert['event_sha256']:
        archival_errors.append('Raw snapshot hash mismatch: HOLD')
    try:
        if transient.get('_close_sink') and transient['_close_sink']()!=1:archival_errors.append('Mapped attempt sink flush failed: HOLD')
    except BaseException as exc:archival_errors.append('Mapped attempt close failed: '+str(exc))
    result=dict(transient['answer']) if transient['answer'] is not None else {}
    trace=dict(transient['trace_metadata']) if transient['trace_metadata'] is not None else {}
    for key in tuple(trace):
        if key.startswith('_'):del trace[key]
    compressed=zlib.compress(raw,1);trace.update(data=base64.b64encode(compressed).decode(),compressed_bytes=len(compressed))
    result.update(trace=trace,fast_certificate=cert,fast_owned_high=base64.b64encode(transient['owned_high']).decode(),
        fast_violations=base64.b64encode(transient['violations']).decode(),fast_error=transient['error'],
        fast_critical_sha256=transient.get('critical_sha256'),fast_timing=transient['timing'],
        both_seat_reset=transient.get('both_seat_reset'),attempt_sink=transient['attempt_sink'],
        engine_attempt_started=transient['engine_attempt_started'],
        archival_errors=archival_errors,
        archival_timing={'elapsed_ms':(time.perf_counter()-tick)*1000,'charged_to_lease':True,'encoding_only_after_owned_validated_snapshot':True})
    return result

archive_attempt=archive_decision

def trace_bytes(trace):return old.trace_bytes(trace)
def iter_trace_states(trace):yield from old.iter_trace_states(trace)
def compile_footprint(trace):return old.compile_footprint(trace)

def validate_decision(answer,engine,state,decision_id,identity=None):
    reader.validate_decision(answer,engine,state,decision_id,identity)
    cert=answer['fast_certificate'];raw=trace_bytes(answer['trace'])
    need(answer.get('fast_error') is None and not answer.get('archival_errors') and cert['valid'] and cert['guard_passed'] and
         cert==answer['trace']['fast_certificate'] and cert['event_sha256']==hashlib.sha256(raw).hexdigest(),
         'Fast archived certificate differs')
    order=cert.get('puct_order_witness')
    if order is not None:
        expected_order=witness.pure(raw,answer['trace']['binding']['provider_identity']['puct_witness']['dll_sha256'])
        need(order==expected_order,'Native/pure completed-evaluation order witness differs')
    owned=base64.b64decode(answer['fast_owned_high'],validate=True);violations=base64.b64decode(answer['fast_violations'],validate=True)
    need(len(owned)==cert['owned_count']*40 and hashlib.sha256(owned).hexdigest()==cert['owned_sha256'] and
         not violations and cert['violations_count']==0,'Archived owned/guard map differs')
    expected={}
    from .neurodiag_data import quotient
    for event in old.iter_events(answer['trace']):
        p=event['state']
        if sum(map(sum,p['pits']))>=25:
            q=bytes.fromhex(quotient(p));expected.setdefault(q,OWNED.pack(q,event['index'],bytes(__import__('congklak.native',fromlist=['flatten']).flatten(p)),p['controller']))
    need(owned==b''.join(v for k,v in sorted(expected.items())),'Full raw/native high footprint mismatch')
    reset=answer['both_seat_reset'];before=reset['before'];after=reset['after']
    bound=answer['trace']['binding']['provider_identity']
    need(reset['schema']=='ownership-both-seat-reset-v1' and reset['decision_id']==decision_id and
         reset['source_identity_sha256']==digest(bound) and reset['actual_controls_replaced']==2 and
         after['created']-before['created']==2 and after['destroyed']-before['destroyed']==2 and
         all(after['retained_inference_controls'][name]>before['retained_inference_controls'][name]
             for name in ('control','candidate')),'Both-seat physical reset receipt differs')
    critical=dict(decision=answer['decision'],transition=answer['transition'],coverage=answer['coverage'],
        lifecycle=answer['lifecycle'],trace_sha256=cert['event_sha256'],certificate=cert,
        both_seat_reset=reset,binding=answer['trace']['binding'])
    need(answer['fast_critical_sha256']==digest(critical),'Archived compact critical binding differs')
    return {'passed':True,'event_count':cert['event_count'],'owned_count':cert['owned_count'],'production_admission':False}

def read_attempt_sink(path):
    """Pure process-kill prefix recovery; this is never a completed decision."""
    data=Path(path).read_bytes();need(len(data)>=64 and data[:8]==b'OTRACE2\0','Unknown mapped trace sink')
    capacity,count,flags=struct.unpack_from('<QQQ',data,8)
    need(1<=capacity<=16000000 and count<=capacity and len(data)>=64+count*24,'Mapped prefix length differs')
    raw=data[64:64+count*24]
    for cells,actor,operation,scope,reserved,action,depth in PACK.iter_unpack(raw):
        need(sum(cells)==98 and actor in (0,1) and 1<=operation<=37 and scope in SCOPES and reserved==0 and action>=-1 and depth>=-1,
             'Mapped complete prefix invalid')
    return dict(schema='native-mapped-attempt-prefix-v1',raw_trace=raw,event_count=count,capacity=capacity,flags=flags,
        event_sha256=hashlib.sha256(raw).hexdigest(),complete_decision=False,hold=True,powerloss_recovery_claimed=False)
