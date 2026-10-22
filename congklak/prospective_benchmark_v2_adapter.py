"""Versioned complete-state engine traces and cold decision providers.

Search math and deterministic ties come from frozen source projections. The
relay baseline's decision-local residual cache additionally binds raw stores.
Packed native events preserve every complete state without Python callbacks.
Completed trace readers below use only persisted bytes and the standard library.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from contextlib import ExitStack
import base64
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import time
import zlib
from . import config

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'build/ownership-implementation-v1/native'
DLL=DIRECTORY/'congklak_ownership_trace_v1.dll'
BUILD=DIRECTORY/'build.json'
CONTROL=ROOT/'data/quality-train-v1/model.json'
CANDIDATE=ROOT/'data/value-training-v1/training/model.json'
RELAY=ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json'
MODEL_SHA={
 'control':'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558',
 'candidate':'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319',
 'H8_relay_CPU':'4796c375398a155fb5edfcfdf1be2442734b7ff1563a32a81d7a131a74103b6b'}
ENGINES=('H_v1','H8_relay_CPU','control','candidate')
OP_NAMES=('decode','normalize_in','normalize_out','legal','terminal_test','margin_eval',
 'move_in','move_out','alpha_node','alpha_tt_probe','alpha_tt_hit','alpha_tt_slot',
 'relay_node','relay_tt_probe','relay_tt_hit','relay_tt_slot','relay_eval',
 'relay_cache_probe','relay_cache_hit','model_eval','model_forward','feature_input',
 'feature_move_in','feature_move_out','sidecar_eval','sidecar_branch','sidecar_bypass',
 'puct_node','puct_select','puct_expand','played_input','played_output',
 'alpha_pv_node','relay_pv_node','model_done','sidecar_done','alpha_root_node')
OPERATIONS={name:i+1 for i,name in enumerate(OP_NAMES)}
SCOPES={1:'H_v1',2:'H8_relay_CPU',3:'relay_feature',4:'PUCT'}
PACK=struct.Struct('<16sBBBBhh')
ABI='congklak-complete-state-ownership-trace-v1/packed24'
DEFAULT_CAPACITY=200000
RESET_FIELDS=('tree','transposition_table','evaluator_cache','feature_cache','ordering_history',
              'root_policy','visits_values','rng_state')

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(body):return hashlib.sha256(config.canonical(body).encode()).hexdigest()

class TraceHold(RuntimeError):
    def __init__(self,message,trace=None):super().__init__(message);self.trace=trace

def build_record():
    record=read(BUILD)
    need(record.get('schema')=='complete-state-ownership-native-build-v1' and
         record.get('dll_sha256')==sha(DLL),'Ownership native build identity differs')
    need(record.get('flags')==['/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD'] and
         record.get('event_bytes')==PACK.size==24 and record.get('capacity_events_max')==2000000,
         'Ownership native ABI/build contract differs')
    for name,checksum in record['inputs'].items():need(sha(ROOT/name)==checksum,'Native source changed: '+name)
    need(sha(DIRECTORY/'source-projection.json')==record['source_projection_sha256'],'Native source projection changed')
    projection=read(DIRECTORY/'source-projection.json')
    for item in projection['copies'].values():
        need(sha(ROOT/item['original'])==item['original_sha256'] and sha(ROOT/item['generated'])==item['generated_sha256'],
             'Frozen or traced source projection changed')
    py=read(DIRECTORY/'python-projection.json')
    need(sha(ROOT/py['original'])==py['original_sha256'] and sha(ROOT/py['generated'])==py['generated_sha256'],
         'Frozen or traced Python search projection changed')
    return record

def source_identity():
    record=build_record()
    for name,path in [('control',CONTROL),('candidate',CANDIDATE),('H8_relay_CPU',RELAY)]:
        need(sha(path)==MODEL_SHA[name],'Frozen model changed: '+name)
    return dict(schema='complete-state-ownership-provider-identity-v1',rules_digest=config.RULES_DIGEST,
        model_sha256=dict(MODEL_SHA),dll_sha256=record['dll_sha256'],build_sha256=sha(BUILD),
        native_sources=record['inputs'],module_sha256=sha(__file__),
        python_projection_sha256=sha(DIRECTORY/'python-projection.json'),
        encoder='frozen127/exact-relay',numeric_backend='avx2-double-packed/fp-strict',
        search=dict(PUCT='fixed-player0/legalsoftmax/cpuct0.25/FPUparent',cache_capacity=0,
                    max_backups=100000,max_nodes=2048,root_noise=None,baseline_max_depth=16,
                    baseline_TT_MiB=16,PVS=True,per_decision_reset=True),
        relay_cache='decision-local full raw14 pits+both stores+actor; immutable model/rules/backend/search private context',
        move_rng='unused; deterministic ties',production_admission=False)

def _attach(native):
    lib=native.lib
    lib.cc_ownership_trace_abi.argtypes=[];lib.cc_ownership_trace_abi.restype=C.c_char_p
    need(lib.cc_ownership_trace_abi().decode()==ABI,'Ownership native trace ABI differs')
    lib.cc_ownership_trace_begin.argtypes=[C.c_uint64,C.c_ubyte];lib.cc_ownership_trace_begin.restype=C.c_int
    for name in ('cc_ownership_trace_count','cc_ownership_trace_attempted'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_uint64
    for name in ('cc_ownership_trace_overflow','cc_ownership_trace_invalid'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_int
    lib.cc_ownership_trace_op_count.argtypes=[C.c_int];lib.cc_ownership_trace_op_count.restype=C.c_uint64
    lib.cc_ownership_trace_data.argtypes=[];lib.cc_ownership_trace_data.restype=C.c_void_p
    lib.cc_ownership_trace_end.argtypes=[];lib.cc_ownership_trace_end.restype=None
    for name in ('cc_ownership_controls_created','cc_ownership_controls_destroyed'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_uint64
    lib.cc_ownership_control_serial.argtypes=[C.c_void_p];lib.cc_ownership_control_serial.restype=C.c_uint64
    lib.cc_ownership_trace_mark.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_ubyte,C.c_int,C.c_int]
    lib.cc_ownership_trace_mark.restype=C.c_int

class _TraceMixin:
    def trace_mark(self,state,operation,action=-1,depth=-1):
        from .native import flatten
        p=config.position(state);cells,_=self._array(flatten(p),7,98)
        need(operation in OPERATIONS,'Unknown trace operation')
        if not self.lib.cc_ownership_trace_mark(cells,p['controller'],OPERATIONS[operation],action,depth):
            raise TraceHold('Native ownership marker failed: overflow/invalid state')

def _classes():
    # Deferred: importing/reading completed evidence must not load DLLs/models.
    from .native import Native,Control,flatten
    from .neural_relay import NeuralNative,Model as BaseModel
    from .puct_cpu_inference import CpuBatchEvaluator,COUNTERS,PredictionResult,FeatureResult
    from .guarded_value_pilot_v2_native import sidecar_model as SidecarModel
    from .value_sidecar_v2 import SidecarResult
    class TracedNative(_TraceMixin,Native):
        def __init__(self):super().__init__(DLL);_attach(self)
    class TracedNeural(_TraceMixin,NeuralNative):
        def __init__(self,model):super().__init__(model,path=DLL,backend='avx2',mode='value-none');_attach(self)
    class Evaluator(CpuBatchEvaluator):
        def __init__(self,candidate=False):
            self.model=SidecarModel(CANDIDATE) if candidate else BaseModel(CONTROL)
            self.lock=threading.RLock();self.closed=False;self.cache_capacity=0;self.max_batch=32
            self._cache=OrderedDict();self._counters={key:0 for key in COUNTERS};self.last_metrics={}
            self._control=None;self._sidecar=None;self.branch_counters={}
            self.native=TracedNeural(self.model.base if candidate else self.model);lib=self.native.lib
            lib.cc_puct_begin.argtypes=[C.c_void_p,C.c_uint64];lib.cc_puct_begin.restype=C.c_int
            lib.cc_puct_features.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.POINTER(FeatureResult)];lib.cc_puct_features.restype=C.c_int
            lib.cc_puct_predict.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p,C.POINTER(PredictionResult)];lib.cc_puct_predict.restype=C.c_int
            if candidate:
                self.branch_counters={'branch_calls':0,'branch_bypasses':0,'base_ms':0.,'branch_ms':0.}
                lib.cc_sidecar_result_size.argtypes=[];lib.cc_sidecar_result_size.restype=C.c_uint32
                need(lib.cc_sidecar_result_size()==C.sizeof(SidecarResult),'Sidecar result layout differs')
                lib.cc_sidecar_create.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_int];lib.cc_sidecar_create.restype=C.c_void_p
                lib.cc_sidecar_destroy.argtypes=[C.c_void_p];lib.cc_sidecar_destroy.restype=None
                self._sidecar=lib.cc_sidecar_create(self.native.handle,self.model.array,2065)
                need(self._sidecar,'Traced sidecar admission failed')
                lib.cc_sidecar_predict.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p,C.POINTER(SidecarResult)];lib.cc_sidecar_predict.restype=C.c_int
                self._predict_native=lib.cc_sidecar_predict
                def predict(cells,actor,_base,control,output):
                    result=SidecarResult();code=self._predict_native(cells,actor,self._sidecar,control,C.byref(result))
                    need(result.abi_version==1 and result.struct_size==C.sizeof(SidecarResult) and result.status==code and code in (0,1,2),'Traced sidecar result status differs')
                    for key in self.branch_counters:self.branch_counters[key]+=getattr(result,key)
                    C.memmove(output,C.byref(result),C.sizeof(PredictionResult))
                    C.cast(output,C.POINTER(PredictionResult)).contents.struct_size=C.sizeof(PredictionResult)
                    return code
                lib.cc_puct_predict=predict
            self._control=Control(self.native,max_depth=128,time_ms=0,node_budget=0,memory_mb=4096,tt_mb=0,use_pvs=False)
            self.backend=self.native.backend;self.identity={'traced':True,'model_sha256':self.model.file_sha256}
        def begin_decision(self):
            self.reset_cache()
            self._control.close()
            self._control=Control(self.native,max_depth=128,time_ms=0,node_budget=0,memory_mb=4096,tt_mb=0,use_pvs=False)
        def snapshot_counters(self):
            return {**super().snapshot_counters(),**self.branch_counters}
        def close(self):
            with self.lock:
                if not self.closed and self._sidecar:
                    self.native.lib.cc_sidecar_destroy(self._sidecar);self._sidecar=None
                super().close()
    return TracedNative,TracedNeural,Evaluator

def trace_snapshot(native,identity,name,decision_id,state,capacity):
    """Copy owned native memory and hash it before the whole-decision clock ends."""
    lib=native.lib;count=int(lib.cc_ownership_trace_count());size=count*PACK.size
    raw=C.string_at(lib.cc_ownership_trace_data(),size) if size else b''
    compressed=zlib.compress(raw,1)
    return dict(schema='complete-state-ownership-trace-v1',encoding='base64/zlib/packed24/little-endian',
        event_bytes=PACK.size,event_count=count,byte_count=size,capacity_events=capacity,
        attempted_events=int(lib.cc_ownership_trace_attempted()),overflow=bool(lib.cc_ownership_trace_overflow()),
        invalid=bool(lib.cc_ownership_trace_invalid()),truncated=False,
        operation_counts={op:int(lib.cc_ownership_trace_op_count(k)) for op,k in OPERATIONS.items()},
        scope_names=dict(SCOPES),event_sha256=hashlib.sha256(raw).hexdigest(),
        data=base64.b64encode(compressed).decode('ascii'),compressed_bytes=len(compressed),
        binding=dict(provider_identity=identity,engine=name,decision_id=decision_id,root_state=state),
        ownership_domain='complete98seed decision states only; never seeds-in-hand',
        production_admission=False)

def trace_bytes(trace):
    need(trace.get('schema')=='complete-state-ownership-trace-v1' and trace.get('encoding')=='base64/zlib/packed24/little-endian',
         'Unknown trace format')
    need(trace.get('event_bytes')==24 and type(trace.get('event_count')) is int and
         0<=trace['event_count']<=trace.get('capacity_events',0)<=16000000,'Trace capacity differs')
    compressed=base64.b64decode(trace['data'],validate=True)
    need(len(compressed)==trace['compressed_bytes'],'Compressed trace size differs')
    dec=zlib.decompressobj();raw=dec.decompress(compressed,trace['event_count']*24+1)
    need(dec.eof and not dec.unused_data and not dec.unconsumed_tail,'Trace compression boundary differs')
    need(len(raw)==trace['byte_count']==trace['event_count']*24 and
         hashlib.sha256(raw).hexdigest()==trace['event_sha256'],'Trace payload hash/length differs')
    return raw

def iter_events(trace):
    """Pure lossless persisted-evidence iterator: no writer, DLL, forward or search."""
    raw=trace_bytes(trace)
    for index,(cells,actor,operation,scope,reserved,action,depth) in enumerate(PACK.iter_unpack(raw)):
        need(actor in (0,1) and 1<=operation<=len(OP_NAMES) and scope in SCOPES and reserved==0,
             'Invalid trace event metadata')
        p=dict(pits=[list(cells[:7]),list(cells[8:15])],stores=[cells[7],cells[15]],controller=actor)
        need(sum(cells)==98,'Ownership trace contains an incomplete seed state')
        yield dict(index=index,state=p,operation=OP_NAMES[operation-1],scope=SCOPES[scope],action=action,depth=depth)

def iter_trace_states(trace):
    """Alias returning full event dictionaries, including complete raw state."""
    yield from iter_events(trace)

def validate_trace(trace,identity=None,decision_id=None):
    need(not trace['overflow'] and not trace['invalid'] and not trace['truncated'] and
         trace['attempted_events']==trace['event_count'],'Incomplete ownership trace: HOLD')
    counts=[0]*(len(OP_NAMES)+1);scopes=set();played_in=[];played_out=[]
    for index,(cells,actor,operation,scope,reserved,action,depth) in enumerate(PACK.iter_unpack(trace_bytes(trace))):
        need(actor in (0,1) and 1<=operation<=len(OP_NAMES) and scope in SCOPES and reserved==0 and
             action>=-1 and depth>=-1 and sum(cells)==98,'Invalid complete-state trace event')
        counts[operation]+=1;scopes.add(scope)
        if operation in (OPERATIONS['played_input'],OPERATIONS['played_output']):
            row=dict(index=index,state=dict(pits=[list(cells[:7]),list(cells[8:15])],stores=[cells[7],cells[15]],controller=actor),action=action)
            (played_in if operation==OPERATIONS['played_input'] else played_out).append(row)
    need({k:counts[v] for k,v in OPERATIONS.items()}==trace['operation_counts'],'Trace counter partition differs')
    need(sum(trace['operation_counts'].values())==trace['event_count'],'Trace events were not fully partitioned')
    if identity is not None:need(trace['binding']['provider_identity']==identity,'Trace provider identity differs')
    if decision_id is not None:need(trace['binding']['decision_id']==decision_id,'Trace decision identity differs')
    return dict(passed=True,event_count=trace['event_count'],operation_counts=trace['operation_counts'],
        scopes=sorted(scopes),played_inputs=played_in,played_outputs=played_out,production_admission=False)

def validate_decision_coverage(envelope,identity=None,decision_id=None):
    """Link a lossless trace to independent engine work and committed wrapper data.

    An event partition alone is not search coverage. This additionally binds
    played endpoints, native per-operation capture counters, physical search
    nodes, feature attempts and completed forwards. Failed partial operations
    keep their attempted inputs and are explicit rather than silently paired.
    """
    trace=envelope['trace'];summary=validate_trace(trace,identity,decision_id);pending_evaluation=0
    binding=trace['binding'];name=binding['engine'];need(name in ENGINES,'Unknown traced engine')
    answer=envelope['decision'];coverage=envelope['coverage'];ops=trace['operation_counts']
    need(coverage.get('native_operation_counters')==ops and
         coverage.get('native_event_count')==trace['event_count'],'Decision-native trace coverage counters differ')
    played_in=summary['played_inputs'];played_out=summary['played_outputs']
    need(len(played_in)==1 and played_in[0]['state']==binding['root_state'],'Missing/mismatched trace played input')
    transition=envelope['transition'];chosen=answer['chosen']
    need(len(played_out)==int(chosen is not None) and (chosen is None or
         transition and transition.get('ok') is True and played_out[0]['state']==transition['position'] and
         played_out[0]['action']==chosen and transition['pit']==chosen),'Missing/mismatched chosen trace endpoint')
    allowed={1} if name=='H_v1' else {2,3} if name=='H8_relay_CPU' else {3,4}
    need(set(summary['scopes'])<=allowed,'Cross-engine native trace scope')
    if name in ('control','candidate'):
        report=answer['report'];delta=envelope['evaluator']['delta']
        need(ops['puct_node']==report['nodes']+1,'PUCT complete tree/validation node footprint differs')
        from .trace_puct_witness_v1 import pending_allowed
        pending_evaluation=pending_allowed(report,ops,trace)
        need(ops['model_done']+ops['sidecar_done']==delta['native_actual_evals'],'Completed dense forward footprint differs')
        need(ops['feature_move_in']==delta['feature_simulations'],'Physical feature-attempt footprint differs')
        need(0<=ops['feature_move_in']-ops['feature_move_out']<=1,'Feature operation output coverage differs')
        if report['stop_reason'] in ('simulations','terminal'):
            need(ops['feature_move_in']==ops['feature_move_out'],'Completed search has unpaired feature output')
        need(ops['move_out']==ops['feature_move_out']+report['work']['transitions']+int(chosen is not None),
             'Completed search/feature/chosen transition footprint differs')
        need(0<=ops['move_in']-ops['move_out']<=1,'Incomplete move footprint not explicitly bounded')
        need(ops['alpha_node']==ops['alpha_tt_probe']==ops['alpha_tt_hit']==ops['relay_node']==ops['relay_tt_hit']==0,
             'Foreign search engine events in PUCT trace')
        need(name=='candidate' or ops['sidecar_eval']==ops['sidecar_branch']==ops['sidecar_bypass']==ops['sidecar_done']==0,
             'Control trace consulted candidate branch')
        need(ops['sidecar_branch']==0 if sum(map(sum,binding['root_state']['pits']))<=24 else True,
             'Shared-root search entered changed branch')
    elif name=='H_v1':
        need(ops['alpha_node']+ops['alpha_pv_node']+ops['alpha_root_node']==answer['nodes'],
             'Alpha/PV/root physical node footprint differs')
        need(ops['model_eval']==ops['feature_input']==ops['puct_node']==ops['relay_node']==0,
             'Heuristic baseline has foreign engine events')
    else:
        need(ops['relay_node']+ops['relay_pv_node']==answer['nodes'],'Relay/PV physical node footprint differs')
        need(ops['alpha_node']==ops['puct_node']==ops['sidecar_eval']==0,'Relay baseline has foreign engine events')
        report=answer['report']
        if report and answer['stop_reason']=='max_depth':
            need(ops['feature_move_in']==report['feature_simulations'] and
                 ops['feature_move_out']==ops['feature_move_in'],'Completed relay feature footprint differs')
            need(ops['model_done']==report['nn_evaluations'] and ops['relay_cache_hit']==report['nn_cache_hits'] and
                 ops['relay_tt_hit']==report['tt_hits'],'Completed relay physical inference/TT footprint differs')
    return dict(passed=True,event_count=trace['event_count'],engine=name,
        physical_partition_checked=True,timed_completed_evaluation_not_expanded=pending_evaluation,production_admission=False)

def validate_decision(answer,engine,state,decision_id,identity=None):
    """Pure persisted envelope validation; no DLL/model construction or search."""
    state=config.position(state);trace=answer['trace'];binding=trace['binding']
    need(engine in ENGINES and binding['engine']==engine and binding['root_state']==state and
         binding['decision_id']==decision_id,'Decision trace root/engine/cursor differs')
    observed=binding['provider_identity']
    need(observed.get('schema')=='complete-state-ownership-provider-identity-v1' and
         observed.get('rules_digest')==config.RULES_DIGEST and observed.get('model_sha256')==MODEL_SHA,
         'Decision complete model/rules scope differs')
    if identity is not None:need(observed==identity,'Decision provider source/build scope differs')
    a=answer['decision'];expected_kind='h' if engine=='H_v1' else 'relay' if engine=='H8_relay_CPU' else 'puct'
    need(a['kind']==expected_kind and type(a['requested_ms']) is int and 0<=a['requested_ms']<=100 and
         type(a['wall_ms']) in (int,float) and math.isfinite(a['wall_ms']) and a['wall_ms']>0 and
         a['overshoot_ms']==(max(0.,a['wall_ms']-a['requested_ms']) if a['requested_ms'] else 0.),
         'Decision engine/whole-clock fields differ')
    legal=[i for i,v in enumerate(state['pits'][state['controller']]) if v]
    need(a['chosen'] is None or type(a['chosen']) is int and a['chosen'] in legal,'Illegal traced recommendation')
    need(a['report'] is None or a['report']['position']==state,'Decision report root differs')
    life=answer['lifecycle'];before=life['before'];after=life['after']
    required=2 if expected_kind=='puct' else 1
    need(after['created']-before['created']==required and after['destroyed']-before['destroyed']==required,
         'Decision did not create/destroy fresh native controls')
    need(after['created']-after['destroyed']==before['created']-before['destroyed'],'Decision leaked native controls')
    if expected_kind=='puct':
        need(after['retained_inference_controls'][engine]>before['retained_inference_controls'][engine],
             'Decision reused native inference-control generation')
    return validate_decision_coverage(answer,identity,decision_id)

def compile_footprint(trace):
    """Pure trace compiler; graph admission/source closure remain separate."""
    from .neurodiag_data import quotient
    validate_trace(trace)
    rows={}
    for event in iter_events(trace):
        p=event['state'];q=quotient(p);a=sum(map(sum,p['pits']))
        row=rows.setdefault(q,dict(quotient=q,active_seeds=a,owned=a>=25,
            first_event=event['index'],state=p,operations=set(),scopes=set(),occurrences=0))
        row['operations'].add(event['operation']);row['scopes'].add(event['scope']);row['occurrences']+=1
    return {'schema':'complete-state-ownership-footprint-v1','source_trace_sha256':digest(trace),
        'decision_id':trace['binding']['decision_id'],'rows':[{**r,'operations':sorted(r['operations']),'scopes':sorted(r['scopes'])} for _,r in sorted(rows.items())],
        'source_event_count':trace['event_count'],'new_games':0,'searches':0,'production_admission':False}

class Providers:
    def __init__(self,stack,capacity=DEFAULT_CAPACITY):
        need(type(capacity) is int and 1<=capacity<=16000000,'Trace event capacity must be bounded')
        self.capacity=capacity;self.identity=source_identity();self.lock=threading.RLock()
        Native,Neural,Evaluator=_classes()
        self.native=Native()
        self.engines={'H_v1':('h',self.native),
            'H8_relay_CPU':('relay',stack.enter_context(Neural(RELAY))),
            'control':('puct',stack.enter_context(Evaluator(False))),
            'candidate':('puct',stack.enter_context(Evaluator(True)))}
    def timed_decision(self,name,state,ms=25,decision_id=None):
        need(type(ms) is int and 1<=ms<=100,'Only positive remaining whole-decision clocks allowed')
        return self._decision(name,state,ms,128,decision_id)
    def fixed_decision(self,name,state,simulations=128,decision_id=None):
        need(name in ENGINES and simulations==128,'Only declared fixed128 engineering budget is admitted')
        return self._decision(name,state,0,simulations,decision_id,baseline_nodes=128 if name in ('H_v1','H8_relay_CPU') else None)
    def fixed_depth_decision(self,name,state,depth=4,decision_id=None):
        need(name in ('H_v1','H8_relay_CPU') and type(depth) is int and 1<=depth<=10,'Engineering fixed-depth baseline bound differs')
        return self._decision(name,state,0,depth,decision_id,baseline_depth=depth)
    def lifecycle(self):
        lib=self.native.lib
        return {'created':int(lib.cc_ownership_controls_created()),'destroyed':int(lib.cc_ownership_controls_destroyed()),
            'retained_inference_controls':{name:int(lib.cc_ownership_control_serial(obj._control.handle))
                for name,(kind,obj) in self.engines.items() if kind=='puct'}}
    def reset_all(self,reason='handoff',decision_id=None):
        need(isinstance(reason,str) and reason and isinstance(decision_id,str) and decision_id,'Reset needs reason and decision identity')
        with self.lock:
            before=self.lifecycle()
            for kind,obj in self.engines.values():
                if kind=='puct':obj.begin_decision()
            after=self.lifecycle()
            need(after['created']-before['created']==2 and after['destroyed']-before['destroyed']==2 and
                 all(after['retained_inference_controls'][name]>before['retained_inference_controls'][name]
                     for name in ('control','candidate')),'Both-seat native reset generations differ')
            return dict(schema='ownership-both-seat-reset-v1',reason=reason,decision_id=decision_id,
                before=before,after=after,actual_controls_replaced=2,baseline_search_objects='decision-local, already destroyed at API return',
                fields=list(RESET_FIELDS),source_identity_sha256=digest(self.identity),production_admission=False)
    def _decision(self,name,state,ms,simulations,decision_id,baseline_depth=None,baseline_nodes=None):
        from . import quality_ownership_v1 as search
        from .native import flatten
        need(name in ENGINES and isinstance(decision_id,str) and decision_id,'Trace needs explicit engine and decision identity')
        state=config.position(state);kind,obj=self.engines[name]
        self.last_failed_envelope=None
        with self.lock:
            started=time.perf_counter();phase_start=time.perf_counter_ns();phase_marks=[];trace=None;controls_before=self.lifecycle()
            need(self.native.lib.cc_ownership_trace_begin(self.capacity,1 if kind=='h' else 2 if kind=='relay' else 4),
                 'Trace already active or allocation failed')
            try:
                self.native.trace_mark(state,'played_input')
                reset_started=time.perf_counter()
                before=obj.snapshot_counters() if kind=='puct' else None
                if kind=='puct':obj.begin_decision()
                reset_ms=(time.perf_counter()-reset_started)*1000
                grant=max(0,math.floor(ms-(time.perf_counter()-started)*1000)) if ms else 0
                if ms and not grant:raise TraceHold('Whole-decision budget exhausted before a positive search grant')
                phase_marks.append(dict(name='trace_and_decision_reset',end_ns=time.perf_counter_ns()))
                if kind=='puct':
                    report=search.search(state,obj,simulations=100000 if ms else 128,time_ms=grant,
                        max_nodes=2048,c_puct=.25,root_noise=None,memory_mb=4096)
                    search.validate_report(report,state,100000 if ms else 128,c_puct=.25,native=obj.native)
                    answer=dict(kind='puct',report=report,stop_reason=report['stop_reason'],engine_elapsed_ms=report['timing']['elapsed_ms'],
                        nodes=report['nodes'],usable_decision=report['usable_decision'],chosen=report['chosen'] if report['usable_decision'] else None,
                        completed_depth=None,completed_simulations=report['completed_simulations'])
                    after=obj.snapshot_counters()
                    telemetry=dict(before=before,after=after,delta={k:after[k]-before[k] for k in after if k not in ('cache_capacity','cache_current_entries')},
                        cache_capacity=0,reset_ms=reset_ms,granted_search_ms=grant)
                else:
                    depth=baseline_depth or 16
                    if kind=='h':raw=obj.analyze(flatten(state),actor=state['controller'],total=98,max_depth=depth,time_ms=grant,
                        node_budget=baseline_nodes or 0,memory_mb=4096,tt_mb=16,use_pvs=True,keep_history=False)
                    else:raw=obj.neural_analyze(state,max_depth=depth,time_ms=grant,node_budget=baseline_nodes or 0,memory_mb=4096,tt_mb=16,use_pvs=True)
                    need(raw.get('ok',True) and 'report' in raw,'Traced baseline search failed: '+str(raw.get('error',raw.get('stop_reason'))))
                    report=raw['report'];usable=bool(report and report['choice_depth']>=1 and report['best_moves'])
                    answer=dict(kind=kind,report=report,stop_reason=raw['stop_reason'],engine_elapsed_ms=raw['elapsed_ms'],nodes=raw['nodes'],
                        usable_decision=usable,chosen=report['best_moves'][0] if usable else None,
                        completed_depth=report['choice_depth'] if report else None,completed_simulations=None)
                    telemetry=None
                phase_marks.append(dict(name='engine_search_and_report',end_ns=time.perf_counter_ns()))
                transition=None
                if answer['chosen'] is not None:
                    transition=self.native.apply(flatten(state),actor=state['controller'],action=answer['chosen'],total=98,trace=False)
                    need(transition.get('ok'),'Chosen move failed literal/native admission')
                    self.native.trace_mark(transition['position'],'played_output',answer['chosen'])
                phase_marks.append(dict(name='chosen_native_transition',end_ns=time.perf_counter_ns()))
                trace=trace_snapshot(self.native,self.identity,name,decision_id,state,self.capacity)
                if trace['overflow'] or trace['invalid'] or trace['attempted_events']!=trace['event_count']:
                    raise TraceHold('Incomplete ownership trace: HOLD',trace)
                phase_marks.append(dict(name='complete_trace_finalize_snapshot_hash_guard',end_ns=time.perf_counter_ns()))
                controls_after=self.lifecycle()
                elapsed=(time.perf_counter()-started)*1000
                answer.update(wall_ms=elapsed,requested_ms=ms,overshoot_ms=max(0.,elapsed-ms) if ms else 0.)
                result=dict(decision=answer,evaluator=telemetry,trace=trace,transition=transition,
                    reset_attestation=dict(scope='fresh every decision; two-seat handoff caller resets both providers',
                        fields=list(RESET_FIELDS),fresh_tree=True,fresh_control=True,baseline_TT_lifetime='single analyze call',
                        inference_cache_capacity=0,relay_cache_key='raw14pits+bothstores+actor+private full context',
                        mutable_measurement_counters_only=True,move_rng='unused deterministic ties',
                        no_prefix_state_reuse=True,live_two_seat_handoff_tested=False),
                    coverage=dict(native_complete_states=True,python_PUCT_nodes_select_expand=True,
                        feature_hypothetical_both_actors=True,TT_probes_hits_and_resident_slots=True,
                        event_partition=sum(trace['operation_counts'].values()),native_operation_counters=dict(trace['operation_counts']),
                        native_event_count=trace['event_count'],
                        capture_and_dump_and_hash_inside_whole_clock=False,capture_owned_copy_hash_validate_inside_whole_clock=True,
                        archive_encoding_and_fsync_outside_clock=True,production_admission=False),
                    lifecycle={'before':controls_before,'after':controls_after},
                    work_budget={'kind':'whole_wall_ms' if ms else 'decision_nodes' if baseline_nodes else 'choice_depth' if baseline_depth else 'PUCT_edge_backups',
                        'value':ms or baseline_nodes or baseline_depth or 128})
                # Full decoded validation is deliberately inside the whole clock.
                result['coverage']['validation']=validate_decision_coverage(result,self.identity,decision_id)
                phase_marks.append(dict(name='physical_coverage_validation',end_ns=time.perf_counter_ns()))
                result['implementation_phases']=dict(start_ns=phase_start,marks=phase_marks,observational_only=True)
                elapsed=(time.perf_counter()-started)*1000
                answer.update(wall_ms=elapsed,overshoot_ms=max(0.,elapsed-ms) if ms else 0.)
                return result
            except BaseException as exc:
                # Preserve every known search result/work counter before HOLD.
                self.last_failed_envelope=dict(decision=locals().get('answer') or {},decision_known=locals().get('answer') is not None,
                    evaluator=locals().get('telemetry'),transition=locals().get('transition'),
                    partial_report=locals().get('report'),baseline_raw=locals().get('raw'),
                    lifecycle={'before':controls_before,'after':self.lifecycle()},
                    elapsed_ms=(time.perf_counter()-started)*1000,
                    requested_ms=ms,decision_id=decision_id,error=str(exc),
                    completed_decision=False,production_admission=False)
                if trace is None:trace=trace_snapshot(self.native,self.identity,name,decision_id,state,self.capacity)
                if isinstance(exc,TraceHold):exc.trace=trace;raise
                raise TraceHold(str(exc),trace) from exc
            finally:self.native.lib.cc_ownership_trace_end()

def open_providers(stack,capacity=DEFAULT_CAPACITY):
    need(isinstance(stack,ExitStack),'Use an ExitStack to close model/control resources')
    return Providers(stack,capacity)
