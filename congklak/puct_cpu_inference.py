"""Exact single-pass CPU inference and decision-local bounded raw-state LRU.

Frozen PUCT counts logical row evaluations. This adapter separately reports
physical forwards, cache events and partially completed feature work. Diagnostic
feature extraction is a separate operation and is never part of prediction.
"""
from __future__ import annotations
from collections import OrderedDict
import ctypes as C
import json
import math
from pathlib import Path
import threading
import time
from . import config
from .native import ROOT,NativeError,Control,flatten
from .neural_relay import Model,NeuralNative,sha
from .phase6_inference import InferenceStopped,validate_prediction

ABI='congklak-puct-cpu-structured-v1'
PROTOCOL='puct-cpu/frozen-double/single-feature-pass/raw17-decision-local-lru-v1'
DLL=ROOT/'build/congklak_puct_cpu.dll'
BUILD_INPUTS={'native/engine.hpp','native/rules.cpp','native/search.cpp','native/neural_relay.cpp','native/neural_puct_cpu.cpp',
              'tools/build-puct-cpu.cmd','tools/build-puct-cpu.ps1'}
BUILD_FLAGS=['/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD']
BUILD_DISPATCH='Frozen CPUID/OSXSAVE/XGETBV/AVX2 packed-double dispatch; scalar fallback, separate mul/add'
COUNTERS=('positions_submitted','evaluations_requested','native_actual_evals','predict_successes','native_calls',
          'cache_hits','cache_misses','cache_evictions','feature_simulations','feature_deposits','cache_resets','terminal_returns')

class PredictionResult(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('struct_size',C.c_uint32),('status',C.c_int32),('terminal',C.c_int32),
              ('feature_simulations',C.c_uint64),('feature_deposits',C.c_uint64),('neural_evaluations',C.c_uint64),
              ('value_player0',C.c_double),('policy_logits',C.c_double*7),('reason',C.c_char*32),('message',C.c_char*192)]
class FeatureResult(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('struct_size',C.c_uint32),('status',C.c_int32),('reserved',C.c_int32),
              ('feature_simulations',C.c_uint64),('feature_deposits',C.c_uint64),('x',C.c_double*127),('reason',C.c_char*32),('message',C.c_char*192)]

def _cancelled(cancel):return bool(cancel() if callable(cancel) else cancel.is_set()) if cancel is not None else False
def _poll(cancel,deadline):
    if _cancelled(cancel):raise InferenceStopped('cancelled')
    if deadline is not None and time.perf_counter()>=deadline:raise InferenceStopped('time_budget')
def _raw_key(position):return bytes(flatten(position)+[position['controller']])

class CpuBatchEvaluator:
    supports_cancel=True
    def __init__(self,modelpath,cache_capacity=0,max_batch=32,backend='auto'):
        config.integer(cache_capacity,'cache_capacity',0,4096);config.integer(max_batch,'max_batch',1,32)
        if backend not in ('auto','scalar','avx2'):raise ValueError('Unknown CPU backend')
        started=time.perf_counter();self.model=modelpath if isinstance(modelpath,Model) else Model(Path(modelpath))
        if self.model.width!=64 or self.model.input_count!=127:raise NativeError('CPU PUCT requires width64/input127')
        self.lock=threading.RLock();self.closed=False;self.cache_capacity=cache_capacity;self.max_batch=max_batch
        self._cache=OrderedDict();self._counters={key:0 for key in COUNTERS};self.last_metrics={};self._control=None
        record=json.loads((ROOT/'build/puct-cpu-build.json').read_text(encoding='utf-8'))
        if not isinstance(record,dict) or set(record)!={'schema','visual_studio','flags','dispatch','inputs','dll_sha256'}:raise NativeError('CPU PUCT build witness fields differ')
        if record['schema']!='puct-cpu-structured-build-v1' or record['dll_sha256']!=sha(DLL):raise NativeError('CPU PUCT build witness differs')
        if not isinstance(record['inputs'],dict) or set(record['inputs'])!=BUILD_INPUTS:raise NativeError('CPU PUCT build witness source set differs')
        if record['flags']!=BUILD_FLAGS or record['dispatch']!=BUILD_DISPATCH:raise NativeError('CPU PUCT strict build flags/dispatch differ')
        if not isinstance(record['visual_studio'],str) or not record['visual_studio'] or not Path(record['visual_studio']).is_absolute():raise NativeError('CPU PUCT compiler witness is invalid')
        for name,checksum in record['inputs'].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=checksum:raise NativeError('CPU PUCT build source changed: '+name)
        self.native=NeuralNative(self.model,path=DLL,backend=backend,mode='value-none')
        try:
            lib=self.native.lib
            lib.cc_puct_cpu_abi.argtypes=[];lib.cc_puct_cpu_abi.restype=C.c_char_p
            for name in ('cc_puct_result_size','cc_puct_features_size'):
                getattr(lib,name).argtypes=[];getattr(lib,name).restype=C.c_uint32
            lib.cc_puct_layout.argtypes=[C.c_int];lib.cc_puct_layout.restype=C.c_uint32
            lib.cc_puct_begin.argtypes=[C.c_void_p,C.c_uint64];lib.cc_puct_begin.restype=C.c_int
            lib.cc_puct_predict.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p,C.POINTER(PredictionResult)];lib.cc_puct_predict.restype=C.c_int
            lib.cc_puct_features.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.POINTER(FeatureResult)];lib.cc_puct_features.restype=C.c_int
            if lib.cc_puct_cpu_abi().decode()!=ABI or lib.cc_puct_result_size()!=C.sizeof(PredictionResult) or lib.cc_puct_features_size()!=C.sizeof(FeatureResult):raise NativeError('Structured CPU ABI/layout mismatch')
            offsets=[PredictionResult.feature_simulations.offset,PredictionResult.value_player0.offset,PredictionResult.policy_logits.offset,PredictionResult.reason.offset,FeatureResult.x.offset]
            if [lib.cc_puct_layout(i) for i in range(5)]!=offsets:raise NativeError('Structured CPU ABI field offsets mismatch')
            self._control=Control(self.native,max_depth=128,time_ms=0,node_budget=0,memory_mb=8192,tt_mb=0,use_pvs=False)
            self.backend=self.native.backend
            self.identity=dict(protocol=PROTOCOL,abi=ABI,backend=self.backend,requested_backend=backend,max_batch=max_batch,
                model_file_sha256=self.model.file_sha256,model_weights_sha256=self.model.id,rules_digest=config.RULES_DIGEST,encoder=self.model.encoder,
                source_sha256=sha(__file__),dll_sha256=self.native.build_id,build_record_sha256=sha(ROOT/'build/puct-cpu-build.json'),build_inputs=record['inputs'],
                dependencies={name:sha(ROOT/name) for name in ('congklak/config.py','congklak/native.py','congklak/neural_relay.py','congklak/features_relay.py','congklak/phase6_inference.py')},
                numeric_policy='Frozen strict-double packed scalar/AVX2 math; no score quantization',
                cache=dict(capacity=cache_capacity,key='FULL INPUT16 cells including stores + inputcontroller byte; evaluator/model private',replacement='exact bounded LRU',lifetime='reset inside EVERY decision/timing-unit whole clock',values='immutable(value,tuple(logits)); copy on return'),
                counter_policy=dict(evaluations_requested='processed nonterminal prefix; exactly cache_hits+cache_misses',positions_submitted='validated rows admitted after lock acquisition',
                    native_actual_evals='physical completed dense forwards, including post-forward deadline/cancellation discards',predict_successes='ABI-successful prediction rows before final Python poll',
                    feature_work='actual single-pass counts including stopped partial hypothetical moves',terminal_returns='exact terminal responses; no cache lookup/forward/features',
                    puct_neural_evaluations='frozen completed-batch logical row count includes cache hits; stopped-batch per_position.evaluated reports physical forwards',diagnostics='separate feature-only operation excluded from prediction counters'),
                timing_policy='whole_call_ms includes every Python/control/cache/native operation; native_call_ms combines features+dense+native boundary; individual feature/forward time not measured',
                cancellation='token native control within sowing; callable/event checked at batch/row/lock boundaries; exact absolute Python deadline before/after native call')
            self.startup_ms=(time.perf_counter()-started)*1000
        except BaseException:
            if self._control:self._control.close()
            self.native.close();raise

    def close(self):
        with self.lock:
            if not self.closed:self._control.close();self.native.close();self._cache.clear();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def reset_cache(self):
        with self.lock:
            if self.closed:raise NativeError('Evaluator is closed')
            self._cache.clear();self._counters['cache_resets']+=1
    def begin_decision(self):self.reset_cache()
    def snapshot_counters(self):
        with self.lock:return {**self._counters,'cache_current_entries':len(self._cache),'cache_capacity':self.cache_capacity}
    def metrics_snapshot(self):return self.snapshot_counters()
    def _begin(self,deadline):
        remaining=0 if deadline is None else max(1,int((deadline-time.perf_counter())*1e9))
        if not self.native.lib.cc_puct_begin(self._control.handle,remaining):raise NativeError('Cannot begin CPU inference control')
    @staticmethod
    def _validate_result(out,code,kind):
        if out.abi_version!=1 or out.struct_size!=C.sizeof(kind) or out.status!=code or code not in (0,1,2):raise NativeError('Damaged structured CPU response')
        if code==1:raise InferenceStopped(out.reason.decode() or 'cancelled')
        if code==2:raise NativeError(out.reason.decode()+': '+out.message.decode())

    def evaluate_many(self,positions,cancel=None,deadline=None):
        started=time.perf_counter();config.integer(len(positions),'batch_size',1,self.max_batch)
        if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline)):raise ValueError('Deadline must be an absolute finite perf-counter time')
        rows=[config.position(p) for p in positions]
        metrics=dict(batch_size=len(rows),queue_wait_ms=0.,native_call_ms=0.,
            whole_call_ms=0.,feature_simulations=0,feature_deposits=0,evaluated_positions=0,native_actual_evals=0,cache_hits=0,cache_misses=0,
            per_position=[dict(feature_simulations=0,feature_deposits=0,evaluated=False) for _ in rows])
        acquired=False;registered=False;running=False
        try:
            _poll(cancel,deadline);wait=time.perf_counter()
            while not self.lock.acquire(timeout=.005):_poll(cancel,deadline)
            acquired=True;metrics['queue_wait_ms']=(time.perf_counter()-wait)*1000
            if self.closed:raise NativeError('Evaluator is closed')
            self._counters['positions_submitted']+=len(rows);_poll(cancel,deadline);self._begin(deadline)
            if cancel is not None and hasattr(cancel,'_register') and hasattr(cancel,'_remove'):cancel._register(self._control);registered=True
            with self._control._lock:self._control._running=True;running=True
            outputs=[]
            for index,p in enumerate(rows):
                _poll(cancel,deadline);work=metrics['per_position'][index]
                if not sum(map(sum,p['pits'])):
                    self._counters['terminal_returns']+=1
                    value=dict(value_player0=p['stores'][0]-p['stores'][1],policy_logits=[0.]*7,feature_simulations=0,feature_deposits=0,cache_hit=False,native_actual_evals=0)
                else:
                    key=_raw_key(p);self._counters['evaluations_requested']+=1
                    if self.cache_capacity and key in self._cache:
                        cached=self._cache.pop(key);self._cache[key]=cached
                        self._counters['cache_hits']+=1;metrics['cache_hits']+=1
                        value=dict(value_player0=cached[0],policy_logits=list(cached[1]),feature_simulations=0,feature_deposits=0,cache_hit=True,native_actual_evals=0)
                    else:
                        self._counters['cache_misses']+=1;metrics['cache_misses']+=1
                        cells,_=self.native._array(flatten(p),7,98);out=PredictionResult()
                        self._counters['native_calls']+=1;tick=time.perf_counter()
                        code=self.native.lib.cc_puct_predict(cells,p['controller'],self.native.handle,self._control.handle,C.byref(out))
                        elapsed=(time.perf_counter()-tick)*1000;metrics['native_call_ms']+=elapsed
                        for counter in ('feature_simulations','feature_deposits'):
                            amount=int(getattr(out,counter));self._counters[counter]+=amount;metrics[counter]+=amount;work[counter]+=amount
                        actual=int(out.neural_evaluations);self._counters['native_actual_evals']+=actual;metrics['native_actual_evals']+=actual;work['evaluated']=bool(actual)
                        if code==0:self._counters['predict_successes']+=1
                        self._validate_result(out,code,PredictionResult)
                        if actual!=1 or out.terminal:raise NativeError('Nonterminal prediction failed physical-forward accounting')
                        value=dict(value_player0=out.value_player0,policy_logits=list(out.policy_logits),feature_simulations=int(out.feature_simulations),feature_deposits=int(out.feature_deposits),cache_hit=False,native_actual_evals=actual)
                        validate_prediction(value,p);_poll(cancel,deadline)
                        if self.cache_capacity:
                            self._cache[key]=(value['value_player0'],tuple(value['policy_logits']))
                            if len(self._cache)>self.cache_capacity:self._cache.popitem(last=False);self._counters['cache_evictions']+=1
                validate_prediction(value,p);_poll(cancel,deadline);outputs.append(value);metrics['evaluated_positions']+=int(bool(sum(map(sum,p['pits']))))
                # Frozen PUCT charges each returned row logically. For a stopped
                # batch, evaluated records physical forwards only (hits stay False).
            _poll(cancel,deadline);metrics['whole_call_ms']=(time.perf_counter()-started)*1000
            for value in outputs:value['inference_ms']=metrics['whole_call_ms']/len(outputs)
            self.last_metrics=metrics;return outputs
        except InferenceStopped as exc:
            metrics['whole_call_ms']=(time.perf_counter()-started)*1000;self.last_metrics=metrics;exc.work=metrics;raise
        finally:
            if running:
                with self._control._lock:self._control._running=False
            if registered:cancel._remove(self._control)
            if acquired:self.lock.release()

    def feature_vector(self,position,cancel=None,deadline=None):
        """Separate diagnostics; never called by evaluate_many or the cache."""
        p=config.position(position)
        with self.lock:
            if self.closed:raise NativeError('Evaluator is closed')
            _poll(cancel,deadline);self._begin(deadline);cells,_=self.native._array(flatten(p),7,98);out=FeatureResult();registered=False
            if cancel is not None and hasattr(cancel,'_register') and hasattr(cancel,'_remove'):cancel._register(self._control);registered=True
            try:
                code=self.native.lib.cc_puct_features(cells,p['controller'],self._control.handle,C.byref(out))
                self._validate_result(out,code,FeatureResult);_poll(cancel,deadline)
                return dict(features=list(out.x),feature_simulations=int(out.feature_simulations),feature_deposits=int(out.feature_deposits))
            except InferenceStopped as exc:
                exc.work=dict(feature_simulations=int(out.feature_simulations),feature_deposits=int(out.feature_deposits),diagnostic_only=True);raise
            finally:
                if registered:cancel._remove(self._control)
