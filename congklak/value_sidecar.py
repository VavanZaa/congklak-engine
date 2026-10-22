"""Versioned, value-only native sidecar. Initialization and publication never optimize.

The eight base tensors, relay encoder and policy arithmetic are immutable. The
new export explicitly binds its input reservation, sources and native build.
"""
from __future__ import annotations
from collections import OrderedDict
import argparse
import ctypes as C
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from . import config,features_relay,quality_search
from .native import ROOT,NativeError,Control,flatten
from .neural_relay import Model as BaseModel,NeuralNative,sha,need,flatten_weights,rounded,SCALE
from .puct_cpu_inference import CpuBatchEvaluator,PredictionResult,FeatureResult,COUNTERS,BUILD_FLAGS,BUILD_DISPATCH

SCHEMA='congklak-value-sidecar-v1'
ABI='congklak-value-sidecar-native-v1'
BASE=ROOT/'data/quality-train-v1/model.json'
BASE_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
INITIAL_TENSOR_SHA='6f9fef8c07f00e755c2ec12eee8c2f6b3b7818cf05470f208ad2084de3e91256'
DLL=ROOT/'build/congklak_value_sidecar.dll'
BUILD=ROOT/'build/value-sidecar-build.json'
BUILD_INPUTS={'native/engine.hpp','native/rules.cpp','native/search.cpp','native/neural_relay.cpp','native/neural_puct_cpu.cpp',
              'native/neural_value_sidecar.cpp','tools/build-value-sidecar.cmd','tools/build-value-sidecar.ps1'}
DEPENDENCIES=('congklak/value_sidecar.py','congklak/config.py','congklak/native.py','congklak/neural_relay.py',
              'congklak/features_relay.py','congklak/features_v3.py','congklak/puct_cpu_inference.py','congklak/phase6_inference.py','congklak/puct.py','congklak/quality_search.py','tools/export-value-sidecar.py')
SHAPES={'A':(16,127),'b':(16,),'v':(1,16),'c':(1,)}
GATE={'minimum_active':25,'maximum_active':74,'outside':'direct-frozen-base-call','terminal':'exact-frozen-base'}
PROTOCOL='value-sidecar/single-feature-pass/frozen-policy/strict-double-v1'
HOOKS=('before_model_commit','after_model_commit','before_certificate_commit','after_certificate_commit')

def _json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _fp32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def tensor_sha(weights):
    h=hashlib.sha256()
    for name in sorted(SHAPES):
        values=list(flatten_weights(weights[name]));h.update(name.encode());h.update(struct.pack('<'+'f'*len(values),*values))
    return h.hexdigest()
def _weights(weights):
    need(isinstance(weights,dict) and set(weights)==set(SHAPES),'Unknown sidecar tensor fields')
    out={}
    for name,dims in SHAPES.items():
        row=weights[name];need(isinstance(row,list) and len(row)==dims[0],'Wrong sidecar tensor dimensions')
        if len(dims)==2:need(all(isinstance(v,list) and len(v)==dims[1] for v in row),'Wrong sidecar matrix dimensions')
        values=list(flatten_weights(row));need(all(type(v) in (int,float) and math.isfinite(v) for v in values),'Nonfinite sidecar tensor')
        # The exported weights are precisely FP32, represented losslessly in JSON.
        originals=[x for item in row for x in item] if len(dims)==2 else row
        need(all(type(x) in (int,float) and x==f for x,f in zip(originals,values)),'Sidecar weights must be exact FP32 values')
        out[name]=[values[i*dims[1]:(i+1)*dims[1]] for i in range(dims[0])] if len(dims)==2 else values
    return out
def build_record():
    r=_json(BUILD)
    need(isinstance(r,dict) and set(r)=={'schema','visual_studio','flags','dispatch','inputs','dll_sha256'},'Sidecar build witness fields differ')
    need(r['schema']=='value-sidecar-native-build-v1' and r['dll_sha256']==sha(DLL),'Sidecar DLL differs')
    need(set(r['inputs'])==BUILD_INPUTS and r['flags']==BUILD_FLAGS and r['dispatch']==BUILD_DISPATCH,'Sidecar build protocol differs')
    need(isinstance(r['visual_studio'],str) and Path(r['visual_studio']).is_absolute(),'Compiler identity missing')
    for name,digest in r['inputs'].items():need(sha(ROOT/name)==digest,'Sidecar native source changed: '+name)
    return r
def bindings(manifest):
    path=Path(manifest).resolve();r=build_record()
    need(path.is_file(),'Input reservation manifest missing');need(sha(BASE)==BASE_SHA,'Frozen base export changed')
    reservation_sources=({name:sha(ROOT/name) for name in ('congklak/valuepilot_reservations.py','tools/reserve-valuepilot.py')}
                         if path==ROOT/'data/neural-value-pilot-v1/reservations/manifest.json' else None)
    return dict(base_file_sha256=BASE_SHA,base_weights_sha256=BaseModel(BASE).id,
        input_manifest={'path':str(path),'sha256':sha(path)},gate=GATE,encoder=features_relay.ENCODER127,rules_digest=config.RULES_DIGEST,
        abi=ABI,protocol=PROTOCOL,numeric_backend='strict-double/scalar-or-AVX2/separate-multiply-add/FP32-export-weights',
        build_record_sha256=sha(BUILD),dll_sha256=r['dll_sha256'],build_inputs=r['inputs'],
        search={'protocol':quality_search.PROTOCOL,'c_puct':0.25,'cache_capacity':0,'max_backups':100000,'max_nodes':2048,'root_noise':None},
        sources={name:sha(ROOT/name) for name in DEPENDENCIES},reservation_interface_sources=reservation_sources)
def make_export(weights,manifest,provenance):
    w=_weights(weights);b=bindings(manifest)
    model=dict(schema=SCHEMA,bindings=b,base_export=_json(BASE),branch_weights=w,branch_weights_sha256=tensor_sha(w),
        parameter_counts={'base_frozen':12872,'branch_trainable':2065,'total':14937},provenance=provenance)
    return {'model':model,'sha256':config.digest(model)}

class Model:
    def __init__(self,path):
        self.path=Path(path).resolve();need(not (self.path.parent/'quarantine.json').exists(),'Sidecar is quarantined')
        e=_json(self.path);need(isinstance(e,dict) and set(e)=={'model','sha256'},'Unknown sidecar envelope')
        m=e['model'];need(isinstance(m,dict) and set(m)=={'schema','bindings','base_export','branch_weights','branch_weights_sha256','parameter_counts','provenance'},'Unknown sidecar model fields')
        need(m['schema']==SCHEMA and config.digest(m)==e['sha256'],'Unknown/damaged sidecar export')
        need(isinstance(m['bindings'],dict) and isinstance(m['bindings'].get('input_manifest'),dict),'Missing sidecar bindings')
        manifest=m['bindings']['input_manifest'];need(set(manifest)=={'path','sha256'},'Unknown reservation binding fields')
        need(m['bindings']==bindings(manifest['path']),'Sidecar sources/input/build identity changed')
        self.base=BaseModel(BASE);need(m['base_export']==self.base.envelope,'Embedded frozen base changed')
        need(m['parameter_counts']=={'base_frozen':12872,'branch_trainable':2065,'total':14937},'Parameter contract differs')
        self.weights=_weights(m['branch_weights']);need(tensor_sha(self.weights)==m['branch_weights_sha256'],'Sidecar tensor digest changed')
        need(isinstance(m['provenance'],dict),'Missing sidecar provenance')
        self.envelope=e;self.identity=m['bindings'];self.id=e['sha256'];self.file_sha256=sha(self.path)
        need(self.id!=self.base.id,'Candidate identity must differ from base')
        self.width=64;self.input_count=127;self.encoder=self.base.encoder
        values=[]
        for name in ('A','b','v','c'):values.extend(flatten_weights(self.weights[name]))
        self.array=(C.c_float*2065)(*values)
    def reference(self,position):
        active=sum(map(sum,position['pits']))
        if not 25<=active<=74:return self.base.reference(position)
        x,d,r=features_relay.encode(position,127)
        h=x
        for prefix in ('layers.0','layers.2'):
            h=[max(0,sum(a*v for a,v in zip(row,h))+b) for row,b in zip(self.base.weights[prefix+'.weight'],self.base.weights[prefix+'.bias'])]
        u=sum(a*v for a,v in zip(self.base.weights['layers.4.weight'][0],h))+self.base.weights['layers.4.bias'][0]
        logits=[sum(a*v for a,v in zip(row,h))+b for row,b in zip(self.base.weights['policy.weight'],self.base.weights['policy.bias'])]
        hidden=[max(0,sum(a*v for a,v in zip(row,x))+b) for row,b in zip(self.weights['A'],self.weights['b'])]
        correction=sum(a*v for a,v in zip(self.weights['v'][0],hidden))+self.weights['c'][0]
        residual=r*math.tanh(u+correction)
        return d+residual,d*SCALE+rounded(residual*SCALE),logits

class SidecarResult(C.Structure):
    _fields_=PredictionResult._fields_+[('branch_calls',C.c_uint64),('branch_bypasses',C.c_uint64),('base_ms',C.c_double),('branch_ms',C.c_double),('value_units',C.c_int32),('reserved',C.c_int32)]

class SidecarEvaluator(CpuBatchEvaluator):
    """Frozen evaluator protocol with a separately identified native extension."""
    def __init__(self,modelpath,cache_capacity=0,max_batch=32,backend='auto'):
        config.integer(cache_capacity,'cache_capacity',0,4096);config.integer(max_batch,'max_batch',1,32)
        need(backend in ('auto','scalar','avx2'),'Unknown sidecar CPU backend')
        started=time.perf_counter();self.model=modelpath if isinstance(modelpath,Model) else Model(modelpath);record=build_record()
        self.lock=threading.RLock();self.closed=False;self.cache_capacity=cache_capacity;self.max_batch=max_batch
        self._cache=OrderedDict();self._counters={key:0 for key in COUNTERS};self.last_metrics={};self._control=None
        self.branch_counters={'branch_calls':0,'branch_bypasses':0,'base_ms':0.,'branch_ms':0.}
        self.native=NeuralNative(self.model.base,path=DLL,backend=backend,mode='value-none');lib=self.native.lib;self._sidecar=None
        try:
            lib.cc_sidecar_abi.argtypes=[];lib.cc_sidecar_abi.restype=C.c_char_p
            lib.cc_sidecar_result_size.argtypes=[];lib.cc_sidecar_result_size.restype=C.c_uint32
            lib.cc_sidecar_layout.argtypes=[C.c_int];lib.cc_sidecar_layout.restype=C.c_uint32
            need(lib.cc_sidecar_abi().decode()==ABI and lib.cc_sidecar_result_size()==C.sizeof(SidecarResult),'Sidecar ABI/layout mismatch')
            need([lib.cc_sidecar_layout(i) for i in range(4)]==[getattr(SidecarResult,k).offset for k in self.branch_counters],'Sidecar ABI offsets differ')
            need(lib.cc_sidecar_layout(4)==SidecarResult.value_units.offset,'Sidecar score-unit ABI offset differs')
            lib.cc_sidecar_create.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.c_int];lib.cc_sidecar_create.restype=C.c_void_p
            lib.cc_sidecar_destroy.argtypes=[C.c_void_p];lib.cc_sidecar_destroy.restype=None
            self._sidecar=lib.cc_sidecar_create(self.native.handle,self.model.array,2065);need(self._sidecar,'Sidecar native admission failed')
            lib.cc_puct_begin.argtypes=[C.c_void_p,C.c_uint64];lib.cc_puct_begin.restype=C.c_int
            lib.cc_puct_features.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.POINTER(FeatureResult)];lib.cc_puct_features.restype=C.c_int
            lib.cc_sidecar_predict.argtypes=[C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p,C.POINTER(SidecarResult)];lib.cc_sidecar_predict.restype=C.c_int
            self._predict_native=lib.cc_sidecar_predict
            # The inherited batch/cache/cancellation implementation remains intact.
            # Translate the extension result into its frozen result prefix.
            def predict(cells,actor,_base,control,output):
                result=SidecarResult();code=self._predict_native(cells,actor,self._sidecar,control,C.byref(result))
                need(result.abi_version==1 and result.struct_size==C.sizeof(SidecarResult) and result.status==code and code in (0,1,2),'Damaged sidecar result ABI/status')
                for key in self.branch_counters:self.branch_counters[key]+=getattr(result,key)
                C.memmove(output,C.byref(result),C.sizeof(PredictionResult))
                C.cast(output,C.POINTER(PredictionResult)).contents.struct_size=C.sizeof(PredictionResult)
                return code
            lib.cc_puct_predict=predict
            self._control=Control(self.native,max_depth=128,time_ms=0,node_budget=0,memory_mb=8192,tt_mb=0,use_pvs=False)
            self.backend=self.native.backend
            self.identity=dict(schema=SCHEMA,protocol=PROTOCOL,abi=ABI,backend=self.backend,requested_backend=backend,max_batch=max_batch,
                model_file_sha256=self.model.file_sha256,model_sha256=self.model.id,bindings=self.model.identity,
                cache={'capacity':cache_capacity,'key':'full raw16 + inputcontroller, private to complete sidecar identity','lifetime':'reset every decision'},
                measurement={'base_ms':'feature extraction + frozen base forward','branch_ms':'branch dense/tanh only','whole_call_ms':'inherited entire provider call'})
            self.startup_ms=(time.perf_counter()-started)*1000
        except BaseException:
            if self._sidecar:lib.cc_sidecar_destroy(self._sidecar)
            if self._control:self._control.close()
            self.native.close();raise
    def snapshot_counters(self):
        with self.lock:return {**super().snapshot_counters(),**self.branch_counters}
    def evaluate_units(self,position):
        """Separate scalar conformance operation, excluded from provider counters.

        Uses the native residual directly for round-away-from-zero scale256.
        The PUCT provider continues to use unquantized values.
        """
        p=config.position(position)
        with self.lock:
            need(not self.closed,'Evaluator is closed');self._begin(None)
            cells,_=self.native._array(flatten(p),7,98);out=SidecarResult()
            code=self._predict_native(cells,p['controller'],self._sidecar,self._control.handle,C.byref(out))
            self._validate_result(out,code,SidecarResult)
            return dict(value_player0=out.value_player0,value_units=out.value_units,policy_logits=list(out.policy_logits),
                feature_simulations=int(out.feature_simulations),feature_deposits=int(out.feature_deposits),
                native_actual_evals=int(out.neural_evaluations),branch_calls=int(out.branch_calls),branch_bypasses=int(out.branch_bypasses),diagnostic_only=True)
    def close(self):
        with self.lock:
            if not self.closed:
                self.native.lib.cc_sidecar_destroy(self._sidecar);self._sidecar=None
                super().close()

def _atomic(path,body):
    temp=path.with_name(path.name+'.pending');data=(config.canonical(body)+'\n').encode()
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
@contextmanager
def _writer(directory):
    import msvcrt
    path=directory/'writer.lock'
    with path.open('a+b') as f:
        if not path.stat().st_size:f.write(b'0');f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise NativeError('Sidecar publisher already running') from exc
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
def _hook(directory,name,pause_at,fixture):
    if pause_at!=name:return
    need(fixture,'Pause hooks require explicit engineering-fixture flag')
    _atomic(directory/'pause.json',{'event':name,'pid':os.getpid()})
    print(config.canonical({'event':name,'pid':os.getpid()}),flush=True)
    while True:time.sleep(.1)
def _initial_model(model):
    provenance=model.envelope['model']['provenance']
    fields={'initialization_seed','generator','torch','optimizer_steps','cuda_allocations','initialization','engineering_fixture','training_eligible'}
    need(set(provenance)==fields and provenance['initialization_seed']==20261016 and provenance['generator']=='isolated torch.Generator(cpu)'
         and isinstance(provenance['torch'],str) and provenance['torch'] and provenance['optimizer_steps']==0 and provenance['cuda_allocations']==0
         and provenance['initialization']=='A,b uniform +/-1/sqrt127; v,c exact zero'
         and type(provenance['engineering_fixture']) is bool and provenance['training_eligible'] is (not provenance['engineering_fixture']),
         'Step-zero provenance differs')
    need(tensor_sha(model.weights)==INITIAL_TENSOR_SHA and model.weights['v']==[[0.]*16] and model.weights['c']==[0.],
         'Initialization tensors differ from deterministic zero-output recipe')
    if provenance['training_eligible']:
        from . import valuepilot_reservations
        manifest=Path(model.identity['input_manifest']['path'])
        need(manifest==valuepilot_reservations.DEFAULT.resolve()/'manifest.json','Production initialization cannot use scratch reservations')
        admitted=valuepilot_reservations.admit(manifest.parent)
        need(all(admitted.get('counts',{}).get(k)==v for k,v in
             {'train':512,'fit_check':128,'fit_cases':32,'arena':24,'globally_distinct_components':664}.items()),
             'Production initialization requires admitted complete reservations')
    return provenance
def verify(directory,manifest=None):
    """Completed validation is read-only and happens before any writer lock."""
    directory=Path(directory).resolve();cert=_json(directory/'certificate.json')
    need(isinstance(cert,dict) and set(cert)=={'body','sha256'} and config.digest(cert['body'])==cert['sha256'],'Sidecar certificate corrupt')
    body=cert['body'];need(set(body)=={'schema','model_sha256','model_file_sha256','bindings','optimizer_steps'},'Unknown initialization certificate')
    model=Model(directory/'model.json')
    provenance=_initial_model(model)
    need(body==dict(schema='value-sidecar-initialization-certificate-v1',model_sha256=model.id,model_file_sha256=model.file_sha256,bindings=model.identity,optimizer_steps=0),'Initialization evidence differs')
    if manifest is not None:need(model.identity['input_manifest']=={'path':str(Path(manifest).resolve()),'sha256':sha(manifest)},'Initialization input differs')
    if (directory/'COMPLETED.json').exists():need(_json(directory/'COMPLETED.json')=={'certificate_sha256':cert['sha256']},'Completion marker differs')
    return {'schema':'value-sidecar-initialization-status-v1','state':'complete','model_sha256':model.id,'model_file_sha256':model.file_sha256,'optimizer_steps':0,'policy_fixed':True,'base_tensors_frozen':8,
            'engineering_fixture':provenance['engineering_fixture'],'training_eligible':provenance['training_eligible']}
def initialize(manifest,output,pause_at=None,fixture=False):
    directory=Path(output).resolve()
    need(pause_at is None or (pause_at in HOOKS and fixture),'Invalid or non-fixture pause hook')
    need(not fixture or directory.is_relative_to(ROOT/'build'),'Engineering exports must remain in build fixtures')
    if (directory/'certificate.json').exists() or (directory/'COMPLETED.json').exists():return verify(directory,manifest)
    if not fixture:
        from . import valuepilot_reservations
        need(Path(manifest).resolve()==valuepilot_reservations.DEFAULT.resolve()/'manifest.json','Production initialization cannot use scratch reservations')
        admitted=valuepilot_reservations.admit(Path(manifest).resolve().parent)
        need(all(admitted.get('counts',{}).get(k)==v for k,v in
             {'train':512,'fit_check':128,'fit_cases':32,'arena':24,'globally_distinct_components':664}.items()),'Production initialization requires admitted complete reservations')
    directory.mkdir(parents=True,exist_ok=True)
    with _writer(directory):
        if (directory/'certificate.json').exists() or (directory/'COMPLETED.json').exists():return verify(directory,manifest)
        expected=bindings(manifest);modelpath=directory/'model.json'
        if modelpath.exists():
            model=Model(modelpath);need(model.identity==expected,'Resumed initialization identity differs')
            _initial_model(model)
            need(model.envelope['model']['provenance']['engineering_fixture'] is fixture,'Initialization fixture/production role differs')
        else:
            import torch
            torch.set_num_threads(1)
            if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            generator=torch.Generator(device='cpu');generator.manual_seed(20261016);limit=1/math.sqrt(127)
            w={'A':torch.empty((16,127),dtype=torch.float32,device='cpu').uniform_(-limit,limit,generator=generator).tolist(),
               'b':torch.empty((16,),dtype=torch.float32,device='cpu').uniform_(-limit,limit,generator=generator).tolist(),
               'v':[[0.]*16],'c':[0.]}
            export=make_export(w,manifest,{'initialization_seed':20261016,'generator':'isolated torch.Generator(cpu)','torch':torch.__version__,
                'optimizer_steps':0,'cuda_allocations':0,'initialization':'A,b uniform +/-1/sqrt127; v,c exact zero',
                'engineering_fixture':fixture,'training_eligible':not fixture})
            _hook(directory,'before_model_commit',pause_at,fixture);_atomic(modelpath,export)
            _hook(directory,'after_model_commit',pause_at,fixture);model=Model(modelpath)
        _initial_model(model)
        body=dict(schema='value-sidecar-initialization-certificate-v1',model_sha256=model.id,model_file_sha256=model.file_sha256,bindings=model.identity,optimizer_steps=0)
        cert={'body':body,'sha256':config.digest(body)}
        _hook(directory,'before_certificate_commit',pause_at,fixture);_atomic(directory/'certificate.json',cert)
        _hook(directory,'after_certificate_commit',pause_at,fixture);_atomic(directory/'COMPLETED.json',{'certificate_sha256':cert['sha256']})
    return verify(directory,manifest)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('initialize','verify'))
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pause-at',choices=HOOKS);parser.add_argument('--engineering-fixture',action='store_true')
    args=parser.parse_args()
    try:
        result=initialize(args.manifest,args.output,args.pause_at,args.engineering_fixture) if args.operation=='initialize' else verify(args.output,args.manifest)
        print(config.canonical(result),flush=True);return 0
    except Exception as exc:print(config.canonical({'state':'hold','error':str(exc),'optimizer_steps':0}),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
