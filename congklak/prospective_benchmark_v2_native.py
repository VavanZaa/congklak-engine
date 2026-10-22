"""Fixed-model private providers with a strictly default-deny high-state guard."""
from contextlib import ExitStack
from functools import lru_cache
import ctypes as C,hashlib,importlib.util,json,struct,sys,uuid
from pathlib import Path
from . import config,neurodiag_common as common
from . import guarded_value_pilot_v2_native as original,native_trace_fast_v2 as frozen
ROOT=common.ROOT
DIRECTORY=ROOT/'build/prospective-benchmark-v2/native'
DLL=DIRECTORY/'congklak_prospective_benchmark_v2.dll'
BUILD=DIRECTORY/'build.json'
CORE=ROOT/'congklak/prospective_benchmark_v2_core.py'
ADAPTER=ROOT/'congklak/prospective_benchmark_v2_adapter.py'
MODEL=ROOT/'data/guarded-value-pilot-v2/training/model.json'
MODEL_SHA='1668e8a9b2734a212102e8db0500641fe1f204f88a5303f9e915268a3b4f531a'
POLICY=dict(original.POLICY)
TEACHER=dict(original.TEACHER)
need,read,sha=common.require,common.read,common.sha
TraceHold=frozen.TraceHold
def _same(a,b):return config.canonical(a)==config.canonical(b)
def candidate_binding(path=None):
 path=MODEL if path is None else Path(path).resolve()
 need(path==MODEL and sha(path)==MODEL_SHA,'Only the unchanged final400 model is admitted')
 binding=original.candidate_binding(path)
 need(binding['provenance']['kind']=='fixed-final-step' and binding['provenance']['optimizer_steps']==400,'Model provenance changed')
 return binding
def _check_binding(binding):
 need(type(binding) is dict and _same(binding,candidate_binding(ROOT/binding['path'])),'Fixed candidate binding changed')
 return binding
def sidecar_model(path):
 _check_binding(candidate_binding(path));return original.sidecar_model(path)
def input_pins():
 paths=[Path(__file__),CORE,ADAPTER,DLL,BUILD,DIRECTORY/'source-projection.json',ROOT/'build/prospective-benchmark-v2/project_execution.py']
 paths.extend(ROOT/name for name in read(BUILD)['inputs'])
 paths.extend(ROOT/name for name in original.input_pins())
 return common.pins(paths)
def source_identity(binding=None):
 binding=candidate_binding() if binding is None else _check_binding(binding)
 record=read(BUILD)
 need(record['dll_sha256']==sha(DLL) and record['maximum_events']==16000000 and record['maximum_unique_owned']==4000000 and record['maximum_claims']==8000000 and record['claim_slots']==16777216,'Native capacity/build changed')
 for name,h in record['inputs'].items():need(sha(ROOT/name)==h,'Native input changed: '+name)
 projection=read(DIRECTORY/'source-projection.json')
 need(projection['unknown_high_default']==2 and projection['search_math_unchanged'] is True,'Strict guard projection changed')
 for name,entry in projection['copies'].items():
  a,b=ROOT/entry['original'],ROOT/entry['generated']
  need(sha(a)==entry['original_sha256'] and sha(b)==entry['generated_sha256'],'Physical projection changed: '+name)
  if name=='trace.cpp':
   before=b'return i!=statics.end()&&i->key==key?i->value:0;';after=b'return i!=statics.end()&&i->key==key?i->value:2;'
   need(a.read_bytes().count(before)==1 and b.read_bytes()==a.read_bytes().replace(before,after),'Strict default-deny delta differs')
  elif name in ('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp','trace.hpp'):
   need(a.read_bytes()==b.read_bytes(),'Frozen algorithm or trace ABI changed')
 base=frozen.source_identity();models=dict(base['model_sha256']);models['candidate']=binding['sha256']
 return dict(base,model_sha256=models,candidate_binding=binding,module_sha256=sha(__file__),dll_sha256=sha(DLL),build_sha256=sha(BUILD),
  native_sources=record['inputs'],projected_adapter_sha256=sha(ADAPTER),projected_fast_core_sha256=sha(CORE),
  source_projection_sha256=sha(DIRECTORY/'source-projection.json'),adapter_projection_sha256=sha(DIRECTORY/'source-projection.json'),
  maximum_events=16000000,maximum_unique_owned=4000000,maximum_claims=8000000,arena_maximum_events=2000000,
  fast_trace_protocol='complete-packed24/native-partition/SHA256/high15/staged-guard-v2-timed-pending',
  puct_witness=__import__('congklak.trace_puct_witness_v1',fromlist=['build_identity']).build_identity(),
  unknown_high_default_deny=True,epoch_scope='prospective-benchmark-v2-certified-joint-frontier',frozen_policy=True,production_admission=False)
def _load(path,stem):
 name='congklak._prospective_benchmark_v2_'+stem+'_'+uuid.uuid4().hex
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
@lru_cache(maxsize=1)
def _configured_cached(binding_json):
 binding=json.loads(binding_json);_check_binding(binding);core=_load(CORE,'core');reader=_load(ADAPTER,'reader')
 core.TraceHold=TraceHold;models=dict(frozen.MODEL_SHA);models['candidate']=binding['sha256']
 reader.MODEL_SHA=dict(models);reader.CANDIDATE=MODEL;core.reader=reader;core.source_identity=lambda:source_identity(binding)
 old_private=core._private_adapter
 def private(owner):
  module=old_private(owner);module.CANDIDATE=MODEL;module.MODEL_SHA=dict(models);return module
 core._private_adapter=private
 return core
def _configured(binding):return _configured_cached(config.canonical(_check_binding(binding)))
class Providers:
 def __init__(self,stack,capacity=2000000,binding=None,policy=None):
  need(isinstance(stack,ExitStack) and binding is not None,'Explicit provider lifetime and frozen binding required')
  need(policy is None or _same(policy,POLICY),'Fixed common timing policy changed')
  self.binding=_check_binding(binding);self.core=_configured(self.binding)
  self.provider=self.core.open_providers(stack,capacity,dict(POLICY));self.native=self.provider.native;self.identity=self.provider.identity
  self.allocation_receipt=self.provider.allocation_receipt
  need(_same(self.identity,source_identity(self.binding)),'Provider identity differs')
 def attach_guard(self,compiled):return self.provider.attach_guard(compiled)
 def decide(self,*args,**kwargs):return self.provider.decide(*args,**kwargs)
 def reset_all(self,*args,**kwargs):return self.provider.reset_all(*args,**kwargs)
 def lifecycle(self):return self.provider.lifecycle()
def open_providers(stack,capacity=2000000,binding=None,policy=None):return Providers(stack,capacity,binding,policy)
def archive_decision(transient):
 from .prospective_benchmark_v2_core import archive_decision as encode
 return encode(transient)
archive_attempt=archive_decision
def validate_decision(answer,engine,state,decision_id,identity=None):
 bound=answer['trace']['binding']['provider_identity'];need(identity is None or _same(bound,identity),'Saved native identity differs')
 need(_same(bound,source_identity(bound['candidate_binding'])),'Saved source identity changed')
 return _configured(bound['candidate_binding']).validate_decision(answer,engine,state,decision_id,bound)
def trace_bytes(trace):return original.trace_bytes(trace)
def iter_trace_states(trace):yield from original.iter_trace_states(trace)
def read_attempt_sink(path):return original.read_attempt_sink(path)
def guard_projection(records,allowed_ordinal):
 need(type(records) is bytes and len(records)%19==0 and type(allowed_ordinal) is int,'Malformed compact guard input')
 return hashlib.sha256(b''.join(q+struct.pack('<I',0 if ordinal==allowed_ordinal else 2) for q,ordinal in struct.iter_unpack('<15sI',records))).hexdigest()
