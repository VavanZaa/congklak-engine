"""Finite source/literal admission for the additive fixed-model benchmark."""
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary
from . import config,neurodiag_common as common
from .value_diagnostics import atomic as _atomic
ROOT=common.ROOT
SCHEMA='prospective-benchmark-v4'
FOLDER=ROOT/'build/prospective-benchmark-v4'
DATA=ROOT/'data/prospective-benchmark-v4'
PLAN=FOLDER/'plan.json'
PREFLIGHT=FOLDER/'audit/preflight.json'
MODEL_PATHS=('data/quality-train-v1/model.json','data/guarded-value-pilot-v2/training/model.json','data/training-stage5-deeper-h8-f127-seed2718/model.json')
CANDIDATE_SHA='1668e8a9b2734a212102e8db0500641fe1f204f88a5303f9e915268a3b4f531a'
REQUIRED_SOURCES=tuple('congklak/prospective_benchmark_v3_'+name+'.py' for name in ('common','guard','execute','runtime','recovery','cli','worker','engineering'))+tuple('congklak/prospective_benchmark_v2_'+name+'.py' for name in ('literal','timing','native','core','adapter','cohort','reservations','extension'))+('tests/reference.py','build/prospective-benchmark-v2/native/source-projection.json','build/prospective-benchmark-v4/engineering.json','build/prospective-benchmark-v4/execution-projection.json')
need,read,sha=common.require,common.read,common.sha
_ACTIVE=ContextVar('prospective_benchmark_v2_verified_session',default=None)
_SEALS=WeakKeyDictionary()
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def ref(path):return dict(path=relative(path),sha256=sha(path))
def path_of(path):
 path=Path(path).resolve();need(any(path==b or path.is_relative_to(b) for b in (FOLDER,DATA)),'Write outside new benchmark namespace');return path
def atomic(path,body):return _atomic(path_of(path),json.loads(config.canonical(body)))
def _review(body):
 review=read(PREFLIGHT)
 need(review['schema']==SCHEMA+'-independent-preflight' and review['passed'] is True and review['independent_of_implementation'] is True and review['plan_ref']==ref(PLAN) and review['source_inputs']==body['source_inputs'],'Independent source-bound benchmark review missing/stale')
 return review
def admit_plan(require_review=True):
 body=read(PLAN)
 need(body['schema']==SCHEMA+'-plan' and body['policy_fixed'] is True and body['base_frozen'] is True and body['training_authorized'] is False,'Benchmark scope changed')
 need(set(REQUIRED_SOURCES)<=set(body['source_inputs']),'Required physical source omitted')
 boundary=body.get('entry_closure_inputs',{})
 need(type(boundary) is dict and all(body['source_inputs'].get(k)==v for k,v in boundary.items()),'Boundary source subset differs')
 need(not any(Path(k).suffix.lower() in ('.py','.cpp','.hpp','.ps1','.cmd','.dll') or k in MODEL_PATHS for k in boundary),'Executable/model sources cannot skip microchecks')
 view=_ACTIVE.get()
 if view is None:common.current(body['source_inputs'])
 else:view.verify(body)
 need(body['models']==common.pins(MODEL_PATHS) and body['models'][MODEL_PATHS[1]]==CANDIDATE_SHA,'Fixed control/final400/H8 model differs')
 for name in ('reservations_ref','cohort_ref','protected_inputs_ref'):
  r=body[name];need(body['source_inputs'].get(r['path'])==r['sha256'],'Required catalog omitted from source pins')
  if view is None or r['path'] not in boundary:need(r==ref(ROOT/r['path']),'Catalog ref changed: '+name)
 if require_review:_review(body)
 return body
class _SourceView:
 def __init__(self,body):
  need(_ACTIVE.get() is None and body==read(PLAN),'Source constructor body/session differs')
  common.current(body['source_inputs']);_review(body)
  self.full=MappingProxyType(dict(body['source_inputs']));self.compact=MappingProxyType({k:v for k,v in self.full.items() if k not in body.get('entry_closure_inputs',{})})
  transitive={}
  for reference in (body['protected_inputs_ref'],body['resource']['carry_ref']):
   need(self.full.get(reference['path'])==reference['sha256'],'Transitive catalog is not source-pinned')
   catalog=read(ROOT/reference['path'])
   for name,h in catalog.get('inputs',catalog.get('source_inputs',{})).items():
    need(name not in transitive or transitive[name]==h,'Conflicting transitive source input');transitive[name]=h
  common.current(transitive);self.transitive=MappingProxyType(transitive)
  self.transitive_compact=MappingProxyType({k:v for k,v in transitive.items() if Path(k).suffix.lower() in ('.py','.cpp','.hpp','.ps1','.cmd','.dll') or k in MODEL_PATHS})
  self.plan_sha=sha(PLAN);self.review_sha=sha(PREFLIGHT);self.closed=False
  from .prospective_benchmark_v2_literal import LiteralRules
  from .prospective_benchmark_v2_timing import GcObserver
  self.literal=LiteralRules(body['source_inputs']['tests/reference.py']);self.timing=GcObserver()
  _SEALS[self]=(self.full,self.compact,self.plan_sha,self.review_sha,self.literal,self.timing,False,self.transitive,self.transitive_compact)
 def verify(self,body):
  seal=_SEALS.get(self)
  need(type(self) is _SourceView and seal is not None and _ACTIVE.get() is self and not self.closed and
   self.full is seal[0] and self.compact is seal[1] and self.plan_sha==seal[2] and self.review_sha==seal[3] and self.literal is seal[4] and self.timing is seal[5] and seal[6] is False and self.transitive is seal[7] and self.transitive_compact is seal[8],'Source session lacks constructor-issued seal')
  need(body['source_inputs']==dict(self.full) and sha(PLAN)==self.plan_sha and sha(PREFLIGHT)==self.review_sha,'Plan/review/source inventory changed')
  common.current(self.compact);common.current(self.transitive_compact);self.literal.verify()
 def close(self):
  self.verify(read(PLAN));common.current(self.full);common.current(self.transitive)
  from . import prospective_benchmark_v2_extension as reservations
  reservations.verify(full=True)
  self.literal_close=self.literal.close();self.timing_close=self.timing.close();self.closed=True
  seal=_SEALS[self];_SEALS[self]=(*seal[:6],True,*seal[7:])
@contextmanager
def worker_views():
 body=admit_plan();view=_SourceView(body);token=_ACTIVE.set(view)
 try:yield view
 finally:
  try:view.close()
  finally:_ACTIVE.reset(token)
def in_view():return _ACTIVE.get() is not None
def literal_rules():
 view=_ACTIVE.get();need(view is not None and not view.closed,'Literal calls require a finite source session');view.literal.verify();return view.literal
def timing_observer():
 view=_ACTIVE.get();need(view is not None and not view.closed,'Timing calls require a finite source session');return view.timing
def close_views():
 need(not in_view(),'Close finite source/literal session before durable worker completion');admit_plan();return True
def current_inputs(inputs):
 """Fresh bytes, or a constructor-issued transitive boundary proof plus code checks."""
 view=_ACTIVE.get()
 if view is None:return common.current(inputs)
 view.verify(read(PLAN))
 combined=dict(view.full,**dict(view.transitive))
 need(all(combined.get(k)==v for k,v in inputs.items()),'Input set was not covered by the full constructor proof')
 common.current({k:v for k,v in inputs.items() if Path(k).suffix.lower() in ('.py','.cpp','.hpp','.ps1','.cmd','.dll') or k in MODEL_PATHS})
 return True
