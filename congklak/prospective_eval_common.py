"""Admission and immutable writes for the additive prospective evaluation."""
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/prospective-eval-v1'
DATA=ROOT/'data/prospective-eval-v1'
PLAN=FOLDER/'plan.json'
PREFLIGHT=FOLDER/'audit/preflight.json'
SCHEMA='prospective-eval-v1'
MODEL_PATHS=('data/quality-train-v1/model.json','data/guarded-value-pilot-v2/training/model.json',
             'data/training-stage5-deeper-h8-f127-seed2718/model.json')
CANDIDATE_SHA='1668e8a9b2734a212102e8db0500641fe1f204f88a5303f9e915268a3b4f531a'
REQUIRED_SOURCES=('congklak/prospective_eval_common.py','congklak/prospective_eval_reservations.py',
 'congklak/prospective_eval_guard.py','congklak/prospective_eval_execute.py',
 'congklak/prospective_eval_runtime.py','congklak/prospective_eval_recovery.py',
 'congklak/prospective_eval_annex.py','build/prospective-eval-v1/annex/policy-v2.json',
 'congklak/prospective_eval_cli.py','congklak/prospective_eval_worker.py',
 'build/prospective-eval-v1/source-projection.json')
need,read,sha=common.require,common.read,common.sha
_ACTIVE=ContextVar('prospective_evaluation_verified_sources',default=None)
_SEALS=WeakKeyDictionary()
# These exact immutable data objects are read and hashed in full at entry and
# again at closure. Executable sources and compact catalogs stay freshly hashed
# on every microcheck. No stat/mtime observation can issue an admission proof.
_LARGE=frozenset(('build/prospective-eval-v1/reservations/components.bin',
                  'build/prospective-eval-v1/reservations/report.json'))

def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def ref(path):return dict(path=relative(path),sha256=sha(path))
def path_of(path):
    path=Path(path).resolve()
    need(any(path==base or path.is_relative_to(base) for base in (FOLDER,DATA)),
         'Write outside the prospective evaluation namespace')
    return path
def atomic(path,body):return _atomic(path_of(path),json.loads(config.canonical(body)))

def admit_plan(require_review=True):
    body=read(PLAN)
    need(body['schema']==SCHEMA+'-plan' and body['policy_fixed'] is True and
         body['base_frozen'] is True and body['training_authorized'] is False,
         'Prospective plan scope changed')
    need(set(REQUIRED_SOURCES)<=set(body['source_inputs']), 'Required physical source omitted')
    view=_ACTIVE.get()
    if view is None:common.current(body['source_inputs'])
    else:view.verify(body)
    need(body['models']==common.pins(MODEL_PATHS) and
         body['models'][MODEL_PATHS[1]]==CANDIDATE_SHA,'Fixed final400/control/H8 model changed')
    r=body['reservations_ref']
    if view is not None and r['path'] in _LARGE:
        need(view.full.get(r['path'])==r['sha256'],'Scoped reservation reference changed')
    else:need(r==ref(ROOT/r['path']),'Prospective reservation binding changed')
    if require_review:
        review=read(PREFLIGHT)
        need(review['schema']==SCHEMA+'-independent-preflight' and review['passed'] is True and
             review['independent_of_implementation'] is True and review['plan_ref']==ref(PLAN) and
             review['source_inputs']==body['source_inputs'], 'Independent prospective source review missing/stale')
    return body

class _SourceView:
    def __init__(self,body):
        need(_ACTIVE.get() is None,'Nested source views are forbidden')
        need(body==read(PLAN),'Source constructor body differs from the saved plan')
        common.current(body['source_inputs'])
        review=read(PREFLIGHT)
        need(review['schema']==SCHEMA+'-independent-preflight' and review['passed'] is True and
             review['independent_of_implementation'] is True and review['plan_ref']==ref(PLAN) and
             review['source_inputs']==body['source_inputs'],'Source constructor lacks independent review')
        self.full=MappingProxyType(dict(body['source_inputs']))
        self.compact=MappingProxyType({k:v for k,v in self.full.items() if k not in _LARGE})
        self.plan_sha256=sha(PLAN);self.review_sha256=sha(PREFLIGHT);self.closed=False
        _SEALS[self]=(self.full,self.compact,self.plan_sha256,self.review_sha256,False)
    def verify(self,body):
        seal=_SEALS.get(self)
        need(type(self) is _SourceView and seal is not None and _ACTIVE.get() is self and not self.closed
             and self.full is seal[0] and self.compact is seal[1] and self.plan_sha256==seal[2]
             and self.review_sha256==seal[3] and seal[4] is False,'Source view lacks its constructor-issued live seal')
        need(body['source_inputs']==dict(self.full) and sha(PLAN)==self.plan_sha256
             and sha(PREFLIGHT)==self.review_sha256,'Scoped plan/review/source inventory changed')
        common.current(self.compact)
    def close(self):
        need(not self.closed,'Source view closed twice')
        self.verify(read(PLAN));common.current(self.full)
        from . import prospective_eval_reservations as reservations
        reservations.verify()
        self.closed=True;seal=_SEALS[self];_SEALS[self]=(*seal[:4],True)

@contextmanager
def worker_views():
    """A finite full-byte source proof, rechecked before durable completion."""
    body=admit_plan();view=_SourceView(body);token=_ACTIVE.set(view)
    try:yield view
    finally:
        try:view.close()
        finally:_ACTIVE.reset(token)

def close_views():
    """Rehash every admitted source/model and the complete new reservation snapshot."""
    need(_ACTIVE.get() is None,'Close the finite source scope before completion')
    admit_plan()
    from . import prospective_eval_reservations as reservations
    reservations.verify()
    return True
