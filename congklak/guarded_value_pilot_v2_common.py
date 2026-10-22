"""Additive pilot identity; every previously sealed byte remains immutable."""
import json
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from pathlib import Path
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/guarded-value-pilot-v2'
DATA=ROOT/'data/guarded-value-pilot-v2'
PLAN=FOLDER/'plan.json'
PREFLIGHT=FOLDER/'audit/preflight.json'
PRIOR=ROOT/'build/footprint-screen-v1/acceptance.json'
SCHEMA='guarded-value-pilot-v2'
MODELS=('data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
        'data/training-stage5-deeper-h8-f127-seed2718/model.json')
need,read,sha=common.require,common.read,common.sha
PRIOR_SHA='9648d0c7b3d18555b52e8fa038f55b24aa19a09d24eaaf454f9ce02398d72714'
INHERITED_NS=8451261000000
_VIEW=ContextVar('guarded_value_pilot_v2_source_view',default=None)
_BASELINE_CACHE=None

def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def path_of(path):
    p=Path(path).resolve()
    need(any(p==b or p.is_relative_to(b) for b in (FOLDER.resolve(),DATA.resolve())), 'New pilot namespace differs')
    return p
def atomic(path,body):return _atomic(path_of(path),json.loads(config.canonical(body)))
def ref(path):return dict(path=relative(path),sha256=sha(path))

def baseline():
    global _BASELINE_CACHE
    path=FOLDER/'baseline.json'
    if path.exists():
        stamp=sha(path)
        if _BASELINE_CACHE is not None and _BASELINE_CACHE[0]==stamp:
            body=json.loads(_BASELINE_CACHE[1])
        else:
            body=read(path);_BASELINE_CACHE=(stamp,config.canonical(body))
        need(body['prior_acceptance']==ref(PRIOR) and body['inherited_charged_ns']==INHERITED_NS,'Prior baseline differs')
        return body
    need(sha(PRIOR)==PRIOR_SHA,'Accepted screening marker changed')
    marker=read(PRIOR);cref=marker['closure_ref'];closure_path=ROOT/cref['path']
    need(marker['passed'] is True and marker['reservation_admission'] is True and sha(closure_path)==cref['sha256'], 'Screened reservations not accepted')
    closure=read(closure_path);inputs=dict(closure['inputs'])
    inputs.update(common.pins([PRIOR,closure_path]))
    need(closure['passed'] is True and len(closure['inputs'])==closure['inputs_count']==93296,'Expected-byte closure differs')
    body=dict(schema=SCHEMA+'-baseline',prior_acceptance=ref(PRIOR),prior_closure=cref,
              inputs=inputs,inputs_count=len(inputs),inherited_charged_ns=INHERITED_NS,
              prior_bytes=sum((ROOT/key).stat().st_size for key in inputs),models=common.pins(MODELS))
    atomic(path,body);return body

def epoch():return dict(epoch_id=SCHEMA,prerequisites_ready=True,historically_unseen=False,inherited_unknown_background=True)

class _WorkerView:
    def __init__(self):
        self.stack=ExitStack();self.pins=None;self.closed=False
    def admit(self,body,protocol):
        need(not self.closed,'Worker source view already closed')
        if self.pins is None:
            need(body['source_inputs']==protocol.source_inputs() and body['reservation_inputs']==protocol.reservation_inputs() and body['models']==common.pins(MODELS),'Required source/reservation/model bindings differ')
            self.pins={**body['source_inputs'],**body['reservation_inputs'],**body['models'],relative(PLAN):sha(PLAN)}
        else:
            need(body['source_inputs']==protocol.source_inputs() and body['models']==common.pins(MODELS),'Worker compact source/model bindings changed')
            common.current({k:v for k,v in self.pins.items() if not k.endswith(('quotient-components.sqlite3','closure.json','baseline.json'))})
    def close(self):
        if self.closed:return
        self.closed=True
        try:self.stack.close()
        finally:
            if self.pins is not None:common.current(self.pins)

@contextmanager
def worker_views():
    """Finite immutable source views; closing proofs precede worker completion."""
    need(_VIEW.get() is None,'Nested worker views are forbidden')
    view=_WorkerView();token=_VIEW.set(view)
    try:
        from . import guarded_value_pilot_v2_data as data
        view.stack.enter_context(data.admission_scope())
        from . import guarded_value_pilot_v2_execute as execute
        view.stack.enter_context(execute.admission_scope())
        yield view
    finally:
        try:view.close()
        finally:_VIEW.reset(token)

def close_views():
    view=_VIEW.get()
    if view is not None:view.close()

def admit_plan(require_review=True):
    from . import guarded_value_pilot_v2_plan as protocol
    body=read(PLAN);expected=protocol.definition()
    need(config.canonical({k:v for k,v in body.items() if k not in ('source_inputs','reservation_inputs','models')})==config.canonical(expected),'Prospective recipe changed')
    view=_VIEW.get()
    if view is None:
        need(body['source_inputs']==protocol.source_inputs() and body['reservation_inputs']==protocol.reservation_inputs() and body['models']==common.pins(MODELS),'Required source/reservation/model bindings differ')
    else:view.admit(body,protocol)
    need(body['policy_fixed'] is True and body['training_authorized'] is True and sha(PRIOR)==PRIOR_SHA,'Pilot admission scope differs')
    if require_review:
        audit=read(PREFLIGHT)
        need(audit['passed'] is True and audit['independent_of_implementation'] is True and audit['plan_sha256']==sha(PLAN) and audit['source_inputs']==body['source_inputs'] and audit['unresolved_findings']==[], 'Independent prospective preflight missing or stale')
    return body
