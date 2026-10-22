"""Source-bound ordinary-only development experiment; no historical release claim."""
import json
from pathlib import Path
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/ordinary-epoch-pilot-v1'
DATA=ROOT/'data/ordinary-epoch-pilot-v1'
PRIOR=ROOT/'build/ownership-resolution-v1/acceptance.json'
PRIOR_SHA='7d89313c5326f8fe7e4cd153e6cf71e4bda37f147e03d3d455069f8aa3c76dd6'
JOINT=ROOT/'build/ownership-resolution-v1/production/units/000695.json'
EPOCH=ROOT/'build/ownership-resolution-v1/production/units/000696.json'
PLAN=FOLDER/'plan-v3.json'
PREFLIGHT=FOLDER/'audit/preflight-v3.json'
SCHEMA='ordinary-epoch-pilot-v1'
need,read,sha=common.require,common.read,common.sha
MODELS=('data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
        'data/training-stage5-deeper-h8-f127-seed2718/model.json')

def normalized(body):return json.loads(config.canonical(body))
def atomic(path,body):return _atomic(Path(path),normalized(body))
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def path_of(path):
    p=Path(path).resolve()
    need(any(p==b or p.is_relative_to(b) for b in (FOLDER.resolve(),DATA.resolve())), 'Ordinary pilot output namespace differs')
    return p
def baseline():
    need(sha(PRIOR)==PRIOR_SHA,'Accepted new reservations changed')
    p=read(PRIOR);need(p['passed'] is True and p['epoch']['prerequisites_ready'] is True,'Reservations not accepted')
    inputs=dict(p['inputs']);inputs[relative(PRIOR)]=PRIOR_SHA
    body=dict(schema=SCHEMA+'-baseline',previous_acceptance=relative(PRIOR),previous_acceptance_sha256=PRIOR_SHA,
              inputs=inputs,inputs_count=len(inputs),models=common.pins(MODELS))
    out=FOLDER/'baseline.json'
    if out.exists():need(config.canonical(read(out))==config.canonical(body),'Baseline cannot change')
    else:atomic(out,body)
    return body
def joint():return read(JOINT)['result']
def epoch():return read(EPOCH)['result']
def admit_plan(require_review=True):
    p=read(PLAN)
    need(p['schema']==SCHEMA+'-plan' and p['policy_fixed'] is True and p['training_authorized'] is True,
         'New pilot recipe not frozen')
    need(p['previous_acceptance_sha256']==PRIOR_SHA and sha(PRIOR)==PRIOR_SHA,'Reservation acceptance differs')
    from . import ordinary_epoch_plan
    expected=ordinary_epoch_plan.definition()
    need(config.canonical({k:v for k,v in p.items() if k not in ('source_inputs','reservation_inputs','models')})==config.canonical(expected),
         'Frozen recipe differs from the one authorized treatment')
    need(config.canonical({k:p[k] for k in ('source_inputs','reservation_inputs','models')})==
         config.canonical(ordinary_epoch_plan.bindings()),'Required full source/reservation/model closure differs')
    if require_review:
        a=read(PREFLIGHT)
        need(set(a)=={'schema','passed','independent_of_implementation','plan_sha256','source_inputs',
                    'checks','unresolved_findings','activity'} and a['schema']==SCHEMA+'-preflight' and
             a['passed'] is True and a['independent_of_implementation'] is True and
             a['plan_sha256']==sha(PLAN) and a['unresolved_findings']==[] and
             config.canonical(a['source_inputs'])==config.canonical(p['source_inputs']) and
             set(a['checks'])=={'recipe','reservations','resources','native','training','recovery_protocol'} and
             all(x is True for x in a['checks'].values()) and
             config.canonical(a['activity'])==config.canonical(dict(optimizer_steps=0,native_searches=0,model_forwards=0)),
             'Independent pre-run audit missing or stale')
    return p
