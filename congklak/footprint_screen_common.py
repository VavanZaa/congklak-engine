"""Isolated, source-bound acquisition; model training is never admitted here."""
import json
from pathlib import Path
from types import MappingProxyType
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/footprint-screen-v1'
DATA=ROOT/'data/footprint-screen-v1'
PLAN=FOLDER/'plan.json'
PREFLIGHT=FOLDER/'audit/preflight.json'
PRIOR=ROOT/'build/ordinary-epoch-pilot-v1/held-closure-v1.json'
PRIOR_SHA='f332496dab1c62b08d9661ba64018df4756d8a0ce210ed38d307b3a0e5f37d00'
SCHEMA='footprint-screen-v1'
MODELS=('data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
        'data/training-stage5-deeper-h8-f127-seed2718/model.json')
INHERITED_CHARGED_NS=737420000000
TOTAL_WORKER_NS=10800000000000
need,read,sha=common.require,common.read,common.sha
_PRIOR_INPUTS=None

def normalized(body):return json.loads(config.canonical(body))
def atomic(path,body):return _atomic(Path(path),normalized(body))
def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def path_of(path):
    p=Path(path).resolve()
    need(any(p==b or p.is_relative_to(b) for b in (FOLDER.resolve(),DATA.resolve())),
         'Footprint-screen output outside isolated namespace')
    return p
def prior_inputs():
    global _PRIOR_INPUTS
    need(sha(PRIOR)==PRIOR_SHA,'Held-run preservation closure changed')
    if _PRIOR_INPUTS is None:
        old=read(PRIOR)
        need(old['passed'] is True and old['pilot_completed'] is False and old['production_steps']==0,
             'Previous held-run scope differs')
        _PRIOR_INPUTS=MappingProxyType(dict(old['inputs']))
    return _PRIOR_INPUTS
def baseline():
    need(sha(PRIOR)==PRIOR_SHA,'Held-run preservation closure changed')
    old=read(PRIOR)
    need(old['passed'] is True and old['pilot_completed'] is False and
         old['production_steps']==0 and old['original_expected_hashes_retained']==80205,
         'Previous held-run scope differs')
    inputs=dict(old['inputs']);inputs[relative(PRIOR)]=PRIOR_SHA
    body=dict(schema=SCHEMA+'-baseline',prior=relative(PRIOR),prior_sha256=PRIOR_SHA,
        inputs_count=len(inputs),inputs=inputs,inherited_charged_ns=INHERITED_CHARGED_NS,
        models=common.pins(MODELS),old_pilot_stays_held=True)
    output=FOLDER/'baseline.json'
    if output.exists():need(config.canonical(read(output))==config.canonical(body),'Baseline cannot change')
    else:atomic(output,body)
    return body
def parent_budget():
    """Read the sealed parent journal; never initialize or mutate it."""
    from . import ordinary_epoch_resources as previous
    need(sha(PRIOR)==PRIOR_SHA,'Parent closure changed')
    old=prior_inputs();folder=ROOT/'data/ordinary-epoch-pilot-v1/resources'
    for p in (folder/'identity.json',folder/'head.json'):
        need(old.get(relative(p))==sha(p),'Sealed parent resource journal changed')
    status=previous.status()
    need(status['pending'] is None and status['charged_ns']==INHERITED_CHARGED_NS and
         status['remaining_ns']==TOTAL_WORKER_NS-INHERITED_CHARGED_NS,
         'Parent budget is unsettled or changed')
    return dict(charged_ns=INHERITED_CHARGED_NS,total_ns=TOTAL_WORKER_NS,
        parent_identity=dict(path=relative(folder/'identity.json'),sha256=sha(folder/'identity.json')),
        parent_head=dict(path=relative(folder/'head.json'),sha256=sha(folder/'head.json')))
def admit_plan(require_review=True):
    from . import footprint_screen_plan as protocol
    plan=read(PLAN)
    need(config.canonical(plan)==config.canonical(protocol.expected()),'Frozen acquisition recipe/source closure differs')
    need(plan['training_authorized'] is False and plan['models_fixed'] is True and plan['policy_fixed'] is True,
         'Acquisition cannot authorize model updates')
    if require_review:
        audit=read(PREFLIGHT)
        need(audit['schema']==SCHEMA+'-preflight' and audit['passed'] is True and
             audit['independent_of_implementation'] is True and audit['plan_sha256']==sha(PLAN) and
             config.canonical(audit['source_inputs'])==config.canonical(plan['source_inputs']) and
             audit['unresolved_findings']==[] and
             config.canonical(audit['activity'])==config.canonical(dict(new_queries=0,model_forwards=0,optimizer_steps=0)),
             'Independent acquisition preflight missing or stale')
    return plan
