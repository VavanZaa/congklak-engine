"""Lightweight independent native-label recovery admission before gradients.

Full raw/native recovery is performed by the independent auditor. This gate
checks its immutable receipt and every bound small proof, without repeating
all128 teacher traces inside the optimizer worker.
"""
from pathlib import Path
from . import config,ordinary_epoch_common as s,ordinary_epoch_query_contract as contract

REPORT=s.FOLDER/'audit/native-label-recovery/result.json'
SCHEMA=s.SCHEMA+'-native-label-recovery'

def _ref(path):return dict(path=s.relative(path),sha256=s.sha(path))

def admit():
    s.admit_plan();a=s.read(REPORT)
    fields={'schema','passed','independent_of_implementation','plan_sha256','accepted_hooks','process_kills',
        'unknown_intent_zero_queries','unknown_intent_never_retried','subset_only_new_claims',
        'label_catalog_ref','label_ownership_prefix_ref','unknown_intent_marker_ref','evidence_inputs','unresolved_findings'}
    s.need(set(a)==fields and a['schema']==SCHEMA and a['passed'] is True and
        a['independent_of_implementation'] is True and a['plan_sha256']==s.sha(s.PLAN) and
        a['accepted_hooks']==list(contract.LABEL_KILL_HOOKS) and type(a['process_kills']) is int and a['process_kills']>=7 and
        a['unknown_intent_zero_queries'] is True and a['unknown_intent_never_retried'] is True and
        type(a['subset_only_new_claims']) is int and a['subset_only_new_claims']==0 and a['unresolved_findings']==[],
        'Independent native label recovery is incomplete or stale')
    required=s.common.pins([s.PLAN,s.FOLDER/'jobs/labels/catalog.json',s.FOLDER/'jobs/labels/ownership-prefix.json',s.FOLDER/'jobs/labels/report.json'])
    label_prefix=s.read(s.FOLDER/'jobs/labels/ownership-prefix.json');label_report=s.read(s.FOLDER/'jobs/labels/report.json')
    s.need(config.canonical(a['label_catalog_ref'])==config.canonical(_ref(s.FOLDER/'jobs/labels/catalog.json')) and
        config.canonical(a['label_ownership_prefix_ref'])==config.canonical(_ref(s.FOLDER/'jobs/labels/ownership-prefix.json')),
        'Recovery covers another label prefix')
    for hook in contract.LABEL_KILL_HOOKS:
        path=contract.FIXTURE_BASE/hook
        required.update(s.common.pins([path/'catalog.json',path/'identity.json',path/'report.json',path/'ownership-prefix.json']))
        report=s.read(path/'report.json');identity=s.read(path/'identity.json')
        s.need(report['state']=='complete' and report['fixture'] is True and report['ownership_passed'] is True and
            type(report['completed_units']) is int and report['completed_units']==2 and
            identity['scope']['subset_only'] is True and identity['scope']['hook']==hook,
            'A declared label-subset recovery fixture did not complete')
        prefix=s.read(path/'ownership-prefix.json')
        s.need(report['canonical_claims_sha256']==prefix['canonical_claims_sha256']==label_prefix['canonical_claims_sha256'] and
            type(report['global_claims']) is int and report['global_claims']==label_report['global_claims'],
            'A recovery fixture introduced claims outside its inherited label subset')
    unknown=contract.FIXTURE_BASE/contract.UNKNOWN_HOOK
    ref=a['unknown_intent_marker_ref'];s.need(set(ref)=={'path','sha256'},'Unknown-intent marker reference differs')
    marker_path=s.path_of(s.ROOT/ref['path']);s.need(config.canonical(ref)==config.canonical(_ref(marker_path)), 'Unknown-intent marker changed')
    marker=s.read(marker_path)
    s.need(marker['point']=='before_query' and marker['query_entry_reached'] is False and
        marker['source_sha256']==s.sha(s.ROOT/'congklak/ordinary_epoch_execute.py') and
        marker['identity_sha256']==s.sha(unknown/'identity.json'),'Zero-query HOLD is not bound to the actual pre-entry marker')
    from . import ordinary_epoch_execute as execute
    identity=s.read(unknown/'identity.json');events,_=execute._events(unknown,identity)
    s.need([e['kind'] for e in events]==['guard_restore_context','query_intent'] and
        not list((unknown/'archives').glob('*')) and not list((unknown/'attempts').glob('*')),
        'The unknown-intent fixture has native work or was retried')
    required.update(s.common.pins([marker_path,unknown/'identity.json',
        *list((unknown/'events').glob('*.json')),*list((unknown/'commits').glob('*.json'))]))
    inputs=a['evidence_inputs'];s.need(type(inputs) is dict and bool(inputs) and
        all(inputs.get(k)==v for k,v in required.items()), 'Recovery receipt omits required concrete proofs')
    for name in inputs:
        s.need(type(name) is str and (s.ROOT/name).resolve().is_relative_to(s.ROOT), 'Recovery proof outside workspace')
    s.common.current(inputs);inputs=dict(inputs);inputs[s.relative(REPORT)]=s.sha(REPORT)
    return dict(schema=s.SCHEMA+'-recovery-gate-admission',passed=True,inputs=inputs,
        actual_kills=a['process_kills'],accepted_label_boundaries=6,zero_query_hold_verified=True)
