"""Derive and seal screened reservations without authorizing optimization."""
from collections import Counter
from . import config,footprint_screen_common as s

METHODS=('source_previous_preservation','population_no_quality_selection','complete_physical_traces',
 'whole_component_closure','joint_quotas_support','interruption_recovery','completed_readers_resources','observer_claims')
def _ref(path):return dict(path=s.relative(path),sha256=s.sha(path))
def _same(a,b):return config.canonical(a)==config.canonical(b)
def _recovery(name,schema,count,hooks):
    path=s.FOLDER/'audit'/name/'result.json';body=s.read(path)
    s.need(body['schema']==s.SCHEMA+schema and body['passed'] is True and
        body['independent_of_implementation'] is True and type(body['process_interruptions']) is int and
        body['process_interruptions']==count,'Actual independent recovery evidence incomplete')
    s.common.current(body['source_inputs'])
    cases=body['cases'].values() if type(body['cases']) is dict else body['cases']
    cases=list(cases)
    s.need(len(cases)==len(hooks) and {row['hook'] for row in cases}==set(hooks) and
        all(row['passed'] is True and row['actual_process_interruption'] is True for row in cases),
        'Actual recovery boundaries differ from frozen protocol')
    for row in cases:
        for ref in row.get('evidence_refs',[]):s.need(s.sha(s.ROOT/ref['path'])==ref['sha256'],'Recovery evidence changed')
    return body
def derive_report():
    from . import footprint_screen_worker as worker,footprint_screen_review as review
    from . import footprint_screen_reservations as reservations,footprint_screen_resources as resources
    from . import footprint_screen_data as data,footprint_screen_plan as protocol
    plan=s.admit_plan();acquisition=worker.completed();snapshot=worker.completed_snapshot()
    raw=review.completed();selected=reservations.completed()
    s.need(not acquisition['publication_gap'] and not raw['publication_gap'] and not selected['publication_gap'],
           'Publication gaps must close before reservation report')
    a=acquisition['report'];r=raw['report'];v=selected['report']
    s.need(a['state']=='complete' and a['fixture'] is False and r['fixture'] is False and v['fixture'] is False,
           'Partial, held or fixture data cannot become primary reservations')
    native=_recovery('native-recovery','-independent-native-recovery',7,protocol.NATIVE_HOOKS)
    rr=_recovery('raw-review-recovery','-independent-review-recovery',2,('before_unit_commit','after_unit_commit'))
    sr=_recovery('reservation-recovery','-independent-reservation-recovery',2,('before_catalog','after_catalog'))
    roles=v['roles'];fit=roles['fit'];arena=roles['arena'];cases=roles['cases']
    fit_counts=Counter(row['cell'] for row in fit);case_counts=Counter(row['cell'] for row in cases)
    arena_counts=Counter(row['cell'] for row in arena);ordinary=data.CELLS[:4]
    models=s.common.pins(s.MODELS);s.need(_same(models,plan['models']),'An original fixed model changed')
    support=v['support192_preserved'];support_counts=Counter(row['cell'] for row in support)
    metrics=dict(acquired_roots=a['settled_roots'],captured_queries=a['completed_queries'],
        complete_teachers=a['complete_teachers'],incomplete_teachers=a['incomplete_teachers'],
        raw_query_sources=r['query_count'],raw_events=r['raw_events'],high_owned_events=r['high_owned_events'],
        low_shared_events=r['low_shared_events'],upper_owned_events=r['upper_owned_events'],
        fit_roots=len(fit),case_roots=len(cases),arena_origins=len(arena),fit_counts=dict(fit_counts),
        case_counts=dict(case_counts),arena_counts=dict(arena_counts),
        distinct_selected_components=len({row['component'] for row in fit+arena}),
        support_roots=len(support),support_cells=dict(support_counts),
        rejected_candidates=sum(not row['eligible'] for row in v['root_dispositions']),
        candidate_dispositions=len(v['root_dispositions']),shortage=v['shortage'],
        native_process_interruptions=native['process_interruptions'],
        review_process_interruptions=rr['process_interruptions'],reservation_process_interruptions=sr['process_interruptions'])
    gates=dict(all512_acquired=a['settled_roots']==512 and a['completed_queries']==1024,
        all_actual_query_sources_replayed=r['passed'] is True and r['all_actual_query_sources_checked'] is True and
            r['production_query_count']==1024 and r['snapshot_sha256']==config.digest(snapshot),
        all_high_footprints_retained=v['all_acquired_footprints_retained'] is True and v['rejected_old_high_quotients_retained']==6669,
        disk_closure_complete=v['disk_backed_full_incidence'] is True and v['owner_union_find_only'] is True,
        metadata_only_selection=v['quality_values_used_for_selection'] is False and v['selection_after_complete_batch_only'] is True,
        joint_quotas=v['metadata_quotas_satisfied'] is True and v['reservation_admission'] is True and
            len(fit)==128 and len(cases)==32 and len(arena)==24 and metrics['distinct_selected_components']==152 and
            fit_counts==Counter({cell:16 for cell in data.CELLS}) and case_counts==Counter({cell:4 for cell in data.CELLS}) and
            arena_counts==Counter({cell:6 for cell in ordinary}),
        ordinary_support_preserved=len(support)==192 and
            support_counts==Counter(dict(zip(ordinary,(39,34,58,61)))),
        old_quarantine_preserved=v['old320_quarantine_preserved'] is True,
        native_recovery=native['subset_only_new_quotients']==0 and native['zero_query_case']['actual_query_count']==0 and
            native['zero_query_case']['no_retry'] is True,
        raw_review_recovery=rr['process_interruptions']==2,
        reservation_recovery=sr['process_interruptions']==2,
        models_policy_fixed=v['models_fixed'] is True and v['policy_fixed'] is True and r['model_forwards']==0 and r['optimizer_steps']==0)
    ledger=resources.status();s.need(ledger['pending'] is None and not(resources.DEFAULT/'hold.json').exists(),
        'Resource accounting unsettled')
    live=s.common.pins([s.PLAN,s.PREFLIGHT,s.FOLDER/'pool.json',s.FOLDER/'acquisition/identity.json',
        s.FOLDER/'acquisition/catalog.json',s.FOLDER/'acquisition/report.json',
        s.FOLDER/'audit/raw-review/identity.json',s.FOLDER/'audit/raw-review/catalog.json',s.FOLDER/'audit/raw-review/report.json',
        s.FOLDER/'reservations/identity.json',s.FOLDER/'reservations/catalog.json',s.FOLDER/'reservations/certificate.json',
        s.FOLDER/'reservations/report.json',resources.DEFAULT/'identity.json',resources.DEFAULT/'head.json',
        *[s.FOLDER/'audit'/name/'result.json' for name in ('native-recovery','raw-review-recovery','reservation-recovery')]])
    live.update(plan['source_inputs']);live.update(models)
    archival=dict(snapshot['inputs']);mapping=v['quotient_component_map'];archival[mapping['path']]=mapping['sha256']
    return dict(schema=s.SCHEMA+'-report',engineering_execution_complete=True,
        reservation_admission=all(gates.values()),gates=gates,metrics=metrics,plan_sha256=s.sha(s.PLAN),
        models_fixed=True,policy_fixed=True,training_authorized=False,model_forwards=0,optimizer_steps=0,
        promotion=False,default_engine='H_v1',old_pilot_stays_held=True,
        inherited_unknown_background=True,historically_unseen=False,natural_game_representativeness_claim=False,
        resource_charged_ns=ledger['charged_ns'],resource_remaining_ns=ledger['remaining_ns'],
        inherited_charged_ns=ledger['inherited_charged_ns'],live_inputs=live,
        archival_inputs=archival,
        decision=dict(ready_for_future_training_recipe=all(gates.values()),training_authorized=False,
            automatic_training=False,automatic_pool_expansion=False,old_candidate_rehabilitated=False,
            runtime_guards_still_required=True,conditional_custom_development_set=True))
def report():
    body=derive_report();s.atomic(s.FOLDER/'report.json',body);return body

def audit_report(body):
    plan=s.admit_plan();path=s.FOLDER/'audit/audit.json';audit=s.read(path)
    s.need(audit['schema']==s.SCHEMA+'-independent-audit' and audit['passed'] is True and
        audit['independent_of_implementation'] is True and audit['plan_sha256']==s.sha(s.PLAN) and
        _same(audit['source_inputs'],plan['source_inputs']) and audit['unresolved_findings']==[] and
        audit['models_fixed'] is True and audit['policy_fixed'] is True and
        audit['training_authorized'] is False and type(audit['model_forwards']) is int and audit['model_forwards']==0 and
        type(audit['optimizer_steps']) is int and audit['optimizer_steps']==0 and
        audit['historically_unseen'] is False and audit['inherited_unknown_background'] is True,
        'Independent final audit is missing, stale or outside the declared scope')
    methods=audit['methods'];s.need(set(methods)==set(METHODS),'Independent audit methods differ')
    for name in METHODS:
        method=methods[name]
        s.need(method['passed'] is True and type(method['evidence']) is list and method['evidence'],
            'Independent audit method lacks concrete evidence: '+name)
        for ref in method['evidence']:
            s.need(set(ref)=={'path','sha256'} and s.sha(s.ROOT/ref['path'])==ref['sha256'],
                'Independent audit evidence changed: '+name)
    s.need(audit['report_ref']==_ref(s.FOLDER/'report.json'),'Audit does not bind the final report')
    s.need(audit['reservation_admission'] is body['reservation_admission'],
        'Audit reservation decision differs from complete-source report')
    return audit

def _files():
    excluded={s.FOLDER/'closure.json',s.FOLDER/'acceptance.json'}
    return sorted(p for base in (s.FOLDER,s.DATA) for p in base.rglob('*') if p.is_file() and p not in excluded)

def _latest_event():
    events=sorted((s.FOLDER/'events').glob('*.json'))
    s.need(bool(events),'Final dashboard event missing');return events[-1]

def seal():
    """Preserve inherited EXPECTED hashes, then check every old and new byte."""
    body=s.read(s.FOLDER/'report.json')
    s.need(_same(body,derive_report()),'Final report is stale')
    audit_report(body)
    baseline=s.baseline();inputs=dict(baseline['inputs'])
    fresh=s.common.pins(_files())
    for key,value in fresh.items():
        s.need(key not in inputs or inputs[key]==value,'Cannot rebase an inherited expected hash: '+key)
        inputs[key]=value
    # Source files outside the isolated artifact namespace remain covered too.
    for key,value in {**body['live_inputs'],**body['archival_inputs']}.items():
        s.need(key not in inputs or inputs[key]==value,'Source hash conflict: '+key);inputs[key]=value
    s.common.current(inputs)
    closure=dict(schema=s.SCHEMA+'-closure',passed=True,inputs=inputs,inputs_count=len(inputs),
        inherited_expected_hashes_retained=len(baseline['inputs']),
        report_ref=_ref(s.FOLDER/'report.json'),audit_ref=_ref(s.FOLDER/'audit/audit.json'),
        reservation_admission=body['reservation_admission'],training_authorized=False,
        models_fixed=True,policy_fixed=True,optimizer_steps=0,model_forwards=0)
    s.atomic(s.FOLDER/'closure.json',closure)
    compact=dict(body['live_inputs']);compact.update(s.common.pins([
        s.FOLDER/'report.json',s.FOLDER/'audit/audit.json',_latest_event()]))
    # The running observer's exact creation/argv/listener proof is audited and sealed.
    marker=dict(schema=s.SCHEMA+'-acceptance',passed=True,engineering_execution_complete=True,
        reservation_admission=body['reservation_admission'],training_authorized=False,
        report_ref=_ref(s.FOLDER/'report.json'),audit_ref=_ref(s.FOLDER/'audit/audit.json'),
        closure_ref=_ref(s.FOLDER/'closure.json'),compact_inputs=compact,
        latest_event_ref=_ref(_latest_event()),decision=body['decision'],metrics=body['metrics'],
        models_fixed=True,policy_fixed=True,model_forwards=0,optimizer_steps=0,
        inherited_unknown_background=True,historically_unseen=False)
    s.atomic(s.FOLDER/'acceptance.json',marker);return marker

def valid_compact(marker):
    s.need(marker['schema']==s.SCHEMA+'-acceptance' and marker['passed'] is True and
        marker['training_authorized'] is False and marker['models_fixed'] is True and marker['policy_fixed'] is True,
        'Acceptance scope differs')
    body=s.read(s.FOLDER/'report.json');audit=s.read(s.FOLDER/'audit/audit.json')
    expected=dict(body['live_inputs']);expected.update(s.common.pins([
        s.FOLDER/'report.json',s.FOLDER/'audit/audit.json',_latest_event()]))
    s.need(_same(marker['compact_inputs'],expected),'Compact receipt omits required source bindings')
    s.common.current(marker['compact_inputs'])
    s.need(marker['latest_event_ref']==_ref(_latest_event()),'Dashboard event changed after acceptance')
    # Full raw/old closure bytes are deliberately verified by verify(), not each poll.
    for key in ('report_ref','audit_ref'):
        ref=marker[key];s.need(s.sha(s.ROOT/ref['path'])==ref['sha256'],'Acceptance receipt changed')
    s.need(marker['report_ref']==_ref(s.FOLDER/'report.json') and marker['audit_ref']==_ref(s.FOLDER/'audit/audit.json') and
        marker['closure_ref']['path']==s.relative(s.FOLDER/'closure.json') and
        marker['reservation_admission'] is body['reservation_admission'] and
        marker['reservation_admission'] is audit['reservation_admission'] and audit['passed'] is True and
        marker['engineering_execution_complete'] is body['engineering_execution_complete'] and
        _same(marker['metrics'],body['metrics']) and _same(marker['decision'],body['decision']) and
        type(marker['model_forwards']) is int and marker['model_forwards']==0 and
        type(marker['optimizer_steps']) is int and marker['optimizer_steps']==0 and
        marker['historically_unseen'] is False and marker['inherited_unknown_background'] is True,
        'Live acceptance claims differ from sealed source reports')
    return True

def verify():
    marker=s.read(s.FOLDER/'acceptance.json');valid_compact(marker)
    ref=marker['closure_ref'];path=s.ROOT/ref['path']
    s.need(s.sha(path)==ref['sha256'],'Full closure marker changed')
    closure=s.read(path);baseline=s.baseline()
    s.need(closure['schema']==s.SCHEMA+'-closure' and closure['passed'] is True and
        closure['inputs_count']==len(closure['inputs']) and
        closure['inherited_expected_hashes_retained']==len(baseline['inputs']) and
        all(closure['inputs'].get(k)==v for k,v in baseline['inputs'].items()),
        'Inherited expected preservation closure differs')
    s.need(all(closure['inputs'].get(s.relative(p))==s.sha(p) for p in _files()),
        'Unsealed or changed new artifact')
    s.common.current(closure['inputs'])
    s.need(_same(s.read(s.FOLDER/'report.json'),derive_report()),'Full report no longer follows actual sources')
    audit_report(s.read(s.FOLDER/'report.json'))
    return dict(passed=True,inputs_verified=len(closure['inputs']),
        inherited_expected_hashes_retained=len(baseline['inputs']),
        reservation_admission=marker['reservation_admission'],training_authorized=False,
        model_forwards=0,optimizer_steps=0)
