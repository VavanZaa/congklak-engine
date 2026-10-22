"""Derive the one fixed pilot decision from strict completed artifact readers."""
from . import config,guarded_value_pilot_v2_common as s,guarded_value_pilot_v2_results as results

def derive_report():
    s.admit_plan()
    from . import guarded_value_pilot_v2_train as training,guarded_value_pilot_v2_execute as execute,guarded_value_pilot_v2_fit as fit
    from . import guarded_value_pilot_v2_resources as resources
    for operation in ('conformance','guards','arena'):
        closed=execute.completed_job(s.FOLDER/'jobs'/operation,operation)
        s.need(closed['report']['state']=='complete' and closed['report']['fixture'] is False and
               not closed['publication_gap'],'A partial or held primary campaign cannot claim completed execution')
    t=training.completed();f=fit.completed();m=execute.summary_metrics()
    s.need(set(m)=={'labels','cases','conformance','endgame','arena'},'Executor measured groups differ')
    m['training']=dict(steps=t['steps'],draws=t['draws'],policy_fixed=t['policy_fixed'],
        base_frozen=t['frozen_tensors']==8 and t['frozen_parameters']==12872)
    fm=f['metrics'];m['fit']=dict(roots=fm['overall']['control']['count'],ordinary_roots=fm['ordinary']['control']['count'],
        capture_roots=fm['capture']['control']['count'],bonus_roots=fm['bonus']['control']['count'],
        control_huber=fm['overall']['control']['normalized_huber'],candidate_huber=fm['overall']['candidate']['normalized_huber'],
        control_mae=fm['overall']['control']['seed_mae'],candidate_mae=fm['overall']['candidate']['seed_mae'],
        control_capture_mae=fm['capture']['control']['seed_mae'],candidate_capture_mae=fm['capture']['candidate']['seed_mae'],
        ownership_passed=execute.admit_labels()['ownership_passed'] is True)
    rows=execute.arena_game_rows();gates=results.derive_gates(m,None)
    eligible=('fit_coverage','case_coverage','native_conformance','endgame_guard','identical_low_fixed_work',
        'arena_coverage','arena_no_caps_or_fallbacks','whole_component_ownership','equal_time_25ms')
    stats=results.paired_statistics(rows) if all(gates[k] for k in eligible) else None
    gates=results.derive_gates(m,stats)
    ledger=resources.status();s.need(ledger['pending'] is None and not(resources.DEFAULT/'hold.json').exists(),
        'Unsettled resource accounting cannot close')
    archival=dict(execute.result_inputs())
    compact_names={'identity.json','catalog.json','report.json','ownership-prefix.json','fit-manifest.json'}
    live={k:v for k,v in archival.items() if s.ROOT.joinpath(k).name in compact_names and
          '/archives/' not in k and '/events/' not in k and '/commits/' not in k}
    live.update(f['input_pins'])
    live.update(s.common.pins([fit.DEFAULT/'report.json',fit.DEFAULT/'certificate.json',
        *[training.DEFAULT/n for n in ('identity.json','catalog.json','certificate.json','result.json','model.json')],
        resources.DEFAULT/'identity.json',resources.DEFAULT/'head.json']))
    archival.update(s.common.pins(p for p in training.DEFAULT.rglob('*') if p.is_file() and p.name!='writer.lock' and not p.name.endswith('.pending')))
    decision=dict(all_scientific_gates_passed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v],
        promotion=False,default_engine='H_v1',fixed_final_step=400,checkpoint_selection=False,
        additional_training=False,expanded_training_data=False,
        strength_interpretation_allowed=stats is not None,
        candidate_retained='development artifact; no release promotion')
    return dict(schema=s.SCHEMA+'-report',engineering_execution_complete=True,policy_fixed=True,base_frozen=True,
        plan_sha256=s.sha(s.PLAN),metrics=m,scientific_gates=gates,statistics=stats,game_rows=rows,decision=decision,
        inherited_unknown_background=True,historically_unseen=False,historical_provenance_complete=False,
        old_candidate_rehabilitated=False,resource_charged_ns=ledger['charged_ns'],
        resource_remaining_ns=ledger['remaining_ns'],resource_leases=sum(e['kind']=='reserve' for e in ledger['events']),
        live_inputs=live,archival_inputs=archival,
        observer_integrity_scope='Full byte verification at audit/seal/explicit verify; compact source and receipt checks during live polling')

def report():
    body=derive_report();path=s.FOLDER/'report.json'
    if path.exists():s.need(config.canonical(s.read(path))==config.canonical(body),'Completed pilot report changed')
    else:s.atomic(path,body)
    return body

def seal():
    """Call only after independent actual-work audit. Rehash the combined closure once."""
    body=report();audit=s.read(s.FOLDER/'audit/audit.json');results.audit_valid(audit)
    baseline=s.read(s.FOLDER/'baseline.json');inputs=dict(baseline['inputs'])
    for key,value in body['archival_inputs'].items():
        s.need(key not in inputs or inputs[key]==value,'Archive attempted to overwrite an old expected digest')
        inputs[key]=value
    # Every old expected hash survives unchanged, including any accidentally
    # reused relative path: a new current digest can never overwrite it.
    for folder in (s.FOLDER,s.DATA):
        for path in folder.rglob('*'):
            if not path.is_file() or path.name in ('acceptance.json','writer.lock') or path.suffix=='.pending':continue
            key=s.relative(path);value=s.sha(path)
            s.need(key not in inputs or inputs[key]==value,'Attempt to overwrite an old expected digest')
            inputs[key]=value
    required=s.common.pins([s.PLAN,s.FOLDER/'baseline.json',s.FOLDER/'report.json',s.FOLDER/'audit/audit.json',
        s.PREFLIGHT,s.ROOT/'congklak/guarded_value_pilot_v2_results.py',
        s.ROOT/'congklak/guarded_value_pilot_v2_progress.py',s.ROOT/'tools/progress-guarded-value-pilot-v2.py'])
    event=s.FOLDER/'status.json';s.need(event.is_file(),'Final progress status is missing')
    required.update(s.common.pins([event]))
    required.update(s.read(s.PLAN)['source_inputs']);required.update(body['live_inputs']);required.update(audit['source_inputs'])
    for method in audit['methods'].values():required.update({r['path']:r['sha256'] for r in method['evidence']})
    for key,value in required.items():
        s.need(key not in inputs or inputs[key]==value,'Live closure attempted to overwrite an expected digest')
        inputs[key]=value
    acceptance=dict(schema=s.SCHEMA+'-acceptance',passed=True,policy_fixed=True,base_frozen=True,
        plan_sha256=s.sha(s.PLAN),report_sha256=s.sha(s.FOLDER/'report.json'),audit_sha256=s.sha(s.FOLDER/'audit/audit.json'),
        previous_files_preserved=baseline['inputs_count'],inputs=inputs,live_inputs=required,
        progress_event=s.relative(event),progress_event_sha256=s.sha(event),
        decision=body['decision'],promotion=False,default_engine='H_v1',historically_unseen=False,inherited_unknown_background=True)
    s.need(results.valid_closure(acceptance),'Final live closure refused before full seal')
    s.common.current(inputs)
    path=s.FOLDER/'acceptance.json'
    if path.exists():s.need(config.canonical(s.read(path))==config.canonical(acceptance),'Immutable acceptance differs')
    else:s.atomic(path,acceptance)
    return acceptance
