"""Recoverable, predeclared four-profile equal-clock engineering panel."""
from contextlib import ExitStack
import base64
import hashlib
import math
from pathlib import Path
import time
from . import config,tracing_speed_common as stage,tracing_speed_contract as contract
from . import ownership_fast_guard as guards,value_run_arena as literal,tracing_speed_resources as resources
from .timing_v2 import PerformanceCore
from .valuepilot_reservations import Writer

ROOT,FOLDER=stage.ROOT,stage.FOLDER
SOURCES=('congklak/tracing_speed_timing.py','congklak/tracing_speed_contract.py',
    'congklak/tracing_speed_common.py','congklak/tracing_speed_launcher_v4.py','congklak/tracing_speed_resources.py',
    'build/tracing-speed-v1/launcher-v4/entry.py','congklak/value_run_arena.py','tests/reference.py','congklak/timing_v2.py')
ORIGIN='timing-consumed-panel'
PANEL_SHA='31d52d496ba35f550d1691fa12bf6b0e8f315de2c87af315c6e7b736a49b2574'
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic

def _identity(spec_path):
    from . import native_trace_fast_v2 as fast,tracing_speed_launcher_v4 as launcher
    spec_path=stage.path_of(spec_path);spec=read(spec_path);launcher.validate_spec(spec_path)
    need(spec['kind']=='timing' and spec['schema']=='trace-speed-timing-protocol-v1' and
         spec['clock_contract_sha256']==config.digest(contract.definition()) and
         spec['panel_path']=='build/tracing-speed-v1/timing/panel.json' and
         spec['pressure_path']=='build/tracing-speed-v1/timing/pressure.json','Timing protocol differs')
    panel=read(ROOT/spec['panel_path']);need(sha(ROOT/spec['panel_path'])==spec['panel_sha256'] and
         spec['panel_sha256']==PANEL_SHA and len(panel['schedule'])==240 and
         panel['clock_contract_sha256']==spec['clock_contract_sha256'],'Timing panel differs')
    calibration=stage.FOLDER/'native/calibration-v4/report.json';cal=read(calibration)
    need(spec['inputs'].get(calibration.relative_to(ROOT).as_posix())==sha(calibration) and
        cal['policy_frozen'] is True and cal['search_policy']==spec['search_policy'] and
        spec['search_policy']['calibrated'] is True and spec['capacity_events']==2000000 and
        spec['guard']==read(ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json')['guard'],
        'Timing exact calibration, capacity or consumed guard differs')
    stage.common.current(spec['inputs']);loaded=guards.load_pure(spec['guard_refs'],spec)
    need(fast.source_identity()==spec['native_identity'],'Timing frozen native provider identity differs')
    identity=dict(schema='trace-speed-timing-identity-v1',spec_path=spec_path.relative_to(ROOT).as_posix(),
        spec_sha256=sha(spec_path),guard_refs=spec['guard_refs'],sources=stage.common.pins(SOURCES),native=fast.source_identity(),
        cpu_declared=read(ROOT/'build/value-plan.json')['cpu_declared'],training_counters=stage.ZERO,
        models_fixed=True,policy_fixed=True)
    return stage.normalized(identity),spec,panel,loaded

def prepare(output,spec_path):
    output=stage.path_of(output,'timing');identity,_,_,_=_identity(spec_path)
    atomic(output/'identity.json',identity);return dict(prepared=True,identity_sha256=sha(output/'identity.json'))

def _commit(output,event):
    index=len(list((output/'commits').glob('*.json')));event=stage.normalized(event)
    digest=hashlib.sha256((config.canonical(event)+'\n').encode()).hexdigest()
    rel=f'events/{index:06d}-{digest}.json';atomic(output/rel,event)
    path=output/'commits'/f'{index:06d}.json'
    atomic(path,dict(index=index,identity_sha256=sha(output/'identity.json'),event=rel,event_sha256=digest,
        previous_sha256=sha(output/'commits'/f'{index-1:06d}.json') if index else None))
    return dict(index=index,event_path=str(output/rel),event_sha256=digest,commit_path=str(path),commit_sha256=sha(path))

def _pressure(spec,loaded):
    pressure=read(ROOT/spec['pressure_path']);need(sha(ROOT/spec['pressure_path'])==spec['pressure_sha256'] and
        len(pressure['keys'])==4096 and pressure['keys']==sorted(set(pressure['keys'])) and
        pressure['origin']==ORIGIN,'Pressure fixture differs')
    source=ROOT/'build/ownership-implementation-v1/provenance/engineering-supplement-v1/report.json'
    need(pressure['source_path']==source.relative_to(ROOT).as_posix() and pressure['source_sha256']==sha(source) and
        pressure['keys']==sorted(read(source)['combined_known_high_quotient_components'])[:4096],
        'Pressure fixture source/selection differs')
    claimed={q:ORIGIN for q in pressure['keys']};need(len(claimed)==4096,'Pressure claims collapsed')
    loaded.claims_blob(claimed);return claimed

def _check_row(output,row,scheduled,panel,loaded,claimed,identity):
    from . import native_trace_fast_v2 as fast
    need(all(row[k]==scheduled[k] and type(row[k]) is type(scheduled[k]) for k in ('index','root','repetition','engine')) and
         row['archive_path']==f'attempts/{scheduled["index"]:06d}.json','Timing schedule order/path differs')
    packet=output/row['archive_path'];need(sha(packet)==row['archive_sha256'],'Attempt archive differs')
    saved=read(packet);answer=saved['answer'];root=panel['panel'][row['root']]['state']
    intent=read(output/'intents'/f'{row["index"]:06d}.json')
    need(saved['decision_id']==f'timing/{scheduled["index"]:06d}','Timing exact decision cursor differs')
    need(saved['scheduled']==scheduled and saved['identity_sha256']==sha(output/'identity.json') and
         saved['root']==root and saved['row']=={k:v for k,v in row.items() if k!='archive_sha256'},'Timing attempt cursor differs')
    need(intent['scheduled']==scheduled and intent['root']==root and intent['decision_id']==saved['decision_id'] and
         intent['identity_sha256']==sha(output/'identity.json') and
         intent['claims_sha256']==hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),'Timing durable intent differs')
    cert=answer['fast_certificate'];policy=read(ROOT/identity['spec_path'])['search_policy']
    clock=answer['fast_timing'];flags=dict(
        coverage_passed=bool(cert.get('valid') and cert.get('attempted_events')==cert.get('event_count')),
        guard_passed=bool(cert.get('guard_passed')))
    need(all(type(row[k]) is bool and row[k]==v for k,v in flags.items()) and
        row['event_count']==cert.get('event_count',0) and row['owned_count']==cert.get('owned_count',0) and
        row['archival_ms']==answer['archival_timing']['elapsed_ms'] and answer['archival_timing']['charged_to_lease'] is True and
        row['native_search_ms']==answer.get('decision',{}).get('requested_ms') and clock['policy']==policy,
        'Timing row/native coverage, cost, count or policy differs')
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in
        (row['critical_ms'],row['archival_ms'],clock['outer_elapsed_ms'])) and
        row['critical_ms']>=clock['outer_elapsed_ms']>0,'Timing critical clock below native return cost')
    expected_usable=row['error'] is None and answer.get('fast_error') is None and not answer.get('archival_errors') and bool(cert.get('valid') and cert.get('guard_passed'))
    need(type(row['usable_decision']) is bool and row['usable_decision']==expected_usable and
        (row['error'] is None or type(row['error']) is str and bool(row['error'])),'Timing usability/error differs')
    if answer.get('engine_attempt_started'):
        from . import native_ownership_trace as codec
        raw=codec.trace_bytes(answer['trace'])
        need(hashlib.sha256(raw).hexdigest()==cert['event_sha256'] and len(raw)==24*cert['event_count'],
             'Timing returned raw hash/count differs')
        owned=base64.b64decode(answer['fast_owned_high'],validate=True)
        violations=base64.b64decode(answer['fast_violations'],validate=True)
        need(len(owned)==40*cert['owned_count'] and hashlib.sha256(owned).hexdigest()==cert['owned_sha256'] and
            len(violations)==16*cert['violations_count'] and hashlib.sha256(violations).hexdigest()==cert['violations_sha256'],
            'Timing returned ownership/prefix telemetry differs')
        if not cert['invalid'] and not cert['overflow']:
            reference=guards.reference_trace(answer['trace'],loaded,ORIGIN,claimed)
            need(reference['owned_records']==owned and reference['violation_records']==violations and
                reference['passed']==cert['guard_passed'] and
                reference['owned_event_count']==cert['owned_occurrences'],'Timing full raw/native guard partition differs')
    else:
        need(not row['usable_decision'] and row['event_count']==0 and row['owned_count']==0 and
             cert.get('valid') is False,'Unstarted attempt claimed physical work or usability')
    if row['usable_decision']:
        need(clock['requested_outer_ms']==25 and clock['archive_outside_clock'] is True and
            type(clock['search_window_ms']) is int and policy['minimum_search_ms']<=clock['search_window_ms']<=
            math.floor(25-policy['reserve_ms']) and row['native_search_ms']==clock['search_window_ms'],
            'Timing useful native search grant differs')
        fast.validate_decision(answer,row['engine'],root,saved['decision_id'],identity['native'])
        reset=answer['both_seat_reset']
        from .ownership_protocol import RESET_FIELDS
        need(reset['reason']=='every_decision' and reset['fields']==list(RESET_FIELDS) and
            reset['after']==answer['lifecycle']['before'] and saved['cpu']==dict(group=0,
                logical=identity['cpu_declared']['logical'],mask=1<<identity['cpu_declared']['logical']),
            'Timing reset ordering/fields or actual CPU differs')
        if scheduled['index']==0:
            need(reset['before']==dict(created=2,destroyed=0,retained_inference_controls=dict(control=1,candidate=2)),
                'First timed attempt was prewarmed with engine controls')
        need(reference['owned_records']==owned and reference['passed'] and
             reference['raw_sha256']==cert['event_sha256'] and reference['operation_counts']==answer['trace']['operation_counts'],
             'Native/reference trace coverage or guard differs')
        expected=guards.stage_reference(owned,loaded,ORIGIN,claimed,reference['claims_before_sha256'],1,0)
        need(all(cert.get(k)==v for k,v in expected.items() if k not in ('delta_records','canonical_restored_claim_hash')),
             'Timing staged-map reference differs')
        need(saved['transition']==literal.transition(root,answer['decision']['chosen'],literal._literal()) and
             saved['transition']['after']==answer['transition']['position'],'Timing literal choice differs')
    return row

def summary(rows):
    result={}
    for engine in contract.ENGINES:
        subset=[r for r in rows if r['engine']==engine];values=sorted(r['critical_ms'] for r in subset)
        result[engine]=dict(attempts=len(values),p95_ms=values[math.ceil(.95*len(values))-1],
            maximum_ms=max(values),median_ms=values[len(values)//2],usable=sum(r['usable_decision'] for r in subset),
            total_events=sum(r['event_count'] for r in subset),archival_ms=sum(r['archival_ms'] for r in subset))
    return result

def _rows(output):return [read(p) for p in sorted((output/'rows').glob('*.json'))]

def _prefix(output,rows,panel,loaded,claimed,identity):
    """Pure full committed-prefix check before adopting work or loading engines."""
    need({p.name for p in (output/'rows').glob('*.json')}=={f'{i:06d}.json' for i in range(len(rows))},
         'Timing partial row filename gap')
    epochs={}
    for path in sorted((output/'epochs').glob('*.json')):
        epoch=read(path);start=epoch['start_index']
        need(type(start) is int and 0<=start<=len(rows) and start<240 and path.name==f'{start:06d}.json' and
            start not in epochs and epoch==dict(schema='trace-speed-provider-epoch-v1',start_index=start,
                committed_prefix_sha256=sha(output/'commits'/f'{2*start-1:06d}.json') if start else None,
                initial_lifecycle=dict(created=2,destroyed=0,retained_inference_controls=dict(control=1,candidate=2)),
                native_identity_sha256=config.digest(identity['native'])),
            'Timing source-bound cold provider epoch differs')
        epochs[start]=dict(path=path.relative_to(output).as_posix(),sha256=sha(path))
    need(not rows or 0 in epochs,'Timing initial physical provider epoch missing')
    previous_lifecycle=None;current_epoch=None
    for index in range(len(rows)+(1 if (output/'attempts'/f'{len(rows):06d}.json').exists() else 0)):
        saved=read(output/'attempts'/f'{index:06d}.json')
        starts=[start for start in epochs if start<=index];need(starts,'Timing attempt has no declared provider epoch')
        start=max(starts);need(saved['provider_epoch']==epochs[start],'Timing archive/provider epoch binding differs')
        reset=saved['answer'].get('both_seat_reset');life=saved['answer'].get('lifecycle')
        if saved['row']['usable_decision'] or saved['answer'].get('engine_attempt_started'):
            expected=read(output/epochs[start]['path'])['initial_lifecycle'] if start!=current_epoch else previous_lifecycle
            need(type(reset) is dict and type(life) is dict and reset['before']==expected and reset['after']==life['before'],
                 'Timing unlogged native controls or incomplete lifecycle between declared attempts')
        previous_lifecycle=life['after'] if life else None;current_epoch=start
    events=[];previous=None
    for index,path in enumerate(sorted((output/'commits').glob('*.json'))):
        commit=read(path)
        need(path.name==f'{index:06d}.json' and commit['index']==index and
            commit['identity_sha256']==sha(output/'identity.json') and commit['previous_sha256']==previous and
            commit['event']==f'events/{index:06d}-{commit["event_sha256"]}.json' and
            sha(output/commit['event'])==commit['event_sha256'],'Timing partial commit chain differs')
        events.append(read(output/commit['event']));previous=sha(path)
    need(2*len(rows)<=len(events)<=min(480,2*len(rows)+2),'Timing partial event count differs')
    for index in range((len(events)+1)//2):
        restore=events[2*index]
        need(restore==dict(kind='guard_restore_context',attempt=index,generation=0,
            canonical_claim_hash=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),guard_stamp=loaded.guard_stamp,
            prefix_sha256=sha(output/'commits'/f'{2*index-1:06d}.json') if index else None),
            'Timing partial restore/prefix differs')
        if 2*index+1>=len(events):continue
        archivepath=output/'attempts'/f'{index:06d}.json';need(archivepath.exists(),'Timing check has no archive')
        saved=read(archivepath);row=dict(saved['row'],archive_sha256=sha(archivepath))
        if index>=len(rows):_check_row(output,row,panel['schedule'][index],panel,loaded,claimed,identity)
        else:need(row==rows[index],'Timing committed row/archive differs')
        check=dict(events[2*index+1]);recovered=check.pop('recovered_pure',None)
        need(recovered in (None,True),'Timing recovery receipt differs')
        expected=dict(kind='ownership_check' if row['usable_decision'] else 'attempt_hold',attempt=index,archive_sha256=row['archive_sha256'])
        if row['usable_decision']:expected['result']=dict(passed=True,origin=ORIGIN,fast_guard_certificate=saved['answer']['fast_certificate'])
        else:expected.update(error=row['error'],production_admission=False)
        need(check==expected,'Timing partial check/archive authorization differs')
    for part in ('intents','attempts'):
        names={p.name for p in (output/part).glob('*.json')}
        required={f'{i:06d}.json' for i in range(len(rows))}
        allowed=required|({f'{len(rows):06d}.json'} if len(rows)<240 else set())
        need(required<=names<=allowed,'Timing partial artifact cursor gap')
    return events

def _adopt(output,rows,panel,loaded,claimed,identity):
    """Repair a returned attempt before opening any provider or CPU context."""
    while len(rows)<240:
        scheduled=panel['schedule'][len(rows)];index=scheduled['index']
        intentpath=output/'intents'/f'{index:06d}.json';archivepath=output/'attempts'/f'{index:06d}.json'
        if not intentpath.exists():break
        need(archivepath.exists(),'Interrupted search has no complete returned archive: HOLD; mapped prefix retained')
        saved=read(archivepath);row=dict(saved['row'],archive_sha256=sha(archivepath))
        _check_row(output,row,scheduled,panel,loaded,claimed,identity)
        existing=_prefix(output,rows,panel,loaded,claimed,identity)
        kind='ownership_check' if row['usable_decision'] else 'attempt_hold'
        checks=[e for e in existing if e['kind']==kind and e['attempt']==index]
        if not checks:
            event=dict(kind=kind,attempt=index,archive_sha256=row['archive_sha256'],recovered_pure=True)
            if row['usable_decision']:event['result']=dict(passed=True,origin=ORIGIN,fast_guard_certificate=saved['answer']['fast_certificate'])
            else:event.update(error=row['error'],production_admission=False)
            _commit(output,event)
        need(len(checks)<=1,'Duplicate recovered ownership check')
        atomic(output/'rows'/f'{index:06d}.json',row);rows.append(row)
    return rows

def _verify(output,require_catalog=True):
    """Pure full evidence reader; does not construct native providers."""
    output=stage.path_of(output,'timing');identity=read(output/'identity.json')
    expected,spec,panel,loaded=_identity(ROOT/identity['spec_path']);need(identity==expected,'Timing identity changed')
    claimed=_pressure(spec,loaded);rows=_rows(output)
    need(len(rows)==240,'Timing panel incomplete')
    expected_names={f'{i:06d}.json' for i in range(240)}
    need(all({p.name for p in (output/part).glob('*.json')}==expected_names for part in ('rows','intents','attempts')),
         'Timing exact attempt artifact coverage differs')
    for row,scheduled in zip(rows,panel['schedule']):_check_row(output,row,scheduled,panel,loaded,claimed,identity)
    _prefix(output,rows,panel,loaded,claimed,identity)
    previous=None;events=[]
    for index,path in enumerate(sorted((output/'commits').glob('*.json'))):
        c=read(path);need(path.name==f'{index:06d}.json' and c['index']==index and c['identity_sha256']==sha(output/'identity.json') and
            c['previous_sha256']==previous and sha(output/c['event'])==c['event_sha256'],'Timing journal chain differs')
        events.append(read(output/c['event']));previous=sha(path)
    checks=[e for e in events if e['kind']=='ownership_check']
    need(len(checks)==sum(r['usable_decision'] for r in rows),'Timing guard check journal differs')
    need(len(events)==480,'Timing exact restore/check event coverage differs')
    for row in rows:
        restore,decision=events[2*row['index']:2*row['index']+2]
        need(restore==dict(kind='guard_restore_context',attempt=row['index'],generation=0,
            canonical_claim_hash=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),guard_stamp=loaded.guard_stamp,
            prefix_sha256=sha(output/'commits'/f'{2*row["index"]-1:06d}.json') if row['index'] else None) and
            decision['attempt']==row['index'] and decision['kind']==('ownership_check' if row['usable_decision'] else 'attempt_hold') and
            decision['archive_sha256']==row['archive_sha256'],'Timing independent restore/check cursor differs')
        matched=[e for e in checks if e['attempt']==row['index']]
        if row['usable_decision']:
            need(len(matched)==1 and matched[0]['archive_sha256']==row['archive_sha256'] and
                 matched[0]['result']['fast_guard_certificate']==read(output/row['archive_path'])['answer']['fast_certificate'],
                 'Timing check/archive binding differs')
    report=read(output/'report.json');gate=contract.timing_gate(rows,sha(ROOT/identity['spec_path']))
    need(report['schema']=='trace-speed-timing-report-v1' and report['rows']==rows and report['gate']==gate and
         report['protocol_sha256']==sha(ROOT/identity['spec_path']) and report['summary']==summary(rows) and
         report['training_counters']==stage.ZERO and report['models_fixed'] is True and report['policy_fixed'] is True and
         report['strength_claim'] is False,'Timing report/gate/scope differs')
    if require_catalog:
        catalog=read(output/'catalog.json');need(catalog['report_sha256']==sha(output/'report.json') and
             catalog['identity_sha256']==sha(output/'identity.json') and catalog['last_commit_sha256']==previous,'Timing catalog differs')
    return dict(passed=True,report=report['summary'],timing_gate=gate,completed_reader_no_engine=True)

def completed(output):return _verify(output,True)

def _finish(output,spec_path,rows):
    gate=contract.timing_gate(rows,sha(spec_path))
    atomic(output/'report.json',dict(schema='trace-speed-timing-report-v1',protocol_sha256=sha(spec_path),rows=rows,gate=gate,
        summary=summary(rows),training_counters=stage.ZERO,models_fixed=True,policy_fixed=True,strength_claim=False))
    commits=sorted((output/'commits').glob('*.json'));verified=_verify(output,False)
    atomic(output/'catalog.json',dict(schema='trace-speed-timing-catalog-v1',identity_sha256=sha(output/'identity.json'),
        report_sha256=sha(output/'report.json'),last_commit_sha256=sha(commits[-1])))
    return verified

def run(output,spec_path,lease_path,pause_at=None,pause_marker=None):
    output=stage.path_of(output,'timing')
    if (output/'catalog.json').exists():return completed(output)
    need(pause_at is None,'Timing uses immutable attempt recovery; handoff worker owns kill fixtures')
    prepare(output,spec_path);identity,spec,panel,loaded=_identity(spec_path);claimed=_pressure(spec,loaded)
    output.mkdir(parents=True,exist_ok=True)
    with Writer(output):
        rows=_rows(output)
        for row,scheduled in zip(rows,panel['schedule']):_check_row(output,row,scheduled,panel,loaded,claimed,identity)
        _prefix(output,rows,panel,loaded,claimed,identity)
        # An unreturned attempt is not silently rerun. Complete archived orphan
        # adoption is explicit, source/cursor checked and never runs the engine.
        rows=_adopt(output,rows,panel,loaded,claimed,identity)
        _prefix(output,rows,panel,loaded,claimed,identity)
        if len(rows)==240:return _finish(output,spec_path,rows)
        need(all(row['usable_decision'] for row in rows),'Failed timing prefix requires review: HOLD before any new query')
        from . import native_trace_fast_v2 as fast
        with PerformanceCore(identity['cpu_declared']) as cpu,ExitStack() as stack:
            providers=fast.open_providers(stack,capacity=spec['capacity_events'],policy=spec['search_policy'])
            epochpath=output/'epochs'/f'{len(rows):06d}.json'
            atomic(epochpath,dict(schema='trace-speed-provider-epoch-v1',start_index=len(rows),
                committed_prefix_sha256=sha(output/'commits'/f'{2*len(rows)-1:06d}.json') if rows else None,
                initial_lifecycle=providers.lifecycle(),native_identity_sha256=config.digest(identity['native'])))
            epochref=dict(path=epochpath.relative_to(output).as_posix(),sha256=sha(epochpath))
            (output/'raw').mkdir(exist_ok=True)
            guard=stack.enter_context(providers.attach_guard(loaded))
            guard.bind_run(dict(path=str(output/'identity.json'),sha256=sha(output/'identity.json')),
                len(list((output/'commits').glob('*.json')))-1)
            ref=literal._literal()
            for scheduled in panel['schedule'][len(rows):]:
                index=scheduled['index'];intentpath=output/'intents'/f'{index:06d}.json';archivepath=output/'attempts'/f'{index:06d}.json'
                rowpath=output/'rows'/f'{index:06d}.json';root=panel['panel'][scheduled['root']]['state']
                decision_id=f'timing/{index:06d}';resources.assert_worker_lease(lease_path)
                need(resources.namespace_bytes()+2*spec['capacity_events']*24+128<read(lease_path)['protocol']['quota_bytes'],
                     'Insufficient namespace headroom for complete raw evidence')
                guard.restore(claimed,generation=0)
                need(not intentpath.exists(),'Recovery cursor changed after pure adoption')
                restore=dict(kind='guard_restore_context',attempt=index,generation=0,
                    canonical_claim_hash=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),guard_stamp=loaded.guard_stamp,
                    prefix_sha256=sha(output/'commits'/f'{2*index-1:06d}.json') if index else None)
                prior_restore=output/'commits'/f'{2*index:06d}.json'
                if prior_restore.exists():need(read(output/read(prior_restore)['event'])==restore,'Uncommitted intent restore context differs')
                else:_commit(output,restore)
                atomic(intentpath,dict(scheduled=scheduled,root=root,identity_sha256=sha(output/'identity.json'),decision_id=decision_id,
                    claims_sha256=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest(),unreturned_attempt='HOLD, never retry'))
                transient=None;error=None;transition=None;cpu_receipt=None;tick=time.perf_counter()
                try:
                    transient=providers.decide(scheduled['engine'],root,decision_id,guard=guard,origin=ORIGIN,
                        outer_started=tick,deadline=tick+.025,attempt_sink=output/'raw'/f'{index:06d}.bin')
                    transition=literal.transition(root,transient['answer']['decision']['chosen'],ref)
                    need(transition['after']==transient['answer']['transition']['position'],'Chosen literal/native mismatch')
                    cpu_receipt=cpu.verify()
                except fast.TraceHold as exc:transient=exc.transient;error=str(exc)
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
                critical_ms=(time.perf_counter()-tick)*1000
                need(transient is not None,'Native return capture unavailable: HOLD; intent and mapped prefix retained')
                answer=fast.archive_attempt(transient)
                if answer.get('archival_errors'):error='Archival HOLD: '+str(answer['archival_errors'])
                cert=answer['fast_certificate'];usable=error is None and bool(cert.get('valid') and cert.get('guard_passed'))
                row=dict(scheduled,critical_ms=critical_ms,usable_decision=usable,
                    coverage_passed=bool(cert.get('valid') and cert.get('attempted_events')==cert.get('event_count')),
                    guard_passed=bool(cert.get('guard_passed')),event_count=cert.get('event_count',0),
                    owned_count=cert.get('owned_count',0),native_search_ms=answer.get('decision',{}).get('requested_ms'),
                    archival_ms=answer['archival_timing']['elapsed_ms'],error=error,
                    archive_path=archivepath.relative_to(output).as_posix())
                # Avoid self-referential archive SHA: row lives in packet without
                # its hash; persisted row adds it after the immutable packet.
                saved=dict(scheduled=scheduled,root=root,decision_id=decision_id,identity_sha256=sha(output/'identity.json'),
                    answer=answer,transition=transition,cpu=cpu_receipt,provider_epoch=epochref,row=dict(row))
                atomic(archivepath,saved);row['archive_sha256']=sha(archivepath)
                if usable:
                    proof=_commit(output,dict(kind='ownership_check',attempt=index,archive_sha256=row['archive_sha256'],
                        result=dict(passed=True,origin=ORIGIN,fast_guard_certificate=cert)))
                    guard.publish_stage(cert['generation'],ORIGIN,proof)
                else:
                    guard.discard_stage();_commit(output,dict(kind='attempt_hold',attempt=index,error=error,
                        archive_sha256=row['archive_sha256'],production_admission=False))
                atomic(rowpath,row);rows.append(row)
                print(config.canonical(dict(event='timing_attempt_archived',**scheduled,critical_ms=critical_ms,usable=usable)),flush=True)
                need(usable,'Timing attempt failed: captured archive and row retained; HOLD before further engine queries')
        return _finish(output,spec_path,rows)
