"""Recoverable consumed-state handoffs using the complete native fast guard.

The journal is the sole authority.  A saved complete orphan archive is checked
before any engine loads; an attempt intent without a complete archive holds.
No training or production evaluation admission is granted by these fixtures.
"""
from contextlib import ExitStack
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
from . import config,tracing_speed_common as stage,tracing_speed_contract as contract
from . import ownership_fast_guard as guard,ownership_protocol as protocol
from . import value_run_arena as literal
from .valuepilot_reservations import Writer

ROOT=stage.ROOT
DEFAULT=stage.FOLDER/'jobs/engineering-fixed'
SPEC=stage.FOLDER/'jobs/engineering-spec.json'
SCHEMA='tracing-speed-worker-v2'
SOURCES=('congklak/tracing_speed_worker_v2.py','congklak/ownership_fast_guard.py',
    'congklak/tracing_speed_common.py','congklak/tracing_speed_contract.py',
    'congklak/native_trace_fast_v1.py','congklak/ownership_protocol.py','congklak/value_run_arena.py',
    'congklak/timing_v2.py','tests/reference.py')
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic


def sources():
    from . import native_trace_fast_v1 as native
    return dict(files=stage.common.pins(SOURCES),native=native.source_identity())


def prepare_spec(path=SPEC,mode='fixed128',policy=None,policy_source=None):
    """Declare only the old consumed geometry; never acquire fresh starts."""
    from . import tracing_speed_launcher_v2 as launcher,native_trace_fast_v1 as native
    path=stage.path_of(path,'jobs');old_path=ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json'
    old=read(old_path);schedule=[]
    for index in (4,7,8):
        row=dict(old['schedule'][index]);row['index']=len(schedule);schedule.append(row)
    inputs=dict(old['inputs']);inputs.update(stage.common.pins(SOURCES+(old_path.relative_to(ROOT).as_posix(),'build/value-plan.json')))
    calibration=None
    if mode=='timed25':
        need(policy_source is not None,'Timed handoff requires immutable calibration source')
        calibration_path=Path(policy_source).resolve()
        need(calibration_path==stage.FOLDER/'native/calibration-v2/report.json','Timed calibration path differs')
        report=read(calibration_path)
        need(report['policy_frozen'] is True and report['search_policy']==policy,'Calibrated policy/source relation differs')
        calibration=dict(path=calibration_path.relative_to(ROOT).as_posix(),sha256=sha(calibration_path))
        inputs[calibration['path']]=calibration['sha256']
    spec=dict(schema=SCHEMA+'-engineering-spec',kind='handoff',engineering_fixture=True,mode=mode,
        schedule=schedule,guard=old['guard'],inputs=inputs,models_fixed=True,policy_fixed=True,
        production_games=0,training_counters=stage.ZERO,launcher_identity=launcher.identity(),
        policy=dict(native.DEFAULT_POLICY if policy is None else policy),capacity_events=2000000,
        calibration_source=calibration,
        scope='Consumed engineering starts only; support320/missing-historical-trace HOLDs unchanged')
    need(mode in ('fixed128','timed25'),'Unknown engineering mode')
    spec['guard_refs']=guard.prepare(spec,guard.DEFAULT/f'handoff-v2-{mode}-compiled')
    atomic(path,spec);return stage.normalized(spec)


def _validate_spec(spec,spec_path):
    from . import tracing_speed_launcher_v2 as launcher
    launcher.validate_spec(spec_path)
    need(spec['schema']==SCHEMA+'-engineering-spec' and spec['kind']=='handoff' and
         spec['engineering_fixture'] is True and spec['production_games']==0 and
         spec['mode'] in ('fixed128','timed25') and spec['capacity_events']==2000000 and
         spec['training_counters']==stage.ZERO and 1<=len(spec['schedule'])<=3,'Unadmitted handoff scope')
    old_path=ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json';old=read(old_path)
    expected_schedule=[]
    for index in (4,7,8):
        row=dict(old['schedule'][index]);row['index']=len(expected_schedule);expected_schedule.append(row)
    required=dict(old['inputs']);required.update(stage.common.pins(SOURCES+(
        old_path.relative_to(ROOT).as_posix(),'build/value-plan.json')))
    need(spec['schedule']==expected_schedule and spec['guard']==old['guard'] and
         all(spec['inputs'].get(name)==checksum for name,checksum in required.items()),
         'Engineering schedule/guard/source authorization differs from exact consumed fixtures')
    need(spec['mode']!='timed25' or spec['policy'].get('calibrated') is True,
         'Timed handoffs require a prospectively frozen calibrated common debt policy')
    if spec['mode']=='timed25':
        calibration=spec['calibration_source'];calibration_path=ROOT/calibration['path'];report=read(calibration_path)
        need(calibration_path==stage.FOLDER/'native/calibration-v2/report.json' and
             spec['inputs'].get(calibration['path'])==calibration['sha256']==sha(calibration_path) and
             report['policy_frozen'] is True and report['search_policy']==spec['policy'],
             'Timed policy lacks exact frozen calibration evidence')
    for index,row in enumerate(spec['schedule']):
        need(row['index']==index and row['profile'] in protocol.PROFILES and row['baseline'] in protocol.BASELINES and
             row['seat'] in (0,1) and row['source'] in spec['inputs'] and
             literal.normalize(row['start'])['state']==row['start'],'Handoff frozen schedule differs')
    guard.load_pure(spec['guard_refs'],spec)


def prepare(path=DEFAULT,spec_path=SPEC):
    path=stage.path_of(path,'jobs');spec_path=stage.path_of(spec_path,'jobs');spec=read(spec_path)
    _validate_spec(spec,spec_path)
    identity=dict(schema=SCHEMA+'-identity',spec_path=spec_path.relative_to(ROOT).as_posix(),spec_sha256=sha(spec_path),
        spec=spec,sources=sources(),guard_refs=spec['guard_refs'],clock_contract_sha256=config.digest(contract.definition()),
        rules_digest=config.RULES_DIGEST,models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,
        cpu_declared=read(ROOT/'build/value-plan.json')['cpu_declared'])
    atomic(path/'identity.json',identity);return stage.normalized(identity)


def _journal(path,identity):
    rows=[];previous=None;ids=sha(path/'identity.json')
    for index,p in enumerate(sorted((path/'commits').glob('*.json'))):
        need(p.name==f'{index:06d}.json','Trace-speed commit gap')
        commit=read(p);rel=f'events/{index:06d}-{commit["event_sha256"]}.json'
        need(commit==dict(index=index,identity_sha256=ids,event=rel,event_sha256=sha(path/rel),previous_sha256=previous),
             'Trace-speed immutable commit/prefix differs')
        rows.append(read(path/rel));previous=sha(p)
    return rows,previous


def append(path,events,event):
    event=stage.normalized(event);index=len(events);payload=(config.canonical(event)+'\n').encode()
    digest=hashlib.sha256(payload).hexdigest();rel=f'events/{index:06d}-{digest}.json'
    atomic(path/rel,event);commit=dict(index=index,identity_sha256=sha(path/'identity.json'),event=rel,
        event_sha256=digest,previous_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None)
    cp=path/'commits'/f'{index:06d}.json';atomic(cp,commit);events.append(event)
    print(config.canonical(dict(event='trace_speed_commit',index=index,kind=event['kind'],game=event.get('game'))),flush=True)
    return dict(event_path=str(path/rel),event_sha256=digest,commit_path=str(cp),commit_sha256=sha(cp),index=index)


def _archive_path(path,relative):
    result=(path/relative).resolve()
    need(result.is_relative_to(path/'archives') and result.suffix=='.json','Decision archive path differs');return result


def _archive(path,intent):
    ap=_archive_path(path,intent['archive']);body=read(ap)
    need(body['schema']==SCHEMA+'-decision-archive' and body['decision_id']==intent['decision_id'] and
         body['before']==intent['before'] and body['engine']==intent['engine'] and
         body['identity_sha256']==sha(path/'identity.json'),'Archived decision/intent/run identity differs')
    return body


def check_saved(archive,identity,compiled,origin,claimed,claim_hash,generation):
    """Full literal archive validation without constructing engines or models."""
    from . import native_trace_fast_v1 as native
    answer=archive['answer'];native.validate_decision(answer,archive['engine'],archive['before'],archive['decision_id'],identity['sources']['native'])
    legal=literal.normalize(archive['before'])['legal'];action=answer['decision']['chosen']
    need(action in legal and archive['legal']==legal and
         archive['transition']==literal.transition(archive['before'],action) and
         answer['transition']['position']==archive['transition']['after'],'Saved chosen literal move differs')
    ref=guard.reference_trace(answer['trace'],compiled,origin,claimed)
    owned=base64.b64decode(answer['fast_owned_high'],validate=True)
    violations=base64.b64decode(answer['fast_violations'],validate=True)
    need(owned==ref['owned_records'] and violations==ref['violation_records'] and ref['passed'],
         'Saved complete high footprint/guard differs')
    cert=answer['fast_certificate'];expected=guard.stage_reference(owned,compiled,origin,claimed,claim_hash,generation+1,generation)
    for name,value in expected.items():
        if name not in ('canonical_restored_claim_hash','delta_records'):need(cert.get(name)==value,'Native stage certificate differs: '+name)
    need(cert['event_sha256']==ref['raw_sha256'] and cert['owned_sha256']==ref['owned_sha256'] and
         cert['owned_count']==ref['owned_unique_count'] and cert['guard_passed'] is True,'Native full trace certificate differs')
    clock=archive['clock'];timed=identity['spec']['mode']=='timed25'
    need(type(clock['critical_ms']) in (float,int) and math.isfinite(clock['critical_ms']) and clock['critical_ms']>0 and
         clock['whole_decision_contract_sha256']==identity['clock_contract_sha256'] and
         clock['raw_hash_guard_claim_staging_included'] is True and clock['archival_outside_clock'] is True and
         clock['requested_ms']==(25 if timed else None),'Saved whole decision clock differs')
    native_elapsed=answer['fast_timing']['outer_elapsed_ms']
    need(type(native_elapsed) in (int,float) and math.isfinite(native_elapsed) and
         0<native_elapsed<=clock['critical_ms'] and clock['critical_ms']>=answer['decision']['wall_ms'] and
         answer['fast_timing']['archive_outside_clock'] is True,'Critical interval omits native/provider work')
    timing=answer['fast_timing'];archival=answer['archival_timing']
    need(timing['policy']==identity['spec']['policy'] and timing['requested_outer_ms']==(25 if timed else None) and
         timing['search_window_ms']==answer['decision']['requested_ms'] and
         type(timing['finalization_ms']) in (int,float) and math.isfinite(timing['finalization_ms']) and
         0<timing['finalization_ms']<=native_elapsed and
         type(timing['search_window_ms']) is int and
         (not timed or timing['policy']['minimum_search_ms']<=timing['search_window_ms']<=math.floor(25-timing['policy']['reserve_ms'])) and
         (timed or timing['search_window_ms']==0 and answer['work_budget']==dict(
             kind='decision_nodes' if archive['engine'] in ('H_v1','H8_relay_CPU') else 'PUCT_edge_backups',value=128)) and
         type(archival['elapsed_ms']) in (int,float) and math.isfinite(archival['elapsed_ms']) and archival['elapsed_ms']>=0 and
         archival['charged_to_lease'] is True and archival['encoding_only_after_owned_validated_snapshot'] is True,
         'Policy/grant/finalization/archival telemetry differs')
    cpu=identity['cpu_declared']
    need(archive['cpu']==dict(group=cpu['group'],logical=cpu['logical'],mask=1<<cpu['logical']),
         'Actual decision CPU differs from declared core')
    reset=answer['both_seat_reset'];need(reset['actual_controls_replaced']==2 and reset['decision_id']==archive['decision_id'] and
         reset['reason']=='every_decision' and reset['fields']==list(protocol.RESET_FIELDS) and
         reset['source_identity_sha256']==config.digest(identity['sources']['native']) and
         reset['after']['created']-reset['before']['created']==2 and
         reset['after']['destroyed']-reset['before']['destroyed']==2 and
         all(reset['after']['retained_inference_controls'][p]>reset['before']['retained_inference_controls'][p]
             for p in ('control','candidate')) and answer['lifecycle']['before']==reset['after'],
         'Both-seat reset receipt/native lifecycle differs')
    return dict(schema=SCHEMA+'-ownership-check',origin=origin,passed=True,fast_guard_certificate=cert,
        raw_sha256=ref['raw_sha256'],owned_sha256=ref['owned_sha256'],owned_count=ref['owned_unique_count'],
        guard_stamp=compiled.guard_stamp,engineering_exception=True,production_admission=False),owned


def replay(path,identity,compiled=None):
    compiled=guard.load_pure(identity['guard_refs'],identity['spec']) if compiled is None else compiled
    events,prefix_sha=_journal(path,identity);spec=identity['spec'];game=0;state=None;suffix=False;choices=0
    claimed={};generation=0;claim_hash=hashlib.sha256(b'').hexdigest();intent=None;pending=None;hold=None
    games=[];decisions=[];handoffs=[];checks=[];restores=[];last_guard_index=-1;last_lifecycle=None
    for index,event in enumerate(events):
        kind=event['kind'];need(not hold,'Trace-speed journal continued after HOLD')
        if kind=='guard_restore_context':
            need(intent is None and pending is None and event['generation']==generation and
                 event['canonical_claims_sha256']==hashlib.sha256(compiled.claims_blob(claimed)).hexdigest() and
                 event['guard_stamp']==compiled.guard_stamp and event['committed_prefix_sha256']==
                     (sha(path/'commits'/f'{index-1:06d}.json') if index else None),
                 'Cold restore context/prefix differs')
            claim_hash=event['canonical_claims_sha256'];restores.append(event);last_lifecycle=None;continue
        need(event['game']==game and game<len(spec['schedule']),'Trace-speed game cursor differs');row=spec['schedule'][game]
        if kind=='game_start':
            need(state is None and intent is None and pending is None and event['start']==row['start'],'Trace-speed start differs')
            state=row['start'];suffix=False;choices=0
        elif kind=='handoff':
            need(state is not None and intent is None and pending is None and not suffix and protocol.active(state)<=24 and
                 event['state']==state and event['terminal']==literal.normalize(state)['terminal'] and
                 event['reset_fields']==list(protocol.RESET_FIELDS),'Frozen endgame handoff differs')
            suffix=True;handoffs.append(event)
        elif kind=='decision_intent':
            need(restores and state is not None and intent is None and pending is None and not literal.normalize(state)['terminal'] and
                 event['before']==state and event['choice']==choices and event['decision_id']==f'{game:03d}/{choices:03d}',
                 'Decision intent cursor differs')
            dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],suffix)
            need(not dispatch['handoff_required'] and event['engine']==dispatch['engine'],'Decision dispatch differs')
            need(event['archive']==f'archives/{game:03d}-{choices:03d}.json' and
                 event['attempt_sink']==f'attempts/{game:03d}-{choices:03d}.bin','Attempt archive/sink cursor differs')
            _archive_path(path,event['archive']);intent=event
        elif kind=='decision':
            need(intent is not None and pending is None and event['decision_id']==intent['decision_id'] and
                 event['archive']==intent['archive'] and sha(_archive_path(path,event['archive']))==event['archive_sha256'],
                 'Decision archive/intent binding differs')
            archive=_archive(path,intent);result,owned=check_saved(archive,identity,compiled,row['origin'],claimed,claim_hash,generation)
            need(last_lifecycle is None or archive['answer']['both_seat_reset']['before']==last_lifecycle,
                 'Decision reset/native lifecycle is discontinuous within a worker epoch')
            initial=archive['answer']['both_seat_reset']['before']
            need(last_lifecycle is not None or initial['created']==2 and initial['destroyed']==0 and
                 sorted(initial['retained_inference_controls'].values())==[1,2],
                 'Cold provider performed prior hidden control work before its first decision')
            last_lifecycle=archive['answer']['lifecycle']['after']
            pending=dict(event=event,archive=archive,result=result,owned=owned);decisions.append(pending)
        elif kind=='ownership_check':
            need(pending is not None and event['decision_id']==intent['decision_id'] and event['result']==pending['result'],
                 'Committed ownership check/certificate differs')
            for owned in guard.owned_rows(pending['owned']):claimed[owned['quotient']]=row['origin']
            generation+=1;claim_hash=event['result']['fast_guard_certificate']['staged_claim_hash']
            state=pending['archive']['transition']['after'];choices+=1;intent=None;pending=None;checks.append(event);last_guard_index=index
        elif kind=='game_end':
            need(state is not None and intent is None and pending is None and suffix and literal.normalize(state)['terminal'] and
                 event['final']==state and event['choices']==choices and event['terminal_margin']==state['stores'][0]-state['stores'][1],
                 'Trace-speed terminal summary differs')
            games.append(event);game+=1;state=None;suffix=False;choices=0
        elif kind=='hold':need(event['terminal'] is True and event['reason'],'Malformed retained HOLD');hold=event
        else:need(False,'Unknown trace-speed event')
    return dict(events=events,prefix_sha256=prefix_sha,game=game,state=state,suffix=suffix,choices=choices,claimed=claimed,
        generation=generation,claim_hash=claim_hash,intent=intent,pending=pending,hold=hold,games=games,
        decisions=decisions,handoffs=handoffs,checks=checks,restores=restores,last_guard_index=last_guard_index)


def summary(data,identity):
    ds=data['decisions'];spec=identity['spec']
    return dict(schema=SCHEMA+'-report',passed=True,engineering_fixture=True,models_fixed=True,policy_fixed=True,
        production_admission=False,training_admission=False,evaluation_admission=False,training_counters=stage.ZERO,
        mode=spec['mode'],games=len(data['games']),decisions=len(ds),handoffs=len(data['handoffs']),
        terminal_boundary_events=sum(h['terminal'] for h in data['handoffs']),owned_claims=len(data['claimed']),
        successful_ownership_checks=data['generation'],restore_contexts=len(data['restores']),
        internal_trace_events=sum(d['archive']['answer']['trace']['event_count'] for d in ds),
        maximum_critical_ms=max((d['archive']['clock']['critical_ms'] for d in ds),default=0),
        overshoot_decisions=sum(d['archive']['clock']['critical_ms']>25 for d in ds) if spec['mode']=='timed25' else None,
        logical_results=[dict(game=d['event']['game'],decision_id=d['archive']['decision_id'],engine=d['archive']['engine'],
            before=d['archive']['before'],action=d['archive']['answer']['decision']['chosen'],after=d['archive']['transition']['after'],
            trace_raw_sha256=d['archive']['answer']['trace']['event_sha256']) for d in ds],
        final_states=[g['final'] for g in data['games']],strength_interpretation_allowed=False,
        confidence_intervals_enabled=False,source_holds_preserved=['320 existing support conflicts','missing historical footprints'])


def completed(path=DEFAULT):
    """Pure: validation precedes any writer, lease reservation or engine load."""
    path=stage.path_of(path,'jobs');identity=read(path/'identity.json');spec=identity['spec']
    need(identity['schema']==SCHEMA+'-identity' and identity['rules_digest']==config.RULES_DIGEST and
         identity['models_fixed'] is True and identity['policy_fixed'] is True and identity['training_counters']==stage.ZERO and
         identity['clock_contract_sha256']==config.digest(contract.definition()) and
         identity['spec_sha256']==sha(ROOT/identity['spec_path']) and read(ROOT/identity['spec_path'])==spec,
         'Completed trace-speed identity differs')
    stage.common.current(spec['inputs']);need(identity['sources']==sources(),'Completed source identity differs')
    _validate_spec(spec,ROOT/identity['spec_path'])
    compiled=guard.load_pure(identity['guard_refs'],spec);need(identity['guard_refs']==spec['guard_refs'],'Identity/spec guard differs')
    data=replay(path,identity,compiled);cat=read(path/'catalog.json');report=read(path/'report.json')
    need(data['game']==len(spec['schedule']) and data['intent'] is None and data['pending'] is None and not data['hold'] and
         cat==dict(schema=SCHEMA+'-complete',passed=True,identity_sha256=sha(path/'identity.json'),
             commits=stage.common.pins(sorted((path/'commits').glob('*.json'))),report_sha256=sha(path/'report.json')) and
         report==summary(data,identity),'Completed trace-speed journal/report differs')
    return dict(catalog=cat,report=report)


completed_status=completed


def _pause(path,point,requested,marker,lease_path):
    if point!=requested:return
    from . import tracing_speed_resources as resources
    resources.assert_worker_lease(lease_path);marker=stage.path_of(marker,'jobs/fixtures')
    atomic(marker,dict(point=point,pid=os.getpid()))
    while True:time.sleep(.05)


def _capacity(lease_path):
    from . import tracing_speed_resources as resources
    lease=resources.assert_worker_lease(lease_path)
    need(resources.namespace_bytes()+2*2000000*24+64<=lease['protocol']['quota_bytes'],
         'Insufficient whole raw/archive namespace quota')


def run(path=DEFAULT,spec_path=SPEC,lease_path=None,pause_at=None,pause_marker=None):
    path=stage.path_of(path,'jobs')
    if (path/'catalog.json').exists():return completed(path)
    need(pause_at is None or pause_at in contract.KILL_POINTS,'Unknown trace-speed pause')
    need(pause_at is None or path.is_relative_to(stage.FOLDER/'jobs/fixtures'),'Pause requires fixture namespace')
    need(lease_path is not None,'Contained cumulative lease required');_capacity(lease_path)
    from . import tracing_speed_resources as resources
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return completed(path)
        identity=prepare(path,spec_path);compiled=guard.load_pure(identity['guard_refs'],identity['spec'])
        data=replay(path,identity,compiled);events=data['events'];need(not data['hold'],'Retained HOLD forbids engine use')
        def refresh():return replay(path,identity,compiled)
        def pause(point):return _pause(path,point,pause_at,pause_marker,lease_path)
        def saved_decision():
            nonlocal data
            intent=data['intent'];ap=_archive_path(path,intent['archive'])
            need(ap.is_file(),'Attempt intent has no complete archive: HOLD; no retry permitted')
            _archive(path,intent)
            pause('before_decision_commit')
            append(path,events,dict(kind='decision',game=data['game'],decision_id=intent['decision_id'],
                archive=intent['archive'],archive_sha256=sha(ap)))
            pause('after_decision_commit');data=refresh()
        def saved_check(native_guard=None):
            nonlocal data
            pending=data['pending'];pause('before_ownership_check_commit')
            proof=append(path,events,dict(kind='ownership_check',game=data['game'],decision_id=pending['archive']['decision_id'],result=pending['result']))
            pause('after_ownership_check_commit')
            if native_guard is not None:
                certificate=pending['result']['fast_guard_certificate']
                native_guard.publish_stage(certificate['generation'],pending['result']['origin'],proof)
            data=refresh()
        # Resolve complete saved work before loading providers, including orphans.
        if data['intent'] is not None and data['pending'] is None:
            if not _archive_path(path,data['intent']['archive']).is_file():
                intent=data['intent'];sink=path/intent['attempt_sink'];partial=None
                if sink.exists():
                    from . import native_trace_fast_v1 as native
                    prefix=native.read_attempt_sink(sink);raw=prefix.pop('raw_trace');partial=dict(prefix,raw_base64=base64.b64encode(raw).decode())
                    atomic(path/'failed-prefixes'/f'{len(events):06d}.json',partial)
                append(path,events,dict(kind='hold',game=data['game'],terminal=True,reason='Attempt intent without complete archive; retained prefix cannot authorize retry',
                    decision_id=intent['decision_id'],attempt_sink=intent['attempt_sink'],prefix_retained=partial is not None))
                need(False,'Missing complete attempt archive: retained HOLD')
            saved_decision()
        if data['pending'] is not None:saved_check()
        # The reset is explicit and durable before engines, including low-only prefixes.
        append(path,events,dict(kind='guard_restore_context',generation=data['generation'],
            canonical_claims_sha256=hashlib.sha256(compiled.claims_blob(data['claimed'])).hexdigest(),
            committed_prefix_sha256=data['prefix_sha256'],guard_stamp=compiled.guard_stamp))
        data=refresh()
        from .timing_v2 import PerformanceCore
        from . import native_trace_fast_v1 as native
        with ExitStack() as stack:
            core=stack.enter_context(PerformanceCore(identity['cpu_declared']));providers=None;live_guard=None
            while data['game']<len(identity['spec']['schedule']):
                _capacity(lease_path);core.verify();row=identity['spec']['schedule'][data['game']];game=data['game']
                if data['state'] is None:
                    append(path,events,dict(kind='game_start',game=game,start=row['start']));data=refresh()
                state=data['state'];normal=literal.normalize(state)
                if protocol.active(state)<=24 and not data['suffix']:
                    pause('before_handoff_commit')
                    append(path,events,dict(kind='handoff',game=game,state=state,terminal=normal['terminal'],reset_fields=list(protocol.RESET_FIELDS)))
                    pause('after_handoff_commit');data=refresh()
                if normal['terminal']:
                    append(path,events,dict(kind='game_end',game=game,final=state,choices=data['choices'],terminal_margin=state['stores'][0]-state['stores'][1]))
                    data=refresh();continue
                if data['choices']>=512:
                    append(path,events,dict(kind='hold',game=game,terminal=True,reason='Engineering choice cap'))
                    need(False,'Engineering choice cap: retained HOLD')
                if providers is None:
                    providers=native.open_providers(stack,capacity=identity['spec']['capacity_events'],policy=identity['spec']['policy'])
                    need(providers.identity==identity['sources']['native'],'Provider identity differs after prepare')
                    live_guard=providers.attach_guard(compiled);stack.callback(live_guard.close)
                    live_guard.restore(data['claimed'],generation=data['generation'])
                    live_guard.bind_run({'path':str(path/'identity.json'),'sha256':sha(path/'identity.json')},data['last_guard_index'])
                decision_id=f'{game:03d}/{data["choices"]:03d}';dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],data['suffix'])
                stem=f'{game:03d}-{data["choices"]:03d}';archive_relative=f'archives/{stem}.json';sink_relative=f'attempts/{stem}.bin'
                append(path,events,dict(kind='decision_intent',game=game,choice=data['choices'],decision_id=decision_id,
                    engine=dispatch['engine'],before=state,archive=archive_relative,attempt_sink=sink_relative));data=refresh()
                (path/'attempts').mkdir(exist_ok=True);started=time.perf_counter();transient=None
                try:
                    transient=providers.decide(dispatch['engine'],state,decision_id,origin=row['origin'],outer_started=started,
                        fixed_work=128 if identity['spec']['mode']=='fixed128' else None,attempt_sink=path/sink_relative)
                    answer=transient['answer'];legal=literal.normalize(state)['legal'];action=answer['decision']['chosen']
                    need(action in legal,'Native decision is unusable');transition=literal.transition(state,action)
                    need(answer['transition']['position']==transition['after'],'Chosen native/literal endpoint differs')
                    cpu=core.verify();elapsed=(time.perf_counter()-started)*1000
                    clock=dict(critical_ms=elapsed,requested_ms=25 if identity['spec']['mode']=='timed25' else None,
                        raw_hash_guard_claim_staging_included=True,archival_outside_clock=True,
                        whole_decision_contract_sha256=identity['clock_contract_sha256'])
                    archived=native.archive_decision(transient)
                    body=dict(schema=SCHEMA+'-decision-archive',identity_sha256=sha(path/'identity.json'),decision_id=decision_id,before=state,engine=dispatch['engine'],
                        legal=legal,transition=transition,cpu=cpu,clock=clock,answer=archived)
                    atomic(path/archive_relative,body)
                except BaseException as exc:
                    transient=getattr(exc,'transient',None) or transient
                    if transient is not None:
                        failure=native.archive_attempt(transient);atomic(path/'failed-attempts'/f'{stem}.json',failure)
                    append(path,events,dict(kind='hold',game=game,terminal=True,reason=str(exc),decision_id=decision_id,
                        attempt_sink=sink_relative,failed_archive_retained=transient is not None));raise
                resources.assert_worker_lease(lease_path);saved_decision();saved_check(live_guard)
            report=summary(data,identity);atomic(path/'report.json',report)
            need(identity['sources']==sources(),'Worker sources changed during execution')
            catalog=dict(schema=SCHEMA+'-complete',passed=True,identity_sha256=sha(path/'identity.json'),
                commits=stage.common.pins(sorted((path/'commits').glob('*.json'))),report_sha256=sha(path/'report.json'))
            pause('before_catalog');atomic(path/'catalog.json',catalog);pause('after_catalog')
    return completed(path)
