"""Recoverable traced handoffs, with ownership checks committed before more search.

Only consumed-state engineering schedules are admitted in this implementation
stage. A future production treatment must supply independently admitted joint
reservations and a frozen schedule; these fixtures make no strength claim.
"""
from contextlib import ExitStack
from functools import lru_cache
import math
import os
from pathlib import Path
import time
from . import config,ownership_implementation_common as stage,ownership_protocol as protocol
from . import value_run_arena as literal,neurodiag_data as metadata
from .valuepilot_reservations import Writer

ROOT,FOLDER=stage.ROOT,stage.FOLDER
DEFAULT=FOLDER/'handoff/engineering-fixed'
SPEC=FOLDER/'handoff/engineering-spec.json'
SCHEMA='live-traced-handoff-v1'
SOURCES=('congklak/ownership_handoff_worker.py','congklak/ownership_handoff_resources.py',
    'congklak/ownership_implementation_common.py','congklak/ownership_protocol.py',
    'congklak/provenance_reader_v2.py','tools/run-ownership-handoff.py','tests/reference.py',
    'congklak/value_run_arena.py','congklak/timing_v2.py')
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic
_FOOTPRINT_CACHE={}

@lru_cache(maxsize=2)
def _graph(path,expected):
    need(sha(ROOT/path)==expected,'Ownership graph source changed')
    return read(ROOT/path)

def sources():
    from . import native_ownership_trace as trace
    return dict(files=stage.common.pins(SOURCES),native=trace.source_identity())

def prepare(path=DEFAULT,spec_path=SPEC):
    path=stage.path_of(path,'handoff');spec_path=stage.path_of(spec_path,'handoff')
    spec=read(spec_path)
    need(spec['schema']==SCHEMA+'-engineering-spec' and spec['engineering_fixture'] is True and
         spec['production_games']==0 and spec['mode'] in ('fixed128','timed25') and
         spec['policy_fixed'] is True and spec['models_fixed'] is True,'Unadmitted handoff schedule')
    stage.common.current(spec['inputs'])
    need(1<=len(spec['schedule'])<=10,'Engineering schedule bound differs')
    ref=literal._literal()
    for i,row in enumerate(spec['schedule']):
        need(row['index']==i and row['profile'] in protocol.PROFILES and row['baseline'] in protocol.BASELINES and
             row['seat'] in (0,1) and row['origin'] and row['source'] in spec['inputs'],'Invalid engineering schedule unit')
        n=literal.normalize(row['start'],ref)
        need(n['state']==row['start'],'Engineering start must be normalized')
    identity=dict(schema=SCHEMA+'-identity',spec_path=spec_path.relative_to(ROOT).as_posix(),
        spec_sha256=sha(spec_path),spec=spec,sources=sources(),training_counters=stage.TRAINING_ZERO,
        rules_digest=config.RULES_DIGEST,models_fixed=True,policy_fixed=True,
        cpu_declared=read(ROOT/'build/value-plan.json')['cpu_declared'])
    atomic(path/'identity.json',identity)
    return stage.normalized(identity)

def _events(path,identity):
    out=[];previous=None;ids=sha(path/'identity.json')
    for i,p in enumerate(sorted((path/'commits').glob('*.json'))):
        need(p.name==f'{i:06d}.json','Handoff commit gap')
        commit=read(p)
        need(commit['index']==i and commit['identity_sha256']==ids and commit['previous_sha256']==previous,
             'Handoff commit identity differs')
        rel=f'events/{i:06d}-{commit["event_sha256"]}.json'
        need(commit['event']==rel and sha(path/rel)==commit['event_sha256'],'Handoff event missing/corrupt')
        out.append(read(path/rel));previous=sha(p)
    return out

def append(path,identity,events,event):
    event=stage.normalized(event);i=len(events)
    digest=__import__('hashlib').sha256((config.canonical(event)+'\n').encode()).hexdigest()
    rel=f'events/{i:06d}-{digest}.json';atomic(path/rel,event)
    commit=dict(index=i,identity_sha256=sha(path/'identity.json'),event=rel,event_sha256=digest,
        previous_sha256=sha(path/'commits'/f'{i-1:06d}.json') if i else None)
    atomic(path/'commits'/f'{i:06d}.json',commit);events.append(event)
    print(config.canonical(dict(event='handoff_event_committed',index=i,kind=event['kind'],game=event.get('game'))),flush=True)
    return event

def footprint(answer):
    from . import native_ownership_trace as trace
    key=config.digest(answer['trace'])
    if key not in _FOOTPRINT_CACHE:
        if len(_FOOTPRINT_CACHE)>=512:_FOOTPRINT_CACHE.clear()
        _FOOTPRINT_CACHE[key]=trace.compile_footprint(answer['trace'])
    return _FOOTPRINT_CACHE[key]

def check_ownership(answer,origin,spec,claimed):
    """Check high footprints conservatively; low rows retain exact raw trace."""
    fp=footprint(answer);guard=spec['guard'];violations=[]
    # The engineering exception is explicit and cannot admit a production run.
    graph=_graph(guard['graph_path'],spec['inputs'][guard['graph_path']])
    lookup=graph['quotient_components'];allowed=set(guard['allowed_components'])
    need(guard['engineering_consumed_only'] is True,'Fresh arena admission is not implemented by fixture exemption')
    for row in fp['rows']:
        if not row['owned']:continue
        q=row['quotient'];cid=lookup.get(q)
        if q in guard['forbidden_quotients']:
            violations.append(dict(quotient=q,reason='explicit protected quotient',first_event=row['first_event']))
        if cid is not None and cid not in allowed:
            violations.append(dict(quotient=q,component=cid,reason='unpermitted historical component',first_event=row['first_event']))
        if q in claimed and claimed[q]!=origin:
            violations.append(dict(quotient=q,reason='different live origin',other_origin=claimed[q],first_event=row['first_event']))
    return dict(schema=SCHEMA+'-ownership-check',origin=origin,trace_sha256=config.digest(answer['trace']),
        footprint=fp,violations=violations,passed=not violations,engineering_exception=True,
        production_admission=False)

def replay(path,identity):
    """Read only: literal choices, strict native envelopes and durable cursor."""
    from . import native_ownership_trace as trace
    events=_events(path,identity);spec=identity['spec'];ref=literal._literal();game=0;state=None;suffix=False
    pending=None;choices=0;claimed={};games=[];decisions=[];handoffs=[];hold=None
    for event in events:
        kind=event['kind'];need(event['game']==game and game<len(spec['schedule']),'Handoff game order differs')
        row=spec['schedule'][game]
        if kind=='game_start':
            need(state is None and pending is None and event['start']==row['start'],'Handoff start differs')
            state=row['start'];suffix=False;choices=0
        elif kind=='handoff':
            need(state is not None and pending is None and not suffix and protocol.active(state)<=24 and
                 event['state']==state and event['terminal']==literal.normalize(state,ref)['terminal'] and
                 event['reset_fields']==list(protocol.RESET_FIELDS),'Handoff boundary cursor differs')
            suffix=True;handoffs.append(event)
        elif kind=='decision':
            need(state is not None and pending is None and not literal.normalize(state,ref)['terminal'] and
                 event['before']==state and event['choice']==choices and event['ownership_pending'] is True,
                 'Handoff decision cursor differs')
            dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],suffix)
            need(not dispatch['handoff_required'] and event['engine']==dispatch['engine'],'Handoff dispatch differs')
            trace.validate_decision(event['answer'],event['engine'],state,event['decision_id'],identity=identity['sources']['native'])
            action=event['answer']['decision']['chosen'];expected=literal.transition(state,action,ref)
            need(event['transition']==expected and event['legal']==literal.normalize(state,ref)['legal'] and
                 event['answer']['transition']['position']==expected['after'],'Literal handoff decision differs')
            clock=event['clock'];timed=spec['mode']=='timed25'
            need(all(type(clock[k]) in (int,float) and math.isfinite(clock[k]) and clock[k]>=0
                 for k in ('outer_ms','validation_ms','pre_search_ms','overshoot_ms')) and
                 clock['mode']==spec['mode'] and clock['outer_ms']>=event['answer']['decision']['wall_ms'] and
                 clock['outer_ms']>=clock['validation_ms']+clock['pre_search_ms'] and
                 clock['requested_ms']==(25 if timed else None) and
                 clock['overshoot_ms']==(max(0.,clock['outer_ms']-25) if timed else 0.) and
                 clock['native_grant_ms']==event['answer']['decision']['requested_ms']==
                     (math.floor(25-clock['pre_search_ms']) if timed else 0) and
                 (not timed or clock['native_grant_ms']>0) and clock['positive_search_grant'] is True and
                 clock['trace_hash_validation_and_guard_included'] is True,
                 'Handoff whole-clock evidence differs')
            att=event['both_seat_reset'];need(att['actual_controls_replaced']==2 and
                att['fields']==list(protocol.RESET_FIELDS) and att['reason']=='every_decision' and
                att['decision_id']==event['decision_id'] and
                att['source_identity_sha256']==config.digest(identity['sources']['native']) and
                att['after']['created']-att['before']['created']==2 and
                att['after']['destroyed']-att['before']['destroyed']==2 and
                all(att['after']['retained_inference_controls'][p]>att['before']['retained_inference_controls'][p]
                    for p in ('control','candidate')),
                'Both-seat decision reset missing')
            pending=event;decisions.append(event)
        elif kind=='ownership_check':
            need(pending is not None and event['decision_id']==pending['decision_id'],'Ownership check has no pending decision')
            expected=check_ownership(pending['answer'],row['origin'],spec,claimed)
            need(event['result']==expected,'Ownership check semantics differ')
            if expected['passed']:
                for q in expected['footprint']['rows']:
                    if q['owned']:claimed[q['quotient']]=row['origin']
                state=pending['transition']['after'];choices+=1;pending=None
            else:hold=expected
        elif kind=='hold':
            need(event.get('reason') and event.get('terminal') is True,'Handoff HOLD event malformed');hold=event
        elif kind=='game_end':
            need(state is not None and pending is None and literal.normalize(state,ref)['terminal'] and suffix and
                 event['final']==state and event['choices']==choices and event['terminal_margin']==state['stores'][0]-state['stores'][1],
                 'Handoff terminal completion differs')
            games.append(event);game+=1;state=None;suffix=False;choices=0
        else:need(False,'Unknown handoff event')
        if hold:need(event is events[-1],'Handoff continued after HOLD')
    return dict(events=events,game=game,state=state,suffix=suffix,pending=pending,choices=choices,
        claimed=claimed,games=games,decisions=decisions,handoffs=handoffs,hold=hold)

def completed(path=DEFAULT):
    path=stage.path_of(path,'handoff');cat=read(path/'catalog.json');identity=read(path/'identity.json')
    need(identity['schema']==SCHEMA+'-identity' and identity['models_fixed'] is True and identity['policy_fixed'] is True and
         identity['training_counters']==stage.TRAINING_ZERO and identity['rules_digest']==config.RULES_DIGEST,
         'Handoff completed identity differs')
    stage.common.current(identity['spec']['inputs']);need(sha(ROOT/identity['spec_path'])==identity['spec_sha256'],
         'Handoff source schedule differs')
    need(read(ROOT/identity['spec_path'])==identity['spec'] and identity['sources']==sources(),'Handoff source identity changed')
    data=replay(path,identity)
    need(cat['schema']==SCHEMA+'-complete' and cat['passed'] is True and cat['identity_sha256']==sha(path/'identity.json') and
         cat['commits']==stage.common.pins(sorted((path/'commits').glob('*.json'))) and
         data['game']==len(identity['spec']['schedule']) and data['pending'] is None and not data['hold'],
         'Handoff completion differs')
    report=read(path/'report.json');expected=summary(data,identity)
    need(cat['report_sha256']==sha(path/'report.json') and report==expected,'Handoff completion report differs')
    return dict(catalog=cat,report=report)

def summary(data,identity):
    timed=identity['spec']['mode']=='timed25';ds=data['decisions']
    return stage.normalized(dict(schema=SCHEMA+'-report',passed=True,engineering_fixture=True,production_admission=False,
        games=len(data['games']),decisions=len(ds),handoffs=len(data['handoffs']),
        terminal_boundary_events=sum(h['terminal'] for h in data['handoffs']),pending_ownership_checks=0,
        internal_trace_events=sum(d['answer']['trace']['event_count'] for d in ds),
        engineering_searches=len(ds),engineering_games=len(data['games']),training_counters=stage.TRAINING_ZERO,
        mode=identity['spec']['mode'],whole_clock_ms=25 if timed else None,
        overshoot_decisions=sum(d['clock']['overshoot_ms']>0 for d in ds) if timed else 0,
        maximum_outer_ms=max((d['clock']['outer_ms'] for d in ds),default=0),
        logical_results=[dict(game=d['game'],choice=d['choice'],engine=d['engine'],before=d['before'],
            action=d['answer']['decision']['chosen'],after=d['transition']['after'],work=d['answer']['work_budget'],
            trace_raw_sha256=d['answer']['trace']['event_sha256']) for d in ds],
        final_states=[g['final'] for g in data['games']],confidence_intervals_enabled=False,
        strength_interpretation_allowed=False,models_fixed=True,policy_fixed=True))

def pause(path,point,requested,marker):
    if point!=requested:return
    marker=stage.path_of(marker,'handoff/fixtures')
    atomic(marker,dict(point=point,pid=os.getpid()))
    while True:time.sleep(.05)

def run(path=DEFAULT,spec_path=SPEC,lease=None,pause_at=None,pause_marker=None):
    path=stage.path_of(path,'handoff')
    if (path/'catalog.json').exists():return completed(path)
    need(pause_at in (None,'before_handoff_commit','after_handoff_commit','before_decision_commit',
         'after_decision_commit','before_catalog'),'Unknown handoff pause')
    need(pause_at is None or path.is_relative_to(FOLDER/'handoff/fixtures'),'Pause hooks require fixture namespace')
    from . import ownership_handoff_resources as resources
    need(lease is not None,'Live worker requires contained cumulative resource lease')
    resources.assert_worker_lease(lease)
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return completed(path)
        identity=prepare(path,spec_path);data=replay(path,identity);events=data['events']
        need(not data['hold'],'Retained ownership HOLD forbids further searches')
        # Complete the saved trace check before loading native models/providers.
        def flush_pending():
            nonlocal data
            pending=data['pending']
            if pending is not None:
                row=identity['spec']['schedule'][data['game']]
                result=check_ownership(pending['answer'],row['origin'],identity['spec'],data['claimed'])
                append(path,identity,events,dict(kind='ownership_check',game=data['game'],decision_id=pending['decision_id'],result=result))
                data=replay(path,identity);need(not data['hold'],'Ownership conflict: retained run HOLD')
        flush_pending()
        from . import native_ownership_trace as trace
        from .timing_v2 import PerformanceCore
        with ExitStack() as stack:
            core=stack.enter_context(PerformanceCore(identity['cpu_declared']));providers=None
            while data['game']<len(identity['spec']['schedule']):
                resources.assert_worker_lease(lease);core.verify()
                row=identity['spec']['schedule'][data['game']];game=data['game']
                if data['state'] is None:
                    append(path,identity,events,dict(kind='game_start',game=game,start=row['start']))
                    data=replay(path,identity)
                state=data['state'];normalized=literal.normalize(state)
                if protocol.active(state)<=24 and not data['suffix']:
                    pause(path,'before_handoff_commit',pause_at,pause_marker)
                    append(path,identity,events,dict(kind='handoff',game=game,state=state,terminal=normalized['terminal'],
                        reset_fields=list(protocol.RESET_FIELDS),reset_before_next_decision=True))
                    pause(path,'after_handoff_commit',pause_at,pause_marker)
                    data=replay(path,identity)
                if normalized['terminal']:
                    append(path,identity,events,dict(kind='game_end',game=game,final=state,choices=data['choices'],
                        terminal_margin=state['stores'][0]-state['stores'][1]))
                    data=replay(path,identity);continue
                if data['choices']>=512:
                    append(path,identity,events,dict(kind='hold',game=game,terminal=True,reason='Engineering game choice cap'))
                    need(False,'Engineering game choice cap: HOLD')
                if providers is None:providers=trace.open_providers(stack)
                decision_id=f'{game:03d}/{data["choices"]:03d}'
                dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],data['suffix'])
                started=time.perf_counter();reset=providers.reset_all(reason='every_decision',decision_id=decision_id)
                timed=identity['spec']['mode']=='timed25'
                pre_search=(time.perf_counter()-started)*1000
                grant=math.floor(25-pre_search) if timed else 0
                need(not timed or grant>0,'No positive remaining whole-decision grant: HOLD')
                try:
                    answer=(providers.timed_decision(dispatch['engine'],state,ms=grant,decision_id=decision_id) if timed else
                            providers.fixed_decision(dispatch['engine'],state,decision_id=decision_id))
                except trace.TraceHold as exc:
                    append(path,identity,events,dict(kind='hold',game=game,terminal=True,reason=str(exc),
                        before=state,engine=dispatch['engine'],decision_id=decision_id,failed_trace=exc.trace))
                    raise
                validation_started=time.perf_counter()
                trace.validate_decision(answer,dispatch['engine'],state,decision_id,identity=identity['sources']['native'])
                legal=literal.normalize(state)['legal'];action=answer['decision']['chosen']
                need(action in legal,'Traced search did not return a usable legal move: HOLD')
                transition=literal.transition(state,action)
                need(answer['transition']['position']==transition['after'],'Native chosen endpoint differs from literal rules: HOLD')
                ownership=check_ownership(answer,row['origin'],identity['spec'],data['claimed'])
                cpu=core.verify();finished=time.perf_counter();elapsed=(finished-started)*1000
                event=dict(kind='decision',game=game,choice=data['choices'],decision_id=decision_id,engine=dispatch['engine'],
                    before=state,legal=legal,answer=answer,transition=transition,ownership_pending=True,both_seat_reset=reset,
                    cpu=cpu,clock=dict(mode=identity['spec']['mode'],outer_ms=elapsed,
                        validation_ms=(finished-validation_started)*1000,pre_search_ms=pre_search,native_grant_ms=grant,
                        requested_ms=25 if timed else None,
                        overshoot_ms=max(0.,elapsed-25) if timed else 0.,positive_search_grant=True,
                        trace_hash_validation_and_guard_included=True,fsync_outside_decision_clock=True))
                pause(path,'before_decision_commit',pause_at,pause_marker)
                append(path,identity,events,event)
                pause(path,'after_decision_commit',pause_at,pause_marker)
                append(path,identity,events,dict(kind='ownership_check',game=game,decision_id=decision_id,result=ownership))
                data=replay(path,identity);need(not data['hold'],'Ownership conflict: retained run HOLD')
            report=summary(data,identity);atomic(path/'report.json',report)
            need(identity['sources']==sources(),'Handoff sources changed during execution')
            cat=dict(schema=SCHEMA+'-complete',passed=True,identity_sha256=sha(path/'identity.json'),
                commits=stage.common.pins(sorted((path/'commits').glob('*.json'))),report_sha256=sha(path/'report.json'))
            pause(path,'before_catalog',pause_at,pause_marker);atomic(path/'catalog.json',cat)
    return completed(path)
