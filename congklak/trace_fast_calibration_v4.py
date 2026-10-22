"""Finite source-bound native conformance/calibration, with every attempt retained."""
from contextlib import ExitStack
import base64,hashlib,math,time
from pathlib import Path
from . import config,tracing_speed_common as stage,tracing_speed_resources as resources
from . import ownership_fast_guard as guards,value_run_arena as literal
from .timing_v2 import PerformanceCore
from .valuepilot_reservations import Writer

ROOT=stage.ROOT
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic
ORIGIN='engineering-consumed-139'
SOURCES=('congklak/trace_fast_calibration_v4.py','congklak/native_trace_fast_v2.py',
 'congklak/trace_fast_adapter_v2.py','congklak/trace_puct_witness_v1.py','congklak/trace_fast_comparison_v1.py','congklak/value_run_arena.py','tests/reference.py','congklak/timing_v2.py')

def identity(spec_path):
 from . import native_trace_fast_v2 as fast,tracing_speed_launcher_v4 as launcher
 spec_path=stage.path_of(spec_path);spec=read(spec_path);launcher.validate_spec(spec_path)
 need(spec['kind']=='calibration' and spec['schema']=='fast-native-bounded-calibration-v1' and
  len(spec['schedule'])==32 and spec['capacity_events']==2000000,'Calibration finite scope differs')
 loaded=guards.load_pure(spec['guard_refs'],spec)
 return stage.normalized(dict(schema='fast-native-calibration-identity-v1',spec_path=spec_path.relative_to(ROOT).as_posix(),
  spec_sha256=sha(spec_path),guard_refs=spec['guard_refs'],sources=stage.common.pins(SOURCES),
  native=fast.source_identity(),cpu_declared=read(ROOT/'build/value-plan.json')['cpu_declared'],
  training_counters=stage.ZERO)),spec,loaded

def prepare(output,spec_path):
 output=stage.path_of(output);bound,_,_=identity(spec_path);atomic(output/'identity.json',bound)
 return dict(prepared=True,identity_sha256=sha(output/'identity.json'))

def commit(output,event):
 index=len(list((output/'commits').glob('*.json')));event=stage.normalized(event)
 digest=hashlib.sha256((config.canonical(event)+'\n').encode()).hexdigest();rel=f'events/{index:06d}-{digest}.json'
 atomic(output/rel,event);path=output/'commits'/f'{index:06d}.json'
 atomic(path,dict(index=index,identity_sha256=sha(output/'identity.json'),event=rel,event_sha256=digest,
  previous_sha256=sha(output/'commits'/f'{index-1:06d}.json') if index else None))
 return dict(index=index,event_path=str(output/rel),event_sha256=digest,commit_path=str(path),commit_sha256=sha(path))

from .trace_fast_comparison_v1 import semantic_decision as _strip_time

def _verify_attempt(saved,scheduled,bound,loaded,claimed,previous_hash,generation):
 from . import native_trace_fast_v2 as fast,native_ownership_trace as old
 need(saved['scheduled']==scheduled and saved['identity']==bound,'Calibration attempt cursor/source differs')
 answer=saved['answer'];cert=answer['fast_certificate'];root=scheduled['state'];decision_id=f'calibration/{scheduled["index"]:06d}'
 need(saved['decision_id']==decision_id and not answer.get('archival_errors'),'Calibration archive incomplete')
 need(answer['trace']['binding']['root_state']==root,'Calibration root binding differs')
 raw=old.trace_bytes(answer['trace']);owned=base64.b64decode(answer['fast_owned_high'],validate=True)
 clock=answer['fast_timing'];cost=saved['critical_ms']
 need(type(cost) in (int,float) and math.isfinite(cost) and cost>=clock['outer_elapsed_ms']>0 and
  clock['archive_outside_clock'] is True and clock['policy']==read(ROOT/bound['spec_path'])['search_policy'] and
  answer['archival_timing']['charged_to_lease'] is True and
  answer['archival_timing']['encoding_only_after_owned_validated_snapshot'] is True and
  answer['engine_attempt_started'] is True,'Calibration critical/archive clock boundary differs')
 if scheduled['mode']=='timed':
  need(clock['requested_outer_ms']==25 and type(clock['search_window_ms']) is int and
   clock['policy']['minimum_search_ms']<=clock['search_window_ms']<=math.floor(25-clock['policy']['reserve_ms']) and
   answer['decision']['requested_ms']==clock['search_window_ms'],'Calibration common useful grant differs')
 need(hashlib.sha256(raw).hexdigest()==cert['event_sha256'],'Calibration complete prefix SHA differs')
 reference=guards.reference_trace(answer['trace'],loaded,ORIGIN,claimed)
 need(reference['owned_records']==owned and reference['passed']==cert['guard_passed'],'Calibration guard/reference differs')
 if saved['error'] is not None:return None
 fast.validate_decision(answer,scheduled['engine'],root,decision_id,bound['native'])
 need(saved['literal_transition']==literal.transition(root,answer['decision']['chosen'],literal._literal()) and
  saved['literal_transition']['after']==answer['transition']['position'],'Calibration chosen literal state differs')
 expected=guards.stage_reference(owned,loaded,ORIGIN,claimed,previous_hash,cert['generation'],generation)
 need(all(cert.get(k)==v for k,v in expected.items() if k not in ('delta_records','canonical_restored_claim_hash')),
  'Calibration staged map/reference differs')
 if scheduled['mode']=='fixed128':
  oracle=read(ROOT/scheduled['oracle']);need(sha(ROOT/scheduled['oracle'])==scheduled['oracle_sha256'],'Saved conformance oracle changed')
  need(raw==old.trace_bytes(oracle['trace']) and _strip_time(answer['decision'])==_strip_time(oracle['decision']),
   'Frozen fixed-work raw event sequence/choice/root Q/PV differs')
 return cert

def _state(output,spec_path,require_all=False):
 bound,spec,loaded=identity(spec_path);need(read(output/'identity.json')==bound,'Calibration source identity changed')
 claimed={};previous_hash=hashlib.sha256(b'').hexdigest();generation=0;rows=[];events=[];previous=None
 for index,path in enumerate(sorted((output/'commits').glob('*.json'))):
  receipt=read(path);event_path=output/receipt['event']
  need(path.name==f'{index:06d}.json' and receipt['index']==index and receipt['identity_sha256']==sha(output/'identity.json') and
   receipt['previous_sha256']==previous and sha(event_path)==receipt['event_sha256'] and
   receipt['event']==f'events/{index:06d}-{receipt["event_sha256"]}.json','Calibration journal chain/source differs')
  event=read(event_path);need(event['kind'] in ('guard_restore_context','ownership_check'),'Unexpected calibration event')
  events.append(event);previous=sha(path)
 cursor=0
 for scheduled in spec['schedule']:
  index=scheduled['index'];path=output/'attempts'/f'{index:06d}.json';intent=output/'intents'/f'{index:06d}.json'
  while cursor<len(events) and events[cursor]['kind']=='guard_restore_context':
   restore=events[cursor];canonical=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest()
   need(restore['committed_attempts']==index and restore['generation']==generation and
    restore['canonical_claim_hash']==canonical and restore['guard_stamp']==loaded.guard_stamp and
    restore['prefix_sha256']==(sha(output/'commits'/f'{cursor-1:06d}.json') if cursor else None),
    'Calibration cold-restore context/hash epoch differs')
   previous_hash=canonical;cursor+=1
  if not path.exists():
   need(not intent.exists(),'Unreturned calibration attempt: HOLD; mapped prefix retained, never retry');break
  saved=read(path);authorization=read(intent)
  need(authorization==dict(scheduled=scheduled,identity_sha256=sha(output/'identity.json'),claims_hash=previous_hash,
   decision_id=f'calibration/{index:06d}',unreturned_attempt='HOLD, never retry') and
   saved['intent_sha256']==sha(intent),'Calibration durable intent source/cursor/claims differs')
  cert=_verify_attempt(saved,scheduled,bound,loaded,claimed,previous_hash,generation)
  need(cursor<len(events) and events[cursor]==dict(kind='ownership_check',attempt=index,
   archive_sha256=sha(path),result=dict(passed=True,origin=ORIGIN,fast_guard_certificate=cert)),
   'Calibration archive lacks exact committed ownership check: HOLD')
  cursor+=1
  rows.append(saved)
  if cert:
   for q,_,_,_ in guards.OWNED.iter_unpack(base64.b64decode(saved['answer']['fast_owned_high'],validate=True)):claimed[q.hex()]=ORIGIN
   generation=cert['generation'];previous_hash=cert['staged_claim_hash']
  else:need(False,'Retained calibration failed attempt: HOLD; no repetition or filtering')
 need(cursor==len(events),'Calibration journal has extra or reordered event')
 if require_all:
  need(len(rows)==32 and len(list((output/'intents').glob('*.json')))==32 and
   len(list((output/'attempts').glob('*.json')))==32 and len(list((output/'raw').glob('*.bin')))==32,
   'Calibration attempt artifact panel incomplete/extra')
 return bound,spec,loaded,claimed,previous_hash,generation,rows

def _summary(rows):
 result={}
 for engine in ('H_v1','H8_relay_CPU','control','candidate'):
  subset=[r for r in rows if r['scheduled']['mode']=='timed' and r['scheduled']['engine']==engine]
  times=sorted(r['critical_ms'] for r in subset)
  result[engine]=dict(attempts=len(subset),median_ms=times[len(times)//2],p95_ms=times[math.ceil(.95*len(times))-1],
   maximum_ms=max(times),overshoots=sum(t>25 for t in times),
   minimum_search_ms=min(r['answer']['fast_timing']['search_window_ms'] for r in subset),
   maximum_events=max(r['answer']['fast_certificate']['event_count'] for r in subset))
 return result

def completed(output):
 output=stage.path_of(output);bound=read(output/'identity.json')
 _,spec,_,_,_,_,rows=_state(output,ROOT/bound['spec_path'],True)
 report=read(output/'report.json');catalog=read(output/'catalog.json')
 need(report['summary']==_summary(rows) and report['search_policy']==spec['search_policy'] and report['policy_frozen'] is True and
  report['fixed_conformance_passed']==8 and report['training_counters']==stage.ZERO and report['models_fixed'] is True and
  report['policy_fixed'] is True and report['strength_claim'] is False and report['no_worst_case_timing_guarantee'] is True and
  catalog['last_commit_sha256']==sha(sorted((output/'commits').glob('*.json'))[-1]) and
  catalog['report_sha256']==sha(output/'report.json') and catalog['identity_sha256']==sha(output/'identity.json'),
  'Calibration report/policy/catalog differs')
 return dict(passed=True,attempts=32,source_bound=True,completed_reader_no_engine=True,summary=report['summary'])

def run(output,spec_path,lease_path,pause_at=None,pause_marker=None):
 from . import native_trace_fast_v2 as fast
 output=stage.path_of(output)
 if (output/'catalog.json').exists():return completed(output)
 need(pause_at is None,'Calibration never injects search interruptions; handoff fixtures own those')
 prepare(output,spec_path)
 with Writer(output):
  bound,spec,loaded,claimed,previous_hash,generation,rows=_state(output,spec_path)
  # Cold resume is explicit; canonical restored hash is a distinct chain epoch.
  previous_hash=hashlib.sha256(loaded.claims_blob(claimed)).hexdigest()
  commit(output,dict(kind='guard_restore_context',generation=generation,canonical_claim_hash=previous_hash,
   guard_stamp=loaded.guard_stamp,committed_attempts=len(rows),
   prefix_sha256=sha(sorted((output/'commits').glob('*.json'))[-1]) if list((output/'commits').glob('*.json')) else None))
  with PerformanceCore(bound['cpu_declared']) as cpu,ExitStack() as stack:
   providers=fast.open_providers(stack,capacity=spec['capacity_events'],policy=spec['search_policy'])
   guard=stack.enter_context(providers.attach_guard(loaded));guard.restore(claimed,generation)
   guard.bind_run(dict(path=str(output/'identity.json'),sha256=sha(output/'identity.json')),
    len(list((output/'commits').glob('*.json')))-1)
   reference=literal._literal()
   for scheduled in spec['schedule'][len(rows):]:
    index=scheduled['index'];resources.assert_worker_lease(lease_path)
    need(resources.namespace_bytes()+2*spec['capacity_events']*24+128<read(lease_path)['protocol']['quota_bytes'],'Calibration raw archive quota: HOLD')
    cursor=f'{index:06d}';decision_id='calibration/'+cursor;intent=output/'intents'/(cursor+'.json')
    atomic(intent,dict(scheduled=scheduled,identity_sha256=sha(output/'identity.json'),claims_hash=previous_hash,
     decision_id=decision_id,unreturned_attempt='HOLD, never retry'))
    (output/'raw').mkdir(exist_ok=True);transient=None;error=None;literal_transition=None;tick=time.perf_counter()
    try:
     transient=providers.decide(scheduled['engine'],scheduled['state'],decision_id,guard=guard,origin=ORIGIN,
      outer_started=tick,deadline=tick+.025,attempt_sink=output/'raw'/(cursor+'.bin'),
      fixed_work=128 if scheduled['mode']=='fixed128' else None)
     literal_transition=literal.transition(scheduled['state'],transient['answer']['decision']['chosen'],reference)
     need(literal_transition['after']==transient['answer']['transition']['position'],'Calibration literal/native mismatch')
     cpu.verify()
    except fast.TraceHold as exc:transient=exc.transient;error=str(exc)
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
     critical_ms=(time.perf_counter()-tick)*1000
     if transient is not None:
      # Archival is always outside the measured decision, inside its resource lease.
      answer=fast.archive_attempt(transient)
      saved=dict(identity=bound,scheduled=scheduled,decision_id=decision_id,intent_sha256=sha(intent),
       answer=answer,error=error,literal_transition=literal_transition,critical_ms=critical_ms)
      atomic(output/'attempts'/(cursor+'.json'),saved)
    need(transient is not None,'Unreturned calibration attempt: HOLD; mapped prefix retained')
    need(error is None,'Retained failed calibration attempt: '+str(error))
    cert=_verify_attempt(saved,scheduled,bound,loaded,claimed,previous_hash,generation)
    proof=commit(output,dict(kind='ownership_check',attempt=index,archive_sha256=sha(output/'attempts'/(cursor+'.json')),
     result=dict(passed=True,origin=ORIGIN,fast_guard_certificate=cert)))
    guard.publish_stage(cert['generation'],ORIGIN,proof)
    for q,_,_,_ in guards.OWNED.iter_unpack(transient['owned_high']):claimed[q.hex()]=ORIGIN
    generation=cert['generation'];previous_hash=cert['staged_claim_hash'];rows.append(saved)
    print(config.canonical(dict(event='calibration_attempt_archived',index=index,engine=scheduled['engine'],
     mode=scheduled['mode'],critical_ms=critical_ms,event_count=cert['event_count'])),flush=True)
  report=dict(schema='fast-native-bounded-calibration-report-v1',summary=_summary(rows),fixed_conformance_passed=8,
   search_policy=spec['search_policy'],policy_frozen=True,policy_is_bounded_hypothesis=True,
   no_worst_case_timing_guarantee=True,models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,strength_claim=False)
  atomic(output/'report.json',report)
  atomic(output/'catalog.json',dict(schema='fast-native-calibration-catalog-v1',identity_sha256=sha(output/'identity.json'),
   report_sha256=sha(output/'report.json'),last_commit_sha256=sha(sorted((output/'commits').glob('*.json'))[-1])))
 return completed(output)
