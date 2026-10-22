"""Prospective queries with unchanged complete trace, durable ownership and clock checks.

Only the new namespace/reservation/source admission and finite workload differ
from the held executor. A saved unknown query is never searched again.
"""
from contextlib import ExitStack
import argparse,base64,hashlib,json,math,os,struct,sys,time
from pathlib import Path
from . import config, prospective_eval_common as s
from .valuepilot_reservations import Writer
ROOT,FOLDER=s.ROOT,s.FOLDER
DEFAULT=FOLDER/'jobs/endgame'
SCHEMA=s.SCHEMA+'-queries'
OPERATIONS=('endgame','arena')
HANDOFF_KILL_HOOKS=('before_handoff_commit','after_handoff_commit')
FIXTURE_BASE=FOLDER/'fixtures/native-recovery'
HOOKS=('before_decision_commit','after_decision_commit','before_ownership_check_commit',
 'after_ownership_check_commit','before_handoff_commit','after_handoff_commit','before_catalog','after_catalog')
need,read,sha,atomic=s.need,s.read,s.sha,s.atomic
SOURCES=s.REQUIRED_SOURCES
def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return s.ref(path)
def _salt(text,value):return hashlib.sha256((text+value).encode()).hexdigest()

def membership():
 from . import prospective_eval_reservations as data
 return data.membership()

def schedule(operation,fixture=False):
 need(operation in OPERATIONS and type(fixture) is bool,'Unknown prospective workload')
 members=membership();rows=[]
 if operation=='endgame':
  need(not fixture,'Endgame has no replacement engineering subset')
  roots=sorted(members['exact'],key=lambda r:_salt('prospective-eval-endgame-20261017/',r['state_key']))
  need(len(roots)==32,'Exactly32 inherited endgame roots required')
  for index,meta in enumerate(roots):
   for mi,mode in enumerate(('timed25','fixed128') if index%2==0 else ('fixed128','timed25')):
    for engine in (('control','candidate') if (index+mi)%2==0 else ('candidate','control')):
     rows.append(dict(kind='decision',engine=engine,mode=mode,meta=meta,cohort='endgame'))
 else:
  roots=sorted(members['arena'],key=lambda r:_salt('prospective-eval-arena-20261017/',r['state_key']))
  need(len(roots)==24,'Exactly24 inherited arena starts required')
  if fixture:return [dict(index=0,kind='game',meta=roots[0],baseline='H_v1',seat=0,profile='candidate')]
  cycle=[('H_v1',0),('H8_relay_CPU',1),('H_v1',1),('H8_relay_CPU',0)]
  for index,meta in enumerate(roots):
   offset=index%4;matches=cycle[offset:]+cycle[:offset]
   for matchup,(baseline,seat) in enumerate(matches):
    for profile in (('control','candidate') if (index+matchup)%2==0 else ('candidate','control')):
     rows.append(dict(kind='game',meta=meta,baseline=baseline,seat=seat,profile=profile))
 return [dict(row,index=i) for i,row in enumerate(rows)]
_schedule=schedule
def schedules():return {op:dict(rows=schedule(op),sha256=config.digest(schedule(op))) for op in OPERATIONS}

def classify(output,operation,fixture=False):
 path=s.path_of(output);need(operation in OPERATIONS and type(fixture) is bool,'Unknown query scope')
 if not fixture:
  need(path==FOLDER/'jobs'/operation,'Unregistered production query path')
  return dict(scope='prospective_evaluation',subset_only=False,hook=None)
 need(path.parent==FIXTURE_BASE and path.name in HANDOFF_KILL_HOOKS and operation=='arena','Unregistered recovery fixture')
 return dict(scope='handoff_recovery',subset_only=False,hook=path.name)

def predecessor(output,operation,fixture=False):
 scope=classify(output,operation,fixture)
 if operation=='endgame':return None
 if scope['scope']=='handoff_recovery':
  if scope['hook']==HANDOFF_KILL_HOOKS[0]:path,op=FOLDER/'jobs/endgame','endgame'
  else:path,op=FIXTURE_BASE/HANDOFF_KILL_HOOKS[0],'arena'
 else:path,op=FIXTURE_BASE/HANDOFF_KILL_HOOKS[1],'arena'
 return dict(path=s.relative(path),operation=op)

def prepare_registry():
 from . import prospective_eval_reservations as data
 return data.registry()

def _candidate(operation,fixture,model):
 from . import guarded_value_pilot_v2_native as native
 expected=ROOT/'data/guarded-value-pilot-v2/training/model.json'
 path=Path(model).resolve() if model else expected
 need(path==expected and sha(path)==s.CANDIDATE_SHA,'Only the exact sealed final400 treatment is permitted')
 binding=native.candidate_binding(path)
 need(binding['provenance']['kind']=='fixed-final-step' and binding['provenance']['optimizer_steps']==400,
      'Candidate is not the fixed final400 export')
 return binding

def _seed_for(path,operation,fixture):
 previous=predecessor(path,operation,fixture)
 if previous is None:
  registry=prepare_registry();old=ROOT/'build/guarded-value-pilot-v2/jobs/guards/ownership-prefix.json'
  prefix=read(old)
  return dict(claims_ref=registry['seed_claims'],generation=prefix['generation'],
   predecessor_held_prefix_ref=s.ref(old),operation='retained_complete_footprint',ownership_passed=True,
   failed_returned_footprint_included=True)
 folder=ROOT/previous['path'];done=completed_job(folder,previous['operation'])
 need(done['report']['state']=='complete' and not done['publication_gap'],'Prospective predecessor is not sealed complete')
 prefix=read(folder/'ownership-prefix.json')
 return dict(catalog_ref=_ref(folder/'catalog.json'),report_ref=_ref(folder/'report.json'),
  ownership_prefix_ref=_ref(folder/'ownership-prefix.json'),claims_ref=prefix['claims_ref'],generation=prefix['generation'],
  operation=previous['operation'],state='complete',ownership_passed=True)

def _identity(path,operation,cache,fixture=False,model=None):
 from . import guarded_value_pilot_v2_native as native, prospective_eval_runtime as resources
 s.admit_plan();scope=classify(path,operation,fixture)
 need(Path(cache).resolve()==(s.DATA/'cache').resolve(),'Prospective cache marker path differs')
 rows=schedule(operation,fixture);registry=prepare_registry();batch=_guard_batch(registry)
 refs={o:_guard_refs(o,registry,batch) for o in sorted({r['meta']['origin'] for r in rows})};batch.verify_current()
 binding=_candidate(operation,fixture,model)
 identity=dict(schema=SCHEMA+'-identity',operation=operation,fixture=fixture,cache=str(Path(cache).resolve()),
  plan_sha256=sha(s.PLAN),schedule=rows,schedule_sha256=config.digest(rows),
  registry=_ref(FOLDER/'reservations/registry.json'),guard_refs_by_origin=refs,binding=binding,
  provider_identity=native.source_identity(binding),seed=_seed_for(path,operation,fixture),
  source_inputs=read(s.PLAN)['source_inputs'],native_inputs=native.input_pins(),rules_digest=config.RULES_DIGEST,
  scope=scope,policy_fixed=True,base_frozen=True,inherited_unknown_background=True,
  operational_v4=resources.operational_binding())
 atomic(path/'identity.json',identity);return identity

def _validate_identity(path,operation):
 from . import guarded_value_pilot_v2_native as native, prospective_eval_runtime as resources
 identity=read(path/'identity.json');plan=s.admit_plan();resources.verify_operational_binding(identity['operational_v4'])
 need(identity['schema']==SCHEMA+'-identity' and identity['operation']==operation and
      identity['plan_sha256']==sha(s.PLAN) and identity['rules_digest']==config.RULES_DIGEST and
      identity['policy_fixed'] is True and identity['base_frozen'] is True,'Prospective identity scope changed')
 need(identity['cache']==str((s.DATA/'cache').resolve()) and identity['source_inputs']==plan['source_inputs'],
      'Query source/cache identity changed')
 need(identity['schedule']==schedule(operation,identity['fixture']) and identity['schedule_sha256']==config.digest(identity['schedule'])
      and identity['scope']==classify(path,operation,identity['fixture']),'Finite query schedule changed')
 registry=prepare_registry();need(identity['registry']==_ref(FOLDER/'reservations/registry.json'),'Prospective registry changed')
 batch=_guard_batch(registry)
 need(identity['guard_refs_by_origin']=={o:_expected_guard_refs(o,registry,batch) for o in
      sorted({r['meta']['origin'] for r in identity['schedule']})},'Guard projection inventory changed')
 batch.verify_current()
 need(identity['binding']==_candidate(operation,identity['fixture'],ROOT/identity['binding']['path']) and
      identity['provider_identity']==native.source_identity(identity['binding']) and identity['native_inputs']==native.input_pins(),
      'Frozen provider/model/source identity changed')
 need(identity['seed']==_seed_for(path,operation,identity['fixture']),'Cumulative complete prefix predecessor changed')
 return identity

def _annex_expected(path,body,loaded,claimed,generation,claim_hash):
 from . import prospective_eval_guard as guard
 owned=base64.b64decode(body['answer']['fast_owned_high'],validate=True)
 proof=dict(guard.derive_annex(owned,loaded,body['origin'],claimed));delta=proof.pop('delta_records')
 need(type(delta) is bytes,'Whole-component annex lacks packed delta')
 after=claimed.merge_delta(delta)
 name='annex/'+body['decision_id'].replace('/','-')+'.bin'
 delta_ref=dict(path=s.relative(path/name),sha256=hashlib.sha256(delta).hexdigest()) if delta else None
 event=dict(kind='component_annex',decision_id=body['decision_id'],origin=body['origin'],generation=generation,
  annex=proof,delta_ref=delta_ref,canonical_before=claimed.sha256,canonical_after=after.sha256,
  previous_claim_hash=claim_hash,post_claim_hash=after.sha256 if delta else claim_hash)
 return event,delta,after

def _accept_annex(path,identity,data,pending,event,after):
 data['claimed']=after;data['claim_hash']=event['post_claim_hash']
 pending['annex_components']=event['annex']['components'];pending['annex_delta_ref']=event['delta_ref']
 data['results'].append(_thin_result(pending));data['last_lifecycle']=pending['body']['answer']['lifecycle']['after']
 if identity['operation']=='arena':
  data['state']=pending['body']['transition']['after'];data['choices']+=1
 data['intent']=None;data['pending']=None

def _guard_spec(origin,registry,batch=None):
 from . import prospective_eval_guard as guard
 return guard.spec(origin,registry,batch)


def _guard_batch(registry):
 from .prospective_eval_guard import AdmissionBatch
 return AdmissionBatch(_guard_spec(registry['origins'][0],registry))


def _guard_refs(origin,registry,batch=None):
 from . import prospective_eval_guard as guard
 spec=_guard_spec(origin,registry,batch);folder=FOLDER/'guard'/config.digest(origin)
 refs=guard.prepare(spec,folder,batch=batch);return refs


def _expected_guard_refs(origin,registry,batch=None):
 from . import prospective_eval_guard as guard
 return guard.expected_refs(origin,registry,batch)


def _events(path,identity):
 events=[];previous=None;ids=sha(path/'identity.json')
 for index,cp in enumerate(sorted((path/'commits').glob('*.json'))):
  c=read(cp);need(cp.name==f'{index:06d}.json' and c['index']==index,'Query commit sequence gap')
  event_path=path/c['event'];expected=dict(index=index,identity_sha256=ids,event=c['event'],
   event_sha256=sha(event_path),previous_sha256=previous)
  need(_same(c,expected) and c['event']==f'events/{index:06d}-{c["event_sha256"]}.json','Query immutable commit binding differs')
  events.append(read(event_path));previous=sha(cp)
 return events,previous


def _append(path,events,event):
 index=len(events);event=json.loads(config.canonical(event))
 payload=(config.canonical(event)+'\n').encode();digest=hashlib.sha256(payload).hexdigest()
 ep=path/'events'/f'{index:06d}-{digest}.json';atomic(ep,event)
 cp=path/'commits'/f'{index:06d}.json';atomic(cp,dict(index=index,identity_sha256=sha(path/'identity.json'),
  event=s.relative(ep).split(s.relative(path)+'/',1)[1],event_sha256=digest,
  previous_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None))
 events.append(event)
 print(config.canonical(dict(event='guarded_value_pilot_v2_query_commit',index=index,kind=event['kind'],cursor=event.get('decision_id'))),flush=True)
 return dict(event_path=str(ep),event_sha256=digest,commit_path=str(cp),commit_sha256=sha(cp),index=index)


def _seed_claims(identity,registry):
 from .prospective_eval_guard import PackedClaims
 ids={o:i+1 for i,o in enumerate(registry['origins'])}
 if identity['seed'] is None:return PackedClaims(b'',ids),0
 ref=identity['seed']['claims_ref'];need(sha(ROOT/ref['path'])==ref['sha256'],'Sealed claims seed changed')
 return PackedClaims((ROOT/ref['path']).read_bytes(),ids),identity['seed']['generation']


def _archive_check(body,identity,loaded,claimed,claim_hash,generation,identity_file_sha256,last_lifecycle):
 from . import guarded_value_pilot_v2_native as native,prospective_eval_guard as guard
 answer=body['answer'];state=body['state'];engine=body['engine'];cursor=body['decision_id']
 need(body['identity_sha256']==identity_file_sha256,
  'Query archive identity differs')
 if body['query_kind'] in ('prediction','features'):native.validate_probe(answer,engine,state,cursor,identity['provider_identity'])
 else:
  native.validate_decision(answer,engine,state,cursor,identity['provider_identity'])
  from . import value_run_arena as literal
  chosen=answer['decision']['chosen']
  need(type(chosen) is int and chosen in literal.normalize(state)['legal'] and
   _same(body['transition'],literal.transition(state,chosen)) and
   _same(answer['transition']['position'],body['transition']['after']),'Saved chosen literal transition differs')
 _clock_check(body,identity)
 reset=answer['both_seat_reset'];need(_same(reset['after'],answer['lifecycle']['before']),
  'Both-seat reset and query controls are not physically continuous')
 need(last_lifecycle is not None and _same(reset['before'],last_lifecycle),
  'Unpublished or recomputed native work changed control generations')
 owned=base64.b64decode(answer['fast_owned_high'],validate=True);violations=base64.b64decode(answer['fast_violations'],validate=True)
 need(not violations and not guard.reference_owned(owned,loaded,body['origin'],claimed),'Whole-component query ownership failed')
 cert=answer['fast_certificate'];expected=guard.stage_reference(owned,loaded,body['origin'],claimed,claim_hash,generation+1,generation)
 for key,value in expected.items():
  if key not in ('canonical_restored_claim_hash','delta_records'):need(_same(cert.get(key),value),'Staged global claim certificate differs: '+key)
 result=dict(passed=True,origin=body['origin'],fast_guard_certificate=cert,
  owned_count=cert['owned_count'],owned_sha256=cert['owned_sha256'],raw_sha256=cert['event_sha256'],
  ownership_passed=True,engineering_exception=False,query_kind=body['query_kind'])
 return result,expected['delta_records']


def _clock_check(body,identity):
 """Bind saved clocks to the scheduled treatment and the declared CPU."""
 from . import guarded_value_pilot_v2_native as native
 if identity['operation']=='arena':mode='timed25'
 else:
  cursor=body['decision_id'];index=int(cursor.rsplit('/',1)[1])
  need(cursor==f'query/{index:06d}' and 0<=index<len(identity['schedule']),'Invalid clock schedule cursor')
  row=identity['schedule'][index];mode=row.get('mode')
  need(body['query_kind']==row['kind'] and body['engine']==row['engine'],'Clock query treatment differs')
 clock=body['clock'];timing=body['answer']['fast_timing'];requested=25 if mode=='timed25' else None
 need(_same(clock['requested_ms'],requested) and _same(timing['requested_outer_ms'],requested),
  'Saved requested whole-clock budget differs from scheduled mode')
 def finite(value):return type(value) in (float,int) and math.isfinite(value) and value>=0
 need(finite(clock['critical_ms']) and finite(timing['outer_elapsed_ms']) and
  clock['critical_ms']>=timing['outer_elapsed_ms'] and clock['raw_hash_guard_staging_included'] is True and
  clock['archive_outside_clock'] is True and timing['archive_outside_clock'] is True,
  'Recorded whole decision clock does not cover captured native work')
 if body['query_kind'] in ('teacher','decision'):
  decision=body['answer']['decision'];window=decision['requested_ms']
  need(type(window) is int and finite(decision['wall_ms']) and timing['outer_elapsed_ms']>=decision['wall_ms'] and
   _same(timing['search_window_ms'],window),'Native clock/window is outside the provider clock')
  if mode=='timed25':
   need(native.POLICY['minimum_search_ms']<=window<=25-native.POLICY['reserve_ms'] and
    _same(timing['policy'],native.POLICY),'Timed search policy or nominal window differs')
  else:
   need(window==0 and _same(timing['policy'],native.TEACHER if body['query_kind']=='teacher' else native.POLICY),
    'Untimed teacher/fixed work received a timed native budget')
 selected=read(s.PLAN)['resource']['native_timed_cpu']
 expected=dict(group=selected['group'],logical=selected['logical'],mask=1<<selected['logical'])
 need(_same(body['cpu'],expected),'Query CPU/affinity receipt differs from the fixed declared P-core')
 return True


def _loaded(identity,origin,registry,batch=None):
 from . import prospective_eval_guard as guard
 return guard.load_pure(identity['guard_refs_by_origin'][origin],_guard_spec(origin,registry,batch),batch=batch)


def _replay(path,identity):
 registry=read(ROOT/identity['registry']['path']);need(sha(ROOT/identity['registry']['path'])==identity['registry']['sha256'],'Origin registry changed')
 claimed,generation=_seed_claims(identity,registry);claim_hash=claimed.sha256
 events,prefix=_events(path,identity);intent=None;pending=None;held=None;results=[];games=[];state=None;suffix=False;choices=0;game=0;restored=None
 loaded_cache={};last_lifecycle=None;epoch_ref=None;batch=_guard_batch(registry)
 for index,event in enumerate(events):
  kind=event['kind']
  need(held is None,'Terminal HOLD cannot be followed by more work')
  if kind=='guard_restore_context':
   need(intent is None and pending is None and event['generation']==generation and event['canonical_claims_sha256']==claimed.sha256,
    'Guard restore context is not the complete durable prefix')
   claim_hash=claimed.sha256;restored=event['origin']
  elif kind=='provider_epoch':
   need(intent is not None and pending is None and event['decision_id']==intent['decision_id'] and
    event['generation']==generation and event['canonical_claims_sha256']==claimed.sha256 and
    _same(event['provider_identity'],identity['provider_identity']),'Provider epoch durable prefix differs')
   life=event['initial_lifecycle']
   need(type(life['created']) is int and life['created']==2 and type(life['destroyed']) is int and life['destroyed']==0 and
    sorted(life['retained_inference_controls'].values())==[1,2],'Cold provider was not physically fresh')
   need(index>0 and event['previous_commit_sha256']==sha(path/'commits'/f'{index-1:06d}.json'),'Provider epoch predecessor differs')
   last_lifecycle=life;epoch_ref=_ref(path/f'events/{index:06d}-{hashlib.sha256((config.canonical(event)+chr(10)).encode()).hexdigest()}.json')
  elif kind=='game_start':
   need(identity['operation']=='arena','Game events outside arena')
   row=identity['schedule'][game];need(state is None and event['game']==game and _same(event['state'],row['meta']['state']),'Game root differs')
   state=event['state'];suffix=False;choices=0
  elif kind=='handoff':
   need(identity['operation']=='arena','Handoff events outside arena')
   need(state is not None and not suffix and sum(map(sum,state['pits']))<=24 and _same(event['state'],state),
    'Invalid or repeated irreversible shared handoff');suffix=True
  elif kind=='query_intent':
   need(intent is None and pending is None and held is None and restored==event['origin'],'New query precedes durable claims/restore')
   if identity['operation']=='arena':
    from . import ownership_protocol as protocol
    row=identity['schedule'][game];dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],suffix)
    expected=dict(decision_id=f'game/{game:03d}/{choices:03d}',query_kind='decision',engine=dispatch['engine'],
     origin=row['meta']['origin'],state=state,mode='timed25')
   else:
    row=identity['schedule'][len(results)]
    expected=dict(decision_id=f'query/{len(results):06d}',query_kind=row['kind'],engine=row['engine'],
     origin=row['meta']['origin'],state=row['meta']['state'],mode=row.get('mode'))
   need(all(_same(event.get(k),v) for k,v in expected.items()),'Query intent differs from its finite predeclared cell')
   stem=expected['decision_id'].replace('/','-')
   need(event['archive']==f'archives/{stem}.json' and event['attempt_sink']==f'attempts/{stem}.bin',
    'Query archive/prefix path differs')
   intent=event
  elif kind=='query_result':
   need(intent is not None and pending is None and event['decision_id']==intent['decision_id'],'Query result has no matching intent')
   ap=path/intent['archive'];need(sha(ap)==event['archive_sha256'],'Query archive changed');body=read(ap)
   need(body['decision_id']==intent['decision_id'] and body['origin']==intent['origin'] and
    _same(body['state'],intent['state']) and body['engine']==intent['engine'] and body['query_kind']==intent['query_kind'],
    'Query envelope and intent differ')
   need(_same(body['provider_epoch_ref'],epoch_ref),'Query belongs to another provider epoch')
   origin=body['origin']
   if origin not in loaded_cache:loaded_cache.clear();loaded_cache[origin]=_loaded(identity,origin,registry,batch)
   check,delta=_archive_check(body,identity,loaded_cache[origin],claimed,claim_hash,generation,sha(path/'identity.json'),last_lifecycle)
   pending=dict(body=body,result=check,delta=delta,archive_ref=_ref(ap),event_index=index)
  elif kind=='ownership_check':
   need(pending is not None and not pending.get('ownership_committed') and event['decision_id']==intent['decision_id'] and _same(event['result'],pending['result']),
    'Durable ownership check differs from complete archive')
   claimed=claimed.merge_delta(pending['delta']);generation+=1;claim_hash=event['result']['fast_guard_certificate']['staged_claim_hash']
   pending['ownership_ref']=_ref(path/f'events/{index:06d}-{hashlib.sha256((config.canonical(event)+chr(10)).encode()).hexdigest()}.json')
   pending['ownership_committed']=True
  elif kind=='component_annex':
   need(pending is not None and pending.get('ownership_committed') and intent is not None,
        'Whole-component annex lacks its committed exact-key stage')
   origin=pending['body']['origin'];loaded=loaded_cache[origin]
   expected,delta,after=_annex_expected(path,pending['body'],loaded,claimed,generation,claim_hash)
   need(event==expected,'Whole-component annex differs from complete immutable component derivation')
   if delta:need((ROOT/event['delta_ref']['path']).read_bytes()==delta,'Whole-component annex packed delta changed')
   pending['annex_ref']=_ref(path/f'events/{index:06d}-{hashlib.sha256((config.canonical(event)+chr(10)).encode()).hexdigest()}.json')
   pending['annex_components']=event['annex']['components'];pending['annex_delta_ref']=event['delta_ref']
   claimed=after;claim_hash=event['post_claim_hash'];results.append(_thin_result(pending))
   if identity['operation']=='arena':
    from . import value_run_arena as literal
    action=pending['body']['answer']['decision']['chosen'];transition=literal.transition(state,action)
    need(transition==pending['body']['transition'],'Arena annex barrier transition differs')
    state=transition['after'];choices+=1
   last_lifecycle=pending['body']['answer']['lifecycle']['after'];pending=None;intent=None
  elif kind=='game_end':
   need(identity['operation']=='arena','Game completion outside arena')
   from . import value_run_arena as literal
   need(state is not None and literal.normalize(state)['terminal'] and suffix and _same(event['final'],state) and event['choices']==choices,
    'Unfinished game cannot become a terminal observation')
   row=identity['schedule'][game];margin=(state['stores'][0]-state['stores'][1])*(1 if row['seat']==0 else -1)
   need(_same(event,dict(kind='game_end',game=game,final=state,choices=choices,origin=row['meta']['origin'],
    profile=row['profile'],baseline=row['baseline'],seat=row['seat'],margin=margin,terminal=True)),
    'Terminal margin/profile/seat differs from its literal scheduled game')
   games.append(event);game+=1;state=None;suffix=False;choices=0
  elif kind=='hold':
   need(held is None,'Duplicate terminal HOLD')
   if event.get('failed_attempt_ref') is not None:
    ref=event['failed_attempt_ref'];fp=ROOT/ref['path']
    need(fp.is_relative_to(path/'failed-attempts') and sha(fp)==ref['sha256'],'Failed attempt archive changed')
    failed=read(fp);witness=intent
    if witness is None and results:
     witness=dict(results[-1]['body']);witness['mode']='timed25' if identity['operation']=='arena' else \
      identity['schedule'][len(results)-1].get('mode')
    need(witness is not None and failed['identity_sha256']==sha(path/'identity.json') and
     failed['decision_id']==witness['decision_id'] and failed['mode']==witness['mode'] and
     failed['engine']==witness['engine'] and failed['query_kind']==witness['query_kind'] and failed['origin']==witness['origin'],
     'Failed attempt is not its durable scheduled query')
    clock=failed['clock'];elapsed=clock['critical_ms'];requested=25 if witness['mode']=='timed25' else None
    need(_same(clock['requested_ms'],requested) and type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0 and
     _same(event['known_critical_ms'],elapsed) and event['known_critical_overshoot'] is (requested==25 and elapsed>25),
     'Failed attempt whole clock differs from its retained HOLD')
   held=event
  else:need(False,'Unknown executor journal event')
 batch.verify_current()
 return dict(events=events,prefix_sha256=prefix,claimed=claimed,generation=generation,claim_hash=claim_hash,guard_batch=batch,
  intent=intent,pending=pending,hold=held,results=results,games=games,state=state,suffix=suffix,choices=choices,game=game,
  registry=registry,last_origin=restored,loaded_cache=loaded_cache,last_lifecycle=last_lifecycle,provider_epoch_ref=epoch_ref)


def _thin_result(result):
 body=dict(result['body']);answer=dict(body['answer']);trace=dict(answer['trace']);trace.pop('data',None)
 answer['trace']=trace;answer.pop('fast_owned_high',None);answer.pop('fast_violations',None);body['answer']=answer
 return {k:(body if k=='body' else v) for k,v in result.items() if k!='delta'}


def _phase_summary(identity,data):
 held=data['hold'];operation=identity['operation'];expected=len(identity['schedule'])
 completed=len(data['games']) if operation=='arena' else len(data['results'])
 complete=not held and data['intent'] is None and data['pending'] is None and completed==expected
 overshoots=sum(r['body']['clock']['critical_ms']>25 for r in data['results'] if r['body']['clock'].get('requested_ms')==25)
 result=dict(schema=SCHEMA+'-'+operation+'-report',operation=operation,state='held' if held else 'complete' if complete else 'checkpoint',
  fixture=identity['fixture'],required_units=expected,completed_units=completed,queries=len(data['results']),
  generation=data['generation'],global_claims=len(data['claimed']),canonical_claims_sha256=data['claimed'].sha256,
  pending_attempts=int(data['intent'] is not None or data['pending'] is not None),hold=held,
  ownership_passed=not held and all(r['result']['passed'] is True for r in data['results']),
  critical_overshoots=overshoots,failed_returned_attempts=int(bool(held and held.get('failed_attempt_ref'))),
  failed_known_critical_overshoots=int(bool(held and held.get('known_critical_overshoot'))),
  policy_fixed=True,base_frozen=True,production_evaluation_admitted=True,
  inherited_unknown_background=True,historically_unseen=False)
 if operation=='arena':result.update(games=len(data['games']),terminal_games=len(data['games']),caps=0,fallbacks=0)
 return result


def _complete_data(identity,data):
 report=_phase_summary(identity,data)
 return report['state'] in ('complete','held')


def _claim_path(path):return path/'claims.bin'


def _write_bytes(path,raw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():need(path.read_bytes()==raw,'Immutable binary evidence changed');return
 pending=path.with_name(path.name+'.pending')
 with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 os.replace(pending,path)


def _finalize(path,identity,data,pause=lambda point:None):
 report=_phase_summary(identity,data);need(report['state'] in ('complete','held'),'Cannot seal partial operation')
 _write_bytes(_claim_path(path),data['claimed'].records)
 atomic(path/'ownership-prefix.json',_prefix_document(path,identity,data,report))
 atomic(path/'report.json',report)
 catalog=dict(schema=SCHEMA+'-complete',identity_ref=_ref(path/'identity.json'),report_ref=_ref(path/'report.json'),
  ownership_prefix_ref=_ref(path/'ownership-prefix.json'),commits=s.common.pins(sorted((path/'commits').glob('*.json'))),fit_manifest_ref=None)
 pause('before_catalog');atomic(path/'catalog.json',catalog);pause('after_catalog')
 return dict(report=report,catalog=catalog,fit_manifest=None)


def _prefix_document(path,identity,data,report):
 return dict(schema=SCHEMA+'-ownership-prefix',generation=data['generation'],claim_hash=data['claim_hash'],
  canonical_claims_sha256=data['claimed'].sha256,claims_ref=_ref(_claim_path(path)),
  commit_prefix_sha256=sha(path/'commits'/f'{len(data['events'])-1:06d}.json') if data['events'] else None,
  ownership_passed=report['ownership_passed'],hold=report['hold'],registry=identity['registry'])


def _completed_job_uncached(output=DEFAULT,operation='endgame'):
 path=s.path_of(output);identity=_validate_identity(path,operation);data=_replay(path,identity)
 _check_orphans(path,data)
 need(_complete_data(identity,data),'Executor prefix is not settled')
 expected=_phase_summary(identity,data);need(_same(read(path/'report.json'),expected),'Executor report differs from actual journal')
 prefix=read(path/'ownership-prefix.json');need(_same(prefix,_prefix_document(path,identity,data,expected)) and
  _claim_path(path).read_bytes()==data['claimed'].records,
  'Settled global claims differ from the actual journal')
 catalog=dict(schema=SCHEMA+'-complete',identity_ref=_ref(path/'identity.json'),report_ref=_ref(path/'report.json'),
  ownership_prefix_ref=_ref(path/'ownership-prefix.json'),commits=s.common.pins(sorted((path/'commits').glob('*.json'))),
  fit_manifest_ref=_ref(path/'fit-manifest.json') if (path/'fit-manifest.json').exists() else None)
 if (path/'catalog.json').exists():need(_same(read(path/'catalog.json'),catalog),'Executor catalog differs')
 manifest=None
 return dict(report=expected,catalog=catalog,fit_manifest=manifest,publication_gap=not(path/'catalog.json').exists())


def publish_completed(output=DEFAULT,operation='endgame'):
 done=completed_job(output,operation)
 if not done['publication_gap']:return done
 need(done['report']['state']=='complete','A HOLD cannot be repaired as successful completion')
 path=s.path_of(output)
 with Writer(path):
  identity=_validate_identity(path,operation);data=_replay(path,identity);_check_orphans(path,data)
  need(_phase_summary(identity,data)['state']=='complete','Settled prefix changed during metadata recovery')
  _finalize(path,identity,data)
 return completed_job(path,operation)


def _pause(path,point,requested,marker,lease):
 if point!=requested:return
 need(path.is_relative_to(FOLDER/'fixtures') and marker is not None,'Pause outside explicit fixture')
 from .prospective_eval_runtime import process_identity
 atomic(s.path_of(marker),dict(point=point,process=process_identity(),argv=list(sys.argv),
  source_sha256=sha(__file__),identity_sha256=sha(path/'identity.json'),lease=str(lease),
  query_entry_reached=point!='before_query'))
 while True:time.sleep(.05)


def _adopt(path,identity,data,pause=lambda point:None,live_guard=None,compiled=None):
 intent=data['intent'];need(intent is not None,'No query to adopt')
 ap=path/intent['archive'];need(ap.is_file(),'Unknown unreturned query: HOLD; never retry')
 body=read(ap);need(body['decision_id']==intent['decision_id'] and body['origin']==intent['origin'] and
  body['engine']==intent['engine'] and body['query_kind']==intent['query_kind'] and _same(body['state'],intent['state']),
  'Orphan archive differs from durable intent')
 loaded=compiled if compiled is not None else _loaded(identity,body['origin'],data['registry'],data.get('guard_batch'))
 if data['pending'] is None:
  need(_same(body['provider_epoch_ref'],data['provider_epoch_ref']),'Orphan archive belongs to another provider epoch')
  result,delta=_archive_check(body,identity,loaded,data['claimed'],data['claim_hash'],data['generation'],sha(path/'identity.json'),data['last_lifecycle'])
  pause('before_decision_commit')
  _append(path,data['events'],dict(kind='query_result',decision_id=intent['decision_id'],archive_sha256=sha(ap)))
  pause('after_decision_commit')
  data['pending']=dict(body=body,result=result,delta=delta,archive_ref=_ref(ap))
 pending=data['pending']
 if not pending.get('ownership_committed'):
  pause('before_ownership_check_commit')
  proof=_append(path,data['events'],dict(kind='ownership_check',decision_id=intent['decision_id'],result=pending['result']))
  pause('after_ownership_check_commit')
  if live_guard is not None:live_guard.publish_stage(pending['result']['fast_guard_certificate']['generation'],body['origin'],proof)
  pending['ownership_ref']=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
  data['claimed']=data['claimed'].merge_delta(pending['delta']);data['generation']+=1
  data['claim_hash']=pending['result']['fast_guard_certificate']['staged_claim_hash'];pending['ownership_committed']=True
 # No score, played-state acceptance, handoff or next query precedes this
 # deterministic complete-component barrier, including its empty case.
 event,delta,after=_annex_expected(path,body,loaded,data['claimed'],data['generation'],data['claim_hash'])
 if delta:_write_bytes(ROOT/event['delta_ref']['path'],delta)
 proof=_append(path,data['events'],event)
 pending['annex_ref']=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
 if delta and live_guard is not None:live_guard.restore(after,data['generation'])
 _accept_annex(path,identity,data,pending,event,after)
 return body



def _check_orphans(path,data):
 referenced={r['archive_ref']['path'] for r in data['results']}
 if data['intent'] is not None:referenced.add(s.relative(path/data['intent']['archive']))
 for p in (path/'archives').glob('*.json'):need(s.relative(p) in referenced,'Unreferenced query archive forbids replay')
 sinks={s.relative(path/e['attempt_sink']) for e in data['events'] if e['kind']=='query_intent'}
 for p in (path/'attempts').glob('*.bin'):need(s.relative(p) in sinks,'Unreferenced raw attempt forbids replay')
 committed={read(p)['event'] for p in (path/'commits').glob('*.json')}
 extras=[p for p in (path/'events').glob('*.json') if p.relative_to(path).as_posix() not in committed]
 need(len(extras)<=1 and all(p.name.startswith(f'{len(data["events"]):06d}-') for p in extras),'Unreferenced journal events beyond adoptable prefix')

 annex_refs={r['annex_delta_ref']['path'] for r in data['results'] if r.get('annex_delta_ref') is not None}
 if data['intent'] is not None and data['pending'] is not None and data['pending'].get('ownership_committed'):
  annex_refs.add(s.relative(path/'annex'/(data['intent']['decision_id'].replace('/','-')+'.bin')))
 for p in (path/'annex').glob('*.bin'):need(s.relative(p) in annex_refs,'Unreferenced whole-component annex delta forbids replay')


def run(output=DEFAULT,cache=None,lease=None,operation='endgame',fixture=False,model=None,pause_at=None,pause_marker=None):
 if has_completion(output,operation):return completed_job(output,operation)
 from . import prospective_eval_runtime as resources
 need(lease is not None,'A contained cumulative resource lease is required')
 resources.assert_worker_lease(lease);s.admit_plan();path=s.path_of(output)
 need(pause_at is None or fixture and pause_at in (*HOOKS,'before_query'),'Unadmitted pause')
 path.mkdir(parents=True,exist_ok=True)
 with Writer(path):
  identity=_validate_identity(path,operation) if (path/'identity.json').exists() else _identity(path,operation,cache,fixture,model)
  data=_replay(path,identity);_check_orphans(path,data);need(data['hold'] is None,'Retained campaign HOLD')
  pause=lambda point:_pause(path,point,pause_at,pause_marker,lease)
  if data['intent'] is not None:
   if not(path/data['intent']['archive']).is_file():
    _append(path,data['events'],dict(kind='hold',reason='Unknown unreturned query; no retry',decision_id=data['intent']['decision_id']))
    data['hold']=data['events'][-1];return _finalize(path,identity,data,pause)
   _adopt(path,identity,data,pause)
  if _complete_data(identity,data):return _finalize(path,identity,data,pause)
  # All saved work and ownership checks are resolved before loading providers.
  from . import guarded_value_pilot_v2_native as native,prospective_eval_guard as guard,value_run_arena as literal,ownership_protocol as protocol
  from .timing_v2 import PerformanceCore
  capacity=2000000
  with ExitStack() as stack:
   core=stack.enter_context(PerformanceCore(read(s.PLAN)['resource']['native_timed_cpu']))
   providers=None;live_guard=None;loaded=None;active_origin=None
   try:
    while True:
     if resources.remaining_ns(lease)<30_000_000_000:
      return dict(report=_phase_summary(identity,data),checkpoint=True)
     resources.assert_worker_lease(lease)
     if operation=='arena':
      if data['game']==len(identity['schedule']):break
      row=identity['schedule'][data['game']];meta=row['meta']
      if data['state'] is None:
       _append(path,data['events'],dict(kind='game_start',game=data['game'],state=meta['state']))
       data['state']=meta['state'];data['choices']=0;data['suffix']=False
      state=data['state'];normal=literal.normalize(state)
      if protocol.active(state)<=24 and not data['suffix']:
       pause('before_handoff_commit');_append(path,data['events'],dict(kind='handoff',game=data['game'],state=state));pause('after_handoff_commit');data['suffix']=True
      if normal['terminal']:
       margin=(state['stores'][0]-state['stores'][1])*(1 if row['seat']==0 else -1)
       event=dict(kind='game_end',game=data['game'],final=state,choices=data['choices'],origin=meta['origin'],
        profile=row['profile'],baseline=row['baseline'],seat=row['seat'],margin=margin,terminal=True)
       _append(path,data['events'],event);data['games'].append(event);data['game']+=1;data['state']=None;continue
      if data['choices']>=512:
       data['hold']=dict(kind='hold',reason='Predeclared512 choice cap',game=data['game']);_append(path,data['events'],data['hold']);break
      dispatch=protocol.dispatch(state,row['profile'],row['baseline'],row['seat'],data['suffix'])
      kind='decision';engine=dispatch['engine'];mode='timed25';cursor=f'game/{data["game"]:03d}/{data["choices"]:03d}'
     else:
      index=len(data['results'])
      if index==len(identity['schedule']):break
      row=identity['schedule'][index];meta=row['meta'];state=meta['state'];kind=row['kind'];engine=row['engine'];mode=row.get('mode');cursor=f'query/{index:06d}'
     origin=meta['origin']
     if active_origin!=origin:
      if live_guard is not None:live_guard.close();live_guard=None
      loaded=_loaded(identity,origin,data['registry'],data['guard_batch'])
      _append(path,data['events'],dict(kind='guard_restore_context',origin=origin,generation=data['generation'],canonical_claims_sha256=data['claimed'].sha256))
      data['claim_hash']=data['claimed'].sha256;data['last_origin']=origin
      if providers is not None:
       live_guard=providers.attach_guard(loaded);live_guard.restore(data['claimed'],data['generation'])
       live_guard.bind_run(_ref_absolute(path/'identity.json'),len(data['events'])-1)
      active_origin=origin
     stem=cursor.replace('/','-');intent=dict(kind='query_intent',decision_id=cursor,query_kind=kind,engine=engine,origin=origin,state=state,mode=mode,
      archive=f'archives/{stem}.json',attempt_sink=f'attempts/{stem}.bin')
     _append(path,data['events'],intent);data['intent']=intent
     pause('before_query')
     if providers is None:
      providers=native.open_providers(stack,capacity=capacity,binding=identity['binding'],policy=native.POLICY)
      live_guard=providers.attach_guard(loaded);live_guard.restore(data['claimed'],data['generation'])
      live_guard.bind_run(_ref_absolute(path/'identity.json'),len(data['events'])-1)
      life=providers.lifecycle();need(life['created']==2 and life['destroyed']==0 and
       sorted(life['retained_inference_controls'].values())==[1,2],'Provider constructor performed unexpected work')
      proof=_append(path,data['events'],dict(kind='provider_epoch',decision_id=cursor,provider_identity=identity['provider_identity'],
       initial_lifecycle=life,generation=data['generation'],canonical_claims_sha256=data['claimed'].sha256,
       previous_commit_sha256=sha(path/'commits'/f'{len(data["events"])-1:06d}.json')))
      data['last_lifecycle']=life;data['provider_epoch_ref']=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
     (path/'attempts').mkdir(exist_ok=True);transient=None;started=time.perf_counter();critical=None;cpu=None
     try:
      need(kind in ('features','prediction','decision'),'No new teacher-query kind is admitted')
      if kind in ('features','prediction'):transient=providers.probe(engine,state,cursor,origin,kind,path/intent['attempt_sink'])
      else:transient=providers.decide(engine,state,cursor,origin=origin,outer_started=started,
       fixed_work=128 if mode=='fixed128' else None,attempt_sink=path/intent['attempt_sink'])
      answer=transient['answer'];transition=None
      if kind in ('teacher','decision'):
       chosen=answer['decision']['chosen'];need(type(chosen) is int and chosen in literal.normalize(state)['legal'],'Query produced no usable recommendation')
       transition=literal.transition(state,chosen);need(_same(answer['transition']['position'],transition['after']),'Native and literal chosen endpoint differ')
      cpu=core.verify();critical=(time.perf_counter()-started)*1000
      archived=native.archive_decision(transient)
      body=dict(schema=SCHEMA+'-archive',identity_sha256=sha(path/'identity.json'),decision_id=cursor,query_kind=kind,engine=engine,
       origin=origin,state=state,transition=transition,cpu=cpu,answer=archived,provider_epoch_ref=data['provider_epoch_ref'],
       clock=dict(critical_ms=critical,requested_ms=25 if mode=='timed25' else None,raw_hash_guard_staging_included=True,archive_outside_clock=True))
      atomic(path/intent['archive'],body)
      _adopt(path,identity,data,pause,live_guard,loaded)
     except BaseException as exc:
      failure_elapsed=(time.perf_counter()-started)*1000
      transient=getattr(exc,'transient',None) or transient
      failed_ref=None
      if transient is not None:
       failure=native.archive_attempt(transient)
       failed=dict(schema=SCHEMA+'-failed-attempt',decision_id=cursor,engine=engine,query_kind=kind,mode=mode,origin=origin,
        identity_sha256=sha(path/'identity.json'),answer=failure,
        clock=dict(requested_ms=25 if mode=='timed25' else None,critical_ms=critical if critical is not None else failure_elapsed,
         failed_elapsed_ms=failure_elapsed,phase='post_clock_validation' if critical is not None else 'engine_or_clock',cpu_verified=cpu is not None,cpu=cpu))
       fp=path/'failed-attempts'/f'{stem}.json';atomic(fp,failed);failed_ref=_ref(fp)
      known_elapsed=critical if critical is not None else failure_elapsed
      data['hold']=dict(kind='hold',reason=type(exc).__name__+': '+str(exc),decision_id=cursor,
       failed_attempt_ref=failed_ref,failed_attempt_retained=transient is not None,
       known_critical_ms=known_elapsed,known_critical_overshoot=mode=='timed25' and known_elapsed>25)
      _append(path,data['events'],data['hold']);return _finalize(path,identity,data,pause)
    return _finalize(path,identity,data,pause)
   finally:
    if live_guard is not None:live_guard.close()


def _ref_absolute(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))



def preflight_job(output=DEFAULT,operation='endgame',cache=None,fixture=False):
 path=s.path_of(output);s.admit_plan();classify(path,operation,fixture)
 if (path/'identity.json').exists():
  identity=_validate_identity(path,operation);need(identity['fixture'] is fixture,'Fixture mode changed')
  data=_replay(path,identity);need(data['hold'] is None,'Retained terminal query HOLD forbids another lease')
  _check_orphans(path,data)
  if data['intent'] is not None:need((path/data['intent']['archive']).is_file(),'Unknown unreturned query: HOLD; no retry')
  return dict(state='ready',operation=operation)
 need(not path.exists() or not any(path.rglob('*')),'Unbound query output contains evidence')
 _candidate(operation,fixture,None);_seed_for(path,operation,fixture)
 return dict(state='ready',operation=operation)

def completed_job(output=DEFAULT,operation='endgame'):
 return _completed_job_uncached(output,operation)

def has_completion(output=DEFAULT,operation='endgame'):
 path=s.path_of(output)
 return (path/'report.json').exists() and read(path/'report.json')['state'] in ('complete','held')

def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','completed','status','publish'))
 parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--cache',default=str(s.DATA/'cache'))
 parser.add_argument('--lease');parser.add_argument('--operation',choices=OPERATIONS,default='endgame')
 parser.add_argument('--engineering-fixture',action='store_true');parser.add_argument('--model')
 parser.add_argument('--pause-at');parser.add_argument('--pause-marker');args=parser.parse_args(argv)
 if args.command=='run':
  from . import prospective_eval_runtime as resources
  need(args.lease is not None,'Contained prospective native worker lease absent')
  resources.bind_worker(args.lease)
  with s.worker_views():
   answer=run(args.output,args.cache,args.lease,args.operation,args.engineering_fixture,args.model,args.pause_at,args.pause_marker)
  resources.worker_complete(args.lease,dict(state=answer['report']['state'],operation=args.operation,
      source_views_closed_before_completion=True))
 elif args.command=='completed':answer=completed_job(args.output,args.operation)
 elif args.command=='publish':answer=publish_completed(args.output,args.operation)
 else:
  path=s.path_of(args.output)
  answer=dict(state='not_started') if not(path/'identity.json').exists() else _phase_summary(_validate_identity(path,args.operation),_replay(path,_validate_identity(path,args.operation)))
 print(config.canonical(answer));return answer

if __name__=='__main__':main()
