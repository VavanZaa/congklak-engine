"""Recoverable prospective queries; durable evidence precedes every new query.

Every operation extends a separate immutable journal from the sealed labels
prefix. Completed readers perform no engine construction, inference or search.
"""
from contextlib import ExitStack
from collections import Counter
import argparse,base64,hashlib,json,math,os,struct,sys,time
from pathlib import Path
from . import config,ordinary_epoch_common as s
from .valuepilot_reservations import Writer

ROOT,FOLDER=s.ROOT,s.FOLDER
DEFAULT=FOLDER/'jobs/labels'
SCHEMA=s.SCHEMA+'-queries'
SELECTION=ROOT/'build/ownership-implementation-v1/reservations/production-v2/selection.json'
EXACT=ROOT/'build/ordinary-value-benchmark-projection.json'
OPERATIONS=('labels','conformance','guards','arena')
HOOKS=('before_decision_commit','after_decision_commit','before_ownership_check_commit',
 'after_ownership_check_commit','before_handoff_commit','after_handoff_commit','before_catalog','after_catalog')
SOURCES=('congklak/ordinary_epoch_execute.py','tools/ordinary-epoch-execute.py',
 'congklak/ordinary_epoch_query_contract.py',
 'congklak/ordinary_epoch_common.py','congklak/ordinary_epoch_guard.py',
 'congklak/ordinary_epoch_native.py','congklak/value_run_arena.py','congklak/ownership_protocol.py','congklak/timing_v2.py')
need,read,sha,atomic=s.need,s.read,s.sha,s.atomic


def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return dict(path=s.relative(path),sha256=sha(path))
def _salt(text,value):return hashlib.sha256((text+value).encode()).hexdigest()


def membership():
 from .ordinary_epoch_query_contract import membership as derive
 return derive()


def _schedule(operation,fixture=False):
 from .ordinary_epoch_query_contract import schedule
 return schedule(operation,fixture)


def schedules():
 return {op:dict(rows=_schedule(op),sha256=config.digest(_schedule(op))) for op in OPERATIONS}


def prepare_registry():
 """Pure source-bound static lookup; no native model or target operation."""
 from .ordinary_epoch_query_contract import derive_registry
 s.admit_plan(False);registry,graph=derive_registry()
 atomic(FOLDER/'guard/known-graph.json',graph)
 atomic(FOLDER/'guard/registry.json',registry)
 return registry


def _guard_spec(origin,registry,batch=None):
 inputs=dict(batch.inputs) if batch is not None else s.common.pins([s.PLAN,s.JOINT,FOLDER/'guard/registry.json',ROOT/registry['graph']['path']])
 return dict(guard=dict(graph_path=registry['graph']['path'],allowed_components=[registry['origin_components'][origin]],
  forbidden_quotients=[],epoch_owned_component_only=True,owner_origin=origin),
  schedule=[dict(origin=name) for name in registry['origins']],inputs=inputs)


def _guard_batch(registry):
 from .ordinary_epoch_guard import AdmissionBatch
 return AdmissionBatch(_guard_spec(registry['origins'][0],registry))


def _guard_refs(origin,registry,batch=None):
 from . import ordinary_epoch_guard as guard
 spec=_guard_spec(origin,registry,batch);folder=FOLDER/'guard'/config.digest(origin)
 refs=guard.prepare(spec,folder,batch=batch);return refs

def _expected_guard_refs(origin,registry,batch=None):
 from . import ordinary_epoch_guard as guard
 folder=FOLDER/'guard'/config.digest(origin);manifest=read(folder/'manifest.json')
 refs=dict(manifest_path=s.relative(folder/'manifest.json'),manifest_sha256=sha(folder/'manifest.json'),
  records_path=s.relative(folder/'records.bin'),records_sha256=sha(folder/'records.bin'),guard_stamp=manifest['guard_stamp'])
 guard.load_pure(refs,_guard_spec(origin,registry,batch),batch=batch);return refs


def _candidate(operation,fixture,model):
 from . import ordinary_epoch_native as native
 if operation=='labels':return native.candidate_binding()
 expected=s.DATA/('initial-v2/model.json' if fixture and operation=='conformance' else 'training-v2/model.json')
 path=Path(model).resolve() if model else expected
 need(path==expected.resolve(),'Candidate model path is not the prospectively fixed treatment')
 from .ordinary_epoch_sidecar import verify_export
 export=verify_export(path);provenance=export['model']['provenance']
 if fixture and operation=='conformance':
  need(provenance['kind']=='stepzero' and type(provenance['optimizer_steps']) is int and provenance['optimizer_steps']==0,
   'Initial conformance requires the exact stepzero export')
 else:
  from . import ordinary_epoch_training as training
  training.completed(path.parent)
  need(provenance['kind']=='fixed-final-step' and provenance['optimizer_steps']==400 and
   provenance['engineering_fixture'] is False,'Development executor requires the one final400 candidate')
 return native.candidate_binding(path)


def _identity(path,operation,cache,fixture=False,model=None):
 from .ordinary_epoch_query_contract import classify
 s.admit_plan();need(operation in OPERATIONS and type(fixture) is bool,'Unknown executor scope');scope=classify(path,operation,fixture)
 schedule=_schedule(operation,fixture);registry=prepare_registry()
 origins=sorted({r['meta']['origin'] for r in schedule});batch=_guard_batch(registry)
 refs={o:_guard_refs(o,registry,batch) for o in origins};batch.verify_current()
 from . import ordinary_epoch_native as native
 binding=_candidate(operation,fixture,model);provider=native.source_identity(binding)
 seed=_seed_for(path,operation,fixture)
 identity=dict(schema=SCHEMA+'-identity',operation=operation,fixture=fixture,cache=str(Path(cache).resolve()),
  plan_sha256=sha(s.PLAN),schedule=schedule,schedule_sha256=config.digest(schedule),registry=_ref(FOLDER/'guard/registry.json'),
  guard_refs_by_origin=refs,binding=binding,provider_identity=provider,seed=seed,
  source_inputs=s.common.pins(SOURCES),native_inputs=native.input_pins(),rules_digest=config.RULES_DIGEST,
  scope=scope,policy_fixed=True,base_frozen=True,inherited_unknown_background=True)
 atomic(path/'identity.json',identity);return identity


def _validate_identity(path,operation):
 from .ordinary_epoch_query_contract import derive_registry,classify
 identity=read(path/'identity.json');s.admit_plan()
 need(identity['schema']==SCHEMA+'-identity' and identity['operation']==operation and
  identity['plan_sha256']==sha(s.PLAN) and identity['rules_digest']==config.RULES_DIGEST and
  identity['policy_fixed'] is True and identity['base_frozen'] is True,'Query identity scope changed')
 need(_same(identity['schedule'],_schedule(operation,identity['fixture'])) and
  identity['schedule_sha256']==config.digest(identity['schedule']),'Prospective query order changed')
 need(type(identity['fixture']) is bool and _same(identity['scope'],classify(path,operation,identity['fixture'])),
  'Query scope/path/fixture contract differs')
 registry,graph=derive_registry()
 need(_same(read(ROOT/identity['registry']['path']),registry) and
  _same(read(ROOT/registry['graph']['path']),graph),'Origin registry or whole graph differs from accepted derivation')
 batch=_guard_batch(registry)
 need(_same(identity['guard_refs_by_origin'],{o:_expected_guard_refs(o,registry,batch) for o in
  sorted({r['meta']['origin'] for r in identity['schedule']})}),'Guard reference inventory differs')
 batch.verify_current()
 need(_same(identity['source_inputs'],s.common.pins(SOURCES)),'Executor source inventory was altered')
 s.common.current(identity['source_inputs']);s.common.current(identity['native_inputs'])
 from . import ordinary_epoch_native as native
 need(_same(identity['provider_identity'],native.source_identity(identity['binding'])),'Provider/model source scope changed')
 need(_same(identity['binding'],_candidate(operation,identity['fixture'],ROOT/identity['binding']['path'])),
  'Executor candidate provenance/path differs')
 need(_same(identity['native_inputs'],native.input_pins()),'Native source inventory was altered')
 need(_same(identity['seed'],_seed_for(path,operation,identity['fixture'])),'Cumulative operation predecessor changed')
 return identity


def _seed_for(path,operation,fixture):
 from . import ordinary_epoch_query_contract as contract
 predecessor=contract.predecessor(path,operation,fixture)
 if predecessor is None:return None
 previous=ROOT/predecessor['path'];done=completed_job(previous,predecessor['operation'])
 need(done['report']['state']=='complete' and not done['publication_gap'],'Campaign predecessor is not sealed and complete')
 prefix=read(previous/'ownership-prefix.json')
 return dict(catalog_ref=_ref(previous/'catalog.json'),report_ref=_ref(previous/'report.json'),
  ownership_prefix_ref=_ref(previous/'ownership-prefix.json'),claims_ref=prefix['claims_ref'],generation=prefix['generation'],
  operation=predecessor['operation'],state='complete',ownership_passed=done['report']['ownership_passed'])


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
 index=len(events);event=s.normalized(event)
 payload=(config.canonical(event)+'\n').encode();digest=hashlib.sha256(payload).hexdigest()
 ep=path/'events'/f'{index:06d}-{digest}.json';atomic(ep,event)
 cp=path/'commits'/f'{index:06d}.json';atomic(cp,dict(index=index,identity_sha256=sha(path/'identity.json'),
  event=s.relative(ep).split(s.relative(path)+'/',1)[1],event_sha256=digest,
  previous_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None))
 events.append(event)
 print(config.canonical(dict(event='ordinary_epoch_query_commit',index=index,kind=event['kind'],cursor=event.get('decision_id'))),flush=True)
 return dict(event_path=str(ep),event_sha256=digest,commit_path=str(cp),commit_sha256=sha(cp),index=index)


def _seed_claims(identity,registry):
 from .ordinary_epoch_guard import PackedClaims
 ids={o:i+1 for i,o in enumerate(registry['origins'])}
 if identity['seed'] is None:return PackedClaims(b'',ids),0
 ref=identity['seed']['claims_ref'];need(sha(ROOT/ref['path'])==ref['sha256'],'Sealed claims seed changed')
 return PackedClaims((ROOT/ref['path']).read_bytes(),ids),identity['seed']['generation']


def _archive_check(body,identity,loaded,claimed,claim_hash,generation,identity_file_sha256,last_lifecycle):
 from . import ordinary_epoch_native as native,ordinary_epoch_guard as guard
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
 if identity['fixture'] and identity['operation']=='labels':
  need(expected['stage_new_count']==0,'Recovery fixture discovered new owned states; campaign HOLD')
 for key,value in expected.items():
  if key not in ('canonical_restored_claim_hash','delta_records'):need(_same(cert.get(key),value),'Staged global claim certificate differs: '+key)
 result=dict(passed=True,origin=body['origin'],fast_guard_certificate=cert,
  owned_count=cert['owned_count'],owned_sha256=cert['owned_sha256'],raw_sha256=cert['event_sha256'],
  ownership_passed=True,engineering_exception=False,query_kind=body['query_kind'])
 return result,expected['delta_records']


def _clock_check(body,identity):
 """Bind saved clocks to the scheduled treatment and the declared CPU."""
 from . import ordinary_epoch_native as native
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
 from . import ordinary_epoch_guard as guard
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
   need(pending is not None and event['decision_id']==intent['decision_id'] and _same(event['result'],pending['result']),
    'Durable ownership check differs from complete archive')
   claimed=claimed.merge_delta(pending['delta']);generation+=1;claim_hash=event['result']['fast_guard_certificate']['staged_claim_hash']
   pending['ownership_ref']=_ref(path/f'events/{index:06d}-{hashlib.sha256((config.canonical(event)+chr(10)).encode()).hexdigest()}.json')
   results.append(_thin_result(pending))
   if identity['operation']=='arena':
    from . import value_run_arena as literal
    answer=pending['body']['answer'];action=answer['decision']['chosen'];need(type(action) is int,'Unusable arena choice')
    transition=literal.transition(state,action);need(_same(transition,pending['body']['transition']),'Arena literal transition differs')
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


def _labels_rows(identity,data):
 from . import ordinary_epoch_native as native
 rows=[];by_cursor={r['body']['decision_id']:r for r in data['results']}
 for index in range(0,len(identity['schedule']),2):
  a,b=by_cursor[f'query/{index:06d}'],by_cursor[f'query/{index+1:06d}']
  meta=identity['schedule'][index]['meta'];teacher=native.teacher_certificate(a['body']['answer'],meta['state'])
  x=b['body']['answer']['probe']['result']['features'];D=meta['state']['stores'][0]-meta['state']['stores'][1]
  r=sum(map(sum,meta['state']['pits']))*(1 if meta['state']['controller']==0 else -1)
  y=teacher['target_player0'];z=(y-D)/r
  need(len(x)==127 and all(type(v) in (int,float) and math.isfinite(v) for v in x) and -1<=z<=1,
   'Feature/target residual interval differs')
  teacher.update(source_archive_ref=a['archive_ref'],ownership_check_ref=a['ownership_ref'])
  rows.append(dict(state=meta['state'],state_key=meta['state_key'],family=meta['family'],cell=meta['cell'],
   component=meta['component'],joint_component=meta['component'],stratum=meta.get('actual_stratum',meta['cell'].rsplit('/actor',1)[0]),
   actor=meta['state']['controller'],x=[struct.unpack('<f',struct.pack('<f',v))[0] for v in x],x_native=x,
   D=D,r=r,target_player0=y,target_z=z,is_case=meta['state_key'] in membership()['cases'],teacher=teacher,
   feature_source=dict(source_archive_ref=b['archive_ref'],ownership_check_ref=b['ownership_ref'])))
 return rows


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
  policy_fixed=True,base_frozen=True,production_evaluation_admitted=False,
  inherited_unknown_background=True,historically_unseen=False)
 if operation=='labels':
  result.update(roots=len(data['results'])//2,complete_h8_roots=0,fit_cases=32 if not identity['fixture'] else 0)
  if complete:
   rows=_labels_rows(identity,data);result.update(roots=len(rows),complete_h8_roots=len(rows),features_complete=True,
    all_legal_Q_complete=True,horizon=8,support192_disjoint=True)
 elif operation=='arena':
  result.update(games=len(data['games']),terminal_games=len(data['games']),caps=0,fallbacks=0)
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
 prefix=_prefix_document(path,identity,data,report)
 atomic(path/'ownership-prefix.json',prefix)
 if identity['operation']=='labels' and report['state']=='complete':
  rows=_labels_rows(identity,data);atomic(path/'fit-rows.json',rows)
  fit_manifest=_fit_manifest_document(path,rows)
  atomic(path/'fit-manifest.json',fit_manifest)
 atomic(path/'report.json',report)
 catalog=dict(schema=SCHEMA+'-complete',identity_ref=_ref(path/'identity.json'),report_ref=_ref(path/'report.json'),
  ownership_prefix_ref=_ref(path/'ownership-prefix.json'),commits=s.common.pins(sorted((path/'commits').glob('*.json'))),
  fit_manifest_ref=_ref(path/'fit-manifest.json') if (path/'fit-manifest.json').exists() else None)
 pause('before_catalog');atomic(path/'catalog.json',catalog);pause('after_catalog')
 return dict(report=report,catalog=catalog,fit_manifest=read(path/'fit-manifest.json') if (path/'fit-manifest.json').exists() else None)


def _prefix_document(path,identity,data,report):
 return dict(schema=SCHEMA+'-ownership-prefix',generation=data['generation'],claim_hash=data['claim_hash'],
  canonical_claims_sha256=data['claimed'].sha256,claims_ref=_ref(_claim_path(path)),
  commit_prefix_sha256=sha(path/'commits'/f'{len(data['events'])-1:06d}.json') if data['events'] else None,
  ownership_passed=report['ownership_passed'],hold=report['hold'],registry=identity['registry'])


def _fit_manifest_document(path,rows):
 return dict(schema=SCHEMA+'-fit-manifest',rows_ref=_ref(path/'fit-rows.json'),roots=len(rows),
  case_state_keys=sorted(r['state_key'] for r in rows if r['is_case']),cases=sum(r['is_case'] for r in rows),
  ownership_prefix_ref=_ref(path/'ownership-prefix.json'),identity_ref=_ref(path/'identity.json'),
  primary_precision='captured native binary64 features; inherited optimizer support usesFP32',
  targets='H_v1 complete choice-depth8 full legal rootQ player0',gradients_from_these_rows=False)


def completed_job(output=DEFAULT,operation='labels'):
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
 if operation=='labels' and expected['state']=='complete':
  rows=_labels_rows(identity,data);manifest=read(path/'fit-manifest.json')
  need(_same(manifest,_fit_manifest_document(path,rows)) and _same(read(path/'fit-rows.json'),rows),
   'Sealed fit rows/membership differ')
 return dict(report=expected,catalog=catalog,fit_manifest=manifest,publication_gap=not(path/'catalog.json').exists())


def has_completion(output=DEFAULT,operation='labels'):
 path=s.path_of(output)
 if not (path/'identity.json').exists():return False
 identity=_validate_identity(path,operation);data=_replay(path,identity)
 _check_orphans(path,data)
 return _complete_data(identity,data) and (path/'report.json').exists() and (path/'ownership-prefix.json').exists()


def preflight_job(output=DEFAULT,operation='labels',cache=None,fixture=False):
 from .ordinary_epoch_query_contract import classify
 path=s.path_of(output);s.admit_plan();need(operation in OPERATIONS,'Unknown operation')
 classify(path,operation,fixture)
 if (path/'identity.json').exists():
  identity=_validate_identity(path,operation);need(identity['fixture'] is fixture,'Executor fixture mode changed')
  data=_replay(path,identity);need(data['hold'] is None,'Existing terminal query HOLD forbids another lease')
  _check_orphans(path,data)
  if data['intent'] is not None:
   need((path/data['intent']['archive']).exists(),'Unknown unreturned query: HOLD; no retry')
  return dict(state='complete' if has_completion(path,operation) else 'ready',operation=operation)
 need(not path.exists() or not any(path.rglob('*')),'Unbound executor output contains evidence; refusing overwrite')
 _candidate(operation,fixture,None)
 _seed_for(path,operation,fixture)
 return dict(state='ready',operation=operation)


def publish_completed(output=DEFAULT,operation='labels'):
 done=completed_job(output,operation)
 if not done['publication_gap']:return done
 need(done['report']['state']=='complete','A HOLD cannot be repaired as successful completion')
 path=s.path_of(output)
 with Writer(path):
  identity=_validate_identity(path,operation);data=_replay(path,identity);_check_orphans(path,data)
  need(_phase_summary(identity,data)['state']=='complete','Settled prefix changed during metadata recovery')
  _finalize(path,identity,data)
 return completed_job(path,operation)


def completed_labels(path=DEFAULT):return completed_job(path,'labels')
def admit_labels(path=DEFAULT):
 path=s.path_of(path);done=completed_labels(path);r=done['report']
 need(r['state']=='complete' and r['fixture'] is False and r['roots']==r['complete_h8_roots']==128 and
  r['fit_cases']==32 and r['all_legal_Q_complete'] is True and r['features_complete'] is True and
  r['ownership_passed'] is True and r['support192_disjoint'] is True and r['pending_attempts']==0 and
  r['hold'] is None and not done['publication_gap'],'The required sealed128 complete labels are not admitted')
 p=read(path/'ownership-prefix.json')
 return dict(schema=s.SCHEMA+'-label-admission',required_roots=128,completed_roots=128,fit_cases=32,horizon=8,
  all_legal_Q_complete=True,features_complete=True,ownership_passed=True,support192_disjoint=True,
  pending_attempts=0,hold=False,catalog_ref=_ref(path/'catalog.json'),ownership_prefix_ref=_ref(path/'ownership-prefix.json'),
  fit_manifest_ref=_ref(path/'fit-manifest.json'),claims_ref=p['claims_ref'],generation=p['generation'])
def load_fit(path=DEFAULT):
 done=completed_labels(path);need(done['report']['state']=='complete','Fit labels incomplete')
 return read(ROOT/done['fit_manifest']['rows_ref']['path'])


def _pause(path,point,requested,marker,lease):
 if point!=requested:return
 need(path.is_relative_to(FOLDER/'fixtures') and marker is not None,'Pause outside explicit fixture')
 from .ordinary_epoch_resources import process_identity
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
 if data['pending'] is None:
  origin=body['origin'];loaded=compiled if compiled is not None else _loaded(identity,origin,data['registry'],data.get('guard_batch'))
  need(_same(body['provider_epoch_ref'],data['provider_epoch_ref']),'Orphan archive belongs to another provider epoch')
  result,delta=_archive_check(body,identity,loaded,data['claimed'],data['claim_hash'],data['generation'],sha(path/'identity.json'),data['last_lifecycle'])
  pause('before_decision_commit')
  _append(path,data['events'],dict(kind='query_result',decision_id=intent['decision_id'],archive_sha256=sha(ap)))
  pause('after_decision_commit')
  data['pending']=dict(body=body,result=result,delta=delta,archive_ref=_ref(ap))
 pending=data['pending'];pause('before_ownership_check_commit')
 proof=_append(path,data['events'],dict(kind='ownership_check',decision_id=intent['decision_id'],result=pending['result']))
 pause('after_ownership_check_commit')
 if live_guard is not None:live_guard.publish_stage(pending['result']['fast_guard_certificate']['generation'],body['origin'],proof)
 pending['ownership_ref']=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
 data['claimed']=data['claimed'].merge_delta(pending['delta']);data['generation']+=1
 data['claim_hash']=pending['result']['fast_guard_certificate']['staged_claim_hash'];data['results'].append(_thin_result(pending))
 data['last_lifecycle']=body['answer']['lifecycle']['after']
 if identity['operation']=='arena':
  data['state']=body['transition']['after'];data['choices']+=1
 data['intent']=None;data['pending']=None
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


def run(output=DEFAULT,cache=None,lease=None,operation='labels',fixture=False,model=None,pause_at=None,pause_marker=None):
 if has_completion(output,operation):return completed_job(output,operation)
 from . import ordinary_epoch_resources as resources
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
  from . import ordinary_epoch_native as native,ordinary_epoch_guard as guard,value_run_arena as literal,ownership_protocol as protocol
  from .timing_v2 import PerformanceCore
  capacity=16000000 if operation=='labels' else 2000000
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
      if kind=='teacher':transient=providers.teacher(state,cursor,origin,path/intent['attempt_sink'])
      elif kind in ('features','prediction'):transient=providers.probe(engine,state,cursor,origin,kind,path/intent['attempt_sink'])
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
      if kind=='teacher':native.teacher_certificate(archived,state)
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


def _closed_data(path,operation):
 completed_job(path,operation);identity=_validate_identity(Path(path),operation)
 return identity,_replay(Path(path),identity)


def _fixed_view(answer):
 report=answer['decision']['report']
 keys=('chosen','visits','policy','root_priors','estimated_value_player0','action_mean_value_player0',
  'completed_simulations','nodes','root_expanded','unvisited_root_q_player0','stop_reason')
 return {key:report[key] for key in keys}


def _conformance_metrics(identity,data):
 from . import ordinary_epoch_fit as arithmetic
 control=read(ROOT/'data/quality-train-v1/model.json')['model']['weights']
 candidate=read(ROOT/identity['binding']['path'])['model']['branch_weights']
 fit={r['state_key']:r for r in load_fit()};feature_exact=True;value_exact=True;policy_exact=True;count=0
 fixed={}
 for item in data['results']:
  body=item['body'];index=int(body['decision_id'].rsplit('/',1)[1]);row=identity['schedule'][index];meta=row['meta']
  x=fit[meta['state_key']]['x_native'];answer=body['answer']
  if body['query_kind']=='features':feature_exact &= _same(answer['probe']['result']['features'],x);continue
  if body['query_kind']=='decision':fixed[body['engine']]=_fixed_view(answer);continue
  value=answer['probe']['result'];u,policy=arithmetic._base(x,control)
  correction=arithmetic._correction(x,candidate) if body['engine']=='candidate' else 0.
  reference=fit[meta['state_key']]['D']+fit[meta['state_key']]['r']*math.tanh(u+correction)
  value_exact &= abs(value['value_player0']-reference)<=1e-9
  policy_exact &= all(abs(a-b)<=1e-12 for a,b in zip(value['policy_logits'],policy));count+=1
 equivalent=not identity['fixture'] or set(fixed)=={'control','candidate'} and _same(fixed['control'],fixed['candidate'])
 return dict(passed=feature_exact and value_exact and policy_exact and equivalent,policy_fixed=policy_exact,
  features_exact=feature_exact,control_candidate_predictions_exact=value_exact and policy_exact,
  fixed128_Q_exact=equivalent,fixed128_visits_exact=equivalent,rows=count,
  ownership_passed=_phase_summary(identity,data)['ownership_passed'],
  critical_overshoots=_phase_summary(identity,data)['critical_overshoots'])


def _guard_metrics(identity,data):
 fit={r['state_key']:r for r in load_fit()};case_rows=[];end_rows=[];fixed={};branch=0
 for entry in data['results']:
  body=entry['body'];index=int(body['decision_id'].rsplit('/',1)[1]);row=identity['schedule'][index]
  meta=row['meta'];answer=body['answer'];chosen=answer['decision']['chosen'];actor=meta['state']['controller']
  target=fit[meta['state_key']]['teacher'] if row['cohort']=='cases' else meta['reference']
  action_q=target.get('action_q',target.get('action_scores'));q={r['pit']:r['value_player0'] for r in action_q}
  best=(max if actor==0 else min)(q.values());regret=(best-q[chosen])*(1 if actor==0 else -1)
  r=dict(state_key=meta['state_key'],cell=meta['cell'],engine=body['engine'],mode=row['mode'],regret=regret,optimal=regret==0)
  if row['cohort']=='cases':case_rows.append(r)
  else:
   end_rows.append(r)
   if body['engine']=='candidate':
    delta=answer['evaluator']['delta'];need(type(delta['branch_calls']) is int and delta['branch_calls']==0 and
     type(delta['branch_bypasses']) is int and delta['branch_bypasses']==delta['native_actual_evals'],
     'Shared endgame candidate did not physically bypass its changed branch')
    branch+=delta['branch_calls']
   if row['mode']=='fixed128':fixed.setdefault(meta['state_key'],{})[body['engine']]=_fixed_view(answer)
 def average(engine,mode):
  rows=[r['regret'] for r in case_rows if r['engine']==engine and r['mode']==mode and '/capture/' in r['cell']]
  return sum(rows)/len(rows) if rows else None
 summary=_phase_summary(identity,data);case_roots={r['state_key'] for r in case_rows};end_roots={r['state_key'] for r in end_rows}
 case=dict(roots=len(case_roots),ordinary_roots=len({r['state_key'] for r in case_rows if '/ordinary/' in r['cell']}),
  capture_roots=len({r['state_key'] for r in case_rows if '/capture/' in r['cell']}),
  bonus_roots=len({r['state_key'] for r in case_rows if '/bonus/' in r['cell']}),ownership_passed=summary['ownership_passed'],
  critical_overshoots=sum(x['body']['clock']['critical_ms']>25 for x in data['results']
   if x['body']['clock'].get('requested_ms')==25 and identity['schedule'][int(x['body']['decision_id'].rsplit('/',1)[1])]['cohort']=='cases'))
 for engine in ('control','candidate'):
  for mode in ('timed25','fixed128'):case[engine+'_capture_regret_'+mode]=average(engine,mode)
 end=dict(roots=len(end_roots),control_optimal=all(r['optimal'] for r in end_rows if r['engine']=='control'),
  candidate_optimal=all(r['optimal'] for r in end_rows if r['engine']=='candidate'),
  fixed_work_equal=len(fixed)==len(end_roots) and all(set(v)=={'control','candidate'} and _same(v['control'],v['candidate']) for v in fixed.values()),
  learned_branch_calls=branch,ownership_passed=summary['ownership_passed'],critical_overshoots=summary['critical_overshoots']-case['critical_overshoots'])
 return case,end


def admit_initial_conformance():
 path=FOLDER/'jobs/initial-conformance';done=completed_job(path,'conformance');identity,data=_closed_data(path,'conformance')
 metrics=_conformance_metrics(identity,data)
 need(done['report']['state']=='complete' and identity['fixture'] is True and len(data['results'])==5 and
  metrics['passed'] is True and metrics['ownership_passed'] is True and not done['publication_gap'],
  'Stepzero native/search equivalence is not complete')
 return dict(schema=s.SCHEMA+'-initial-conformance-admission',
  features_exact=metrics['features_exact'],control_candidate_predictions_exact=metrics['control_candidate_predictions_exact'],
  fixed128_Q_exact=metrics['fixed128_Q_exact'],fixed128_visits_exact=metrics['fixed128_visits_exact'],ownership_passed=True,
  pending_attempts=0,hold=False,model_sha256=identity['binding']['sha256'],initial_model_ref=_ref(ROOT/identity['binding']['path']),
  catalog_ref=_ref(path/'catalog.json'),ownership_prefix_ref=_ref(path/'ownership-prefix.json'))


def arena_game_rows():
 identity,data=_closed_data(FOLDER/'jobs/arena','arena')
 return [{k:r[k] for k in ('origin','profile','baseline','seat','margin','terminal')} for r in data['games']]


def summary_metrics():
 labels=completed_labels()['report'];ci,cd=_closed_data(FOLDER/'jobs/conformance','conformance')
 gi,gd=_closed_data(FOLDER/'jobs/guards','guards');ai,ad=_closed_data(FOLDER/'jobs/arena','arena')
 cases,endgame=_guard_metrics(gi,gd)
 arena={k:_phase_summary(ai,ad)[k] for k in ('games','terminal_games','caps','fallbacks','ownership_passed','critical_overshoots')}
 from .ordinary_epoch_query_contract import FIXTURE_BASE,HANDOFF_KILL_HOOKS
 for hook in HANDOFF_KILL_HOOKS:
  fi,fd=_closed_data(FIXTURE_BASE/hook,'arena');fm=_phase_summary(fi,fd)
  arena['ownership_passed'] &= fm['ownership_passed'];arena['critical_overshoots']+=fm['critical_overshoots']
 return dict(labels={k:labels[k] for k in ('roots','complete_h8_roots','ownership_passed')},
  cases=cases,conformance=_conformance_metrics(ci,cd),endgame=endgame,
  arena=arena)


def result_inputs():
 inputs={}
 for folder in ('labels','initial-conformance','conformance','guards','arena'):
  path=FOLDER/'jobs'/folder;operation='conformance' if folder=='initial-conformance' else folder
  completed_job(path,operation)
  inputs.update(s.common.pins(p for p in path.rglob('*') if p.is_file() and p.name!='writer.lock' and not p.name.endswith('.pending')))
 from .ordinary_epoch_query_contract import FIXTURE_BASE
 inputs.update(s.common.pins(p for p in FIXTURE_BASE.rglob('*') if p.is_file() and p.name!='writer.lock' and not p.name.endswith('.pending')))
 return inputs


def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','completed','status','publish'))
 parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--cache',default=str(s.DATA/'cache-v2'))
 parser.add_argument('--lease');parser.add_argument('--operation',choices=OPERATIONS,default='labels')
 parser.add_argument('--engineering-fixture',action='store_true');parser.add_argument('--model')
 parser.add_argument('--pause-at');parser.add_argument('--pause-marker');args=parser.parse_args(argv)
 if args.command=='run':
  if has_completion(args.output,args.operation):answer=publish_completed(args.output,args.operation)
  else:
   from . import ordinary_epoch_resources as resources
   need(args.lease is not None,'Contained native worker lease absent');resources.bind_worker(args.lease)
   answer=run(args.output,args.cache,args.lease,args.operation,args.engineering_fixture,args.model,args.pause_at,args.pause_marker)
   resources.worker_complete(args.lease,dict(state=answer['report']['state'],operation=args.operation))
 elif args.command=='completed':answer=completed_job(args.output,args.operation)
 elif args.command=='publish':
  answer=publish_completed(args.output,args.operation)
 else:
  path=s.path_of(args.output)
  if not(path/'identity.json').exists():answer=dict(state='not_started')
  else:
   identity=_validate_identity(path,args.operation);answer=_phase_summary(identity,_replay(path,identity))
 print(config.canonical(answer));return answer


if __name__=='__main__':main()
