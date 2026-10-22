"""Finite recoverable acquisition, separate from reservation/model admission."""
from contextlib import ExitStack
import argparse,base64,hashlib,json,os,sys,time
from pathlib import Path
from . import config,footprint_screen_common as s
from .valuepilot_reservations import Writer

DEFAULT=s.FOLDER/'acquisition'
FIXTURE_BASE=s.FOLDER/'fixtures/native-recovery'
HOOKS=('before_query','before_result_commit','after_result_commit','before_trace_commit',
       'after_trace_commit','before_catalog','after_catalog')
SCHEMA=s.SCHEMA+'-worker'
SOURCES=('congklak/footprint_screen_worker.py','congklak/footprint_screen_native.py',
         'tools/footprint-screen-worker.py','congklak/footprint_screen_common.py')
ZERO=hashlib.sha256(b'').hexdigest()
need,read,sha,atomic=s.need,s.read,s.sha,s.atomic


def same(a,b):return config.canonical(a)==config.canonical(b)
def ref(p):return dict(path=s.relative(p),sha256=sha(p))
def scope(output,fixture):
 p=s.path_of(output);need(type(fixture) is bool,'Fixture flag must be boolean')
 if fixture:need(p.parent==FIXTURE_BASE and p.name in (*HOOKS,'reference'),'Unregistered acquisition fixture')
 else:need(p==DEFAULT,'Production acquisition output differs')
 return p


def pool():
 p=read(s.FOLDER/'pool.json');rows=p['roots']
 need(p['schema']==s.SCHEMA+'-pool' and len(rows)==512 and
  [r['index'] for r in rows]==list(range(512)) and len({r['id'] for r in rows})==512,'Frozen finite pool differs')
 for row in rows:
  state=config.position(row['state'])
  need(row['id']==row['family'] and row['origin']=='footprint-screen/'+row['family'] and
   row['state_key']==s.common.rawkey(state),'Pool true family/state/origin differs')
 return p


def schedule(p,fixture):
 rows=p['roots'][:1] if fixture else p['roots']
 return [dict(index=i,root_index=r['index'],kind=k,root_id=r['id'],origin=r['origin'])
  for i,(r,k) in enumerate((r,k) for r in rows for k in ('teacher','features'))]


def identity_expected(output,fixture):
 from . import footprint_screen_native as native
 p=pool();rows=schedule(p,fixture)
 return dict(schema=SCHEMA+'-identity',fixture=fixture,output=s.relative(scope(output,fixture)),
  pool=ref(s.FOLDER/'pool.json'),plan_sha256=sha(s.PLAN),schedule=rows,schedule_sha256=config.digest(rows),
  sources=s.common.pins(SOURCES),native_inputs=native.input_pins(),native_identity=native.source_identity(),
  models_fixed=True,policy_fixed=True,training_admitted=False,reservations_admitted=False)


def prepare(output=DEFAULT,fixture=False):
 s.admit_plan();p=scope(output,fixture);expected=identity_expected(p,fixture)
 atomic(p/'identity.json',expected);return expected


def _identity(path):
 identity=read(path/'identity.json');s.admit_plan()
 need(same(identity,identity_expected(path,identity['fixture'])),'Acquisition identity/source scope changed')
 return identity


def _events(path):
 events=[];previous=None;identity_sha=sha(path/'identity.json')
 files=sorted((path/'commits').glob('*.json'))
 for i,cp in enumerate(files):
  need(cp.name==f'{i:06d}.json','Acquisition commit cursor gap');c=read(cp);ep=path/c['event']
  expected=dict(index=i,identity_sha256=identity_sha,event=c['event'],event_sha256=sha(ep),previous_sha256=previous)
  need(same(c,expected) and c['event']==f'events/{i:06d}-{c["event_sha256"]}.json','Acquisition commit binding differs')
  events.append(read(ep));previous=sha(cp)
 return events,previous


def _append(path,events,event):
 event=s.normalized(event);i=len(events);payload=(config.canonical(event)+'\n').encode();digest=hashlib.sha256(payload).hexdigest()
 ep=path/f'events/{i:06d}-{digest}.json';extras=[p for p in (path/'events').glob(f'{i:06d}-*.json') if p!=ep]
 need(not extras,'An incompatible uncommitted event forbids new work')
 atomic(ep,event);cp=path/f'commits/{i:06d}.json'
 atomic(cp,dict(index=i,identity_sha256=sha(path/'identity.json'),event=s.relative(ep).split(s.relative(path)+'/',1)[1],
  event_sha256=digest,previous_sha256=sha(path/f'commits/{i-1:06d}.json') if i else None))
 events.append(event);print(config.canonical(dict(event='footprint_screen_commit',kind=event['kind'],index=i,cursor=event.get('cursor'))),flush=True)
 return dict(event_path=str(ep),commit_path=str(cp),event_sha256=digest,commit_sha256=sha(cp))


def _cursor(i):return f'query/{i:06d}'
def _archive_path(i):return f'archives/query-{i:06d}.json'
def _sink_path(i):return f'attempts/query-{i:06d}.bin'


def _archive_write(path,body):
 """Stream canonical JSON once; never hold a second entire JSON document."""
 from . import footprint_screen_resources as resources
 need(not path.exists(),'Acquisition archive already exists; adopt it instead of rewriting')
 path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.pending')
 encoder=json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
 with pending.open('xb') as stream:
  for chunk in encoder.iterencode(body):stream.write(chunk.encode())
  stream.write(b'\n');stream.flush();os.fsync(stream.fileno())
 resources.check_namespace();os.replace(pending,path)


def _check_capture(body,identity,meta,row,life,claims,claim_hash,generation):
 from . import footprint_screen_native as native
 need(body['schema']==SCHEMA+'-archive' and body['identity_sha256']==config.digest(identity) and
  body['cursor']==_cursor(row['index']) and same(body['scheduled'],row) and
  body['root_id']==meta['id'] and body['family']==meta['family'] and body['origin']==meta['origin'],
  'Acquisition returned envelope differs from finite intent')
 need(body['status']=='recorded' and body['recording_only'] is True,'Failed acquisition cannot become a complete capture')
 expected=native.capture_proof(body['answer'],row['kind'],meta,body['cursor'],identity['native_identity']['native'],life)
 need(same(body['capture_proof'],expected),'Runtime compact capture proof differs')
 stage,after,keys=native.stage_check(body['answer'],claims,claim_hash,generation)
 return dict(capture=expected,stage=stage,claims=after,keys=keys)


def _read(path,identity):
 p=pool();events,prefix=_events(path);results=[];intent=None;pending=None;hold=None;life=None
 claims=b'';generation=0;claim_hash=ZERO;root_index=None;restored=None;epoch=None
 for i,event in enumerate(events):
  kind=event['kind'];need(hold is None,'HOLD cannot be followed by further queries')
  if kind=='guard_restore':
   row=identity['schedule'][len(results)];ri=row['root_index']
   if ri!=root_index:claims=b'';generation=0;claim_hash=ZERO;root_index=ri
   need(intent is None and pending is None and event['root_index']==ri and event['generation']==generation and
    event['canonical_claims_sha256']==hashlib.sha256(claims).hexdigest(),'Root-local cold restore differs')
   claim_hash=event['canonical_claims_sha256'];restored=ri
  elif kind=='intent':
   row=identity['schedule'][len(results)];need(intent is None and pending is None and restored==row['root_index'] and
    same(event,dict(kind='intent',cursor=_cursor(row['index']),scheduled=row,archive=_archive_path(row['index']),sink=_sink_path(row['index']))),
    'Acquisition intent differs from fixed order')
   intent=event
  elif kind=='provider_epoch':
   need(intent is not None and pending is None and event['cursor']==intent['cursor'] and
    event['previous_commit_sha256']==sha(path/f'commits/{i-1:06d}.json') and
    event['generation']==generation and event['canonical_claims_sha256']==hashlib.sha256(claims).hexdigest() and
    same(event['native_identity'],identity['native_identity']),'Cold provider epoch differs')
   life=event['lifecycle'];need(type(life['created']) is int and life['created']==2 and type(life['destroyed']) is int and
    life['destroyed']==0 and sorted(life['retained_inference_controls'].values())==[1,2],'Provider was not physically fresh')
   epoch=ref(path/f'events/{i:06d}-{hashlib.sha256((config.canonical(event)+chr(10)).encode()).hexdigest()}.json')
  elif kind=='result':
   need(intent is not None and pending is None and event['cursor']==intent['cursor'],'Acquisition result lacks intent')
   ap=path/intent['archive'];need(same(event['archive_ref'],ref(ap)),'Returned acquisition bytes changed');body=read(ap)
   need(same(body['provider_epoch_ref'],epoch),'Returned query belongs to another provider epoch')
   row=intent['scheduled'];meta=p['roots'][row['root_index']]
   checked=_check_capture(body,identity,meta,row,life,claims,claim_hash,generation)
   pending=dict(body=body,checked=checked,archive_ref=ref(ap))
  elif kind=='trace_commit':
   need(pending is not None and intent is not None and event['cursor']==intent['cursor'] and
    same(event['stage'],pending['checked']['stage']) and same(event['archive_ref'],pending['archive_ref']),
    'Durable capture publication differs')
   claims=pending['checked']['claims'];generation+=1;claim_hash=event['stage']['staged_claim_hash']
   proof=pending['checked']['capture'];life=proof['lifecycle_after']
   results.append(dict(index=len(results),scheduled=intent['scheduled'],proof=proof,
    archive_ref=pending['archive_ref']))
   intent=None;pending=None
  elif kind=='hold':
   need(event['cursor']==(intent['cursor'] if intent else event['cursor']),'HOLD cursor differs')
   if event.get('failure_ref') is not None:
    fp=s.ROOT/event['failure_ref']['path'];need(fp.is_relative_to(path/'failures') and sha(fp)==event['failure_ref']['sha256'],
     'Failed raw capture changed')
    failed=read(fp);need(failed['identity_sha256']==config.digest(identity) and failed['cursor']==event['cursor'],
     'Failed attempt belongs to another query')
   hold=event
  else:need(False,'Unknown acquisition journal event')
 data=dict(events=events,prefix=prefix,results=results,intent=intent,pending=pending,hold=hold,claims=claims,
  generation=generation,claim_hash=claim_hash,root_index=root_index,restored=restored,life=life,epoch=epoch)
 _orphans(path,data);return data


def _orphans(path,data):
 allowed={r['archive_ref']['path'] for r in data['results']}
 if data['intent']:allowed.add(s.relative(path/data['intent']['archive']))
 for p in (path/'archives').glob('*.json'):need(s.relative(p) in allowed,'Unreferenced returned query prevents replay')
 sinks={s.relative(path/e['sink']) for e in data['events'] if e['kind']=='intent'}
 for p in (path/'attempts').glob('*.bin'):need(s.relative(p) in sinks,'Unreferenced raw attempt prevents replay')
 committed={read(p)['event'] for p in (path/'commits').glob('*.json')}
 extra=[p for p in (path/'events').glob('*.json') if p.relative_to(path).as_posix() not in committed]
 need(len(extra)<=1 and all(p.name.startswith(f'{len(data["events"]):06d}-') for p in extra),'Unbound journal suffix prevents replay')


def _report(identity,data):
 complete=len(data['results'])==len(identity['schedule']) and data['intent'] is None and data['pending'] is None and data['hold'] is None
 return dict(schema=SCHEMA+'-report',state='held' if data['hold'] else 'complete' if complete else 'checkpoint',
  fixture=identity['fixture'],required_queries=len(identity['schedule']),completed_queries=len(data['results']),
  required_roots=len(identity['schedule'])//2,settled_roots=len(data['results'])//2,
  complete_teachers=sum(r['proof']['math_complete'] for r in data['results'] if r['scheduled']['kind']=='teacher'),
  incomplete_teachers=sum(not r['proof']['math_complete'] for r in data['results'] if r['scheduled']['kind']=='teacher'),
  model_forwards=0,policy_fixed=True,models_fixed=True,training_admitted=False,reservations_admitted=False,
  independent_full_raw_review_required=True,hold=data['hold'],pending_attempts=int(data['intent'] is not None))


def _catalog(path,identity,data):
 return dict(schema=SCHEMA+'-catalog',identity_ref=ref(path/'identity.json'),report_ref=ref(path/'report.json'),
  commits=s.common.pins(sorted((path/'commits').glob('*.json'))),archives={r['archive_ref']['path']:r['archive_ref']['sha256'] for r in data['results']},
  failed_refs=[e['failure_ref'] for e in data['events'] if e['kind']=='hold' and e.get('failure_ref')],
  recording_only=True,full_raw_audit_pending=True)


def _finalize(path,identity,data,pause=lambda p:None):
 report=_report(identity,data);need(report['state'] in ('complete','held'),'Cannot seal unfinished acquisition')
 atomic(path/'report.json',report);catalog=_catalog(path,identity,data)
 pause('before_catalog');atomic(path/'catalog.json',catalog);pause('after_catalog')
 return dict(report=report,catalog=catalog,publication_gap=False)


def has_completion(output=DEFAULT):
 path=s.path_of(output)
 if not(path/'identity.json').exists():return False
 identity=_identity(path);data=_read(path,identity)
 return _report(identity,data)['state'] in ('complete','held') and (path/'report.json').exists()


def completed(output=DEFAULT):
 path=s.path_of(output);identity=_identity(path);data=_read(path,identity);expected=_report(identity,data)
 need(expected['state'] in ('complete','held') and same(read(path/'report.json'),expected),'Acquisition is not a settled exact report')
 catalog=_catalog(path,identity,data)
 if (path/'catalog.json').exists():need(same(read(path/'catalog.json'),catalog),'Acquisition catalog differs')
 return dict(report=expected,catalog=catalog,publication_gap=not(path/'catalog.json').exists())


def publish_completed(output=DEFAULT):
 done=completed(output)
 if not done['publication_gap']:return done
 need(done['report']['state']=='complete','Cannot repair terminal HOLD as successful completion')
 path=s.path_of(output)
 with Writer(path):_finalize(path,_identity(path),_read(path,_identity(path)))
 return completed(path)


def preflight(output=DEFAULT,fixture=False):
 s.admit_plan();path=scope(output,fixture)
 if (path/'identity.json').exists():
  identity=_identity(path);need(identity['fixture'] is fixture,'Fixture treatment changed');data=_read(path,identity)
  need(data['hold'] is None,'Retained acquisition HOLD; no retry')
  need(data['intent'] is None or (path/data['intent']['archive']).exists(),'Unknown unreturned acquisition; HOLD, no retry')
 else:need(not path.exists() or not any(path.rglob('*')),'Unbound acquisition evidence forbids overwrite')
 return dict(state='ready',fixture=fixture)


def _pause(path,point,wanted,marker,lease,provider_constructed):
 if point!=wanted:return
 from . import footprint_screen_resources as resources
 target=s.path_of(marker);atomic(target,dict(schema=SCHEMA+'-pause',point=point,identity_sha256=sha(path/'identity.json'),
  process=resources.process_identity(),argv=sys.argv,lease=str(Path(lease).resolve()),
  source_sha256=sha(__file__),query_entry_reached=point!='before_query',provider_constructed=provider_constructed))
 while True:time.sleep(.05)


def _adopt(path,identity,data,pause=lambda p:None,live=None):
 intent=data['intent'];need(intent is not None,'No returned intent to adopt');ap=path/intent['archive'];body=read(ap)
 row=intent['scheduled'];meta=pool()['roots'][row['root_index']]
 need(same(body['provider_epoch_ref'],data['epoch']),'Recorded return belongs to another provider epoch')
 if data['pending'] is None:
  check=_check_capture(body,identity,meta,row,data['life'],data['claims'],data['claim_hash'],data['generation'])
  pause('before_result_commit');_append(path,data['events'],dict(kind='result',cursor=intent['cursor'],archive_ref=ref(ap)));pause('after_result_commit')
  data['pending']=dict(body=body,checked=check,archive_ref=ref(ap))
 pending=data['pending'];pause('before_trace_commit')
 proof=_append(path,data['events'],dict(kind='trace_commit',cursor=intent['cursor'],archive_ref=pending['archive_ref'],stage=pending['checked']['stage']))
 pause('after_trace_commit')
 if live is not None:live.publish(pending['checked']['stage'],proof)
 data['claims']=pending['checked']['claims'];data['generation']+=1;data['claim_hash']=pending['checked']['stage']['staged_claim_hash']
 capture=pending['checked']['capture'];data['life']=capture['lifecycle_after']
 data['results'].append(dict(index=len(data['results']),scheduled=row,proof=capture,
  archive_ref=pending['archive_ref']))
 data['pending']=None;data['intent']=None


def run(output=DEFAULT,fixture=False,stop_after=None,pause_at=None,pause_marker=None,lease=None):
 if has_completion(output):return publish_completed(output)
 from . import footprint_screen_resources as resources
 need(lease is not None,'A contained acquisition lease is required');resources.assert_worker_lease(lease)
 s.admit_plan();path=scope(output,fixture);need(pause_at is None or fixture and pause_at in HOOKS,'Unadmitted acquisition pause')
 need(stop_after is None or type(stop_after) is int and stop_after>0,'Invalid cooperative stop cursor')
 path.mkdir(parents=True,exist_ok=True)
 with Writer(path):
  identity=_identity(path) if (path/'identity.json').exists() else prepare(path,fixture)
  data=_read(path,identity);need(data['hold'] is None,'Retained acquisition HOLD');session=None
  pause=lambda point:_pause(path,point,pause_at,pause_marker,lease,session is not None)
  if data['intent'] is not None:
   need((path/data['intent']['archive']).exists(),'Unknown unreturned acquisition: HOLD; no retry')
   _adopt(path,identity,data,pause)
  if len(data['results'])==len(identity['schedule']):return _finalize(path,identity,data,pause)
  from . import footprint_screen_native as native
  p=pool();active_root=None
  with ExitStack() as stack:
   try:
    while len(data['results'])<len(identity['schedule']):
     if resources.remaining_ns(lease)<30_000_000_000 or stop_after is not None and len(data['results'])>=stop_after:
      return dict(report=_report(identity,data),checkpoint=True)
     resources.assert_worker_lease(lease);row=identity['schedule'][len(data['results'])];meta=p['roots'][row['root_index']]
     cursor=_cursor(row['index'])
     # Raw mapped16m prefix is384MB; the new archive and its atomic pending file
     # share one name/inode. Reserve2GiB for the raw prefix plus complete packed
     # archive/report, then check actual namespace size before publication.
     try:resources.check_namespace(additional_bytes=2*1024**3)
     except BaseException as error:
      hold=dict(kind='hold',cursor=cursor,reason='Pre-query namespace capacity: '+str(error),failure_ref=None,
       raw_return_retained=False,retry_permitted=False,query_entry_reached=False)
      _append(path,data['events'],hold);data['hold']=hold;return _finalize(path,identity,data,pause)
     if active_root!=row['root_index']:
      if row['root_index']!=data['root_index']:data.update(claims=b'',generation=0,claim_hash=ZERO,root_index=row['root_index'])
      _append(path,data['events'],dict(kind='guard_restore',root_index=row['root_index'],generation=data['generation'],canonical_claims_sha256=hashlib.sha256(data['claims']).hexdigest()))
      data['claim_hash']=hashlib.sha256(data['claims']).hexdigest();data['restored']=row['root_index'];active_root=row['root_index']
      if session is not None:session.open_root(meta,data['claims'],data['generation'])
     intent=dict(kind='intent',cursor=cursor,scheduled=row,archive=_archive_path(row['index']),sink=_sink_path(row['index']))
     _append(path,data['events'],intent);data['intent']=intent;pause('before_query')
     if session is None:
      session=native.open_session(stack,p);session.open_root(meta,data['claims'],data['generation']);life=session.lifecycle()
      need(life['created']==2 and life['destroyed']==0 and sorted(life['retained_inference_controls'].values())==[1,2],'Provider is not physically fresh')
      proof=_append(path,data['events'],dict(kind='provider_epoch',cursor=cursor,native_identity=identity['native_identity'],lifecycle=life,
       generation=data['generation'],canonical_claims_sha256=hashlib.sha256(data['claims']).hexdigest(),
       previous_commit_sha256=sha(path/f'commits/{len(data["events"])-1:06d}.json')))
      data['life']=life;data['epoch']=dict(path=s.relative(proof['event_path']),sha256=proof['event_sha256'])
     (path/'attempts').mkdir(exist_ok=True);transient=None;started=time.perf_counter()
     try:
      transient=session.query(row['kind'],meta,cursor,path/intent['sink']);answer=native.archive(transient)
      capture=native.capture_proof(answer,row['kind'],meta,cursor,identity['native_identity']['native'],data['life'])
      body=dict(schema=SCHEMA+'-archive',identity_sha256=config.digest(identity),cursor=cursor,scheduled=row,
       root_id=meta['id'],family=meta['family'],origin=meta['origin'],answer=answer,capture_proof=capture,
       provider_epoch_ref=data['epoch'],status='recorded',recording_only=True,elapsed_ms=(time.perf_counter()-started)*1000)
      _archive_write(path/intent['archive'],body);_adopt(path,identity,data,pause,session.guard)
      del body,answer,capture,transient
     except BaseException as error:
      transient=getattr(error,'transient',None) or transient;failure_ref=None
      if transient is not None:
       failed=native.archive_failure(transient);fp=path/f'failures/query-{row["index"]:06d}.json'
       _archive_write(fp,dict(schema=SCHEMA+'-failure',identity_sha256=config.digest(identity),cursor=cursor,scheduled=row,
        family=meta['family'],origin=meta['origin'],answer=failed,error=type(error).__name__+': '+str(error),
        elapsed_ms=(time.perf_counter()-started)*1000,recording_only=True));failure_ref=ref(fp)
      hold=dict(kind='hold',cursor=cursor,reason=type(error).__name__+': '+str(error),failure_ref=failure_ref,
       raw_return_retained=transient is not None,retry_permitted=False)
      _append(path,data['events'],hold);data['hold']=hold;return _finalize(path,identity,data,pause)
    return _finalize(path,identity,data,pause)
   finally:
    if session is not None:session.close()


def _checked_ref(reference):
 path=s.path_of(s.ROOT/reference['path']);need(sha(path)==reference['sha256'],'Independent recovery evidence changed')
 return path


def _recovery_gate():
 """Pure external process-interruption proof; never infer missing==zero."""
 rp=s.FOLDER/'audit/native-recovery/result.json';r=read(rp);plan=s.admit_plan();inputs=s.common.pins([rp])
 hooks=set(HOOKS)-{'before_query'}
 need(r['schema']==s.SCHEMA+'-independent-native-recovery' and r['passed'] is True and
  r['independent_of_implementation'] is True and r['plan_sha256']==sha(s.PLAN) and
  type(r['process_interruptions']) is int and r['process_interruptions']==7 and
  type(r['subset_only_new_quotients']) is int and r['subset_only_new_quotients']==0 and
  r['unresolved_findings']==[] and set(r['cases'])==hooks,'Independent acquisition recovery gate is missing or incomplete')
 need(r['source_inputs'] and all(plan['source_inputs'].get(k)==v for k,v in r['source_inputs'].items()),
  'Independent recovery lacks exact frozen source witnesses')
 s.common.current(r['source_inputs']);inputs.update(r['source_inputs'])
 for hook in sorted(hooks):
  path=FIXTURE_BASE/hook;case=r['cases'][hook];done=completed(path)
  need(case['passed'] is True and case['actual_process_interruption'] is True and
   done['report']['state']=='complete' and done['report']['fixture'] is True and not done['publication_gap'] and
   same(case['fixture_identity_ref'],ref(path/'identity.json')) and same(case['catalog_ref'],ref(path/'catalog.json')),
   'A recovery fixture is not the complete actual interrupted source')
  references=[case['fixture_identity_ref'],case['catalog_ref'],*case['evidence_refs']]
  need(case['evidence_refs'],'Actual interruption evidence is required')
  for reference in references:inputs.update(s.common.pins([_checked_ref(reference)]))
 zero=r['zero_query_case'];path=FIXTURE_BASE/'before_query';identity=_identity(path);data=_read(path,identity)
 need(zero['hook']=='before_query' and type(zero['actual_query_count']) is int and zero['actual_query_count']==0 and
  zero['provider_constructed'] is False and zero['query_entry_reached'] is False and zero['no_retry'] is True and
  same(zero['identity_ref'],ref(path/'identity.json')) and len(data['results'])==0 and data['pending'] is None and
  data['intent'] is not None and data['intent']['cursor']=='query/000000' and
  not any(e['kind'] in ('provider_epoch','result','trace_commit') for e in data['events']) and
  not any((path/'archives').glob('*')) and not any((path/'attempts').glob('*')),
  'Unreturned fixture has no independently proven zero-query boundary')
 marker_path=_checked_ref(zero['marker_ref']);marker=read(marker_path)
 need(marker['schema']==SCHEMA+'-pause' and marker['point']=='before_query' and
  marker['identity_sha256']==sha(path/'identity.json') and marker['source_sha256']==sha(__file__) and
  marker['provider_constructed'] is False and marker['query_entry_reached'] is False,
  'Zero-query marker is not the source-bound pre-provider boundary')
 lease_path=s.path_of(marker['lease']);lease=read(lease_path)
 worker_path=s.path_of(lease['worker_identity_file']);worker=read(worker_path)
 permit_path=s.path_of(lease['permit_file']);permit=read(permit_path)
 need(same(marker['process'],worker['process']) and marker['argv']==worker['argv'] and
  permit['worker']==worker['process'] and permit['token']==lease['token'] and
  Path(lease['worker_output']).resolve()==path.resolve() and lease['plan_sha256']==sha(s.PLAN),
  'Zero-query marker process/argv/lease/permit differs')
 inputs.update(s.common.pins([path/'identity.json',marker_path,lease_path,worker_path,permit_path]));return inputs


def _unit_quotients(result):
 """Load one immutable footprint only when building a post-acquisition view."""
 from . import footprint_screen_native as native
 ap=s.path_of(s.ROOT/result['archive_ref']['path'])
 need(sha(ap)==result['archive_ref']['sha256'],'Snapshot capture bytes changed')
 body=read(ap);_,keys=native.owned_records(body['answer'])
 return [q.hex() for q in keys]


def _snapshot_body(include_fixtures=True):
 done=completed();need(done['report']['state']=='complete' and not done['publication_gap'],'Full acquisition pool is not sealed')
 identity=_identity(DEFAULT);data=_read(DEFAULT,identity);p=pool();roots=[];inputs={}
 for meta in p['roots']:
  units=[r for r in data['results'] if r['scheduled']['root_index']==meta['index']]
  need(len(units)==2,'A finite acquisition root is missing a query')
  sources=[r['archive_ref'] for r in units]
  roots.append(dict(id=meta['id'],index=meta['index'],teacher_complete=units[0]['proof']['math_complete'],
   features_complete=units[1]['proof']['math_complete'],trace_complete=True,hold=False,
   quotients=sorted(set(q for r in units for q in _unit_quotients(r))),source_refs=sources))
 for path in (DEFAULT/'identity.json',DEFAULT/'report.json',DEFAULT/'catalog.json',s.FOLDER/'pool.json',s.PLAN):inputs.update(s.common.pins([path]))
 inputs.update(s.common.pins(f for f in DEFAULT.rglob('*') if f.is_file() and f.name!='writer.lock' and not f.name.endswith('.pending')))
 for r in data['results']:inputs[r['archive_ref']['path']]=r['archive_ref']['sha256']
 # Every actual engineering episode stays a source even when its deterministic
 # raw footprint duplicates a primary episode. Family incidence remains true.
 if include_fixtures:
  inputs.update(_recovery_gate())
  for folder in sorted(FIXTURE_BASE.glob('*')):
   if not(folder/'identity.json').exists():continue
   fi=_identity(folder);fd=_read(folder,fi)
   if folder.name!='before_query':
    done=completed(folder);need(done['report']['state']=='complete' and not done['publication_gap'],
     'Fixture has an unadopted return or unknown query')
   for r in fd['results']:
    roots[r['scheduled']['root_index']]['source_refs'].append(r['archive_ref'])
    roots[r['scheduled']['root_index']]['quotients']=sorted(set(roots[r['scheduled']['root_index']]['quotients'])|set(_unit_quotients(r)))
    inputs[r['archive_ref']['path']]=r['archive_ref']['sha256']
   inputs.update(s.common.pins(f for f in folder.rglob('*') if f.is_file() and f.name!='writer.lock' and not f.name.endswith('.pending')))
 return dict(schema=s.SCHEMA+'-completed-snapshot',pool_sha256=sha(s.FOLDER/'pool.json'),roots=roots,inputs=inputs)


def production_snapshot():
 """Preliminary primary captures for recovery subset checks; no admission."""
 return _snapshot_body(False)


def completed_snapshot():return _snapshot_body(True)


def validate_snapshot(snapshot):
 need(same(snapshot,_snapshot_body()),'Screen snapshot differs from immutable recorded acquisition and fixture journals')
 return True


def full_raw_review(archive_ref):
 """One pure expensive audit unit; callers may journal independent units."""
 from . import footprint_screen_native as native
 ap=s.path_of(s.ROOT/archive_ref['path']);need(sha(ap)==archive_ref['sha256'],'Raw audit input archive changed')
 need(ap.parent.name=='archives','Raw review requires a recorded complete-return archive')
 path=ap.parent.parent;identity=_identity(path);body=read(ap);row=body['scheduled'];meta=pool()['roots'][row['root_index']]
 need(body['identity_sha256']==config.digest(identity) and same(row,identity['schedule'][row['index']]),
  'Raw audit query is outside frozen acquisition')
 result=native.full_raw_review(body['answer'],row['kind'],meta,body['cursor'],identity['native_identity']['native'])
 raw=native.old.trace_bytes(body['answer']['trace']);low=0
 for cells,actor,*_ in native.old.frozen.PACK.iter_unpack(raw):
  if 98-cells[7]-cells[15]<=24:low+=1
 owned,keys=native.owned_records(body['answer'])
 return dict(schema=SCHEMA+'-full-raw-unit',passed=True,archive_ref=archive_ref,root_id=meta['id'],root_index=meta['index'],
  family=meta['family'],origin=meta['origin'],kind=row['kind'],raw_sha256=result['raw_sha256'],
  events=len(raw)//24,low_shared_events=low,high_owned_events=len(raw)//24-low,
  canonical_high_quotients=[q.hex() for q in keys],high_keys_sha256=hashlib.sha256(b''.join(keys)).hexdigest(),
  first_occurrence_owned40_base64=base64.b64encode(owned).decode(),owned40_sha256=hashlib.sha256(owned).hexdigest(),
  model_forwards=0,new_queries=0,pool_sha256=sha(s.FOLDER/'pool.json'),source_inputs=s.common.pins(SOURCES))


def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','completed','status','publish','snapshot'))
 parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--lease');parser.add_argument('--engineering-fixture',action='store_true')
 parser.add_argument('--stop-after',type=int);parser.add_argument('--pause-at');parser.add_argument('--pause-marker');args=parser.parse_args(argv)
 if args.command=='run':
  if has_completion(args.output):answer=publish_completed(args.output)
  else:
   from . import footprint_screen_resources as resources
   need(args.lease is not None,'Contained acquisition lease missing');resources.bind_worker(args.lease)
   answer=run(args.output,args.engineering_fixture,args.stop_after,args.pause_at,args.pause_marker,args.lease)
   resources.worker_complete(args.lease,dict(state=answer['report']['state']))
 elif args.command=='completed':answer=completed(args.output)
 elif args.command=='publish':answer=publish_completed(args.output)
 elif args.command=='snapshot':answer=completed_snapshot()
 else:
  path=s.path_of(args.output);answer=dict(state='not_started') if not(path/'identity.json').exists() else _report(_identity(path),_read(path,_identity(path)))
 print(config.canonical(answer));return answer


if __name__=='__main__':main()
