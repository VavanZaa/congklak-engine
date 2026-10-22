"""Pure actual-source inventory and known-query overlay; unknown history stays unknown.

Only paths in the preceding accepted closure may supply historical evidence.
Actual query runs are selected by their durable leases, not by recursively
mistaking copied reader fixtures or synthetic bytes for executed searches.
"""
from collections import Counter,defaultdict
from functools import lru_cache
import base64,hashlib,json,struct
from pathlib import Path
from . import config,ownership_resolution_common as stage,native_ownership_trace as codec

ROOT=stage.ROOT
SOURCES=('congklak/ownership_resolution_provenance.py','congklak/ownership_resolution_common.py',
 'congklak/native_ownership_trace.py','congklak/config.py')
BASE='build/ownership-implementation-v1/provenance/certified-v2/report.json'
SUPPLEMENT='build/ownership-implementation-v1/provenance/engineering-supplement-v1/report.json'
PRIOR_SHA='9b7da69df92fe6b7ca8bf0d126b3cd89cb6ff717c341b89db1ad0b573a0c717c'
STATIC=(BASE,SUPPLEMENT,
 'build/ownership-implementation-v1/provenance/certified-v2/certificate.json',
 'build/ownership-implementation-v1/provenance/engineering-supplement-v1/certificate.json',
 'build/ownership-implementation-v1/native/author-v1/failed-attempt.json',
 'build/ownership-implementation-v1/native/author-v1/source-witness.json',
 'build/ownership-implementation-v1/native/author-v1/source/tools/check-ownership-native-v1.py',
 'build/ownership-implementation-v1/native/smoke-control-v1.json',
 'build/ownership-implementation-v1/native/smoke-H-v1.json',
 'build/ownership-implementation-v1/reservations/pre-draw-binding-v4.json',
 'build/ownership-implementation-v1/audit/worker-recovery-v2.py',
 'build/ownership-implementation-v1/audit/worker-recovery-v2.log',
 'build/ownership-implementation-v1/audit/worker-recovery/case-before_decision_commit.json',
 'docs/NATIVE_OWNERSHIP_TRACE_IMPLEMENTATION.md','docs/OWNERSHIP_IMPLEMENTATION.md',
 'build/tracing-speed-v1/audit/audit.json','build/tracing-speed-v1/audit/resource-final/result.json',
 'build/tracing-speed-v1/audit/calibration-final/result.json',
 'build/tracing-speed-v1/audit/timing-final/result.json')
OWNED=struct.Struct('<15sQ16sB')
need=stage.need

@lru_cache(maxsize=1)
def _accepted():
 raw=stage.PRIOR.read_bytes();need(hashlib.sha256(raw).hexdigest()==PRIOR_SHA,'Prior acceptance changed')
 body=json.loads(raw);need(body['passed'] is True and len(body['inputs'])==78403,'Prior acceptance scope differs')
 return body['inputs']

def _read(path):
 pins=_accepted();need(path in pins,'Historical source is outside accepted closure: '+path)
 raw=(ROOT/path).read_bytes();need(hashlib.sha256(raw).hexdigest()==pins[path],'Accepted historical source changed: '+path)
 return json.loads(raw)

def _ref(path):return dict(path=path,sha256=_accepted()[path])

def _same(a,b):return config.canonical(a)==config.canonical(b)

def _family(identity_path,origin):
 return 'recorded-engineering-origin/'+config.digest(dict(run_identity_path=identity_path,declared_origin=origin))

@lru_cache(maxsize=1)
def _layout():
 pins=_accepted();runs={}
 for path in sorted(pins):
  if path.startswith('build/tracing-speed-v1/resources/leases/') and path.endswith('/lease.json'):
   lease=_read(path);folder=Path(lease['worker_output']).resolve().relative_to(ROOT.resolve()).as_posix()
   entry=runs.setdefault(folder,dict(leases=[],spec_path=Path(lease['cache_path']).resolve().relative_to(ROOT.resolve()).as_posix()))
   need(entry['spec_path']==Path(lease['cache_path']).resolve().relative_to(ROOT.resolve()).as_posix(),'Run changed spec across leases')
   entry['leases'].append(path)
 tasks=[];classified=set();exclusions=[]
 for folder,run in sorted(runs.items()):
  identity_path=folder+'/identity.json';identity=_read(identity_path);spec=_read(run['spec_path'])
  need(identity['spec_path']==run['spec_path'] and identity['spec_sha256']==pins[run['spec_path']],'Actual run spec binding differs')
  kind=spec['kind'];need(kind in ('calibration','timing','handoff'),'Unknown executed worker kind')
  middle='/archives/' if kind=='handoff' else '/attempts/'
  archives=sorted(p for p in pins if p.startswith(folder+middle) and p.endswith('.json'))
  intent_map={};cursor_map={}
  if kind=='handoff':
   intent_sources=sorted(p for p in pins if p.startswith(folder+'/events/') and p.endswith('.json'))
   for path in intent_sources:
    event=_read(path)
    if event['kind']!='decision_intent':continue
    target=folder+'/'+event['archive'];cursor=event['decision_id']
    need(target not in intent_map and cursor not in cursor_map,'Duplicate actual handoff attempt intent')
    intent_map[target]=path;cursor_map[cursor]=target
  else:
   intent_sources=sorted(p for p in pins if p.startswith(folder+'/intents/') and p.endswith('.json'))
   for path in intent_sources:
    intent=_read(path);target=folder+middle+Path(path).name;cursor=intent['decision_id']
    need(target not in intent_map and cursor not in cursor_map,'Duplicate actual calibration/timing attempt intent')
    need(intent['identity_sha256']==pins[identity_path],'Actual intent identity differs')
    intent_map[target]=path;cursor_map[cursor]=target
  need(set(intent_map)==set(archives),'Missing or orphan actual query intent/archive')
  run['intent_sources']=intent_sources
  run['query_coverage']=dict(run=folder,kind=kind,identity=_ref(identity_path),spec=_ref(run['spec_path']),
   durable_intent_count=len(intent_map),actual_archive_count=len(archives),
   one_to_one=True,missing_intents=0,missing_archives=0,duplicate_cursors=0,
   durable_intent_sources=[_ref(intent_map[p]) for p in archives])
  for path in archives:
   stem=Path(path).stem
   if kind=='handoff':
    game,choice=map(int,stem.split('-'));origin=spec['schedule'][game]['origin'];cursor=f'{game:03d}/{choice:03d}'
   elif kind=='calibration':
    index=int(stem);origin=spec['schedule'][index]['origin'];cursor=f'calibration/{index:06d}'
   else:
    index=int(stem);need(len(spec['schedule'])==1,'Timing origin scope changed')
    origin=spec['schedule'][0]['origin'];cursor=f'timing/{index:06d}'
   proof=None
   candidate='build/tracing-speed-v1/audit/'+('timing-final' if kind=='timing' else 'calibration-final')+'/result.json'
   if kind in ('timing','calibration'):
    audit=_read(candidate)
    if audit.get('passed') is True and audit.get('source_inputs',{}).get(path)==pins[path]:proof=_ref(candidate)
   family=_family(identity_path,origin)
   need(cursor_map.get(cursor)==path,'Archive filename/intent decision cursor differs')
   task=dict(index=len(tasks),source=path,source_sha256=pins[path],kind=kind,run=folder,
    identity=_ref(identity_path),spec=_ref(run['spec_path']),leases=[_ref(p) for p in run['leases']],
    attempt_intent=_ref(intent_map[path]),declared_origin=origin,family=family,decision_id=cursor,coverage_proof=proof)
   tasks.append(task);classified.add(path)
  for path in pins:
   if (path.startswith(folder+'/failed-prefixes/') or path.startswith(folder+'/failed-attempts/')) and path.endswith('.json'):
    # No actual kernel-loss run exists in this accepted stage. If one appears,
    # refuse instead of silently turning it into a synthetic exclusion.
    need(False,'Unclassified actual-run failed prefix requires a versioned task: '+path)
 for path in sorted(pins):
  if not path.startswith('build/tracing-speed-v1/') or not path.endswith('.json'):continue
  if any(token in path for token in ('/archives/','/attempts/','/failed-prefixes/')) and path not in classified:
   reason='retained source copy; no additional executed query' if '/source/' in path else 'reader/synthetic fixture without an actual worker lease; not an executed query'
   exclusions.append(dict(source=path,sha256=pins[path],classification=reason,actual_query=False))
 need(len(tasks)==692 and len(runs)==16,'Finite accepted actual-query inventory changed')
 return tasks,runs,exclusions

def plan():return stage.normalized(_layout()[0])

def input_pins():
 pins=_accepted();selected=set(STATIC)
 for proof_path in STATIC[16:19]:
  for path,expected in _read(proof_path)['source_inputs'].items():
   need(pins.get(path)==expected,'Accepted independent proof dependency is not source-bound: '+path)
   selected.add(path)
 for task in _layout()[0]:
  selected.add(task['source']);selected.update(r['path'] for r in [task['identity'],task['spec'],task['attempt_intent'],*task['leases']])
  if task['coverage_proof']:selected.add(task['coverage_proof']['path'])
 for run in _layout()[1].values():selected.update(run['intent_sources'])
 # Every excluded source is still bound by the accepted manifest; hash it in
 # the stage closure so exclusions cannot be manufactured by dropping files.
 selected.update(x['source'] for x in _layout()[2])
 missing=selected-set(pins);need(not missing,'Required historical evidence not accepted: '+str(sorted(missing)))
 return dict({p:pins[p] for p in sorted(selected)},**{stage.relative(stage.PRIOR):PRIOR_SHA})

def _inventory_document():
 base=_read(BASE);supp=_read(SUPPLEMENT);failed=_read(STATIC[4]);binding=_read(STATIC[9])
 tasks,runs,excluded=_layout();ledger=[]
 ledger.append(dict(id='broad-historical-native-internals',category='unknown_query_footprint',
  scope=base['coverage_scope'],limitation=base['historical_internal_search_feature_tt_leaves'],
  evidence=[_ref(BASE)],exact_missing_attempt_count=None,permanently_unknown=True,
  optimized_unseen_states_claimed=False,exact_state_set_recoverable=False))
 ledger.append(dict(id='native-author-v1-timed-overflow',category='unknown_engineering_query_footprint',
  evidence=[_ref(STATIC[4]),_ref(STATIC[5]),_ref(STATIC[6])],reported_attempts=failed['uncommitted_timed_search_attempts'],
  exact_state_set_recoverable=False,partial_raw_saved=False,permanently_unknown=True,
  reported_completed_nodes=failed['last_completed_native_report_nodes'],
  reported_completed_depth=failed['last_completed_native_report_choice_depth'],
  no_replay_as_historical_truth=True,possible_root_active_upper_bound=25,
  bound_status='descriptive root bound only; no exact footprint or transposition/context completeness claim'))
 ledger.append(dict(id='ad-hoc-native-smokes',category='partly_recorded_engineering_query_group',
  evidence=[_ref(SUPPLEMENT),_ref(STATIC[13]),_ref(STATIC[7]),_ref(STATIC[8])],
  reported_group=dict(PUCT=5,baseline=6,total=11),known_saved_complete_envelopes=[_ref(STATIC[7]),_ref(STATIC[8])],
  exact_missing_attempt_count=None,exact_missing_state_set=None,permanently_unknown=True,
  accounting_discrepancy=dict(reported_cumulative=106,reported_author_v1=45,reported_author_v2=51,
   residual_smokes_from_breakdown=10,reported_smoke_group=11,reconciled=False),
  no_subtraction_without_episode_identity=True))
 additional=binding['additional_missing_attempt_ledger']
 need(len(additional)==1 and additional[0]['exact_actual_attempt_capture_claim'] is False,'Old worker gap qualifier differs')
 ledger.append(dict(id='old-worker-before-decision-commit',category='unknown_engineering_query_footprint',
  evidence=[_ref(STATIC[9]),_ref(STATIC[10]),_ref(STATIC[11]),_ref(STATIC[12])],
  old_qualification=additional[0],exact_state_set_recoverable=False,permanently_unknown=True,
  expected_duplicate_is_not_actual_capture=True,possible_root_active_upper_bound=28,
  bound_status='descriptive root bound only; retained TT/cache/input scope must not be inferred from root alone'))
 for i,prefix in enumerate(supp['partial_failed_trace_prefixes']):
  ledger.append(dict(id=f'old-persisted-incomplete-prefix/{i}',category='actual_saved_prefix_with_unknown_tail',
   evidence=[prefix['source'],_ref(SUPPLEMENT)],origin=prefix['origin'],
   saved_event_interval=[0,prefix['saved_events']],attempted_event_count=prefix['attempted_events'],
   complete_trace=False,unknown_is_not_empty=True,exact_suffix_recoverable=False,
   historical_binding=prefix['binding']))
 return dict(schema='ownership-resolution-historical-inventory-v1',inputs=input_pins(),
  actual_current_query_episodes=len(tasks),actual_current_run_namespaces=len(runs),
  actual_current_kind_counts=dict(sorted(Counter(t['kind'] for t in tasks).items())),
  actual_run_query_coverage=[runs[folder]['query_coverage'] for folder in sorted(runs)],
  durable_actual_intents_paired_one_to_one=True,
  latest_queries_all_archived=True,actual_current_missing_query_archives=0,
  recorded_reconstruction=dict(source=_ref(BASE),literal_replayed_actions=base['literal_replayed_actions'],
   current_components=len(base['components']),coverage_scope=base['coverage_scope']),
  missing_attempts=ledger,excluded_nonquery_artifacts=excluded,
  accepted_coverage_proofs=[dict(source=_ref(path),dependencies=_read(path)['source_inputs']) for path in STATIC[16:19]],
  exposure_types=['known optimizer support','inherited checkpoint/model-selection background',
   'recorded actual query exposure','unknown historical query footprint','metadata-only synthetic/copy evidence'],
  historical_provenance_complete=False,historically_unseen_claim=False,ancestry_unknown=True,
  unknown_queries_are_not_proof_of_gradient_use=True,old_failed_results_rehabilitated=False,
  reconstruction='Actual saved bytes and accepted source-bound coverage proofs only',
  model_files_hashed_only=True,models_fixed=True,policy_fixed=True,activity=stage.ZERO,
  training_authorized=False,production_evaluation_admitted=False)

def inventory():return stage.normalized(_inventory_document())
def validate_inventory(report):
 need(_same(report,stage.normalized(_inventory_document())),'Historical coverage inventory semantics differ')
 return True

def _raw_key(cells,actor):return min(cells[:7]+cells[8:15]+bytes([actor]),cells[8:15]+cells[:7]+bytes([1-actor]))

def _unit_document(task):
 need(type(task) is dict and type(task.get('index')) is int and 0<=task['index']<len(_layout()[0]) and
  _same(task,_layout()[0][task['index']]),'Actual episode task changed')
 body=_read(task['source']);answer=body['answer'];trace=answer['trace'];cert=answer['fast_certificate']
 intent=_read(task['attempt_intent']['path'])
 need(body['decision_id']==task['decision_id'] and trace['binding']['decision_id']==task['decision_id'],
  'Actual query decision cursor differs')
 need(intent['decision_id']==task['decision_id'],'Actual query intent cursor differs')
 if task['kind']=='handoff':
  need(_same(intent['before'],body['before']) and intent['engine']==body['engine'] and
   _same(trace['binding']['root_state'],body['before']),'Handoff query intent/root/engine differs')
 else:
  need(_same(intent['scheduled'],body['scheduled']),'Calibration/timing scheduled intent differs')
  if task['kind']=='calibration':
   need(body['intent_sha256']==task['attempt_intent']['sha256'] and
    _same(trace['binding']['root_state'],body['scheduled']['state']),'Calibration intent/root differs')
  else:need(_same(intent['root'],body['root']) and _same(trace['binding']['root_state'],body['root']),
   'Timing query intent/root differs')
 identity_hash=task['identity']['sha256']
 need(body.get('identity_sha256',config.digest(body.get('identity',{})))==identity_hash or
  (body.get('identity') is not None and body['identity']==_read(task['identity']['path'])),
  'Actual query run identity differs')
 raw=codec.trace_bytes(trace);need(cert['event_sha256']==trace['event_sha256'],'Raw/certificate binding differs')
 complete=not(trace['overflow'] or trace['invalid'] or trace['truncated'] or trace['attempted_events']!=trace['event_count'])
 need(complete,'Current actual incomplete query requires explicit saved-prefix task')
 compact=base64.b64decode(answer['fast_owned_high'],validate=True)
 need(len(compact)==cert['owned_count']*40 and hashlib.sha256(compact).hexdigest()==cert['owned_sha256'],
  'Owned records hash/count differs')
 observed=[];last=None
 for i,(q,first,cells,actor) in enumerate(OWNED.iter_unpack(compact)):
  need(len(q)==15 and sum(cells)==98 and actor in (0,1) and sum(q[:14])>=25 and
   q==_raw_key(cells,actor) and (last is None or last<q) and first<trace['event_count'],
   'Canonical full owned-record geometry differs')
  packed=codec.PACK.unpack_from(raw,first*24);need(packed[0]==cells and packed[1]==actor,
   'Owned first-occurrence witness differs from raw bytes')
  observed.append([q.hex(),first,(cells+bytes([actor])).hex()]);last=q
 mode='accepted-independent-complete-footprint-proof'
 if task['coverage_proof'] is not None:
  proof=_read(task['coverage_proof']['path'])
  need(proof['passed'] is True and proof['source_inputs'].get(task['source'])==task['source_sha256'] and
   type(proof['model_forwards']) is int and type(proof['new_searches']) is int and
   proof['model_forwards']==proof['new_searches']==0,'Accepted compact-coverage proof differs')
  if task['kind']=='timing':
   need(proof['schema']=='tracing-speed-independent-all240-v1' and proof['all_raw_and_owned_bytes_verified'] is True,
    'Timing proof does not certify owned records')
  else:need(proof['schema']=='tracing-speed-independent-completed-calibration-v1','Calibration proof schema differs')
 else:
  # The smaller non-panel histories are genuinely re-decoded from their ACTUAL
  # saved bytes, never searched again or inferred from a deterministic rerun.
  expected={};counts=Counter();high=0
  for index,(cells,actor,op,scope,reserved,action,depth) in enumerate(codec.PACK.iter_unpack(raw)):
   need(sum(cells)==98 and actor in (0,1) and 1<=op<=37 and scope in (1,2,3,4) and not reserved and action>=-1 and depth>=-1,
    'Actual archived complete-state prefix metadata differs')
   counts[codec.OP_NAMES[op-1]]+=1
   if 98-cells[7]-cells[15]>=25:
    high+=1;q=_raw_key(cells,actor);expected.setdefault(q,OWNED.pack(q,index,cells,actor))
  need(compact==b''.join(v for _,v in sorted(expected.items())) and
   {name:counts[name] for name in codec.OP_NAMES}==trace['operation_counts'],'Raw/owned full coverage differs')
  need(high==cert['owned_occurrences'],'Actual high-event partition differs')
  mode='reconstructed-from-actual-lossless-saved-bytes'
 high=cert['owned_occurrences'];need(0<=high<=trace['event_count'],'High/shared event partition differs')
 return dict(schema='ownership-resolution-actual-query-unit-v1',task_sha256=config.digest(task),
  index=task['index'],family=task['family'],declared_origin=task['declared_origin'],run=task['run'],
  episode=config.digest(dict(run_identity=task['identity'],decision_id=task['decision_id'],event_sha256=trace['event_sha256'])),
  source=_ref(task['source']),identity=task['identity'],spec=task['spec'],leases=task['leases'],attempt_intent=task['attempt_intent'],
  decision_id=task['decision_id'],engine=trace['binding']['engine'],root_state=trace['binding']['root_state'],
  event_sha256=trace['event_sha256'],event_count=trace['event_count'],owned_event_count=high,
  shared_event_count=trace['event_count']-high,owned_records=observed,owned_records_sha256=cert['owned_sha256'],
  coverage_mode=mode,coverage_proof=task['coverage_proof'],raw_link_preserved=True,complete_trace=True,
  native_or_prediction_work_repeated=False,actual_incidence=True,
  exposure_role='engineering acquisition only; no new optimizer/target/strength use',
  failed_attempt_retained=bool(body.get('error') or answer.get('fast_error')),
  models_fixed=True,policy_fixed=True,activity=stage.ZERO)

def unit(task):return stage.normalized(_unit_document(task))
def validate_unit(task,result):
 need(_same(result,stage.normalized(_unit_document(task))),'Actual-query unit source/footprint semantics differ')
 return True

class _Graph:
 def __init__(self):self.p={}
 def find(self,n):
  self.p.setdefault(n,n);r=n
  while self.p[r]!=r:r=self.p[r]
  while self.p[n]!=n:n,self.p[n]=self.p[n],r
  return r
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[b]=a

def _projection_document(units):
 tasks=_layout()[0];need(len(units)==len(tasks) and {u['index'] for u in units}==set(range(len(tasks))),
  'Known-query projection missing or duplicate actual episodes')
 base=_read(BASE);supp=_read(SUPPLEMENT);graph=_Graph();causes={};origin_q=defaultdict(set);edge_sources=defaultdict(dict)
 flags=('optimizer_blocked','fit_blocked','arena_blocked','ancestral_training','acquisition_exposed')
 for cid,component in base['components'].items():
  anchor=('prior_component',cid);graph.find(anchor)
  for f in component['families']:graph.join(anchor,('family',f))
  for q in component['quotients']:
   need(sum(bytes.fromhex(q)[:14])>=25,'Prior owned component contains shared q');graph.join(anchor,('quotient',q))
  causes[anchor]=dict(kind='accepted reconstructed high-component constraint',source=_ref(BASE),pointer='/components/'+cid,
   marker_indices=component['marker_indices'],**{f:component[f] for f in flags},actual_incidence=False)
 for cid,component in supp['components'].items():
  anchor=('prior_engineering_component',cid);graph.find(anchor)
  for previous in component['prior_components']:graph.join(anchor,('prior_component',previous))
  for q in component['quotients']:graph.join(anchor,('quotient',q))
  for origin in component['origins']:graph.join(anchor,('family','prior-engineering/'+origin))
  causes[anchor]=dict(kind='accepted saved engineering component constraint',source=_ref(SUPPLEMENT),pointer='/components/'+cid,
   optimizer_blocked=False,fit_blocked=False,arena_blocked=False,ancestral_training=False,acquisition_exposed=True,actual_incidence=False)
 # Schedule origin vertices remain even when terminal/shared-only starts
 # required no engine query. These are declared metadata constraints, never
 # invented origin-to-state incidences or claims that an unrun start was seen.
 declared=[];query_counts=Counter(u['family'] for u in units)
 for folder,run in sorted(_layout()[1].items()):
  identity_path=folder+'/identity.json';spec=_read(run['spec_path']);indices=defaultdict(list)
  for index,row in enumerate(spec['schedule']):indices[row['origin']].append(index)
  for name,schedule_indices in sorted(indices.items()):
   family=_family(identity_path,name);origin=('family',family);graph.find(origin);origin_q[family]
   entry=dict(family=family,declared_origin=name,run=folder,identity=_ref(identity_path),spec=_ref(run['spec_path']),
    schedule_indices=schedule_indices,actual_query_episodes=query_counts[family],actual_incidence=False)
   declared.append(entry)
   causes[origin]=dict(kind='declared schedule origin; no invented state incidence',**entry,
    optimizer_blocked=False,fit_blocked=False,arena_blocked=False,ancestral_training=False,
    acquisition_exposed=query_counts[family]>0)
 episode_refs=[]
 for u in sorted(units,key=lambda x:x['index']):
  task=tasks[u['index']]
  need(u['schema']=='ownership-resolution-actual-query-unit-v1' and u['task_sha256']==config.digest(task) and
   _same(u['source'],_ref(task['source'])) and u['family']==task['family'] and _same(u['activity'],stage.ZERO),
   'Projection unit provenance differs')
  origin=('family',u['family']);graph.find(origin)
  causes.setdefault(origin,dict(kind='actual declared engineering acquisition origin',declared_origin=u['declared_origin'],
   run=u['run'],identity=u['identity'],spec=u['spec'],optimizer_blocked=False,fit_blocked=False,arena_blocked=False,
   ancestral_training=False,acquisition_exposed=True,actual_incidence=True))
  for record_index,(q,first,rawkey) in enumerate(u['owned_records']):
   graph.join(origin,('quotient',q));origin_q[u['family']].add(q)
   edge_sources[u['family']].setdefault(q,[u['index'],record_index])
  episode_refs.append({k:u[k] for k in ('index','episode','family','source','decision_id','event_sha256','event_count','owned_event_count','shared_event_count','coverage_proof','coverage_mode')})
 grouped=defaultdict(lambda:dict(families=[],quotients=[],prior_components=[],prior_engineering_components=[],marker_causes=[]))
 for (kind,value) in graph.p:
  group=grouped[graph.find((kind,value))]
  if kind=='family':group['families'].append(value)
  elif kind=='quotient':group['quotients'].append(value)
  elif kind=='prior_component':group['prior_components'].append(value)
  else:group['prior_engineering_components'].append(value)
 for node,cause in causes.items():grouped[graph.find(node)]['marker_causes'].append(cause)
 components={};family_map={};qmap={};remap={}
 for body in grouped.values():
  for k in ('families','quotients','prior_components','prior_engineering_components'):body[k].sort()
  body['marker_causes'].sort(key=config.canonical)
  body.update({f:any(c[f] for c in body['marker_causes']) for f in flags})
  cid=config.digest(body);components[cid]=body
  for f in body['families']:family_map[f]=cid
  for q in body['quotients']:qmap[q]=cid
  for previous in body['prior_components']:remap[previous]=cid
 return dict(schema='ownership-resolution-known-provenance-overlay-v1',inputs=input_pins(),
  components=components,family_components=family_map,quotient_components=qmap,prior_component_remap=remap,
  protected_quotients=sorted(qmap),origin_quotients={f:sorted(qs) for f,qs in sorted(origin_q.items())},
  origin_incidence_witnesses={f:[[q,*edge_sources[f][q]] for q in sorted(qs)] for f,qs in sorted(origin_q.items())},
  actual_episode_units=episode_refs,actual_query_episodes=len(units),actual_declared_origins=len(origin_q),
  declared_schedule_origins=declared,zero_query_schedule_origins=sum(x['actual_query_episodes']==0 for x in declared),
  owned_incidence_observations=sum(len(u['owned_records']) for u in units),
  recorded_events=sum(u['event_count'] for u in units),shared_events=sum(u['shared_event_count'] for u in units),
  low_states_shared_and_raw_linked=True,upper_band_retained=True,no_cartesian_incidence_invention=True,
  graph_scope='certified legacy and old saved-engineering component constraints plus all692accepted actual latest query episodes; population constraints are joined separately',
  source_complete_for_declared_persisted=True,ancestry_unknown=True,historical_provenance_complete=False,
  historically_unseen_claim=False,missing_attempts=_inventory_document()['missing_attempts'],
  historical_failures_permanently_quarantined=True,optimizer_use_inferred_from_engineering_exposure=False,
  models_fixed=True,policy_fixed=True,activity=stage.ZERO,training_authorized=False,production_evaluation_admitted=False)

def final_projection(units):return stage.normalized(_projection_document(units))
def validate_projection(report,units):
 # Standalone validation also verifies each finite actual source. It does not
 # depend on the producer having validated units before passing them here.
 for result in units:
  index=result.get('index');need(type(index) is int and 0<=index<len(_layout()[0]),'Projection unit index invalid')
  need(_same(result,stage.normalized(_unit_document(_layout()[0][index]))),'Projection contains a resealed invalid unit')
 need(_same(report,stage.normalized(_projection_document(units))),'Known-query graph projection differs')
 return True
