"""Whole-origin ownership closure and outcome-blind joint reservations.

Accepted components are constraints, never invented family-to-quotient edges.
Actual traces contribute every observed high quotient, including rejected and
capped acquisitions. Shared low states remain linked through lossless archives.
"""
from collections import Counter,defaultdict
import argparse,base64,hashlib,json,struct,sys,time,zlib
from pathlib import Path
from . import config,footprint_screen_common as s

ROOT=s.ROOT
JOINT=ROOT/'build/ownership-resolution-v1/production/units/000695.json'
POPULATION=ROOT/'build/ownership-implementation-v1/reservations/production-v2/selection.json'
OLD=ROOT/'build/ordinary-epoch-pilot-v1/jobs/labels'
SCHEMA=s.SCHEMA+'-reservations'
DEFAULT=s.FOLDER/'reservations'
FIXTURE_BASE=s.FOLDER/'fixtures/reservation-recovery'
HOOKS=('before_catalog','after_catalog')
SOURCES=('congklak/footprint_screen_reservations.py','congklak/footprint_screen_common.py','congklak/config.py',
         'tools/reserve-footprint-screen.py','congklak/footprint_screen_graph.py')
PACK=struct.Struct('<16sBBBBhh');OWNED=struct.Struct('<15sQ16sB')
OP_NAMES=('decode','normalize_in','normalize_out','legal','terminal_test','margin_eval','move_in','move_out',
 'alpha_node','alpha_tt_probe','alpha_tt_hit','alpha_tt_slot','relay_node','relay_tt_probe','relay_tt_hit','relay_tt_slot',
 'relay_eval','relay_cache_probe','relay_cache_hit','model_eval','model_forward','feature_input','feature_move_in',
 'feature_move_out','sidecar_eval','sidecar_branch','sidecar_bypass','puct_node','puct_select','puct_expand',
 'played_input','played_output','alpha_pv_node','relay_pv_node','model_done','sidecar_done','alpha_root_node')
_PRIOR_CACHE={}
need,read,sha=s.need,s.read,s.sha

def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return dict(path=s.relative(path),sha256=sha(path))
def _old_read(path):
    need(s.prior_inputs().get(s.relative(path))==sha(path),'Held source changed: '+s.relative(path))
    return read(path)
def quotient(cells,actor):
    need(len(cells)==16 and sum(cells)==98 and actor in (0,1),'Complete98 state geometry differs')
    return min(cells[:7]+cells[8:15]+bytes([actor]),cells[8:15]+cells[:7]+bytes([1-actor]))
def high_key(q):
    need(type(q) is str and len(q)==30 and q==q.lower(),'Canonical high quotient width differs')
    try:raw=bytes.fromhex(q)
    except ValueError as error:raise ValueError('Malformed high quotient') from error
    need(len(raw)==15 and raw[14] in (0,1) and 25<=sum(raw[:14])<=98 and
         raw==min(raw,raw[7:14]+raw[:7]+bytes([1-raw[14]])),'Shared/noncanonical quotient cannot connect owned graph')
    return q

class Union:
    def __init__(self):self.p={}
    def find(self,a):
        self.p.setdefault(a,a);r=a
        while self.p[r]!=r:r=self.p[r]
        while self.p[a]!=a:a,self.p[a]=self.p[a],r
        return r
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)

def historical_sources():
    return [JOINT,POPULATION,OLD/'identity.json',OLD/'catalog.json',OLD/'report.json',
        *[OLD/'archives'/f'query-{i:06d}.json' for i in range(4)],OLD/'failed-attempts/query-000004.json']

def input_pins():
    files=historical_sources()
    for p in files:need(s.prior_inputs().get(s.relative(p))==sha(p),'Historical closure source changed')
    return s.common.pins([*files,s.PRIOR,*[ROOT/p for p in SOURCES]])

def _historical_footprint(path,index,identity):
    body=_old_read(path);answer=body['answer'];trace=answer['trace'];cert=answer['fast_certificate']
    meta=identity['schedule'][index]['meta'];state=meta['state'];origin=meta['origin']
    need(body['identity_sha256']==sha(OLD/'identity.json') and body['decision_id']==f'query/{index:06d}' and
         body['origin']==origin and body['query_kind']==('teacher' if index%2==0 else 'features') and
         _same(trace['binding']['root_state'],state),'Historical query/root/origin binding differs')
    need(trace['schema']=='complete-state-ownership-trace-v1' and trace['encoding']=='base64/zlib/packed24/little-endian' and
         trace['event_bytes']==24 and type(trace['event_count']) is int and
         0<=trace['event_count']<=trace['capacity_events']==16000000,'Historical trace format/capacity differs')
    compressed=base64.b64decode(trace['data'],validate=True);decoder=zlib.decompressobj()
    raw=decoder.decompress(compressed,trace['event_count']*24+1)
    need(len(compressed)==trace['compressed_bytes'] and decoder.eof and not decoder.unused_data and
         not decoder.unconsumed_tail,'Historical trace compression boundary differs')
    need(type(trace['event_count']) is int and trace['attempted_events']==trace['event_count'] and
         trace['overflow'] is False and trace['invalid'] is False and trace['truncated'] is False and
         len(raw)==24*trace['event_count'] and hashlib.sha256(raw).hexdigest()==trace['event_sha256']==cert['event_sha256'],
         'Historical query has an unknown/invalid raw tail')
    expected={};counts=Counter();low=high=0
    for i,(cells,actor,op,scope,reserved,action,depth) in enumerate(PACK.iter_unpack(raw)):
        need(sum(cells)==98 and actor in (0,1) and 1<=op<=37 and scope in (1,2,3,4) and
             reserved==0 and action>=-1 and depth>=-1,'Historical raw event geometry differs')
        counts[OP_NAMES[op-1]]+=1
        if 98-cells[7]-cells[15]>=25:
            high+=1;q=quotient(cells,actor);expected.setdefault(q,OWNED.pack(q,i,cells,actor))
        else:low+=1
    packed=base64.b64decode(answer['fast_owned_high'],validate=True)
    need(packed==b''.join(v for _,v in sorted(expected.items())) and len(expected)==cert['owned_count'] and
         hashlib.sha256(packed).hexdigest()==cert['owned_sha256'] and high==cert['owned_occurrences'] and
         _same({name:counts[name] for name in OP_NAMES},trace['operation_counts']),
         'Historical complete high footprint differs from raw source')
    return dict(index=index,family=meta['family'],origin=origin,state_key=meta['state_key'],source_ref=_ref(path),
        quotients=[q.hex() for q in sorted(expected)],event_count=trace['event_count'],owned_occurrences=high,
        shared_occurrences=low,raw_sha256=trace['event_sha256'],owned_sha256=cert['owned_sha256'],
        rejected_attempt=index==4,complete_footprint=True,quality_values_read=False)

def _derive_prior():
    pins=input_pins();joint=_old_read(JOINT)['result'];identity=_old_read(OLD/'identity.json')
    report=_old_read(OLD/'report.json');_old_read(OLD/'catalog.json')
    need(report['state']=='held' and report['queries']==4 and report['failed_returned_attempts']==1,
         'Previous actual acquisition count differs')
    graph=Union();flags={};families=joint['joint_family_components'];qmap=joint['joint_quotient_components']
    for cid,c in joint['joint_components'].items():
        graph.find(('prior',cid));flags[cid]=dict(known=bool(c['known_components']),ancestral=bool(c['ancestral_training']))
    for family,cid in families.items():graph.join(('prior',cid),('family',family))
    for q,cid in qmap.items():graph.join(('prior',cid),('quotient',q))
    support=joint['future_support']['roots'];need(len(support)==192,'Ordinary support count changed')
    support_ids={r['joint_component'] for r in support};quarantine=set(joint['whole_quarantine_joint_components'])
    previous_roles={r['joint_component'] for r in joint['conditional_role_revalidation']}
    footprints=[_historical_footprint(OLD/'archives'/f'query-{i:06d}.json',i,identity) for i in range(4)]
    footprints.append(_historical_footprint(OLD/'failed-attempts/query-000004.json',4,identity))
    need(len(footprints[-1]['quotients'])==6669,'The entire rejected6669-high footprint must remain')
    for row in footprints:
        family=('family',row['family']);graph.join(family,('actual_origin',row['origin']))
        for q in row['quotients']:graph.join(family,('quotient',q))
    groups=defaultdict(list)
    for node in graph.p:groups[graph.find(node)].append(node)
    ids={root:config.digest(dict(schema=SCHEMA+'-prior-component',nodes=sorted(nodes))) for root,nodes in groups.items()}
    components={};remap={}
    for root,nodes in groups.items():
        old=sorted(n[1] for n in nodes if n[0]=='prior');cid=ids[root];reasons=[]
        if any(flags[c]['known'] or flags[c]['ancestral'] for c in old):reasons.append('known historical/ancestral component')
        if support_ids.intersection(old):reasons.append('current192ordinary training support')
        if quarantine.intersection(old):reasons.append('whole320-root quarantine')
        if previous_roles.intersection(old):reasons.append('previous fixed evaluation reservations')
        if any(n[0]=='actual_origin' for n in nodes):reasons.append('actual previous query exposure')
        components[cid]=dict(prior_components=old,evaluation_blocked=bool(reasons),reasons=reasons,
            contains_training_support=bool(support_ids.intersection(old)),contains_quarantine=bool(quarantine.intersection(old)))
        for old_cid in old:remap[old_cid]=cid
    result=dict(schema=SCHEMA+'-prior-closure',components=components,
        family_components={f:ids[graph.find(('family',f))] for f in families},
        quotient_components={n[1]:ids[graph.find(n)] for n in graph.p if n[0]=='quotient'},prior_component_remap=remap,
        previous_actual_footprints=footprints,all_four_committed_queries_retained=True,rejected_high_quotients_retained=6669,
        old_population_components_retained=16383,component_constraints_are_not_individual_incidence_edges=True,
        training_support=support,quarantine_components=sorted({remap[c] for c in quarantine}),
        metadata_prior_evaluation_components=sorted(previous_roles),input_pins=pins,
        inherited_unknown_background=True,historically_unseen=False,old_pilot_stays_held=True,
        training_authorized=False,models_fixed=True,policy_fixed=True)
    need(input_pins()==pins,'Prior closure source changed during pure reconstruction')
    return result

def prior_closure():
    pins=input_pins();key=config.digest(pins)
    if key not in _PRIOR_CACHE:
        _PRIOR_CACHE.clear();_PRIOR_CACHE[key]=config.canonical(_derive_prior()).encode()
    return json.loads(_PRIOR_CACHE[key])

def _rank(salt,value):return hashlib.sha256(config.canonical(dict(salt=salt,value=value)).encode()).hexdigest()

def _matching(eligible,components,cells):
    slots=[]
    for cell in cells:
        for role,count in (('fit',16),('arena',6 if '/ordinary/' in cell else 0)):
            slots.extend((role,cell,i) for i in range(count))
    slots.sort(key=lambda x:_rank('footprint-screen-v1/joint-slots/20261017',x))
    by_cell=defaultdict(set)
    for row in eligible:by_cell[row['cell']].add(components[row['id']])
    choices={slot:sorted(by_cell[slot[1]],key=lambda c:_rank('footprint-screen-v1/slot-component/20261017',[slot,c])) for slot in slots}
    owner={};assigned={}
    def augment(slot,seen):
        for cid in choices[slot]:
            if cid in seen:continue
            seen.add(cid)
            if cid not in owner or augment(owner[cid],seen):owner[cid]=slot;assigned[slot]=cid;return True
        return False
    for slot in slots:augment(slot,set())
    # The component owner map is authoritative; earlier assignment entries may
    # have been displaced while an augmenting path was constructed.
    assigned={slot:cid for cid,slot in owner.items()}
    return slots,assigned,choices

def _matching_shortage_witness(slots,assigned,choices):
    """Alternating Hall witness, including shortages caused across strata."""
    owner={cid:slot for slot,cid in assigned.items()};left=set(slot for slot in slots if slot not in assigned)
    right=set();todo=list(left)
    while todo:
        slot=todo.pop()
        for cid in choices[slot]:
            if cid in right:continue
            right.add(cid)
            if cid in owner and owner[cid] not in left:left.add(owner[cid]);todo.append(owner[cid])
    return dict(unfilled_slots=len(slots)-len(assigned),reachable_slots=sorted(left),
        reachable_components=sorted(right),deficiency=len(left)-len(right))

def _acquired_graph(prior,records):
    """Join all actual footprints before eligibility, retaining capped roots."""
    graph=Union()
    for cid in prior['components']:graph.find(('prior',cid))
    for f,cid in prior['family_components'].items():graph.join(('prior',cid),('family',f))
    for q,cid in prior['quotient_components'].items():graph.join(('prior',cid),('quotient',high_key(q)))
    for root,result in records:
        for q in result['quotients']:graph.join(('family',root['family']),('quotient',high_key(q)))
        graph.join(('family',root['family']),('actual_origin',root['origin']))
    groups=defaultdict(list)
    for n in graph.p:groups[graph.find(n)].append(n)
    ids={r:config.digest(dict(schema=SCHEMA+'-acquired-component',nodes=sorted(ns))) for r,ns in groups.items()}
    summaries={}
    for r,nodes in groups.items():
        inherited=sorted(n[1] for n in nodes if n[0]=='prior')
        blocked=[c for c in inherited if prior['components'][c]['evaluation_blocked']]
        acquisition=any(n[0]=='actual_origin' for n in nodes)
        summaries[ids[r]]=dict(prior_components=inherited,evaluation_blocked=bool(blocked),blocking_prior_components=blocked,
            acquisition_exposed=acquisition,acquired_origins_optimizer_blocked=acquisition,
            acquisition_is_optimizer_use=False,support192_permission_not_reassessed=True)
    return dict(components=summaries,
        family_components={n[1]:ids[graph.find(n)] for n in graph.p if n[0]=='family'},
        quotient_components={n[1]:ids[graph.find(n)] for n in graph.p if n[0]=='quotient'})

def scope(output=DEFAULT,fixture=None):
    path=s.path_of(output)
    actual=path!=DEFAULT
    need(fixture is None or type(fixture) is bool and fixture is actual,'Reservation fixture treatment changed')
    need(path==DEFAULT or path.parent==FIXTURE_BASE and path.name in HOOKS,'Unregistered reservation publication scope')
    return path,actual

def _records(pool,snapshot):
    need(snapshot['schema']==s.SCHEMA+'-completed-snapshot' and snapshot['pool_sha256']==sha(s.FOLDER/'pool.json') and
         len(snapshot['roots'])==512,'Acquisition snapshot is not the complete fixed pool')
    records=[]
    for root,result in zip(pool['roots'],snapshot['roots']):
        need(result['id']==root['id'] and type(result['index']) is int and result['index']==root['index'] and
             type(result['teacher_complete']) is bool and type(result['features_complete']) is bool and
             result['trace_complete'] is True and result['hold'] is False,
             'Unknown/partial/misbound acquisition cannot become a reservation')
        need(type(result['quotients']) is list and result['quotients']==sorted(set(result['quotients'])) and
             bool(result['quotients']),'Complete acquired high-footprint inventory is missing')
        for q in result['quotients']:high_key(q)
        need(root['metadata_root_quotient'] in result['quotients'],'Acquisition footprint omits its root')
        need(type(result['source_refs']) is list and len(result['source_refs'])>=2 and
             len({ref['path'] for ref in result['source_refs']})==len(result['source_refs']),
             'Both actual teacher and feature archives must be retained')
        for ref in result['source_refs']:
            need(set(ref)=={'path','sha256'} and sha(s.path_of(ROOT/ref['path']))==ref['sha256'],
                 'Acquisition source reference changed')
        records.append((root,result))
    s.common.current(snapshot['inputs'])
    return records

def _audit(snapshot,audit):
    if audit is None:return False
    from . import footprint_screen_review as review
    receipt=review.completed()
    need(_same(audit,receipt),'Final independent review receipt differs from its immutable completed journal')
    report=receipt['report']
    need(receipt['publication_gap'] is False and receipt['catalog_ref'] is not None and
         report['schema']==s.SCHEMA+'-independent-raw-review-report' and report['passed'] is True and
         report['independent_of_implementation'] is True and type(report['acquired_roots']) is int and
         report['acquired_roots']==512 and type(report['production_query_count']) is int and
         report['production_query_count']==1024 and report['pool_sha256']==sha(s.FOLDER/'pool.json') and
         report['snapshot_sha256']==config.digest(snapshot) and _same(report['source_inputs'],snapshot['inputs']),
         'Independent full raw review does not bind the complete current acquisition')
    return True

def _graph_identity(snapshot,audit,prior,path,fixture):
    from . import footprint_screen_graph as graph
    return dict(schema=graph.SCHEMA+'-identity',publication=_identity(snapshot,audit,path,fixture),
        prior_inputs_sha256=config.digest(prior['input_pins']),source_sha256=sha(ROOT/'congklak/footprint_screen_graph.py'))

def expected(snapshot,audit=None,output=DEFAULT,fixture=False):
    from . import footprint_screen_data as data,footprint_screen_graph as graph
    path,_=scope(output,fixture);pool=data.membership();prior=prior_closure();records=_records(pool,snapshot)
    closed=graph.load(path,prior,records,_graph_identity(snapshot,audit,prior,path,fixture))
    summaries=closed['components'];root_components={};eligible=[];dispositions=[]
    for root,result in records:
        cid=closed['family_components'][root['family']];root_components[root['id']]=cid;reasons=[]
        if summaries[cid]['evaluation_blocked']:reasons.append('complete acquired component touches protected prior ownership')
        if not result['teacher_complete']:reasons.append('H8 target did not complete under fixed cap')
        if not result['features_complete']:reasons.append('native127 features did not complete')
        dispositions.append(dict(id=root['id'],index=root['index'],cell=root['cell'],component=cid,eligible=not reasons,reasons=reasons,
            teacher_complete=result['teacher_complete'],features_complete=result['features_complete'],full_footprint_retained=True,
            acquisition_exposed=True,optimizer_blocked=True,optimizer_exclusion_is_not_optimizer_use=True,
            acquired_origin_scope='evaluation/development only; never newly admitted optimizer support'))
        if not reasons:eligible.append(root)
    slots,assigned,choices=_matching(eligible,root_components,data.CELLS);roles=dict(fit=[],arena=[],cases=[])
    for slot,cid in assigned.items():
        role,cell,_=slot;options=[r for r in eligible if r['cell']==cell and root_components[r['id']]==cid]
        root=min(options,key=lambda r:_rank('footprint-screen-v1/role-root/20261017',[role,r['id'],r['state_key']]))
        roles[role].append(dict(root,component=cid,role=role))
    for role in ('fit','arena'):roles[role].sort(key=lambda r:(data.CELLS.index(r['cell']),r['index']))
    if len(roles['fit'])==128:
        for cell in data.CELLS:
            rows=[r for r in roles['fit'] if r['cell']==cell]
            roles['cases'].extend(dict(r,role='cases',fit_root_id=r['id']) for r in sorted(rows,
                key=lambda r:_rank('footprint-screen-v1/case-subset/20261017',[r['id'],r['state_key']]))[:4])
    shortage={role:{cell:sum(slot[0]==role and slot[1]==cell and slot not in assigned for slot in slots) for cell in data.CELLS}
              for role in ('fit','arena')}
    audit_passed=_audit(snapshot,audit)
    complete=len(assigned)==152 and len(roles['cases'])==32
    return dict(schema=SCHEMA+'-report',metadata_quotas_satisfied=complete,independent_full_raw_audit_passed=audit_passed,
        reservation_admission=complete and audit_passed and not fixture,roles=roles,root_dispositions=dispositions,shortage=shortage,
        fixture=fixture,output=s.relative(path),
        acquired_roots=512,all_acquired_footprints_retained=True,selection_after_complete_batch_only=True,
        selected_distinct_components=len({r['component'] for role in ('fit','arena') for r in roles[role]}),
        components=summaries,family_components=closed['family_components'],quotient_component_map=closed['quotient_component_map'],
        disk_backed_full_incidence=True,owner_union_find_only=True,
        matching_shortage_witness=_matching_shortage_witness(slots,assigned,choices),
        support192_preserved=prior['training_support'],old320_quarantine_preserved=True,
        support192_current_components={r['state_key']:closed['family_components'][r['family']] for r in prior['training_support']},
        support192_cells=dict(Counter(r['cell'] for r in prior['training_support'])),
        quarantine_prior_components=prior['quarantine_components'],
        previous_actual_footprints=prior['previous_actual_footprints'],rejected_old_high_quotients_retained=6669,
        snapshot_sha256=config.digest(snapshot),pool_sha256=sha(s.FOLDER/'pool.json'),snapshot_inputs=snapshot['inputs'],
        audit=audit,prior_inputs=prior['input_pins'],source_inputs=s.common.pins([ROOT/p for p in SOURCES]),
        quality_values_used_for_selection=False,models_fixed=True,policy_fixed=True,training_authorized=False,
        historically_unseen=False,inherited_unknown_background=True,old_pilot_stays_held=True)

def _identity(snapshot,audit,path,fixture):
    return dict(schema=SCHEMA+'-identity',output=s.relative(path),fixture=fixture,plan_sha256=sha(s.PLAN),
        pool_ref=_ref(s.FOLDER/'pool.json'),snapshot_sha256=config.digest(snapshot),audit_sha256=config.digest(audit),
        sources=s.common.pins(SOURCES),models_fixed=True,policy_fixed=True,training_authorized=False)

def _certificate(path,report):
    return dict(schema=SCHEMA+'-certificate',report_ref=_ref(path/'report.json'),identity_ref=_ref(path/'identity.json'),
        snapshot_sha256=report['snapshot_sha256'],pool_ref=_ref(s.FOLDER/'pool.json'),fixture=report['fixture'],
        reservation_admission=report['reservation_admission'])

def _catalog(path,report):
    return dict(schema=SCHEMA+'-catalog',identity_ref=_ref(path/'identity.json'),report_ref=_ref(path/'report.json'),
        certificate_ref=_ref(path/'certificate.json'),fixture=report['fixture'],
        reservation_admission=report['reservation_admission'],training_authorized=False)

def prepare(snapshot,audit,output=DEFAULT,fixture=False,pause=lambda point:None,should_pause=lambda:False):
    path,_=scope(output,fixture);s.admit_plan()
    from . import footprint_screen_worker as worker,footprint_screen_data as data,footprint_screen_graph as graph
    worker.validate_snapshot(snapshot)
    need(_audit(snapshot,audit),'Independent full raw replay is required before sealing')
    prior=prior_closure();records=_records(data.membership(),snapshot)
    s.atomic(path/'identity.json',_identity(snapshot,audit,path,fixture))
    graph.build(path,prior,records,_graph_identity(snapshot,audit,prior,path,fixture),should_pause)
    report=expected(snapshot,audit,path,fixture)
    s.atomic(path/'report.json',report)
    s.atomic(path/'certificate.json',_certificate(path,report));pause('before_catalog')
    s.atomic(path/'catalog.json',_catalog(path,report));pause('after_catalog')
    return dict(report=report,catalog=read(path/'catalog.json'),publication_gap=False)

def validate(report,snapshot,audit,output=DEFAULT,fixture=False):
    need(_same(report,expected(snapshot,audit,output,fixture)),'Whole-footprint reservations differ from complete sources');return True

def completed(output=DEFAULT):
    """Strict pure reader: no publication, writer, DLL, encoder or engine."""
    from . import footprint_screen_worker as worker,footprint_screen_review as review
    s.admit_plan();path,fixture=scope(output);report=read(path/'report.json');certificate=read(path/'certificate.json')
    snapshot=worker.completed_snapshot();audit=review.completed();worker.validate_snapshot(snapshot)
    need(_same(read(path/'identity.json'),_identity(snapshot,audit,path,fixture)),'Reservation source/scope identity changed')
    validate(report,snapshot,audit,path,fixture)
    need(_same(certificate,_certificate(path,report)),'Final reservation certificate/source binding differs')
    catalog=_catalog(path,report)
    if (path/'catalog.json').exists():need(_same(read(path/'catalog.json'),catalog),'Final reservation catalog differs')
    return dict(schema=SCHEMA+'-completion',report=report,report_ref=_ref(path/'report.json'),
        certificate_ref=_ref(path/'certificate.json'),reservation_admission=report['reservation_admission'],
        catalog=catalog,publication_gap=not(path/'catalog.json').exists(),
        training_authorized=False,models_fixed=True,policy_fixed=True)

def has_completion(output=DEFAULT):
    path,_=scope(output)
    if not(path/'report.json').exists() or not(path/'certificate.json').exists():
        need(not(path/'catalog.json').exists(),'A catalog without its exact report/certificate is corrupt');return False
    completed(path);return True

def publish_completed(output=DEFAULT):
    done=completed(output)
    if done['publication_gap']:
        from .valuepilot_reservations import Writer
        path,_=scope(output)
        with Writer(path):s.atomic(path/'catalog.json',done['catalog'])
        done['publication_gap']=False
    return done

def preflight(output=DEFAULT,fixture=False):
    path,_=scope(output,fixture);s.admit_plan()
    from . import footprint_screen_worker as worker,footprint_screen_review as review
    snapshot=worker.completed_snapshot();worker.validate_snapshot(snapshot);audit=review.completed()
    if (path/'identity.json').exists():
        need(_same(read(path/'identity.json'),_identity(snapshot,audit,path,fixture)),'Existing reservation source identity differs')
    else:need(not path.exists() or not any(path.rglob('*')),'Unbound reservation evidence forbids overwrite')
    return dict(state='ready',fixture=fixture,acquired_roots=512,full_raw_review_passed=True)

def run(output=DEFAULT,fixture=False,lease_path=None,pause_at=None,pause_marker=None):
    path,_=scope(output,fixture)
    if has_completion(path):return publish_completed(path)
    from . import footprint_screen_worker as worker,footprint_screen_review as review,footprint_screen_resources as resources
    from .valuepilot_reservations import Writer
    need(lease_path is not None,'Reservation metadata publication requires its contained resource lease')
    resources.assert_worker_lease(lease_path);preflight(path,fixture)
    need(pause_at is None or fixture and pause_at in HOOKS and pause_marker is not None,'Undeclared publication pause')
    def pause(point):
        if point!=pause_at:return
        marker=s.path_of(pause_marker)
        need(marker.parent==path,'Publication pause marker outside its fixture')
        s.atomic(marker,dict(schema=SCHEMA+'-pause',point=point,identity_ref=_ref(path/'identity.json'),
            process=resources.process_identity(),argv=sys.argv,lease=str(Path(lease_path).resolve()),
            source_sha256=sha(__file__),new_queries=0,model_forwards=0,optimizer_steps=0))
        while True:time.sleep(.05)
    with Writer(path):
        resources.assert_worker_lease(lease_path)
        from . import footprint_screen_graph as graph
        try:return prepare(worker.completed_snapshot(),review.completed(),path,fixture,pause,
            lambda:resources.remaining_ns(lease_path)<30_000_000_000)
        except graph.Paused as stopped:
            return dict(report=dict(state='paused',reservation_admission=False,closure_cursor=stopped.cursor),
                catalog=None,publication_gap=False)

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','completed','publish'))
    parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--lease');parser.add_argument('--engineering-fixture',action='store_true')
    parser.add_argument('--pause-at');parser.add_argument('--pause-marker');args=parser.parse_args(argv)
    if args.command=='run':
        if has_completion(args.output):answer=publish_completed(args.output)
        else:
            from . import footprint_screen_resources as resources
            need(args.lease is not None,'Contained metadata lease missing');resources.bind_worker(args.lease)
            answer=run(args.output,args.engineering_fixture,args.lease,args.pause_at,args.pause_marker)
            resources.worker_complete(args.lease,dict(state=answer['report'].get('state','complete'),
                reservation_admission=answer['report']['reservation_admission']))
    elif args.command=='publish':answer=publish_completed(args.output)
    else:answer=completed(args.output)
    print(config.canonical(dict(state=answer['report'].get('state','complete'),publication_gap=answer['publication_gap'],
        reservation_admission=answer['report']['reservation_admission'],training_authorized=False)))
    return answer
