"""Recoverable metadata-only custom reservations with pre-draw review binding.

No targets, features, model calls, searches, optimization or production games.
Every counter and legal child probe is retained, including rejected candidates.
"""
from collections import Counter,defaultdict
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import time
import uuid
from . import config,neurodiag_common as common
from . import provenance_population as population,provenance_reconstruction_v2 as graph_source
from . import provenance_reconstruction as provenance
from . import pilot
from . import engineering_provenance_supplement as supplement
from .native import Native,flatten
from .valuepilot_reservations import Writer

ROOT=common.ROOT
DEFAULT=ROOT/'build/ownership-implementation-v1/reservations/production-v2'
FOLDER=DEFAULT.parent
SCHEMA='prospective-custom-reservations-v2'
MATCHER_SCHEMA='prospective-custom-reservations-v1'
SOURCES=('congklak/prospective_reservations_v2.py','congklak/provenance_population.py',
    'congklak/provenance_reconstruction_v2.py','congklak/pilot.py','congklak/native.py',
    'congklak/config.py','congklak/neurodiag_data.py','build/congklak_native.dll',
    'rules/id-betawi-relay7-v0.json','tools/reserve-prospective-ownership-v2.py')
SOURCES=(*SOURCES,'congklak/engineering_provenance_supplement.py')
need,read,sha=common.require,common.read,common.sha
CHUNK=64


def publish(path,body):
    """Immutable publication of exact UTF-8/LF bytes on every platform."""
    path=Path(path);payload=(config.canonical(body)+'\n').encode('utf-8')
    if path.exists():
        need(path.read_bytes()==payload,'Immutable V2 reservation output differs: '+str(path))
        return body
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return body
QUOTA_BYTES=5*1024**3
RESERVE_BYTES=2*1024**3


def namespace_bytes(path):
    total=0
    def failed(error):raise error
    for folder,dirs,names in os.walk(path,onerror=failed):
        for name in names:
            candidate=Path(folder)/name
            try:info=candidate.stat()
            except FileNotFoundError:
                # The pinned immutable JSON helper uses UUID.tmp files; only
                # those declared temporary renames may disappear mid-scan.
                parts=name.rsplit('.',2)
                temporary=(len(parts)==3 and parts[-1]=='tmp' and len(parts[-2])==32 and
                           all(ch in '0123456789abcdef' for ch in parts[-2]))
                if temporary:continue
                raise
            if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total


def resource_guard(needed=0):
    FOLDER.mkdir(parents=True,exist_ok=True)
    need(namespace_bytes(FOLDER)+needed<=QUOTA_BYTES,'Reservation namespace5GiB quota reached')
    need(shutil.disk_usage(FOLDER).free>=RESERVE_BYTES+needed,'Reservation2GiB free-space reserve reached')


def matcher_spec():
    return dict(schema=MATCHER_SCHEMA+'-matcher',arena_salt=population.ARENA_SALT,fit_salt=population.FIT_SALT,
        rank='SHA256(ASCII(role_salt + / + state_key)), then state_key; literal one slash delimiter',
        cells=list(population.CELLS),slots='arena first, then fit; within each role CELLS order, then integer slot0..quota-1',
        arena_per_cell=population.specification()['roles']['arena_per_cell'],fit_per_cell=16,
        matching='Depth-first augmenting-path matching of slots to complete new-domain components, candidate options sorted by declared rank; each recursion visits each component once per slot',
        within_component='Only the candidate row chosen by deterministic augmenting path; entire origin component retained/reserved',
        cases='Within each cell sort the16selected fit rows by fit rank/state_key and take first4; cases remain fit-role subset',
        known_graph_policy='Reject every new component touching any known prior high-domain component, conservatively including unprotected prior metadata; do not erase those edges',
        rejected_candidate_bridges='All root/probe incidences, including mechanism mismatches, enter graph before matching; no first-fit component deletion',
        component_id='SHA256 canonical JSON of schema,families,quotients,prior_components,prior_graph_report_sha256',
        shortage='Emit empty role reservations with exact per-cell roots/components, requested quotas and matching failure; no fillers or smaller quota',
        support_reuse='Exactly current512 unchanged root identities; report320genuine existing conflicts separately; never override to permit training',
        no_training_admission=True)


def binding():
    """Pure declaration for the auditor; calling this constructs no position."""
    base=graph_source.DEFAULT
    need((base/'certificate.json').is_file(),'Corrected source graph must be certified before pre-draw review')
    pins=common.pins([*SOURCES,base/'certificate.json',base/'report.json',*provenance.MODELS])
    extension=supplement.completed()
    pins.update(extension['inputs'])
    pins.update(common.pins([supplement.DEFAULT/'certificate.json',supplement.DEFAULT/'report.json']))
    prior_failure_path=ROOT/'build/ownership-implementation-v1/audit/reservation-bytehash-failure.json'
    failure=read(prior_failure_path)
    need(failure['counter_draws']==128 and failure['native_action_probes']==571 and failure['actual_process_kill'] is True,
         'Preserved V1 metadata attempt receipt differs')
    failed_folder=ROOT/failure['folder']
    need(failed_folder.resolve().is_relative_to(FOLDER/'fixtures'),'Preserved V1 attempt source outside fixture namespace')
    pins.update(common.pins([prior_failure_path,failed_folder/'identity.json',failed_folder/'work.sqlite3',
                            *sorted((failed_folder/'chunks').glob('*.json'))]))
    pins.update(common.pins([ROOT/'build/ownership-implementation-v1/audit/worker-recovery-v2.py',
                            ROOT/'build/ownership-implementation-v1/audit/worker-recovery-v2.log']))
    return dict(schema=SCHEMA+'-pre-draw-binding',source_inputs=pins,
        population_specification=population.specification(),population_specification_sha256=config.digest(population.specification()),
        matcher=matcher_spec(),matcher_sha256=config.digest(matcher_spec()),
        graph_certificate_sha256=sha(base/'certificate.json'),graph_report_sha256=sha(base/'report.json'),
        engineering_supplement_certificate_sha256=sha(supplement.DEFAULT/'certificate.json'),
        engineering_supplement_report_sha256=sha(supplement.DEFAULT/'report.json'),
        additional_missing_attempt_ledger=[dict(attempt='worker before_decision_commit actual kill',
            source='build/ownership-implementation-v1/audit/worker-recovery-v2.py',
            qualification='Uncommitted first fixed128 H_v1 decision envelope was lost; same frozen schedule0/root/engine as saved root9/resumed evidence, expected deterministic duplicate, actual attempt not persisted',
            exact_actual_attempt_capture_claim=False,global_missing_trace_admission_hold_preserved=True)],
        conditional_metadata_acquisition_only=True,missing_material_engineering_trace_hold=True,
        complete_engineering_source_absence_claim=False,training_admission=False,evaluation_admission=False,
        rules_digest=config.RULES_DIGEST,classifier='pinned pilot._mechanism with a recording adapter; every legal apply captured',
        native_build_sha256=pins['build/congklak_native.dll'],candidate_constructor_calls=128,
        native_classifier_calls=128,new_version_constructor_calls=0,new_version_native_classifier_calls=0,
        prior_attempt_counter_draws=128,prior_attempt_native_action_probes=571,
        prior_attempt_failure_receipt=provenance.rel(prior_failure_path),
        retry_population='Exact unchanged deterministic constructor, seed, cells, counters, salts, matcher and case subset; prior128 observations retained, no fresh-origin claim',
        no_targets_or_models=True,
        resource_contract=dict(namespace_quota_bytes=QUOTA_BYTES,min_free_bytes=RESERVE_BYTES,
            chunk_counters=CHUNK,max_counters=population.MAX_DRAWS,
            enforced_private_memory_cap=False,enforced_wall_time_cap=False,
            description='Fixed finite counter/probe work and disk guards; this metadata producer does not claim arena90-minute/private-memory watchdog enforcement'))


def _path(path):
    path=Path(path).resolve();need(path.is_relative_to(FOLDER),'Reservation path outside new namespace');return path


def _readonly(path):return sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)


def prefix(path,identity):
    need((path/'work.sqlite3').exists(),'Reservation journal missing')
    with closing(_readonly(path/'work.sqlite3')) as db:
        need(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Reservation journal corrupt')
        meta=dict(db.execute('SELECT key,value FROM meta'))
        need(meta.get('identity_sha256')==sha(path/'identity.json'),'Reservation journal identity differs')
        count=int(meta['committed_chunks']);rows=list(db.execute('SELECT idx,name,digest,previous FROM chunks ORDER BY idx'))
    need(len(rows)==count,'Reservation committed high-water differs')
    out=[];previous=None
    for index,(idx,name,digest,ancestor) in enumerate(rows):
        need(idx==index and ancestor==previous and name==f'chunks/{index:06d}-{digest}.json','Reservation journal prefix gap')
        need(sha(path/name)==digest,'Reservation committed chunk missing/corrupt')
        chunk=read(path/name)
        need(chunk['schema']==SCHEMA+'-chunk' and chunk['index']==index and
             chunk['identity_sha256']==sha(path/'identity.json') and chunk['start_counter']==index*CHUNK,
             'Reservation chunk identity/counter differs')
        need(0<len(chunk['records'])<=CHUNK and [r['counter'] for r in chunk['records']]==list(range(index*CHUNK,index*CHUNK+len(chunk['records']))),
             'Reservation chunk counter coverage differs')
        out.extend(chunk['records']);previous=digest
    need(len(out)<=identity['max_draws'],'Reservation counter exceeds declared population')
    return out,rows


class Recorder:
    def __init__(self,native):self.native=native;self.facts=[];self.normalizations=[]
    def normalize(self,*args,**kwargs):
        answer=self.native.normalize(*args,**kwargs);self.normalizations.append(answer);return answer
    def apply(self,*args,**kwargs):
        answer=self.native.apply(*args,**kwargs);self.facts.append((kwargs['action'],answer));return answer


def candidate_record(counter,native):
    row=population.candidate(counter);state=row['state'];recorder=Recorder(native)
    active,stratum=pilot._mechanism(state,recorder)
    need(active==row['active_seeds'],'Population active classifier differs')
    normal=recorder.normalizations[0];need(normal.get('ok'),'Population root normalization invalid')
    probes=[]
    for action,answer in recorder.facts:
        need(answer.get('ok'),'Population classification action invalid')
        normalized=native.normalize(flatten(answer['position']),actor=answer['position']['controller'],total=98)
        need(normalized.get('ok'),'Population child normalization invalid')
        child=normalized['position'];provenance.domain.mass_transition(state,child)
        probes.append(dict(action=action,reason=answer['reason'],passes=answer['passes'],
                           state=child,raw_state_key=common.rawkey(child),quotient=provenance.metadata.quotient(child),
                           active_seeds=sum(map(sum,child['pits'])),terminal=normalized['terminal']))
    reasons=[]
    if normal['position']!=state or normal['terminal']:reasons.append('normalization changes root')
    if row['legal_mask'].bit_count()<2:reasons.append('not at least two legal choices')
    if abs(state['stores'][0]-state['stores'][1])>=active:reasons.append('mathematically decided store margin')
    if stratum!=row['cell'].rsplit('/actor',1)[0]:reasons.append('metadata mechanism differs from declared counter cell')
    return dict(row,actual_stratum=stratum,quotient=provenance.metadata.quotient(state),probes=probes,
        classifier_source='pilot._mechanism',metadata_eligible=not reasons,rejection_reasons=reasons,
        acquisition_exposed=True,model_or_outcome_inspected=False,native_action_probes=len(probes),
        native_normalizations=len(probes)+len(recorder.normalizations))


def rank(role,row):
    salt=population.ARENA_SALT if role=='arena' else population.FIT_SALT
    return hashlib.sha256((salt+'/'+row['state_key']).encode('ascii')).hexdigest(),row['state_key']


def select(records,prior,prior_report_sha):
    graph=provenance.Graph();prior_q=prior['quotient_components'];by_family={};incidences=0;low=0
    for row in records:
        family=row['family'];need(family not in by_family,'Generated origin repeated');by_family[family]=row
        graph.find(('family',family))
        states=[dict(quotient=row['quotient'],active_seeds=row['active_seeds'])]+row['probes']
        for observation in states:
            q=observation['quotient'];active=provenance.domain.quotient_active(q)
            need(active==observation['active_seeds'],'Generated observation seed domain differs');incidences+=1
            if active<25:low+=1;continue
            graph.join(('family',family),('quotient',q))
            if q in prior_q:graph.join(('quotient',q),('prior_component',prior_q[q]))
    groups=defaultdict(lambda:dict(families=[],quotients=[],prior_components=[]))
    for kind,value in graph.parent:
        groups[graph.find((kind,value))][{'family':'families','quotient':'quotients','prior_component':'prior_components'}[kind]].append(value)
    components={};lookup={}
    for body in groups.values():
        base={k:sorted(v) for k,v in body.items()};base.update(schema=MATCHER_SCHEMA+'-component',prior_graph_report_sha256=prior_report_sha)
        cid=config.digest(base);components[cid]=base
        for family in body['families']:lookup[family]=cid
    arena_quota=population.specification()['roles']['arena_per_cell']
    slots=[(role,cell,index) for role in ('arena','fit') for cell in population.CELLS
           for index in range(arena_quota[cell] if role=='arena' else 16)]
    options={};availability={};excluded=[]
    for role in ('arena','fit'):
        for cell in population.CELLS:
            values=[dict(row,new_domain_component=lookup[row['family']]) for row in records
                    if row['cell']==cell and row['metadata_eligible'] and not components[lookup[row['family']]]['prior_components']]
            values.sort(key=lambda row:rank(role,row));options[role,cell]=values
            availability[role+'/'+cell]=dict(roots=len(values),components=len({r['new_domain_component'] for r in values}),
                                           requested=arena_quota[cell] if role=='arena' else 16)
    assigned={};chosen={}
    def place(slot,seen):
        for row in options[slot[0],slot[1]]:
            cid=row['new_domain_component']
            if cid in seen:continue
            seen.add(cid)
            if cid not in assigned or place(assigned[cid],seen):assigned[cid]=slot;chosen[slot]=row;return True
        return False
    failure=None
    for slot in slots:
        if not place(slot,set()):failure=list(slot);break
    roles={'arena':[],'fit':[],'cases':[]}
    if failure is None:
        for slot in slots:roles[slot[0]].append(chosen[slot])
        for cell in population.CELLS:
            roles['cases'].extend(sorted((r for r in roles['fit'] if r['cell']==cell),key=lambda r:rank('fit',r))[:4])
        for role in roles:roles[role].sort(key=lambda r:r['state_key'])
        need(len(roles['fit'])==128 and len(roles['arena'])==24 and len(roles['cases'])==32,
             'Prospective exact role quota differs')
        need(len({r['new_domain_component'] for r in roles['fit']+roles['arena']})==152,'Prospective role components overlap')
    rejected=Counter(reason for row in records for reason in row['rejection_reasons'])
    rejected['known high-domain component']=sum(bool(components[lookup[row['family']]]['prior_components']) for row in records)
    selected_roles=defaultdict(list)
    for role,rows in roles.items():
        for row in rows:selected_roles[row['family']].append(role)
    counter_ledger=[]
    for row in records:
        cid=lookup[row['family']];known=components[cid]['prior_components'];assigned_roles=sorted(selected_roles[row['family']])
        reasons=list(row['rejection_reasons'])
        if known:reasons.append('known high-domain component')
        status=('selected_fit' if 'fit' in assigned_roles else 'selected_arena' if 'arena' in assigned_roles else
                'rejected_metadata_or_graph' if reasons else 'unselected_development_only')
        counter_ledger.append(dict(counter=row['counter'],family=row['family'],state_key=row['state_key'],
            quotient=row['quotient'],cell=row['cell'],actual_stratum=row.get('actual_stratum'),
            new_domain_component=cid,known_prior_components=known,selection_roles=assigned_roles,
            disposition=status,rejection_reasons=reasons,acquisition_exposed=True,
            analysis_or_outcome_exposed=False,optimizer_blocked=True,
            optimizer_reason='All generated origins are fit/arena/development-only; exact old512 support is the only permitted training pool',
            chunk_index=row['counter']//CHUNK,record_index=row['counter']%CHUNK))
    return dict(schema=MATCHER_SCHEMA+'-selection',metadata_reservations_complete=failure is None,
        status='conditional_metadata_reserved_training_and_evaluation_hold' if failure is None else 'HOLD_quota_shortage',
        roles=roles,availability=availability,matching_failure=failure,components=dict(sorted(components.items())),
        selected_components=sorted({r['new_domain_component'] for r in roles['fit']+roles['arena']}),
        counter_draws=len(records),native_action_probes=sum(r['native_action_probes'] for r in records),
        native_normalizations=sum(r['native_normalizations'] for r in records),
        actual_observation_incidences=incidences,shared_observation_incidences=low,
        rejected_reason_counts=dict(sorted(rejected.items())),all_rejections_retained=True,
        counter_disposition_ledger=sorted(counter_ledger,key=lambda row:row['counter']),
        frozen_support_reuse=prior['candidate_pools']['support_reuse'] if 'candidate_pools' in prior else [],
        prior_support_roots=512,prior_support_roots_with_genuine_exclusion=prior['support_roots_with_genuine_exclusion'],
        support_pool_changed=False,training_admission=False,evaluation_admission=False,
        conditional_metadata_acquisition_only=True,missing_material_engineering_trace_hold=True,
        complete_engineering_source_absence_claim=False,prior_support_permission_not_reassessed=True,
        historical_source_hold_preserved=True,models_fixed=True,policy_fixed=True,
        optimizer_steps=0,optimizer_draws=0,searches=0,new_games=0,new_teacher_labels=0,model_forwards=0,feature_simulations=0,
        target_values_read=0,opening_reachability_claimed=False,historically_unseen_claim=False)


def _review(path):
    review=read(path);declaration=binding()
    need(review.get('schema')==SCHEMA+'-independent-pre-draw-review' and
         review.get('approved_before_first_draw') is True and
         review.get('independent_of_implementation') is True and
         review.get('binding_sha256')==config.digest(declaration) and
         review.get('unresolved_findings')==[],'Independent pre-draw review does not bind this exact population/matcher/graph/build')
    common.current(declaration['source_inputs']);return declaration


def prepare(path,review_path,fixture_draws=None):
    declaration=_review(review_path)
    need(fixture_draws is None or (path.is_relative_to(FOLDER/'fixtures') and type(fixture_draws) is int and
         64<=fixture_draws<=256 and fixture_draws%CHUNK==0),'Invalid reservation recovery fixture')
    need(fixture_draws is not None or path==DEFAULT.resolve(),'Full declared population must use production namespace')
    identity=dict(schema=SCHEMA+'-identity',binding=declaration,binding_sha256=config.digest(declaration),
        review=dict(path=provenance.rel(review_path),sha256=sha(review_path)),max_draws=population.MAX_DRAWS if fixture_draws is None else fixture_draws,
        fixture_draws=fixture_draws,engineering_fixture=fixture_draws is not None,chunk_size=CHUNK,model_and_policy_fixed=True)
    if (path/'identity.json').exists():
        need(read(path/'identity.json')==identity,'Reservation resume identity changed')
        if (path/'work.sqlite3').exists():
            with closing(_readonly(path/'work.sqlite3')) as db:
                tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables:return identity
        need(not list((path/'chunks').glob('*.json')),'Missing initialized journal with observed candidate chunks')
    else:
        need(not (path/'work.sqlite3').exists() and not list((path/'chunks').glob('*.json')),'Reservation journal cannot be recreated over prior work')
        path.mkdir(parents=True,exist_ok=True);publish(path/'identity.json',identity)
    with closing(sqlite3.connect(path/'work.sqlite3')) as db:
        db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
        db.execute('BEGIN IMMEDIATE')
        db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('CREATE TABLE chunks(idx INTEGER PRIMARY KEY,name TEXT NOT NULL,digest TEXT NOT NULL,previous TEXT)')
        db.executemany('INSERT INTO meta VALUES(?,?)',[('identity_sha256',sha(path/'identity.json')),('committed_chunks','0')]);db.commit()
    return identity


def pause(path,point,wanted,marker,index):
    if point!=wanted:return
    need(path.is_relative_to(FOLDER/'fixtures') and marker is not None,'Reservation pause hooks are fixtures only')
    publish(marker,dict(point=point,index=index,pid=os.getpid()))
    while True:time.sleep(.1)


def completed(path=DEFAULT):
    path=_path(path);certificate=read(path/'certificate.json');identity=read(path/'identity.json')
    need(certificate['schema']==SCHEMA+'-certificate' and certificate['passed'] is True and
         certificate['identity_sha256']==sha(path/'identity.json') and
         certificate['engineering_fixture']==identity['engineering_fixture']==(identity['fixture_draws'] is not None) and
         certificate['population_scan_complete'] is True and certificate['production_training_admission'] is False and
         certificate['model_and_policy_fixed'] is True and identity['model_and_policy_fixed'] is True,
         'Reservation certificate identity differs')
    need(identity['schema']==SCHEMA+'-identity' and identity['chunk_size']==CHUNK,
         'Reservation identity schema/chunk scope differs')
    fixture=identity['fixture_draws']
    need((fixture is None and path==DEFAULT.resolve() and identity['max_draws']==population.MAX_DRAWS) or
         (type(fixture) is int and path.is_relative_to(FOLDER/'fixtures') and
          64<=fixture<=256 and fixture%CHUNK==0 and identity['max_draws']==fixture),
         'Reservation production/fixture counter scope differs')
    common.current(identity['binding']['source_inputs'])
    need(sha(ROOT/identity['review']['path'])==identity['review']['sha256'],'Reservation independent review changed')
    need(_review(ROOT/identity['review']['path'])==identity['binding'] and
         identity['binding_sha256']==config.digest(identity['binding']),
         'Reservation prospective binding/review changed')
    need(sha(path/'work.sqlite3')==certificate['journal_sha256'],'Reservation committed high-water changed')
    records,rows=prefix(path,identity)
    need(len(records)==identity['max_draws'] and len(rows)==certificate['committed_chunks'],'Reservation complete counter prefix differs')
    need(sha(path/'selection.json')==certificate['selection_sha256'],'Reservation selection changed')
    prior=extended_prior();selection=read(path/'selection.json')
    need(selection==json.loads(config.canonical(select(records,prior,identity['binding']['graph_report_sha256']))),
         'Reservation joint matching semantics differ')
    return dict(identity=identity,certificate=certificate,selection=selection)


def extended_prior():
    prior=read(graph_source.DEFAULT/'report.json');extension=supplement.completed()
    prior['quotient_components']=extension['combined_known_high_quotient_components']
    return prior


def run(path=DEFAULT,review_path=None,fixture_draws=None,pause_at=None,pause_marker=None,pause_index=0):
    path=_path(path)
    if (path/'certificate.json').exists():return completed(path)
    need(review_path is not None,'Reservation producer requires independent pre-draw review')
    need(pause_at in (None,'before_chunk_commit','after_chunk_commit','before_certificate'),'Unknown reservation recovery boundary')
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        resource_guard()
        identity=prepare(path,Path(review_path),fixture_draws);records,rows=prefix(path,identity)
        native=Native();need(native.build_id==identity['binding']['native_build_sha256'],'Reservation native classifier build differs')
        prior=extended_prior();identity_sha=sha(path/'identity.json')
        with closing(sqlite3.connect(path/'work.sqlite3')) as db:
            db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
            for start in range(len(records),identity['max_draws'],CHUNK):
                common.current(identity['binding']['source_inputs'])
                index=start//CHUNK;batch=[candidate_record(counter,native) for counter in range(start,min(start+CHUNK,identity['max_draws']))]
                body=dict(schema=SCHEMA+'-chunk',index=index,identity_sha256=identity_sha,start_counter=start,records=batch)
                resource_guard(2*len((config.canonical(body)+'\n').encode())+1024**2)
                digest=hashlib.sha256((config.canonical(body)+'\n').encode()).hexdigest();name=f'chunks/{index:06d}-{digest}.json'
                publish(path/name,body)
                if index==pause_index:pause(path,'before_chunk_commit',pause_at,pause_marker,index)
                previous=rows[-1][2] if rows else None
                with db:
                    db.execute('INSERT INTO chunks VALUES(?,?,?,?)',(index,name,digest,previous))
                    db.execute("UPDATE meta SET value=? WHERE key='committed_chunks'",(str(index+1),))
                rows.append((index,name,digest,previous));records.extend(batch)
                print(json.dumps(dict(event='population_chunk_committed',chunks=index+1,counters=len(records),maximum=identity['max_draws'])),flush=True)
                if index==pause_index:pause(path,'after_chunk_commit',pause_at,pause_marker,index)
        selection=select(records,prior,identity['binding']['graph_report_sha256'])
        resource_guard(2*len((config.canonical(selection)+'\n').encode())+1024**2)
        publish(path/'selection.json',selection)
        certificate=dict(schema=SCHEMA+'-certificate',passed=True,identity_sha256=identity_sha,
            journal_sha256=sha(path/'work.sqlite3'),selection_sha256=sha(path/'selection.json'),committed_chunks=len(rows),
            population_scan_complete=len(records)==identity['max_draws'],engineering_fixture=identity['engineering_fixture'],
            production_training_admission=False,model_and_policy_fixed=True)
        pause(path,'before_certificate',pause_at,pause_marker,len(rows));publish(path/'certificate.json',certificate)
    return completed(path)
