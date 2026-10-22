"""Prospective whole-component reservations from the complete held footprint.

Pure metadata publication: no engine, feature encoder, teacher, or model calls.
The predecessor stays held. All its returned attempts, including the failure,
contribute complete high-domain incidences before the new reservation is sealed.
"""
from collections import defaultdict
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import uuid
from . import config, neurodiag_common as common
from . import guarded_value_pilot_v2_data as old_data
from . import guarded_value_pilot_v2_guard as old_guard
from . import native_ownership_trace as codec
from .value_diagnostics import atomic
from .valuepilot_reservations import Writer

ROOT=common.ROOT
FOLDER=ROOT/'build/prospective-eval-v1'
DEFAULT=FOLDER/'reservations'
OLD=ROOT/'build/guarded-value-pilot-v2'
SCHEMA='prospective-eval-v1-reservations'
SOURCES=('congklak/prospective_eval_reservations.py',)
need,read,sha=common.require,common.read,common.sha
STATIC,OWNED,EVENT=old_guard.STATIC,old_guard.OWNED,old_guard.EVENT


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def ref(path):return dict(path=relative(path),sha256=sha(path))
def path_of(path):
    path=Path(path).resolve()
    need(path==DEFAULT or path.is_relative_to(FOLDER/'fixtures'),'Reservation output outside new namespace')
    return path


class Union:
    def __init__(self):self.parent={}
    def find(self,a):
        self.parent.setdefault(a,a)
        if self.parent[a]!=a:self.parent[a]=self.find(self.parent[a])
        return self.parent[a]
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def close_units(components,origin_components,origin_roles,footprints,lookup):
    """Close every incidence, then refuse protected or multi-unit bridges.

    ``lookup`` maps a canonical high key to its entire predecessor component.
    Every unknown key is also a graph node, so even previously unseen dynamic
    intersections join whole units instead of disappearing from the closure.
    """
    union=Union();unknown={};by_component=defaultdict(set)
    for cid in components:union.find(('component',cid))
    for origin,cid in origin_components.items():
        if cid==old_data.SHARED:continue
        need(cid in components,'Registered owner component missing')
        by_component[cid].add(origin)
    for footprint in footprints:
        origin=footprint['origin'];need(origin in origin_components,'Unregistered historical footprint')
        owner=origin_components[origin]
        for text in footprint['quotients']:
            old_guard.key(text,True);cid=lookup.get(text)
            need(owner!=old_data.SHARED,'Shared endgame footprint entered owned domain')
            node=('component',cid) if cid is not None else ('quotient',text)
            union.join(('component',owner),node)
            if cid is None:unknown[text]=origin
    groups=defaultdict(lambda:dict(prior_components=[],new_quotients=[],origins=[]))
    for node in list(union.parent):
        group=groups[union.find(node)]
        group['prior_components' if node[0]=='component' else 'new_quotients'].append(node[1])
    translated={};summaries={};extra={}
    for body in groups.values():
        for cid in body['prior_components']:body['origins'].extend(by_component[cid])
        for name in body:body[name]=sorted(set(body[name]))
        origins=body['origins'];protected=[cid for cid in body['prior_components'] if components[cid]['evaluation_blocked']]
        need(len(origins)<=1,'Whole-component closure crosses separate selected evaluation/optimizer units')
        if origins and origin_roles[origins[0]]!='optimizer_support':
            need(not protected,'Evaluation whole-component closure touches protected prior support/quarantine')
        cid=config.digest(dict(schema=SCHEMA+'-component',**body))
        summaries[cid]=dict(body,roles=sorted({origin_roles[o] for o in origins}),
            evaluation_blocked=bool(protected),blocking_prior_components=protected,
            acquired_origins_optimizer_blocked=any(components[c]['acquired_origins_optimizer_blocked'] for c in body['prior_components']),
            prospective_only=True,historically_unseen=False)
        for oldcid in body['prior_components']:translated[oldcid]=cid
        for q in body['new_quotients']:extra[q]=cid
    return dict(components=summaries,prior_to_prospective=translated,new_quotient_components=extra,
        origin_components={o:(c if c==old_data.SHARED else translated[c]) for o,c in origin_components.items()})


def returned_footprints():
    """Reconstruct high incidences literally from every old returned raw trace."""
    out=[];pins={}
    paths=sorted([*OLD.glob('jobs/*/archives/*.json'),*OLD.glob('jobs/*/failed-attempts/*.json')])
    need(len(paths)==354,'Expected complete 353 accepted plus one failed returned attempt')
    for path in paths:
        body=read(path);answer=body['answer'];trace=answer['trace'];cert=answer['fast_certificate']
        need(trace['attempted_events']==trace['event_count'] and trace['overflow'] is False and
             trace['invalid'] is False and trace['truncated'] is False,'Old returned footprint has unknown raw tail')
        raw=codec.trace_bytes(trace);first={};high=0
        for index,(cells,actor,op,scope,reserved,action,depth) in enumerate(EVENT.iter_unpack(raw)):
            need(sum(cells)==98 and actor in (0,1) and 1<=op<=37 and scope in (1,2,3,4) and
                 reserved==0 and action>=-1 and depth>=-1,'Historical raw event malformed')
            q=old_guard.raw_key(cells,actor)
            if sum(q[:14])<25:continue
            high+=1;first.setdefault(q,OWNED.pack(q,index,cells,actor))
        owned=base64.b64decode(answer['fast_owned_high'],validate=True)
        need(owned==b''.join(first[q] for q in sorted(first)) and len(first)==cert['owned_count'] and
             hashlib.sha256(owned).hexdigest()==cert['owned_sha256'] and high==cert['owned_occurrences'],
             'Old compact footprint differs from complete raw trace')
        out.append(dict(origin=body['origin'],decision_id=body['decision_id'],source_ref=ref(path),
            quotients=[q.hex() for q in sorted(first)],raw_sha256=hashlib.sha256(raw).hexdigest(),
            raw_events=trace['event_count'],owned_occurrences=high,failed_attempt=path.parent.name=='failed-attempts'))
        pins.update(common.pins([path]))
        sink=answer.get('attempt_sink')
        if isinstance(sink,dict):
            source=sink.get('ref',sink)
            if isinstance(source,dict) and source.get('path') and (ROOT/source['path']).is_file():
                pins.update(common.pins([ROOT/source['path']]))
    need(sum(row['failed_attempt'] for row in out)==1,'Failed returned footprint missing/duplicated')
    return out,pins


def _publish_bytes(path,payload):
    path=Path(path)
    if path.exists():need(path.read_bytes()==payload,'Immutable reservation binary differs');return
    path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def prepare(output=DEFAULT,pause=lambda point:None):
    path=path_of(output)
    if (path/'catalog.json').exists():return verify(path)
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        old_report=read(old_data.RESERVATIONS);old_registry=read(OLD/'guard/registry.json')
        members=old_data.membership();members['cases']=sorted(members['cases'])
        origin_roles={row['origin']:row['role'] for role in ('train','fit','arena','exact') for row in members[role]}
        need(set(origin_roles)==set(old_registry['origins']) and len(origin_roles)==345,'Old global owner registry differs')
        shared=read(OLD/'guard/shared.json');records=(OLD/'guard/components.bin').read_bytes()
        need(hashlib.sha256(records).hexdigest()==shared['records_ref']['sha256'] and len(records)==8135078*19,
             'Complete predecessor mapping bytes changed')
        lookup=old_guard.Lookup(records,shared['component_ordinals'])
        footprints,pins=returned_footprints()
        closure=close_units(old_report['components'],old_registry['origin_components'],origin_roles,footprints,lookup)
        # Protect the exact192 trained components even if a source summary's
        # generic evaluation flag is ever weakened in a future source namespace.
        support={row['component'] for row in members['train']}
        for body in closure['components'].values():
            if support.intersection(body['prior_components']):
                need(len(body['prior_components'])==1 and body['new_quotients']==[] and body['roles']==['optimizer_support'],
                     'Prospective overlay changes trained support component')
        ordinals={cid:i+1 for i,cid in enumerate(sorted(closure['components']))};ordinals[old_data.SHARED]=0
        old_ordinals={v:ordinals[closure['prior_to_prospective'][k]] for k,v in shared['component_ordinals'].items() if v}
        extras=sorted((bytes.fromhex(q),ordinals[c]) for q,c in closure['new_quotient_components'].items())
        packed=bytearray();cursor=0
        for q,ordinal in STATIC.iter_unpack(records):
            while cursor<len(extras) and extras[cursor][0]<q:
                packed.extend(STATIC.pack(*extras[cursor]));cursor+=1
            need(cursor==len(extras) or extras[cursor][0]!=q,'New quotient unexpectedly present in predecessor map')
            packed.extend(STATIC.pack(q,old_ordinals[ordinal]))
        for row in extras[cursor:]:packed.extend(STATIC.pack(*row))
        _publish_bytes(path/'components.bin',bytes(packed))
        for role in ('train','fit','arena','exact'):
            for row in members[role]:
                row['predecessor_component']=row['component'];cid=closure['origin_components'][row['origin']]
                row.update(component=cid,current_component=cid,joint_component=cid)
        origin_ids={o:i+1 for i,o in enumerate(old_registry['origins'])}
        claims=old_guard.PackedClaims((OLD/'jobs/guards/claims.bin').read_bytes(),origin_ids)
        allclaims=dict(claims.items())
        for footprint in footprints:
            for q in footprint['quotients']:
                need(allclaims.get(q,footprint['origin'])==footprint['origin'],'Historical global claims cross owners')
                allclaims[q]=footprint['origin']
        _publish_bytes(path/'claims.bin',b''.join(STATIC.pack(bytes.fromhex(q),origin_ids[o]) for q,o in sorted(allclaims.items())))
        pins.update(common.pins([old_data.RESERVATIONS,OLD/'held-report.json',OLD/'audit/hold-v1/forensics.json',
            OLD/'guard/shared.json',OLD/'guard/components.bin',OLD/'guard/registry.json',OLD/'jobs/guards/claims.bin',
            *[ROOT/name for name in SOURCES]]))
        policy=dict(schema=SCHEMA+'-policy',complete_predecessor_component_unions=True,
            full_returned_failed_footprint_retained=True,selected_unit_bridges='HOLD',optimizer_support_bridges='HOLD',
            inherited_quarantine_bridges='HOLD',new_static_foreign_component='HOLD',unknown_dynamic_key='claim exact key before next query',
            role_membership_fixed=True,quality_values_used_for_selection=False,historically_unseen=False,
            inherited_unknown_background=True,old_pilot_stays_held=True,training_authorized=False,models_fixed=True,policy_fixed=True)
        report=dict(schema=SCHEMA+'-report',**closure,policy=policy,source_inputs=pins,
            returned_attempts=len(footprints),failed_attempts=sum(r['failed_attempt'] for r in footprints),
            complete_raw_events=sum(r['raw_events'] for r in footprints),footprints=footprints,
            original_support192_preserved=True,fit_roots=128,arena_origins=24,selected_units=344,
            inherited_claims=len(claims),all_retained_claims=len(allclaims),record_count=len(packed)//19,
            shared_records_ref=ref(path/'components.bin'),seed_claims_ref=ref(path/'claims.bin'),
            component_ordinals=ordinals,reservation_admission=True,production_searches=0,optimizer_steps=0)
        atomic(path/'report.json',report);atomic(path/'membership.json',members)
        registry=dict(schema=SCHEMA+'-registry',origins=old_registry['origins'],origin_components=closure['origin_components'],
            component_ordinals=ordinals,shared_records=ref(path/'components.bin'),graph=ref(path/'report.json'),
            seed_claims=ref(path/'claims.bin'),role_roots=[{k:r[k] for k in ('state_key','family','role','origin','component')}
                for role in ('train','fit','arena','exact') for r in members[role]])
        atomic(path/'registry.json',registry)
        certificate=dict(schema=SCHEMA+'-certificate',passed=True,reservation_admission=True,
            files=common.pins([path/n for n in ('report.json','membership.json','registry.json','components.bin','claims.bin')]))
        atomic(path/'certificate.json',certificate);pause('before_catalog')
        atomic(path/'catalog.json',dict(schema=SCHEMA+'-catalog',certificate=ref(path/'certificate.json'),passed=True));pause('after_catalog')
    return verify(path)


def verify(output=DEFAULT,full=True):
    path=path_of(output);certificate=read(path/'certificate.json');catalog=read(path/'catalog.json')
    need(catalog==dict(schema=SCHEMA+'-catalog',certificate=ref(path/'certificate.json'),passed=True) and
         certificate['schema']==SCHEMA+'-certificate' and certificate['passed'] is True,'Reservation completion differs')
    common.current(certificate['files']);report=read(path/'report.json')
    if full:common.current(report['source_inputs'])
    need(report['reservation_admission'] is True and report['original_support192_preserved'] is True and
         report['policy']['historically_unseen'] is False,'Prospective reservation scope differs')
    return report


def membership():
    verify(full=False);members=read(DEFAULT/'membership.json');members['cases']=set(members['cases']);return members


def registry():
    verify(full=False);return read(DEFAULT/'registry.json')
