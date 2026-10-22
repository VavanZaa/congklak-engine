"""Immutable known-state protection from newly persisted engineering evidence.

Search envelopes supply actual decision-origin/state incidences.  Standalone
saved raw states are protected quotient markers, never an invented shared
origin for every position in an artifact.  Reading uses no engine/model/DLL.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
from . import config,neurodiag_common as common,native_ownership_trace as codec
from . import provenance_reconstruction as provenance,provenance_reconstruction_v2 as base

ROOT=common.ROOT
DEFAULT=provenance.DEFAULT/'engineering-supplement-v1'
SCHEMA='known-engineering-provenance-supplement-v1'
SCOPES=('native','handoff','audit')
SOURCES=('congklak/engineering_provenance_supplement.py','congklak/native_ownership_trace.py',
         'congklak/provenance_reconstruction.py','congklak/provenance_reconstruction_v2.py',
         'congklak/neurodiag_data.py')
need,read,sha=common.require,common.read,common.sha


def _walk(value,pointer=''):
    if isinstance(value,dict):
        if value.get('schema')=='complete-state-ownership-trace-v1':
            yield 'trace',pointer,value;return
        if set(value)=={'pits','stores','controller'}:
            try:state=config.position(value)
            except ValueError:return
            yield 'standalone_state',pointer,state;return
        for key,child in value.items():
            yield from _walk(child,pointer+'/'+str(key).replace('~','~0').replace('/','~1'))
    elif isinstance(value,list):
        for index,child in enumerate(value):yield from _walk(child,pointer+'/'+str(index))


def discover():
    """Find only material trace/state JSON artifacts; atomic temporaries ignored."""
    material=[];scanned=0
    for scope in SCOPES:
        directory=ROOT/'build/ownership-implementation-v1'/scope
        for path in sorted(directory.rglob('*.json')):
            body=read(path);scanned+=1;items=list(_walk(body))
            if items:material.append((path,sha(path),items))
    return material,scanned


def collect():
    prior=read(base.DEFAULT/'report.json');prior_q=prior['quotient_components']
    base_sha=sha(base.DEFAULT/'report.json')
    material,scanned=discover();graph=provenance.Graph();traces={};markers=[];events=0
    source_pins={};partial=[]
    for path,digest,items in material:
        source=provenance.rel(path);source_pins[source]=digest
        for kind,pointer,value in items:
            witness=dict(path=source,sha256=digest,pointer=pointer)
            if kind=='standalone_state':
                q=provenance.metadata.quotient(value);a=provenance.domain.quotient_active(q)
                markers.append(dict(raw_state_key=common.rawkey(value),quotient=q,active_seeds=a,witness=witness,
                                    kind='protected saved engineering raw-state marker',actual_incidence=False))
                if a>=25:
                    graph.find(('quotient',q))
                    if q in prior_q:graph.join(('quotient',q),('prior_component',prior_q[q]))
                continue
            trace=value;raw=codec.trace_bytes(trace)
            ident=config.digest(dict(binding=trace['binding'],event_sha256=trace['event_sha256']))
            origin='engineering-trace/'+ident
            if ident in traces:traces[ident]['source_witnesses'].append(witness);continue
            graph.find(('origin',origin));by_raw={};counts=Counter();trace_events=0
            for event in codec.iter_events(trace):
                p=event['state'];key=common.rawkey(p);q=provenance.metadata.quotient(p);a=provenance.domain.quotient_active(q)
                counts[event['operation']]+=1;trace_events+=1
                if key not in by_raw:
                    by_raw[key]=dict(raw_state_key=key,quotient=q,active_seeds=a,first_event=event['index'],
                                    operation=event['operation'],scope=event['scope'],occurrences=0)
                by_raw[key]['occurrences']+=1
                if a>=25:
                    graph.join(('origin',origin),('quotient',q))
                    if q in prior_q:graph.join(('quotient',q),('prior_component',prior_q[q]))
            need(trace_events==trace['event_count'],'Engineering trace event cardinality differs')
            incomplete=bool(trace['overflow'] or trace['invalid'] or trace['truncated'] or trace['attempted_events']!=trace['event_count'])
            if incomplete:partial.append(dict(origin=origin,binding=trace['binding'],source=witness,
                overflow=trace['overflow'],invalid=trace['invalid'],truncated=trace['truncated'],
                attempted_events=trace['attempted_events'],saved_events=trace['event_count'],
                interpretation='Only persisted valid complete-state prefix is witnessed; failed attempt is retained, no absent-event coverage claim'))
            traces[ident]=dict(origin=origin,event_sha256=trace['event_sha256'],binding=trace['binding'],
                source_witnesses=[witness],event_count=trace_events,raw_observations=list(by_raw.values()),
                complete_trace=not incomplete,operation_counts=dict(sorted(counts.items())))
            events+=trace_events
    grouped=defaultdict(lambda:dict(origins=[],quotients=[],prior_components=[]))
    for kind,value in graph.parent:
        grouped[graph.find((kind,value))][{'origin':'origins','quotient':'quotients','prior_component':'prior_components'}[kind]].append(value)
    components={};observed_q={};remap={}
    for body in grouped.values():
        body={k:sorted(v) for k,v in body.items()};body.update(schema=SCHEMA+'-component',base_graph_report_sha256=base_sha)
        cid=config.digest(body);components[cid]=body
        for q in body['quotients']:observed_q[q]=cid
        for old in body['prior_components']:remap[old]=cid
    combined={q:remap.get(cid,cid) for q,cid in prior_q.items()};combined.update(observed_q)
    source_pins.update(common.pins([*SOURCES,base.DEFAULT/'certificate.json',base.DEFAULT/'report.json']))
    missing=[dict(attempt='native author-v1 branching timed H_v1 overflow',
        source='build/ownership-implementation-v1/native/author-v1/failed-attempt.json',
        reason='Failed author harness explicitly did not persist partial trace; unreported inflight work possible',
        missing_high_footprints_possible=True),
        dict(attempt='5 ad-hoc fixed PUCT and6 baseline smoke searches reported by native owner',
             source='Native implementation owner report relayed before supplement freeze',
             reason='Not all envelopes were persisted; exact missing-state sets cannot be reconstructed from aggregate counts',
             missing_high_footprints_possible=True)]
    source_pins.update(common.pins([ROOT/missing[0]['source'],ROOT/'build/ownership-implementation-v1/native/author-v2/result.json']))
    return dict(schema=SCHEMA+'-report',inputs=dict(sorted(source_pins.items())),
        scopes=list(SCOPES),scanned_json_files=scanned,material_json_files=len(material),
        trace_episodes=list(traces.values()),unique_trace_episodes=len(traces),decoded_saved_events=events,
        standalone_state_markers=markers,partial_failed_trace_prefixes=partial,
        missing_attempt_ledger=missing,missing_material_engineering_footprints_hold=True,
        known_persisted_source_scope_only=True,complete_engineering_source_absence_claim=False,
        conditional_metadata_acquisition_only=True,training_admission=False,evaluation_admission=False,
        high_quotient_components=dict(sorted(observed_q.items())),base_component_remap=dict(sorted(remap.items())),
        combined_known_high_quotient_components=dict(sorted(combined.items())),components=dict(sorted(components.items())),
        all_observed_high_states_protected=True,no_invented_family_quotient_edges=True,
        historical_unknown_leaves_still_unknown=True,models_and_policy_fixed=True,
        model_forwards=0,searches=0,new_games=0,new_teacher_labels=0,optimizer_steps=0,feature_simulations=0,
        metadata_generation_started=False,production_admission=False)


def run():
    need(not (DEFAULT/'certificate.json').exists(),'Supplement snapshot is immutable; use completed() or a new namespace')
    report=collect();common.publish(DEFAULT/'report.json',report)
    common.publish(DEFAULT/'certificate.json',dict(schema=SCHEMA+'-certificate',passed=True,
        inputs=report['inputs'],report_sha256=sha(DEFAULT/'report.json'),known_state_protection=True,
        incomplete_attempts_retained=len(report['partial_failed_trace_prefixes']),metadata_generation_started=False))
    return report


def completed():
    certificate=read(DEFAULT/'certificate.json');need(certificate['schema']==SCHEMA+'-certificate' and certificate['passed'] is True,
                                                  'Engineering supplement certificate scope differs')
    common.current(certificate['inputs']);need(sha(DEFAULT/'report.json')==certificate['report_sha256'],'Engineering supplement report changed')
    report=read(DEFAULT/'report.json')
    need(report['inputs']==certificate['inputs'] and report['all_observed_high_states_protected'] is True and
         report['no_invented_family_quotient_edges'] is True,'Engineering supplement scope changed')
    return report
