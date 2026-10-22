"""Fixed metadata-only acquisition population; no new positions are generated.

The old declared cells and mechanism classifications are reused verbatim. All
16,384 disposition rows remain referenced, including rejected metadata. Native
targets, values and outcomes cannot affect which 512 roots are acquired.
"""
from collections import Counter,defaultdict
from contextlib import closing
import json,math,sqlite3,statistics
from . import config,footprint_screen_common as s

ROOT=s.ROOT
POOL=s.FOLDER/'pool.json'
CELLS=('25-49/ordinary/actor0','25-49/ordinary/actor1',
       '50-74/ordinary/actor0','50-74/ordinary/actor1',
       '25-49/capture/actor0','25-49/capture/actor1',
       '25-49/bonus/actor0','25-49/bonus/actor1')
SALT='footprint-screen-v1/acquisition/20261017'
SOURCES=('congklak/footprint_screen_data.py','congklak/footprint_screen_reservations.py',
         'congklak/footprint_screen_common.py','congklak/config.py')
_CACHE={}
need,read,sha=s.need,s.read,s.sha

def protocol():
    return dict(schema=s.SCHEMA+'-population-protocol',source='existing frozen16384 metadata population',
        cells=list(CELLS),acquisition_roots=512,acquisition_per_cell=64,salt=SALT,
        rank='sha256(canonical JSON {salt,value:[cell,family,state_key]}); hexadecimal ascending',
        order='round-robin rank0..63, each declared cell in fixed cells order',
        eligibility='original declared cell and actual mechanism agree; original rejection list empty; '
          'no protected prior closure, previous reservations or duplicate metadata component',
        component_uniqueness_before_acquisition=True,full_batch_required=True,early_stop_when_quotas_filled=False,
        teacher_horizon=8,teacher='H_v1/oracle-free/final-player0-margin/full-legal-root-Q',
        fit_per_cell=16,case_subset_per_cell=4,arena_per_ordinary_cell=6,
        selection='joint maximum bipartite component-slot matching after all complete footprints',
        capped_targets='ineligible without replacement; complete observed footprint still joins closure',
        unknown_or_lost_raw='campaign HOLD; no retry',new_positions=0,quality_values_read=False,
        optimizer_pool='exact existing192 ordinary support; no widening',training_authorized=False,
        inherited_unknown_background=True,historically_unseen=False,models_fixed=True,policy_fixed=True)

def input_pins():
    from . import footprint_screen_reservations as reservations
    pins=reservations.input_pins()
    pins.update(s.common.pins([*[ROOT/p for p in SOURCES],*[ROOT/p for p in s.MODELS]]))
    return pins

def _state(key):
    need(type(key) is str and len(key)==34 and key==key.lower(),'Metadata state key geometry differs')
    raw=bytes.fromhex(key)
    need(len(raw)==17,'Metadata state key width differs')
    state=config.position(dict(pits=[list(raw[:7]),list(raw[8:15])],stores=[raw[7],raw[15]],controller=raw[16]))
    active=sum(sum(row) for row in state['pits'])
    need(25<=active<=74 and sum(v>0 for v in state['pits'][state['controller']])>=2,
         'Declared development root is outside acquisition domain')
    return state,raw

def _derive():
    from . import footprint_screen_reservations as r
    pins=input_pins();prior=r.prior_closure();population=r._old_read(r.POPULATION)
    ledger=population['counter_disposition_ledger']
    need(type(ledger) is list and len(ledger)==16384 and population['counter_draws']==16384,
         'Frozen finite population size differs')
    candidates=defaultdict(list);dispositions=[]
    source_sha=sha(r.POPULATION)
    for index,row in enumerate(ledger):
        need(type(row['counter']) is int and row['counter']==index and row['acquisition_exposed'] is True and
             row['analysis_or_outcome_exposed'] is False and row['optimizer_blocked'] is True,
             'Existing population acquisition ledger differs')
        reasons=list(row['rejection_reasons']);state,raw=_state(row['state_key'])
        component=prior['family_components'][row['family']]
        need(row['family'] in population['components'][row['new_domain_component']]['families'] and
             r.quotient(raw[:16],raw[16]).hex()==row['quotient'] and
             row['quotient'] in population['components'][row['new_domain_component']]['quotients'],
             'Existing root/component witness differs')
        if row['cell'] not in CELLS or row['actual_stratum']+'/actor'+str(raw[16])!=row['cell']:
            reasons.append('original declared cell and actual mechanism disagree')
        if prior['components'][component]['evaluation_blocked']:
            reasons.append('whole known prior component protected')
        if row['selection_roles']:
            reasons.append('previous fixed reservation')
        item=dict(counter=index,family=row['family'],state_key=row['state_key'],cell=row['cell'],
            source_component=component,reasons=sorted(set(reasons)),selected_for_acquisition=False)
        dispositions.append(item)
        if reasons:continue
        candidates[row['cell']].append(dict(id=row['family'],state=state,state_key=row['state_key'],family=row['family'],
            origin='footprint-screen/'+row['family'],cell=row['cell'],source_component=component,
            source_population_component=row['new_domain_component'],metadata_root_quotient=row['quotient'],
            source_ref=dict(path=s.relative(r.POPULATION),sha256=source_sha,
                pointer='/counter_disposition_ledger/'+str(index),component_pointer='/components/'+row['new_domain_component']),
            source_counter=index))
    for cell in CELLS:
        candidates[cell].sort(key=lambda row:r._rank(SALT,[cell,row['family'],row['state_key']]))
    available={cell:len(candidates[cell]) for cell in CELLS}
    cursors={cell:0 for cell in CELLS};seen=set();selected=[]
    for rank in range(64):
        for cell in CELLS:
            rows=candidates[cell]
            while cursors[cell]<len(rows) and rows[cursors[cell]]['source_component'] in seen:
                row=rows[cursors[cell]];cursors[cell]+=1
                dispositions[row['source_counter']]['reasons'].append('duplicate pre-acquisition metadata component')
            need(cursors[cell]<len(rows),'Fixed512 metadata pool shortage in '+cell)
            row=rows[cursors[cell]];cursors[cell]+=1;seen.add(row['source_component'])
            row=dict(row,index=len(selected));selected.append(row)
            dispositions[row['source_counter']]['selected_for_acquisition']=True
    for row in dispositions:
        row['disposition']=('selected_for_complete_footprint_acquisition' if row['selected_for_acquisition'] else
            'rejected_metadata_or_protected' if row['reasons'] else 'unselected_fixed_finite_population')
    need(Counter(row['cell'] for row in selected)==Counter({cell:64 for cell in CELLS}) and
         len({row['family'] for row in selected})==len(seen)==512,'Fixed acquisition pool quotas/ownership differ')
    result=dict(schema=s.SCHEMA+'-pool',roots=selected,input_pins=pins,acquisition=protocol(),
        metadata_available_per_cell=available,metadata_dispositions=dispositions,
        full_population_constraints_retained=16383,all_population_dispositions_retained=16384,
        old_actual_query_count=5,old_rejected_high_footprint_retained=6669,
        prior_component_count=len(prior['components']),prior_high_quotient_count=len(prior['quotient_components']),
        training_support_roots=192,training_support_cells=dict(Counter(row['cell'] for row in prior['training_support'])),
        old320_quarantine_retained=True,inherited_unknown_background=True,historically_unseen=False,
        new_positions=0,new_queries=0,model_forwards=0,optimizer_steps=0,
        quality_values_read=False,training_authorized=False,models_fixed=True,policy_fixed=True)
    need(input_pins()==pins,'Metadata sources changed during fixed pool construction')
    return result

def expected_pool():
    key=config.digest(input_pins())
    if key not in _CACHE:
        _CACHE.clear();_CACHE[key]=config.canonical(_derive()).encode()
    return json.loads(_CACHE[key])

def prepare_pool():
    body=expected_pool();s.atomic(POOL,body)
    return dict(schema=s.SCHEMA+'-pool-manifest',pool_ref=dict(path=s.relative(POOL),sha256=sha(POOL)),
        roots=512,cells={cell:64 for cell in CELLS},metadata_available_per_cell=body['metadata_available_per_cell'],
        old_rejected_high_footprint_retained=6669,training_authorized=False)

def membership():
    body=read(POOL)
    need(config.canonical(body)==config.canonical(expected_pool()),'Acquisition pool differs from frozen metadata recipe')
    return body

def validate_pool(body):
    need(config.canonical(body)==config.canonical(expected_pool()),'Fixed population derivation differs');return True

def workload_estimate():
    """Describe source-pinned workload scenarios, never infer target quality."""
    from . import footprint_screen_reservations as r
    prior=r.prior_closure();dbpath=ROOT/'data/deeper-9000-v2/work.sqlite3'
    need(s.prior_inputs().get(s.relative(dbpath))==sha(dbpath),'Saved workload source changed')
    with closing(sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)) as db:
        saved=[db.execute("SELECT json_extract(body,'$.report.nodes'),json_extract(body,'$.report.elapsed_ms') "
            "FROM labels WHERE state=?",(row['state_key'],)).fetchone() for row in prior['training_support']]
    need(len(saved)==192 and all(row is not None and type(row[0]) is int and row[0]>0 and
         type(row[1]) is int and row[1]>=0 for row in saved),'Saved node/timing metadata is incomplete')
    def stats(values):
        values=sorted(values)
        return dict(count=len(values),minimum=values[0],median=statistics.median(values),mean=statistics.mean(values),
            p95=values[math.ceil(.95*len(values))-1],maximum=values[-1],total=sum(values))
    observed=[]
    for proof in prior['previous_actual_footprints']:
        path=ROOT/proof['source_ref']['path'];body=r._old_read(path);answer=body['answer'];trace=answer['trace']
        nodes=answer.get('decision',{}).get('nodes',0)
        observed.append(dict(kind=body['query_kind'],source_ref=proof['source_ref'],events=proof['event_count'],
            nodes=nodes,owned=len(proof['quotients']),raw_bytes=24*proof['event_count'],
            archive_bytes=path.stat().st_size,critical_ms=body['clock']['critical_ms'],
            rejected=proof['rejected_attempt']))
    teachers=[row for row in observed if row['kind']=='teacher'];features=[row for row in observed if row['kind']=='features']
    need(len(teachers)==3 and all(type(row['nodes']) is int and row['nodes']>0 for row in teachers),'Teacher workload metadata differs')
    ratio=max(row['events']/row['nodes'] for row in teachers)
    node_stats=stats([row[0] for row in saved]);scenarios={}
    per_event_archive=max(row['archive_bytes']/row['events'] for row in teachers)
    for name,nodes in (('ordinary_mean_proxy',node_stats['mean']),('ordinary_p95_proxy',node_stats['p95'])):
        events=math.ceil(nodes*ratio);raw_bytes=events*24*512
        archive_bytes=math.ceil(events*per_event_archive)*512+sum(row['raw_bytes']+row['archive_bytes'] for row in features)*256
        scenarios[name]=dict(events_per_teacher=events,raw_sink_bytes=raw_bytes,archive_plus_feature_bytes=archive_bytes,
            combined_bytes=raw_bytes+archive_bytes)
    return dict(schema=s.SCHEMA+'-workload-estimate',input_pins=dict(input_pins(),**{s.relative(dbpath):sha(dbpath)}),
        saved_ordinary_h8_nodes=node_stats,saved_ordinary_h8_ms=stats([row[1] for row in saved]),
        saved_traced_queries=observed,scenarios=scenarios,
        worst_declared_raw_sinks_bytes=512*2*16000000*24,
        quota_guaranteed=False,completion_within_budget_guaranteed=False,
        scenario_limitation='Only192 historical ordinary roots and5 traced queries; capture/bonus and unique-high rates may differ',
        resource_response='Fixed batch does not shrink or replace; insufficient namespace/unknown raw means retained campaign HOLD',
        target_or_Q_values_read=False,new_queries=0,model_forwards=0,optimizer_steps=0)
