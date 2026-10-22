"""Pure, source-witnessed quarantine and next-update support projection.

The old 512-root recipe and every old exclusion remain immutable.  A projected
ordinary-only subset is a different future recipe, never a repaired old fit or
a claim that inherited model ancestry has been erased.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from . import config,ownership_resolution_common as stage,ownership_fast_guard as keys

ROOT=stage.ROOT
BASE=ROOT/'build/ownership-implementation-v1/provenance/certified-v2/report.json'
CERT=BASE.parent/'certificate.json'
PROVENANCE=ROOT/'build/ownership-implementation-v1/provenance/production'
SELECTION=ROOT/'build/ownership-implementation-v1/reservations/production-v2/selection.json'
SELECTION_CERT=SELECTION.parent/'certificate.json'
SCHEMA='ownership-resolution-support-v1'
SOURCES=('congklak/ownership_resolution_support.py','congklak/ownership_resolution_common.py',
         'congklak/ownership_fast_guard.py','congklak/config.py','congklak/neurodiag_common.py')
OPENING='07'*14+'00'
FLAGS=('optimizer_blocked','fit_blocked','arena_blocked','ancestral_training','acquisition_exposed')
ORDINARY_CELLS=tuple(band+'/ordinary/actor'+str(actor) for band in ('25-49','50-74') for actor in (0,1))
ALL_CELLS=tuple(sorted(ORDINARY_CELLS+tuple('25-49/'+mechanism+'/actor'+str(actor)
    for mechanism in ('bonus','capture') for actor in (0,1))))
need,read,sha=stage.need,stage.read,stage.sha
LOADED_SHA=sha(__file__)
_INVENTORY_CACHE={}


def _unit_refs():
    catalog=read(PROVENANCE/'catalog.json');identity=read(PROVENANCE/'identity.json');refs=[];previous=None
    need(catalog['passed'] is True and catalog['committed_units']==354 and
         catalog['identity_sha256']==sha(PROVENANCE/'identity.json'),'Frozen provenance catalog differs')
    for index,p in enumerate(sorted((PROVENANCE/'commits').glob('*.json'))):
        c=read(p);relative=f'units/{index:06d}-{c["result_sha256"]}.json'
        need(p.name==f'{index:06d}.json' and c['index']==index and c['identity_sha256']==catalog['identity_sha256'] and
             c['previous_commit_sha256']==previous and c['task_sha256']==config.digest(identity['tasks'][index]) and
             c['result']==relative and catalog['commits'].get(stage.relative(p))==sha(p),
             'Frozen provenance commit identity/prefix differs')
        refs.append((index,PROVENANCE/relative,c['result_sha256']));previous=sha(p)
    need(len(refs)==354,'Frozen provenance unit coverage differs');return refs,identity


def input_pins():
    need(sha(__file__)==LOADED_SHA,'Support implementation changed after import')
    refs,_=_unit_refs()
    return stage.common.pins([BASE,CERT,PROVENANCE/'identity.json',PROVENANCE/'catalog.json',
        *sorted((PROVENANCE/'commits').glob('*.json')),*[p for _,p,_ in refs],SELECTION,SELECTION_CERT,
        ROOT/'build/ownership-resolution-v1/contract.json',*SOURCES,*stage.MODELS])


def _base():
    cert=read(CERT);r=read(BASE)
    need(cert['passed'] is True and cert['report_sha256']==sha(BASE) and
         r['schema']=='source-witnessed-provenance-role-correction-v2-report' and
         r['models_and_policy_fixed'] is True and r['training_authorized'] is False and
         r['support_roots_exactly_reused']==512 and r['support_roots_with_genuine_exclusion']==320 and
         r['support_training_admission'] is False and r['raw_incidence_and_component_partition_changed'] is False,
         'Frozen support/role correction scope differs')
    pool=r['candidate_pools']['support_reuse']
    need(len(pool)==512 and len({x['state_key'] for x in pool})==512,'Frozen support identities differ')
    for row in pool:
        raw=bytes.fromhex(row['state_key']);need(len(raw)==17 and keys.raw_key(raw[:16],raw[16]).hex()==row['quotient'],
                                               'Frozen support raw state/ownership identity differs')
    blocked={x['new_domain_component'] for x in pool if not x['eligible']}
    need(len(blocked)==1 and sum(not x['eligible'] for x in pool)==320,'Frozen whole-component conflict changed')
    return r,pool,next(iter(blocked))


def _counts(rows):
    return {cell:dict(roots=sum(r['cell']==cell for r in rows),components=len({r.get('joint_component',r.get('new_domain_component'))
            for r in rows if r['cell']==cell})) for cell in ALL_CELLS}


def _edge_proof(edge,index,path,digest,edge_index,source_inputs):
    raw=bytes.fromhex(edge['raw_state_key']);need(len(raw)==17,'Witness raw state width differs')
    q=keys.raw_key(raw[:16],raw[16]).hex()
    need(q==edge['quotient'] and sum(bytes.fromhex(q)[:14])==edge['active_seeds']>=25 and
         edge['domain']=='owned' and source_inputs.get(edge['witness']['path'])==edge['witness']['sha256'],
         'Actual high incidence geometry/source binding differs')
    return dict(unit_index=index,unit_path=stage.relative(path),unit_sha256=digest,pointer='/edges/'+str(edge_index),
        incidence=edge)


def _compute_inventory(source_pins):
    base,pool,giant=_base();refs,identity=_unit_refs();blocked=[r for r in pool if not r['eligible']]
    component=base['components'][giant];ledger=base['marker_ledger']
    anchors=[(index,m) for index,m in enumerate(ledger) if m['kind']=='family' and m['optimizer_blocked'] and
        m['reason']=='current frozen fit_check full origin' and m['value'] in component['families']]
    need(anchors,'No genuine fit exclusion witness remains')
    marker_index,anchor=min(anchors,key=lambda im:(im[1]['value'],config.canonical(im[1]['witness'])))
    wanted={r['family'] for r in blocked}|{anchor['value']};root_by_raw={r['state_key']:r for r in blocked}
    root_edges={};opening_edges={};scanned=0
    for index,path,digest in refs:
        payload=path.read_bytes();need(hashlib.sha256(payload).hexdigest()==digest,'Frozen incidence unit corrupt')
        unit=json.loads(payload);need(unit['index']==index,'Frozen incidence unit index differs')
        for edge_index,edge in enumerate(unit['edges']):
            scanned+=1
            if edge['family'] not in wanted:continue
            is_open=edge['quotient']==OPENING
            root=root_by_raw.get(edge['raw_state_key'])
            is_root=root is not None and edge['family']==root['family'] and edge['quotient']==root['quotient']
            if not is_open and not is_root:continue
            proof=_edge_proof(edge,index,path,digest,edge_index,identity['inputs'])
            if is_open:opening_edges.setdefault(edge['family'],proof)
            if is_root:root_edges.setdefault(root['state_key'],proof)
    need(scanned==base['actual_incidence_occurrences'],'Frozen incidence observation count differs')
    need(anchor['value'] in opening_edges and all(r['family'] in opening_edges and r['state_key'] in root_edges for r in blocked),
         'A quarantined root lacks an actual complete causal path; retain HOLD')
    marker_ref=dict(path=stage.relative(BASE),sha256=sha(BASE),pointer='/marker_ledger/'+str(marker_index),marker=anchor)
    dispositions=[]
    for index,row in enumerate(pool):
        old=base['components'][row['new_domain_component']]
        result=dict(index=index,state_key=row['state_key'],quotient=row['quotient'],family=row['family'],cell=row['cell'],
            active_seeds=sum(bytes.fromhex(row['quotient'])[:14]),old_component=row['new_domain_component'],old_eligible=row['eligible'],
            original_flags={k:old[k] for k in FLAGS},source=dict(path=stage.relative(BASE),sha256=sha(BASE),
                pointer='/candidate_pools/support_reuse/'+str(index)),
            disposition='quarantined_whole_component' if not row['eligible'] else 'candidate_for_ordinary_only_future_recipe')
        if not row['eligible']:
            result['causal_path']=dict(nodes=[['quotient',row['quotient']],['family',row['family']],
                ['quotient',OPENING],['family',anchor['value']]],
                actual_incidence_witnesses=[root_edges[row['state_key']],opening_edges[row['family']],opening_edges[anchor['value']]],
                protected_role_marker=marker_ref,shared_ancestor_active_seeds=98,
                reason='All high states of a source origin remain connected; the public98-seed ancestor is inside owned>=25')
        dispositions.append(result)
    eligible=[r for r in pool if r['eligible']]
    return dict(schema=SCHEMA+'-inventory',original_pool_roots=512,original_balanced_recipe_available=False,
        original_support_training_admission=False,original_pool_and_markers_unchanged=True,
        original_marker_count=len(ledger),original_marker_ledger_sha256=config.digest(ledger),
        original_512_dispositions=dispositions,genuine_conflicting_roots=320,projected_before_overlay=len(eligible),
        old_cells=_counts(pool),quarantined_cells=_counts([r for r in pool if not r['eligible']]),ordinary_candidate_cells=_counts(eligible),
        whole_component_tombstone=dict(component=giant,source_path=stage.relative(BASE),source_sha256=sha(BASE),
            source_pointer='/components/'+giant,component_sha256=config.digest(component),families=len(component['families']),
            quotients=len(component['quotients']),all_high_aliases_and_future_merged_descendants_quarantined=True),
        causal_paths_complete=320,actual_incidence_observations_scanned=scanned,incidences_invented=0,
        input_pins=source_pins,future_scope='ordinary-only known-provenance next-update development',
        new_recipe_and_explicit_run_required=True,inherited_model_ancestry_unknown=True,historically_unseen_claim=False,
        training_authorized=False,models_fixed=True,policy_fixed=True,activity=stage.ZERO)


def _derive_inventory():
    # Rehash only this helper's bounded declared sources. A cached result is
    # immutable canonical bytes and every caller receives a fresh decoded copy.
    pins=input_pins();identity=config.digest(pins)
    if identity not in _INVENTORY_CACHE:
        report=_compute_inventory(pins)
        need(input_pins()==pins,'Support source changed during causal reconstruction')
        payload=config.canonical(report).encode();_INVENTORY_CACHE.clear();_INVENTORY_CACHE[identity]=payload
    return json.loads(_INVENTORY_CACHE[identity])


def inventory():return _derive_inventory()


def validate_inventory(report):
    need(config.canonical(report)==config.canonical(_derive_inventory()),
         'Support inventory differs from immutable witnesses');return True


class Union:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def _joint_graph(overlay,selection):
    """Union certified connected-component constraints, never inferred edges."""
    union=Union();families={};quotients={};sources={};flags={}
    for namespace,components in (('known',overlay['components']),('population',selection['components'])):
        for cid,c in components.items():
            node=(namespace,cid);union.find(node);sources[node]=c
            need(all(type(c.get(k,False)) is bool for k in FLAGS),'Component protection flags must be boolean')
            flags[node]={k:c.get(k,False) for k in FLAGS}
            if namespace=='population':
                flags[node]['optimizer_blocked']=True
                flags[node]['acquisition_exposed']=True
            family_values=sorted(set(c['families'])|set(c.get('origins',[])))
            for kind,lookup,values in (('family',families,family_values),('quotient',quotients,c['quotients'])):
                for value in values:
                    if kind=='quotient':keys.key(value,True)
                    if value in lookup:union.join(node,lookup[value])
                    else:lookup[value]=node
    groups=defaultdict(list)
    for node in union.parent:groups[union.find(node)].append(node)
    ids={root:config.digest(dict(schema=SCHEMA+'-joint-constraint-component',constraints=sorted(nodes))) for root,nodes in groups.items()}
    summary={ids[root]:dict(constraints=[list(n) for n in sorted(nodes)],
        known_components=sorted(n[1] for n in nodes if n[0]=='known'),population_components=sorted(n[1] for n in nodes if n[0]=='population'),
        **{k:any(flags[n][k] for n in nodes) for k in FLAGS}) for root,nodes in groups.items()}
    return dict(components=summary,family_components={f:ids[union.find(n)] for f,n in families.items()},
        quotient_components={q:ids[union.find(n)] for q,n in quotients.items()},
        known_components={cid:ids[union.find(('known',cid))] for cid in overlay['components']})


def _weights(rows):
    counts=Counter(r['cell'] for r in rows)
    missing=[c for c in ORDINARY_CELLS if not counts[c]]
    if missing:return dict(available=False,missing_cells=missing,weights=[],rule='No renormalization over missing cells')
    weighted=[]
    for row in rows:
        need(row['cell'] in ORDINARY_CELLS,'Ordinary-only sampler received another mechanism')
        weight=Fraction(1,4*counts[row['cell']]);weighted.append(dict(state_key=row['state_key'],cell=row['cell'],
            numerator=weight.numerator,denominator=weight.denominator))
    need(sum(Fraction(r['numerator'],r['denominator']) for r in weighted)==1,'Prospective weights do not sum exactly to one')
    return dict(available=True,missing_cells=[],weights=weighted,
        rule='Four ordinary actor/band cells each1/4; uniform roots within each remaining cell',
        weight_values_read=False,component_independence_claim=False,recipe_not_executed=True)


def _validate_overlay(base,overlay):
    # The coordinator independently validates the latest actual-source overlay;
    # here we additionally require preservation of every old constraint/flag.
    need(overlay['ancestry_unknown'] is True and overlay['historical_provenance_complete'] is False and
         overlay['source_complete_for_declared_persisted'] is True,'Known overlay scope/ancestry differs')
    for cid,c in base['components'].items():
        assigned={overlay['family_components'].get(f) for f in c['families']}
        assigned.update(overlay['quotient_components'].get(q) for q in c['quotients'])
        need(None not in assigned and len(assigned)==1,'Known overlay split/dropped an old high component')
        new=overlay['components'][next(iter(assigned))]
        need(all(type(new[k]) is bool for k in FLAGS),'Overlay protection flags must be boolean')
        need(all(not c[k] or new[k] for k in FLAGS),'Known overlay erased an inherited protection flag')


def _validate_population(selection):
    ledger=selection['counter_disposition_ledger']
    need(selection['all_rejections_retained'] is True and selection['conditional_metadata_acquisition_only'] is True and
         type(selection['target_values_read']) is int and selection['target_values_read']==0 and
         selection['training_admission'] is False and
         len(ledger)==16384 and len({r['family'] for r in ledger})==16384 and
         all(r['optimizer_blocked'] is True and r['acquisition_exposed'] is True and
             r['analysis_or_outcome_exposed'] is False for r in ledger) and
         all(not c['prior_components'] for c in selection['components'].values()),
         'Frozen population acquisition/protection ledger or prior linkage changed')


def _derive_projection(overlay,conditional_selection):
    base,pool,giant=_base();selection=read(SELECTION);certificate=read(SELECTION_CERT)
    need(certificate['passed'] is True and certificate['selection_sha256']==sha(SELECTION) and
         config.canonical(selection)==config.canonical(conditional_selection) and selection['metadata_reservations_complete'] is True,
         'Conditional reservation identity differs')
    _validate_population(selection)
    _validate_overlay(base,overlay);joint=_joint_graph(overlay,selection)
    quarantine={joint['family_components'][f] for f in base['components'][giant]['families']}
    need(len(quarantine)==1,'Whole giant component was split by the joint overlay')
    dispositions=[];eligible=[]
    for index,row in enumerate(pool):
        cid=joint['family_components'][row['family']]
        need(cid==joint['quotient_components'][row['quotient']],'Support family/raw quotient joint ownership differs')
        c=joint['components'][cid];reasons=[]
        if not row['eligible']:reasons.append('original genuine exclusion remains quarantined')
        if cid in quarantine:reasons.append('whole quarantined component or merged descendant')
        if c['optimizer_blocked']:reasons.append('joint component has optimizer exclusion')
        result=dict(index=index,state_key=row['state_key'],family=row['family'],quotient=row['quotient'],cell=row['cell'],
            active_seeds=sum(bytes.fromhex(row['quotient'])[:14]),old_component=row['new_domain_component'],joint_component=cid,
            original_eligible=row['eligible'],eligible_for_future_ordinary_recipe=not reasons,reasons=reasons,
            disposition='quarantined' if reasons else 'projected_future_ordinary_support')
        dispositions.append(result)
        if not reasons:eligible.append(result)
    roles=selection['roles'];need({k:len(v) for k,v in roles.items()}==dict(fit=128,cases=32,arena=24),'Conditional role quotas changed')
    need({r['state_key'] for r in roles['cases']}<={r['state_key'] for r in roles['fit']},'Conditional cases no longer subset fit')
    fit_cells=Counter(r['cell'] for r in roles['fit']);case_cells=Counter(r['cell'] for r in roles['cases'])
    need(fit_cells==Counter({c:16 for c in ALL_CELLS}) and case_cells==Counter({c:4 for c in ALL_CELLS}),
         'Frozen fit/case cell quotas changed')
    arena_cells=Counter(r['cell'] for r in roles['arena'])
    expected_arena={c:(2 if '/ordinary/' in c else 4) for c in ALL_CELLS}
    need(arena_cells==Counter(expected_arena),'Frozen arena mechanism/actor/band quotas changed')
    support_components={joint['family_components'][r['family']] for r in pool}
    role_components={role:Counter(joint['family_components'][r['family']] for r in roles[role]) for role in ('fit','arena')}
    reservations=[]
    for role in ('fit','arena','cases'):
        for row in roles[role]:
            cid=joint['family_components'][row['family']];need(cid==joint['quotient_components'][row['quotient']],
                'Conditional root family/quotient ownership differs')
            c=joint['components'][cid];reasons=[]
            if c['known_components']:reasons.append('intersects latest known recorded graph through whole component')
            if cid in support_components:reasons.append('intersects full original512 support ownership')
            if role in ('fit','arena') and role_components[role][cid]!=1:reasons.append('distinct origin-component quota no longer holds')
            if role=='cases' and role_components['fit'][cid]!=1:reasons.append('case no longer inherits a distinct fit component')
            if cid in role_components['fit'] and cid in role_components['arena']:reasons.append('fit/arena components overlap')
            reservations.append(dict(role=role,state_key=row['state_key'],family=row['family'],cell=row['cell'],
                joint_component=cid,known_recorded_disjoint=not reasons,reasons=reasons))
    held=[r for r in reservations if r['reasons']]
    weights=_weights(eligible);weight_lookup={r['state_key']:{k:r[k] for k in ('numerator','denominator')} for r in weights['weights']}
    future_roots=[dict(r,weight=weight_lookup.get(r['state_key'])) for r in eligible]
    inventory_ref=dict(path=stage.relative(BASE),sha256=sha(BASE),marker_ledger_sha256=config.digest(base['marker_ledger']))
    return dict(schema=SCHEMA+'-projection',old_support_pool_roots=512,original_balanced512_recipe_available=False,
        original_support_training_admission=False,old_support_and_all_protections_retained=inventory_ref,
        original_512_dispositions=dispositions,whole_quarantine_joint_components=sorted(quarantine),
        projected_ordinary_support=eligible,projected_ordinary_support_count=len(eligible),
        old_cells=_counts(pool),projected_cells=_counts(eligible),prospective_weights=weights,
        future_support=dict(roots=future_roots,scope='ordinary-only known-provenance next-update development',
            metadata_only=True,recipe_executed=False,independent_component_sample_claim=False),
        ordinary_cells=list(ORDINARY_CELLS),original_512_recipe_admitted=False,
        fit_roots=128,fit_cases=32,arena_origins=24,evaluation_known_graph_disjoint=not held,
        exact_eval_quotas_satisfied=not held,ordinary_support_ready=weights['available'] and bool(eligible) and not held,
        readiness_scope='Metadata only; new recipe and explicit run required',inherited_unknown_background=True,
        balanced_bonus_capture_support_available=False,new_data_or_scope_change_required_for_balanced_training=True,
        conditional_role_revalidation=reservations,conditional_roles_known_recorded_disjoint=not held,
        conditional_role_collision_count=len(held),conditional_quotas_preserved=True,conditional_reservations_reselected=False,
        joint_component_constraint_count=len(overlay['components'])+len(selection['components']),
        joint_components=joint['components'],joint_family_components=joint['family_components'],
        joint_quotient_components=joint['quotient_components'],component_constraints_are_not_actual_incidence_edges=True,
        population_constraints_retained=len(selection['components']),all_counter_acquisition_dispositions_retained=True,
        population_optimizer_exclusion_is_not_optimizer_use=True,
        counter_ledger_source=dict(path=stage.relative(SELECTION),sha256=sha(SELECTION),pointer='/counter_disposition_ledger'),
        known_overlay_inputs=overlay['inputs'],input_pins=input_pins(),
        future_scope='ordinary-only known-provenance next-update development',historically_unseen_claim=False,
        inherited_model_ancestry_unknown=True,missing_historical_attempts_preserved=overlay['missing_attempts'],
        historical_provenance_complete=False,old_candidate_validation_repaired=False,
        new_recipe_and_explicit_run_required=True,training_authorized=False,models_fixed=True,policy_fixed=True,activity=stage.ZERO)


def final_projection(overlay,conditional_selection):return _derive_projection(overlay,conditional_selection)


def validate_projection(report,overlay,conditional_selection):
    need(config.canonical(report)==config.canonical(_derive_projection(overlay,conditional_selection)),
         'Prospective support projection differs');return True
