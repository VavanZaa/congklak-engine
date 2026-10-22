"""Narrow future-update prerequisites and a pure recorded-footprint gate.

No function here authorizes an optimizer, target producer or arena run. The
footprint primitive requires a complete, independently validated trace upstream.
"""
from fractions import Fraction
from . import config, neurodiag_data as metadata, ownership_domain_projection as domain
from . import ownership_resolution_common as s, ownership_resolution_contract as contract
from . import ownership_fast_guard as keys

SOURCES=('congklak/ownership_resolution_epoch.py','congklak/ownership_resolution_common.py',
         'congklak/ownership_resolution_contract.py','congklak/neurodiag_data.py',
         'congklak/ownership_domain_projection.py','congklak/ownership_fast_guard.py')
SELECTION='build/ownership-implementation-v1/reservations/production-v2/selection.json'
CERTIFICATE='build/ownership-implementation-v1/reservations/production-v2/certificate.json'


def input_pins():
    previous=s.read(s.PRIOR)['inputs']
    return {p:previous[p] for p in (SELECTION,CERTIFICATE)}


def selection():
    pins=input_pins();s.common.current(pins);return s.read(s.ROOT/SELECTION)


def _derive(history,support,overlay,joint):
    s.need(support['original_pool_roots']==512 and support['genuine_conflicting_roots']==320 and
        support['causal_paths_complete']==320 and support['original_pool_and_markers_unchanged'] is True,
        'Unresolved original support dispositions')
    s.need(overlay['ancestry_unknown'] is True and overlay['historical_provenance_complete'] is False and
        overlay['source_complete_for_declared_persisted'] is True and
        joint['historical_provenance_complete'] is False and joint['inherited_unknown_background'] is True and
        joint['original_512_recipe_admitted'] is False and joint['old_candidate_validation_repaired'] is False,
        'Epoch attempted to erase historical uncertainty or admit the old recipe')
    ready=(joint['ordinary_support_ready'] is True and joint['evaluation_known_graph_disjoint'] is True and
        joint['exact_eval_quotas_satisfied'] is True)
    roots=joint['future_support']['roots'];weights=joint['prospective_weights']
    if ready:
        s.need(weights['available'] is True and roots and
            sum(Fraction(r['weight']['numerator'],r['weight']['denominator']) for r in roots)==1,
            'Ordinary support probabilities differ')
    reason='retired only for the named prospective ordinary-development epoch' if ready else 'retained: prospective support/role quota or graph collision'
    result=dict(schema=s.SCHEMA+'-epoch',scope='ordinary-only known-provenance next-update development',
        contract_sha256=config.digest(contract.definition()),prerequisites_ready=ready,
        operational_hold_dispositions=[dict(id='320 prior support conflicts',resolution=reason,
            method='Permanent whole-component quarantine; original512 pool remains unchanged',
            original_conflicting_roots=320,original_pool_roots=512),
            dict(id='missing historical engineering footprints',resolution=reason,
            method='Permanent unknown ledger and historical-result exclusion; new inherited-background estimand',
            historical_coverage_repaired=False)],
        ordinary_support_roots=len(roots),ordinary_support_cells=joint['projected_cells'],
        sampler=weights,fit_roots=joint['fit_roots'],fit_cases=joint['fit_cases'],arena_origins=joint['arena_origins'],
        evaluation_known_graph_disjoint=joint['evaluation_known_graph_disjoint'],
        exact_eval_quotas_satisfied=joint['exact_eval_quotas_satisfied'],
        original_balanced512_recipe_available=False,bonus_capture_training_support_available=False,
        future_capture_bonus_regression_guards_required=True,ordinary_only_generalization_limit=True,
        inherited_unknown_background=True,inherited_model_ancestry=True,
        historical_provenance_complete=False,historically_unseen=False,
        legacy_strength_results_rehabilitated=False,old_candidate_quarantined=True,
        independent_strength_established=False,default_engine='H_v1',
        required_next_run=dict(explicit_new_recipe=True,fixed_model_sources=s.common.pins(s.MODELS),
            reserve_all_new_footprints_before_reuse=True,complete_trace_validator_upstream=True,
            reject_cross_role_components=True,no_untraced_retry=True,
            shared_endgames_frozen_max_active=24,all_owned_states_min_active=25,owned_upper_max_active=98,
            both_baselines_and_capture_endgame_guards=True,training_admitted_now=False),
        known_graph_projection_sha256=config.digest(overlay),joint_projection_sha256=config.digest(joint),
        historical_inventory_sha256=config.digest(history),support_inventory_sha256=config.digest(support),
        models_fixed=True,policy_fixed=True,training_authorized=False,
        labels_authorized=False,production_evaluation_admitted=False,model_promotion_authorized=False,activity=s.ZERO)
    result['epoch_id']=config.digest(result)
    return s.normalized(result)


def build(history,support,overlay,joint):return _derive(history,support,overlay,joint)


def validate(report,history,support,overlay,joint):
    s.need(config.canonical(report)==config.canonical(_derive(history,support,overlay,joint)),
        'Future epoch scope/ancestry/prerequisites differ')
    return True


def check_known_footprint(joint,role,state_key,family,quotients):
    """Pure reservation primitive; it is not a trace certificate or run permit.

    A future caller must first validate the full source-bound trace, then pass
    EVERY canonical quotient, including high hypothetical, feature and TT states.
    Missing or partial traces cannot be represented by an empty footprint.
    """
    s.need(role in ('optimizer_support','fit','arena'),'Unknown prospective role')
    s.need(type(state_key) is str and type(family) is str and
        isinstance(quotients,list) and bool(quotients),'Complete nonempty footprint required')
    if role=='optimizer_support':
        matches=[r for r in joint['future_support']['roots'] if r['state_key']==state_key and r['family']==family]
    else:
        matches=[r for r in joint['conditional_role_revalidation'] if r['role']==role and
                 r['state_key']==state_key and r['family']==family and r['known_recorded_disjoint'] is True]
    s.need(len(matches)==1,'Root/family absent from the named prospective role')
    try:raw=bytes.fromhex(state_key)
    except ValueError as exc:raise ValueError('Malformed raw root identity') from exc
    s.need(len(raw)==17,'Raw root identity width differs')
    root_q=keys.raw_key(raw[:16],raw[16]).hex()
    s.need(root_q in quotients,'Footprint omits its selected root')
    allowed=matches[0]['joint_component'];s.need(allowed not in joint['whole_quarantine_joint_components'],
        'Root is in a quarantined component')
    by_q=joint['joint_quotient_components'];components=joint['joint_components'];shared=0;high=0
    for q in quotients:
        active=domain.quotient_active(q)
        if active<=24:shared+=1;continue
        high+=1;cid=by_q.get(q)
        if cid is None:continue # A caller must durably claim this new q before another origin uses it.
        s.need(cid==allowed,'Recorded footprint crosses a reserved/protected component')
        if role=='optimizer_support':s.need(components[cid]['optimizer_blocked'] is False,
            'Optimizer footprint touches an excluded component')
    s.need(high>0,'An ordinary-only root cannot have an entirely low-state footprint')
    return dict(passed=True,role=role,owner_component=allowed,owned_observations=high,
        shared_observations=shared,unknown_high_quotients_require_durable_claim=True,
        full_trace_certificate=False,training_authorized=False,production_execution=False)
