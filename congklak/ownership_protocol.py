"""Executable contracts for a prospective switched-engine experiment.

These functions classify metadata and scratch fixtures. They never authorize
production admission, instantiate an engine, train, or reinterpret old results.
"""
from collections import defaultdict
from . import config, neurodiag_data as metadata

SCHEMA = 'prospective-ownership-contract-v1'
BOUNDARY = 24
PROFILES = ('control', 'candidate')
BASELINES = ('H_v1', 'H8_relay_CPU')
ROLES = {'optimizer_support', 'fit_check', 'arena', 'protected'}
RESET_FIELDS = ('tree', 'transposition_table', 'evaluator_cache', 'feature_cache',
                'ordering_history', 'root_policy', 'visits_values', 'rng_state')


def need(ok, message):
    if not ok: raise ValueError(message)


def active(state):
    state = config.position(state)
    return sum(map(sum, state['pits']))


def domain(state):
    a = active(state)
    return 'shared_frozen' if a <= BOUNDARY else 'owned_updated' if a <= 74 else 'owned_upper'


def root_admissible(state):
    state = config.position(state)
    return 25 <= active(state) <= 74 and sum(n > 0 for n in state['pits'][state['controller']]) >= 2


def dispatch(state, profile, baseline, seat, suffix_started=False):
    """Describe the REQUIRED future dispatcher; this does not execute/reset it."""
    state = config.position(state)
    need(profile in PROFILES and baseline in BASELINES and type(seat) is int and seat in (0, 1) and
         type(suffix_started) is bool, 'Unknown dispatch identity')
    a = active(state)
    need(not suffix_started or a <= BOUNDARY, 'An absorbing suffix cannot return to the owned domain')
    if a == 0:
        return dict(phase='terminal', engine=None, handoff_required=False, decision_required=False,
                    boundary_event_required=not suffix_started,
                    suffix_started=True, production_execution=False)
    need(any(state['pits'][state['controller']]), 'Normalize forced passes before dispatch')
    suffix = suffix_started or a <= BOUNDARY
    engine = ('control' if suffix else profile) if state['controller'] == seat else baseline
    return dict(phase='shared_frozen' if suffix else 'prefix', engine=engine,
                handoff_required=suffix and not suffix_started, decision_required=True,
                boundary_event_required=suffix and not suffix_started,
                suffix_started=suffix, reset_required=list(RESET_FIELDS),
                reset_scope='both_seats_at_handoff_and_every_decision',
                clock_ms=25, ownership_tracing_inside_clock=True,
                cache_capacity=0, production_execution=False)


def leaf_contract(state, profile):
    need(profile in PROFILES, 'Unknown leaf profile')
    a = active(state)
    return dict(value_path='frozen_original' if a <= 24 or a >= 75 or profile == 'control' else 'candidate_sidecar',
                owned=a >= 25, quotient=metadata.quotient(config.position(state)),
                cache_key='raw_state+actor+rules+full_model+encoder+backend+search_version',
                quotient_is_not_a_value_cache_key=True)


def _quotient(value):
    need(type(value) is str and len(value) == 30, 'Malformed ownership quotient')
    try: raw = bytes.fromhex(value)
    except ValueError as exc: raise ValueError('Malformed ownership quotient') from exc
    need(len(raw) == 15 and value == raw.hex() and raw[-1] in (0, 1) and sum(raw[:14]) <= 98,
         'Invalid ownership quotient geometry')
    other = raw[7:14] + raw[:7] + bytes([1 - raw[-1] if any(raw[:14]) else 0])
    need(raw <= other and (any(raw[:14]) or raw[-1] == 0), 'Quotient is not canonical')
    return sum(raw[:14])


def graph_cut(edges, coverage=None):
    """Cut actual low-q edges, preserving every origin vertex and high edge.

    Caller coverage is a scratch contract, never a source certificate. A real
    provenance builder and independent admission are still required afterward.
    """
    need(isinstance(edges, list), 'Incidence must be a list of actual edges')
    parent = {}; roles = defaultdict(set); arena_ids = defaultdict(set); low = []; high = []
    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node: parent[node] = find(parent[node])
        return parent[node]
    def join(left, right):
        left, right = find(left), find(right)
        if left != right: parent[right] = left
    for edge in edges:
        need({'family', 'quotient', 'roles', 'source'} <= set(edge) <= {'family', 'quotient', 'roles', 'source', 'arena_origin'} and
             type(edge['family']) is str and edge['family'] and type(edge['source']) is str and edge['source'] and
             isinstance(edge['roles'], list) and set(edge['roles']) <= ROLES,
             'Actual family/quotient incidence with source and roles is required')
        family = ('family', edge['family']); find(family)
        roles[edge['family']].update(edge['roles'])
        if 'arena' in edge['roles']:
            arena_origin = edge.get('arena_origin', edge['family'])
            need(type(arena_origin) is str and arena_origin, 'Reserved arena origin identity missing')
            arena_ids[edge['family']].add(arena_origin)
        else: need('arena_origin' not in edge, 'Arena origin annotation without arena role')
        a = _quotient(edge['quotient'])
        if a >= 25:
            join(family, ('quotient', edge['quotient'])); high.append(edge)
        else: low.append(edge)
    groups = defaultdict(lambda: dict(families=[], quotients=[], roles=set(), arena_origins=set()))
    for node in parent:
        group = groups[find(node)]
        group['families' if node[0] == 'family' else 'quotients'].append(node[1])
        if node[0] == 'family':
            group['roles'].update(roles[node[1]]); group['arena_origins'].update(arena_ids[node[1]])
    components = []
    conflicts = []
    for group in groups.values():
        body = {k: sorted(v) for k, v in group.items()}
        body['id'] = config.digest(body)
        # A protected family remains protected even if its observed low edges
        # are cut; it cannot become an unowned new family through that cut.
        conflict = len(body['arena_origins']) > 1 or len(set(body['roles']) & {'optimizer_support', 'fit_check', 'arena'}) > 1 or (
            'protected' in body['roles'] and bool(set(body['roles']) & {'optimizer_support', 'fit_check', 'arena'}))
        if conflict: conflicts.append(body['id'])
        components.append(body)
    components.sort(key=lambda v: v['id'])
    sufficient = bool(coverage and coverage.get('raw_incidence') is True and
        coverage.get('declared_sources') and coverage.get('declared_sources') == coverage.get('verified_sources') and
        coverage.get('missing_sources') == [] and coverage.get('unresolved_unions') == [])
    return dict(schema=SCHEMA + '-graph-fixture', components=components,
        owned_edges=len(high), shared_edges=len(low), shared_ledger=low,
        conflicting_components=sorted(conflicts), source_coverage_sufficient=sufficient,
        contract_graph_consistent=sufficient and not conflicts,
        actual_source_certificate=False, production_admission=False,
        historical_component_ids_reused=False, missing_incidence_means_hold=True)


def contract():
    return dict(schema=SCHEMA, prospective_only=True, retroactive_old_arena_reclassification=False,
        models_fixed=True, policy_fixed=True, training_authorized=False, execution_ready=False,
        boundary=dict(shared_max=24, owned_min=25, updated_max=74, upper_min=75,
                      primary_root_min=25, primary_root_max=74, normalized_choice_states_only=True),
        suffix=dict(learned_seat_engine='control', opposing_seat_engine='assigned_baseline',
                    arm_independent=True, irreversible=True, exact_oracle_at_24=False,
                    reset_fields=list(RESET_FIELDS), live_implementation_verified=False),
        ownership=dict(raw_origin_incidence_required=True, include_upper_states=True,
                       arena_reservation_identity_separate_from_source_family=True,
                       include_search_feature_and_tt_states=True, drop_low_edges_only=True,
                       flattened_component_splitting_forbidden=True, no_quotient_target_transfer=True,
                       missing_incidence_holds=True, old_exclusions_preserved=True),
        evaluation=dict(primary='finite_panel_terminal_margin_for_switched_composite',
                        families=24, games=192, ordinary=8, capture=8, bonus=8,
                        baselines=list(BASELINES), profiles=list(PROFILES), whole_decision_ms=25,
                        confidence_intervals_enabled=False, iid_components_assumed=False,
                        posthoc_filtering=False, posthoc_cluster_merging=False,
                        current_strength_interpretation_allowed=False),
        pending=['reconstruct_and_certify_actual_origin_incidence',
                 'instrument_complete_future_search_feature_tt_footprints',
                 'implement_and_audit_live_suffix_dispatch_reset_and_recovery',
                 'declare_generator_population_and_joint_metadata_reservations',
                 'freeze_separate_training_and_evaluation_treatment'])


def validate_spec(spec):
    """Bind the designer's section spec to this exact executable contract."""
    need(spec.get('schema') == 'prospective-ownership-section-spec-v1' and
         spec.get('machine_contract') == contract() and
         spec.get('training_authorized') is False and spec.get('execution_ready') is False and
         spec.get('old_arena_hold_preserved') is True, 'Prospective section spec differs from the fixed contract')
    return True


def fixture_checks():
    def state(a, actor=0):
        pits = [[0]*7, [0]*7]
        if a: pits[actor][0] = a
        return dict(pits=pits, stores=[98-a, 0], controller=actor)
    checks = 0
    for a in (0, 4, 5, 24, 25, 49, 50, 74, 75, 98):
        for actor in (0, 1):
            p = state(a, actor)
            for profile in PROFILES:
                for baseline in BASELINES:
                    for seat in (0, 1):
                        result = dispatch(p, profile, baseline, seat)
                        expected = None if not a else baseline if actor != seat else 'control' if a <= 24 else profile
                        need(result['engine'] == expected, 'Scratch dispatcher changed arm/seat semantics')
                        need(leaf_contract(p, profile)['owned'] == (a >= 25), 'Scratch leaf domain mismatch')
                        checks += 1
    q5, q25, q75 = [metadata.quotient(state(a)) for a in (5, 25, 75)]
    def edge(f, q, r=()): return dict(family=f, quotient=q, roles=list(r), source='scratch')
    coverage = dict(raw_incidence=True, declared_sources=['scratch'], verified_sources=['scratch'],
                    missing_sources=[], unresolved_unions=[])
    low_link = graph_cut([edge('a',q5),edge('b',q5)], coverage)
    need(len(low_link['components']) == 2 and low_link['shared_edges'] == 2, 'Low-only quotient joins families')
    same_origin = graph_cut([edge('a',q25),edge('a',q75),edge('b',q75)], coverage)
    need(len(same_origin['components']) == 1 and same_origin['owned_edges'] == 3, 'High origin or upper alias dropped')
    conflict = graph_cut([edge('a',q25,['optimizer_support']),edge('b',q25,['arena'])],coverage)
    need(bool(conflict['conflicting_components']), 'Cross-role high-domain bridge admitted')
    missing = graph_cut([edge('a',q25)])
    need(not missing['source_coverage_sufficient'] and not missing['production_admission'], 'Missing source coverage admitted')
    arena_collision = graph_cut([edge('a',q25,['arena']),edge('b',q25,['arena'])],coverage)
    need(bool(arena_collision['conflicting_components']), 'Same-role cross-arena collision admitted')
    same_arena = [dict(edge(f,q25,['arena']),arena_origin='one_reserved_origin') for f in ('a','b')]
    need(not graph_cut(same_arena,coverage)['conflicting_components'], 'One reserved origin counted twice')
    return dict(schema=SCHEMA + '-fixture-checks', passed=True, dispatch_leaf_cases=checks,
                graph_cases=6, production_engine_calls=0, optimizer_steps=0, new_games=0,
                searches=0, new_teacher_labels=0, feature_simulations=0,
                live_suffix_worker_tested=False, production_admission=False)
