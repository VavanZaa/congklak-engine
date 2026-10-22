"""Independent allocation traces, prospective reservations and H12 assessment."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse, ast, copy, hashlib, json, math, os, shutil, sqlite3, struct
import subprocess, sys, time, unittest, uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from congklak import config
from congklak.native import NativeError
from data_pilot_independent_audit import sha, load, save, relative, pins, dbopen, norm, position, mechanism, rows, run_child
from divergence_independent_audit import backup, kill, sql_rows, patch_sql, bits, semantic, teacher_semantic
from stage5_deeper_audit_acceptance import value_and_policy
from natural_study_independent_audit import check_pvs
import reference as literal

WORK = ROOT / 'build/allocation-audit'
BASELINE = ROOT / 'build/allocation-baseline.json'
FREEZE = ROOT / 'build/allocation-audit-source-freeze.json'
MODEL = ROOT / 'data/quality-train-v1/model.json'
CANDIDATE = ROOT / 'data/ordinary-value-pilot-v1/model.json'
EVIDENCE = {}
STRATA = ('50-74/ordinary', '25-49/ordinary', '25-49/bonus', '25-49/capture')
CELLS = tuple(s + '/actor' + str(a) for s in STRATA for a in (0, 1))
GROUPS = {
 'preflight': ['test_01_prior4255_panel_roles_and_predeclared_rules',
               'test_02_trace_real_pre_post_simulation_kills_and_semantic_resume',
               'test_03_teacher_real_pre_post_attempt_kills_and_H12_gate'],
 'trace': ['test_04_all8_production_traces_literal_selection_backup_and_parity'],
 'labels': ['test_05_assessment_H12_fullQ_histories_and_literal_PVs',
            'test_06_H8_H10_rule_weights_H12_denominators_and_sensitivity'],
 'final': ['test_07_rehashed_corruption_missing_publication_and_lookahead_refusal',
           'test_08_closed_API_CLI_no_work_and_prior_preservation']}


def raw(p):
    return bytes([*p['pits'][0], p['stores'][0], *p['pits'][1], p['stores'][1], p['controller']]).hex()


def quotient(p):
    a, b = tuple(p['pits'][0]), tuple(p['pits'][1])
    actor = p['controller'] if any(a + b) else 0
    return min(bytes(a + b + (actor,)), bytes(b + a + ((1 - actor) if any(a + b) else 0,))).hex()


def metadata_rows(journal, table, key='state'):
    with closing(dbopen(ROOT / 'data' / journal / 'work.sqlite3')) as db:
        return rows_from_db(db, table, key)


def rows_from_db(db, table, key):
    result = {}
    for ident, text, checksum in db.execute('SELECT ' + key + ',body,hash FROM ' + table + ' ORDER BY ' + key):
        item = json.loads(text)
        assert config.digest(item) == checksum, 'Corrupt accepted metadata'
        result[ident] = item
    return result


def requested_keys(journal, stage, include_attempts=False):
    with closing(dbopen(ROOT / 'data' / journal / 'work.sqlite3')) as db:
        tables = {r[0] for r in db.execute('SELECT name FROM sqlite_master')}
        wanted = [stage + '_labels', stage + '_references']
        if include_attempts:
            wanted.append(stage + '_attempts')
        return set().union(*(set(r[0] for r in db.execute('SELECT state FROM ' + table))
                             for table in wanted if table in tables))


def exact_weight(h8, h10, legal):
    left, right = h8['label'], h10['label']
    q8 = {r['pit']: r['value_player0'] for r in left['action_scores']}
    q10 = {r['pit']: r['value_player0'] for r in right['action_scores']}
    assert set(q8) == set(q10) == set(legal)
    assert all(type(x) is int for x in [left['target_player0'], right['target_player0'], *q8.values(), *q10.values()])
    dv = abs(right['target_player0'] - left['target_player0'])
    dq = sum((F(abs(q10[a] - q8[a])) for a in legal), F()) / len(legal)
    return dv, dq, max(F(1, 4), F(4, 1) / (4 + dv + dq))


def rational(value):
    value = F(value)
    return dict(numerator=value.numerator, denominator=value.denominator, value=float(value))


def independent_inventory():
    """Reconstruct component ownership from metadata, without H12 target bodies."""
    old = load(ROOT / 'build/ordinary-value-registry.json')
    protected_q, protected_f = set(old['quotients']), set(old['families'])
    observation_count = 0
    with closing(dbopen(ROOT / 'data/ordinary-value-benchmark-v1/work.sqlite3')) as db:
        games = rows_from_db(db, 'games', 'idx')
        assert len(games) == 96
        for text, checksum in db.execute('SELECT body,hash FROM decisions ORDER BY game,choice'):
            item = json.loads(text); assert config.digest(item) == checksum
            for p in (item['state'], item['after']):
                protected_q.add(quotient(p)); observation_count += 1
        for game in games.values():
            protected_f.add(game['family']); protected_q.add(quotient(game['final']))
    for row in load(ROOT / 'build/divergence-panel.json')['roots']:
        protected_q.add(quotient(row['state'])); protected_f.add(row['family'])
    exposed, exposed_q, exposed_f = set(), set(), set()
    for journal in ('horizon-probe-v1', 'horizon-finish-v1', 'midgame-label-pilot-v1',
                    'ordinary-value-labels-v1', 'divergence-teacher-v1'):
        keys = set().union(*(requested_keys(journal, stage, True) for stage in ('h12', 'h14')))
        exposed.update(keys)
        for key, row in metadata_rows(journal, 'roots').items():
            if key in keys:
                exposed_q.add(quotient(row['state'])); exposed_f.add(row['family'])
    candidates = {}
    for journal in ('natural-study-v1', 'midgame-label-pilot-v1'):
        support = requested_keys(journal, 'h8') & requested_keys(journal, 'h10')
        for key, row in metadata_rows(journal, 'roots').items():
            if key in support and row['cohort'] == 'training_candidate':
                assert key not in candidates
                candidates[key] = dict(source=journal, source_row=row,
                    source_wrapper_sha256=config.digest(row), state_key=key,
                    state=row['state'], family=row['family'], stratum=row['stratum'],
                    actor=row['state']['controller'], legal_mask=row['legal_mask'],
                    source_cohort=row['cohort'])
    # Undirected graph traversal is independent of the producer's disjoint-set union.
    graph = defaultdict(set); owners = {}; candidate_observations = 0
    for journal in ('ordinary-data-pilot-v1', 'natural-study-v1', 'midgame-data-pilot-v1'):
        for family in metadata_rows(journal, 'families', 'idx').values():
            fid = family['family']; assert fid not in owners
            owners[fid] = family['cohort']; node = ('family', fid); graph[node]
            for row in family['candidates']:
                if row['stratum'] in STRATA and row['legal_mask'].bit_count() >= 2:
                    q = ('quotient', quotient(row['state']))
                    graph[node].add(q); graph[q].add(node); candidate_observations += 1
    components = {}; by_family = {}; seen = set()
    for first in graph:
        if first in seen:
            continue
        pending = [first]; seen.add(first); found = []
        while pending:
            at = pending.pop(); found.append(at)
            for other in graph[at]:
                if other not in seen:
                    seen.add(other); pending.append(other)
        families = sorted(v for kind, v in found if kind == 'family')
        qs = sorted(v for kind, v in found if kind == 'quotient')
        cohorts = sorted({owners[f] for f in families}); reasons = []
        if set(families) & protected_f:
            reasons.append('protected_origin_family')
        if set(qs) & protected_q:
            reasons.append('protected_eligible_quotient_bridge')
        if set(families) & exposed_f or set(qs) & exposed_q:
            reasons.append('prior_requested_H12_H14_exposed_component')
        if len(cohorts) > 1:
            reasons.append('mixed_source_ownership_component')
        body = dict(families=families, quotients=qs, source_cohorts=cohorts, blocked_reasons=reasons)
        cid = config.digest(body); components[cid] = body
        by_family.update({fid: cid for fid in families})
    eligible = {}; excluded = {}
    for key, row in candidates.items():
        cid = by_family[row['family']]; reasons = list(components[cid]['blocked_reasons'])
        if key in exposed:
            reasons.append('prior_requested_H12_H14_exposure')
        if row['stratum'] not in STRATA or row['legal_mask'].bit_count() < 2:
            reasons.append('outside_declared_choice_support')
        item = dict(row, component=cid, cell=row['stratum'] + '/actor' + str(row['actor']), quotient=quotient(row['state']))
        if reasons:
            excluded[key] = dict(item, reasons=reasons)
        else:
            eligible[key] = item
    return dict(eligible=eligible, excluded=excluded, components=components,
                protected_quotients=sorted(protected_q), protected_families=sorted(protected_f),
                prior_H12_H14_exposed_rawkeys=sorted(exposed),
                prior_H12_H14_exposed_quotients=sorted(exposed_q),
                prior_H12_H14_exposed_families=sorted(exposed_f),
                raw_H8_H10_training_roots=len(candidates), eligible_roots=len(eligible),
                eligible_families=len({r['family'] for r in eligible.values()}),
                eligible_cells=dict(sorted(Counter(r['cell'] for r in eligible.values()).items())),
                source_families=len(owners), eligible_candidate_observations=candidate_observations,
                blocked_source_families=sum(len(c['families']) for c in components.values() if c['blocked_reasons']),
                latest_evaluation_state_observations=observation_count)


def independent_selection(inv):
    """Deterministic augmenting-path matching over distinct components."""
    rank = lambda salt, key: hashlib.sha256((salt + key).encode()).hexdigest()
    selection_salt = 'allocation-weighting-panel/20261030/v1/'
    split_salt = 'allocation-weighting-development-assessment/20261030/v1/'
    options = {cell: sorted((r for r in inv['eligible'].values() if r['cell'] == cell),
                key=lambda r: (rank(selection_salt, r['state_key']), r['state_key'])) for cell in CELLS}
    owner, chosen = {}, {}
    def place(slot, touched):
        for candidate in options[slot[0]]:
            component = candidate['component']
            if component in touched:
                continue
            touched.add(component)
            if component not in owner or place(owner[component], touched):
                owner[component] = slot; chosen[slot] = candidate; return True
        return False
    for slot in [(cell, index) for cell in CELLS for index in range(2)]:
        assert place(slot, set()), 'Independent metadata matching shortfall'
    result = {}
    for cell in CELLS:
        pair = sorted((chosen[cell, i] for i in range(2)), key=lambda r: (rank(split_salt, r['state_key']), r['state_key']))
        for group, row in zip(('development', 'assessment'), pair):
            result[row['state_key']] = dict(row, group=group, future_optimizer_reserved=True)
    return dict(sorted(result.items()))


def literal_transition(before, pit):
    t = literal.apply(before, pit)
    last = next(e for e in reversed(t.events) if e['kind'] == 'deposit')['index']
    reason = ('own_store' if t.bonus else 'own_capture' if t.captured else
              'own_empty' if last in literal.row_indices(7, t.actor) else 'opponent_empty')
    return t, dict(position=position(t.position), reason=reason,
                   store_delta=[t.position.counts[7] - before.counts[7], t.position.counts[15] - before.counts[15]],
                   passes=[dict(zip(('from', 'to'), pair)) for pair in t.final_passes],
                   relays=t.relay_pickups, drops=t.deposits, terminal=t.terminal)


def audit_trace_worker(spec):
    """Replay saved predictions through independent dictionaries and literal rules."""
    from congklak import allocation_trace as producer
    from congklak.puct_cpu_inference import COUNTERS
    from stage5_deeper_audit_acceptance import literal_features
    roots, stores, manifest = producer.load_trace(Path(spec['run']), fixture=spec.get('fixture', False))
    plan, admitted_roots, units, identity = producer.admit(fixture=spec.get('fixture', False))
    assert roots == admitted_roots
    models = {'control': load(MODEL)['model'], 'candidate': load(CANDIDATE)['model']}
    predictions = {}; checked_predictions = 0; internal_choices = 0; bonus_edges = 0
    total_simulations = 0; reference_rows = []; decision_paths = {}; final_nodes = {}

    def same(a, b):
        assert a == b, 'Independent exact replay differs'

    def check_prediction(prediction, counters, state, is_root):
        nonlocal checked_predictions
        if prediction is None:
            same(counters, {key: 0 for key in COUNTERS}); return
        key = (profile, raw(state))
        if key not in predictions:
            value, _, logits = value_and_policy(norm(state), models[profile])
            x, _, _ = literal_features(norm(state), 127)
            simulations = sum(n > 0 for side in state['pits'] for n in side)
            deposits = sum(int(round(x[43 + i * 6 + 1] * 98)) for i in range(14))
            predictions[key] = (value, logits, simulations, deposits); checked_predictions += 1
        value, logits, simulations, deposits = predictions[key]
        assert abs(prediction['value_player0'] - value) <= 1e-10
        assert all(abs(a - b) <= 1e-10 for a, b in zip(prediction['policy_logits'], logits))
        assert len(prediction['policy_logits']) == len(logits) == 7
        same(prediction['feature_simulations'], simulations); same(prediction['feature_deposits'], deposits)
        expected = {key: 0 for key in COUNTERS}
        for field in ('positions_submitted', 'evaluations_requested', 'native_actual_evals',
                      'predict_successes', 'native_calls', 'cache_misses'):
            expected[field] = 1
        expected.update(feature_simulations=simulations, feature_deposits=deposits, cache_resets=int(is_root))
        same(counters, expected)

    def node(state, index, prediction):
        q = norm(state); terminal = literal.normalize(q).terminal
        legal = tuple(literal.legal_actions(q)); bootstrap = literal.evaluate(q) / 98
        priors = [0.] * 7
        if not terminal:
            assert prediction is not None
            bootstrap = prediction['value_player0'] / 98
            top = max(prediction['policy_logits'][a] for a in legal)
            exp = {a: math.exp(prediction['policy_logits'][a] - top) for a in legal}
            denominator = sum(exp.values()); priors = [exp.get(a, 0.) / denominator for a in range(7)]
        return dict(id=index, q=q, actor=q.controller, legal=legal, terminal=terminal,
                    bootstrap=bootstrap, priors=priors, visits=[0] * 7, sums=[0.] * 7, children={})

    def packed(n):
        return dict(id=n['id'], state=position(n['q']), actor=n['actor'], legal=list(n['legal']),
                    terminal=n['terminal'], bootstrap_normalized=n['bootstrap'], priors=n['priors'])

    def select(n):
        total = sum(n['visits']); sign = 1 if n['actor'] == 0 else -1; scores = []
        for pit in n['legal']:
            observed = n['sums'][pit] / n['visits'][pit] if n['visits'][pit] else None
            q = observed if observed is not None else n['bootstrap']
            exploration = .25 * n['priors'][pit] * math.sqrt(max(1, total)) / (1 + n['visits'][pit])
            value = sign * q + exploration
            scores.append(dict(pit=pit, visits=n['visits'][pit], prior=n['priors'][pit],
                observed_Q_player0_normalized=observed, parent_bootstrap_player0_normalized=n['bootstrap'],
                Q_source='visited_mean' if observed is not None else 'parent_bootstrap_FPU',
                selected_Q_player0_normalized=q, actor_signed_Q_normalized=sign*q,
                exploration_normalized=exploration, total_score_normalized=value,
                total_score_seed_display=value*98))
        chosen = min(scores, key=lambda r: (-r['total_score_normalized'], r['pit']))['pit']
        return dict(actor=n['actor'], total_visits=total, rows=scores, chosen=chosen,
                    tie_break='Maximum normalized score, then lowest pit; seed display is never used for selection')

    def leader(n):
        count = sum(n['visits'])
        return min(n['legal'], key=lambda a: (-n['visits'][a], -n['priors'][a] if not count else 0, a))

    for unit in units:
        cell = unit['index']; profile = unit['profile']; root = roots[unit['root_index']]
        setup = stores['root_setups'][cell]
        same(setup['unit'], unit); same(setup['root_sha256'], config.digest(root))
        check_prediction(setup['prediction'], setup['physical_counts'], root['state'], True)
        nodes = {0: node(root['state'], 0, setup['prediction'])}; same(setup['root_node'], packed(nodes[0]))
        work = dict(transitions=0, deposits=0, feature_simulations=0, feature_deposits=0,
                    neural_evaluations=0, neural_calls=0, root_expansions=1)
        counts = {key: 0 for key in COUNTERS}
        def charge(prediction, physical):
            if prediction is not None:
                for key in ('feature_simulations', 'feature_deposits'):
                    work[key] += prediction[key]
                work['neural_evaluations'] += 1; work['neural_calls'] += 1
            for key in COUNTERS:
                counts[key] += physical[key]
        charge(setup['prediction'], setup['physical_counts'])
        decision_paths[cell] = []
        for index, step in enumerate(stores['steps'][cell], 1):
            same(step['cell'], cell); same(step['simulation'], index)
            same(step['root_selection'], select(nodes[0]))
            current = nodes[0]; path = []; created = False; selections = []
            for level, (node_id, actor, pit) in enumerate(step['path']):
                same((node_id, actor), (current['id'], current['actor']))
                details = select(current); same(pit, details['chosen']); internal_choices += 1; path.append((node_id, pit))
                selections.append(dict(depth=level, state=position(current['q']), state_key=raw(position(current['q'])), node_id=node_id, selection=details))
                if pit not in current['children']:
                    assert level == len(step['path']) - 1
                    new = step['new_edge']; assert new is not None
                    same((new['parent'], new['pit'], new['child']), (node_id, pit, len(nodes)))
                    t, transition = literal_transition(current['q'], pit)
                    same(new['transition'], transition)
                    state = position(t.position); check_prediction(step['prediction'], step['physical_counts'], state, False)
                    child = node(state, len(nodes), step['prediction']); same(new['node'], packed(child))
                    bonus_edges += int(child['actor'] == current['actor'] and not child['terminal'])
                    current['children'][pit] = child['id']; nodes[child['id']] = child
                    current = child; created = True; work['transitions'] += 1; work['deposits'] += t.deposits
                else:
                    current = nodes[current['children'][pit]]
                    assert level < len(step['path']) - 1 or current['terminal']
            assert path
            if not created:
                assert step['new_edge'] is None and current['terminal'] and step['prediction'] is None
                check_prediction(None, step['physical_counts'], position(current['q']), False)
            charge(step['prediction'], step['physical_counts'])
            raw_value = literal.evaluate(current['q']) if current['terminal'] else step['prediction']['value_player0']
            same(step['leaf'], dict(node_id=current['id'], state_key=raw(position(current['q'])),
                 state=position(current['q']), controller=current['actor'], terminal=current['terminal'],
                 source='terminal_store_margin' if current['terminal'] else 'network',
                 raw_value_player0=raw_value, backup_normalized=current['bootstrap'],
                 backup_player0_display=current['bootstrap']*98))
            for node_id, pit in path:
                nodes[node_id]['visits'][pit] += 1; nodes[node_id]['sums'][pit] += current['bootstrap']
            same(step['root_after'], dict(visits=nodes[0]['visits'], sums_normalized=nodes[0]['sums'], visit_leader=leader(nodes[0])))
            decision_paths[cell].append(selections)
        completed = len(stores['steps'][cell]); saved = stores['cells'][cell]
        same(completed, unit['simulations']); n = nodes[0]; count = sum(n['visits'])
        report = dict(schema='quality-puct-report-v1', protocol='quality/fixed-player0/parent-bootstrap-fpu/legal-softmax/declared-cpuct-v1',
             rules_digest=config.RULES_DIGEST, position=position(n['q']), chosen=leader(n), visits=n['visits'],
             policy=[v/count for v in n['visits']], root_priors=n['priors'], estimated_value_player0=sum(n['sums'])/count*98,
             action_mean_value_player0=[n['sums'][a]/n['visits'][a]*98 if n['visits'][a] else None for a in range(7)],
             root_expanded=True, completed_simulations=completed, requested_simulations=unit['simulations'], nodes=len(nodes),
             max_nodes=unit['max_nodes'], usable_decision=True, memory_mb=8192, stop_reason='simulations', proof_status='monte_carlo_estimate',
             work={**work, 'total_deposits':work['deposits']+work['feature_deposits']}, forest_batch_calls=work['neural_calls'],
             forest_size=1, value_perspective='fixed_player0_seeds', first_play_urgency='parent_neural_bootstrap',
             unvisited_root_q_player0=n['bootstrap']*98, c_puct=.25)
        same(saved['semantic_report'], report); same(saved['physical_counts'], counts)
        if not spec.get('fixture'):
            same(report, plan['references'][profile+'/'+str(unit['simulations'])])
        total_simulations += completed
        final_nodes[cell] = nodes[0]
        reference_rows.append(dict(cell=cell, profile=profile, simulations=completed,
                                   chosen=report['chosen'], semantic_sha256=config.digest(report), nodes=len(nodes)))
    allocation = []
    for ri, root in enumerate(roots):
        for budget in sorted({u['simulations'] for u in units if u['root_index']==ri}):
            pair = {u['profile']:u for u in units if u['root_index']==ri and u['simulations']==budget}
            records = {p:stores['steps'][u['index']] for p,u in pair.items()}
            count = min(map(len, records.values())); first_root = first_internal = None
            matched = different = 0; switches = {p:[] for p in pair}; all_visited = {p:None for p in pair}; fpu = {p:0 for p in pair}
            leaders = {p:min(final_nodes[u['index']]['legal'],key=lambda a:(-stores['root_setups'][u['index']]['root_node']['priors'][a],a)) for p,u in pair.items()}
            for index in range(count):
                a,b = records['control'][index],records['candidate'][index]
                matched += a['leaf']['state_key']==b['leaf']['state_key']; different += a['leaf']['state_key']!=b['leaf']['state_key']
                for profile, entries in records.items():
                    record=entries[index]; chosen=record['root_after']['visit_leader']
                    if chosen!=leaders[profile]:switches[profile].append(dict(simulation=index+1,from_pit=leaders[profile],to_pit=chosen,visits=record['root_after']['visits']))
                    leaders[profile]=chosen
                    chosenrow=next(r for r in record['root_selection']['rows'] if r['pit']==record['root_selection']['chosen'])
                    fpu[profile]+=chosenrow['Q_source']=='parent_bootstrap_FPU'
                    if all_visited[profile] is None and all(record['root_after']['visits'][a]>0 for a in final_nodes[pair[profile]['index']]['legal']):all_visited[profile]=index+1
                if first_root is None and a['root_selection']['chosen']!=b['root_selection']['chosen']:
                    first_root=dict(simulation=index+1,state_key=root['state_key'],control=a['root_selection'],candidate=b['root_selection'])
                if first_internal is None:
                    for left,right in zip(decision_paths[pair['control']['index']][index],decision_paths[pair['candidate']['index']][index]):
                        if left['state_key']!=right['state_key']:break
                        if left['selection']['chosen']!=right['selection']['chosen']:
                            if left['depth']>0:first_internal=dict(simulation=index+1,depth=left['depth'],state=left['state'],state_key=left['state_key'],control=left['selection'],candidate=right['selection'])
                            break
            sampled={}
            for profile,unit in pair.items():
                n=final_nodes[unit['index']]; observed={a:n['sums'][a]/n['visits'][a] for a in n['legal'] if n['visits'][a]}
                best=(max if n['actor']==0 else min)(observed.values())
                ties=[a for a,q in observed.items() if q==best]
                sampled[profile]=dict(visit_winner=leader(n),actor_best_among_sampled_means=ties,
                    unvisited_legal_actions=[a for a in n['legal'] if not n['visits'][a]],vote_winner_in_best_sampled_mean=leader(n) in ties)
            allocation.append(dict(root_id=root['id'],state_key=root['state_key'],simulations=budget,eligible_models=2,completed_models=2,
                first_root_allocation_difference=first_root,first_internal_difference_on_shared_path=first_internal,
                matching_leaf_states_at_same_simulation=matched,different_leaf_states_at_same_simulation=different,
                visit_leader_switches=switches,first_all_root_actions_visited_after_simulation=all_visited,root_selected_FPU_uses=fpu,
                final_vote_vs_sampled_mean=sampled,
                FPU_limit='After every root action has a visit, FPU is no longer directly used at the root; it remains active on unvisited descendant edges. Allocation is not attributed to root FPU alone.',
                control_final_chosen=leader(final_nodes[pair['control']['index']]),candidate_final_chosen=leader(final_nodes[pair['candidate']['index']])))
    same(manifest['allocation'],allocation)
    return dict(schema='allocation-independent-literal-trace-v1', passed=True, cells=len(units),
                root_simulations=total_simulations, internal_selection_backups=internal_choices,
                same_actor_new_edges=bonus_edges, unique_literal_model_predictions=checked_predictions,
                references=reference_rows, allocation=allocation, all_root_and_internal_scores_rederived=True,
                all_transitions_seed_by_seed=True, fixed_player0_bonus_backup_verified=True,
                native_inference_for_audit=0, new_teacher_searches=0, new_games=0,
                training_updates=0, torch_imported=False)


def from_rational(value):
    assert set(value)=={'numerator','denominator','value'}
    assert type(value['numerator']) is type(value['denominator']) is int and value['denominator']>0
    answer=F(value['numerator'],value['denominator'])
    assert rational(answer)==value
    return answer


def independent_metric(values):
    complete=[r for r in values if r['H12_complete']]
    allmass=sum((from_rational(r['weight']) for r in values),F())
    mass=sum((from_rational(r['weight']) for r in complete),F()); n=len(complete)
    fields=('root_drift','action_Q_drift','H10_best_H12_regret_min','H10_best_H12_regret_max','H10_best_H12_regret_mean')
    uniform={}; weighted={}
    for field in fields:
        uniform[field]=rational(sum((from_rational(r[field]) for r in complete),F())/n) if n else None
        weighted[field]=rational(sum((from_rational(r[field])*from_rational(r['weight']) for r in complete),F())/mass) if n else None
    ess=mass**2/sum((from_rational(r['weight'])**2 for r in complete),F()) if n else None
    return dict(requested_roots=len(values),complete_roots=n,missing_roots=len(values)-n,
        requested_state_keys=sorted(r['state_key'] for r in values),complete_state_keys=sorted(r['state_key'] for r in complete),
        missing_state_keys=sorted(r['state_key'] for r in values if not r['H12_complete']),original_weight_mass=rational(allmass),
        complete_weight_mass=rational(mass),missing_original_weight_mass=rational(allmass-mass),
        missing_original_weight_fraction=rational((allmass-mass)/allmass) if allmass else None,
        ESS=rational(ess) if n else None,relative_ESS=rational(ess/n) if n else None,uniform=uniform,weighted=weighted,
        support='Identical requested-H12 complete intersection for both averages; missing remains null, not zero')


def independent_mass(values):
    total=sum((from_rational(r['weight']) for r in values),F())
    cmass=sum((from_rational(r['weight']) for r in values if r['H12_complete']),F()); result={}
    for field in ('group','stratum','actor','cell'):
        result[field]={}
        for key in sorted({str(r[field]) for r in values}):
            subset=[r for r in values if str(r[field])==key]
            mass=sum((from_rational(r['weight']) for r in subset),F())
            complete=sum((from_rational(r['weight']) for r in subset if r['H12_complete']),F())
            result[field][key]=dict(requested_roots=len(subset),complete_roots=sum(r['H12_complete'] for r in subset),
                original_fraction=rational(mass/total),complete_mass_fraction_of_original=rational(complete/total),
                complete_intersection_fraction=rational(complete/cmass) if cmass else None)
    return result


class AllocationIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.base = WORK / ('retained-' + uuid.uuid4().hex); cls.base.mkdir()

    def close(self, a, b):
        self.assertAlmostEqual(a, b, delta=1e-10)

    def current(self, values):
        for key, checksum in values.items():
            self.assertEqual(sha(ROOT / key), checksum, key)

    def worker(self, spec, path):
        path.mkdir(); spec = dict(spec, output=str(path / 'proof.json'))
        save(path / 'spec.json', spec)
        run_child([sys.executable, str(ROOT / 'tools/allocation-audit.py'), str(path / 'spec.json')], path / 'worker.log', timeout=900)
        proof = load(path / 'proof.json'); self.assertTrue(proof['passed'])
        return dict(path=relative(path / 'proof.json'), **proof)

    def clone(self, source, name):
        result = self.base / name; shutil.copytree(source, result); return result

    def command(self, kind, path):
        if kind == 'trace':
            return [sys.executable, str(ROOT/'tools/trace-allocation.py'), 'run', '--fixture', '--path', str(path)]
        return [sys.executable, str(ROOT/'tools/validate-allocation-weighting.py'), 'run', '--fixture', '--run', str(path)]

    def closed_worker(self, kind, path, output, fixture=False, watch_prior=False):
        from congklak import allocation_trace as trace, allocation_weighting as weighting
        producer = trace if kind == 'trace' else weighting
        calls = [dict(function='run', args=[str(path)], kwargs=dict(fixture=fixture))]
        calls.append(dict(function='verify', args=[str(path)], kwargs=dict(fixture=fixture)) if kind == 'trace'
                     else dict(function='load_rows', args=[str(path)]))
        extra = []
        if not fixture:
            extra = [relative(trace.OUTPUT), relative(trace.RESOURCES)] if kind == 'trace' else [relative(weighting.OUTPUT)]
            if kind != 'trace':
                calls += [dict(function='assess'), dict(function='verify')]
                extra += [relative(trace.OUTPUT),relative(trace.RESOURCES)]
        cli = 'tools/trace-allocation.py' if kind == 'trace' else 'tools/validate-allocation-weighting.py'
        argv = ['verify', '--path', str(path)] if kind == 'trace' else ['verify-labels', '--run', str(path)]
        if fixture and kind == 'trace':
            argv.append('--fixture')
        return self.worker(dict(kind='closed', module=producer.__name__, run=str(path),
                 calls=calls, cli=cli, argv=argv, extra_files=extra, watch_baseline=watch_prior,
                 watch_directories=[str(trace.DEFAULT)] if not fixture and kind!='trace' else [],
                 forbidden_attributes=['CpuBatchEvaluator','PerformanceCore','_advance'] if kind=='trace' else ['history.execute'],
                 input_names=[relative(producer.PLAN), relative(producer.IMPLEMENTATION),
                              relative(path/'work.sqlite3'), relative(path/'manifest.json'), relative(FREEZE),
                              'tests/allocation_independent_audit.py']), output)

    def test_01_prior4255_panel_roles_and_predeclared_rules(self):
        from congklak import allocation_trace as trace, allocation_weighting as weighting
        b = load(BASELINE); self.assertEqual((b['inputs_count'], b['prior_accepted_inputs']), (4255,4252))
        self.assertEqual(len(b['inputs']), 4255); self.current(b['inputs'])
        self.assertEqual(sha(MODEL), 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558')
        a, c = load(MODEL)['model']['weights'], load(CANDIDATE)['model']['weights']
        self.assertEqual({key for key in a if bits(a[key]) != bits(c[key])}, {'layers.4.weight','layers.4.bias'})
        inv = independent_inventory(); actual = load(weighting.INVENTORY)
        for field, value in inv.items():
            self.assertEqual(actual[field], value, field)
        self.assertEqual((inv['eligible_roots'],inv['eligible_families'],inv['source_families']), (62,53,1280))
        selected = independent_selection(inv); registry = load(weighting.REGISTRY); self.assertEqual(registry['rows'], selected)
        self.assertEqual(len({r['component'] for r in selected.values()}),16)
        self.assertEqual(len({r['family'] for r in selected.values()}),16)
        self.assertEqual(Counter((r['group'],r['cell']) for r in selected.values()), Counter({(g,c):1 for g in ('development','assessment') for c in CELLS}))
        rules = load(weighting.RULE); reservations = load(weighting.RESERVATIONS)
        self.assertEqual(registry['inventory_sha256'],sha(weighting.INVENTORY))
        self.assertEqual(rules['registry_sha256'],sha(weighting.REGISTRY));self.assertEqual(reservations['registry_sha256'],sha(weighting.REGISTRY))
        oldkeys=set(load(ROOT/'build/quality-data.json')['train_state_keys'])
        def decode(key):
            b=bytes.fromhex(key);return dict(pits=[list(b[:7]),list(b[8:15])],stores=[b[7],b[15]],controller=b[16])
        oldq={quotient(decode(k)) for k in oldkeys}
        current=[r for group in load(ROOT/'build/ordinary-value-rows.json')['groups'].values() for r in group]
        currentkeys={r['state_key'] for r in current};currentq={quotient(r['state']) for r in current}
        exposure=registry['direct_model_root_exposure'];self.assertFalse(exposure['unseen_release_holdout_claim'])
        for key,row in selected.items():
            self.assertEqual(exposure['rows'][key],dict(control_ancestral_root_raw=key in oldkeys,control_ancestral_root_quotient=row['quotient'] in oldq,
                candidate_additional_root_raw=key in currentkeys,candidate_additional_root_quotient=row['quotient'] in currentq))
        for key, row in selected.items():
            self.assertEqual(raw(row['state']), key); self.assertEqual(row['source_cohort'],'training_candidate')
            self.assertTrue(row['future_optimizer_reserved']); self.assertGreaterEqual(row['legal_mask'].bit_count(),2)
            self.assertEqual(mechanism(row['state'])[:2], (sum(map(sum,row['state']['pits'])),row['stratum']))
            legal = list(literal.legal_actions(norm(row['state']))); self.assertEqual(row['legal_mask'],sum(1<<a for a in legal))
            saved = rules['rows'][key]; refs = {}
            for stage in ('h8','h10'):
                with closing(dbopen(ROOT/'data'/row['source']/'work.sqlite3')) as db:
                    text,checksum = db.execute('SELECT body,hash FROM '+stage+'_labels WHERE state=?',(key,)).fetchone()
                refs[stage] = json.loads(text); self.assertEqual(checksum,config.digest(refs[stage])); self.assertEqual(saved[stage],refs[stage])
                self.assertEqual(saved[stage+'_item_sha256'],checksum)
                self.assertEqual(refs[stage]['root_sha256'],row['source_row']['source_root_sha256'])
            dv,dq,w = exact_weight(refs['h8'],refs['h10'],legal)
            self.assertEqual(saved['rule'],dict(root_shift_seeds=dv,mean_action_shift_seeds=rational(dq),weight=rational(w),formula=weighting.CONTRACT['rule'],requested_inputs=[8,10],H12_input=False))
            self.assertEqual(saved['source_wrapper_sha256'],config.digest(row['source_row']))
        for group in ('development','assessment'):
            subset = [r for r in selected.values() if r['group']==group]
            expected = dict(origin_families=sorted({f for r in subset for f in inv['components'][r['component']]['families']}),
                eligible_quotients=sorted({q for r in subset for q in inv['components'][r['component']]['quotients']}),
                selected_state_keys=sorted(r['state_key'] for r in subset))
            self.assertEqual(reservations['roles'][group], expected)
        self.assertFalse(set(reservations['roles']['development']['eligible_quotients']) & set(reservations['roles']['assessment']['eligible_quotients']))
        self.assertTrue(reservations['future_optimizer_reserved']); self.assertFalse(reservations['historical_roles_rewritten'])
        # SQLite's authorizer rejects reading any H12/H14 target body while the
        # complete metadata/rule projection rederives. Key-only queries remain legal.
        original_ro = weighting.ro
        def guarded_ro(name):
            db = original_ro(name)
            def guard(action, table, column, database, trigger):
                return sqlite3.SQLITE_DENY if action==sqlite3.SQLITE_READ and column=='body' and any(s in table for s in ('h12','h14')) else sqlite3.SQLITE_OK
            db.set_authorizer(guard); return db
        with mock.patch.object(weighting,'ro',side_effect=guarded_ro):
            prepared,_ = weighting.prepare(); self.assertEqual(prepared,registry)
        plan, roots, units, identity = trace.admit(); weighting.admit()
        self.assertEqual((len(units),sum(u['simulations'] for u in units)),(8,3840))
        self.assertEqual(roots,[next(r for r in load(ROOT/'build/divergence-panel.json')['roots'] if r['pair']==7)])
        self.assertEqual((identity['c_puct'],identity['time_ms'],identity['root_noise'],identity['cache_capacity'],identity['max_batch'],identity['max_nodes']),(.25,0,None,0,1,2048))
        self.assertEqual(weighting.CONTRACT['h12'],dict(depth=12,node_budget=10000000))
        self.assertEqual(weighting.CONTRACT['maximum_new_nodes'],160000000)
        with mock.patch.object(trace,'primary_witness',side_effect=ValueError('Independent production trace not ready')):
            with self.assertRaises(ValueError):weighting.run()
        self.assertFalse(weighting.DEFAULT.exists()); self.assertFalse(trace.DEFAULT.exists()); self.assertNotIn('torch',sys.modules)
        EVIDENCE['identity_and_blinding'] = dict(prior4255_inputs_current=True,full_pool_families=1280,
            eligible_roots=62,eligible_origins=53,selected16_distinct_components=True,development8_assessment8=True,
            H12_H14_target_body_SQL_reads_forbidden_during_prepare=True,exact_Fraction_rule_verified=True,
            future_component_reservations_verified=True,models_and_policy_fixed=True,no_training_or_promotion=True)

    def test_02_trace_real_pre_post_simulation_kills_and_semantic_resume(self):
        from congklak import allocation_trace as trace
        fresh = self.base/'trace-fresh'; run_child(self.command('trace',fresh),self.base/'trace-fresh.log',timeout=300)
        roots, reference, manifest = trace.load_trace(fresh,fixture=True)
        self.assertEqual((len(reference['cells']),manifest['counts']['committed_root_simulations']),(8,96)); tests=[]
        for event,count in (('allocation_trace_simulation_precommit',1),('allocation_trace_simulation_committed',2)):
            run = self.base/('trace-'+str(count)); probe = self.base/('trace-probe-'+str(count))
            marker = kill(self.command('trace',run),probe,event,count,dict(ALLOCATION_TRACE_BLOCK_EVENT=event,
                ALLOCATION_TRACE_BLOCK_AFTER=str(count),ALLOCATION_TRACE_BLOCK_MARKER=str(probe/'declared-marker.json')))
            backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
            prefix = {table:sql_rows(probe/'validated-prefix.sqlite3',table,key) for table,key in
                      (('roots','state'),('startup','id'),('root_setups','cell'),('steps','cell,seq'),('cells','idx'))}
            self.assertEqual(len(prefix['steps']),count); self.assertEqual(len(prefix['root_setups']),1); self.assertEqual(len(prefix['cells']),0)
            run_child(self.command('trace',run),probe/'resume.log',timeout=300); rr,actual,_ = trace.load_trace(run,fixture=True)
            self.assertEqual(rr,roots)
            for table,key in (('roots','state'),('startup','id'),('root_setups','cell'),('steps','cell,seq'),('cells','idx')):
                self.assertEqual(prefix[table],sql_rows(run/'work.sqlite3',table,key)[:len(prefix[table])])
            self.assertEqual(trace.semantic_trace(actual),trace.semantic_trace(reference))
            tests.append(dict(event=event,actual_process_kill=True,committed_prefix=count,marker=marker,probe=relative(probe),
                run=relative(run),resumed_simulations=96,committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        replay = self.worker(dict(kind='literal_trace',run=str(fresh),fixture=True,input_names=[relative(fresh/'work.sqlite3'),relative(fresh/'manifest.json'),relative(trace.PLAN),relative(trace.IMPLEMENTATION),relative(FREEZE),'tests/allocation_independent_audit.py']),self.base/'trace-fixture-literal')
        capped = trace.run(self.base/'trace-capped',fixture=True,variant='node_cap',on_event=lambda e:None)
        self.assertFalse(capped['scope_finished']); self.assertGreater(capped['counts']['incomplete_cells'],0)
        self.assertTrue(all(c['status']=='node_budget' and c['semantic_report']['completed_simulations']<c['unit']['simulations'] for c in capped['cells']))
        bad = self.clone(fresh,'trace-corrupt-marker'); (bad/'result.json').unlink()
        patch_sql(bad/'work.sqlite3','meta','allocation_trace_complete','key',lambda b:b.update(result_sha256='0'*64))
        with self.assertRaises(ValueError):trace.run(bad,fixture=True)
        self.assertFalse((bad/'result.json').exists())
        badid = self.clone(fresh,'trace-corrupt-identity'); patch_sql(badid/'work.sqlite3','meta','identity','key',lambda b:b.update(rng=False))
        with self.assertRaises(ValueError):trace.verify(badid,True)
        cancelled = self.base/'trace-root-cancelled'; result=trace.run(cancelled,fixture=True,cancel=lambda:True,on_event=lambda e:None)
        self.assertEqual(result['status'],'interrupted'); self.assertEqual(result['root_setups'],0)
        trace.run(cancelled,fixture=True,on_event=lambda e:None); _,cs,_=trace.load_trace(cancelled,True)
        self.assertEqual([c['semantic_report'] for c in cs['cells'].values()],[c['semantic_report'] for c in reference['cells'].values()])
        closed = self.closed_worker('trace',fresh,self.base/'trace-fixture-closed',True)
        EVIDENCE['live_trace_recovery'] = dict(tests=tests,fixture_cells=8,fixture_root_simulations=96,
            literal_replay=replay,root_setup_cancellation_resume=True,caps_not_complete=True,
            rehashed_marker_identity_refused_before_missing_export_repair=True,closed=closed)

    def test_03_teacher_real_pre_post_attempt_kills_and_H12_gate(self):
        from congklak import allocation_weighting as weighting
        fresh=self.base/'teacher-fresh'; run_child(self.command('teacher',fresh),self.base/'teacher-fresh.log',timeout=240)
        rootmap,reference,manifest=weighting.load_rows(fresh); self.assertEqual(len(rootmap),2)
        self.assertEqual(manifest['identity']['settings']['h12']['depth'],4); tests=[]
        for event,count in (('allocation_weighting_h12_precommit',1),('allocation_weighting_h12_committed',2)):
            run=self.base/('teacher-'+str(count)); probe=self.base/('teacher-probe-'+str(count))
            marker=kill(self.command('teacher',run),probe,event,count,dict(ALLOCATION_WEIGHTING_BLOCK_EVENT=event,
                ALLOCATION_WEIGHTING_BLOCK_STEP=str(count),ALLOCATION_WEIGHTING_MARKER=str(probe/'declared-marker.json')))
            backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
            prefix={t:sql_rows(probe/'validated-prefix.sqlite3',t,'state') for t in ('roots',*weighting.TABLES)}
            self.assertEqual(len(prefix['h12_attempts']),count)
            self.assertEqual(len(prefix['h8_references']),2); self.assertEqual(len(prefix['h10_references']),2)
            run_child(self.command('teacher',run),probe/'resume.log',timeout=240); rr,actual,_=weighting.load_rows(run)
            self.assertEqual(rr,rootmap)
            for table,values in prefix.items():self.assertEqual(values,sql_rows(run/'work.sqlite3',table,'state')[:len(values)])
            self.assertEqual(teacher_semantic(actual),teacher_semantic(reference))
            tests.append(dict(event=event,actual_process_kill=True,committed_prefix=count,marker=marker,probe=relative(probe),run=relative(run),
                fixture_requested_depth=4,resumed_attempts=2,committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        for key,row in rootmap.items():
            for report in reference['h12_attempts'][key]['history']:
                check_pvs(self,row,weighting.history.item_for(report,row),report['choice_depth'])
        bad=self.clone(fresh,'teacher-corrupt-history'); key=next(iter(rootmap))
        patch_sql(bad/'work.sqlite3','h12_attempts',key,'state',lambda a:a['history'].pop(0))
        with self.assertRaises(ValueError):weighting.load_rows(bad)
        badmarker=self.clone(fresh,'teacher-corrupt-marker'); (badmarker/'manifest.json').unlink()
        patch_sql(badmarker/'work.sqlite3','meta','complete','key',lambda b:b['summary'].update(attempts=99))
        with self.assertRaises(ValueError):weighting.run(badmarker,fixture=True)
        self.assertFalse((badmarker/'manifest.json').exists())
        badid=self.clone(fresh,'teacher-corrupt-identity');patch_sql(badid/'work.sqlite3','meta','identity','key',lambda b:b.update(fixture=1))
        with self.assertRaises(ValueError):weighting.load_rows(badid)
        closed=self.closed_worker('teacher',fresh,self.base/'teacher-fixture-closed',True)
        self.assertFalse(weighting.DEFAULT.exists())
        EVIDENCE['live_teacher_recovery']=dict(tests=tests,old_consumed_engineering_roots=2,actual_requested_fixture_depth=4,
            current_assessment_H12_answers_generated=False,rehashed_history_identity_complete_refused=True,closed=closed)

    def test_04_all8_production_traces_literal_selection_backup_and_parity(self):
        from congklak import allocation_trace as trace,allocation_weighting as weighting
        roots,stores,manifest=trace.load_trace(); self.assertEqual(len(roots),1)
        self.assertEqual((manifest['counts']['complete_cells'],manifest['counts']['committed_root_simulations']),(8,3840))
        self.assertEqual(manifest['counts']['incomplete_cells'],0); self.assertEqual(len(stores['interruptions']),0)
        proof=self.worker(dict(kind='literal_trace',run=str(trace.DEFAULT),fixture=False,input_names=[relative(trace.DEFAULT/'work.sqlite3'),
            relative(trace.DEFAULT/'manifest.json'),relative(trace.PLAN),relative(trace.IMPLEMENTATION),relative(trace.OUTPUT),relative(trace.RESOURCES),
            relative(FREEZE),'tests/allocation_independent_audit.py']),self.base/'production-all8-literal')
        self.assertEqual((proof['cells'],proof['root_simulations']),(8,3840))
        plan,_,units,identity=trace.admit()
        for budget in (128,256,512,1024):
            pair=[c for c in stores['cells'].values() if c['unit']['simulations']==budget]
            self.assertEqual(len(pair),2); self.assertEqual(pair[0]['semantic_report']['root_priors'],pair[1]['semantic_report']['root_priors'])
        for cell in stores['cells'].values():
            report=cell['semantic_report']; self.assertTrue(cell['sealed_reference_matches'])
            self.assertEqual(cell['semantic_report_sha256'],config.digest(report))
            self.assertEqual(report,plan['references'][cell['unit']['profile']+'/'+str(cell['unit']['simulations'])])
            self.assertEqual(sum(report['visits']),cell['unit']['simulations'])
            for n,q in zip(report['visits'],report['action_mean_value_player0']):self.assertEqual(q is None,n==0)
        witness=trace.primary_witness(); self.assertEqual(witness['completed_root_simulations'],3840)
        self.assertTrue(witness['all_sealed_reports_match']); self.assertFalse(weighting.DEFAULT.exists())
        EVIDENCE['production_trace']=dict(cells=8,root_simulations=3840,all_sealed_budget_reports_equal=True,
            independent_replay=proof,root_and_descendant_allocation_verified=True,root_priors_fixed=True,
            committed_work_separate_from_lost_work=True,no_timing_strength_or_energy_claim=True)

    def test_05_assessment_H12_fullQ_histories_and_literal_PVs(self):
        from congklak import allocation_weighting as weighting,reports
        from congklak.native import Native
        rootmap,stores,manifest=weighting.load_rows(); registry=load(weighting.REGISTRY)['rows']; rules=load(weighting.RULE)['rows']
        self.assertEqual(set(rootmap),set(registry)); self.assertEqual(len(rootmap),16)
        self.assertEqual(set(stores['h12_attempts']),set(rootmap)); self.assertEqual(manifest['requested_roots'],16)
        native=Native(); histories=paths=choices=total=0; complete=0
        for key,row in rootmap.items():
            original=registry[key]['source_row']; expected=dict(original,allocation=dict(group=registry[key]['group'],component=registry[key]['component'],
                cell=registry[key]['cell'],source_cohort='training_candidate',future_optimizer_reserved=True,source_wrapper_sha256=config.digest(original)))
            self.assertEqual(row,expected)
            for stage in ('h8','h10'):self.assertEqual(stores[stage+'_references'][key],rules[key][stage])
            attempt=stores['h12_attempts'][key]; history=attempt['history']; target=stores['h12_labels'].get(key)
            self.assertEqual(attempt['study_row_sha256'],config.digest(row)); self.assertEqual(attempt['source_root_sha256'],row['source_root_sha256'])
            self.assertEqual((attempt['stage'],attempt['requested_depth']),('h12',12))
            self.assertEqual([r['choice_depth'] for r in history],list(range(1,attempt['completed_depth']+1)))
            self.assertGreater(attempt['work']['nodes'],0); self.assertLessEqual(attempt['work']['nodes'],10000000); total+=attempt['work']['nodes']
            if attempt['status']=='complete':
                complete+=1; self.assertEqual(len(history),12); self.assertEqual(attempt['stop_reason'],'max_depth')
                self.assertEqual(target,weighting.history.item_for(history[-1],row))
            else:
                self.assertEqual(attempt['status'],'budget_rejected'); self.assertIsNone(target); self.assertIsNone(attempt['label_sha256'])
                self.assertLess(len(history),12); self.assertEqual(attempt['stop_reason'],'node_budget'); self.assertEqual(attempt['work']['nodes'],10000000)
            last={k:0 for k in attempt['work']}
            for report in history:
                reports.validate(report,row['state'],config.identity(native.build_id),native)
                self.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True))
                for field,value in last.items():self.assertGreaterEqual(report[field],value); self.assertLessEqual(report[field],attempt['work'][field])
                last={k:report[k] for k in last}; item=weighting.history.item_for(report,row)
                a,b=check_pvs(self,row,item,report['choice_depth']);paths+=a;choices+=b;histories+=1
                if report['choice_depth'] in (8,10):
                    ref=rules[key]['h'+str(report['choice_depth'])]['label']
                    for field in ('target_player0','best_moves','action_scores','choice_depth'):self.assertEqual(item['label'][field],ref[field])
        self.assertEqual((manifest['summary']['attempts'],manifest['summary']['complete'],manifest['summary']['rejected']),(16,complete,16-complete))
        self.assertEqual(manifest['total_new_nodes'],total);self.assertLessEqual(total,160000000)
        self.assertEqual(manifest['auxiliary_completed_reports'],histories)
        self.assertFalse(manifest['weights_applied']);self.assertFalse(manifest['training_performed']);self.assertFalse(manifest['promotion'])
        EVIDENCE['production_H12']=dict(attempts=16,complete=complete,capped=16-complete,all16_unconditional=True,actual_nodes=total,node_ceiling=160000000,
            completed_iteration_reports=histories,literal_PV_paths=paths,literal_PV_choices=choices,full_Q_and_bonus_actor_verified=True,
            original_root_vs_study_wrapper_links_verified=True,auxiliary_not_requested_labels=True,finite_reference_not_game_truth=True)

    def test_06_H8_H10_rule_weights_H12_denominators_and_sensitivity(self):
        from congklak import allocation_weighting as weighting
        registry=load(weighting.REGISTRY)['rows'];rule=load(weighting.RULE)['rows'];rootmap,stores,manifest=weighting.load_rows()
        output=load(weighting.OUTPUT); values=[]
        for key,row in registry.items():
            a,b=rule[key]['h8']['label'],rule[key]['h10']['label']; target=stores['h12_labels'].get(key)
            legal=[p for p in range(7) if row['legal_mask']&(1<<p)];dv,dq,w=exact_weight(rule[key]['h8'],rule[key]['h10'],legal)
            self.assertEqual(rule[key]['rule']['weight'],rational(w)); attempt=stores['h12_attempts'][key]
            expected=dict(state_key=key,family=row['family'],component=row['component'],stratum=row['stratum'],actor=row['actor'],cell=row['cell'],
                group=row['group'],source_cohort='training_candidate',future_optimizer_reserved=True,state=row['state'],legal_actions=legal,weight=rational(w),
                H12_complete=target is not None,H12_status=attempt['status'],requested_depth=12,completed_depth=attempt['completed_depth'],
                source_wrapper_sha256=row['source_wrapper_sha256'],source_root_sha256=rule[key]['source_root_sha256'],allocation_wrapper_sha256=config.digest(rootmap[key]),
                H8_item_sha256=config.digest(rule[key]['h8']),H10_item_sha256=config.digest(rule[key]['h10']),H12_item_sha256=config.digest(target) if target else None,
                root_drift=None,action_Q_drift=None,H10_best_H12_regret_min=None,H10_best_H12_regret_max=None,H10_best_H12_regret_mean=None,H10_best_H12_regrets=None,
                H8_player0=a['target_player0'],H10_player0=b['target_player0'],H12_player0=None,target_kind='approximate_requested_finite_H_v1_choice_horizon')
            if target is not None:
                c=target['label'];q10={r['pit']:r['value_player0'] for r in b['action_scores']};q12={r['pit']:r['value_player0'] for r in c['action_scores']}
                best=(max if row['actor']==0 else min)(q12.values())
                regrets={str(p):best-q12[p] if row['actor']==0 else q12[p]-best for p in b['best_moves']}
                expected.update(root_drift=rational(abs(c['target_player0']-b['target_player0'])),action_Q_drift=rational(F(sum(abs(q12[p]-q10[p]) for p in legal),len(legal))),
                    H10_best_H12_regret_min=rational(min(regrets.values())),H10_best_H12_regret_max=rational(max(regrets.values())),
                    H10_best_H12_regret_mean=rational(F(sum(regrets.values()),len(regrets))),H10_best_H12_regrets=regrets,H12_player0=c['target_player0'],
                    H10_Q_player0={str(p):q10[p] for p in q10},H12_Q_player0={str(p):q12[p] for p in q12},H10_best_moves=b['best_moves'],H12_best_moves=c['best_moves'])
            self.assertEqual(output['per_root'][key],expected);values.append(expected)
        metrics={g:independent_metric([r for r in values if r['group']==g]) for g in ('development','assessment')};metrics['all']=independent_metric(values)
        metrics['by_group_stratum']={g:{s:independent_metric([r for r in values if r['group']==g and r['stratum']==s]) for s in STRATA} for g in ('development','assessment')}
        metrics['by_group_actor']={g:{str(a):independent_metric([r for r in values if r['group']==g and r['actor']==a]) for a in (0,1)} for g in ('development','assessment')}
        metrics['by_group_cell']={g:{c:independent_metric([r for r in values if r['group']==g and r['cell']==c]) for c in CELLS} for g in ('development','assessment')}
        metrics['mass_distribution']={g:independent_mass([r for r in values if g=='all' or r['group']==g]) for g in ('development','assessment','all')}
        self.assertEqual(output['metrics'],metrics)
        m=metrics['assessment'];n=m['complete_roots'];a,b=m['uniform'],m['weighted']
        conditions=dict(all_assessment_complete=n==m['requested_roots']==8,
            weighted_root_strictly_lower=bool(n and from_rational(b['root_drift'])<from_rational(a['root_drift'])),
            weighted_action_no_higher=bool(n and from_rational(b['action_Q_drift'])<=from_rational(a['action_Q_drift'])),
            ESS_at_least_4N_over5=bool(n and from_rational(m['ESS'])>=F(4*n,5)))
        self.assertEqual(output['screen'],dict(**conditions,passes=all(conditions.values()),exact_rational_comparisons=True,
            engineering_acceptance_condition=False,calibrated_uncertainty_claim=False,training_application_authorized=False))
        fields=('root_drift','action_Q_drift','H10_best_H12_regret_min','H10_best_H12_regret_max','H10_best_H12_regret_mean')
        for uniform in (True,False):
            synthetic=[dict(state_key=str(i),H12_complete=True,weight=rational(1 if uniform else F(1,4) if i%2 else 1),**{k:rational(i if uniform else 5) for k in fields}) for i in range(8)]
            actual=weighting.metric(synthetic);self.assertEqual(actual,independent_metric(synthetic));self.assertFalse(weighting.descriptive_screen(actual)['passes'])
        synthetic[0]['H12_complete']=False
        for field in fields:synthetic[0][field]=None
        actual=weighting.metric(synthetic);self.assertEqual(actual,independent_metric(synthetic));self.assertFalse(weighting.descriptive_screen(actual)['all_assessment_complete'])
        self.assertFalse(output['weights_applied']);self.assertFalse(output['rule_fitted']);self.assertFalse(output['promotion']);self.assertEqual(output['previous_allocation'],'hold')
        EVIDENCE['weighting_assessment']=dict(selected_roots=16,development8_assessment8=True,all_rational_weights_and_metrics_verified=True,
            complete_intersection_and_missing_mass_verified=True,full_H10_tie_set_regret_verified=True,constant_uniform_missing_probes=True,
            assessment_screen=output['screen'],no_fitting_application_calibration_or_strength_claim=True,future_admission_must_ingest_reservations=True)

    def test_07_rehashed_corruption_missing_publication_and_lookahead_refusal(self):
        from congklak import allocation_trace as trace,allocation_weighting as weighting
        roots,stores,manifest=trace.load_trace();_,_,units,identity=trace.admit();cases=[]
        setup=stores['root_setups'][0];first=stores['steps'][0][0]
        for name,mutate in (('root-score',lambda b:b['root_selection']['rows'][0].update(total_score_normalized=17.0)),
                            ('leaf-value',lambda b:b['leaf'].update(raw_value_player0=b['leaf']['raw_value_player0']+1)),
                            ('actor-bool',lambda b:b['path'][0].__setitem__(1,True))):
            tree=trace._root_setup(setup,units[0],roots[0],identity,stores['startup'],trace.Native());bad=copy.deepcopy(first);mutate(bad)
            save(self.base/('trace-rehashed-'+name+'.json'),dict(body=bad,sha256=config.digest(bad)))
            with self.assertRaises((ValueError,AssertionError)):trace.replay_step(tree,bad,units[0],identity)
            cases.append(name)
        fresh=self.base/'repair-trace-fixture'; run_child(self.command('trace',fresh),self.base/'repair-trace-fixture.log',timeout=300)
        missing=self.clone(fresh,'trace-valid-missing-export');(missing/'result.json').unlink()
        with (mock.patch.object(trace,'CpuBatchEvaluator',side_effect=AssertionError('Missing-export repair inferred')),
              mock.patch.object(trace,'_advance',side_effect=AssertionError('Missing-export repair searched'))):
            repaired=trace.run(missing,fixture=True)
        self.assertEqual(load(missing/'result.json'),repaired)
        teacher=self.base/'repair-teacher-fixture'; run_child(self.command('teacher',teacher),self.base/'repair-teacher-fixture.log',timeout=240)
        bad=self.clone(teacher,'teacher-rehashed-pvs');key=next(iter(rows(bad/'work.sqlite3','roots','state')))
        patch_sql(bad/'work.sqlite3','h12_attempts',key,'state',lambda a:a['history'][0]['search_options'].update(use_pvs=False))
        with self.assertRaises(ValueError):weighting.load_rows(bad)
        missing=self.clone(teacher,'teacher-valid-missing-export');(missing/'manifest.json').unlink()
        with mock.patch.object(weighting.history,'execute',side_effect=AssertionError('Missing-export repair searched')):
            repaired=weighting.run(missing,fixture=True)
        self.assertEqual(load(missing/'manifest.json'),repaired)
        original_read=weighting.read
        for name in ('metric','source'):
            bad=copy.deepcopy(load(weighting.OUTPUT))
            if name=='metric':bad['metrics']['assessment']['uniform']['root_drift']['numerator']+=1
            else:bad['inputs'][relative(weighting.RULE)]='0'*64
            save(self.base/('weighting-rehashed-'+name+'.json'),dict(body=bad,sha256=config.digest(bad)))
            with mock.patch.object(weighting,'read',side_effect=lambda p,b=bad:b if Path(p).resolve()==weighting.OUTPUT.resolve() else original_read(p)):
                with self.assertRaises(ValueError):weighting.verify()
            cases.append('assessment-'+name)
        bad=copy.deepcopy(load(weighting.RULE));bad['H12_used']=True
        bad['rows'][next(iter(bad['rows']))]['rule']['requested_inputs']=[8,10,12]
        save(self.base/'weighting-rehashed-lookahead.json',dict(body=bad,sha256=config.digest(bad)))
        with mock.patch.object(weighting,'read',side_effect=lambda p:bad if Path(p).resolve()==weighting.RULE.resolve() else original_read(p)):
            with self.assertRaises(ValueError):weighting.admit()
        EVIDENCE['production_corruption_refusal']=dict(cases=cases,all_payloads_rehashed=True,all_refused=True,
            teacher_pvs_refused=True,H12_rule_lookahead_refused=True,validated_missing_exports_repaired_without_search=True,
            no_production_artifacts_modified=True)

    def test_08_closed_API_CLI_no_work_and_prior_preservation(self):
        from congklak import allocation_trace as trace,allocation_weighting as weighting
        a=self.closed_worker('trace',trace.DEFAULT,self.base/'production-trace-closed')
        b=self.closed_worker('teacher',weighting.DEFAULT,self.base/'production-weighting-closed',watch_prior=True)
        oldtree=ast.parse((ROOT/'congklak/divergence_progress.py').read_text());newtree=ast.parse((ROOT/'congklak/allocation_progress.py').read_text())
        for name in ('event_message','run_command','save','runs'):
            old=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            new=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        logger=load(ROOT/'build/ordinary-value-review-progress-author.json');self.assertTrue(logger['passed']);self.assertEqual(logger['tests_run'],2);self.current(logger['inputs'])
        self.current(load(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        EVIDENCE['final_no_work']=dict(trace=a,weighting=b,all4255_old_pins_current=True,models_policy_and_HOLD_fixed=True,
            proven_logger_four_functions_AST_identical=True,old_logger_tests_not_recounted=True,
            future_reservation_ledger_requires_new_trainer_admission=True,no_training_selfplay_strength_games_or_promotion=True)


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def source_inputs():
    from congklak import allocation_trace as trace,allocation_weighting as weighting
    result=dict(load(BASELINE)['inputs']);names={relative(Path(__file__)),'tools/allocation-audit.py',relative(BASELINE),
        'docs/ALLOCATION_SCOPE.md','congklak/allocation_common.py','congklak/allocation_progress.py','tools/allocation-dashboard.html',
        relative(trace.PLAN),relative(trace.IMPLEMENTATION),relative(weighting.INVENTORY),relative(weighting.REGISTRY),
        relative(weighting.RULE),relative(weighting.RESERVATIONS),relative(weighting.PLAN),relative(weighting.IMPLEMENTATION),
        *weighting.SOURCES,*weighting.DEPENDENCIES}
    for field in ('sources','science','dependencies'):names.update(load(trace.IMPLEMENTATION)[field])
    result.update({name:sha(ROOT/name) for name in names});return dict(sorted(result.items()))


def collect():
    inputs={};evidence={};covered=[];groups=[]
    for group,methods in GROUPS.items():
        path=WORK/(group+'.json');value=load(path)
        assert value['passed'] is True and value['covered_methods']==methods and value['tests_run']==len(methods)
        assert not any(value[k] for k in ('errors','failures','skips'))
        for name,checksum in value['inputs'].items():
            assert sha(ROOT/name)==checksum,name;assert name not in inputs or inputs[name]==checksum;inputs[name]=checksum
        assert not(set(evidence)&set(value['evidence']));evidence.update(value['evidence']);covered+=methods
        groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods));inputs[relative(path)]=sha(path)
    assert len(covered)==len(set(covered))==8
    body=dict(schema='allocation-independent-audit-v1',passed=True,tests_run=8,errors=0,failures=0,skips=0,
        covered_methods=covered,covered_checks=['__main__.AllocationIndependentAudit.'+name for name in covered],
        groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/allocation-audit.json'
    if output.exists():assert load(output)==body,'Immutable audit aggregate differs'
    else:save(output,body)
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['freeze',*GROUPS,'collect']);args=parser.parse_args()
    if args.operation=='freeze':
        assert not FREEZE.exists(),'Audit source already frozen'
        save(FREEZE,dict(schema='allocation-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=load(FREEZE);assert frozen['groups']==GROUPS
    for name,checksum in frozen['inputs'].items():assert sha(ROOT/name)==checksum,'Executed source changed: '+name
    output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable audit group exists'
    started=time.monotonic();methods=GROUPS[args.operation]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(AllocationIndependentAudit(name) for name in methods))
    from congklak import allocation_trace as trace,allocation_weighting as weighting
    inputs={**frozen['inputs'],relative(FREEZE):sha(FREEZE)}
    files=[p for p in AllocationIndependentAudit.base.rglob('*') if durable(p)]
    if args.operation in ('trace','labels','final'):
        files += [trace.OUTPUT,trace.RESOURCES]+[p for p in trace.DEFAULT.rglob('*') if durable(p)]+[WORK/'preflight.json']
    if args.operation in ('labels','final'):
        files += [weighting.OUTPUT]+[p for p in weighting.DEFAULT.rglob('*') if durable(p)]+[WORK/'trace.json']
    if args.operation=='final':files += [WORK/'labels.json']
    inputs.update(pins(files))
    body=dict(schema='allocation-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        covered_checks=['__main__.AllocationIndependentAudit.'+name for name in methods],duration_seconds=time.monotonic()-started,
        retained_directory=relative(AllocationIndependentAudit.base),inputs=inputs,evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=traceback) for test,traceback in result.errors+result.failures])
    save(output,body);print(config.canonical(dict(event='allocation_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if body['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

