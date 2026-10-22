"""Independent ownership, natural-history, finite-target and live-recovery audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from congklak import config
from congklak.native import Native, flatten
from data_pilot_independent_audit import (sha, load, save, relative, pins, dbopen,
    norm, position, key_of, quotient, mechanism, rows, semantic, kill_barrier,
    run_child, snapshot, from_key)
import reference as literal

WORK = ROOT / 'build/midgame-pilot-audit'
BASELINE = ROOT / 'build/midgame-pilot-baseline.json'
CONTROL = ROOT / 'data/quality-train-v1/model.json'
PILOT = ROOT / 'data/value-pilot-v1/model.json'
CONTROL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
PILOT_SHA = '7d19e6685000406349cf1a7aa862f4a13665ff59de41145a8f29de10eb8c0f1d'
FROZEN = ('layers.0.weight', 'layers.0.bias', 'layers.2.weight', 'layers.2.bias',
          'policy.weight', 'policy.bias')
EVIDENCE = {}


def flat_bytes(value):
    return b''.join(flat_bytes(v) for v in value) if isinstance(value, list) else struct.pack('<f', value)


def latest_games_reference():
    """Replay every current pilot arena state including the final terminal state."""
    protected, families, projection = set(), set(), []
    observations = choices = 0
    for stage, expected in (('pretraining', 48), ('posttraining', 96)):
        path = ROOT / f'data/value-pilot-benchmark-{stage}-v1/work.sqlite3'
        games = rows(path, 'games', 'idx')
        assert len(games) == expected
        for index, game in games.items():
            q = norm(game['start']); family = game['family']; families.add(family)
            assert len(game['moves']) == len(game['decisions'])
            for step, (action, saved) in enumerate(zip(game['moves'], game['decisions'])):
                assert not literal.normalize(q).terminal
                assert action in literal.legal_actions(q) and saved['actor'] == q.controller
                protected.add(quotient(position(q))); observations += 1
                q = literal.apply(q, action).position; choices += 1
            assert literal.normalize(q).terminal and position(q) == game['final']
            protected.add(quotient(position(q))); observations += 1
            projection.append([stage, index, family, game['start'], game['moves']])
    return protected, families, dict(games=144, literal_choices=choices,
        state_observations_including_terminal=observations,
        unique_quotients=len(protected), families=len(families),
        projection_sha256=config.digest(projection))


def old_role_reference():
    roots = rows(ROOT / 'data/ordinary-data-pilot-v1/work.sqlite3', 'roots', 'state')
    natural = rows(ROOT / 'data/natural-study-v1/work.sqlite3', 'roots', 'state')
    for key, wrapper in natural.items():
        if key in roots:
            assert roots[key] == wrapper['root']
        roots[key] = wrapper['root']
    assert len(roots) == 83
    assert Counter(r['cohort'] for r in roots.values()) == dict(training_candidate=61, validation=14, diagnostic=8)
    return roots


def diagnostic_reference():
    report = load(ROOT / 'build/pilot-loss-review-arena.json')
    selected = [r for r in report['seats'] if not r['first_divergence']['identical_action_trajectory']]
    assert len(selected) == 9
    assert sum(r['first_divergence']['existing_Q_comparison']['coverage'] == 'uncovered' for r in selected) == 8
    return selected


def protection_reference():
    protected, families = set(), set()
    for name in ('value-registry', 'data-pilot-registry', 'natural-study-registry'):
        old = load(ROOT / f'build/{name}.json')
        protected.update(old['quotients']); families.update(old['families'])
    root_metadata = []
    with closing(dbopen(ROOT / 'data/deeper-9000-v2/work.sqlite3')) as db:
        for key, state, family in db.execute("SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family') FROM roots ORDER BY state"):
            p = json.loads(state); protected.add(quotient(p)); families.add(family)
            root_metadata.append([key, p, family])
    assert len(root_metadata) == 9000
    groups = load(ROOT / 'build/value-pilot-rows.json')['groups']
    assert len(groups['development']) == 1000
    assert sum(r['stratum'].endswith('/ordinary') for r in groups['development']) == 259
    for row in groups['development']:
        protected.add(quotient(row['state'])); families.add(row['family'])
    old_roles = old_role_reference()
    for row in old_roles.values():
        protected.add(quotient(row['state'])); families.add(row['family'])
    generated = set()
    for name, expected in (('ordinary-data-pilot-v1', 256), ('natural-study-v1', 512)):
        pool = rows(ROOT / f'data/{name}/work.sqlite3', 'families', 'idx')
        assert len(pool) == expected
        generated.update(r['family'] for r in pool.values())
    families.update(generated)
    plan = load(ROOT / 'build/value-pilot-benchmark-plan.json')
    for row in [*plan['cases'], *plan['arena_starts']]:
        protected.add(quotient(row['state'])); families.add(row['family'])
    latest_q, latest_f, latest = latest_games_reference()
    protected.update(latest_q); families.update(latest_f)
    conflicts = {key for key, row in old_roles.items()
                 if quotient(row['state']) in latest_q or row['family'] in latest_f}
    return protected, families, old_roles, dict(latest=latest,
        root_metadata_sha256=config.digest(root_metadata), old_generated_families=len(generated),
        curated_roots=83, ordinary_development_roots=259,
        curated_roles=dict(Counter(r['cohort'] for r in old_roles.values())),
        latest_arena_conflicting_old_role_keys=sorted(conflicts),
        latest_arena_conflicting_old_roles=dict(Counter(old_roles[k]['cohort'] for k in conflicts)))


def transitive_components(candidates, protected_q, protected_f):
    """Independent bipartite fixed point; no quota, mechanism or opening shortcuts."""
    fq, qf, owners = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in candidates:
        f, q = row['family'], row['quotient']
        fq[f].add(q); qf[q].add(f); owners[f].add(row['cohort'])
    remaining, components = set(fq), []
    blocked_f, blocked_q = set(), set()
    while remaining:
        first = min(remaining); fs, qs, frontier = {first}, set(), {first}
        while frontier:
            newq = set().union(*(fq[f] for f in frontier)) - qs
            qs.update(newq)
            newf = (set().union(*(qf[q] for q in newq)) if newq else set()) - fs
            fs.update(newf); frontier = newf
        remaining.difference_update(fs)
        cohort = set().union(*(owners[f] for f in fs))
        reasons = []
        if qs & protected_q: reasons.append('protected_quotient')
        if fs & protected_f: reasons.append('protected_family')
        if len(cohort) > 1: reasons.append('cross_cohort_component')
        if reasons: blocked_f.update(fs); blocked_q.update(qs)
        components.append(dict(families=sorted(fs), quotients=sorted(qs),
                               cohorts=sorted(cohort), reasons=reasons))
    return blocked_f, blocked_q, components


def check_full_label(test, row, item, depth):
    """Reconstruct all legal Q, ties, root linkage and every literal PV leaf."""
    state = row['state']; report = item['report']; label = item['label']
    legal = list(literal.legal_actions(norm(state)))
    values = {r['pit']: r['value_player0'] for r in label['action_scores']}
    sign = 1 if state['controller'] == 0 else -1
    test.assertEqual(list(values), legal)
    optimum = max(sign * v for v in values.values())
    test.assertEqual(sign * label['target_player0'], optimum)
    test.assertEqual(label['best_moves'], [a for a in legal if sign * values[a] == optimum])
    test.assertEqual(report['choice_depth'], depth); test.assertEqual(label['choice_depth'], depth)
    test.assertEqual(label['target_player0'], report['depth_result']['value_player0'])
    test.assertEqual(label['best_moves'], report['best_moves'])
    test.assertEqual(label['coverage'], report['coverage'])
    test.assertEqual(label['action_scores'], [dict(pit=c['pit'], value_player0=c['depth_result']['value_player0']) for c in report['candidates']])
    test.assertEqual(label['work'], {k: report[k] for k in ('nodes', 'deposits', 'elapsed_ms', 'peak_process_bytes')})
    paths = choices = 0
    for pv, value in [(report['pv'], label['target_player0'])] + [(c['pv'], c['depth_result']['value_player0']) for c in report['candidates']]:
        q = norm(state)
        for move in pv:
            test.assertEqual(move['actor'], q.controller)
            test.assertIn(move['pit'], literal.legal_actions(q))
            transition = literal.apply(q, move['pit']); q = transition.position
            test.assertEqual(move['terminal_after'], transition.terminal)
            test.assertEqual(move['bonus_choice'], transition.bonus and not transition.terminal)
            test.assertEqual(move['forced_passes_after'], [dict(zip(('from', 'to'), v)) for v in transition.final_passes])
            choices += 1
        test.assertTrue(literal.normalize(q).terminal or len(pv) == depth)
        test.assertEqual(literal.evaluate(q), value); paths += 1
    return paths, choices


def literal_history(index, settings, evaluator):
    family = f"midgame-history-pilot/{settings['seed']}/{index}"
    rank = int.from_bytes(hashlib.sha256(('midgame-pilot-owner-v1/' + family).encode()).digest()[:8], 'big') % 100
    cohort = 'training_candidate' if rank < 80 else 'validation'
    source = 'random_history' if index % 2 == 0 else 'frozen_policy_epsilon_history'
    rng = random.Random(settings['seed'] + 104729 * index)
    q = norm(config.position()); moves = []; candidates = []; forwards = 0
    eligible = set(settings['caps']['training_candidate'])
    for step in range(settings['max_choices']):
        state = position(q); active = sum(map(sum, state['pits']))
        if literal.normalize(q).terminal or active < 25: break
        key = key_of(state); canonical = from_key(key)
        active, stratum, terminal_children = mechanism(canonical)
        legal_mask = sum(1 << a for a in literal.legal_actions(norm(canonical)))
        if stratum in eligible and legal_mask.bit_count() >= 2:
            swapped = bytes(q.counts + (q.controller,)).hex() != key
            candidates.append(dict(state_key=key, state=canonical, quotient=quotient(canonical),
                family=family, cohort=cohort, purpose='targeted_midgame_pilot',
                band=stratum.split('/')[0], stratum=stratum, active=active,
                legal_mask=legal_mask, terminal_children=terminal_children,
                provenance=dict(source=source, family_index=index, step=step,
                    start=config.position(), moves_prefix=list(moves), source_game_idx=None,
                    source_comparison=None, saved_puct_chosen=None, raw_state=state, swapped=swapped)))
        legal = list(literal.legal_actions(q))
        if index % 2 == 0 or step < settings['random_opening_choices'] or rng.random() < settings['epsilon']:
            action = rng.choice(legal)
        else:
            logits = evaluator.evaluate_many([state])[0]['policy_logits']; forwards += 1
            action = min(legal, key=lambda a: (-logits[a], a))
        moves.append(action); q = literal.apply(q, action).position
    final = position(q); active = sum(map(sum, final['pits']))
    return dict(index=index, family=family, cohort=cohort, source=source,
        rng_seed=settings['seed'] + 104729 * index, start=config.position(), moves=moves,
        final=final, choices=len(moves), candidates=candidates, policy_evaluations=forwards,
        stop_reason='terminal' if not active else 'below25' if active < 25 else 'choice_cap')


def selected_reference(pool, settings, registry):
    candidates = [r for f in pool.values() for r in f['candidates']]
    blocked_f, blocked_q, components = transitive_components(
        candidates, set(registry['quotients']), set(registry['families']))
    available = sorted((r for r in candidates if r['family'] not in blocked_f),
        key=lambda r: (r['provenance']['family_index'], r['provenance']['step'], r['state_key']))
    chosen, seen_q, seen_family_stratum, counts = {}, set(), set(), Counter()
    for row in available:
        key = row['cohort'], row['stratum']; fs = row['family'], row['stratum']
        if counts[key] < settings['caps'][key[0]][key[1]] and row['quotient'] not in seen_q and fs not in seen_family_stratum:
            chosen[row['state_key']] = row; seen_q.add(row['quotient'])
            seen_family_stratum.add(fs); counts[key] += 1
    evidence = dict(eligible_observations=len(candidates), eligible_unique_quotients=len({r['quotient'] for r in candidates}),
        graph_components=len(components), blocked_families=len(blocked_f), blocked_quotients=len(blocked_q),
        counts={c:{s:counts[c,s] for s in caps} for c,caps in settings['caps'].items()},
        shortfall={c:{s:n-counts[c,s] for s,n in caps.items()} for c,caps in settings['caps'].items()})
    return dict(sorted(chosen.items())), evidence, components


def history_item(report, row):
    return dict(root_sha256=row['source_root_sha256'], report=report,
        label=dict(kind='finite_choice_horizon', choice_depth=report['choice_depth'],
            oracle_manifest_sha256=None, target_player0=report['depth_result']['value_player0'],
            best_moves=report['best_moves'], action_scores=[dict(pit=c['pit'],
                value_player0=c['depth_result']['value_player0']) for c in report['candidates']],
            coverage=report['coverage'], work={name:report[name] for name in
                ('nodes','deposits','elapsed_ms','peak_process_bytes')}))


def audit_weights(test, fresh, wrappers, stores):
    report = load(ROOT / 'build/midgame-pilot-weight-review.json')
    eligible = [key for key,row in sorted(fresh.items())
                if row['cohort']=='training_candidate' and key in stores['h10_labels']]
    by_key = {r['state_key']:r for r in report['candidate_rows']}
    test.assertEqual(list(by_key), eligible); test.assertEqual(len(by_key),len(report['candidate_rows']))
    test.assertEqual(report['eligible_complete_H10_training_candidates'],len(eligible))
    factors = {}
    for key in eligible:
        row = fresh[key]; saved = by_key[key]
        depths = [d for d in (8,10) if key in stores[f'h{d}_labels']]
        targets = [stores[f'h{d}_labels'][key]['label']['target_player0'] for d in depths]
        span = max(targets)-min(targets)
        overlap = bool(set(stores['h8_labels'][key]['label']['best_moves']) &
            set(stores['h10_labels'][key]['label']['best_moves'])) if len(depths)==2 else None
        volatility = 4/(4+span) if len(depths)==2 else .5
        factor = volatility*(.5 if overlap is False else 1.)
        factors[key] = factor
        test.assertEqual(saved, dict(state_key=key,family=row['family'],stratum=row['stratum'],
            cohort='training_candidate',source_root_sha256=config.digest(row),
            requested_depths_available=depths,target_player0=targets,H8_H10_best_sets_overlap=overlap,
            target_span_seeds=span,volatility_factor=volatility,disagreement_factor=.5 if overlap is False else 1.,
            proposed_exposure_factor=factor))
    test.assertEqual(report['excluded_fresh_roots'],[dict(state_key=key,
        reason='protected_validation' if row['cohort']!='training_candidate' else 'no_complete_requested_H10')
        for key,row in sorted(fresh.items()) if key not in by_key])
    strata = Counter(fresh[key]['stratum'] for key in eligible)
    expected = {
        'uniform_root':{key:1/len(eligible) for key in eligible},
        'equal_nonempty_target_stratum':{key:1/(len(strata)*strata[fresh[key]['stratum']]) for key in eligible},
        'stability_weighted_root':{key:factors[key]/math.fsum(factors.values()) for key in eligible}}
    for name, probabilities in expected.items():
        actual = report['distributions'][name]
        test.assertEqual((actual['roots'],actual['families']),
            (len(eligible),len({fresh[key]['family'] for key in eligible})))
        test.assertEqual(set(actual['probabilities']),set(probabilities))
        for key,value in probabilities.items():test.assertAlmostEqual(actual['probabilities'][key],value,places=14)
        if not eligible:
            test.assertEqual(actual['stratum_mass'],{})
            for field in ('root_effective_support','family_effective_support','maximum_root_probability'):test.assertIsNone(actual[field])
            continue
        test.assertAlmostEqual(math.fsum(actual['probabilities'].values()),1.,places=14)
        family_mass = {family:math.fsum(value for key,value in probabilities.items() if fresh[key]['family']==family)
                       for family in {fresh[key]['family'] for key in eligible}}
        stratum_mass = {s:math.fsum(value for key,value in probabilities.items() if fresh[key]['stratum']==s) for s in strata}
        for s,value in stratum_mass.items():test.assertAlmostEqual(actual['stratum_mass'][s],value,places=14)
        test.assertAlmostEqual(actual['root_effective_support'],1/math.fsum(v*v for v in probabilities.values()),places=12)
        test.assertAlmostEqual(actual['family_effective_support'],1/math.fsum(v*v for v in family_mass.values()),places=12)
        test.assertAlmostEqual(actual['maximum_root_probability'],max(probabilities.values()),places=14)
    for flag in ('sampling_changed','loss_weights_applied','training_performed','model_changed',
                 'source_data_pruned','sealed_final_test_consumed'):test.assertFalse(report[flag])
    test.assertTrue(report['requested_H10_only']);test.assertTrue(report['no_auxiliary_fallback'])
    test.assertEqual(report['protected_diagnostics'],9)
    test.assertFalse(set(by_key)&{k for k,r in wrappers.items() if r['study_role']=='diagnostic'})
    return dict(eligible_training_candidates=len(eligible),proposals=3,
        all_probabilities_normalized_and_concentration_rederived=True,
        validation_and_diagnostics_zero_eligibility=True,only_complete_requested_H10=True,
        no_auxiliary_or_old_replay_blend=True,no_active_weights_or_model_changed=True)


def audit_metrics(test, wrappers, stores):
    from midgame_pilot_metric_audit import check
    return check(test, wrappers, stores)


class MidgamePilotIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.base = WORK / ('retained-' + uuid.uuid4().hex); cls.base.mkdir()

    def fresh(self, name):
        path = self.base / name; path.mkdir(); return path

    def backup(self, run, destination):
        with closing(dbopen(run / 'work.sqlite3')) as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)

    def collection_command(self, run):
        return [sys.executable, str(ROOT / 'tools/midgame-data-pilot.py'), 'collect',
                '--run', str(run), '--fixture', '--families', '8']

    def label_command(self, run, collected, stage='all'):
        return [sys.executable, str(ROOT / 'tools/midgame-label-pilot.py'), stage,
                '--run', str(run), '--collection', str(collected), '--fixture',
                '--diagnostics', '2', '--fresh-roots', '1']

    def forbidden_search_cli(self, command, path):
        script = path.with_suffix('.py')
        script.write_text("import runpy,sys\nfrom unittest import mock\n"
            + 'sys.path.insert(0,' + repr(str(ROOT)) + ')\n'
            + 'from congklak.native import Native\n'
            + 'sys.argv=' + repr(command[1:]) + '\n'
            + "with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed CLI invoked teacher')) as blocked:\n"
            + " try: runpy.run_path(sys.argv[0],run_name='__main__')\n"
            + " except SystemExit as exc:\n  if exc.code not in (0,None): raise\n"
            + " assert blocked.call_count==0\nassert 'torch' not in sys.modules\n"
            + "print('{\"event\":\"completed_CLI_zero_teacher_calls\",\"passed\":true}',flush=True)\n", encoding='utf-8')
        run_child([sys.executable, str(script)], path)

    def teacher_fixture(self):
        cls = type(self)
        if not hasattr(cls, 'label_fixture'):
            parent = self.fresh('labels'); collected = parent / 'collection'; fresh = parent / 'fresh'
            run_child(self.collection_command(collected), parent / 'collection.log')
            run_child(self.label_command(fresh, collected), parent / 'fresh.log')
            cls.label_fixture = parent, collected, fresh
        return cls.label_fixture

    def test_01_baseline2639_fixed_models_and_frozen_contract(self):
        b = load(BASELINE)
        self.assertEqual((b['schema'], b['inputs_count'], b['prior_accepted_inputs']),
                         ('midgame-pilot-prior-immutability-v1', 2639, 2637))
        self.assertEqual(len(b['inputs']), 2639)
        for name, digest in b['inputs'].items(): self.assertEqual(sha(ROOT / name), digest, name)
        self.assertEqual(sha(CONTROL), CONTROL_SHA); self.assertEqual(sha(PILOT), PILOT_SHA)
        control, pilot = load(CONTROL)['model'], load(PILOT)['model']
        for name in FROZEN: self.assertEqual(flat_bytes(control['weights'][name]), flat_bytes(pilot['weights'][name]))
        self.assertEqual(sum(len(flat_bytes(control['weights'][name])) // 4 for name in FROZEN), 12807)
        self.assertFalse(b['new_training_authorized']); self.assertFalse(b['new_strength_games_authorized'])
        self.assertFalse(b['source_data_pruning_authorized'])
        from congklak import midgame_data_pilot as collection, midgame_label_pilot as labeling
        collection.admit(); labeling.admit()
        c = collection.CONTRACT
        self.assertEqual((c['families'], c['seed'], c['max_choices'], c['random_opening_choices'], c['epsilon']),
                         (512, 20261023, 192, 8, .25))
        self.assertEqual(c['caps'], dict(training_candidate={'50-74/ordinary':16, '25-49/ordinary':8,
            '25-49/bonus':8, '25-49/capture':8}, validation={'50-74/ordinary':4,
            '25-49/ordinary':2, '25-49/bonus':2, '25-49/capture':2}))
        self.assertEqual(c['minimum_legal_actions'], 2)
        l = labeling.CONTRACT
        for stage, depth, budget in (('h8',8,2000000), ('h10',10,5000000), ('h12',12,10000000)):
            self.assertEqual(l[stage], dict(depth=depth, node_budget=budget))
        self.assertEqual((l['diagnostic_roots'], l['maximum_fresh_roots'], l['maximum_new_production_nodes']), (9,50,503000000))
        self.assertEqual((l['time_ms'], l['tt_mb'], l['memory_mb'], l['use_pvs'], l['keep_history']), (0,16,512,True,True))
        for flag in ('automatic_expansion','H14_authorized','training_performed','sampling_applied','sealed_final_test_consumed'):
            self.assertFalse(l[flag])
        EVIDENCE['baseline_and_contract'] = dict(prior_inputs=2639, prior_accepted_inputs=2637,
            all_prior_hashes_current=True, control_and_pilot_unchanged=True,
            six_frozen_tensors_raw_bytes_identical=True, fixed_policy_parameters=12807,
            training_or_new_strength_games_authorized=False)

    def test_02_literal_latest144_ownership_83roles_259dev_and_nine_diagnostics(self):
        from congklak import midgame_data_pilot as collection
        registry = load(collection.REGISTRY)
        protected, families, old_roles, proof = protection_reference()
        self.assertEqual(set(registry['quotients']), protected)
        self.assertEqual(set(registry['families']), families)
        latest_q, latest_f, latest = latest_games_reference()
        self.assertEqual(set(registry['latest_arena_quotients']), latest_q)
        self.assertEqual(set(registry['latest_arena_families']), latest_f)
        self.assertEqual(registry['latest_arena_game_count'], 144)
        self.assertEqual(registry['historical_role_counts'], dict(Counter(r['cohort'] for r in old_roles.values())))
        self.assertEqual(set(registry['historical_roles']), set(old_roles))
        for key, row in old_roles.items():
            self.assertEqual(registry['historical_roles'][key], dict(state_key=key,
                quotient=row['quotient'], family=row['family'], cohort=row['cohort'],
                original_root_sha256=config.digest(row),
                latest_eval_quotient_collision=row['quotient'] in latest_q,
                latest_eval_family_collision=row['family'] in latest_f))
        self.assertEqual(registry['prior_family_pool_count'], 768)
        ordinary = {r['state_key'] for r in load(ROOT / 'build/value-pilot-rows.json')['groups']['development']
                    if r['stratum'].endswith('/ordinary')}
        self.assertEqual({r['state_key'] for r in registry['base_ordinary_development']}, ordinary)
        self.assertEqual(len(ordinary), 259)
        expected_diagnostics = diagnostic_reference(); actual = registry['diagnostics']
        self.assertEqual(len(actual), 9)
        for seat in expected_diagnostics:
            div = seat['first_divergence']; state = div['state']
            key = bytes(flatten(state) + [state['controller']]).hex()
            row = actual[key]; active, stratum, terminal_children = mechanism(state)
            self.assertEqual(row, dict(state_key=key, state=state, quotient=quotient(state),
                family=seat['family'], cohort='diagnostic', purpose='protected_first_divergence',
                band=stratum.split('/')[0], stratum=stratum, active=active,
                legal_mask=sum(1 << a for a in literal.legal_actions(norm(state))),
                terminal_children=terminal_children,
                provenance=dict(source='saved_first_divergence', source_report='build/pilot-loss-review-arena.json',
                    source_report_sha256=sha(ROOT / 'build/pilot-loss-review-arena.json'),
                    pair=seat['pair'], candidate_seat=seat['candidate_seat'],
                    first_different_choice=div['first_different_choice'],
                    control_chosen=div['control']['summary']['chosen'],
                    pilot_chosen=div['pilot']['summary']['chosen'],
                    existing_Q_comparison=div['existing_Q_comparison'], raw_state=state, swapped=False)))
        self.assertEqual(sorted(r['active'] for r in actual.values()), [8,11,15,30,34,39,47,53,66])
        self.assertFalse(registry['sealed_target_bodies_read']); self.assertFalse(registry['historical_roles_rewritten'])
        self.assertTrue(registry['future_protection_overrides_eligibility']); self.assertTrue(registry['ancestry_cannot_be_untrained'])
        EVIDENCE['protection_and_diagnostics'] = dict(**proof, protected_quotients=len(protected),
            protected_families=len(families), diagnostic_roots=9, formerly_uncovered=8,
            sole_old_Q_comparison_preserved=True, exact_raw_diagnostic_orientation=True,
            diagnostics_exempt_from_fresh_band_and_mechanism_filter=True,
            every_terminal_state_protected=True, historical_roles_unchanged=True,
            no_claim_of_ancestral_unlearning=True)

    def test_03_literal_fixture_sampling_and_multihop_component_closure(self):
        from congklak import midgame_data_pilot as collection
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        settings = collection.settings_for(True, 8); actual = {}; expected = {}
        native = Native()
        with CpuBatchEvaluator(CONTROL, cache_capacity=0, max_batch=1) as evaluator:
            for index in range(8):
                expected[index] = literal_history(index, settings, evaluator)
                actual[index] = collection.generate_family(index, settings, native, evaluator)
                self.assertEqual(actual[index], expected[index], f'Fixture family {index}')
        registry = load(collection.REGISTRY)
        wanted, proof, components = selected_reference(expected, settings, registry)
        got, evidence = collection.select(actual, settings, registry)
        self.assertEqual(got, wanted)
        base = next(r for f in expected.values() for r in f['candidates'])
        graph = {}
        links = [('A','p','training_candidate'), ('A','q1','training_candidate'),
                 ('B','q1','training_candidate'), ('B','q2','training_candidate'),
                 ('C','q2','training_candidate'), ('D','q3','validation'),
                 ('E','q3','training_candidate'), ('E','q4','training_candidate'),
                 ('F','q4','training_candidate'), ('G','safe','training_candidate')]
        for index, (family, q, cohort) in enumerate(links):
            row = copy.deepcopy(base)
            row.update(family=family, quotient=q, cohort=cohort, state_key=f'synthetic-{index}')
            row['provenance'].update(family_index=index, step=0)
            graph.setdefault(family, dict(family=family, cohort=cohort, candidates=[]))
            graph[family]['candidates'].append(row)
        synthetic_registry = dict(quotients=['p'], families=[])
        reference, _, comp = selected_reference(graph, settings, synthetic_registry)
        chosen, _ = collection.select(graph, settings, synthetic_registry)
        self.assertEqual(chosen, reference)
        self.assertEqual({r['family'] for r in chosen.values()}, {'G'})
        blocked, _, _ = transitive_components([r for f in graph.values() for r in f['candidates']], {'p'}, set())
        self.assertEqual(blocked, {'A','B','C','D','E','F'})
        # A protected family must also propagate through a multi-hop quotient bridge.
        reference, _, _ = selected_reference(graph, settings, dict(quotients=[], families=['C']))
        chosen, _ = collection.select(graph, settings, dict(quotients=[], families=['C']))
        self.assertEqual(chosen, reference); self.assertEqual({r['family'] for r in chosen.values()}, {'G'})
        EVIDENCE['literal_fixture_and_components'] = dict(families=8,
            literal_choices=sum(f['choices'] for f in expected.values()),
            all_literal_rng_policy_and_canonical_candidates_match=True,
            component_closure_before_selection=True,
            protected_multihop_bridge_excluded=True,
            protected_family_multihop_bridge_excluded=True,
            cross_owner_multihop_component_excluded_both=True,
            unrelated_component_retained=True, one_root_per_family_stratum=True,
            actual_fixture_selection=proof)

    def test_04_real_collection_pre_postcommit_kills(self):
        from congklak import midgame_data_pilot as collection
        parent = self.fresh('collection'); fresh = parent / 'fresh'
        run_child(self.collection_command(fresh), parent / 'fresh.log')
        expected = rows(fresh / 'work.sqlite3', 'families', 'idx')
        expected_roots = rows(fresh / 'work.sqlite3', 'roots', 'state')
        records = []
        for event, step, prefix in (('midgame_collection_precommit', 1, 1),
                                    ('midgame_collection_committed', 2, 2)):
            run = parent / event; probe = parent / (event + '-probe')
            marker = kill_barrier(self.collection_command(run), probe, event,
                env={'MIDGAME_COLLECTION_BLOCK_EVENT':event,
                     'MIDGAME_COLLECTION_BLOCK_STEP':str(step),
                     'MIDGAME_COLLECTION_MARKER':str(probe / 'declared-marker.json')},
                condition=lambda value, s=step: value['completed'] == s)
            before = rows(run / 'work.sqlite3', 'families', 'idx')
            self.assertEqual(list(before), list(range(prefix)))
            self.backup(run, probe / 'validated-prefix.sqlite3')
            run_child(self.collection_command(run), parent / (event + '-resume.log'))
            after = rows(run / 'work.sqlite3', 'families', 'idx')
            self.assertEqual(after, expected); self.assertEqual({k:after[k] for k in before}, before)
            self.assertEqual(rows(run / 'work.sqlite3', 'roots', 'state'), expected_roots)
            stable = snapshot(run)
            with mock.patch.object(Native, 'analyze', side_effect=AssertionError('Completed collection executed teacher search')):
                collection.collect(run, fixture=True, families=8)
            self.assertEqual(snapshot(run), stable)
            self.forbidden_search_cli(self.collection_command(run), parent / (event + '-rerun.log'))
            self.assertEqual(snapshot(run), stable)
            records.append(dict(event=event, committed_prefix=prefix, families_after_resume=8,
                fresh_histories_and_selection_equal=True, prefix_retained=True,
                completed_zero_teacher_calls=True, completed_bytes_mtimes_unchanged=True, marker=marker))
        EVIDENCE['live_collection_recovery'] = dict(actual_kills=True, tests=records,
            raw_SQLite_prefix_backups_and_markers_retained=True)

    def test_05_real_H8_H10_H12_pre_postcommit_kills_and_exact_prefix_resume(self):
        from congklak import midgame_label_pilot as labeling
        parent, collected, fresh = self.teacher_fixture()
        expected_rows, expected_stores, _ = labeling.load_rows(fresh)
        records = []
        for stage in ('h8','h10','h12'):
            stage_index = ('h8','h10','h12').index(stage)
            for suffix, step, prefix in (('precommit',1,1), ('committed',2,2)):
                event = 'midgame_label_' + stage + '_' + suffix
                run = parent / event; run.mkdir(); probe = parent / (event + '-probe')
                self.backup(fresh, run / 'work.sqlite3')
                seed_counts = {}
                with closing(sqlite3.connect(run / 'work.sqlite3')) as db:
                    db.execute("DELETE FROM meta WHERE key='complete'")
                    for index, name in enumerate(('h8','h10','h12')):
                        keep = list(expected_stores[name + '_attempts'])[:1] if index == stage_index else (
                            list(expected_stores[name + '_attempts']) if index < stage_index else [])
                        for table in (name + '_attempts', name + '_labels'):
                            if keep:
                                db.execute('DELETE FROM ' + table + ' WHERE state NOT IN (' + ','.join('?' for _ in keep) + ')', keep)
                            else: db.execute('DELETE FROM ' + table)
                            seed_counts[table] = db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
                    db.commit()
                seed = dict(source='independently_executed_same_budget_fixture', source_journal=relative(fresh / 'work.sqlite3'),
                    source_journal_sha256=sha(fresh / 'work.sqlite3'), seed_journal=relative(run / 'work.sqlite3'),
                    seed_journal_sha256=sha(run / 'work.sqlite3'), counts=seed_counts,
                    current_stage=stage, seeded_current_prefix=1, new_target_index=1,
                    no_seeded_work_claimed_as_fresh=True)
                save(parent / (event + '-seed.json'), seed)
                marker = kill_barrier(self.label_command(run, collected), probe, event,
                    env={'MIDGAME_LABEL_BLOCK_EVENT':event, 'MIDGAME_LABEL_BLOCK_STEP':str(step),
                         'MIDGAME_LABEL_MARKER':str(probe / 'declared-marker.json')},
                    condition=lambda value, s=step: value['completed'] == s)
                before = {name:rows(run / 'work.sqlite3', name, 'state') for name in labeling.TABLES}
                self.assertEqual(len(before[stage + '_attempts']), prefix)
                self.backup(run, probe / 'validated-prefix.sqlite3')
                run_child(self.label_command(run, collected), parent / (event + '-resume.log'))
                actual_rows, actual_stores, manifest = labeling.load_rows(run)
                self.assertEqual(actual_rows, expected_rows)
                self.assertEqual(semantic(actual_stores), semantic(expected_stores))
                for name, values in before.items():
                    self.assertEqual({k:actual_stores[name][k] for k in values}, values)
                for name in ('h8','h10','h12'):
                    for key, attempt in actual_stores[name + '_attempts'].items():
                        for work in ('nodes','deposits'):
                            self.assertEqual(attempt['work'][work], expected_stores[name + '_attempts'][key]['work'][work])
                stable = snapshot(run)
                with mock.patch.object(Native, 'analyze', side_effect=AssertionError('Completed labels searched')):
                    labeling.run(run, fixture=True, collection_path=collected, diagnostics=2, fresh_roots=1)
                self.assertEqual(snapshot(run), stable)
                self.forbidden_search_cli(self.label_command(run, collected), parent / (event + '-rerun.log'))
                self.assertEqual(snapshot(run), stable)
                records.append(dict(stage=stage, event=event, committed_attempt_rows=prefix,
                    seed_record=relative(parent / (event + '-seed.json')),
                    seeded_prefix_from_real_independent_fixture=True,
                    actual_second_target_executed_and_killed=True,
                    exact_prefix_preserved=True, fresh_semantics_and_deterministic_work_equal=True,
                    completed_zero_teacher_calls_in_API_and_CLI=True,
                    completed_bytes_mtimes_unchanged=True, marker=marker,
                    final_counts={name:len(value) for name,value in actual_stores.items()}))
        EVIDENCE['live_label_recovery'] = dict(actual_kills=True, tests=records,
            requested_stages=['h8','h10','h12'], both_transaction_barriers_each_stage=True,
            explicit_seed_provenance=True, raw_SQLite_prefix_backups_and_markers_retained=True)

    def test_06_corruption_partial_ownership_prefix_and_prepublication_refusals(self):
        from congklak import midgame_data_pilot as collection, midgame_label_pilot as labeling
        from congklak.pilot import PilotError
        parent, collected, fresh = self.teacher_fixture()
        root_rows, stores, manifest = labeling.load_rows(fresh)
        settings = labeling.settings_for(True, 2, 1); native = Native()
        key = next(k for k in stores['h8_labels'] if k in stores['h10_labels'])
        row = root_rows[key]; h8 = stores['h8_labels'][key]
        budget = h8['report']['nodes'] + 1
        p = row['state']
        partial = native.analyze(flatten(p), actor=p['controller'], total=98, max_depth=10,
            time_ms=0, node_budget=budget, memory_mb=512, tt_mb=16, use_pvs=True, keep_history=True)
        self.assertEqual(partial['stop_reason'], 'node_budget')
        self.assertGreaterEqual(len(partial['reports']), 8); self.assertLess(len(partial['reports']), 10)
        save(parent / 'actual-bounded-H10-partial.json', partial)
        attempt = dict(study_row_sha256=config.digest(row), source_root_sha256=row['source_root_sha256'],
            stage='h10', requested_depth=10, status='budget_rejected', stop_reason='node_budget',
            completed_depth=len(partial['reports']), native_last_completed_depth=partial['last_completed_depth'],
            work={name:partial[name] for name in ('nodes','deposits','elapsed_ms','peak_process_bytes')},
            history=partial['reports'], label_sha256=None)
        bounded = copy.deepcopy(settings); bounded['h10']['node_budget'] = budget
        labeling.validate_attempt(row, attempt, None, 'h10', bounded, native)
        with self.assertRaises(ValueError): labeling.validate_attempt(row, attempt, h8, 'h10', bounded, native)
        forged = copy.deepcopy(attempt); forged.update(status='complete', stop_reason='max_depth', completed_depth=10)
        with self.assertRaises(ValueError): labeling.validate_attempt(row, forged, h8, 'h10', bounded, native)
        forged = copy.deepcopy(stores['h8_attempts'][key]); forged['work']['nodes'] = 0
        with self.assertRaises(ValueError): labeling.validate_attempt(row, forged, h8, 'h8', settings, native)
        forged = copy.deepcopy(stores['h10_attempts'][key]); forged['history'][0]['choice_depth'] = 2
        with self.assertRaises(ValueError): labeling.validate_attempt(row, forged, stores['h10_labels'][key], 'h10', settings, native)
        bad_item = copy.deepcopy(h8); bad_item['report']['search_options'].update(tt_mb=0, tt_allocated_bytes=0, tt_capacity=0, use_pvs=False)
        with self.assertRaises(ValueError): labeling.natural_study.validate_label(bad_item, row, 8, native)
        refused = []
        for kind in ('identity', 'root_role', 'covered_reference', 'attempt_prefix', 'complete_missing_export'):
            run = parent / ('bad-' + kind); shutil.copytree(fresh, run)
            with closing(sqlite3.connect(run / 'work.sqlite3')) as db:
                def put(table, key_, value):
                    db.execute('UPDATE ' + table + ' SET body=?,hash=? WHERE ' + ('key' if table=='meta' else 'state') + '=?',
                               (config.canonical(value), config.digest(value), key_))
                if kind == 'identity':
                    identity = copy.deepcopy(manifest['identity']); identity['schema'] = 'incompatible'
                    complete = copy.deepcopy(manifest); complete['identity'] = identity
                    put('meta','identity',identity); put('meta','complete',complete)
                elif kind == 'root_role':
                    k = next(k for k,r in root_rows.items() if r['study_role']=='diagnostic')
                    body = copy.deepcopy(root_rows[k]); body.update(cohort='training_candidate', study_role='fresh')
                    put('roots', k, body)
                elif kind == 'covered_reference':
                    k, body = next(iter(stores['old_h8_references'].items())); body = copy.deepcopy(body)
                    body['label']['target_player0'] += 1; put('old_h8_references', k, body)
                elif kind == 'attempt_prefix':
                    first = next(iter(stores['h10_attempts']))
                    db.execute('DELETE FROM h10_attempts WHERE state=?',(first,))
                else:
                    body = copy.deepcopy(manifest); body['roots'] += 1; put('meta','complete',body)
                db.commit()
            (run / 'manifest.json').unlink()
            with self.assertRaises((ValueError,KeyError,FileNotFoundError)):
                labeling.load_rows(run)
            with mock.patch.object(Native, 'analyze', side_effect=AssertionError('Corruption restarted search')):
                with self.assertRaises((ValueError,PilotError)):
                    labeling.run(run, fixture=True, collection_path=collected, diagnostics=2, fresh_roots=1)
            self.assertFalse((run / 'manifest.json').exists(), 'Invalid label marker was published')
            refused.append(kind)
        bad = parent / 'bad-collection-marker'; shutil.copytree(collected, bad)
        with closing(sqlite3.connect(bad / 'work.sqlite3')) as db:
            body = json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0])
            body['families'] += 1
            db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(body),config.digest(body)))
            db.commit()
        (bad / 'manifest.json').unlink()
        with self.assertRaises(ValueError): collection.collect(bad, fixture=True, families=8)
        self.assertFalse((bad / 'manifest.json').exists(), 'Invalid collection marker was published')
        repair = parent / 'missing-export'; shutil.copytree(fresh, repair)
        (repair / 'manifest.json').unlink(); before = snapshot(repair)
        with mock.patch.object(Native, 'analyze', side_effect=AssertionError('Export repair searched')):
            labeling.run(repair, fixture=True, collection_path=collected, diagnostics=2, fresh_roots=1)
        after = snapshot(repair)
        self.assertEqual({k:after[k] for k in before}, before)
        self.assertEqual(load(repair / 'manifest.json'), manifest)
        EVIDENCE['refusals_and_repair'] = dict(real_H10_partial_completed_depth=len(partial['reports']),
            bounded_probe_node_cap=budget, partial_not_requested_label=True,
            forged_full_depth_zero_work_history_and_wrong_TT_PVS_refused=True,
            rehashed_corruptions_refused=refused, invalid_collection_complete_marker_refused=True,
            no_invalid_export_published=True, valid_missing_export_repaired_without_search=True,
            all_existing_repair_inputs_bytes_mtimes_unchanged=True,
            bounded_probe_separate_from_production_budget=True)

    def test_07_all512_production_histories_transitive_ownership_and_honest_yield(self):
        from congklak import midgame_data_pilot as collection
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        fresh, manifest = collection.load_rows()
        saved = rows(collection.DEFAULT / 'work.sqlite3','families','idx')
        self.assertEqual(list(saved),list(range(512))); expected = {}
        with CpuBatchEvaluator(CONTROL,cache_capacity=0,max_batch=1) as evaluator:
            for index in range(512):
                expected[index] = literal_history(index,collection.CONTRACT,evaluator)
                self.assertEqual(saved[index],expected[index],f'Production family {index}')
        registry = load(collection.REGISTRY)
        selected, proof, components = selected_reference(expected,collection.CONTRACT,registry)
        self.assertEqual(fresh,selected)
        selection = manifest['selection']
        self.assertEqual(selection['counts'],proof['counts']);self.assertEqual(selection['shortfall'],proof['shortfall'])
        self.assertEqual(selection['observations'],proof['eligible_observations'])
        self.assertEqual(selection['unique_candidate_quotients'],proof['eligible_unique_quotients'])
        self.assertEqual(selection['component_count'],proof['graph_components'])
        self.assertEqual(selection['blocked_family_count'],proof['blocked_families'])
        reason_map = {'protected_quotient':'protected_eligible_quotient',
                      'protected_family':'old_protected_family','cross_cohort_component':'mixed_ownership_component'}
        converted = [dict(c,reasons=[reason_map[r] for r in c['reasons']]) for c in components]
        self.assertEqual(selection['components'],sorted(converted,key=lambda c:c['families']))
        candidates = [r for f in expected.values() for r in f['candidates']]
        self.assertEqual(selection['observed_by_stratum'],dict(sorted(Counter(r['stratum'] for r in candidates).items())))
        blocked = {f for c in components if c['reasons'] for f in c['families']}
        self.assertEqual(selection['blocked_families'],sorted(blocked))
        self.assertEqual(selection['blocked_candidate_observations'],sum(r['family'] in blocked for r in candidates))
        protected_q = set(registry['quotients'])
        self.assertEqual(selection['direct_protected_candidate_observations'],sum(r['quotient'] in protected_q for r in candidates))
        self.assertLessEqual(len(fresh),50)
        self.assertEqual(len({(r['family'],r['stratum']) for r in fresh.values()}),len(fresh))
        self.assertEqual(len({r['quotient'] for r in fresh.values()}),len(fresh))
        for row in fresh.values():
            self.assertNotIn(row['family'],registry['families']);self.assertNotIn(row['quotient'],registry['quotients'])
            self.assertIn(row['stratum'],collection.STRATA);self.assertGreaterEqual(row['legal_mask'].bit_count(),2)
            self.assertEqual(mechanism(row['state'])[:2],(row['active'],row['stratum']))
        self.assertEqual(manifest,load(collection.OUTPUT));self.assertEqual(manifest,load(collection.DEFAULT/'manifest.json'))
        self.assertEqual(Counter(f['source'] for f in expected.values()),dict(random_history=256,frozen_policy_epsilon_history=256))
        self.assertFalse(manifest['teacher_searches']);self.assertFalse(manifest['training_performed'])
        EVIDENCE['production_collection'] = dict(families=512, selected_fresh_roots=len(fresh),
            literal_choices=sum(f['choices'] for f in expected.values()),
            policy_evaluations=sum(f['policy_evaluations'] for f in expected.values()),
            source_families=dict(Counter(f['source'] for f in expected.values())),
            family_cohorts=dict(Counter(f['cohort'] for f in expected.values())),
            all_literal_histories_and_candidates_equal=True,
            every_connected_component_and_selection_independently_rederived=True,
            selected_counts=proof['counts'],shortfall=proof['shortfall'],
            blocked_families=len(blocked), no_constructed_or_relaxed_quota_filler=True,
            one_root_per_family_stratum=True,shared_ineligible_prefixes_not_independent_claim=True)

    def test_08_all_requested_auxiliary_Q_PVs_fixed_metrics_weights_and_closed_reruns(self):
        from congklak import midgame_data_pilot as collection,midgame_label_pilot as labeling,reports
        from congklak.neural_relay import NeuralNative
        fresh,collected = collection.load_rows(); wrappers,stores,manifest = labeling.load_rows()
        self.assertEqual({k:r['root'] for k,r in wrappers.items() if r['study_role']=='fresh'},fresh)
        self.assertEqual({k:r['root'] for k,r in wrappers.items() if r['study_role']=='diagnostic'},load(collection.REGISTRY)['diagnostics'])
        self.assertEqual(len(stores['old_h8_references']),1)
        self.assertEqual(stores['old_h8_references'],load(labeling.REGISTRY)['old_h8_references'])
        native=Native();paths=steps=history_count=labels_count=total_nodes=0
        signatures={key:{} for key in wrappers}
        for key,item in stores['old_h8_references'].items():
            signatures[key][8]={field:item['label'][field] for field in ('target_player0','best_moves','action_scores')}
        for stage,depth,budget in (('h8',8,2000000),('h10',10,5000000),('h12',12,10000000)):
            targets=[k for k,r in wrappers.items() if stage!='h12' or r['study_role']=='diagnostic']
            attempts,labels=stores[stage+'_attempts'],stores[stage+'_labels']
            self.assertEqual(list(attempts),targets);self.assertLessEqual(set(labels),set(attempts))
            self.assertEqual(manifest['stages'][stage]['targets'],len(targets))
            self.assertEqual(manifest['stages'][stage]['attempts'],len(attempts))
            self.assertEqual(manifest['stages'][stage]['labels'],len(labels))
            self.assertEqual(manifest['stages'][stage]['rejected'],sum(a['status']=='budget_rejected' for a in attempts.values()))
            self.assertEqual(manifest['stages'][stage]['pending'],0)
            for key,attempt in attempts.items():
                row=wrappers[key];history=attempt['history'];total_nodes+=attempt['work']['nodes']
                self.assertEqual(attempt['study_row_sha256'],config.digest(row));self.assertEqual(attempt['source_root_sha256'],config.digest(row['root']))
                self.assertEqual((attempt['stage'],attempt['requested_depth'],attempt['completed_depth']),(stage,depth,len(history)))
                self.assertGreater(attempt['work']['nodes'],0);self.assertLessEqual(attempt['work']['nodes'],budget)
                previous={name:0 for name in attempt['work']}
                for expected_depth,report in enumerate(history,1):
                    self.assertEqual(report['choice_depth'],expected_depth)
                    reports.validate(report,row['state'],config.identity(native.build_id),native)
                    self.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True))
                    for name,value in previous.items():
                        self.assertLessEqual(value,report[name]);self.assertLessEqual(report[name],attempt['work'][name])
                    previous={name:report[name] for name in previous}
                    item=history_item(report,row);pvs,choices=check_full_label(self,row,item,expected_depth)
                    paths+=pvs;steps+=choices;history_count+=1
                    signature={field:item['label'][field] for field in ('target_player0','best_moves','action_scores')}
                    if expected_depth in signatures[key]:self.assertEqual(signatures[key][expected_depth],signature)
                    signatures[key][expected_depth]=signature
                if key in labels:
                    item=labels[key];self.assertEqual(attempt['status'],'complete');self.assertEqual(attempt['stop_reason'],'max_depth')
                    self.assertEqual(len(history),depth);self.assertEqual(item,history_item(history[-1],row))
                    self.assertEqual(attempt['label_sha256'],config.digest(item));labels_count+=1
                else:
                    self.assertEqual((attempt['status'],attempt['stop_reason']),('budget_rejected','node_budget'))
                    self.assertLess(len(history),depth);self.assertEqual(attempt['work']['nodes'],budget)
                    self.assertIsNone(attempt['label_sha256'])
        self.assertEqual(total_nodes,manifest['total_new_nodes']);self.assertLessEqual(total_nodes,503000000)
        self.assertEqual(history_count,manifest['auxiliary_completed_reports'])
        comparison=audit_metrics(self,wrappers,stores)
        weights=audit_weights(self,fresh,wrappers,stores)
        watched=[collection.DEFAULT, labeling.DEFAULT]
        before={relative(path):snapshot(path) for path in watched}
        reports_=[collection.OUTPUT,labeling.OUTPUT,ROOT/'build/midgame-label-pilot-comparison.json',ROOT/'build/midgame-pilot-weight-review.json']
        published={relative(path):(sha(path),path.stat().st_mtime_ns) for path in reports_}
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed production performed teacher search')) as blocked:
            collection.collect();labeling.run()
        self.assertEqual(blocked.call_count,0)
        for path in watched:self.assertEqual(snapshot(path),before[relative(path)])
        for path in reports_:self.assertEqual((sha(path),path.stat().st_mtime_ns),published[relative(path)])
        self.forbidden_search_cli([sys.executable,str(ROOT/'tools/midgame-data-pilot.py'),'collect'],self.base/'production-collection-rerun.log')
        self.forbidden_search_cli([sys.executable,str(ROOT/'tools/midgame-label-pilot.py'),'all'],self.base/'production-label-rerun.log')
        self.forbidden_search_cli([sys.executable,str(ROOT/'tools/analyze-midgame-label-pilot.py'),'verify'],self.base/'comparison-rerun.log')
        self.forbidden_search_cli([sys.executable,str(ROOT/'tools/review-midgame-pilot-weights.py')],self.base/'weights-rerun.log')
        for path in watched:self.assertEqual(snapshot(path),before[relative(path)])
        for path in reports_:self.assertEqual((sha(path),path.stat().st_mtime_ns),published[relative(path)])
        for name,digest in load(BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(CONTROL),CONTROL_SHA);self.assertEqual(sha(PILOT),PILOT_SHA)
        self.assertNotIn('torch',sys.modules)
        EVIDENCE['production_labels_metrics_and_weights']=dict(roots=len(wrappers),fresh_roots=len(fresh),diagnostic_roots=9,
            requested_labels=labels_count,auxiliary_completed_reports=history_count,
            literal_PV_paths=paths,literal_PV_choices=steps,total_new_production_nodes=total_nodes,
            all_full_legal_Q_best_sets_and_PV_leaves_verified=True,
            all_same_depth_old_new_and_cross_stage_signatures_equal=True,
            capped_attempts_never_promoted_to_requested_labels=True,
            metrics=comparison,offline_weights=weights,
            completed_zero_teacher_calls_in_API_and_CLI=True,
            completed_all_durable_bytes_mtimes_unchanged=True,all2639_prior_pins_current=True,
            both_models_and_shared_policy_unchanged=True,training_or_ingestion_performed=False)


GROUPS = {
    'preflight': [
        'test_01_baseline2639_fixed_models_and_frozen_contract',
        'test_02_literal_latest144_ownership_83roles_259dev_and_nine_diagnostics',
        'test_03_literal_fixture_sampling_and_multihop_component_closure',
        'test_04_real_collection_pre_postcommit_kills',
        'test_05_real_H8_H10_H12_pre_postcommit_kills_and_exact_prefix_resume',
        'test_06_corruption_partial_ownership_prefix_and_prepublication_refusals'],
    'final': [
        'test_07_all512_production_histories_transitive_ownership_and_honest_yield',
        'test_08_all_requested_auxiliary_Q_PVs_fixed_metrics_weights_and_closed_reruns']}


def collect():
    expected = [name for methods in GROUPS.values() for name in methods]
    inputs, evidence, groups, covered = {}, {}, [], []
    for name, methods in GROUPS.items():
        path = WORK / (name + '.json'); report = load(path)
        assert report['passed'] is True and report['covered_methods'] == methods
        assert report['tests_run'] == len(methods) and not any(report[k] for k in ('failures','errors','skips'))
        for filename, digest in report['inputs'].items():
            assert sha(ROOT / filename) == digest, 'Audit input changed: ' + filename
            assert filename not in inputs or inputs[filename] == digest
            inputs[filename] = digest
        assert not (set(evidence) & set(report['evidence']))
        evidence.update(report['evidence']); covered.extend(methods)
        inputs[relative(path)] = sha(path)
        groups.append(dict(path=relative(path), sha256=sha(path), tests_run=report['tests_run'], covered_methods=methods))
    assert covered == expected and len(set(covered)) == 8
    inputs[relative(Path(__file__))] = sha(__file__)
    body = dict(schema='midgame-pilot-independent-audit-v1', passed=True, tests_run=8,
                failures=0, errors=0, skips=0, covered_methods=covered,
                groups=groups, inputs=inputs, evidence=evidence)
    output = ROOT / 'build/midgame-pilot-audit.json'
    if output.exists(): assert load(output) == body, 'Immutable combined audit differs'
    else: save(output, body)
    return body


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=[*GROUPS,'collect']); args = parser.parse_args()
    if args.group == 'collect':
        body = collect(); print(config.canonical(dict(event='midgame_pilot_independent_audit',
            group='collect', passed=True, tests=body['tests_run'])),flush=True); return 0
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(
        MidgamePilotIndependentAudit(name) for name in GROUPS[args.group]))
    files = [Path(__file__),BASELINE,
        ROOT/'congklak/midgame_data_pilot.py',ROOT/'tools/midgame-data-pilot.py',
        ROOT/'congklak/midgame_label_pilot.py',ROOT/'tools/midgame-label-pilot.py',
        ROOT/'build/midgame-data-pilot-registry.json',ROOT/'build/midgame-data-pilot-plan.json',
        ROOT/'build/midgame-data-pilot-implementation.json',ROOT/'build/midgame-label-pilot-registry.json',
        ROOT/'build/midgame-label-pilot-plan.json',ROOT/'build/midgame-label-pilot-implementation.json',
        ROOT/'tools/analyze-midgame-label-pilot.py',ROOT/'tools/review-midgame-pilot-weights.py']
    files += [p for p in MidgamePilotIndependentAudit.base.rglob('*')
              if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if args.group == 'final':
        files += [ROOT/name for name in ('tests/midgame_pilot_metric_audit.py',
            'data/midgame-data-pilot-v1/work.sqlite3','data/midgame-data-pilot-v1/manifest.json',
            'data/midgame-label-pilot-v1/work.sqlite3','data/midgame-label-pilot-v1/manifest.json',
            'build/midgame-data-pilot-result.json','build/midgame-label-pilot-result.json',
            'build/midgame-label-pilot-comparison.json','build/midgame-pilot-weight-review.json')]
    output = WORK / (args.group + '.json')
    body = dict(schema='midgame-pilot-independent-group-v1',passed=result.wasSuccessful(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        covered_methods=GROUPS[args.group],inputs=pins([p for p in files if p.is_file()]),
        evidence=EVIDENCE,duration_seconds=time.monotonic()-started,
        retained_directory=relative(MidgamePilotIndependentAudit.base),
        failure_details=[dict(method=str(t),trace=trace) for t,trace in result.failures+result.errors])
    save(output,body)
    print(config.canonical(dict(event='midgame_pilot_independent_audit',group=args.group,
                               passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
