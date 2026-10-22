"""Independent exact arithmetic and literal saved-line capture investigation."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import unittest
import uuid
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import reference as literal

WORK = ROOT / 'build/capture-audit'
FREEZE = ROOT / 'build/capture-audit-source-freeze.json'
BASELINE = ROOT / 'build/capture-baseline.json'
EVIDENCE = {}
GROUPS = {
 'preflight': ['test_01_prior4923_exact_support_sources_reservations_and_models',
               'test_02_exact_attribution_signed_ranks_ties_and_cancellation',
               'test_03_real_case_pre_post_commit_kills_and_exact_resume'],
 'production': ['test_04_all_capture_reports_actions_histories_and_literal_PVs',
                'test_05_all16_exact_contributions_covariance_and_forensic_reports'],
 'final': ['test_06_rehashed_corruption_missing_export_and_closed_no_work']}
STAGES = ('h8', 'h10', 'h12')


def canonical(body): return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
def digest(body): return hashlib.sha256(canonical(body).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def save(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Immutable audit artifact exists: ' + str(path)
    path.write_text(canonical(body) + '\n', encoding='utf-8', newline='\n')
def pins(paths): return {rel(p): sha(p) for p in sorted(set(map(Path, paths))) if p.is_file()}
def current(mapping):
    for name, checksum in mapping.items(): assert sha(ROOT / name) == checksum, name
def durable(path):
    return path.is_file() and path.suffix != '.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size > 0)
def files(path): return [p for p in Path(path).rglob('*') if durable(p)] if Path(path).exists() else []
def qread(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
def rows(path, table, key):
    assert table.replace('_', '').isalnum() and key.replace('_', '').isalnum()
    with closing(qread(path)) as db:
        out = {}
        for k, text, checksum in db.execute('SELECT ' + key + ',body,hash FROM ' + table + ' ORDER BY ' + key):
            body = json.loads(text); assert checksum == digest(body); assert k not in out; out[k] = body
        return out
def physical_rows(path, table, key):
    with closing(qread(path)) as db: return [list(v) for v in db.execute('SELECT ' + key + ',body,hash FROM ' + table + ' ORDER BY ' + key)]
def backup(source, target):
    with closing(qread(source)) as a, closing(sqlite3.connect(target)) as b: a.backup(b)
def mutate_row(path, table, key, column, callback):
    with closing(sqlite3.connect(path)) as db:
        body = json.loads(db.execute('SELECT body FROM ' + table + ' WHERE ' + column + '=?', (key,)).fetchone()[0])
        callback(body)
        with db: db.execute('UPDATE ' + table + ' SET body=?,hash=? WHERE ' + column + '=?', (canonical(body), digest(body), key))
def run_child(command, log, timeout=600, env=None):
    with Path(log).open('w', encoding='utf-8', newline='\n') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, encoding='utf-8', timeout=timeout,
                                env={**os.environ, **(env or {})})
    assert result.returncode == 0, Path(log).read_text(encoding='utf-8')[-18000:]
def kill(command, probe, event, completed, env):
    probe.mkdir(); marker = probe / 'marker.json'; ready = threading.Event(); output = []
    with (probe / 'worker.log').open('w', encoding='utf-8', newline='\n') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', env={**os.environ, **env})
        def consume():
            for line in child.stdout:
                stream.write(line); stream.flush(); output.append(line)
                try: body = json.loads(line)
                except ValueError: continue
                if isinstance(body, dict) and body.get('event') == event and body.get('completed') == completed:
                    save(marker, body); ready.set(); return
        thread = threading.Thread(target=consume, daemon=True); thread.start()
        try:
            deadline = time.monotonic() + 180
            while not ready.wait(.05) and child.poll() is None and time.monotonic() < deadline: pass
            assert marker.exists(), 'Barrier not observed: ' + ''.join(output)[-12000:]
            assert child.poll() is None, 'Child exited before actual process kill'
            child.kill(); child.wait(10); thread.join(5)
        finally:
            if child.poll() is None: child.kill(); child.wait(10)
    return read(marker)
def frac(body):
    assert set(body) == {'numerator', 'denominator', 'value'}
    assert type(body['numerator']) is int and type(body['denominator']) is int and body['denominator'] > 0
    v = F(body['numerator'], body['denominator']); assert body == rational(v); return v
def rational(value):
    v = F(value); return dict(numerator=v.numerator, denominator=v.denominator, value=float(v))
def pos(state):
    assert set(state) == {'pits', 'stores', 'controller'}
    values = state['pits'][0] + [state['stores'][0]] + state['pits'][1] + [state['stores'][1]]
    assert len(values) == 16 and all(type(v) is int and 0 <= v <= 98 for v in values) and sum(values) == 98
    assert type(state['controller']) is int and state['controller'] in (0, 1)
    p = literal.Position(7, tuple(values), state['controller']); assert literal.normalize(p).controller == p.controller
    return p
def state(p): return dict(pits=[list(p.counts[:7]), list(p.counts[8:15])], stores=[p.counts[7], p.counts[15]], controller=p.controller)
def raw(p): return bytes(list(pos(p).counts) + [p['controller']]).hex()
def bits(value):
    if isinstance(value, list): return b''.join(bits(v) for v in value)
    return struct.pack('<d', value)


def source_data():
    registry = read(ROOT / 'build/stratum-registry.json')
    rule = read(ROOT / 'build/stratum-rule.json')
    result = read(ROOT / 'build/stratum-result.json')
    db = ROOT / 'data/stratum-weighting-v1/work.sqlite3'
    stores = {s + suffix: rows(db, s + suffix, 'state') for s in STAGES for suffix in ('_attempts', '_labels')}
    return registry, rule, result, rows(db, 'roots', 'state'), stores


def label_q(item, actor):
    label = item['label']; q = {r['pit']: r['value_player0'] for r in label['action_scores']}
    assert len(q) == len(label['action_scores']) and all(type(v) is int for v in q.values())
    sign = 1 if actor == 0 else -1; best = max(sign * v for v in q.values())
    assert label['best_moves'] == [a for a in q if sign * q[a] == best]
    assert sign * label['target_player0'] == best
    return q


def drift_parts(deltas):
    assert deltas
    mean = F(sum(deltas), len(deltas)); absolute = F(sum(abs(v) for v in deltas), len(deltas))
    return dict(signed_mean=mean, absolute_mean=absolute, translation=abs(mean),
                signed_cancellation=absolute-abs(mean), centered_dispersion=sum((abs(F(v)-mean) for v in deltas), F()) / len(deltas))


def covariance(raw_weights, drifts):
    assert len(raw_weights) == len(drifts) == 2 and all(v > 0 for v in raw_weights)
    r, s = raw_weights; d, e = drifts
    conditional = (r*d+s*e)/(r+s)-(d+e)/2
    formula = (r-s)*(d-e)/(2*(r+s)); assert conditional == formula
    return conditional


def independent_attribution(registry, rule, stores):
    derived = {}; cells = defaultdict(list)
    for key, row in sorted(registry['rows'].items()):
        actor = row['actor']; sign = 1 if actor == 0 else -1
        q = {s: label_q(stores[s+'_labels'][key], actor) for s in STAGES}
        legal = list(literal.legal_actions(pos(row['state'])))
        assert all(list(values) == legal for values in q.values())
        root_shift = abs(stores['h10_labels'][key]['label']['target_player0']-stores['h8_labels'][key]['label']['target_player0'])
        q_shift = F(sum(abs(q['h10'][a]-q['h8'][a]) for a in legal), len(legal))
        raw_weight = max(F(1, 4), 1/(1+F(root_shift, 4)+q_shift/4))
        projection = rule['projection']['rows'][key]
        assert raw_weight == frac(projection['raw_rule']['raw_weight'])
        w = frac(projection['weighted_probability']); u = frac(projection['uniform_probability']); assert u == F(1, 16)
        delta = {a: q['h12'][a]-q['h10'][a] for a in legal}; parts = drift_parts(list(delta.values()))
        best = stores['h10_labels'][key]['label']['best_moves']
        action_contributions = {a: (w-u)*abs(delta[a])/len(legal) for a in legal}
        contribution = sum(action_contributions.values(), F()); assert contribution == (w-u)*parts['absolute_mean']
        rank_pairs = []
        for i, a in enumerate(legal):
            for b in legal[i+1:]:
                before = sign*(q['h10'][a]-q['h10'][b]); after = sign*(q['h12'][a]-q['h12'][b])
                rank_pairs.append(dict(a=a, b=b, gap10=before, gap12=after,
                                       strict_reversal=before*after < 0, tie_change=(before == 0) != (after == 0)))
        derived[key] = dict(actor=actor, cell=row['cell'], stratum=row['stratum'], legal=legal, q=q,
            uniform=u, weighted=w, raw_weight=raw_weight, delta=delta, actor_delta={a:sign*v for a,v in delta.items()},
            parts=parts, action_contributions=action_contributions, contribution=contribution,
            best_contribution=sum((action_contributions[a] for a in best), F()),
            nonbest_contribution=sum((v for a,v in action_contributions.items() if a not in best), F()), ranks=rank_pairs)
        cells[row['cell']].append(key)
    assert len(derived) == 16 and len(cells) == 8 and all(len(v) == 2 for v in cells.values())
    cell_contributions = {}
    for cell, keys in cells.items():
        assert sum((derived[k]['weighted'] for k in keys), F()) == F(1, 8)
        value = covariance([derived[k]['raw_weight'] for k in keys], [derived[k]['parts']['absolute_mean'] for k in keys])/8
        assert value == sum((derived[k]['contribution'] for k in keys), F()); cell_contributions[cell] = value
    total = sum((r['contribution'] for r in derived.values()), F())
    return derived, cell_contributions, total


def replay_pv(start, moves, depth, value):
    p = pos(start); summaries = []
    for index, move in enumerate(moves):
        assert move['actor'] == p.controller and move['pit'] in literal.legal_actions(p)
        transition = literal.apply(p, move['pit']); after = transition.position
        assert literal.replay_trace(transition) == after.counts and sum(after.counts) == 98
        assert move['terminal_after'] is transition.terminal
        assert move['next_controller'] == (None if transition.terminal else after.controller)
        assert move['bonus_choice'] is (transition.bonus and not transition.terminal)
        assert move['forced_passes_after'] == [dict(zip(('from', 'to'), v)) for v in transition.final_passes]
        summaries.append(dict(index=index, actor=p.controller, pit=move['pit'], before=state(p), after=state(after),
            capture_seeds=transition.captured, relays=transition.relay_pickups, deposits=transition.deposits,
            bonus=transition.bonus, terminal=transition.terminal, passes=[list(v) for v in transition.final_passes],
            store_gain=[after.counts[j]-p.counts[j] for j in (7, 15)]))
        p = after
    assert len(moves) == depth or literal.normalize(p).terminal
    assert p.counts[7]-p.counts[15] == value
    return dict(steps=summaries, leaf=state(p), choices=len(moves), terminal=literal.normalize(p).terminal,
                leaf_active=sum(p.counts[i] for i in range(16) if i not in (7, 15)), value_player0=value)


def report_literal(root, report):
    actor = root['state']['controller']; sign = 1 if actor == 0 else -1
    p = pos(root['state']); legal = list(literal.legal_actions(p)); depth = report['choice_depth']
    assert report['evaluator'] == 'H_v1' and report['depth_result']['evaluator'] == 'H_v1'
    assert report['position'] == root['state'] and report['legal_actions'] == legal
    assert report['normalized_controller'] == actor and report['completed_iteration'] is True
    assert [c['pit'] for c in report['candidates']] == legal
    assert report['search_options']['tt_mb'] == 16 and report['search_options']['use_pvs'] is True
    values = {c['pit']: c['depth_result']['value_player0'] for c in report['candidates']}
    best_value = max(sign*v for v in values.values()); best = [a for a in legal if sign*values[a] == best_value]
    assert report['best_moves'] == best and sign*report['depth_result']['value_player0'] == best_value
    evidence = {'root': replay_pv(root['state'], report['pv'], depth, sign*best_value), 'actions': {}}
    for candidate in report['candidates']:
        a = candidate['pit']; t = literal.apply(p, a); native = candidate['transition']; score = candidate['depth_result']
        assert score['choice_depth'] == depth and score['value_actor'] == sign*values[a] and score['bound'] == 'exact'
        assert candidate['actor'] == actor and candidate['analysis_complete'] is True
        assert native['position'] == state(t.position) and native['actor'] == actor and native['pit'] == a
        assert native['drops'] == t.deposits and native['relays'] == t.relay_pickups and native['terminal'] is t.terminal
        assert native['store_delta'] == [t.position.counts[j]-p.counts[j] for j in (7, 15)]
        assert sum(e['count'] for e in native['events'] if e['type'] == 'capture') == t.captured
        assert candidate['pv'][0]['pit'] == a
        evidence['actions'][str(a)] = replay_pv(root['state'], candidate['pv'], depth, values[a])
    return evidence


def check_line_projection(test, trace):
    expected = replay_pv(trace['initial_state'], trace['pv'], trace['requested_depth'], trace['leaf']['saved_score_player0'])
    test.assertEqual(trace['leaf']['state'], expected['leaf']); test.assertEqual(trace['leaf']['state_key'], raw(expected['leaf']))
    test.assertEqual(trace['leaf']['terminal'], expected['terminal']); test.assertEqual(trace['leaf']['active_seeds'], expected['leaf_active'])
    test.assertEqual(trace['leaf']['saved_score_minus_leaf_store_margin'], 0)
    captured = [0, 0]; gains = [0, 0]; totals = Counter()
    for index, (actual, old) in enumerate(zip(trace['steps'], expected['steps']), 1):
        test.assertEqual(actual['choice'], index)
        for field in ('actor', 'pit', 'before', 'after', 'store_gain', 'relays', 'deposits'):
            test.assertEqual(actual[field], old[field])
        test.assertEqual(actual['display_pit'], old['pit']+1); test.assertEqual(actual['before_state_key'], raw(old['before']))
        test.assertEqual(actual['after_state_key'], raw(old['after'])); test.assertEqual(actual['captured_seeds'], old['capture_seeds'])
        test.assertEqual(actual['bonus_choice'], old['bonus'] and not old['terminal']); test.assertEqual(actual['terminal_after'], old['terminal'])
        test.assertEqual(actual['forced_passes_after'], [dict(zip(('from', 'to'), p)) for p in old['passes']])
        t = literal.apply(pos(old['before']), old['pit']); last = next(e['index'] for e in reversed(t.events) if e['kind'] == 'deposit')
        reason = 'own_store' if t.bonus else 'own_capture' if t.captured else 'own_empty' if last in literal.row_indices(7, t.actor) else 'opponent_empty'
        test.assertEqual(actual['reason'], reason); test.assertEqual(actual['literal_events_sha256'], digest(list(t.events)))
        captures = [dict(index=e['index'], opposite=e['opposite'], seeds=e['seeds']) for e in t.events if e['kind'] == 'capture']
        test.assertEqual(actual['capture_events'], captures)
        test.assertEqual(actual['relay_events'], [dict(index=e['index'], seeds=e['seeds']) for e in t.events if e['kind'] == 'relay'])
        captured[old['actor']] += old['capture_seeds']; gains = [gains[a]+old['store_gain'][a] for a in (0, 1)]
        test.assertEqual(actual['cumulative'], dict(captured_seeds_by_player=captured, store_gain_by_player=gains, store_margin_change_player0=gains[0]-gains[1]))
        totals.update(dict(capture_events=len(captures), bonus_choices=int(old['bonus'] and not old['terminal']),
                           relays=old['relays'], deposits=old['deposits'], forced_passes=len(old['passes'])))
    test.assertEqual(len(trace['steps']), len(expected['steps']))
    test.assertEqual(trace['totals'], dict(choices=len(expected['steps']), captured_seeds_by_player=captured, store_gain_by_player=gains,
        store_margin_change_player0=gains[0]-gains[1], capture_margin_change_player0=captured[0]-captured[1],
        sowing_store_margin_change_player0=gains[0]-gains[1]-captured[0]+captured[1], **dict(totals)))
    return expected


def check_comparison(test, result, left, right):
    prefix = 0; different = None
    for a, b in zip(left['steps'], right['steps']):
        test.assertEqual(a['before'], b['before'])
        if (a['actor'], a['pit']) != (b['actor'], b['pit']): different = (a, b); break
        test.assertEqual(a['after'], b['after']); prefix += 1
    relation = 'different_choice' if different else 'identical_recorded_choices' if len(left['steps']) == len(right['steps']) else 'one_recorded_line_is_a_prefix'
    test.assertEqual(result['relationship'], relation); test.assertEqual(result['common_choice_prefix'], prefix)
    test.assertEqual(result['left_leaf'], left['leaf']); test.assertEqual(result['right_leaf'], right['leaf'])
    if different:
        a, b = different; actual = result['first_different_choice']; test.assertEqual(actual['choice'], prefix+1)
        test.assertEqual(actual['state'], a['before']); test.assertEqual(actual['state_key'], a['before_state_key'])
        test.assertEqual((actual['left_actor'], actual['right_actor'], actual['left_pit'], actual['right_pit']), (a['actor'], b['actor'], a['pit'], b['pit']))
        test.assertEqual(actual['left_mechanics'], a); test.assertEqual(actual['right_mechanics'], b)
    else: test.assertIsNone(result['first_different_choice'])
    for side, line in (('left', left), ('right', right)):
        tail = line['steps'][prefix:]; captures = [sum(s['captured_seeds'] for s in tail if s['actor'] == a) for a in (0, 1)]
        gains = [sum(s['store_gain'][a] for s in tail) for a in (0, 1)]
        test.assertEqual(result[side+'_tail_after_common_prefix'], dict(choices=len(tail), captured_seeds_by_player=captures,
            store_gain_by_player=gains, store_margin_change_player0=gains[0]-gains[1], bonus_choices=sum(s['bonus_choice'] for s in tail),
            forced_passes=sum(len(s['forced_passes_after']) for s in tail)))


def all_action_ledger(registry, rule, rootmap, stores):
    """Retain every action's global contribution, including non-capture context."""
    out = {}
    for key, row in sorted(registry['rows'].items()):
        sign = 1 if row['actor'] == 0 else -1
        q = {s:label_q(stores[s+'_labels'][key], row['actor']) for s in STAGES}
        best = {s:stores[s+'_labels'][key]['label']['best_moves'] for s in STAGES}; legal = sorted(q['h10'])
        probability = rule['projection']['rows'][key]; u = frac(probability['uniform_probability']); w = frac(probability['weighted_probability'])
        entry = dict(state_key=key, raw_state=row['state'], source_metadata_sha256=digest(row), source_root_sha256=rootmap[key]['source_root_sha256'],
            study_wrapper_sha256=digest(rootmap[key]), actor=row['actor'], cell=row['cell'], stratum=row['stratum'], family=row['family'],
            legal_actions=legal, uniform_probability=rational(u), weighted_probability=rational(w),
            endpoints={s:dict(item_sha256=digest(stores[s+'_labels'][key]), target_player0=stores[s+'_labels'][key]['label']['target_player0'],
                              Q_player0={str(a):v for a,v in q[s].items()}, best_moves=best[s]) for s in STAGES}, pairs={})
        for before, after in (('h8','h10'), ('h10','h12'), ('h8','h12')):
            deltas={a:q[after][a]-q[before][a] for a in legal}; parts=drift_parts(list(deltas.values())); actions={}; ranks=[]
            for a in legal:
                actions[str(a)]=dict(pit=a, before_Q_player0=q[before][a], after_Q_player0=q[after][a],
                    delta_player0=deltas[a], delta_actor=sign*deltas[a], absolute_drift=abs(deltas[a]),
                    before_best=a in best[before], after_best=a in best[after], H10_best=a in best['h10'],
                    gap_before_actor=sign*(stores[before+'_labels'][key]['label']['target_player0']-q[before][a]),
                    gap_after_actor=sign*(stores[after+'_labels'][key]['label']['target_player0']-q[after][a]),
                    root_mean_absolute_contribution=rational(F(abs(deltas[a]),len(legal))),
                    uniform_global_contribution=rational(u*abs(deltas[a])/len(legal)),
                    weighted_global_contribution=rational(w*abs(deltas[a])/len(legal)),
                    global_weighted_minus_uniform=rational((w-u)*abs(deltas[a])/len(legal)))
            for i,a in enumerate(legal):
                for b in legal[i+1:]:
                    old=sign*(q[before][a]-q[before][b]); new=sign*(q[after][a]-q[after][b])
                    ranks.append(dict(actions=[a,b], before_gap_actor=old, after_gap_actor=new, strict_reversal=old*new<0, tie_change=(old==0)!=(new==0)))
            partitions={}
            for name, selected in (('H10_best',set(best['h10'])),('H10_nonbest',set(legal)-set(best['h10']))):
                part=F(sum(abs(deltas[a]) for a in selected),len(legal))
                partitions[name]=dict(actions=sorted(selected), all_legal_denominator=len(legal), root_drift_part=rational(part),
                    uniform_global_contribution=rational(u*part), weighted_global_contribution=rational(w*part), global_weighted_minus_uniform=rational((w-u)*part))
            entry['pairs'][before+'_to_'+after]=dict(actions=actions, pairwise_ranks=ranks, best_nonbest=partitions,
                signed_mean_player0=rational(parts['signed_mean']),signed_mean_actor=rational(sign*parts['signed_mean']),
                D=rational(parts['absolute_mean']),C=rational(parts['translation']),S=rational(parts['signed_cancellation']),E=rational(parts['centered_dispersion']),
                global_delta=rational((w-u)*parts['absolute_mean']), before_best_moves=best[before], after_best_moves=best[after])
        out[key]=entry
    return dict(schema='capture-independent-all16-action-attribution-v1',roots=out,root_count=16,
        endpoint_Q_entries=sum(len(v['legal_actions'])*3 for v in out.values()),horizon_pairs=['h8_to_h10','h10_to_h12','h8_to_h12'],
        all_root_probabilities_preserved=True,all_action_denominators_preserved=True,exact_rational=True,weights_applied=False,causal_claim=False)


class CaptureIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True, exist_ok=True)
        cls.base = WORK / ('retained-' + uuid.uuid4().hex); cls.base.mkdir()

    def worker(self, spec, path):
        path.mkdir(); spec = dict(spec, output=str(path / 'proof.json')); save(path / 'spec.json', spec)
        run_child([sys.executable, str(ROOT / 'tools/capture-audit.py'), str(path / 'spec.json')], path / 'worker.log', timeout=900)
        proof = read(path / 'proof.json'); self.assertTrue(proof['passed']); return dict(path=rel(path / 'proof.json'), **proof)

    def clone(self, source, name):
        target = self.base / name; shutil.copytree(source, target); return target

    def fixture_command(self, path):
        spec = self.base / (path.name + '-cli-spec.json')
        save(spec, dict(mode='fixture_cli', output=str(self.base / (path.name + '-cli-proof.json')),
            calls=[], clis=[dict(path='tools/investigate-capture-drift.py', argv=['run', '--fixture', '--cases', '2', '--run', str(path)])],
            input_names=['tools/capture-audit.py', 'tests/capture_independent_audit.py', rel(FREEZE),
                         'build/capture-drift-plan.json', 'build/capture-drift-implementation.json']))
        return [sys.executable, str(ROOT / 'tools/capture-audit.py'), str(spec)]

    def test_01_prior4923_exact_support_sources_reservations_and_models(self):
        from congklak import capture_drift as p
        baseline = read(BASELINE); previous_seal = read(ROOT / 'build/stratum-acceptance.json')
        self.assertEqual((baseline['inputs_count'], len(baseline['inputs'])), (4923, 4923)); current(baseline['inputs'])
        self.assertEqual(len(previous_seal['inputs']), 4920)
        self.assertTrue(set(previous_seal['inputs']) <= set(baseline['inputs']))
        self.assertEqual({k: baseline['inputs'][k] for k in previous_seal['inputs']}, previous_seal['inputs'])
        registry, rule, result, rootmap, stores = source_data(); overlay = read(ROOT / 'build/stratum-reservations.json')
        self.assertEqual(set(rootmap), set(registry['rows'])); self.assertEqual(len(rootmap), 16)
        self.assertEqual(set(overlay['selected_state_keys']), set(rootmap)); self.assertTrue(overlay['future_optimizer_reserved'])
        self.assertFalse(overlay['historical_roles_rewritten']); self.assertFalse(result['screen']['passes'])
        prefix = ROOT / 'data/stratum-weighting-v1/references-prefix.sqlite3'
        self.assertEqual(sha(prefix), rule['references_prefix_sha256'])
        self.assertEqual(sha(ROOT / 'data/stratum-weighting-v1/references-manifest.json'), rule['references_manifest_sha256'])
        self.assertEqual(rows(prefix, 'roots', 'state'), rootmap)
        origins = 0
        for key, row in registry['rows'].items():
            wrapper = rootmap[key]; self.assertEqual(raw(row['state']), key)
            self.assertEqual(wrapper['source_root_sha256'], digest(wrapper['root'])); self.assertEqual(wrapper['source_metadata_sha256'], digest(row))
            self.assertEqual(wrapper['state'], row['state']); self.assertEqual(wrapper['root'], row['source']['source_candidate'])
            self.assertEqual(row['source_cohort'], 'training_candidate'); self.assertEqual(row['group'], 'assessment')
            self.assertIn(row['family'], overlay['origin_families']); self.assertIn(row['component'], overlay['components'])
            self.assertIn(row['quotient'], overlay['eligible_quotients']); self.assertTrue(row['future_optimizer_reserved'])
            self.assertEqual(row['legal_mask'], sum(1 << a for a in literal.legal_actions(pos(row['state']))))
            for origin in row['all_origins']:
                source_db = ROOT / 'data' / origin['source_journal'] / 'work.sqlite3'
                with closing(qread(source_db)) as db:
                    text, checksum = db.execute('SELECT body,hash FROM families WHERE idx=?', (origin['source_family_index'],)).fetchone()
                family = json.loads(text); self.assertEqual(checksum, digest(family)); self.assertEqual(checksum, origin['source_family_sha256'])
                candidate = family['candidates'][origin['source_candidate_index']]
                self.assertEqual(candidate, origin['source_candidate']); self.assertEqual(digest(candidate), origin['source_candidate_sha256'])
                self.assertEqual(candidate['state'], row['state']); self.assertIn(candidate['family'], overlay['origin_families']); origins += 1
            for stage in ('h8', 'h10'):
                self.assertEqual(rows(prefix, stage + '_labels', 'state')[key], stores[stage + '_labels'][key])
                self.assertEqual(rule['projection']['rows'][key]['endpoints'][stage], stores[stage + '_labels'][key])
        capture = sorted(k for k, r in registry['rows'].items() if r['stratum'] == '25-49/capture')
        self.assertEqual(len(capture), 4); self.assertEqual(Counter(registry['rows'][k]['actor'] for k in capture), {0: 2, 1: 2})
        self.assertEqual([len(literal.legal_actions(pos(rootmap[k]['state']))) for k in capture], [4, 5, 5, 7])
        for key in capture:
            transitions = [literal.apply(pos(rootmap[key]['state']), a) for a in literal.legal_actions(pos(rootmap[key]['state']))]
            self.assertTrue(any(t.captured > 0 for t in transitions)); self.assertFalse(any(t.initial_passes or t.final_passes for t in transitions))
        control = read(ROOT / 'data/quality-train-v1/model.json')['model']; candidate = read(ROOT / 'data/ordinary-value-pilot-v1/model.json')['model']
        self.assertEqual(sha(ROOT / 'data/quality-train-v1/model.json'), 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558')
        changed = {name for name in control['weights'] if bits(control['weights'][name]) != bits(candidate['weights'][name])}
        self.assertEqual(changed, {'layers.4.weight', 'layers.4.bias'})
        data = p.admit(); self.assertEqual(data['capture_keys'], capture); self.assertEqual(data['rows'], rootmap)
        self.assertEqual(read(p.PANEL)['all_state_keys'], sorted(rootmap)); self.assertEqual(read(p.PLAN)['contract'], p.CONTRACT)
        self.assertFalse(p.CONTRACT['new_reservations']); self.assertTrue(p.CONTRACT['retrospective'])
        self.assertEqual((p.CONTRACT['new_teacher_searches'], p.CONTRACT['new_NN_forwards'], p.CONTRACT['new_games']), (0, 0, 0))
        self.assertFalse(p.DEFAULT.exists())
        with mock.patch.object(p, 'gate', side_effect=ValueError('Independent preflight unavailable')):
            with self.assertRaises(ValueError): p.run()
        self.assertFalse(p.DEFAULT.exists()); self.assertNotIn('torch', sys.modules)
        EVIDENCE['identity'] = dict(prior4923_current=True, previous4920_acceptance_preserved=True,
            all16_state_keys=sorted(rootmap), capture_state_keys=capture, exact_origin_occurrences_checked=origins,
            reservations_unchanged=True, historical_roles_preserved=True, policy_and_models_fixed=True,
            production_preflight_gate_refused_before_directory_creation=True, new_runtime_calls=0)

    def test_02_exact_attribution_signed_ranks_ties_and_cancellation(self):
        from congklak import capture_drift as p
        for deltas in ([-2, 6], [-4, 4], [3, 3], [-3, -3], [0, 0], [0, 1, -5, 8]):
            expected = drift_parts(deltas); actual = p.translation(deltas)
            for field, metric in (('mean_delta_player0', 'signed_mean'), ('D_mean_absolute', 'absolute_mean'),
                                  ('C_absolute_common_mean', 'translation'), ('S_signed_cancellation_remainder', 'signed_cancellation'),
                                  ('E_centered_dispersion', 'centered_dispersion')):
                self.assertEqual(frac(actual[field]), expected[metric])
            self.assertEqual(expected['absolute_mean'], expected['translation'] + expected['signed_cancellation'])
        counterexample = p.translation([-2, 6]); self.assertFalse(counterexample['D_equals_C_plus_E'])
        self.assertEqual((frac(counterexample['D_mean_absolute']), frac(counterexample['C_absolute_common_mean']), frac(counterexample['E_centered_dispersion'])), (F(4), F(2), F(4)))
        for raw_weights, drifts in (([F(2), F(1)], [F(3), F(1)]), ([F(1), F(1)], [F(8), F(2)]),
                                    ([F(1, 4), F(3, 4)], [F(5), F(5)]), ([F(2, 5), F(1, 3)], [F(2, 3), F(7, 8)])):
            expected = covariance(raw_weights, drifts); actual = p.covariance_identity(raw_weights, drifts)
            self.assertEqual(frac(actual['conditional_delta']), expected); self.assertEqual(frac(actual['global_delta']), expected / 8)
            self.assertEqual(frac(actual['uniform_covariance']) / frac(actual['raw_mean']), expected)
        probability = dict(uniform_probability=rational(F(1, 16)), weighted_probability=rational(F(3, 40)))
        for actor in (0, 1):
            sign = 1 if actor == 0 else -1; endpoints = {}
            for stage, scores in (('h8', [5, 5, 2]), ('h10', [4, 4, 1]), ('h12', [2, 3, 5])):
                q = {a: sign*v for a, v in zip((0, 2, 6), scores)}; best = max(sign*v for v in q.values())
                endpoints[stage] = dict(label=dict(action_scores=[dict(pit=a, value_player0=v) for a, v in q.items()],
                    target_player0=sign*best, best_moves=[a for a,v in q.items() if sign*v == best]))
            actual = p.action_analysis(dict(actor=actor), endpoints, probability); pair = actual['pairs']['h10_to_h12']
            self.assertEqual(actual['H10_best_moves'], [0, 2]); self.assertEqual(actual['H12_best_moves'], [6])
            self.assertEqual(actual['H10_best_H12_regrets'], {0: 3, 2: 2})
            self.assertEqual(pair['strict_ranking_reversals'], 2); self.assertEqual(pair['tie_changes'], 1)
            self.assertEqual(pair['action_groups']['H10_best']['actions'], [0, 2]); self.assertEqual(pair['action_groups']['H10_top2']['actions'], [0, 2])
            deltas = [sign*v for v in (-2, -1, 4)]
            self.assertEqual(frac(pair['mean_delta_player0']), F(sum(deltas), 3)); self.assertEqual(frac(pair['mean_delta_actor']), F(1, 3))
            self.assertEqual(frac(pair['D_mean_absolute']), F(7, 3))
            total = F()
            for a, delta in zip((0, 2, 6), deltas):
                item = actual['actions'][a]['pairs']['h10_to_h12']; self.assertEqual(item['delta_player0'], delta)
                self.assertEqual(item['delta_actor'], sign*delta)
                self.assertEqual(frac(item['global_weighted_minus_uniform']), (F(3, 40)-F(1, 16))*abs(delta)/3)
                total += frac(item['global_weighted_minus_uniform'])
                self.assertEqual([g['change_actor'] for g in item['H10_best_anchor_gap_changes']], [sign*(deltas[i]-delta) for i in (0, 1)])
            self.assertEqual(total, (F(3, 40)-F(1, 16))*F(7, 3))
        EVIDENCE['synthetic_exact_algebra'] = dict(translation_cases=6, covariance_cases=4, actor_cases=2,
            full_tie_regrets_and_boundary_ties=True, strict_reversal_vs_tie_change=True,
            D_equals_C_plus_S=True, D_equals_C_plus_E_counterexample=True, all_assertions_exact_Fractions=True)

    def test_03_real_case_pre_post_commit_kills_and_exact_resume(self):
        from congklak import capture_drift as p
        fresh = self.base / 'fresh'; run_child(self.fixture_command(fresh), self.base / 'fresh.log', timeout=900)
        reference, manifest = p.load_rows(fresh); self.assertEqual(len(reference), 2); self.assertEqual(manifest['cases'], 2)
        guard = read(self.base / 'fresh-cli-proof.json'); self.assertEqual(guard['native_calls'], 0)
        self.assertTrue(guard['fixture_case_commits_allowed']); self.assertIsNone(guard['new_case_commits'])
        records = []
        for event, count in (('capture_drift_precommit', 1), ('capture_drift_committed', 2)):
            run = self.base / ('resume-' + str(count)); probe = self.base / ('kill-' + str(count))
            command = self.fixture_command(run)
            env = dict(CAPTURE_DRIFT_BLOCK_EVENT=event, CAPTURE_DRIFT_BLOCK_STEP=str(count),
                       CAPTURE_DRIFT_MARKER=str(probe / 'declared-marker.json'))
            marker = kill(command, probe, event, count, env)
            backup(run / 'work.sqlite3', probe / 'validated-prefix.sqlite3')
            prefix = physical_rows(probe / 'validated-prefix.sqlite3', 'cases', 'idx')
            self.assertEqual(len(prefix), count); self.assertEqual(prefix, physical_rows(fresh / 'work.sqlite3', 'cases', 'idx')[:count])
            saved_meta = rows(probe / 'validated-prefix.sqlite3', 'meta', 'key'); self.assertNotIn('complete', saved_meta)
            run_child(command, probe / 'resume.log', timeout=900)
            actual, complete = p.load_rows(run); self.assertEqual(actual, reference); self.assertEqual(complete, manifest)
            self.assertEqual(prefix, physical_rows(run / 'work.sqlite3', 'cases', 'idx')[:count])
            guard = read(self.base / (run.name + '-cli-proof.json')); self.assertEqual(guard['native_calls'], 0)
            records.append(dict(event=event, actual_process_kill=True, committed_prefix=count, marker=marker,
                probe=rel(probe), run=rel(run), committed_prefix_bytes_preserved=True, fresh_semantics_equal=True,
                resumed_cases=len(actual), public_CLI_under_runtime_guards=True,
                producer_declared_marker_published=(probe / 'declared-marker.json').exists()))
        EVIDENCE['live_case_recovery'] = dict(tests=records, fresh=rel(fresh), fresh_manifest_sha256=sha(fresh / 'manifest.json'),
            raw_case_bodies_exact=True, physical_kills=2, unique_barriers=2, no_Native_NN_search_training=True,
            marker_policy='Helper marker and flushed worker event required; producer atomic marker optional if killed before rename.')

    def test_04_all_capture_reports_actions_histories_and_literal_PVs(self):
        from congklak import capture_drift as p
        registry, rule, previous, rootmap, stores = source_data(); cases, manifest = p.load_rows()
        selected = sorted(k for k, r in registry['rows'].items() if r['stratum'] == '25-49/capture')
        self.assertEqual([case['state_key'] for case in cases.values()], selected); self.assertEqual(manifest['cases'], 4)
        counts = Counter(); per_case = {}; replayed = {}
        for key in selected:
            row = rootmap[key]; case = next(c for c in cases.values() if c['state_key'] == key); overlap = {}; case_counts = Counter()
            self.assertEqual(case['state'], row['state']); self.assertEqual(case['source_root_sha256'], digest(row['root']))
            self.assertEqual(case['source_wrapper_sha256'], digest(row)); self.assertNotEqual(case['source_root_sha256'], case['source_wrapper_sha256'])
            for stage, depth in zip(STAGES, (8, 10, 12)):
                item = stores[stage + '_labels'][key]; attempt = stores[stage + '_attempts'][key]
                self.assertEqual(case['endpoint_item_sha256'][stage], digest(item)); self.assertEqual(case['source_attempt_sha256'][stage], digest(attempt))
                self.assertEqual(attempt['study_row_sha256'], digest(row)); self.assertEqual(item['root_sha256'], row['source_root_sha256'])
                self.assertEqual((attempt['stage'], attempt['requested_depth'], attempt['completed_depth'], attempt['status']), (stage, depth, depth, 'complete'))
                self.assertEqual([report['choice_depth'] for report in attempt['history']], list(range(1, depth + 1)))
                self.assertEqual(item['report'], attempt['history'][-1]); self.assertEqual(item['label']['choice_depth'], depth)
                self.assertEqual(item['label']['target_player0'], item['report']['depth_result']['value_player0'])
                self.assertEqual(item['label']['action_scores'], [dict(pit=c['pit'], value_player0=c['depth_result']['value_player0']) for c in item['report']['candidates']])
                self.assertEqual(item['label']['best_moves'], item['report']['best_moves'])
                case_counts['endpoint_Q_values'] += len(item['label']['action_scores'])
                for report in attempt['history']:
                    signature = dict(value=report['depth_result']['value_player0'], best=report['best_moves'],
                        q=[(c['pit'], c['depth_result']['value_player0']) for c in report['candidates']])
                    d = report['choice_depth']
                    if d in overlap: self.assertEqual(signature, overlap[d])
                    overlap[d] = signature
                    proof = report_literal(row, report); replayed[(key, stage, d)] = proof
                    case_counts['stage_reports'] += 1; case_counts['PV_references'] += 1 + len(proof['actions'])
                    case_counts['PV_choices'] += proof['root']['choices'] + sum(v['choices'] for v in proof['actions'].values())
            self.assertEqual(sorted(overlap), list(range(1, 13))); self.assertEqual(case_counts['stage_reports'], 30)
            per_case[key] = dict(case_counts); counts.update(case_counts)
        self.assertEqual(counts['stage_reports'], 120); self.assertEqual(counts['endpoint_Q_values'], 63); self.assertEqual(counts['PV_references'], 750)
        save(self.base / 'literal-reference-counts.json', dict(totals=dict(counts), cases=per_case))
        # Detailed per-choice checks are repeated against the forensic serialization
        # in method05, without treating duplicated stage/PV references as samples.
        EVIDENCE['production_literal_sources'] = dict(cases=4, all16_context_roots=len(rootmap), counts=dict(counts), per_case=per_case,
            all_legal_actions=True, all_consecutive_histories=True, shared_depth_Q_best_signatures_equal=True,
            raw_source_and_study_hashes_distinct=True, literal_seed_conservation=True, leaf_Hv1_store_margin_exact=True,
            no_native_calls=True, no_new_teacher_or_NN=True)

    def test_05_all16_exact_contributions_covariance_and_forensic_reports(self):
        from congklak import capture_drift as p
        registry, rule, previous, rootmap, stores = source_data(); cases, manifest = p.load_rows(); report = p.verify()
        ledger=all_action_ledger(registry,rule,rootmap,stores); ledger_path=self.base/'all16-action-attribution.json'; save(ledger_path,ledger)
        derived, cell_delta, total = independent_attribution(registry, rule, stores)
        ctx = report['context']; self.assertEqual((ctx['global_roots'], ctx['capture_roots'], ctx['uniform_global_denominator']), (16, 4, 16))
        self.assertEqual(set(ctx['roots']), set(rootmap)); self.assertFalse(ctx['probabilities_changed'])
        for before, after in (('h8', 'h10'), ('h10', 'h12'), ('h8', 'h12')):
            pair = before+'_to_'+after; all_drifts = {}; all_changes = {}
            for key, row in registry['rows'].items():
                q0 = label_q(stores[before+'_labels'][key], row['actor']); q1 = label_q(stores[after+'_labels'][key], row['actor'])
                drift = F(sum(abs(q1[a]-q0[a]) for a in q0), len(q0)); entry = ctx['roots'][key]['pairs'][pair]
                u = derived[key]['uniform']; w = derived[key]['weighted']
                self.assertEqual(entry, dict(drift=rational(drift), uniform_contribution=rational(u*drift),
                    weighted_contribution=rational(w*drift), global_delta=rational((w-u)*drift)))
                all_drifts[key] = drift; all_changes[key] = (w-u)*drift
                self.assertEqual(frac(ledger['roots'][key]['pairs'][pair]['global_delta']), (w-u)*drift)
            expected_total = sum(all_changes.values(), F()); selected = [k for k, r in registry['rows'].items() if r['stratum'] == '25-49/capture']
            self.assertEqual(frac(ctx['pairs'][pair]['global_delta']), expected_total)
            self.assertEqual(frac(ctx['pairs'][pair]['capture_global_delta']), sum((all_changes[k] for k in selected), F()))
            self.assertEqual(frac(ctx['pairs'][pair]['non_capture_global_delta']), sum((v for k,v in all_changes.items() if k not in selected), F()))
            for comparator in ('uniform', 'weighted'):
                self.assertEqual(frac(ctx['pairs'][pair][comparator+'_global_drift']), sum((derived[k][comparator]*all_drifts[k] for k in derived), F()))
            for cell, entry in ctx['cells'].items():
                keys = sorted(k for k, r in registry['rows'].items() if r['cell'] == cell); self.assertEqual(entry['state_keys'], keys)
                self.assertEqual(frac(entry['weighted_mass']), F(1, 8)); self.assertEqual(frac(entry['uniform_mass']), F(1, 8))
                delta = covariance([derived[k]['raw_weight'] for k in keys], [all_drifts[k] for k in keys])
                self.assertEqual(frac(entry['pairs'][pair]['conditional_delta']), delta)
                self.assertEqual(frac(entry['pairs'][pair]['global_delta']), delta/8)
            self.assertEqual(sum((frac(ctx['pairs'][pair]['by_stratum'][s]) for s in ctx['pairs'][pair]['by_stratum']), F()), expected_total)
        self.assertEqual(frac(report['summary']['global_Q_drift_delta']), total)
        self.assertEqual(frac(previous['metrics']['overall']['weighted']['action_Q_drift'])-frac(previous['metrics']['overall']['uniform']['action_Q_drift']), total)
        totals = Counter()
        for case in cases.values():
            key = case['state_key']; old = derived[key]; row = rootmap[key]; sign = 1 if old['actor'] == 0 else -1
            analysis = case['analysis']; forensic = case['forensics']; q = old['q']; legal = old['legal']
            self.assertEqual(set(map(int, analysis['actions'])), set(legal)); self.assertEqual(forensic['legal_actions'], legal)
            self.assertEqual(case['raw_rule'], rule['projection']['rows'][key]['raw_rule'])
            self.assertTrue(analysis['best_H10_H12_same']); self.assertTrue(all(v == 0 for v in analysis['H10_best_H12_regrets'].values()))
            for before, after in (('h8', 'h10'), ('h10', 'h12'), ('h8', 'h12')):
                pair = before+'_to_'+after; actual = analysis['pairs'][pair]
                delta = {a:q[after][a]-q[before][a] for a in legal}; parts = drift_parts(list(delta.values()))
                for field, expected in (('mean_delta_player0', parts['signed_mean']), ('mean_delta_actor', sign*parts['signed_mean']),
                    ('D_mean_absolute', parts['absolute_mean']), ('C_absolute_common_mean', parts['translation']),
                    ('S_signed_cancellation_remainder', parts['signed_cancellation']), ('E_centered_dispersion', parts['centered_dispersion'])):
                    self.assertEqual(frac(actual[field]), expected)
                ranking = []
                for i, a in enumerate(legal):
                    for b in legal[i+1:]:
                        x = sign*(q[before][a]-q[before][b]); y = sign*(q[after][a]-q[after][b])
                        ranking.append(dict(actions=[a,b], old_actor_difference=x, new_actor_difference=y, strict_reversal=x*y<0,
                            tie_changed=(x==0)!=(y==0), order_changed=((x>0)-(x<0))!=((y>0)-(y<0))))
                self.assertEqual(actual['pairwise_ranks'], ranking)
                for a in legal:
                    entry = analysis['actions'][str(a)]; change = entry['pairs'][pair]
                    self.assertEqual(entry['Q_player0'], {s:q[s][a] for s in STAGES}); self.assertEqual(entry['Q_actor'], {s:sign*q[s][a] for s in STAGES})
                    self.assertEqual((change['delta_player0'], change['delta_actor'], change['absolute_drift']), (delta[a], sign*delta[a], abs(delta[a])))
                    self.assertEqual(frac(change['global_weighted_minus_uniform']), (old['weighted']-old['uniform'])*abs(delta[a])/len(legal))
                    self.assertEqual(frac(change['centered_delta_player0']), delta[a]-parts['signed_mean'])
                self.assertEqual(sum((frac(analysis['actions'][str(a)]['pairs'][pair]['global_weighted_minus_uniform']) for a in legal), F()),
                    frac(ctx['roots'][key]['pairs'][pair]['global_delta']))
                h10_best = set(stores['h10_labels'][key]['label']['best_moves'])
                for group, selected in (('H10_best', h10_best), ('H10_nonbest', set(legal)-h10_best)):
                    g = actual['action_groups'][group]; self.assertEqual(g['actions'], sorted(selected))
                    self.assertEqual(frac(g['root_mean_absolute_contribution']), F(sum(abs(delta[a]) for a in selected), len(legal)))
                f_rank = forensic['ranking'][before+'_'+after]
                self.assertEqual([(x['pit_a'], x['pit_b'], x['gap_before_actor'], x['gap_after_actor'], x['strict_reversal'], x['tie_changed']) for x in f_rank['pairs']],
                    [(x['actions'][0], x['actions'][1], x['old_actor_difference'], x['new_actor_difference'], x['strict_reversal'], x['tie_changed']) for x in ranking])
                for gap in f_rank['before_best_anchor_gaps']:
                    b = gap['before_best_anchor']; a = gap['pit']; self.assertIn(b, stores[before+'_labels'][key]['label']['best_moves'])
                    self.assertEqual(gap['gap_change_actor'], sign*(delta[b]-delta[a]))
            refs = set(); referenced_choices = 0
            for line_id, line in forensic['lines'].items():
                trace = line['trace']; check_line_projection(self, trace)
                self.assertEqual(line_id, digest(dict(root_state=row['state'], pv=trace['pv'], depth=trace['requested_depth'], value_player0=trace['leaf']['saved_score_player0'])))
                self.assertTrue(line['source_references'])
                for source in line['source_references']:
                    stage, d, kind, pit = (source[k] for k in ('stage', 'actual_depth', 'kind', 'pit'))
                    identity = (stage,d,kind,pit); self.assertNotIn(identity, refs); refs.add(identity)
                    original = stores[stage+'_attempts'][key]['history'][d-1]
                    self.assertEqual(source['report_sha256'], digest(original))
                    selected = original if kind == 'root' else next(v for v in original['candidates'] if v['pit'] == pit)
                    self.assertEqual(trace['pv'], selected['pv']); self.assertEqual(trace['requested_depth'], d)
                    self.assertEqual(trace['leaf']['saved_score_player0'], selected['depth_result']['value_player0'])
                    referenced_choices += len(trace['pv'])
            self.assertEqual(len(refs), (len(legal)+1)*30); self.assertEqual(forensic['counts']['PV_references'], len(refs))
            self.assertEqual(forensic['counts']['deduplicated_literal_lines'], len(forensic['lines']))
            self.assertEqual(forensic['counts']['referenced_choices'], referenced_choices)
            self.assertEqual([r['depth'] for r in forensic['H12_job_depth_trajectory']], list(range(1,13)))
            for stage in STAGES:
                self.assertEqual(forensic['source_item_sha256'][stage], digest(stores[stage+'_labels'][key]))
                for entry in forensic['report_index'][stage]:
                    original = stores[stage+'_attempts'][key]['history'][entry['actual_depth']-1]
                    self.assertEqual(entry['report_sha256'], digest(original))
                    self.assertEqual(entry['signature']['action_scores'], [dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in original['candidates']])
            for before, after in (('h8','h10'), ('h10','h12'), ('h8','h12')):
                pair = before+'_'+after
                left = forensic['lines'][forensic['report_index'][before][-1]['root_line']]['trace']
                right = forensic['lines'][forensic['report_index'][after][-1]['root_line']]['trace']
                check_comparison(self, forensic['root_PV_comparisons'][pair], left, right)
                for a in legal:
                    action = forensic['actions'][str(a)]
                    left = forensic['lines'][action['scores'][before]['line_id']]['trace']; right = forensic['lines'][action['scores'][after]['line_id']]['trace']
                    check_comparison(self, action['PV_comparisons'][pair], left, right)
                    first = left['steps'][0]; self.assertEqual(action['immediate'], first)
                    self.assertEqual(action['immediate_capture'], first['captured_seeds']>0)
            totals.update(dict(PV_references=len(refs), unique_lines=len(forensic['lines']), referenced_choices=referenced_choices))
        self.assertEqual(totals['PV_references'], 750)
        self.assertFalse(report['weights_applied']); self.assertFalse(report['training_performed']); self.assertFalse(report['new_reservations'])
        EVIDENCE['exact_attribution_and_forensics'] = dict(context_roots=16, cells=8, horizon_pairs=3, all_exact_Fractions=True,
            global_delta=rational(total), capture_delta=report['summary']['capture_Q_drift_delta'], non_capture_delta=report['summary']['non_capture_Q_drift_delta'],
            per_cell_global_delta={k:rational(v) for k,v in cell_delta.items()}, all4_stable_best_sets=True, zero_capture_best_action_regret=True,
            literal_serialization_counts=dict(totals), source_references_exhaustive=True, no_duplicate_reference_sample_claim=True,
            PV_mechanics_are_observations_not_causal_evidence=True, new_weights_or_reservations=False)
        EVIDENCE['exact_attribution_and_forensics']['all16_action_evidence']=dict(path=rel(ledger_path),sha256=sha(ledger_path),
            schema=ledger['schema'],roots=16,endpoint_Q_entries=ledger['endpoint_Q_entries'],horizon_pairs=3,
            all_action_signed_deltas_contributions_best_nonbest_and_ranks=True)

    def test_06_rehashed_corruption_missing_export_and_closed_no_work(self):
        from congklak import capture_drift as p
        preflight = read(WORK / 'preflight.json'); fresh = ROOT / preflight['evidence']['live_case_recovery']['fresh']
        original_cases, original_manifest = p.load_rows(fresh); checks = []
        for kind in ('metric', 'literal', 'identity', 'complete'):
            target = self.clone(fresh, 'bad-' + kind); (target / 'manifest.json').unlink()
            if kind == 'metric':
                def corrupt(body): body['analysis']['pairs']['h10_to_h12']['root_value_delta_player0'] += 1
                mutate_row(target/'work.sqlite3', 'cases', 0, 'idx', corrupt)
            elif kind == 'literal':
                def corrupt(body):
                    line = next(iter(body['forensics']['lines'].values())); line['trace']['steps'][0]['captured_seeds'] += 1
                mutate_row(target/'work.sqlite3', 'cases', 0, 'idx', corrupt)
            elif kind == 'identity':
                mutate_row(target/'work.sqlite3', 'meta', 'identity', 'key', lambda body: body['settings'].update(fixture=1))
            else:
                mutate_row(target/'work.sqlite3', 'meta', 'complete', 'key', lambda body: body.update(case_sha256='0'*64))
            with self.assertRaises(ValueError): p.run(target, fixture=True)
            self.assertFalse((target/'manifest.json').exists()); checks.append(kind)
        repair = self.clone(fresh, 'missing-final-export'); expected_bytes = (repair/'manifest.json').read_bytes(); (repair/'manifest.json').unlink()
        p.run(repair, fixture=True); self.assertEqual((repair/'manifest.json').read_bytes(), expected_bytes)
        self.assertEqual(p.load_rows(repair)[0], original_cases)
        required = self.clone(p.SOURCE_DEFAULT, 'missing-required-prefix'); (required/'references-prefix.sqlite3').unlink()
        with mock.patch.object(p, 'SOURCE_DEFAULT', required):
            with self.assertRaises((ValueError, FileNotFoundError, sqlite3.OperationalError)): p.source()
        self.assertFalse((required/'references-prefix.sqlite3').exists())
        bad_path = ROOT/'data/capture-invalid-fixture-path-v1'
        with self.assertRaises(ValueError): p.run(bad_path, fixture=True)
        self.assertFalse(bad_path.exists())
        result = p.verify()
        bad = copy.deepcopy(result); bad['context']['pairs']['h10_to_h12']['global_delta'] = rational(frac(bad['context']['pairs']['h10_to_h12']['global_delta'])+1)
        with self.assertRaises(ValueError): p.validate_report(bad)
        bad = copy.deepcopy(result); bad['inputs']['congklak/capture_drift.py'] = '0'*64
        with self.assertRaises(ValueError): p.validate_report(bad)
        alternate = self.base/'missing-new-result.json'
        with mock.patch.object(p, 'OUTPUT', alternate):
            p.assess(); first = (sha(alternate), alternate.stat().st_size, alternate.stat().st_mtime_ns)
            p.assess(); p.verify(); self.assertEqual(first, (sha(alternate), alternate.stat().st_size, alternate.stat().st_mtime_ns))
        self.assertEqual(read(alternate), result)
        events = []; p.run(on_event=events.append)
        self.assertFalse(any(e['event'] in ('capture_drift_precommit', 'capture_drift_committed') for e in events))
        proof = self.worker(dict(watch_directories=[str(p.DEFAULT)], extra_files=[rel(p.OUTPUT)], watch_baseline=True,
            calls=[dict(module=p.__name__, function=name) for name in ('run', 'load_rows', 'build', 'assess', 'verify')],
            clis=[dict(path='tools/investigate-capture-drift.py', argv=['run']),
                  dict(path='tools/investigate-capture-drift.py', argv=['verify-cases']),
                  dict(path='tools/investigate-capture-drift.py', argv=['verify'])],
            input_names=[rel(p.PLAN),rel(p.IMPLEMENTATION),rel(p.SOURCE_FREEZE),rel(p.DEFAULT/'work.sqlite3'),
                         rel(p.DEFAULT/'manifest.json'),rel(p.OUTPUT),rel(FREEZE),'tests/capture_independent_audit.py']), self.base/'closed-worker')
        self.assertEqual(proof['native_calls'], 0); self.assertEqual(proof['new_case_commits'], 0)
        self.assertTrue(proof['all_bytes_sizes_mtimes_unchanged']); self.assertFalse(proof['torch_imported'])
        old = ast.parse((ROOT/'congklak/stratum_progress.py').read_text(encoding='utf-8-sig'))
        new = ast.parse((ROOT/'congklak/capture_progress.py').read_text(encoding='utf-8-sig'))
        functions = lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
        before, after = functions(old), functions(new)
        for name in ('save','event_message','runs','run_command'): self.assertEqual(before[name], after[name])
        current(read(BASELINE)['inputs']); self.assertNotIn('torch',sys.modules)
        EVIDENCE['completed_recovery_and_refusal'] = dict(rehashed_SQL_refusals=checks, rehashed_metric_and_source_report_refused=True,
            missing_final_export_repaired_from_valid_journal=True, required_prefix_not_rebuilt=True, invalid_fixture_path_refused=True,
            missing_new_result_rebuilt_stably=True, zero_new_case_commit_events=True, completed_worker=proof,
            prior4923_bytes_sizes_mtimes_preserved=True, Native_NN_training_calls_prohibited=True, logger4_AST_identity=True,
            old_logger_tests_not_recounted=True)


def source_inputs():
    from congklak import capture_drift as p
    mapping = dict(read(BASELINE)['inputs'])
    paths = [Path(__file__), ROOT/'tools/capture-audit.py', BASELINE, p.PANEL, p.PLAN, p.IMPLEMENTATION, p.SOURCE_FREEZE,
        ROOT/'build/capture-drift-author.json', ROOT/'build/capture-forensics-source-freeze.json', ROOT/'build/capture-forensics-author.json',
        ROOT/'congklak/capture_progress.py', ROOT/'tools/capture-dashboard.html', ROOT/'native/search.cpp', ROOT/'native/rules.cpp']
    paths += [ROOT/n for n in (*p.SOURCES,*p.DEPENDENCIES)]
    assert all(path.is_file() for path in paths), 'Required source/owner proof missing before audit freeze'
    mapping.update(pins(paths)); return dict(sorted(mapping.items()))


def collect():
    frozen = read(FREEZE); current(frozen['inputs']); groups=[]; inputs={}; evidence={}; covered=[]
    for name, methods in GROUPS.items():
        path=WORK/(name+'.json'); body=read(path); checksum=sha(path)
        assert body['schema']=='capture-independent-group-v1' and body['group']==name and body['passed'] is True
        assert body['tests_run']==len(methods) and body['errors']==body['failures']==body['skips']==0 and body['covered_methods']==methods
        current(body['inputs'])
        for k,v in body['inputs'].items():
            assert k not in inputs or inputs[k]==v, 'Input conflict: '+k
            inputs[k]=v
        assert not set(evidence)&set(body['evidence']); evidence.update(body['evidence']); covered+=methods
        inputs[rel(path)]=checksum
        groups.append(dict(path=rel(path),sha256=checksum,tests_run=len(methods),covered_methods=methods,
                           executed_source=dict(path=rel(__file__),sha256=sha(__file__))))
    assert len(covered)==len(set(covered))==6
    body=dict(schema='capture-independent-audit-v1',passed=True,tests_run=6,errors=0,failures=0,skips=0,
        covered_methods=covered,covered_checks=['__main__.CaptureIndependentAudit.'+m for m in covered],
        groups=groups,inputs=dict(sorted(inputs.items())),evidence=evidence)
    save(ROOT/'build/capture-audit.json',body); return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('freeze',*GROUPS,'collect'));args=parser.parse_args()
    if args.operation=='freeze':
        save(FREEZE,dict(schema='capture-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=read(FREEZE);assert frozen['groups']==GROUPS;current(frozen['inputs'])
    output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable group exists'
    spec=importlib.util.spec_from_file_location('_capture_audit_runtime_guards',ROOT/'tools/capture-audit.py')
    guarded=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guarded;spec.loader.exec_module(guarded)
    started=time.monotonic();methods=GROUPS[args.operation]
    with guarded.engine_guards():
        result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(CaptureIndependentAudit(name) for name in methods))
    from congklak import capture_drift as p
    inputs={**frozen['inputs'],rel(FREEZE):sha(FREEZE)}
    retained=getattr(CaptureIndependentAudit,'base',None);captured=files(retained) if retained else []
    if args.operation in ('production','final'):
        captured+=files(p.DEFAULT)+[p.OUTPUT,WORK/'preflight.json']
    if args.operation=='final':
        captured += [WORK/'production.json']
        captured += [p for p in (ROOT/'tools').glob('*capture*.py') if p.name!='capture-audit.py']
    inputs.update(pins(captured))
    body=dict(schema='capture-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        covered_checks=['__main__.CaptureIndependentAudit.'+m for m in methods],duration_seconds=time.monotonic()-started,
        retained_directory=rel(retained) if retained else None,inputs=dict(sorted(inputs.items())),evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=traceback) for test,traceback in result.errors+result.failures])
    save(output,body);print(canonical(dict(event='capture_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True)
    return int(not body['passed'])


if __name__=='__main__':raise SystemExit(main())
