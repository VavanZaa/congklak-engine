"""Fixed-checkpoint capture diagnostics from existing cached features and labels.

This module deliberately imports only the standard library.  Its bounded units
are executed and journaled by the enclosing diagnostic coordinator.  It never
extracts features, opens an engine, chooses a checkpoint, or optimizes a model.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'value-capture-diagnostic-v1'
CONTROL = 'data/quality-train-v1/model.json'
CANDIDATE = 'data/value-training-v1/training/model.json'
CONTROL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
CANDIDATE_SHA = 'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319'
CACHE = 'data/value-training-v1/cache/'
CASES = 'build/value-run-v1/cases/'
FIT = 'build/value-run-v1/cases-fit-v2.json'
ROLES = {'train': ('train.json', 512), 'fit_check': ('fit-check.json', 128)}
CHUNK = 32


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(body):
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def _pins(names):
    return {name: sha(ROOT / name) for name in sorted(set(names))}


def current(pins):
    for name, checksum in pins.items():
        p = (ROOT / name).resolve()
        need(p.is_relative_to(ROOT) and p.is_file() and sha(p) == checksum,
             'Diagnostic input changed: ' + name)


def counters(rows=0, records=0):
    return dict(cached_rows=rows, saved_case_records=records,
                frozen_base_forwards=rows, sidecar_forwards=rows,
                evaluated_value_predictions=2 * rows, policy_forwards=rows,
                feature_simulations=0, optimizer_steps=0, optimizer_draws=0,
                new_teacher_labels=0, new_games=0, new_searches=0)


def plan():
    """Return deterministic JSON tasks and the entire direct input certificate."""
    names = [CONTROL, CANDIDATE, FIT, CACHE + 'train.json', CACHE + 'fit-check.json',
             CACHE + 'manifest.json', CACHE + 'certificate.json',
             'congklak/value_capture_diagnostic.py',
             'tests/test_value_capture_diagnostic.py', CASES + 'tasks.json',
             CASES + 'catalog.json', CASES + 'report.json', CASES + 'certificate.json']
    need(sha(ROOT / CONTROL) == CONTROL_SHA and sha(ROOT / CANDIDATE) == CANDIDATE_SHA,
         'Only the frozen control and final400 candidate are admitted')
    manifest = read(CACHE + 'manifest.json')
    tasks = []
    for role, (filename, count) in ROLES.items():
        meta = manifest['role_files'][role]
        need(meta['filename'] == filename and meta['count'] == count and
             sha(ROOT / CACHE / filename) == meta['sha256'], 'Cache role binding differs')
        for start in range(0, count, CHUNK):
            tasks.append(dict(id=f'{role}-{start:04d}', kind='cached_rows',
                              role=role, start=start, stop=min(start + CHUNK, count)))
    catalog = read(CASES + 'catalog.json')
    case_tasks = read(CASES + 'tasks.json')
    need(len(case_tasks) == catalog['count'] == len(catalog['records']) == 650,
         'The saved case corpus is incomplete')
    records = []
    for index, (entry, task) in enumerate(zip(catalog['records'], case_tasks)):
        need(entry['index'] == index, 'Saved case catalog order differs')
        if task['root']['cohort'] == 'fit_H8' and task['engine'] in ('control', 'candidate'):
            name = CASES + 'records/' + entry['name']
            need(sha(ROOT / name) == entry['sha256'], 'Saved case bytes differ')
            names.append(name)
            records.append(dict(index=index, path=name, sha256=entry['sha256']))
    need(len(records) == 192, 'Saved neural fit-case coverage differs')
    tasks.append(dict(id='saved-fit-searches', kind='saved_case_records', records=records))
    pins = _pins(names)
    return dict(schema=SCHEMA + '-plan', inputs=pins, tasks=tasks,
                task_sha256=digest(tasks), policy_fixed=True, models_fixed=True,
                selected_checkpoint='final400/predeclared/no-selection',
                counters=counters(640, 192))


def _models():
    need(sha(ROOT / CONTROL) == CONTROL_SHA and sha(ROOT / CANDIDATE) == CANDIDATE_SHA,
         'Frozen model bytes changed')
    control, candidate = read(CONTROL), read(CANDIDATE)
    need(candidate['model']['base_export'] == control,
         'The candidate changed a frozen base tensor or policy')
    return control['model']['weights'], candidate['model']['branch_weights']


def forward(x, base, branch):
    """Literal F64 arithmetic over already FP32-rounded cache vectors."""
    h = x
    for prefix in ('layers.0', 'layers.2'):
        h = [max(0., sum(a * v for a, v in zip(w, h)) + b)
             for w, b in zip(base[prefix + '.weight'], base[prefix + '.bias'])]
    u = sum(a * v for a, v in zip(base['layers.4.weight'][0], h)) + base['layers.4.bias'][0]
    logits = [sum(a * v for a, v in zip(w, h)) + b
              for w, b in zip(base['policy.weight'], base['policy.bias'])]
    activations = [max(0., sum(a * v for a, v in zip(w, x)) + b)
                   for w, b in zip(branch['A'], branch['b'])]
    contributions = [v * a for v, a in zip(branch['v'][0], activations)]
    correction = sum(contributions) + branch['c'][0]
    return dict(base_logit=u, correction_logit=correction,
                branch_bias=branch['c'][0], hidden_activations=activations,
                hidden_contributions=contributions, policy_logits_sha256=digest(logits),
                control_z=math.tanh(u), candidate_z=math.tanh(u + correction))


def huber(error):
    a = abs(error)
    return .5 * error * error if a <= .1 else .1 * (a - .05)


def _nearest(row, candidates):
    ranked = sorted((sum((a - b) ** 2 for a, b in zip(row['x'], other['x'])),
                     other['state_key'], other) for other in candidates)
    distance, key, other = ranked[0]
    return dict(state_key=key, cell=other['cell'], component=other['component'],
                feature_l2=math.sqrt(distance), target_z=other['target_z'],
                target_z_difference=row['target_z'] - other['target_z'],
                exact_feature_match=distance == 0., actor=other['actor'])


def describe(row, base, branch, training_rows):
    """One cached root; all target and mechanism checks consume saved values."""
    x, state, actor, source = row['x'], row['state'], row['actor'], row['target_source']
    active = sum(map(sum, state['pits']))
    sign = 1 if actor == 0 else -1
    need(len(x) == 127 and all(math.isfinite(v) for v in x), 'Bad feature vector')
    need(actor == state['controller'] and active * sign == row['r'] and
         state['stores'][0] - state['stores'][1] == row['D'] and 25 <= active <= 74,
         'Cached score perspective or branch band differs')
    need(bytes(state['pits'][0] + [state['stores'][0]] + state['pits'][1] +
               [state['stores'][1], actor]).hex() == row['state_key'], 'Cached state key differs')
    q = {entry['pit']: entry['value_player0'] for entry in source['action_q']}
    legal = [pit for pit, count in enumerate(state['pits'][actor]) if count]
    need(set(q) == set(legal) and len(q) == len(source['action_q']), 'Incomplete legal H8 Q values')
    best = (max if actor == 0 else min)(q.values())
    need(best == row['target_player0'] and source['best_moves'] == [a for a in legal if q[a] == best]
         and source['requested_depth'] == source['completed_depth'] == 8
         and source['target_is_solved_truth'] is False,
         'Mixed, capped, substituted, or differently oriented H8 target')
    need(row['target_z'] == (best - row['D']) / row['r'], 'Normalized target differs')
    f = forward(x, base, branch)
    predictions = {}
    for profile in ('control', 'candidate'):
        z = f[profile + '_z']; value = row['D'] + row['r'] * z
        error = value - best; normalized_error = z - row['target_z']
        predictions[profile] = dict(value_player0=value, seed_error=error,
                                    actor_seed_error=sign * error,
                                    normalized_error=normalized_error, huber=huber(normalized_error))
    c, n = predictions['control'], predictions['candidate']
    delta = abs(n['seed_error']) - abs(c['seed_error'])
    shift = n['value_player0'] - c['value_player0']
    own = [x[43 + 6 * a:49 + 6 * a] for a in range(7)]
    opponent = [x[85 + 6 * a:91 + 6 * a] for a in range(7)]
    for features in own + opponent:
        need(all(features[index] in (0., 1.) for index in (3, 4, 5)), 'Cached mechanism flag differs')
    capture = [a for a in legal if own[a][3] == 1.]
    best_actions = source['best_moves']
    ordered = sorted((sign * value for value in q.values()), reverse=True)
    gap = ordered[0] - ordered[1]
    toward = shift * c['seed_error'] < 0
    away = shift * c['seed_error'] > 0
    out = dict(state_key=row['state_key'], role=row['reserved_role'], family=row['family'],
               component=row['component'], actor=actor, cell=row['cell'], stratum=row['stratum'],
               active=active, D=row['D'], r=row['r'], target_z=row['target_z'],
               target_player0=best, target_source=source, feature_source=row['feature_source'],
               feature_sha256=digest(x), predictions=predictions, forward=f,
               delta_seed_absolute_error=delta, delta_normalized_huber=n['huber'] - c['huber'],
               value_shift_player0=shift, value_shift_actor=sign * shift,
               correction_toward_original_target=toward,
               correction_away_from_original_target=away,
               crossed_target=c['seed_error'] * n['seed_error'] < 0,
               overshoot_regression=toward and c['seed_error'] * n['seed_error'] < 0 and delta > 0,
               regression=delta > 0, improved=delta < 0,
               current_logit_loss_gradient=max(-.1, min(.1, n['normalized_error'])) *
                                          (1 - f['candidate_z'] ** 2),
               teacher_action_gap_actor=gap,
               teacher_action_spread=max(q.values()) - min(q.values()),
               best_actions=best_actions, own_capture_actions=capture,
               best_capture_actions=[a for a in best_actions if a in capture],
               all_best_actions_capture=all(a in capture for a in best_actions),
               own_max_store_gain_cached=98 * max(v[0] for v in own),
               own_max_relay_pickups_cached=98 * max(v[2] for v in own),
               opponent_max_relay_pickups_cached=98 * max(v[2] for v in opponent),
               own_any_terminal=any(v[5] for v in own),
               feature_distance_interpretation='Unweighted L2 in existing 127-input units; descriptive only.')
    if row['reserved_role'] == 'fit_check':
        out['nearest_train_all'] = _nearest(row, training_rows)
        out['nearest_train_same_cell'] = _nearest(row, [r for r in training_rows if r['cell'] == row['cell']])
    return out


def unit(task):
    """Compute a single deterministic unit, with no writes or engine calls."""
    approved = plan()
    need(task in approved['tasks'], 'Unknown or modified capture diagnostic task')
    current(approved['inputs'])
    if task['kind'] == 'cached_rows':
        base, branch = _models()
        rows = read(CACHE + ROLES[task['role']][0])['rows']
        training_rows = read(CACHE + 'train.json')['rows'] if task['role'] == 'fit_check' else []
        described = [describe(row, base, branch, training_rows)
                     for row in rows[task['start']:task['stop']]]
        body = dict(rows=described, counters=counters(len(described)))
    else:
        records = []
        for entry in task['records']:
            record = read(entry['path']); t = record['task']; root = t['root']
            need(record['index'] == entry['index'] and root['cohort'] == 'fit_H8' and
                 t['engine'] in ('control', 'candidate'), 'Saved case membership differs')
            chosen = record['answer']['decision']['chosen']
            q = {a['pit']: a['value_player0'] for a in root['reference']['action_scores']}
            actor = root['state']['controller']; sign = 1 if actor == 0 else -1
            need(chosen in q, 'Saved search has no legal recommendation')
            regret = sign * (root['reference']['target_player0'] - q[chosen])
            need(regret == record['score']['regret'] and regret >= 0,
                 'Saved action regret or player perspective differs')
            records.append(dict(state_key=root['state_key'], stratum=root['stratum'], actor=actor,
                                engine=t['engine'], mode=t['mode'], ms=t['ms'], chosen=chosen,
                                regret=regret, source=entry, decision_sha256=digest(record)))
        body = dict(records=records, counters=counters(records=len(records)))
    return dict(schema=SCHEMA + '-unit', id=task['id'], task=task,
                plan_sha256=digest(approved), inputs=approved['inputs'], **body)


def _metrics(rows):
    need(rows, 'Empty metric group')
    result = dict(roots=len(rows))
    for profile in ('control', 'candidate'):
        p = [r['predictions'][profile] for r in rows]
        result[profile] = dict(seed_mae=statistics.mean(abs(v['seed_error']) for v in p),
                               seed_rmse=math.sqrt(statistics.mean(v['seed_error'] ** 2 for v in p)),
                               seed_bias_player0=statistics.mean(v['seed_error'] for v in p),
                               seed_bias_actor=statistics.mean(v['actor_seed_error'] for v in p),
                               normalized_huber=statistics.mean(v['huber'] for v in p),
                               normalized_mae=statistics.mean(abs(v['normalized_error']) for v in p))
    result.update(delta_seed_mae=result['candidate']['seed_mae'] - result['control']['seed_mae'],
                  delta_huber=result['candidate']['normalized_huber'] - result['control']['normalized_huber'],
                  worsening_roots=sum(r['regression'] for r in rows),
                  improving_roots=sum(r['improved'] for r in rows),
                  adverse_direction_regressions=sum(r['regression'] and r['correction_away_from_original_target'] for r in rows),
                  overshoot_regressions=sum(r['overshoot_regression'] for r in rows),
                  mean_actor_value_shift=statistics.mean(r['value_shift_actor'] for r in rows),
                  mean_logit_correction=statistics.mean(r['forward']['correction_logit'] for r in rows),
                  positive_logit_corrections=sum(r['forward']['correction_logit'] > 0 for r in rows),
                  sum_delta_absolute_error=sum(r['delta_seed_absolute_error'] for r in rows),
                  mean_hidden_contributions=[statistics.mean(r['forward']['hidden_contributions'][j] for r in rows) for j in range(16)])
    return result


def summarize(results):
    """Assemble all units and clearly separate scalar fit from saved move quality."""
    p = plan(); expected = p['tasks']
    need(len(results) == len(expected) and [r['id'] for r in results] == [t['id'] for t in expected],
         'Missing, reordered or duplicated capture units')
    for result, task in zip(results, expected):
        need(result['schema'] == SCHEMA + '-unit' and result['task'] == task and
             result['plan_sha256'] == digest(p) and result['inputs'] == p['inputs'], 'Capture unit identity differs')
    current(p['inputs'])
    rows = [r for result in results for r in result.get('rows', [])]
    cases = [r for result in results for r in result.get('records', [])]
    need(Counter(r['role'] for r in rows) == {'train': 512, 'fit_check': 128} and
         len({r['state_key'] for r in rows}) == 640 and len(cases) == 192,
         'Capture diagnostic corpus coverage differs')
    groups = defaultdict(list)
    for row in rows:
        role = row['role']
        for group in ('all', row['stratum'], row['cell'], 'actor/' + str(row['actor'])):
            groups[role + '/' + group].append(row)
        if row['stratum'].endswith('/capture'):
            groups[role + '/capture_best_action/' + ('capture' if row['best_capture_actions'] else 'noncapture')].append(row)
            groups[role + '/capture_gap/' + ('0-2' if row['teacher_action_gap_actor'] <= 2 else '3+')].append(row)
    metrics = {key: _metrics(value) for key, value in sorted(groups.items())}
    fit_saved = {r['state_key']: r for r in read(FIT)['rows']}
    maximum_fit_difference = 0.
    for row in rows:
        if row['role'] != 'fit_check':
            continue
        saved = fit_saved[row['state_key']]
        need(saved['target_source'] == row['target_source'], 'Saved fit target was changed')
        for profile in ('control', 'candidate'):
            maximum_fit_difference = max(maximum_fit_difference, abs(
                saved['predictions'][profile]['value_player0'] - row['predictions'][profile]['value_player0']))
    # Saved report uses fresh literal F64 features; the training cache is FP32.
    need(maximum_fit_difference <= 1e-4, 'Cached FP32 diagnostic no longer agrees with prior literal fit')
    casegroups = defaultdict(list)
    for record in cases:
        key = f"{record['mode']}/{record['ms']}/{record['stratum']}/{record['engine']}"
        casegroups[key].append(record)
    case_metrics = {key: dict(roots=len(group), mean_regret=statistics.mean(r['regret'] for r in group),
                              optimal=sum(r['regret'] == 0 for r in group))
                    for key, group in sorted(casegroups.items())}
    bycase = defaultdict(dict)
    for record in cases:
        bycase[(record['state_key'], record['mode'], record['ms'])][record['engine']] = record
    associations = []
    bykey = {row['state_key']: row for row in rows if row['role'] == 'fit_check'}
    for (key, mode, ms), profiles in sorted(bycase.items()):
        need(set(profiles) == {'control', 'candidate'}, 'Saved search pair incomplete')
        row = bykey[key]
        associations.append(dict(state_key=key, mode=mode, ms=ms, stratum=row['stratum'],
                                 root_delta_absolute_error=row['delta_seed_absolute_error'],
                                 control_regret=profiles['control']['regret'],
                                 candidate_regret=profiles['candidate']['regret'],
                                 delta_regret=profiles['candidate']['regret'] - profiles['control']['regret'],
                                 action_changed=profiles['candidate']['chosen'] != profiles['control']['chosen']))
    capture_fit = [r for r in rows if r['role'] == 'fit_check' and r['stratum'].endswith('/capture')]
    ordered = sorted(capture_fit, key=lambda r: (-r['delta_seed_absolute_error'], r['state_key']))
    top = ordered[:5]
    coverage = dict(fit_capture_roots=len(capture_fit),
                    zero_feature_distance_to_train=sum(r['nearest_train_all']['exact_feature_match'] for r in capture_fit),
                    nearest_train_same_cell_distance_min=min(r['nearest_train_same_cell']['feature_l2'] for r in capture_fit),
                    nearest_train_same_cell_distance_median=statistics.median(r['nearest_train_same_cell']['feature_l2'] for r in capture_fit),
                    nearest_train_same_cell_distance_max=max(r['nearest_train_same_cell']['feature_l2'] for r in capture_fit))
    return dict(schema=SCHEMA + '-report', passed=True, inputs=p['inputs'], plan_sha256=digest(p),
                counters=counters(640, 192), policy_fixed=True, models_fixed=True,
                fit_check_is_unseen=False, training_authorized=False, candidate_quarantined=True,
                cached_fp32_vs_saved_f64_max_seed_difference=maximum_fit_difference,
                cached_fp32_vs_saved_f64_seed_tolerance=1e-4,
                metrics=metrics, rows=rows, saved_search_metrics=case_metrics,
                saved_search_root_associations=associations, saved_search_records=cases,
                capture_coverage=coverage,
                worst_five_fit_capture=dict(state_keys=[r['state_key'] for r in top],
                    sum_delta_absolute_error=sum(r['delta_seed_absolute_error'] for r in top),
                    all32_sum_delta_absolute_error=sum(r['delta_seed_absolute_error'] for r in capture_fit)),
                interpretation=[
                    'The fixed sidecar is the arithmetic source of prediction changes; no policy or feature change is involved.',
                    'Per-actor differences are descriptive: all 127 inputs and the learned residual are actor-relative; actor is not an explicit input.',
                    'Train/fit divergence and local feature distances identify possible coverage or fitting failures; they do not prove their cause.',
                    'Root scalar error and action regret measure different objects. Saved search associations do not reconstruct internal leaf errors.',
                    'All labels are one complete H_v1 depth-8 finite-horizon target, not solved outcomes; this audit cannot establish target truth.',
                    'Fit roots were excluded from these 400 updates but have historical ancestry exposure and repeated diagnostic use; no unseen-validation claim.'],
                proposed_next_experiment=[
                    'Before any training, predeclare a fresh ordinary/capture ownership-safe evaluation panel with coverage of actor-relative target signs and relay/capture mechanisms.',
                    'Keep the frozen policy and the single normalized H8 target. Test a small regularized value-only fit against the current final400 and control with a predeclared step budget and no checkpoint selection.',
                    'Track each capture cell and individual high-regret roots, normalized Huber plus seed MAE, both baseline equal-time games, and the unchanged exact endgame guard.',
                    'Use the current 32 capture fit roots only as a consumed diagnostic regression suite; do not turn their error ranks into training weights or release criteria.'])
