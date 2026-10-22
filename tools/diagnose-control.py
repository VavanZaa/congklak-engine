"""Read-only diagnosis of the retained learned control, using existing evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, pilot
from congklak.native import Native, flatten
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.oracle import Oracle, DEFAULT_PATH as ORACLE_PATH

OUTPUT = ROOT / 'build/control-diagnosis.json'
DOCUMENT = ROOT / 'docs/CONTROL_DIAGNOSIS.md'
INPUTS = {}


def pin(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), str(path)
    key = path.relative_to(ROOT).as_posix()
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[key] = value
    return value


def load(path):
    pin(ROOT / path)
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def emit(stage, **values):
    print(config.canonical(dict(event='control_diagnosis_progress', stage=stage, **values)), flush=True)


def database(path):
    pin(ROOT / path)
    return sqlite3.connect((ROOT / path).as_uri() + '?mode=ro', uri=True)


def source(row):
    return 'constructed' if row['family'].startswith('constructed-') else 'recorded_history'


def summarize(rows):
    return dict(rows=len(rows), mae_seeds=statistics.mean(r['absolute_error'] for r in rows),
                signed_actor_error=statistics.mean(r['signed_actor_error'] for r in rows),
                policy_teacher_regret=statistics.mean(r['policy_regret'] for r in rows),
                policy_optimal_accuracy=statistics.mean(r['policy_regret'] == 0 for r in rows)) if rows else dict(rows=0)


def dataset():
    manifests = {name: load('build/' + name + '-data.json') for name in ('quality', 'value')}
    rows = {}
    labels = {}
    with closing(database('data/deeper-9000-v2/work.sqlite3')) as db:
        for key, text, digest in db.execute("SELECT state,body,hash FROM roots WHERE json_extract(body,'$.split') IN ('train','development') ORDER BY state"):
            row = json.loads(text)
            assert config.digest(row) == digest and row['state_key'] == key
            rows[key] = row
        for key, text, digest in db.execute("SELECT l.state,l.body,l.hash FROM labels l JOIN roots r ON l.state=r.state WHERE json_extract(r.body,'$.split') IN ('train','development') ORDER BY l.state"):
            item = json.loads(text)
            assert config.digest(item) == digest and item['root_sha256'] == config.digest(rows[key])
            assert item['label']['choice_depth'] == 8 and item['label']['kind'] == 'finite_choice_horizon'
            labels[key] = item['label']
    assert len(rows) == len(labels) == 9000
    coverage = {}
    for name, manifest in manifests.items():
        admitted = [rows[key] for key in manifest['train_state_keys']]
        assert len(admitted) == manifest['train_rows'] and all(r['split'] == 'train' for r in admitted)
        groups = defaultdict(list)
        for index, row in enumerate(admitted):
            groups[row['stratum']].append(index)
        strata = sorted(groups)
        rng = random.Random(20261015 if name == 'quality' else 20261018)
        draws = Counter(rng.choice(groups[rng.choice(strata)]) for _ in range(384000))
        stats = {}
        for stratum in strata:
            ids = groups[stratum]
            group = [admitted[i] for i in ids]
            stats[stratum] = dict(rows=len(group), families=len({r['family'] for r in group}),
                recorded_history_rows=sum(source(r) == 'recorded_history' for r in group),
                constructed_rows=sum(source(r) == 'constructed' for r in group),
                actual_draws=sum(draws[i] for i in ids), max_draws_single_root=max(draws[i] for i in ids))
        rare = [i for ids in groups.values() if len(ids) <= 5 for i in ids]
        coverage[name] = dict(rows=len(admitted), families=len({r['family'] for r in admitted}), strata=stats,
            unique_roots_drawn=len(draws), total_draws=sum(draws.values()), rare_le5_rows=len(rare),
            rare_le5_draws=sum(draws[i] for i in rare),
            sampler_scope='Reconstructed frozen uniform-stratum schedule from accepted seed and admitted row order; no new optimizer execution')
    development_labels = {k: v for k, v in labels.items() if rows[k]['split'] == 'development'}
    assert len(development_labels) == 1000
    return rows, development_labels, coverage, labels


def development(rows, labels):
    metrics = []
    control_path = ROOT / 'data/quality-train-v1/model.json'
    refit_path = ROOT / 'data/value-train-v1/model.json'
    pin(control_path); pin(refit_path)
    control = Model(control_path); refit = Model(refit_path)
    frozen = set(control.weights) - {'layers.4.weight', 'layers.4.bias'}
    assert all(config.canonical(control.weights[k]) == config.canonical(refit.weights[k]) for k in frozen)
    with CpuBatchEvaluator(control, cache_capacity=0, max_batch=1) as ce, CpuBatchEvaluator(refit, cache_capacity=0, max_batch=1) as re:
        for index, key in enumerate(sorted(labels)):
            row = rows[key]; p = row['state']; label = labels[key]
            a = ce.evaluate_many([p])[0]; b = re.evaluate_many([p])[0]
            assert a['policy_logits'] == b['policy_logits']
            legal = [q['pit'] for q in label['action_scores']]
            chosen = max(legal, key=lambda i: (a['policy_logits'][i], -i))
            sign = 1 if p['controller'] == 0 else -1
            q = {x['pit']: x['value_player0'] for x in label['action_scores']}
            y = label['target_player0']
            metrics.append(dict(state_key=key, family=row['family'], stratum=row['stratum'], source=source(row),
                state=p, teacher_player0=y, control_player0=a['value_player0'], refit_player0=b['value_player0'],
                absolute_error=abs(a['value_player0'] - y), refit_absolute_error=abs(b['value_player0'] - y),
                signed_actor_error=sign * (a['value_player0'] - y),
                policy_chosen=chosen, policy_regret=sign * (y - q[chosen])))
            if (index + 1) % 250 == 0:
                emit('development', completed=index + 1, total=1000)
    result = {}
    for stratum in sorted({r['stratum'] for r in metrics}):
        group = [r for r in metrics if r['stratum'] == stratum]
        result[stratum] = {**summarize(group), 'by_source': {s: summarize([r for r in group if r['source'] == s]) for s in ('recorded_history', 'constructed')}}
    expected = load('data/value-train-v1/result.json')
    for key, field in [('initial_development_this_invocation', 'absolute_error'), ('development', 'refit_absolute_error')]:
        m = expected[key]
        reference = sum(x['rows'] * x['mae_seeds'] for x in m['strata'].values()) / m['rows']
        assert abs(statistics.mean(x[field] for x in metrics) - reference) < 0.0001
    worst = sorted(metrics, key=lambda r: (-r['absolute_error'], r['state_key']))[:12]
    return dict(control=summarize(metrics), strata=result, worst_examples=worst,
                native_policy_equal_on_all1000=True, all_frozen_export_tensors_equal=True)


def case_diagnosis():
    records = []; endgame = []
    for campaign, profile, stem in [('quality', 'refined_search', 'quality'), ('value', 'policy_control', 'value')]:
        plan = load('build/' + stem + '-plan.json')
        with closing(database('data/' + stem + '-benchmark-v1/work.sqlite3')) as db:
            for index, text, digest in db.execute('SELECT idx,body,hash FROM cases ORDER BY idx'):
                body = json.loads(text); assert config.digest(body) == digest
                case = plan['cases'][index]; ref = case['reference']; p = case['state']
                for ms in (25, 100):
                    row = body['measurements'][str(ms)][profile]
                    d = row['decision']['decision']; report = d['report']; chosen = d['chosen']
                    assert chosen is not None and d['usable_decision']
                    sign = 1 if p['controller'] == 0 else -1
                    q = {r['pit']: r['value_player0'] for r in ref['action_scores']}
                    regret = sign * (ref['target_player0'] - q[chosen])
                    assert regret == row['reference_regret']
                    best = ref['best_moves']; visits = report['visits']; priors = report['root_priors']; total = sum(visits)
                    item = dict(campaign=campaign, state_key=case['state_key'], stratum=case['stratum'], time_ms=ms,
                        regret=regret, chosen=chosen, best_moves=best,
                        teacher_best_prior_mass=sum(priors[i] for i in best),
                        teacher_best_visit_fraction=sum(visits[i] for i in best) / total,
                        teacher_best_unvisited=not any(visits[i] for i in best),
                        chosen_is_top_prior=priors[chosen] == max(priors),
                        mean_estimates_player0=report['action_mean_value_player0'], visits=visits, priors=priors)
                    if ref['kind'] == 'exact_optimal_margin':
                        endgame.append(item)
                    else:
                        records.append(item)
    summary = {}
    for kind in ('ordinary', 'capture', 'bonus'):
        for ms in (25, 100):
            group = [r for r in records if r['stratum'].endswith('/' + kind) and r['time_ms'] == ms]
            errors = [r for r in group if r['regret'] > 0]
            summary[kind + '/' + str(ms)] = dict(measurements=len(group), unique_roots=len({r['state_key'] for r in group}),
                disagreements=len(errors), mean_teacher_regret=statistics.mean(r['regret'] for r in group),
                error_teacher_best_unvisited=sum(r['teacher_best_unvisited'] for r in errors),
                error_teacher_best_visits_le5pct=sum(r['teacher_best_visit_fraction'] <= .05 for r in errors),
                error_chosen_is_top_prior=sum(r['chosen_is_top_prior'] for r in errors),
                mean_best_prior_mass_in_errors=statistics.mean(r['teacher_best_prior_mass'] for r in errors) if errors else None)
    assert len(records) == 192 and len(endgame) == 128 and all(r['regret'] == 0 for r in endgame)
    return dict(summary=summary, error_examples=sorted((r for r in records if r['regret'] > 0), key=lambda r: (-r['regret'], r['state_key'], r['time_ms']))[:16],
                existing_measurements=192, unique_H8_roots=len({r['state_key'] for r in records}),
                exact_diagnostic_measurements=128, unique_exact_roots=len({r['state_key'] for r in endgame}),
                exact_regrets_all_zero=True, endgame_scope='32 existing K4 diagnostic roots, each observed at two budgets in two campaigns; no new endgame generalization claim',
                interpretation='Teacher disagreements and allocation symptoms; H8 labels are approximate and do not certify blunders or causal search defects')


def arena(rows, labels):
    native = Native(); counts = Counter(); choices = Counter(); matched = []; candidates = 0; games = 0; distinct = set()
    starts = Counter(); by_start = defaultdict(Counter); ordinary_initial = Counter(); ordinary_later = Counter()
    exact = Counter(); label_splits = Counter(); non_K4 = 0
    label_index = {bytes.fromhex(k): k for k in labels}
    pin(ORACLE_PATH)
    with closing(database('data/value-benchmark-v1/work.sqlite3')) as db, Oracle(native=native) as oracle:
        for text, digest in db.execute("SELECT body,hash FROM games WHERE comparison IN ('policy_control_vs_H_v1','policy_control_vs_H8_relay_CPU') ORDER BY idx"):
            body = json.loads(text); assert config.digest(body) == digest
            p = body['start']; games += 1
            start_source = 'constructed_start' if body['family'].startswith('constructed-') else 'recorded_history_start'
            starts[start_source] += 1
            for index, (action, decision) in enumerate(zip(body['moves'], body['decisions'])):
                p = native.normalize(flatten(p), actor=p['controller'], total=98)['position']
                assert decision['actor'] == p['controller']
                if decision['engine'] == 'policy_control':
                    _, stratum = pilot._mechanism(p, native)
                    counts[stratum] += 1; candidates += 1
                    by_start[start_source][stratum] += 1
                    if stratum.endswith('/ordinary'):
                        (ordinary_initial if index == 0 else ordinary_later)[stratum] += 1
                    key, canonical, swapped = pilot.canonicalize(p, native); distinct.add(key.hex())
                    if sum(map(sum, p['pits'])) > 4:
                        non_K4 += 1
                    if key in label_index:
                        label = labels[label_index[key]]; sign = 1 if canonical['controller'] == 0 else -1
                        q = {r['pit']: r['value_player0'] for r in label['action_scores']}
                        matched.append(dict(stratum=stratum, regret=sign * (label['target_player0'] - q[action])))
                        label_splits[rows[label_index[key]]['split']] += 1
                step = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                if decision['engine'] == 'policy_control': choices[step['reason']] += 1
                if decision['engine'] == 'policy_control' and sum(map(sum, p['pits'])) <= 4:
                    root_exact = oracle.lookup(p); child_exact = oracle.lookup(step['position'])
                    regret = (1 if p['controller'] == 0 else -1) * (root_exact['value_player0'] - child_exact['value_player0'])
                    assert regret >= 0 and ((action in root_exact['best_moves']) == (regret == 0))
                    exact['choices'] += 1; exact['nonoptimal'] += regret > 0; exact['total_regret'] += regret
                p = step['position']
            assert p == body['final']
    assert games == 96 and candidates > 0
    return dict(games=games, candidate_choices=candidates, distinct_candidate_roots=len(distinct), strata=dict(sorted(counts.items())),
                game_start_sources=dict(starts), strata_by_start_source={k: dict(v) for k, v in by_start.items()},
                ordinary_initial_root_counts=dict(ordinary_initial), ordinary_later_root_counts=dict(ordinary_later),
                chosen_move_reasons=dict(choices), exact_canonical_H8_label_matches=len(matched),
                matched_teacher_disagreements=sum(r['regret'] > 0 for r in matched),
                H8_matches_by_split=dict(label_splits), non_K4_choices=non_K4,
                unlabeled_non_K4_choices=non_K4-len(matched), exact_K4=dict(exact),
                scope='Observed control decisions in its existing96 games only; repeated roots correlated. All9000 existing train/development H8 roots were checked; K4 uses existing read-only verified exact lookup. Unmatched non-K4 moves cannot be called teacher errors. Ordinary root means no legal pass/capture/bonus under the priority classifier; it is not every ordinary chosen move.')


def document(body):
    coverage = body['training_coverage']; dev = body['development']; cases = body['cases']; arena_body = body['arena']
    lines = ['# Retained control: data and mistake diagnosis', '',
        'The improved policy/control remains the learned candidate. H_v1 stays the default. This diagnosis performs no training, changes no model/search settings and generates no new positions or teacher labels.', '',
        '## Unique data and sampling', '',
        'The retained control used 7,669 selected roots from 4,576 families; the later value refit used 7,592 roots from 4,546 families. Each run made 384,000 draws with replacement. Repeated optimizer draws are not additional positions. The warm ancestor had already used the original 8,000 roots; current filters do not erase that exposure.', '',
        '| Stratum | Control roots | Control recorded-history roots | Refit roots | Refit recorded-history roots | Control draws |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, row in coverage['quality']['strata'].items():
        other = coverage['value']['strata'][name]
        lines.append(f"| {name} | {row['rows']} | {row['recorded_history_rows']} | {other['rows']} | {other['recorded_history_rows']} | {row['actual_draws']} |")
    rare = coverage['quality']
    lines += ['', f"The two strata with at most five roots contain {rare['rare_le5_rows']} positions but receive {rare['rare_le5_draws']:,} of 384,000 control draws ({100*rare['rare_le5_draws']/384000:.2f}%). This is a consequence of intentional uniform-stratum weighting. It signals repetition risk, not proof that the weighting caused a particular loss. Constructed examples are valid positions without a recorded game-history reachability witness; they must not be presented as observed game diversity.", '',
        '## Development errors by source', '',
        'These are the same 1,000 reused development roots and approximate H8 targets. Counts below expose coverage and provenance; they are not a fresh release test.', '',
        '| Stratum | Roots | Control MAE, seeds | Policy regret, seeds | Recorded-history roots / MAE | Constructed roots / MAE |',
        '| --- | ---: | ---: | ---: | --- | --- |']
    for name, row in dev['strata'].items():
        def cell(value): return str(value['rows']) + (' / ' + f"{value['mae_seeds']:.3f}" if value['rows'] else ' / unavailable')
        lines.append(f"| {name} | {row['rows']} | {row['mae_seeds']:.3f} | {row['policy_teacher_regret']:.3f} | {cell(row['by_source']['recorded_history'])} | {cell(row['by_source']['constructed'])} |")
    lines += ['', 'There is no development support for 75-98/ordinary. Five training roots in that stratum do not establish reliable early-game behavior. Active-counter bands are phase proxies, and the root mechanism classifier is stricter than the reason for one chosen move.', '',
        '## Existing search decisions', '',
        '| Mechanism / milliseconds | Measurements | Teacher disagreements | Mean regret | Disagreements with <=5% visits on teacher best |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, row in cases['summary'].items():
        lines.append(f"| {name} | {row['measurements']} | {row['disagreements']} | {row['mean_teacher_regret']:.3f} | {row['error_teacher_best_visits_le5pct']} |")
    lines += ['', 'The 192 H8 search measurements cover 96 roots at two budgets. Low visits on the teacher-preferred move are an allocation symptom; approximate H8 disagreements do not prove a correct alternative or establish causality. Worst examples with states, action estimates, priors and visits are retained in the JSON report.', '',
        '## Existing arena coverage', '',
        f"The control's 96 existing games contain {arena_body['candidate_choices']:,} control decisions at {arena_body['distinct_candidate_roots']:,} distinct canonical roots. Of {arena_body['non_K4_choices']:,} decisions outside K4, {arena_body['exact_canonical_H8_label_matches']:,} match the existing 9,000-root H8 corpus ({arena_body['H8_matches_by_split'].get('development',0)} development and {arena_body['H8_matches_by_split'].get('train',0)} training); {arena_body['unlabeled_non_K4_choices']:,} have no existing H8 root label. The other {arena_body['exact_K4']['choices']} decisions are covered by the existing exact K4 oracle. Final losses cannot be assigned to individual non-K4 moves from this evidence alone.", '',
        '| Root stratum | Observed control decisions |', '| --- | ---: |']
    for name, count in arena_body['strata'].items(): lines.append(f'| {name} | {count} |')
    lines += ['', 'These arenas use selected starting positions: 72 games start from recorded-history roots and 24 from constructed roots. Of the 12 ordinary 50-74 decisions, 10 are initial constructed-root decisions; only two occur in recorded-history-start games. No 75-98/ordinary root occurs in this selected arena sample. These observations do not estimate its frequency in full games from the standard opening. They argue against choosing a large rare-stratum quota merely from five training examples.', '',
        '## Endgame and next decision', '',
        f"The retained control has zero regret on all 32 existing exact K4 diagnostic roots at both budgets in both campaigns (128 observations, not 128 independent positions). Additionally, read-only verified oracle lookups certify {arena_body['exact_K4']['choices']} observed arena choices: {arena_body['exact_K4']['nonoptimal']} nonoptimal and {arena_body['exact_K4']['total_regret']} total seed regret. No new oracle was built or integrated into search, and these correlated observations are not an unseen release test. This diagnosis leaves the model and endgame code unchanged. Learned low-active values above K4 remain approximate.", '',
        f"Keep the improved policy/control. First audit repetition weights and the non-K4 label gap: {arena_body['unlabeled_non_K4_choices']:,} observed choices lack an existing H8 root label after checking both training and development. A small diagnostic labeling pilot on saved positions can distinguish policy/value/search errors; those protected evaluation trajectories must not silently become training rows. If the errors point to missing experience, collect new independent game-history families for training and separately owned validation, measuring ordinary-state yield before choosing quotas. Retain useful constructed examples, low-active coverage and the exact regression cases. Any later model change still needs equal-time games against both baselines.", '',
        'No expansion or retraining is performed here. The corpus is small and uneven, so targeted data expansion is a justified hypothesis; the existing measurements do not prove that more data will improve playing strength. Availability/rarity, teacher horizon error, limited representation and search allocation remain distinct hypotheses. New labels or counterfactual searches would be needed to distinguish causes on currently unlabeled game decisions.', '',
        'Evidence: `build/control-diagnosis.json`, the unchanged accepted quality/value manifests, H8 development roots, and the existing quality/value case and value arena journals. All prior 1,829 accepted input hashes are checked before and after diagnosis.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    accepted = load('build/value-acceptance.json')
    assert accepted['passed'] and len(accepted['inputs']) == 1829
    for name, digest in accepted['inputs'].items(): assert pin(ROOT / name) == digest, name
    pin(__file__); emit('admitted', prior_inputs=1829, training=False, new_teacher_labels=0)
    rows, labels, coverage, all_labels = dataset(); emit('coverage', retained_control_roots=coverage['quality']['rows'])
    dev = development(rows, labels); case = case_diagnosis(); emit('existing_search_cases', measurements=case['existing_measurements'])
    games = arena(rows, all_labels)
    assert 'torch' not in sys.modules
    body = dict(schema='retained-control-diagnosis-v1', passed=True, training_coverage=coverage, development=dev, cases=case, arena=games,
        control_model='data/quality-train-v1/model.json', candidate_unchanged=True, default_engine='H_v1',
        training_performed=False, generated_positions=0, generated_teacher_labels=0, sealed_final_test_consumed=False,
        torch_imported=False, causal_attribution=False, independent_unseen_validation=False,
        prior_inputs_preserved=1829, inputs=dict(sorted(INPUTS.items())))
    for name, digest in accepted['inputs'].items(): assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    text = document(body)
    for path, value in [(OUTPUT, config.canonical(body) + '\n'), (DOCUMENT, text)]:
        if path.exists(): assert path.read_text(encoding='utf-8') == value, 'Existing diagnostic differs: ' + str(path)
        elif not args.check_only:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(value, encoding='utf-8'); temporary.replace(path)
    emit('complete', prior_inputs_preserved=1829, control_roots=coverage['quality']['rows'], development_roots=1000,
         case_measurements=case['existing_measurements'], arena_decisions=games['candidate_choices'], artifacts_written=not args.check_only)


if __name__ == '__main__': main()
