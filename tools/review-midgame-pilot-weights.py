"""Compare offline exposure proposals on fresh completed-H10 training candidates only."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as prior

OUTPUT = ROOT / 'build/midgame-pilot-weight-review.json'


def distribution(rows, raw):
    if not rows:
        return dict(roots=0, families=0, probabilities={}, root_effective_support=None,
                    family_effective_support=None, maximum_root_probability=None, stratum_mass={})
    total = math.fsum(raw)
    assert total > 0 and all(math.isfinite(x) and x > 0 for x in raw)
    p = [x / total for x in raw]
    family = defaultdict(float)
    strata = defaultdict(float)
    for row, mass in zip(rows, p):
        family[row['family']] += mass
        strata[row['stratum']] += mass
    assert abs(math.fsum(p) - 1) < 1e-12
    return dict(roots=len(rows), families=len(family),
        probabilities={row['state_key']:x for row, x in zip(rows, p)},
        root_effective_support=1 / math.fsum(x*x for x in p),
        family_effective_support=1 / math.fsum(x*x for x in family.values()),
        maximum_root_probability=max(p), stratum_mass=dict(sorted(strata.items())))


def build():
    from congklak import midgame_data_pilot as collection, midgame_label_pilot as labeling
    fresh, collected = collection.load_rows()
    wrappers, stores, labeled = labeling.load_rows()
    rows, excluded = [], []
    for key, root in sorted(fresh.items()):
        wrapper = wrappers[key]
        assert wrapper['root'] == root and wrapper['study_role'] == 'fresh'
        assert wrapper['cohort'] == root['cohort']
        if root['cohort'] != 'training_candidate':
            excluded.append(dict(state_key=key, reason='protected_validation'))
            continue
        if key not in stores['h10_labels']:
            excluded.append(dict(state_key=key, reason='no_complete_requested_H10'))
            continue
        depths = [d for d in (8, 10) if key in stores[f'h{d}_labels']]
        targets = [stores[f'h{d}_labels'][key]['label']['target_player0'] for d in depths]
        overlap = None
        if len(depths) == 2:
            overlap = bool(set(stores['h8_labels'][key]['label']['best_moves']) &
                           set(stores['h10_labels'][key]['label']['best_moves']))
        span = max(targets) - min(targets)
        volatility = 1 / (1 + span / 4.) if len(depths) > 1 else .5
        disagreement = .5 if overlap is False else 1.
        rows.append(dict(state_key=key, family=root['family'], stratum=root['stratum'],
            cohort=root['cohort'], source_root_sha256=config.digest(root),
            requested_depths_available=depths, target_player0=targets,
            H8_H10_best_sets_overlap=overlap, target_span_seeds=span,
            volatility_factor=volatility, disagreement_factor=disagreement,
            proposed_exposure_factor=volatility * disagreement))
    n = Counter(row['stratum'] for row in rows)
    distributions = dict(uniform_root=distribution(rows, [1.] * len(rows)),
        equal_nonempty_target_stratum=distribution(rows, [1 / n[r['stratum']] for r in rows]),
        stability_weighted_root=distribution(rows, [r['proposed_exposure_factor'] for r in rows]))
    diagnostics = [r for r in wrappers.values() if r['study_role'] == 'diagnostic']
    assert len(diagnostics) == 9 and not ({r['state_key'] for r in rows} & {r['state_key'] for r in diagnostics})
    inputs = {name:prior.sha(ROOT/name) for name in (
        'tools/review-midgame-pilot-weights.py', 'build/midgame-pilot-baseline.json',
        'congklak/midgame_data_pilot.py', 'congklak/midgame_label_pilot.py',
        'data/midgame-data-pilot-v1/work.sqlite3', 'data/midgame-data-pilot-v1/manifest.json',
        'data/midgame-label-pilot-v1/work.sqlite3', 'data/midgame-label-pilot-v1/manifest.json')}
    return dict(schema='midgame-pilot-offline-weight-review-v1', passed=True,
        inputs=inputs, collection_manifest_sha256=config.digest(collected),
        label_manifest_sha256=config.digest(labeled), selected_fresh_roots=len(fresh),
        fresh_cohorts=dict(sorted(Counter(r['cohort'] for r in fresh.values()).items())),
        eligible_complete_H10_training_candidates=len(rows), protected_diagnostics=9,
        candidate_rows=rows, excluded_fresh_roots=excluded, distributions=distributions,
        effective_support_definition='Inverse probability concentration; not independent games or gradient ESS.',
        interpretation='Offline proposals over ONLY this new training-candidate slice. No replay blend or optimizer admission is made.',
        exposure_factor_interpretation='Uncalibrated horizon-stability heuristic, not probability of target correctness.',
        objective='Unit-loss expectation changes with the proposed draw distribution; no common target distribution or importance correction.',
        no_auxiliary_fallback=True, requested_H10_only=True,
        sampling_changed=False, loss_weights_applied=False, training_performed=False,
        model_changed=False, source_data_pruned=False, sealed_final_test_consumed=False,
        future_gate='Rebuild the full role-aware optimizer admission set before any replay blend, then freeze a separate learning/strength experiment.')


def main():
    body = build()
    prior.publish(OUTPUT, body)
    assert 'torch' not in sys.modules
    print(config.canonical(dict(event='midgame_pilot_offline_weight_review', passed=True,
        selected=body['selected_fresh_roots'], eligible=body['eligible_complete_H10_training_candidates'],
        protected_diagnostics=9, sampling_changed=False)), flush=True)


if __name__ == '__main__':
    main()
