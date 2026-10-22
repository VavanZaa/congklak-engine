"""Apply the predeclared value-pilot gates without selecting another treatment."""
from pathlib import Path
from . import neurodiag_common as common

ROOT = common.ROOT
DEFAULT = ROOT / 'build/value-run-v1/decision.json'
SOURCES = ('congklak/value_run_decision.py', 'tools/decide-value-run.py')


def classify(fit, cases, arena, conformance):
    exact = all(cases['exact_guard'][str(ms)]['passed'] for ms in (25, 100))
    regret = all(cases['h8_regret_gates'][key]['passed'] for key in ('timed/25', 'timed/100', 'fixed128/0'))
    fallback_count = sum(row['fallbacks'] for row in cases['metrics'].values())
    gates = dict(fit=fit['passed'], native_conformance=conformance['passed'],
        timed_and_fixed_H8_regret=regret, exact_optimality=exact,
        low_active_fixed_work=cases['low_active']['passed'],
        case_coverage=cases['timed_measurements'] == 512 and cases['fixed_searches'] == 138 and cases['fixed_completed_backups'] == 17664,
        no_case_fallbacks=fallback_count == 0,
        arena_coverage=arena['games'] == arena['terminal_games'] == 192,
        no_arena_caps_or_fallbacks=arena['capped_games'] == arena['fallbacks'] == 0,
        trajectory_ownership=arena['ownership']['ownership_passed'],
        nonnegative_paired_means=arena['nonnegative_paired_means_both_baselines'])
    reasons = [key for key, passed in gates.items() if not passed]
    strength_allowed = (arena['cluster_analysis_valid'] and fallback_count == 0 and exact and
                        gates['low_active_fixed_work'] and gates['case_coverage'])
    return dict(gates=gates, pilot_promising=all(gates.values()), failed_gates=reasons,
        candidate_quarantined=not gates['trajectory_ownership'] or not gates['low_active_fixed_work'],
        strength_interpretation_allowed=strength_allowed,
        exploratory_paired_improvement=strength_allowed and arena['positive_paired_lower_bounds_both_baselines'],
        candidate_beats_both_baselines_exploratory=strength_allowed and arena['positive_candidate_absolute_lower_bounds_both_baselines'],
        promotion=False, default_engine='H_v1', checkpoint_selection=False,
        expanded_data=False, additional_training=False)


def run(output=DEFAULT):
    path = Path(output).resolve()
    common.require(path.is_relative_to(ROOT / 'build/value-run-v1'), 'Decision output outside new experiment')
    if path.exists():
        body = common.read(path)
        common.require(body['schema'] == 'value-run-final-decision-v1', 'Unknown decision report')
        common.current(body['inputs'])
        return body
    from . import value_training as training, value_run_cases_v2 as cases, value_run_arena as arena
    production = training.completed()
    common.require(production['steps'] == 400 and production['draws'] == 51200 and production['policy_fixed'],
                   'Fixed final production training is incomplete')
    fit_path = ROOT / 'build/value-run-v1/cases-fit-v2.json'
    conformance_path = ROOT / 'build/value-run-v1/cases-conformance-v2.json'
    fit = cases.fit(fit_path)
    conformance = cases.conformance(conformance_path)
    case_report = cases.completed()
    arena_report = arena.verify()
    common.require(case_report['decisions'] == 650 and not case_report['engineering_fixture'] and
                   arena_report['completed'] and not arena_report['fixture'], 'Evaluation is partial or a fixture')
    common.require(conformance['passed'] is True and fit['policy_fixed'] is True, 'Policy or numeric conformance failed')
    decision = classify(fit, case_report, arena_report, conformance)
    prefix = arena._prefix(arena.DEFAULT)
    descriptive = {}
    for profile in arena.PROFILES:
        for baseline in arena.BASELINES:
            values = [g['model_margin'] for g in prefix['games']
                if prefix['identity']['schedule'][g['index']]['profile'] == profile and
                   prefix['identity']['schedule'][g['index']]['baseline'] == baseline and g['terminal']]
            descriptive[profile + '/' + baseline] = dict(games=len(values),
                mean_margin=sum(values) / len(values) if values else None,
                strength_interpretation_allowed=decision['strength_interpretation_allowed'])
    for baseline in arena.BASELINES:
        a = descriptive['control/' + baseline]['mean_margin']; b = descriptive['candidate/' + baseline]['mean_margin']
        descriptive['candidate-minus-control/' + baseline] = dict(mean_margin=b-a if a is not None and b is not None else None,
            strength_interpretation_allowed=decision['strength_interpretation_allowed'])
    files = [ROOT / n for n in SOURCES]
    files += [fit_path, conformance_path, cases.DEFAULT / 'certificate.json', cases.DEFAULT / 'catalog.json',
        cases.DEFAULT / 'report.json', arena.DEFAULT / 'certificate.json', arena.DEFAULT / 'head.json',
        arena.DEFAULT / 'result.json', training.DEFAULT / 'certificate.json', training.DEFAULT / 'model.json',
        ROOT / 'build/value-run-v1/production-complete.json', ROOT / 'docs/NEURAL_VALUE_PILOT_PLAN.md']
    body = dict(schema='value-run-final-decision-v1', evidence_complete=True, **decision,
        production_steps=400, production_draws=51200, new_labels=0,
        fit_gate=fit['gate'], fit_metrics=fit['metrics'], exact_guard=case_report['exact_guard'],
        H8_regret=case_report['h8_regret_gates'], low_active=case_report['low_active'],
        timed_measurements=512, fixed_searches=138, case_fallbacks=sum(r['fallbacks'] for r in case_report['metrics'].values()),
        arena_games=arena_report['games'], arena_terminal_games=arena_report['terminal_games'],
        arena_fallbacks=arena_report['fallbacks'], arena_capped_games=arena_report['capped_games'],
        descriptive_raw_comparisons=descriptive, cluster_statistics=arena_report['cluster_statistics'],
        ownership=arena_report['ownership'], new_optimizer_steps_here=0,
        limitations=['Fit-check roots have ancestral training exposure.',
                     'Ordinary arena starts are synthetic custom positions; opening reachability is unproven.',
                     'Bootstrap intervals are exploratory.',
                     'Negative gates retain the fixed control; no replacement checkpoint or treatment is selected.'],
        inputs=common.pins(files))
    return common.publish(path, body)
