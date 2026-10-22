"""Integrate saved-data findings without authorizing another experiment."""
from .native import ROOT
from .ordinary_value_review_common import baseline, pins, current, publish, read, require

OUTPUT = ROOT / 'build/ordinary-value-review-conclusions.json'
REPORTS = ('build/ordinary-value-overlap-review.json', 'build/ordinary-value-loss-review.json',
           'build/ordinary-value-validation-review.json')


def build():
    frozen = baseline()
    overlap, arena, validation = [read(ROOT / name) for name in REPORTS]
    require(all(body['passed'] is True for body in (overlap, arena, validation)), 'A review report is incomplete')
    os, hv, vs = overlap['summary'], arena['summary']['H_v1'], validation['summary']
    require(os['original_allocation'] == 'hold' and os['action_taken'] == 'none', 'Original hold changed')
    source_names = ('build/ordinary-value-review-baseline.json', 'docs/ORDINARY_VALUE_REVIEW_SCOPE.md',
                    'congklak/ordinary_value_review_common.py', 'congklak/ordinary_value_review_conclusions.py',
                    'tools/review-ordinary-value-conclusions.py') + REPORTS
    inputs = pins(source_names)
    for report in (overlap, arena, validation):
        current(report['inputs'])
    body = dict(schema='ordinary-value-review-conclusions-v1', passed=True, inputs=inputs,
                prior_inputs_preserved=frozen['inputs_count'],
                summary='Keep policy and control fixed. Preserve the original overlap HOLD. No further training or collection; diagnose fixed-work capture choices and target stability before another bounded value experiment.',
                facts=dict(overlap_admitted_observations=os['admitted_quotient_match_observations'],
                    overlap_sampled_observations=os['sampled_quotient_match_observations'],
                    overlap_raw_full_state_observations=os['raw_state_match_observations'],
                    prior_training_draw_exposure=overlap['ancestral_exposure']['matching_row_positive_prior_draw_exposure'],
                    H_v1_control_wdl=hv['control_wdl'], H_v1_candidate_wdl=hv['candidate_wdl'],
                    H_v1_identical_trajectories=hv['identical_action_trajectories'],
                    H_v1_mean_seed_margin_delta=hv['paired_mean_candidate_minus_control'],
                    H_v1_negative_delta_seats=hv['negative_delta_seats'],
                    H8_context_mean_seed_margin_delta=arena['summary']['H8_relay_CPU']['paired_mean_candidate_minus_control'],
                    H10_control_mae=vs['H10_control_mae'], H10_candidate_mae=vs['H10_candidate_mae'],
                    H10_mae_delta=vs['H10_mae_delta'], H10_improved_roots=vs['improved_roots'],
                    H10_worsened_roots=vs['worsened_roots'], policy_logits_identical=vs['policy_logits_identical'],
                    high_validation_actor_band_confound=vs['high_roots_all_actor1']),
                interpretation=dict(original_gate_correctly_applied=True, current_gradient_leakage_from_matching_row_established=False,
                    zero_current_draws_implies_no_ancestry=False, identical_encoder_pattern_is_irrelevant=False,
                    value_head_isolated_as_game_loss_cause=False, H12_is_exact_truth=False,
                    six_root_panel_is_population_or_release_test=False,
                    interpolation_selects_a_new_model=False, small_training_fit_gain_justifies_scale=False),
                prospective_next_gates=[
                    dict(order=1, goal='Separate value/search effects from clock-driven visit differences.',
                         proposal='Use a new bounded fixed-simulation comparison on every adverse H_v1 shared-prefix divergence, with unchanged-margin divergence controls and identical search settings. Keep both models and policy fixed; do not call it a new equal-time strength result.', executed=False),
                    dict(order=2, goal='Establish action-ranking and value-target reliability.',
                         proposal='Where exact-full-state saved Q is missing, predeclare limited deeper per-action teacher work at the identified states and difficult ordinary roots. Retain horizon disagreement and capped evidence; no best-move claim from local captures or a single horizon.', executed=False),
                    dict(order=3, goal='Define protection before any new data or optimizer draws.',
                         proposal='Keep all96 latest trajectories and original family/quotient masks. Predeclare separate ancestral exposure, actual new draws, admitted-only and raw/transformed-state tiers. Store-offset/player-swap encoder equivalence is deliberate; never retroactively relabel consumed evaluation as unseen.', executed=False),
                    dict(order=4, goal='Allocate data only after the diagnosis identifies a target need.',
                         proposal='Use more distinct starts for weak ordinary coverage, and more independent rollouts per original start for noisy control-policy outcomes. Keep terminal-policy returns separate from finite-search targets and preserve capture/endgame rehearsal. Reconsider the declared unit-loss mixture before one small fixed-policy value pilot; no broad scale-up yet.', executed=False)],
                decision=dict(previous_allocation='hold', current_allocation='hold', further_training_authorized=False,
                    new_data_collection_authorized=False, new_model_selected=False, default_engine='H_v1',
                    learned_candidate='data/quality-train-v1/model.json', candidate_promoted=False,
                    audit_required_before_acceptance=True),
                work=dict(new_training_updates=0, new_teacher_searches=0, new_PUCT_searches=0, new_games=0, torch_imported=False))
    current(inputs)
    return body


def validate_report(body):
    require(body == build(), 'Review conclusions or evidence identities differ')
    return body


def run(path=OUTPUT): return publish(path, build())


def verify(path=OUTPUT): return validate_report(read(path))
