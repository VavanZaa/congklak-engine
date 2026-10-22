"""Prospective-update scope, with no repair-by-erasure of historical exposure."""
from . import ownership_resolution_common as stage

METHODS = ('source_integrity', 'support_dispositions', 'historical_coverage',
           'known_trace_overlay', 'prospective_reservations', 'recovery',
           'completed_readers', 'observer', 'strength_claims')
KILL_POINTS = ('before_unit_commit', 'after_unit_commit', 'before_catalog', 'after_catalog')


def definition():
    return dict(schema=stage.SCHEMA+'-contract', models_fixed=True, policy_fixed=True,
        activity=stage.ZERO, training_authorized=False, labels_authorized=False,
        production_games_authorized=False, model_promotion_authorized=False,
        old_support=dict(roots=512, genuine_conflicting_roots=320,
            original_pool_and_roles_immutable=True, quarantine_whole_components=True,
            no_training_cache_reclassification=True, no_exposure_unlearning_claim=True,
            future_subset_requires_new_explicit_recipe=True),
        historical=dict(actual_saved_bytes_only=True, recorded_provenance_reconstruction_retained=True,
            replay_is_not_historical_trace=True, unknown_is_not_empty=True,
            missing_attempts_remain_permanent_unknowns=True,
            old_results_and_old_candidate_strength_not_rehabilitated=True),
        known_overlay=dict(include_all_accepted_trace_episodes_and_saved_prefixes=True,
            retain_all_complete_owned_states_ge25=True, retain_upper_band_75_98=True,
            low_states_remain_shared_and_recorded=True, actual_origin_incidence_only=True,
            accepted_source_bound_coverage_proof_reuse=True,
            unknown_or_unbound_attempt_is_not_silently_dropped=True),
        future_epoch=dict(estimand='known-provenance next-update development with inherited unknown background',
            historically_unseen_claim=False, exhaustive_historical_absence_claim=False,
            historical_provenance_complete=False, inherited_model_ancestry=True,
            prior_strength_confidence_not_restored=True,
            reserve_jointly_against_all_known_recorded_sources=True,
            no_new_root_construction=True, no_target_prediction_outcome_selection=True,
            existing_conditional_fit_roots=128, existing_fit_case_subset=32,
            existing_conditional_arena_origins=24, preserve_declared_cell_quotas=True,
            quota_shortage='HOLD; no silent shrink, filler, relabel or new population',
            new_training_recipe_and_explicit_user_run_required=True,
            fresh_complete_future_search_traces_and_runtime_guard_required=True),
        recovery=dict(commit_journal_immutable=True, full_prefix_verified_before_writes=True,
            reuse_complete_units=True, completed_catalog_before_writer=True,
            no_engines_or_torch=True, kill_points=list(KILL_POINTS)),
        admission=dict(historical_absence_remains_unproved=True,
            operational_resolution_only_with_named_narrow_epoch=True,
            no_automatic_permission_from_passed_flag=True), methods=list(METHODS))
