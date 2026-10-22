"""Prospective acquisition and reservation contract; no training permission."""
from . import config,footprint_screen_common as s

NATIVE_HOOKS=('before_result_commit','after_result_commit','before_trace_commit','after_trace_commit',
              'before_catalog','after_catalog')
def definition():
    from . import footprint_screen_data as data
    return dict(schema=s.SCHEMA+'-plan',prior_closure_sha256=s.PRIOR_SHA,
        profile=config.PROFILE,rules_digest=config.RULES_DIGEST,
        scope='Complete recorded-footprint acquisition and conditional development reservations',
        models_fixed=True,policy_fixed=True,training_authorized=False,model_forwards_authorized=False,
        model_promotion_authorized=False,previous_pilot_stays_held=True,
        population=data.protocol(),
        acquisition=dict(roots=512,queries_per_root=['teacher','features'],total_queries=1024,
            all_roots_before_reservations=True,stop_when_quotas_filled=False,
            teacher=dict(engine='H_v1',depth=8,time_ms=0,node_cap=1000000,tt_mb=16,use_pvs=True,memory_mb=4096,
                target='all legal full-root finite-choice-depth8 exact player0 seed margins',oracle=False),
            features=dict(encoder='frozen native127 relay-aware',precision='binary64',model_forwards=0),
            raw_event_cap=16000000,unique_owned_cap=4000000,global_claim_cap=8000000,
            recording_guard='empty static root-local workspace only; final ownership determined from full joint closure',
            source_family='true frozen root family retained for every candidate and fixture',
            capped_h8='complete valid physical trace retained; candidate ineligible; paired features still required; no retry',
            fatal_hold=['unknown unreturned native work','incomplete raw capture','raw or owned overflow',
                'invalid source or physical coverage','missing or unaccounted fixture footprint'],
            candidate_rejections=['protected prior component intersection','H8 target incomplete under fixed cap'],
            complete_batch_required=True,no_numeric_quality_selection=True),
        ownership=dict(all_previous_actual_queries=5,previous_failed_high_quotients=6669,
            all_old_components_and_families_retained=True,whole_quarantine_320=True,
            support192_fixed=True,support_cells=[39,34,58,61],owned_min=25,owned_max=98,shared_max=24,
            all_actual_candidates_join_before_selection=True,all_actual_fixtures_join=True,
            reject_protected_component_merges=True,one_selected_root_per_joint_component=True,
            numeric_Q_value_regret_used=False,full_independent_raw_replay_required=True),
        reservations=dict(fit_per_cell=16,fit_roots=128,cases_per_cell=4,cases_subset=32,
            arena_per_ordinary_cell=6,arena_origins=24,distinct_components=152,
            method='deterministic maximum bipartite component-slot matching',
            shortage='retain complete batch, maximum matching and shortage witness; no automatic expansion',
            shared_endgames='inherited fixed control suffix at active<=24; exact32 K4 guards unchanged'),
        recovery=dict(native_hooks=list(NATIVE_HOOKS),zero_query_hook='before_query',
            native_receipt='build/footprint-screen-v1/audit/native-recovery/result.json',
            review_hooks=['before_unit_commit','after_unit_commit'],
            reservation_hooks=['before_catalog','after_catalog'],actual_process_interruptions=11,
            fixtures='declared first-root native subsets; review/publication read captured data only',
            orphan_complete_result='adopt before any new provider or query',
            unknown_query='retain HOLD; no retry',completed_reader='zero new engine/provider/worker/lease work',
            power_loss_guarantee=False),
        resources=dict(total_worker_ns=s.TOTAL_WORKER_NS,inherited_charged_ns=s.INHERITED_CHARGED_NS,
            lease_seconds=300,private_commit_gib=4,combined_old_new_quota_gib=20,
            memory_deadline_poll_seconds=.025,quota_poll_seconds=1.0,
            large_publication='exact quota checks plus bounded pre-query headroom',
            unclean='full durable reservation charged',unknown_overrun='HOLD',
            heavy_workers=['acquisition','independent raw review','reservation publication'],
            read_only_control_cost='pure source/preservation inspection recorded separately'),
        reporting=dict(all512_dispositions=True,all_rejections_and_caps=True,quotas_and_shortage=True,
            raw_audit_and_recovery_receipts=True,preserve_all_prior_expected_hashes=True,
            live_integrity='compact source and receipt checks; full bytes checked at audit/seal/explicit verify'),
        inherited_unknown_background=True,historically_unseen=False,historical_provenance_complete=False,
        natural_game_representativeness_claim=False,conditional_on_structural_screening=True,
        independent_strength_established=False,default_engine='H_v1')
def bindings():
    from . import footprint_screen_data as data,footprint_screen_native as native
    files=list((s.ROOT/'congklak').glob('footprint_screen*.py'))
    files+=list((s.ROOT/'tools').glob('*footprint-screen*'))
    files+=list((s.ROOT/'tests').glob('*footprint_screen*'))
    sources=s.common.pins(p for p in files if p.is_file())
    sources.update(native.input_pins())
    inputs=data.input_pins()
    inputs.update(s.common.pins([s.PRIOR,s.FOLDER/'baseline.json',s.FOLDER/'pool.json']))
    old=s.prior_inputs()
    for key,value in {**sources,**inputs}.items():
        if key in old:s.need(value==old[key],'An inherited frozen input changed: '+key)
    models=s.common.pins(s.MODELS)
    s.need(all(old.get(k)==v for k,v in models.items()),'A fixed model changed')
    return dict(source_inputs=sources,metadata_inputs=inputs,models=models,
        pool=dict(path=s.relative(s.FOLDER/'pool.json'),sha256=s.sha(s.FOLDER/'pool.json')),
        parent_budget=s.parent_budget())
def expected():
    body=definition();body.update(bindings());return body
def freeze():
    from . import footprint_screen_data as data
    s.baseline();data.validate_pool(s.read(s.FOLDER/'pool.json'))
    body=expected();s.atomic(s.PLAN,body);return body
