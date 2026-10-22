"""Freeze the one prospective recipe before independent pre-run admission."""
from . import config,ordinary_epoch_common as s

def definition():
    return dict(schema=s.SCHEMA+'-plan',previous_acceptance_sha256=s.PRIOR_SHA,
        epoch_id=s.epoch()['epoch_id'],policy_fixed=True,base_frozen=True,training_authorized=True,
        continuation=dict(version=3,superseded_plan='build/ordinary-epoch-pilot-v1/plan-v2.json',
            superseded_plan_sha256='594ad3e4a45012d286049405d1487812ea9e144ec0e8ca9ecdabfe5445300064',
            original_plan_sha256='a68c23d927f9486bcf3bd3045706b01bb2ed884cf64d12d6c00c9354acc4500c',
            reason='Deduplicate path hashing and batch exact common ownership verification with constructor-issued snapshots, preserving provenance and scientific recipe',
            cache='data/ordinary-epoch-pilot-v1/cache-v2',initial='data/ordinary-epoch-pilot-v1/initial-v2',
            training='data/ordinary-epoch-pilot-v1/training-v2',prior_native_queries=0,prior_optimizer_steps=0),
        labels_authorized=True,development_evaluation_authorized=True,promotion_authorized=False,
        scope='ordinary-only conditional inherited-background development',
        train=dict(roots=192,cells=[39,34,58,61],cell_order=['25-49/ordinary/actor0','25-49/ordinary/actor1','50-74/ordinary/actor0','50-74/ordinary/actor1'],
            sampler='torch randint four cells per batch, then uniform roots within cells in fixed cell order',
            cell_probability='1/4',loss_weight=1,steps=400,batch_size=128,draws=51200,
            sampler_seed=20261016,initialization_seed=20261016,initialization='exact accepted zero-output sidecar tensors',
            optimizer='AdamW',learning_rate=.0003,betas=[.9,.999],epsilon=1e-8,weight_decay=0,
            huber_delta=.1,gradient_clip_l2=1,checkpoint_interval=20,final_step_only=True,
            runtime='recorded Torch CPU FP32 single-thread deterministic',cuda=False,
            target='original completed H_v1 choice-depth8 full-legal-root-Q player0 seed-margin',
            normalization='(margin-store_difference)/signed_active_seeds',target_clipping=False,
            cached_features_inherited=True,capture_training=False,policy_training=False),
        teacher=dict(engine='H_v1/untrained/oracle-free',depth=8,full_root_Q=True,memory_mb=4096,
            inherited_source_memory_mb=512,memory_cap_semantics='process-wide resource only; incomplete horizon forbidden',tt_mb=16,use_pvs=True,
            time_ms=0,node_cap=1000000,trace_event_cap=16000000,required_fit_labels=128,
            required_completed_depth=8,cap_or_overflow='retain full evidence and HOLD; no fallback target'),
        evaluation=dict(fit_roots=128,fit_class_counts=dict(ordinary=64,capture=32,bonus=32),
            fit_cases_subset=32,case_class_counts=dict(ordinary=16,capture=8,bonus=8),arena_origins=24,profiles=['control','candidate'],
            baselines=['H_v1','H8_relay_CPU'],seats=[0,1],games=192,critical_budget_ms=25,
            maximum_choices_per_game=512,
            reserve_ms=12,minimum_nominal_window_ms=10,shared_frozen_endgames_max_active=24,
            per_decision_reset=True,whole_component_guards=True,all_new_quotients_claimed_before_next_query=True,
            arena_order=dict(origin_salt='ordinary-epoch-arena-20261016/',origin_sort='SHA256(salt+state_key)',
                matchup_cycle=[['H_v1',0],['H8_relay_CPU',1],['H_v1',1],['H8_relay_CPU',0]],
                rotate_matchups='origin_index modulo4',adjacent_profile_pairs=True,
                first_profile='control when(origin_index+matchup_index) even; candidate otherwise'),
            case_order=dict(root_salt='ordinary-epoch-cases-20261016/',root_sort='SHA256(salt+state_key)',
                modes=['timed25','fixed128'],reverse_modes_on_odd_roots=True,adjacent_profile_pairs=True,
                first_profile='control when(root_index+mode_index) even; candidate otherwise'),
            failures='retain all attempts; cross-role bridge or unknown incomplete query HOLDs campaign'),
        gates=dict(lower_overall_huber=True,no_higher_overall_seed_mae=True,no_higher_capture_seed_mae=True,
            no_higher_capture_case_regret=True,shared_endgame_exact_equality=True,
            both_baselines_complete_equal_time=True,paired_origin_uncertainty=True,
            no_automatic_promotion=True,no_reseed_no_retune_no_checkpoint_selection=True,
            minimum_huber_improvement=1e-12,mae_and_regret_tolerance=1e-9,
            native_value_tolerance=1e-9,native_policy_tolerance=1e-12,
            fit_numeric_backend='strict scalar double on full-precision native127features; training cache remains FP32',
            low_active_identical_fixed_work='exact value/policy/selected move; no learned branch calls',
            critical_overshoot_rule='ANY critical elapsed>25ms fails equal-time interpretation; retain all games'),
        uncertainty=dict(unit='reserved origin',origins=24,origin_weights='equal',seats='average before resampling',
            contrasts=['candidate absolute margin vs H_v1','candidate absolute margin vs H8 relay CPU',
                       'candidate minus control margin vs H_v1','candidate minus control margin vs H8 relay CPU'],
            method='paired percentile bootstrap',resamples=20000,seed=20261016,
            per_interval_two_sided_coverage=.9875,multiplicity='Bonferroni four contrasts, family coverage at least .95 nominal',
            positive_strength='all four lower bounds>0 plus every fit/capture/endgame/ownership/timing gate',
            limitations='Exploratory bootstrap over24 custom origins; nominal coverage does not prove calibration or natural-play generalization'),
        resource=dict(cumulative_worker_seconds=10800,lease_seconds=300,private_commit_gib=4,namespace_gib=20,
            clean_stop_margin_seconds=15,unclean='full lease charged',unknown_overrun='HOLD',
            native_timed_cpu=dict(cache=0,core=2,efficiency_class=1,group=0,id=258,logical=2,numa=0),
            containment='independent Windows Job watchdog bound to actual process and exact durable argv'),
        observer=dict(full_byte_verification='independent audit, final combined seal, explicit verification',
            live_poll='exact current sources, model/control files and compact immutable catalogs/reports/prefix certificates',
            raw_archives_continuously_rehashed=False,mtime_only_cache=False,prior_files_preserved=True),
        recovery=dict(trainer='actual interrupted and uninterrupted six-step fixtures; optimizer/RNG/sample identity',
            fixture_root='build/ordinary-epoch-pilot-v1/fixtures/independent-training-recovery-v1',
            fixture_names=['uninterrupted','before_checkpoint_catalog','after_checkpoint_catalog'],
            fixture_replay='build/ordinary-epoch-pilot-v1/audit/training-replay-fixtures',
            production_replay='build/ordinary-epoch-pilot-v1/audit/training-replay-production',
            independent_replay_optimizer_steps=418,independent_replay_candidate_exports=0,
            production_cooperative_cursor=20,
            native_fixture_root='build/ordinary-epoch-pilot-v1/fixtures/native-recovery-v1',
            label_subset_kill_hooks=['before_decision_commit','after_decision_commit','before_ownership_check_commit',
                'after_ownership_check_commit','before_catalog','after_catalog'],
            label_subset_rule='first reserved fit root; all actual high quotients already owned by same-origin completed labels; zero new claims',
            unknown_intent_hook='before_query',unknown_intent_rule='actual kill before native query entry; source-bound zero-query proof; fixture HOLD and no retry',
            handoff_kill_hooks=['before_handoff_commit','after_handoff_commit'],engineering_handoff_games=2,
            handoff_fixture_rule='one game each, first salted reserved arena origin, final candidate vs H_v1 seat0; cumulative claims seed primary arena',
            mandatory_initial_equivalence='jobs/initial-conformance first fit root, stepzero features/predictions/fixed128 twins; receipt gates production training',
            operation_chain=['jobs/labels','jobs/initial-conformance','production training400','jobs/conformance',
                'jobs/guards','fixtures/native-recovery-v1/before_handoff_commit',
                'fixtures/native-recovery-v1/after_handoff_commit','jobs/arena'],
            queries='complete returned archive adoption; mmap-only unknown query HOLD, no retry',
            completed_reader='zero new optimizer/search/provider work',power_loss_guarantee=False),
        historical_provenance_complete=False,historically_unseen=False,inherited_unknown_background=True,
        old_candidate_rehabilitated=False,default_engine='H_v1')

def bindings():
    """Exact mandatory closure; an empty/partial resealed map cannot admit work."""
    sources=list((s.ROOT/'congklak').glob('ordinary_epoch*.py'))
    sources+=list((s.ROOT/'tools').glob('*ordinary-epoch*'))
    sources+=list((s.ROOT/'tests').glob('*ordinary_epoch*'))
    sources+=list((s.FOLDER/'audit').glob('*.py'))
    sources+=[s.ROOT/'docs/ORDINARY_EPOCH_PILOT.md']
    sources=[p for p in sources if p.is_file()]
    # The native provider contributes its own strict copied-source/build closure.
    from . import ordinary_epoch_native as native
    if hasattr(native,'input_pins'):
        native_pins=native.input_pins()
    else:native_pins={}
    source_inputs=s.common.pins(sources);source_inputs.update(native_pins)
    inherited_reservations=s.common.pins([s.JOINT,s.EPOCH,s.ROOT/'build/training-runtime.json',s.ROOT/'build/value-plan.json'])
    reservation_inputs=dict(inherited_reservations);reservation_inputs.update(s.common.pins([s.FOLDER/'baseline.json']))
    models=s.common.pins(s.MODELS);prior=s.read(s.PRIOR)
    s.need(all(prior['inputs'].get(k)==v for k,v in {**models,**inherited_reservations}.items()),'An inherited fixed model/reservation/runtime changed')
    baseline=s.read(s.FOLDER/'baseline.json');expected=dict(prior['inputs']);expected[s.relative(s.PRIOR)]=s.PRIOR_SHA
    s.need(config.canonical(baseline['inputs'])==config.canonical(expected) and
        type(baseline['inputs_count']) is int and baseline['inputs_count']==len(expected),'Immutable baseline expectation map differs')
    return dict(source_inputs=source_inputs,reservation_inputs=reservation_inputs,models=models)

def freeze():
    s.baseline();b=definition();b.update(bindings())
    if s.PLAN.exists():s.need(config.canonical(s.read(s.PLAN))==config.canonical(b),'Frozen recipe cannot change')
    else:s.atomic(s.PLAN,b)
    return b
