"""One fixed ordinary value treatment using the screened development roles."""
from pathlib import Path
from . import config, guarded_value_pilot_v2_common as s

def definition():
    from .ordinary_epoch_plan import definition as inherited
    body=inherited()
    body.update(schema=s.SCHEMA+'-plan',previous_acceptance_sha256=s.PRIOR_SHA,epoch_id=s.SCHEMA,
        labels_authorized=False,scope='ordinary-only conditional screened-development pilot',
        continuation=dict(version=2,reason='Use sealed complete-footprint reservations; inherited192 numerical rows and unchanged value recipe',
            old_pilot_stays_held=True,cache='data/guarded-value-pilot-v2/cache',initial='data/guarded-value-pilot-v2/initial',
            training='data/guarded-value-pilot-v2/training',new_teacher_queries=0),
        source_reuse=dict(training_roots=192,screened_candidates=512,fit_roots=128,cases_subset=32,arena_origins=24,
            acquired_origins_optimizer_blocked=True,new_optimizer_rows_admitted=0,labels_and_features_reused=True),
        native_guard=dict(representation='sorted19/q15/componentOrdinal32',maximum_static=16000000,maximum_claims=8000000,
            shared_index=True,full_owned_domain=[25,98],shared_endgames_max_active=24))
    body['resource'].update(inherited_worker_seconds=s.INHERITED_NS/1e9,cumulative_worker_seconds=10800,
                            clean_stop_margin_seconds=15,namespace_includes_inherited=True)
    recovery=body['recovery']
    recovery.update(fixture_root='build/guarded-value-pilot-v2/fixtures/independent-training-recovery-v2',
        fixture_replay='build/guarded-value-pilot-v2/audit/training-replay-fixtures',
        production_replay='build/guarded-value-pilot-v2/audit/training-replay-production',
        native_fixture_root='build/guarded-value-pilot-v2/fixtures/native-recovery-v2',
        label_subset_kill_hooks=[],label_subset_rule='No label worker: accepted captured targets/features reused',
        unknown_intent_hook=None,unknown_intent_rule='Reuse accepted prior zero-query unknown-intent kill proof; unchanged native trace protocol remains fail closed',
        inherited_actual_recovery_proofs=11,new_actual_kills=4,
        mandatory_initial_equivalence='first screened fit root; stepzero features/predictions/fixed128 exact twins',
        operation_chain=['prepare cached labels/features','jobs/initial-conformance','production training400',
            'jobs/conformance','jobs/guards','native handoff recovery','jobs/arena'])
    return body

def source_inputs():
    paths=list((s.ROOT/'congklak').glob('guarded_value_pilot_v2*.py'))
    paths+=list((s.ROOT/'tools').glob('*guarded-value-pilot-v2*'))
    paths+=list((s.ROOT/'tests').glob('*guarded_value_pilot_v2*'))
    paths+=list((s.FOLDER/'audit').glob('*.py'))
    paths+=list((s.ROOT/'native/guarded_value_pilot_v2').glob('*'))
    paths+=[s.FOLDER/'native/build.json',s.FOLDER/'native/source-projection.json',
        s.FOLDER/'native/adapter-projection-v2.json',s.FOLDER/'native/guard-projection-v3.json',
        s.FOLDER/'native/integration-projection-v3.json',s.FOLDER/'audit/execute-projection.json',
        s.FOLDER/'native/congklak_guarded_value_pilot_v2.dll']
    return s.common.pins([p for p in paths if p.is_file()])

def reservation_inputs():
    marker=s.read(s.PRIOR);r=s.ROOT/'build/footprint-screen-v1/reservations'
    paths=[s.PRIOR,s.ROOT/marker['closure_ref']['path'],s.FOLDER/'baseline.json',r/'report.json',r/'certificate.json',
           r/'quotient-components.sqlite3',s.ROOT/'data/ordinary-epoch-pilot-v1/cache-v2/manifest.json',
           s.ROOT/'data/ordinary-epoch-pilot-v1/cache-v2/train.json',s.ROOT/'build/training-runtime.json']
    return s.common.pins(paths)

def freeze():
    s.baseline();body=definition();body.update(source_inputs=source_inputs(),reservation_inputs=reservation_inputs(),models=s.common.pins(s.MODELS))
    s.atomic(s.PLAN,body);return body
