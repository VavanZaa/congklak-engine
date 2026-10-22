"""Read-only closure of implementation evidence; this never admits training."""
import json
from . import ownership_implementation_common as stage

PATH=stage.FOLDER/'acceptance.json'
SCHEMA='ownership-implementation-acceptance-v1'
SOURCES=('congklak/ownership_implementation_acceptance.py','tools/verify-ownership-implementation.py',
    'tools/finalize-ownership-implementation.py',
    'docs/OWNERSHIP_IMPLEMENTATION.md','docs/NATIVE_OWNERSHIP_TRACE_IMPLEMENTATION.md',
    'congklak/ownership_implementation_progress.py','tools/ownership-implementation-dashboard.html',
    'congklak/ownership_implementation_progress_v2.py','tools/ownership-implementation-dashboard-v2.html',
    'congklak/ownership_implementation_common.py','congklak/provenance_reader_v2.py',
    'congklak/ownership_handoff_launcher_v2.py','tools/run-ownership-handoff-v2.py')
METHODS=('facts','provenance_ranges','provenance_recovery','role_correction','native_trace',
         'worker_recovery','reservation_review','reservation_recovery','completed_readers')

def validate(body):
    stage.need(body['schema']==SCHEMA and body['passed'] is True and
        body['execution_ready'] is False and body['training_authorized'] is False and
        body['models_fixed'] is True and body['policy_fixed'] is True and
        body['training_counters']==stage.TRAINING_ZERO,'Implementation acceptance scope differs')
    baseline=stage.baseline()
    stage.need(body['prior_files_preserved']==59154 and all(body['inputs'].get(k)==v for k,v in baseline['inputs'].items()),
        'Implementation accepted history coverage differs')
    stage.need(all(body['inputs'].get(k)==v for k,v in stage.common.pins(SOURCES).items()),
        'Implementation closure source coverage differs')
    required_live=stage.common.pins([*SOURCES,stage.FOLDER/'baseline.json',stage.ROOT/body['progress_event'],
        stage.ROOT/body['independent_audit'],'data/quality-train-v1/model.json',
        'data/value-training-v1/training/model.json','data/training-stage5-deeper-h8-f127-seed2718/model.json'])
    stage.need(all(body['live_inputs'].get(k)==v for k,v in required_live.items()) and
        all(body['inputs'].get(k)==v for k,v in body['live_inputs'].items()),
        'Implementation live observer coverage differs')
    stage.common.current(body['inputs'])
    from . import provenance_reader_v2 as provenance
    from . import provenance_reconstruction_v2 as roles
    from . import ownership_handoff_worker as worker
    from . import ownership_handoff_launcher_v2 as launcher
    from . import prospective_reservations_v2 as reservations
    original=provenance.completed()['report'];graph=roles.completed()
    fixed=worker.completed()['report'];timed=launcher.completed(stage.FOLDER/'handoff/engineering-timed-v2')['report']
    reserved=reservations.completed()['selection']
    stage.need(original['known_material_source_coverage_complete'] is True and original['units']==354 and
        original['cartesian_flattened_edges_created']==0 and
        graph['support_roots_exactly_reused']==512 and graph['support_roots_with_genuine_exclusion']==320 and
        graph['support_training_admission'] is False,'Certified provenance admission scope differs')
    stage.need(fixed['passed'] and fixed['games']==9 and fixed['decisions']==144 and fixed['handoffs']==9 and
        fixed['pending_ownership_checks']==0 and timed['passed'] and timed['games']==2,
        'Live handoff acceptance differs')
    stage.need(reserved['metadata_reservations_complete'] is True and reserved['training_admission'] is False and
        reserved['evaluation_admission'] is False and reserved['historically_unseen_claim'] is False and
        reserved['opening_reachability_claimed'] is False and
        len(reserved['roles']['fit'])==128 and len(reserved['roles']['arena'])==24 and
        len(reserved['roles']['cases'])==32 and len(reserved['selected_components'])==152 and
        reserved['support_pool_changed'] is False,'Joint reservation acceptance differs')
    audit=stage.read(stage.ROOT/body['independent_audit'])
    stage.need(audit['schema']=='ownership-implementation-independent-audit-v1' and audit['passed'] is True and
        audit['independent_of_implementation'] is True and audit['unresolved_findings']==[] and
        audit['completed_reader_no_engine'] is True and body['independent_audit'] in body['inputs'] and
        all(audit['methods'].get(k,{}).get('passed') is True and audit['methods'][k]['evidence'] for k in METHODS) and
        all(audit['recovery_counts'].get(k,0)>=v for k,v in
            dict(provenance_kills=3,worker_kills=5,reservation_kills=3).items()),
        'Independent implementation audit coverage failed')
    for group in METHODS:
        for evidence in audit['methods'][group]['evidence']:
            stage.need(body['inputs'].get(evidence['path'])==evidence['sha256'],
                'Independent method evidence omitted from closure')
    stage.need(all(body['inputs'].get(k)==v for k,v in audit['source_inputs'].items()),
        'Independent source evidence omitted from closure')
    eventpath=stage.path_of(stage.ROOT/body['progress_event'],'events');event=stage.read(eventpath)
    stage.need(eventpath==sorted((stage.FOLDER/'events').glob('*.json'))[-1] and
        body['progress_event_sha256']==body['inputs'].get(body['progress_event'])==stage.sha(eventpath) and
        event['schema']=='ownership-implementation-progress-v1' and
        event['acceptance_schema']==SCHEMA and event['baseline_sha256']==stage.sha(stage.FOLDER/'baseline.json') and
        event['training_counters']==stage.TRAINING_ZERO and event['models_fixed'] is True and event['policy_fixed'] is True and
        event['execution_ready'] is False and event['training_authorized'] is False and
        event['models_sha256']==stage.common.pins(['data/quality-train-v1/model.json',
            'data/value-training-v1/training/model.json','data/training-stage5-deeper-h8-f127-seed2718/model.json']),
        'Implementation final dashboard binding differs')
    return body

def completed():return validate(stage.read(PATH))

def main():
    body=completed()
    print(json.dumps(dict(passed=True,inputs=len(body['inputs']),training_counters=stage.TRAINING_ZERO,
        metadata_fit_roots=128,metadata_arena_origins=24,support_roots_on_hold=320,
        execution_ready=False,training_authorized=False),sort_keys=True),flush=True)

if __name__=='__main__':main()
