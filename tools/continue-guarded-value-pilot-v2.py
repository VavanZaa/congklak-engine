"""Explicit restartable stages for the one fixed value-only experiment."""
import argparse,json,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import guarded_value_pilot_v2_common as s

def result(command):
    from congklak import guarded_value_pilot_v2_data as data,guarded_value_pilot_v2_sidecar as side
    from congklak import guarded_value_pilot_v2_resources as resources,guarded_value_pilot_v2_execute as execute
    if command=='freeze':
        from congklak.guarded_value_pilot_v2_plan import freeze
        b=freeze();return dict(plan_sha256=s.sha(s.PLAN),source_files=len(b['source_inputs']))
    with s.worker_views():
        if command=='prepare':
            from congklak.guarded_value_pilot_v2_guard import prepare_shared
            train=data.prepare_train();fit=data.prepare_fit();initial=side.initialize();guard=prepare_shared()
            return dict(train=train['role_files']['train']['count'],fit=fit['role_files']['fit']['count'],initial=initial,origins=len(guard['origins']))
        if command in ('training-recovery','handoff-recovery'):
            filename='training_recovery.py' if command=='training-recovery' else 'handoff_recovery.py'
            runpy.run_path(str(s.FOLDER/'audit'/filename),run_name='__main__')
            return dict(state='complete',stage=command)
        if command in ('replay-fixtures','replay-production'):
            fixture=command=='replay-fixtures';operation='verify_fixture_states' if fixture else 'verify_production_state'
            output=s.FOLDER/('audit/training-replay-fixtures' if fixture else 'audit/training-replay-production')
            return resources.launch(output,data.DEFAULT,entry=s.ROOT/'tools/audit-guarded-value-pilot-v2.py',extra_args=['--operation',operation])
        if command in ('initial','conformance','guards','arena'):
            fixture=command=='initial';operation='conformance' if fixture else command
            output=s.FOLDER/'jobs'/('initial-conformance' if fixture else command)
            return resources.launch(output,data.DEFAULT,fixture=fixture,entry=s.ROOT/'tools/guarded-value-pilot-v2-execute.py',extra_args=['--operation',operation])
        if command=='admit':
            initial=execute.admit_initial_conformance()
            recovery=s.read(s.FOLDER/'audit/training-recovery/result.json');replay=s.read(s.FOLDER/'audit/training-replay-fixtures/result.json')
            s.need(recovery['passed'] is True and recovery['actual_process_kills']==2 and replay['passed'] is True and replay['diagnostic_optimizer_steps']==18,'Independent checkpoint recovery is incomplete')
            inputs=s.common.pins([s.PREFLIGHT,s.FOLDER/'audit/training-recovery/result.json',s.FOLDER/'audit/training-replay-fixtures/result.json',
                s.FOLDER/'audit/data-check.json',s.FOLDER/'audit/shared-guard-check.json',s.FOLDER/'audit/native-component-check.json',
                s.FOLDER/'jobs/initial-conformance/catalog.json',s.FOLDER/'jobs/initial-conformance/ownership-prefix.json'])
            for name in ('data-check.json','shared-guard-check.json','native-component-check.json'):
                s.need(s.read(s.FOLDER/'audit'/name)['passed'] is True,'Preparation diagnostic failed')
            receipt=dict(schema=s.SCHEMA+'-preparation-admission',passed=True,plan_sha256=s.sha(s.PLAN),reservations_sha256=s.sha(s.PRIOR),
                cache_manifest_sha256=s.sha(data.DEFAULT/'manifest.json'),initial_model_sha256=s.sha(side.INITIAL/'model.json'),initial_conformance_passed=True,
                independent_checkpoint_recovery_passed=True,policy_fixed=True,production_optimizer_steps=0,inputs=inputs,initial_conformance=initial)
            return s.atomic(s.FOLDER/'preparation/admission.json',receipt)
        if command in ('train20','train'):
            from congklak.guarded_value_pilot_v2_train import DEFAULT
            return resources.launch(DEFAULT,data.DEFAULT,stop_after=20 if command=='train20' else None)
        if command=='fit':
            from congklak.guarded_value_pilot_v2_fit import run
            report=run();return dict(passed=report['passed'],checks=report['checks'],metrics=report['metrics']['overall'])
        if command in ('report','seal'):
            from congklak import guarded_value_pilot_v2_finish as finish
            report=finish.report() if command=='report' else finish.seal()
            return dict(state='complete',decision=report['decision'])
    raise ValueError(command)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','prepare','training-recovery','replay-fixtures','initial','admit','train20','train','replay-production','fit','conformance','guards','handoff-recovery','arena','report','seal'))
    args=parser.parse_args()
    try:print(json.dumps(result(args.stage),sort_keys=True),flush=True)
    except Exception as error:
        print(json.dumps(dict(state='hold',stage=args.stage,error=type(error).__name__+': '+str(error))),flush=True)
        raise
