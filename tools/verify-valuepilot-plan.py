"""Seal planning evidence and a read-only dashboard; execute no experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import neurodiag_common as common,valuepilot_plan_progress as progress
FOLDER=ROOT/'build/valuepilot-plan-v1'
OUTPUT=FOLDER/'acceptance.json'
NEW=('docs/NEURAL_VALUE_PILOT_PLAN.md','docs/NEURAL_VALUE_PILOT_PLAN_AUDIT.md',
    'build/valuepilot-plan-v1/plan.json','build/valuepilot-plan-v1/feasibility.json','build/valuepilot-plan-v1/prior.json',
    'build/valuepilot-plan-v1/audit.json','congklak/valuepilot_plan_progress.py','tools/valuepilot-plan-dashboard.html',
    'tools/prepare-valuepilot-plan.py','tools/verify-valuepilot-plan.py')


def checks():
    p=common.read(FOLDER/'plan.json');a=common.read(FOLDER/'audit.json');f=common.read(FOLDER/'feasibility.json')
    prior=common.read(FOLDER/'prior.json');require=common.require
    require(p['schema']=='neural-value-pilot-plan-v1' and p['training_started'] is p['implementation_started'] is False,'Plan execution status changed')
    require(p['target']['kind']=='approximate_H8_root_margin' and p['target']['requested_depth']==p['target']['completed_depth']==8
        and p['target']['mixed_targets'] is False and p['target']['new_teacher_labels']==0,'Target contract differs')
    d=p['data'];t=p['training'];arch=p['architecture'];c=p['cases'];arena=p['arena']
    require(len(d['cells'])==8 and d['train']==8*d['train_per_cell']==512 and d['fit_check']==8*d['fit_check_per_cell']==128
        and d['globally_distinct_components']==d['train']+d['fit_check']==640,'Data quota arithmetic differs')
    branch=16*127+16+16+1
    require(arch['trainable_parameters']==branch==2065 and arch['total_parameters']==arch['frozen_base_parameters']+branch==14937
        and arch['frozen_base_tensors']==8 and arch['gate_active_inclusive']==[25,74]
        and arch['additional_fp32_weight_bytes']==branch*4==8260 and arch['total_dense_MAC_in_gate']==arch['base_dense_MAC']+arch['sidecar_dense_MAC']==14784,'Architecture arithmetic differs')
    require(t['steps']==t['selected_step']==400 and t['batch_size']==128 and t['draws']==t['steps']*t['batch_size']==51200
        and t['scheduled_training_checkpoints']==t['steps']//t['checkpoint_every']==20 and t['device']=='cpu'
        and t['metric_selected_checkpoint'] is t['GPU_in_scope'] is False,'Fixed training contract differs')
    require(arena['games']==arena['families']*len(arena['profiles'])*len(arena['baselines'])*arena['seats']==192
        and arena['globally_distinct_components']==arena['bootstrap']['clusters']==arena['families']==24
        and set(arena['baselines'])=={'H_v1','H8_relay_CPU'},'Arena cluster/game arithmetic differs')
    require(c['timed_measurements']==(c['fit_H8']+c['known_exact'])*len(c['engines'])*len(c['budgets_ms'])==512
        and c['fixed_sim_searches']==(c['fit_H8']+c['known_exact']+c['reused_Hv1_divergences'])*c['fixed_sim_profiles']==138
        and c['requested_fixed_backups']==c['fixed_sim_searches']*c['fixed_backups']==17664,'Case arithmetic differs')
    require(a['schema']=='neural-value-pilot-independent-plan-audit-v1' and a['passed'] is True
        and a['independent_of_design_and_implementation'] is True and a['static_review_item_count']==len(a['static_review_items'])==22
        and all(v is True for v in a['static_review_items'].values()) and not a['unresolved_blocking_findings'],'Independent plan review incomplete')
    require(all(value==0 for key,value in a['actual_work_this_audit'].items() if key!='metadata_worksheet_reexecuted_by_auditor')
        and all(a[key] is False for key in ('implementation_verified','training_verified','recovery_verified','strength_verified','numerical_conformance_verified')),
        'Static review was misrepresented as execution')
    require(f['passed'] is True and f['requested_components']==f['matching_components']==640
        and f['matched_by_cell']=={cell:80 for cell in d['cells']} and f['actual_selection_published'] is f['new_reservations'] is False
        and all(f[key]==0 for key in ('target_bodies_read','model_forwards','searches','optimizer_steps','new_games')),'Metadata worksheet differs')
    require(prior['passed'] is True and prior['inputs_count']==len(prior['inputs'])==5616,'Preceding protected count differs')
    common.current(a['inputs']);common.current(f['inputs']);common.current(prior['inputs'])
    require(common.sha(ROOT/p['source_model'])==p['source_sha256'],'Fixed control model changed')
    return p,a,f,prior


def listener():
    script="$connections=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($connections | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine} }) | ConvertTo-Json -Compress"
    r=subprocess.run(['powershell','-NoProfile','-Command',script],check=True,capture_output=True,text=True,timeout=30)
    rows=json.loads(r.stdout)
    if isinstance(rows,dict):rows=[rows]
    common.require(len(rows)==1 and rows[0]['address']=='127.0.0.1'
        and '-m congklak.valuepilot_plan_progress serve --port 18766' in rows[0]['command'],'Planning dashboard listener differs')
    return rows[0]


def build():
    p,a,f,prior=checks()
    common.require(not any((ROOT/name).exists() for name in p['planned_future_namespaces']),'Experiment artifacts already exist; planning view cannot claim zero work')
    network=listener();inputs=dict(prior['inputs']);inputs.update(a['inputs']);new=common.pins(NEW)
    common.require(all(k not in inputs or inputs[k]==v for k,v in new.items()),'Conflicting source binding');inputs.update(new)
    result=common.publish(OUTPUT,dict(schema='neural-value-pilot-plan-acceptance-v1',passed=True,scope='planning_only',
        independent_static_review_items=22,automated_implementation_test_methods_executed=0,
        previous_inputs_preserved=5616,metadata_matching_components=640,actual_optimizer_memberships_frozen=False,
        implementation_started=False,training_steps=0,optimizer_draws=0,model_forwards=0,new_teacher_labels=0,new_games=0,process_kills=0,
        models_and_policy_fixed=True,promotion=False,backend_and_OS_listener_checked=True,browser_or_HTTP_inspection=False,
        listener=network,new_inputs=new,inputs=dict(sorted(inputs.items()))))
    state=progress.snapshot()
    common.require([row['state'] for row in state['tools'][:9]]==['complete']+['planned']*8
        and state['current_work']['state']=='complete' and state['execution_counters']==dict(training_steps=0,optimizer_draws=0,new_games=0,new_teacher_labels=0)
        and not state['active_tools'],'Dashboard overstates planning/execution progress')
    common.require('torch' not in sys.modules and 'congklak.quality_search' not in sys.modules,'Planning verification imported an execution runtime')
    return result


def verify():
    body=common.read(OUTPUT);common.require(body['passed'] is True and body['scope']=='planning_only','No accepted planning package')
    common.current(body['inputs']);checks();return body


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('build','verify'));args=parser.parse_args()
    result=build() if args.operation=='build' else verify()
    print(json.dumps(dict(event='valuepilot_plan_'+args.operation,passed=True,previous_inputs=result['previous_inputs_preserved'],
        independent_static_review_items=22,implementation_steps=0,training_steps=0,new_games=0),sort_keys=True),flush=True)
