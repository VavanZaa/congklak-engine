"""Accept a protocol definition, without admitting data or authorizing training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import neurodiag_common as common,ownership_protocol_flow_v2 as flow,ownership_protocol_progress_v2 as progress

FOLDER=flow.FOLDER
SOURCES=(*flow.SOURCES,'tools/verify-ownership-protocol-v2.py','congklak/ownership_protocol_progress_v2.py',
    'tools/ownership-protocol-dashboard-v2.html','tests/test_ownership_protocol_progress_v2.py',
    'tests/ownership_protocol_independent_audit.py','docs/OWNERSHIP_PROTOCOL_RESULTS.md',
    'docs/OWNERSHIP_PROTOCOL_AUDIT.md','docs/OWNERSHIP_PROTOCOL_RESULTS_V2.md',
    'congklak/ownership_protocol_flow.py','tools/run-ownership-protocol.py',
    'congklak/ownership_protocol_progress.py','tools/ownership-protocol-dashboard.html',
    'tests/test_ownership_protocol_progress.py','tools/verify-ownership-protocol.py')


def listener():
    script="$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); $proc=Get-Process -Id $_.OwningProcess; [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine;affinity=[long]$proc.ProcessorAffinity} }) | ConvertTo-Json -Compress"
    done=subprocess.run(['powershell','-NoProfile','-Command',script],capture_output=True,text=True,check=True,timeout=30)
    rows=json.loads(done.stdout)
    if isinstance(rows,dict):rows=[rows]
    common.require(len(rows)==1 and rows[0]['address']=='127.0.0.1' and rows[0]['affinity']==1 and
        '-m congklak.ownership_protocol_progress_v2 serve --port 18766' in rows[0]['command'],'Unexpected protocol listener')
    startup=[json.loads(line) for line in (FOLDER/'server-v2.stdout.log').read_text().splitlines() if line.startswith('{')]
    common.require(startup and startup[-1]['event']=='ownership_protocol_observer_v2_ready' and
        startup[-1]['pid']==rows[0]['pid'] and startup[-1]['source_sha256']==common.sha(ROOT/'congklak/ownership_protocol_progress_v2.py'),
        'Protocol observer did not load reviewed source')
    return rows[0]


def build():
    baseline=flow.baseline();production=flow.completed();audit=common.read(FOLDER/'audit.json')
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
        audit['unresolved_findings']==[] and len(audit['recovery_cases'])>=3 and audit['production_counters']==flow.ZERO,
        'Independent protocol audit not closed')
    common.current(audit['inputs']);state=progress.snapshot()
    common.require(state['current_work']['state']=='complete' and state['counters']==flow.ZERO and
        state['definition_only'] is True and state['execution_ready'] is False and state['training_authorized'] is False and
        state['live_suffix_worker_verified'] is False and state['new_reservations_made'] is False and
        state['confidence_intervals_enabled'] is False,'Protocol dashboard falsely implies execution readiness')
    network=listener();files=list(SOURCES)
    for path in FOLDER.rglob('*'):
        if path.is_file() and path!=FOLDER/'acceptance.json' and path.name!='writer.lock' and path.suffix not in ('.pyc','.pending') and '__pycache__' not in path.parts:
            files.append(path)
    inputs=dict(baseline['inputs']);new=dict(audit['inputs']);new.update(common.pins(files))
    common.require(all(k not in inputs or inputs[k]==v for k,v in new.items()),'Previously accepted history changed')
    inputs.update(new)
    live_names=set(SOURCES)|{'build/ownership-protocol-v1/audit.json',
        'build/ownership-protocol-v1/production-v2/catalog.json','build/ownership-protocol-v1/production-v2/identity.json',
        'build/ownership-protocol-v1/production-v2/plan.json','build/ownership-protocol-v1/production-v2/projection.json',
        'build/ownership-protocol-v1/production-v2/contract-checks.json',
        'data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
        'data/training-stage5-deeper-h8-f127-seed2718/model.json'}
    events=sorted((FOLDER/'events').glob('*.json'));event=events[-1].relative_to(ROOT).as_posix()
    live_names.update(p.relative_to(ROOT).as_posix() for p in events)
    body=dict(schema='prospective-ownership-acceptance-v2',passed=True,evidence_accepted=True,
        definition_only=True,execution_ready=False,training_authorized=False,data_admission=False,
        new_reservations_made=False,live_suffix_worker_verified=False,confidence_intervals_enabled=False,
        candidate_quarantined=True,old_arena_hold_preserved=True,strength_interpretation_allowed=False,
        models_fixed=True,policy_fixed=True,promotion=False,default_engine='H_v1',
        counters=flow.ZERO,previous_inputs_preserved=baseline['inputs_count'],
        committed_units=production['catalog']['committed_units'],actual_recovery_cases=len(audit['recovery_cases']),
        recovery_scope='protocol publication and contract fixtures; live switch worker remains pending',
        unresolved_findings=[],listener=network,browser_or_HTTP_inspection=False,progress_event=event,
        inputs=dict(sorted(inputs.items())),live_inputs={k:v for k,v in inputs.items() if k in live_names})
    flow.atomic(FOLDER/'acceptance.json',body)
    common.require(progress.snapshot()['accepted'],'Protocol observer did not accept sealed definition')
    common.require('torch' not in sys.modules,'Protocol closure imported Torch')
    return body


def verify():
    body=common.read(FOLDER/'acceptance.json')
    common.require(body['schema']=='prospective-ownership-acceptance-v2' and body['passed'] is True and
        body['definition_only'] is True and body['execution_ready'] is False and body['training_authorized'] is False and
        body['data_admission'] is False and body['live_suffix_worker_verified'] is False and
        body['confidence_intervals_enabled'] is False and body['counters']==flow.ZERO,'Protocol acceptance scope changed')
    common.current(body['inputs']);flow.completed();listener()
    common.require(progress.snapshot()['accepted'],'Protocol observer reports HOLD')
    common.require('torch' not in sys.modules,'Protocol verification imported Torch')
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('build','verify'))
    body=dict(build=build,verify=verify)[parser.parse_args().operation]()
    print(json.dumps(dict(passed=True,inputs=len(body['inputs']),prior_inputs=body['previous_inputs_preserved'],
        units=body['committed_units'],recovery_cases=body['actual_recovery_cases'],definition_only=True,
        execution_ready=False,training_authorized=False,counters=body['counters']),sort_keys=True),flush=True)


if __name__=='__main__':main()
