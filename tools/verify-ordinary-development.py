"""Seal admitted ordinary development data and inference; never train."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import neurodiag_common as common
from congklak import ordinary_development_progress as progress

FOLDER = ROOT / 'build/ordinary-development-v1'


def listener():
    script = "$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, check=True, timeout=30)
    rows = json.loads(result.stdout)
    if isinstance(rows, dict):
        rows = [rows]
    common.require(len(rows) == 1 and rows[0]['address'] == '127.0.0.1' and
                   '-m congklak.ordinary_development_progress serve --port 18766' in rows[0]['command'],
                   'Unexpected dashboard listener')
    startup = [json.loads(line) for line in (FOLDER / 'server.stdout.log').read_text().splitlines()
               if line.startswith('{')]
    common.require(startup and startup[-1]['event'] == 'ordinary_development_observer_ready' and
                   startup[-1]['pid'] == rows[0]['pid'] and
                   startup[-1]['source_sha256'] == common.sha(ROOT / 'congklak/ordinary_development_progress.py'),
                   'Dashboard process did not load the reviewed source')
    return rows[0]


def baseline():
    previous = common.read(ROOT / 'build/valuepilot-interface-v1/acceptance.json')
    common.require(previous['passed'] is True, 'Previous implementation was not accepted')
    inputs = dict(previous['inputs'])
    inputs.update(common.pins(['build/valuepilot-interface-v1/acceptance.json']))
    common.current(inputs)
    return common.publish(FOLDER / 'baseline.json', dict(schema='ordinary-development-prior-v1',
                          passed=True, inputs_count=len(inputs), inputs=inputs))


def build():
    prior = common.read(FOLDER / 'baseline.json')
    common.require(prior['passed'] is True and prior['inputs_count'] == len(prior['inputs']) == 5950,
                   'Previous input baseline differs')
    common.current(prior['inputs'])
    audit = common.read(FOLDER / 'audit.json')
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
                   not audit['unresolved_findings'] and audit['automated_test_methods'] and audit['process_kill_cases'],
                   'Independent audit is not closed')
    common.current(audit['inputs'])
    common.require(audit['counters'] == progress.ZERO, 'Unexpected optimizer, label or game execution')
    from congklak import ordinary_development as generation
    from congklak import valuepilot_reservations_v2 as reservations
    from congklak import value_sidecar_v2 as sidecar
    generation.admit()
    manifest = reservations.admit()
    common.require(manifest['admitted'] is True and manifest['status'] == 'reserved' and
                   manifest['counts'] == dict(train=512, fit_check=128, fit_cases=32, arena=24, globally_distinct_components=664),
                   'Joint memberships are not admitted')
    initial = sidecar.verify(ROOT / 'data/neural-value-pilot-v2/interface-initial',
                             reservations.DEFAULT / 'manifest.json')
    common.require(initial['state'] == 'complete' and initial['optimizer_steps'] == 0 and
                   initial['training_eligible'] is True and initial['engineering_fixture'] is False,
                   'Production step-zero export not admitted')
    status = common.read(FOLDER / 'stage-status.json')
    common.require(status['state'] == 'complete' and status['execution_counters'] == progress.ZERO and
                   status['committed'] == dict(training_roots=512, fit_check_roots=128, arena_components=24, value_exports=1),
                   'Dashboard committed counts differ')
    common.require(status['generation_counters']['candidate_pool_committed'] == 32 and
                   status['generation_counters']['selected_ordinary_arena_components'] == 8 and
                   [r['state'] for r in status['cards']] == ['complete'] * 4 + ['planned'] * 4 + ['complete'] * 2,
                   'Dashboard stage readiness differs')
    network = listener()
    new = dict(audit['inputs'])
    new.update(common.pins([
        'tools/verify-ordinary-development.py', 'congklak/ordinary_development_progress.py',
        'tools/ordinary-development-dashboard.html', 'tests/test_ordinary_development_progress.py',
        'build/ordinary-development-v1/baseline.json', 'build/ordinary-development-v1/audit.json',
        'build/ordinary-development-v1/stage-status.json',
        'build/valuepilot-interface-v1/stage-status.json', 'build/neurodiag-dashboard-final.json']))
    inputs = dict(prior['inputs'])
    common.require(all(k not in inputs or inputs[k] == v for k, v in new.items()), 'Accepted history changed')
    inputs.update(new)
    sources = {'.py', '.cpp', '.hpp', '.ps1', '.cmd', '.html', '.md'}
    live_names = {
        'build/ordinary-development-v1/audit.json', 'build/ordinary-development-v1/stage-status.json',
        'build/ordinary-development-v1/conformance-owner.json',
        'build/congklak_value_sidecar.dll', 'build/value-sidecar-build.json',
        'data/quality-train-v1/model.json',
        'build/valuepilot-interface-v1/stage-status.json', 'build/neurodiag-dashboard-final.json'}
    for directory in ('data/ordinary-development-v1', 'data/neural-value-pilot-v2/reservations',
                      'data/neural-value-pilot-v2/interface-initial'):
        for filename in ('identity.json', 'manifest.json', 'catalog.json', 'certificate.json',
                         'certificate.pending.json', 'model.json', 'COMPLETED.json'):
            live_names.add(directory + '/' + filename)
    live = {k: v for k, v in inputs.items() if k in live_names or (k in new and Path(k).suffix in sources)}
    result = common.publish(FOLDER / 'acceptance.json', dict(
        schema='ordinary-development-acceptance-v1', passed=True, data_and_inference_ready=True,
        scope='synthetic_ordinary_development_joint_admission_and_stepzero_conformance_before_training',
        previous_inputs_preserved=5950, candidate_pool=32, new_ordinary_arena_components=8,
        joint_counts=manifest['counts'], production_stepzero_exports=1,
        training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0,
        policy_and_original_weights_fixed=True, promotion=False, baseline_default='H_v1',
        independent_test_methods=len(audit['automated_test_methods']),
        accepted_recovery_cases=len(audit['process_kill_cases']),
        synthetic_custom_starts=True, standard_opening_reachability_claim=False,
        training_optimizer_recovery_verified=False, strength_verified=False,
        next_operation='Implement the fixed 400-step value trainer with optimizer/RNG checkpoints, resource leases and training-process recovery fixtures before production optimization.',
        next_step='The ordinary arena shortage is resolved. Joint data reservations, actual-corpus native conformance and step-zero export are accepted. Training remains off; the recoverable fixed training pilot is next.',
        listener=network, browser_or_HTTP_inspection=False,
        live_inputs=live, inputs=dict(sorted(inputs.items()))))
    common.require(progress.snapshot()['accepted'] is True, 'Dashboard did not accept closure')
    common.require('torch' not in sys.modules, 'Acceptance closure imported a training runtime')
    return result


def verify():
    body = common.read(FOLDER / 'acceptance.json')
    common.require(body['passed'] is True and body['schema'] == 'ordinary-development-acceptance-v1' and
                   body['data_and_inference_ready'] is True, 'No accepted admission package')
    common.current(body['inputs'])
    state = progress.snapshot()
    common.require(state['accepted'] is True and state['execution_counters'] == progress.ZERO and
                   state['current_work']['state'] == 'complete', 'Dashboard admission state differs')
    common.require('torch' not in sys.modules, 'Completed verification imported Torch')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('baseline', 'build', 'verify'))
    args = parser.parse_args()
    result = dict(baseline=baseline, build=build, verify=verify)[args.operation]()
    print(json.dumps(dict(event='ordinary_development_' + args.operation, passed=True,
                          inputs_count=len(result['inputs'])), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
