"""Preserve accepted history and close pre-training interface evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak import neurodiag_common as common
from congklak import valuepilot_interface_progress as progress

FOLDER = ROOT / 'build/valuepilot-interface-v1'


def baseline():
    old = common.read(ROOT / 'build/valuepilot-plan-v1/acceptance.json')
    common.require(old['passed'] is True and old['scope'] == 'planning_only', 'Previous plan not accepted')
    inputs = dict(old['inputs'])
    inputs.update(common.pins(['build/valuepilot-plan-v1/acceptance.json']))
    common.current(inputs)
    return common.publish(FOLDER / 'baseline.json', dict(
        schema='valuepilot-interface-prior-v1', passed=True, inputs_count=len(inputs), inputs=inputs))


def listener():
    script = "$connections=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($connections | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine} }) | ConvertTo-Json -Compress"
    row = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                         check=True, text=True, timeout=30)
    rows = json.loads(row.stdout)
    if isinstance(rows, dict):
        rows = [rows]
    common.require(len(rows) == 1 and rows[0]['address'] == '127.0.0.1' and
                   '-m congklak.valuepilot_interface_progress serve --port 18766' in rows[0]['command'],
                   'Unexpected interface dashboard listener')
    startup = [json.loads(line) for line in (FOLDER / 'server.stdout.log').read_text(encoding='utf-8').splitlines()
               if line.startswith('{')]
    common.require(startup and startup[-1]['event'] == 'valuepilot_interface_observer_ready' and
                   startup[-1]['pid'] == rows[0]['pid'] and
                   startup[-1]['source_sha256'] == common.sha(ROOT / 'congklak/valuepilot_interface_progress.py'),
                   'Dashboard process has not loaded the reviewed observer source')
    return rows[0]


def build():
    """Close engineering evidence without turning a data HOLD into admission."""
    prior = common.read(FOLDER / 'baseline.json')
    common.require(prior['schema'] == 'valuepilot-interface-prior-v1' and prior['passed'] is True and
                   prior['inputs_count'] == len(prior['inputs']) == 5627, 'Historical baseline differs')
    common.current(prior['inputs'])
    audit = common.read(FOLDER / 'audit.json')
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
                   not audit['unresolved_findings'], 'Independent implementation audit not closed')
    common.current(audit['inputs'])
    common.require(audit['automated_test_methods'] and audit['process_kill_cases'], 'Executed audit checks missing')
    common.require(all(audit['counters'][key] == 0 for key in
                       ('training_steps', 'optimizer_draws', 'new_games', 'new_teacher_labels')),
                   'This acceptance is only for pre-training implementation')
    from congklak import valuepilot_reservations as data
    manifest = data.verify()
    state = data.status()
    # The current historical corpus cannot fill the accepted ordinary arena
    # quota. A later changed corpus needs a separate acceptance and conformance
    # on the actual admitted states, rather than reusing this engineering seal.
    common.require(manifest.get('admitted') is False and state['status'] == 'hold',
                   'Expected audited data HOLD changed; do not silently admit a different pilot')
    blockers = audit['readiness_blockers']
    common.require(blockers and state.get('readiness_blockers'), 'Missing explicit readiness blocker')
    common.require(not (ROOT / 'data/neural-value-pilot-v1/interface-initial/model.json').exists(),
                   'Production initialization must remain blocked by data admission')
    live = listener()
    status = common.read(FOLDER / 'stage-status.json')
    common.require(status['state'] == 'hold' and status['execution_counters'] == audit['counters'] and
                   status['committed']['training_roots'] == status['committed']['fit_check_roots'] ==
                   status['committed']['arena_components'] == status['committed']['value_exports'] == 0,
                   'Dashboard overstates admitted data or production initialization')
    common.require([r['state'] for r in status['cards']] ==
                   ['complete', 'hold', 'complete', 'planned', 'planned', 'planned', 'planned', 'complete', 'complete'],
                   'Dashboard stage states differ from pre-training HOLD')
    new = dict(audit['inputs'])
    new.update(common.pins([
        'tools/verify-valuepilot-interface.py', 'congklak/valuepilot_interface_progress.py',
        'tools/valuepilot-interface-dashboard.html', 'tests/test_valuepilot_interface_progress.py',
        'build/valuepilot-interface-v1/baseline.json', 'build/valuepilot-interface-v1/audit.json',
        'build/valuepilot-interface-v1/stage-status.json', 'build/neurodiag-dashboard-final.json']))
    inputs = dict(prior['inputs'])
    common.require(all(name not in inputs or inputs[name] == digest for name, digest in new.items()),
                   'Reviewed inputs conflict with accepted history')
    inputs.update(new)
    # Keep the live dashboard cheap: raw crash fixtures remain fully bound in
    # inputs and checked by verify(), rather than rehashing hundreds of MB at
    # every five-second browser poll. Sources, models and closure certificates
    # are checked live; this is not a substitute for full acceptance verify.
    live_names = {
        'build/valuepilot-interface-v1/audit.json', 'build/valuepilot-interface-v1/stage-status.json',
        'build/neurodiag-dashboard-final.json', 'build/value-sidecar-build.json',
        'build/congklak_value_sidecar.dll', 'data/quality-train-v1/model.json',
        'data/neural-value-pilot-v1/reservations/identity.json',
        'data/neural-value-pilot-v1/reservations/manifest.json',
        'data/neural-value-pilot-v1/reservations/catalog.json',
        'data/neural-value-pilot-v1/reservations/certificate.pending.json'}
    extensions = {'.py', '.cpp', '.hpp', '.ps1', '.cmd', '.html', '.md'}
    fast = {k: v for k, v in new.items() if k in live_names or Path(k).suffix in extensions}
    result = common.publish(FOLDER / 'acceptance.json', dict(
        schema='valuepilot-interface-acceptance-v1', passed=True,
        scope='pre_training_implementation_with_data_hold', training_ready=False,
        readiness_blockers=blockers,
        next_step='Interface and recovery checks passed. Training is held: eight clean ordinary arena components are required and none remain in the existing eligible development pool. Add and audit separate ordinary development starts, then rerun admission and conformance on the actual reserved corpus before training.',
        previous_inputs_preserved=len(prior['inputs']), implementation_accepted=True,
        data_status='hold', actual_optimizer_memberships_frozen=False,
        training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0,
        models_and_policy_fixed=True, promotion=False,
        independent_test_methods=len(audit['automated_test_methods']),
        process_kill_cases=len(audit['process_kill_cases']),
        production_value_exports=0, listener=live,
        backend_and_OS_listener_checked=True, browser_or_HTTP_inspection=False,
        new_inputs=fast, inputs=dict(sorted(inputs.items()))))
    common.require(progress.snapshot()['implementation_accepted'] is True,
                   'Observer did not recognize bound implementation evidence')
    return result


def verify():
    body = common.read(FOLDER / 'acceptance.json')
    common.require(body['schema'] == 'valuepilot-interface-acceptance-v1' and body['passed'] is True,
                   'No accepted interface package')
    common.current(body['inputs'])
    state = progress.snapshot()
    common.require(state['implementation_accepted'] and state['current_work']['state'] == ('complete' if body['training_ready'] else 'hold') and
                   state['pilot_training_ready'] == body['training_ready'] and
                   state['execution_counters'] == dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0),
                   'Dashboard implementation/training status differs')
    common.require('torch' not in sys.modules, 'Completed acceptance verification imported training runtime')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('baseline', 'build', 'verify'))
    args = parser.parse_args()
    result = {'baseline': baseline, 'build': build, 'verify': verify}[args.operation]()
    print(json.dumps(dict(event='valuepilot_interface_' + args.operation, passed=True,
                          inputs_count=len(result['inputs'])), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
