"""Seal the bounded trainer implementation; never optimize or run strength games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import neurodiag_common as common
from congklak import value_training_progress as progress

FOLDER = ROOT / 'build/value-training-v1'
ROOT_FILES = ('tools/verify-value-training.py', 'congklak/value_training_progress.py',
              'tools/value-training-dashboard.html', 'tests/test_value_training_progress.py',
              'docs/RECOVERABLE_VALUE_TRAINING.md')


def listener():
    script = "$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, check=True, timeout=30)
    rows = json.loads(result.stdout)
    if isinstance(rows, dict):
        rows = [rows]
    common.require(len(rows) == 1 and rows[0]['address'] == '127.0.0.1' and
                   '-m congklak.value_training_progress serve --port 18766' in rows[0]['command'],
                   'Unexpected dashboard listener')
    startup = [json.loads(line) for line in (FOLDER / 'server.stdout.log').read_text().splitlines()
               if line.startswith('{')]
    common.require(startup and startup[-1]['event'] == 'value_training_observer_ready' and
                   startup[-1]['pid'] == rows[0]['pid'] and
                   startup[-1]['source_sha256'] == common.sha(ROOT / 'congklak/value_training_progress.py'),
                   'Dashboard process did not load the reviewed source')
    return rows[0]


def build():
    prior = common.read(FOLDER / 'baseline.json')
    common.require(prior['passed'] is True and prior['inputs_count'] == len(prior['inputs']) == 7848,
                   'Previous immutable input baseline differs')
    common.current(prior['inputs'])
    audit = common.read(FOLDER / 'audit.json')
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
                   not audit['unresolved_findings'] and audit['automated_test_methods'] and audit['process_kill_cases'],
                   'Independent trainer audit is not closed')
    common.require(audit['production_counters'] == progress.ZERO, 'Unexpected production work')
    common.current(audit['inputs'])
    from congklak import value_training_data as data, value_training as training
    cache = data.admit()
    common.require(training.status()['state'] == 'not_started' and not training.DEFAULT.exists(),
                   'Production training was initialized during implementation acceptance')
    common.require(cache['labels_read'] == 640 and cache['optimizer_steps'] == cache['model_forwards'] == 0,
                   'Training cache activity differs')
    status = common.read(FOLDER / 'stage-status.json')
    common.require(status['state'] == 'complete' and status['production_counters'] == progress.ZERO and
                   status['engineering_counters'] == audit['engineering_counters'] and
                   [r['state'] for r in status['cards']] == ['complete'] * 5 + ['planned'] * 4 + ['complete'],
                   'Dashboard implementation state or counters differ')
    network = listener()
    new = dict(audit['inputs'])
    new.update(common.pins(ROOT_FILES))
    # Preserve the actual cache and every completed/failed new-stage evidence
    # file. Never bind the acceptance to itself or transient bytecode/locks.
    paths = []
    for directory in (FOLDER, data.DEFAULT):
        for path in directory.rglob('*'):
            if (path.is_file() and path.name not in ('acceptance.json', 'writer.lock') and
                    '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pending') and
                    not path.name.endswith(('-shm', '-wal'))):
                paths.append(path)
    new.update(common.pins(paths))
    new.update(common.pins(['build/ordinary-development-v1/stage-status.json']))
    inputs = dict(prior['inputs'])
    common.require(all(k not in inputs or inputs[k] == v for k, v in new.items()), 'Accepted history changed')
    inputs.update(new)
    source_suffixes = {'.py', '.cpp', '.hpp', '.ps1', '.cmd', '.html', '.md'}
    live_names = {'build/value-training-v1/audit.json', 'build/value-training-v1/stage-status.json',
                  'build/value-training-v1/root-checks.json', 'build/congklak_value_sidecar.dll',
                  'build/value-sidecar-build.json', 'data/quality-train-v1/model.json',
                  'data/neural-value-pilot-v2/interface-initial/model.json',
                  'data/neural-value-pilot-v2/reservations/manifest.json',
                  'build/ordinary-development-v1/acceptance.json',
                  'build/ordinary-development-v1/stage-status.json'}
    live_names.update('data/value-training-v1/cache/' + n for n in
                      ('identity.json', 'catalog.json', 'manifest.json', 'certificate.json', 'complete.json'))
    live = {k: v for k, v in inputs.items() if k in live_names or (k in new and Path(k).suffix in source_suffixes)}
    result = common.publish(FOLDER / 'acceptance.json', dict(
        schema='value-training-implementation-acceptance-v1', passed=True, implementation_ready=True,
        previous_inputs_preserved=7848, production_counters=progress.ZERO,
        engineering_counters=audit['engineering_counters'], unresolved_findings=[],
        training_optimizer_recovery_verified=True, resource_watchdog_verified=True,
        policy_and_original_weights_fixed=True, training_parameters=2065, frozen_parameters=12872,
        training_roots=512, fit_check_roots=128, production_target_steps=400,
        production_target_draws=51200, production_training_started=False,
        independent_test_methods=len(audit['automated_test_methods']),
        accepted_recovery_cases=len(audit['process_kill_cases']), strength_verified=False,
        promotion=False, baseline_default='H_v1', listener=network, browser_or_HTTP_inspection=False,
        next_operation='Run the accepted 400-step CPU FP32 value-only pilot through the durable resource supervisor, then fit, endgame and both-baseline strength gates.',
        next_step='The recoverable value-training implementation is accepted. Production remains 0/400 steps. Next run the fixed value-only pilot, then the planned fit, endgame and equal-time tests against both baselines.',
        inputs=dict(sorted(inputs.items())), live_inputs=live))
    common.require(progress.snapshot()['accepted'] is True, 'Dashboard did not accept implementation closure')
    # Exercise the real production preflight without creating a training path.
    gate = training.production_gate()
    common.require(gate['sha256'] == common.sha(FOLDER / 'acceptance.json') and not training.DEFAULT.exists(),
                   'Production gate binding or zero-work condition differs')
    common.require('torch' not in sys.modules, 'Closure imported a training runtime')
    return result


def verify():
    body = common.read(FOLDER / 'acceptance.json')
    common.require(body['passed'] is True and body['schema'] == 'value-training-implementation-acceptance-v1' and
                   body['implementation_ready'] is True and body['production_counters'] == progress.ZERO,
                   'No accepted trainer implementation')
    common.current(body['inputs'])
    state = progress.snapshot()
    common.require(state['accepted'] is True and state['current_work']['state'] == 'complete',
                   'Dashboard implementation state differs')
    common.require('torch' not in sys.modules, 'Completed verification imported Torch')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'verify'))
    result = dict(build=build, verify=verify)[parser.parse_args().operation]()
    print(json.dumps(dict(event='value_training_implementation_accepted', passed=True,
                          inputs_count=len(result['inputs']), production_steps=0), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
