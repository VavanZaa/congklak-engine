"""Seal the complete value-pilot evidence independently of a positive outcome."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import neurodiag_common as common, value_run_progress_v3 as progress

FOLDER = ROOT / 'build/value-run-v1'
ROOT_FILES = ('tools/verify-value-run.py', 'congklak/value_run_progress_v3.py',
              'tools/value-run-dashboard-v3.html', 'congklak/value_run_reader.py', 'tests/test_value_run_reader.py', 'tests/test_value_run_progress_v3.py',
              'congklak/value_run_decision.py', 'tools/decide-value-run.py',
              'docs/VALUE_RUN_RESULTS.md')


def listener():
    script = "$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); $proc=Get-Process -Id $_.OwningProcess; [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine;affinity=[long]$proc.ProcessorAffinity} }) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, check=True, timeout=30)
    rows = json.loads(result.stdout)
    if isinstance(rows, dict): rows = [rows]
    common.require(len(rows) == 1 and rows[0]['address'] == '127.0.0.1' and rows[0]['affinity'] == 1 and
                   '-m congklak.value_run_progress_v3 serve --port 18766' in rows[0]['command'],
                   'Unexpected dashboard listener or affinity')
    startup = [json.loads(line) for line in (FOLDER / 'server-v3.stdout.log').read_text().splitlines() if line.startswith('{')]
    common.require(startup and startup[-1]['event'] == 'value_run_observer_v3_ready' and
                   startup[-1]['pid'] == rows[0]['pid'] and
                   startup[-1]['source_sha256'] == common.sha(ROOT / 'congklak/value_run_progress_v3.py'),
                   'Dashboard did not load the reviewed source')
    return rows[0]


def build():
    prior = common.read(FOLDER / 'baseline.json')
    common.require(prior['passed'] is True and prior['inputs_count'] == len(prior['inputs']) == 8898,
                   'Previous immutable baseline differs')
    common.current(prior['inputs'])
    audit = common.read(FOLDER / 'audit.json')
    counters = dict(training_steps=400, optimizer_draws=51200, new_games=192, new_teacher_labels=0)
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
                   audit['unresolved_findings'] == [] and audit['automated_test_methods'] and audit['process_kill_cases'],
                   'Independent evidence audit is not closed')
    common.require(audit['production_counters'] == counters, 'Actual production counters differ')
    common.current(audit['inputs'])
    from congklak import value_run_decision as decision, value_training as training
    report = decision.run()
    result = training.completed()
    common.require(report['evidence_complete'] and result['steps'] == 400 and result['draws'] == 51200 and
                   not result['engineering_fixture'] and report['promotion'] is False, 'Fixed production evidence differs')
    state = common.read(FOLDER / 'stage-status.json')
    common.require(state['state'] == 'complete' and state['engineering_counters'] == audit['engineering_counters'],
                   'Dashboard final audit counters differ')
    network = listener()
    inputs = dict(prior['inputs'])
    new = dict(audit['inputs'])
    new.update(common.pins(ROOT_FILES))
    files = []
    for directory in (FOLDER, ROOT / 'data/value-run-v1', ROOT / 'data/value-training-v1/training', ROOT / 'data/value-training-v1/resources'):
        for path in directory.rglob('*'):
            if (path.is_file() and path != FOLDER / 'acceptance.json' and path.name != 'writer.lock' and
                    '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pending') and
                    not path.name.endswith(('-shm', '-wal'))): files.append(path)
    new.update(common.pins(files))
    common.require(all(k not in inputs or inputs[k] == value for k, value in new.items()), 'Accepted history changed')
    inputs.update(new)
    live_names = {'build/value-run-v1/audit.json', 'build/value-run-v1/decision.json',
        'build/value-run-v1/stage-status.json', 'build/value-run-v1/production-complete.json',
        'build/value-run-v1/cases-fit-v2.json', 'build/value-run-v1/cases-conformance-v2.json',
        'build/value-run-v1/cases/certificate.json', 'build/value-run-v1/cases/catalog.json',
        'build/value-run-v1/cases/report.json', 'data/value-run-v1/arena/certificate.json',
        'data/value-run-v1/arena/head.json', 'data/value-run-v1/arena/result.json',
        'data/value-training-v1/training/certificate.json', 'data/value-training-v1/training/model.json',
        'data/quality-train-v1/model.json'}
    suffixes = {'.py', '.cpp', '.hpp', '.ps1', '.cmd', '.html', '.md'}
    live = {key: value for key, value in inputs.items() if key in live_names or (key in new and Path(key).suffix in suffixes)}
    message = ('The complete fixed value pilot is accepted experimental evidence. ' +
        ('It passed the predeclared research gates; strength intervals remain exploratory.' if report['pilot_promising'] else
         'It failed predeclared research gates: ' + ', '.join(report['failed_gates']) + '. The control is retained.'))
    next_operation = ('Review the retained ordinary/capture errors and search failures using existing data before specifying another treatment. No automatic retraining or data expansion.')
    body = common.publish(FOLDER / 'acceptance.json', dict(schema='value-run-acceptance-v1', passed=True,
        evidence_accepted=True, pilot_promising=report['pilot_promising'],
        candidate_quarantined=report['candidate_quarantined'], strength_interpretation_allowed=report['strength_interpretation_allowed'],
        failed_gates=report['failed_gates'], production_counters=counters, engineering_counters=audit['engineering_counters'],
        previous_inputs_preserved=8898, frozen_policy=True, promotion=False, default_engine='H_v1',
        independent_test_methods=len(audit['automated_test_methods']), actual_recovery_cases=len(audit['process_kill_cases']),
        unresolved_findings=[], message=message, next_operation=next_operation,
        listener=network, browser_or_HTTP_inspection=False, inputs=dict(sorted(inputs.items())), live_inputs=live))
    common.require(progress.snapshot()['accepted'] is True, 'Dashboard did not accept complete pilot evidence')
    common.require('torch' not in sys.modules, 'Final closure imported Torch')
    return body


def verify():
    body = common.read(FOLDER / 'acceptance.json')
    common.require(body['schema'] == 'value-run-acceptance-v1' and body['passed'] is True and body['evidence_accepted'] is True,
                   'No accepted full pilot evidence')
    common.current(body['inputs'])
    state = progress.snapshot()
    common.require(state['accepted'] and state['production']['steps'] == 400 and state['production']['draws'] == 51200,
                   'Dashboard production or evidence state differs')
    common.require('torch' not in sys.modules, 'Completed evidence verification imported Torch')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'verify'))
    result = dict(build=build, verify=verify)[parser.parse_args().operation]()
    print(json.dumps(dict(passed=result['passed'], evidence_accepted=True, pilot_promising=result['pilot_promising'],
                          inputs_count=len(result['inputs']), failed_gates=result['failed_gates']), sort_keys=True), flush=True)


if __name__ == '__main__': main()
