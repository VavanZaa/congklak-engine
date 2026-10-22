"""Seal the existing-data investigation without promoting the held candidate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import neurodiag_common as common, value_diagnostics as diagnostics
from congklak import value_diagnostics_progress as progress

FOLDER = diagnostics.FOLDER
SOURCES = ('tools/verify-value-diagnostics.py', 'congklak/value_diagnostics.py',
           'tools/run-value-diagnostics.py', 'congklak/value_capture_diagnostic.py',
           'congklak/value_bridge_diagnostic.py', 'congklak/value_diagnostics_progress.py',
           'tools/value-diagnostics-dashboard.html', 'tests/test_value_diagnostics_progress.py',
           'tests/test_value_capture_diagnostic.py', 'tests/test_value_bridge_diagnostic.py',
           'tests/value_diagnostics_independent_audit.py', 'docs/VALUE_DIAGNOSTICS_SCOPE.md',
           'docs/VALUE_DIAGNOSTICS_RESULTS.md', 'docs/VALUE_DIAGNOSTICS_AUDIT.md',
           'docs/VALUE_BRIDGE_DIAGNOSTICS.md')


def listener():
    script = "$rows=@(Get-NetTCPConnection -State Listen | Where-Object LocalPort -eq 18766); @($rows | ForEach-Object { $owner=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_.OwningProcess); $proc=Get-Process -Id $_.OwningProcess; [pscustomobject]@{address=$_.LocalAddress;port=$_.LocalPort;pid=$_.OwningProcess;command=$owner.CommandLine;affinity=[long]$proc.ProcessorAffinity} }) | ConvertTo-Json -Compress"
    process = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                             text=True, check=True, timeout=30)
    rows = json.loads(process.stdout)
    if isinstance(rows, dict): rows = [rows]
    common.require(len(rows) == 1 and rows[0]['address'] == '127.0.0.1' and rows[0]['affinity'] == 1 and
                   '-m congklak.value_diagnostics_progress serve --port 18766' in rows[0]['command'],
                   'Unexpected diagnostic observer listener or affinity')
    startup = [json.loads(line) for line in (FOLDER / 'server.stdout.log').read_text().splitlines() if line.startswith('{')]
    common.require(startup and startup[-1]['event'] == 'value_diagnostics_observer_ready' and
                   startup[-1]['pid'] == rows[0]['pid'] and startup[-1]['source_sha256'] == common.sha(ROOT / 'congklak/value_diagnostics_progress.py'),
                   'Diagnostic observer did not load reviewed source')
    return rows[0]


def build():
    baseline = diagnostics.baseline()
    results = diagnostics.completed()
    audit = common.read(FOLDER / 'audit.json')
    common.require(audit['passed'] is True and audit['independent_of_implementation'] is True and
                   audit['unresolved_findings'] == [] and len(audit['recovery_cases']) >= 3,
                   'Independent diagnostics audit is not closed')
    common.current(audit['inputs'])
    state = progress.snapshot()
    common.require(state['current_work']['state'] == 'complete' and
                   state['counters'] == diagnostics.ZERO and state['candidate_quarantined'] is True and
                   state['strength_interpretation_allowed'] is False, 'Diagnostic dashboard scope differs')
    network = listener()
    files = list(SOURCES)
    for path in FOLDER.rglob('*'):
        if path.is_file() and path != FOLDER / 'acceptance.json' and path.name != 'writer.lock' and path.suffix not in ('.pyc', '.pending') and '__pycache__' not in path.parts:
            files.append(path)
    inputs = dict(baseline['inputs'])
    new = dict(audit['inputs']); new.update(common.pins(files))
    common.require(all(k not in inputs or inputs[k] == v for k, v in new.items()), 'Accepted history changed')
    inputs.update(new)
    live_names = {*SOURCES, 'build/value-diagnostics-v1/audit.json',
                  'build/value-diagnostics-v1/production/catalog.json',
                  'build/value-diagnostics-v1/production/identity.json',
                  'build/value-diagnostics-v1/production/capture.json',
                  'build/value-diagnostics-v1/production/bridges.json',
                  'data/quality-train-v1/model.json', 'data/value-training-v1/training/model.json'}
    live_names.update(k for k in new if '/events/' in k)
    body = dict(schema='value-diagnostics-acceptance-v1', passed=True, evidence_accepted=True,
        candidate_quarantined=True, strength_interpretation_allowed=False, promotion=False,
        default_engine='H_v1', models_fixed=True, policy_fixed=True, counters=diagnostics.ZERO,
        previous_inputs_preserved=baseline['inputs_count'], committed_units=results['catalog']['committed_units'],
        actual_recovery_cases=len(audit['recovery_cases']), unresolved_findings=[], listener=network,
        browser_or_HTTP_inspection=False, inputs=dict(sorted(inputs.items())),
        live_inputs={k: v for k, v in inputs.items() if k in live_names})
    diagnostics.atomic(FOLDER / 'acceptance.json', body)
    common.require(progress.snapshot()['accepted'], 'Observer did not accept sealed investigation')
    common.require('torch' not in sys.modules, 'Diagnostics closure imported Torch')
    return body


def verify():
    body = common.read(FOLDER / 'acceptance.json')
    common.require(body['schema'] == 'value-diagnostics-acceptance-v1' and body['passed'] is True and
                   body['counters'] == diagnostics.ZERO and body['models_fixed'] and body['policy_fixed'] and
                   body['candidate_quarantined'] and not body['strength_interpretation_allowed'], 'Diagnostics acceptance differs')
    common.current(body['inputs'])
    diagnostics.completed()
    common.require(progress.snapshot()['accepted'], 'Diagnostic observer reports HOLD')
    listener()
    common.require('torch' not in sys.modules, 'Completed closure verification imported Torch')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('build', 'verify'))
    body = dict(build=build, verify=verify)[parser.parse_args().operation]()
    print(json.dumps(dict(passed=True, inputs=len(body['inputs']), prior_inputs=body['previous_inputs_preserved'],
                          units=body['committed_units'], recovery_cases=body['actual_recovery_cases'],
                          candidate_quarantined=True, counters=body['counters']), sort_keys=True), flush=True)


if __name__ == '__main__': main()
