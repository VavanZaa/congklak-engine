"""Add static closure and metadata coverage to executed independent evidence.

This does not rerun or manufacture the six accepted methods. Their original
source-bound records remain intact, including the repaired recovery attempt.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import valuepilot_reservations as data
FOLDER = ROOT / 'build/valuepilot-interface-v1'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def save(path, value):
    assert not path.exists(), 'Immutable audit record already exists'
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main():
    executed = read(FOLDER / 'audit-v2-executed.json')
    assert executed['passed'] is True and executed['accepted_methods'] == 6
    assert executed['process_kill_cases_count'] == 6 and executed['actual_process_kills_all_attempts'] == 8
    superseded_static = {}
    for name, digest in executed['inputs'].items():
        current = sha(ROOT / name)
        if name == 'tools/verify-valuepilot-interface.py' and current != digest:
            archive = FOLDER / 'root-closure-reviewed-v1.py'
            assert sha(archive) == digest, 'Exact previously reviewed root closure bytes not retained'
            assert current == '47fd032b431d570665bd56d25ef48ef19a987f5d8f469d5df9a4c6a12acf191b'
            superseded_static[name] = dict(previous_sha256=digest, retained_exact_bytes=rel(archive),
                final_sha256=current, review='Only the small live dashboard pin map changed; full evidence verification is unchanged. This root closure was a static input, not executed by the six independent methods.')
        else: assert current == digest, name
    manifest = data.verify(); assert manifest['admitted'] is False and manifest['status'] == 'hold'
    inventory = read(data.DEFAULT / 'inventory.json')
    by_f = {family: component for component in inventory['components'].values() for family in component['families']}
    by_q = {q: component for component in inventory['components'].values() for q in component['quotients']}
    groups = []
    for name, roles in (
        ('ordinary-selfplay-registry.json', ('ancestral_roles', 'recent_roles', 'diagnostics')),
        ('selfplay-training-pilot-registry.json', ('historical_roles', 'latest_roles', 'diagnostics')),
        ('ordinary-value-registry.json', ('rows',))):
        ledger = read(ROOT / 'build' / name)
        for role in roles:
            values = ledger[role]; values = list(values.values()) if isinstance(values, dict) else values
            for row in values:
                state = row.get('state') or row.get('root', {}).get('state')
                quotient = row.get('quotient') or row.get('root', {}).get('quotient')
                if quotient is None: quotient = data.previous.quotient(state)
                component = by_f[row['family']]
                assert component == by_q[quotient], (name, role, row['family'])
                if row['cohort'] != 'training_candidate':
                    assert any(r.startswith('optimizer/') for r in component['reasons'])
                    assert any(r.startswith('arena/') for r in component['reasons'])
            groups.append(dict(source='build/' + name, group=role, references=len(values),
                               all_origins_resolved=True, nontraining_origins_protected=True))
    dashboard = read(FOLDER / 'dashboard-owner-checks.json')
    assert dashboard['passed'] is True and dashboard['accepted_method_count'] == 5
    for name, digest in dashboard.get('sources', {}).items(): assert sha(ROOT / name) == digest
    source_freeze = read(FOLDER / 'data-owner-acceptance.json')['source_freeze']
    for name, digest in source_freeze.items(): assert sha(ROOT / name) == digest
    interface = read(FOLDER / 'interface-owner-acceptance.json')
    assert interface['passed'] and interface['tests_run'] == 11 and interface['failures'] == interface['errors'] == 0
    assert interface['test_source_sha256'] == sha(ROOT / 'tests/test_value_sidecar.py')
    for name, digest in interface.get('sources', {}).items(): assert sha(ROOT / name) == digest
    # A failed redirector assertion had a cleanup route. Confirm that no export
    # worker is left alive; do not invent its unrecorded launcher PID/exit code.
    script = "$p=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python.*\\.exe$' -and $_.CommandLine -like '*export-value-sidecar.py*' }); @($p | Select-Object ProcessId,ParentProcessId,CommandLine) | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, check=True, timeout=30)
    assert not result.stdout.strip() or json.loads(result.stdout) in (None, []), 'Orphan export worker exists'
    supplement = dict(schema='valuepilot-interface-independent-static-closure-v1', passed=True,
        executed_methods_rerun=0, registry_role_projections=groups,
        reviewed_owner_methods=dict(reservations=8, interface=11, dashboard=5),
        original_independent_methods=6, retained_independent_method_executions=7,
        accepted_distinct_recovery_boundaries=6, recorded_protocol_kills_all_attempts=8,
        failed_launch_cleanup=dict(marker_worker_pid=57580, launcher_pid_not_retained=True,
            not_counted_as_successful_recovery=True, no_orphan_export_worker=True),
        additional_static_inputs=['docs/VALUEPILOT_INTERFACE_AUDIT.md',
            'build/valuepilot-interface-v1/dashboard-owner-checks.json',
            'build/valuepilot-interface-v1/data-owner-acceptance.json',
            'build/valuepilot-interface-v1/interface-owner-acceptance.json'],
        superseded_static_pins=superseded_static,
        counters=executed['counters'], readiness_blockers=executed['readiness_blockers'])
    save(FOLDER / 'audit-static-closure.json', supplement)
    mapping = dict(executed['inputs'])
    # Retain failed owner/audit attempts and publication fixture evidence. Live
    # operational server logs and root's still-editable dashboard status are
    # deliberately excluded; root binds its final status during acceptance.
    excluded = {'stage-status.json', 'server.stdout.log', 'server.stderr.log', 'audit.json', 'acceptance.json'}
    for path in FOLDER.rglob('*'):
        if path.is_file() and path.suffix != '.pyc' and not path.name.endswith('-shm') and path.name not in excluded:
            mapping[rel(path)] = sha(path)
    for path in (Path(__file__), ROOT / 'docs/VALUEPILOT_INTERFACE_AUDIT.md'):
        mapping[rel(path)] = sha(path)
    mapping['tools/verify-valuepilot-interface.py'] = sha(ROOT / 'tools/verify-valuepilot-interface.py')
    final = dict(executed, inputs=dict(sorted(mapping.items())),
        static_closure='build/valuepilot-interface-v1/audit-static-closure.json',
        additive_static_review_does_not_claim_original_execution=True)
    for name, digest in final['inputs'].items(): assert sha(ROOT / name) == digest, name
    save(FOLDER / 'audit.json', final)
    print(json.dumps(dict(event='valuepilot_independent_audit_closed', passed=True,
        accepted_methods=6, accepted_kill_boundaries=6, all_recorded_protocol_kills=8,
        inputs=len(mapping), training_ready=False), sort_keys=True), flush=True)


if __name__ == '__main__': main()
