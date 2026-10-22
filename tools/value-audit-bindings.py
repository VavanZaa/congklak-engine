"""Relocate three actually executed audit groups without claiming a rerun."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS=('plan','targets','training')
SNAPSHOT='tools/value-audit-executed-v1.py'
ORIGINAL='tools/value-audit.py'
SELF='tools/value-audit-bindings.py'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def derived(group):
    if group not in GROUPS:raise ValueError('Not an authorized preserved group')
    original=f'build/value-audit/executed-v1/{group}-tests.json'
    body=json.loads((ROOT/original).read_bytes())
    digest=body['inputs'][ORIGINAL]
    if sha(ROOT/SNAPSHOT)!=digest:raise ValueError('Executed coordinator snapshot differs')
    for path,expected in body['inputs'].items():
        if path!=ORIGINAL and sha(ROOT/path)!=expected:raise ValueError('Original executed input changed '+path)
    del body['inputs'][ORIGINAL]
    body['inputs'].update({SNAPSHOT:digest,original:sha(ROOT/original),SELF:sha(ROOT/SELF)})
    body['executed_source_provenance']={
        'original_report_path':original,'original_report_sha256':sha(ROOT/original),
        'original_coordinator_path':ORIGINAL,'original_coordinator_sha256':digest,
        'coordinator_snapshot_path':SNAPSHOT,'coordinator_snapshot_sha256':digest,
        'unchanged_test_source_path':'tests/value_audit_acceptance.py',
        'unchanged_test_source_sha256':body['inputs']['tests/value_audit_acceptance.py'],
        'tests_rerun':False,
        'derivation':'Retained identical executed coordinator bytes; original outcomes, methods, evidence, durations and all other input bindings unchanged.',
    }
    return body

def validate(group,body):
    if body!=derived(group):raise ValueError('Derived execution report differs from original evidence')

def main():
    for group in GROUPS:
        path=ROOT/f'build/value-audit/{group}-tests.json';value=derived(group)
        current=json.loads(path.read_bytes())
        if 'executed_source_provenance' in current:
            validate(group,current)
        else:
            original=json.loads((ROOT/f'build/value-audit/executed-v1/{group}-tests.json').read_bytes())
            if current!=original:raise ValueError('Refusing to replace an unrecognized report')
            path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print('Three earlier executed groups rebound to exact retained source; no tests rerun.')

if __name__=='__main__':main()
