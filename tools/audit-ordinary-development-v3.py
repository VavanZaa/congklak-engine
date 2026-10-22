"""Re-execute only generation/publication recovery and completed validation."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('ordinary_development_v3_recovery_audit', ROOT / 'tests/test_ordinary_development_audit_v3.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
spec2 = importlib.util.spec_from_file_location('original_ordinary_development_audit_tool', ROOT / 'tools/audit-ordinary-development.py')
original = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(original)
FOLDER = ROOT / 'build/ordinary-development-v1'


def main():
    first = audit.read(FOLDER / 'audit-v1.json')
    repair = audit.read(FOLDER / 'audit-method4-repair-v2.json')
    assert first['passed'] is False and repair['passed'] is True
    for name, digest in first['inputs'].items(): assert audit.sha(ROOT / name) == digest, name
    for name, digest in repair['execution_sources'].items(): assert audit.sha(ROOT / name) == digest, name
    review = audit.read(FOLDER / 'protocol-review-v3.json'); assert review['passed'] is True
    for name, digest in review['inputs'].items(): assert audit.sha(ROOT / name) == digest, name
    work = FOLDER / 'independent-recovery-v3'; work.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'tests/test_ordinary_development_audit_v3.py', ROOT / 'congklak/ordinary_development_v3.py', ROOT / 'tools/generate-ordinary-development-v3.py']
    source_pins = {audit.rel(p): audit.sha(p) for p in sources}
    source_folder = work / 'execution-sources'; source_folder.mkdir()
    for path in sources: shutil.copy2(path, source_folder / path.name)
    audit.WORK = work
    methods = ['test_05_real_generation_reservation_export_kills_and_prefixes', 'test_06_completed_failclosed_history_and_dashboard']
    suite = unittest.TestSuite([audit.OrdinaryDevelopmentIndependentAudit(name) for name in methods])
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = dict(schema='ordinary-development-independent-recovery-methods-v3', passed=result.wasSuccessful(),
        automated_test_methods=methods, tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, execution_sources=source_pins,
        evidence=audit.EVIDENCE, attempts=audit.ATTEMPTS, process_kill_cases=audit.KILLS,
        actual_process_kills_this_execution=len(audit.KILLS),
        scope='V3 full-prefix plus bounded literal-orphan validation; same-data recovery fixtures, not new distinct data.')
    audit.save(work / 'methods.json', proof)
    for name, digest in source_pins.items(): assert audit.sha(ROOT / name) == digest
    report = dict(proof, inputs=original.inputs(work))
    report['inputs'].update(source_pins)
    for p in (FOLDER / 'root-publication-checks.json', FOLDER / 'protocol-review-v3.json',
              FOLDER / 'audit-orphan-corruption-proof.json', FOLDER / 'audit-prefix-corruption-proof.json'):
        report['inputs'][audit.rel(p)] = audit.sha(p)
    audit.save(FOLDER / 'audit-recovery-v3-executed.json', report)
    print(json.dumps(dict(event='ordinary_development_recovery_v3', passed=result.wasSuccessful(),
        methods=result.testsRun, actual_process_kills=len(audit.KILLS), canonical_audit_published=False), sort_keys=True), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__': raise SystemExit(main())
