"""Repair only the independent recovery harness for Windows venv redirectors.

The implementation and six original audit methods remain unchanged. Five
successful v1 methods retain their own executed source/evidence. This version
re-executes only publication recovery, targeting the verified worker PID.
"""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('independent_value_interface_v1_tool', ROOT / 'tools/audit-valuepilot-interface.py')
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
audit = original.audit
FOLDER = original.FOLDER


def process_identity(pid):
    assert type(pid) is int and pid > 0
    script = "$p=Get-CimInstance Win32_Process -Filter 'ProcessId=" + str(pid) + "'; if ($p) { [pscustomobject]@{pid=$p.ProcessId;parent=$p.ParentProcessId;command=$p.CommandLine} | ConvertTo-Json -Compress }"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True,
                            text=True, check=True, timeout=30)
    assert result.stdout.strip(), 'Marked worker is no longer running'
    return json.loads(result.stdout)


def kill_worker(argv, directory, marker, boundary, timeout=120):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); process = None; worker_handle = None; details = None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.TerminateProcess.restype = ctypes.c_int
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetExitCodeProcess.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err)
            deadline = time.monotonic() + timeout
            while not marker.exists():
                if process.poll() is not None:
                    raise AssertionError('Worker ended before pause: ' + (directory / 'stderr.log').read_text(encoding='utf-8', errors='replace')[-6000:])
                assert time.monotonic() < deadline, 'Pause marker timeout'
                time.sleep(.05)
            body = audit.read(marker)
            assert body.get('event', body.get('point')) == boundary
            worker_pid = body['pid']; details = process_identity(worker_pid)
            lineage = [details]; current = details
            while current['pid'] != process.pid and len(lineage) < 5:
                current = process_identity(current['parent']); lineage.append(current)
            assert lineage[-1]['pid'] == process.pid, 'Pause worker is not our launched descendant'
            assert str(Path(argv[1]).resolve()).casefold() in details['command'].casefold(), 'Worker script differs'
            # Open a stable handle only after exact script/lineage validation.
            worker_handle = kernel.OpenProcess(0x0001 | 0x00100000 | 0x1000, False, worker_pid)
            assert worker_handle, 'Cannot open verified worker handle'
            assert kernel.TerminateProcess(worker_handle, 1), 'Actual worker termination failed'
            assert kernel.WaitForSingleObject(worker_handle, 30000) == 0, 'Worker termination wait failed'
            worker_code = ctypes.c_ulong()
            assert kernel.GetExitCodeProcess(worker_handle, ctypes.byref(worker_code))
            assert worker_code.value != 0, 'Worker exited cleanly rather than being killed'
            code = process.wait(timeout=30)
            assert code != 0
        finally:
            if worker_handle: kernel.CloseHandle(worker_handle)
            if process is not None and process.poll() is None:
                process.kill(); process.wait(timeout=30)
    record = dict(boundary=boundary, argv=[str(v) for v in argv], launcher_pid=process.pid,
        worker_pid=worker_pid, worker_identity=details, verified_lineage=lineage,
        returncode=code, worker_returncode=worker_code.value, marker=audit.rel(marker),
        marker_sha256=audit.sha(marker), actual_process_kill=True,
        elapsed_seconds=time.monotonic() - started, logs=audit.rel(directory))
    audit.KILLS.append(record); audit.ATTEMPTS.append(record); audit.save(directory / 'execution.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('operation', choices=('run',))
    args = parser.parse_args()
    old = audit.read(FOLDER / 'audit-v1.json')
    old_methods = audit.read(FOLDER / 'independent-v1/methods.json')
    assert old['passed'] is False and old_methods['failures'] == 1 and old_methods['errors'] == 0
    failed = old_methods['failure_details'][0]
    assert 'test_05_real_publication_kills_and_prefix_recovery' in failed['method']
    assert 'Wrong process marker' in failed['trace']
    for name, digest in old['inputs'].items():
        assert audit.sha(ROOT / name) == digest, 'v1 source/input changed: ' + name
    work = FOLDER / 'independent-recovery-v2'; work.mkdir(parents=True, exist_ok=False)
    source_pins = original.inputs(work)
    source_pins[audit.rel(__file__)] = audit.sha(__file__)
    audit.WORK = work; audit.kill_process = kill_worker
    method = 'test_05_real_publication_kills_and_prefix_recovery'
    suite = unittest.TestSuite([audit.ValuepilotIndependentAudit(method)])
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(suite)
    proof = dict(schema='valuepilot-interface-independent-recovery-v2', passed=result.wasSuccessful(),
        automated_test_methods=[method], tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), failure_details=[dict(method=str(test), trace=trace) for test, trace in result.failures + result.errors],
        elapsed_seconds=time.monotonic() - started, execution_sources=source_pins,
        evidence=audit.EVIDENCE, process_kill_cases=audit.KILLS, attempts=audit.ATTEMPTS,
        repair='Verify Windows venv redirector lineage; terminate actual marked worker handle rather than assuming launcher PID equality.')
    audit.save(work / 'methods.json', proof)
    for name, digest in source_pins.items(): assert audit.sha(ROOT / name) == digest
    if not result.wasSuccessful():
        audit.save(FOLDER / 'audit-recovery-v2-failed.json', proof); return 1
    assert len(audit.KILLS) == 6
    passed_v1 = [name for name in old_methods['automated_test_methods'] if name != method]
    assert len(passed_v1) == 5
    mapping = dict(old['inputs']); mapping.update(original.inputs(work))
    mapping[audit.rel(__file__)] = audit.sha(__file__)
    mapping[audit.rel(FOLDER / 'audit-v1.json')] = audit.sha(FOLDER / 'audit-v1.json')
    merged_evidence = dict(old_methods['evidence']); merged_evidence.update(audit.EVIDENCE)
    record = dict(schema='valuepilot-interface-independent-audit-v2', passed=True, independent_of_implementation=True,
        automated_test_methods=passed_v1 + [method], automated_test_methods_count=6,
        tests_run=7, accepted_methods=6, retained_failed_harness_methods=1,
        process_kill_cases=audit.KILLS, process_kill_cases_count=len(audit.KILLS),
        all_attempted_process_kill_cases=old_methods['process_kill_cases'] + audit.KILLS,
        actual_process_kills_all_attempts=len(old_methods['process_kill_cases']) + len(audit.KILLS),
        distinct_recovery_boundaries=6, repeated_data_kills_retained=2,
        unresolved_findings=[], readiness_blockers=old['readiness_blockers'],
        counters=dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0),
        training_ready=False, production_value_exports=0, inputs=dict(sorted(mapping.items())),
        evidence=merged_evidence, attempts=old_methods['attempts'] + audit.ATTEMPTS,
        executed_groups=[dict(source='tests/test_valuepilot_interface_audit.py', proof='build/valuepilot-interface-v1/independent-v1/methods.json',
                              accepted_methods=passed_v1),
                         dict(source=audit.rel(__file__), proof=audit.rel(work / 'methods.json'), accepted_methods=[method])],
        repair=proof['repair'], limits=old['limits'])
    audit.save(FOLDER / 'audit-v2-executed.json', record)
    print(json.dumps(dict(event='valuepilot_independent_recovery_v2', passed=True, methods=1,
                         actual_kills_this_run=6, actual_kills_all_attempts=8), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
