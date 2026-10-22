"""Source-bound actual process recovery tests, dispatched one stage at a time.

Native fixtures repeat only the already captured first candidate and prove its
full quotient set is a subset of that primary capture. The before-query case
never enters a provider and is permanently retained without retry. All actual
workers share the single inherited-budget resource ledger.
"""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, footprint_screen_common as s, footprint_screen_resources as resources

CLI = ROOT/'tools/run-footprint-screen.py'
NATIVE_HOOKS = ('before_result_commit','after_result_commit','before_trace_commit',
                'after_trace_commit','before_catalog','after_catalog')
REVIEW_HOOKS = ('before_unit_commit','after_unit_commit')
PUBLICATION_HOOKS = ('before_catalog','after_catalog')


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def ref(path):
    return dict(path=s.relative(path), sha256=s.sha(path))


def saved_prefix(folder):
    return {s.relative(p): dict(sha256=s.sha(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file() and p.name != 'writer.lock' and not p.name.endswith('.pending')}


def preserved(before):
    current = {name: dict(sha256=s.sha(ROOT/name), bytes=(ROOT/name).stat().st_size,
                          mtime_ns=(ROOT/name).stat().st_mtime_ns) for name in before}
    need(before == current, 'Saved capture/committed prefix bytes, size or mtime changed')
    return current


def command(operation, folder, hook=None, marker=None):
    args = [sys.executable, str(CLI), operation, '--output', str(folder), '--engineering-fixture']
    if hook:
        args += ['--pause-at', hook, '--pause-marker', str(marker)]
    return args


def run_logged(args, log):
    with log.open('x', encoding='utf-8', newline='\n') as stream:
        done = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=360,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    need(done.returncode == 0, 'Supervisor refused; retained '+str(log))


def kill_case(operation, folder, hook, out, marker):
    need(not folder.exists() and not marker.exists(), 'Recovery case evidence must be fresh')
    args = command(operation, folder, hook, marker)
    first_log = out/(hook+'.first.log')
    with first_log.open('x', encoding='utf-8', newline='\n') as stream:
        supervisor = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic()+230
        while not marker.is_file():
            need(supervisor.poll() is None, 'Worker ended before pause; retained '+str(first_log))
            need(time.monotonic() < deadline, 'Pause timed out; watchdog remains responsible')
            time.sleep(.05)
        pause = s.read(marker)
        need(pause['point'] == hook, 'Actual pause boundary differs')
        report = resources.status()
        lease = report['pending']
        need(lease is not None and Path(lease['worker_output']).resolve() == folder.resolve() and
             lease['fixture'] is True and lease['supervisor_identity']['pid'] == supervisor.pid,
             'Actual paused lease/supervisor differs')
        worker = s.read(lease['worker_identity_file'])
        permit = s.read(lease['permit_file'])
        actual = resources.process_identity(pause['process']['pid'])
        need(actual == pause['process'] == worker['process'] == permit['worker'] and
             worker['argv'] == pause['argv'] and worker['token'] == permit['token'] == lease['token'] and
             permit['kill_on_job_close'] is True and s.sha(lease['worker_entry']) == lease['worker_entry_sha256'],
             'Actual PID creation/source/argv/token/containment differs')
        need(s.read(s.PLAN)['source_inputs'].get(s.relative(lease['worker_entry'])) == lease['worker_entry_sha256'],
             'Paused worker entry is not the frozen plan source')
        before = saved_prefix(folder)
        kernel, handle = resources._open(actual['pid'], 0x1 | 0x1000 | 0x00100000)
        try:
            need(resources.process_identity(actual['pid']) == actual, 'Paused PID identity changed before termination')
            need(kernel.TerminateProcess(handle, 137), 'Exact worker termination failed')
            need(kernel.WaitForSingleObject(handle, 10000) == 0, 'Killed worker remains active')
        finally:
            kernel.CloseHandle(handle)
        supervisor.wait(timeout=45)
    settled = resources.status()
    finish = settled['events'][-1]
    need(settled['pending'] is None and finish['kind'] == 'finish' and finish['token'] == lease['token'] and
         finish['clean'] is False and finish['charged_ns'] == lease['reserved_ns'],
         'Actual interrupted lease was not settled at its full reservation')
    receipt = dict(schema=s.SCHEMA+'-independent-kill', hook=hook, actual_process_interruption=True,
                   process=actual, marker_ref=ref(marker), worker_ref=ref(lease['worker_identity_file']),
                   permit_ref=ref(lease['permit_file']), lease_ref=ref(lease['lease_file']),
                   source_sha256=lease['worker_entry_sha256'], prefix_before=before, finish=finish)
    dest = out/('killed-'+hook+'.json')
    s.atomic(dest, receipt)
    return receipt, before


def sources():
    paths = [Path(__file__), CLI, resources.ENTRY, resources.AUDIT_ENTRY,
             ROOT/'congklak/footprint_screen_worker.py', ROOT/'congklak/footprint_screen_native.py',
             ROOT/'congklak/footprint_screen_review.py', ROOT/'congklak/footprint_screen_resources.py',
             ROOT/'congklak/footprint_screen_reservations.py', ROOT/'tools/reserve-footprint-screen.py']
    pins = s.common.pins(paths)
    plan = s.admit_plan()
    need(all(plan['source_inputs'].get(k) == v for k,v in pins.items()), 'Recovery driver is not source-bound')
    return pins


def native_stage():
    from congklak import footprint_screen_worker as worker
    pins = sources()
    out = s.FOLDER/'audit/native-recovery'
    need(not out.exists(), 'Native recovery evidence already exists; no silent retry')
    need(resources.status()['pending'] is None, 'Another worker lease is active')
    primary = worker.production_snapshot()
    allowed = set(primary['roots'][0]['quotients'])
    out.mkdir(parents=True)
    s.atomic(out/'intent.json', dict(schema=s.SCHEMA+'-native-recovery-intent', source_inputs=pins,
             plan_sha256=s.sha(s.PLAN), hooks=[*NATIVE_HOOKS,'before_query'],
             captured_primary_snapshot_sha256=config.digest(primary), source_family=primary['roots'][0]['id']))
    cases = {}
    for hook in NATIVE_HOOKS:
        folder = worker.FIXTURE_BASE/hook
        marker = out/(hook+'.paused.json')
        killed, before = kill_case('run', folder, hook, out, marker)
        run_logged(command('run', folder), out/(hook+'.resume.log'))
        done = worker.completed(folder)
        need(done['report']['state'] == 'complete' and not done['publication_gap'], 'Recovered native case is incomplete')
        data = worker._read(folder, worker._identity(folder))
        need(len(data['results']) == 2, 'Recovered native case repeated or omitted scheduled queries')
        observed = set()
        for row in data['results']:
            body = s.read(ROOT/row['archive_ref']['path'])
            owned = base64.b64decode(body['answer']['fast_owned_high'], validate=True)
            observed.update(q.hex() for q, _, _, _ in struct.iter_unpack('<15sQ16sB', owned))
            original_path = ROOT/primary['roots'][0]['source_refs'][row['index']]['path']
            original = s.read(original_path)
            need(body['capture_proof']['raw_sha256'] == original['capture_proof']['raw_sha256'],
                 'Fixed teacher/feature raw work differs from primary capture')
        need(not observed-allowed, 'Recovery fixture exposes a new high quotient outside the captured primary root')
        after = preserved(before)
        case_path = out/('case-'+hook+'.json')
        s.atomic(case_path, dict(passed=True, hook=hook, actual_process_interruption=True,
                 completed_queries=2, subset_only_new_quotients=0, fixed_raw_work_equal=True,
                 committed_capture_not_repeated=True, complete_orphan_adopted=True,
                 prefix_before=before, prefix_after=after, prefix_unchanged=True,
                 catalog_ref=ref(folder/'catalog.json')))
        cases[hook] = dict(passed=True, hook=hook, actual_process_interruption=True,
                           fixture_identity_ref=ref(folder/'identity.json'), catalog_ref=ref(folder/'catalog.json'),
                           evidence_refs=[ref(case_path),ref(out/('killed-'+hook+'.json')),ref(marker),
                                          ref(out/(hook+'.first.log')),ref(out/(hook+'.resume.log'))])
        print(config.canonical(dict(stage='native-recovery', hook=hook, passed=True)), flush=True)
    hook = 'before_query'
    folder = worker.FIXTURE_BASE/hook
    marker = out/(hook+'.paused.json')
    killed, before = kill_case('run', folder, hook, out, marker)
    pause = s.read(marker)
    need(pause['provider_constructed'] is False and pause['query_entry_reached'] is False,
         'Unknown-intent fixture crossed the actual provider entry')
    data = worker._read(folder, worker._identity(folder))
    need([e['kind'] for e in data['events']] == ['guard_restore','intent'] and not data['results'] and
         not any((folder/'archives').glob('*')) and not any((folder/'attempts').glob('*')),
         'Pre-provider fixture contains actual query material')
    ledger_before = saved_prefix(resources.DEFAULT)
    try:
        worker.preflight(folder, fixture=True)
    except Exception as error:
        refusal = str(error)
    else:
        raise AssertionError('Unknown unreturned intent allowed retry')
    need('Unknown unreturned acquisition' in refusal, 'Unknown-intent refusal reason differs')
    preserved(before)
    preserved(ledger_before)
    zero_case = dict(hook=hook, identity_ref=ref(folder/'identity.json'), marker_ref=ref(marker),
                     actual_query_count=0, provider_constructed=False, query_entry_reached=False, no_retry=True,
                     evidence_refs=[ref(out/('killed-'+hook+'.json'))], refusal=refusal)
    s.common.current(pins)
    s.atomic(out/'result.json', dict(schema=s.SCHEMA+'-independent-native-recovery', passed=True,
             independent_of_implementation=True, plan_sha256=s.sha(s.PLAN), source_inputs=pins, cases=cases,
             process_interruptions=7, subset_only_new_quotients=0, zero_query_case=zero_case,
             actual_native_queries=12, model_forwards=0, optimizer_steps=0, unresolved_findings=[]))
    print(config.canonical(dict(stage='native-recovery', passed=True, process_interruptions=7)), flush=True)


def review_stage():
    from congklak import footprint_screen_review as review
    pins = sources()
    out = s.FOLDER/'audit/raw-review-recovery'
    need(not out.exists(), 'Raw review recovery evidence already exists')
    out.mkdir(parents=True)
    cases = []
    for hook in REVIEW_HOOKS:
        folder = review.FIXTURES/hook
        marker = out/(hook+'.paused.json')
        killed, before = kill_case('review', folder, hook, out, marker)
        units_before = sorted((folder/'units').glob('*.json'))
        need(len(units_before) == 1, 'Review pause did not retain its first complete raw proof')
        run_logged(command('review', folder), out/(hook+'.resume.log'))
        done = review.completed(folder)
        need(done['report']['passed'] is True and not done['publication_gap'], 'Recovered independent review failed')
        after = preserved(before)
        case = dict(passed=True, hook=hook, actual_process_interruption=True, new_queries=0,
                    complete_proof_not_recomputed=True, prefix_before=before, prefix_after=after,
                    prefix_unchanged=True, fixture_identity_ref=ref(folder/'identity.json'),
                    catalog_ref=ref(folder/'catalog.json'), evidence_refs=[ref(out/('killed-'+hook+'.json')),ref(marker)])
        s.atomic(out/('case-'+hook+'.json'), case)
        cases.append(case)
    s.common.current(pins)
    s.atomic(out/'result.json', dict(schema=s.SCHEMA+'-independent-review-recovery', passed=True,
             independent_of_implementation=True, source_inputs=pins, process_interruptions=2, cases=cases,
             new_queries=0, model_forwards=0, optimizer_steps=0))


def reservation_stage():
    from congklak import footprint_screen_reservations as reservations
    pins = sources()
    out = s.FOLDER/'audit/reservation-recovery'
    need(not out.exists(), 'Reservation recovery evidence already exists')
    out.mkdir(parents=True)
    cases = []
    for hook in PUBLICATION_HOOKS:
        folder = s.FOLDER/'fixtures/reservation-recovery'/hook
        marker = folder/'pause.json'
        killed, before = kill_case('reserve', folder, hook, out, marker)
        run_logged(command('reserve', folder), out/(hook+'.resume.log'))
        done = reservations.completed(folder)
        need(not done['publication_gap'] and done['report']['fixture'] is True and
             done['reservation_admission'] is False, 'Metadata fixture became production reservation authority')
        after = preserved(before)
        case = dict(passed=True, hook=hook, actual_process_interruption=True, new_queries=0,
                    prefix_before=before, prefix_after=after, prefix_unchanged=True,
                    catalog_ref=ref(folder/'catalog.json'), evidence_refs=[ref(out/('killed-'+hook+'.json'))])
        s.atomic(out/('case-'+hook+'.json'), case)
        cases.append(case)
    s.common.current(pins)
    s.atomic(out/'result.json', dict(schema=s.SCHEMA+'-independent-reservation-recovery', passed=True,
             independent_of_implementation=True, source_inputs=pins, process_interruptions=2,
             cases=cases, new_queries=0, model_forwards=0, optimizer_steps=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('native','review','reservations'))
    args = parser.parse_args()
    s.admit_plan()
    {'native': native_stage, 'review': review_stage, 'reservations': reservation_stage}[args.stage]()


if __name__ == '__main__':
    main()
