"""Two new native handoff kills and exact saved-prefix recovery checks.

The660s controller wait covers cold supervisor admission plus a contained120s
worker. It does not extend the independent watchdog's physical worker limit.
"""
import json,os,subprocess,sys,time
from pathlib import Path
from . import prospective_benchmark_v3_common as s, prospective_benchmark_v3_execute as e, prospective_benchmark_v3_runtime as r
ROOT=s.ROOT
CLI=ROOT/'congklak/prospective_benchmark_v3_cli.py'
BASE=e.FIXTURE_BASE
OUT=s.FOLDER/'audit/native-recovery'
def need(ok,message):
 if not ok:raise AssertionError(message)

def command(path,operation,hook=None):
 result=[sys.executable,'-B',str(CLI),'execute','--operation',operation,'--engineering-fixture','--output',str(path),
         '--cache',str(s.DATA/'cache')]
 if hook is not None:result+=['--pause-at',hook,'--pause-marker',str(OUT/(hook+'.paused.json'))]
 return result

def main():
 r.admit();OUT.mkdir(parents=True,exist_ok=True)
 need(not(OUT/'handoff-result.json').exists(),'Completed recovery evidence is immutable')
 need(not(r.DEFAULT/'identity.json').exists() or r.status()['pending'] is None,'A shared worker lease is active')
 sources=dict(s.read(s.PLAN)['source_inputs']);sources.update(s.common.pins([s.PLAN,s.PREFLIGHT]))
 sources.update(s.common.pins([r.CARRY]))
 hooks=list(e.HANDOFF_KILL_HOOKS)
 need(hooks==['before_handoff_commit','after_handoff_commit'],'Recovery hooks differ')
 s.atomic(OUT/'handoff-intent.json',dict(schema=s.SCHEMA+'-native-recovery-intent',source_inputs=sources,hooks=hooks,
  ledger=s.relative(r.DEFAULT),no_ownership_exemption=True,controller_wait_seconds=660,worker_limit_seconds=120))
 cases=[]
 for hook in hooks:
  case=recover_case(hook,'arena');cases.append(case)
  print(json.dumps(dict(hook=hook,passed=True,physical_queries=case['physical_queries'])),flush=True)
 s.common.current(sources);need(r.status()['pending'] is None,'Recovery left an active lease')
 result=dict(schema=s.SCHEMA+'-native-recovery-phase',passed=True,phase='handoff',source_inputs=sources,cases=cases,
  actual_process_kills=2,physical_queries=sum(c['physical_queries'] for c in cases),models_and_policy_fixed=True,
  saved_queries_not_repeated=True,new_label_queries=0)
 s.atomic(OUT/'handoff-result.json',result);return result

def snapshot(path):
    return {s.relative(p): dict(sha256=s.sha(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(path.rglob('*')) if p.is_file() and p.name != 'writer.lock'
            and not p.name.endswith('.pending')}


def preserve(before):
    after = {name: dict(sha256=s.sha(ROOT / name), bytes=(ROOT / name).stat().st_size,
                       mtime_ns=(ROOT / name).stat().st_mtime_ns) for name in before}
    need(before == after, 'Saved native archive/commit/prefix bytes, size or mtime changed')
    return after


def run_logged(args, path):
    with path.open('w', encoding='utf-8', newline='\n') as log:
        p = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=660,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    need(p.returncode == 0, 'Supervisor failed; retained ' + str(path))


def query_archives(path):
    return sorted((path / 'archives').glob('*.json'))


def event_rows(path):
    return [s.read(path / s.read(cp)['event']) for cp in sorted((path / 'commits').glob('*.json'))]


def kill_at(path, operation, hook):
    marker_path = OUT / (hook + '.paused.json')
    need(not path.exists() and not marker_path.exists(), 'Recovery fixture cannot overwrite an attempt')
    need(r.status()['pending'] is None, 'Another global resource lease is active')
    with (OUT / (hook + '.first.log')).open('w', encoding='utf-8', newline='\n') as log:
        supervisor = subprocess.Popen(command(path, operation, hook), cwd=ROOT, stdout=log,
                                      stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + 660
        while not marker_path.is_file():
            need(supervisor.poll() is None, 'Native worker ended before requested pause; log retained')
            need(time.monotonic() < deadline, 'Native pause timeout; watchdog retains enforcement')
            time.sleep(.05)
        marker = s.read(marker_path)
        lease = r.status()['pending']
        need(marker['point'] == hook and lease is not None and lease['worker_output'] == str(path)
             and lease['fixture'] is True and lease['worker_entry'] == str(r.WORKER),
             'Actual pause/worker/fixture/operation scope differs')
        actual = r.process_identity(marker['process']['pid'])
        worker = s.read(lease['worker_identity_file'])
        permit = s.read(lease['permit_file'])
        need(actual == marker['process'] == worker['process'] == permit['worker']
             and worker['token'] == lease['token'] and permit['kill_on_job_close'] is True
             and lease['supervisor_identity']['pid'] == supervisor.pid,
             'Native PID creation/token/supervisor/containment differs')
        expected_argv = [lease['worker_entry'], 'run', '--output', str(path), '--cache', str(s.DATA/'cache'),
                         '--lease', lease['lease_file'], '--engineering-fixture', *lease['extra_args']]
        need(worker['argv'] == marker['argv'] == expected_argv
             and marker['source_sha256'] == s.sha(e.__file__)
             and marker['identity_sha256'] == s.sha(path / 'identity.json')
             and s.sha(lease['worker_entry']) == lease['worker_entry_sha256'],
             'Native paused exact source/argv/identity changed')
        saved = snapshot(path)
        archives = [s.relative(p) for p in query_archives(path)]
        events = event_rows(path)
        if hook == 'before_query':
            need(marker['query_entry_reached'] is False and not archives
                 and not list((path / 'attempts').glob('*.bin'))
                 and sum(x['kind'] == 'query_intent' for x in events) == 1
                 and not any(x['kind'] == 'provider_epoch' for x in events),
                 'Zero-query kill reached native construction/query entry')
        else:
            need(marker['query_entry_reached'] is True, 'Expected completed native work before publication kill')
        kernel, handle = r._open(actual['pid'], 0x1 | 0x1000 | 0x00100000)
        try:
            need(r.process_identity(actual['pid']) == actual, 'Native PID identity changed before exact termination')
            need(kernel.TerminateProcess(handle, 137), 'Exact native worker termination failed')
            need(kernel.WaitForSingleObject(handle, 10000) == 0, 'Native killed process remained alive')
        finally:
            kernel.CloseHandle(handle)
        supervisor.wait(timeout=40)
    settled = r.status()
    finish = settled['events'][-1]
    need(settled['pending'] is None and finish['kind'] == 'finish' and finish['token'] == lease['token']
         and finish['clean'] is False and finish['charged_ns'] == lease['reserved_ns'],
         'Native killed lease was not fully charged on the same cumulative ledger')
    killed = dict(hook=hook, operation=operation, actual_process_kill=True, actual_worker=actual,
                  marker=marker, before=saved, old_archives=archives, finish=finish,
                  ledger=s.relative(r.DEFAULT), lease=s.relative(lease['lease_file']))
    s.atomic(OUT / ('killed-' + hook + '.json'), killed)
    return killed


def recover_case(hook, operation):
    path = BASE / hook
    killed = kill_at(path, operation, hook)
    run_logged(command(path, operation), OUT / (hook + '.resume.log'))
    done = e.completed_job(path, operation)
    need(done['report']['state'] == 'complete' and done['report']['fixture'] is True
         and done['report']['ownership_passed'] is True and done['report']['pending_attempts'] == 0
         and not done['publication_gap'], 'Native recovery did not settle complete ownership/publication')
    after = preserve(killed['before'])
    bodies = [(p, s.read(p)) for p in query_archives(path)]
    old = set(killed['old_archives'])
    new = [(p, b) for p, b in bodies if s.relative(p) not in old]
    if new:
        cold = new[0][1]['answer']['both_seat_reset']['before']
        need(cold['created'] == 2 and cold['destroyed'] == 0
             and sorted(cold['retained_inference_controls'].values()) == [1, 2],
             'Restart physically recomputed saved work before its first new decision')
    need(len({b['decision_id'] for _, b in bodies}) == len(bodies), 'Repeated native query cursor')
    if operation == 'labels':
        need(len(bodies) == 2 and all(b['answer']['fast_certificate']['stage_new_count'] == 0 for _, b in bodies),
             'Recovery label work discovered exposure beyond the already closed same-origin prefix')
    else:
        need(done['report']['games'] == 1 and sum(x['kind'] == 'handoff' for x in event_rows(path)) == 1,
             'Shared suffix was skipped or handed off repeatedly')
    if hook in ('before_catalog', 'after_catalog'):
        need(not new, 'Catalog recovery performed another native query')
    receipt = dict(schema=s.SCHEMA+'-native-recovery-case', passed=True,
                   hook=hook, operation=operation, actual_process_kill=True, fixture=s.relative(path),
                   physical_queries=len(bodies), pre_kill_archives=len(old), new_queries_after_resume=len(new),
                   cold_lifecycle_checked=bool(new), saved_queries_not_repeated=True,
                   prefix_before=killed['before'], prefix_after=after, ownership_passed=True,
                   stage_new_count_zero=operation == 'labels', report_ref=dict(path=s.relative(path / 'report.json'),
                       sha256=s.sha(path / 'report.json')))
    s.atomic(OUT / ('case-' + hook + '.json'), receipt)
    return receipt


