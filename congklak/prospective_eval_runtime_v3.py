"""Prospectively reviewed capacity extension on the existing charged ledger.

The ledger identity and every prior lease remain unchanged. This module uses
the same Windows containment and event replay with the reviewed effective cap.
"""
from contextlib import contextmanager
from pathlib import Path
import os,subprocess,sys,time,uuid
from . import config, prospective_eval_common as stage, prospective_eval_runtime_v2 as previous
base=previous.base
ROOT,DEFAULT,SCHEMA=base.ROOT,base.DEFAULT,base.SCHEMA
CAPACITY=stage.FOLDER/'amendments/v3/capacity.json'
REVIEW=CAPACITY.parent/'preflight.json'
WORKER=ROOT/'congklak/prospective_eval_worker_v3.py'
SUPERVISOR=ROOT/'congklak/prospective_eval_cli_v3.py'
need,read,sha=stage.need,stage.read,stage.sha
Writer=base.Writer
_path,_atomic,_append,_identity=base._path,base._atomic,base._append,base._identity
_open,_job,_alive,private_commit=base._open,base._job,base._alive,base.private_commit
process_identity,boot_identity=base.process_identity,base.boot_identity
namespace_bytes,preflight=base.namespace_bytes,base.preflight
operational_binding,verify_operational_binding=base.operational_binding,base.verify_operational_binding
REQUIRED=('congklak/prospective_eval_execute_v3.py','congklak/prospective_eval_runtime_v3.py',
 'congklak/prospective_eval_worker_v3.py','congklak/prospective_eval_cli_v3.py',
 'build/prospective-eval-v1/amendments/v3/project_v3.py','build/prospective-eval-v1/amendments/v3/source-projection.json',
 'build/prospective-eval-v1/amendments/v3/verify_reader_v3.py','build/prospective-eval-v1/amendments/v3/freeze_v3.py',
 'tests/test_prospective_eval_v3.py')

def admit():
    stage.admit_plan();previous.admit();body=read(CAPACITY);review=read(REVIEW)
    need(body['schema']=='prospective-eval-v1-capacity-v3' and body['base_plan_ref']==stage.ref(stage.PLAN)
         and body['previous_amendment_ref']==stage.ref(previous.AMENDMENT)
         and body['previous_review_ref']==stage.ref(previous.REVIEW),'Capacity predecessor changed')
    need(body['budget_reset'] is False and body['models_policy_search_clock_changed'] is False
         and body['ledger_identity_changed'] is False,'Capacity amendment changed scientific or accounting scope')
    need(body['resource_identity_ref']==stage.ref(DEFAULT/'identity.json'),'Existing resource identity changed')
    protocol=dict(read(DEFAULT/'identity.json')['protocol'],cumulative_ns=18_000_000_000_000,
                  quota_bytes=32*1024**3,lease_ns=1_800_000_000_000)
    need(body['effective_protocol']==protocol and body['legacy_execution_binding']==previous.execution_binding(),
         'Effective cap or exact legacy execution binding changed')
    need(set(REQUIRED)<=set(body['source_inputs']),'Required v3 source omitted')
    stage.common.current(body['source_inputs'])
    need(review['schema']=='prospective-eval-v1-capacity-v3-independent-preflight' and review['passed'] is True
         and review['independent_of_implementation'] is True and review['capacity_ref']==stage.ref(CAPACITY)
         and review['source_inputs']==body['source_inputs'],'Independent effective-capacity review missing or stale')
    return body

def execution_binding():
    body=admit()
    return dict(schema='prospective-eval-v1-capacity-v3-binding',capacity_ref=stage.ref(CAPACITY),review_ref=stage.ref(REVIEW),
      source_inputs=body['source_inputs'],effective_protocol=body['effective_protocol'],same_ledger=str(DEFAULT))

def effective_protocol():return dict(admit()['effective_protocol'])

def verify_execution_identity(identity,path):
    body=admit();path=Path(path).resolve();name=stage.relative(path/'identity.json')
    need(body['legacy_identity_refs'].get(name)==sha(path/'identity.json'),'Only the pinned existing job identity is admitted')
    if 'execution_amendment' in identity:
        need(identity['execution_amendment']==body['legacy_execution_binding'],'Existing job execution binding changed')
    else:
        need(path==stage.FOLDER/'jobs/endgame' and identity['operation']=='endgame' and identity['fixture'] is False
             and (path/'catalog.json').is_file() and read(path/'report.json')['state']=='complete',
             'Unversioned identity is not the sealed endgame predecessor')
    return True

@contextmanager
def execution_views():
    binding=execution_binding()
    try:yield binding
    finally:need(execution_binding()==binding,'Capacity/source closure changed before completion')

def _lease(path):
    path=Path(path).resolve();lease=read(path);report=status(lease['ledger_path'])
    need(lease['schema']==SCHEMA+'-lease' and lease['lease_file']==str(path)
         and report['pending']==lease and lease['resource_identity_sha256']==sha(DEFAULT/'identity.json'),
         'Capacity lease is not the active unchanged ledger reservation')
    need(lease['boot_filetime']==boot_identity(),'Windows boot changed: unknown lease overrun')
    verify_operational_binding(lease['operational_v4'])
    need(lease['worker_entry']==str(WORKER) and lease['worker_entry_sha256']==sha(WORKER)
         and lease['capacity_binding']==execution_binding() and lease['protocol']==effective_protocol(),
         'Actual worker or lease effective capacity/source binding differs')
    return lease

def worker_complete(path,result):
    binding=execution_binding();lease=_lease(path);actual=read(lease['worker_identity_file'])
    need(actual['process']==process_identity() and result['source_views_closed_before_completion'] is True,
         'V3 completion lacks actual contained worker and closed views')
    _atomic(lease['completion_file'],dict(token=lease['token'],worker=actual['process'],
      result=dict(result,execution_amendment=binding),completed_unix_ns=time.time_ns(),completed_monotonic_ns=time.monotonic_ns(),
      operational_v4=operational_binding(),capacity_binding=binding,source_views_closed_before_completion=True),True)

def reserve(path,worker_output,cache_path,fixture=False,protocol=None,entry=None,extra_args=()):
    from . import prospective_eval_execute_v3 as contract
    binding=execution_binding();path=_path(path)
    need(not(path/'hold.json').exists(),'Cumulative resource accounting is held')
    entry=Path(entry or WORKER).resolve()
    need(entry==WORKER and fixture is False and list(extra_args)==['--operation','arena'],
         'Capacity extension authorizes only the exact remaining arena worker')
    contract.classify(worker_output,'arena',False)
    need(Path(worker_output).resolve()==stage.FOLDER/'jobs/arena' and (Path(worker_output)/'identity.json').is_file(),
         'Capacity continuation requires the existing arena identity')
    verify_execution_identity(read(Path(worker_output)/'identity.json'),worker_output)
    need(protocol is None and Path(cache_path).resolve()==stage.DATA/'cache','Capacity recipe/cache override forbidden')
    with Writer(path):
        report=status(path);need(report['pending'] is None,'Another cumulative lease remains active')
        p=report['effective_protocol'];need(report['remaining_ns']>30_000_000_000,'Effective cumulative budget exhausted')
        pre=preflight(p);duration=min(p['lease_ns'],report['remaining_ns']);token=uuid.uuid4().hex
        folder=path/'leases'/token;folder.mkdir(parents=True,exist_ok=False)
        lease=dict(schema=SCHEMA+'-lease',token=token,ledger_path=str(path),worker_output=str(Path(worker_output).resolve()),
          cache_path=str(Path(cache_path).resolve()),worker_entry=str(entry),worker_entry_sha256=sha(entry),extra_args=list(extra_args),
          plan_sha256=sha(stage.PLAN),deadline_unix_ns=time.time_ns()+duration,deadline_monotonic_ns=time.monotonic_ns()+duration,
          reserved_ns=duration,fixture=False,worker_identity_file=str(folder/'worker.json'),permit_file=str(folder/'permit.json'),
          launcher_file=str(folder/'launcher.json'),completion_file=str(folder/'completion.json'),supervisor_identity=process_identity(),
          protocol=p,resource_identity_sha256=sha(path/'identity.json'),preflight=pre,lease_file=str(folder/'lease.json'),
          boot_filetime=boot_identity(),operational_v4=operational_binding(),capacity_binding=binding)
        _atomic(folder/'lease.json',lease,True);_append(path,report,dict(kind='reserve',lease=lease))
    return Path(lease['lease_file'])

def launch(worker_output,cache_path=None,fixture=False,stop_after=None,ledger=DEFAULT,python=None,protocol=None,
           pause_at=None,pause_marker=None,entry=None,extra_args=()):
    from . import prospective_eval_execute_v3 as execute, prospective_eval_reservations as reservations
    admit();need(stop_after is None and python is None and protocol is None and pause_at is None and pause_marker is None
                 and fixture is False,'Effective-capacity runtime recipe cannot be overridden')
    need(list(extra_args)==['--operation','arena'],'Only the remaining arena is authorized');operation='arena'
    cache_path=Path(cache_path or stage.DATA/'cache').resolve()
    if execute.has_completion(worker_output,operation):return execute.completed_job(worker_output,operation)
    reservations.verify();execute.preflight_job(worker_output,operation,cache_path,fixture=False)
    lease_path=reserve(ledger,worker_output,cache_path,False,entry=entry,extra_args=extra_args);lease=read(lease_path)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    kwargs=dict(cwd=ROOT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    watcher=subprocess.Popen([sys.executable,str(SUPERVISOR),'watchdog','--lease',str(lease_path)],**kwargs)
    command=[sys.executable,str(WORKER),'run','--output',str(Path(worker_output).resolve()),'--cache',str(cache_path),'--lease',str(lease_path),*lease['extra_args']]
    try:
        process=subprocess.Popen(command,**kwargs)
        _atomic(lease['launcher_file'],dict(token=lease['token'],process=process_identity(process.pid),supervisor=lease['supervisor_identity'],command=command),True)
    except Exception:
        watcher.wait(timeout=40);raise
    process.wait();watcher.wait(timeout=40);settled=status()
    need(settled['pending'] is None and process.returncode==0 and watcher.returncode==0,'Capacity worker failed; retain evidence')
    need(settled['events'][-1]['clean'] is True,'Capacity worker unclean; no automatic query retry')
    done=read(lease['completion_file'])['result']
    need(done['execution_amendment']==execution_binding() and done['state'] in ('complete','checkpoint','held'),'Capacity completion differs')
    return dict(worker_returncode=process.returncode,watchdog_returncode=watcher.returncode,lease=str(lease_path),
      charged_ns=settled['charged_ns'],remaining_ns=settled['remaining_ns'],state=done['state'])

# BEGIN PROJECTED CAPACITY STATE MACHINES
def status(path=DEFAULT):
    capacity=admit();path=_path(path);identity=read(path/'identity.json')
    need(config.canonical(identity)==config.canonical(_identity(identity['protocol'],identity['fixture'])),'Resource identity/source changed')
    head=read(path/'head.json');events=[];prev=None;charged=identity['inherited_charged_ns'];pending=None
    need(type(head.get('count')) is int and head['count']>=0,'Invalid resource highwater count')
    for index in range(head['count']):
        event=read(path/'events'/('%06d.json'%index))
        need(type(event['index']) is int and event['index']==index and event['previous']==prev,'Resource event prefix differs')
        if event['kind']=='reserve':
            need(pending is None,'Overlapping durable leases')
            pending=event['lease']
            need(type(pending['reserved_ns']) is int and 0<pending['reserved_ns']<=(identity['protocol']['lease_ns'] if index<capacity['carried_head']['count'] else capacity['effective_protocol']['lease_ns']), 'Invalid durable reservation')
            charged+=pending['reserved_ns']
        elif event['kind']=='finish':
            need(pending is not None and event['token']==pending['token'],'Resource finish has no reservation')
            need(type(event['elapsed_ns']) is int and event['elapsed_ns']>=0 and type(event['unknown_overrun']) is bool,'Invalid elapsed/overrun evidence')
            need(type(event['charged_ns']) is int and 0<=event['charged_ns']<=pending['reserved_ns'],'Invalid resource charge')
            need(type(event['clean']) is bool and (event['clean'] is True or event['charged_ns']==pending['reserved_ns']),'Unclean lease was not fully charged')
            charged+=event['charged_ns']-pending['reserved_ns'];pending=None
        else:need(False,'Unknown resource event')
        prev=config.digest(event);events.append(event)
    need(config.canonical(head)==config.canonical(dict(schema=SCHEMA+'-head',count=len(events),last_sha256=prev)),'Resource highwater differs')
    files=list((path/'events').glob('*.json')) if (path/'events').exists() else []
    need(len(files)==len(events),'Uncommitted resource event: HOLD; manual conservative reconciliation required')
    lease_files=list((path/'leases').glob('*/lease.json')) if (path/'leases').exists() else []
    reserved={event['lease']['token']:event['lease'] for event in events if event['kind']=='reserve'}
    need(len(lease_files)==len(reserved) and all(p.parent.name in reserved and config.canonical(read(p))==config.canonical(reserved[p.parent.name])
         for p in lease_files),'Resource lease highwater differs: HOLD; cannot roll back reservations')
    carried=capacity['carried_head'];count=carried['count']
    need(type(count) is int and 0<count<=len(events) and carried['last_sha256']==config.digest(events[count-1]),
         'Reviewed pre-extension ledger prefix changed')
    baseline_charge=identity['inherited_charged_ns'];prior_pending=None
    for prior in events[:count]:
        if prior['kind']=='reserve':
            prior_pending=prior['lease'];baseline_charge+=prior_pending['reserved_ns']
            need(prior_pending['protocol']==identity['protocol'],'A historical lease protocol was changed')
        else:
            baseline_charge+=prior['charged_ns']-prior_pending['reserved_ns'];prior_pending=None
        need(baseline_charge<=identity['protocol']['cumulative_ns'],'Historical original cap was exceeded')
    need(prior_pending is None and baseline_charge==capacity['carried_charged_ns'],
         'Capacity handoff was not the exact settled fully charged prefix')
    binding=execution_binding()
    for event in events[count:]:
        if event['kind']=='reserve':
            future=event['lease']
            need(future.get('capacity_binding')==binding and future['protocol']==capacity['effective_protocol']
                 and future['worker_entry']==str(WORKER) and future['worker_entry_sha256']==sha(WORKER),
                 'Prospective lease lacks the reviewed effective capacity and actual entry')
    need(charged<=capacity['effective_protocol']['cumulative_ns'],'Effective cumulative resource cap exceeded')
    old_lease_count=13
    return dict(identity=identity,head=head,events=events,charged_ns=charged,pending=pending,
                remaining_ns=capacity['effective_protocol']['cumulative_ns']-charged,old_ledger_ref=identity['old_ledger'],
                effective_protocol=capacity['effective_protocol'],original_protocol=identity['protocol'],capacity_ref=stage.ref(CAPACITY),
                old_lease_count=old_lease_count,total_lease_count=old_lease_count+sum(e['kind']=='reserve' for e in events))


def bind_worker(lease_path):
    """Called by actual interpreter before Torch; await contained watchdog permit."""
    lease=_lease(lease_path)
    actual=dict(process=process_identity(),argv=sys.argv,cwd=str(Path.cwd().resolve()),token=lease['token'])
    _atomic(lease['worker_identity_file'],actual,True)
    while not Path(lease['permit_file']).exists():
        need(time.monotonic_ns()<lease['deadline_monotonic_ns'],'Worker gate expired before containment')
        time.sleep(0.01)
    permit=read(lease['permit_file'])
    need(permit['token']==lease['token'] and permit['worker']==actual['process'] and permit['kill_on_job_close'] is True,'Watchdog permit differs')
    need(permit.get('capacity_binding')==lease['capacity_binding'],'Worker permit capacity/source binding differs')
    assert_worker_lease(lease_path)
    return lease

def assert_worker_lease(lease_path):
    lease=_lease(lease_path);actual=read(lease['worker_identity_file']);permit=read(lease['permit_file'])
    need(actual['process']==process_identity() and permit['worker']==actual['process'] and permit['token']==lease['token'],'Worker identity/permit differs')
    need(permit.get('capacity_binding')==lease['capacity_binding'],'Actual permit capacity/source binding differs')
    need(_alive(permit['watchdog']),'Independent watchdog missing')
    need(time.monotonic_ns()<lease['deadline_monotonic_ns']-lease['protocol']['clean_stop_margin_ns'],'Training lease must checkpoint and stop')
    return lease

def remaining_ns(lease_path):
    """Validated active lease clock, allowing a cooperative stop before margin."""
    lease=_lease(lease_path)
    need(read(lease['worker_identity_file'])['process']==process_identity(),'Lease clock caller is not actual worker')
    return lease['deadline_monotonic_ns']-time.monotonic_ns()

def watchdog(lease_path):
    lease=_lease(lease_path);p=lease['protocol'];start=time.monotonic_ns();start_wall=time.time_ns()
    k=handle=job=None;reason='startup failure';clean=False;peak=0;worker=None
    try:
        startup=min(lease['deadline_monotonic_ns'],start+p['startup_ns'])
        while not Path(lease['worker_identity_file']).exists() or not Path(lease['launcher_file']).exists():
            need(time.monotonic_ns()<startup,'Worker/launcher handshake missing');time.sleep(p['poll_seconds'])
        record=read(lease['worker_identity_file']);worker=record['process'];launcher=read(lease['launcher_file'])
        need(record['token']==lease['token'] and record['cwd']==str(ROOT.resolve()),'Worker launch token/cwd differs')
        need(process_identity(worker['pid'])==worker,'Actual worker creation/image differs')
        k,handle=_open(worker['pid'],0x1000|0x0400|0x00100000|0x0001|0x0100|0x0010)
        job=_job(k,handle)
        need(launcher['supervisor']==lease['supervisor_identity'] and launcher['token']==lease['token'],'Launcher binding differs')
        # Venv python.exe may launch a second actual interpreter. Admit only the
        # recorded launcher or its direct child, and verify both creation IDs.
        launched=launcher['process'];need(process_identity(launched['pid'])==launched,'Launcher identity no longer verifiable')
        need(launched['parent_pid']==lease['supervisor_identity']['pid'] and
             (worker==launched or worker['parent_pid']==launched['pid']),'Actual worker lineage differs')
        need(Path(record['argv'][0]).resolve()==Path(lease['worker_entry']) and sha(lease['worker_entry'])==lease['worker_entry_sha256'], 'Unexpected or changed worker entry point')
        from . import prospective_eval_common as stage
        need(sha(stage.PLAN)==lease['plan_sha256'],'Worker plan changed')
        expected=[str(Path(lease['worker_entry']).resolve()),'run','--output',lease['worker_output'],'--cache',lease['cache_path'],'--lease',lease['lease_file']]
        if lease['fixture']:expected.append('--engineering-fixture')
        expected.extend(lease['extra_args'])
        need(record['argv']==expected,'Actual worker arguments differ from durable lease')
        _atomic(lease['permit_file'],dict(schema=SCHEMA+'-permit',token=lease['token'],worker=worker,
             watchdog=process_identity(),kill_on_job_close=True,operational_v4=operational_binding(),capacity_binding=execution_binding()),True)
        reason='worker exited without durable completion'
        while k.WaitForSingleObject(handle,0)==258:
            now=time.monotonic_ns();peak=max(peak,private_commit(handle))
            if now>=lease['deadline_monotonic_ns']:reason='deadline';break
            if peak>p['private_commit_bytes']:reason='private commit';break
            if namespace_bytes()>p['quota_bytes']:reason='namespace quota';break
            time.sleep(p['poll_seconds'])
        else:
            if Path(lease['completion_file']).exists():
                done=read(lease['completion_file'])
                need(done['token']==lease['token'] and done['worker']==worker and
                     done['completed_monotonic_ns']<=lease['deadline_monotonic_ns'],'Worker completion binding/time differs')
                need(done.get('capacity_binding')==lease['capacity_binding'] and done.get('source_views_closed_before_completion') is True,
                     'Capacity worker completion lacks exact source closure')
                clean=True;reason='durable completion'
    except Exception as error:
        reason=type(error).__name__+': '+str(error)
    finally:
        if k and handle:
            if k.WaitForSingleObject(handle,0)==258:k.TerminateProcess(handle,124)
            k.WaitForSingleObject(handle,5000)
            if job:k.CloseHandle(job)
            k.CloseHandle(handle)
        elapsed=time.monotonic_ns()-(lease['deadline_monotonic_ns']-lease['reserved_ns'])
        # A delayed watchdog cannot prove that a lease stopped at its deadline.
        overrun=time.monotonic_ns()>lease['deadline_monotonic_ns']+2_000_000_000
        charged=min(elapsed,lease['reserved_ns']) if clean and not overrun else lease['reserved_ns']
        path=Path(lease['ledger_path'])
        with Writer(path):
            report=status(path)
            need(config.canonical(report['pending'])==config.canonical(lease),'Watchdog reservation changed')
            _append(path,report,dict(kind='finish',token=lease['token'],clean=clean and not overrun,
                 charged_ns=charged,elapsed_ns=elapsed,reason=reason,peak_private_commit_bytes=peak,
                 actual_worker=worker,unknown_overrun=overrun,watchdog=process_identity()))
            if overrun:_atomic(path/'hold.json',dict(reason='unknown deadline overrun',token=lease['token']),True)
    return dict(clean=clean and not overrun,reason=reason,charged_ns=charged,peak_private_commit_bytes=peak)

