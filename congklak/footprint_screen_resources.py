"""One bounded acquisition ledger carrying every sealed parent charge forward."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time,uuid
from . import config,ordinary_epoch_resources as old,footprint_screen_common as s
from .valuepilot_reservations import Writer

ROOT=s.ROOT
DEFAULT=s.DATA/'resources'
SCHEMA=s.SCHEMA+'-resource'
ENTRY=ROOT/'tools/footprint-screen-worker.py'
AUDIT_ENTRY=ROOT/'tools/audit-footprint-screen.py'
RESERVATION_ENTRY=ROOT/'tools/reserve-footprint-screen.py'
SUPERVISOR=ROOT/'tools/run-footprint-screen.py'
SOURCES=('congklak/footprint_screen_resources.py','congklak/footprint_screen_common.py',
         'congklak/ordinary_epoch_resources.py','congklak/footprint_screen_worker.py',
         'tools/footprint-screen-worker.py','tools/run-footprint-screen.py',
         'congklak/footprint_screen_review.py','tools/audit-footprint-screen.py',
         'congklak/footprint_screen_reservations.py','tools/reserve-footprint-screen.py')
PROTOCOL=dict(old.PROTOCOL,quota_poll_seconds=1.0)
need,read,sha=s.need,s.read,s.sha
_kernel,_open,boot_identity,process_identity=old._kernel,old._open,old.boot_identity,old.process_identity
_alive,private_commit,_job=old._alive,old.private_commit,old._job

def _same(a,b):return config.canonical(a)==config.canonical(b)
def _path(path):
    p=Path(path).resolve();need(p==DEFAULT.resolve(),'One carry-forward acquisition ledger required');return p
def _atomic(path,body,immutable=False):
    if immutable:return s.atomic(path,body)
    return old._atomic(path,body)
def namespace_bytes():
    roots=(s.FOLDER,s.DATA,ROOT/'build/ordinary-epoch-pilot-v1',ROOT/'data/ordinary-epoch-pilot-v1')
    return sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())
def check_namespace(additional_bytes=0):
    need(type(additional_bytes) is int and additional_bytes>=0,'Invalid publication byte reservation')
    used=namespace_bytes()
    need(used+additional_bytes<=PROTOCOL['quota_bytes'],'Combined artifact publication quota exceeded')
    return dict(namespace_bytes=used,additional_bytes=additional_bytes,quota_bytes=PROTOCOL['quota_bytes'])
def preflight():
    # The inherited helper checks real Windows commit/RAM/disk. The additional
    # combined size includes both old and new namespaces, retaining one quota.
    result=old.preflight(PROTOCOL);result['namespace_bytes']=namespace_bytes()
    need(result['namespace_bytes']<=PROTOCOL['quota_bytes'],'Combined namespace quota exceeded')
    return result
def _identity():
    parent=s.parent_budget()
    return dict(schema=SCHEMA+'-identity',protocol=PROTOCOL,parent=parent,
        sources=s.common.pins(SOURCES),plan_sha256=sha(s.PLAN))
def status(path=DEFAULT):
    path=_path(path);identity=read(path/'identity.json')
    need(_same(identity,_identity()),'Acquisition resource identity/source changed')
    head=read(path/'head.json');need(type(head.get('count')) is int and head['count']>=0,'Resource head count invalid')
    events=[];previous=None;local=0;pending=None;reserved={}
    for index in range(head['count']):
        event=read(path/'events'/f'{index:06d}.json')
        need(type(event['index']) is int and event['index']==index and event['previous']==previous,'Resource prefix changed')
        if event['kind']=='reserve':
            need(pending is None,'Concurrent acquisition leases refused')
            pending=event['lease'];need(type(pending['reserved_ns']) is int and
                0<pending['reserved_ns']<=PROTOCOL['lease_ns'],'Invalid lease duration')
            need(pending['token'] not in reserved,'Duplicate resource token')
            reserved[pending['token']]=pending;local+=pending['reserved_ns']
        elif event['kind']=='finish':
            need(pending is not None and event['token']==pending['token'],'Resource finish lacks active lease')
            need(type(event['charged_ns']) is int and 0<=event['charged_ns']<=pending['reserved_ns'] and
                type(event['clean']) is bool and type(event['unknown_overrun']) is bool and
                type(event['elapsed_ns']) is int and event['elapsed_ns']>=0 and
                (event['clean'] or event['charged_ns']==pending['reserved_ns']), 'Resource charge invalid')
            local+=event['charged_ns']-pending['reserved_ns'];pending=None
        else:need(False,'Unknown resource event')
        previous=config.digest(event);events.append(event)
    need(_same(head,dict(schema=SCHEMA+'-head',count=len(events),last_sha256=previous)),'Resource head differs')
    files=list((path/'events').glob('*.json'));leases=list((path/'leases').glob('*/lease.json'))
    need(len(files)==len(events) and len(leases)==len(reserved) and
         all(p.parent.name in reserved and _same(read(p),reserved[p.parent.name]) for p in leases),
         'Uncommitted resource journal; conservative reconciliation required')
    total=identity['parent']['charged_ns']+local
    need(total<=PROTOCOL['cumulative_ns'],'Carry-forward cumulative worker budget exceeded')
    return dict(identity=identity,head=head,events=events,local_charged_ns=local,
        inherited_charged_ns=identity['parent']['charged_ns'],charged_ns=total,pending=pending,
        remaining_ns=PROTOCOL['cumulative_ns']-total)
def initialize(path=DEFAULT):
    path=_path(path)
    if (path/'identity.json').exists():return status(path)
    need(not path.exists() or not any(p.name!='writer.lock' for p in path.iterdir()),'Missing resource identity refuses budget reset')
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        _atomic(path/'identity.json',_identity(),True)
        _atomic(path/'head.json',dict(schema=SCHEMA+'-head',count=0,last_sha256=None),True)
    return status(path)
def _append(path,report,body):
    index=report['head']['count'];event=dict(body,index=index,previous=report['head']['last_sha256'])
    _atomic(path/'events'/f'{index:06d}.json',event,True)
    _atomic(path/'head.json',dict(schema=SCHEMA+'-head',count=index+1,last_sha256=config.digest(event)))
    return event
def _implementation(entry):
    entry=Path(entry).resolve()
    if entry==ENTRY:
        from . import footprint_screen_worker as implementation
    elif entry==AUDIT_ENTRY:
        from . import footprint_screen_review as implementation
    elif entry==RESERVATION_ENTRY:
        from . import footprint_screen_reservations as implementation
    else:need(False,'Unregistered footprint-screen entry')
    return implementation
def reserve(output,fixture=False,extra_args=(),entry=ENTRY):
    s.admit_plan();path=DEFAULT;need(not(path/'hold.json').exists(),'Resource accounting held')
    need(type(fixture) is bool and all(type(a) is str for a in extra_args),'Worker argument types differ')
    need(len(extra_args)%2==0 and len(set(extra_args[::2]))==len(extra_args[::2]) and
         all(a in ('--stop-after','--pause-at','--pause-marker') for a in extra_args[::2]),'Unexpected worker option')
    need('--pause-at' not in extra_args or fixture,'Only declared recovery fixtures may pause')
    entry=Path(entry).resolve();worker=_implementation(entry)
    need(entry!=RESERVATION_ENTRY or '--stop-after' not in extra_args,'Reservation publication has no query cursor')
    worker.preflight(output,fixture=fixture)
    initialize()
    with Writer(path):
        report=status();need(report['pending'] is None,'Previous acquisition lease unsettled')
        need(report['remaining_ns']>PROTOCOL['clean_stop_margin_ns'],'Carry-forward worker budget exhausted')
        before=preflight();duration=min(PROTOCOL['lease_ns'],report['remaining_ns']);token=uuid.uuid4().hex
        folder=path/'leases'/token;folder.mkdir(parents=True,exist_ok=False)
        lease=dict(schema=SCHEMA+'-lease',token=token,ledger_path=str(path),worker_output=str(s.path_of(output)),
            worker_entry=str(entry),worker_entry_sha256=sha(entry),extra_args=list(extra_args),fixture=fixture,
            plan_sha256=sha(s.PLAN),deadline_unix_ns=time.time_ns()+duration,
            deadline_monotonic_ns=time.monotonic_ns()+duration,reserved_ns=duration,
            worker_identity_file=str(folder/'worker.json'),permit_file=str(folder/'permit.json'),
            launcher_file=str(folder/'launcher.json'),completion_file=str(folder/'completion.json'),
            supervisor_identity=process_identity(),protocol=PROTOCOL,resource_identity_sha256=sha(path/'identity.json'),
            preflight=before,lease_file=str(folder/'lease.json'),boot_filetime=boot_identity())
        _atomic(folder/'lease.json',lease,True);_append(path,report,dict(kind='reserve',lease=lease))
    return folder/'lease.json'
def _lease(path):
    path=Path(path).resolve();lease=read(path)
    need(lease['schema']==SCHEMA+'-lease' and lease['lease_file']==str(path),'Lease path/schema changed')
    report=status(lease['ledger_path'])
    need(_same(report['pending'],lease) and lease['resource_identity_sha256']==sha(DEFAULT/'identity.json'),
         'Lease not durably active')
    need(lease['boot_filetime']==boot_identity(),'Windows boot changed with unsettled lease')
    return lease
def bind_worker(path):
    lease=_lease(path);actual=dict(process=process_identity(),argv=sys.argv,cwd=str(ROOT.resolve()),token=lease['token'])
    need(Path.cwd().resolve()==ROOT.resolve(),'Worker cwd differs')
    _atomic(lease['worker_identity_file'],actual,True)
    while not Path(lease['permit_file']).exists():
        need(time.monotonic_ns()<lease['deadline_monotonic_ns'],'Containment permit timed out');time.sleep(.01)
    permit=read(lease['permit_file'])
    need(permit['worker']==actual['process'] and permit['token']==lease['token'] and permit['kill_on_job_close'] is True,
         'Actual worker permit differs')
    assert_worker_lease(path);return lease
def assert_worker_lease(path):
    lease=_lease(path);actual=read(lease['worker_identity_file']);permit=read(lease['permit_file'])
    need(actual['process']==process_identity() and permit['worker']==actual['process'] and
         permit['token']==lease['token'] and _alive(permit['watchdog']),'Worker identity/watchdog differs')
    need(time.monotonic_ns()<lease['deadline_monotonic_ns']-PROTOCOL['clean_stop_margin_ns'],'Worker must checkpoint and stop')
    return lease
def remaining_ns(path):
    lease=_lease(path);need(read(lease['worker_identity_file'])['process']==process_identity(),'Lease clock caller differs')
    return lease['deadline_monotonic_ns']-time.monotonic_ns()
def worker_complete(path,result):
    lease=_lease(path);actual=read(lease['worker_identity_file']);need(actual['process']==process_identity(),'Completion caller differs')
    _atomic(lease['completion_file'],dict(token=lease['token'],worker=actual['process'],result=result,
        completed_unix_ns=time.time_ns(),completed_monotonic_ns=time.monotonic_ns()),True)
def watchdog(path):
    lease=_lease(path);p=PROTOCOL;start=time.monotonic_ns();k=handle=job=None;worker=None;reason='startup';clean=False;peak=0
    try:
        startup=min(lease['deadline_monotonic_ns'],start+p['startup_ns'])
        while not Path(lease['worker_identity_file']).exists() or not Path(lease['launcher_file']).exists():
            need(time.monotonic_ns()<startup,'Worker/launcher handshake missing');time.sleep(p['poll_seconds'])
        record=read(lease['worker_identity_file']);worker=record['process'];launcher=read(lease['launcher_file'])
        need(record['token']==lease['token'] and record['cwd']==str(ROOT.resolve()),'Actual worker token/cwd differs')
        need(process_identity(worker['pid'])==worker and launcher['supervisor']==lease['supervisor_identity'] and
             launcher['token']==lease['token'],'Actual process creation/launcher differs')
        launched=launcher['process'];need(process_identity(launched['pid'])==launched,'Launcher process differs')
        need(launched['parent_pid']==lease['supervisor_identity']['pid'] and
             (worker==launched or worker['parent_pid']==launched['pid']),'Actual worker lineage differs')
        entry=Path(lease['worker_entry']);need(entry in (ENTRY,AUDIT_ENTRY,RESERVATION_ENTRY),'Unregistered actual worker entry')
        need(Path(record['argv'][0]).resolve()==entry and sha(entry)==lease['worker_entry_sha256'] and
             sha(s.PLAN)==lease['plan_sha256'],'Worker source/entry/plan changed')
        expected=[str(entry),'run','--output',lease['worker_output'],'--lease',lease['lease_file']]
        if lease['fixture']:expected.append('--engineering-fixture')
        expected.extend(lease['extra_args']);need(record['argv']==expected,'Actual worker argv differs')
        k,handle=_open(worker['pid'],0x1000|0x0400|0x00100000|0x0001|0x0100|0x0010);job=_job(k,handle)
        _atomic(lease['permit_file'],dict(schema=SCHEMA+'-permit',token=lease['token'],worker=worker,
            watchdog=process_identity(),kill_on_job_close=True),True)
        reason='exit without durable completion';next_quota=time.monotonic_ns()
        while k.WaitForSingleObject(handle,0)==258:
            now=time.monotonic_ns();peak=max(peak,private_commit(handle))
            if now>=lease['deadline_monotonic_ns']:reason='deadline';break
            if peak>p['private_commit_bytes']:reason='private commit';break
            if now>=next_quota:
                if namespace_bytes()>p['quota_bytes']:reason='combined namespace quota';break
                next_quota=now+int(p['quota_poll_seconds']*1000000000)
            time.sleep(p['poll_seconds'])
        else:
            if Path(lease['completion_file']).exists():
                done=read(lease['completion_file'])
                need(done['token']==lease['token'] and done['worker']==worker and
                     done['completed_monotonic_ns']<=lease['deadline_monotonic_ns'],'Worker completion identity/time differs')
                clean=True;reason='durable completion'
    except Exception as error:reason=type(error).__name__+': '+str(error)
    finally:
        if k and handle:
            if k.WaitForSingleObject(handle,0)==258:k.TerminateProcess(handle,124)
            k.WaitForSingleObject(handle,5000)
            if job:k.CloseHandle(job)
            k.CloseHandle(handle)
        now=time.monotonic_ns();elapsed=now-(lease['deadline_monotonic_ns']-lease['reserved_ns'])
        overrun=now>lease['deadline_monotonic_ns']+2000000000
        charged=min(elapsed,lease['reserved_ns']) if clean and not overrun else lease['reserved_ns']
        with Writer(DEFAULT):
            report=status();need(_same(report['pending'],lease),'Watchdog active reservation differs')
            _append(DEFAULT,report,dict(kind='finish',token=lease['token'],clean=clean and not overrun,
                charged_ns=charged,elapsed_ns=elapsed,reason=reason,peak_private_commit_bytes=peak,
                actual_worker=worker,unknown_overrun=overrun,watchdog=process_identity()))
            if overrun:_atomic(DEFAULT/'hold.json',dict(reason='unknown deadline overrun',token=lease['token']),True)
    return dict(clean=clean and not overrun,reason=reason,charged_ns=charged,peak_private_commit_bytes=peak)
def launch(output,fixture=False,stop_after=None,pause_at=None,pause_marker=None,entry=ENTRY):
    entry=Path(entry).resolve();worker=_implementation(entry)
    output=s.path_of(output)
    # The strict pure reader precedes every mutable operation and process launch.
    if worker.has_completion(output):
        done=worker.completed(output)
        return worker.publish_completed(output) if done.get('publication_gap') else done
    s.admit_plan();worker.preflight(output,fixture=fixture)
    args=[]
    if stop_after is not None:
        need(type(stop_after) is int and stop_after>0,'Invalid cooperative cursor');args+=['--stop-after',str(stop_after)]
    if pause_at is not None:
        need(fixture and pause_marker is not None,'Pause requires declared recovery fixture')
        args+=['--pause-at',str(pause_at),'--pause-marker',str(s.path_of(pause_marker))]
    lease_path=reserve(output,fixture,args,entry);lease=read(lease_path)
    kwargs=dict(cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    watchdog_process=subprocess.Popen([sys.executable,str(SUPERVISOR),'watchdog','--lease',str(lease_path)],**kwargs)
    command=[sys.executable,str(entry),'run','--output',str(output),'--lease',str(lease_path)]
    if fixture:command.append('--engineering-fixture')
    command.extend(args)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        process=subprocess.Popen(command,env=env,**kwargs)
        _atomic(lease['launcher_file'],dict(token=lease['token'],process=process_identity(process.pid),
            supervisor=lease['supervisor_identity'],command=command),True)
    except Exception:watchdog_process.wait(timeout=20);raise
    process.wait();watchdog_process.wait(timeout=20);report=status();need(report['pending'] is None,'Lease settlement missing')
    return dict(worker_returncode=process.returncode,watchdog_returncode=watchdog_process.returncode,
        lease=str(lease_path),inherited_charged_ns=report['inherited_charged_ns'],
        charged_ns=report['charged_ns'],remaining_ns=report['remaining_ns'])
