"""Durable training leases and an independent Windows Job Object watchdog.

The actual Python worker waits for the watchdog's permit before importing Torch.
Only the watchdog holds the kill-on-close Job handle. Killing the supervisor
therefore leaves enforcement alive; killing the watchdog kills its worker.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from . import config, neurodiag_common as common
from .valuepilot_reservations import Writer

ROOT = common.ROOT
DEFAULT = ROOT / 'data/ordinary-epoch-pilot-v1/resources'
SCHEMA = 'ordinary-epoch-resource-v1'
SOURCES = ('congklak/ordinary_epoch_resources.py', 'tools/run-ordinary-epoch.py', 'tools/train-ordinary-epoch.py', 'tools/ordinary-epoch-execute.py', 'tools/audit-ordinary-epoch-training.py')
GIB = 1024 ** 3
PROTOCOL = dict(lease_ns=300_000_000_000, cumulative_ns=10_800_000_000_000,
    private_commit_bytes=4*GIB, minimum_free_commit_bytes=4*GIB,
    minimum_free_ram_bytes=4*GIB, minimum_disk_bytes=2*GIB, quota_bytes=20*GIB,
    poll_seconds=0.025, startup_ns=15_000_000_000, clean_stop_margin_ns=15_000_000_000,
    unknown_overrun='HOLD', unclean_charge='entire durable reservation')
need, read, sha = common.require, common.read, common.sha


def _path(path):
    path = Path(path).resolve()
    need(path == DEFAULT.resolve(), 'One cumulative pilot ledger is required for production and recovery fixtures')
    return path


def _atomic(path, body, immutable=False):
    path=Path(path); text=config.canonical(body)+'\n'
    if immutable and path.exists():
        need(path.read_text(encoding='utf-8')==text, 'Immutable resource record differs'); return body
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    with temp.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path);return body


def _kernel():
    need(os.name=='nt','Windows process containment is required')
    dll=C.WinDLL('kernel32',use_last_error=True)
    dll.OpenProcess.argtypes=(W.DWORD,W.BOOL,W.DWORD);dll.OpenProcess.restype=W.HANDLE
    dll.CloseHandle.argtypes=(W.HANDLE,);dll.CloseHandle.restype=W.BOOL
    dll.GetProcessTimes.argtypes=(W.HANDLE,C.POINTER(W.FILETIME),C.POINTER(W.FILETIME),C.POINTER(W.FILETIME),C.POINTER(W.FILETIME));dll.GetProcessTimes.restype=W.BOOL
    dll.QueryFullProcessImageNameW.argtypes=(W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD));dll.QueryFullProcessImageNameW.restype=W.BOOL
    dll.WaitForSingleObject.argtypes=(W.HANDLE,W.DWORD);dll.WaitForSingleObject.restype=W.DWORD
    dll.TerminateProcess.argtypes=(W.HANDLE,W.UINT);dll.TerminateProcess.restype=W.BOOL
    return dll


def _open(pid, access=0x1000|0x00100000):
    k=_kernel();handle=k.OpenProcess(access,False,int(pid))
    need(bool(handle), 'Cannot open actual process handle: '+str(pid));return k,handle


def boot_identity():
    """Bind cross-process monotonic deadlines to this Windows boot."""
    need(os.name=='nt','Windows boot identity required')
    buffer=(C.c_longlong*6)();returned=W.ULONG();nt=C.WinDLL('ntdll')
    query=nt.NtQuerySystemInformation
    query.argtypes=(W.ULONG,C.c_void_p,W.ULONG,C.POINTER(W.ULONG));query.restype=C.c_long
    need(query(3,C.byref(buffer),C.sizeof(buffer),C.byref(returned))==0,'Cannot read Windows boot identity')
    return int(buffer[0])


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid;k,h=_open(pid)
    try:
        times=[W.FILETIME() for _ in range(4)]
        need(k.GetProcessTimes(h,*[C.byref(v) for v in times]),'Cannot read process creation time')
        buffer=C.create_unicode_buffer(32768);length=W.DWORD(len(buffer))
        need(k.QueryFullProcessImageNameW(h,0,buffer,C.byref(length)),'Cannot read process image')
        class Basic(C.Structure):
            _fields_=[('reserved1',C.c_void_p),('peb',C.c_void_p),('reserved2',C.c_void_p*2),('unique',C.c_size_t),('parent',C.c_size_t)]
        basic=Basic();nt=C.WinDLL('ntdll');query=nt.NtQueryInformationProcess
        query.argtypes=(W.HANDLE,W.ULONG,C.c_void_p,W.ULONG,C.c_void_p);query.restype=C.c_long
        need(query(h,0,C.byref(basic),C.sizeof(basic),None)==0,'Cannot read process lineage')
        return dict(pid=int(pid),creation_filetime=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,
                    image=buffer.value,parent_pid=int(basic.parent))
    finally:k.CloseHandle(h)


def _alive(identity):
    try:
        current=process_identity(identity['pid'])
        if current!=identity:return False
        k,h=_open(identity['pid'])
        try:return k.WaitForSingleObject(h,0)==258
        finally:k.CloseHandle(h)
    except (OSError,ValueError,RuntimeError):return False


def private_commit(handle):
    class Counters(C.Structure):
        _fields_=[('cb',W.DWORD),('PageFaultCount',W.DWORD)]+[(name,C.c_size_t) for name in
          ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
           'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    counters=Counters();counters.cb=C.sizeof(counters)
    psapi=C.WinDLL('psapi',use_last_error=True);fn=psapi.GetProcessMemoryInfo
    fn.argtypes=(W.HANDLE,C.c_void_p,W.DWORD);fn.restype=W.BOOL
    need(fn(handle,C.byref(counters),counters.cb),'Cannot read private commit')
    return int(counters.PrivateUsage)


def namespace_bytes():
    roots=(ROOT/'data/ordinary-epoch-pilot-v1',ROOT/'build/ordinary-epoch-pilot-v1')
    return sum(p.stat().st_size for root in roots if root.exists() for p in root.rglob('*') if p.is_file())


def preflight(protocol=PROTOCOL):
    class Memory(C.Structure):
        _fields_=[('length',W.DWORD),('load',W.DWORD)]+[(name,C.c_ulonglong) for name in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    status=Memory();status.length=C.sizeof(status);k=_kernel()
    k.GlobalMemoryStatusEx.argtypes=(C.POINTER(Memory),);k.GlobalMemoryStatusEx.restype=W.BOOL
    need(k.GlobalMemoryStatusEx(C.byref(status)),'Cannot read Windows commit status')
    disk=shutil.disk_usage(ROOT).free;used=namespace_bytes()
    result=dict(free_commit_bytes=int(status.avail_page),free_ram_bytes=int(status.avail_phys),
                free_disk_bytes=disk,namespace_bytes=used)
    need(result['free_commit_bytes']>=protocol['minimum_free_commit_bytes'],'Insufficient Windows free commit')
    need(result['free_ram_bytes']>=protocol['minimum_free_ram_bytes'],'Insufficient free RAM')
    need(disk>=protocol['minimum_disk_bytes'],'Insufficient disk headroom')
    need(used<=protocol['quota_bytes'],'Training namespace quota exceeded')
    return result


def _identity(protocol, fixture):
    need(type(fixture) is bool,'Fixture flag must be boolean')
    need(fixture is False and config.canonical(protocol)==config.canonical(PROTOCOL),'Pilot resource limits cannot be overridden, including fixtures')
    need(all(type(protocol[k]) is int and protocol[k]>0 for k in ('lease_ns','cumulative_ns','private_commit_bytes','minimum_free_commit_bytes','minimum_free_ram_bytes','minimum_disk_bytes','quota_bytes','startup_ns','clean_stop_margin_ns')),'Invalid resource limits')
    need(type(protocol['poll_seconds']) in (int,float) and 0<protocol['poll_seconds']<=0.1 and protocol['clean_stop_margin_ns']<protocol['lease_ns'],'Invalid watchdog cadence/margin')
    return dict(schema=SCHEMA+'-identity',protocol=protocol,fixture=fixture,sources=common.pins(SOURCES))


def status(path=DEFAULT):
    path=_path(path);identity=read(path/'identity.json')
    need(config.canonical(identity)==config.canonical(_identity(identity['protocol'],identity['fixture'])),'Resource identity/source changed')
    head=read(path/'head.json');events=[];prev=None;charged=0;pending=None
    need(type(head.get('count')) is int and head['count']>=0,'Invalid resource highwater count')
    for index in range(head['count']):
        event=read(path/'events'/('%06d.json'%index))
        need(type(event['index']) is int and event['index']==index and event['previous']==prev,'Resource event prefix differs')
        if event['kind']=='reserve':
            need(pending is None,'Overlapping durable leases')
            pending=event['lease']
            need(type(pending['reserved_ns']) is int and 0<pending['reserved_ns']<=identity['protocol']['lease_ns'], 'Invalid durable reservation')
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
    need(charged<=identity['protocol']['cumulative_ns'],'Cumulative resource cap exceeded')
    return dict(identity=identity,head=head,events=events,charged_ns=charged,pending=pending,
                remaining_ns=identity['protocol']['cumulative_ns']-charged)


def initialize(path=DEFAULT,fixture=False,protocol=None):
    path=_path(path);protocol=dict(PROTOCOL if protocol is None else protocol)
    if (path/'identity.json').exists():
        report=status(path);need(config.canonical(report['identity'])==config.canonical(_identity(protocol,fixture)),'Existing resource settings differ');return report
    need(not path.exists() or not any(p.name!='writer.lock' for p in path.iterdir()),
         'Missing resource identity with existing artifacts: HOLD; cannot reset accounting')
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        _atomic(path/'identity.json',_identity(protocol,fixture),True)
        _atomic(path/'head.json',dict(schema=SCHEMA+'-head',count=0,last_sha256=None),True)
    return status(path)


def _append(path,report,body):
    index=report['head']['count'];event=dict(body,index=index,previous=report['head']['last_sha256'])
    _atomic(path/'events'/('%06d.json'%index),event,True)
    _atomic(path/'head.json',dict(schema=SCHEMA+'-head',count=index+1,last_sha256=config.digest(event)))
    return event


def reserve(path,worker_output,cache_path,fixture=False,protocol=None,entry=None,extra_args=()):
    path=_path(path);need(not (path/'hold.json').exists(),'Resource accounting is on HOLD')
    from . import ordinary_epoch_common as stage
    stage.admit_plan()
    entry=Path(entry or ROOT/'tools/train-ordinary-epoch.py').resolve()
    need(entry in (ROOT/'tools/train-ordinary-epoch.py',ROOT/'tools/ordinary-epoch-execute.py',ROOT/'tools/audit-ordinary-epoch-training.py'),'Unregistered pilot entry')
    need(all(type(x) is str for x in extra_args),'Invalid worker arguments')
    allowed=({'--operation','--stop-after','--pause-at','--pause-marker'} if entry.name in ('ordinary-epoch-execute.py','audit-ordinary-epoch-training.py')
             else {'--stop-after','--pause-at','--pause-marker'})
    need(len(extra_args)%2==0 and len(set(extra_args[::2]))==len(extra_args[::2]) and
         all(x in allowed for x in extra_args[::2]),'Unregistered or duplicate worker option')
    if '--operation' in extra_args:
        operations=(('verify_fixture_states','verify_production_state') if entry.name=='audit-ordinary-epoch-training.py' else ('labels','arena','conformance','guards'))
        need(extra_args[extra_args.index('--operation')+1] in operations,'Unregistered native/audit operation')
    if '--pause-at' in extra_args:need(fixture is True,'Production pause hook forbidden')
    stage.path_of(worker_output);stage.path_of(cache_path)
    need(type(fixture) is bool,'Worker fixture flag must be boolean')
    initialize(path,False,protocol)
    with Writer(path):
        report=status(path);need(report['pending'] is None,'Previous lease has unknown overrun: HOLD')
        p=report['identity']['protocol'];need(report['remaining_ns']>p['clean_stop_margin_ns'],'Cumulative training budget exhausted')
        pre=preflight(p);duration=min(p['lease_ns'],report['remaining_ns']);token=uuid.uuid4().hex
        folder=path/'leases'/token;folder.mkdir(parents=True,exist_ok=False)
        lease=dict(schema='ordinary-epoch-resource-lease-v1',token=token,ledger_path=str(path),
            worker_output=str(Path(worker_output).resolve()),cache_path=str(Path(cache_path).resolve()),
            worker_entry=str(entry),worker_entry_sha256=sha(entry),extra_args=list(extra_args),plan_sha256=sha(stage.PLAN),
            deadline_unix_ns=time.time_ns()+duration,deadline_monotonic_ns=time.monotonic_ns()+duration,
            reserved_ns=duration,fixture=fixture,
            worker_identity_file=str(folder/'worker.json'),permit_file=str(folder/'permit.json'),
            launcher_file=str(folder/'launcher.json'),completion_file=str(folder/'completion.json'),
            supervisor_identity=process_identity(),protocol=p,resource_identity_sha256=sha(path/'identity.json'),
            preflight=pre,lease_file=str(folder/'lease.json'),boot_filetime=boot_identity())
        _atomic(folder/'lease.json',lease,True)
        _append(path,report,dict(kind='reserve',lease=lease))
    return Path(lease['lease_file'])


def _lease(path):
    path=Path(path).resolve();lease=read(path)
    need(lease['schema']=='ordinary-epoch-resource-lease-v1' and lease['lease_file']==str(path),'Lease path/schema differs')
    report=status(lease['ledger_path'])
    need(config.canonical(report['pending'])==config.canonical(lease) and lease['resource_identity_sha256']==sha(Path(lease['ledger_path'])/'identity.json'),'Lease not durably active')
    need(lease['boot_filetime']==boot_identity(),'Windows boot changed: pending lease has unknown overrun; HOLD')
    return lease


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
    assert_worker_lease(lease_path)
    return lease


def assert_worker_lease(lease_path):
    lease=_lease(lease_path);actual=read(lease['worker_identity_file']);permit=read(lease['permit_file'])
    need(actual['process']==process_identity() and permit['worker']==actual['process'] and permit['token']==lease['token'],'Worker identity/permit differs')
    need(_alive(permit['watchdog']),'Independent watchdog missing')
    need(time.monotonic_ns()<lease['deadline_monotonic_ns']-lease['protocol']['clean_stop_margin_ns'],'Training lease must checkpoint and stop')
    return lease

def remaining_ns(lease_path):
    """Validated active lease clock, allowing a cooperative stop before margin."""
    lease=_lease(lease_path)
    need(read(lease['worker_identity_file'])['process']==process_identity(),'Lease clock caller is not actual worker')
    return lease['deadline_monotonic_ns']-time.monotonic_ns()


def worker_complete(lease_path,result):
    lease=_lease(lease_path);actual=read(lease['worker_identity_file'])
    need(actual['process']==process_identity(),'Completion writer is not actual worker')
    _atomic(lease['completion_file'],dict(token=lease['token'],worker=actual['process'],result=result,
         completed_unix_ns=time.time_ns(),completed_monotonic_ns=time.monotonic_ns()),True)


def _job(k,handle):
    class Basic(C.Structure):
        _fields_=[('ProcessTime',C.c_longlong),('JobTime',C.c_longlong),('Flags',W.DWORD),
          ('Minimum',C.c_size_t),('Maximum',C.c_size_t),('Active',W.DWORD),('Affinity',C.c_size_t),
          ('Priority',W.DWORD),('Scheduling',W.DWORD)]
    class IO(C.Structure):_fields_=[(n,C.c_ulonglong) for n in ('ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes')]
    class Extended(C.Structure):_fields_=[('Basic',Basic),('IO',IO),('ProcessMemoryLimit',C.c_size_t),('JobMemoryLimit',C.c_size_t),('PeakProcessMemory',C.c_size_t),('PeakJobMemory',C.c_size_t)]
    k.CreateJobObjectW.argtypes=(C.c_void_p,W.LPCWSTR);k.CreateJobObjectW.restype=W.HANDLE
    k.SetInformationJobObject.argtypes=(W.HANDLE,C.c_int,C.c_void_p,W.DWORD);k.SetInformationJobObject.restype=W.BOOL
    k.AssignProcessToJobObject.argtypes=(W.HANDLE,W.HANDLE);k.AssignProcessToJobObject.restype=W.BOOL
    job=k.CreateJobObjectW(None,None);need(bool(job),'Cannot create watchdog Job Object')
    info=Extended();info.Basic.Flags=0x00002000
    if not k.SetInformationJobObject(job,9,C.byref(info),C.sizeof(info)) or not k.AssignProcessToJobObject(job,handle):
        error=C.get_last_error();k.CloseHandle(job);need(False,'Cannot contain actual worker in Job Object: '+str(error))
    return job


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
        need(launcher['supervisor']==lease['supervisor_identity'] and launcher['token']==lease['token'],'Launcher binding differs')
        # Venv python.exe may launch a second actual interpreter. Admit only the
        # recorded launcher or its direct child, and verify both creation IDs.
        launched=launcher['process'];need(process_identity(launched['pid'])==launched,'Launcher identity no longer verifiable')
        need(launched['parent_pid']==lease['supervisor_identity']['pid'] and
             (worker==launched or worker['parent_pid']==launched['pid']),'Actual worker lineage differs')
        need(Path(record['argv'][0]).resolve()==Path(lease['worker_entry']) and sha(lease['worker_entry'])==lease['worker_entry_sha256'], 'Unexpected or changed worker entry point')
        from . import ordinary_epoch_common as stage
        need(sha(stage.PLAN)==lease['plan_sha256'],'Worker plan changed')
        expected=[str(Path(lease['worker_entry']).resolve()),'run','--output',lease['worker_output'],'--cache',lease['cache_path'],'--lease',lease['lease_file']]
        if lease['fixture']:expected.append('--engineering-fixture')
        expected.extend(lease['extra_args'])
        need(record['argv']==expected,'Actual worker arguments differ from durable lease')
        k,handle=_open(worker['pid'],0x1000|0x0400|0x00100000|0x0001|0x0100|0x0010)
        job=_job(k,handle)
        _atomic(lease['permit_file'],dict(schema=SCHEMA+'-permit',token=lease['token'],worker=worker,
             watchdog=process_identity(),kill_on_job_close=True),True)
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


def launch(worker_output,cache_path,fixture=False,stop_after=None,ledger=DEFAULT,python=None,protocol=None,pause_at=None,pause_marker=None,entry=None,extra_args=()):
    """Supervisor orchestration; the watchdog, not this process, owns enforcement."""
    from . import ordinary_epoch_common as stage
    stage.path_of(worker_output);stage.path_of(cache_path)
    # Pure completion/dependency checks precede even a durable reservation. A
    # repeated completed command must not create a watchdog or burn a lease.
    entry=Path(entry or ROOT/'tools/train-ordinary-epoch.py').resolve()
    if entry==ROOT/'tools/train-ordinary-epoch.py':
        from . import ordinary_epoch_training as training,ordinary_epoch_data as data,ordinary_epoch_sidecar as side
        if training._has_completion(worker_output):return training.completed(worker_output)
        data.admit(cache_path);side.verify_initial()
        if not fixture:training.production_gate()
    elif entry==ROOT/'tools/ordinary-epoch-execute.py':
        from . import ordinary_epoch_execute as execute
        need(list(extra_args[::2])==['--operation'],'Executor requires one frozen operation')
        operation=extra_args[1]
        if execute.has_completion(worker_output,operation):
            done=execute.completed_job(worker_output,operation)
            return execute.publish_completed(worker_output,operation) if done['publication_gap'] else done
        execute.preflight_job(worker_output,operation,cache_path,fixture=fixture)
    elif entry==ROOT/'tools/audit-ordinary-epoch-training.py':
        import importlib.util
        need(list(extra_args[::2])==['--operation'],'Audit requires one frozen operation')
        spec=importlib.util.spec_from_file_location('_ordinary_epoch_replay_reader',ROOT/'tests/ordinary_epoch_training_replay.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        output=Path(worker_output).resolve();cache=Path(cache_path).resolve();operation=extra_args[1]
        need(operation in module.OPERATIONS,'Unknown diagnostic replay operation')
        expected=stage.FOLDER/('audit/training-replay-fixtures' if operation=='verify_fixture_states' else 'audit/training-replay-production')
        need(output==expected and cache==stage.DATA/'cache-v2','Diagnostic replay output/cache differs')
        if (output/'result.json').exists():return module.completed(output,operation,cache)
        module.bindings(operation,cache)
    else:need(False,'Unregistered pilot entry')
    ledger=_path(ledger);need(not (ledger/'hold.json').exists(),'Resource accounting is on HOLD')
    args=list(extra_args)
    if stop_after is not None:args.extend(['--stop-after',str(stop_after)])
    if pause_at is not None:
        need(fixture and pause_marker is not None,'Production pause hook forbidden')
        args.extend(['--pause-at',str(pause_at),'--pause-marker',str(pause_marker)])
    lease_path=reserve(ledger,worker_output,cache_path,fixture,protocol,entry,args);lease=read(lease_path)
    python=python or read(ROOT/'build/training-runtime.json')['python']
    supervisor_entry=ROOT/'tools/run-ordinary-epoch.py'
    watchdog_command=[sys.executable,str(supervisor_entry),'watchdog','--lease',str(lease_path)]
    kwargs=dict(cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    watchdog_process=subprocess.Popen(watchdog_command,**kwargs)
    command=[str(python),lease['worker_entry'],'run','--output',str(Path(worker_output).resolve()),
             '--cache',str(Path(cache_path).resolve()),'--lease',str(lease_path)]
    if fixture:command.append('--engineering-fixture')
    command.extend(lease['extra_args'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        process=subprocess.Popen(command,env=env,**kwargs)
        _atomic(lease['launcher_file'],dict(token=lease['token'],process=process_identity(process.pid),
             supervisor=lease['supervisor_identity'],command=command),True)
    except Exception:
        # Watchdog will charge this already durable reservation in full.
        watchdog_process.wait(timeout=20);raise
    process.wait();watchdog_process.wait(timeout=20)
    report=status(ledger)
    need(report['pending'] is None,'Watchdog did not settle resource lease')
    return dict(worker_returncode=process.returncode,watchdog_returncode=watchdog_process.returncode,
                lease=str(lease_path),charged_ns=report['charged_ns'],remaining_ns=report['remaining_ns'])
