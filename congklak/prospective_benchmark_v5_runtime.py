"""Carried accounting with fixed-purpose workers and an independent Windows watchdog."""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path
from . import config, neurodiag_common as common, prospective_benchmark_v5_common as stage
from .valuepilot_reservations import Writer
ROOT=common.ROOT
DEFAULT=ROOT/'data/prospective-benchmark-v5/resources'
SCHEMA='prospective-benchmark-v5-resource'
GIB=1024**3
PROTOCOL=dict(lease_ns=1_800_000_000_000,fixture_lease_ns=120_000_000_000,cumulative_ns=18_000_000_000_000,
 private_commit_bytes=4*GIB,minimum_free_commit_bytes=4*GIB,minimum_free_ram_bytes=4*GIB,
 minimum_disk_bytes=2*GIB,quota_bytes=32*GIB,poll_seconds=.025,startup_ns=15_000_000_000,
 clean_stop_margin_ns=2_000_000_000,unknown_overrun='HOLD',unclean_charge='entire durable reservation')
WORKER=ROOT/'congklak/prospective_benchmark_v5_worker.py'
SUPERVISOR=ROOT/'congklak/prospective_benchmark_v5_cli.py'
CARRY=ROOT/'build/prospective-benchmark-v5/resource-carry.json'
need,read,sha=common.require,common.read,common.sha

def _path(path):
 path=Path(path).resolve();need(path==DEFAULT,'One carried benchmark ledger is required');return path

def admit(full=True):
 body=stage.admit_plan();r=body['resource'];carry=read(CARRY)
 for key in ('lease_ns','fixture_lease_ns','cumulative_ns','private_commit_bytes','minimum_free_commit_bytes',
             'minimum_free_ram_bytes','minimum_disk_bytes','quota_bytes','clean_stop_margin_ns'):
  need(r[key]==PROTOCOL[key],'Resource protocol differs: '+key)
 need(r['carry_ref']==stage.ref(CARRY) and carry['passed'] is True and carry['budget_reset'] is False and
      carry['charged_ns']==r['inherited_charged_ns']==12_836_415_000_000 and carry['physical_worker_leases']==29 and
      carry['pending'] is None and r['inherited_namespace_bytes']==carry['inherited_namespace_bytes'],
      'Settled prior resources cannot reset or advance')
 stage.current_inputs(carry['inputs'])
 return body

def operational_binding():
 body=admit(False)
 return dict(schema=SCHEMA+'-operational-binding',plan_ref=stage.ref(stage.PLAN),review_ref=stage.ref(stage.PREFLIGHT),
  physical_sources=body['source_inputs'],models=body['models'],carry_ref=stage.ref(CARRY),resource_protocol=PROTOCOL)

def verify_operational_binding(binding):
 need(binding==operational_binding(),'Actual benchmark source/resource binding changed');return True

def _identity(protocol,fixture):
 need(fixture is False and protocol==PROTOCOL,'Resource protocol override forbidden')
 body=admit(False)
 return dict(schema=SCHEMA+'-identity',protocol=protocol,fixture=False,sources=body['source_inputs'],
  inherited_charged_ns=body['resource']['inherited_charged_ns'],old_ledger=read(CARRY),
  operational_binding=operational_binding())

def namespace_bytes():
 inherited=read(CARRY)['inherited_namespace_bytes']
 return inherited+sum(p.stat().st_size for root in (stage.FOLDER,stage.DATA) if root.exists()
                      for p in root.rglob('*') if p.is_file())

def _purpose_limit(fixture,operation):
 need(type(fixture) is bool and operation in ('engineering','endgame','arena'),'Unknown worker purpose')
 need(not fixture or operation=='arena','Only arena recovery fixtures are registered')
 return PROTOCOL['fixture_lease_ns'] if fixture or operation=='engineering' else PROTOCOL['lease_ns']

def _validate_lease_shape(lease,identity):
 from . import prospective_benchmark_v5_execute as contract
 args=lease['extra_args']
 need(type(args) is list and len(args)%2==0 and all(type(v) is str for v in args) and
      len(set(args[::2]))==len(args[::2]) and set(args[::2])<={'--operation','--pause-at','--pause-marker'} and
      '--operation' in args,'Malformed historical lease options')
 options=dict(zip(lease['extra_args'][::2],lease['extra_args'][1::2]))
 contract.classify(lease['worker_output'],options['--operation'],lease['fixture'])
 need(Path(lease['cache_path']).resolve()==stage.DATA/'cache','Historical cache path changed')
 if '--pause-at' in options:
  need(lease['fixture'] and options['--pause-at'] in contract.HANDOFF_KILL_HOOKS and '--pause-marker' in options,
       'Historical pause scope changed');stage.path_of(options['--pause-marker'])
 need(lease['worker_entry']==str(WORKER) and lease['worker_entry_sha256']==sha(WORKER) and
      lease['protocol']==identity['protocol'] and lease['plan_sha256']==sha(stage.PLAN),
      'Worker source/protocol changed')
 expected=_purpose_limit(lease['fixture'],options['--operation'])
 need(lease['purpose_limit_ns']==expected and lease['reserved_ns']<=expected and
      lease['purpose']==('recovery' if lease['fixture'] else options['--operation']),
      'Engineering/recovery lease limit changed')

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


def status(path=DEFAULT):
    path=_path(path);identity=read(path/'identity.json')
    need(config.canonical(identity)==config.canonical(_identity(identity['protocol'],identity['fixture'])),'Resource identity/source changed')
    head=read(path/'head.json');events=[];prev=None;charged=identity['inherited_charged_ns'];pending=None;seen_tokens=set()
    need(type(head.get('count')) is int and head['count']>=0,'Invalid resource highwater count')
    for index in range(head['count']):
        event=read(path/'events'/('%06d.json'%index))
        need(type(event['index']) is int and event['index']==index and event['previous']==prev,'Resource event prefix differs')
        if event['kind']=='reserve':
            need(pending is None,'Overlapping durable leases')
            pending=event['lease']
            need(pending['token'] not in seen_tokens,'Repeated durable reservation token')
            seen_tokens.add(pending['token'])
            need(type(pending['reserved_ns']) is int and 0<pending['reserved_ns']<=identity['protocol']['lease_ns'], 'Invalid durable reservation')
            _validate_lease_shape(pending,identity)
            charged+=pending['reserved_ns']
            need(charged<=identity['protocol']['cumulative_ns'],'Reservation exceeded cumulative resource cap')
        elif event['kind']=='finish':
            need(pending is not None and event['token']==pending['token'],'Resource finish has no reservation')
            need(type(event['elapsed_ns']) is int and event['elapsed_ns']>=0 and type(event['unknown_overrun']) is bool,'Invalid elapsed/overrun evidence')
            need(type(event['charged_ns']) is int and 0<=event['charged_ns']<=pending['reserved_ns'],'Invalid resource charge')
            need(type(event['clean']) is bool and (event['clean'] is True or event['charged_ns']==pending['reserved_ns']),'Unclean lease was not fully charged')
            need(not event['unknown_overrun'] or (event['clean'] is False and event['charged_ns']==pending['reserved_ns']),'Unknown overrun must be unclean and fully charged')
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
    old_lease_count=29
    return dict(identity=identity,head=head,events=events,charged_ns=charged,pending=pending,
                remaining_ns=identity['protocol']['cumulative_ns']-charged,old_ledger_ref=identity['old_ledger'],
                old_lease_count=old_lease_count,total_lease_count=old_lease_count+sum(e['kind']=='reserve' for e in events))


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


def _lease(path):
    path=Path(path).resolve();lease=read(path)
    need(lease['schema']==SCHEMA+'-lease' and lease['lease_file']==str(path),'Lease path/schema differs')
    report=status(lease['ledger_path'])
    need(config.canonical(report['pending'])==config.canonical(lease) and lease['resource_identity_sha256']==sha(Path(lease['ledger_path'])/'identity.json'),'Lease not durably active')
    need(lease['boot_filetime']==boot_identity(),'Windows boot changed: pending lease has unknown overrun; HOLD')
    verify_operational_binding(lease['operational_binding'])
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
    k=handle=job=None;reason='startup failure';clean=False;peak=0;worker=None;worker_exit_code=None
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
        from . import prospective_benchmark_v5_common as stage
        need(sha(stage.PLAN)==lease['plan_sha256'],'Worker plan changed')
        expected=[str(Path(lease['worker_entry']).resolve()),'run','--output',lease['worker_output'],'--cache',lease['cache_path'],'--lease',lease['lease_file']]
        if lease['fixture']:expected.append('--engineering-fixture')
        expected.extend(lease['extra_args'])
        need(record['argv']==expected,'Actual worker arguments differ from durable lease')
        _atomic(lease['permit_file'],dict(schema=SCHEMA+'-permit',token=lease['token'],worker=worker,
             watchdog=process_identity(),kill_on_job_close=True,operational_binding=operational_binding()),True)
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
                     done['completed_monotonic_ns']<=lease['deadline_monotonic_ns'] and
                     done['operational_binding']==lease['operational_binding'] and
                     done['source_views_closed_before_completion'] is True and
                     done['result']['source_views_closed_before_completion'] is True,'Worker completion binding/time differs')
                clean=True;reason='durable completion'
    except Exception as error:
        reason=type(error).__name__+': '+str(error)
    finally:
        if k and handle:
            if k.WaitForSingleObject(handle,0)==258:k.TerminateProcess(handle,124)
            k.WaitForSingleObject(handle,5000)
            k.GetExitCodeProcess.argtypes=(W.HANDLE,C.POINTER(W.DWORD));k.GetExitCodeProcess.restype=W.BOOL
            exit_value=W.DWORD()
            if k.GetExitCodeProcess(handle,C.byref(exit_value)):worker_exit_code=exit_value.value
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
                 actual_worker=worker,worker_exit_code=worker_exit_code,unknown_overrun=overrun,watchdog=process_identity()))
            if overrun:_atomic(path/'hold.json',dict(reason='unknown deadline overrun',token=lease['token']),True)
    return dict(clean=clean and not overrun,reason=reason,charged_ns=charged,peak_private_commit_bytes=peak)


def reserve(path,worker_output,cache_path,fixture=False,protocol=None,entry=None,extra_args=()):
 from . import prospective_benchmark_v5_execute as contract
 admit();path=_path(path);need(not(path/'hold.json').exists(),'Resource accounting held')
 entry=Path(entry or WORKER).resolve();need(entry==WORKER and protocol is None,'Worker entry/protocol override forbidden')
 need(type(fixture) is bool and len(extra_args)%2==0 and all(type(v) is str for v in extra_args),'Malformed worker options')
 need(len(set(extra_args[::2]))==len(extra_args[::2]) and set(extra_args[::2])<={'--operation','--pause-at','--pause-marker'} and
      '--operation' in extra_args,'Worker option override forbidden')
 options=dict(zip(extra_args[::2],extra_args[1::2]));operation=options['--operation']
 purpose_limit=_purpose_limit(fixture,operation);contract.classify(worker_output,operation,fixture)
 if '--pause-at' in options:
  need(fixture and options['--pause-at'] in contract.HANDOFF_KILL_HOOKS and '--pause-marker' in options,
       'Only declared recovery boundaries may pause')
  stage.path_of(options['--pause-marker'])
 need(Path(cache_path).resolve()==stage.DATA/'cache','Cache marker override forbidden')
 initialize(path,False)
 with Writer(path):
  report=status(path);need(report['pending'] is None,'Another worker lease is active')
  need(report['remaining_ns']>30_000_000_000,'Cumulative resource budget exhausted')
  pre=preflight(PROTOCOL);duration=min(purpose_limit,report['remaining_ns']);token=uuid.uuid4().hex
  folder=path/'leases'/token;folder.mkdir(parents=True,exist_ok=False)
  lease=dict(schema=SCHEMA+'-lease',token=token,ledger_path=str(path),worker_output=str(Path(worker_output).resolve()),
   cache_path=str(Path(cache_path).resolve()),worker_entry=str(entry),worker_entry_sha256=sha(entry),extra_args=list(extra_args),
   plan_sha256=sha(stage.PLAN),deadline_unix_ns=time.time_ns()+duration,deadline_monotonic_ns=time.monotonic_ns()+duration,
   reserved_ns=duration,fixture=fixture,purpose='recovery' if fixture else operation,purpose_limit_ns=purpose_limit,
   worker_identity_file=str(folder/'worker.json'),permit_file=str(folder/'permit.json'),launcher_file=str(folder/'launcher.json'),
   completion_file=str(folder/'completion.json'),supervisor_identity=process_identity(),protocol=PROTOCOL,
   resource_identity_sha256=sha(path/'identity.json'),preflight=pre,lease_file=str(folder/'lease.json'),
   boot_filetime=boot_identity(),operational_binding=operational_binding())
  _atomic(folder/'lease.json',lease,True);_append(path,report,dict(kind='reserve',lease=lease))
 return Path(lease['lease_file'])

def worker_complete(lease_path,result):
 admit();binding=operational_binding();lease=_lease(lease_path);actual=read(lease['worker_identity_file'])
 need(actual['process']==process_identity() and result['source_views_closed_before_completion'] is True,
      'Completion lacks actual worker identity or closed source views')
 _atomic(lease['completion_file'],dict(token=lease['token'],worker=actual['process'],result=result,
  completed_unix_ns=time.time_ns(),completed_monotonic_ns=time.monotonic_ns(),operational_binding=binding,
  source_views_closed_before_completion=True),True)


def launch(worker_output,cache_path=None,fixture=False,stop_after=None,ledger=DEFAULT,python=None,protocol=None,
           pause_at=None,pause_marker=None,entry=None,extra_args=()):
 from . import prospective_benchmark_v5_execute as execute, prospective_benchmark_v2_extension as reservations
 admit();need(stop_after is None and python is None and protocol is None,'Runtime recipe cannot be overridden')
 need(list(extra_args[::2])==['--operation'],'One native operation is required');operation=extra_args[1]
 cache_path=Path(cache_path or stage.DATA/'cache').resolve()
 if execute.has_completion(worker_output,operation):
  done=execute.completed_job(worker_output,operation)
  return execute.publish_completed(worker_output,operation) if done['publication_gap'] else done
 # These potentially long exact-byte reads precede the durable300s lease.
 reservations.verify();execute.preflight_job(worker_output,operation,cache_path,fixture=fixture)
 args=list(extra_args)
 if pause_at is not None:
  need(fixture and pause_marker is not None,'Production pause forbidden')
  args.extend(['--pause-at',str(pause_at),'--pause-marker',str(pause_marker)])
 lease_path=reserve(ledger,worker_output,cache_path,fixture,entry=entry,extra_args=args);lease=read(lease_path)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',PYTHONFAULTHANDLER='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
          OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 kwargs=dict(cwd=ROOT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 with (Path(lease_path).parent/'watchdog.log').open('xb') as watchdog_log, (Path(lease_path).parent/'worker.log').open('xb') as worker_log:
  watchdog_process=subprocess.Popen([sys.executable,'-B','-X','faulthandler','-u',str(SUPERVISOR),'watchdog','--lease',str(lease_path)],stdout=watchdog_log,stderr=subprocess.STDOUT,**kwargs)
  command=[sys.executable,'-B','-X','faulthandler','-u',str(WORKER),'run','--output',str(Path(worker_output).resolve()),'--cache',str(cache_path),'--lease',str(lease_path)]
  if fixture:command.append('--engineering-fixture')
  command.extend(lease['extra_args'])
  try:
   process=subprocess.Popen(command,stdout=worker_log,stderr=subprocess.STDOUT,**kwargs)
   _atomic(lease['launcher_file'],dict(token=lease['token'],process=process_identity(process.pid),supervisor=lease['supervisor_identity'],command=command),True)
  except Exception:
   watchdog_process.wait(timeout=40);raise
  process.wait();watchdog_process.wait(timeout=40);settled=status()
  _atomic(Path(lease_path).parent/'child-exits.json',dict(worker_returncode=process.returncode,watchdog_returncode=watchdog_process.returncode,worker_log=str(worker_log.name),watchdog_log=str(watchdog_log.name),operational_binding=lease['operational_binding']),True)
  need(settled['pending'] is None and process.returncode==0 and watchdog_process.returncode==0,'Contained native worker failed; retain evidence')
  finish=settled['events'][-1];need(finish['clean'] is True,'Worker did not complete cleanly; no automatic query retry')
  done=read(lease['completion_file'])['result'];need(done['state'] in ('complete','checkpoint','held'),'Worker completion state invalid')
  return dict(worker_returncode=process.returncode,watchdog_returncode=watchdog_process.returncode,lease=str(lease_path),
   charged_ns=settled['charged_ns'],remaining_ns=settled['remaining_ns'],state=done['state'])


