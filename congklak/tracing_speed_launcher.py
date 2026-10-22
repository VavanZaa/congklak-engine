"""Source-bound trace-speed supervisor and independently contained worker."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from . import config,tracing_speed_common as stage,tracing_speed_resources as resources

ROOT=stage.ROOT
CLI=ROOT/'tools/run-tracing-speed-worker.py'
SOURCES=('congklak/tracing_speed_launcher.py','tools/run-tracing-speed-worker.py')
SCHEMA='tracing-speed-launcher-v1'
need,read,sha=stage.need,stage.read,stage.sha

def identity():
    return dict(schema=SCHEMA+'-identity',sources=stage.common.pins(SOURCES),
        resource_sources=stage.common.pins(resources.SOURCES),entry=CLI.relative_to(ROOT).as_posix(),
        management_before_completion=True,production_admission=False)

def module(spec):
    if spec['kind']=='timing':
        from . import tracing_speed_timing as result
    elif spec['kind']=='calibration':
        from . import trace_fast_calibration_v1 as result
    else:
        need(spec['kind']=='handoff','Unknown engineering task kind')
        from . import tracing_speed_worker as result
    return result

def validate_spec(path):
    spec=read(path);adapter=identity()
    need(spec['launcher_identity']==adapter,'Launcher source/spec differs')
    need(spec['models_fixed'] is True and spec['policy_fixed'] is True and
         spec['training_counters']==stage.ZERO,'Unadmitted scope')
    stage.common.current(spec['inputs']);return adapter

def verify_permit(lease_path):
    lease=read(lease_path);permit=read(lease['permit_file']);actual=read(lease['worker_identity_file'])
    adapter=validate_spec(lease['cache_path']);launch=read(lease['launcher_file']);att=permit.get('adapter')
    expected=[str(CLI),'watchdog','--output',lease['worker_output'],'--spec',lease['cache_path'],
              '--ledger',lease['ledger_path'],'--lease',str(Path(lease_path).resolve())]
    need(type(att) is dict and att==dict(identity=adapter,lease_sha256=sha(lease_path),
        spec_sha256=sha(lease['cache_path']),worker_output=lease['worker_output'],
        actual_worker_argv=actual['argv'],actual_watchdog_argv=expected) and
        Path(actual['argv'][0]).resolve()==CLI and actual['argv']==launch['command'][1:] and
        launch['adapter']==adapter and permit['worker']==actual['process'] and
        permit['watchdog']==launch['watchdog_process'] and permit['token']==lease['token'] and
        permit['kill_on_job_close'] is True,'Actual contained command/source attestation differs')
    return permit

def watchdog(lease_path):
    lease=resources._lease(lease_path);original=resources._atomic
    def attested(path,body,immutable=False):
        if Path(path).resolve()==Path(lease['permit_file']).resolve():
            adapter=validate_spec(lease['cache_path']);actual=read(lease['worker_identity_file']);launch=read(lease['launcher_file'])
            need(launch['adapter']==adapter and actual['argv']==launch['command'][1:] and
                 Path(actual['argv'][0]).resolve()==CLI and body['watchdog']==resources.process_identity()==launch['watchdog_process'],
                 'Watchdog actual entry/source differs')
            body=dict(body,adapter=dict(identity=adapter,lease_sha256=sha(lease_path),
                spec_sha256=sha(lease['cache_path']),worker_output=lease['worker_output'],
                actual_worker_argv=actual['argv'],actual_watchdog_argv=list(sys.argv)))
        return original(path,body,immutable)
    resources._atomic=attested
    try:return resources.watchdog(lease_path)
    finally:resources._atomic=original

def settled(ledger,lease):
    report=resources.status(ledger)
    need(report['pending'] is None,'Resource lease not settled')
    rows=[e for e in report['events'] if e['kind']=='finish' and e['token']==lease['token']]
    need(len(rows)==1,'Resource finish missing/duplicate');return report,rows[0]

def completed(output):
    """Pure source-bound reader. No reservation, writer or engine constructor."""
    output=stage.path_of(output);source=read(output/'identity.json');spec_path=ROOT/source['spec_path']
    spec=read(spec_path);validate_spec(spec_path);result=module(spec).completed(output)
    seal=read(output/'launcher-completed.json')
    need(seal['schema']==SCHEMA+'-completed' and seal['launcher_identity']==identity() and
         seal['worker_catalog_sha256']==sha(output/'catalog.json') and
         seal['worker_identity_sha256']==sha(output/'identity.json'),'Completed source/catalog seal differs')
    lease=read(seal['lease']);verify_permit(seal['lease'])
    status,finish=settled(lease['ledger_path'],lease)
    need(lease['worker_output']==str(output) and not finish['unknown_overrun'] and
         sha(seal['finish_event'])==seal['finish_event_sha256'] and read(seal['finish_event'])==finish and
         (seal['recovered_catalog_gap'] is True or
          (finish['clean'] is True and seal['worker_returncode']==seal['watchdog_returncode']==0)),
         'Completed resource finish differs')
    return dict(result,launcher_verified=True,resource=dict(charged_ns=status['charged_ns'],remaining_ns=status['remaining_ns']))

def seal_catalog(output,lease_path,recovered=False,codes=(0,0)):
    """Recover a durable catalog/launcher gap without a new lease or search."""
    output=stage.path_of(output);lease=read(lease_path);verify_permit(lease_path)
    status,finish=settled(lease['ledger_path'],lease)
    need(not finish['unknown_overrun'] and lease['worker_output']==str(output),'Unverifiable catalog recovery lease')
    source=read(output/'identity.json');spec=read(ROOT/source['spec_path']);module(spec).completed(output)
    finish_file=Path(lease['ledger_path'])/'events'/f'{finish["index"]:06d}.json'
    stage.atomic(output/'launcher-completed.json',dict(schema=SCHEMA+'-completed',launcher_identity=identity(),
        lease=str(Path(lease_path).resolve()),worker_returncode=codes[0],watchdog_returncode=codes[1],
        recovered_catalog_gap=recovered,worker_catalog_sha256=sha(output/'catalog.json'),
        worker_identity_sha256=sha(output/'identity.json'),finish_event=str(finish_file),finish_event_sha256=sha(finish_file)))
    return completed(output)

def launch(output,spec_path,ledger=resources.DEFAULT,pause_at=None,pause_marker=None,protocol=None):
    output=stage.path_of(output);spec_path=stage.path_of(spec_path);ledger=stage.path_of(ledger)
    adapter=validate_spec(spec_path);spec=read(spec_path)
    if (output/'catalog.json').exists():
        if not (output/'launcher-completed.json').exists():
            status=resources.status(ledger);need(status['pending'] is None,'Catalog lease is still active')
            leases=[e['lease'] for e in status['events'] if e['kind']=='reserve' and e['lease']['worker_output']==str(output)]
            need(leases,'Catalog has no accountable lease');return seal_catalog(output,leases[-1]['lease_file'],True,(None,None))
        return completed(output)
    module(spec).prepare(output,spec_path)
    lease_path=resources.reserve(ledger,output,spec_path,True,protocol);lease=read(lease_path)
    kwargs=dict(cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    wd_command=[sys.executable,str(CLI),'watchdog','--output',str(output),'--spec',str(spec_path),'--ledger',str(ledger),'--lease',str(lease_path)]
    wd=subprocess.Popen(wd_command,**kwargs)
    command=[sys.executable,str(CLI),'worker','--output',str(output),'--spec',str(spec_path),'--ledger',str(ledger),'--lease',str(lease_path)]
    if pause_at:
        need(pause_marker is not None,'Pause marker missing');command+=['--pause-at',pause_at,'--pause-marker',str(pause_marker)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    process=subprocess.Popen(command,env=env,**kwargs)
    resources._atomic(lease['launcher_file'],dict(token=lease['token'],process=resources.process_identity(process.pid),
        supervisor=lease['supervisor_identity'],command=command,watchdog_command=wd_command,
        watchdog_process=resources.process_identity(wd.pid),adapter=adapter),True)
    process.wait();wd.wait(timeout=30);report,finish=settled(ledger,lease)
    need(process.returncode==0 and wd.returncode==0 and finish['clean'] and not finish['unknown_overrun'],
         'Worker/watchdog failed; full evidence and accounting retained: '+str((process.returncode,wd.returncode,finish['reason'])))
    return seal_catalog(output,lease_path,False,(process.returncode,wd.returncode))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','run','worker','verify','resources','watchdog'))
    p.add_argument('--output',type=Path,default=stage.FOLDER/'jobs/engineering-fixed')
    p.add_argument('--spec',type=Path,default=stage.FOLDER/'jobs/engineering-spec.json')
    p.add_argument('--ledger',type=Path,default=resources.DEFAULT);p.add_argument('--lease',type=Path)
    p.add_argument('--pause-at');p.add_argument('--pause-marker',type=Path);a=p.parse_args(argv)
    try:
        if a.operation=='resources':result=resources.status(a.ledger)
        elif a.operation=='watchdog':result=watchdog(a.lease)
        elif a.operation=='verify':result=completed(a.output)
        elif a.operation=='prepare':validate_spec(a.spec);result=module(read(a.spec)).prepare(a.output,a.spec)
        elif a.operation=='worker':
            resources.bind_worker(a.lease);verify_permit(a.lease)
            result=module(read(a.spec)).run(a.output,a.spec,a.lease,a.pause_at,a.pause_marker)
            resources.worker_complete(a.lease,dict(completed=True,catalog_sha256=sha(a.output/'catalog.json')))
        else:result=launch(a.output,a.spec,a.ledger,a.pause_at,a.pause_marker)
        print(config.canonical(result),flush=True);return 0
    except Exception as error:
        print(config.canonical(dict(state='hold',error=type(error).__name__+': '+str(error))),flush=True);return 2
