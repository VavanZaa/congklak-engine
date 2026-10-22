"""Source-bound launcher adapter; management operations precede completed readers.

The accepted worker and lease/watchdog implementation remain immutable. A
source-bound engineering entry shim uses the old watchdog's explicit build/
fixture allow-list. No argv path is disguised and actual process identities
remain the independent watchdog's authority.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
from . import config,neurodiag_common as common
from . import ownership_handoff_worker as worker,ownership_handoff_resources as resources

ROOT=common.ROOT
CLI=ROOT/'tools/run-ownership-handoff-v2.py'
ENTRY=ROOT/'build/ownership-implementation-v1/launcher-v2/worker-entry.py'
SOURCES=('congklak/ownership_handoff_launcher_v2.py','tools/run-ownership-handoff-v2.py',
         'build/ownership-implementation-v1/launcher-v2/worker-entry.py')
SCHEMA='ownership-handoff-launcher-v2'
need,read,sha=common.require,common.read,common.sha

def identity():
    return dict(schema=SCHEMA+'-identity',sources=common.pins(SOURCES),
        actual_worker_entry=ENTRY.relative_to(ROOT).as_posix(),watchdog_entry=CLI.relative_to(ROOT).as_posix(),
        legacy_resource_sources=common.pins(resources.SOURCES),
        management_dispatch_before_completion=True,production_admission=False)

def prepare_spec(source,destination):
    """Create a new version; never modify a source specification or old identity."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    need(destination.is_relative_to(ROOT/'build/ownership-implementation-v1/handoff') and source!=destination,
         'Launcher spec needs a fresh handoff namespace')
    body=read(source);body['launcher_adapter']=identity()
    body['inputs']=dict(body['inputs'],**body['launcher_adapter']['sources'])
    resources._atomic(destination,body,True)
    validate_spec(destination)
    return body

def validate_spec(spec_path):
    spec=read(spec_path);expected=identity()
    need(spec.get('launcher_adapter')==expected,'Launcher adapter source/spec identity differs')
    need(all(spec['inputs'].get(name)==checksum for name,checksum in expected['sources'].items()),
         'Launcher source omitted from worker input pins')
    common.current(spec['inputs']);return expected

def verify_permit(lease_path):
    lease=read(lease_path);permit=read(lease['permit_file']);actual=read(lease['worker_identity_file'])
    adapter=validate_spec(lease['cache_path']);att=permit.get('launcher_adapter');launched=read(lease['launcher_file'])
    expected_wd=[str(CLI),'watchdog','--output',lease['worker_output'],'--spec',lease['cache_path'],
        '--ledger',lease['ledger_path'],'--lease',str(Path(lease_path).resolve())]
    need(isinstance(att,dict) and att.get('identity')==adapter and
         att.get('lease_sha256')==sha(lease_path) and att.get('spec_sha256')==sha(lease['cache_path']) and
         att.get('worker_output')==lease['worker_output'] and att.get('actual_worker_argv')==actual['argv'] and
         Path(actual['argv'][0]).resolve()==ENTRY and permit['worker']==actual['process'] and
         permit['token']==lease['token'] and permit['kill_on_job_close'] is True and
         att.get('actual_watchdog_argv')==expected_wd and att.get('actual_watchdog_sources')==adapter['sources'] and
         launched.get('launcher_adapter')==adapter and actual['argv']==launched['command'][1:] and
         permit['watchdog']==launched.get('watchdog_process'),
         'Actual launcher/worker permit attestation differs')
    return permit

def watchdog(lease_path):
    """Run unchanged containment; append verified adapter sources before permit."""
    lease=resources._lease(lease_path);old_atomic=resources._atomic
    def attested(path,body,immutable=False):
        if Path(path).resolve()==Path(lease['permit_file']).resolve():
            adapter=validate_spec(lease['cache_path']);record=read(lease['worker_identity_file'])
            launched=read(lease['launcher_file'])
            need(launched.get('launcher_adapter')==adapter and record['argv']==launched['command'][1:] and
                 Path(record['argv'][0]).resolve()==ENTRY and body['watchdog']==resources.process_identity()==launched.get('watchdog_process'),
                 'Actual v2 command/source handoff differs')
            body=dict(body,launcher_adapter=dict(identity=adapter,lease_sha256=sha(lease_path),
                spec_sha256=sha(lease['cache_path']),worker_output=lease['worker_output'],
                actual_worker_argv=record['argv'],actual_watchdog_argv=list(sys.argv),
                actual_watchdog_sources=common.pins(SOURCES)))
        return old_atomic(path,body,immutable)
    resources._atomic=attested
    try:return resources.watchdog(lease_path)
    finally:resources._atomic=old_atomic

def _settled_finish(ledger,lease):
    report=resources.status(ledger)
    need(report['pending'] is None,'Watchdog did not settle the exact resource lease')
    matches=[e for e in report['events'] if e['kind']=='finish' and e['token']==lease['token']]
    need(len(matches)==1,'Missing/duplicate exact resource finish')
    return report,matches[0]

def completed(output=worker.DEFAULT):
    """Pure completed reader: no reservation, process, gate, writer or engine."""
    output=Path(output).resolve();report=worker.completed(output)
    seal=read(output/'launcher-completed-v2.json')
    need(seal.get('schema')==SCHEMA+'-completed' and seal['launcher_identity']==identity() and
         seal['worker_catalog_sha256']==sha(output/'catalog.json') and
         seal['worker_identity_sha256']==sha(output/'identity.json'),
         'Completed launcher/source/catalog attestation differs')
    source=read(output/'identity.json');validate_spec(ROOT/source['spec_path'])
    lease=read(seal['lease']);need(lease['worker_output']==str(output),'Completed launcher output differs')
    verify_permit(seal['lease'])
    need(sha(seal['finish_event'])==seal['finish_event_sha256'],'Completed resource finish changed')
    finish=read(seal['finish_event'])
    need(finish['kind']=='finish' and finish['token']==lease['token'] and finish['clean'] is True and
         not finish['unknown_overrun'] and seal['worker_returncode']==seal['watchdog_returncode']==0,
         'Failed worker/watchdog cannot be completed')
    return dict(report,launcher=dict(schema=SCHEMA,verified=True,lease=seal['lease'],
        source_identity_sha256=config.digest(seal['launcher_identity']),production_admission=False))

def launch(worker_output,spec_path,fixture=True,ledger=resources.DEFAULT,pause_at=None,pause_marker=None,protocol=None):
    """Compatible supervisor API with actual output passed to both children."""
    output=Path(worker_output).resolve();spec_path=Path(spec_path).resolve();ledger=Path(ledger).resolve()
    need(fixture is True,'Production schedule admission remains blocked')
    if (output/'catalog.json').exists():return completed(output)
    adapter=validate_spec(spec_path);worker.prepare(output,spec_path)
    lease_path=resources.reserve(ledger,output,spec_path,fixture=True,protocol=protocol)
    lease=read(lease_path)
    kwargs=dict(cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    wd_command=[sys.executable,str(CLI),'watchdog','--output',str(output),'--spec',str(spec_path),
                '--ledger',str(ledger),'--lease',str(lease_path)]
    watchdog_process=subprocess.Popen(wd_command,**kwargs)
    command=[sys.executable,str(ENTRY),'worker','--output',str(output),'--spec',str(spec_path),
             '--ledger',str(ledger),'--lease',str(lease_path)]
    if pause_at is not None:
        need(pause_marker is not None,'Pause marker missing')
        command.extend(['--pause-at',pause_at,'--pause-marker',str(pause_marker)])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    process=subprocess.Popen(command,env=env,**kwargs)
    resources._atomic(lease['launcher_file'],dict(token=lease['token'],process=resources.process_identity(process.pid),
        supervisor=lease['supervisor_identity'],command=command,watchdog_command=wd_command,
        watchdog_process=resources.process_identity(watchdog_process.pid),launcher_adapter=adapter),True)
    process.wait();watchdog_process.wait(timeout=30)
    status,finish=_settled_finish(ledger,lease)
    need(process.returncode==0 and watchdog_process.returncode==0 and finish['clean'] is True and not finish['unknown_overrun'],
         'Worker/watchdog failed; retained lease charged, no completed shortcut allowed: '+str((process.returncode,watchdog_process.returncode,finish['reason'])))
    worker.completed(output)
    finish_file=ledger/'events'/f'{finish["index"]:06d}.json'
    seal=dict(schema=SCHEMA+'-completed',launcher_identity=adapter,lease=str(lease_path),
        worker_returncode=process.returncode,watchdog_returncode=watchdog_process.returncode,
        worker_catalog_sha256=sha(output/'catalog.json'),worker_identity_sha256=sha(output/'identity.json'),
        finish_event=str(finish_file),finish_event_sha256=sha(finish_file))
    resources._atomic(output/'launcher-completed-v2.json',seal,True)
    return dict(completed(output),resource=dict(charged_ns=status['charged_ns'],remaining_ns=status['remaining_ns']))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','worker','verify','resources','watchdog','prepare-spec'))
    parser.add_argument('--output',type=Path,default=worker.DEFAULT);parser.add_argument('--spec',type=Path,default=worker.SPEC)
    parser.add_argument('--ledger',type=Path,default=resources.DEFAULT);parser.add_argument('--lease',type=Path)
    parser.add_argument('--source-spec',type=Path);parser.add_argument('--pause-at');parser.add_argument('--pause-marker',type=Path)
    args=parser.parse_args(argv)
    try:
        # Management routes are unconditional and never consult default output.
        if args.operation=='resources':result=resources.status(args.ledger)
        elif args.operation=='watchdog':result=watchdog(args.lease)
        elif args.operation=='prepare-spec':result=prepare_spec(args.source_spec,args.spec)
        elif args.operation=='verify':result=completed(args.output)
        elif args.operation=='prepare':validate_spec(args.spec);result=worker.prepare(args.output,args.spec)
        elif args.operation=='worker':
            resources.bind_worker(args.lease);verify_permit(args.lease)
            result=worker.run(args.output,args.spec,args.lease,args.pause_at,args.pause_marker)
            resources.worker_complete(args.lease,dict(completed=True,engineering_games=result['report']['games']))
        else:result=launch(args.output,args.spec,ledger=args.ledger,pause_at=args.pause_at,pause_marker=args.pause_marker)
        print(config.canonical(result),flush=True);return 0
    except Exception as exc:
        print(config.canonical(dict(state='hold',error=type(exc).__name__+': '+str(exc))),flush=True);return 2
