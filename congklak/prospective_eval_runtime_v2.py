"""Reviewed new entry routing over the unchanged cumulative ledger/watchdog.

The frozen base module is never patched. Only reservation and launch select the
explicit v2 reader/worker; all charges use the existing identity and event chain.
"""
from contextlib import contextmanager
from pathlib import Path
import os,subprocess,sys,time,uuid
from . import prospective_eval_common as stage, prospective_eval_runtime as base
from . import config
ROOT=base.ROOT
DEFAULT=base.DEFAULT
SCHEMA=base.SCHEMA
PROTOCOL=base.PROTOCOL
AMENDMENT=stage.FOLDER/'amendments/v2/amendment.json'
REVIEW=AMENDMENT.parent/'preflight.json'
PROJECTION=AMENDMENT.parent/'source-projection.json'
WORKER=ROOT/'congklak/prospective_eval_worker_v2.py'
SUPERVISOR=ROOT/'congklak/prospective_eval_cli_v2.py'
need,read,sha=stage.need,stage.read,stage.sha
Writer=base.Writer
status=base.status
initialize=base.initialize
preflight=base.preflight
process_identity=base.process_identity
boot_identity=base.boot_identity
operational_binding=base.operational_binding
verify_operational_binding=base.verify_operational_binding
_path=base._path
_atomic=base._atomic
_append=base._append
_open=base._open
REQUIRED=('congklak/prospective_eval_execute_v2.py','congklak/prospective_eval_runtime_v2.py',
 'congklak/prospective_eval_worker_v2.py','congklak/prospective_eval_cli_v2.py','congklak/prospective_eval_recovery_v2.py',
 'build/prospective-eval-v1/amendments/v2/project_v2.py','build/prospective-eval-v1/amendments/v2/source-projection.json',
 'tests/test_prospective_eval_v2.py')

def admit():
    stage.admit_plan();body=read(AMENDMENT);review=read(REVIEW)
    need(body['schema']=='prospective-eval-v1-reader-v2-amendment' and body['base_plan_ref']==stage.ref(stage.PLAN)
         and body['base_review_ref']==stage.ref(stage.PREFLIGHT),'Reader amendment base scope changed')
    need(body['budget_reset'] is False and body['ledger_identity_changed'] is False and body['engine_policy_changed'] is False
         and body['guard_seal_requirement_relaxed'] is False,'Reader amendment changed execution semantics')
    need(set(REQUIRED)<=set(body['source_inputs']),'Required v2 physical source omitted')
    stage.common.current(body['source_inputs'])
    need(review['schema']=='prospective-eval-v1-reader-v2-independent-preflight' and review['passed'] is True
         and review['independent_of_implementation'] is True and review['amendment_ref']==stage.ref(AMENDMENT)
         and review['source_inputs']==body['source_inputs'],'Independent v2 entry/reader review missing or stale')
    return body

def execution_binding():
    body=admit()
    return dict(schema='prospective-eval-v1-reader-v2-binding',amendment_ref=stage.ref(AMENDMENT),review_ref=stage.ref(REVIEW),
                source_inputs=body['source_inputs'],base_plan_ref=body['base_plan_ref'],same_ledger=str(DEFAULT))

def verify_execution_identity(identity,path):
    if 'execution_amendment' in identity:
        need(identity['execution_amendment']==execution_binding(),'New query identity lost its reviewed v2 source binding')
    else:
        path=Path(path).resolve()
        need(path==stage.FOLDER/'jobs/endgame' and identity['operation']=='endgame' and identity['fixture'] is False
             and (path/'catalog.json').is_file() and read(path/'report.json')['state']=='complete',
             'Only the sealed legacy endgame may use a read-only original identity')
    return True

@contextmanager
def execution_views():
    binding=execution_binding()
    try:yield binding
    finally:need(execution_binding()==binding,'V2 physical source closure changed before completion')

def _lease(path):
    lease=base._lease(path)
    need(lease['worker_entry']==str(WORKER) and lease['worker_entry_sha256']==sha(WORKER)
         and lease.get('operational_amendment')==execution_binding(),'Actual lease is not the reviewed v2 entry/source set')
    return lease

def bind_worker(path):
    _lease(path);answer=base.bind_worker(path);_lease(path);return answer

def assert_worker_lease(path):
    _lease(path);return base.assert_worker_lease(path)

def remaining_ns(path):
    _lease(path);return base.remaining_ns(path)

def worker_complete(path,result):
    binding=execution_binding();_lease(path)
    need(result['source_views_closed_before_completion'] is True,'V2 source view did not close')
    base.worker_complete(path,dict(result,execution_amendment=binding))

def watchdog(path):
    _lease(path)
    return base.watchdog(path)

def reserve(path,worker_output,cache_path,fixture=False,protocol=None,entry=None,extra_args=()):
    from . import prospective_eval_execute_v2 as contract
    binding=execution_binding();path=_path(path)
    need(not(path/'hold.json').exists(),'Cumulative resource accounting is held')
    entry=Path(entry or WORKER).resolve();need(entry==WORKER,'Only the exact reviewed v2 native worker is registered')
    need(type(fixture) is bool and len(extra_args)%2==0 and all(type(v) is str for v in extra_args),'Malformed v2 worker arguments')
    need(len(set(extra_args[::2]))==len(extra_args[::2]) and set(extra_args[::2])<={'--operation','--pause-at','--pause-marker'}
         and '--operation' in extra_args,'Unregistered v2 worker options')
    options=dict(zip(extra_args[::2],extra_args[1::2]));contract.classify(worker_output,options['--operation'],fixture)
    need(options['--operation']=='arena','Completed endgame queries cannot be repeated')
    if '--pause-at' in options:need(fixture and options['--pause-at'] in contract.HANDOFF_KILL_HOOKS and '--pause-marker' in options,'Unregistered pause')
    stage.path_of(worker_output);need(Path(cache_path).resolve()==(stage.DATA/'cache').resolve(),'Cache marker changed')
    initialize(path,False,protocol)
    with Writer(path):
        report=status(path);need(report['pending'] is None,'Another cumulative worker lease remains active')
        p=report['identity']['protocol'];need(report['remaining_ns']>30_000_000_000,'Cumulative budget exhausted')
        pre=preflight(p);duration=min(p['lease_ns'],report['remaining_ns']);token=uuid.uuid4().hex
        folder=path/'leases'/token;folder.mkdir(parents=True,exist_ok=False)
        lease=dict(schema=SCHEMA+'-lease',token=token,ledger_path=str(path),worker_output=str(Path(worker_output).resolve()),
          cache_path=str(Path(cache_path).resolve()),worker_entry=str(entry),worker_entry_sha256=sha(entry),extra_args=list(extra_args),
          plan_sha256=sha(stage.PLAN),deadline_unix_ns=time.time_ns()+duration,deadline_monotonic_ns=time.monotonic_ns()+duration,
          reserved_ns=duration,fixture=fixture,worker_identity_file=str(folder/'worker.json'),permit_file=str(folder/'permit.json'),
          launcher_file=str(folder/'launcher.json'),completion_file=str(folder/'completion.json'),supervisor_identity=process_identity(),
          protocol=p,resource_identity_sha256=sha(path/'identity.json'),preflight=pre,lease_file=str(folder/'lease.json'),
          boot_filetime=boot_identity(),operational_v4=operational_binding(),operational_amendment=binding)
        _atomic(folder/'lease.json',lease,True);_append(path,report,dict(kind='reserve',lease=lease))
    return Path(lease['lease_file'])

def launch(worker_output,cache_path=None,fixture=False,stop_after=None,ledger=DEFAULT,python=None,protocol=None,
           pause_at=None,pause_marker=None,entry=None,extra_args=()):
    from . import prospective_eval_execute_v2 as execute, prospective_eval_reservations as reservations
    admit();need(stop_after is None and python is None and protocol is None,'Runtime recipe cannot be overridden')
    need(list(extra_args[::2])==['--operation'],'One native operation required');operation=extra_args[1]
    cache_path=Path(cache_path or stage.DATA/'cache').resolve()
    if execute.has_completion(worker_output,operation):
        done=execute.completed_job(worker_output,operation)
        need(done['publication_gap'] is False,'V2 cannot rewrite a sealed predecessor publication')
        return done
    reservations.verify();execute.preflight_job(worker_output,operation,cache_path,fixture=fixture)
    args=list(extra_args)
    if pause_at is not None:
        need(fixture and pause_marker is not None,'Production pause forbidden')
        args.extend(['--pause-at',str(pause_at),'--pause-marker',str(pause_marker)])
    lease_path=reserve(ledger,worker_output,cache_path,fixture,entry=entry,extra_args=args);lease=read(lease_path)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    kwargs=dict(cwd=ROOT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    watcher=subprocess.Popen([sys.executable,str(SUPERVISOR),'watchdog','--lease',str(lease_path)],**kwargs)
    command=[sys.executable,str(WORKER),'run','--output',str(Path(worker_output).resolve()),'--cache',str(cache_path),'--lease',str(lease_path)]
    if fixture:command.append('--engineering-fixture')
    command.extend(lease['extra_args'])
    try:
        process=subprocess.Popen(command,**kwargs)
        _atomic(lease['launcher_file'],dict(token=lease['token'],process=process_identity(process.pid),supervisor=lease['supervisor_identity'],command=command),True)
    except Exception:
        watcher.wait(timeout=40);raise
    process.wait();watcher.wait(timeout=40);settled=status()
    need(settled['pending'] is None and process.returncode==0 and watcher.returncode==0,'Contained v2 worker failed; retain evidence')
    need(settled['events'][-1]['clean'] is True,'V2 worker unclean; no automatic query retry')
    done=read(lease['completion_file'])['result']
    need(done['execution_amendment']==execution_binding() and done['state'] in ('complete','checkpoint','held'),'V2 completion source/state differs')
    return dict(worker_returncode=process.returncode,watchdog_returncode=watcher.returncode,lease=str(lease_path),
                charged_ns=settled['charged_ns'],remaining_ns=settled['remaining_ns'],state=done['state'])
