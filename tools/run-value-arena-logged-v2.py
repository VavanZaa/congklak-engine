"""Logging-only supervisor for the unchanged frozen arena worker.

This adds explicit inherited log handles and compares every pre-interruption raw
commit. It changes no worker argv, models, settings, clock, evaluator or journal.
"""
from pathlib import Path
import json
import subprocess
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag_common as common,value_run_arena as arena,value_run_resources as resources,value_run_watchdog_v2 as adapter


def _unchanged(snapshot):
    for relative,expected in snapshot['raw_committed'].items():
        p=ROOT/relative
        current=dict(sha256=common.sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        if current!=expected:raise RuntimeError('Previously committed raw prefix changed: '+relative)


def main():
    prefix=common.read(ROOT/'build/value-run-v1/arena-pre-resume-prefix-v2.json');_unchanged(prefix)
    freeze=common.read(ROOT/'build/value-run-v1/arena-source-freeze.json');common.current(freeze['sources'])
    folder=ROOT/'build/value-run-v1/arena-logged-invocations'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    original=subprocess.Popen;old_size=resources.namespace_bytes;handles=[];calls=[]
    resources.namespace_bytes=adapter.namespace_bytes
    def logged(command,**kwargs):
        role='worker' if 'worker' in command else 'watchdog'
        if role=='watchdog':command=[command[0],str(ROOT/'tools/watch-value-arena-v2.py'),'--lease',command[command.index('--lease')+1]]
        log=folder/('%02d-%s.log'%(len(calls),role));stream=log.open('w',encoding='utf-8');handles.append(stream)
        kwargs.update(stdout=stream,stderr=subprocess.STDOUT)
        calls.append(dict(command=list(command),role=role,log=str(log),launcher_sources=freeze['sources'],enforcement_adapter=adapter.witness(),harness_source_sha256=common.sha(__file__)))
        common.publish(folder/('%02d-command.json'%(len(calls)-1)),calls[-1])
        return original(command,**kwargs)
    resources.subprocess.Popen=logged
    try:
        while not (arena.DEFAULT/'certificate.json').exists():
            result=resources.launch(arena.DEFAULT,arena.CANDIDATE)
            print(config.canonical(dict(event='unchanged_arena_resume_lease',**result)),flush=True)
            _unchanged(prefix)
            common.publish(folder/('settlement-%02d.json'%(len(calls)//2)),result)
            if result['worker_returncode']!=0 or result['watchdog_returncode']!=0:
                raise RuntimeError('Arena worker/watchdog failed; explicit child logs and unchanged raw prefix retained')
        result=arena.verify()
        _unchanged(prefix)
        common.publish(folder/'prefix-preserved.json',dict(schema='value-run-arena-production-prefix-preserved-v1',
            passed=True,committed_decisions_preserved=prefix['decisions'],committed_games_preserved=prefix['games'],
            verified_files=len(prefix['raw_committed']),sha_size_mtime_unchanged=True,
            original_head_sha256=prefix['head_sha256'],final_head_sha256=result['head_sha256'],
            harness_source_sha256=common.sha(__file__),frozen_sources=freeze['sources']))
        print(config.canonical(dict(completed=True,games=result['games'],decisions=result['decisions'],logs=str(folder))),flush=True)
        return 0
    except Exception as error:
        print(config.canonical(dict(state='hold',error=str(error),logs=str(folder))),flush=True);return 2
    finally:
        resources.subprocess.Popen=original
        resources.namespace_bytes=old_size
        for handle in handles:handle.close()


if __name__=='__main__':raise SystemExit(main())
