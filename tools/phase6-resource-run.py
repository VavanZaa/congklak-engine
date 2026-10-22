"""Observe the scoped process/allocator costs of frozen timing producers."""
from __future__ import annotations
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import phase6_common as common,config

def host_peak():
    if os.name!='nt':return None
    class Counters(C.Structure):
        _fields_=[('cb',C.c_ulong),('faults',C.c_ulong)]+[(k,C.c_size_t) for k in ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    kernel=C.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=C.c_void_p;psapi=C.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(Counters),C.c_ulong];value=Counters();value.cb=C.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),C.byref(value),value.cb):raise C.WinError(C.get_last_error())
    return dict(peak_working_set_bytes=value.peak,working_set_bytes=value.working,private_commit_bytes=value.pagefile,peak_private_commit_bytes=value.peak_pagefile)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('producer',choices=('speed','benchmark'));a=p.parse_args()
    manifest=json.loads((ROOT/'build/phase6-implementation-v2.json').read_text(encoding='utf-8'))
    common.require(manifest['sha256']==config.digest({k:v for k,v in manifest.items() if k!='sha256'}),'Corrupt implementation mapping')
    for group in ('sources','science_inputs'):
        for name,digest in manifest[group].items():
            actual=(ROOT/name).resolve();common.require(actual.is_relative_to(ROOT) and actual.is_file() and common.sha(actual)==digest,'Implementation input differs: '+name)
    source=ROOT/manifest['roles'][a.producer];output=ROOT/manifest['artifacts'][a.producer+'_output'];record=ROOT/f'build/phase6-{a.producer}-resources.json'
    models={'warm':common.sha(common.WARM)}
    if a.producer=='benchmark':models['updated']=common.sha(common.UPDATE)
    identity=dict(producer=a.producer,producer_sha256=common.sha(source),wrapper_sha256=common.sha(__file__),implementation_sha256=config.digest(manifest),model_file_sha256=models)
    if record.exists():
        saved=json.loads(record.read_text(encoding='utf-8'));common.require(saved['identity']==identity and saved['report_file_sha256']==common.sha(output),'Resource evidence differs')
        print(config.canonical(dict(event='phase6_resources_already_complete',producer=a.producer,new_semantic_test_run=False)),flush=True);return
    common.require(not output.exists(),'Completed producer has no original resource capture; a rerun cannot reconstruct its cost')
    prior_counts={};run=ROOT/manifest['artifacts'][a.producer+'_run']
    journals=sorted(run.glob('*/work.sqlite3')) if a.producer=='speed' else [run/'work.sqlite3']
    for journal in journals:
      if journal.exists():
        with sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True) as db:
            counts={}
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('units','cases','games'):
                if table in tables:counts[table]=db.execute('SELECT count(*) FROM '+table).fetchone()[0]
            prior_counts[str(journal.relative_to(ROOT)).replace('\\','/')]=counts
    started=time.monotonic();spec=importlib.util.spec_from_file_location('phase6_scoped_'+a.producer,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if a.producer=='speed':module.run(argparse.Namespace(model=str(common.WARM),backend='both',blocks=7,repetitions=128,output=str(output)))
    else:module.run()
    elapsed=time.monotonic()-started;torch=sys.modules.get('torch');gpu=None
    if torch is not None and torch.cuda.is_initialized():gpu=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    observed=dict(schema='phase6-scoped-producer-resources-v1',identity=identity,elapsed_seconds_including_module_load=elapsed,prior_committed_units=prior_counts,host=host_peak(),cuda_allocator=gpu,
                  report_file_sha256=common.sha(output),energy_measured=False,scope='One producer invocation; CUDA allocator excludes driver/context and other processes; wall clock includes module load, startup, remaining units, journal work and publication, excludes any prior interrupted invocation; no power claim')
    common.publish(record,observed);print(config.canonical(dict(event='phase6_scoped_resources_complete',producer=a.producer,elapsed_seconds=elapsed,host=observed['host'],cuda_allocator=gpu)),flush=True)

if __name__=='__main__':main()
