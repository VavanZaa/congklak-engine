"""Capture original CPU-only producer invocation costs, including imports and admission."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as prior,puct_cpu_common as common

def host_peak():
    spec=importlib.util.spec_from_file_location('cpu_resource_os_dependency',ROOT/'tools/phase6-resource-run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.host_peak()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('producer',choices=('speed','benchmark'));args=parser.parse_args()
    started=time.perf_counter();_,manifest=common.admit();source=ROOT/manifest['roles'][args.producer];output=ROOT/manifest['artifacts'][args.producer+'_output'];run=ROOT/manifest['artifacts'][args.producer+'_run']
    record=ROOT/f'build/puct-cpu-{args.producer}-process-resources.json'
    identity=dict(producer=args.producer,producer_sha256=prior.sha(source),wrapper_sha256=prior.sha(__file__),implementation_sha256=manifest['sha256'])
    if record.exists():
        body=json.loads(record.read_text(encoding='utf-8'));prior.require(body['identity']==identity and body['report_file_sha256']==prior.sha(output),'Original resource observation differs')
        print(config.canonical(dict(event='puct_cpu_resources_reused',producer=args.producer,new_semantic_test_run=False)),flush=True);return
    prior.require(not output.exists(),'Producer already complete without original outer resource capture; cost cannot be reconstructed')
    previous={};journal=run/'work.sqlite3'
    if journal.exists():
        with sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            previous={name:db.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in ('units','cases','games') if name in tables}
    spec=importlib.util.spec_from_file_location('cpu_scoped_'+args.producer,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.run()
    prior.require('torch' not in sys.modules,'CPU producer unexpectedly imported Torch')
    body=dict(schema='puct-cpu-process-resources-v1',identity=identity,elapsed_seconds_including_module_load=time.perf_counter()-started,
              prior_committed_units=previous,host=host_peak(),torch_imported=False,cuda_used=False,energy_measured=False,report_file_sha256=prior.sha(output),
              scope='One original CPU producer invocation after argument parsing, including source guards/producer module load/admission/startup/remaining units/journals/publication; Python startup and previous invocations excluded. No power claim.')
    prior.publish(record,body);print(config.canonical(dict(event='puct_cpu_resources_complete',producer=args.producer,elapsed_seconds=body['elapsed_seconds_including_module_load'],host=body['host'])),flush=True)

if __name__=='__main__':main()
