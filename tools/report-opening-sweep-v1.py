"""Bounded, resumable n=1..64 majority campaign for the engine report."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from congklak.dakon_opening import Run, status

OUT=ROOT/'build/report-openings-v1'
RUNS=ROOT/'data/dakon-opening-v1/report-20261017'

def write(path,value):
    raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
    temporary=path.with_suffix('.tmp')
    temporary.write_text(raw,encoding='utf-8');temporary.replace(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    identity={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ('congklak/dakon_opening.py','congklak/dakon_opening_verify.py')}
    rows=[]
    start=time.monotonic()
    for n in range(1,65):
        path=RUNS/f'n{n:02d}-majority-A'
        t=time.monotonic()
        create=None if (path/'identity.json').exists() else dict(n=n,goal='majority',starter=0,cache_entries=10000)
        with Run(path,create=create) as run:
            value=run.search(max_actions=100000,max_seconds=3,checkpoint_actions=256,progress_seconds=1)
        if value['state']=='pending_verification':
            with Run(path) as run:value=run.search(max_actions=0,max_seconds=10)
        elapsed=time.monotonic()-t
        checked=value.get('verification')
        row=dict(n=n,total=2*n*n,required=n*n+1,state=value['state'],outcome=value['outcome'],
                 choices=checked['choices'] if checked else None,
                 starter_store=checked['starter_store'] if checked else None,
                 opponent_max=checked['opponent_max_total'] if checked else None,
                 simulated_actions=value['simulated_actions'],expanded_positions=value['expanded_positions'],
                 generation=value['generation'],stop_reason=value['stop_reason'],
                 invocation_seconds=round(elapsed,6),run_dir=path.relative_to(ROOT).as_posix(),
                 certificate_sha256=hashlib.sha256((path/'certificate.json').read_bytes()).hexdigest() if checked else None)
        rows.append(row)
        write(OUT/'sweep-progress.json',dict(state='running',completed=n,total=64,rows=rows,source_sha256=identity))
        print(json.dumps(row),flush=True)
    result=dict(schema='report-opening-sweep-v1',source_sha256=identity,python=sys.version,
                platform=platform.platform(),max_seconds_per_initial_invocation=3,
                max_actions_per_initial_invocation=100000,verification_resume_seconds=10,
                starter=0,goal='majority',cache_entries=10000,rows=rows,
                total_invocation_seconds=round(time.monotonic()-start,6),
                timing_scope='wall time on this host including checkpoints and in-search verification; not a cross-engine benchmark')
    write(OUT/'sweep.json',result)
    write(OUT/'sweep-progress.json',dict(state='complete',completed=64,total=64,
              verified=sum(r['outcome']=='guaranteed_win' for r in rows),rows=rows,source_sha256=identity))
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(event='sweep_complete',verified=sum(r['outcome']=='guaranteed_win' for r in rows),
                         no_witness=[r['n'] for r in rows if r['state']=='exhausted'],
                         incomplete=[r['n'] for r in rows if r['state'] not in ('found','exhausted')])),flush=True)

if __name__=='__main__':main()
