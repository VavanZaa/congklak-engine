"""Read-only Phase6 observer and scoped streamed-command logs; prior evidence intact."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from . import progress as old,config
from .native import ROOT

FOLDER=ROOT/'build/phase6-progress'

def read(path):return old.read_json(path)

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,value)

def tracked_runs():
    rows=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:12]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        rows.append(row)
    return rows

def training(folder,title,total):
    run=ROOT/'data'/folder;result=read(run/'result.json');step=0
    metrics=None
    if (run/'catalog.sqlite3').exists():
        with old.database(run/'catalog.sqlite3') as db:
            step=db.execute('SELECT COALESCE(MAX(step),0) FROM checkpoints').fetchone()[0]
            latest=db.execute('SELECT body,hash FROM checkpoints ORDER BY step DESC LIMIT 1').fetchone()
            if latest:
                value=json.loads(latest[0])
                if config.digest(value)!=latest[1]:raise ValueError('Corrupt checkpoint catalog metrics')
                metrics=value['metrics']
    complete=bool(result and result.get('status')=='complete' and step==total and (run/'model.json').exists())
    return old.card('phase6-'+folder,title,'complete' if complete else 'partial' if step else 'planned',
        f'{step}/{total} optimizer steps durably checkpointed.'+(f" Teacher policy CE {metrics['policy']['uniform_stratum_teacher_cross_entropy']:.4f}; H8 value MAE {metrics['finite_choice_horizon']['mae_seeds']:.3f} seeds." if metrics else '')+(f" Selected step {result['best_step']}." if complete else ''),str(run.relative_to(ROOT)),
        'Keep target sources separate; verify sampler, optimizer and all RNGs on recovery.',[old.count('committed optimizer steps',step,total)])

def snapshot():
    base=old.snapshot(ROOT);cards=[]
    plan=read(ROOT/'build/phase6-plan.json')
    cards.append(old.card('phase6-plan','Betawi rules and frozen pilot','complete' if plan else 'running',
        'Betawi id-betawi-relay7-v0; fixed-player0 final single-round seed margin. Existing accepted simulator retained.',
        'build/phase6-plan.json; build/phase6-protected-registry.json','Any local rule variation needs a separate versioned rule contract.'))
    cards.append(training('training-phase6-warm-v1','Policy head warm start',1000))
    cpu=read(ROOT/'build/phase6-puct-cpu-acceptance.json');gpu=read(ROOT/'build/phase6-puct-gpu-acceptance.json')
    cards.append(old.card('phase6-puct','Bounded PUCT and actor-aware backups','complete' if cpu and cpu.get('passed') and gpu and gpu.get('passed') else 'partial' if cpu else 'running',
        'CPU reference checks passed; GPU conformance pending.' if cpu and not gpu else 'Independent trees, legal priors, same-player bonus turns, bounded completed backups.',
        'build/phase6-puct-cpu-acceptance.json; build/phase6-puct-gpu-acceptance.json','Measure native PUCT in a new arena experiment; retain estimate/proof labels and bounded cancellation.'))
    speed=read(ROOT/'build/phase6-inference-speed-results.json')
    speed_message='Measure batch1/2/4/8/16/32, CPU features, transfers, GPU and whole-call latency; no batch-fill delay.'
    if speed and speed.get('complete'):
        single={k:v['batches']['1']['whole_call_ns']['median']/1e6 for k,v in speed['backends'].items()}
        speed_message=f"98/98 timing units. Single-position CPU {single['native']:.3f} ms; CUDA {single['cuda']:.3f} ms. CPU adapter faster at every measured batch size."
    cards.append(old.card('phase6-gpu','GPU batching and feature costs','complete' if speed and speed.get('complete') else 'planned',
        speed_message,
        'build/phase6-inference-speed-results.json','Measure reusable buffers and cheaper features before scaling; native telemetry charges two feature passes, CUDA one.'))
    run=ROOT/'data/phase6-selfplay-v1';manifest=read(run/'manifest.json');decisions=terminal=0
    if (run/'work.sqlite3').exists():
        with old.database(run/'work.sqlite3') as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='sp_games'").fetchone():
                terminal=db.execute("SELECT count(*) FROM sp_games WHERE json_extract(body,'$.status')='terminal'").fetchone()[0]
                decisions=db.execute('SELECT count(*) FROM sp_decisions').fetchone()[0]
    info=manifest['summary'] if manifest else {}
    cards.append(old.card('phase6-selfplay','Recoverable self-play and replay admission','complete' if manifest else 'partial' if decisions else 'planned',
        f'{terminal}/64 terminal games; {decisions} committed decisions. '+(f"{info['retained_rows']} retained rows; {info['excluded_families']} families excluded." if manifest else 'Incomplete games have no value labels.'),
        'data/phase6-selfplay-v1/manifest.json','Improve diversity: only12 families admitted; rarest stratum has2 rows. Preserve provenance and individual returns.',
        [old.count('terminal games',terminal,64),old.count('decision commits',decisions)]))
    cards.append(training('training-phase6-update-v1','Small self-play learning update',500))
    benchmark=read(ROOT/'build/phase6-benchmark.json');cases=games=0;bdb=ROOT/'data/phase6-benchmark-v1/work.sqlite3'
    if bdb.exists():
        with old.database(bdb) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'cases' in tables:cases=db.execute('SELECT count(*) FROM cases').fetchone()[0]
            if 'games' in tables:games=db.execute('SELECT count(*) FROM games').fetchone()[0]
    gate_message=''
    if benchmark:
        gate_message=' Both-baseline gates: '+', '.join(k+' '+('passed' if v else 'failed') for k,v in sorted(benchmark['both_baseline_primary_gates'].items()))+'.'
    cards.append(old.card('phase6-strength','Equal-time tests against both CPU baselines','complete' if benchmark and benchmark.get('cases')==80 and benchmark.get('games')==192 else 'partial' if cases or games else 'planned',
        f'{cases}/80 cases; {games}/192 paired games. H_v1 remains default.'+gate_message, 'build/phase6-benchmark.json',
        'Primary paired terminal seed margin with family uncertainty; report WDL, fallbacks and whole-decision overshoot separately.',
        [old.count('cases',cases,80),old.count('games',games,192)]))
    acceptance=read(ROOT/'build/phase6-acceptance.json');audit=read(ROOT/'build/phase6-audit/audit.json')
    stale=old.current_evidence(ROOT,acceptance) if acceptance and acceptance.get('passed') else []
    accepted=bool(acceptance and acceptance.get('passed') and not stale)
    cards.append(old.card('phase6-audit','Independent audit and interruption recovery','complete' if accepted else 'stale' if stale else 'partial' if audit else 'running',
        f"{acceptance.get('tests_run')} current acceptance checks passed." if accepted else 'Acceptance inputs changed.' if stale else 'Independent review active; actual training/game/arena process-kill tests required.',
        'build/phase6-audit/audit.json; build/phase6-acceptance.json','Verify all retained games, target provenance, completed rerun stability and all878 prior evidence pins.'))
    cards.append(old.card('phase6-readiness','Work remaining for the finalized analyzer','held',
        'Fresh release validation, cheaper evaluation, typed exact lookup integration, unified analyzer interface, packaging and rollback remain.',
        'docs/PHASE6_SCOPE_AND_ENGINE_READINESS.md','A successful pilot does not prove the initial board solved. Larger models/data follow measured benefit.'))
    for c in cards:c['needs_improvement']=c['state']!='complete'
    if speed and speed.get('complete'):cards[3]['needs_improvement']=True
    if benchmark and not all(benchmark['both_baseline_primary_gates'].values()):cards[6]['needs_improvement']=True
    invocations=tracked_runs();active=[r for r in invocations if r['state']=='running'];activity=read(FOLDER/'activity.json')
    return {**base,'schema':'congklak-phase6-progress-v1','tools':cards+base['tools'],'recent_runs':invocations,'active_tools':active,
            'current_work':dict(state='running',message=' · '.join(r['tool']+' — '+r['last_event'] for r in active)) if active else activity or dict(state='idle',message='Phase6 implementation and audit in progress; committed catalogs shown below.'),
            'scope':'Betawi confirmed. Committed work counters measure completion, not strength or solving. Prior Stage5 acceptance remains separately preserved.'}

def run_command(command,tool):
    FOLDER.mkdir(parents=True,exist_ok=True);uid=uuid.uuid4().hex;path=FOLDER/f'run-{uid}.json';log=FOLDER/f'run-{uid}.log';started=time.monotonic();child=None
    row=dict(tool=tool,command=command,pid=os.getpid(),process_identity=old.process_identity(os.getpid()),state='running',started_at=old.utc(),updated_at=old.utc(),last_event='Launching',log=str(log.relative_to(ROOT)))
    save(path,row)
    try:
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
                try:event=json.loads(line);message=old.describe(event)
                except (ValueError,KeyError,TypeError):message=line.strip()[:300]
                row.update(last_event=message,updated_at=old.utc());save(path,row)
            code=child.wait();row.update(state='complete' if code==0 else 'failed',exit_code=code,updated_at=old.utc(),elapsed_seconds=time.monotonic()-started);save(path,row);return code
    except BaseException:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        row.update(state='interrupted',updated_at=old.utc(),elapsed_seconds=time.monotonic()-started);save(path,row);raise

def serve(port=18766):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':data=(ROOT/'tools/phase6-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Phase6 observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
    sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot');activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='phase6');runner.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.operation=='serve':serve(a.port)
    elif a.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif a.operation=='activity':save(FOLDER/'activity.json',dict(state=a.state,message=a.message))
    else:return run_command(a.command[1:] if a.command[:1]==['--'] else a.command,a.tool)

if __name__=='__main__':raise SystemExit(main())
