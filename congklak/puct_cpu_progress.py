"""CPU PUCT workshop observer; previously accepted progress sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,phase6_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/puct-cpu-progress'
def read(path):return old.read_json(path)
def save(path,body):path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,body)

def runs():
    values=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:15]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        values.append(row)
    return values

def snapshot():
    base=previous.snapshot();cards=[];plan=read(ROOT/'build/puct-cpu-plan.json');implementation=read(ROOT/'build/puct-cpu-implementation.json')
    cards.append(old.card('puct-cpu-plan','Frozen CPU comparison and ownership','complete' if implementation else 'partial' if plan else 'planned',
        'Both fixed models × legacy / single pass / bounded cache; 80 cases and 576 paired games. No training; reused development families.',
        'build/puct-cpu-plan.json; build/puct-cpu-implementation.json','Keep all variants visible; exploratory intervals do not establish a release result.'))
    inference=read(ROOT/'build/puct-cpu-inference-acceptance.json');checks=inference.get('tests',inference.get('tests_run',0)) if inference else 0
    inference_ok=bool(inference and inference.get('passed') and not old.current_evidence(ROOT,inference))
    cards.append(old.card('puct-cpu-inference','Single-pass native inference and bounded LRU','complete' if inference_ok else 'running',
        f'{checks} current author checks passed.' if inference_ok else 'Building structured native ABI and exact double predictions; current source checks pending.',
        'build/puct-cpu-inference-acceptance.json','Count physical forwards separately from logical requests; empty cache every decision; verify cancellation and full-state keys.'))
    speed=read(ROOT/'build/puct-cpu-speed.json');units=0;database=ROOT/'data/puct-cpu-speed-v1/work.sqlite3'
    if database.exists():
        with old.database(database) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='units'").fetchone():units=db.execute('SELECT count(*) FROM units').fetchone()[0]
    message=f'{units}/1410 committed units: empty-cache calls, retrieval ceiling and real 64-backup trees.'
    if speed:
        a=speed['profiles']['warm_legacy']['micro']['cold']['1']['whole_call_ns']['median']/1000
        b=speed['profiles']['warm_single']['micro']['cold']['1']['whole_call_ns']['median']/1000
        message+=f' Warm cold-cache CPU call: legacy {a:.1f} µs; single pass {b:.1f} µs.'
    cards.append(old.card('puct-cpu-speed','Evaluation costs and fixed-tree conformance','complete' if speed and speed.get('complete') else 'partial' if units else 'planned',message,
        'build/puct-cpu-speed.json','Hot-cache speed is a retrieval ceiling; use actual tree hit rates and whole-decision clocks for practical gains.',[old.count('cost units',units,1410)]))
    benchmark=read(ROOT/'build/puct-cpu-benchmark.json');cases=games=0;database=ROOT/'data/puct-cpu-benchmark-v1/work.sqlite3'
    if database.exists():
        with old.database(database) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'cases' in tables:cases=db.execute('SELECT count(*) FROM cases').fetchone()[0]
            if 'games' in tables:games=db.execute('SELECT count(*) FROM games').fetchone()[0]
    message=f'{cases}/80 cases and {games}/576 games committed. H_v1 stays default.'
    if benchmark:
        gates=benchmark.get('both_baseline_primary_gates',{})
        message+=' Both-baseline gates: '+', '.join(name+' '+('passed' if passed else 'failed') for name,passed in gates.items())+'.'
    cards.append(old.card('puct-cpu-strength','Equal-time CPU PUCT strength','complete' if benchmark and cases==80 and games==576 else 'partial' if cases or games else 'planned',message,
        'build/puct-cpu-benchmark.json','Report all six candidates, paired seed margins, fallbacks and deadline overshoot; no automatic promotion.',[old.count('cases',cases,80),old.count('games',games,576)]))
    acceptance=read(ROOT/'build/puct-cpu-acceptance.json');stale=old.current_evidence(ROOT,acceptance) if acceptance and acceptance.get('passed') else []
    accepted=bool(acceptance and acceptance.get('passed') and not stale)
    cards.append(old.card('puct-cpu-audit','Independent audit and interruption recovery','complete' if accepted else 'stale' if stale else 'running',
        f"{acceptance['tests_run']} current acceptance checks passed; prior 1,192 inputs preserved." if accepted else 'Independent feature/tree/counter tests and real analysis/case/arena process kills.',
        'build/puct-cpu-audit/audit.json; build/puct-cpu-acceptance.json','Retain raw recovery states; validate journals and completed reruns, not just file checksums.'))
    cards.append(old.card('puct-cpu-readiness','Next decisions before more training','held','Measure native PUCT gains first. Replay diversity, fresh release validation, exact lookup integration and packaging remain.',
        'docs/PUCT_CPU_RESULTS.md','Larger models and datasets follow measured quality and cost; a development win does not solve the initial board.'))
    for c in cards:c['needs_improvement']=c['state']!='complete'
    if benchmark and not all(benchmark.get('both_baseline_primary_gates',{}).values()):cards[3]['needs_improvement']=True
    recent=runs();active=[r for r in recent if r['state']=='running'];activity=read(FOLDER/'activity.json')
    return {**base,'schema':'puct-cpu-progress-v1','tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,
            'current_work':dict(state='running',message=' · '.join(r['tool']+' — '+r['last_event'] for r in active)) if active else activity or dict(state='running',message='Preparing native CPU comparisons; committed catalogs appear below.'),
            'scope':'Betawi CPU PUCT efficiency experiment; models and training frozen. Prior Phase 6 and Stage 5 evidence preserved.'}

def run_command(command,tool):
    FOLDER.mkdir(parents=True,exist_ok=True);uid=uuid.uuid4().hex;path=FOLDER/f'run-{uid}.json';log=FOLDER/f'run-{uid}.log';started=time.monotonic();child=None
    import os
    row=dict(tool=tool,command=command,pid=os.getpid(),process_identity=list(old.process_identity(os.getpid())),state='running',started_at=old.utc(),updated_at=old.utc(),last_event='Launching',log=str(log.relative_to(ROOT)))
    save(path,row)
    try:
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
                try:message=old.describe(json.loads(line))
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
                if self.path=='/':data=(ROOT/'tools/puct-cpu-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'CPU PUCT observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='cpu-puct');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':main()
