"""Value-only refit observer; all prior accepted sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,quality_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/value-progress'
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
    base=previous.snapshot();cards=[];plan=read(ROOT/'build/value-plan.json');impl=read(ROOT/'build/value-implementation.json');data=read(ROOT/'build/value-data.json')
    admitted=data['train_rows'] if data else 0;excluded=len(data['excluded_rows']) if data else 0
    cards.append(old.card('value-plan','Protected data and frozen value experiment','complete' if plan and impl else 'running',
        f'{admitted}/8000 rows admitted; {excluded} ownership exclusions. Two profiles share improved policy/search; fresh case and arena families remain development.',
        'build/value-plan.json; build/value-registry.json; build/value-data.json','Exclusions restrict new draws; preserve ancestral exposure and original data.'))
    steps=0;catalog=ROOT/'data/value-train-v1/catalog.sqlite3'
    if catalog.exists():
        with old.database(catalog) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='checkpoints'").fetchone():steps=db.execute('SELECT coalesce(max(step),0) FROM checkpoints').fetchone()[0]
    trained=read(ROOT/'data/value-train-v1/result.json')
    cards.append(old.card('value-training','Value-head-only seed-space training','complete' if trained else 'partial' if steps else 'planned',
        f'{steps}/1500 committed steps. Train65 final-value parameters; seed-space Huber(delta4), fixed final checkpoint, no policy loss.',
        'data/value-train-v1/result.json; build/value-model-witness.json','Frozen representation can limit value fit; measure teacher error and arena strength separately.',[old.count('optimizer steps',steps,1500)]))
    proof=read(ROOT/'build/value-model-witness.json')
    cards.append(old.card('value-policy','Improved policy and search held fixed','complete' if proof and proof.get('passed') else 'running',
        'Six trunk/policy tensors (12807 parameters) stay bit-identical; unchanged relay encoder, native forward and c_puct=0.25 search.',
        'build/value-model-witness.json; build/value-audit/audit.json','Verify every checkpoint and export; sampled policy parity supports the frozen-weight proof.'))
    cases=games=0;dbpath=ROOT/'data/value-benchmark-v1/work.sqlite3'
    if dbpath.exists():
        with old.database(dbpath) as db:
            tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'cases' in tables:cases=db.execute('SELECT count(*) FROM cases').fetchone()[0]
            if 'games' in tables:games=db.execute('SELECT count(*) FROM games').fetchone()[0]
    result=read(ROOT/'build/value-benchmark.json');message=f'{cases}/80 cases and {games}/192 games committed;640 scalar measurements. Both candidates face H_v1 and H8 on matched families.'
    if result:message+=' Both-baseline gates: '+', '.join(k+' '+('passed' if v else 'failed') for k,v in result['both_baseline_primary_gates'].items())+'.'
    cards.append(old.card('value-strength','Equal-time fixed-policy value comparison','complete' if result and cases==80 and games==192 else 'partial' if cases or games else 'planned',message,
        'build/value-benchmark.json','Report both profiles, matched value-only differences, all margins/intervals and overshoots.',[old.count('cases',cases,80),old.count('games',games,192)]))
    acceptance=read(ROOT/'build/value-acceptance.json');stale=old.current_evidence(ROOT,acceptance) if acceptance and acceptance.get('passed') else [];accepted=bool(acceptance and acceptance.get('passed') and not stale)
    cards.append(old.card('value-audit','Independent audit and real kill/recovery','complete' if accepted else 'stale' if stale else 'running',
        f"{acceptance['tests_run']} current checks passed; all1647 prior inputs preserved." if accepted else 'Auditing frozen policy, targets, actual CUDA checkpoint kills and analyzer/case/arena recovery.',
        'build/value-audit/audit.json; build/value-acceptance.json','Require exact resumed optimizer/RNG equality; preserve raw killed states and prior evidence.'))
    analysis=ROOT/'data/value-analysis-demo/analysis-128.json'
    cards.append(old.card('value-analysis','Recoverable analysis and release work','complete' if analysis.exists() else 'planned',
        'Keep the latest completed recommendation durable while unfinished trees recompute; margins and visits remain estimates.',
        'data/value-analysis-demo/analysis-128.json; docs/VALUE_RESULTS.md','Release still needs strength evidence, fresh validation, typed exact lookup and packaging.'))
    for c in cards:c['needs_improvement']=c['state']!='complete'
    if result and not all(result['both_baseline_primary_gates'].values()):cards[3]['needs_improvement']=True
    recent=runs();active=[r for r in recent if r['state']=='running'];activity=read(FOLDER/'activity.json')
    return {**base,'schema':'value-progress-v1','tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,
            'current_work':dict(state='running',message=' · '.join(r['tool']+' — '+r['last_event'] for r in active)) if active else activity or dict(state='running',message='Building fixed-policy value-only experiment.'),
            'scope':'Betawi fixed-policy value-head experiment; prior quality/CPU/Phase6 evidence preserved; H_v1 remains default.'}

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
                if self.path=='/':data=(ROOT/'tools/value-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Value observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='cpu-puct');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
