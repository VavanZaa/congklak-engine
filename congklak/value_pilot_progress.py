"""Read-only fixed-policy value pilot observer; all prior accepted sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,horizon_finish_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/value-pilot-progress'
def read(path):return old.read_json(path)
def save(path,body):path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,body)

def runs():
    values=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:15]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        values.append(row)
    return values

_PRIOR_CACHE=None
_PRIOR_AT=0.


def previous_snapshot():
    global _PRIOR_CACHE,_PRIOR_AT
    now=time.monotonic()
    if _PRIOR_CACHE is None or now-_PRIOR_AT>=15:
        _PRIOR_CACHE=previous.snapshot();_PRIOR_AT=now
    return _PRIOR_CACHE



def committed(stage):
    path=ROOT/('data/value-pilot-benchmark-'+stage+'-v1/work.sqlite3');out=dict(cases=0,games=0)
    if path.exists():
        with old.database(path) as db:
            tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('cases','games'):
                if table in tables:out[table]=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
    return out


def snapshot():
    base=previous_snapshot();cards=[];plan=read(ROOT/'build/value-pilot-benchmark-plan.json');implementation=read(ROOT/'build/value-pilot-benchmark-implementation.json');data=read(ROOT/'build/value-pilot-data.json')
    cards.append(old.card('value-pilot-plan','Frozen sampling objective and small pilot','complete' if plan and implementation and data else 'running',
        'Fixed source policy/trunk; only65 value-head parameters may train. Original data remain intact. The pilot uses reviewed weights and newly protected cohorts.',
        'build/value-pilot-data.json; build/value-pilot-training-plan.json; build/value-pilot-benchmark-plan.json',
        'Report exact target objective, draw probabilities and loss weights. Distribution effective support measures concentration, not independent games or gradient ESS. No automatic promotion.'))
    pre=committed('pretraining');pre_result=read(ROOT/'build/value-pilot-benchmark-pretraining.json')
    cards.append(old.card('value-pilot-pretraining','Pretraining control versus both baselines','complete' if pre_result and pre['cases']==44 and pre['games']==48 else 'partial' if pre['cases'] or pre['games'] else 'planned',
        f"{pre['cases']}/44 diagnostics and {pre['games']}/48 games committed;264 scalar measurements. The unchanged control must finish before the first optimizer step.",
        'build/value-pilot-benchmark-pretraining.json; data/value-pilot-benchmark-pretraining-v1/work.sqlite3',
        'Twelve reused development families, paired seats and25ms whole decisions. Keep this control as a descriptive timing reference, not a fresh strength test.',[old.count('diagnostic cases',pre['cases'],44),old.count('pretraining control games',pre['games'],48)]))
    steps=0;catalog=ROOT/'data/value-pilot-v1/catalog.sqlite3'
    if catalog.exists():
        with old.database(catalog) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='checkpoints'").fetchone():steps=db.execute('SELECT coalesce(max(step),0) FROM checkpoints').fetchone()[0]
    trained=read(ROOT/'data/value-pilot-v1/result.json')
    cards.append(old.card('value-pilot-training','Recoverable fixed-final value-head training','complete' if trained else 'partial' if steps else 'planned',
        f'{steps}/200 committed steps; batch256, checkpoint every20. Fixed final checkpoint and deterministic CPU training; no checkpoint selection from benchmark outcomes.',
        'data/value-pilot-v1/catalog.sqlite3; data/value-pilot-v1/result.json; build/value-pilot-model-witness.json',
        'Verify exact optimizer, draw and RNG recovery. A new value head can change PUCT choices through backups despite identical raw policy.',[old.count('committed optimizer steps',steps,200)]))
    proof=read(ROOT/'build/value-pilot-model-witness.json')
    cards.append(old.card('value-pilot-policy','Frozen global policy and search','complete' if proof and proof.get('passed') else 'running',
        'Six trunk/policy tensors stay bit-identical to the retained control;65 head parameters may train. Same relay encoder, uncached CPU inference, c_puct0.25 and search limits.',
        'build/value-pilot-model-witness.json; build/value-pilot-training-implementation.json',
        'Validate every checkpoint and export. Raw-policy invariance does not imply identical searched actions or improved value estimates.'))
    post=committed('posttraining');result=read(ROOT/'build/value-pilot-benchmark-posttraining.json');text=f"{post['cases']}/44 diagnostics and {post['games']}/96 games committed;352 scalar measurements. Contemporaneous control and candidate face H_v1 and H8 relay on matched paired families."
    if result:text+=' Both-baseline exploratory gates: '+json.dumps(result.get('both_baseline_primary_gates'),sort_keys=True)+'. Matched candidate-minus-control differences: '+json.dumps(result.get('secondary_factor_contrasts'),sort_keys=True)+'.'
    cards.append(old.card('value-pilot-strength','Equal-time contemporaneous value comparison','complete' if result and post['cases']==44 and post['games']==96 else 'partial' if post['cases'] or post['games'] else 'planned',text,
        'build/value-pilot-benchmark-posttraining.json; data/value-pilot-benchmark-posttraining-v1/work.sqlite3',
        'Exploratory12-family intervals only; preserve both baseline results, unfinished games, fallbacks, overshoots and physical feature/forward counts.',[old.count('diagnostic cases',post['cases'],44),old.count('posttraining games',post['games'],96)]))
    acceptance=read(ROOT/'build/value-pilot-acceptance.json');stale=old.current_evidence(ROOT,acceptance) if acceptance and acceptance.get('passed') else [];okay=bool(acceptance and acceptance.get('passed') and not stale)
    cards.append(old.card('value-pilot-audit','Independent audit, recovery and release decision','complete' if okay else 'stale' if stale else 'running',
        str(acceptance.get('tests_run'))+' unique checks passed;2426 prior pins preserved.' if okay else 'Auditing draw/weight objective, frozen policy, real training and game pre/postcommit kills, all retained traces and completed no-work reruns.',
        'build/value-pilot-audit.json; build/value-pilot-acceptance.json; docs/VALUE_PILOT_RESULTS.md',
        'H_v1 remains default. No silent rollout, new sealed test use or automatic promotion follows from pilot completion.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if result and not all(result.get('both_baseline_primary_gates',{}).values()):cards[4]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Building the fixed-policy small value pilot and frozen pretraining control.')
    return {**base,'schema':'value-pilot-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Betawi small value-head pilot. Reused development:48 pretraining+96 posttraining games,616 scalar measurements. Fixed policy/search; no sealed test or automatic default promotion.'}


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
                if self.path=='/':data=(ROOT/'tools/value-pilot-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Value pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='value-pilot');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
