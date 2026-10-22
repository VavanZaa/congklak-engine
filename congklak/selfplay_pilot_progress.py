"""Committed self-play, joint-training and equal-time benchmark progress observer."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,midgame_pilot_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/selfplay-pilot-progress'
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



def committed(path,tables):
    """Read only committed rows, without importing producer/native evaluators."""
    import sqlite3
    if not path.exists():return {}
    try:
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2) as db:
            db.execute('BEGIN');present={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            return {name:{key:json.loads(body) for key,body in db.execute('SELECT '+('key' if name=='meta' else 'state')+',body FROM '+name+' ORDER BY '+('key' if name=='meta' else 'state'))} for name in tables if name in present}
    except (sqlite3.Error,ValueError):return {}


def snapshot():
    base=previous_snapshot();cards=[]
    projection=read(ROOT/'build/selfplay-pilot-benchmark-projection.json') or {};sp=read(ROOT/'build/selfplay-training-pilot-selfplay.json') or {};trained=read(ROOT/'data/selfplay-training-pilot-v1/result.json') or {}
    result=read(ROOT/'build/selfplay-pilot-benchmark.json') or {};modelproof=read(ROOT/'build/selfplay-training-pilot-model-witness.json') or {}
    def counts(path,tables):
        import sqlite3
        if not path.exists():return {}
        try:
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2) as db:
                db.execute('BEGIN');present={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                return {name:db.execute('SELECT count(*) FROM '+name).fetchone()[0] for name in tables if name in present}
        except sqlite3.Error:return {}
    sc=counts(ROOT/'data/selfplay-training-pilot-v1/work.sqlite3',('sp_games','sp_decisions','sp_rows'))
    bc=counts(ROOT/'data/selfplay-pilot-benchmark-v1/work.sqlite3',('cases','games','decisions'))
    cards.append(old.card('selfplay-pilot-reservation','Frozen control and protected evaluation reservations','complete' if projection else 'running',
        'Control bc1160e...; joint challenger has the same127x64 architecture. '+str(len(projection.get('arena_starts',[])))+'/12 arena starts reserved;32 reused known-exact diagnostics. Protect latest144 evaluation trajectories,10 new validation roots and9 diagnostics before self-play.',
        'build/selfplay-pilot-baseline.json; build/selfplay-pilot-benchmark-projection.json; docs/SELFPLAY_PILOT_SCOPE.md',
        'New-to-prior-arena family IDs are still consumed development. Future trajectory collisions are checked after evaluation and flagged; no sealed release or energy claim.'))
    text=str(sc.get('sp_games',0))+'/32 maximum self-play trajectories committed; '+str(sc.get('sp_decisions',0))+' durable decisions; '+str(sc.get('sp_rows',0))+' replay rows committed.'
    if sp:text+=' Terminal '+str(sp.get('terminal_games','see result'))+'; capped '+str(sp.get('capped_games','see result'))+'; admitted families '+str(sp.get('admitted_families','see result'))+'; retained rows '+str(sp.get('retained_rows','see result'))+'; training admission '+str(sp.get('training_admission','see result'))+'.'
    cards.append(old.card('selfplay-pilot-generation','Fixed-control CPU PUCT self-play','complete' if sp else 'running' if sc.get('sp_decisions',0) else 'held',text,
        'data/selfplay-training-pilot-v1/work.sqlite3; build/selfplay-training-pilot-selfplay.json',
        '16 fixed training starts,2 trajectories each,64 simulations, c_puct0.25, no model update during generation. Capped games do not supply terminal-return targets; no filler expansion.'))
    cards.append(old.card('selfplay-pilot-replay','Terminal returns, visits and role-aware replay','complete' if sp else 'running' if sc.get('sp_rows',0) else 'held',
        'Self-play admission is closed; inspect admitted families/terminal games/retained rows in the result. At least8 admitted families are required before training.' if sp else 'Optimizer candidates require at least25 active seeds; all later states are omitted before the unchanged evaluation mask and whole-family collision ban. Actor-correct visits and fixed-player0 terminal margins; at most4 observations per raw state.',
        'build/selfplay-training-pilot-selfplay.json; data/selfplay-training-pilot-v1/selfplay-manifest.json',
        'Joint losses use only admitted self-play roots with at least25 active seeds. No teacher rehearsal. Root/decision counts are not independent games or gradient ESS.'))
    cards.append(old.card('selfplay-pilot-training','Joint trunk, policy and value CPU training','complete' if trained else 'running' if (ROOT/'data/selfplay-training-pilot-v1/catalog.sqlite3').exists() else 'held',
        'Fixed-final training result is available; no benchmark-driven checkpoint selection.' if trained else 'Declared200steps x256draws. All8 tensors/12,872parameters may train from the frozen control. Training waits for source reservations and admitted terminal replay.',
        'data/selfplay-training-pilot-v1/result.json; build/selfplay-training-pilot-model-witness.json',
        'Optimizer/checkpoint/RNG state must resume exactly. The updated challenger never generates its own games in this one-shot pilot.'))
    cards.append(old.card('selfplay-pilot-model','Actual checkpoint audit and CPU export admission','complete' if modelproof else 'held',
        'Actual Torch checkpoint/export inspection proof is available; evaluation itself remains CPU-only.' if modelproof else 'Waiting for actual fixed-final checkpoint inspection and CPU-only witness. No placeholder model is admitted to production evaluation.',
        'build/selfplay-training-pilot-model-witness.json',
        'Training can change policy and trunk in this scope. Preserve model SHA, data/source/reservation identity and actual optimizer/RNG replay proof.'))
    cards.append(old.card('selfplay-pilot-exact','Known-exact endgame guard','complete' if result else 'running' if bc.get('cases',0) else 'held',
        str(bc.get('cases',0))+'/32 exact cases committed;256 scheduled scalar measurements at25/100ms across4 engines. '+(str(result.get('endgame_guard',{})) if result else 'No completed guard result yet.'),
        'build/selfplay-pilot-benchmark.json: endgame_guard',
        'Reused K<=4 exact diagnostics measure completed recommendations and true action regret. They are not an unseen release test and cannot auto-promote the challenger.'))
    cards.append(old.card('selfplay-pilot-strength','Equal-time paired control/challenger versus both baselines','complete' if result else 'running' if bc.get('decisions',0) else 'held',
        str(bc.get('games',0))+'/96 games and '+str(bc.get('decisions',0))+' decisions committed. '+('Both-baseline gates: '+str(result.get('both_baseline_primary_gates'))+'.' if result else '12 paired families,2seats, H_v1 and H8relay; contemporaneous interleaving at25ms.'),
        'data/selfplay-pilot-benchmark-v1/work.sqlite3; build/selfplay-pilot-benchmark.json',
        'Whole clocks include feature/inference costs. Resume exact committed decision prefixes. Report paired margin intervals, full tails, fallback/terminal gates and actual posthoc replay overlap.'))
    accepted=read(ROOT/'build/selfplay-pilot-acceptance.json') or {};stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else [];okay=bool(accepted.get('passed') and not stale)
    cards.append(old.card('selfplay-pilot-audit','Independent audit, interruptions and immutable completion','complete' if okay else 'stale' if stale else 'running',
        str(accepted.get('tests_run'))+' unique current checks passed; all2839 old pins retained. Completed pipeline makes no new games/searches/optimizer work.' if okay else 'Building and testing real self-play/checkpoint/benchmark kill-resume barriers, role masks, terminal-return sign, policy targets and completed zero-work readers.',
        'build/selfplay-pilot-audit.json; build/selfplay-pilot-acceptance.json',
        'No automatic model promotion. Small development results and any training-trajectory collision remain descriptive; H_v1 stays default.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Building reserved self-play, joint-training and equal-time benchmark/recovery checks; no production launched yet.')
    return {**base,'schema':'selfplay-pilot-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'One fixed-control self-play generation round, joint policy/value/trunk training, then interleaved equal-time comparisons versus H_v1 and H8relay with exact endgame guard. No automatic promotion or sealed final access.'}


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
                if self.path=='/':data=(ROOT/'tools/selfplay-pilot-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Self-play pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='selfplay-pilot');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
