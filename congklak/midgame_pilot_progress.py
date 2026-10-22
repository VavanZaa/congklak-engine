"""Targeted midgame collection and full-root labeling observer."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,pilot_loss_review_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/midgame-pilot-progress'
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
    registry=read(ROOT/'build/midgame-data-pilot-registry.json') or {};collection=read(ROOT/'build/midgame-data-pilot-result.json') or {};labels=read(ROOT/'build/midgame-label-pilot-result.json') or {}
    comparison=read(ROOT/'build/midgame-label-pilot-comparison.json');weights=read(ROOT/'build/midgame-pilot-weight-review.json')
    cs=committed(ROOT/'data/midgame-data-pilot-v1/work.sqlite3',('meta','roots'))
    ls=committed(ROOT/'data/midgame-label-pilot-v1/work.sqlite3',('meta','roots','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels'))
    family_count=collection.get('families',0)
    if not collection:
        import sqlite3
        path=ROOT/'data/midgame-data-pilot-v1/work.sqlite3'
        if path.exists():
            try:
                with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2) as db:family_count=db.execute('SELECT count(*) FROM families').fetchone()[0]
            except sqlite3.Error:pass
    fresh_count=collection.get('selected_roots',len(cs.get('roots',{})))
    cards.append(old.card('midgame-pilot-protection','Role protection and fixed models','complete' if registry else 'running',
        '2639 prior inputs preserved. Latest 144 evaluation trajectories and historical diagnostic/validation roles are protected; eligible family-quotient components are blocked before quotas. Nine original first-divergence states remain diagnostic only.',
        'build/midgame-pilot-baseline.json; build/midgame-data-pilot-registry.json',
        'Control bc1160e... and pilot 7d19e668... remain fixed. No training, PUCT games, ingestion, applied sampling or automatic promotion.'))
    selection=collection.get('selection',{});shortfalls=selection.get('shortfall',{})
    cards.append(old.card('midgame-pilot-collection','Natural midgame collection and quotas','complete' if collection else 'running',
        str(family_count)+'/512 standard-start prefixes committed; '+str(fresh_count)+'/50 maximum fresh roots selected. '+('Shortfalls: '+json.dumps(shortfalls,sort_keys=True)+'. Blocked families: '+str(selection.get('blocked_family_count'))+'.' if collection else 'The full candidate pool closes before selection; pending work is not counted as selected data.'),
        'data/midgame-data-pilot-v1/work.sqlite3; build/midgame-data-pilot-result.json',
        '50-74 ordinary: caps 16 training / 4 validation. 25-49 ordinary, bonus and capture: 8 / 2 each. No quota fillers or expansion.'))
    roots=ls.get('roots',{});stages=labels.get('stages',{})
    for stage,budget in (('h8',2000000),('h10',5000000),('h12',10000000)):
        target=sum(stage!='h12' or row['study_role']=='diagnostic' for row in roots.values())
        attempts=ls.get(stage+'_attempts',{});done=ls.get(stage+'_labels',{});rejected=sum(row['status']=='budget_rejected' for row in attempts.values())
        counts=stages.get(stage,dict(targets=target,attempts=len(attempts),labels=len(done),rejected=rejected,pending=target-len(attempts)))
        state='complete' if labels else 'running'
        text=str(counts['attempts'])+'/'+str(counts['targets'])+' attempts committed; '+str(counts['labels'])+' complete requested labels; '+str(counts['rejected'])+' budget rejections; '+str(counts['pending'])+' pending. Per-root cap '+format(budget,',')+' nodes.'
        if not roots:text='Waiting for closed collection and label-root initialization. '+('Nine diagnostics only.' if stage=='h12' else 'All fresh roots plus all nine diagnostics.')+' Per-root cap '+format(budget,',')+' nodes.'
        cards.append(old.card('midgame-pilot-'+stage,stage.upper()+' full-root Q and typed auxiliary history',state,text,
            'data/midgame-label-pilot-v1/work.sqlite3: '+stage+'_attempts / '+stage+'_labels',
            'Every eligible root is attempted despite earlier rejection. Only complete requested-depth full-root Q is a label; completed partial-depth reports remain auxiliary. No H14 in this scope.'))
    text='Comparing fixed control/pilot forwards and original saved actions at exactly the same full state. Only completed requested-depth intersections enter stability means.'
    if comparison:text='Requested-depth comparison published. '+json.dumps(comparison.get('counts',{}),sort_keys=True)+'. Deeper finite-horizon labels are not solved-game truth.'
    cards.append(old.card('midgame-pilot-comparison','Fixed-model errors and diagnostic horizon movement','complete' if comparison else 'running',text,
        'build/midgame-label-pilot-comparison.json',
        'Candidate-owned and opponent-owned first divergences are distinguished. Saved action regret is relative to this teacher/horizon and does not isolate a causal game blunder.'))
    cards.append(old.card('midgame-pilot-weights','Fresh training-candidate exposure review','held' if weights else 'running',
        'Offline probability proposals published for fresh training candidates with complete requested H10 only; no optimizer or data ingestion has used them.' if weights else 'Waiting for closed requested H10 labels. Review will compare uniform-root, nonempty-target-stratum and stability-weighted distributions offline.',
        'build/midgame-pilot-weight-review.json',
        'Validation and all nine diagnostics are excluded from proposed optimizer distributions. Confidence factors are uncalibrated; changing probabilities changes the risk objective unless explicitly corrected.'))
    accepted=read(ROOT/'build/midgame-pilot-acceptance.json');audit=read(ROOT/'build/midgame-pilot-audit.json')
    stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    cards.append(old.card('midgame-pilot-audit','Recovery, independent audit and immutable completion','complete' if okay else 'stale' if stale else 'running',
        str(accepted.get('tests_run'))+' unique checks passed; all 2639 prior pins preserved; completed rerun makes zero new searches.' if okay else 'Author checks and independent audit cover two collection plus six H8/H10/H12 real kill barriers, retained prefixes, partial-label refusal and completed zero-search reruns. '+('Independent report is available.' if audit else 'Independent final report pending.'),
        'build/midgame-pilot-audit.json; build/midgame-pilot-acceptance.json; docs/MIDGAME_PILOT_RESULTS.md',
        'The 503M declared maximum is the requested production attempt sum. Audit fixtures and lost uncommitted work after interruptions are separate costs. No blind large-data or training decision follows automatically.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if shortfalls and any(n for group in shortfalls.values() for n in group.values()):cards[1]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Building protected midgame collection, bounded labels and independent recovery checks.')
    return {**base,'schema':'midgame-pilot-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Fixed-model targeted midgame data pilot. Fresh natural states and nine protected diagnostics have separate ownership; bounded H8/H10/H12 labels, offline weights only. No training or new PUCT games.'}


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
                if self.path=='/':data=(ROOT/'tools/midgame-pilot-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Midgame pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='midgame-pilot');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
