"""Read-only exact-six H12/H14 horizon probe observer; prior sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,natural_study_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/horizon-probe-progress'
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


def committed_counts():
    result=dict(roots=0,h8_references=0,h10_references=0,h12_attempts=0,h12_labels=0,h12_rejected=0,h14_attempts=0,h14_labels=0,h14_rejected=0,requested_complete_pairs=0,cases=[])
    path=ROOT/'data/horizon-probe-v1/work.sqlite3'
    if not path.exists():return result
    with old.database(path) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ('roots','h8_references','h10_references','h12_attempts','h12_labels','h14_attempts','h14_labels'):
            if table in tables:result[table]=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
        for stage in ('h12','h14'):
            if stage+'_attempts' in tables:
                result[stage+'_rejected']=db.execute("SELECT COUNT(*) FROM "+stage+"_attempts WHERE json_extract(body,'$.status')='budget_rejected'").fetchone()[0]
        if {'h12_labels','h14_labels'}<=tables:
            result['requested_complete_pairs']=db.execute('SELECT COUNT(*) FROM h12_labels a JOIN h14_labels b ON a.state=b.state').fetchone()[0]
        if 'roots' in tables:
            for key,body in db.execute('SELECT state,body FROM roots ORDER BY state'):
                root=json.loads(body);case=dict(state_key=key,cohort=root.get('cohort','pending'),stages={})
                for stage in ('h12','h14'):
                    if stage+'_attempts' not in tables:continue
                    item=db.execute("SELECT json_extract(body,'$.status'),json_extract(body,'$.requested_depth'),json_extract(body,'$.completed_depth') FROM "+stage+"_attempts WHERE state=?",(key,)).fetchone()
                    if item:
                        depths=[r[0] for r in db.execute("SELECT json_extract(j.value,'$.choice_depth') FROM "+stage+"_attempts a,json_each(a.body,'$.history') j WHERE a.state=? ORDER BY j.key",(key,))]
                        case['stages'][stage]=dict(status=item[0],requested_depth=item[1],completed_depth=item[2],history_depths=depths)
                result['cases'].append(case)
    return result


def compact(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def snapshot():
    base=previous_snapshot();cards=[];baseline=read(ROOT/'build/horizon-probe-baseline.json');plan=read(ROOT/'build/horizon-probe-plan.json');impl=read(ROOT/'build/horizon-probe-implementation.json')
    fixed=baseline.get('fixed_model_sha256','pending') if baseline else 'pending'
    cards.append(old.card('horizon-probe-plan','Exactly six prior disagreements and fixed model','complete' if baseline and plan and impl else 'running',
        'Control model '+fixed[:16]+'... stays fixed. Only the six prior disjoint H8/H10 best-set roots are investigated; old roots, references and original ownership stay unchanged.',
        'build/horizon-probe-baseline.json; build/horizon-probe-plan.json; build/horizon-probe-registry.json',
        'Investigation only: no training ingestion, active weights or promotion. Requested H12/H14 complete labels remain separate from auxiliary completed-depth reports.'))
    counts=committed_counts();report=read(ROOT/'build/horizon-probe-result.json')
    for stage,depth,cap in (('h12',12,'10M'),('h14',14,'25M')):
        attempts=counts[stage+'_attempts'];labels=counts[stage+'_labels'];rejects=counts[stage+'_rejected']
        text=f"{attempts}/6 attempts committed, {labels}/6 complete H{depth} labels, {rejects} budget rejects and {6-attempts} pending. The {cap}-node cap stays fixed."
        cards.append(old.card('horizon-probe-'+stage,'Requested H'+str(depth)+' labels and bounded attempts','complete' if attempts==6 else 'partial' if attempts else 'planned',text,
            'data/horizon-probe-v1/work.sqlite3: '+stage+'_attempts, '+stage+'_labels',
            'A budget stop retains its completed history at each actual depth; it does not become a complete requested-horizon label or trigger a larger budget.',
            [old.count('committed H'+str(depth)+' attempts',attempts,6),old.count('complete H'+str(depth)+' labels',labels,6)]))
    forensic=read(ROOT/'build/horizon-probe-forensics.json');valid=bool(forensic and forensic.get('passed') and not old.current_evidence(ROOT,forensic))
    text=f"{counts['requested_complete_pairs']}/6 roots have complete requested H12 AND H14 labels. Incomplete roots stay outside that comparison; auxiliary depths retain their source job."
    if forensic:text+=' Forensic counts: '+compact(forensic['counts'])+'.'
    cards.append(old.card('horizon-probe-comparison','Comparison eligibility and horizon diagnosis','complete' if valid else 'stale' if forensic else 'planned',text,
        'build/horizon-probe-forensics.json; tools/analyze-horizon-probe.py',
        'Deeper agreement is finite-horizon evidence, not proof that a case or the full game is solved. No new playing-strength test is performed.',
        [old.count('complete requested H12/H14 pairs',counts['requested_complete_pairs'],6)]))
    parts=[]
    for case in counts['cases']:
        details=[]
        for stage,row in case['stages'].items():details.append(stage.upper()+': '+row['status']+', actual completed depths '+compact(row['history_depths']))
        parts.append(case['state_key'][:12]+'... '+case['cohort']+' ['+'; '.join(details)+']')
    text=' | '.join(parts) if parts else 'Awaiting committed exact-six roots and typed auxiliary history. No additional positions are selected.'
    if forensic:
        outcomes=[]
        for row in forensic['cases']:
            requested=[]
            for depth in (8,10,12,14):
                entry=row['depth_evidence'].get(str(depth))
                if depth<=10 or depth in row['requested_depths_available']:
                    requested.append('H'+str(depth)+' best pits '+compact(entry['display_best_moves']))
                else:requested.append('H'+str(depth)+' requested unavailable')
            auxiliary={str(depth):row['depth_evidence'][str(depth)]['source_job'] for depth in row['auxiliary_completed_depths']}
            outcomes.append(row['case_id']+' '+row['cohort']+': '+row['classification']+'; '+', '.join(requested)+'; auxiliary actual-depth/source '+compact(auxiliary))
        text=' | '.join(outcomes)+'. Display pits are 1-7.'
    cards.append(old.card('horizon-probe-cases','Six-case outcomes and auxiliary actual depths','complete' if valid else 'partial' if counts['cases'] else 'planned',text,
        'data/horizon-probe-v1/work.sqlite3; build/horizon-probe-forensics.json',
        'Inspect PV changes, store gains, captures, bonuses, passes and leaf states with actor-correct semantics. Human pit numbers are 1-7; retained machine moves are 0-6.'))
    accepted=read(ROOT/'build/horizon-probe-acceptance.json');stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    text=f"{accepted['tests_run']} unique new checks passed; all 2197 prior sealed inputs preserved. Recovery and completed reruns retain durable artifacts." if okay else 'Verifying actual H12/H14 pre/postcommit kills, typed history, complete-label boundaries, forensic PVs and unchanged completed artifacts.'
    cards.append(old.card('horizon-probe-audit','Recovery, independent audit and next decision','complete' if okay else 'stale' if stale else 'running',text,
        'build/horizon-probe-audit.json; build/horizon-probe-acceptance.json; docs/HORIZON_PROBE_RESULTS.md',
        'Keep the bounded outcome and any deeper rejects visible. No training, ingestion, automatic budget expansion or promotion follows from engineering acceptance.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if counts['h12_rejected']:cards[1]['needs_improvement']=True
    if counts['h14_rejected']:cards[2]['needs_improvement']=True
    if counts['requested_complete_pairs']!=6:cards[3]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Investigating exactly six H8/H10 disagreements with bounded H12/H14 searches. Fixed model; no training, ingestion or active weight changes.')
    return {**base,'schema':'horizon-probe-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Betawi exact-six horizon investigation. Fixed model, original ownership and references; complete requested labels separate from auxiliary actual depths. No training, ingestion, active weights, budget expansion, promotion or exact-solution claim.'}


def any_positive(value):
    if isinstance(value,dict):return any(any_positive(v) for v in value.values())
    if isinstance(value,list):return any(any_positive(v) for v in value)
    return type(value) in (int,float) and value>0


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
                if self.path=='/':data=(ROOT/'tools/horizon-probe-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Horizon probe observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='horizon-probe');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
