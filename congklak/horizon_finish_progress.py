"""Read-only five-root H14 finish observer; all prior accepted sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,horizon_probe_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/horizon-finish-progress'
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
    result=dict(roots=0,H12_references=0,reused_H14=0,tier1_attempts=0,tier1_labels=0,tier1_rejected=0,tier2_eligible=0,tier2_attempts=0,tier2_labels=0,tier2_rejected=0,new_H14_complete=0,combined_H14_complete=0,cases=[])
    path=ROOT/'data/horizon-finish-v1/work.sqlite3'
    if not path.exists():return result
    with old.database(path) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table,key in (('roots','roots'),('h12_references','H12_references'),('prior_h14_labels','reused_H14'),('tier1_attempts','tier1_attempts'),('tier1_labels','tier1_labels'),('tier2_attempts','tier2_attempts'),('tier2_labels','tier2_labels')):
            if table in tables:result[key]=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
        for tier in ('tier1','tier2'):
            if tier+'_attempts' in tables:
                result[tier+'_rejected']=db.execute("SELECT COUNT(*) FROM "+tier+"_attempts WHERE json_extract(body,'$.status')='budget_rejected'").fetchone()[0]
        result['tier2_eligible']=result['tier1_rejected']
        if {'tier1_labels','tier2_labels'}<=tables:
            result['new_H14_complete']=db.execute('SELECT COUNT(*) FROM (SELECT state FROM tier1_labels UNION SELECT state FROM tier2_labels)').fetchone()[0]
        if {'prior_h14_labels','tier1_labels','tier2_labels'}<=tables:
            result['combined_H14_complete']=db.execute('SELECT COUNT(*) FROM (SELECT state FROM prior_h14_labels UNION SELECT state FROM tier1_labels UNION SELECT state FROM tier2_labels)').fetchone()[0]
        if 'roots' in tables:
            for index,(key,body) in enumerate(db.execute('SELECT state,body FROM roots ORDER BY state'),1):
                root=json.loads(body);case=dict(case_id='D'+str(index),state_key=key,cohort=root.get('cohort','pending'),tiers={})
                case['reused_prior_H14']=bool('prior_h14_labels' in tables and db.execute('SELECT 1 FROM prior_h14_labels WHERE state=?',(key,)).fetchone())
                for tier in ('tier1','tier2'):
                    if tier+'_attempts' not in tables:continue
                    row=db.execute("SELECT json_extract(body,'$.status'),json_extract(body,'$.node_budget'),json_extract(body,'$.completed_depth') FROM "+tier+"_attempts WHERE state=?",(key,)).fetchone()
                    if row:case['tiers'][tier]=dict(status=row[0],node_budget=row[1],completed_depth=row[2])
                result['cases'].append(case)
    return result


def compact(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def snapshot():
    base=previous_snapshot();cards=[];baseline=read(ROOT/'build/horizon-finish-baseline.json');plan=read(ROOT/'build/horizon-finish-plan.json');impl=read(ROOT/'build/horizon-finish-implementation.json')
    fixed=baseline.get('fixed_model_sha256','pending') if baseline else 'pending'
    cards.append(old.card('horizon-finish-plan','Finish exactly five prior caps with fixed model','complete' if baseline and plan and impl else 'running',
        'Control model '+fixed[:16]+'... remains fixed. Only the five old capped H14 roots get new attempts: 250M nodes first, then 1B only if still capped. Old D2 H14 is reused unchanged.',
        'build/horizon-finish-baseline.json; build/horizon-finish-plan.json; build/horizon-finish-registry.json',
        'No new roots, collection, training, ingestion, active weights or promotion. This scope explicitly uses larger budgets; prior 25M outcomes remain historical evidence.'))
    counts=committed_counts();report=read(ROOT/'build/horizon-finish-result.json')
    text=f"{counts['tier1_attempts']}/5 initial attempts committed; {counts['tier1_labels']}/5 complete H14 labels and {counts['tier1_rejected']} retained 250M budget rejects. {5-counts['tier1_attempts']} initial attempts pending."
    cards.append(old.card('horizon-finish-tier1','Initial H14 attempts | 250M per root','complete' if counts['tier1_attempts']==5 else 'partial' if counts['tier1_attempts'] else 'planned',text,
        'data/horizon-finish-v1/work.sqlite3: tier1_attempts, tier1_labels',
        'Retain full actual-depth histories and the initial cap outcome. A rejected initial attempt does not become a complete requested label.',
        [old.count('committed initial attempts',counts['tier1_attempts'],5),old.count('initial complete H14 labels',counts['tier1_labels'],5)]))
    eligible=counts['tier2_eligible'];tier2_done=counts['tier1_attempts']==5 and counts['tier2_attempts']==eligible
    text=f"{counts['tier2_attempts']}/{eligible} currently eligible fallback attempts committed; {counts['tier2_labels']} complete H14 labels and {counts['tier2_rejected']} rejects under the 1B cap."
    if counts['tier1_attempts']<5:text+=' Eligibility is not final until all five initial attempts are committed.'
    elif not eligible:text+=' No fallback is needed.'
    cards.append(old.card('horizon-finish-tier2','Conditional H14 fallback | 1B per eligible root','complete' if tier2_done else 'partial' if counts['tier2_attempts'] else 'planned',text,
        'data/horizon-finish-v1/work.sqlite3: tier2_attempts, tier2_labels',
        'Fallback is authorized only for a genuinely capped initial attempt. Preserve both histories. Any root still capped at 1B blocks an all-five-complete acceptance.',
        [old.count('committed eligible fallbacks',counts['tier2_attempts'],eligible)]))
    comparison=read(ROOT/'build/horizon-finish-comparison.json');valid=bool(comparison and comparison.get('passed') and not old.current_evidence(ROOT,comparison))
    exhausted=bool(report and report.get('status')=='budget_exhausted')
    text=f"{counts['new_H14_complete']}/5 new requested H14 labels; {counts['reused_H14']}/1 reused old H14 label; {counts['combined_H14_complete']}/6 combined H14 labels with {counts['H12_references']}/6 unchanged H12 references."
    if comparison:text+=' Comparison counts: '+compact(comparison['counts'])+'.'
    if exhausted:text+=' Authorized tiers are exhausted with unresolved roots; no completion claim is made.'
    cards.append(old.card('horizon-finish-comparison','Combined six-root H12/H14 comparison','complete' if valid else 'held' if exhausted else 'partial' if counts['new_H14_complete'] else 'planned',text,
        'build/horizon-finish-result.json; build/horizon-finish-comparison.json',
        'Require all five new complete H14 labels plus the old unchanged sixth label. Equal best sets at two finite depths do not establish convergence or solve the positions.',
        [old.count('new requested H14 labels',counts['new_H14_complete'],5),old.count('combined H12/H14 coverage',counts['combined_H14_complete'],6)]))
    parts=[]
    if comparison:
        for case in comparison['cases']:
            p=case['H14_provenance'];best12=case['depth_evidence']['12']['display_best_moves'];best14=case['depth_evidence']['14']['display_best_moves']
            parts.append(case['case_id']+' '+case['cohort']+': H12 '+compact(best12)+' -> H14 '+compact(best14)+'; '+case['classification']+'; source '+p['source_job']+' ('+str(p['node_budget'])+' nodes)')
    else:
        for case in counts['cases']:
            detail='reused unchanged prior full H14' if case['reused_prior_H14'] else compact(case['tiers'])
            parts.append(case['case_id']+' '+case['cohort']+': '+detail)
    cards.append(old.card('horizon-finish-cases','Case outcomes, ties and H14 provenance','complete' if valid else 'partial' if parts else 'planned',
        ' | '.join(parts)+'. Human pit numbers are 1-7.' if parts else 'Awaiting committed six-root references and new five-root attempts.',
        'data/horizon-finish-v1/work.sqlite3; build/horizon-finish-comparison.json',
        'Keep original ownership, full best-move ties and actual source budgets. Investigation labels are not ingested into training.'))
    accepted=read(ROOT/'build/horizon-finish-acceptance.json');stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    text=f"{accepted['tests_run']} unique new checks passed; all 2314 prior artifact pins preserved. All five new requested H14 labels are complete." if okay else 'Verifying initial/fallback real process kills, budget eligibility, original references, all five H14 completions, six-root comparison and zero-search completed reruns.'
    cards.append(old.card('horizon-finish-audit','Recovery, independent audit and completion gate','complete' if okay else 'stale' if stale else 'held' if exhausted else 'running',text,
        'build/horizon-finish-audit.json; build/horizon-finish-acceptance.json; docs/HORIZON_FINISH_RESULTS.md',
        'No final acceptance while a requested root remains unresolved. Engineering completion does not authorize collection, training, weights or promotion.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if counts['tier1_rejected']:cards[1]['needs_improvement']=True
    if counts['tier2_rejected']:cards[2]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Finishing the five capped H14 roots with fixed 250M initial and conditional 1B fallback budgets. No collection or training.')
    historical=[{**card,'title':'Prior 25M probe | '+card['title']} if card['id'].startswith('horizon-probe-') else card for card in base['tools']]
    return {**base,'schema':'horizon-finish-progress-v1','observed_at':old.utc(),'tools':cards+historical,'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Betawi five-root H14 finish with authorized larger budgets. Preserve the prior 25M scope, old D2 full H14, all H12 references and ownership. No collection, training, ingestion, active weights, promotion or exact-solution claim.'}


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
                if self.path=='/':data=(ROOT/'tools/horizon-finish-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Horizon finish observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='horizon-finish');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
