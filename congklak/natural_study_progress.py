"""Read-only natural 50-74 and H8/H10 study observer; prior sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,data_pilot_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/natural-study-progress'
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
    result=dict(families=0,selected_roots=0,fresh_selected=0,followups=0,h8_new_labels=0,h8_inherited_labels=0,h8_attempts=0,h8_rejected=0,h10_labels=0,h10_attempts=0,h10_rejected=0,paired_complete=0,selection=None)
    path=ROOT/'data/natural-study-v1/work.sqlite3'
    if not path.exists():return result
    with old.database(path) as db:
        tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table,key in (('families','families'),('roots','selected_roots'),('h10_attempts','h10_attempts'),('h10_labels','h10_labels')):
            if table in tables:result[key]=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
        if 'roots' in tables:
            for role,n in db.execute("SELECT json_extract(body,'$.study_role'),COUNT(*) FROM roots GROUP BY 1"):
                if role=='fresh':result['fresh_selected']=n
                elif role=='followup':result['followups']=n
        for stage in ('h8','h10'):
            if stage+'_attempts' in tables:
                result[stage+'_rejected']=db.execute("SELECT COUNT(*) FROM "+stage+"_attempts WHERE json_extract(body,'$.status')='budget_rejected'").fetchone()[0]
        if {'roots','h8_attempts'}<=tables:
            result['h8_attempts']=db.execute("SELECT COUNT(*) FROM h8_attempts a JOIN roots r ON a.state=r.state WHERE json_extract(r.body,'$.study_role')='fresh'").fetchone()[0]
        if {'roots','h8_labels'}<=tables:
            for role,n in db.execute("SELECT json_extract(r.body,'$.study_role'),COUNT(*) FROM h8_labels l JOIN roots r ON l.state=r.state GROUP BY 1"):
                if role=='fresh':result['h8_new_labels']=n
                elif role=='followup':result['h8_inherited_labels']=n
        if {'h8_labels','h10_labels'}<=tables:
            result['paired_complete']=db.execute('SELECT COUNT(*) FROM h8_labels a JOIN h10_labels b ON a.state=b.state').fetchone()[0]
        if 'meta' in tables:
            row=db.execute("SELECT body,hash FROM meta WHERE key='selection'").fetchone()
            if row:
                value=json.loads(row[0])
                if config.digest(value)!=row[1]:raise ValueError('Corrupt committed natural-study selection')
                result['selection']=value
    return result


def compact(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def snapshot():
    base=previous_snapshot();cards=[];baseline=read(ROOT/'build/natural-study-baseline.json');plan=read(ROOT/'build/natural-study-plan.json');impl=read(ROOT/'build/natural-study-implementation.json')
    fixed=baseline.get('fixed_model_sha256','pending') if baseline else 'pending'
    cards.append(old.card('natural-study-plan','Fixed model, teacher horizons and ownership','complete' if baseline and plan and impl else 'running',
        'Control model '+fixed[:16]+'... stays fixed. Fresh H8 cap 2M nodes; H10 cap 5M. No training, active weights, pruning or promotion.',
        'build/natural-study-baseline.json; build/natural-study-plan.json; build/natural-study-implementation.json',
        'Keep 15 existing 50-74 follow-ups in original ownership; protect all old roots and the prior pilot families. H10 is deeper approximate evidence, not exact truth.'))
    counts=committed_counts();report=read(ROOT/'build/natural-study-result.json');selected=report.get('selection') if report else counts['selection']
    shortfall=selected.get('shortfall',selected.get('shortfalls')) if selected else None
    text=f"{counts['families']}/512 standard-opening prefixes committed; {counts['fresh_selected']}/32 fresh ordinary 50-74 roots selected. Prefixes stop below 25 active seeds or at 192 choices."
    if selected:text+=' Selection: '+compact(selected)+'.'
    cards.append(old.card('natural-study-collection','Natural 50-74 yield and cap shortfalls','complete' if counts['families']==512 and selected else 'partial' if counts['families'] else 'planned',text,
        'data/natural-study-v1/work.sqlite3; build/natural-study-result.json',
        'Use the whole 512-family pool before collision closure, with one root per family. Report shortfalls without fillers or automatic expansion.',
        [old.count('committed history prefixes',counts['families'],512),old.count('fresh roots selected',counts['fresh_selected'],32)]))
    text=f"{counts['h8_new_labels']}/{counts['fresh_selected']} fresh complete H8 labels; {counts['h8_attempts']} committed attempts, {counts['h8_rejected']} rejects, {counts['fresh_selected']-counts['h8_attempts']} pending. {counts['h8_inherited_labels']}/{counts['followups']} inherited H8 labels remain unchanged."
    h8_done=counts['families']==512 and counts['followups']==15 and counts['h8_inherited_labels']==15 and counts['h8_attempts']==counts['fresh_selected']
    cards.append(old.card('natural-study-h8','Fresh H8 labels and immutable old targets','complete' if h8_done else 'partial' if counts['h8_attempts'] or counts['h8_inherited_labels'] else 'planned',text,
        'data/natural-study-v1/work.sqlite3: h8_attempts, h8_labels',
        'Only completed full-root H8 labels are admitted. Follow-up targets are copied byte-for-byte with original cohort ownership; rejected attempts remain visible.',
        [old.count('fresh complete H8 labels',counts['h8_new_labels'],counts['fresh_selected']),old.count('inherited complete H8 labels',counts['h8_inherited_labels'],15)]))
    text=f"{counts['h10_labels']}/{counts['selected_roots']} complete H10 labels; {counts['h10_attempts']} committed attempts, {counts['h10_rejected']} rejects, {counts['selected_roots']-counts['h10_attempts']} pending. Complete H8/H10 intersection: {counts['paired_complete']}/{counts['selected_roots']}; unpaired {counts['selected_roots']-counts['paired_complete']}."
    h10_done=counts['families']==512 and counts['followups']==15 and counts['h10_attempts']==counts['selected_roots']
    cards.append(old.card('natural-study-h10','H10 completion and paired-label denominator','complete' if h10_done else 'partial' if counts['h10_attempts'] else 'planned',text,
        'data/natural-study-v1/work.sqlite3: h10_attempts, h10_labels',
        'Never substitute a partial H10 target. Report rejection work and keep complete-pair comparisons restricted to the H8/H10 intersection.',
        [old.count('complete H10 labels',counts['h10_labels'],counts['selected_roots']),old.count('complete H8/H10 pairs',counts['paired_complete'],counts['selected_roots'])]))
    stability=read(ROOT/'build/natural-study-stability.json');valid=bool(stability and stability.get('passed') and not old.current_evidence(ROOT,stability))
    text='Awaiting complete paired targets. Root-level model disagreement and teacher shifts will be descriptive; no new strength games or training.'
    if stability:
        c=stability['counts'];text=f"{c['complete_pairs']}/{c['selected_roots']} complete pairs, {c['unpaired_roots']} excluded from paired means. "
        parts=[]
        def number(value):return 'n/a' if value is None else f'{value:.3f}'
        for name,row in stability['groups'].items():
            means=row['paired_means']
            parts.append(name+': '+str(row['complete_pairs'])+'/'+str(row['selected_roots'])+' pairs; target shift '+number(means['absolute_target_shift'])+'; model MAE '+number(means['value_mae_h8'])+' -> '+number(means['value_mae_h10'])+'; policy regret '+number(means['policy_regret_h8'])+' -> '+number(means['policy_regret_h10']))
        text+=' | '.join(parts)+'. Seeds against approximate teachers; H10 is not exact game truth.'
    cards.append(old.card('natural-study-stability','Teacher stability and fixed-model errors','complete' if valid else 'stale' if stability else 'planned',text,
        'build/natural-study-stability.json; tools/analyze-natural-study.py',
        'Separate fresh/follow-up origin and original cohorts. Best-set changes, target shifts and model error changes do not establish improved playing strength.'))
    accepted=read(ROOT/'build/natural-study-acceptance.json');stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    text=f"{accepted['tests_run']} unique new checks passed; all 2058 prior sealed inputs preserved. No training, weight change or promotion." if okay else 'Verifying actual collection, fresh H8 and H10 pre/postcommit kills; complete targets, rejection truth, paired metrics and closed reruns.'
    cards.append(old.card('natural-study-audit','Recovery, independent audit and next decision','complete' if okay else 'stale' if stale else 'running',text,
        'build/natural-study-audit.json; build/natural-study-acceptance.json; docs/NATURAL_STUDY_RESULTS.md',
        'Decide the next bounded study from paired evidence and missing coverage. No bulk training or model promotion follows automatically from engineering acceptance.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if any_positive(shortfall):cards[1]['needs_improvement']=True
    if counts['h8_rejected']:cards[2]['needs_improvement']=True
    if counts['h10_rejected'] or counts['selected_roots']!=counts['paired_complete']:cards[3]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Bounded natural 50-74 collection and H8/H10 teacher-stability study. Fixed model; no training or active weight changes.')
    return {**base,'schema':'natural-study-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Betawi natural 50-74 and H8/H10 study. Fixed model, separate original ownership, complete approximate labels and visible shortfalls/rejections. No training, active weights, pruning, promotion or unseen-release claim.'}


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
                if self.path=='/':data=(ROOT/'tools/natural-study-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Natural teacher study observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='natural-teacher-study');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
