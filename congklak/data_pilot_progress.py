"""Read-only ordinary-data pilot observer; prior accepted sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,value_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/data-pilot-progress'
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
    result=dict(families=0,root_records=0,attempt_records=0,accepted_labels=0,budget_rejected=0,cohort_roots={},selection=None)
    path=ROOT/'data/ordinary-data-pilot-v1/work.sqlite3'
    if not path.exists():return result
    with old.database(path) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table,key in (('families','families'),('roots','root_records'),('attempts','attempt_records'),('labels','accepted_labels')):
            if table in tables:result[key]=db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
        if 'roots' in tables:
            for cohort,band,n in db.execute("SELECT json_extract(body,'$.cohort'),json_extract(body,'$.band'),count(*) FROM roots GROUP BY 1,2"):
                if cohort in ('diagnostic','training_candidate','validation'):
                    result['cohort_roots'][str(cohort)+'/'+str(band)]=n
        if 'attempts' in tables:
            result['budget_rejected']=db.execute("SELECT count(*) FROM attempts WHERE json_extract(body,'$.status')='budget_rejected'").fetchone()[0]
        if 'meta' in tables:
            row=db.execute("SELECT body,hash FROM meta WHERE key='selection'").fetchone()
            if row:
                value=json.loads(row[0])
                if config.digest(value)!=row[1]:raise ValueError('Corrupt committed pilot selection')
                result['selection']=value
    return result


def compact(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def snapshot():
    base=previous_snapshot();cards=[];baseline=read(ROOT/'build/data-pilot-baseline.json');plan=read(ROOT/'build/data-pilot-plan.json');impl=read(ROOT/'build/data-pilot-implementation.json')
    fixed=baseline.get('fixed_model_sha256','pending') if baseline else 'pending'
    cards.append(old.card('data-pilot-plan','Fixed model and data ownership','complete' if baseline and plan and impl else 'running',
        'Control model '+fixed[:16]+'... remains fixed. Training is stopped; no new optimizer weights or model promotion.',
        'build/data-pilot-baseline.json; build/data-pilot-plan.json; build/data-pilot-registry.json',
        'Keep diagnostic, training_candidate and validation ownership separate; old protected trajectories cannot become training rows.'))
    sampling=read(ROOT/'build/sampling-review.json');sampling_ok=bool(sampling and sampling.get('passed') and not old.current_evidence(ROOT,sampling))
    text='Comparing uniform-root, uniform-stratum, square-root count and 90/10 root/stratum exposure offline. No distribution is applied.'
    if sampling:
        strata=sampling['distributions']['uniform_stratum'];sqrt=sampling['distributions']['sqrt_count'];mix=sampling['distributions']['root90_stratum10']
        text=f"Distribution effective root support: stratum {strata['root_effective_support']:.1f}, sqrt {sqrt['root_effective_support']:.1f}, mixture {mix['root_effective_support']:.1f}. No optimizer runs."
    cards.append(old.card('data-pilot-sampling','Sampling repetition and objective review','complete' if sampling_ok else 'running',text,
        'build/sampling-review.json; tools/review-sampling.py',
        'Changing draws with unit loss weights changes the objective. Effective support measures concentration, not independent games or gradient ESS.'))
    counts=committed_counts();report=read(ROOT/'build/data-pilot-result.json');selected=report.get('selection') if report else counts['selection']
    selection=selected.get('counts') if selected else None;shortfall=selected.get('shortfall') if selected else None
    text=f"{counts['families']}/256 standard-opening history prefixes committed; {counts['root_records']} unique root records. Histories stop below 25 active seeds or at 192 choices."
    if selection is not None:text+=' Selected cohort/band counts: '+compact(selection)+'.'
    if shortfall is not None:text+=' Shortfalls: '+compact(shortfall)+'.'
    if selected:text+=' Yield '+str(selected['ordinary_observations'])+' ordinary observations / '+str(selected['ordinary_unique_quotients'])+' unique quotients. Selected legal-action histogram: '+compact(selected['selected_legal_action_counts'])+'; diagnostic shortfall '+str(selected['diagnostic_shortfall'])+'.'
    cards.append(old.card('data-pilot-collection','Ordinary history yield and protected cohorts','complete' if counts['families']==256 and selected else 'partial' if counts['families'] else 'planned',text,
        'data/ordinary-data-pilot-v1/work.sqlite3; build/data-pilot-result.json',
        'The fixed 256-family pool is measured without quota fillers. Ordinary roots can have one legal move; choice-bearing diversity must be reported separately.',
        [old.count('committed history prefixes',counts['families'],256),old.count('unique root records',counts['root_records'],None)]))
    text=f"{counts['accepted_labels']} accepted complete H8 labels committed; {counts['attempt_records']}/{counts['root_records']} attempt records, {counts['budget_rejected']} budget-rejected, {counts['root_records']-counts['attempt_records']} pending. Pending/rejected results are not complete targets."
    cards.append(old.card('data-pilot-labels','Complete H8 targets and label shortfalls','complete' if report else 'partial' if counts['attempt_records'] else 'planned',text,
        'data/ordinary-data-pilot-v1/work.sqlite3; build/data-pilot-result.json',
        'Retain bounded-attempt rejections and provenance. Approximate H8 labels are finite-choice estimates, not exact solutions.',
        [old.count('committed complete labels',counts['accepted_labels'],None)]))
    diagnosis=read(ROOT/'build/data-pilot-model-diagnosis.json')
    diagnosis_text='Awaiting committed complete labels; no new model, search tuning or strength claim.'
    if diagnosis:
        pieces=[]
        for name,row in diagnosis['choice_bearing_cohorts'].items():
            value=row['value_mae_seeds'];regret=row['policy_teacher_regret']
            pieces.append(name+': '+str(row['rows'])+' choice-bearing labels, value MAE '+('n/a' if value is None else f'{value:.3f}')+', policy H8 regret '+('n/a' if regret is None else f'{regret:.3f}'))
        diagnosis_text='; '.join(pieces)+'. Descriptive seeds; single-legal roots are separated and no new strength test is run.'
    cards.append(old.card('data-pilot-diagnosis','Fixed-model errors on the small labeled pilot','complete' if diagnosis and diagnosis.get('passed') else 'planned',
        diagnosis_text,
        'build/data-pilot-model-diagnosis.json; tools/analyze-data-pilot.py',
        'Use labeled diagnostic errors to choose a future controlled experiment; a tiny sample does not justify large training weight or a generalization claim.'))
    accepted=read(ROOT/'build/data-pilot-acceptance.json');stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    cards.append(old.card('data-pilot-audit','Recovery, independent audit and next decision','complete' if okay else 'stale' if stale else 'running',
        f"{accepted['tests_run']} current checks passed; prior 1829 accepted inputs and completed diagnosis preserved. Model and training weights unchanged." if okay else 'Verifying real collection/label pre/postcommit kills, corruption refusal, complete targets and honest shortfalls.',
        'build/data-pilot-audit.json; build/data-pilot-acceptance.json; docs/DATA_PILOT_RESULTS.md',
        'No rows are pruned and no pilot weights are applied. Further data, sampler or model work needs a new frozen plan and appropriate strength tests.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if selected and (any_positive(shortfall) or selected['diagnostic_shortfall']>0):cards[2]['needs_improvement']=True
    if counts['budget_rejected']>0:cards[3]['needs_improvement']=True
    recent=runs();active=[r for r in recent if r['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(r['tool']+' - '+r['last_event'] for r in active)) if active else activity or dict(state='running',message='Small fixed-model ordinary-data pilot; offline exposure review, no training or promotion.')
    return {**base,'schema':'data-pilot-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Betawi ordinary-data pilot. Fixed control model, separate protected cohorts, complete approximate H8 labels, retained shortfalls. No training, pruning, promotion, energy or unseen-release claim.'}


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
                if self.path=='/':data=(ROOT/'tools/data-pilot-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Data pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='cpu-puct');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
