"""Read-only value-head pilot, validation, equal-time and allocation observer."""
from collections import Counter
from contextlib import closing
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import sqlite3
from . import config,progress as old,ordinary_selfplay_progress as previous
from .native import ROOT

FOLDER=ROOT/'build/ordinary-value-progress'
read=old.read_json


def save(path,body):path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,body)


def committed(path):
    if not path.exists():return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
            db.execute('BEGIN');tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};result={}
            for name in ('roots','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels','checkpoints','cases','games','decisions','families','sp_games','sp_decisions','sp_rows'):
                if name not in tables:continue
                result[name]=db.execute('SELECT count(*) FROM '+name).fetchone()[0]
                if name in ('games','sp_games'):
                    result[name+'_statuses']=dict(Counter(json.loads(text).get('status','unknown') for text, in db.execute('SELECT body FROM '+name)))
            return result
    except sqlite3.OperationalError as exc:return dict(observation_pending=str(exc))


def runs():
    values=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:15]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        values.append(row)
    return values


def snapshot():
    base=previous.snapshot();cards=[]
    projection=read(ROOT/'build/ordinary-value-benchmark-projection.json') or {};labels=read(ROOT/'build/ordinary-value-labels.json') or {};data=read(ROOT/'build/ordinary-value-data.json') or {};training=read(ROOT/'data/ordinary-value-pilot-v1/result.json') or {}
    proof=read(ROOT/'build/ordinary-value-model-witness.json') or {};result=read(ROOT/'build/ordinary-value-benchmark.json') or {};followup=read(ROOT/'build/ordinary-value-followup.json') or {};accepted=read(ROOT/'build/ordinary-value-acceptance.json') or {}
    lc=committed(ROOT/'data/ordinary-value-labels-v1/work.sqlite3');tc=committed(ROOT/'data/ordinary-value-pilot-v1/catalog.sqlite3');bc=committed(ROOT/'data/ordinary-value-benchmark-v1/work.sqlite3');fc=committed(ROOT/'data/ordinary-value-followup-v1/work.sqlite3')
    def card(key,title,state,text,evidence,improvement):
        value=old.card('ordinary-value-'+key,title,state,text,evidence,improvement);value['needs_improvement']=state!='complete';cards.append(value)
    card('reservation','Fixed policy and evaluation reservations','complete' if projection else 'running',
        str(len(projection.get('arena_starts',[])))+'/12 arena starts; '+str(len(projection.get('validation_roots',[])))+'/6 reserved validation roots;32 reused exact diagnostics. Control stays bc1160e...; only65 value-head parameters may train.',
        'build/ordinary-value-benchmark-projection.json; docs/ORDINARY_VALUE_SCOPE.md','New-to-prior-arena IDs remain consumed development with ancestral exposure. Six trunk/policy tensors must stay bit-exact; no release or promotion claim.')
    label_text='; '.join(f"H{depth}: {lc.get('h'+str(depth)+'_attempts',0)}/6 committed attempts / {lc.get('h'+str(depth)+'_labels',0)} complete labels / {lc.get('h'+str(depth)+'_attempts',0)-lc.get('h'+str(depth)+'_labels',0)} rejected" for depth in (8,10,12))
    card('labels','Six protected roots: requested horizons and capped evidence','complete' if labels.get('status')=='complete' else 'running' if lc.get('h8_attempts',0) else 'held',label_text+'. RequestedH10 is primary; missing requested labels never borrow an auxiliary depth.',
        'data/ordinary-value-labels-v1/work.sqlite3','H8/H10/H12 caps2M/5M/10M independently per root. Retain incomplete attempts as typed evidence; finite-H targets are approximate and all6 are needed only for favorable allocation.')
    card('data','Protected optimizer pools and declared unit-loss mixture','complete' if data else 'held',
        'Closed optimizer groups '+str(data.get('groups',{}))+'; paired initial starts '+str(data.get('paired_starts',{}))+'. Excluded origin families '+str(len(data.get('excluded_origin_families',{})))+'; source rehearsal roots '+str(data.get('source_base_rows',0))+'; excluded descendants '+str(len(data.get('excluded_descendants',[])))+'.' if data else 'Waiting for protected high/low initial-root observations plus admitted historical rehearsal. Exclude14 ordinary descendants and validation/evaluation ownership before any optimizer draws.',
        'build/ordinary-value-data.json; build/ordinary-value-rows.json; build/ordinary-value-registry.json','New arena reservations, latest96 full trajectories and the whole ordinary validation mask must enter origin-family closure. No posthoc filtering or retraining after arena outcomes.')
    card('training','Small CPU value-head optimization','complete' if training else 'running' if tc.get('checkpoints',0) else 'held',
        'Fixed-final result available: '+str(training.get('steps_completed',training.get('steps','see result')))+' steps; '+str(training.get('draws','see result'))+' actual draws.' if training else str(tc.get('checkpoints',0))+' checkpoints committed; declared200 steps x64draws, CPU deterministic. All six trunk/policy tensors stay frozen.',
        'data/ordinary-value-pilot-v1/result.json; data/ordinary-value-pilot-v1/catalog.sqlite3','No checkpoint or sampling selection from validation/arena outcomes. Full optimizer/RNG recovery must precede the single production launch.')
    card('model','Actual checkpoint replay and fixed-policy CPU export','complete' if proof else 'held',
        'Actual exported-model/checkpoint proof available; production benchmark remains CPU-only.' if proof else 'Waiting for actual final checkpoint inspection, frozen-tensor proof and CPU-only witness; no placeholder candidate admitted.',
        'build/ordinary-value-model-witness.json','Value updates can change PUCT choices through backups while global raw policy stays identical. Model source/hash and real optimizer states must be independently verified.')
    states=bc.get('games_statuses',{})
    text=f"{bc.get('cases',0)}/38 cases; {bc.get('games',0)}/96 game traces / {states.get('terminal',0)} terminal; {bc.get('decisions',0)} decisions."
    if result:text+=' Exact-only guard '+str(result['endgame_guard'])+'; primary-H10 label coverage '+str(result['validation_model_fit']['complete_reference_roots'])+'/6; recorded train/arena contamination '+str(result['trajectory_replay_overlap']['contaminated_evaluation'])+'.'
    card('benchmark','Equal whole-time strength, exact guard and finite-H validation','complete' if result else 'running' if bc.get('cases',0) or bc.get('decisions',0) else 'held',text,
        'build/ordinary-value-benchmark.json; build/ordinary-value-benchmark-resources.json','25ms paired/interleaved games versus BOTH H_v1 and H8relay;32 exact roots and6 separate approximate-H10 cases at25/100ms. Missing references keep recommendations but no invented regret.')
    branch=followup.get('identity',{}).get('branch',followup.get('branch','pending'));summary=followup.get('summary',{});statuses=fc.get('sp_games_statuses',{})
    card('allocation','Bounded result-based data allocation; further training held','complete' if followup.get('status') in ('complete','held') else 'running' if fc.get('sp_decisions',0) or fc.get('families',0) else 'held',
        'Branch '+str(branch)+f"; {fc.get('families',0)} prefixes, {fc.get('sp_games',0)} initialized game slots / {statuses.get('terminal',0)} terminal / {statuses.get('capped',0)} capped; {fc.get('sp_decisions',0)} committed choices. "+('Closed summary '+str(summary) if followup else 'Awaiting strict benchmark and independent proof before allocation.'),
        'build/ordinary-value-followup.json; build/ordinary-value-followup-plan.json','Clean favorable fit/paired means allows at most24 newstarts/48games; otherwise valid mixed/incomplete results repeat each original12 highstarts twice (24games). Engineering/endgame/collision failures hold. No new training or promotion follows automatically.')
    stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else [];okay=bool(accepted.get('passed') and not stale)
    card('audit','Independent recovery, no-work completion and preserved history','complete' if okay else 'stale' if stale else 'running',
        str(accepted.get('tests_run'))+' actual unique checks passed; all3364 prior pins current.' if okay else 'Owner and independent audits are building/source-bound; real label/checkpoint/benchmark/followup interruption barriers must pass before their production stages.',
        'build/ordinary-value-audit.json; build/ordinary-value-acceptance.json','Keep exact committed prefixes and capped-evidence semantics. Completed reruns perform no new games/searches/optimization; literal CPU verification remains permitted. Browser access is not used.')
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Building frozen validation, fixed-policy value pilot, equal-time tests and bounded data allocation; production held for preflight.')
    return {**base,'schema':'ordinary-value-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Six protected finite-H validation labels, one fixed-final value-head pilot, equal-time both-baseline development tests, exact endgame guard and one bounded data-only allocation. No automatic model promotion or subsequent training.'}


def run_command(command,tool):
    import os,subprocess,time,uuid
    FOLDER.mkdir(parents=True,exist_ok=True);uid=uuid.uuid4().hex;path=FOLDER/f'run-{uid}.json';log=FOLDER/f'run-{uid}.log';started=time.monotonic();child=None
    row=dict(tool=tool,command=command,pid=os.getpid(),process_identity=list(old.process_identity(os.getpid())),state='running',started_at=old.utc(),updated_at=old.utc(),last_event='Launching',log=str(log.relative_to(ROOT)));save(path,row)
    try:
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
                try:event=json.loads(line);message=event.get('event',line.strip()[:200])
                except ValueError:message=line.strip()[:200]
                row.update(last_event=message,updated_at=old.utc());save(path,row)
            code=child.wait()
        row.update(state='complete' if code==0 else 'failed',returncode=code,elapsed_seconds=time.monotonic()-started,updated_at=old.utc());save(path,row);return code
    except BaseException:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        row.update(state='interrupted',elapsed_seconds=time.monotonic()-started,updated_at=old.utc());save(path,row);raise


def serve(port=18766):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':data=(ROOT/'tools/ordinary-value-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Ordinary value pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True);sub.add_parser('snapshot');sub.add_parser('serve').add_argument('--port',type=int,default=18766)
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    run=sub.add_parser('run');run.add_argument('--tool',default='ordinary-value');run.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='serve':serve(args.port)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)


if __name__=='__main__':raise SystemExit(main())
