"""Read-only committed data coverage and offline sampling review observer."""
from collections import Counter
from contextlib import closing
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import sqlite3
from . import config, progress as old, selfplay_pilot_live_progress as previous
from .native import ROOT

FOLDER=ROOT/'build/ordinary-selfplay-progress'
read=old.read_json


def save(path,body):path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,body)


def committed(path):
    if not path.exists():return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
            db.execute('BEGIN');tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};result={}
            for name in ('families','roots','sp_games','sp_decisions','sp_rows'):
                if name not in tables:continue
                result[name]=db.execute('SELECT count(*) FROM '+name).fetchone()[0]
                if name in ('roots','sp_games','sp_rows'):
                    rows=[]
                    for text,digest in db.execute('SELECT body,hash FROM '+name):
                        row=json.loads(text)
                        if config.digest(row)!=digest:raise ValueError('Corrupt committed '+name+' row')
                        rows.append(row)
                    if name=='roots':result['root_cohorts']=dict(Counter(row.get('cohort','unknown') for row in rows));result['root_strata']=dict(Counter(row.get('stratum','unknown') for row in rows))
                    elif name=='sp_games':result['game_statuses']=dict(Counter(row.get('status','unknown') for row in rows))
                    else:result['retained_strata']=dict(Counter(row.get('stratum','unknown') for row in rows))
            return result
    except sqlite3.OperationalError as exc:return dict(observation_pending=str(exc))


def runs():
    result=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:15]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        result.append(row)
    return result


def snapshot():
    prior=previous.snapshot();cards=[]
    baseline=read(ROOT/'build/ordinary-selfplay-baseline.json') or {};registry=read(ROOT/'build/ordinary-selfplay-registry.json') or {};arena=read(ROOT/'build/ordinary-selfplay-prior-arena-review.json') or {}
    collection=read(ROOT/'build/ordinary-selfplay-collection.json') or {};replay=read(ROOT/'build/ordinary-selfplay-replay.json') or {}
    sampling=read(ROOT/'build/ordinary-selfplay-sampling-review.json') or {};audit=read(ROOT/'build/ordinary-selfplay-audit.json') or {};accepted=read(ROOT/'build/ordinary-selfplay-acceptance.json') or {}
    counts=committed(ROOT/'data/ordinary-selfplay-pilot-v1/work.sqlite3');statuses=counts.get('game_statuses',{});summary=replay.get('summary',{})
    selection=collection.get('selection',{});protection=collection.get('rollout_protection',{})
    def card(key,title,state,result,evidence,improvement):
        row=old.card('ordinary-selfplay-'+key,title,state,result,evidence,improvement);row['needs_improvement']=state!='complete';cards.append(row)
    card('diagnosis','Existing 96 games: ordinary decision coverage','complete' if arena else 'running',
        ('96 historical games / 2,958 decisions reviewed. Ordinary>=25: '+str(arena['coverage']['all_decisions']['ordinary_active_ge25_decisions'])+' observations / '+str(arena['coverage']['all_decisions']['ordinary_active_ge25_unique_raw_states'])+' raw states. Four ordinary first divergences are all below25; no stored same-state Q.') if arena else 'Read-only replay of 96 retained games / 2,958 saved decisions; ordinary support and paired outcomes by original start mechanism. No new strength matches.',
        'build/ordinary-selfplay-prior-arena-review.json; congklak/ordinary_selfplay_review.py','First divergence is a shared-state observation. Later trajectories are unmatched; opponent-owned differences and uncovered Q cannot establish a causal blunder.')
    card('protection','Frozen models and role-aware protection','complete' if baseline and registry else 'held',
        'Control and challenger fixed; '+str(baseline.get('inputs_count',0))+' prior hashes reserved. '+str(len(registry.get('latest96_games',[])))+'/96 historical arena trajectories indexed. Entire eligible validation pool is additionally protected in the closed collection before rollouts.',
        'build/ordinary-selfplay-baseline.json; build/ordinary-selfplay-registry.json; docs/ORDINARY_SELFPLAY_SCOPE.md','Data-only scope. Never admit validation roots, protected diagnostics or evaluation trajectory collisions to optimizer candidates; no source-data deletion.')
    card('collection','Natural-prefix collection and independent validation panel','complete' if collection else 'running' if counts.get('families',0) else 'held',
        str(counts.get('families',0))+'/512 families committed; '+str(counts.get('roots',0))+' selected roots. '+('Counts '+str(selection.get('counts',{}))+'; shortfalls '+str(selection.get('shortfall',{}))+'. Entire validation pool protected: '+str(protection.get('new_validation_candidate_observations',0))+' observations / '+str(protection.get('new_validation_unique_quotients',0))+' quotients / '+str(len(protection.get('new_validation_family_ids',[])))+' families.' if collection else 'Declared maxima: 24 training roots (12 high +12 low ordinary) and 6 validation roots (3+3); natural shortfalls stay explicit.'),
        'data/ordinary-selfplay-pilot-v1/work.sqlite3; build/ordinary-selfplay-collection.json','All legal actions must satisfy ordinary classification, active>=25 and at least2 legal moves. Block full-pool ownership components before quotas; no fillers or automatic expansion.')
    card('rollouts','Fixed-control root rollouts: terminal versus initialized','complete' if replay.get('status')=='complete' else 'running' if counts.get('sp_decisions',0) else 'held',
        str(counts.get('sp_games',0))+'/48 maximum initialized game slots; '+str(statuses.get('terminal',0))+' terminal; '+str(statuses.get('capped',0))+' capped; '+str(statuses.get('active',0))+' active; '+str(counts.get('sp_decisions',0))+' durable choices.',
        'data/ordinary-selfplay-pilot-v1/work.sqlite3; build/ordinary-selfplay-replay.json','Only selected training roots roll out twice at64 CPU PUCT simulations. Validation remains protected. Terminal returns are noisy trajectory outcomes, not minimax teacher values.')
    card('admission','Paired initial-root support and ordinary descendants','complete' if replay.get('status')=='complete' else 'held',
        ('Usable start roots '+str(summary.get('usable_start_roots',{}))+'; gate '+str(summary.get('usable_start_gate_passed'))+'; validation panel complete '+str(summary.get('validation_panel_complete'))+'. '+str(summary.get('retained_rows',0))+' retained records / '+str(summary.get('unique_rawstates',0))+' unique raw states / '+str(summary.get('retained_families',0))+' families. Support '+str(summary.get('support',{}))+'.') if summary else str(counts.get('sp_rows',0))+' retained rows committed; waiting for closed summary. Gate: at least8 high and8 low roots with both terminal root-choice0 observations; descendants are counted separately.',
        'build/ordinary-selfplay-replay.json; data/ordinary-selfplay-pilot-v1/work.sqlite3','A raw-state cap of4 retains root-choice0 first, then deterministic hashes. Counts are records/states/families, not independent gradient samples. Below-gate output cannot justify training.')
    card('sampling','Offline family/root/stratum sampling alternatives','complete' if sampling else 'held',
        ('Offline probability review available; no sampler was applied. '+str(sampling.get('counts',{}))) if sampling else 'Waiting for actual closed support. Compare family/root/stratum distributions, inverse-concentration support and maximum exposure; hypothetical25,600 draws for comparability only.',
        'build/ordinary-selfplay-sampling-review.json; tools/review-ordinary-selfplay-sampling.py','Unit loss for all proposals: changing draw probabilities changes the objective. No double weighting or automatic winner. Distribution support is not optimizer ESS or independent games.')
    card('training','Training and model promotion remain held','held','No optimizer, checkpoint, new teacher targets or new equal-time strength games in this scope. Both model files and policies stay fixed.',
        'docs/ORDINARY_SELFPLAY_SCOPE.md','Coverage and rollout-return fit are descriptive. A later training experiment requires a separate justified scope and contemporary evaluation; H_v1 remains default.')
    stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else [];okay=bool(accepted.get('passed') and not stale)
    card('audit','Recovery, independent audit and immutable completion','complete' if okay else 'stale' if stale else 'running',
        str(accepted.get('tests_run'))+' current unique methods passed; complete no-work rerun verified.' if okay else 'Independent audit '+str(audit.get('tests_run',0))+' actual methods reported. Four actual collection/decision kill barriers plus strict replay publication/corruption checks precede acceptance.',
        'build/ordinary-selfplay-audit.json; build/ordinary-selfplay-acceptance.json','Only executed unique methods count. Preserve exact committed prefixes and terminal-target sign; completed reruns must do no search or durable mutation.')
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Reviewing existing ordinary coverage and building protected data-only collection/recovery; no training launched.')
    return {**prior,'schema':'ordinary-selfplay-progress-v1','observed_at':old.utc(),'tools':cards+prior['tools'],'recent_runs':recent+prior['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Fixed-model ordinary data and offline sampling pilot. Historical 96 strength games are read-only evidence; at most48 training-root rollouts create data. No training, new strength games, teacher search or model promotion.'}


def run_command(command,tool):
    import os,subprocess,time,uuid
    FOLDER.mkdir(parents=True,exist_ok=True);uid=uuid.uuid4().hex;path=FOLDER/f'run-{uid}.json';log=FOLDER/f'run-{uid}.log';started=time.monotonic();child=None
    row=dict(tool=tool,command=command,pid=os.getpid(),process_identity=list(old.process_identity(os.getpid())),state='running',started_at=old.utc(),updated_at=old.utc(),last_event='Launching',log=str(log.relative_to(ROOT)));save(path,row)
    try:
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True);text=line.strip()
                try:event=json.loads(text);message=event.get('event',text[:200])
                except ValueError:message=text[:200]
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
                if self.path=='/':data=(ROOT/'tools/ordinary-selfplay-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Ordinary data pilot observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True);sub.add_parser('snapshot');sub.add_parser('serve').add_argument('--port',type=int,default=18766)
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    run=sub.add_parser('run');run.add_argument('--tool',default='ordinary-selfplay');run.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='serve':serve(args.port)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)


if __name__=='__main__':raise SystemExit(main())
