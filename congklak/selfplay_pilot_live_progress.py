"""Read-only display corrections over the frozen self-play pilot observer."""
from collections import Counter
from contextlib import closing
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import sqlite3
from . import config,selfplay_pilot_progress as frozen
from .native import ROOT


def committed_counts(path):
    """Initialized game slots and terminal games are distinct committed facts."""
    if not path.exists():return {}
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.execute('BEGIN');tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};result={}
        if 'sp_games' in tables:
            games=[]
            for body,digest in db.execute('SELECT body,hash FROM sp_games ORDER BY id'):
                game=json.loads(body)
                if config.digest(game)!=digest:raise ValueError('Observer found a corrupt committed game')
                games.append(game)
            result.update(initialized_games=len(games),game_statuses=dict(Counter(game['status'] for game in games)))
        for name in ('sp_decisions','sp_rows','cases','games','decisions','checkpoints'):
            if name in tables:result[name]=db.execute('SELECT count(*) FROM '+name).fetchone()[0]
        if 'games' in tables:
            result['game_statuses']={status:n for status,n in db.execute("SELECT json_extract(body,'$.status'),count(*) FROM games GROUP BY json_extract(body,'$.status')")}
        return result


def snapshot():
    original=frozen.snapshot();current=[dict(card) for card in original['tools'][:8]];cards={card['id']:card for card in current}
    read=frozen.read;sp=read(ROOT/'build/selfplay-training-pilot-selfplay.json') or {};summary=sp.get('summary',{})
    plan=read(ROOT/'build/selfplay-training-pilot-plan-v2.json') or {};contract=plan.get('contract',{});settings=contract.get('training',{})
    training=read(ROOT/'data/selfplay-training-pilot-v1/result.json') or {};benchmark=read(ROOT/'build/selfplay-pilot-benchmark.json') or {}
    sc=committed_counts(ROOT/'data/selfplay-training-pilot-v1/work.sqlite3');bc=committed_counts(ROOT/'data/selfplay-pilot-benchmark-v1/work.sqlite3')
    statuses=sc.get('game_statuses',{});terminal=statuses.get('terminal',0);capped=statuses.get('capped',0);active=statuses.get('active',0)
    card=cards['selfplay-pilot-generation'];card['title']='Fixed-control self-play: terminal versus initialized'
    card['result']=f"{sc.get('initialized_games',0)}/32 initialized slots; {terminal}/32 terminal; {capped} capped; {active} active; {sc.get('sp_decisions',0)} durable decisions."
    card['state']='complete' if sp.get('status')=='complete' and not active else 'running' if sc.get('sp_decisions',0) else 'held'
    card['evidence']+='; NEW display: congklak/selfplay_pilot_live_progress.py'
    card=cards['selfplay-pilot-replay']
    if summary:
        card['result']=f"{summary['admitted_families']}/16 admitted families; {summary['retained_rows']} retained observations / {summary['rawstate_unique']} unique raw states. Training gate: {summary['training_admission']} (minimum {summary['minimum_admitted_families']} families). Skipped {summary['exact_domain_rows_skipped']} at active<=4 and {summary['late5_24_rows_skipped']} at5-24 before the unchanged protection mask; duplicate-cap pruning {summary['pruned_duplicate_rows']}."
        card['state']='complete' if summary['training_admission']=='ready' else 'held'
    card=cards['selfplay-pilot-training'];batch=settings.get('batch_size');steps=settings.get('steps')
    if training:
        card['result']=f"{training['steps_completed']}/{steps} optimizer steps; batch {batch}; {training['draws']} replacement draws; {training['train_rows']} replay observations. All8 tensors/12,872parameters may train; fixed-final checkpoint {training['selected_step']}."
    else:card['result']=f"Declared {steps} steps x batch {batch}; CPU1thread, joint trunk/policy/value from admitted active>=25 self-play only. No teacher rehearsal. Waiting for training or its completed result."
    projection=read(ROOT/'build/selfplay-pilot-benchmark-projection.json') or {};exact=projection.get('cases',[])
    forced=sum(sum(count>0 for count in row['state']['pits'][row['state']['controller']])==1 for row in exact)
    card=cards['selfplay-pilot-exact'];card['result']=f"{bc.get('cases',0)}/32 exact cases committed; {len(exact)-forced} choice-bearing and {forced} forced-move roots. {benchmark.get('scalar_measurements',0)}/256 measurements published."
    if benchmark:card['result']+=' Numeric guard: '+str(benchmark['endgame_guard'])+'.'
    card=cards['selfplay-pilot-strength'];statuses=bc.get('game_statuses',{})
    card['result']=f"{bc.get('games',0)}/96 full game traces committed; {statuses.get('terminal',0)} terminal / {statuses.get('unfinished',0)} unfinished; {bc.get('decisions',0)} durable decisions."
    if benchmark:
        overlap=benchmark['trajectory_replay_overlap'];card['result']+=' Numeric both-baseline CI gates: '+str(benchmark['both_baseline_primary_gates'])+f". Actual replay contamination: {overlap['contaminated_evaluation']}; {overlap['colliding_observations']} overlapping observations."
        card['improvement']='Numeric intervals do not authorize promotion. Actual replay hits make evaluation contaminated/descriptive-only; all cohorts retain ancestral/development exposure. H_v1 stays default.'
    for card in current:card['needs_improvement']=card['state']!='complete'
    if capped:cards['selfplay-pilot-generation']['needs_improvement']=True
    if benchmark and benchmark['trajectory_replay_overlap'].get('contaminated_evaluation'):cards['selfplay-pilot-strength']['needs_improvement']=True
    return {**original,'schema':'selfplay-pilot-live-progress-v1','tools':current+original['tools'][8:],
        'observer_corrections':dict(schema='selfplay-pilot-readonly-display-corrections-v1',frozen_sources_unchanged=True,
            summary_source='selfplay result.summary',generation_counter_basis='Initialized slots and status terminal/capped/active from committed SQLite rows',
            training_batch_source='Frozen training-plan-v2 contract.training.batch_size',training_batch_size=batch,
            imports_training_or_search_producers=False)}


def html_bytes():
    data=(ROOT/'tools/selfplay-pilot-dashboard.html').read_bytes()
    return data.replace(b'congklak.selfplay_pilot_progress',b'congklak.selfplay_pilot_live_progress')


def serve(port=18766):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':data=html_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Self-play pilot live observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True);sub.add_parser('snapshot');sub.add_parser('serve').add_argument('--port',type=int,default=18766)
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    run=sub.add_parser('run');run.add_argument('--tool',default='selfplay-pilot');run.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='serve':serve(args.port)
    elif args.operation=='activity':frozen.save(frozen.FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return frozen.run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)


if __name__=='__main__':raise SystemExit(main())
