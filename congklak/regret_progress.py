"""Read-only root178 and capture regression forensic progress."""
import argparse
from collections import Counter
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sqlite3
import subprocess
import time
import uuid
from . import config, progress as old
from .native import ROOT

FOLDER = ROOT / 'build/regret-progress'
read = old.read_json


def committed(path):
    if not path.is_file(): return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result = {}
            for name in ('cases','roots','root_setups','cells','steps','simulations','attempts','labels','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels'):
                if name in tables:
                    result[name] = db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                    columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
                    if 'body' in columns and name in ('cells','attempts','h8_attempts','h10_attempts','h12_attempts'):
                        result[name + '_statuses'] = dict(Counter(json.loads(text).get('status','unspecified') for text, in db.execute('SELECT body FROM ' + name)))
            return result
    except sqlite3.OperationalError as exc: return dict(observation_pending=str(exc))


def snapshot():
    baseline=read(ROOT/'build/regret-baseline.json') or {}
    plan=read(ROOT/'build/regret-drift-plan.json') or {}
    result=read(ROOT/'build/regret-drift-result.json') or {}
    accepted=read(ROOT/'build/regret-acceptance.json') or {}
    preflight=read(ROOT/'build/regret-audit/preflight.json') or {}
    production=read(ROOT/'build/regret-audit/production.json') or {}
    counts=committed(ROOT/'data/regret-drift-v1/work.sqlite3');done=counts.get('cases',0)
    cards=[]
    def add(key,title,state,text,evidence,improvement):
        card=old.card('regret-'+key,title,state,text,evidence,improvement);card['needs_improvement']=state!='complete';cards.append(card)
    add('baseline','Protected history and fixed models','complete' if baseline else 'running',
        '5,252 previous files protected; models, policy and existing sampling/loss settings unchanged.',
        'build/regret-baseline.json','Existing data only: no Native, new labels, neural inference, games or training.')
    add('support','Root178, capture cases and full comparison context','complete' if plan else 'running',
        'Root178 plus all4 consumed capture roots and all16 prior assessment roots; all legal actions and saved horizons.',
        'build/regret-drift-plan.json; docs/REGRET_SCOPE.md','Retrospective evidence; preserve reservations and fixed denominators. No new validation or training eligibility.')
    add('attribution','Exact Q-drift and best-action regret attribution','complete' if result else 'running' if done else 'held',
        'Root178 has 4 seeds of finite-H12 regret; capture mean Q drift rises 1.708333 to 1.719188. The weighting rule remains rejected.',
        'build/regret-drift-result.json; docs/REGRET_RESULTS.md','Keep player-0 and actor signs, best/non-best actions, ranking gaps and cancellation separate; no fitted weights.')
    add('mechanics','Literal saved-line mechanics','complete' if result else 'running' if done else 'held',
        'Replay existing PVs only: capture, relay, bonus, store swings, shared prefixes and finite leaves.',
        'build/regret-drift-result.json; docs/REGRET_FINDINGS.md','H12 is a finite teacher reference. Next: diagnose neural value targets and fit, then test one bounded treatment at equal time.')
    add('recovery','Independent interruption preflight','complete' if preflight.get('passed') else 'running',
        str(preflight.get('tests_run',0))+'/3 methods; actual pre/post case-commit kills and exact prefix resume.',
        'build/regret-audit/preflight.json','Semantic validation may replay arithmetic; completed reopens add no cases or engine/training work.')
    add('cases','Recoverable regret-case reports','complete' if result.get('status')=='complete' else 'running' if done else 'held',
        str(done)+'/5 complete case commits; '+str(production.get('tests_run',0))+'/2 independent production methods.',
        'data/regret-drift-v1/work.sqlite3; build/regret-audit/production.json','Retain source-bound case rows and original inputs; refuse rehashed corruption before export repair.')
    if result:cards[2]['needs_improvement']=True;cards[3]['needs_improvement']=True
    stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else []
    add('audit','Final independent audit and closure','complete' if accepted.get('passed') and not stale else 'stale' if stale else 'running',
        str(accepted.get('tests_run',0))+' current accepted checks; all5,252 prior files preserved.',
        'build/regret-audit.json; build/regret-acceptance.json','Actual source-bound methods, no-work reopen, idempotent seal and backend/listener verification; no training.')
    recent=runs();active=[r for r in recent if r['state']=='running']
    activity=read(FOLDER/'activity.json') or dict(state='running',message='Preparing existing-data regret investigation; no training.')
    current=dict(state='running',message=' | '.join(r['tool']+': '+r['last_event'] for r in active)) if active else activity
    history=[]
    for name in ('qstability-dashboard-final.json','capture-dashboard-final.json'):history.extend((read(ROOT/'build'/name) or {}).get('cards',[]))
    return dict(schema='regret-progress-v1',observed_at=old.utc(),tools=cards+history,recent_runs=recent,active_tools=active,current_work=current,
        scope='Existing consumed data only: root178 and4capture roots/all16 context, literal replay and exact attribution; models/policy fixed, no Native/search/labels/training.')


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    old.atomic_json(path, body)


def event_message(line):
    try: value = json.loads(line)
    except ValueError: return line.strip()[:200]
    if isinstance(value, dict) and isinstance(value.get('event'), str):
        return value['event'][:200]
    return line.strip()[:200]


def runs():
    rows = []
    for path in sorted(FOLDER.glob('run-*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)[:15]:
        row = read(path)
        if row['state'] == 'running' and list(old.process_identity(row['pid'])) != row['process_identity']:
            row = {**row, 'state': 'interrupted_or_exited'}
        rows.append(row)
    return rows


def run_command(command, tool):
    FOLDER.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    path, log = FOLDER / ('run-' + uid + '.json'), FOLDER / ('run-' + uid + '.log')
    started = time.monotonic()
    row = dict(tool=tool, command=command, pid=os.getpid(), process_identity=list(old.process_identity(os.getpid())),
               state='running', started_at=old.utc(), updated_at=old.utc(), last_event='Launching', log=str(log.relative_to(ROOT)))
    save(path, row)
    child = None
    try:
        with log.open('w', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace')
            for line in child.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
                row.update(last_event=event_message(line), updated_at=old.utc()); save(path, row)
            code = child.wait()
        row.update(state='complete' if code == 0 else 'failed', returncode=code,
                   elapsed_seconds=time.monotonic() - started, updated_at=old.utc()); save(path, row)
        return code
    except BaseException:
        if child is not None and child.poll() is None: child.terminate(); child.wait()
        row.update(state='interrupted', elapsed_seconds=time.monotonic() - started, updated_at=old.utc()); save(path, row)
        raise


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/': data = (ROOT / 'tools/regret-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Regret Q-drift observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='regret'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
