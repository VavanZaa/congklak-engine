"""Read-only neural target-consistency and value-fit progress."""
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

FOLDER = ROOT / 'build/neurodiag-progress'
read = old.read_json


def committed(path):
    if not path.is_file(): return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result = {}
            for name in ('batches','observations','cases','roots','root_setups','cells','steps','simulations','attempts','labels','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels'):
                if name in tables:
                    result[name] = db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                    columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
                    if 'body' in columns and name in ('cells','attempts','h8_attempts','h10_attempts','h12_attempts'):
                        result[name + '_statuses'] = dict(Counter(json.loads(text).get('status','unspecified') for text, in db.execute('SELECT body FROM ' + name)))
            return result
    except sqlite3.OperationalError as exc: return dict(observation_pending=str(exc))


def snapshot():
    baseline=read(ROOT/'build/neurodiag-baseline.json') or {}
    inventory=read(ROOT/'build/neurodiag-inventory.json') or {}
    panel=read(ROOT/'build/neurodiag-panel.json') or {}
    plan=read(ROOT/'build/neurodiag-plan.json') or {}
    result=read(ROOT/'build/neurodiag-result.json') or {}
    accepted=read(ROOT/'build/neurodiag-acceptance.json') or {}
    preflight=read(ROOT/'build/neurodiag-audit/preflight.json') or {}
    witness=read(ROOT/'build/neurodiag-audit/observation-witness.json') or {}
    closed=read(ROOT/'build/neurodiag-rerun.json') or {}
    counts=committed(ROOT/'data/neurodiag-v1/work.sqlite3')
    batches=counts.get('batches',0); observations=counts.get('observations',0)
    cards=[]
    def add(key,title,state,text,evidence,improvement):
        card=old.card('neurodiag-'+key,title,state,text,evidence,improvement);card['needs_improvement']=state!='complete';cards.append(card)
    add('baseline','Protected history and fixed models','complete' if baseline else 'running',
        '5,377 prior files protected; both model exports, policy and active training weights fixed.',
        'build/neurodiag-baseline.json','Frozen CPU forwards and analytical diagnostics only. No optimizer, teacher search, new games or promotion.')
    add('eligibility','Eligible existing data and fixed panel','complete' if panel and plan else 'running',
        'Metadata-first full family/quotient closure, all origins and accumulated reservation overlays before target bodies.',
        'build/neurodiag-inventory.json; build/neurodiag-panel.json; docs/NEURODIAG_SCOPE.md','Keep protected/sealed/evaluation data excluded; report coverage shortages instead of generating or substituting data.')
    add('targets','Target consistency by objective','complete' if result else 'running' if batches else 'held',
        'Compare saved finite teacher horizons and completed control-policy returns separately where eligible support exists.',
        'build/neurodiag-result.json; docs/NEURODIAG_FINDINGS.md','Finite-horizon estimates, policy-return variability and minimax truth are different quantities; no relabeling or fitted sampling rule.')
    add('forward','Recoverable frozen model observations','complete' if witness else 'running' if batches else 'held',
        str(batches)+' committed batches; '+str(observations)+' observation rows. Scalar conformance and policy identity checked.',
        'data/neurodiag-v1/work.sqlite3; build/neurodiag-audit/observation-witness.json','CPU NumPy with bounded batches and pinned numeric conventions; no CUDA/Torch or model parameter updates.')
    add('fit','Value error, sensitivity and fixed features','complete' if result else 'running' if batches else 'held',
        'Residual normalization, saturation, frozen-head derivatives, hidden-feature rank and interior target-logit projection.',
        'build/neurodiag-result.json; docs/NEURODIAG_METRICS.md','No head coefficients or candidate model are fitted/exported. In-sample feature diagnostics do not establish playing strength.')
    add('recovery','Independent process-kill recovery','complete' if preflight.get('passed') else 'running',
        str(preflight.get('tests_run',0))+'/3 preflight methods; physical pre/post batch-commit kills and exact committed-prefix resume.',
        'build/neurodiag-audit/preflight.json','Only unfinished batches may compute new forwards after interruption; preserve all committed observations.')
    add('cache','Completed cache and no-forward reopening','complete' if closed.get('passed') else 'held',
        'Validate source-bound cached observations and independent witness; rebuild reports without model forwards.',
        'build/neurodiag-rerun.json','Corrupt rows, changed identities and missing required evidence must refuse; valid final exports repair from certified cache only.')
    stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else []
    add('audit','Independent audit and final acceptance','complete' if accepted.get('passed') and not stale else 'stale' if stale else 'running',
        str(accepted.get('tests_run',0))+' accepted current checks; no training or promotion.',
        'build/neurodiag-audit.json; build/neurodiag-acceptance.json','Close source-bound audit, immutable rerun and backend/listener proof before choosing any training treatment.')
    if result: cards[2]['needs_improvement']=True; cards[4]['needs_improvement']=True
    recent=runs();active=[r for r in recent if r['state']=='running']
    activity=read(FOLDER/'activity.json') or dict(state='running',message='Preparing eligible existing-data neural diagnostics; models and policy fixed; no training.')
    current=dict(state='running',message=' | '.join(r['tool']+': '+r['last_event'] for r in active)) if active else activity
    history=[]
    for name in ('regret-dashboard-final.json','qstability-dashboard-final.json'):history.extend((read(ROOT/'build'/name) or {}).get('cards',[]))
    return dict(schema='neurodiag-progress-v1',observed_at=old.utc(),tools=cards+history,recent_runs=recent,active_tools=active,current_work=current,
        scope='Eligible existing data only: target consistency and frozen neural value/feature diagnostics. Models/policy fixed; no training, teacher search, new games, pruning or promotion.')


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
                if self.path == '/': data = (ROOT / 'tools/neurodiag-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Neural diagnostics observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='neurodiag'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
