"""Read-only fixed-simulation, full-action-Q and recovery dashboard."""
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

FOLDER = ROOT / 'build/divergence-progress'
read = old.read_json


def committed(path):
    if not path.is_file(): return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result = {}
            for name in ('roots','cells','attempts','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels'):
                if name in tables:
                    result[name] = db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                    columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
                    if 'body' in columns:
                        result[name + '_statuses'] = dict(Counter(json.loads(text).get('status','unspecified') for text, in db.execute('SELECT body FROM ' + name)))
            return result
    except sqlite3.OperationalError as exc: return dict(observation_pending=str(exc))


def snapshot():
    baseline = read(ROOT / 'build/divergence-baseline.json') or {}
    panel = read(ROOT / 'build/divergence-panel.json') or {}
    search = read(ROOT / 'build/divergence-search-result.json') or {}
    teacher = read(ROOT / 'build/divergence-teacher-result.json') or {}
    comparison = read(ROOT / 'build/divergence-comparison.json') or {}
    accepted = read(ROOT / 'build/divergence-acceptance.json') or {}
    preflight = read(ROOT / 'build/divergence-audit/preflight.json') or {}
    checked_search = read(ROOT / 'build/divergence-audit/search.json') or {}
    checked_labels = read(ROOT / 'build/divergence-audit/labels.json') or {}
    sc = committed(ROOT / 'data/divergence-search-v1/work.sqlite3')
    tc = committed(ROOT / 'data/divergence-teacher-v1/work.sqlite3')
    cards = []
    def add(key, title, state, text, evidence, improvement):
        card = old.card('divergence-' + key, title, state, text, evidence, improvement)
        card['needs_improvement'] = state != 'complete'
        cards.append(card)
    add('panel', 'Exactly five saved H_v1 divergence roots', 'complete' if panel and baseline else 'running',
        str(len(panel.get('roots',[]))) + '/5 roots;3914 prior files pinned; three adverse-margin cases and two unchanged controls. All five actors are player1.',
        'build/divergence-panel.json; build/divergence-baseline.json',
        'Consumed diagnostic states preserve raw stores/orientation. No population, symmetry or unseen-release claim.')
    changed = [name for name,digest in panel.get('models',{}).items() if not (ROOT/name).is_file() or old.sha(ROOT/name) != digest]
    add('models', 'Fixed control, value candidate and policy', 'complete' if panel and not changed else 'stale' if changed else 'held',
        'Both fixed model files match the panel identities. No weights or training settings change.' if panel and not changed else 'Waiting for fixed model identity or inspecting changed files: ' + str(changed),
        'data/quality-train-v1/model.json; data/ordinary-value-pilot-v1/model.json',
        'Only the already-trained value heads differ between models. No training, self-play, game arena or automatic promotion.')
    add('recovery', 'Independent preflight and real interruption recovery', 'complete' if preflight.get('passed') else 'running',
        str(preflight.get('tests_run',0)) + '/3 preflight methods; four actual pre/postcommit kills required across search and teacher fixtures.',
        'build/divergence-audit/preflight.json',
        'Use already-consumed fixtures only. Physical committed prefixes survive; uncommitted cells or attempts can recompute.')
    add('search', 'Both models at128/256/512/1024 simulations', 'complete' if search.get('status') == 'complete' else 'running' if sc.get('cells',0) else 'held',
        str(sc.get('cells',0)) + '/40 production cells committed. Requested total19200 root simulations; no clocks, noise or cache. Completed root visits must equal the requested budget.',
        'build/divergence-search-result.json; data/divergence-search-v1/work.sqlite3',
        'Each cell uses a fresh tree. Capped/cancelled work remains incomplete; unvisited action Q stays missing. Fixed work is not new strength evidence.')
    add('search-audit', 'Independent deterministic search verification', 'complete' if checked_search.get('passed') else 'running' if search else 'held',
        'All40 cells independently checked; teacher may proceed under its frozen caps.' if checked_search.get('passed') else 'Teacher production waits for the closed40-cell witness and independent method04.',
        'build/divergence-audit/search.json',
        'Replay selection/backups and work using frozen scalar leaf evaluation; verify fixed-policy parity separately.')
    parts = ['H' + str(depth) + ': ' + str(tc.get('h'+str(depth)+'_attempts',0)) + '/5 attempts, ' + str(tc.get('h'+str(depth)+'_labels',0)) + ' complete Q labels' for depth in (8,10,12)]
    add('teacher', 'All legal action Q at requested H8/H10/H12', 'complete' if teacher.get('status') == 'complete' else 'running' if tc.get('h8_attempts',0) else 'held',
        '; '.join(parts) + '. Fixed85M requested-node ceiling; capped histories never substitute for requested labels.',
        'build/divergence-teacher-result.json; data/divergence-teacher-v1/work.sqlite3',
        'Finite H_v1-leaf references, not game truth. H10 is primary; H8/H12 remain separate, with all5 roots and missing counts visible.')
    add('comparison', 'Action ranking, actor-signed regret and next gate', 'complete' if comparison.get('passed') else 'running' if teacher else 'held',
        comparison.get('summary','Waiting for search and finite-horizon action-Q evidence; every budget/horizon remains reported.'),
        'build/divergence-comparison.json; docs/DIVERGENCE_STUDY_RESULTS.md',
        'Keep sampled PUCT Q distinct from teacher Q. No inferred full-game blunder, target ingestion, revised old HOLD or automatic training.')
    stale = old.current_evidence(ROOT, accepted) if accepted.get('passed') else []
    add('audit', 'Final audit, no-work reopen and protected history', 'complete' if accepted.get('passed') and not stale else 'stale' if stale else 'running',
        str(accepted.get('tests_run',0)) + ' current source-bound checks; all3914 prior inputs preserved.' if accepted.get('passed') and not stale else
        'Independent labels/comparison checks ' + str(checked_labels.get('tests_run',0)) + '/2; final corruption and no-work API/CLI validation pending.',
        'build/divergence-audit.json; build/divergence-acceptance.json',
        'Count eight unique independent methods by staged source-bound groups. Retain actual failed attempts separately and keep all model weights fixed.')
    if comparison: cards[6]['needs_improvement'] = True
    if teacher and any(tc.get('h'+str(depth)+'_labels',0) < 5 for depth in (8,10,12)):
        cards[5]['needs_improvement'] = True
    recent = runs(); active = [row for row in recent if row['state']=='running']
    activity = read(FOLDER/'activity.json') or dict(state='running', message='Building fixed-work diagnostics and full-action teacher labels; training held.')
    current = dict(state='running',message=' | '.join(row['tool']+': '+row['last_event'] for row in active)) if active else activity
    historical = read(ROOT/'build/ordinary-value-review-dashboard-final.json') or {}
    return dict(schema='divergence-progress-v1',observed_at=old.utc(),tools=cards+historical.get('cards',[]),
                recent_runs=recent,active_tools=active,current_work=current,
                scope='Five consumed states, fixed models/policy, all40 fixed-simulation cells then15 bounded teacher attempts. No training or new games.')


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
                if self.path == '/': data = (ROOT / 'tools/divergence-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Divergence search and Q observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='divergence'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
