"""Read-only allocation, weighting assessment and recovery dashboard."""
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

FOLDER = ROOT / 'build/allocation-progress'
read = old.read_json


def committed(path):
    if not path.is_file(): return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result = {}
            for name in ('roots','root_setups','cells','steps','simulations','attempts','labels','h12_attempts','h12_labels'):
                if name in tables:
                    result[name] = db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                    columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
                    if 'body' in columns and name in ('cells','attempts','h12_attempts'):
                        result[name + '_statuses'] = dict(Counter(json.loads(text).get('status','unspecified') for text, in db.execute('SELECT body FROM ' + name)))
            return result
    except sqlite3.OperationalError as exc: return dict(observation_pending=str(exc))


def snapshot():
    baseline = read(ROOT / 'build/allocation-baseline.json') or {}
    registry = read(ROOT / 'build/allocation-weighting-registry.json') or {}
    rule = read(ROOT / 'build/allocation-weighting-rule.json') or {}
    trace = read(ROOT / 'build/allocation-trace-result.json') or {}
    weighting = read(ROOT / 'build/allocation-weighting-result.json') or {}
    accepted = read(ROOT / 'build/allocation-acceptance.json') or {}
    preflight = read(ROOT / 'build/allocation-audit/preflight.json') or {}
    trace_audit = read(ROOT / 'build/allocation-audit/trace.json') or {}
    labels_audit = read(ROOT / 'build/allocation-audit/labels.json') or {}
    tc = committed(ROOT / 'data/allocation-trace-v1/work.sqlite3')
    wc = committed(ROOT / 'data/allocation-weighting-v1/work.sqlite3')
    selected = len(registry.get('rows', {}))
    cards = []
    def add(key, title, state, result, evidence, improvement):
        card = old.card('allocation-' + key, title, state, result, evidence, improvement)
        card['needs_improvement'] = state != 'complete'
        cards.append(card)
    add('baseline', 'Protected history and fixed models', 'complete' if baseline else 'running',
        '4,255 prior files protected, including the 4,252 accepted inputs and three closing proofs. Both model files and policy stay fixed.',
        'build/allocation-baseline.json', 'No training, self-play, strength games, pruning or promotion in this study.')
    add('data', 'Separate eligible assessment and frozen rule', 'complete' if registry and rule else 'running',
        str(selected) + ' selected roots; role, family and state exclusions precede new teacher answers. Rule uses H8/H10 only.',
        'build/allocation-weighting-registry.json; build/allocation-weighting-rule.json',
        'Exclude forensic and protected evaluation data. Report coverage shortfalls and historical exposure; no quota filler.')
    add('recovery', 'Independent interruption preflight', 'complete' if preflight.get('passed') else 'running',
        str(preflight.get('tests_run', 0)) + '/3 methods; four real process kills around trace-simulation and teacher-attempt commits.',
        'build/allocation-audit/preflight.json', 'Committed records survive; replay uses saved evidence. Uncommitted work alone may recompute.')
    add('trace', 'Pair 7 selection and visit allocation', 'complete' if trace.get('status') == 'complete' else 'running' if tc else 'held',
        str(tc.get('steps', 0)) + '/3,840 simulation commits; ' + str(tc.get('cells', 0)) + '/8 cells. Both models at 128/256/512/1024, with identical original settings.',
        'build/allocation-trace-result.json; data/allocation-trace-v1/work.sqlite3',
        'Preserve score components, parent-bootstrap FPU, controller, visits and backup values. No search retuning or strength inference.')
    add('trace-audit', 'Independent trace and original-result parity', 'complete' if trace_audit.get('passed') else 'running' if trace else 'held',
        'Recorded selection/backups and all eight old semantic reports independently checked.' if trace_audit.get('passed') else 'New assessment labeling waits for the completed trace witness and independent method 4.',
        'build/allocation-audit/trace.json', 'Use literal seed transitions and independent selection calculations; preserve ties and bonus actors.')
    add('references', 'New H12 reference on selected eligible roots', 'complete' if weighting.get('status') == 'complete' else 'running' if wc else 'held',
        str(wc.get('h12_attempts', wc.get('attempts', 0))) + '/' + str(selected) + ' attempts committed, ' + str(wc.get('h12_labels', wc.get('labels', 0))) + ' complete requested references. Ten million nodes per H12; no replacements.',
        'data/allocation-weighting-v1/work.sqlite3; build/allocation-weighting-result.json',
        'H12 is a finite sensitivity reference. Caps remain missing; H12 never changes weights or selected IDs.')
    assessment = weighting.get('metrics', {}).get('assessment', {})
    def display(value): return 'missing' if not value else format(value['value'], '.4f')
    assessment_text = ('Descriptive screen: ' + ('passed' if weighting.get('screen', {}).get('passes') else 'not passed') +
        '; complete ' + str(assessment.get('complete_roots', 0)) + '/8; uniform root drift ' + display(assessment.get('uniform', {}).get('root_drift')) +
        ', weighted ' + display(assessment.get('weighted', {}).get('root_drift')) + ' seeds; ESS ' + display(assessment.get('ESS')) + '. No weights applied.') if weighting else \
        'Compare the same complete support; retain uniform metrics, missing mass, effective support and actor/stratum coverage.'
    add('weighting', 'Uniform versus frozen weighting assessment', 'complete' if weighting.get('passed') else 'running' if wc else 'held',
        assessment_text, 'build/allocation-weighting-result.json; docs/ALLOCATION_RESULTS.md',
        'Lower weighted drift measures concentration toward stability. It does not prove calibrated uncertainty or better trained predictions.')
    stale = old.current_evidence(ROOT, accepted) if accepted.get('passed') else []
    add('audit', 'Final audit and completed no-work reopening', 'complete' if accepted.get('passed') and not stale else 'stale' if stale else 'running',
        str(accepted.get('tests_run', 0)) + ' current checks; ' + str(labels_audit.get('tests_run', 0)) + '/2 label and assessment methods.',
        'build/allocation-audit.json; build/allocation-acceptance.json',
        'Verify source-bound checks, rehashed corruption refusals, zero-work completed API/CLI and preservation of all 4,255 old inputs.')
    if weighting:
        cards[6]['needs_improvement'] = True
        if weighting.get('capped_roots', 0): cards[5]['needs_improvement'] = True
    recent = runs(); active = [r for r in recent if r['state'] == 'running']
    activity = read(FOLDER / 'activity.json') or dict(state='running', message='Preparing allocation and separate target-weighting diagnostics; training held.')
    current = dict(state='running', message=' | '.join(r['tool'] + ': ' + r['last_event'] for r in active)) if active else activity
    historical = []
    for name in ('divergence-dashboard-final.json', 'ordinary-value-review-dashboard-final.json'):
        historical.extend((read(ROOT / 'build' / name) or {}).get('cards', []))
    return dict(schema='allocation-progress-v1', observed_at=old.utc(), tools=cards + historical,
                recent_runs=recent, active_tools=active, current_work=current,
                scope='Fixed models/policy; pair 7 simulation traces and a separate frozen H8/H10 weighting rule assessed with new H12. No training.')


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
                if self.path == '/': data = (ROOT / 'tools/allocation-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Allocation and weighting observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='allocation'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
