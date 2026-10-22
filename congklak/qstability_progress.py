"""Read-only within-cell weighting, reference work and recovery dashboard."""
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

FOLDER = ROOT / 'build/qstability-progress'
read = old.read_json


def committed(path):
    if not path.is_file(): return {}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.2)) as db:
            db.execute('BEGIN')
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result = {}
            for name in ('roots','root_setups','cells','steps','simulations','attempts','labels','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels'):
                if name in tables:
                    result[name] = db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                    columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
                    if 'body' in columns and name in ('cells','attempts','h8_attempts','h10_attempts','h12_attempts'):
                        result[name + '_statuses'] = dict(Counter(json.loads(text).get('status','unspecified') for text, in db.execute('SELECT body FROM ' + name)))
            return result
    except sqlite3.OperationalError as exc: return dict(observation_pending=str(exc))


def snapshot():
    baseline=read(ROOT/'build/qstability-baseline.json') or {}
    registry=read(ROOT/'build/qstability-registry.json') or {}
    rule=read(ROOT/'build/qstability-rule.json') or {}
    result=read(ROOT/'build/qstability-result.json') or {}
    accepted=read(ROOT/'build/qstability-acceptance.json') or {}
    preflight=read(ROOT/'build/qstability-audit/preflight.json') or {}
    rule_audit=read(ROOT/'build/qstability-audit/reference.json') or {}
    counts=committed(ROOT/'data/qstability-v1/work.sqlite3')
    selected=len(registry.get('rows',{}))
    cards=[]
    def add(key,title,state,text,evidence,improvement):
        card=old.card('qstability-'+key,title,state,text,evidence,improvement)
        card['needs_improvement']=state!='complete';cards.append(card)
    add('baseline','Protected history and fixed models','complete' if baseline else 'running',
        '5,050 previous accepted files protected; models, policy and active optimizer weights fixed.',
        'build/qstability-baseline.json','No training, self-play, strength games, pruning or promotion.')
    research=ROOT/'docs/QSTABILITY_METRICS.md'
    add('research','Revised rule and prospective design','complete' if research.exists() and (ROOT/'build/qstability-implementation.json').exists() else 'running',
        'Worst legal-action H8/H10 Q shift; same floor/scale and fixed cell totals. Strict Q benefit and regret/missing-mass guards.',
        'docs/QSTABILITY_METRICS.md; docs/QSTABILITY_SCOPE.md','Maximum shift is an uncalibrated stability heuristic; no parameter fit, sweep or H12 input.')
    add('data','Separate eligible metadata panel','complete' if registry else 'running',
        str(selected)+'/16 assessment roots, two per stratum/actor cell; old consumed and protected components excluded.',
        'build/qstability-inventory.json; build/qstability-registry.json; build/qstability-reservations.json',
        'The selected saved roots are newly reserved; historical H8/H10 exposure remains explicit.')
    h8=counts.get('h8_attempts',0);h10=counts.get('h10_attempts',0);l8=counts.get('h8_labels',0);l10=counts.get('h10_labels',0)
    add('references','Fresh H8/H10 measurements','complete' if selected==16 and h8==h10==16 else 'running' if counts else 'held',
        str(h8)+'/16 H8 attempts, '+str(h10)+'/16 H10 attempts; complete '+str(l8)+'/'+str(l10)+'.',
        'data/qstability-v1/work.sqlite3','Fixed10M nodes per requested horizon; caps block H12, with no lower-depth substitutes.')
    if h8==h10==16 and (l8<16 or l10<16):cards[-1]['needs_improvement']=True
    add('rule','Immutable within-cell weighting projection','complete' if rule and rule_audit.get('passed') else 'running' if rule else 'held',
        'Each of eight cells keeps1/8 mass; strata25%, actors50%, ordinary50%. Independent rule audit precedes H12.',
        'build/qstability-rule.json; build/qstability-audit/reference.json','Revised worst-action rule; realized endpoint-linked probabilities freeze before any H12.')
    add('recovery','Independent interruption preflight','complete' if preflight.get('passed') else 'running',
        str(preflight.get('tests_run',0))+'/3 methods; actual pre/post teacher-commit kills and deterministic resume.',
        'build/qstability-audit/preflight.json','Validate exact committed prefixes, phase identities and missing/export refusal; serialize writer APIs.')
    attempts=counts.get('h12_attempts',0);labels=counts.get('h12_labels',0)
    add('h12','H12 decision-quality reference','complete' if selected==16 and attempts==16 else 'running' if attempts else 'held',
        str(attempts)+'/16 attempts, '+str(labels)+' complete requested references; fixed10M nodes each.',
        'data/qstability-v1/work.sqlite3','Missing endpoints retain original probability mass and hold favorable metrics; finite H12 is not solved truth.')
    if attempts==16 and labels<16:cards[-1]['needs_improvement']=True
    screen=result.get('screen') or {}
    metrics=(result.get('metrics') or {}).get('overall',{})
    def number(value):return 'unavailable' if not isinstance(value,dict) or 'value' not in value else format(value['value'],'.4f')
    u=metrics.get('uniform',{});w=metrics.get('weighted',{})
    text=('Descriptive screen: '+('passed' if screen.get('passes') else 'HOLD')+'; Q drift '+number(u.get('action_Q_drift'))+' -> '+number(w.get('action_Q_drift'))+'; best-action regret '+number(u.get('H10_best_H12_regret_mean'))+' -> '+number(w.get('H10_best_H12_regret_mean'))+'; no weights applied.') if result else 'Regret, drift, cell/actor/ordinary coverage and concentration checked on all16 fixed assessment roots.'
    add('metrics','Q drift, decision regret and coverage','complete' if result else 'held',text,
        'build/qstability-result.json; docs/QSTABILITY_RESULTS.md','Require strict mean-Q benefit, max/root Q and capture guards, tied-best regret and all-cell concentration; no weights applied.')
    if result:cards[-1]['needs_improvement']=not screen.get('passes',False)
    stale=old.current_evidence(ROOT,accepted) if accepted.get('passed') else []
    add('audit','Independent audit and no-work closure','complete' if accepted.get('passed') and not stale else 'stale' if stale else 'running',
        str(accepted.get('tests_run',0))+' current unique accepted checks; prior5,050 inputs preserved.',
        'build/qstability-audit.json; build/qstability-acceptance.json','Source-bound actual checks, rehashed corruption refusal, closed no-work, idempotent seal and backend/listener proof.')
    recent=runs();active=[r for r in recent if r['state']=='running']
    activity=read(FOLDER/'activity.json') or dict(state='running',message='Preparing separate within-cell weighting assessment; no training.')
    current=dict(state='running',message=' | '.join(r['tool']+': '+r['last_event'] for r in active)) if active else activity
    historical=[]
    for name in ('capture-dashboard-final.json','stratum-dashboard-final.json','allocation-dashboard-final.json'):
        historical.extend((read(ROOT/'build'/name) or {}).get('cards',[]))
    return dict(schema='qstability-progress-v1',observed_at=old.utc(),tools=cards+historical,recent_runs=recent,active_tools=active,current_work=current,
        scope='Fixed models/policy; separate16-root within-stratum-and-actor assessment, bounded H8/H10/H12, worst-action stability, no training.')


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
                if self.path == '/': data = (ROOT / 'tools/qstability-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Within-cell weighting observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='qstability'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
