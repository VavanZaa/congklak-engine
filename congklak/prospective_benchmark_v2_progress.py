"""Compact live progress for the protected, fixed-model benchmark."""
import argparse
import datetime
import json
import os
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import config, neurodiag_common as common
from . import guarded_value_pilot_v2_progress as prior

ROOT = common.ROOT
FOLDER = ROOT / 'build/prospective-benchmark-v2'
DATA = ROOT / 'data/prospective-benchmark-v2'
STATUS = FOLDER / 'status.json'
ENTRY = ROOT / 'tools/progress-prospective-benchmark-v2.py'
LOADED = common.pins([__file__, ENTRY, ROOT/'congklak/config.py', ROOT/'congklak/neurodiag_common.py',
                      ROOT/'congklak/guarded_value_pilot_v2_progress.py'])
HTML = prior.HTML.replace('<title>Congklak value pilot</title>', '<title>Congklak protected benchmark</title>').replace(
    'Congklak · guarded value pilot', 'Congklak · protected benchmark').replace(
    '400 value-only training steps. Policy, shared trunk and baseline models stay fixed. Evaluation uses the screened development reservations.',
    'Fixed models and policy. Certified finite high-state closure. Strict 25 ms timing, both baselines, and actual interruption recovery.').replace(
    'Development results remain conditional on inherited model history. Acquired evaluation origins remain excluded from optimization. Live progress does not promote a model.',
    'Synthetic boundary positions test a narrow domain. Protected historical and fit data stay excluded. No model promotion or natural-play strength claim.')


def publish(state, message, next_operation='', tools=None, errors=None):
    FOLDER.mkdir(parents=True, exist_ok=True)
    body = dict(schema='prospective-benchmark-v2-progress', state=state, message=message,
                next_operation=next_operation, tools=tools or [], errors=errors or [],
                policy_fixed=True, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    pending = STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:
        stream.write((config.canonical(body)+'\n').encode()); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, STATUS)
    return body


def small(path, errors):
    path = Path(path)
    if not path.exists():
        return None
    try:
        common.require(path.stat().st_size <= 4*1024**2, 'Compact observer input too large')
        body = common.read(path)
        if type(body) is dict and set(body) == {'body', 'sha256'}:
            common.require(config.digest(body['body']) == body['sha256'], 'Compact envelope changed')
            body = body['body']
        common.require(type(body) is dict, 'Compact input must be an object')
        return body
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(path.name+': '+str(error))
        return None


def brief(body):
    if body is None:
        return None
    return {key:body[key] for key in ('state', 'passed', 'completed_units', 'required_units', 'queries',
        'games', 'terminal_games', 'critical_overshoots', 'ownership_passed', 'hold', 'actual_process_kills',
        'promotion', 'decision', 'summary', 'engineering_execution_complete', 'boundary_roots',
        'broader_roots', 'allowlist_keys') if key in body}


def resources(errors):
    folder = DATA/'resources'
    identity, head = small(folder/'identity.json', errors), small(folder/'head.json', errors)
    if identity is None or head is None:
        return dict(state='not_started', charged_seconds=11254.65, carried_seconds=11254.65,
                    remaining_seconds=6745.35, limit_seconds=18000, budget_reset=False, pending=None)
    try:
        charged, pending, previous, last = identity['inherited_charged_ns'], None, None, None
        common.require(type(head['count']) is int and 0 <= head['count'] <= 200, 'Bad resource cursor')
        for index in range(head['count']):
            event = small(folder/'events'/f'{index:06d}.json', errors)
            common.require(event['index'] == index and event['previous'] == previous, 'Resource prefix changed')
            if event['kind'] == 'reserve':
                common.require(pending is None, 'Resource leases overlap')
                pending = event['lease']; charged += pending['reserved_ns']
            else:
                common.require(event['kind'] == 'finish' and pending and event['token'] == pending['token'],
                               'Resource settlement changed')
                charged += event['charged_ns']-pending['reserved_ns']; pending = None; last = event
            previous = config.digest(event)
        common.require(previous == head['last_sha256'], 'Resource highwater changed')
        limit = identity['protocol']['cumulative_ns']
        return dict(state='held' if (folder/'hold.json').exists() else 'active' if pending else 'idle',
            charged_seconds=round(charged/1e9, 3), carried_seconds=identity['inherited_charged_ns']/1e9,
            remaining_seconds=round((limit-charged)/1e9, 3), limit_seconds=limit/1e9, budget_reset=False,
            pending=None if not pending else {k:pending[k] for k in ('token', 'worker_output', 'reserved_ns')},
            last_finish=None if last is None else {k:last[k] for k in ('clean', 'reason', 'elapsed_ns', 'peak_private_commit_bytes')})
    except (KeyError, ValueError, TypeError) as error:
        errors.append('Resources: '+str(error))
        return dict(state='requires verification')


def job(name, errors):
    folder = FOLDER/'jobs'/name
    report = small(folder/'report.json', errors)
    queries = len(list((folder/'archives').glob('*.json')))
    state = (report or {}).get('state', 'running' if queries else 'not_started')
    if state == 'held':
        errors.append(name+': '+str(report.get('hold')))
    return dict(name={'engineering':'Timing and native conformance', 'endgame':'Shared endgame guard',
                      'arena':'Both-baseline paired games'}[name], state=state,
        completed=(report or {}).get('completed_units', 0), total=(report or {}).get('required_units', 0),
        captured_queries=queries, report=brief(report))


def snapshot():
    errors = []
    status = small(STATUS, errors) or dict(state='building', message='Building the protected fixed-model benchmark.')
    errors.extend(status.get('errors', []))
    cohort = small(FOLDER/'cohort/report.json', errors)
    reservations = small(FOLDER/'reservations/report.json', errors)
    review = small(FOLDER/'audit/preflight.json', errors)
    recovery = small(FOLDER/'recovery/result.json', errors)
    audit = small(FOLDER/'audit/audit.json', errors)
    result = small(FOLDER/'report.json', errors)
    tools = [dict(name='Frozen models and policy', state='complete', new_optimizer_steps=0,
                  candidate='Final step400', baselines=['H_v1', 'H8 relay CPU']),
        dict(name='Certified prospective cohort', state='complete' if cohort and cohort.get('passed') else 'building', report=brief(cohort)),
        dict(name='Default-deny protected reservations', state='complete' if reservations and reservations.get('passed') else 'building', report=brief(reservations)),
        dict(name='Independent preflight', state='passed' if review and review.get('passed') else 'not_started'),
        job('engineering', errors), job('endgame', errors), job('arena', errors),
        dict(name='Actual interruption recovery', state='passed' if recovery and recovery.get('passed') else 'not_started',
             completed=(recovery or {}).get('actual_process_kills', 0), total=2, report=brief(recovery)),
        dict(name='Independent final audit', state='passed' if audit and audit.get('passed') else 'not_started', report=brief(audit))]
    tools.extend(status.get('tools', []))
    resource = resources(errors)
    current = all(common.sha(ROOT/name) == digest for name,digest in LOADED.items())
    if not current:
        errors.append('Observer source changed; restart required.')
    held = resource.get('state') == 'held' or any(t.get('state') == 'held' for t in tools)
    return dict(schema='prospective-benchmark-v2-dashboard',
        current_work=dict(state='held' if held else status['state'], message=status['message']),
        tools=tools, errors=list(dict.fromkeys(errors)), resources=resource,
        updated_at=status.get('updated_at'), next_operation=status.get('next_operation', ''),
        results=brief(result), acceptance_recorded=(FOLDER/'acceptance.json').exists(),
        observer_source_current=current, observer_inputs=LOADED, policy_fixed=True, model_promotion=False,
        training=dict(steps=400, policy_fixed=True, new_steps=0), historically_unseen=False,
        integrity_mode='Compact live counters only; complete traces and literal closure are independently audited.')


def serve(port=18766):
    common.require(port == 18766, 'Registered observer port is18766')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                raw, kind = HTML.encode(), 'text/html; charset=utf-8'
            elif self.path == '/api/status':
                raw, kind = config.canonical(snapshot()).encode(), 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff'); self.end_headers(); self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(config.canonical(dict(event='prospective_benchmark_v2_observer_ready', pid=os.getpid(),
                                port=port, loaded_inputs=LOADED)), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('serve', 'snapshot')); parser.add_argument('--port', type=int, default=18766)
    args = parser.parse_args()
    if args.operation == 'serve':
        serve(args.port)
    else:
        print(config.canonical(snapshot()))


if __name__ == '__main__':
    main()
