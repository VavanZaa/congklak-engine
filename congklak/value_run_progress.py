"""Read-only live observer for the fixed value pilot and its acceptance gates."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from .valuepilot_interface_progress import current, read

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/value-run-v1'
LOADED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def optional(path):
    return read(path) if path.is_file() else {}


def snapshot():
    from . import value_training as training
    state = read(FOLDER / 'stage-status.json')
    if state.get('schema') != 'value-run-stage-status-v1' or len(state['cards']) != 9:
        raise ValueError('Unknown value-run stage status')
    source_current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == LOADED_SHA
    accepted = optional(FOLDER / 'acceptance.json')
    closed = source_current and accepted.get('passed') is True and current(accepted.get('live_inputs'))
    production_error = None
    try:
        production = training.status()
    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as exc:
        production = dict(state='hold', steps=None, draws=None, target_steps=400)
        production_error = str(exc)
    cases = optional(FOLDER / 'cases/progress.json')
    arena = optional(ROOT / 'data/value-run-v1/arena/progress.json')
    cards = []
    invalid = not source_current or production_error is not None or (bool(accepted) and not closed)
    for original in state['cards']:
        row = dict(original)
        if row.get('inputs') and not current(row['inputs']):
            row.update(state='hold', result='Bound source or accepted evidence changed.')
            invalid = True
        if row['id'] == 'training':
            row['result'] = f"Committed {production['steps']} / 400 steps; {production['draws']} / 51,200 draws. Only final step 400 is evaluated."
            row['state'] = 'hold' if production_error else 'complete' if production['state'] == 'complete' else row['state']
        row.pop('inputs', None)
        row['needs_improvement'] = row['state'] not in ('complete', 'planned')
        cards.append(row)
    history = read(ROOT / 'build/value-training-v1/stage-status.json')['cards']
    message = ('HOLD: source or required evidence differs. ' + (production_error or 'Revalidate before continuing.')) if invalid else accepted.get('message', state['message'])
    return dict(schema='value-run-progress-v1', current_cards=9, tools=cards + history,
        accepted=closed and not invalid, evidence_accepted=closed and not invalid,
        pilot_promising=accepted.get('pilot_promising', False) if closed and not invalid else False,
        current_work=dict(state='hold' if invalid else 'complete' if closed else state['state'], message=message),
        production=production, case_progress=cases, arena_progress=arena,
        engineering_counters=state.get('engineering_counters', {}),
        active_tools=[] if closed else state.get('active_tools', []),
        observer_source_current=source_current, observer_source_sha256=LOADED_SHA,
        next_resumable_operation=accepted.get('next_operation', state.get('next_operation')),
        scope='One fixed 400-step CPU FP32 value-only treatment. Policy and all original tensors remain fixed. H_v1 stays the default. Fit-check has historical ancestry; ordinary arena starts are synthetic. Negative gates and trajectory overlaps are retained. Live progress is derived; final acceptance checks all bound evidence.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/':
                    body = (ROOT / 'tools/value-run-dashboard.html').read_bytes()
                    kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status':
                    body = json.dumps(snapshot(), sort_keys=True, allow_nan=False).encode()
                    kind = 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(503, str(exc))

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(json.dumps(dict(event='value_run_observer_ready', pid=os.getpid(),
        source_sha256=LOADED_SHA, port=port), sort_keys=True), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot')
    sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    args = parser.parse_args()
    if args.operation == 'snapshot':
        print(json.dumps(snapshot(), sort_keys=True, allow_nan=False), flush=True)
    else:
        serve(args.port)


if __name__ == '__main__':
    main()
