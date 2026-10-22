"""Read-only observer for bounded value-training implementation acceptance."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from .valuepilot_interface_progress import current, read

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/value-training-v1'
LOADED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ZERO = dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0)


def snapshot():
    state = read(FOLDER / 'stage-status.json')
    if state.get('schema') != 'value-training-implementation-status-v1' or state['production_counters'] != ZERO:
        raise ValueError('Unknown implementation status or unexpected production work')
    if len(state['cards']) != 10:
        raise ValueError('Expected ten current stages')
    source_current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == LOADED_SHA
    path = FOLDER / 'acceptance.json'
    accepted = read(path) if path.exists() else {}
    closed = source_current and accepted.get('passed') is True and current(accepted.get('live_inputs'))
    cards = []
    changed = False
    for original in state['cards']:
        row = dict(original)
        if row.get('inputs') and not current(row['inputs']):
            row.update(state='hold', result='Bound evidence changed or is missing.')
            changed = True
        row.pop('inputs', None)
        row['needs_improvement'] = row['state'] not in ('complete', 'planned')
        cards.append(row)
    invalid = not source_current or changed or (path.exists() and not closed)
    # This observer certifies implementation only. Production state must never be
    # silently hidden under the accepted zero-step implementation snapshot.
    if (ROOT / 'data/value-training-v1/training').exists():
        invalid = True
        message = 'Production training has been initialized. This implementation snapshot is historical; publish a training-run observer before reporting run progress.'
    elif invalid:
        message = 'HOLD: reviewed source or evidence changed. Revalidate before training.'
    else:
        message = accepted.get('next_step', state['message'])
    history = read(ROOT / 'build/ordinary-development-v1/stage-status.json')['cards']
    return dict(schema='value-training-implementation-progress-v1', tools=cards + history,
        current_cards=len(cards), accepted=closed and not invalid,
        implementation_ready=closed and not invalid,
        current_work=dict(state='hold' if invalid else 'complete' if closed else state['state'], message=message),
        production_counters=state['production_counters'], execution_counters=state['production_counters'],
        engineering_counters=state.get('engineering_counters', {}), committed=state.get('committed', {}),
        observer_source_sha256=LOADED_SHA, observer_source_current=source_current,
        active_tools=[] if closed else state.get('active_tools', []),
        next_resumable_operation=accepted.get('next_operation', state.get('next_operation')),
        scope='Implementation acceptance and bounded engineering fixtures. Production remains 0/400 steps and 0/51,200 draws. Fixture optimization is counted separately. Fit, endgame and equal-time strength gates follow the production run. H_v1 remains the default. Live checks cover sources and certificates; full verify checks every bound artifact.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/':
                    body = (ROOT / 'tools/value-training-dashboard.html').read_bytes()
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
    print(json.dumps(dict(event='value_training_observer_ready', pid=os.getpid(),
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
