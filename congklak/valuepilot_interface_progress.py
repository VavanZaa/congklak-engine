"""Read-only observer for pre-training data and value-interface implementation."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/valuepilot-interface-v1'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def current(inputs):
    if not isinstance(inputs, dict) or not inputs:
        return False
    for name, checksum in inputs.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            return False
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != checksum:
                return False
    return True


def snapshot():
    status = read(FOLDER / 'stage-status.json')
    if status.get('schema') != 'valuepilot-interface-stage-status-v1':
        raise ValueError('Unknown implementation status')
    counters = status['execution_counters']
    if counters != dict(training_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0):
        raise ValueError('Pre-training observer cannot certify training or game execution')
    accepted_path = FOLDER / 'acceptance.json'
    accepted = read(accepted_path) if accepted_path.exists() else {}
    source_current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == LOADED_SOURCE_SHA256
    closed = source_current and accepted.get('passed') is True and current(accepted.get('new_inputs'))
    cards = []
    for row in status['cards']:
        row = dict(row)
        if row['state'] == 'complete' and row.get('inputs') and not current(row['inputs']):
            row.update(state='hold', result='Bound evidence changed or is missing; this stage must be checked again.')
        row.pop('inputs', None)
        row['needs_improvement'] = row['state'] not in ('complete', 'planned')
        cards.append(row)
    if len(cards) != 9:
        raise ValueError('Expected nine pilot stages')
    invalid = not source_current or (accepted_path.exists() and not closed)
    if not source_current:
        message = 'HOLD: the dashboard source changed after this observer started. Restart the observer to load the current code.'
    elif invalid:
        message = 'HOLD: the accepted implementation evidence changed. Training remains off.'
    elif closed:
        message = accepted['next_step']
    else:
        message = status['message']
    history = read(ROOT / 'build/neurodiag-dashboard-final.json')['cards']
    return dict(schema='valuepilot-interface-progress-v1', tools=cards + history,
                observer_source_sha256=LOADED_SOURCE_SHA256, observer_source_current=source_current,
                current_work=dict(state='hold' if invalid or (closed and not accepted.get('training_ready')) else 'complete' if closed else status['state'], message=message),
                execution_counters=counters, implementation_accepted=closed,
                pilot_training_ready=closed and accepted.get('training_ready') is True,
                readiness_blockers=accepted.get('readiness_blockers', status.get('readiness_blockers', [])),
                committed=status.get('committed', {}), active_tools=[] if closed else status.get('active_tools', []),
                scope='Pre-training implementation only. Synthetic inference, fixed-work conformance and publication recovery are engineering checks. The 400-step run and 192-game both-baseline arena remain planned. H_v1 remains the default.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/':
                    body = (ROOT / 'tools/valuepilot-interface-dashboard.html').read_bytes()
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

    print(json.dumps(dict(event='valuepilot_interface_observer_ready', pid=os.getpid(),
                          source_sha256=LOADED_SOURCE_SHA256, port=port), sort_keys=True), flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


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
