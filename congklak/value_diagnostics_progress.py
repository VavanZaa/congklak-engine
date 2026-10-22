"""Immutable-event observer for fixed-model existing-data diagnostics."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from .value_run_reader import read
from .valuepilot_interface_progress import current

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'build/value-diagnostics-v1'
LOADED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def snapshot():
    events = sorted((FOLDER / 'events').glob('*.json'))
    if not events: raise ValueError('No diagnostic progress event')
    state = read(events[-1])
    if state['schema'] != 'value-diagnostics-progress-v1': raise ValueError('Unknown diagnostic progress schema')
    production = FOLDER / 'production'
    # No mutable worker head/progress/ledger opens on this Windows volume.
    physical = sum(1 for _ in (production / 'commits').glob('*.json'))
    catalog = read(production / 'catalog.json') if (production / 'catalog.json').is_file() else None
    acceptance_path = FOLDER / 'acceptance.json'
    acceptance = read(acceptance_path) if acceptance_path.is_file() else {}
    source_ok = hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == LOADED_SHA
    accepted = source_ok and bool(acceptance) and acceptance.get('passed') is True and current(acceptance.get('live_inputs'))
    invalid = not source_ok or (bool(acceptance) and not accepted)
    message = 'HOLD: bound diagnostic source or accepted evidence changed.' if invalid else state['message']
    history = read(ROOT / 'build/value-run-v1/stage-status.json')['cards']
    return dict(schema='value-diagnostics-dashboard-v1', accepted=accepted,
        current_cards=len(state['cards']), tools=state['cards'] + history,
        current_work=dict(state='hold' if invalid else 'complete' if accepted else state['state'], message=message),
        progress=dict(committed_units=catalog['committed_units'] if catalog else None,
            physical_commit_files=physical, physical_counts_are_provisional=catalog is None),
        counters=dict(optimizer_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0, searches=0, feature_simulations=0),
        candidate_quarantined=True, policy_fixed=True, models_fixed=True, strength_interpretation_allowed=False,
        observer_source_current=source_ok, observer_source_sha256=LOADED_SHA,
        next_operation=state['next_operation'],
        scope='Existing-data diagnosis only. Historical pilot: 400 updates, 51,200 draws and 192 games. Current diagnostics perform zero updates, new games or teacher searches. H_v1 remains default; arena strength intervals remain withheld.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/':
                    body = (ROOT / 'tools/value-diagnostics-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status':
                    body = json.dumps(snapshot(), sort_keys=True, allow_nan=False).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(body)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(json.dumps(dict(event='value_diagnostics_observer_ready', pid=os.getpid(), source_sha256=LOADED_SHA, port=port)), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(json.dumps(snapshot(), sort_keys=True, allow_nan=False), flush=True)
    else: serve(args.port)


if __name__ == '__main__': main()
