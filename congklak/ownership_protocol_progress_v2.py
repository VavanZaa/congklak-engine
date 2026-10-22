"""Read immutable protocol progress; never open an active worker's mutable files."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from .value_run_reader import read
from .valuepilot_interface_progress import current

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'build/ownership-protocol-v1'
LOADED_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def snapshot():
    events=sorted((FOLDER/'events').glob('*.json'))
    if not events: raise ValueError('No prospective protocol progress event')
    state=read(events[-1])
    if state['schema']!='ownership-protocol-progress-v1':raise ValueError('Unknown protocol progress schema')
    production=FOLDER/'production-v2'
    catalog=read(production/'catalog.json') if (production/'catalog.json').is_file() else None
    accepted=read(FOLDER/'acceptance.json') if (FOLDER/'acceptance.json').is_file() else {}
    source_ok=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==LOADED_SHA
    closed=source_ok and accepted.get('passed') is True and current(accepted.get('live_inputs'))
    if closed and accepted.get('progress_event')!=events[-1].relative_to(ROOT).as_posix():closed=False
    invalid=not source_ok or (bool(accepted) and not closed)
    old_events=sorted((ROOT/'build/value-diagnostics-v1/events').glob('*.json'))
    history=read(old_events[-1])['cards']+read(ROOT/'build/value-run-v1/stage-status.json')['cards']
    return dict(schema='ownership-protocol-dashboard-v2',accepted=closed,current_cards=len(state['cards']),
        tools=state['cards']+history,current_work=dict(state='hold' if invalid else 'complete' if closed else state['state'],
            message='HOLD: bound source, evidence or final progress event differs.' if invalid else state['message']),
        progress=dict(committed_units=catalog['committed_units'] if catalog else None,
            physical_commit_files=sum(1 for _ in (production/'commits').glob('*.json')),
            physical_counts_are_provisional=catalog is None,planned_publication_units=195),
        counters=dict(optimizer_steps=0,optimizer_draws=0,new_games=0,new_teacher_labels=0,searches=0,feature_simulations=0),
        definition_only=True,execution_ready=False,training_authorized=False,new_reservations_made=False,
        models_fixed=True,policy_fixed=True,candidate_quarantined=True,strength_interpretation_allowed=False,
        live_suffix_worker_verified=False,confidence_intervals_enabled=False,
        observer_source_current=source_ok,observer_source_sha256=LOADED_SHA,
        next_operation=state['next_operation'],
        scope='Prospective protocol definition and existing-data projection only. Current model/policy fixed. Publication recovery is tested; live shared-suffix handoff, trace instrumentation and new reservations remain pending. Historical arena holds remain; no training or new games are authorized.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(ROOT/'tools/ownership-protocol-dashboard-v2.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers();self.wfile.write(body)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='ownership_protocol_observer_v2_ready',pid=os.getpid(),source_sha256=LOADED_SHA,port=port)),flush=True)
    server.serve_forever()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    sub.add_parser('snapshot');sub.add_parser('serve').add_argument('--port',type=int,default=18766)
    args=parser.parse_args()
    if args.operation=='snapshot':print(json.dumps(snapshot(),sort_keys=True,allow_nan=False),flush=True)
    else:serve(args.port)


if __name__=='__main__':main()
