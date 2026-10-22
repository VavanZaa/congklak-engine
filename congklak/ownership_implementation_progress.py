"""Immutable implementation evidence observer; never reads mutable worker heads."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from . import ownership_implementation_common as stage

ROOT,FOLDER=stage.ROOT,stage.FOLDER
LOADED_SHA=stage.sha(__file__)

def publish(state,message,cards,next_operation):
    files=sorted((FOLDER/'events').glob('*.json'))
    return stage.atomic(FOLDER/'events'/f'{len(files):06d}.json',dict(schema='ownership-implementation-progress-v1',
        state=state,message=message,cards=cards,next_operation=next_operation,models_fixed=True,policy_fixed=True))

def snapshot():
    events=sorted((FOLDER/'events').glob('*.json'));state=stage.read(events[-1])
    accepted=stage.read(FOLDER/'acceptance.json') if (FOLDER/'acceptance.json').exists() else None
    source_ok=stage.sha(__file__)==LOADED_SHA
    closed=bool(accepted and accepted['passed'] and source_ok)
    if closed:
        try:stage.common.current(accepted['live_inputs'])
        except (OSError,ValueError):closed=False
        closed=closed and accepted['progress_event']==events[-1].relative_to(ROOT).as_posix()
    counters=dict(stage.TRAINING_ZERO)
    for key,path in [('engineering',FOLDER/'handoff/engineering-fixed/report.json'),
                     ('timed_engineering',FOLDER/'handoff/engineering-timed-v2/report.json')]:
        if (path.parent/'catalog.json').exists():
            report=stage.read(path);counters[key+'_games']=report['engineering_games'];counters[key+'_searches']=report['engineering_searches']
    if accepted:
        counters.update(accepted.get('additional_engineering_counters',{}))
    return dict(schema='ownership-implementation-dashboard-v1',accepted=closed,current_cards=len(state['cards']),tools=state['cards'],
        current_work=dict(state=state['state'] if source_ok else 'hold',message=state['message'] if source_ok else 'HOLD: observer source changed'),
        progress=dict(provenance_commit_files=sum(1 for _ in (FOLDER/'provenance/production/commits').glob('*.json')),
            handoff_commit_files=sum(1 for _ in (FOLDER/'handoff/engineering-fixed/commits').glob('*.json')),
            physical_counts_are_provisional=not closed),counters=counters,definition_only=False,
        execution_ready=False,training_authorized=False,live_suffix_worker_verified=closed,
        confidence_intervals_enabled=False,models_fixed=True,policy_fixed=True,
        observer_source_current=source_ok,observer_source_sha256=LOADED_SHA,next_operation=state['next_operation'],
        scope='Engineering fixtures and metadata reservations only. Historical arena results remain on HOLD. Candidate quarantined; H_v1 default. Training and production arena games require a separate admitted treatment.')

def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(ROOT/'tools/ownership-implementation-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers();self.wfile.write(body)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='ownership_implementation_observer_ready',pid=os.getpid(),source_sha256=LOADED_SHA,port=port)),flush=True)
    server.serve_forever()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('snapshot','serve'));p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='snapshot':print(json.dumps(snapshot(),sort_keys=True,allow_nan=False))
    else:serve(a.port)

if __name__=='__main__':main()
