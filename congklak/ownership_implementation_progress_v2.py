"""Fail-closed lightweight observer over the immutable implementation evidence."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from . import ownership_implementation_progress as previous
from . import ownership_implementation_acceptance as acceptance
from . import ownership_implementation_common as stage
from .native_ownership_trace import MODEL_SHA

ROOT,FOLDER=stage.ROOT,stage.FOLDER
LOADED_SHA=stage.sha(__file__)
HTML='tools/ownership-implementation-dashboard-v2.html'

def valid_closure(body,eventpath):
    """No full historical rehash, writers, native constructors or searches."""
    try:
        need=stage.need
        need(body['schema']==acceptance.SCHEMA and body['passed'] is True and
            body['prior_files_preserved']==59154 and len(body['inputs'])>=59154 and
            body['models_fixed'] is True and body['policy_fixed'] is True and
            body['training_counters']==stage.TRAINING_ZERO and body['execution_ready'] is False and
            body['training_authorized'] is False and body['evaluation_admission'] is False and
            body['conditional_metadata_only'] is True and body['candidate_quarantined'] is True and
            body['default_engine']=='H_v1','Observer acceptance scope differs')
        need(body['independent_audit']=='build/ownership-implementation-v1/audit/audit.json',
            'Observer audit scope differs')
        audit=stage.read(ROOT/body['independent_audit'])
        need(audit['schema']=='ownership-implementation-independent-audit-v1' and audit['passed'] is True and
            audit['independent_of_implementation'] is True and audit['unresolved_findings']==[] and
            audit['completed_reader_no_engine'] is True and all(audit['methods'][g]['passed'] is True and
                audit['methods'][g]['evidence'] for g in acceptance.METHODS) and
            all(audit['recovery_counts'][k]>=v for k,v in dict(provenance_kills=3,worker_kills=5,reservation_kills=3).items()),
            'Observer independent coverage differs')
        event=stage.read(eventpath);rel=eventpath.relative_to(ROOT).as_posix()
        models=stage.common.pins(['data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
            'data/training-stage5-deeper-h8-f127-seed2718/model.json'])
        need(models==dict(zip(('data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
            'data/training-stage5-deeper-h8-f127-seed2718/model.json'),
            (MODEL_SHA['control'],MODEL_SHA['candidate'],MODEL_SHA['H8_relay_CPU']))),'Observer frozen models differ')
        need(body['progress_event']==rel and body['progress_event_sha256']==stage.sha(eventpath) and
            event['acceptance_schema']==acceptance.SCHEMA and
            event['baseline_sha256']==stage.sha(FOLDER/'baseline.json') and event['models_sha256']==models and
            event['training_counters']==stage.TRAINING_ZERO and event['models_fixed'] is True and
            event['policy_fixed'] is True and event['execution_ready'] is False and event['training_authorized'] is False,
            'Observer final progress binding differs')
        required=stage.common.pins([*acceptance.SOURCES,ROOT/body['independent_audit'],eventpath,FOLDER/'baseline.json'])
        required.update(models);required.update(audit['source_inputs'])
        for group in acceptance.METHODS:
            for witness in audit['methods'][group]['evidence']:required[witness['path']]=witness['sha256']
        need(all(body['live_inputs'].get(k)==v==body['inputs'].get(k) for k,v in required.items()) and
            all(body['inputs'].get(k)==v for k,v in body['live_inputs'].items()),'Observer live coverage differs')
        # previous.snapshot already hashes the entire supplied live map. This
        # second gate establishes that the map includes the required evidence.
        return True
    except (KeyError,TypeError,ValueError,OSError,IndexError):return False

def snapshot():
    base=previous.snapshot();source_ok=stage.sha(__file__)==LOADED_SHA and base['observer_source_current']
    path=FOLDER/'acceptance.json';exists=path.exists()
    closed=bool(source_ok and exists and base['accepted'] and
        valid_closure(stage.read(path),sorted((FOLDER/'events').glob('*.json'))[-1]))
    base.update(schema='ownership-implementation-dashboard-v2',accepted=closed,
        live_suffix_worker_verified=closed,observer_source_current=source_ok,
        observer_source_sha256=LOADED_SHA,observer_gate='independent audit, exact required live/source/model/progress bindings')
    if not source_ok or exists and not closed:
        base['current_work']=dict(state='hold',message='HOLD: acceptance scope, required evidence or observer source differs.')
    return base

def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(ROOT/HTML).read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers();self.wfile.write(body)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='ownership_implementation_observer_v2_ready',pid=os.getpid(),source_sha256=LOADED_SHA,port=port)),flush=True)
    server.serve_forever()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('snapshot','serve'));p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='snapshot':print(json.dumps(snapshot(),sort_keys=True,allow_nan=False))
    else:serve(a.port)

if __name__=='__main__':main()
