"""Read-only live acquisition and reservation status; never constructs an engine."""
import argparse,json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from . import footprint_screen_common as s

HTML=s.ROOT/'tools/footprint-screen-dashboard.html'
LOADED=s.common.pins([__file__,s.__file__,s.ROOT/'congklak/config.py'])
def publish(state,message,tools,next_operation):
    events=sorted((s.FOLDER/'events').glob('*.json'))
    return s.atomic(s.FOLDER/'events'/f'{len(events):06d}.json',dict(schema=s.SCHEMA+'-progress',
        state=state,message=message,tools=tools,next_operation=next_operation,
        policy_fixed=True,models_fixed=True,training_authorized=False))
def _optional(path):return s.read(path) if path.exists() else None
def _brief(body):
    if body is None:return None
    keys=('state','completed_queries','completed_roots','settled_roots','complete_teachers','incomplete_teachers','queries','roots','required_queries','required_roots',
          'teacher_complete_roots','capped_roots','pending_attempts','hold','acquired_roots',
          'metadata_quotas_satisfied','independent_full_raw_audit_passed','reservation_admission',
          'selected_distinct_components','shortage')
    return {k:body[k] for k in keys if k in body}
def snapshot():
    events=sorted((s.FOLDER/'events').glob('*.json'))
    event=s.read(events[-1]) if events else dict(state='preparing',message='Preparing complete teacher-footprint screening.',
        tools=[],next_operation='Freeze and independently review the acquisition recipe.')
    current=all(s.sha(s.ROOT/key)==value for key,value in LOADED.items())
    pool=_optional(s.FOLDER/'pool.json');catalog=_optional(s.FOLDER/'acquisition/catalog.json')
    report=_optional(s.FOLDER/'acquisition/report.json');reservations=_optional(s.FOLDER/'reservations/report.json')
    ledger=_optional(s.DATA/'resources/head.json');acceptance=_optional(s.FOLDER/'acceptance.json')
    valid=False
    if acceptance:
        try:
            from . import footprint_screen_finish as finish
            valid=current and finish.valid_compact(acceptance)
        except (ValueError,OSError,RuntimeError,KeyError):valid=False
    held=(s.FOLDER/'acquisition/hold.json').exists() or bool(report and report.get('state')=='held')
    tools=list(event['tools'])
    if pool:tools.insert(0,dict(name='Frozen acquisition pool',state='declared',roots=len(pool.get('roots',[])),
        selection='Metadata order only; every candidate retained'))
    commits=s.FOLDER/'acquisition/commits'
    if commits.exists():tools.insert(0,dict(name='Live acquisition journal',state='held' if held else (report or {}).get('state','running'),
        durable_records=len(list(commits.glob('*.json'))),report=_brief(report)))
    if reservations:tools.insert(0,dict(name='Rebuilt reservations',state='recorded',report=_brief(reservations)))
    state='held' if held else 'observer restart required' if not current else 'evidence requires verification' if acceptance and not valid else event['state']
    return dict(schema=s.SCHEMA+'-dashboard',accepted=valid,
        current_work=dict(state=state,message=event['message']),
        observer_source_current=current,observer_inputs=LOADED,tools=tools,next_operation=event['next_operation'],
        policy_fixed=True,models_fixed=True,training_authorized=False,production_training_steps=0,
        previous_pilot=dict(state='held',production_steps=0,preserved=True),
        resource_head=ledger,inherited_worker_seconds=737.42,cumulative_worker_limit_seconds=10800,
        scope='Complete-footprint acquisition and development reservations conditional on inherited model background. Historical unknowns remain.',
        integrity_mode='Live checks use compact sources and receipts. Full raw traces are independently replayed before reservation acceptance.',
        decision=acceptance.get('decision') if valid else None)
def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':raw=HTML.read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':raw=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
            except Exception as error:self.send_error(503,str(error))
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='footprint_screen_observer_ready',pid=os.getpid(),port=port,loaded_inputs=LOADED)),flush=True)
    server.serve_forever()
def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('serve','snapshot'));p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='serve':serve(a.port)
    else:print(json.dumps(snapshot(),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
