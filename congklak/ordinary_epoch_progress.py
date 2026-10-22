"""Local read-only observer for the ordinary-only pilot and its retained gates."""
import argparse,json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from . import ordinary_epoch_common as s

HTML='tools/ordinary-epoch-dashboard.html'
LOADED_SHA=s.sha(__file__)
def publish(state,message,tools,next_operation):
    files=sorted((s.FOLDER/'events').glob('*.json'))
    return s.atomic(s.FOLDER/'events'/('%06d.json'%len(files)),dict(schema=s.SCHEMA+'-progress',
        state=state,message=message,tools=tools,next_operation=next_operation,policy_fixed=True))
def snapshot():
    files=sorted((s.FOLDER/'events').glob('*.json'))
    event=s.read(files[-1]) if files else dict(state='preparing',message='Preparing the bounded ordinary-only value pilot.',tools=[],next_operation='Freeze and independently audit the recipe.')
    ap=s.FOLDER/'acceptance.json';a=s.read(ap) if ap.exists() else None
    current=s.sha(__file__)==LOADED_SHA
    if a:
        try:
            from . import ordinary_epoch_results as results
            valid=current and results.valid_closure(a)
        except (OSError,ValueError,RuntimeError):valid=False
    else:valid=False
    training=s.DATA/'training-v2/catalog.json';catalog=s.read(training) if training.exists() else None
    resource=s.DATA/'resources/head.json';head=s.read(resource) if resource.exists() else None
    jobs={}
    held=[]
    for name in ('labels','conformance','guards','arena'):
        folder=s.FOLDER/'jobs'/name
        report=s.read(folder/'report.json') if (folder/'report.json').exists() else None
        jobs[name]=dict(committed_records=len(list((folder/'commits').glob('*.json'))),
            report_present=report is not None,hold_present=bool(report and report.get('state')=='held') or (folder/'hold.json').exists(),
            state=report.get('state') if report else 'not sealed',completed_units=report.get('completed_units') if report else None)
        if jobs[name]['hold_present']:held.append((name,(report or {}).get('hold')))
    tc=(catalog or {}).get('body',catalog or {})
    live_tools=[]
    if catalog:live_tools.append(dict(name='Live training checkpoint',state='complete' if tc.get('complete') else 'in progress',
        steps=str(tc.get('step'))+'/400',draws=tc.get('draws')))
    for name,row in jobs.items():
        if row['committed_records'] or row['report_present']:
            live_tools.append(dict(name='Live '+name,state=row['state'],committed_records=row['committed_records'],
                completed_units=row['completed_units'],hold=row['hold_present']))
    return dict(schema=s.SCHEMA+'-dashboard',accepted=valid,policy_fixed=True,
        current_work=dict(state='hold' if held or not current or a is not None and not valid else event['state'],
            message=('HOLD in '+held[0][0]+': '+str(held[0][1])) if held else event['message']),
        tools=live_tools+event['tools'],next_operation=event['next_operation'],training_catalog=catalog,
        resource_head=head,jobs=jobs,observer_source_current=current,observer_source_sha256=LOADED_SHA,
        pilot=dict(train_roots=192,fit_roots=128,cases_subset=32,arena_origins=24,
            planned_steps=400,planned_draws=51200,planned_games=192,engineering_handoff_games=2,ordinary_cells=[39,34,58,61],
            capture_training=False,policy_training=False,checkpoint_selection=False,default_engine='H_v1'),
        scope='New ordinary-only development comparison conditional on inherited model background. Historical unknowns and old result exclusions remain.',
        integrity_mode='Full byte verification at audit/seal/explicit verification; live status checks compact sources and receipts.',
        decision=a.get('decision') if a and valid else None)
def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':raw=(s.ROOT/HTML).read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':raw=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='ordinary_epoch_observer_ready',pid=os.getpid(),port=port,source_sha256=LOADED_SHA)),flush=True)
    server.serve_forever()
def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('serve','snapshot'));p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='serve':serve(a.port)
    else:print(json.dumps(snapshot(),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
