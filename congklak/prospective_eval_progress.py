"""Read compact prospective evaluation progress without starting engine work."""
import argparse,datetime,json,os,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from . import config,neurodiag_common as c,guarded_value_pilot_v2_progress as old
ROOT=c.ROOT
FOLDER=ROOT/'build/prospective-eval-v1'
STATUS=FOLDER/'status.json'
ENTRY=ROOT/'tools/progress-prospective-eval.py'
LOADED=c.pins([__file__,ENTRY,ROOT/'congklak/config.py',ROOT/'congklak/neurodiag_common.py',ROOT/'congklak/guarded_value_pilot_v2_progress.py'])
HTML=old.HTML.replace('<title>Congklak value pilot</title>','<title>Congklak prospective evaluation</title>').replace('Congklak · guarded value pilot','Congklak · prospective evaluation').replace('400 value-only training steps. Policy, shared trunk and baseline models stay fixed. Evaluation uses the screened development reservations.','Frozen 400-step candidate. New whole-component reservations. 128 endgame checks, two native recovery interruptions, and 192 equal-time games.').replace('Development results remain conditional on inherited model history. Acquired evaluation origins remain excluded from optimization. Live progress does not promote a model.','Models and policy remain fixed. The prior held run stays preserved. Development results remain conditional on inherited history; live progress does not promote a model.')

def publish(state,message,next_operation='',tools=None,errors=None):
    FOLDER.mkdir(parents=True,exist_ok=True)
    body=dict(schema='prospective-eval-v1-progress',state=state,message=message,next_operation=next_operation,
        tools=tools or [],errors=errors or [],policy_fixed=True,
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    p=STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.pending')
    with p.open('xb') as stream:stream.write((config.canonical(body)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    os.replace(p,STATUS);return body

def small(path,errors,limit=4*1024*1024):
    path=Path(path)
    if not path.exists():return None
    try:
        c.require(path.stat().st_size<=limit,'Compact observer input too large')
        body=c.read(path)
        if type(body) is dict and set(body)=={'body','sha256'}:
            c.require(config.digest(body['body'])==body['sha256'],'Compact envelope differs');body=body['body']
        c.require(type(body) is dict,'Compact status is not an object');return body
    except (OSError,ValueError,KeyError,TypeError) as ex:errors.append(path.name+': '+str(ex));return None

def brief(body):
    if not body:return None
    keep=('state','passed','completed_units','required_units','queries','games','terminal_games','critical_overshoots',
        'ownership_passed','pending_attempts','hold','actual_process_kills','promotion','decision','summary','engineering_execution_complete')
    return {k:body[k] for k in keep if k in body}

def job(name,total,errors):
    folder=FOLDER/'jobs'/name;report=small(folder/'report.json',errors)
    archives=len(list((folder/'archives').glob('*.json')))
    state=(report or {}).get('state','running' if archives else 'not_started')
    if state=='held':errors.append(name+': '+str((report or {}).get('hold')))
    return dict(name={'endgame':'Shared endgame guards','arena':'Both-baseline equal-time games'}[name],state=state,
        completed=(report or {}).get('completed_units',0),total=total,captured_queries=archives,report=brief(report),
        timing='25 ms per decision for both seats' if name=='arena' else 'Timed 25 ms and fixed 128 simulations')

def resources(errors):
    folder=ROOT/'data/prospective-eval-v1/resources';identity=small(folder/'identity.json',errors);head=small(folder/'head.json',errors)
    if not identity or not head:return dict(state='not_started',charged_seconds=9924.807,limit_seconds=14400,remaining_seconds=4475.193)
    try:
        charged=identity['inherited_charged_ns'];pending=None;previous=None;last=None
        c.require(type(head['count']) is int and 0<=head['count']<=200,'Resource cursor malformed')
        for index in range(head['count']):
            event=small(folder/'events'/f'{index:06d}.json',errors)
            c.require(event['index']==index and event['previous']==previous,'Resource event prefix is updating or damaged')
            if event['kind']=='reserve':
                c.require(pending is None,'Resource leases overlap');pending=event['lease'];charged+=pending['reserved_ns']
            else:
                c.require(event['kind']=='finish' and pending and event['token']==pending['token'],'Resource settlement differs')
                charged+=event['charged_ns']-pending['reserved_ns'];pending=None;last=event
            previous=config.digest(event)
        c.require(previous==head['last_sha256'],'Resource highwater is updating or damaged')
        held=small(folder/'hold.json',errors);limit=identity['protocol']['cumulative_ns']
        return dict(state='held' if held else 'active' if pending else 'idle',charged_seconds=round(charged/1e9,3),
            remaining_seconds=round((limit-charged)/1e9,3),limit_seconds=limit/1e9,carried_seconds=9924.807,budget_reset=False,
            pending=None if not pending else {k:pending[k] for k in ('token','worker_output','reserved_ns')},
            last_finish=None if not last else {k:last[k] for k in ('clean','reason','elapsed_ns','peak_private_commit_bytes')},hold=held)
    except (KeyError,TypeError,ValueError) as ex:
        errors.append('Resources: '+str(ex));return dict(state='requires verification')

def snapshot():
    errors=[];status=small(STATUS,errors) or dict(state='preparing',message='Preparing the prospective campaign.')
    errors.extend(status.get('errors',[]));reservation=small(FOLDER/'progress-reservations.json',errors)
    pre=small(FOLDER/'audit/preflight.json',errors);prior=small(FOLDER/'prior-preservation.json',errors)
    recovered=small(FOLDER/'audit/native-recovery/handoff-result.json',errors)
    if recovered is None:recovered=small(FOLDER/'recovery/result.json',errors)
    audit=small(FOLDER/'audit/audit.json',errors);report=small(FOLDER/'report.json',errors)
    tools=[dict(name='Frozen models and policy',state='complete',training_steps=400,new_optimizer_steps=0,
        candidate='Final step 400',policy='Frozen',baselines=['H_v1','H8 relay CPU']),
        dict(name='Prospective whole-component reservations',state='complete' if reservation else 'in progress',
            report=reservation),
        dict(name='Independent prospective admission',state='passed' if pre and pre.get('passed') else 'not_started'),
        dict(name='Prior evidence preservation',state='passed' if prior and prior.get('passed') else 'in progress',
            verified_files=(prior or {}).get('verified_files'),verified_bytes=(prior or {}).get('verified_bytes')),
        job('endgame',128,errors),
        dict(name='Native interruption recovery',state='passed' if recovered and recovered.get('passed') else 'not_started',
            completed=(recovered or {}).get('actual_process_kills',0),total=2,report=brief(recovered)),
        job('arena',192,errors),
        dict(name='Independent final audit',state='passed' if audit and audit.get('passed') else 'not_started',
            report=brief(audit))]
    tools.extend(status.get('tools',[]))
    resource=resources(errors)
    # Only compact event and status files are read. Complete raw traces, maps,
    # checkpoints and source closures belong to explicit admission/audit.
    try:current=all(c.sha(ROOT/name)==value for name,value in LOADED.items())
    except OSError:current=False
    if not current:errors.append('Observer source changed; restart before trusting the display.')
    held=any(t.get('state')=='held' for t in tools) or resource.get('state')=='held'
    return dict(schema='prospective-eval-v1-dashboard',current_work=dict(state='held' if held else status.get('state','preparing'),
        message=status.get('message','')),tools=tools,updated_at=status.get('updated_at'),
        next_operation=status.get('next_operation',''),errors=list(dict.fromkeys(errors)),resources=resource,
        training=dict(steps=400,policy_fixed=True,new_steps=0),results=brief(report),acceptance_recorded=(FOLDER/'acceptance.json').exists(),
        observer_source_current=current,observer_inputs=LOADED,policy_fixed=True,model_promotion=False,
        historically_unseen=False,integrity_mode='Compact live evidence only; source, raw-footprint and numerical checks are independently audited.')

def serve(port=18766):
    c.require(port==18766,'Registered observer uses port 18766')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/':raw=HTML.encode();kind='text/html; charset=utf-8'
            elif self.path=='/api/status':raw=config.canonical(snapshot()).encode();kind='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(config.canonical(dict(event='prospective_eval_v1_observer_ready',pid=os.getpid(),port=port,loaded_inputs=LOADED)),flush=True)
    server.serve_forever()

def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('serve','snapshot'));p.add_argument('--port',type=int,default=18766)
    args=p.parse_args()
    if args.operation=='serve':serve(args.port)
    else:print(config.canonical(snapshot()))
