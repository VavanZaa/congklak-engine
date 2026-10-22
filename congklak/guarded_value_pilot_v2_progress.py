"""Compact, read-only pilot observer. No archive, engine, Torch or worker entry.

The observer reads durable catalogs and small reports. These are live progress
facts, not a substitute for the independent final verifier. Only ``publish``
writes, exclusively to this pilot's mutable status document.
"""
import argparse,datetime,hashlib,json,os,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from . import guarded_value_pilot_v2_common as s,config

STATUS=s.FOLDER/'status.json'
LOADED=s.common.pins([__file__,s.__file__,s.ROOT/'congklak/config.py',s.ROOT/'tools/progress-guarded-value-pilot-v2.py'])
SCHEMA=s.SCHEMA+'-dashboard'

HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Congklak value pilot</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#101722;color:#e6edf6;font:15px system-ui,sans-serif}main{max-width:1230px;margin:34px auto;padding:0 22px}h1{font-size:29px;margin-bottom:8px}h2{font-size:20px}h3{margin:0 0 12px;font-size:17px}.muted{color:#a4b3c6;line-height:1.55}.headline{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.pill{border:1px solid #53708b;border-radius:30px;padding:5px 12px;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(275px,1fr));gap:14px}.card{background:#1a2635;border:1px solid #34475e;border-radius:13px;padding:19px}.card.done{border-color:#4a987d}.card.held{border-color:#d99065}.state{font-size:14px;color:#9fc3e8}.big{font-size:25px;font-weight:650;margin:9px 0}.bar{height:8px;background:#0f1722;border-radius:8px;overflow:hidden;margin:14px 0}.fill{height:100%;background:#74b7ed}.done .fill{background:#79d2aa}.held .fill{background:#e0a078}p{line-height:1.5}.detail{font-size:13px;color:#b6c6d8;overflow-wrap:anywhere}.errors{border-left:4px solid #db976e;background:#352b27;padding:12px 18px;border-radius:5px;margin:18px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;color:#a9c7e8}.foot{font-size:12px;margin-top:28px}</style>
<main><h1>Congklak · guarded value pilot</h1><p class="muted">400 value-only training steps. Policy, shared trunk and baseline models stay fixed. Evaluation uses the screened development reservations.</p>
<div class="headline"><h2 id="state">Loading observer…</h2><span class="pill" id="policy">Policy fixed</span><span class="pill" id="updated">Waiting for status</span></div><p id="message"></p><div id="errors"></div><div class="grid" id="cards"></div>
<h2>Next operation</h2><p id="next"></p><details><summary>Resources and recorded evidence</summary><pre id="raw"></pre></details><p class="muted foot">Development results remain conditional on inherited model history. Acquired evaluation origins remain excluded from optimization. Live progress does not promote a model.</p></main>
<script>let busy=false;const el=id=>document.getElementById(id);async function refresh(){if(busy)return;busy=true;try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('Observer returned '+r.status);let d=await r.json();el('state').textContent=d.current_work.state;el('message').textContent=d.current_work.message;el('next').textContent=d.next_operation;el('updated').textContent=d.updated_at||'Live catalog view';let errors=el('errors');errors.replaceChildren();if(d.errors.length){let box=document.createElement('div');box.className='errors';for(let e of d.errors){let p=document.createElement('p');p.textContent=e;box.append(p)}errors.append(box)}let cards=el('cards');cards.replaceChildren();for(let c of d.tools){let box=document.createElement('section');box.className='card'+(c.state==='complete'||c.state==='passed'?' done':c.state==='held'||c.state==='failed'?' held':'');let h=document.createElement('h3');h.textContent=c.name;box.append(h);let st=document.createElement('div');st.className='state';st.textContent=c.state;box.append(st);if(Number.isFinite(c.total)){let big=document.createElement('div');big.className='big';big.textContent=c.completed+' / '+c.total;box.append(big);let bar=document.createElement('div');bar.className='bar';let fill=document.createElement('div');fill.className='fill';fill.style.width=Math.max(0,Math.min(100,100*c.completed/Math.max(1,c.total)))+'%';bar.append(fill);box.append(bar)}for(let [k,v] of Object.entries(c)){if(['name','state','completed','total'].includes(k)||v===null)continue;let p=document.createElement('p');p.className='detail';p.textContent=k.replaceAll('_',' ')+': '+(typeof v==='object'?JSON.stringify(v):v);box.append(p)}cards.append(box)}el('raw').textContent=JSON.stringify({resources:d.resources,training:d.training,results:d.results,acceptance_recorded:d.acceptance_recorded,observer_source_current:d.observer_source_current,integrity_mode:d.integrity_mode},null,2)}catch(e){el('state').textContent='Observer unavailable';el('message').textContent=e.message}finally{busy=false}}refresh();setInterval(refresh,5000)</script></html>'''

def publish(state,message,tools=None,next_operation='',errors=None):
    """Mutable status publication for the controller, never for the observer."""
    s.need(not (s.FOLDER/'acceptance.json').exists(),'Cannot change pilot status after acceptance')
    body=dict(schema=s.SCHEMA+'-progress',state=state,message=message,tools=tools or [],
        next_operation=next_operation,errors=errors or [],policy_fixed=True,
        updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path=s.path_of(STATUS);path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:stream.write((config.canonical(body)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path);return body

def _optional(path,errors,limit=4*1024*1024):
    if not path.exists():return None
    try:
        s.need(path.stat().st_size<=limit,'Compact report exceeds observer size limit')
        body=s.read(path)
        if type(body) is dict and set(body)=={'body','sha256'}:
            s.need(config.digest(body['body'])==body['sha256'],'Compact envelope digest differs');body=body['body']
        s.need(type(body) is dict,'Compact status is not an object');return body
    except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(s.relative(path)+': '+str(exc));return None

def _brief(body):
    if not body:return None
    keys=('state','passed','steps','target_steps','production_steps','required_units','completed_units','games','terminal_games',
          'queries','ownership_passed','critical_overshoots','pending_attempts','hold','checks','decision','metrics',
          'actual_process_kills','independent_checkpoint_recovery_passed','native_handoff_recovery_passed')
    return {k:body[k] for k in keys if k in body}

def _resources(errors):
    folder=s.DATA/'resources';identity=_optional(folder/'identity.json',errors);head=_optional(folder/'head.json',errors)
    if identity is None or head is None:return dict(state='not_started',inherited_seconds=s.INHERITED_NS/1e9)
    try:
        count=head['count'];s.need(type(count) is int and 0<=count<=1000,'Resource head count is invalid')
        charged=identity['inherited_charged_ns'];pending=None;previous=None;last=None
        for index in range(count):
            event=_optional(folder/'events'/f'{index:06d}.json',errors,128*1024)
            s.need(event is not None and event['index']==index and event['previous']==previous,'Resource event prefix is updating or damaged')
            if event['kind']=='reserve':
                s.need(pending is None,'Resource leases overlap');pending=event['lease'];charged+=pending['reserved_ns']
            elif event['kind']=='finish':
                s.need(pending is not None and event['token']==pending['token'],'Resource finish binding differs')
                charged+=event['charged_ns']-pending['reserved_ns'];pending=None;last=event
            else:raise ValueError('Unknown resource event')
            previous=config.digest(event)
        s.need(head['last_sha256']==previous,'Resource head is updating or damaged')
        limit=identity['protocol']['cumulative_ns'];held=_optional(folder/'hold.json',errors)
        return dict(state='held' if held else 'active' if pending else 'idle',charged_seconds=round(charged/1e9,3),
            remaining_seconds=round((limit-charged)/1e9,3),limit_seconds=limit/1e9,
            inherited_seconds=identity['inherited_charged_ns']/1e9,event_count=count,
            pending=None if pending is None else {k:pending[k] for k in ('token','worker_output','reserved_ns') if k in pending},
            last_finish=None if last is None else {k:last[k] for k in ('clean','reason','elapsed_ns','peak_private_commit_bytes') if k in last},hold=held)
    except (ValueError,KeyError,TypeError) as exc:errors.append('Resources: '+str(exc));return dict(state='requires verification')

def _job(name,total,errors):
    path=s.FOLDER/'jobs'/name;report=_optional(path/'report.json',errors)
    # Final job catalogs may contain many thousands of compact commit hashes.
    # Presence is enough for live progress; do not reread the large map on polls.
    catalog=(path/'catalog.json').is_file()
    commits=list((path/'commits').glob('*.json')) if (path/'commits').exists() else []
    state=(report or {}).get('state','running' if commits else 'not_started')
    if report and report.get('state')=='held':errors.append(name+': '+str(report.get('hold') or 'Ownership or execution hold'))
    done=(report or {}).get('completed_units',0)
    if type(done) is not int or not 0<=done<=total:errors.append(name+': invalid recorded unit count');done=0
    return dict(name={'initial-conformance':'Initial native equivalence','conformance':'Final native conformance',
        'guards':'Capture and endgame guards','arena':'Equal-time strength tests'}[name],state=state,
        completed=done,total=total,durable_records=len(commits),catalog_present=catalog,
        baselines=['H_v1','H8 relay CPU'] if name=='arena' else None,
        timing='25 ms per decision for both seats' if name=='arena' else None,report=_brief(report))

def snapshot():
    errors=[];status=_optional(STATUS,errors) or dict(state='preparing',message='Preparing the guarded value pilot.',
        tools=[],next_operation='Freeze the implementation and complete the independent preflight.')
    if isinstance(status.get('errors'),list):errors.extend(str(e) for e in status['errors'])
    elif status.get('error'):errors.append(str(status['error']))
    try:current=all(s.sha(s.ROOT/name)==value for name,value in LOADED.items())
    except OSError:current=False
    if not current:errors.append('Observer source changed; restart the observer before trusting its status.')
    plan=_optional(s.PLAN,errors);preflight=_optional(s.PREFLIGHT,errors)
    train_cache=_optional(s.DATA/'cache/catalog.json',errors);fit_cache=_optional(s.FOLDER/'fit-data/catalog.json',errors)
    shared=_optional(s.FOLDER/'guard/shared.json',errors);initial=(s.DATA/'initial/model.json').is_file()
    prep=[bool(plan),bool(train_cache and train_cache.get('complete')),bool(fit_cache and fit_cache.get('complete')),
        bool(shared and shared.get('records')==8135078),initial,bool(preflight and preflight.get('passed') is True)]
    tools=[dict(name='Preparation',state='complete' if all(prep) else 'in progress',completed=sum(prep),total=len(prep),
        frozen_plan=prep[0],training_cache_192=prep[1],development_cache_128=prep[2],complete_shared_guard=prep[3],
        stepzero_export=prep[4],independent_preflight=prep[5])]
    catalog=_optional(s.DATA/'training/catalog.json',errors);result=_optional(s.DATA/'training/result.json',errors)
    steps=(catalog or {}).get('step',0)
    if type(steps) is not int or not 0<=steps<=400:errors.append('Training: invalid committed step count');steps=0
    complete=bool(result and result.get('state')=='complete' and result.get('steps')==400 and (s.DATA/'training/certificate.json').is_file())
    training=dict(state='complete' if complete else 'finalizing' if steps==400 else 'checkpointed' if steps else 'not_started',
        steps=steps,target_steps=400,checkpoints=len((catalog or {}).get('checkpoints',[])),policy_fixed=True,
        value_branch_only=True,result=_brief(result))
    tools.append(dict(name='Value-only training',state=training['state'],completed=steps,total=400,
        durable_checkpoints=training['checkpoints'],policy='Frozen',trainable='16-unit residual value branch'))
    tools.append(_job('initial-conformance',5,errors));fit=_optional(s.FOLDER/'fit/report.json',errors)
    fit_complete=bool(fit and (s.FOLDER/'fit/certificate.json').is_file())
    tools.append(dict(name='Reserved-position value fit',state='passed' if fit_complete and fit.get('passed') else
        'failed' if fit_complete else 'not_started',completed=128 if fit_complete else 0,total=128,report=_brief(fit)))
    tools.extend(_job(name,total,errors) for name,total in (('conformance',256),('guards',256),('arena',192)))
    recovery=[_optional(s.FOLDER/'audit/training-recovery/result.json',errors),
        _optional(s.FOLDER/'audit/native-recovery/handoff-result.json',errors)]
    kills=sum(r.get('actual_process_kills',0) for r in recovery if r and r.get('passed') is True)
    tools.append(dict(name='Interruption recovery',state='passed' if kills==4 else 'in progress' if any(recovery) else 'not_started',
        completed=kills,total=4,training_checkpoint_tests=bool(recovery[0] and recovery[0].get('passed')),
        native_handoff_tests=bool(recovery[1] and recovery[1].get('passed'))))
    audit=_optional(s.FOLDER/'audit/audit.json',errors);acceptance_present=(s.FOLDER/'acceptance.json').is_file()
    tools.append(dict(name='Independent final audit',state='passed' if audit and audit.get('passed') is True else
        'failed' if audit and audit.get('passed') is False else 'not_started',
        unresolved_findings=(audit or {}).get('unresolved_findings',[]),sealed_marker_present=acceptance_present))
    if isinstance(status.get('tools'),list):tools.extend(t for t in status['tools'] if type(t) is dict and 'name' in t and 'state' in t)
    resources=_resources(errors);held=resources.get('state')=='held' or any(t['state']=='held' for t in tools)
    state='held' if held else 'observer restart required' if not current else status.get('state','preparing')
    results=_optional(s.FOLDER/'report.json',errors)
    return dict(schema=SCHEMA,current_work=dict(state=state,message=status.get('message','')),tools=tools,
        updated_at=status.get('updated_at'),next_operation=status.get('next_operation',''),errors=list(dict.fromkeys(errors)),
        observer_source_current=current,observer_inputs=LOADED,policy_fixed=True,baseline_models_fixed=True,
        production_training_steps=steps,training=training,resources=resources,results=_brief(results),
        acceptance_recorded=acceptance_present,accepted=False,
        historically_unseen=False,inherited_unknown_background=True,model_promotion=False,
        integrity_mode='Compact live catalogs and reports only. Full source, ownership and numerical checks belong to the independent verifier.')

def serve(port=18766):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':raw=HTML.encode();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':raw=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
            except Exception as error:self.send_error(503,str(error))
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(config.canonical(dict(event='guarded_value_pilot_v2_observer_ready',pid=os.getpid(),port=port,loaded_inputs=LOADED)),flush=True)
    server.serve_forever()

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('serve','snapshot'));parser.add_argument('--port',type=int,default=18766)
    args=parser.parse_args(argv)
    if args.operation=='serve':serve(args.port)
    else:print(config.canonical(snapshot()))

if __name__=='__main__':main()
