"""Immutable trace-speed progress observer; never opens active worker heads."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from . import tracing_speed_common as stage
from . import tracing_speed_contract as contract,config
from .native_ownership_trace import MODEL_SHA

ROOT,FOLDER=stage.ROOT,stage.FOLDER
LOADED_SHA=stage.sha(__file__)
HTML='tools/tracing-speed-dashboard.html'
MODELS=('data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
        'data/training-stage5-deeper-h8-f127-seed2718/model.json')

def publish(state,message,cards,next_operation):
    files=sorted((FOLDER/'events').glob('*.json'))
    return stage.atomic(FOLDER/'events'/f'{len(files):06d}.json',dict(schema='tracing-speed-progress-v1',
        state=state,message=message,cards=cards,next_operation=next_operation,models_fixed=True,policy_fixed=True))

def valid_closure(body,eventpath):
    try:
        need=stage.need;event=stage.read(eventpath)
        need(body['schema']=='tracing-speed-acceptance-v1' and body['passed'] is True and
            body['previous_files_preserved']==62200 and body['models_fixed'] is True and body['policy_fixed'] is True and
            body['training_counters']==stage.ZERO and body['training_authorized'] is False and
            all(type(body['training_counters'][k]) is int for k in stage.ZERO) and body['execution_ready'] is False and
            body['production_evaluation_admitted'] is False and len(body['inputs'])>=62200 and
            body['open_holds']==['320 prior support conflicts','missing historical engineering footprints'],'Observer scope differs')
        need(body['independent_audit']=='build/tracing-speed-v1/audit/audit.json','Observer audit scope differs')
        audit=stage.read(ROOT/body['independent_audit'])
        need(audit['schema']=='tracing-speed-independent-audit-v1' and audit['passed'] is True and
            audit['independent_of_implementation'] is True and audit['unresolved_findings']==[] and
            audit['completed_reader_no_engine'] is True and
            audit['models_fixed'] is True and audit['policy_fixed'] is True and audit['training_counters']==stage.ZERO and
            all(type(audit['training_counters'][k]) is int for k in stage.ZERO) and
            audit['training_admitted'] is False and audit['production_evaluation_admitted'] is False and
            all(audit['methods'].get(k,{}).get('passed') is True and audit['methods'][k].get('evidence') for k in contract.METHODS) and
            all(type(audit['recovery_counts'].get(k)) is int and audit['recovery_counts'][k]>=1 for k in contract.KILL_POINTS),'Observer independent audit differs')
        models=stage.common.pins(MODELS)
        need(models==dict(zip(MODELS,(MODEL_SHA['control'],MODEL_SHA['candidate'],MODEL_SHA['H8_relay_CPU']))),
             'Observer frozen model differs')
        rel=eventpath.relative_to(ROOT).as_posix()
        need(body['progress_event']==rel and body['progress_event_sha256']==stage.sha(eventpath) and
            event['acceptance_schema']==body['schema'] and event['training_counters']==stage.ZERO and
            event['baseline_sha256']==stage.sha(FOLDER/'baseline.json') and event['models_sha256']==models,
            'Observer final event differs')
        required=stage.common.pins([__file__,HTML,FOLDER/'baseline.json',stage.PRIOR,eventpath,ROOT/body['independent_audit']])
        panelpath=FOLDER/'timing/panel.json';protocolpath=FOLDER/'timing/protocol.json';reportpath=FOLDER/'timing/production/report.json'
        panel=stage.read(panelpath);protocol=stage.read(protocolpath);report=stage.read(reportpath)
        need(protocol['clock_contract_sha256']==panel['clock_contract_sha256']==config.digest(contract.definition()) and
            protocol['panel_sha256']==stage.sha(panelpath) and report['protocol_sha256']==stage.sha(protocolpath) and
            report['schema']=='trace-speed-timing-report-v1' and len(report['rows'])==len(panel['schedule'])==240,
            'Observer timing protocol differs')
        need(all(all(row[k]==scheduled[k] for k in ('index','root','repetition','engine'))
            for row,scheduled in zip(report['rows'],panel['schedule'])) and
            body['timing_gate']==report['gate']==contract.timing_gate(report['rows'],stage.sha(protocolpath)),
            'Observer timing gate derivation differs')
        timedpath=FOLDER/'jobs/engineering-timed/report.json';calpath=FOLDER/'native/calibration-v4/report.json'
        timed=stage.read(timedpath);calibration=stage.read(calpath)
        fixedpath=FOLDER/'jobs/engineering-fixed/report.json';fixed=stage.read(fixedpath)
        need(timed['passed'] is True and timed['mode']=='timed25' and timed['models_fixed'] is True and
            timed['policy_fixed'] is True and timed['training_counters']==stage.ZERO and timed['decisions']>0 and
            body['timed_handoff_overshoots']==timed['overshoot_decisions'] and
            body['calibration_overshoots']==sum(v['overshoots'] for v in calibration['summary'].values()) and
            body['target_25ms_met']==(report['gate']['passed'] and timed['overshoot_decisions']==0 and
                timed['maximum_critical_ms']<=25 and body['calibration_overshoots']==0),
            'Observer additional timed counterexamples differ')
        audit_counters=audit['engineering_counters']
        need(all(type(v) is int and v>=0 for v in audit_counters.values()),'Observer independent counters differ')
        expected_counters=dict(fixed_games=fixed['games'],fixed_decisions=fixed['decisions'],
            timed_games=timed['games'],timed_decisions=timed['decisions'],timing_attempts=240,
            calibration_prior_attempts=24,calibration_final_attempts=32,calibration_engine_queries=56,
            **{'independent_audit_'+k:v for k,v in audit_counters.items()})
        need(body['engineering_counters']==expected_counters and all(type(v) is int for v in body['engineering_counters'].values()),
            'Observer primary/scoped engineering counter derivation differs')
        clockpath=FOLDER/'clock-summary.json';clocks=stage.read(clockpath)
        need(clocks['schema']=='trace-speed-clock-summary-v1' and body['resource_clock_summary']==clocks['timing'] and
            clocks['timing']['attempts']==240 and clocks['timing']['critical_p95_ms']==report['gate']['p95_ms'] and
            clocks['timing']['critical_maximum_ms']==report['gate']['maximum_ms'],
            'Observer contained versus critical clock summary differs')
        required.update(stage.common.pins([panelpath,protocolpath,reportpath,'congklak/tracing_speed_contract.py']))
        required.update(stage.common.pins([timedpath,calpath,fixedpath,clockpath]))
        required.update(models);required.update(audit['source_inputs'])
        for method in audit['methods'].values():
            need(method['passed'] is True and method['evidence'],'Observer method coverage differs')
            required.update({e['path']:e['sha256'] for e in method['evidence']})
        need(all(body['live_inputs'].get(k)==v==body['inputs'].get(k) for k,v in required.items()) and
            all(body['inputs'].get(k)==v for k,v in body['live_inputs'].items()),'Observer live inputs differ')
        stage.common.current(body['live_inputs']);return True
    except (KeyError,TypeError,ValueError,OSError,IndexError):return False

def snapshot():
    files=sorted((FOLDER/'events').glob('*.json'));state=stage.read(files[-1])
    path=FOLDER/'acceptance.json';body=stage.read(path) if path.exists() else None
    source_ok=stage.sha(__file__)==LOADED_SHA
    closed=bool(source_ok and body and valid_closure(body,files[-1]))
    invalid=not source_ok or body is not None and not closed
    history=stage.read(sorted((ROOT/'build/ownership-implementation-v1/events').glob('*.json'))[-1])['cards']
    return dict(schema='tracing-speed-dashboard-v1',accepted=closed,current_cards=len(state['cards']),
        tools=state['cards']+history,current_work=dict(state='hold' if invalid else state['state'],
            message='HOLD: bound observer, source or evidence differs.' if invalid else state['message']),
        progress=dict(worker_commits=sum(1 for _ in (FOLDER/'jobs/engineering-fixed/commits').glob('*.json')),
            timed_handoff_commits=sum(1 for _ in (FOLDER/'jobs/engineering-timed/commits').glob('*.json')),
            calibration_commits=sum(1 for _ in (FOLDER/'native/calibration-v4/commits').glob('*.json')),
            recovery_cases=sum(1 for _ in (FOLDER/'audit/worker-recovery').glob('case-*.json')),
            timing_commits=sum(1 for _ in (FOLDER/'timing/production/commits').glob('*.json')),
            physical_counts_are_provisional=not closed),
        counters=dict(stage.ZERO,**(body.get('engineering_counters',{}) if closed else {})),
        definition_only=False,execution_ready=False,training_authorized=False,
        live_suffix_worker_verified=closed,confidence_intervals_enabled=False,
        target_25ms_met=body['target_25ms_met'] if closed else False,
        timing_metrics=body['timing_gate'] if closed else None,
        resource_clocks=body['resource_clock_summary'] if closed else None,
        models_fixed=True,policy_fixed=True,observer_source_current=source_ok,observer_source_sha256=LOADED_SHA,
        next_operation=state['next_operation'],scope='Fixed-model engineering only. The critical decision includes reset, search, complete raw capture, owned bytes, hashing, validation and ownership checks. Archival time is reported separately and charged to the cumulative lease. Earlier support/provenance holds remain; no training or production strength admission.')

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
    print(json.dumps(dict(event='tracing_speed_observer_ready',pid=os.getpid(),source_sha256=LOADED_SHA,port=port)),flush=True)
    server.serve_forever()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('snapshot','serve'));p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='snapshot':print(json.dumps(snapshot(),sort_keys=True,allow_nan=False))
    else:serve(a.port)

if __name__=='__main__':main()
