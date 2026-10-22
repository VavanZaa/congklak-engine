"""Read-only dashboard for scoped resolution, never an execution admission."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from . import config, ownership_resolution_common as s, ownership_resolution_contract as c

HTML='tools/ownership-resolution-dashboard.html'
LOADED_SHA=s.sha(__file__)


def publish(state,message,cards,next_operation):
    files=sorted((s.FOLDER/'events').glob('*.json'))
    return s.atomic(s.FOLDER/'events'/f'{len(files):06d}.json',dict(
        schema=s.SCHEMA+'-progress',state=state,message=message,cards=cards,
        next_operation=next_operation,models_fixed=True,policy_fixed=True,
        training_authorized=False))


def valid_closure(body,eventpath):
    try:
        need=s.need;event=s.read(eventpath);audit=s.read(s.FOLDER/'audit/audit.json')
        need(body['schema']==s.SCHEMA+'-acceptance' and body['passed'] is True and
            body['models_fixed'] is True and body['policy_fixed'] is True and
            body['activity']==s.ZERO and all(type(v) is int for v in body['activity'].values()) and
            body['training_authorized'] is False and body['production_evaluation_admitted'] is False and
            body['historical_provenance_complete'] is False and body['historically_unseen'] is False and
            body['inherited_unknown_background'] is True,'Resolution scope differs')
        need(audit['schema']==s.SCHEMA+'-independent-audit' and audit['passed'] is True and
            audit['independent_of_implementation'] is True and audit['unresolved_findings']==[] and
            audit['models_fixed'] is True and audit['policy_fixed'] is True and
            audit['activity']==s.ZERO and all(type(v) is int for v in audit['activity'].values()) and
            audit['training_authorized'] is False and audit['production_evaluation_admitted'] is False and
            all(audit['methods'].get(k,{}).get('passed') is True and audit['methods'][k].get('evidence') for k in c.METHODS) and
            all(type(audit['recovery_counts'].get(k)) is int and audit['recovery_counts'][k]>=1 for k in c.KILL_POINTS),
            'Independent resolution audit differs')
        baseline=s.read(s.FOLDER/'baseline.json');report=s.read(s.FOLDER/'production/report.json')
        need(body['previous_files_preserved']==baseline['inputs_count'] and
            all(body['inputs'].get(k)==v for k,v in baseline['inputs'].items()),'Preserved baseline differs')
        need(config.canonical(body['contract'])==config.canonical(c.definition()) and
            config.canonical(body['epoch'])==config.canonical(report['epoch']) and
            report['passed'] is True and report['fixture'] is False and report['activity']==s.ZERO and
            all(type(v) is int for v in report['activity'].values()) and
            body['production_report_sha256']==s.sha(s.FOLDER/'production/report.json'),
            'Resolution report/contract differs')
        epoch=report['epoch']
        need(epoch['schema']==s.SCHEMA+'-epoch' and type(epoch['prerequisites_ready']) is bool and
            epoch['scope']=='ordinary-only known-provenance next-update development' and
            epoch['contract_sha256']==config.digest(c.definition()) and
            epoch['epoch_id']==config.digest({k:v for k,v in epoch.items() if k!='epoch_id'}),
            'Resolution epoch identity differs')
        need(epoch['models_fixed'] is True and epoch['policy_fixed'] is True and
            epoch['training_authorized'] is False and epoch['labels_authorized'] is False and
            epoch['production_evaluation_admitted'] is False and epoch['model_promotion_authorized'] is False and
            epoch['activity']==s.ZERO and all(type(v) is int for v in epoch['activity'].values()) and
            epoch['historical_provenance_complete'] is False and epoch['historically_unseen'] is False and
            epoch['inherited_unknown_background'] is True and epoch['inherited_model_ancestry'] is True and
            epoch['legacy_strength_results_rehabilitated'] is False and epoch['old_candidate_quarantined'] is True and
            epoch['independent_strength_established'] is False and
            epoch['original_balanced512_recipe_available'] is False and
            epoch['bonus_capture_training_support_available'] is False and
            epoch['required_next_run']['training_admitted_now'] is False,
            'Nested epoch contradicts fixed-model historical or execution scope')
        need(body['progress_event']==s.relative(eventpath) and body['progress_event_sha256']==s.sha(eventpath) and
            event['acceptance_schema']==body['schema'] and event['baseline_sha256']==s.sha(s.FOLDER/'baseline.json') and
            event['models']==s.common.pins(s.MODELS),'Final progress binding differs')
        required=s.common.pins([__file__,HTML,s.FOLDER/'baseline.json',s.PRIOR,
            s.FOLDER/'contract.json',s.FOLDER/'plan.json',s.FOLDER/'production/report.json',
            s.FOLDER/'production/catalog.json',s.FOLDER/'audit/audit.json',eventpath,*s.MODELS])
        required.update(audit['source_inputs'])
        for method in audit['methods'].values():
            required.update({r['path']:r['sha256'] for r in method['evidence']})
        need(all(body['live_inputs'].get(k)==v==body['inputs'].get(k) for k,v in required.items()) and
            all(body['inputs'].get(k)==v for k,v in body['live_inputs'].items()),'Resolution live source closure differs')
        s.common.current(body['live_inputs']);return True
    except (KeyError,ValueError,TypeError,OSError,IndexError):return False


def snapshot():
    paths=sorted((s.FOLDER/'events').glob('*.json'));event=s.read(paths[-1])
    ap=s.FOLDER/'acceptance.json';acceptance=s.read(ap) if ap.exists() else None
    current=s.sha(__file__)==LOADED_SHA
    accepted=bool(current and acceptance and valid_closure(acceptance,paths[-1]))
    invalid=not current or acceptance is not None and not accepted
    prior=s.read(sorted((s.ROOT/'build/tracing-speed-v1/events').glob('*.json'))[-1])
    return dict(schema=s.SCHEMA+'-dashboard',accepted=accepted,models_fixed=True,policy_fixed=True,
        training_authorized=False,production_evaluation_admitted=False,
        historical_provenance_complete=False,historically_unseen=False,inherited_unknown_background=True,
        current_work=dict(state='hold' if invalid else event['state'],
            message='HOLD: source or acceptance evidence differs.' if invalid else event['message']),
        tools=event['cards']+prior['cards'],next_operation=event['next_operation'],
        progress=dict(committed_units=len(list((s.FOLDER/'production/commits').glob('*.json'))),
            total_units=len(s.read(s.FOLDER/'plan.json')['tasks']) if (s.FOLDER/'plan.json').exists() else None,
            provisional=not accepted),activity=s.ZERO,
        epoch=acceptance['epoch'] if accepted else None,
        observer_source_current=current,observer_source_sha256=LOADED_SHA,
        scope='Known-provenance next-update development only. Historical unknowns and old result exclusions remain. Prerequisite readiness grants no permission to train, label, evaluate or promote.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':raw=(s.ROOT/HTML).read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':raw=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers();self.wfile.write(raw)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(json.dumps(dict(event='ownership_resolution_observer_ready',pid=os.getpid(),port=port,
        source_sha256=LOADED_SHA)),flush=True);server.serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('serve','snapshot'))
    p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if a.operation=='serve':serve(a.port)
    else:print(json.dumps(snapshot(),sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
