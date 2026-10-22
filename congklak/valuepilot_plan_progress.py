"""Read-only plan view; it never imports an engine or training runtime."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'build/valuepilot-plan-v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def current(mapping):
    return all((ROOT/name).is_file() and sha(ROOT/name)==checksum for name,checksum in mapping.items())


def snapshot():
    plan=read(FOLDER/'plan.json')
    audit=read(FOLDER/'audit.json') if (FOLDER/'audit.json').exists() else {}
    accepted=read(FOLDER/'acceptance.json') if (FOLDER/'acceptance.json').exists() else {}
    reviewed=audit.get('passed') is True and current(audit.get('inputs',{}))
    closed=accepted.get('passed') is True and current(accepted.get('new_inputs',{}))
    artifacts=[name for name in plan['planned_future_namespaces'] if (ROOT/name).exists()]
    planned='hold' if artifacts else 'planned'
    cards=[]
    def card(key,title,state,result,improvement,evidence='docs/NEURAL_VALUE_PILOT_PLAN.md'):
        cards.append(dict(id='valuepilot-'+key,title=title,state=state,result=result,improvement=improvement,
            evidence=evidence,needs_improvement=state not in ('complete','planned')))
    card('plan','Audited pilot plan','complete' if reviewed and closed else 'reviewed' if reviewed else 'running',
        'One complete H8 root target; 2,065 value parameters; original policy and value path frozen.',
        'Implementation starts with metadata ownership and the versioned native interface.',
        'docs/NEURAL_VALUE_PILOT_PLAN.md; docs/NEURAL_VALUE_PILOT_PLAN_AUDIT.md')
    card('data','Protected data and component split',planned,
        'Planned: 512 training roots, 128 fit-check roots and 24 distinct arena components. Actual IDs are not reserved.',
        'Rebuild every ownership overlay before targets, then freeze actual IDs; stop on a quota shortage.')
    card('export','Value branch and inference contract',planned,
        '12,872 base parameters frozen. A 127 to 16 to 1 branch operates only at 25–74 active seeds.',
        'Check policy equality, zero initialization, gate boundaries and native/scalar/batched CPU parity.')
    card('training','Fixed final training run',planned,
        '0 / 400 steps; 0 / 51,200 draws; 0 / 20 scheduled checkpoints. CPU FP32, one thread.',
        'One fixed seed and final checkpoint. Preserve optimizer, sampler and all active RNG state.')
    card('fit','Value fit and action quality',planned,
        'Planned: 128 fit-check roots, 256 timed H8 measurements, 138 fixed-work searches across all cohorts.',
        'Track normalized Huber, seed errors, capture errors and best-action regret without selecting checkpoints.')
    card('exact','Endgame and low-active guards',planned,
        'Planned: 256 timed exact measurements across four engines; 64 candidate optimality checks.',
        'All 32 reused exact roots must pass; 16 are forced. Low-active fixed-work behavior must match control.')
    card('arena','Both baseline strength tests',planned,
        '0 / 192 games. Twenty-four components, both seats, H_v1 and H8 relay CPU at whole-decision 25 ms.',
        'Keep control paired with candidate; report 24-cluster intervals, actual clocks and any overlap/fallback.')
    card('recovery','Interrupted and completed sessions',planned,
        '0 / 2 training kills; 0 / 4 evaluation barriers. Completed access must acquire no writer lock.',
        'Verify exact optimizer/RNG continuation and committed prefixes; charge uncommitted work to resource leases.')
    card('audit','Implementation audit and final decision',planned,
        'Implementation tests and strength evidence have not run. H_v1 remains the default.',
        'Separate valid negative results from invalid evidence. No automatic promotion, expanded data or retraining.')
    if artifacts:
        work=dict(state='hold',message='Implementation artifacts exist; this plan view does not monitor their live work. Install the implementation observer before claiming runtime counts.')
    elif reviewed and closed:
        work=dict(state='complete',message='Plan finalized and independently reviewed. Implementation and training have not started. Next: data admission and the native value-branch contract.')
    else:
        work=dict(state='running',message='Closing the pilot plan and independent design review. Models and policy remain fixed; no training is running.')
    history=read(ROOT/'build/neurodiag-dashboard-final.json')['cards']
    return dict(schema='neural-value-pilot-plan-progress-v1',tools=cards+history,current_work=work,
        plan_audited=reviewed,plan_package_current=closed,implementation_started=bool(artifacts),
        execution_counters=dict(training_steps=0,optimizer_draws=0,new_games=0,new_teacher_labels=0) if not artifacts else None,
        active_tools=[],scope='Plan view only. Training, labels, game results and implementation acceptance are planned; the preceding certified diagnostics are history.')


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':body=(ROOT/'tools/valuepilot-plan-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':body=json.dumps(snapshot(),sort_keys=True,allow_nan=False).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers();self.wfile.write(body)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    print('Neural value pilot plan http://127.0.0.1:'+str(port)+'/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    sub.add_parser('snapshot');sub.add_parser('serve').add_argument('--port',type=int,default=18766)
    args=parser.parse_args()
    if args.operation=='snapshot':print(json.dumps(snapshot(),sort_keys=True,allow_nan=False),flush=True)
    else:serve(args.port)


if __name__=='__main__':main()
