"""Metadata-only product and saved-benchmark review dashboard."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import config, neurodiag_common as common
from .guarded_value_pilot_v2_progress import HTML as BASE_HTML

ROOT = common.ROOT
FOLDER = ROOT / 'build/benchmark-loss-review-v1'
STATUS = FOLDER / 'status.json'
ENTRY = ROOT / 'tools/progress-benchmark-loss-review-v1.py'
LOADED = common.pins([__file__, ENTRY, ROOT/'congklak/config.py',
                     ROOT/'congklak/neurodiag_common.py',
                     ROOT/'congklak/guarded_value_pilot_v2_progress.py'])
BENCHMARK = ROOT / 'build/prospective-benchmark-v5'
SEALED = {
    'report.json': '0f3ef98eec37989825c29dd51d484beb28bbc99e55913f4507791fba951db8b3',
    'acceptance.json': '4113e4c6bef7426037c5c4f34afd21318a273243701a414fe8cfa273bbe79451',
    'audit/closure.json': '680c60b60f7cd5a2e7fd4b869ceccc4d1be1a272f4b699dfd748a371391f4423',
}
MODELS = {
    'data/guarded-value-pilot-v2/training/model.json': '1668e8a9b2734a212102e8db0500641fe1f204f88a5303f9e915268a3b4f531a',
    'data/quality-train-v1/model.json': 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558',
    'data/training-stage5-deeper-h8-f127-seed2718/model.json': '4796c375398a155fb5edfcfdf1be2442734b7ff1563a32a81d7a131a74103b6b',
}
HTML = BASE_HTML.replace('Congklak value pilot', 'Congklak product progress').replace(
    'Congklak · guarded value pilot', 'Congklak · product delivery').replace(
    '400 value-only training steps. Policy, shared trunk and baseline models stay fixed. Evaluation uses the screened development reservations.',
    'Building a local playable Betawi board and analyzer. Reviewing existing benchmark losses; models and policy stay fixed.').replace(
    'Development results remain conditional on inherited model history. Acquired evaluation origins remain excluded from optimization. Live progress does not promote a model.',
    'The sealed96-game benchmark is retained unchanged. Product history is separate from research data. No new training or model promotion.').replace(
    '<h2>Next operation</h2>',
    '<p><a href="http://127.0.0.1:18767/" style="color:#a9c7e8">Open the local Congklak app</a></p><h2>Next operation</h2>')


def publish(state, message, next_operation='', tools=None, errors=None):
    FOLDER.mkdir(parents=True, exist_ok=True)
    body = dict(schema='benchmark-loss-review-v1-progress', state=state, message=message,
                next_operation=next_operation, tools=tools or [], errors=errors or [],
                policy_fixed=True, new_optimizer_steps=0,
                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temporary = STATUS.with_name(STATUS.name+'.'+uuid.uuid4().hex+'.pending')
    with temporary.open('xb') as stream:
        stream.write((config.canonical(body)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, STATUS)
    return body


def small(path, errors):
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 4*1024**2:
            raise ValueError('Observer input exceeds compact limit')
        body = json.loads(path.read_bytes())
        if type(body) is not dict:
            raise ValueError('Observer input is not an object')
        return body
    except (OSError, ValueError) as error:
        errors.append(str(path.relative_to(ROOT))+': '+str(error))
        return None


def snapshot():
    errors = []
    status = small(STATUS, errors) or dict(state='building product', message='Preparing the local Congklak app.', next_operation='Finish the playable board and recoverable saved-game review.')
    errors.extend(status.get('errors', []))
    protected = all(common.sha(BENCHMARK/name) == digest for name, digest in SEALED.items())
    if not protected:
        errors.append('The sealed benchmark metadata changed.')
    models_fixed = all(common.sha(ROOT/name) == digest for name, digest in MODELS.items())
    if not models_fixed:
        errors.append('A fixed model changed.')
    analysis = small(FOLDER/'analysis/progress.json', errors) or {}
    summary = small(FOLDER/'analysis/summary.json', errors) or {}
    recovery = small(FOLDER/'recovery/result.json', errors) or {}
    product = small(ROOT/'build/product-v1/progress.json', errors) or {}
    product_acceptance = small(ROOT/'build/product-v1/acceptance.json', errors) or {}
    product_recovery = small(ROOT/'build/product-v1/recovery.json', errors) or {}
    audit = small(FOLDER/'audit/audit.json', errors) or {}
    current = all(common.sha(ROOT/name) == digest for name, digest in LOADED.items())
    if not current:
        errors.append('Observer source changed; restart required.')
    completed = analysis.get('completed_games', analysis.get('completed_units', analysis.get('completed', 0)))
    tools = [
        dict(name='Fixed models and policy', state='complete' if models_fixed else 'failed', new_optimizer_steps=0, product_engine='H_v1'),
        dict(name='Sealed benchmark retained', state='passed' if protected else 'failed', games=96,
             timed_decisions=2609, overruns=0, maximum_ms=22.1252),
        dict(name='Saved-game loss review', state=analysis.get('state', 'building'), completed=completed,
             total=96, summary=summary.get('summary', summary.get('decision'))),
        dict(name='Analysis interruption recovery', state='passed' if recovery.get('passed') else 'pending',
             actual_process_kills=recovery.get('actual_process_kills', 0)),
        dict(name='Local browser product', state=product.get('state', 'building'),
             message=product.get('message'), url='http://127.0.0.1:18767/'),
        dict(name='Product acceptance', state='passed' if product_acceptance.get('passed') else 'pending',
             summary=product_acceptance.get('summary')),
        dict(name='Product saved-session recovery', state='passed' if product_recovery.get('passed') else 'pending',
             actual_process_kills=product_recovery.get('actual_process_kills', 0)),
        dict(name='Independent final audit', state='passed' if audit.get('passed') else 'pending'),
    ]
    tools.extend(status.get('tools', []))
    return dict(schema='benchmark-loss-review-v1-dashboard',
                current_work=dict(state=status['state'], message=status['message']),
                tools=tools, errors=list(dict.fromkeys(errors)), next_operation=status.get('next_operation', ''),
                updated_at=status.get('updated_at'), observer_source_current=current,
                observer_inputs=LOADED, acceptance_recorded=(FOLDER/'decision.json').exists(),
                resources=dict(research_benchmark='Sealed and complete; no new research search',
                               analysis='Existing saved evidence only', product='Local CPU analyzer; separate user history'),
                training=dict(new_steps=0, policy_fixed=True, models_fixed=True),
                results=dict(candidate_vs_H_v1='9 wins,3 draws,12 losses in24 games',
                             candidate_vs_H8='8 wins,4 draws,12 losses in24 games',
                             ordinary_vs_H_v1='2 wins,2 draws,4 losses in8 games',
                             ordinary_vs_H8='2 wins,1 draw,5 losses in8 games',
                             known_joint_cohort=True, product_engine='H_v1'),
                integrity_mode='Read-only compact metadata; benchmark sources and models remain sealed.')


def serve(port=18766):
    if port != 18766:
        raise ValueError('This dashboard is registered for18766')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                raw, kind = HTML.encode(), 'text/html; charset=utf-8'
            elif self.path == '/api/status':
                raw, kind = config.canonical(snapshot()).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(config.canonical(dict(event='benchmark_loss_review_v1_observer_ready', pid=os.getpid(),
                                port=port, loaded_inputs=LOADED)), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('serve', 'snapshot'))
    parser.add_argument('--port', type=int, default=18766)
    args = parser.parse_args()
    if args.operation == 'serve':
        serve(args.port)
    else:
        print(config.canonical(snapshot()))

if __name__ == '__main__':
    main()
