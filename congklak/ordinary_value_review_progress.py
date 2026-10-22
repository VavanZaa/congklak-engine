"""Read-only observer for saved value-pilot overlap, losses and validation review."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import time
import uuid
from . import config, progress as old
from .native import ROOT

FOLDER = ROOT / 'build/ordinary-value-review-progress'
read = old.read_json


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    old.atomic_json(path, body)


def event_message(line):
    try: value = json.loads(line)
    except ValueError: return line.strip()[:200]
    if isinstance(value, dict) and isinstance(value.get('event'), str):
        return value['event'][:200]
    return line.strip()[:200]


def runs():
    rows = []
    for path in sorted(FOLDER.glob('run-*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)[:15]:
        row = read(path)
        if row['state'] == 'running' and list(old.process_identity(row['pid'])) != row['process_identity']:
            row = {**row, 'state': 'interrupted_or_exited'}
        rows.append(row)
    return rows


def snapshot():
    # Historical cards are already sealed. Rebuilding every nested historical
    # observer on a three-second refresh repeats expensive ancestor validation.
    historical = read(ROOT / 'build/ordinary-value-dashboard-final.json') or {}
    historical_cards = historical.get('cards', [])
    baseline = read(ROOT / 'build/ordinary-value-review-baseline.json') or {}
    overlap = read(ROOT / 'build/ordinary-value-overlap-review.json') or {}
    arena = read(ROOT / 'build/ordinary-value-loss-review.json') or {}
    validation = read(ROOT / 'build/ordinary-value-validation-review.json') or {}
    decision = read(ROOT / 'build/ordinary-value-review-conclusions.json') or {}
    accepted = read(ROOT / 'build/ordinary-value-review-acceptance.json') or {}
    cards = []
    def add(key, title, result, evidence, complete=False, improvement=''):
        card = old.card('ordinary-value-review-' + key, title, 'complete' if complete else 'running', result, evidence, improvement)
        card['needs_improvement'] = not complete
        cards.append(card)
    add('baseline', 'Protected models and prior evidence',
        str(baseline.get('inputs_count', 0)) + '/3779 prior files preserved. Policy, trunk and value weights fixed. No new training, teacher, PUCT or strength games.',
        'build/ordinary-value-review-baseline.json; docs/ORDINARY_VALUE_REVIEW_SCOPE.md', bool(baseline),
        'This is consumed-development diagnosis. The previous HOLD and H_v1 default remain unchanged.')
    osummary = overlap.get('summary', {})
    add('overlap', 'What triggered the overlap hold',
        'Rebuilding admitted/drawn, exact full-state, swapped full-state, store-offset pattern and family matches.' if not overlap else
        json.dumps(osummary, sort_keys=True) + '; prior control draw exposure confirmed: ' +
        str(overlap['ancestral_exposure']['matching_row_positive_prior_draw_exposure']),
        'build/ordinary-value-overlap-review.json', bool(overlap),
        'Zero current draws does not erase ancestral exposure. Identical encoder patterns remain relevant; the original admitted-pool guard is preserved.')
    hv = arena.get('summary', {}).get('H_v1', {})
    add('arena', 'All H_v1 losses and matched margins',
        'Reviewing all96 saved games and both seats, including unchanged losses and H8 relay context.' if not arena else
        'H_v1 control WDL ' + str(hv.get('control_wdl')) + '; candidate ' + str(hv.get('candidate_wdl')) +
        '; mean margin change ' + str(hv.get('paired_mean_candidate_minus_control')) +
        '; identical trajectories ' + str(hv.get('identical_action_trajectories')) + '/24.',
        'build/ordinary-value-loss-review.json', bool(arena), 'WDL and seed margins measure different things; all families remain in the denominator.')
    add('divergence', 'Shared-prefix decisions and timing ambiguity',
        'Locating every first divergent saved move and exact-state reference coverage.' if not arena else
        'All-baseline first divergences ' + str(arena['counts'].get('first_divergences')) +
        '; exact saved Q covered ' + str(arena['counts'].get('indexed_Q_covered_divergences')) +
        '; uncovered ' + str(arena['counts'].get('indexed_Q_uncovered_divergences')) + '.',
        'build/ordinary-value-loss-review.json: seats, families, label_index_scope', bool(arena),
        'Different visits, value estimates and opponent search depths coexist in timed games. Saved traces do not isolate a value-caused blunder.')
    vs = validation.get('summary', {})
    add('validation', 'Six-root value errors and horizon sensitivity',
        'Recomputing player0/actor errors, head-shift algebra and fixed-prediction H8/H10/H12 sensitivity.' if not validation else
        'H10 MAE ' + str(vs.get('H10_control_mae')) + ' -> ' + str(vs.get('H10_candidate_mae')) +
        '; improved/worsened ' + str(vs.get('improved_roots')) + '/' + str(vs.get('worsened_roots')) +
        '; frozen policy ' + str(vs.get('policy_logits_identical')) + '.',
        'build/ordinary-value-validation-review.json', bool(validation),
        'All high-band validation actors are player1: phase and actor are confounded. Deeper finite labels are sensitivity checks, not exact truth.')
    add('sampling', 'Actual draws, noisy outcomes and support',
        'Reconstructing12800 draws and24 paired starts; target-blind nearest support is descriptive.' if not validation else
        '12800 draws:3200 high /3200 low /6400 rehearsal. Self-play support24 starts; ' +
        str(vs.get('pairs_with_different_returns')) + '/24 pairs differ; median return gap ' +
        str(vs.get('median_pair_gap_seeds')) + ', maximum ' + str(vs.get('max_pair_gap_seeds')) +
        ' seeds. Five target-blind rehearsal neighbors per validation root. No weights changed.',
        'build/ordinary-value-validation-review.json: exposure, ordinary_training_pairs, coverage', bool(validation),
        'Two dependent outcomes per start do not establish calibrated terminal targets. Similar pit counts do not guarantee similar values.')
    add('decision', 'Next gates before more training',
        decision.get('summary', 'Review recommendations pending; further training and collection remain held.'),
        'build/ordinary-value-review-conclusions.json; docs/ORDINARY_VALUE_REVIEW_RESULTS.md', bool(decision),
        'Any revised overlap policy or controlled diagnostic is prospective. No retrospective filtering, candidate selection or promotion.')
    stale = old.current_evidence(ROOT, accepted) if accepted.get('passed') else []
    add('audit', 'Independent arithmetic, corruption and completion checks',
        str(accepted.get('tests_run', 0)) + ' source-bound current checks; old3779 pins preserved.' if accepted.get('passed') and not stale else
        'Independent auditor checking raw games, transforms, draws, model math and immutable review reruns.',
        'build/ordinary-value-review-audit.json; build/ordinary-value-review-acceptance.json', bool(accepted.get('passed') and not stale),
        'Read-only publication recovery is tested in new fixture paths; old engine interruption tests are not counted again.')
    # A finished investigation can still leave a model or validation problem.
    if overlap: cards[1]['needs_improvement'] = True
    if arena:
        cards[2]['needs_improvement'] = hv['paired_mean_candidate_minus_control'] < 0
        cards[3]['needs_improvement'] = arena['counts']['indexed_Q_uncovered_divergences'] > 0
    if validation:
        cards[4]['needs_improvement'] = vs['H10_mae_delta'] > 0
        cards[5]['needs_improvement'] = vs.get('pairs_with_different_returns', 0) > 0
    if decision: cards[6]['needs_improvement'] = True
    recent = runs()
    active = [row for row in recent if row['state'] == 'running']
    activity = read(FOLDER / 'activity.json') or dict(state='running', message='Investigating saved overlap, H_v1 losses and H10 errors; policy fixed and training held.')
    current = dict(state='running', message=' | '.join(row['tool'] + ': ' + row['last_event'] for row in active)) if active else activity
    return {'schema': 'ordinary-value-review-progress-v1', 'observed_at': old.utc(),
            'tools': cards + historical_cards, 'recent_runs': recent,
            'active_tools': active, 'current_work': current,
            'scope': 'Saved-data review only. No new training, teacher/PUCT searches, self-play or strength games. Previous HOLD retained.'}


def run_command(command, tool):
    FOLDER.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    path, log = FOLDER / ('run-' + uid + '.json'), FOLDER / ('run-' + uid + '.log')
    started = time.monotonic()
    row = dict(tool=tool, command=command, pid=os.getpid(), process_identity=list(old.process_identity(os.getpid())),
               state='running', started_at=old.utc(), updated_at=old.utc(), last_event='Launching', log=str(log.relative_to(ROOT)))
    save(path, row)
    child = None
    try:
        with log.open('w', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding='utf-8', errors='replace')
            for line in child.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
                row.update(last_event=event_message(line), updated_at=old.utc()); save(path, row)
            code = child.wait()
        row.update(state='complete' if code == 0 else 'failed', returncode=code,
                   elapsed_seconds=time.monotonic() - started, updated_at=old.utc()); save(path, row)
        return code
    except BaseException:
        if child is not None and child.poll() is None: child.terminate(); child.wait()
        row.update(state='interrupted', elapsed_seconds=time.monotonic() - started, updated_at=old.utc()); save(path, row)
        raise


def serve(port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path == '/': data = (ROOT / 'tools/ordinary-value-review-dashboard.html').read_bytes(); kind = 'text/html; charset=utf-8'
                elif self.path == '/api/status': data = config.canonical(snapshot()).encode(); kind = 'application/json'
                else: self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type', kind); self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers(); self.wfile.write(data)
            except Exception as exc: self.send_error(503, str(exc))
        def log_message(self, *_): pass
    class Server(ThreadingHTTPServer): allow_reuse_address = False
    print('Saved value review observer http://127.0.0.1:' + str(port) + '/', flush=True)
    Server(('127.0.0.1', port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    sub.add_parser('snapshot'); sub.add_parser('serve').add_argument('--port', type=int, default=18766)
    activity = sub.add_parser('activity'); activity.add_argument('--message', required=True); activity.add_argument('--state', choices=('running', 'complete', 'idle'), default='running')
    runner = sub.add_parser('run'); runner.add_argument('--tool', default='ordinary-value-review'); runner.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.operation == 'snapshot': print(config.canonical(snapshot()), flush=True)
    elif args.operation == 'serve': serve(args.port)
    elif args.operation == 'activity': save(FOLDER / 'activity.json', dict(state=args.state, message=args.message))
    else: return run_command(args.command[1:] if args.command[:1] == ['--'] else args.command, args.tool)


if __name__ == '__main__': raise SystemExit(main())
