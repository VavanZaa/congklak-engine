"""Read-only saved-trace pilot loss review observer; all prior accepted sources stay immutable."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import time
import uuid
from . import config,progress as old,value_pilot_progress as previous
from .native import ROOT
FOLDER=ROOT/'build/pilot-loss-review-progress'
def read(path):return old.read_json(path)
def save(path,body):path.parent.mkdir(parents=True,exist_ok=True);old.atomic_json(path,body)

def runs():
    values=[]
    for path in sorted(FOLDER.glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[:15]:
        row=read(path)
        if row['state']=='running' and list(old.process_identity(row['pid']))!=row['process_identity']:row={**row,'state':'interrupted_or_exited'}
        values.append(row)
    return values

_PRIOR_CACHE=None
_PRIOR_AT=0.


def previous_snapshot():
    global _PRIOR_CACHE,_PRIOR_AT
    now=time.monotonic()
    if _PRIOR_CACHE is None or now-_PRIOR_AT>=15:
        _PRIOR_CACHE=previous.snapshot();_PRIOR_AT=now
    return _PRIOR_CACHE



def snapshot():
    base=previous_snapshot();cards=[];arena=read(ROOT/'build/pilot-loss-review-arena.json');ordinary=read(ROOT/'build/pilot-loss-review-ordinary.json');ready=read(ROOT/'build/engine-readiness-current.json')
    text='Replaying all 72 retained H8-relay games: pretraining control, current control and pilot on 12 families / 24 seats. No new searches, games, labels or training.'
    if arena:
        a=arena['aggregate'];text=f"Pilot W/D/L {a['pilot']['wdl']}; control {a['control']['wdl']}. Matched pilot-minus-control seed margin {a['pilot_minus_control_paired_mean_margin']:.4f}; same-control post-minus-pre drift {a['same_control_post_minus_pre_mean_margin']:.4f}. WDL clips loss severity; margins retain magnitude."
    cards.append(old.card('pilot-loss-review-arena','Saved H8-relay losses and paired margins','complete' if arena else 'running',text,
        'build/pilot-loss-review-arena.json; tools/analyze-pilot-losses.py',
        'All 12 families and both seats remain visible. Reused development and wall-clock decisions cannot establish a unique value-head blunder.'))
    text='Locating the first distinct saved action after a literal shared prefix, retaining actor/profile, visits/Q/prior/bootstrap, actual depth and physical cost. No invented teacher Q.'
    if arena:
        a=arena['aggregate'];c=arena['counts'];text=f"{c['first_divergent_seats']}/24 first-divergent seats; {c['opponent_first_divergences']} first divergences belong to H8 relay. Stored exact-full-state Q covers {c['divergent_seats_with_exact_state_Q']} and leaves {c['divergent_seats_without_stored_Q']} uncovered. Worst families: "+' | '.join(str(row['pair'])+': '+row['family']+' delta '+str(row['pilot_minus_control_paired_mean_margin']) for row in arena['adverse_families_ranked'][:3])+f". Leave-one-family-out means {a['leave_one_family_out_min']:.4f} to {a['leave_one_family_out_max']:.4f}."
    cards.append(old.card('pilot-loss-review-divergence','Loss tails, timing variation and label gaps','complete' if arena else 'running',text,
        'build/pilot-loss-review-arena.json: families, seats, first_divergence, existing_Q_comparison',
        'Opponent-first means first divergence actor, not first game move. Nearby captures/bonuses/stores are observations. H8 names supervision horizon; actual search depths are retained.'))
    text='Reviewing all 14 ordinary validation roots, 61 ordinary optimizer roots, 8 diagnostics and 259 retained ordinary development roots with fixed CPU model forwards.'
    if ordinary:
        g=ordinary['groups']['ordinary_validation'];all_rows=g['summary']['all'];text=f"Ordinary validation {all_rows['rows']} roots: native-forward MAE {all_rows['control_mae_seeds']:.4f} -> {all_rows['pilot_mae_seeds']:.4f}, {all_rows['improved']} improved/{all_rows['worse']} worse. "+' | '.join(band+': '+str(row['rows'])+' roots, delta '+str(round(row['delta_mae_seeds'],4)) for band,row in g['summary']['by_band'].items())
    cards.append(old.card('pilot-loss-review-ordinary','Ordinary error and actual draw exposure','complete' if ordinary else 'running',text,
        'build/pilot-loss-review-ordinary.json',
        'Small mixed-horizon cohorts and uncalibrated exposure factors remain descriptive; analysis does not apply new weights, targets or optimizer steps.'))
    data=ready.get('larger_data',{}) if ready else {};product=ready.get('product',{}) if ready else {}
    text='Readiness assessment pending.' if not ready else data['engineering']+'; learning: '+data['learning_evidence']+'. '+data['next_gate']
    cards.append(old.card('pilot-loss-review-data-readiness','Larger data: bounded engineering versus payoff','held' if ready else 'running',text,
        'build/engine-readiness-current.json; docs/ENGINE_READINESS_CURRENT.md',
        'Existing single-writer versioned collection can expand in bounded chunks. Parallel/sharded throughput and blind 100k/long training are not established. Protect latest evaluation trajectories and ownership first.'))
    text='Readiness assessment pending.' if not ready else 'Internal analyzer: '+product['internal_analyzer']+'. Finished strong analyzer: '+product['finished_strong_analyzer']+'. Default: '+product['default']+'. Open blocks: '+', '.join(product['blocks'])+'.'
    cards.append(old.card('pilot-loss-review-product-readiness','Finished engine: usable prototype and open gates','held' if ready else 'running',text,
        'build/engine-readiness-current.json; docs/ENGINE_READINESS_CURRENT.md',
        'Separate usable H_v1 analysis from validated learned strength, fresh release validation, unified exact/estimated modes and portable packaging. No invented completion percentage.'))
    accepted=read(ROOT/'build/pilot-loss-review-acceptance.json');stale=old.current_evidence(ROOT,accepted) if accepted and accepted.get('passed') else [];okay=bool(accepted and accepted.get('passed') and not stale)
    cards.append(old.card('pilot-loss-review-audit','Independent review and read-only completion','complete' if okay else 'stale' if stale else 'running',
        str(accepted.get('tests_run'))+' unique checks passed; all 2600 prior artifact pins preserved. No new training/search/game.' if okay else 'Auditing literal matched games, exact label states, raw model forwards, draw replay, readiness facts and stable publication.',
        'build/pilot-loss-review-audit.json; build/pilot-loss-review-acceptance.json; docs/PILOT_LOSS_REVIEW_RESULTS.md',
        'The review proposes next gates only. It starts no collection, training or release and never promotes the pilot.'))
    for card in cards:card['needs_improvement']=card['state']!='complete'
    if arena and arena['counts']['divergent_seats_without_stored_Q']:cards[1]['needs_improvement']=True
    if ordinary and ordinary['groups']['ordinary_validation']['summary']['all']['delta_mae_seeds']>0:cards[2]['needs_improvement']=True
    recent=runs();active=[row for row in recent if row['state']=='running'];activity=read(FOLDER/'activity.json')
    current=dict(state='running',message=' | '.join(row['tool']+' - '+row['last_event'] for row in active)) if active else activity or dict(state='running',message='Reviewing saved pilot losses and ordinary errors before any further training.')
    return {**base,'schema':'pilot-loss-review-progress-v1','observed_at':old.utc(),'tools':cards+base['tools'],'recent_runs':recent+base['recent_runs'][:5],'active_tools':active,'current_work':current,
        'scope':'Read-only pilot loss review and readiness assessment. Fixed policy/models, saved 72 H8 games, no new games/searches/teacher labels/training. Bounded collection engineering and finished-engine release gates are separate.'}


def run_command(command,tool):
    FOLDER.mkdir(parents=True,exist_ok=True);uid=uuid.uuid4().hex;path=FOLDER/f'run-{uid}.json';log=FOLDER/f'run-{uid}.log';started=time.monotonic();child=None
    import os
    row=dict(tool=tool,command=command,pid=os.getpid(),process_identity=list(old.process_identity(os.getpid())),state='running',started_at=old.utc(),updated_at=old.utc(),last_event='Launching',log=str(log.relative_to(ROOT)))
    save(path,row)
    try:
        with log.open('w',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
                try:message=old.describe(json.loads(line))
                except (ValueError,KeyError,TypeError):message=line.strip()[:300]
                row.update(last_event=message,updated_at=old.utc());save(path,row)
            code=child.wait();row.update(state='complete' if code==0 else 'failed',exit_code=code,updated_at=old.utc(),elapsed_seconds=time.monotonic()-started);save(path,row);return code
    except BaseException:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        row.update(state='interrupted',updated_at=old.utc(),elapsed_seconds=time.monotonic()-started);save(path,row);raise

def serve(port=18766):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path=='/':data=(ROOT/'tools/pilot-loss-review-dashboard.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/api/status':data=config.canonical(snapshot()).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'");self.end_headers();self.wfile.write(data)
            except Exception as exc:self.send_error(503,str(exc))
        def log_message(self,*_):pass
    class Server(ThreadingHTTPServer):allow_reuse_address=False
    print(f'Pilot loss review observer http://127.0.0.1:{port}/',flush=True);Server(('127.0.0.1',port),Handler).serve_forever()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True);sub.add_parser('serve').add_argument('--port',type=int,default=18766);sub.add_parser('snapshot')
    activity=sub.add_parser('activity');activity.add_argument('--message',required=True);activity.add_argument('--state',choices=('running','complete','idle'),default='running')
    runner=sub.add_parser('run');runner.add_argument('--tool',default='pilot-loss-review');runner.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.operation=='serve':serve(args.port)
    elif args.operation=='snapshot':print(config.canonical(snapshot()),flush=True)
    elif args.operation=='activity':save(FOLDER/'activity.json',dict(state=args.state,message=args.message))
    else:return run_command(args.command[1:] if args.command[:1]==['--'] else args.command,args.tool)

if __name__=='__main__':raise SystemExit(main())
