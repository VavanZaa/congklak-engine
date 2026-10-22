"""Finite, recoverable review of sealed benchmark evidence; no engine imports.

The only executed game code is the independent literal rules interpreter.
Saved Monte Carlo Q values remain estimates, never teacher labels or regret.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'build/benchmark-loss-review-v1/analysis'
OLD = 'build/prospective-benchmark-v5'
SOURCE = 'congklak/benchmark_loss_review_v1.py'
SCHEMA = 'benchmark-loss-review-v1'
SEALED_CLOSURE_SHA = '680c60b60f7cd5a2e7fd4b869ceccc4d1be1a272f4b699dfd748a371391f4423'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(body):
    return (json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def reference(path):
    return dict(path=relative(path), sha256=sha(path))


def checked(ref):
    path = (ROOT / ref['path']).resolve()
    need(path.is_relative_to(ROOT), 'Input escaped workspace')
    need(sha(path) == ref['sha256'], 'Input hash differs: ' + ref['path'])
    return path


def atomic(path, body, replace=False):
    path = Path(path).resolve()
    need(path.is_relative_to(HOME.resolve()), 'Write outside new analysis namespace')
    value = canonical(body)
    if path.exists() and not replace:
        need(path.read_bytes() == value, 'Immutable output differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _closure():
    closure_path = ROOT / OLD / 'audit/closure.json'
    need(sha(closure_path) == SEALED_CLOSURE_SHA, 'Independent sealed closure identity differs')
    closure = read(closure_path)
    inventory_path = checked(closure['inventory_ref'])
    inventory = {}
    with inventory_path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                need(row['path'] not in inventory, 'Duplicate sealed inventory input')
                inventory[row['path']] = row
    return closure_path, closure, inventory


def freeze():
    """Bind every saved archive to the independently sealed byte inventory."""
    manifest_path = HOME / 'manifest.json'
    need(not manifest_path.exists(), 'Manifest already frozen; use run to resume')
    closure_path, closure, inventory = _closure()
    pins = {}
    for name in ('plan.json', 'report.json', 'audit/arena-games.json', 'audit/audit.json',
                 'jobs/arena/identity.json', 'jobs/arena/catalog.json'):
        path = OLD + '/' + name
        row = inventory[path]
        ref = dict(path=path, sha256=row['sha256'])
        checked(ref)
        pins[path] = ref['sha256']
    plan = read(ROOT / OLD / 'plan.json')
    games = read(ROOT / OLD / 'audit/arena-games.json')['games']
    schedule = plan['schedules']['arena']['rows']
    need(len(games) == len(schedule) == 96, 'Require complete frozen 96 games')
    rows = []
    for index, (game, row) in enumerate(zip(games, schedule)):
        need(game['game'] == row['index'] == index, 'Finite game schedule differs')
        refs = []
        for move in range(game['choices']):
            path = OLD + f'/jobs/arena/archives/game-{index:03d}-{move:03d}.json'
            item = inventory[path]
            refs.append(dict(path=path, sha256=item['sha256']))
        actual_names = {p for p in inventory if p.startswith(OLD + f'/jobs/arena/archives/game-{index:03d}-')}
        need(actual_names == {r['path'] for r in refs}, 'Unaccounted or missing game archive')
        need(refs[0] == game['first_archive_ref'] and refs[-1] == game['final_archive_ref'], 'Audited archive endpoints differ')
        rows.append(dict(index=index, schedule=row, audited=game, archives=refs))
    for path in (SOURCE, 'tests/reference.py', 'tests/test_benchmark_loss_review_v1.py'):
        pins[path] = sha(ROOT / path)
    for path, h in plan['models'].items():
        need(sha(ROOT / path) == h, 'Frozen model differs')
        pins[path] = h
    body = dict(schema=SCHEMA + '-manifest', source_inputs=pins,
                predecessor_closure_ref=reference(closure_path), inventory_ref=closure['inventory_ref'],
                models_fixed=True, policy_fixed=True, training_authorized=False,
                saved_evidence_only=True, raw_trace_payloads_read=False,
                new_native_queries=0, new_model_forwards=0, new_teacher_labels=0,
                games=rows, finite_games=96,
                limitations=['Known synthetic 12-root joint cohort; one ownership unit.',
                             'No proven natural opening reachability or independent strength estimate.',
                             'Saved PUCT Q values are estimates; no best-action regret is inferred.',
                             'An unfavorable initial store margin is not proof that a position is lost.',
                             'After active seeds fall to 24 the candidate seat uses the shared control.'])
    atomic(manifest_path, body)
    return dict(manifest_ref=reference(manifest_path), games=len(rows), archive_inputs=sum(len(g['archives']) for g in rows))


def admit(manifest_path, verify_archives=True):
    body = read(manifest_path)
    need(body['schema'] == SCHEMA + '-manifest' and body['finite_games'] == 96 and len(body['games']) == 96,
         'Finite review manifest differs')
    need(body['models_fixed'] and body['policy_fixed'] and not body['training_authorized'], 'Fixed review scope differs')
    for path, h in body['source_inputs'].items():
        checked(dict(path=path, sha256=h))
    closure = read(checked(body['predecessor_closure_ref']))
    need(body['predecessor_closure_ref']['sha256'] == SEALED_CLOSURE_SHA, 'Review predecessor is not the fixed sealed closure')
    need(closure['inventory_ref'] == body['inventory_ref'], 'Sealed inventory binding differs')
    checked(body['inventory_ref'])
    if verify_archives:
        for game in body['games']:
            for ref in game['archives']:
                checked(ref)
    return body


def literal_module():
    name = '_loss_review_literal_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/reference.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def position(state, literal):
    return literal.Position(7, tuple(state['pits'][0] + [state['stores'][0]] + state['pits'][1] + [state['stores'][1]]), state['controller'])


def raw(position):
    c = position.counts
    return dict(pits=[list(c[:7]), list(c[8:15])], stores=[c[7], c[15]], controller=position.controller)


def active(state):
    return sum(map(sum, state['pits']))


def margin(state, seat):
    return state['stores'][seat] - state['stores'][1 - seat]


def analyze_game(row, literal):
    schedule, audited = row['schedule'], row['audited']
    seat, profile, baseline = schedule['seat'], schedule['profile'], schedule['baseline']
    state = schedule['meta']['state']
    root = state
    need(root == audited['root_state'] and len(row['archives']) == audited['choices'], 'Audited game root/count differs')
    moves = []
    locked_loss = None
    for ply, ref in enumerate(row['archives']):
        archive = read(checked(ref))
        expected_id = f"game/{row['index']:03d}/{ply:03d}"
        need(archive['decision_id'] == expected_id and archive['state'] == state, 'Saved decision chain differs')
        answer = archive['answer']
        choice = answer['decision']['chosen']
        p = position(state, literal)
        need(choice in literal.legal_actions(p), 'Saved move illegal')
        transition = literal.apply(p, choice)
        after = raw(transition.position)
        saved_transition = dict(after=after, actor=transition.actor, bonus=transition.bonus,
                                captured=transition.captured, deposits=transition.deposits,
                                relay_pickups=transition.relay_pickups, terminal=transition.terminal,
                                passes=[list(v) for v in transition.final_passes])
        need(archive['transition'] == saved_transition and sum(p.counts) == 98, 'Literal archived transition differs')
        n = active(state)
        learned_turn = transition.actor == seat
        expected_engine = (profile if n > 24 else 'control') if learned_turn else baseline
        need(archive['engine'] == expected_engine, 'Saved branch/opponent assignment differs')
        report = answer['decision'].get('report') or {}
        delta = (answer.get('evaluator') or {}).get('delta', {})
        lower, upper = margin(state, seat) - n, margin(state, seat) + n
        if upper < 0 and locked_loss is None:
            locked_loss = ply
        moves.append(dict(ply=ply, archive_ref=ref, state=state, after=after, active=n,
                          actor=transition.actor, engine=archive['engine'], learned_seat_turn=learned_turn,
                          candidate_high_branch=archive['engine'] == 'candidate', action=choice,
                          legal=list(literal.legal_actions(p)), captured=transition.captured,
                          bonus=transition.bonus, relay_pickups=transition.relay_pickups,
                          deposits=transition.deposits, store_margin=margin(state, seat),
                          final_margin_arithmetic_bounds=[lower, upper],
                          branch_calls=delta.get('branch_calls', 0), branch_bypasses=delta.get('branch_bypasses', 0),
                          full_clock_ms=archive['clock']['critical_ms'],
                          completed_simulations=answer['decision'].get('completed_simulations'),
                          completed_depth=answer['decision'].get('completed_depth'),
                          search_value_estimate_player0=report.get('estimated_value_player0'),
                          action_q_estimates_player0=report.get('action_mean_value_player0'),
                          root_priors=report.get('root_priors'), visits=report.get('visits'),
                          proof_status=report.get('proof_status'), stop_reason=answer['decision'].get('stop_reason')))
        state = after
    need(state == audited['final'] and active(state) == 0 and margin(state, seat) == audited['margin'], 'Audited terminal outcome differs')
    start_margin = margin(root, seat)
    result = dict(game=row['index'], profile=profile, baseline=baseline, seat=seat,
                  root_id=schedule['meta']['id'], state_key=schedule['meta']['state_key'],
                  stratum=schedule['meta']['benchmark_stratum'], root_state=root,
                  root_active=active(root), root_actor=root['controller'],
                  profile_acts_at_root=root['controller'] == seat,
                  initial_store_margin=start_margin,
                  initial_margin_arithmetic_bounds=[start_margin-active(root), start_margin+active(root)],
                  initial_unfavorable_store_balance=start_margin < 0,
                  initial_loss_mathematically_locked=start_margin+active(root) < 0,
                  final=state, margin=margin(state, seat), outcome='win' if margin(state, seat) > 0 else 'loss' if margin(state, seat) < 0 else 'draw',
                  first_arithmetic_locked_loss_ply=locked_loss,
                  candidate_branch_decisions=sum(m['candidate_high_branch'] for m in moves),
                  candidate_value_branch_calls=sum(m['branch_calls'] for m in moves),
                  shared_control_suffix_decisions=sum(m['engine']=='control' and m['active']<=24 for m in moves),
                  learned_seat_captures=sum(m['captured'] for m in moves if m['learned_seat_turn']),
                  opponent_captures=sum(m['captured'] for m in moves if not m['learned_seat_turn']),
                  opponent_store_gains=state['stores'][1-seat]-root['stores'][1-seat],
                  learned_store_gains=state['stores'][seat]-root['stores'][seat],
                  decisions=len(moves), moves=moves,
                  best_action_regret=None, theoretical_root_value=None,
                  causal_attribution='Unproven; observed outcomes and literal paths only.')
    return result


def load_receipts(output, manifest_ref, manifest=None):
    folder = output / 'receipts'
    paths = sorted(folder.glob('*.json')) if folder.exists() else []
    receipts = []
    previous = None
    for index, path in enumerate(paths):
        body = read(path)
        need(body['schema'] == SCHEMA+'-game-receipt' and body['manifest_ref'] == manifest_ref and
             body['index'] == index and body['previous_sha256'] == previous and body['result']['game'] == index,
             'Receipt prefix corrupt, duplicated or mismatched')
        h = sha(path)
        need(path.name == f'{index:03d}-{h}.json' and path.read_bytes() == canonical(body), 'Receipt filename/bytes differ')
        if manifest is not None:
            row = manifest['games'][index]
            schedule, audited, result = row['schedule'], row['audited'], body['result']
            need(body['input_archive_refs'] == row['archives'] and
                 result['root_state'] == audited['root_state'] == schedule['meta']['state'] and
                 result['profile'] == audited['profile'] == schedule['profile'] and
                 result['baseline'] == audited['baseline'] == schedule['baseline'] and
                 result['seat'] == audited['seat'] == schedule['seat'] and
                 result['margin'] == audited['margin'] and result['final'] == audited['final'] and
                 result['decisions'] == audited['choices'] == len(result['moves']) and
                 [move['archive_ref'] for move in result['moves']] == row['archives'],
                 'Cached receipt differs from frozen game semantics')
        previous = h
        receipts.append(body)
    need(len(receipts) <= 96, 'Review exceeded finite schedule')
    progress_path = output/'progress.json'
    if progress_path.exists():
        progress = read(progress_path)
        committed = progress['completed']
        need(progress['manifest_ref'] == manifest_ref and type(committed) is int and
             0 <= committed <= len(receipts), 'Published committed prefix is missing or stale')
        expected = hashlib.sha256(canonical(receipts[committed-1])).hexdigest() if committed else None
        need(progress['last_committed_sha256'] == expected, 'Published prefix hash differs')
    if (output/'result.json').exists():
        final = read(output/'result.json')
        need(len(receipts) == 96 and final['manifest_ref'] == manifest_ref and final['games'] == 96,
             'Published final result has an incomplete receipt prefix')
    return receipts


def bucket(rows, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        result.append(dict(zip(fields, key), games=len(values), wins=sum(r['margin'] > 0 for r in values),
                           draws=sum(r['margin'] == 0 for r in values), losses=sum(r['margin'] < 0 for r in values),
                           mean_margin=sum(r['margin'] for r in values)/len(values),
                           margins=[r['margin'] for r in values], game_ids=[r['game'] for r in values]))
    return result


def summarize(receipts, manifest_ref):
    rows = [r['result'] for r in receipts]
    need(len(rows) == 96, 'No partial aggregate may become final')
    pairs = defaultdict(dict)
    for row in rows:
        key = (row['state_key'], row['baseline'], row['seat'])
        need(row['profile'] not in pairs[key], 'Duplicate paired game')
        pairs[key][row['profile']] = row
    paired = []
    for key, pair in sorted(pairs.items()):
        need(set(pair) == {'control', 'candidate'}, 'Incomplete comparison pair')
        candidate, control = pair['candidate'], pair['control']
        divergence = None
        for c, b in zip(candidate['moves'], control['moves']):
            if c['state'] != b['state']:
                need(divergence is not None, 'Paths differ without earlier chosen action')
                break
            if c['action'] != b['action']:
                divergence = dict(ply=c['ply'], actor=c['actor'], active=c['active'],
                                  actor_is_learned_seat=c['learned_seat_turn'],
                                  candidate_engine=c['engine'], control_engine=b['engine'],
                                  candidate_action=c['action'], control_action=b['action'],
                                  candidate_archive_ref=c['archive_ref'], control_archive_ref=b['archive_ref'],
                                  same_legal_actions=c['legal']==b['legal'],
                                  candidate_search_value_estimate_player0=c['search_value_estimate_player0'],
                                  control_search_value_estimate_player0=b['search_value_estimate_player0'],
                                  candidate_simulations=c['completed_simulations'], control_simulations=b['completed_simulations'])
                break
        same_choices = divergence is None
        need(not same_choices or (candidate['decisions'] == control['decisions'] and candidate['final'] == control['final']), 'Identical path has different outcome')
        paired.append(dict(root_id=candidate['root_id'], state_key=key[0], baseline=key[1], seat=key[2],
                           stratum=candidate['stratum'], candidate_game=candidate['game'], control_game=control['game'],
                           candidate_margin=candidate['margin'], control_margin=control['margin'],
                           candidate_minus_control_margin=candidate['margin']-control['margin'],
                           candidate_outcome=candidate['outcome'], control_outcome=control['outcome'],
                           identical_action_path=same_choices, first_action_divergence=divergence,
                           candidate_branch_decisions=candidate['candidate_branch_decisions']))
    candidate_rows = [r for r in rows if r['profile']=='candidate']
    ordinary = [r for r in candidate_rows if r['stratum']!='boundary25']
    return dict(schema=SCHEMA+'-result', state='complete', manifest_ref=manifest_ref,
                games=96, paired_games=48, models_fixed=True, policy_fixed=True, optimizer_steps=0,
                native_queries=0, model_forwards=0, teacher_labels=0, raw_trace_payloads_read=0,
                same_known_joint_benchmark_unit=True, independently_owned_clusters=1,
                by_profile_baseline=bucket(rows, ['profile','baseline']),
                by_profile_stratum_baseline=bucket(rows, ['profile','stratum','baseline']),
                by_root_baseline_profile_seat=bucket(rows, ['root_id','stratum','baseline','profile','seat']),
                by_stratum_baseline_profile_root_actor=bucket(rows, ['stratum','baseline','profile','profile_acts_at_root']),
                paired=paired,
                candidate_losses=[{k:v for k,v in r.items() if k!='moves'} for r in candidate_rows if r['margin']<0],
                ordinary_candidate_games=[{k:v for k,v in r.items() if k!='moves'} for r in ordinary],
                summary=dict(candidate_games=48, candidate_losses=sum(r['margin']<0 for r in candidate_rows),
                             ordinary_games=len(ordinary), ordinary_losses=sum(r['margin']<0 for r in ordinary),
                             ordinary_unfavorable_store_starts=sum(r['initial_unfavorable_store_balance'] for r in ordinary),
                             ordinary_mathematically_locked_losses_at_start=sum(r['initial_loss_mathematically_locked'] for r in ordinary),
                             identical_pair_paths=sum(p['identical_action_path'] for p in paired),
                             ordinary_identical_pair_paths=sum(p['identical_action_path'] for p in paired if p['stratum']!='boundary25'),
                             paired_outcome_changes=sum(p['candidate_outcome'] != p['control_outcome'] for p in paired),
                             candidate_high_domain_decisions=sum(r['candidate_branch_decisions'] for r in candidate_rows)),
                teacher_regret_inferred=False, theoretical_root_values_inferred=False,
                interpretation='Finite saved-path diagnosis; model effects cannot be separated from timed-search allocation without new controlled tests.',
                promotion=False, next_product_action='Keep H_v1 default; expose analysis, move replay, alternatives and honest confidence before expanding training.')


def pause_hook(output, hook, index, manifest_ref):
    marker = dict(schema=SCHEMA+'-recovery-hook', hook=hook, game=index, pid=os.getpid(),
                  manifest_ref=manifest_ref, native_queries=0, model_forwards=0)
    atomic(output/'recovery-hook.json', marker)
    print(json.dumps(marker), flush=True)
    while True:
        time.sleep(.05)


def run(output, hook=None, pause_game=3, stop_after=None):
    output = Path(output).resolve()
    need(output.is_relative_to(HOME.resolve()), 'Output outside new analysis namespace')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = HOME/'manifest.json'
    body = admit(manifest_path)
    manifest_ref = reference(manifest_path)
    # Windows holds the file handle across the finite worker; another writer
    # cannot acquire this byte lock, including after an interrupted process.
    lock_path = output/'writer.lock'
    lock = lock_path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if lock_path.stat().st_size == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = load_receipts(output, manifest_ref, body)
        resumed_at = len(receipts)
        def progress():
            snapshot = dict(schema=SCHEMA+'-progress', state='complete' if len(receipts)==96 else 'running',
                            completed=len(receipts), total=96, resumed_at=resumed_at,
                            last_committed_sha256=hashlib.sha256(canonical(receipts[-1])).hexdigest() if receipts else None,
                            models_fixed=True, policy_fixed=True, native_queries=0, model_forwards=0,
                            output=relative(output), manifest_ref=manifest_ref)
            atomic(output/'progress.json', snapshot, replace=True)
            if output == (HOME/'production').resolve():
                atomic(HOME/'progress.json', snapshot, replace=True)
        progress()
        literal = literal_module()
        for index in range(resumed_at, 96):
            result = analyze_game(body['games'][index], literal)
            receipt = dict(schema=SCHEMA+'-game-receipt', manifest_ref=manifest_ref, index=index,
                           previous_sha256=sha(output/'receipts'/f"{index-1:03d}-{hashlib.sha256(canonical(receipts[-1])).hexdigest()}.json") if receipts else None,
                           input_archive_refs=body['games'][index]['archives'], result=result,
                           native_queries=0, model_forwards=0, optimizer_steps=0)
            if hook == 'before_commit' and index == pause_game:
                pause_hook(output, hook, index, manifest_ref)
            h = hashlib.sha256(canonical(receipt)).hexdigest()
            atomic(output/'receipts'/f'{index:03d}-{h}.json', receipt)
            receipts.append(receipt)
            progress()
            if hook == 'after_commit' and index == pause_game:
                pause_hook(output, hook, index, manifest_ref)
            print(json.dumps(dict(kind='game_reviewed', index=index, completed=len(receipts), total=96)), flush=True)
            if stop_after is not None and len(receipts) >= stop_after:
                break
        # Verify immutable sources and metadata again; archive bytes were checked
        # at entry and immediately before their per-game analysis.
        admit(manifest_path, verify_archives=False)
        reloaded = load_receipts(output, manifest_ref, body)
        need(len(reloaded) == len(receipts), 'Durable receipt prefix differs')
        if len(receipts) == 96:
            result = summarize(receipts, manifest_ref)
            atomic(output/'result.json', result)
            if output == (HOME/'production').resolve():
                atomic(HOME/'summary.json', result)
        return dict(state='complete' if len(receipts)==96 else 'checkpointed', completed=len(receipts),
                    resumed_at=resumed_at, newly_committed=len(receipts)-resumed_at,
                    native_queries=0, model_forwards=0, result_ref=reference(output/'result.json') if len(receipts)==96 else None)
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['freeze','run','status'])
    parser.add_argument('--output', default=str(HOME/'production'))
    parser.add_argument('--pause-hook', choices=['before_commit','after_commit'])
    parser.add_argument('--pause-game', type=int, default=3)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.operation == 'freeze':
        result = freeze()
    elif args.operation == 'run':
        result = run(args.output, args.pause_hook, args.pause_game, args.stop_after)
    else:
        manifest_ref = reference(HOME/'manifest.json')
        result = dict(completed=len(load_receipts(Path(args.output), manifest_ref, read(HOME/'manifest.json'))), total=96)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
