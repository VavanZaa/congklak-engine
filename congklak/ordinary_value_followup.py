"""One preregistered, data-only self-play allocation after the value pilot.

Uses the sealed ordinary producer's pure history/closure/replay validators with
explicit roots. It never changes that producer's globals or saved artifacts.
"""
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from . import config, ordinary_selfplay_pilot as old, phase6_store, pilot
from .native import ROOT, Native, flatten
from .puct_cpu_inference import CpuBatchEvaluator

SCHEMA = 'ordinary-value-conditional-data-followup-v1'
DEFAULT = ROOT / 'data/ordinary-value-followup-v1'
PLAN = ROOT / 'build/ordinary-value-followup-plan.json'
IMPLEMENTATION = ROOT / 'build/ordinary-value-followup-implementation.json'
OUTPUT = ROOT / 'build/ordinary-value-followup.json'
BASELINE = ROOT / 'build/ordinary-value-baseline.json'
PROJECTION = ROOT / 'build/ordinary-value-benchmark-projection.json'
SOURCES = ('congklak/ordinary_value_followup.py', 'tools/ordinary-value-followup.py',
           'docs/ORDINARY_VALUE_SCOPE.md')
DEPENDENCIES = ('congklak/ordinary_selfplay_pilot.py', 'congklak/selfplay_training_pilot.py',
                'congklak/ordinary_value_pilot.py', 'congklak/ordinary_value_benchmark.py',
                'congklak/quality_search.py', 'congklak/puct_cpu_inference.py',
                'congklak/phase6_store.py', 'congklak/features_relay.py')
CONTRACT = dict(prefix_seed=20261030, rollout_seed=20261031, prefixes=512,
                new_start_caps={kind: 12 for kind in old.STRATA},
                repeat_high='Every original12 high-band training start; two extra games each',
                branch_rule='Engineering failure holds; clean+all6H10+MAE<=control+both matched means>=0 gives new_starts; otherwise repeat_high',
                root_model_sha256=old.MODEL_SHA, simulations=64, max_nodes=2048,
                max_choices=512, rawstate_cap=4, previous_observations_consume_cap=True,
                automatic_retraining=False, promotion=False)
require, read, sha, publish = old.require, old.read, old.sha, old.publish


def plan():
    baseline = read(BASELINE)
    require(baseline['passed'] and len(baseline['inputs']) == 3364, 'Wrong followup baseline')
    body = dict(schema=SCHEMA, contract=CONTRACT, baseline_sha256=sha(BASELINE),
                projection_sha256=sha(PROJECTION), rules_digest=config.RULES_DIGEST)
    publish(PLAN, body)
    return body


def freeze():
    plan()
    body = dict(schema=SCHEMA, plan_sha256=sha(PLAN),
                sources={name: sha(ROOT / name) for name in SOURCES},
                dependencies={name: sha(ROOT / name) for name in DEPENDENCIES})
    publish(IMPLEMENTATION, body)
    return body


def admit():
    value = read(PLAN)
    require(value == dict(schema=SCHEMA, contract=CONTRACT, baseline_sha256=sha(BASELINE),
                          projection_sha256=sha(PROJECTION), rules_digest=config.RULES_DIGEST), 'Followup plan changed')
    require(read(IMPLEMENTATION) == dict(schema=SCHEMA, plan_sha256=sha(PLAN),
                sources={name: sha(ROOT / name) for name in SOURCES},
                dependencies={name: sha(ROOT / name) for name in DEPENDENCIES}), 'Followup source changed')
    for name, digest in read(BASELINE)['inputs'].items():
        require(sha(ROOT / name) == digest, 'Prior input changed: ' + name)


def allocation(fixture=False):
    if fixture:
        return dict(branch='repeat_high', fixture=True, reason='Explicit first-original-high-root recovery fixture')
    from . import ordinary_value_benchmark as benchmark
    decision = benchmark.allocation_decision()
    proofs = {}
    for stage in ('preflight', 'labels', 'model', 'recovery', 'benchmark'):
        path = ROOT / 'build/ordinary-value-audit' / (stage + '.json')
        body = read(path)
        require(body['passed'] is True and body['errors'] == body['failures'] == body['skips'] == 0,
                'Followup requires independent stage: ' + stage)
        for name, digest in body['inputs'].items():
            require(sha(ROOT / name) == digest, 'Followup audit input changed: ' + name)
        proofs[path.relative_to(ROOT).as_posix()] = sha(path)
    require(decision['branch'] in ('new_starts', 'repeat_high', 'hold'), 'Unknown allocation branch')
    return dict(decision=decision, branch=decision['branch'], independent_stages=proofs,
                benchmark_sha256=sha(ROOT / 'build/ordinary-value-benchmark.json'))


def registry(fixture=False):
    previous = read(old.PROTECTION)
    projection = read(PROJECTION)
    q, f = set(previous['quotients']), set(previous['families'])
    q.update(projection['reserved_quotients']); f.update(projection['reserved_families'])
    trajectories = []
    if not fixture:
        native = Native()
        games = old.db_rows('data/ordinary-value-benchmark-v1/work.sqlite3', 'games', 'idx')
        require(len(games) == 96, 'Followup needs all96 benchmark trajectories')
        for index, game in games.items():
            p = game['start']; f.add(game['family']); states = []
            for action in [*game['moves'], None]:
                p = old.previous.normalize(p, native)
                quotient = pilot.quotient(p).hex(); q.add(quotient)
                states.append(dict(state_key=old.previous.rawkey(p), quotient=quotient))
                if action is not None:
                    result = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                    require(result['ok'], 'Illegal new protected arena trajectory'); p = result['position']
            require(p == game['final'], 'Protected arena final differs')
            trajectories.append(dict(index=index, game_sha256=config.digest(game), family=game['family'], states=states))
    prior = read(old.REGISTRY)
    nq, nf = set(prior['novelty_quotients']) | q, set(prior['novelty_families']) | f
    pool = old.db_rows('data/ordinary-selfplay-pilot-v1/work.sqlite3', 'families', 'idx')
    for family in pool.values():
        nf.add(family['family']); nq.update(row['quotient'] for row in family['candidates'])
    return dict(true_evaluation_quotients=sorted(q), true_evaluation_families=sorted(f),
                novelty_quotients=sorted(nq), novelty_families=sorted(nf),
                latest96_trajectories=trajectories, fixture=fixture)


def identity_for(fixture=False):
    admit(); choice = allocation(fixture); roles = registry(fixture)
    settings = json.loads(config.canonical(old.CONTRACT))
    settings.update(seed=CONTRACT['prefix_seed'], families=32 if fixture else CONTRACT['prefixes'],
                    caps={'training_candidate': {kind: 12 for kind in old.STRATA},
                          'validation': {kind: 0 for kind in old.STRATA}},
                    minimum_usable_roots={kind: 0 for kind in old.STRATA})
    settings['rollouts'].update(seed=CONTRACT['rollout_seed'], simulations=4 if fixture else 64)
    return dict(schema=SCHEMA, fixture=fixture, branch=choice['branch'], allocation=choice,
                roles=roles, settings=settings, rules_digest=config.RULES_DIGEST,
                model_sha256=sha(old.MODEL), native_build_sha256=Native().build_id,
                plan_sha256=sha(PLAN), implementation_sha256=sha(IMPLEMENTATION))


def explicit_starts(identity, pool):
    if identity['branch'] == 'new_starts':
        return old.select(pool, identity['settings'], identity['roles'])
    original = old.db_rows('data/ordinary-selfplay-pilot-v1/work.sqlite3', 'roots', 'state')
    roots = {key: row for key, row in original.items()
             if row['cohort'] == 'training_candidate' and row['stratum'] == '50-74/ordinary'}
    require(len(roots) == 12, 'Original12 high starts differ')
    if identity['fixture']:
        roots = dict(list(sorted(roots.items()))[:1])
    counts = {'training_candidate': {kind: sum(r['stratum'] == kind for r in roots.values()) for kind in old.STRATA},
              'validation': {kind: 0 for kind in old.STRATA}}
    roles = identity['roles']
    protection = dict(quotients=roles['true_evaluation_quotients'], families=roles['true_evaluation_families'],
                      new_validation_candidate_observations=0, new_validation_unique_quotients=0,
                      new_validation_family_ids=[], selected_validation_roots=0)
    return roots, dict(counts=counts, source='ALL original12 high starts; no outcome selection',
                       fixture_first_original_high_only=identity['fixture']), protection


def collection(db, identity, on_event=None, cancel=None):
    event = on_event or (lambda value: None)
    pool = old.table(db, 'families', 'idx')
    require(list(pool) == list(range(len(pool))), 'Followup prefix not contiguous')
    target = identity['settings']['families'] if identity['branch'] == 'new_starts' else 0
    require(len(pool) <= target, 'Followup pool over budget')
    if target:
        with CpuBatchEvaluator(old.MODEL, cache_capacity=0, max_batch=1) as evaluator:
            for index, family in pool.items():
                require(old.equal(family, old.generate_family(index, identity['settings'], Native(), evaluator)), 'Followup saved history differs')
            for index in range(len(pool), target):
                if cancel and cancel(): return None
                family = old.generate_family(index, identity['settings'], Native(), evaluator)
                event(dict(event='ordinary_value_followup_collection_precommit', completed=index, target=target))
                with db: db.execute('INSERT INTO families VALUES(?,?,?)', (index, config.canonical(family), config.digest(family)))
                pool[index] = family
                event(dict(event='ordinary_value_followup_collection_committed', completed=index + 1, target=target))
    roots, selection, protection = explicit_starts(identity, pool)
    body = dict(schema=SCHEMA, roots_sha256=config.digest(roots), families_sha256=config.digest(pool),
                branch=identity['branch'], selection=selection, rollout_protection=protection)
    saved = db.execute("SELECT body FROM meta WHERE key='collection_complete'").fetchone()
    if saved:
        require(old.equal(old.metadata(db, 'collection_complete'), body) and old.equal(old.table(db, 'roots', 'state'), roots), 'Followup ownership changed')
    else:
        require(not old.table(db, 'roots', 'state'), 'Orphan followup roots')
        with db:
            for key, row in roots.items(): db.execute('INSERT INTO roots VALUES(?,?,?)', (key, config.canonical(row), config.digest(row)))
            db.execute('INSERT INTO meta VALUES(?,?,?)', ('collection_complete', config.canonical(body), config.digest(body)))
    return roots, body


def replay(db, identity, roots, collected, closed=False):
    # The delegated validator must consume exactly these explicit roots. Its
    # old fixture=True path intentionally selects different fixture roots.
    delegate = {**identity, 'fixture': False}
    body, rows = old.replay_body(db, delegate, roots, collected, closed)
    historical = old.db_rows('data/ordinary-selfplay-pilot-v1/work.sqlite3', 'sp_rows', 'id')
    older = old.db_rows('data/selfplay-training-pilot-v1/work.sqlite3', 'sp_rows', 'id')
    used = Counter(row['state_key'] for row in [*historical.values(), *older.values()])
    by_state = defaultdict(list)
    for row in rows: by_state[row['state_key']].append(row)
    retained, pruned = [], []
    for key, values in by_state.items():
        values.sort(key=lambda row: (row['choice'] != 0, hashlib.sha256(('ordinary-value-followup-cap/' + row['id']).encode()).digest()))
        room = max(0, 4 - used[key]); retained.extend(values[:room]); pruned.extend(row['id'] for row in values[room:])
    retained = [dict(row, qualified_id='ordinary-value-followup-v1/' + row['id'],
                     dataset_namespace='ordinary-value-followup-v1') for row in retained]
    retained.sort(key=lambda row: row['id'])
    retained_ids = {row['id'] for row in retained}
    starts = old.starts_for(roots, delegate)
    usable = {}; usable_counts = Counter()
    for start in starts:
        ids = [old.previous.initial_game(start, index, identity['settings']['rollouts'], Native())['id'] + '/000'
               for index in range(2)]
        okay = all(key in retained_ids for key in ids)
        usable[start['root']['state_key']] = dict(family=start['root']['family'], stratum=start['root']['stratum'],
                usable=okay, required_record_ids=ids, root_sha256=start['root_sha256'])
        if okay: usable_counts[start['root']['stratum']] += 1
    support = {kind: dict(records=sum(row['stratum'] == kind for row in retained),
                        unique_rawstates=len({row['state_key'] for row in retained if row['stratum'] == kind}),
                        families=len({row['family'] for row in retained if row['stratum'] == kind}),
                        start_records=sum(row['stratum'] == kind and row['choice'] == 0 for row in retained),
                        descendant_records=sum(row['stratum'] == kind and row['choice'] > 0 for row in retained))
               for kind in old.STRATA}
    summary = dict(body['summary'], retained_rows=len(retained), unique_rawstates=len({r['state_key'] for r in retained}),
                   retained_families=len({r['family'] for r in retained}), cross_batch_cap_pruned=len(pruned),
                   cross_batch_cap_pruned_ids=sorted(pruned), actual_fixture=identity['fixture'],
                   support=support, usable_starts=usable,
                   usable_start_roots={kind: usable_counts[kind] for kind in old.STRATA},
                   future_use_ready=False, usable_start_gate_passed=False, validation_panel_complete=False,
                   pruned_duplicate_rows=body['summary']['pruned_duplicate_rows'] + len(pruned),
                   pruned_row_ids=sorted(body['summary']['pruned_row_ids'] + pruned),
                   prior_counts_sha256=config.digest(dict(sorted(used.items()))))
    return dict(schema=SCHEMA, status='complete' if closed else 'in_progress', identity=identity,
                summary=summary, delegated_replay=body, rows_sha256=config.digest(retained),
                game_namespace='ordinary-value-followup-v1',
                training_performed=False, default_promotion=False, future_training_authorized=False), retained


def run(path=DEFAULT, fixture=False, on_event=None, cancel=None, decision_limit=None):
    path = Path(path).resolve(); identity = identity_for(fixture)
    require(path.is_relative_to(ROOT / 'build') if fixture else path == DEFAULT.resolve(), 'Followup path differs')
    require('torch' not in sys.modules, 'Followup must be CPU-only')
    event = on_event or (lambda value: None)
    if identity['branch'] == 'hold':
        body = dict(schema=SCHEMA, status='held', identity=identity, summary=dict(games=0, committed_decisions=0),
                    training_performed=False, default_promotion=False, future_training_authorized=False)
        publish(path / 'result.json', body)
        if not fixture: publish(OUTPUT, body)
        return body
    work = phase6_store.Work(path, identity); native = Native(); roll = identity['settings']['rollouts']
    try:
        work.db.executescript('CREATE TABLE IF NOT EXISTS families(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS sp_games(id TEXT PRIMARY KEY,body TEXT,hash TEXT);CREATE TABLE IF NOT EXISTS sp_decisions(gameid TEXT,choice INTEGER,body TEXT,hash TEXT,PRIMARY KEY(gameid,choice));CREATE TABLE IF NOT EXISTS sp_rows(id TEXT PRIMARY KEY,body TEXT,hash TEXT);')
        collected = collection(work.db, identity, event, cancel)
        if collected is None: return dict(status='interrupted')
        roots, body = collected
        rollout = dict(schema=SCHEMA, collection_sha256=config.digest(body), explicit_roots=True)
        if work.get('rollout_identity') is None:
            with work.db: work.put('rollout_identity', rollout)
        require(work.get('rollout_identity') == rollout, 'Followup rollout identity differs')
        starts = old.starts_for(roots, {**identity, 'fixture': False})
        if not old.table(work.db, 'sp_games'):
            require(work.db.execute('SELECT COUNT(*) FROM sp_decisions').fetchone()[0] == 0, 'Orphan followup decisions')
            with work.db:
                for start in starts:
                    for index in range(2):
                        game = old.previous.initial_game(start, index, roll, native)
                        work.db.execute('INSERT INTO sp_games VALUES(?,?,?)', (game['id'], config.canonical(game), config.digest(game)))
        replay(work.db, identity, roots, body)
        completed = work.get('replay_complete'); count = work.db.execute('SELECT COUNT(*) FROM sp_decisions').fetchone()[0]; new = 0
        require(completed is not None or not old.table(work.db, 'sp_rows'), 'Orphan followup replay')
        if completed is None:
            with CpuBatchEvaluator(old.MODEL, cache_capacity=0, max_batch=1) as evaluator:
                while True:
                    active = [g for g in old.table(work.db, 'sp_games').values() if g['status'] == 'active']
                    if not active: break
                    if (cancel and cancel()) or (decision_limit is not None and new >= decision_limit):
                        return dict(status='interrupted', committed_decisions=count)
                    game = active[0]; rng = old.previous.rng_restore(game['rng']); before = rng.getstate(); p = game['position']
                    report = old.previous.quality_search.search(p, evaluator, simulations=roll['simulations'], time_ms=0,
                            max_nodes=roll['max_nodes'], c_puct=roll['c_puct'], root_noise=roll['root_noise'], rng=rng,
                            cancel=cancel, native=native, memory_mb=roll['memory_mb'])
                    if report['stop_reason'] != 'simulations': return dict(status='interrupted', committed_decisions=count)
                    legal = native.normalize(flatten(p), actor=p['controller'], total=98)['legal_actions']
                    action = old.previous.behavior_action(report, legal, rng, game['choices'], roll)
                    transition = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                    require(transition['ok'], 'Illegal followup action'); after = old.previous.normalize(transition['position'], native)
                    row = dict(gameid=game['id'], choice=game['choices'], state=p, action=action, after=after,
                               rng_before=before, rng_after=rng.getstate(), policy_target=report['policy'], search=report)
                    terminal = not sum(map(sum, after['pits']))
                    updated = dict(game, position=after, rng=rng.getstate(), choices=game['choices'] + 1,
                                   status='terminal' if terminal else 'capped' if game['choices'] + 1 >= roll['max_choices'] else 'active',
                                   terminal_margin=after['stores'][0] - after['stores'][1] if terminal else None)
                    event(dict(event='ordinary_value_followup_decision_precommit', committed_decisions=count, gameid=game['id']))
                    work.admit((len(config.canonical(row)) + len(config.canonical(updated))) * 3)
                    with work.db:
                        work.db.execute('INSERT INTO sp_decisions VALUES(?,?,?,?)', (game['id'], game['choices'], config.canonical(row), config.digest(row)))
                        work.db.execute('UPDATE sp_games SET body=?,hash=? WHERE id=?', (config.canonical(updated), config.digest(updated), game['id']))
                    count += 1; new += 1
                    event(dict(event='ordinary_value_followup_decision_committed', committed_decisions=count, gameid=game['id']))
            result, rows = replay(work.db, identity, roots, body, True)
            with work.db:
                for row in rows: work.db.execute('INSERT INTO sp_rows VALUES(?,?,?)', (row['id'], config.canonical(row), config.digest(row)))
                work.put('replay_complete', result)
            completed = result
        result, rows = replay(work.db, identity, roots, body, True)
        require(old.equal(completed, result) and old.equal(list(old.table(work.db, 'sp_rows').values()), rows), 'Closed followup marker/replay differs')
        publish(path / 'result.json', result)
        if not fixture: publish(OUTPUT, result)
        event(dict(event='ordinary_value_followup_complete', branch=identity['branch'], summary=result['summary']))
        return result
    finally: work.close()


def load_run(path=DEFAULT, fixture=False):
    path = Path(path).resolve(); identity = identity_for(fixture)
    saved = read(path / 'result.json')
    if identity['branch'] == 'hold':
        require(saved['status'] == 'held' and saved['identity'] == identity, 'Held allocation differs')
        return saved, []
    with closing(sqlite3.connect((path / 'work.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        require(old.metadata(db, 'identity') == identity, 'Saved followup identity differs')
        roots = old.table(db, 'roots', 'state'); collected = old.metadata(db, 'collection_complete')
        pool = old.table(db, 'families', 'idx')
        target = identity['settings']['families'] if identity['branch'] == 'new_starts' else 0
        require(list(pool) == list(range(target)), 'Closed followup pool incomplete')
        if target:
            with CpuBatchEvaluator(old.MODEL, cache_capacity=0, max_batch=1) as evaluator:
                for index, family in pool.items():
                    require(old.equal(family, old.generate_family(index, identity['settings'], Native(), evaluator)), 'Closed followup history differs')
        expected, selection, protection = explicit_starts(identity, pool)
        require(old.equal(roots, expected) and old.equal(collected,
                dict(schema=SCHEMA, roots_sha256=config.digest(expected), families_sha256=config.digest(pool),
                     branch=identity['branch'], selection=selection, rollout_protection=protection)), 'Followup selection/protection differs')
        require(old.equal(old.metadata(db, 'rollout_identity'),
                dict(schema=SCHEMA, collection_sha256=config.digest(collected), explicit_roots=True)), 'Closed followup rollout identity differs')
        value, rows = replay(db, identity, roots, collected, True)
        require(old.equal(old.metadata(db, 'replay_complete'), value) and old.equal(saved, value) and
                old.equal(list(old.table(db, 'sp_rows').values()), rows), 'Followup completed publication differs')
        return value, rows
