"""Recoverable first-witness opening search for Dakon-n, without neural training."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from .dakon_opening_verify import (RULES, MAX_N, MAX_BYTES, canonical, digest,
                                   integer, target, cells_valid, verify, read_certificate)

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOTS = (ROOT/'data'/'dakon-opening-v1', ROOT/'build'/'dakon-opening-v1')


def transition(cells, n, actor, action, poll=lambda: None):
    """Exact arithmetic sowing: distribute full circuits in bulk, then the tail."""
    integer(n, 1, MAX_N); integer(actor, 0, 1); integer(action, 0, n-1)
    cells_valid(cells, n)
    out = list(cells)
    source, own = actor*(n+1)+action, actor*(n+1)+n
    forbidden = (1-actor)*(n+1)+n
    hand, cursor = out[source], source
    if not hand:
        raise ValueError('Selected an empty pit')
    out[source] = 0
    deposits = relays = captured = 0
    while True:
        poll()
        route = [(cursor+i) % len(out) for i in range(1, len(out)+1)
                 if (cursor+i) % len(out) != forbidden]
        turns, tail = divmod(hand, len(route))
        if turns:
            for slot in route:
                out[slot] += turns
        for slot in route[:tail]:
            out[slot] += 1
        deposits += hand
        cursor = route[tail-1] if tail else route[-1]
        before_last = out[cursor]-1
        if cursor == own:
            reason = 'own_store'; break
        if before_last:
            hand, out[cursor] = out[cursor], 0
            relays += 1
            continue
        reason = 'opponent_empty'
        if actor*(n+1) <= cursor < actor*(n+1)+n:
            opposite = 2*n-cursor
            reason = 'own_empty'
            if out[opposite]:
                captured = out[cursor]+out[opposite]
                out[own] += captured
                out[cursor] = out[opposite] = 0
                reason = 'own_capture'
        break
    rows = [any(out[p*(n+1):p*(n+1)+n]) for p in (0, 1)]
    terminal = not any(rows)
    controller = actor if terminal or reason == 'own_store' else 1-actor
    passes = []
    if not terminal and not rows[controller]:
        passes = [[controller, 1-controller]]; controller = 1-controller
    cells_valid(out, n)
    poll()
    return dict(cells=out, controller=controller, reason=reason, terminal=terminal,
                passes=passes, deposits=deposits, relays=relays, captured=captured)


def recipe(n, goal, starter, cache_entries):
    integer(n, 1, MAX_N); integer(starter, 0, 1); integer(cache_entries, 0, 10000)
    target(n, goal)
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
               [('searcher', Path(__file__)), ('verifier', Path(__file__).with_name('dakon_opening_verify.py'))]}
    return dict(schema='dakon-opening-run-v1', n=n, goal=goal, starter=starter,
                cache_entries=cache_entries, rules=RULES, sources=sources,
                ordering='canonical_pit_ascending')


def run_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(base.resolve()) and path != base.resolve() for base in RUN_ROOTS):
        raise ValueError('Use a dedicated run directory below data/dakon-opening-v1 or build/dakon-opening-v1')
    return path


def envelope(body):
    return dict(body=body, sha256=digest(body))


def read_envelope(path):
    value = read_certificate(path)
    if not isinstance(value, dict) or set(value) != {'body', 'sha256'} or digest(value['body']) != value['sha256']:
        raise ValueError(f'Checksum mismatch: {path.name}')
    return value['body']


def read_recipe(path):
    value = read_envelope(path/'identity.json')
    if not isinstance(value, dict):
        raise ValueError('Invalid saved recipe')
    expected = recipe(value.get('n'), value.get('goal'), value.get('starter'), value.get('cache_entries'))
    if canonical(value) != canonical(expected):
        raise ValueError('Saved rules, source version, or recipe is incompatible; use a new run directory')
    return value


class Stopped(Exception):
    pass


class Run:
    def __init__(self, path, *, create=None, hook=None):
        self.path = run_path(path)
        prepared = recipe(**create) if create is not None else None
        self.lock = None
        self.hook = hook or (lambda stage: None)
        self.generation = -1
        self.recovered_from_older_checkpoint = False
        if create is not None and self.path.exists() and any(self.path.iterdir()):
            raise ValueError('Run directory already exists; use resume or a new directory')
        if create is None and not self.path.is_dir():
            raise ValueError('Saved run directory does not exist')
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.lock = (self.path/'writer.lock').open('a+b')
            if self.lock.seek(0, os.SEEK_END) == 0:
                self.lock.write(b'0'); self.lock.flush()
            self.lock.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError('Another finder owns this run directory') from error
            if create is not None:
                if any(p.name != 'writer.lock' for p in self.path.iterdir()):
                    raise ValueError('Run directory was initialized by another writer; use resume')
                self.recipe = prepared
                self.write('identity.json', envelope(self.recipe))
            else:
                self.recipe = read_recipe(self.path)
            self.n, self.first = self.recipe['n'], self.recipe['starter']
            self.required = target(self.n, self.recipe['goal'])
            self.own = self.first*(self.n+1)+self.n
            self.other = (1-self.first)*(self.n+1)+self.n
            slots = list(self.path.glob('checkpoint-[01].json'))
            if slots:
                candidates = []
                for path in slots:
                    try:
                        body = read_envelope(path)
                        self.validate_frontier(body)
                        candidates.append(body)
                    except (ValueError, OSError, KeyError, TypeError):
                        pass
                if not candidates:
                    raise ValueError('No valid checkpoint remains; refusing to fabricate progress')
                self.state = max(candidates, key=lambda x: x['generation'])
                self.generation = self.state['generation']
                self.recovered_from_older_checkpoint = len(candidates) != len(slots)
            else:
                if any(self.path.glob('certificate.json')):
                    raise ValueError('Certificate exists without a valid frontier')
                self.state = dict(schema='dakon-opening-frontier-v1', recipe_hash=digest(self.recipe),
                                  generation=-1, state='searching', stop_reason=None,
                                  stack=[dict(cells=[self.n]*self.n+[0]+[self.n]*self.n+[0], next=0, via=None)],
                                  failed=[], witness=None,
                                  stats=dict(expanded_positions=1, simulated_actions=0, cache_hits=0,
                                             dead_ends=0, best_store=0))
                self.checkpoint()
            self.failed = deque(tuple(c) for c in self.state['failed'])
            self.failed_set = set(self.failed)
        except BaseException:
            self.close(); raise

    def close(self):
        if self.lock is not None:
            self.lock.close(); self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def write(self, name, value, *, checkpoint=False):
        raw = (canonical(value)+'\n').encode()
        if len(raw) > MAX_BYTES:
            raise ValueError('Record exceeds the 16 MiB checkpoint/certificate limit')
        used = sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
        if used+len(raw)+1024*1024 > 128*1024*1024:
            raise ValueError('Run exceeds its 128 MiB storage allowance')
        temporary = self.path/('.'+name+'.tmp')
        with temporary.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if checkpoint:
            self.hook('before_checkpoint_replace')
        os.replace(temporary, self.path/name)
        if checkpoint:
            self.hook('after_checkpoint_replace')

    def validate_frontier(self, body):
        if not isinstance(body, dict) or set(body) != {'schema', 'recipe_hash', 'generation', 'state', 'stop_reason', 'stack', 'failed', 'witness', 'stats'}:
            raise ValueError('Invalid frontier fields')
        if body['schema'] != 'dakon-opening-frontier-v1' or body['recipe_hash'] != digest(self.recipe):
            raise ValueError('Frontier identity mismatch')
        integer(body['generation'], 0, 2**63-1)
        if body['state'] not in ('searching', 'pending_verification', 'found', 'exhausted'):
            raise ValueError('Invalid frontier state')
        if body['stop_reason'] not in (None, 'time_limit', 'action_limit', 'cancelled'):
            raise ValueError('Invalid stop reason')
        stats = body['stats']
        if not isinstance(stats, dict) or set(stats) != {'expanded_positions', 'simulated_actions', 'cache_hits', 'dead_ends', 'best_store'}:
            raise ValueError('Invalid work counters')
        for value in stats.values():
            integer(value, 0, 2**63-1)
        integer(stats['best_store'], 0, 2*self.n*self.n)
        stack, failed = body['stack'], body['failed']
        if not isinstance(stack, list) or len(stack) > self.required+1 or not isinstance(failed, list) or len(failed) > self.recipe['cache_entries']:
            raise ValueError('Invalid bounded frontier/cache')
        for counts in failed:
            cells_valid(counts, self.n)
            if counts[self.other] or counts[self.own] >= self.required:
                raise ValueError('Invalid failed-state cache')
        if len({tuple(c) for c in failed}) != len(failed):
            raise ValueError('Duplicate cache state')
        previous = None
        for frame in stack:
            if not isinstance(frame, dict) or set(frame) != {'cells', 'next', 'via'}:
                raise ValueError('Invalid stack frame')
            cells_valid(frame['cells'], self.n); integer(frame['next'], 0, self.n)
            if frame['cells'][self.own] >= self.required or frame['cells'][self.other]:
                raise ValueError('Invalid opening frame')
            if previous is None:
                if frame['via'] is not None or frame['cells'] != [self.n]*self.n+[0]+[self.n]*self.n+[0]:
                    raise ValueError('Invalid search root')
            else:
                integer(frame['via'], 0, self.n-1)
                step = transition(previous['cells'], self.n, self.first, frame['via'])
                if previous['next'] != frame['via']+1 or step['cells'] != frame['cells'] or not self.continues(step):
                    raise ValueError('Stack is not a replayable bonus chain')
            previous = frame
        witness = body['witness']
        if body['state'] in ('found', 'pending_verification'):
            if not isinstance(witness, list) or not 1 <= len(witness) <= 2*self.n*self.n+1:
                raise ValueError('Missing bounded witness')
            for action in witness:
                integer(action, 0, self.n-1)
        elif witness is not None:
            raise ValueError('Unexpected witness')
        if (body['state'] == 'exhausted') != (not stack):
            raise ValueError('Inconsistent exhausted frontier')
        if stats['expanded_positions'] != stats['dead_ends']+len(stack) or stats['simulated_actions'] < stats['expanded_positions']-1:
            raise ValueError('Inconsistent frontier counters')

    def checkpoint(self):
        if hasattr(self, 'failed'):
            self.state['failed'] = [list(c) for c in self.failed]
        self.generation += 1
        self.state['generation'] = self.generation
        self.write(f'checkpoint-{self.generation % 2}.json', envelope(self.state), checkpoint=True)
        self.write('status.json', self.summary())

    def continues(self, step):
        return step['reason'] == 'own_store' and not step['passes'] and not step['terminal'] and step['controller'] == self.first

    def summary(self):
        state = self.state['state']
        result = dict(schema='dakon-opening-progress-v1', state=state,
                      outcome='guaranteed_win' if state == 'found' else 'no_first_turn_witness' if state == 'exhausted' else 'unknown',
                      n=self.n, goal=self.recipe['goal'], required=self.required, starter=self.first,
                      generation=self.generation, stop_reason=self.state['stop_reason'],
                      depth=max(0, len(self.state['stack'])-1), **self.state['stats'],
                      recovered_from_older_checkpoint=self.recovered_from_older_checkpoint,
                      counters_scope='committed logical work; unfinished work may be recomputed',
                      shortest_proved=False)
        if state == 'found':
            result.update(certificate=str(self.path/'certificate.json'),
                          line_right_store=[a+1 for a in self.state['witness']],
                          line_left_store=[self.n-a for a in self.state['witness']])
        return result

    def finish(self, poll):
        path = self.path/'certificate.json'
        if self.state['state'] == 'found' and path.exists():
            cert = read_certificate(path)
            if not isinstance(cert, dict) or not isinstance(cert.get('body'), dict) or canonical(cert['body'].get('recipe')) != canonical(self.recipe) or cert['body'].get('actions') != self.state['witness']:
                raise ValueError('Published certificate does not match this saved run')
            return dict(**self.summary(), verification=verify(cert, poll))
        cells = [self.n]*self.n+[0]+[self.n]*self.n+[0]
        steps = []
        for action in self.state['witness']:
            step = transition(cells, self.n, self.first, action, poll)
            steps.append(step); cells = step['cells']
        proof = dict(total=2*self.n*self.n, required=self.required, starter_store=cells[self.own],
                     opponent_max_total=2*self.n*self.n-cells[self.own], opponent_choices_before_goal=0,
                     shortest_proved=False)
        cert = envelope(dict(schema='dakon-opening-certificate-v1', recipe=self.recipe,
                             actions=self.state['witness'], steps=steps, proof=proof))
        checked = verify(cert, poll)
        if path.exists():
            if canonical(read_certificate(path)) != canonical(cert):
                raise ValueError('Existing certificate differs; refusing to overwrite it')
        else:
            self.write('certificate.json', cert)
            self.hook('after_certificate_publish')
        if self.state['state'] != 'found':
            self.state.update(state='found', stop_reason=None)
            self.checkpoint()
        return dict(**self.summary(), verification=checked)

    def search(self, *, max_actions=1000000, max_seconds=10, checkpoint_actions=256,
               progress_seconds=1, cancel=lambda: False, emit=lambda value: None):
        integer(max_actions, 0, 2**63-1); integer(checkpoint_actions, 1, 1000000)
        if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or not 0 <= max_seconds <= 86400:
            raise ValueError('max_seconds must be finite, between zero and 86400')
        if not isinstance(progress_seconds, (int, float)) or isinstance(progress_seconds, bool) or not math.isfinite(progress_seconds) or not .1 <= progress_seconds <= 60:
            raise ValueError('progress_seconds must be finite, between 0.1 and 60')
        start = last_progress = time.monotonic()
        start_actions = last_saved = self.state['stats']['simulated_actions']
        def poll():
            if cancel():
                raise Stopped('cancelled')
            if max_seconds and time.monotonic()-start >= max_seconds:
                raise Stopped('time_limit')
        if self.state['state'] == 'exhausted':
            return self.summary()
        try:
            if self.state['state'] in ('found', 'pending_verification'):
                return self.finish(poll)
            self.state['stop_reason'] = None
            while self.state['stack']:
                poll()
                stats = self.state['stats']
                if max_actions and stats['simulated_actions']-start_actions >= max_actions:
                    raise Stopped('action_limit')
                frame = self.state['stack'][-1]
                action = frame['next']
                if action == self.n:
                    key = tuple(frame['cells'])
                    if self.recipe['cache_entries'] and key not in self.failed_set:
                        if len(self.failed) == self.recipe['cache_entries']:
                            self.failed_set.remove(self.failed.popleft())
                        self.failed.append(key); self.failed_set.add(key)
                    self.state['stack'].pop(); stats['dead_ends'] += 1
                    continue
                if not frame['cells'][self.first*(self.n+1)+action]:
                    frame['next'] += 1; continue
                step = transition(frame['cells'], self.n, self.first, action, poll)
                # Mutate the frontier only after the complete private transition succeeds.
                frame['next'] += 1; stats['simulated_actions'] += 1
                stats['best_store'] = max(stats['best_store'], step['cells'][self.own])
                if step['cells'][self.own] >= self.required:
                    self.state.update(state='pending_verification', witness=[f['via'] for f in self.state['stack'][1:]]+[action])
                    self.checkpoint(); emit(self.summary())
                    return self.finish(poll)
                if self.continues(step):
                    if tuple(step['cells']) in self.failed_set:
                        stats['cache_hits'] += 1
                    else:
                        self.state['stack'].append(dict(cells=step['cells'], next=0, via=action))
                        stats['expanded_positions'] += 1
                now = time.monotonic()
                if stats['simulated_actions']-last_saved >= checkpoint_actions or now-last_progress >= progress_seconds:
                    self.checkpoint(); emit(self.summary())
                    last_saved, last_progress = stats['simulated_actions'], now
            self.state.update(state='exhausted', stop_reason=None)
            self.checkpoint()
            return self.summary()
        except Stopped as stop:
            if self.state['state'] == 'found':
                # Preserve the old completed run, but never claim this interrupted read verified it.
                return dict(self.summary(), state='verification_interrupted', outcome='unknown',
                            stop_reason=str(stop))
            self.state['stop_reason'] = str(stop); self.checkpoint()
            return self.summary()


def status(path):
    """Read the two durable slots, without locks, writes, native code, or search."""
    path = run_path(path)
    saved = read_recipe(path)
    reader = Run.__new__(Run)
    reader.path = path
    reader.recipe, reader.n, reader.first = saved, saved['n'], saved['starter']
    reader.required = target(reader.n, saved['goal'])
    reader.own, reader.other = reader.first*(reader.n+1)+reader.n, (1-reader.first)*(reader.n+1)+reader.n
    valid = []
    for slot in path.glob('checkpoint-[01].json'):
        try:
            body = read_envelope(slot)
            reader.validate_frontier(body)
            valid.append(body)
        except (ValueError, OSError, KeyError, TypeError):
            pass
    if not valid:
        raise ValueError('No checksummed checkpoint is available')
    body = max(valid, key=lambda b: b['generation'])
    if body['state'] == 'found':
        certificate = read_certificate(path/'certificate.json')
        verify(certificate)
        if canonical(certificate['body']['recipe']) != canonical(saved) or certificate['body']['actions'] != body['witness']:
            raise ValueError('Published certificate does not match the saved run')
    reader.state, reader.generation = body, body['generation']
    reader.recovered_from_older_checkpoint = len(valid) != len(list(path.glob('checkpoint-[01].json')))
    return reader.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('find', 'resume'):
        command = commands.add_parser(name)
        command.add_argument('--run-dir', type=Path, required=True)
        command.add_argument('--max-actions', type=int, default=1000000)
        command.add_argument('--max-seconds', type=float, default=10)
        command.add_argument('--checkpoint-actions', type=int, default=256)
        if name == 'find':
            command.add_argument('--n', type=int, required=True)
            command.add_argument('--goal', choices=['majority', 'refill_threshold'], default='majority')
            command.add_argument('--starter', choices=['A', 'B'], default='A')
            command.add_argument('--cache-entries', type=int, default=10000)
    command = commands.add_parser('status'); command.add_argument('--run-dir', type=Path, required=True)
    command = commands.add_parser('verify'); command.add_argument('certificate', type=Path)
    args = parser.parse_args(argv)
    cancelled = False
    def interrupt(*_):
        nonlocal cancelled
        cancelled = True
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        if args.command == 'verify':
            result = verify(read_certificate(args.certificate), lambda: (_ for _ in ()).throw(ValueError('Verification cancelled')) if cancelled else None)
        elif args.command == 'status':
            result = status(args.run_dir)
        else:
            create = dict(n=args.n, goal=args.goal, starter=0 if args.starter == 'A' else 1,
                          cache_entries=args.cache_entries) if args.command == 'find' else None
            with Run(args.run_dir, create=create) as run:
                print(canonical(dict(event='started', **run.summary())), flush=True)
                result = run.search(max_actions=args.max_actions, max_seconds=args.max_seconds,
                                    checkpoint_actions=args.checkpoint_actions, cancel=lambda: cancelled,
                                    emit=lambda value: print(canonical(dict(event='progress', **value)), flush=True))
        print(canonical(dict(event='result', **result)), flush=True)
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
        parser.exit(2, f'Finder error: {error}\n')
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == '__main__':
    main()
