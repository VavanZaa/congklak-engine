"""Exactness, certificate rejection, bounded work, and real process recovery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch

from congklak import dakon_opening as finder
from congklak.dakon_opening_verify import literal, verify, read_certificate

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT/'build'/'dakon-opening-v1'/'tests'


def fresh():
    return BASE/uuid.uuid4().hex


def create(n=7, goal='majority', starter=0, cache_entries=100):
    return dict(n=n, goal=goal, starter=starter, cache_entries=cache_entries)


def solve(path, **options):
    with finder.Run(path, create=create(**options)) as run:
        answer = run.search(max_seconds=10)
    return answer, read_certificate(path/'certificate.json')


def fingerprints(path):
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in path.glob('*.json')}


class OpeningFinder(unittest.TestCase):
    def test_bulk_sowing_matches_literal_small_and_wide_boards(self):
        randomizer = random.Random(20261017)
        comparisons = 0
        for n in (1, 2, 3, 4, 7, 8, 12, 16, 64):
            for first in (0, 1):
                for _ in range(12 if n <= 8 else 3):
                    cells = [0]*(2*n+2)
                    for _ in range(2*n*n):
                        cells[randomizer.randrange(len(cells))] += 1
                    legal = [a for a in range(n) if cells[first*(n+1)+a]]
                    for action in legal[:3]:
                        self.assertEqual(finder.transition(cells, n, first, action), literal(cells, n, first, action))
                        comparisons += 1
                cells = [0]*(2*n+2)
                cells[first*(n+1)] = 2*n*n
                self.assertEqual(finder.transition(cells, n, first, 0), literal(cells, n, first, 0))
        self.assertGreater(comparisons, 300)

    def test_existing_native_kernel_and_old_independent_reference_agree(self):
        from congklak.native import Native, flatten
        spec = importlib.util.spec_from_file_location('opening_old_reference', ROOT/'tests'/'reference.py')
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference; spec.loader.exec_module(reference)
        native = Native()
        for n in range(1, 8):
            for first in (0, 1):
                cells = [n]*n+[0]+[n]*n+[0]
                for action in range(n):
                    actual = finder.transition(cells, n, first, action)
                    old = reference.apply(reference.Position(n, tuple(cells), first), action)
                    compiled = native.apply(cells, pits=n, actor=first, action=action, total=2*n*n)
                    self.assertTrue(compiled['ok'])
                    self.assertEqual(actual['cells'], flatten(compiled['position']))
                    self.assertEqual(tuple(actual['cells']), old.position.counts)
                    self.assertEqual(actual['controller'], old.position.controller)
                    self.assertEqual(actual['deposits'], old.deposits)
                    self.assertEqual(actual['relays'], old.relay_pickups)

    def test_majority_witness_and_display_coordinates_for_both_starters(self):
        for first in (0, 1):
            path = fresh(); answer, certificate = solve(path, starter=first)
            checked = verify(certificate)
            self.assertEqual(answer['outcome'], 'guaranteed_win')
            self.assertEqual(checked['choices'], 13)
            self.assertEqual(checked['starter_store'], 54)
            self.assertEqual(checked['required'], 50)
            self.assertEqual(checked['line_left_store'], [7,6,3,6,7,3,6,6,5,4,5,7,7])
            self.assertFalse(checked['shortest_proved'])
            self.assertEqual(certificate['body']['steps'][-1]['reason'], 'own_capture')

    def test_stricter_threshold_is_a_separate_objective(self):
        path = fresh(); answer, certificate = solve(path, goal='refill_threshold')
        checked = verify(certificate)
        self.assertEqual(checked['required'], 92)
        self.assertEqual(checked['starter_store'], 92)
        self.assertLess(answer['simulated_actions'], 10000)

    def test_n8_and_n12_have_no_fixed_16_slot_or_byte_count_limit(self):
        for n in (8, 12):
            path = fresh(); answer, certificate = solve(path, n=n)
            checked = verify(certificate)
            self.assertEqual(answer['outcome'], 'guaranteed_win')
            self.assertEqual(checked['total'], 2*n*n)
            self.assertGreaterEqual(checked['starter_store'], n*n+1)

    def test_paused_search_resumes_identically(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            paused = run.search(max_actions=5, max_seconds=0, checkpoint_actions=2)
            self.assertEqual(paused['outcome'], 'unknown')
            self.assertEqual(paused['stop_reason'], 'action_limit')
        with finder.Run(path) as run:
            resumed = run.search()
        fresh_result, fresh_cert = solve(fresh())
        self.assertEqual(read_certificate(path/'certificate.json'), fresh_cert)
        for key in ('simulated_actions', 'expanded_positions', 'dead_ends', 'cache_hits'):
            self.assertEqual(resumed[key], fresh_result[key])

    def test_pending_witness_is_not_reported_as_a_verified_win(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            paused = run.search(cancel=lambda: run.state['state'] == 'pending_verification')
        self.assertEqual(paused['state'], 'pending_verification')
        self.assertEqual(paused['outcome'], 'unknown')
        self.assertFalse((path/'certificate.json').exists())
        with finder.Run(path) as run:
            self.assertEqual(run.search()['outcome'], 'guaranteed_win')

    def test_small_exhausted_tree_is_not_a_full_game_loss_claim(self):
        path = fresh()
        with finder.Run(path, create=create(n=3, goal='refill_threshold')) as run:
            result = run.search()
        self.assertEqual(result['state'], 'exhausted')
        self.assertEqual(result['outcome'], 'no_first_turn_witness')
        self.assertFalse((path/'certificate.json').exists())

    def test_completed_resume_and_status_do_not_rewrite_results(self):
        path = fresh(); solve(path)
        before = fingerprints(path)
        with finder.Run(path) as run:
            self.assertEqual(run.search()['outcome'], 'guaranteed_win')
        self.assertEqual(finder.status(path)['state'], 'found')
        self.assertEqual(fingerprints(path), before)

    def test_cancelled_completed_verification_does_not_claim_a_fresh_pass(self):
        path = fresh(); solve(path)
        before = fingerprints(path)
        with finder.Run(path) as run:
            result = run.search(cancel=lambda: True)
        self.assertEqual(result['outcome'], 'unknown')
        self.assertEqual(result['state'], 'verification_interrupted')
        self.assertEqual(fingerprints(path), before)
        certificate = read_certificate(path/'certificate.json')
        certificate['body']['proof']['starter_store'] = 98
        (path/'certificate.json').write_text(json.dumps(certificate))
        with finder.Run(path) as run:
            with self.assertRaises(ValueError):
                run.search()
        with self.assertRaises(ValueError):
            finder.status(path)

    def test_capture_pass_back_is_not_an_opening_bonus(self):
        # A capture empties B's row; A retains control solely because B passes.
        cells = [1,0,1,15,0,1,0,0]
        step = finder.transition(cells, 3, 0, 0)
        self.assertEqual(step['reason'], 'own_capture')
        self.assertEqual(step['controller'], 0)
        self.assertEqual(step['passes'], [[1,0]])
        with finder.Run(fresh(), create=create(n=3)) as run:
            self.assertFalse(run.continues(step))

    def test_corrupt_latest_checkpoint_falls_back_and_both_bad_refused(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            run.search(max_actions=5, checkpoint_actions=2)
        slots = sorted(path.glob('checkpoint-[01].json'), key=lambda p: finder.read_envelope(p)['generation'])
        slots[-1].write_text('truncated')
        with finder.Run(path) as run:
            self.assertTrue(run.recovered_from_older_checkpoint)
            self.assertEqual(run.search()['outcome'], 'guaranteed_win')
        for slot in path.glob('checkpoint-[01].json'):
            slot.write_text('truncated')
        with self.assertRaisesRegex(ValueError, 'No valid checkpoint'):
            finder.Run(path)

    def test_checksum_valid_invalid_frontier_is_refused(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            run.search(max_actions=5, checkpoint_actions=2)
        for slot in path.glob('checkpoint-[01].json'):
            body = finder.read_envelope(slot)
            body['stack'][0]['cells'][0] -= 1; body['stack'][0]['cells'][7] += 1
            slot.write_text(finder.canonical(finder.envelope(body)))
        with self.assertRaisesRegex(ValueError, 'No valid checkpoint'):
            finder.Run(path)

    def test_certificates_are_recomputed_not_trusted(self):
        _, certificate = solve(fresh())
        for mutate in (
            lambda b: b['actions'].__setitem__(0, 6),
            lambda b: b['proof'].__setitem__('starter_store', 98),
            lambda b: b['proof'].__setitem__('shortest_proved', True),
            lambda b: b['recipe'].__setitem__('goal', 'refill_threshold'),
            lambda b: b['recipe']['rules'].__setitem__('skip_opponent_store', 1),
            lambda b: b['steps'][0]['cells'].__setitem__(0, 98),
        ):
            body = deepcopy(certificate['body']); mutate(body)
            with self.assertRaises(ValueError):
                verify(finder.envelope(body))

    def test_source_and_recipe_mismatch_and_busy_writer_are_refused(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            with self.assertRaisesRegex(ValueError, 'Another finder'):
                finder.Run(path)
            run.search(max_actions=2)
        identity = finder.read_envelope(path/'identity.json')
        identity['sources']['searcher'] = '0'*64
        (path/'identity.json').write_text(finder.canonical(finder.envelope(identity)))
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            finder.Run(path)

    def test_product_paths_invalid_sizes_and_nonfinite_budgets_refused(self):
        with self.assertRaisesRegex(ValueError, 'dedicated run directory'):
            finder.Run(ROOT/'data'/'product-v1')
        for n in (0, 65, True):
            path = fresh()
            with self.assertRaises(ValueError):
                finder.Run(path, create=create(n=n))
            self.assertFalse(path.exists())
        with finder.Run(fresh(), create=create()) as run:
            for bad in (float('inf'), float('nan'), True, -1):
                with self.assertRaises(ValueError):
                    run.search(max_seconds=bad)

    def test_checkpoint_write_failure_preserves_previous_frontier(self):
        path = fresh()
        with finder.Run(path, create=create()) as run:
            with patch.object(finder.os, 'replace', side_effect=OSError('Synthetic disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    run.search(max_actions=5, checkpoint_actions=5)
        with finder.Run(path) as run:
            self.assertEqual(run.state['stats']['simulated_actions'], 0)
            self.assertEqual(run.search()['outcome'], 'guaranteed_win')

    def test_real_kills_before_after_checkpoint_and_certificate(self):
        worker = r'''
import sys,time
from pathlib import Path
from congklak.dakon_opening import Run
path,barrier=Path(sys.argv[1]),sys.argv[2]
run=None
def hook(stage):
    if (stage==barrier and run is not None and run.state['stats']['simulated_actions']>=10) or (barrier=='initial' and stage=='before_checkpoint_replace'):
        (path/'barrier.txt').write_text(stage)
        while True:time.sleep(.02)
with Run(path,create=dict(n=7,goal='refill_threshold',starter=0,cache_entries=100),hook=hook if barrier=='initial' else None) as run:
    run.hook=hook
    run.search(max_seconds=15,checkpoint_actions=5)
'''
        expected, certificate = solve(fresh(), goal='refill_threshold')
        receipts = []
        for barrier in ('initial', 'before_checkpoint_replace', 'after_checkpoint_replace', 'after_certificate_publish'):
            path = fresh()
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-c', worker, str(path), barrier], cwd=ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            try:
                deadline = time.monotonic()+15
                while not (path/'barrier.txt').exists() and child.poll() is None and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue((path/'barrier.txt').exists(), f'Worker did not reach {barrier}: {child.poll()}')
                published = (path/'certificate.json').read_bytes() if (path/'certificate.json').exists() else None
                child.kill(); child.wait(timeout=5)
                with finder.Run(path) as run:
                    result = run.search()
                self.assertEqual(result['outcome'], 'guaranteed_win')
                self.assertEqual(read_certificate(path/'certificate.json'), certificate)
                self.assertEqual(result['simulated_actions'], expected['simulated_actions'])
                if published is not None:
                    self.assertEqual((path/'certificate.json').read_bytes(), published)
                receipts.append(dict(barrier=barrier, actual_process_kill=True, passed=True,
                                     simulated_actions=result['simulated_actions']))
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=5)
                child.communicate(timeout=5)
        output = ROOT/'build'/'dakon-opening-v1'/'recovery.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(passed=True, cases=receipts), indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
