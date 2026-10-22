"""Exact quotient coverage, independent minimax and durable builder checks."""
from __future__ import annotations

from functools import lru_cache
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from congklak.native import Native, ROOT
from congklak.oracle import Oracle, OracleError, OracleInterrupted, build, verify, _reference


class OracleAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="congklak-oracle-")
        cls.base = Path(cls.temp.name).resolve()
        cls.path = cls.base / "complete.sqlite3"
        cls.native = Native()
        cls.manifest = build(cls.path, native=cls.native, chunk_size=512)

    @classmethod
    def tearDownClass(cls):
        # tempfile's target is a concrete, newly-created task-owned directory.
        cls.temp.cleanup()

    def test_full_coverage_independent_values_masks_and_forced_pass_inputs(self):
        ref = _reference()
        visiting = set()
        @lru_cache(None)
        def solve(counts, actor):
            state = ref.Position(7, counts[:7]+(0,)+counts[7:]+(0,), actor)
            normal = ref.normalize(state)
            if normal.terminal:
                return 0, 0
            if normal.controller != actor:
                return solve(counts, normal.controller)
            key = counts, actor
            self.assertNotIn(key, visiting, "Cycle must never become a draw")
            visiting.add(key)
            scores = {}
            for action in ref.legal_actions(state):
                transition = ref.apply(state, action).position
                child = transition.counts[:7]+transition.counts[8:15]
                scores[action] = ref.evaluate(transition)+solve(child, transition.controller)[0]
            best = (max if actor == 0 else min)(scores.values())
            mask = sum(1 << action for action, score in scores.items() if score == best)
            visiting.remove(key)
            return best, mask
        count = passes = 0
        with Oracle(self.path, self.native) as oracle:
            for active in range(5):
                for counts in ref.compositions(active, 14):
                    for actor in (0, 1):
                        stores = [(98-active)//2, (98-active)-(98-active)//2]
                        position = {"pits": [list(counts[:7]), list(counts[7:])],
                                    "stores": stores, "controller": actor}
                        result = oracle.lookup(position)
                        value, mask = solve(counts, actor)
                        self.assertEqual(result["value_player0"], stores[0]-stores[1]+value)
                        self.assertEqual(result["optimal_mask"], mask)
                        self.assertEqual(result["best_moves"], [i for i in range(7) if mask & (1 << i)])
                        normal = ref.normalize(ref.Position(7, counts[:7]+(stores[0],)+counts[7:]+(stores[1],), actor))
                        self.assertEqual(result["normalized_controller"], normal.controller)
                        self.assertEqual(result["passes"], [{"from": a, "to": b} for a,b in normal.passes])
                        self.assertEqual(result["input_controller"], actor)
                        count += 1; passes += len(normal.passes)
        self.assertEqual(count, 6120)
        self.assertEqual(passes, 658)
        self.assertEqual(self.manifest["covered_normalized_states"], 5462)
        self.assertEqual(self.manifest["verification"]["edges_checked"], 9520)

    def test_uncovered_and_terminal_distinction(self):
        with Oracle(self.path, self.native) as oracle:
            result = oracle.lookup({"pits": [[7]*7, [7]*7], "stores": [0,0], "controller": 0})
            self.assertFalse(result["covered"])
            self.assertNotIn("value_player0", result)
            result = oracle.lookup({"pits": [[0]*7, [0]*7], "stores": [48,50], "controller": 1})
            self.assertTrue(result["covered"])
            self.assertTrue(result["terminal"])
            self.assertEqual(result["optimal_mask"], 0)
            self.assertEqual(result["value_player0"], -2)

    def test_cooperative_resume_at_each_stage_and_incomplete_reader_refusal(self):
        for target in ("enumerating", "generating", "solving", "verifying"):
            with self.subTest(stage=target):
                path = self.base/(target+"-cooperative.sqlite3")
                def stop(event):
                    if event["stage"] == target:
                        return False
                with self.assertRaises(OracleInterrupted):
                    build(path, self.native, chunk_size=512, on_progress=stop)
                with self.assertRaises(OracleError) as caught:
                    Oracle(path, self.native)
                self.assertEqual(caught.exception.code, "incomplete_oracle")
                resumed = build(path, self.native, chunk_size=512)
                self.assertEqual(resumed, self.manifest)

    def test_real_process_kill_at_committed_stage_boundaries_then_resume(self):
        # The helper blocks after a flushed post-commit marker. Parent kills
        # that exact child process, then reopens/resumes the SQLite artifact.
        script = """
import sys, threading
from congklak.oracle import build
stage=sys.argv[2]
def committed(event):
    eligible = (event['stage']==stage and
                (stage!='generating' or event['generated']>=512) and
                (stage!='solving' or event['solved']>=512))
    if eligible:
        print('committed', flush=True)
        threading.Event().wait()
build(sys.argv[1], chunk_size=512, on_progress=committed)
"""
        for target in ("enumerating", "generating", "solving", "verifying", "complete"):
            with self.subTest(stage=target):
                path = self.base/(target+"-killed.sqlite3")
                process = subprocess.Popen([sys.executable, "-c", script, str(path), target],
                                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(process.stdout.readline().strip(), "committed")
                    process.kill()
                    process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=10)
                    process.stdout.close(); process.stderr.close()
                if target != "complete":
                    with self.assertRaises(OracleError) as caught:
                        Oracle(path, self.native)
                    self.assertEqual(caught.exception.code, "incomplete_oracle")
                resumed = build(path, self.native, chunk_size=512)
                self.assertEqual(resumed, self.manifest)
                self.assertEqual(verify(path, self.native), self.manifest)

    def test_real_process_kill_before_chunk_commit_and_before_manifest_publication(self):
        script = """
import sys, threading
import congklak.oracle as oracle
target=sys.argv[2]
checkpoint=oracle._checkpoint
put=oracle._put
def before_chunk(db,path,quota,reserve,stage,progress,cancel):
    if target=='chunk' and stage=='generating' and db.execute('SELECT COUNT(*) FROM states WHERE generated=1').fetchone()[0]>=512:
        print('uncommitted', flush=True)
        threading.Event().wait()
    return checkpoint(db,path,quota,reserve,stage,progress,cancel)
def before_manifest(db,name,value):
    if target=='manifest' and name=='manifest':
        print('uncommitted', flush=True)
        threading.Event().wait()
    return put(db,name,value)
oracle._checkpoint=before_chunk
oracle._put=before_manifest
oracle.build(sys.argv[1], chunk_size=512)
"""
        for target in ("chunk", "manifest"):
            with self.subTest(boundary=target):
                path = self.base/(target+"-before-commit.sqlite3")
                process = subprocess.Popen([sys.executable, "-c", script, str(path), target],
                                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertEqual(process.stdout.readline().strip(), "uncommitted")
                    process.kill(); process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait(timeout=10)
                    process.stdout.close(); process.stderr.close()
                with self.assertRaises(OracleError) as caught:
                    Oracle(path, self.native)
                self.assertEqual(caught.exception.code, "incomplete_oracle")
                self.assertEqual(build(path, self.native, chunk_size=512), self.manifest)

    def test_corrupt_value_missing_entry_and_incompatible_identity_are_refused(self):
        for change in ("UPDATE states SET value=99 WHERE key=(SELECT key FROM states LIMIT 1)",
                       "DELETE FROM states WHERE key=(SELECT key FROM states LIMIT 1)",
                       "UPDATE metadata SET checksum='bad' WHERE name='identity'"):
            with self.subTest(change=change):
                path = self.base/("damaged-"+str(abs(hash(change)))+".sqlite3")
                with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(path)) as target:
                    source.backup(target)
                    with target:
                        target.execute(change)
                with self.assertRaises(OracleError):
                    Oracle(path, self.native)
        class Incompatible:
            build_id = "0"*64
        with self.assertRaises(OracleError) as caught:
            verify(self.path, Incompatible())
        self.assertEqual(caught.exception.code, "incompatible_oracle")

    def test_single_owner_and_capacity_admission(self):
        path = self.base/"owner.sqlite3"
        seen = []
        def progress(event):
            with self.assertRaises(OracleError) as caught:
                build(path, self.native, chunk_size=512)
            self.assertEqual(caught.exception.code, "oracle_busy")
            seen.append(True)
            return False
        with self.assertRaises(OracleInterrupted):
            build(path, self.native, chunk_size=512, on_progress=progress)
        self.assertTrue(seen)
        with self.assertRaises(OracleError) as caught:
            build(self.base/"quota.sqlite3", self.native, quota_mb=0)
        self.assertEqual(caught.exception.code, "storage_error")
        with self.assertRaises(OracleError) as caught:
            build(self.base/"space.sqlite3", self.native, min_free_mb=10**12)
        self.assertEqual(caught.exception.code, "storage_error")

    def test_checksum_valid_malformed_manifest_and_sqlite_full_are_structured(self):
        from congklak.config import canonical, digest
        for value in ([], None, 4, {"status": "complete"},
                      {**self.manifest, "verification": {**self.manifest["verification"], "cycles": False}}):
            with self.subTest(value=value):
                path = self.base/("manifest-"+str(abs(hash(canonical(value))))+".sqlite3")
                with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(path)) as target:
                    source.backup(target)
                    with target:
                        target.execute("UPDATE metadata SET body=?,checksum=? WHERE name='manifest'", (canonical(value), digest(value)))
                with self.assertRaises(OracleError) as caught:
                    verify(path, self.native)
                self.assertEqual(caught.exception.code, "corrupt_oracle")
                with self.assertRaises(OracleError):
                    build(path, self.native)
        path = self.base/"sqlite-full.sqlite3"
        with patch("congklak.oracle._put", side_effect=sqlite3.OperationalError("database or disk is full")):
            with self.assertRaises(OracleError) as caught:
                build(path, self.native)
            self.assertEqual(caught.exception.code, "storage_error")
            self.assertIsInstance(caught.exception.__cause__, sqlite3.OperationalError)
        self.assertEqual(build(path, self.native, chunk_size=512), self.manifest)


if __name__ == "__main__":
    unittest.main()
