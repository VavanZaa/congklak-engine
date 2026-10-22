"""Pilot provenance, weighting, immutable publication and interruption acceptance."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from congklak import config, pilot
from congklak.native import ROOT, Native
from congklak.oracle import build as build_oracle
from tests import reference as ref


class PilotAcceptance(unittest.TestCase):
    target = 128
    seed = 20261014
    families = 32

    @classmethod
    def setUpClass(cls):
        acceptance = (ROOT / "build" / "acceptance").resolve()
        acceptance.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="pilot-", dir=acceptance)
        cls.base = Path(cls.temp.name).resolve()
        if not cls.base.is_relative_to(acceptance):
            raise RuntimeError("Temporary acceptance path escaped the workspace")
        cls.native = Native()
        cls.oracle_path = cls.base / "oracle-k4.sqlite3"
        build_oracle(cls.oracle_path, native=cls.native, chunk_size=512)
        cls.path = cls.base / "fresh"
        cls.manifest = cls.build(cls.path)

    @classmethod
    def tearDownClass(cls):
        # The absolute path was checked against our task-owned acceptance root.
        cls.temp.cleanup()

    @classmethod
    def build(cls, path, **kwargs):
        return pilot.build(path, cls.oracle_path, target=cls.target, seed=cls.seed,
                           family_count=cls.families, native=cls.native, **kwargs)

    @staticmethod
    def independent_key(position):
        normal = ref.normalize(position)
        actor = 0 if normal.terminal else normal.controller
        cells = position.counts
        a = bytes(cells + (actor,))
        exchanged = cells[8:15] + (cells[15],) + cells[:7] + (cells[7],)
        b = bytes(exchanged + (0 if normal.terminal else 1 - actor,))
        return min(a, b), b < a

    @staticmethod
    def reference_position(position):
        return ref.Position(7, tuple(position["pits"][0] + [position["stores"][0]]
                                     + position["pits"][1] + [position["stores"][1]]),
                            position["controller"])

    def assert_semantic_resume(self, manifest):
        self.assertEqual(manifest["records"], self.target)
        self.assertEqual(manifest["semantic_sha256"], self.manifest["semantic_sha256"])
        self.assertEqual(manifest["splits"], self.manifest["splits"])
        self.assertEqual(manifest["label_kinds"], self.manifest["label_kinds"])

    def test_01_exact_size_independent_provenance_canonical_labels_and_splits(self):
        records = list(pilot.records(self.path))
        self.assertEqual(len(records), self.target)
        self.assertEqual(len({r["state_key"] for r in records}), self.target)
        self.assertEqual(self.manifest["record_bytes"], 52)
        self.assertEqual(set(self.manifest["sources"]),
                         {"random_history", "heuristic_history", "search_history", "constructed_exact"})
        self.assertEqual(set(self.manifest["splits"]), set(pilot.SPLITS))
        database = self.path / self.manifest["directory"] / "dataset.sqlite3"
        with closing(sqlite3.connect(database)) as db:
            archives = {}
            for family, split, source, seed, body, checksum in db.execute("SELECT * FROM families"):
                archive = json.loads(body)
                self.assertEqual(config.digest(archive), checksum)
                if source != "constructed_exact":
                    rank = int.from_bytes(hashlib.sha256(("split-v1/" + family).encode())
                                          .digest()[:8], "big") % 100
                    self.assertEqual(split, pilot.SPLITS[0 if rank < 80 else 1 if rank < 90 else 2])
                else:
                    self.assertNotEqual(split, "train")
                states = [self.reference_position(archive["start"])]
                for action in archive["moves"]:
                    states.append(ref.apply(states[-1], action).position)
                archives[family] = (split, source, states)
            exclusions = {bytes(row[0]) for row in db.execute(
                "SELECT state FROM exclusions WHERE kind='state'")}
        visiting = set()
        @lru_cache(None)
        def exact(position):
            if ref.normalize(position).terminal:
                return ref.evaluate(position)
            self.assertNotIn(position, visiting, "An endgame cycle cannot become a draw")
            visiting.add(position)
            values = [exact(ref.apply(position, move).position)
                      for move in ref.legal_actions(position)]
            visiting.remove(position)
            return (max if ref.normalize(position).controller == 0 else min)(values)
        for record in records:
            p = self.reference_position(record["state"])
            key, swapped = self.independent_key(p)
            self.assertEqual(key.hex(), record["state_key"])
            self.assertFalse(swapped)
            split, source, states = archives[record["family"]]
            origin_key, origin_swapped = self.independent_key(states[record["step"]])
            self.assertEqual(origin_key, key)
            self.assertEqual(origin_swapped, record["source_swapped"])
            self.assertEqual(record["split"], split)
            active = sum(p.counts[:7]) + sum(p.counts[8:15])
            legal = ref.legal_actions(p)
            self.assertEqual(record["legal_mask"], sum(1 << move for move in legal))
            label = record["label"]
            if label["kind"] == "finite_choice_horizon":
                self.assertEqual(label["choice_depth"], 4)
                scores = ref.root_scores(p, 4)
                target = ref.minimax(p, 4)
                self.assertIsNone(label["oracle_manifest_sha256"])
            else:
                self.assertLessEqual(active, 4)
                self.assertNotEqual(split, "train")
                self.assertIsNone(label["choice_depth"])
                scores = {a: exact(ref.apply(p, a).position) for a in legal}
                target = exact(p)
                self.assertEqual(target % 2, 0)
                self.assertEqual(label["oracle_manifest_sha256"],
                                 self.manifest["identity"]["oracle_manifest_sha256"])
            self.assertEqual(label["target_player0"], target)
            self.assertEqual({r["pit"]: r["value_player0"] for r in label["action_scores"]}, scores)
            best = (max if p.controller == 0 else min)(scores.values()) if scores else target
            self.assertEqual(label["best_moves"], [a for a in legal if scores[a] == best])
            if split == "train":
                self.assertGreater(active, 4)
                self.assertNotIn(key, exclusions)
        self.assertTrue(pilot.verify(self.path, self.native, relabel=True)["verified"])

    def test_02_unit_weights_probabilities_and_uniform_stratum_sampler(self):
        rows = list(pilot.records(self.path))
        counts = Counter((r["split"], r["stratum"]) for r in rows)
        support = {split: len({s for sp, s in counts if sp == split}) for split in pilot.SPLITS}
        for r in rows:
            self.assertEqual(r["label_weight"], 1.0)
            self.assertEqual(r["importance_ratio"], 1.0)
            self.assertEqual(r["sampling_priority"], 1.0)
            self.assertEqual(r["relabel_priority"], 0.0)
            self.assertEqual(r["draw_probability"],
                             1 / (support[r["split"]] * counts[(r["split"], r["stratum"])]))
            self.assertGreater(r["selection_inclusion_probability"], 0)
            self.assertLessEqual(r["selection_inclusion_probability"], 1)
        a = list(pilot.sample(self.path, 8000, seed=37, split="train"))
        b = list(pilot.sample(self.path, 8000, seed=37, split="train"))
        self.assertEqual([r["state_key"] for r in a], [r["state_key"] for r in b])
        self.assertTrue(all(r["split"] == "train" for r in a))
        strata = Counter(r["stratum"] for r in a)
        self.assertEqual(len(strata), support["train"])
        expected = len(a) / support["train"]
        self.assertTrue(all(abs(c - expected) < 200 for c in strata.values()), strata)

    def test_03_cooperative_cancel_then_resume_each_phase(self):
        for stage in ("collect", "label", "publish"):
            with self.subTest(stage=stage):
                path = self.base / ("cancel-" + stage)
                stopped = False
                def progress(event):
                    nonlocal stopped
                    if event["stage"] == stage:
                        stopped = True
                with self.assertRaises(pilot.Interrupted):
                    self.build(path, on_progress=progress, cancel=lambda: stopped)
                with self.assertRaises(pilot.PilotError) as caught:
                    pilot.load(path)
                self.assertEqual(caught.exception.code, "incomplete_dataset")
                self.assert_semantic_resume(self.build(path))
                self.assertTrue(pilot.verify(path, self.native)["verified"])

    def kill_builder(self, path, boundary):
        script = r'''
import sys, threading
from congklak import pilot
path, oracle, boundary, target, seed, families = sys.argv[1:]
original_put = pilot.Work.put
def pause():
    print('boundary-ready', flush=True)
    threading.Event().wait()
def put(self, key, value):
    hit = ((boundary == 'collect-before-commit' and key == 'progress'
            and value.get('stage') == 'collect' and value.get('cursor') == 1)
           or (boundary == 'label-before-commit' and key == 'progress'
               and value.get('stage') == 'label' and value.get('labels') == 1)
           or (boundary == 'catalog-before-commit' and key == 'current'))
    if hit: pause()
    return original_put(self, key, value)
pilot.Work.put = put
def progress(event):
    hit = ((boundary == 'collect-after-commit' and event['stage'] == 'collect')
           or (boundary == 'label-after-commit' and event['stage'] == 'label'
               and event.get('labels', 0) >= 100)
           or (boundary == 'publish-files-before-catalog' and event['stage'] == 'publish'
               and 'rows_written' in event)
           or (boundary == 'catalog-after-commit' and event['stage'] == 'complete'))
    if hit: pause()
pilot.build(path, oracle, target=int(target), seed=int(seed), family_count=int(families),
            on_progress=progress)
'''
        process = subprocess.Popen([sys.executable, "-c", script, str(path), str(self.oracle_path),
                                    boundary, str(self.target), str(self.seed), str(self.families)],
                                   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        messages = queue.Queue()
        def read():
            for line in process.stdout:
                messages.put(line.strip())
            messages.put("<process-eof>")
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            message = messages.get(timeout=45)
            if message != "boundary-ready":
                process.wait(timeout=5)
                self.fail(f"{boundary}: {message}: {process.stderr.read()}")
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
            reader.join(timeout=5)
            process.stdout.close(); process.stderr.close()

    def test_04_real_process_kills_before_after_commits_and_publication(self):
        boundaries = ("collect-before-commit", "collect-after-commit", "label-before-commit",
                      "label-after-commit", "publish-files-before-catalog", "catalog-before-commit",
                      "catalog-after-commit")
        for boundary in boundaries:
            with self.subTest(boundary=boundary):
                path = self.base / boundary
                self.kill_builder(path, boundary)
                if boundary != "catalog-after-commit":
                    with self.assertRaises(pilot.PilotError) as caught:
                        pilot.load(path)
                    self.assertEqual(caught.exception.code, "incomplete_dataset")
                else:
                    self.assert_semantic_resume(pilot.load(path))
                self.assert_semantic_resume(self.build(path))
                self.assertTrue(pilot.verify(path, self.native, relabel=True)["verified"])

    def test_05_sqlite_full_preserves_prior_commits_and_resumes(self):
        path = self.base / "sqlite-full"
        original = pilot.Work.admit
        observed = []
        def actual_full(work, estimate):
            original(work, estimate)
            if work.db and work.get("progress")["stage"] == "label":
                labels = work.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
                if labels >= 3 and not observed:
                    observed.append(labels)
                    work.db.execute("CREATE TABLE full_fault(payload BLOB)")
                    pages = work.db.execute("PRAGMA page_count").fetchone()[0]
                    work.db.execute(f"PRAGMA max_page_count={pages}")
                    work.db.execute("INSERT INTO full_fault VALUES(zeroblob(8388608))")
        with patch.object(pilot.Work, "admit", actual_full):
            with self.assertRaises(pilot.PilotError) as caught:
                self.build(path)
        self.assertEqual(caught.exception.code, "storage_error")
        self.assertIn("full", str(caught.exception).lower())
        self.assertEqual(observed, [3])
        with closing(sqlite3.connect(path / "work.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM labels").fetchone()[0], 3)
        self.assert_semantic_resume(self.build(path))

    def test_06_corruption_locking_and_storage_admission(self):
        path = self.base / "bad-pending-row"
        stopped = False
        def progress(event):
            nonlocal stopped
            if event["stage"] == "label" and event.get("labels", 0) >= 100:
                stopped = True
        with self.assertRaises(pilot.Interrupted):
            self.build(path, on_progress=progress, cancel=lambda: stopped)
        with closing(sqlite3.connect(path / "work.sqlite3")) as db:
            db.execute("UPDATE labels SET body='{}' WHERE state=(SELECT state FROM labels LIMIT 1)")
            db.commit()
        with self.assertRaises(pilot.PilotError):
            self.build(path)
        with self.assertRaises(pilot.PilotError):
            pilot.load(path)
        work = pilot.Work(self.path, self.manifest["identity"])
        try:
            with self.assertRaises(pilot.PilotError):
                self.build(self.path)
        finally:
            work.close()
        self.assertEqual(pilot.load(self.path)["generation"], self.manifest["generation"])
        with patch.object(pilot.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(1, 1, 0)):
            with self.assertRaises(pilot.PilotError) as caught:
                self.build(self.base / "no-free-space")
        self.assertEqual(caught.exception.code, "storage_error")
        with self.assertRaises(pilot.PilotError) as caught:
            self.build(self.base / "quota", quota_mb=1, min_free_mb=0)
        self.assertEqual(caught.exception.code, "storage_error")
        damaged = self.base / "damaged-published"
        manifest = self.build(damaged)
        shard = damaged / manifest["directory"] / next(f["name"] for f in manifest["files"]
                                                       if f["name"].endswith(".cgp"))
        with shard.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0); stream.write(bytes([first[0] ^ 1]))
        with self.assertRaises(pilot.PilotError) as caught:
            pilot.verify(damaged, self.native)
        self.assertEqual(caught.exception.code, "corrupt_saved_data")


if __name__ == "__main__":
    unittest.main()
