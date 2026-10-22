"""Independent phase-two audit: helpers use only the independent interpreter.

Production oracle builders and search transitions are never used to calculate
expected values.  Artifact and recovery checks are appended as the public APIs
are finalized; a missing native/oracle artifact is a failure, never a skip.
"""
from __future__ import annotations

import ctypes
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import queue
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

from congklak.native import Control, Native, ROOT
import reference as ref


def reference_k4():
    """Literal award recurrence, DFS cycle detection and complete normalized keys."""
    solved = {}
    pending = set()
    counts = {"edges": 0, "same_layer_edges": 0}

    def solve(pits, actor):
        position = ref.Position(7, tuple(pits[:7]) + (0,) + tuple(pits[7:]) + (0,), actor)
        normalized = ref.normalize(position)
        key = (tuple(pits), normalized.controller)
        if key in solved:
            return solved[key]
        if normalized.terminal:
            solved[key] = (0, 0)
            return solved[key]
        if key in pending:
            raise AssertionError("Independent K=4 recurrence found a dependency cycle")
        pending.add(key)
        values = []
        for action in ref.legal_actions(position):
            child = ref.apply(position, action).position
            child_pits = child.counts[:7] + child.counts[8:15]
            counts["edges"] += 1
            counts["same_layer_edges"] += sum(child_pits) == sum(pits)
            future, _ = solve(child_pits, child.controller)
            values.append((action, ref.evaluate(child) + future))
        optimum = (max if normalized.controller == 0 else min)(value for _, value in values)
        mask = sum(1 << action for action, value in values if value == optimum)
        pending.remove(key)
        solved[key] = (optimum, mask)
        return solved[key]

    for active in range(5):
        for pits in ref.compositions(active, 14):
            for actor in (0, 1):
                solve(pits, actor)
    return solved, counts


def reference_pv(position, depth, first=None):
    """Choose the first lexicographic tied optimum by full minimax at each node."""
    result = []
    while depth and not ref.normalize(position).terminal:
        scores = ref.root_scores(position, depth)
        actor = ref.normalize(position).controller
        best = (max if actor == 0 else min)(scores.values())
        move = first if first is not None else next(a for a, value in scores.items() if value == best)
        first = None
        result.append((actor, move))
        position = ref.apply(position, move).position
        depth -= 1
    return result, ref.evaluate(position), ref.normalize(position).terminal


def product_position(counts, actor):
    return {"pits": [list(counts[:7]), list(counts[8:15])],
            "stores": [counts[7], counts[15]], "controller": actor}


def independent_canonical(value):
    counts = tuple(value["pits"][0]) + (value["stores"][0],) + tuple(value["pits"][1]) + (value["stores"][1],)
    original = ref.Position(7, counts, value["controller"])
    normalized = ref.normalize(original)
    actor = 0 if normalized.terminal else normalized.controller
    a = bytes(counts + (actor,))
    b = bytes(counts[8:] + counts[:8] + (0 if normalized.terminal else 1-actor,))
    return min(a, b), b < a


class OptimizedIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = Native()
        setter = cls.native.lib.cc_set_search_options
        setter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        setter.restype = ctypes.c_char_p
        cls.comparisons = 0

    def run_search(self, position, depth, tt, pvs):
        with Control(self.native, max_depth=depth, time_ms=0, node_budget=0, memory_mb=512) as control:
            configured = json.loads(self.native.lib.cc_set_search_options(control.handle, tt, int(pvs)))
            self.assertTrue(configured["ok"], configured)
            result = self.native.analyze(position.counts, pits=position.pits,
                                         actor=position.controller, max_depth=depth,
                                         start_depth=depth, control=control)
        self.assertTrue(result["ok"], result)
        self.assertIsNotNone(result["report"], result)
        self.assertEqual(result["stop_reason"], "max_depth")
        return result["report"]

    def check_report(self, position, depth, report):
        expected_value = ref.minimax(position, depth)
        self.assertEqual(report["depth_result"]["value_player0"], expected_value, (position, depth))
        expected_actions = ref.root_scores(position, depth)
        self.assertEqual({c["pit"]: c["depth_result"]["value_player0"] for c in report["candidates"]
                          if c["depth_result"] is not None}, expected_actions)
        self.assertEqual(tuple(report["best_moves"]), ref.optimal_actions(position, depth))
        pv, leaf, terminal = reference_pv(position, depth)
        self.assertEqual([(m["actor"], m["pit"]) for m in report["pv"]], pv)
        self.assertEqual(report["pv_truncated"], not terminal)
        self.assertEqual(leaf, expected_value)
        if depth:
            for candidate in report["candidates"]:
                expected_pv, leaf, terminal = reference_pv(position, depth, candidate["pit"])
                self.assertEqual([(m["actor"], m["pit"]) for m in candidate["pv"]], expected_pv)
                self.assertEqual(candidate["depth_result"]["value_player0"], leaf)
                self.assertEqual(candidate["pv_truncated"], not terminal)
        type(self).comparisons += 1

    def test_01_tt_pvs_independent_small_max_min_bonus_pass_and_horizon(self):
        corpus = []
        for counts in ref.compositions(4, 6):
            for actor in (0, 1):
                corpus.append(ref.Position(2, counts, actor))
        # Explicit depth-zero checks prevent deeper TT values replacing H_v1.
        for depth in (0, 1, 3, 5):
            for position in corpus:
                for tt, pvs in ((0, False), (1, False), (0, True), (1, True)):
                    report = self.run_search(position, depth, tt, pvs)
                    self.check_report(position, 0 if ref.normalize(position).terminal else depth, report)

    def test_02_product_history_scores_all_ties_and_canonical_pv(self):
        for seed in range(48):
            rng = random.Random(31000 + seed)
            position = ref.Position(7, (7,) * 7 + (0,) + (7,) * 7 + (0,), seed % 2)
            for _ in range(rng.randrange(1, 60)):
                if ref.normalize(position).terminal:
                    break
                position = ref.apply(position, rng.choice(ref.legal_actions(position))).position
            depth = 0 if ref.normalize(position).terminal else 4
            for tt, pvs in ((1, False), (0, True), (1, True)):
                self.check_report(position, depth, self.run_search(position, depth, tt, pvs))


class OracleIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from congklak import oracle
        cls.module = oracle
        cls.expected, cls.expected_counts = reference_k4()
        cls.temp_root = ROOT / "build" / "acceptance"
        cls.temp_root.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="independent-oracle-", dir=cls.temp_root)
        cls.path = Path(cls.directory.name).resolve() / "oracle.sqlite3"
        if not cls.path.is_relative_to(cls.temp_root.resolve()):
            raise AssertionError("Refuse cleanup outside checked workspace audit directory")
        cls.manifest = oracle.build(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_01_every_quotient_value_mask_and_complete_domain(self):
        self.assertEqual(len(self.expected), 5462)
        self.assertEqual(self.expected_counts, {"edges": 9520, "same_layer_edges": 5968})
        with closing(sqlite3.connect(self.path)) as db:
            actual = { (tuple(key[:14]), key[14]): (value, mask)
                       for key, value, mask in db.execute("SELECT key,value,optimal_mask FROM states") }
            self.assertEqual(actual, self.expected)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM edges").fetchone()[0], 9520)
        self.assertEqual(self.manifest["covered_normalized_states"], 5462)
        self.assertEqual(self.manifest["excluded_unnormalized_slots"], 658)
        for (counts, actor), (future, mask) in actual.items():
            self.assertLessEqual(abs(future), sum(counts))
            self.assertEqual((future - sum(counts)) % 2, 0)
            self.assertTrue(sum(counts) == 0 or mask != 0)

    def test_02_every_raw_controller_alias_store_offset_and_mask_owner(self):
        checked = 0
        with self.module.Oracle(self.path) as oracle:
            for active in range(5):
                for counts in ref.compositions(active, 14):
                    for input_actor in (0, 1):
                        ring = counts[:7] + (0,) + counts[7:] + (0,)
                        original = ref.Position(7, ring, input_actor)
                        normalized = ref.normalize(original)
                        future, mask = self.expected[(counts, normalized.controller)]
                        for stores in ((98-active, 0), (0, 98-active)):
                            value = product_position(counts[:7] + (stores[0],) + counts[7:] + (stores[1],), input_actor)
                            actual = oracle.lookup(value)
                            self.assertTrue(actual["covered"])
                            self.assertEqual(actual["future_awards_player0"], future)
                            self.assertEqual(actual["value_player0"], stores[0] - stores[1] + future)
                            self.assertEqual(actual["optimal_mask"], mask)
                            self.assertEqual(actual["normalized_controller"], normalized.controller)
                            self.assertEqual([(p["from"], p["to"]) for p in actual["passes"]], list(normalized.passes))
                            self.assertEqual(actual["terminal"], normalized.terminal)
                            self.assertEqual(actual["best_moves"], [a for a in ref.legal_actions(original) if mask & (1 << a)])
                            self.assertEqual(actual["value_actor"], actual["value_player0"] * (1 if normalized.controller == 0 else -1))
                            checked += 1
            self.assertFalse(oracle.lookup(product_position((7,)*7+(0,)+(7,)*7+(0,), 0))["covered"])
        self.assertEqual(checked, 12240)

    def test_03_hard_kill_committed_generation_then_resume(self):
        for stage in ("enumerating", "generating", "solving", "verifying"):
            with tempfile.TemporaryDirectory(prefix="oracle-kill-", dir=self.temp_root) as directory:
                path = Path(directory).resolve() / "oracle.sqlite3"
                self.assertTrue(path.is_relative_to(self.temp_root.resolve()))
                child = subprocess.Popen([sys.executable, "-m", "congklak.oracle", "build", "--path", str(path),
                                          "--chunk-size", "32"], cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8")
                try:
                    while True:
                        line = child.stdout.readline()
                        if not line:
                            self.fail("Oracle child exited before target stage: " + child.stderr.read())
                        event = json.loads(line)
                        if event.get("stage") == stage:
                            break
                    child.kill()
                    child.wait(timeout=15)
                finally:
                    if child.poll() is None:
                        child.kill(); child.wait(timeout=15)
                    child.stdout.close(); child.stderr.close()
                resumed = self.module.build(path, chunk_size=32)
                self.assertEqual(resumed, self.manifest, stage)

    def test_04_checksum_valid_malformed_manifest_is_structured_failure(self):
        with tempfile.TemporaryDirectory(prefix="oracle-damaged-", dir=self.temp_root) as directory:
            path = Path(directory).resolve() / "oracle.sqlite3"
            self.assertTrue(path.is_relative_to(self.temp_root.resolve()))
            with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(path)) as target:
                source.backup(target)
            for value in ([], 1, "complete", {"status": "complete"}):
                body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                checksum = hashlib.sha256(body.encode()).hexdigest()
                with closing(sqlite3.connect(path)) as db:
                    db.execute("UPDATE metadata SET body=?,checksum=? WHERE name='manifest'", (body, checksum))
                    db.commit()
                with self.assertRaises(self.module.OracleError) as failure:
                    self.module.verify(path)
                self.assertEqual(failure.exception.code, "corrupt_oracle")


class PilotPublishedIndependentAudit(unittest.TestCase):
    def test_01_final_10000_rows_provenance_split_weights_and_independent_labels(self):
        """A release without its requested 10,000-row artifact cannot pass."""
        from congklak import pilot
        path = ROOT / "data" / "pilot-10000"
        manifest = pilot.load(path)
        self.assertEqual(manifest["records"], 10000)
        self.assertEqual(manifest["identity"]["target"], 10000)
        self.assertEqual(manifest["record_bytes"], 52)
        self.assertTrue(manifest["protected"])
        self.assertTrue(manifest["q_equals_P"])
        root = path / manifest["directory"]
        expected_oracle, _ = reference_k4()
        with closing(sqlite3.connect(root / "dataset.sqlite3")) as db:
            db.row_factory = sqlite3.Row
            records = [json.loads(row["body"]) for row in db.execute("SELECT * FROM examples ORDER BY state")]
            families = {}
            for row in db.execute("SELECT * FROM families"):
                archive = json.loads(row["body"])
                position = archive["start"]
                state = ref.Position(7, tuple(position["pits"][0]) + (position["stores"][0],) +
                                     tuple(position["pits"][1]) + (position["stores"][1],), position["controller"])
                states = [state]
                for action in archive["moves"]:
                    state = ref.apply(state, action).position
                    states.append(state)
                self.assertEqual(archive["reachability"], "counterfactual_valid_stable" if row["source"] == "constructed_exact" else "observed_legal_history")
                families[row["id"]] = (row["split"], archive, states)
            exclusions = {(row["kind"], bytes(row["state"])) for row in db.execute("SELECT * FROM exclusions")}
        self.assertEqual(len(records), 10000)
        counts = Counter((r["split"], r["stratum"]) for r in records)
        support = {split: len({s for sp, s in counts if sp == split}) for split in ("train", "development", "sealed_final_test")}
        seen = set()
        family_owners = defaultdict(set)
        # Select evenly spaced deterministic samples; their distribution is not
        # used to choose a model, teacher budget or retention policy.
        horizon_samples = set(range(0, 10000, 173))
        horizon_checked = exact_checked = 0
        for index, record in enumerate(records):
            state = record["state"]
            key, swapped = independent_canonical(state)
            self.assertEqual(key.hex(), record["state_key"])
            self.assertFalse(swapped)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(sum(key[:16]), 98)
            self.assertIn(record["split"], support)
            family_owners[record["family"]].add(record["split"])
            owner, archive, family_states = families[record["family"]]
            self.assertEqual(record["split"], owner)
            original = family_states[record["step"]]
            original_value = product_position(original.counts, original.controller)
            original_key, original_swapped = independent_canonical(original_value)
            self.assertEqual(original_key, key)
            self.assertEqual(record["source_swapped"], original_swapped)
            self.assertIs(type(record["source_swapped"]), bool)
            self.assertEqual(record["teacher_revision"], manifest["identity"]["teacher_revision"])
            self.assertEqual(record["rules_digest"], "c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e")
            self.assertEqual(record["state_encoding"], "canonical-player-swap/normalized/terminal-actor0-v1")
            self.assertEqual(record["objective"], "final-store0-minus-store1")
            self.assertEqual(record["label_weight"], 1.0)
            self.assertEqual(record["sampling_priority"], 1.0)
            self.assertEqual(record["relabel_priority"], 0.0)
            self.assertEqual(record["importance_ratio"], 1.0)
            self.assertEqual(record["retention_reason"], "validated-uniform-stratum-pilot")
            self.assertEqual(record["draw_probability"], 1 / (support[owner] * counts[(owner, record["stratum"])]))
            selection = manifest["selection"]
            self.assertEqual(record["selection_inclusion_probability"], selection["quotas"][record["stratum"]] / selection["eligible_counts"][record["stratum"]])
            position = ref.Position(7, tuple(key[:16]), key[16])
            active = sum(key[:7]) + sum(key[8:15])
            legal = ref.legal_actions(position)
            self.assertEqual(record["legal_mask"], sum(1 << a for a in legal))
            label = record["label"]
            self.assertIn(label["kind"], ("finite_choice_horizon", "exact_optimal_margin"))
            self.assertIs(type(label["target_player0"]), int)
            actual_scores = {r["pit"]: r["value_player0"] for r in label["action_scores"]}
            self.assertEqual(set(actual_scores), set(legal))
            self.assertEqual(len(actual_scores), len(label["action_scores"]))
            self.assertTrue(all(type(a["pit"]) is int and type(a["value_player0"]) is int for a in label["action_scores"]))
            if owner == "train":
                self.assertGreater(active, 4)
                self.assertEqual(label["kind"], "finite_choice_horizon")
                self.assertNotIn(("state", key), exclusions)
            if label["kind"] == "exact_optimal_margin":
                self.assertNotEqual(owner, "train")
                self.assertLessEqual(active, 4)
                self.assertIsNone(label["choice_depth"])
                quotient_counts = position.counts[:7] + position.counts[8:15]
                future, mask = expected_oracle[(quotient_counts, position.controller)]
                self.assertEqual(label["target_player0"], ref.evaluate(position) + future)
                self.assertEqual(label["best_moves"], [a for a in legal if mask & (1 << a)])
                for action in legal:
                    child = ref.apply(position, action).position
                    child_counts = child.counts[:7] + child.counts[8:15]
                    self.assertEqual(actual_scores[action], ref.evaluate(child) + expected_oracle[(child_counts, child.controller)][0])
                exact_checked += 1
            else:
                self.assertEqual(label["choice_depth"], 4)
                self.assertIsNone(label["oracle_manifest_sha256"])
                if index in horizon_samples:
                    self.assertEqual(label["target_player0"], ref.minimax(position, 4))
                    self.assertEqual(actual_scores, ref.root_scores(position, 4))
                    self.assertEqual(tuple(label["best_moves"]), ref.optimal_actions(position, 4))
                    horizon_checked += 1
        self.assertTrue(all(len(owners) == 1 for owners in family_owners.values()))
        self.assertGreater(exact_checked, 0)
        self.assertGreater(horizon_checked, 30)
        self.assertEqual(Counter(r["split"] for r in records), manifest["splits"])
        print(f"PHASE2_INDEPENDENT_PILOT rows={len(records)} exact_labels={exact_checked} horizon_labels={horizon_checked}")

    def test_02_all_packed_rows_decode_independently(self):
        from congklak import pilot
        path = ROOT / "data" / "pilot-10000"
        manifest = pilot.load(path)
        root = path / manifest["directory"]
        # Independent decoder for the declared little-endian wire contract.
        decoder = struct.Struct("<17shBBBB7hfdHB")
        self.assertEqual(decoder.size, 52)
        split_order = ("train", "development", "sealed_final_test")
        decoded = 0
        with closing(sqlite3.connect(root / "dataset.sqlite3")) as db:
            for body, shard, offset in db.execute("SELECT body,shard,offset FROM examples ORDER BY state"):
                record = json.loads(body)
                label = record["label"]
                with (root / shard).open("rb") as stream:
                    stream.seek(offset)
                    fields = decoder.unpack(stream.read(decoder.size))
                self.assertEqual(fields[0].hex(), record["state_key"])
                self.assertEqual(fields[1], label["target_player0"])
                self.assertEqual(fields[2], 1 if label["kind"] == "exact_optimal_margin" else 2)
                self.assertEqual(fields[3], sum(1 << move for move in label["best_moves"]))
                self.assertEqual(fields[4], record["legal_mask"])
                self.assertEqual(fields[5], 0 if label["choice_depth"] is None else label["choice_depth"])
                scores = {item["pit"]: item["value_player0"] for item in label["action_scores"]}
                self.assertEqual(fields[6:13], tuple(scores.get(a, 32767) for a in range(7)))
                self.assertEqual(fields[13], record["label_weight"])
                self.assertEqual(fields[14], record["draw_probability"])
                self.assertEqual(fields[15], manifest["strata"].index(record["stratum"]))
                self.assertEqual(fields[16], split_order.index(record["split"]))
                decoded += 1
        self.assertEqual(decoded, 10000)


class PilotRecoveryIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from congklak import pilot
        cls.module = pilot
        cls.temp_root = (ROOT / "build" / "acceptance").resolve()
        cls.temp_root.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="independent-pilot-", dir=cls.temp_root)
        cls.base = Path(cls.directory.name).resolve()
        if not cls.base.is_relative_to(cls.temp_root):
            raise AssertionError("Refuse cleanup of a pilot path outside the checked workspace")
        cls.oracle_path = ROOT / "data" / "oracle-k4.sqlite3"
        cls.fresh = cls.build(cls.base / "fresh")

    @classmethod
    def build(cls, path, **kwargs):
        return cls.module.build(path, cls.oracle_path, target=128, family_count=32, **kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_01_real_kill_label_and_unpublished_files_resume_same_content(self):
        script = """
import sys,threading
from congklak import pilot
path,oracle,stage=sys.argv[1:]
def boundary(event):
    if ((stage=='label' and event['stage']=='label' and event.get('labels',0)>=100)
        or (stage=='publish' and event['stage']=='publish' and 'rows_written' in event)):
        print('committed-boundary',flush=True)
        threading.Event().wait()
pilot.build(path,oracle,target=128,family_count=32,on_progress=boundary)
"""
        for stage in ("label", "publish"):
            path = self.base / ("killed-" + stage)
            child = subprocess.Popen([sys.executable, "-c", script, str(path), str(self.oracle_path), stage],
                                     cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            messages = queue.Queue()
            def reader():
                for line in child.stdout:
                    messages.put(line.strip())
                messages.put("eof")
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            try:
                message = messages.get(timeout=45)
                if message != "committed-boundary":
                    child.wait(timeout=10)
                    self.fail("Pilot exited before target boundary: " + child.stderr.read())
                child.kill(); child.wait(timeout=10)
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=10)
                thread.join(timeout=10)
                child.stdout.close(); child.stderr.close()
            with self.assertRaises(self.module.PilotError) as failure:
                self.module.load(path)
            self.assertEqual(failure.exception.code, "incomplete_dataset")
            resumed = self.build(path)
            self.assertEqual(resumed["semantic_sha256"], self.fresh["semantic_sha256"])
            self.assertEqual(resumed["splits"], self.fresh["splits"])
            self.assertTrue(self.module.verify(path, relabel=True)["verified"])

    def incomplete_labels(self, path):
        stopped = False
        def progress(event):
            nonlocal stopped
            if event["stage"] == "label" and event.get("labels", 0) >= 100:
                stopped = True
        with self.assertRaises(self.module.Interrupted):
            self.build(path, on_progress=progress, cancel=lambda: stopped)

    def test_02_checksum_valid_identity_and_database_key_corruption_never_publish(self):
        for mutation in ("identity", "key_binding"):
            path = self.base / ("corrupt-" + mutation)
            self.incomplete_labels(path)
            with closing(sqlite3.connect(path / "work.sqlite3")) as db:
                rows = db.execute("SELECT state,body,hash FROM labels ORDER BY state LIMIT 2").fetchall()
                self.assertEqual(len(rows), 2)
                if mutation == "identity":
                    record = json.loads(rows[0][1])
                    record["teacher_revision"] = "0" * 64
                    body = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                    checksum = hashlib.sha256(body.encode()).hexdigest()
                    db.execute("UPDATE labels SET body=?,hash=? WHERE state=?", (body, checksum, rows[0][0]))
                else:
                    # A copied valid record still must not bind to another key.
                    db.execute("UPDATE labels SET body=?,hash=? WHERE state=?", (rows[0][1], rows[0][2], rows[1][0]))
                db.commit()
            with self.assertRaises(self.module.PilotError) as failure:
                self.build(path)
            self.assertEqual(failure.exception.code, "corrupt_saved_data")
            with self.assertRaises(self.module.PilotError) as failure:
                self.module.load(path)
            self.assertEqual(failure.exception.code, "incomplete_dataset")

    def test_03_checksum_valid_collection_cursor_cannot_skip_committed_units(self):
        path = self.base / "forged-cursor"
        stopped = False
        def progress(event):
            nonlocal stopped
            if event["stage"] == "collect" and event["cursor"] >= 11:
                stopped = True
        with self.assertRaises(self.module.Interrupted):
            self.module.build(path, self.oracle_path, target=32, family_count=32,
                              on_progress=progress, cancel=lambda: stopped)
        with closing(sqlite3.connect(path / "work.sqlite3")) as db:
            body = db.execute("SELECT body FROM meta WHERE key='progress'").fetchone()[0]
            saved = json.loads(body)
            self.assertEqual(saved["cursor"], 11)
            saved["cursor"] = 32
            body = json.dumps(saved, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            checksum = hashlib.sha256(body.encode()).hexdigest()
            db.execute("UPDATE meta SET body=?,hash=? WHERE key='progress'", (body, checksum))
            db.commit()
        with self.assertRaises(self.module.PilotError) as failure:
            self.module.build(path, self.oracle_path, target=32, family_count=32)
        self.assertEqual(failure.exception.code, "corrupt_saved_data")
        with self.assertRaises(self.module.PilotError) as failure:
            self.module.load(path)
        self.assertEqual(failure.exception.code, "incomplete_dataset")


if __name__ == "__main__":
    unittest.main()
