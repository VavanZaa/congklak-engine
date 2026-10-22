"""Verified, resumable exact K=4 quotient oracle for the frozen single round.

The store offset is deliberately absent from keys. Every edge stores the change
in store margin; F is future awards, and a product lookup returns D + F.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
import hashlib
import importlib.util
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

from .config import PROFILE, RULES_DIGEST, ENCODING, InputError, canonical, digest, position, verify_rules
from .native import Native, NativeError, ROOT, flatten

K = 4
SCHEMA = "congklak-oracle-k4-v1"
DEFAULT_PATH = ROOT / "data" / "oracle-k4.sqlite3"
MIB = 1024 * 1024


class OracleError(RuntimeError):
    def __init__(self, message, code="oracle_error"):
        super().__init__(message)
        self.code = code


class OracleInterrupted(OracleError):
    def __init__(self):
        super().__init__("Oracle work stopped at a committed boundary", "interrupted")


def _storage_guard(operation):
    @wraps(operation)
    def guarded(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except (sqlite3.Error, OSError) as exc:
            raise OracleError("Oracle persistence failure: " + str(exc), "storage_error") from exc
    return guarded


def _identity(native):
    return {"schema": SCHEMA, "profile": PROFILE, "rules_digest": RULES_DIGEST,
            "state_encoding": ENCODING, "native_build_sha256": native.build_id,
            "objective": "final-store0-minus-store1", "quotient": "future-awards-F-v1",
            "active_seed_limit": K, "pits_per_side": 7,
            "domain": "all-normalized-stable-pit-compositions-R0-through4",
            "recurrence": "terminal-F0/edge-deltaD/fixed-player0-max-min-v1"}


def _compositions(total, slots):
    if slots == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for tail in _compositions(total - first, slots - 1):
                yield (first,) + tail


def _states():
    for active in range(K + 1):
        for counts in _compositions(active, 14):
            for actor in (0, 1):
                if active == 0 or any(counts[7*actor:7*(actor+1)]):
                    yield bytes(counts) + bytes([actor]), counts, actor


def _ring(counts, stores=(0, 0)):
    return list(counts[:7]) + [stores[0]] + list(counts[7:]) + [stores[1]]


def _key(pos):
    return bytes(pos["pits"][0] + pos["pits"][1]) + bytes([pos["controller"]])


@contextmanager
def _owner(path):
    lock = Path(str(path) + ".lock").open("a+b")
    try:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OracleError("Another builder owns this oracle", "oracle_busy") from exc
        yield
    finally:
        lock.close()


def _admit(path, quota_mb, min_free_mb, projected=4*MIB):
    # The parent directory is the managed-data scope, including sibling pilot,
    # analysis, WAL, temporary and lock artifacts, each physical file once.
    used = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    if used + projected > quota_mb*MIB:
        raise OracleError("Oracle write exceeds managed physical quota", "storage_error")
    if shutil.disk_usage(path.parent).free < min_free_mb*MIB + projected:
        raise OracleError("Insufficient disk space plus oracle recovery reserve", "storage_error")


def _connect(path):
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        _initialize(db)
        return db
    except Exception:
        db.close()
        raise


def _initialize(db):
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1):
        raise OracleError("Unsupported oracle schema", "incompatible_oracle")
    if version == 0 and db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
        raise OracleError("Unrecognized oracle schema", "incompatible_oracle")
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise OracleError("Damaged oracle SQLite file", "corrupt_oracle")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, body TEXT NOT NULL, checksum TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS states (
        key BLOB PRIMARY KEY, counts BLOB NOT NULL, actor INTEGER NOT NULL,
        active INTEGER NOT NULL, generated INTEGER NOT NULL DEFAULT 0,
        value INTEGER, optimal_mask INTEGER, solved INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS edges (
        source BLOB NOT NULL REFERENCES states(key), pit INTEGER NOT NULL,
        child BLOB NOT NULL REFERENCES states(key), delta INTEGER NOT NULL,
        PRIMARY KEY(source,pit));
      CREATE INDEX IF NOT EXISTS edges_child ON edges(child);
      PRAGMA user_version=1;
    """)


def _put(db, name, value):
    db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?,?)", (name, canonical(value), digest(value)))


def _get(db, name):
    size = db.execute("SELECT length(CAST(body AS BLOB)) FROM metadata WHERE name=?", (name,)).fetchone()
    if size is not None and size[0] > 65536:
        raise OracleError("Oversized oracle metadata", "corrupt_oracle")
    row = db.execute("SELECT body,checksum FROM metadata WHERE name=?", (name,)).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
        if value is None:
            raise ValueError("null metadata")
        if digest(value) != row[1]:
            raise ValueError("checksum")
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise OracleError("Damaged oracle metadata " + name, "corrupt_oracle") from exc


def _manifest_shape(manifest, identity):
    fields = {"status", "identity", "verification", "raw_allocation_upper_bound",
              "covered_normalized_states", "excluded_unnormalized_slots",
              "forced_pass_lookup", "terminal_zero_masks_are_valid"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise OracleError("Malformed oracle completion manifest fields", "corrupt_oracle")
    if (manifest["status"] != "complete" or manifest["identity"] != identity
            or type(manifest["covered_normalized_states"]) is not int
            or manifest["covered_normalized_states"] != 5462
            or type(manifest["excluded_unnormalized_slots"]) is not int
            or manifest["excluded_unnormalized_slots"] != 658
            or type(manifest["raw_allocation_upper_bound"]) is not int
            or manifest["raw_allocation_upper_bound"] != 6120
            or manifest["terminal_zero_masks_are_valid"] is not True
            or manifest["forced_pass_lookup"] != "normalize-then-lookup/return-original-controller-and-pass"
            or not isinstance(manifest["verification"], dict)):
        raise OracleError("Malformed oracle completion manifest values", "corrupt_oracle")


def _checkpoint(db, path, quota_mb, min_free_mb, stage, on_progress, cancel):
    _admit(path, quota_mb, min_free_mb)
    with db:
        _put(db, "stage", stage)
    event = {"stage": stage, "states": db.execute("SELECT COUNT(*) FROM states").fetchone()[0],
             "generated": db.execute("SELECT COUNT(*) FROM states WHERE generated=1").fetchone()[0],
             "solved": db.execute("SELECT COUNT(*) FROM states WHERE solved=1").fetchone()[0]}
    if on_progress is not None and on_progress(event) is False:
        raise OracleInterrupted()
    if cancel is not None and cancel():
        raise OracleInterrupted()


def _graph(db):
    if (db.execute("SELECT COUNT(*) FROM states").fetchone()[0] > 5462
            or db.execute("SELECT COUNT(*) FROM edges").fetchone()[0] > 9520):
        raise OracleError("Oversized oracle coverage", "corrupt_oracle")
    if db.execute("SELECT 1 FROM states WHERE typeof(key)!='blob' OR length(key)!=15 OR typeof(counts)!='blob' OR length(counts)!=14 LIMIT 1").fetchone():
        raise OracleError("Malformed oracle state BLOB dimensions", "corrupt_oracle")
    rows = {}
    for row in db.execute("SELECT * FROM states ORDER BY key"):
        if (type(row["key"]) is not bytes or len(row["key"]) != 15
                or type(row["counts"]) is not bytes or len(row["counts"]) != 14
                or type(row["actor"]) is not int or row["actor"] not in (0, 1)
                or type(row["active"]) is not int or not 0 <= row["active"] <= K
                or row["generated"] not in (0, 1) or row["solved"] not in (0, 1)
                or (row["value"] is not None and (type(row["value"]) is not int or not -K <= row["value"] <= K))
                or (row["optimal_mask"] is not None and (type(row["optimal_mask"]) is not int or not 0 <= row["optimal_mask"] < 128))):
            raise OracleError("Malformed oracle state encoding", "corrupt_oracle")
        if (row["key"] != row["counts"]+bytes([row["actor"]])
                or sum(row["counts"]) != row["active"]
                or (row["active"] != 0 and not any(row["counts"][7*row["actor"]:7*(row["actor"]+1)]))):
            raise OracleError("Oracle key/counts/normalization disagree", "corrupt_oracle")
        rows[row["key"]] = dict(row)
    edges = defaultdict(list)
    reverse = defaultdict(set)
    for row in db.execute("SELECT source,pit,child,delta FROM edges ORDER BY source,pit"):
        if (type(row[0]) is not bytes or len(row[0]) != 15 or type(row[2]) is not bytes or len(row[2]) != 15
                or type(row[1]) is not int or not 0 <= row[1] < 7
                or type(row[3]) is not int or not -4 <= row[3] <= 4):
            raise OracleError("Malformed oracle edge", "corrupt_oracle")
        source, pit, child, delta = bytes(row[0]), row[1], bytes(row[2]), row[3]
        if source not in rows or child not in rows:
            raise OracleError("Missing oracle dependency", "corrupt_oracle")
        edges[source].append((pit, child, delta))
        reverse[child].add(source)
    return rows, edges, reverse


def _solve(rows, edges, reverse):
    """Complete sink-first graph treatment; cycles are refused, never drawn."""
    pending = {key: len({child for _, child, _ in edges[key]}) for key in rows}
    ready = deque(sorted(key for key, n in pending.items() if n == 0))
    values = {}
    order = []
    while ready:
        key = ready.popleft()
        row = rows[key]
        outgoing = edges[key]
        if row["active"] == 0:
            if outgoing:
                raise OracleError("Terminal state has outgoing moves", "corrupt_oracle")
            value, mask = 0, 0
        else:
            if not outgoing:
                raise OracleError("Choice state has no legal successors", "corrupt_oracle")
            scores = [(pit, delta + values[child][0]) for pit, child, delta in outgoing]
            value = (max if row["actor"] == 0 else min)(q for _, q in scores)
            mask = sum(1 << pit for pit, q in scores if q == value)
        if abs(value) > row["active"] or (value-row["active"]) % 2:
            raise OracleError("Exact quotient violates range/conservation parity", "corrupt_oracle")
        values[key] = (value, mask)
        order.append(key)
        for parent in sorted(reverse[key]):
            pending[parent] -= 1
            if pending[parent] == 0:
                ready.append(parent)
    if len(values) != len(rows):
        raise OracleError("Dependency graph contains unresolved cycles; complete SCC semantics are required", "unsupported_cycle")
    return values, order


def _content_checksums(db):
    graph = hashlib.sha256()
    values = hashlib.sha256()
    for row in db.execute("SELECT key,counts,actor,active,generated,value,optimal_mask,solved FROM states ORDER BY key"):
        values.update(canonical([bytes(row[0]).hex(), bytes(row[1]).hex(), *row[2:]]).encode("ascii") + b"\n")
    for row in db.execute("SELECT source,pit,child,delta FROM edges ORDER BY source,pit"):
        graph.update(canonical([bytes(row[0]).hex(), row[1], bytes(row[2]).hex(), row[3]]).encode("ascii") + b"\n")
    return {"values_sha256": values.hexdigest(), "graph_sha256": graph.hexdigest()}


def _reference():
    # Ship the independent acceptance interpreter as verification evidence. It
    # imports no production module and sowing is literally seed-by-seed.
    name = "congklak_independent_oracle_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / "reference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _verify(db, identity, cancel=None):
    """Regenerate the entire domain and graph with the independent interpreter."""
    ref = _reference()
    rows, native_edges, _ = _graph(db)
    expected = {}
    for active in range(K+1):
        for counts in ref.compositions(active, 14):
            for actor in (0, 1):
                pos = ref.Position(7, tuple(_ring(counts)), actor)
                norm = ref.normalize(pos)
                if norm.controller == actor:
                    expected[bytes(counts)+bytes([actor])] = (counts, actor, active)
    if set(rows) != set(expected):
        raise OracleError("Oracle normalized coverage is incomplete or unexpected", "corrupt_oracle")
    independent = defaultdict(list)
    reverse = defaultdict(set)
    offsets_checked = edges_checked = 0
    for index, key in enumerate(sorted(expected)):
        if cancel is not None and index % 128 == 0 and cancel():
            raise OracleInterrupted()
        counts, actor, active = expected[key]
        saved = rows[key]
        if (bytes(saved["counts"]) != bytes(counts) or saved["actor"] != actor
                or saved["active"] != active or saved["generated"] != 1 or saved["solved"] != 1):
            raise OracleError("Malformed or incomplete oracle state", "corrupt_oracle")
        pos = ref.Position(7, tuple(_ring(counts)), actor)
        for pit in ref.legal_actions(pos):
            tr = ref.apply(pos, pit)
            child_counts = tr.position.counts[:7] + tr.position.counts[8:15]
            child = bytes(child_counts) + bytes([tr.position.controller])
            delta = ref.evaluate(tr.position)
            if child not in expected or sum(child_counts) > active:
                raise OracleError("Independent graph escapes coverage", "corrupt_oracle")
            expected_delta = active-sum(child_counts) if actor == 0 else -(active-sum(child_counts))
            if delta != expected_delta:
                raise OracleError("Store award accounting failed", "corrupt_oracle")
            independent[key].append((pit, child, delta))
            reverse[child].add(key)
            edges_checked += 1
            # Exercise two extreme product-98 offsets for every move. The proof
            # is structural; these checks also protect quotient implementation.
            for stores in ((98-active, 0), (0, 98-active)):
                shifted = ref.apply(ref.Position(7, tuple(_ring(counts, stores)), actor), pit)
                shifted_pits = shifted.position.counts[:7] + shifted.position.counts[8:15]
                if (shifted_pits != child_counts or shifted.position.controller != tr.position.controller
                        or ref.evaluate(shifted.position)-stores[0]+stores[1] != delta):
                    raise OracleError("Store-offset quotient does not hold", "corrupt_oracle")
                offsets_checked += 1
        if independent[key] != native_edges[key]:
            raise OracleError("Native graph differs from independent successor/award graph", "corrupt_oracle")
    values, order = _solve(rows, independent, reverse)
    for key, (value, mask) in values.items():
        if (rows[key]["value"], rows[key]["optimal_mask"]) != (value, mask):
            raise OracleError("Value or tied optimal mask differs from independent recurrence", "corrupt_oracle")
    return {"states_checked": len(values), "edges_checked": edges_checked,
            "store_offset_transitions_checked": offsets_checked,
            "dependency_graph": "acyclic-complete-sink-first", "cycles": 0,
            "independent_reference_sha256": hashlib.sha256((ROOT/"tests"/"reference.py").read_bytes()).hexdigest(),
            "all_tied_moves_checked": True, "range_and_parity_checked": True,
            "identity_sha256": digest(identity), **_content_checksums(db)}


@_storage_guard
def build(path=DEFAULT_PATH, native=None, chunk_size=128, on_progress=None, cancel=None,
          quota_mb=51200, min_free_mb=256):
    """Resume a single-owner build; only independently verified coverage publishes.

    on_progress runs *after* its chunk transaction commits. Returning False
    cooperatively interrupts, preserving the latest consistent boundary.
    """
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise OracleError("chunk_size must be 1..4096")
    if type(quota_mb) is not int or quota_mb < 0 or type(min_free_mb) is not int or min_free_mb < 0:
        raise OracleError("Storage quotas and reserve must be nonnegative integers")
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    native = native or Native()
    verify_rules()
    identity = _identity(native)
    with _owner(path):
        _admit(path, quota_mb, min_free_mb)
        db = _connect(path)
        try:
            saved = _get(db, "identity")
            if saved is not None and saved != identity:
                raise OracleError("Oracle rules/schema/native build identity changed", "incompatible_oracle")
            if saved is None:
                with db:
                    _put(db, "identity", identity)
                    _put(db, "stage", "enumerating")
            manifest = _get(db, "manifest")
            if manifest is not None:
                _manifest_shape(manifest, identity)
                if _get(db, "stage") != "complete":
                    raise OracleError("Manifest and build stage disagree", "corrupt_oracle")
                verification = _verify(db, identity, cancel)
                if canonical(verification) != canonical(manifest["verification"]):
                    raise OracleError("Published oracle checksum or verification differs", "corrupt_oracle")
                return manifest
            # INSERT OR IGNORE is idempotent after interrupted enumeration.
            count = 0
            for key, counts, actor in _states():
                if count % chunk_size == 0:
                    _admit(path, quota_mb, min_free_mb)
                db.execute("INSERT OR IGNORE INTO states(key,counts,actor,active) VALUES(?,?,?,?)",
                           (key, bytes(counts), actor, sum(counts)))
                count += 1
                if count % chunk_size == 0:
                    _checkpoint(db, path, quota_mb, min_free_mb, "enumerating", on_progress, cancel)
            _checkpoint(db, path, quota_mb, min_free_mb, "generating", on_progress, cancel)
            _graph(db)  # Validate bounded persisted keys before native work.
            rows = db.execute("SELECT key,counts,actor,active FROM states WHERE generated=0 ORDER BY active,key").fetchall()
            for index, row in enumerate(rows, 1):
                if (index-1) % chunk_size == 0:
                    _admit(path, quota_mb, min_free_mb)
                key, counts, actor, active = bytes(row[0]), tuple(row[1]), row[2], row[3]
                for pit in range(7):
                    if counts[actor*7+pit] == 0:
                        continue
                    tr = native.apply(_ring(counts), actor=actor, action=pit, total=active, trace=False)
                    if not tr.get("ok"):
                        raise OracleError("Native transition failed: " + canonical(tr))
                    child = _key(tr["position"])
                    delta = tr["store_delta"][0] - tr["store_delta"][1]
                    db.execute("INSERT INTO edges VALUES(?,?,?,?)", (key, pit, child, delta))
                db.execute("UPDATE states SET generated=1 WHERE key=?", (key,))
                if index % chunk_size == 0:
                    _checkpoint(db, path, quota_mb, min_free_mb, "generating", on_progress, cancel)
            _checkpoint(db, path, quota_mb, min_free_mb, "solving", on_progress, cancel)
            rows, edges, reverse = _graph(db)
            values, order = _solve(rows, edges, reverse)
            pending = [key for key in order if rows[key]["solved"] != 1]
            for index, key in enumerate(pending, 1):
                if (index-1) % chunk_size == 0:
                    _admit(path, quota_mb, min_free_mb)
                value, mask = values[key]
                db.execute("UPDATE states SET value=?,optimal_mask=?,solved=1 WHERE key=?", (value, mask, key))
                if index % chunk_size == 0:
                    _checkpoint(db, path, quota_mb, min_free_mb, "solving", on_progress, cancel)
            _checkpoint(db, path, quota_mb, min_free_mb, "verifying", on_progress, cancel)
            verification = _verify(db, identity, cancel)
            manifest = {"status": "complete", "identity": identity, "verification": verification,
                        "raw_allocation_upper_bound": 6120,
                        "covered_normalized_states": len(rows),
                        "excluded_unnormalized_slots": 6120-len(rows),
                        "forced_pass_lookup": "normalize-then-lookup/return-original-controller-and-pass",
                        "terminal_zero_masks_are_valid": True}
            _admit(path, quota_mb, min_free_mb)
            with db:
                _put(db, "manifest", manifest)
                _put(db, "stage", "complete")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if on_progress is not None:
                on_progress({"stage": "complete", "states": len(rows), "manifest": manifest})
            return manifest
        finally:
            db.close()


def _read_and_verify(db, native):
    if db.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise OracleError("Unsupported oracle schema", "incompatible_oracle")
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise OracleError("Damaged oracle SQLite file", "corrupt_oracle")
    identity = _get(db, "identity")
    if identity != _identity(native):
        raise OracleError("Oracle identity is incompatible", "incompatible_oracle")
    manifest = _get(db, "manifest")
    if manifest is None:
        # A NULL JSON manifest is corruption, rather than an absent publication.
        if db.execute("SELECT 1 FROM metadata WHERE name='manifest'").fetchone():
            raise OracleError("Malformed NULL completion manifest", "corrupt_oracle")
        raise OracleError("Oracle is not published complete", "incomplete_oracle")
    _manifest_shape(manifest, identity)
    if _get(db, "stage") != "complete":
        raise OracleError("Manifest and build stage disagree", "corrupt_oracle")
    verification = _verify(db, identity)
    if canonical(manifest["verification"]) != canonical(verification):
        raise OracleError("Oracle manifest differs from verification", "corrupt_oracle")
    return manifest


@_storage_guard
def verify(path=DEFAULT_PATH, native=None):
    native = native or Native()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise OracleError("Oracle artifact is absent", "oracle_not_found")
    db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        return _read_and_verify(db, native)
    finally:
        db.close()


class Oracle:
    """Read-only lookup of independently verified coverage, never incomplete rows."""
    @_storage_guard
    def __init__(self, path=DEFAULT_PATH, native=None):
        self.path = Path(path).expanduser().resolve()
        self.native = native or Native()
        if not self.path.is_file():
            raise OracleError("Oracle artifact is absent", "oracle_not_found")
        self.db = sqlite3.connect(self.path.as_uri()+"?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            # Keep the validated SQLite read snapshot for this reader's whole
            # lifetime, so an external writer cannot swap verified lookups.
            self.db.execute("BEGIN")
            self.manifest = _read_and_verify(self.db, self.native)
            self.identity = self.manifest["identity"]
            self.manifest_sha256 = digest(self.manifest)
        except Exception:
            self.close()
            raise

    @_storage_guard
    def lookup(self, value):
        value = position(value)
        active = sum(map(sum, value["pits"]))
        input_controller = value["controller"]
        normal = self.native.normalize(flatten(value), actor=input_controller, total=98)
        base = {"covered": active <= K, "input_controller": input_controller,
                "normalized_controller": normal["normalized_controller"],
                "passes": normal["passes"], "terminal": normal["terminal"],
                "active_seeds": active, "identity": self.identity,
                "oracle_manifest_sha256": self.manifest_sha256}
        if active > K:
            return base
        row = self.db.execute("SELECT value,optimal_mask,solved FROM states WHERE key=?", (_key(normal["position"]),)).fetchone()
        if row is None or row["solved"] != 1:
            raise OracleError("Published coverage has a missing lookup", "corrupt_oracle")
        score = value["stores"][0] - value["stores"][1] + row["value"]
        base.update({"future_awards_player0": row["value"], "value_player0": score,
                     "value_actor": score if normal["normalized_controller"] == 0 else -score,
                     "optimal_mask": row["optimal_mask"],
                     "best_moves": [pit for pit in range(7) if row["optimal_mask"] & (1 << pit)],
                     "label_kind": "exact_optimal_margin", "scope": "eventual-optimal-final-margin"})
        return base

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("build", "verify", "lookup"))
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--quota-mb", type=int, default=51200)
    parser.add_argument("--min-free-mb", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        if args.operation == "build":
            def progress(event):
                print(canonical({"event": "oracle_progress", **event}), flush=True)
            result = build(args.path, chunk_size=args.chunk_size, on_progress=progress,
                           quota_mb=args.quota_mb, min_free_mb=args.min_free_mb)
        elif args.operation == "verify":
            result = verify(args.path)
        else:
            with Oracle(args.path) as oracle:
                result = oracle.lookup(json.loads(sys.stdin.readline()))
        print(canonical({"ok": True, "result": result}), flush=True)
        return 0
    except (OracleError, InputError, NativeError, ValueError, KeyboardInterrupt) as exc:
        print(canonical({"ok": False, "error": {"code": getattr(exc, "code", "interrupted"), "message": str(exc)}}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
