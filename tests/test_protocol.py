"""Black-box NDJSON protocol checks using a real CLI subprocess and local data.

Run with ``py -3.12 -m unittest discover -s tests -p test_protocol.py -v``.
The subprocess's reports stay in pipes; assertions print only small summaries.
"""

import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parent.parent
WORK = (ROOT / "build" / "acceptance").resolve()
OPENING = {"pits": [[7] * 7, [7] * 7], "stores": [0, 0], "controller": 0}
FORCED_PASS = {"pits": [[0] * 7, [0] * 6 + [1]],
               "stores": [49, 48], "controller": 0}
TERMINAL = {"pits": [[0] * 7, [0] * 7], "stores": [49, 49], "controller": 1}
UNLIMITED = {"time_ms": 0, "node_budget": 0}


def brief(event):
    if not isinstance(event, dict):
        return repr(event)[:160]
    return {key: event[key] for key in
            ("event", "id", "ok", "job_id", "stop_reason", "last_completed_depth", "error")
            if key in event}


class ProtocolClient:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.is_relative_to(WORK):
            raise AssertionError("Protocol test data must remain under build/acceptance")
        self.messages = queue.Queue()
        self.pending = []
        self.seen = []
        self.errors = []
        self.process = subprocess.Popen(
            ["py", "-3.12", "-m", "congklak", "--data-dir", str(self.data_dir)],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.error_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader.start()
        self.error_reader.start()
        self.ready = self.wait("ready", timeout=15)

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except (ValueError, TypeError) as exc:
                    self.messages.put({"event": "invalid_output", "error": str(exc),
                                       "excerpt": line[:160]})
        finally:
            self.messages.put({"event": "process_eof"})

    def _read_stderr(self):
        for line in self.process.stderr:
            if sum(map(len, self.errors)) < 8192:
                self.errors.append(line[:1024])

    def send(self, request):
        self.send_raw(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")

    def send_raw(self, line):
        if isinstance(line, bytes):
            self.process.stdin.buffer.write(line)
            self.process.stdin.buffer.flush()
        else:
            self.process.stdin.write(line)
            self.process.stdin.flush()

    def wait(self, event, request_id=None, timeout=15):
        def matches(item):
            return item.get("event") == event and (request_id is None or item.get("id") == request_id)
        for index, item in enumerate(self.pending):
            if matches(item):
                return self.pending.pop(index)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"Timed out waiting for {event}/{request_id}; "
                                     f"recent={[brief(x) for x in self.seen[-4:]]}; "
                                     f"stderr={''.join(self.errors)[-1000:]}")
            try:
                item = self.messages.get(timeout=remaining)
            except queue.Empty:
                continue
            self.seen.append(item)
            if matches(item):
                return item
            if item.get("event") in ("invalid_output", "process_eof"):
                raise AssertionError(f"Unexpected CLI output/end: {brief(item)}; "
                                     f"stderr={''.join(self.errors)[-1000:]}")
            self.pending.append(item)

    def finish_input(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def wait_exit(self, timeout=15):
        code = self.process.wait(timeout=timeout)
        self.reader.join(timeout=2)
        self.error_reader.join(timeout=2)
        return code

    def close(self):
        if self.process.poll() is None:
            try:
                if self.process.stdin and not self.process.stdin.closed:
                    self.send({"op": "stop", "id": "cleanup-stop"})
                    self.finish_input()
                self.wait_exit(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=10)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ProtocolAcceptance(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="protocol_", dir=WORK)
        self.data_dir = Path(self.directory.name).resolve()
        self.assertTrue(self.data_dir.is_relative_to(WORK))
        self.client = ProtocolClient(self.data_dir)
        self.sequence = 0

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def request(self, op, event, **values):
        self.sequence += 1
        request_id = f"r{self.sequence}"
        self.client.send({"op": op, "id": request_id, **values})
        return self.client.wait(event, request_id)

    def invalid(self, request, code=None):
        self.sequence += 1
        request_id = f"bad{self.sequence}"
        self.client.send({**request, "id": request_id})
        response = self.client.wait("error", request_id)
        self.assertFalse(response["ok"], brief(response))
        self.assertIsInstance(response["error"]["code"], str)
        if code is not None:
            self.assertEqual(response["error"]["code"], code, brief(response))
        return response

    def analyze(self, request_id, limits, position=None):
        request = {"op": "analyze", "id": request_id, "limits": limits}
        if position is not None:
            request["position"] = position
        self.client.send(request)
        started = self.client.wait("job_started", request_id)
        done = self.client.wait("analysis_done", request_id)
        self.assertTrue(done["ok"], brief(done))
        return started, done

    def test_01_ready_defaults_and_default_analysis(self):
        ready = self.client.ready
        self.assertTrue(ready["ok"])
        self.assertEqual(ready["protocol"], "congklak-ndjson-v1")
        self.assertEqual(ready["profile_status"], "experimental_engineering_contract")
        self.assertEqual(ready["defaults"],
                         {"max_depth": 16, "time_ms": 1000, "node_budget": 1000000, "memory_mb": 512})
        self.assertEqual(Path(ready["data_dir"]), self.data_dir)
        self.client.send({"op": "analyze", "id": "default"})
        started = self.client.wait("job_started", "default")
        self.assertEqual(started["limits"], {**ready["defaults"],**ready["search_defaults"]})
        self.assertEqual(started["start_depth"], 1)
        self.assertEqual([c["pit"] for c in started["candidates"]], list(range(7)))
        self.assertTrue(all(c["actor"] == 0 and c["depth_result"] is None
                            for c in started["candidates"]))
        done = self.client.wait("analysis_done", "default")
        self.assertTrue(done["ok"], brief(done))
        self.assertIn(done["stop_reason"], ("max_depth", "node_budget", "time_budget"))
        self.assertLessEqual(done["work"]["nodes"], 1000000)
        self.assertLessEqual(done["last_completed_depth"], 16)
        self.assertIsNotNone(done["report"])

    def test_02_validate_apply_forced_pass_terminal_and_resume(self):
        valid = self.request("validate", "validated", position=OPENING)
        self.assertTrue(valid["ok"])
        self.assertEqual(valid["legal_actions"], list(range(7)))
        self.assertEqual(valid["position"], OPENING)
        applied = self.request("apply", "applied", position=OPENING, action=0, trace=True)
        self.assertTrue(applied["ok"])
        self.assertEqual(applied["actor"], 0)
        self.assertTrue(applied["events"])
        self.assertEqual(sum(sum(row) for row in applied["position"]["pits"])
                         + sum(applied["position"]["stores"]), 98)
        no_trace = self.request("apply", "applied", position=OPENING, action=0, trace=False)
        self.assertEqual(no_trace["position"], applied["position"])
        self.assertEqual(no_trace["events"], [])
        passed = self.request("validate", "validated", position=FORCED_PASS)
        self.assertEqual(passed["input_controller"], 0)
        self.assertEqual(passed["normalized_controller"], 1)
        self.assertEqual(passed["passes"], [{"from": 0, "to": 1}])
        illegal = self.request("apply", "applied", position=FORCED_PASS, action=6)
        self.assertFalse(illegal["ok"])
        self.assertEqual(illegal["error"]["code"], "forced_pass_required")
        explicit = self.request("apply", "applied", position=FORCED_PASS, action="pass")
        self.assertEqual(explicit["action"], "pass")
        self.assertEqual(explicit["events"], [{"type": "pass", "actor": 0, "to": 1}])
        self.assertEqual(explicit["position"]["controller"], 1)
        started, done = self.analyze("forced", {**UNLIMITED, "max_depth": 2}, FORCED_PASS)
        self.assertEqual(started["candidates"], [{"actor": 1, "pit": 6, "depth_result": None}])
        report = done["report"]
        self.assertEqual(report["input_controller"], 0)
        self.assertEqual(report["normalized_controller"], 1)
        self.assertEqual(report["forced_passes"], [{"from": 0, "to": 1}])
        self.assertEqual(report["score_player"], 1)
        self.assertEqual(report["pv"][0]["actor"], 1)
        self.assertFalse(report["pv_truncated"])
        _, terminal_done = self.analyze("terminal", {**UNLIMITED, "max_depth": 4}, TERMINAL)
        terminal_report = terminal_done["report"]
        self.assertEqual(terminal_done["last_completed_depth"], 0)
        self.assertTrue(terminal_report["terminal"])
        self.assertEqual(terminal_report["score_player"], 1)
        self.assertEqual(terminal_report["best_moves"], [])
        self.assertEqual(terminal_report["candidates"], [])
        self.assertFalse(terminal_report["pv_truncated"])
        first, _ = self.analyze("one", {**UNLIMITED, "max_depth": 1})
        loaded = self.request("load", "loaded", job_id=first["job_id"])["job"]
        self.assertEqual(loaded["last_completed_depth"], 1)
        self.client.send({"op": "resume", "id": "resume", "job_id": first["job_id"],
                          "limits": {**UNLIMITED, "max_depth": 2}})
        resumed = self.client.wait("job_started", "resume")
        self.assertEqual(resumed["start_depth"], 2)
        resumed_done = self.client.wait("analysis_done", "resume")
        self.assertTrue(resumed_done["ok"], brief(resumed_done))
        self.assertEqual(resumed_done["last_completed_depth"], 2)
        self.assertEqual([x["report"]["choice_depth"] for x in self.client.seen
                          if x.get("event") == "iteration" and x.get("id") == "resume"], [2])
        already = self.request("resume", "analysis_done", job_id=first["job_id"])
        self.assertEqual(already["stop_reason"], "already_complete")
        self.invalid({"op": "load", "job_id": "not-a-saved-job"}, "invalid_job_id")
        self.invalid({"op": "load", "job_id": "0" * 32}, "job_not_found")

    def test_03_malformed_requests_are_structured_and_recoverable(self):
        for line in ("{\n", "null\n", "[]\n", '{"op":"validate","x":NaN}\n', b"\xff\n"):
            with self.subTest(raw=repr(line)[:60]):
                self.client.send_raw(line)
                response = self.client.wait("error")
                self.assertFalse(response["ok"])
                self.assertIsInstance(response["error"]["code"], str)
        invalid_requests = [
            {"op": "missing"}, {"op": "validate", "profile": "other-profile"},
            {"op": "validate", "rules_digest": "wrong"},
            {"op": "validate", "state_encoding": "uncertain"},
            {"op": "validate", "unexpected": True}, {"op": "apply"},
            {"op": "apply", "action": True}, {"op": "apply", "action": 7},
            {"op": "apply", "action": "pass"},
            {"op": "apply", "position": TERMINAL, "action": "pass"},
            {"op": "apply", "action": 0, "trace": "yes"},
            {"op": "apply", "position": FORCED_PASS, "action": "pass", "trace": 1},
        ]
        for value in (True, -1, 1.5, "7", 99):
            position = copy.deepcopy(OPENING)
            position["pits"][0][0] = value
            invalid_requests.append({"op": "validate", "position": position})
        for value in (True, 2, "0"):
            position = copy.deepcopy(OPENING)
            position["controller"] = value
            invalid_requests.append({"op": "validate", "position": position})
        position = copy.deepcopy(OPENING)
        position["pits"][0].pop()
        invalid_requests.append({"op": "validate", "position": position})
        position = copy.deepcopy(OPENING)
        del position["controller"]
        invalid_requests.append({"op": "validate", "position": position})
        position = copy.deepcopy(OPENING)
        position["pits"][0][0] = 6
        invalid_requests.append({"op": "validate", "position": position})
        for request in invalid_requests:
            with self.subTest(request=repr(request)[:160]):
                self.invalid(request)
        self.client.send_raw(" " * 70000 + "\n")
        too_large = self.client.wait("error")
        self.assertEqual(too_large["error"]["code"], "request_too_large")
        self.client.send_raw("[" * 2000 + "0" + "]" * 2000 + "\n")
        nested = self.client.wait("error")
        self.assertFalse(nested["ok"])
        after = self.request("validate", "validated")
        self.assertTrue(after["ok"])

    def test_04_limits_no_completed_ranking_and_explicit_depth_zero(self):
        bad_limits = [
            {"max_depth": -1}, {"max_depth": 129}, {"max_depth": True},
            {"time_ms": -1}, {"time_ms": 86400001}, {"time_ms": 1.5},
            {"node_budget": -1}, {"node_budget": 2**63}, {"node_budget": "100"},
            {"memory_mb": 15}, {"memory_mb": 24577}, {"memory_mb": True},
            {"workers": 2},
        ]
        for limits in bad_limits:
            with self.subTest(limits=limits):
                self.invalid({"op": "analyze", "limits": limits})
        started, done = self.analyze("node-one", {"max_depth": 4, "time_ms": 0, "node_budget": 1})
        self.assertEqual(done["stop_reason"], "node_budget")
        self.assertEqual(done["work"]["nodes"], 1)
        self.assertEqual(done["last_completed_depth"], -1)
        self.assertIsNone(done["report"])
        self.assertTrue(all(c["depth_result"] is None for c in started["candidates"]))
        saved = self.request("load", "loaded", job_id=started["job_id"])["job"]
        self.assertIsNone(saved["report"])
        self.assertEqual(saved["last_completed_depth"], -1)
        _, done = self.analyze("time", {"max_depth": 128, "time_ms": 1, "node_budget": 0})
        self.assertEqual(done["stop_reason"], "time_budget")
        if done["report"] is not None:
            self.assertEqual(done["report"]["choice_depth"], done["last_completed_depth"])
            self.assertTrue(done["report"]["completed_iteration"])
        _, done = self.analyze("memory", {**UNLIMITED, "max_depth": 4, "memory_mb": 16})
        self.assertEqual(done["stop_reason"], "memory_budget")
        self.assertIsNone(done["report"])
        self.assertGreaterEqual(done["work"]["peak_process_bytes"], 16 * 1024 * 1024)
        zero, done = self.analyze("zero", {**UNLIMITED, "max_depth": 0})
        self.assertEqual(zero["start_depth"], 0)
        self.assertEqual(done["last_completed_depth"], 0)
        report = done["report"]
        self.assertEqual(report["depth_result"]["value_player0"], 0)
        self.assertEqual(report["best_moves"], [])
        self.assertTrue(all(c["depth_result"] is None for c in report["candidates"]))
        self.assertTrue(report["pv_truncated"])

    def test_05_single_worker_busy_stop_and_read_responsiveness(self):
        self.client.send({"op": "analyze", "id": "long", "limits": {**UNLIMITED, "max_depth": 128}})
        started = self.client.wait("job_started", "long")
        began = time.monotonic()
        self.client.send({"op": "analyze", "id": "busy", "limits": {"max_depth": 0}})
        busy = self.client.wait("error", "busy", timeout=3)
        self.assertEqual(busy["error"]["code"], "busy")
        self.client.send({"op": "stop", "id": "wrong-stop", "job_id": "different"})
        wrong = self.client.wait("error", "wrong-stop", timeout=3)
        self.assertEqual(wrong["error"]["code"], "job_not_active")
        read = self.request("validate", "validated")
        self.assertTrue(read["ok"])
        self.client.send({"op": "stop", "id": "stop", "job_id": started["job_id"]})
        stopped = self.client.wait("stop_requested", "stop", timeout=3)
        self.assertTrue(stopped["active"])
        done = self.client.wait("analysis_done", "long", timeout=3)
        self.assertLess(time.monotonic() - began, 3)
        self.assertEqual(done["stop_reason"], "cancelled")
        self.assertTrue(done["ok"], brief(done))
        iterations = [x["report"] for x in self.client.seen
                      if x.get("event") == "iteration" and x.get("id") == "long"]
        self.assertEqual(done["last_completed_depth"], iterations[-1]["choice_depth"] if iterations else -1)
        self.assertEqual(done["report"], iterations[-1] if iterations else None)
        inactive = self.request("stop", "stop_requested")
        self.assertFalse(inactive["active"])
        loaded = self.request("load", "loaded", job_id=started["job_id"])["job"]
        self.assertEqual(loaded["report"], done["report"])
        self.assertEqual(loaded["status"], "stopped")

    def test_06_eof_waits_for_active_analysis_and_saved_report(self):
        self.client.send({"op": "analyze", "id": "eof", "limits": {**UNLIMITED, "max_depth": 3}})
        self.client.finish_input()
        started = self.client.wait("job_started", "eof")
        done = self.client.wait("analysis_done", "eof")
        self.assertTrue(done["ok"], brief(done))
        self.assertEqual(done["stop_reason"], "max_depth")
        self.assertEqual(done["last_completed_depth"], 3)
        self.assertEqual(self.client.wait_exit(), 0)
        self.assertEqual([x["report"]["choice_depth"] for x in self.client.seen
                          if x.get("event") == "iteration" and x.get("id") == "eof"], [1, 2, 3])
        with ProtocolClient(self.data_dir) as restarted:
            restarted.send({"op": "load", "id": "reloaded", "job_id": started["job_id"]})
            loaded = restarted.wait("loaded", "reloaded")["job"]
            self.assertEqual(loaded["last_completed_depth"], 3)
            self.assertEqual(loaded["report"], done["report"])
            self.assertEqual(loaded["status"], "complete")
            self.assertTrue(loaded["compatible"])


if __name__ == "__main__":
    unittest.main()
