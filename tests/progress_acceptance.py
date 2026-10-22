"""Progress authority, recovery observation and tool exit-code acceptance."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from congklak import config, progress


class ProgressAcceptance(unittest.TestCase):
    def test_01_observer_uses_committed_catalog_not_historical_manifest_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data/stronger-10000-v2"
            path.mkdir(parents=True)
            generation = path / "generation-a"
            generation.mkdir()
            (generation / "rows.cgp").write_bytes(b"rows")
            db = sqlite3.connect(path / "work.sqlite3")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("CREATE TABLE meta(key TEXT,body TEXT,hash TEXT); CREATE TABLE artifacts(id TEXT,status TEXT,hash TEXT); CREATE TABLE collection_units(idx INTEGER); CREATE TABLE labels(state BLOB); CREATE TABLE candidates(state BLOB);")
            identity = {"target": 1, "family_count": 1, "imported_registry_sha256": "registry"}
            manifest = dict(identity=identity, records=1, generation="a", directory="generation-a",
                            semantic_sha256="semantic", splits={"train": 1, "development": 0, "sealed_final_test": 0},
                            files=[dict(name="rows.cgp", size=4)], generation_statistics={"stage": "publish"}, q_equals_P=True)
            for key, body in [("identity", identity), ("progress", {"stage": "complete"}), ("current", manifest)]:
                db.execute("INSERT INTO meta VALUES(?,?,?)", (key, config.canonical(body), config.digest(body)))
            db.execute("INSERT INTO artifacts VALUES(?,?,?)", ("a", "complete", config.digest(manifest)))
            db.execute("INSERT INTO labels VALUES(?)", (b"committed",))
            db.execute("INSERT INTO collection_units VALUES(0)")
            db.commit()
            # An uncommitted row must not appear, and the observer must not
            # acquire the writer lock or mutate the producer's transaction.
            db.execute("INSERT INTO labels VALUES(?)", (b"not committed",))
            value = progress.corpus(path, "campaign", "Corpus")
            self.assertEqual(value["state"], "complete")
            self.assertEqual(value["counters"][1]["completed"], 1)
            self.assertTrue(db.in_transaction)
            db.rollback()
            db.execute("UPDATE meta SET hash='wrong' WHERE key='current'")
            db.commit()
            value = progress.snapshot(root)
            c = next(t for t in value["tools"] if t["id"] == "campaign")
            self.assertEqual(c["state"], "read_error")
            db.close()

    def test_02_finished_artifact_is_not_trusted_without_required_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            db = sqlite3.connect(path / "catalog.sqlite3")
            db.executescript("CREATE TABLE config(body TEXT,hash TEXT); CREATE TABLE checkpoints(id INTEGER,step INTEGER,name TEXT,sha TEXT,body TEXT,hash TEXT);")
            identity = {"width": 256, "seed": 1, "steps": 2000, "device": "cuda"}
            db.execute("INSERT INTO config VALUES(?,?)", (config.canonical(identity), config.digest(identity)))
            meta = {"metrics": {}}
            db.execute("INSERT INTO checkpoints VALUES(1,2000,'missing.pt','sha',?,?)", (config.canonical(meta), config.digest(meta)))
            db.commit()
            db.close()
            with self.assertRaisesRegex(ValueError, "checkpoint missing"):
                progress.training(path)
            progress.atomic_json(path / "quarantine.json", {"reason": "alias leak"})
            self.assertEqual(progress.training(path)["state"], "quarantined")

    def test_03_stale_acceptance_and_reused_pid_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("new version")
            report = {"inputs": {"source.py": "old digest"}}
            self.assertEqual(progress.current_evidence(root, report), ["source.py"])
            progress.atomic_json(root / "build/progress/run-fake.json", dict(state="running", pid=123, process_birth=1, tool="campaign"))
            with patch.object(progress, "process_identity", return_value=(True, 2)):
                self.assertEqual(progress.runs(root)[0]["state"], "interrupted_or_exited")
            with patch.object(progress, "process_identity", return_value=(True, 1)):
                self.assertEqual(progress.runs(root)[0]["state"], "running")

    def test_04_wrapper_streams_events_preserves_logs_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            program = "import json,sys; print(json.dumps({'event':'benchmark_case_committed','completed':7,'total':10}),flush=True); sys.exit(3)"
            output = io.StringIO()
            with redirect_stdout(output):
                code = progress.run_command("benchmark", [sys.executable, "-u", "-c", program], root)
            self.assertEqual(code, 3)
            self.assertIn("completed=7 | total=10", output.getvalue())
            state = progress.read_json(next((root / "build/progress").glob("run-*.json")))
            self.assertEqual((state["state"], state["exit_code"]), ("failed", 3))
            self.assertIn('"completed": 7', Path(state["log"]).read_text())

    def test_05_project_status_and_protected_path_behavior(self):
        value = progress.snapshot()
        by_id = {t["id"]: t for t in value["tools"]}
        self.assertEqual(by_id["campaign"]["state"], "complete")
        self.assertEqual(by_id["campaign"]["counters"][1]["completed"], 10000)
        self.assertEqual(by_id["benchmark"]["counters"][1]["completed"], 24)
        self.assertEqual(by_id["promotion"]["state"], "held")
        self.assertNotIn('"state_key"', json.dumps(value))
        with self.assertRaises(ValueError):
            progress.inside(progress.ROOT, "../outside")
        with self.assertRaisesRegex(ValueError, "Unknown flow tool"):
            progress.command("unknown", [])

    def test_06_atomic_progress_retries_transient_windows_sharing_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"status.json"
            progress.atomic_json(path,{"completed":1})
            real_replace=progress.os.replace; attempts=[]
            def replace(source,target):
                attempts.append(str(source))
                if len(attempts)==1:
                    self.assertEqual(progress.read_json(path),{"completed":1})
                    raise PermissionError("simulated Windows reader sharing collision")
                return real_replace(source,target)
            with patch.object(progress.os,"replace",side_effect=replace):
                progress.atomic_json(path,{"completed":2})
            self.assertEqual(progress.read_json(path),{"completed":2})
            self.assertEqual(len(attempts),2)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_07_stage5_counters_override_unfinished_static_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"data/stage5-benchmark";path.mkdir(parents=True)
            db=sqlite3.connect(path/"work.sqlite3")
            db.executescript("CREATE TABLE meta(key TEXT,body TEXT,hash TEXT); CREATE TABLE cases(idx INTEGER); CREATE TABLE games(pair INTEGER);")
            db.execute("INSERT INTO cases VALUES(0)");db.commit();db.close()
            progress.atomic_json(root/"build/stage5-progress.json",{"schema":"stage5-work-v1","tasks":[{
                "id":"stage5-strength","title":"Stage 5 strength","state":"complete","result":"stale manual text","improvement":"Finish the ledger."},
                {"id":"stage5-features","title":"Completed training","state":"complete","result":"Ordinary gate failed.","improvement":"Reduce ordinary bias."}]})
            value=progress.snapshot(root)
            tool=next(t for t in value["tools"] if t["id"]=="stage5-strength")
            self.assertEqual(tool["state"],"partial")
            self.assertEqual(tool["counters"][0],{"label":"position cases committed","completed":1,"total":407})
            self.assertTrue(next(t for t in value["tools"] if t["id"]=="stage5-features")["needs_improvement"])

    def test_09_action_progress_requires_committed_counts_and_matching_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"data/stage5-action-benchmark";path.mkdir(parents=True)
            db=sqlite3.connect(path/"work.sqlite3")
            db.executescript("CREATE TABLE meta(key TEXT,body TEXT,hash TEXT); CREATE TABLE cases(idx INTEGER); CREATE TABLE games(pair INTEGER);")
            db.executemany("INSERT INTO cases VALUES(?)",((i,) for i in range(391)))
            db.executemany("INSERT INTO games VALUES(?)",((i,) for i in range(144)))
            report={"cases":391,"passed":True}; marker={"report_sha256":config.digest(report)}
            db.execute("INSERT INTO meta VALUES('action_complete',?,?)",(config.canonical(marker),config.digest(marker)))
            db.commit();db.close()
            progress.atomic_json(root/"build/stage5-action-benchmark.json",report)
            progress.atomic_json(root/"build/stage5-action-progress.json",{"schema":"stage5-action-work-v1","tasks":[{
                "id":"stage5-action-strength","title":"Action strength","state":"building","result":"stale manual text","improvement":"Audit the ledger."}]})
            tool=next(t for t in progress.snapshot(root)["tools"] if t["id"]=="stage5-action-strength")
            self.assertEqual(tool["state"],"complete")
            self.assertEqual([c["completed"] for c in tool["counters"]],[391,144])
            progress.atomic_json(root/"build/stage5-action-benchmark.json",{"changed":True})
            tool=next(t for t in progress.snapshot(root)["tools"] if t["id"]=="stage5-action-strength")
            self.assertEqual(tool["state"],"partial")

    def test_10_deeper_data_counts_override_static_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"data/deeper-9000-v2";path.mkdir(parents=True)
            with sqlite3.connect(path/"work.sqlite3") as db:
                db.executescript("CREATE TABLE meta(key TEXT,body TEXT,hash TEXT); CREATE TABLE labels(state TEXT); CREATE TABLE features(state TEXT);")
                db.execute("INSERT INTO labels VALUES('a')");db.executemany("INSERT INTO features VALUES(?)",[('a',),('b',)])
            db.close()
            progress.atomic_json(root/"build/stage5-deeper-progress.json",{"schema":"stage5-deeper-work-v1","teacher_run":"data/deeper-9000-v2","tasks":[{
                "id":"stage5-deeper-data","title":"H8 generation","state":"complete","result":"outdated manual status","improvement":"Finish labels."}]})
            tool=next(t for t in progress.snapshot(root)["tools"] if t["id"]=="stage5-deeper-data")
            self.assertEqual(tool["state"],"partial")
            self.assertEqual([c["completed"] for c in tool["counters"]],[1,2])
            self.assertEqual([c["total"] for c in tool["counters"]],[9000,9000])

    def test_08_retention_preserves_logs_pinned_by_scientific_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);logs=root/"build/progress";logs.mkdir(parents=True)
            for i in range(101):
                file=logs/f"run-{i:04d}.json"
                progress.atomic_json(file,{"state":"complete"})
                file.with_suffix(".log").write_text(f"evidence {i}")
                progress.os.utime(file,(i+1,i+1))
            pinned=logs/"run-0000.log"
            progress.atomic_json(root/"build/phase3-acceptance.json",{"passed":True,"inputs":{str(pinned.relative_to(root)):progress.sha(pinned)}})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(progress.run_command("test",[sys.executable,"-c","print('done')"],root),0)
            self.assertEqual(pinned.read_text(),"evidence 0")
            self.assertTrue(pinned.with_suffix(".json").exists())
            self.assertFalse((logs/"run-0001.log").exists())


if __name__ == "__main__":
    unittest.main()
