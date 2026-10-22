"""Real child-process termination plus transactional storage fault acceptance."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from congklak import config
from congklak.native import ROOT,Native
from congklak.storage import Store,StorageError,MIB

TEMP_ROOT=ROOT/"build"/"acceptance"
TEMP_ROOT.mkdir(parents=True,exist_ok=True)

class Session:
    def __init__(self,path,*args):
        self.process=subprocess.Popen([sys.executable,"-m","congklak","--data-dir",str(path),*args],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8")
        self.events=queue.Queue()
        def reader():
            for line in self.process.stdout:
                try: self.events.put(json.loads(line))
                except Exception as exc: self.events.put({"event":"reader_error","message":str(exc)})
            self.events.put({"event":"eof"})
        self.reader=threading.Thread(target=reader,daemon=True)
        self.reader.start()
        self.ready=self.get()

    def send(self,op,**args):
        self.process.stdin.write(config.canonical({"op":op,**args})+"\n")
        self.process.stdin.flush()

    def get(self,event=None,timeout=20):
        while True:
            item=self.events.get(timeout=timeout)
            if event is None or item["event"] in (event,"error","eof"):
                return item

    def kill(self):
        self.process.kill()
        self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr): stream.close()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr): stream.close()

def semantics(report):
    return {"root_value":report["depth_result"]["value_player0"],"depth":report["choice_depth"],
            "best_moves":report["best_moves"],"pv":report["pv"],"truncated":report["pv_truncated"],
            "candidates":[(c["pit"],c["depth_result"]["value_player0"],c["pv"],c["pv_truncated"])
                          for c in report["candidates"]]}

class RecoveryAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native=Native()
        cls.identity=config.identity(cls.native.build_id)
        cls.root=config.position()
        cls.full_limits=config.limits({"max_depth":6,"time_ms":0,"node_budget":0})
        cls.fresh=cls.native.analyze([7]*7+[0]+[7]*7+[0],**cls.full_limits)["report"]

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="recovery-",dir=TEMP_ROOT)
        self.path=Path(self.temp.name).resolve()
        self.assertTrue(self.path.is_relative_to(TEMP_ROOT.resolve()))

    def tearDown(self):
        # Verified workspace-contained temporary directory only.
        self.temp.cleanup()

    def store(self,**kwargs):
        return Store(self.path,self.identity,**kwargs)

    def report(self,depth):
        return self.native.analyze([7]*7+[0]+[7]*7+[0],max_depth=depth)["report"]

    def test_01_kill_after_report_commit_resume_matches_fixed_horizon(self):
        first=Session(self.path)
        try:
            self.assertEqual(first.ready["event"],"ready")
            first.send("analyze",limits={"max_depth":128,"time_ms":0,"node_budget":0})
            start=first.get("job_started")
            complete=first.get("iteration")
            self.assertTrue(complete["report"]["completed_iteration"])
            job_id=start["job_id"]
        finally: first.kill()
        second=Session(self.path)
        try:
            second.send("load",job_id=job_id)
            loaded=second.get("loaded")["job"]
            self.assertIsNotNone(loaded["report"])
            self.assertGreaterEqual(loaded["last_completed_depth"],complete["report"]["choice_depth"])
            self.assertEqual(loaded["status"],"interrupted")
            target=max(6,loaded["last_completed_depth"]+1)
            self.assertLessEqual(target,10,"Kill test did not interrupt promptly")
            second.send("resume",job_id=job_id,limits={"max_depth":target,"time_ms":0,"node_budget":0})
            done=second.get("analysis_done")
            self.assertTrue(done["ok"],done)
            expected=self.fresh if target==6 else self.report(target)
            self.assertEqual(semantics(done["report"]),semantics(expected))
            second.send("load",job_id=job_id)
            loaded=second.get("loaded")["job"]
            self.assertEqual(len(loaded["runs"]),2)
            self.assertEqual(loaded["runs"][0]["stop_reason"],"process_interrupted")
            self.assertEqual(loaded["runs"][1]["settings"]["max_depth"],target)
        finally: second.close()

    def test_02_kill_before_first_report_and_after_job_commit(self):
        # A 1-node budget deterministically cannot finish depth1, preserving a
        # durable unranked job. Hard kill then restart must never invent a score.
        first=Session(self.path)
        try:
            first.send("analyze",limits={"max_depth":128,"time_ms":0,"node_budget":1})
            start=first.get("job_started")
            job_id=start["job_id"]
        finally: first.kill()
        second=Session(self.path)
        try:
            second.send("load",job_id=job_id)
            loaded=second.get("loaded")["job"]
            self.assertIsNone(loaded["report"])
            self.assertEqual(loaded["last_completed_depth"],-1)
            second.send("resume",job_id=job_id,limits={"max_depth":6,"time_ms":0,"node_budget":0})
            done=second.get("analysis_done")
            self.assertEqual(semantics(done["report"]),semantics(self.fresh))
        finally: second.close()

    def test_03_pending_job_before_run_commit(self):
        store=self.store()
        job_id=store.create(self.root,self.full_limits)
        store.close()
        session=Session(self.path)
        try:
            session.send("load",job_id=job_id)
            saved=session.get("loaded")["job"]
            self.assertEqual(saved["status"],"pending")
            self.assertIsNone(saved["report"])
            session.send("resume",job_id=job_id,limits={"max_depth":6,"time_ms":0,"node_budget":0})
            self.assertEqual(semantics(session.get("analysis_done")["report"]),semantics(self.fresh))
        finally: session.close()

    def test_04_damaged_latest_falls_back_and_recomputes(self):
        store=self.store()
        job_id=store.create(self.root,self.full_limits)
        for depth in (1,2,3): store.commit_report(job_id,self.report(depth))
        self.assertEqual(store.db.execute("SELECT count(*) FROM reports").fetchone()[0],2)
        with store.db: store.db.execute("UPDATE reports SET body='broken' WHERE depth=3")
        saved=store.load(job_id,compatible=True)
        self.assertEqual(saved["last_completed_depth"],2)
        self.assertTrue(saved["recovery"]["restored_previous_generation"])
        store.close()
        session=Session(self.path)
        try:
            session.send("resume",job_id=job_id,limits={"max_depth":6,"time_ms":0,"node_budget":0})
            self.assertEqual(semantics(session.get("analysis_done")["report"]),semantics(self.fresh))
        finally: session.close()

    def test_05_malformed_and_incompatible_input(self):
        store=self.store()
        job_id=store.create(self.root,self.full_limits)
        payload=json.loads(store.db.execute("SELECT payload FROM jobs").fetchone()[0])
        payload["identity"]["native_build_sha256"]="other-build"
        with store.db:
            store.db.execute("UPDATE jobs SET payload=?,payload_hash=?",(config.canonical(payload),config.digest(payload)))
        self.assertFalse(store.load(job_id)["compatible"])
        with self.assertRaisesRegex(StorageError,"differ"): store.load(job_id,compatible=True)
        with store.db: store.db.execute("UPDATE jobs SET payload='{}'")
        with self.assertRaisesRegex(StorageError,"Malformed"): store.load(job_id)
        store.close()

    def test_06_schema_and_disk_admission_fail_before_search(self):
        store=self.store(); store.db.execute("PRAGMA user_version=99"); store.close()
        session=Session(self.path)
        try:
            self.assertEqual(session.ready["event"],"error")
            self.assertEqual(session.ready["error"]["code"],"incompatible_saved_data")
            session.process.wait(timeout=10)
        finally: session.close()
        with tempfile.TemporaryDirectory(prefix="quota-",dir=TEMP_ROOT) as directory:
            path=Path(directory).resolve()
            self.assertTrue(path.is_relative_to(TEMP_ROOT.resolve()))
            (path/"quota-filler.bin").write_bytes(b"x"*MIB)
            session=Session(path,"--storage-quota-mb","1")
            try:
                self.assertEqual(session.ready["event"],"error")
                self.assertIn("quota",session.ready["error"]["message"])
                self.assertFalse((path/"analysis.sqlite3").exists())
            finally: session.close()
        with tempfile.TemporaryDirectory(prefix="disk-",dir=TEMP_ROOT) as directory:
            session=Session(directory,"--min-free-mb",str(2**40))
            try:
                self.assertEqual(session.ready["event"],"error")
                self.assertIn("free disk",session.ready["error"]["message"])
                self.assertFalse((Path(directory)/"analysis.sqlite3").exists())
            finally: session.close()

    def test_07_failed_report_commit_keeps_previous(self):
        store=self.store()
        try:
            job_id=store.create(self.root,self.full_limits)
            store.commit_report(job_id,self.report(1))
            with patch("congklak.storage.shutil.disk_usage",return_value=shutil_disk(0)):
                with self.assertRaises(StorageError): store.commit_report(job_id,self.report(2))
            self.assertEqual(store.load(job_id)["last_completed_depth"],1)
            # SQLite rollback after inserting but before moving progression.
            store.db.execute("CREATE TRIGGER prevent_progress BEFORE UPDATE OF latest_depth ON jobs BEGIN SELECT RAISE(ABORT,'simulated full disk write'); END")
            with self.assertRaises(StorageError): store.commit_report(job_id,self.report(2))
            self.assertEqual(store.load(job_id)["last_completed_depth"],1)
            self.assertEqual(store.db.execute("SELECT count(*) FROM reports").fetchone()[0],1)
        finally: store.close()

    def test_08_exclusive_coordinator_and_resume_identity(self):
        first=Session(self.path)
        second=Session(self.path)
        try:
            self.assertEqual(first.ready["event"],"ready")
            self.assertEqual(second.ready["event"],"error")
        finally:
            second.close(); first.close()

    def test_09_kill_inside_uncommitted_job_and_report_transactions(self):
        store=self.store()
        job_id=store.create(self.root,self.full_limits)
        store.commit_report(job_id,self.report(1))
        store.close()
        staged_job="f"*32
        script="""
import sys,json
from congklak import config
from congklak.native import Native
from congklak.storage import Store
path,mode,job_id,staged=sys.argv[1:]
native=Native()
store=Store(path,config.identity(native.build_id),native=native)
store.db.execute('BEGIN IMMEDIATE')
if mode=='job':
    payload={'identity':store.identity,'position':config.position(),'initial_limits':config.limits()}
    store.db.execute("INSERT INTO jobs VALUES(?,?,?,'pending',-1,NULL,0,0)",(staged,config.canonical(payload),config.digest(payload)))
else:
    report=native.analyze([7]*7+[0]+[7]*7+[0],max_depth=2)['report']
    store.db.execute('INSERT INTO reports VALUES(?,?,?,?,0)',(job_id,2,config.canonical(report),config.digest(report)))
    store.db.execute('UPDATE jobs SET latest_depth=2 WHERE id=?',(job_id,))
print('staged',flush=True)
sys.stdin.readline()
raise RuntimeError('Test parent should kill before commit')
"""
        for mode in ("job","report"):
            child=subprocess.Popen([sys.executable,"-c",script,str(self.path),mode,job_id,staged_job],cwd=ROOT,
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(),"staged")
                child.kill(); child.wait(timeout=10)
            finally:
                if child.poll() is None: child.kill(); child.wait(timeout=10)
                for stream in (child.stdin,child.stdout,child.stderr): stream.close()
            store=self.store()
            try:
                self.assertEqual(store.load(job_id)["last_completed_depth"],1)
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM jobs WHERE id=?",(staged_job,)).fetchone()[0],0)
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM reports WHERE depth=2").fetchone()[0],0)
            finally: store.close()

    def test_10_sqlite_full_preserves_durable_generation(self):
        # Force real SQLITE_FULL inside the transaction without filling a disk.
        store=self.store()
        try:
            job_id=store.create(self.root,self.full_limits)
            store.commit_report(job_id,self.report(1))
            pages=store.db.execute("PRAGMA page_count").fetchone()[0]
            self.assertEqual(store.db.execute("PRAGMA max_page_count="+str(pages)).fetchone()[0],pages)
            with self.assertRaises(StorageError) as failure:
                store.commit_report(job_id,self.report(2))
            self.assertEqual(failure.exception.__cause__.sqlite_errorcode,sqlite3.SQLITE_FULL)
            self.assertEqual(store.load(job_id)["last_completed_depth"],1)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM reports").fetchone()[0],1)
            store.close()
            store=self.store()
            self.assertEqual(store.load(job_id)["last_completed_depth"],1)
        finally: store.close()

def shutil_disk(free):
    import shutil
    return type(shutil.disk_usage(TEMP_ROOT))(1000,1000-free,free)

if __name__=="__main__": unittest.main()
