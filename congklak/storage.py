"""Inline, checksummed, transactionally published analysis checkpoints."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import uuid
from .config import InputError, canonical, digest, position, limits
from .native import Native
from .reports import validate as validate_report

MIB=1024*1024
MAX_REPORT_BYTES=8*MIB

class StorageError(RuntimeError):
    def __init__(self, message, code="storage_error"):
        super().__init__(message)
        self.code=code

class Store:
    def __init__(self,path,identity,quota_mb=51200,min_free_mb=256,native=None):
        self.path=Path(path).expanduser().resolve()
        self.identity=identity
        self.quota=quota_mb*MIB
        self.headroom=min_free_mb*MIB
        self.mutex=threading.RLock()
        self.db=None
        self.lockfile=None
        self.native=native or Native()
        try:
            self.path.mkdir(parents=True,exist_ok=True)
            self.lockfile=(self.path/"coordinator.lock").open("a+b")
            if self.lockfile.seek(0,os.SEEK_END)==0:
                self.lockfile.write(b"0"); self.lockfile.flush()
            self.lockfile.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.lockfile.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._admit(MIB)
            self.db=sqlite3.connect(self.path/"analysis.sqlite3",check_same_thread=False,timeout=5)
            self.db.row_factory=sqlite3.Row
            version=self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0,1):
                raise StorageError("Unsupported saved database schema", "incompatible_saved_data")
            if version==0 and self.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                raise StorageError("Unrecognized database schema", "incompatible_saved_data")
            if self.db.execute("PRAGMA quick_check").fetchone()[0]!="ok":
                raise StorageError("SQLite integrity check failed", "corrupt_saved_data")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
              status TEXT NOT NULL, latest_depth INTEGER NOT NULL DEFAULT -1,
              stop_reason TEXT, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS reports (
              job_id TEXT NOT NULL REFERENCES jobs(id), depth INTEGER NOT NULL,
              body TEXT NOT NULL, body_hash TEXT NOT NULL, committed REAL NOT NULL,
              PRIMARY KEY(job_id,depth));
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
              settings TEXT NOT NULL, settings_hash TEXT NOT NULL, started REAL NOT NULL,
              ended REAL, stop_reason TEXT);
            PRAGMA user_version=1;
            """)
            with self.db:
                self.db.execute("UPDATE jobs SET status='interrupted',stop_reason='process_interrupted' WHERE status='running'")
                self.db.execute("UPDATE runs SET ended=?,stop_reason='process_interrupted' WHERE ended IS NULL",(time.time(),))
        except Exception as exc:
            self.close()
            if isinstance(exc,StorageError): raise
            raise StorageError(str(exc)) from exc

    def _admit(self,projected):
        physical=sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
        if physical+projected>self.quota:
            raise StorageError("Projected database/WAL write exceeds physical data-directory quota")
        if shutil.disk_usage(self.path).free < self.headroom+projected:
            raise StorageError("Insufficient free disk for write plus recovery headroom")

    def create(self,root,settings):
        if root is None or not isinstance(settings,dict) or set(settings)!=set(limits()):
            raise InputError("invalid_saved_input","Saved position and all limits are required")
        root=position(root); settings=limits(settings)
        payload={"identity":self.identity,"position":root,"initial_limits":settings}
        job_id=uuid.uuid4().hex
        now=time.time()
        with self.mutex:
            self._admit(MIB)
            try:
                with self.db:
                    self.db.execute("INSERT INTO jobs(id,payload,payload_hash,status,created,updated) VALUES(?,?,?,'pending',?,?)",
                                    (job_id,canonical(payload),digest(payload),now,now))
            except sqlite3.Error as exc: raise StorageError(str(exc)) from exc
        return job_id

    def load(self,job_id,compatible=False):
        if not isinstance(job_id,str) or len(job_id)!=32 or any(c not in "0123456789abcdef" for c in job_id):
            raise InputError("invalid_job_id", "job_id must be the 32-character identifier returned by analyze")
        with self.mutex:
            size=self.db.execute("SELECT length(CAST(payload AS BLOB)) FROM jobs WHERE id=?",(job_id,)).fetchone()
            if size and size[0]>65536:
                raise StorageError("Oversized saved input","corrupt_saved_data")
            row=self.db.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
            if row is None: raise InputError("job_not_found", "Saved analysis job does not exist")
            try:
                payload=json.loads(row["payload"])
                if digest(payload)!=row["payload_hash"]: raise ValueError("payload checksum")
                if set(payload)!={"identity","position","initial_limits"}: raise ValueError("payload fields")
                if payload["position"] is None or not isinstance(payload["initial_limits"],dict) or set(payload["initial_limits"])!=set(limits()):
                    raise ValueError("missing saved position/limits")
                position(payload["position"]); limits(payload["initial_limits"])
                ident=payload["identity"]
                if not isinstance(ident,dict) or set(ident)!=set(self.identity): raise ValueError("identity fields")
                if row["latest_depth"] < -1 or row["latest_depth"]>128: raise ValueError("depth")
                if row["status"] not in {"pending","running","interrupted","stopped","complete","error"}: raise ValueError("status")
            except (ValueError,TypeError,KeyError,RecursionError) as exc:
                raise StorageError("Malformed saved input/settings: "+str(exc),"corrupt_saved_data") from exc
            is_compatible=ident==self.identity
            if compatible and not is_compatible:
                raise StorageError("Saved rules, evaluator, ABI/build or encoding differ; start a new job", "incompatible_saved_data")
            report=None
            invalid=[]
            report_rows=self.db.execute("SELECT depth,length(CAST(body AS BLOB)) AS bytes FROM reports WHERE job_id=? AND depth<=? ORDER BY depth DESC LIMIT 129",(job_id,row["latest_depth"])).fetchall()
            for meta in report_rows:
                try:
                    if meta["bytes"]>MAX_REPORT_BYTES: raise ValueError("oversized report")
                    saved=self.db.execute("SELECT * FROM reports WHERE job_id=? AND depth=?",(job_id,meta["depth"])).fetchone()
                    item=json.loads(saved["body"])
                    if digest(item)!=saved["body_hash"]: raise ValueError("report checksum")
                    if item["choice_depth"]!=saved["depth"]: raise ValueError("report depth")
                    validate_report(item,payload["position"],ident,self.native)
                    report=item
                    break
                except (ValueError,KeyError,TypeError,RecursionError):
                    invalid.append(meta["depth"])
            loaded=dict(row)
            loaded.pop("payload_hash")
            loaded["payload"]=payload
            loaded["report"]=report
            loaded["last_completed_depth"]=report["choice_depth"] if report else -1
            loaded["compatible"]=is_compatible
            loaded["recovery"]={"discarded_invalid_depths":invalid,"restored_previous_generation":bool(invalid and report)}
            runs=[]
            run_count=self.db.execute("SELECT COUNT(*),MAX(length(CAST(settings AS BLOB))) FROM runs WHERE job_id=?",(job_id,)).fetchone()
            if run_count[0]>2048 or (run_count[1] or 0)>2048:
                raise StorageError("Oversized saved run ledger","corrupt_saved_data")
            for saved in self.db.execute("SELECT * FROM runs WHERE job_id=? ORDER BY id",(job_id,)):
                run=dict(saved)
                try:
                    settings=json.loads(run["settings"])
                    if digest(settings)!=run.pop("settings_hash"): raise ValueError("run checksum")
                    if not isinstance(settings,dict) or set(settings)!=set(limits()): raise ValueError("missing run limits")
                    run["settings"]=limits(settings)
                except (ValueError,TypeError,KeyError,RecursionError) as exc:
                    raise StorageError("Malformed saved run settings","corrupt_saved_data") from exc
                runs.append(run)
            loaded["runs"]=runs
            return loaded

    def begin(self,job_id,settings):
        with self.mutex:
            self.load(job_id,compatible=True)
            self._admit(MIB)
            if self.db.execute("SELECT COUNT(*) FROM runs WHERE job_id=?",(job_id,)).fetchone()[0]>=2048:
                raise StorageError("Job run ledger limit reached; create a new job")
            now=time.time()
            try:
                with self.db:
                    self.db.execute("UPDATE jobs SET status='running',stop_reason=NULL,updated=? WHERE id=?",(now,job_id))
                    run=self.db.execute("INSERT INTO runs(job_id,settings,settings_hash,started) VALUES(?,?,?,?)",
                                        (job_id,canonical(settings),digest(settings),now))
                return run.lastrowid
            except sqlite3.Error as exc: raise StorageError(str(exc)) from exc

    def commit_report(self,job_id,report):
        body=canonical(report)
        size=len(body.encode("utf-8"))
        if size>MAX_REPORT_BYTES:
            raise StorageError("Report exceeds bounded 8 MiB inline buffer")
        with self.mutex:
            saved=self.load(job_id,compatible=True)
            try: validate_report(report,saved["payload"]["position"],self.identity,self.native)
            except (ValueError,TypeError,KeyError) as exc:
                raise StorageError("Malformed report cannot be committed: "+str(exc),"corrupt_saved_data") from exc
            self._admit(4*size+MIB)
            try:
                with self.db:
                    # A recovery may restart below damaged/orphaned generations.
                    # They must not displace this result's valid fallback.
                    self.db.execute("DELETE FROM reports WHERE job_id=? AND depth>?",(job_id,report["choice_depth"]))
                    self.db.execute("INSERT OR REPLACE INTO reports VALUES(?,?,?,?,?)",(job_id,report["choice_depth"],body,digest(report),time.time()))
                    self.db.execute("UPDATE jobs SET latest_depth=?,updated=? WHERE id=?",(report["choice_depth"],time.time(),job_id))
                    # Keep a fallback. Delete older generations only in the same
                    # transaction that makes the new complete result reachable.
                    self.db.execute("DELETE FROM reports WHERE job_id=? AND depth NOT IN (SELECT depth FROM reports WHERE job_id=? ORDER BY depth DESC LIMIT 2)",(job_id,job_id))
            except sqlite3.Error as exc: raise StorageError(str(exc)) from exc

    def finish(self,job_id,run_id,reason,complete=False,error=False):
        with self.mutex:
            try:
                with self.db:
                    self.db.execute("UPDATE jobs SET status=?,stop_reason=?,updated=? WHERE id=?",("error" if error else "complete" if complete else "stopped",reason,time.time(),job_id))
                    self.db.execute("UPDATE runs SET ended=?,stop_reason=? WHERE id=?",(time.time(),reason,run_id))
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc: raise StorageError(str(exc)) from exc

    def close(self):
        if self.db is not None:
            self.db.close(); self.db=None
        if self.lockfile is not None:
            # Closing releases the OS lock, including constructor-error paths.
            self.lockfile.close(); self.lockfile=None
