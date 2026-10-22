"""Exclusive FULL/WAL journals whose completed reopen does not rewrite SQLite."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import sqlite3
from . import config
from .pilot import PilotError,MIB

class Work:
    def __init__(self,path,identity,quota_mb=51200,min_free_mb=2048):
        self.path=Path(path).resolve();self.identity=identity;self.quota=quota_mb*MIB;self.reserve=min_free_mb*MIB;self.db=None;self.lock=None
        self.path.mkdir(parents=True,exist_ok=True)
        try:
            self.lock=(self.path/'writer.lock').open('a+b')
            if self.lock.seek(0,2)==0:self.lock.write(b'0');self.lock.flush()
            self.lock.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.admit(8*MIB);self.db=sqlite3.connect(self.path/'work.sqlite3');self.db.row_factory=sqlite3.Row
            version=self.db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0,6):raise PilotError('Incompatible Phase6 journal schema','incompatible_saved_data')
            if self.db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise PilotError('Corrupt Phase6 SQLite','corrupt_saved_data')
            self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
            if version==0:
                with self.db:
                    self.db.execute('BEGIN IMMEDIATE')
                    self.db.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
                    self.db.execute('PRAGMA user_version=6')
                    self.put('identity',identity)
            previous=self.get('identity')
            if previous!=identity:raise PilotError('Phase6 identities/settings differ; use another directory','incompatible_saved_data')
        except BaseException:self.close();raise

    def admit(self,estimate):
        physical=sum(f.stat().st_size for f in self.path.parent.rglob('*') if f.is_file())
        if physical+estimate+self.reserve>self.quota:raise PilotError('Phase6 managed physical quota/recovery reserve exceeded','storage_error')
        if shutil.disk_usage(self.path).free<estimate+self.reserve:raise PilotError('Insufficient disk plus2GiB recovery headroom','storage_error')

    def get(self,key):
        row=self.db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone()
        if row is None:return None
        try:
            value=json.loads(row[0])
            if config.digest(value)!=row[1]:raise ValueError('checksum')
            return value
        except (ValueError,TypeError,RecursionError) as exc:raise PilotError('Corrupt Phase6 metadata '+key,'corrupt_saved_data') from exc

    def put(self,key,value):self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?,?)',(key,config.canonical(value),config.digest(value)))

    def close(self):
        if self.db:self.db.close();self.db=None
        if self.lock:self.lock.close();self.lock=None
