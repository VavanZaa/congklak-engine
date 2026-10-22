"""Versioned completed reader: certify without acquiring the pinned writer lock.

Observation generation remains the frozen v1 producer. Certified or completed
data always take the read-only branch, including refusal of missing/corrupt
certificates. Certification is published only after the collector has closed.
"""
from contextlib import closing
from pathlib import Path
import sqlite3
from . import neurodiag as producer

ROOT=producer.ROOT
DEFAULT=producer.DEFAULT
OUTPUT=producer.OUTPUT
FREEZE=ROOT/'build/neurodiag-completion-implementation.json'
SOURCES=('congklak/neurodiag_v2.py','tools/diagnose-neural-v2.py')


def implementation():
    return dict(schema='neurodiag-completed-reader-implementation-v1',
        sources=producer.common.pins(SOURCES),producer_source_freeze_sha256=producer.sha(producer.SOURCE_FREEZE),
        contract='Completed/certificate-present caches read and certify before any Work/writer-lock construction; unfinished uncertified collection delegates frozen v1.')


def freeze():return producer.publish(FREEZE,implementation())


def admit():
    producer.require(producer.same(producer.read(FREEZE),implementation()),'Completed-reader implementation differs')


def completed(path):
    database=Path(path)/'work.sqlite3'
    if not database.exists():return False
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone():return False
        return db.execute("SELECT 1 FROM meta WHERE key='complete'").fetchone() is not None


def run(path=DEFAULT,fixture=False,on_event=None,cancel=None):
    admit();producer.require(type(fixture) is bool,'Fixture flag must be Boolean')
    path=producer.path_check(path,fixture)
    # Either certificate file is sufficient to prohibit collection fallback.
    # A partially published certificate or damaged completed cache must refuse.
    if producer.WITNESS.exists() or producer.PRODUCTION_AUDIT.exists() or completed(path):
        state=producer.raw_complete(path);producer.certify(path,state)
        if on_event:on_event(dict(event='neurodiag_already_complete',completed=len(state['batch_rows']),units=len(state['observations'])))
        return state['manifest']
    return producer.run(path,fixture,on_event,cancel)


def raw_complete(path=DEFAULT,require_exports=True):
    admit();return producer.raw_complete(path,require_exports)


def build():
    admit();return producer.build()


def assess(output=OUTPUT):
    admit();return producer.assess(output)


def verify(output=OUTPUT):
    admit();return producer.verify(output)
