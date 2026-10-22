"""Versioned quota-sampling repair around the frozen arena watchdog.

Only vanished atomic *.pending files are ignored. Missing immutable committed
files, permission failures and every other filesystem error still stop sampling.
The original deadline, private-commit, Job Object and cumulative ledger stay in
force; the actual helper sources are bound in the worker permit.
"""
import os
import stat
from pathlib import Path
from . import neurodiag_common as common,value_run_resources as frozen
ROOT=common.ROOT
SOURCES=('congklak/value_run_watchdog_v2.py','tools/watch-value-arena-v2.py','tools/run-value-arena-logged-v2.py')
ROOTS=tuple(ROOT/p for p in ('data/value-training-v1','build/value-training-v1','data/value-run-v1','build/value-run-v1'))


def _size(path):
    try:
        info=path.stat()
        return info.st_size if stat.S_ISREG(info.st_mode) else 0
    except FileNotFoundError:
        if path.name.endswith('.pending'):return 0
        raise


def _walk_error(error):
    raise error


def namespace_bytes():
    total=0
    for root in ROOTS:
        if not root.exists():continue
        for directory,subdirs,files in os.walk(root,onerror=_walk_error):
            for name in files:total+=_size(Path(directory)/name)
    return total


def witness():
    return dict(schema='value-run-watchdog-enforcement-v2',change='ignore only vanished atomic .pending quota samples',
        sources=common.pins(SOURCES),frozen_resource_sources=common.pins(frozen.SOURCES),
        resource_protocol=frozen.PROTOCOL,engine_worker_unchanged=True)


def watchdog(lease):
    old_size=frozen.namespace_bytes;old_atomic=frozen._atomic
    active=frozen._lease(lease)
    bound=dict(witness(),lease_file=str(Path(lease).resolve()),lease_sha256=common.sha(lease),
               token=active['token'],actual_watchdog=frozen.process_identity())
    def certified_atomic(path,body,immutable=False):
        if Path(path).name=='permit.json':
            attestation=Path(path).with_name('enforcement-attestation-v2.json')
            old_atomic(attestation,bound,True)
            body=dict(body,enforcement_implementation=bound,
                      enforcement_attestation=dict(path=str(attestation),sha256=common.sha(attestation)))
        return old_atomic(path,body,immutable)
    frozen.namespace_bytes=namespace_bytes;frozen._atomic=certified_atomic
    try:return frozen.watchdog(lease)
    finally:frozen.namespace_bytes=old_size;frozen._atomic=old_atomic
