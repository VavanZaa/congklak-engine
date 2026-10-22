"""Constructor-issued completed-job proofs scoped to one contained worker.

Large immutable sources and raw journals are byte verified on admission and
again before the scope closes. Each reuse freshly hashes compact closing and
source bindings. No timestamp, file-size, inference or historical module cache
can admit a result. Returned results are independent canonical copies.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary
import hashlib
import json
from . import config, guarded_value_pilot_v2_common as stage

_ACTIVE = ContextVar('guarded_value_pilot_v2_completed_jobs', default=None)
_LAST = ContextVar('guarded_value_pilot_v2_completed_jobs_receipt', default=None)
_ISSUER = object()
_SEALS = WeakKeyDictionary()
need, read, sha = stage.need, stage.read, stage.sha


def _path(path):
    path = Path(path).resolve()
    need(path.is_relative_to(stage.FOLDER), 'Job proof path outside new build namespace')
    return path


def _inventory(path):
    return tuple(sorted(p.relative_to(path).as_posix() for p in path.rglob('*')
                        if p.is_file() and p.name != 'writer.lock' and not p.name.endswith('.pending')))


def _dependencies(path):
    identity = read(path / 'identity.json')
    paths = {path / name for name in _inventory(path)}
    for field in ('source_inputs', 'native_inputs'):
        for name in identity[field]:
            filename = (stage.ROOT / name).resolve()
            need(filename.is_relative_to(stage.ROOT), 'Job proof dependency outside workspace')
            paths.add(filename)
    paths.update((stage.PLAN, stage.PREFLIGHT, stage.PRIOR, Path(__file__).resolve()))
    paths.add(stage.ROOT / identity['binding']['path'])
    paths.add(stage.ROOT / identity['cached_fit_manifest']['path'])
    registry_path = stage.ROOT / identity['registry']['path']
    registry = read(registry_path)
    shared_path = stage.ROOT / registry['graph']['path']
    shared = read(shared_path)
    paths.update((registry_path, shared_path, stage.ROOT / shared['records_ref']['path'],
                  stage.ROOT / shared['source_mapping']['path']))
    for name in shared['source_inputs']:
        paths.add(stage.ROOT / name)
    for refs in identity['guard_refs_by_origin'].values():
        paths.add(stage.ROOT / refs['manifest_path'])
    for ref in (identity.get('seed') or {}).values():
        if isinstance(ref, dict) and set(ref) == {'path', 'sha256'}:
            paths.add(stage.ROOT / ref['path'])
    return {Path(item).resolve() for item in paths}


def _compact(path):
    identity = read(path / 'identity.json')
    paths = {path / name for name in ('identity.json', 'catalog.json', 'report.json',
                                     'ownership-prefix.json', 'claims.bin')}
    for field in ('source_inputs', 'native_inputs'):
        paths.update(stage.ROOT / name for name in identity[field])
    paths.update((stage.PLAN, stage.PREFLIGHT, stage.PRIOR, Path(__file__).resolve(),
                  stage.ROOT / identity['binding']['path'], stage.ROOT / identity['registry']['path'],
                  stage.ROOT / identity['cached_fit_manifest']['path']))
    registry = read(stage.ROOT / identity['registry']['path'])
    paths.add(stage.ROOT / registry['graph']['path'])
    paths.update(stage.ROOT / refs['manifest_path'] for refs in identity['guard_refs_by_origin'].values())
    for ref in (identity.get('seed') or {}).values():
        if isinstance(ref, dict) and set(ref) == {'path', 'sha256'}:
            paths.add(stage.ROOT / ref['path'])
    return {Path(item).resolve() for item in paths}


def _pins(paths):
    return {str(path): sha(path) for path in sorted(paths)}


def _current(pins):
    for name, checksum in pins.items():
        need(Path(name).is_file() and sha(name) == checksum, 'Scoped completed evidence changed: ' + name)


@dataclass(frozen=True, eq=False)
class _VerifiedJob:
    result: bytes
    compact: object
    full: object
    inventory: tuple
    path: Path
    operation: str
    issuer: object


class _Scope:
    def __init__(self):
        self.tokens = {}
        self.inputs = {}
        self.closed = False
        self.hits = 0
        self.cold = 0
        self.source_sha256 = sha(__file__)

    def assert_live(self):
        need(type(self) is _Scope and _ACTIVE.get() is self and not self.closed and
             sha(__file__) == self.source_sha256, 'Completed-job admission scope is not live/source bound')

    def get(self, path, operation):
        self.assert_live()
        token = self.tokens.get((path, operation))
        if token is None:
            return None
        seal = _SEALS.get(token)
        need(type(token) is _VerifiedJob and token.issuer is _ISSUER and seal is not None and
             token.path == path and token.operation == operation and
             token.result is seal[0] and token.compact is seal[1] and token.full is seal[2] and
             token.inventory is seal[3] and seal[4] is self,
             'Completed-job proof was not issued by this constructor')
        need(_inventory(path) == token.inventory, 'Completed-job physical inventory changed')
        _current(token.compact)
        self.hits += 1
        return json.loads(token.result)

    def admit(self, path, operation, producer):
        self.assert_live()
        inventory = _inventory(path)
        dependencies = _dependencies(path)
        # Previously admitted large immutable dependencies share this finite
        # source view; every new byte source is freshly read before validation.
        before = {str(item): self.inputs[str(item)] if str(item) in self.inputs else sha(item)
                  for item in sorted(dependencies)}
        result = producer()
        if result['publication_gap']:
            return result  # Closing metadata may still be published; never cache it.
        need(_inventory(path) == inventory, 'Completed-job files changed during cold admission')
        _current({name: checksum for name, checksum in before.items() if name not in self.inputs})
        compact = _pins(_compact(path))
        need(all(before.get(name) == checksum for name, checksum in compact.items()),
             'Compact completed bindings do not match the cold full-byte snapshot')
        for name, checksum in before.items():
            need(name not in self.inputs or self.inputs[name] == checksum, 'Completed scopes disagree on source bytes')
            self.inputs[name] = checksum
        token = _VerifiedJob(config.canonical(result).encode(), MappingProxyType(compact),
            MappingProxyType(before), inventory, path, operation, _ISSUER)
        _SEALS[token] = (token.result, token.compact, token.full, token.inventory, self)
        self.tokens[(path, operation)] = token
        self.cold += 1
        return json.loads(token.result)

    def close(self):
        self.assert_live()
        for token in self.tokens.values():
            need(_inventory(token.path) == token.inventory, 'Completed-job inventory changed before scope close')
        _current(self.inputs)
        self.closed = True
        _LAST.set(dict(schema=stage.SCHEMA + '-completed-job-view', passed=True,
            cold_jobs=self.cold, canonical_copy_reuses=self.hits, files=len(self.inputs),
            expected_inputs_sha256=config.digest(self.inputs), source_sha256=self.source_sha256,
            full_bytes_verified_before_and_after=True, mtime_cache=False,
            native_queries=0, optimizer_steps=0, model_forwards=0))


@contextmanager
def admission_scope():
    existing = _ACTIVE.get()
    if existing is not None:
        existing.assert_live()
        yield existing
        return
    scope = _Scope()
    token = _ACTIVE.set(scope)
    try:
        yield scope
        scope.close()
    finally:
        _ACTIVE.reset(token)


def completed(path, operation, producer):
    scope = _ACTIVE.get()
    if scope is None:
        return producer()
    path = _path(path)
    cached = scope.get(path, operation)
    return cached if cached is not None else scope.admit(path, operation, producer)


def admission_receipt():
    value = _LAST.get()
    return None if value is None else json.loads(config.canonical(value))
