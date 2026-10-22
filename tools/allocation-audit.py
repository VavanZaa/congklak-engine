"""Isolated, source-bound independent workers for the allocation study."""
from __future__ import annotations
from contextlib import ExitStack
import argparse
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from congklak import config


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in this audit worker')
        return None


sys.meta_path.insert(0, NoTorch())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def name(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def durable(path):
    return (path.is_file() and path.suffix != '.pyc'
            and not path.name.endswith('-shm')
            and (not path.name.endswith('-wal') or path.stat().st_size > 0))


def files_at(path):
    path = Path(path)
    return {p.resolve() for p in path.rglob('*') if durable(p)} if path.exists() else set()


def snapshot(paths):
    return {name(p): dict(sha256=sha(p), size=p.stat().st_size,
                         mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}


def save(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Immutable worker proof already exists'
    path.write_text(config.canonical(body) + '\n', encoding='utf-8')


def forbidden(*args, **kwargs):
    raise AssertionError('Completed verification attempted new engine work')


def closed(spec):
    from congklak import native, quality_search
    from congklak.neural_relay import NeuralNative, Model
    from congklak import allocation_trace
    module = importlib.import_module(spec['module'])
    run = Path(spec['run']).resolve()
    roots = [run, *[Path(p).resolve() for p in spec.get('watch_directories', [])]]
    watched = set().union(*(files_at(p) for p in roots))
    watched.update((ROOT / p).resolve() for p in spec.get('extra_files', []))
    if spec.get('watch_baseline'):
        watched.update((ROOT / p).resolve() for p in read(ROOT / 'build/allocation-baseline.json')['inputs'])
    before = snapshot(watched)
    inventory_before = {name(p) for root in roots for p in files_at(root)}
    results = []
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(native.Native, 'analyze', forbidden))
        stack.enter_context(mock.patch.object(NeuralNative, 'analyze', forbidden))
        stack.enter_context(mock.patch.object(Model, 'reference', forbidden))
        for attr in ('search', 'forest', '_select', '_backup', '_expand'):
            stack.enter_context(mock.patch.object(quality_search, attr, forbidden))
        for attr in ('CpuBatchEvaluator', 'PerformanceCore', '_advance'):
            stack.enter_context(mock.patch.object(allocation_trace, attr, forbidden))
        for attr in spec.get('forbidden_attributes', []):
            owner = module
            parts = attr.split('.')
            for part in parts[:-1]:
                owner = getattr(owner, part)
            assert hasattr(owner, parts[-1]), 'Unknown declared engine boundary: ' + attr
            stack.enter_context(mock.patch.object(owner, parts[-1], forbidden))
        for call in spec['calls']:
            function = getattr(module, call['function'])
            result = function(*call.get('args', []), **call.get('kwargs', {}))
            results.append(dict(function=call['function'], returned=type(result).__name__))
        old_argv = sys.argv
        try:
            sys.argv = [str(ROOT / spec['cli']), *spec['argv']]
            try:
                runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as exc:
                assert exc.code in (None, 0), 'Completed CLI failed: ' + str(exc.code)
        finally:
            sys.argv = old_argv
    after = snapshot(watched)
    assert before == after, 'Completed API/CLI changed durable bytes, size or mtime'
    inventory_after = {name(p) for root in roots for p in files_at(root)}
    assert inventory_before == inventory_after, 'Completed API/CLI changed durable inventory'
    assert 'torch' not in sys.modules
    return dict(schema='allocation-independent-completed-no-work-v1', passed=True,
                before=before, after=after, durable_inventory=sorted(inventory_after),
                all_bytes_sizes_mtimes_unchanged=True, calls=results,
                cli_completed=True, new_teacher_searches=0, new_PUCT_searches=0,
                new_model_forwards=0, new_training_updates=0, new_games=0,
                torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec')
    args = parser.parse_args()
    path = Path(args.spec).resolve()
    spec = read(path)
    if spec['kind'] == 'closed':
        proof = closed(spec)
    elif spec['kind'] == 'literal_trace':
        from allocation_independent_audit import audit_trace_worker
        proof = audit_trace_worker(spec)
    else:
        raise ValueError('Unknown independent worker')
    inputs = {name(path): sha(path), name(__file__): sha(__file__)}
    for p in spec.get('input_names', []):
        inputs[p] = sha(ROOT / p)
    proof['inputs'] = inputs
    save(spec['output'], proof)
    print(config.canonical(dict(event='allocation_independent_worker',
                                kind=spec['kind'], passed=True)), flush=True)


if __name__ == '__main__':
    main()
