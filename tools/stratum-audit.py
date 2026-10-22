"""Isolated source-bound completed-path checks for the stratum study."""
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from congklak import config


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in this data-only audit')
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
    return (path.is_file() and path.suffix != '.pyc' and not path.name.endswith('-shm')
            and (not path.name.endswith('-wal') or path.stat().st_size > 0))


def files_at(path):
    path = Path(path)
    return {p.resolve() for p in path.rglob('*') if durable(p)} if path.exists() else set()


def snapshot(paths):
    return {name(p): dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(paths)}


def forbidden(*args, **kwargs):
    raise AssertionError('Completed validation attempted new engine work')


def closed(spec):
    from congklak import native, quality_search, puct_cpu_inference
    from congklak.neural_relay import NeuralNative, Model
    module = importlib.import_module(spec['module'])
    directories = [Path(p).resolve() for p in spec['watch_directories']]
    watched = set().union(*(files_at(p) for p in directories))
    watched.update((ROOT / p).resolve() for p in spec.get('extra_files', []))
    if spec.get('watch_baseline'):
        watched.update((ROOT / p).resolve() for p in read(ROOT / 'build/stratum-baseline.json')['inputs'])
    before = snapshot(watched)
    inventory = {name(p) for root in directories for p in files_at(root)}
    results = []
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(native.Native, 'analyze', forbidden))
        stack.enter_context(mock.patch.object(NeuralNative, 'analyze', forbidden))
        stack.enter_context(mock.patch.object(Model, 'reference', forbidden))
        for attr in ('evaluate', 'measure_inference', 'neural_analyze'):
            stack.enter_context(mock.patch.object(NeuralNative, attr, forbidden))
        stack.enter_context(mock.patch.object(puct_cpu_inference.CpuBatchEvaluator, 'evaluate_many', forbidden))
        for attr in ('search', 'forest', '_select', '_backup', '_expand'):
            stack.enter_context(mock.patch.object(quality_search, attr, forbidden))
        for attr in spec.get('forbidden_attributes', []):
            owner = module
            parts = attr.split('.')
            for part in parts[:-1]: owner = getattr(owner, part)
            assert hasattr(owner, parts[-1]), 'Unknown engine boundary: ' + attr
            stack.enter_context(mock.patch.object(owner, parts[-1], forbidden))
        for call in spec['calls']:
            result = getattr(module, call['function'])(*call.get('args', []), **call.get('kwargs', {}))
            results.append(dict(function=call['function'], returned=type(result).__name__))
        previous = sys.argv
        try:
            sys.argv = [str(ROOT / spec['cli']), *spec['argv']]
            try: runpy.run_path(sys.argv[0], run_name='__main__')
            except SystemExit as exc: assert exc.code in (None, 0), str(exc.code)
        finally: sys.argv = previous
    assert before == snapshot(watched), 'Durable bytes, size or mtime changed'
    assert inventory == {name(p) for root in directories for p in files_at(root)}, 'Durable inventory changed'
    assert 'torch' not in sys.modules
    return dict(schema='stratum-independent-completed-no-work-v1', passed=True, before=before,
                after=snapshot(watched), durable_inventory=sorted(inventory),
                all_bytes_sizes_mtimes_unchanged=True, calls=results, cli_completed=True,
                new_teacher_searches=0, new_PUCT_searches=0, new_model_forwards=0,
                new_training_updates=0, new_games=0, torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('spec'); args = parser.parse_args()
    path = Path(args.spec).resolve(); spec = read(path); proof = closed(spec)
    proof['inputs'] = {name(path): sha(path), name(__file__): sha(__file__),
                       **{p: sha(ROOT / p) for p in spec.get('input_names', [])}}
    output = Path(spec['output']); assert not output.exists(), 'Immutable worker proof exists'
    output.write_text(config.canonical(proof) + '\n', encoding='utf-8')
    print(config.canonical(dict(event='stratum_independent_worker', passed=True)), flush=True)


if __name__ == '__main__': main()
