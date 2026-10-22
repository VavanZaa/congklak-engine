"""Isolated literal-only completed-path checks for the regret review."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
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


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in this existing-data review')
        return None


sys.meta_path.insert(0, NoTorch())


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def name(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def durable(path):
    return path.is_file() and path.suffix != '.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size > 0)
def files_at(path):
    path = Path(path)
    return {p.resolve() for p in path.rglob('*') if durable(p)} if path.exists() else set()
def snapshot(paths):
    return {name(p): dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}
def forbidden(*args, **kwargs): raise AssertionError('Regret review attempted Native, inference, search or training work')


@contextmanager
def engine_guards():
    from congklak import native, quality_search, puct_cpu_inference, ordinary_value_pilot, value_learning
    from congklak.neural_relay import NeuralNative, Model
    with ExitStack() as stack:
        # Prohibit the runtime itself, including apparently harmless Native
        # normalization or application. Only the independent literal replay runs.
        for cls in (native.Native, native.Control, NeuralNative, puct_cpu_inference.CpuBatchEvaluator):
            stack.enter_context(mock.patch.object(cls, '__init__', forbidden))
        for cls, attrs in ((native.Native, ('analyze', 'normalize', 'apply', 'validate')),
                           (NeuralNative, ('analyze', 'evaluate', 'measure_inference', 'neural_analyze')),
                           (Model, ('reference',)),
                           (puct_cpu_inference.CpuBatchEvaluator, ('evaluate_many', 'feature_vector'))):
            for attr in attrs:
                if hasattr(cls, attr): stack.enter_context(mock.patch.object(cls, attr, forbidden))
        for attr in ('search', 'forest', '_select', '_backup', '_expand'):
            stack.enter_context(mock.patch.object(quality_search, attr, forbidden))
        stack.enter_context(mock.patch.object(ordinary_value_pilot, 'train', forbidden))
        stack.enter_context(mock.patch.object(value_learning, 'runtime', forbidden))
        for owner in (ordinary_value_pilot.cache, ordinary_value_pilot.joint):
            stack.enter_context(mock.patch.object(owner, 'train', forbidden))
        yield


def closed(spec):
    fixture = spec.get('mode') == 'fixture_cli'
    assert spec.get('mode', 'closed') in ('closed', 'fixture_cli')
    directories = [Path(p).resolve() for p in spec.get('watch_directories', [])]
    assert not fixture or not directories, 'Fixture writes must not be described as unchanged completed directories'
    watched = set().union(*(files_at(p) for p in directories)) if directories else set()
    watched.update((ROOT / p).resolve() for p in spec.get('extra_files', []))
    if spec.get('watch_baseline'):
        watched.update((ROOT / p).resolve() for p in read(ROOT / 'build/regret-baseline.json')['inputs'])
    before = snapshot(watched)
    inventory = {name(p) for root in directories for p in files_at(root)}
    results = []
    with engine_guards():
        for call in spec.get('calls', []):
            module = importlib.import_module(call['module'])
            result = getattr(module, call['function'])(*call.get('args', []), **call.get('kwargs', {}))
            results.append(dict(module=call['module'], function=call['function'], returned=type(result).__name__))
        for call in spec.get('clis', []):
            previous = sys.argv
            try:
                sys.argv = [str(ROOT / call['path']), *call.get('argv', [])]
                try: runpy.run_path(sys.argv[0], run_name='__main__')
                except SystemExit as exc: assert exc.code in (None, 0), str(exc.code)
            finally: sys.argv = previous
    assert before == snapshot(watched), 'Durable bytes, size or mtime changed'
    assert inventory == {name(p) for root in directories for p in files_at(root)}, 'Durable inventory changed'
    assert 'torch' not in sys.modules
    return dict(schema='regret-independent-fixture-runtime-guard-v1' if fixture else 'regret-independent-completed-no-work-v1',
                passed=True, fixture_case_commits_allowed=fixture, before=before,
                after=snapshot(watched), durable_inventory=sorted(inventory),
                all_bytes_sizes_mtimes_unchanged=True, calls=results, clis_completed=len(spec.get('clis', [])),
                new_case_commits=None if fixture else 0, native_calls=0, new_model_forwards=0, new_teacher_searches=0,
                new_PUCT_searches=0, new_training_updates=0, new_games=0, torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('spec'); args = parser.parse_args()
    path = Path(args.spec).resolve(); spec = read(path); proof = closed(spec)
    proof['inputs'] = {name(path): sha(path), name(__file__): sha(__file__),
                       **{p: sha(ROOT / p) for p in spec.get('input_names', [])}}
    output = Path(spec['output']); assert not output.exists(), 'Immutable worker proof exists'
    output.write_text(json.dumps(proof, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(event='regret_independent_worker', passed=True)), flush=True)


if __name__ == '__main__': main()
