"""Verify combined completed review replays without changing prior evidence."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_overlap_review as overlap, ordinary_value_loss_review as arena
from congklak import ordinary_value_validation_review as validation, ordinary_value_review_conclusions as conclusions
from congklak import quality_search, ordinary_value_pilot as producer, ordinary_value_benchmark as benchmark
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from congklak.ordinary_value_review_common import baseline, publish, pins


def signature(path):
    with path.open('rb') as stream: checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=checksum, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    frozen = baseline()
    modules = (overlap, arena, validation, conclusions)
    reports = [module.OUTPUT.relative_to(ROOT).as_posix() for module in modules]
    names = set(frozen['inputs']) | set(reports) | {'build/ordinary-value-review-baseline.json'}
    for module in modules:
        names.update(json.loads(module.OUTPUT.read_text())['inputs'])
    def record(): return {name: signature(ROOT / name) for name in sorted(names)}
    before = record()
    assert all(before[name]['sha256'] == digest for name, digest in frozen['inputs'].items())
    def forbidden(*args, **kwargs): raise AssertionError('Completed review attempted new experimental work')
    with ExitStack() as stack:
        for target, name in ((Native, 'analyze'), (NeuralNative, 'analyze'), (quality_search, 'search'),
                             (quality_search, 'forest_search'), (producer, 'train'), (producer, 'runtime'),
                             (producer, 'run_labels'), (benchmark, 'run'), (benchmark, 'decide')):
            if hasattr(target, name): stack.enter_context(patch.object(target, name, forbidden))
        for module in modules:
            expected = json.loads(module.OUTPUT.read_text())
            assert module.run() == module.verify() == expected
    after = record()
    assert before == after and 'torch' not in sys.modules
    source_names = ['tools/verify-ordinary-value-review-rerun.py', 'congklak/ordinary_value_review_common.py']
    source_names += [Path(module.__file__).relative_to(ROOT).as_posix() for module in modules]
    body = dict(schema='ordinary-value-review-completed-rerun-v1', passed=True, before=before, after=after,
                all_bytes_sizes_mtimes_unchanged=True, old_inputs_preserved=len(frozen['inputs']),
                reports_rebuilt_and_verified=len(modules), sources=pins(source_names),
                new_training_updates=0, new_teacher_searches=0, new_PUCT_searches=0, new_games=0,
                torch_imported=False, literal_replay_and_feature_arithmetic_permitted=True)
    publish(ROOT / 'build/ordinary-value-review-rerun.json', body)
    print(config.canonical(dict(event='ordinary_value_review_completed_rerun_verified', passed=True,
                               old_inputs=len(frozen['inputs']), reports=len(modules))), flush=True)


if __name__ == '__main__': main()
