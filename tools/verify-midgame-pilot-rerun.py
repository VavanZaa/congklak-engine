"""Verify completed midgame producers and reviews without new search or training."""
import contextlib
import hashlib
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as prior, value_pilot_learning as learning
from congklak import value_pilot_benchmark as benchmark, quality_search
from congklak.native import Native
from congklak.neural_relay import NeuralNative

COMMANDS = (
    ('tools/midgame-data-pilot.py', 'all'),
    ('tools/midgame-label-pilot.py', 'all'),
    ('tools/analyze-midgame-label-pilot.py', 'verify'),
    ('tools/review-midgame-pilot-weights.py',))
WATCHED = (
    'data/midgame-data-pilot-v1/work.sqlite3', 'data/midgame-data-pilot-v1/manifest.json',
    'data/midgame-label-pilot-v1/work.sqlite3', 'data/midgame-label-pilot-v1/manifest.json',
    'build/midgame-data-pilot-result.json', 'build/midgame-label-pilot-result.json',
    'build/midgame-label-pilot-comparison.json', 'build/midgame-pilot-weight-review.json',
    'data/quality-train-v1/model.json', 'data/value-pilot-v1/model.json')


def signature(name):
    path = ROOT / name
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def main():
    before = {name:signature(name) for name in WATCHED}
    stream = io.StringIO()
    argv = sys.argv
    try:
        with (patch.object(Native, 'analyze', side_effect=AssertionError('Completed run started teacher search')) as native,
              patch.object(NeuralNative, 'analyze', side_effect=AssertionError('Completed run started neural search')) as neural,
              patch.object(learning, 'runtime', side_effect=AssertionError('Training runtime forbidden')) as runtime,
              patch.object(learning, 'train', side_effect=AssertionError('Training forbidden')) as train,
              patch.object(benchmark, 'decide', side_effect=AssertionError('New strength game forbidden')) as decide,
              patch.object(quality_search, 'search', side_effect=AssertionError('New PUCT search forbidden')) as puct):
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                for command in COMMANDS:
                    sys.argv = [str(ROOT / command[0]), *command[1:]]
                    try:
                        runpy.run_path(sys.argv[0], run_name='__main__')
                    except SystemExit as exc:
                        if exc.code not in (None, 0):
                            raise
            assert all(p.call_count == 0 for p in (native, neural, runtime, train, decide, puct))
    finally:
        sys.argv = argv
    assert 'torch' not in sys.modules
    after = {name:signature(name) for name in WATCHED}
    assert before == after, 'Completed producer/review rerun changed durable bytes or mtimes'
    baseline = prior.read(ROOT / 'build/midgame-pilot-baseline.json')
    assert len(baseline['inputs']) == 2639
    assert all(prior.sha(ROOT/name) == digest for name, digest in baseline['inputs'].items())
    log = ROOT / 'build/midgame-pilot-completed-rerun.log'
    if not log.exists():
        log.write_bytes(stream.getvalue().encode('utf-8'))
    sources = [command[0] for command in COMMANDS]
    sources += ['congklak/midgame_data_pilot.py', 'congklak/midgame_label_pilot.py',
                'tools/verify-midgame-pilot-rerun.py']
    proof = dict(schema='midgame-pilot-completed-rerun-v1', passed=True,
        before=before, after=after, completed_bytes_mtimes_unchanged=True,
        prior_inputs_preserved=2639, commands=[list(command) for command in COMMANDS],
        new_native_teacher_searches=0, new_neural_searches=0, new_PUCT_searches=0,
        new_training_runtime_activations=0, new_training_calls=0, new_strength_decisions=0,
        torch_imported=False, saved_history_validation_allowed=True,
        sources={name:prior.sha(ROOT/name) for name in sources}, log_sha256=prior.sha(log))
    prior.publish(ROOT/'build/midgame-pilot-root-verification.json', proof)
    print(config.canonical(dict(event='midgame_pilot_completed_rerun_verified', passed=True,
        stable_files=len(before), prior_inputs_preserved=2639,
        new_teacher_searches=0, training_calls=0, torch_imported=False)), flush=True)


if __name__ == '__main__':
    main()
