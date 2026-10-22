"""Repeat existing-data diagnostics while prohibiting search, games and training."""
import contextlib
import hashlib
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as prior, value_pilot_learning as learning, value_pilot_benchmark as benchmark, quality_search
from congklak.native import Native

SCRIPTS = ('tools/analyze-pilot-ordinary.py', 'tools/analyze-pilot-losses.py', 'tools/assess-engine-readiness.py')
OUTPUTS = ('build/pilot-loss-review-ordinary.json', 'build/pilot-loss-review-arena.json', 'build/engine-readiness-current.json')
PROTECTED = ('data/quality-train-v1/model.json', 'data/value-pilot-v1/model.json',
             'data/value-pilot-v1/catalog.sqlite3', 'build/value-pilot-rows.json',
             'data/value-pilot-benchmark-pretraining-v1/work.sqlite3',
             'data/value-pilot-benchmark-posttraining-v1/work.sqlite3')


def signature(name):
    p = ROOT / name
    return dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size,
                mtime_ns=p.stat().st_mtime_ns)


def main():
    before = {name:signature(name) for name in OUTPUTS + PROTECTED}
    stream = io.StringIO()
    original_argv = sys.argv
    try:
        with (patch.object(Native, 'analyze', side_effect=AssertionError('No new native search in existing-data review')) as searches,
              patch.object(learning, 'runtime', side_effect=AssertionError('No training runtime in existing-data review')) as runtime,
              patch.object(learning, 'train', side_effect=AssertionError('No training in existing-data review')) as training,
              patch.object(benchmark, 'decide', side_effect=AssertionError('No new benchmark decision in existing-data review')) as decisions,
              patch.object(quality_search, 'search', side_effect=AssertionError('No new PUCT search in existing-data review')) as puct):
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                for script in SCRIPTS:
                    sys.argv = [str(ROOT/script)]
                    try:
                        runpy.run_path(sys.argv[0], run_name='__main__')
                    except SystemExit as exc:
                        if exc.code not in (0, None):
                            raise
            assert searches.call_count == runtime.call_count == training.call_count == decisions.call_count == puct.call_count == 0
    finally:
        sys.argv = original_argv
    assert 'torch' not in sys.modules
    after = {name:signature(name) for name in OUTPUTS + PROTECTED}
    assert before == after, 'Read-only diagnostic rerun changed existing artifacts'
    baseline = prior.read(ROOT/'build/pilot-loss-review-baseline.json')
    assert len(baseline['inputs']) == 2600
    assert all(prior.sha(ROOT/name) == digest for name, digest in baseline['inputs'].items())
    log = ROOT/'build/pilot-loss-review-rerun.log'
    if log.exists():
        assert log.read_bytes() == stream.getvalue().encode('utf-8')
    else:
        log.write_bytes(stream.getvalue().encode('utf-8'))
    proof = dict(schema='pilot-loss-review-completed-rerun-v1', passed=True,
        prior_inputs_preserved=2600, before=before, after=after, completed_bytes_mtimes_unchanged=True,
        new_native_searches=0, new_PUCT_searches=0, new_training_runtime_activations=0,
        new_training_calls=0, new_benchmark_decisions=0, torch_imported=False,
        sources={name:prior.sha(ROOT/name) for name in (*SCRIPTS, 'tools/verify-pilot-loss-review-rerun.py')})
    prior.publish(ROOT/'build/pilot-loss-review-root-verification.json', proof)
    print(config.canonical(dict(event='pilot_loss_review_readonly_rerun_verified', passed=True,
        stable_files=len(before), prior_inputs_preserved=2600, new_searches=0,
        new_training_calls=0, torch_imported=False)), flush=True)


if __name__ == '__main__':
    main()
