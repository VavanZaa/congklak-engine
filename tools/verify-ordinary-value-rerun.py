"""Verify completed labels, value pilot, arena and followup without new work."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, ordinary_value_pilot as learning
from congklak import ordinary_value_benchmark as benchmark, ordinary_value_followup as followup, quality_search
from congklak.native import Native
from congklak.neural_relay import NeuralNative


def signature(path):
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(sha256=checksum, size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def paths():
    files = {learning.BASELINE, learning.DATA, learning.ROWS, learning.REGISTRY, learning.PLAN,
             learning.IMPLEMENTATION, learning.LABEL_OUTPUT, learning.WITNESS,
             benchmark.PLAN, benchmark.IMPLEMENTATION, benchmark.PROJECTION,
             ROOT / 'build/ordinary-value-benchmark.json', ROOT / 'build/ordinary-value-benchmark-resources.json',
             followup.PLAN, followup.IMPLEMENTATION, followup.OUTPUT, learning.SOURCE}
    for folder in (learning.DEFAULT, learning.LABEL_DEFAULT,
                   ROOT / 'data/ordinary-value-benchmark-v1', followup.DEFAULT):
        files.update(p for p in folder.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm')))
    return sorted(files)


def main():
    watched = paths(); assert all(p.is_file() for p in watched)
    before = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    events, capture = [], io.StringIO()
    def prohibited(*args, **kwargs):
        raise AssertionError('Completed replay attempted search, inference worker or training runtime')
    with patch.object(Native, 'analyze', prohibited), patch.object(NeuralNative, 'analyze', prohibited), \
         patch.object(quality_search, 'search', prohibited), patch.object(quality_search, 'forest', prohibited), \
         patch.object(benchmark, 'decide', prohibited), patch.object(benchmark, 'puct_search', prohibited), \
         patch.object(learning, 'runtime', prohibited), \
         contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        learning.run_labels(on_event=events.append)
        learning.load_labels()
        learning.train(on_event=events.append)
        learning.verify_run()
        learning.witness()
        benchmark.run(on_event=events.append)
        benchmark.closed_witness()
        followup.run(on_event=events.append)
        followup_value, _ = followup.load_run()
        if followup_value['status'] == 'held':
            assert followup_value == dict(schema=followup.SCHEMA, status='held', identity=followup_value['identity'],
                summary=dict(games=0, committed_decisions=0), training_performed=False,
                default_promotion=False, future_training_authorized=False)
    assert 'torch' not in sys.modules
    assert not any(row.get('event', '').endswith(('_precommit', '_committed', '_file_ready')) for row in events)
    assert watched == paths()
    after = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    assert before == after
    baseline = learning.read(learning.BASELINE)
    assert all(learning.sha(ROOT / name) == digest for name, digest in baseline['inputs'].items())
    log = ROOT / 'build/ordinary-value-rerun.log'
    if log.exists(): assert log.read_bytes() == capture.getvalue().encode()
    else: log.write_bytes(capture.getvalue().encode())
    sources = ('tools/verify-ordinary-value-rerun.py', 'congklak/ordinary_value_pilot.py',
               'congklak/ordinary_value_benchmark.py', 'congklak/ordinary_value_followup.py')
    proof = dict(schema='ordinary-value-completed-no-work-v1', passed=True, before=before, after=after,
                 all_durable_bytes_sizes_mtimes_unchanged=True, prior_inputs_preserved=len(baseline['inputs']),
                 new_teacher_searches=0, new_optimizer_steps=0, new_benchmark_decisions=0,
                 new_followup_decisions=0, torch_imported=False,
                 literal_CPU_validation_and_history_rechecks_allowed=True,
                 sources={name: learning.sha(ROOT / name) for name in sources})
    learning.publish(ROOT / 'build/ordinary-value-root-verification.json', proof)
    print(config.canonical(dict(event='ordinary_value_completed_rerun_verified', passed=True,
                               durable_files=len(watched), prior_inputs_preserved=len(baseline['inputs']),
                               new_searches=0, new_optimizer_steps=0, torch_imported=False)), flush=True)


if __name__ == '__main__': main()
