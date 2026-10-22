"""Prove that a finished self-play/training/comparison cycle resumes without new work."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, quality_search, quality_learning, value_pilot_learning
from congklak import selfplay_training_pilot as learning, selfplay_pilot_benchmark as benchmark
from congklak.native import Native
_spec = importlib.util.spec_from_file_location('selfplay_pilot_weights_review',
                                              ROOT / 'tools/review-selfplay-pilot-weights.py')
weights = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(weights)


def signature(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)


def paths():
    result = [learning.BASELINE, learning.REGISTRY, learning.PLAN, learning.IMPLEMENTATION,
              learning.OUTPUT, learning.WITNESS, benchmark.PLAN, benchmark.PROJECTION,
              benchmark.IMPLEMENTATION, weights.OUTPUT]
    for folder in (learning.DEFAULT, ROOT / 'data/selfplay-pilot-benchmark-v1'):
        result += [p for p in folder.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))]
    output = ROOT / 'build/selfplay-pilot-benchmark.json'
    result += [output, output.with_name(output.stem + '-resources.json')]
    return sorted(set(result))


def main():
    watched = paths()
    assert all(p.is_file() for p in watched), 'Pilot is not complete'
    before = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    stream = io.StringIO()
    with (patch.object(learning, 'runtime', side_effect=AssertionError('No training runtime on closed resume')) as training,
          patch.object(quality_learning, 'runtime', side_effect=AssertionError('No indirect Torch activation')),
          patch.object(value_pilot_learning, 'runtime', side_effect=AssertionError('No inherited Torch activation')),
          patch.object(quality_search, 'forest', side_effect=AssertionError('No new self-play search')) as forests,
          patch.object(quality_search, 'search', side_effect=AssertionError('No new PUCT search')),
          patch.object(benchmark, 'decide', side_effect=AssertionError('No new comparison decisions')) as decisions,
          patch.object(Native, 'analyze', side_effect=AssertionError('No native search on closed resume')) as searches):
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            learning.generate()
            learning.load_replay()
            learning.train()
            learning.verify_run()
            learning.witness()
            weights.review()
            benchmark.run(stage='evaluation')
            benchmark.closed_witness('evaluation')
    assert training.call_count == forests.call_count == decisions.call_count == searches.call_count == 0
    assert 'torch' not in sys.modules, 'Closed resume imported Torch'
    assert paths() == watched, 'Closed resume created a durable file'
    after = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    assert before == after, 'Closed resume changed durable bytes or modification times'
    baseline = json.loads(learning.BASELINE.read_text(encoding='utf-8'))
    for name, digest in baseline['inputs'].items():
        assert learning.sha(ROOT / name) == digest, 'Prior accepted input changed: ' + name
    log = ROOT / 'build/selfplay-pilot-rerun.log'
    if log.exists():
        assert log.read_bytes() == stream.getvalue().encode('utf-8')
    else:
        log.write_bytes(stream.getvalue().encode('utf-8'))
    proof = dict(schema='selfplay-pilot-completed-no-work-rerun-v1', passed=True,
                 prior_inputs_preserved=len(baseline['inputs']), before=before, after=after,
                 completed_bytes_mtimes_unchanged=True, new_optimizer_activations=0,
                 new_selfplay_searches=0, new_comparison_decisions=0, new_native_searches=0,
                 torch_imported=False, candidate_model_sha256=learning.sha(learning.DEFAULT / 'model.json'),
                 sources={name: learning.sha(ROOT / name) for name in (
                     'tools/verify-selfplay-pilot-rerun.py', 'congklak/selfplay_training_pilot.py',
                     'congklak/selfplay_pilot_benchmark.py', 'tools/review-selfplay-pilot-weights.py')})
    learning.publish(ROOT / 'build/selfplay-pilot-root-verification.json', proof)
    print(config.canonical(dict(event='selfplay_pilot_completed_rerun_verified', passed=True,
                               durable_files=len(watched), prior_inputs_preserved=len(baseline['inputs']),
                               new_optimizer_activations=0, new_selfplay_searches=0,
                               new_comparison_decisions=0, torch_imported=False)), flush=True)


if __name__ == '__main__':
    main()
