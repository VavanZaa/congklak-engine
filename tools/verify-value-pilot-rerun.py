"""Verify completed training and both timed campaigns without new optimization/decisions."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, value_pilot_data as data, value_pilot_learning as learning, value_pilot_benchmark as benchmark
from congklak.native import Native


def signature(path):
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns)


def paths():
    found = [ROOT / 'build/value-pilot-model-witness.json', data.MANIFEST, data.ROWS, data.REVIEW]
    found += [p for p in learning.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))]
    for stage in benchmark.STAGES:
        folder, output = benchmark.paths(stage)
        found += [p for p in folder.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))]
        found += [output, output.with_name(output.stem + '-resources.json')]
    return sorted(set(found))


def main():
    watched = paths()
    before = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    stream = io.StringIO()
    with (patch.object(learning, 'runtime', side_effect=AssertionError('No Torch/optimizer activation during completed rerun')) as optimizer,
         patch.object(benchmark, 'decide', side_effect=AssertionError('No new benchmark decisions during completed rerun')) as decisions,
         patch.object(Native, 'analyze', side_effect=AssertionError('No new native search during completed rerun')) as searches):
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            data.prepare()
            learning.train()
            learning.witness()
            for stage in benchmark.STAGES:
                benchmark.run(stage=stage)
                benchmark.closed_witness(stage)
        assert optimizer.call_count == decisions.call_count == searches.call_count == 0
    assert 'torch' not in sys.modules, 'Completed rerun imported Torch'
    assert paths() == watched, 'Completed rerun added durable files'
    after = {p.relative_to(ROOT).as_posix(): signature(p) for p in watched}
    assert before == after, 'Completed rerun changed durable bytes or mtimes'
    baseline = data.read(data.BASELINE)
    for name, digest in baseline['inputs'].items():
        assert data.sha(ROOT / name) == digest, 'Prior artifact changed: ' + name
    log = ROOT / 'build/value-pilot-rerun.log'
    if log.exists():
        assert log.read_bytes() == stream.getvalue().encode('utf-8')
    else:
        log.write_bytes(stream.getvalue().encode('utf-8'))
    proof = dict(schema='value-pilot-completed-no-work-rerun-v1', passed=True,
        prior_inputs_preserved=len(baseline['inputs']), before=before, after=after,
        completed_bytes_mtimes_unchanged=True, new_optimizer_activations=0, new_decisions=0,
        new_native_searches=0, torch_imported=False,
        sources={name:data.sha(ROOT/name) for name in ('tools/verify-value-pilot-rerun.py',
            'congklak/value_pilot_learning.py', 'congklak/value_pilot_benchmark.py', 'congklak/value_pilot_data.py')},
        model_sha256=data.sha(learning.DEFAULT/'model.json'))
    data.publish(ROOT / 'build/value-pilot-root-verification.json', proof)
    print(config.canonical(dict(event='value_pilot_completed_rerun_verified', passed=True,
        durable_files=len(watched), prior_inputs_preserved=len(baseline['inputs']),
        new_optimizer_activations=0, new_decisions=0, torch_imported=False)), flush=True)


if __name__ == '__main__':
    main()
