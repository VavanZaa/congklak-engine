"""Verify completed forensic artifacts without permitting teacher re-execution."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.native import Native
from congklak.deeper_teacher_v2 import atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = ('build/horizon-probe-forensics.json', 'data/horizon-probe-v1/work.sqlite3',
             'data/horizon-probe-v1/manifest.json', 'build/horizon-probe-result.json')
    def snapshot():
        return {name: dict(sha256=sha(ROOT / name), size=(ROOT / name).stat().st_size,
                          mtime_ns=(ROOT / name).stat().st_mtime_ns) for name in paths}
    before = snapshot(); output = io.StringIO(); original_argv = sys.argv
    try:
        sys.argv = [str(ROOT / 'tools/analyze-horizon-probe.py')]
        with patch.object(Native, 'analyze', side_effect=AssertionError('No teacher search allowed during forensic rerun')) as teacher_calls:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                runpy.run_path(sys.argv[0], run_name='__main__')
            assert teacher_calls.call_count == 0
    finally:
        sys.argv = original_argv
    after = snapshot(); assert before == after, 'Completed forensic rerun changed artifacts'
    baseline = json.loads((ROOT / 'build/horizon-probe-baseline.json').read_text())
    for name, digest in baseline['inputs'].items():
        assert sha(ROOT / name) == digest, 'Prior artifact changed: ' + name
    log = ROOT / 'build/horizon-probe-forensics-rerun.log'; data = output.getvalue().encode('utf-8')
    if log.exists():
        assert log.read_bytes() == data
    else:
        log.write_bytes(data)
    proof = dict(schema='fixed-model-horizon-probe-root-verification-v1', passed=True,
        prior_inputs_preserved=len(baseline['inputs']), completed_artifacts_bytes_mtimes_unchanged=True,
        before=before, after=after, native_teacher_calls=0, training_performed=False,
        model_sha256=sha(ROOT / 'data/quality-train-v1/model.json'),
        analysis_source_sha256=sha(ROOT / 'tools/analyze-horizon-probe.py'),
        verifier_source_sha256=sha(Path(__file__)), baseline_sha256=sha(ROOT / 'build/horizon-probe-baseline.json'))
    path = ROOT / 'build/horizon-probe-root-verification.json'
    if path.exists():
        assert json.loads(path.read_text()) == proof
    else:
        atomic_json(path, proof)
    print(config.canonical(dict(event='horizon_probe_rerun_verified', passed=True,
        prior_inputs_preserved=len(baseline['inputs']), stable_completed_artifacts=len(paths), native_teacher_calls=0)), flush=True)


if __name__ == '__main__':
    main()
