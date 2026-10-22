"""Verify immutable completed analysis and prior inputs; no new teacher search."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = ('build/natural-study-stability.json', 'data/natural-study-v1/work.sqlite3',
             'data/natural-study-v1/manifest.json', 'build/natural-study-result.json')
    def snapshot():
        return {name: dict(sha256=sha(ROOT / name), size=(ROOT / name).stat().st_size,
                          mtime_ns=(ROOT / name).stat().st_mtime_ns) for name in paths}
    before = snapshot()
    result = subprocess.run([sys.executable, str(ROOT / 'tools/analyze-natural-study.py')],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
    after = snapshot()
    assert before == after, 'Completed analysis changed an artifact'
    baseline = json.loads((ROOT / 'build/natural-study-baseline.json').read_text())
    for name, digest in baseline['inputs'].items():
        assert sha(ROOT / name) == digest, 'Prior input changed: ' + name
    log = ROOT / 'build/natural-study-stability-rerun.log'
    data = (result.stdout + result.stderr).encode('utf-8')
    if log.exists():
        assert log.read_bytes() == data
    else:
        log.write_bytes(data)
    proof = dict(schema='fixed-model-natural-study-root-verification-v1', passed=True,
        prior_inputs_preserved=len(baseline['inputs']), analysis_rerun_exit_code=result.returncode,
        completed_artifacts_bytes_mtimes_unchanged=True, before=before, after=after,
        training_performed=False,
        model_sha256=sha(ROOT / 'data/quality-train-v1/model.json'),
        analysis_source_sha256=sha(ROOT / 'tools/analyze-natural-study.py'),
        verifier_source_sha256=sha(Path(__file__)),
        baseline_sha256=sha(ROOT / 'build/natural-study-baseline.json'))
    output = ROOT / 'build/natural-study-root-verification.json'
    if output.exists():
        assert json.loads(output.read_text()) == proof, 'Completed verification differs'
    else:
        atomic_json(output, proof)
    print(config.canonical(dict(event='natural_study_rerun_verified', passed=True,
        prior_inputs_preserved=proof['prior_inputs_preserved'], stable_completed_artifacts=len(paths))), flush=True)


if __name__ == '__main__':
    main()
