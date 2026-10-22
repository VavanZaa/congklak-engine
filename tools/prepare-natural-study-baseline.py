"""Freeze the preserved pilot acceptance and rerun witness for the natural study."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

OUTPUT = ROOT / 'build/natural-study-baseline.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    prior_path = 'build/data-pilot-acceptance.json'
    rerun_path = 'build/data-pilot-acceptance-rerun.json'
    prior, rerun = read(prior_path), read(rerun_path)
    assert prior['passed'] is True and len(prior['inputs']) == 2058
    assert prior['model_file_sha256'] == MODEL_SHA
    assert rerun['passed'] is True and rerun['completed_bytes_mtimes_unchanged'] is True
    assert rerun['before'] == rerun['after'] and rerun['after']['sha256'] == sha(ROOT / prior_path)
    assert rerun['after']['size'] == (ROOT / prior_path).stat().st_size
    assert rerun['after']['mtime_ns'] == (ROOT / prior_path).stat().st_mtime_ns
    inputs = dict(prior['inputs'])
    inputs.update({path: sha(ROOT / path) for path in (prior_path, rerun_path)})
    assert len(inputs) == 2060
    for name, digest in inputs.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, 'Prior input changed: ' + name
    assert sha(ROOT / 'data/quality-train-v1/model.json') == MODEL_SHA
    value = dict(schema='natural-study-prior-immutability-v1', passed=True,
        inputs_count=len(inputs), prior_accepted_inputs=2058, prior_acceptance_sha256=sha(ROOT / prior_path),
        fixed_model_sha256=MODEL_SHA, inputs=dict(sorted(inputs.items())))
    text = config.canonical(value) + '\n'
    if OUTPUT.exists():
        assert OUTPUT.read_text(encoding='utf-8') == text, 'Immutable natural-study baseline differs'
    else:
        atomic_json(OUTPUT, value)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, 'Input changed during baseline publication: ' + name
    print(config.canonical(dict(event='natural_study_baseline', passed=True, prior_accepted_inputs=2058,
        preserved_inputs=len(inputs), baseline_sha256=sha(OUTPUT), fixed_model_sha256=MODEL_SHA)), flush=True)


if __name__ == '__main__':
    main()
