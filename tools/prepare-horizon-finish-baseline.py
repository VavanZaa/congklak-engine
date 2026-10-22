"""Preserve the sealed six-case probe before the authorized larger H14 run."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

OUTPUT = ROOT / 'build/horizon-finish-baseline.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    acceptance = ROOT / 'build/horizon-probe-acceptance.json'
    rerun = ROOT / 'build/horizon-probe-acceptance-rerun.json'
    prior = json.loads(acceptance.read_text(encoding='utf-8'))
    witness = json.loads(rerun.read_text(encoding='utf-8'))
    assert prior['passed'] is True and len(prior['inputs']) == 2312
    assert prior['model_file_sha256'] == MODEL_SHA
    current = dict(sha256=sha(acceptance), size=acceptance.stat().st_size,
                   mtime_ns=acceptance.stat().st_mtime_ns)
    assert witness['passed'] is True and witness['completed_bytes_mtimes_unchanged'] is True
    assert current == witness['before'] == witness['after']
    inputs = dict(prior['inputs'])
    inputs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (acceptance, rerun)})
    assert len(inputs) == 2314
    for name, checksum in inputs.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == checksum, 'Prior input changed: ' + name
    assert sha(ROOT / 'data/quality-train-v1/model.json') == MODEL_SHA
    value = dict(schema='horizon-finish-prior-immutability-v1', passed=True,
                 inputs_count=len(inputs), prior_accepted_inputs=2312,
                 prior_acceptance_sha256=sha(acceptance), fixed_model_sha256=MODEL_SHA,
                 inputs=dict(sorted(inputs.items())))
    text = config.canonical(value) + '\n'
    if OUTPUT.exists():
        assert OUTPUT.read_text(encoding='utf-8') == text, 'Immutable finish baseline differs'
    else:
        atomic_json(OUTPUT, value)
    assert all(sha(ROOT / name) == checksum for name, checksum in inputs.items())
    print(config.canonical(dict(event='horizon_finish_baseline', passed=True,
          prior_accepted_inputs=2312, preserved_inputs=len(inputs),
          baseline_sha256=sha(OUTPUT), fixed_model_sha256=MODEL_SHA)), flush=True)


if __name__ == '__main__':
    main()
