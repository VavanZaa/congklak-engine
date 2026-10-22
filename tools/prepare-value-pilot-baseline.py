"""Preserve the completed H14 investigation before the authorized value pilot."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json

MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    acceptance = ROOT / 'build/horizon-finish-acceptance.json'
    witness_path = ROOT / 'build/horizon-finish-acceptance-rerun.json'
    prior = json.loads(acceptance.read_text(encoding='utf-8'))
    witness = json.loads(witness_path.read_text(encoding='utf-8'))
    assert prior['passed'] is prior['scope_finished'] is True and len(prior['inputs']) == 2424
    assert prior['model_file_sha256'] == MODEL_SHA
    signature = dict(sha256=sha(acceptance), size=acceptance.stat().st_size, mtime_ns=acceptance.stat().st_mtime_ns)
    assert witness['passed'] is True and witness['completed_bytes_mtimes_unchanged'] is True
    assert witness['before'] == witness['after'] == signature
    inputs = dict(prior['inputs'])
    inputs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in (acceptance, witness_path)})
    assert len(inputs) == 2426
    for name, digest in inputs.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, 'Prior artifact changed: ' + name
    assert sha(ROOT / 'data/quality-train-v1/model.json') == MODEL_SHA
    body = dict(schema='value-pilot-prior-immutability-v1', passed=True, inputs_count=len(inputs),
                prior_accepted_inputs=2424, prior_acceptance_sha256=sha(acceptance),
                fixed_model_sha256=MODEL_SHA, inputs=dict(sorted(inputs.items())))
    output = ROOT / 'build/value-pilot-baseline.json'
    text = config.canonical(body) + '\n'
    if output.exists():
        assert output.read_text(encoding='utf-8') == text, 'Immutable baseline differs'
    else:
        atomic_json(output, body)
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items())
    print(config.canonical(dict(event='value_pilot_baseline', passed=True, inputs=len(inputs),
                               baseline_sha256=sha(output))), flush=True)


if __name__ == '__main__':
    main()
