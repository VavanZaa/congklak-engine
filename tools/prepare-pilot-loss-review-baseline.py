"""Preserve completed pilot evidence before a read-only loss/readiness review."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = ROOT / 'build/value-pilot-acceptance.json'
    w = ROOT / 'build/value-pilot-acceptance-rerun.json'
    prior = json.loads(p.read_text(encoding='utf-8'))
    witness = json.loads(w.read_text(encoding='utf-8'))
    sig = dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    assert prior['passed'] is prior['policy_and_trunk_bit_exact'] is True
    assert prior['default_promotion'] is False and len(prior['inputs']) == 2598
    assert witness['passed'] is witness['completed_bytes_mtimes_unchanged'] is True
    assert witness['before'] == witness['after'] == sig
    inputs = dict(prior['inputs'])
    inputs.update({q.relative_to(ROOT).as_posix(): sha(q) for q in (p, w)})
    assert len(inputs) == 2600
    for name, digest in inputs.items():
        q = (ROOT / name).resolve()
        assert q.is_relative_to(ROOT) and sha(q) == digest, 'Prior evidence changed: ' + name
    control = sha(ROOT / 'data/quality-train-v1/model.json')
    candidate = sha(ROOT / 'data/value-pilot-v1/model.json')
    assert control == prior['source_model_sha256']
    assert candidate == prior['candidate_model_sha256']
    body = dict(schema='pilot-loss-review-prior-immutability-v1', passed=True,
        inputs_count=len(inputs), prior_accepted_inputs=2598, prior_acceptance_sha256=sha(p),
        fixed_model_sha256=control, fixed_candidate_model_sha256=candidate,
        inputs=dict(sorted(inputs.items())), new_training_authorized=False, new_teacher_labeling_authorized=False)
    out = ROOT / 'build/pilot-loss-review-baseline.json'
    if out.exists():
        assert out.read_text(encoding='utf-8') == config.canonical(body) + '\n'
    else:
        atomic_json(out, body)
    assert all(sha(ROOT / name) == digest for name, digest in inputs.items())
    print(config.canonical(dict(event='pilot_loss_review_baseline', passed=True, inputs=len(inputs),
        fixed_model_sha256=control, fixed_candidate_model_sha256=candidate, baseline_sha256=sha(out))), flush=True)


if __name__ == '__main__':
    main()
