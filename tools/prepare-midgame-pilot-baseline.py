"""Preserve sealed loss-review evidence before a bounded data/teacher pilot."""
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
    accepted = ROOT / 'build/pilot-loss-review-acceptance.json'
    rerun = ROOT / 'build/pilot-loss-review-acceptance-rerun.json'
    prior = json.loads(accepted.read_text(encoding='utf-8'))
    proof = json.loads(rerun.read_text(encoding='utf-8'))
    signature = dict(sha256=sha(accepted), size=accepted.stat().st_size,
                     mtime_ns=accepted.stat().st_mtime_ns)
    assert prior['passed'] is prior['policy_and_trunk_fixed'] is True
    assert prior['default_promotion'] is False and len(prior['inputs']) == 2637
    assert proof['passed'] is proof['acceptance_bytes_size_mtime_unchanged'] is True
    assert proof['before'] == proof['after'] == signature
    inputs = {**prior['inputs'], **{p.relative_to(ROOT).as_posix():sha(p) for p in (accepted, rerun)}}
    assert len(inputs) == 2639
    for name, digest in inputs.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == digest, 'Sealed evidence changed: ' + name
    control = sha(ROOT / 'data/quality-train-v1/model.json')
    pilot = sha(ROOT / 'data/value-pilot-v1/model.json')
    assert control == prior['control_model_sha256']
    assert pilot == prior['pilot_model_sha256']
    body = dict(schema='midgame-pilot-prior-immutability-v1', passed=True,
        inputs_count=len(inputs), prior_accepted_inputs=2637,
        prior_acceptance_sha256=sha(accepted), fixed_model_sha256=control,
        fixed_candidate_model_sha256=pilot, inputs=dict(sorted(inputs.items())),
        new_data_collection_authorized=True, new_teacher_labeling_authorized=True,
        new_training_authorized=False, new_strength_games_authorized=False,
        source_data_pruning_authorized=False)
    output = ROOT / 'build/midgame-pilot-baseline.json'
    if output.exists():
        assert output.read_text(encoding='utf-8') == config.canonical(body) + '\n'
    else:
        atomic_json(output, body)
    assert all(sha(ROOT / n) == d for n, d in inputs.items())
    print(config.canonical(dict(event='midgame_pilot_baseline', passed=True,
        inputs=len(inputs), fixed_model_sha256=control,
        fixed_candidate_model_sha256=pilot, baseline_sha256=sha(output))), flush=True)


if __name__ == '__main__':
    main()
