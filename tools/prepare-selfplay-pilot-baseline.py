"""Freeze all previously accepted evidence before the current-model self-play pilot."""
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


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    acceptance_name = 'build/midgame-pilot-acceptance.json'
    rerun_name = 'build/midgame-pilot-acceptance-rerun.json'
    accepted, rerun = read(acceptance_name), read(rerun_name)
    assert accepted['passed'] is True and rerun['passed'] is True
    signature = dict(sha256=sha(ROOT / acceptance_name),
                     size=(ROOT / acceptance_name).stat().st_size,
                     mtime_ns=(ROOT / acceptance_name).stat().st_mtime_ns)
    assert rerun['before'] == rerun['after'] == signature
    assert rerun['acceptance_bytes_size_mtime_unchanged'] is True
    inputs = dict(accepted['inputs'])

    def merge(mapping):
        for name, digest in mapping.items():
            path = (ROOT / name).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), name
            assert sha(path) == digest, 'Previously accepted input changed: ' + name
            assert name not in inputs or inputs[name] == digest, 'Conflicting pin: ' + name
            inputs[name] = digest

    reference = 'build/algorithm-reference-audit-20261015/'
    local, review = read(reference + 'local-checks.json'), read(reference + 'review.json')
    assert local['passed'] is True and review['passed'] is True
    assert local['tests_run'] == review['tests_run'] == 8
    assert review['training_performed'] is review['source_modified'] is False
    merge(local['inputs'])
    merge(review['inputs'])
    for name in (acceptance_name, rerun_name, reference + 'review.json'):
        merge({name: sha(ROOT / name)})
    merge(inputs.copy())
    model = 'data/quality-train-v1/model.json'
    assert sha(ROOT / model) == 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
    body = dict(schema='selfplay-pilot-prior-immutability-v1', passed=True,
                inputs_count=len(inputs), prior_midgame_inputs=len(accepted['inputs']),
                prior_reference_tests=review['tests_run'], fixed_model=model,
                fixed_model_sha256=sha(ROOT / model), inputs=dict(sorted(inputs.items())),
                new_selfplay_authorized=True, joint_policy_value_training_authorized=True,
                new_strength_games_authorized=True, source_pruning_authorized=False,
                default_promotion=False)
    path = ROOT / 'build/selfplay-pilot-baseline.json'
    if path.exists():
        assert path.read_text(encoding='utf-8') == config.canonical(body) + '\n'
    else:
        atomic_json(path, body)
    merge(inputs.copy())
    print(config.canonical(dict(event='selfplay_pilot_baseline', passed=True,
                               inputs=len(inputs), model_sha256=sha(ROOT / model),
                               baseline_sha256=sha(path))), flush=True)


if __name__ == '__main__':
    main()
