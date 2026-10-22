"""Preserve the sealed self-play experiment before an ordinary-coverage data study."""
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
    accepted_name = 'build/selfplay-pilot-acceptance.json'
    rerun_name = 'build/selfplay-pilot-acceptance-rerun.json'
    dashboard_name = 'build/selfplay-pilot-dashboard-final.json'
    accepted, rerun, dashboard = (read(n) for n in (accepted_name, rerun_name, dashboard_name))
    assert accepted['passed'] is rerun['passed'] is dashboard['passed'] is True
    assert accepted['tests_run'] == 20 and accepted['actual_process_kills'] == 8
    assert accepted['default_promotion'] is False
    path = ROOT / accepted_name
    signature = dict(sha256=sha(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    assert rerun['before'] == rerun['after'] == signature
    assert rerun['acceptance_bytes_size_mtime_unchanged'] is True
    inputs = dict(accepted['inputs'])
    assert len(inputs) == 3168
    for name, digest in dashboard['inputs'].items():
        assert name not in inputs or inputs[name] == digest
        inputs[name] = digest
    for name in (accepted_name, rerun_name, dashboard_name):
        assert name not in inputs or inputs[name] == sha(ROOT / name)
        inputs[name] = sha(ROOT / name)
    for name, digest in inputs.items():
        target = (ROOT / name).resolve()
        assert target.is_relative_to(ROOT) and target.is_file() and sha(target) == digest, name
    control, challenger = 'data/quality-train-v1/model.json', 'data/selfplay-training-pilot-v1/model.json'
    body = dict(schema='ordinary-selfplay-prior-immutability-v1', passed=True,
                inputs_count=len(inputs), prior_accepted_inputs=3168, inputs=dict(sorted(inputs.items())),
                control_model=control, control_model_sha256=sha(ROOT / control),
                challenger_model=challenger, challenger_model_sha256=sha(ROOT / challenger),
                new_data_generation_authorized=True, offline_sampling_review_authorized=True,
                new_training_authorized=False, new_teacher_searches_authorized=False,
                new_strength_games_authorized=False, source_data_pruning_authorized=False,
                default_promotion=False)
    assert body['control_model_sha256'] == 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
    output = ROOT / 'build/ordinary-selfplay-baseline.json'
    if output.exists():
        assert json.loads(output.read_text(encoding='utf-8')) == body
    else:
        atomic_json(output, body)
    print(config.canonical(dict(event='ordinary_selfplay_baseline', passed=True,
                               prior_inputs=len(inputs), baseline_sha256=sha(output),
                               training_authorized=False)), flush=True)


if __name__ == '__main__':
    main()
