"""Preserve the accepted ordinary coverage study before a fixed-policy value pilot."""
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
    names = ['build/ordinary-selfplay-acceptance.json',
             'build/ordinary-selfplay-acceptance-rerun.json',
             'build/ordinary-selfplay-dashboard-final.json']
    accepted, rerun, dashboard = map(read, names)
    assert accepted['passed'] is rerun['passed'] is dashboard['passed'] is True
    assert accepted['tests_run'] == 16 and accepted['actual_process_kills'] == 4
    assert accepted['training_performed'] is accepted['default_promotion'] is False
    p = ROOT / names[0]
    assert rerun['before'] == rerun['after'] == dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    assert dashboard['before'] == dashboard['after'] and dashboard['all_bytes_sizes_mtimes_unchanged'] is True
    inputs = dict(accepted['inputs'])
    assert len(inputs) == 3361
    for name, digest in dashboard['before'].items():
        assert name not in inputs or inputs[name] == digest['sha256']
        inputs[name] = digest['sha256']
    for name in names:
        assert name not in inputs or inputs[name] == sha(ROOT / name)
        inputs[name] = sha(ROOT / name)
    assert len(inputs) == 3364
    for name, digest in inputs.items():
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and sha(p) == digest, name
    control = 'data/quality-train-v1/model.json'
    body = dict(schema='ordinary-value-prior-immutability-v1', passed=True,
                inputs_count=len(inputs), prior_accepted_inputs=3361, inputs=dict(sorted(inputs.items())),
                control_model=control, control_model_sha256=sha(ROOT / control),
                validation_labeling_authorized=True, value_only_training_authorized=True,
                equal_time_strength_games_authorized=True, conditional_bounded_selfplay_authorized=True,
                source_data_pruning_authorized=False, default_promotion=False)
    assert body['control_model_sha256'] == 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
    out = ROOT / 'build/ordinary-value-baseline.json'
    if out.exists():
        assert read(out.relative_to(ROOT).as_posix()) == body
    else:
        atomic_json(out, body)
    print(config.canonical(dict(event='ordinary_value_baseline', passed=True, prior_inputs=len(inputs))), flush=True)


if __name__ == '__main__':
    main()
