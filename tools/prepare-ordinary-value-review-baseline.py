"""Preserve the sealed value-only pilot before read-only diagnosis."""
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
    names = ['build/ordinary-value-acceptance.json',
             'build/ordinary-value-acceptance-rerun.json',
             'build/ordinary-value-dashboard-final.json']
    accepted, rerun, dashboard = [json.loads((ROOT / name).read_text()) for name in names]
    assert accepted['passed'] is rerun['passed'] is dashboard['passed'] is True
    assert accepted['tests_run'] == 18 and accepted['actual_process_kills'] == 10
    assert accepted['allocation']['branch'] == 'hold' and accepted['default_promotion'] is False
    target = ROOT / names[0]
    signature = dict(sha256=sha(target), size=target.stat().st_size, mtime_ns=target.stat().st_mtime_ns)
    assert rerun['before'] == rerun['after'] == signature
    assert dashboard['before'] == dashboard['after'] and dashboard['all_bytes_sizes_mtimes_unchanged'] is True
    inputs = dict(accepted['inputs'])
    assert len(inputs) == 3776
    for name, record in dashboard['before'].items():
        assert name not in inputs or inputs[name] == record['sha256']
        inputs[name] = record['sha256']
    for name in names:
        assert name not in inputs or inputs[name] == sha(ROOT / name)
        inputs[name] = sha(ROOT / name)
    assert len(inputs) == 3779
    for name, digest in inputs.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    body = dict(schema='ordinary-value-review-prior-immutability-v1', passed=True,
                inputs_count=len(inputs), prior_accepted_inputs=3776, inputs=dict(sorted(inputs.items())),
                old_allocation='hold', policy_fixed=True, training_authorized=False,
                new_searches_authorized=False, new_strength_games_authorized=False,
                source_data_pruning_authorized=False, default_promotion=False)
    output = ROOT / 'build/ordinary-value-review-baseline.json'
    if output.exists():
        assert json.loads(output.read_text()) == body
    else:
        atomic_json(output, body)
    print(config.canonical(dict(event='ordinary_value_review_baseline', passed=True, prior_inputs=len(inputs))), flush=True)


if __name__ == '__main__': main()
