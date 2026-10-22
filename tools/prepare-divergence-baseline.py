"""Protect the accepted read-only value review before new bounded diagnostics."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config
from congklak.deeper_teacher_v2 import atomic_json


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    names = ['build/ordinary-value-review-acceptance.json', 'build/ordinary-value-review-acceptance-rerun.json',
             'build/ordinary-value-review-dashboard-final.json']
    accepted, rerun, dashboard = [json.loads((ROOT / name).read_text()) for name in names]
    assert accepted['passed'] is rerun['passed'] is dashboard['passed'] is True
    assert accepted['tests_run'] == 17 and len(accepted['inputs']) == 3911
    assert accepted['policy_fixed'] is accepted['all_model_weights_fixed'] is True
    assert accepted['training_performed'] is accepted['default_promotion'] is False
    assert accepted['current_allocation'] == 'hold'
    target = ROOT / names[0]
    assert rerun['before'] == rerun['after'] == dict(sha256=sha(target), size=target.stat().st_size, mtime_ns=target.stat().st_mtime_ns)
    assert dashboard['before'] == dashboard['after'] and dashboard['all_bytes_sizes_mtimes_unchanged'] is True
    inputs = dict(accepted['inputs'])
    for name, record in dashboard['before'].items():
        assert name not in inputs or inputs[name] == record['sha256']
        inputs[name] = record['sha256']
    for name in names:
        assert name not in inputs or inputs[name] == sha(ROOT / name)
        inputs[name] = sha(ROOT / name)
    assert len(inputs) == 3914
    for name, digest in inputs.items():
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file() and sha(p) == digest, name
    body = dict(schema='divergence-prior-immutability-v1', passed=True, inputs_count=3914, prior_accepted_inputs=3911,
                inputs=dict(sorted(inputs.items())), models_fixed=True, policy_fixed=True,
                fixed_simulation_search_authorized=True, bounded_action_Q_labeling_authorized=True,
                training_authorized=False, strength_games_authorized=False, selfplay_authorized=False,
                source_data_pruning_authorized=False, previous_allocation='hold', default_promotion=False)
    output = ROOT / 'build/divergence-baseline.json'
    if output.exists(): assert json.loads(output.read_text()) == body
    else: atomic_json(output, body)
    print(config.canonical(dict(event='divergence_baseline', passed=True, prior_inputs=3914)), flush=True)


if __name__ == '__main__': main()
