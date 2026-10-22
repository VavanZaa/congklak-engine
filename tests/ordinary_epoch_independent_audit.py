"""Independent ordinary-epoch checks; never a producer or training entry point.

The initial facts check reads accepted reservation metadata and existing model
envelopes only. Further checks bind actual frozen campaign artifacts.
"""
from collections import Counter
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'build/ordinary-epoch-pilot-v1/audit'
CONTROL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
PRIOR_SHA = '7d89313c5326f8fe7e4cd153e6cf71e4bda37f147e03d3d455069f8aa3c76dd6'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def pins(paths):
    return {Path(p).resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths}


def publish(name, body):
    path = OUT / name
    need(not path.exists(), 'Independent evidence is immutable: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write((canonical(body) + '\n').encode())
    return body


def facts():
    """Derive treatment/role counts directly from the accepted joint graph."""
    acceptance = ROOT / 'build/ownership-resolution-v1/acceptance.json'
    need(sha(acceptance) == PRIOR_SHA, 'Wrong accepted ownership epoch')
    accepted = read(acceptance)
    joint_path = ROOT / 'build/ownership-resolution-v1/production/units/000695.json'
    epoch_path = ROOT / 'build/ownership-resolution-v1/production/units/000696.json'
    for path in (joint_path, epoch_path):
        need(accepted['inputs'][path.relative_to(ROOT).as_posix()] == sha(path), 'Changed accepted reservation')
    joint, epoch = read(joint_path)['result'], read(epoch_path)['result']
    roots = joint['future_support']['roots']
    counts = Counter(r['cell'] for r in roots)
    expected = {b + '/ordinary/actor' + str(a): n for b, a, n in (
        ('25-49', 0, 39), ('25-49', 1, 34), ('50-74', 0, 58), ('50-74', 1, 61))}
    need(dict(counts) == expected and len(roots) == 192, 'Wrong four-cell ordinary population')
    need(len({r['state_key'] for r in roots}) == len({r['joint_component'] for r in roots}) == 192,
         'Duplicate admitted raw roots or components')
    masses = Counter()
    for row in roots:
        w = row['weight']
        need(type(w['numerator']) is int and type(w['denominator']) is int,
             'Noninteger exact sampling weight')
        weight = Fraction(w['numerator'], w['denominator'])
        need(weight == Fraction(1, 4 * counts[row['cell']]), 'Changed within-cell sampling weight')
        masses[row['cell']] += weight
        need(row['eligible_for_future_ordinary_recipe'] is True and
             row['joint_component'] not in joint['whole_quarantine_joint_components'],
             'Whole-component quarantine was weakened')
    need(set(masses.values()) == {Fraction(1, 4)}, 'Cell probability mass differs')
    need(epoch['prerequisites_ready'] is True and epoch['historical_provenance_complete'] is False and
         joint['inherited_unknown_background'] is True and joint['original_512_recipe_admitted'] is False,
         'Historical background or old-pool restrictions differ')
    control = ROOT / 'data/quality-train-v1/model.json'
    need(sha(control) == CONTROL_SHA, 'Policy/control source changed')
    source = read(control)
    need(hashlib.sha256(canonical(source['model']).encode()).hexdigest() == source['sha256'],
         'Control model envelope differs')
    w = source['model']['weights']
    def number_count(value):
        return sum(map(number_count, value)) if isinstance(value, list) else 1
    need(len(w) == 8 and sum(number_count(v) for v in w.values()) == 12872, 'Frozen model shape differs')
    need('torch' not in sys.modules, 'Metadata facts unexpectedly imported Torch')
    return dict(schema='ordinary-epoch-independent-facts-v1', passed=True,
                source_inputs=pins([acceptance, joint_path, epoch_path, control, Path(__file__)]),
                preserved_inputs=len(accepted['inputs']) + 1,
                ordinary_counts=expected, cell_mass={c: '1/4' for c in expected},
                train_roots=192, frozen_parameters=12872, frozen_tensor_count=8,
                policy_fixed=True, inherited_unknown_background=True,
                historical_provenance_complete=False, original_512_recipe_admitted=False,
                new_searches=0, new_labels=0, optimizer_steps=0, model_forwards=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('facts',))
    args = parser.parse_args()
    body = facts()
    publish('facts-v1.json', body)
    print(canonical(body))


if __name__ == '__main__':
    main()
