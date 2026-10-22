"""Describe the fixed family-balanced distribution and reconstructed committed draws."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from congklak import config, selfplay_training_pilot as learning

OUTPUT = ROOT / 'build/selfplay-pilot-sampling-review.json'


def review():
    manifest, rows = learning.load_replay()
    proof = learning.witness()
    result = learning.read(learning.DEFAULT / 'result.json')
    audit_path = ROOT / 'build/selfplay-pilot-audit/model.json'
    audit = learning.read(audit_path)
    independent = audit['evidence']['production_optimizer_audit']
    assert audit['passed'] is independent['passed'] is True
    assert independent['steps_replayed'] == 200 and independent['draws_replayed'] == 25600
    assert len(independent['checkpoints']) == 10
    assert independent['all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal'] is True
    for name, digest in independent['inputs'].items():
        assert learning.sha(ROOT / name) == digest, 'Independent update input changed: ' + name
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        assert sum(map(sum, row['state']['pits'])) >= 25, 'Late position entered midgame replay'
        groups[row['family']].append(index)
    families = sorted(groups)
    assert len(families) >= 8 and groups == result['identity']['sampler_groups']
    assert result['draws'] == result['steps_completed'] * result['identity']['batch_size'] == 25600
    rng = random.Random(result['identity']['seed'])
    draws = Counter()
    for _ in range(result['draws']):
        # Separate reconstruction from the producer's sampler; same declared
        # two-choice Python RNG contract, audited against actual checkpoints.
        family = families[rng.randrange(len(families))]
        index = groups[family][rng.randrange(len(groups[family]))]
        draws[index] += 1
    draw_ledger_sha = config.digest([[row['id'], draws[index]] for index, row in enumerate(rows)])
    assert draw_ledger_sha == independent['sampler_row_draw_ledger_sha256']
    probabilities = [1 / (len(families) * len(groups[row['family']])) for row in rows]
    assert abs(math.fsum(probabilities) - 1) < 1e-12
    expected = defaultdict(float)
    observed = defaultdict(Counter)
    records = []
    for index, (row, p) in enumerate(zip(rows, probabilities)):
        legal = int(row['legal_mask']).bit_count()
        for category, label in (('family', row['family']), ('stratum', row['stratum']),
                                ('band', row['stratum'].split('/')[0]), ('legal_actions', str(legal))):
            expected[(category, label)] += p
            observed[category][label] += draws[index]
        records.append(dict(id=row['id'], family=row['family'], stratum=row['stratum'],
                            legal_actions=legal, sampling_probability=p,
                            expected_draws=p * result['draws'], reconstructed_draws=draws[index]))
    distributions = {}
    for category in ('family', 'stratum', 'band', 'legal_actions'):
        labels = sorted(label for cat, label in expected if cat == category)
        distributions[category] = {label: dict(probability=expected[(category, label)],
            expected_draws=expected[(category, label)] * result['draws'],
            reconstructed_draws=observed[category][label]) for label in labels}
        assert sum(observed[category].values()) == result['draws']
    inputs = (Path(__file__), learning.DEFAULT / 'work.sqlite3',
              learning.DEFAULT / 'selfplay-manifest.json', learning.DEFAULT / 'result.json',
              learning.WITNESS, ROOT / 'congklak/selfplay_training_pilot.py', learning.REGISTRY,
              learning.PLAN, learning.IMPLEMENTATION, audit_path)
    body = dict(schema='selfplay-pilot-fixed-sampling-review-v1', passed=True,
                train_rows=len(rows), families=len(families), draws=result['draws'],
                root_effective_support=1 / math.fsum(p * p for p in probabilities),
                family_effective_support=len(families), maximum_root_probability=max(probabilities),
                distributions=distributions, row_exposures=records,
                draw_accounting='Reconstructed from the fixed seed and two-choice sampler; actual sampler RNG and all checkpoints independently audited. Draw IDs were not separately logged.',
                support_definition='Inverse probability concentration, not independent games or gradient ESS',
                loss_weights='Unit per draw; family/root sampling is the only weighting layer',
                teacher_rehearsal=False, optimization_distribution_changed=False,
                source_rows_pruned=False, duplicate_pruning=manifest['summary']['pruned_duplicate_rows'],
                checkpoint_witness_sha256=proof['proof_file_sha256'],
                sampler_row_draw_ledger_sha256=draw_ledger_sha,
                inputs={p.relative_to(ROOT).as_posix(): learning.sha(p) for p in inputs})
    learning.publish(OUTPUT, body)
    return body


if __name__ == '__main__':
    value = review()
    print(config.canonical(dict(event='selfplay_pilot_sampling_review', passed=True,
                               rows=value['train_rows'], families=value['families'], draws=value['draws'],
                               root_effective_support=value['root_effective_support'])), flush=True)
