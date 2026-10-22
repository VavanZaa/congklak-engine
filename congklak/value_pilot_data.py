"""CPU-only reviewed targets and explicit sampling probabilities for one value pilot."""
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from . import config, data_pilot, natural_study, horizon_finish, deeper_teacher_v2 as teacher, features_relay, pilot
from .native import ROOT, Native, flatten

BASELINE = ROOT / 'build/value-pilot-baseline.json'
MANIFEST = ROOT / 'build/value-pilot-data.json'
ROWS = ROOT / 'build/value-pilot-rows.json'
REVIEW = ROOT / 'build/value-pilot-sampling-review.json'
MODEL = ROOT / 'data/quality-train-v1/model.json'
MODEL_SHA = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
CONTRACT = dict(base_mass=.95, ordinary_mass=.05, base_uniform_root_fraction=.9,
                base_uniform_stratum_fraction=.1, confidence_requested_depths=[8, 10, 12, 14],
                confidence_last_depths=3, confidence_span_scale_seeds=4., single_depth_factor=.5,
                disjoint_last_two_best_sets_factor=.5, loss_weight=1., draws=51200,
                target_selection='Deepest complete requested root label; replace followups, never duplicate',
                confidence_interpretation='Uncalibrated exposure heuristic; not accuracy probability or a proof',
                objective='Unit-loss expectation under the declared sampling distribution; no importance correction')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def publish(path, body):
    text = config.canonical(body) + '\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Immutable data publication differs: ' + str(path))
    else:
        teacher.atomic_json(path, body)


def baseline():
    b = read(BASELINE)
    require(b['passed'] is True and b['inputs_count'] == len(b['inputs']) == 2426, 'Pilot baseline differs')
    for name, digest in b['inputs'].items():
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT) and sha(p) == digest, 'Prior artifact changed: ' + name)
    require(sha(MODEL) == b['fixed_model_sha256'] == MODEL_SHA, 'Retained source model changed')
    return b


def confidence(evidence):
    depths = sorted(evidence)[-CONTRACT['confidence_last_depths']:]
    values = [evidence[d]['label']['target_player0'] for d in depths]
    span = max(values) - min(values)
    overlap = bool(set(evidence[depths[-2]]['label']['best_moves']) & set(evidence[depths[-1]]['label']['best_moves'])) if len(depths) > 1 else None
    volatility = 1 / (1 + span / 4.) if len(depths) > 1 else .5
    disagreement = .5 if overlap is False else 1.
    return dict(requested_depths_available=sorted(evidence), depths_used=depths, targets_player0=values,
                span_seeds=span, last_two_best_sets_overlap=overlap, volatility_factor=volatility,
                disagreement_factor=disagreement, exposure_factor=volatility * disagreement)


def distribution_summary(rows):
    mass = math.fsum(r['sampling_probability'] for r in rows)
    require(abs(mass - 1) < 1e-12 and all(r['sampling_probability'] > 0 for r in rows), 'Invalid sampling mass')
    family = defaultdict(float)
    strata = defaultdict(float)
    for row in rows:
        family[row['family']] += row['sampling_probability']
        strata[row['stratum']] += row['sampling_probability']
    return dict(roots=len(rows), families=len(family), probability=mass,
        root_effective_support=1 / math.fsum(r['sampling_probability'] ** 2 for r in rows),
        family_effective_support=1 / math.fsum(p ** 2 for p in family.values()),
        maximum_root_probability=max(r['sampling_probability'] for r in rows),
        maximum_expected_root_draws=CONTRACT['draws'] * max(r['sampling_probability'] for r in rows),
        strata=dict(sorted(strata.items())),
        low_active_5_24_probability=math.fsum(r['sampling_probability'] for r in rows if r['stratum'].startswith('5-24/')),
        effective_support_definition='Inverse probability concentration, not independent games or gradient ESS')


def build():
    b = baseline()
    _, corpus, _ = teacher.load(teacher_depth=8)
    source_keys = set(read(ROOT / 'build/value-data.json')['train_state_keys'])
    require(len(source_keys) == 7592, 'Previous admitted replay set differs')
    pr, pl, _, _ = data_pilot.load_rows()
    nr, n8, n10, _, _, _ = natural_study.load_rows()
    _, hs, _ = horizon_finish.load_rows()
    h14 = horizon_finish.combined_labels(hs)
    roots = dict(pr)
    for key, wrapper in nr.items():
        if key in roots:
            require(roots[key] == wrapper['root'], 'Followup wrapper differs')
        roots[key] = wrapper['root']
    evidence = defaultdict(dict)
    for depth, items in ((8, pl), (8, n8), (10, n10), (12, hs['h12_references']), (14, h14)):
        for key, item in items.items():
            if depth in evidence[key]:
                require(evidence[key][depth] == item, 'Duplicate same-depth target differs')
            evidence[key][depth] = item
    require(Counter(r['cohort'] for r in roots.values()) == dict(training_candidate=61, validation=14, diagnostic=8), 'Unique ordinary cohorts differ')
    projection = read(ROOT / 'build/value-pilot-benchmark-projection.json')
    previous_registry = read(ROOT / 'build/value-registry.json')
    protected_q = set(previous_registry['quotients']) | set(projection['quotients'])
    protected_f = set(previous_registry['families']) | set(projection['families'])
    native = Native()
    old_games = 0
    with closing(sqlite3.connect((ROOT / 'data/value-benchmark-v1/work.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        for body, digest in db.execute('SELECT body,hash FROM games ORDER BY idx'):
            game = json.loads(body)
            require(config.digest(game) == digest, 'Previous arena row corrupt')
            p = game['start']
            protected_f.add(game['family'])
            for action in [*game['moves'], None]:
                p = native.normalize(flatten(p), actor=p['controller'], total=98)['position']
                protected_q.add(pilot.quotient(p).hex())
                if action is not None:
                    t = native.apply(flatten(p), actor=p['controller'], action=action, total=98, trace=False)
                    require(t['ok'], 'Previous arena action invalid')
                    p = t['position']
            require(p == game['final'], 'Previous arena final differs')
            old_games += 1
    require(old_games == 192, 'Previous value arena coverage differs')
    for row in corpus['development']:
        protected_f.add(row['family'])
        protected_q.add(pilot.quotient(row['state']).hex())
    for row in roots.values():
        if row['cohort'] != 'training_candidate':
            protected_f.add(row['family'])
            protected_q.add(pilot.quotient(row['state']).hex())
    base_candidates = [r for r in corpus['train'] if r['state_key'] in source_keys]
    require(len(base_candidates) == len(source_keys), 'Replay keys missing')
    all_candidates = [*base_candidates, *[r for r in roots.values() if r['cohort'] == 'training_candidate']]
    bridge_f = {r['family'] for r in all_candidates if pilot.quotient(r['state']).hex() in protected_q}
    excluded = []
    def admitted(row):
        reasons = []
        if row['family'] in protected_f:
            reasons.append('protected_family')
        if pilot.quotient(row['state']).hex() in protected_q:
            reasons.append('protected_quotient')
        if row['family'] in bridge_f:
            reasons.append('protected_family_via_quotient')
        if reasons:
            excluded.append(dict(state_key=row['state_key'], family=row['family'], reasons=reasons))
        return not reasons
    base = []
    for row in base_candidates:
        if admitted(row):
            base.append(dict(state_key=row['state_key'], state=row['state'], family=row['family'],
                cohort='replay_training', stratum=row['stratum'], legal_mask=row['legal_mask'],
                cached_features=row['cached_features'], label=row['label'], teacher_depth=8,
                pool='base', label_source='original_H8_replay', loss_weight=1.))
    ordinary = dict(train=[], ordinary_development=[], diagnostic=[])
    new_evidence = []
    for key, root in sorted(roots.items()):
        depth = max(evidence[key])
        item = evidence[key][depth]
        natural_study.validate_label(item, dict(root=root, source_root_sha256=config.digest(root)), depth, native)
        x, d, r = features_relay.encode(root['state'], 127, native)
        row = dict(state_key=key, state=root['state'], family=root['family'], cohort=root['cohort'],
                   stratum=root['stratum'], legal_mask=root['legal_mask'], cached_features=dict(x=x, d=d, r=r),
                   label=item['label'], teacher_depth=depth, pool='ordinary', label_source='complete_requested_H' + str(depth),
                   original_root_sha256=config.digest(root), label_item_sha256=config.digest(item), loss_weight=1.)
        cf = confidence(evidence[key])
        row['exposure_review'] = cf
        row['target_evidence'] = [dict(depth=k, item_sha256=config.digest(v), target_player0=v['label']['target_player0'],
                                       best_moves=v['label']['best_moves']) for k, v in sorted(evidence[key].items())]
        if root['cohort'] == 'training_candidate':
            if admitted(root):
                ordinary['train'].append(row)
        else:
            ordinary['diagnostic' if root['cohort'] == 'diagnostic' else 'ordinary_development'].append(row)
    require(len(ordinary['train']) == 61, 'Protected filtering changed expected ordinary optimizer pool; review before freezing')
    base = sorted(base, key=lambda r: r['state_key'])
    counts = Counter(r['stratum'] for r in base)
    n, s = len(base), len(counts)
    require(n > 0 and s == 14, 'Replay strata lost coverage')
    for row in base:
        row['sampling_probability'] = .95 * (.9 / n + .1 / (s * counts[row['stratum']]))
    z = math.fsum(r['exposure_review']['exposure_factor'] for r in ordinary['train'])
    for row in ordinary['train']:
        row['sampling_probability'] = .05 * row['exposure_review']['exposure_factor'] / z
        new_evidence.append(dict(state_key=row['state_key'], family=row['family'], teacher_depth=row['teacher_depth'],
            **row['exposure_review'], sampling_probability=row['sampling_probability'], expected_draws=51200 * row['sampling_probability']))
    train = [*base, *ordinary['train']]
    require(len({r['state_key'] for r in train}) == len(train), 'Optimizer duplicate root')
    require(len({pilot.quotient(r['state']) for r in ordinary['train']}) == 61, 'Ordinary quotient duplicate')
    dev = [dict(state_key=r['state_key'], state=r['state'], family=r['family'], cohort='reused_development',
                stratum=r['stratum'], legal_mask=r['legal_mask'], cached_features=r['cached_features'],
                label=r['label'], teacher_depth=8, pool='base', label_source='original_H8_development', loss_weight=1.)
           for r in corpus['development']]
    groups = dict(train=train, development=dev, ordinary_development=ordinary['ordinary_development'], diagnostic=ordinary['diagnostic'])
    for name in ('development', 'ordinary_development', 'diagnostic'):
        require(not ({r['family'] for r in train} & {r['family'] for r in groups[name]}), 'Family split leakage: ' + name)
        require(not ({pilot.quotient(r['state']) for r in train} & {pilot.quotient(r['state']) for r in groups[name]}), 'Quotient split leakage: ' + name)
    files = ('congklak/value_pilot_data.py', 'tools/prepare-value-pilot-data.py', 'build/value-pilot-baseline.json',
             'build/value-pilot-benchmark-projection.json')
    inputs = dict(b['inputs'])
    inputs.update({p: sha(ROOT / p) for p in files})
    dataset = dict(schema='value-pilot-explicit-row-distribution-v1', groups=groups)
    publish(ROWS, dataset)
    review = dict(schema='value-pilot-sampling-review-v1', passed=True, contract=CONTRACT,
        summary=distribution_summary(train), pool_mass=dict(base=math.fsum(r['sampling_probability'] for r in base),
            ordinary=math.fsum(r['sampling_probability'] for r in ordinary['train'])),
        base_rows=len(base), ordinary_rows=61,
        ordinary_families=len({r['family'] for r in ordinary['train']}),
        ordinary_family_probabilities={family: math.fsum(r['sampling_probability'] for r in ordinary['train'] if r['family'] == family)
                                       for family in sorted({r['family'] for r in ordinary['train']})},
        old_replay_rows=7592, excluded=excluded,
        ordinary_teacher_depths=dict(sorted(Counter(r['teacher_depth'] for r in ordinary['train']).items())),
        ordinary_per_root=new_evidence, historical_uniform_stratum_tiny9_mass=.14285714285714285,
        active_weights_changed_only_for_new_pilot=True, loss_weights_are_all_one=True,
        uncertainty='Finite-depth targets and uncalibrated exposure factors; selected roots do not estimate general reliability.',
        ancestral_exposure='Frozen trunk/policy retain ancestral training and consumed-development exposure. New exclusions are not unlearning.',
        attribution='One bundled data/sampler/small-head-refit experiment; no isolated claim about any component.',
        pruning='No source rows deleted. Protected rows receive zero new optimizer exposure; all admitted replay strata retain positive mass.')
    publish(REVIEW, review)
    inputs[ROWS.relative_to(ROOT).as_posix()] = sha(ROWS)
    inputs[REVIEW.relative_to(ROOT).as_posix()] = sha(REVIEW)
    body = dict(schema='value-pilot-reviewed-data-v1', passed=True, rules_digest=config.RULES_DIGEST,
        source_model_sha256=MODEL_SHA, sampling=CONTRACT, groups={k:len(v) for k,v in groups.items()},
        train_rows=len(train), train_keys=[r['state_key'] for r in train],
        row_groups_sha256={k:config.digest(v) for k,v in groups.items()}, dataset_sha256=sha(ROWS),
        base_rows=len(base), new_ordinary_train_rows=61, excluded_replay_rows=7592-len(base),
        sealed_final_test_consumed=False, protected_old_value_games=192,
        protected_families=sorted(protected_f | bridge_f), protected_quotients=sorted(protected_q),
        inputs=dict(sorted(inputs.items())), no_teacher_searches=True, no_optimizer_execution=True)
    for name, digest in inputs.items():
        require(sha(ROOT / name) == digest, 'Data input changed: ' + name)
    require('torch' not in sys.modules, 'CPU data preparation imported Torch')
    publish(MANIFEST, body)
    return body, groups


def prepare():
    manifest = read(MANIFEST)
    require(manifest['schema'] == 'value-pilot-reviewed-data-v1' and manifest['passed'] is True and
            manifest['sampling'] == CONTRACT and manifest['source_model_sha256'] == sha(MODEL) == MODEL_SHA,
            'Reviewed data identity differs')
    for name, digest in manifest['inputs'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'Reviewed data input changed: ' + name)
    dataset = read(ROWS)
    groups = dataset['groups']
    require(dataset['schema'] == 'value-pilot-explicit-row-distribution-v1' and
            {k:len(v) for k,v in groups.items()} == manifest['groups'] and
            {k:config.digest(v) for k,v in groups.items()} == manifest['row_groups_sha256'], 'Reviewed rows changed')
    require(len(groups['train']) == manifest['train_rows'] and [r['state_key'] for r in groups['train']] == manifest['train_keys'], 'Optimizer row order differs')
    distribution_summary(groups['train'])
    return manifest, groups
