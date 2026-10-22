"""Scalar-double diagnostics on the screened, cached development positions.

The development targets never enter the optimizer. Native features and H8
targets are read from their accepted captures; diagnostics generate no new
search, feature simulation or native inference.
"""
from collections import Counter
import math
from . import config, guarded_value_pilot_v2_common as stage
from . import guarded_value_pilot_v2_data as data
from . import guarded_value_pilot_v2_sidecar as side
from . import guarded_value_pilot_v2_train as training
from . import ordinary_epoch_fit as arithmetic
from .valuepilot_reservations import Writer

ROOT, DEFAULT = stage.ROOT, stage.FOLDER / 'fit'
SCHEMA = stage.SCHEMA + '-fit'
CANDIDATE = training.DEFAULT / 'model.json'
SOURCES = ('congklak/guarded_value_pilot_v2_fit.py',
    'congklak/guarded_value_pilot_v2_common.py', 'congklak/guarded_value_pilot_v2_data.py',
    'congklak/guarded_value_pilot_v2_sidecar.py', 'congklak/guarded_value_pilot_v2_train.py',
    'congklak/ordinary_epoch_fit.py', 'congklak/config.py')
need, read, sha = stage.need, stage.read, stage.sha
_dot, _base, _correction = arithmetic._dot, arithmetic._base, arithmetic._correction
_huber, _metrics = arithmetic._huber, arithmetic._metrics


def _same(a, b):
    return config.canonical(a) == config.canonical(b)


def _validate_rows(rows, manifest):
    members = data.membership()
    lookup = {row['state_key']: row for row in members['fit']}
    cases = set(members['cases'])
    need(type(rows) is list and len(rows) == len(lookup) == 128 and
         len({row['state_key'] for row in rows}) == 128 and
         {row['state_key'] for row in rows} == set(lookup) and len(cases) == 32 and
         set(manifest['case_state_keys']) == cases, 'Screened fit/case membership differs')
    for row in rows:
        meta = lookup[row['state_key']]
        state = config.position(row['state'])
        actor, active = state['controller'], sum(map(sum, state['pits']))
        need(row['state_key'] == stage.common.rawkey(state) and
             row['family'] == meta['family'] and row['cell'] == meta['cell'] and
             row['component'] == row['joint_component'] == meta['component'] and
             meta['optimizer_blocked'] is True and type(row['is_case']) is bool and
             row['is_case'] is (row['state_key'] in cases) and row['actor'] == actor and
             row['D'] == state['stores'][0] - state['stores'][1] and
             row['r'] == active * (1 if actor == 0 else -1) and 25 <= active <= 74 and
             len(row['x']) == len(row['x_native']) == 127 and
             all(type(value) in (float, int) and math.isfinite(value) for value in row['x_native']) and
             row['x'] == [data.fp32(value) for value in row['x_native']] and
             type(row['target_player0']) is int and -1 <= row['target_z'] <= 1 and
             row['target_z'] == (row['target_player0'] - row['D']) / row['r'],
             'Screened development features, normalization or ownership differ')
        teacher = row['teacher']
        legal = [index for index, seeds in enumerate(state['pits'][actor]) if seeds]
        q = {item['pit']: item['value_player0'] for item in teacher['action_q']}
        need(type(teacher['requested_depth']) is int and teacher['requested_depth'] == 8 and
             type(teacher['completed_depth']) is int and teacher['completed_depth'] == 8 and
             teacher['all_legal_Q_complete'] is True and set(q) == set(legal) and
             len(q) == len(teacher['action_q']) and
             all(type(item['pit']) is int for item in teacher['action_q']) and
             all(type(value) is int for value in q.values()), 'Incomplete or mixed H8 development targets')
        target = (max if actor == 0 else min)(q.values())
        need(target == row['target_player0'] == teacher['target_player0'] and
             teacher['best_moves'] == [action for action in legal if q[action] == target],
             'Captured H8 value and action targets disagree')
    need(sum(row['is_case'] for row in rows) == 32, 'Screened case subset changed')


def _derive():
    stage.admit_plan()
    rows, manifest = data.load_fit()
    result = training.completed()
    need(result['steps'] == 400 and result['engineering_fixture'] is False,
         'Development fit requires the one final400 candidate')
    _validate_rows(rows, manifest)
    control = read(side.BASE)
    candidate = side.verify_export(CANDIDATE)
    need(_same(candidate['model']['base_export'], control), 'Candidate changed the frozen policy or trunk')
    control_weights = control['model']['weights']
    candidate_weights = candidate['model']['base_export']['model']['weights']
    branch = candidate['model']['branch_weights']
    predictions = []
    for row in rows:
        x = row['x_native']
        u, policy = _base(x, control_weights)
        candidate_u, candidate_policy = _base(x, candidate_weights)
        need(u == candidate_u and policy == candidate_policy, 'Frozen base/policy arithmetic changed')
        z, candidate_z = math.tanh(u), math.tanh(candidate_u + _correction(x, branch))
        predictions.append(dict(state_key=row['state_key'], family=row['family'], cell=row['cell'], actor=row['actor'],
            active_seeds=abs(row['r']), is_case=row['is_case'], target_player0=row['target_player0'], target_z=row['target_z'],
            control=dict(z=z, value_player0=row['D'] + row['r'] * z),
            candidate=dict(z=candidate_z, value_player0=row['D'] + row['r'] * candidate_z),
            policy_logits_exact=True, source_row_sha256=config.digest(row)))
    groups = dict(overall=predictions, cases=[row for row in predictions if row['is_case']])
    groups.update({kind: [row for row in predictions if '/' + kind + '/' in row['cell']]
                   for kind in ('ordinary', 'capture', 'bonus')})
    groups.update({'actor' + str(actor): [row for row in predictions if row['actor'] == actor] for actor in (0, 1)})
    groups.update({'band' + band: [row for row in predictions if (row['active_seeds'] < 50) == (band == '25-49')]
                   for band in ('25-49', '50-74')})
    groups.update({cell: [row for row in predictions if row['cell'] == cell] for cell in sorted({row['cell'] for row in predictions})})
    metrics = {name: {model: _metrics(group, model) for model in ('control', 'candidate')} for name, group in groups.items()}
    overall, capture = metrics['overall'], metrics['capture']
    checks = dict(normalized_huber_improved=overall['candidate']['normalized_huber'] < overall['control']['normalized_huber'] - 1e-12,
        overall_seed_mae_nonregressed=overall['candidate']['seed_mae'] <= overall['control']['seed_mae'] + 1e-9,
        capture_seed_mae_nonregressed=capture['candidate']['seed_mae'] <= capture['control']['seed_mae'] + 1e-9,
        policy_logits_exact=all(row['policy_logits_exact'] for row in predictions),
        target_kind_preserved=True, fit_case_membership_preserved=True)
    files = [ROOT / name for name in SOURCES] + [stage.PLAN, stage.PRIOR, side.BASE, CANDIDATE]
    files += [training.DEFAULT / name for name in ('identity.json', 'catalog.json', 'certificate.json', 'result.json')]
    files += [p for p in data.FIT_DEFAULT.iterdir() if p.is_file() and p.suffix == '.json']
    return dict(schema=SCHEMA + '-report', passed=all(checks.values()), checks=checks, metrics=metrics, rows=predictions,
        input_pins=stage.common.pins(files), fit_manifest_sha256=sha(data.FIT_DEFAULT / 'manifest.json'),
        counts=dict(Counter(row['cell'] for row in predictions)),
        target_kind='complete-H8/H_v1-untrained-oraclefree/full-legal-root-Q/player0-margin',
        target_normalization='(y-D)/signed_active; no clipping; endpoints retained',
        numeric_backend='Python scalar-double/input-order separate multiply/add/captured native-double127features/exportedFP32weights',
        primary_feature_precision='native captured double', training_feature_precision='existing FP32 cache unchanged',
        native_conformance_separate=True, policy_proof=side.policy_proof(),
        cached_feature_model_calculations=256, new_position_queries=0, new_encodes=0, new_searches=0,
        native_model_forwards=0, optimizer_steps=0, policy_fixed=True, model_promotion=False,
        acquired_origins_optimizer_blocked=True, inherited_unknown_background=True, historically_unseen=False)


def completed(output=DEFAULT):
    path = stage.path_of(output)
    need(path == DEFAULT.resolve(), 'Fit output scope differs')
    report = read(path / 'report.json')
    need(_same(read(path / 'certificate.json'), dict(schema=SCHEMA + '-certificate',
        report_sha256=sha(path / 'report.json'), source_inputs=report['input_pins'])), 'Fit certificate differs')
    need(_same(report, _derive()), 'Fit report differs from accepted captures/exports')
    return report


def run(output=DEFAULT):
    path = stage.path_of(output)
    need(path == DEFAULT.resolve(), 'Fit output scope differs')
    if (path / 'certificate.json').exists():
        return completed(path)
    report = _derive()
    path.mkdir(parents=True, exist_ok=True)
    with Writer(path):
        if (path / 'certificate.json').exists():
            return completed(path)
        if (path / 'report.json').exists():
            need(_same(read(path / 'report.json'), report), 'Uncommitted fit report differs')
        else:
            stage.atomic(path / 'report.json', report)
        stage.atomic(path / 'certificate.json', dict(schema=SCHEMA + '-certificate',
            report_sha256=sha(path / 'report.json'), source_inputs=report['input_pins']))
    return completed(path)
