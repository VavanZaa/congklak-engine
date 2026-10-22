"""Immutable evidence identities and exact five-root diagnostic projection."""
from pathlib import Path
import hashlib
import json
from . import config, pilot
from .native import ROOT, Native, flatten
from .deeper_teacher_v2 import atomic_json

BASELINE = ROOT / 'build/divergence-baseline.json'
PANEL = ROOT / 'build/divergence-panel.json'
SCOPE = ROOT / 'docs/DIVERGENCE_STUDY_SCOPE.md'
CONTROL = ROOT / 'data/quality-train-v1/model.json'
CANDIDATE = ROOT / 'data/ordinary-value-pilot-v1/model.json'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def pins(names):
    result = {}
    for name in names:
        p = Path(name) if Path(name).is_absolute() else ROOT / name
        p = p.resolve(); require(p.is_relative_to(ROOT), 'Input outside workspace')
        result[p.relative_to(ROOT).as_posix()] = sha(p)
    return result


def current(mapping):
    for name, digest in mapping.items():
        p = (ROOT / name).resolve()
        require(p.is_relative_to(ROOT) and p.is_file() and sha(p) == digest, 'Input changed: ' + name)


def publish(path, body):
    path = Path(path)
    if path.exists(): require(path.read_text(encoding='utf-8') == config.canonical(body) + '\n', 'Immutable divergence output differs: ' + str(path))
    else: atomic_json(path, body)
    return body


def baseline(verify_all=False):
    body = read(BASELINE)
    require(body['schema'] == 'divergence-prior-immutability-v1' and body['passed'] is True and
            type(body['inputs_count']) is int and len(body['inputs']) == body['inputs_count'] == 3914, 'Wrong divergence baseline')
    if verify_all: current(body['inputs'])
    return body


def rawkey(p): return bytes(flatten(p) + [p['controller']]).hex()


def projection():
    frozen = baseline()
    name = 'build/ordinary-value-loss-review.json'
    source = read(ROOT / name)
    require(source['passed'] is True and sha(ROOT / name) == frozen['inputs'][name], 'Sealed loss review changed')
    seats = sorted((row for row in source['seats'] if row['baseline'] == 'H_v1' and
                    not row['first_divergence']['identical_action_trajectory']), key=lambda r: (r['pair'], r['candidate_seat']))
    require([(row['pair'], row['candidate_seat']) for row in seats] == [(n, 1) for n in (2, 4, 6, 7, 8)], 'Five-root selection differs')
    native = Native(); roots = []
    for row in seats:
        d = row['first_divergence']; p = config.position(d['state'])
        normal = native.normalize(flatten(p), actor=p['controller'], total=98)
        active, stratum = pilot._mechanism(p, native)
        require(not normal['terminal'] and normal['position'] == p and p['controller'] == row['candidate_seat'] == 1,
                'Diagnostic actor/normalization differs')
        require(stratum == d['divergence_stratum'] and d['candidate_priors_equal'] is True, 'Saved divergence mechanism/policy differs')
        control = d['control']; candidate = d['candidate']
        roots.append(dict(id='pair' + str(row['pair']).zfill(2) + '-seat1', pair=row['pair'], candidate_seat=1,
            state=p, state_key=rawkey(p), legal_mask=sum(1 << a for a in normal['legal_actions']),
            legal_actions=normal['legal_actions'], active_seeds=active, stratum=stratum, family=row['family'],
            actor=p['controller'], choice_index=d['choice_index'], control_action=control['saved_decision']['decision']['decision']['chosen'],
            candidate_action=candidate['saved_decision']['decision']['decision']['chosen'],
            timed_margin_delta=row['candidate_minus_control_margin'], source_seat_sha256=config.digest(row),
            source_divergence_sha256=config.digest(d), source_control_decision_sha256=control['source']['decision_sha256'],
            source_candidate_decision_sha256=candidate['source']['decision_sha256'],
            source_control_game=row['control_game_index'], source_candidate_game=row['candidate_game_index'],
            role='consumed_forensic_diagnostic/no_optimizer_or_release_holdout',
            existing_Q_coverage=d['existing_Q_comparison']['coverage']))
    require([r['active_seeds'] for r in roots] == [10, 16, 36, 63, 27] and
            [len(r['legal_actions']) for r in roots] == [4, 5, 4, 6, 5] and len({r['state_key'] for r in roots}) == 5,
            'Exact root or action coverage differs')
    body = dict(schema='five-Hv1-divergence-panel-v1', roots=roots, rules_digest=config.RULES_DIGEST,
                source_report_sha256=sha(ROOT / name), baseline_sha256=sha(BASELINE),
                actor_band_confounds='All five roots are player1; no population or player-symmetry inference from this panel.',
                selection='All five sealed H_v1 first divergences, including three adverse and two unchanged-margin controls.',
                models=pins([CONTROL, CANDIDATE]), previous_allocation='hold', training_authorized=False)
    body['sha256'] = config.digest(body)
    return body


def prepare_panel(): return publish(PANEL, projection())


def load_panel():
    body = read(PANEL)
    require(config.canonical(body) == config.canonical(projection()), 'Fixed five-root projection changed')
    return body
