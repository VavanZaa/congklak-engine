"""Isolated, immutable ownership-resolution evidence and scope helpers."""
import json
from pathlib import Path
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _publish

ROOT = common.ROOT
FOLDER = ROOT / 'build/ownership-resolution-v1'
PRIOR = ROOT / 'build/tracing-speed-v1/acceptance.json'
SCHEMA = 'ownership-resolution-v1'
MODELS = ('data/quality-train-v1/model.json',
          'data/value-training-v1/training/model.json',
          'data/training-stage5-deeper-h8-f127-seed2718/model.json')
ZERO = dict(optimizer_steps=0, optimizer_draws=0, new_teacher_labels=0,
            model_forwards=0, native_searches=0, production_games=0,
            new_metadata_candidates=0)
need, read, sha = common.require, common.read, common.sha


def normalized(body):
    return json.loads(config.canonical(body))


def atomic(path, body):
    return _publish(path, normalized(body))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def path_of(path, subtree=''):
    result = Path(path).resolve()
    base = (FOLDER / subtree).resolve()
    need(result == base or result.is_relative_to(base), 'Resolution output namespace differs')
    return result


def baseline():
    """Bind the prior closure; the independent/final passes verify all bytes."""
    prior = read(PRIOR)
    need(prior['schema'] == 'tracing-speed-acceptance-v1' and prior['passed'] is True and
         prior['models_fixed'] is True and prior['policy_fixed'] is True and
         prior['training_authorized'] is False, 'Prior accepted scope differs')
    inputs = dict(prior['inputs'])
    inputs[relative(PRIOR)] = sha(PRIOR)
    body = dict(schema=SCHEMA+'-baseline', previous_acceptance=relative(PRIOR),
                previous_acceptance_sha256=sha(PRIOR), inputs=inputs,
                inputs_count=len(inputs), models=common.pins(MODELS),
                inherited_open_holds=prior['open_holds'], full_rehash_pending=True)
    return atomic(FOLDER/'baseline.json', body)
