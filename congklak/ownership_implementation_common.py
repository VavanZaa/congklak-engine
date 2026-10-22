"""Immutable identities for the traced handoff implementation stage."""
from pathlib import Path
import json
from . import config, neurodiag_common as common
from .value_diagnostics import atomic as _atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/ownership-implementation-v1'
PRIOR=ROOT/'build/ownership-protocol-v1/acceptance.json'
need,read,sha=common.require,common.read,common.sha
TRAINING_ZERO=dict(optimizer_steps=0,optimizer_draws=0,new_teacher_labels=0,production_games=0)


def normalized(body):return json.loads(config.canonical(body))


def atomic(path,body):return _atomic(path,normalized(body))


def baseline():
    path=FOLDER/'baseline.json'
    if path.exists():
        result=read(path)
        need(result['schema']=='ownership-implementation-baseline-v1' and result['passed'] is True and
             result['inputs_count']==len(result['inputs'])==59154,'Implementation baseline differs')
    else:
        prior=read(PRIOR)
        need(prior['passed'] is True and prior['execution_ready'] is False and len(prior['inputs'])==59153,
             'Prior protocol acceptance differs')
        inputs=dict(prior['inputs']);inputs.update(common.pins([PRIOR]))
        result=dict(schema='ownership-implementation-baseline-v1',passed=True,inputs_count=len(inputs),inputs=inputs)
    common.current(result['inputs'])
    return atomic(path,result)


def path_of(path,subtree):
    path=Path(path).resolve();base=(FOLDER/subtree).resolve()
    need(path==base or path.is_relative_to(base),'Implementation output namespace differs')
    return path
