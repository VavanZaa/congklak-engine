"""Isolated trace-speed stage identities and exact immutable publication."""
import json
from pathlib import Path
from . import config,neurodiag_common as common
from .value_diagnostics import atomic as publish

ROOT=common.ROOT
FOLDER=ROOT/'build/tracing-speed-v1'
PRIOR=ROOT/'build/ownership-implementation-v1/acceptance.json'
ZERO=dict(optimizer_steps=0,optimizer_draws=0,new_teacher_labels=0,production_games=0)
need,read,sha=common.require,common.read,common.sha

def normalized(body):return json.loads(config.canonical(body))
def atomic(path,body):return publish(path,normalized(body))

def path_of(path,subtree=''):
    path=Path(path).resolve();base=(FOLDER/subtree).resolve()
    need(path==base or path.is_relative_to(base),'Trace-speed output namespace differs')
    return path

def baseline():
    path=FOLDER/'baseline.json'
    if path.exists():
        body=read(path)
        need(body['schema']=='tracing-speed-baseline-v1' and body['passed'] is True and
             body['inputs_count']==len(body['inputs'])==62200,'Trace-speed baseline differs')
    else:
        prior=read(PRIOR)
        need(prior['passed'] is True and prior['models_fixed'] is True and prior['policy_fixed'] is True and
             prior['training_counters']==ZERO and len(prior['inputs'])==62199,'Prior accepted scope differs')
        inputs=dict(prior['inputs']);inputs.update(common.pins([PRIOR]))
        body=dict(schema='tracing-speed-baseline-v1',passed=True,inputs_count=len(inputs),inputs=inputs)
    common.current(body['inputs']);return atomic(path,body)

if __name__=='__main__':
    result=baseline();print(json.dumps(dict(passed=True,preserved_files=result['inputs_count'])),flush=True)
