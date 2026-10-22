"""Independent audit helpers for witnessed ownership and live shared suffixes.

Historical evidence and original exports are immutable. The facts operation
imports no production builder, native engine, Torch or training module.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'build/ownership-implementation-v1'
OUT=FOLDER/'audit'
CONTROL='data/quality-train-v1/model.json'
CANDIDATE='data/value-training-v1/training/model.json'
CONTROL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
CANDIDATE_SHA='c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319'


def need(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(body):return json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def publish(name,body):
    path=OUT/name;content=canonical(body)+'\n';path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.exists(),'Independent evidence path already exists: '+str(path))
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(content)
    return body


def active(state):
    need(set(state)=={'pits','stores','controller'} and len(state['pits'])==2 and
         all(len(row)==7 for row in state['pits']) and len(state['stores'])==2 and
         type(state['controller']) is int and state['controller'] in (0,1),'State geometry differs')
    values=state['pits'][0]+state['pits'][1]+state['stores']
    need(all(type(value) is int and value>=0 for value in values) and sum(values)==98,'State seed conservation differs')
    return sum(map(sum,state['pits']))


def quotient(state):
    a=active(state);cells=state['pits'][0]+state['pits'][1];actor=state['controller'] if a else 0
    return min(bytes(cells+[actor]),bytes(cells[7:]+cells[:7]+[1-actor if a else 0])).hex()


def reference():
    spec=importlib.util.spec_from_file_location('ownership_implementation_independent_reference',ROOT/'tests/reference.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def literal_transition(state,action):
    ref=reference();active(state)
    ring=tuple(state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1]])
    move=ref.apply(ref.Position(7,ring,state['controller']),action)
    result=dict(pits=[list(move.position.counts[:7]),list(move.position.counts[8:15])],
                stores=[move.position.counts[7],move.position.counts[15]],controller=move.position.controller)
    before,after=active(state),active(result)
    need(after<=before and before-after==sum(result['stores'])-sum(state['stores']) and
         all(v>=u for u,v in zip(state['stores'],result['stores'])),'Nonabsorbing complete transition')
    return dict(after=result,terminal=move.terminal,captured=move.captured,bonus=move.bonus,
                relay_pickups=move.relay_pickups,deposits=move.deposits,
                initial_passes=[list(p) for p in move.initial_passes],
                final_passes=[list(p) for p in move.final_passes],actor=move.actor)


def facts():
    path=ROOT/'build/ownership-protocol-v1/acceptance.json';accepted=read(path)
    need(accepted['schema']=='prospective-ownership-acceptance-v3' and accepted['passed'] is True and
         accepted['candidate_quarantined'] is True and accepted['training_authorized'] is False and
         len(accepted['inputs'])==59153,'Prior protocol acceptance differs')
    for name,expected in accepted['inputs'].items():need(sha(ROOT/name)==expected,'Previous accepted file changed: '+name)
    need(sha(ROOT/CONTROL)==CONTROL_SHA and sha(ROOT/CANDIDATE)==CANDIDATE_SHA,'Fixed model changed')
    control,candidate=read(ROOT/CONTROL),read(ROOT/CANDIDATE)
    need(candidate['model']['base_export']==control,'Original model or policy differs')
    need('torch' not in sys.modules,'Independent facts imported Torch')
    inputs={name:sha(ROOT/name) for name in (CONTROL,CANDIDATE,'tests/ownership_implementation_independent_audit.py',
        'build/ownership-protocol-v1/acceptance.json','tests/reference.py')}
    result=dict(schema='ownership-implementation-independent-facts-v1',passed=True,
        previous_inputs_verified=59153,previous_inputs_preserved_including_acceptance=59154,
        policy_fixed=True,models_fixed=True,embedded_base_export_exact=True,
        candidate_quarantined=True,inputs=inputs,
        engineering_counters=dict(model_forwards=0,searches=0,new_games=0,literal_replays=0),
        training_counters=dict(optimizer_steps=0,optimizer_draws=0,new_teacher_labels=0),torch_imported=False)
    publish('facts.json',result)
    print(canonical(dict(passed=True,prior_files=59154,models_fixed=True,policy_fixed=True)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['facts']);args=parser.parse_args()
    facts()
