"""Independent, existing-data audit for fixed-model value/ownership diagnostics.

The facts path deliberately imports no production diagnostics, engine, Torch or
feature extractor. Dense arithmetic uses saved vectors and exported JSON only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build/value-diagnostics-v1/audit'
CONTROL = ROOT / 'data/quality-train-v1/model.json'
CANDIDATE = ROOT / 'data/value-training-v1/training/model.json'
EXPECTED_CONTROL = 'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
EXPECTED_CANDIDATE = 'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(name, value):
    path = OUT / name
    text = canonical(value) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Immutable audit evidence differs')
    else:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    return value


def close(a, b, label, tol=1e-10):
    require(abs(a-b) <= tol, label + ': ' + repr((a, b)))


def prior_closure():
    path = ROOT / 'build/value-run-v1/acceptance.json'
    body = read(path)
    require(body['passed'] and len(body['inputs']) == 40540, 'Prior closure incomplete')
    for name, expected in body['inputs'].items():
        require(sha(ROOT/name) == expected, 'Prior immutable input changed: '+name)
    return dict(previous_inputs_verified=40540, previous_acceptance_sha256=sha(path),
                preserved_including_acceptance=40541)


def quotient(state):
    c = state['pits'][0] + state['pits'][1]
    require(len(c) == 14 and sum(c+state['stores']) == 98, 'State geometry/total differs')
    actor = state['controller'] if any(c) else 0
    return min(bytes(c+[actor]), bytes(c[7:]+c[:7]+[1-actor if any(c) else 0])).hex()


def dense(weights, prefix, vector, relu=False):
    out = [sum(a*x for a,x in zip(w,vector))+b for w,b in
           zip(weights[prefix+'.weight'], weights[prefix+'.bias'])]
    return [max(0.0,x) for x in out] if relu else out


def predictions(row, base, branch):
    x = row['x']
    require(len(x) == 127, 'Cached feature dimension differs')
    h = dense(base, 'layers.0', x, True)
    h = dense(base, 'layers.2', h, True)
    u = dense(base, 'layers.4', h)[0]
    correction = sum(a*max(0.0,sum(w*v for w,v in zip(weights,x))+b)
                     for a,weights,b in zip(branch['v'][0],branch['A'],branch['b'])) + branch['c'][0]
    require(25 <= abs(row['r']) <= 74, 'Training band differs')
    return dict(control=row['D']+row['r']*math.tanh(u),
                candidate=row['D']+row['r']*math.tanh(u+correction),
                logit_correction=correction)


def metrics(rows):
    result = {}
    for name in ('control', 'candidate'):
        errors = [(r[name]-r['target_player0']) for r in rows]
        zerrors = [e/r['r'] for e,r in zip(errors,rows)]
        n = len(rows)
        result[name] = dict(roots=n, seed_mae=sum(abs(e) for e in errors)/n,
            seed_rmse=math.sqrt(sum(e*e for e in errors)/n), seed_bias=sum(errors)/n,
            normalized_huber=sum(.5*e*e if abs(e)<=.1 else .1*(abs(e)-.05) for e in zerrors)/n)
    return result


def capture_facts():
    require(sha(CONTROL) == EXPECTED_CONTROL and sha(CANDIDATE) == EXPECTED_CANDIDATE, 'Frozen model changed')
    base = read(CONTROL)
    sidecar = read(CANDIDATE)['model']
    require(sidecar['base_export'] == base, 'Base/policy export differs')
    weights = base['model']['weights']
    saved = read(ROOT/'build/value-run-v1/cases-fit-v2.json')
    saved_rows = {r['state_key']:r for r in saved['rows']}
    all_rows = {}; saved_prediction_maximum=0.0
    for role,filename in [('train','train.json'),('fit_check','fit-check.json')]:
        source = read(ROOT/'data/value-training-v1/cache'/filename)
        rows=[]
        for row in source['rows']:
            require(row['reserved_role'] == role, 'Reserved role differs')
            a=sum(map(sum,row['state']['pits']))
            require(row['D']==row['state']['stores'][0]-row['state']['stores'][1], 'Offset differs')
            require(row['r']==(a if row['actor']==0 else -a), 'Residual perspective differs')
            close(row['target_z'],(row['target_player0']-row['D'])/row['r'],'Target normalization')
            q=[v['value_player0'] for v in row['target_source']['action_q']]
            require(row['target_player0'] == (max(q) if row['actor']==0 else min(q)), 'H8 root target differs')
            pred=predictions(row,weights,sidecar['branch_weights'])
            output={k:row[k] for k in ('state_key','actor','cell','stratum','target_player0','r')}
            output.update(pred)
            rows.append(output)
            if role=='fit_check':
                for name in ('control','candidate'):
                    old=saved_rows[row['state_key']]['predictions'][name]['value_player0']
                    saved_prediction_maximum=max(saved_prediction_maximum,abs(pred[name]-old))
                    close(pred[name],old,'Cached-FP32 vs fresh-F64 feature prediction',tol=2e-5)
        all_rows[role]=rows
    summaries={}
    for role,rows in all_rows.items():
        groups={'all':rows, 'capture':[r for r in rows if r['stratum'].endswith('/capture')]}
        groups.update({cell:[r for r in rows if r['cell']==cell] for cell in sorted({r['cell'] for r in rows})})
        summaries[role]={label:metrics(rs) for label,rs in groups.items()}
    for name in ('control','candidate'):
        for group in ('all','capture'):
            for key,actual in summaries['fit_check'][group][name].items():
                close(actual,saved['metrics'][name][group][key],'Independent '+group+'/'+name+'/'+key,tol=2e-5)
    captures=[r for r in all_rows['fit_check'] if r['stratum'].endswith('/capture')]
    for row in captures:
        row['absolute_error_delta']=abs(row['candidate']-row['target_player0'])-abs(row['control']-row['target_player0'])
        row['value_shift']=row['candidate']-row['control']
    return dict(rows=640,fit_rows=128,train_rows=512,base_and_policy_exact=True,
        summaries=summaries,capture_fit_rows=captures,
        capture_worse_count=sum(r['absolute_error_delta']>0 for r in captures),
        capture_improved_count=sum(r['absolute_error_delta']<0 for r in captures),
        no_new_features_or_targets=True,cached_fp32_vs_saved_fresh_f64_maximum_seed_error=saved_prediction_maximum)


def load_reference():
    spec=importlib.util.spec_from_file_location('value_diagnostics_audit_reference',ROOT/'tests/reference.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def arena_facts():
    path=ROOT/'data/value-run-v1/arena'
    identity=read(path/'identity.json'); head=read(path/'head.json'); saved=read(path/'result.json')
    require(head['identity_sha256']==sha(path/'identity.json'),'Arena identity digest differs')
    decisions=[]; games=[]
    for key,target in [('decisions',decisions),('games',games)]:
        for entry in head[key]:
            require(sha(path/entry['name'])==entry['sha256'],'Raw arena entry changed')
            target.append(read(path/entry['name']))
    require(len(games)==192 and len(decisions)==6663,'Arena cardinality differs')
    inv=read(ROOT/'data/neural-value-pilot-v2/reservations/inventory.json')
    selection=read(ROOT/'data/neural-value-pilot-v2/reservations/selection.json')
    qcomponents=defaultdict(set);qroles=defaultdict(set);froles=defaultdict(set)
    for cid,component in inv['components'].items():
        for q in component['quotients']:qcomponents[q].add(cid)
    for role in ('train','fit_check'):
        for row in selection[role]:
            component=inv['components'][row['component']]
            for q in component['quotients']:qroles[q].add(role)
            for f in component['families']:froles[f].add(role)
    qowners=defaultdict(set);cowners=defaultdict(set);by_game=defaultdict(list)
    arena_components={s['family_index']:s['component'] for s in identity['schedule']}
    for fi,cid in arena_components.items():cowners[cid].add(fi)
    for d in decisions:by_game[d['game_index']].append(d)
    role_hits=[];observed=[];ancestral_hits=[];ref=load_reference()
    for game in games:
        gi=game['index'];schedule=identity['schedule'][gi];ds=by_game[gi];state=schedule['start']
        require(len(ds)==game['choices'],'Raw game decision cardinality differs')
        require([digest(d) for d in ds]==game['decision_sha256'],'Raw game decision hash chain differs')
        states=[state]
        for choice,d in enumerate(ds):
            require(d['before']==state and d['choice_index']==choice,'Broken literal trajectory')
            ring=tuple(state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1]])
            t=ref.apply(ref.Position(7,ring,state['controller']),d['action'])
            state=dict(pits=[list(t.position.counts[:7]),list(t.position.counts[8:15])],
                       stores=[t.position.counts[7],t.position.counts[15]],controller=t.position.controller)
            require(state==d['transition']['after'],'Literal transition differs')
            require(t.deposits==d['transition']['deposits'] and t.captured==d['transition']['captured'] and
                    t.relay_pickups==d['transition']['relay_pickups'],'Literal mechanisms differ')
            states.append(state)
        require(state==game['final'] and game['terminal'] and not game['capped'] and not game['fallbacks'],'Game not exact terminal')
        for choice,state in enumerate(states):
            active=sum(map(sum,state['pits']))
            if active<=4:continue
            q=quotient(state); fi=schedule['family_index'];cid=schedule['component']; owners=qcomponents[q]
            row=dict(game=gi,choice=choice,family_index=fi,component=cid,quotient=q)
            observed.append(dict(row,state=state,active=active,profile=schedule['profile'],baseline=schedule['baseline']))
            qowners[q].add(fi)
            for owner in owners:cowners[owner].add(fi)
            roles=qroles[q]|froles[schedule['family']]
            if roles:role_hits.append(dict(row,roles=sorted(roles)))
            for other,othercid in arena_components.items():
                if other!=fi and othercid in owners:qowners[q].add(other)
            if owners-{cid}:ancestral_hits.append(dict(row,ancestral_components=sorted(owners-{cid})))
    cross=[dict(quotient=q,family_indices=sorted(f)) for q,f in sorted(qowners.items()) if len(f)>1]
    bridges=[dict(historical_component=cid,family_indices=sorted(f)) for cid,f in sorted(cowners.items()) if len(f)>1]
    own=saved['ownership']
    require(role_hits==own['support_or_fit_hits'],'Support/fit ownership witnesses differ')
    require(cross==own['cross_arena_component_hits'],'Direct ownership bridges differ')
    require(bridges==own['cross_arena_alias_component_hits'],'Historical ownership bridges differ')
    require(ancestral_hits==own['broader_ancestral_hits'],'Historical observation witnesses differ')
    require(len(observed)==own['observations'] and len(qowners)==own['distinct_observed_quotients'],'Ownership denominator differs')
    require(not own['ownership_passed'] and not saved['cluster_analysis_valid'] and not saved['cluster_statistics'] and
            not saved['family_cluster_rows'],'Invalid arena strength interpretation was emitted')
    supportq={r['quotient'] for r in role_hits};crossq={r['quotient'] for r in cross}
    return dict(games=len(games),decisions=len(decisions),literal_transitions=len(decisions),eligible_observations=len(observed),
        support_hits=role_hits,direct_bridges=cross,historical_bridges=bridges,
        support_state_witnesses=[r for r in observed if r['quotient'] in supportq],
        direct_observation_active_counts=dict((str(a),sum(r['active']==a for r in observed if r['quotient'] in crossq))
                                            for a in sorted({r['active'] for r in observed if r['quotient'] in crossq})),
        ownership_hold_preserved=True,statistics_withheld=True)


def facts():
    previous=prior_closure()
    capture=capture_facts()
    arena=arena_facts()
    require('torch' not in sys.modules,'Independent existing-data path imported Torch')
    files=[Path(__file__), ROOT/'tests/reference.py',CONTROL,CANDIDATE,
           ROOT/'build/value-run-v1/cases-fit-v2.json', ROOT/'data/value-run-v1/arena/head.json',
           ROOT/'data/value-run-v1/arena/result.json', ROOT/'data/neural-value-pilot-v2/reservations/inventory.json',
           ROOT/'data/neural-value-pilot-v2/reservations/selection.json',
           ROOT/'data/value-training-v1/cache/train.json',ROOT/'data/value-training-v1/cache/fit-check.json']
    result=dict(schema='value-diagnostics-independent-facts-v1',passed=True,
                prior=previous,capture=capture,arena=arena,
                inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files},
                new_optimizer_steps=0,new_teacher_labels=0,new_games=0,policy_fixed=True,models_fixed=True,
                production_searches=0,feature_extractions=0,literal_reference_replays=6663,
                independent_dense_model_evaluations=1280,torch_imported=False)
    publish('facts.json',result)
    print(canonical(dict(passed=True,capture_fit_mae={n:capture['summaries']['fit_check']['capture'][n]['seed_mae'] for n in ('control','candidate')},
                         support_hits=len(arena['support_hits']),direct_bridges=len(arena['direct_bridges']),historical_bridges=len(arena['historical_bridges']))))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['facts'])
    args=parser.parse_args()
    if args.command=='facts':facts()
