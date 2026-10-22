"""Deterministic value diagnostics on sealed cached fit features.

No feature encoding, native operation, search or Torch is used. Values use
scalar-double arithmetic over captured native-double features/FP32 weights; native feature and
backend conformance is a separate required check. Completed verification
re-derives these numerical projections from the pinned source rows/exports.
"""
from collections import Counter
import math
from pathlib import Path
from . import config,ordinary_epoch_common as stage,ordinary_epoch_sidecar as side
from . import ordinary_epoch_training as training,ordinary_epoch_data as data
from .valuepilot_reservations import Writer

ROOT=stage.ROOT
DEFAULT=stage.FOLDER/'fit'
SCHEMA='ordinary-epoch-fit-v1'
CANDIDATE=training.DEFAULT/'model.json'
SOURCES=('congklak/ordinary_epoch_fit.py','congklak/ordinary_epoch_common.py',
 'congklak/ordinary_epoch_data.py','congklak/ordinary_epoch_sidecar.py',
 'congklak/ordinary_epoch_training.py','congklak/ordinary_epoch_execute.py','congklak/config.py')
need,read,sha=stage.need,stage.read,stage.sha

def _same(a,b):return config.canonical(a)==config.canonical(b)

def _dot(a,b):
    # Native scalar and AVX2 kernels both accumulate inputs0,1,... separately.
    # Python3.12's built-in float sum may compensate, so use explicit additions.
    value=0.
    for weight,x in zip(a,b):value+=weight*x
    return value

def _base(x,weights):
    h=x
    for prefix in ('layers.0','layers.2'):
        h=[max(0,_dot(row,h)+bias)
           for row,bias in zip(weights[prefix+'.weight'],weights[prefix+'.bias'])]
    u=_dot(weights['layers.4.weight'][0],h)+weights['layers.4.bias'][0]
    policy=[_dot(row,h)+bias
            for row,bias in zip(weights['policy.weight'],weights['policy.bias'])]
    return u,policy

def _correction(x,weights):
    h=[max(0,_dot(row,x)+bias) for row,bias in zip(weights['A'],weights['b'])]
    return _dot(weights['v'][0],h)+weights['c'][0]

def _huber(error):
    a=abs(error)
    return .5*error*error if a<=.1 else .1*(a-.05)

def _metrics(rows,name):
    if not rows:return dict(count=0)
    z=[r[name]['z']-r['target_z'] for r in rows];seeds=[r[name]['value_player0']-r['target_player0'] for r in rows]
    return dict(count=len(rows),normalized_huber=sum(_huber(e) for e in z)/len(z),
        normalized_mae=sum(abs(e) for e in z)/len(z),normalized_rmse=math.sqrt(sum(e*e for e in z)/len(z)),
        normalized_bias=sum(z)/len(z),seed_mae=sum(abs(e) for e in seeds)/len(seeds),
        seed_rmse=math.sqrt(sum(e*e for e in seeds)/len(seeds)),seed_bias_player0=sum(seeds)/len(seeds))

def _validate_rows(rows,receipt,manifest):
    selected=[r for r in stage.joint()['conditional_role_revalidation'] if r['role']=='fit']
    lookup={r['state_key']:r for r in selected}
    cases={r['state_key'] for r in stage.joint()['conditional_role_revalidation'] if r['role']=='cases'}
    need(type(rows) is list and len(rows)==len(lookup)==128 and len({r['state_key'] for r in rows})==128 and
         {r['state_key'] for r in rows}==set(lookup) and len(cases)==32 and
         set(manifest['case_state_keys'])==cases,'Sealed fit/case membership differs')
    for row in rows:
        m=lookup[row['state_key']];state=config.position(row['state']);actor=state['controller'];active=sum(map(sum,state['pits']))
        need(row['state_key']==stage.common.rawkey(state) and row['family']==m['family'] and row['cell']==m['cell'] and
             row['component']==row['joint_component']==m['joint_component'] and m['known_recorded_disjoint'] is True and
             type(row['is_case']) is bool and row['is_case']==(row['state_key'] in cases) and row['actor']==actor and
             row['D']==state['stores'][0]-state['stores'][1] and row['r']==active*(1 if actor==0 else -1) and
             25<=active<=74 and len(row['x'])==127 and
             all(type(v) is float and math.isfinite(v) and data.fp32(v)==v for v in row['x']) and
             len(row['x_native'])==127 and all(type(v) in (float,int) and math.isfinite(v) for v in row['x_native']) and
             row['x']==[data.fp32(v) for v in row['x_native']] and
             type(row['target_player0']) is int and -1<=row['target_z']<=1 and
             row['target_z']==(row['target_player0']-row['D'])/row['r'],
             'Sealed fit features/normalization/ownership differ')
        teacher=row['teacher'];legal=[i for i,n in enumerate(state['pits'][actor]) if n]
        q={r['pit']:r['value_player0'] for r in teacher['action_q']}
        need(type(teacher['requested_depth']) is int and teacher['requested_depth']==8 and
             type(teacher['completed_depth']) is int and teacher['completed_depth']==8 and
             teacher['all_legal_Q_complete'] is True and set(q)==set(legal) and len(q)==len(teacher['action_q']) and
             all(type(r['pit']) is int for r in teacher['action_q']) and
             all(type(v) is int for v in q.values()),'Fit teacher mixes capped/incomplete targets')
        y=(max if actor==0 else min)(q.values())
        need(y==row['target_player0']==teacher['target_player0'] and
             teacher['best_moves']==[a for a in legal if q[a]==y],'Fit teacher target/action Q disagree')
    need(sum(r['is_case'] for r in rows)==receipt['fit_cases']==32,'Fit case count differs')

def _derive():
    stage.admit_plan()
    from . import ordinary_epoch_execute as execute
    admission=execute.admit_labels();closed=execute.completed_labels();manifest=closed['fit_manifest'];rows=execute.load_fit()
    result=training.completed();need(result['steps']==400 and result['engineering_fixture'] is False,'Fit requires final400 candidate')
    _validate_rows(rows,admission,manifest)
    control=read(side.BASE);candidate=side.verify_export(CANDIDATE)
    need(_same(candidate['model']['base_export'],control),'Fit candidate changed frozen trunk/policy')
    control_weights=control['model']['weights'];candidate_weights=candidate['model']['base_export']['model']['weights']
    branch=candidate['model']['branch_weights'];predictions=[]
    for row in rows:
        x=row['x_native'];u,policy=_base(x,control_weights);cu,cp=_base(x,candidate_weights)
        need(policy==cp and u==cu,'Cached policy/base arithmetic differs')
        z=math.tanh(u);cz=math.tanh(cu+_correction(x,branch))
        predictions.append(dict(state_key=row['state_key'],family=row['family'],cell=row['cell'],actor=row['actor'],
            active_seeds=abs(row['r']),is_case=row['is_case'],target_player0=row['target_player0'],target_z=row['target_z'],
            control=dict(z=z,value_player0=row['D']+row['r']*z),
            candidate=dict(z=cz,value_player0=row['D']+row['r']*cz),policy_logits_exact=True,
            source_row_sha256=config.digest(row)))
    groups=dict(overall=predictions,cases=[r for r in predictions if r['is_case']])
    groups.update({m:[r for r in predictions if '/'+m+'/' in r['cell']] for m in ('ordinary','capture','bonus')})
    groups.update({'actor'+str(a):[r for r in predictions if r['actor']==a] for a in (0,1)})
    groups.update({'band'+b:[r for r in predictions if (r['active_seeds']<50)==(b=='25-49')] for b in ('25-49','50-74')})
    groups.update({c:[r for r in predictions if r['cell']==c] for c in sorted({r['cell'] for r in predictions})})
    metrics={g:{m:_metrics(rs,m) for m in ('control','candidate')} for g,rs in groups.items()}
    overall=metrics['overall'];capture=metrics['capture']
    checks=dict(normalized_huber_improved=overall['candidate']['normalized_huber']<overall['control']['normalized_huber']-1e-12,
        overall_seed_mae_nonregressed=overall['candidate']['seed_mae']<=overall['control']['seed_mae']+1e-9,
        capture_seed_mae_nonregressed=capture['candidate']['seed_mae']<=capture['control']['seed_mae']+1e-9,
        policy_logits_exact=all(r['policy_logits_exact'] for r in predictions),
        target_kind_preserved=True,fit_case_membership_preserved=True)
    files=[ROOT/p for p in SOURCES]+[stage.PLAN,stage.PRIOR,stage.JOINT,stage.EPOCH,side.BASE,CANDIDATE]
    files.extend(training.DEFAULT/n for n in ('identity.json','catalog.json','certificate.json','result.json'))
    files.extend(ROOT/admission[k]['path'] for k in ('catalog_ref','ownership_prefix_ref','fit_manifest_ref'))
    files.append(ROOT/manifest['rows_ref']['path'])
    return dict(schema=SCHEMA+'-report',passed=all(checks.values()),checks=checks,metrics=metrics,rows=predictions,
        input_pins=stage.common.pins(files),fit_admission=admission,
        target_kind='complete-H8/H_v1-untrained-oraclefree/full-legal-root-Q/player0-margin',
        target_normalization='(y-D)/signed_active; no clipping; endpoints retained',
        numeric_backend='Python scalar-double/input-order separate multiply/add on captured native-double127features and exportedFP32weights',
        primary_feature_precision='native captured double',training_feature_precision='existing FP32 cache unchanged',
        native_conformance_separate=True,policy_proof=side.policy_proof(),
        cached_feature_model_calculations=256,new_position_queries=0,new_encodes=0,new_searches=0,
        native_model_forwards=0,optimizer_steps=0,policy_fixed=True,model_promotion=False,
        inherited_unknown_background=True,historically_unseen=False)

def completed(output=DEFAULT):
    path=stage.path_of(output);certificate=read(path/'certificate.json');report=read(path/'report.json')
    need(_same(certificate,dict(schema=SCHEMA+'-certificate',report_sha256=sha(path/'report.json'),
        source_inputs=report['input_pins'])),'Fit report closure differs')
    need(_same(report,_derive()),'Fit report differs from sealed rows/exports')
    return report

def run(output=DEFAULT):
    path=stage.path_of(output)
    if (path/'certificate.json').exists():return completed(path)
    report=_derive();path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'certificate.json').exists():return completed(path)
        if (path/'report.json').exists():need(_same(read(path/'report.json'),report),'Uncommitted fit report differs')
        else:stage.atomic(path/'report.json',report)
        stage.atomic(path/'certificate.json',dict(schema=SCHEMA+'-certificate',report_sha256=sha(path/'report.json'),
            source_inputs=report['input_pins']))
    return completed(path)
