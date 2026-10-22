"""Independent, durable per-simulation reconstruction of the frozen PUCT rule.

The accepted engine is never modified or used for selection/expansion/backup.
Committed predictions restore a tree without inference. All primary cells must
match the corresponding already-sealed engine semantics at their exact budget.
"""
from __future__ import annotations
from contextlib import ExitStack,closing
from dataclasses import dataclass,field
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import uuid

from . import config,allocation_common as common,phase6_store,quality_search
from .native import ROOT,Native,Control,flatten
from .neural_relay import Model
from .ordinary_value_benchmark import verify_candidate,host_peak
from .phase6_inference import InferenceStopped,validate_prediction
from .puct_cpu_inference import CpuBatchEvaluator,COUNTERS,DLL
from .timing_v2 import PerformanceCore

DEFAULT=ROOT/'data/allocation-trace-v1'
PLAN=ROOT/'build/allocation-trace-plan.json'
IMPLEMENTATION=ROOT/'build/allocation-trace-implementation.json'
OUTPUT=ROOT/'build/allocation-trace-result.json'
RESOURCES=ROOT/'build/allocation-trace-resources.json'
CLI=ROOT/'tools/trace-allocation.py'
SCHEMA='allocation-independent-trace-v1'
PROFILES=('control','candidate')
MAX_STEP_BYTES=128*1024
RUN_QUOTA_BYTES=1024**3
TABLES={'meta','roots','startup','root_setups','steps','cells','interruptions'}
require=common.require
same=common.same


def _integer(x):return type(x) is int and x>=0
def _hashed(text,digest):
    body=json.loads(text);require(config.digest(body)==digest,'Corrupt allocation row');return body
def _ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
def _meta(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone();return _hashed(*row) if row else None
def _poll(cancel):
    stopped=bool(cancel() if callable(cancel) else cancel.is_set()) if cancel is not None else False
    if stopped:raise InferenceStopped('cancelled')
def _path(path,fixture):
    path=Path(path).resolve();require(path.is_relative_to((ROOT/'build').resolve()) if fixture else path==DEFAULT.resolve(),'Fixture paths must be build-only; production uses its declared directory');return path
def _exports(path):return (path/'manifest.json',OUTPUT,RESOURCES) if path==DEFAULT.resolve() else (path/'manifest.json',path/'result.json',path/'resources.json')


@dataclass
class Node:
    id:int
    state:dict
    actor:int
    legal:tuple
    terminal:bool
    bootstrap:float
    priors:list=field(default_factory=lambda:[0.0]*7)
    visits:list=field(default_factory=lambda:[0]*7)
    sums:list=field(default_factory=lambda:[0.0]*7)
    children:dict=field(default_factory=dict)


@dataclass
class Tree:
    nodes:dict
    completed:int=0
    work:dict=field(default_factory=lambda:dict(transitions=0,deposits=0,feature_simulations=0,feature_deposits=0,neural_evaluations=0,neural_calls=0,root_expansions=0))
    counts:dict=field(default_factory=lambda:{name:0 for name in COUNTERS})


def _node(native,p,index):
    normal=native.normalize(flatten(p),actor=p['controller'],total=98);require(normal['ok'],'Cannot normalize trace node');p=normal['position']
    return Node(index,p,p['controller'],tuple(normal['legal_actions']),normal['terminal'],(p['stores'][0]-p['stores'][1])/98)


def _expand(node,prediction):
    validate_prediction(prediction,node.state);require(not node.terminal,'Terminal nodes must not infer');node.bootstrap=prediction['value_player0']/98
    logits=prediction['policy_logits'];top=max(logits[a] for a in node.legal);values={a:math.exp(logits[a]-top) for a in node.legal};total=sum(values.values())
    node.priors=[values.get(a,0.0)/total for a in range(7)]


def selection_details(node):
    total=sum(node.visits);sign=1 if node.actor==0 else -1;rows=[]
    for pit in node.legal:
        observed=node.sums[pit]/node.visits[pit] if node.visits[pit] else None;q=observed if observed is not None else node.bootstrap
        # Preserve the frozen multiplication/division order in normalized units.
        exploration=.25*node.priors[pit]*math.sqrt(max(1,total))/(1+node.visits[pit]);signed=sign*q;score=signed+exploration
        rows.append(dict(pit=pit,visits=node.visits[pit],prior=node.priors[pit],observed_Q_player0_normalized=observed,parent_bootstrap_player0_normalized=node.bootstrap,
            Q_source='visited_mean' if observed is not None else 'parent_bootstrap_FPU',selected_Q_player0_normalized=q,actor_signed_Q_normalized=signed,exploration_normalized=exploration,total_score_normalized=score,
            total_score_seed_display=score*98))
    chosen=max(rows,key=lambda row:(row['total_score_normalized'],-row['pit']))['pit']
    return dict(actor=node.actor,total_visits=total,rows=rows,chosen=chosen,tie_break='Maximum normalized score, then lowest pit; seed display is never used for selection')


def _leader(node):
    count=sum(node.visits);return max(node.legal,key=lambda a:(node.visits[a],node.priors[a] if not count else 0,-a))


def _packed(node):
    return dict(id=node.id,state=node.state,actor=node.actor,legal=list(node.legal),terminal=node.terminal,bootstrap_normalized=node.bootstrap,priors=node.priors)


def _charge(tree,prediction,counts,root=False):
    for key in ('feature_simulations','feature_deposits'):tree.work[key]+=prediction[key]
    tree.work['neural_evaluations']+=1;tree.work['neural_calls']+=1;tree.work['root_expansions']+=int(root)
    for name in COUNTERS:tree.counts[name]+=counts[name]


def _counter_delta(before,after):return {name:after[name]-before[name] for name in COUNTERS}


def validate_prediction_work(prediction,counts,state,reset):
    require(isinstance(counts,dict) and set(counts)==set(COUNTERS) and all(_integer(x) for x in counts.values()),'Physical counter delta changed')
    if prediction is None:
        require(not any(counts.values()),'Terminal/nonprediction step claims a forward');return
    validate_prediction(prediction,state)
    for key in ('positions_submitted','evaluations_requested','native_actual_evals','predict_successes','native_calls','cache_misses'):require(type(counts[key]) is int and counts[key]==1,'A trace prediction must be exactly one physical uncached forward')
    require(counts['cache_hits']==counts['cache_evictions']==counts['terminal_returns']==0 and counts['cache_resets']==reset,'Prediction cache/reset differs')
    require(1<=counts['feature_simulations']<=14 and counts['feature_deposits']>=counts['feature_simulations'],'Prediction lacks single-pass feature work')
    require(same(prediction['feature_simulations'],counts['feature_simulations']) and same(prediction['feature_deposits'],counts['feature_deposits']) and prediction['cache_hit'] is False and type(prediction['native_actual_evals']) is int and prediction['native_actual_evals']==1,'Prediction physical result differs')
    require(type(prediction.get('inference_ms')) in (int,float) and math.isfinite(prediction['inference_ms']) and prediction['inference_ms']>=0,'Invalid observed inference clock')


def expected_plan():
    common.baseline();panel=common.read(ROOT/'build/divergence-panel.json');root=next(row for row in panel['roots'] if row['pair']==7)
    require(root['actor']==1 and root['state_key']=='090a03000405010409010a010400061f01','Pair7 selection changed')
    old=common.read(ROOT/'build/divergence-search-plan.json');source=common.read(ROOT/'build/divergence-search-result.json');references={}
    with closing(_ro(ROOT/'data/divergence-search-v1/work.sqlite3')) as db:old_startup=_hashed(*db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone())
    for cell in source['cells']:
        if cell['unit']['state_key']!=root['state_key']:continue
        report=copy.deepcopy(cell['report']);report.pop('timing');report['work'].pop('inference_ms');references[cell['unit']['profile']+'/'+str(cell['unit']['simulations'])]=report
    require(len(references)==8,'Missing sealed pair7 budget reference')
    fixtures=[];native=Native()
    for index,item in enumerate(common.read(ROOT/'build/value-pilot-rows.json')['groups']['diagnostic'][:2]):
        n=_node(native,item['state'],0);require(not n.terminal and same(n.state,item['state']),'Fixture normalization differs')
        fixtures.append(dict(id='consumed-fixture-'+str(index),state=n.state,state_key=common.rawkey(n.state),actor=n.actor,source_row_sha256=config.digest(item),role='old_consumed_engineering_diagnostic'))
    return dict(schema=SCHEMA+'-plan',roots=[root],profiles=list(PROFILES),budgets=[128,256,512,1024],cells=8,total_root_simulations=3840,maximum_completed_forwards=3848,
        models=common.pins([common.CONTROL,common.CANDIDATE]),source_inputs=common.pins([common.BASELINE,common.SCOPE,ROOT/'build/divergence-panel.json',ROOT/'build/divergence-search-plan.json',ROOT/'build/divergence-search-result.json',ROOT/'data/divergence-search-v1/work.sqlite3',ROOT/'build/divergence-comparison.json',ROOT/'build/value-pilot-rows.json']),
        references=references,c_puct=.25,max_nodes=2048,time_ms=0,root_noise=None,rng=0,cache_capacity=0,max_batch=1,memory_mb=8192,cpu_declared=old['cpu_declared'],cpu_metadata=old['cpu_metadata'],backend=old['expected_backend'],evaluator_identities=old_startup['evaluators'],
        work_bound_scope='Maxima cover uninterrupted committed production only. Retries, lost uncommitted work, legal verification replay, fixtures and audits are additional.',
        max_step_bytes=MAX_STEP_BYTES,run_quota_bytes=RUN_QUOTA_BYTES,min_free_mb=2048,storage='Atomic FULL/WAL simulation commits; nonempty WAL remains durable; SHM and absent/empty WAL are ephemeral',
        fixture=dict(roots=fixtures,budgets=[8,16],variants=dict(normal=dict(max_nodes=2048),node_cap=dict(max_nodes=2))),
        algorithm='Independent normalized-float scoring/expansion/fixed-player0 backups; no frozen helper calls; no transpositions or noise',
        trace_semantic_volatile_fields=['root_setup.observation.elapsed_ms','root_setup.prediction.inference_ms','step.observation.elapsed_ms','step.prediction.inference_ms','interrupt.observation.elapsed_ms'],
        training_authorized=False,parameter_tuning=False,promotion=False)


def prepare():return common.publish(PLAN,expected_plan())


def implementation_body():
    body=dict(schema=SCHEMA+'-implementation',sources=common.pins(['congklak/allocation_trace.py','tools/trace-allocation.py','tests/allocation_trace_author.py']),
        science=common.pins([common.BASELINE,common.SCOPE,PLAN,ROOT/'congklak/allocation_common.py']),
        dependencies=common.pins(['congklak/quality_search.py','congklak/puct_cpu_inference.py','congklak/phase6_store.py','congklak/timing_v2.py','congklak/native.py','congklak/config.py','congklak/neural_relay.py','congklak/features_relay.py','congklak/phase6_inference.py','congklak/ordinary_value_benchmark.py','build/puct-cpu-build.json','build/congklak_puct_cpu.dll']),rules_digest=config.RULES_DIGEST)
    return {**body,'sha256':config.digest(body)}


def freeze():prepare();return common.publish(IMPLEMENTATION,implementation_body())


def admit(fixture=False,variant='normal',verify_prior=False):
    require(type(fixture) is bool and variant in ('normal','node_cap') and (fixture or variant=='normal'),'Unknown allocation configuration');require('torch' not in sys.modules,'Torch is forbidden')
    common.baseline(verify_all=verify_prior);plan=common.read(PLAN);require(same(plan,expected_plan()),'Allocation plan changed');implementation=implementation_body();require(same(common.read(IMPLEMENTATION),implementation),'Allocation freeze changed')
    models={name:Model(path) for name,path in (('control',common.CONTROL),('candidate',common.CANDIDATE))};contract=verify_candidate(models['control'],models['candidate'])
    roots=plan['fixture']['roots'] if fixture else plan['roots'];budgets=plan['fixture']['budgets'] if fixture else plan['budgets'];cap=plan['fixture']['variants'][variant]['max_nodes'] if fixture else plan['max_nodes']
    units=[dict(index=i,root_index=ri,state_key=root['state_key'],profile=name,simulations=n,max_nodes=cap) for i,(ri,root,n,name) in enumerate((ri,r,n,name) for ri,r in enumerate(roots) for n in budgets for name in PROFILES)]
    identity=dict(schema=SCHEMA,fixture=fixture,variant=variant,plan_sha256=common.sha(PLAN),implementation_file_sha256=common.sha(IMPLEMENTATION),implementation_sha256=implementation['sha256'],baseline_sha256=common.sha(common.BASELINE),
        roots_sha256=config.digest(roots),units_sha256=config.digest(units),unit_count=len(units),model_files={n:m.file_sha256 for n,m in models.items()},model_weights={n:m.id for n,m in models.items()},policy_contract=contract,
        cpu_metadata=plan['cpu_metadata'],cpu_declared=plan['cpu_declared'],backend=plan['backend'],evaluator_identities=plan['evaluator_identities'],max_nodes=cap,c_puct=.25,memory_mb=8192,time_ms=0,root_noise=None,rng=0,cache_capacity=0,max_batch=1,rules_digest=config.RULES_DIGEST)
    return plan,roots,units,identity


def _validate_startup(body,identity):
    require(set(body)=={'cpu','evaluators'} and same(body['cpu'],identity['cpu_metadata']) and same(body['evaluators'],identity['evaluator_identities']),'Runtime startup identity differs')
    for name,evaluator in body['evaluators'].items():
        require(evaluator['model_file_sha256']==identity['model_files'][name] and evaluator['model_weights_sha256']==identity['model_weights'][name] and evaluator['backend']==identity['backend'] and evaluator['requested_backend']=='auto','Actual model/backend changed')
        require(same(evaluator['max_batch'],1) and same(evaluator['cache']['capacity'],0) and evaluator['dll_sha256']==common.sha(DLL) and evaluator['build_record_sha256']==common.sha(ROOT/'build/puct-cpu-build.json'),'CPU adapter configuration changed')
        common.current(evaluator['dependencies']);common.current(evaluator['build_inputs'])


def _root_setup(body,unit,root,identity,startup,native):
    fields={'schema','cell','unit','root_sha256','prediction','physical_counts','root_node','evaluator_identity_sha256','cpu_verified','observation'}
    require(set(body)==fields and body['schema']==SCHEMA+'-root-setup' and type(body['cell']) is int and body['cell']==unit['index'] and same(body['unit'],unit) and body['root_sha256']==config.digest(root),'Root expansion linkage differs')
    node=_node(native,root['state'],0);require(not node.terminal,'Primary/engineering roots are nonterminal');validate_prediction_work(body['prediction'],body['physical_counts'],node.state,1);_expand(node,body['prediction'])
    require(same(body['root_node'],_packed(node)) and body['evaluator_identity_sha256']==config.digest(startup['evaluators'][unit['profile']]),'Root expansion/model identity differs')
    _validate_observation(body['observation']);_validate_cpu(body['cpu_verified'],identity)
    tree=Tree({0:node});_charge(tree,body['prediction'],body['physical_counts'],root=True);return tree


def _validate_cpu(body,identity):require(same(body,dict(group=0,logical=identity['cpu_declared']['logical'],mask=1<<identity['cpu_declared']['logical'])),'Search left declared CPU')
def _validate_observation(body):require(isinstance(body,dict) and set(body)=={'elapsed_ms'} and type(body['elapsed_ms']) in (int,float) and math.isfinite(body['elapsed_ms']) and body['elapsed_ms']>=0,'Invalid diagnostic observation')
def _transition(t):return {key:t[key] for key in ('position','reason','store_delta','passes','relays','drops','terminal')}


def _backup(tree,path,value):
    for nid,actor,pit in path:tree.nodes[nid].visits[pit]+=1;tree.nodes[nid].sums[pit]+=value
    tree.completed+=1


def _root_after(tree):
    node=tree.nodes[0];return dict(visits=list(node.visits),sums_normalized=list(node.sums),visit_leader=_leader(node))


def replay_step(tree,body,unit,identity,native=None):
    """Recompute every internal selection and backup; never evaluate a model."""
    native=native or Native();fields={'schema','cell','simulation','root_selection','path','new_edge','prediction','physical_counts','leaf','root_after','cpu_verified','observation'}
    require(set(body)==fields and body['schema']==SCHEMA+'-step' and type(body['cell']) is int and body['cell']==unit['index'] and type(body['simulation']) is int and body['simulation']==tree.completed+1,'Simulation identity differs')
    require(len(config.canonical(body).encode())<=MAX_STEP_BYTES and tree.completed<unit['simulations'],'Step storage/budget exceeded');root=tree.nodes[0];require(same(body['root_selection'],selection_details(root)),'Root scoring/FPU changed')
    path=body['path'];require(isinstance(path,list) and 0<len(path)<unit['max_nodes'] and all(isinstance(p,list) and len(p)==3 and all(type(x) is int and x>=0 for x in p) for p in path),'Malformed compressed path')
    node=root;decisions=[];new=body['new_edge']
    for depth,(nid,actor,pit) in enumerate(path):
        require(not node.terminal and nid==node.id and actor==node.actor,'Path controller/node identity differs');details=selection_details(node);require(details['chosen']==pit,'Internal selection/tie-break changed')
        decisions.append(dict(depth=depth,state_key=common.rawkey(node.state),state=node.state,node_id=node.id,selection=details))
        if pit not in node.children:
            require(depth==len(path)-1 and len(tree.nodes)<unit['max_nodes'] and isinstance(new,dict) and set(new)=={'parent','pit','child','transition','node'},'Missing/extra new edge')
            require(type(new['parent']) is type(new['pit']) is type(new['child']) is int and new['parent']==nid and new['pit']==pit and new['child']==len(tree.nodes),'New node identity differs')
            t=native.apply(flatten(node.state),actor=node.actor,action=pit,total=98,trace=False);require(t['ok'] and same(_transition(t),new['transition']),'Recorded legal transition differs')
            child=_node(native,t['position'],len(tree.nodes));validate_prediction_work(body['prediction'],body['physical_counts'],child.state,0)
            if child.terminal:require(body['prediction'] is None,'Terminal leaf has a prediction')
            else:require(body['prediction'] is not None,'Nonterminal leaf lacks prediction');_expand(child,body['prediction'])
            require(same(new['node'],_packed(child)),'Leaf state/bootstrap/priors changed');node.children[pit]=child.id;tree.nodes[child.id]=child;tree.work['transitions']+=1;tree.work['deposits']+=t['drops'];node=child
        else:
            node=tree.nodes[node.children[pit]]
            require(depth<len(path)-1 or node.terminal,'Path stopped before a new/terminal leaf')
    if new is None:
        require(node.terminal and body['prediction'] is None,'Existing leaf must be terminal');validate_prediction_work(None,body['physical_counts'],node.state,0)
    elif body['prediction'] is not None:_charge(tree,body['prediction'],body['physical_counts'])
    raw_value=node.state['stores'][0]-node.state['stores'][1] if node.terminal else body['prediction']['value_player0']
    expected=dict(node_id=node.id,state_key=common.rawkey(node.state),state=node.state,controller=node.actor,terminal=node.terminal,source='terminal_store_margin' if node.terminal else 'network',raw_value_player0=raw_value,backup_normalized=node.bootstrap,backup_player0_display=node.bootstrap*98)
    require(same(body['leaf'],expected),'Leaf/controller/backup provenance differs');_validate_cpu(body['cpu_verified'],identity);_validate_observation(body['observation'])
    _backup(tree,path,node.bootstrap);require(same(body['root_after'],_root_after(tree)),'Fixed-player0 backup or root leader differs');return decisions


def semantic_report(tree,unit,stop='simulations'):
    root=tree.nodes[0];count=sum(root.visits);policy=[v/count for v in root.visits] if count else list(root.priors);estimate=sum(root.sums)/count if count else root.bootstrap
    return dict(schema=quality_search.SCHEMA,protocol=quality_search.PROTOCOL,rules_digest=config.RULES_DIGEST,position=root.state,chosen=_leader(root),visits=list(root.visits),policy=policy,root_priors=list(root.priors),estimated_value_player0=estimate*98,
        action_mean_value_player0=[root.sums[a]/root.visits[a]*98 if root.visits[a] else None for a in range(7)],root_expanded=True,completed_simulations=tree.completed,requested_simulations=unit['simulations'],nodes=len(tree.nodes),max_nodes=unit['max_nodes'],usable_decision=bool(tree.completed),memory_mb=8192,
        stop_reason=stop,proof_status='monte_carlo_estimate',work={**tree.work,'total_deposits':tree.work['deposits']+tree.work['feature_deposits']},forest_batch_calls=tree.work['neural_calls'],forest_size=1,value_perspective='fixed_player0_seeds',first_play_urgency='parent_neural_bootstrap',unvisited_root_q_player0=root.bootstrap*98,c_puct=.25)


def semantic_trace(stores):
    """Only clocks are volatile. Predictions, all deltas and selections remain."""
    body=copy.deepcopy(stores)
    for row in body['root_setups'].values():
        row.pop('observation');row['prediction'].pop('inference_ms')
    for rows in body['steps'].values():
        for row in rows:
            row.pop('observation')
            if row['prediction'] is not None:row['prediction'].pop('inference_ms')
    for row in body['interruptions']:row.pop('observation')
    return body


def replay_prefix(db,roots,units,identity):
    require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok' and db.execute('PRAGMA user_version').fetchone()[0]==6 and {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}==TABLES,'Unknown/damaged trace journal')
    require(same(_meta(db,'identity'),identity) and {r[0] for r in db.execute('SELECT key FROM meta')}<={'identity','allocation_trace_complete'},'Trace identity/metadata changed')
    saved={key:_hashed(text,digest) for key,text,digest in db.execute('SELECT state,body,hash FROM roots')};require(same(saved,{r['state_key']:r for r in roots}),'Trace roots changed')
    startup_rows=db.execute('SELECT id,body,hash FROM startup').fetchall();require(len(startup_rows)<=1 and (not startup_rows or startup_rows[0][0]==0),'Extra startup identity')
    startup=_hashed(*startup_rows[0][1:]) if startup_rows else None
    if startup:_validate_startup(startup,identity)
    stores=dict(startup=startup,root_setups={},steps={},cells={},interruptions=[]);trees={};cells=[];native=Native()
    for idx,text,digest in db.execute('SELECT idx,body,hash FROM cells ORDER BY idx'):
        require(type(idx) is int and idx==len(cells) and idx<len(units),'Noncontiguous cell completion');cells.append(_hashed(text,digest))
    for idx,text,digest in db.execute('SELECT cell,body,hash FROM root_setups ORDER BY cell'):
        require(type(idx) is int and idx==len(stores['root_setups']) and idx<=len(cells) and idx<len(units) and startup is not None,'Orphan root expansion')
        setup=_hashed(text,digest);unit=units[idx];tree=_root_setup(setup,unit,roots[unit['root_index']],identity,startup,native);stores['root_setups'][idx]=setup;rows=[]
        for seq,body_text,body_sha in db.execute('SELECT seq,body,hash FROM steps WHERE cell=? ORDER BY seq',(idx,)):
            require(type(seq) is int and seq==len(rows),'Noncontiguous simulation prefix');row=_hashed(body_text,body_sha);replay_step(tree,row,unit,identity,native);rows.append(row)
        stores['steps'][idx]=rows;trees[idx]=tree
        if idx<len(cells):
            cell=cells[idx];require(same(cell,_cell_result(tree,unit,identity)),'Completed cell/report/parity changed');stores['cells'][idx]=cell
    require(set(stores['cells'])==set(range(len(cells))) and all(idx in stores['root_setups'] for idx, in db.execute('SELECT DISTINCT cell FROM steps')),'Missing/orphan steps or cells')
    for idx,text,digest in db.execute('SELECT idx,body,hash FROM interruptions ORDER BY idx'):
        body=_hashed(text,digest);require(type(idx) is int and idx==len(stores['interruptions']) and set(body)=={'index','cell','completed_prefix','reason','phase','physical_counts','uncommitted_native_transition_work','observation'} and type(body['index']) is int and body['index']==idx and type(body['cell']) is int and 0<=body['cell']<len(units),'Invalid interruption provenance')
        require(type(body['completed_prefix']) is int and 0<=body['completed_prefix']<=len(stores['steps'].get(body['cell'],[])) and body['reason'] in ('cancelled','memory_budget') and set(body['physical_counts'])==set(COUNTERS) and all(_integer(v) for v in body['physical_counts'].values()),'Invalid interrupted work')
        require(body['phase'] in ('root_setup','simulation') and (same(body['uncommitted_native_transition_work'],dict(transitions=0,deposits=0)) if body['phase']=='root_setup' else body['uncommitted_native_transition_work'] is None),'Interrupted transition work cannot be invented')
        require(body['completed_prefix']==0 if body['phase']=='root_setup' else body['cell'] in stores['root_setups'],'Invalid pre-root or simulation cancellation prefix')
        _validate_observation(body['observation']);stores['interruptions'].append(body)
    return stores,trees


def _cell_result(tree,unit,identity):
    complete=tree.completed==unit['simulations'];stop='simulations' if complete else 'node_budget';report=semantic_report(tree,unit,stop)
    require(complete or len(tree.nodes)==unit['max_nodes'],'Incomplete cell has no declared cap')
    expected=None if identity['fixture'] else common.read(PLAN)['references'][unit['profile']+'/'+str(unit['simulations'])]
    if complete and expected is not None:require(same(report,expected),'Instrumented tree differs from sealed engine semantics')
    return dict(schema=SCHEMA+'-cell',index=unit['index'],unit=unit,status='complete' if complete else 'node_budget',semantic_report=report,semantic_report_sha256=config.digest(report),physical_counts=tree.counts,
        sealed_reference_sha256=config.digest(expected) if expected is not None else None,sealed_reference_matches=same(report,expected) if complete and expected is not None else None)


def _storage(path,estimate=MAX_STEP_BYTES*3):
    require(sum(p.stat().st_size for p in path.rglob('*') if p.is_file())+estimate<=RUN_QUOTA_BYTES,'Trace journal reached fixed1GiB storage bound')
    require(shutil.disk_usage(path).free>=estimate+2048*1024**2,'Insufficient trace plus2GiB recovery headroom')


def _advance(tree,evaluator,native,unit,identity,cpu,cancel):
    started=time.perf_counter();_poll(cancel);root_selection=selection_details(tree.nodes[0]);node=tree.nodes[0];path=[];edge=None;prediction=None;counts={name:0 for name in COUNTERS}
    while not node.terminal:
        _poll(cancel);details=selection_details(node);pit=details['chosen'];path.append([node.id,node.actor,pit])
        if pit not in node.children:
            if len(tree.nodes)>=unit['max_nodes']:return None
            control=Control(native,max_depth=128,time_ms=0,node_budget=0,memory_mb=8192,tt_mb=0,use_pvs=False)
            registered=hasattr(cancel,'_register') and hasattr(cancel,'_remove')
            if registered:cancel._register(control)
            try:t=native.apply(flatten(node.state),actor=node.actor,action=pit,total=98,trace=False,control=control)
            finally:
                if registered:cancel._remove(control)
                control.close()
            if not t['ok']:
                if t.get('error',{}).get('code')=='cancelled':raise InferenceStopped('cancelled')
                raise ValueError('Illegal trace transition')
            _poll(cancel);child=_node(native,t['position'],len(tree.nodes))
            if not child.terminal:
                before=evaluator.snapshot_counters();prediction=evaluator.evaluate_many([child.state],cancel=cancel)[0];counts=_counter_delta(before,evaluator.snapshot_counters());validate_prediction_work(prediction,counts,child.state,0);_expand(child,prediction)
            edge=dict(parent=node.id,pit=pit,child=child.id,transition=_transition(t),node=_packed(child));node.children[pit]=child.id;tree.nodes[child.id]=child;tree.work['transitions']+=1;tree.work['deposits']+=t['drops'];node=child;break
        node=tree.nodes[node.children[pit]]
    _poll(cancel)
    if prediction is not None:_charge(tree,prediction,counts)
    leaf=dict(node_id=node.id,state_key=common.rawkey(node.state),state=node.state,controller=node.actor,terminal=node.terminal,source='terminal_store_margin' if node.terminal else 'network',raw_value_player0=node.state['stores'][0]-node.state['stores'][1] if node.terminal else prediction['value_player0'],backup_normalized=node.bootstrap,backup_player0_display=node.bootstrap*98)
    _backup(tree,path,node.bootstrap)
    return dict(schema=SCHEMA+'-step',cell=unit['index'],simulation=tree.completed,root_selection=root_selection,path=path,new_edge=edge,prediction=prediction,physical_counts=counts,leaf=leaf,root_after=_root_after(tree),cpu_verified=cpu.verify(),observation=dict(elapsed_ms=(time.perf_counter()-started)*1000))


def _compare_selection_paths(stores,units,roots,identity):
    """Paired fresh-tree replay finds first differences at a common raw state."""
    results=[];native=Native()
    for ri,root in enumerate(roots):
        budgets=sorted({u['simulations'] for u in units if u['root_index']==ri})
        for budget in budgets:
            pair={u['profile']:u for u in units if u['root_index']==ri and u['simulations']==budget}
            if not all(u['index'] in stores['cells'] for u in pair.values()):continue
            trees={n:_root_setup(stores['root_setups'][u['index']],u,root,identity,stores['startup'],native) for n,u in pair.items()}
            count=min(len(stores['steps'][u['index']]) for u in pair.values());first_root=None;first_internal=None;matched_leaves=0;different_leaves=0;switches={n:[] for n in PROFILES};leaders={n:_leader(t.nodes[0]) for n,t in trees.items()};all_visited={n:None for n in PROFILES};fpu_used={n:0 for n in PROFILES}
            for i in range(count):
                rows={n:stores['steps'][u['index']][i] for n,u in pair.items()}
                traces={n:replay_step(trees[n],rows[n],pair[n],identity,native) for n in PROFILES}
                if rows['control']['leaf']['state_key']==rows['candidate']['leaf']['state_key']:matched_leaves+=1
                else:different_leaves+=1
                for n in PROFILES:
                    leader=rows[n]['root_after']['visit_leader']
                    if leader!=leaders[n]:switches[n].append(dict(simulation=i+1,from_pit=leaders[n],to_pit=leader,visits=rows[n]['root_after']['visits']))
                    leaders[n]=leader
                    table=rows[n]['root_selection'];chosen=next(x for x in table['rows'] if x['pit']==table['chosen']);fpu_used[n]+=int(chosen['Q_source']=='parent_bootstrap_FPU')
                    if all_visited[n] is None and all(rows[n]['root_after']['visits'][a]>0 for a in trees[n].nodes[0].legal):all_visited[n]=i+1
                a=rows['control']['root_selection'];b=rows['candidate']['root_selection']
                if first_root is None and a['chosen']!=b['chosen']:first_root=dict(simulation=i+1,state_key=root['state_key'],control=a,candidate=b)
                if first_internal is None:
                    for left,right in zip(traces['control'],traces['candidate']):
                        if left['state_key']!=right['state_key']:break
                        if left['selection']['chosen']!=right['selection']['chosen']:
                            if left['depth']>0:first_internal=dict(simulation=i+1,depth=left['depth'],state=left['state'],state_key=left['state_key'],control=left['selection'],candidate=right['selection'])
                            break
            sampled={}
            for n,t in trees.items():
                node=t.nodes[0];observed={a:node.sums[a]/node.visits[a] for a in node.legal if node.visits[a]};sign=1 if node.actor==0 else -1;top=max(sign*q for q in observed.values()) if observed else None
                best=[a for a,q in observed.items() if sign*q==top];sampled[n]=dict(visit_winner=_leader(node),actor_best_among_sampled_means=best,unvisited_legal_actions=[a for a in node.legal if not node.visits[a]],vote_winner_in_best_sampled_mean=(_leader(node) in best))
            results.append(dict(root_id=root['id'],state_key=root['state_key'],simulations=budget,eligible_models=2,completed_models=sum(stores['cells'][u['index']]['status']=='complete' for u in pair.values()),
                first_root_allocation_difference=first_root,first_internal_difference_on_shared_path=first_internal,matching_leaf_states_at_same_simulation=matched_leaves,different_leaf_states_at_same_simulation=different_leaves,visit_leader_switches=switches,
                first_all_root_actions_visited_after_simulation=all_visited,root_selected_FPU_uses=fpu_used,final_vote_vs_sampled_mean=sampled,
                FPU_limit='After every root action has a visit, FPU is no longer directly used at the root; it remains active on unvisited descendant edges. Allocation is not attributed to root FPU alone.',
                control_final_chosen=stores['cells'][pair['control']['index']]['semantic_report']['chosen'],candidate_final_chosen=stores['cells'][pair['candidate']['index']]['semantic_report']['chosen']))
    return results


def summarize(roots,units,identity,stores):
    cells=[stores['cells'][i] for i in sorted(stores['cells'])];complete=sum(c['status']=='complete' for c in cells);steps=sum(len(rows) for rows in stores['steps'].values())
    counts={name:sum(row['physical_counts'][name] for row in stores['root_setups'].values())+sum(row['physical_counts'][name] for rows in stores['steps'].values() for row in rows) for name in COUNTERS}
    return dict(schema=SCHEMA+'-result',identity=identity,status='complete' if complete==len(units) else 'budget_exhausted',scope_finished=complete==len(units),roots=roots,cells=cells,
        counts=dict(eligible_cells=len(units),finalized_cells=len(cells),complete_cells=complete,incomplete_cells=len(cells)-complete,root_setups=len(stores['root_setups']),requested_root_simulations=sum(u['simulations'] for u in units),committed_root_simulations=steps,interruptions=len(stores['interruptions']),physical_committed_counts=counts),
        allocation=_compare_selection_paths(stores,units,roots,identity),semantic_trace_sha256=config.digest(semantic_trace(stores)),raw_stores_sha256=config.digest(stores),
        actual_vs_committed_work='Counters describe committed root expansions/simulations. Interrupted inference counters are separate, not total work; uncommitted simulation transition/deposit totals are unavailable. Killed uncommitted work may be recomputed and is additional. Validation replays legal transitions without inference; fixtures/audits are separate.',
        units='Selection and sums use fixed-player0 seed margin/98; actor sign only in selection. Seed displays multiply normalized scores by98 and never select. Unvisited observed Q remains None; FPU is separate.',
        limits=['Eight fresh production trees; every declared budget retained. All are the same consumed pair7 player1 diagnostic, not independent population evidence.',
            'Per-simulation FULL/WAL trace overhead precludes speed/energy/strength comparisons. H8/H10/H12 reference preferences remain horizon dependent.',
            'No model/policy change, parameter tuning, optimizer, strength game or automatic promotion. Previous HOLD and H_v1 default remain fixed.'],training_authorized=False,torch_imported=False)


def _validate_resources(body,identity,result,stores):
    fields={'schema','identity_sha256','result_sha256','elapsed_seconds_through_prepublication_validation','initial_committed_simulations','final_committed_simulations','initial_root_setups','final_root_setups','host_peak','torch_imported','energy_measured','scope'}
    require(set(body)==fields and body['schema']==SCHEMA+'-resources' and body['identity_sha256']==config.digest(identity) and body['result_sha256']==config.digest(result),'Resource identity differs')
    require(type(body['elapsed_seconds_through_prepublication_validation']) in (int,float) and math.isfinite(body['elapsed_seconds_through_prepublication_validation']) and body['elapsed_seconds_through_prepublication_validation']>=0,'Invalid trace clock')
    for key in ('initial_committed_simulations','final_committed_simulations','initial_root_setups','final_root_setups'):require(_integer(body[key]),'Invalid resource prefix')
    require(body['initial_committed_simulations']<=body['final_committed_simulations']==sum(len(x) for x in stores['steps'].values()) and body['initial_root_setups']<=body['final_root_setups']==len(stores['root_setups']),'Resource committed work differs')
    require(body['torch_imported'] is False and body['energy_measured'] is False and body['scope']=='This finalizing invocation through validation; excludes subsequent completion/export, prior/lost uncommitted work, fixtures and audits. Includes diagnostic persistence/replay; not an inference benchmark','Resource claim differs')
    require(isinstance(body['host_peak'],dict) and set(body['host_peak'])=={'peak_working_set_bytes','peak_private_commit_bytes'} and all(type(x) is int and x>0 for x in body['host_peak'].values()),'Missing host peak')


def _closed(path,fixture=False,variant='normal',require_exports=False):
    path=_path(path,fixture);plan,roots,units,identity=admit(fixture,variant);require((path/'work.sqlite3').is_file(),'Missing trace journal')
    with closing(_ro(path/'work.sqlite3')) as db:
        stores,trees=replay_prefix(db,roots,units,identity);marker=_meta(db,'allocation_trace_complete')
        require(marker is not None and len(stores['cells'])==len(units),'Trace is not closed');body=summarize(roots,units,identity,stores)
        require(set(marker)=={'schema','identity_sha256','stores_sha256','result_sha256','resource_path','resource_file_sha256'} and marker['schema']==SCHEMA+'-complete' and marker['identity_sha256']==config.digest(identity) and marker['stores_sha256']==config.digest(stores) and marker['result_sha256']==config.digest(body),'Invalid complete trace marker')
    relative=Path(marker['resource_path']);resource_path=(path/relative).resolve();require(not relative.is_absolute() and resource_path.is_relative_to(path) and resource_path.is_file() and common.sha(resource_path)==marker['resource_file_sha256'],'Missing/corrupt resource observation')
    resource=common.read(resource_path);_validate_resources(resource,identity,body,stores)
    for file,expected in zip(_exports(path),(body,body,resource)):
        if file.exists():require(file.read_text(encoding='utf-8')==config.canonical(expected)+'\n','Existing trace export differs')
        else:require(not require_exports,'Missing trace export')
    return roots,stores,body,resource


def emit(event):
    print(config.canonical(event),flush=True)
    if os.environ.get('ALLOCATION_TRACE_BLOCK_EVENT')!=event['event']:return
    after=int(os.environ.get('ALLOCATION_TRACE_BLOCK_AFTER','-1'))
    if after>=0 and event.get('completed')!=after:return
    marker=os.environ.get('ALLOCATION_TRACE_BLOCK_MARKER');require(marker,'Missing trace kill marker');common.publish(Path(marker),event)
    while True:time.sleep(.05)


def run(path=DEFAULT,fixture=False,variant='normal',simulation_limit=None,on_event=None,cancel=None):
    started=time.perf_counter();path=_path(path,fixture);plan,roots,units,identity=admit(fixture,variant,verify_prior=not fixture);on_event=on_event or emit
    require(simulation_limit is None or _integer(simulation_limit),'Invalid invocation simulation limit')
    if (path/'work.sqlite3').exists():
        with closing(_ro(path/'work.sqlite3')) as db:
            marker=_meta(db,'allocation_trace_complete');tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables==TABLES:replay_prefix(db,roots,units,identity)
            else:require(tables=={'meta'} and same(_meta(db,'identity'),identity) and marker is None,'Invalid incomplete initialization')
        if marker is not None:
            _,_,body,resource=_closed(path,fixture,variant)
            for file,value in zip(_exports(path),(body,body,resource)):common.publish(file,value)
            common.baseline(verify_all=not fixture);return body
    with ExitStack() as stack:
        work=phase6_store.Work(path,identity);stack.callback(work.close);db=work.db
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS startup(id INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS root_setups(cell INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS steps(cell INTEGER,seq INTEGER,body TEXT NOT NULL,hash TEXT NOT NULL,PRIMARY KEY(cell,seq))')
            db.execute('CREATE TABLE IF NOT EXISTS cells(idx INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS interruptions(idx INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            if db.execute('SELECT count(*) FROM roots').fetchone()[0]==0:
                for root in roots:db.execute('INSERT INTO roots VALUES(?,?,?)',(root['state_key'],config.canonical(root),config.digest(root)))
        stores,trees=replay_prefix(db,roots,units,identity);initial_steps=sum(len(x) for x in stores['steps'].values());initial_setups=len(stores['root_setups']);new_steps=0;cpu=None;evaluators=None;native=Native();paused=False
        def runtime():
            nonlocal cpu,evaluators
            if evaluators is not None:return
            cpu=stack.enter_context(PerformanceCore(plan['cpu_declared']));evaluators={n:stack.enter_context(CpuBatchEvaluator(p,cache_capacity=0,max_batch=1,backend='auto')) for n,p in (('control',common.CONTROL),('candidate',common.CANDIDATE))}
            actual=dict(cpu=cpu.metadata,evaluators={n:e.identity for n,e in evaluators.items()});_validate_startup(actual,identity)
            if stores['startup'] is not None:require(same(actual,stores['startup']),'Resumed backend identity differs')
            else:
                with db:db.execute('INSERT INTO startup VALUES(0,?,?)',(config.canonical(actual),config.digest(actual)))
                stores['startup']=actual
        def interrupted(unit,before,reason,elapsed,phase):
            counts={name:0 for name in COUNTERS}
            if evaluators is not None and before is not None:counts=_counter_delta(before,evaluators[unit['profile']].snapshot_counters())
            body=dict(index=len(stores['interruptions']),cell=unit['index'],completed_prefix=len(stores['steps'].get(unit['index'],[])),reason=reason,phase=phase,physical_counts=counts,uncommitted_native_transition_work=dict(transitions=0,deposits=0) if phase=='root_setup' else None,observation=dict(elapsed_ms=elapsed))
            with db:db.execute('INSERT INTO interruptions VALUES(?,?,?)',(body['index'],config.canonical(body),config.digest(body)))
            stores['interruptions'].append(body)
        on_event(dict(event='allocation_trace_started',fixture=fixture,completed=initial_steps,target=sum(u['simulations'] for u in units),finalized_cells=len(stores['cells']),complete_cells=sum(c['status']=='complete' for c in stores['cells'].values())))
        for index in range(len(stores['cells']),len(units)):
            unit=units[index];root=roots[unit['root_index']]
            if simulation_limit is not None and new_steps>=simulation_limit:paused=True;break
            if index not in stores['root_setups']:
                runtime();e=evaluators[unit['profile']];before=e.snapshot_counters();tick=time.perf_counter()
                try:
                    _poll(cancel);e.reset_cache();node=_node(native,root['state'],0);prediction=e.evaluate_many([node.state],cancel=cancel)[0];_poll(cancel);counts=_counter_delta(before,e.snapshot_counters());validate_prediction_work(prediction,counts,node.state,1);_expand(node,prediction)
                except (InferenceStopped,MemoryError) as exc:
                    interrupted(unit,before,getattr(exc,'reason','memory_budget'),(time.perf_counter()-tick)*1000,'root_setup');paused=True;break
                setup=dict(schema=SCHEMA+'-root-setup',cell=index,unit=unit,root_sha256=config.digest(root),prediction=prediction,physical_counts=counts,root_node=_packed(node),evaluator_identity_sha256=config.digest(e.identity),cpu_verified=cpu.verify(),observation=dict(elapsed_ms=(time.perf_counter()-tick)*1000))
                tree=_root_setup(setup,unit,root,identity,stores['startup'],native);_storage(path);work.admit(MAX_STEP_BYTES)
                with db:db.execute('INSERT INTO root_setups VALUES(?,?,?)',(index,config.canonical(setup),config.digest(setup)))
                stores['root_setups'][index]=setup;stores['steps'][index]=[];trees[index]=tree
                on_event(dict(event='allocation_trace_root_committed',cell=index,profile=unit['profile'],simulations=unit['simulations'],completed=initial_steps+new_steps))
            tree=trees[index]
            # Independent shadow replay checks each generated record before commit.
            shadow=_root_setup(stores['root_setups'][index],unit,root,identity,stores['startup'],native)
            for row in stores['steps'][index]:replay_step(shadow,row,unit,identity,native)
            while tree.completed<unit['simulations']:
                if simulation_limit is not None and new_steps>=simulation_limit:paused=True;break
                runtime();e=evaluators[unit['profile']];before=e.snapshot_counters();tick=time.perf_counter()
                try:row=_advance(tree,e,native,unit,identity,cpu,cancel)
                except (InferenceStopped,MemoryError) as exc:
                    interrupted(unit,before,getattr(exc,'reason','memory_budget'),(time.perf_counter()-tick)*1000,'simulation');trees[index]=shadow;tree=shadow;paused=True;break
                if row is None:break
                replay_step(shadow,row,unit,identity,native);encoded=config.canonical(row);require(len(encoded.encode())<=MAX_STEP_BYTES,'Trace step exceeds fixed byte cap');_storage(path)
                completed=initial_steps+new_steps
                on_event(dict(event='allocation_trace_simulation_precommit',cell=index,simulation=row['simulation'],completed=completed,target=sum(u['simulations'] for u in units),profile=unit['profile']))
                with db:db.execute('INSERT INTO steps VALUES(?,?,?,?)',(index,row['simulation']-1,encoded,config.digest(row)))
                stores['steps'][index].append(row);new_steps+=1
                on_event(dict(event='allocation_trace_simulation_committed',cell=index,simulation=row['simulation'],completed=completed+1,target=sum(u['simulations'] for u in units),profile=unit['profile']))
            if paused:break
            cell=_cell_result(tree,unit,identity)
            with db:db.execute('INSERT INTO cells VALUES(?,?,?)',(index,config.canonical(cell),config.digest(cell)))
            stores['cells'][index]=cell
            on_event(dict(event='allocation_trace_cell_committed',cell=index,finalized_cells=len(stores['cells']),complete_cells=sum(c['status']=='complete' for c in stores['cells'].values()),completed=initial_steps+new_steps,target=sum(u['simulations'] for u in units),status=cell['status']))
        if paused:return dict(schema=SCHEMA+'-progress',status='interrupted',finalized_cells=len(stores['cells']),complete_cells=sum(c['status']=='complete' for c in stores['cells'].values()),committed_simulations=initial_steps+new_steps,root_setups=len(stores['root_setups']),interruptions=len(stores['interruptions']))
        stores,trees=replay_prefix(db,roots,units,identity);require(len(stores['cells'])==len(units),'Missing final cells');body=summarize(roots,units,identity,stores);common.baseline(verify_all=not fixture)
        resource=dict(schema=SCHEMA+'-resources',identity_sha256=config.digest(identity),result_sha256=config.digest(body),elapsed_seconds_through_prepublication_validation=time.perf_counter()-started,initial_committed_simulations=initial_steps,final_committed_simulations=sum(len(x) for x in stores['steps'].values()),initial_root_setups=initial_setups,final_root_setups=len(stores['root_setups']),host_peak=host_peak(),torch_imported='torch' in sys.modules,energy_measured=False,
            scope='This finalizing invocation through validation; excludes subsequent completion/export, prior/lost uncommitted work, fixtures and audits. Includes diagnostic persistence/replay; not an inference benchmark')
        _validate_resources(resource,identity,body,stores);relative='resource-observations/'+uuid.uuid4().hex+'.json';common.publish(path/relative,resource)
        marker=dict(schema=SCHEMA+'-complete',identity_sha256=config.digest(identity),stores_sha256=config.digest(stores),result_sha256=config.digest(body),resource_path=relative,resource_file_sha256=common.sha(path/relative))
        with db:work.put('allocation_trace_complete',marker)
    _,_,body,resource=_closed(path,fixture,variant)
    for file,value in zip(_exports(path),(body,body,resource)):common.publish(file,value)
    on_event(dict(event='allocation_trace_complete',fixture=fixture,**body['counts']));return body


def verify(path=DEFAULT,fixture=False,variant='normal'):return _closed(path,fixture,variant,require_exports=True)[2]


def load_trace(path=DEFAULT,fixture=False,variant='normal'):
    roots,stores,body,_=_closed(path,fixture,variant,require_exports=True);return roots,stores,body


def primary_witness():
    body=verify();require(body['scope_finished'] is True and body['identity']['fixture'] is False and body['counts']['complete_cells']==8 and body['counts']['committed_root_simulations']==3840 and all(c['sealed_reference_matches'] is True for c in body['cells']),'Production trace gate incomplete')
    return dict(schema=SCHEMA+'-primary-witness',passed=True,fixture=False,complete_cells=8,completed_root_simulations=3840,plan_file_sha256=common.sha(PLAN),implementation_file_sha256=common.sha(IMPLEMENTATION),
        journal_file_sha256=common.sha(DEFAULT/'work.sqlite3'),manifest_file_sha256=common.sha(DEFAULT/'manifest.json'),result_file_sha256=common.sha(OUTPUT),resource_file_sha256=common.sha(RESOURCES),semantic_trace_sha256=body['semantic_trace_sha256'],all_sealed_reports_match=True,model_files=body['identity']['model_files'],model_weights=body['identity']['model_weights'],torch_imported_by_gate=False)
