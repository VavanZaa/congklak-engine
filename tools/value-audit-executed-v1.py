"""Independent frozen-policy value-head acceptance coordinator."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import closing
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from value_audit_acceptance import ValueIndependentAcceptance,EVIDENCE,WORK,sha,load,save,module,normalize_literal,json_position,literal,semantic

GROUPS={
    'plan':['test_01_prior_1647_inputs_unchanged','test_02_predeclared_value_only_plan'],
    'targets':['test_03_training_filter_and_seed_target_derivation'],
    'models':['test_04_global_frozen_policy_structure_and_native_literal_outputs'],
    'contract':['test_05_frozen_checkpoint_and_two_parameter_optimizer_refusals'],
    'training':['test_06_actual_training_pre_postcommit_kills_and_optimizer_rng_recovery'],
    'analysis':['test_07_actual_analysis_pre_postcommit_kills_and_rehashed_corruption'],
    'case':['test_08_actual_case_kill_prefix_and_rehashed_regret_refusal'],
    'arena':['test_09_actual_arena_kill_192_literal_replays_and_completed_stability'],
    'deadline':['test_10_expired_deadline_and_physical_counter_admission'],
}
SOURCES=['tests/value_audit_acceptance.py','tools/value-audit.py']

def require(value,message):
    if not value:raise AssertionError(message)
def input_map(paths):return {Path(p).relative_to(ROOT).as_posix():sha(p) for p in paths}

def group(name):
    EVIDENCE.clear();started=time.monotonic();methods=GROUPS[name]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(ValueIndependentAcceptance(n) for n in methods))
    paths=[ROOT/p for p in SOURCES]+[ROOT/'build/quality-acceptance.json']
    dependency_names=['congklak/value_learning.py','tools/value-train.py','build/value-registry.json','build/value-data.json','build/training-runtime.json'] if name=='training' else ['build/value-plan.json','build/value-implementation.json','build/value-registry.json','build/value-data.json','congklak/value_common.py','congklak/quality_search.py','congklak/value_learning.py','congklak/value_analysis.py','tools/benchmark-value.py']
    for p in dependency_names:
        if (ROOT/p).is_file():paths.append(ROOT/p)
    if name!='training':
        for p in ('data/value-train-v1/model.json','data/quality-train-v1/model.json','build/value-model-witness.json'):
            if (ROOT/p).is_file():paths.append(ROOT/p)
    retained=[p for p in ValueIndependentAcceptance.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    body={**EVIDENCE,'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'duration_seconds':time.monotonic()-started,'covered_methods':methods,'inputs':input_map(paths+retained),'retained_directory':str(ValueIndependentAcceptance.base.relative_to(ROOT)).replace('\\','/'),'failure_details':[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors]}
    save(WORK/(name+'-tests.json'),body)
    print(config.canonical(dict(event='value_independent_group',group=name,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1

def collect():
    covered=[];groups=[];evidence={};metadata={'passed','tests_run','failures','errors','skips','duration_seconds','covered_methods','inputs','retained_directory','failure_details'}
    for name,methods in GROUPS.items():
        path=WORK/(name+'-tests.json');body=load(path)
        require(body['passed'] and body['tests_run']==len(methods) and body['covered_methods']==methods and not body['failures']+body['errors']+body['skips'],'Incomplete independent group '+name)
        for file,digest in body['inputs'].items():require(sha(ROOT/file)==digest,'Stale independent input '+file)
        for key,value in body.items():
            if key not in metadata:require(key not in evidence,'Duplicate independent evidence '+key);evidence[key]=value
        covered+=methods;groups.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),methods=methods))
    expected=unittest.defaultTestLoader.getTestCaseNames(ValueIndependentAcceptance)
    require(len(covered)==len(set(covered)) and sorted(covered)==expected,'Independent unique method coverage differs')
    body={**evidence,'passed':True,'tests_run':len(covered),'failures':0,'errors':0,'skips':0,'covered_methods':sorted(covered),'groups':groups};save(WORK/'tests.json',body);return body

def training_state_worker(path):
    """Exact state comparison in a separate pinned Torch interpreter."""
    import torch
    spec=load(path);left=torch.load(spec['left'],map_location='cpu',weights_only=True);right=torch.load(spec['right'],map_location='cpu',weights_only=True);count=0
    def equal(a,b,where):
        nonlocal count
        if torch.is_tensor(a):
            require(torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),'Tensor state differs '+where);count+=1
        elif isinstance(a,dict):
            require(isinstance(b,dict) and a.keys()==b.keys(),'State keys differ '+where)
            for key in a:equal(a[key],b[key],where+'.'+str(key))
        elif isinstance(a,(list,tuple)):
            require(type(a) is type(b) and len(a)==len(b),'State sequence differs '+where)
            for index,(x,y) in enumerate(zip(a,b)):equal(x,y,where+'.'+str(index))
        else:require(type(a) is type(b) and a==b,'State value differs '+where)
    equal(left,right,'checkpoint');save(spec['output'],dict(passed=True,step=left['step'],exact_model_optimizer_and_rng=True,tensor_states_compared=count));return 0

def production_training_worker(output):
    """Independently parse every production PT tensor, not a proof's assertions."""
    import hashlib
    import torch
    from torch import nn
    from congklak import value_common as common
    plan=load(common.PLAN);run=(ROOT/plan['models']['refit']).parent;export=load(run/'model.json')['model'];identity=export['identity'];admitted=load(ROOT/'build/value-data.json');seed=identity['seed']
    torch.set_num_threads(1);torch.manual_seed(seed)
    layers=nn.Sequential(nn.Linear(127,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh());policy=nn.Linear(64,7)
    expected_cpu_rng=torch.get_rng_state();expected_cuda_rng=torch.cuda.get_rng_state_all() if identity['device']=='cuda' else [];expected_python_rng=random.Random(seed).getstate()
    parameters=list(layers[4].parameters());expected_opt=torch.optim.AdamW(parameters,lr=identity['lr'],weight_decay=0.,foreach=False).state_dict()['param_groups']
    dimensions={'layers.0.weight':(64,127),'layers.0.bias':(64,),'layers.2.weight':(64,64),'layers.2.bias':(64,),'layers.4.weight':(1,64),'layers.4.bias':(1,),'policy.weight':(7,64),'policy.bias':(7,)}
    initial=load(ROOT/plan['models']['control'])['model']['weights'];trainable={'layers.4.weight','layers.4.bias'};frozen=set(dimensions)-trainable
    with closing(sqlite3.connect((ROOT/'data/deeper-9000-v2/work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        rowstrata={(key if isinstance(key,str) else key.hex()):stratum for key,stratum in db.execute("SELECT state,json_extract(body,'$.stratum') FROM roots WHERE json_extract(body,'$.split')='train' ORDER BY state")}
    groups=defaultdict(list)
    for index,key in enumerate(admitted['train_state_keys']):groups[rowstrata[key]].append(index)
    strata=sorted(groups);sampler=random.Random(seed);draws=0;rows=[];paths=[run/'model.json',run/'result.json',run/'catalog.sqlite3',ROOT/'build/value-data.json',common.PLAN];last=None
    def weights_digest(values):
        digest=hashlib.sha256()
        for name,value in sorted(values.items()):digest.update(name.encode());digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
        return digest.hexdigest()
    with closing(sqlite3.connect((run/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        configs=db.execute('SELECT body,hash FROM config').fetchall();require(len(configs)==1 and json.loads(configs[0][0])==identity and configs[0][1]==config.digest(identity),'Independent training identity differs')
        entries=db.execute('SELECT id,step,name,sha,body,hash FROM checkpoints ORDER BY id').fetchall();steps=[r[1] for r in entries];require(steps==sorted(set(steps)) and set(range(50,1501,50))<=set(steps) and steps[-1]==1500,'Independent training schedule differs')
        for index,step,name,digest,text,bodyhash in entries:
            file=run/name;require(file.resolve().parent==run.resolve() and sha(file)==digest,'Independent PT checksum differs');meta=json.loads(text);require(config.digest(meta)==bodyhash and meta['step']==step and meta['identity_sha256']==config.digest(identity),'Independent training metadata differs')
            state=torch.load(file,map_location='cpu',weights_only=True);require(set(state)=={'identity','step','draws','model','optimizer','sampler_rng','python_rng','torch_rng','cuda_rng','scheduler','amp_scaler','pending_batches'} and state['identity']==identity and type(state['step']) is int and state['step']==step and type(state['draws']) is int and state['draws']==step*256,'Independent training state fields/counters differ')
            require(set(state['model'])==set(dimensions),'Independent parameter ownership differs')
            for key,value in state['model'].items():require(value.shape==dimensions[key] and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Independent model tensor differs '+key)
            for key in frozen:require(torch.equal(state['model'][key],torch.tensor(initial[key],dtype=torch.float32)),'Independent frozen policy tensor changed '+key)
            require(weights_digest(state['model'])==meta['weights_sha256'],'Independent model weight digest differs');opt=state['optimizer'];require(set(opt)=={'state','param_groups'} and opt['param_groups']==expected_opt and set(opt['state'])=={0,1},'Independent two-parameter optimizer configuration differs')
            for number,values in opt['state'].items():
                require(set(values)=={'step','exp_avg','exp_avg_sq'} and values['step'].numel()==1 and values['step'].item()==step,'Independent optimizer step differs')
                for key in ('exp_avg','exp_avg_sq'):require(values[key].shape==parameters[number].shape and values[key].dtype==torch.float32 and bool(torch.isfinite(values[key]).all()),'Independent optimizer moment differs')
                require(bool((values['exp_avg_sq']>=0).all()),'Independent squared moment negative')
            for _ in range(state['draws']-draws):sampler.choice(groups[sampler.choice(strata)])
            draws=state['draws'];require(sampler.getstate()==state['sampler_rng'] and state['python_rng']==expected_python_rng and torch.equal(state['torch_rng'],expected_cpu_rng),'Independent sampler/Python/Torch RNG differs')
            require(len(state['cuda_rng'])==len(expected_cuda_rng) and all(torch.equal(a,b) for a,b in zip(state['cuda_rng'],expected_cuda_rng)),'Independent CUDA RNG differs')
            require(state['scheduler'] is None and state['amp_scaler'] is None and state['pending_batches']==[],'Independent pending optimizer state differs')
            history=db.execute('SELECT body,hash FROM history WHERE step=?',(step,)).fetchall();require(len(history)==1 and json.loads(history[0][0])==meta['metrics'] and config.digest(meta['metrics'])==history[0][1],'Independent checkpoint history differs')
            rows.append(dict(step=step,weight_sha256=meta['weights_sha256']));paths.append(file);last=state
    require(export['step']==1500 and export['weights_sha256']==weights_digest(last['model']) and all(export['weights'][name]==value.tolist() for name,value in last['model'].items()),'Independent final export mapping differs')
    changed=[name for name,value in last['model'].items() if initial[name]!=value.tolist()];require(bool(changed) and set(changed)<=trainable,'Changes extend beyond the value head or head did not change')
    save(output,dict(passed=True,checkpoints=len(rows),steps=rows,all_optimizer_tensors_and_sampler_python_torch_cuda_rng_independently_verified=True,final_export_equals_PT=True,changed_tensors=changed,all_frozen_tensors_bit_exact=True,frozen_parameters=12807,trainable_parameters=65,optimizer_parameter_ids=[0,1],draws=draws,inputs=input_map(paths+[ROOT/plan['models']['control']])));return 0

def state_refusal_worker(output):
    import copy
    from congklak import value_learning as learning
    torch,_=learning.runtime();run=learning.DEFAULT;model=learning.evaluator()
    with closing(sqlite3.connect((run/'catalog.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:name=db.execute('SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1').fetchone()[0]
    state=torch.load(run/name,map_location='cpu',weights_only=True);learning.validate_state(state,state['identity'],model);frozen=set(state['model'])-set(learning.TRAINABLE);refused=0
    def rejects(value):
        try:learning.validate_state(value,value['identity'],model)
        except (ValueError,RuntimeError,KeyError):return True
        return False
    for key in frozen:
        forged=copy.deepcopy(state);forged['model'][key].view(-1)[0]+=.25;require(rejects(forged),'Frozen tensor mutation accepted '+key);refused+=1
    for key in learning.TRAINABLE:
        allowed=copy.deepcopy(state);allowed['model'][key].view(-1)[0]+=.01;learning.validate_state(allowed,allowed['identity'],model)
    for kind in ('draws','weight_decay','optimizer_ids','moment_shape'):
        forged=copy.deepcopy(state)
        if kind=='draws':forged['draws']+=1
        if kind=='weight_decay':forged['optimizer']['param_groups'][0]['weight_decay']=.0001
        if kind=='optimizer_ids':forged['optimizer']['state'][2]=copy.deepcopy(forged['optimizer']['state'][0])
        if kind=='moment_shape':forged['optimizer']['state'][0]['exp_avg']=torch.zeros(64)
        require(rejects(forged),'Corrupt checkpoint contract accepted '+kind)
    save(output,dict(passed=True,frozen_tensors_mutated_and_refused=refused,allowed_value_head_mutations=2,invalid_optimizer_and_draws_refused=True,inputs=input_map([run/name,ROOT/'congklak/value_learning.py'])));return 0

def prior_audit():
    accepted=load(ROOT/'build/quality-acceptance.json');require(accepted['passed'] and len(accepted['inputs'])==1647,'Prior CPU acceptance differs')
    for path,digest in accepted['inputs'].items():require(sha(ROOT/path)==digest,'Prior accepted input changed '+path)
    return dict(prior_inputs=1647,all_current_hashes_match=True)

def registry_audit():
    from congklak import value_common as common,pilot
    registry=common.read_registry();old=load(common.previous.REGISTRY)
    quotients=set(old['quotients']);families=set(old['families']);games=choices=active=0
    for folder in ('data/quality-benchmark-v1',):
        with closing(sqlite3.connect((ROOT/folder/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
            for start,moves,family,final in db.execute("SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.final') FROM games ORDER BY hash"):
                q=normalize_literal(json.loads(start));families.add(family)
                for action in [*json.loads(moves),None]:
                    p=json_position(q)
                    if sum(map(sum,p['pits']))>4:quotients.add(pilot.quotient(p).hex());active+=1
                    if action is not None:q=literal.apply(q,action).position;choices+=1
                require(json_position(q)==json.loads(final),'Literal protected recent game differs');games+=1
    require(games==384 and registry['recent_arena_games']==384 and set(registry['quotients'])==quotients and set(registry['families'])==families and registry['recent_active_observations']==active,'Independent protected root registry differs')
    require(registry['sealed_target_bodies_read'] is False and registry['metadata_only_dataset_access'] is True,'Unsupported protected-target access')
    return dict(recent_games=games,literal_choices=choices,recent_active_observations=active,quotients=len(quotients),families=len(families),literal_rederived=True,sealed_target_bodies_read=False,scope='Actual evaluated roots and family metadata; no enumeration of every possible family trajectory')

def model_audit(plan):
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    from stage5_deeper_audit_acceptance import value_and_policy
    from congklak import deeper_teacher_v2 as teacher
    _,data,_=teacher.load();roots=[r['state'] for r in data['development']];observed=[];control_policies=[]
    for name in ('control','refit'):
        path=ROOT/plan['models'][name]
        with CpuBatchEvaluator(path,cache_capacity=0,max_batch=32) as evaluator:
            worst=0.
            for offset in range(0,len(roots),32):
                batch=roots[offset:offset+32]
                for index,(p,row) in enumerate(zip(batch,evaluator.evaluate_many(batch))):
                    value,_,logits=value_and_policy(normalize_literal(p),evaluator.model.envelope['model']);error=max(abs(value-row['value_player0']),max(abs(a-b) for a,b in zip(logits,row['policy_logits'])))
                    require(error<=1e-10,'Literal model forward differs');worst=max(worst,error)
                    if name=='control':control_policies.append(row['policy_logits'])
                    else:require(row['policy_logits']==control_policies[offset+index],'Native frozen policy logits differ')
            observed.append(dict(model=name,roots=1000,literal_forward=True,max_error=worst,file_sha256=sha(path),weights_sha256=evaluator.model.id))
    left=load(ROOT/plan['models']['control'])['model'];right=load(ROOT/plan['models']['refit'])['model'];frozen=set(left['weights'])-{'layers.4.weight','layers.4.bias'}
    require(left['encoder']==right['encoder'] and all(left['weights'][key]==right['weights'][key] for key in frozen),'Global policy subgraph differs')
    return dict(models=observed,all_refit_outputs_literal_verified=True,all_frozen_tensors_bit_exact=True,global_policy_unchanged_by_identical_subgraph=True,policy_logits_bit_exact=True)

def benchmark_audit(plan):
    from congklak import value_common as common
    bench=module('value_benchmark_full_audit','tools/benchmark-value.py');manifest,witness=bench.admit(plan);report=load(common.OUTPUT);counts=defaultdict(int);cohorts=defaultdict(list);bycomparison=defaultdict(list)
    with closing(sqlite3.connect((common.RUN/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        text,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();identity=json.loads(text)
        require(config.digest(identity)==digest and identity==report['identity'] and identity['implementation']==manifest and identity['plan_sha256']==config.digest(plan),'Benchmark identity differs')
        for index,text,digest in db.execute('SELECT idx,body,hash FROM cases ORDER BY idx'):
            require(index==counts['cases'],'Case index prefix differs');case=json.loads(text);require(config.digest(case)==digest,'Case checksum differs');bench.validate_case(case,index,plan)
            reference=plan['cases'][index]['reference'];sign=1 if plan['cases'][index]['state']['controller']==0 else -1;q={r['pit']:r['value_player0'] for r in reference['action_scores']}
            for ms,engines in case['measurements'].items():
                for name,row in engines.items():
                    decision=row['decision']['decision'];chosen=decision['chosen'];require(row['reference_optimal']==(chosen in reference['best_moves'] if chosen is not None else None) and row['reference_regret']==(sign*(reference['target_player0']-q[chosen]) if chosen is not None else None),'Independent root reference metric differs')
                    counts['scalar_measurements']+=1
            counts['cases']+=1
        comparisons={c['id']:c for c in plan['arena_comparisons']}
        for index,name,pair,seat,text,digest in db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'):
            require(index==counts['games'] and plan['arena_work_order'][index]==dict(comparison=name,pair=pair,seat=seat),'Arena declared order differs');game=json.loads(text);require(config.digest(game)==digest,'Arena hash differs');bench.replay_game(game,comparisons[name],pair,seat,plan,bench.Native())
            q=normalize_literal(game['start']);fallbacks=Counter()
            for action,row in zip(game['moves'],game['decisions']):
                require(row['actor']==q.controller and action in literal.legal_actions(q),'Literal actor/action differs')
                expected=comparisons[name]['candidate'] if q.controller==seat else comparisons[name]['opponent'];require(row['engine']==expected,'Literal engine seat assignment differs')
                if row['fallback']:require(action==literal.legal_actions(q)[0],'Fallback action differs');fallbacks[row['engine']]+=1
                q=literal.apply(q,action).position;counts['literal_choices']+=1
            terminal=literal.normalize(q).terminal;margin=literal.evaluate(q)*(1 if seat==0 else -1)
            require(json_position(q)==game['final'] and dict(fallbacks)==game['fallbacks'] and game['status']==('terminal' if terminal else 'unfinished') and game['candidate_final_margin']==(margin if terminal else None),'Literal arena final result differs')
            bycomparison[name].append(game);counts['games']+=1
        require((counts['cases'],counts['scalar_measurements'],counts['games'])==(80,640,192),'Incomplete benchmark coverage')
        require(report==bench.summarize(db,identity,plan),'Published benchmark summary differs')
    primary={};paired={}
    for name,games in bycomparison.items():
        margins=[];scores=[];paired[name]={}
        for pair in range(24):
            group=[g for g in games if g['pair']==pair];require(len(group)==2 and {g['candidate_seat'] for g in group}=={0,1},'Unpaired arena family')
            if all(g['status']=='terminal' for g in group):
                margin=sum(g['candidate_final_margin'] for g in group)/2;margins.append(margin);paired[name][pair]=margin;scores.append(sum(g['candidate_match_score'] for g in group)/2)
        rng=random.Random(20261019);samples=sorted(sum(rng.choices(margins,k=len(margins)))/len(margins) for _ in range(4096)) if margins else [];interval=[samples[102],samples[3993]] if samples else None;mean=statistics.mean(margins) if margins else None;fallbacks=sum(sum(g['fallbacks'].values()) for g in games);gate=bool(len(margins)==24 and not fallbacks and interval and interval[0]>0)
        row=report['arenas'][name];require(row['primary_paired_mean_final_margin_seeds']==mean and row['primary_margin_bootstrap_95_interval']==interval and row['primary_strength_gate_passed']==gate,'Independent paired interval or gate differs')
        require(row['paired_mean_score']==(statistics.mean(scores) if scores else None),'Independent paired WDL differs');primary[name]=dict(mean_margin_seeds=mean,interval=interval,gate_passed=gate,complete_pairs=len(margins),fallbacks=fallbacks)
    gates={name:all(primary[name+'_vs_'+base]['gate_passed'] for base in ('H_v1','H8_relay_CPU')) for name in ('policy_control','value_refit')}
    require(report['both_baseline_primary_gates']==gates and report['default_promotion'] is False and report['sealed_final_test_consumed'] is False and report['energy_measured'] is False,'Unsupported promotion/energy claim')
    contrasts={}
    for baseline in ('H_v1','H8_relay_CPU'):
        a=paired['value_refit_vs_'+baseline];b=paired['policy_control_vs_'+baseline];matched=sorted(set(a)&set(b));contrasts[baseline]=dict(treatment='value_refit',control='policy_control',expected_families=24,complete_matched_families=len(matched),paired_mean_seed_margin_difference=statistics.mean(a[p]-b[p] for p in matched) if len(matched)==24 else None,confidence_interval=None,primary_gate=False)
    require(report['secondary_factor_contrasts']==contrasts,'Independent matched family factor contrasts differ')
    resource_path=common.OUTPUT.with_name(common.OUTPUT.stem+'-resources.json');resources=load(resource_path);bench.validate_resources(resources,common.OUTPUT,manifest)
    return dict(**counts,primary=primary,both_baseline_gates=gates,secondary_factor_contrasts=contrasts,all_terminal_games=sum(g['status']=='terminal' for games in bycomparison.values() for g in games),all_fallbacks=sum(p['fallbacks'] for p in primary.values()),whole_decision_clocks=True,original_resources=resources)

def full():
    from congklak import value_common as common
    if (WORK/'audit.json').is_file():
        previous=load(WORK/'audit.json')
        if previous.get('passed') and previous.get('inputs') and all((ROOT/path).is_file() and sha(ROOT/path)==digest for path,digest in previous['inputs'].items()):
            print(config.canonical(dict(event='value_independent_full_audit_verified',passed=True,files_rewritten=False)),flush=True);return 0
        index=0
        while (WORK/f'audit-before-rerun-{index}.json').exists():index+=1
        (WORK/f'audit-before-rerun-{index}.json').write_bytes((WORK/'audit.json').read_bytes())
    started=time.monotonic();result=dict(passed=False);paths=[ROOT/p for p in SOURCES]+[ROOT/'build/quality-acceptance.json']
    try:
        result['tests']=collect();plan,manifest=common.admit();result['prior']=prior_audit();result['registry']=registry_audit();result['models']=model_audit(plan);result['benchmark']=benchmark_audit(plan)
        runtime=load(ROOT/'build/training-runtime.json')['python'];training_output=WORK/'production-training.json'
        import subprocess
        with (WORK/'production-training.log').open('w',encoding='utf-8') as stream:child=subprocess.run([runtime,str(ROOT/'tools/value-audit.py'),'production-training-worker','--spec',str(training_output)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
        require(child.returncode==0,'Independent production training payload audit failed; see '+str(WORK/'production-training.log'));result['training']=load(training_output);require(result['training']['passed'],'Independent production training audit incomplete')
        paths += [ROOT/path for path in manifest['sources']]+[ROOT/path for path in manifest['science_inputs']]+[common.IMPLEMENTATION,common.OUTPUT,common.RUN/'work.sqlite3',common.OUTPUT.with_name(common.OUTPUT.stem+'-resources.json')]
        paths += [WORK/'tests.json']+[WORK/(name+'-tests.json') for name in GROUPS]
        paths += [ROOT/path for name in GROUPS for path in load(WORK/(name+'-tests.json'))['inputs']]
        for model in plan['models'].values():paths.append(ROOT/model)
        paths += [training_output,WORK/'production-training.log']+[ROOT/path for path in result['training']['inputs']]
        result.update(passed=True,inputs=input_map(paths),default_promotion=False,sealed_target_bodies_read=False,energy_measured=False,new_value_cuda_actual_kill_verified=True,unrelated_prior_warm_kill_claimed=False)
    except Exception as exc:result['error']=str(exc)
    result['duration_seconds']=time.monotonic()-started;save(WORK/'audit.json',result);print(config.canonical(dict(event='value_independent_full_audit',passed=result['passed'],error=result.get('error'))),flush=True);return 0 if result['passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['group','collect','full','training-state-worker','production-training-worker','state-refusal-worker']);p.add_argument('--group',choices=list(GROUPS));p.add_argument('--spec');a=p.parse_args()
    if a.operation=='group':require(a.group,'Missing group');return group(a.group)
    if a.operation=='training-state-worker':require(a.spec,'Missing state comparison spec');return training_state_worker(a.spec)
    if a.operation=='production-training-worker':require(a.spec,'Missing production training output');return production_training_worker(a.spec)
    if a.operation=='state-refusal-worker':require(a.spec,'Missing checkpoint refusal output');return state_refusal_worker(a.spec)
    if a.operation=='full':return full()
    collect();return 0
if __name__=='__main__':raise SystemExit(main())
