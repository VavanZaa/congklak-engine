"""Independent Phase6 evidence collection and full published-game audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,pilot,campaign,phase6_common as common,phase6_selfplay as sp
from congklak.neural_relay import Model,NeuralNative
from phase6_audit_acceptance import Phase6IndependentAcceptance,EVIDENCE,normalize_literal,json_position,sha,load
import reference as literal
from stage5_deeper_audit_acceptance import literal_features,value_and_policy
WORK=ROOT/'build/phase6-audit';WORK.mkdir(parents=True,exist_ok=True)
NAMES=unittest.defaultTestLoader.getTestCaseNames(Phase6IndependentAcceptance)
GROUPS={'core':NAMES[:7]+[NAMES[12]],'tensor':[NAMES[7]],'warm':[NAMES[8]],'update':[NAMES[9]],'case':[NAMES[10]],'game':[NAMES[11]]}
from phase6_cuda_selfplay_audit import Phase6CudaSelfplayAcceptance,EVIDENCE as CUDA_GAME_EVIDENCE
from phase6_cuda_training_audit_v2 import Phase6CudaTrainingV2Acceptance
from phase6_versioned_pipeline_audit_v3 import Phase6VersionedPipelineAuditV3
from phase6_warm_evidence_review import Phase6AuthorWarmEvidenceReview
WARM_REVIEW_METHOD='test_09_independent_author_warm_cuda_recovery_evidence_review'
GROUPS['warm']=[WARM_REVIEW_METHOD]
CUDA_GAME_METHOD='test_14_cuda_selfplay_actual_pre_postcommit_kills_and_legal_resume'
GROUPS['cuda-selfplay']=[CUDA_GAME_METHOD]
SOURCES=['tests/phase6_audit_acceptance.py','tests/reference.py','congklak/phase6_common.py','congklak/phase6_store.py','congklak/phase6_selfplay.py','congklak/phase6_analysis.py','congklak/puct.py','congklak/phase6_inference.py','congklak/phase6_training.py','tools/benchmark-phase6.py']

def require(value,message):
    if not value:raise AssertionError(message)
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def inputs(paths):return {str(Path(p).relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}

def group(name):
    started=time.monotonic();EVIDENCE.clear();CUDA_GAME_EVIDENCE.clear();methods=GROUPS[name]
    case=Phase6CudaSelfplayAcceptance if name=='cuda-selfplay' else Phase6AuthorWarmEvidenceReview if name=='warm' else Phase6VersionedPipelineAuditV3 if name in ('update','case','game') else Phase6IndependentAcceptance
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(case(n) for n in methods))
    paths=[ROOT/p for p in SOURCES]+[common.PLAN,common.REGISTRY]
    if name=='cuda-selfplay':paths.append(ROOT/'tests/phase6_cuda_selfplay_audit.py')
    if name in ('update','case','game'):paths += [ROOT/'tests/phase6_versioned_pipeline_audit_v3.py',ROOT/'congklak/phase6_training_v2.py',ROOT/'tools/benchmark-phase6-v2.py',ROOT/'build/phase6-implementation-v2.json']
    if name in ('update','case','game'):
        retained=[p for p in case.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
        paths += retained
        EVIDENCE[name+'_retained_raw_fixtures']=dict(directory=str(case.base.relative_to(ROOT)).replace('\\','/'),files=inputs(retained),preserved_after_test=True)
    if name=='warm':paths.append(ROOT/'tests/phase6_warm_evidence_review.py')
    if name=='warm':paths += [ROOT/p for p in EVIDENCE['author_warm_recovery_evidence_review']['inputs']]
    body={**EVIDENCE,**CUDA_GAME_EVIDENCE,'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
          'duration_seconds':time.monotonic()-started,'covered_methods':methods,'test_class':case.__name__,'source_sha256':sha(ROOT/SOURCES[0]),'inputs':inputs(paths),
          'failure_details':[dict(method=str(t),trace=trace) for t,trace in result.failures+result.errors]}
    save(WORK/(name+'-tests.json'),body);print(config.canonical(dict(event='phase6_independent_group',group=name,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1

def collect():
    covered=[];groups=[];evidence={};seconds=0;source=sha(ROOT/SOURCES[0])
    metadata={'passed','tests_run','failures','errors','skips','duration_seconds','covered_methods','test_class','source_sha256','inputs','failure_details'}
    for name,methods in GROUPS.items():
        path=WORK/(name+'-tests.json');body=load(path)
        require(body['passed'] and body['tests_run']==len(methods) and not body['failures']+body['errors']+body['skips'],'Incomplete group '+name)
        require(body['source_sha256']==source and body['covered_methods']==methods,'Stale method coverage '+name)
        for p,digest in body['inputs'].items():require(sha(ROOT/p)==digest,'Stale group input '+p)
        for key,value in body.items():
            if key not in metadata:require(key not in evidence,'Duplicate group evidence '+key);evidence[key]=value
        covered+=methods;seconds+=body['duration_seconds'];groups.append(dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path),methods=methods))
    accepted=[n for n in NAMES if n!=NAMES[8]]+[WARM_REVIEW_METHOD,CUDA_GAME_METHOD]
    require(len(covered)==len(set(covered))==14 and sorted(covered)==sorted(accepted),'Independent method coverage differs')
    result={**evidence,'passed':True,'tests_run':14,'failures':0,'errors':0,'skips':0,'duration_seconds':seconds,'covered_methods':sorted(covered),'source_sha256':source,'additional_test_source_sha256':sha(ROOT/'tests/phase6_cuda_selfplay_audit.py'),'groups':groups}
    save(WORK/'tests.json',result);return result

def registry_audit():
    registry=common.read_registry();qs=set();fs=set();old_family={}
    for folder in (pilot.DEFAULT_PATH,campaign.DEFAULT):
        manifest=pilot.load(folder);dbpath=Path(folder)/manifest['directory']/'dataset.sqlite3'
        with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)) as db:
            for key,split,family in db.execute("SELECT state,json_extract(body,'$.split'),json_extract(body,'$.family') FROM examples"):
                old_family[bytes(key).hex()]=family
                if split!='train':qs.add(pilot.quotient(pilot.from_key(bytes(key))).hex());fs.add(family)
            for kind,key,split in db.execute('SELECT kind,state,split FROM exclusions'):
                if split!='train':qs.add(bytes(key).hex() if kind=='quotient' else pilot.quotient(pilot.from_key(bytes(key))).hex())
            fs.update(f for f,s in db.execute('SELECT id,split FROM families') if s!='train')
        with closing(sqlite3.connect(Path(folder)/'work.sqlite3')) as db:
            for family, in db.execute('SELECT family FROM collection_units'):
                split=('development' if hashlib.sha256(bytes.fromhex(family.split('/')[1])).digest()[0]<128 else 'sealed_final_test') if family.startswith('constructed-k4/') else pilot.family_split(family)
                if split!='train':fs.add(family)
    with closing(sqlite3.connect(ROOT/'data/stronger-10000/work.sqlite3')) as db:
        fs.update(f for f,s in db.execute('SELECT id,split FROM families') if s!='train')
        for key, in db.execute("SELECT DISTINCT o.state FROM origins o JOIN families f ON f.id=o.family WHERE f.split!='train'"):
            qs.add(pilot.quotient(pilot.from_key(bytes(key))).hex())
        for key, in db.execute("SELECT state FROM selected WHERE split!='train'"):qs.add(pilot.quotient(pilot.from_key(bytes(key))).hex())
        for kind,key,split in db.execute('SELECT kind,state,split FROM exclusions'):
            if split!='train':qs.add(bytes(key).hex() if kind=='quotient' else pilot.quotient(pilot.from_key(bytes(key))).hex())
    games=choices=observations=0
    for folder in common.HISTORY:
        dbpath=ROOT/folder/'work.sqlite3';declared=next(s for s in registry['sources'] if s['path']==folder)
        require(sha(dbpath)==declared['database_sha256'],'Historical ledger changed')
        with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)) as db:
            sql="SELECT json_extract(body,'$.start'),json_extract(body,'$.moves'),json_extract(body,'$.family'),json_extract(body,'$.start_key'),json_extract(body,'$.final') FROM games"
            for start,moves,family,key,final in db.execute(sql):
                fs.add(family or old_family[key]);q=normalize_literal(json.loads(start));moves=json.loads(moves)
                for action in moves+[None]:
                    p=json_position(q)
                    if sum(map(sum,p['pits']))>4:qs.add(pilot.quotient(p).hex());observations+=1
                    if action is not None:q=literal.apply(q,action).position;choices+=1
                require(json_position(q)==json.loads(final),'Historical literal replay differs');games+=1
    require(qs==set(registry['quotients']) and fs==set(registry['families']),'Protected registry incomplete or fabricated')
    require(games==registry['historical_games']==720 and observations==registry['historical_active_observations'],'Historical registry coverage differs')
    return dict(protected_quotients=len(qs),protected_families=len(fs),literal_history_games=games,literal_history_choices=choices,metadata_only=True,sealed_target_bodies_read=False)

def mechanism(p):
    q=normalize_literal(p);r=sum(map(sum,p['pits']));band='5-24' if r<=24 else '25-49' if r<=49 else '50-74' if r<=74 else '75-98'
    transitions=[literal.apply(q,a) for a in literal.legal_actions(q)]
    kind='pass' if any(t.initial_passes or t.final_passes for t in transitions) else 'capture' if any(t.captured for t in transitions) else 'bonus' if any(t.bonus for t in transitions) else 'ordinary'
    return band+'/'+kind

def cohort_audit():
    plan=common.read_plan();registry=common.read_registry();allrows={}
    with closing(sqlite3.connect(ROOT/'data/deeper-9000-v2/work.sqlite3')) as db:
        sql="SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family'),json_extract(body,'$.stratum'),json_extract(body,'$.split') FROM roots"
        for key,state,family,stratum,split in db.execute(sql):
            encoded=key if isinstance(key,str) else bytes(key).hex();allrows[encoded]=dict(state_key=encoded,state=json.loads(state),family=family,stratum=stratum,split=split)
        require(Counter(r['split'] for r in allrows.values())=={'train':8000,'development':1000},'H8 cohort ownership differs')
        for case in plan['cases'][:48]:
            row=allrows[case['state_key']];require(row['split']=='development' and all(case[k]==row[k] for k in ('state_key','state','family','stratum')),'Proxy ownership differs')
            label=json.loads(db.execute("SELECT json_extract(body,'$.label') FROM labels WHERE state=?",(case['state_key'],)).fetchone()[0]);require(label==case['reference'],'H8 proxy reference differs')
    prior=load(ROOT/'build/stage5-deeper-benchmark-plan.json');require(plan['cases'][48:]==[r for r in prior['cases'] if r['reference']['kind']=='exact_optimal_margin'],'Exact diagnostic cohort differs')
    historical={r['family'] for name in ('stage5-benchmark-plan.json','stage5-deeper-benchmark-plan.json') for r in load(ROOT/'build'/name)['arena_starts']}
    require(len(historical)==60,'Prior arena family count differs')
    def contested(r):p=r['state'];active=sum(map(sum,p['pits']));return active>=25 and abs(p['stores'][0]-p['stores'][1])<active
    def choose(rows,counts,namespace,excluded):
        ranked=sorted(rows,key=lambda r:hashlib.sha256((namespace+'/'+r['state_key']).encode()).digest());used=set(excluded);out=[]
        for kind,quota in counts:
            group=[]
            for r in ranked:
                if r['family'] not in used and r['stratum'].endswith('/'+kind):group.append(r);used.add(r['family'])
                if len(group)==quota:break
            require(len(group)==quota,'Predeclared selector quota unavailable');out+=group
        return [{k:r[k] for k in ('state_key','state','family','stratum')} for r in out]
    train=[r for r in allrows.values() if r['split']=='train' and contested(r) and r['family'] not in registry['families'] and pilot.quotient(r['state']).hex() not in registry['quotients']]
    dev=[r for r in allrows.values() if r['split']=='development']
    require(plan['selfplay_starts']==choose(train,[('ordinary',12),('capture',10),('bonus',10)],'phase6-selfplay-start-v1',[]),'Selfplay starts changed by outcomes')
    require(plan['arena_starts']==choose([r for r in dev if contested(r)],[('ordinary',8),('capture',8),('bonus',8)],'phase6-arena-start-v1',historical),'Arena starts changed by outcomes')
    require([{k:r[k] for k in ('state_key','state','family','stratum')} for r in plan['cases'][:48]]==choose(dev,[('ordinary',16),('capture',16),('bonus',16)],'phase6-H8-proxy-v1',[]),'Proxy starts changed by outcomes')
    return dict(train_source_roots=8000,development_source_roots=1000,selfplay_start_families=32,new_arena_families=24,prior_arena_families_excluded=60,H8_proxy_references=48,verified_exact_diagnostics=32,deterministic_predeclared_selection_reproduced=True)

def selfplay_audit():
    manifest,rows=sp.load_training_rows();identity=manifest['identity'];plan=common.read_plan();registry=common.read_registry()
    require(identity['settings']==plan['selfplay'] and identity['starts']==plan['selfplay_starts'] and identity['plan_sha256']==config.digest(plan),'Production selfplay plan differs')
    require(identity['model_file_sha256']==sha(common.WARM) and identity['model_weights_sha256']==Model(common.WARM).id,'Selfplay generator model differs')
    protected=set(registry['quotients']);families=defaultdict(list);expected=[];decisions=0;bygame={};excluded={};lowactive=0
    with closing(sqlite3.connect(common.SELFPLAY/'work.sqlite3')) as db:
        games=sp.read_games(db);require(len(games)==64,'Pilot must retain fixed64 games')
        for game in games:
            families[game['family']].append(game);q=normalize_literal(game['start']);rng=random.Random(game['seed']);observations=[]
            bridge=pilot.quotient(json_position(q)).hex() in protected and sum(map(sum,json_position(q)['pits']))>4
            for index,body,digest in db.execute('SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):
                row=json.loads(body);require(config.digest(row)==digest and row['state']==json_position(q),'Selfplay literal source differs')
                require(sp.tuples(row['rng_before'])==rng.getstate(),'RNG before differs');legal=literal.legal_actions(q)
                for _ in legal:rng.gammavariate(.3,1)
                policy=[v/64 for v in row['search']['visits']];require(policy==row['policy_target'],'Visit target differs')
                if index<8:
                    draw=rng.random();mass=0;chosen=legal[-1]
                    for a in legal:
                        mass+=policy[a]
                        if draw<mass:chosen=a;break
                else:chosen=max(legal,key=lambda a:(row['search']['visits'][a],-a))
                require(row['action']==chosen and sp.tuples(row['rng_after'])==rng.getstate(),'Literal sampled action/RNG differs')
                q=literal.apply(q,chosen).position;require(json_position(q)==row['after'],'Literal selfplay transition differs')
                if sum(map(sum,row['after']['pits']))>4 and pilot.quotient(row['after']).hex() in protected:bridge=True
                observations.append((row,digest));decisions+=1
            require(json_position(q)==game['position'] and game['choices']==len(observations),'Final selfplay cursor differs')
            terminal=literal.normalize(q).terminal
            require(game['status']==('terminal' if terminal else 'capped') and game['terminal_margin']==(literal.evaluate(q) if terminal else None),'Terminal selfplay target fabricated')
            bygame[game['id']]=observations
            if not terminal:excluded[game['family']]='incomplete_or_capped_family'
            if game['family'] in registry['families']:excluded[game['family']]='protected_family'
            if bridge:excluded[game['family']]='protected_active_quotient_bridge'
        for family,games in families.items():
            require(len(games)==2,'Missing seeded family run')
            if family in excluded:continue
            for game in games:
                for row,digest in bygame[game['id']]:
                    p=row['state'];active=sum(map(sum,p['pits']))
                    if active<=4:lowactive+=1;continue
                    x,d,r=literal_features(normalize_literal(p),127);x=list(x)
                    expected.append(dict(id=f"{game['id']}/{row['choiceindex']:04d}",labelkind='selfplay_terminal_seed_margin',value_target_player0=game['terminal_margin'],policy_target=row['policy_target'],gameid=game['id'],state=p,stratum=mechanism(p),family=family,choiceindex=row['choiceindex'],cached_features=dict(x=x,d=d,r=r),source_decision_sha256=digest,weight=1,split='train'))
    before=len(expected);buckets=defaultdict(list)
    for row in expected:buckets[common.rawkey(row['state'])].append(row)
    kept=[];pruned=[]
    for raw,bucket in sorted(buckets.items()):
        ordered=sorted(bucket,key=lambda r:config.digest(dict(namespace='phase6-rawstate-cap-v1',id=r['id'])));kept+=ordered[:4];pruned += [r['id'] for r in ordered[4:]]
    kept.sort(key=lambda r:r['id']);require(rows==kept,'Independent literal feature/terminal/policy/ownership/pruning differs')
    summary=manifest['summary'];require(summary['family_exclusions']==excluded and summary['observed_decisions']==decisions and summary['low_active_rows_omitted']==lowactive,'Admission counts differ')
    require(summary['eligible_rows_before_cap']==before and summary['pruned_row_ids']==sorted(pruned) and summary['retained_rows_by_game']==dict(sorted(Counter(r['gameid'] for r in kept).items())),'Cap/provenance counts differ')
    return dict(games=64,literal_decisions=decisions,terminal_games=summary['terminal_games'],retained_rows=len(rows),families_excluded=len(excluded),families_admitted=32-len(excluded),pruned_rows=len(pruned),cached_features_rederived=len(rows),individual_terminal_targets=True,rows_uniform_weight=True,manifest_sha256=config.digest(manifest))

def model_audit():
    source=Model(ROOT/common.read_plan()['source_model']);warm=Model(common.WARM);updated=Model(common.UPDATE);roots=[]
    with closing(sqlite3.connect((ROOT/'data/deeper-9000-v2/work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        sql="SELECT json_extract(r.body,'$.state'),json_extract(r.body,'$.stratum'),json_extract(l.body,'$.label.target_player0') FROM roots r JOIN labels l ON r.state=l.state WHERE json_extract(r.body,'$.split')='development' ORDER BY r.state"
        for state,stratum,target in db.execute(sql):
            require(type(target) is int and isinstance(stratum,str),'Development target provenance differs')
            roots.append(dict(state=json.loads(state),stratum=stratum,target=target))
    require(len(roots)==1000,'Development model audit coverage differs')
    for k in source.weights:
        if k.startswith('layers.'):require(warm.weights[k]==source.weights[k],'Warm value/trunk tensor changed')
    require(warm.weights['policy.weight']!=source.weights['policy.weight'] and updated.id!=warm.id,'No learning change in claimed iteration')
    out=[]
    for name,path,model in [('warm',common.WARM,warm),('updated',common.UPDATE,updated)]:
        errors=[];strata=defaultdict(list)
        with NeuralNative(path,backend='avx2',mode='value-none') as native:
            for row in roots:
                p=row['state']
                exact,units,logits=value_and_policy(normalize_literal(p),model.envelope['model']);raw=native.evaluate(p)
                # Frozen packed-double forward is checked against literal
                # features and separately accumulated dense layers.
                actual=raw['value_player0'];actual_logits=raw['policy_logits']
                require(abs(actual-exact)<=1e-10 and max(abs(a-b) for a,b in zip(logits,actual_logits))<=1e-10,'Literal native model forward differs')
                errors.append(abs(actual-exact))
                strata[row['stratum']].append((abs(exact-row['target']),abs(p['stores'][0]-p['stores'][1]-row['target'])))
        checked={s:dict(count=len(v),mae_seeds=sum(a for a,b in v)/len(v),H_v1_mae_seeds=sum(b for a,b in v)/len(v)) for s,v in strata.items()}
        published=load(path.parent/'result.json')['selected_development']['finite_choice_horizon']
        require(published['count']==1000 and set(published['strata'])==set(checked),'Development regression scope differs')
        for s,value in checked.items():
            saved=published['strata'][s]
            require(saved['count']==value['count'] and saved['H_v1_mae_seeds']==value['H_v1_mae_seeds'] and abs(saved['mae_seeds']-value['mae_seeds'])<=1e-4,'Independent stratum value MAE differs '+s)
        total=sum(v['mae_seeds']*v['count'] for v in checked.values())/1000;baseline=sum(v['H_v1_mae_seeds']*v['count'] for v in checked.values())/1000
        require(abs(published['mae_seeds']-total)<=1e-4 and published['H_v1_mae_seeds']==baseline and abs(published['uniform_stratum_mae_seeds']-sum(v['mae_seeds'] for v in checked.values())/len(checked))<=1e-4,'Independent overall value MAE differs')
        eligible={s:v for s,v in checked.items() if s.endswith('/ordinary') and v['count']>=20}
        require(eligible and all((v['mae_seeds']<=v['H_v1_mae_seeds'])==(published['strata'][s]['mae_seeds']<=published['strata'][s]['H_v1_mae_seeds']) for s,v in eligible.items()),'Independent ordinary preservation gate differs')
        out.append(dict(model=name,roots=1000,literal_feature_and_double_forward=True,max_value_difference=max(errors),weights_sha256=model.id,file_sha256=sha(path),
                        literal_development_regression=True,fp32_to_literal_MAE_tolerance_seeds=1e-4,mae_seeds=total,H_v1_mae_seeds=baseline,strata=checked,ordinary_H8_gate_passed=all(v['mae_seeds']<=v['H_v1_mae_seeds'] for v in eligible.values())))
    return dict(models=out,warm_six_value_tensors_unchanged=True,policy_and_selfplay_update_changed=True)

def checkpoint_worker():
    # Called only in an isolated CPU process after timing slots are released.
    import torch
    from congklak import phase6_training as warm_tr,phase6_training_v2 as update_tr
    torch.set_num_threads(1);manifest,data,_=warm_tr.teacher.load(teacher_depth=8);spmanifest,sprows=sp.load_training_rows()
    teacher_groups=defaultdict(list);spgroups=defaultdict(list)
    for i,r in enumerate(data['train']):teacher_groups[r['stratum']].append(i)
    for i,r in enumerate(sprows):spgroups[r['stratum']].append(i)
    out=[];files={}
    for stage,path,target,batch,count in [('warm',common.WARM.parent,1000,256,20),('selfplay',common.UPDATE.parent,500,128,10)]:
        tr=warm_tr if stage=='warm' else update_tr
        result=load(path/'result.json');model=Model(path/'model.json');identity=result['identity'];rng=random.Random(identity['seed']);checkpointed=0;history=[];last=None
        with closing(sqlite3.connect(path/'catalog.sqlite3')) as db:
            rows=db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY step').fetchall();require(len(rows)==count and [x[0] for x in rows]==list(range(50,target+1,50)),'Checkpoint cadence differs')
            require(load_json_row(db.execute('SELECT body,hash FROM config').fetchone())==identity,'Training catalog identity differs')
            for step,name,digest,body,bodyhash in rows:
                file=path/name;require(sha(file)==digest and config.digest(json.loads(body))==bodyhash,'Corrupt checkpoint file/catalog');files[str(file.relative_to(ROOT)).replace('\\','/')]=digest
                state=torch.load(file,map_location='cpu',weights_only=True);require(state['identity']==identity and state['step']==step,'Checkpoint identity differs')
                for _ in range(checkpointed,step):
                    n=batch if stage=='warm' else batch//2
                    for _ in range(n):rng.choice(teacher_groups[rng.choice(sorted(teacher_groups))])
                    if stage=='selfplay':
                        for _ in range(n):rng.choice(spgroups[rng.choice(sorted(spgroups))])
                checkpointed=step
                require(state['sampler_rng']==rng.getstate() and state['draws']==step*batch and state['teacher_draws']==step*(batch if stage=='warm' else batch//2) and state['selfplay_draws']==(0 if stage=='warm' else step*batch//2),'Sampler/RNG/mixed counter differs')
                require(state['scheduler'] is None and state['amp_scaler'] is None and state['pending_batches']==[],'Unsupported runtime state')
                require(len(state['optimizer']['state'])==(2 if stage=='warm' else 8) and len(state['cuda_rng'])==1 and state['torch_rng'].dtype==torch.uint8,'Optimizer/RNG coverage differs')
                require(tr.weight_digest(state['model'])==json.loads(body)['weights_sha256'],'State weight catalog differs')
                for t in state['model'].values():require(bool(torch.isfinite(t).all()),'Nonfinite checkpoint tensor')
                if stage=='warm':require(tr.weight_digest(tr.value_state(state['model']))==tr.SOURCE_VALUE_WEIGHTS and tr.weight_digest(tr.value_state(state['best_model']))==tr.SOURCE_VALUE_WEIGHTS,'Warm six value tensors changed')
                else:require((state['best_model'] is None and step<target) or (step==target==state['best_step']),'SP result selected by development')
                history.append((step,json.loads(body)['metrics']['policy']['uniform_stratum_teacher_cross_entropy']));last=state
        winner=min(history,key=lambda v:v[1])[0] if stage=='warm' else target
        require(result['status']=='complete' and result['steps_completed']==target and result['best_step']==winner==model.envelope['model']['step'],'Final selection differs')
        require(model.envelope['model']==tr.export_payload(last['best_model'],identity,winner) and tr.weight_digest(last['best_model'])==model.id==result['weights_sha256'],'Selected export differs from checkpoint')
        require(identity['source_sha256']==sha(tr.__file__) and identity['sealed_final_test_consumed'] is False,'Trainer source/ownership differs')
        out.append(dict(stage=stage,checkpoints=count,steps=target,draws=target*batch,teacher_draws=last['teacher_draws'],selfplay_draws=last['selfplay_draws'],selected_step=winner,optimizer_parameters=2 if stage=='warm' else 8,sampler_replayed=True))
    save(WORK/'checkpoint-audit.json',dict(passed=True,runs=out,checkpoint_files=30,files=files,safe_weights_only=True,all_cuda_rng_preserved=True))
    print(config.canonical(dict(event='phase6_checkpoint_audit',passed=True,files=30)),flush=True)

def load_json_row(row):
    body,digest=row;v=json.loads(body);require(config.digest(v)==digest,'Corrupt metadata');return v

def prior_artifact_audit():
    accepted=load(ROOT/'build/stage5-deeper-acceptance.json')
    require(accepted['passed'] and len(accepted['inputs'])==878,'Earlier acceptance witness differs')
    for path,digest in accepted['inputs'].items():require(sha(ROOT/path)==digest,'Earlier accepted artifact changed '+path)
    return dict(accepted_inputs=878,all_current_hashes_match=True,checked_after_phase6_production=True)

def benchmark_audit():
    implementation=load(ROOT/'build/phase6-implementation-v2.json');path=ROOT/implementation['roles']['benchmark'];spec=importlib.util.spec_from_file_location('phase6_benchmark_audited',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    require(m.implementation_manifest()==implementation,'Versioned producer implementation differs')
    plan=common.read_plan();report=load(ROOT/'build/phase6-benchmark.json');dbpath=m.DEFAULT_RUN/'work.sqlite3';casecount=gamecount=choices=0;buckets=defaultdict(list);games_by=defaultdict(list)
    with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)) as db:
        identity=load_json_row(db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone());require(identity==report['identity'] and identity['plan_sha256']==config.digest(plan),'Benchmark source identity differs')
        for p,digest in identity['sources'].items():require(sha(ROOT/p)==digest,'Benchmark source stale '+p)
        for index,body,digest in db.execute('SELECT idx,body,hash FROM cases ORDER BY idx'):
            require(index==casecount and config.digest(json.loads(body))==digest,'Case coverage/hash differs');case=json.loads(body);m.validate_case(case,index,plan);casecount+=1
            for ms,answers in case['measurements'].items():
                for name,v in answers.items():buckets[(case['cohort'],ms,name)].append(v)
        comparisons={c['id']:c for c in plan['arena_comparisons']}
        for comparison,pair,seat,body,digest in db.execute('SELECT comparison,pair,seat,body,hash FROM games ORDER BY comparison,pair,seat'):
            game=json.loads(body);require(config.digest(game)==digest,'Game hash differs');m.replay_game(game,comparisons[comparison],pair,seat,plan,m.Native())
            q=normalize_literal(game['start']);fallback=Counter()
            for action,saved in zip(game['moves'],game['decisions']):
                require(saved['actor']==q.controller,'Literal actor differs');legal=literal.legal_actions(q);require(action in legal,'Illegal literal arena choice')
                if saved['fallback']:require(action==legal[0],'Fallback is not first legal');fallback[saved['engine']]+=1
                q=literal.apply(q,action).position;choices+=1
            require(json_position(q)==game['final'] and dict(fallback)==game['fallbacks'],'Literal final/fallback differs')
            terminal=literal.normalize(q).terminal;margin=literal.evaluate(q)*(1 if seat==0 else -1)
            require(game['status']==('terminal' if terminal else 'unfinished') and game['candidate_final_margin']==(margin if terminal else None),'Terminal score differs')
            games_by[comparison].append(game);gamecount+=1
        require(casecount==80 and gamecount==192 and report==m.summarize(db,identity,plan),'Full benchmark publication differs')
    primary={}
    for name,games in games_by.items():
        margins=[];scores=[]
        for pair in range(24):
            pairgames=[g for g in games if g['pair']==pair]
            require(len(pairgames)==2 and {g['candidate_seat'] for g in pairgames}=={0,1},'Unpaired family')
            if all(g['status']=='terminal' for g in pairgames):margins.append(sum(g['candidate_final_margin'] for g in pairgames)/2);scores.append(sum(g['candidate_match_score'] for g in pairgames)/2)
        rng=random.Random(20261017);samples=sorted(sum(rng.choices(margins,k=len(margins)))/len(margins) for _ in range(4096)) if margins else []
        ci=[samples[102],samples[3993]] if samples else None;observed=report['arenas'][name]
        require(ci==observed['primary_margin_bootstrap_95_interval'] and (statistics.mean(margins) if margins else None)==observed['primary_paired_mean_final_margin_seeds'],'Independent paired bootstrap differs')
        gate=len(margins)==24 and not sum(sum(g['fallbacks'].values()) for g in games) and ci and ci[0]>0
        require(bool(gate)==observed['primary_strength_gate_passed'],'Primary gate differs');primary[name]=dict(mean_margin_seeds=observed['primary_paired_mean_final_margin_seeds'],interval=ci,gate_passed=bool(gate),complete_pairs=len(margins))
    require(report['default_promotion'] is False and report['energy_measured'] is False and report['sealed_final_test_consumed'] is False,'Unsupported release/energy/proof claim')
    return dict(cases=80,games=192,literal_choices=choices,primary=primary,full_source_and_cpu_identity=True,whole_wrapper_clock_checked=True,case_scalar_measurements=sum(len(v) for v in buckets.values()))

def full():
    started=time.monotonic();paths=[ROOT/p for p in SOURCES]+[Path(__file__),common.PLAN,common.REGISTRY,ROOT/'build/stage5-deeper-acceptance.json'];result={'passed':False,'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        result['tests']=collect();result['registry']=registry_audit();result['cohorts']=cohort_audit();result['selfplay']=selfplay_audit();result['models']=model_audit()
        child=subprocess.run([sys.executable,__file__,'checkpoints'],cwd=ROOT,capture_output=True,text=True);require(child.returncode==0,child.stdout+child.stderr);result['checkpoints']=load(WORK/'checkpoint-audit.json')
        result['benchmark']=benchmark_audit();result['prior_artifacts']=prior_artifact_audit();paths += [WORK/(n+'-tests.json') for n in GROUPS]+[WORK/'tests.json',WORK/'checkpoint-audit.json',common.WARM,common.UPDATE,common.SELFPLAY/'manifest.json',common.SELFPLAY/'work.sqlite3',ROOT/'build/phase6-benchmark.json',ROOT/'data/phase6-benchmark-v1/work.sqlite3',ROOT/'build/phase6-implementation-v2.json',ROOT/'congklak/phase6_training_v2.py',ROOT/'tests/phase6_versioned_pipeline_audit_v3.py',ROOT/'tools/benchmark-phase6-v2.py']
        paths += list((WORK/'cuda-selfplay-kills').glob('*'))
        paths += [p for p in (WORK/'warm-resource-failure.json',WORK/'warm-batch32-host-memory-failure.json',ROOT/'tests/phase6_cuda_selfplay_audit.py',ROOT/'tests/phase6_cuda_training_audit_v2.py',ROOT/'tests/phase6_warm_evidence_review.py') if p.exists()]
        paths += [ROOT/p for p in result['tests']['author_warm_recovery_evidence_review']['inputs']]
        paths += [ROOT/p for name in GROUPS for p in load(WORK/(name+'-tests.json'))['inputs']]
        result['inputs']=inputs(paths);result['passed']=True;result['sealed_target_bodies_read']=False;result['default_promotion']=False;result['energy_measured']=False
    except Exception as exc:result['error']=str(exc)
    result['duration_seconds']=time.monotonic()-started;save(WORK/'audit.json',result);print(config.canonical(dict(event='phase6_independent_full_audit',passed=result['passed'],error=result.get('error'))),flush=True)
    return 0 if result['passed'] else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['group','collect','full','checkpoints']);p.add_argument('--group',choices=list(GROUPS));a=p.parse_args()
    if a.operation=='group':require(a.group,'Missing group');return group(a.group)
    if a.operation=='collect':collect();return 0
    if a.operation=='checkpoints':checkpoint_worker();return 0
    return full()
if __name__=='__main__':raise SystemExit(main())
