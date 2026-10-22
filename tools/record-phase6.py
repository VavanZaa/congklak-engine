"""Bind completed Phase6 engineering, recovery and strength evidence using stdlib."""
from __future__ import annotations
from contextlib import closing
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import struct
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'build/phase6-acceptance.json'
RULES='c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e'
SOURCE_VALUE='29f90512e59ebaeb93eedf74d247684161e6b211a9b21640dc4f9333d213d69f'
WARM_RUN='data/training-phase6-warm-v1'
UPDATE_RUN='data/training-phase6-update-v1'
FILES=set()

def require(condition,message):
    if not condition:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def inside(name):
    path=(ROOT/name).resolve()
    require(path.is_relative_to(ROOT),'Evidence path escapes workspace: '+str(name));return path
def sha(name):
    path=inside(name);require(path.is_file(),'Missing required evidence: '+str(name))
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def pin(name):
    path=inside(name);require(path.is_file(),'Missing required evidence: '+str(name))
    name=path.relative_to(ROOT).as_posix();FILES.add(name);return name
def load(name):
    path=inside(name);pin(path)
    return json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+x)))
def current(mapping):
    require(isinstance(mapping,dict) and mapping,'Missing evidence input mapping')
    for name,expected in mapping.items():require(sha(name)==expected,'Stale evidence input: '+name);pin(name)
def passed(body,count,key='tests'):
    require(body.get('passed') is True and body.get(key)==count,'Incomplete acceptance group')
    require(all(body.get(k,0)==0 for k in ('errors','failures','skips','skipped')),'Acceptance has failures/errors/skips')
def methods(name,cls):
    tree=ast.parse(inside(name).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return {n.name for n in node.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
def open_db(name):
    path=inside(name);pin(path)
    return sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
def catalog_value(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone()
    require(row is not None,'Missing committed completion/identity: '+key)
    body=json.loads(row[0]);require(digest(body)==row[1],'Corrupt catalog metadata: '+key);return body
def finite(value,minimum=0):return type(value) in (int,float) and math.isfinite(value) and value>=minimum
def tree_files(folder):
    path=inside(folder);require(path.is_dir(),'Missing retained evidence directory: '+str(folder))
    for item in path.rglob('*'):
        if item.is_file() and not item.name.endswith(('-wal','-shm','.tmp')) and '__pycache__' not in item.parts:pin(item)
def model(name):
    envelope=load(name);body=envelope['model']
    require(set(envelope)=={'model','sha256'} and digest(body)==envelope['sha256'],'Damaged model envelope')
    require(body['schema']=='congklak-mlp-export-v4' and body['uses_oracle'] is False,'Unexpected model export/oracle use')
    shapes={'layers.0.weight':(64,127),'layers.0.bias':(64,), 'layers.2.weight':(64,64),'layers.2.bias':(64,),
            'layers.4.weight':(1,64),'layers.4.bias':(1,), 'policy.weight':(7,64),'policy.bias':(7,)}
    require(set(body['weights'])==set(shapes),'Model tensor ownership differs')
    def flatten(value,shape):
        require(isinstance(value,list) and len(value)==shape[0],'Model shape differs')
        if len(shape)>1:return [v for row in value for v in flatten(row,shape[1:])]
        require(all(type(x) in (int,float) and math.isfinite(x) for x in value),'Nonfinite model tensor');return value
    full=hashlib.sha256();value=hashlib.sha256();params=0
    for key in sorted(shapes):
        vals=flatten(body['weights'][key],shapes[key]);params+=len(vals);packed=key.encode()+struct.pack('<'+'f'*len(vals),*vals)
        full.update(packed)
        if key.startswith('layers.'):value.update(packed)
    require(params==12872 and full.hexdigest()==body['weights_sha256'],'Model parameter/digest differs')
    return body,value.hexdigest()

def production_training(run,stage,implementation,plan,initial):
    settings=plan['warm' if stage=='warm' else 'update'];target=settings['steps'];batch=settings['batch_size']
    body,value_digest=model(run+'/model.json');result=load(run+'/result.json');identity=body['identity']
    role='warm_trainer' if stage=='warm' else 'update_trainer'
    schema='phase6-policy-warmstart-and-selfplay-training-v1' if stage=='warm' else 'phase6-selfplay-deferred-cuda-training-v2'
    require(result['status']=='complete' and result['steps_completed']==result['optimizer_steps_target']==target and result['identity']==identity,'Production training incomplete')
    require(identity['stage']==stage and identity['schema']==schema and identity['source_sha256']==implementation['sources'][implementation['roles'][role]],'Wrong production trainer/source')
    require(identity['rules_digest']==RULES and identity['width']==64 and identity['input_count']==127 and identity['device']=='cuda','Production architecture/runtime differs')
    require(identity['amp'] is False and identity['oracle_training'] is False and identity['sealed_final_test_consumed'] is False and result['sealed_final_test_consumed'] is False,'Unsupported numeric/data claim')
    require(result['parameters']==12872 and result['trainable_parameters']==identity['trainable_parameters']==(455 if stage=='warm' else 12872),'Parameter ownership differs')
    for key in ('steps','batch_size','seed','lr','checkpoint_every'):require(identity[key]==settings[key],'Fixed training budget changed: '+key)
    current({name.replace('.','/')+'.py':checksum for name,checksum in identity['source_dependencies_sha256'].items()})
    teacher=load('data/deeper-9000-v2/manifest.json')
    require(identity['dataset_manifest_sha256']==digest(teacher) and identity['dataset_semantic_sha256']==teacher['semantic_sha256'] and identity['policy_temperature_seeds']==4. and identity['policy_perspective']=='Normalized root actor irrespective of successor controller','Teacher/policy lineage differs')
    initial_body,initial_value=model(initial)
    require(identity['initial_model_file_sha256']==sha(initial) and identity['initial_weights_sha256']==initial_body['weights_sha256'] and identity['initial_value_weights_sha256']==initial_value==SOURCE_VALUE,'Production ancestry differs')
    require(result['weights_sha256']==body['weights_sha256'] and result['value_weights_sha256']==value_digest,'Production model/result differs')
    require(result['draws']==target*batch and result['teacher_draws']==target*(batch if stage=='warm' else batch//2) and result['selfplay_draws']==(0 if stage=='warm' else target*batch//2),'Production draw counters differ')
    if stage=='warm':require(value_digest==SOURCE_VALUE,'Frozen warm value/trunk changed')
    else:
        manifest=load('data/phase6-selfplay-v1/manifest.json')
        require(identity['selfplay_manifest_sha256']==digest(manifest) and identity['selfplay_loader_sha256']==sha(implementation['roles']['selfplay']),'Update replay lineage differs')
        require(identity['warm_witness_implementation_sha256']==sha(implementation['roles']['warm_trainer']) and result['best_step']==target,'Update selection/warm witness differs')
    entries=[];scores=[];selected=None
    with closing(open_db(run+'/catalog.sqlite3')) as db:
        db.execute('BEGIN');require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt production catalog')
        rows=db.execute('SELECT body,hash FROM config').fetchall();require(len(rows)==1 and json.loads(rows[0][0])==identity and digest(identity)==rows[0][1],'Production catalog identity differs')
        rows=db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id').fetchall()
        require([row[0] for row in rows]==list(range(50,target+1,50)),'Missing or extra scheduled checkpoints')
        for step,name,checksum,text,bodyhash in rows:
            path=inside(run)/name;require(path.resolve().parent==inside(run),'Checkpoint path escapes run');require(sha(path)==checksum,'Damaged checkpoint bytes');pin(path)
            meta=json.loads(text);require(digest(meta)==bodyhash and meta['step']==step and meta['identity_sha256']==digest(identity),'Damaged checkpoint metadata')
            ce=meta['metrics']['policy']['uniform_stratum_teacher_cross_entropy'];require(finite(ce),'Invalid selection metric')
            scores.append((step,ce));entries.append((step,pin(path),checksum))
            if step==result['best_step']:selected=meta
    expected=min(scores,key=lambda row:row[1])[0] if stage=='warm' else target
    require(body['step']==result['best_step']==expected and selected['weights_sha256']==body['weights_sha256'],'Wrong scheduled/final export')
    require(result['selected_development']=={k:v for k,v in selected['metrics'].items() if k!='last_training_batch_loss'},'Selected metrics differ from committed checkpoint')
    verify_name='build/phase6-training-warm-verification.json' if stage=='warm' else 'build/phase6-training-selfplay-verification-v2.json'
    verification=load(verify_name);require(verification['passed'] and verification['stage']==stage and verification['checkpoints_verified']==len(entries) and verification['latest_step']==target and verification['selected_step']==expected,'Production verification missing')
    require(verification['trainer_sha256']==identity['source_sha256'] and verification['source_sha256']==sha(implementation['roles']['warm_tool' if stage=='warm' else 'update_tool']),'Production verification source changed')
    ordinary={k:v for k,v in result['selected_development']['finite_choice_horizon']['strata'].items() if k.endswith('/ordinary') and v['count']>=20}
    require(ordinary and all(finite(v['mae_seeds']) and finite(v['H_v1_mae_seeds']) for v in ordinary.values()),'No eligible ordinary H8 validation strata')
    return dict(model_file=run+'/model.json',weights_sha256=body['weights_sha256'],selected_step=expected,checkpoint_files=len(entries),teacher_draws=result['teacher_draws'],selfplay_draws=result['selfplay_draws'],ordinary_H8=ordinary,ordinary_H8_gate_passed=all(v['mae_seeds']<=v['H_v1_mae_seeds'] for v in ordinary.values()),checkpoint_bytes=entries),result

def unique_test_groups():
    owner=load('build/phase6-training-acceptance.json');passed(owner,6)
    require(owner['source_sha256']==sha('tests/phase6_training_acceptance.py') and owner['trainer_sha256']==sha('congklak/phase6_training.py'),'Warm acceptance source changed')
    unit=sorted(n for n in methods('tests/phase6_training_acceptance.py','Phase6TrainingAcceptance') if n[:7] in ('test_01','test_02','test_03','test_04'))
    require(len(unit)==4 and len(owner['records'])==2 and {r['kind'] for r in owner['records']}=={'precommit','preexport'},'Warm acceptance coverage differs')
    require(all(r['passed'] and r['actual_cuda_kill'] and r['full_state_bitwise'] and r['completed_bytes_and_mtimes_unchanged'] for r in owner['records']),'Warm reported recovery incomplete')
    warm_methods=['Phase6TrainingAcceptance.'+n for n in unit]+['author_CUDA_'+r['kind'] for r in owner['records']]
    mixed=load('build/phase6-selfplay-training-acceptance-v2.json');passed(mixed,3,'checks')
    require(mixed['trainer_sha256']==sha('congklak/phase6_training_v2.py') and mixed['probe_sha256']==sha('build/phase6-selfplay-training-acceptance-resume2-v2.py'),'Mixed acceptance source changed')
    require(mixed['warm_model_sha256']==sha('build/phase6-training-acceptance/serial-cuda-warm-fixture/model.json') and mixed['selfplay_manifest_sha256']==sha('build/phase6-training-selfplay-fixture-v1/manifest.json'),'Mixed acceptance fixture changed')
    require(mixed['production_training'] is False and mixed['all_workflow_artifacts_retained'] is True,'Mixed acceptance scope differs')
    names=[r['kind'] for r in mixed['records']];require(len(set(names))==3 and set(names)=={'precommit','preexport','independent_mixed_coefficients_and_target_routes'},'Mixed unique recovery coverage differs')
    for row in mixed['records'][:2]:require(all(row[k] is True for k in ('passed','actual_cuda_kill','whole_state_bitwise','all_checkpoint_states_bitwise','completed_bytes_and_mtimes_unchanged','fixed_final_export','orphan_bytes_and_mtimes_retained')),'Retained mixed recovery incomplete')
    require(mixed['records'][2]['observed_total']==mixed['records'][2]['expected_total']==6.25,'Mixed loss routing differs')
    tree_files(mixed['artifacts_path'])
    puct=load('build/phase6-puct-gpu-acceptance.json');passed(puct,8);current(puct['inputs'])
    require(puct['gpu_test_scheduled'] is True and puct['tested_model_sha256']==sha(puct['tested_model'])==sha(WARM_RUN+'/model.json'),'PUCT actual warm/GPU scope differs')
    puct_methods=sorted(methods('tests/phase6_puct_acceptance.py','Tests'));require(len(puct_methods)==len(puct['checks'])==8 and all(r['passed'] for r in puct['checks']),'PUCT coverage differs')
    updated_puct=load('build/phase6-puct-updated-gpu-acceptance.json');passed(updated_puct,8);current(updated_puct['inputs'])
    require(updated_puct['gpu_test_scheduled'] is True and updated_puct['tested_model_sha256']==sha(updated_puct['tested_model'])==sha(UPDATE_RUN+'/model.json'),'PUCT actual updated/GPU scope differs')
    require(updated_puct['inputs']==puct['inputs'] and updated_puct['checks']==puct['checks'],'Updated-model parity repeated a different suite')
    bench=load('build/phase6-benchmark-v2-acceptance.json');passed(bench,11);current(bench['inputs'])
    historical_source=load('build/phase6-benchmark-acceptance.json')
    require(sha('tests/phase6_benchmark_acceptance.py')==historical_source['inputs']['tests/phase6_benchmark_acceptance.py'],'Inherited benchmark contracts changed')
    shared=methods('tests/phase6_benchmark_acceptance.py','Tests')|methods('tests/phase6_benchmark_v2_acceptance.py','SharedContracts')
    bench_methods=['SharedContracts.'+n for n in sorted(shared)]+['Admission.'+n for n in sorted(methods('tests/phase6_benchmark_v2_acceptance.py','Admission'))]
    require(len(bench_methods)==len(bench['checks'])==11 and len({r['name'] for r in bench['checks']})==11 and all(r['passed'] for r in bench['checks']),'Guarded benchmark unique coverage differs')
    independent=load('build/phase6-audit/tests.json');passed(independent,14,'tests_run')
    require(independent['source_sha256']==sha('tests/phase6_audit_acceptance.py'),'Independent acceptance source changed')
    expected=methods('tests/phase6_audit_acceptance.py','Phase6IndependentAcceptance')
    expected.remove('test_09_cuda_warm_actual_kill_full_state_and_finished_stability')
    expected |= methods('tests/phase6_warm_evidence_review.py','Phase6AuthorWarmEvidenceReview')|methods('tests/phase6_cuda_selfplay_audit.py','Phase6CudaSelfplayAcceptance')
    covered=independent['covered_methods'];require(len(covered)==len(set(covered))==14 and set(covered)==expected,'Independent method coverage omits or duplicates checks')
    delivered=[]
    for group in independent['groups']:
        require(sha(group['path'])==group['sha256'],'Independent group report changed');body=load(group['path']);passed(body,len(group['methods']),'tests_run');current(body['inputs'])
        require(body['covered_methods']==group['methods'] and body['source_sha256']==independent['source_sha256'],'Independent group/source coverage changed');delivered.extend(group['methods'])
    require(len(delivered)==len(set(delivered))==14 and set(delivered)==expected and len(independent['groups'])==7,'Independent delivered coverage differs')
    narrow=independent['author_warm_recovery_evidence_review']
    require(narrow['raw_successful_kill_fixtures_retained'] is False and narrow['independent_raw_killed_state_recomparison'] is False and narrow['independent_live_warm_kill_passed'] is False and narrow['standalone_fixture_is_not_kill_proof'] is True and narrow['limitation'],'Warm review scope inaccurately expanded')
    cuda_update=independent['cuda_selfplay']
    require(cuda_update['steps']==6 and cuda_update['raw_fresh_and_resumed_states_retained'] is True and len(cuda_update['checks'])==2 and all(all(row[k] is True for k in ('actual_cuda_hard_kill','whole_state_bitwise_equal','finished_bytes_mtime_stable','checksum_valid_export_refused')) for row in cuda_update['checks']),'Independent live update recovery incomplete')
    cuda_games=independent['cuda_selfplay_recovery']
    require(cuda_games['model_file_sha256']==sha(WARM_RUN+'/model.json') and len(cuda_games['tests'])==2 and all(all(row[k] is True for k in ('actual_cuda_hard_kill','committed_bytes_preserved','literal_terminal_replay','finished_bytes_mtime_stable')) for row in cuda_games['tests']),'Independent live CUDA selfplay recovery incomplete')
    require(independent['benchmark_case_kill']['actual_hard_kill'] is True and independent['benchmark_case_kill']['forged_reference_refused'] is True and all(independent['benchmark_game_kill'][k] is True for k in ('actual_hard_kill','all_games_literal_replayed','finished_bytes_mtime_stable','forged_outcome_refused')),'Independent case/arena recovery incomplete')
    groups={'author_warm_training':warm_methods,'author_mixed_training':names,'puct_actual_warm_CPU_CUDA':puct_methods,'guarded_benchmark':bench_methods,'independent_auditor':covered}
    return groups,independent,narrow

def speed_and_resources(implementation,benchmark):
    speed=load('build/phase6-inference-speed-results.json');require(speed['complete'] is True,'Incomplete inference sweep')
    identity=speed['identity'];require(identity['source_sha256']==sha(implementation['roles']['speed']) and identity['batch_sizes']==[1,2,4,8,16,32] and identity['blocks']==7 and identity['repetitions']==128 and identity['warmup_calls_per_unit']==4,'Speed runtime/geometry differs')
    require(identity['cpu']==benchmark['identity']['cpu'] and set(identity['engines'])==set(speed['backends'])=={'native','cuda'},'Speed and strength CPU/backend declarations differ')
    require(identity['timing_sha256']==sha('congklak/timing_v2.py') and identity['store_sha256']==sha('congklak/phase6_store.py'),'Speed supporting sources differ')
    for value in identity['engines'].values():require(value['model_file_sha256']==sha(WARM_RUN+'/model.json'),'Speed model differs')
    units={}
    with closing(open_db(speed['ledger']+'/work.sqlite3')) as db:
        db.execute('BEGIN');require(catalog_value(db,'identity')==identity,'Speed ledger identity differs')
        for kind,backend,batch,block,text,checksum in db.execute('SELECT kind,backend,batch,block,body,hash FROM units'):
            body=json.loads(text);require(digest(body)==checksum,'Damaged speed unit');units[(kind,backend,batch,block)]=body
        expected={('startup',b,0,0) for b in ('native','cuda')}|{('parity',b,n,0) for b in ('native','cuda') for n in identity['batch_sizes']}|{('micro',b,n,k) for b in ('native','cuda') for n in identity['batch_sizes'] for k in range(7)}
        require(set(units)==expected and len(units)==98,'Speed unit coverage differs')
        require(catalog_value(db,'phase6_inference_complete')==dict(units=98,report_sha256=digest(speed)),'Speed completion/report differs')
    def summary(values):return dict(median=statistics.median(values),min=min(values),max=max(values),samples=len(values))
    for backend in ('native','cuda'):
        require(speed['backends'][backend]['startup']==units[('startup',backend,0,0)],'Speed startup summary differs')
        for size in identity['batch_sizes']:
            row=speed['backends'][backend]['batches'][str(size)];parity=units[('parity',backend,size,0)]
            require(row['parity']==parity and parity['passed'] is True and parity['roots']==64 and 0<=parity['max_value_error_seeds']<=1e-4 and 0<=parity['max_logit_error']<=1e-5,'Inference parity tolerance failed')
            values=[units[('micro',backend,size,k)] for k in range(7)]
            require(all(finite(v['whole_call_ns'],1) and v['position_ns']==v['whole_call_ns']/size and v['repetitions']==128 and finite(v['checksum'],-float('inf')) for v in values),'Speed sample invalid')
            for key in ('whole_call_ns','position_ns'):require(row[key]==summary([v[key] for v in values]),'Speed time summary differs')
            require(row['positions_per_second']==summary([1e9/v['position_ns'] for v in values]),'Speed throughput summary differs')
            for key,observed in row['timing_mean_ms'].items():require(observed==summary([v['timing_mean_ms'][key] for v in values]) and all(finite(v['timing_mean_ms'][key]) for v in values),'Speed component summary differs')
            require(speed['native_to_cuda_whole_call_ratio'][str(size)]==summary([units[('micro','native',size,k)]['whole_call_ns']/units[('micro','cuda',size,k)]['whole_call_ns'] for k in range(7)]),'Speed adapter ratio differs')
    resources={}
    for kind,report in (('speed',speed),('benchmark',benchmark)):
        body=load('build/phase6-'+kind+'-resources.json');ident=body['identity']
        require(ident['producer']==kind and ident['producer_sha256']==sha(implementation['roles'][kind]) and ident['wrapper_sha256']==sha(implementation['roles']['resource_wrapper']) and ident['implementation_sha256']==digest(implementation),'Resource producer identity differs')
        require(body['report_file_sha256']==sha(implementation['artifacts'][kind+'_output']) and body['energy_measured'] is False and finite(body['elapsed_seconds_including_module_load'],.000001),'Resource scope/report differs')
        expected_models={'warm':sha(WARM_RUN+'/model.json')}
        if kind=='benchmark':expected_models['updated']=sha(UPDATE_RUN+'/model.json')
        require(ident['model_file_sha256']==expected_models,'Resource model differs')
        require(body['host'] and all(type(v) is int and v>=0 for v in body['host'].values()) and body['cuda_allocator'] and 0<=body['cuda_allocator']['peak_allocated_bytes']<=body['cuda_allocator']['peak_reserved_bytes']<12*1024**3,'Scoped resource measurements missing/exceeded')
        resources[kind]=body
    return speed,resources

def build_report():
    FILES.clear();pin(__file__)
    prior=load('build/stage5-deeper-acceptance.json');require(prior['passed'] and len(prior['inputs'])==878,'Prior Stage5 acceptance changed');current(prior['inputs'])
    implementation=load('build/phase6-implementation-v2.json');require(implementation['schema']=='phase6-implementation-v2' and implementation['rules_digest']==RULES and implementation['sha256']==digest({k:v for k,v in implementation.items() if k!='sha256'}),'Implementation mapping damaged');current(implementation['sources']);current(implementation['science_inputs'])
    admission=load('build/phase6-implementation-admission.json');require(admission['passed'] and admission['implementation_file_sha256']==sha('build/phase6-implementation-v2.json') and admission['implementation_sha256']==implementation['sha256'] and admission['torch_imported'] is False,'Frozen implementation admission differs')
    plan=load('build/phase6-plan.json');require(plan['rules_digest']==RULES and len(plan['cases'])==80 and len(plan['arena_starts'])==24 and len(plan['selfplay_starts'])==32 and plan['bootstrap_seed']==20261017 and plan['bootstrap_samples']==4096,'Frozen science geometry differs')
    warm,warm_result=production_training(WARM_RUN,'warm',implementation,plan,plan['source_model'])
    updated,updated_result=production_training(UPDATE_RUN,'selfplay',implementation,plan,WARM_RUN+'/model.json')
    groups,independent,narrow=unique_test_groups()
    audit=load('build/phase6-audit/audit.json');require(audit['passed'] and audit['tests']==independent,'Independent full audit incomplete');current(audit['inputs'])
    dashboard_review=load('build/phase6-audit/dashboard-review.json');current(dashboard_review['inputs'])
    require(all(dashboard_review[key] is True for key in ('passed','actual_live_process_identity_verified','json_list_matches_live_tuple','exited_or_reused_pid_detected','observer_preserves_log_bytes_and_mtimes','completed_work_does_not_hide_failed_strength_gates')),'Independent progress observer review incomplete')
    require(audit['prior_artifacts']['accepted_inputs']==878 and audit['prior_artifacts']['all_current_hashes_match'] is True,'Prior exhaustive audit missing')
    require(audit['selfplay']['games']==audit['selfplay']['terminal_games']==64 and audit['selfplay']['literal_decisions']==2154 and audit['selfplay']['retained_rows']==audit['selfplay']['cached_features_rederived']==623 and audit['selfplay']['families_excluded']==20 and audit['selfplay']['families_admitted']==12,'Selfplay literal provenance audit incomplete')
    checkpoints=audit['checkpoints'];require(checkpoints['passed'] and checkpoints['checkpoint_files']==len(checkpoints['files'])==30 and checkpoints['safe_weights_only'] and checkpoints['all_cuda_rng_preserved'] and all(r['sampler_replayed'] for r in checkpoints['runs']),'Independent tensor/sampler checkpoint audit incomplete');current(checkpoints['files'])
    require({name:checksum for _,name,checksum in warm['checkpoint_bytes']+updated['checkpoint_bytes']}==checkpoints['files'],'Producer/independent checkpoint manifests differ')
    require(audit['benchmark']['cases']==80 and audit['benchmark']['games']==192 and audit['benchmark']['case_scalar_measurements']==800 and audit['benchmark']['whole_wrapper_clock_checked'] is True,'Full benchmark audit incomplete')
    require(audit['models']['warm_six_value_tensors_unchanged'] is True and all(r['roots']==1000 and r['literal_feature_and_double_forward'] for r in audit['models']['models']) and len(audit['models']['models'])==2,'Independent model forward audit incomplete')
    audited_models={row['model']:row for row in audit['models']['models']}
    require(set(audited_models)=={'warm','updated'},'Independent development regression model set differs')
    for name,selection,result in (('warm',warm,warm_result),('updated',updated,updated_result)):
        row=audited_models[name];published=result['selected_development']['finite_choice_horizon']
        require(row['literal_development_regression'] is True and row['fp32_to_literal_MAE_tolerance_seeds']==1e-4 and row['ordinary_H8_gate_passed']==selection['ordinary_H8_gate_passed'],'Independent ordinary preservation verification incomplete')
        require(row['file_sha256']==sha(selection['model_file']) and row['weights_sha256']==selection['weights_sha256'],'Independent regression model differs')
        require(set(row['strata'])==set(published['strata']) and abs(row['mae_seeds']-published['mae_seeds'])<=1e-4 and row['H_v1_mae_seeds']==published['H_v1_mae_seeds'],'Independent overall regression differs')
        for stratum,metric in row['strata'].items():
            target=published['strata'][stratum]
            require(metric['count']==target['count'] and metric['H_v1_mae_seeds']==target['H_v1_mae_seeds'] and abs(metric['mae_seeds']-target['mae_seeds'])<=1e-4,'Independent per-stratum regression differs: '+stratum)
    require(audit['energy_measured'] is False and audit['default_promotion'] is False and audit['sealed_target_bodies_read'] is False,'Unsupported audit claim')
    cpu=load('build/phase6-cpu-producer-equivalence.json');require(cpu['passed'] and cpu['production'] is False and cpu['cuda_initialized'] is False and cpu['all_checkpoint_steps']==[5,10] and cpu['model_optimizer_sampler_python_torch_cpu_best_bitwise'] and cpu['export_model_exact_except_declared_identity'] and cpu['selected_development_exact'],'Source-equivalent CPU evidence incomplete')
    require(cpu['v1_trainer_sha256']==sha('congklak/phase6_training.py') and cpu['v2_trainer_sha256']==sha('congklak/phase6_training_v2.py') and cpu['probe_sha256']==sha('build/phase6-cpu-producer-equivalence.py'),'CPU equivalence source changed');tree_files(cpu['artifacts_path'])
    require(cpu['fixture_warm_model_file_sha256']==sha('build/phase6-training-acceptance/cpu-warm-fixture-v1/model.json') and cpu['fixture_selfplay_manifest_file_sha256']==sha('build/phase6-cpu-equivalence-selfplay-fixture-v1/manifest.json'),'CPU equivalence fixture changed')
    benchmark=load('build/phase6-benchmark.json');require(benchmark['cases']==80 and benchmark['games']==192 and benchmark['identity']['schema']=='phase6-benchmark-v2' and benchmark['identity']['implementation']==implementation and benchmark['identity']['plan_sha256']==digest(plan),'Benchmark incomplete/source changed');current(benchmark['identity']['sources'])
    with closing(open_db('data/phase6-benchmark-v1/work.sqlite3')) as db:
        db.execute('BEGIN');require(catalog_value(db,'identity')==benchmark['identity'] and catalog_value(db,'phase6_benchmark_complete')==dict(cases=80,games=192,report_sha256=digest(benchmark)),'Benchmark completion/report differs')
        require(db.execute('SELECT COUNT(*) FROM cases').fetchone()[0]==80 and db.execute('SELECT COUNT(*) FROM games').fetchone()[0]==192,'Benchmark durable coverage differs')
    speed,resources=speed_and_resources(implementation,benchmark)
    manifest=load('data/phase6-selfplay-v1/manifest.json');require(manifest['complete'] is True and manifest['training_admission']=='ready' and manifest['summary']['targets_averaged'] is False and manifest['summary']['unit_row_weights'] is True,'Replay admission/weighting incomplete')
    for name in ('docs/PHASE6_RESULTS.md','docs/PHASE6_INDEPENDENT_AUDIT.md','docs/PHASE6_SCOPE_AND_ENGINE_READINESS.md','docs/PHASE6_RUN_GUIDE.md','tools/phase6-dashboard.html','congklak/phase6_progress.py'):pin(name)
    # Production opening evidence is supplied by the parent only after the
    # recovered recommendations and their durable source identities exist.
    opening=load('build/phase6-analysis-recovery.json')
    require(opening['schema']=='phase6-analysis-recovery-v1' and all(opening[key] is True for key in ('passed','retained_prior_after_kill','fresh_semantic_equal','completed_bytes_mtimes_unchanged')),'Production analyzer recovery demo incomplete')
    current(opening['inputs'])
    require(UPDATE_RUN+'/model.json' in opening['inputs'] and opening['inputs'][UPDATE_RUN+'/model.json']==sha(UPDATE_RUN+'/model.json') and 'congklak/phase6_analysis.py' in opening['inputs'],'Production analyzer recovery uses wrong model/source')
    for folder in (WARM_RUN,UPDATE_RUN,'data/phase6-selfplay-v1','data/phase6-analysis-demo','build/phase6-audit','build/phase6-training-acceptance','build/phase6-training-selfplay-fixture-v1','build/phase6-cpu-equivalence-selfplay-fixture-v1'):tree_files(folder)
    for pattern in ('congklak/phase6*.py','tools/phase6*.py','tools/benchmark-phase6*.py','tests/phase6*.py','build/phase6*.json','build/phase6*.py'):
        for path in ROOT.glob(pattern):
            # Acceptance and a later idempotence witness derive from this
            # manifest; including those would create a recursive dependency.
            if path!=OUTPUT and not path.name.startswith('phase6-acceptance-'):pin(path)
    for path in (ROOT/'build/phase6-progress').glob('run-*.json'):
        body=json.loads(path.read_text(encoding='utf-8'))
        if body.get('state') in ('complete','failed') and not any('record-phase6.py' in str(part) for part in body.get('command',[])):pin(path);pin(body['log'])
    counts={name:len(names) for name,names in groups.items()};require(counts==dict(author_warm_training=6,author_mixed_training=3,puct_actual_warm_CPU_CUDA=8,guarded_benchmark=11,independent_auditor=14),'Unique test counts differ')
    total=sum(counts.values());require(total==42,'Aggregate test coverage differs')
    ordinary={'warm':warm['ordinary_H8_gate_passed'],'updated':updated['ordinary_H8_gate_passed']};primary=benchmark['both_baseline_primary_gates']
    require(set(primary)=={'warm','updated'} and all(type(v) is bool for v in primary.values()),'Arena baseline gate types differ')
    for comparison in plan['arena_comparisons']:
        name=comparison['id'];arena=benchmark['arenas'][name];checked=audit['benchmark']['primary'][name]
        require(arena['pairs']==24 and arena['terminal_games']+arena['unfinished_games']==48 and checked['mean_margin_seeds']==arena['primary_paired_mean_final_margin_seeds'] and checked['interval']==arena['primary_margin_bootstrap_95_interval'] and checked['gate_passed']==arena['primary_strength_gate_passed'],'Independent arena outcome/gate differs')
    for candidate in ('warm','updated'):require(primary[candidate] is all(benchmark['arenas'][candidate+'_cuda_vs_'+baseline]['primary_strength_gate_passed'] for baseline in ('H_v1','H8_relay_CPU')),'Both-baseline gate differs')
    combined={k:ordinary[k] and primary[k] for k in ordinary}
    return dict(schema='phase6-acceptance-v1',passed=True,engineering_passed=True,tests_run=total,groups=counts,covered_checks=groups,failures=0,errors=0,skips=0,
                independent_audit_complete=True,prior_inputs_preserved=878,training_runs=2,checkpoint_files_verified=30,selfplay_games=64,selfplay_decisions=2154,selfplay_training_rows=623,
                position_cases=80,arena_games=192,inference_speed_units=98,ordinary_H8_gates=ordinary,both_baseline_primary_gates=primary,combined_candidate_gates=combined,
                selected={'warm':warm,'updated':updated},arenas=benchmark['arenas'],resources=resources,
                supporting_verifications={'updated_model_PUCT_CPU_CUDA':dict(passed=True,repeated_checks=8,additional_unique_test_count=0,artifact='build/phase6-puct-updated-gpu-acceptance.json',model_file_sha256=sha(UPDATE_RUN+'/model.json')),
                                          'independent_literal_development_regression':dict(passed=True,models=2,roots_per_model=1000,MAE_tolerance_seeds=1e-4,additional_unique_test_count=0),
                                          'independent_dashboard_review':dict(passed=True,additional_unique_test_count=0,artifact='build/phase6-audit/dashboard-review.json'),
                                          'production_checkpoints':dict(passed=True,files=30,additional_unique_test_count=0),'source_equivalent_CPU_producers':dict(passed=True,additional_unique_test_count=0)},
                evidence_scope='42 unique executed checks across five suites; production artifact verification and CPU equivalence are supporting evidence, not additional test counts. Independent warm evidence review is narrower than a live independent warm kill.',
                author_warm_recovery_review=narrow,author_warm_raw_successful_kill_fixtures_retained=False,independent_live_warm_recovery_verified=False,
                recovery_scope='Retained author mixed-update states and independent live selfplay/update/case/arena kills; production opening recommendations recover through durable publications. GPU different-batch-shape bitwise continuation is not promised.',
                failed_resource_reports_and_probe_diagnostics_retained=True,sealed_target_bodies_read=False,sealed_final_test_consumed=False,energy_measured=False,full_game_solved=False,default_promotion=False,
                promotion='H_v1 remains default; no automatic promotion from this development pilot',
                inputs={name:sha(name) for name in sorted(FILES)})

def publish(body):
    encoded=canonical(body)+'\n'
    if OUTPUT.exists() and OUTPUT.read_text(encoding='utf-8')==encoded:return False
    temp=OUTPUT.with_name(OUTPUT.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,OUTPUT)
    finally:
        if temp.exists():temp.unlink()
    return True
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    try:
        body=build_report();require('torch' not in sys.modules,'Recorder imported Torch')
        rewritten=False if args.check_only else publish(body)
        print(canonical(dict(event='phase6_acceptance_checked' if args.check_only else 'phase6_acceptance_recorded',passed=True,tests_run=body['tests_run'],groups=body['groups'],ordinary_H8_gates=body['ordinary_H8_gates'],both_baseline_primary_gates=body['both_baseline_primary_gates'],files_rewritten=rewritten,torch_imported=False)),flush=True);return 0
    except Exception as exc:
        print(canonical(dict(event='phase6_acceptance_refused',passed=False,error=str(exc),acceptance_written=False,torch_imported='torch' in sys.modules)),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
