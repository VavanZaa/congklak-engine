"""Publish the fixed-policy value-head experiment only from complete audited evidence."""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
from contextlib import closing
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,value_common as common
from congklak.native import Native,flatten
require=common.require
INPUTS={}


def pin(path):
    path=(ROOT/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(),'Missing acceptance input: '+str(path))
    name=path.relative_to(ROOT).as_posix();digest=common.sha(path);INPUTS[name]=digest;return digest


def load(path):
    pin(path);return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def current(mapping):
    require(isinstance(mapping,dict) and bool(mapping),'Evidence has no immutable input pins')
    for name,digest in mapping.items():require(pin(name)==digest,'Evidence input changed: '+name)


def module(name,path):
    pin(path);spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def methods(path):
    pin(path);tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'));result=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            result.extend((node.name,item.name) for item in node.body if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)) and item.name.startswith('test_'))
    require(len(result)==len(set(result)) and len({name for _,name in result})==len(result),'Duplicated source methods: '+path)
    return result


def execution_source(group,report):
    """Check mixed executed coordinators without relabeling a prior run."""
    coordinator='tools/value-audit.py';provenance=report.get('executed_source_provenance')
    pin(coordinator)
    if provenance is None:
        require(report['inputs'].get(coordinator)==pin(coordinator),'Direct group does not bind its current executed coordinator')
        return dict(coordinator_path=coordinator,coordinator_sha256=pin(coordinator),tests_rerun=False,provenance='Direct execution, not derived')
    require(group in ('plan','targets','training'),'Unregistered historical execution group')
    original_path=f'build/value-audit/executed-v1/{group}-tests.json';snapshot='tools/value-audit-executed-v1.py';test_source='tests/value_audit_acceptance.py';helper='tools/value-audit-bindings.py'
    original=load(original_path);require('executed_source_provenance' not in original,'Executed original was itself derived')
    old_sha=original['inputs'][coordinator]
    require(pin(snapshot)==old_sha,'Retained coordinator bytes differ from original executed binding')
    require(original['inputs'][test_source]==pin(test_source),'Executed test source changed')
    expected_provenance=dict(original_report_path=original_path,original_report_sha256=pin(original_path),
        original_coordinator_path=coordinator,original_coordinator_sha256=old_sha,
        coordinator_snapshot_path=snapshot,coordinator_snapshot_sha256=old_sha,
        unchanged_test_source_path=test_source,unchanged_test_source_sha256=pin(test_source),tests_rerun=False,
        derivation='Retained identical executed coordinator bytes; original outcomes, methods, evidence, durations and all other input bindings unchanged.')
    expected_inputs={name:digest for name,digest in original['inputs'].items() if name!=coordinator}
    expected_inputs.update({snapshot:old_sha,original_path:pin(original_path),helper:pin(helper)})
    require(provenance==expected_provenance and report=={**original,'inputs':expected_inputs,'executed_source_provenance':expected_provenance},'Historical execution derivation changed outcomes, methods, evidence, duration or unrelated bindings')
    # These routines were not called by the three retained groups. Every other
    # function, import and group definition must be structurally unchanged.
    permitted={'collect','production_training_worker','model_audit'}
    old_tree=ast.parse((ROOT/snapshot).read_text(encoding='utf-8-sig'));new_tree=ast.parse((ROOT/coordinator).read_text(encoding='utf-8-sig'))
    old_functions={n.name:ast.dump(n,include_attributes=False) for n in old_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_functions={n.name:ast.dump(n,include_attributes=False) for n in new_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    require(set(old_functions)==set(new_functions) and {name for name in old_functions if old_functions[name]!=new_functions[name]}==permitted,'Coordinator correction exceeds its declared three-function scope')
    old_tree.body=[n for n in old_tree.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in permitted)]
    new_tree.body=[n for n in new_tree.body if not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in permitted)]
    require(ast.dump(old_tree,include_attributes=False)==ast.dump(new_tree,include_attributes=False),'Previously executed group or state-comparison code changed')
    return expected_provenance


def group(path,source,count):
    body=load(path);current(body['inputs']);actual=body.get('tests_run',body.get('tests'))
    require(body.get('passed') is True and actual==count and body.get('errors')==body.get('failures')==0 and body.get('skips',body.get('skipped',0))==0,'Acceptance group incomplete: '+path)
    covered=body.get('covered_methods',body.get('methods'));expected=methods(source)
    require(isinstance(covered,list) and len(covered)==len(set(covered))==count==len(expected),'Missing or duplicate executed methods: '+path)
    require({item.rsplit('.',1)[-1] for item in covered}=={name for _,name in expected},'Executed/source method coverage differs: '+path)
    ids=[source+'::'+cls+'.'+name for cls,name in expected]
    return body,ids


def database(path):
    pin(path);db=sqlite3.connect((ROOT/path).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('BEGIN')
    require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt acceptance SQLite: '+path);return db


def meta(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone();require(row is not None,'Missing committed metadata: '+key)
    body=json.loads(row[0]);require(config.digest(body)==row[1],'Corrupt committed metadata: '+key);return body


def training(plan):
    from congklak.value_learning import witness
    body=witness();proof=load('build/value-model-witness.json');require(body==proof,'Training model witness changed');current(proof['inputs'])
    require(proof['passed'] is True and proof['selected_step']==1500 and proof['all_optimizer_sampler_rng_states_verified'] is True and proof['final_export_equals_checkpoint_tensors'] is True,'Genuine final training proof incomplete')
    require(proof['all_frozen_tensors_bit_exact'] is True and proof['frozen_parameter_count']==12807 and proof['trainable_parameter_count']==65,'Frozen-policy tensor ownership proof incomplete')
    require(proof['model_file_sha256']==pin(plan['models']['refit']),'Wrong final refit export')
    require(len(proof['checkpoints'])==proof['checkpoints_verified'] and proof['checkpoints_verified']>=30,'Incomplete actual checkpoint proof')
    with closing(database('data/value-train-v1/catalog.sqlite3')) as db:
        rows=db.execute('SELECT step,name,sha,body,hash FROM checkpoints ORDER BY id').fetchall();steps=[]
        for row in rows:
            value=json.loads(row['body']);steps.append(row['step'])
            require(pin('data/value-train-v1/'+row['name'])==row['sha'] and config.digest(value)==row['hash'],'Training checkpoint catalog/file differs')
        require(len(rows)==proof['checkpoints_verified'] and set(range(50,1501,50))<=set(steps) and steps==sorted(set(steps)) and steps[-1]==1500,'Training checkpoint schedule differs')
    pin('data/value-train-v1/result.json');return proof


def evaluator_identities(values,plan):
    require(set(values)==set(common.PROFILES),'Registered evaluator identities incomplete')
    for name,value in values.items():
        path=plan['models'][plan['profiles'][name]['model']];envelope=load(path)
        require(value['model_file_sha256']==pin(path) and value['model_weights_sha256']==envelope['model']['weights_sha256'],'Evaluator uses wrong model')
        require(value['source_sha256']==pin('congklak/puct_cpu_inference.py') and value['dll_sha256']==pin('build/congklak_puct_cpu.dll') and value['build_record_sha256']==pin('build/puct-cpu-build.json'),'Evaluator build/source differs')
        require(value['rules_digest']==config.RULES_DIGEST and value['cache']['capacity']==0 and value['max_batch']==32,'Evaluator experiment bounds differ')
        current(value['build_inputs']);current(value['dependencies'])


def benchmark(plan,manifest):
    bench=module('value_recorder_benchmark','tools/benchmark-value.py');implementation,witness=bench.admit(plan)
    require(implementation==manifest,'Benchmark admission implementation differs')
    body=load('build/value-benchmark.json');resources=load('build/value-benchmark-resources.json');identity=body['identity']
    require(body['schema']=='value-benchmark-results-v1' and body['cases']==80 and body['games']==192 and identity['implementation']==manifest and identity['plan_sha256']==config.digest(plan),'Benchmark campaign/identity incomplete')
    require(identity['admission']==witness and identity['implementation_file_sha256']==pin('build/value-implementation.json'),'Benchmark current model admission differs')
    require(identity['baseline_dll_sha256']==pin('build/congklak_native.dll') and identity['source_model']['file_sha256']==pin(plan['source_model']) and identity['source_model']['dll_sha256']==pin('build/congklak_neural_relay.dll'),'Baseline engine identity differs')
    evaluator_identities(identity['evaluators'],plan)
    require(identity['cpu']['selected']==plan['cpu_declared'] and identity['cpu']['mask']==1<<plan['cpu_declared']['logical'] and identity['cpu']['power_settings_changed'] is False and identity['cpu']['priority_changed'] is False,'Benchmark declared CPU differs')
    bench.validate_resources(resources,ROOT/'build/value-benchmark.json',manifest)
    literal=module('value_recorder_literal','tests/reference.py');counts=Counter(scalar_measurements=0,literal_choices=0,terminal_games=0,unfinished_games=0,fallbacks=0);games=[]
    with closing(database('data/value-benchmark-v1/work.sqlite3')) as db:
        require(meta(db,'identity')==identity,'Benchmark report/journal identity differs')
        cases,n=bench.validate_saved_units(db,plan,Native());require((cases,n)==(80,192),'Benchmark journal unit coverage differs')
        startup=db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone();require(startup is not None and config.digest(json.loads(startup[0]))==startup[1] and db.execute('SELECT COUNT(*) FROM setup').fetchone()[0]==1,'Missing or extra benchmark startup witness')
        require(meta(db,'value_benchmark_complete')==dict(cases=80,games=192,report_sha256=config.digest(body),resource_file_sha256=pin('build/value-benchmark-resources.json')),'Benchmark durable completion witness differs')
        counts['scalar_measurements']=sum(len(engines) for row in db.execute('SELECT body FROM cases') for engines in json.loads(row[0])['measurements'].values());require(counts['scalar_measurements']==640,'Missing registered scalar measurements')
        for text, in db.execute('SELECT body FROM games ORDER BY idx'):
            game=json.loads(text);q=literal.Position(7,tuple(flatten(game['start'])),game['start']['controller']);fallbacks=Counter()
            comp=next(c for c in plan['arena_comparisons'] if c['id']==game['comparison'])
            for action,row in zip(game['moves'],game['decisions']):
                normal=literal.normalize(q);q=literal.Position(7,q.counts,normal.controller);legal=literal.legal_actions(q)
                require(not normal.terminal and row['actor']==q.controller and action in legal,'Literal arena controller/action differs')
                expected=comp['candidate'] if q.controller==game['candidate_seat'] else comp['opponent'];require(row['engine']==expected,'Literal engine seat assignment differs')
                if row['fallback']:require(action==legal[0],'Illegal fallback order');fallbacks[expected]+=1
                q=literal.apply(q,action).position;counts['literal_choices']+=1
            final=dict(pits=[list(q.counts[:7]),list(q.counts[8:15])],stores=[q.counts[7],q.counts[15]],controller=q.controller)
            terminal=literal.normalize(q).terminal;margin=literal.evaluate(q)*(1 if game['candidate_seat']==0 else -1)
            require(game['final']==final and game['status']==('terminal' if terminal else 'unfinished') and game['candidate_final_margin']==(margin if terminal else None) and game['fallbacks']==dict(fallbacks),'Literal arena final payoff differs')
            counts['terminal_games' if terminal else 'unfinished_games']+=1;counts['fallbacks']+=sum(fallbacks.values());games.append(game)
        require(body==bench.summarize(db,identity,plan),'Benchmark summary does not reproduce from durable units')
    # Recompute paired means, bootstrap intervals and secondary contrasts here.
    paired={};primary={}
    for comp in plan['arena_comparisons']:
        rows=[g for g in games if g['comparison']==comp['id']];byfamily={};fallbacks=sum(sum(g['fallbacks'].values()) for g in rows)
        for pair in range(24):
            group=[g for g in rows if g['pair']==pair];require(len(group)==2 and {g['candidate_seat'] for g in group}=={0,1},'Missing paired seat games')
            if all(g['status']=='terminal' for g in group):byfamily[pair]=statistics.mean(g['candidate_final_margin'] for g in group)
        values=list(byfamily.values());rng=random.Random(plan['bootstrap_seed'])
        draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(plan['bootstrap_samples'])) if values else []
        ci=[draws[int(.025*len(draws))],draws[int(.975*len(draws))]] if draws else None
        mean=statistics.mean(values) if values else None;gate=bool(len(values)==24 and not fallbacks and ci and ci[0]>0);actual=body['arenas'][comp['id']]
        require(actual['primary_paired_mean_final_margin_seeds']==mean and actual['primary_margin_bootstrap_95_interval']==ci and actual['primary_strength_gate_passed']==gate,'Recomputed paired interval/gate differs')
        paired[comp['id']]=byfamily;primary[comp['id']]=dict(mean_margin_seeds=mean,interval=ci,gate_passed=gate,complete_pairs=len(values),fallbacks=fallbacks)
    contrasts={}
    for base in common.BASELINES:
        t=paired['value_refit_vs_'+base];c=paired['policy_control_vs_'+base];matched=sorted(set(t)&set(c))
        contrasts[base]=dict(treatment='value_refit',control='policy_control',expected_families=24,complete_matched_families=len(matched),paired_mean_seed_margin_difference=statistics.mean(t[p]-c[p] for p in matched) if len(matched)==24 else None,confidence_interval=None,primary_gate=False)
    gates={name:all(primary[name+'_vs_'+base]['gate_passed'] for base in common.BASELINES) for name in common.PROFILES}
    require(body['secondary_factor_contrasts']==contrasts and body['both_baseline_primary_gates']==gates,'Recomputed secondary contrasts or both-baseline gates differ')
    require(all(body[key] is False for key in ('default_promotion','sealed_final_test_consumed','energy_measured')),'Unsupported promotion/sealed/energy claim')
    return body,dict(counts),primary,resources


def independent():
    body=load('build/value-audit/audit.json');require(body['passed'] is True,'Independent full audit incomplete');current(body['inputs'])
    require(body['prior']==dict(prior_inputs=1647,all_current_hashes_match=True),'Independent prior preservation differs')
    require(body['new_value_cuda_actual_kill_verified'] is True and body['unrelated_prior_warm_kill_claimed'] is False and all(body[k] is False for k in ('default_promotion','sealed_target_bodies_read','energy_measured')),'Independent scope differs')
    collector=load('build/value-audit/tests.json');require(collector==body['tests'] and collector['passed'] is True,'Independent test collector differs')
    expected=methods('tests/value_audit_acceptance.py');require(len(expected)==10,'Independent source method count differs')
    covered=[];groups={};versions={}
    for row in collector['groups']:
        path=row['path'];report=load(path);require(pin(path)==row['sha256'],'Independent group changed');current(report['inputs'])
        require(report['passed'] is True and report['errors']==report['failures']==report['skips']==0 and report['covered_methods']==row['methods'] and report['tests_run']==len(report['covered_methods']),'Independent group incomplete')
        name=Path(path).stem.removesuffix('-tests');versions[name]=execution_source(name,report)
        covered+=report['covered_methods'];groups[Path(path).stem]=dict(path=path,tests_run=report['tests_run'],covered_methods=report['covered_methods'],executed_coordinator=versions[name])
    require(len(covered)==len(set(covered))==collector['tests_run']==10 and set(covered)==set(collector['covered_methods'])=={name for _,name in expected},'Independent unique method execution coverage differs')
    require(collector['coordinator_versions']==versions,'Independent collector mislabels executed coordinator versions')
    require({name for name,value in versions.items() if 'original_report_path' in value}=={'plan','targets','training'},'Preserved executed coordinator group coverage differs')
    source_review=load('build/value-provenance-review-final.json');current(source_review['inputs'])
    require(source_review['passed'] is True and source_review['methods']==4 and {r['group'] for r in source_review['groups']}=={'plan','targets','training'} and source_review['all_other_ast_unchanged'] is True and source_review['unchanged_test_source'] is True,'Final independent source-provenance review incomplete')
    training=body['training'];current(training['inputs'])
    require(training['passed'] is True and training['checkpoints']>=30 and training['all_optimizer_tensors_and_sampler_python_torch_cuda_rng_independently_verified'] is True and training['final_export_equals_PT'] is True and bool(training['changed_tensors']) and set(training['changed_tensors'])<={'layers.4.weight','layers.4.bias'},'Independent actual training state proof incomplete')
    require(training['all_frozen_tensors_bit_exact'] is True and training['frozen_parameters']==12807 and training['trainable_parameters']==65 and training['optimizer_parameter_ids']==[0,1],'Independent frozen-tensor or optimizer ownership proof incomplete')
    recovery=collector['live_training_recovery'];require(recovery['device']=='cuda' and recovery['actual_value_cuda_training_killed'] is True and recovery['raw_fixtures_retained'] is True,'Independent CUDA recovery absent')
    require({r['kill_event'] for r in recovery['tests']}=={'checkpoint_file_ready','checkpoint_committed'} and all(r['resumed_step']==6 and r['exact_model_optimizer_and_rng'] and r['completed_bytes_mtimes_unchanged'] for r in recovery['tests']),'Independent pre/postcommit training proof incomplete')
    recovery=collector['live_analysis_recovery'];require(recovery['profile']=='value_refit' and recovery['cache_capacity']==0 and recovery['raw_fixtures_retained'] is True,'Independent analyzer profile differs')
    flags=('actual_process_kill','previous64_export_bytes_mtime_retained','fresh128_search_and_physical_counters_equal','missing_export_repaired','completed_bytes_mtimes_unchanged','rehashed_corruption_refused')
    require({r['event'] for r in recovery['records']}=={'value_analysis_precommit','value_analysis_committed'} and all(all(r[k] is True for k in flags) for r in recovery['records']),'Independent analyzer recovery incomplete')
    case=collector['live_case_recovery'];arena=collector['live_arena_recovery']
    require(case['actual_process_kill'] and case['committed_prefix_retained']==1 and case['cases_after_resume']==2 and case['rehashed_reference_regret_refused'] and case['raw_journal_backup_marker_retained'],'Independent case recovery incomplete')
    require(arena['actual_process_kill'] and arena['committed_prefix_retained']==191 and arena['games_after_resume']==192 and arena['all_games_literal_replayed'] and arena['completed_bytes_mtimes_unchanged'] and arena['rehashed_final_margin_refused'] and arena['raw_journal_backup_marker_retained'],'Independent arena recovery incomplete')
    return body,['tests/value_audit_acceptance.py::'+cls+'.'+name for cls,name in expected],groups


def analysis_demo(plan,manifest,proof):
    from congklak import value_analysis as analysis,quality_search
    folder='data/value-analysis-demo';body=load(folder+'/analysis-128.json')
    with closing(database(folder+'/work.sqlite3')) as db:identity=meta(db,'identity');saved=meta(db,'analysis_result')
    p=dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0)
    require(body==saved and body['status']=='complete' and body['identity_sha256']==config.digest(identity),'Analysis durable publication differs')
    require(identity['schema']=='value-resumable-analysis-v1' and identity['profile']=='value_refit' and identity['settings']==plan['profiles']['value_refit'] and identity['position']==p and identity['max_nodes']==2048 and identity['seed']==20261015,'Production analyzer declared profile differs')
    require(identity['implementation_sha256']==manifest['sha256'] and identity['plan_sha256']==config.digest(plan) and identity['trained_witness']==proof and identity['rules_digest']==config.RULES_DIGEST,'Production analyzer lineage differs')
    evaluator=identity['model'];require(evaluator['model_file_sha256']==pin(plan['models']['refit']) and evaluator['model_weights_sha256']==proof['model_weights_sha256'] and evaluator['cache']['capacity']==0 and evaluator['max_batch']==1 and evaluator['source_sha256']==pin('congklak/puct_cpu_inference.py') and evaluator['dll_sha256']==pin('build/congklak_puct_cpu.dll'),'Production analyzer model/backend differs')
    current(evaluator['build_inputs']);current(evaluator['dependencies'])
    report=body['search'];native=Native();quality_search.validate_report(report,p,128,c_puct=.25,native=native,allow_interrupted=False);analysis.validate_counters(body['evaluator'],report,0)
    ranking,explanation=analysis.describe(report,p,native)
    require(body['rankings']==ranking and body['explanation']==explanation and body['proof_status']==report['proof_status'] and body['uncertainty']==analysis.UNCERTAINTY and body['clock_scope']==analysis.CLOCK,'Production analyzer explanation/uncertainty differs')
    require(type(body['attempt_ms']) in (int,float) and math.isfinite(body['attempt_ms']) and body['attempt_ms']>=report['timing']['elapsed_ms'],'Production analyzer clock differs')
    rerun=load('build/value-analysis-rerun.json');require(rerun['schema']=='value-analysis-rerun-v1' and rerun['passed'] is True and rerun['before']==rerun['after'],'Production completed analyzer stability proof differs')
    require(len(rerun['after'])==2 and {r['path'] for r in rerun['after']}=={folder+'/analysis-128.json',folder+'/work.sqlite3'},'Production analyzer stability coverage differs')
    for row in rerun['after']:
        require(pin(row['path'])==row['sha256'],'Production analyzer file changed after stable rerun')
        stamp=(ROOT/row['path']).stat().st_mtime_ns
        require((row.get('mtime_ns')==stamp) if 'mtime_ns' in row else row.get('mtime_utc_ticks')==stamp//100+621355968000000000,'Production analyzer mtime changed after stable rerun')
    return dict(path=folder+'/analysis-128.json',profile='value_refit',simulations=128,chosen_action=report['chosen'],completed_bytes_mtimes_unchanged=True,independent_actual_pre_postcommit_kills=True)


def retained():
    result=[]
    for folder in ('build/value-audit','build/value-learning-acceptance','build/value-benchmark-fixtures'):
        base=ROOT/folder
        if base.exists():
            for path in sorted(base.rglob('*')):
                if path.is_file() and not path.name.endswith(('-wal','-shm')):pin(path);result.append(path.relative_to(ROOT).as_posix())
    # Superseded diagnostics are bound as raw evidence. Their historic input
    # maps describe their publication instant and need not be current again.
    patterns=('value-implementation-prefreeze-*.json','value-data-prefreeze-*.json','value-*-failed-*.json','value-*-failure*.json','value-benchmark-pretraining-probe*','value-prefreeze-*','value-provenance-review*.json','value-prefinal-*')
    for pattern in patterns:
        for path in sorted((ROOT/'build').glob(pattern)):
            if path.is_file():pin(path);result.append(path.relative_to(ROOT).as_posix())
    return sorted(set(result))


def build_report():
    INPUTS.clear();pin('tools/record-value.py');plan,manifest=common.admit();pin('build/value-plan.json');pin('build/value-implementation.json');current(manifest['sources']);current(manifest['science_inputs'])
    prior=load('build/quality-acceptance.json');require(prior['passed'] and len(prior['inputs'])==1647,'Prior accepted evidence differs');current(prior['inputs'])
    training_proof=training(plan)
    learning,learning_ids=group('build/value-learning-acceptance.json','tests/value_learning_acceptance.py',4)
    bench_checks,bench_ids=group('build/value-benchmark-acceptance.json','tests/value_benchmark_acceptance.py',13)
    audit,audit_ids,audit_groups=independent();strength,counts,primary,resources=benchmark(plan,manifest)
    ids=learning_ids+bench_ids+audit_ids;require(len(ids)==len(set(ids))==27,'Aggregate unique executed checks differ')
    audited=audit['benchmark'];require(audited['cases']==80 and audited['scalar_measurements']==640 and audited['games']==192 and audited['literal_choices']==counts['literal_choices'],'Independent exhaustive benchmark coverage differs')
    require(audited['primary']==primary and audited['secondary_factor_contrasts']==strength['secondary_factor_contrasts'] and audited['both_baseline_gates']==strength['both_baseline_primary_gates'],'Independent arena statistics/gates differ')
    require(audited['original_resources']==resources and audited['all_terminal_games']==counts['terminal_games'] and audited['all_fallbacks']==counts['fallbacks'],'Independent resource/completion counts differ')
    models=audit['models'];require(models['all_refit_outputs_literal_verified'] is True and len(models['models'])==2 and {r['model'] for r in models['models']}=={'control','refit'} and all(r['roots']==1000 and r['literal_forward'] and r['max_error']<=1e-10 for r in models['models']),'Independent literal neural outputs incomplete')
    require(models['all_frozen_tensors_bit_exact'] is True and models['global_policy_unchanged_by_identical_subgraph'] is True and models['policy_logits_bit_exact'] is True,'Independent fixed-policy structure/native proof incomplete')
    demo=analysis_demo(plan,manifest,training_proof)
    for path in ('docs/VALUE_SCOPE.md','docs/VALUE_RESULTS.md','docs/VALUE_INDEPENDENT_AUDIT.md','docs/VALUE_RUN_GUIDE.md','build/training-runtime.json'):pin(path)
    raw=retained();require('torch' not in sys.modules,'CPU-only acceptance recorder imported Torch')
    return dict(schema='value-acceptance-v1',passed=True,engineering_passed=True,tests_run=27,errors=0,failures=0,skips=0,covered_checks=ids,
        groups=dict(author_learning=dict(tests_run=4,covered_methods=learning_ids),author_benchmark=dict(tests_run=13,covered_methods=bench_ids),independent=audit_groups),executed_audit_coordinators=audit['tests']['coordinator_versions'],
        prior_inputs_preserved=dict(quality=1647,all_current_hashes_match=True),independent_audit_complete=True,implementation_sha256=manifest['sha256'],
        training=dict(optimizer_steps=1500,checkpoint_files_verified=training_proof['checkpoints_verified'],model_file_sha256=training_proof['model_file_sha256'],model_weights_sha256=training_proof['model_weights_sha256'],trainable_parameters=65,frozen_parameters=12807,all_frozen_tensors_bit_exact=True,policy_logits_bit_exact=True,selection='Fixed final optimizer step',all_optimizer_and_rng_states_verified=True),
        analysis_demo=demo,position_cases=80,scalar_choices=640,arena_games=192,terminal_games=counts['terminal_games'],unfinished_games=counts['unfinished_games'],fallbacks=counts['fallbacks'],literal_choices=counts['literal_choices'],
        both_baseline_primary_gates=strength['both_baseline_primary_gates'],arenas=strength['arenas'],secondary_factor_contrasts=strength['secondary_factor_contrasts'],resources=resources,
        training_performed=True,cuda_training=True,cpu_inference=True,torch_imported=False,energy_measured=False,full_game_solved=False,sealed_final_test_consumed=False,default_promotion=False,development_reuse=True,
        scoped_failed_probes_and_raw_recovery_preserved=raw,inputs=dict(sorted(INPUTS.items())))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args();body=build_report()
    if not args.check_only:common.publish(ROOT/'build/value-acceptance.json',body)
    print(config.canonical(dict(event='value_acceptance_checked' if args.check_only else 'value_accepted',engineering_passed=True,tests_run=27,cases=80,scalar_choices=640,games=192,inputs=len(INPUTS),both_baseline_primary_gates=body['both_baseline_primary_gates'],acceptance_written=not args.check_only)),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:print(config.canonical(dict(event='value_record_refused',error=str(exc))),flush=True);raise SystemExit(2)
