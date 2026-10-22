"""Publish CPU PUCT engineering acceptance only from complete audited evidence."""
from __future__ import annotations
import argparse
import ast
from contextlib import closing
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,phase6_common as prior,puct_cpu_common as common
from congklak.native import Native
require=prior.require
INPUTS={}

def pin(path):
    path=(ROOT/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(),'Acceptance input absent or outside workspace: '+str(path))
    name=str(path.relative_to(ROOT)).replace('\\','/');checksum=prior.sha(path);INPUTS[name]=checksum;return checksum

def load(path):
    pin(path);return json.loads((ROOT/path).read_text(encoding='utf-8'))

def current(mapping):
    require(isinstance(mapping,dict) and bool(mapping),'Evidence has no source/input pins')
    for name,digest in mapping.items():require(pin(name)==digest,'Evidence source/input differs: '+name)

def module(name,path):
    pin(path);spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def methods(path):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'));found=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            for child in node.body:
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and child.name.startswith('test_'):
                    found.append((node.name,child.name))
    return found

def group(path,test_source,expected_count):
    body=load(path);current(body['inputs']);count=body.get('tests_run',body.get('tests'))
    require(body.get('passed') is True and count==expected_count and all(body.get(k)==0 for k in ('errors','failures')) and body.get('skips',body.get('skipped',0))==0,'Incomplete acceptance group: '+path)
    covered=body.get('covered_methods');require(isinstance(covered,list) and len(covered)==len(set(covered))==count,'Duplicated/missing executed methods: '+path)
    expected=methods(test_source);pin(test_source)
    require(len(expected)==count and len({name for _,name in expected})==count,'Source test methods are duplicated or missing: '+test_source)
    require({name.rsplit('.',1)[-1] for name in covered}=={name for _,name in expected},'Source/test method coverage differs: '+path)
    prefix=Path(test_source).stem
    return body,[prefix+'.'+name if '.' not in name else name for name in covered]

def meta(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone();require(row is not None,'Missing durable completion metadata: '+key)
    body=json.loads(row[0]);require(config.digest(body)==row[1],'Corrupt completion metadata: '+key);return body

def database(path):
    path=(ROOT/path).resolve();pin(path)
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('BEGIN')
    require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Damaged production SQLite: '+str(path));return db

def evaluator_identities(values,plan,manifest):
    require(set(values)==set(common.PROFILES),'Missing registered evaluator identities')
    for name,value in values.items():
        model=plan['profiles'][name]['model'];envelope=load(plan['models'][model])
        require(value['model_file_sha256']==pin(plan['models'][model])==plan['model_files_sha256'][model] and value['model_weights_sha256']==envelope['model']['weights_sha256'] and value['rules_digest']==config.RULES_DIGEST,'Wrong evaluator model/rules identity')
        if name.endswith('_legacy'):
            require(value['backend']=='native' and value['source_sha256']==pin('congklak/phase6_inference.py') and value['cache_capacity']==0 and value['feature_passes_per_native_eval']==2,'Legacy adapter identity differs')
        else:
            require(value['source_sha256']==manifest['sources'][manifest['roles']['inference']] and value['dll_sha256']==pin('build/congklak_puct_cpu.dll') and value['cache']['capacity']==plan['profiles'][name]['cache_capacity'],'Structured adapter/cache identity differs')

def speed_keys(plan):
    profiles=list(common.PROFILES);settings=plan['speed'];keys=[('startup',name,0,0,-1) for name in profiles]
    keys += [('parity',name,size,0,-1) for name in profiles for size in settings['batch_sizes']]
    for block in range(settings['blocks']):
        order=profiles[block%6:]+profiles[:block%6];sizes=settings['batch_sizes'];sizes=sizes[block%6:]+sizes[:block%6]
        for kind in ('cold','hot'):
            for size in sizes:
                for name in order:keys.append((kind,name,size,block,-1))
    for block in range(settings['tree_blocks']):
        order=profiles[block%6:]+profiles[:block%6]
        for index in range(48):
            for name in order:keys.append(('tree',name,1,block,index))
    return keys

def verify_speed(plan,manifest,benchmark):
    speed=module('record_cpu_speed','tools/puct-cpu-speed.py');body=load(manifest['artifacts']['speed_output'])
    require(body.get('schema')=='puct-cpu-speed-results-v1' and body.get('complete') is True and body.get('units')==1410 and body.get('fixed_tree_semantics_equal') is True,'CPU speed campaign incomplete')
    require(body['identity']['implementation_sha256']==manifest['sha256'] and body['identity']['plan_sha256']==config.digest(plan),'Speed lineage differs')
    evaluator_identities(body['identity']['evaluators'],plan,manifest)
    path=manifest['artifacts']['speed_run'];require(body['journal']==path,'Unexpected speed journal path');saved={}
    with closing(database(path+'/work.sqlite3')) as db:
        require(meta(db,'identity')==body['identity'],'Speed SQLite/report identity differs')
        for row in db.execute('SELECT kind,profile,batch,block,root,body,hash FROM units'):
            key=tuple(row[name] for name in ('kind','profile','batch','block','root'));value=json.loads(row['body'])
            require(config.digest(value)==row['hash'] and key not in saved,'Damaged/duplicate speed unit');speed.check(key,value,plan,benchmark);saved[key]=value
        require(len(saved)==1410 and set(saved)==set(speed_keys(plan)),'Speed journal registered unit coverage differs')
        require(meta(db,'puct_cpu_speed_complete')==dict(units=1410,report_sha256=config.digest(body)),'Speed completion witness differs')
    require(body==speed.summarize(saved,body['identity'],plan,ROOT/path),'Speed report does not reproduce from all1410 units')
    details={}
    for name in common.PROFILES:
        p=body['profiles'][name];counts=p['trees']['counts'];details[name]=dict(cold_batch1_us=p['micro']['cold']['1']['whole_call_ns']['median']/1000,cold_batch8_us=p['micro']['cold']['8']['whole_call_ns']['median']/1000,
            median_fixed64_tree_ms=p['trees']['wall_ns']['median']/1e6,tree_requests=counts['evaluations_requested'],tree_physical_forwards=counts['native_actual_evals'],tree_cache_hits=counts['cache_hits'],tree_hit_fraction=counts['cache_hits']/counts['evaluations_requested'] if counts['evaluations_requested'] else None)
    cost_declaration=load('build/puct-cpu-cost-gate.json');require(manifest['science_inputs'].get('build/puct-cpu-cost-gate.json')==pin('build/puct-cpu-cost-gate.json'),'Cost gate was not implementation-pinned')
    require(cost_declaration['schema']=='puct-cpu-cost-gate-v1' and cost_declaration['predeclared_before_production'] is True and cost_declaration['models']==['warm','updated'] and cost_declaration['energy_measured'] is False,'Invalid predeclared cost gate')
    gate_models={}
    for model in cost_declaration['models']:
        root_ratios=[];cache_ratios=[];half_work=True
        for index in range(48):
            ratios=[];cached=[]
            for block in range(3):
                legacy=saved[('tree',model+'_legacy',1,block,index)];single=saved[('tree',model+'_single',1,block,index)];cache=saved[('tree',model+'_cache',1,block,index)]
                ratios.append(single['wall_ns']/legacy['wall_ns']);cached.append(cache['wall_ns']/single['wall_ns'])
                half_work=half_work and all(single['counts'][key]*2==legacy['counts'][key] for key in ('feature_simulations','feature_deposits'))
            root_ratios.append(statistics.median(ratios));cache_ratios.append(statistics.median(cached))
        ratio=statistics.median(root_ratios)
        gate_models[model]=dict(matched_roots=48,matched_blocks_per_root=3,per_root_median_single_legacy_ratios=root_ratios,median_matched_single_legacy_ratio=ratio,
            exact_half_features_at_every_match=half_work,passed=bool(ratio<1 and half_work and body['fixed_tree_semantics_equal']),median_matched_cache_single_ratio_descriptive=statistics.median(cache_ratios))
    gate=dict(schema=cost_declaration['schema'],predeclared_file_sha256=pin('build/puct-cpu-cost-gate.json'),passed=all(v['passed'] for v in gate_models.values()),models=gate_models,
              confidence_interval_claim=False,cache_gain_required=False,scope='Predeclared matched fixed64 whole-tree point ratios; cache comparisons descriptive; no electricity claim')
    return body,details,gate

def verify_benchmark(plan,manifest,benchmark):
    body=load(manifest['artifacts']['benchmark_output']);require(body.get('schema')=='puct-cpu-benchmark-results-v1' and body.get('cases')==80 and body.get('games')==576,'CPU strength campaign incomplete')
    require(body['identity']['implementation']==manifest and body['identity']['plan_sha256']==config.digest(plan),'Benchmark lineage differs')
    evaluator_identities(body['identity']['evaluators'],plan,manifest)
    require(body['identity']['baseline_dll_sha256']==pin('build/congklak_native.dll'),'Baseline DLL identity differs')
    require(body.get('default_promotion') is False and body.get('sealed_final_test_consumed') is False and body.get('energy_measured') is False,'Unexpected promotion/sealed/energy claim')
    resource_path=manifest['artifacts'].get('benchmark_producer_resources','build/puct-cpu-benchmark-resources.json');resource=load(resource_path)
    with closing(database(manifest['artifacts']['benchmark_run']+'/work.sqlite3')) as db:
        require(meta(db,'identity')==body['identity'],'Benchmark SQLite/report identity differs')
        cases,games=benchmark.validate_saved_units(db,plan,Native());require(cases==80 and games==576,'Missing registered benchmark units')
        setup=db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone();require(setup is not None and config.digest(json.loads(setup[0]))==setup[1],'Missing/damaged startup warmup observation')
        require(meta(db,'puct_cpu_benchmark_complete')==dict(cases=80,games=576,report_sha256=config.digest(body),resource_file_sha256=pin(resource_path)),'Benchmark completion witness differs')
        scalar=sum(len(engines) for text, in db.execute('SELECT body FROM cases') for engines in json.loads(text)['measurements'].values());require(scalar==1280,'Missing registered scalar choices')
        literal_choices=sum(len(json.loads(text)['moves']) for text, in db.execute('SELECT body FROM games'))
        require(body==benchmark.summarize(db,body['identity'],plan),'Benchmark report does not reproduce from cases and games')
    require(resource['schema']=='puct-cpu-benchmark-resources-v1' and resource['implementation_sha256']==manifest['sha256'] and resource['report_file_sha256']==pin(manifest['artifacts']['benchmark_output']) and resource['torch_imported'] is False and resource['energy_measured'] is False,'Inner benchmark resources differ')
    return body,dict(position_cases=80,scalar_choices=1280,games=576,literal_choices=literal_choices,terminal_games=sum(row['terminal_games'] for row in body['arenas'].values()),unfinished_games=sum(row['unfinished_games'] for row in body['arenas'].values()),fallbacks=sum(sum(row['fallback_counts'].values()) for row in body['arenas'].values()))

def resources(manifest):
    result={}
    for producer in ('speed','benchmark'):
        path='build/puct-cpu-'+producer+'-process-resources.json';body=load(path);source=manifest['roles'][producer]
        expected=dict(producer=producer,producer_sha256=pin(source),wrapper_sha256=pin(manifest['roles']['resource_wrapper']),implementation_sha256=manifest['sha256'])
        require(body['schema']=='puct-cpu-process-resources-v1' and body['identity']==expected and body['report_file_sha256']==pin(manifest['artifacts'][producer+'_output']),'Outer producer resource lineage differs')
        require(body['torch_imported'] is False and body['cuda_used'] is False and body['energy_measured'] is False,'CPU resource scope misreported')
        elapsed=body['elapsed_seconds_including_module_load'];require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>0,'Invalid scoped resource clock')
        result[producer]=body
    return result

def independent_audit():
    body=load('build/puct-cpu-audit/audit.json');require(body.get('passed') is True,'Independent full CPU audit incomplete');current(body['inputs'])
    require(all(body.get(key) is False for key in ('default_promotion','sealed_target_bodies_read','energy_measured','training_performed','independent_live_warm_kill_claimed')),'Independent evidence scope differs')
    test_source='tests/puct_cpu_audit_acceptance.py';source_methods=methods(test_source);expected={name for _,name in source_methods};pin(test_source);covered=[];groups={}
    require(len(source_methods)==len(expected)==13,'Independent source method identities differ')
    collector=load('build/puct-cpu-audit/tests.json');require(collector==body.get('tests') and collector.get('passed') is True,'Independent full/test collector differs')
    records=collector.get('groups');require(isinstance(records,list) and bool(records),'Independent group catalog missing')
    for row in records:
        path=row['path'];report=load(path);require(pin(path)==row['sha256'],'Independent group file changed');current(report['inputs'])
        count=report['tests_run'];require(report['passed'] is True and report['errors']==report['failures']==report['skips']==0 and len(report['covered_methods'])==count,'Independent group incomplete')
        require(row['methods']==report['covered_methods'],'Independent group/catalog coverage differs')
        covered.extend(report['covered_methods']);name=Path(path).stem;groups[name]=dict(path=path,tests_run=count,covered_methods=report['covered_methods'])
    require(len(covered)==len(set(covered)) and set(covered)==expected,'Independent unique executed methods differ from full source')
    require(len(covered)==13,'Independent unique test count differs')
    require(set(collector['covered_methods'])==set(covered) and collector['tests_run']==len(covered),'Independent summary coverage differs')
    recovery=collector['analysis_actual_recovery'];require(recovery['model']=='updated' and recovery['cache_capacity']==4096 and recovery['raw_states_and_logs_retained'] is True,'Independent analyzer recovery scope differs')
    checks=recovery['checks'];require({row['event'] for row in checks}=={'puct_cpu_analysis_precommit','puct_cpu_analysis_committed'} and len(checks)==2,'Independent analyzer kill barriers differ')
    require(all(all(row[key] is True for key in ('actual_process_kill','prior_completed_bytes_retained','fresh_semantic_and_counters_equal','missing_export_repaired','finished_bytes_mtimes_stable','rehashed_counter_corruption_refused')) for row in checks),'Independent analyzer recovery proof incomplete')
    require(body['prior_artifacts']==dict(phase6_inputs=1192,stage5_inputs=878,all_current_hashes_match=True),'Independent prior preservation coverage differs')
    return body,['puct_cpu_audit_acceptance.'+name for name in covered],groups


def analysis_demo(plan,manifest):
    """Validate the user-visible completed recommendation without running a new tree."""
    from congklak.puct_cpu_analysis import validate_counters
    from congklak.phase6_analysis import describe
    from congklak.phase6_selfplay import validate_search
    folder=manifest['artifacts']['analysis_demo_run'];body=load(folder+'/analysis-128.json')
    with closing(database(folder+'/work.sqlite3')) as db:
        identity=meta(db,'identity');saved=meta(db,'analysis_result')
    require(body==saved and body['status']=='complete' and body['identity_sha256']==config.digest(identity),'Production analysis publication differs from durable recommendation')
    require(identity['schema']=='puct-cpu-resumable-analysis-v1' and identity['implementation_sha256']==manifest['sha256'] and identity['plan_sha256']==config.digest(plan) and identity['source_sha256']==pin(manifest['roles']['analysis']),'Production analysis source/plan differs')
    require(identity['rules_digest']==config.RULES_DIGEST and identity['position']==dict(pits=[[7]*7,[7]*7],stores=[0,0],controller=0) and identity['max_nodes']==2048 and identity['seed']==20261015,'Production analysis position/search identity differs')
    model=identity['model'];envelope=load(plan['models']['updated'])
    require(model['model_file_sha256']==pin(plan['models']['updated']) and model['model_weights_sha256']==envelope['model']['weights_sha256'] and model['cache']['capacity']==4096,'Production analysis model/cache differs')
    require(model['source_sha256']==manifest['sources'][manifest['roles']['inference']] and model['dll_sha256']==pin('build/congklak_puct_cpu.dll') and model['build_record_sha256']==pin('build/puct-cpu-build.json') and model['rules_digest']==config.RULES_DIGEST and model['max_batch']==1,'Production analysis adapter/build identity differs')
    current(model['build_inputs']);current(model['dependencies'])
    report=body['search'];native=Native();validate_search(report,identity['position'],128,native=native);validate_counters(body['evaluator'],report,4096)
    rankings,explanation=describe(report,identity['position'],native)
    require(report['completed_simulations']==128 and body['rankings']==rankings and body['explanation']==explanation and body['proof_status']==report['proof_status'],'Production analysis search/explanation differs')
    require(type(body['attempt_ms']) in (int,float) and math.isfinite(body['attempt_ms']) and body['attempt_ms']>=report['timing']['elapsed_ms'],'Production analysis whole-attempt clock differs')
    rerun_path='build/puct-cpu-analysis-rerun.json';pin(rerun_path);rerun=json.loads((ROOT/rerun_path).read_text(encoding='utf-8-sig'))
    require(rerun['schema']=='puct-cpu-analysis-rerun-v1' and rerun['passed'] is True and rerun['before']==rerun['after'],'Production analyzer finished-rerun stability witness differs')
    require({row['path'] for row in rerun['after']}=={folder+'/analysis-128.json',folder+'/work.sqlite3'} and len(rerun['after'])==2,'Production analyzer rerun file coverage differs')
    for row in rerun['after']:
        require(pin(row['path'])==row['sha256'] and (ROOT/row['path']).stat().st_mtime_ns//100+621355968000000000==row['mtime_utc_ticks'],'Production analyzer rerun file changed')
    pin('build/puct-cpu-progress/analysis-demo-rerun.log')
    return dict(path=folder+'/analysis-128.json',model='updated',simulations=128,chosen_action=report['chosen'],durable_publication_matches=True,
                completed_bytes_mtimes_unchanged=True,independent_live_pre_postcommit_kills=True,recovery_scope='Independent retained fixtures verify interruption recovery; this completed production demo supplies the user-facing recommendation')

def retained():
    paths=[]
    for folder in ('build/puct-cpu-audit','build/puct-cpu-benchmark-fixtures'):
        base=ROOT/folder
        if base.exists():
            for path in sorted(base.rglob('*')):
                if path.is_file() and not path.name.endswith(('-wal','-shm')):paths.append(str(path.relative_to(ROOT)).replace('\\','/'));pin(path)
    # Superseded prefinal documents/acceptance are retained as raw diagnostics.
    # Their old input mappings describe their publication instant and must not
    # recursively become current acceptance requirements after correction.
    for pattern in ('puct-cpu-implementation-prefreeze-*.json','puct-cpu-*-failure*.json','puct-cpu-benchmark-acceptance-missing-*.json','puct-cpu-prefinal-doc-correction-*'):
        for path in sorted((ROOT/'build').glob(pattern)):
            name=str(path.relative_to(ROOT)).replace('\\','/');paths.append(name);pin(path)
    return sorted(set(paths))

def build_report():
    INPUTS.clear();pin('tools/record-puct-cpu.py');plan,manifest=common.admit();pin('build/puct-cpu-plan.json');pin('build/puct-cpu-implementation.json');current(manifest['sources']);current(manifest['science_inputs'])
    benchmark=module('record_cpu_benchmark','tools/benchmark-puct-cpu.py');_,witness=benchmark.admit(plan)
    older=load('build/phase6-acceptance.json');require(len(older['inputs'])==1192,'Prior exhaustive input count changed');current(older['inputs'])
    author,author_methods=group('build/puct-cpu-inference-acceptance.json','tests/puct_cpu_inference_acceptance.py',9)
    bench,bench_methods=group('build/puct-cpu-benchmark-acceptance.json','tests/puct_cpu_benchmark_acceptance.py',12)
    audit,audit_methods,groups=independent_audit();speed,costs,cost_gate=verify_speed(plan,manifest,benchmark);strength,units=verify_benchmark(plan,manifest,benchmark);cost_resources=resources(manifest)
    covered=author_methods+bench_methods+audit_methods;require(len(covered)==len(set(covered))==34,'Aggregate unique executed check coverage differs')
    audited_strength=audit['benchmark'];require(audited_strength['cases']==80 and audited_strength['scalar_measurements']==1280 and audited_strength['games']==576 and audited_strength['literal_choices']==units['literal_choices'],'Independent benchmark coverage differs')
    require(set(audited_strength['primary'])==set(strength['arenas']),'Independent arena comparison catalog differs')
    for name,row in strength['arenas'].items():
        checked=audited_strength['primary'][name]
        require(checked['mean_margin_seeds']==row['primary_paired_mean_final_margin_seeds'] and checked['interval']==row['primary_margin_bootstrap_95_interval'] and checked['gate_passed']==row['primary_strength_gate_passed'],'Independent arena outcome/gate differs: '+name)
    audited_speed=audit['speed'];require(audited_speed['units']==1410 and audited_speed['trees']==864 and audited_speed['exact_fixed_tree_semantics'] is True and audited_speed['cache_leaf_trace_counts_rederived'] is True and audited_speed['point_cost_gate_passed']==cost_gate['passed'],'Independent speed coverage/gate differs')
    for model in ('warm','updated'):require(audited_speed['single_vs_legacy_tree_cost'][model]['median_root_ratio']==cost_gate['models'][model]['median_matched_single_legacy_ratio'],'Independent matched cost ratio differs')
    demo=analysis_demo(plan,manifest)
    for name in ('docs/PUCT_CPU_SCOPE.md','docs/PUCT_CPU_RESULTS.md','docs/PUCT_CPU_INDEPENDENT_AUDIT.md','docs/PUCT_CPU_RUN_GUIDE.md'):pin(name)
    require('torch' not in sys.modules,'Stdlib recorder unexpectedly imported Torch')
    preserved=retained()
    return dict(schema='puct-cpu-acceptance-v1',passed=True,engineering_passed=True,tests_run=len(covered),errors=0,failures=0,skips=0,covered_checks=covered,
        groups=dict(author_inference=dict(tests_run=9,covered_methods=author_methods),author_benchmark=dict(tests_run=12,covered_methods=bench_methods),independent=groups),
        prior_inputs_preserved=dict(phase6=1192,stage5=878,all_current_hashes_match=True),independent_audit_complete=True,implementation_sha256=manifest['sha256'],model_witnesses=witness['model_witnesses'],analysis_demo=demo,
        inference_speed_units=1410,fixed_tree_semantics_equal=True,position_cases=units['position_cases'],scalar_choices=units['scalar_choices'],arena_games=units['games'],terminal_games=units['terminal_games'],unfinished_games=units['unfinished_games'],fallbacks=units['fallbacks'],
        both_baseline_primary_gates=strength['both_baseline_primary_gates'],arenas=strength['arenas'],costs=costs,cost_gate=cost_gate,resources=cost_resources,
        source_equivalence='Fixed64 trees exactly match each model legacy control; source/math unchanged',development_reuse=True,sealed_final_test_consumed=False,training_performed=False,torch_imported=False,cuda_used=False,energy_measured=False,full_game_solved=False,default_promotion=False,
        scoped_failed_probes_and_raw_recovery_preserved=preserved,inputs=dict(sorted(INPUTS.items())))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    output=build_report()
    if not args.check_only:prior.publish(ROOT/'build/puct-cpu-acceptance.json',output)
    print(config.canonical(dict(event='puct_cpu_acceptance_checked' if args.check_only else 'puct_cpu_accepted',engineering_passed=True,tests_run=output['tests_run'],cases=80,scalar_choices=1280,games=576,both_baseline_primary_gates=output['both_baseline_primary_gates'],cost_gate_passed=output['cost_gate']['passed'],inputs=len(INPUTS),acceptance_written=not args.check_only)),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:print(config.canonical(dict(event='puct_cpu_record_refused',error=str(exc))),flush=True);raise SystemExit(2)
