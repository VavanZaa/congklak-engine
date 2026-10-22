"""Seal executed neural diagnostics, independent certification and durable recovery."""
import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag as p,neurodiag_common as common
require=common.require
OUTPUT=ROOT/'build/neurodiag-acceptance.json'
EXCLUDED={'neurodiag-acceptance.json','neurodiag-acceptance-rerun.json','neurodiag-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def physical(path,table,key):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute('SELECT '+key+',body,hash FROM '+table+' ORDER BY '+key).fetchall()


def raw_recovery(record):
    probe=(ROOT/record['probe']).resolve();run=(ROOT/record['run']).resolve()
    require(probe.is_relative_to(ROOT/'build') and run.is_relative_to(ROOT/'build'),'Recovery escaped build')
    marker=common.read(ROOT/record['marker']);prefix=record['committed_prefix']
    require(marker['event']==record['event'] and marker['completed']==prefix,'Physical kill marker differs')
    events=[]
    for line in (probe/'worker.log').read_text(encoding='utf-8').splitlines():
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict):events.append(value)
    require(any(common.same(value,marker) for value in events),'Kill marker lacks flushed event')
    before=probe/'validated-prefix.sqlite3';after=run/'work.sqlite3'
    oldmeta,newmeta=physical(before,'meta','key'),physical(after,'meta','key')
    require([r[0] for r in oldmeta]==['identity'] and {r[0] for r in newmeta}=={'identity','summary','complete'},'Recovery summary/complete boundary differs')
    require(oldmeta[0]==next(r for r in newmeta if r[0]=='identity'),'Recovery identity changed')
    for table,key,count in (('batches','idx',prefix),('observations','unit',16*prefix)):
        prior,current=physical(before,table,key),physical(after,table,key)
        require(len(prior)==count and all(row in current for row in prior),'Committed raw '+table+' prefix changed')
    require(len(physical(after,'batches','idx'))==3 and len(physical(after,'observations','unit'))==48,'Recovered fixture incomplete')


def executed(report_name,source_name):
    body=common.read(ROOT/report_name)
    require(body['passed'] is True and all(type(body[k]) is int and body[k]==0 for k in ('errors','failures','skips')),'Owner checks failed')
    ids=body.get('covered_checks',body.get('covered_methods'))
    require(type(body['tests_run']) is int and len(ids)==len(set(ids))==body['tests_run']==4,'Owner executed count differs')
    tree=ast.parse((ROOT/source_name).read_text(encoding='utf-8-sig'))
    available={node.name+'.'+method.name for node in tree.body if isinstance(node,ast.ClassDef)
        for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
    require({'.'.join(name.split('.')[-2:]) for name in ids}==available,'Owner source method identities differ')
    require(body['inputs'].get(source_name)==common.sha(ROOT/source_name),'Executed owner source unpinned')
    return body,[source_name+'::'+'.'.join(name.split('.')[-2:]) for name in ids]


def audit_repair():
    body=common.read(ROOT/'build/neurodiag-audit-harness-repair.json')
    require(body['schema']=='neurodiag-audit-harness-repair-v1' and body['passed'] is True
        and body['originals_unchanged'] is True and body['originals_before']==body['originals_after']
        and body['no_repeated_producer_work'] is body['no_repeated_preflight_or_kills'] is body['no_science_model_policy_or_source_changes'] is True,
        'Audit repair changed protected producer/recovery evidence')
    common.current(body['originals_after']);common.current(body['retained_failure_inputs'])
    failed=body['failed_attempt'];report=common.read(ROOT/failed['path'])
    require(common.sha(ROOT/failed['path'])==failed['sha256'] and report['passed'] is False
        and report['tests_run']==failed['tests_run']==2 and report['errors']==failed['errors']==1
        and report['failures']==failed['failures']==1 and report['skips']==failed['skips']==0,'Original failed audit not retained')
    require(failed['failed_methods']==['__main__.NeurodiagIndependentAudit.test_04_actual_panel_cache_scalar_forward_and_targets']
        and failed['errored_methods']==['__main__.NeurodiagIndependentAudit.test_05_analytic_gradients_svd_summaries_and_observation_witness'],
        'Original failed method identities differ')
    for item in (body['executed_source_copy'],body['versioned_source'],body['versioned_freeze']):
        require(common.sha(ROOT/item['path'])==item['sha256'],'Audit repair source/freeze bytes differ')
    def methods(path):
        tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
        return {m.name:ast.dump(m,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='NeurodiagIndependentAudit'
            for m in c.body if isinstance(m,ast.FunctionDef)}
    old,new=methods('tests/neurodiag_independent_audit.py'),methods(body['versioned_source']['path'])
    require(set(old)==set(new) and sorted(k for k in old if old[k]!=new[k])==body['changed_class_methods']==[
        'test_04_actual_panel_cache_scalar_forward_and_targets'],'Unexpected independent audit class changes')
    return body


def lock_failure():
    directory=ROOT/'build/neurodiag-audit-final-lock-failure-v1';body=common.read(directory/'diagnostic.json')
    require(body['schema']=='neurodiag-completed-writer-lock-failure-v1' and body['passed'] is False and body['producer_issue'] is True
        and body['numeric_cache_issue'] is False and body['cache_bytes_equal_certificate'] is True
        and body['tests_run']==1 and body['failures']==1 and body['errors']==body['skips']==0,'Completed Windows lock failure not retained')
    for values in (body['cache_sha256'],body['originals_unchanged'],body['retained_inputs'],body['original_retained_fixture_inputs']):common.current(values)
    path=directory/'build/neurodiag-audit/final.json';report=common.read(path)
    require(common.sha(path)==body['canonical_failed_report_sha256'] and report['passed'] is False
        and report['tests_run']==1 and report['failures']==1 and report['errors']==report['skips']==0
        and report['covered_methods']==['test_06_corruption_repair_and_no_forward_completed_idempotence'],'Original lock-failure execution differs')
    return body


def independent_groups(audit):
    expected={'preflight':3,'production':2,'final':1};original='tests/neurodiag_independent_audit.py'
    sources={'preflight':original,'production':'tests/neurodiag_independent_audit_v2.py','final':'tests/neurodiag_independent_audit_v3.py'}
    freezes={'preflight':'build/neurodiag-audit-source-freeze.json',
        'production':'build/neurodiag-audit-source-freeze-v2.json','final':'build/neurodiag-audit-source-freeze-v3.json'}
    tree=ast.parse((ROOT/original).read_text(encoding='utf-8-sig'))
    available={m.name for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='NeurodiagIndependentAudit'
        for m in c.body if isinstance(m,ast.FunctionDef) and m.name.startswith('test_')}
    require(len(audit['groups'])==3,'Three independent execution groups required')
    seen=set();methods=[];evidence={};bindings=[]
    for entry in audit['groups']:
        name=Path(entry['path']).stem
        require(name in expected and name not in seen and entry['path']=='build/neurodiag-audit/'+name+'.json','Independent group identity differs');seen.add(name)
        source=sources[name];freeze=common.read(ROOT/freezes[name])
        require(freeze['schema']=='neurodiag-independent-executed-source-v1' and freeze['inputs'].get(source)==common.sha(ROOT/source),
            'Executed audit source freeze differs')
        executed_tree=ast.parse((ROOT/source).read_text(encoding='utf-8-sig'))
        executed_methods={m.name for c in executed_tree.body if isinstance(c,ast.ClassDef) and c.name=='NeurodiagIndependentAudit'
            for m in c.body if isinstance(m,ast.FunctionDef) and m.name.startswith('test_')}
        part=common.read(ROOT/entry['path']);checksum=common.sha(ROOT/entry['path'])
        require(entry['sha256']==checksum==audit['inputs'].get(entry['path']),'Group file binding differs')
        require(part['schema']=='neurodiag-independent-group-v1' and part['group']==name and part['passed'] is True,'Group did not pass')
        require(all(type(part[k]) is int and part[k]==0 for k in ('errors','failures','skips')),'Independent failures or skips')
        ids=part['covered_methods']
        require(type(part['tests_run']) is int and part['tests_run']==entry['tests_run']==expected[name] and len(ids)==len(set(ids))==expected[name]
            and ids==entry['covered_methods']==freeze['groups'][name],'Group executed coverage differs')
        require(entry['executed_source']['path']==source and entry['executed_source']['sha256']==common.sha(ROOT/source)==part['inputs'].get(source),'Independent executed source differs')
        require(set(ids)<=available and set(ids)<=executed_methods and part['covered_checks']==['__main__.NeurodiagIndependentAudit.'+m for m in ids],'Independent class IDs differ')
        require(all(audit['inputs'].get(k)==v for k,v in part['inputs'].items()),'Aggregate omits group source/cache inputs')
        require(not set(evidence)&set(part['evidence']),'Duplicated independent evidence')
        evidence.update(part['evidence']);methods.extend(ids);bindings.extend(source+'::NeurodiagIndependentAudit.'+m for m in ids)
    require(seen==set(expected) and len(methods)==len(set(methods))==6 and set(methods)==available and methods==audit['covered_methods'],'Independent six-method union differs')
    require(audit['covered_checks']==['__main__.NeurodiagIndependentAudit.'+m for m in methods] and common.same(evidence,audit['evidence']),'Independent aggregate evidence differs')
    return bindings


def build():
    baseline=common.baseline();inputs=dict(baseline['inputs']);checks=[];groups={}
    def add(values):
        require(all(k not in inputs or inputs[k]==v for k,v in values.items()),'Conflicting pinned evidence')
        inputs.update(values)
    repair=audit_repair();add(repair['originals_after']);add(repair['retained_failure_inputs'])
    lock=lock_failure()
    for values in (lock['cache_sha256'],lock['originals_unchanged'],lock['retained_inputs'],lock['original_retained_fixture_inputs']):add(values)
    for group,report,source in (
        ('data_owner','build/neurodiag-data-author.json','tests/neurodiag_data_author.py'),
        ('metrics_owner','build/neurodiag-metrics-author.json','tests/neurodiag_metrics_author.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(ids);add(proof['inputs'])
    audit=common.read(ROOT/'build/neurodiag-audit.json')
    require(audit['schema']=='neurodiag-independent-audit-v1' and audit['passed'] is True and type(audit['tests_run']) is int and audit['tests_run']==6
        and all(type(audit[k]) is int and audit[k]==0 for k in ('errors','failures','skips')),'Independent audit incomplete')
    add(audit['inputs']);ids=independent_groups(audit);groups['independent']=len(ids);checks.extend(ids)
    require(len(checks)==len(set(checks))==14,'Current unique executed methods differ')
    recovery=audit['evidence']['live_batch_recovery']['tests']
    require(len(recovery)==2 and {r['event'] for r in recovery}=={'neurodiag_precommit','neurodiag_committed'}
        and [r['committed_prefix'] for r in recovery]==[1,2] and [r['committed_units'] for r in recovery]==[16,32]
        and all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),'Actual kill/recovery evidence missing')
    for record in recovery:raw_recovery(record)
    prepared=common.read(ROOT/'build/neurodiag-input-preparation.json');add(prepared['inputs'])
    require(prepared['passed'] is prepared['models_fixed'] is prepared['policy_fixed'] is True and prepared['torch_imported'] is False
        and all(prepared[k]==0 for k in ('model_forwards','native_calls','new_teacher_searches','new_PUCT_searches','new_games','optimizer_updates')),'Input guard contract differs')
    require(prepared['counts']==common.read(p.INPUTS)['counts'] and prepared['counts']['literal_encoder_calls']==28
        and prepared['counts']['literal_generated_missing_vectors']==13 and prepared['counts']['literal_cache_comparisons']==15,'Declared feature generation differs')
    guarded=common.read(ROOT/'build/neurodiag-production.json');add(guarded['inputs'])
    require(guarded['passed'] is guarded['models_fixed'] is guarded['policy_fixed'] is guarded['public_CLI_used'] is True and guarded['torch_imported'] is False
        and guarded['unit_count']==308 and guarded['target_records']==360 and guarded['batch_count']==20
        and guarded['actual_calls']==dict(observe_batches=20,shared_trunk_forwards=308,value_head_evaluations=616,target_diagnostics=720,summary_calls=1)
        and all(guarded[k]==0 for k in ('native_calls','new_teacher_searches','new_PUCT_searches','new_games','optimizer_updates')),'Guarded production count/scope differs')
    result=p.verify();cache=p.raw_complete();p.certify(p.DEFAULT,cache)
    require(result['certified'] is result['passed'] is True and result['unit_count']==308 and result['target_records']==360 and result['batch_count']==20,'Certified support differs')
    require(result['policy_fixed'] is result['models_fixed'] is True and result['training_performed'] is result['weights_applied'] is result['promotion'] is False
        and result['new_teacher_searches']==result['new_PUCT_searches']==result['new_games']==0,'Diagnostic scope exceeded')
    rerun=common.read(ROOT/'build/neurodiag-rerun.json');add(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True
        and rerun['old_inputs_preserved']==5377 and rerun['producer_durable_file_inventory_unchanged'] is True and rerun['torch_imported'] is False
        and rerun['versioned_reader']==2 and rerun['certified_completed_writer_lock_acquisitions']==0,'Completed cache reopened with work')
    require(all(rerun[k]==0 for k in ('new_observation_commits','new_batch_commits','native_calls','new_training_updates','new_teacher_searches','new_PUCT_searches',
        'new_model_forwards','new_derivatives','new_SVDs','new_summary_calls','new_feature_encodings','new_games')),'Unexpected reopened computation')
    for name,sig in rerun['after'].items():
        path=ROOT/name;require(common.sha(path)==sig['sha256'] and path.stat().st_size==sig['size'] and path.stat().st_mtime_ns==sig['mtime_ns'],'Reopened evidence changed: '+name)
    names=list(result['inputs'])+['congklak/neurodiag_progress.py','tools/neurodiag-dashboard.html','congklak/neurodiag_common.py']
    names += [path.relative_to(ROOT).as_posix() for path in (ROOT/'tools').glob('*neurodiag*.py')]
    names += [path.relative_to(ROOT).as_posix() for path in (ROOT/'docs').glob('NEURODIAG*.md')]
    for path in (ROOT/'build').glob('neurodiag*'):
        if path.name=='neurodiag-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(item.relative_to(ROOT).as_posix() for item in path.rglob('*') if durable(item))
    names.extend(path.relative_to(ROOT).as_posix() for path in p.DEFAULT.rglob('*') if durable(path))
    from congklak import neurodiag_progress as progress
    require(not progress.snapshot()['active_tools'],'Diagnostic worker still active')
    interrupted=[]
    for path in (ROOT/'build/neurodiag-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running');interrupted.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    add(common.pins(names));common.current(inputs)
    return common.publish(OUTPUT,dict(schema='neurodiag-acceptance-v1',passed=True,tests_run=14,covered_checks=checks,groups=groups,errors=0,failures=0,skips=0,
        independent_complete=True,old_inputs_preserved=5377,actual_interruption_recovery_kills=2,accepted_interruption_recovery_kills=2,
        unit_count=308,target_records=360,batch_count=20,summary_sha256=config.digest(result['summary']),policy_fixed=True,all_model_weights_fixed=True,
        weights_applied=False,training_performed=False,new_teacher_searches=0,new_PUCT_searches=0,new_games=0,promotion=False,
        production_calls=guarded['actual_calls'],engineering_fixture_work_separate=True,new_reservations=False,diagnostic_only=True,
        retained_failed_audit_attempts=2,retained_failed_method_executions=3,failed_attempt_counted_as_passed=False,
        completed_Windows_lock_issue_fixed_by_versioned_reader=True,public_completed_reader='congklak.neurodiag_v2',
        independent_execution_sources={Path(g['path']).stem:g['executed_source'] for g in audit['groups']},
        completed_no_work=True,interrupted_observer_runs=sorted(interrupted),inputs=dict(sorted(inputs.items()))))


def verify():
    body=common.read(OUTPUT);expected=build();require(common.same(body,expected),'Acceptance does not rederive');return body


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('build','verify'));args=parser.parse_args()
    result=build() if args.operation=='build' else verify()
    print(config.canonical(dict(event='neurodiag_acceptance_'+args.operation,passed=True,tests_run=result['tests_run'],inputs=len(result['inputs']))),flush=True)
