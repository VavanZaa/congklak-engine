"""Seal executed source-bound diagnostic and recovery evidence."""
import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,stratum_common as common,stratum_weighting as weighting
require=common.require
OUTPUT=ROOT/'build/stratum-acceptance.json'
EXCLUDED={'stratum-acceptance.json','stratum-acceptance-rerun.json','stratum-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def saved_rows(path,table,columns,order):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute('SELECT '+columns+' FROM '+table+' ORDER BY '+order).fetchall()


def raw_recovery(record):
    probe=(ROOT/record['probe']).resolve();run=(ROOT/record['run']).resolve()
    require(probe.is_relative_to(ROOT/'build') and run.is_relative_to(ROOT/'build'),'Recovery fixture escaped build')
    marker=common.read(probe/'marker.json');require(common.same(marker,record['marker']) and marker['event']==record['event'],'Independent kill marker differs')
    stdout=[]
    for line in (probe/'worker.log').read_text(encoding='utf-8').splitlines():
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict):stdout.append(value)
    require(any(common.same(value,marker) for value in stdout),'Kill marker lacks flushed worker event')
    prefix=probe/'validated-prefix.sqlite3';final=run/'work.sqlite3';count=record['committed_prefix']
    expected=dict(roots=2,h8_attempts=2,h8_labels=2,h10_attempts=count,h10_labels=count,h12_attempts=0,h12_labels=0)
    for table,n in expected.items():
        before=saved_rows(prefix,table,'state,body,hash','state')
        require(len(before)==n and before==saved_rows(final,table,'state,body,hash','state')[:n],'Raw teacher prefix changed: '+table)
    require(expected['h8_attempts']+expected['h10_attempts']==record['committed_aggregate_prefix'],'Aggregate prefix differs')


def executed(report_name,source_name):
    body=common.read(ROOT/report_name)
    require(body['passed'] is True and all(type(body[k]) is int and body[k]==0 for k in ('errors','failures','skips')),'Suite failed: '+report_name)
    ids=body['covered_checks'] if 'covered_checks' in body else body['covered_methods']
    require(type(body['tests_run']) is int and len(ids)==len(set(ids))==body['tests_run']>0,'Executed coverage differs')
    tree=ast.parse((ROOT/source_name).read_text(encoding='utf-8'))
    expected={node.name+'.'+method.name for node in tree.body if isinstance(node,ast.ClassDef) for method in node.body if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name.startswith('test_')}
    actual={'.'.join(name.split('.')[-2:]) for name in ids}
    require(actual==expected,'Source method coverage differs: '+source_name)
    common.current(body['inputs']);require(body['inputs'].get(source_name)==common.sha(ROOT/source_name),'Unpinned executed test source')
    return body,ids


def independent_groups(audit):
    expected={'preflight':3,'rule':1,'labels':2,'final':2}
    original='tests/stratum_independent_audit.py';late='build/stratum-audit-final-v2.py'
    original_freeze=common.read(ROOT/'build/stratum-audit-source-freeze.json')
    records=audit['groups'];require(isinstance(records,list) and len(records)==4,'Four independent group records required')
    seen=set();methods=[];evidence={};bindings=[]
    for entry in records:
        path=entry['path'];name='final' if path=='build/stratum-audit/final-v2.json' else Path(path).stem
        require(name in expected and name not in seen and path=='build/stratum-audit/'+('final-v2' if name=='final' else name)+'.json','Independent group identity differs');seen.add(name)
        part=common.read(ROOT/path);checksum=common.sha(ROOT/path)
        require(entry['sha256']==checksum==audit['inputs'].get(path),'Independent group byte binding differs')
        require(part['schema']=='stratum-independent-group-v1' and part['group']==name and part['passed'] is True,'Independent group failed or misidentified')
        require(all(type(part[k]) is int and part[k]==0 for k in ('errors','failures','skips')),'Independent group has failed checks')
        ids=part['covered_methods'];require(type(part['tests_run']) is int and part['tests_run']==entry['tests_run']==expected[name] and len(ids)==len(set(ids))==expected[name] and ids==entry['covered_methods'],'Independent group coverage differs')
        require(ids==original_freeze['groups'][name],'Independent group changed original semantic coverage')
        source=late if name=='final' else original
        require(entry['executed_source']['path']==source and entry['executed_source']['sha256']==common.sha(ROOT/source)==part['inputs'].get(source),'Actual executed test-source binding differs')
        tree=ast.parse((ROOT/source).read_text(encoding='utf-8'))
        source_methods={method.name for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='StratumIndependentAudit' for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
        require(set(ids)<=source_methods,'Executed methods absent from their actual source')
        bindings.extend(source+'::StratumIndependentAudit.'+method for method in ids)
        require(part['covered_checks']==['__main__.StratumIndependentAudit.'+m for m in ids],'Independent group full method IDs differ')
        require(all(audit['inputs'].get(key)==value for key,value in part['inputs'].items()),'Aggregate omitted or changed group inputs')
        require(not set(evidence).intersection(part['evidence']),'Independent evidence duplicated across groups')
        evidence.update(part['evidence']);methods.extend(ids)
    require(seen==set(expected) and len(methods)==len(set(methods))==8 and methods==audit['covered_methods'],'Independent group union differs')
    require(audit['covered_checks']==['__main__.StratumIndependentAudit.'+m for m in methods] and common.same(evidence,audit['evidence']),'Independent aggregate evidence or method binding differs')
    require(common.read(ROOT/'build/stratum-audit-final-source-freeze-v2.json')['groups']==original_freeze['groups'],'Late runner changed declared scope')
    return bindings


def build():
    baseline=common.baseline(verify_all=True);inputs=dict(baseline['inputs']);checks=[];groups={}
    for group,report,source in (
        ('metrics_owner','build/stratum-metrics-author.json','tests/stratum_metrics_author.py'),
        ('weighting_owner','build/stratum-weighting-author.json','tests/stratum_weighting_author.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(source+'::'+'.'.join(x.split('.')[-2:]) for x in ids);inputs.update(proof['inputs'])
    audit=common.read(ROOT/'build/stratum-audit.json')
    require(audit['schema']=='stratum-independent-audit-v1' and audit['passed'] is True and audit['tests_run']==8 and all(type(audit[k]) is int and audit[k]==0 for k in ('errors','failures','skips')),'Independent aggregate failed')
    common.current(audit['inputs']);inputs.update(audit['inputs'])
    bindings=independent_groups(audit);groups['independent']=len(bindings);checks.extend(bindings)
    require(groups['independent']==8 and len(checks)==len(set(checks)),'Independent/unique coverage differs')
    recovery=audit['evidence']['live_teacher_recovery']['tests']
    require(len(recovery)==2 and [r['committed_prefix'] for r in recovery]==[1,2] and
            [r['committed_aggregate_prefix'] for r in recovery]==[3,4] and
            all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),
            'Two actual interruption/recovery proofs missing')
    for record in recovery:raw_recovery(record)
    historical=common.read(ROOT/'build/stratum-audit/first-definition-order-failure-v1/diagnostic.json')
    require(historical['actual_process_kills']==2 and historical['accepted_current_coverage'] is False and historical['formal_group_report_published'] is False,'Historical failed-run recovery provenance differs')
    common.current(historical['raw_inputs'])
    rows,stores,manifest=weighting.load_rows();result=weighting.verify()
    require(len(rows)==16 and manifest['total_new_nodes']<=480000000,'Assessment support or node bound differs')
    attempts={stage:len(stores[stage+'_attempts']) for stage in ('h8','h10','h12')}
    require(attempts==dict(h8=16,h10=16,h12=16),'Missing unconditional requested attempts')
    require(result['weights_applied'] is result['training_performed'] is result['promotion'] is False and result['previous_allocation']=='hold','Assessment changed training allocation')
    reservations=common.read(weighting.RESERVATIONS)
    require(reservations['future_optimizer_reserved'] is True and reservations['historical_roles_rewritten'] is False,'Prospective protection changed')
    rerun=common.read(ROOT/'build/stratum-rerun.json');common.current(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved']==4546 and rerun['closed_paths_verified']==2 and rerun['producer_durable_file_inventory_unchanged'] is True and rerun['torch_imported'] is False,'No-work proof differs')
    require(all(rerun[k]==0 for k in ('new_training_updates','new_teacher_searches','new_PUCT_searches','new_inference_runtime_activations','new_games')),'Unexpected reopening work')
    for name,sig in rerun['after'].items():
        p=ROOT/name;require(common.sha(p)==sig['sha256'] and p.stat().st_size==sig['size'] and p.stat().st_mtime_ns==sig['mtime_ns'],'Reopened artifact changed: '+name)
    names=list(result['inputs'])+['congklak/stratum_progress.py','tools/stratum-dashboard.html','congklak/stratum_common.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*stratum*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').glob('STRATUM*.md')]
    for path in (ROOT/'build').glob('stratum*'):
        if path.name=='stratum-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    for path in (ROOT/'data').glob('stratum*-v1'):
        names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    from congklak import stratum_progress as progress
    require(not progress.snapshot()['active_tools'],'A diagnostic worker is active')
    interrupted_observer_runs=[]
    for path in (ROOT/'build/stratum-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running')
            interrupted_observer_runs.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    inputs.update(common.pins(names));common.current(inputs)
    body=dict(schema='stratum-acceptance-v1',passed=True,tests_run=len(checks),covered_checks=checks,groups=groups,
        errors=0,failures=0,skips=0,independent_complete=True,old_inputs_preserved=4546,
        actual_interruption_recovery_kills=4,accepted_interruption_recovery_kills=2,historical_interruption_recovery_kills=2,
        teacher_summary=manifest['summary'],teacher_attempts=attempts,teacher_total_nodes=manifest['total_new_nodes'],
        weighting_screen=result['screen'],weights_applied=False,future_optimizer_reservations=True,
        policy_fixed=True,all_model_weights_fixed=True,training_performed=False,new_games=0,promotion=False,previous_allocation='hold',
        diagnostic_only=True,completed_no_work=True,interrupted_observer_runs=sorted(interrupted_observer_runs),inputs=dict(sorted(inputs.items())))
    return common.publish(OUTPUT,body)


def verify():
    body=common.read(OUTPUT);require(body['passed'] is body['policy_fixed'] is body['all_model_weights_fixed'] is True and body['training_performed'] is body['promotion'] is False,'Acceptance scope invalid')
    expected=build();require(config.canonical(body)==config.canonical(expected),'Acceptance does not fully rederive');return body


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('build','verify'));a=p.parse_args();r=build() if a.operation=='build' else verify()
    print(config.canonical(dict(event='stratum_acceptance_'+a.operation,passed=True,tests_run=r['tests_run'],inputs=len(r['inputs']))),flush=True)
