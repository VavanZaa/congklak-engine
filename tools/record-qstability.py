"""Seal the prospective Q-stability assessment and executed independent checks."""
import ast
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,qstability_common as common,qstability_weighting as weighting
require=common.require
OUTPUT=ROOT/'build/qstability-acceptance.json'
EXCLUDED={'qstability-acceptance.json','qstability-acceptance-rerun.json','qstability-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def saved_rows(path,table):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute('SELECT state,body,hash FROM '+table+' ORDER BY state').fetchall()


def raw_recovery(record):
    import json
    probe=(ROOT/record['probe']).resolve();run=(ROOT/record['run']).resolve()
    require(probe.is_relative_to(ROOT/'build') and run.is_relative_to(ROOT/'build'),'Recovery escaped build')
    marker=common.read(probe/'marker.json')
    require(common.same(marker,record['marker']) and marker['event']==record['event'] and marker['completed']==record['committed_aggregate_prefix'] and marker['stage_completed']==record['committed_prefix'],'Kill marker differs')
    events=[]
    for line in (probe/'worker.log').read_text(encoding='utf-8').splitlines():
        try:body=json.loads(line)
        except ValueError:continue
        if isinstance(body,dict):events.append(body)
    require(any(common.same(value,marker) for value in events),'Kill lacks flushed event')
    prefix=probe/'validated-prefix.sqlite3';final=run/'work.sqlite3'
    def metadata(path):
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            return {key:(body,digest) for key,body,digest in db.execute('SELECT key,body,hash FROM meta')}
    old_meta,new_meta=metadata(prefix),metadata(final)
    require('complete' not in old_meta and 'complete' in new_meta and old_meta['identity']==new_meta['identity'],'Recovery identity/complete boundary differs')
    count=record['committed_prefix']
    expected=dict(roots=2,h8_attempts=2,h8_labels=2,h10_attempts=count,h10_labels=count,h12_attempts=0,h12_labels=0)
    for table,n in expected.items():
        before=saved_rows(prefix,table);after=saved_rows(final,table)
        require(len(before)==n and before==after[:n],'Raw committed teacher prefix changed: '+table)
    require(2+count==record['committed_aggregate_prefix'],'Aggregate prefix differs')


def executed(report_name,source_name):
    body=common.read(ROOT/report_name)
    require(body['passed'] is True and all(type(body[k]) is int and body[k]==0 for k in ('errors','failures','skips')),'Owner suite failed')
    ids=body.get('covered_checks',body.get('covered_methods'))
    require(type(body['tests_run']) is int and len(ids)==len(set(ids))==body['tests_run']==4,'Owner coverage differs')
    tree=ast.parse((ROOT/source_name).read_text(encoding='utf-8-sig'))
    expected={node.name+'.'+method.name for node in tree.body if isinstance(node,ast.ClassDef) for method in node.body if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name.startswith('test_')}
    actual={'.'.join(name.split('.')[-2:]) for name in ids}
    require(actual==expected,'Source owner methods differ')
    require(body['inputs'].get(source_name)==common.sha(ROOT/source_name),'Owner source unpinned')
    return body,[source_name+'::'+'.'.join(name.split('.')[-2:]) for name in ids]


def independent_groups(audit):
    expected={'preflight':3,'reference':1,'final':2}
    source='tests/qstability_independent_audit.py'
    freeze=common.read(ROOT/'build/qstability-audit-source-freeze.json')
    tree=ast.parse((ROOT/source).read_text(encoding='utf-8-sig'))
    available={method.name for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='QStabilityIndependentAudit' for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
    records=audit['groups'];require(len(records)==3,'Three independent groups required')
    seen=set();methods=[];evidence={};bindings=[]
    for entry in records:
        path=entry['path'];name=Path(path).stem
        require(name in expected and name not in seen and path=='build/qstability-audit/'+name+'.json','Group identity differs');seen.add(name)
        part=common.read(ROOT/path);checksum=common.sha(ROOT/path)
        require(entry['sha256']==checksum==audit['inputs'].get(path),'Group byte binding differs')
        require(part['schema']=='qstability-independent-group-v1' and part['group']==name and part['passed'] is True,'Independent group failed')
        require(all(type(part[k]) is int and part[k]==0 for k in ('errors','failures','skips')),'Independent failures/skips')
        ids=part['covered_methods']
        require(type(part['tests_run']) is int and part['tests_run']==entry['tests_run']==expected[name] and len(ids)==len(set(ids))==expected[name] and ids==entry['covered_methods']==freeze['groups'][name],'Group coverage differs')
        require(entry['executed_source']['path']==source and entry['executed_source']['sha256']==common.sha(ROOT/source)==part['inputs'].get(source),'Executed source differs')
        require(set(ids)<=available and part['covered_checks']==['__main__.QStabilityIndependentAudit.'+m for m in ids],'Method identities differ')
        require(all(audit['inputs'].get(k)==v for k,v in part['inputs'].items()),'Aggregate omitted group inputs')
        require(not set(evidence).intersection(part['evidence']),'Duplicated independent evidence')
        evidence.update(part['evidence']);methods.extend(ids);bindings.extend(source+'::QStabilityIndependentAudit.'+m for m in ids)
    require(seen==set(expected) and len(methods)==len(set(methods))==6 and set(methods)==available and methods==audit['covered_methods'],'Independent union differs')
    require(audit['covered_checks']==['__main__.QStabilityIndependentAudit.'+m for m in methods] and common.same(evidence,audit['evidence']),'Aggregate evidence differs')
    return bindings


def build():
    baseline=common.baseline();inputs=dict(baseline['inputs']);checks=[];groups={}
    def add(values):
        require(all(k not in inputs or inputs[k]==v for k,v in values.items()),'Conflicting evidence input identity')
        inputs.update(values)
    for group,report,source in (
        ('producer_owner','build/qstability-weighting-author.json','tests/qstability_weighting_author.py'),
        ('metrics_owner','build/qstability-metrics-author.json','tests/qstability_metrics_author.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(ids);add(proof['inputs'])
    audit=common.read(ROOT/'build/qstability-audit.json')
    require(audit['schema']=='qstability-independent-audit-v1' and audit['passed'] is True and audit['tests_run']==6 and all(type(audit[k]) is int and audit[k]==0 for k in ('errors','failures','skips')),'Independent audit incomplete')
    add(audit['inputs']);ids=independent_groups(audit);groups['independent']=len(ids);checks.extend(ids)
    require(len(checks)==len(set(checks))==14,'Current unique method count differs')
    recovery=audit['evidence']['live_teacher_recovery']['tests']
    require(len(recovery)==2 and [r['committed_prefix'] for r in recovery]==[1,2] and [r['committed_aggregate_prefix'] for r in recovery]==[3,4] and all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),'Actual recovery proofs missing')
    for record in recovery:raw_recovery(record)
    rows,stores,manifest=weighting.load_rows();result=weighting.verify()
    require(len(rows)==result['requested_roots']==16 and manifest['status'] in ('complete','references_incomplete_hold') and result['status']==manifest['status'],'Closed assessment support differs')
    attempts={stage:len(stores[stage+'_attempts']) for stage in weighting.STAGES}
    complete={stage:len(stores[stage+'_labels']) for stage in weighting.STAGES}
    require(attempts['h8']==attempts['h10']==16 and manifest['total_new_nodes']<=480000000,'Reference attempts or bound differ')
    if manifest['status']=='references_incomplete_hold':
        require(attempts['h12']==complete['h12']==0 and (complete['h8']<16 or complete['h10']<16) and not weighting.RULE.exists() and result['assessment_available'] is False and result['protocol_closed'] is True,'Reference HOLD contract differs')
    else:
        require(complete['h8']==complete['h10']==attempts['h12']==16 and weighting.RULE.is_file(),'H12 admission/support differs')
        require(common.read(ROOT/'build/qstability-audit/reference.json')['evidence']['references']['H12_ready'] is True,'Explicit H12 readiness absent')
        require(result['screen']['passes'] is False or complete['h12']==16,'Incomplete endpoint screen passed')
    require(result['weights_applied'] is result['training_performed'] is result['promotion'] is False and result['previous_allocation']=='hold','Scope exceeded')
    reserve=common.read(weighting.RESERVATIONS)
    require(reserve['future_optimizer_reserved'] is True and reserve['historical_roles_rewritten'] is False,'Reservations changed')
    rerun=common.read(ROOT/'build/qstability-rerun.json');add(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved']==5050 and rerun['producer_durable_file_inventory_unchanged'] is True and rerun['torch_imported'] is False,'Completed reopen differs')
    require(all(rerun[k]==0 for k in ('new_attempt_commits','new_training_updates','new_teacher_searches','new_PUCT_searches','new_model_forwards','new_games')),'Unexpected reopened work')
    for name,sig in rerun['after'].items():
        path=ROOT/name;require(common.sha(path)==sig['sha256'] and path.stat().st_size==sig['size'] and path.stat().st_mtime_ns==sig['mtime_ns'],'Reopened evidence changed: '+name)
    names=list(result['inputs'])+['congklak/qstability_progress.py','tools/qstability-dashboard.html','congklak/qstability_common.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*qstability*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').glob('QSTABILITY*.md')]
    for path in (ROOT/'build').glob('qstability*'):
        if path.name=='qstability-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    names.extend(p.relative_to(ROOT).as_posix() for p in weighting.DEFAULT.rglob('*') if durable(p))
    from congklak import qstability_progress as progress
    require(not progress.snapshot()['active_tools'],'Worker active')
    interrupted=[]
    for path in (ROOT/'build/qstability-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running');interrupted.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    add(common.pins(names));common.current(inputs)
    return common.publish(OUTPUT,dict(schema='qstability-acceptance-v1',passed=True,tests_run=14,covered_checks=checks,groups=groups,errors=0,failures=0,skips=0,independent_complete=True,old_inputs_preserved=5050,actual_interruption_recovery_kills=2,accepted_interruption_recovery_kills=2,teacher_attempts=attempts,teacher_complete=complete,teacher_total_nodes=manifest['total_new_nodes'],weighting_screen=result['screen'],assessment_status=result['status'],policy_fixed=True,all_model_weights_fixed=True,weights_applied=False,training_performed=False,new_model_forwards=0,new_games=0,promotion=False,previous_allocation='hold',future_optimizer_reservations=True,diagnostic_only=True,completed_no_work=True,interrupted_observer_runs=sorted(interrupted),inputs=dict(sorted(inputs.items()))))


def verify():
    body=common.read(OUTPUT);expected=build();require(common.same(body,expected),'Acceptance does not rederive');return body


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('build','verify'));a=p.parse_args();r=build() if a.operation=='build' else verify()
    print(config.canonical(dict(event='qstability_acceptance_'+a.operation,passed=True,tests_run=r['tests_run'],inputs=len(r['inputs']))),flush=True)
