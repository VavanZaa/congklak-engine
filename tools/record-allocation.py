"""Seal executed source-bound diagnostic and recovery evidence."""
import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,allocation_common as common,allocation_trace as trace,allocation_weighting as weighting
require=common.require
OUTPUT=ROOT/'build/allocation-acceptance.json'
EXCLUDED={'allocation-acceptance.json','allocation-acceptance-rerun.json','allocation-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def saved_rows(path,table,columns,order):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute('SELECT '+columns+' FROM '+table+' ORDER BY '+order).fetchall()


def raw_recovery(record,kind):
    probe=(ROOT/record['probe']).resolve();run=(ROOT/record['run']).resolve()
    require(probe.is_relative_to(ROOT/'build') and run.is_relative_to(ROOT/'build'),'Recovery fixture escaped build')
    marker=common.read(probe/'marker.json');require(common.same(marker,record['marker']) and marker['event']==record['event'] and marker['completed']==record['committed_prefix'],'Independent kill marker differs')
    stdout=[]
    for line in (probe/'worker.log').read_text(encoding='utf-8').splitlines():
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict):stdout.append(value)
    require(any(common.same(value,marker) for value in stdout),'Kill marker lacks flushed worker event')
    prefix=probe/'validated-prefix.sqlite3';final=run/'work.sqlite3';count=record['committed_prefix']
    if kind=='trace':
        before=saved_rows(prefix,'steps','cell,seq,body,hash','cell,seq')
        require(len(before)==count and before==saved_rows(final,'steps','cell,seq,body,hash','cell,seq')[:count],'Raw trace prefix changed')
        setups=saved_rows(prefix,'root_setups','cell,body,hash','cell')
        require(len(setups)==1 and setups==saved_rows(final,'root_setups','cell,body,hash','cell')[:1] and not saved_rows(prefix,'cells','idx,body,hash','idx'),'Wrong physical trace boundary')
    else:
        for table in ('roots','h8_references','h10_references','h12_attempts','h12_labels'):
            before=saved_rows(prefix,table,'state,body,hash','state')
            require(len(before)==(2 if table in ('roots','h8_references','h10_references') else count) and before==saved_rows(final,table,'state,body,hash','state')[:len(before)],'Raw teacher prefix or references changed')


def executed(report_name,source_name):
    body=common.read(ROOT/report_name)
    require(body['passed'] is True and all(type(body[k]) is int and body[k]==0 for k in ('errors','failures','skips')),'Suite failed: '+report_name)
    ids=body.get('covered_checks',body['covered_methods'])
    require(type(body['tests_run']) is int and len(ids)==len(set(ids))==body['tests_run']>0,'Executed coverage differs')
    tree=ast.parse((ROOT/source_name).read_text(encoding='utf-8'))
    expected={node.name+'.'+method.name for node in tree.body if isinstance(node,ast.ClassDef) for method in node.body if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)) and method.name.startswith('test_')}
    actual={'.'.join(name.split('.')[-2:]) for name in ids}
    require(actual==expected,'Source method coverage differs: '+source_name)
    common.current(body['inputs']);require(body['inputs'].get(source_name)==common.sha(ROOT/source_name),'Unpinned executed test source')
    return body,ids


def build():
    baseline=common.baseline(verify_all=True);inputs=dict(baseline['inputs']);checks=[];groups={}
    for group,report,source in (
        ('trace_owner','build/allocation-trace-author.json','tests/allocation_trace_author.py'),
        ('weighting_owner','build/allocation-weighting-author.json','tests/allocation_weighting_author.py'),
        ('independent','build/allocation-audit.json','tests/allocation_independent_audit.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(source+'::'+'.'.join(x.split('.')[-2:]) for x in ids);inputs.update(proof['inputs'])
    require(groups['independent']==8 and len(checks)==len(set(checks)),'Independent/unique coverage differs')
    audit=common.read(ROOT/'build/allocation-audit.json')
    recovery=audit['evidence']['live_trace_recovery']['tests']+audit['evidence']['live_teacher_recovery']['tests']
    require(len(recovery)==4 and [r['committed_prefix'] for r in recovery]==[1,2,1,2] and
            all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),
            'Four actual interruption/recovery proofs missing')
    for index,record in enumerate(recovery):raw_recovery(record,'trace' if index<2 else 'teacher')
    witness=trace.primary_witness();rows,stores,manifest=weighting.load_rows();result=weighting.verify()
    require(len(rows)==16 and manifest['total_new_nodes']<=160000000,'Assessment support or requested node bound differs')
    require(len(stores['h12_attempts'])==16,'Missing unconditional H12 attempts')
    require(result['weights_applied'] is result['training_performed'] is result['promotion'] is False and result['previous_allocation']=='hold','Assessment changed training allocation')
    reservations=common.read(weighting.RESERVATIONS)
    require(reservations['future_optimizer_reserved'] is True and reservations['historical_roles_rewritten'] is False,'Prospective protection changed')
    rerun=common.read(ROOT/'build/allocation-rerun.json');common.current(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved']==4255 and rerun['closed_paths_verified']==3 and rerun['producer_durable_file_inventory_unchanged'] is True and rerun['torch_imported'] is False,'No-work proof differs')
    require(all(rerun[k]==0 for k in ('new_training_updates','new_teacher_searches','new_PUCT_searches','new_inference_runtime_activations','new_games')),'Unexpected reopening work')
    for name,sig in rerun['after'].items():
        p=ROOT/name;require(common.sha(p)==sig['sha256'] and p.stat().st_size==sig['size'] and p.stat().st_mtime_ns==sig['mtime_ns'],'Reopened artifact changed: '+name)
    names=list(result['inputs'])+['congklak/allocation_progress.py','tools/allocation-dashboard.html','congklak/allocation_common.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*allocation*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').glob('ALLOCATION*.md')]
    for path in (ROOT/'build').glob('allocation*'):
        if path.name=='allocation-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    for path in (ROOT/'data').glob('allocation*-v1'):
        names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    from congklak import allocation_progress as progress
    require(not progress.snapshot()['active_tools'],'A diagnostic worker is active')
    interrupted_observer_runs=[]
    for path in (ROOT/'build/allocation-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running')
            interrupted_observer_runs.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    inputs.update(common.pins(names));common.current(inputs)
    body=dict(schema='allocation-acceptance-v1',passed=True,tests_run=len(checks),covered_checks=checks,groups=groups,
        errors=0,failures=0,skips=0,independent_complete=True,old_inputs_preserved=4255,
        actual_interruption_recovery_kills=4,
        fixed_work=witness,teacher_summary=manifest['summary'],teacher_attempts=16,teacher_total_nodes=manifest['total_new_nodes'],
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
    print(config.canonical(dict(event='allocation_acceptance_'+a.operation,passed=True,tests_run=r['tests_run'],inputs=len(r['inputs']))),flush=True)
