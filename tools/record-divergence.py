"""Seal executed source-bound diagnostic and recovery evidence."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,divergence_study_common as common,divergence_search as search,divergence_teacher as teacher,divergence_comparison as comparison
require=common.require
OUTPUT=ROOT/'build/divergence-acceptance.json'
EXCLUDED={'divergence-acceptance.json','divergence-acceptance-rerun.json','divergence-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


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
        ('search_owner','build/divergence-search-author.json','tests/divergence_search_author.py'),
        ('teacher_owner','build/divergence-teacher-author.json','tests/divergence_teacher_author.py'),
        ('independent','build/divergence-audit.json','tests/divergence_independent_audit.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(source+'::'+'.'.join(x.split('.')[-2:]) for x in ids);inputs.update(proof['inputs'])
    require(groups['independent']==8 and len(checks)==len(set(checks)),'Independent/unique coverage differs')
    audit=common.read(ROOT/'build/divergence-audit.json')
    recovery=audit['evidence']['live_search_recovery']['tests']+audit['evidence']['live_teacher_recovery']['tests']
    require(len(recovery)==4 and [r['committed_prefix'] for r in recovery]==[1,2,3,4] and
            all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),
            'Four actual interruption/recovery proofs missing')
    witness=search.primary_witness();rows,stores,tm=teacher.load_rows();result=comparison.verify()
    require(tm['requested_attempts']==15 and tm['total_new_nodes']<=85000000,'Teacher requested coverage differs')
    require(all(len(stores[s+'_attempts'])==5 for s in teacher.STAGES),'Missing requested teacher attempts')
    require(result['policy_fixed'] is result['all_model_weights_fixed'] is True and result['training_performed'] is result['promotion'] is False,'Scope changed')
    rerun=common.read(ROOT/'build/divergence-rerun.json');common.current(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved']==3914 and rerun['reports_rebuilt_and_verified']==3 and rerun['torch_imported'] is False,'No-work proof differs')
    require(all(rerun[k]==0 for k in ('new_training_updates','new_teacher_searches','new_PUCT_searches','new_games')),'Unexpected reopening work')
    for name,sig in rerun['after'].items():
        p=ROOT/name;require(common.sha(p)==sig['sha256'] and p.stat().st_size==sig['size'] and p.stat().st_mtime_ns==sig['mtime_ns'],'Reopened artifact changed: '+name)
    names=list(result['inputs'])+['congklak/divergence_progress.py','tools/divergence-dashboard.html','congklak/divergence_study_common.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*divergence*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').glob('DIVERGENCE*.md')]
    for path in (ROOT/'build').glob('divergence*'):
        if path.name=='divergence-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    for path in (ROOT/'data').glob('divergence*-v1'):
        names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    from congklak import divergence_progress as progress
    require(not progress.snapshot()['active_tools'],'A diagnostic worker is active')
    interrupted_observer_runs=[]
    for path in (ROOT/'build/divergence-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running')
            interrupted_observer_runs.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    inputs.update(common.pins(names));common.current(inputs)
    body=dict(schema='divergence-acceptance-v1',passed=True,tests_run=len(checks),covered_checks=checks,groups=groups,
        errors=0,failures=0,skips=0,independent_complete=True,old_inputs_preserved=3914,
        actual_interruption_recovery_kills=4,
        fixed_work=witness,teacher_summary=tm['summary'],teacher_attempts=15,teacher_total_nodes=tm['total_new_nodes'],
        policy_fixed=True,all_model_weights_fixed=True,training_performed=False,new_games=0,promotion=False,previous_allocation='hold',
        diagnostic_only=True,completed_no_work=True,interrupted_observer_runs=sorted(interrupted_observer_runs),inputs=dict(sorted(inputs.items())))
    return common.publish(OUTPUT,body)


def verify():
    body=common.read(OUTPUT);require(body['passed'] is body['policy_fixed'] is body['all_model_weights_fixed'] is True and body['training_performed'] is body['promotion'] is False,'Acceptance scope invalid')
    expected=build();require(config.canonical(body)==config.canonical(expected),'Acceptance does not fully rederive');return body


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('build','verify'));a=p.parse_args();r=build() if a.operation=='build' else verify()
    print(config.canonical(dict(event='divergence_acceptance_'+a.operation,passed=True,tests_run=r['tests_run'],inputs=len(r['inputs']))),flush=True)
