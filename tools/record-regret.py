"""Seal the existing-data regret investigation and executed independent checks."""
import ast
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,regret_common as common,regret_drift as drift
require=common.require
OUTPUT=ROOT/'build/regret-acceptance.json'
EXCLUDED={'regret-acceptance.json','regret-acceptance-rerun.json','regret-dashboard-final.json'}


def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)


def saved_cases(path):
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        return db.execute('SELECT idx,body,hash FROM cases ORDER BY idx').fetchall()


def raw_recovery(record):
    import json
    probe=(ROOT/record['probe']).resolve();run=(ROOT/record['run']).resolve()
    require(probe.is_relative_to(ROOT/'build') and run.is_relative_to(ROOT/'build'),'Recovery escaped build')
    marker=common.read(probe/'marker.json')
    require(common.same(marker,record['marker']) and marker['event']==record['event'] and marker['completed']==record['committed_prefix'],'Kill marker differs')
    events=[]
    for line in (probe/'worker.log').read_text(encoding='utf-8').splitlines():
        try:body=json.loads(line)
        except ValueError:continue
        if isinstance(body,dict):events.append(body)
    require(any(common.same(value,marker) for value in events),'Kill marker lacks flushed worker event')
    prefix=probe/'validated-prefix.sqlite3';final=run/'work.sqlite3'
    def metadata(path):
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            return {key:(body,digest) for key,body,digest in db.execute('SELECT key,body,hash FROM meta')}
    old_meta,new_meta=metadata(prefix),metadata(final)
    require(set(old_meta)=={'identity'} and set(new_meta)=={'identity','complete'} and old_meta['identity']==new_meta['identity'],'Recovery metadata identity/complete boundary differs')
    before=saved_cases(prefix);after=saved_cases(final)
    require(len(before)==record['committed_prefix'] and before==after[:len(before)],'Raw committed case prefix changed')


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
    expected={'preflight':3,'production':2,'final':1}
    source='tests/regret_independent_audit.py'
    freeze=common.read(ROOT/'build/regret-audit-source-freeze.json')
    tree=ast.parse((ROOT/source).read_text(encoding='utf-8-sig'))
    available={method.name for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RegretIndependentAudit' for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
    records=audit['groups'];require(len(records)==3,'Three independent groups required')
    seen=set();methods=[];evidence={};bindings=[]
    for entry in records:
        path=entry['path'];name=Path(path).stem
        require(name in expected and name not in seen and path=='build/regret-audit/'+name+'.json','Group identity differs');seen.add(name)
        part=common.read(ROOT/path);checksum=common.sha(ROOT/path)
        require(entry['sha256']==checksum==audit['inputs'].get(path),'Group byte binding differs')
        require(part['schema']=='regret-independent-group-v1' and part['group']==name and part['passed'] is True,'Independent group failed')
        require(all(type(part[k]) is int and part[k]==0 for k in ('errors','failures','skips')),'Independent failures/skips')
        ids=part['covered_methods']
        require(type(part['tests_run']) is int and part['tests_run']==entry['tests_run']==expected[name] and len(ids)==len(set(ids))==expected[name] and ids==entry['covered_methods']==freeze['groups'][name],'Group coverage differs')
        require(entry['executed_source']['path']==source and entry['executed_source']['sha256']==common.sha(ROOT/source)==part['inputs'].get(source),'Executed source differs')
        require(set(ids)<=available and part['covered_checks']==['__main__.RegretIndependentAudit.'+m for m in ids],'Method identities differ')
        require(all(audit['inputs'].get(k)==v for k,v in part['inputs'].items()),'Aggregate omitted group inputs')
        require(not set(evidence).intersection(part['evidence']),'Duplicated independent evidence')
        evidence.update(part['evidence']);methods.extend(ids);bindings.extend(source+'::RegretIndependentAudit.'+m for m in ids)
    require(seen==set(expected) and len(methods)==len(set(methods))==6 and set(methods)==available and methods==audit['covered_methods'],'Independent union differs')
    require(audit['covered_checks']==['__main__.RegretIndependentAudit.'+m for m in methods] and common.same(evidence,audit['evidence']),'Aggregate evidence differs')
    return bindings


def build():
    baseline=common.baseline();inputs=dict(baseline['inputs']);checks=[];groups={}
    def add(values):
        require(all(k not in inputs or inputs[k]==v for k,v in values.items()),'Conflicting evidence input identity')
        inputs.update(values)
    for group,report,source in (
        ('producer_owner','build/regret-drift-author.json','tests/regret_drift_author.py'),
        ('forensics_owner','build/regret-forensics-author.json','tests/regret_forensics_author.py')):
        proof,ids=executed(report,source);groups[group]=len(ids);checks.extend(ids);add(proof['inputs'])
    audit=common.read(ROOT/'build/regret-audit.json')
    require(audit['schema']=='regret-independent-audit-v1' and audit['passed'] is True and audit['tests_run']==6 and all(type(audit[k]) is int and audit[k]==0 for k in ('errors','failures','skips')),'Independent audit incomplete')
    add(audit['inputs'])
    ids=independent_groups(audit);groups['independent']=len(ids);checks.extend(ids)
    require(len(checks)==len(set(checks))==14,'Current unique method count differs')
    recovery=audit['evidence']['live_case_recovery']['tests']
    require(len(recovery)==2 and [r['committed_prefix'] for r in recovery]==[1,2] and all(r['actual_process_kill'] is r['committed_prefix_bytes_preserved'] is r['fresh_semantics_equal'] is True for r in recovery),'Actual recovery proofs missing')
    for record in recovery:raw_recovery(record)
    guarded=common.read(ROOT/'build/regret-production.json');add(guarded['inputs'])
    require(guarded['passed'] is guarded['models_fixed'] is guarded['policy_fixed'] is guarded['public_CLI_used'] is True and guarded['cases']==guarded['new_case_commits']==5 and guarded['torch_imported'] is False and all(guarded[k]==0 for k in ('native_calls','new_model_forwards','new_teacher_searches','new_PUCT_searches','new_training_updates','new_games')),'Guarded public production evidence differs')
    cases,manifest=drift.load_rows();result=drift.verify()
    require(len(cases)==manifest['cases']==result['cases']==5 and result['status']=='complete' and result['summary']['all16_context'] is True,'Case support differs')
    ledger_ref=audit['evidence']['exact_attribution_and_forensics']['all16_action_evidence']
    require(ledger_ref['roots']==16 and ledger_ref['endpoint_Q_entries']==246 and ledger_ref['horizon_pairs']==3 and ledger_ref['all_action_signed_deltas_contributions_best_nonbest_and_ranks'] is True,'All16 action evidence absent')
    ledger=common.read(ROOT/ledger_ref['path'])
    require(common.sha(ROOT/ledger_ref['path'])==ledger_ref['sha256']==audit['inputs'].get(ledger_ref['path']) and ledger['schema']=='regret-independent-all16-action-attribution-v1' and ledger['root_count']==16 and set(ledger['roots'])==set(result['context']['roots']),'All16 action evidence identity differs')
    from fractions import Fraction
    for pair in ledger['horizon_pairs']:
        total=Fraction()
        for key,row in ledger['roots'].items():
            values=row['pairs'][pair];actions=values['actions'];legal=row['legal_actions']
            require(set(actions)=={str(a) for a in legal},'Missing context action')
            contribution=sum((drift.fraction(v['global_weighted_minus_uniform']) for v in actions.values()),Fraction())
            require(contribution==drift.fraction(values['global_delta'])==drift.fraction(result['context']['roots'][key]['pairs'][pair]['global_delta']),'All16 action/root contribution join differs')
            total+=contribution
        require(total==drift.fraction(result['context']['pairs'][pair]['global_delta']),'All16 action/global sum differs')
    require(result['weights_applied'] is result['training_performed'] is result['promotion'] is False and result['previous_allocation']=='hold' and result['new_teacher_searches']==result['new_NN_forwards']==result['new_games']==0,'Scope exceeded')
    rerun=common.read(ROOT/'build/regret-rerun.json');add(rerun['sources'])
    require(rerun['passed'] is True and rerun['before']==rerun['after'] and rerun['all_bytes_sizes_mtimes_unchanged'] is True and rerun['old_inputs_preserved']==5252 and rerun['producer_durable_file_inventory_unchanged'] is True and rerun['torch_imported'] is False,'Completed reopen differs')
    require(all(rerun[k]==0 for k in ('new_case_commits','native_calls','new_training_updates','new_teacher_searches','new_PUCT_searches','new_model_forwards','new_games')),'Unexpected reopened work')
    for name,sig in rerun['after'].items():
        p=ROOT/name;require(common.sha(p)==sig['sha256'] and p.stat().st_size==sig['size'] and p.stat().st_mtime_ns==sig['mtime_ns'],'Reopened evidence changed: '+name)
    names=list(result['inputs'])+['congklak/regret_progress.py','tools/regret-dashboard.html','congklak/regret_common.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*regret*.py')]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').glob('REGRET*.md')]
    for path in (ROOT/'build').glob('regret*'):
        if path.name=='regret-progress':continue
        if path.is_file() and path.name not in EXCLUDED:names.append(path.relative_to(ROOT).as_posix())
        elif path.is_dir():names.extend(p.relative_to(ROOT).as_posix() for p in path.rglob('*') if durable(p))
    names.extend(p.relative_to(ROOT).as_posix() for p in drift.DEFAULT.rglob('*') if durable(p))
    from congklak import regret_progress as progress
    require(not progress.snapshot()['active_tools'],'Diagnostic worker active')
    interrupted=[]
    for path in (ROOT/'build/regret-progress').glob('run-*.json'):
        run=common.read(path)
        if run['state']=='running':
            require(list(progress.old.process_identity(run['pid']))!=run['process_identity'],'Worker still running');interrupted.append(path.relative_to(ROOT).as_posix())
        names.extend([path.relative_to(ROOT).as_posix(),run['log']])
    add(common.pins(names));common.current(inputs)
    return common.publish(OUTPUT,dict(schema='regret-acceptance-v1',passed=True,tests_run=14,covered_checks=checks,groups=groups,errors=0,failures=0,skips=0,independent_complete=True,old_inputs_preserved=5252,actual_interruption_recovery_kills=2,accepted_interruption_recovery_kills=2,summary=result['summary'],cases=5,context_roots=16,policy_fixed=True,all_model_weights_fixed=True,weights_applied=False,training_performed=False,new_teacher_searches=0,new_model_forwards=0,new_games=0,promotion=False,previous_allocation='hold',new_reservations=False,diagnostic_only=True,completed_no_work=True,interrupted_observer_runs=sorted(interrupted),inputs=dict(sorted(inputs.items()))))


def verify():
    body=common.read(OUTPUT);expected=build();require(common.same(body,expected),'Acceptance does not rederive');return body


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('build','verify'));a=p.parse_args();r=build() if a.operation=='build' else verify()
    print(config.canonical(dict(event='regret_acceptance_'+a.operation,passed=True,tests_run=r['tests_run'],inputs=len(r['inputs']))),flush=True)
