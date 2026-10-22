"""Recoverable metadata-only ownership resolution; never loads an engine."""
import argparse
import os
from pathlib import Path
import sys
import time
from . import config, ownership_resolution_common as stage
from . import ownership_resolution_contract as contract
from .valuepilot_reservations import Writer

ROOT, FOLDER = stage.ROOT, stage.FOLDER
DEFAULT, PLAN = FOLDER/'production', FOLDER/'plan.json'
SOURCES = ('congklak/ownership_resolution_job.py',
           'congklak/ownership_resolution_common.py',
           'congklak/ownership_resolution_contract.py',
           'congklak/ownership_resolution_epoch.py',
           'tools/run-ownership-resolution.py')
need, read, sha, atomic = stage.need, stage.read, stage.sha, stage.atomic


def modules():
    from . import ownership_resolution_provenance as history
    from . import ownership_resolution_support as support
    from . import ownership_resolution_epoch as epoch
    return history, support, epoch


def input_pins():
    history,support,epoch=modules()
    inputs=stage.common.pins([*SOURCES,*history.SOURCES,*support.SOURCES,
        *epoch.SOURCES,FOLDER/'contract.json',FOLDER/'baseline.json',stage.PRIOR,*stage.MODELS])
    inputs.update(history.input_pins());inputs.update(support.input_pins());inputs.update(epoch.input_pins())
    return inputs


def prepare_plan(path=PLAN, fixture=False):
    path=stage.path_of(path);history,support,epoch=modules()
    tasks=[dict(kind='history_inventory'),dict(kind='support_inventory')]
    if not fixture:
        tasks.extend(dict(kind='trace_episode', episode=t) for t in history.plan())
        tasks.extend([dict(kind='overlay_projection'),dict(kind='joint_projection'),dict(kind='epoch')])
    tasks=[dict(t,index=i) for i,t in enumerate(tasks)]
    inputs=input_pins()
    body=dict(schema=stage.SCHEMA+'-plan', fixture=fixture,tasks=tasks,inputs=inputs,
        contract=contract.definition(),contract_sha256=config.digest(contract.definition()),
        models_fixed=True,policy_fixed=True,activity=stage.ZERO,
        training_authorized=False,production_evaluation_admitted=False)
    need(not fixture or path.is_relative_to(FOLDER/'fixtures'),'Fixture plan outside fixture namespace')
    atomic(path,body);return stage.normalized(body)


def validate_plan(path, check_sources=True):
    path=stage.path_of(path);p=read(path)
    need(p['schema']==stage.SCHEMA+'-plan' and config.digest(p['contract'])==config.digest(contract.definition()) and
        p['contract_sha256']==config.digest(contract.definition()) and
        p['models_fixed'] is True and p['policy_fixed'] is True and
        p['activity']==stage.ZERO and all(type(v) is int for v in p['activity'].values()) and
        p['training_authorized'] is False and p['production_evaluation_admitted'] is False,
        'Resolution plan scope differs')
    need(type(p['fixture']) is bool and len(p['tasks'])>=2 and
        all(type(t['index']) is int and t['index']==i for i,t in enumerate(p['tasks'])) and
        [t['kind'] for t in p['tasks'][:2]]==['history_inventory','support_inventory'],
        'Resolution task prefix differs')
    if p['fixture']:
        need(path.is_relative_to(FOLDER/'fixtures') and len(p['tasks'])==2,
            'Fixture contains non-metadata work')
    else:
        history,_,_=modules()
        expected=[dict(kind='history_inventory'),dict(kind='support_inventory')]
        expected.extend(dict(kind='trace_episode',episode=t) for t in history.plan())
        expected.extend([dict(kind='overlay_projection'),dict(kind='joint_projection'),dict(kind='epoch')])
        need(config.canonical(p['tasks'])==config.canonical([dict(t,index=i) for i,t in enumerate(expected)]),'Finite accepted-source task plan differs')
    need(p['inputs']==input_pins(),'Resolution plan source closure differs')
    if check_sources:stage.common.current(p['inputs'])
    return p


def prepare(path=DEFAULT, plan_path=PLAN):
    path=stage.path_of(path);plan_path=stage.path_of(plan_path);p=validate_plan(plan_path)
    need(not p['fixture'] or path.is_relative_to(FOLDER/'fixtures'),'Fixture output outside fixture namespace')
    identity=dict(schema=stage.SCHEMA+'-identity',plan_path=stage.relative(plan_path),
        plan_sha256=sha(plan_path),plan=p,models=stage.common.pins(stage.MODELS),
        models_fixed=True,policy_fixed=True,activity=stage.ZERO,
        pre_run_review=_review_ref(plan_path,p))
    atomic(path/'identity.json',identity);return stage.normalized(identity)


def _identity(path, check_sources=True):
    identity=read(path/'identity.json');p=validate_plan(ROOT/identity['plan_path'],check_sources)
    expected=dict(schema=stage.SCHEMA+'-identity',plan_path=identity['plan_path'],
        plan_sha256=sha(ROOT/identity['plan_path']),plan=p,models=stage.common.pins(stage.MODELS),
        models_fixed=True,policy_fixed=True,activity=stage.ZERO,
        pre_run_review=_review_ref(ROOT/identity['plan_path'],p))
    need(config.canonical(identity)==config.canonical(expected),'Resolution identity differs')
    return identity


def _review_ref(plan_path,plan):
    if plan['fixture']:return None
    path=FOLDER/'audit/pre-run-review.json';review=read(path)
    need(review['schema']==stage.SCHEMA+'-pre-run-review' and review['passed'] is True and
        review['independent_of_implementation'] is True and review['plan_path']==stage.relative(plan_path) and
        review['plan_sha256']==sha(plan_path) and review['source_inputs']==plan['inputs'] and
        review['models_fixed'] is True and review['policy_fixed'] is True and
        review['activity']==stage.ZERO and all(type(v) is int for v in review['activity'].values()),
        'Independent pre-run source/plan review differs')
    return dict(path=stage.relative(path),sha256=sha(path))


def _prior(results,kind):
    selected=[r['result'] for r in results if r['kind']==kind]
    need(len(selected)==1,'Missing/duplicate prior unit '+kind);return selected[0]


def _compute(task,results):
    history,support,epoch=modules();kind=task['kind']
    if kind=='history_inventory':return history.inventory()
    if kind=='support_inventory':return support.inventory()
    if kind=='trace_episode':return history.unit(task['episode'])
    traces=[r['result'] for r in results if r['kind']=='trace_episode']
    if kind=='overlay_projection':return history.final_projection(traces)
    if kind=='joint_projection':return support.final_projection(_prior(results,'overlay_projection'),epoch.selection())
    need(kind=='epoch','Unknown metadata task')
    return epoch.build(_prior(results,'history_inventory'),_prior(results,'support_inventory'),
        _prior(results,'overlay_projection'),_prior(results,'joint_projection'))


def _validate(task,result,results):
    history,support,epoch=modules();kind=task['kind']
    need(type(result) is dict,'Resolution unit is not an object')
    need(result.get('models_fixed') is True and result.get('policy_fixed') is True and
        result.get('activity')==stage.ZERO and
        all(type(result['activity'][k]) is int for k in stage.ZERO),
        'Metadata unit contradicts fixed-model or zero-work scope')
    if kind=='history_inventory':history.validate_inventory(result)
    elif kind=='support_inventory':support.validate_inventory(result)
    elif kind=='trace_episode':history.validate_unit(task['episode'],result)
    elif kind=='overlay_projection':history.validate_projection(result,[r['result'] for r in results if r['kind']=='trace_episode'])
    elif kind=='joint_projection':support.validate_projection(result,_prior(results,'overlay_projection'),epoch.selection())
    else:
        need(kind=='epoch','Unknown metadata task')
        epoch.validate(result,_prior(results,'history_inventory'),_prior(results,'support_inventory'),
            _prior(results,'overlay_projection'),_prior(results,'joint_projection'))


def _envelope(path,identity,task,results):
    up=path/'units'/f'{task["index"]:06d}.json';u=read(up)
    need(set(u)=={'schema','identity_sha256','index','kind','task_sha256',
        'producer_invocation_id','result_sha256','result'} and
        u['schema']==stage.SCHEMA+'-unit' and u['identity_sha256']==sha(path/'identity.json') and
        type(u['index']) is int and u['index']==task['index'] and u['kind']==task['kind'] and
        u['task_sha256']==config.digest(task) and type(u['producer_invocation_id']) is str and
        u['result_sha256']==config.digest(u['result']), 'Unit identity/task/result differs')
    _validate(task,u['result'],results);return u


def replay(path,identity):
    tasks=identity['plan']['tasks'];commits=sorted((path/'commits').glob('*.json'))
    need(len(commits)<=len(tasks),'Too many committed units');results=[];previous=None
    for i,cp in enumerate(commits):
        need(cp.name==f'{i:06d}.json','Resolution commit sequence gap')
        up=path/'units'/f'{i:06d}.json';c=read(cp)
        expected=dict(schema=stage.SCHEMA+'-commit',index=i,
            identity_sha256=sha(path/'identity.json'),task_sha256=config.digest(tasks[i]),
            unit_path=stage.relative(up),unit_sha256=sha(up),previous_sha256=previous)
        need(config.canonical(c)==config.canonical(expected),'Resolution commit chain differs')
        u=_envelope(path,identity,tasks[i],results);results.append(u);previous=sha(cp)
    # At most the immediately next immutable result may be uncommitted.
    files=sorted((path/'units').glob('*.json'))
    need([p.name for p in files]==[f'{i:06d}.json' for i in range(len(files))] and
         len(commits)<=len(files)<=min(len(tasks),len(commits)+1),'Unexpected orphan unit or result gap')
    if len(files)>len(commits):_envelope(path,identity,tasks[len(commits)],results)
    return dict(results=results,previous_sha256=previous)


def summary(identity,results):
    need(len(results)==len(identity['plan']['tasks']),'Incomplete metadata plan')
    result=dict(schema=stage.SCHEMA+'-report',passed=True,fixture=identity['plan']['fixture'],
        units=len(results),unit_results=[dict(index=r['index'],kind=r['kind'],result_sha256=r['result_sha256']) for r in results],
        activity=stage.ZERO,models_fixed=True,policy_fixed=True,training_authorized=False,
        production_evaluation_admitted=False,historical_provenance_complete=False,
        historically_unseen=False,inherited_unknown_background=True)
    if not identity['plan']['fixture']:result['epoch']=_prior(results,'epoch')
    return stage.normalized(result)


def completed(path=DEFAULT):
    """Pure verification: no writer, producer, engine or new reservation."""
    path=stage.path_of(path);identity=_identity(path);data=replay(path,identity)
    expected=summary(identity,data['results']);report=read(path/'report.json');cat=read(path/'catalog.json')
    expected_cat=dict(schema=stage.SCHEMA+'-complete',passed=True,
        identity_sha256=sha(path/'identity.json'),report_sha256=sha(path/'report.json'),
        commits=stage.common.pins(sorted((path/'commits').glob('*.json'))))
    need(config.canonical(report)==config.canonical(expected) and config.canonical(cat)==config.canonical(expected_cat),
        'Resolution report/catalog differs')
    return dict(report=report,catalog=cat)


def _pause(path,point,requested,marker):
    if point!=requested:return
    from .tracing_speed_resources import process_identity
    need(path.is_relative_to(FOLDER/'fixtures'),'Pause requires fixture output')
    marker=stage.path_of(marker,'fixtures')
    atomic(marker,dict(point=point,process=process_identity(),argv=list(sys.argv),
        worker_source_sha256=sha(__file__),identity_sha256=sha(path/'identity.json')))
    while True:time.sleep(.05)


def run(path=DEFAULT,plan_path=PLAN,pause_at=None,pause_marker=None):
    path=stage.path_of(path)
    if (path/'catalog.json').exists():return completed(path)
    need(pause_at is None or pause_at in contract.KILL_POINTS,'Unknown metadata pause boundary')
    need(pause_at is None or path.is_relative_to(FOLDER/'fixtures'),'Fixture pause outside fixture namespace')
    if (path/'identity.json').exists():
        identity=_identity(path);replay(path,identity) # Refuse corruption before the writer.
    else:
        p=validate_plan(plan_path);_review_ref(plan_path,p)
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return completed(path)
        identity=prepare(path,plan_path);data=replay(path,identity);results=data['results'];previous=data['previous_sha256']
        for task in identity['plan']['tasks'][len(results):]:
            i=task['index'];up=path/'units'/f'{i:06d}.json'
            if up.exists():u=_envelope(path,identity,task,results)
            else:
                result=stage.normalized(_compute(task,results));_validate(task,result,results)
                u=dict(schema=stage.SCHEMA+'-unit',identity_sha256=sha(path/'identity.json'),
                    index=i,kind=task['kind'],task_sha256=config.digest(task),
                    producer_invocation_id=f'{os.getpid()}/{time.monotonic_ns()}',
                    result_sha256=config.digest(result),result=result)
                atomic(up,u)
            _pause(path,'before_unit_commit',pause_at,pause_marker)
            cp=path/'commits'/f'{i:06d}.json'
            atomic(cp,dict(schema=stage.SCHEMA+'-commit',index=i,
                identity_sha256=sha(path/'identity.json'),task_sha256=config.digest(task),
                unit_path=stage.relative(up),unit_sha256=sha(up),previous_sha256=previous))
            previous=sha(cp);results.append(stage.normalized(u))
            _pause(path,'after_unit_commit',pause_at,pause_marker)
        # No restore/context append: completed-prefix publication reuses exact bytes.
        atomic(path/'report.json',summary(identity,results))
        cat=dict(schema=stage.SCHEMA+'-complete',passed=True,identity_sha256=sha(path/'identity.json'),
            report_sha256=sha(path/'report.json'),commits=stage.common.pins(sorted((path/'commits').glob('*.json'))))
        _pause(path,'before_catalog',pause_at,pause_marker);atomic(path/'catalog.json',cat)
        _pause(path,'after_catalog',pause_at,pause_marker)
    return completed(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('plan','prepare','run','verify'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--fixture',action='store_true');parser.add_argument('--pause-at');parser.add_argument('--pause-marker',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.operation=='plan':result=prepare_plan(args.plan,args.fixture)
        elif args.operation=='prepare':result=prepare(args.output,args.plan)
        elif args.operation=='verify':result=completed(args.output)
        else:result=run(args.output,args.plan,args.pause_at,args.pause_marker)
        print(config.canonical(dict(passed=True,operation=args.operation,
            report=result.get('report'),plan_sha256=sha(args.plan))),flush=True);return 0
    except Exception as exc:
        print(config.canonical(dict(passed=False,state='hold',error=type(exc).__name__+': '+str(exc))),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
