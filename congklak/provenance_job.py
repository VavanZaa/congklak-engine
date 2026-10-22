"""Recoverable, source-bound publication of actual provenance witnesses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from . import config,ownership_implementation_common as stage
from .valuepilot_reservations import Writer

ROOT,FOLDER=stage.ROOT,stage.FOLDER
DEFAULT=FOLDER/'provenance/production'
SCHEMA='certified-provenance-job-v1'
SOURCES=('congklak/provenance_job.py','congklak/provenance_reconstruction.py',
         'congklak/ownership_implementation_common.py','tools/run-provenance-job.py')
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic


def module():
    from . import provenance_reconstruction
    return provenance_reconstruction


def prepare(path,fixture_tasks=None):
    baseline=stage.baseline();plan=module().plan()
    inputs=dict(baseline['inputs']);new=dict(plan['inputs'])
    new.update(stage.common.pins([FOLDER/'baseline.json',*SOURCES]))
    need(all(k not in inputs or inputs[k]==v for k,v in new.items()),'Provenance input conflicts with accepted history')
    inputs.update(new);stage.common.current(inputs)
    need(fixture_tasks is None or (path.is_relative_to(FOLDER/'provenance/fixtures') and
         type(fixture_tasks) is int and 2<=fixture_tasks<=len(plan['tasks'])),'Invalid provenance fixture task limit')
    tasks=plan['tasks'] if fixture_tasks is None else plan['tasks'][:fixture_tasks]
    body=dict(schema=SCHEMA+'-identity',inputs=inputs,tasks=tasks,full_plan_tasks=len(plan['tasks']),
              fixture_tasks=fixture_tasks,engineering_fixture=fixture_tasks is not None,
              plan_sha256=config.digest(plan),models_fixed=True,policy_fixed=True,training_counters=stage.TRAINING_ZERO)
    return atomic(path/'identity.json',body)


def prefix(path,identity):
    result=[];previous=None;identity_sha=sha(path/'identity.json')
    for index,filename in enumerate(sorted((path/'commits').glob('*.json'))):
        need(filename.name==f'{index:06d}.json' and index<len(identity['tasks']),'Provenance commit gap/extra record')
        row=read(filename)
        need(row['index']==index and row['identity_sha256']==identity_sha and row['previous_commit_sha256']==previous and
             row['task_sha256']==config.digest(identity['tasks'][index]),'Provenance commit identity differs')
        expected=f'units/{index:06d}-{row["result_sha256"]}.json'
        need(row['result']==expected and sha(path/expected)==row['result_sha256'],'Provenance result missing/corrupt')
        result.append(read(path/expected));previous=sha(filename)
    return result


def completed(path=DEFAULT):
    path=stage.path_of(path,'provenance');cat=read(path/'catalog.json');identity=read(path/'identity.json')
    need(cat['schema']==SCHEMA+'-complete' and cat['passed'] is True and
         cat['identity_sha256']==sha(path/'identity.json') and identity['schema']==SCHEMA+'-identity' and
         identity['models_fixed'] is True and identity['policy_fixed'] is True and
         identity['training_counters']==stage.TRAINING_ZERO,'Provenance completion scope differs')
    stage.common.current(identity['inputs']);units=prefix(path,identity)
    need(len(units)==len(identity['tasks'])==cat['committed_units'] and
         cat['commits']==stage.common.pins(sorted((path/'commits').glob('*.json'))),'Provenance completion prefix differs')
    need(sha(path/'report.json')==cat['report_sha256'],'Provenance report corrupt')
    report=read(path/'report.json')
    need(cat['training_counters']==stage.TRAINING_ZERO,'Provenance training counters differ')
    planned=module().plan()
    need(identity['plan_sha256']==config.digest(planned) and
         identity['full_plan_tasks']==len(planned['tasks']) and
         all(identity['inputs'].get(k)==v for k,v in planned['inputs'].items()),
         'Provenance source plan differs')
    fixture=identity['fixture_tasks']
    need(identity['engineering_fixture']==(fixture is not None) and
         identity['tasks']==(planned['tasks'] if fixture is None else planned['tasks'][:fixture]),
         'Provenance task coverage differs')
    expected=(dict(schema=SCHEMA+'-fixture-report',certificate_eligible=False,engineering_fixture=True,
                   units=len(units),source_plan_sha256=identity['plan_sha256'],training_counters=stage.TRAINING_ZERO)
              if fixture is not None else module().summarize(units))
    need(report==stage.normalized(expected),'Provenance report semantics differ')
    return dict(catalog=cat,identity=identity,report=report)


def pause(path,point,requested,marker,index):
    if point!=requested:return
    need(path!=DEFAULT.resolve() and marker is not None,'Provenance pause hooks are fixtures only')
    marker=stage.path_of(marker,'provenance/fixtures')
    atomic(marker,dict(point=point,index=index,pid=os.getpid()))
    while True:time.sleep(.1)


def run(path=DEFAULT,pause_at=None,pause_marker=None,pause_index=1,fixture_tasks=None):
    path=stage.path_of(path,'provenance')
    if (path/'catalog.json').exists():return completed(path)
    need(pause_at in (None,'before_unit_commit','after_unit_commit','before_catalog'),'Unknown provenance pause')
    need(pause_at is None or path!=DEFAULT.resolve(),'Production pause forbidden')
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return completed(path)
        identity=prepare(path,fixture_tasks);units=prefix(path,identity);identity_sha=sha(path/'identity.json')
        sources={k:identity['inputs'][k] for k in SOURCES}
        for index in range(len(units),len(identity['tasks'])):
            stage.common.current(sources)
            body=stage.normalized(module().unit(identity['tasks'][index]));stage.common.current(sources)
            digest=hashlib.sha256((config.canonical(body)+'\n').encode()).hexdigest()
            relative=f'units/{index:06d}-{digest}.json';atomic(path/relative,body)
            if index==pause_index:pause(path,'before_unit_commit',pause_at,pause_marker,index)
            row=dict(index=index,identity_sha256=identity_sha,task_sha256=config.digest(identity['tasks'][index]),
                result=relative,result_sha256=digest,
                previous_commit_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None)
            atomic(path/'commits'/f'{index:06d}.json',row);units.append(body)
            if index==pause_index:pause(path,'after_unit_commit',pause_at,pause_marker,index)
            print(json.dumps(dict(event='provenance_unit_committed',index=index,total=len(identity['tasks']))),flush=True)
        report=(dict(schema=SCHEMA+'-fixture-report',certificate_eligible=False,engineering_fixture=True,
                     units=len(units),source_plan_sha256=identity['plan_sha256'],training_counters=stage.TRAINING_ZERO)
                if identity['engineering_fixture'] else module().summarize(units))
        report=stage.normalized(report);atomic(path/'report.json',report)
        stage.common.current(identity['inputs'])
        cat=dict(schema=SCHEMA+'-complete',passed=True,identity_sha256=identity_sha,committed_units=len(units),
            commits=stage.common.pins(sorted((path/'commits').glob('*.json'))),report_sha256=sha(path/'report.json'),
            training_counters=stage.TRAINING_ZERO)
        pause(path,'before_catalog',pause_at,pause_marker,len(units));atomic(path/'catalog.json',cat)
    return completed(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('baseline','run','verify'))
    parser.add_argument('--path',type=Path,default=DEFAULT)
    parser.add_argument('--pause-at',choices=('before_unit_commit','after_unit_commit','before_catalog'))
    parser.add_argument('--pause-marker',type=Path);parser.add_argument('--pause-index',type=int,default=1)
    parser.add_argument('--fixture-tasks',type=int)
    args=parser.parse_args()
    if args.operation=='baseline':result=dict(passed=True,prior_inputs=stage.baseline()['inputs_count'])
    else:
        body=completed(args.path) if args.operation=='verify' else run(args.path,args.pause_at,args.pause_marker,args.pause_index,args.fixture_tasks)
        result=dict(passed=True,units=body['catalog']['committed_units'],training_counters=stage.TRAINING_ZERO)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
