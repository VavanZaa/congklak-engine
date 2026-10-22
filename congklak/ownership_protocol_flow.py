"""Recoverable publication of a prospective protocol; execution stays blocked."""
import argparse
import json
from pathlib import Path
import sys
import time
import os
from . import config, neurodiag_common as common, ownership_protocol as contract
from .value_diagnostics import atomic
from .valuepilot_reservations import Writer

ROOT = common.ROOT
FOLDER = ROOT / 'build/ownership-protocol-v1'
DEFAULT = FOLDER / 'production'
SCHEMA = 'prospective-ownership-publication-v1'
ZERO = dict(optimizer_steps=0, optimizer_draws=0, new_games=0, new_teacher_labels=0,
            searches=0, feature_simulations=0)
DESIGN = 'build/ownership-protocol-v1/design/section-spec.json'
SOURCES = ('congklak/ownership_protocol.py', 'congklak/ownership_protocol_flow.py',
           'congklak/ownership_domain_projection.py', 'tools/run-ownership-protocol.py',
           'tests/test_ownership_protocol.py', 'tests/test_ownership_domain_projection.py',
           'docs/PROSPECTIVE_OWNERSHIP_PROTOCOL.md', DESIGN,
           'build/ownership-protocol-v1/design/compiled-section-spec.json')
need, read, sha = common.require, common.read, common.sha


def path_of(path):
    path = Path(path).resolve()
    need(path == DEFAULT.resolve() or path.is_relative_to(FOLDER / 'fixtures'), 'Protocol output namespace differs')
    return path


def baseline():
    path = FOLDER / 'baseline.json'
    if path.exists():
        body = read(path)
        need(body['schema'] == SCHEMA + '-baseline' and body['passed'] is True and
             body['inputs_count'] == len(body['inputs']) == 45202, 'Protocol baseline differs')
        common.current(body['inputs']); return body
    accepted = ROOT / 'build/value-diagnostics-v1/acceptance.json'; prior = read(accepted)
    need(prior['passed'] is True and len(prior['inputs']) == 45201, 'Previous diagnostics acceptance differs')
    inputs = dict(prior['inputs']); inputs.update(common.pins([accepted])); common.current(inputs)
    return atomic(path, dict(schema=SCHEMA + '-baseline', passed=True, inputs_count=len(inputs), inputs=inputs))


def projection_module():
    from . import ownership_domain_projection
    return ownership_domain_projection


def section_spec():
    design = read(ROOT / DESIGN)
    need(design['schema'] == 'prospective-ownership-protocol-design-v1' and
         design['definition_only'] is True and design['policy_fixed'] is True and
         design['training_authorized'] is False and design['execution_ready'] is False and
         design['new_games_authorized'] is False and design['new_starts_authorized'] is False and
         design['new_labels_authorized'] is False and design['old_arena_reclassification'] is False and
         design['prior_holds_preserved'] is True and design['candidate_quarantined'] is True and
         design['current_counters'] == ZERO, 'Design authorization or historical scope differs')
    for model in design['fixed_models'].values():
        need(sha(ROOT / model['path']) == model['sha256'], 'Design model changed')
    active = design['active_seeds']
    need([active[k] for k in ('shared_domain_max','ownership_domain_min','changed_branch_max','arena_start_min','arena_start_max')] ==
         [24,25,74,25,74] and active['temporary_seeds_in_hand_arrays_define_domain'] is False and
         active['K24_exact_claim'] is False, 'Design boundary differs')
    arena = design['future_arena']; statistics = design['statistics']; graph = design['graph']
    need(arena['planned_games'] == 192 and arena['whole_decision_ms'] == 25 and
         arena['baselines'] == ['H_v1','H8_relay_CPU'] and
         statistics['uncertainty_enabled'] is False and statistics['graph_disjointness_proves_IID'] is False and
         graph['retained_incidence_predicate'] == 'active_seeds >=25' and
         graph['family_to_quotient_cartesian_inference'] is False and
         graph['public_opening_ancestor_exemption'] is False and
         design['estimand']['suffix_depends_on_comparison_arm'] is False,
         'Design ownership/statistical contract differs')
    result = dict(schema='prospective-ownership-section-spec-v1', machine_contract=contract.contract(),
                  design=design, training_authorized=False, execution_ready=False, old_arena_hold_preserved=True)
    contract.validate_spec(result)
    compiled=FOLDER/'design/compiled-section-spec.json'
    if compiled.exists():need(read(compiled)==result,'Compiled section spec differs from source design/contract')
    return result


def prepare(path):
    prior = baseline(); projection = projection_module().plan()
    tasks = [dict(index=0,kind='contract',task=dict(id='scratch-contract'))]
    for task in projection['tasks']: tasks.append(dict(index=len(tasks),kind='projection',task=task))
    tasks.append(dict(index=len(tasks),kind='section',task=dict(id='prospective-section')))
    inputs = dict(prior['inputs']); new = dict(projection['inputs'])
    new.update(common.pins([FOLDER / 'baseline.json', *SOURCES]))
    need(all(k not in inputs or inputs[k] == v for k,v in new.items()), 'Prior protocol input changed')
    inputs.update(new); common.current(inputs)
    identity = dict(schema=SCHEMA+'-identity',inputs=inputs,tasks=tasks,counters=ZERO,
                    models_fixed=True,policy_fixed=True,execution_ready=False,training_authorized=False)
    section_spec()
    return atomic(path/'identity.json',identity)


def validate_identity(identity):
    need(identity['schema']==SCHEMA+'-identity' and identity['counters']==ZERO and
         identity['models_fixed'] is True and identity['policy_fixed'] is True and
         identity['execution_ready'] is False and identity['training_authorized'] is False,
         'Protocol identity scope differs')
    old=read(FOLDER/'baseline.json')
    need(old['inputs_count']==len(old['inputs'])==45202 and
         all(identity['inputs'].get(k)==v for k,v in old['inputs'].items()) and
         all(k in identity['inputs'] for k in SOURCES), 'Protocol inherited input coverage differs')
    tasks=identity['tasks']
    need(len(tasks)==195 and [t['index'] for t in tasks]==list(range(195)) and
         [t['kind'] for t in tasks]==['contract']+['projection']*193+['section'], 'Protocol task sequence differs')


def prefix(path,identity):
    results=[]; previous=None; identity_sha=sha(path/'identity.json')
    for index,filename in enumerate(sorted((path/'commits').glob('*.json'))):
        need(filename.name==f'{index:06d}.json' and index<len(identity['tasks']), 'Protocol commit gap or extra commit')
        row=read(filename)
        need(row['index']==index and row['identity_sha256']==identity_sha and
             row['task_sha256']==config.digest(identity['tasks'][index]) and
             row['previous_commit_sha256']==previous, 'Protocol commit chain differs')
        expected=f'units/{index:06d}-{row["result_sha256"]}.json'
        need(row['result']==expected and sha(path/expected)==row['result_sha256'], 'Protocol unit hash differs')
        results.append(read(path/expected)); previous=sha(filename)
    return results


def compute(entry):
    if entry['kind']=='contract': return contract.fixture_checks()
    if entry['kind']=='projection': return projection_module().unit(entry['task'])
    need(entry['kind']=='section','Unknown protocol unit')
    return section_spec()


def reports_check(plan,projection,checks):
    contract.validate_spec(plan['section_spec'])
    need(plan['schema']=='prospective-ownership-plan-v1' and plan['definition_only'] is True and
         plan['execution_ready'] is False and plan['training_authorized'] is False and
         plan['data_admission'] is False and plan['models_fixed'] is True and plan['policy_fixed'] is True and
         plan['candidate_quarantined'] is True and plan['old_arena_hold_preserved'] is True and
         plan['confidence_intervals_enabled'] is False and plan['counters']==ZERO,
         'Protocol plan falsely authorizes execution or changes historical scope')
    need(checks['passed'] is True and checks['dispatch_leaf_cases']==160 and checks['graph_cases']==6 and
         checks['live_suffix_worker_tested'] is False and checks['production_admission'] is False,
         'Protocol fixture coverage or live scope differs')
    need(projection['fresh_domain_components'] is None and projection['production_admission'] is False,
         'Flattened metadata cannot certify future admission')


def completed(path=DEFAULT):
    path=path_of(path); catalog=read(path/'catalog.json'); identity=read(path/'identity.json')
    need(catalog['schema']==SCHEMA+'-complete' and catalog['passed'] is True and
         catalog['identity_sha256']==sha(path/'identity.json') and catalog['counters']==ZERO and
         catalog['training_authorized'] is False and catalog['execution_ready'] is False,
         'Protocol catalog identity or scope differs')
    validate_identity(identity); common.current(identity['inputs'])
    results=prefix(path,identity)
    need(len(results)==catalog['committed_units']==len(identity['tasks'])==195 and
         catalog['commits']==common.pins(sorted((path/'commits').glob('*.json'))), 'Protocol completion prefix differs')
    reports={}
    for name in ('plan','projection','contract-checks'):
        need(sha(path/(name+'.json'))==catalog['reports'][name], 'Protocol report changed')
        reports[name]=read(path/(name+'.json'))
    reports_check(reports['plan'],reports['projection'],reports['contract-checks'])
    need('torch' not in sys.modules,'Protocol reader imported Torch')
    return dict(catalog=catalog,reports=reports)


def pause(path,point,requested,marker,index):
    if point!=requested:return
    need(path!=DEFAULT.resolve() and marker is not None,'Pause hooks are fixture only')
    marker=Path(marker).resolve();need(marker.is_relative_to(FOLDER/'fixtures'),'Pause marker namespace differs')
    atomic(marker,dict(point=point,index=index,pid=os.getpid()))
    while True:time.sleep(.1)


def run(path=DEFAULT,pause_at=None,pause_marker=None,pause_index=1):
    path=path_of(path)
    if (path/'catalog.json').exists():return completed(path)
    need(pause_at in (None,'before_unit_commit','after_unit_commit','before_catalog'),'Unknown protocol pause hook')
    need(pause_at is None or path!=DEFAULT.resolve(),'Production pause forbidden')
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return completed(path)
        identity=prepare(path);validate_identity(identity);results=prefix(path,identity)
        source_pins={k:identity['inputs'][k] for k in SOURCES}
        for entry in identity['tasks'][len(results):]:
            common.current(source_pins);body=compute(entry);common.current(source_pins)
            index=entry['index']
            import hashlib
            body_sha=hashlib.sha256((config.canonical(body)+'\n').encode()).hexdigest()
            relative=f'units/{index:06d}-{body_sha}.json';atomic(path/relative,body)
            if index==pause_index:pause(path,'before_unit_commit',pause_at,pause_marker,index)
            record=dict(index=index,identity_sha256=sha(path/'identity.json'),task_sha256=config.digest(entry),
                result=relative,result_sha256=body_sha,
                previous_commit_sha256=sha(path/'commits'/f'{index-1:06d}.json') if index else None)
            atomic(path/'commits'/f'{index:06d}.json',record);results.append(body)
            if index==pause_index:pause(path,'after_unit_commit',pause_at,pause_marker,index)
            print(json.dumps(dict(event='protocol_unit_committed',index=index,total=195,kind=entry['kind'])),flush=True)
        projection=projection_module().summarize(results[1:-1])
        plan=dict(schema='prospective-ownership-plan-v1',definition_only=True,execution_ready=False,
            training_authorized=False,data_admission=False,models_fixed=True,policy_fixed=True,
            candidate_quarantined=True,old_arena_hold_preserved=True,confidence_intervals_enabled=False,
            counters=ZERO,section_spec=results[-1],pending_gates=results[-1]['design']['pending_gates'],
            current_projection_is_not_new_reservations=True,
            recovery_scope='protocol publication and scratch contracts; live handoff recovery pending')
        reports_check(plan,projection,results[0]); reports={}
        for name,body in (('plan',plan),('projection',projection),('contract-checks',results[0])):
            atomic(path/(name+'.json'),body);reports[name]=sha(path/(name+'.json'))
        common.current(identity['inputs'])
        catalog=dict(schema=SCHEMA+'-complete',passed=True,identity_sha256=sha(path/'identity.json'),
            committed_units=len(results),commits=common.pins(sorted((path/'commits').glob('*.json'))),
            reports=reports,counters=ZERO,training_authorized=False,execution_ready=False)
        pause(path,'before_catalog',pause_at,pause_marker,len(results));atomic(path/'catalog.json',catalog)
    return completed(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('baseline','run','verify'))
    parser.add_argument('--path',type=Path,default=DEFAULT)
    parser.add_argument('--pause-at',choices=('before_unit_commit','after_unit_commit','before_catalog'))
    parser.add_argument('--pause-marker',type=Path);parser.add_argument('--pause-index',type=int,default=1)
    args=parser.parse_args()
    if args.operation=='baseline': result=dict(passed=True,prior_inputs=baseline()['inputs_count'])
    else:
        body=completed(args.path) if args.operation=='verify' else run(args.path,args.pause_at,args.pause_marker,args.pause_index)
        result=dict(passed=True,units=body['catalog']['committed_units'],execution_ready=False,counters=ZERO)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
