"""Bounded full-root labels for protected diagnostics and fresh midgame roots."""
from collections import Counter
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from . import config,horizon_probe as helper,natural_study,deeper_teacher_v2 as teacher,phase6_store
from .native import ROOT,Native,flatten
from .neural_relay import Model

SCHEMA='targeted-midgame-full-root-label-pilot-v1'
DEFAULT=ROOT/'data/midgame-label-pilot-v1'
BASELINE=ROOT/'build/midgame-pilot-baseline.json'
REGISTRY=ROOT/'build/midgame-label-pilot-registry.json'
PLAN=ROOT/'build/midgame-label-pilot-plan.json'
IMPLEMENTATION=ROOT/'build/midgame-label-pilot-implementation.json'
OUTPUT=ROOT/'build/midgame-label-pilot-result.json'
MODELS={'control':'data/quality-train-v1/model.json','pilot':'data/value-pilot-v1/model.json'}
MODEL_SHAS={'control':'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558','pilot':'7d19e6685000406349cf1a7aa862f4a13665ff59de41145a8f29de10eb8c0f1d'}
SOURCES=('congklak/midgame_label_pilot.py','tools/midgame-label-pilot.py','tests/midgame_label_pilot_author.py','congklak/midgame_pilot_progress.py','tools/midgame-pilot-dashboard.html','tools/analyze-midgame-label-pilot.py','docs/MIDGAME_PILOT_SCOPE.md')
CONTRACT=dict(diagnostic_roots=9,uncovered_diagnostics=8,covered_reference_diagnostics=1,maximum_fresh_roots=50,
    h8=dict(depth=8,node_budget=2000000),h10=dict(depth=10,node_budget=5000000),h12=dict(depth=12,node_budget=10000000),
    h8_scope='All diagnostics and fresh roots',h10_scope='All diagnostics and fresh roots regardless of H8 rejection',h12_scope='All nine diagnostics regardless of earlier rejection; no fresh roots',
    keep_history=True,time_ms=0,tt_mb=16,use_pvs=True,memory_mb=512,maximum_new_production_nodes=503000000,
    automatic_expansion=False,H14_authorized=False,training_performed=False,sampling_applied=False,sealed_final_test_consumed=False)
TABLES=('old_h8_references','h8_attempts','h8_labels','h10_attempts','h10_labels','h12_attempts','h12_labels')
require,sha,read,publish,table,stopped=helper.require,helper.sha,helper.read,helper.publish,helper.table,helper.stopped
item_for,q_signature,validate_attempt,execute=helper.item_for,helper.q_signature,helper.validate_attempt,helper.execute


def collector():
    from . import midgame_data_pilot
    return midgame_data_pilot


def checked_baseline():
    body=read(BASELINE)
    require(body['schema']=='midgame-pilot-prior-immutability-v1' and body['passed'] is True and body['inputs_count']==len(body['inputs'])==2639 and body['prior_accepted_inputs']==2637,'Midgame baseline coverage differs')
    for name,digest in body['inputs'].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT) and sha(path)==digest,'Prior midgame input changed: '+name)
    require(all(sha(ROOT/path)==MODEL_SHAS[name] for name,path in MODELS.items()),'Fixed model changed')
    return body


def make_registry():
    checked_baseline();source=read(collector().REGISTRY);arena=read(ROOT/'build/pilot-loss-review-arena.json');dataset=read(ROOT/'build/value-pilot-rows.json')
    diagnostics=source['diagnostics'];require(isinstance(diagnostics,dict) and len(diagnostics)==9,'Diagnostic panel must contain nine literal roots')
    expected={(row['pair'],row['candidate_seat']):row for row in arena['seats'] if not row['first_divergence']['identical_action_trajectory']}
    require(len(expected)==9,'Prior diagnostic selection differs');seen=set();old_refs={}
    native=Native()
    for key,row in sorted(diagnostics.items()):
        p=row['state'];provenance=row['provenance'];pair=(provenance['pair'],provenance['candidate_seat']);previous=expected[pair];d=previous['first_divergence']
        normal=native.normalize(flatten(p),actor=p['controller'],total=98)
        require(pair not in seen and row['cohort']=='diagnostic' and row['state_key']==key==collector().raw_key(p) and p==d['state']==provenance['raw_state']==normal['position'] and provenance['swapped'] is False,'Diagnostic literal state/ownership changed')
        require(row['family']==previous['family'] and row['legal_mask']==sum(1<<pit for pit in normal['legal_actions']) and row['active']==sum(map(sum,p['pits'])) and
            provenance['control_chosen']==d['control']['summary']['chosen'] and provenance['pilot_chosen']==d['pilot']['summary']['chosen'] and provenance['existing_Q_comparison']==d['existing_Q_comparison'],'Diagnostic actions/reference provenance changed')
        seen.add(pair)
        references=d['existing_Q_comparison']['comparisons']
        if references:
            require(len(references)==1 and pair==(1,1),'Only covered ninth H8 reference is admitted');reference=references[0];origin=reference['source'];original=dataset['groups'][origin['group']][origin['row']]
            require(original['state']==p and config.digest(original)==origin['row_sha256'] and config.digest(original['label'])==reference['label_sha256'] and original['teacher_depth']==8,'Old H8 full-state reference differs')
            teacher.validate_label(original['label'],p,row['legal_mask'],8,native)
            old_refs[key]=dict(source=origin,original_row=original,label=original['label'],label_sha256=config.digest(original['label']))
    require(len(old_refs)==1 and seen==set(expected),'Covered/uncovered panel differs')
    body=dict(schema='midgame-label-diagnostic-registry-v1',diagnostics=dict(sorted(diagnostics.items())),old_h8_references=old_refs,
        collector_registry_sha256=sha(collector().REGISTRY),arena_sha256=sha(ROOT/'build/pilot-loss-review-arena.json'),baseline_sha256=sha(BASELINE),literal_state_orientation=True)
    body['sha256']=config.digest(body);publish(REGISTRY,body);return body


def plan():
    registry=make_registry();c=collector();c.admit()
    value=dict(schema=SCHEMA,contract=CONTRACT,rules_digest=config.RULES_DIGEST,model_files=MODELS,model_sha256=MODEL_SHAS,
        registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),collector_plan_sha256=sha(c.PLAN),collector_implementation_sha256=sha(c.IMPLEMENTATION),collector_registry_sha256=sha(c.REGISTRY),
        diagnostic_state_keys=sorted(registry['diagnostics']),fresh_source='Strict closed collector output, bound in runtime identity after collection; never inferred before selection',
        full_root_Q_only=True,deeper_is_not_game_truth=True,no_training_or_ingestion=True)
    publish(PLAN,value);return value


def freeze():
    plan();baseline=checked_baseline();c=collector()
    sources={name:sha(ROOT/name) for name in SOURCES}
    sources[Path(c.__file__).resolve().relative_to(ROOT).as_posix()]=sha(c.__file__)
    value=dict(schema='midgame-label-implementation-v1',plan_sha256=sha(PLAN),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),sources=sources,prior_inputs=baseline['inputs'])
    publish(IMPLEMENTATION,value);return value


def admit():
    baseline=checked_baseline();p=read(PLAN);implementation=read(IMPLEMENTATION);c=collector();c.admit()
    require(p['schema']==SCHEMA and p['contract']==CONTRACT and p['rules_digest']==config.RULES_DIGEST and p['model_files']==MODELS and p['model_sha256']==MODEL_SHAS and
        p['registry_sha256']==sha(REGISTRY) and p['baseline_sha256']==sha(BASELINE) and p['collector_plan_sha256']==sha(c.PLAN) and p['collector_implementation_sha256']==sha(c.IMPLEMENTATION) and p['collector_registry_sha256']==sha(c.REGISTRY),'Fixed label plan differs')
    expected={name:sha(ROOT/name) for name in SOURCES};expected[Path(c.__file__).resolve().relative_to(ROOT).as_posix()]=sha(c.__file__)
    require(set(implementation)=={'schema','plan_sha256','registry_sha256','baseline_sha256','sources','prior_inputs'} and implementation['schema']=='midgame-label-implementation-v1' and implementation['sources']==expected and implementation['prior_inputs']==baseline['inputs'] and implementation['plan_sha256']==sha(PLAN) and implementation['registry_sha256']==sha(REGISTRY) and implementation['baseline_sha256']==sha(BASELINE),'Label source freeze differs')
    return p,implementation


def settings_for(fixture=False,diagnostics=None,fresh_roots=None):
    value=json.loads(config.canonical(CONTRACT))
    if fixture:
        value['diagnostic_roots']=2 if diagnostics is None else config.integer(diagnostics,'fixture_diagnostics',2,9)
        value['maximum_fresh_roots']=1 if fresh_roots is None else config.integer(fresh_roots,'fixture_fresh_roots',1,8)
    else:require(diagnostics is None and fresh_roots is None,'Production counts cannot be overridden')
    return value


def source_rows(collection_path,settings,fixture):
    c=collector();collection_path=Path(collection_path).resolve()
    if fixture:require(collection_path.is_relative_to(ROOT/'build'),'Fixture collection must be under build')
    else:require(collection_path==c.DEFAULT.resolve(),'Unregistered collection input')
    fresh,manifest=c.load_rows(collection_path)
    require(manifest['identity']['fixture'] is fixture,'Collection fixture/production identity differs')
    registry=read(REGISTRY);require(registry['sha256']==config.digest({k:v for k,v in registry.items() if k!='sha256'}),'Label diagnostic registry checksum differs')
    covered=next(iter(registry['old_h8_references']));diagnostic_keys=[covered]+[key for key in sorted(registry['diagnostics']) if key!=covered]
    if fixture:diagnostic_keys=diagnostic_keys[:settings['diagnostic_roots']];fresh={key:fresh[key] for key in sorted(fresh)[:settings['maximum_fresh_roots']]}
    require(len(diagnostic_keys)==settings['diagnostic_roots'] and len(fresh)<=settings['maximum_fresh_roots'] and (not fixture or len(fresh)>=1),'Missing fixed diagnostics or fresh fixture source')
    rows={}
    for role,group in (('diagnostic',{key:registry['diagnostics'][key] for key in diagnostic_keys}),('fresh',fresh)):
        for key,root in sorted(group.items()):
            require(key not in rows and root['state_key']==key and root['cohort'] in (('diagnostic',) if role=='diagnostic' else ('training_candidate','validation')),'Root source/ownership collision')
            rows[key]={name:root[name] for name in ('state_key','state','family','cohort','stratum','band','active','legal_mask')}
            rows[key].update(study_role=role,root=root,source_root_sha256=config.digest(root))
    lineage=dict(path=collection_path.relative_to(ROOT).as_posix(),journal_sha256=sha(collection_path/'work.sqlite3'),manifest_sha256=sha(collection_path/'manifest.json'),identity_sha256=config.digest(manifest['identity']),roots_sha256=manifest['roots_sha256'])
    return dict(sorted(rows.items())),{key:registry['old_h8_references'][key] for key in diagnostic_keys if key in registry['old_h8_references']},lineage


def identity_for(collection_path,settings,fixture):
    rows,references,lineage=source_rows(collection_path,settings,fixture);native=Native()
    identity=dict(schema=SCHEMA,settings=settings,fixture=fixture,rules_digest=config.RULES_DIGEST,model_file_sha256={name:sha(ROOT/path) for name,path in MODELS.items()},model_weights_sha256={name:Model(ROOT/path).id for name,path in MODELS.items()},
        source_sha256=sha(__file__),tool_sha256=sha(ROOT/'tools/midgame-label-pilot.py'),registry_sha256=sha(REGISTRY),baseline_sha256=sha(BASELINE),native_build_sha256=native.build_id,
        plan_sha256=None if fixture else sha(PLAN),implementation_sha256=None if fixture else sha(IMPLEMENTATION),collection=lineage,rows_sha256=config.digest(rows),old_references_sha256=config.digest(references),reserve_bytes=2*1024**3,quota_bytes=50*1024**3)
    require(identity['model_file_sha256']==MODEL_SHAS and 'torch' not in sys.modules,'Fixed CPU-only models changed')
    return identity,rows,references,native


def eligible(rows,stage):return [key for key,row in rows.items() if stage!='h12' or row['study_role']=='diagnostic']


def validate_tables(rows,stores,settings,references,native,closed=False):
    require(stores['old_h8_references']==references,'Old H8 reference was altered')
    signatures={key:{} for key in rows}
    for key,reference in references.items():
        teacher.validate_label(reference['label'],rows[key]['state'],rows[key]['legal_mask'],8,native)
        signatures[key][8]={name:reference['label'][name] for name in ('target_player0','best_moves','action_scores')}
    for stage in ('h8','h10','h12'):
        targets=eligible(rows,stage);attempts=stores[stage+'_attempts'];labels=stores[stage+'_labels']
        require(list(attempts)==targets[:len(attempts)] and set(labels)<=set(attempts) and (not closed or len(attempts)==len(targets)),'Requested stage prefix/ownership differs')
        for key,attempt in attempts.items():
            validate_attempt(rows[key],attempt,labels.get(key),stage,settings,native)
            for report in attempt['history']:
                depth=report['choice_depth'];signature=q_signature(item_for(report,rows[key]))
                require(depth not in signatures[key] or signatures[key][depth]==signature,'Overlapping same-depth old/new Q/best differs')
                signatures[key][depth]=signature
    require(not stores['h10_attempts'] or len(stores['h8_attempts'])==len(rows),'H10 before all H8 attempts')
    require(not stores['h12_attempts'] or len(stores['h10_attempts'])==len(rows),'H12 before all H10 attempts')


def counts(rows,stores):
    stages={}
    for stage in ('h8','h10','h12'):
        targets=eligible(rows,stage);attempts=stores[stage+'_attempts'];labels=stores[stage+'_labels']
        stages[stage]=dict(targets=len(targets),attempts=len(attempts),labels=len(labels),rejected=sum(row['status']=='budget_rejected' for row in attempts.values()),pending=len(targets)-len(attempts),
            by_role={role:dict(targets=sum(rows[key]['study_role']==role for key in targets),labels=sum(rows[key]['study_role']==role for key in labels),rejected=sum(rows[key]['study_role']==role and attempt['status']=='budget_rejected' for key,attempt in attempts.items())) for role in ('fresh','diagnostic')})
    return dict(roots=len(rows),fresh_roots=sum(row['study_role']=='fresh' for row in rows.values()),diagnostic_roots=sum(row['study_role']=='diagnostic' for row in rows.values()),old_h8_reference_labels=len(stores['old_h8_references']),
        cohort_counts=dict(Counter(row['cohort'] for row in rows.values())),stages=stages,completed_H8_H10_pairs=len(set(stores['h8_labels'])&set(stores['h10_labels'])),diagnostic_complete_H8_H10_H12=sum(rows[key]['study_role']=='diagnostic' for key in set(stores['h8_labels'])&set(stores['h10_labels'])&set(stores['h12_labels'])),
        auxiliary_completed_reports=sum(len(attempt['history']) for stage in ('h8','h10','h12') for attempt in stores[stage+'_attempts'].values()),total_new_nodes=sum(attempt['work']['nodes'] for stage in ('h8','h10','h12') for attempt in stores[stage+'_attempts'].values()))


def validate_complete_marker(manifest,identity,rows,stores):
    expected=counts(rows,stores)
    fields={'schema','identity','status',*expected,'table_sha256','training_performed','data_ingested','sampling_applied','torch_imported','sealed_final_test_consumed','elapsed_seconds_initial_run'}
    require(set(manifest)==fields and manifest['schema']==SCHEMA and manifest['identity']==identity and manifest['status']=='complete' and all(manifest[name]==value for name,value in expected.items()),'Complete label marker fields/counters/identity differ')
    require(all(not stage['pending'] for stage in expected['stages'].values()) and expected['diagnostic_roots']==identity['settings']['diagnostic_roots'] and expected['fresh_roots']<=identity['settings']['maximum_fresh_roots'],'Complete marker has pending targets')
    require(expected['total_new_nodes']<=503000000 and manifest['table_sha256']=={'roots':config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},'Complete table hashes or physical work differ')
    require(all(manifest[name] is False for name in ('training_performed','data_ingested','sampling_applied','torch_imported','sealed_final_test_consumed')) and type(manifest['elapsed_seconds_initial_run']) in (int,float) and math.isfinite(manifest['elapsed_seconds_initial_run']) and manifest['elapsed_seconds_initial_run']>0,'Complete label scope/clock differs')


def run(path=DEFAULT,stage='all',fixture=False,collection_path=None,diagnostics=None,fresh_roots=None,on_event=None,cancel=None):
    require(stage in ('h8','h10','h12','all'),'Unknown label stage');path=Path(path).resolve();settings=settings_for(fixture,diagnostics,fresh_roots)
    if fixture:require(path.is_relative_to(ROOT/'build'),'Fixture label path must be under build')
    else:require(path==DEFAULT.resolve(),'Unregistered production label path');admit()
    collection_path=collector().DEFAULT if collection_path is None else Path(collection_path)
    identity,expected,references,native=identity_for(collection_path,settings,fixture);work=phase6_store.Work(path,identity);started=time.monotonic()
    def event(name,**fields):
        if on_event:on_event(dict(event=name,**fields))
    try:
        work.db.executescript('CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT,hash TEXT);'+''.join('CREATE TABLE IF NOT EXISTS '+name+'(state TEXT PRIMARY KEY,body TEXT,hash TEXT);' for name in TABLES))
        rows=table(work.db,'roots');stores={name:table(work.db,name) for name in TABLES}
        if not rows:
            require(not any(stores.values()),'Orphan label tables')
            with work.db:
                for key,row in expected.items():work.db.execute('INSERT INTO roots VALUES(?,?,?)',(key,config.canonical(row),config.digest(row)))
                for key,reference in references.items():work.db.execute('INSERT INTO old_h8_references VALUES(?,?,?)',(key,config.canonical(reference),config.digest(reference)))
            rows=expected;stores['old_h8_references']=references
        require(rows==expected,'Committed selected roots or ownership differ');validate_tables(rows,stores,settings,references,native)
        complete=work.get('complete')
        if complete is None:
            for target in ('h8','h10','h12'):
                if stage!='all' and int(target[1:])>int(stage[1:]):break
                targets=eligible(rows,target);attempts=stores[target+'_attempts'];labels=stores[target+'_labels']
                for key in targets[len(attempts):]:
                    if stopped(cancel):break
                    item,attempt=execute(rows[key],target,settings,native,cancel)
                    if attempt is None:break
                    proposed={name:dict(value) for name,value in stores.items()};proposed[target+'_attempts'][key]=attempt
                    if item is not None:proposed[target+'_labels'][key]=item
                    validate_tables(rows,proposed,settings,references,native)
                    event('midgame_label_'+target+'_precommit',state_key=key,completed=len(attempts),target=len(targets),labels=len(labels),completed_depth=attempt['completed_depth'],study_role=rows[key]['study_role'])
                    work.admit(2*1024**2)
                    with work.db:
                        work.db.execute('INSERT INTO '+target+'_attempts VALUES(?,?,?)',(key,config.canonical(attempt),config.digest(attempt)))
                        if item is not None:work.db.execute('INSERT INTO '+target+'_labels VALUES(?,?,?)',(key,config.canonical(item),config.digest(item)))
                    attempts[key]=attempt
                    if item is not None:labels[key]=item
                    event('midgame_label_'+target+'_committed',state_key=key,completed=len(attempts),target=len(targets),labels=len(labels),rejected=sum(attempt['status']=='budget_rejected' for attempt in attempts.values()),completed_depth=attempt['completed_depth'],study_role=rows[key]['study_role'])
                if len(attempts)!=len(targets):break
        if complete is None and all(len(stores[target+'_attempts'])==len(eligible(rows,target)) for target in ('h8','h10','h12')):
            complete=dict(schema=SCHEMA,identity=identity,status='complete',**counts(rows,stores),table_sha256={'roots':config.digest(rows),**{name:config.digest(value) for name,value in stores.items()}},training_performed=False,data_ingested=False,sampling_applied=False,torch_imported=False,sealed_final_test_consumed=False,elapsed_seconds_initial_run=time.monotonic()-started)
            validate_complete_marker(complete,identity,rows,stores)
            with work.db:work.put('complete',complete)
        if complete is not None:
            validate_complete_marker(complete,identity,rows,stores);publish(path/'manifest.json',complete)
            if not fixture:publish(OUTPUT,complete)
            load_rows(path);event('midgame_label_complete',**counts(rows,stores));return complete
        return dict(status='interrupted',**counts(rows,stores))
    finally:work.close()


def load_rows(path=DEFAULT):
    path=Path(path).resolve()
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN')
        def meta(key):
            row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone();require(row is not None,'Missing committed '+key);value=json.loads(row[0]);require(config.digest(value)==row[1],'Corrupt '+key);return value
        manifest,identity=meta('complete'),meta('identity');fixture=identity['fixture'];settings=identity['settings']
        require(type(fixture) is bool and settings==settings_for(fixture,settings['diagnostic_roots'] if fixture else None,settings['maximum_fresh_roots'] if fixture else None),'Closed fixture/settings differ')
        if fixture:require(path.is_relative_to(ROOT/'build'),'Fixture closed path differs')
        else:require(path==DEFAULT.resolve(),'Production closed path differs');admit()
        current,expected,references,native=identity_for(ROOT/identity['collection']['path'],settings,fixture);require(identity==current,'Closed source/model/collection identity differs')
        rows=table(db,'roots');stores={name:table(db,name) for name in TABLES};require(rows==expected,'Closed roots/ownership differ')
        validate_tables(rows,stores,settings,references,native,closed=True);validate_complete_marker(manifest,identity,rows,stores)
        require((path/'manifest.json').read_text(encoding='utf-8')==config.canonical(manifest)+'\n','Closed file/commit differs')
        return rows,stores,manifest
