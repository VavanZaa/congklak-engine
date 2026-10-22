"""Recoverable fixed-model observations and independently certified cached reports."""
from contextlib import closing
import ast
import math
import os
from pathlib import Path
import sqlite3
import sys
from . import config, phase6_store, neurodiag_common as common, neurodiag_data as data

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

ROOT=common.ROOT
DEFAULT=ROOT/'data/neurodiag-v1'
PANEL,INVENTORY,INPUTS=data.PANEL,data.INVENTORY,data.INPUTS
PLAN=ROOT/'build/neurodiag-plan.json'
IMPLEMENTATION=ROOT/'build/neurodiag-implementation.json'
SOURCE_FREEZE=ROOT/'build/neurodiag-source-freeze.json'
OUTPUT=ROOT/'build/neurodiag-result.json'
WITNESS=ROOT/'build/neurodiag-audit/observation-witness.json'
PRODUCTION_AUDIT=ROOT/'build/neurodiag-audit/production.json'
SCHEMA='neurodiag-frozen-model-cache-v1'
SOURCES=('congklak/neurodiag.py','congklak/neurodiag_data.py','congklak/neurodiag_metrics.py',
    'congklak/neurodiag_common.py','tools/diagnose-neural.py','tests/neurodiag_data_author.py',
    'tests/neurodiag_metrics_author.py','docs/NEURODIAG_SCOPE.md','docs/NEURODIAG_DATA.md',
    'docs/NEURODIAG_METRICS.md','congklak/config.py','congklak/phase6_store.py',
    'congklak/divergence_study_common.py','congklak/features_relay.py','congklak/features_v3.py',
    'congklak/neural_relay.py','congklak/native.py','congklak/pilot.py','tests/reference.py',
    'build/neurodiag-metrics-implementation.json',
    'build/neurodiag-prefreeze/executed_schema_probe.py',
    'build/neurodiag-prefreeze/schema_probe_stdout.txt',
    'build/neurodiag-prefreeze/schema_probe_exposure.json',
    'build/neurodiag-prefreeze/inventory_probe.py')
CONTRACT=dict(batch_size=16,fixture_batches=3,primary_roots=256,primary_cell_quota=32,
    paired_horizon_roots=28,MC_records=48,MC_starts=24,source_records=332,target_records=360,
    precision='Fixed FP32 export weights; FP64 features, scalar-compatible model arithmetic',
    objective='Separate finite H8/H10 estimates and individual fixed-policy terminal returns',
    role='Training-source in-sample diagnostics; no new assessment or optimizer reservation',
    summary='Cached numeric summary before complete marker; no model/gradient/SVD on certified reopen',
    witness='Independent production2 must pass and pin external observation witness; no DB mutation',
    training_performed=False,policy_fixed=True,models_fixed=True,weights_applied=False,
    new_teacher_searches=0,new_PUCT_searches=0,new_games=0,promotion=False)
AUDIT_METHODS=[
    'test_01_training_roles_all_origins_protection_closure_and_metadata_selection',
    'test_02_frozen_forward_derivatives_and_projection_contract',
    'test_03_live_batch_commit_recovery_and_completed_certification_gate']
PRODUCTION_METHODS=['test_04_actual_panel_cache_scalar_forward_and_targets',
    'test_05_analytic_gradients_svd_summaries_and_observation_witness']
WITNESS_CHECKS=('scalar_forward_conformance','analytic_derivatives','numeric_summary_and_SVD',
    'eligible_target_and_feature_sources','models_and_policy_fixed')
require,read,sha,publish=common.require,common.read,common.sha,common.publish
same=common.same


def metrics():
    from . import neurodiag_metrics
    return neurodiag_metrics


def finite(value):
    if isinstance(value,float): require(math.isfinite(value),'Nonfinite diagnostic cache')
    elif isinstance(value,dict):
        for item in value.values(): finite(item)
    elif isinstance(value,(list,tuple)):
        for item in value: finite(item)


def runtime_info():
    import numpy as np
    require('torch' not in sys.modules,'Torch is forbidden in fixed CPU diagnostics')
    return dict(python=sys.version,numpy=np.__version__,float64_epsilon=float(np.finfo(np.float64).eps),
        threads={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})


def prepare(): return data.prepare()


def plan_body():
    panel=read(PANEL)
    require(panel['source_records']==332 and panel['target_records']==360,'Declared panel support changed')
    return dict(schema=SCHEMA,contract=CONTRACT,baseline_sha256=sha(common.BASELINE),
        inventory_sha256=sha(INVENTORY),panel_sha256=sha(PANEL),source_inputs=panel['source_inputs'],
        model_files=common.pins([common.CONTROL,common.CANDIDATE]),rules_digest=config.RULES_DIGEST,
        runtime=runtime_info())


def plan():
    prepare();return publish(PLAN,plan_body())


def implementation_body():
    return dict(schema=SCHEMA+'-implementation',plan_sha256=sha(PLAN),sources=common.pins(SOURCES))


def freeze():
    plan();body=implementation_body();publish(IMPLEMENTATION,body);return publish(SOURCE_FREEZE,body)


def admit():
    require('torch' not in sys.modules,'Torch is forbidden in diagnostics')
    require(same(read(PLAN),plan_body()),'Frozen plan/panel/model/runtime differs')
    implementation=implementation_body()
    require(same(read(IMPLEMENTATION),implementation) and same(read(SOURCE_FREEZE),implementation),'Frozen implementation differs')
    require(all(common.baseline()['inputs'].get(k)==v for k,v in read(PLAN)['source_inputs'].items()),'Prior source ownership differs')
    common.current(read(PLAN)['source_inputs'])
    return read(PANEL)


def prepare_inputs():
    admit();pack=data.resolve_inputs();finite(pack);publish(INPUTS,pack);data.validate_inputs(pack);return pack


def owner_gate():
    proof=read(ROOT/'build/neurodiag-audit/preflight.json')
    require(proof['schema']=='neurodiag-independent-group-v1' and proof['group']=='preflight' and proof['passed'] is True,
        'Independent preflight is incomplete')
    require(type(proof['tests_run']) is int and proof['tests_run']==3 and all(type(proof[k]) is int and proof[k]==0 for k in ('errors','failures','skips')) and proof['covered_methods']==AUDIT_METHODS,'Preflight coverage differs')
    require(proof['covered_checks']==['__main__.NeurodiagIndependentAudit.'+name for name in AUDIT_METHODS],'Preflight class identities differ')
    prior=common.baseline()['inputs']
    require(all(proof['inputs'].get(k)==v for k,v in prior.items()),'Preflight omits protected prior inputs')
    needed=set(SOURCES)|{p.relative_to(ROOT).as_posix() for p in (common.BASELINE,INVENTORY,PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE,INPUTS,common.CONTROL,common.CANDIDATE)}
    needed.update(('tests/neurodiag_independent_audit.py','tools/neurodiag-audit.py','build/neurodiag-audit-source-freeze.json',
        'build/neurodiag-data-author.json','build/neurodiag-metrics-author.json'))
    require(needed<=set(proof['inputs']),'Preflight omits current source/inputs/audit bindings');common.current(proof['inputs'])
    owners={}
    for filename,source in (('neurodiag-data-author.json','tests/neurodiag_data_author.py'),('neurodiag-metrics-author.json','tests/neurodiag_metrics_author.py')):
        body=read(ROOT/'build'/filename)
        require(body['passed'] is True and type(body['tests_run']) is int and body['tests_run']==4 and all(type(body[k]) is int and body[k]==0 for k in ('errors','failures','skips')),'Owner suite incomplete')
        ids=body.get('covered_methods',body.get('covered_checks'))
        tree=ast.parse((ROOT/source).read_text(encoding='utf-8-sig'))
        available={node.name+'.'+method.name for node in tree.body if isinstance(node,ast.ClassDef)
            for method in node.body if isinstance(method,ast.FunctionDef) and method.name.startswith('test_')}
        require(len(ids)==len(set(ids))==4 and {'.'.join(name.split('.')[-2:]) for name in ids}==available,'Owner executed coverage differs')
        require(body['inputs'].get(source)==sha(ROOT/source),'Owner source is unpinned');common.current(body['inputs'])
        owners[filename]=sha(ROOT/'build'/filename)
    return dict(preflight_sha256=sha(ROOT/'build/neurodiag-audit/preflight.json'),owners=owners)


def settings_for(pack,fixture=False):
    require(type(fixture) is bool,'Fixture flag must be Boolean')
    ids=sorted(pack['units'])
    if fixture:
        require(len(ids)>=48,'Fixture requires first48 frozen units');ids=ids[:48]
    return dict(fixture=fixture,unit_ids=ids,batch_size=16,batches=(len(ids)+15)//16)


def identity_for(pack,settings):
    return dict(schema=SCHEMA,settings=settings,rules_digest=config.RULES_DIGEST,
        artifacts=common.pins([common.BASELINE,PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE,INPUTS]),
        model_files=common.pins([common.CONTROL,common.CANDIDATE]),runtime=runtime_info(),
        production_gate=None if settings['fixture'] else owner_gate())


def path_check(path,fixture):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT/'build') if fixture else path==DEFAULT.resolve(),'Diagnostic run path differs')
    return path


def table(db,name,key):
    out={};hashes={}
    import json
    for identity,text,digest in db.execute('SELECT '+key+',body,hash FROM '+name+' ORDER BY '+key):
        body=json.loads(text);require(config.digest(body)==digest,'Corrupt '+name+' row');finite(body)
        out[identity]=body;hashes[str(identity)]=digest
    return out,hashes


def core(db,pack,identity,complete=False):
    require(db.execute('PRAGMA user_version').fetchone()[0]==6 and db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Invalid diagnostic SQLite journal')
    names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    require(names=={'meta','batches','observations'},'Unexpected journal tables')
    meta,mh=table(db,'meta','key');batches,bh=table(db,'batches','idx');obs,oh=table(db,'observations','unit')
    require(set(meta)>={'identity'} and set(meta)<={'identity','summary','complete'} and same(meta['identity'],identity),'Journal metadata/identity differs')
    require(('summary' in meta)==('complete' in meta),'Partial summary/complete transaction')
    settings=identity['settings'];ids=settings['unit_ids'];require(list(batches)==list(range(len(batches))) and len(batches)<=settings['batches'],'Batch prefix differs')
    expected=ids[:min(len(batches)*16,len(ids))]
    require(set(obs)==set(expected),'Observation prefix differs')
    full=[]
    for idx,body in batches.items():
        unit_ids=ids[idx*16:(idx+1)*16]
        require(set(body)=={'idx','unit_ids','observation_hashes','envelope','batch_sha256'} and type(body['idx']) is int and body['idx']==idx and body['unit_ids']==unit_ids and body['observation_hashes']=={u:oh[u] for u in unit_ids},'Batch binding differs')
        envelope=dict(body['envelope'],observations={u:obs[u] for u in unit_ids})
        require(config.digest(envelope)==body['batch_sha256'],'Batch envelope differs')
        metrics().validate_cached(envelope,units={u:pack['units'][u] for u in unit_ids})
        if full:require(same(envelope['models'],full[0]['models']) and same(envelope['runtime'],full[0]['runtime']),'Committed batch model/runtime identities differ')
        full.append(envelope)
    if 'complete' in meta:
        require('summary' in meta and len(batches)==settings['batches'] and len(obs)==len(ids),'Premature complete marker')
        expected_manifest=manifest_for(identity,batches,obs,meta['summary'])
        require(same(meta['complete'],expected_manifest),'Complete marker differs')
    elif complete: raise ValueError('Incomplete diagnostic cache')
    return dict(identity=identity,batches=full,batch_rows=batches,observations=obs,summary=meta.get('summary'),manifest=meta.get('complete'),
        row_hashes=dict(meta=mh,batches=bh,observations=oh))


def manifest_for(identity,batches,observations,summary):
    return dict(schema=SCHEMA+'-complete',status='complete_uncertified',identity=identity,
        batch_count=len(batches),unit_count=len(observations),target_records=sum(len(v['targets']) for v in observations.values()),
        batch_sha256=config.digest(batches),observations_sha256=config.digest(observations),summary_sha256=config.digest(summary),
        training_performed=False,policy_fixed=True,models_fixed=True,weights_applied=False,new_games=0,new_teacher_searches=0,new_PUCT_searches=0,promotion=False)


def stopped(cancel): return bool(cancel and (cancel.is_set() if hasattr(cancel,'is_set') else cancel()))


def run(path=DEFAULT,fixture=False,on_event=None,cancel=None):
    admit();pack=data.load_inputs();settings=settings_for(pack,fixture);path=path_check(path,fixture);identity=identity_for(pack,settings)
    event=lambda name,**values:on_event(dict(event=name,**values)) if on_event else None
    work=phase6_store.Work(path,identity);fresh=False
    try:
        work.db.execute('CREATE TABLE IF NOT EXISTS batches(idx INTEGER PRIMARY KEY,body TEXT,hash TEXT)')
        work.db.execute('CREATE TABLE IF NOT EXISTS observations(unit TEXT PRIMARY KEY,body TEXT,hash TEXT)');work.db.commit()
        state=core(work.db,pack,identity)
        if state['manifest'] is not None:
            certify(path,state);event('neurodiag_already_complete',completed=len(state['batch_rows']),units=len(state['observations']));return state['manifest']
        fresh=True;models=None
        for idx in range(len(state['batch_rows']),settings['batches']):
            if stopped(cancel):return dict(status='interrupted',batch_count=idx,unit_count=len(state['observations']))
            unit_ids=settings['unit_ids'][idx*16:(idx+1)*16]
            if models is None:models=metrics().load_models()
            envelope=metrics().observe({u:pack['units'][u] for u in unit_ids},models)
            metrics().validate_cached(envelope,units={u:pack['units'][u] for u in unit_ids});finite(envelope)
            if state['batches']:
                require(same(envelope['models'],state['batches'][0]['models']) and same(envelope['runtime'],state['batches'][0]['runtime']),'New batch model/numerical runtime differs from committed prefix')
            observations=envelope['observations'];require(set(observations)==set(unit_ids),'Forward observation units differ')
            hashes={u:config.digest(observations[u]) for u in unit_ids}
            body=dict(idx=idx,unit_ids=unit_ids,observation_hashes=hashes,envelope={k:v for k,v in envelope.items() if k!='observations'},batch_sha256=config.digest(envelope))
            event('neurodiag_precommit',completed=idx,target=settings['batches'],unit_count=len(state['observations']),batch_index=idx)
            work.admit(len(config.canonical(envelope))*3+2*1024**2)
            with work.db:
                for u in unit_ids:work.db.execute('INSERT INTO observations VALUES(?,?,?)',(u,config.canonical(observations[u]),hashes[u]))
                work.db.execute('INSERT INTO batches VALUES(?,?,?)',(idx,config.canonical(body),config.digest(body)))
            state['batch_rows'][idx]=body;state['batches'].append(envelope);state['observations'].update(observations)
            event('neurodiag_committed',completed=idx+1,target=settings['batches'],unit_count=len(state['observations']),batch_index=idx)
        summary=metrics().summarize_cached(state['batches']);finite(summary)
        manifest=manifest_for(identity,state['batch_rows'],state['observations'],summary)
        with work.db:work.put('summary',summary);work.put('complete',manifest)
        state=core(work.db,pack,identity,True)
    finally:work.close()
    require(fresh,'Unexpected cache publication path')
    publish(path/'summary.json',state['summary']);publish(path/'manifest.json',state['manifest'])
    event('neurodiag_cache_complete',completed=len(state['batch_rows']),units=len(state['observations']),certified=False)
    return state['manifest']


def raw_complete(path=DEFAULT,require_exports=True):
    """Audit-only uncertified cached data; never constitutes model conformance."""
    admit();pack=data.load_inputs();path=Path(path).resolve()
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');meta,_=table(db,'meta','key');saved=meta['identity'];settings=saved['settings']
        path_check(path,settings['fixture']);require(same(settings,settings_for(pack,settings['fixture'])),'Saved settings differ')
        identity=identity_for(pack,settings);state=core(db,pack,identity,True)
    if require_exports:
        require(same(read(path/'manifest.json'),state['manifest']) and same(read(path/'summary.json'),state['summary']),'Required cache exports differ')
    return state


def repair_raw_exports(path=DEFAULT):
    """Repair interrupted, still-uncertified publication, without model work."""
    require(not WITNESS.exists() and not PRODUCTION_AUDIT.exists(),'Certified cache source exports cannot be repaired')
    state=raw_complete(path,False);path=Path(path)
    publish(path/'summary.json',state['summary']);publish(path/'manifest.json',state['manifest']);return state['manifest']


def certify(path,state):
    require(Path(path).resolve()==DEFAULT.resolve() and not state['identity']['settings']['fixture'],'Fixtures cannot be certified production')
    witness=read(WITNESS);group=read(PRODUCTION_AUDIT)
    require(witness['schema']=='neurodiag-independent-observation-witness-v1' and witness['passed'] is True and set(witness['checks'])==set(WITNESS_CHECKS) and all(witness['checks'][k] is True for k in WITNESS_CHECKS),'Independent observation witness is incomplete')
    require(group['schema']=='neurodiag-independent-group-v1' and group['passed'] is True and group['group']=='production' and type(group['tests_run']) is int and group['tests_run']==2 and all(type(group[k]) is int and group[k]==0 for k in ('errors','failures','skips')),'Independent production certification incomplete')
    require(group['covered_methods']==PRODUCTION_METHODS and group['covered_checks']==['__main__.NeurodiagIndependentAudit.'+name for name in PRODUCTION_METHODS],'Independent production coverage differs')
    needed=set(SOURCES)|{p.relative_to(ROOT).as_posix() for p in (common.BASELINE,INVENTORY,PANEL,PLAN,IMPLEMENTATION,SOURCE_FREEZE,INPUTS,
        DEFAULT/'work.sqlite3',DEFAULT/'manifest.json',DEFAULT/'summary.json',common.CONTROL,common.CANDIDATE)}
    needed.update(('tests/neurodiag_independent_audit.py','tools/neurodiag-audit.py','build/neurodiag-audit-source-freeze.json',
        'build/neurodiag-audit/preflight.json','build/neurodiag-data-author.json','build/neurodiag-metrics-author.json'))
    require(needed<=set(witness['inputs']) and needed<=set(group['inputs']),'Certificate omits required source/cache/audit bindings')
    require(group['inputs'].get(WITNESS.relative_to(ROOT).as_posix())==sha(WITNESS),'Production audit does not pin witness')
    require(witness['database_sha256']==sha(Path(path)/'work.sqlite3') and witness['manifest_sha256']==sha(Path(path)/'manifest.json') and witness['summary_sha256']==sha(Path(path)/'summary.json'),'Certified cache bytes differ')
    require(witness['identity_sha256']==config.digest(state['identity']) and witness['cached_summary_sha256']==config.digest(state['summary']) and same(witness['cache_row_sha256'],state['row_hashes']),'Certified cache identities/rows differ')
    common.current(witness['inputs']);common.current(group['inputs'])
    require(same(read(Path(path)/'manifest.json'),state['manifest']) and same(read(Path(path)/'summary.json'),state['summary']),'Required certified exports differ')
    return witness


def build():
    state=raw_complete();witness=certify(DEFAULT,state)
    return dict(schema=SCHEMA+'-result',status='complete',passed=True,certified=True,
        unit_count=state['manifest']['unit_count'],target_records=state['manifest']['target_records'],batch_count=state['manifest']['batch_count'],
        summary=state['summary'],witness_sha256=sha(WITNESS),production_audit_sha256=sha(PRODUCTION_AUDIT),
        training_performed=False,policy_fixed=True,models_fixed=True,weights_applied=False,promotion=False,
        new_teacher_searches=0,new_PUCT_searches=0,new_games=0,previous_allocation='hold',
        inputs=common.pins([common.BASELINE,PANEL,INVENTORY,PLAN,IMPLEMENTATION,SOURCE_FREEZE,INPUTS,
            DEFAULT/'work.sqlite3',DEFAULT/'manifest.json',DEFAULT/'summary.json',WITNESS,PRODUCTION_AUDIT,*SOURCES]))


def assess(output=OUTPUT):return publish(Path(output),build())


def verify(output=OUTPUT):
    body=read(Path(output));require(same(body,build()),'Published diagnostic report differs');return body
