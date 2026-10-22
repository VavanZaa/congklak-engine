"""Fixed-work PUCT on a frozen five-root panel, with durable cell transactions.

Requested budgets are independent fresh trees. Timings remain in the raw record;
only their explicit fields are excluded from deterministic semantic comparison.
No teacher, optimizer, Torch, arena, or adaptive budget is invoked here.
"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack, closing
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from . import config, divergence_study_common as common, quality_search, phase6_store
from .native import Native, ROOT, flatten
from .neural_relay import Model
from .ordinary_value_benchmark import verify_candidate, host_peak
from .puct_cpu_inference import CpuBatchEvaluator, COUNTERS, DLL
from .timing_v2 import PerformanceCore

DEFAULT=ROOT/'data/divergence-search-v1'
PLAN=ROOT/'build/divergence-search-plan.json'
IMPLEMENTATION=ROOT/'build/divergence-search-implementation.json'
OUTPUT=ROOT/'build/divergence-search-result.json'
RESOURCES=ROOT/'build/divergence-search-resources.json'
CLI=ROOT/'tools/probe-divergence-search.py'
SCHEMA='five-root-fixed-simulation-search-v1'
PROFILES=('control','candidate')
TABLES={'meta','roots','setup','attempts','cells'}
require=common.require


def _number(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
def _integer(value):return type(value) is int and value>=0
def _same(left,right):return config.canonical(left)==config.canonical(right)
def _hashrow(text,digest):
    body=json.loads(text);require(config.digest(body)==digest,'Corrupt divergence journal row');return body
def _ro(path):return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
def _meta(db,key):
    row=db.execute('SELECT body,hash FROM meta WHERE key=?',(key,)).fetchone()
    return _hashrow(*row) if row else None
def _cancelled(cancel):return bool(cancel() if callable(cancel) else cancel.is_set()) if cancel is not None else False


def expected_plan():
    panel=common.load_panel();old=common.read(ROOT/'build/ordinary-value-benchmark-plan.json')
    with closing(_ro(ROOT/'data/ordinary-value-benchmark-v1/work.sqlite3')) as db:
        old_setup=_hashrow(*db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone())
    old_backends={old_setup['warmup'][name]['identity']['backend'] for name in ('ordinary_control','ordinary_value')}
    require(len(old_backends)==1,'Prior models used different backends');expected_backend=next(iter(old_backends))
    diagnostics=common.read(ROOT/'build/value-pilot-rows.json')['groups']['diagnostic'][:2]
    fixture=[];native=Native()
    for index,item in enumerate(diagnostics):
        p=config.position(item['state']);normal=native.normalize(flatten(p),actor=p['controller'],total=98)
        require(not normal['terminal'] and normal['position']==p,'Fixture diagnostic must be normalized/nonterminal')
        fixture.append(dict(id='consumed-fixture-'+str(index),state=p,state_key=common.rawkey(p),legal_actions=normal['legal_actions'],legal_mask=sum(1<<a for a in normal['legal_actions']),actor=p['controller'],
            source=dict(path='build/value-pilot-rows.json',group='diagnostic',index=index,row_sha256=config.digest(item)),role='consumed_engineering_fixture_only'))
    require(len(fixture)==2 and len({r['state_key'] for r in fixture})==2,'Fixture state support differs')
    return dict(schema=SCHEMA+'-plan',panel_file_sha256=common.sha(common.PANEL),panel_sha256=panel['sha256'],baseline_sha256=common.sha(common.BASELINE),scope_sha256=common.sha(common.SCOPE),
        roots=panel['roots'],profiles=list(PROFILES),models=common.pins([common.CONTROL,common.CANDIDATE]),simulations=[128,256,512,1024],
        requested_cells=40,requested_completed_simulations=19200,c_puct=.25,time_ms=0,root_noise=None,rng=0,max_nodes=2048,memory_mb=8192,cache_capacity=0,max_batch=1,backend='auto',expected_backend=expected_backend,
        cpu_declared=old['cpu_declared'],cpu_metadata=common.read(ROOT/'build/ordinary-value-benchmark.json')['identity']['cpu'],search_protocol=quality_search.PROTOCOL,order='root order, ascending budget, control then candidate; fresh tree and cache for every cell',
        fixture=dict(roots=fixture,simulations=[8,16],variants=dict(normal=dict(max_nodes=2048),node_cap=dict(max_nodes=2)),production_witness=False),
        incomplete='Cancelled attempts are retained and retried only by explicit resume. Fixed node/memory caps finalize typed incomplete cells; no automatic budget expansion.',
        primary_witness='All40 production cells must complete their exact budget before teacher production; fixtures never satisfy the gate',training_authorized=False,default_promotion=False)


def prepare():
    return common.publish(PLAN,expected_plan())


def implementation_body():
    sources=('congklak/divergence_search.py','tools/probe-divergence-search.py','tests/divergence_search_author.py')
    science=(common.BASELINE,common.PANEL,common.SCOPE,PLAN,ROOT/'congklak/divergence_study_common.py')
    dependencies=('congklak/quality_search.py','congklak/puct_cpu_inference.py','congklak/phase6_store.py','congklak/timing_v2.py','congklak/ordinary_value_benchmark.py','congklak/config.py','congklak/native.py','congklak/neural_relay.py','congklak/features_relay.py','congklak/phase6_inference.py','build/puct-cpu-build.json','build/congklak_puct_cpu.dll')
    body=dict(schema=SCHEMA+'-implementation',sources=common.pins(sources),science_inputs=common.pins(science),dependencies=common.pins(dependencies),rules_digest=config.RULES_DIGEST)
    return {**body,'sha256':config.digest(body)}


def freeze():
    prepare();return common.publish(IMPLEMENTATION,implementation_body())


def admit(fixture=False,variant='normal',verify_prior=False):
    require(type(fixture) is bool and variant in ('normal','node_cap') and (fixture or variant=='normal'),'Unknown search configuration')
    require('torch' not in sys.modules,'Torch is forbidden in this search pilot')
    common.baseline(verify_all=verify_prior);plan=common.read(PLAN);require(_same(plan,expected_plan()),'Search plan changed');implementation=implementation_body();require(_same(common.read(IMPLEMENTATION),implementation),'Search source freeze changed')
    models={name:Model(path) for name,path in (('control',common.CONTROL),('candidate',common.CANDIDATE))};contract=verify_candidate(models['control'],models['candidate'])
    roots=plan['fixture']['roots'] if fixture else plan['roots'];budgets=plan['fixture']['simulations'] if fixture else plan['simulations'];max_nodes=plan['fixture']['variants'][variant]['max_nodes'] if fixture else plan['max_nodes']
    units=[dict(index=index,root_index=ri,state_key=root['state_key'],profile=name,simulations=budget,max_nodes=max_nodes) for index,(ri,root,budget,name) in enumerate((ri,root,budget,name) for ri,root in enumerate(roots) for budget in budgets for name in PROFILES)]
    identity=dict(schema=SCHEMA,fixture=fixture,variant=variant,plan_file_sha256=common.sha(PLAN),implementation_file_sha256=common.sha(IMPLEMENTATION),implementation_sha256=implementation['sha256'],panel_file_sha256=common.sha(common.PANEL),baseline_sha256=common.sha(common.BASELINE),
        model_files={name:models[name].file_sha256 for name in PROFILES},model_weights={name:models[name].id for name in PROFILES},policy_contract=contract,roots_sha256=config.digest(roots),units_sha256=config.digest(units),units=len(units),
        max_nodes=max_nodes,c_puct=.25,time_ms=0,root_noise=None,rng=0,cache_capacity=0,max_batch=1,memory_mb=8192,cpu_declared=plan['cpu_declared'],cpu_metadata=plan['cpu_metadata'],backend=plan['expected_backend'],rules_digest=config.RULES_DIGEST,training_authorized=False)
    return plan,roots,units,identity


def semantic_report(report):
    body=copy.deepcopy(report);body.pop('timing');body['work'].pop('inference_ms');return body


def _snapshot(evaluator):return evaluator.snapshot_counters()


def validate_attempt(body,seq,unit,root,identity,setup,native=None):
    fields={'schema','seq','cell_index','unit','root_sha256','status','report','report_sha256','semantic_report','semantic_sha256','evaluator_before','evaluator_after','counts','cpu_verified','evaluator_identity_sha256'}
    require(isinstance(body,dict) and set(body)==fields and body['schema']==SCHEMA+'-attempt' and type(body['seq']) is int and body['seq']==seq and type(body['cell_index']) is int and body['cell_index']==unit['index'] and _same(body['unit'],unit) and body['root_sha256']==config.digest(root),'Attempt identity/fields changed')
    report=body['report'];quality_search.validate_report(report,root['state'],unit['simulations'],max_nodes=unit['max_nodes'],c_puct=.25,native=native,allow_interrupted=True)
    require(report['memory_mb']==8192 and report['timing']['requested_time_ms']==0 and report['timing']['overshoot_ms']==0 and report['forest_size']==1,'Search bounds or clock changed')
    require(body['report_sha256']==config.digest(report) and _same(body['semantic_report'],semantic_report(report)) and body['semantic_sha256']==config.digest(body['semantic_report']),'Semantic/raw report digest changed')
    expected='complete' if report['stop_reason']=='simulations' and report['completed_simulations']==unit['simulations'] else 'cancelled' if report['stop_reason']=='cancelled' else 'budget_rejected'
    require(body['status']==expected and report['stop_reason'] in ('simulations','node_budget','cancelled','memory_budget'),'Incomplete work relabeled or undeclared time stop')
    if expected=='complete':require(report['root_expanded'] and report['usable_decision'] and sum(report['visits'])==unit['simulations'],'Incomplete full-budget cell')
    counts=body['counts'];before=body['evaluator_before'];after=body['evaluator_after'];keys=set(COUNTERS)|{'cache_current_entries','cache_capacity'}
    require(set(before)==set(after)==keys and all(_integer(v) for value in (before,after) for v in value.values()) and set(counts)==set(COUNTERS),'Counter snapshot schema changed')
    require(all(counts[name]==after[name]-before[name] and _integer(counts[name]) for name in COUNTERS),'Counter delta changed')
    require(all(value['cache_capacity']==value['cache_current_entries']==0 for value in (before,after)) and counts['cache_resets']==1 and counts['cache_hits']==counts['cache_evictions']==counts['terminal_returns']==0,'Uncached fresh-tree contract changed')
    require(counts['evaluations_requested']==counts['cache_misses'] and counts['predict_successes']<=counts['native_actual_evals']<=counts['native_calls']<=counts['cache_misses']<=counts['positions_submitted'],'Physical counter conservation changed')
    require(report['work']['neural_evaluations']==counts['native_actual_evals'] and report['work']['feature_simulations']==counts['feature_simulations'] and report['work']['feature_deposits']==counts['feature_deposits'],'Search work differs from physical inference')
    require(counts['feature_simulations']<=14*counts['native_calls'],'Feature calls exceed single-pass legal support')
    if expected!='cancelled':
        require(len({counts[name] for name in ('positions_submitted','evaluations_requested','native_actual_evals','predict_successes','native_calls','cache_misses')})==1,'Finished uncached call prefix incomplete')
        require(counts['feature_simulations']>=counts['native_actual_evals'] and counts['feature_deposits']>=counts['feature_simulations'],'Physical forwards lack actual feature work')
        require(report['work']['neural_calls']==counts['native_calls'],'Physical calls relabeled')
    require(_same(body['cpu_verified'],dict(group=0,logical=identity['cpu_declared']['logical'],mask=1<<identity['cpu_declared']['logical'])),'Search left declared CPU')
    require(body['evaluator_identity_sha256']==config.digest(setup['evaluators'][unit['profile']]),'Evaluator admission identity changed')
    return body


def _validate_setup(setup,identity):
    require(set(setup)=={'evaluators','cpu'} and set(setup['evaluators'])==set(PROFILES),'Missing evaluator startup identities')
    require(_same(setup['cpu'],identity['cpu_metadata']),'CPU topology or exact declaration changed')
    for name,value in setup['evaluators'].items():
        require(value['model_file_sha256']==identity['model_files'][name] and value['model_weights_sha256']==identity['model_weights'][name] and value['rules_digest']==config.RULES_DIGEST,'Model startup identity changed')
        require(value['requested_backend']=='auto' and value['backend']==identity['backend'] and _same(value['max_batch'],1) and _same(value['cache']['capacity'],0),'CPU adapter settings changed')
        require(value['dll_sha256']==common.sha(DLL) and value['build_record_sha256']==common.sha(ROOT/'build/puct-cpu-build.json') and value['source_sha256']==common.sha(ROOT/'congklak/puct_cpu_inference.py'),'CPU binary/source changed')
        common.current(value['dependencies']);common.current(value['build_inputs'])
    require(setup['evaluators']['control']['backend']==setup['evaluators']['candidate']['backend'],'Models use different numerical backends')


def validate_prefix(db,roots,units,identity):
    require({row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}==TABLES and db.execute('PRAGMA user_version').fetchone()[0]==6 and db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Unknown/damaged search journal')
    require(_same(_meta(db,'identity'),identity),'Search journal identity changed');require({row[0] for row in db.execute('SELECT key FROM meta')}<= {'identity','divergence_search_complete'},'Unknown journal metadata')
    saved={key:_hashrow(text,digest) for key,text,digest in db.execute('SELECT state,body,hash FROM roots')}
    require(_same(saved,{row['state_key']:row for row in roots}),'Frozen roots changed')
    startup=db.execute('SELECT id,body,hash FROM setup').fetchall();require(len(startup)<=1 and (not startup or startup[0][0]==0),'Startup identity has duplicates')
    setup=_hashrow(startup[0][1],startup[0][2]) if startup else None
    if setup:_validate_setup(setup,identity)
    native=Native();attempts=[];finalized=[]
    for seq,text,digest in db.execute('SELECT seq,body,hash FROM attempts ORDER BY seq'):
        body=_hashrow(text,digest);require(seq==len(attempts) and len(finalized)<len(units) and setup is not None,'Orphan/noncontiguous attempt')
        unit=units[len(finalized)];validate_attempt(body,seq,unit,roots[unit['root_index']],identity,setup,native);attempts.append(body)
        if body['status']!='cancelled':finalized.append(body)
    cells=[]
    for index,text,digest in db.execute('SELECT idx,body,hash FROM cells ORDER BY idx'):
        body=_hashrow(text,digest);require(index==len(cells) and index<len(finalized) and _same(body,finalized[index]),'Committed cells differ from attempt ledger');cells.append(body)
    require(_same(cells,finalized),'Missing finalized cell ledger');return setup,attempts,cells


def _paths(path):
    path=Path(path).resolve();official=path==DEFAULT.resolve()
    return path/'manifest.json',OUTPUT if official else path/'result.json',RESOURCES if official else path/'resources.json'


def _admit_path(path,fixture):
    path=Path(path).resolve()
    require(path.is_relative_to((ROOT/'build').resolve()) if fixture else path==DEFAULT.resolve(),'Fixtures require a build-only directory; production requires its declared directory')
    return path


def summarize(roots,cells,identity):
    complete=sum(row['status']=='complete' for row in cells);comparisons=[]
    for ri,root in enumerate(roots):
        budgets=sorted({row['unit']['simulations'] for row in cells if row['unit']['root_index']==ri})
        for budget in budgets:
            items={row['unit']['profile']:row for row in cells if row['unit']['root_index']==ri and row['unit']['simulations']==budget}
            if set(items)!=set(PROFILES):continue
            available=all(row['status']=='complete' for row in items.values());a=items['control']['report'];b=items['candidate']['report']
            if available:require(_same(a['root_priors'],b['root_priors']),'Fixed-policy priors differ at exact root')
            comparisons.append(dict(state_key=root['state_key'],root_id=root['id'],simulations=budget,eligible_models=2,completed_models=sum(row['status']=='complete' for row in items.values()),missing_models=sum(row['status']!='complete' for row in items.values()),complete_pair=available,
                control_chosen=a['chosen'] if items['control']['status']=='complete' else None,candidate_chosen=b['chosen'] if items['candidate']['status']=='complete' else None,
                same_chosen_action=a['chosen']==b['chosen'] if available else None,root_priors_identical=a['root_priors']==b['root_priors'] if available else None,
                control_action_sampled_mean_player0=a['action_mean_value_player0'][a['chosen']] if items['control']['status']=='complete' else None,candidate_action_sampled_mean_player0=b['action_mean_value_player0'][b['chosen']] if items['candidate']['status']=='complete' else None,
                interpretation='Sampled PUCT means and parent bootstrap are not teacher action Q. Unvisited means stay None; incomplete requested work is missing.'))
    return dict(schema=SCHEMA+'-result',identity=identity,status='complete' if complete==len(cells)==identity['units'] else 'budget_exhausted',scope_finished=complete==len(cells)==identity['units'],
        roots=roots,cells=cells,counts=dict(roots=len(roots),requested_cells=identity['units'],committed_cells=len(cells),complete_cells=complete,capped_cells=len(cells)-complete,
            requested_simulations=sum(row['unit']['simulations'] for row in cells),completed_simulations=sum(row['report']['completed_simulations'] for row in cells),
            transitions=sum(row['report']['work']['transitions'] for row in cells),tree_nodes=sum(row['report']['nodes'] for row in cells),physical_forwards=sum(row['counts']['native_actual_evals'] for row in cells),
            feature_simulations=sum(row['counts']['feature_simulations'] for row in cells),feature_deposits=sum(row['counts']['feature_deposits'] for row in cells)),
        comparisons=comparisons,semantic_cells_sha256=config.digest([dict(unit=row['unit'],status=row['status'],semantic_report=row['semantic_report'],counts=row['counts']) for row in cells]),
        limits=['All budgets and both unchanged-margin controls retained; consumed selected player1 diagnostics, not release evidence or a strength result.',
            'Raw timings are observations and excluded only from semantic equality. No equal-time or energy claim. Completed reopening runs no new search.',
            'An incomplete simulation budget is never a complete smaller-budget result. Models, policy, previous HOLD and H_v1 default remain unchanged.'],
        training_authorized=False,new_strength_games=0,torch_imported=False)


def _closed(path,fixture=False,variant='normal',require_exports=False):
    path=_admit_path(path,fixture);plan,roots,units,identity=admit(fixture,variant);require((path/'work.sqlite3').is_file(),'No search journal')
    with closing(_ro(path/'work.sqlite3')) as db:
        setup,attempts,cells=validate_prefix(db,roots,units,identity);marker=_meta(db,'divergence_search_complete')
        require(marker is not None and len(cells)==len(units),'Search is not closed');body=summarize(roots,cells,identity)
        fields={'schema','identity_sha256','cells_sha256','attempts_sha256','result_sha256','resource_path','resource_file_sha256'}
        require(set(marker)==fields and marker['schema']==SCHEMA+'-complete' and marker['identity_sha256']==config.digest(identity) and marker['cells_sha256']==config.digest(cells) and marker['attempts_sha256']==config.digest(attempts) and marker['result_sha256']==config.digest(body),'Invalid search completion marker')
    relative=Path(marker['resource_path']);resource_path=(path/relative).resolve();require(not relative.is_absolute() and resource_path.is_relative_to(path) and resource_path.is_file() and common.sha(resource_path)==marker['resource_file_sha256'],'Missing/corrupt finalizing resource observation')
    resource=common.read(resource_path);validate_resources(resource,identity,body)
    require(resource['final_attempts']==len(attempts),'Final resource attempt count differs')
    manifest,report,export=_paths(path)
    for file,expected in ((manifest,body),(report,body),(export,resource)):
        if file.exists():require(file.read_text(encoding='utf-8')==config.canonical(expected)+'\n','Existing search export differs')
        else:require(not require_exports,'Missing search export')
    return roots,cells,body,resource


def validate_resources(body,identity,result):
    fields={'schema','identity_sha256','result_sha256','elapsed_seconds_through_prepublication_validation','initial_committed_cells','final_committed_cells','initial_attempts','final_attempts','host_peak','torch_imported','energy_measured','scope'}
    require(set(body)==fields and body['schema']==SCHEMA+'-resources' and body['identity_sha256']==config.digest(identity) and body['result_sha256']==config.digest(result),'Resource identity differs')
    require(_number(body['elapsed_seconds_through_prepublication_validation']) and all(_integer(body[k]) for k in ('initial_committed_cells','final_committed_cells','initial_attempts','final_attempts')),'Invalid resource bounds')
    require(body['initial_committed_cells']<=body['final_committed_cells']==identity['units'] and body['initial_attempts']<=body['final_attempts'] and body['final_attempts']>=identity['units'],'Resource prefix differs')
    require(body['torch_imported'] is False and body['energy_measured'] is False and body['scope']=='This finalizing invocation, admission through validation before completion/public export; prior invocations, lost uncommitted kill work, fixtures and audits excluded','Resource scope differs')
    host=body['host_peak'];require(isinstance(host,dict) and set(host)=={'peak_working_set_bytes','peak_private_commit_bytes'} and all(type(v) is int and v>0 for v in host.values()),'Invalid host peak observation')


def emit(event):
    print(config.canonical(event),flush=True)
    if os.environ.get('DIVERGENCE_SEARCH_BLOCK_EVENT')!=event['event']:return
    after=int(os.environ.get('DIVERGENCE_SEARCH_BLOCK_AFTER','-1'))
    if after>=0 and event.get('completed')!=after:return
    marker=os.environ.get('DIVERGENCE_SEARCH_BLOCK_MARKER');require(marker,'Missing physical barrier marker path')
    common.publish(Path(marker),event)
    while True:time.sleep(.05)


def run(path=DEFAULT,fixture=False,variant='normal',cell_limit=None,on_event=None,cancel=None):
    started=time.perf_counter();path=_admit_path(path,fixture);plan,roots,units,identity=admit(fixture,variant,verify_prior=not fixture)
    require(cell_limit is None or (_integer(cell_limit) and cell_limit<=len(units)),'Invalid cell limit')
    if (path/'work.sqlite3').exists():
        with closing(_ro(path/'work.sqlite3')) as db:
            marker=_meta(db,'divergence_search_complete');tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables==TABLES:validate_prefix(db,roots,units,identity)
            else:require(tables=={'meta'} and _same(_meta(db,'identity'),identity) and marker is None,'Corrupt incomplete journal initialization')
        if marker is not None:
            _,_,body,resource=_closed(path,fixture,variant)
            manifest,report,export=_paths(path)
            for file,value in ((manifest,body),(report,body),(export,resource)):common.publish(file,value)
            common.baseline(verify_all=not fixture);return body
    on_event=on_event or emit
    with ExitStack() as stack:
        work=phase6_store.Work(path,identity);stack.callback(work.close);db=work.db
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS roots(state TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS setup(id INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts(seq INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS cells(idx INTEGER PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL)')
            if db.execute('SELECT COUNT(*) FROM roots').fetchone()[0]==0:
                for root in roots:db.execute('INSERT INTO roots VALUES(?,?,?)',(root['state_key'],config.canonical(root),config.digest(root)))
        setup,attempts,cells=validate_prefix(db,roots,units,identity);initial_cells=len(cells);initial_attempts=len(attempts)
        cpu=stack.enter_context(PerformanceCore(plan['cpu_declared']));evaluators={name:stack.enter_context(CpuBatchEvaluator(pathname,cache_capacity=0,max_batch=1,backend='auto')) for name,pathname in (('control',common.CONTROL),('candidate',common.CANDIDATE))}
        actual_setup=dict(evaluators={name:value.identity for name,value in evaluators.items()},cpu=cpu.metadata);_validate_setup(actual_setup,identity)
        if setup:require(_same(setup,actual_setup),'Resumed actual CPU/backend identity differs')
        else:
            with db:db.execute('INSERT INTO setup VALUES(0,?,?)',(config.canonical(actual_setup),config.digest(actual_setup)))
            setup=actual_setup
        target=len(units) if cell_limit is None else cell_limit
        on_event(dict(event='divergence_search_started',completed=len(cells),target=len(units),fixture=fixture,variant=variant))
        for index in range(len(cells),target):
            if _cancelled(cancel):break
            unit=units[index];root=roots[unit['root_index']];evaluator=evaluators[unit['profile']];before=_snapshot(evaluator);evaluator.reset_cache();cpu.verify()
            raw=quality_search.search(root['state'],evaluator,simulations=unit['simulations'],time_ms=0,max_nodes=unit['max_nodes'],rng=0,root_noise=None,cancel=cancel,c_puct=.25,memory_mb=8192)
            after=_snapshot(evaluator);counts={name:after[name]-before[name] for name in COUNTERS};semantic=semantic_report(raw)
            status='complete' if raw['stop_reason']=='simulations' and raw['completed_simulations']==unit['simulations'] else 'cancelled' if raw['stop_reason']=='cancelled' else 'budget_rejected'
            body=dict(schema=SCHEMA+'-attempt',seq=len(attempts),cell_index=index,unit=unit,root_sha256=config.digest(root),status=status,report=raw,report_sha256=config.digest(raw),semantic_report=semantic,semantic_sha256=config.digest(semantic),evaluator_before=before,evaluator_after=after,counts=counts,cpu_verified=cpu.verify(),evaluator_identity_sha256=config.digest(evaluator.identity))
            validate_attempt(body,len(attempts),unit,root,identity,setup,evaluator.native);work.admit(1024*1024)
            on_event(dict(event='divergence_search_cell_precommit',index=index,completed=len(cells),target=len(units),status=status,profile=unit['profile'],simulations=unit['simulations'],state_key=root['state_key']))
            with db:
                db.execute('INSERT INTO attempts VALUES(?,?,?)',(len(attempts),config.canonical(body),config.digest(body)))
                if status!='cancelled':db.execute('INSERT INTO cells VALUES(?,?,?)',(index,config.canonical(body),config.digest(body)))
            attempts.append(body)
            if status!='cancelled':cells.append(body)
            on_event(dict(event='divergence_search_cell_committed',index=index,completed=len(cells),target=len(units),status=status,profile=unit['profile'],simulations=unit['simulations'],state_key=root['state_key']))
            if status=='cancelled':break
        if len(cells)!=len(units):return dict(schema=SCHEMA+'-progress',status='interrupted',fixture=fixture,committed_cells=len(cells),complete_cells=sum(x['status']=='complete' for x in cells),requested_cells=len(units),attempts=len(attempts))
        _,attempts,cells=validate_prefix(db,roots,units,identity);body=summarize(roots,cells,identity)
        common.baseline(verify_all=not fixture)
        resource=dict(schema=SCHEMA+'-resources',identity_sha256=config.digest(identity),result_sha256=config.digest(body),elapsed_seconds_through_prepublication_validation=time.perf_counter()-started,initial_committed_cells=initial_cells,final_committed_cells=len(cells),initial_attempts=initial_attempts,final_attempts=len(attempts),host_peak=host_peak(),torch_imported='torch' in sys.modules,energy_measured=False,
            scope='This finalizing invocation, admission through validation before completion/public export; prior invocations, lost uncommitted kill work, fixtures and audits excluded')
        validate_resources(resource,identity,body);relative='resource-observations/'+uuid.uuid4().hex+'.json';file=path/relative;common.publish(file,resource)
        marker=dict(schema=SCHEMA+'-complete',identity_sha256=config.digest(identity),cells_sha256=config.digest(cells),attempts_sha256=config.digest(attempts),result_sha256=config.digest(body),resource_path=relative,resource_file_sha256=common.sha(file))
        with db:work.put('divergence_search_complete',marker)
    _,_,body,resource=_closed(path,fixture,variant)
    for file,value in zip(_paths(path),(body,body,resource)):common.publish(file,value)
    on_event(dict(event='divergence_search_complete',fixture=fixture,**body['counts']));return body


def verify(path=DEFAULT,fixture=False,variant='normal'):
    roots,cells,body,_=_closed(path,fixture,variant,require_exports=True);return body


def load_cells(path=DEFAULT,fixture=False,variant='normal'):
    body=verify(path,fixture,variant);return body['roots'],body['cells'],body


def primary_witness():
    body=verify();require(body['scope_finished'] is True and body['identity']['fixture'] is False and body['counts']['complete_cells']==40 and body['counts']['completed_simulations']==19200,'Primary fixed-simulation gate is incomplete')
    return dict(schema=SCHEMA+'-primary-witness',passed=True,fixture=False,complete_cells=40,completed_simulations=19200,plan_file_sha256=common.sha(PLAN),implementation_file_sha256=common.sha(IMPLEMENTATION),panel_file_sha256=common.sha(common.PANEL),
        journal_file_sha256=common.sha(DEFAULT/'work.sqlite3'),manifest_file_sha256=common.sha(DEFAULT/'manifest.json'),result_file_sha256=common.sha(OUTPUT),resource_file_sha256=common.sha(RESOURCES),semantic_cells_sha256=body['semantic_cells_sha256'],model_files=body['identity']['model_files'],model_weights=body['identity']['model_weights'],torch_imported_by_gate=False)
