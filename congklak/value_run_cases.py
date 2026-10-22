"""Frozen final-step value pilot: fit, conformance and recoverable search cases.

Completed readers use the standard library and certified persisted evidence.
Only a new decision opens native engines; each committed decision is immutable.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import ExitStack
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'build/value-run-v1/cases'
CANDIDATE = ROOT / 'data/value-training-v1/training/model.json'
CONTROL = ROOT / 'data/quality-train-v1/model.json'
RELAY = ROOT / 'data/training-stage5-deeper-h8-f127-seed2718/model.json'
RESERVATIONS = ROOT / 'data/neural-value-pilot-v2/reservations'
CACHE = ROOT / 'data/value-training-v1/cache'
SCHEMA = 'value-run-cases-v1'
ENGINES = ('H_v1', 'H8_relay_CPU', 'control', 'candidate')
PROFILES = ('control', 'candidate')
CPU = dict(id=258, group=0, logical=2, core=2, cache=0, numa=0, efficiency_class=1)
LIMITS = dict(max_depth=16, tt_mb=16, use_pvs=True,
              puct=dict(simulations=100000, max_nodes=2048))
SOURCES = ('congklak/value_run_cases.py', 'tools/evaluate-value-run.py',
           'tests/test_value_run_cases.py',
           'congklak/quality_search.py', 'congklak/value_sidecar_v2.py',
           'congklak/puct_cpu_inference.py', 'congklak/neural_relay.py',
           'congklak/native.py', 'congklak/timing_v2.py', 'congklak/config.py',
           'congklak/features_relay.py', 'congklak/features_v3.py',
           'tools/benchmark-phase6-v2.py')
HOOKS = ('before_decision_commit', 'after_decision_commit')
_PRIOR = None


def need(ok, message):
    if not ok: raise RuntimeError(message)


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(body): return hashlib.sha256(canonical(body).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('wb') as stream:
        stream.write((canonical(body) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def pins(paths): return {str(Path(p).resolve()): sha(p) for p in paths}


def current(mapping):
    need(isinstance(mapping, dict) and mapping, 'Missing source certificate')
    for name, checksum in mapping.items():
        need(Path(name).is_file() and sha(name) == checksum, 'Input changed: ' + name)


def _prior():
    global _PRIOR
    if _PRIOR is not None: return _PRIOR
    spec = importlib.util.spec_from_file_location('_value_run_frozen_baselines', ROOT / 'tools/benchmark-phase6-v2.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _PRIOR = module
    return _PRIOR


def _counts(obj): return dict(obj.snapshot_counters())


def open_engines(stack, candidate_path=None):
    """Identical engine construction shared by case and arena workers."""
    from .native import Native
    from .neural_relay import NeuralNative
    from .puct_cpu_inference import CpuBatchEvaluator
    from .value_sidecar_v2 import SidecarEvaluator
    _prior()  # Import/startup is outside every measured decision, for all engines.
    return {'H_v1': ('h', Native()),
            'H8_relay_CPU': ('relay', stack.enter_context(NeuralNative(RELAY, mode='value-none', backend='avx2'))),
            'control': ('puct', stack.enter_context(CpuBatchEvaluator(CONTROL, cache_capacity=0, backend='avx2'))),
            'candidate': ('puct', stack.enter_context(SidecarEvaluator(candidate_path or CANDIDATE, cache_capacity=0, backend='avx2')))}


def _empty(p, simulations):
    from . import quality_search as q, config
    legal = [a for a, n in enumerate(p['pits'][p['controller']]) if n]
    d = p['stores'][0] - p['stores'][1]
    return dict(schema=q.SCHEMA, protocol=q.PROTOCOL, rules_digest=config.RULES_DIGEST,
        position=p, chosen=legal[0], visits=[0]*7,
        policy=[1/len(legal) if a in legal else 0. for a in range(7)], root_priors=[0.]*7,
        estimated_value_player0=d, action_mean_value_player0=[None]*7,
        root_expanded=False, completed_simulations=0, requested_simulations=simulations,
        nodes=1, max_nodes=2048, usable_decision=False, memory_mb=8192,
        stop_reason='time_budget', proof_status='monte_carlo_estimate',
        work=dict(transitions=0, deposits=0, feature_simulations=0, feature_deposits=0,
                  neural_evaluations=0, neural_calls=0, inference_ms=0., root_expansions=0, total_deposits=0),
        timing=dict(elapsed_ms=0., requested_time_ms=0, overshoot_ms=0., whole_decision_includes_inference=True),
        forest_batch_calls=0, forest_size=1, value_perspective='fixed_player0_seeds',
        first_play_urgency='parent_neural_bootstrap', unvisited_root_q_player0=d, c_puct=.25)


def _decision(name, engine, p, ms, simulations, cpu):
    from . import quality_search as q
    kind, obj = engine
    if cpu is not None: cpu.verify()
    start = time.perf_counter()
    if kind != 'puct':
        need(ms > 0, 'Fixed backups apply only to neural PUCT')
        # Every baseline uses the same positive floored remaining deadline.
        grant = max(0, math.floor(ms - (time.perf_counter()-start)*1000))
        if grant:
            answer = _prior().decide(engine, p, grant, LIMITS)
        else:
            answer = dict(kind=kind, report=None, stop_reason='time_budget', engine_elapsed_ms=0.,
                          nodes=0, usable_decision=False, chosen=None, completed_depth=None,
                          completed_simulations=None, wall_ms=.001, requested_ms=ms, overshoot_ms=0.)
        answer['requested_ms'] = ms
        telemetry = None
    else:
        before = _counts(obj); tick = time.perf_counter(); obj.begin_decision()
        reset_ms = (time.perf_counter()-tick)*1000; reset = _counts(obj)
        grant = max(0, math.floor(ms-(time.perf_counter()-start)*1000)) if ms else 0
        report = q.search(p, obj, simulations=simulations, time_ms=grant,
                          max_nodes=2048, c_puct=.25, root_noise=None, memory_mb=8192) if (not ms or grant) else _empty(p, simulations)
        after = _counts(obj)
        answer = dict(kind='puct', report=report, stop_reason=report['stop_reason'],
            engine_elapsed_ms=report['timing']['elapsed_ms'], nodes=report['nodes'],
            usable_decision=report['usable_decision'], chosen=report['chosen'] if report['usable_decision'] else None,
            completed_depth=None, completed_simulations=report['completed_simulations'],
            wall_ms=.001, requested_ms=ms, overshoot_ms=0.)
        telemetry = dict(protocol='value-run-physical-counters-v1', profile=name, before=before,
            after_reset=reset, after=after, delta={k:after[k]-before[k] for k in after if k not in
                ('cache_capacity', 'cache_occupancy', 'cache_current_entries')},
            reset_ms=reset_ms, granted_search_ms=grant, cache_capacity=0,
            deadline_policy='floor positive remaining whole milliseconds after reset; zero never unlimited',
            branch_bookkeeping=('branch_calls', 'branch_bypasses', 'base_ms', 'branch_ms'))
    elapsed = (time.perf_counter()-start)*1000
    if ms and answer['stop_reason'] == 'time_budget' and elapsed < ms:
        time.sleep((ms-elapsed)/1000)
    elapsed = (time.perf_counter()-start)*1000
    answer.update(wall_ms=elapsed, overshoot_ms=max(0., elapsed-ms) if ms else 0.)
    result = dict(decision=answer, evaluator=telemetry)
    validate_decision(result, p, ms, name)
    if cpu is not None: cpu.verify()
    elapsed = (time.perf_counter()-start)*1000
    answer.update(wall_ms=elapsed, overshoot_ms=max(0., elapsed-ms) if ms else 0.)
    return result


def timed_decision(name, engine, p, ms, cpu=None):
    need(type(ms) is int and ms in (25, 100), 'Only frozen 25/100 ms clocks admitted')
    return _decision(name, engine, p, ms, 100000, cpu)


def fixed_decision(name, engine, p, simulations=128, cpu=None):
    need(simulations == 128 and name in PROFILES, 'Only fixed128 neural profiles admitted')
    return _decision(name, engine, p, 0, 128, cpu)


def validate_decision(envelope, p, ms, name, native=None):
    """Live decision validation; completed paths validate certificates instead."""
    from . import quality_search as q
    need(set(envelope) == {'decision','evaluator'} and name in ENGINES, 'Decision envelope differs')
    a = envelope['decision']; kind = 'h' if name == 'H_v1' else 'relay' if name == 'H8_relay_CPU' else 'puct'
    need(a['kind'] == kind and a['requested_ms'] == ms, 'Decision engine/clock differs')
    need(math.isfinite(a['wall_ms']) and a['wall_ms'] > 0 and a['overshoot_ms'] == (max(0.,a['wall_ms']-ms) if ms else 0.), 'Whole decision clock differs')
    if kind != 'puct':
        _prior().validate_decision(a, p, ms, LIMITS)
        need(envelope['evaluator'] is None, 'Baseline fabricated neural telemetry'); return
    r = a['report']; requested = 100000 if ms else 128
    q.validate_report(r, p, requested, c_puct=.25, native=native)
    need(a['chosen'] == (r['chosen'] if r['usable_decision'] else None) and
         a['usable_decision'] == r['usable_decision'] and a['nodes'] == r['nodes'] and
         a['completed_simulations'] == r['completed_simulations'] and a['completed_depth'] is None and
         a['engine_elapsed_ms'] == r['timing']['elapsed_ms'] and a['engine_elapsed_ms'] <= a['wall_ms']+1,
         'External recommendation or work differs')
    t = envelope['evaluator']; need(t['profile'] == name and t['cache_capacity'] == 0, 'Cache/profile differs')
    b, reset, after = t['before'], t['after_reset'], t['after']
    need(t['delta'] == {k:after[k]-b[k] for k in after if k not in ('cache_capacity','cache_occupancy','cache_current_entries')}, 'Physical counter delta differs')
    d = t['delta']; need(d['cache_resets'] == 1 and d['cache_hits'] == d['cache_evictions'] == 0, 'Cache carries decision state')
    need(after['cache_capacity'] == 0 and after.get('cache_current_entries',after.get('cache_occupancy',0)) == 0, 'Cache enabled')
    need(d['native_actual_evals'] == r['work']['neural_evaluations'] and
         d['feature_simulations'] == r['work']['feature_simulations'] and
         d['feature_deposits'] == r['work']['feature_deposits'], 'Physical work differs')
    need(r['timing']['requested_time_ms'] == t['granted_search_ms'] and
         (not ms or 0 <= t['granted_search_ms'] <= math.floor(ms-t['reset_ms'])), 'Reset granted extra clock')
    if ms and a['stop_reason'] == 'time_budget': need(a['wall_ms'] >= ms, 'Premature timeout clock')


def physical_view(envelope):
    """Fixed-work equivalence excludes clocks and branch bypass bookkeeping."""
    r = envelope['decision']['report']
    report = {k:v for k,v in r.items() if k not in ('timing','work')}
    report['work'] = {k:v for k,v in r['work'].items() if k != 'inference_ms'}
    # Evaluator counters are cumulative per provider. Compare per-decision deltas.
    report['physical_delta'] = {k:v for k,v in envelope['evaluator']['delta'].items()
        if k not in ('branch_calls','branch_bypasses','base_ms','branch_ms')}
    return report


def _model_binding(candidate):
    from . import value_training as training
    report = training.completed(Path(candidate).parent)
    need(report['production_steps'] == 400 and not report['engineering_fixture'], 'Only fixed final400 production candidate')
    return dict(path=str(Path(candidate).resolve()), sha256=sha(candidate), training_report_sha256=digest(report))


def panel():
    from . import value_training_data as data
    rows, manifest = data.load_fit(CACHE)
    selection = read(RESERVATIONS/'selection.json')
    bykey = {r['state_key']:r for r in rows}
    fit = []
    for metadata in selection['fit_cases']:
        row = bykey[metadata['state_key']]
        need(row['metadata_sha256'] == digest(metadata), 'Selected fit case metadata differs')
        fit.append(dict(cohort='fit_H8', state=row['state'], state_key=row['state_key'],
            family=row['family'], component=row['component'], stratum=row['stratum'], cell=row['cell'],
            reference=dict(action_scores=row['target_source']['action_q'], best_moves=row['target_source']['best_moves'],
                           target_player0=row['target_player0']), target_source=row['target_source']))
    exact = read(ROOT/'build/ordinary-value-benchmark-projection.json')['cases']
    forensic = read(ROOT/'build/divergence-panel.json')['roots']
    need(len(rows) == 128 and len(fit) == len(exact) == 32 and len(forensic) == 5, 'Panel count differs')
    exact = [{**r, 'cohort':'exact_guard'} for r in exact]
    forensic = [{**r, 'cohort':'reused_forensic', 'reference':None} for r in forensic]
    return dict(fit_rows=rows, fit_cases=fit, exact=exact, forensic=forensic,
                fit_check_is_unseen=False, source_manifest_sha256=digest(manifest))


def _tasks(p, fixture=False):
    tasks = []
    roots = p['fit_cases'] + p['exact']
    if not fixture:
        for index, row in enumerate(roots):
            for bi, ms in enumerate((25,100)):
                offset = (index+bi)%4; order = list(ENGINES[offset:]+ENGINES[:offset])
                for name in order: tasks.append(dict(mode='timed', root=row, engine=name, ms=ms))
    fixed = [p['fit_cases'][0],p['exact'][0],p['forensic'][0]] if fixture else roots+p['forensic']
    for index,row in enumerate(fixed):
        for name in PROFILES if index%2 == 0 else tuple(reversed(PROFILES)):
            tasks.append(dict(mode='fixed128',root=row,engine=name,ms=0))
    need(len(tasks) == (6 if fixture else 650), 'Search schedule differs')
    return tasks


def identity(candidate=CANDIDATE, fixture=False):
    p = panel(); paths = [ROOT/n for n in SOURCES]
    paths += [CONTROL,RELAY,ROOT/'build/congklak_native.dll',ROOT/'build/congklak_neural_relay.dll',
        ROOT/'build/congklak_puct_cpu.dll',ROOT/'build/congklak_value_sidecar.dll',
        RESERVATIONS/'manifest.json',RESERVATIONS/'selection.json',
        CACHE/'manifest.json',CACHE/'fit-check.json',
        ROOT/'build/ordinary-value-benchmark-projection.json',ROOT/'build/divergence-panel.json']
    return dict(schema=SCHEMA+'-identity', engineering_fixture=fixture,
        candidate=_model_binding(candidate), inputs=pins(paths), cpu=CPU, limits=LIMITS,
        puct=dict(c_puct=.25, cache_capacity=0, root_noise=None, backend='avx2-double-packed'),
        tasks_sha256=digest(_tasks(p, fixture)), task_count=6 if fixture else 650,
        fit_selection_sha256=digest(p['fit_rows']), equal_time_exclusive=True,
        no_result_selection=True, new_teacher_labels=0, optimizer_updates=0)


def _writer(path):
    from .valuepilot_reservations_v2 import Writer
    return Writer(path)


def _catalog(path):
    frozen = read(path/'identity.json'); anchor = read(path/'initialized.json')
    need(anchor == dict(schema=SCHEMA+'-initialized', identity_sha256=sha(path/'identity.json')), 'Initialization identity differs')
    current(frozen['inputs']); need(_model_binding(frozen['candidate']['path']) == frozen['candidate'], 'Candidate training evidence changed')
    schedule = read(path/'tasks.json')
    need(len(schedule) == frozen['task_count'] and digest(schedule) == frozen['tasks_sha256'], 'Frozen case schedule differs')
    cat = read(path/'catalog.json')
    need(cat['schema'] == SCHEMA+'-catalog' and cat['identity_sha256'] == sha(path/'identity.json') and
         cat['count'] == len(cat['records']) <= frozen['task_count'] and cat['prefix_sha256'] == digest(cat['records']), 'Committed prefix differs')
    for index, entry in enumerate(cat['records']):
        need(entry['index'] == index and Path(entry['name']).name == entry['name'] and
             sha(path/'records'/entry['name']) == entry['sha256'], 'Committed decision missing or corrupt')
        item = read(path/'records'/entry['name'])
        need(item['index'] == index and item['identity_sha256'] == cat['identity_sha256'] and item['task'] == schedule[index], 'Committed decision identity differs')
        need(item['score'] == _score(item), 'Committed regret differs from frozen full action Q')
    return frozen, cat


def completed(output=DEFAULT):
    path = Path(output).resolve(); frozen, cat = _catalog(path)
    need(cat['count'] == frozen['task_count'], 'Case journal incomplete')
    cert = read(path/'certificate.json')
    need(cert['schema'] == SCHEMA+'-certificate' and cert['passed'] is True and cert['catalog_sha256'] == sha(path/'catalog.json'), 'Case certificate differs')
    current(cert['files'])
    result = read(path/'report.json'); need(digest(result) == cert['report_digest'] and result == summarize(path,frozen,cat), 'Case report differs')
    return result


def status(output=DEFAULT):
    path=Path(output).resolve()
    if not path.exists(): return dict(status='not_started', decisions=0, total=650)
    frozen, cat = _catalog(path)
    if (path/'certificate.json').exists(): return completed(path)
    return dict(status='partial',decisions=cat['count'],total=frozen['task_count'])


def _progress(path,cat,tasks,operation,reason=None):
    done=tasks[:cat['count']]
    timed=sum(r['mode']=='timed' for r in done);fixed=len(done)-timed
    atomic(path/'progress.json',dict(schema=SCHEMA+'-progress',authoritative=False,
        timed=dict(committed=timed,target=0 if len(tasks)==6 else 512),
        fixed=dict(committed=fixed,target=6 if len(tasks)==6 else 138),
        fit=dict(committed=128 if (ROOT/'build/value-run-v1/cases-fit.json').exists() else 0,target=128),
        conformance=dict(committed=733 if (ROOT/'build/value-run-v1/cases-conformance.json').exists() else 0,target=733),
        operation=operation,stop_reason=reason,source_pins=pins([ROOT/n for n in SOURCES])))


def _pause(path, point, requested, marker, index, fixture, pause_index):
    if point != requested or index != pause_index: return
    need(fixture and marker, 'Pause hooks require engineering fixture')
    atomic(marker,dict(event=point,index=index,pid=os.getpid(),output=str(path)))
    while True: time.sleep(.05)


def _attempt_event(log,event,**fields):
    body=dict(event=event,pid=os.getpid(),utc_ns=time.time_ns(),**fields)
    with log.open('ab') as stream:
        stream.write((canonical(body)+'\n').encode());stream.flush();os.fsync(stream.fileno())


def attempt_accounting(path):
    logs=[];started=calculated=committed=warmups=backups=0
    for file in sorted((path/'attempts').glob('*.jsonl')):
        events=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
        for event in events:
            started+=event['event']=='decision_start';calculated+=event['event']=='decision_computed'
            committed+=event['event']=='decision_committed';warmups+=event['event']=='warmup_complete'
            if event['event']=='decision_computed':backups+=event['completed_backups']
        logs.append(dict(path=file.name,sha256=sha(file),events=len(events),
                         clean_exit=bool(events and events[-1]['event']=='invocation_end')))
    return dict(invocations=logs,started_decisions=started,computed_decisions=calculated,
        committed_decisions=committed,warmup_profile_calls=warmups,
        computed_actual_backups_including_uncommitted=backups,
        interrupted_before_computation_report=started-calculated,
        warmup_rows_per_profile_call=8,counts_include_engineering_fixture=path!=DEFAULT.resolve())


def _score(record):
    root=record['task']['root']; ref=root.get('reference'); chosen=record['answer']['decision']['chosen']
    if ref is None or chosen is None:return dict(optimal=None,regret=None)
    q={r['pit']:r['value_player0'] for r in ref['action_scores']}
    need(chosen in q, 'Decision absent from saved legal Q')
    sign=1 if root['state']['controller']==0 else -1
    return dict(optimal=chosen in ref['best_moves'],regret=sign*(ref['target_player0']-q[chosen]))


def summarize(path, frozen, cat):
    records=[read(path/'records'/r['name']) for r in cat['records']]
    buckets=defaultdict(list); low=defaultdict(dict)
    for r in records:
        task=r['task']; buckets[(task['mode'],task['root']['cohort'],str(task['ms']),task['engine'])].append(r)
        if task['mode']=='fixed128' and sum(map(sum,task['root']['state']['pits']))<=24:
            low[task['root']['state_key']][task['engine']]=r['answer']
    metrics={}
    for key,values in buckets.items():
        scored=[r['score'] for r in values if r['score']['regret'] is not None]
        answers=[r['answer']['decision'] for r in values]
        metrics['/'.join(key)]=dict(positions=len(values),fallbacks=sum(not a['usable_decision'] for a in answers),
            scored=len(scored),optimal=sum(r['optimal'] for r in scored),
            mean_regret=statistics.mean(r['regret'] for r in scored) if scored else None,
            mean_wall_ms=statistics.mean(a['wall_ms'] for a in answers),max_overshoot_ms=max(a['overshoot_ms'] for a in answers),
            completed_backups=sum(a['completed_simulations'] or 0 for a in answers),
            physical_feature_simulations=sum((r['answer']['evaluator'] or {}).get('delta',{}).get('feature_simulations',0) for r in values))
    equal=all(set(pair)==set(PROFILES) and physical_view(pair['control'])==physical_view(pair['candidate']) for pair in low.values())
    guard={str(ms):dict(roots=32, candidate_optimal=metrics.get(f'timed/exact_guard/{ms}/candidate',{}).get('optimal',0),
        candidate_fallbacks=metrics.get(f'timed/exact_guard/{ms}/candidate',{}).get('fallbacks',32),
        passed=metrics.get(f'timed/exact_guard/{ms}/candidate',{}).get('optimal')==32 and metrics.get(f'timed/exact_guard/{ms}/candidate',{}).get('fallbacks')==0)
        for ms in (25,100)}
    regret={}
    for mode,clock in (('timed',25),('timed',100),('fixed128',0)):
        a=metrics.get(f'{mode}/fit_H8/{clock}/control',{});b=metrics.get(f'{mode}/fit_H8/{clock}/candidate',{})
        regret[f'{mode}/{clock}']=dict(control=a.get('mean_regret'),candidate=b.get('mean_regret'),
            passed=a.get('scored')==b.get('scored')==32 and b['mean_regret']<=a['mean_regret'])
    timed=sum(r['task']['mode']=='timed' for r in records);fixed=len(records)-timed
    return dict(schema=SCHEMA+'-report',status='complete',engineering_fixture=frozen['engineering_fixture'],
        identity=frozen,decisions=len(records),timed_measurements=timed,fixed_searches=fixed,
        fixed_requested_backups=fixed*128,fixed_completed_backups=sum(r['answer']['decision']['completed_simulations'] for r in records if r['task']['mode']=='fixed128'),
        metrics=metrics, exact_guard=guard, h8_regret_gates=regret,
        low_active=dict(roots=len(low),searches=sum(len(v) for v in low.values()),passed=equal,
                        exact_action_visits_q_physical_work=equal),
        engineering_valid=equal, new_labels=0,optimizer_updates=0,default_promotion=False,
        fit_check_is_unseen=False,forensic_reuse=True,attempt_accounting=attempt_accounting(path))


def run(output=DEFAULT,candidate=CANDIDATE,fixture=False,stop_after=None,pause_at=None,pause_marker=None,pause_index=1):
    path=Path(output).resolve()
    if (path/'certificate.json').exists():return completed(path)
    need(path.is_relative_to(ROOT/'build/value-run-v1'),'Case output outside new namespace')
    need(pause_at in (None,*HOOKS) and (fixture or pause_at is None),'Production pause forbidden')
    frozen=identity(candidate,fixture);p=panel();tasks=_tasks(p,fixture)
    path.mkdir(parents=True,exist_ok=True)
    with _writer(path):
        if (path/'initialized.json').exists():
            actual,cat=_catalog(path);need(actual==frozen,'Resume identity/source/schedule differs')
        else:
            need(not (path/'identity.json').exists() and not (path/'records').exists(),'Incomplete initialization requires review')
            atomic(path/'identity.json',frozen)
            atomic(path/'tasks.json',tasks)
            atomic(path/'initialized.json',dict(schema=SCHEMA+'-initialized',identity_sha256=sha(path/'identity.json')))
            cat=dict(schema=SCHEMA+'-catalog',identity_sha256=sha(path/'identity.json'),count=0,records=[],prefix_sha256=digest([]))
            atomic(path/'catalog.json',cat)
        from .timing_v2 import PerformanceCore
        _progress(path,cat,tasks,'opening engines')
        (path/'attempts').mkdir(exist_ok=True)
        log=path/'attempts'/f'{time.time_ns()}-{os.getpid()}.jsonl'
        _attempt_event(log,'invocation_start',committed_prefix=cat['count'])
        with ExitStack() as stack:
            cpu=stack.enter_context(PerformanceCore(CPU));engines=open_engines(stack,candidate)
            # Same declared warmup each invocation; never counted as case decisions.
            for name in PROFILES:
                _attempt_event(log,'warmup_start',profile=name,rows=8)
                before=_counts(engines[name][1]);engines[name][1].evaluate_many([r['state'] for r in p['fit_cases'][:8]])
                after=_counts(engines[name][1]);_attempt_event(log,'warmup_complete',profile=name,rows=8,
                    physical_delta={k:after[k]-before[k] for k in after if k not in ('cache_capacity','cache_occupancy','cache_current_entries')})
            for index in range(cat['count'],min(len(tasks),stop_after if stop_after is not None else len(tasks))):
                task=tasks[index];root=task['root'];name=task['engine'];cpu.verify()
                _attempt_event(log,'decision_start',index=index,mode=task['mode'],engine=name)
                answer=(timed_decision(name,engines[name],root['state'],task['ms'],cpu) if task['mode']=='timed' else
                        fixed_decision(name,engines[name],root['state'],cpu=cpu))
                _attempt_event(log,'decision_computed',index=index,completed_backups=answer['decision']['completed_simulations'] or 0,
                    answer_sha256=digest(answer),physical_delta=(answer['evaluator'] or {}).get('delta'))
                record=dict(schema=SCHEMA+'-decision',index=index,identity_sha256=cat['identity_sha256'],task=task,answer=answer,cpu_verified=cpu.verify())
                record['score']=_score(record)
                # Unique physical files retain every uncommitted attempt after a kill.
                filename=f'{index:04d}-{time.time_ns()}-{os.getpid()}.json'
                atomic(path/'records'/filename,record)
                _pause(path,'before_decision_commit',pause_at,pause_marker,index,fixture,pause_index)
                entry=dict(index=index,name=filename,sha256=sha(path/'records'/filename))
                cat['records'].append(entry);cat['count']+=1;cat['prefix_sha256']=digest(cat['records']);atomic(path/'catalog.json',cat)
                _attempt_event(log,'decision_committed',index=index,record_sha256=entry['sha256'])
                _pause(path,'after_decision_commit',pause_at,pause_marker,index,fixture,pause_index)
                _progress(path,cat,tasks,'timed cases' if task['mode']=='timed' else 'fixed128 diagnostics')
                print(canonical(dict(event='value_case_committed',completed=cat['count'],total=len(tasks),mode=task['mode'])),flush=True)
        _attempt_event(log,'invocation_end',committed_prefix=cat['count'])
        if cat['count']<len(tasks):
            _progress(path,cat,tasks,'resumable','cooperative decision limit')
            return dict(status='partial',decisions=cat['count'],total=len(tasks))
        result=summarize(path,frozen,cat);atomic(path/'report.json',result)
        files=pins([path/'identity.json',path/'tasks.json',path/'initialized.json',path/'report.json']+
            [path/'records'/r['name'] for r in cat['records']]+list((path/'attempts').glob('*.jsonl')))
        atomic(path/'certificate.json',dict(schema=SCHEMA+'-certificate',passed=True,catalog_sha256=sha(path/'catalog.json'),files=files,report_digest=digest(result)))
        _progress(path,cat,tasks,'complete')
    return completed(path)


def _aggregate(rows):
    return dict(roots=len(rows),normalized_huber=statistics.mean(r['huber'] for r in rows),
        normalized_mae=statistics.mean(abs(r['normalized_error']) for r in rows),
        normalized_rmse=math.sqrt(statistics.mean(r['normalized_error']**2 for r in rows)),
        seed_mae=statistics.mean(abs(r['seed_error']) for r in rows),
        seed_rmse=math.sqrt(statistics.mean(r['seed_error']**2 for r in rows)),seed_bias=statistics.mean(r['seed_error'] for r in rows))


def fit(output=ROOT/'build/value-run-v1/cases-fit.json',candidate=CANDIDATE):
    output=Path(output)
    if output.exists():
        result=read(output);current(result['inputs']);need(_model_binding(candidate)==result['candidate'],'Fit candidate training evidence changed');return result
    from .neural_relay import Model
    from .value_sidecar_v2 import Model as SidecarModel
    p=panel();base=Model(CONTROL);model=SidecarModel(candidate);_model_binding(candidate)
    rows=[]
    for row in p['fit_rows']:
        predictions={'control':base.reference(row['state']),'candidate':model.reference(row['state'])}
        need(predictions['control'][2]==predictions['candidate'][2],'Frozen policy changed in fit')
        values={}
        for name,value in predictions.items():
            error=(value[0]-row['target_player0'])/row['r'];a=abs(error)
            values[name]=dict(value_player0=value[0],normalized_error=error,seed_error=value[0]-row['target_player0'],
                huber=.5*error*error if a<=.1 else .1*(a-.05))
        rows.append(dict(state_key=row['state_key'],cell=row['cell'],stratum=row['stratum'],actor=row['actor'],
            target_player0=row['target_player0'],target_source=row['target_source'],predictions=values))
    summaries={}
    for name in PROFILES:
        summaries[name]={'all':_aggregate([r['predictions'][name] for r in rows])}
        for field in ('actor','stratum','cell'):
            for value in sorted({str(r[field]) for r in rows}):
                summaries[name][field+'/'+value]=_aggregate([r['predictions'][name] for r in rows if str(r[field])==value])
        summaries[name]['capture']=_aggregate([r['predictions'][name] for r in rows if 'capture' in r['stratum']])
    a,b=summaries['control'],summaries['candidate']
    gate=dict(strictly_lower_huber=b['all']['normalized_huber']<a['all']['normalized_huber'],
              no_higher_overall_seed_mae=b['all']['seed_mae']<=a['all']['seed_mae'],
              no_higher_capture_seed_mae=b['capture']['seed_mae']<=a['capture']['seed_mae'])
    result=dict(schema=SCHEMA+'-fit',passed=all(gate.values()),gate=gate,metrics=summaries,rows=rows,candidate=_model_binding(candidate),
        policy_fixed=True,fit_check_is_unseen=False,optimizer_updates=0,
        method='literal sequential float64 exported FP32 weights; original complete finite-H8 targets',
        inputs=pins([ROOT/n for n in SOURCES]+[CONTROL,candidate,CACHE/'fit-check.json',CACHE/'manifest.json']))
    atomic(output,result);return result


def conformance(output=ROOT/'build/value-run-v1/cases-conformance.json',candidate=CANDIDATE,torch_check=False,
                fp32_proof=ROOT/'build/value-run-v1/audit-training-v1/FP32-admission.json'):
    output=Path(output)
    if output.exists():
        result=read(output);current(result['inputs']);return result
    from . import value_sidecar_v2 as side
    from .neural_relay import Model
    from .puct_cpu_inference import CpuBatchEvaluator
    _model_binding(candidate);model=side.Model(candidate);base=Model(CONTROL)
    # Reuse the already certified 733 occurrence corpus without selecting outcomes.
    corpus_path=ROOT/'build/ordinary-development-v1/conformance-owner-corpus.json';p=read(corpus_path)
    rows=[r['state'] for group in ('selected','guards','boundaries') for r in p[group]]
    need(len(rows)==733,'Learned conformance corpus differs')
    maxima=dict(seed=0.,logit=0.,score_units=0);outside=policy=0;results=[]
    for backend in ('scalar','avx2'):
        with CpuBatchEvaluator(CONTROL,backend=backend) as a,side.SidecarEvaluator(model,backend=backend) as b:
            current_rows=[]
            for start in range(0,len(rows),32):
                chunk=rows[start:start+32]
                aa=a.evaluate_many(chunk);bb=b.evaluate_many(chunk)
                for state,x,y in zip(chunk,aa,bb):
                    need(struct.pack('<7d',*x['policy_logits'])==struct.pack('<7d',*y['policy_logits']),'Learned native policy changed');policy+=1
                    ref=model.reference(state);units=b.evaluate_units(state)
                    maxima['seed']=max(maxima['seed'],abs(y['value_player0']-ref[0]))
                    maxima['logit']=max(maxima['logit'],max(abs(v-w) for v,w in zip(y['policy_logits'],ref[2])))
                    maxima['score_units']=max(maxima['score_units'],abs(units['value_units']-ref[1]))
                    need(maxima['seed']<=1e-10 and maxima['logit']<=1e-10 and maxima['score_units']==0,'Native learned algebra mismatch')
                    for k in ('feature_simulations','feature_deposits','native_actual_evals'):need(x[k]==y[k],'Sidecar repeated feature work')
                    if not 25<=sum(map(sum,state['pits']))<=74:
                        need(x['value_player0']==y['value_player0'],'Outside gate value changed');outside+=1
                    current_rows.append({k:y[k] for k in ('value_player0','policy_logits','feature_simulations','feature_deposits')})
            if backend=='scalar':results=current_rows
            else:need(current_rows==results,'Scalar/AVX2 outputs differ')
    fp32=None;proof_inputs={}
    if Path(fp32_proof).exists():
        proof=read(fp32_proof)
        need(proof['schema']=='value-run-independent-FP32-admission-v1' and proof['passed'] is True and
             proof['candidate_sha256']==sha(candidate) and Path(proof['candidate_path']).resolve()==Path(candidate).resolve() and
             proof['corpus_rows']==733 and proof['nonterminal_rows']==731 and proof['terminal_rows']==2 and
             proof['cuda_initialized'] is False and proof['optimizer_steps']==0 and
             proof['limits']==dict(logits=1e-5,score_units=1,seed=.001) and
             all(proof['maximum'][k]<=v for k,v in proof['limits'].items()),'Independent FP32 admission differs')
        for key in ('methods','worker_report','worker_spec'):
            ref=proof[key];file=ROOT/ref['path'];need(sha(file)==ref['sha256'],'Independent FP32 evidence changed')
            proof_inputs[str(file.resolve())]=sha(file)
        proof_inputs[str(Path(fp32_proof).resolve())]=sha(fp32_proof)
        fp32=dict(passed=True,independent=True,proof=proof,source_sha256=sha(fp32_proof))
    if torch_check:
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        frozen={k:torch.tensor(v,dtype=torch.float32) for k,v in base.weights.items()};branch={k:torch.tensor(v,dtype=torch.float32) for k,v in model.weights.items()}
        mx=dict(seeds=0.,policy_logits=0.,score_units=0)
        with torch.no_grad():
            for state in rows:
                x,d,r=side.features_relay.encode(state,127)
                if not r:continue
                h=torch.tensor(x,dtype=torch.float32)
                for prefix in ('layers.0','layers.2'):h=torch.relu(torch.mv(frozen[prefix+'.weight'],h)+frozen[prefix+'.bias'])
                u=(torch.mv(frozen['layers.4.weight'],h)+frozen['layers.4.bias'])[0]
                logits=(torch.mv(frozen['policy.weight'],h)+frozen['policy.bias']).tolist()
                if 25<=sum(map(sum,state['pits']))<=74:u=u+(torch.mv(branch['v'],torch.relu(torch.mv(branch['A'],torch.tensor(x,dtype=torch.float32))+branch['b']))+branch['c'])[0]
                residual=float((r*torch.tanh(u)).item());ref=model.reference(state)
                mx['seeds']=max(mx['seeds'],abs(d+residual-ref[0]));mx['policy_logits']=max(mx['policy_logits'],max(abs(v-w) for v,w in zip(logits,ref[2])))
                mx['score_units']=max(mx['score_units'],abs(d*256+side.rounded(residual*256)-ref[1]))
        need(mx['seeds']<=.001 and mx['policy_logits']<=1e-5 and mx['score_units']<=1,'FP32 learned conformance failed')
        need(not torch.cuda.is_initialized(),'CUDA initialized')
        fp32=dict(passed=True,maximum_errors=mx,torch=torch.__version__,device='cpu',cuda_initialized=False)
    need(fp32 is not None,'Full learned conformance requires separately granted Torch execution')
    result=dict(schema=SCHEMA+'-conformance',passed=True,rows=733,policy_comparisons=policy,policy_bit_exact=True,
        additional_feature_simulations=0,native_maximum_errors=maxima,torch_fp32=fp32,outside_direct_equal=outside,
        scalar_avx2_equal=True,optimizer_updates=0,inputs={**pins([ROOT/n for n in SOURCES]+[CONTROL,candidate,corpus_path]),**proof_inputs})
    atomic(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('run','verify','status','fit','conformance'))
    parser.add_argument('--output');parser.add_argument('--candidate',default=str(CANDIDATE))
    parser.add_argument('--engineering-fixture',action='store_true');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--pause-at',choices=HOOKS);parser.add_argument('--pause-marker');parser.add_argument('--pause-index',type=int,default=1)
    parser.add_argument('--torch-check',action='store_true');args=parser.parse_args()
    if args.operation=='run':result=run(args.output or DEFAULT,args.candidate,args.engineering_fixture,args.stop_after,args.pause_at,args.pause_marker,args.pause_index)
    elif args.operation=='verify':result=completed(args.output or DEFAULT)
    elif args.operation=='status':result=status(args.output or DEFAULT)
    elif args.operation=='fit':result=fit(args.output or ROOT/'build/value-run-v1/cases-fit.json',args.candidate)
    else:result=conformance(args.output or ROOT/'build/value-run-v1/cases-conformance.json',args.candidate,args.torch_check)
    print(canonical({k:v for k,v in result.items() if k in ('schema','status','passed','decisions','total','timed_measurements','fixed_searches','gate','metrics','engineering_valid')}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
