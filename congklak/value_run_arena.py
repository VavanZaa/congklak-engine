"""Fixed-policy value pilot arena: immutable decisions, literal replay, clustered results.

Raw committed choices are never searched again. Uncommitted files remain retained
orphans; only the full ordered head prefix is authoritative. Completed verification
uses standard-library algebra/rules before any writer or engine entry point.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import ExitStack
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid
from . import config, neurodiag_common as common, neurodiag_data as metadata
from .valuepilot_reservations import Writer

ROOT=common.ROOT
DEFAULT=ROOT/'data/value-run-v1/arena'
CANDIDATE=ROOT/'data/value-training-v1/training/model.json'
RESERVATIONS=ROOT/'data/neural-value-pilot-v2/reservations'
SCHEMA='value-run-arena-v1'
SOURCES=('congklak/value_run_arena.py','congklak/value_run_resources.py','congklak/value_run_cases.py',
         'tools/run-value-arena.py','tests/reference.py','congklak/timing_v2.py')
PROFILES=('control','candidate')
BASELINES=('H_v1','H8_relay_CPU')
SETTINGS=dict(arena_ms=25,max_choices=512,c_puct=.25,cache_capacity=0,max_backups=100000,
              max_nodes=2048,root_noise=None,baseline_depth=16,tt_mb=16,use_pvs=True,
              bootstrap_samples=4096,bootstrap_seed=20261019,whole_family_clusters=24,
              posthoc_filtering=False,default_promotion=False)
need,read,sha=common.require,common.read,common.sha


def path_of(value):
    p=Path(value).resolve()
    need(p==DEFAULT.resolve() or p.is_relative_to(ROOT/'build/value-run-v1'),'Arena namespace differs')
    return p


def atomic(path,body,immutable=False):
    path=Path(path);text=config.canonical(body)+'\n'
    if immutable and path.exists():
        need(path.read_text(encoding='utf-8')==text,'Immutable arena artifact differs: '+str(path));return body
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with temp.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path);return body


def _literal():
    spec=importlib.util.spec_from_file_location('value_run_arena_literal_reference',ROOT/'tests/reference.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def _position(p,reference):
    config.position(p)
    return reference.Position(7,tuple(p['pits'][0]+[p['stores'][0]]+p['pits'][1]+[p['stores'][1]]),p['controller'])


def _raw(p):
    c=p.counts
    return dict(pits=[list(c[:7]),list(c[8:15])],stores=[c[7],c[15]],controller=p.controller)


def normalize(p,reference=None):
    ref=reference or _literal();rp=_position(p,ref);n=ref.normalize(rp)
    return dict(state=dict(p,controller=n.controller),terminal=n.terminal,legal=list(ref.legal_actions(rp)))


def transition(p,action,reference=None):
    ref=reference or _literal();t=ref.apply(_position(p,ref),action)
    return dict(after=_raw(t.position),terminal=t.terminal,deposits=t.deposits,
                relay_pickups=t.relay_pickups,captured=t.captured,bonus=t.bonus,
                passes=[list(v) for v in t.final_passes],actor=t.actor)


def schedule(starts,fixture=False):
    result=[]
    # Every family keeps all eight games together. Rotate all four comparisons
    # and both seat orders without consulting decisions or outcomes.
    comparisons=[(p,b) for p in PROFILES for b in BASELINES]
    for family_index,row in enumerate(starts):
        pairs=comparisons[family_index%4:]+comparisons[:family_index%4]
        if fixture:pairs=[('control','candidate')]
        for profile,baseline in pairs:
            for seat in ((0,1) if family_index%2==0 else (1,0)):
                result.append(dict(index=len(result),family_index=family_index,family=row['family'],
                    component=row['component'],profile=profile,baseline=baseline,seat=seat,start=row['state']))
    need(len(result)==(2 if fixture else 192),'Arena schedule cardinality differs')
    return result


def _engine_sources():
    from . import value_run_cases as cases
    files=list(SOURCES)+list(getattr(cases,'SOURCES',()))
    files+=['congklak/quality_search.py','congklak/native.py','congklak/value_sidecar_v2.py',
            'congklak/puct_cpu_inference.py','congklak/neural_relay.py',
            'build/congklak_native.dll','build/congklak_value_sidecar.dll',
            'build/congklak_neural_relay.dll','build/congklak_puct_cpu.dll',
            'build/native-build.json','build/neural-relay-build.json','build/puct-cpu-build.json',
            'build/value-sidecar-build.json','build/value-training-v1/acceptance.json']
    return common.pins(sorted(set(files)))


def prepare(path=DEFAULT,candidate_path=CANDIDATE,fixture=False):
    path=path_of(path);candidate_path=Path(candidate_path).resolve()
    if (path/'identity.json').exists():return status(path)['identity']
    need(not path.exists() or not any(path.iterdir()),'Existing arena artifacts without identity: HOLD')
    from . import value_training, valuepilot_reservations_v2 as reservations
    reservations.admit()
    production=value_training.completed(CANDIDATE.parent)
    need(production['steps']==400 and production['production_steps']==400 and production['engineering_fixture'] is False,
         'Production candidate is not the fixed final400 export')
    need(candidate_path==CANDIDATE.resolve(),'Arena must use only the fixed final candidate')
    inv=read(RESERVATIONS/'inventory.json');selection=read(RESERVATIONS/'selection.json')
    starts=selection['arena'][:1] if fixture else selection['arena']
    need(len(selection['train'])==512 and len(selection['fit_check'])==128 and len(selection['arena'])==24,'Actual reservation geometry differs')
    cpu=read(ROOT/'build/value-plan.json')['cpu_declared']
    need(cpu['group']==0 and cpu['logical']==2,'Saved CPU core differs')
    files=[RESERVATIONS/n for n in ('identity.json','selection.json','inventory.json','manifest.json','catalog.json','certificate.pending.json')]
    files += [CANDIDATE,CANDIDATE.parent/'certificate.json',ROOT/'data/quality-train-v1/model.json',
              ROOT/'data/training-stage5-deeper-h8-f127-seed2718/model.json']
    files=[p for p in files if p.exists()]
    identity=dict(schema=SCHEMA+'-identity',rules_digest=config.RULES_DIGEST,fixture=fixture,
        settings=SETTINGS,sources=_engine_sources(),inputs=common.pins(files),cpu_declared=cpu,
        candidate_path=str(candidate_path),candidate_sha256=sha(candidate_path),
        reservation_manifest_sha256=sha(RESERVATIONS/'manifest.json'),schedule=schedule(starts,fixture),
        preparation_runtime=dict(executable=sys.executable,version=sys.version),policy_fixed=True,
        training_steps=400,optimizer_steps_here=0,fixture_fixed_simulations=128 if fixture else None)
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        atomic(path/'identity.json',identity,True)
        atomic(path/'head.json',dict(schema=SCHEMA+'-head',identity_sha256=sha(path/'identity.json'),decisions=[],games=[]))
        atomic(path/'initialized.json',dict(schema=SCHEMA+'-anchor',identity_sha256=sha(path/'identity.json'),initialized=True),True)
    return identity


def _identity(path):
    identity=read(path/'identity.json')
    need(identity['schema']==SCHEMA+'-identity' and identity['rules_digest']==config.RULES_DIGEST and identity['settings']==SETTINGS,'Arena identity/settings differ')
    need(identity['sources']==_engine_sources(),'Arena executed source changed')
    common.current(identity['inputs'])
    need(identity['candidate_path']==str(CANDIDATE.resolve()) and identity['candidate_sha256']==sha(CANDIDATE),'Fixed final candidate differs')
    need(read(path/'initialized.json')==dict(schema=SCHEMA+'-anchor',identity_sha256=sha(path/'identity.json'),initialized=True),'Arena initialized highwater missing/changed')
    selection=read(RESERVATIONS/'selection.json')
    need(identity['schedule']==schedule(selection['arena'][:1] if identity['fixture'] else selection['arena'],identity['fixture']),'Arena frozen schedule differs')
    return identity


def _read_record(path,entry,kind):
    need(isinstance(entry,dict) and set(entry)=={'name','sha256'},'Arena prefix record fields differ')
    name=entry['name'];relative=Path(name)
    need(not relative.is_absolute() and relative.parts[0]==kind and len(relative.parts)==2 and relative.suffix=='.json','Invalid arena prefix path')
    record=path/relative
    need(record.is_file() and sha(record)==entry['sha256'],'Arena committed prefix missing/corrupt: '+name)
    return read(record)


def _validate_answer(answer,p,engine,fixture):
    # Full engine validator is run BEFORE commit. The closed reader deliberately
    # avoids its native constructor and verifies the retained recommendation,
    # legal position/clock/work claims plus independently literal transitions.
    need(isinstance(answer,dict) and set(answer)=={'decision','evaluator'},'Unknown arena decision envelope')
    d=answer['decision'];need(isinstance(d,dict),'Malformed retained decision')
    need(d['requested_ms']==(0 if fixture else 25),'Arena decision budget differs')
    need(d['kind']==('puct' if engine in PROFILES else 'h' if engine=='H_v1' else 'relay'),'Arena engine family differs')
    need(type(d['usable_decision']) is bool and (d['chosen'] is None or type(d['chosen']) is int),'Invalid decision recommendation')
    need(d['usable_decision']==(d['chosen'] is not None),'Decision usability/chosen differ')
    need(all(type(d[k]) in (int,float) and math.isfinite(d[k]) and d[k]>=0 for k in ('wall_ms','overshoot_ms','engine_elapsed_ms')),'Invalid arena timing')
    need(abs(d['overshoot_ms']-max(0.,d['wall_ms']-d['requested_ms']))<1e-9 if not fixture else d['overshoot_ms']==0,'Arena whole overshoot differs')
    if engine in PROFILES:
        report=d['report'];need(report['position']==p and report['c_puct']==.25,'Arena PUCT position/coefficient differs')
        need(report['requested_simulations']==(128 if fixture else 100000) and report['max_nodes']==2048,'Arena PUCT search bounds differ')
        need(sum(report['visits'])==report['completed_simulations']==d['completed_simulations'],'Arena backups differ')
        need(report['chosen']==d['chosen'] if d['usable_decision'] else report['completed_simulations']==0,'Arena root choice differs')


def _prefix(path):
    identity=_identity(path);head=read(path/'head.json')
    need(set(head)=={'schema','identity_sha256','decisions','games'} and head['schema']==SCHEMA+'-head' and head['identity_sha256']==sha(path/'identity.json'),'Arena full prefix head differs')
    ref=_literal();games=[];decisions=[];pending=[];p=None;game_index=0;choice_index=0
    scheduled=identity['schedule'];names=[]
    for entry in head['decisions']:
        row=_read_record(path,entry,'decisions');names.append(entry['name'])
        while game_index<len(head['games']) and row['game_index']>game_index:
            # A game commit is valid only after its full decision segment.
            game_entry=head['games'][game_index];game=_read_record(path,game_entry,'games')
            _validate_game(game,scheduled[game_index],pending,p,ref)
            games.append(game);game_index+=1;choice_index=0;pending=[];p=None
        need(game_index<len(scheduled) and row['game_index']==game_index,'Arena decision game cursor differs')
        unit=scheduled[game_index]
        if p is None:p=normalize(unit['start'],ref)['state']
        need(row['schema']==SCHEMA+'-decision' and row['index']==len(decisions) and row['choice_index']==choice_index and row['before']==p,'Arena committed decision state/cursor differs')
        need(row['schedule_sha256']==config.digest(unit),'Arena decision schedule differs')
        normal=normalize(p,ref);need(not normal['terminal'] and p==normal['state'],'Arena decision on nonnormalized/terminal state')
        engine=unit['profile'] if p['controller']==unit['seat'] else unit['baseline']
        need(row['engine']==engine and row['cpu_verified']==dict(group=0,logical=2,mask=4),'Arena engine/core assignment differs')
        _validate_answer(row['answer'],p,engine,identity['fixture'])
        d=row['answer']['decision'];fallback=not d['usable_decision'];action=normal['legal'][0] if fallback else d['chosen']
        need(row['fallback']==fallback and row['action']==action and action in normal['legal'],'Arena fallback/action differs')
        need(row['transition']==transition(p,action,ref),'Arena literal transition/terminal differs')
        p=row['transition']['after'];pending.append(row);decisions.append(row);choice_index+=1
        need(choice_index<=512,'Arena game choice cap exceeded')
    while game_index<len(head['games']):
        game=_read_record(path,head['games'][game_index],'games')
        _validate_game(game,scheduled[game_index],pending,p,ref)
        games.append(game);game_index+=1;choice_index=0;pending=[];p=None
    need(len(names)==len(set(names)) and len(head['games'])<=len(scheduled),'Duplicate prefix filename/too many games')
    need(all(g['index']==i for i,g in enumerate(games)),'Arena game indices differ')
    return dict(identity=identity,head=head,games=games,decisions=decisions,pending=pending,next_state=p,
                completed_games=len(games),committed_decisions=len(decisions),pending_choices=choice_index)


def _validate_game(game,unit,rows,p,ref):
    need(game['schema']==SCHEMA+'-game' and game['index']==unit['index'] and game['schedule_sha256']==config.digest(unit),'Arena game identity differs')
    expected=normalize(unit['start'],ref)['state'] if p is None else p
    normal=normalize(expected,ref);terminal=normal['terminal']
    need(terminal or len(rows)==512,'Arena game ended before terminal/cap')
    margin=expected['stores'][0]-expected['stores'][1]
    margin=margin if unit['seat']==0 else -margin
    need(game['final']==expected and game['terminal']==terminal and game['capped']==(not terminal) and game['choices']==len(rows),'Arena final state/cap differs')
    need(game['model_margin']==(margin if terminal else None),'Arena terminal margin differs')
    need(game['decision_sha256']==[config.digest(r) for r in rows] and game['fallbacks']==sum(r['fallback'] for r in rows),'Arena game decision closure differs')


def status(path=DEFAULT):
    path=path_of(path)
    if (path/'certificate.json').exists() or (path/'COMPLETED.json').exists():return verify(path)
    prefix=_prefix(path)
    return dict(schema=SCHEMA+'-status',identity=prefix['identity'],completed=False,
        completed_games=prefix['completed_games'],committed_decisions=prefix['committed_decisions'],pending_choices=prefix['pending_choices'],
        scheduled_games=len(prefix['identity']['schedule']),head_sha256=sha(path/'head.json'),next_operation='resume from the complete committed decision prefix')


def ownership(games,decisions,identity):
    inv=read(RESERVATIONS/'inventory.json');selection=read(RESERVATIONS/'selection.json')
    q_roles=defaultdict(set);f_roles=defaultdict(set);ancestral=defaultdict(set)
    for cid,component in inv['components'].items():
        for q in component['quotients']:ancestral[q].add(cid)
    for role in ('train','fit_check'):
        for row in selection[role]:
            c=inv['components'][row['component']]
            for q in c['quotients']:q_roles[q].add(role)
            for f in c['families']:f_roles[f].add(role)
    arena_components={u['family_index']:u['component'] for u in identity['schedule']}
    quotient_owners=defaultdict(set);component_owners=defaultdict(set);observations=[];role_hits=[];old_hits=[]
    for family_index,cid in arena_components.items():component_owners[cid].add(family_index)
    by_game=defaultdict(list)
    for d in decisions:by_game[d['game_index']].append(d)
    for game in games:
        u=identity['schedule'][game['index']];family_index=u['family_index'];cid=u['component']
        states=[u['start']]+[d['transition']['after'] for d in by_game[game['index']]]
        for choice,p in enumerate(states):
            if sum(map(sum,p['pits']))<=4:continue
            q=metadata.quotient(p);owners=set(ancestral.get(q,()))
            row=dict(game=game['index'],choice=choice,family_index=family_index,component=cid,quotient=q)
            observations.append(row);quotient_owners[q].add(family_index)
            for owner_component in owners:component_owners[owner_component].add(family_index)
            roles=q_roles.get(q,set())|f_roles.get(u['family'],set())
            if roles:role_hits.append(dict(row,roles=sorted(roles)))
            for other_index,other_component in arena_components.items():
                if other_index!=family_index and other_component in owners:quotient_owners[q].add(other_index)
            foreign=owners-{cid}
            if foreign:old_hits.append(dict(row,ancestral_components=sorted(foreign)))
    cross=[dict(quotient=q,family_indices=sorted(v)) for q,v in sorted(quotient_owners.items()) if len(v)>1]
    alias_cross=[dict(historical_component=cid,family_indices=sorted(v)) for cid,v in sorted(component_owners.items()) if len(v)>1]
    return dict(checked=True,eligible_predicate='active seeds >4, full origin family/quotient aliases',
        frozen_support_roots=512,frozen_fit_roots=128,all_support_counted_even_if_never_drawn=True,
        observations=len(observations),distinct_observed_quotients=len(quotient_owners),
        support_or_fit_hits=role_hits,cross_arena_component_hits=cross,cross_arena_alias_component_hits=alias_cross,
        broader_ancestral_hits=old_hits,broader_ancestral_hits_not_automatic_gradient_pool=True,
        ownership_passed=not role_hits and not cross and not alias_cross,posthoc_filtering=False,posthoc_cluster_merging=False)


def _interval(values,rng,samples):
    n=len(values);means=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(samples))
    def quantile(q):
        at=(samples-1)*q;lo=math.floor(at);hi=math.ceil(at)
        return means[lo]+(means[hi]-means[lo])*(at-lo)
    return [quantile(.025),quantile(.975)]


def summarize(prefix):
    identity=prefix['identity'];games=prefix['games'];decisions=prefix['decisions'];fixture=identity['fixture']
    all_terminal=all(g['terminal'] for g in games);fallbacks=sum(g['fallbacks'] for g in games)
    overlap=ownership(games,decisions,identity)
    valid=not fixture and len(games)==192 and all_terminal and fallbacks==0 and overlap['ownership_passed']
    statistics={};family_rows=[]
    if valid:
        by=defaultdict(dict)
        for g in games:
            u=identity['schedule'][g['index']];by[u['family_index']][(u['profile'],u['baseline'],u['seat'])]=g['model_margin']
        for family_index in range(24):
            values=by[family_index];r=dict(family_index=family_index,family=identity['schedule'][family_index*8]['family'])
            for profile in PROFILES:
                for baseline in BASELINES:r[profile+'/'+baseline]=sum(values[(profile,baseline,s)] for s in (0,1))/2
            for baseline in BASELINES:r['candidate-minus-control/'+baseline]=r['candidate/'+baseline]-r['control/'+baseline]
            family_rows.append(r)
        keys=[p+'/'+b for p in PROFILES for b in BASELINES]+['candidate-minus-control/'+b for b in BASELINES]
        # One bootstrap draws24 WHOLE family indices and applies those same
        # sampled clusters jointly to all six estimates, retaining all8games.
        rng=random.Random(20261019);sampled={k:[] for k in keys}
        for _ in range(4096):
            ids=[rng.randrange(24) for _ in range(24)]
            for k in keys:sampled[k].append(sum(family_rows[i][k] for i in ids)/24)
        for k in keys:
            values=sorted(sampled[k]);a=.025*4095;b=.975*4095
            q=lambda x:values[math.floor(x)]+(values[math.ceil(x)]-values[math.floor(x)])*(x-math.floor(x))
            statistics[k]=dict(mean=sum(r[k] for r in family_rows)/24,interval95=[q(a),q(b)],clusters=24,games=192,
                resamples=4096,seed=20261019,exploratory=True,interpretation_allowed=valid)
    work=Counter();overshoot=0.;per=Counter()
    for d in decisions:
        answer=d['answer']['decision'];overshoot+=answer['overshoot_ms'];per[d['engine']]+=1
        if d['engine'] in PROFILES:
            for k,v in answer['report']['work'].items():
                if type(v) is int:work[k]+=v
    return dict(schema=SCHEMA+'-result',completed=len(games)==len(identity['schedule']),fixture=fixture,
        games=len(games),scheduled_games=len(identity['schedule']),decisions=len(decisions),terminal_games=sum(g['terminal'] for g in games),
        capped_games=sum(g['capped'] for g in games),fallbacks=fallbacks,per_engine_decisions=dict(per),
        per_baseline_seats=dict(Counter(u['profile']+'/'+u['baseline']+'/seat'+str(u['seat']) for u in identity['schedule'][:len(games)])),
        whole_decision_overshoot_ms=overshoot,puct_work=dict(work),ownership=overlap,
        cluster_statistics=statistics,family_cluster_rows=family_rows,cluster_analysis_valid=valid,
        nonnegative_paired_means_both_baselines=valid and all(statistics['candidate-minus-control/'+b]['mean']>=0 for b in BASELINES),
        positive_paired_lower_bounds_both_baselines=valid and all(statistics['candidate-minus-control/'+b]['interval95'][0]>0 for b in BASELINES),
        positive_candidate_absolute_lower_bounds_both_baselines=valid and all(statistics['candidate/'+b]['interval95'][0]>0 for b in BASELINES),
        hard_stop_reasons=(['ownership bridge discovered'] if not overlap['ownership_passed'] else [])+(['fallbacks retained'] if fallbacks else [])+(['game choice cap reached'] if any(g['capped'] for g in games) else []),
        policy_fixed=True,training_steps=400,optimizer_steps_here=0,default_promotion=False,
        source_certificate_sha256=config.digest(identity),head_sha256=sha(prefix['path']/'head.json') if 'path' in prefix else None)


def verify(path=DEFAULT):
    path=path_of(path);prefix=_prefix(path);certificate=read(path/'certificate.json')
    need(certificate['schema']==SCHEMA+'-certificate' and certificate['completed'] is True,'Arena completed certificate missing/invalid')
    if (path/'COMPLETED.json').exists():
        need(read(path/'COMPLETED.json')==dict(certificate_sha256=sha(path/'certificate.json')),'Arena completion anchor differs')
    files=[path/'identity.json',path/'initialized.json',path/'head.json']+[path/e['name'] for k in ('decisions','games') for e in prefix['head'][k]]
    need(certificate['files']==common.pins(files),'Arena certificate does not bind complete committed prefix')
    common.current(certificate['files'])
    need(prefix['completed_games']==len(prefix['identity']['schedule']) and not prefix['pending'],'Arena completion prefix is partial')
    prefix['path']=path;result=summarize(prefix)
    need(certificate['result_sha256']==config.digest(result),'Arena derived result differs from certified evidence')
    result_path=path/'result.json'
    if result_path.exists():need(read(result_path)==result,'Arena derived report corrupt')
    return result


def _pause(path,point,requested,marker):
    if point!=requested:return
    need(path!=DEFAULT.resolve() and marker is not None,'Production hook forbidden')
    atomic(Path(marker),dict(schema=SCHEMA+'-pause',point=point,pid=os.getpid(),path=str(path)),True)
    while True:time.sleep(.1)


def _commit(path,prefix,kind,body,pause_at=None,pause_marker=None):
    folder=path/kind;index=len(prefix['head'][kind]);name=kind+'/%06d-%s.json'%(index,uuid.uuid4().hex)
    atomic(path/name,body,True)
    singular='decision' if kind=='decisions' else 'game'
    _pause(path,'before_'+singular+'_commit',pause_at,pause_marker)
    head=dict(prefix['head']);head[kind]=head[kind]+[dict(name=name,sha256=sha(path/name))]
    atomic(path/'head.json',head);prefix['head']=head
    _pause(path,'after_'+singular+'_commit',pause_at,pause_marker)


def progress(path,prefix,stop_reason=None):
    games=prefix['head']['games'];decisions=prefix['head']['decisions'];identity=prefix['identity']
    resource_path=ROOT/'data/value-run-v1/arena-resources'
    resource=None
    if (resource_path/'identity.json').exists():
        from . import value_run_resources as resources
        ledger=resources.status(resource_path)
        resource=dict(charged_ns=ledger['charged_ns'],remaining_ns=ledger['remaining_ns'],pending=ledger['pending'] is not None)
    body=dict(schema=SCHEMA+'-progress',games_committed=len(games),games_planned=len(identity['schedule']),
        choices_committed=len(decisions),pending_game=len(games) if len(games)<len(identity['schedule']) else None,
        pending_choices=len(prefix.get('pending',[])),
        per_baseline_seats=dict(Counter(u['profile']+'/'+u['baseline']+'/seat'+str(u['seat']) for u in identity['schedule'][:len(games)])),
        resources=resource,stop_reason=stop_reason,next_operation='verify completed arena' if len(games)==len(identity['schedule']) else 'resume next uncommitted decision',
        policy_fixed=True,optimizer_steps_here=0,fixture=identity['fixture'],head_sha256=sha(Path(path)/'head.json'))
    atomic(Path(path)/'progress.json',body)
    return body


def run(path=DEFAULT,candidate_path=CANDIDATE,fixture=False,fixed_simulations=None,
        stop_after_decisions=None,pause_at=None,pause_marker=None,lease=None):
    path=path_of(path)
    # No completed path reaches resources, locks, DLLs, encoders or search.
    if (path/'certificate.json').exists() or (path/'COMPLETED.json').exists():return verify(path)
    if not (path/'identity.json').exists():prepare(path,candidate_path,fixture)
    prefix=_prefix(path);identity=prefix['identity']
    need(identity['fixture']==fixture and (fixed_simulations==128 if fixture else fixed_simulations is None),'Arena fixture settings differ')
    need(fixture or lease is not None,'Production arena requires independent resource lease')
    need(pause_at in (None,'before_decision_commit','after_decision_commit','before_game_commit','after_game_commit'),'Unknown arena pause point')
    need(not pause_at or fixture,'Production pause hook forbidden')
    from . import value_run_cases as cases, value_run_resources as resources
    from .timing_v2 import PerformanceCore
    if lease:resources.assert_worker_lease(lease)
    ref=_literal();added=0
    with Writer(path),ExitStack() as stack:
        # Prefix was verified before the exclusive writer; verify it once more
        # to reject concurrent metadata changes before making any engine calls.
        prefix=_prefix(path);identity=prefix['identity']
        cpu=stack.enter_context(PerformanceCore(identity['cpu_declared']))
        engines=cases.open_engines(stack,Path(identity['candidate_path']))
        for game_index in range(len(prefix['games']),len(identity['schedule'])):
            unit=identity['schedule'][game_index];rows=prefix['pending'] if game_index==len(prefix['games']) else []
            p=prefix['next_state'] if rows else normalize(unit['start'],ref)['state']
            while not normalize(p,ref)['terminal'] and len(rows)<512:
                if stop_after_decisions is not None and added>=stop_after_decisions:
                    prefix['pending']=rows;return progress(path,prefix,'cooperative decision limit')
                if lease:
                    try:resources.assert_worker_lease(lease)
                    except ValueError as exc:
                        if 'must checkpoint and stop' in str(exc):
                            prefix['pending']=rows;return progress(path,prefix,'lease boundary; raw prefix preserved')
                        raise
                normal=normalize(p,ref);name=unit['profile'] if p['controller']==unit['seat'] else unit['baseline']
                cpu_verified=cpu.verify()
                answer=cases.fixed_decision(name,engines[name],p,128) if fixture else cases.timed_decision(name,engines[name],p,25,cpu)
                fallback=not answer['decision']['usable_decision'];action=normal['legal'][0] if fallback else answer['decision']['chosen']
                need(action in normal['legal'],'Arena recommended illegal action')
                row=dict(schema=SCHEMA+'-decision',index=len(prefix['head']['decisions']),game_index=game_index,choice_index=len(rows),
                    schedule_sha256=config.digest(unit),engine=name,before=p,answer=answer,fallback=fallback,action=action,
                    transition=transition(p,action,ref),cpu_verified=cpu_verified)
                _validate_answer(answer,p,name,fixture)
                _commit(path,prefix,'decisions',row,pause_at,pause_marker)
                rows=rows+[row];p=row['transition']['after'];added+=1
                prefix['pending']=rows;progress(path,prefix)
            normal=normalize(p,ref);margin=p['stores'][0]-p['stores'][1];margin=margin if unit['seat']==0 else -margin
            game=dict(schema=SCHEMA+'-game',index=game_index,schedule_sha256=config.digest(unit),choices=len(rows),
                final=p,terminal=normal['terminal'],capped=not normal['terminal'],model_margin=margin if normal['terminal'] else None,
                fallbacks=sum(r['fallback'] for r in rows),decision_sha256=[config.digest(r) for r in rows])
            _validate_game(game,unit,rows,p,ref);_commit(path,prefix,'games',game,pause_at,pause_marker)
            # Local cursors avoid re-reading/replaying the already validated raw
            # prefix for every next decision. The final close rechecks all facts.
            prefix['games'].append(game);prefix['pending']=[];prefix['next_state']=None;progress(path,prefix)
        prefix=_prefix(path);prefix['path']=path;result=summarize(prefix)
        files=[path/'identity.json',path/'initialized.json',path/'head.json']+[path/e['name'] for k in ('decisions','games') for e in prefix['head'][k]]
        atomic(path/'result.json',result,True)
        atomic(path/'certificate.json',dict(schema=SCHEMA+'-certificate',completed=True,files=common.pins(files),result_sha256=config.digest(result)),True)
        atomic(path/'COMPLETED.json',dict(certificate_sha256=sha(path/'certificate.json')),True)
    return verify(path)
