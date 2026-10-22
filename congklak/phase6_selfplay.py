"""Decision-transaction self-play, terminal-only replay admission and bounded pruning."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
import time
from . import config,pilot,features_relay
from . import phase6_common as common
from .native import Native,flatten
from .puct import forest,PROTOCOL
from .phase6_inference import BatchEvaluator
from .phase6_store import Work

SCHEMA='phase6-decision-transaction-terminal-replay-v1'
require=common.require

def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value

def rng_restore(value):
    r=random.Random();r.setstate(tuples(value));return r

def event(value,callback=None):
    if callback:callback(value)
    target=os.environ.get('PHASE6_BLOCK_EVENT');after=int(os.environ.get('PHASE6_BLOCK_AFTER','1'))
    if value['event']==target and value.get('committed_decisions',0)>=after:
        print(config.canonical(dict(event='phase6_test_barrier',blocked=value)),flush=True)
        while True:time.sleep(.1)

def unpack(body,digest):
    value=json.loads(body);require(config.digest(value)==digest,'Corrupt Phase6 game/decision hash');return value

def read_games(db):
    games=[]
    for gid,body,digest in db.execute('SELECT id,body,hash FROM sp_games ORDER BY id'):
        game=unpack(body,digest);require(game['id']==gid,'Game table/body ID differs');games.append(game)
    return games

def validate_search(report,p,requested,max_nodes=2048,native=None):
    """Check the typed statistical result, rather than trusting a body checksum."""
    native=native or Native();n=native.normalize(flatten(p),actor=p['controller'],total=98);legal=n['legal_actions'];terminal=n['terminal']
    numeric=lambda x:type(x) in (int,float) and math.isfinite(x)
    integer=lambda x:type(x) is int and x>=0
    require(report['schema']=='phase6-puct-report-v1' and report['protocol']==PROTOCOL and report['rules_digest']==config.RULES_DIGEST,'Search protocol differs')
    require(report['position']==p and report['value_perspective']=='fixed_player0_seeds' and report['first_play_urgency']=='parent_neural_bootstrap','Search perspective/FPU differs')
    require(report['requested_simulations']==requested and report['max_nodes']==max_nodes and report['memory_mb']==8192,'Search limits differ')
    visits=report['visits'];priors=report['root_priors'];policy=report['policy'];means=report['action_mean_value_player0'];completed=report['completed_simulations']
    require(all(isinstance(v,list) and len(v)==7 for v in (visits,priors,policy,means)) and all(integer(v) for v in visits),'Malformed root arrays')
    require(integer(completed) and completed==(0 if terminal else requested) and sum(visits)==completed,'Incomplete committed search')
    require(type(report['nodes']) is int and 1<=report['nodes']<=max_nodes and report['nodes']<=completed+1,'Invalid tree bound')
    require(all(numeric(v) and v>=0 and (a in legal or v==0) for seq in (priors,policy) for a,v in enumerate(seq)),'Invalid legal priors/policy')
    require(all(a in legal or v==0 for a,v in enumerate(visits)),'Illegal edge visits')
    d=p['stores'][0]-p['stores'][1];r=sum(map(sum,p['pits']))
    require(numeric(report['estimated_value_player0']) and d-r-1e-4<=report['estimated_value_player0']<=d+r+1e-4,'Estimate violates physical bounds')
    require(numeric(report['unvisited_root_q_player0']) and d-r-1e-4<=report['unvisited_root_q_player0']<=d+r+1e-4,'FPU violates physical bounds')
    require(report['c_puct']==1.5 and type(report['usable_decision']) is bool and report['usable_decision'],'Unusable completed result')
    if terminal:
        require(report['chosen'] is None and report['proof_status']=='terminal_outcome' and report['stop_reason']=='terminal' and report['estimated_value_player0']==d and not report['root_expanded'],'Fabricated terminal result')
        require(not any(priors+policy) and all(m is None for m in means),'Terminal policy should be empty')
    else:
        require(report['proof_status']=='monte_carlo_estimate' and report['stop_reason']=='simulations' and report['root_expanded'] is True,'Nonterminal proof/stop differs')
        require(abs(sum(priors)-1)<=1e-12 and policy==[v/completed for v in visits],'Invalid root probability mass')
        require(report['chosen']==max(legal,key=lambda a:(visits[a],-a)),'Choice differs from completed visits')
        require(all((m is None if visits[a]==0 else numeric(m) and d-r-1e-4<=m<=d+r+1e-4) for a,m in enumerate(means)),'Invalid edge means')
        expected=sum(visits[a]*means[a] for a in legal if visits[a])/completed
        require(abs(expected-report['estimated_value_player0'])<=1e-10,'Root estimate differs from edge means')
    work=report['work'];required=('transitions','deposits','feature_simulations','feature_deposits','neural_evaluations','neural_calls','root_expansions','total_deposits')
    require(all(integer(work[k]) for k in required) and work['transitions']==report['nodes']-1 and work['total_deposits']==work['deposits']+work['feature_deposits'],'Invalid work counters')
    require(work['root_expansions']==(0 if terminal else 1) and work['neural_evaluations']<=report['nodes'] and work['neural_calls']==work['neural_evaluations'],'Invalid neural work')
    require(numeric(work['inference_ms']) and work['inference_ms']>=0,'Invalid inference time')
    timing=report['timing'];require(timing['requested_time_ms']==0 and timing['whole_decision_includes_inference'] is True and numeric(timing['elapsed_ms']) and timing['elapsed_ms']>=0 and timing['overshoot_ms']==0,'Fixed-simulation timing differs')
    require(type(report['forest_size']) is int and 1<=report['forest_size']<=32 and integer(report['forest_batch_calls']),'Invalid batch accounting')
    return report

def validate_game(db,game,native,settings):
    p=common.normalize(game['start'],native);r=random.Random(game['seed']);count=0
    for index,body,digest in db.execute('SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):
        row=unpack(body,digest);require(index==count and row['gameid']==game['id'] and row['choiceindex']==count and row['state']==p,'Noncontiguous selfplay replay')
        require(tuples(row['rng_before'])==r.getstate(),'Selfplay RNG chain differs')
        n=native.normalize(flatten(p),actor=p['controller'],total=98);legal=n['legal_actions'];report=validate_search(row['search'],p,settings['simulations'],settings['max_nodes'],native)
        require(report['position']==p and report['completed_simulations']==settings['simulations'] and sum(report['visits'])==settings['simulations'] and report['usable_decision'],'Incomplete search committed')
        require(all(type(v) is int and v>=0 and (a in legal or v==0) for a,v in enumerate(report['visits'])),'Invalid legal visits')
        expected=[v/settings['simulations'] for v in report['visits']]
        require(report['policy']==expected and row['policy_target']==expected and row['action'] in legal,'Invalid committed selfplay policy/action')
        if settings['root_noise'] is not None:
            for _ in legal:r.gammavariate(settings['root_noise']['alpha'],1)
        if count<settings['action_temperature_first_choices']:
            draw=r.random();running=0;expected_action=legal[-1]
            for a in legal:
                running+=expected[a]
                if draw<running:expected_action=a;break
        else:expected_action=max(legal,key=lambda a:(report['visits'][a],-a))
        require(row['action']==expected_action and tuples(row['rng_after'])==r.getstate(),'Action/RNG violates declared sampling')
        t=native.apply(flatten(p),actor=p['controller'],action=row['action'],total=98,trace=False);require(t.get('ok'),'Illegal selfplay transition')
        p=common.normalize(t['position'],native);require(row['after']==p,'Selfplay saved transition differs');count+=1
    require(count==game['choices'] and game['position']==p and tuples(game['rng'])==r.getstate(),'Selfplay checkpoint differs from decision chain')
    terminal=sum(map(sum,p['pits']))==0
    require(game['status']==('terminal' if terminal else 'capped' if count>=settings['max_choices'] else 'active'),'Fabricated game completion')
    require(game['terminal_margin']==(p['stores'][0]-p['stores'][1] if terminal else None),'Fabricated terminal outcome')
    return p

def initialize(work,starts,settings,native):
    work.db.executescript('''CREATE TABLE IF NOT EXISTS sp_games(id TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS sp_decisions(gameid TEXT NOT NULL,choice INTEGER NOT NULL,body TEXT NOT NULL,hash TEXT NOT NULL,PRIMARY KEY(gameid,choice));
      CREATE TABLE IF NOT EXISTS sp_rows(id TEXT PRIMARY KEY,body TEXT NOT NULL,hash TEXT NOT NULL);''')
    expected=[]
    for i,start in enumerate(starts):
        for run in range(settings['runs_per_family']):
            gid=f'{i:03d}/{run}';seed=settings['seed']+i*settings['runs_per_family']+run;p=common.normalize(start['state'],native)
            expected.append(gid)
            row=work.db.execute('SELECT body,hash FROM sp_games WHERE id=?',(gid,)).fetchone()
            if row is None:
                game=dict(id=gid,family=start['family'],start=p,position=p,seed=seed,rng=random.Random(seed).getstate(),choices=0,status='active',terminal_margin=None)
                require(sum(map(sum,p['pits']))>4,'Selfplay start must be nonterminal active>4')
                with work.db:work.db.execute('INSERT INTO sp_games VALUES(?,?,?)',(gid,config.canonical(game),config.digest(game)))
            else:
                game=unpack(*row);require(game['id']==gid and game['family']==start['family'] and game['start']==p and game['seed']==seed,'Game source differs');validate_game(work.db,game,native,settings)
    require([r[0] for r in work.db.execute('SELECT id FROM sp_games ORDER BY id')]==expected,'Unexpected selfplay game IDs')

def build_rows(db,identity,registry,native):
    games=read_games(db);settings=identity['settings'];protected=set(registry['quotients']);protected_families=set(registry['families'])
    exclusions={};byfamily=defaultdict(list);allrows=[];semantic=[];observed=0;lowactive=0
    expected={f'{i:03d}/{run}':(start,settings['seed']+i*settings['runs_per_family']+run)
              for i,start in enumerate(identity['starts']) for run in range(settings['runs_per_family'])}
    require({g['id'] for g in games}==set(expected) and len(games)==settings['games'],'Game catalog differs from frozen starts')
    for game in games:
        start,seed=expected[game['id']];require(game['family']==start['family'] and game['start']==common.normalize(start['state'],native) and game['seed']==seed,'Game source ownership differs')
    require(db.execute('SELECT count(*) FROM sp_decisions').fetchone()[0]==sum(g['choices'] for g in games),'Orphan selfplay decisions')
    require(not db.execute('SELECT 1 FROM sp_decisions d LEFT JOIN sp_games g ON g.id=d.gameid WHERE g.id IS NULL LIMIT 1').fetchone(),'Unknown game in decision ledger')
    for game in games:
        validate_game(db,game,native,settings);byfamily[game['family']].append(game)
        if game['status']!='terminal':exclusions[game['family']]='incomplete_or_capped_family'
        if game['family'] in protected_families:exclusions[game['family']]='protected_family'
        states=[game['start']]
        for body,digest in db.execute('SELECT body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):
            row=unpack(body,digest);semantic.append([game['id'],row['choiceindex'],digest]);states.append(row['after']);observed+=1
        if any(sum(map(sum,p['pits']))>4 and pilot.quotient(p).hex() in protected for p in states):exclusions[game['family']]='protected_active_quotient_bridge'
    for family,familygames in byfamily.items():
        require(len(familygames)==settings['runs_per_family'],'Missing seeded family game')
        if family in exclusions:continue
        for game in familygames:
            for body,digest in db.execute('SELECT body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(game['id'],)):
                row=unpack(body,digest);p=row['state']
                if sum(map(sum,p['pits']))<=4:lowactive+=1;continue
                x,d,r=features_relay.encode(p,127);stratum=pilot._mechanism(p,native)[1]
                value=dict(id=f"{game['id']}/{row['choiceindex']:04d}",labelkind='selfplay_terminal_seed_margin',value_target_player0=game['terminal_margin'],
                           policy_target=row['policy_target'],gameid=game['id'],state=p,stratum=stratum,family=family,choiceindex=row['choiceindex'],
                           cached_features=dict(x=x,d=d,r=r),source_decision_sha256=digest,weight=1,split='train')
                allrows.append(value)
    grouped=defaultdict(list)
    for row in allrows:grouped[common.rawkey(row['state'])].append(row)
    retained=[];pruned=[]
    for key,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:config.digest({'namespace':'phase6-rawstate-cap-v1','id':r['id']}))
        retained.extend(rows[:settings['rawstate_observation_cap']]);pruned.extend(r['id'] for r in rows[settings['rawstate_observation_cap']:])
    retained.sort(key=lambda r:r['id'])
    summary=dict(games=len(games),terminal_games=sum(g['status']=='terminal' for g in games),capped_games=sum(g['status']=='capped' for g in games),
                 family_exclusions=exclusions,excluded_families=len(exclusions),admitted_families=len(byfamily)-len(exclusions),observed_decisions=observed,
                 low_active_rows_omitted=lowactive,eligible_rows_before_cap=len(allrows),retained_rows=len(retained),rawstate_unique=len(grouped),
                 capped_duplicate_rows=len(pruned),pruned_row_ids=sorted(pruned),strata=dict(sorted(Counter(r['stratum'] for r in retained).items())),
                 retained_rows_by_game=dict(sorted(Counter(r['gameid'] for r in retained).items())),unit_row_weights=True,targets_averaged=False)
    ledger=dict(games=[[g['id'],config.digest(g)] for g in games],decisions=semantic)
    manifest=dict(schema=SCHEMA,identity=identity,rules_digest=config.RULES_DIGEST,registry_sha256=registry['sha256'],summary=summary,
                  records_semantic_sha256=config.digest([[r['id'],config.digest(r)] for r in retained]),game_ledger_semantic_sha256=config.digest(ledger),
                  training_admission='ready' if retained else 'failed_no_admitted_rows',complete=True,sealed_target_bodies_read=False,sealed_final_test_consumed=False)
    return manifest,retained

def run(path=common.SELFPLAY,modelpath=common.WARM,backend='cuda',plan=None,registry=None,on_event=None,cancel=None,decision_limit=None):
    plan=plan or common.read_plan();registry=registry or common.read_registry();settings=plan['selfplay'];starts=plan['selfplay_starts'];native=Native()
    with BatchEvaluator(modelpath,backend,max_batch=settings['parallel_roots']) as evaluator:
        identity=dict(schema=SCHEMA,settings=settings,starts=starts,plan_sha256=config.digest(plan),registry_sha256=registry['sha256'],rules_digest=config.RULES_DIGEST,
                      source_sha256=common.sha(__file__),puct_sha256=common.sha(Path(__file__).with_name('puct.py')),common_sha256=common.sha(common.__file__),
                      evaluator=evaluator.identity,model_file_sha256=evaluator.model.file_sha256,model_weights_sha256=evaluator.model.id,store_sha256=common.sha(Path(__file__).with_name('phase6_store.py')))
        work=Work(path,identity)
        try:
            initialize(work,starts,settings,native)
            done=work.get('selfplay_manifest')
            if done is not None:
                manifest,rows=build_rows(work.db,identity,registry,native);require(manifest==done,'Completed selfplay ledger differs');common.publish(Path(path)/'manifest.json',manifest);return manifest
            committed=work.db.execute('SELECT count(*) FROM sp_decisions').fetchone()[0];new_decisions=0
            while True:
                active=[g for g in read_games(work.db) if g['status']=='active'][:settings['parallel_roots']]
                if not active:break
                if (cancel and cancel()) or (decision_limit is not None and new_decisions>=decision_limit):return dict(status='interrupted',committed_decisions=committed)
                rngs=[rng_restore(g['rng']) for g in active];before=[r.getstate() for r in rngs]
                results=forest([g['position'] for g in active],evaluator,simulations=settings['simulations'],max_nodes=settings['max_nodes'],rng=rngs,
                               root_noise=settings['root_noise'],max_trees=settings['parallel_roots'],cancel=cancel,native=native)
                if any(not q['usable_decision'] or q['completed_simulations']!=settings['simulations'] for q in results):return dict(status='interrupted',committed_decisions=committed,reason='unfinished_batch')
                decisions=[];updated=[]
                for game,rng,state,report in zip(active,rngs,before,results):
                    p=game['position'];legal=[a for a,v in enumerate(p['pits'][p['controller']]) if v]
                    if game['choices']<settings['action_temperature_first_choices']:
                        draw=rng.random();running=0;action=legal[-1]
                        for a in legal:
                            running+=report['policy'][a]
                            if draw<running:action=a;break
                    else:action=max(legal,key=lambda a:(report['visits'][a],-a))
                    t=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=False);require(t.get('ok'),'Illegal generated selfplay action');after=common.normalize(t['position'],native)
                    row=dict(gameid=game['id'],choiceindex=game['choices'],state=p,action=action,after=after,rng_before=state,rng_after=rng.getstate(),policy_target=report['policy'],search=report)
                    game={**game,'position':after,'rng':rng.getstate(),'choices':game['choices']+1}
                    terminal=sum(map(sum,after['pits']))==0;game['status']='terminal' if terminal else 'capped' if game['choices']>=settings['max_choices'] else 'active'
                    game['terminal_margin']=after['stores'][0]-after['stores'][1] if terminal else None;decisions.append(row);updated.append(game)
                event(dict(event='phase6_selfplay_precommit',committed_decisions=committed),on_event)
                work.admit(sum(len(config.canonical(r)) for r in decisions+updated)*3)
                with work.db:
                    for row,game in zip(decisions,updated):
                        work.db.execute('INSERT INTO sp_decisions VALUES(?,?,?,?)',(row['gameid'],row['choiceindex'],config.canonical(row),config.digest(row)))
                        work.db.execute('UPDATE sp_games SET body=?,hash=? WHERE id=?',(config.canonical(game),config.digest(game),game['id']))
                    committed+=len(decisions);new_decisions+=len(decisions)
                    work.put('progress',dict(stage='selfplay',committed_decisions=committed,terminal_games=sum(g['status']=='terminal' for g in read_games(work.db)),games=len(starts)*settings['runs_per_family']))
                event(dict(event='phase6_selfplay_committed',committed_decisions=committed,terminal_games=work.get('progress')['terminal_games']),on_event)
            manifest,rows=build_rows(work.db,identity,registry,native);work.admit(sum(len(config.canonical(r)) for r in rows)*3)
            event(dict(event='phase6_replay_precommit',committed_decisions=committed,rows=len(rows)),on_event)
            with work.db:
                require(work.db.execute('SELECT count(*) FROM sp_rows').fetchone()[0]==0,'Orphan replay rows')
                for row in rows:work.db.execute('INSERT INTO sp_rows VALUES(?,?,?)',(row['id'],config.canonical(row),config.digest(row)))
                work.put('selfplay_manifest',manifest);work.put('progress',dict(stage='complete',committed_decisions=committed,terminal_games=manifest['summary']['terminal_games'],games=manifest['summary']['games']))
            event(dict(event='phase6_replay_committed',committed_decisions=committed,rows=len(rows)),on_event)
            common.publish(Path(path)/'manifest.json',manifest);return manifest
        finally:work.close()

def load_training_rows(path=common.SELFPLAY):
    path=Path(path).resolve();manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'));require(manifest['schema']==SCHEMA and manifest['complete'],'Incomplete selfplay catalog')
    identity=manifest['identity'];require(identity['source_sha256']==common.sha(__file__) and identity['rules_digest']==config.RULES_DIGEST,'Generator source/rules differs')
    registry=common.read_registry();require(registry['sha256']==identity['registry_sha256'],'Protected registry differs')
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN');require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corrupt selfplay database')
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone();require(unpack(body,digest)==identity,'Selfplay work identity differs')
        body,digest=db.execute("SELECT body,hash FROM meta WHERE key='selfplay_manifest'").fetchone();require(unpack(body,digest)==manifest,'Uncommitted published replay manifest')
        observed,expected=build_rows(db,identity,registry,Native());require(observed==manifest,'Selfplay provenance/admission differs')
        rows=[unpack(*r) for r in db.execute('SELECT body,hash FROM sp_rows ORDER BY id')];require(rows==expected,'Published replay rows differ')
    return manifest,rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=('generate','verify'));p.add_argument('--path',default=str(common.SELFPLAY));p.add_argument('--model',default=str(common.WARM));p.add_argument('--backend',choices=('native','cuda'),default='cuda');p.add_argument('--decision-limit',type=int)
    a=p.parse_args();stopped=False
    def stop(*_):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGINT,stop)
    if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,stop)
    output=lambda v:print(config.canonical(v),flush=True)
    if a.operation=='generate':result=run(a.path,a.model,a.backend,on_event=output,cancel=lambda:stopped,decision_limit=a.decision_limit)
    else:
        manifest,rows=load_training_rows(a.path);result=dict(status='verified',rows=len(rows),manifest_sha256=config.digest(manifest))
    output(dict(event='phase6_selfplay_done',result=result))

if __name__=='__main__':main()
