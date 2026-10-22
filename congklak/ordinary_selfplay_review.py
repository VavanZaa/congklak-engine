"""Read-only ordinary coverage and matched-trajectory review of the sealed pilot.

This module never generates games, teacher targets, or optimizer draws.  Ordinary
means the existing simulator's mechanism priority: no legal action causes a pass,
capture, or own-store finish.  A start's mechanism does not describe later states.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

from . import config, pilot, selfplay_pilot_benchmark as bench
from .native import Native, ROOT, flatten
from .deeper_teacher_v2 import atomic_json

BASELINE=ROOT/'build/ordinary-selfplay-baseline.json'
ARENA_OUTPUT=ROOT/'build/ordinary-selfplay-prior-arena-review.json'
CLI=ROOT/'tools/review-ordinary-selfplay-pilot.py'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def raw_key(state):return bytes(flatten(state)+[state['controller']]).hex()
def relative(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def baseline():
    body=read(BASELINE)
    require(body['schema']=='ordinary-selfplay-prior-immutability-v1' and body['passed'] is True and len(body['inputs'])==body['inputs_count']==3171,'Wrong ordinary review prior coverage')
    for name,digest in body['inputs'].items():require(sha(ROOT/name)==digest,'Prior input changed: '+name)
    return body


def label_index(plan):
    """Index *literal full states*, never canonical/player/store-offset aliases."""
    result=defaultdict(list);sources=[]
    dataset=ROOT/'build/value-pilot-rows.json';sources.append(dataset)
    groups=read(dataset)['groups'];require(set(groups)=={'development','diagnostic','ordinary_development','train'},'Unexpected label source group; sealed final-test labels are forbidden')
    for group,rows in groups.items():
        for ordinal,row in enumerate(rows):
            result[config.canonical(row['state'])].append(dict(state=row['state'],label=row['label'],source=dict(path=relative(dataset),group=group,row=ordinal,row_sha256=config.digest(row))))
    for ordinal,row in enumerate(plan['cases']):
        result[config.canonical(row['state'])].append(dict(state=row['state'],label=row['reference'],source=dict(path=relative(bench.PLAN),case=ordinal,row_sha256=config.digest(row))))
    databases={
        'data/ordinary-data-pilot-v1/work.sqlite3':('labels',),
        'data/natural-study-v1/work.sqlite3':('h8_labels','h10_labels'),
        'data/horizon-finish-v1/work.sqlite3':('h8_references','h10_references','h12_references','prior_h14_labels','tier1_labels','tier2_labels'),
        'data/midgame-label-pilot-v1/work.sqlite3':('h8_labels','h10_labels','h12_labels')}
    for name,tables in databases.items():
        path=ROOT/name;sources.append(path)
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            db.execute('BEGIN');roots={}
            for key,text,digest in db.execute('SELECT state,body,hash FROM roots'):
                row=json.loads(text);require(config.digest(row)==digest,'Corrupt saved label root');roots[key]=row
            for table in tables:
                for key,text,digest in db.execute('SELECT state,body,hash FROM '+table):
                    item=json.loads(text);require(config.digest(item)==digest and key in roots,'Corrupt saved complete label')
                    root=roots[key];label=item['label'];require(label['choice_depth']>0 and len(label['action_scores'])==root['legal_mask'].bit_count(),'Incomplete root-Q label indexed')
                    result[config.canonical(root['state'])].append(dict(state=root['state'],label=label,source=dict(path=name,table=table,state_key=key.hex() if isinstance(key,bytes) else key,item_sha256=digest,root_sha256=config.digest(root))))
    return result,sources


def existing_Q(state,left_action,right_action,index):
    found=index.get(config.canonical(state),[]);items=[];sign=1 if state['controller']==0 else -1
    for row in found:
        require(row['state']==state,'Saved label lookup transformed full state')
        label=row['label'];q={entry['pit']:entry['value_player0'] for entry in label['action_scores']}
        require(left_action in q and right_action in q,'Saved full-root Q lacks action')
        a=sign*(label['target_player0']-q[left_action]);b=sign*(label['target_player0']-q[right_action]);require(a>=0 and b>=0,'Saved label actor sign differs')
        items.append(dict(source=row['source'],label_sha256=config.digest(label),label_kind=label['kind'],choice_depth=label.get('choice_depth'),target_player0=label['target_player0'],best_moves=label['best_moves'],display_best_moves=[x+1 for x in label['best_moves']],control_action_Q_player0=q[left_action],challenger_action_Q_player0=q[right_action],control_actor_regret=a,challenger_actor_regret=b))
    return dict(coverage='exact_raw_full_state' if items else 'uncovered',comparisons=items,orientation='Same normalized pits, stores and controller; no player swap, quotient or store-offset transform',interpretation='Retained finite-depth estimates retain their horizon; no causal game-blunder or deeper-truth claim')


def decision_summary(row):
    envelope=row['decision'];d=envelope['decision'];r=d['report'];telemetry=envelope['evaluator']
    return dict(engine=row['engine'],actor=row['actor'],chosen=d['chosen'],display_chosen=d['chosen']+1,kind=d['kind'],requested_ms=d['requested_ms'],wall_ms=d['wall_ms'],overshoot_ms=d['overshoot_ms'],nodes=d['nodes'],completed_depth=d['completed_depth'],completed_simulations=d['completed_simulations'],estimated_value_player0=r.get('estimated_value_player0',r.get('value_player0')),root_priors=r.get('root_priors'),root_visits=r.get('visits'),action_mean_value_player0=r.get('action_mean_value_player0'),bootstrap_player0=r.get('unvisited_root_q_player0'),physical_forwards=telemetry['physical_forward_executions'] if telemetry else r.get('nn_evaluations',0),feature_simulations=telemetry['counts']['feature_simulations'] if telemetry else r.get('feature_simulations',0),feature_deposits=telemetry['counts']['feature_deposits'] if telemetry else r.get('feature_deposits',0))


def wdl(values):return dict(wins=sum(x>0 for x in values),draws=sum(x==0 for x in values),losses=sum(x<0 for x in values))


def group_summary(seats):
    families=sorted(set(row['pair'] for row in seats));deltas=[]
    for pair in families:
        rows=[row for row in seats if row['pair']==pair];require(len(rows)==2,'Summary lacks paired seats');deltas.append(statistics.mean(row['challenger_minus_control_margin'] for row in rows))
    return dict(families=len(families),matched_seats=len(seats),control_wdl=wdl([row['control_margin'] for row in seats]),challenger_wdl=wdl([row['challenger_margin'] for row in seats]),control_mean_margin=statistics.mean(row['control_margin'] for row in seats),challenger_mean_margin=statistics.mean(row['challenger_margin'] for row in seats),paired_mean_challenger_minus_control=statistics.mean(deltas),family_deltas=deltas,confidence_interval=None,descriptive_only=True)


def arena_build():
    prior=baseline();inputs=dict(prior['inputs'])
    for path in (BASELINE,Path(__file__),CLI):inputs[relative(path)]=sha(path)
    witness=bench.closed_witness('evaluation');plan=bench.read_plan('evaluation');native=Native();labels,label_sources=label_index(plan)
    run,_=bench.paths('evaluation');journal=run/'work.sqlite3';comparisons={row['id']:row for row in plan['arena_comparisons']}
    cached_facts={};games={};ordinary=[];coverage=defaultdict(list);all_records=[]
    def facts(state):
        key=config.canonical(state)
        if key not in cached_facts:
            normal=native.normalize(flatten(state),actor=state['controller'],total=98);require(normal['position']==state and not normal['terminal'],'Saved decision not normalized/nonterminal')
            active,stratum=pilot._mechanism(state,native);cached_facts[key]=dict(active=active,stratum=stratum,band=stratum.split('/')[0],mechanism=stratum.split('/')[1],legal_actions=normal['legal_actions'],legal_action_count=len(normal['legal_actions']))
        return cached_facts[key]
    with closing(sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN')
        for idx,name,pair,seat,text,digest in db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'):
            game=json.loads(text);require(config.digest(game)==digest and game['status']=='terminal','Invalid saved terminal game');comp=comparisons[name];records=bench.records_for(db,idx);p=game['start'];steps=[]
            require(len(records)==len(game['moves'])==len(game['decisions']),'Saved decisions/game disagree')
            for choice,record in enumerate(records):
                p=bench.validate_record(record,idx,choice,p,dict(pair=pair,seat=seat),comp,plan,native);row=record['row'];state=record['state'];f=facts(state);role='candidate_profile' if row['actor']==seat else 'opponent_baseline'
                require(record['action']==game['moves'][choice] and row==game['decisions'][choice] and not row['fallback'],'Saved terminal trace/fallback differs')
                step=dict(source=dict(path=relative(journal),game_index=idx,choice_index=choice,decision_sha256=config.digest(record)),comparison=name,baseline=comp['opponent'],profile=comp['candidate'],pair=pair,family=game['family'],candidate_seat=seat,start_stratum=plan['arena_starts'][pair]['stratum'],state=state,state_key=raw_key(state),actor=row['actor'],actor_role=role,engine=row['engine'],action=record['action'],display_action=record['action']+1,**f)
                steps.append(step);all_records.append(step)
                for grouping,key in (('profile_actor',(comp['candidate'],role)),('engine',(row['engine'],)),('profile_baseline_actor',(comp['candidate'],comp['opponent'],role)),('start_stratum',(plan['arena_starts'][pair]['stratum'],))):coverage[(grouping,key)].append(step)
                if f['mechanism']=='ordinary':ordinary.append({**step,'saved_decision_summary':decision_summary(row)})
            require(p==game['final'] and not sum(map(sum,p['pits'])),'Literal saved final differs')
            margin=(p['stores'][0]-p['stores'][1])*(1 if seat==0 else -1);require(margin==game['candidate_final_margin'],'Saved margin differs')
            key=(comp['candidate'],comp['opponent'],pair,seat);require(key not in games,'Duplicate retained game')
            games[key]=dict(index=idx,game=game,steps=steps,records=records,source_sha256=digest)
    require(len(games)==96 and len(all_records)==2958,'Sealed arena geometry differs')
    seats=[];shared_ordinary=[]
    for opponent in bench.BASELINES:
        for pair in range(12):
            for seat in (0,1):
                a=games[('selfplay_control',opponent,pair,seat)];b=games[('selfplay_challenger',opponent,pair,seat)];require(a['game']['start']==b['game']['start'] and a['game']['family']==b['game']['family'],'Matched start/family differs')
                index=0;limit=min(len(a['records']),len(b['records']))
                while index<limit and a['records'][index]['action']==b['records'][index]['action']:
                    require(a['records'][index]['state']==b['records'][index]['state'] and a['records'][index]['after']==b['records'][index]['after'],'Shared prefix states differ')
                    if a['steps'][index]['mechanism']=='ordinary':shared_ordinary.append(dict(baseline=opponent,pair=pair,candidate_seat=seat,choice_index=index,actor_role=a['steps'][index]['actor_role'],state_key=a['steps'][index]['state_key'],stratum=a['steps'][index]['stratum'],action=a['steps'][index]['action']))
                    index+=1
                divergence=dict(identical_action_trajectory=index==limit,shared_choices=index,first_different_choice=None)
                if index==limit:require(len(a['records'])==len(b['records']) and a['game']['final']==b['game']['final'],'Completed game cannot be a strict prefix')
                else:
                    left=a['records'][index];right=b['records'][index];require(left['state']==right['state'],'First divergence not same raw state');f=facts(left['state']);role=a['steps'][index]['actor_role']
                    divergence=dict(identical_action_trajectory=False,shared_choices=index,first_different_choice=index+1,state=left['state'],state_key=raw_key(left['state']),actor=left['row']['actor'],first_changed_actor_role=role,opponent_first=role=='opponent_baseline',**f,
                        control=dict(source=a['steps'][index]['source'],saved_envelope=left['row']['decision'],summary=decision_summary(left['row'])),challenger=dict(source=b['steps'][index]['source'],saved_envelope=right['row']['decision'],summary=decision_summary(right['row'])),existing_Q_comparison=existing_Q(left['state'],left['action'],right['action'],labels),
                        attribution='First distinct saved action after a shared literal prefix. Opponent-first means an opponent decision at this divergence; candidate-first can reflect model and whole-clock search-work differences. Later states are unmatched; no unique causal blunder is inferred.')
                seats.append(dict(baseline=opponent,pair=pair,family=a['game']['family'],candidate_seat=seat,start_stratum=plan['arena_starts'][pair]['stratum'],start_mechanism=plan['arena_starts'][pair]['stratum'].split('/')[1],control_game_index=a['index'],challenger_game_index=b['index'],control_game_sha256=a['source_sha256'],challenger_game_sha256=b['source_sha256'],control_margin=a['game']['candidate_final_margin'],challenger_margin=b['game']['candidate_final_margin'],challenger_minus_control_margin=b['game']['candidate_final_margin']-a['game']['candidate_final_margin'],control_ordinary_decisions=sum(x['mechanism']=='ordinary' for x in a['steps']),challenger_ordinary_decisions=sum(x['mechanism']=='ordinary' for x in b['steps']),first_divergence=divergence))
    def coverage_summary(rows):
        normals=[row for row in rows if row['mechanism']=='ordinary'];eligible=[row for row in normals if row['active']>=25]
        return dict(decisions=len(rows),unique_raw_states=len({row['state_key'] for row in rows}),actual_strata=dict(sorted(Counter(row['stratum'] for row in rows).items())),ordinary_decisions=len(normals),ordinary_unique_raw_states=len({row['state_key'] for row in normals}),ordinary_legal_action_counts=dict(sorted(Counter(str(row['legal_action_count']) for row in normals).items())),ordinary_choice_bearing=sum(row['legal_action_count']>=2 for row in normals),ordinary_forced=sum(row['legal_action_count']==1 for row in normals),ordinary_active_ge25_decisions=len(eligible),ordinary_active_ge25_unique_raw_states=len({row['state_key'] for row in eligible}),ordinary_high_decisions=sum(row['band']=='50-74' for row in normals),ordinary_low_decisions=sum(row['band']=='25-49' for row in normals))
    matched_families=[]
    for opponent in bench.BASELINES:
        for pair in range(12):
            group=[row for row in seats if row['baseline']==opponent and row['pair']==pair];matched_families.append(dict(baseline=opponent,pair=pair,family=group[0]['family'],start_stratum=group[0]['start_stratum'],**group_summary(group)))
    aggregate={opponent:group_summary([row for row in seats if row['baseline']==opponent]) for opponent in bench.BASELINES}
    retained=read(bench.paths()[1])
    for opponent in bench.BASELINES:require(abs(aggregate[opponent]['paired_mean_challenger_minus_control']-retained['secondary_factor_contrasts'][opponent]['paired_mean_seed_margin_difference'])<1e-12,'Saved paired contrast differs')
    divergences=[row['first_divergence'] for row in seats if not row['first_divergence']['identical_action_trajectory']];ordinary_divergences=[row for row in divergences if row['mechanism']=='ordinary']
    counts=dict(games=96,decisions=len(all_records),matched_seats=len(seats),families=12,first_divergent_seats=len(divergences),opponent_first_divergences=sum(row['opponent_first'] for row in divergences),candidate_first_divergences=sum(not row['opponent_first'] for row in divergences),ordinary_first_divergences=len(ordinary_divergences),ordinary_candidate_first_divergences=sum(not row['opponent_first'] for row in ordinary_divergences),ordinary_opponent_first_divergences=sum(row['opponent_first'] for row in ordinary_divergences),ordinary_first_divergences_with_saved_Q=sum(row['existing_Q_comparison']['coverage']!='uncovered' for row in ordinary_divergences),shared_prefix_equal_ordinary_decisions=len(shared_ordinary),ordinary_observations=len(ordinary))
    for path in label_sources+[journal,bench.PLAN,bench.IMPLEMENTATION,bench.paths()[1],ROOT/'congklak/pilot.py',ROOT/'congklak/selfplay_pilot_benchmark.py']:
        inputs[relative(path)]=sha(path)
    for name,digest in inputs.items():require(sha(ROOT/name)==digest,'Input changed during read-only review: '+name)
    require('torch' not in sys.modules,'Read-only review imported Torch')
    return dict(schema='ordinary-selfplay-prior-arena-review-v1',passed=True,prior_inputs_preserved=3171,inputs=dict(sorted(inputs.items())),fixed_control_model_sha256=sha(bench.CONTROL),fixed_challenger_model_sha256=sha(bench.PILOT),rules_digest=config.RULES_DIGEST,closed_arena_witness=witness,counts=counts,
        coverage=dict(all_decisions=coverage_summary(all_records),groups=[dict(grouping=grouping,keys=list(keys),**coverage_summary(rows)) for (grouping,keys),rows in sorted(coverage.items())]),aggregate_by_baseline=aggregate,matched_families=matched_families,matched_seats=seats,
        by_baseline_start_mechanism=[dict(baseline=opponent,start_mechanism=mechanism,**group_summary([row for row in seats if row['baseline']==opponent and row['start_mechanism']==mechanism])) for opponent in bench.BASELINES for mechanism in ('ordinary','capture','bonus')],
        by_baseline_start_stratum=[dict(baseline=opponent,start_stratum=stratum,**group_summary([row for row in seats if row['baseline']==opponent and row['start_stratum']==stratum])) for opponent in bench.BASELINES for stratum in sorted({row['start_stratum'] for row in seats})],
        ordinary_observations=ordinary,shared_prefix_equal_ordinary_decisions=shared_ordinary,
        interpretation=dict(ordinary='Existing simulator classification across ALL legal actions: pass, capture and own-store bonus take priority; ordinary is not synonymous with quiet chosen action. Forced single-legal roots are counted separately.',start_stratum='Only describes the original root. Paired start groups contain four mechanism families, with finer band groups sometimes only one family; means are descriptive, no new confidence gates.',divergences='Only the first action divergence is an assured same-state comparison. Later ordinary observations measure coverage, not matched causal action errors.',labels='Exact raw-full-state retained completed labels only; horizon-sensitive approximate targets are not game truth. H8 relay names its training horizon; actual timed-search completed depths are saved.',readiness='Consumed development arenas and historically trained ancestors cannot establish unseen release strength; this diagnosis precedes fixed-model data collection and applies no sampler change.'),
        new_searches=0,new_games=0,new_teacher_labels=0,new_model_forwards=0,training_performed=False,sampling_applied=False,models_changed=False,source_data_pruned=False,default_promotion=False,torch_imported=False)


def arena_main():
    from . import quality_search
    from .neural_relay import NeuralNative
    def forbidden(*_args,**_kwargs):raise AssertionError('Ordinary review may not search or generate games')
    saved=(Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide)
    Native.analyze=NeuralNative.analyze=quality_search.search=bench.decide=forbidden
    try:body=arena_build()
    finally:Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide=saved
    expected=config.canonical(body)+'\n'
    if ARENA_OUTPUT.exists():require(ARENA_OUTPUT.read_text(encoding='utf-8')==expected,'Immutable ordinary arena review differs')
    else:atomic_json(ARENA_OUTPUT,body)
    print(config.canonical(dict(event='ordinary_selfplay_arena_review_complete',passed=True,counts=body['counts'],coverage=body['coverage']['all_decisions'])),flush=True)
    return body
