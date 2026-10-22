"""Explain retained pilot/control H8-relay margins without new games or searches."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import ExitStack,closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,value_pilot_benchmark as bench,value_pilot_data,quality_search
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.deeper_teacher_v2 import atomic_json

BASELINE=ROOT/'build/pilot-loss-review-baseline.json'
OUTPUT=ROOT/'build/pilot-loss-review-arena.json'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def state_key(state):return config.canonical(state)


def baseline():
    body=read(BASELINE)
    require(body['schema']=='pilot-loss-review-prior-immutability-v1' and body['passed'] is True and len(body['inputs'])==2600,'Wrong loss-review baseline')
    for name,digest in body['inputs'].items():require(sha(ROOT/name)==digest,'Prior input changed: '+name)
    return body


def read_games(stage,candidate):
    path,_=bench.paths(stage);comparison=candidate+'_vs_H8_relay_CPU';result={}
    with closing(sqlite3.connect((path/'work.sqlite3').as_uri()+'?mode=ro',uri=True)) as db:
        for index,text,digest in db.execute('SELECT idx,body,hash FROM games WHERE comparison=? ORDER BY pair,seat',(comparison,)):
            game=json.loads(text);require(config.digest(game)==digest and game['status']=='terminal','Invalid retained game')
            key=(game['pair'],game['candidate_seat']);require(key not in result,'Duplicate matched seat')
            result[key]=dict(body=game,source=dict(path=(path/'work.sqlite3').relative_to(ROOT).as_posix(),index=index,comparison=comparison,pair=key[0],candidate_seat=key[1],game_sha256=digest,stage=stage))
    require(set(result)=={(pair,seat) for pair in range(12) for seat in (0,1)},'Missing matched H8 games')
    return result


def replay(record,native,plan):
    game=record['body'];p=game['start'];steps=[];costs=defaultdict(lambda:dict(decisions=0,requested_ms=0,wall_ms=0.,overshoot_ms=0.,max_overshoot_ms=0.,nodes=0,completed_simulations=0,physical_forwards=0,feature_simulations=0,feature_deposits=0))
    require(len(game['moves'])==len(game['decisions']),'Game decision count differs')
    for index,(action,row) in enumerate(zip(game['moves'],game['decisions'])):
        normal=native.normalize(flatten(p),actor=p['controller'],total=98)
        require(not normal['terminal'],'Saved choices after terminal');p=normal['position'];before=p
        require(row['actor']==p['controller'] and action in normal['legal_actions'] and row['decision']['decision']['chosen']==action and not row['fallback'],'Saved move/actor/fallback differs')
        bench.validate_envelope(row['decision'],p,25,plan,row['engine'])
        transition=native.apply(flatten(p),actor=p['controller'],action=action,total=98,trace=True)
        require(transition['ok'],'Illegal retained move');p=transition['position']
        gains=[p['stores'][actor]-before['stores'][actor] for actor in (0,1)]
        captures=[event for event in transition['events'] if event['type']=='capture']
        require(gains==transition['store_delta'] and all(event['after']-event['before']==event['count'] for event in captures),'Literal store/capture facts differ')
        steps.append(dict(choice_index=index,choice=index+1,before=before,after=p,actor=row['actor'],engine=row['engine'],pit=action,display_pit=action+1,
            store_gain=gains,stores_after=p['stores'],capture_events=captures,capture_transfer=sum(event['count'] for event in captures),bonus_choice=transition['reason']=='own_store' and not transition['terminal'],
            forced_passes_after=transition['passes'],terminal_after=transition['terminal'],reason=transition['reason'],relays=transition['relays'],deposits=transition['drops']))
        decision=row['decision']['decision'];work=costs[row['engine']];work['decisions']+=1;work['requested_ms']+=decision['requested_ms'];work['wall_ms']+=decision['wall_ms'];work['overshoot_ms']+=decision['overshoot_ms'];work['max_overshoot_ms']=max(work['max_overshoot_ms'],decision['overshoot_ms']);work['nodes']+=decision['nodes'];work['completed_simulations']+=decision['completed_simulations'] or 0
        report=decision['report'];telemetry=row['decision']['evaluator']
        work['physical_forwards']+=(telemetry['physical_forward_executions'] if telemetry else report.get('nn_evaluations',0))
        work['feature_simulations']+=(telemetry['counts']['feature_simulations'] if telemetry else report.get('feature_simulations',0))
        work['feature_deposits']+=(telemetry['counts']['feature_deposits'] if telemetry else report.get('feature_deposits',0))
    require(p==game['final'] and not sum(map(sum,p['pits'])),'Retained final position differs')
    margin=(p['stores'][0]-p['stores'][1])*(1 if game['candidate_seat']==0 else -1)
    require(margin==game['candidate_final_margin'] and game['candidate_match_score']==(1 if margin>0 else 0 if margin<0 else .5),'Retained margin/WDL differs')
    return dict(steps=steps,summary=dict(source=record['source'],family=game['family'],candidate_seat=game['candidate_seat'],margin_seeds=margin,match_score=game['candidate_match_score'],choices=len(steps),fallbacks=game['fallbacks'],cost_by_engine=dict(costs),
        capture_events=sum(len(step['capture_events']) for step in steps),capture_transfer=sum(step['capture_transfer'] for step in steps),bonus_choices=sum(step['bonus_choice'] for step in steps),literal_deposits=sum(step['deposits'] for step in steps),
        largest_capture_observations=sorted([step for step in steps if step['capture_events']],key=lambda step:(-step['capture_transfer'],step['choice_index']))[:3]))


def label_index(groups,plan):
    result=defaultdict(list)
    for group,rows in groups.items():
        for index,row in enumerate(rows):
            result[state_key(row['state'])].append(dict(source=dict(path='build/value-pilot-rows.json',group=group,row=index,state_key=row['state_key'],row_sha256=config.digest(row)),label=row['label'],teacher_depth=row['teacher_depth'],label_kind=row['label'].get('kind'),state=row['state']))
    for index,row in enumerate(plan['cases']):
        result[state_key(row['state'])].append(dict(source=dict(path='build/value-pilot-benchmark-plan.json',case=index,state_key=row['state_key'],row_sha256=config.digest(row)),label=row['reference'],teacher_depth=row['reference'].get('choice_depth'),label_kind=row['reference']['kind'],state=row['state']))
    return result


def stored_q(state,left_action,right_action,index):
    found=index.get(state_key(state),[]);comparisons=[];sign=1 if state['controller']==0 else -1
    for item in found:
        require(item['state']==state,'Label lookup is not exact full state')
        label=item['label'];q={row['pit']:row['value_player0'] for row in label['action_scores']}
        require(left_action in q and right_action in q,'Same-state label lacks selected legal actions')
        left=sign*(label['target_player0']-q[left_action]);right=sign*(label['target_player0']-q[right_action])
        require(left>=0 and right>=0,'Wrong label actor perspective')
        comparisons.append(dict(source=item['source'],teacher_depth=item['teacher_depth'],label_kind=item['label_kind'],label_sha256=config.digest(label),target_player0=label['target_player0'],best_moves=label['best_moves'],display_best_moves=[pit+1 for pit in label['best_moves']],
            control_action_Q_player0=q[left_action],pilot_action_Q_player0=q[right_action],control_actor_regret=left,pilot_actor_regret=right,pilot_minus_control_actor_regret=right-left))
    return dict(coverage='exact_full_state_stored_label' if found else 'uncovered',state_match='Exact normalized pits, stores and controller; no quotient or store-offset transform',comparisons=comparisons,
        interpretation='Stored finite-horizon Q is descriptive; only kind exact_optimal_margin carries its original verified-domain exact status. Uncovered positions have no new Q search.')


def snapshot(record,index):
    row=record['body']['decisions'][index];decision=row['decision']['decision'];report=decision['report']
    return dict(source={**record['source'],'choice_index':index,'decision_sha256':config.digest(row)},saved_decision=row,
        summary=dict(engine=row['engine'],actor=row['actor'],chosen=decision['chosen'],display_chosen=decision['chosen']+1,kind=decision['kind'],wall_ms=decision['wall_ms'],requested_ms=decision['requested_ms'],overshoot_ms=decision['overshoot_ms'],nodes=decision['nodes'],completed_depth=decision['completed_depth'],completed_simulations=decision['completed_simulations'],
            estimated_value_player0=report.get('estimated_value_player0',report.get('value_player0')),neural_bootstrap_player0=report.get('unvisited_root_q_player0'),visits=report.get('visits'),root_priors=report.get('root_priors'),visit_policy=report.get('policy'),action_mean_value_player0=report.get('action_mean_value_player0'),
            work=report.get('work',{key:report.get(key) for key in ('deposits','feature_simulations','feature_deposits','nn_evaluations')}),evaluator=row['decision']['evaluator']))


def divergence(left,right,left_replay,right_replay,labels=None,evaluators=None,cache=None,left_name='control',right_name='pilot'):
    a=left_replay['steps'];b=right_replay['steps'];index=0
    while index<min(len(a),len(b)) and (a[index]['actor'],a[index]['pit'])==(b[index]['actor'],b[index]['pit']):
        require(a[index]['before']==b[index]['before'] and a[index]['after']==b[index]['after'],'Shared actions produced different states');index+=1
    if index==min(len(a),len(b)):
        require(len(a)==len(b) and left['body']['final']==right['body']['final'],'One completed game cannot be a strict prefix')
        return dict(shared_choices=index,identical_action_trajectory=True,first_different_choice=None)
    state=a[index]['before'];require(state==b[index]['before'],'First divergence is not the same literal state')
    seat=left['body']['candidate_seat'];require(seat==right['body']['candidate_seat'],'Compared different seats')
    actor=state['controller'];role='candidate_profile' if actor==seat else 'opponent_H8_relay'
    result=dict(shared_choices=index,identical_action_trajectory=False,first_different_choice=index+1,state=state,actor=actor,candidate_seat=seat,first_changed_actor_role=role,opponent_first=actor!=seat,
        attribution='Unidentified from independent wall-clock runs. Opponent-first means the first distinct saved action belongs to H8 relay, not the first game decision or an absence of earlier candidate evaluations. Candidate-first differences can include changed values and timing/search-count variation. Later paths are unmatched, not counterfactual evaluations.')
    result[left_name]=snapshot(left,index);result[right_name]=snapshot(right,index)
    result[left_name+'_nearby_observations']=a[max(0,index-2):index+4];result[right_name+'_nearby_observations']=b[max(0,index-2):index+4]
    if right_name=='posttraining_control':result['attribution']='Both candidate and opponent models/search are unchanged. This is descriptive same-control wall-clock variation, not a training effect; later paths are unmatched.'
    if labels is not None:result['existing_Q_comparison']=stored_q(state,a[index]['pit'],b[index]['pit'],labels)
    if evaluators is not None:
        key=state_key(state)
        if key not in cache:
            prediction={name:{key:value for key,value in evaluator.evaluate_many([state])[0].items() if key!='inference_ms'} for name,evaluator in evaluators.items()}
            require(all(value['native_actual_evals']==1 and value['cache_hit'] is False for value in prediction.values()),'Direct forward count differs')
            require(config.canonical(prediction['control']['policy_logits'])==config.canonical(prediction['pilot']['policy_logits']),'Frozen raw policy changed')
            cache[key]=dict(control=prediction['control'],pilot=prediction['pilot'],pilot_minus_control_value_player0=prediction['pilot']['value_player0']-prediction['control']['value_player0'],policy_logits_identical=True,
                new_forward_timing_measured=False,scope='Two direct uncached CPU forwards at this literal state; no search or chosen-move counterfactual; per-call inference timing omitted')
        result['analysis_model_forwards']=cache[key]
    return result


def wdls(values):
    return dict(wins=sum(value>0 for value in values),draws=sum(value==0 for value in values),losses=sum(value<0 for value in values))


def build():
    prior=baseline();inputs=dict(prior['inputs'])
    for path in (Path(__file__),BASELINE):inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    pre_witness=bench.pretraining_witness();post_witness=bench.closed_witness('posttraining')
    _,groups=value_pilot_data.prepare();plan=bench.read_plan('posttraining');preplan=bench.read_plan('pretraining')
    labels=label_index(groups,plan);native=Native();pre=read_games('pretraining','policy_control');control=read_games('posttraining','policy_control');pilot=read_games('posttraining','value_pilot')
    replayed={name:{key:replay(record,native,preplan if name=='pretraining_control' else plan) for key,record in source.items()} for name,source in (('pretraining_control',pre),('control',control),('pilot',pilot))}
    require(all(pre[key]['body']['start']==control[key]['body']['start']==pilot[key]['body']['start'] and pre[key]['body']['family']==control[key]['body']['family']==pilot[key]['body']['family'] for key in control),'Matched starts/families changed')
    seats=[];forward_cache={}
    with ExitStack() as stack:
        evaluators={name:stack.enter_context(CpuBatchEvaluator(path,cache_capacity=0,max_batch=1)) for name,path in (('control',bench.CONTROL),('pilot',bench.PILOT))}
        for key in sorted(control):
            c=replayed['control'][key];t=replayed['pilot'][key];p=replayed['pretraining_control'][key]
            seats.append(dict(pair=key[0],candidate_seat=key[1],family=control[key]['body']['family'],stratum=plan['arena_starts'][key[0]]['stratum'],control=c['summary'],pilot=t['summary'],pretraining_control=p['summary'],
                pilot_minus_control_margin=t['summary']['margin_seeds']-c['summary']['margin_seeds'],same_control_post_minus_pre_margin=c['summary']['margin_seeds']-p['summary']['margin_seeds'],
                first_divergence=divergence(control[key],pilot[key],c,t,labels,evaluators,forward_cache),same_control_drift_divergence=divergence(pre[key],control[key],p,c,left_name='pretraining_control',right_name='posttraining_control')))
    families=[]
    for pair in range(12):
        rows=[row for row in seats if row['pair']==pair];require(len(rows)==2,'Missing paired seats')
        families.append(dict(pair=pair,family=rows[0]['family'],stratum=rows[0]['stratum'],control_mean_margin=statistics.mean(row['control']['margin_seeds'] for row in rows),pilot_mean_margin=statistics.mean(row['pilot']['margin_seeds'] for row in rows),pretraining_control_mean_margin=statistics.mean(row['pretraining_control']['margin_seeds'] for row in rows),
            pilot_minus_control_paired_mean_margin=statistics.mean(row['pilot_minus_control_margin'] for row in rows),same_control_post_minus_pre_paired_mean_margin=statistics.mean(row['same_control_post_minus_pre_margin'] for row in rows),
            seat_contributions=[dict(seat=row['candidate_seat'],delta=row['pilot_minus_control_margin'],control_margin=row['control']['margin_seeds'],pilot_margin=row['pilot']['margin_seeds'],first_changed_actor_role=row['first_divergence'].get('first_changed_actor_role'),shared_choices=row['first_divergence']['shared_choices']) for row in rows]))
    deltas=[row['pilot_minus_control_paired_mean_margin'] for row in families];negative=sorted([row for row in families if row['pilot_minus_control_paired_mean_margin']<0],key=lambda row:(row['pilot_minus_control_paired_mean_margin'],row['pair']));positive=[row for row in families if row['pilot_minus_control_paired_mean_margin']>0]
    leave_one=[dict(excluded_pair=row['pair'],excluded_family=row['family'],families_remaining=11,mean_pilot_minus_control_margin=statistics.mean(value for index,value in enumerate(deltas) if index!=row['pair'])) for row in families]
    control_values=[row['control']['margin_seeds'] for row in seats];pilot_values=[row['pilot']['margin_seeds'] for row in seats];pre_values=[row['pretraining_control']['margin_seeds'] for row in seats]
    main_divergences=[row['first_divergence'] for row in seats if not row['first_divergence']['identical_action_trajectory']];drift_divergences=[row['same_control_drift_divergence'] for row in seats if not row['same_control_drift_divergence']['identical_action_trajectory']]
    negative_mass=-sum(row['pilot_minus_control_paired_mean_margin'] for row in negative)
    aggregate=dict(control=dict(wdl=wdls(control_values),mean_margin_seeds=statistics.mean(control_values),match_score=statistics.mean(row['control']['match_score'] for row in seats)),pilot=dict(wdl=wdls(pilot_values),mean_margin_seeds=statistics.mean(pilot_values),match_score=statistics.mean(row['pilot']['match_score'] for row in seats)),pretraining_control=dict(wdl=wdls(pre_values),mean_margin_seeds=statistics.mean(pre_values)),
        pilot_minus_control_paired_mean_margin=statistics.mean(deltas),sum_seat_margin_deltas=sum(row['pilot_minus_control_margin'] for row in seats),same_control_post_minus_pre_mean_margin=statistics.mean(control_values)-statistics.mean(pre_values),
        negative_families=len(negative),positive_families=len(positive),unchanged_families=sum(value==0 for value in deltas),gross_negative_paired_margin_mass=negative_mass,gross_positive_paired_margin_mass=sum(row['pilot_minus_control_paired_mean_margin'] for row in positive),
        worst_family_pair=negative[0]['pair'] if negative else None,top_two_fraction_of_gross_adverse_mass=(-sum(row['pilot_minus_control_paired_mean_margin'] for row in negative[:2])/negative_mass if negative_mass else None),
        leave_one_family_out_min=min(row['mean_pilot_minus_control_margin'] for row in leave_one),leave_one_family_out_max=max(row['mean_pilot_minus_control_margin'] for row in leave_one),leave_one_family_out_positive=sum(row['mean_pilot_minus_control_margin']>0 for row in leave_one),
        first_changed_actor_counts=dict(Counter(row['first_changed_actor_role'] for row in main_divergences)),same_control_drift_first_changed_actor_counts=dict(Counter(row['first_changed_actor_role'] for row in drift_divergences)),
        interpretation='WDL clips outcome severity; seed margin retains magnitude. Tail contributions and leave-one-family-out are descriptive sensitivity, not confidence intervals or new selection/promotion gates.')
    expected=read(ROOT/'build/value-pilot-benchmark-posttraining.json')['secondary_factor_contrasts']['H8_relay_CPU']['paired_mean_seed_margin_difference']
    require(abs(aggregate['pilot_minus_control_paired_mean_margin']-expected)<1e-12,'Matched delta differs from frozen benchmark')
    counts=dict(families=12,matched_seats=24,replayed_games=72,literal_choices=sum(row['summary']['choices'] for group in replayed.values() for row in group.values()),first_divergent_seats=len(main_divergences),identical_action_trajectories=24-len(main_divergences),
        opponent_first_divergences=sum(row['opponent_first'] for row in main_divergences),candidate_first_divergences=sum(not row['opponent_first'] for row in main_divergences),same_control_drift_divergent_seats=len(drift_divergences),
        divergent_seats_with_exact_state_Q=sum(row['existing_Q_comparison']['coverage']!='uncovered' for row in main_divergences),divergent_seats_without_stored_Q=sum(row['existing_Q_comparison']['coverage']=='uncovered' for row in main_divergences),
        analysis_forward_states=len(forward_cache),analysis_CPU_model_forwards=2*len(forward_cache))
    for path in (ROOT/'congklak/value_pilot_benchmark.py',ROOT/'congklak/value_pilot_data.py',ROOT/'congklak/puct_cpu_inference.py',ROOT/'build/value-pilot-model-witness.json'):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in inputs.items():require(sha(ROOT/name)==digest,'Input changed during loss review: '+name)
    require('torch' not in sys.modules,'Loss review imported Torch')
    return dict(schema='pilot-loss-review-arena-v1',passed=True,prior_inputs_preserved=2600,inputs=dict(sorted(inputs.items())),control_model_sha256=sha(bench.CONTROL),pilot_model_sha256=sha(bench.PILOT),rules_digest=config.RULES_DIGEST,
        counts=counts,aggregate=aggregate,families=families,adverse_families_ranked=[dict(rank=index+1,**row) for index,row in enumerate(negative)],leave_one_family_out=leave_one,seats=seats,pretraining_witness=pre_witness,posttraining_witness=post_witness,
        source_scope='All12 consumed development H8-relay families,24 seats, three retained runs. Exact full-state labels only; no quotient/store-offset transformation.',
        definitions=dict(values='Fixed-player0 seed margins; candidate final margins use its declared seat. Stored-label regrets use actor sign.',first_divergence='First distinct saved(actor,pit) after a literal common prefix from identical start and seat; later trajectories are unmatched.',opponent_first='The first divergence is an H8-relay decision, not necessarily the first game decision or before candidate evaluation. H8 labels its training horizon; actual completed search depths are retained separately.',observations='Nearby captures, bonuses, forced passes and store gains are replay facts, not causal blunders.',readiness='This diagnostic exposes failure concentration and missing label coverage; it cannot certify expert/finished strength.'),
        new_games=0,new_searches=0,new_teacher_labels=0,training_performed=False,model_changed=False,policy_changed=False,data_ingested=False,sealed_final_test_consumed=False,default_promotion=False,energy_measured=False,torch_imported=False)


def main():
    def prohibited(*_args,**_kwargs):raise AssertionError('Loss review must not run searches or new decisions')
    original=(Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide)
    Native.analyze=NeuralNative.analyze=quality_search.search=bench.decide=prohibited
    try:body=build()
    finally:Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide=original
    if OUTPUT.exists():require(OUTPUT.read_text(encoding='utf-8')==config.canonical(body)+'\n','Immutable loss review differs')
    else:atomic_json(OUTPUT,body)
    print(config.canonical(dict(event='pilot_loss_review_arena_complete',passed=True,counts=body['counts'],matched_delta=body['aggregate']['pilot_minus_control_paired_mean_margin'])),flush=True)
    return body


if __name__=='__main__':main()
