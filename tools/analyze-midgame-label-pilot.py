"""Fixed-model, same-state review of complete requested midgame labels only."""
from collections import Counter,defaultdict
import argparse
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,midgame_label_pilot as p
from congklak.neural_relay import Model
from congklak.puct_cpu_inference import CpuBatchEvaluator

OUTPUT=ROOT/'build/midgame-label-pilot-comparison.json'
SCHEMA='fixed-model-midgame-requested-label-comparison-v1'


def regret(label,action,actor):
    q={row['pit']:row['value_player0'] for row in label['action_scores']}
    p.require(action in q,'Saved/raw-policy action is not legal in its original state')
    value=(1 if actor==0 else -1)*(label['target_player0']-q[action])
    p.require(value>=0,'Actor-signed action regret is negative')
    return dict(action=action,display_pit=action+1,Q_player0=q[action],actor_signed_regret=value,is_best=action in label['best_moves'])


def relation(left,right):
    a,b=left['label'],right['label'];aq={row['pit']:row['value_player0'] for row in a['action_scores']};bq={row['pit']:row['value_player0'] for row in b['action_scores']}
    p.require(set(aq)==set(bq),'Paired full-root legal actions differ')
    return dict(left_depth=a['choice_depth'],right_depth=b['choice_depth'],
        target_shift_player0=b['target_player0']-a['target_player0'],absolute_target_shift=abs(b['target_player0']-a['target_player0']),
        action_Q_shift_mae=math.fsum(abs(bq[pit]-aq[pit]) for pit in sorted(aq))/len(aq),
        same_best_set=set(a['best_moves'])==set(b['best_moves']),best_sets_overlap=bool(set(a['best_moves'])&set(b['best_moves'])),
        best_sets_disjoint=not bool(set(a['best_moves'])&set(b['best_moves'])),
        left_best_moves=a['best_moves'],right_best_moves=b['best_moves'])


def summarize(rows):
    out=dict(selected_roots=len(rows),requested_depths={},pairs={},complete_H8_H10_H12=sum(len(row['requested_depths_available'])==3 for row in rows))
    for depth in ('8','10','12'):
        group=[row['by_requested_depth'][depth] for row in rows if depth in row['by_requested_depth']]
        eligible=sum(row['attempts']['h'+depth]['eligible'] for row in rows)
        n=len(group);entry=dict(eligible_roots=eligible,out_of_scope_roots=len(rows)-eligible,complete_labels=n,missing_labels=eligible-n)
        for model in ('control','pilot'):
            entry[model]=dict(value_mae_seeds=math.fsum(row[model]['absolute_value_error_seeds'] for row in group)/n if n else None,
                raw_policy_regret=math.fsum(row[model]['raw_policy']['actor_signed_regret'] for row in group)/n if n else None)
        out['requested_depths'][depth]=entry
    for pair in ('8_10','10_12','8_12'):
        group=[row['pairs'][pair] for row in rows if row['pairs'][pair] is not None];n=len(group)
        a,b=pair.split('_');eligible=sum(row['attempts']['h'+a]['eligible'] and row['attempts']['h'+b]['eligible'] for row in rows)
        out['pairs'][pair]=dict(eligible_roots=eligible,out_of_scope_roots=len(rows)-eligible,complete_pairs=n,eligible_unpaired_roots=eligible-n,
            unpaired_roots=len(rows)-n,unpaired_denominator='all_selected_roots_including_out_of_scope',
            best_sets_changed=sum(not row['same_best_set'] for row in group),best_sets_disjoint=sum(row['best_sets_disjoint'] for row in group),
            mean_absolute_target_shift=math.fsum(row['absolute_target_shift'] for row in group)/n if n else None,
            mean_action_Q_shift=math.fsum(row['action_Q_shift_mae'] for row in group)/n if n else None)
    return out


def build(path=p.DEFAULT):
    prior=p.checked_baseline();rows,stores,manifest=p.load_rows(path);models={name:Model(ROOT/file) for name,file in p.MODELS.items()}
    p.require('torch' not in sys.modules,'Comparison must be CPU-only')
    frozen=[key for key in models['control'].weights if key not in ('layers.4.weight','layers.4.bias')]
    p.require(all(models['control'].weights[key]==models['pilot'].weights[key] for key in frozen),'Fixed trunk/policy differs')
    names=['tools/analyze-midgame-label-pilot.py',*p.SOURCES,'build/midgame-pilot-baseline.json',
        'build/midgame-label-pilot-registry.json','build/midgame-label-pilot-plan.json','build/midgame-label-pilot-implementation.json',
        'build/midgame-data-pilot-registry.json','build/midgame-data-pilot-plan.json','build/midgame-data-pilot-implementation.json',
        *p.MODELS.values(),'congklak/puct_cpu_inference.py','build/puct-cpu-build.json','build/congklak_puct_cpu.dll']
    path=Path(path).resolve();collection=ROOT/manifest['identity']['collection']['path']
    names.extend(str(file.relative_to(ROOT)).replace('\\','/') for directory in (path,collection) for file in (directory/'work.sqlite3',directory/'manifest.json'))
    if not manifest['identity']['fixture']:names.extend(('build/midgame-label-pilot-result.json','build/midgame-data-pilot-result.json'))
    inputs={name:p.sha(ROOT/name) for name in sorted(set(names))}
    output=[];identities={};groups=defaultdict(list);forward_count=0
    with CpuBatchEvaluator(models['control'],cache_capacity=0,max_batch=1,backend='scalar') as control,CpuBatchEvaluator(models['pilot'],cache_capacity=0,max_batch=1,backend='scalar') as pilot:
        identities=dict(control=control.identity,pilot=pilot.identity)
        for key,row in rows.items():
            state=row['state'];actor=state['controller'];legal=[pit for pit in range(7) if row['legal_mask']&(1<<pit)]
            predictions={}
            for name,evaluator in (('control',control),('pilot',pilot)):
                result=evaluator.evaluate_many([state])[0]
                # Per-call timings are deliberately unmeasured in this deterministic review.
                predictions[name]={name:value for name,value in result.items() if name!='inference_ms'}
                p.require(predictions[name]['native_actual_evals']==1 and predictions[name]['feature_simulations']>=1,'Expected one uncached nonterminal forward')
                forward_count+=1
            p.require(predictions['control']['policy_logits']==predictions['pilot']['policy_logits'],'Identical fixed policy logits differ')
            actions={name:max(legal,key=lambda pit:(result['policy_logits'][pit],-pit)) for name,result in predictions.items()}
            diagnostic=row['study_role']=='diagnostic';provenance=row['root']['provenance']
            case=dict(state_key=key,state=state,family=row['family'],cohort=row['cohort'],study_role=row['study_role'],band=row['band'],stratum=row['stratum'],active=row['active'],
                actor=actor,legal_actions=legal,source_root_sha256=row['source_root_sha256'],root_sha256=config.digest(row),
                analysis_model_forwards=predictions,requested_depths_available=[],attempts={},by_requested_depth={},pairs={})
            if diagnostic:
                p.require(provenance['raw_state']==state and provenance['swapped'] is False,'Diagnostic original orientation changed')
                case['saved_first_divergence']=dict(pair=provenance['pair'],candidate_seat=provenance['candidate_seat'],first_different_choice=provenance['first_different_choice'],
                    actor_role='candidate' if actor==provenance['candidate_seat'] else 'opponent',
                    original_state_unchanged=True,control_game_action=provenance['control_chosen'],pilot_game_action=provenance['pilot_chosen'],
                    action_names_refer_to_games=True,opponent_first=actor!=provenance['candidate_seat'])
            reference=stores['old_h8_references'].get(key)
            case['old_H8_reference']=None if reference is None else dict(source=reference['source'],label_sha256=reference['label_sha256'],
                old_label_unchanged=True,requested_H8_same_Q_best=(key in stores['h8_labels'] and p.q_signature(stores['h8_labels'][key])=={name:reference['label'][name] for name in ('target_player0','best_moves','action_scores')}))
            for stage in ('h8','h10','h12'):
                attempt=stores[stage+'_attempts'].get(key)
                if attempt is None:
                    case['attempts'][stage]=dict(eligible=False,status='out_of_scope');continue
                depth=attempt['requested_depth'];item=stores[stage+'_labels'].get(key)
                case['attempts'][stage]=dict(eligible=True,status=attempt['status'],requested_depth=depth,completed_depth=attempt['completed_depth'],
                    requested_label_admitted=item is not None,auxiliary_completed_depths=[report['choice_depth'] for report in attempt['history'] if report['choice_depth']!=depth],
                    node_budget=manifest['identity']['settings'][stage]['node_budget'],actual_work=attempt['work'],attempt_sha256=config.digest(attempt))
                if item is None:continue
                label=item['label'];case['requested_depths_available'].append(depth)
                entry=dict(label_sha256=config.digest(item),target_player0=label['target_player0'],best_moves=label['best_moves'],display_best_pits=[pit+1 for pit in label['best_moves']],action_scores=label['action_scores'])
                for name,result in predictions.items():
                    signed=result['value_player0']-label['target_player0']
                    entry[name]=dict(value_player0=result['value_player0'],value_error_player0=signed,value_error_actor=(1 if actor==0 else -1)*signed,absolute_value_error_seeds=abs(signed),raw_policy=regret(label,actions[name],actor))
                entry['saved_actions']=None if not diagnostic else {name:regret(label,provenance[name+'_chosen'],actor) for name in ('control','pilot')}
                case['by_requested_depth'][str(depth)]=entry
            for a,b in ((8,10),(10,12),(8,12)):
                left,right=stores['h'+str(a)+'_labels'].get(key),stores['h'+str(b)+'_labels'].get(key)
                case['pairs'][str(a)+'_'+str(b)]=relation(left,right) if left is not None and right is not None else None
            output.append(case);groups[row['study_role']+'/'+row['cohort']].append(case)
    counts=p.counts(rows,stores)
    result=dict(schema=SCHEMA,passed=True,inputs=inputs,counts=counts,groups={name:summarize(values) for name,values in sorted(groups.items())},cases=output,
        model_identities=identities,analysis_model_forwards=forward_count,new_forward_timing_measured=False,all_fixed_policy_logits_equal=True,diagnostic_original_state_orientation=True,
        comparisons_use_complete_requested_intersections=True,no_auxiliary_fallback=True,old_Q_reference_compared_in_original_orientation=True,
        deeper_is_not_game_truth=True,saved_action_regret_is_not_causal_game_error=True,training_performed=False,new_games=False,new_teacher_searches=False,
        weights_applied=False,data_ingested=False,torch_imported=False,sealed_final_test_consumed=False,
        clock_scope='Historical attempt work remains saved. New direct-forward timing is unmeasured; inference_ms omitted for deterministic publication.',
        interpretation='Selected diagnostic/development data only. Regrets are actor-signed within the same literal full state and completed finite horizon. Opponent-first divergences do not isolate value-head causation.')
    p.require(all(p.sha(ROOT/name)==digest for name,digest in inputs.items()),'Comparison input changed during execution')
    p.require(p.checked_baseline()==prior and 'torch' not in sys.modules,'Prior inputs/runtime changed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('build','verify'));args=parser.parse_args();body=build()
    if args.operation=='build':p.publish(OUTPUT,body)
    else:p.require(p.read(OUTPUT)==body,'Comparison differs from independently recomputed current inputs')
    print(config.canonical(dict(event='midgame_label_comparison_verified',sha256=config.digest(body),counts=body['counts'])),flush=True)


if __name__=='__main__':main()
