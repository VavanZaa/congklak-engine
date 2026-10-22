"""Independent arithmetic for the bounded midgame comparison, used by method08."""
from collections import Counter
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.puct_cpu_inference import CpuBatchEvaluator
from data_pilot_independent_audit import load,sha,norm
from stage5_deeper_audit_acceptance import value_and_policy
from quality_audit_acceptance import module


def regret(label,action,actor):
    q={r['pit']:r['value_player0'] for r in label['action_scores']}
    sign=1 if actor==0 else -1
    optimum=max(sign*v for v in q.values())
    return dict(action=action,display_pit=action+1,Q_player0=q[action],
        actor_signed_regret=optimum-sign*q[action],is_best=action in label['best_moves'])


def relation(left,right):
    a,b=left['label'],right['label']
    qa={r['pit']:r['value_player0'] for r in a['action_scores']}
    qb={r['pit']:r['value_player0'] for r in b['action_scores']}
    delta=b['target_player0']-a['target_player0']
    return dict(left_depth=a['choice_depth'],right_depth=b['choice_depth'],
        target_shift_player0=delta,absolute_target_shift=abs(delta),
        action_Q_shift_mae=math.fsum(abs(qb[p]-qa[p]) for p in sorted(qa))/len(qa),
        same_best_set=set(a['best_moves'])==set(b['best_moves']),
        best_sets_overlap=bool(set(a['best_moves'])&set(b['best_moves'])),
        best_sets_disjoint=not bool(set(a['best_moves'])&set(b['best_moves'])),
        left_best_moves=a['best_moves'],right_best_moves=b['best_moves'])


def summary(cases):
    result=dict(selected_roots=len(cases),requested_depths={},pairs={},
        complete_H8_H10_H12=sum(set(r['requested_depths_available'])=={8,10,12} for r in cases))
    for depth in ('8','10','12'):
        eligible=[r for r in cases if r['attempts']['h'+depth]['eligible']]
        complete=[r['by_requested_depth'][depth] for r in eligible if depth in r['by_requested_depth']]
        n=len(complete)
        result['requested_depths'][depth]=dict(eligible_roots=len(eligible),
            out_of_scope_roots=len(cases)-len(eligible),complete_labels=n,missing_labels=len(eligible)-n)
        for model in ('control','pilot'):
            result['requested_depths'][depth][model]=dict(
                value_mae_seeds=math.fsum(r[model]['absolute_value_error_seeds'] for r in complete)/n if n else None,
                raw_policy_regret=math.fsum(r[model]['raw_policy']['actor_signed_regret'] for r in complete)/n if n else None)
    for left,right in ((8,10),(10,12),(8,12)):
        key=f'{left}_{right}'
        eligible=[r for r in cases if r['attempts'][f'h{left}']['eligible'] and r['attempts'][f'h{right}']['eligible']]
        paired=[r['pairs'][key] for r in eligible if r['pairs'][key] is not None];n=len(paired)
        result['pairs'][key]=dict(eligible_roots=len(eligible),out_of_scope_roots=len(cases)-len(eligible),
            complete_pairs=n,eligible_unpaired_roots=len(eligible)-n,unpaired_roots=len(cases)-n,
            unpaired_denominator='all_selected_roots_including_out_of_scope',
            best_sets_changed=sum(not r['same_best_set'] for r in paired),
            best_sets_disjoint=sum(r['best_sets_disjoint'] for r in paired),
            mean_absolute_target_shift=math.fsum(r['absolute_target_shift'] for r in paired)/n if n else None,
            mean_action_Q_shift=math.fsum(r['action_Q_shift_mae'] for r in paired)/n if n else None)
    return result


def check(test,rows,stores):
    report=load(ROOT/'build/midgame-label-pilot-comparison.json')
    test.assertEqual(report['schema'],'fixed-model-midgame-requested-label-comparison-v1')
    test.assertEqual([r['state_key'] for r in report['cases']],list(rows))
    for name,digest in report['inputs'].items():test.assertEqual(sha(ROOT/name),digest,name)
    for name,digest in load(ROOT/'build/midgame-pilot-weight-review.json')['inputs'].items():test.assertEqual(sha(ROOT/name),digest,name)
    stages={}
    for stage in ('h8','h10','h12'):
        targets=[k for k,r in rows.items() if stage!='h12' or r['study_role']=='diagnostic']
        attempts,labels=stores[stage+'_attempts'],stores[stage+'_labels']
        stages[stage]=dict(targets=len(targets),attempts=len(attempts),labels=len(labels),
            rejected=sum(a['status']=='budget_rejected' for a in attempts.values()),pending=len(targets)-len(attempts),
            by_role={role:dict(targets=sum(rows[k]['study_role']==role for k in targets),
                labels=sum(rows[k]['study_role']==role for k in labels),
                rejected=sum(rows[k]['study_role']==role and a['status']=='budget_rejected' for k,a in attempts.items()))
                for role in ('fresh','diagnostic')})
    test.assertEqual(report['counts'],dict(roots=len(rows),fresh_roots=sum(r['study_role']=='fresh' for r in rows.values()),
        diagnostic_roots=sum(r['study_role']=='diagnostic' for r in rows.values()),old_h8_reference_labels=len(stores['old_h8_references']),
        cohort_counts=dict(Counter(r['cohort'] for r in rows.values())),stages=stages,
        completed_H8_H10_pairs=len(set(stores['h8_labels'])&set(stores['h10_labels'])),
        diagnostic_complete_H8_H10_H12=sum(rows[k]['study_role']=='diagnostic' for k in
            set(stores['h8_labels'])&set(stores['h10_labels'])&set(stores['h12_labels'])),
        auxiliary_completed_reports=sum(len(a['history']) for stage in ('h8','h10','h12') for a in stores[stage+'_attempts'].values()),
        total_new_nodes=sum(a['work']['nodes'] for stage in ('h8','h10','h12') for a in stores[stage+'_attempts'].values())))
    actual={r['state_key']:r for r in report['cases']};direct=literal=depth_count=pair_count=0
    control_path=ROOT/'data/quality-train-v1/model.json';pilot_path=ROOT/'data/value-pilot-v1/model.json'
    models={'control':load(control_path)['model'],'pilot':load(pilot_path)['model']}
    with CpuBatchEvaluator(control_path,cache_capacity=0,max_batch=1,backend='scalar') as control, \
         CpuBatchEvaluator(pilot_path,cache_capacity=0,max_batch=1,backend='scalar') as pilot:
        test.assertEqual(report['model_identities'],dict(control=control.identity,pilot=pilot.identity))
        for key,row in rows.items():
            case=actual[key];state=row['state'];actor=state['controller'];sign=1 if actor==0 else -1
            legal=[p for p in range(7) if row['legal_mask']&(1<<p)]
            for field in ('state','family','cohort','study_role','band','stratum','active','source_root_sha256'):
                test.assertEqual(case[field],row[field])
            test.assertEqual(case['root_sha256'],config.digest(row));test.assertEqual(case['actor'],actor)
            test.assertEqual(case['legal_actions'],legal)
            predictions={};actions={}
            for name,evaluator in (('control',control),('pilot',pilot)):
                prediction=evaluator.evaluate_many([state])[0]
                prediction={k:v for k,v in prediction.items() if k!='inference_ms'}
                test.assertEqual(case['analysis_model_forwards'][name],prediction)
                value,_,logits=value_and_policy(norm(state),models[name]);literal+=1;direct+=1
                test.assertAlmostEqual(prediction['value_player0'],value,delta=1e-10)
                test.assertLessEqual(max(abs(a-b) for a,b in zip(prediction['policy_logits'],logits)),1e-10)
                test.assertEqual(prediction['native_actual_evals'],1)
                test.assertNotIn('inference_ms',case['analysis_model_forwards'][name])
                predictions[name]=prediction;actions[name]=min(legal,key=lambda p:(-logits[p],p))
            test.assertEqual(predictions['control']['policy_logits'],predictions['pilot']['policy_logits'])
            available=[];expected_attempts={}
            for stage in ('h8','h10','h12'):
                attempt=stores[stage+'_attempts'].get(key);item=stores[stage+'_labels'].get(key)
                if attempt is None:
                    expected_attempts[stage]=dict(eligible=False,status='out_of_scope')
                    test.assertNotIn(stage[1:],case['by_requested_depth']);continue
                depth=attempt['requested_depth']
                expected_attempts[stage]=dict(eligible=True,status=attempt['status'],requested_depth=depth,
                    completed_depth=attempt['completed_depth'],requested_label_admitted=item is not None,
                    auxiliary_completed_depths=[r['choice_depth'] for r in attempt['history'] if r['choice_depth']!=depth],
                    node_budget={'h8':2000000,'h10':5000000,'h12':10000000}[stage],
                    actual_work=attempt['work'],attempt_sha256=config.digest(attempt))
                if item is None:
                    test.assertNotIn(str(depth),case['by_requested_depth']);continue
                available.append(depth);label=item['label'];entry=case['by_requested_depth'][str(depth)];depth_count+=1
                expected=dict(label_sha256=config.digest(item),target_player0=label['target_player0'],
                    best_moves=label['best_moves'],display_best_pits=[p+1 for p in label['best_moves']],
                    action_scores=label['action_scores'])
                for model,pred in predictions.items():
                    delta=pred['value_player0']-label['target_player0']
                    expected[model]=dict(value_player0=pred['value_player0'],value_error_player0=delta,
                        value_error_actor=sign*delta,absolute_value_error_seeds=abs(delta),
                        raw_policy=regret(label,actions[model],actor))
                expected['saved_actions']=None if row['study_role']!='diagnostic' else {
                    name:regret(label,row['root']['provenance'][name+'_chosen'],actor) for name in ('control','pilot')}
                test.assertEqual(entry,expected)
            test.assertEqual(case['attempts'],expected_attempts)
            test.assertEqual(case['requested_depths_available'],available)
            test.assertEqual(set(case['by_requested_depth']),{str(d) for d in available})
            for a,b in ((8,10),(10,12),(8,12)):
                left=stores[f'h{a}_labels'].get(key);right=stores[f'h{b}_labels'].get(key)
                expected=relation(left,right) if left is not None and right is not None else None
                test.assertEqual(case['pairs'][f'{a}_{b}'],expected)
                pair_count+=expected is not None
            reference=stores['old_h8_references'].get(key)
            test.assertEqual(case['old_H8_reference'],None if reference is None else dict(
                source=reference['source'],label_sha256=reference['label_sha256'],old_label_unchanged=True,
                requested_H8_same_Q_best=key in stores['h8_labels'] and all(
                    stores['h8_labels'][key]['label'][name]==reference['label'][name]
                    for name in ('target_player0','best_moves','action_scores'))))
            if row['study_role']=='diagnostic':
                p=row['root']['provenance']
                test.assertEqual(case['saved_first_divergence'],dict(pair=p['pair'],candidate_seat=p['candidate_seat'],
                    first_different_choice=p['first_different_choice'],actor_role='candidate' if actor==p['candidate_seat'] else 'opponent',
                    original_state_unchanged=True,control_game_action=p['control_chosen'],pilot_game_action=p['pilot_chosen'],
                    action_names_refer_to_games=True,opponent_first=actor!=p['candidate_seat']))
            else:test.assertNotIn('saved_first_divergence',case)
    groups={name:[r for r in actual.values() if r['study_role']+'/'+r['cohort']==name]
        for name in {r['study_role']+'/'+r['cohort'] for r in actual.values()}}
    test.assertEqual(report['groups'],{name:summary(values) for name,values in groups.items()})
    # Explicit censoring/eligibility probe: one complete diagnostic, one censored
    # diagnostic and one fresh root whose H12 is outside the declared scope.
    tool=module('midgame_comparison_denominator_audit','tools/analyze-midgame-label-pilot.py')
    proto=next(r for r in actual.values() if len(r['requested_depths_available'])==3)
    import copy
    paired=copy.deepcopy(proto);censored=copy.deepcopy(proto);fresh=copy.deepcopy(proto)
    censored['requested_depths_available']=[8];censored['by_requested_depth']={'8':censored['by_requested_depth']['8']}
    censored['pairs']={name:None for name in ('8_10','10_12','8_12')}
    for stage in ('h10','h12'):
        censored['attempts'][stage].update(status='budget_rejected',requested_label_admitted=False,
            completed_depth=8,auxiliary_completed_depths=list(range(1,9)))
    fresh['requested_depths_available']=[8,10];fresh['by_requested_depth'].pop('12')
    fresh['attempts']['h12']=dict(eligible=False,status='out_of_scope')
    fresh['pairs']['10_12']=None;fresh['pairs']['8_12']=None
    fixture=[paired,censored,fresh]
    test.assertEqual(tool.summarize(fixture),summary(fixture))
    test.assertEqual(tool.summarize(fixture)['requested_depths']['12']['missing_labels'],1)
    test.assertEqual(tool.summarize(fixture)['requested_depths']['12']['out_of_scope_roots'],1)
    test.assertEqual(report['analysis_model_forwards'],2*len(rows))
    for flag in ('new_forward_timing_measured','training_performed','new_games','new_teacher_searches',
                 'weights_applied','data_ingested','torch_imported','sealed_final_test_consumed'):test.assertFalse(report[flag])
    for flag in ('all_fixed_policy_logits_equal','diagnostic_original_state_orientation',
                 'comparisons_use_complete_requested_intersections','no_auxiliary_fallback',
                 'old_Q_reference_compared_in_original_orientation','deeper_is_not_game_truth',
                 'saved_action_regret_is_not_causal_game_error'):test.assertTrue(report[flag])
    return dict(cases=len(rows),direct_native_forwards=direct,literal_feature_and_dense_forwards=literal,
        requested_label_comparisons=depth_count,complete_horizon_pairs=pair_count,
        all_same_state_actor_errors_policy_and_saved_action_regrets_rederived=True,
        all_group_denominators_eligibility_and_out_of_scope_counts_rederived=True,
        synthetic_censoring_and_out_of_scope_probe_passed=True,
        all_requested_only_no_auxiliary_fallback=True,no_causal_or_strength_claim=True)
