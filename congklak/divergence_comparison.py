"""Deterministic forensic comparisons, without any search or training entry point."""
from pathlib import Path
import sys
from . import config, divergence_study_common as common
from . import divergence_search as search, divergence_teacher as teacher

ROOT=common.ROOT
OUTPUT=ROOT/'build/divergence-comparison.json'
IMPLEMENTATION=ROOT/'build/divergence-comparison-implementation.json'
SCHEMA='five-divergence-action-regret-comparison-v1'
SOURCES=('congklak/divergence_comparison.py','tools/compare-divergence.py',
         'congklak/divergence_study_common.py','docs/DIVERGENCE_STUDY_SCOPE.md')
require=common.require


def implementation():
    return dict(schema=SCHEMA+'-implementation',sources=common.pins(SOURCES),
                fixed_inputs=common.pins([common.BASELINE,common.PANEL,search.PLAN,search.IMPLEMENTATION,
                                          teacher.REGISTRY,teacher.PLAN,teacher.IMPLEMENTATION]),
                contract='All five raw roots, all four simulation budgets, each requested horizon separately; H10 primary. Actor-signed seed regret with full best-set ties; no fallback or causal/strength inference.')


def freeze():
    return common.publish(IMPLEMENTATION,implementation())


def regret(action,q,actor):
    if q is None:return None
    sign=1 if actor==0 else -1
    best=max(sign*value for value in q.values())
    return best-sign*q[action]


def build():
    require('torch' not in sys.modules,'Comparison must remain CPU-only')
    require(config.canonical(common.read(IMPLEMENTATION))==config.canonical(implementation()),'Comparison source freeze changed')
    panel=common.load_panel();sr,cells,sm=search.load_cells();rows,stores,tm=teacher.load_rows()
    require(config.canonical(sr)==config.canonical(panel['roots']) and set(rows)=={r['state_key'] for r in sr},'Comparison exact support differs')
    search.primary_witness()
    index={(c['unit']['state_key'],c['unit']['simulations'],c['unit']['profile']):c for c in cells}
    results=[];metrics=[];shifts=[]
    for root in sr:
        key=root['state_key'];actor=root['actor'];horizons=[];fixed=[]
        for stage in teacher.STAGES:
            attempt=stores[stage+'_attempts'][key];item=stores[stage+'_labels'].get(key)
            q={a['pit']:a['value_player0'] for a in item['label']['action_scores']} if item else None
            if q is not None:require(set(q)==set(root['legal_actions']),'Missing or extra action Q')
            best=sorted(a for a in q if regret(a,q,actor)==0) if q is not None else None
            if item:require(best==sorted(item['label']['best_moves']),'Actor-signed teacher best set differs')
            horizons.append(dict(stage=stage,requested_depth=attempt['requested_depth'],status=attempt['status'],
                completed_depth=attempt['completed_depth'],nodes=attempt['work']['nodes'],
                requested_action_Q_player0=[dict(pit=a,value_player0=q[a]) for a in sorted(q)] if q is not None else None,
                best_moves=best,missing_actions=len(root['legal_actions']) if q is None else 0,
                timed_control_regret=regret(root['control_action'],q,actor),timed_candidate_regret=regret(root['candidate_action'],q,actor),
                retained_completed_depths=[x['choice_depth'] for x in attempt['history']]))
        for budget in (128,256,512,1024):
            models={}
            for profile in search.PROFILES:
                cell=index[key,budget,profile];report=cell['report']
                require(cell['status']=='complete','Incomplete fixed-work cell')
                models[profile]=dict(chosen=report['chosen'],visits=report['visits'],root_priors=report['root_priors'],
                    sampled_action_Q_player0=report['action_mean_value_player0'],parent_bootstrap_player0=report['unvisited_root_q_player0'],
                    completed_simulations=report['completed_simulations'],physical_forwards=cell['counts']['native_actual_evals'],
                    matches_saved_timed_action=report['chosen']==root[profile+'_action'])
            refs=[]
            for h in horizons:
                q={a['pit']:a['value_player0'] for a in h['requested_action_Q_player0']} if h['requested_action_Q_player0'] is not None else None
                cr=regret(models['control']['chosen'],q,actor);vr=regret(models['candidate']['chosen'],q,actor)
                refs.append(dict(stage=h['stage'],requested_depth=h['requested_depth'],eligible=q is not None,
                    control_regret=cr,candidate_regret=vr,candidate_minus_control_regret=vr-cr if q is not None else None,
                    control_in_best_set=cr==0 if q is not None else None,candidate_in_best_set=vr==0 if q is not None else None))
            fixed.append(dict(simulations=budget,models=models,same_action=models['control']['chosen']==models['candidate']['chosen'],references=refs))
        for left,right in zip(horizons,horizons[1:]):
            eligible=left['best_moves'] is not None and right['best_moves'] is not None
            shifts.append(dict(state_key=key,left=left['stage'],right=right['stage'],eligible=eligible,
                best_sets_disjoint=not bool(set(left['best_moves'])&set(right['best_moves'])) if eligible else None))
        results.append(dict(id=root['id'],state_key=key,actor=actor,legal_actions=root['legal_actions'],
            timed_control_action=root['control_action'],timed_candidate_action=root['candidate_action'],timed_margin_delta=root['timed_margin_delta'],
            horizons=horizons,fixed_work=fixed))
    for stage in teacher.STAGES:
        for budget in (128,256,512,1024):
            refs=[next(h for h in next(f for f in r['fixed_work'] if f['simulations']==budget)['references'] if h['stage']==stage) for r in results]
            valid=[x for x in refs if x['eligible']]
            metrics.append(dict(stage=stage,simulations=budget,requested_roots=5,eligible_roots=len(valid),missing_roots=5-len(valid),
                control_regret_sum=sum(x['control_regret'] for x in valid),candidate_regret_sum=sum(x['candidate_regret'] for x in valid),
                control_best_count=sum(x['control_in_best_set'] for x in valid),candidate_best_count=sum(x['candidate_in_best_set'] for x in valid),
                candidate_lower_regret=sum(x['candidate_minus_control_regret']<0 for x in valid),same_regret=sum(x['candidate_minus_control_regret']==0 for x in valid),
                candidate_higher_regret=sum(x['candidate_minus_control_regret']>0 for x in valid),
                different_actions_denominator=5,different_actions=sum(not next(f for f in r['fixed_work'] if f['simulations']==budget)['same_action'] for r in results)))
    inputs=common.pins([common.BASELINE,common.PANEL,IMPLEMENTATION,search.PLAN,search.IMPLEMENTATION,search.OUTPUT,search.RESOURCES,
        search.DEFAULT/'work.sqlite3',search.DEFAULT/'manifest.json',teacher.REGISTRY,teacher.PLAN,teacher.IMPLEMENTATION,teacher.OUTPUT,
        teacher.DEFAULT/'work.sqlite3',teacher.DEFAULT/'manifest.json',common.CONTROL,common.CANDIDATE,*SOURCES])
    return dict(schema=SCHEMA,passed=True,roots=results,metrics=metrics,horizon_shifts=shifts,inputs=inputs,
        primary_horizon='h10',fixed_cells=40,completed_root_simulations=sm['counts']['completed_simulations'],teacher_attempts=tm['requested_attempts'],
        teacher_total_nodes=tm['total_new_nodes'],summary='Five fixed diagnostic roots: all simulation budgets and requested H8/H10/H12 action references reported; H10 primary. Models/policy fixed; training remains held.',
        policy_fixed=True,all_model_weights_fixed=True,training_performed=False,new_games=0,promotion=False,previous_allocation='hold',
        limits=['All five selected actors are player1; no actor0 or unseen-release generalization.',
                'Actor-signed finite-horizon regret is in seeds. Ties count as best; incomplete requested references remain missing.',
                'Sampled PUCT Q, neural bootstrap and finite-horizon teacher Q are distinct. No cross-horizon fallback.',
                'Fixed simulations do not prove equal-time strength, energy savings or the cause of a later full-game loss.'])


def run():return common.publish(OUTPUT,build())


def verify():
    actual=common.read(OUTPUT);require(config.canonical(actual)==config.canonical(build()),'Comparison does not rederive');return actual
