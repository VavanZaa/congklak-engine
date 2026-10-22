"""Offline probability objectives for fixed-model ordinary rollout observations.

All probabilities are proposals with unit per-draw loss.  No optimizer or sampler
is run here; 25,600 expected draws are a common hypothetical comparison scale.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import ExitStack
import math
from pathlib import Path
import sys
from . import config
from .native import ROOT,Native
from .ordinary_selfplay_review import baseline,require,sha,read,relative

OUTPUT=ROOT/'build/ordinary-selfplay-sampling-review.json'
CLI=ROOT/'tools/review-ordinary-selfplay-sampling.py'
HYPOTHETICAL_DRAWS=25600
MIXTURE_ROOT_FRACTION=.9
MODES=('observation_uniform','family_uniform','raw_state_uniform','stratum_uniform','root90_stratum10')


def probabilities(rows,mode):
    require(mode in MODES,'Unknown offline distribution')
    if not rows:return []
    groups={field:Counter(row[field] for row in rows) for field in ('family','state_key','stratum')}
    def uniform(field):return [1/(len(groups[field])*groups[field][row[field]]) for row in rows]
    if mode=='observation_uniform':answer=[1/len(rows)]*len(rows)
    elif mode=='family_uniform':answer=uniform('family')
    elif mode=='raw_state_uniform':answer=uniform('state_key')
    elif mode=='stratum_uniform':answer=uniform('stratum')
    else:answer=[MIXTURE_ROOT_FRACTION*a+(1-MIXTURE_ROOT_FRACTION)*b for a,b in zip(uniform('state_key'),uniform('stratum'))]
    require(all(math.isfinite(p) and p>0 for p in answer) and abs(math.fsum(answer)-1)<1e-12,'Offline probabilities do not conserve positive support')
    return answer


def inverse_concentration(probabilities):
    return 1/math.fsum(p*p for p in probabilities) if probabilities else None


def exposure(rows,mode,draws=HYPOTHETICAL_DRAWS):
    require(type(draws) is int and draws>0,'Invalid hypothetical comparison scale');p=probabilities(rows,mode)
    masses={};counts={}
    for field in ('family','state_key','stratum'):
        sums=defaultdict(list);n=Counter()
        for row,prob in zip(rows,p):sums[row[field]].append(prob);n[row[field]]+=1
        masses[field]={key:math.fsum(values) for key,values in sorted(sums.items())};counts[field]=dict(sorted(n.items()))
    def category(field):
        q=masses[field]
        return dict(support=len(q),distribution_effective_support=inverse_concentration(q.values()),minimum_probability=min(q.values()) if q else None,maximum_probability=max(q.values()) if q else None,groups={key:dict(observations=counts[field][key],probability=value,hypothetical_expected_draws=value*draws) for key,value in q.items()})
    rare=[key for key,count in counts['stratum'].items() if count==min(counts['stratum'].values())] if rows else []
    return dict(mode=mode,observations=len(rows),probability_sum=math.fsum(p),observation_distribution_effective_support=inverse_concentration(p),minimum_observation_probability=min(p) if p else None,maximum_observation_probability=max(p) if p else None,maximum_observation_hypothetical_expected_draws=max(p)*draws if p else None,
        raw_states=category('state_key'),families=category('family'),strata=category('stratum'),rarest_observed_strata=rare,rarest_observed_strata_probability=math.fsum(masses['stratum'][key] for key in rare),
        observation_exposures=[dict(id=row['id'],family=row['family'],state_key=row['state_key'],stratum=row['stratum'],sampling_probability=prob,hypothetical_expected_draws=prob*draws,loss_weight=1) for row,prob in zip(rows,p)])


def pool_summary(rows):
    return dict(observations=len(rows),unique_raw_states=len({r['state_key'] for r in rows}),families=len({r['family'] for r in rows}),strata=dict(sorted(Counter(r['stratum'] for r in rows).items())),initial_choice0_observations=sum(r['choice']==0 for r in rows),descendant_observations=sum(r['choice']>0 for r in rows))


def fixed_model_predictions(rows):
    """Pure CPU forwards; compare noisy source-policy returns, not teacher truth."""
    from .puct_cpu_inference import CpuBatchEvaluator
    models={'control':ROOT/'data/quality-train-v1/model.json','challenger':ROOT/'data/selfplay-training-pilot-v1/model.json'}
    unique={row['state_key']:row['state'] for row in rows};outputs={};identities={}
    with ExitStack() as stack:
        evaluators={name:stack.enter_context(CpuBatchEvaluator(path,cache_capacity=0,max_batch=1,backend='scalar')) for name,path in models.items()}
        for name,evaluator in evaluators.items():
            outputs[name]={key:{k:v for k,v in evaluator.evaluate_many([state])[0].items() if k!='inference_ms'} for key,state in sorted(unique.items())};identities[name]=evaluator.identity
            require(all(value['native_actual_evals']==1 and value['cache_hit'] is False for value in outputs[name].values()),'Descriptive forward accounting differs')
    metrics=[]
    for row in rows:
        target=row['terminal_return_player0'];sign=1 if row['actor']==0 else -1;per_model={}
        for name,values in outputs.items():
            prediction=values[row['state_key']];logits=prediction['policy_logits'];legal=[i for i in range(7) if row['legal_mask']&(1<<i)];maximum=max(logits[i] for i in legal);denom=math.fsum(math.exp(logits[i]-maximum) for i in legal)
            policy=[math.exp(logits[i]-maximum)/denom if i in legal else 0 for i in range(7)]
            cross_entropy=-math.fsum(row['policy_target'][i]*math.log(max(policy[i],1e-300)) for i in legal)
            per_model[name]=dict(value_player0=prediction['value_player0'],absolute_error_to_own_rollout_return=abs(prediction['value_player0']-target),signed_actor_error_to_own_rollout_return=sign*(prediction['value_player0']-target),legal_policy_cross_entropy_to_saved_visits=cross_entropy)
        metrics.append(dict(id=row['id'],family=row['family'],state_key=row['state_key'],choice=row['choice'],stratum=row['stratum'],terminal_return_player0=target,models=per_model))
    def summarize(values):
        result=dict(observations=len(values),families=len({r['family'] for r in values}))
        for name in models:
            errors=[r['models'][name]['absolute_error_to_own_rollout_return'] for r in values];ces=[r['models'][name]['legal_policy_cross_entropy_to_saved_visits'] for r in values]
            families=sorted({r['family'] for r in values});family_errors=[math.fsum(r['models'][name]['absolute_error_to_own_rollout_return'] for r in values if r['family']==family)/sum(r['family']==family for r in values) for family in families]
            result[name]=dict(row_weighted_mae=math.fsum(errors)/len(errors) if errors else None,family_macro_mae=math.fsum(family_errors)/len(family_errors) if family_errors else None,row_weighted_policy_cross_entropy=math.fsum(ces)/len(ces) if ces else None)
        return result
    return dict(model_identities=identities,unique_forward_states=len(unique),actual_CPU_forwards=2*len(unique),new_forward_timing_measured=False,forwards=outputs,per_observation=metrics,
        all_rows=summarize(metrics),initial_choice0=summarize([row for row in metrics if row['choice']==0]),descendants=summarize([row for row in metrics if row['choice']>0]),by_stratum={key:summarize([r for r in metrics if r['stratum']==key]) for key in sorted({r['stratum'] for r in metrics})},
        interpretation='Both fixed models are evaluated against dependent terminal returns and noisy visit targets generated by the control policy, not minimax values or independent validation truth. These are unfitted descriptive errors, not training gains or proof of stronger play.')


def build():
    from . import ordinary_selfplay_pilot as producer,selfplay_training_pilot as old
    prior=baseline();collection_roots,collection=producer.load_rows();manifest,rows=producer.load_replay();old_manifest,old_rows=old.load_replay()
    require(all(row['purpose']=='ordinary_selfplay_training_candidate' and row.get('cohort','training_candidate')=='training_candidate' and row['stratum'] in ('25-49/ordinary','50-74/ordinary') and row['is_start_record']==(row['choice']==0) and sum(map(sum,row['state']['pits']))>=25 for row in rows),'Nontraining or nonordinary observation entered offline pool')
    roots_by_key=collection_roots if isinstance(collection_roots,dict) else {row['state_key']:row for row in collection_roots}
    for row in rows:
        root=roots_by_key[row['selected_start_state_key']]
        require(root['cohort']=='training_candidate' and root['family']==row['family'] and config.digest(root)==row['selected_start_root_sha256'],'Retained observation has no admitted training origin')
    old_ordinary=[row for row in old_rows if row['stratum'].endswith('/ordinary')]
    pools={'combined_admitted':rows,'paired_initial_choice0':[r for r in rows if r['choice']==0],'ordinary_descendants':[r for r in rows if r['choice']>0]}
    proposals={name:dict(support=pool_summary(values),distributions={mode:exposure(values,mode) for mode in MODES}) for name,values in pools.items()}
    original_distribution=exposure(old_rows,'family_uniform');original_ordinary_mass=math.fsum(row['sampling_probability'] for row in original_distribution['observation_exposures'] if row['stratum'].endswith('/ordinary'))
    inputs=dict(prior['inputs'])
    paths=[Path(__file__),CLI,ROOT/'build/ordinary-selfplay-baseline.json',ROOT/'congklak/ordinary_selfplay_review.py',ROOT/'build/ordinary-selfplay-prior-arena-review.json',producer.DEFAULT/'work.sqlite3',producer.DEFAULT/'collection-manifest.json',producer.DEFAULT/'replay-manifest.json',producer.DEFAULT/'rollout-protection.json',producer.COLLECTION,producer.PROTECTION,producer.OUTPUT,producer.PLAN,producer.REGISTRY,producer.IMPLEMENTATION,ROOT/'congklak/ordinary_selfplay_pilot.py',ROOT/'tools/ordinary-selfplay-pilot.py',old.DEFAULT/'work.sqlite3',old.DEFAULT/'selfplay-manifest.json',ROOT/'build/selfplay-pilot-sampling-review.json',ROOT/'congklak/puct_cpu_inference.py']
    # Identity-selected paths are supplied by the closed producer, not guessed
    # final model/training outputs: this scope never has an optimizer catalog.
    for path in paths:
        require(path.is_file(),'Missing closed sampling input: '+str(path));inputs[relative(path)]=sha(path)
    descriptions=fixed_model_predictions(rows)
    paired_returns=[]
    for key,item in sorted(manifest['summary']['usable_starts'].items()):
        values=[row for row in rows if row['selected_start_state_key']==key and row['choice']==0]
        if item['usable']:
            require(len(values)==2 and {row['id'] for row in values}==set(item['required_record_ids']),'Paired initial records differ')
        returns=[row['terminal_return_player0'] for row in values]
        paired_returns.append(dict(selected_start_state_key=key,family=item['family'],stratum=item['stratum'],usable=item['usable'],retained_initial_observations=len(values),terminal_returns_player0=returns,terminal_return_spread=max(returns)-min(returns) if returns else None))
    for name,digest in inputs.items():require(sha(ROOT/name)==digest,'Input changed during offline probability review: '+name)
    require('torch' not in sys.modules,'Offline review imported Torch')
    return dict(schema='ordinary-selfplay-offline-sampling-review-v1',passed=True,prior_inputs_preserved=3171,inputs=dict(sorted(inputs.items())),counts=dict(selected_roots=len(collection_roots),admitted_observations=len(rows),new_unique_raw_states=len({row['state_key'] for row in rows}),new_families=len({row['family'] for row in rows}),historical_ordinary_observations=len(old_ordinary),historical_ordinary_unique_raw_states=len({row['state_key'] for row in old_ordinary})),
        collection=collection,replay_manifest=manifest,historical_support=dict(all=pool_summary(old_rows),ordinary=pool_summary(old_ordinary),family_uniform_ordinary_probability=original_ordinary_mass,observed_historical_optimizer_draws=read(ROOT/'build/selfplay-pilot-sampling-review.json')['distributions']['stratum'],historical_draws=25600),
        hypothetical_draws=HYPOTHETICAL_DRAWS,actual_new_optimizer_draws=0,proposals=proposals,paired_initial_return_variability=paired_returns,fixed_model_own_return_diagnostics=descriptions,
        objective_definitions=dict(observation_uniform='Each retained observation gets equal probability.',family_uniform='Each admitted origin family gets equal mass, then each retained observation in that family is uniform.',raw_state_uniform='Each distinct literal full state gets equal mass, then repeated observations of that state are uniform.',stratum_uniform='Each nonempty actual row stratum gets equal mass, then observations within that stratum are uniform.',root90_stratum10='Exactly0.9 raw-state-uniform +0.1 equal-nonempty-stratum. This is a conservative comparison proposal, not a selected winner.'),
        interpretation=dict(objective='Unit loss per hypothetical draw under every proposal. Changing q changes the optimized population risk; no importance correction or second label weight is applied or implicitly claimed.',support='Distribution effective support is inverse concentration1/sum(p^2) over observations, raw states or origin families. It is not gradient ESS, independent games, or uncertainty.',rare='All admitted observations keep positive probability; minimum and rarest-observed-stratum mass are reported. Unobserved states cannot be recovered by weighting.',comparability='25,600 expected draws are hypothetical for comparison with the earlier accepted pilot; no new optimizer draws were made. Old actual draws are retained historical evidence.',ownership='Validation roots and every protected eligible validation-pool component are excluded by the closed producer before rollouts/admission; this review does not reinterpret ownership.',recommendation='No proposal is applied or selected as a strength winner. Review initial-root and descendant pools separately; repeated source-policy returns and finite dependent trajectories cannot justify blind longer training.'),
        training_performed=False,sampling_applied=False,model_changed=False,source_rows_pruned=False,new_teacher_searches=0,new_strength_games=0,default_promotion=False,torch_imported=False)


def main():
    from . import quality_search
    from .neural_relay import NeuralNative
    from . import selfplay_pilot_benchmark as bench
    def forbidden(*_args,**_kwargs):raise AssertionError('Offline sampling review must not search or optimize')
    saved=(Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide)
    Native.analyze=NeuralNative.analyze=quality_search.search=bench.decide=forbidden
    try:body=build()
    finally:Native.analyze,NeuralNative.analyze,quality_search.search,bench.decide=saved
    from .deeper_teacher_v2 import atomic_json
    expected=config.canonical(body)+'\n'
    if OUTPUT.exists():require(OUTPUT.read_text(encoding='utf-8')==expected,'Immutable offline sampling review differs')
    else:atomic_json(OUTPUT,body)
    print(config.canonical(dict(event='ordinary_selfplay_sampling_review_complete',passed=True,counts=body['counts'],actual_new_optimizer_draws=0)),flush=True)
    return body
