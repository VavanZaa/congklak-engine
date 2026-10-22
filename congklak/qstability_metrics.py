"""Exact prospective max-action-Q weighting; pure standard-library arithmetic."""
from fractions import Fraction as F
import hashlib
import json

SCHEMA='qstability-max-action-fixed-cell-exact-metrics-v1'
STRATA=('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
CELLS=tuple(s+'/actor'+str(a) for s in STRATA for a in (0,1))
RULE='max(1/4,1/(1+max_legal_abs(Q10-Q8)/4))'
METRICS=('H10_best_H12_regret_mean','root_drift','action_Q_drift','H10_best_H12_regret_min','H10_best_H12_regret_max','H10_lexicographic_H12_regret','max_action_Q_drift','H10_best_action_Q_drift_contribution','H10_nonbest_action_Q_drift_contribution')


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(body):return json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(body):return hashlib.sha256(canonical(body).encode()).hexdigest()
def same(a,b):return canonical(a)==canonical(b)
def integer(x):return type(x) is int


def rational(value):
    require(type(value) is F or integer(value),'Exact arithmetic requires an integer or Fraction')
    q=F(value);return dict(numerator=q.numerator,denominator=q.denominator,value=float(q))


def fraction(body):
    require(isinstance(body,dict) and set(body)=={'numerator','denominator','value'},'Malformed rational')
    require(integer(body['numerator']) and integer(body['denominator']) and body['denominator']>0,'Rational identity requires exact integers')
    q=F(body['numerator'],body['denominator']);require(same(body,rational(q)),'Noncanonical rational/display');return q


def rawkey(state):
    require(isinstance(state,dict) and set(state)=={'pits','stores','controller'},'Unknown full state')
    require(type(state['controller']) is int and state['controller'] in (0,1),'Invalid controller')
    require(isinstance(state['pits'],list) and len(state['pits'])==2 and all(isinstance(v,list) and len(v)==7 for v in state['pits']),'Invalid pit rows')
    require(isinstance(state['stores'],list) and len(state['stores'])==2,'Invalid stores')
    cells=state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1]]
    require(all(integer(v) and 0<=v<=98 for v in cells) and sum(cells)==98,'Invalid seed conservation')
    return bytes(cells+[state['controller']]).hex()


def validate_rows(rows):
    require(isinstance(rows,dict) and len(rows)==16,'Assessment requires exactly16 roots')
    counts={cell:0 for cell in CELLS};families=set();components=set()
    for key,row in rows.items():
        require(isinstance(row,dict) and key==row.get('state_key')==rawkey(row['state']),'Root/raw state identity differs')
        actor=row['state']['controller'];require(type(row['actor']) is int and row['actor']==actor,'Root actor differs')
        require(row['stratum'] in STRATA and row['group']=='assessment','Unknown stratum/assessment role')
        cell=row['stratum']+'/actor'+str(actor);require(row['cell']==cell,'Declared cell differs');counts[cell]+=1
        active=sum(map(sum,row['state']['pits']));lower,upper=map(int,row['stratum'].split('/')[0].split('-'));require(lower<=active<=upper,'Root active band differs')
        legal=[a for a,n in enumerate(row['state']['pits'][actor]) if n]
        require(len(legal)>=2 and type(row['legal_mask']) is int and row['legal_mask']==sum(1<<a for a in legal),'Root legal support differs')
        for field,seen in (('family',families),('component',components)):
            require(isinstance(row[field],str) and row[field] and row[field] not in seen,'Selected origin/component must be distinct');seen.add(row[field])
    require(all(n==2 for n in counts.values()),'Exactly2 roots are required in every declared cell');return rows


def _label(item,row,depth):
    require(isinstance(item,dict) and isinstance(item.get('label'),dict),'Missing requested label')
    label=item['label'];require(type(label['choice_depth']) is int and label['choice_depth']==depth,'Auxiliary/wrong requested horizon cannot substitute')
    require(type(label['target_player0']) is int,'Integer finite-reference target required');scores=label['action_scores']
    require(isinstance(scores,list) and all(isinstance(v,dict) and type(v.get('pit')) is int and type(v.get('value_player0')) is int for v in scores),'Integer full-action Q required')
    q={v['pit']:v['value_player0'] for v in scores};legal=[a for a in range(7) if row['legal_mask']&(1<<a)]
    require(len(q)==len(scores) and set(q)==set(legal),'Requested full-action support differs')
    actor=row['actor'];best_value=(max if actor==0 else min)(q.values());best=[a for a in legal if q[a]==best_value]
    require(same(label['best_moves'],best) and label['target_player0']==best_value,'Requested best value/full tie set differs')
    difference=row['state']['stores'][0]-row['state']['stores'][1];active=sum(map(sum,row['state']['pits']))
    require(all(difference-active<=v<=difference+active for v in q.values()),'Reference Q exceeds seed bounds')
    return label,q,best


def raw_rule(row,h8,h10):
    a,qa,_=_label(h8,row,8);b,qb,_=_label(h10,row,10);legal=sorted(qa)
    deltas={p:qb[p]-qa[p] for p in legal};maximum=max(abs(v) for v in deltas.values());r=max(F(1,4),1/(1+F(maximum,4)))
    root=abs(b['target_player0']-a['target_player0']);action=F(sum(abs(v) for v in deltas.values()),len(legal));sign=1 if row['actor']==0 else -1
    require(root<=maximum and action<=maximum,'Full-Q supremum must bound extremum/mean shift')
    return dict(max_action_shift_seeds=maximum,max_shift_moves=[p for p in legal if abs(deltas[p])==maximum],
        action_shifts=[dict(pit=p,Q8_player0=qa[p],Q10_player0=qb[p],delta_player0=deltas[p],delta_actor=sign*deltas[p],absolute_delta=abs(deltas[p])) for p in legal],
        root_shift_seeds=root,mean_action_shift_seeds=rational(action),root_and_mean_are_rule_inputs=False,
        raw_weight=rational(r),formula=RULE,requested_depths=[8,10],H12_used=False)


def _masses(rows,probabilities):
    groups={'cells':CELLS,'strata':STRATA,'actors':(0,1)};out={}
    for name,keys in groups.items():
        field={'cells':'cell','strata':'stratum','actors':'actor'}[name]
        out[name]={str(key):rational(sum((probabilities[k] for k,row in rows.items() if row[field]==key),F())) for key in keys}
    out['ordinary']=rational(sum((probabilities[k] for k,row in rows.items() if row['stratum'].endswith('/ordinary')),F()))
    return out


def freeze_weights(rows,endpoints):
    validate_rows(rows);require(isinstance(endpoints,dict) and set(endpoints)==set(rows),'All16 same-root H8/H10 endpoints required')
    rules={}
    for key,row in rows.items():
        require(isinstance(endpoints[key],dict) and set(endpoints[key])=={'h8','h10'},'Only H8/H10 inputs may define weights')
        rules[key]=raw_rule(row,endpoints[key]['h8'],endpoints[key]['h10'])
    cell_sums={cell:sum((fraction(rules[k]['raw_weight']) for k,row in rows.items() if row['cell']==cell),F()) for cell in CELLS}
    weighted={key:F(1,8)*fraction(rules[key]['raw_weight'])/cell_sums[row['cell']] for key,row in rows.items()};uniform={key:F(1,16) for key in rows}
    require(sum(weighted.values())==sum(uniform.values())==1,'Frozen probabilities must sum exactly1')
    entries={key:dict(root_sha256=digest(row),h8_item_sha256=digest(endpoints[key]['h8']),h10_item_sha256=digest(endpoints[key]['h10']),endpoints=endpoints[key],raw_rule=rules[key],weighted_probability=rational(weighted[key]),uniform_probability=rational(uniform[key])) for key,row in sorted(rows.items())}
    masses={name:_masses(rows,p) for name,p in (('uniform',uniform),('weighted',weighted))};require(same(masses['uniform'],masses['weighted']),'Fixed mixture changed')
    return dict(schema=SCHEMA+'-projection',rows_sha256=digest(rows),roots=16,per_cell=2,rows=entries,masses=masses,rule=RULE,normalization='(1/8)*raw_weight/sum_same_cell(raw_weight)',all_assessment=True,H12_used=False,weights_applied=False,
        objective='Declared8-cell assessment mixture; heuristic changes within-cell focus. Not a density ratio or calibrated confidence.',unit_loss='No sampler/loss changed. A future draw at these probabilities would use unit loss, not the same weights twice.')


def weight_projection(rows,h8_labels,h10_labels):
    require(set(h8_labels)==set(h10_labels)==set(rows),'All16 requested H8/H10 labels required')
    return freeze_weights(rows,{k:dict(h8=h8_labels[k],h10=h10_labels[k]) for k in rows})


def validate_projection(rows,projection):
    require(isinstance(projection,dict) and isinstance(projection.get('rows'),dict),'Missing frozen projection')
    endpoints={k:v['endpoints'] for k,v in projection['rows'].items()};require(same(projection,freeze_weights(rows,endpoints)),'Frozen weights/endpoint identities differ');return projection


def _root_result(key,row,frozen,h12):
    b,q10,best10=_label(frozen['endpoints']['h10'],row,10)
    base=dict(state_key=key,root_sha256=digest(row),cell=row['cell'],stratum=row['stratum'],actor=row['actor'],family=row['family'],component=row['component'],weighted_probability=frozen['weighted_probability'],uniform_probability=frozen['uniform_probability'],H12_complete=h12 is not None,
        h8_item_sha256=frozen['h8_item_sha256'],h10_item_sha256=frozen['h10_item_sha256'],h12_item_sha256=digest(h12) if h12 is not None else None,H10_best_moves=best10,H12_best_moves=None,best_sets_changed=None,best_sets_disjoint=None,regrets_by_H10_best_move=None,action_changes=None,ranking=None,metrics=None)
    if h12 is None:return base
    c,q12,best12=_label(h12,row,12);best=c['target_player0'];sign=1 if row['actor']==0 else -1;regrets={a:sign*(best-q12[a]) for a in best10}
    require(all(v>=0 for v in regrets.values()),'Actor regret became negative')
    metrics=dict(H10_best_H12_regret_mean=rational(F(sum(regrets.values()),len(regrets))),root_drift=rational(abs(best-b['target_player0'])),action_Q_drift=rational(F(sum(abs(q12[a]-q10[a]) for a in q10),len(q10))),
        H10_best_H12_regret_min=rational(min(regrets.values())),H10_best_H12_regret_max=rational(max(regrets.values())),H10_lexicographic_H12_regret=rational(regrets[min(best10)]),
        max_action_Q_drift=rational(max(abs(q12[a]-q10[a]) for a in q10)),
        H10_best_action_Q_drift_contribution=rational(F(sum(abs(q12[a]-q10[a]) for a in best10),len(q10))),
        H10_nonbest_action_Q_drift_contribution=rational(F(sum(abs(q12[a]-q10[a]) for a in q10 if a not in best10),len(q10))))
    deltas={a:q12[a]-q10[a] for a in q10};mean=F(sum(deltas.values()),len(deltas));wdiff=fraction(frozen['weighted_probability'])-fraction(frozen['uniform_probability'])
    actions=[dict(pit=a,Q10_player0=q10[a],Q12_player0=q12[a],Q10_actor=sign*q10[a],Q12_actor=sign*q12[a],delta_player0=deltas[a],delta_actor=sign*deltas[a],absolute_delta=abs(deltas[a]),centered_delta_player0=rational(deltas[a]-mean),
        H10_best=a in best10,H12_best=a in best12,H10_actor_gap=sign*(b['target_player0']-q10[a]),H12_actor_gap=sign*(best-q12[a]),
        H10_best_anchor_gap_changes=[dict(anchor=z,change_actor=sign*(deltas[z]-deltas[a])) for z in best10],
        global_weighted_minus_uniform_Q_drift=rational(wdiff*F(abs(deltas[a]),len(q10)))) for a in sorted(q10)]
    ranks=[]
    legal=sorted(q10)
    for index,a in enumerate(legal):
        for z in legal[index+1:]:
            old=sign*(q10[a]-q10[z]);new=sign*(q12[a]-q12[z])
            ranks.append(dict(actions=[a,z],H10_actor_difference=old,H12_actor_difference=new,gap_change_actor=new-old,strict_reversal=old*new<0,tie_changed=(old==0)!=(new==0)))
    ranking=dict(pairs=ranks,strict_reversals=sum(v['strict_reversal'] for v in ranks),tie_changes=sum(v['tie_changed'] for v in ranks),numeric_gap_changes=sum(v['gap_change_actor']!=0 for v in ranks),
        action_shift_mean_player0=rational(mean),action_shift_mean_actor=rational(sign*mean),interpretation='Signed Q/rank diagnostics only; finite H12 is not game truth. Nonbest contributions retain the full legal-action denominator.')
    return dict(base,H12_best_moves=best12,best_sets_changed=best10!=best12,best_sets_disjoint=not(set(best10)&set(best12)),regrets_by_H10_best_move=[dict(pit=a,regret_seeds=regrets[a]) for a in best10],action_changes=actions,ranking=ranking,metrics=metrics)


def _scope(keys,results):
    complete=[k for k in keys if results[k]['H12_complete']];output=dict(selected_roots=len(keys),complete_roots=len(complete),missing_roots=len(keys)-len(complete),selected_state_keys=keys,complete_state_keys=complete,uniform={},weighted={})
    for name in ('uniform','weighted'):
        p={key:fraction(results[key][name+'_probability']) for key in keys};mass=sum(p.values(),F());covered=sum((p[k] for k in complete),F());missing=mass-covered
        ess=mass*mass/sum((v*v for v in p.values()),F()) if p else None
        full={metric:rational(sum((p[k]*fraction(results[k]['metrics'][metric]) for k in keys),F())/mass) if len(complete)==len(keys) and mass else None for metric in METRICS}
        partial={metric:rational(sum((p[k]*fraction(results[k]['metrics'][metric]) for k in complete),F())) for metric in METRICS}
        output[name]=dict(**full,original_selected_mass=rational(mass),original_complete_mass=rational(covered),original_missing_mass=rational(missing),ESS=rational(ess) if ess is not None else None,max_conditional_probability=rational(max(p.values())/mass) if p and mass else None,
            complete_risk=full,covered_unnormalized_totals=partial,conditional_complete_diagnostic={metric:rational(fraction(partial[metric])/covered) if covered else None for metric in METRICS},conditional_diagnostic_warning='Explicit covered-support diagnostic only; not the fixed full-cell objective. No missing mass is redistributed.')
    output['weighted_minus_uniform']={metric:rational(fraction(output['weighted']['complete_risk'][metric])-fraction(output['uniform']['complete_risk'][metric])) if output['weighted']['complete_risk'][metric] is not None else None for metric in METRICS}
    return output


def _nonincrease(scope,metric):
    u=scope['uniform']['complete_risk'][metric];w=scope['weighted']['complete_risk'][metric]
    return fraction(w)<=fraction(u) if u is not None and w is not None else None


def favorable_screen(groups,projection):
    all_scope=groups['all'];ordinary_scopes={k:v for k,v in groups['strata'].items() if k.endswith('/ordinary')};ordinary_cells={k:v for k,v in groups['cells'].items() if '/ordinary/' in k}
    nontrivial={cell:len({fraction(v['raw_rule']['raw_weight']) for key,v in projection['rows'].items() if key in groups['cells'][cell]['selected_state_keys']})>1 for cell in CELLS}
    complete=all_scope['complete_roots']==16
    guard=lambda scope,metric:_nonincrease(scope,metric) if complete else None
    strict=fraction(all_scope['weighted']['action_Q_drift'])<fraction(all_scope['uniform']['action_Q_drift']) if complete else None
    expected_masses=dict(cells={c:rational(F(1,8)) for c in CELLS},strata={s:rational(F(1,4)) for s in STRATA},actors={str(a):rational(F(1,2)) for a in (0,1)},ordinary=rational(F(1,2)))
    ordinary_actor_mass={str(a):sum((fraction(projection['rows'][k]['weighted_probability']) for k in groups['ordinary']['selected_state_keys'] if k in groups['actors'][str(a)]['selected_state_keys']),F()) for a in (0,1)}
    conditions=dict(all16_requested_H8_H10_H12_complete=complete,
        fixed_cell_stratum_actor_ordinary_masses=same(projection['masses']['uniform'],expected_masses) and same(projection['masses']['weighted'],expected_masses) and all(v==F(1,4) for v in ordinary_actor_mass.values()),
        nontrivial_within_cell_rule=any(nontrivial.values()),primary_fulltie_regret_nonincrease_overall=guard(all_scope,'H10_best_H12_regret_mean'),
        primary_fulltie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_mean') for k,v in ordinary_scopes.items()},
        primary_fulltie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_mean') for k,v in ordinary_cells.items()},
        worst_tie_regret_nonincrease_overall=guard(all_scope,'H10_best_H12_regret_max'),worst_tie_regret_nonincrease_ordinary_strata={k:guard(v,'H10_best_H12_regret_max') for k,v in ordinary_scopes.items()},
        worst_tie_regret_nonincrease_ordinary_actor_cells={k:guard(v,'H10_best_H12_regret_max') for k,v in ordinary_cells.items()},
        root_drift_nonincrease_overall=guard(all_scope,'root_drift'),max_action_Q_drift_nonincrease_overall=guard(all_scope,'max_action_Q_drift'),
        primary_action_Q_drift_strictly_lower_overall=strict,capture_stratum_action_Q_drift_nonincrease=guard(groups['strata']['25-49/capture'],'action_Q_drift'),
        ESS_at_least_four_fifths_overall=fraction(all_scope['weighted']['ESS'])>=F(4,5)*all_scope['selected_roots'],
        ESS_at_least_four_fifths_all_actor_cells={k:fraction(v['weighted']['ESS'])>=F(4,5)*v['selected_roots'] for k,v in groups['cells'].items()})
    def flags(x):return [v for value in x.values() for v in (flags(value) if isinstance(value,dict) else [value])]
    passed=all(v is True for v in flags(conditions))
    return dict(conditions=conditions,passes=passed,status='passed' if passed else 'not_passed' if complete else 'incomplete_reference_hold',nontrivial_cells=nontrivial,
        equal_score_cells=[cell for cell,changed in nontrivial.items() if not changed],
        screen_scope='Predeclared descriptive safeguards on selected16assessment roots; not training admission, uncertainty calibration, population inference or promotion',weights_applied=False)


def evaluate(rows,projection,h12_labels):
    validate_projection(rows,projection);require(isinstance(h12_labels,dict) and set(h12_labels)<=set(rows),'Unexpected H12 root')
    require(all(v is not None for v in h12_labels.values()),'Missing references use absent keys, never surrogate/None items')
    results={key:_root_result(key,row,projection['rows'][key],h12_labels.get(key)) for key,row in sorted(rows.items())}
    groups=dict(all=_scope(sorted(rows),results),ordinary=_scope(sorted(k for k,r in rows.items() if r['stratum'].endswith('/ordinary')),results),
        cells={cell:_scope(sorted(k for k,r in rows.items() if r['cell']==cell),results) for cell in CELLS},
        strata={stratum:_scope(sorted(k for k,r in rows.items() if r['stratum']==stratum),results) for stratum in STRATA},actors={str(a):_scope(sorted(k for k,r in rows.items() if r['actor']==a),results) for a in (0,1)})
    return dict(schema=SCHEMA+'-result',projection_sha256=digest(projection),rows_sha256=digest(rows),h12_labels_sha256=digest(h12_labels),roots=results,requested_roots=16,complete_roots=len(h12_labels),metrics=dict(overall=groups['all'],ordinary=groups['ordinary'],by_cell=groups['cells'],by_stratum=groups['strata'],by_actor=groups['actors']),screen=favorable_screen(groups,projection),
        counts=dict(selected_roots=16,H8_complete=16,H10_complete=16,H12_complete=len(h12_labels),H12_missing=16-len(h12_labels),complete_pairs=len(h12_labels)),
        masses=projection['masses'],interpretation='H12 is a finite same-state requested reference. ESS is distribution concentration, not independent games or gradient ESS. Missing references retain original mass. No optimizer, sampling/loss change or strength claim.',weights_applied=False,training_performed=False)


def validate_report(rows,projection,h12_labels,report):
    require(same(report,evaluate(rows,projection,h12_labels)),'Fixed-mixture metrics/guards/identities differ');return report
