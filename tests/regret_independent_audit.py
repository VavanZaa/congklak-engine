"""Independent existing-data regret, full-action attribution and literal replay."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse,ast,copy,hashlib,importlib.util,json,os,shutil,sqlite3,sys,time,unittest,uuid
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import capture_independent_audit as cap
from capture_independent_audit import read,sha,save,rel,pins,current,durable,files,qread,rows,physical_rows,backup,mutate_row,run_child,kill,frac,rational,pos,state,raw,bits,report_literal,digest,canonical
import reference as literal

WORK=ROOT/'build/regret-audit';FREEZE=ROOT/'build/regret-audit-source-freeze.json';BASELINE=ROOT/'build/regret-baseline.json'
STAGES=('h8','h10','h12');PAIRS=(('h8','h10'),('h10','h12'),('h8','h12'))
STRATA=('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
CELLS=tuple(s+'/actor'+str(a) for s in STRATA for a in (0,1))
CASE_SUFFIXES=(178,91,382,92,390)
EVIDENCE={}
GROUPS={
 'preflight':['test_01_prior5252_sources_panel_reservations_and_models','test_02_all16_signed_attribution_fullties_covariance_and_crossing','test_03_real_case_pre_post_commit_kills_and_exact_resume'],
 'production':['test_04_all5_histories_actions_and_literal_PVs','test_05_all16_exact_context_and_case_forensics'],
 'final':['test_06_rehashed_corruption_missing_export_closed_no_work']}


def source_data():
    """Read exact consumed records; never invoke the old teacher reader."""
    registry=read(ROOT/'build/qstability-registry.json');rule=read(ROOT/'build/qstability-rule.json');previous=read(ROOT/'build/qstability-result.json')
    path=ROOT/'data/qstability-v1/work.sqlite3';rootmap=rows(path,'roots','state')
    stores={s+t:rows(path,s+t,'state') for s in STAGES for t in ('_attempts','_labels')}
    return registry,rule,previous,rootmap,stores


def case_order(rootmap):
    out=[]
    for suffix in CASE_SUFFIXES:
        matches=[k for k,r in rootmap.items() if r['family']=='midgame-history-pilot/20261023/'+str(suffix)]
        assert len(matches)==1;out.append(matches[0])
    assert len(set(out))==5;return out


def qvalues(item,actor):
    label=item['label'];q={c['pit']:c['value_player0'] for c in label['action_scores']};sign=1 if actor==0 else -1
    assert len(q)==len(label['action_scores']) and q and all(type(v) is int for v in q.values())
    target=sign*max(sign*v for v in q.values());assert label['target_player0']==target
    assert label['best_moves']==[a for a in sorted(q) if q[a]==target]
    return q


def regret(q,before_best,actor):
    sign=1 if actor==0 else -1;best=sign*max(sign*v for v in q.values());values={a:sign*(best-q[a]) for a in before_best}
    assert values and min(values.values())>=0
    return dict(by_action=values,mean=F(sum(values.values()),len(values)),minimum=F(min(values.values())),maximum=F(max(values.values())),lowest_pit=F(values[min(values)]))


def independent_context(registry,rule,rootmap,stores):
    """Every root/action and all three pairs retain the original denominator."""
    ledger=cap.all_action_ledger(registry,rule,rootmap,stores);ledger['schema']='regret-independent-all16-action-attribution-v1'
    raw_weights={};by_cell=defaultdict(list)
    for key,row in rootmap.items():
        actor=row['actor'];q={s:qvalues(stores[s+'_labels'][key],actor) for s in STAGES};legal=sorted(q['h10'])
        assert all(sorted(v)==legal for v in q.values());assert legal==list(literal.legal_actions(pos(row['state'])))
        maximum=max(abs(q['h10'][a]-q['h8'][a]) for a in legal);r=max(F(1,4),F(4,4+maximum));raw_weights[key]=r;by_cell[row['cell']].append(key)
        entry=rule['projection']['rows'][key];assert frac(entry['raw_rule']['raw_weight'])==r and entry['raw_rule']['max_action_shift_seeds']==maximum
        ledger['roots'][key]['raw_reliability']=rational(r);ledger['roots'][key]['max_early_action_shift']=maximum
        for before,after in PAIRS:
            pair=ledger['roots'][key]['pairs'][before+'_to_'+after];values=regret(q[after],stores[before+'_labels'][key]['label']['best_moves'],actor)
            pair['before_best_after_regret']=dict(by_action={str(a):rational(v) for a,v in values['by_action'].items()},mean=rational(values['mean']),minimum=rational(values['minimum']),maximum=rational(values['maximum']),lowest_pit=rational(values['lowest_pit']))
    assert set(by_cell)==set(CELLS) and all(len(v)==2 for v in by_cell.values())
    cells={};totals={}
    for cell,keys in sorted(by_cell.items()):
        keys.sort();raws=[raw_weights[k] for k in keys];pairs={}
        for key in keys:
            assert frac(rule['projection']['rows'][key]['weighted_probability'])==F(1,8)*raw_weights[key]/sum(raws)
            assert frac(rule['projection']['rows'][key]['uniform_probability'])==F(1,16)
        for before,after in PAIRS:
            pair=before+'_to_'+after;drifts=[frac(ledger['roots'][k]['pairs'][pair]['D']) for k in keys];conditional=cap.covariance(raws,drifts)
            assert sum((frac(ledger['roots'][k]['pairs'][pair]['global_delta']) for k in keys),F())==conditional/8
            pairs[pair]=dict(Q_drift_conditional_delta=rational(conditional),Q_drift_global_delta=rational(conditional/8),regrets={})
            for metric in ('mean','minimum','maximum','lowest_pit'):
                values=[frac(ledger['roots'][k]['pairs'][pair]['before_best_after_regret'][metric]) for k in keys]
                delta=cap.covariance(raws,values);pairs[pair]['regrets'][metric]=dict(conditional_delta=rational(delta),global_delta=rational(delta/8))
        cells[cell]=dict(state_keys=keys,raw_reliabilities=[rational(v) for v in raws],pairs=pairs,uniform_mass=rational(F(1,8)),weighted_mass=rational(F(1,8)))
    for before,after in PAIRS:
        pair=before+'_to_'+after
        def total(field,keys):return sum((frac(ledger['roots'][k]['pairs'][pair][field]) for k in keys),F())
        capture=[k for k,r in rootmap.items() if r['stratum']=='25-49/capture'];noncapture=sorted(set(rootmap)-set(capture));allkeys=sorted(rootmap)
        delta=total('global_delta',allkeys);assert delta==sum((frac(c['pairs'][pair]['Q_drift_global_delta']) for c in cells.values()),F())
        totals[pair]=dict(global_delta=rational(delta),capture_global_delta=rational(total('global_delta',capture)),non_capture_global_delta=rational(total('global_delta',noncapture)),
            uniform_global_drift=rational(sum((frac(ledger['roots'][k]['uniform_probability'])*frac(ledger['roots'][k]['pairs'][pair]['D']) for k in allkeys),F())),
            weighted_global_drift=rational(sum((frac(ledger['roots'][k]['weighted_probability'])*frac(ledger['roots'][k]['pairs'][pair]['D']) for k in allkeys),F())),regrets={})
        for metric in ('mean','minimum','maximum','lowest_pit'):
            values={k:frac(ledger['roots'][k]['pairs'][pair]['before_best_after_regret'][metric]) for k in allkeys}
            u=sum(values.values(),F())/16;w=sum((frac(ledger['roots'][k]['weighted_probability'])*values[k] for k in allkeys),F());cov=sum((frac(c['pairs'][pair]['regrets'][metric]['global_delta']) for c in cells.values()),F());assert w-u==cov
            totals[pair]['regrets'][metric]=dict(uniform=rational(u),weighted=rational(w),global_delta=rational(w-u))
    ledger['cells']=cells;ledger['pair_totals']=totals
    assert ledger['endpoint_Q_entries']==246 and len(ledger['roots'])==16
    return ledger


def indexed(mapping,key):return mapping[str(key)] if str(key) in mapping else mapping[key]


def metric_values(entry,pair):
    item=entry['pairs'][pair];regrets=item['before_best_after_regret'];before,after=pair.split('_to_');q0=entry['endpoints'][before];q1=entry['endpoints'][after];best=set(q0['best_moves']);actions=item['actions'];n=len(actions)
    return dict(action_Q_drift=frac(item['D']),max_action_Q_drift=F(max(v['absolute_drift'] for v in actions.values())),root_drift=F(abs(q1['target_player0']-q0['target_player0'])),
        before_best_regret_mean=frac(regrets['mean']),before_best_regret_min=frac(regrets['minimum']),before_best_regret_max=frac(regrets['maximum']),before_best_lexicographic_regret=frac(regrets['lowest_pit']),
        best_action_Q_drift_contribution=F(sum(v['absolute_drift'] for a,v in actions.items() if int(a) in best),n),nonbest_action_Q_drift_contribution=F(sum(v['absolute_drift'] for a,v in actions.items() if int(a) not in best),n))


def check_analysis(test,actual,row,endpoints,probability):
    sign=1 if row['actor']==0 else -1;q={s:qvalues(endpoints[s],row['actor']) for s in STAGES};legal=sorted(q['h10']);best={s:endpoints[s]['label']['best_moves'] for s in STAGES};union=set().union(*(set(v) for v in best.values()));threshold=sorted((sign*v for v in q['h10'].values()),reverse=True)[min(1,len(legal)-1)];top={a for a in legal if sign*q['h10'][a]>=threshold}
    u,w=frac(probability['uniform_probability']),frac(probability['weighted_probability']);test.assertEqual(actual['H10_best_moves'],best['h10']);test.assertEqual(actual['H12_best_moves'],best['h12']);test.assertEqual(actual['best_H10_H12_same'],best['h10']==best['h12'])
    for a in best['h10']:test.assertEqual(indexed(actual['H10_best_H12_regrets'],a),sign*(endpoints['h12']['label']['target_player0']-q['h12'][a]))
    test.assertEqual({int(a) for a in actual['actions']},set(legal))
    for a in legal:
        item=indexed(actual['actions'],a);test.assertEqual(item['Q_player0'],{s:q[s][a] for s in STAGES});test.assertEqual(item['Q_actor'],{s:sign*q[s][a] for s in STAGES});test.assertEqual(item['best_membership'],{s:a in best[s] for s in STAGES});test.assertEqual(item['H10_top2_including_boundary_ties'],a in top);test.assertEqual(item['ever_requested_best'],a in union)
        test.assertEqual(item['gap_to_best_actor'],{s:sign*(endpoints[s]['label']['target_player0']-q[s][a]) for s in STAGES})
    for before,after in PAIRS:
        pair=before+'_to_'+after;deltas={a:q[after][a]-q[before][a] for a in legal};parts=cap.drift_parts(list(deltas.values()));old=actual['pairs'][pair]
        for field,name in (('D_mean_absolute','absolute_mean'),('C_absolute_common_mean','translation'),('S_signed_cancellation_remainder','signed_cancellation'),('E_centered_dispersion','centered_dispersion'),('mean_delta_player0','signed_mean')):test.assertEqual(frac(old[field]),parts[name])
        test.assertEqual(frac(old['mean_delta_actor']),sign*parts['signed_mean']);test.assertEqual(old['root_value_delta_player0'],endpoints[after]['label']['target_player0']-endpoints[before]['label']['target_player0']);test.assertEqual(old['best_sets_before'],best[before]);test.assertEqual(old['best_sets_after'],best[after])
        test.assertTrue(old['D_equals_C_plus_S']);test.assertEqual(old['D_equals_C_plus_E'],parts['absolute_mean']==parts['translation']+parts['centered_dispersion'])
        ranking=[]
        for i,a in enumerate(legal):
            for b in legal[i+1:]:
                x=sign*(q[before][a]-q[before][b]);y=sign*(q[after][a]-q[after][b]);ranking.append(dict(actions=[a,b],old_actor_difference=x,new_actor_difference=y,strict_reversal=x*y<0,tie_changed=(x==0)!=(y==0),order_changed=((x>0)-(x<0))!=((y>0)-(y<0))))
        test.assertEqual(old['pairwise_ranks'],ranking);test.assertEqual(old['strict_ranking_reversals'],sum(v['strict_reversal'] for v in ranking));test.assertEqual(old['tie_changes'],sum(v['tie_changed'] for v in ranking))
        for group,selected in (('H10_best',set(best['h10'])),('H10_nonbest',set(legal)-set(best['h10'])),('H10_top2',top),('ever_requested_best',union),('never_requested_best',set(legal)-union)):
            item=old['action_groups'][group];total=sum(abs(deltas[a]) for a in selected);part=F(total,len(legal));test.assertEqual(item['actions'],sorted(selected));test.assertEqual(item['absolute_sum'],total);test.assertEqual(item['count'],len(selected));test.assertEqual(item['within_subset_mean_absolute'],rational(F(total,len(selected))) if selected else None)
            for field,value in (('root_mean_absolute_contribution',part),('uniform_global_contribution',u*part),('weighted_global_contribution',w*part),('global_weighted_minus_uniform',(w-u)*part)):test.assertEqual(frac(item[field]),value)
        values=regret(q[after],best[before],row['actor']);test.assertEqual(old['before_best_regrets'],{str(a):v for a,v in values['by_action'].items()})
        for field,name in (('before_best_regret_mean','mean'),('before_best_regret_min','minimum'),('before_best_regret_max','maximum'),('before_best_lexicographic_regret','lowest_pit')):test.assertEqual(frac(old[field]),values[name])
        for a in legal:
            item=indexed(actual['actions'],a)['pairs'][pair];test.assertEqual(item['delta_player0'],deltas[a]);test.assertEqual(item['delta_actor'],sign*deltas[a]);test.assertEqual(item['absolute_drift'],abs(deltas[a]));test.assertEqual(frac(item['centered_delta_player0']),deltas[a]-parts['signed_mean'])
            for field,value in (('root_mean_absolute_contribution',F(abs(deltas[a]),len(legal))),('uniform_global_contribution',u*F(abs(deltas[a]),len(legal))),('weighted_global_contribution',w*F(abs(deltas[a]),len(legal))),('global_weighted_minus_uniform',(w-u)*F(abs(deltas[a]),len(legal)))):test.assertEqual(frac(item[field]),value)
            test.assertEqual(item['H10_best_anchor_gap_changes'],[dict(anchor=b,anchor_still_best=b in best[after],change_actor=sign*(deltas[b]-deltas[a])) for b in best['h10']] if before=='h10' else None)


def check_scope(test,actual,keys,pair,ledger):
    test.assertEqual(actual['state_keys'],keys);test.assertEqual(actual['roots'],len(keys));metrics={k:metric_values(ledger['roots'][k],pair) for k in keys}
    for name in ('uniform','weighted'):
        p={k:frac(ledger['roots'][k][name+'_probability']) for k in keys};mass=sum(p.values(),F());item=actual[name];test.assertEqual(frac(item['original_selected_mass']),mass);test.assertEqual(item['selected_roots'],len(keys))
        test.assertEqual(item['ESS'],rational(mass*mass/sum((v*v for v in p.values()),F())) if mass else None)
        names=tuple(metric_values(next(iter(ledger['roots'].values())),pair))
        for metric in names:
            total=sum((p[k]*metrics[k][metric] for k in keys),F());test.assertEqual(frac(item['global_contributions'][metric]),total);test.assertEqual(item['conditional_risks'][metric],rational(total/mass) if mass else None)
    for metric in metric_values(next(iter(ledger['roots'].values())),pair):
        u=frac(actual['uniform']['global_contributions'][metric]);w=frac(actual['weighted']['global_contributions'][metric]);test.assertEqual(frac(actual['weighted_minus_uniform_global'][metric]),w-u)
        test.assertEqual(actual['weighted_minus_uniform_conditional'][metric],rational(frac(actual['weighted']['conditional_risks'][metric])-frac(actual['uniform']['conditional_risks'][metric])) if keys else None)


def check_forensics(test,forensic,row,stores):
    key=row['state_key'];legal=list(literal.legal_actions(pos(row['state'])));test.assertEqual(forensic['state'],row['state']);test.assertEqual(forensic['legal_actions'],legal);test.assertEqual(forensic['root_sha256'],digest(row));test.assertEqual(forensic['source_root_sha256'],row['source_root_sha256']);test.assertFalse(forensic['H12_is_game_truth'])
    test.assertEqual(forensic['reference_source_sha256'],sha(ROOT/'tests/reference.py'));test.assertEqual(forensic['literal_helper_source_sha256'],sha(ROOT/'congklak/capture_forensics.py'));test.assertEqual(forensic['overlap']['unique_actual_depths'],list(range(1,13)));test.assertTrue(forensic['overlap']['Q_best_signatures_equal'])
    referenced_choices=0;references=[]
    for line_id,line in forensic['lines'].items():
        trace=line['trace'];cap.check_line_projection(test,trace);test.assertEqual(line_id,digest(dict(root_state=row['state'],pv=trace['pv'],depth=trace['requested_depth'],value_player0=trace['leaf']['saved_score_player0'])))
        for ref in line['source_references']:
            stage=ref['stage'];d=ref['actual_depth'];report=stores[stage+'_attempts'][key]['history'][d-1];test.assertEqual(ref['report_sha256'],digest(report));test.assertEqual(trace['requested_depth'],d)
            if ref['kind']=='root':test.assertIsNone(ref['pit']);expected=report
            else:test.assertEqual(ref['kind'],'action');expected=next(c for c in report['candidates'] if c['pit']==ref['pit'])
            test.assertEqual(trace['pv'],expected['pv']);test.assertEqual(trace['leaf']['saved_score_player0'],expected['depth_result']['value_player0']);referenced_choices+=len(trace['pv']);references.append((stage,d,ref['kind'],ref['pit']))
    test.assertEqual(len(references),len(set(references)));test.assertEqual(len(references),(len(legal)+1)*30);test.assertEqual(forensic['counts']['PV_references'],len(references));test.assertEqual(forensic['counts']['referenced_choices'],referenced_choices);test.assertEqual(forensic['counts']['deduplicated_literal_lines'],len(forensic['lines']));test.assertEqual(forensic['counts']['literal_choices'],sum(len(v['trace']['pv']) for v in forensic['lines'].values()))
    test.assertEqual([v['depth'] for v in forensic['H12_job_depth_trajectory']],list(range(1,13)))
    for stage in STAGES:
        test.assertEqual(forensic['source_item_sha256'][stage],digest(stores[stage+'_labels'][key]));test.assertEqual(forensic['source_attempt_sha256'][stage],digest(stores[stage+'_attempts'][key]));test.assertEqual([v['actual_depth'] for v in forensic['report_index'][stage]],list(range(1,int(stage[1:])+1)))
        for v in forensic['report_index'][stage]:
            report=stores[stage+'_attempts'][key]['history'][v['actual_depth']-1];test.assertEqual(v['report_sha256'],digest(report));test.assertEqual(v['signature']['action_scores'],[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in report['candidates']])
    for before,after in PAIRS:
        pair=before+'_'+after;left=forensic['lines'][forensic['report_index'][before][-1]['root_line']]['trace'];right=forensic['lines'][forensic['report_index'][after][-1]['root_line']]['trace'];cap.check_comparison(test,forensic['root_PV_comparisons'][pair],left,right)
        for a in legal:
            action=forensic['actions'][str(a)];left=forensic['lines'][action['scores'][before]['line_id']]['trace'];right=forensic['lines'][action['scores'][after]['line_id']]['trace'];cap.check_comparison(test,action['PV_comparisons'][pair],left,right);test.assertEqual(action['immediate'],left['steps'][0]);test.assertEqual(action['immediate_capture'],left['steps'][0]['captured_seeds']>0);test.assertEqual(action['immediate_bonus'],left['steps'][0]['bonus_choice'])
        q0=qvalues(stores[before+'_labels'][key],row['actor']);q1=qvalues(stores[after+'_labels'][key],row['actor']);best0=stores[before+'_labels'][key]['label']['best_moves'];best1=stores[after+'_labels'][key]['label']['best_moves'];values=regret(q1,best0,row['actor']);item=forensic['requested_decision_transitions'][pair]
        test.assertEqual(item['before_best_moves'],best0);test.assertEqual(item['after_best_moves'],best1);test.assertEqual(item['after_regrets_by_before_best'],[dict(pit=a,regret_seeds=values['by_action'][a]) for a in best0]);test.assertEqual(item['best_sets_changed'],best0!=best1);test.assertEqual(item['best_sets_disjoint'],not(set(best0)&set(best1)))
        for field,name in (('regret_min','minimum'),('regret_mean','mean'),('regret_max','maximum')):test.assertEqual(frac(item[field]),values[name])
        test.assertEqual(item['lowest_pit_before_best_regret'],values['lowest_pit']);test.assertEqual(item['root_delta_player0'],stores[after+'_labels'][key]['label']['target_player0']-stores[before+'_labels'][key]['label']['target_player0']);test.assertEqual(item['root_delta_actor'],item['root_delta_player0']*(1 if row['actor']==0 else -1))
    test.assertEqual(forensic['H10_best_H12_regret'],forensic['requested_decision_transitions']['h10_h12'])
    expected_pairs={str(a)+'_'+str(b):(a,b) for i,a in enumerate(legal) for b in legal[i+1:]};test.assertEqual(set(forensic['action_pair_trajectories']),set(expected_pairs));sign=1 if row['actor']==0 else -1
    for name,(a,b) in expected_pairs.items():
        trajectory=forensic['action_pair_trajectories'][name];test.assertEqual((trajectory['pit_a'],trajectory['pit_b'],trajectory['actor']),(a,b,row['actor']));test.assertEqual(len(trajectory['H12_job']),12);test.assertEqual(len(trajectory['adjacent_changes']),11)
        def point(actual,stage,d):
            report=stores[stage+'_attempts'][key]['history'][d-1];q={c['pit']:c['depth_result']['value_player0'] for c in report['candidates']};gap=sign*(q[a]-q[b]);test.assertEqual(actual['actual_depth'],d);test.assertEqual(actual['report_sha256'],digest(report));test.assertEqual((actual['Q_a_player0'],actual['Q_b_player0']),(q[a],q[b]));test.assertEqual((actual['Q_a_actor'],actual['Q_b_actor']),(sign*q[a],sign*q[b]));test.assertEqual(actual['advantage_a_over_b_actor'],gap);test.assertEqual(actual['relation'],'a_preferred' if gap>0 else 'b_preferred' if gap<0 else 'tied');test.assertEqual(actual['best_moves'],report['best_moves']);test.assertEqual(actual['a_in_best_set'],a in report['best_moves']);test.assertEqual(actual['b_in_best_set'],b in report['best_moves'])
            index=forensic['report_index'][stage][d-1];test.assertEqual(actual['a_line'],index['action_lines'][str(a)]);test.assertEqual(actual['b_line'],index['action_lines'][str(b)])
        for stage in STAGES:point(trajectory['endpoints'][stage],stage,int(stage[1:]))
        for d,actual in enumerate(trajectory['H12_job'],1):point(actual,'h12',d)
        for i,change in enumerate(trajectory['adjacent_changes']):
            x=trajectory['H12_job'][i]['advantage_a_over_b_actor'];y=trajectory['H12_job'][i+1]['advantage_a_over_b_actor'];test.assertEqual(change,dict(before_depth=i+1,after_depth=i+2,before_advantage_actor=x,after_advantage_actor=y,numeric_gap_change_actor=y-x,strict_reversal=x*y<0,tie_entered=x!=0 and y==0,tie_left=x==0 and y!=0))
    test.assertEqual(forensic['counts']['action_pairs'],len(expected_pairs));test.assertEqual(forensic['counts']['action_pair_history_rows'],12*len(expected_pairs));return dict(PV_references=len(references),unique_lines=len(forensic['lines']),referenced_choices=referenced_choices,action_pairs=len(expected_pairs),action_pair_history_rows=12*len(expected_pairs))


class RegretIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def worker(self,spec,target):
        target.mkdir();spec=dict(spec,output=str(target/'proof.json'));save(target/'spec.json',spec)
        run_child([sys.executable,str(ROOT/'tools/regret-audit.py'),str(target/'spec.json')],target/'worker.log',timeout=1200)
        proof=read(target/'proof.json');self.assertTrue(proof['passed']);return dict(path=rel(target/'proof.json'),**proof)

    def clone(self,source,name):
        target=self.base/name;shutil.copytree(source,target);return target

    def fixture_command(self,path):
        spec=self.base/(path.name+'-cli-spec.json')
        save(spec,dict(mode='fixture_cli',output=str(self.base/(path.name+'-cli-proof.json')),calls=[],
            clis=[dict(path='tools/investigate-regret.py',argv=['run','--fixture','--cases','2','--run',str(path)])],
            input_names=['tools/regret-audit.py','tests/regret_independent_audit.py',rel(FREEZE),'build/regret-drift-plan.json','build/regret-drift-implementation.json']))
        return [sys.executable,str(ROOT/'tools/regret-audit.py'),str(spec)]

    def test_01_prior5252_sources_panel_reservations_and_models(self):
        from congklak import regret_drift as p
        baseline=read(BASELINE);old=read(ROOT/'build/qstability-acceptance.json');self.assertEqual((baseline['inputs_count'],len(baseline['inputs'])),(5252,5252));current(baseline['inputs'])
        self.assertEqual(len(old['inputs']),5249);self.assertEqual({k:baseline['inputs'][k] for k in old['inputs']},old['inputs'])
        registry,rule,previous,rootmap,stores=source_data();overlay=read(ROOT/'build/qstability-reservations.json');self.assertEqual(len(rootmap),16);self.assertEqual(set(rootmap),set(registry['rows']));self.assertEqual(set(overlay['selected_state_keys']),set(rootmap))
        self.assertTrue(overlay['future_optimizer_reserved']);self.assertFalse(overlay['historical_roles_rewritten']);self.assertFalse(previous['screen']['passes'])
        prefix=ROOT/'data/qstability-v1/references-prefix.sqlite3';self.assertEqual(sha(prefix),rule['references_prefix_sha256']);self.assertEqual(sha(ROOT/'data/qstability-v1/references-manifest.json'),rule['references_manifest_sha256']);self.assertEqual(rows(prefix,'roots','state'),rootmap)
        origins=0
        for key,row in registry['rows'].items():
            wrapper=rootmap[key];self.assertEqual(raw(row['state']),key);self.assertEqual(wrapper['source_root_sha256'],digest(wrapper['root']));self.assertEqual(wrapper['source_metadata_sha256'],digest(row));self.assertEqual(wrapper['state'],row['state']);self.assertEqual(wrapper['root'],row['source']['source_candidate'])
            self.assertEqual(row['source_cohort'],'training_candidate');self.assertEqual(row['group'],'assessment');self.assertTrue(row['future_optimizer_reserved']);self.assertIn(row['family'],overlay['origin_families']);self.assertIn(row['component'],overlay['components']);self.assertIn(row['quotient'],overlay['eligible_quotients'])
            self.assertEqual(row['legal_mask'],sum(1<<a for a in literal.legal_actions(pos(row['state']))))
            for origin in row['all_origins']:
                with closing(qread(ROOT/'data'/origin['source_journal']/'work.sqlite3')) as db:text,checksum=db.execute('SELECT body,hash FROM families WHERE idx=?',(origin['source_family_index'],)).fetchone()
                family=json.loads(text);self.assertEqual(checksum,digest(family));self.assertEqual(checksum,origin['source_family_sha256']);candidate=family['candidates'][origin['source_candidate_index']]
                self.assertEqual(candidate,origin['source_candidate']);self.assertEqual(digest(candidate),origin['source_candidate_sha256']);self.assertEqual(candidate['state'],row['state']);self.assertIn(candidate['family'],overlay['origin_families']);origins+=1
            for stage in ('h8','h10'):
                self.assertEqual(rows(prefix,stage+'_labels','state')[key],stores[stage+'_labels'][key]);self.assertEqual(rule['projection']['rows'][key]['endpoints'][stage],stores[stage+'_labels'][key])
        ordered=case_order(rootmap);self.assertEqual([rootmap[k]['actor'] for k in ordered],[1,0,0,1,1]);self.assertEqual([rootmap[k]['stratum'] for k in ordered],['25-49/bonus']+['25-49/capture']*4)
        for key in ordered:
            transition=[literal.apply(pos(rootmap[key]['state']),a) for a in literal.legal_actions(pos(rootmap[key]['state']))]
            self.assertFalse(any(t.initial_passes or t.final_passes for t in transition))
            if key==ordered[0]:self.assertFalse(any(t.captured for t in transition));self.assertTrue(any(t.bonus for t in transition))
            else:self.assertTrue(any(t.captured for t in transition))
        control=read(ROOT/'data/quality-train-v1/model.json')['model'];candidate=read(ROOT/'data/ordinary-value-pilot-v1/model.json')['model'];self.assertEqual(sha(ROOT/'data/quality-train-v1/model.json'),'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558')
        self.assertEqual({name for name in control['weights'] if bits(control['weights'][name])!=bits(candidate['weights'][name])},{'layers.4.weight','layers.4.bias'})
        data=p.admit();self.assertEqual(data['rows'],rootmap);self.assertEqual(data['case_keys'],ordered);self.assertEqual(read(p.PANEL)['all_state_keys'],sorted(rootmap));self.assertEqual(read(p.PLAN)['contract'],p.CONTRACT)
        self.assertEqual(p.CONTRACT['cases'],5);self.assertTrue(p.CONTRACT['retrospective']);self.assertFalse(p.CONTRACT['new_reservations']);self.assertEqual((p.CONTRACT['new_teacher_searches'],p.CONTRACT['new_NN_forwards'],p.CONTRACT['new_games']),(0,0,0))
        with mock.patch.object(p,'gate',side_effect=ValueError('Independent preflight unavailable')):
            with self.assertRaises(ValueError):p.run()
        self.assertFalse(p.DEFAULT.exists());self.assertNotIn('torch',sys.modules)
        EVIDENCE['identity']=dict(prior5252_current=True,previous5249_acceptance_preserved=True,all16_state_keys=sorted(rootmap),case_state_keys=ordered,case_family_suffixes=list(CASE_SUFFIXES),exact_origin_occurrences_checked=origins,reservations_unchanged=True,historical_roles_preserved=True,policy_and_models_fixed=True,production_gate_refused_before_directory_creation=True,new_runtime_calls=0)

    def test_02_all16_signed_attribution_fullties_covariance_and_crossing(self):
        from congklak import regret_drift as p
        for deltas in ([-2,6],[-4,4],[3,3],[-3,-3],[0,0],[0,1,-5,8]):
            expected=cap.drift_parts(deltas);actual=p.translation(deltas)
            for field,metric in (('mean_delta_player0','signed_mean'),('D_mean_absolute','absolute_mean'),('C_absolute_common_mean','translation'),('S_signed_cancellation_remainder','signed_cancellation'),('E_centered_dispersion','centered_dispersion')):self.assertEqual(frac(actual[field]),expected[metric])
            self.assertEqual(expected['absolute_mean'],expected['translation']+expected['signed_cancellation'])
        self.assertFalse(p.translation([-2,6])['D_equals_C_plus_E'])
        for raw_weights,drifts in (([F(2),F(1)],[F(3),F(1)]),([F(1),F(1)],[F(8),F(2)]),([F(1,4),F(3,4)],[F(5),F(5)]),([F(2,5),F(1,3)],[F(2,3),F(7,8)])):
            expected=cap.covariance(raw_weights,drifts);actual=p.covariance_identity(raw_weights,drifts);self.assertEqual(frac(actual['conditional_delta']),expected);self.assertEqual(frac(actual['global_delta']),expected/8);self.assertEqual(frac(actual['uniform_covariance'])/frac(actual['raw_mean']),expected)
        for actor in (0,1):
            sign=1 if actor==0 else -1;before={0:sign*4,2:sign*4,6:sign*1};after={0:sign*2,2:sign*3,6:sign*5};values=regret(after,[0,2],actor)
            self.assertEqual(values,dict(by_action={0:3,2:2},mean=F(5,2),minimum=F(2),maximum=F(3),lowest_pit=F(3)))
            endpoints={}
            for stage,scores in (('h8',{0:sign*5,2:sign*5,6:sign*2}),('h10',before),('h12',after)):
                target=sign*max(sign*v for v in scores.values());endpoints[stage]=dict(label=dict(action_scores=[dict(pit=a,value_player0=v) for a,v in sorted(scores.items())],target_player0=target,best_moves=[a for a,v in sorted(scores.items()) if v==target]))
            probability=dict(uniform_probability=rational(F(1,16)),weighted_probability=rational(F(3,40)));check_analysis(self,p.action_analysis(dict(actor=actor),endpoints,probability),dict(actor=actor),endpoints,probability)
        registry,rule,previous,rootmap,stores=source_data();ledger=independent_context(registry,rule,rootmap,stores);root=case_order(rootmap)[0];actor=rootmap[root]['actor'];self.assertEqual(actor,1)
        q={s:qvalues(stores[s+'_labels'][root],actor) for s in STAGES};self.assertEqual([(q[s][5],q[s][6]) for s in STAGES],[(14,16),(13,14),(16,12)])
        self.assertEqual([stores[s+'_labels'][root]['label']['best_moves'] for s in STAGES],[[5],[5],[6]])
        self.assertEqual(regret(q['h12'],[5],1)['mean'],4);self.assertEqual(frac(rule['projection']['rows'][root]['uniform_probability']),F(1,16));self.assertEqual(frac(rule['projection']['rows'][root]['weighted_probability']),F(1,14))
        total=ledger['pair_totals']['h10_to_h12'];self.assertEqual(frac(total['regrets']['mean']['uniform']),F(1,4));self.assertEqual(frac(total['regrets']['mean']['weighted']),F(2,7));self.assertEqual(frac(total['regrets']['mean']['global_delta']),F(1,28))
        self.assertEqual(total['uniform_global_drift'],previous['metrics']['overall']['uniform']['action_Q_drift']);self.assertEqual(total['weighted_global_drift'],previous['metrics']['overall']['weighted']['action_Q_drift'])
        EVIDENCE['synthetic_exact_algebra']=dict(translation_cases=6,covariance_cases=4,actor_cases=2,D_equals_C_plus_S=True,D_equals_C_plus_E_counterexample=True,entire_tied_best_regrets_and_lowest_pit=True,all16_original_weights_recomputed=True,root178_actor1_minimization_crossing=True,root178_H10_regret4_probability_increased=True,overall_regret_delta=rational(F(1,28)),all_assertions_exact_Fractions=True)

    def test_03_real_case_pre_post_commit_kills_and_exact_resume(self):
        from congklak import regret_drift as p
        fresh=self.base/'fresh';run_child(self.fixture_command(fresh),self.base/'fresh.log',timeout=1200);reference,manifest=p.load_rows(fresh);self.assertEqual(len(reference),2);self.assertEqual(manifest['cases'],2)
        guard=read(self.base/'fresh-cli-proof.json');self.assertEqual(guard['native_calls'],0);self.assertTrue(guard['fixture_case_commits_allowed']);self.assertIsNone(guard['new_case_commits']);records=[]
        for event,count in (('regret_precommit',1),('regret_committed',2)):
            run=self.base/('resume-'+str(count));probe=self.base/('kill-'+str(count));command=self.fixture_command(run)
            marker=kill(command,probe,event,count,dict(REGRET_BLOCK_EVENT=event,REGRET_BLOCK_STEP=str(count),REGRET_MARKER=str(probe/'declared-marker.json')))
            backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3');prefix=physical_rows(probe/'validated-prefix.sqlite3','cases','idx');self.assertEqual(len(prefix),count);self.assertEqual(prefix,physical_rows(fresh/'work.sqlite3','cases','idx')[:count]);self.assertNotIn('complete',rows(probe/'validated-prefix.sqlite3','meta','key'))
            run_child(command,probe/'resume.log',timeout=1200);actual,complete=p.load_rows(run);self.assertEqual(actual,reference);self.assertEqual(complete,manifest);self.assertEqual(prefix,physical_rows(run/'work.sqlite3','cases','idx')[:count]);self.assertEqual(read(self.base/(run.name+'-cli-proof.json'))['native_calls'],0)
            records.append(dict(event=event,actual_process_kill=True,committed_prefix=count,marker=marker,probe=rel(probe),run=rel(run),committed_prefix_bytes_preserved=True,fresh_semantics_equal=True,resumed_cases=len(actual),public_CLI_under_runtime_guards=True,producer_declared_marker_published=(probe/'declared-marker.json').exists()))
        EVIDENCE['live_case_recovery']=dict(tests=records,fresh=rel(fresh),fresh_manifest_sha256=sha(fresh/'manifest.json'),raw_case_bodies_exact=True,physical_kills=2,unique_barriers=2,no_Native_NN_search_training=True,marker_policy='Helper marker and flushed worker event required; producer atomic marker optional if killed before rename.')

    def test_04_all5_histories_actions_and_literal_PVs(self):
        from congklak import regret_drift as p
        registry,rule,previous,rootmap,stores=source_data();cases,manifest=p.load_rows();ordered=case_order(rootmap);self.assertEqual([c['state_key'] for c in cases.values()],ordered);self.assertEqual(manifest['cases'],5);counts=Counter();per_case={}
        for key in ordered:
            row=rootmap[key];case=next(c for c in cases.values() if c['state_key']==key);overlap={};found=Counter()
            self.assertEqual(case['state'],row['state']);self.assertEqual(case['source_root_sha256'],digest(row['root']));self.assertEqual(case['source_wrapper_sha256'],digest(row));self.assertNotEqual(case['source_root_sha256'],case['source_wrapper_sha256'])
            for stage,depth in zip(STAGES,(8,10,12)):
                item=stores[stage+'_labels'][key];attempt=stores[stage+'_attempts'][key];self.assertEqual(case['endpoint_item_sha256'][stage],digest(item));self.assertEqual(case['source_attempt_sha256'][stage],digest(attempt));self.assertEqual(attempt['study_row_sha256'],digest(row));self.assertEqual(item['root_sha256'],row['source_root_sha256'])
                self.assertEqual((attempt['stage'],attempt['requested_depth'],attempt['completed_depth'],attempt['status']),(stage,depth,depth,'complete'));self.assertEqual([r['choice_depth'] for r in attempt['history']],list(range(1,depth+1)));self.assertEqual(item['report'],attempt['history'][-1]);self.assertEqual(item['label']['choice_depth'],depth)
                self.assertEqual(item['label']['target_player0'],item['report']['depth_result']['value_player0']);self.assertEqual(item['label']['action_scores'],[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in item['report']['candidates']]);self.assertEqual(item['label']['best_moves'],item['report']['best_moves']);found['endpoint_Q_values']+=len(item['label']['action_scores'])
                for report in attempt['history']:
                    signature=dict(value=report['depth_result']['value_player0'],best=report['best_moves'],q=[(c['pit'],c['depth_result']['value_player0']) for c in report['candidates']]);d=report['choice_depth']
                    if d in overlap:self.assertEqual(signature,overlap[d])
                    overlap[d]=signature;proof=report_literal(row,report);found['stage_reports']+=1;found['PV_references']+=1+len(proof['actions']);found['PV_choices']+=proof['root']['choices']+sum(v['choices'] for v in proof['actions'].values())
            self.assertEqual(sorted(overlap),list(range(1,13)));self.assertEqual(found['stage_reports'],30);per_case[key]=dict(found);counts.update(found)
        self.assertEqual(counts['stage_reports'],150);save(self.base/'literal-reference-counts.json',dict(totals=dict(counts),cases=per_case))
        EVIDENCE['production_literal_sources']=dict(cases=5,case_family_suffixes=list(CASE_SUFFIXES),all16_context_roots=len(rootmap),counts=dict(counts),per_case=per_case,all_completed_history_and_full_legal_action_endpoints=True,literal_controller_bonus_capture_relay_pass_leaf_store_checks=True,source_root_and_wrapper_identities_separate=True,all_repeated_references_counted_not_independent_samples=True)


    def test_05_all16_exact_context_and_case_forensics(self):
        from congklak import regret_drift as p
        registry,rule,previous,rootmap,stores=source_data();cases,manifest=p.load_rows();report=p.verify();context=report['context'];ledger=independent_context(registry,rule,rootmap,stores);ledger_path=self.base/'all16-action-attribution.json';save(ledger_path,ledger)
        self.assertEqual((context['global_roots'],context['capture_roots'],context['detailed_case_roots'],context['uniform_global_denominator']),(16,4,5,16));self.assertFalse(context['probabilities_changed']);self.assertEqual(set(context['roots']),set(rootmap));ordered=case_order(rootmap)
        for key,row in rootmap.items():
            actual=context['roots'][key];entry=ledger['roots'][key];u,w=frac(entry['uniform_probability']),frac(entry['weighted_probability']);self.assertEqual(actual['state_key'],key);self.assertEqual(actual['actor'],row['actor']);self.assertEqual(actual['cell'],row['cell']);self.assertEqual(actual['legal_actions'],entry['legal_actions']);self.assertEqual(actual['source_root_sha256'],row['source_root_sha256']);self.assertEqual(actual['source_wrapper_sha256'],digest(row));self.assertEqual(actual['detailed_case'],key in ordered)
            self.assertEqual(actual['raw_rule'],rule['projection']['rows'][key]['raw_rule']);self.assertEqual(actual['raw_reliability'],entry['raw_reliability']);self.assertEqual(actual['uniform_probability'],entry['uniform_probability']);self.assertEqual(actual['weighted_probability'],entry['weighted_probability']);self.assertEqual(actual['endpoint_item_sha256'],{s:digest(stores[s+'_labels'][key]) for s in STAGES})
            check_analysis(self,actual['analysis'],row,{s:stores[s+'_labels'][key] for s in STAGES},rule['projection']['rows'][key])
            for before,after in PAIRS:
                pair=before+'_to_'+after;values=metric_values(entry,pair);item=actual['pairs'][pair];d=values['action_Q_drift']
                for field,value in (('drift',d),('uniform_contribution',u*d),('weighted_contribution',w*d),('global_delta',(w-u)*d)):self.assertEqual(frac(item[field]),value)
                self.assertEqual(set(item['metrics']),set(values))
                for name,value in values.items():
                    self.assertEqual(item['metrics'][name],dict(value=rational(value),uniform_global_contribution=rational(u*value),weighted_global_contribution=rational(w*value),global_weighted_minus_uniform=rational((w-u)*value)))
        for cell in CELLS:
            keys=sorted(k for k,r in rootmap.items() if r['cell']==cell);item=context['cells'][cell];raws=[frac(ledger['roots'][k]['raw_reliability']) for k in keys];self.assertEqual(item['state_keys'],keys);self.assertEqual(frac(item['uniform_mass']),F(1,8));self.assertEqual(frac(item['weighted_mass']),F(1,8))
            for before,after in PAIRS:
                pair=before+'_to_'+after
                for name in metric_values(ledger['roots'][keys[0]],pair):
                    values=[metric_values(ledger['roots'][k],pair)[name] for k in keys];conditional=cap.covariance(raws,values);actual=item['pairs'][pair]['metrics'][name];mean=sum(raws)/2;cov=(raws[0]-raws[1])*(values[0]-values[1])/4
                    self.assertEqual(actual['raw_reliabilities'],[rational(v) for v in raws]);self.assertEqual(actual['root_drifts'],[rational(v) for v in values]);self.assertEqual(frac(actual['conditional_uniform']),sum(values)/2);self.assertEqual(frac(actual['conditional_weighted']),sum((r*v for r,v in zip(raws,values)),F())/sum(raws));self.assertEqual(frac(actual['uniform_covariance']),cov);self.assertEqual(frac(actual['raw_mean']),mean);self.assertEqual(frac(actual['conditional_delta']),conditional);self.assertEqual(frac(actual['global_delta']),conditional/8);self.assertEqual(frac(actual['cell_mass']),F(1,8));self.assertFalse(actual['causal_claim']);self.assertTrue(actual['identity_verified'])
        capture=sorted(k for k,r in rootmap.items() if r['stratum']=='25-49/capture');allkeys=sorted(rootmap)
        for before,after in PAIRS:
            pair=before+'_to_'+after;expected=ledger['pair_totals'][pair];actual=context['pairs'][pair]
            for field in ('uniform_global_drift','weighted_global_drift','global_delta','capture_global_delta','non_capture_global_delta'):self.assertEqual(actual[field],expected[field])
            for stratum in STRATA:self.assertEqual(frac(actual['by_stratum'][stratum]),sum((frac(ledger['roots'][k]['pairs'][pair]['global_delta']) for k,r in rootmap.items() if r['stratum']==stratum),F()))
            group=context['scopes'][pair];check_scope(self,group['overall'],allkeys,pair,ledger);check_scope(self,group['capture'],capture,pair,ledger);check_scope(self,group['non_capture'],sorted(set(rootmap)-set(capture)),pair,ledger);check_scope(self,group['ordinary'],[k for k,r in rootmap.items() if r['stratum'].endswith('/ordinary')],pair,ledger);check_scope(self,group['detailed_cases'],ordered,pair,ledger);check_scope(self,group['other_cases'],sorted(set(rootmap)-set(ordered)),pair,ledger)
            for stratum in STRATA:check_scope(self,group['by_stratum'][stratum],[k for k,r in rootmap.items() if r['stratum']==stratum],pair,ledger)
            for cell in CELLS:check_scope(self,group['by_cell'][cell],[k for k,r in rootmap.items() if r['cell']==cell],pair,ledger)
            for actor in (0,1):check_scope(self,group['by_actor'][str(actor)],[k for k,r in rootmap.items() if r['actor']==actor],pair,ledger)
        totals=Counter()
        for case in cases.values():
            key=case['state_key'];row=rootmap[key];self.assertEqual(case['analysis'],context['roots'][key]['analysis']);self.assertEqual(case['uniform_probability'],rule['projection']['rows'][key]['uniform_probability']);self.assertEqual(case['weighted_probability'],rule['projection']['rows'][key]['weighted_probability']);self.assertFalse(case['weights_changed']);self.assertTrue(case['finite_H12_not_game_truth']);proof=check_forensics(self,case['forensics'],row,stores);totals.update(proof)
            self.assertEqual(case['forensics']['root_mechanism'],'bonus' if key==ordered[0] else 'capture')
        focus=cases[0]['forensics']['action_pair_trajectories']['5_6'];self.assertEqual([focus['endpoints'][s]['advantage_a_over_b_actor'] for s in STAGES],[2,1,-4]);self.assertEqual([focus['endpoints'][s]['best_moves'] for s in STAGES],[[5],[5],[6]])
        self.assertEqual(totals['PV_references'],900);self.assertEqual(totals['action_pairs'],55);self.assertEqual(totals['action_pair_history_rows'],660)
        self.assertFalse(report['weights_applied']);self.assertFalse(report['training_performed']);self.assertFalse(report['promotion']);self.assertFalse(report['new_reservations']);self.assertEqual(report['previous_allocation'],'hold')
        EVIDENCE['exact_attribution_and_forensics']=dict(context_roots=16,cells=8,horizon_pairs=3,all_exact_Fractions=True,all9metric_root_cell_covariance_and_scope_joins=True,original_global_vs_conditional_masses_explicit=True,all_fullties_best_nonbest_boundaryties_rank_signs_verified=True,literal_serialization_counts=dict(totals),all55_H12_pair_trajectories_and_root178_crossing=True,source_references_exhaustive=True,no_duplicate_reference_sample_claim=True,PV_mechanics_not_causal_or_strength_evidence=True,new_weights_or_reservations=False,
            all16_action_evidence=dict(path=rel(ledger_path),sha256=sha(ledger_path),schema=ledger['schema'],roots=16,endpoint_Q_entries=246,horizon_pairs=3,all_action_signed_deltas_contributions_best_nonbest_and_ranks=True),global_Q_delta=context['pairs']['h10_to_h12']['global_delta'],global_regret_delta=ledger['pair_totals']['h10_to_h12']['regrets']['mean']['global_delta'])

    def test_06_rehashed_corruption_missing_export_closed_no_work(self):
        from congklak import regret_drift as p
        preflight=read(WORK/'preflight.json');fresh=ROOT/preflight['evidence']['live_case_recovery']['fresh'];original_cases,original_manifest=p.load_rows(fresh);checks=[]
        for kind in ('metric','regret','literal','trajectory','identity','complete'):
            target=self.clone(fresh,'bad-'+kind);(target/'manifest.json').unlink()
            if kind=='metric':mutate_row(target/'work.sqlite3','cases',0,'idx',lambda b:b['analysis']['pairs']['h10_to_h12'].update(root_value_delta_player0=999))
            elif kind=='regret':mutate_row(target/'work.sqlite3','cases',0,'idx',lambda b:b['forensics']['H10_best_H12_regret'].update(regret_mean=rational(0)))
            elif kind=='literal':
                def corrupt(b):next(iter(b['forensics']['lines'].values()))['trace']['steps'][0]['captured_seeds']+=1
                mutate_row(target/'work.sqlite3','cases',0,'idx',corrupt)
            elif kind=='trajectory':mutate_row(target/'work.sqlite3','cases',0,'idx',lambda b:b['forensics']['action_pair_trajectories']['5_6']['H12_job'][0].update(advantage_a_over_b_actor=999))
            elif kind=='identity':mutate_row(target/'work.sqlite3','meta','identity','key',lambda b:b['settings'].update(fixture=1))
            else:mutate_row(target/'work.sqlite3','meta','complete','key',lambda b:b.update(case_sha256='0'*64))
            with self.assertRaises(ValueError):p.run(target,fixture=True)
            self.assertFalse((target/'manifest.json').exists());checks.append(kind)
        repair=self.clone(fresh,'missing-final-export');expected=(repair/'manifest.json').read_bytes();(repair/'manifest.json').unlink();p.run(repair,fixture=True);self.assertEqual((repair/'manifest.json').read_bytes(),expected);self.assertEqual(p.load_rows(repair)[0],original_cases)
        required=self.clone(p.SOURCE_DEFAULT,'missing-required-source-prefix');(required/'references-prefix.sqlite3').unlink()
        with mock.patch.object(p,'SOURCE_DEFAULT',required):
            with self.assertRaises((ValueError,FileNotFoundError,sqlite3.OperationalError)):p.source()
        self.assertFalse((required/'references-prefix.sqlite3').exists())
        bad_path=ROOT/'data/regret-invalid-fixture-path-v1'
        with self.assertRaises(ValueError):p.run(bad_path,fixture=True)
        self.assertFalse(bad_path.exists());report=p.verify()
        for kind in ('metric','source'):
            bad=copy.deepcopy(report)
            if kind=='metric':bad['context']['pairs']['h10_to_h12']['global_delta']=rational(frac(bad['context']['pairs']['h10_to_h12']['global_delta'])+1)
            else:bad['inputs']['congklak/regret_drift.py']='0'*64
            with self.assertRaises(ValueError):p.validate_report(bad)
        alternate=self.base/'missing-new-result.json'
        with mock.patch.object(p,'OUTPUT',alternate):
            p.assess();first=(sha(alternate),alternate.stat().st_size,alternate.stat().st_mtime_ns);p.assess();p.verify();self.assertEqual(first,(sha(alternate),alternate.stat().st_size,alternate.stat().st_mtime_ns))
        self.assertEqual(read(alternate),report);events=[];p.run(on_event=events.append);self.assertFalse(any(e['event'] in ('regret_precommit','regret_committed') for e in events))
        proof=self.worker(dict(watch_directories=[str(p.DEFAULT)],extra_files=[rel(x) for x in (p.OUTPUT,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE)],watch_baseline=True,
            calls=[dict(module=p.__name__,function=name) for name in ('run','load_rows','build','assess','verify')],
            clis=[dict(path='tools/investigate-regret.py',argv=['all']),dict(path='tools/investigate-regret.py',argv=['verify-cases']),dict(path='tools/investigate-regret.py',argv=['verify'])],
            input_names=[rel(p.PLAN),rel(p.IMPLEMENTATION),rel(p.SOURCE_FREEZE),rel(p.DEFAULT/'work.sqlite3'),rel(p.DEFAULT/'manifest.json'),rel(p.OUTPUT),rel(FREEZE),'tests/regret_independent_audit.py']),self.base/'closed-worker')
        for field in ('native_calls','new_model_forwards','new_teacher_searches','new_PUCT_searches','new_training_updates','new_games','new_case_commits'):self.assertEqual(proof[field],0)
        self.assertTrue(proof['all_bytes_sizes_mtimes_unchanged']);self.assertFalse(proof['torch_imported']);old=ast.parse((ROOT/'congklak/qstability_progress.py').read_text(encoding='utf-8-sig'));new=ast.parse((ROOT/'congklak/regret_progress.py').read_text(encoding='utf-8-sig'))
        for name in ('save','event_message','runs','run_command'):
            before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        roadmap=(ROOT/'docs/REGRET_NEURAL_NEXT.md').read_text(encoding='utf-8')
        for phrase in ('target inconsistency','representability','saturation','gradient scale','not authorization','both H_v1','head-only','policy outputs','in the current stage'):self.assertIn(phrase,roadmap)
        current(read(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        EVIDENCE['completed_recovery_and_refusal']=dict(rehashed_SQL_refusals=checks,rehashed_metric_and_source_report_refused=True,missing_final_export_repaired_from_valid_journal=True,required_source_prefix_not_rebuilt=True,invalid_fixture_path_refused=True,missing_new_result_rebuilt_stably=True,zero_new_case_commit_events=True,completed_worker=proof,prior5252_bytes_sizes_mtimes_preserved=True,Native_NN_training_calls_prohibited=True,logger4_AST_identity=True,old_logger_tests_not_recounted=True,neural_recommendation_requires_new_target_and_fit_diagnostics=True,no_third_weighting_or_current_training=True)


def source_inputs():
    from congklak import regret_drift as p
    mapping=dict(read(BASELINE)['inputs']);paths=[Path(__file__),ROOT/'tools/regret-audit.py',BASELINE,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,ROOT/'build/regret-drift-author.json',ROOT/'build/regret-forensics-source-freeze.json',ROOT/'build/regret-forensics-author.json',ROOT/'congklak/regret_progress.py',ROOT/'tools/regret-dashboard.html',ROOT/'docs/REGRET_NEURAL_NEXT.md']
    paths += [ROOT/n for n in (*p.SOURCES,*p.DEPENDENCIES)];assert all(path.is_file() for path in paths),'Required source/owner proof missing before audit freeze';mapping.update(pins(paths));return dict(sorted(mapping.items()))


def collect():
    current(read(FREEZE)['inputs']);groupentries=[];inputs={};evidence={};covered=[]
    for name,methods in GROUPS.items():
        path=WORK/(name+'.json');body=read(path);checksum=sha(path);assert body['schema']=='regret-independent-group-v1' and body['group']==name and body['passed'] is True and body['tests_run']==len(methods)
        assert body['errors']==body['failures']==body['skips']==0 and body['covered_methods']==methods and body['covered_checks']==['__main__.RegretIndependentAudit.'+m for m in methods];current(body['inputs'])
        for key,value in body['inputs'].items():assert key not in inputs or inputs[key]==value,'Input conflict: '+key;inputs[key]=value
        assert not set(evidence)&set(body['evidence']);evidence.update(body['evidence']);covered+=methods;inputs[rel(path)]=checksum
        groupentries.append(dict(path=rel(path),sha256=checksum,tests_run=len(methods),covered_methods=methods,executed_source=dict(path=rel(__file__),sha256=sha(__file__))))
    assert len(covered)==len(set(covered))==6
    body=dict(schema='regret-independent-audit-v1',passed=True,tests_run=6,errors=0,failures=0,skips=0,covered_methods=covered,covered_checks=['__main__.RegretIndependentAudit.'+m for m in covered],groups=groupentries,inputs=dict(sorted(inputs.items())),evidence=evidence);save(ROOT/'build/regret-audit.json',body);return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('freeze',*GROUPS,'collect'));args=parser.parse_args()
    if args.operation=='freeze':save(FREEZE,dict(schema='regret-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=read(FREEZE);assert frozen['groups']==GROUPS;current(frozen['inputs']);output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable group exists'
    spec=importlib.util.spec_from_file_location('_regret_audit_runtime_guards',ROOT/'tools/regret-audit.py');guarded=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guarded;spec.loader.exec_module(guarded)
    started=time.monotonic();methods=GROUPS[args.operation]
    with guarded.engine_guards():result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(RegretIndependentAudit(name) for name in methods))
    from congklak import regret_drift as p
    inputs={**frozen['inputs'],rel(FREEZE):sha(FREEZE)};retained=getattr(RegretIndependentAudit,'base',None);captured=files(retained) if retained else []
    if args.operation in ('production','final'):captured+=files(p.DEFAULT)+[p.OUTPUT,WORK/'preflight.json']
    if args.operation=='final':captured+=[WORK/'production.json']
    inputs.update(pins(captured));body=dict(schema='regret-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,covered_checks=['__main__.RegretIndependentAudit.'+m for m in methods],duration_seconds=time.monotonic()-started,retained_directory=rel(retained) if retained else None,inputs=dict(sorted(inputs.items())),evidence=EVIDENCE,failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures])
    save(output,body);print(canonical(dict(event='regret_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True);return int(not body['passed'])


if __name__=='__main__':raise SystemExit(main())
