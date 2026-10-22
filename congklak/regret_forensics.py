"""Generic saved bonus/capture forensics using the accepted literal replay helpers."""
from fractions import Fraction as F
import hashlib
from pathlib import Path
from . import capture_forensics as saved
from .capture_forensics import (
    require,canonical,digest,same,integer,rational,literal,position,rawkey,active,
    replay_line,compare_lines,rank_diagnostics,_signature,_transition_matches,
)

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=saved.REFERENCE
CAPTURE_HELPER=Path(saved.__file__).resolve()
SCHEMA='regret-saved-bonus-capture-literal-forensics-v1'
STAGES=saved.STAGES
DEPTHS=saved.DEPTHS


def _mechanism(immediate):
    return ('pass' if any(v['forced_passes_after'] for v in immediate.values()) else
        'capture' if any(v['reason']=='own_capture' for v in immediate.values()) else
        'bonus' if any(v['reason']=='own_store' for v in immediate.values()) else 'ordinary')


def _build_history_case(row,items_by_stage,attempts_by_stage):
    require(isinstance(row,dict) and row['stratum'] in ('25-49/bonus','25-49/capture') and row['state_key']==rawkey(row['state']) and type(row['actor']) is int and row['actor']==row['state']['controller'],'Bonus/capture raw-root/actor differs')
    require(row['source_root_sha256']==digest(row['root']) and same(row['root']['state'],row['state']) and 25<=active(row['state'])<=49,'Source/study root/band linkage differs');legal=list(literal().legal_actions(position(row['state'])));require(len(legal)>=2 and row['legal_mask']==sum(1<<a for a in legal) and type(row['legal_mask']) is int,'Root legal mask differs')
    require(set(items_by_stage)==set(attempts_by_stage)==set(STAGES),'All3 requested horizons required')
    lines={};reports={};overlap={};same_depth_pv_differences=[];immediate={};pv_references=0
    def line_for(pv,depth,value,link):
        nonlocal pv_references
        key=digest(dict(root_state=row['state'],pv=pv,depth=depth,value_player0=value))
        if key not in lines:lines[key]=dict(trace=replay_line(row['state'],pv,depth,value),source_references=[])
        lines[key]['source_references'].append(link);pv_references+=1;return key
    for stage in STAGES:
        depth=DEPTHS[stage];item=items_by_stage[stage];attempt=attempts_by_stage[stage]
        require(attempt['stage']==stage and attempt['status']=='complete' and attempt['stop_reason']=='max_depth' and type(attempt['requested_depth']) is int and attempt['requested_depth']==depth and type(attempt['completed_depth']) is int and attempt['completed_depth']==depth and type(attempt['native_last_completed_depth']) is int and attempt['native_last_completed_depth']==depth,'Incomplete/wrong requested attempt')
        require(attempt['source_root_sha256']==item['root_sha256']==row['source_root_sha256'] and attempt['study_row_sha256']==digest(row) and attempt['label_sha256']==digest(item),'Endpoint/source/study hashes differ')
        history=attempt['history'];require(isinstance(history,list) and len(history)==depth and same(item['report'],history[-1]),'Endpoint/history linkage differs');stage_reports=[]
        for d,report in enumerate(history,1):
            signature=_signature(report,row,d);require(d not in overlap or same(overlap[d]['signature'],signature),'Overlapping completed-depth Q/best signature differs')
            refs=dict(stage=stage,actual_depth=d,report_sha256=digest(report));rootline=line_for(report['pv'],d,signature['target_player0'],dict(refs,kind='root',pit=None));candidate_lines={}
            require(type(report['pv_truncated']) is bool and report['pv_truncated']==(not lines[rootline]['trace']['leaf']['terminal']) and lines[rootline]['trace']['steps'][0]['pit'] in signature['best_moves'],'Root PV best action/truncation metadata differs')
            for c in report['candidates']:
                pit=c['pit'];value=c['depth_result']['value_player0'];lid=line_for(c['pv'],d,value,dict(refs,kind='action',pit=pit));trace=lines[lid]['trace'];require(trace['steps'] and trace['steps'][0]['pit']==pit,'Action PV starts with a different legal choice');require(type(c['pv_truncated']) is bool and c['pv_truncated']==(not trace['leaf']['terminal']),'Action PV truncation metadata differs')
                first=trace['steps'][0];_transition_matches(c['transition'],first)
                if pit in immediate:require(same(immediate[pit],first),'Same immediate action differs across stages')
                else:immediate[pit]=first
                candidate_lines[str(pit)]=lid
            endpoint=dict(actual_depth=d,report_sha256=digest(report),signature=signature,root_line=rootline,action_lines=candidate_lines)
            if d in overlap and (overlap[d]['root_line']!=rootline or overlap[d]['action_lines']!=candidate_lines):same_depth_pv_differences.append(dict(actual_depth=d,earlier_stage=overlap[d]['stage'],later_stage=stage,earlier_report_sha256=overlap[d]['report_sha256'],later_report_sha256=digest(report)))
            else:overlap.setdefault(d,dict(endpoint,stage=stage))
            stage_reports.append(endpoint)
        label=item['label'];signature=stage_reports[-1]['signature'];require(type(label['choice_depth']) is int and label['choice_depth']==depth and same({k:label[k] for k in signature},signature),'Requested label/full-action report differs');reports[stage]=stage_reports
    mechanism=_mechanism(immediate);require(row['stratum']=='25-49/'+mechanism,'Declared root mechanism differs from all literal legal actions')
    endpoints={stage:reports[stage][-1] for stage in STAGES};actions={};sign=1 if row['actor']==0 else -1
    q={stage:{c['pit']:c['value_player0'] for c in value['signature']['action_scores']} for stage,value in endpoints.items()}
    for pit in legal:
        scores={stage:dict(depth=DEPTHS[stage],value_player0=q[stage][pit],value_actor=sign*q[stage][pit],in_best_set=pit in endpoints[stage]['signature']['best_moves'],gap_from_best_actor=sign*(endpoints[stage]['signature']['target_player0']-q[stage][pit]),line_id=endpoints[stage]['action_lines'][str(pit)]) for stage in STAGES}
        changes={};comparisons={}
        for a,b in (('h8','h10'),('h10','h12'),('h8','h12')):
            delta=q[b][pit]-q[a][pit];changes[a+'_'+b]=dict(delta_player0=delta,delta_actor=sign*delta,absolute_delta=abs(delta));comparisons[a+'_'+b]=compare_lines(lines[scores[a]['line_id']]['trace'],lines[scores[b]['line_id']]['trace'])
        actions[str(pit)]=dict(pit=pit,display_pit=pit+1,immediate=immediate[pit],immediate_capture=immediate[pit]['captured_seeds']>0,immediate_bonus=immediate[pit]['bonus_choice'],immediate_relays=immediate[pit]['relays'],scores=scores,changes=changes,PV_comparisons=comparisons)
    ranks={a+'_'+b:rank_diagnostics(q[a],q[b],row['actor']) for a,b in (('h8','h10'),('h10','h12'),('h8','h12'))}
    root_comparisons={a+'_'+b:compare_lines(lines[endpoints[a]['root_line']]['trace'],lines[endpoints[b]['root_line']]['trace']) for a,b in (('h8','h10'),('h10','h12'),('h8','h12'))}
    trend=[dict(depth=r['actual_depth'],signature=r['signature'],root_line=r['root_line'],action_lines=r['action_lines'],report_sha256=r['report_sha256'],versus_previous=rank_diagnostics({v['pit']:v['value_player0'] for v in reports['h12'][i-1]['signature']['action_scores']},{v['pit']:v['value_player0'] for v in r['signature']['action_scores']},row['actor']) if i else None,
        root_PV_versus_previous=compare_lines(lines[reports['h12'][i-1]['root_line']]['trace'],lines[r['root_line']]['trace']) if i else None) for i,r in enumerate(reports['h12'])]
    return dict(schema=SCHEMA,state_key=row['state_key'],state=row['state'],root_sha256=digest(row),source_root_sha256=row['source_root_sha256'],family=row['family'],actor=row['actor'],stratum=row['stratum'],legal_actions=legal,
        source_item_sha256={s:digest(items_by_stage[s]) for s in STAGES},source_attempt_sha256={s:digest(attempts_by_stage[s]) for s in STAGES},report_index=reports,lines=lines,actions=actions,ranking=ranks,H10_best_H12_anchor_gaps=ranks['h10_h12']['before_best_anchor_gaps'],root_PV_comparisons=root_comparisons,H12_job_depth_trajectory=trend,
        overlap=dict(unique_actual_depths=sorted(overlap),Q_best_signatures_equal=True,same_depth_PV_differences=same_depth_pv_differences),
        counts=dict(legal_actions=len(legal),endpoint_action_Q_values=len(legal)*3,stage_report_references=30,unique_actual_depths=12,PV_references=pv_references,deduplicated_literal_lines=len(lines),literal_choices=sum(v['trace']['totals']['choices'] for v in lines.values()),
            referenced_choices=sum(v['trace']['totals']['choices']*len(v['source_references']) for v in lines.values()),immediate_capture_actions=sum(v['captured_seeds']>0 for v in immediate.values())),
        limits=['Repeated stage histories and root/action PVs are not independent samples. H12 depth1–12 is one saved descriptive trajectory.',
            'Root mechanism describes available actions, not every selected action. Observed captures/bonus/store swings do not isolate a cause of Q error.',
            'No new leaves, unrecorded branches, Native evaluation, inference, search, labels, games, training or weight application.'],reference_source_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),H12_is_game_truth=False)


def pair_trajectory(case,pit_a,pit_b):
    """Actor-signed score crossings on one saved H12-job trajectory, not samples."""
    legal=case['legal_actions'];require(integer(pit_a) and integer(pit_b) and pit_a in legal and pit_b in legal and pit_a<pit_b,'Ordered distinct legal pair required')
    actor=case['actor'];sign=1 if actor==0 else -1
    def point(report):
        q={v['pit']:v['value_player0'] for v in report['signature']['action_scores']};gap=sign*(q[pit_a]-q[pit_b])
        return dict(actual_depth=report['actual_depth'],report_sha256=report['report_sha256'],Q_a_player0=q[pit_a],Q_b_player0=q[pit_b],Q_a_actor=sign*q[pit_a],Q_b_actor=sign*q[pit_b],
            advantage_a_over_b_actor=gap,relation='a_preferred' if gap>0 else 'b_preferred' if gap<0 else 'tied',
            best_moves=report['signature']['best_moves'],a_in_best_set=pit_a in report['signature']['best_moves'],b_in_best_set=pit_b in report['signature']['best_moves'],
            a_line=report['action_lines'][str(pit_a)],b_line=report['action_lines'][str(pit_b)])
    history=[point(v) for v in case['report_index']['h12']];changes=[]
    for before,after in zip(history,history[1:]):
        g0=before['advantage_a_over_b_actor'];g1=after['advantage_a_over_b_actor']
        changes.append(dict(before_depth=before['actual_depth'],after_depth=after['actual_depth'],before_advantage_actor=g0,after_advantage_actor=g1,
            numeric_gap_change_actor=g1-g0,strict_reversal=g0*g1<0,tie_entered=g0!=0 and g1==0,tie_left=g0==0 and g1!=0))
    return dict(pit_a=pit_a,pit_b=pit_b,display_pit_a=pit_a+1,display_pit_b=pit_b+1,actor=actor,
        endpoints={stage:point(case['report_index'][stage][-1]) for stage in STAGES},H12_job=history,adjacent_changes=changes,
        interpretation='Same saved job, every consecutive actual depth; not independent outcomes, convergence proof or a causal explanation of the changed gap.')


def build_case(row,items_by_stage,attempts_by_stage):
    case=_build_history_case(row,items_by_stage,attempts_by_stage);sign=1 if row['actor']==0 else -1;decisions={}
    for before,after in (('h8','h10'),('h10','h12'),('h8','h12')):
        a=case['report_index'][before][-1]['signature'];b=case['report_index'][after][-1]['signature'];q={v['pit']:v['value_player0'] for v in b['action_scores']}
        regrets={pit:sign*(b['target_player0']-q[pit]) for pit in a['best_moves']};require(all(v>=0 for v in regrets.values()),'Actor-correct best-set regret cannot be negative')
        decisions[before+'_'+after]=dict(before_depth=DEPTHS[before],after_depth=DEPTHS[after],before_best_moves=a['best_moves'],after_best_moves=b['best_moves'],
            best_sets_changed=a['best_moves']!=b['best_moves'],best_sets_disjoint=not(set(a['best_moves'])&set(b['best_moves'])),
            after_regrets_by_before_best=[dict(pit=pit,regret_seeds=regrets[pit]) for pit in a['best_moves']],
            regret_min=rational(min(regrets.values())),regret_mean=rational(F(sum(regrets.values()),len(regrets))),regret_max=rational(max(regrets.values())),
            lowest_pit_before_best_regret=regrets[min(regrets)],root_delta_player0=b['target_player0']-a['target_player0'],root_delta_actor=sign*(b['target_player0']-a['target_player0']))
    legal=case['legal_actions'];pairs={str(a)+'_'+str(b):pair_trajectory(case,a,b) for index,a in enumerate(legal) for b in legal[index+1:]}
    mechanism=_mechanism({int(p):v['immediate'] for p,v in case['actions'].items()})
    case.update(root_mechanism=mechanism,requested_decision_transitions=decisions,H10_best_H12_regret=decisions['h10_h12'],action_pair_trajectories=pairs,
        literal_helper_source_sha256=hashlib.sha256(CAPTURE_HELPER.read_bytes()).hexdigest(),
        pure_reuse=dict(module='congklak.capture_forensics',line_schema=saved.SCHEMA+'-line',reused=['replay_line','compare_lines','rank_diagnostics','_signature','_transition_matches']))
    case['counts'].update(action_pairs=len(pairs),action_pair_history_rows=sum(len(v['H12_job']) for v in pairs.values()),
        immediate_bonus_actions=sum(v['immediate_bonus'] for v in case['actions'].values()))
    return case


def validate_case(row,items_by_stage,attempts_by_stage,case):
    require(same(case,build_case(row,items_by_stage,attempts_by_stage)),'Saved regret/bonus/capture facts, sources, pair trajectory or coverage differ')
    return case
