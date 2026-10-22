"""Saved-PV facts from literal Python sowing; no Native, model or search imports."""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'tests/reference.py'
SCHEMA='capture-saved-literal-forensics-v1'
STAGES=('h8','h10','h12')
DEPTHS=dict(h8=8,h10=10,h12=12)
DEPOSIT_GUARD=1000000


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(body):return json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(body):return hashlib.sha256(canonical(body).encode()).hexdigest()
def same(a,b):return canonical(a)==canonical(b)
def integer(x):return type(x) is int
def rational(q):
    require(integer(q) or type(q) is F,'Integer/Fraction required');q=F(q);return dict(numerator=q.numerator,denominator=q.denominator,value=float(q))


def literal():
    name='_capture_seed_reference'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,REFERENCE);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name];require(Path(module.__file__).resolve()==REFERENCE.resolve(),'Literal interpreter path differs');return module


def position(state):
    require(isinstance(state,dict) and set(state)=={'pits','stores','controller'},'Full state required')
    require(integer(state['controller']) and state['controller'] in (0,1),'Invalid controller')
    require(isinstance(state['pits'],list) and len(state['pits'])==2 and all(isinstance(r,list) and len(r)==7 for r in state['pits']),'Invalid pits')
    require(isinstance(state['stores'],list) and len(state['stores'])==2,'Invalid stores')
    counts=state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1]];require(all(integer(v) and 0<=v<=98 for v in counts) and sum(counts)==98,'Invalid conserved seeds')
    q=literal().Position(7,tuple(counts),state['controller']);require(literal().normalize(q).controller==q.controller,'Unnormalized raw controller');return q


def state(q):return dict(pits=[list(q.counts[:7]),list(q.counts[8:15])],stores=[q.counts[7],q.counts[15]],controller=q.controller)
def rawkey(body):q=position(body);return bytes(list(q.counts)+[q.controller]).hex()
def margin(body):return body['stores'][0]-body['stores'][1]
def active(body):return sum(map(sum,body['pits']))


def _step(q,pit,index):
    ref=literal();before=state(q);t=ref.apply(q,pit,deposit_guard=DEPOSIT_GUARD);require(ref.replay_trace(t)==t.position.counts,'Literal event reconstruction differs')
    after=state(t.position);last=next(e['index'] for e in reversed(t.events) if e['kind']=='deposit');reason='own_store' if t.bonus else 'own_capture' if t.captured else 'own_empty' if last in ref.row_indices(7,t.actor) else 'opponent_empty'
    gain=[after['stores'][a]-before['stores'][a] for a in (0,1)];captures=[dict(index=e['index'],opposite=e['opposite'],seeds=e['seeds']) for e in t.events if e['kind']=='capture']
    body=dict(choice=index,actor=t.actor,pit=pit,display_pit=pit+1,before=before,after=after,before_state_key=rawkey(before),after_state_key=rawkey(after),reason=reason,
        store_gain=gain,store_margin_change_player0=gain[0]-gain[1],captured_seeds=t.captured,capture_events=captures,relay_events=[dict(index=e['index'],seeds=e['seeds']) for e in t.events if e['kind']=='relay'],
        relays=t.relay_pickups,deposits=t.deposits,bonus_choice=t.bonus and not t.terminal,forced_passes_after=[dict(zip(('from','to'),p)) for p in t.final_passes],terminal_after=t.terminal,
        leaf_store_margin_player0=margin(after),active_after=active(after),literal_events_sha256=digest(list(t.events)))
    return t.position,body


def replay_line(initial,pv,depth,expected_value=None):
    require(integer(depth) and 1<=depth<=12 and isinstance(pv,list) and len(pv)<=depth,'Saved PV/depth is invalid')
    require(expected_value is None or integer(expected_value),'Saved score must be integer');q=position(initial);steps=[];captured=[0,0];gains=[0,0];bonuses=relays=deposits=passes=0
    for i,move in enumerate(pv,1):
        require(isinstance(move,dict) and set(move)=={'actor','pit','bonus_choice','forced_passes_after','next_controller','terminal_after'},'Saved move metadata differs')
        require(integer(move['actor']) and move['actor']==q.controller and integer(move['pit']) and move['pit'] in literal().legal_actions(q),'Illegal saved controller/action')
        q,body=_step(q,move['pit'],i)
        expected=dict(actor=body['actor'],pit=body['pit'],bonus_choice=body['bonus_choice'],forced_passes_after=body['forced_passes_after'],next_controller=None if body['terminal_after'] else q.controller,terminal_after=body['terminal_after'])
        require(same(move,expected),'Saved move/bonus/pass/terminal metadata differs')
        captured[body['actor']]+=body['captured_seeds'];gains=[gains[a]+body['store_gain'][a] for a in (0,1)];bonuses+=int(body['bonus_choice']);relays+=body['relays'];deposits+=body['deposits'];passes+=len(body['forced_passes_after'])
        body['cumulative']=dict(captured_seeds_by_player=list(captured),store_gain_by_player=list(gains),store_margin_change_player0=gains[0]-gains[1]);steps.append(body)
    end=state(q);terminal=literal().normalize(q).terminal;require(terminal or len(pv)==depth,'Saved line ends before its stated finite horizon')
    if expected_value is not None:require(expected_value==margin(end),'Saved H_v1 PV score must equal its literal leaf store margin')
    return dict(schema=SCHEMA+'-line',initial_state=initial,requested_depth=depth,pv=pv,steps=steps,leaf=dict(state=end,state_key=rawkey(end),controller=q.controller,terminal=terminal,active_seeds=active(end),stores=end['stores'],store_margin_player0=margin(end),saved_score_player0=expected_value,
        saved_score_minus_leaf_store_margin=expected_value-margin(end) if expected_value is not None else None,interpretation='Store facts on this saved line; unvisited alternatives and later choices are not evaluated.'),
        totals=dict(choices=len(steps),captured_seeds_by_player=captured,store_gain_by_player=gains,store_margin_change_player0=gains[0]-gains[1],capture_margin_change_player0=captured[0]-captured[1],sowing_store_margin_change_player0=gains[0]-gains[1]-captured[0]+captured[1],capture_events=sum(len(s['capture_events']) for s in steps),bonus_choices=bonuses,relays=relays,deposits=deposits,forced_passes=passes))


def _tail(line,prefix):
    steps=line['steps'][prefix:];captures=[sum(s['captured_seeds'] for s in steps if s['actor']==a) for a in (0,1)];gains=[sum(s['store_gain'][a] for s in steps) for a in (0,1)]
    return dict(choices=len(steps),captured_seeds_by_player=captures,store_gain_by_player=gains,store_margin_change_player0=gains[0]-gains[1],bonus_choices=sum(s['bonus_choice'] for s in steps),forced_passes=sum(len(s['forced_passes_after']) for s in steps))


def compare_lines(left,right):
    require(same(left['initial_state'],right['initial_state']),'PV comparisons require the identical full raw root')
    prefix=0;difference=None
    for a,b in zip(left['steps'],right['steps']):
        require(same(a['before'],b['before']),'Shared prefix lost identical raw-state semantics')
        if (a['actor'],a['pit'])!=(b['actor'],b['pit']):
            difference=dict(choice=prefix+1,state=a['before'],state_key=a['before_state_key'],left_actor=a['actor'],right_actor=b['actor'],left_pit=a['pit'],right_pit=b['pit'],left_mechanics=a,right_mechanics=b);break
        require(same(a['after'],b['after']),'Same state/action produced differing literal transition');prefix+=1
    kind='different_choice' if difference else 'identical_recorded_choices' if len(left['steps'])==len(right['steps']) else 'one_recorded_line_is_a_prefix'
    return dict(left_depth=left['requested_depth'],right_depth=right['requested_depth'],common_choice_prefix=prefix,relationship=kind,first_different_choice=difference,
        left_leaf=left['leaf'],right_leaf=right['leaf'],left_tail_after_common_prefix=_tail(left,prefix),right_tail_after_common_prefix=_tail(right,prefix),
        interpretation='Different horizon-selected downstream lines/leaves are observations, not a controlled causal decomposition or later-game proof.')


def rank_diagnostics(before,after,actor):
    require(type(actor) is int and actor in (0,1) and set(before)==set(after) and len(before)>=2 and all(integer(a) and 0<=a<7 and integer(v) for q in (before,after) for a,v in q.items()),'Malformed full-action comparison')
    sign=1 if actor==0 else -1;best_before=(max if actor==0 else min)(before.values());best_after=(max if actor==0 else min)(after.values());old=[a for a in sorted(before) if before[a]==best_before];new=[a for a in sorted(after) if after[a]==best_after]
    pairs=[]
    for a in sorted(before):
        for b in sorted(before):
            if b<=a:continue
            g0=sign*(before[a]-before[b]);g1=sign*(after[a]-after[b]);pairs.append(dict(pit_a=a,pit_b=b,gap_before_actor=g0,gap_after_actor=g1,numeric_gap_change_actor=g1-g0,strict_reversal=g0*g1<0,tie_changed=(g0==0)!=(g1==0)))
    anchors=[dict(before_best_anchor=b,pit=a,gap_before_actor=sign*(before[b]-before[a]),gap_after_actor=sign*(after[b]-after[a]),gap_change_actor=sign*((after[b]-before[b])-(after[a]-before[a]))) for b in old for a in sorted(before)]
    return dict(actor=actor,best_before=old,best_after=new,best_sets_changed=old!=new,best_sets_disjoint=not(set(old)&set(new)),pairs=pairs,strict_reversals=sum(p['strict_reversal'] for p in pairs),tie_changes=sum(p['tie_changed'] for p in pairs),before_best_anchor_gaps=anchors)


def _signature(report,row,depth):
    require(report['completed_iteration'] is True and report['evaluator']=='H_v1' and report['policy']=='terminal-first/no-oracle/full-root-window-v2' and same(report['position'],row['state']) and type(report['choice_depth']) is int and report['choice_depth']==depth,'Saved complete report/root/depth/policy differs')
    require(type(report['input_controller']) is int and type(report['normalized_controller']) is int and report['input_controller']==report['normalized_controller']==row['actor'] and report['forced_passes']==[] and report['terminal'] is False,'Report controller/root status differs')
    legal=list(literal().legal_actions(position(row['state'])));require(same(report['legal_actions'],legal),'Report legal support differs')
    q={}
    for candidate in report['candidates']:
        a=candidate['pit'];d=candidate['depth_result'];require(integer(a) and a in legal and a not in q and candidate['analysis_complete'] is True and type(candidate['actor']) is int and candidate['actor']==row['actor'],'Candidate support/actor differs')
        require(type(d['choice_depth']) is int and d['choice_depth']==depth and integer(d['value_player0']) and d['bound']=='exact' and d['scope']=='finite_choice_horizon' and d['evaluator']=='H_v1','Candidate finite full-action Q differs');q[a]=d['value_player0']
        require(type(d['value_actor']) is int and d['value_actor']==(1 if row['actor']==0 else -1)*q[a],'Candidate actor/P0 sign differs')
    require(set(q)==set(legal),'Missing full legal-action score');best=(max if row['actor']==0 else min)(q.values());best_moves=[a for a in legal if q[a]==best]
    require(same(report['best_moves'],best_moves) and type(report['depth_result']['value_player0']) is int and report['depth_result']['value_player0']==best,'Root target/full best set differs')
    return dict(target_player0=best,best_moves=best_moves,action_scores=[dict(pit=a,value_player0=q[a]) for a in legal])


def _transition_matches(saved,step):
    require(isinstance(saved,dict),'Missing saved immediate transition');expected=dict(ok=True,actor=step['actor'],pit=step['pit'],position=step['after'],store_delta=step['store_gain'],relays=step['relays'],drops=step['deposits'],passes=step['forced_passes_after'],terminal=step['terminal_after'],reason=step['reason'])
    require(same({key:saved[key] for key in expected},expected),'Saved immediate transition differs from literal sowing')
    require(sum(e['count'] for e in saved['events'] if e['type']=='capture')==step['captured_seeds'],'Saved immediate capture transfer differs')


def build_case(row,items_by_stage,attempts_by_stage):
    require(isinstance(row,dict) and row['stratum']=='25-49/capture' and row['state_key']==rawkey(row['state']) and type(row['actor']) is int and row['actor']==row['state']['controller'],'Capture raw-root/actor differs')
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
    require(any(v['captured_seeds']>0 for v in immediate.values()),'Capture classification needs an actually capturing root action')
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
            'Capture-class root does not mean every action captures. Observed later captures/bonus/store swings do not isolate a cause of Q error.',
            'No new leaves, unrecorded branches, Native evaluation, inference, search, labels, games, training or weight application.'],reference_source_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),H12_is_game_truth=False)


def validate_case(row,items_by_stage,attempts_by_stage,case):
    require(same(case,build_case(row,items_by_stage,attempts_by_stage)),'Saved forensic facts/links/coverage differ');return case
