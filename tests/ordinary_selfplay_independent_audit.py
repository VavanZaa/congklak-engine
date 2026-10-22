"""Independent fixed-model ordinary-coverage, ownership and recovery audit."""
from __future__ import annotations
from collections import Counter,defaultdict,deque
from contextlib import closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import unittest
from unittest import mock
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from data_pilot_independent_audit import (sha,load,save,relative,pins,dbopen,norm,from_key,
    position,key_of,quotient,rows,mechanism,kill_barrier,run_child,snapshot)
from selfplay_pilot_independent_audit import literal_puct,tuples,raw_key,sp_semantic
import reference as literal

WORK=ROOT/'build/ordinary-selfplay-audit'
BASELINE=ROOT/'build/ordinary-selfplay-baseline.json'
MODELS={
    'data/quality-train-v1/model.json':'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558',
    'data/selfplay-training-pilot-v1/model.json':'4e26d33043853c3cabd867a2ec1d9c53ea1db03cfdb44d4af7dfb58c04aefc4b'}
EVIDENCE={}


def components_reference(families,protected_q,protected_f):
    """BFS connected components, independent of producer disjoint-set union."""
    by_family={body['family']:body for body in families.values() if body['candidates']};owners=defaultdict(set)
    for family,body in by_family.items():
        for row in body['candidates']:owners[row['quotient']].add(family)
    unseen=set(by_family);components=[]
    while unseen:
        queue=deque([min(unseen)]);member=set();quotients=set()
        while queue:
            family=queue.popleft()
            if family in member:continue
            member.add(family);unseen.discard(family)
            for row in by_family[family]['candidates']:
                q=row['quotient'];quotients.add(q)
                queue.extend(sorted(owners[q]-member))
        cohorts={by_family[f]['cohort'] for f in member};reasons=[]
        if quotients&protected_q:reasons.append('protected_quotient_component')
        if member&protected_f:reasons.append('protected_family_component')
        if len(cohorts)>1:reasons.append('cross_cohort_component')
        components.append(dict(families=sorted(member),quotients=sorted(quotients),cohorts=sorted(cohorts),reasons=reasons))
    return sorted(components,key=lambda c:c['families'][0])


def complete_game_literal(game,decisions):
    q=norm(game['start']);bonus=0
    for index,row in enumerate(decisions):
        assert row['choice']==index and position(q)==row['state']
        assert row['action'] in literal.legal_actions(q) and not literal.normalize(q).terminal
        transition=literal.apply(q,row['action']);bonus+=bool(transition.bonus and not transition.terminal)
        q=transition.position;assert position(q)==row['after']
    assert position(q)==game['position'] and game['choices']==len(decisions)
    terminal=literal.normalize(q).terminal
    assert game['status']==('terminal' if terminal else 'capped')
    assert game['terminal_margin']==(literal.evaluate(q) if terminal else None)
    return bonus


def evaluation_reference():
    """Accepted true-role ledger plus literal states of all latest96 matches."""
    previous=load(ROOT/'build/selfplay-training-pilot-registry.json')
    qset=set(previous['quotients']);fset=set(previous['families']);observations=[]
    path=ROOT/'data/selfplay-pilot-benchmark-v1/work.sqlite3';games=rows(path,'games','idx')
    assert len(games)==96;choices=0
    for index,game in games.items():
        q=norm(game['start']);fset.add(game['family']);trajectory=[]
        for choice,action in enumerate([*game['moves'],None]):
            state=position(q);keyq=quotient(state);qset.add(keyq)
            trajectory.append(dict(state_key=raw_key(state),quotient=keyq))
            if action is not None:
                assert game['decisions'][choice]['actor']==q.controller and action in literal.legal_actions(q)
                q=literal.apply(q,action).position;choices+=1
        assert position(q)==game['final'] and literal.normalize(q).terminal
        observations.append(dict(index=index,family=game['family'],game_sha256=config.digest(game),states=trajectory))
    assert choices==2958
    return qset,fset,observations


def literal_family(index,settings,evaluator):
    family=f"ordinary-selfplay-history/{settings['seed']}/{index}"
    rank=int.from_bytes(hashlib.sha256(('ordinary-selfplay-owner-v1/'+family).encode()).digest()[:8],'big')%100
    cohort='training_candidate' if rank<80 else 'validation';source='random_history' if index%2==0 else 'frozen_policy_epsilon_history'
    rng=random.Random(settings['seed']+index*104729);q=norm(config.position());moves=[];candidates=[];forwards=0
    for step in range(settings['max_choices']):
        p=position(q)
        if literal.normalize(q).terminal or sum(map(sum,p['pits']))<25:break
        state=from_key(key_of(p));canonical=norm(state);active,stratum,terminal_children=mechanism(state)
        legal=list(literal.legal_actions(canonical))
        if stratum in settings['eligible_strata'] and len(legal)>=2:
            candidates.append(dict(state_key=key_of(p),state=state,quotient=quotient(state),family=family,cohort=cohort,
                purpose='ordinary_selfplay_natural_coverage',band=stratum.split('/')[0],stratum=stratum,
                active=active,legal_mask=sum(1<<a for a in legal),terminal_children=terminal_children,
                provenance=dict(source=source,family_index=index,step=step,start=config.position(),moves_prefix=list(moves),
                    source_game_idx=None,source_comparison=None,saved_puct_chosen=None,raw_state=p,swapped=raw_key(p)!=key_of(p))))
        legal=list(literal.legal_actions(q))
        if index%2==0 or step<8 or rng.random()<.25:action=legal[rng.randrange(len(legal))]
        else:
            prediction=evaluator.evaluate_many([p])[0];forwards+=1
            action=min(legal,key=lambda a:(-prediction['policy_logits'][a],a))
        moves.append(action);q=literal.apply(q,action).position
    final=position(q);active=sum(map(sum,final['pits']))
    return dict(index=index,family=family,cohort=cohort,source=source,rng_seed=settings['seed']+index*104729,
        start=config.position(),moves=moves,final=final,choices=len(moves),candidates=candidates,
        policy_evaluations=forwards,stop_reason='terminal' if not active else 'below25' if active<25 else 'choice_cap')


def maximum_distinct_roots(pool,settings,blocked):
    """Independent residual-network maximum flow: band/owner -> q -> family."""
    graph=defaultdict(dict)
    def edge(a,b,n):graph[a][b]=n;graph[b].setdefault(a,0)
    seen=set()
    for family in pool.values():
        if family['family'] in blocked:continue
        for row in family['candidates']:
            category=('category',row['cohort'],row['stratum']);qn=('q',row['quotient']);fn=('family',row['family'])
            if category not in seen:edge('source',category,settings['caps'][row['cohort']][row['stratum']]);seen.add(category)
            if (category,qn) not in seen:edge(category,qn,1);seen.add((category,qn))
            if (qn,fn) not in seen:edge(qn,fn,1);seen.add((qn,fn))
            if fn not in seen:edge(fn,'sink',1);seen.add(fn)
    total=0
    while True:
        queue=deque(['source']);parent={'source':None}
        while queue and 'sink' not in parent:
            a=queue.popleft()
            for b,n in graph[a].items():
                if n>0 and b not in parent:parent[b]=a;queue.append(b)
        if 'sink' not in parent:return total
        node='sink'
        while parent[node] is not None:
            prior=parent[node];graph[prior][node]-=1;graph[node][prior]+=1;node=prior
        total+=1


def check_selection(test,pool,selected,body,registry):
    settings=body['identity']['settings'];components=components_reference(pool,set(registry['novelty_quotients']),set(registry['novelty_families']))
    expected={(tuple(c['families']),tuple(c['quotients']),tuple(c['cohorts'])) for c in components}
    actual={(tuple(c['families']),tuple(c['quotients']),tuple(c['cohorts'])) for c in body['selection']['components']}
    test.assertEqual(expected,actual);blocked={f for c in components if c['reasons'] for f in c['families']}
    test.assertEqual(set(body['selection']['blocked_families']),blocked)
    allrows={config.digest(r):r for f in pool.values() for r in f['candidates']};counts=Counter()
    for row in selected.values():
        test.assertIn(config.digest(row),allrows);test.assertNotIn(row['family'],blocked)
        test.assertGreaterEqual(row['legal_mask'].bit_count(),2);test.assertTrue(row['stratum'].endswith('/ordinary'))
        test.assertEqual(mechanism(row['state'])[1],row['stratum']);counts[row['cohort'],row['stratum']]+=1
    test.assertEqual(len(selected),len({r['family'] for r in selected.values()}));test.assertEqual(len(selected),len({r['quotient'] for r in selected.values()}))
    test.assertEqual(len(selected),maximum_distinct_roots(pool,settings,blocked))
    for cohort,limits in settings['caps'].items():
        for stratum,cap in limits.items():
            test.assertLessEqual(counts[cohort,stratum],cap);test.assertEqual(body['selection']['counts'][cohort][stratum],counts[cohort,stratum])
            test.assertEqual(body['selection']['shortfall'][cohort][stratum],cap-counts[cohort,stratum])
    val=[row for family in pool.values() if family['cohort']=='validation' for row in family['candidates']]
    protection=body['rollout_protection'];test.assertEqual(set(protection['quotients']),set(registry['true_evaluation_quotients'])|{r['quotient'] for r in val})
    valfamilies={f['family'] for f in pool.values() if f['cohort']=='validation'}
    test.assertEqual(set(protection['families']),set(registry['true_evaluation_families'])|valfamilies)
    test.assertEqual(protection['new_validation_family_ids'],sorted(valfamilies));test.assertEqual(protection['new_validation_candidate_observations'],len(val))
    test.assertEqual(protection['new_validation_unique_quotients'],len({r['quotient'] for r in val}))
    return dict(families=len(pool),selected_roots=len(selected),blocked_families=len(blocked),
        independent_BFS_transitive_components_equal=True,independent_maximum_flow_root_count_equal=True,
        true_mask_covers_all_validation_owned_pool_not_just_selected=True,
        validation_pool_families=len(valfamilies),validation_pool_observations=len(val),counts=body['selection']['counts'],shortfall=body['selection']['shortfall'])


def replay_semantic(value):
    if isinstance(value,dict):return {k:replay_semantic(v) for k,v in value.items() if k not in {
        'timing','inference_ms','decision_sha256','rows_sha256','source_decision_sha256'}}
    if isinstance(value,list):return [replay_semantic(v) for v in value]
    return value


def check_replay(test,path,check_trees=False):
    from congklak import ordinary_selfplay_pilot as p,features_relay
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    roots,collection=p.load_rows(path);manifest,actual=p.load_replay(path);settings=manifest['identity']['settings'];roll=settings['rollouts']
    if manifest['identity']['fixture']:
        metadata=rows(ROOT/'data/midgame-data-pilot-v1/work.sqlite3','roots','state');chosen=[];used=set()
        for kind in p.STRATA:
            root=next(row for key,row in sorted(metadata.items()) if row['cohort']=='training_candidate' and row['stratum']==kind and row['family'] not in used)
            chosen.append(root);used.add(root['family'])
    else:chosen=[row for key,row in sorted(roots.items()) if row['cohort']=='training_candidate']
    starts=[dict(slot=i,root=r,root_sha256=config.digest(r)) for i,r in enumerate(chosen)]
    test.assertEqual(manifest['starts'],starts);games=rows(path/'work.sqlite3','sp_games','id')
    expected={f"{s['slot']:03d}/{run}":s for s in starts for run in (0,1)};test.assertEqual(set(games),set(expected))
    decisions={};trees=0;bonuses=0;native=Native()
    with CpuBatchEvaluator(p.MODEL,cache_capacity=0,max_batch=1) as evaluator:
        for gid,game in games.items():
            start=expected[gid];seed=roll['seed']+start['slot']*104729+int(gid[-1])*1009;rng=random.Random(seed)
            test.assertEqual(game['start'],start['root']['state']);test.assertEqual(game['seed'],seed)
            saved=[]
            with closing(dbopen(path/'work.sqlite3')) as db:
                for choice,body,digest in db.execute('SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(gid,)):
                    row=json.loads(body);test.assertEqual(config.digest(row),digest);test.assertEqual(choice,len(saved));saved.append((row,digest))
            bonuses+=complete_game_literal(game,[row for row,digest in saved])
            for row,digest in saved:
                state=row['state'];legal=list(literal.legal_actions(norm(state)));report=row['search'];test.assertEqual(tuples(row['rng_before']),rng.getstate())
                out=evaluator.evaluate_many([state])[0];top=max(out['policy_logits'][a] for a in legal)
                weights={a:math.exp(out['policy_logits'][a]-top) for a in legal};den=sum(weights.values());priors=[weights.get(a,0.)/den for a in range(7)]
                tree_rng=random.Random();tree_rng.setstate(rng.getstate());noise=[rng.gammavariate(.3,1) for _ in legal];den=sum(noise)
                for a,n in zip(legal,noise):priors[a]=.75*priors[a]+.25*n/den
                test.assertEqual(priors,report['root_priors']);test.assertEqual(sum(report['visits']),roll['simulations'])
                test.assertEqual(row['policy_target'],[n/roll['simulations'] for n in report['visits']]);test.assertEqual(report['stop_reason'],'simulations')
                if check_trees and trees<2 and (trees==0 or state['controller']!=previous_actor):
                    def forward(state):
                        prediction=evaluator.evaluate_many([state])[0];return prediction['value_player0'],prediction['policy_logits']
                    tree,nodes=literal_puct(state,roll['simulations'],tree_rng,forward,noise=roll['root_noise'])
                    test.assertEqual(tree['visits'],report['visits']);test.assertEqual(nodes,report['nodes'])
                    for a,n in enumerate(tree['visits']):
                        if n:test.assertAlmostEqual(98*tree['sums'][a]/n,report['action_mean_value_player0'][a],delta=1e-12)
                    trees+=1;previous_actor=state['controller']
                if row['choice']<8:
                    draw=rng.random();running=0.;action=legal[-1]
                    for a in legal:
                        running+=row['policy_target'][a]
                        if draw<running:action=a;break
                else:action=min(legal,key=lambda a:(-report['visits'][a],a))
                test.assertEqual(row['action'],action);test.assertEqual(tuples(row['rng_after']),rng.getstate())
            test.assertEqual(tuples(game['rng']),rng.getstate());decisions[gid]=saved
    protection=collection['rollout_protection'];pq=set(protection['quotients']);pf=set(protection['families'])
    excluded={};eligible={};skips=Counter();byfamily=defaultdict(list)
    for g in games.values():byfamily[g['family']].append(g)
    for family,members in byfamily.items():
        reasons=[]
        if any(g['status']!='terminal' for g in members):reasons.append('both_trajectories_not_terminal')
        if family in pf:reasons.append('protected_family')
        for game in members:
            for row,digest in decisions[game['id']]:
                state=row['state'];active,stratum,_=mechanism(state);legal=list(literal.legal_actions(norm(state)))
                if active<=4:skips['exact_domain_le4']+=1;continue
                if active<25:skips['late5_24']+=1;continue
                if not stratum.endswith('/ordinary'):skips['nonordinary_ge25']+=1;continue
                if len(legal)<2:skips['forced_ordinary_ge25']+=1;continue
                eligible[game['id'],row['choice']]=(row,digest,stratum,legal)
                if quotient(state) in pq:reasons.append('protected_eligible_ordinary_quotient_bridge')
        if reasons:excluded[family]=sorted(set(reasons))
    candidates=[];byfamily={s['root']['family']:s for s in starts}
    for (gid,choice),(decision,digest,stratum,legal) in eligible.items():
        game=games[gid]
        if game['family'] in excluded:continue
        state=decision['state'];x,d,r=features_relay.encode(state,127,native);start=byfamily[game['family']];y=game['terminal_margin'];active=sum(map(sum,state['pits']))
        test.assertEqual(r,active*(1 if state['controller']==0 else -1));test.assertEqual(d,state['stores'][0]-state['stores'][1]);test.assertLessEqual(abs(y-d),active)
        candidates.append(dict(id=gid+'/'+f'{choice:03d}',gameid=gid,choice=choice,is_start_record=choice==0,
            family=game['family'],selected_start_state_key=start['root']['state_key'],selected_start_root_sha256=start['root_sha256'],
            purpose='ordinary_selfplay_training_candidate',state=state,state_key=raw_key(state),quotient=quotient(state),stratum=stratum,
            actor=state['controller'],legal_mask=sum(1<<a for a in legal),root_visits=decision['search']['visits'],policy_target=decision['policy_target'],
            terminal_return_player0=y,labelkind='observed_fixed_control_selfplay_terminal_seed_margin',
            terminal_game_sha256=config.digest(game),source_decision_sha256=digest,features=dict(x=x,d=d,r=r)))
    grouped=defaultdict(list)
    for row in candidates:grouped[row['state_key']].append(row)
    expected_rows=[];pruned=[]
    for group in grouped.values():
        ordered=sorted(group,key=lambda row:(row['choice']!=0,hashlib.sha256(('ordinary-selfplay-rawcap-v1/'+row['id']).encode()).digest()))
        expected_rows+=ordered[:4];pruned += [r['id'] for r in ordered[4:]]
    expected_rows.sort(key=lambda r:r['id']);test.assertEqual(actual,expected_rows);summary=manifest['summary']
    test.assertEqual(summary['skipped_rows'],dict(skips));test.assertEqual(summary['family_exclusions'],excluded);test.assertEqual(summary['pruned_row_ids'],sorted(pruned))
    retained={r['id'] for r in actual};counts=Counter()
    for start in starts:
        ids=[f"{start['slot']:03d}/{run}/000" for run in (0,1)];usable=all(i in retained for i in ids)
        row=summary['usable_starts'][start['root']['state_key']];test.assertEqual(row['usable'],usable);test.assertEqual(row['required_record_ids'],ids)
        if usable:counts[start['root']['stratum']]+=1
    test.assertEqual(summary['usable_start_roots'],{kind:counts[kind] for kind in p.STRATA})
    gate=all(counts[kind]>=settings['minimum_usable_roots'][kind] for kind in p.STRATA)
    val=collection['selection']['counts']['validation'];validation_complete=all(val[kind]==settings['caps']['validation'][kind] for kind in p.STRATA)
    test.assertEqual(summary['usable_start_gate_passed'],gate);test.assertEqual(summary['validation_panel_complete'],validation_complete);test.assertEqual(summary['future_use_ready'],gate and validation_complete)
    for kind,observed in summary['support'].items():
        selected=[r for r in actual if r['stratum']==kind];descendants=[r for r in selected if r['choice']>0]
        start_states={raw_key(s['root']['state']) for s in starts}
        test.assertEqual(observed,dict(records=len(selected),unique_rawstates=len({r['state_key'] for r in selected}),families=len({r['family'] for r in selected}),
            start_records=sum(r['choice']==0 for r in selected),descendant_records=len(descendants),descendant_unique_rawstates=len({r['state_key'] for r in descendants}),
            descendant_novel_vs_selected_start_rawstates=len({r['state_key'] for r in descendants}-start_states)))
    return dict(games=len(games),terminal_games=sum(g['status']=='terminal' for g in games.values()),decisions=sum(len(v) for v in decisions.values()),
        bonus_choices=bonuses,literal_search_trees=trees,retained_rows=len(actual),unique_rawstates=len({r['state_key'] for r in actual}),
        all_noise_priors_behavior_RNG_visits_and_terminal_targets_independently_equal=True,
        ordinary_choice_bearing_ge25_eligibility_before_true_mask=True,whole_family_exclusions=excluded,
        root_first_cap4_and_both_choice0_gate_equal=True,usable_start_roots=summary['usable_start_roots'],
        usable_start_gate_passed=gate,validation_panel_complete=validation_complete,future_use_ready=gate and validation_complete,
        support=summary['support'],skipped_rows=dict(skips),models_fixed=True,no_training=True)


class OrdinarySelfplayIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def fresh(self,name):
        path=self.base/name;path.mkdir();return path

    def backup(self,source,destination):
        with closing(dbopen(source)) as db,closing(sqlite3.connect(destination)) as target:db.backup(target)

    def worker(self,spec,path):
        spec={**spec,'output':str(path.with_suffix('.json'))};save(path.with_suffix('.spec.json'),spec)
        run_child([sys.executable,str(ROOT/'tools/ordinary-selfplay-audit.py'),str(path.with_suffix('.spec.json'))],path.with_suffix('.log'),timeout=300)
        result=load(path.with_suffix('.json'));self.assertTrue(result['passed']);return result

    def test_01_fixed_models_prior_inputs_and_independent_old_ordinary_exposure(self):
        baseline=load(BASELINE)
        self.assertEqual((baseline['schema'],baseline['prior_accepted_inputs'],baseline['inputs_count']),
            ('ordinary-selfplay-prior-immutability-v1',3168,3171))
        self.assertEqual(len(baseline['inputs']),3171)
        for name,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        for field in ('new_strength_games_authorized','new_teacher_searches_authorized','new_training_authorized','source_data_pruning_authorized'):
            self.assertFalse(baseline[field])
        for name,digest in MODELS.items():self.assertEqual(sha(ROOT/name),digest,name)
        replay=list(rows(ROOT/'data/selfplay-training-pilot-v1/work.sqlite3','sp_rows','id').values())
        self.assertEqual(len(replay),276);groups=defaultdict(list)
        for index,row in enumerate(replay):groups[row['family']].append(index)
        rng=random.Random(20261025);families=sorted(groups);draws=Counter()
        for _ in range(25600):
            family=families[rng.randrange(len(families))];ids=groups[family]
            draws[ids[rng.randrange(len(ids))]]+=1
        summary={}
        for stratum,expected in (('25-49/ordinary',(15,10,2079)),('50-74/ordinary',(8,4,504))):
            indices=[i for i,row in enumerate(replay) if mechanism(row['state'])[1]==stratum]
            observed=(len(indices),len({replay[i]['state_key'] for i in indices}),sum(draws[i] for i in indices))
            self.assertEqual(observed,expected)
            self.assertTrue(all(replay[i]['legal_mask'].bit_count()>=2 for i in indices))
            if stratum.startswith('50'):self.assertTrue(all(replay[i]['choice']==0 for i in indices))
            summary[stratum]=dict(records=observed[0],raw_states=observed[1],draws=observed[2],
                families=len({replay[i]['family'] for i in indices}),choice_histogram=dict(Counter(str(replay[i]['choice']) for i in indices)))
        self.assertEqual(sum(x['records'] for x in summary.values()),23)
        self.assertEqual(sum(x['draws'] for x in summary.values()),2583)
        EVIDENCE['old_ordinary_support']=dict(total_records=276,total_raw_states=231,
            ordinary_records=23,ordinary_raw_states=14,ordinary_draws=2583,total_draws=25600,
            strata=summary,independent_literal_mechanism_and_sampler_reconstruction=True,
            forced_ordinary_records=0,reweighting_cannot_create_new_states=True,models_fixed=True)
        EVIDENCE['baseline']=dict(accepted_inputs=3168,all3171_prior_pins_current=True,
            fixed_models=MODELS,no_new_training_teacher_strength_games_or_pruning=True)

    def test_02_true_roles_latest96_literal_trajectories_and_separate_novelty(self):
        from congklak import ordinary_selfplay_pilot as p
        registry=load(p.REGISTRY);q,f,observations=evaluation_reference()
        self.assertEqual(set(registry['true_evaluation_quotients']),q);self.assertEqual(set(registry['true_evaluation_families']),f)
        self.assertEqual(registry['latest96_games'],observations);self.assertEqual(registry['latest96_state_observations'],3054)
        old=load(ROOT/'build/midgame-data-pilot-registry.json');nq=set(old['quotients']);nf=set(old['families'])
        for row in rows(ROOT/'data/midgame-data-pilot-v1/work.sqlite3','roots','state').values():nq.add(row['quotient']);nf.add(row['family'])
        priorpool=rows(ROOT/'data/midgame-data-pilot-v1/work.sqlite3','families','idx');self.assertEqual(len(priorpool),512)
        nf.update(r['family'] for r in priorpool.values());oldreplay=rows(ROOT/'data/selfplay-training-pilot-v1/work.sqlite3','sp_rows','id')
        for row in oldreplay.values():nq.add(row['quotient']);nf.add(row['family'])
        metadata=[];unbanned_training=0
        with closing(dbopen(ROOT/'data/deeper-9000-v2/work.sqlite3')) as db:
            for key,body,digest in db.execute('SELECT state,body,hash FROM roots ORDER BY state'):
                row=json.loads(body);self.assertEqual(config.digest(row),digest);keyq=quotient(row['state'])
                nq.add(keyq);nf.add(row['family']);metadata.append([key,keyq,row['family'],row['split']])
                unbanned_training += row['split']=='train' and keyq not in q and row['family'] not in f
        nq.update(q);nf.update(f)
        self.assertEqual(set(registry['novelty_quotients']),nq);self.assertEqual(set(registry['novelty_families']),nf)
        self.assertEqual(registry['metadata9000'],dict(rows=9000,semantic_sha256=config.digest(metadata),blanket_rollout_evaluation_ban=False))
        self.assertGreater(unbanned_training,1000);self.assertTrue(registry['prior_training_used_only_as_collection_novelty'])
        previous=load(ROOT/'build/selfplay-training-pilot-registry.json')
        for a,b in (('ancestral_roles','historical_roles'),('recent_roles','latest_roles'),('diagnostics','diagnostics')):
            self.assertEqual(registry[a],previous[b])
        EVIDENCE['ownership']=dict(latest_games=96,literal_choices=2958,state_observations=3054,
            all_terminal_states_included=True,true_quotients=len(q),true_families=len(f),
            novelty_quotients=len(nq),novelty_families=len(nf),historical9000_metadata_not_blanket_eval_ban=True,
            original_training_rows_still_not_evaluation_banned=unbanned_training,prior_roles_unchanged=True)

    def test_03_frozen_contract_multihop_components_matching_and_fixture_boundary(self):
        from congklak import ordinary_selfplay_pilot as p
        p.admit();s=p.CONTRACT
        self.assertEqual((s['families'],s['seed'],s['max_choices'],s['random_opening_choices'],s['epsilon']),(512,20261026,192,8,.25))
        self.assertEqual(s['minimum_legal_actions'],2);self.assertEqual(set(s['eligible_strata']),{'25-49/ordinary','50-74/ordinary'})
        self.assertEqual(s['caps'],{'training_candidate':dict.fromkeys(p.STRATA,12),'validation':dict.fromkeys(p.STRATA,3)})
        self.assertEqual(s['minimum_usable_roots'],dict.fromkeys(p.STRATA,8))
        r=s['rollouts'];self.assertEqual((r['simulations'],r['max_nodes'],r['runs_per_family'],r['max_choices'],r['c_puct']),(64,2048,2,512,.25))
        self.assertEqual(r['root_noise'],dict(alpha=.3,fraction=.25));self.assertEqual((r['rawstate_observation_cap'],r['temperature_first_choices']),(4,8))
        for field in ('automatic_expansion','constructed_filler','teacher_searches','training_performed','new_strength_games','default_promotion','sealed_final_test_consumed'):self.assertFalse(s[field])
        # The only protected q is two edges away from C, and the shared
        # ineligible opening is intentionally not represented in candidates.
        prior=rows(ROOT/'data/midgame-data-pilot-v1/work.sqlite3','roots','state')
        template=next(x for x in prior.values() if x['cohort']=='training_candidate' and x['stratum']=='50-74/ordinary')
        pool={}
        for index,(family,owner,quotients) in enumerate((('A','training_candidate',['q1']),('B','training_candidate',['q1','q2']),('C','validation',['q2']),('D','training_candidate',['q3']),('opening-only','validation',[]))):
            candidates=[]
            for j,key in enumerate(quotients):
                row=copy.deepcopy(template);row.update(family=family,cohort=owner,quotient=key,state_key=family+str(j));row['provenance'].update(family_index=index,step=j);candidates.append(row)
            pool[index]=dict(family=family,cohort=owner,candidates=candidates)
        registry=dict(novelty_quotients=['q1'],novelty_families=[],true_evaluation_quotients=[],true_evaluation_families=[])
        selected,selection,protection=p.select(pool,p.settings_for(True),registry)
        self.assertEqual(set(selection['blocked_families']),{'A','B','C'});self.assertEqual({x['family'] for x in selected.values()},{'D'})
        self.assertEqual(set(protection['new_validation_family_ids']),{'C','opening-only'})
        self.assertEqual(maximum_distinct_roots(pool,p.settings_for(True),{'A','B','C'}),1)
        expected=components_reference(pool,{'q1'},set());self.assertEqual(len(expected),2)
        self.assertEqual(expected[0]['families'],['A','B','C']);self.assertTrue(expected[0]['reasons'])
        EVIDENCE['frozen_contract']=dict(collection_families=512,maximum_training_roots=24,maximum_validation_roots=6,
            maximum_rollout_games=48,model_control_sha256=MODELS['data/quality-train-v1/model.json'],
            min_usable_distinct_start_roots_high=8,min_usable_distinct_start_roots_low=8,
            both_choice0_records_required=True,root_first_global_rawcap4=True,
            multihop_BFS_exclusion_and_ineligible_opening_fixture=True,fixture_prior_root_provenance_explicit=True,
            no_production_fallback_or_outcome_replacement=True)

    def test_04_actual_collection_pre_post_kills_literal_fixture_and_closed_refusals(self):
        from congklak import ordinary_selfplay_pilot as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        base=self.fresh('collection');fresh=base/'fresh'
        def command(run):return [sys.executable,str(ROOT/'tools/ordinary-selfplay-pilot.py'),'collect','--fixture','--run',str(run)]
        run_child(command(fresh),base/'fresh.log',timeout=300);selected,body=p.load_rows(fresh);pool=rows(fresh/'work.sqlite3','families','idx')
        self.assertEqual(len(pool),32)
        with CpuBatchEvaluator(p.MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,family in pool.items():self.assertEqual(family,literal_family(index,body['identity']['settings'],evaluator))
        check_selection(self,pool,selected,body,load(p.REGISTRY));records=[]
        for suffix,prefix in (('precommit',1),('committed',2)):
            event='ordinary_selfplay_collection_'+suffix;run=base/suffix;probe=base/(suffix+'-kill')
            marker=kill_barrier(command(run),probe,event,env={'ORDINARY_SELFPLAY_BLOCK_EVENT':event,'ORDINARY_SELFPLAY_BLOCK_STEP':str(prefix),'ORDINARY_SELFPLAY_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=prefix:v['completed']==n,timeout=300)
            self.backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3');before=list(rows(run/'work.sqlite3','families','idx').items());self.assertEqual(len(before),prefix)
            run_child(command(run),base/(suffix+'-resume.log'),timeout=300)
            finalroots,finalbody=p.load_rows(run);self.assertEqual(finalroots,selected);self.assertEqual(finalbody,body)
            self.assertEqual(rows(run/'work.sqlite3','families','idx'),pool)
            completed=self.worker(dict(kind='closed',run=str(run),stage='collection',fixture=True),base/(suffix+'-closed'))
            records.append(dict(event=event,actual_process_kill=True,committed_families=prefix,marker=marker,
                probe=relative(probe),run=relative(run),prefix_exact_after_resume=True,completed=completed))
        # Strict identity and marker checks must run before any repair.
        for name,key,mutation in (('identity','identity',lambda x:x.update(schema='wrong-schema')),('marker','collection_complete',lambda x:x.update(selected_roots=x['selected_roots']+1))):
            bad=base/('bad-'+name);shutil.copytree(fresh,bad);(bad/'collection-manifest.json').unlink()
            with closing(sqlite3.connect(bad/'work.sqlite3')) as db:
                value=json.loads(db.execute('SELECT body FROM meta WHERE key=?',(key,)).fetchone()[0]);mutation(value)
                db.execute('UPDATE meta SET body=?,hash=? WHERE key=?',(config.canonical(value),config.digest(value),key));db.commit()
            self.assertNotEqual(run_child(command(bad),base/(name+'-refused.log'),timeout=300,success=False),0);self.assertFalse((bad/'collection-manifest.json').exists())
        repaired=base/'repair';shutil.copytree(fresh,repaired);expected=(repaired/'collection-manifest.json').read_bytes();(repaired/'collection-manifest.json').unlink()
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Repair searched teacher')):p.collect(repaired,fixture=True)
        self.assertEqual((repaired/'collection-manifest.json').read_bytes(),expected)
        EVIDENCE['live_collection_recovery']=dict(tests=records,fresh_collection=relative(fresh),
            literal32_histories_equal=True,identity_and_marker_refused_before_repair=True,
            valid_missing_publication_repaired=True,actual_pre_post_process_kills=True)

    def test_05_actual_decision_pre_post_kills_terminal_returns_and_no_work(self):
        from congklak import ordinary_selfplay_pilot as p
        base=self.fresh('rollouts');source=ROOT/EVIDENCE['live_collection_recovery']['fresh_collection'];fresh=base/'fresh';shutil.copytree(source,fresh)
        def command(run):return [sys.executable,str(ROOT/'tools/ordinary-selfplay-pilot.py'),'generate','--fixture','--run',str(run)]
        run_child(command(fresh),base/'fresh.log',timeout=300);manifest,replay=p.load_replay(fresh);records=[]
        self.assertEqual(manifest['summary']['games'],4);self.assertEqual(manifest['summary']['terminal_games'],4)
        proof=check_replay(self,fresh,check_trees=True)
        for suffix,prefix in (('precommit',1),('committed',2)):
            event='ordinary_selfplay_decision_'+suffix;run=base/suffix;shutil.copytree(source,run);probe=base/(suffix+'-kill')
            marker=kill_barrier(command(run),probe,event,env={'ORDINARY_SELFPLAY_BLOCK_EVENT':event,'ORDINARY_SELFPLAY_BLOCK_STEP':str(prefix),'ORDINARY_SELFPLAY_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=prefix:v['committed_decisions']==n,timeout=300)
            self.backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
            with closing(dbopen(run/'work.sqlite3')) as db:before=db.execute('SELECT * FROM sp_decisions ORDER BY gameid,choice').fetchall()
            self.assertEqual(len(before),prefix);run_child(command(run),base/(suffix+'-resume.log'),timeout=300)
            actual,rows_=p.load_replay(run);self.assertEqual(replay_semantic(actual),replay_semantic(manifest));self.assertEqual(replay_semantic(rows_),replay_semantic(replay))
            self.assertEqual(rows(run/'work.sqlite3','sp_games','id'),rows(fresh/'work.sqlite3','sp_games','id'))
            with closing(dbopen(run/'work.sqlite3')) as db:self.assertEqual(db.execute('SELECT * FROM sp_decisions ORDER BY gameid,choice LIMIT ?',(prefix,)).fetchall(),before)
            completed=self.worker(dict(kind='closed',run=str(run),stage='rollouts',fixture=True),base/(suffix+'-closed'))
            records.append(dict(event=event,actual_process_kill=True,committed_decisions=prefix,marker=marker,
                probe=relative(probe),run=relative(run),committed_prefix_exact=True,
                semantic_moves_visits_returns_RNG_equal_fresh=True,uncommitted_clock_hashes_excluded_only=True,completed=completed))
        # Rehashed policy/return/marker corruption cannot become a new label.
        for name in ('policy','return','marker'):
            bad=base/('bad-'+name);shutil.copytree(fresh,bad)
            with closing(sqlite3.connect(bad/'work.sqlite3')) as db:
                if name=='policy':
                    game,choice,body=db.execute('SELECT gameid,choice,body FROM sp_decisions ORDER BY gameid,choice LIMIT 1').fetchone();value=json.loads(body);value['policy_target'][0]=.123
                    db.execute('UPDATE sp_decisions SET body=?,hash=? WHERE gameid=? AND choice=?',(config.canonical(value),config.digest(value),game,choice))
                elif name=='return':
                    game,body=db.execute('SELECT id,body FROM sp_games ORDER BY id LIMIT 1').fetchone();value=json.loads(body);value['terminal_margin']+=2
                    db.execute('UPDATE sp_games SET body=?,hash=? WHERE id=?',(config.canonical(value),config.digest(value),game))
                else:
                    (bad/'replay-manifest.json').unlink();value=json.loads(db.execute("SELECT body FROM meta WHERE key='replay_complete'").fetchone()[0]);value['summary']['retained_rows']+=1
                    db.execute("UPDATE meta SET body=?,hash=? WHERE key='replay_complete'",(config.canonical(value),config.digest(value)))
                db.commit()
            if name=='marker':
                self.assertNotEqual(run_child(command(bad),base/(name+'-refused.log'),timeout=300,success=False),0);self.assertFalse((bad/'replay-manifest.json').exists())
            else:
                with self.assertRaises(Exception):p.load_replay(bad)
        EVIDENCE['live_selfplay_recovery']=dict(tests=records,fixture_fresh=relative(fresh),literal_fixture=proof,
            two_prior_metadata_roots_explicit=True,production_no_fallback=True,
            corrupt_policy_terminal_return_and_marker_refused=True,actual_pre_post_process_kills=True)

    def test_06_all512_literal_histories_closure_validation_mask_and_closed_stage_backup(self):
        from congklak import ordinary_selfplay_pilot as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        selected,body=p.load_rows();pool=rows(p.DEFAULT/'work.sqlite3','families','idx');self.assertEqual(len(pool),512)
        with CpuBatchEvaluator(p.MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,family in pool.items():self.assertEqual(family,literal_family(index,body['identity']['settings'],evaluator))
        proof=check_selection(self,pool,selected,body,load(p.REGISTRY))
        backup=self.base/'production-collection-closed.sqlite3';source_sha=sha(p.DEFAULT/'work.sqlite3');self.backup(p.DEFAULT/'work.sqlite3',backup)
        self.assertEqual(rows(backup,'families','idx'),pool);self.assertEqual(rows(backup,'roots','state'),selected)
        stable=snapshot(p.DEFAULT)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Teacher search')),mock.patch.object(p.previous.quality_search,'search',side_effect=AssertionError('Collection searched PUCT')):
            p.collect();p.load_rows()
        self.assertEqual(snapshot(p.DEFAULT),stable)
        EVIDENCE['production_collection']=dict(**proof,all512_histories_literal_equal=True,
            backup=relative(backup),backup_sha256=sha(backup),capture_source_journal_sha256=source_sha,
            source_journal_sha_is_historical_not_current_after_rollouts=True,
            families_semantic_sha256=config.digest(pool),roots_semantic_sha256=config.digest(selected),
            collection_manifest_sha256=sha(p.DEFAULT/'collection-manifest.json'),
            rollout_protection_file_sha256=sha(p.DEFAULT/'rollout-protection.json'),
            completed_collection_no_teacher_no_PUCT_stable=True,models_fixed=MODELS)

    def test_07_all_terminal_rollouts_ordinary_targets_root_support_and_preserved_collection(self):
        from congklak import ordinary_selfplay_pilot as p
        prior=load(WORK/'collection.json')['evidence']['production_collection'];backup=ROOT/prior['backup']
        self.assertEqual(sha(backup),prior['backup_sha256'])
        for table,key in (('families','idx'),('roots','state')):
            with closing(dbopen(backup)) as left,closing(dbopen(p.DEFAULT/'work.sqlite3')) as right:
                self.assertEqual(left.execute(f'SELECT * FROM {table} ORDER BY {key}').fetchall(),right.execute(f'SELECT * FROM {table} ORDER BY {key}').fetchall())
        self.assertEqual(sha(p.DEFAULT/'collection-manifest.json'),prior['collection_manifest_sha256'])
        self.assertEqual(sha(p.DEFAULT/'rollout-protection.json'),prior['rollout_protection_file_sha256'])
        proof=check_replay(self,p.DEFAULT,check_trees=True);self.assertLessEqual(proof['games'],48)
        manifest,_=p.load_replay();self.assertEqual(manifest['summary']['games'],2*manifest['summary']['selected_training_starts'])
        for name,digest in MODELS.items():self.assertEqual(sha(ROOT/name),digest)
        EVIDENCE['production_replay']=dict(**proof,collection_tables_byte_exact_to_closed_backup=True,
            collection_manifest_and_full_validation_mask_unchanged=True,training_performed=False,
            honest_shortfalls_and_all_excluded_histories_retained=True)

    def test_08_offline_probabilities_fixed_forward_errors_old_arena_diagnosis_and_closed_APIs(self):
        from congklak import ordinary_selfplay_pilot as p,ordinary_selfplay_sampling as sampling
        from congklak import ordinary_selfplay_review as review,selfplay_pilot_benchmark as oldbench
        from stage5_deeper_audit_acceptance import value_and_policy
        manifest,data=p.load_replay();report=load(sampling.OUTPUT);arena=load(review.ARENA_OUTPUT)
        self.assertEqual(report['actual_new_optimizer_draws'],0);self.assertEqual(report['hypothetical_draws'],25600)
        self.assertFalse(report['sampling_applied']);self.assertFalse(report['training_performed']);self.assertFalse(report['model_changed'])
        pools={'combined_admitted':data,'paired_initial_choice0':[r for r in data if r['choice']==0],
            'ordinary_descendants':[r for r in data if r['choice']>0]};distributions=0
        for name,values in pools.items():
            proposal=report['proposals'][name];n=len(values);counts={field:Counter(r[field] for r in values) for field in ('family','state_key','stratum')}
            self.assertEqual(proposal['support']['observations'],n)
            for mode in sampling.MODES:
                probabilities=[]
                for row in values:
                    byfamily=1/(len(counts['family'])*counts['family'][row['family']]);bystate=1/(len(counts['state_key'])*counts['state_key'][row['state_key']]);bystratum=1/(len(counts['stratum'])*counts['stratum'][row['stratum']])
                    probabilities.append(1/n if mode=='observation_uniform' else byfamily if mode=='family_uniform' else bystate if mode=='raw_state_uniform' else bystratum if mode=='stratum_uniform' else .9*bystate+.1*bystratum)
                observed=proposal['distributions'][mode];self.assertEqual(observed['observations'],n)
                self.assertAlmostEqual(observed['probability_sum'],math.fsum(probabilities),places=12)
                self.assertEqual(len(observed['observation_exposures']),n)
                for row,probability,saved in zip(values,probabilities,observed['observation_exposures']):
                    self.assertEqual(saved['id'],row['id']);self.assertEqual(saved['loss_weight'],1)
                    self.assertAlmostEqual(saved['sampling_probability'],probability,places=14);self.assertAlmostEqual(saved['hypothetical_expected_draws'],25600*probability,places=9)
                self.assertAlmostEqual(observed['observation_distribution_effective_support'],1/math.fsum(x*x for x in probabilities),places=9) if n else self.assertIsNone(observed['observation_distribution_effective_support'])
                for field,category in (('family','families'),('state_key','raw_states'),('stratum','strata')):
                    mass=defaultdict(list)
                    for row,probability in zip(values,probabilities):mass[row[field]].append(probability)
                    mass={key:math.fsum(ps) for key,ps in mass.items()};item=observed[category]
                    self.assertEqual(item['support'],len(mass));self.assertEqual(set(item['groups']),set(mass))
                    if mass:self.assertAlmostEqual(item['distribution_effective_support'],1/math.fsum(x*x for x in mass.values()),places=9)
                    else:self.assertIsNone(item['distribution_effective_support'])
                    for key,value in mass.items():
                        self.assertEqual(item['groups'][key]['observations'],counts[field][key]);self.assertAlmostEqual(item['groups'][key]['probability'],value,places=12)
                distributions+=1
        metrics=report['fixed_model_own_return_diagnostics'];self.assertFalse(metrics['new_forward_timing_measured'])
        self.assertEqual(metrics['actual_CPU_forwards'],2*len({r['state_key'] for r in data}));models={
            'control':load(ROOT/'data/quality-train-v1/model.json')['model'],
            'challenger':load(ROOT/'data/selfplay-training-pilot-v1/model.json')['model']}
        for row,saved in zip(data,metrics['per_observation']):
            self.assertEqual(saved['id'],row['id']);self.assertEqual(saved['terminal_return_player0'],row['terminal_return_player0'])
            for name,model in models.items():
                value,_,logits=value_and_policy(norm(row['state']),model);prediction=metrics['forwards'][name][row['state_key']]
                self.assertAlmostEqual(prediction['value_player0'],value,delta=1e-10);self.assertLessEqual(max(abs(a-b) for a,b in zip(logits,prediction['policy_logits'])),1e-10)
                self.assertNotIn('inference_ms',prediction)
                legal=[a for a in range(7) if row['legal_mask']&(1<<a)];top=max(logits[a] for a in legal);den=math.fsum(math.exp(logits[a]-top) for a in legal)
                ce=-math.fsum(row['policy_target'][a]*math.log(max(math.exp(logits[a]-top)/den,1e-300)) for a in legal)
                current=saved['models'][name];self.assertAlmostEqual(current['absolute_error_to_own_rollout_return'],abs(value-row['terminal_return_player0']),delta=1e-10)
                self.assertAlmostEqual(current['signed_actor_error_to_own_rollout_return'],(1 if row['actor']==0 else -1)*(value-row['terminal_return_player0']),delta=1e-10)
                self.assertAlmostEqual(current['legal_policy_cross_entropy_to_saved_visits'],ce,delta=1e-10)
        games=rows(ROOT/'data/selfplay-pilot-benchmark-v1/work.sqlite3','games','idx');facts={};ordinary=[];bygame={}
        for index,game in games.items():
            q=norm(game['start']);steps=[]
            for choice,action in enumerate(game['moves']):
                state=position(q);active,stratum,_=mechanism(state);count=len(literal.legal_actions(q))
                item=(index,choice,raw_key(state),stratum,active,count);steps.append(item)
                if stratum.endswith('/ordinary'):ordinary.append(item)
                q=literal.apply(q,action).position
            self.assertEqual(position(q),game['final']);bygame[index]=steps
        self.assertEqual(len(games),96);self.assertEqual(sum(map(len,bygame.values())),2958)
        self.assertEqual([(x['source']['game_index'],x['source']['choice_index'],x['state_key'],x['stratum'],x['active'],x['legal_action_count']) for x in arena['ordinary_observations']],ordinary)
        summary=arena['coverage']['all_decisions'];self.assertEqual(summary['ordinary_decisions'],445);self.assertEqual(summary['ordinary_forced'],184)
        self.assertEqual(sum(x[5]==1 for x in ordinary),184);mid=[x for x in ordinary if x[4]>=25];self.assertEqual((len(mid),len({x[2] for x in mid})),(50,11))
        indexed={(g['comparison'].split('_vs_')[0],g['comparison'].split('_vs_')[1],g['pair'],g['candidate_seat']):(idx,g) for idx,g in games.items()}
        divergent=ordinary_divergent=0
        for seat in arena['matched_seats']:
            key=(seat['baseline'],seat['pair'],seat['candidate_seat']);ia,a=indexed['selfplay_control',*key];ib,b=indexed['selfplay_challenger',*key]
            first=next((i for i,(x,y) in enumerate(zip(a['moves'],b['moves'])) if x!=y),None);observed=seat['first_divergence']
            self.assertEqual(seat['challenger_minus_control_margin'],b['candidate_final_margin']-a['candidate_final_margin'])
            if first is None:self.assertTrue(observed['identical_action_trajectory']);continue
            divergent+=1;self.assertEqual(observed['first_different_choice'],first+1);self.assertEqual(observed['state_key'],bygame[ia][first][2]);self.assertEqual(bygame[ia][first][2],bygame[ib][first][2])
            self.assertEqual(observed['opponent_first'],observed['actor']!=seat['candidate_seat'])
            if bygame[ia][first][3].endswith('/ordinary'):
                ordinary_divergent+=1;self.assertLess(observed['active'],25);self.assertEqual(observed['existing_Q_comparison']['coverage'],'uncovered')
        self.assertEqual((divergent,ordinary_divergent),(31,4))
        stable=snapshot(p.DEFAULT);watched=[p.COLLECTION,p.PROTECTION,p.OUTPUT,sampling.OUTPUT,review.ARENA_OUTPUT]
        signatures={file:(sha(file),file.stat().st_mtime_ns) for file in watched}
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed searched teacher')),mock.patch.object(p.previous.quality_search,'search',side_effect=AssertionError('Completed searched PUCT')):
            p.collect();p.generate();p.load_replay()
        self.assertEqual(snapshot(p.DEFAULT),stable)
        for file,signature in signatures.items():self.assertEqual((sha(file),file.stat().st_mtime_ns),signature)
        for name,digest in MODELS.items():self.assertEqual(sha(ROOT/name),digest)
        for name,digest in load(BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest)
        self.assertNotIn('torch',sys.modules)
        cli=self.worker(dict(kind='closed',run=str(p.DEFAULT),stage='rollouts'),self.base/'production-completed-CLI')
        EVIDENCE['offline_review']=dict(distributions=distributions,unit_loss_positive_mass_and_concentration_independently_rederived=True,
            actual_new_optimizer_draws=0,fixed_model_literal_forwards=2*len(data),all_own_return_errors_and_visit_cross_entropies_equal=True,
            target_truth_claim=False,proposal_applied=False)
        EVIDENCE['prior_arena_diagnosis']=dict(games=96,literal_choices=2958,ordinary_observations=445,forced_ordinary=184,
            ordinary_ge25_observations=50,ordinary_ge25_rawstates=11,first_divergences=31,ordinary_first_divergences=4,
            all_four_ordinary_divergences_late_and_index_uncovered=True,no_causal_loss_fix_claim=True)
        EVIDENCE['closed_final']=dict(all_durable_bytes_mtimes_unchanged=True,zero_new_search_or_commit=True,
            torch_imported=False,both_models_fixed=True,prior3171_inputs_current=True,no_training_or_strength_games=True,
            completed_API_CLI_proof=cli)


GROUPS={
    'preflight':[
        'test_01_fixed_models_prior_inputs_and_independent_old_ordinary_exposure',
        'test_02_true_roles_latest96_literal_trajectories_and_separate_novelty',
        'test_03_frozen_contract_multihop_components_matching_and_fixture_boundary',
        'test_04_actual_collection_pre_post_kills_literal_fixture_and_closed_refusals',
        'test_05_actual_decision_pre_post_kills_terminal_returns_and_no_work'],
    'collection':['test_06_all512_literal_histories_closure_validation_mask_and_closed_stage_backup'],
    'final':[
        'test_07_all_terminal_rollouts_ordinary_targets_root_support_and_preserved_collection',
        'test_08_offline_probabilities_fixed_forward_errors_old_arena_diagnosis_and_closed_APIs']}


def collect():
    inputs={};evidence={};groups=[];covered=[]
    for name,methods in GROUPS.items():
        path=WORK/(name+'.json');body=load(path)
        assert body['passed'] is True and body['covered_methods']==methods and body['tests_run']==len(methods)
        assert not any(body[k] for k in ('errors','failures','skips'))
        for filename,digest in body['inputs'].items():
            assert sha(ROOT/filename)==digest,'Audit input changed: '+filename
            assert filename not in inputs or inputs[filename]==digest;inputs[filename]=digest
        assert not(set(evidence)&set(body['evidence']));evidence.update(body['evidence'])
        inputs[relative(path)]=sha(path);covered+=methods
        groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods))
    assert len(covered)==len(set(covered))==8
    result=dict(schema='ordinary-selfplay-independent-audit-v1',passed=True,tests_run=8,
        errors=0,failures=0,skips=0,covered_methods=covered,groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/ordinary-selfplay-audit.json'
    if output.exists():assert load(output)==result,'Immutable aggregate differs'
    else:save(output,result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=[*GROUPS,'collect']);args=parser.parse_args()
    if args.group=='collect':
        result=collect();print(config.canonical(dict(event='ordinary_selfplay_independent_audit',group='collect',passed=True,tests=8)),flush=True);return 0
    started=time.monotonic();methods=GROUPS[args.group]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(OrdinarySelfplayIndependentAudit(name) for name in methods))
    from congklak import ordinary_selfplay_pilot as producer
    files=[Path(__file__),ROOT/'tools/ordinary-selfplay-audit.py',BASELINE,producer.PLAN,producer.REGISTRY,producer.IMPLEMENTATION]
    files += [ROOT/name for name in MODELS]
    files += [ROOT/name for name in producer.SOURCES+producer.DEPENDENCIES]
    files += [ROOT/name for name in ('congklak/ordinary_selfplay_review.py','congklak/ordinary_selfplay_sampling.py',
        'tools/review-ordinary-selfplay-pilot.py','tools/review-ordinary-selfplay-sampling.py','build/ordinary-selfplay-prior-arena-review.json')]
    files += [p for p in OrdinarySelfplayIndependentAudit.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if args.group=='collection':
        files += [producer.COLLECTION,producer.PROTECTION,producer.DEFAULT/'collection-manifest.json',producer.DEFAULT/'rollout-protection.json']
        # The closed backup above is the immutable witness. The shared live
        # database will validly gain selfplay tables; never bind it here.
    if args.group=='final':
        files += [producer.COLLECTION,producer.PROTECTION,producer.OUTPUT,ROOT/'build/ordinary-selfplay-sampling-review.json']
        files += [p for p in producer.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    body=dict(schema='ordinary-selfplay-independent-group-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        duration_seconds=time.monotonic()-started,retained_directory=relative(OrdinarySelfplayIndependentAudit.base),
        inputs=pins([p for p in files if p.is_file()]),evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures])
    save(WORK/(args.group+'.json'),body)
    print(config.canonical(dict(event='ordinary_selfplay_independent_audit',group=args.group,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
