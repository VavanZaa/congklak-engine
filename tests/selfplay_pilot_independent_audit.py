"""Independent current-model self-play, joint optimizer and equal-time audit."""
from __future__ import annotations
from collections import Counter,defaultdict
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
import struct
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
from data_pilot_independent_audit import (sha,load,save,relative,pins,dbopen,norm,
    position,key_of,quotient,rows,semantic,kill_barrier,run_child,snapshot)
from quality_audit_acceptance import module
from value_pilot_independent_audit import tensor_equal,pt_catalog,literal_game
import reference as literal

WORK=ROOT/'build/selfplay-pilot-audit'
BASELINE=ROOT/'build/selfplay-pilot-baseline.json'
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
ALL_PARAMETERS={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias',
                'layers.4.weight','layers.4.bias','policy.weight','policy.bias'}
EVIDENCE={}


def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def raw_key(state):return bytes(flatten(state)+[state['controller']]).hex()


def literal_puct(state,simulations,rng,prediction,c_puct=.25,noise=None):
    """Dictionary recursion with literal sowing and no production search helpers."""
    nodes=1
    def node(q):
        n=literal.normalize(q);q=literal.Position(7,q.counts,n.controller)
        return dict(q=q,terminal=n.terminal,priors=None,bootstrap=literal.evaluate(q)/98,
            visits=[0]*7,sums=[0.]*7,children={})
    def expand(tree):
        value,logits=prediction(position(tree['q']));legal=list(literal.legal_actions(tree['q']))
        top=max(logits[a] for a in legal);weights={a:math.exp(logits[a]-top) for a in legal};z=sum(weights.values())
        tree['priors']=[weights.get(a,0.)/z for a in range(7)];tree['bootstrap']=value/98
    def visit(tree):
        nonlocal nodes
        if tree['terminal']:return literal.evaluate(tree['q'])/98
        if tree['priors'] is None:expand(tree);return tree['bootstrap']
        actor=literal.normalize(tree['q']).controller;sign=1 if actor==0 else -1
        scores={a:sign*(tree['sums'][a]/tree['visits'][a] if tree['visits'][a] else tree['bootstrap'])+
            c_puct*tree['priors'][a]*math.sqrt(max(1,sum(tree['visits'])))/(1+tree['visits'][a])
            for a in literal.legal_actions(tree['q'])}
        action=min(scores,key=lambda a:(-scores[a],a))
        if action not in tree['children']:
            tree['children'][action]=node(literal.apply(tree['q'],action).position);nodes+=1
        value=visit(tree['children'][action]);tree['visits'][action]+=1;tree['sums'][action]+=value
        return value
    root=node(norm(state))
    if root['terminal']:return root,nodes
    expand(root)
    if noise:
        legal=list(literal.legal_actions(root['q']));draws=[rng.gammavariate(noise['alpha'],1) for _ in legal];z=sum(draws)
        for action,x in zip(legal,draws):root['priors'][action]=(1-noise['fraction'])*root['priors'][action]+noise['fraction']*x/z
    for _ in range(simulations):visit(root)
    return root,nodes


def literal_trajectory(state,actions):
    q=norm(state);states=[];bonuses=passes=0
    for action in actions:
        assert not literal.normalize(q).terminal and action in literal.legal_actions(q)
        states.append(position(q));t=literal.apply(q,action);q=t.position
        bonuses+=bool(t.bonus and not t.terminal);passes+=len(t.final_passes)
    return states,position(q),dict(bonus_choices=bonuses,forced_passes=passes,
        terminal=literal.normalize(q).terminal,player0_margin=literal.evaluate(q))


def torch_python():
    value=load(ROOT/'build/training-runtime.json')
    for key in ('python','python_executable','executable'):
        if isinstance(value.get(key),str) and Path(value[key]).is_file():return value[key]
    # This location is already recorded by the accepted workspace runtime.
    expected=Path(r'D:\3D Project\Ministral-3-14B-Reasoning-2512\.venv\Scripts\python.exe')
    assert expected.is_file();return str(expected)


def independent_registry():
    true=load(ROOT/'build/value-pilot-data.json');prior=load(ROOT/'build/midgame-data-pilot-registry.json')
    q=set(true['protected_quotients']);f=set(true['protected_families'])
    q.update(prior['latest_arena_quotients']);f.update(prior['latest_arena_families'])
    for row in prior['evaluation_roots']:
        q.add(row['quotient'])
        if row['family']:f.add(row['family'])
    for row in prior['historical_roles'].values():
        if row['cohort']!='training_candidate' or row['latest_eval_quotient_collision'] or row['latest_eval_family_collision']:
            q.add(row['quotient']);f.add(row['family'])
    latest=rows(ROOT/'data/midgame-data-pilot-v1/work.sqlite3','roots','state')
    assert Counter(r['cohort'] for r in latest.values())==dict(training_candidate=39,validation=10)
    for row in [*latest.values(),*prior['diagnostics'].values()]:
        if row['cohort']!='training_candidate':q.add(quotient(row['state']));f.add(row['family'])
    projection=load(ROOT/'build/selfplay-pilot-benchmark-projection.json')
    q.update(projection['reserved_quotients']);f.update(projection['reserved_families'])
    metadata=[];train=[]
    with closing(dbopen(ROOT/'data/deeper-9000-v2/work.sqlite3')) as db:
        for key,text,family,split in db.execute("SELECT state,json_extract(body,'$.state'),json_extract(body,'$.family'),json_extract(body,'$.split') FROM roots ORDER BY state"):
            state=json.loads(text);keyq=quotient(state);metadata.append([key,keyq,family,split])
            if split!='train':q.add(keyq);f.add(family)
            else:train.append((keyq,family))
    assert len(metadata)==9000 and len(train)==8000
    return q,f,latest,metadata,dict(original_train_rows=8000,
        train_rows_not_evaluation_banned=sum(keyq not in q and family not in f for keyq,family in train),
        true_protected_quotients=len(q),true_protected_families=len(f),
        old9000_metadata_are_not_blanket_bans=True)


def matching_reference(candidates,protected_q,protected_f):
    """Iterative augmenting paths, independent of producer recursion."""
    kinds=('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
    options={kind:sorted((r for r in candidates if r['stratum']==kind and r['family'] not in protected_f and r['quotient'] not in protected_q),
        key=lambda r:hashlib.sha256(('selfplay-pilot-start-v1/'+r['state_key']).encode()).digest()) for kind in kinds}
    slots=[kind for kind in kinds for _ in range(4)];assigned={};owners={}
    for start in range(16):
        seen=set();stack=[(start,iter(options[slots[start]]))];path=[]
        while stack:
            slot,iterator=stack[-1]
            try:row=next(iterator)
            except StopIteration:
                stack.pop()
                if path:path.pop()
                continue
            family=row['family']
            if family in seen:continue
            seen.add(family)
            if family in owners:
                path.append((slot,row));target=owners[family]
                stack.append((target,iter(options[slots[target]])));continue
            assigned[slot]=row;owners[family]=slot
            for prior_slot,prior_row in reversed(path):
                assigned[prior_slot]=prior_row;owners[prior_row['family']]=prior_slot
            break
    return [dict(slot=slot,root=assigned[slot],root_sha256=config.digest(assigned[slot])) for slot in sorted(assigned)]


def sp_semantic(value):
    if isinstance(value,dict):return {k:sp_semantic(v) for k,v in value.items() if k not in {
        'timing','inference_ms','source_decision_sha256','decisions_sha256','rows_sha256'}}
    if isinstance(value,list):return [sp_semantic(v) for v in value]
    return value


def sqlite_rows(path,table,order):
    with closing(dbopen(path)) as db:return list(db.execute(f'SELECT * FROM {table} ORDER BY {order}'))


def independent_replay(path,check_trees=False):
    """Literal rules, independently rebuilt noise/behavior/eligibility/cap/targets."""
    from congklak import selfplay_training_pilot as producer,features_relay
    from congklak.puct_cpu_inference import CpuBatchEvaluator
    manifest,retained=producer.load_replay(path);settings=manifest['identity']['settings']
    games=rows(path/'work.sqlite3','sp_games','id');registry=load(producer.REGISTRY)
    protected_q=set(registry['quotients']);protected_f=set(registry['families']);all_decisions={}
    starts={f"{item['slot']:03d}/{run}":item for item in manifest['identity']['starts'] for run in range(2)}
    assert set(games)==set(starts);bonus=0;checked_trees=0;native=Native()
    with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
        for gameid,game in games.items():
            start=starts[gameid];run=int(gameid[-1]);seed=settings['seed']+start['slot']*104729+run*1009
            rng=random.Random(seed);q=norm(start['root']['state']);decisions=[]
            with closing(dbopen(path/'work.sqlite3')) as db:
                saved=list(db.execute('SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(gameid,)))
            for expected,(choice,text,digest) in enumerate(saved):
                row=json.loads(text);assert choice==expected and config.digest(row)==digest
                p=position(q);legal=list(literal.legal_actions(q));assert p==row['state'] and not literal.normalize(q).terminal
                assert tuples(row['rng_before'])==rng.getstate();report=row['search']
                prediction=evaluator.evaluate_many([p])[0];logits=prediction['policy_logits'];top=max(logits[a] for a in legal)
                exp={a:math.exp(logits[a]-top) for a in legal};total=sum(exp.values());priors=[exp.get(a,0.)/total for a in range(7)]
                tree_rng=random.Random();tree_rng.setstate(rng.getstate())
                draws=[rng.gammavariate(.3,1) for _ in legal];total=sum(draws)
                for a,draw in zip(legal,draws):priors[a]=.75*priors[a]+.25*draw/total
                assert priors==report['root_priors'];assert row['policy_target']==[v/settings['simulations'] for v in report['visits']]
                assert sum(report['visits'])==settings['simulations'] and all(v==0 for a,v in enumerate(report['visits']) if a not in legal)
                if check_trees and checked_trees<2 and (checked_trees==0 or p['controller']!=tree_actor):
                    def forward(s):
                        out=evaluator.evaluate_many([s])[0];return out['value_player0'],out['policy_logits']
                    tree,nodes=literal_puct(p,settings['simulations'],tree_rng,forward,noise=settings['root_noise'])
                    assert tree['visits']==report['visits'] and tree['priors']==report['root_priors'] and nodes==report['nodes']
                    for a,n in enumerate(tree['visits']):
                        if n:assert math.isclose(98*tree['sums'][a]/n,report['action_mean_value_player0'][a],abs_tol=1e-12)
                    checked_trees+=1;tree_actor=p['controller']
                if choice<8:
                    u=rng.random();running=0.;action=legal[-1]
                    for a in legal:
                        running+=row['policy_target'][a]
                        if u<running:action=a;break
                else:action=min(legal,key=lambda a:(-report['visits'][a],a))
                assert action==row['action'] and tuples(row['rng_after'])==rng.getstate()
                transition=literal.apply(q,action);bonus+=bool(transition.bonus and not transition.terminal);q=transition.position
                assert position(q)==row['after'];decisions.append((row,digest))
            assert game['choices']==len(saved) and position(q)==game['position'] and tuples(game['rng'])==rng.getstate()
            terminal=literal.normalize(q).terminal;assert game['status']==('terminal' if terminal else 'capped')
            assert game['terminal_margin']==(literal.evaluate(q) if terminal else None)
            all_decisions[gameid]=decisions
    exclusions={};candidates=[];skipped=0
    for family in sorted({g['family'] for g in games.values()}):
        members=[g for g in games.values() if g['family']==family];reasons=[]
        if any(g['status']!='terminal' for g in members):reasons.append('both_trajectories_not_terminal')
        if family in protected_f:reasons.append('protected_family')
        for g in members:
            for row,_ in all_decisions[g['id']]:
                active=sum(map(sum,row['state']['pits']))
                if active<25:skipped+=1;continue
                if quotient(row['state']) in protected_q:reasons.append('protected_eligible_active_ge25_quotient_bridge')
        if reasons:exclusions[family]=sorted(set(reasons))
    for gameid,game in games.items():
        if game['family'] in exclusions:continue
        for row,digest in all_decisions[gameid]:
            p=row['state'];active=sum(map(sum,p['pits']))
            if active<25:continue
            legal=list(literal.legal_actions(norm(p)));features,d,r=features_relay.encode(p,127,native)
            assert r==active*(1 if p['controller']==0 else -1) and d==p['stores'][0]-p['stores'][1]
            candidates.append(dict(id=gameid+'/'+f"{row['choice']:03d}",gameid=gameid,choice=row['choice'],family=game['family'],
                purpose='selfplay_training',state=p,state_key=raw_key(p),quotient=quotient(p),actor=p['controller'],
                legal_mask=sum(1<<a for a in legal),root_visits=row['search']['visits'],policy_target=row['policy_target'],
                terminal_return_player0=game['terminal_margin'],labelkind='selfplay_terminal_seed_margin',
                terminal_game_sha256=config.digest(game),source_decision_sha256=digest,features=dict(x=features,d=d,r=r),
                stratum=producer.pilot._mechanism(p,native)[1]))
    grouped=defaultdict(list)
    for row in candidates:grouped[row['state_key']].append(row)
    expected=[];pruned=[]
    for group in grouped.values():
        ordered=sorted(group,key=lambda row:hashlib.sha256(('selfplay-pilot-rawcap-v1/'+row['id']).encode()).digest())
        expected+=ordered[:4];pruned += [r['id'] for r in ordered[4:]]
    expected.sort(key=lambda r:r['id']);assert expected==retained
    summary=manifest['summary'];assert summary['family_exclusions']==exclusions and summary['retained_rows']==len(expected)
    assert summary['pruned_row_ids']==sorted(pruned)
    assert len({r['family'] for r in retained})==summary['admitted_families']
    return dict(games=len(games),terminal_games=sum(g['status']=='terminal' for g in games.values()),
        decisions=sum(len(v) for v in all_decisions.values()),bonus_choices=bonus,retained_rows=len(retained),
        admitted_families=summary['admitted_families'],excluded_families=exclusions,
        all_root_noise_priors_behavior_and_RNG_independently_rederived=True,
        literal_tree_replays=checked_trees,all_terminal_player0_targets_and_raw_visits_exact=True,
        eligibility_active_ge25_before_unchanged_protection=True,rawstate_cap4_independently_equal=True)


class SelfplayPilotIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()

    def fresh(self,name):
        path=self.base/name;path.mkdir();return path

    def backup(self,source,destination):
        with closing(dbopen(source)) as db,closing(sqlite3.connect(destination)) as target:db.backup(target)

    def worker(self,kind,spec,path,torch=False):
        spec={**spec,'kind':kind,'output':str(path.with_suffix('.json'))};save(path.with_suffix('.spec.json'),spec)
        command=[torch_python() if torch else sys.executable,str(ROOT/'tools/selfplay-pilot-audit.py'),str(path.with_suffix('.spec.json'))]
        run_child(command,path.with_suffix('.log'),timeout=300)
        result=load(path.with_suffix('.json'));self.assertTrue(result['passed']);return result

    def test_01_baseline_current_model_frozen_search_training_and_evaluation_contract(self):
        baseline=load(BASELINE)
        self.assertEqual((baseline['schema'],baseline['inputs_count']),('selfplay-pilot-prior-immutability-v1',2839))
        self.assertEqual(len(baseline['inputs']),2839)
        for name,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA)
        self.assertEqual(set(load(MODEL)['model']['weights']),ALL_PARAMETERS)
        from congklak import selfplay_training_pilot as producer,selfplay_pilot_benchmark as bench
        producer.admit();gate=bench.pretraining_witness()
        self.assertTrue(gate['passed']);self.assertFalse(gate['games_performed_by_gate'])
        s=producer.SP;t=producer.TRAIN
        self.assertEqual((s['families'],s['runs_per_family'],s['simulations'],s['max_nodes'],s['max_choices'],s['c_puct']),
            (16,2,64,2048,512,.25));self.assertEqual(s['root_noise'],dict(alpha=.3,fraction=.25))
        self.assertEqual((s['temperature_first_choices'],s['rawstate_observation_cap'],s['minimum_admitted_families']),(8,4,8))
        self.assertEqual(s['min_training_active'],25)
        self.assertEqual((t['steps'],t['batch_size'],t['seed'],t['checkpoint_every'],t['lr'],t['parameters']),
            (200,128,20261025,20,.0001,12872))
        self.assertEqual((t['device'],t['cpu_threads'],t['value_huber_delta'],t['policy_loss_weight']),('cpu',1,.1,.25))
        self.assertFalse(t['teacher_rehearsal']);self.assertFalse(t['amp'])
        projection=load(bench.PROJECTION);self.assertEqual(len(projection['arena_starts']),12)
        self.assertEqual(len(projection['cases']),32);self.assertFalse(projection['sealed_final_test_consumed'])
        self.assertEqual(Counter(r['stratum'].split('/')[-1] for r in projection['arena_starts']),dict(ordinary=4,capture=4,bonus=4))
        self.assertFalse(set(r['family'] for r in projection['arena_starts'])&set(projection['prior_arena_families']))
        EVIDENCE['frozen_contract']=dict(prior_inputs=2839,all_current=True,current_model_exact=True,
            source_policy_and_value_both_loaded=True,trainable_parameters=12872,trainable_tensors=8,
            selfplay_games_requested=32,minimum_admitted_families=8,fixed_final200=True,
            arena_games=96,exact_guard_roots=32,exact_guard_measurements=256,
            no_sealed_release_test_or_promotion=True)

    def test_02_independent_true_roles_reservations_start_matching_and_exposure_contract(self):
        from congklak import selfplay_training_pilot as producer
        q,f,latest,metadata,proof=independent_registry();registry=load(producer.REGISTRY)
        self.assertEqual(set(registry['quotients']),q);self.assertEqual(set(registry['families']),f)
        self.assertEqual(registry['metadata_only_reservations'],dict(rows=9000,
            semantic_sha256=config.digest(metadata),blanket_evaluation_ban=False,train_rows=8000))
        self.assertGreater(proof['train_rows_not_evaluation_banned'],1000)
        self.assertEqual(registry['latest_roles'],{key:dict(family=r['family'],cohort=r['cohort'],quotient=r['quotient'],root_sha256=config.digest(r)) for key,r in latest.items()})
        candidates=[r for r in latest.values() if r['cohort']=='training_candidate']
        wanted=matching_reference(candidates,q,f)
        self.assertEqual(registry['starts'],wanted);self.assertEqual(len(wanted),16)
        self.assertEqual(len({r['root']['family'] for r in wanted}),16)
        self.assertEqual(Counter(r['root']['stratum'] for r in wanted),dict.fromkeys(producer.SP['quotas'],4))
        for item in wanted:
            r=item['root'];self.assertEqual(r['cohort'],'training_candidate');self.assertNotIn(r['quotient'],q);self.assertNotIn(r['family'],f)
        self.assertTrue(registry['ancestral_exposure_not_removed']);self.assertFalse(registry['sealed_target_bodies_read'])
        EVIDENCE['role_registry']=dict(**proof,new_validation_roots_protected=10,new_diagnostic_roots_protected=9,
            exactly16_distinct_start_families=True,iterative_matching_equal=True,
            four_starts_per_declared_stratum=True,new39_H10_targets_not_gradient_labels=True,
            family_uniform_then_row_uniform=True,raw_state_cap4_not_independence_claim=True)

    def test_03_literal_noisy_PUCT_bonus_orientation_raw_visits_and_legal_policy(self):
        from congklak import selfplay_training_pilot as producer,quality_search
        registry=load(producer.REGISTRY);states=[item['root']['state'] for item in registry['starts']]
        selected=[]
        for state in states:
            q=norm(state)
            if any(literal.apply(q,a).bonus and not literal.apply(q,a).terminal for a in literal.legal_actions(q)):
                selected.append(state)
            if len(selected)==2:break
        self.assertTrue(selected)
        for actor in (0,1):
            state=copy.deepcopy(selected[0])
            if state['controller']!=actor:state=dict(pits=state['pits'][::-1],stores=state['stores'][::-1],controller=actor)
            selected.append(state)
        def prediction(p):
            active=sum(map(sum,p['pits']));d=p['stores'][0]-p['stores'][1]
            return d+active*.375*(1 if p['controller']==0 else -1),[.13*(a-3) for a in range(7)]
        class Evaluator:
            max_batch=1
            def evaluate_many(self,positions):
                return [dict(value_player0=prediction(p)[0],policy_logits=prediction(p)[1],
                    feature_simulations=0,feature_deposits=0) for p in positions]
        for i,state in enumerate(selected):
            a=random.Random(7123+i);b=random.Random(7123+i)
            expected,nodes=literal_puct(state,64,a,prediction,noise=producer.SP['root_noise'])
            actual=quality_search.search(state,Evaluator(),simulations=64,max_nodes=2048,
                rng=b,root_noise=producer.SP['root_noise'],c_puct=.25,time_ms=0)
            self.assertEqual(actual['visits'],expected['visits']);self.assertEqual(actual['nodes'],nodes)
            self.assertEqual(actual['root_priors'],expected['priors']);self.assertEqual(a.getstate(),b.getstate())
            self.assertEqual(actual['policy'],[n/64 for n in expected['visits']])
            self.assertAlmostEqual(actual['estimated_value_player0'],98*sum(expected['sums'])/64,places=12)
            for pit,n in enumerate(expected['visits']):
                if n:self.assertAlmostEqual(actual['action_mean_value_player0'][pit],98*expected['sums'][pit]/n,places=12)
                else:self.assertIsNone(actual['action_mean_value_player0'][pit])
        EVIDENCE['literal_search']=dict(roots=len(selected),simulations_each=64,
            root_noise_priors_visits_Q_and_RNG_independently_equal=True,
            both_actor_orientations_and_bonus_turns=True,backup_fixed_player0_not_alternating=True,
            targets_raw_visits_not_behavior_onehot=True)

    def test_04_actual_selfplay_decision_kills_terminal_targets_corruption_and_no_work(self):
        from congklak import selfplay_training_pilot as producer
        base=self.fresh('selfplay-recovery');fresh=base/'fresh'
        def command(path):return [sys.executable,str(ROOT/'tools/selfplay-training-pilot.py'),'generate','--fixture','--run',str(path)]
        run_child(command(fresh),base/'fresh.log',timeout=300)
        fresh_manifest,fresh_rows=producer.load_replay(fresh);proof=independent_replay(fresh,check_trees=True)
        self.assertGreaterEqual(proof['admitted_families'],1);records=[]
        for suffix,prefix in (('precommit',1),('committed',2)):
            event='selfplay_pilot_decision_'+suffix;run=base/suffix;probe=base/(suffix+'-kill')
            marker=kill_barrier(command(run),probe,event,env={'SELFPLAY_PILOT_BLOCK_EVENT':event,
                'SELFPLAY_PILOT_BLOCK_STEP':str(prefix),'SELFPLAY_PILOT_MARKER':str(probe/'declared-marker.json')},
                condition=lambda v,n=prefix:v['committed_decisions']==n,timeout=300)
            committed=sqlite_rows(run/'work.sqlite3','sp_decisions','gameid,choice');self.assertEqual(len(committed),prefix)
            self.backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
            run_child(command(run),base/(suffix+'-resume.log'),timeout=300)
            final=sqlite_rows(run/'work.sqlite3','sp_decisions','gameid,choice');self.assertEqual(final[:prefix],committed)
            actual_manifest,actual_rows=producer.load_replay(run)
            self.assertEqual(sp_semantic(actual_manifest),sp_semantic(fresh_manifest))
            self.assertEqual(sp_semantic(actual_rows),sp_semantic(fresh_rows))
            self.assertEqual(rows(run/'work.sqlite3','sp_games','id'),rows(fresh/'work.sqlite3','sp_games','id'))
            stable=snapshot(run)
            with mock.patch.object(producer.quality_search,'search',side_effect=AssertionError('Completed selfplay searched')),mock.patch.object(Native,'analyze',side_effect=AssertionError('Teacher search')):
                producer.generate(run,fixture=True);producer.load_replay(run)
            self.assertEqual(snapshot(run),stable);self.assertNotIn('torch',sys.modules)
            # A fully rehashed wrong actor/terminal return still fails literal replay.
            broken=base/(suffix+'-bad');shutil.copytree(run,broken)
            with closing(sqlite3.connect(broken/'work.sqlite3')) as db:
                gid,body=db.execute('SELECT id,body FROM sp_games ORDER BY id LIMIT 1').fetchone();value=json.loads(body);value['terminal_margin']+=2
                db.execute('UPDATE sp_games SET body=?,hash=? WHERE id=?',(config.canonical(value),config.digest(value),gid));db.commit()
            with self.assertRaises(Exception):producer.load_replay(broken)
            records.append(dict(event=event,actual_process_kill=True,committed_decisions=prefix,
                marker=marker,probe=relative(probe),run=relative(run),
                committed_prefix_exact=True,resumed_moves_returns_policy_RNG_equal_fresh=True,
                uncommitted_clock_fields_excluded_only=True,completed_bytes_mtimes_unchanged=True))
        # Wrong completed marker cannot repair a missing selfplay publication.
        bad=base/'bad-marker';shutil.copytree(fresh,bad);(bad/'selfplay-manifest.json').unlink()
        with closing(sqlite3.connect(bad/'work.sqlite3')) as db:
            value=json.loads(db.execute("SELECT body FROM meta WHERE key='selfplay_manifest'").fetchone()[0]);value['summary']['retained_rows']+=1
            db.execute("UPDATE meta SET body=?,hash=? WHERE key='selfplay_manifest'",(config.canonical(value),config.digest(value)));db.commit()
        with self.assertRaises(Exception):producer.generate(bad,fixture=True)
        self.assertFalse((bad/'selfplay-manifest.json').exists())
        # Valid missing publication is recoverable with no new search.
        repaired=base/'repair';shutil.copytree(fresh,repaired);expected=(repaired/'selfplay-manifest.json').read_bytes();(repaired/'selfplay-manifest.json').unlink()
        with mock.patch.object(producer.quality_search,'search',side_effect=AssertionError('Repair searched')):producer.generate(repaired,fixture=True)
        self.assertEqual((repaired/'selfplay-manifest.json').read_bytes(),expected)
        EVIDENCE['live_selfplay_recovery']=dict(tests=records,fresh_replay=relative(fresh),literal_fixture=proof,
            actual_decision_pre_post_kills=True,corrupt_rehashed_terminal_and_marker_refused=True,
            valid_missing_publication_repaired_without_search=True,all_raw_retained=True)

    def test_05_actual_checkpoint_file_catalog_kills_exact_joint_resume_and_CPU_completed(self):
        from congklak import selfplay_training_pilot as producer
        # This method is in the same preflight process, sharing the exact replay identity.
        replay=ROOT/EVIDENCE['live_selfplay_recovery']['fresh_replay'];base=self.fresh('training-recovery');fresh=base/'fresh'
        def command(path):return [torch_python(),str(ROOT/'tools/selfplay-training-pilot.py'),'train','--fixture','--run',str(path),'--replay',str(replay)]
        run_child(command(fresh),base/'fresh.log',timeout=300);reference=pt_catalog(fresh)
        self.assertEqual([r[1] for r in reference],[2,4,6]);records=[]
        for event,expected_steps in (('checkpoint_file_ready',[2]),('checkpoint_committed',[2,4])):
            run=base/event;probe=base/(event+'-kill')
            marker=kill_barrier(command(run),probe,event,env={'SELFPLAY_PILOT_BLOCK_EVENT':event,
                'SELFPLAY_PILOT_BLOCK_STEP':'4','SELFPLAY_PILOT_MARKER':str(probe/'declared-marker.json')},
                condition=lambda v:v['step']==4,timeout=300)
            before=pt_catalog(run);self.assertEqual([r[1] for r in before],expected_steps)
            self.backup(run/'catalog.sqlite3',probe/'validated-prefix.sqlite3')
            committed={r[2]:(sha(run/r[2]),(run/r[2]).stat().st_mtime_ns) for r in before}
            run_child(command(run),base/(event+'-resume.log'),timeout=300);actual=pt_catalog(run)
            self.assertEqual(actual[:len(before)],before);self.assertEqual(load(fresh/'model.json'),load(run/'model.json'))
            for name,witness in committed.items():self.assertEqual((sha(run/name),(run/name).stat().st_mtime_ns),witness)
            stateproof=self.worker('compare_state',dict(left=str(fresh/reference[-1][2]),right=str(run/actual[-1][2])),base/(event+'-state'),torch=True)
            completed=self.worker('completed_train',dict(run=str(run),replay=str(replay),fixture=True),base/(event+'-completed'))
            records.append(dict(event=event,actual_process_kill=True,marker=marker,probe=relative(probe),run=relative(run),
                committed_prefix_steps=expected_steps,resumed_step=6,committed_PT_bytes_mtimes_exact=True,
                exact_state_proof=stateproof,completed_no_work_proof=completed,
                orphan_file_retained=event=='checkpoint_file_ready'))
        optimizer=self.worker('joint_optimizer',dict(run=str(fresh),fixture=True),base/'independent-six-step',torch=True)
        self.assertEqual(optimizer['steps_replayed'],6)
        EVIDENCE['live_training_recovery']=dict(tests=records,runtime=torch_python(),device='cpu',fixture_steps=6,
            batch_size=32,checkpoint_every=2,actual_joint_checkpoint_pre_post_kills=True,
            same_replay_path_identity=True,all_raw_PT_optimizer_and_RNG_proofs_retained=True)

    def test_06_all_production_episodes_noise_returns_admission_and_200_optimizer_updates(self):
        from congklak import selfplay_training_pilot as producer
        proof=independent_replay(producer.DEFAULT,check_trees=True)
        self.assertEqual(proof['games'],32);self.assertGreaterEqual(proof['admitted_families'],8)
        self.assertEqual(proof['terminal_games'],32)
        optimizer=self.worker('joint_optimizer',dict(run=str(producer.DEFAULT)),self.base/'production-optimizer',torch=True)
        self.assertEqual(optimizer['steps_replayed'],200);self.assertEqual(optimizer['draws_replayed'],25600)
        self.assertEqual(len(optimizer['checkpoints']),10)
        with mock.patch.object(producer,'runtime',side_effect=AssertionError('CPU witness activated runtime')):witness=producer.witness()
        self.assertTrue(witness['passed']);self.assertTrue({'policy.weight','layers.4.weight'}<=set(optimizer['changed_tensors']))
        EVIDENCE['production_selfplay']=proof;EVIDENCE['production_optimizer_audit']=optimizer
        EVIDENCE['joint_model']=dict(source_model_file_sha256=MODEL_SHA,candidate_file_sha256=sha(producer.DEFAULT/'model.json'),
            current_model_loaded_exact=True,all8_tensors_trainable=True,policy_and_value_changed=True,
            fixed_final_step200=True,CPU_witness_no_Torch=True,teacher_rehearsal=False)

    def test_07_benchmark_decision_game_pre_post_kills_prefix_deadline_and_no_work(self):
        from congklak import selfplay_pilot_benchmark as bench
        base=self.fresh('benchmark-recovery');records=[];plan=bench.read_plan(fixture=True)
        def command(path,out):return [sys.executable,str(ROOT/'tools/benchmark-selfplay-pilot.py'),'run',
            '--fixture','--fixture-control','--run',str(path),'--output',str(out)]
        for unit in ('decision','game'):
            for suffix,prefix in (('precommit',1),('committed',2)):
                event=('selfplay_pilot_benchmark_decision_' if unit=='decision' else 'selfplay_pilot_game_')+suffix
                run=base/(unit+'-'+suffix);out=base/(unit+'-'+suffix+'.json');probe=base/(unit+'-'+suffix+'-kill')
                marker=kill_barrier(command(run,out),probe,event,env={'SELFPLAY_PILOT_BLOCK_EVENT':event,
                    'SELFPLAY_PILOT_BLOCK_STEP':str(prefix),'SELFPLAY_PILOT_BLOCK_MARKER':str(probe/'declared-marker.json')},
                    condition=lambda v,n=prefix:v['completed']==n,timeout=300)
                self.backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3')
                before_decisions=sqlite_rows(run/'work.sqlite3','decisions','game,choice');before_games=sqlite_rows(run/'work.sqlite3','games','idx')
                self.assertEqual(len(before_decisions) if unit=='decision' else len(before_games),prefix)
                run_child(command(run,out),base/(unit+'-'+suffix+'-resume.log'),timeout=300)
                final_decisions=sqlite_rows(run/'work.sqlite3','decisions','game,choice');final_games=sqlite_rows(run/'work.sqlite3','games','idx')
                self.assertEqual(final_decisions[:len(before_decisions)],before_decisions);self.assertEqual(final_games[:len(before_games)],before_games)
                self.assertEqual(len(final_games),8);choices=0
                for _,_,_,_,body,digest in final_games:
                    game=json.loads(body);self.assertEqual(config.digest(game),digest);choices+=literal_game(self,game)
                with closing(dbopen(run/'work.sqlite3')) as db:bench.validate_saved_units(db,plan,Native())
                stable=snapshot(run);exports={p:(sha(p),p.stat().st_mtime_ns) for p in (out,out.with_name(out.stem+'-resources.json'))}
                with mock.patch.object(bench,'decide',side_effect=AssertionError('Completed benchmark decided')),mock.patch.object(Native,'analyze',side_effect=AssertionError('Teacher search')):
                    bench.run(path=run,output=out,fixture=True,fixture_control=True,on_event=lambda _:None)
                self.assertEqual(snapshot(run),stable)
                for p,witness in exports.items():self.assertEqual((sha(p),p.stat().st_mtime_ns),witness)
                valid=base/(unit+'-'+suffix+'-valid.sqlite3');self.backup(run/'work.sqlite3',valid)
                broken=base/(unit+'-'+suffix+'-bad');shutil.copytree(run,broken);badout=base/(unit+'-'+suffix+'-missing.json')
                shutil.copy2(out.with_name(out.stem+'-resources.json'),badout.with_name(badout.stem+'-resources.json'))
                with closing(sqlite3.connect(broken/'work.sqlite3')) as db:
                    value=json.loads(db.execute("SELECT body FROM meta WHERE key='selfplay_pilot_benchmark_complete'").fetchone()[0]);value['games']+=1
                    db.execute("UPDATE meta SET body=?,hash=? WHERE key='selfplay_pilot_benchmark_complete'",(config.canonical(value),config.digest(value)));db.commit()
                self.assertNotEqual(run_child(command(broken,badout),base/(unit+'-'+suffix+'-refused.log'),success=False,timeout=300),0)
                self.assertFalse(badout.exists())
                records.append(dict(event=event,unit=unit,actual_process_kill=True,marker=marker,probe=relative(probe),
                    run=relative(run),validated_complete_journal=relative(valid),committed_prefix=prefix,
                    all_committed_decisions_and_games_exact=True,resumed_terminal_games=8,literal_choices=choices,
                    completed_bytes_mtimes_unchanged=True,corrupt_marker_refused_before_publication=True,
                    timed_uncommitted_choices_not_required_equal_fresh=True))
        EVIDENCE['live_benchmark_recovery']=dict(tests=records,actual_decision_and_game_pre_post_kills=True,
            fixture_control_substitution_explicit=True,production_no_substitution=True,raw_markers_prefix_backups_logs_retained=True)

    def test_08_all_equal_time_games_CIs_exact_guards_overlap_exposure_and_closed_reruns(self):
        from congklak import selfplay_pilot_benchmark as bench,selfplay_training_pilot as producer
        plan=bench.read_plan();path,out=bench.paths();report=load(out);games=rows(path/'work.sqlite3','games','idx')
        self.assertEqual((len(games),report['games'],report['cases'],report['scalar_measurements']),(96,96,32,256))
        choices=sum(literal_game(self,g) for g in games.values());paired={};primary={}
        def interval(values):
            rng=random.Random(plan['bootstrap_seed']);samples=[]
            for _ in range(plan['bootstrap_samples']):samples.append(statistics.mean(values[min(len(values)-1,int(rng.random()*len(values)))] for _ in values))
            samples.sort();return [samples[int(.025*len(samples))],samples[int(.975*len(samples))]]
        for comparison in plan['arena_comparisons']:
            selected=[g for g in games.values() if g['comparison']==comparison['id']];margins={};scores=[]
            for family in range(12):
                pair=[g for g in selected if g['pair']==family];self.assertEqual(len(pair),2)
                if all(g['status']=='terminal' for g in pair):margins[family]=statistics.mean(g['candidate_final_margin'] for g in pair);scores.append(statistics.mean(g['candidate_match_score'] for g in pair))
            values=list(margins.values());paired[comparison['id']]=margins;observed=report['arenas'][comparison['id']]
            self.assertEqual(observed['complete_pairs'],len(values));self.assertEqual(observed['primary_margin_bootstrap_95_interval'],interval(values) if values else None)
            self.assertEqual(observed['primary_paired_mean_final_margin_seeds'],statistics.mean(values) if values else None)
            fallback=Counter()
            for g in selected:fallback.update(g['fallbacks'])
            gate=bool(len(values)==12 and not sum(fallback.values()) and interval(values)[0]>0)
            self.assertEqual(observed['primary_strength_gate_passed'],gate);primary[comparison['id']]=dict(mean=observed['primary_paired_mean_final_margin_seeds'],interval=observed['primary_margin_bootstrap_95_interval'],gate=gate)
        contrasts={}
        for base in bench.BASELINES:
            treatment=paired['selfplay_challenger_vs_'+base];control=paired['selfplay_control_vs_'+base]
            values=[treatment[i]-control[i] for i in sorted(treatment.keys()&control.keys())]
            expected=dict(mean=statistics.mean(values) if len(values)==12 else None,interval=interval(values) if len(values)==12 else None)
            row=report['secondary_factor_contrasts'][base];self.assertEqual(row['paired_mean_seed_margin_difference'],expected['mean']);self.assertEqual(row['confidence_interval'],expected['interval']);contrasts[base]=expected
        cases=rows(path/'work.sqlite3','cases','idx');measurements=0
        for index,case in cases.items():
            reference=plan['cases'][index]['reference'];state=plan['cases'][index]['state'];actor=1 if state['controller']==0 else -1
            q={a['pit']:a['value_player0'] for a in reference['action_scores']}
            for ms,engines in case['measurements'].items():
                for name,saved in engines.items():
                    bench.validate_envelope(saved['decision'],state,int(ms),plan,name);decision=saved['decision']['decision'];measurements+=1
                    if decision['usable_decision']:
                        action=decision['chosen'];self.assertEqual(saved['reference_regret'],actor*(reference['target_player0']-q[action]));self.assertEqual(saved['reference_optimal'],action in reference['best_moves'])
                    self.assertEqual(decision['requested_ms'],int(ms))
        self.assertEqual(measurements,256)
        for ms in ('25','100'):
            group=report['endgame_guard'][ms]
            for name in bench.CANDIDATES:
                saved=[c['measurements'][ms][name] for c in cases.values()];usable=[x for x in saved if x['decision']['decision']['usable_decision']]
                self.assertEqual(group[name]['complete_choices'],len(usable));self.assertEqual(group[name]['optimal_choices'],sum(bool(x['reference_optimal']) for x in usable))
                self.assertEqual(group[name]['mean_exact_regret'],statistics.mean(x['reference_regret'] for x in usable) if usable else None)
        _,replay=producer.load_replay();qset={r['quotient'] for r in replay};fset={r['family'] for r in replay};overlap=[]
        for index,g in games.items():
            q=norm(g['start'])
            for choice,action in enumerate(g['moves']):
                p=position(q)
                if sum(map(sum,p['pits']))>4 and (quotient(p) in qset or g['family'] in fset):overlap.append((index,choice))
                q=literal.apply(q,action).position
        observed=report['trajectory_replay_overlap'];self.assertEqual([(x['game'],x['choice']) for x in observed['collisions']],overlap)
        self.assertEqual(observed['contaminated_evaluation'],bool(overlap))
        sampling=load(ROOT/'build/selfplay-pilot-sampling-review.json');groups=defaultdict(list)
        for row in replay:groups[row['family']].append(row['id'])
        rng=random.Random(20261025);familyids=sorted(groups);draws=Counter()
        for _ in range(25600):
            f=familyids[rng.randrange(len(familyids))];draws[groups[f][rng.randrange(len(groups[f]))]]+=1
        self.assertEqual(sampling['draws'],25600)
        for row,exposure in zip(replay,sampling['row_exposures']):
            probability=1/(len(groups)*len(groups[row['family']]))
            self.assertEqual(exposure['id'],row['id']);self.assertEqual(exposure['sampling_probability'],probability);self.assertEqual(exposure['reconstructed_draws'],draws[row['id']])
        stable=snapshot(producer.DEFAULT);arena_stable=snapshot(path);exports={p:(sha(p),p.stat().st_mtime_ns) for p in (out,out.with_name(out.stem+'-resources.json'))}
        with mock.patch.object(producer,'runtime',side_effect=AssertionError('Completed imported Torch')),mock.patch.object(producer.quality_search,'search',side_effect=AssertionError('Completed selfplay searched')),mock.patch.object(bench,'decide',side_effect=AssertionError('Completed arena decided')),mock.patch.object(Native,'analyze',side_effect=AssertionError('Teacher search')):
            producer.generate();producer.train();producer.witness();bench.run(on_event=lambda _:None);bench.closed_witness()
        self.assertEqual(snapshot(producer.DEFAULT),stable);self.assertEqual(snapshot(path),arena_stable)
        for p,witness in exports.items():self.assertEqual((sha(p),p.stat().st_mtime_ns),witness)
        for name,digest in load(BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['equal_time_final']=dict(games=96,literal_choices=choices,cases=32,exact_measurements=256,
            primary=primary,secondary_contrasts=contrasts,all_six_intervals_independently_rederived=True,
            actual_trajectory_overlap_hits=len(overlap),contaminated_evaluation=bool(overlap),development_only=True,
            raw_numeric_gates_not_promotion=True,all_current_saved_envelopes_validated=True)
        EVIDENCE['exposure_and_closed']=dict(draws=25600,family_uniform_row_uniform_independently_replayed=True,
            all_row_probabilities_and_draw_counts_equal=True,completed_no_search_no_Torch_no_optimizer_work=True,
            all_durable_bytes_mtimes_unchanged=True,prior2839_inputs_unchanged=True)


GROUPS={
    'preflight':[
        'test_01_baseline_current_model_frozen_search_training_and_evaluation_contract',
        'test_02_independent_true_roles_reservations_start_matching_and_exposure_contract',
        'test_03_literal_noisy_PUCT_bonus_orientation_raw_visits_and_legal_policy',
        'test_04_actual_selfplay_decision_kills_terminal_targets_corruption_and_no_work',
        'test_05_actual_checkpoint_file_catalog_kills_exact_joint_resume_and_CPU_completed'],
    'model':['test_06_all_production_episodes_noise_returns_admission_and_200_optimizer_updates'],
    'recovery':['test_07_benchmark_decision_game_pre_post_kills_prefix_deadline_and_no_work'],
    'final':['test_08_all_equal_time_games_CIs_exact_guards_overlap_exposure_and_closed_reruns']}


def collect():
    inputs={};evidence={};groups=[];covered=[]
    for group,methods in GROUPS.items():
        path=WORK/(group+'.json');body=load(path)
        assert body['passed'] and body['tests_run']==len(methods) and body['covered_methods']==methods
        assert not any(body[k] for k in ('errors','failures','skips'))
        for name,digest in body['inputs'].items():
            assert sha(ROOT/name)==digest,'Independent input changed: '+name
            assert name not in inputs or inputs[name]==digest;inputs[name]=digest
        assert not(set(evidence)&set(body['evidence']));evidence.update(body['evidence'])
        inputs[relative(path)]=sha(path);covered+=methods
        groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods))
    assert len(covered)==len(set(covered))==8
    result=dict(schema='selfplay-pilot-independent-audit-v1',passed=True,tests_run=8,
        failures=0,errors=0,skips=0,covered_methods=covered,groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/selfplay-pilot-audit.json'
    if output.exists():assert load(output)==result,'Immutable independent aggregate differs'
    else:save(output,result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=[*GROUPS,'collect']);args=parser.parse_args()
    if args.group=='collect':
        result=collect();print(config.canonical(dict(event='selfplay_pilot_independent_audit',group='collect',passed=True,tests=result['tests_run'])),flush=True);return 0
    started=time.monotonic();methods=GROUPS[args.group]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SelfplayPilotIndependentAudit(name) for name in methods))
    from congklak import selfplay_training_pilot as producer,selfplay_pilot_benchmark as bench
    files=[Path(__file__),ROOT/'tools/selfplay-pilot-audit.py',BASELINE,MODEL,producer.REGISTRY,
        producer.PLAN,producer.IMPLEMENTATION,bench.PLAN,bench.PROJECTION,bench.IMPLEMENTATION,
        ROOT/'congklak/selfplay_training_pilot.py',ROOT/'tools/selfplay-training-pilot.py',
        ROOT/'congklak/selfplay_pilot_benchmark.py',ROOT/'tools/benchmark-selfplay-pilot.py']
    files += [p for p in SelfplayPilotIndependentAudit.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if args.group in ('model','final'):
        files += [p for p in producer.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
        files += [producer.WITNESS]
    if args.group=='final':
        arena,out=bench.paths();files += [p for p in arena.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
        files += [out,out.with_name(out.stem+'-resources.json'),ROOT/'build/selfplay-pilot-sampling-review.json',ROOT/'tools/review-selfplay-pilot-weights.py']
    body=dict(schema='selfplay-pilot-independent-group-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        duration_seconds=time.monotonic()-started,retained_directory=relative(SelfplayPilotIndependentAudit.base),
        inputs=pins([p for p in files if p.is_file()]),evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors])
    save(WORK/(args.group+'.json'),body)
    print(config.canonical(dict(event='selfplay_pilot_independent_audit',group=args.group,passed=body['passed'],tests=result.testsRun)),flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
