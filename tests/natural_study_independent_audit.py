"""Independent natural 50–74 history and paired H8/H10 study verification."""
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
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,key_of,quotient,mechanism,literal_family,rows,semantic,kill_barrier,run_child,snapshot
import reference as literal

WORK=ROOT/'build/natural-study-audit';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
MEANS=('value_mae_h8','value_mae_h10','value_mae_change_h10_minus_h8','policy_regret_h8','policy_regret_h10','policy_regret_change_h10_minus_h8','absolute_target_shift','signed_actor_target_shift','H10_regret_H8_chosen','H10_regret_H8_tie_min','H10_regret_H8_tie_max','action_Q_shift_mae')

def old_reference():
    registry=load(ROOT/'build/data-pilot-registry.json');run=ROOT/'data/ordinary-data-pilot-v1'
    roots=rows(run/'work.sqlite3','roots','state');labels=rows(run/'work.sqlite3','labels','state');families=rows(run/'work.sqlite3','families','idx')
    protected=set(registry['quotients'])|{quotient(r['state']) for r in roots.values()}
    protected_families=set(registry['families'])|{f['family'] for f in families.values()}
    followups={k:r for k,r in roots.items() if r['band']=='50-74'}
    assert len(roots)==62 and len(families)==256 and len(followups)==15
    return protected,protected_families,followups,{k:labels[k] for k in followups}

def wrap_reference(row,role,item=None):
    fields=('state_key','state','legal_mask','family','cohort','band','stratum','quotient','active')
    return {**{k:row[k] for k in fields},'study_role':role,'root':row,'source_root_sha256':config.digest(row),'prior_h8_item_sha256':config.digest(item) if item is not None else None}

def select_reference(pool,settings,registry):
    observations=[r for f in pool.values() for r in f['candidates'] if r['band']=='50-74'];owners=defaultdict(set)
    for r in observations:owners[r['quotient']].add(r['cohort'])
    collisions={q for q,v in owners.items() if len(v)>1};protected=set(registry['quotients']);families=set(registry['families'])
    available=sorted((r for r in observations if r['quotient'] not in protected and r['quotient'] not in collisions and r['family'] not in families),key=lambda r:(r['provenance']['family_index'],r['provenance']['step'],r['state_key']))
    selected={};inherited={};usedq=set();usedfamily=set();counts=Counter()
    for r in available:
        c=r['cohort']
        if counts[c]<settings['fresh_caps'][c] and r['quotient'] not in usedq and r['family'] not in usedfamily:
            selected[r['state_key']]=wrap_reference(r,'fresh');usedq.add(r['quotient']);usedfamily.add(r['family']);counts[c]+=1
    for key in sorted(registry['followups'])[:settings['followups']]:
        old=registry['followups'][key];selected[key]=wrap_reference(old['root'],'followup',old['h8']);inherited[key]=old['h8']
    return dict(sorted(selected.items())),dict(sorted(inherited.items())),dict(ordinary_50_74_observations=len(observations),unique_candidate_quotients=len(owners),cross_cohort_collisions=len(collisions),protected_observations=sum(r['quotient'] in protected for r in observations),fresh_counts={c:counts[c] for c in settings['fresh_caps']},shortfall={c:settings['fresh_caps'][c]-counts[c] for c in settings['fresh_caps']},followups=len(inherited),observed_legal_action_counts=dict(sorted(Counter(str(r['legal_mask'].bit_count()) for r in observations).items())))

def check_pvs(test,row,item,depth):
    state=row['state'];report=item['report'];label=item['label'];legal=list(literal.legal_actions(norm(state)))
    values={r['pit']:r['value_player0'] for r in label['action_scores']};sign=1 if state['controller']==0 else -1
    test.assertEqual(list(values),legal);best=max(sign*v for v in values.values());test.assertEqual(sign*label['target_player0'],best);test.assertEqual(label['best_moves'],[a for a in legal if sign*values[a]==best])
    test.assertEqual(item['root_sha256'],row['source_root_sha256']);test.assertEqual(config.digest(row['root']),item['root_sha256']);test.assertEqual(report['choice_depth'],depth);test.assertEqual(label['choice_depth'],depth)
    test.assertEqual(label['target_player0'],report['depth_result']['value_player0']);test.assertEqual(label['action_scores'],[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in report['candidates']]);test.assertEqual(label['best_moves'],report['best_moves']);test.assertEqual(label['coverage'],report['coverage']);test.assertEqual(label['work'],{k:report[k] for k in ('nodes','deposits','elapsed_ms','peak_process_bytes')})
    paths=choices=0
    for pv,value in [(report['pv'],label['target_player0'])]+[(c['pv'],c['depth_result']['value_player0']) for c in report['candidates']]:
        q=norm(state)
        for move in pv:
            test.assertEqual(move['actor'],q.controller);test.assertIn(move['pit'],literal.legal_actions(q));t=literal.apply(q,move['pit']);q=t.position
            test.assertEqual(move['terminal_after'],t.terminal);test.assertEqual(move['bonus_choice'],t.bonus and not t.terminal);test.assertEqual(move['forced_passes_after'],[dict(zip(('from','to'),v)) for v in t.final_passes]);choices+=1
        test.assertTrue(literal.normalize(q).terminal or len(pv)==depth);test.assertEqual(literal.evaluate(q),value);paths+=1
    return paths,choices

def check_summary(test,summary,values):
    paired=[r for r in values if r['complete_pair']];choices=[r for r in paired if r['legal_action_count']>1]
    test.assertEqual(summary['selected_roots'],len(values));test.assertEqual(summary['families'],len({r['family'] for r in values}));test.assertEqual(summary['complete_pairs'],len(paired));test.assertEqual(summary['unpaired_roots'],len(values)-len(paired));test.assertEqual(summary['h8_complete'],sum(r['h8_complete'] for r in values));test.assertEqual(summary['h10_complete'],sum(r['h10_complete'] for r in values));test.assertEqual(summary['choice_bearing_pairs'],len(choices));test.assertEqual(summary['single_legal_action_pairs'],len(paired)-len(choices))
    for key in MEANS:
        test.assertEqual(summary['paired_means'][key],statistics.mean(r[key] for r in paired) if paired else None)
        test.assertEqual(summary['paired_choice_bearing_means'][key],statistics.mean(r[key] for r in choices) if choices else None)
    for key in ('best_sets_changed','best_sets_disjoint','H8_choice_still_best','all_H8_ties_still_best'):test.assertEqual(summary[key],sum(r[key] for r in paired))
    for key,op in (('model_MAE_lower_at_H10',lambda x:x<0),('model_MAE_higher_at_H10',lambda x:x>0),('model_MAE_equal_at_H10',lambda x:x==0)):test.assertEqual(summary[key],sum(op(r['value_mae_change_h10_minus_h8']) for r in paired))
    test.assertEqual(summary['unpaired_statuses'],dict(sorted(Counter(r['h8_status']+'/'+r['h10_status'] for r in values if not r['complete_pair']).items())))

class NaturalStudyIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):
        p=self.base/name;p.mkdir();return p
    def backup(self,run,path):
        with closing(dbopen(run/'work.sqlite3')) as source,closing(sqlite3.connect(path)) as target:source.backup(target)
    def command(self,operation,path):return [sys.executable,str(ROOT/'tools/natural-study.py'),operation,'--run',str(path),'--fixture','--families','8','--followups','2']
    def test_01_baseline_fixed_model_and_contract(self):
        baseline=load(ROOT/'build/natural-study-baseline.json')
        self.assertEqual(baseline['inputs_count'],2060);self.assertEqual(len(baseline['inputs']),2060)
        self.assertEqual(sha(MODEL),MODEL_SHA)
        for path,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['baseline']=dict(prior_inputs=2060,old_accepted_inputs=2058,all_hashes_current=True,fixed_model_sha256=MODEL_SHA)
        from congklak import natural_study as p
        s=p.CONTRACT;self.assertEqual((s['families'],s['seed'],s['max_choices'],s['random_opening_choices'],s['epsilon']),(512,20261021,192,8,.25));self.assertEqual(s['fresh_band'],'50-74');self.assertEqual(s['fresh_caps'],{'training_candidate':24,'validation':8});self.assertEqual(s['followups'],15);self.assertTrue(s['one_fresh_root_per_family'])
        self.assertEqual(s['h8'],{'depth':8,'node_budget':2000000});self.assertEqual(s['h10'],{'depth':10,'node_budget':5000000});self.assertEqual((s['time_ms'],s['tt_mb'],s['memory_mb'],s['use_pvs']),(0,16,512,True));self.assertFalse(s['training_performed']);self.assertFalse(s['automatic_expansion']);self.assertFalse(s['sealed_final_test_consumed']);p.admit()

    def test_02_protection_and_exact_original_followup_references(self):
        from congklak import natural_study as p
        protected,families,followups,labels=old_reference();registry=load(p.REGISTRY)
        self.assertEqual(set(registry['quotients']),protected);self.assertEqual(set(registry['families']),families);self.assertEqual(registry['followups'],{k:dict(root=r,h8=labels[k]) for k,r in followups.items()});self.assertEqual(Counter(r['cohort'] for r in followups.values()),{'training_candidate':12,'validation':2,'diagnostic':1})
        for key,row in followups.items():self.assertEqual(key,key_of(row['state']));self.assertEqual(mechanism(row['state'])[:2],(row['active'],'50-74/ordinary'))
        self.assertFalse(registry['sealed_target_bodies_read']);self.assertTrue(registry['no_prefix_disjointness_claim'])
        EVIDENCE['protection_and_followups']=dict(old_pilot_roots=62,old_pilot_families=256,protected_quotients=len(protected),protected_families=len(families),exact_old_H8_references=15,cohorts=dict(Counter(r['cohort'] for r in followups.values())),old_H8_hashes_not_rewritten=True)

    def test_03_literal_fixture_histories_ordinary_scope_and_collision_closure(self):
        from congklak import natural_study as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        settings=p.settings_for(True,8,2);native=Native();pool={}
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index in range(8):
                expected=literal_family(index,settings,evaluator);actual=p.prior.generate_family(index,settings,native,evaluator);self.assertEqual(actual,expected);pool[index]=expected
        registry=load(p.REGISTRY);expected=select_reference(pool,settings,registry);self.assertEqual(p.select(pool,settings,registry),expected)
        selected,inherited,evidence=expected;fresh=[r for r in selected.values() if r['study_role']=='fresh'];self.assertEqual(len(fresh),1);self.assertEqual(len(inherited),2)
        r=copy.deepcopy(fresh[0]['root']);other=copy.deepcopy(r);other['cohort']='validation' if r['cohort']=='training_candidate' else 'training_candidate';other['family']='new-cross-owner-family';other['provenance']['family_index']=999
        x,h,d=p.select({0:{'candidates':[r]},1:{'candidates':[other]}},settings,{'quotients':[],'families':[],'followups':{}});self.assertFalse(x);self.assertFalse(h);self.assertEqual(d['cross_cohort_collisions'],1)
        EVIDENCE['literal_fixture']=dict(families=8,all_literal_rng_and_policy_choices_match=True,fresh_H8_targets=1,inherited_H8_targets=2,H10_targets=3,cross_owner_collision_excluded_both=True,ordinary_scope_unchanged=True)

    def test_04_real_collection_pre_postcommit_kills(self):
        parent=self.fresh('collection');fresh=parent/'fresh';run_child(self.command('collect',fresh),parent/'fresh.log');expected=rows(fresh/'work.sqlite3','families','idx');rootexpected=rows(fresh/'work.sqlite3','roots','state');records=[]
        for event,prefix in (('natural_study_family_precommit',1),('natural_study_family_committed',2)):
            run=parent/event;probe=parent/(event+'-probe');marker=kill_barrier(self.command('collect',run),probe,event,env={'NATURAL_STUDY_BLOCK_EVENT':event,'NATURAL_STUDY_BLOCK_STEP':'1','NATURAL_STUDY_MARKER':str(probe/'declared-marker.json')},condition=lambda v:v['index']==1)
            before=rows(run/'work.sqlite3','families','idx');self.assertEqual(list(before),list(range(prefix)));self.backup(run,probe/'validated-prefix.sqlite3');run_child(self.command('collect',run),parent/(event+'-resume.log'));after=rows(run/'work.sqlite3','families','idx');self.assertEqual(after,expected);self.assertEqual({k:after[k] for k in before},before);self.assertEqual(rows(run/'work.sqlite3','roots','state'),rootexpected)
            stable=snapshot(run);run_child(self.command('collect',run),parent/(event+'-rerun.log'));self.assertEqual(snapshot(run),stable);records.append(dict(event=event,committed_prefix=prefix,families_after_resume=8,fresh_histories_and_selection_equal=True,prefix_retained=True,bytes_mtimes_unchanged=True,marker=marker))
        EVIDENCE['live_collection_recovery']=dict(actual_kills=True,tests=records,raw_backups_retained=True)

    def test_05_real_H8_and_H10_pre_postcommit_kills(self):
        from congklak import natural_study as p
        parent=self.fresh('teachers');fresh=parent/'fresh';run_child(self.command('all',fresh),parent/'fresh.log');expected=p.load_rows(fresh);records=[]
        for stage,event,step,prefix in (('h8','natural_study_h8_precommit',0,2),('h8','natural_study_h8_committed',1,3),('h10','natural_study_h10_precommit',1,1),('h10','natural_study_h10_committed',2,2)):
            run=parent/event;probe=parent/(event+'-probe');marker=kill_barrier(self.command('all',run),probe,event,env={'NATURAL_STUDY_BLOCK_EVENT':event,'NATURAL_STUDY_BLOCK_STEP':str(step),'NATURAL_STUDY_MARKER':str(probe/'declared-marker.json')},condition=lambda v,s=step:v['completed']==s)
            before=rows(run/'work.sqlite3',stage+'_attempts','state');beforelabels=rows(run/'work.sqlite3',stage+'_labels','state');self.assertEqual(len(before),prefix);self.backup(run,probe/'validated-prefix.sqlite3');run_child(self.command('all',run),parent/(event+'-resume.log'));actual=p.load_rows(run)
            for left,right in zip(actual[:5],expected[:5]):self.assertEqual(semantic(left),semantic(right))
            after=rows(run/'work.sqlite3',stage+'_attempts','state');afterlabels=rows(run/'work.sqlite3',stage+'_labels','state');self.assertEqual({k:after[k] for k in before},before);self.assertEqual({k:afterlabels[k] for k in beforelabels},beforelabels)
            stable=snapshot(run)
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed fixture executed teacher')):p.run(run,fixture=True,families=8,followups=2)
            self.assertEqual(snapshot(run),stable);records.append(dict(stage=stage,event=event,committed_attempt_rows=prefix,committed_new_prefix=step,inherited_H8_rows=2 if stage=='h8' else 0,final_H8_labels=len(actual[1]),final_H10_labels=len(actual[2]),fresh_semantics_and_counters_equal=True,prefix_exactly_retained=True,completed_zero_teacher_calls=True,bytes_mtimes_unchanged=True,marker=marker))
        EVIDENCE['live_teacher_recovery']=dict(actual_kills=True,tests=records,H8_and_H10_both_barriers=True,raw_backups_retained=True)

    def test_06_partial_identity_ownership_publication_and_pair_denominator_refusals(self):
        from congklak import natural_study as p,reports
        from quality_audit_acceptance import module
        parent=self.fresh('refusals');fresh=parent/'fresh';run_child(self.command('all',fresh),parent/'fresh.log');r,h8,h10,a8,a10,m=p.load_rows(fresh);settings=p.settings_for(True,8,2);registry=load(p.REGISTRY);native=Native();key=next(iter(h8));row=r[key]
        budget=h8[key]['report']['nodes']+1
        partial=native.analyze(flatten(row['state']),actor=row['state']['controller'],total=98,max_depth=10,time_ms=0,node_budget=budget,memory_mb=512,tt_mb=16,use_pvs=True,keep_history=False)
        self.assertEqual(partial['stop_reason'],'node_budget');self.assertIn(partial['report']['choice_depth'],(8,9));save(parent/'actual-H10-partial-after-H8.json',partial)
        small=copy.deepcopy(settings);small['h10']['node_budget']=budget;attempt=p.attempt_for(None,row,'h10',partial);p.validate_attempts({key:row},{key:attempt},{},'h10',small,registry,native)
        forged=copy.deepcopy(attempt);forged['completed_depth']=10
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:forged},{},'h10',small,registry,native)
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:attempt},{key:h8[key]},'h10',small,registry,native)
        item=copy.deepcopy(h8[key]);item['report']['search_options'].update(tt_mb=0,tt_allocated_bytes=0,tt_capacity=0,use_pvs=False)
        reports.validate(item['report'],row['state'],config.identity(native.build_id),native)
        with self.assertRaises(ValueError):p.validate_label(item,row,8,native)
        item=copy.deepcopy(h8[key]);item['label']['target_player0']+=1
        with self.assertRaises((ValueError,p.teacher.DeeperError)):p.validate_label(item,row,8,native)
        item=copy.deepcopy(h10[key]);item['report']['completed_iteration']=False
        with self.assertRaises((ValueError,p.teacher.DeeperError)):p.validate_label(item,row,10,native)
        forged=copy.deepcopy(a10[key]);forged['work']={k:0 for k in forged['work']}
        with self.assertRaises(ValueError):p.validate_attempts({key:row},{key:forged},{key:h10[key]},'h10',settings,registry,native)
        for kind in ('identity_schema','ownership','H8_reference','complete_missing_export'):
            run=parent/kind;shutil.copytree(fresh,run)
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                if kind=='identity_schema':
                    ident=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0]);ident['schema']='incompatible-study';body=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0]);body['identity']=ident
                    for name,value in (('identity',ident),('complete',body)):db.execute('UPDATE meta SET body=?,hash=? WHERE key=?',(config.canonical(value),config.digest(value),name))
                    (run/'manifest.json').write_text(config.canonical(body)+'\n',encoding='utf-8')
                elif kind=='ownership':
                    key0=next(k for k,v in r.items() if v['study_role']=='followup');body=copy.deepcopy(r[key0]);body['study_role']='fresh';db.execute('UPDATE roots SET body=?,hash=? WHERE state=?',(config.canonical(body),config.digest(body),key0))
                elif kind=='H8_reference':
                    key0=next(k for k,v in r.items() if v['study_role']=='followup');body=copy.deepcopy(h8[key0]);body['report']['elapsed_ms']+=1;body['label']['work']['elapsed_ms']+=1;db.execute('UPDATE h8_labels SET body=?,hash=? WHERE state=?',(config.canonical(body),config.digest(body),key0))
                else:
                    body=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0]);body['selected_fresh']+=1;db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(body),config.digest(body)));(run/'manifest.json').unlink()
                db.commit()
            with self.assertRaises((ValueError,KeyError,p.teacher.DeeperError)):p.load_rows(run)
            self.assertNotEqual(run_child(self.command('all',run),parent/(kind+'.log'),success=False),0)
            if kind=='complete_missing_export':self.assertFalse((run/'manifest.json').exists(),'Invalid complete marker was republished')
        tool=module('natural_stability_pair_fixture','tools/analyze-natural-study.py');paired={k:float(i+1) for i,k in enumerate(MEANS)};paired.update(complete_pair=True,h8_complete=True,h10_complete=True,h8_status='complete',h10_status='complete',legal_action_count=3,family='a',best_sets_changed=True,best_sets_disjoint=False,H8_choice_still_best=True,all_H8_ties_still_best=False)
        unpaired={**paired,**{k:None for k in MEANS},'complete_pair':False,'h8_complete':False,'h8_status':'budget_rejected','family':'b'}
        summary=tool.summarize([paired,unpaired]);check_summary(self,summary,[paired,unpaired]);self.assertEqual(summary['complete_pairs'],1);self.assertEqual(summary['unpaired_roots'],1);self.assertEqual(summary['paired_means'],{k:paired[k] for k in MEANS})
        EVIDENCE['refusals_and_censoring']=dict(real_H10_partial_depth=partial['report']['choice_depth'],H8_or_H9_partial_allowed_only_as_rejection=True,partial_as_full_H10_refused=True,wrong_TT_PVS_refused=True,report_target_link_refused=True,incomplete_report_refused=True,zero_work_refused=True,rehashed_identity_schema_ownership_H8_reference_refused=True,corrupt_complete_not_published=True,synthetic_unpaired_summary_uses_only_complete_intersection=True)

    def test_07_all512_literal_histories_selection_and_unchanged_followups(self):
        from congklak import natural_study as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        roots,h8,h10,a8,a10,manifest=p.load_rows();saved=rows(p.DEFAULT/'work.sqlite3','families','idx');self.assertEqual(list(saved),list(range(512)));expected={}
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index in range(512):expected[index]=literal_family(index,p.CONTRACT,evaluator);self.assertEqual(saved[index],expected[index],f'Family {index}')
        registry=load(p.REGISTRY);selected,inherited,evidence=select_reference(expected,p.CONTRACT,registry);self.assertEqual(roots,selected);self.assertEqual(manifest['selection'],evidence);self.assertEqual(manifest,load(p.OUTPUT));self.assertEqual(manifest,load(p.DEFAULT/'manifest.json'))
        protected,oldfamilies,followups,oldh8=old_reference();self.assertEqual({k:r['root'] for k,r in roots.items() if r['study_role']=='followup'},followups);self.assertEqual(inherited,oldh8)
        for key in inherited:self.assertEqual(h8[key],oldh8[key]);self.assertEqual(a8[key]['status'],'inherited_complete')
        fresh=[r for r in roots.values() if r['study_role']=='fresh'];self.assertLessEqual(len(fresh),32);self.assertEqual(len({r['family'] for r in fresh}),len(fresh));owners=defaultdict(set)
        for r in fresh:
            self.assertNotIn(r['quotient'],protected);self.assertNotIn(r['family'],oldfamilies);self.assertEqual(mechanism(r['state'])[:2],(r['active'],'50-74/ordinary'));owners[r['cohort']].add(r['quotient'])
        self.assertFalse(owners['training_candidate']&owners['validation']);self.assertEqual(set(a8),set(roots));self.assertEqual(set(a10),set(roots));self.assertEqual(manifest['selected_fresh'],len(fresh));self.assertEqual(manifest['followups'],15)
        EVIDENCE['production_collection']=dict(families=512,literal_choices=sum(f['choices'] for f in expected.values()),policy_evaluations=sum(f['policy_evaluations'] for f in expected.values()),source_families=dict(Counter(f['source'] for f in expected.values())),cohort_families=dict(Counter(f['cohort'] for f in expected.values())),all_literal_rng_and_policy_choices_match=True,full_pool_closure_and_selection_match=True,selection=evidence,fresh_selected=len(fresh),unchanged_old_followups=15,exact_old_H8_items_reused=True,all_H10_targets_attempted_independent_of_H8_status=True)

    def test_08_all_teacher_PVs_paired_metrics_and_completed_stability(self):
        from congklak import natural_study as p,reports
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        roots,h8,h10,a8,a10,manifest=p.load_rows();native=Native();pvs=choices=0
        for depth,labels in ((8,h8),(10,h10)):
            for key,item in labels.items():
                reports.validate(item['report'],roots[key]['state'],config.identity(native.build_id),native)
                paths,steps=check_pvs(self,roots[key],item,depth);pvs+=paths;choices+=steps
        report=load(ROOT/'build/natural-study-stability.json');actual={r['state_key']:r for r in report['metrics']};self.assertEqual(len(actual),len(report['metrics']));self.assertEqual(set(actual),set(roots))
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for key,row in roots.items():
                value,_,logits=value_and_policy(norm(row['state']),evaluator.model.envelope['model']);prediction=evaluator.evaluate_many([row['state']])[0]
                self.assertAlmostEqual(value,prediction['value_player0'],places=10);self.assertEqual(logits,prediction['policy_logits']);r=actual[key];legal=list(literal.legal_actions(norm(row['state'])));chosen=min(legal,key=lambda a:(-logits[a],a));paired=key in h8 and key in h10
                self.assertEqual(r['complete_pair'],paired);self.assertEqual(r['h8_complete'],key in h8);self.assertEqual(r['h10_complete'],key in h10);self.assertEqual(r['h8_status'],a8[key]['status']);self.assertEqual(r['h10_status'],a10[key]['status']);self.assertEqual(r['legal_action_count'],len(legal));self.assertEqual(r['policy_chosen'],chosen);self.assertAlmostEqual(r['model_value_player0'],value,places=10)
                for field in ('study_role','cohort','family','state'):self.assertEqual(r[field],row[field])
                if not paired:
                    for metric in MEANS+('best_sets_changed','best_sets_disjoint','H8_choice_still_best','all_H8_ties_still_best'):self.assertIsNone(r[metric])
                    continue
                l8,l10=h8[key]['label'],h10[key]['label'];q8={a['pit']:a['value_player0'] for a in l8['action_scores']};q10={a['pit']:a['value_player0'] for a in l10['action_scores']};v8=l8['target_player0'];v10=l10['target_player0'];sign=1 if row['state']['controller']==0 else -1;b8=l8['best_moves'];b10=l10['best_moves']
                err8=abs(value-v8);err10=abs(value-v10);reg8=sign*(v8-q8[chosen]);reg10=sign*(v10-q10[chosen]);ties=[sign*(v10-q10[a]) for a in b8]
                expected=dict(value_mae_h8=err8,value_mae_h10=err10,value_mae_change_h10_minus_h8=err10-err8,policy_regret_h8=reg8,policy_regret_h10=reg10,policy_regret_change_h10_minus_h8=reg10-reg8,absolute_target_shift=abs(v10-v8),signed_actor_target_shift=sign*(v10-v8),H10_regret_H8_chosen=sign*(v10-q10[min(b8)]),H10_regret_H8_tie_min=min(ties),H10_regret_H8_tie_max=max(ties),action_Q_shift_mae=statistics.mean(abs(q10[a]-q8[a]) for a in legal),best_sets_changed=set(b8)!=set(b10),best_sets_disjoint=not bool(set(b8)&set(b10)),H8_choice_still_best=min(b8) in b10,all_H8_ties_still_best=set(b8)<=set(b10),H8_target_player0=v8,H10_target_player0=v10,H8_best_moves=b8,H10_best_moves=b10,H8_action_scores=l8['action_scores'],H10_action_scores=l10['action_scores'])
                for name,v in expected.items():
                    if isinstance(v,float):self.assertAlmostEqual(r[name],v,places=10)
                    else:self.assertEqual(r[name],v)
        check_summary(self,report['overall'],list(actual.values()));self.assertEqual(set(report['groups']),{r['study_role']+'/'+r['cohort'] for r in actual.values()})
        for group,summary in report['groups'].items():check_summary(self,summary,[r for r in actual.values() if r['study_role']+'/'+r['cohort']==group])
        self.assertEqual(report['counts'],dict(selected_roots=len(roots),h8_complete=len(h8),h10_complete=len(h10),complete_pairs=len(set(h8)&set(h10)),unpaired_roots=len(roots)-len(set(h8)&set(h10)),h8_rejected=sum(a['status']=='budget_rejected' for a in a8.values()),h10_rejected=sum(a['status']=='budget_rejected' for a in a10.values()),old_h8_references=sum(r['study_role']=='followup' for r in roots.values()),new_h8_labels=sum(roots[k]['study_role']=='fresh' for k in h8),new_h10_labels=len(h10)))
        for flag in ('training_performed','model_changed','active_sampling_weights_changed','new_PUCT_or_strength_tests','strength_improvement_claimed','sealed_final_test_consumed','independent_unseen_release_test'):self.assertFalse(report[flag])
        fresh=[]
        for depth,labels in ((8,h8),(10,h10)):
            if not labels:continue
            key=next(iter(labels));state=roots[key]['state'];result=native.analyze(flatten(state),actor=state['controller'],total=98,max_depth=depth,time_ms=0,node_budget=2000000 if depth==8 else 5000000,tt_mb=16,use_pvs=True,memory_mb=512,keep_history=False)
            self.assertEqual(result['stop_reason'],'max_depth');self.assertEqual(semantic(result['report']),semantic(labels[key]['report']));fresh.append(dict(depth=depth,state_key=key))
        before=snapshot(p.DEFAULT);stability_path=ROOT/'build/natural-study-stability.json';stability=(sha(stability_path),stability_path.stat().st_mtime_ns)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed verification invoked teacher search')):p.run()
        self.assertEqual(snapshot(p.DEFAULT),before)
        run_child([sys.executable,str(ROOT/'tools/analyze-natural-study.py')],self.base/'stability-rerun.log');self.assertEqual((sha(stability_path),stability_path.stat().st_mtime_ns),stability)
        for path,digest in load(ROOT/'build/natural-study-baseline.json')['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        EVIDENCE['production_labels_and_metrics']=dict(roots=len(roots),h8_labels=len(h8),h10_labels=len(h10),complete_pairs=len(set(h8)&set(h10)),literal_PV_paths=pvs,literal_PV_choices=choices,literal_model_forwards=len(roots),fresh_teacher_reports=fresh,all_horizon_links_and_ties_verified=True,all_paired_values_regrets_shifts_and_denominators_rederived=True,completed_zero_teacher_calls=True,completed_bytes_mtimes_unchanged=True,stability_bytes_mtimes_unchanged=True,all2060_prior_hashes_current=True)

GROUPS={'preflight':['test_01_baseline_fixed_model_and_contract','test_02_protection_and_exact_original_followup_references','test_03_literal_fixture_histories_ordinary_scope_and_collision_closure','test_04_real_collection_pre_postcommit_kills','test_05_real_H8_and_H10_pre_postcommit_kills','test_06_partial_identity_ownership_publication_and_pair_denominator_refusals'],
    'final':['test_07_all512_literal_histories_selection_and_unchanged_followups','test_08_all_teacher_PVs_paired_metrics_and_completed_stability']}

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=list(GROUPS));a=parser.parse_args();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(NaturalStudyIndependentAudit(n) for n in GROUPS[a.group]))
    files=['tests/natural_study_independent_audit.py','congklak/natural_study.py','tools/natural-study.py','tools/analyze-natural-study.py','build/natural-study-baseline.json','build/natural-study-registry.json','build/natural-study-plan.json','build/natural-study-implementation.json','build/natural-study-audit-fixture-feasibility.json']
    paths=[ROOT/f for f in files if (ROOT/f).is_file()];paths += [p for p in NaturalStudyIndependentAudit.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if a.group=='final':
        paths += [ROOT/f for f in ('data/natural-study-v1/work.sqlite3','data/natural-study-v1/manifest.json','build/natural-study-result.json','build/natural-study-stability.json','build/natural-study-audit/preflight.json')]
        paths += [ROOT/f for f in load(WORK/'preflight.json')['inputs']]
    body=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=GROUPS[a.group],evidence=EVIDENCE,inputs=pins(paths),duration_seconds=time.monotonic()-started,retained_directory=relative(NaturalStudyIndependentAudit.base),failure_details=[dict(method=str(t),trace=trace) for t,trace in result.failures+result.errors])
    save(WORK/(a.group+'.json'),body);print(config.canonical(dict(event='natural_study_independent_audit',group=a.group,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

