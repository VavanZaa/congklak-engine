"""Independent five-root H14 completion, tier eligibility and live recovery audit."""
from __future__ import annotations
from collections import Counter
from contextlib import closing
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import unittest
from unittest import mock
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,key_of,rows,semantic,kill_barrier,run_child,snapshot
from natural_study_independent_audit import check_pvs
from horizon_probe_independent_audit import q_semantic,item_view,check_histories,check_forensic_line
WORK=ROOT/'build/horizon-finish-audit';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def old_reference():
    path=ROOT/'data/horizon-probe-v1/work.sqlite3'
    names={'h8_references':'h8_references','h10_references':'h10_references','h12_references':'h12_labels',
           'prior_h12_attempts':'h12_attempts','prior_h14_attempts':'h14_attempts','prior_h14_labels':'h14_labels'}
    root=rows(path,'roots','state');stores={name:rows(path,table,'state') for name,table in names.items()}
    keys=sorted(k for k,a in stores['prior_h14_attempts'].items() if a['status']=='budget_rejected')
    assert len(root)==6 and len(keys)==5 and len(stores['prior_h14_labels'])==1
    return keys,root,stores


def audit_new_histories(test,root,stores,settings,targets,literal_pvs=True):
    from congklak import reports
    known={key:{} for key in root};native=Native();history_reports=paths=steps=0;nodes=0
    for name in ('prior_h12_attempts','prior_h14_attempts'):
        for key,a in stores[name].items():
            for report in a['history']:known[key][report['choice_depth']]=q_semantic(report)
    for tier in ('tier1','tier2'):
        attempts=stores[tier+'_attempts'];labels=stores[tier+'_labels']
        eligible=targets if tier=='tier1' else [k for k in targets if stores['tier1_attempts'][k]['status']=='budget_rejected']
        test.assertEqual(list(attempts),eligible);test.assertLessEqual(set(labels),set(attempts))
        cap=settings[tier]['node_budget']
        for key,a in attempts.items():
            row=root[key];h=a['history'];nodes+=a['work']['nodes']
            test.assertEqual(a['tier'],tier);test.assertEqual(a['schema'],'h14-budget-tier-attempt-v1');test.assertEqual(a['node_budget'],cap)
            test.assertEqual((a['requested_depth'],a['stage']),(14,'h14'));test.assertEqual(a['study_row_sha256'],config.digest(row));test.assertEqual(a['source_root_sha256'],row['source_root_sha256'])
            test.assertEqual([r['choice_depth'] for r in h],list(range(1,len(h)+1)));test.assertEqual(a['completed_depth'],len(h));test.assertEqual(a['native_last_completed_depth'],len(h) if h else -1)
            test.assertGreater(a['work']['nodes'],0);test.assertLessEqual(a['work']['nodes'],cap)
            if a['status']=='complete':
                test.assertEqual(len(h),14);test.assertEqual(a['stop_reason'],'max_depth');test.assertEqual(labels[key],item_view(h[-1],row));test.assertEqual(a['label_sha256'],config.digest(labels[key]))
            else:
                test.assertEqual(a['status'],'budget_rejected');test.assertEqual(a['stop_reason'],'node_budget');test.assertLess(len(h),14);test.assertEqual(a['work']['nodes'],cap);test.assertNotIn(key,labels);test.assertIsNone(a['label_sha256'])
            previous={name:0 for name in a['work']}
            for report in h:
                reports.validate(report,row['state'],config.identity(native.build_id),native)
                test.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True))
                for name,total in a['work'].items():test.assertGreaterEqual(report[name],previous[name]);test.assertLessEqual(report[name],total)
                previous={name:report[name] for name in previous};depth=report['choice_depth'];signature=q_semantic(report)
                if depth in known[key]:test.assertEqual(known[key][depth],signature)
                known[key][depth]=signature;history_reports+=1
                if literal_pvs:
                    n,m=check_pvs(test,row,item_view(report,row),depth);paths+=n;steps+=m
    return dict(new_history_reports=history_reports,literal_new_PV_paths=paths,literal_new_PV_choices=steps,total_new_nodes=nodes,
                tier2_eligible_keys=[k for k in targets if stores['tier1_attempts'][k]['status']=='budget_rejected'])


class HorizonFinishIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p
    def backup(self,run,path):
        with closing(dbopen(run/'work.sqlite3')) as source,closing(sqlite3.connect(path)) as target:source.backup(target)
    def command(self,stage,path):return [sys.executable,str(ROOT/'tools/horizon-finish.py'),stage,'--run',str(path),'--fixture','--roots','1']
    def shared_fixture(self):
        from congklak import horizon_finish as p
        path=self.base/'shared-independent-fresh'
        if not path.exists():run_child(self.command('all',path),self.base/'shared-independent-fresh.log',timeout=1800)
        p.load_rows(path);return path
    def seeded_probe(self,parent,tier,event):
        source=self.shared_fixture();run=parent/event;shutil.copytree(source,run);removed=[]
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            for name in ('tier1_attempts','tier1_labels','tier2_attempts','tier2_labels'):
                if tier=='tier1' or name.startswith('tier2'):
                    removed.append(dict(table=name,keys=[r[0] for r in db.execute('SELECT state FROM '+name+' ORDER BY state')]))
                    db.execute('DELETE FROM '+name)
            db.execute("DELETE FROM meta WHERE key='complete'");db.commit()
        (run/'manifest.json').unlink()
        retained=rows(run/'work.sqlite3','tier1_attempts','state')
        save(parent/(event+'-seed.json'),dict(source=relative(source),source_journal_sha256=sha(source/'work.sqlite3'),
            retained_initial_keys=list(retained),retained_initial_digest=config.digest(retained),removed=removed,completion_and_export_removed=True))
        return run

    def test_01_baseline_fixed_model_and_two_distinct_budget_contracts(self):
        from congklak import horizon_finish as p
        baseline=load(p.BASELINE);self.assertEqual((baseline['inputs_count'],baseline['prior_accepted_inputs']),(2314,2312));self.assertEqual(len(baseline['inputs']),2314)
        for name,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA);c=p.CONTRACT
        self.assertEqual(c['roots'],5);self.assertEqual(c['tier1'],dict(depth=14,node_budget=250000000));self.assertEqual(c['tier2'],dict(depth=14,node_budget=1000000000));self.assertEqual(c['maximum_new_production_nodes'],6250000000)
        self.assertEqual((c['keep_history'],c['time_ms'],c['tt_mb'],c['use_pvs'],c['memory_mb']),(True,0,16,True,512));self.assertTrue(c['investigation_only']);self.assertFalse(c['automatic_expansion']);self.assertFalse(c['training_performed']);self.assertFalse(c['sealed_final_test_consumed'])
        fixture=p.settings_for(True,1);self.assertEqual(fixture['tier1'],dict(depth=14,node_budget=25000000));self.assertEqual(fixture['tier2'],dict(depth=14,node_budget=250000000));self.assertEqual(fixture['roots'],1)
        with self.assertRaises(ValueError):p.settings_for(False,1)
        p.admit();EVIDENCE['baseline']=dict(prior_inputs=2314,old_accepted_inputs=2312,all_hashes_current=True,fixed_model_sha256=MODEL_SHA,maximum_production_nodes=6250000000,fixture_caps_separate_from_production=True)

    def test_02_exact_five_selection_six_original_references_and_ownership(self):
        from congklak import horizon_finish as p
        targets,root,prior=old_reference();registry=load(p.REGISTRY)
        self.assertEqual(registry['state_keys'],targets);self.assertEqual(registry['comparison_keys'],list(root));self.assertEqual(registry['rows'],root)
        for name,value in prior.items():self.assertEqual(registry[name],value)
        self.assertTrue(registry['investigation_only']);self.assertTrue(registry['no_ingest_or_training'])
        for key,row in root.items():self.assertEqual(key,key_of(row['state']));self.assertEqual(row['source_root_sha256'],config.digest(row['root']))
        EVIDENCE['exact_selection']=dict(target_roots=5,comparison_roots=6,state_keys=targets,comparison_keys=list(root),original_rows_and_all_six_prior_tables_exact=True,
            target_cohorts=dict(Counter(root[k]['cohort'] for k in targets)),target_roles=dict(Counter(root[k]['study_role'] for k in targets)),old_completed_H14_key=next(iter(prior['prior_h14_labels'])),ownership_unchanged=True,investigation_only=True)

    def test_03_actual_fixture_rejection_fallback_completion_and_literal_histories(self):
        from congklak import horizon_finish as p
        run=self.shared_fixture();root,stores,m=p.load_rows(run);targets=load(p.REGISTRY)['state_keys'][:1]
        proof=audit_new_histories(self,root,stores,p.settings_for(True,1),targets)
        self.assertEqual(len(stores['tier1_attempts']),1);self.assertFalse(stores['tier1_labels']);self.assertEqual(stores['tier1_attempts'][targets[0]]['completed_depth'],13)
        self.assertEqual(len(stores['tier2_labels']),1);self.assertTrue(m['scope_finished']);self.assertEqual(m['status'],'complete');self.assertEqual(m['total_new_nodes'],proof['total_new_nodes'])
        EVIDENCE['fresh_fixture']=dict(**proof,target_roots=1,requested_depth=14,tier1_rejected=1,tier1_auxiliary_depth=13,tier2_complete=1,initial_cap=25000000,fallback_cap=250000000,cost_scope='Independent explicitly versioned fixture; separate from production6.25b allocation')

    def recovery(self,tier):
        from congklak import horizon_finish as p
        parent=self.fresh(tier+'-kills');source=self.shared_fixture();root,fresh,_=p.load_rows(source);records=[]
        for suffix,prefix in (('precommit',0),('committed',1)):
            event='horizon_finish_'+tier+'_'+suffix;run=self.seeded_probe(parent,tier,event);probe=parent/(event+'-probe')
            initial_before=rows(run/'work.sqlite3','tier1_attempts','state')
            marker=kill_barrier(self.command(tier,run),probe,event,env={'HORIZON_FINISH_BLOCK_EVENT':event,'HORIZON_FINISH_BLOCK_STEP':str(prefix),'HORIZON_FINISH_MARKER':str(probe/'declared-marker.json')},condition=lambda value,n=prefix:value['completed']==n,timeout=1800)
            before=rows(run/'work.sqlite3',tier+'_attempts','state');beforelabels=rows(run/'work.sqlite3',tier+'_labels','state');self.assertEqual(len(before),prefix);self.backup(run,probe/'validated-prefix.sqlite3')
            run_child(self.command(tier,run),parent/(event+'-resume.log'),timeout=1800)
            actual=rows(run/'work.sqlite3',tier+'_attempts','state');labels=rows(run/'work.sqlite3',tier+'_labels','state')
            self.assertEqual(semantic(actual),semantic(fresh[tier+'_attempts']));self.assertEqual(semantic(labels),semantic(fresh[tier+'_labels']));self.assertEqual({k:actual[k] for k in before},before);self.assertEqual({k:labels[k] for k in beforelabels},beforelabels)
            if tier=='tier2':self.assertEqual(rows(run/'work.sqlite3','tier1_attempts','state'),initial_before)
            stable=snapshot(run)
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed tier reran teacher')):p.run(run,stage=tier,fixture=True,roots=1)
            self.assertEqual(snapshot(run),stable)
            records.append(dict(event=event,tier=tier,committed_prefix=prefix,targets_after_resume=1,prefix_seeded_from_actual_independent_fresh_fixture=True,
                seed_record=relative(parent/(event+'-seed.json')),initial_tier_retained_exactly=tier=='tier2',prefix_exactly_retained=True,fresh_semantics_and_counters_equal=True,completed_zero_teacher_calls=True,bytes_mtimes_unchanged=True,marker=marker))
        return dict(actual_kills=True,tier=tier,tests=records,raw_seed_edits_markers_backups_retained=True)

    def test_04_actual_initial_tier_pre_postcommit_recovery(self):EVIDENCE['live_tier1_recovery']=self.recovery('tier1')
    def test_05_actual_fallback_tier_pre_postcommit_recovery(self):EVIDENCE['live_tier2_recovery']=self.recovery('tier2')

    def test_06_identity_budget_fallback_ownership_history_and_publication_refusals(self):
        from congklak import horizon_finish as p,reports
        parent=self.fresh('refusals');source=self.shared_fixture();root,stores,m=p.load_rows(source);settings=p.settings_for(True,1);native=Native();key=next(iter(stores['tier1_attempts']));row=root[key];a=stores['tier2_attempts'][key];label=stores['tier2_labels'][key]
        for change in ('history_gap','zero_work','wrong_budget','wrong_TT_PVS','auxiliary_as_H14'):
            bad=copy.deepcopy(a);item=label
            if change=='history_gap':bad['history'].pop(0);bad['completed_depth']-=1;bad['native_last_completed_depth']-=1
            elif change=='zero_work':bad['work']={name:0 for name in bad['work']}
            elif change=='wrong_budget':bad['node_budget']=1000000000
            elif change=='wrong_TT_PVS':
                bad['history'][0]['search_options'].update(tt_mb=0,tt_allocated_bytes=0,tt_capacity=0,use_pvs=False)
                reports.validate(bad['history'][0],row['state'],config.identity(native.build_id),native)
            else:
                bad=copy.deepcopy(stores['tier1_attempts'][key]);bad.update(tier='tier2',node_budget=250000000,status='complete',stop_reason='max_depth');item=p.item_for(bad['history'][-1],row);bad['label_sha256']=config.digest(item)
            with self.assertRaises(ValueError,msg=change):p.validate_attempt(row,bad,item,'tier2',settings,native)
        # A successful initial attempt cannot be eligible for fallback, even with fully hashed records.
        synthetic=copy.deepcopy(stores);successful=copy.deepcopy(a);successful.update(tier='tier1',node_budget=250000000);synthetic['tier1_attempts']={key:successful};synthetic['tier1_labels']={key:label}
        with self.assertRaises(ValueError):p.validate_tables(root,synthetic,p.CONTRACT,load(p.REGISTRY),native)
        for kind in ('identity','fixture_production_caps','ownership','reference','history','prefix','fallback_eligibility','complete_missing_export'):
            run=parent/kind;shutil.copytree(source,run)
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                def update(table,k,value,column='state'):db.execute('UPDATE '+table+' SET body=?,hash=? WHERE '+column+'=?',(config.canonical(value),config.digest(value),k))
                if kind in ('identity','fixture_production_caps'):
                    ident=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0]);marker=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0])
                    if kind=='identity':ident['schema']='incompatible-finish'
                    else:ident['fixture']=False;ident['settings']=copy.deepcopy(p.CONTRACT)
                    marker['identity']=ident;update('meta','identity',ident,'key');update('meta','complete',marker,'key');(run/'manifest.json').write_text(config.canonical(marker)+'\n',encoding='utf-8')
                elif kind=='ownership':bad=copy.deepcopy(row);bad['cohort']='diagnostic';update('roots',key,bad)
                elif kind=='reference':bad=copy.deepcopy(stores['h8_references'][key]);bad['label']['target_player0']+=1;update('h8_references',key,bad)
                elif kind=='history':bad=copy.deepcopy(a);bad['history'][0]['depth_result']['value_player0']+=1;update('tier2_attempts',key,bad)
                elif kind=='prefix':db.execute('DELETE FROM tier1_attempts')
                elif kind=='fallback_eligibility':bad=copy.deepcopy(a);bad['tier']='tier1';bad['node_budget']=25000000;update('tier1_attempts',key,bad)
                else:
                    marker=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0]);marker['auxiliary_completed_reports']+=1;update('meta','complete',marker,'key');(run/'manifest.json').unlink()
                db.commit()
            with self.assertRaises((ValueError,KeyError,p.teacher.DeeperError)):p.load_rows(run)
            self.assertNotEqual(run_child(self.command('all',run),parent/(kind+'.log'),success=False,timeout=1800),0)
            if kind=='complete_missing_export':self.assertFalse((run/'manifest.json').exists())
        repaired=parent/'repair';shutil.copytree(source,repaired);before=snapshot(repaired);export=(repaired/'manifest.json').read_bytes();(repaired/'manifest.json').unlink()
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Repair ran teacher')):p.run(repaired,fixture=True,roots=1)
        self.assertEqual((repaired/'manifest.json').read_bytes(),export);after=snapshot(repaired);self.assertEqual({k:v for k,v in before.items() if k!='manifest.json'},{k:v for k,v in after.items() if k!='manifest.json'})
        EVIDENCE['refusals_and_eligibility']=dict(history_gap_zero_work_wrong_budget_wrong_TT_PVS_refused=True,auxiliary_H13_not_H14=True,successful_initial_root_cannot_fallback=True,
            rehashed_identity_fixture_caps_ownership_reference_history_prefix_fallback_refused=True,invalid_complete_not_published=True,missing_export_repaired_without_teacher=True,repair_preserves_journal_bytes_mtime=True)

    def test_07_all_production_histories_PVs_ledger_and_fresh_H14_verifier(self):
        from congklak import horizon_finish as p,horizon_probe as old
        root,stores,m=p.load_rows();targets,priorroot,prior=old_reference();self.assertEqual(root,priorroot)
        for name,value in prior.items():self.assertEqual(stores[name],value)
        historical=check_histories(self,root,stores['h8_references'],stores['h10_references'],stores['h12_references'],stores['prior_h14_labels'],stores['prior_h12_attempts'],stores['prior_h14_attempts'],old.CONTRACT)
        proof=audit_new_histories(self,root,stores,p.CONTRACT,targets);combined={**stores['prior_h14_labels'],**stores['tier1_labels'],**stores['tier2_labels']};unresolved=sorted(set(targets)-set(combined))
        self.assertEqual(p.combined_labels(stores),combined);self.assertFalse(set(stores['tier1_labels'])&set(stores['tier2_labels']));self.assertLessEqual(proof['total_new_nodes'],6250000000)
        expected=dict(comparison_roots=6,target_roots=5,h8_references=6,h10_references=6,h12_references=6,prior_h14_labels=1,
            tier1_attempts=len(stores['tier1_attempts']),tier1_labels=len(stores['tier1_labels']),tier1_rejected=sum(a['status']=='budget_rejected' for a in stores['tier1_attempts'].values()),
            tier2_eligible=len(proof['tier2_eligible_keys']),tier2_attempts=len(stores['tier2_attempts']),tier2_labels=len(stores['tier2_labels']),tier2_rejected=sum(a['status']=='budget_rejected' for a in stores['tier2_attempts'].values()),
            new_h14_labels=len(set(targets)&set(combined)),combined_h14_labels=len(combined),unresolved_target_roots=len(unresolved),auxiliary_completed_reports=proof['new_history_reports'],total_new_nodes=proof['total_new_nodes'])
        for name,value in expected.items():self.assertEqual(m[name],value)
        self.assertEqual(m['unresolved_keys'],unresolved);self.assertEqual(m['scope_finished'],not unresolved);self.assertEqual(m['status'],'complete' if not unresolved else 'budget_exhausted');self.assertEqual(m,load(p.OUTPUT));self.assertEqual(m,load(p.DEFAULT/'manifest.json'))
        key=targets[0];winner='tier2' if key in stores['tier2_labels'] else 'tier1';saved=stores[winner+'_attempts'][key];state=root[key]['state'];native=Native()
        fresh=native.analyze(flatten(state),actor=state['controller'],total=98,max_depth=14,time_ms=0,node_budget=saved['node_budget'],memory_mb=512,tt_mb=16,use_pvs=True,keep_history=True)
        save(self.base/'fresh-first-target-H14-bounded.json',fresh);self.assertEqual(fresh['stop_reason'],saved['stop_reason']);self.assertEqual([q_semantic(r) for r in fresh['reports']],[q_semantic(r) for r in saved['history']]);self.assertLessEqual(fresh['nodes'],saved['node_budget'])
        EVIDENCE['production_teacher_audit']=dict(**expected,**{k:v for k,v in proof.items() if k not in expected},historical_reports_and_PVs=historical,
            all_six_original_rows_and_prior_tables_exact=True,all_five_target_initial_attempts_present=True,all_fallback_eligibility_and_caps_verified=True,all_overlap_Q_best_semantics_match=True,
            fresh_bounded_check=dict(state_key=key,requested_depth=14,node_cap=saved['node_budget'],actual_nodes=fresh['nodes'],last_completed_depth=fresh['last_completed_depth'],all_history_Q_best_semantics_equal=True),fresh_check_cost_separate_from_production=True)

    def test_08_independent_combined_sixroot_forensic_math_provenance_and_literal_PVs(self):
        from congklak import horizon_finish as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        root,stores,m=p.load_rows();report=load(ROOT/'build/horizon-finish-comparison.json');cases={c['state_key']:c for c in report['cases']};self.assertEqual(list(cases),list(root));self.assertEqual(len(cases),6)
        h8,h10,h12=(stores[name] for name in ('h8_references','h10_references','h12_references'));h14={**stores['prior_h14_labels'],**stores['tier1_labels'],**stores['tier2_labels']};self.assertEqual(set(h14),set(root));line_count=step_count=captures=bonuses=depth_count=0
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,(key,row) in enumerate(root.items(),1):
                case=cases[key];state=row['state'];value,_,logits=value_and_policy(norm(state),evaluator.model.envelope['model']);pred=evaluator.evaluate_many([state])[0];self.assertAlmostEqual(value,pred['value_player0'],places=10);self.assertEqual(logits,pred['policy_logits']);legal=h8[key]['report']['legal_actions'];policy=min(legal,key=lambda a:(-logits[a],a));sign=1 if state['controller']==0 else -1;b8=h8[key]['label']['best_moves'];b10=h10[key]['label']['best_moves'];b12=h12[key]['label']['best_moves'];b14=h14[key]['label']['best_moves']
                if key in stores['prior_h14_labels']:job='prior_h14';a14=stores['prior_h14_attempts'][key];budget=25000000;provenance=dict(source_kind='prior_complete_H14',source_job=job,node_budget=budget,new_teacher_work=False)
                else:
                    job='tier2' if key in stores['tier2_labels'] else 'tier1';a14=stores[job+'_attempts'][key];budget=a14['node_budget'];provenance=dict(source_kind='larger_budget_complete_H14',source_job=job,node_budget=budget,new_teacher_work=True)
                self.assertEqual(case['H14_provenance'],provenance);self.assertEqual(case['prior_requested_H14_status'],stores['prior_h14_attempts'][key]['status']);self.assertEqual(case['prior_auxiliary_H13_best_moves'],stores['prior_h14_attempts'][key]['history'][12]['best_moves'])
                evidence={8:(h8[key]['report'],'reference','old_h8',8,None),10:(h10[key]['report'],'reference','old_h10',10,None),12:(h12[key]['report'],'requested_label','h12',12,10000000),14:(h14[key]['report'],'requested_label',job,14,budget)}
                for stage,attempt,requested,cap in ((job,a14,14,budget),('h12',stores['prior_h12_attempts'][key],12,10000000)):
                    for r in attempt['history']:
                        d=r['choice_depth']
                        if d>=8 and d not in evidence:evidence[d]=(r,'auxiliary_completed_iteration',stage,requested,cap)
                depths=sorted(evidence);equal=set(b12)==set(b14)
                expected_case=dict(case_id='D'+str(index),state=state,family=row['family'],source_role=row['study_role'],cohort=row['cohort'],investigation_only=True,requested_status=dict(h12='complete',h14='complete'),requested_depths_available=[12,14],auxiliary_completed_depths=[d for d in depths if evidence[d][1]=='auxiliary_completed_iteration'],requested_complete_pair=True,same_best_set_H12_H14=equal,classification='same_best_set_at_H12_H14' if equal else 'different_best_sets_at_H12_H14',deepest_requested_depth=14,deepest_actual_completed_depth=14,deepest_requested_best_overlap_H8=bool(set(b14)&set(b8)),deepest_requested_best_overlap_H10=bool(set(b14)&set(b10)),available_depth12_and14_best_sets_equal=equal,adjacent_actual_depth_best_set_changes=sum(set(evidence[a][0]['best_moves'])!=set(evidence[b][0]['best_moves']) for a,b in zip(depths,depths[1:]) if b==a+1),raw_policy_chosen=policy,reference_H8_best_moves=b8,reference_H10_best_moves=b10)
                for name,v in expected_case.items():self.assertEqual(case[name],v)
                self.assertAlmostEqual(case['model_value_player0'],value,places=10);self.assertEqual(set(case['depth_evidence']),{str(d) for d in depths})
                for d,(source,kind,sourcejob,requested,cap) in sorted(evidence.items()):
                    item=case['depth_evidence'][str(d)];q={c['pit']:c['depth_result']['value_player0'] for c in source['candidates']};target=source['depth_result']['value_player0'];best=source['best_moves'];depth_count+=1
                    expected=dict(actual_depth=d,source_kind=kind,source_job=sourcejob,source_requested_depth=requested,source_node_budget=cap,source_report_sha256=config.digest(source),target_player0=target,best_moves=best,display_best_moves=[a+1 for a in best],action_scores=[dict(pit=a,display_pit=a+1,value_player0=q[a]) for a in legal],H8_lexicographic_choice_regret=sign*(target-q[min(b8)]),H10_lexicographic_choice_regret=sign*(target-q[min(b10)]),H8_best_tie_regret_min=min(sign*(target-q[a]) for a in b8),H8_best_tie_regret_max=max(sign*(target-q[a]) for a in b8),H10_best_tie_regret_min=min(sign*(target-q[a]) for a in b10),H10_best_tie_regret_max=max(sign*(target-q[a]) for a in b10),raw_policy_regret=sign*(target-q[policy]))
                    for name,v in expected.items():self.assertEqual(item[name],v)
                    self.assertAlmostEqual(item['model_value_absolute_error'],abs(value-target),places=10);actions=sorted(set(b8)|set(b10)|set(best));self.assertEqual(set(item['lines']),{str(a) for a in actions})
                    for action in actions:
                        candidate=next(c for c in source['candidates'] if c['pit']==action);n,c,b=check_forensic_line(self,state,candidate,d,item['lines'][str(action)],h8[key],h10[key]);line_count+=1;step_count+=n;captures+=c;bonuses+=b
                q14={c['pit']:c['depth_result']['value_player0'] for c in h14[key]['report']['candidates']};target=h14[key]['label']['target_player0']
                self.assertEqual(case['H14_regret_of_H12_best_ties'],[dict(pit=a,display_pit=a+1,regret=sign*(target-q14[a])) for a in b12]);self.assertEqual(case['H13_H14_same_best_set'],set(evidence[13][0]['best_moves'])==set(b14))
        expected=dict(roots=6,new_H14_complete=5,reused_H14_complete=1,H12_complete=6,H14_complete=6,requested_complete_pairs=6,requested_incomplete_cases=0,same_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] for c in cases.values()),changed_best_sets_H12_H14=sum(not c['same_best_set_H12_H14'] for c in cases.values()),same_best_sets_H13_H14=sum(c['H13_H14_same_best_set'] for c in cases.values()))
        self.assertEqual(report['counts'],expected);self.assertEqual(report['case_classifications'],dict(sorted(Counter(c['classification'] for c in cases.values()).items())))
        for flag in ('training_performed','model_changed','active_sampling_weights_changed','data_ingested','teacher_search_performed_by_analysis','new_PUCT_or_strength_tests','sealed_final_test_consumed','convergence_proved','full_game_solved'):self.assertFalse(report[flag])
        before=snapshot(p.DEFAULT);path=ROOT/'build/horizon-finish-comparison.json';oldreport=(sha(path),path.stat().st_mtime_ns)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed production reran teacher')):p.run()
        self.assertEqual(snapshot(p.DEFAULT),before)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Forensics reran teacher')):
            import runpy
            runpy.run_path(str(ROOT/'tools/analyze-horizon-finish.py'),run_name='__main__')
        self.assertEqual((sha(path),path.stat().st_mtime_ns),oldreport)
        for name,digest in load(p.BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['forensic_audit']=dict(cases=6,depth_evidence_rows=depth_count,literal_lines=line_count,literal_PV_choices=step_count,literal_capture_events=captures,literal_bonus_choices=bonuses,literal_model_forwards=6,all_regrets_model_errors_classifications_and_denominators_rederived=True,all_source_job_budget_depth_provenance_verified=True,all_literal_store_capture_bonus_pass_terminal_leaf_cutoff_and_common_prefix_facts_verified=True,counts=expected,completed_zero_teacher_calls=True,completed_bytes_mtimes_unchanged=True,forensics_zero_teacher_calls=True,forensics_bytes_mtimes_unchanged=True,all2314_prior_hashes_current=True)


GROUPS={'preflight':['test_01_baseline_fixed_model_and_two_distinct_budget_contracts','test_02_exact_five_selection_six_original_references_and_ownership','test_03_actual_fixture_rejection_fallback_completion_and_literal_histories','test_04_actual_initial_tier_pre_postcommit_recovery','test_05_actual_fallback_tier_pre_postcommit_recovery','test_06_identity_budget_fallback_ownership_history_and_publication_refusals'],
        'final':['test_07_all_production_histories_PVs_ledger_and_fresh_H14_verifier','test_08_independent_combined_sixroot_forensic_math_provenance_and_literal_PVs']}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=list(GROUPS));args=parser.parse_args();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(HorizonFinishIndependentAudit(name) for name in GROUPS[args.group]))
    names=['tests/horizon_finish_independent_audit.py','congklak/horizon_finish.py','tools/horizon-finish.py','tools/analyze-horizon-finish.py','build/horizon-finish-baseline.json','build/horizon-finish-registry.json','build/horizon-finish-plan.json','build/horizon-finish-implementation.json'];paths=[ROOT/name for name in names]
    paths += [path for path in HorizonFinishIndependentAudit.base.rglob('*') if path.is_file() and not path.name.endswith(('-wal','-shm'))]
    if args.group=='final':
        paths += [ROOT/name for name in ('data/horizon-finish-v1/work.sqlite3','data/horizon-finish-v1/manifest.json','build/horizon-finish-result.json','build/horizon-finish-comparison.json','build/horizon-finish-audit/preflight.json')];paths += [ROOT/name for name in load(WORK/'preflight.json')['inputs']]
    body=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=GROUPS[args.group],evidence=EVIDENCE,inputs=pins(paths),duration_seconds=time.monotonic()-started,retained_directory=relative(HorizonFinishIndependentAudit.base),failure_details=[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors]);save(WORK/(args.group+'.json'),body);print(config.canonical(dict(event='horizon_finish_independent_audit',group=args.group,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
