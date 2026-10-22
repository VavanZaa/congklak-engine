"""Independent exact-six H12/H14 probe, typed histories and live recovery."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import copy
import hashlib
import json
import math
from pathlib import Path
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
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,key_of,quotient,mechanism,rows,semantic,kill_barrier,run_child,snapshot
from natural_study_independent_audit import check_pvs
from quality_audit_acceptance import module
import reference as literal
WORK=ROOT/'build/horizon-probe-audit';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'

def old_reference():
    report=load(ROOT/'build/natural-study-stability.json')
    keys=sorted(r['state_key'] for r in report['metrics'] if r['best_sets_disjoint'] is True)
    old=ROOT/'data/natural-study-v1/work.sqlite3';root=rows(old,'roots','state');h8=rows(old,'h8_labels','state');h10=rows(old,'h10_labels','state')
    assert len(keys)==6
    return keys,{k:root[k] for k in keys},{k:h8[k] for k in keys},{k:h10[k] for k in keys}

def q_semantic(report):
    return dict(depth=report['choice_depth'],state=report['position'],legal=report['legal_actions'],value=report['depth_result']['value_player0'],best=report['best_moves'],Q=[(c['pit'],c['depth_result']['value_player0']) for c in report['candidates']])

def item_view(report,row):
    label=dict(kind='finite_choice_horizon',choice_depth=report['choice_depth'],oracle_manifest_sha256=None,target_player0=report['depth_result']['value_player0'],best_moves=report['best_moves'],action_scores=[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in report['candidates']],coverage=report['coverage'],work={k:report[k] for k in ('nodes','deposits','elapsed_ms','peak_process_bytes')})
    return dict(root_sha256=row['source_root_sha256'],label=label,report=report)

def check_histories(test,rootmap,h8,h10,l12,l14,a12,a14,settings,literal_pvs=True):
    from congklak import reports
    native=Native();paths=steps=history_reports=0;known={k:{8:q_semantic(h8[k]['report']),10:q_semantic(h10[k]['report'])} for k in rootmap}
    for stage,labels,attempts in (('h12',l12,a12),('h14',l14,a14)):
        depth=settings[stage]['depth'];budget=settings[stage]['node_budget'];test.assertEqual(set(attempts),set(rootmap))
        for key,a in attempts.items():
            row=rootmap[key];history=a['history'];test.assertEqual([r['choice_depth'] for r in history],list(range(1,len(history)+1)));test.assertEqual(a['completed_depth'],len(history));test.assertEqual(a['requested_depth'],depth);test.assertEqual(a['stage'],stage);test.assertEqual(a['study_row_sha256'],config.digest(row));test.assertEqual(a['source_root_sha256'],row['source_root_sha256']);test.assertGreater(a['work']['nodes'],0);test.assertLessEqual(a['work']['nodes'],budget)
            if a['status']=='complete':test.assertEqual(len(history),depth);test.assertEqual(labels[key],item_view(history[-1],row));test.assertEqual(a['stop_reason'],'max_depth')
            else:test.assertEqual(a['status'],'budget_rejected');test.assertLess(len(history),depth);test.assertNotIn(key,labels);test.assertIsNone(a['label_sha256']);test.assertEqual(a['stop_reason'],'node_budget');test.assertEqual(a['work']['nodes'],budget)
            previous={k:0 for k in a['work']}
            for r in history:
                reports.validate(r,row['state'],config.identity(native.build_id),native);test.assertEqual((r['search_options']['tt_mb'],r['search_options']['use_pvs']),(16,True))
                for name,total in a['work'].items():test.assertGreaterEqual(r[name],previous[name]);test.assertLessEqual(r[name],total)
                previous={name:r[name] for name in previous};sig=q_semantic(r);d=r['choice_depth']
                if d in known[key]:test.assertEqual(known[key][d],sig)
                known[key][d]=sig;history_reports+=1
                if literal_pvs:
                    n,m=check_pvs(test,row,item_view(r,row),d);paths+=n;steps+=m
    return dict(history_reports=history_reports,literal_PV_paths=paths,literal_PV_choices=steps,actual_depths={k:sorted(v) for k,v in known.items()})

def check_forensic_line(test,root,candidate,depth,line,old8,old10):
    q=norm(root);expected_steps=[];snapshots={};captures=bonuses=0
    for index,move in enumerate(candidate['pv'],1):
        before=q;t=literal.apply(q,move['pit']);test.assertEqual(t.actor,move['actor']);q=t.position;state=position(q);gain=[q.counts[7]-before.counts[7],q.counts[15]-before.counts[15]]
        captured=[e for e in t.events if e['kind']=='capture'];store=7 if t.actor==0 else 15;store_before=before.counts[store]+sum(e['kind']=='deposit' and e['index']==store for e in t.events)
        events=[dict(actor=t.actor,after=store_before+e['seeds'],before=store_before,count=e['seeds'],hand=0,opposite=e['opposite'],slot=e['index'],type='capture') for e in captured]
        last=next(e for e in reversed(t.events) if e['kind']=='deposit')['index'];own=last in literal.row_indices(7,t.actor)
        reason='own_store' if t.bonus else 'own_capture' if t.captured else 'own_empty' if own else 'opponent_empty'
        expected_steps.append(dict(choice=index,actor=t.actor,pit=move['pit'],display_pit=move['pit']+1,reason=reason,store_gain=gain,stores_after=state['stores'],margin_player0_after=literal.evaluate(q),active_after=sum(state['pits'][0]+state['pits'][1]),capture_events=events,capture_transfer=t.captured,bonus_choice=t.bonus and not t.terminal,forced_passes_after=[dict(zip(('from','to'),v)) for v in t.final_passes],terminal_after=t.terminal,relays=t.relay_pickups,deposits=t.deposits))
        captures+=len(events);bonuses+=t.bonus and not t.terminal
        if index in (8,10,12,14):snapshots[str(index)]=dict(stores=state['stores'],margin_player0=literal.evaluate(q),active=sum(state['pits'][0]+state['pits'][1]),interpretation='Store margin along THIS reported line; not the independently optimized shorter-depth target')
    test.assertEqual(line['steps'],expected_steps);test.assertEqual(line['pv'],candidate['pv']);test.assertEqual(line['target_player0'],candidate['depth_result']['value_player0']);test.assertEqual(line['pit'],candidate['pit']);test.assertEqual(line['display_pit'],candidate['pit']+1)
    terminal=literal.normalize(q).terminal;state=position(q);test.assertTrue(terminal or len(expected_steps)==depth);test.assertEqual(line['leaf'],dict(state=state,terminal=terminal,remaining_active=sum(state['pits'][0]+state['pits'][1]),margin_player0=literal.evaluate(q)));test.assertEqual(line['leaf']['margin_player0'],line['target_player0']);test.assertEqual(line['same_line_cutoff_snapshots'],snapshots)
    tail=[s for s in expected_steps if s['choice']>8];test.assertEqual(line['tail_after_choice8'],dict(choices=len(tail),capture_events=sum(len(s['capture_events']) for s in tail),capture_transfer=sum(s['capture_transfer'] for s in tail),bonus_choices=sum(s['bonus_choice'] for s in tail),store_gain=[sum(s['store_gain'][a] for s in tail) for a in (0,1)]))
    for d,item in ((8,old8),(10,old10)):
        old=next(c['pv'] for c in item['report']['candidates'] if c['pit']==candidate['pit']);n=0
        for a,b in zip(old,candidate['pv']):
            if (a['actor'],a['pit'])!=(b['actor'],b['pit']):break
            n+=1
        test.assertEqual(line['common_prefix_vs_H'+str(d)],dict(shared_choices=n,first_different_choice=n+1 if n<min(len(old),len(candidate['pv'])) else None,shorter_line_is_prefix=n==min(len(old),len(candidate['pv']))))
    return len(expected_steps),captures,bonuses

class HorizonProbeIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p
    def backup(self,run,path):
        with closing(dbopen(run/'work.sqlite3')) as source,closing(sqlite3.connect(path)) as target:source.backup(target)
    def command(self,stage,path):return [sys.executable,str(ROOT/'tools/horizon-probe.py'),stage,'--run',str(path),'--fixture','--roots','2']
    def shared_fixture(self):
        from congklak import horizon_probe as p
        path=self.base/'shared-independent-fresh'
        if not path.exists():run_child(self.command('all',path),self.base/'shared-independent-fresh.log',timeout=900)
        p.load_rows(path);return path
    def seeded_probe(self,parent,stage,event):
        source=self.shared_fixture();run=parent/event;shutil.copytree(source,run);keys=sorted(rows(run/'work.sqlite3','roots','state'));removed=[]
        with closing(sqlite3.connect(run/'work.sqlite3')) as db:
            for name in ('h12_attempts','h12_labels','h14_attempts','h14_labels'):
                if stage=='h12' and name.startswith('h14'):db.execute('DELETE FROM '+name);removed.append(dict(table=name,keys='all'))
                elif name.startswith(stage):db.execute('DELETE FROM '+name+' WHERE state=?',(keys[-1],));removed.append(dict(table=name,keys=[keys[-1]]))
            db.execute("DELETE FROM meta WHERE key='complete'");db.commit()
        (run/'manifest.json').unlink();save(parent/(event+'-seed.json'),dict(source=relative(source),source_journal_sha256=sha(source/'work.sqlite3'),retained_first_target=keys[0],removed=removed,completion_and_export_removed=True));return run
    def test_01_baseline_fixed_model_and_bounded_contract(self):
        baseline=load(ROOT/'build/horizon-probe-baseline.json');self.assertEqual(baseline['inputs_count'],2199);self.assertEqual(len(baseline['inputs']),2199)
        for path,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/path),digest,path)
        self.assertEqual(sha(MODEL),MODEL_SHA)
        EVIDENCE['baseline']=dict(prior_inputs=2199,old_accepted_inputs=2197,all_hashes_current=True,fixed_model_sha256=MODEL_SHA)
        from congklak import horizon_probe as p
        c=p.CONTRACT;self.assertEqual(c['roots'],6);self.assertEqual(c['h12'],{'depth':12,'node_budget':10000000});self.assertEqual(c['h14'],{'depth':14,'node_budget':25000000});self.assertEqual((c['keep_history'],c['time_ms'],c['tt_mb'],c['use_pvs'],c['memory_mb']),(True,0,16,True,512));self.assertEqual(c['maximum_new_production_nodes'],210000000);self.assertTrue(c['investigation_only']);self.assertFalse(c['automatic_expansion']);self.assertFalse(c['training_performed']);self.assertFalse(c['sealed_final_test_consumed']);p.admit()

    def test_02_exact_six_original_wrappers_references_and_ownership(self):
        from congklak import horizon_probe as p
        keys,old,h8,h10=old_reference();registry=load(p.REGISTRY);self.assertEqual(registry['state_keys'],keys);self.assertEqual(registry['rows'],old);self.assertEqual(registry['h8_references'],h8);self.assertEqual(registry['h10_references'],h10);self.assertTrue(registry['investigation_only']);self.assertTrue(registry['no_ingest_or_training'])
        self.assertEqual(Counter(r['cohort'] for r in old.values()),{'training_candidate':5,'validation':1});self.assertEqual(Counter(r['study_role'] for r in old.values()),{'followup':2,'fresh':4})
        for k in keys:self.assertEqual(k,key_of(old[k]['state']));self.assertFalse(set(h8[k]['label']['best_moves'])&set(h10[k]['label']['best_moves']))
        EVIDENCE['exact_selection']=dict(roots=6,state_keys=keys,cohorts=dict(Counter(r['cohort'] for r in old.values())),original_roles=dict(Counter(r['study_role'] for r in old.values())),all_original_wrappers_and_H8_H10_items_exact=True,investigation_only=True)

    def test_03_real_fresh_fixture_complete_and_auxiliary_history_types(self):
        from congklak import horizon_probe as p
        run=self.shared_fixture();r,h8,h10,l12,l14,a12,a14,m=p.load_rows(run);result=check_histories(self,r,h8,h10,l12,l14,a12,a14,p.settings_for(True,2))
        self.assertEqual(len(r),2);self.assertEqual(len(l12),2);self.assertEqual(len(l14),1);self.assertEqual(sum(a['status']=='budget_rejected' for a in a14.values()),1);rejected=next(a for a in a14.values() if a['status']=='budget_rejected');self.assertEqual(rejected['completed_depth'],13);self.assertEqual(m['total_new_nodes'],sum(a['work']['nodes'] for a in [*a12.values(),*a14.values()]))
        EVIDENCE['fresh_fixture']=dict(**result,roots=2,H12_complete=2,H14_complete=1,H14_budget_rejected=1,rejected_H14_auxiliary_depth=13,all_overlap_Q_best_semantics_match=True,total_fixture_nodes=m['total_new_nodes'],cost_scope='Independent fixture, separate from production210m cap')

    def recovery(self,stage):
        from congklak import horizon_probe as p
        parent=self.fresh(stage+'-kills');source=self.shared_fixture();fresh=p.load_rows(source);attemptpos=5 if stage=='h12' else 6;labelpos=3 if stage=='h12' else 4;records=[]
        for suffix,prefix in (('precommit',1),('committed',2)):
            event='horizon_probe_'+stage+'_'+suffix;run=self.seeded_probe(parent,stage,event);probe=parent/(event+'-probe')
            marker=kill_barrier(self.command(stage,run),probe,event,env={'HORIZON_PROBE_BLOCK_EVENT':event,'HORIZON_PROBE_BLOCK_STEP':str(prefix),'HORIZON_PROBE_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=prefix:v['completed']==n,timeout=900)
            before=rows(run/'work.sqlite3',stage+'_attempts','state');beforelabels=rows(run/'work.sqlite3',stage+'_labels','state');self.assertEqual(len(before),prefix);self.backup(run,probe/'validated-prefix.sqlite3');run_child(self.command(stage,run),parent/(event+'-resume.log'),timeout=900)
            actual=rows(run/'work.sqlite3',stage+'_attempts','state');actual_labels=rows(run/'work.sqlite3',stage+'_labels','state');self.assertEqual(semantic(actual),semantic(fresh[attemptpos]));self.assertEqual(semantic(actual_labels),semantic(fresh[labelpos]));self.assertEqual({k:actual[k] for k in before},before);self.assertEqual({k:actual_labels[k] for k in beforelabels},beforelabels)
            stable=snapshot(run)
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed stage rerun executed teacher')):p.run(run,stage=stage,fixture=True,roots=2)
            self.assertEqual(snapshot(run),stable);records.append(dict(event=event,stage=stage,committed_prefix=prefix,targets_after_resume=2,prefix_seeded_from_actual_independent_fresh_fixture=True,seed_record=relative(parent/(event+'-seed.json')),prefix_exactly_retained=True,fresh_semantics_and_counters_equal=True,completed_zero_teacher_calls=True,bytes_mtimes_unchanged=True,marker=marker))
        return dict(actual_kills=True,stage=stage,tests=records,raw_seed_edits_markers_backups_retained=True)

    def test_04_actual_H12_pre_postcommit_recovery(self):EVIDENCE['live_H12_recovery']=self.recovery('h12')
    def test_05_actual_H14_pre_postcommit_recovery(self):EVIDENCE['live_H14_recovery']=self.recovery('h14')

    def test_06_history_identity_ownership_publication_and_auxiliary_refusals(self):
        from congklak import horizon_probe as p,reports
        parent=self.fresh('refusals');source=self.shared_fixture();r,h8,h10,l12,l14,a12,a14,m=p.load_rows(source);settings=p.settings_for(True,2);native=Native();key=next(iter(a12));row=r[key]
        bad=copy.deepcopy(a12[key]);bad['history'].pop(0);bad['completed_depth']-=1;bad['native_last_completed_depth']-=1
        with self.assertRaises(ValueError):p.validate_attempt(row,bad,l12[key],'h12',settings,native)
        bad=copy.deepcopy(a12[key]);bad['work']={name:0 for name in bad['work']}
        with self.assertRaises(ValueError):p.validate_attempt(row,bad,l12[key],'h12',settings,native)
        bad=copy.deepcopy(a12[key]);bad['history'][0]['search_options'].update(tt_mb=0,tt_allocated_bytes=0,tt_capacity=0,use_pvs=False)
        reports.validate(bad['history'][0],row['state'],config.identity(native.build_id),native)
        with self.assertRaises(ValueError):p.validate_attempt(row,bad,l12[key],'h12',settings,native)
        rejected_key=next(k for k,a in a14.items() if a['status']=='budget_rejected');rejected=copy.deepcopy(a14[rejected_key]);self.assertEqual(rejected['completed_depth'],13)
        rejected['status']='complete';rejected['stop_reason']='max_depth';fake=p.item_for(rejected['history'][-1],r[rejected_key]);rejected['label_sha256']=config.digest(fake)
        with self.assertRaises(ValueError):p.validate_attempt(r[rejected_key],rejected,fake,'h14',settings,native)
        for kind in ('identity','ownership','reference','history','prefix','complete_missing_export'):
            run=parent/kind;shutil.copytree(source,run)
            with closing(sqlite3.connect(run/'work.sqlite3')) as db:
                if kind=='identity':
                    ident=json.loads(db.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0]);ident['schema']='incompatible-probe';marker=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0]);marker['identity']=ident
                    for name,v in (('identity',ident),('complete',marker)):db.execute('UPDATE meta SET body=?,hash=? WHERE key=?',(config.canonical(v),config.digest(v),name))
                    (run/'manifest.json').write_text(config.canonical(marker)+'\n',encoding='utf-8')
                elif kind=='ownership':
                    v=copy.deepcopy(row);v['cohort']='diagnostic';db.execute('UPDATE roots SET body=?,hash=? WHERE state=?',(config.canonical(v),config.digest(v),key))
                elif kind=='reference':
                    v=copy.deepcopy(h8[key]);v['report']['elapsed_ms']+=1;v['label']['work']['elapsed_ms']+=1;db.execute('UPDATE h8_references SET body=?,hash=? WHERE state=?',(config.canonical(v),config.digest(v),key))
                elif kind=='history':
                    v=copy.deepcopy(a12[key]);v['history'][0]['depth_result']['value_player0']+=1;db.execute('UPDATE h12_attempts SET body=?,hash=? WHERE state=?',(config.canonical(v),config.digest(v),key))
                elif kind=='prefix':db.execute('DELETE FROM h12_attempts WHERE state=?',(key,))
                else:
                    marker=json.loads(db.execute("SELECT body FROM meta WHERE key='complete'").fetchone()[0]);marker['auxiliary_completed_reports']+=1;db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)));(run/'manifest.json').unlink()
                db.commit()
            with self.assertRaises((ValueError,KeyError,p.teacher.DeeperError)):p.load_rows(run)
            self.assertNotEqual(run_child(self.command('all',run),parent/(kind+'.log'),success=False,timeout=900),0)
            if kind=='complete_missing_export':self.assertFalse((run/'manifest.json').exists())
        repaired=parent/'repair';shutil.copytree(source,repaired);before=snapshot(repaired);export=(repaired/'manifest.json').read_bytes();(repaired/'manifest.json').unlink()
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Export repair executed search')):p.run(repaired,fixture=True,roots=2)
        self.assertEqual((repaired/'manifest.json').read_bytes(),export);after=snapshot(repaired);self.assertEqual({k:v for k,v in before.items() if k!='manifest.json'},{k:v for k,v in after.items() if k!='manifest.json'})
        # Synthetic eligibility check: available auxiliary12 must not complete a missing requested12 label.
        tool=module('horizon_forensics_aux_eligibility','tools/analyze-horizon-probe.py');complete14=next(iter(l14));no12={k:v for k,v in l12.items() if k!=complete14};a12_synthetic=copy.deepcopy(a12);a12_synthetic[complete14]['status']='budget_rejected'
        case=tool.analyze_case(complete14,r[complete14],h8,h10,no12,l14,a12_synthetic,a14,dict(value_player0=0.,policy_logits=[0.]*7),native,{12:10000000,14:25000000});save(parent/'synthetic-auxiliary12-not-requested12.json',case)
        self.assertFalse(case['requested_complete_pair']);self.assertIsNone(case['same_best_set_H12_H14']);self.assertEqual(case['classification'],'requested_comparison_incomplete');self.assertEqual(case['requested_depths_available'],[14]);self.assertEqual(case['depth_evidence']['12']['source_kind'],'auxiliary_completed_iteration');self.assertEqual(case['depth_evidence']['12']['source_job'],'h14');self.assertEqual(case['depth_evidence']['12']['source_node_budget'],25000000)
        EVIDENCE['refusals_and_eligibility']=dict(history_gap_refused=True,zero_work_refused=True,wrong_TT_PVS_refused=True,real_H14_auxiliary13_not_admitted_as_H14=True,rehashed_identity_ownership_reference_history_prefix_refused=True,invalid_complete_not_published=True,missing_export_repaired_without_teacher=True,repair_preserves_journal_bytes_mtime=True,synthetic_auxiliary12_does_not_complete_requested12=True)

    def test_07_all_production_histories_PVs_overlap_and_bounded_fresh_deeper_check(self):
        from congklak import horizon_probe as p
        r,h8,h10,l12,l14,a12,a14,m=p.load_rows();keys,old,old8,old10=old_reference();self.assertEqual(list(r),keys);self.assertEqual(r,old);self.assertEqual(h8,old8);self.assertEqual(h10,old10)
        proof=check_histories(self,r,h8,h10,l12,l14,a12,a14,p.CONTRACT)
        expected=dict(roots=6,h8_references=6,h10_references=6,h12_labels=len(l12),h12_rejected=sum(a['status']=='budget_rejected' for a in a12.values()),h14_labels=len(l14),h14_rejected=sum(a['status']=='budget_rejected' for a in a14.values()),h12_h14_complete_pairs=len(set(l12)&set(l14)),auxiliary_completed_reports=sum(len(a['history']) for a in [*a12.values(),*a14.values()]),total_new_nodes=sum(a['work']['nodes'] for a in [*a12.values(),*a14.values()]))
        for name,v in expected.items():self.assertEqual(m[name],v)
        self.assertLessEqual(expected['total_new_nodes'],210000000);self.assertEqual(m,load(p.OUTPUT));self.assertEqual(m,load(p.DEFAULT/'manifest.json'))
        key=keys[0];state=r[key]['state'];native=Native();fresh=native.analyze(flatten(state),actor=state['controller'],total=98,max_depth=12,time_ms=0,node_budget=10000000,memory_mb=512,tt_mb=16,use_pvs=True,keep_history=True)
        save(self.base/'fresh-first-key-H12-bounded.json',fresh);self.assertEqual(fresh['stop_reason'],a12[key]['stop_reason']);self.assertEqual([q_semantic(v) for v in fresh['reports']],[q_semantic(v) for v in a12[key]['history']]);self.assertLessEqual(fresh['nodes'],10000000)
        EVIDENCE['production_teacher_audit']=dict(**expected,**proof,all_six_original_rows_and_references_exact=True,all_H14_roots_attempted=True,requested_and_auxiliary_types_verified=True,all_overlap_Q_best_semantics_match=True,fresh_bounded_check=dict(state_key=key,requested_depth=12,node_cap=10000000,actual_nodes=fresh['nodes'],last_completed_depth=fresh['last_completed_depth'],all_history_Q_best_semantics_equal=True),fresh_check_cost_separate_from_production=True)

    def test_08_independent_forensic_provenance_math_and_literal_PV_facts(self):
        from congklak import horizon_probe as p
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        r,h8,h10,l12,l14,a12,a14,m=p.load_rows();report=load(ROOT/'build/horizon-probe-forensics.json');cases={c['state_key']:c for c in report['cases']};self.assertEqual(list(cases),list(r));self.assertEqual(len(cases),6);line_count=step_count=captures=bonuses=depth_count=0
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
            for index,(key,row) in enumerate(r.items(),1):
                case=cases[key];state=row['state'];value,_,logits=value_and_policy(norm(state),evaluator.model.envelope['model']);pred=evaluator.evaluate_many([state])[0];self.assertAlmostEqual(value,pred['value_player0'],places=10);self.assertEqual(logits,pred['policy_logits']);legal=h8[key]['report']['legal_actions'];policy=min(legal,key=lambda a:(-logits[a],a));sign=1 if state['controller']==0 else -1;b8=h8[key]['label']['best_moves'];b10=h10[key]['label']['best_moves']
                evidence={8:(h8[key]['report'],'reference','old_h8',8,None),10:(h10[key]['report'],'reference','old_h10',10,None)}
                if key in l12:evidence[12]=(l12[key]['report'],'requested_label','h12',12,10000000)
                if key in l14:evidence[14]=(l14[key]['report'],'requested_label','h14',14,25000000)
                for stage,attempt,requested,budget in (('h14',a14[key],14,25000000),('h12',a12[key],12,10000000)):
                    for item in attempt['history']:
                        d=item['choice_depth']
                        if d>=8 and d not in evidence:evidence[d]=(item,'auxiliary_completed_iteration',stage,requested,budget)
                paired=key in l12 and key in l14;equal=set(l12[key]['label']['best_moves'])==set(l14[key]['label']['best_moves']) if paired else None;requested=[d for d in (12,14) if (key in l12 if d==12 else key in l14)];deep_requested=max([8,10]+requested);deep_best=evidence[deep_requested][0]['best_moves'];depths=sorted(evidence)
                expected_case=dict(case_id='D'+str(index),state=state,family=row['family'],source_role=row['study_role'],cohort=row['cohort'],investigation_only=True,requested_status=dict(h12=a12[key]['status'],h14=a14[key]['status']),requested_depths_available=requested,auxiliary_completed_depths=[d for d in depths if evidence[d][1]=='auxiliary_completed_iteration'],requested_complete_pair=paired,same_best_set_H12_H14=equal,classification=('same_best_set_at_H12_H14' if equal else 'different_best_sets_at_H12_H14') if paired else 'requested_comparison_incomplete',deepest_requested_depth=deep_requested,deepest_actual_completed_depth=max(evidence),deepest_requested_best_overlap_H8=bool(set(deep_best)&set(b8)),deepest_requested_best_overlap_H10=bool(set(deep_best)&set(b10)),available_depth12_and14_best_sets_equal=(set(evidence[12][0]['best_moves'])==set(evidence[14][0]['best_moves'])) if 12 in evidence and 14 in evidence else None,adjacent_actual_depth_best_set_changes=sum(set(evidence[a][0]['best_moves'])!=set(evidence[b][0]['best_moves']) for a,b in zip(depths,depths[1:]) if b==a+1),raw_policy_chosen=policy,reference_H8_best_moves=b8,reference_H10_best_moves=b10)
                for name,v in expected_case.items():self.assertEqual(case[name],v)
                self.assertAlmostEqual(case['model_value_player0'],value,places=10);self.assertEqual(set(case['depth_evidence']),{str(d) for d in depths})
                for d,(source,kind,job,requested_depth,budget) in sorted(evidence.items()):
                    item=case['depth_evidence'][str(d)];q={c['pit']:c['depth_result']['value_player0'] for c in source['candidates']};target=source['depth_result']['value_player0'];best=source['best_moves'];depth_count+=1
                    expected=dict(actual_depth=d,source_kind=kind,source_job=job,source_requested_depth=requested_depth,source_node_budget=budget,source_report_sha256=config.digest(source),target_player0=target,best_moves=best,display_best_moves=[a+1 for a in best],action_scores=[dict(pit=a,display_pit=a+1,value_player0=q[a]) for a in legal],H8_lexicographic_choice_regret=sign*(target-q[min(b8)]),H10_lexicographic_choice_regret=sign*(target-q[min(b10)]),H8_best_tie_regret_min=min(sign*(target-q[a]) for a in b8),H8_best_tie_regret_max=max(sign*(target-q[a]) for a in b8),H10_best_tie_regret_min=min(sign*(target-q[a]) for a in b10),H10_best_tie_regret_max=max(sign*(target-q[a]) for a in b10),raw_policy_regret=sign*(target-q[policy]))
                    for name,v in expected.items():self.assertEqual(item[name],v)
                    self.assertAlmostEqual(item['model_value_absolute_error'],abs(value-target),places=10);actions=sorted(set(b8)|set(b10)|set(best));self.assertEqual(set(item['lines']),{str(a) for a in actions})
                    for action in actions:
                        candidate=next(c for c in source['candidates'] if c['pit']==action);n,c,b=check_forensic_line(self,state,candidate,d,item['lines'][str(action)],h8[key],h10[key]);line_count+=1;step_count+=n;captures+=c;bonuses+=b
        counts=dict(roots=6,h12_attempts=6,h14_attempts=6,h12_complete=len(l12),h14_complete=len(l14),requested_complete_pairs=len(set(l12)&set(l14)),h12_rejected=sum(a['status']=='budget_rejected' for a in a12.values()),h14_rejected=sum(a['status']=='budget_rejected' for a in a14.values()),stable_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] is True for c in cases.values()),changed_best_sets_H12_H14=sum(c['same_best_set_H12_H14'] is False for c in cases.values()),requested_incomplete_cases=sum(not c['requested_complete_pair'] for c in cases.values()),cases_with_deeper_requested_label=sum(c['deepest_requested_depth']>10 for c in cases.values()))
        self.assertEqual(report['counts'],counts);self.assertEqual(report['case_classifications'],dict(sorted(Counter(c['classification'] for c in cases.values()).items())))
        for flag in ('training_performed','model_changed','active_sampling_weights_changed','data_ingested','teacher_search_performed_by_analysis','new_PUCT_or_strength_tests','sealed_final_test_consumed'):self.assertFalse(report[flag])
        before=snapshot(p.DEFAULT);path=ROOT/'build/horizon-probe-forensics.json';oldreport=(sha(path),path.stat().st_mtime_ns)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed probe executed teacher')):p.run()
        self.assertEqual(snapshot(p.DEFAULT),before);run_child([sys.executable,str(ROOT/'tools/analyze-horizon-probe.py')],self.base/'forensics-rerun.log',timeout=300);self.assertEqual((sha(path),path.stat().st_mtime_ns),oldreport)
        for name,digest in load(ROOT/'build/horizon-probe-baseline.json')['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        EVIDENCE['forensic_audit']=dict(cases=6,depth_evidence_rows=depth_count,literal_lines=line_count,literal_PV_choices=step_count,literal_capture_events=captures,literal_bonus_choices=bonuses,literal_model_forwards=6,all_regrets_model_errors_and_case_classifications_rederived=True,all_source_job_budget_depth_provenance_verified=True,all_literal_store_capture_bonus_pass_terminal_leaf_and_cutoff_facts_verified=True,all_common_prefix_comparisons_verified=True,requested_pairs_distinct_from_auxiliary_availability=True,requested_counts=counts,completed_zero_teacher_calls=True,completed_bytes_mtimes_unchanged=True,forensics_bytes_mtimes_unchanged=True,all2199_prior_hashes_current=True)

GROUPS={'preflight':['test_01_baseline_fixed_model_and_bounded_contract','test_02_exact_six_original_wrappers_references_and_ownership','test_03_real_fresh_fixture_complete_and_auxiliary_history_types','test_04_actual_H12_pre_postcommit_recovery','test_05_actual_H14_pre_postcommit_recovery','test_06_history_identity_ownership_publication_and_auxiliary_refusals'],
        'final':['test_07_all_production_histories_PVs_overlap_and_bounded_fresh_deeper_check','test_08_independent_forensic_provenance_math_and_literal_PV_facts']}

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=list(GROUPS));a=parser.parse_args();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(HorizonProbeIndependentAudit(n) for n in GROUPS[a.group]))
    files=['tests/horizon_probe_independent_audit.py','congklak/horizon_probe.py','tools/horizon-probe.py','tools/analyze-horizon-probe.py','build/horizon-probe-baseline.json','build/horizon-probe-registry.json','build/horizon-probe-plan.json','build/horizon-probe-implementation.json'];paths=[ROOT/f for f in files]
    paths += [p for p in HorizonProbeIndependentAudit.base.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]
    if a.group=='final':
        paths += [ROOT/f for f in ('data/horizon-probe-v1/work.sqlite3','data/horizon-probe-v1/manifest.json','build/horizon-probe-result.json','build/horizon-probe-forensics.json','build/horizon-probe-audit/preflight.json')];paths += [ROOT/f for f in load(WORK/'preflight.json')['inputs']]
    body=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=GROUPS[a.group],evidence=EVIDENCE,inputs=pins(paths),duration_seconds=time.monotonic()-started,retained_directory=relative(HorizonProbeIndependentAudit.base),failure_details=[dict(method=str(t),trace=trace) for t,trace in result.failures+result.errors]);save(WORK/(a.group+'.json'),body);print(config.canonical(dict(event='horizon_probe_independent_audit',group=a.group,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

