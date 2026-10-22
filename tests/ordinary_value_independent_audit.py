"""Independent ordinary-value targets, exact optimizer, recovery and allocation audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import bisect,copy,hashlib,json,math,random,shutil,sqlite3,statistics,sys,time,unittest,uuid
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,quotient,mechanism,rows,semantic,kill_barrier,run_child,snapshot
from value_pilot_independent_audit import pt_catalog,literal_game,FROZEN,TRAINABLE
from horizon_probe_independent_audit import item_view,q_semantic
from natural_study_independent_audit import check_pvs
from ordinary_selfplay_independent_audit import components_reference,literal_family,complete_game_literal
from selfplay_pilot_independent_audit import raw_key,tuples
import reference as literal
WORK=ROOT/'build/ordinary-value-audit';BASELINE=ROOT/'build/ordinary-value-baseline.json';FREEZE=ROOT/'build/ordinary-value-audit-source-freeze.json';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json';MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'


def backup(path,output):
    with closing(dbopen(path)) as source,closing(sqlite3.connect(output)) as target:source.backup(target)


def patch_sql(path,table,key,mutate):
    with closing(sqlite3.connect(path)) as db:
        text=db.execute('SELECT body FROM '+table+' WHERE '+('key' if table=='meta' else 'state')+'=?',(key,)).fetchone()[0]
        body=json.loads(text);mutate(body);db.execute('UPDATE '+table+' SET body=?,hash=? WHERE '+('key' if table=='meta' else 'state')+'=?',(config.canonical(body),config.digest(body),key));db.commit()


def check_labels(test,rootmap,stores,settings):
    from congklak import reports
    histories=paths=choices=nodes=0;known=defaultdict(dict);summary={}
    for stage in ('h8','h10','h12'):
        depth=settings[stage]['depth'];budget=settings[stage]['node_budget'];attempts=stores[stage+'_attempts'];labels=stores[stage+'_labels']
        test.assertEqual(list(attempts),list(rootmap));test.assertLessEqual(set(labels),set(attempts));summary[stage]=dict(complete=len(labels),rejected=len(attempts)-len(labels))
        for key,a in attempts.items():
            row=rootmap[key];test.assertEqual(a['study_row_sha256'],config.digest(row));test.assertEqual(a['source_root_sha256'],config.digest(row['root']));test.assertEqual(a['stage'],stage);test.assertEqual(a['requested_depth'],depth)
            test.assertEqual([r['choice_depth'] for r in a['history']],list(range(1,a['completed_depth']+1)));test.assertGreater(a['work']['nodes'],0);test.assertLessEqual(a['work']['nodes'],budget);nodes+=a['work']['nodes']
            if a['status']=='complete':test.assertEqual(a['completed_depth'],depth);test.assertEqual(labels[key],item_view(a['history'][-1],row));test.assertEqual(a['stop_reason'],'max_depth')
            else:test.assertEqual(a['status'],'budget_rejected');test.assertNotIn(key,labels);test.assertLess(a['completed_depth'],depth);test.assertEqual(a['work']['nodes'],budget);test.assertEqual(a['stop_reason'],'node_budget');test.assertIsNone(a['label_sha256'])
            previous={name:0 for name in a['work']}
            for report in a['history']:
                reports.validate(report,row['state'],config.identity(Native().build_id),Native());test.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True))
                for name,value in previous.items():test.assertGreaterEqual(report[name],value);test.assertLessEqual(report[name],a['work'][name])
                previous={name:report[name] for name in previous};d=report['choice_depth'];signature=q_semantic(report)
                if d in known[key]:test.assertEqual(known[key][d],signature)
                known[key][d]=signature;n,m=check_pvs(test,row,item_view(report,row),d);paths+=n;choices+=m;histories+=1
    return dict(roots=len(rootmap),requested_attempts=3*len(rootmap),requested_depths=[8,10,12],summary=summary,
        all_complete_history_reports=histories,literal_PV_paths=paths,literal_PV_choices=choices,new_nodes=nodes,
        partials_never_admitted_as_requested_labels=True,finite_horizon_not_truth=True)


def interval_reference(values,plan):
    if not values:return None
    rng=random.Random(plan['bootstrap_seed']);n=len(values);samples=sorted(statistics.mean(values[int(rng.random()*n)] for _ in range(n)) for _ in range(plan['bootstrap_samples']))
    # Match the declared nearest-rank percentile convention, independently.
    return [samples[int(.025*len(samples))],samples[min(len(samples)-1,int(.975*len(samples)))]]


class OrdinaryValueIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):path=self.base/name;path.mkdir();return path
    def worker(self,spec,path,torch=False):
        path.mkdir();spec={**spec,'output':str(path/'proof.json')};save(path/'spec.json',spec)
        runtime=load(ROOT/'build/training-runtime.json')['python'] if torch else sys.executable
        run_child([runtime,str(ROOT/'tools/ordinary-value-audit.py'),str(path/'spec.json')],path/'worker.log',timeout=600)
        value=load(path/'proof.json');self.assertTrue(value['passed']);return dict(path=relative(path/'proof.json'),**value)
    def label_command(self,path):return [sys.executable,str(ROOT/'tools/ordinary-value-pilot.py'),'labels','--fixture','--run',str(path)]
    def train_command(self,path):return [load(ROOT/'build/training-runtime.json')['python'],str(ROOT/'tools/ordinary-value-pilot.py'),'train','--fixture','--run',str(path)]

    def test_01_frozen_sources_true_roles_independent_admission_and_sampler(self):
        from congklak import ordinary_value_pilot as p,ordinary_value_benchmark as b,ordinary_value_followup as f,features_relay
        baseline=load(BASELINE);self.assertEqual((baseline['inputs_count'],baseline['prior_accepted_inputs']),(3364,3361));self.assertEqual(len(baseline['inputs']),3364)
        for name,digest in baseline['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA);p.admit();b.pretraining_witness();f.admit()
        train=p.TRAIN;self.assertEqual({k:train[k] for k in ('steps','batch_size','checkpoint_every','seed','lr','trainable_parameters','device')},dict(steps=200,batch_size=64,checkpoint_every=20,seed=20261028,lr=.0001,trainable_parameters=65,device='cpu'))
        self.assertEqual(set(p.TRAINABLE),TRAINABLE)
        original=ROOT/'data/ordinary-selfplay-pilot-v1/work.sqlite3';oldroots=rows(original,'roots','state');initial=[r for r in rows(original,'sp_rows','id').values() if r['choice']==0];self.assertEqual(len(initial),48)
        expectedval={key:p.wrap(row) for key,row in oldroots.items() if row['cohort']=='validation'};registry=load(p.REGISTRY);self.assertEqual(registry['rows'],expectedval);self.assertEqual(len(expectedval),6)
        prior=load(ROOT/'data/ordinary-selfplay-pilot-v1/rollout-protection.json');projection=load(b.PROJECTION);qset=set(prior['quotients'])|set(projection['reserved_quotients']);fset=set(prior['families'])|set(projection['reserved_families'])
        self.assertEqual(set(registry['quotients']),qset);self.assertEqual(set(registry['families']),fset);self.assertEqual(len(registry['validation_pool_families']),100);self.assertEqual(registry['validation_pool_unique_quotients'],38)
        # The entire new validation ownership pool remains protected, not only6 selected.
        pool=rows(original,'families','idx');valq={r['quotient'] for body in pool.values() if body['cohort']=='validation' for r in body['candidates']};valf={body['family'] for body in pool.values() if body['cohort']=='validation'}
        self.assertEqual(set(prior['new_validation_family_ids']),valf);self.assertLessEqual(valq,qset);self.assertLessEqual(valf,fset)
        oldbase=[r for r in load(ROOT/'build/value-pilot-rows.json')['groups']['train'] if r['pool']=='base'];self.assertEqual(len(oldbase),7563)
        candidates=[*oldbase,*rows(original,'sp_rows','id').values()];blocked={r['family'] for r in candidates if r['family'] in fset or quotient(r['state']) in qset}
        manifest,groups=p.load_data();self.assertEqual(manifest['excluded_origin_families'].keys(),blocked)
        self.assertEqual([r['source_id'] for r in groups['rehearsal']],sorted(r['state_key'] for r in oldbase if r['family'] not in blocked))
        for band,name in (('50-74/ordinary','high'),('25-49/ordinary','low')):
            expected=sorted((r for r in initial if r['stratum']==band and r['family'] not in blocked),key=lambda r:r['id']);self.assertEqual([r['source_id'] for r in groups[name]],[r['id'] for r in expected]);self.assertGreaterEqual(len({r['state_key'] for r in groups[name]}),8)
        bysource={**{r['state_key']:r for r in oldbase},**{r['id']:r for r in initial}};encoded=0
        for name,values in groups.items():
            for row in values:
                old=bysource[row['source_id']];self.assertEqual(row['source_row_sha256'],config.digest(old));self.assertEqual((row['state'],row['family'],row['stratum']),(old['state'],old['family'],old['stratum']));self.assertNotIn(row['family'],fset|blocked);self.assertNotIn(quotient(row['state']),qset)
                target=old['label']['target_player0'] if name=='rehearsal' else old['terminal_return_player0'];self.assertEqual(row['target_player0'],target);self.assertEqual(row['cached_features'],old['cached_features'] if name=='rehearsal' else old['features'])
                # Recompute every cached relay input; independent target orientation below.
                x,d,r=features_relay.encode(row['state'],127,Native());self.assertEqual({key:row['cached_features'][key] for key in ('x','d','r')},dict(x=x,d=d,r=r));self.assertEqual(r,(1 if row['state']['controller']==0 else -1)*sum(map(sum,row['state']['pits'])));self.assertGreaterEqual((target-d)/r,-1);self.assertLessEqual((target-d)/r,1);encoded+=1
        counts=Counter(r['stratum'] for r in groups['rehearsal']);self.assertAlmostEqual(math.fsum(r['sampling_probability'] for r in groups['rehearsal']),1.)
        for row in groups['rehearsal']:self.assertEqual(row['sampling_probability'],.9/len(groups['rehearsal'])+.1/(len(counts)*counts[row['stratum']]))
        sampler=p.sampler_for(groups);rng=random.Random(20261028);batch=p.sample_batch(rng,sampler);ind=random.Random(20261028);expected={}
        for name in ('high','low'):
            roots=defaultdict(list)
            for i,row in enumerate(groups[name]):roots[row['state_key']].append(i)
            keys=sorted(roots);expected[name]=[ind.choice(roots[ind.choice(keys)]) for _ in range(16)]
        cdf=[];total=0.
        for row in groups['rehearsal']:total+=row['sampling_probability'];cdf.append(total)
        expected['rehearsal']=[bisect.bisect_right(cdf,ind.random()*cdf[-1]) for _ in range(32)];self.assertEqual(batch,expected);self.assertEqual(rng.getstate(),ind.getstate())
        self.assertEqual(len(projection['arena_starts']),12);self.assertFalse(set(r['family'] for r in projection['arena_starts'])&set(projection['prior_arena_families']));self.assertEqual(len(projection['prior_arena_families']),144)
        self.assertNotIn('torch',sys.modules);EVIDENCE['contract_and_admission']=dict(prior_inputs=3364,all_current=True,source_model_sha256=MODEL_SHA,groups={k:len(v) for k,v in groups.items()},cached_rows_rederived=encoded,paired_starts=manifest['paired_starts'],whole_origin_family_bridge_independently_verified=True,validation_pool_families=100,validation_pool_unique_quotients=38,exact_batch_order_high16_low16_rehearsal32=True,unit_loss_only=True,no_validation_or_descendant_draws=True)

    def test_02_actual_label_pre_post_commit_kills_missing_export_and_corruption(self):
        from congklak import ordinary_value_pilot as p
        base=self.fresh('label-kills');fresh=base/'fresh';run_child(self.label_command(fresh),base/'fresh.log',timeout=600);rootmap,stores,manifest=p.load_labels(fresh);check_labels(self,rootmap,stores,manifest['identity']['settings']);records=[]
        for event,count in (('ordinary_value_h10_precommit',1),('ordinary_value_h10_committed',2)):
            path=base/event;probe=base/(event+'-probe');marker=kill_barrier(self.label_command(path),probe,event,env={'ORDINARY_VALUE_BLOCK_EVENT':event,'ORDINARY_VALUE_BLOCK_STEP':str(count),'ORDINARY_VALUE_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=count:v['completed']==n,timeout=600)
            before=rows(path/'work.sqlite3','h10_attempts','state');self.assertEqual(len(before),count);labels=rows(path/'work.sqlite3','h10_labels','state');backup(path/'work.sqlite3',probe/'validated-prefix.sqlite3')
            run_child(self.label_command(path),base/(event+'-resume.log'),timeout=600);r,s,m=p.load_labels(path);self.assertEqual({k:s['h10_attempts'][k] for k in before},before);self.assertEqual({k:s['h10_labels'][k] for k in labels},labels)
            for stage in ('h8','h10','h12'):
                self.assertEqual(set(stores[stage+'_labels']),set(s[stage+'_labels']))
                for key in s[stage+'_labels']:self.assertEqual(q_semantic(s[stage+'_labels'][key]['report']),q_semantic(stores[stage+'_labels'][key]['report']))
            closed=self.worker(dict(kind='closed',stage='labels',run=str(path),fixture=True),base/(event+'-closed'))
            records.append(dict(event=event,actual_process_kill=True,marker=marker,probe=relative(probe),run=relative(path),committed_prefix=count,prefix_exact=True,completed_proof=closed))
        # Missing export repairs only from a valid committed marker.
        repair=base/'missing-export';shutil.copytree(fresh,repair);(repair/'manifest.json').unlink()
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Repair searched')):p.run_labels(repair,fixture=True)
        self.assertEqual(load(repair/'manifest.json'),manifest);bad=base/'bad-marker';shutil.copytree(fresh,bad);(bad/'manifest.json').unlink();patch_sql(bad/'work.sqlite3','meta','complete',lambda body:body.update(roots=99))
        with self.assertRaises((ValueError,RuntimeError,KeyError)):p.run_labels(bad,fixture=True)
        self.assertFalse((bad/'manifest.json').exists())
        corrupt=base/'bad-history';shutil.copytree(fresh,corrupt);key=next(iter(rootmap));patch_sql(corrupt/'work.sqlite3','h10_attempts',key,lambda body:body['history'][0]['search_options'].update(use_pvs=False))
        with self.assertRaises((ValueError,RuntimeError,KeyError)):p.load_labels(corrupt)
        EVIDENCE['live_label_recovery']=dict(tests=records,actual_pre_and_post_H10_root_commit_kills=True,full_requested_fixture_budgets=True,missing_export_repaired_without_search=True,rehashed_bad_marker_refused_before_publication=True,corrupt_history_refused=True)

    def test_03_actual_checkpoint_kills_exact_state_rng_and_completed_no_work(self):
        base=self.fresh('training-kills');fresh=base/'fresh';run_child(self.train_command(fresh),base/'fresh.log',timeout=600);reference=pt_catalog(fresh);self.assertEqual([r[1] for r in reference],[2,4,6]);records=[]
        for event in ('checkpoint_file_ready','checkpoint_committed'):
            path=base/event;probe=base/(event+'-probe');marker=kill_barrier(self.train_command(path),probe,event,env={'ORDINARY_VALUE_BLOCK_EVENT':event,'ORDINARY_VALUE_BLOCK_STEP':'4','ORDINARY_VALUE_MARKER':str(probe/'declared-marker.json')},condition=lambda v:v['step']==4,timeout=600)
            prefix=pt_catalog(path);steps=[2] if event=='checkpoint_file_ready' else [2,4];self.assertEqual([r[1] for r in prefix],steps);backup(path/'catalog.sqlite3',probe/'validated-prefix.sqlite3');saved={r[2]:(sha(path/r[2]),(path/r[2]).stat().st_mtime_ns) for r in prefix}
            run_child(self.train_command(path),base/(event+'-resume.log'),timeout=600);actual=pt_catalog(path);self.assertEqual(actual[:len(prefix)],prefix)
            for name,value in saved.items():self.assertEqual((sha(path/name),(path/name).stat().st_mtime_ns),value)
            self.assertEqual(load(path/'model.json'),load(fresh/'model.json'));state=self.worker(dict(kind='state',left=str(fresh/reference[-1][2]),right=str(path/actual[-1][2])),base/(event+'-state'),torch=True)
            closed=self.worker(dict(kind='closed',stage='training',run=str(path),fixture=True),base/(event+'-closed'))
            records.append(dict(event=event,actual_process_kill=True,marker=marker,probe=relative(probe),run=relative(path),committed_prefix_steps=steps,resumed_step=6,prefix_bytes_mtimes_exact=True,state_proof=state,completed_proof=closed,orphan_checkpoint_retained=event=='checkpoint_file_ready'))
        EVIDENCE['live_training_recovery']=dict(tests=records,fixture_steps=6,batch_size=64,checkpoint_every=2,exact_full_model_head_optimizer_sampler_python_torch_rng=True)

    def test_04_all_production_requested_labels_actions_PVs_and_primary_coverage(self):
        from congklak import ordinary_value_pilot as p
        rootmap,stores,manifest=p.load_labels();registry=load(p.REGISTRY);self.assertEqual(rootmap,registry['rows']);self.assertEqual(Counter(row['stratum'] for row in rootmap.values()),{'50-74/ordinary':3,'25-49/ordinary':3})
        proof=check_labels(self,rootmap,stores,manifest['identity']['settings']);self.assertLessEqual(proof['new_nodes'],102000000);self.assertEqual(manifest['all_six_primary_H10_complete'],len(stores['h10_labels'])==6)
        for stage in ('h8','h10','h12'):self.assertEqual(manifest['summary'][stage]['missing_requested_reference'],sorted(set(rootmap)-set(stores[stage+'_labels'])))
        self.worker(dict(kind='closed',stage='labels',run=str(p.LABEL_DEFAULT)),self.base/'production-labels-closed');EVIDENCE['production_labels']=proof

    def test_05_all200_updates_ten_actual_checkpoints_and_fixed_policy_bits(self):
        from congklak import ordinary_value_pilot as p
        proof=self.worker(dict(kind='optimizer',run=str(p.DEFAULT)),self.base/'production-optimizer-audit',torch=True);self.assertEqual(proof['draws'],12800);self.assertEqual(proof['checkpoints'],10);self.assertTrue(proof['all_frozen_tensors_bit_exact']);p.witness()
        closed=self.worker(dict(kind='closed',stage='training',run=str(p.DEFAULT)),self.base/'production-training-closed');EVIDENCE['production_optimizer_audit']=proof;EVIDENCE['production_training_closed']=closed

    def test_06_actual_benchmark_decision_game_and_followup_decision_recovery(self):
        from congklak import ordinary_value_benchmark as b,ordinary_value_followup as f
        base=self.fresh('benchmark-kills');records=[]
        def command(path,out):return [sys.executable,str(ROOT/'tools/benchmark-ordinary-value.py'),'run','--fixture','--fixture-control','--run',str(path),'--output',str(out)]
        for event,count,table in (('ordinary_value_benchmark_decision_precommit',1,'decisions'),('ordinary_value_benchmark_decision_committed',2,'decisions'),('ordinary_value_game_precommit',0,'games'),('ordinary_value_game_committed',1,'games')):
            path=base/event;out=base/(event+'.json');probe=base/(event+'-probe');marker=kill_barrier(command(path,out),probe,event,env={'ORDINARY_VALUE_BLOCK_EVENT':event,'ORDINARY_VALUE_BLOCK_STEP':str(count),'ORDINARY_VALUE_BLOCK_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=count:v['completed']==n,timeout=600)
            backup(path/'work.sqlite3',probe/'validated-prefix.sqlite3');prefix=rows(path/'work.sqlite3','games','idx') if table=='games' else None
            with closing(dbopen(path/'work.sqlite3')) as db:decisionprefix=list(db.execute('SELECT game,choice,body,hash FROM decisions ORDER BY game,choice'));self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],count)
            run_child(command(path,out),base/(event+'-resume.log'),timeout=600)
            with closing(dbopen(path/'work.sqlite3')) as db:
                for game,choice,text,digest in decisionprefix:self.assertEqual(db.execute('SELECT body,hash FROM decisions WHERE game=? AND choice=?',(game,choice)).fetchone(),(text,digest))
            games=rows(path/'work.sqlite3','games','idx');self.assertEqual(len(games),8)
            if prefix:self.assertEqual({key:games[key] for key in prefix},prefix)
            choices=sum(literal_game(self,game) for game in games.values());closed=self.worker(dict(kind='closed',stage='benchmark',run=str(path),report=str(out),fixture=True),base/(event+'-closed'))
            records.append(dict(event=event,actual_process_kill=True,marker=marker,probe=relative(probe),run=relative(path),report=relative(out),prefix_table=table,committed_prefix=count,prefix_exact=True,resumed_terminal_games=8,literal_choices=choices,completed_proof=closed))
        EVIDENCE['live_benchmark_recovery']=dict(tests=records,uncommitted_timed_choices_need_not_match_fresh=True)
        # A rehashed completion marker must fail before a missing export repairs.
        corrupt=base/'rehashed-invalid-complete';shutil.copytree(path,corrupt);badout=base/'bad-missing-report.json'
        shutil.copyfile(out.with_name(out.stem+'-resources.json'),badout.with_name(badout.stem+'-resources.json'))
        patch_sql(corrupt/'work.sqlite3','meta','ordinary_value_benchmark_complete',lambda marker:marker.update(games=999))
        with self.assertRaises((ValueError,RuntimeError,KeyError)):b.run(path=corrupt,output=badout,fixture=True,fixture_control=True)
        self.assertFalse(badout.exists())
        follow=self.fresh('followup-kills');frecords=[]
        def followcommand(path):return [sys.executable,str(ROOT/'tools/ordinary-value-followup.py'),'run','--fixture','--run',str(path)]
        for event,count in (('ordinary_value_followup_decision_precommit',1),('ordinary_value_followup_decision_committed',2)):
            path=follow/event;probe=follow/(event+'-probe');marker=kill_barrier(followcommand(path),probe,event,env={'ORDINARY_VALUE_FOLLOWUP_BLOCK_EVENT':event,'ORDINARY_VALUE_FOLLOWUP_BLOCK_STEP':str(count),'ORDINARY_VALUE_FOLLOWUP_MARKER':str(probe/'declared-marker.json')},condition=lambda v,n=count:v['committed_decisions']==n,timeout=600)
            backup(path/'work.sqlite3',probe/'validated-prefix.sqlite3')
            with closing(dbopen(path/'work.sqlite3')) as db:prefix=list(db.execute('SELECT gameid,choice,body,hash FROM sp_decisions ORDER BY gameid,choice'));self.assertEqual(len(prefix),count)
            run_child(followcommand(path),follow/(event+'-resume.log'),timeout=600);m,data=f.load_run(path,fixture=True);games=rows(path/'work.sqlite3','sp_games','id');self.assertEqual(len(games),2)
            self.assertEqual(m['summary']['terminal_games'],2)
            with closing(dbopen(path/'work.sqlite3')) as db:
                for game,choice,text,digest in prefix:self.assertEqual(db.execute('SELECT body,hash FROM sp_decisions WHERE gameid=? AND choice=?',(game,choice)).fetchone(),(text,digest))
            for key,game in games.items():
                with closing(dbopen(path/'work.sqlite3')) as db:decisions=[json.loads(v[0]) for v in db.execute('SELECT body FROM sp_decisions WHERE gameid=? ORDER BY choice',(key,))]
                complete_game_literal(game,decisions)
            closed=self.worker(dict(kind='closed',stage='followup',run=str(path),fixture=True),follow/(event+'-closed'))
            frecords.append(dict(event=event,actual_process_kill=True,marker=marker,probe=relative(probe),run=relative(path),committed_prefix=count,prefix_exact=True,completed_proof=closed))
        EVIDENCE['live_followup_recovery']=dict(tests=frecords,fixture_explicit_first_original_high_root=True,fixture_simulations=4,production_simulations=64)

    def test_07_full_equal_time_arenas_guards_H10_errors_and_allocation(self):
        from congklak import ordinary_value_benchmark as b,ordinary_value_pilot as p
        from stage5_deeper_audit_acceptance import value_and_policy
        witness=b.closed_witness();plan=b.read_plan();run,out=b.paths();result=load(out);games=rows(run/'work.sqlite3','games','idx');cases=rows(run/'work.sqlite3','cases','idx');self.assertEqual((len(cases),len(games),result['scalar_measurements']),(38,96,304))
        self.assertEqual(plan['arena_ms'],25);self.assertEqual(plan['budgets_ms'],[25,100]);self.assertTrue(all(v['c_puct']==.25 and v['cache_capacity']==0 for v in plan['profiles'].values()))
        primary={};paired={};literal_choices=0;train=load(p.ROWS)['groups'];tq={quotient(r['state']) for group in train.values() for r in group};tf={r['family'] for group in train.values() for r in group};collisions=[];decision_count=0
        for index,game in games.items():
            literal_choices+=literal_game(self,game);q=norm(game['start'])
            with closing(dbopen(run/'work.sqlite3')) as db:records=[json.loads(row[0]) for row in db.execute('SELECT body FROM decisions WHERE game=? ORDER BY choice',(index,))]
            self.assertEqual(len(records),len(game['moves']))
            for choice,record in enumerate(records):
                self.assertEqual(record['state'],position(q));self.assertEqual(record['row'],game['decisions'][choice]);self.assertEqual(record['action'],game['moves'][choice]);b.validate_envelope(record['row']['decision'],position(q),25,plan,record['row']['engine'])
                if sum(map(sum,position(q)['pits']))>4 and (quotient(position(q)) in tq or game['family'] in tf):collisions.append((index,choice))
                q=literal.apply(q,record['action']).position;self.assertEqual(record['after'],position(q));decision_count+=1
        self.assertEqual(decision_count,result['decisions']);self.assertEqual(bool(collisions),result['trajectory_replay_overlap']['contaminated_evaluation']);self.assertEqual(len(collisions),result['trajectory_replay_overlap']['colliding_observations'])
        for comp in plan['arena_comparisons']:
            selected=[g for g in games.values() if g['comparison']==comp['id']];values=[];pairmap={}
            for pair in range(12):
                group=[g for g in selected if g['pair']==pair];self.assertEqual({g['candidate_seat'] for g in group},{0,1});value=statistics.mean(g['candidate_final_margin'] for g in group);values.append(value);pairmap[pair]=value
            paired[comp['id']]=pairmap;actual=result['arenas'][comp['id']];self.assertEqual(actual['primary_paired_mean_final_margin_seeds'],statistics.mean(values));ci=interval_reference(values,plan);self.assertEqual(actual['primary_margin_bootstrap_95_interval'],ci);primary[comp['id']]=values
            for field,score in (('candidate_wins',1),('candidate_draws',.5),('candidate_losses',0)):self.assertEqual(actual[field],sum(game['candidate_match_score']==score for game in selected))
            fallbacks=sum(sum(game['fallbacks'].values()) for game in selected);self.assertEqual(actual['terminal_games'],sum(game['status']=='terminal' for game in selected));self.assertEqual(actual['primary_strength_gate_passed'],not fallbacks and len(values)==12 and ci[0]>0)
        for base in b.BASELINES:
            values=[paired['ordinary_value_vs_'+base][i]-paired['ordinary_control_vs_'+base][i] for i in range(12)];contrast=result['secondary_factor_contrasts'][base];self.assertEqual(contrast['paired_mean_seed_margin_difference'],statistics.mean(values));self.assertEqual(contrast['confidence_interval'],interval_reference(values,plan))
        for name in b.CANDIDATES:self.assertEqual(result['both_baseline_primary_gates'][name],all(result['arenas'][name+'_vs_'+base]['primary_strength_gate_passed'] for base in b.BASELINES))
        measurements=0;guards=defaultdict(list)
        for index,case in cases.items():
            ref=plan['cases'][index]['reference'];state=plan['cases'][index]['state'];sign=1 if state['controller']==0 else -1;qvalues={r['pit']:r['value_player0'] for r in ref['action_scores']} if ref else {}
            for ms,engines in case['measurements'].items():
                for engine,value in engines.items():
                    b.validate_envelope(value['decision'],state,int(ms),plan,engine);chosen=value['decision']['decision']['chosen'];measurements+=1
                    self.assertEqual(value['reference_regret'],sign*(ref['target_player0']-qvalues[chosen]) if ref and chosen is not None else None);self.assertEqual(value['reference_optimal'],chosen in ref['best_moves'] if ref and chosen is not None else None)
                    if case['cohort']=='exact_guard':guards[ms,engine].append(value)
        self.assertEqual(measurements,304)
        for ms in ('25','100'):
            for engine in b.CANDIDATES:
                values=guards[ms,engine];actual=result['endgame_guard'][ms][engine];self.assertEqual(actual['roots'],32);self.assertEqual(actual['optimal_choices'],sum(v['reference_optimal'] is True for v in values));self.assertEqual(actual['mean_exact_regret'],statistics.mean(v['reference_regret'] for v in values))
        rootmap,stores,_=p.load_labels();models={'control':load(MODEL)['model'],'candidate':load(p.DEFAULT/'model.json')['model']};errors={k:[] for k in models}
        for row in result['validation_model_fit']['rows']:
            key=row['state_key'];target=stores['h10_labels'].get(key);self.assertEqual(row['reference_target_player0'],target['label']['target_player0'] if target else None);logits=[]
            for name,model in models.items():
                value,_,policy=value_and_policy(norm(rootmap[key]['state']),model);logits.append(policy);self.assertAlmostEqual(row['predictions'][name]['value_player0'],value,places=10)
                if target:errors[name].append(abs(value-target['label']['target_player0']));self.assertAlmostEqual(row['predictions'][name]['absolute_seed_error'],errors[name][-1],places=10)
                else:self.assertIsNone(row['predictions'][name]['absolute_seed_error'])
            self.assertEqual(*logits)
        for name in models:
            expected=statistics.mean(errors[name]) if errors[name] else None;actual=result['validation_model_fit'][name+'_mae']
            if expected is None:self.assertIsNone(actual)
            else:self.assertAlmostEqual(actual,expected,places=10)
        exact=all(result['endgame_guard'][ms]['non_regression_observed'] for ms in ('25','100'));engineering=result['terminal_zero_fallback_gate'] and exact and not collisions;fit=result['validation_model_fit'];gains={base:result['secondary_factor_contrasts'][base]['paired_mean_seed_margin_difference'] for base in b.BASELINES}
        favorable=engineering and len(stores['h10_labels'])==6 and fit['candidate_mae']<=fit['control_mae'] and all(v>=0 for v in gains.values());expected='hold' if not engineering else 'new_starts' if favorable else 'repeat_high';decision=b.allocation_decision();self.assertEqual(decision['branch'],expected)
        synthetic=copy.deepcopy(result);synthetic['validation_model_fit'].update(complete_reference_roots=5);self.assertEqual(b.allocation_from_report(synthetic,True)['branch'],'repeat_high' if engineering else 'hold');self.assertEqual(b.allocation_from_report(result,False)['branch'],'hold')
        EVIDENCE['production_benchmark']=dict(cases=38,scalar_measurements=304,games=96,literal_choices=literal_choices,all_saved_envelopes_current=True,primary=primary,all_four_primary_and_two_matched_intervals_independently_rederived=True,all_H10_intersection_errors_literal=True,actual_training_arena_collisions=collisions,allocation_branch=expected,allocation_decision=decision,no_promotion=True)

    def test_08_actual_followup_branch_terminal_targets_protection_and_completed_no_work(self):
        from congklak import ordinary_value_followup as f,ordinary_value_benchmark as b,ordinary_value_pilot as p
        manifest,data=f.load_run();identity=manifest['identity'];decision=b.allocation_decision();self.assertEqual(identity['branch'],decision['branch']);self.assertEqual(identity['model_sha256'],MODEL_SHA);self.assertFalse(manifest['training_performed']);self.assertFalse(manifest['future_training_authorized'])
        if identity['branch']=='hold':self.assertEqual(data,[]);self.assertEqual(manifest['summary']['games'],0)
        else:
            roles=identity['roles'];prior=load(ROOT/'data/ordinary-selfplay-pilot-v1/rollout-protection.json');projection=load(b.PROJECTION);qset=set(prior['quotients'])|set(projection['reserved_quotients']);fset=set(prior['families'])|set(projection['reserved_families']);arena=rows(b.paths()[0]/'work.sqlite3','games','idx');trajectory=0
            for game in arena.values():
                q=norm(game['start']);fset.add(game['family'])
                for action in [*game['moves'],None]:
                    qset.add(quotient(position(q)));trajectory+=1
                    if action is not None:q=literal.apply(q,action).position
            self.assertEqual(set(roles['true_evaluation_quotients']),qset);self.assertEqual(set(roles['true_evaluation_families']),fset);self.assertEqual(len(roles['latest96_trajectories']),96)
            roots=rows(f.DEFAULT/'work.sqlite3','roots','state');pool=rows(f.DEFAULT/'work.sqlite3','families','idx')
            if identity['branch']=='repeat_high':
                original=rows(ROOT/'data/ordinary-selfplay-pilot-v1/work.sqlite3','roots','state');self.assertEqual(roots,{k:r for k,r in original.items() if r['cohort']=='training_candidate' and r['stratum']=='50-74/ordinary'});self.assertEqual(len(roots),12);self.assertEqual(pool,{})
            else:
                self.assertEqual(len(pool),512)
                from congklak.puct_cpu_inference import CpuBatchEvaluator
                with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=1) as evaluator:
                    for i,family in pool.items():self.assertEqual(family,literal_family(i,identity['settings'],evaluator))
                components=components_reference(pool,set(roles['novelty_quotients']),set(roles['novelty_families']));blocked={family for c in components if c['reasons'] for family in c['families']};self.assertFalse({r['family'] for r in roots.values()}&blocked);self.assertLessEqual(len(roots),24);self.assertTrue(all(r['cohort']=='training_candidate' for r in roots.values()))
            games=rows(f.DEFAULT/'work.sqlite3','sp_games','id');self.assertEqual(len(games),2*len(roots));self.assertEqual(identity['settings']['rollouts']['simulations'],64);eligible={};excluded={};total=bonuses=0
            protect=f.explicit_starts(identity,pool)[2];pq,pf=set(protect['quotients']),set(protect['families'])
            byfamily=defaultdict(list);all_decisions={}
            for key,game in games.items():
                with closing(dbopen(f.DEFAULT/'work.sqlite3')) as db:decisions=[json.loads(v[0]) for v in db.execute('SELECT body FROM sp_decisions WHERE gameid=? ORDER BY choice',(key,))]
                bonuses+=complete_game_literal(game,decisions);total+=len(decisions);byfamily[game['family']].append(game);all_decisions[key]=decisions
                for d in decisions:
                    self.assertEqual(sum(d['search']['visits']),64);self.assertEqual(d['search']['completed_simulations'],64);self.assertEqual(d['policy_target'],[v/64 for v in d['search']['visits']])
                    state=d['state'];active,stratum,_=mechanism(state);legal=literal.legal_actions(norm(state))
                    if active>=25 and stratum.endswith('/ordinary') and len(legal)>=2:eligible[key,d['choice']]=d
            for family,group in byfamily.items():
                reasons=[]
                if any(g['status']!='terminal' for g in group):reasons.append('both_trajectories_not_terminal')
                if family in pf:reasons.append('protected_family')
                if any(quotient(d['state']) in pq for g in group for d in all_decisions[g['id']] if (g['id'],d['choice']) in eligible):reasons.append('protected_eligible_ordinary_quotient_bridge')
                if reasons:excluded[family]=sorted(set(reasons))
            self.assertEqual(manifest['summary']['family_exclusions'],excluded)
            for row in data:
                game=games[row['gameid']];d=eligible[row['gameid'],row['choice']];self.assertEqual(row['state'],d['state']);self.assertEqual(row['terminal_return_player0'],game['terminal_margin']);self.assertEqual(row['policy_target'],[v/64 for v in d['search']['visits']]);self.assertNotIn(row['family'],excluded);self.assertNotIn(quotient(row['state']),pq)
            historical=Counter(r['state_key'] for db in ('data/selfplay-training-pilot-v1/work.sqlite3','data/ordinary-selfplay-pilot-v1/work.sqlite3') for r in rows(ROOT/db,'sp_rows','id').values());new=Counter(r['state_key'] for r in data)
            for key,n in new.items():self.assertLessEqual(n+historical[key],4)
            candidates=defaultdict(list)
            for (gameid,choice),d in eligible.items():
                if games[gameid]['family'] not in excluded:candidates[raw_key(d['state'])].append(dict(id=gameid+'/'+f'{choice:03d}',choice=choice))
            expectedids=[];localpruned=[];crosspruned=[]
            for key,values in candidates.items():
                first=sorted(values,key=lambda row:(row['choice']!=0,hashlib.sha256(('ordinary-selfplay-rawcap-v1/'+row['id']).encode()).digest()));localpruned.extend(r['id'] for r in first[4:])
                second=sorted(first[:4],key=lambda row:(row['choice']!=0,hashlib.sha256(('ordinary-value-followup-cap/'+row['id']).encode()).digest()));room=max(0,4-historical[key]);expectedids.extend(r['id'] for r in second[:room]);crosspruned.extend(r['id'] for r in second[room:])
            self.assertEqual([r['id'] for r in data],sorted(expectedids));self.assertEqual(manifest['summary']['cross_batch_cap_pruned_ids'],sorted(crosspruned));self.assertEqual(manifest['summary']['pruned_row_ids'],sorted(localpruned+crosspruned))
            self.assertEqual(manifest['summary']['retained_rows'],len(data));self.assertEqual(manifest['summary']['unique_rawstates'],len(new))
            self.assertEqual(manifest['summary']['prior_counts_sha256'],config.digest(dict(sorted(historical.items()))))
            for row in data:self.assertEqual(row['qualified_id'],'ordinary-value-followup-v1/'+row['id']);self.assertEqual(row['dataset_namespace'],'ordinary-value-followup-v1')
            self.assertEqual(manifest['summary']['committed_decisions'],total);self.assertFalse(manifest['summary']['future_use_ready']);self.assertEqual(len({r['family'] for r in roots.values()}),len(roots))
            EVIDENCE['actual_followup_data']=dict(branch=identity['branch'],selected_starts=len(roots),games=len(games),literal_choices=total,bonus_choices=bonuses,retained_rows=len(data),all96_new_trajectories_protected=True,whole_eligible_family_bridges_rederived=True,cross_batch_rawcap4_independently_checked=True,new_training_performed=False)
        followclosed=self.worker(dict(kind='closed',stage='followup',run=str(f.DEFAULT)),self.base/'production-followup-closed');trainclosed=self.worker(dict(kind='closed',stage='training',run=str(p.DEFAULT)),self.base/'final-training-closed');benchclosed=self.worker(dict(kind='closed',stage='benchmark',run=str(b.paths()[0]),report=str(b.paths()[1])),self.base/'final-benchmark-closed')
        for name,digest in load(BASELINE)['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA);self.assertNotIn('torch',sys.modules);EVIDENCE['closed_final']=dict(prior3364_inputs_current=True,control_policy_fixed=True,completed_followup=followclosed,completed_training=trainclosed,completed_benchmark=benchclosed,no_followup_training_or_promotion=True)


GROUPS={'preflight':['test_01_frozen_sources_true_roles_independent_admission_and_sampler','test_02_actual_label_pre_post_commit_kills_missing_export_and_corruption','test_03_actual_checkpoint_kills_exact_state_rng_and_completed_no_work'],
    'labels':['test_04_all_production_requested_labels_actions_PVs_and_primary_coverage'],
    'model':['test_05_all200_updates_ten_actual_checkpoints_and_fixed_policy_bits'],
    'recovery':['test_06_actual_benchmark_decision_game_and_followup_decision_recovery'],
    'benchmark':['test_07_full_equal_time_arenas_guards_H10_errors_and_allocation'],
    'final':['test_08_actual_followup_branch_terminal_targets_protection_and_completed_no_work']}


def collect():
    inputs={};evidence={};groups=[];covered=[]
    for group,methods in GROUPS.items():
        path=WORK/(group+'.json');body=load(path);assert body['passed'] is True and body['tests_run']==len(methods) and body['covered_methods']==methods and not any(body[k] for k in ('errors','failures','skips'))
        for name,digest in body['inputs'].items():assert sha(ROOT/name)==digest,'Changed audit input '+name;assert name not in inputs or inputs[name]==digest;inputs[name]=digest
        assert not(set(evidence)&set(body['evidence']));evidence.update(body['evidence']);inputs[relative(path)]=sha(path);groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods));covered+=methods
    assert len(covered)==len(set(covered))==8
    result=dict(schema='ordinary-value-independent-audit-v1',passed=True,tests_run=8,errors=0,failures=0,skips=0,covered_methods=covered,groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/ordinary-value-audit.json'
    if output.exists():assert load(output)==result,'Immutable final audit differs'
    else:save(output,result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=[*GROUPS,'collect']);args=parser.parse_args()
    if args.group=='collect':collect();return 0
    frozen=load(FREEZE);assert frozen['schema']=='ordinary-value-independent-executed-source-v1' and frozen['groups']==GROUPS
    for name,digest in frozen['inputs'].items():assert sha(ROOT/name)==digest,'Independent executed source changed: '+name
    started=time.monotonic();methods=GROUPS[args.group];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(OrdinaryValueIndependentAudit(name) for name in methods))
    from congklak import ordinary_value_pilot as p,ordinary_value_benchmark as b,ordinary_value_followup as f
    files=[Path(__file__),ROOT/'tools/ordinary-value-audit.py',FREEZE,BASELINE,MODEL,p.DATA,p.ROWS,p.REGISTRY,p.PLAN,p.IMPLEMENTATION,b.PLAN,b.PROJECTION,b.IMPLEMENTATION,f.PLAN,f.IMPLEMENTATION]
    files += [ROOT/name for name in p.SOURCES+p.DEPENDENCIES+f.SOURCES+f.DEPENDENCIES]
    if b.IMPLEMENTATION.is_file():
        body=load(b.IMPLEMENTATION)
        for role in ('sources','dependencies','science_inputs'):files += [ROOT/name for name in body[role]]
    files += [file for file in OrdinaryValueIndependentAudit.base.rglob('*') if file.is_file() and not file.name.endswith(('-wal','-shm'))]
    if args.group in ('labels','model','recovery','benchmark','final'):files += [p.LABEL_OUTPUT,p.LABEL_DEFAULT/'manifest.json',p.LABEL_DEFAULT/'work.sqlite3']
    if args.group in ('model','recovery','benchmark','final'):files += [p.WITNESS]+[file for file in p.DEFAULT.rglob('*') if file.is_file() and not file.name.endswith(('-wal','-shm'))]
    if args.group in ('benchmark','final'):files += [b.paths()[1],b.paths()[1].with_name(b.paths()[1].stem+'-resources.json'),b.paths()[0]/'work.sqlite3']
    if args.group=='final':files += [f.OUTPUT]+[file for file in f.DEFAULT.rglob('*') if file.is_file() and not file.name.endswith(('-wal','-shm'))]
    body=dict(schema='ordinary-value-independent-group-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,duration_seconds=time.monotonic()-started,retained_directory=relative(OrdinaryValueIndependentAudit.base),inputs=pins([file for file in files if file.is_file()]),evidence=EVIDENCE,failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures])
    save(WORK/(args.group+'.json'),body);print(config.canonical(dict(event='ordinary_value_independent_audit',group=args.group,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if body['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
