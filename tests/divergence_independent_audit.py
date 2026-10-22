"""Independent fixed-work/action-Q audit, actual kills and immutable recovery."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import argparse,ast,copy,json,math,os,random,shutil,sqlite3,struct,subprocess,sys,threading,time,unittest,uuid
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,quality_search
from congklak.native import NativeError
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,mechanism,rows,run_child,snapshot
from stage5_deeper_audit_acceptance import literal_features,value_and_policy
from natural_study_independent_audit import check_pvs
import reference as literal
WORK=ROOT/'build/divergence-audit';BASELINE=ROOT/'build/divergence-baseline.json';PANEL=ROOT/'build/divergence-panel.json';FREEZE=ROOT/'build/divergence-audit-source-freeze.json';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json';CANDIDATE=ROOT/'data/ordinary-value-pilot-v1/model.json'
FROZEN={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'}


def raw(p):return bytes([*p['pits'][0],p['stores'][0],*p['pits'][1],p['stores'][1],p['controller']]).hex()
def flat(v):return [x for row in v for x in row]if isinstance(v[0],list)else v
def bits(v):
    v=flat(v);return struct.pack('<'+'f'*len(v),*v)
def semantic(raw):
    value=copy.deepcopy(raw);value.pop('timing');value['work'].pop('inference_ms');return value
def teacher_semantic(value):
    if isinstance(value,dict):return {k:teacher_semantic(v)for k,v in value.items()if k not in {'elapsed_ms','peak_process_bytes','label_sha256'}}
    if isinstance(value,list):return [teacher_semantic(v)for v in value]
    return value
def sql_rows(path,table,key):
    with closing(dbopen(path))as db:return [list(v)for v in db.execute('SELECT '+key+',body,hash FROM '+table+' ORDER BY '+key)]
def backup(source,dest):
    with closing(dbopen(source))as a,closing(sqlite3.connect(dest))as b:a.backup(b)
def patch_sql(path,table,key,column,mutate):
    with closing(sqlite3.connect(path))as db:
        text=db.execute('SELECT body FROM '+table+' WHERE '+column+'=?',(key,)).fetchone()[0];body=json.loads(text);mutate(body)
        with db:db.execute('UPDATE '+table+' SET body=?,hash=? WHERE '+column+'=?',(config.canonical(body),config.digest(body),key))
def kill(command,probe,event,completed,env):
    """Physical stdout-triggered kill; declared atomic marker may not exist yet."""
    probe.mkdir();marker=probe/'marker.json';done=threading.Event();lines=[]
    with(probe/'worker.log').open('w',encoding='utf-8')as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',env={**os.environ,**env})
        def reader():
            for line in child.stdout:
                stream.write(line);stream.flush();lines.append(line)
                try:body=json.loads(line)
                except ValueError:continue
                if isinstance(body,dict)and body.get('event')==event and body.get('completed')==completed:save(marker,body);done.set();return
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        try:
            deadline=time.monotonic()+180
            while not done.wait(.05)and child.poll()is None and time.monotonic()<deadline:pass
            if not marker.exists():raise AssertionError('Kill barrier missing: '+''.join(lines)[-6000:])
            assert child.poll()is None,'Child exited before physical kill';child.kill();child.wait(10);thread.join(5)
        finally:
            if child.poll()is None:child.kill();child.wait(10)
    return load(marker)


class DivergenceIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p
    def close(self,a,b):self.assertAlmostEqual(a,b,delta=1e-10)
    def current(self,inputs):
        for name,digest in inputs.items():self.assertEqual(sha(ROOT/name),digest,name)
    def worker(self,spec,path):
        path.mkdir();spec={**spec,'output':str(path/'proof.json')};save(path/'spec.json',spec);run_child([sys.executable,str(ROOT/'tools/divergence-audit.py'),str(path/'spec.json')],path/'worker.log',timeout=600);proof=load(path/'proof.json');self.assertTrue(proof['passed']);return dict(path=relative(path/'proof.json'),**proof)
    def command(self,kind,path):
        if kind=='search':return [sys.executable,str(ROOT/'tools/probe-divergence-search.py'),'run','--fixture','--path',str(path)]
        return [sys.executable,str(ROOT/'tools/label-divergence-actions.py'),'all','--fixture','--roots','2','--run',str(path)]
    def clone(self,source,name):
        dest=self.base/name;shutil.copytree(source,dest);return dest

    def test_01_prior3914_exact_panel_models_and_predeclared_budgets(self):
        from congklak import divergence_search as s,divergence_teacher as t,divergence_study_common as common
        base=load(BASELINE);self.assertEqual((base['inputs_count'],base['prior_accepted_inputs']),(3914,3911));self.assertEqual(len(base['inputs']),3914);self.current(base['inputs']);self.assertEqual(sha(MODEL),'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558')
        previous=load(ROOT/'build/ordinary-value-loss-review.json');selected=sorted([v for v in previous['seats']if v['baseline']=='H_v1'and not v['first_divergence']['identical_action_trajectory']],key=lambda v:(v['pair'],v['candidate_seat']));panel=load(PANEL);self.assertEqual([v['pair']for v in panel['roots']],[2,4,6,7,8]);self.assertEqual(len(selected),5)
        for saved,root in zip(selected,panel['roots']):
            d=saved['first_divergence'];self.assertEqual(root['state'],d['state']);self.assertEqual(root['state_key'],raw(d['state']));self.assertEqual(root['source_seat_sha256'],config.digest(saved));self.assertEqual(root['source_divergence_sha256'],config.digest(d));self.assertEqual(root['legal_actions'],list(literal.legal_actions(norm(root['state']))));self.assertEqual(root['legal_mask'],sum(1<<a for a in root['legal_actions']));self.assertEqual(root['actor'],1);self.assertEqual(mechanism(root['state'])[:2],(root['active_seeds'],root['stratum']));self.assertEqual(root['timed_margin_delta'],saved['candidate_minus_control_margin'])
            for name in('control','candidate'):self.assertEqual(root[name+'_action'],d[name]['saved_decision']['decision']['decision']['chosen']);self.assertEqual(root['source_'+name+'_decision_sha256'],d[name]['source']['decision_sha256'])
        self.assertEqual([v['active_seeds']for v in panel['roots']],[10,16,36,63,27]);self.assertEqual([len(v['legal_actions'])for v in panel['roots']],[4,5,4,6,5]);self.assertEqual([v['timed_margin_delta']for v in panel['roots']],[0,-6,-4,-6,0])
        a=load(MODEL)['model'];b=load(CANDIDATE)['model'];self.assertEqual(a['identity']['rules_digest'],config.RULES_DIGEST);self.assertEqual(b['identity']['rules_digest'],config.RULES_DIGEST)
        for name in FROZEN:self.assertEqual(bits(a['weights'][name]),bits(b['weights'][name]))
        self.assertEqual({k for k in a['weights']if bits(a['weights'][k])!=bits(b['weights'][k])},{'layers.4.weight','layers.4.bias'})
        plan,_,units,identity=s.admit();t.admit();self.assertEqual((len(units),sum(r['simulations']for r in units)),(40,19200));self.assertEqual(plan['simulations'],[128,256,512,1024]);self.assertEqual({k:identity[k]for k in('max_nodes','c_puct','time_ms','root_noise','rng','cache_capacity','max_batch','memory_mb')},dict(max_nodes=2048,c_puct=.25,time_ms=0,root_noise=None,rng=0,cache_capacity=0,max_batch=1,memory_mb=8192));self.assertEqual(identity['cpu_declared']['logical'],2)
        self.assertEqual({stage:t.CONTRACT[stage]for stage in t.STAGES},dict(h8=dict(depth=8,node_budget=2000000),h10=dict(depth=10,node_budget=5000000),h12=dict(depth=12,node_budget=10000000)));self.assertEqual(t.CONTRACT['maximum_new_nodes'],85000000)
        for root in panel['roots']:
            wrapper=t.panel()[root['state_key']];self.assertEqual(wrapper['root'],root);self.assertEqual(wrapper['source_root_sha256'],config.digest(root));self.assertEqual(wrapper['cohort'],'diagnostic')
        with mock.patch.object(s,'primary_witness',side_effect=ValueError('No independently audited fixed work')),mock.patch.object(t.Native,'analyze',side_effect=AssertionError('Teacher bypassed gate')):
            with self.assertRaises(ValueError):t.identity_for(t.settings_for(False),False)
        self.assertFalse(t.DEFAULT.exists());self.assertNotIn('torch',sys.modules)
        EVIDENCE['identity']=dict(prior3914_inputs_current=True,exact_five_roots=[2,4,6,7,8],all_actor1=True,source_model_and_policy_fixed=True,root_simulations19200=True,teacher_maximum_nodes85000000=True,no_training_or_promotion=True)

    def test_02_search_real_pre_post_commit_kills_prefix_semantics_and_refusal(self):
        from congklak import divergence_search as s
        fresh=self.base/'search-fresh';run_child(self.command('search',fresh),self.base/'search-fresh.log',timeout=180);roots,reference,manifest=s.load_cells(fresh,True);self.assertEqual((len(reference),manifest['counts']['completed_simulations']),(8,96));records=[]
        for event,count in(('divergence_search_cell_precommit',1),('divergence_search_cell_committed',2)):
            run=self.base/('search-'+str(count));probe=self.base/('search-kill-'+str(count));env=dict(DIVERGENCE_SEARCH_BLOCK_EVENT=event,DIVERGENCE_SEARCH_BLOCK_AFTER=str(count),DIVERGENCE_SEARCH_BLOCK_MARKER=str(probe/'declared-marker.json'));marker=kill(self.command('search',run),probe,event,count,env);backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3');prefix=sql_rows(probe/'validated-prefix.sqlite3','cells','idx');self.assertEqual(len(prefix),count)
            run_child(self.command('search',run),probe/'resume.log',timeout=180);rr,actual,_=s.load_cells(run,True);self.assertEqual(rr,roots);self.assertEqual(prefix,sql_rows(run/'work.sqlite3','cells','idx')[:count])
            self.assertEqual([dict(unit=c['unit'],status=c['status'],semantic=semantic(c['report']),counts=c['counts'])for c in actual],[dict(unit=c['unit'],status=c['status'],semantic=semantic(c['report']),counts=c['counts'])for c in reference]);records.append(dict(event=event,actual_process_kill=True,committed_prefix=count,marker=marker,probe=relative(probe),run=relative(run),resumed_cells=len(actual),committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        literalproof=self.worker(dict(kind='literal_cells',run=str(fresh),fixture=True,input_names=[relative(s.PLAN),relative(s.IMPLEMENTATION),relative(fresh/'work.sqlite3'),relative(fresh/'manifest.json')]),self.base/'search-fixture-literal')
        capped=s.run(self.base/'search-node-cap',fixture=True,variant='node_cap',on_event=lambda e:None);self.assertGreater(capped['counts']['capped_cells'],0);self.assertFalse(capped['scope_finished'])
        for c in capped['cells']:
            if c['status']=='budget_rejected':self.assertEqual(c['report']['stop_reason'],'node_budget');self.assertLess(c['report']['completed_simulations'],c['unit']['simulations']);self.assertEqual(c['report']['nodes'],2)
        bad=self.clone(fresh,'search-bad-marker');(bad/'result.json').unlink();patch_sql(bad/'work.sqlite3','meta','divergence_search_complete','key',lambda v:v.update(cells_sha256='0'*64))
        with self.assertRaises(ValueError):s.run(bad,fixture=True)
        self.assertFalse((bad/'result.json').exists())
        identitybad=self.clone(fresh,'search-bad-identity');patch_sql(identitybad/'work.sqlite3','meta','identity','key',lambda v:v.update(rng=False))
        with self.assertRaises(ValueError):s.verify(identitybad,True)
        partial=self.base/'search-cancelled';calls=0
        def cancel():
            nonlocal calls
            calls+=1;return calls>=9
        paused=s.run(partial,fixture=True,cancel=cancel,on_event=lambda e:None);self.assertEqual(paused['committed_cells'],0);self.assertEqual(paused['status'],'interrupted');a=rows(partial/'work.sqlite3','attempts','seq');self.assertEqual(len(a),1);self.assertEqual(a[0]['status'],'cancelled');self.assertEqual(rows(partial/'work.sqlite3','cells','idx'),{})
        s.run(partial,fixture=True,on_event=lambda e:None);self.assertEqual(s.verify(partial,True)['semantic_cells_sha256'],manifest['semantic_cells_sha256'])
        closed=self.worker(dict(kind='closed',producer='search',run=str(fresh),fixture=True),self.base/'search-fixture-closed');EVIDENCE['live_search_recovery']=dict(tests=records,fixture_cells=8,fixture_simulations96=True,literal_replay=literalproof,cap_or_cancel_not_complete=True,corrupt_identity_and_marker_refused=True,closed=closed)

    def test_03_teacher_real_pre_post_commit_kills_history_prefix_and_refusal(self):
        from congklak import divergence_teacher as t
        fresh=self.base/'teacher-fresh';run_child(self.command('teacher',fresh),self.base/'teacher-fresh.log',timeout=180);rootmap,reference,manifest=t.load_rows(fresh);self.assertEqual(len(rootmap),2);self.assertEqual([manifest['identity']['settings'][s]['depth']for s in t.STAGES],[2,3,4]);records=[]
        for event,count,within in(('divergence_teacher_h10_precommit',3,1),('divergence_teacher_h10_committed',4,2)):
            run=self.base/('teacher-'+str(count));probe=self.base/('teacher-kill-'+str(count));env=dict(DIVERGENCE_TEACHER_BLOCK_EVENT=event,DIVERGENCE_TEACHER_BLOCK_STEP=str(count),DIVERGENCE_TEACHER_MARKER=str(probe/'declared-marker.json'));marker=kill(self.command('teacher',run),probe,event,count,env);backup(run/'work.sqlite3',probe/'validated-prefix.sqlite3');prefix={table:sql_rows(probe/'validated-prefix.sqlite3',table,'state')for table in('roots',*t.TABLES)};self.assertEqual(len(prefix['h8_attempts']),2);self.assertEqual(len(prefix['h10_attempts']),within);self.assertEqual(len(prefix['h12_attempts']),0)
            run_child(self.command('teacher',run),probe/'resume.log',timeout=180);rr,actual,body=t.load_rows(run);self.assertEqual(rr,rootmap)
            for table,values in prefix.items():self.assertEqual(values,sql_rows(run/'work.sqlite3',table,'state')[:len(values)])
            self.assertEqual(teacher_semantic(actual),teacher_semantic(reference));records.append(dict(event=event,actual_process_kill=True,committed_prefix=count,stage_prefix=within,fixture_requested_depth=3,marker=marker,probe=relative(probe),run=relative(run),resumed_attempts=body['requested_attempts'],committed_prefix_bytes_preserved=True,fresh_semantics_equal=True))
        for stage in t.STAGES:
            for key,row in rootmap.items():
                attempt=reference[stage+'_attempts'][key];self.assertEqual([r['choice_depth']for r in attempt['history']],list(range(1,attempt['completed_depth']+1)))
                for report in attempt['history']:check_pvs(self,row,t.item_for(report,row),report['choice_depth'])
        bad=self.clone(fresh,'teacher-bad-history');key=next(iter(rootmap));patch_sql(bad/'work.sqlite3','h10_attempts',key,'state',lambda a:a['history'].pop(0))
        with self.assertRaises(ValueError):t.load_rows(bad)
        missing=self.clone(fresh,'teacher-bad-marker');(missing/'manifest.json').unlink();patch_sql(missing/'work.sqlite3','meta','complete','key',lambda v:v.update(requested_attempts=999))
        with self.assertRaises(ValueError):t.run(missing,fixture=True)
        self.assertFalse((missing/'manifest.json').exists())
        identitybad=self.clone(fresh,'teacher-bad-identity');patch_sql(identitybad/'work.sqlite3','meta','identity','key',lambda v:v.update(fixture=1))
        with self.assertRaises(ValueError):t.load_rows(identitybad)
        closed=self.worker(dict(kind='closed',producer='teacher',run=str(fresh),fixture=True),self.base/'teacher-fixture-closed');EVIDENCE['live_teacher_recovery']=dict(tests=records,old_consumed_fixture_roots=2,true_requested_fixture_depths=[2,3,4],production_labels_generated=False,rehash_history_identity_complete_refused=True,closed=closed)

    def test_04_all40_fixedwork_cells_literal_tree_and_teacher_gate(self):
        from congklak import divergence_search as s,divergence_teacher as t
        rootlist,cells,manifest=s.load_cells();plan,roots,units,identity=s.admit();self.assertEqual(rootlist,load(PANEL)['roots']);self.assertEqual(len(cells),40);self.assertEqual(manifest['counts']['complete_cells'],40);self.assertEqual(manifest['counts']['completed_simulations'],19200)
        proof=self.worker(dict(kind='literal_cells',run=str(s.DEFAULT),fixture=False,input_names=[relative(s.PLAN),relative(s.IMPLEMENTATION),relative(s.DEFAULT/'work.sqlite3'),relative(s.DEFAULT/'manifest.json'),relative(s.OUTPUT),relative(s.RESOURCES)]),self.base/'all40-literal-fixedwork');self.assertEqual((proof['cells'],proof['root_simulations']),(40,19200))
        for unit,cell in zip(units,cells):
            rawreport=cell['report'];self.assertEqual(cell['unit'],unit);self.assertEqual(cell['status'],'complete');self.assertEqual(rawreport['stop_reason'],'simulations');self.assertEqual(rawreport['completed_simulations'],unit['simulations']);self.assertEqual(sum(rawreport['visits']),unit['simulations']);self.assertLessEqual(rawreport['nodes'],2048);self.assertEqual(rawreport['timing']['requested_time_ms'],0);self.assertIsNone(identity['root_noise'])
            self.assertEqual(semantic(rawreport),cell['semantic_report']);self.assertEqual(config.digest(semantic(rawreport)),cell['semantic_sha256']);self.assertEqual(cell['counts']['native_actual_evals'],rawreport['work']['neural_evaluations']);self.assertEqual(cell['counts']['feature_deposits'],rawreport['work']['feature_deposits']);self.assertEqual(cell['counts']['cache_hits'],0);self.assertEqual(cell['counts']['cache_resets'],1)
            for action,value in enumerate(rawreport['action_mean_value_player0']):self.assertEqual(value is None,rawreport['visits'][action]==0)
        for comparison in manifest['comparisons']:
            selected=[c for c in cells if c['unit']['state_key']==comparison['state_key']and c['unit']['simulations']==comparison['simulations']];self.assertEqual(len(selected),2);self.assertEqual(selected[0]['report']['root_priors'],selected[1]['report']['root_priors']);self.assertEqual((comparison['eligible_models'],comparison['completed_models'],comparison['missing_models']),(2,2,0))
        witness=s.primary_witness();self.assertTrue(witness['passed']);self.assertFalse(witness['fixture']);self.assertEqual(witness['completed_simulations'],19200);self.current(load(WORK/'preflight.json')['inputs']);self.assertFalse(t.DEFAULT.exists())
        EVIDENCE['production_fixedwork']=dict(cells=40,completed_root_simulations=19200,literal_reference=proof,all_budget_visits_nodes_physical_work_verified=True,unvisited_Q_kept_None=True,global_policy_identity=True,primary_witness=witness,teacher_not_started_before_independent_stage04=True,auditor_replay_is_not_a_strength_match=True)

    def test_05_all15_requested_attempts_fullQ_histories_and_literal_PVs(self):
        from congklak import divergence_teacher as t
        rootmap,stores,manifest=t.load_rows();self.assertEqual(set(rootmap),{r['state_key']for r in load(PANEL)['roots']});self.assertEqual(manifest['requested_attempts'],15);self.assertEqual(manifest['roots'],5);histories=paths=choices=nodes=0;signatures=defaultdict(dict);counts={}
        for stage in t.STAGES:
            attempts=stores[stage+'_attempts'];labels=stores[stage+'_labels'];settings=manifest['identity']['settings'][stage];self.assertEqual(list(attempts),list(rootmap));self.assertLessEqual(set(labels),set(attempts));counts[stage]=dict(eligible=5,complete=len(labels),missing=5-len(labels))
            for key,a in attempts.items():
                row=rootmap[key];self.assertEqual(a['study_row_sha256'],config.digest(row));self.assertEqual(a['source_root_sha256'],config.digest(row['root']));self.assertEqual(a['requested_depth'],settings['depth']);self.assertEqual(a['stage'],stage);self.assertEqual([r['choice_depth']for r in a['history']],list(range(1,a['completed_depth']+1)));self.assertGreater(a['work']['nodes'],0);self.assertLessEqual(a['work']['nodes'],settings['node_budget']);nodes+=a['work']['nodes']
                previous={k:0 for k in a['work']}
                for report in a['history']:
                    depth=report['choice_depth'];self.assertEqual((report['search_options']['tt_mb'],report['search_options']['use_pvs']),(16,True));item=t.item_for(report,row);n,m=check_pvs(self,row,item,depth);paths+=n;choices+=m;histories+=1
                    for name,value in previous.items():self.assertGreaterEqual(report[name],value);self.assertLessEqual(report[name],a['work'][name])
                    previous={name:report[name]for name in previous};label=item['label'];sig=dict(target=label['target_player0'],q=label['action_scores'],best=label['best_moves'],coverage=label['coverage'])
                    if depth in signatures[key]:self.assertEqual(signatures[key][depth],sig)
                    signatures[key][depth]=sig
                if a['status']=='complete':
                    self.assertEqual(a['completed_depth'],settings['depth']);self.assertEqual(a['stop_reason'],'max_depth');self.assertEqual(labels[key],t.item_for(a['history'][-1],row));self.assertEqual(a['label_sha256'],config.digest(labels[key]));self.assertEqual(set(r['pit']for r in labels[key]['label']['action_scores']),set(row['root']['legal_actions']))
                else:self.assertEqual(a['status'],'budget_rejected');self.assertEqual(a['stop_reason'],'node_budget');self.assertEqual(a['work']['nodes'],settings['node_budget']);self.assertLess(a['completed_depth'],settings['depth']);self.assertNotIn(key,labels);self.assertIsNone(a['label_sha256'])
            self.assertEqual(manifest['summary'][stage]['complete'],len(labels));self.assertEqual(manifest['summary'][stage]['missing_requested_reference'],sorted(set(rootmap)-set(labels)))
        self.assertEqual(nodes,manifest['total_new_nodes']);self.assertLessEqual(nodes,85000000);self.assertEqual(histories,manifest['auxiliary_completed_reports']);self.assertFalse(manifest['auxiliary_horizon_fallback']);self.assertTrue(manifest['diagnostic_only']);self.assertFalse(manifest['training_performed']);self.assertFalse(manifest['torch_imported'])
        gate=manifest['identity']['fixed_simulation_witness'];self.assertEqual(set(gate['independent']),{'preflight','search'})
        for group,record in gate['independent'].items():self.assertEqual(record['sha256'],sha(ROOT/record['path']));self.current(load(ROOT/record['path'])['inputs'])
        EVIDENCE['production_teacher']=dict(attempts=15,roots=5,requested_depths=[8,10,12],counts=counts,all_complete_iteration_reports=histories,literal_PV_paths=paths,literal_PV_choices=choices,actual_new_nodes=nodes,ceiling=85000000,all_legal_Q_actor_bonus_and_report_links_verified=True,auxiliary_history_never_promoted=True,H10_primary_finite_reference_not_truth=True)

    def test_06_cross_model_budget_horizon_regret_and_comparison_denominators(self):
        from congklak import divergence_search as s,divergence_teacher as t,divergence_comparison as c
        output=load(c.OUTPUT);roots,cells,sm=s.load_cells();rowmap,stores,tm=t.load_rows();cellmap={(x['unit']['state_key'],x['unit']['simulations'],x['unit']['profile']):x for x in cells};index={r['state_key']:r for r in output['roots']};references={};shifts=[]
        def regret(action,q,actor):return (max if actor==0 else min)(q.values())-q[action]if actor==0 else q[action]-min(q.values())
        self.assertEqual(len(index),5);self.assertEqual(output['primary_horizon'],'h10');self.assertEqual(output['fixed_cells'],40);self.assertEqual(output['completed_root_simulations'],19200);self.assertEqual(output['teacher_attempts'],15);self.assertEqual(output['teacher_total_nodes'],tm['total_new_nodes']);self.current(output['inputs'])
        for root in roots:
            key=root['state_key'];saved=index[key];actor=root['actor'];self.assertEqual((saved['actor'],saved['legal_actions'],saved['timed_control_action'],saved['timed_candidate_action'],saved['timed_margin_delta']),(actor,root['legal_actions'],root['control_action'],root['candidate_action'],root['timed_margin_delta']));horizonmap={}
            for h in saved['horizons']:
                stage=h['stage'];a=stores[stage+'_attempts'][key];item=stores[stage+'_labels'].get(key);q={r['pit']:r['value_player0']for r in item['label']['action_scores']}if item else None;horizonmap[stage]=q
                self.assertEqual((h['status'],h['requested_depth'],h['completed_depth'],h['nodes']),(a['status'],a['requested_depth'],a['completed_depth'],a['work']['nodes']));self.assertEqual(h['retained_completed_depths'],list(range(1,a['completed_depth']+1)))
                self.assertEqual(h['requested_action_Q_player0'],[dict(pit=a,value_player0=q[a])for a in sorted(q)]if q is not None else None);self.assertEqual(h['best_moves'],sorted(a for a in q if regret(a,q,actor)==0)if q is not None else None);self.assertEqual(h['missing_actions'],0 if q is not None else len(root['legal_actions']))
                for profile in('control','candidate'):self.assertEqual(h['timed_'+profile+'_regret'],regret(root[profile+'_action'],q,actor)if q is not None else None)
            self.assertEqual(set(horizonmap),set(t.STAGES));self.assertEqual([f['simulations']for f in saved['fixed_work']],[128,256,512,1024])
            for fixed in saved['fixed_work']:
                budget=fixed['simulations'];chosen={}
                for profile in('control','candidate'):
                    cell=cellmap[key,budget,profile];r=cell['report'];model=fixed['models'][profile];chosen[profile]=r['chosen'];self.assertEqual(model,dict(chosen=r['chosen'],visits=r['visits'],root_priors=r['root_priors'],sampled_action_Q_player0=r['action_mean_value_player0'],parent_bootstrap_player0=r['unvisited_root_q_player0'],completed_simulations=r['completed_simulations'],physical_forwards=cell['counts']['native_actual_evals'],matches_saved_timed_action=r['chosen']==root[profile+'_action']))
                self.assertEqual(fixed['same_action'],chosen['control']==chosen['candidate'])
                for ref in fixed['references']:
                    q=horizonmap[ref['stage']];cr=regret(chosen['control'],q,actor)if q is not None else None;vr=regret(chosen['candidate'],q,actor)if q is not None else None;expected=dict(stage=ref['stage'],requested_depth=int(ref['stage'][1:]),eligible=q is not None,control_regret=cr,candidate_regret=vr,candidate_minus_control_regret=vr-cr if q is not None else None,control_in_best_set=cr==0 if q is not None else None,candidate_in_best_set=vr==0 if q is not None else None);self.assertEqual(ref,expected);references[key,budget,ref['stage']]=expected
            for left,right in zip(t.STAGES,t.STAGES[1:]):
                a,b=horizonmap[left],horizonmap[right];eligible=a is not None and b is not None;abest={x for x in a if regret(x,a,actor)==0}if a is not None else None;bbest={x for x in b if regret(x,b,actor)==0}if b is not None else None;shifts.append(dict(state_key=key,left=left,right=right,eligible=eligible,best_sets_disjoint=not bool(abest&bbest)if eligible else None))
        self.assertEqual(output['horizon_shifts'],shifts);self.assertEqual(len(output['metrics']),12)
        for metric in output['metrics']:
            stage,budget=metric['stage'],metric['simulations'];valid=[references[r['state_key'],budget,stage]for r in roots if references[r['state_key'],budget,stage]['eligible']];different=sum(cellmap[r['state_key'],budget,'control']['report']['chosen']!=cellmap[r['state_key'],budget,'candidate']['report']['chosen']for r in roots)
            expected=dict(stage=stage,simulations=budget,requested_roots=5,eligible_roots=len(valid),missing_roots=5-len(valid),control_regret_sum=sum(r['control_regret']for r in valid),candidate_regret_sum=sum(r['candidate_regret']for r in valid),control_best_count=sum(r['control_in_best_set']for r in valid),candidate_best_count=sum(r['candidate_in_best_set']for r in valid),candidate_lower_regret=sum(r['candidate_minus_control_regret']<0 for r in valid),same_regret=sum(r['candidate_minus_control_regret']==0 for r in valid),candidate_higher_regret=sum(r['candidate_minus_control_regret']>0 for r in valid),different_actions=different,different_actions_denominator=5);self.assertEqual(metric,expected)
        self.assertFalse(output['training_performed']);self.assertFalse(output['promotion']);self.assertEqual(output['previous_allocation'],'hold');EVIDENCE['comparison']=dict(roots=5,fixedwork_pairs=20,budget_horizon_metrics=12,horizon_shifts=10,all_eligible_missing_regret_and_bestset_denominators_verified=True,actor1_minimum_Q_correct=True,no_favorable_budget_or_horizon_selection=True,no_Q_transform_or_auxiliary_fallback=True,no_causal_game_or_strength_claim=True)

    def test_07_rehashed_production_corruption_and_missing_output_refusal(self):
        from congklak import divergence_search as s,divergence_teacher as t,divergence_comparison as c
        roots,cells,manifest=s.load_cells();plan,_,units,identity=s.admit()
        with closing(dbopen(s.DEFAULT/'work.sqlite3'))as db:setup=s._hashrow(*db.execute('SELECT body,hash FROM setup WHERE id=0').fetchone())
        cases=[]
        for name,mutate in(('unit-float',lambda v:v['unit'].update(simulations=float(v['unit']['simulations']))),('visit-work',lambda v:v['report']['visits'].__setitem__(v['report']['chosen'],v['report']['visits'][v['report']['chosen']]+1))):
            bad=copy.deepcopy(cells[0]);mutate(bad);bad['report_sha256']=config.digest(bad['report']);bad['semantic_report']=semantic(bad['report']);bad['semantic_sha256']=config.digest(bad['semantic_report']);save(self.base/('search-corrupt-'+name+'.json'),bad)
            with self.assertRaises((ValueError,NativeError,AssertionError)):s.validate_attempt(bad,0,units[0],roots[0],identity,setup)
            cases.append(name)
        oldmeta=s._meta;badidentity=copy.deepcopy(identity);badidentity['rng']=False;save(self.base/'search-rehashed-identity.json',badidentity)
        with mock.patch.object(s,'_meta',side_effect=lambda db,key:badidentity if key=='identity'else oldmeta(db,key)):
            with self.assertRaises(ValueError):s.verify()
        rootmap,stores,tm=t.load_rows();key=next(iter(rootmap));attempt=stores['h10_attempts'][key];label=stores['h10_labels'].get(key)
        for name,mutate in(('history-prefix',lambda a:a['history'].pop(0)),('pvs-option',lambda a:a['history'][0]['search_options'].update(use_pvs=False)),('root-source',lambda a:a.update(source_root_sha256='0'*64))):
            bad=copy.deepcopy(attempt);mutate(bad);save(self.base/('teacher-corrupt-'+name+'.json'),bad)
            with self.assertRaises(ValueError):t.validate_attempt(rootmap[key],bad,label,'h10',tm['identity']['settings'],t.Native())
            cases.append(name)
        original=load(c.OUTPUT);read=c.common.read
        for name in('metric','source'):
            bad=copy.deepcopy(original)
            if name=='metric':bad['metrics'][0]['candidate_regret_sum']+=1
            else:bad['inputs'][relative(PANEL)]='0'*64
            save(self.base/('comparison-rehashed-'+name+'.json'),dict(body=bad,rehash=config.digest(bad)))
            with mock.patch.object(c.common,'read',side_effect=lambda path,body=bad:body if Path(path).resolve()==c.OUTPUT.resolve()else read(path)):
                with self.assertRaises(ValueError):c.verify()
            cases.append('comparison-'+name)
        EVIDENCE['production_corruption_refusal']=dict(cases=cases,all_hashes_recomputed_or_payload_rehashed=True,all_refused=True,production_files_never_modified=True,prepublication_missing_export_guards_covered_by_preflight_fixtures=True)

    def test_08_closed_API_CLI_no_work_source_immutability_and_final_preservation(self):
        from congklak import divergence_search as s,divergence_teacher as t,divergence_comparison as c
        searchclosed=self.worker(dict(kind='closed',producer='search',run=str(s.DEFAULT),extra_files=[relative(s.OUTPUT),relative(s.RESOURCES)]),self.base/'production-search-closed');teacherclosed=self.worker(dict(kind='closed',producer='teacher',run=str(t.DEFAULT),extra_files=[relative(t.OUTPUT)]),self.base/'production-teacher-closed')
        extra=[relative(c.OUTPUT),relative(s.OUTPUT),relative(s.RESOURCES),relative(t.OUTPUT)]+[relative(p)for p in t.DEFAULT.rglob('*')if p.is_file()and not p.name.endswith(('-wal','-shm'))]
        comparisonclosed=self.worker(dict(kind='closed',producer='comparison',run=str(s.DEFAULT),extra_files=extra,watch_baseline=True),self.base/'production-comparison-closed')
        oldtree=ast.parse((ROOT/'congklak/ordinary_value_review_progress.py').read_text());newtree=ast.parse((ROOT/'congklak/divergence_progress.py').read_text());functions=('event_message','run_command','save','runs')
        for name in functions:
            old=next(x for x in oldtree.body if isinstance(x,ast.FunctionDef)and x.name==name);new=next(x for x in newtree.body if isinstance(x,ast.FunctionDef)and x.name==name);self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        logger=load(ROOT/'build/ordinary-value-review-progress-author.json');self.assertTrue(logger['passed']);self.assertEqual(logger['tests_run'],2);self.current(logger['inputs']);self.current(load(BASELINE)['inputs']);self.assertNotIn('torch',sys.modules)
        EVIDENCE['final_no_work']=dict(search=searchclosed,teacher=teacherclosed,comparison=comparisonclosed,all3914_prior_inputs_current=True,copied_logger_four_functions_AST_identical_to_sealed_two_physical_child_tests=True,new_logger_tests_recounted=False,all_models_and_policy_fixed=True,previous_allocation_hold=True,new_training_or_strength_games=False)


GROUPS={'preflight':['test_01_prior3914_exact_panel_models_and_predeclared_budgets','test_02_search_real_pre_post_commit_kills_prefix_semantics_and_refusal','test_03_teacher_real_pre_post_commit_kills_history_prefix_and_refusal'],
 'search':['test_04_all40_fixedwork_cells_literal_tree_and_teacher_gate'],
 'labels':['test_05_all15_requested_attempts_fullQ_histories_and_literal_PVs','test_06_cross_model_budget_horizon_regret_and_comparison_denominators'],
 'final':['test_07_rehashed_production_corruption_and_missing_output_refusal','test_08_closed_API_CLI_no_work_source_immutability_and_final_preservation']}


def source_inputs():
    from congklak import divergence_search as s,divergence_teacher as t,divergence_comparison as c
    names={relative(Path(__file__)),'tools/divergence-audit.py',relative(BASELINE),relative(PANEL),'docs/DIVERGENCE_STUDY_SCOPE.md','tests/data_pilot_independent_audit.py','tests/reference.py','tests/selfplay_pilot_independent_audit.py','tests/quality_audit_acceptance.py','tests/stage5_deeper_audit_acceptance.py','tests/stage5_audit_acceptance.py','tests/natural_study_independent_audit.py','congklak/divergence_progress.py'}
    inputs=dict(load(BASELINE)['inputs']);names.update(t.SOURCES+t.DEPENDENCIES+c.SOURCES);names.update((relative(t.REGISTRY),relative(t.PLAN),relative(t.IMPLEMENTATION),relative(s.PLAN),relative(s.IMPLEMENTATION),relative(c.IMPLEMENTATION)))
    for role in('sources','science_inputs','dependencies'):names.update(load(s.IMPLEMENTATION)[role])
    inputs.update({name:sha(ROOT/name)for name in names});return dict(sorted(inputs.items()))


def collect():
    inputs={};evidence={};covered=[];groups=[]
    for name,methods in GROUPS.items():
        path=WORK/(name+'.json');value=load(path);assert value['passed']is True and value['covered_methods']==methods and value['tests_run']==len(methods)and not any(value[k]for k in('errors','failures','skips'))
        for key,digest in value['inputs'].items():assert sha(ROOT/key)==digest,key;assert key not in inputs or inputs[key]==digest;inputs[key]=digest
        assert not(set(evidence)&set(value['evidence']));evidence.update(value['evidence']);covered.extend(methods);groups.append(dict(path=relative(path),sha256=sha(path),tests_run=len(methods),covered_methods=methods));inputs[relative(path)]=sha(path)
    assert len(covered)==len(set(covered))==8
    body=dict(schema='divergence-independent-audit-v1',passed=True,tests_run=8,errors=0,failures=0,skips=0,covered_methods=covered,covered_checks=['__main__.DivergenceIndependentAudit.'+name for name in covered],groups=groups,inputs=inputs,evidence=evidence)
    output=ROOT/'build/divergence-audit.json'
    if output.exists():assert load(output)==body,'Immutable audit aggregate differs'
    else:save(output,body)
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['freeze',*GROUPS,'collect']);args=parser.parse_args()
    if args.operation=='freeze':
        assert not FREEZE.exists(),'Audit source already frozen';save(FREEZE,dict(schema='divergence-independent-executed-source-v1',groups=GROUPS,inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=load(FREEZE);assert frozen['groups']==GROUPS
    for name,digest in frozen['inputs'].items():assert sha(ROOT/name)==digest,'Executed source/input changed: '+name
    output=WORK/(args.operation+'.json');assert not output.exists(),'Immutable group already exists';started=time.monotonic();methods=GROUPS[args.operation];result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(DivergenceIndependentAudit(name)for name in methods))
    from congklak import divergence_search as s,divergence_teacher as t,divergence_comparison as c
    inputs={**frozen['inputs'],relative(FREEZE):sha(FREEZE)};files=[p for p in DivergenceIndependentAudit.base.rglob('*')if p.is_file()and not p.name.endswith(('-wal','-shm'))]
    if args.operation in('search','labels','final'):files +=[s.OUTPUT,s.RESOURCES]+[p for p in s.DEFAULT.rglob('*')if p.is_file()and not p.name.endswith(('-wal','-shm'))]
    if args.operation in('labels','final'):files +=[t.OUTPUT,c.OUTPUT]+[p for p in t.DEFAULT.rglob('*')if p.is_file()and not p.name.endswith(('-wal','-shm'))]
    if args.operation in('labels','final'):files +=[WORK/'preflight.json',WORK/'search.json']
    if args.operation=='final':files +=[WORK/'labels.json']
    inputs.update(pins(files));body=dict(schema='divergence-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,covered_checks=['__main__.DivergenceIndependentAudit.'+name for name in methods],duration_seconds=time.monotonic()-started,retained_directory=relative(DivergenceIndependentAudit.base),inputs=inputs,evidence=EVIDENCE,failure_details=[dict(method=str(test),trace=trace)for test,trace in result.errors+result.failures]);save(output,body);print(config.canonical(dict(event='divergence_independent_group',group=args.operation,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if body['passed']else 1
if __name__=='__main__':raise SystemExit(main())
