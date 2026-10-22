"""Independent reviewed-data, fixed-policy value pilot and equal-time audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import statistics
import struct
import sys
import time
import unittest
from unittest import mock
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native,flatten
from data_pilot_independent_audit import sha,load,save,relative,pins,dbopen,norm,position,key_of,quotient,rows,semantic,kill_barrier,run_child,snapshot
from quality_audit_acceptance import module
from horizon_probe_independent_audit import item_view
import reference as literal
WORK=ROOT/'build/value-pilot-audit';EVIDENCE={}
MODEL=ROOT/'data/quality-train-v1/model.json'
MODEL_SHA='bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558'
TRAINABLE={'layers.4.weight','layers.4.bias'}
FROZEN={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'}


def pt_catalog(path):
    with closing(dbopen(path/'catalog.sqlite3')) as db:return list(db.execute('SELECT id,step,name,sha,body,hash FROM checkpoints ORDER BY id'))


def literal_game(test,game):
    q=norm(game['start']);counts=Counter();steps=0
    test.assertEqual(len(game['moves']),len(game['decisions']))
    for action,decision in zip(game['moves'],game['decisions']):
        test.assertEqual(decision['actor'],q.controller);test.assertIn(action,literal.legal_actions(q))
        expected=game['comparison'].split('_vs_')[0] if q.controller==game['candidate_seat'] else game['comparison'].split('_vs_')[1]
        test.assertEqual(decision['engine'],expected)
        if decision['fallback']:counts[expected]+=1
        else:test.assertEqual(action,decision['decision']['decision']['chosen'])
        q=literal.apply(q,action).position;steps+=1
    test.assertEqual(position(q),game['final']);terminal=literal.normalize(q).terminal
    test.assertEqual(game['status'],'terminal' if terminal else 'unfinished');margin=literal.evaluate(q)*(1 if game['candidate_seat']==0 else -1)
    test.assertEqual(game['candidate_final_margin'],margin if terminal else None)
    test.assertEqual(game['candidate_match_score'],(1 if margin>0 else 0 if margin<0 else .5) if terminal else None);test.assertEqual(game['fallbacks'],dict(counts))
    return steps


def tensor_equal(torch,a,b,where='state'):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape,where
        assert torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)),where
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),where
        for key in a:tensor_equal(torch,a[key],b[key],where+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):tensor_equal(torch,x,y,where+'.'+str(i))
    else:assert type(a) is type(b) and a==b,where


def state_worker(spec):
    import torch
    left=torch.load(spec['left'],map_location='cpu',weights_only=True);right=torch.load(spec['right'],map_location='cpu',weights_only=True)
    tensor_equal(torch,left,right);save(Path(spec['output']),dict(passed=True,all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal=True,step=left['step'],draws=left['draws']));return 0


def completed_training_worker(spec):
    from congklak import value_pilot_learning as p
    path=Path(spec['run']);before=snapshot(path)
    with mock.patch.object(p,'runtime',side_effect=AssertionError('Completed run imported Torch')):
        value=p.train(path,fixture=spec.get('fixture',False))
        verified=p.verify_run(path,fixture=spec.get('fixture',False))
    assert value['status']=='complete' and verified['passed'] and 'torch' not in sys.modules and snapshot(path)==before
    save(Path(spec['output']),dict(passed=True,zero_optimizer_steps=True,torch_imported=False,files_bytes_mtimes_unchanged=True));return 0


def optimizer_worker(spec):
    """Fresh independent CPU operator replay against every actual production PT."""
    import torch
    from torch import nn
    from congklak import value_pilot_learning as producer
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    run=Path(spec['run']);export=load(run/'model.json')['model'];identity=export['identity'];source=load(MODEL)['model']['weights']
    torch.manual_seed(identity['seed']);random.seed(identity['seed'])
    class Network(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.Sequential(nn.Linear(127,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh());self.policy=nn.Linear(64,7)
    model=Network();expected_torch_rng=torch.get_rng_state().clone();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in source.items()})
    for name,p in model.named_parameters():p.requires_grad_(name in TRAINABLE)
    data=load(ROOT/'build/value-pilot-rows.json')['groups'];hidden={};inputs={};cache=dict(source_model_sha256=MODEL_SHA,frozen_weights_sha256=identity['initial_frozen_weights_sha256'],rows_sha256={},features_sha256={},hidden_sha256={},policy_sha256={})
    def digest(values):
        h=hashlib.sha256()
        for name,value in sorted(values.items()):h.update(name.encode());h.update(bytes(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
        return h.hexdigest()
    with torch.no_grad():
        for split,group in data.items():
            x=torch.tensor([r['cached_features']['x'] for r in group],dtype=torch.float32);h=model.layers[3](model.layers[2](model.layers[1](model.layers[0](x))));hidden[split]=h
            inputs[split]={name:torch.tensor([r['cached_features'][name] for r in group],dtype=torch.float32) for name in ('d','r')};inputs[split]['y']=torch.tensor([r['label']['target_player0'] for r in group],dtype=torch.float32)
            cache['rows_sha256'][split]=config.digest([r['state_key'] for r in group]);cache['features_sha256'][split]=digest({'x':x});cache['hidden_sha256'][split]=digest({'h':h});cache['policy_sha256'][split]=digest({'logits':model.policy(h)})
    assert cache==identity['hidden_cache'] and config.digest(cache)==identity['hidden_cache_sha256']
    def descriptive():
        result={}
        with torch.no_grad():
            for split,group in data.items():
                if split=='train':continue
                value=inputs[split]['d']+inputs[split]['r']*model.layers[5](model.layers[4](hidden[split])).squeeze(-1)
                error=(value-inputs[split]['y']).abs().tolist();huber=torch.nn.functional.huber_loss(value,inputs[split]['y'],delta=4.,reduction='none').tolist();strata=defaultdict(list)
                for i,r in enumerate(group):strata[str(r['teacher_depth'])+'/'+r['stratum']].append(i)
                mean=lambda ids:dict(rows=len(ids),mae_seeds=math.fsum(error[i] for i in ids)/len(ids),seed_huber_delta4=math.fsum(huber[i] for i in ids)/len(ids))
                result[split]={**mean(list(range(len(group)))),'strata':{k:mean(v) for k,v in sorted(strata.items())},'scope':'Descriptive finite-horizon development targets; fixed final step'}
        return result
    initial_descriptive=descriptive()
    optimizer=torch.optim.AdamW([model.layers[4].weight,model.layers[4].bias],lr=.0003,weight_decay=0.,foreach=False)
    cumulative=[];total=0.
    for row in data['train']:total+=row['sampling_probability'];cumulative.append(total)
    sampler=random.Random(identity['seed']);entries=pt_catalog(run);assert [r[1] for r in entries]==list(range(20,201,20));by_step={r[1]:r for r in entries};verified=[];paths=[];draws=0
    for step in range(1,201):
        ids=torch.tensor([bisect.bisect_right(cumulative,sampler.random()*cumulative[-1]) for _ in range(256)],dtype=torch.int64);draws+=256
        optimizer.zero_grad(set_to_none=True);z=model.layers[5](model.layers[4](hidden['train'][ids])).squeeze(-1);prediction=inputs['train']['d'][ids]+inputs['train']['r'][ids]*z
        loss=torch.nn.functional.huber_loss(prediction,inputs['train']['y'][ids],delta=4.);loss.backward();torch.nn.utils.clip_grad_norm_([model.layers[4].weight,model.layers[4].bias],1.);optimizer.step()
        if step not in by_step:continue
        row=by_step[step];_,_,name,checksum,text,bodyhash=row;file=run/name;meta=json.loads(text)
        assert sha(file)==checksum and config.digest(meta)==bodyhash and meta['step']==step and meta['identity_sha256']==config.digest(identity)
        actual=torch.load(file,map_location='cpu',weights_only=True);assert actual['identity']==identity and actual['step']==step and actual['draws']==draws
        expected={'identity':identity,'step':step,'draws':draws,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampler_rng':sampler.getstate(),'python_rng':random.Random(identity['seed']).getstate(),'torch_rng':expected_torch_rng,'cuda_rng':[],'scheduler':None,'amp_scaler':None,'pending_batches':[]}
        tensor_equal(torch,actual,expected,'checkpoint'+str(step));assert set(actual['model'])==TRAINABLE|FROZEN and set(actual['optimizer']['state'])=={0,1}
        for name in FROZEN:tensor_equal(torch,actual['model'][name],torch.tensor(source[name],dtype=torch.float32),'frozen'+name)
        assert digest(actual['model'])==meta['weights_sha256'];verified.append(dict(step=step,weights_sha256=meta['weights_sha256']));paths.append(file)
    last=actual;assert export['step']==200 and export['weights_sha256']==digest(last['model']) and all(export['weights'][k]==v.tolist() for k,v in last['model'].items())
    result=load(run/'result.json');assert result['initial_descriptive']==initial_descriptive and result['descriptive']==descriptive()
    changed=sorted(k for k,v in last['model'].items() if source[k]!=v.tolist());assert set(changed)==TRAINABLE
    # Admission must reject all frozen tensor mutations and incorrect optimizer ownership.
    accepted_model=producer.evaluator();producer.validate_state(last,identity,accepted_model)
    rejected=[]
    for name in sorted(FROZEN):
        bad=copy.deepcopy(last);bad['model'][name].view(-1)[0]+=.25
        try:producer.validate_state(bad,identity,accepted_model)
        except (ValueError,RuntimeError,KeyError):rejected.append(name)
        else:raise AssertionError('Frozen mutation admitted '+name)
    for kind in ('draws','ids','moment_shape','weight_decay','sampler'):
        bad=copy.deepcopy(last)
        if kind=='draws':bad['draws']+=1
        elif kind=='ids':bad['optimizer']['state'][2]=copy.deepcopy(bad['optimizer']['state'][0])
        elif kind=='moment_shape':bad['optimizer']['state'][0]['exp_avg']=torch.zeros(64)
        elif kind=='weight_decay':bad['optimizer']['param_groups'][0]['weight_decay']=.01
        else:bad['sampler_rng']=random.Random(999).getstate()
        try:producer.validate_state(bad,identity,accepted_model)
        except (ValueError,RuntimeError,KeyError):rejected.append(kind)
        else:raise AssertionError('Corrupt optimizer admitted '+kind)
    save(Path(spec['output']),dict(passed=True,checkpoints=10,steps=verified,all200_optimizer_steps_independently_replayed=True,all_checkpoint_model_optimizer_sampler_python_torch_rng_raw_bytes_equal=True,
        all_frozen_tensors_bit_exact=True,frozen_parameters=12807,trainable_parameters=65,optimizer_parameter_ids=[0,1],changed_tensors=changed,draws=draws,hidden_cache_independently_reconstructed=True,initial_and_final_all_cohort_metrics_independently_rederived=True,final_export_equals_PT=True,checkpoint_corruptions_refused=rejected,inputs=pins(paths+[run/'model.json',run/'catalog.sqlite3',MODEL,ROOT/'build/value-pilot-rows.json'])));return 0


class ValuePilotIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):WORK.mkdir(parents=True,exist_ok=True);cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir()
    def fresh(self,name):p=self.base/name;p.mkdir();return p

    def test_01_all_prior_inputs_and_fixed_model_contract(self):
        b=load(ROOT/'build/value-pilot-baseline.json');self.assertEqual((b['inputs_count'],b['prior_accepted_inputs']),(2426,2424));self.assertEqual(len(b['inputs']),2426)
        for name,digest in b['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA);EVIDENCE['baseline']=dict(prior_inputs=2426,old_accepted_inputs=2424,all_hashes_current=True,fixed_model_sha256=MODEL_SHA)

    def test_02_independent_ownership_targets_probability_and_source_reconstruction(self):
        from congklak import value_pilot_data as p,features_relay
        manifest=load(p.MANIFEST);dataset=load(p.ROWS);groups=dataset['groups'];train=groups['train'];base=[r for r in train if r['pool']=='base'];ordinary=[r for r in train if r['pool']=='ordinary'];review=load(p.REVIEW)
        self.assertEqual((len(train),len(base),len(ordinary)),(7624,7563,61));self.assertEqual({k:len(v) for k,v in groups.items()},dict(train=7624,development=1000,ordinary_development=14,diagnostic=8));self.assertEqual(len({r['family'] for r in ordinary}),59)
        old=ROOT/'data/deeper-9000-v2/work.sqlite3';oldroots=rows(old,'roots','state');oldlabels=rows(old,'labels','state');oldfeatures=rows(old,'features','state')
        pilot=ROOT/'data/ordinary-data-pilot-v1/work.sqlite3';natural=ROOT/'data/natural-study-v1/work.sqlite3';finish=ROOT/'data/horizon-finish-v1/work.sqlite3'
        original=rows(pilot,'roots','state');evidence=defaultdict(dict)
        for depth,db,table in ((8,pilot,'labels'),(8,natural,'h8_labels'),(10,natural,'h10_labels'),(12,finish,'h12_references'),(14,finish,'prior_h14_labels'),(14,finish,'tier1_labels'),(14,finish,'tier2_labels')):
            for key,item in rows(db,table,'state').items():
                if depth in evidence[key]:self.assertEqual(evidence[key][depth],item)
                evidence[key][depth]=item
        for key,wrapper in rows(natural,'roots','state').items():
            if key in original:self.assertEqual(original[key],wrapper['root'])
            original[key]=wrapper['root']
        self.assertEqual(Counter(r['cohort'] for r in original.values()),dict(training_candidate=61,validation=14,diagnostic=8))
        projection=load(ROOT/'build/value-pilot-benchmark-projection.json');oldregistry=load(ROOT/'build/value-registry.json');protectedq=set(oldregistry['quotients'])|set(projection['quotients']);protectedf=set(oldregistry['families'])|set(projection['families']);gamecount=choices=0
        with closing(dbopen(ROOT/'data/value-benchmark-v1/work.sqlite3')) as db:
            for text,digest in db.execute('SELECT body,hash FROM games ORDER BY idx'):
                game=json.loads(text);self.assertEqual(config.digest(game),digest);q=norm(game['start']);protectedf.add(game['family'])
                for action in [*game['moves'],None]:
                    protectedq.add(quotient(position(q)))
                    if action is not None:q=literal.apply(q,action).position;choices+=1
                self.assertEqual(position(q),game['final']);gamecount+=1
        self.assertEqual(gamecount,192)
        dev=[r for r in oldroots.values() if r['split']=='development']
        for r in [*dev,*[r for r in original.values() if r['cohort']!='training_candidate']]:protectedf.add(r['family']);protectedq.add(quotient(r['state']))
        oldkeys=set(load(ROOT/'build/value-data.json')['train_state_keys']);self.assertEqual(len(oldkeys),7592);candidates=[oldroots[k] for k in oldkeys]+[r for r in original.values() if r['cohort']=='training_candidate'];bridge={r['family'] for r in candidates if quotient(r['state']) in protectedq}
        allowed=lambda r:r['family'] not in protectedf|bridge and quotient(r['state']) not in protectedq
        expected=sorted(k for k in oldkeys if allowed(oldroots[k]));self.assertEqual([r['state_key'] for r in base],expected);self.assertEqual(manifest['protected_families'],sorted(protectedf|bridge));self.assertEqual(manifest['protected_quotients'],sorted(protectedq))
        for r in base:
            key=r['state_key'];source=oldroots[key];self.assertEqual(r['state'],source['state']);self.assertEqual((r['family'],r['stratum'],r['legal_mask']),(source['family'],source['stratum'],source['legal_mask']));self.assertEqual(r['label'],oldlabels[key]['label']);self.assertEqual(r['cached_features'],oldfeatures[key]);self.assertEqual((r['teacher_depth'],r['loss_weight']),(8,1.))
        ns=Counter(r['stratum'] for r in base)
        for r in base:self.assertEqual(r['sampling_probability'],.95*(.9/len(base)+.1/(len(ns)*ns[r['stratum']])))
        self.assertEqual([r['state_key'] for r in ordinary],sorted(k for k,r in original.items() if r['cohort']=='training_candidate' and allowed(r)));factors=[];depthcounts=Counter();native=Native()
        for r in [*ordinary,*groups['ordinary_development'],*groups['diagnostic']]:
            key=r['state_key'];source=original[key];ev=evidence[key];depth=max(ev);item=ev[depth];self.assertEqual(r['state'],source['state']);self.assertEqual((r['family'],r['cohort'],r['legal_mask']),(source['family'],source['cohort'],source['legal_mask']));self.assertEqual(r['original_root_sha256'],config.digest(source));self.assertEqual(r['label_item_sha256'],config.digest(item));self.assertEqual((r['teacher_depth'],r['label']),(depth,item['label']))
            self.assertEqual(r['target_evidence'],[dict(depth=d,item_sha256=config.digest(v),target_player0=v['label']['target_player0'],best_moves=v['label']['best_moves']) for d,v in sorted(ev.items())]);self.assertLessEqual(set(ev),{8,10,12,14})
            tail=sorted(ev)[-3:];values=[ev[d]['label']['target_player0'] for d in tail];span=max(values)-min(values);overlap=bool(set(ev[tail[-2]]['label']['best_moves'])&set(ev[tail[-1]]['label']['best_moves'])) if len(tail)>1 else None;volatility=1/(1+span/4) if len(tail)>1 else .5;disagreement=.5 if overlap is False else 1.;factor=volatility*disagreement
            self.assertEqual(r['exposure_review'],dict(requested_depths_available=sorted(ev),depths_used=tail,targets_player0=values,span_seeds=span,last_two_best_sets_overlap=overlap,volatility_factor=volatility,disagreement_factor=disagreement,exposure_factor=factor))
            x,d,scale=features_relay.encode(r['state'],127,native);self.assertEqual(r['cached_features'],dict(x=x,d=d,r=scale));self.assertEqual(item['root_sha256'],config.digest(source));self.assertEqual(r['label']['target_player0'],item['report']['depth_result']['value_player0']);self.assertEqual(r['label']['action_scores'],[dict(pit=c['pit'],value_player0=c['depth_result']['value_player0']) for c in item['report']['candidates']])
            if r['cohort']=='training_candidate':factors.append(factor);depthcounts[depth]+=1
        self.assertEqual(depthcounts,{8:32,10:24,14:5});z=math.fsum(factors)
        for r,factor in zip(ordinary,factors):self.assertEqual(r['sampling_probability'],.05*factor/z)
        self.assertAlmostEqual(math.fsum(r['sampling_probability'] for r in base),.95);self.assertAlmostEqual(math.fsum(r['sampling_probability'] for r in ordinary),.05);self.assertTrue(all(r['loss_weight']==1. for r in train));self.assertEqual(len({r['state_key'] for r in train}),len(train))
        for name in ('development','ordinary_development','diagnostic'):
            self.assertFalse({r['family'] for r in train}&{r['family'] for r in groups[name]});self.assertFalse({quotient(r['state']) for r in train}&{quotient(r['state']) for r in groups[name]})
        probability=[r['sampling_probability'] for r in train];family=defaultdict(float)
        for r in train:family[r['family']]+=r['sampling_probability']
        self.assertAlmostEqual(review['summary']['root_effective_support'],1/math.fsum(v*v for v in probability));self.assertAlmostEqual(review['summary']['family_effective_support'],1/math.fsum(v*v for v in family.values()));self.assertFalse(manifest['sealed_final_test_consumed']);self.assertTrue(manifest['no_teacher_searches'])
        EVIDENCE['independent_data_review']=dict(train_rows=7624,base_rows=7563,new_ordinary_rows=61,new_ordinary_families=59,new_depths=dict(depthcounts),development_rows=1000,new_validation_rows=14,diagnostic_rows=8,excluded_replay_rows=29,protected_old_games=192,literal_protection_choices=choices,protected_quotients=len(protectedq),protected_families=len(protectedf|bridge),all_original_labels_and_followup_replacements_exact=True,requested_depth_only_uncertainty_inputs=True,all_probabilities_factors_and_effective_support_rederived=True,base_mass=.95,new_mass=.05,unit_loss_weights=True,no_validation_or_diagnostic_optimizer_draws=True,ancestral_exposure_not_unlearned=True)

    def test_03_predeclared_schedule_and_complete_pretraining_control(self):
        from congklak import value_pilot_benchmark as b
        full=load(b.PLAN);prior=load(ROOT/'build/value-plan.json');projection=load(b.PROJECTION)
        def select(rows,tag):return [r for kind in ('ordinary','capture','bonus') for r in sorted([r for r in rows if r['stratum'].endswith('/'+kind)],key=lambda r:hashlib.sha256((tag+'|'+r['family']+'|'+r['state_key']).encode()).hexdigest())[:4]]
        self.assertEqual(projection['cases'],select(prior['cases'][:48],'value-pilot-diagnostics-v1')+prior['cases'][48:]);self.assertEqual(projection['arena_starts'],select(prior['arena_starts'],'value-pilot-arena-v1'));self.assertEqual(full['cases'],projection['cases']);self.assertEqual(full['arena_starts'],projection['arena_starts'])
        self.assertEqual(full['profiles'],dict(policy_control=dict(model='control',c_puct=.25,cache_capacity=0),value_pilot=dict(model='pilot',c_puct=.25,cache_capacity=0)));self.assertEqual((full['arena_ms'],full['budgets_ms'],full['tt_mb'],full['use_pvs'],full['max_depth']),(25,[25,100],16,True,16));self.assertFalse(full['default_promotion']);self.assertFalse(full['sealed_final_test_consumed'])
        for stage,profiles,games,scalars in [('pretraining',['policy_control'],48,264),('posttraining',['policy_control','value_pilot'],96,352)]:
            plan=b.read_plan(stage);comparisons=[dict(id=c+'_vs_'+base,candidate=c,opponent=base,pairs=12) for c in profiles for base in ('H_v1','H8_relay_CPU')];self.assertEqual(plan['arena_comparisons'],comparisons);order=[];names=[c['id'] for c in comparisons]
            for pair in range(12):
                for name in names[pair%len(names):]+names[:pair%len(names)]:
                    for seat in ((0,1) if pair%2==0 else (1,0)):order.append(dict(comparison=name,pair=pair,seat=seat))
            self.assertEqual(plan['arena_work_order'],order);self.assertEqual(len(order),games);self.assertEqual(len(plan['cases'])*2*len(plan['engines']),scalars)
        with mock.patch.object(Native,'analyze',side_effect=AssertionError('Witness searched teacher')):witness=b.pretraining_witness()
        self.assertTrue(witness['passed']);self.assertEqual((witness['games'],witness['cases'],witness['scalar_measurements']),(48,44,264))
        from congklak import value_pilot_learning as learning
        self.assertEqual((learning.CONTRACT['steps'],learning.CONTRACT['batch_size'],learning.CONTRACT['checkpoint_every'],learning.CONTRACT['seed'],learning.CONTRACT['lr']),(200,256,20,20261022,.0003))
        EVIDENCE['frozen_schedule']=dict(families=12,reused_development=True,cases_per_stage=44,pretraining_games=48,posttraining_games=96,total_games=144,total_scalar_measurements=616,pretraining_control_completed_before_training=True,paired_seats_interleaved_order_independently_rederived=True,same_c025_uncached_whole_clock=True,fixed_final_step=200,no_selection_or_promotion=True)

    def test_04_actual_CPU_checkpoint_kills_exact_optimizer_rng_and_completed_no_work(self):
        runtime=load(ROOT/'build/training-runtime.json')['python'];base=self.fresh('training-kills')
        def command(path):return [runtime,str(ROOT/'tools/value-pilot-train.py'),'train','--fixture','--run',str(path)]
        fresh=base/'fresh';run_child(command(fresh),base/'fresh.log',timeout=300);freshrows=pt_catalog(fresh);self.assertEqual([r[1] for r in freshrows],[2,4,6]);records=[]
        for event in ('checkpoint_file_ready','checkpoint_committed'):
            path=base/event;probe=base/(event+'-probe')
            marker=kill_barrier(command(path),probe,event,env={'VALUE_PILOT_BLOCK_EVENT':event,'VALUE_PILOT_BLOCK_STEP':'4','VALUE_PILOT_MARKER':str(probe/'declared-marker.json')},condition=lambda value:value['step']==4,timeout=300)
            before=pt_catalog(path);expected=[2] if event=='checkpoint_file_ready' else [2,4];self.assertEqual([r[1] for r in before],expected);prefix={r[2]:(sha(path/r[2]),(path/r[2]).stat().st_mtime_ns) for r in before}
            with closing(dbopen(path/'catalog.sqlite3')) as source,closing(sqlite3.connect(probe/'validated-prefix.sqlite3')) as target:source.backup(target)
            run_child(command(path),base/(event+'-resume.log'),timeout=300);actual=pt_catalog(path);self.assertEqual(actual[:len(before)],before)
            for name,digest_mtime in prefix.items():self.assertEqual((sha(path/name),(path/name).stat().st_mtime_ns),digest_mtime)
            self.assertEqual(load(path/'model.json'),load(fresh/'model.json'))
            output=base/(event+'-exact-state.json');spec=base/(event+'-state-spec.json');save(spec,dict(left=str(fresh/freshrows[-1][2]),right=str(path/actual[-1][2]),output=str(output)))
            run_child([runtime,str(Path(__file__)),'worker','state','--spec',str(spec)],base/(event+'-state.log'),timeout=300);self.assertTrue(load(output)['passed'])
            complete_output=base/(event+'-completed.json');complete_spec=base/(event+'-completed-spec.json');save(complete_spec,dict(run=str(path),fixture=True,output=str(complete_output)))
            run_child([sys.executable,str(Path(__file__)),'worker','completed-training','--spec',str(complete_spec)],base/(event+'-completed.log'),timeout=300);self.assertTrue(load(complete_output)['passed'])
            records.append(dict(event=event,actual_process_kill=True,marker=marker,committed_prefix_steps=expected,resumed_step=6,prefix_bytes_mtimes_exact=True,all_model_optimizer_sampler_python_torch_rng_raw_bytes_equal=True,completed_no_torch_no_optimizer_work=True,completed_bytes_mtimes_unchanged=True,orphan_checkpoint_retained=event=='checkpoint_file_ready'))
        EVIDENCE['live_training_recovery']=dict(runtime=runtime,device='cpu',fixture_steps=6,batch_size=32,checkpoint_every=2,tests=records,actual_new_value_training_killed=True,raw_markers_catalog_backups_and_tensor_comparison_proofs_retained=True)

    def test_05_every_production_optimizer_step_checkpoint_and_global_fixed_policy(self):
        from congklak import value_pilot_learning as p,value_pilot_benchmark as b
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        from stage5_deeper_audit_acceptance import value_and_policy
        runtime=load(ROOT/'build/training-runtime.json')['python'];output=self.base/'actual-production-tensors.json';spec=self.base/'actual-production-spec.json';save(spec,dict(run=str(p.DEFAULT),output=str(output)))
        run_child([runtime,str(Path(__file__)),'worker','optimizer','--spec',str(spec)],self.base/'actual-production-tensors.log',timeout=300);proof=load(output);self.assertTrue(proof['passed']);self.assertEqual(proof['checkpoints'],10)
        with mock.patch.object(p,'runtime',side_effect=AssertionError('CPU witness imported Torch')):witness=p.witness();self.assertTrue(witness['passed'])
        control=load(MODEL)['model'];candidate=load(p.DEFAULT/'model.json')['model'];self.assertEqual(control['encoder'],candidate['encoder']);self.assertEqual(set(control['weights']),TRAINABLE|FROZEN)
        def raw(value):
            if isinstance(value,list):return b''.join(raw(v) for v in value)
            return struct.pack('<f',value)
        for name in FROZEN:self.assertEqual(raw(control['weights'][name]),raw(candidate['weights'][name]))
        self.assertEqual(sum(len(raw(control['weights'][k]))//4 for k in FROZEN),12807)
        groups=load(ROOT/'build/value-pilot-rows.json')['groups'];roots=[r['state'] for r in b.read_plan()['cases']]+[r['state'] for name in ('ordinary_development','diagnostic') for r in groups[name]]+[dict(pits=[[0]*7,[1,2,0,0,0,0,1]],stores=[47,47],controller=0),dict(pits=[[0]*7,[0]*7],stores=[40,58],controller=1)];checked=0
        with CpuBatchEvaluator(MODEL,cache_capacity=0,max_batch=32) as left,CpuBatchEvaluator(p.DEFAULT/'model.json',cache_capacity=0,max_batch=32) as right:
            for offset in range(0,len(roots),32):
                batch=roots[offset:offset+32]
                for state,x,y in zip(batch,left.evaluate_many(batch),right.evaluate_many(batch)):
                    self.assertEqual(x['policy_logits'],y['policy_logits']);self.assertEqual((x['feature_simulations'],x['feature_deposits']),(y['feature_simulations'],y['feature_deposits']))
                    for prediction,model in ((x,control),(y,candidate)):
                        value,_,logits=value_and_policy(norm(state),model);self.assertAlmostEqual(prediction['value_player0'],value,delta=1e-10);self.assertLessEqual(max(abs(a-b) for a,b in zip(prediction['policy_logits'],logits)),1e-10)
                    checked+=1
        EVIDENCE['production_optimizer_audit']=proof
        EVIDENCE['global_policy_proof']=dict(frozen_tensors=sorted(FROZEN),frozen_parameters=12807,trainable_parameters=65,changed_tensors=proof['changed_tensors'],native_positions=checked,literal_forwards=2*checked,policy_logits_bit_exact=True,global_policy_unchanged_by_identical_encoder_and_frozen_subgraph=True,CPU_witness_no_Torch=True)

    def test_06_actual_game_pre_postcommit_recovery_deadline_and_corruption_refusals(self):
        from congklak import value_pilot_benchmark as b
        from congklak.puct_cpu_inference import CpuBatchEvaluator
        base=self.fresh('game-kills');plan=b.read_plan('posttraining',True);records=[]
        def command(path,output):return [sys.executable,str(ROOT/'tools/benchmark-value-pilot.py'),'run','--stage','posttraining','--fixture','--run',str(path),'--output',str(output)]
        for suffix,prefix in (('precommit',0),('committed',1)):
            event='value_pilot_game_'+suffix;path=base/suffix;output=base/(suffix+'-result.json');probe=base/(suffix+'-probe')
            marker=kill_barrier(command(path,output),probe,event,env={'VALUE_PILOT_BLOCK_EVENT':event,'VALUE_PILOT_BLOCK_STEP':str(prefix),'VALUE_PILOT_BLOCK_MARKER':str(probe/'declared-marker.json')},condition=lambda value,n=prefix:value['completed']==n,timeout=300)
            with closing(dbopen(path/'work.sqlite3')) as db:
                before=list(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'));self.assertEqual(len(before),prefix)
                with closing(sqlite3.connect(probe/'validated-prefix.sqlite3')) as target:db.backup(target)
            run_child(command(path,output),base/(suffix+'-resume.log'),timeout=300);choices=0
            with closing(dbopen(path/'work.sqlite3')) as db:
                actual=list(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'));self.assertEqual(len(actual),8);self.assertEqual(actual[:len(before)],before)
                for _,_,_,_,text,digest in actual:
                    game=json.loads(text);self.assertEqual(config.digest(game),digest);choices+=literal_game(self,game)
                b.validate_saved_units(db,plan,Native())
            stable=snapshot(path);export=(sha(output),output.stat().st_mtime_ns);resource=output.with_name(output.stem+'-resources.json');resource_before=(sha(resource),resource.stat().st_mtime_ns)
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed benchmark searched')),mock.patch.object(b,'puct_search',side_effect=AssertionError('Completed benchmark searched PUCT')),mock.patch.object(b,'NeuralNative',side_effect=AssertionError('Completed benchmark allocated relay')):
                b.run('posttraining',path,output,fixture=True,on_event=lambda _:None)
            self.assertEqual(snapshot(path),stable);self.assertEqual((sha(output),output.stat().st_mtime_ns),export);self.assertEqual((sha(resource),resource.stat().st_mtime_ns),resource_before)
            valid=base/(suffix+'-valid.sqlite3')
            with closing(dbopen(path/'work.sqlite3')) as db,closing(sqlite3.connect(valid)) as target:db.backup(target)
            # Invalid rehashed completion cannot repair a missing publication.
            broken=base/(suffix+'-broken');shutil.copytree(path,broken);badout=base/(suffix+'-missing.json');shutil.copy2(resource,badout.with_name(badout.stem+'-resources.json'))
            with closing(sqlite3.connect(broken/'work.sqlite3')) as db:
                marker_body=json.loads(db.execute("SELECT body FROM meta WHERE key='value_pilot_benchmark_complete'").fetchone()[0]);marker_body['games']+=1
                db.execute("UPDATE meta SET body=?,hash=? WHERE key='value_pilot_benchmark_complete'",(config.canonical(marker_body),config.digest(marker_body)));db.commit()
            self.assertNotEqual(run_child(command(broken,badout),base/(suffix+'-refused-marker.log'),success=False,timeout=300),0);self.assertFalse(badout.exists())
            # Repair only the missing valid report, without allocating a search evaluator.
            oldbytes=output.read_bytes();output.unlink()
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Repair searched')),mock.patch.object(b,'puct_search',side_effect=AssertionError('Repair searched PUCT')),mock.patch.object(b,'NeuralNative',side_effect=AssertionError('Repair allocated relay')):b.run('posttraining',path,output,fixture=True,on_event=lambda _:None)
            self.assertEqual(output.read_bytes(),oldbytes);self.assertEqual(snapshot(path),stable)
            with closing(sqlite3.connect(path/'work.sqlite3')) as db:
                index,name,pair,seat,text,digest=actual[0];bad=json.loads(text);bad['candidate_final_margin']=(bad['candidate_final_margin'] or 0)+2;db.execute('UPDATE games SET body=?,hash=? WHERE idx=?',(config.canonical(bad),config.digest(bad),index));db.commit()
            self.assertNotEqual(run_child(command(path,output),base/(suffix+'-refused-game.log'),success=False,timeout=300),0)
            records.append(dict(event=event,actual_process_kill=True,marker=marker,committed_prefix=prefix,games_after_resume=8,prefix_exactly_retained=True,literal_choices=choices,all_resumed_games_literally_replayed=True,timed_games_not_required_to_match_fresh_choices=True,completed_no_search_and_bytes_mtimes_unchanged=True,missing_export_repaired_without_search=True,invalid_rehashed_marker_not_published=True,rehashed_final_margin_refused=True))
        full=b.read_plan('posttraining');state=full['cases'][0]['state'];deadline_profiles=0
        for name in full['profiles']:
            with CpuBatchEvaluator(ROOT/full['models'][full['profiles'][name]['model']],cache_capacity=0,max_batch=32) as evaluator:
                ticks=iter(i*.002 for i in range(100))
                with mock.patch.object(b.time,'perf_counter',side_effect=lambda:next(ticks)),mock.patch.object(b.time,'sleep'),mock.patch.object(b,'puct_search',side_effect=AssertionError('Expired budget became unlimited')):empty=b.decide(name,('puct',evaluator),state,1,full)
                self.assertEqual(empty['evaluator']['granted_search_ms'],0);self.assertFalse(empty['decision']['usable_decision']);self.assertEqual(empty['decision']['completed_simulations'],0)
                answer=b.decide(name,('puct',evaluator),state,25,full);forged=copy.deepcopy(answer);counts=forged['evaluator']['counts'];self.assertGreater(counts['native_actual_evals'],0);counts['feature_simulations']=counts['feature_deposits']=0;work=forged['decision']['report']['work'];work['feature_simulations']=work['feature_deposits']=0;work['total_deposits']=work['deposits']
                with self.assertRaises(Exception):b.validate_envelope(forged,state,25,full,name)
                deadline_profiles+=1
        EVIDENCE['live_game_recovery']=dict(stage='posttraining',tests=records,actual_initial_and_committed_game_kills=True,raw_markers_prefix_backups_and_valid_closed_journals_retained=True)
        EVIDENCE['deadline_and_physical_work']=dict(profiles=deadline_profiles,expired_zero_grant_no_search=True,zero_never_unlimited=True,all_zero_feature_work_forgery_refused=True)

    def test_07_all144_games_616_measurements_and_paired_strength_arithmetic(self):
        from congklak import value_pilot_benchmark as b
        stages={};total_choices=measurements=games_count=cases_count=0
        for stage,expectedgames,expectedscalars in (('pretraining',48,264),('posttraining',96,352)):
            plan=b.read_plan(stage);path,output=b.paths(stage);result=load(output);paired={};choices=0
            with closing(dbopen(path/'work.sqlite3')) as db:
                b.validate_saved_units(db,plan,Native());case_rows=list(db.execute('SELECT idx,body,hash FROM cases ORDER BY idx'));game_rows=list(db.execute('SELECT idx,comparison,pair,seat,body,hash FROM games ORDER BY idx'))
                self.assertEqual(len(case_rows),44);self.assertEqual(len(game_rows),expectedgames);bucket=defaultdict(list);scalars=0
                for index,text,digest in case_rows:
                    case=json.loads(text);self.assertEqual(config.digest(case),digest);reference=plan['cases'][index]['reference'];self.assertEqual(case['reference'],reference);q={x['pit']:x['value_player0'] for x in reference['action_scores']};sign=1 if plan['cases'][index]['state']['controller']==0 else -1
                    for bi,ms in enumerate((25,100)):
                        names=plan['engines'][index%len(plan['engines']):]+plan['engines'][:index%len(plan['engines'])];names=names[::-1] if bi else names;self.assertEqual(case['orders'][str(ms)],names)
                        for name,value in case['measurements'][str(ms)].items():
                            decision=value['decision']['decision'];choice=decision['chosen'];self.assertEqual(decision['requested_ms'],ms);self.assertEqual(value['reference_optimal'],choice in reference['best_moves'] if choice is not None else None);self.assertEqual(value['reference_regret'],sign*(reference['target_player0']-q[choice]) if choice is not None else None);self.assertGreaterEqual(decision['wall_ms'],decision['engine_elapsed_ms']);self.assertAlmostEqual(decision['overshoot_ms'],max(0.,decision['wall_ms']-ms));bucket[(case['cohort'],str(ms),name)].append(value);scalars+=1
                self.assertEqual(scalars,expectedscalars)
                for key,values in bucket.items():
                    summary=result['case_results']['/'.join(key)];usable=[v for v in values if v['decision']['decision']['usable_decision']]
                    self.assertEqual(summary['positions'],len(values));self.assertEqual(summary['no_completed_recommendation'],len(values)-len(usable));self.assertEqual(summary['reference_optimal_accuracy'],statistics.mean(v['reference_optimal'] for v in usable) if usable else None);self.assertEqual(summary['mean_reference_regret_seeds'],statistics.mean(v['reference_regret'] for v in usable) if usable else None)
                actual_games=[]
                for index,name,pair,seat,text,digest in game_rows:
                    game=json.loads(text);self.assertEqual(config.digest(game),digest);self.assertEqual(plan['arena_work_order'][index],dict(comparison=name,pair=pair,seat=seat));choices+=literal_game(self,game);actual_games.append(game)
                for comp in plan['arena_comparisons']:
                    gs=[g for g in actual_games if g['comparison']==comp['id']];values=[];scores=[];by_family={}
                    for pair in range(12):
                        group=[g for g in gs if g['pair']==pair]
                        if len(group)==2 and all(g['status']=='terminal' for g in group):
                            margin=statistics.mean(g['candidate_final_margin'] for g in group);values.append(margin);by_family[pair]=margin;scores.append(statistics.mean(g['candidate_match_score'] for g in group))
                    paired[comp['id']]=by_family;rng=random.Random(plan['bootstrap_seed']);bootstrap=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(4096)) if values else [];ci=[bootstrap[int(.025*len(bootstrap))],bootstrap[int(.975*len(bootstrap))]] if values else None;summary=result['arenas'][comp['id']];fallbacks={name:sum(g['fallbacks'].get(name,0) for g in gs) for name in (comp['candidate'],comp['opponent'])}
                    expected=dict(pairs=12,complete_pairs=len(values),incomplete_pairs=12-len(values),terminal_games=sum(g['status']=='terminal' for g in gs),unfinished_games=sum(g['status']!='terminal' for g in gs),candidate_wins=sum(g['candidate_match_score']==1 for g in gs),candidate_draws=sum(g['candidate_match_score']==.5 for g in gs),candidate_losses=sum(g['candidate_match_score']==0 for g in gs),paired_mean_score=statistics.mean(scores) if scores else None,primary_paired_mean_final_margin_seeds=statistics.mean(values) if values else None,primary_margin_bootstrap_95_interval=ci,primary_strength_gate_passed=bool(len(values)==12 and not sum(fallbacks.values()) and ci and ci[0]>0),fallback_counts=fallbacks)
                    self.assertEqual(summary,expected)
                gates={name:all(result['arenas'][name+'_vs_'+base]['primary_strength_gate_passed'] for base in b.BASELINES) for name in plan['profiles']};self.assertEqual(result['both_baseline_primary_gates'],gates)
                if stage=='posttraining':
                    for base in b.BASELINES:
                        treatment=paired['value_pilot_vs_'+base];control=paired['policy_control_vs_'+base];matched=sorted(set(treatment)&set(control));self.assertEqual(result['secondary_factor_contrasts'][base],dict(treatment='value_pilot',control='policy_control',expected_families=12,complete_matched_families=len(matched),paired_mean_seed_margin_difference=statistics.mean(treatment[k]-control[k] for k in matched) if len(matched)==12 else None,confidence_interval=None,primary_gate=False))
                self.assertFalse(result['default_promotion']);self.assertFalse(result['sealed_final_test_consumed']);self.assertTrue(result['development_reuse'])
            stages[stage]=dict(cases=44,games=expectedgames,scalar_measurements=scalars,literal_choices=choices,paired_margins={name:dict(sorted(values.items())) for name,values in paired.items()},both_baseline_gates=result['both_baseline_primary_gates'],secondary_factor_contrasts=result['secondary_factor_contrasts']);total_choices+=choices;measurements+=scalars;games_count+=expectedgames;cases_count+=44
        EVIDENCE['exhaustive_benchmark']=dict(stages=stages,games=games_count,cases=cases_count,scalar_measurements=measurements,literal_choices=total_choices,all_full_game_traces_literally_replayed=True,all_case_regrets_and_usable_denominators_rederived=True,all_paired_margins_bootstrap_intervals_gates_and_contemporaneous_contrasts_independently_rederived=True,all_budgets_orders_and_physical_telemetry_validated=True)

    def test_08_completed_CPU_no_work_stability_ownership_and_old_immutability(self):
        from congklak import value_pilot_learning as p,value_pilot_benchmark as b
        output=self.base/'completed-production.json';spec=self.base/'completed-production-spec.json';save(spec,dict(run=str(p.DEFAULT),fixture=False,output=str(output)))
        run_child([sys.executable,str(Path(__file__)),'worker','completed-training','--spec',str(spec)],self.base/'completed-production.log',timeout=300);self.assertTrue(load(output)['passed']);records=[]
        for stage in b.STAGES:
            path,report=b.paths(stage);resource=report.with_name(report.stem+'-resources.json');before=snapshot(path);publications={p:(sha(p),p.stat().st_mtime_ns) for p in (report,resource)}
            with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed stage searched teacher')),mock.patch.object(b,'puct_search',side_effect=AssertionError('Completed stage searched PUCT')),mock.patch.object(b,'NeuralNative',side_effect=AssertionError('Completed stage allocated relay')):
                b.run(stage,on_event=lambda _:None);b.closed_witness(stage)
            self.assertEqual(snapshot(path),before)
            for p,digestmtime in publications.items():self.assertEqual((sha(p),p.stat().st_mtime_ns),digestmtime)
            records.append(dict(stage=stage,zero_teacher_or_PUCT_searches=True,no_relay_allocation=True,all_journal_report_resource_bytes_mtimes_unchanged=True))
        for name,digest in load(ROOT/'build/value-pilot-baseline.json')['inputs'].items():self.assertEqual(sha(ROOT/name),digest,name)
        self.assertEqual(sha(MODEL),MODEL_SHA);self.assertNotIn('torch',sys.modules)
        EVIDENCE['completed_pipeline_stability']=dict(training_CPU_verify_no_Torch_or_optimizer_work=True,stages=records,all2426_prior_hashes_current=True,model_source_unchanged=True,independent_parent_Torch_imported=False,no_new_teacher_labeling=True,no_default_promotion=True,consumed_development_not_release_test=True)


GROUPS={'data':['test_01_all_prior_inputs_and_fixed_model_contract','test_02_independent_ownership_targets_probability_and_source_reconstruction'],
    'plan':['test_03_predeclared_schedule_and_complete_pretraining_control'],
    'training':['test_04_actual_CPU_checkpoint_kills_exact_optimizer_rng_and_completed_no_work'],
    'models':['test_05_every_production_optimizer_step_checkpoint_and_global_fixed_policy'],
    'recovery':['test_06_actual_game_pre_postcommit_recovery_deadline_and_corruption_refusals'],
    'final':['test_07_all144_games_616_measurements_and_paired_strength_arithmetic','test_08_completed_CPU_no_work_stability_ownership_and_old_immutability']}


def main():
    import argparse
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        parser=argparse.ArgumentParser();parser.add_argument('worker');parser.add_argument('kind',choices=('state','optimizer','completed-training'));parser.add_argument('--spec',required=True);args=parser.parse_args();spec=load(Path(args.spec));return {'state':state_worker,'optimizer':optimizer_worker,'completed-training':completed_training_worker}[args.kind](spec)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('group',choices=list(GROUPS));args=parser.parse_args();started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(ValuePilotIndependentAudit(name) for name in GROUPS[args.group]))
    names=['tests/value_pilot_independent_audit.py','congklak/value_pilot_data.py','tools/prepare-value-pilot-data.py','congklak/value_pilot_learning.py','tools/value-pilot-train.py','congklak/value_pilot_benchmark.py','tools/benchmark-value-pilot.py','build/value-pilot-baseline.json','build/value-pilot-data.json','build/value-pilot-rows.json','build/value-pilot-sampling-review.json','build/value-pilot-benchmark-plan.json','build/value-pilot-benchmark-implementation.json','build/value-pilot-training-plan.json','build/value-pilot-training-implementation.json'];paths=[ROOT/name for name in names if (ROOT/name).is_file()]
    paths += [path for path in ValuePilotIndependentAudit.base.rglob('*') if path.is_file() and not path.name.endswith(('-wal','-shm'))]
    if args.group not in ('data','plan','training'):
        paths += [ROOT/name for name in ('data/value-pilot-v1/model.json','data/value-pilot-v1/result.json','data/value-pilot-v1/catalog.sqlite3','build/value-pilot-model-witness.json')]
    if args.group in ('plan','models','recovery','final'):
        for stage in ('pretraining','posttraining'):
            for name in ('data/value-pilot-benchmark-'+stage+'-v1/work.sqlite3','build/value-pilot-benchmark-'+stage+'.json','build/value-pilot-benchmark-'+stage+'-resources.json'):
                if (ROOT/name).exists():paths.append(ROOT/name)
    body=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),covered_methods=GROUPS[args.group],evidence=EVIDENCE,inputs=pins(paths),duration_seconds=time.monotonic()-started,retained_directory=relative(ValuePilotIndependentAudit.base),failure_details=[dict(method=str(test),trace=trace) for test,trace in result.failures+result.errors]);save(WORK/(args.group+'.json'),body);print(config.canonical(dict(event='value_pilot_independent_audit',group=args.group,passed=body['passed'],tests=result.testsRun)),flush=True);return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
