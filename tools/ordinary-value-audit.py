"""Independent isolated CPU workers for the ordinary value pilot audit."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import bisect,copy,hashlib,json,random,runpy,sys
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from congklak.native import Native
from data_pilot_independent_audit import sha,load,save,pins,snapshot
from value_pilot_independent_audit import tensor_equal,pt_catalog
MODEL=ROOT/'data/quality-train-v1/model.json'
FROZEN={'layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias'}
HEAD={'layers.4.weight','layers.4.bias'}


def state(spec):
    import torch
    a=torch.load(spec['left'],map_location='cpu',weights_only=True);b=torch.load(spec['right'],map_location='cpu',weights_only=True)
    tensor_equal(torch,a,b)
    save(spec['output'],dict(passed=True,step=a['step'],draws=a['draws'],all_model_optimizer_sampler_python_torch_rng_raw_bytes_equal=True,
        inputs=pins([Path(spec['left']),Path(spec['right']),Path(__file__),Path(spec['spec_path'])])))


def closed(spec):
    from congklak import ordinary_value_pilot as p,ordinary_value_benchmark as b,ordinary_value_followup as f,quality_search
    path=Path(spec['run']);before=snapshot(path);events=[];kind=spec['stage'];fixture=spec.get('fixture',False)
    with mock.patch.object(Native,'analyze',side_effect=AssertionError('Completed teacher search')) as teacher,\
         mock.patch.object(quality_search,'search',side_effect=AssertionError('Completed PUCT search')) as search,\
         mock.patch.object(quality_search,'forest',side_effect=AssertionError('Completed forest search')) as forest,\
         mock.patch.object(p,'runtime',side_effect=AssertionError('Completed activated Torch')) as runtime,\
         mock.patch.object(b,'decide',side_effect=AssertionError('Completed benchmark decision')) as decide:
        if kind=='labels':p.run_labels(path,fixture=fixture,on_event=events.append);p.load_labels(path)
        elif kind=='training':p.train(path,fixture=fixture,on_event=events.append);p.verify_run(path,fixture=fixture)
        elif kind=='benchmark':b.run(path=path,output=Path(spec['report']),fixture=fixture,fixture_control=fixture,on_event=events.append)
        else:f.run(path,fixture=fixture,on_event=events.append);f.load_run(path,fixture=fixture)
    assert before==snapshot(path) and 'torch' not in sys.modules
    assert all(v.call_count==0 for v in (teacher,search,forest,runtime,decide))
    assert not any(v['event'].endswith(('_precommit','_committed')) for v in events)
    save(spec['output'],dict(passed=True,stage=kind,zero_teacher_PUCT_optimizer_benchmark_work=True,torch_imported=False,
        all_durable_bytes_sizes_mtimes_unchanged=True,events=events,inputs=pins([Path(__file__),Path(spec['spec_path'])])))


def optimizer(spec):
    """Recompute all gradients from source tensors, not source trainer operators."""
    import torch
    from torch import nn
    from congklak import ordinary_value_pilot as p
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    run=Path(spec['run']);export=load(run/'model.json')['model'];identity=export['identity'];source=load(MODEL)['model']['weights']
    assert identity['steps']==200 and identity['batch_size']==64 and identity['checkpoint_every']==20
    assert identity['seed']==20261028 and identity['lr']==.0001 and identity['device']=='cpu'
    random.seed(identity['seed']);torch.manual_seed(identity['seed'])
    class Network(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.Sequential(nn.Linear(127,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh());self.policy=nn.Linear(64,7)
    model=Network();torch_rng=torch.get_rng_state().clone();model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in source.items()})
    for name,param in model.named_parameters():param.requires_grad_(name in HEAD)
    def digest(values):
        h=hashlib.sha256()
        for name,value in sorted(values.items()):h.update(name.encode());h.update(bytes(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
        return h.hexdigest()
    groups=load(ROOT/'build/ordinary-value-rows.json')['groups'];hidden={};target={};initial_policy={}
    cache=dict(source_model_sha256=sha(MODEL),frozen_weights_sha256=identity['initial_frozen_weights_sha256'],rows_sha256={},features_sha256={},hidden_sha256={},policy_sha256={})
    with torch.no_grad():
        for name,group in groups.items():
            x=torch.tensor([row['cached_features']['x'] for row in group],dtype=torch.float32)
            h=model.layers[3](model.layers[2](model.layers[1](model.layers[0](x))));hidden[name]=h;initial_policy[name]=model.policy(h)
            target[name]={key:torch.tensor([row['cached_features'][key] for row in group],dtype=torch.float32) for key in ('d','r')}
            target[name]['y']=torch.tensor([row['target_player0'] for row in group],dtype=torch.float32)
            target[name]['z']=torch.tensor([(row['target_player0']-row['cached_features']['d'])/row['cached_features']['r'] for row in group],dtype=torch.float32)
            cache['rows_sha256'][name]=config.digest([row['state_key'] for row in group]);cache['features_sha256'][name]=digest({'x':x});cache['hidden_sha256'][name]=digest({'h':h});cache['policy_sha256'][name]=digest({'logits':initial_policy[name]})
    assert cache==identity['hidden_cache'] and config.digest(cache)==identity['hidden_cache_sha256']
    def metrics():
        answer={}
        with torch.no_grad():
            for name,group in groups.items():
                z=model.layers[5](model.layers[4](hidden[name])).squeeze(-1)
                answer[name]=dict(rows=len(group),mae_seeds=(target[name]['d']+target[name]['r']*z-target[name]['y']).abs().mean().item(),residual_huber_delta_point1=torch.nn.functional.huber_loss(z,target[name]['z'],delta=.1).item(),target_kind=group[0]['target_kind'])
        return answer
    initial=metrics();rootids={}
    for name in ('high','low'):
        byroot=defaultdict(list)
        for i,row in enumerate(groups[name]):byroot[row['state_key']].append(i)
        rootids[name]=dict(sorted(byroot.items()));assert all(len(v)==2 for v in byroot.values())
    cdf=[];total=0.;strata=Counter(r['stratum'] for r in groups['rehearsal'])
    for row in groups['rehearsal']:
        prob=.9/len(groups['rehearsal'])+.1/(len(strata)*strata[row['stratum']]);assert prob==row['sampling_probability'];total+=prob;cdf.append(total)
    assert identity['sampler']==dict(high=rootids['high'],low=rootids['low'],rehearsal_cdf=cdf)
    opt=torch.optim.AdamW([model.layers[4].weight,model.layers[4].bias],lr=.0001,weight_decay=0.,foreach=False)
    rng=random.Random(identity['seed']);entries=pt_catalog(run);assert [r[1] for r in entries]==list(range(20,201,20));by_step={r[1]:r for r in entries};verified=[];paths=[];ledger=Counter()
    for step in range(1,201):
        batch={}
        for name in ('high','low'):
            keys=sorted(rootids[name]);batch[name]=[rng.choice(rootids[name][rng.choice(keys)]) for _ in range(16)]
        batch['rehearsal']=[bisect.bisect_right(cdf,rng.random()*cdf[-1]) for _ in range(32)]
        for name,ids in batch.items():ledger.update(groups[name][i]['id'] for i in ids)
        h=torch.cat([hidden[name][batch[name]] for name in ('high','low','rehearsal')]);y=torch.cat([target[name]['z'][batch[name]] for name in ('high','low','rehearsal')])
        opt.zero_grad(set_to_none=True);z=model.layers[5](model.layers[4](h)).squeeze(-1);loss=torch.nn.functional.huber_loss(z,y,delta=.1);loss.backward()
        assert all((param.grad is None)==(name in FROZEN) for name,param in model.named_parameters())
        torch.nn.utils.clip_grad_norm_([model.layers[4].weight,model.layers[4].bias],1.);opt.step()
        if step not in by_step:continue
        _,_,filename,checksum,text,bodyhash=by_step[step];file=run/filename;meta=json.loads(text);assert sha(file)==checksum and config.digest(meta)==bodyhash
        actual=torch.load(file,map_location='cpu',weights_only=True)
        expected=dict(identity=identity,step=step,draws=64*step,model=model.state_dict(),optimizer=opt.state_dict(),sampler_rng=rng.getstate(),python_rng=random.Random(identity['seed']).getstate(),torch_rng=torch_rng,cuda_rng=[],scheduler=None,amp_scaler=None,pending_batches=[])
        tensor_equal(torch,actual,expected,'checkpoint'+str(step));assert set(actual['optimizer']['state'])=={0,1}
        for name in FROZEN:tensor_equal(torch,actual['model'][name],torch.tensor(source[name],dtype=torch.float32),'frozen'+name)
        assert digest(actual['model'])==meta['weights_sha256'] and metrics()==meta['metrics'];verified.append(dict(step=step,weights_sha256=meta['weights_sha256']));paths.append(file)
    changed=sorted(k for k,v in actual['model'].items() if source[k]!=v.tolist());assert set(changed)==HEAD
    assert export['step']==200 and export['weights_sha256']==digest(actual['model']) and all(export['weights'][k]==v.tolist() for k,v in actual['model'].items())
    result=load(run/'result.json');assert initial==result['initial_metrics'] and metrics()==result['metrics']
    for name,h in hidden.items():tensor_equal(torch,model.policy(h),initial_policy[name],'policy'+name)
    accepted=p.evaluator();p.validate_state(actual,identity,accepted);refused=[]
    for kind in [*sorted(FROZEN),'draws','ids','moment_shape','weight_decay','sampler']:
        bad=copy.deepcopy(actual)
        if kind in FROZEN:bad['model'][kind].reshape(-1)[0]+=.25
        elif kind=='draws':bad['draws']+=1
        elif kind=='ids':bad['optimizer']['state'][2]=copy.deepcopy(bad['optimizer']['state'][0])
        elif kind=='moment_shape':bad['optimizer']['state'][0]['exp_avg']=torch.zeros(64)
        elif kind=='weight_decay':bad['optimizer']['param_groups'][0]['weight_decay']=.01
        else:bad['sampler_rng']=random.Random(999).getstate()
        try:p.validate_state(bad,identity,accepted)
        except (ValueError,RuntimeError,KeyError):refused.append(kind)
        else:raise AssertionError('Corrupt state admitted '+kind)
    save(spec['output'],dict(schema='ordinary-value-independent-optimizer-v1',passed=True,checkpoints=10,steps=verified,
        all200_optimizer_steps_independently_replayed=True,all_checkpoint_model_optimizer_sampler_python_torch_rng_raw_bytes_equal=True,
        all_frozen_tensors_bit_exact=True,all_dataset_policy_logits_bit_exact=True,trainable_parameters=65,frozen_parameters=12807,
        optimizer_parameter_ids=[0,1],changed_tensors=changed,draws=12800,draw_ledger=dict(sorted(ledger.items())),
        hidden_cache_independently_reconstructed=True,initial_and_final_metrics_independently_rederived=True,final_export_equals_PT=True,
        checkpoint_corruptions_refused=refused,inputs=pins(paths+[run/'catalog.sqlite3',run/'model.json',run/'result.json',MODEL,ROOT/'build/ordinary-value-rows.json',Path(__file__),Path(spec['spec_path'])])))


def main():
    spec=load(sys.argv[1]);spec['spec_path']=str(Path(sys.argv[1]).resolve());assert Path(spec['output']).resolve().is_relative_to(ROOT/'build')
    {'state':state,'closed':closed,'optimizer':optimizer}[spec['kind']](spec)
    print(config.canonical(dict(event='ordinary_value_independent_worker',kind=spec['kind'],passed=True)),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
