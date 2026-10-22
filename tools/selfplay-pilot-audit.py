"""Separate CPU-only audit workers; no accepted artifact or report is rewritten."""
from __future__ import annotations
from collections import Counter,defaultdict
from contextlib import closing
import copy
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config
from data_pilot_independent_audit import sha,load,save,pins,dbopen,snapshot
from value_pilot_independent_audit import tensor_equal,pt_catalog


def compare_state(spec):
    import torch
    left=torch.load(spec['left'],map_location='cpu',weights_only=True)
    right=torch.load(spec['right'],map_location='cpu',weights_only=True)
    tensor_equal(torch,left,right)
    save(spec['output'],dict(passed=True,step=left['step'],draws=left['draws'],
        all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal=True,
        inputs=pins([Path(spec['left']),Path(spec['right']),Path(__file__)])))


def completed_train(spec):
    from congklak import selfplay_training_pilot as producer
    path=Path(spec['run']);before=snapshot(path)
    with mock.patch.object(producer,'runtime',side_effect=AssertionError('Completed trainer activated Torch')) as blocked:
        value=producer.train(path,replay_path=Path(spec['replay']),fixture=spec.get('fixture',False))
        proof=producer.verify_run(path,fixture=spec.get('fixture',False))
    assert value['status']=='complete' and proof['passed'] and blocked.call_count==0
    assert 'torch' not in sys.modules and snapshot(path)==before
    save(spec['output'],dict(passed=True,zero_training_runtime_activations=True,
        zero_optimizer_steps=True,torch_imported=False,all_durable_bytes_mtimes_unchanged=True))


def joint_optimizer(spec):
    """Independently reconstruct every update, not merely inspect optimizer shapes."""
    import torch
    from congklak import features_relay
    from congklak.native import Native
    from congklak.puct_cpu_inference import Model
    from congklak import selfplay_training_pilot as producer
    run=Path(spec['run']);cpu=producer.verify_run(run,fixture=spec.get('fixture',False))
    identity=cpu['identity'];manifest,rows=producer.load_replay(ROOT/identity['replay_path'])
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);random.seed(identity['seed']);torch.manual_seed(identity['seed'])
    class IndependentNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=torch.nn.Sequential(torch.nn.Linear(127,64),torch.nn.ReLU(),torch.nn.Linear(64,64),
                torch.nn.ReLU(),torch.nn.Linear(64,1),torch.nn.Tanh())
            self.policy=torch.nn.Linear(64,7)
        def both(self,x):
            h=torch.relu(self.layers[0](x));h=torch.relu(self.layers[2](h))
            return torch.tanh(self.layers[4](h)).squeeze(-1),self.policy(h)
    model=IndependentNetwork();source=Model(producer.SOURCE)
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in source.weights.items()})
    assert len(list(model.parameters()))==8 and sum(v.numel() for v in model.parameters())==12872
    groups=defaultdict(list);native=Native();x=[];legal=[];pi=[];z=[];d=[];r=[];y=[]
    for index,row in enumerate(rows):
        groups[row['family']].append(index);p=row['state'];active=sum(map(sum,p['pits']))
        assert active>=25 and row['labelkind']=='selfplay_terminal_seed_margin'
        feature,offset,scale=features_relay.encode(p,127,native)
        assert row['features']==dict(x=feature,d=offset,r=scale)
        assert scale==(active if p['controller']==0 else -active)
        assert offset==p['stores'][0]-p['stores'][1]
        x.append(feature);d.append(offset);r.append(scale);y.append(row['terminal_return_player0'])
        legal.append([bool(row['legal_mask']&(1<<a)) for a in range(7)]);pi.append(row['policy_target'])
        z.append((y[-1]-offset)/scale)
        assert pi[-1]==[v/manifest['identity']['settings']['simulations'] for v in row['root_visits']]
    groups=dict(sorted(groups.items()));assert groups==identity['sampler_groups'];families=sorted(groups)
    x=torch.tensor(x,dtype=torch.float32);legal=torch.tensor(legal,dtype=torch.bool)
    pi=torch.tensor(pi,dtype=torch.float32);z=torch.tensor(z,dtype=torch.float32)
    d=torch.tensor(d,dtype=torch.float32);r=torch.tensor(r,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32)
    def losses(ids):
        value,logits=model.both(x[ids])
        logp=torch.nn.functional.log_softmax(logits.masked_fill(~legal[ids],-math.inf),dim=1).masked_fill(~legal[ids],0.)
        return torch.nn.functional.huber_loss(value,z[ids],delta=.1),-(pi[ids]*logp).sum(dim=1).mean()
    def metrics():
        with torch.no_grad():
            v,logits=model.both(x)
            logp=torch.nn.functional.log_softmax(logits.masked_fill(~legal,-math.inf),dim=1).masked_fill(~legal,0.)
            return dict(rows=len(rows),mae_seeds=(d+r*v-y).abs().mean().item(),
                residual_huber_delta_point1=torch.nn.functional.huber_loss(v,z,delta=.1).item(),
                visit_policy_cross_entropy=(-(pi*logp).sum(dim=1)).mean().item(),
                scope='In-sample admitted selfplay targets; no checkpoint selection')
    assert metrics()==cpu['result']['initial_metrics']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001,foreach=False)
    rng=random.Random(identity['seed']);entries=pt_catalog(run);by_step={entry[1]:entry for entry in entries}
    assert len(entries)==math.ceil(identity['steps']/identity['checkpoint_every'])
    draw_counts=Counter();checked=[];loss_values=[]
    for step in range(1,identity['steps']+1):
        selected=[]
        for _ in range(identity['batch_size']):
            family=families[rng.randrange(len(families))];ids=groups[family]
            selected.append(ids[rng.randrange(len(ids))]);draw_counts[rows[selected[-1]]['id']]+=1
        indices=torch.tensor(selected,dtype=torch.int64);optimizer.zero_grad(set_to_none=True)
        vl,pl=losses(indices);loss=vl+.25*pl;assert torch.isfinite(loss)
        loss.backward();assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_values.append(float(loss))
        if step in by_step:
            entry=by_step[step];state=torch.load(run/entry[2],map_location='cpu',weights_only=True)
            expected=dict(identity=identity,step=step,draws=step*identity['batch_size'],
                model=model.state_dict(),optimizer=optimizer.state_dict(),sampler_rng=rng.getstate(),
                python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=[],
                scheduler=None,amp_scaler=None,pending_batches=[])
            tensor_equal(torch,expected,state)
            assert json.loads(entry[4])['metrics']==metrics()
            checked.append(dict(step=step,file=entry[2],sha256=sha(run/entry[2])))
    candidate=Model(run/'model.json')
    for name,tensor in model.state_dict().items():
        tensor_equal(torch,tensor,torch.tensor(candidate.weights[name],dtype=torch.float32),name)
    assert metrics()==cpu['result']['metrics'];changed=[k for k in source.weights if source.weights[k]!=candidate.weights[k]]
    assert 'policy.weight' in changed and 'layers.4.weight' in changed and any(k.startswith('layers.0') for k in changed)
    save(spec['output'],dict(schema='selfplay-pilot-independent-all-update-replay-v1',passed=True,
        steps_replayed=identity['steps'],draws_replayed=identity['steps']*identity['batch_size'],
        checkpoints=checked,optimizer_parameter_ids=list(range(8)),all_model_optimizer_sampler_python_torch_rng_raw_tensor_bytes_equal=True,
        all_checkpoint_metrics_independently_equal=True,final_export_equals_replayed_tensors=True,
        all8_tensors_trainable=True,changed_tensors=changed,source_model_exact=True,
        policy_and_value_gradients=True,teacher_rehearsal=False,features_targets_and_visits_rederived=True,
        sampler_row_draw_ledger_sha256=config.digest([[row['id'],draw_counts[row['id']]] for row in rows]),
        inputs={**cpu['inputs'],**pins([Path(__file__),Path(spec['spec_path'])])}))


def main():
    spec=load(Path(sys.argv[1]));spec['spec_path']=str(Path(sys.argv[1]));output=Path(spec['output']).resolve()
    assert output.is_relative_to(ROOT/'build')
    kinds={'compare_state':compare_state,'completed_train':completed_train,'joint_optimizer':joint_optimizer}
    kinds[spec['kind']](spec)
    print(config.canonical(dict(event='selfplay_pilot_independent_worker',kind=spec['kind'],passed=True)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
