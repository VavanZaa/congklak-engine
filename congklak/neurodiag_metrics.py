"""Fixed-export CPU diagnostics. No Native, Torch, optimizer or producer imports."""
from fractions import Fraction as F
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'tests/reference.py'
SCHEMA='neurodiag-fixed-cached-value-fit-v1'
ENCODER='current-controller-count43-own-opponent-choice14-storegain-drops-relays-capture-retain-terminal84/store-offset-exact-v5'
RULES='c521cc9fea211aeb63d5942d80cc084ea095da74effec13c041325e80345414e'
NAMES=('control','candidate')
SHAPES={'layers.0.weight':(64,127),'layers.0.bias':(64,),'layers.2.weight':(64,64),'layers.2.bias':(64,),
    'layers.4.weight':(1,64),'layers.4.bias':(1,),'policy.weight':(7,64),'policy.bias':(7,)}
FROZEN=('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias')
SETTINGS=dict(numpy_version='2.3.2',threadpoolctl_version='3.6.0',dtype='float64',weight_storage='rounded_float32',input_storage='unrounded_cached_float64',
    blas_threads=1,svd_rcond=1e-10,model_saturation=0.95,target_near_boundary=0.95,relu_kink_abs=1e-8,residual_huber_delta=0.1,
    scalar_atol=1e-10,scalar_rtol=1e-12,finite_difference_step=1e-6,finite_difference_atol=1e-6,finite_difference_rtol=1e-4,
    feature_atol=1e-12,deposit_guard=1000000,max_units=384,max_target_records=384)
_NP=None
_LIMIT=None


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(body):return json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(body):return hashlib.sha256(canonical(body).encode()).hexdigest()
def same(a,b):return canonical(a)==canonical(b)
def integer(v):return type(v) is int
def number(v):return type(v) in (int,float) and math.isfinite(v)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fp32(v):
    require(number(v),'Finite model number required');return struct.unpack('<f',struct.pack('<f',v))[0]
def rational(v):
    require(type(v) is F or integer(v),'Exact rational required');q=F(v);return dict(numerator=q.numerator,denominator=q.denominator,value=float(q))
def _norm(v):return math.sqrt(sum(float(x)*float(x) for x in v))
def _hex(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def _vector(v,n):return isinstance(v,list) and len(v)==n and all(number(x) for x in v)


def load_models():
    """Read the two fixed exports; no model construction or inference."""
    paths=dict(control=ROOT/'data/quality-train-v1/model.json',candidate=ROOT/'data/ordinary-value-pilot-v1/model.json')
    return {name:dict(file_sha256=sha(path),envelope=json.loads(path.read_text(encoding='utf-8'))) for name,path in paths.items()}


def numpy_runtime():
    """The only NumPy entry point; completed public readers must never call it."""
    global _NP,_LIMIT
    if _NP is None:
        for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
        require(importlib.metadata.version('numpy')==SETTINGS['numpy_version'] and importlib.metadata.version('threadpoolctl')==SETTINGS['threadpoolctl_version'],'Pinned diagnostic numerical runtime differs')
        import numpy as np
        from threadpoolctl import threadpool_limits
        _LIMIT=threadpool_limits(limits=1);_NP=np
    return _NP


def runtime_metadata():
    numpy_runtime()
    from threadpoolctl import threadpool_info
    libraries=[]
    for item in threadpool_info():
        if item.get('user_api') not in ('blas','openmp'):continue
        require(item['num_threads']==1,'Numerical library is not single-threaded')
        libraries.append({k:item.get(k) for k in ('user_api','internal_api','prefix','version','num_threads','threading_layer','architecture')})
        libraries[-1]['library_sha256']=sha(item['filepath'])
    return dict(python=sys.version.split()[0],numpy=SETTINGS['numpy_version'],threadpoolctl=SETTINGS['threadpoolctl_version'],libraries=sorted(libraries,key=canonical),settings=SETTINGS)


def _state(state):
    require(isinstance(state,dict) and set(state)=={'pits','stores','controller'},'Exact full state required')
    require(integer(state['controller']) and state['controller'] in (0,1),'Invalid controller')
    require(isinstance(state['pits'],list) and len(state['pits'])==2 and all(isinstance(r,list) and len(r)==7 for r in state['pits']) and isinstance(state['stores'],list) and len(state['stores'])==2,'Wrong state shape')
    counts=state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1]]
    require(all(integer(v) and 0<=v<=98 for v in counts) and sum(counts)==98,'State seed conservation differs')
    return counts


def rawkey(state):return bytes(_state(state)+[state['controller']]).hex()


def _cheap_features(state):
    _state(state);active=sum(map(sum,state['pits']));actor=state['controller']
    if active and not sum(state['pits'][actor]):actor=1-actor
    counts=state['pits'][actor]+state['pits'][1-actor]
    x=[v/98 for v in counts]+[active/98]+[float(v>0) for v in counts]+[float(v==1) for v in counts]
    return x,state['stores'][0]-state['stores'][1],active*(1 if actor==0 else -1),actor


def literal_features(state):
    """Exact accepted43+84 specification with literal sowing, never Native."""
    name='_neurodiag_seed_reference'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,REFERENCE);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    ref=sys.modules[name];require(Path(ref.__file__).resolve()==REFERENCE.resolve(),'Literal encoder interpreter differs')
    x,d,r,actor=_cheap_features(state);counts=_state(state)
    for side in (actor,1-actor):
        for pit,count in enumerate(state['pits'][side]):
            if not count:x.extend([0.0]*6);continue
            transition=ref.apply(ref.Position(7,tuple(counts),side),pit,deposit_guard=SETTINGS['deposit_guard']);store=7 if side==0 else 15
            x.extend([(transition.position.counts[store]-counts[store])/98,transition.deposits/98,transition.relay_pickups/98,
                float(transition.captured>0),float(not transition.terminal and transition.position.controller==side),float(transition.terminal)])
    result=dict(x=x,d=d,r=r);validate_features(state,result);return result


def validate_features(state,features):
    require(isinstance(features,dict) and set(features)=={'x','d','r'} and _vector(features['x'],127),'Cached127 feature shape/numbers differ')
    x,d,r,actor=_cheap_features(state);require(integer(features['d']) and integer(features['r']) and features['d']==d and features['r']==r,'Cached d/signed-active perspective differs')
    require(all(abs(float(a)-b)<=SETTINGS['feature_atol'] for a,b in zip(features['x'][:43],x)),'Cached cheap43/count orientation differs')
    for i,count in enumerate(state['pits'][actor]+state['pits'][1-actor]):
        values=features['x'][43+6*i:49+6*i]
        require(all(v>=0 for v in values[:3]) and all(v in (0,1) for v in values[3:]),'Relay feature ranges differ')
        if not count:require(all(v==0 for v in values),'Empty legal pit has nonzero relay features')
    return features


def prepare_models(models):
    require(isinstance(models,dict) and set(models)==set(NAMES),'Exactly control/candidate exports required');admitted={};ids={}
    for name,entry in models.items():
        require(isinstance(entry,dict) and set(entry)=={'file_sha256','envelope'} and _hex(entry['file_sha256']),'Fixed model file binding differs')
        envelope=entry['envelope'];require(isinstance(envelope,dict) and set(envelope)=={'model','sha256'},'Unknown export envelope')
        body=envelope['model'];identity=body['identity'];require(digest(body)==envelope['sha256'] and body['schema']=='congklak-mlp-export-v4' and body['encoder']==ENCODER and body['uses_oracle'] is False,'Unknown/damaged fixed export')
        require(integer(identity['width']) and identity['width']==64 and integer(identity['input_count']) and identity['input_count']==127 and identity['rules_digest']==RULES,'Fixed export architecture/rules differ')
        require(set(body['weights'])==set(SHAPES),'Unexpected tensor set');weights={};fingerprints={};allweights=hashlib.sha256()
        for key in sorted(SHAPES):
            value=body['weights'][key];shape=SHAPES[key]
            require(isinstance(value,list) and len(value)==shape[0] and (len(shape)==1 or all(isinstance(row,list) and len(row)==shape[1] for row in value)),'Tensor dimensions differ')
            weights[key]=[[fp32(v) for v in row] for row in value] if len(shape)==2 else [fp32(v) for v in value]
            flat=[v for row in weights[key] for v in row] if len(shape)==2 else weights[key];payload=struct.pack('<'+'f'*len(flat),*flat);fingerprints[key]=hashlib.sha256(payload).hexdigest();allweights.update(key.encode());allweights.update(payload)
        require(allweights.hexdigest()==body['weights_sha256'],'Stored FP32 weight digest differs');admitted[name]=weights
        ids[name]=dict(file_sha256=entry['file_sha256'],envelope_sha256=envelope['sha256'],weights_sha256=body['weights_sha256'],tensor_sha256=fingerprints)
    require(all(ids['control']['tensor_sha256'][k]==ids['candidate']['tensor_sha256'][k] for k in FROZEN),'Trunk/policy six tensors must remain bit-identical')
    return admitted,dict(models=ids,frozen_six_tensors=list(FROZEN),shared_trunk_policy_bit_identical=True,encoder=ENCODER)


def scalar_forward(weights,features):
    """Accepted sequential-sum algebra over stored FP32 weights and f64 cache."""
    x=list(features['x']);d=features['d'];r=features['r']
    if r==0:return dict(value_player0=d,value_units=256*d,policy_logits=[0.0]*7,terminal_early_return=True)
    for layer in ('layers.0','layers.2'):
        x=[max(0,sum(a*v for a,v in zip(row,x))+b) for row,b in zip(weights[layer+'.weight'],weights[layer+'.bias'])]
    u=sum(a*v for a,v in zip(weights['layers.4.weight'][0],x))+weights['layers.4.bias'][0];residual=r*math.tanh(u)
    rounded=math.floor(residual*256+.5) if residual>=0 else math.ceil(residual*256-.5)
    return dict(value_player0=d+residual,value_units=d*256+rounded,policy_logits=[sum(a*v for a,v in zip(row,x))+b for row,b in zip(weights['policy.weight'],weights['policy.bias'])],terminal_early_return=False)


def forward_arrays(weights,features):
    """One nonterminal shared trunk/policy, with current value-head sensitivities."""
    np=numpy_runtime();require(features['r']!=0,'Observation panel must be nonterminal')
    x=np.asarray(features['x'],dtype=np.float64);w0=np.asarray(weights['layers.0.weight'],dtype=np.float64);w2=np.asarray(weights['layers.2.weight'],dtype=np.float64)
    a0=w0@x+np.asarray(weights['layers.0.bias'],dtype=np.float64);h0=np.where(a0>0,a0,0.0);a2=w2@h0+np.asarray(weights['layers.2.bias'],dtype=np.float64);h=np.where(a2>0,a2,0.0)
    logits=np.asarray(weights['policy.weight'],dtype=np.float64)@h+np.asarray(weights['policy.bias'],dtype=np.float64)
    return dict(x=x,a0=a0,h0=h0,a2=a2,h=h,logits=logits,w0=w0,w2=w2)


def head_diagnostics(weights,features,trunk):
    np=numpy_runtime();head=np.asarray(weights['layers.4.weight'][0],dtype=np.float64);h=trunk['h'];u=float(head@h+weights['layers.4.bias'][0]);t=math.tanh(u);gain=1-t*t;r=features['r']
    hidden= gain*head;a2=hidden*(trunk['a2']>0);a0=(trunk['w2'].T@a2)*(trunk['a0']>0);gx=trunk['w0'].T@a0;gh=gain*np.concatenate([h,[1.0]])
    norms=dict(input=_norm(gx),head=_norm(gh),hidden=_norm(hidden),layer0_bias=_norm(a0),layer0_weight=_norm(a0)*_norm(trunk['x']),layer2_bias=_norm(a2),layer2_weight=_norm(a2)*_norm(trunk['h0']))
    residual=r*t;rounded=math.floor(residual*256+.5) if residual>=0 else math.ceil(residual*256-.5)
    return dict(preactivation=u,tanh=t,value_player0=features['d']+residual,value_units=features['d']*256+rounded,normalized_output_gain=gain,seed_output_gain=r*gain,
        saturated=abs(t)>=SETTINGS['model_saturation'],input_jacobian_normalized=gx.tolist(),input_jacobian_player0=(r*gx).tolist(),head_jacobian_normalized=gh.tolist(),head_jacobian_player0=(r*gh).tolist(),
        normalized_jacobian_norms=norms,seed_jacobian_norms={k:abs(r)*v for k,v in norms.items()},
        sensitivity_scope='Local continuous encoded-input/output sensitivity, not legal discrete-board derivatives, causal feature importance or confidence.')


def target_diagnostics(target,features,head):
    require(integer(target),'Saved scalar seed target must be an exact integer');d=features['d'];r=features['r'];require(r!=0,'Terminal targets are outside the observation panel')
    z=F(target-d,r);require(-1<=z<=1,'Target exceeds the conserved remaining-seed interval')
    interior=abs(z)<1;error=head['tanh']-float(z);delta=SETTINGS['residual_huber_delta'];psi=max(-delta,min(delta,error))
    return dict(target_player0=target,target_normalized=rational(z),target_status='strict_interior' if interior else 'endpoint',target_logit=math.atanh(float(z)) if interior else None,
        target_near_boundary=abs(z)>=F(19,20),error_player0=head['value_player0']-target,absolute_error_player0=abs(head['value_player0']-target),normalized_residual_error=error,
        residual_huber_delta=delta,residual_huber_loss=0.5*error*error if abs(error)<=delta else delta*(abs(error)-0.5*delta),residual_huber_psi=psi,
        residual_huber_gradient_pre_activation=psi*head['normalized_output_gain'],residual_huber_gradient_head=[psi*v for v in head['head_jacobian_normalized']],
        gradient_scope='Unreduced normalized-residual Huber(delta=.1) at fixed parameters; not saved training gradients, seed-MAE gradients or an optimizer update.')


def observe(units,models):
    require(isinstance(units,dict) and 0<len(units)<=SETTINGS['max_units'],'Bounded nonempty unit batch required')
    admitted,model_identity=prepare_models(models);observations={};record_ids=set();states=set();records_count=0
    for uid,unit in sorted(units.items()):
        require(isinstance(unit,dict) and isinstance(unit.get('records'),list) and unit['records'],'Unit/source records missing');state=unit['state'];features=validate_features(state,unit['features']);key=rawkey(state)
        require(unit['state_key']==key and uid==digest(dict(state_key=key,features=features)) and key not in states,'Unit identity/duplicate raw state differs');states.add(key)
        require(features['r']!=0,'Production observation units must be nonterminal');require(same(unit['records'],sorted(unit['records'],key=lambda r:r['id'])),'Unit records must have sorted stable IDs')
        trunk=forward_arrays(admitted['control'],features);heads={name:head_diagnostics(admitted[name],features,trunk) for name in NAMES};actor=0 if features['r']>0 else 1;legal=[p for p,n in enumerate(state['pits'][actor]) if n]
        logits=trunk['logits'].tolist();maximum=max(logits[p] for p in legal);exps={p:math.exp(logits[p]-maximum) for p in legal};total=sum(exps.values());probability=[exps.get(p,0)/total for p in range(7)];targets={}
        for record in unit['records']:
            require(isinstance(record,dict) and isinstance(record.get('id'),str) and record['id'] and record['id'] not in record_ids,'Duplicate/invalid scalar record ID');record_ids.add(record['id'])
            require(record['state_key']==key and integer(record['actor']) and record['actor']==state['controller'] and all(isinstance(record.get(k),str) and record[k] for k in ('family','cohort','target_type','stratum')),'Record state/actor/group identity differs')
            targets[record['id']]=dict(record=record,record_sha256=digest(record),models={name:target_diagnostics(record['target_player0'],features,heads[name]) for name in NAMES})
        records_count+=len(targets);require(records_count<=SETTINGS['max_target_records'],'Target record bound exceeded')
        observations[uid]=dict(schema=SCHEMA+'-observation',unit_id=uid,source_unit=unit,source_unit_sha256=digest(unit),feature_sha256=digest(features),state_key=key,normalized_actor=actor,
            shared=dict(layer0_pre_activation=trunk['a0'].tolist(),layer0_hidden=trunk['h0'].tolist(),layer2_pre_activation=trunk['a2'].tolist(),hidden=trunk['h'].tolist(),policy_logits=logits,
                policy_probabilities=probability,legal_actions=legal,policy_sha256=digest(logits),layer0_relu_active=[bool(v>0) for v in trunk['a0']],layer2_relu_active=[bool(v>0) for v in trunk['a2']],layer0_kink=[bool(abs(v)<=SETTINGS['relu_kink_abs']) for v in trunk['a0']],layer2_kink=[bool(abs(v)<=SETTINGS['relu_kink_abs']) for v in trunk['a2']]),
            models=heads,targets=targets,model_identity_sha256=digest(model_identity))
    batch=dict(schema=SCHEMA+'-batch',source_units_sha256=digest(units),models=model_identity,runtime=runtime_metadata(),observations=observations,
        counts=dict(distinct_units=len(units),shared_trunk_forwards=len(units),value_head_evaluations=2*len(units),policy_evaluations=len(units),model_view_evaluations=2*len(units),target_records=records_count))
    validate_cached(batch,units,model_identity);return batch


def validate_cached(batch,units=None,model_identity=None):
    """Structural/source admission only; independent certificate proves numerics.

    No encoder, model parsing/forward, gradient, NumPy or SVD is called here.
    Production completed readers additionally require the external certificate.
    """
    require(isinstance(batch,dict) and set(batch)=={'schema','source_units_sha256','models','runtime','observations','counts'} and batch['schema']==SCHEMA+'-batch','Unknown cached batch schema')
    require(isinstance(batch['observations'],dict) and 0<len(batch['observations'])<=SETTINGS['max_units'],'Cached units missing/out of bounds')
    require(same(batch['runtime']['settings'],SETTINGS) and batch['runtime']['numpy']==SETTINGS['numpy_version'] and batch['runtime']['threadpoolctl']==SETTINGS['threadpoolctl_version'],'Cached numerical settings differ')
    require(batch['models']['shared_trunk_policy_bit_identical'] is True and set(batch['models']['models'])==set(NAMES),'Cached fixed-model identity differs')
    if model_identity is not None:require(same(model_identity,batch['models']),'Expected fixed-model identity differs')
    sources={};records=set();states=set()
    for uid,item in batch['observations'].items():
        require(isinstance(item,dict) and item['schema']==SCHEMA+'-observation' and item['unit_id']==uid and item['model_identity_sha256']==digest(batch['models']),'Cached observation identity differs')
        unit=item['source_unit'];sources[uid]=unit;require(item['source_unit_sha256']==digest(unit) and item['feature_sha256']==digest(unit['features']) and item['state_key']==unit['state_key'],'Cached source linkage differs')
        require(uid==digest(dict(state_key=unit['state_key'],features=unit['features'])) and item['state_key'] not in states,'Cached unit identity/duplicate state differs');states.add(item['state_key'])
        shared=item['shared'];require(all(_vector(shared[k],n) for k,n in [('layer0_pre_activation',64),('layer0_hidden',64),('layer2_pre_activation',64),('hidden',64),('policy_logits',7),('policy_probabilities',7)]),'Cached activation/output shape differs')
        require(shared['policy_sha256']==digest(shared['policy_logits']) and all(isinstance(shared[k],list) and len(shared[k])==64 and all(type(v) is bool for v in shared[k]) for k in ('layer0_kink','layer2_kink','layer0_relu_active','layer2_relu_active')),'Cached policy/kink identity differs')
        require(set(item['models'])==set(NAMES) and set(item['targets'])=={v['id'] for v in unit['records']},'Cached targets/models differ')
        originals={v['id']:v for v in unit['records']}
        for name,head in item['models'].items():
            require(all(number(head[k]) for k in ('preactivation','tanh','value_player0','normalized_output_gain','seed_output_gain')) and -1<=head['tanh']<=1 and type(head['value_units']) is int and type(head['saturated']) is bool,'Cached head scalar differs')
            require(all(_vector(head[k],n) for k,n in [('input_jacobian_normalized',127),('input_jacobian_player0',127),('head_jacobian_normalized',65),('head_jacobian_player0',65)]),'Cached sensitivity shape differs')
        for rid,target in item['targets'].items():
            require(rid not in records and same(target['record'],originals[rid]) and target['record_sha256']==digest(originals[rid]) and set(target['models'])==set(NAMES),'Cached scalar source/duplicate record differs');records.add(rid)
            for name,fit in target['models'].items():
                require(fit['target_player0']==originals[rid]['target_player0'] and integer(fit['target_player0']) and fit['target_status'] in ('strict_interior','endpoint') and _vector(fit['residual_huber_gradient_head'],65),'Cached target/sensitivity fields differ')
                require(fit['error_player0']==item['models'][name]['value_player0']-fit['target_player0'] and fit['absolute_error_player0']==abs(fit['error_player0']),'Cached error arithmetic differs')
    require(batch['source_units_sha256']==digest(sources),'Cached source-set checksum differs')
    if units is not None:require(same(sources,units),'Expected cached source units differ')
    count=len(sources);expected=dict(distinct_units=count,shared_trunk_forwards=count,value_head_evaluations=2*count,policy_evaluations=count,model_view_evaluations=2*count,target_records=len(records))
    require(same(batch['counts'],expected) and len(records)<=SETTINGS['max_target_records'],'Cached physical/target counts differ');return batch


def record_weights(records,mode):
    """Exact target-record masses, retaining repeated outcomes at a raw root."""
    require(records and mode in ('raw_root_balanced','family_balanced'),'Unknown/empty weighting definition')
    roots={};families={}
    for record in records:
        roots.setdefault(record['state_key'],[]).append(record['id']);families.setdefault(record['family'],set()).add(record['state_key'])
    require(all(len({r['family'] for r in records if r['state_key']==key})==1 for key in roots),'Raw root has conflicting family ownership within target group')
    result={}
    for record in records:
        n=len(roots[record['state_key']]);result[record['id']]=F(1,len(roots)*n) if mode=='raw_root_balanced' else F(1,len(families)*len(families[record['family']])*n)
    require(sum(result.values(),F())==1,'Target-record mass is not conserved');return result


def _weighted(values,weights):return sum(float(p)*float(v) for p,v in zip(weights,values))


def _concentration(records,weights):
    root={};family={}
    for record,p in zip(records,weights):
        key=record['state_key'];root[key]=root.get(key,F())+p;key=record['family'];family[key]=family.get(key,F())+p
    return dict(record_mass=[dict(record_id=r['id'],probability=rational(p)) for r,p in zip(records,weights)],
        record_effective_support=rational(1/sum((p*p for p in weights),F())),raw_root_effective_support=rational(1/sum((p*p for p in root.values()),F())),family_effective_support=rational(1/sum((p*p for p in family.values()),F())),
        maximum_record_probability=rational(max(weights)),maximum_root_probability=rational(max(root.values())),maximum_family_probability=rational(max(family.values())),
        interpretation='Distribution concentration only; not independent samples, importance-sampling ESS, confidence or effective games.')


def projection_diagnostic(hidden,normalized_targets,weights):
    """Residual-only weighted projection of interior artanh targets onto [h,1].

    No affine coefficients are constructed, optimized, returned or persisted.
    The sample-space U projection is not a seed-error capacity bound.
    """
    require(len(hidden)==len(normalized_targets)==len(weights)>0 and all(_vector(v,64) for v in hidden),'Projection dimensions differ')
    require(all(type(z) is F and abs(z)<=1 for z in normalized_targets) and all(type(p) is F and p>0 for p in weights) and sum(weights,F())==1,'Projection target/mass identity differs')
    interior=[i for i,z in enumerate(normalized_targets) if abs(z)<1];mass=sum((weights[i] for i in interior),F());endpoint_mass=1-mass
    base=dict(design='cached_frozen_hidden64_plus_intercept',target='artanh((target_player0-d)/signed_active_r)',dtype='float64',rcond=SETTINGS['svd_rcond'],selected_records=len(hidden),
        strict_interior_records=len(interior),endpoint_records=len(hidden)-len(interior),original_selected_mass=rational(1),original_interior_mass=rational(mass),original_endpoint_mass=rational(endpoint_mass),
        endpoints_clipped=False,coefficients_constructed=False,conditioning_scope='Identified retained singular subspace only.',
        interpretation='In-sample preactivation/logit-space representability at frozen hidden features; not a seed-error lower bound, trained head, generalization or strength evidence.')
    if not interior:return dict(base,available=False,rank=None,singular_values=[],weighted_logit_projection_residual_rmse=None,full_row_rank=None,small_sample_interpolation_warning=None)
    np=numpy_runtime();p=np.asarray([float(weights[i]/mass) for i in interior],dtype=np.float64);a=np.asarray([hidden[i]+[1.] for i in interior],dtype=np.float64);b=np.asarray([math.atanh(float(normalized_targets[i])) for i in interior],dtype=np.float64)
    aw=a*np.sqrt(p)[:,None];bw=b*np.sqrt(p);u,s,_discard=np.linalg.svd(aw,full_matrices=False);cut=SETTINGS['svd_rcond']*float(s[0]) if len(s) else 0.;rank=int(np.count_nonzero(s>cut));basis=u[:,:rank]
    residual=bw-basis@(basis.T@bw);full=rank==len(interior)
    return dict(base,available=True,rank=rank,columns=65,rows=len(interior),singular_values=s.tolist(),rank_cutoff=cut,identified_condition_number=float(s[0]/s[rank-1]) if rank else None,
        column_rank_deficient=rank<65,full_row_rank=full,small_sample_interpolation_warning=bool(full and len(interior)<=65),
        weighted_logit_projection_residual_rmse=math.sqrt(float(residual@residual)),weighted_logit_target_rms=math.sqrt(float(bw@bw)))


def feature_geometry(vectors,weights):
    """Centered weighted feature geometry; repeats retain their declared mass."""
    require(vectors and len(vectors)==len(weights) and all(_vector(v,len(vectors[0])) for v in vectors),'Feature geometry shape differs')
    np=numpy_runtime();x=np.asarray(vectors,dtype=np.float64);p=np.asarray([float(v) for v in weights],dtype=np.float64);mean=p@x;centered=x-mean;variance=p@(centered*centered);a=centered*np.sqrt(p)[:,None]
    singular=np.linalg.svd(a,compute_uv=False);cut=SETTINGS['svd_rcond']*float(singular[0]) if len(singular) else 0.;rank=int(np.count_nonzero(singular>cut))
    return dict(rows=len(vectors),columns=x.shape[1],centered=True,weighted_mean=mean.tolist(),weighted_variance=variance.tolist(),exact_zero_variance_columns=[int(i) for i,v in enumerate(variance) if v==0],
        rank=rank,rank_cutoff=cut,singular_values=singular.tolist(),identified_condition_number=float(singular[0]/singular[rank-1]) if rank else None,full_column_rank=rank==x.shape[1],
        interpretation='Descriptive geometry of this fixed feature support; correlation/rank is not causal feature importance or predictive capacity.')


def _gradient_cancellation(gradients,weights):
    means=[sum(float(p)*g[j] for p,g in zip(weights,gradients)) for j in range(65)];denom=_weighted([_norm(g) for g in gradients],weights);numerator=_norm(means)
    return dict(weighted_mean_gradient_norm=numerator,weighted_mean_individual_gradient_norm=denom,cancellation_ratio=numerator/denom if denom else None,zero_denominator=denom==0,
        scope='Fixed-parameter normalized-residual Huber head gradients; descriptive cancellation, no optimizer and not actual historical training gradients.')


def _risk(records,observations,weights,name):
    fits=[observations[r['unit_id']]['targets'][r['id']]['models'][name] for r in records];heads=[observations[r['unit_id']]['models'][name] for r in records]
    mean=lambda key:_weighted([fit[key] for fit in fits],weights)
    errors=[fit['error_player0'] for fit in fits];normalized=[fit['normalized_residual_error'] for fit in fits]
    result=dict(mae_player0=mean('absolute_error_player0'),rmse_player0=math.sqrt(_weighted([e*e for e in errors],weights)),bias_player0=_weighted(errors,weights),
        normalized_mae=_weighted([abs(v) for v in normalized],weights),normalized_rmse=math.sqrt(_weighted([v*v for v in normalized],weights)),residual_huber_loss=mean('residual_huber_loss'),
        model_saturated_mass=_weighted([int(h['saturated']) for h in heads],weights),target_near_boundary_mass=_weighted([int(v['target_near_boundary']) for v in fits],weights),
        seed_gain_signed_mean=_weighted([h['seed_output_gain'] for h in heads],weights),seed_gain_absolute_mean=_weighted([abs(h['seed_output_gain']) for h in heads],weights),
        normalized_gain_mean=_weighted([h['normalized_output_gain'] for h in heads],weights),
        normalized_jacobian_norm_means={k:_weighted([h['normalized_jacobian_norms'][k] for h in heads],weights) for k in heads[0]['normalized_jacobian_norms']},
        seed_jacobian_norm_means={k:_weighted([h['seed_jacobian_norms'][k] for h in heads],weights) for k in heads[0]['seed_jacobian_norms']},
        head_gradient_cancellation=_gradient_cancellation([v['residual_huber_gradient_head'] for v in fits],weights))
    families={}
    for family in sorted({r['family'] for r in records}):
        indices=[i for i,r in enumerate(records) if r['family']==family];mass=sum((weights[i] for i in indices),F());families[family]=dict(original_mass=rational(mass),
            gradients=_gradient_cancellation([fits[i]['residual_huber_gradient_head'] for i in indices],[weights[i]/mass for i in indices]))
    result['by_family_head_gradient_cancellation']=families;return result


def _group_summary(records,observations):
    require(records,'Empty target group');units=sorted({r['unit_id'] for r in records});target_status={}
    for r in records:
        fit=observations[r['unit_id']]['targets'][r['id']]['models']['control'];target_status[fit['target_status']]=target_status.get(fit['target_status'],0)+1
    result=dict(records=len(records),distinct_raw_roots=len({r['state_key'] for r in records}),families=len({r['family'] for r in records}),distinct_forward_units=len(units),
        source_record_ids=[r['id'] for r in records],target_status_counts=target_status,weightings={})
    for mode in ('raw_root_balanced','family_balanced'):
        mapping=record_weights(records,mode);weights=[mapping[r['id']] for r in records];hidden=[observations[r['unit_id']]['shared']['hidden'] for r in records];features=[observations[r['unit_id']]['source_unit']['features']['x'] for r in records]
        targets=[F(v['numerator'],v['denominator']) for v in (observations[r['unit_id']]['targets'][r['id']]['models']['control']['target_normalized'] for r in records)]
        result['weightings'][mode]=dict(concentration=_concentration(records,weights),models={name:_risk(records,observations,weights,name) for name in NAMES},
            head_logit_projection=projection_diagnostic(hidden,targets,weights),encoded_feature_geometry=feature_geometry(features,weights),frozen_hidden_geometry=feature_geometry(hidden,weights))
    result['subgroups']={}
    for axis in ('stratum','actor'):
        result['subgroups'][axis]={}
        for value in sorted({str(r[axis]) for r in records}):
            subset=[r for r in records if str(r[axis])==value];mapping=record_weights(subset,'raw_root_balanced');weights=[mapping[r['id']] for r in subset]
            result['subgroups'][axis][value]=dict(records=len(subset),raw_roots=len({r['state_key'] for r in subset}),families=len({r['family'] for r in subset}),models={name:_risk(subset,observations,weights,name) for name in NAMES})
    return result


def summarize_cached(batches):
    """Initial/audit numerical summary only. NEVER call on public completed path.

    This consumes cached activations/gradients; it runs no encoder or forward.
    SVD is computed here once before completion and independently in the audit.
    """
    require(isinstance(batches,list) and batches,'A nonempty cached batch list is required');observations={};records=[];model_identity=None;runtime=None;statekeys=set()
    for batch in batches:
        validate_cached(batch)
        if model_identity is None:model_identity=batch['models'];runtime=batch['runtime']
        require(same(model_identity,batch['models']) and same(runtime,batch['runtime']),'Batch model/runtime identity differs')
        for uid,item in sorted(batch['observations'].items()):
            require(uid not in observations and item['state_key'] not in statekeys,'Duplicate forward unit/raw state across batches');observations[uid]=item;statekeys.add(item['state_key'])
            for rid,entry in sorted(item['targets'].items()):records.append(dict(entry['record'],unit_id=uid))
    require(len(observations)<=SETTINGS['max_units'] and len(records)<=SETTINGS['max_target_records'] and len({r['id'] for r in records})==len(records),'Global unit/target record bound or uniqueness differs');records.sort(key=lambda r:r['id'])
    groups={}
    for cohort,target_type in sorted({(r['cohort'],r['target_type']) for r in records}):
        subset=[r for r in records if r['cohort']==cohort and r['target_type']==target_type];groups[cohort+'/'+target_type]=_group_summary(subset,observations)
    activation={}
    for layer in ('layer0','layer2'):
        activation[layer]=dict(active_units_per_neuron=[sum(item['shared'][layer+'_relu_active'][i] for item in observations.values()) for i in range(64)],
            near_kink_units_per_neuron=[sum(item['shared'][layer+'_kink'][i] for item in observations.values()) for i in range(64)],
            never_active_neurons=[i for i in range(64) if not any(item['shared'][layer+'_relu_active'][i] for item in observations.values())],units=len(observations))
    pairs=[];mc=[]
    for uid,item in sorted(observations.items()):
        entries=[entry['record'] for entry in item['targets'].values()];paired={r['target_type']:r for r in entries if r['cohort']=='paired_H8_H10'}
        h8=paired.get('approximate_H8_root_margin');h10=paired.get('approximate_H10_root_margin')
        if h8 is not None and h10 is not None:pairs.append(dict(unit_id=uid,state_key=item['state_key'],family=h8['family'],H8=h8['target_player0'],H10=h10['target_player0'],signed_shift_player0=h10['target_player0']-h8['target_player0'],absolute_shift_seeds=abs(h10['target_player0']-h8['target_player0'])))
        outcomes=[r for r in entries if r['cohort']=='ordinary_MC']
        if outcomes:
            values=[r['target_player0'] for r in outcomes];mc.append(dict(unit_id=uid,state_key=item['state_key'],family=outcomes[0]['family'],records=len(outcomes),returns_player0=values,empirical_spread_seeds=max(values)-min(values),
                interpretation='Dependent same-start outcomes from fixed control policy; empirical two-trajectory conflict, not a calibrated noise floor or generalization bound.'))
    count=len(observations)
    return dict(schema=SCHEMA+'-summary',models=model_identity,runtime=runtime,batch_sha256=[digest(v) for v in batches],
        counts=dict(distinct_units=count,shared_trunk_forwards=count,value_head_evaluations=2*count,policy_evaluations=count,model_view_evaluations=2*count,target_records=len(records),batches=len(batches)),
        observation_sha256={uid:digest(item) for uid,item in sorted(observations.items())},groups=groups,activation_coverage=activation,paired_horizon_targets=pairs,ordinary_MC_within_start=mc,
        scope=dict(parameters_fixed=True,policy_trunk_six_tensors_bit_identical=True,coefficients_constructed=False,models_exported=False,optimizer_steps=0,
            numeric_contract='Stored FP32-rounded weights, unrounded cached float64 inputs, float64 export algebra; not bit parity with historical Torch FP32 training.',
            target_contract='Separate cohorts and target estimands; no merged finite-horizon/MC risk or projection.',
            completed_contract='Public completed readers use independently certified persisted summary only; no feature/forward/gradient/SVD calls.',
            sensitivity_contract='Local continuous encoded-feature and fixed-parameter residual-Huber sensitivity; not board-state causal importance or confidence.',
            support_contract='Retrospective eligible training-source diagnostic support with family dependence; no strength, promotion, training solution or capacity-fix claim.'))
