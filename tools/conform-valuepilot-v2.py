"""Conform the unchanged-policy sidecar on its actual admitted v2 corpus.

No labels are extracted, optimizer constructed, weights updated or games played.
The two model exports used by this check are explicitly engineering fixtures.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import struct
import sys

for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,neurodiag_common as common,value_sidecar_v2 as side
from congklak import valuepilot_reservations_v2 as data
from congklak import quality_search
from congklak.neural_relay import Model as BaseModel,rounded
from congklak.puct_cpu_inference import CpuBatchEvaluator

FOLDER=ROOT/'build/ordinary-development-v1'
WORK=FOLDER/'conformance-owner-fixtures'
REPORT=side.CONFORMANCE
CORPUS=FOLDER/'conformance-owner-corpus.json'
SOURCES=('congklak/value_sidecar_v2.py','tools/export-value-sidecar-v2.py',
         'tools/conform-valuepilot-v2.py','tests/test_value_sidecar_v2.py',
         'congklak/valuepilot_reservations_v2.py','tools/reserve-valuepilot-v2.py')

def boundary(active,actor=0,forced=False):
    pits=[[0]*7,[0]*7]
    if forced:pits[1-actor][0]=active
    else:
        for i in range(active):pits[(i%14)//7][i%7]+=1
    return dict(pits=pits,stores=[(98-active)//2,98-active-(98-active)//2],controller=actor)

def corpus(manifest):
    manifest=Path(manifest).resolve();admitted=data.admit(manifest.parent)
    common.require(manifest==data.DEFAULT.resolve()/'manifest.json','Conformance requires actual v2 production reservations')
    selection=common.read(manifest.parent/'selection.json')
    selected=[]
    for role,count in (('train',512),('fit_check',128),('arena',24)):
        common.require(len(selection[role])==count,'Incomplete selected role')
        for row in selection[role]:
            selected.append(dict(role=role,state=row['state'],state_key=row['state_key'],
                component=row['component'],family=row['family'],stratum=row['stratum']))
    common.require(len({r['state_key'] for r in selected})==len({r['component'] for r in selected})==664,
                   'Admitted roles overlap')
    # Metadata projections deliberately do not decode reference/target bodies.
    exact=data.old._json_metadata('build/ordinary-value-benchmark-projection.json','cases')
    forensic=data.old._json_metadata('build/divergence-panel.json','roots')
    common.require(len(exact)==32 and len(forensic)==5,'Preserved guard cardinality differs')
    guards=[dict(role=role,state=r['state'],state_key=r['state_key'],family=r['family'],source=source)
            for role,rows,source in (('reused_exact_guard',exact,'build/ordinary-value-benchmark-projection.json'),
                                    ('reused_Hv1_forensic',forensic,'build/divergence-panel.json')) for r in rows]
    boundaries=[dict(role='synthetic_boundary',state=boundary(a,c,forced),active=a,actor=c,forced_pass=forced)
        for forced,actives in ((False,(0,4,5,24,25,49,50,74,75,98)),(True,(4,24,25,49,74,75)))
        for a in actives for c in (0,1)]
    return dict(schema='valuepilot-admitted-conformance-corpus-v2',manifest={'path':str(manifest),'sha256':common.sha(manifest)},
                reservation_bindings=side.reservation_bindings(manifest),selected=selected,guards=guards,boundaries=boundaries,
                source_inputs=common.pins([manifest.parent/n for n in
                    ('manifest.json','identity.json','inventory.json','selection.json','catalog.json','certificate.pending.json')]+
                    [ROOT/'build/ordinary-value-benchmark-projection.json',ROOT/'build/divergence-panel.json']),
                targets_read=0,optimizer_steps=0)

def positions(panel):return [row['state'] for group in ('selected','guards','boundaries') for row in panel[group]]

def literal(base,weights,p):
    active=sum(map(sum,p['pits']))
    if not 25<=active<=74:return base.reference(p)
    x,d,r=side.features_relay.encode(p,127);h=x
    for prefix in ('layers.0','layers.2'):
        h=[max(0.,sum(a*b for a,b in zip(row,h))+bias)
           for row,bias in zip(base.weights[prefix+'.weight'],base.weights[prefix+'.bias'])]
    u=sum(a*b for a,b in zip(base.weights['layers.4.weight'][0],h))+base.weights['layers.4.bias'][0]
    policy=[sum(a*b for a,b in zip(row,h))+bias for row,bias in zip(base.weights['policy.weight'],base.weights['policy.bias'])]
    hidden=[max(0.,sum(a*b for a,b in zip(row,x))+bias) for row,bias in zip(weights['A'],weights['b'])]
    residual=r*math.tanh(u+sum(a*b for a,b in zip(weights['v'][0],hidden))+weights['c'][0])
    return d+residual,d*256+rounded(residual*256),policy

def native_check(initial,synthetic,rows):
    maxima=dict(zero_seed=0.,nonzero_seed=0.,logit=0.,score_units=0)
    policy_comparisons=outside=changed=single_calls=0;backends=[];scalar_results=[]
    for backend in ('scalar','avx2'):
        with CpuBatchEvaluator(side.BASE,backend=backend) as base,side.SidecarEvaluator(initial,backend=backend) as zero,side.SidecarEvaluator(synthetic,backend=backend) as candidate:
            backends.append(candidate.backend)
            for start in range(0,len(rows),32):
                group=rows[start:start+32];aa=base.evaluate_many(group);zz=zero.evaluate_many(group);bb=candidate.evaluate_many(group)
                for index,(p,a,z,b) in enumerate(zip(group,aa,zz,bb),start):
                    policy_comparisons+=1
                    assert struct.pack('<7d',*a['policy_logits'])==struct.pack('<7d',*z['policy_logits'])==struct.pack('<7d',*b['policy_logits'])
                    assert a['value_player0']==z['value_player0']
                    for key in ('feature_simulations','feature_deposits','native_actual_evals'):
                        assert a[key]==z[key]==b[key],key
                    rz=literal(initial.base,initial.weights,p);rb=literal(initial.base,synthetic.weights,p)
                    maxima['zero_seed']=max(maxima['zero_seed'],abs(z['value_player0']-rz[0]))
                    maxima['nonzero_seed']=max(maxima['nonzero_seed'],abs(b['value_player0']-rb[0]))
                    maxima['logit']=max(maxima['logit'],max(abs(x-y) for x,y in zip(b['policy_logits'],rb[2])))
                    assert abs(z['value_player0']-rz[0])<=1e-10 and abs(b['value_player0']-rb[0])<=1e-10
                    assert max(abs(x-y) for x,y in zip(b['policy_logits'],rb[2]))<=1e-10
                    units=candidate.evaluate_units(p);maxima['score_units']=max(maxima['score_units'],abs(units['value_units']-rb[1]))
                    assert units['value_units']==rb[1] and zero.evaluate_units(p)['value_units']==rz[1]
                    if not 25<=sum(map(sum,p['pits']))<=74:
                        outside+=1;assert a['value_player0']==b['value_player0']
                    else:changed+=int(a['value_player0']!=b['value_player0'])
                    one=candidate.evaluate_many([p])[0];single_calls+=1
                    normalized={k:b[k] for k in ('value_player0','policy_logits','feature_simulations','feature_deposits')}
                    assert normalized=={k:one[k] for k in normalized}
                    if backend=='scalar':scalar_results.append(normalized)
                    else:assert normalized==scalar_results[index]
            assert candidate.snapshot_counters()['branch_calls']>0 and candidate.snapshot_counters()['branch_bypasses']>0
    assert changed>0
    return dict(passed=True,rows=len(rows),backends=backends,policy_comparisons=policy_comparisons,
        policy_bit_exact=True,zero_branch_value_and_units_exact=True,scalar_single_batch_avx2_equal=True,
        single_row_parity_calls=single_calls,outside_gate_direct_equality=outside,synthetic_in_gate_changed=changed,
        additional_feature_simulations=0,maximum_errors=maxima,native_literal_tolerance=1e-10)

def torch_check(initial,synthetic,rows):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    generator=torch.Generator(device='cpu');generator.manual_seed(20261016);bound=1/math.sqrt(127)
    assert initial.weights['A']==torch.empty((16,127),dtype=torch.float32,device='cpu').uniform_(-bound,bound,generator=generator).tolist()
    assert initial.weights['b']==torch.empty((16,),dtype=torch.float32,device='cpu').uniform_(-bound,bound,generator=generator).tolist()
    assert side.tensor_sha(initial.weights)==side.INITIAL_TENSOR_SHA
    base={k:torch.tensor(v,dtype=torch.float32,device='cpu') for k,v in initial.base.weights.items()}
    maxima=dict(seeds=0.,policy_logits=0.,score_units=0);compared=0
    with torch.no_grad():
        for model in (initial,synthetic):
            branch={k:torch.tensor(v,dtype=torch.float32,device='cpu') for k,v in model.weights.items()}
            for p in rows:
                x,d,r=side.features_relay.encode(p,127)
                if not r:continue
                compared+=1;x=torch.tensor(x,dtype=torch.float32,device='cpu');h=x
                for prefix in ('layers.0','layers.2'):h=torch.relu(torch.mv(base[prefix+'.weight'],h)+base[prefix+'.bias'])
                u=(torch.mv(base['layers.4.weight'],h)+base['layers.4.bias'])[0]
                before=torch.mv(base['policy.weight'],h)+base['policy.bias']
                if 25<=sum(map(sum,p['pits']))<=74:u=u+(torch.mv(branch['v'],torch.relu(torch.mv(branch['A'],x)+branch['b']))+branch['c'])[0]
                after=torch.mv(base['policy.weight'],h)+base['policy.bias'];assert torch.equal(before,after)
                residual=float((r*torch.tanh(u)).item());ref=literal(initial.base,model.weights,p)
                maxima['seeds']=max(maxima['seeds'],abs(d+residual-ref[0]))
                maxima['policy_logits']=max(maxima['policy_logits'],max(abs(a-b) for a,b in zip(after.tolist(),ref[2])))
                maxima['score_units']=max(maxima['score_units'],abs(d*256+rounded(residual*256)-ref[1]))
    assert maxima['seeds']<=.001 and maxima['policy_logits']<=1e-5 and maxima['score_units']<=1,maxima
    assert not torch.cuda.is_initialized()
    # Synthetic derivative check only: all branch entries are checked against
    # algebra, with no training labels, optimizer or parameter update.
    x=torch.linspace(-.1,.4,127,dtype=torch.float64,device='cpu')
    branch={k:torch.tensor(v,dtype=torch.float64,device='cpu',requires_grad=True) for k,v in synthetic.weights.items()}
    pre=branch['A']@x+branch['b'];hidden=torch.relu(pre)
    out=torch.tanh(.2+(branch['v']@hidden+branch['c'])[0]);out.backward()
    factor=1-float(out.detach())**2;gradient=factor*branch['v'].detach()[0]*(pre.detach()>0)
    assert torch.allclose(branch['A'].grad,gradient[:,None]*x[None,:],atol=1e-14,rtol=0)
    assert torch.allclose(branch['b'].grad,gradient,atol=1e-14,rtol=0)
    assert torch.allclose(branch['v'].grad,factor*hidden.detach()[None,:],atol=1e-14,rtol=0)
    assert abs(float(branch['c'].grad[0])-factor)<=1e-14
    return dict(passed=True,rows=len(rows),zero_and_nonzero_nonterminal_comparisons=compared,maximum_errors=maxima,
        tolerances=dict(seeds=.001,policy_logits=1e-5,score_units=1),policy_bit_exact=True,device='cpu',dtype='float32',
        torch=torch.__version__,threads=1,interop_threads=1,initial_generator_exact=True,
        analytic_scratch_derivative_entries=2065,optimizer_steps=0,cuda_initialized=False)

def low_active_check(synthetic,panel):
    rows=[r['state'] for r in panel['guards'] if r['role']=='reused_exact_guard']
    rows += [boundary(a,c) for a in (4,5,24) for c in (0,1)]+[boundary(24,c,True) for c in (0,1)]
    reports=[]
    with CpuBatchEvaluator(side.BASE,backend='scalar') as control,side.SidecarEvaluator(synthetic,backend='scalar') as candidate:
        for p in rows:
            assert sum(map(sum,p['pits']))<=24
            control.begin_decision();candidate.begin_decision()
            a=quality_search.search(p,control,simulations=128,c_puct=.25);b=quality_search.search(p,candidate,simulations=128,c_puct=.25)
            quality_search.validate_report(a,p,128,c_puct=.25);quality_search.validate_report(b,p,128,c_puct=.25)
            for k in a:
                if k not in ('timing','work'):assert a[k]==b[k],k
            for k in a['work']:
                if k!='inference_ms':assert a['work'][k]==b['work'][k],k
            reports.append(dict(state=p,control=a,synthetic=b))
    side._atomic(FOLDER/'conformance-owner-low-active.json',reports)
    return dict(passed=True,roots=len(rows),searches=len(rows)*2,requested_backups=len(rows)*256,
        completed_backups=sum(r[sidekey]['completed_simulations'] for r in reports for sidekey in ('control','synthetic')),
        actions_visits_backed_values_physical_work_exact=True,protocol=quality_search.PROTOCOL,
        ignored_fields=['timing','work.inference_ms'],proof_sha256=common.sha(FOLDER/'conformance-owner-low-active.json'))

def verify(manifest=data.DEFAULT/'manifest.json'):
    report=common.read(REPORT)
    common.require(report['passed'] is True and report['schema']=='valuepilot-admitted-corpus-conformance-v2','Conformance incomplete')
    common.current(report['inputs'])
    panel=corpus(manifest)
    common.require(common.read(CORPUS)==panel,'Admitted conformance corpus changed')
    side.conformance_admission(manifest)
    common.require(report['corpus_rows']==len(positions(panel)) and report['guard_counts']==dict(reused_exact_guard=32,reused_Hv1_forensic=5,synthetic_boundary=32),
                   'Conformance corpus coverage differs')
    return report

def run(manifest=data.DEFAULT/'manifest.json'):
    # A complete report is consumed read-only, before model locks or Torch.
    if REPORT.exists():return verify(manifest)
    panel=corpus(manifest);rows=positions(panel)
    FOLDER.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    # These are engineering exports even though they bind admitted membership.
    side._atomic(CORPUS,panel)
    side.initialize(manifest,WORK/'initial',fixture=True)
    initial=side.Model(WORK/'initial/model.json')
    weights=copy.deepcopy(initial.weights);weights['v']=[[(i-7)/128 for i in range(16)]];weights['c']=[.03125]
    synthetic_path=WORK/'synthetic.json'
    side._atomic(synthetic_path,side.make_export(weights,manifest,{'engineering_fixture':True,'training_eligible':False,'optimizer_steps':0,'purpose':'nonzero-branch-conformance-only'}))
    synthetic=side.Model(synthetic_path)
    print(config.canonical(dict(event='actual_corpus_frozen',selected=664,guards=37,boundaries=32,total=len(rows),optimizer_steps=0)),flush=True)
    native=native_check(initial,synthetic,rows)
    print(config.canonical(dict(event='native_conformance_complete',rows=len(rows),backends=native['backends'])),flush=True)
    fp32=torch_check(initial,synthetic,rows)
    low=low_active_check(synthetic,panel)
    inputs=common.pins([ROOT/n for n in SOURCES]+[CORPUS,WORK/'initial/model.json',WORK/'initial/certificate.json',WORK/'initial/COMPLETED.json',synthetic_path,FOLDER/'conformance-owner-low-active.json',side.BUILD,side.DLL,side.BASE])
    inputs.update(panel['source_inputs']);inputs.update(common.pins([ROOT/n for n in side.DEPENDENCIES]))
    report=dict(schema='valuepilot-admitted-corpus-conformance-v2',passed=True,manifest=panel['manifest'],
        reservation_bindings=panel['reservation_bindings'],interface_bindings=initial.identity,selected_counts=dict(train=512,fit_check=128,arena=24),
        unique_selected_states=664,unique_selected_components=664,guard_counts=dict(reused_exact_guard=32,reused_Hv1_forensic=5,synthetic_boundary=32),
        corpus_rows=len(rows),corpus_sha256=common.sha(CORPUS),initial_tensor_sha256=side.INITIAL_TENSOR_SHA,
        checks=dict(native=native,torch_fp32=fp32,low_active_fixed128=low,
                    initialization=dict(passed=True,branch_parameters=2065,base_parameters_frozen=12872,base_tensors_frozen=8,
                        initial_tensor_sha256=side.tensor_sha(initial.weights),zero_output=True,optimizer_steps=0)),
        inputs=inputs,policy_fixed=True,optimizer_steps=0,cuda_initialized=False,
        new_teacher_labels=0,new_training_games=0,scope='admitted-corpus interface conformance; not strength evidence',
        production_export_written=False)
    side._atomic(REPORT,report)
    return verify(manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('run','verify'))
    parser.add_argument('--manifest',default=str(data.DEFAULT/'manifest.json'));args=parser.parse_args()
    try:
        report=run(args.manifest) if args.operation=='run' else verify(args.manifest)
        print(config.canonical({k:report[k] for k in ('schema','passed','selected_counts','corpus_rows','policy_fixed','optimizer_steps','cuda_initialized')}),flush=True)
        return 0
    except Exception as exc:
        print(config.canonical(dict(passed=False,state='hold',error=str(exc),optimizer_steps=0)),flush=True)
        return 2
if __name__=='__main__':raise SystemExit(main())
