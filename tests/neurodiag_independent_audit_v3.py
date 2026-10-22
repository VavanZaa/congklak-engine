"""Independent existing-data neural value diagnostics and certified cache audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction as F
import argparse, ast, copy, hashlib, importlib.util, json, math, os, re
from pathlib import Path
import shutil, sqlite3, struct, sys, time, unittest, uuid
from unittest import mock

# Set these before any local or third-party NumPy import, including subprocesses.
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from capture_independent_audit import (read,sha,save,rel,pins,current,durable,files,qread,
    rows,physical_rows,backup,mutate_row,run_child,kill,pos,state,raw,digest,canonical)
import reference as literal

WORK=ROOT/'build/neurodiag-audit'
FREEZE=ROOT/'build/neurodiag-audit-source-freeze-v3.json'
ORIGINAL_MAIN=ROOT/'tests/neurodiag_independent_audit.py'
ORIGINAL_FREEZE=ROOT/'build/neurodiag-audit-source-freeze.json'
FAILURE_ARCHIVE=ROOT/'build/neurodiag-audit-production-role-failure-v1'
REPAIR=ROOT/'build/neurodiag-audit-harness-repair.json'
V2_MAIN=ROOT/'tests/neurodiag_independent_audit_v2.py'
V2_FREEZE=ROOT/'build/neurodiag-audit-source-freeze-v2.json'
FINAL_FAILURE=ROOT/'build/neurodiag-audit-final-lock-failure-v1'
COMPLETION_FREEZE=ROOT/'build/neurodiag-completion-implementation.json'
BASELINE=ROOT/'build/neurodiag-baseline.json'
WITNESS=WORK/'observation-witness.json'
STRATA=('50-74/ordinary','25-49/ordinary','25-49/bonus','25-49/capture')
CELLS=tuple(s+'/actor'+str(a) for s in STRATA for a in (0,1))
EVIDENCE={}
GROUPS={
 'preflight':['test_01_training_roles_all_origins_protection_closure_and_metadata_selection',
              'test_02_frozen_forward_derivatives_and_projection_contract',
              'test_03_live_batch_commit_recovery_and_completed_certification_gate'],
 'production':['test_04_actual_panel_cache_scalar_forward_and_targets',
               'test_05_analytic_gradients_svd_summaries_and_observation_witness'],
 'final':['test_06_corruption_repair_and_no_forward_completed_idempotence']}


def quotient(p):
    counts=p['pits'][0]+p['pits'][1];actor=p['controller'] if any(counts) else 0
    return min(bytes(counts+[actor]),bytes(counts[7:]+counts[:7]+[1-actor if any(counts) else 0])).hex()


def literal_features(p):
    """Independent exact-choice encoder; no Native or accepted encoder calls."""
    active=sum(map(sum,p['pits']));actor=p['controller']
    if active and not sum(p['pits'][actor]):actor=1-actor
    counts=p['pits'][actor]+p['pits'][1-actor]
    x=[v/98 for v in counts]+[active/98]
    x += [float(v>0) for v in counts]+[float(v==1) for v in counts]
    for side in (actor,1-actor):
        for pit,count in enumerate(p['pits'][side]):
            if not count:x += [0.0]*6;continue
            before=literal.Position(7,tuple(p['pits'][0]+[p['stores'][0]]+p['pits'][1]+[p['stores'][1]]),side)
            transition=literal.apply(before,pit)
            after=state(transition.position)
            x += [(after['stores'][side]-p['stores'][side])/98,
                  transition.deposits/98,transition.relay_pickups/98,float(transition.captured>0),
                  float(not transition.terminal and after['controller']==side),float(transition.terminal)]
    assert len(x)==127
    return dict(x=x,d=p['stores'][0]-p['stores'][1],r=active*(1 if actor==0 else -1))


def fp32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def frozen_weights(path):
    envelope=read(path);payload=envelope['model'];assert envelope['sha256']==digest(payload)
    def rounded(value):return [rounded(v) for v in value] if isinstance(value,list) else fp32(value)
    return {name:rounded(value) for name,value in payload['weights'].items()}


def scalar_forward(weights,features):
    """Separate loop implementation, including signed player-0 output Jacobians."""
    x=list(features['x']);d=features['d'];r=features['r']
    a1=[sum(w*v for w,v in zip(row,x))+b for row,b in zip(weights['layers.0.weight'],weights['layers.0.bias'])]
    h1=[max(0.0,v) for v in a1]
    a2=[sum(w*v for w,v in zip(row,h1))+b for row,b in zip(weights['layers.2.weight'],weights['layers.2.bias'])]
    h2=[max(0.0,v) for v in a2]
    u=sum(w*v for w,v in zip(weights['layers.4.weight'][0],h2))+weights['layers.4.bias'][0]
    t=math.tanh(u);du=r*(1-t*t)
    hidden=[du*v for v in weights['layers.4.weight'][0]]
    ga2=[g if a>0 else 0.0 for g,a in zip(hidden,a2)]
    gh1=[sum(ga2[i]*weights['layers.2.weight'][i][j] for i in range(64)) for j in range(64)]
    ga1=[g if a>0 else 0.0 for g,a in zip(gh1,a1)]
    gx=[sum(ga1[i]*weights['layers.0.weight'][i][j] for i in range(64)) for j in range(127)]
    norm=lambda values:math.sqrt(sum(v*v for v in values))
    logits=[sum(w*v for w,v in zip(row,h2))+b for row,b in zip(weights['policy.weight'],weights['policy.bias'])]
    head=[du*v for v in h2]+[du]
    return dict(a1=a1,h1=h1,a2=a2,h=h2,u=u,tanh=t,prediction_player0=d+r*t,policy_logits=logits,
        derivative_u_player0=du,derivative_head_player0=head,derivative_hidden_player0=hidden,derivative_input_player0=gx,
        gradient_norms_player0=dict(head=norm(head),input=norm(gx),hidden=norm(hidden),
            layer2_weight=norm(ga2)*norm(h1),layer2_bias=norm(ga2),layer1_weight=norm(ga1)*norm(x),layer1_bias=norm(ga1)),
        model_saturated=abs(t)>=.95,relu_kinks=dict(layer1=sum(v==0 for v in a1),layer2=sum(v==0 for v in a2)))


def target_facts(features,target):
    d,r=features['d'],features['r'];assert r!=0
    z=(target-d)/r;inside=-1<z<1
    return dict(z=z,physically_bounded=abs(z)<=1,finite_head_representable=inside,
        saturation_endpoint=abs(z)==1,target_near_boundary=abs(z)>=.95,
        preactivation_target=math.atanh(z) if inside else None)


def independent_projection(hidden,targets,rcond=1e-10):
    """Sample-space projection only; never form or export fitted coefficients."""
    import numpy as np
    A=np.column_stack((np.asarray(hidden,dtype=np.float64),np.ones(len(hidden))))
    t=np.asarray(targets,dtype=np.float64)
    U,s,_=np.linalg.svd(A,full_matrices=False)
    rank=int(np.count_nonzero(s>(s[0]*rcond))) if len(s) and s[0]>0 else 0
    basis=U[:,:rank];projection=basis@(basis.T@t);residual=t-projection
    assert np.allclose(basis.T@residual,0,rtol=0,atol=1e-9)
    assert math.isclose(float(t@t),float(projection@projection+residual@residual),rel_tol=1e-10,abs_tol=1e-9)
    return dict(rows=len(hidden),columns=65,rank=rank,rcond=rcond,singular_values=s.tolist(),
        residual=residual.tolist(),residual_l2=float(np.linalg.norm(residual)),
        residual_rmse=float(np.linalg.norm(residual)/math.sqrt(len(t))),
        projection_is_preactivation_only=True,coefficients_exported=False)


def metadata_projection(journal,table,fields):
    """SQL extracts declared metadata; no whole supervision-body reads."""
    path=ROOT/'data'/journal/'work.sqlite3'
    with closing(qread(path)) as db:
        query='SELECT '+','.join("json_extract(body,'$."+f+"')" for f in fields)+' FROM '+table
        return [{k:json.loads(v) if k=='state' else v for k,v in zip(fields,values)} for values in db.execute(query)]


def independent_metadata_graph():
    """BFS union of all saved origins, old closure, trajectories and reservations."""
    graph=defaultdict(set);reasons=defaultdict(set)
    def edge(a,b):graph[a].add(b);graph[b].add(a)
    def add(row,reason=None):
        f=('family',row['family']);edge(f,('quotient',quotient(row['state'])))
        if reason:reasons[f].add(reason)
    old=read(ROOT/'build/stratum-inventory.json');mask=read(ROOT/'build/allocation-weighting-inventory.json')
    protected_q=set(mask['protected_quotients'])|set(mask['prior_H12_H14_exposed_quotients'])
    protected_f=set(mask['protected_families'])|set(mask['prior_H12_H14_exposed_families'])
    for filename in ('ordinary-value-registry.json','selfplay-training-pilot-registry.json','ordinary-selfplay-registry.json'):
        ledger=read(ROOT/'build'/filename)
        protected_q.update(ledger['true_evaluation_quotients'] if 'true_evaluation_quotients' in ledger else ledger['quotients'])
        protected_f.update(ledger['true_evaluation_families'] if 'true_evaluation_families' in ledger else ledger['families'])
    for component in old['components'].values():
        anchor=('family',component['families'][0])
        for fid in component['families']:edge(anchor,('family',fid))
        for q in component['quotients']:edge(anchor,('quotient',q))
        if component['blocked_reasons']:reasons[anchor].add('ancestral_blocked_component')
    for filename in ('allocation-weighting-reservations.json','stratum-reservations.json','qstability-reservations.json'):
        reservation=read(ROOT/'build'/filename)
        for role in reservation.get('roles',{'all':reservation}).values():
            protected_q.update(role['eligible_quotients']);protected_f.update(role['origin_families'])
    original=metadata_projection('deeper-9000-v2','roots',['state_key','state','family','stratum','legal_mask','split'])
    for row in original:add(row,'original_development' if row['split']!='train' else None)
    with closing(qread(ROOT/'data/ordinary-selfplay-pilot-v1/work.sqlite3')) as db:
        for fid,cohort,text in db.execute("SELECT json_extract(body,'$.family'),json_extract(body,'$.cohort'),json_extract(body,'$.candidates') FROM families"):
            for candidate in json.loads(text):
                if candidate['stratum'] in STRATA[:2] and candidate['legal_mask'].bit_count()>=2:
                    add(dict(candidate,family=fid),'ordinary_validation_family' if cohort!='training_candidate' else None)
    sp={}
    for journal in ('selfplay-training-pilot-v1','ordinary-selfplay-pilot-v1'):
        sp[journal]=metadata_projection(journal,'sp_rows',['id','state_key','state','family','stratum','legal_mask','choice'])
        for row in sp[journal]:add(row)
    for q in protected_q:reasons[('quotient',q)].add('protected_or_reserved_quotient')
    for fid in protected_f:reasons[('family',fid)].add('protected_or_reserved_family')
    found={};seen=set();components=[]
    for first in graph:
        if first in seen:continue
        todo=[first];seen.add(first);nodes=set()
        while todo:
            node=todo.pop();nodes.add(node)
            for other in graph[node]:
                if other not in seen:seen.add(other);todo.append(other)
        blocked=sorted(set().union(*(reasons[node] for node in nodes)))
        body=dict(families=sorted(v for kind,v in nodes if kind=='family'),quotients=sorted(v for kind,v in nodes if kind=='quotient'),blocked_reasons=blocked)
        components.append(body)
        for node in nodes:found[node]=body
    return dict(components=components,by_node=found,original=original,selfplay=sp,protected_q=protected_q,protected_f=protected_f)


def cached_metadata():
    result=defaultdict(list)
    for filename in ('build/ordinary-value-rows.json','build/value-pilot-rows.json'):
        text=(ROOT/filename).read_text(encoding='utf-8')
        with closing(sqlite3.connect(':memory:')) as db:
            groups=[r[0] for r in db.execute("SELECT key FROM json_each(?,'$.groups')",(text,))]
            for group in groups:
                query="SELECT key,json_extract(value,'$.state_key'),json_array_length(value,'$.cached_features.x') FROM json_each(?,'$.groups."+group+"')"
                for index,key,length in db.execute(query,(text,)):
                    if length==127:result[key].append(dict(path=filename,group=group,index=index))
    return result


def requested_hashes(journal,table):
    with closing(qread(ROOT/'data'/journal/'work.sqlite3')) as db:
        names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return dict(db.execute('SELECT state,hash FROM '+table+' ORDER BY state')) if table in names else {}


def independent_pools(graph):
    cache=cached_metadata()
    def clean(row):return not graph['by_node'][('family',row['family'])]['blocked_reasons']
    rehearsal={key for key,refs in cache.items() if any(r['path']=='build/ordinary-value-rows.json' and r['group']=='rehearsal' for r in refs)}
    primary=[r for r in graph['original'] if r['state_key'] in rehearsal and clean(r)]
    assert all(r['split']=='train' for r in primary)
    pairs=[]
    for journal in ('ordinary-data-pilot-v1','natural-study-v1','midgame-label-pilot-v1'):
        h8=requested_hashes(journal,'h8_labels') or requested_hashes(journal,'labels')
        h10=requested_hashes(journal,'h10_labels')
        for row in metadata_projection(journal,'roots',['state_key','state','family','stratum','legal_mask','cohort']):
            if row['cohort']=='training_candidate' and clean(row) and row['state_key'] in h8 and row['state_key'] in h10:
                pairs.append(dict(row,source_journal=journal))
    mc=[r for r in graph['selfplay']['ordinary-selfplay-pilot-v1'] if r['choice']==0 and clean(r)]
    return dict(primary=primary,paired=pairs,MC=mc,cache=cache)


def independent_selection(pool,components,salt):
    component_by_family={f:cid for cid,c in components.items() for f in c['families']}
    slots=[(cell,index) for cell in CELLS for index in range(32)]
    ordered={cell:sorted((r for r in pool if r['stratum']+'/actor'+str(r['state']['controller'])==cell and r['legal_mask'].bit_count()>=2),
        key=lambda r:(digest(salt+r['state_key']),r['state_key'])) for cell in CELLS}
    assigned={};chosen={}
    def augment(slot,seen):
        for row in ordered[slot[0]]:
            cid=component_by_family[row['family']]
            if cid in seen:continue
            seen.add(cid)
            if cid not in assigned or augment(assigned[cid],seen):assigned[cid]=slot;chosen[slot]=row;return True
        return False
    assert all(augment(slot,set()) for slot in slots),'Independent component matching shortage'
    return sorted((r['state_key'] for r in chosen.values()))


def near(left,right,atol=1e-10,rtol=1e-12):
    if isinstance(left,list):
        assert isinstance(right,(list,tuple)) and len(left)==len(right)
        for a,b in zip(left,right):near(a,b,atol,rtol)
    else:assert math.isclose(float(left),float(right),abs_tol=atol,rel_tol=rtol),(left,right)


def exact_fraction(body,value):
    value=F(value);assert body==dict(numerator=value.numerator,denominator=value.denominator,value=float(value))
    assert type(body['numerator']) is int and type(body['denominator']) is int


def selected_source(ref,key_column='state'):
    with closing(qread(ROOT/'data'/ref['journal']/'work.sqlite3')) as db:
        text,checksum=db.execute('SELECT body,hash FROM '+ref['table']+' WHERE '+key_column+'=?',(ref['key'],)).fetchone()
    item=json.loads(text);assert digest(item)==checksum==ref['sha256'];return item


def cached_features_projection(ref):
    text=(ROOT/ref['path']).read_text(encoding='utf-8');base='$.groups.'+ref['group']+'['+str(ref['index'])+']'
    with closing(sqlite3.connect(':memory:')) as db:
        features,p,key,family=db.execute('SELECT json_extract(?,?),json_extract(?,?),json_extract(?,?),json_extract(?,?)',
            (text,base+'.cached_features',text,base+'.state',text,base+'.state_key',text,base+'.family')).fetchone()
    value=json.loads(features)
    return {k:value[k] for k in ('x','d','r')},json.loads(p),key,family


def record_masses(records,mode):
    from fractions import Fraction as F
    roots=defaultdict(list);families=defaultdict(set)
    for record in records:roots[record['state_key']].append(record['id']);families[record['family']].add(record['state_key'])
    return {r['id']:F(1,len(roots)*len(roots[r['state_key']])) if mode=='raw_root_balanced'
        else F(1,len(families)*len(families[r['family']])*len(roots[r['state_key']])) for r in records}


def weighted_projection(hidden,targets,weights):
    """Independent weighted residual, original endpoint mass, no coefficients."""
    from fractions import Fraction as F
    import numpy as np
    eligible=[i for i,z in enumerate(targets) if abs(z)<1]
    mass=sum((weights[i] for i in eligible),F())
    if not eligible:return dict(available=False,interior_mass=mass,rank=None,residual_rmse=None,singular_values=[])
    p=np.asarray([float(weights[i]/mass) for i in eligible]);A=np.asarray([hidden[i]+[1.0] for i in eligible]);t=np.asarray([math.atanh(float(targets[i])) for i in eligible])
    A=A*np.sqrt(p)[:,None];t=t*np.sqrt(p);U,s,_=np.linalg.svd(A,full_matrices=False)
    rank=int(sum(v>s[0]*1e-10 for v in s));basis=U[:,:rank];residual=t-basis@(basis.T@t)
    assert np.allclose(basis.T@residual,0,rtol=0,atol=1e-9)
    return dict(available=True,interior_mass=mass,rank=rank,residual_rmse=float(np.linalg.norm(residual)),singular_values=s.tolist(),
        target_rms=float(np.linalg.norm(t)),rows=len(eligible))


def independent_geometry(vectors,weights):
    import numpy as np
    x=np.asarray(vectors,dtype=np.float64);p=np.asarray([float(v) for v in weights]);mean=np.sum(x*p[:,None],axis=0);center=x-mean
    variance=np.sum(center*center*p[:,None],axis=0);s=np.linalg.svd(center*np.sqrt(p)[:,None],compute_uv=False)
    rank=int(sum(v>s[0]*1e-10 for v in s)) if len(s) else 0
    return dict(mean=mean.tolist(),variance=variance.tolist(),singular_values=s.tolist(),rank=rank)


def check_gradient_cancellation(actual,gradients,weights):
    mean=[sum(float(p)*g[j] for p,g in zip(weights,gradients)) for j in range(65)]
    norms=[math.sqrt(sum(v*v for v in g)) for g in gradients]
    numerator=math.sqrt(sum(v*v for v in mean));denominator=sum(float(p)*v for p,v in zip(weights,norms))
    near(actual['weighted_mean_gradient_norm'],numerator);near(actual['weighted_mean_individual_gradient_norm'],denominator)
    assert actual['zero_denominator']==(denominator==0)
    if denominator:near(actual['cancellation_ratio'],numerator/denominator)
    else:assert actual['cancellation_ratio'] is None


def check_risk(actual,records,observations,weights,name):
    fits=[observations[r['unit_id']]['targets'][r['id']]['models'][name] for r in records]
    heads=[observations[r['unit_id']]['models'][name] for r in records]
    average=lambda values:sum(float(p)*float(v) for p,v in zip(weights,values))
    error=[f['error_player0'] for f in fits];normalized=[f['normalized_residual_error'] for f in fits]
    expected=dict(mae_player0=average([abs(v) for v in error]),rmse_player0=math.sqrt(average([v*v for v in error])),bias_player0=average(error),
        normalized_mae=average([abs(v) for v in normalized]),normalized_rmse=math.sqrt(average([v*v for v in normalized])),
        residual_huber_loss=average([f['residual_huber_loss'] for f in fits]),model_saturated_mass=average([h['saturated'] for h in heads]),
        target_near_boundary_mass=average([f['target_near_boundary'] for f in fits]),seed_gain_signed_mean=average([h['seed_output_gain'] for h in heads]),
        seed_gain_absolute_mean=average([abs(h['seed_output_gain']) for h in heads]),normalized_gain_mean=average([h['normalized_output_gain'] for h in heads]))
    for key,value in expected.items():near(actual[key],value)
    for field in ('normalized_jacobian_norms','seed_jacobian_norms'):
        for key in heads[0][field]:near(actual[field[:-1]+'_means'][key],average([h[field][key] for h in heads]))
    check_gradient_cancellation(actual['head_gradient_cancellation'],[f['residual_huber_gradient_head'] for f in fits],weights)
    assert set(actual['by_family_head_gradient_cancellation'])=={r['family'] for r in records}
    for family,value in actual['by_family_head_gradient_cancellation'].items():
        indices=[i for i,r in enumerate(records) if r['family']==family];mass=sum((weights[i] for i in indices),F())
        assert value['original_mass']['numerator']*mass.denominator==mass.numerator*value['original_mass']['denominator']
        check_gradient_cancellation(value['gradients'],[fits[i]['residual_huber_gradient_head'] for i in indices],[weights[i]/mass for i in indices])


class NeurodiagIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from congklak import neurodiag as producer,neurodiag_data as data,neurodiag_metrics as metrics
        cls.p,cls.d,cls.m=producer,data,metrics
        cls.base=WORK/('retained-'+uuid.uuid4().hex);cls.base.mkdir(parents=True)

    def test_01_training_roles_all_origins_protection_closure_and_metadata_selection(self):
        baseline=read(BASELINE);self.assertEqual(baseline['inputs_count'],5377);self.assertIs(baseline['passed'],True);current(baseline['inputs'])
        self.assertEqual(len(baseline['inputs']),5377)
        # The SQL guard records unexpanded SQL only: bound JSON parameters may
        # contain saved supervision and must never be printed in diagnostic logs.
        original_connect=sqlite3.connect;queries=[]
        class MetadataOnly(sqlite3.Connection):
            def __init__(db,*args,**kwargs):
                super().__init__(*args,**kwargs)
                def authorize(action,table,column,database,origin):
                    if action!=sqlite3.SQLITE_READ or column!='body':return sqlite3.SQLITE_OK
                    table=table or ''
                    return sqlite3.SQLITE_DENY if table=='labels' or any(v in table for v in ('h8_','h10_','h12_','h14_')) else sqlite3.SQLITE_OK
                db.set_authorizer(authorize)
            def execute(db,query,parameters=(),/):
                lowered=query.lower();queries.append(query)
                assert not re.search(r'\bselect\s+(?:\*|body\b)',lowered),query
                assert not any(v in lowered for v in ('$.label','$.terminal_return','$.target_player0')),query
                return super().execute(query,parameters)
        def connect(*args,**kwargs):kwargs['factory']=MetadataOnly;return original_connect(*args,**kwargs)
        with mock.patch.object(sqlite3,'connect',side_effect=connect):actual=self.d.metadata_inventory()
        graph=independent_metadata_graph();pool=independent_pools(graph)
        self.assertEqual(set(actual['protected_quotients']),graph['protected_q']);self.assertEqual(set(actual['protected_families']),graph['protected_f'])
        by_f={f:c for c in actual['components'].values() for f in c['families']}
        for component in graph['components']:
            expected=by_f[component['families'][0]]
            self.assertEqual(expected['families'],component['families']);self.assertEqual(expected['quotients'],component['quotients'])
            self.assertEqual(bool(expected['blocked_reasons']),bool(component['blocked_reasons']))
        for owner,key in (('primary_pool','primary'),('paired_pool','paired')):
            self.assertEqual({r['state_key'] for r in actual[owner]},{r['state_key'] for r in pool[key]})
        self.assertEqual({r['id'] for r in actual['MC_pool']},{r['id'] for r in pool['MC']})
        panel=read(self.d.PANEL)
        self.assertEqual([r['state_key'] for r in panel['primary']],independent_selection(pool['primary'],actual['components'],self.d.SALT))
        self.assertEqual(len(panel['primary']),256);self.assertEqual(len(panel['paired']),28);self.assertEqual(len(panel['MC']),48)
        self.assertEqual(Counter(r['cell'] for r in panel['primary']),Counter({c:32 for c in CELLS}))
        self.assertEqual(len({r['component'] for r in panel['primary']}),256);self.assertEqual(len({r['family'] for r in panel['primary']}),256)
        self.assertEqual(panel['source_records'],332);self.assertEqual(panel['target_records'],360);self.assertLessEqual(panel['unique_rawstates'],308)
        self.assertIs(panel['training_source_diagnostic'],True);self.assertIs(panel['new_assessment_or_holdout_claim'],False);self.assertIs(panel['no_new_optimizer_reservations'],True)
        quality=read(ROOT/'build/quality-data.json');ancestral=read(ROOT/'build/sampling-review.json')
        quality_keys=set(quality['train_state_keys']);observed=ancestral['observed_accepted_run']
        self.assertEqual(len(quality_keys),7669);self.assertEqual(observed['unique_roots_drawn'],7669);self.assertEqual(observed['total_draws'],384000)
        self.assertTrue({r['state_key'] for r in panel['primary']}<=quality_keys)
        # The accepted late-K5 guard is a store-offset/quotient alias. It must
        # poison its whole origin component even though its current draw was0.
        held=next(r for r in graph['original'] if r['state_key']=='000000000000002d010000020100013001')
        self.assertIn(quotient(held['state']),graph['protected_q']);self.assertTrue(graph['by_node'][('family',held['family'])]['blocked_reasons'])
        self.assertNotIn(held['state_key'],{r['state_key'] for r in actual['primary_pool']})
        exposure=read(ROOT/'build/neurodiag-prefreeze/schema_probe_exposure.json')
        self.assertEqual(exposure['target_containing_training_rows_parsed'],3);self.assertIs(exposure['zero_prefreeze_body_access_claim'],False)
        self.assertIs(exposure['protected_or_sealed_target_bodies_read'],False)
        declared=[]
        for item in exposure['rows']:
            with closing(qread(ROOT/item['journal'])) as db:
                checksum=db.execute('SELECT hash FROM '+item['table']+' WHERE '+item['key_column']+'=?',(item['key'],)).fetchone()[0]
            self.assertEqual(checksum,item['body_sha256'])
            comp=graph['by_node'].get(('family',item['family']))
            declared.append(dict(journal=item['journal'],table=item['table'],key=item['key'],historical_training_source=True,
                currently_excluded_component=bool(comp and comp['blocked_reasons'])))
        EVIDENCE['metadata_roles_and_closure']=dict(prior_pins=5377,component_count=len(graph['components']),primary_pool=len(pool['primary']),
            paired_pool=len(pool['paired']),MC_records=len(pool['MC']),primary_selected=256,source_records=332,target_records=360,
            complete_component_memberships_checked=True,late_K5_alias_excluded=True,metadata_sql_statements=len(queries),
            protected_target_body_access_denied=True,prefreeze_declared_training_body_reads=declared,
            training_diagnostic_not_holdout=True,new_optimizer_reservations=False,
            primary_roots_with_positive_ancestral_draw_exposure=256,exact_ancestral_counts_reconstructed=False,
            nonzero_gradient_or_causal_effect_proven=False)

    def test_02_frozen_forward_derivatives_and_projection_contract(self):
        m=self.m
        weights={
            'layers.0.weight':[[fp32(((i*17+j*7)%23+1)/1000) for j in range(127)] for i in range(64)],
            'layers.0.bias':[fp32(.03+i/1000) for i in range(64)],
            'layers.2.weight':[[fp32(((i*11+j*3)%19+1)/1000) for j in range(64)] for i in range(64)],
            'layers.2.bias':[fp32(.02+i/2000) for i in range(64)],
            'layers.4.weight':[[fp32((1 if i%2 else -1)/1000) for i in range(64)]],
            'layers.4.bias':[fp32(.07)],
            'policy.weight':[[fp32((i-j)/1000) for j in range(64)] for i in range(7)],
            'policy.bias':[fp32(i/100) for i in range(7)]}
        features=dict(x=[(j%13+1)/1000 for j in range(127)],d=7,r=-36)
        expected=scalar_forward(weights,features);trunk=m.forward_arrays(weights,features);head=m.head_diagnostics(weights,features,trunk)
        near(head['value_player0'],expected['prediction_player0']);near(trunk['logits'].tolist(),expected['policy_logits'])
        near(head['seed_output_gain'],expected['derivative_u_player0']);self.assertLess(head['seed_output_gain'],0)
        near(head['input_jacobian_player0'],expected['derivative_input_player0']);near(head['head_jacobian_player0'],expected['derivative_head_player0'])
        norm_map={'head':'head','input':'input','hidden':'hidden','layer2_weight':'layer2_weight','layer2_bias':'layer2_bias','layer1_weight':'layer0_weight','layer1_bias':'layer0_bias'}
        for own,other in norm_map.items():near(expected['gradient_norms_player0'][own],head['seed_jacobian_norms'][other])
        self.assertTrue(all(abs(v)>1e-8 for v in expected['a1']+expected['a2']))
        epsilon=1e-6
        def value(w,f):return scalar_forward(w,f)['prediction_player0']
        for j in (0,31,126):
            plus=copy.deepcopy(features);minus=copy.deepcopy(features);plus['x'][j]+=epsilon;minus['x'][j]-=epsilon
            near((value(weights,plus)-value(weights,minus))/(2*epsilon),expected['derivative_input_player0'][j],1e-6,1e-4)
        for j in (0,31,63,64):
            plus=copy.deepcopy(weights);minus=copy.deepcopy(weights)
            if j==64:plus['layers.4.bias'][0]+=epsilon;minus['layers.4.bias'][0]-=epsilon
            else:plus['layers.4.weight'][0][j]+=epsilon;minus['layers.4.weight'][0][j]-=epsilon
            near((value(plus,features)-value(minus,features))/(2*epsilon),expected['derivative_head_player0'][j],1e-6,1e-4)
        for target in (-29,7,43):
            fit=m.target_diagnostics(target,features,head);facts=target_facts(features,target)
            self.assertEqual(fit['target_status'],'strict_interior' if facts['finite_head_representable'] else 'endpoint')
            near(fit['target_normalized']['value'],facts['z']);self.assertEqual(fit['target_near_boundary'],facts['target_near_boundary'])
            if facts['preactivation_target'] is not None:near(fit['target_logit'],facts['preactivation_target'])
            else:self.assertIsNone(fit['target_logit'])
            error=expected['tanh']-facts['z'];psi=max(-.1,min(.1,error))
            near(fit['residual_huber_gradient_pre_activation'],psi*(1-expected['tanh']**2))
        with self.assertRaises(ValueError):m.target_diagnostics(44,features,head)
        terminal=m.scalar_forward(weights,dict(features,r=0));self.assertEqual(terminal['value_player0'],7);self.assertEqual(terminal['policy_logits'],[0.0]*7)
        hidden=[[i/96,(i%7)/7]+[0.0]*62 for i in range(96)];targets=[math.sin(i/3) for i in range(96)]
        projection=independent_projection(hidden,targets);self.assertEqual(projection['rank'],3);self.assertGreater(projection['residual_l2'],0)
        EVIDENCE['prospective_numerical_contract']=dict(signed_actor1_gain=True,independent_scalar_loops=True,
            synthetic_finite_differences_away_from_kinks=True,unchanged_real_model_files=True,
            projection_rank=projection['rank'],projection_residual_not_seed_error_bound=True,
            target_endpoints_not_clipped=True,no_coefficients_or_optimizer=True)

    def worker(self,spec,directory):
        directory.mkdir();spec=dict(spec,output=str(directory/'proof.json'))
        save(directory/'spec.json',spec)
        run_child([sys.executable,str(ROOT/'tools/neurodiag-audit.py'),str(directory/'spec.json')],directory/'worker.log',timeout=900)
        proof=read(directory/'proof.json');self.assertIs(proof['passed'],True);return proof

    def fixture_spec(self,path):
        return dict(mode='fixture_cli',clis=[dict(path='tools/diagnose-neural.py',argv=['collect','--fixture','--run',str(path)])],
            input_names=['tests/neurodiag_independent_audit.py','tools/neurodiag-audit.py',rel(FREEZE)])

    def test_03_live_batch_commit_recovery_and_completed_certification_gate(self):
        p=self.p;fresh=self.base/'fresh'
        fresh_proof=self.worker(self.fixture_spec(fresh),self.base/'fresh-worker')
        self.assertEqual([len(v) for v in fresh_proof['observation_call_unit_ids']],[16,16,16])
        expected=p.raw_complete(fresh)
        self.assertEqual(expected['manifest']['status'],'complete_uncertified');self.assertEqual(len(expected['batch_rows']),3)
        self.assertEqual(len(expected['observations']),48)
        tests=[]
        for event,prefix in (('neurodiag_precommit',1),('neurodiag_committed',2)):
            path=self.base/(event+'-run');probe=self.base/(event+'-probe');spec=self.fixture_spec(path)
            specpath=self.base/(event+'-kill-spec.json');save(specpath,dict(spec,output=str(self.base/(event+'-unexpected-proof.json'))))
            marker=kill([sys.executable,str(ROOT/'tools/neurodiag-audit.py'),str(specpath)],probe,event,prefix,
                dict(NEURODIAG_BLOCK_EVENT=event,NEURODIAG_BLOCK_STEP=str(prefix),NEURODIAG_MARKER=str(probe/'declared-marker.json')))
            self.assertEqual(marker['completed'],prefix)
            db=path/'work.sqlite3';before={table:physical_rows(db,table,key) for table,key in (('meta','key'),('batches','idx'),('observations','unit'))}
            self.assertEqual([r[0] for r in before['batches']],list(range(prefix)));self.assertEqual(len(before['observations']),prefix*16)
            self.assertEqual([r[0] for r in before['meta']],['identity'])
            backup(db,probe/'validated-prefix.sqlite3')
            resume_proof=self.worker(self.fixture_spec(path),self.base/(event+'-resume-worker'))
            self.assertEqual([len(v) for v in resume_proof['observation_call_unit_ids']],[16]*(3-prefix))
            repeated=[u for call in resume_proof['observation_call_unit_ids'] for u in call]
            self.assertEqual(repeated,expected['identity']['settings']['unit_ids'][16*prefix:])
            self.assertFalse(set(repeated)&{r[0] for r in before['observations']})
            actual=p.raw_complete(path)
            self.assertEqual(actual,expected)
            for table,key in (('meta','key'),('batches','idx'),('observations','unit')):
                after={r[0]:r for r in physical_rows(db,table,key)}
                self.assertTrue(all(after[r[0]]==r for r in before[table]))
            tests.append(dict(event=event,actual_process_kill=True,committed_prefix=prefix,committed_units=prefix*16,
                marker=rel(probe/'marker.json'),probe=rel(probe),run=rel(path),
                committed_prefix_bytes_preserved=True,fresh_semantics_equal=True,entire_cache_semantics_preserved=True))
        # Ordinary completed admission must not silently treat a fixture/raw
        # cache as certified numerical evidence.
        forbidden=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected model work on completed refusal'))
        with mock.patch.object(self.m,'observe',forbidden),mock.patch.object(self.m,'summarize_cached',forbidden):
            with self.assertRaises(ValueError):p.run(fresh,True)
        self.assertFalse(p.WITNESS.exists());self.assertFalse(p.PRODUCTION_AUDIT.exists())
        EVIDENCE['live_batch_recovery']=dict(tests=tests,accepted_actual_process_kills=2,
            fixture_provenance='First48 frozen eligible training-panel units;3 batches16; explicitly preproduction engineering forwards.',
            no_zero_preforward_claim=True,unfinished_batches_only_recomputed=True,uncertified_completed_admission_refused=True)

    def test_04_actual_panel_cache_scalar_forward_and_targets(self):
        p,d=self.p,self.d;cache=p.raw_complete();pack=d.load_inputs();observations=cache['observations']
        self.assertEqual(len(observations),308);self.assertEqual(len(cache['batch_rows']),20)
        self.assertEqual(set(observations),set(pack['units']));self.assertEqual(cache['manifest']['target_records'],360)
        weights={name:frozen_weights(path) for name,path in (('control',p.common.CONTROL),('candidate',p.common.CANDIDATE))}
        frozen=('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias','policy.weight','policy.bias')
        self.assertTrue(all(weights['control'][k]==weights['candidate'][k] for k in frozen))
        metadata={digest(row):row for cohort in ('primary','paired','MC') for row in read(d.PANEL)[cohort]}
        literal_cache={};games_checked={};target_counts=Counter();max_error=0.;target_rows=0
        type(self).scalar_observations={}
        for uid,unit in sorted(pack['units'].items()):
            item=observations[uid];features=unit['features'];self.assertEqual(item['source_unit'],unit)
            active=sum(map(sum,unit['state']['pits']));self.assertEqual(features['r'],active*(1 if unit['state']['controller']==0 else -1))
            self.assertEqual(features['d'],unit['state']['stores'][0]-unit['state']['stores'][1]);self.assertGreater(active,4)
            expected={name:scalar_forward(w,features) for name,w in weights.items()}
            type(self).scalar_observations[uid]=expected
            self.assertEqual(expected['control']['policy_logits'],expected['candidate']['policy_logits'])
            near(item['shared']['policy_logits'],expected['control']['policy_logits'])
            near(item['shared']['hidden'],expected['control']['h']);near(item['shared']['layer0_hidden'],expected['control']['h1'])
            near(item['shared']['layer0_pre_activation'],expected['control']['a1']);near(item['shared']['layer2_pre_activation'],expected['control']['a2'])
            for layer,values in (('layer0',expected['control']['a1']),('layer2',expected['control']['a2'])):
                self.assertEqual(item['shared'][layer+'_relu_active'],[v>0 for v in values])
                self.assertEqual(item['shared'][layer+'_kink'],[abs(v)<=1e-8 for v in values])
            legal=[a for a,n in enumerate(unit['state']['pits'][unit['state']['controller']]) if n]
            logits=expected['control']['policy_logits'];maximum=max(logits[a] for a in legal);exps={a:math.exp(logits[a]-maximum) for a in legal};denom=sum(exps.values())
            near(item['shared']['policy_probabilities'],[exps.get(a,0)/denom for a in range(7)]);self.assertEqual(item['shared']['legal_actions'],legal)
            for name,want in expected.items():
                head=item['models'][name];near(head['value_player0'],want['prediction_player0']);near(head['preactivation'],want['u']);near(head['tanh'],want['tanh'])
                max_error=max(max_error,abs(head['value_player0']-want['prediction_player0']))
                residual=features['r']*want['tanh'];quantized=math.floor(residual*256+.5) if residual>=0 else math.ceil(residual*256-.5)
                self.assertEqual(head['value_units'],features['d']*256+quantized)
            for record in unit['records']:
                meta=metadata[record['panel_row_sha256']];self.assertEqual(record['historical_role'],meta['historical_role']);self.assertEqual(meta['state'],unit['state'])
                self.assertEqual(record['historical_role'],{'base_H8':'train','paired_H8_H10':'training_candidate','ordinary_MC':'ordinary_selfplay_training_candidate'}[record['cohort']])
                target_rows+=1;target_counts[(record['cohort'],record['target_type'])]+=1
                ref=record['target_source']['reference']
                if record['cohort']=='ordinary_MC':
                    source=selected_source(ref,'id');self.assertEqual(source['features'],features)
                    self.assertEqual(source['state'],unit['state']);self.assertEqual(source['choice'],0)
                    gameid=source['gameid']
                    if gameid not in games_checked:
                        game=selected_source(dict(journal=ref['journal'],table='sp_games',key=gameid,sha256=source['terminal_game_sha256']),'id')
                        with closing(qread(ROOT/'data'/ref['journal']/'work.sqlite3')) as db:
                            decisions=db.execute('SELECT choice,body,hash FROM sp_decisions WHERE gameid=? ORDER BY choice',(gameid,)).fetchall()
                        position=game['start'];self.assertEqual(len(decisions),game['choices'])
                        for index,text,checksum in decisions:
                            decision=json.loads(text);self.assertEqual(digest(decision),checksum);self.assertEqual(decision['choice'],index);self.assertEqual(position,decision['state'])
                            transition=literal.apply(pos(position),decision['action']);position=state(transition.position);self.assertEqual(position,decision['after'])
                        self.assertEqual(position,game['position']);self.assertFalse(sum(map(sum,position['pits'])));self.assertEqual(game['status'],'terminal')
                        self.assertEqual(game['terminal_margin'],position['stores'][0]-position['stores'][1]);games_checked[gameid]=game
                    self.assertEqual(record['target_player0'],games_checked[gameid]['terminal_margin'])
                else:
                    item_source=selected_source(ref);label=item_source['label'];self.assertEqual(label['choice_depth'],ref['depth'])
                    q={r['pit']:r['value_player0'] for r in label['action_scores']};self.assertEqual(set(q),set(legal));self.assertEqual(len(q),len(label['action_scores']))
                    self.assertTrue(all(type(v) is int for v in q.values()));best=(max if meta['actor']==0 else min)(q.values())
                    self.assertEqual(label['best_moves'],[a for a in legal if q[a]==best]);self.assertEqual(label['target_player0'],best);self.assertEqual(record['target_player0'],best)
                    with closing(qread(ROOT/'data'/ref['journal']/'work.sqlite3')) as db:
                        source_root,nested,wrapper=db.execute("SELECT json_extract(body,'$.source_root_sha256'),json_extract(body,'$.root'),hash FROM roots WHERE state=?",(ref['key'],)).fetchone()
                    expected_root=source_root or (digest(json.loads(nested)) if nested is not None else wrapper)
                    self.assertEqual(item_source['root_sha256'],expected_root);self.assertEqual(record['target_source']['root_sha256'],expected_root)
                    if meta['feature_ref'] is not None:
                        original,position,key,family=cached_features_projection(meta['feature_ref'])
                        self.assertEqual((position,key,family),(meta['state'],meta['state_key'],meta['family']));self.assertEqual(original,features)
                    if record['cohort']=='paired_H8_H10':
                        key=unit['state_key']
                        if key not in literal_cache:literal_cache[key]=literal_features(unit['state'])
                        literal_value=literal_cache[key];self.assertEqual((literal_value['d'],literal_value['r']),(features['d'],features['r']));near(literal_value['x'],features['x'],1e-12,0)
                facts=target_facts(features,record['target_player0']);self.assertTrue(facts['physically_bounded'])
                self.assertEqual(set(item['targets']),{r['id'] for r in unit['records']})
                for name,want in expected.items():
                    fit=item['targets'][record['id']]['models'][name]
                    self.assertEqual(fit['target_status'],'strict_interior' if facts['finite_head_representable'] else 'endpoint')
                    near(fit['target_normalized']['value'],facts['z']);self.assertEqual(fit['target_near_boundary'],facts['target_near_boundary'])
                    if facts['preactivation_target'] is None:self.assertIsNone(fit['target_logit'])
                    else:near(fit['target_logit'],facts['preactivation_target'])
                    error=want['prediction_player0']-record['target_player0'];near(fit['error_player0'],error);near(fit['absolute_error_player0'],abs(error))
        self.assertEqual(target_rows,360);self.assertEqual(len(literal_cache),28);self.assertEqual(len(games_checked),48)
        self.assertEqual(sorted(target_counts.values()),[28,28,48,256])
        EVIDENCE['actual_forward_and_sources']=dict(passed=True,forward_units=308,shared_trunk_forwards=308,
            value_head_evaluations=616,policy_evaluations=308,scalar_audit_model_views=616,target_records=360,
            source_records=332,MC_terminal_games_replayed=48,MC_distinct_starts=24,paired_literal_vectors_compared=28,
            paired_cached_vectors=15,paired_missing_vectors_generated=13,maximum_scalar_value_error=max_error,
            policy_six_tensors_and_identical_subgraph_exact=True,target_types_separate=True,
            no_generalization_or_independent_sample_claim=True)

    def test_05_analytic_gradients_svd_summaries_and_observation_witness(self):
        self.assertIs(EVIDENCE['actual_forward_and_sources']['passed'],True)
        p,m=self.p,self.m;cache=p.raw_complete();obs=cache['observations'];summary=cache['summary']
        scalar=type(self).scalar_observations;self.assertEqual(set(scalar),set(obs))
        records=[]
        for uid,item in sorted(obs.items()):
            r=item['source_unit']['features']['r']
            for name,expected in scalar[uid].items():
                head=item['models'][name]
                near(head['seed_output_gain'],expected['derivative_u_player0']);near(head['normalized_output_gain'],1-expected['tanh']**2)
                near(head['input_jacobian_player0'],expected['derivative_input_player0']);near(head['head_jacobian_player0'],expected['derivative_head_player0'])
                near(head['input_jacobian_normalized'],[v/r for v in expected['derivative_input_player0']]);near(head['head_jacobian_normalized'],[v/r for v in expected['derivative_head_player0']])
                self.assertEqual(head['saturated'],expected['model_saturated'])
                for own,other in {'head':'head','input':'input','hidden':'hidden','layer2_weight':'layer2_weight','layer2_bias':'layer2_bias','layer1_weight':'layer0_weight','layer1_bias':'layer0_bias'}.items():
                    near(head['seed_jacobian_norms'][other],expected['gradient_norms_player0'][own]);near(head['normalized_jacobian_norms'][other],expected['gradient_norms_player0'][own]/abs(r))
            for rid,entry in sorted(item['targets'].items()):
                record=entry['record'];records.append(dict(record,unit_id=uid))
                z=F(record['target_player0']-item['source_unit']['features']['d'],r)
                for name,expected in scalar[uid].items():
                    fit=entry['models'][name];error=expected['tanh']-float(z);psi=max(-.1,min(.1,error));loss=.5*error*error if abs(error)<=.1 else .1*(abs(error)-.05)
                    exact_fraction(fit['target_normalized'],z);near(fit['normalized_residual_error'],error);near(fit['residual_huber_psi'],psi);near(fit['residual_huber_loss'],loss)
                    near(fit['residual_huber_gradient_pre_activation'],psi*(1-expected['tanh']**2));near(fit['residual_huber_gradient_head'],[psi*v/r for v in expected['derivative_head_player0']])
        records.sort(key=lambda r:r['id'])
        expected_groups={cohort+'/'+kind for cohort,kind in {(r['cohort'],r['target_type']) for r in records}}
        self.assertEqual(set(summary['groups']),expected_groups);self.assertEqual(len(expected_groups),4)
        ledger={}
        for group,actual in summary['groups'].items():
            selected=[r for r in records if r['cohort']+'/'+r['target_type']==group]
            self.assertEqual(actual['source_record_ids'],[r['id'] for r in selected]);self.assertEqual(actual['records'],len(selected))
            self.assertEqual(actual['distinct_raw_roots'],len({r['state_key'] for r in selected}));self.assertEqual(actual['families'],len({r['family'] for r in selected}))
            ledger[group]={}
            for mode,value in actual['weightings'].items():
                mapping=record_masses(selected,mode);weights=[mapping[r['id']] for r in selected];self.assertEqual(sum(weights,F()),1)
                concentration=value['concentration'];self.assertEqual([r['record_id'] for r in concentration['record_mass']],[r['id'] for r in selected])
                for entry,weight in zip(concentration['record_mass'],weights):exact_fraction(entry['probability'],weight)
                rootmass=defaultdict(F);familymass=defaultdict(F)
                for record,weight in zip(selected,weights):rootmass[record['state_key']]+=weight;familymass[record['family']]+=weight
                exact_fraction(concentration['record_effective_support'],1/sum((v*v for v in weights),F()))
                exact_fraction(concentration['raw_root_effective_support'],1/sum((v*v for v in rootmass.values()),F()))
                exact_fraction(concentration['family_effective_support'],1/sum((v*v for v in familymass.values()),F()))
                exact_fraction(concentration['maximum_record_probability'],max(weights));exact_fraction(concentration['maximum_root_probability'],max(rootmass.values()));exact_fraction(concentration['maximum_family_probability'],max(familymass.values()))
                for name in ('control','candidate'):check_risk(value['models'][name],selected,obs,weights,name)
                hidden=[obs[r['unit_id']]['shared']['hidden'] for r in selected]
                z=[F(r['target_player0']-obs[r['unit_id']]['source_unit']['features']['d'],obs[r['unit_id']]['source_unit']['features']['r']) for r in selected]
                projection=weighted_projection(hidden,z,weights);reported=value['head_logit_projection']
                self.assertEqual(reported['available'],projection['available']);self.assertEqual(reported['rank'],projection['rank'])
                exact_fraction(reported['original_selected_mass'],1);exact_fraction(reported['original_interior_mass'],projection['interior_mass']);exact_fraction(reported['original_endpoint_mass'],1-projection['interior_mass'])
                self.assertIs(reported['endpoints_clipped'],False);self.assertIs(reported['coefficients_constructed'],False)
                if projection['available']:
                    near(reported['weighted_logit_projection_residual_rmse'],projection['residual_rmse'],1e-9,1e-10)
                    near(reported['singular_values'],projection['singular_values'],1e-9,1e-10);near(reported['weighted_logit_target_rms'],projection['target_rms'])
                    self.assertEqual(reported['rows'],projection['rows']);self.assertEqual(reported['columns'],65)
                    self.assertEqual(reported['full_row_rank'],projection['rank']==projection['rows'])
                    self.assertEqual(reported['small_sample_interpolation_warning'],projection['rank']==projection['rows'] and projection['rows']<=65)
                else:self.assertIsNone(reported['weighted_logit_projection_residual_rmse'])
                for field,vectors in (('encoded_feature_geometry',[obs[r['unit_id']]['source_unit']['features']['x'] for r in selected]),('frozen_hidden_geometry',hidden)):
                    geometry=independent_geometry(vectors,weights);got=value[field]
                    near(got['weighted_mean'],geometry['mean'],1e-9,1e-10);near(got['weighted_variance'],geometry['variance'],1e-9,1e-10)
                    near(got['singular_values'],geometry['singular_values'],1e-9,1e-10);self.assertEqual(got['rank'],geometry['rank'])
                ledger[group][mode]=dict(records=len(selected),raw_roots=len(rootmass),families=len(familymass),
                    projection_available=projection['available'],strict_interior_mass=str(projection['interior_mass']),rank=projection['rank'],
                    weighted_logit_residual_rmse=projection['residual_rmse'],coefficients_exported=False)
            for axis,values in actual['subgroups'].items():
                self.assertEqual(set(values),{str(r[axis]) for r in selected})
                for key,value in values.items():
                    subset=[r for r in selected if str(r[axis])==key];mapping=record_masses(subset,'raw_root_balanced');weights=[mapping[r['id']] for r in subset]
                    self.assertEqual(value['records'],len(subset));self.assertEqual(value['raw_roots'],len({r['state_key'] for r in subset}))
                    for name in ('control','candidate'):check_risk(value['models'][name],subset,obs,weights,name)
        self.assertEqual(m.summarize_cached(cache['batches']),summary)
        self.assertEqual(len(summary['paired_horizon_targets']),28);self.assertEqual(len(summary['ordinary_MC_within_start']),24)
        self.assertTrue(all(v['records']==2 for v in summary['ordinary_MC_within_start']))
        self.assertEqual(summary['counts']['shared_trunk_forwards'],308);self.assertEqual(summary['counts']['model_view_evaluations'],616)
        ledgerpath=self.base/'independent-numeric-summary.json';save(ledgerpath,ledger)
        row_hashes={name:{str(key):checksum for key,_text,checksum in physical_rows(p.DEFAULT/'work.sqlite3',name,column)} for name,column in (('meta','key'),('batches','idx'),('observations','unit'))}
        self.assertEqual(row_hashes,cache['row_hashes'])
        inputs=source_inputs();inputs.update(pins([FREEZE,WORK/'preflight.json',p.DEFAULT/'work.sqlite3',p.DEFAULT/'manifest.json',p.DEFAULT/'summary.json',ledgerpath]))
        self.assertNotIn(rel(WITNESS),inputs);self.assertNotIn(rel(WORK/'production.json'),inputs)
        witness=dict(schema='neurodiag-independent-observation-witness-v1',passed=True,
            identity_sha256=digest(cache['identity']),cache_row_sha256=row_hashes,database_sha256=sha(p.DEFAULT/'work.sqlite3'),
            manifest_sha256=sha(p.DEFAULT/'manifest.json'),summary_sha256=sha(p.DEFAULT/'summary.json'),cached_summary_sha256=digest(summary),
            checks=dict(scalar_forward_conformance=True,analytic_derivatives=True,numeric_summary_and_SVD=True,
                eligible_target_and_feature_sources=True,models_and_policy_fixed=True),inputs=inputs)
        save(WITNESS,witness)
        EVIDENCE['independent_numerical_certification']=dict(passed=True,witness=dict(path=rel(WITNESS),sha256=sha(WITNESS)),
            retained_numeric_ledger=dict(path=rel(ledgerpath),sha256=sha(ledgerpath)),
            all_output_input_head_gradients_checked=True,all_target_Huber_gradients_checked=True,
            all_four_separate_groups_both_weightings_and_subgroups_checked=True,
            weighted_artanh_projection_and_feature_geometry_checked=True,no_model_coefficients_or_updates=True,
            no_future_group_hash_or_database_certification_mutation=True)

    def test_06_corruption_repair_and_no_forward_completed_idempotence(self):
        from congklak import neurodiag_v2 as completed
        p=self.p;self.assertIs(read(WORK/'production.json')['passed'],True);original=p.verify()
        self.assertIs(original['certified'],True)
        cache=p.raw_complete();uid=next(iter(cache['observations']));rejects=[]
        forbidden=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Completed reader acquired Work or fell back to legacy collection'))
        events=[]
        with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
            self.assertEqual(completed.run(on_event=events.append),cache['manifest'])
        self.assertEqual([v['event'] for v in events],['neurodiag_already_complete'])
        def cloned(name):
            path=self.base/name;shutil.copytree(p.DEFAULT,path,ignore=shutil.ignore_patterns('*-shm','*-wal'))
            backup(p.DEFAULT/'work.sqlite3',path/'work.sqlite3')
            return path
        for kind in ('observation','summary','complete','identity'):
            path=cloned('corrupt-'+kind)
            if kind=='observation':mutate_row(path/'work.sqlite3','observations',uid,'unit',lambda value:value['models']['control'].__setitem__('preactivation',value['models']['control']['preactivation']+.5))
            elif kind=='summary':mutate_row(path/'work.sqlite3','meta','summary','key',lambda value:value['counts'].__setitem__('distinct_units',value['counts']['distinct_units']+1))
            elif kind=='complete':mutate_row(path/'work.sqlite3','meta','complete','key',lambda value:value.__setitem__('status','certified'))
            else:mutate_row(path/'work.sqlite3','meta','identity','key',lambda value:value['settings'].__setitem__('fixture',True))
            with mock.patch.object(p,'DEFAULT',path):
                with self.assertRaises(ValueError):p.raw_complete(path)
                with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
                    with self.assertRaises(ValueError):completed.run(path)
            rejects.append(kind)
        incomplete=cloned('incomplete-under-live-certificate')
        with closing(sqlite3.connect(incomplete/'work.sqlite3')) as db:
            with db:db.execute("DELETE FROM meta WHERE key IN ('summary','complete')")
        with mock.patch.object(p,'DEFAULT',incomplete),mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
            with self.assertRaises(ValueError):completed.run(incomplete)
        rejects.append('incomplete-under-live-certificate-no-fallback')
        for filename in ('manifest.json','summary.json'):
            path=cloned('missing-'+filename.replace('.','-'));(path/filename).unlink()
            with mock.patch.object(p,'DEFAULT',path):
                with self.assertRaises((ValueError,FileNotFoundError,sqlite3.OperationalError)):p.build()
            rejects.append('required-'+filename)
        for kind in ('metric','source'):
            bad=copy.deepcopy(original)
            if kind=='metric':bad['summary']['counts']['distinct_units']+=1
            else:bad['inputs']['congklak/neurodiag_metrics.py']='0'*64
            output=self.base/('corrupt-report-'+kind+'.json');save(output,bad)
            with self.assertRaises(ValueError):p.verify(output)
            rejects.append('rehashed-report-'+kind)
        for kind in ('witness-schema','witness-checks','witness-inputs','group-coverage'):
            if kind=='group-coverage':
                value=read(p.PRODUCTION_AUDIT);value['covered_methods']=value['covered_methods'][:1]
                output=self.base/(kind+'.json');save(output,value)
                with mock.patch.object(p,'PRODUCTION_AUDIT',output):
                    with self.assertRaises(ValueError):p.build()
                    with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
                        with self.assertRaises(ValueError):completed.run()
            else:
                value=read(p.WITNESS)
                if kind=='witness-schema':value['schema']='incorrect'
                elif kind=='witness-checks':value['checks']['analytic_derivatives']=1
                else:value['inputs'].pop('tests/neurodiag_independent_audit.py')
                output=self.base/(kind+'.json');save(output,value)
                with mock.patch.object(p,'WITNESS',output):
                    with self.assertRaises(ValueError):p.build()
                    with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
                        with self.assertRaises(ValueError):completed.run()
            rejects.append(kind)
        with mock.patch.object(p,'WITNESS',self.base/'absent-witness.json'):
            with self.assertRaises(FileNotFoundError):p.build()
            with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
                with self.assertRaises(FileNotFoundError):completed.run()
        rejects.append('missing-witness')
        with mock.patch.object(p,'WITNESS',self.base/'absent-witness.json'),mock.patch.object(p,'PRODUCTION_AUDIT',self.base/'absent-production-proof.json'):
            with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',forbidden):
                with self.assertRaises(FileNotFoundError):completed.run()
        rejects.append('complete-marker-with-both-certificates-missing-no-fallback')
        empty=self.base/'empty-startup';empty.mkdir()
        with closing(sqlite3.connect(empty/'work.sqlite3')):pass
        with mock.patch.object(p,'DEFAULT',empty),mock.patch.object(p,'WITNESS',self.base/'absent-witness.json'),mock.patch.object(p,'PRODUCTION_AUDIT',self.base/'absent-production-proof.json'):
            with mock.patch.object(p.phase6_store.Work,'__init__',forbidden),mock.patch.object(p,'run',return_value={'delegated':True}) as legacy:
                self.assertEqual(completed.run(empty),{'delegated':True});legacy.assert_called_once_with(empty,False,None,None)
        new_output=self.base/'repaired-new-report.json';self.assertFalse(new_output.exists())
        entrypoints=[dict(module='congklak.neurodiag_metrics',function=name) for name in
            ('observe','forward_arrays','head_diagnostics','target_diagnostics','literal_features','summarize_cached',
             'projection_diagnostic','feature_geometry','numpy_runtime','scalar_forward','prepare_models','load_models')]
        entrypoints += [dict(module='congklak.neurodiag_data',function='resolve_inputs'),dict(module='congklak.neurodiag',function='prepare_inputs')]
        entrypoints += [dict(module='congklak.phase6_store',function='Work'),dict(module='congklak.neurodiag',function='run')]
        spec=dict(watch_directories=[str(p.DEFAULT)],watch_baseline=True,numeric_entrypoints=entrypoints,
            extra_files=[rel(x) for x in (p.OUTPUT,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,p.INPUTS,WITNESS,p.PRODUCTION_AUDIT)],
            input_names=[rel(x) for x in (p.OUTPUT,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,p.INPUTS,WITNESS,p.PRODUCTION_AUDIT,FREEZE)],
            calls=[dict(module='congklak.neurodiag_v2',function='run'),dict(module='congklak.neurodiag_v2',function='raw_complete'),dict(module='congklak.neurodiag_v2',function='build')],
            clis=[dict(path='tools/diagnose-neural-v2.py',argv=['collect']),dict(path='tools/diagnose-neural-v2.py',argv=['raw']),
                dict(path='tools/diagnose-neural-v2.py',argv=['assess','--output',str(new_output)]),dict(path='tools/diagnose-neural-v2.py',argv=['verify','--output',str(new_output)])])
        proof=self.worker(spec,self.base/'closed-worker')
        self.assertEqual(read(new_output),original)
        again=dict(spec,extra_files=spec['extra_files']+[rel(new_output)],calls=[dict(module='congklak.neurodiag_v2',function='assess',kwargs=dict(output=str(new_output))),dict(module='congklak.neurodiag_v2',function='verify',kwargs=dict(output=str(new_output)))],clis=[])
        stable=self.worker(again,self.base/'repeated-output-worker')
        self.assertEqual(read(new_output),original);self.assertIs(proof['all_bytes_sizes_mtimes_unchanged'],True);self.assertIs(stable['all_bytes_sizes_mtimes_unchanged'],True)
        EVIDENCE['completed_corruption_and_idempotence']=dict(rejections=rejects,missing_new_output_rebuilt_without_compute=True,
            required_cache_and_certificate_missing_refused=True,all5377_prior_pins_current=True,
            completed_worker=rel(self.base/'closed-worker/proof.json'),repeated_output_worker=rel(self.base/'repeated-output-worker/proof.json'),
            new_forwards=0,new_features=0,new_analytic_gradients=0,new_SVD=0,new_batches=0,new_Native=0,new_training=0,new_games=0,
            current_completed_API='congklak.neurodiag_v2',current_completed_CLI='tools/diagnose-neural-v2.py',
            Work_constructor_and_legacy_fallback_trapped=True,empty_startup_legacy_routing_mocked_no_new_work=True)


def source_inputs():
    from congklak import neurodiag as p
    mapping=dict(read(BASELINE)['inputs'])
    paths=[Path(__file__),ORIGINAL_MAIN,ORIGINAL_FREEZE,V2_MAIN,V2_FREEZE,WORK/'preflight.json',WORK/'production.json',REPAIR,*files(FAILURE_ARCHIVE),*files(FINAL_FAILURE),COMPLETION_FREEZE,
        ROOT/'congklak/neurodiag_v2.py',ROOT/'tools/diagnose-neural-v2.py',ROOT/'docs/NEURODIAG_COMPLETED_READER.md',*files(ROOT/'build/neurodiag-completion-prefinal-v1'),ROOT/'tools/neurodiag-audit.py',BASELINE,p.PANEL,p.INVENTORY,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,p.INPUTS,
        ROOT/'build/neurodiag-data-author.json',ROOT/'build/neurodiag-metrics-author.json',ROOT/'build/neurodiag-metrics-implementation.json',
        ROOT/'congklak/neurodiag_progress.py',ROOT/'tools/neurodiag-dashboard.html',*[ROOT/name for name in p.SOURCES]]
    mapping.update(pins(paths));return mapping


def collect():
    mapping=source_inputs();mapping.update(pins([FREEZE]));evidence={};groups=[];methods=[]
    for group,expected in GROUPS.items():
        path=WORK/(group+'.json');body=read(path)
        assert body['schema']=='neurodiag-independent-group-v1' and body['group']==group and body['passed'] is True
        assert body['tests_run']==len(expected) and body['covered_methods']==expected and body['errors']==body['failures']==body['skips']==0
        assert body['covered_checks']==['__main__.NeurodiagIndependentAudit.'+name for name in expected]
        current(body['inputs'])
        for name,checksum in body['inputs'].items():assert name not in mapping or mapping[name]==checksum;mapping[name]=checksum
        for key,value in body['evidence'].items():assert key not in evidence;evidence[key]=value
        mapping[rel(path)]=sha(path);methods.extend(expected)
        executed=ORIGINAL_MAIN if group=='preflight' else V2_MAIN if group=='production' else Path(__file__)
        groups.append(dict(path=rel(path),sha256=sha(path),tests_run=len(expected),covered_methods=expected,
            executed_source=dict(path=rel(executed),sha256=sha(executed))))
    assert len(methods)==len(set(methods))==6
    mapping.update(pins([WITNESS]));current(mapping)
    save(ROOT/'build/neurodiag-audit.json',dict(schema='neurodiag-independent-audit-v1',passed=True,tests_run=6,errors=0,failures=0,skips=0,
        covered_methods=methods,covered_checks=['__main__.NeurodiagIndependentAudit.'+name for name in methods],groups=groups,inputs=mapping,evidence=evidence))
    print(canonical(dict(event='neurodiag_independent_aggregate',passed=True,tests=6)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('freeze','final','collect'));args=parser.parse_args()
    if args.operation=='freeze':
        save(FREEZE,dict(schema='neurodiag-independent-executed-source-v1',groups=GROUPS,execution_scope='final_only',original_source_sha256=sha(ORIGINAL_MAIN),original_freeze_sha256=sha(ORIGINAL_FREEZE),production_source_sha256=sha(V2_MAIN),production_freeze_sha256=sha(V2_FREEZE),inputs=source_inputs()));return 0
    if args.operation=='collect':collect();return 0
    frozen=read(FREEZE);assert frozen['schema']=='neurodiag-independent-executed-source-v1' and frozen['groups']==GROUPS;current(frozen['inputs'])
    output=WORK/'final-v3.json';assert not output.exists(),'Immutable group proof exists'
    started=time.monotonic();methods=GROUPS[args.operation]
    suite=unittest.TestSuite(NeurodiagIndependentAudit(name) for name in methods)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    mapping=source_inputs();mapping.update(pins([FREEZE]));directory=getattr(NeurodiagIndependentAudit,'base',None)
    if directory is not None:mapping.update(pins(files(directory)))
    if args.operation in ('production','final'):mapping.update(pins([WORK/'preflight.json',WITNESS]))
    if args.operation=='production':
        from congklak import neurodiag as p
        mapping.update(pins(files(p.DEFAULT)))
    if args.operation=='final':
        from congklak import neurodiag as p
        mapping.update(pins([WORK/'production.json',p.OUTPUT,*files(p.DEFAULT)]))
    save(output,dict(schema='neurodiag-independent-group-v1',group=args.operation,passed=result.wasSuccessful(),tests_run=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),covered_methods=methods,
        covered_checks=['__main__.NeurodiagIndependentAudit.'+name for name in methods],duration_seconds=time.monotonic()-started,
        retained_directory=rel(directory) if directory is not None else None,inputs=mapping,evidence=EVIDENCE,
        failure_details=[dict(method=str(test),trace=trace) for test,trace in result.errors+result.failures]))
    if args.operation=='final' and result.wasSuccessful():
        # The failed unsealed canonical proof is retained exactly in the archive.
        # Frozen public admission requires this canonical current path. Reissue it
        # only after the versioned executed report has actually passed.
        canonical_path=WORK/'final.json'
        assert sha(canonical_path)==sha(FINAL_FAILURE/'build/neurodiag-audit/final.json')
        temporary=WORK/'final-current-v3.tmp';temporary.write_bytes(output.read_bytes());os.replace(temporary,canonical_path)
    print(canonical(dict(event='neurodiag_independent_group',group=args.operation,passed=result.wasSuccessful(),tests=result.testsRun)),flush=True)
    return int(not result.wasSuccessful())


if __name__=='__main__':raise SystemExit(main())

