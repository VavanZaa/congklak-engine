"""Independent pure derivation of reserved query identities and exact schedules.

No targets, encoders, native providers, model exports or optimizers are opened.
The registry is derived from the accepted role metadata rather than an executor
file's claims about its own origin list.
"""
import hashlib,json
from collections import Counter
from . import config,ordinary_epoch_common as s

SELECTION=s.ROOT/'build/ownership-implementation-v1/reservations/production-v2/selection.json'
SUPPORT=s.ROOT/'data/neural-value-pilot-v2/reservations/selection.json'
EXACT=s.ROOT/'build/ordinary-value-benchmark-projection.json'
SCHEMA=s.SCHEMA+'-queries'
OPERATIONS=('labels','conformance','guards','arena')
SHARED='shared-frozen-exact-guard'
LABEL_KILL_HOOKS=('before_decision_commit','after_decision_commit','before_ownership_check_commit',
                 'after_ownership_check_commit','before_catalog','after_catalog')
HANDOFF_KILL_HOOKS=('before_handoff_commit','after_handoff_commit')
UNKNOWN_HOOK='before_query'
FIXTURE_BASE=s.FOLDER/'fixtures/native-recovery-v1'
_CACHE={}
_LOADED_SHA=s.sha(__file__)

def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return dict(path=s.relative(path),sha256=s.sha(path))
def _salt(prefix,value):return hashlib.sha256((prefix+value).encode()).hexdigest()

def _pins():
    s.need(s.sha(__file__)==_LOADED_SHA,'Loaded query contract source changed')
    s.need(s.sha(s.PRIOR)==s.PRIOR_SHA,'Accepted query reservations changed')
    pins=s.common.pins([s.JOINT,SELECTION,SUPPORT,EXACT]);accepted=s.read(s.PRIOR)
    s.need(all(accepted['inputs'].get(k)==v for k,v in pins.items()),'Query metadata differs from accepted source bytes')
    return pins

def _raw(meta):
    state=config.position(meta['state'])
    s.need(meta['state_key']==s.common.rawkey(state) and type(meta['family']) is str and bool(meta['family']),
           'Reserved metadata raw state/family differs')
    s.need(sum(map(sum,state['pits']))+sum(state['stores'])==98,'Reserved query state seed conservation differs')
    return state

def membership():
    pins=_pins();key=config.digest(pins)
    if key in _CACHE:
        result=json.loads(_CACHE[key]);result['cases']=set(result['cases']);return result
    joint=s.joint();selected=s.read(SELECTION)['roles'];old=s.read(SUPPORT)['train']
    s.need(set(selected)=={'fit','cases','arena'} and len(old)==512,'Original selected role/source pool differs')
    lookup={(r['role'],r['state_key']):r for r in joint['conditional_role_revalidation']}
    s.need(len(lookup)==184,'Conditional role proofs have duplicates or missing roots')
    out={name:[] for name in ('fit','arena','train')}
    for role in ('fit','arena'):
        s.need(len(selected[role])==(128 if role=='fit' else 24),'Reserved evaluation quota differs')
        s.need(len({m['state_key'] for m in selected[role]})==len(selected[role]),'Repeated evaluation raw root')
        for meta in selected[role]:
            state=_raw(meta);p=lookup[(role,meta['state_key'])];cid=p['joint_component']
            s.need(p['known_recorded_disjoint'] is True and p['family']==meta['family'] and p['cell']==meta['cell'] and
                cid in joint['joint_components'] and cid not in joint['whole_quarantine_joint_components'] and
                25<=sum(map(sum,state['pits']))<=74,'Evaluation root lost its complete component reservation')
            out[role].append(dict(meta,component=cid,joint_component=cid,role=role,origin=role+'/'+meta['family']))
    old_lookup={m['state_key']:m for m in old};support=joint['future_support']['roots']
    s.need(len(old_lookup)==512 and len(support)==192,'Ordinary support projection differs')
    for p in support:
        meta=old_lookup[p['state_key']];state=_raw(meta);cid=p['joint_component']
        s.need(p['eligible_for_future_ordinary_recipe'] is True and p['family']==meta['family'] and
            p['cell']==meta['cell'] and p['quotient']==meta['quotient'] and
            cid not in joint['whole_quarantine_joint_components'] and
            joint['joint_components'][cid]['optimizer_blocked'] is False and '/ordinary/' in meta['cell'] and
            25<=sum(map(sum,state['pits']))<=74,'A queried support root is not admitted ordinary support')
        out['train'].append(dict(meta,component=cid,joint_component=cid,role='optimizer_support',
            origin='optimizer_support/'+meta['family']))
    expected=dict(zip(('25-49/ordinary/actor0','25-49/ordinary/actor1','50-74/ordinary/actor0','50-74/ordinary/actor1'),(39,34,58,61)))
    s.need(Counter(m['cell'] for m in out['train'])==Counter(expected),'Ordinary queried support cells differ')
    fit_lookup={r['state_key']:r for r in selected['fit']};cases=selected['cases']
    s.need(len(cases)==len({r['state_key'] for r in cases})==32,'Reserved32 case roots differ')
    for meta in cases:
        s.need(meta['state_key'] in fit_lookup and _same(meta,fit_lookup[meta['state_key']]) and
            lookup[('cases',meta['state_key'])]['joint_component']==lookup[('fit',meta['state_key'])]['joint_component'],
            'Case root is not an exact subset of its fit origin')
    exact=s.read(EXACT)['cases'];s.need(len(exact)==len({r['state_key'] for r in exact})==32,'Inherited exact guard quota differs')
    for meta in exact:
        state=_raw(meta);s.need(sum(map(sum,state['pits']))<=4,'Inherited exact guard outside shared K4 domain')
    out['exact']=[dict(m,role='shared_guard',origin=SHARED,component=SHARED,joint_component=SHARED,cell=m['stratum']) for m in exact]
    core=out['train']+out['fit']+out['arena']
    s.need(len({r['origin'] for r in core})==len({r['component'] for r in core})==344,
           'Reserved roles share an origin/component')
    out['cases']=sorted(r['state_key'] for r in cases)
    _CACHE[key]=config.canonical(out);result=json.loads(_CACHE[key]);result['cases']=set(result['cases']);return result

def schedule(operation,fixture=False):
    s.need(operation in OPERATIONS and type(fixture) is bool,'Unknown query schedule scope')
    m=membership();rows=[]
    if operation=='labels':
        for meta in (m['fit'][:1] if fixture else m['fit']):
            for kind,engine in (('teacher','H_v1'),('features','control')):rows.append(dict(kind=kind,engine=engine,meta=meta))
    elif operation=='conformance':
        if fixture:
            meta=m['fit'][0];rows.append(dict(kind='features',engine='control',meta=meta))
            for engine in ('control','candidate'):rows.append(dict(kind='prediction',engine=engine,meta=meta))
            for engine in ('control','candidate'):rows.append(dict(kind='decision',engine=engine,mode='fixed128',meta=meta,cohort='initial_equivalence'))
        else:
            roots=sorted(m['fit'],key=lambda r:_salt('ordinary-epoch-conformance-20261016/',r['state_key']))
            for i,meta in enumerate(roots):
                for engine in (('control','candidate') if i%2==0 else ('candidate','control')):
                    rows.append(dict(kind='prediction',engine=engine,meta=meta))
    elif operation=='guards':
        cases=[r for r in m['fit'] if r['state_key'] in m['cases']]
        for cohort,roots in (('cases',cases[:1] if fixture else cases),('endgame',m['exact'][:1] if fixture else m['exact'])):
            for i,meta in enumerate(sorted(roots,key=lambda r:_salt('ordinary-epoch-cases-20261016/',r['state_key']))):
                for j,mode in enumerate(('timed25','fixed128') if i%2==0 else ('fixed128','timed25')):
                    for engine in (('control','candidate') if (i+j)%2==0 else ('candidate','control')):
                        rows.append(dict(kind='decision',engine=engine,mode=mode,meta=meta,cohort=cohort))
    else:
        roots=sorted(m['arena'],key=lambda r:_salt('ordinary-epoch-arena-20261016/',r['state_key']))
        if fixture:
            return [dict(index=0,kind='game',meta=roots[0],baseline='H_v1',seat=0,profile='candidate')]
        cycle=[('H_v1',0),('H8_relay_CPU',1),('H_v1',1),('H8_relay_CPU',0)]
        for i,meta in enumerate(roots):
            offset=i%4;matches=cycle[offset:]+cycle[:offset]
            for j,(baseline,seat) in enumerate(matches):
                for profile in (('control','candidate') if (i+j)%2==0 else ('candidate','control')):
                    rows.append(dict(kind='game',meta=meta,baseline=baseline,seat=seat,profile=profile))
    return [dict(row,index=i) for i,row in enumerate(rows)]

def classify(output,operation,fixture=False):
    path=s.path_of(output);s.need(operation in OPERATIONS and type(fixture) is bool,'Unknown campaign scope')
    if not fixture:
        s.need(path==s.FOLDER/'jobs'/operation,'Unregistered production query path')
        return dict(scope='production',subset_only=False,hook=None)
    if path==s.FOLDER/'jobs/initial-conformance':
        s.need(operation=='conformance','Initial equivalence operation differs')
        return dict(scope='initial_equivalence',subset_only=False,hook=None)
    s.need(path.parent==FIXTURE_BASE,'Unregistered native recovery fixture path')
    hook=path.name
    if hook in LABEL_KILL_HOOKS or hook==UNKNOWN_HOOK:
        s.need(operation=='labels','Label recovery fixture operation differs')
        return dict(scope='label_subset_recovery',subset_only=True,hook=hook)
    s.need(hook in HANDOFF_KILL_HOOKS and operation=='arena','Unregistered handoff recovery fixture')
    return dict(scope='handoff_recovery',subset_only=False,hook=hook)

def predecessor(output,operation,fixture=False):
    scope=classify(output,operation,fixture)
    if scope['scope'] in ('label_subset_recovery','initial_equivalence'):
        path=s.FOLDER/'jobs/labels';op='labels'
    elif scope['scope']=='handoff_recovery':
        path=(s.FOLDER/'jobs/guards' if scope['hook']=='before_handoff_commit'
              else FIXTURE_BASE/'before_handoff_commit')
        op='guards' if scope['hook']=='before_handoff_commit' else 'arena'
    elif operation=='labels':return None
    elif operation=='conformance':path=s.FOLDER/'jobs/initial-conformance';op='conformance'
    elif operation=='guards':path=s.FOLDER/'jobs/conformance';op='conformance'
    else:path=FIXTURE_BASE/'after_handoff_commit';op='arena'
    return dict(path=s.relative(path),operation=op)

def derive_registry(plan_sha256=None):
    m=membership();joint=s.joint();rows=m['train']+m['fit']+m['arena']+m['exact']
    origins={r['origin']:r['component'] for r in rows};s.need(len(origins)==345,'Exact global origin registry differs')
    graph=dict(schema=SCHEMA+'-known-graph',components={cid:{} for cid in joint['joint_components']},
        quotient_components=joint['joint_quotient_components'],origin_components=origins,source=_ref(s.JOINT),
        role_roots=[{k:r[k] for k in ('state_key','family','role','origin','component')} for r in rows])
    graph['components'][SHARED]={}
    graph_ref=dict(path=s.relative(s.FOLDER/'guard/known-graph.json'),
        sha256=hashlib.sha256((config.canonical(graph)+'\n').encode()).hexdigest())
    registry=dict(schema=SCHEMA+'-registry',plan_sha256=plan_sha256 or s.sha(s.PLAN),graph=graph_ref,
        origins=sorted(origins),origin_components=origins,selection=_ref(SELECTION),exact_guard=_ref(EXACT),role_roots=graph['role_roots'])
    return registry,graph
