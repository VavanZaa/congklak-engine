"""New guarded pilot bindings to existing support and screened development data.

The optimizer rows are byte-for-byte the earlier 192-row ordinary cache. The
new ownership component is a separate reservation projection, never a changed
label or training example. Development labels/features are extracted only from
the already captured, independently replayed screening archives.
"""
from collections import Counter
from contextlib import closing,contextmanager
from contextvars import ContextVar
from fractions import Fraction
import hashlib,json,math,os,sqlite3,struct,time
from pathlib import Path
from types import MappingProxyType
import weakref
from . import config,guarded_value_pilot_v2_common as stage
from . import ordinary_epoch_data as prior_data,neurodiag_common as common
from .valuepilot_reservations import Writer

ROOT=stage.ROOT
DEFAULT=stage.DATA/'cache'
FIT_DEFAULT=stage.FOLDER/'fit-data'
OLD_CACHE=ROOT/'data/ordinary-epoch-pilot-v1/cache-v2'
INITIAL=prior_data.INITIAL
PRIOR=ROOT/'build/footprint-screen-v1/acceptance.json'
PRIOR_SHA='9648d0c7b3d18555b52e8fa038f55b24aa19a09d24eaaf454f9ce02398d72714'
RESERVATIONS=ROOT/'build/footprint-screen-v1/reservations/report.json'
ACQUISITION=ROOT/'build/footprint-screen-v1/acquisition'
EXACT=ROOT/'build/ordinary-value-benchmark-projection.json'
SHARED='shared-frozen-exact-guard'
CELLS=prior_data.CELLS
COUNTS=prior_data.COUNTS
SCHEMA='guarded-value-pilot-v2-data'
HOOKS=('before_data_catalog','after_data_catalog')
SOURCES=('congklak/guarded_value_pilot_v2_data.py','congklak/guarded_value_pilot_v2_common.py',
         'congklak/ordinary_epoch_data.py','congklak/ordinary_epoch_native.py','congklak/config.py')
need,read,sha,atomic=stage.need,stage.read,stage.sha,stage.atomic
fp32=prior_data.fp32
_SEALED={}
_MEMBERSHIP={}
_TRAIN_VIEWS={}
_VIEW_SEALS=weakref.WeakKeyDictionary()
_ACTIVE_VIEW=ContextVar('guarded_value_pilot_v2_data_source_view',default=None)

class _FrozenDict(dict):
    def _immutable(self,*args,**kwargs):raise TypeError('Verified source snapshot is immutable')
    __setitem__=__delitem__=clear=pop=popitem=setdefault=update=__ior__=_immutable

class _FrozenList(list):
    def _immutable(self,*args,**kwargs):raise TypeError('Verified source snapshot is immutable')
    __setitem__=__delitem__=append=clear=extend=insert=pop=remove=reverse=sort=__iadd__=__imul__=_immutable

def _freeze(value):
    if type(value) is dict:return _FrozenDict((k,_freeze(v)) for k,v in value.items())
    if type(value) is list:return _FrozenList(_freeze(v) for v in value)
    return value

def _same(a,b):return config.canonical(a)==config.canonical(b)
def _ref(path):return dict(path=stage.relative(path),sha256=sha(path))

def accepted():
    """Inspect the sealed admission and exact reservation sources, read-only."""
    need(sha(PRIOR)==PRIOR_SHA,'Screened reservation acceptance changed')
    marker=read(PRIOR)
    need(marker['passed'] is True and marker['reservation_admission'] is True and
         marker['training_authorized'] is False and marker['policy_fixed'] is True and
         marker['inherited_unknown_background'] is True and marker['historically_unseen'] is False,
         'Screened reservations are not admitted for a prospective recipe')
    ref=marker['closure_ref'];path=ROOT/ref['path']
    need(path==PRIOR.parent/'closure.json' and sha(path)==ref['sha256'],'Screening closure changed')
    report_sha=sha(RESERVATIONS);cache_key=(ref['sha256'],report_sha)
    if cache_key not in _SEALED:
        closure=read(path)
        need(closure['passed'] is True and closure['inputs_count']==len(closure['inputs']),
             'Screening preservation closure is incomplete')
        need(closure['inputs'].get(stage.relative(RESERVATIONS))==report_sha,'Screened role report changed')
        report=read(RESERVATIONS)
        need(report['reservation_admission'] is True and report['metadata_quotas_satisfied'] is True and
             report['independent_full_raw_audit_passed'] is True and report['all_acquired_footprints_retained'] is True and
             report['selected_distinct_components']==152 and report['quality_values_used_for_selection'] is False and
             report['old320_quarantine_preserved'] is True and report['training_authorized'] is False,
             'Screening role/closure scope differs')
        need(sha(PRIOR)==PRIOR_SHA and sha(path)==ref['sha256'] and sha(RESERVATIONS)==report_sha,
             'Screening sources changed while constructing verified view')
        _SEALED.clear();_SEALED[cache_key]=(_freeze(report),MappingProxyType(dict(closure['inputs'])))
    report,inputs=_SEALED[cache_key]
    return report,inputs

def _source(path,inputs):
    need(inputs.get(stage.relative(path))==sha(path),'Sealed source changed: '+stage.relative(path))
    return read(path)

def membership():
    """Pure role metadata; neither values, features nor an encoder are opened."""
    report,inputs=accepted()
    pins=common.pins([PRIOR,PRIOR.parent/'closure.json',RESERVATIONS,prior_data.OLD_SELECTION,EXACT,Path(__file__)])
    cache_key=config.digest(pins)
    if cache_key not in _MEMBERSHIP:
        out=_derive_membership(report,inputs);out['cases']=sorted(out['cases'])
        need(common.pins(ROOT/name for name in pins)==pins,'Membership sources changed during derivation')
        _MEMBERSHIP.clear();_MEMBERSHIP[cache_key]=config.canonical(out).encode()
    result=json.loads(_MEMBERSHIP[cache_key]);result['cases']=set(result['cases']);return result

def _derive_membership(report,inputs):
    old=_source(prior_data.OLD_SELECTION,inputs)['train']
    lookup={r['state_key']:r for r in old};support=report['support192_preserved']
    need(len(old)==len(lookup)==512 and len(support)==192,'Inherited support population differs')
    out={name:[] for name in ('train','fit','arena')}
    for reserved in support:
        meta=lookup[reserved['state_key']];cid=report['support192_current_components'][meta['state_key']]
        need(reserved['eligible_for_future_ordinary_recipe'] is True and meta['family']==reserved['family'] and
             meta['cell']==reserved['cell'] and '/ordinary/' in meta['cell'] and
             report['components'][cid]['acquisition_exposed'] is False,
             'Support would admit an acquired or changed origin')
        out['train'].append(dict(meta,role='optimizer_support',origin='optimizer_support/'+meta['family'],
            source_component=reserved['joint_component'],component=cid,current_component=cid,joint_component=cid,
            support_reservation=reserved))
    need(Counter(m['cell'] for m in out['train'])==Counter(COUNTS),'Four support cells changed')
    need(sum(Fraction(m['support_reservation']['weight']['numerator'],m['support_reservation']['weight']['denominator'])
             for m in out['train'])==1,'Support sampling mass changed')
    for role,count in (('fit',128),('arena',24)):
        need(len(report['roles'][role])==count,'Screened role quota differs')
        for meta in report['roles'][role]:
            cid=meta['component'];summary=report['components'][cid]
            need(summary['evaluation_blocked'] is False and summary['acquisition_exposed'] is True and
                 summary['acquired_origins_optimizer_blocked'] is True and
                 meta['origin'] in summary['acquired_origins'],
                 'Selected development component is not protected')
            out[role].append(dict(meta,role=role,joint_component=cid,current_component=cid,
                source_origin=meta['origin'],origin=meta['origin'],optimizer_blocked=True))
    fit={m['state_key']:m for m in report['roles']['fit']};cases=report['roles']['cases']
    need(len(cases)==len({m['state_key'] for m in cases})==32,'Case subset differs')
    for meta in cases:
        expected=dict(fit[meta['state_key']]);expected.update(role='cases',fit_root_id=expected['id'])
        need(_same(meta,expected),'Case is not an exact selected fit root')
    out['cases']={m['state_key'] for m in cases}
    exact=_source(EXACT,inputs)['cases']
    need(len(exact)==len({m['state_key'] for m in exact})==32,'Shared exact guard quota changed')
    out['exact']=[dict(m,role='shared_guard',origin=SHARED,component=SHARED,joint_component=SHARED,
                       current_component=SHARED,cell=m['stratum']) for m in exact]
    for meta in out['exact']:
        need(sum(map(sum,meta['state']['pits']))<=4,'Shared exact guard exceeds K4')
    core=out['train']+out['fit']+out['arena']
    need(len({m['component'] for m in core})==len({m['origin'] for m in core})==344,
         'Support and selected evaluation roles share a component/origin')
    for meta in core+out['exact']:
        state=config.position(meta['state'])
        need(meta['state_key']==common.rawkey(state) and sum(map(sum,state['pits']))+sum(state['stores'])==98,
             'Reserved root state geometry differs')
    return out

class _TrainSourceView:
    """Constructor-issued exact source admission; rows remain canonical bytes."""
    def __setattr__(self,name,value):
        need(not getattr(self,'_sealed',False),'Training source view is immutable')
        object.__setattr__(self,name,value)
    def __init__(self):
        pins,narrow=_source_view_pins()
        # Issuance itself performs the literal old-source derivation. A caller
        # cannot bless invented rows by invoking this constructor directly.
        rows,manifest=prior_data.load_train(OLD_CACHE)
        need(len(rows)==192 and [r['state_key'] for r in rows]==[m['state_key'] for m in membership()['train']],
             'Inherited optimizer row order differs')
        common.current(pins)
        self.pins=MappingProxyType(dict(pins));self.narrow=MappingProxyType(dict(narrow))
        self.rows=config.canonical(rows).encode();self.manifest=config.canonical(manifest).encode();self._sealed=True
        _VIEW_SEALS[self]=(self.pins,self.narrow,self.rows,self.manifest)
    def assert_sealed(self):
        seal=_VIEW_SEALS.get(self)
        need(type(self) is _TrainSourceView and seal is not None and
             all(getattr(self,k) is v for k,v in zip(('pins','narrow','rows','manifest'),seal)),
             'Training source view lacks constructor admission')
    def verify(self,full=True):
        self.assert_sealed();common.current(self.pins if full else self.narrow);return True
    def unpack(self):return json.loads(self.rows),json.loads(self.manifest)

def _source_view_pins():
    _,sealed=accepted();files=[OLD_CACHE/name for name in
        ('identity.json','train.json','manifest.json','certificate.json','catalog.json','complete.json')]
    manifest=_source(OLD_CACHE/'manifest.json',sealed)
    original=manifest['source_identity']['source_inputs']
    for name,expected in original.items():
        need(sealed.get(name)==expected,'Old training proof does not match accepted preservation closure')
    files.extend(ROOT/name for name in original)
    files.extend([PRIOR,PRIOR.parent/'closure.json',RESERVATIONS,prior_data.OLD_SELECTION,EXACT,*[ROOT/name for name in SOURCES]])
    files=sorted(set(Path(p).resolve() for p in files))
    pins=common.pins(files)
    for path in files:
        name=stage.relative(path)
        if name in sealed:need(pins[name]==sealed[name],'Training source changed after accepted closure')
    narrow={name:value for name,value in pins.items() if name not in original}
    # Source-code pins stay in the cheap subset even when an old dependency also
    # appears in the original manifest. Only large immutable data gets scoped.
    narrow.update({name:pins[name] for name in original if name.endswith('.py')})
    return pins,narrow

def _train_view():
    active=_ACTIVE_VIEW.get()
    if active is not None:active.verify(False);return active
    pins,narrow=_source_view_pins();cache_key=config.digest(pins)
    if cache_key not in _TRAIN_VIEWS:
        view=_TrainSourceView()
        need(dict(view.pins)==pins and dict(view.narrow)==narrow,'Sources changed during source-view issuance')
        _TRAIN_VIEWS.clear();_TRAIN_VIEWS[cache_key]=view
    view=_TRAIN_VIEWS[cache_key];view.assert_sealed();return view

@contextmanager
def admission_scope():
    """Finite worker scope: full fresh hashes before AND after any reuse.

    Every inner admission still verifies all cache artifacts, reservations and
    executable source hashes. A changed original teacher/feature source causes
    the closing check to fail; no caller may accept a worker before it closes.
    """
    need(_ACTIVE_VIEW.get() is None,'Nested training-source scope is not admitted')
    view=_train_view();view.verify(True);token=_ACTIVE_VIEW.set(view)
    try:yield view
    finally:
        try:view.verify(True)
        finally:_ACTIVE_VIEW.reset(token)

def _train_rows():return _train_view().unpack()

def train_identity():
    members=membership();rows,old=_train_rows()
    sources=[PRIOR,PRIOR.parent/'closure.json',RESERVATIONS,*[OLD_CACHE/name for name in
        ('identity.json','train.json','manifest.json','certificate.json','catalog.json','complete.json')],
        *[ROOT/name for name in SOURCES]]
    return dict(schema=SCHEMA+'-identity',role='train',plan_sha256=sha(stage.PLAN),
        source_inputs=common.pins(sources),source_cache_manifest_sha256=sha(OLD_CACHE/'manifest.json'),
        source_row_sha256=[config.digest(r) for r in rows],count=192,cells=COUNTS,
        current_components={m['state_key']:m['component'] for m in members['train']},
        old_components_retained_in_training_payload=True,features_and_targets_unchanged=True,
        target=old['source_identity']['target'],sampler=old['source_identity']['sampler'],
        acquired_origins_optimizer_blocked=True,inherited_unknown_background=True,historically_unseen=False,
        policy_fixed=True,optimizer_steps=0,model_forwards=0,feature_simulations=0)

def _pause(path,hook,pause_at,pause_marker):
    if pause_at!=hook:return
    need(path.is_relative_to(stage.FOLDER/'fixtures') and pause_marker is not None,'Data pause requires registered fixture')
    atomic(pause_marker,dict(point=hook,pid=os.getpid(),output=str(path)))
    while True:time.sleep(.05)

def _prepare(path,identity,rows,role,pause_at=None,pause_marker=None):
    path=stage.path_of(path);need(pause_at in (None,*HOOKS),'Unknown data pause hook')
    path.mkdir(parents=True,exist_ok=True)
    if (path/'catalog.json').exists():return _admit(path,identity,rows,role)
    with Writer(path):
        atomic(path/'identity.json',identity)
        atomic(path/(role+'.json'),dict(schema=SCHEMA+'-role',role=role,rows=rows))
        manifest=dict(schema=SCHEMA+'-manifest',role=role,identity_sha256=sha(path/'identity.json'),
            source_identity=identity,role_files={role:dict(filename=role+'.json',sha256=sha(path/(role+'.json')),
                count=len(rows),row_sha256=[config.digest(r) for r in rows],components=[r['component'] for r in rows])},
            current_components=identity['current_components'],labels_read=len(rows),features_packed=len(rows),
            fit_targets_read=0 if role=='train' else len(rows),arena_targets_read=0,
            feature_simulations=0,model_forwards=0,optimizer_steps=0,policy_fixed=True)
        if role=='fit':manifest['case_state_keys']=identity['case_state_keys']
        atomic(path/'manifest.json',manifest)
        atomic(path/'certificate.json',dict(schema=SCHEMA+'-certificate',passed=True,
            files=common.pins(path/name for name in ('identity.json',role+'.json','manifest.json'))))
        _pause(path,'before_data_catalog',pause_at,pause_marker)
        atomic(path/'catalog.json',dict(schema=SCHEMA+'-catalog',complete=True,
            manifest_sha256=sha(path/'manifest.json'),certificate_sha256=sha(path/'certificate.json')))
        _pause(path,'after_data_catalog',pause_at,pause_marker)
        atomic(path/'complete.json',dict(catalog_sha256=sha(path/'catalog.json')))
    return _admit(path,identity,rows,role)

def _admit(path,identity,rows,role):
    path=stage.path_of(path);catalog=read(path/'catalog.json');manifest=read(path/'manifest.json');cert=read(path/'certificate.json')
    need(_same(read(path/'identity.json'),identity),'Data identity no longer follows sealed sources')
    need(catalog==dict(schema=SCHEMA+'-catalog',complete=True,manifest_sha256=sha(path/'manifest.json'),
         certificate_sha256=sha(path/'certificate.json')) and cert['schema']==SCHEMA+'-certificate' and cert['passed'] is True,
         'Data catalog/certificate differs')
    expected=common.pins(path/name for name in ('identity.json',role+'.json','manifest.json'))
    need(_same(cert['files'],expected),'Data certificate closure differs');common.current(expected)
    need(_same(read(path/(role+'.json')),dict(schema=SCHEMA+'-role',role=role,rows=rows)) and
         manifest['source_identity']==identity and manifest['identity_sha256']==sha(path/'identity.json') and
         manifest['current_components']==identity['current_components'] and set(manifest['role_files'])=={role},
         'Data payload or ownership projection differs')
    ref=manifest['role_files'][role]
    need(ref==dict(filename=role+'.json',sha256=sha(path/(role+'.json')),count=len(rows),
        row_sha256=[config.digest(r) for r in rows],components=[r['component'] for r in rows]),'Row evidence differs')
    need(manifest['schema']==SCHEMA+'-manifest' and manifest['role']==role and manifest['labels_read']==len(rows) and
         manifest['features_packed']==len(rows) and manifest['fit_targets_read']==(0 if role=='train' else len(rows)) and
         manifest['arena_targets_read']==0 and manifest['policy_fixed'] is True and
         all(type(manifest[k]) is int and manifest[k]==0 for k in ('feature_simulations','model_forwards','optimizer_steps')),
         'Unexpected data activity')
    if role=='fit':need(manifest['case_state_keys']==identity['case_state_keys'],'Fit case projection differs')
    if (path/'complete.json').exists():
        need(read(path/'complete.json')==dict(catalog_sha256=sha(path/'catalog.json')),'Data completion marker differs')
    return manifest

def prepare_train(output=DEFAULT,pause_at=None,pause_marker=None):
    stage.admit_plan(False);rows,_=_train_rows()
    return _prepare(output,train_identity(),rows,'train',pause_at,pause_marker)

prepare=prepare_train

def admit(output=DEFAULT):
    stage.admit_plan(False);rows,_=_train_rows();return _admit(output,train_identity(),rows,'train')

def load_train(output=DEFAULT):
    manifest=admit(output);return read(stage.path_of(output)/'train.json')['rows'],manifest

def fixture_rows(rows):return prior_data.fixture_rows(rows)

def _fit_rows():
    from .ordinary_epoch_native import teacher_certificate
    report,inputs=accepted();members=membership();catalog=_source(ACQUISITION/'catalog.json',inputs)
    ident=_source(ACQUISITION/'identity.json',inputs);rows=[];source_refs={}
    for meta in members['fit']:
        index=meta['index'];archives=[]
        for offset,kind in ((0,'teacher'),(1,'features')):
            cursor=2*index+offset;path=ACQUISITION/'archives'/f'query-{cursor:06d}.json';body=_source(path,inputs)
            ref=_ref(path);source_refs[ref['path']]=ref['sha256']
            need(catalog['archives'].get(ref['path'])==ref['sha256'] and body['identity_sha256']==config.digest(ident) and
                 body['scheduled']==ident['schedule'][cursor] and body['scheduled']['kind']==kind and
                 body['root_id']==meta['family'] and body['origin']==meta['source_origin'] and body['status']=='recorded' and
                 body['answer'].get('fast_error') is None and not body['answer'].get('archival_errors') and
                 body['answer']['trace']['binding']['root_state']==meta['state'],
                 'Captured teacher/features do not bind the reserved root')
            archives.append((body,ref))
        (a,ar),(b,br)=archives;teacher=teacher_certificate(a['answer'],meta['state'])
        probe=b['answer']['probe'];x=probe['result']['features'];state=meta['state'];actor=state['controller']
        D=state['stores'][0]-state['stores'][1];r=sum(map(sum,state['pits']))*(1 if actor==0 else -1)
        y=teacher['target_player0'];z=(y-D)/r
        need(probe['kind']=='features' and probe['engine']=='control' and probe['state']==state and
             len(x)==127 and all(type(v) in (int,float) and math.isfinite(v) for v in x) and -1<=z<=1,
             'Captured development features/normalization differ')
        replay={name:report['audit'][name] for name in ('catalog_ref','report_ref')}
        teacher.update(source_archive_ref=ar,source_full_raw_replay=replay)
        rows.append(dict(state=state,state_key=meta['state_key'],family=meta['family'],cell=meta['cell'],
            component=meta['component'],joint_component=meta['component'],origin=meta['origin'],
            stratum=meta['cell'].rsplit('/actor',1)[0],actor=actor,x=[fp32(v) for v in x],x_native=x,
            D=D,r=r,target_player0=y,target_z=z,is_case=meta['state_key'] in members['cases'],teacher=teacher,
            feature_source=dict(source_archive_ref=br,source_full_raw_replay=replay)))
    need(len(rows)==128 and sum(r['is_case'] for r in rows)==32,'Fit/case extraction count differs')
    return rows,source_refs

def fit_identity(rows=None,source_refs=None):
    if rows is None:rows,source_refs=_fit_rows()
    members=membership();sources=common.pins([PRIOR,PRIOR.parent/'closure.json',RESERVATIONS,
        ACQUISITION/'identity.json',ACQUISITION/'catalog.json',*[ROOT/name for name in SOURCES]])
    sources.update(source_refs)
    return dict(schema=SCHEMA+'-identity',role='fit',plan_sha256=sha(stage.PLAN),source_inputs=sources,
        count=128,cells=dict(Counter(r['cell'] for r in rows)),case_state_keys=sorted(members['cases']),
        current_components={r['state_key']:r['component'] for r in rows},
        source_row_sha256=[config.digest(r) for r in rows],captured_labels_only=True,new_teacher_queries=0,
        new_features_encoded=0,acquired_origins_optimizer_blocked=True,inherited_unknown_background=True,
        historically_unseen=False,policy_fixed=True,optimizer_steps=0,model_forwards=0,feature_simulations=0)

def prepare_fit(output=FIT_DEFAULT,pause_at=None,pause_marker=None):
    stage.admit_plan(False);rows,refs=_fit_rows()
    return _prepare(output,fit_identity(rows,refs),rows,'fit',pause_at,pause_marker)

def load_fit(output=FIT_DEFAULT):
    stage.admit_plan(False);rows,refs=_fit_rows();manifest=_admit(output,fit_identity(rows,refs),rows,'fit')
    return rows,manifest
