"""Exact192-root ordinary epoch cache from existing H8 labels/features only.

Preparation has no native, model, feature-encoding or Torch operation.  New fit
labels remain a separate sealed role and are never loaded by this module.
"""
from collections import Counter
from contextlib import closing
from fractions import Fraction
import json,math,os,sqlite3,struct,time
from pathlib import Path
from . import config,ordinary_epoch_common as stage,neurodiag_common as common
from . import neurodiag_data as previous,value_training_data as historical
from .valuepilot_reservations import Writer

ROOT=stage.ROOT
DEFAULT=stage.DATA/'cache-v2'
SCHEMA='ordinary-epoch-data-v1'
OLD_SELECTION=ROOT/'data/neural-value-pilot-v2/reservations/selection.json'
INITIAL=ROOT/'data/neural-value-pilot-v2/interface-initial/model.json'
CELLS=tuple(b+'/ordinary/actor'+str(a) for b in ('25-49','50-74') for a in (0,1))
COUNTS=dict(zip(CELLS,(39,34,58,61)))
SOURCES=('congklak/ordinary_epoch_data.py','congklak/ordinary_epoch_common.py',
 'congklak/value_training_data.py','congklak/neurodiag_data.py','congklak/neurodiag_common.py',
 'congklak/features_relay.py','congklak/features_v3.py','congklak/config.py')
HOOKS=('before_data_catalog','after_data_catalog')
need,read,sha=stage.need,stage.read,stage.sha
atomic=stage.atomic
_ROW_CACHE={}

def _same(a,b):return config.canonical(a)==config.canonical(b)
def _path(output):return stage.path_of(output)
def fp32(v):return struct.unpack('<f',struct.pack('<f',v))[0]

def membership():
    """Pure reservation metadata; no teacher outcomes or cached features read."""
    need(sha(stage.PRIOR)==stage.PRIOR_SHA,'New ordinary reservation acceptance changed')
    accepted=read(stage.PRIOR)
    for p in (stage.JOINT,stage.EPOCH,OLD_SELECTION):
        need(accepted['inputs'].get(stage.relative(p))==sha(p),'Reservation membership source changed')
    joint=stage.joint();epoch=stage.epoch();roots=joint['future_support']['roots']
    need(epoch['prerequisites_ready'] is True and epoch['ordinary_support_roots']==192 and
         joint['original_512_recipe_admitted'] is False and joint['historical_provenance_complete'] is False and
         joint['inherited_unknown_background'] is True and len(roots)==192,'Ordinary epoch scope differs')
    old=read(OLD_SELECTION)['train'];lookup={r['state_key']:r for r in old}
    need(len(old)==len(lookup)==512 and len({r['state_key'] for r in roots})==192,'Old support identities differ')
    out=[]
    for selected in roots:
        meta=lookup[selected['state_key']]
        need(meta['family']==selected['family'] and meta['cell']==selected['cell'] and
             meta['quotient']==selected['quotient'] and selected['eligible_for_future_ordinary_recipe'] is True and
             selected['joint_component'] not in joint['whole_quarantine_joint_components'] and
             joint['joint_components'][selected['joint_component']]['optimizer_blocked'] is False,
             'A selected support root is outside the new ordinary epoch')
        need(meta['cohort']=='base_H8' and len(meta['target_refs'])==1 and
             meta['target_refs'][0]['journal']=='deeper-9000-v2' and meta['target_refs'][0]['depth']==8,
             'Mixed or changed target source')
        out.append(dict(metadata=meta,reservation=selected))
    need(Counter(r['metadata']['cell'] for r in out)==Counter(COUNTS),'Ordinary four-cell counts differ')
    need(sum(Fraction(r['reservation']['weight']['numerator'],r['reservation']['weight']['denominator']) for r in out)==1,
         'Prospective root weights do not sum to one')
    return out

def input_pins():
    rows=membership();files=[stage.PRIOR,stage.JOINT,stage.EPOCH,OLD_SELECTION,INITIAL,ROOT/'build/training-runtime.json']
    files.extend(ROOT/p for p in SOURCES)
    files.extend(ROOT/'data'/r['metadata']['target_refs'][0]['journal']/'work.sqlite3' for r in rows)
    files.extend(ROOT/r['metadata']['feature_ref']['path'] for r in rows)
    files.extend(ROOT/p for p in stage.MODELS)
    # pins() overwrites duplicate map keys only after hashing. Resolve/deduplicate
    # first so the same218MB teacher database is read once, not192 times.
    return common.pins(sorted({Path(p).resolve() for p in files}))

def identity():
    selected=membership()
    return dict(schema=SCHEMA+'-identity',rules_digest=config.RULES_DIGEST,
        source_inputs=input_pins(),plan_sha256=sha(stage.PLAN),epoch_id=stage.epoch()['epoch_id'],
        membership=[config.digest(r) for r in selected],count=192,cells=COUNTS,
        target=dict(kind='original-complete-H8/full-legal-root-Q/player0-margin',teacher='H_v1',
            requested_depth=8,completed_depth=8,normalized='(y-D)/signed_active',clipping=False,
            endpoints='retain',solved_truth=False),
        sampler=dict(cells=list(CELLS),cell_mass='1/4',within_cell='uniform-replacement'),
        features=dict(count=127,dtype='IEEE754-binary32',rounding='nearest-even',existing_cache_only=True),
        inherited_unknown_background=True,historically_unseen=False,old_candidate_rehabilitated=False,
        excluded_old_support_roots=320,fit_targets_read=0,arena_targets_read=0,
        feature_simulations=0,model_forwards=0,optimizer_steps=0,policy_fixed=True)

def _derive_rows():
    selected=membership();out=[];cached={}
    with closing(previous.ro('deeper-9000-v2')) as db:
        raw=db.execute("SELECT body,hash FROM meta WHERE key='identity'").fetchone()
        teacher=json.loads(raw[0]);need(config.digest(teacher)==raw[1],'Teacher identity hash differs')
        need(teacher['rules_digest']==config.RULES_DIGEST and teacher['choice_depth']==8 and
             teacher['teacher']=='H_v1/untrained/oracle-free/exact-completed-root-depth8/full-root-Q' and
             _same(teacher['search'],dict(max_depth=8,memory_mb=512,node_budget=0,time_ms=0,tt_mb=16,use_pvs=True)),
             'Original H8 teacher configuration differs')
        for item in selected:
            meta=item['metadata'];reservation=item['reservation']
            y,evidence=historical._teacher(meta,db,teacher)
            ref=meta['feature_ref'];filename=ref['path']
            if filename not in cached:cached[filename]=(ROOT/filename).read_text(encoding='utf-8')
            pointer='$.groups.'+ref['group']+'['+str(ref['index'])+']'
            value=db.execute('SELECT json_extract(?,?),json_extract(?,?),json_extract(?,?),json_extract(?,?)',
                (cached[filename],pointer+'.cached_features',cached[filename],pointer+'.state',
                 cached[filename],pointer+'.state_key',cached[filename],pointer+'.family')).fetchone()
            need(value[0] is not None and _same(json.loads(value[1]),meta['state']) and
                 value[2]==meta['state_key'] and value[3]==meta['family'],'Existing feature source differs')
            full=json.loads(value[0]);features={k:full[k] for k in ('x','d','r')}
            previous.validate_features(features,meta['state']);D,r=features['d'],features['r'];z=(y-D)/r
            need(r and math.isfinite(z) and -1<=z<=1,'Normalized H8 target outside residual range')
            out.append(dict(x=[fp32(v) for v in features['x']],target_z=z,D=D,r=r,target_player0=y,
                state=meta['state'],state_key=meta['state_key'],actor=meta['actor'],family=meta['family'],
                component=reservation['joint_component'],cell=meta['cell'],reserved_role='train',
                weight=reservation['weight'],metadata_sha256=config.digest(meta),reservation_sha256=config.digest(reservation),
                feature_source=dict(reference=ref,full_cached_features_sha256=config.digest(full),
                    extracted_features_sha256=config.digest(features),file_sha256=sha(ROOT/filename)),target_source=evidence))
    return out

def _rows():
    """Source-derived rows cached only as immutable canonical bytes."""
    pins=input_pins();key=config.digest(pins)
    if key not in _ROW_CACHE:
        rows=_derive_rows();need(_same(input_pins(),pins),'Existing target/feature sources changed during extraction')
        _ROW_CACHE.clear();_ROW_CACHE[key]=config.canonical(rows).encode()
    return json.loads(_ROW_CACHE[key])

def _validate_source_rows(role,expected,actual=None):
    expected_sha=[config.digest(row) for row in expected]
    need(_same(role['row_sha256'],expected_sha) and
         _same(role['components'],[row['component'] for row in expected]),
         'Train row target/feature evidence differs from original pinned sources')
    if actual is not None:
        need(type(actual) is list and len(actual)==len(expected) and
             _same([config.digest(row) for row in actual],expected_sha),
             'Actual train rows differ from original pinned sources')

def _pause(path,point,requested,marker):
    if point!=requested:return
    need(path.is_relative_to(stage.FOLDER/'fixtures') and marker is not None,'Data pause requires fixture path')
    atomic(marker,dict(point=point,pid=os.getpid(),output=str(path)))
    while True:time.sleep(.05)

def prepare(output=DEFAULT,pause_at=None,pause_marker=None):
    path=_path(output)
    if (path/'catalog.json').exists() or (path/'complete.json').exists():return admit(path)
    stage.admit_plan();need(pause_at in (None,*HOOKS),'Unknown cache pause hook')
    expected=identity();path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        if (path/'catalog.json').exists():return admit(path)
        for name,body in [('identity.json',expected),('train.json',dict(schema=SCHEMA+'-role',role='train',rows=_rows()))]:
            if (path/name).exists():need(_same(read(path/name),body),'Uncommitted data artifact differs')
            else:atomic(path/name,body)
        rows=read(path/'train.json')['rows'];need(len(rows)==192,'Ordinary support cache count differs')
        manifest=dict(schema=SCHEMA+'-manifest',identity_sha256=sha(path/'identity.json'),source_identity=expected,
            role_files=dict(train=dict(filename='train.json',sha256=sha(path/'train.json'),count=192,
                row_sha256=[config.digest(r) for r in rows],components=[r['component'] for r in rows])),
            labels_read=192,features_packed=192,fit_targets_read=0,arena_targets_read=0,feature_simulations=0,
            model_forwards=0,optimizer_steps=0,policy_fixed=True)
        atomic(path/'manifest.json',manifest)
        atomic(path/'certificate.json',dict(schema=SCHEMA+'-certificate',passed=True,
            files=common.pins(path/n for n in ('identity.json','train.json','manifest.json'))))
        _pause(path,'before_data_catalog',pause_at,pause_marker)
        atomic(path/'catalog.json',dict(schema=SCHEMA+'-catalog',complete=True,
            manifest_sha256=sha(path/'manifest.json'),certificate_sha256=sha(path/'certificate.json')))
        _pause(path,'after_data_catalog',pause_at,pause_marker)
        atomic(path/'complete.json',dict(catalog_sha256=sha(path/'catalog.json')))
    return admit(path)

def admit(output=DEFAULT):
    """Pure source-derived cache verification; no fit rows, forward or encoder.

    Existing H8 labels/features may be re-read for verification. That is source
    inspection, never a fresh teacher target or a new scientific query.
    """
    path=_path(output);stage.admit_plan();catalog=read(path/'catalog.json');frozen=read(path/'identity.json')
    need(_same(frozen,identity()),'Ordinary data source identity differs')
    cert=read(path/'certificate.json');manifest=read(path/'manifest.json')
    need(catalog['schema']==SCHEMA+'-catalog' and catalog['complete'] is True and
         catalog['certificate_sha256']==sha(path/'certificate.json') and catalog['manifest_sha256']==sha(path/'manifest.json') and
         cert['schema']==SCHEMA+'-certificate' and cert['passed'] is True,'Data completion evidence differs')
    common.current(cert['files']);need(set(cert['files'])==set(common.pins(path/n for n in ('identity.json','train.json','manifest.json'))),
         'Data file closure differs')
    need(manifest['schema']==SCHEMA+'-manifest' and manifest['identity_sha256']==sha(path/'identity.json') and
         _same(manifest['source_identity'],frozen) and set(manifest['role_files'])=={'train'},'Cache manifest differs')
    r=manifest['role_files']['train']
    need(r['filename']=='train.json' and type(r['count']) is int and r['count']==len(r['row_sha256'])==len(r['components'])==192 and
         r['sha256']==sha(path/'train.json'),'Train row closure differs')
    expected=_rows()
    actual=read(path/'train.json')
    need(set(actual)=={'schema','role','rows'} and actual['schema']==SCHEMA+'-role' and actual['role']=='train',
         'Actual train role wrapper differs')
    _validate_source_rows(r,expected,actual['rows'])
    need(type(manifest['labels_read']) is int and type(manifest['features_packed']) is int and
         manifest['labels_read']==manifest['features_packed']==192 and
         all(type(manifest[k]) is int and manifest[k]==0 for k in
             ('fit_targets_read','arena_targets_read','feature_simulations','model_forwards','optimizer_steps')) and
         manifest['policy_fixed'] is True,'Unexpected cache activity')
    if (path/'complete.json').exists():need(_same(read(path/'complete.json'),dict(catalog_sha256=sha(path/'catalog.json'))),'Cache marker differs')
    return manifest

def load_train(output=DEFAULT):
    path=_path(output);manifest=admit(path);body=read(path/'train.json');rows=body['rows']
    need(body['schema']==SCHEMA+'-role' and body['role']=='train' and
         [config.digest(r) for r in rows]==manifest['role_files']['train']['row_sha256'],'Train row digests differ')
    selected=membership()
    for row,item in zip(rows,selected):
        meta,res=item['metadata'],item['reservation'];active=sum(map(sum,row['state']['pits']))
        need(row['reserved_role']=='train' and row['state_key']==meta['state_key']==common.rawkey(row['state']) and
             row['family']==meta['family'] and row['component']==res['joint_component'] and row['cell']==meta['cell'] and
             _same(row['weight'],res['weight']) and row['metadata_sha256']==config.digest(meta) and
             row['reservation_sha256']==config.digest(res) and row['actor']==row['state']['controller'] and
             row['D']==row['state']['stores'][0]-row['state']['stores'][1] and
             row['r']==active*(1 if row['actor']==0 else -1) and 25<=active<=74 and len(row['x'])==127 and
             all(type(v) is float and math.isfinite(v) and fp32(v)==v for v in row['x']) and
             type(row['target_player0']) is int and -1<=row['target_z']<=1 and
             row['target_z']==(row['target_player0']-row['D'])/row['r'],'Malformed ordinary training row')
    return rows,manifest

def fixture_rows(rows):
    """Exactly first8 frozen roots per cell, retaining the four-cell sampler."""
    chosen=[]
    for cell in CELLS:
        cell_rows=[r for r in rows if r['cell']==cell];need(len(cell_rows)>=8,'Fixture lacks required ordinary cell')
        chosen.extend(cell_rows[:8])
    return chosen
