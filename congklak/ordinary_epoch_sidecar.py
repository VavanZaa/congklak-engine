"""New-epoch binding for unchanged sidecar arithmetic and byte-fixed policy.

Only manifest/provenance admission differs from the historical512-root loader.
The native ABI, branch shapes, encoder, active gate and all base tensors match
the frozen implementation. No model query occurs during publication/reading.
"""
import ctypes as C
from pathlib import Path
from . import config,ordinary_epoch_common as stage,ordinary_epoch_data as data
from . import value_sidecar_v2 as frozen

ROOT=stage.ROOT
BASE=frozen.BASE
BASE_SHA=frozen.BASE_SHA
INITIAL=stage.DATA/'initial-v2'
SCHEMA='congklak-ordinary-epoch-sidecar-v1'
PROTOCOL='ordinary-epoch/frozen-policy/value-sidecar-v1/fixed-final-step400'
SOURCES=('congklak/ordinary_epoch_sidecar.py','congklak/ordinary_epoch_common.py',
 'congklak/ordinary_epoch_data.py',*frozen.DEPENDENCIES)
SHAPES=frozen.SHAPES
ABI=frozen.ABI
GATE=frozen.GATE
SidecarResult=frozen.SidecarResult
need,read,sha=stage.need,stage.read,stage.sha
tensor_sha=frozen.tensor_sha

def _same(a,b):return config.canonical(a)==config.canonical(b)

def policy_proof():
    base=read(BASE);need(sha(BASE)==BASE_SHA and config.digest(base['model'])==base['sha256'],'Frozen policy/control changed')
    weights=base['model']['weights']
    return dict(base_file_sha256=BASE_SHA,base_export_sha256=config.digest(base),
        all_eight_base_tensors_sha256=config.digest(weights),
        policy_tensors_sha256=config.digest({k:weights[k] for k in ('policy.weight','policy.bias')}),
        shared_trunk_sha256=config.digest({k:weights[k] for k in
            ('layers.0.weight','layers.0.bias','layers.2.weight','layers.2.bias')}),
        frozen_parameters=12872,policy_arithmetic_unchanged=True,policy_training_steps=0)

def bindings(manifest):
    path=Path(manifest).resolve();need(path.name=='manifest.json','Epoch input must be cache manifest')
    admitted=data.admit(path.parent);record=frozen.build_record();stage.admit_plan()
    return dict(schema=SCHEMA+'-bindings',epoch_id=stage.epoch()['epoch_id'],
        previous_acceptance_sha256=stage.PRIOR_SHA,plan_sha256=sha(stage.PLAN),
        input_manifest=dict(path=str(path),sha256=sha(path)),input_identity_sha256=admitted['identity_sha256'],
        base_file_sha256=BASE_SHA,policy_proof=policy_proof(),gate=GATE,
        encoder=frozen.features_relay.ENCODER127,rules_digest=config.RULES_DIGEST,abi=ABI,protocol=PROTOCOL,
        numeric_backend='strict-double/scalar-or-AVX2/separate-multiply-add/FP32-export-weights',
        build_record_sha256=sha(frozen.BUILD),dll_sha256=record['dll_sha256'],build_inputs=record['inputs'],
        sources=stage.common.pins(SOURCES),ancestry='inherited unknown historical background',
        old512_recipe_reused=False,old_candidate_rehabilitated=False,production_evaluation_admitted=False)

def make_export(weights,manifest,provenance):
    w=frozen._weights(weights);b=bindings(manifest)
    need(provenance.get('protocol')==PROTOCOL and provenance.get('fixed_policy') is True,
         'Epoch export provenance protocol differs')
    model=dict(schema=SCHEMA,bindings=b,base_export=read(BASE),branch_weights=w,
        branch_weights_sha256=tensor_sha(w),
        parameter_counts=dict(base_frozen=12872,branch_trainable=2065,total=14937),provenance=provenance)
    return dict(model=model,sha256=config.digest(model))

def verify_export(path):
    """Standard-library envelope/source/policy check, with zero model queries."""
    path=Path(path).resolve();need(not(path.parent/'quarantine.json').exists(),'Epoch model is quarantined')
    e=read(path);need(set(e)=={'model','sha256'} and config.digest(e['model'])==e['sha256'],'Damaged epoch model export')
    m=e['model'];need(set(m)=={'schema','bindings','base_export','branch_weights','branch_weights_sha256','parameter_counts','provenance'} and
        m['schema']==SCHEMA and _same(m['bindings'],bindings(m['bindings']['input_manifest']['path'])),
        'Epoch model manifest/build/plan bindings differ')
    need(_same(m['base_export'],read(BASE)) and
        _same(m['parameter_counts'],dict(base_frozen=12872,branch_trainable=2065,total=14937)),
        'An original base or policy tensor changed')
    weights=frozen._weights(m['branch_weights']);need(tensor_sha(weights)==m['branch_weights_sha256'],'Branch tensor SHA differs')
    p=m['provenance'];need(p['protocol']==PROTOCOL and p['fixed_policy'] is True and
        p['kind'] in ('stepzero','fixed-final-step') and p['inherited_unknown_background'] is True and
        p['production_evaluation_admitted'] is False,'Epoch provenance scope differs')
    if p['kind']=='stepzero':
        need(type(p['optimizer_steps']) is int and p['optimizer_steps']==0 and
             tensor_sha(weights)==frozen.INITIAL_TENSOR_SHA and weights['v']==[[0.]*16] and weights['c']==[0.],
             'Epoch initialization differs from fixed zero-output tensors')
    else:
        need(type(p['engineering_fixture']) is bool and type(p['optimizer_steps']) is int and
             p['optimizer_steps']==(6 if p['engineering_fixture'] else 400) and
             p['selection']=='final-fixed-step-only','Unexpected trained candidate/checkpoint selection')
    return e

class Model(frozen.Model):
    """Same reference method and native model API, new source-bound loader."""
    def __init__(self,path):
        self.path=Path(path).resolve();e=verify_export(self.path);m=e['model']
        self.base=frozen.BaseModel(BASE);self.weights=frozen._weights(m['branch_weights'])
        self.envelope=e;self.identity=m['bindings'];self.id=e['sha256'];self.file_sha256=sha(self.path)
        self.width=64;self.input_count=127;self.encoder=self.base.encoder
        values=[]
        for name in ('A','b','v','c'):values.extend(frozen.flatten_weights(self.weights[name]))
        self.array=(C.c_float*2065)(*values)

def initialize(manifest=data.DEFAULT/'manifest.json',output=INITIAL):
    """Reuse pinned stepzero tensors; no Torch RNG, optimizer or forward."""
    stage.admit_plan();path=stage.path_of(output)
    if (path/'certificate.json').exists():return verify_initial(path,manifest)
    original=read(data.INITIAL);weights=original['model']['branch_weights']
    need(tensor_sha(weights)==frozen.INITIAL_TENSOR_SHA,'Frozen initialization tensors changed')
    p=dict(protocol=PROTOCOL,kind='stepzero',optimizer_steps=0,fixed_policy=True,
        initialization_source=dict(path=stage.relative(data.INITIAL),sha256=sha(data.INITIAL)),
        initialization='reuse exact accepted zero-output tensors; no RNG execution',
        inherited_unknown_background=True,production_evaluation_admitted=False)
    export=make_export(weights,manifest,p);path.mkdir(parents=True,exist_ok=True)
    with data.Writer(path):
        if (path/'certificate.json').exists():return verify_initial(path,manifest)
        if (path/'model.json').exists():need(_same(read(path/'model.json'),export),'Uncommitted stepzero export differs')
        else:stage.atomic(path/'model.json',export)
        stage.atomic(path/'certificate.json',dict(schema=SCHEMA+'-initial-certificate',model_sha256=export['sha256'],
            model_file_sha256=sha(path/'model.json'),input_manifest_sha256=sha(manifest),optimizer_steps=0))
    return verify_initial(path,manifest)

def verify_initial(output=INITIAL,manifest=None):
    path=stage.path_of(output);e=verify_export(path/'model.json');c=read(path/'certificate.json')
    m=e['model']['bindings']['input_manifest'];need(e['model']['provenance']['kind']=='stepzero' and
        _same(c,dict(schema=SCHEMA+'-initial-certificate',model_sha256=e['sha256'],
            model_file_sha256=sha(path/'model.json'),input_manifest_sha256=m['sha256'],optimizer_steps=0)),
        'Initial epoch certificate differs')
    if manifest is not None:need(_same(m,dict(path=str(Path(manifest).resolve()),sha256=sha(manifest))),'Initial cache binding differs')
    return dict(schema=SCHEMA+'-initial-status',state='complete',model_file_sha256=sha(path/'model.json'),
        model_sha256=e['sha256'],optimizer_steps=0,policy_fixed=True)
