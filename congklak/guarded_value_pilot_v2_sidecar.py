"""Explicit screened-reservation binding for the unchanged value sidecar ABI.

Publication and verification are standard-library evidence operations.  The
eight base tensors and policy are byte fixed; only the 2,065 value parameters
may be changed by the separately admitted trainer.
"""
import ctypes as C
from pathlib import Path
from . import config, guarded_value_pilot_v2_common as stage
from . import guarded_value_pilot_v2_data as data, value_sidecar_v2 as frozen
from .ordinary_epoch_sidecar import policy_proof

ROOT = stage.ROOT
BASE, BASE_SHA = frozen.BASE, frozen.BASE_SHA
INITIAL = stage.DATA / 'initial'
SCHEMA = 'congklak-guarded-value-pilot-v2-sidecar'
PROTOCOL = 'guarded-value-pilot-v2/frozen-policy/value-sidecar/fixed-final-step400'
SHAPES, ABI, GATE = frozen.SHAPES, frozen.ABI, frozen.GATE
SidecarResult, tensor_sha = frozen.SidecarResult, frozen.tensor_sha
SOURCES = ('congklak/guarded_value_pilot_v2_sidecar.py',
           'congklak/guarded_value_pilot_v2_common.py',
           'congklak/guarded_value_pilot_v2_data.py',
           'congklak/ordinary_epoch_sidecar.py', *frozen.DEPENDENCIES)
need, read, sha = stage.need, stage.read, stage.sha


def _same(a, b):
    return config.canonical(a) == config.canonical(b)


def bindings(manifest):
    path = Path(manifest).resolve()
    need(path == (data.DEFAULT / 'manifest.json').resolve(), 'Screened cache manifest path differs')
    stage.admit_plan(False)
    admitted = data.admit(path.parent)
    record = frozen.build_record()
    return dict(schema=SCHEMA + '-bindings', plan_sha256=sha(stage.PLAN),
        reservation_acceptance_sha256=sha(stage.PRIOR),
        input_manifest=dict(path=str(path), sha256=sha(path)),
        admitted_cache_sha256=config.digest(admitted),
        base_file_sha256=BASE_SHA, policy_proof=policy_proof(), gate=GATE,
        encoder=frozen.features_relay.ENCODER127, rules_digest=config.RULES_DIGEST,
        abi=ABI, protocol=PROTOCOL,
        numeric_backend='strict-double/scalar-or-AVX2/separate-multiply-add/FP32-export-weights',
        build_record_sha256=sha(frozen.BUILD), dll_sha256=record['dll_sha256'],
        build_inputs=record['inputs'], sources=stage.common.pins(SOURCES),
        inherited_unknown_background=True, historically_unseen=False,
        production_evaluation_admitted=False)


def make_export(weights, manifest, provenance):
    weights = frozen._weights(weights)
    need(provenance.get('protocol') == PROTOCOL and provenance.get('fixed_policy') is True,
         'Guarded value export protocol differs')
    model = dict(schema=SCHEMA, bindings=bindings(manifest), base_export=read(BASE),
        branch_weights=weights, branch_weights_sha256=tensor_sha(weights),
        parameter_counts=dict(base_frozen=12872, branch_trainable=2065, total=14937),
        provenance=provenance)
    return dict(model=model, sha256=config.digest(model))


def verify_export(path):
    """Pure source/envelope/policy verifier; it performs no model queries."""
    path = stage.path_of(path)
    need(not (path.parent / 'quarantine.json').exists(), 'Guarded value model is quarantined')
    export = read(path)
    need(set(export) == {'model', 'sha256'} and config.digest(export['model']) == export['sha256'],
         'Guarded value export envelope differs')
    model = export['model']
    need(set(model) == {'schema', 'bindings', 'base_export', 'branch_weights',
                        'branch_weights_sha256', 'parameter_counts', 'provenance'} and
         model['schema'] == SCHEMA and
         _same(model['bindings'], bindings(model['bindings']['input_manifest']['path'])),
         'Guarded value export source bindings differ')
    need(_same(model['base_export'], read(BASE)) and
         _same(model['parameter_counts'], dict(base_frozen=12872, branch_trainable=2065, total=14937)),
         'Frozen base or policy tensors changed')
    weights = frozen._weights(model['branch_weights'])
    need(tensor_sha(weights) == model['branch_weights_sha256'], 'Guarded value tensor digest differs')
    provenance = model['provenance']
    need(provenance['protocol'] == PROTOCOL and provenance['fixed_policy'] is True and
         provenance['kind'] in ('stepzero', 'fixed-final-step') and
         provenance['inherited_unknown_background'] is True and
         provenance['production_evaluation_admitted'] is False,
         'Guarded value export provenance differs')
    if provenance['kind'] == 'stepzero':
        need(type(provenance['optimizer_steps']) is int and provenance['optimizer_steps'] == 0 and
             tensor_sha(weights) == frozen.INITIAL_TENSOR_SHA and
             weights['v'] == [[0.] * 16] and weights['c'] == [0.],
             'Guarded value zero-output initialization differs')
    else:
        need(type(provenance['engineering_fixture']) is bool and
             type(provenance['optimizer_steps']) is int and
             provenance['optimizer_steps'] == (6 if provenance['engineering_fixture'] else 400) and
             provenance['selection'] == 'final-fixed-step-only', 'Unexpected value training cursor/selection')
    return export


class Model(frozen.Model):
    """Unchanged reference arithmetic and native tensor layout, explicit loader."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        export = verify_export(self.path)
        model = export['model']
        self.base = frozen.BaseModel(BASE)
        self.weights = frozen._weights(model['branch_weights'])
        self.envelope, self.identity = export, model['bindings']
        self.id, self.file_sha256 = export['sha256'], sha(self.path)
        self.width, self.input_count, self.encoder = 64, 127, self.base.encoder
        values = []
        for name in ('A', 'b', 'v', 'c'):
            values.extend(frozen.flatten_weights(self.weights[name]))
        self.array = (C.c_float * 2065)(*values)


def initialize(manifest=None, output=INITIAL):
    """Copy the accepted exact zero-output branch, without RNG or Torch."""
    manifest = data.DEFAULT / 'manifest.json' if manifest is None else Path(manifest)
    stage.admit_plan(False)
    path = stage.path_of(output)
    need(path == INITIAL.resolve(), 'Initial value export path differs')
    if (path / 'certificate.json').exists():
        return verify_initial(path, manifest)
    source = data.INITIAL
    weights = read(source)['model']['branch_weights']
    need(tensor_sha(weights) == frozen.INITIAL_TENSOR_SHA, 'Accepted initial tensor source changed')
    provenance = dict(protocol=PROTOCOL, kind='stepzero', optimizer_steps=0, fixed_policy=True,
        initialization_source=dict(path=stage.relative(source), sha256=sha(source)),
        initialization='reuse exact accepted zero-output tensors; no RNG execution',
        inherited_unknown_background=True, production_evaluation_admitted=False)
    export = make_export(weights, manifest, provenance)
    path.mkdir(parents=True, exist_ok=True)
    from .valuepilot_reservations import Writer
    with Writer(path):
        if (path / 'certificate.json').exists():
            return verify_initial(path, manifest)
        if (path / 'model.json').exists():
            need(_same(read(path / 'model.json'), export), 'Uncommitted initial value export differs')
        else:
            stage.atomic(path / 'model.json', export)
        stage.atomic(path / 'certificate.json', dict(schema=SCHEMA + '-initial-certificate',
            model_sha256=export['sha256'], model_file_sha256=sha(path / 'model.json'),
            input_manifest_sha256=sha(manifest), optimizer_steps=0))
    return verify_initial(path, manifest)


def verify_initial(output=INITIAL, manifest=None):
    path = stage.path_of(output)
    export = verify_export(path / 'model.json')
    ref = export['model']['bindings']['input_manifest']
    need(export['model']['provenance']['kind'] == 'stepzero' and
         _same(read(path / 'certificate.json'), dict(schema=SCHEMA + '-initial-certificate',
            model_sha256=export['sha256'], model_file_sha256=sha(path / 'model.json'),
            input_manifest_sha256=ref['sha256'], optimizer_steps=0)), 'Initial value certificate differs')
    if manifest is not None:
        need(_same(ref, dict(path=str(Path(manifest).resolve()), sha256=sha(manifest))),
             'Initial value cache manifest differs')
    return dict(schema=SCHEMA + '-initial-status', state='complete',
        model_file_sha256=sha(path / 'model.json'), model_sha256=export['sha256'],
        optimizer_steps=0, policy_fixed=True)
