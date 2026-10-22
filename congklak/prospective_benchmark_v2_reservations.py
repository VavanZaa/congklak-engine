"""Protected-data exclusions and compact universally certified benchmark roots."""
import base64
import hashlib
import os
from pathlib import Path
import struct
import uuid
from . import config, neurodiag_common as common
from . import prospective_benchmark_v2_cohort as cohort
from . import ordinary_epoch_guard as packed
from . import guarded_value_pilot_v2_core as old_codec
from .value_diagnostics import atomic

ROOT=common.ROOT
FOLDER=ROOT/'build/prospective-benchmark-v2'
DEFAULT=FOLDER/'reservations'
COHORT=FOLDER/'cohort'
DECLARATION=COHORT/'declaration-v3.json'
PRIOR=ROOT/'build/prospective-eval-v1'
SHARED='shared-frozen-exact-guard'
OWNER='prospective-benchmark-v2/joint-evaluation'
SCHEMA='prospective-benchmark-v2-reservations'
need,read,sha=common.require,common.read,common.sha
SOURCES=('congklak/prospective_benchmark_v2_cohort.py','congklak/prospective_benchmark_v2_reservations.py',
    'congklak/config.py','congklak/value_run_arena.py','tests/reference.py','tests/test_prospective_eval_execute.py',
    'native/guarded_value_pilot_v2/rules.cpp','native/guarded_value_pilot_v2/neural_relay.cpp')
_FAKE_ROOT='build/prospective-eval-v1/fixtures/pure-tests/'
SYNTHETIC_FIXTURES={**{_FAKE_ROOT+name+'/archives/query-000000.json':'7fe922c52116db91e8a40d339a9cbfa08275b08d5baa01af4f79118888b6b22e'
    for name in ('0a0994c73f9e49dc82cb7c3cfb4e0e15','2a4072c906a245bf9de56a607395f0ae',
                 '3882b551c4344ba2998f5b687991201d','3cce603e165448c8b298c07a0a2d3310')},
    **{_FAKE_ROOT+name+'/attempts/extra.bin':'8d48d8ac066725dc647aaf3a483798a1ae5efa1872889b79f180f4386f2eb1e2'
    for name in ('068c539d3130448aa2e90bf759f32222','14a087fb5676417c8e825fb157019954',
                 '4ca6d52ecafc479791354df2860506a6','5a43c8d59452410f9cfc94937cfacbf4','b42ef8e86f1a4be8a7cd52c316fd99e2')}}


def relative(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def ref(path):return dict(path=relative(path),sha256=sha(path))
def publish_bytes(path,data):
    path=Path(path)
    need(path.is_relative_to(FOLDER),'Benchmark artifact outside new namespace')
    if path.exists():need(path.read_bytes()==data,'Immutable benchmark bytes differ');return
    path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with pending.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def prior_files():
    return sorted(set([*[ROOT/name for name in SOURCES],
        PRIOR/'reservations/components.bin',PRIOR/'reservations/report.json',PRIOR/'reservations/registry.json',
        PRIOR/'reservations/certificate.json',PRIOR/'reservations/membership.json',PRIOR/'plan.json',PRIOR/'report.json',
        *PRIOR.glob('**/claims.bin'),*PRIOR.glob('**/archives/*.json'),
        *PRIOR.glob('**/failed-attempts/*.json'),*PRIOR.glob('**/attempts/*.bin')]))


def declare():
    """Freeze sources, population and all known predecessor query evidence."""
    body=dict(schema=SCHEMA+'-declaration',specification=cohort.specification(),source_inputs=common.pins(prior_files()),
        candidate_constructor_calls=0,native_queries=0,model_forwards=0,optimizer_steps=0,
        default_deny_unknown_high=True,old_known_components_all_forbidden=True,old_claims_not_cleared=True,
        independent_metadata_review_required=True,old_evaluations_remain_held=True)
    atomic(DECLARATION,body);return body


def prepare_prior():
    declaration=read(DECLARATION);common.current(declaration['source_inputs'])
    keys=set();evidence=[];covered=set()
    paths=sorted([*PRIOR.glob('**/archives/*.json'),*PRIOR.glob('**/failed-attempts/*.json')])
    for path in paths:
        if relative(path) in SYNTHETIC_FIXTURES:
            need(sha(path)==SYNTHETIC_FIXTURES[relative(path)],'Exact retained synthetic fixture changed')
            evidence.append(dict(synthetic_fixture_ref=ref(path),producer_ref=ref(ROOT/'tests/test_prospective_eval_execute.py'),
                classification='sealed pure test fabricated empty footprint; no native query'));continue
        body=read(path);answer=body['answer'];owned=base64.b64decode(answer['fast_owned_high'],validate=True)
        cert=answer['fast_certificate']
        need(hashlib.sha256(owned).hexdigest()==cert['owned_sha256'] and len(owned)==cert['owned_count']*40,
             'Retained prior compact footprint changed')
        rows=packed.owned_rows(owned)
        keys.update(bytes.fromhex(row['quotient']) for row in rows)
        stem=path.stem;covered.add((path.parent.parent/'attempts'/(stem+'.bin')).resolve())
        evidence.append(dict(source_ref=ref(path),owned_keys=len(rows),failed_attempt=path.parent.name=='failed-attempts'))
    for path in sorted(PRIOR.glob('**/claims.bin')):
        raw=path.read_bytes();need(len(raw)%19==0,'Prior global claim width differs')
        keys.update(q for q,owner in packed.STATIC.iter_unpack(raw));evidence.append(dict(claims_ref=ref(path),keys=len(raw)//19))
    for path in sorted(PRIOR.glob('**/attempts/*.bin')):
        if path.resolve() in covered:continue
        if relative(path) in SYNTHETIC_FIXTURES:
            need(sha(path)==SYNTHETIC_FIXTURES[relative(path)],'Exact retained synthetic sink fixture changed')
            evidence.append(dict(synthetic_fixture_ref=ref(path),producer_ref=ref(ROOT/'tests/test_prospective_eval_execute.py'),
                classification='sealed pure test fabricated non-trace sentinel; no native query'));continue
        prefix=old_codec.read_attempt_sink(path);raw=prefix['raw_trace']
        for cells,actor,*rest in packed.EVENT.iter_unpack(raw):
            q=packed.raw_key(cells,actor)
            if sum(q[:14])>=25:keys.add(q)
        evidence.append(dict(unreturned_mapped_prefix_ref=ref(path),raw_events=prefix['event_count'],unknown_tail_preserved=True))
    publish_bytes(COHORT/'prior-extra-keys.bin',b''.join(sorted(keys)))
    body=dict(schema=SCHEMA+'-prior-evidence',declaration_ref=ref(DECLARATION),
        base_complete_mapping=ref(PRIOR/'reservations/components.bin'),all_retained_extra_keys=ref(COHORT/'prior-extra-keys.bin'),
        retained_extra_key_count=len(keys),sources=evidence,all_known_prior_components_forbidden=True,
        retained_failed_attempts=sum(row.get('failed_attempt',False) for row in evidence),old_claims_not_cleared=True)
    atomic(COHORT/'prior-evidence.json',body);return body


class Exclusions:
    def __init__(self):
        from .prospective_eval_guard import Lookup
        report=read(PRIOR/'reservations/report.json')
        self.lookup=Lookup((PRIOR/'reservations/components.bin').read_bytes(),report['component_ordinals'])
        raw=(COHORT/'prior-extra-keys.bin').read_bytes();self.extra=frozenset(raw[i:i+15] for i in range(0,len(raw),15))
    def __call__(self,q):
        cid=self.lookup.get(q)
        if cid is not None:return dict(kind='any_prior_complete_component',component=cid)
        if bytes.fromhex(q) in self.extra:return dict(kind='retained_prior_query_or_global_claim')
        return None


def build_boundary(review=FOLDER/'audit/cohort-declaration.json'):
    declaration=read(DECLARATION);common.current(declaration['source_inputs']);review=read(review)
    need(review['passed'] is True and review['declaration_ref']==ref(DECLARATION),'Independent pre-draw metadata review missing/stale')
    prepare_prior();forbidden=Exclusions();reference=cohort.literal._literal();accepted=[];seen=set();counts={}
    for counter in range(cohort.MAX_DRAWS):
        candidate=cohort.candidate(counter);reasons=[];certificate=None;hits=[]
        if candidate['constructor_rejected']:reasons.append('minimum legal store-crossing allocation exceeds25')
        else:
            certificate=cohort.certify(candidate['state'],reference)
            if not certificate['ordinary']:reasons.append('actual-actor mechanism is capture or bonus')
            for state in certificate['root_actor_states']:
                cause=forbidden(state['quotient'])
                if cause:hits.append(dict(quotient=state['quotient'],cause=cause))
            if hits:reasons.append('prior known component/query/global claim')
            if candidate['state_key'] in seen:reasons.append('duplicate raw root')
            seen.add(candidate['state_key'])
        row=dict(schema=SCHEMA+'-boundary-counter',candidate=candidate,certificate=certificate,
            eligible=not reasons,reasons=reasons,protected_hits=hits,native_queries=0,model_forwards=0)
        atomic(COHORT/'counters'/f'{counter:04d}.json',row)
        for reason in reasons:counts[reason]=counts.get(reason,0)+1
        if not reasons:accepted.append(row)
    selected=accepted[:cohort.QUOTA]
    body=dict(schema=SCHEMA+'-boundary',declaration_ref=ref(DECLARATION),prior_evidence_ref=ref(COHORT/'prior-evidence.json'),
        all_counters=cohort.MAX_DRAWS,eligible_roots=len(accepted),selected=selected,requested_roots=cohort.QUOTA,
        rejection_counts=counts,complete=len(selected)==cohort.QUOTA,shortage=max(0,cohort.QUOTA-len(selected)),
        fixed_scan_completed=True,selection_uses_native_outputs=False,historically_unseen=False,root_only_value_branch=True)
    atomic(COHORT/'boundary.json',body);return body


def build_optional():
    """Separate bounded high-child stratum; never changes boundary selection."""
    declaration=read(DECLARATION);common.current(declaration['source_inputs'])
    review=read(FOLDER/'audit/cohort-declaration.json')
    need(review['passed'] is True and review['declaration_ref']==ref(DECLARATION),'Independent optional population review missing/stale')
    forbidden=Exclusions();reference=cohort.literal._literal();accepted=[];rows=[]
    for counter in range(cohort.OPTIONAL_DRAWS):
        candidate=cohort.candidate(counter,'optional');closure=None;reasons=[]
        if candidate['constructor_rejected']:reasons.append('minimum allocation exceeds target active seeds')
        else:
            actual=cohort.literal.normalize(candidate['state'],reference)
            transitions=[cohort.literal.transition(candidate['state'],a,reference) for a in actual['legal']]
            if any(t['captured'] or t['bonus'] for t in transitions):reasons.append('actual-actor mechanism is capture or bonus')
            closure=cohort.certify_high_closure(candidate['state'],forbidden,reference)
            if not closure['complete']:reasons.append(closure['reason'])
        row=dict(schema=SCHEMA+'-optional-counter',candidate=candidate,closure=closure,eligible=not reasons,reasons=reasons)
        atomic(COHORT/'optional/counters'/f'{counter:02d}.json',row);rows.append(row)
        if not reasons:accepted.append(row)
    body=dict(schema=SCHEMA+'-optional',declaration_ref=ref(DECLARATION),all_counters=cohort.OPTIONAL_DRAWS,
        selected=accepted[:cohort.OPTIONAL_QUOTA],eligible_roots=len(accepted),requested_roots=cohort.OPTIONAL_QUOTA,
        complete=len(accepted)>=cohort.OPTIONAL_QUOTA,shortage=max(0,cohort.OPTIONAL_QUOTA-len(accepted)),
        independent_of_boundary=True,substitution_with_boundary_roots=False)
    atomic(COHORT/'optional/report.json',body);return body


def publish():
    boundary=read(COHORT/'boundary.json');need(boundary['complete'] is True,'Boundary metadata cohort has a shortage')
    selected=boundary['selected'];high=set();arena=[]
    cid=config.digest(dict(schema=SCHEMA+'-joint-owner',declaration_sha256=sha(DECLARATION)))
    for index,row in enumerate(selected):
        candidate=row['candidate'];state=candidate['state'];high.update(s['quotient'] for s in row['certificate']['root_actor_states'])
        arena.append(dict(id=f'boundary/{index:02d}',state=state,state_key=candidate['state_key'],family=candidate['origin'],
            source_origin=candidate['origin'],origin=OWNER,component=cid,current_component=cid,joint_component=cid,
            role='arena',cell='25-49/ordinary/actor'+str(state['controller']),stratum='25-49/ordinary',
            structural_certificate_ref=ref(COHORT/'counters'/f'{candidate["counter"]:04d}.json')))
    forbidden=Exclusions();need(all(forbidden(q) is None for q in high),'Final allowlist touches old ownership')
    publish_bytes(DEFAULT/'components.bin',b''.join(packed.STATIC.pack(bytes.fromhex(q),1) for q in sorted(high)))
    publish_bytes(DEFAULT/'claims.bin',b'')
    exact=read(PRIOR/'reservations/membership.json')['exact']
    members=dict(arena=arena,fit=[],train=[],cases=[],exact=exact)
    atomic(DEFAULT/'membership.json',members)
    components={cid:dict(origins=[OWNER],roles=['arena'],evaluation_blocked=False),SHARED:dict(origins=[SHARED],roles=['shared_guard'])}
    registry=dict(schema=SCHEMA+'-registry',origins=sorted([OWNER,SHARED]),origin_components={OWNER:cid,SHARED:SHARED},
        component_ordinals={cid:1,SHARED:0},shared_records=ref(DEFAULT/'components.bin'),seed_claims=ref(DEFAULT/'claims.bin'))
    report=dict(schema=SCHEMA+'-report',components=components,component_ordinals=registry['component_ordinals'],
        record_count=len(high),shared_records_ref=registry['shared_records'],seed_claims_ref=registry['seed_claims'],
        source_inputs=read(DECLARATION)['source_inputs'],cohort_ref=ref(COHORT/'boundary.json'),
        prior_evidence_ref=ref(COHORT/'prior-evidence.json'),declaration_ref=ref(DECLARATION),
        reservation_admission=True,all_prior_known_and_claimed_states_default_denied=True,
        old_claims_preserved_as_forbidden=True,one_joint_evaluation_unit=True,historically_unseen=False,
        inherited_unknown_background=True,root_only_value_branch=True,optional_stratum_included=False)
    atomic(DEFAULT/'report.json',report);registry['graph']=ref(DEFAULT/'report.json');atomic(DEFAULT/'registry.json',registry)
    files=[DEFAULT/n for n in ('components.bin','claims.bin','membership.json','report.json','registry.json')]
    certificate=dict(schema=SCHEMA+'-certificate',passed=True,files=common.pins(files))
    atomic(DEFAULT/'certificate.json',certificate);atomic(DEFAULT/'catalog.json',dict(schema=SCHEMA+'-catalog',certificate_ref=ref(DEFAULT/'certificate.json'),passed=True))
    return report


def verify(full=True):
    certificate=read(DEFAULT/'certificate.json');need(certificate['passed'] is True,'Benchmark reservation certificate failed')
    common.current(certificate['files']);report=read(DEFAULT/'report.json')
    if full:common.current(report['source_inputs'])
    return report
def membership():verify(False);return read(DEFAULT/'membership.json')
def registry():verify(False);return read(DEFAULT/'registry.json')
