"""Add the predeclared complete high-child stratum without changing base files."""
from pathlib import Path
from . import config, neurodiag_common as common
from . import prospective_benchmark_v2_reservations as base
from .value_diagnostics import atomic

ROOT,FOLDER,COHORT=base.ROOT,base.FOLDER,base.COHORT
DEFAULT=FOLDER/'reservations-extended'
SHARED,OWNER=base.SHARED,base.OWNER
SCHEMA=base.SCHEMA
need,read,sha=common.require,common.read,common.sha


def publish():
    parent=base.verify();optional=read(COHORT/'optional/report.json')
    need(optional['complete'] is True and optional['declaration_ref']==base.ref(base.DECLARATION),
         'Optional complete protected-safe cohort is unavailable')
    members=base.membership();cid=base.registry()['origin_components'][OWNER]
    high={q.hex() for q,ordinal in base.packed.STATIC.iter_unpack((base.DEFAULT/'components.bin').read_bytes())}
    for row in members['arena']:row['benchmark_stratum']='boundary25'
    for index,row in enumerate(optional['selected']):
        candidate=row['candidate'];state=candidate['state'];closure=row['closure']
        need(closure['complete'] is True and row['eligible'] is True,'Selected optional root lacks full closure')
        high.update(s['quotient'] for s in closure['states'])
        members['arena'].append(dict(id=f'optional/{index:02d}',state=state,state_key=candidate['state_key'],
            family=candidate['origin'],source_origin=candidate['origin'],origin=OWNER,
            component=cid,current_component=cid,joint_component=cid,role='arena',
            cell='25-49/ordinary/actor'+str(state['controller']),stratum='25-49/ordinary',
            benchmark_stratum='optional26-32',active_seeds=candidate['active_seeds'],
            structural_certificate_ref=base.ref(COHORT/'optional/counters'/f'{candidate["counter"]:02d}.json')))
    forbidden=base.Exclusions();need(all(forbidden(q) is None for q in high),'Extended allowlist touches old protected ownership')
    base.publish_bytes(DEFAULT/'components.bin',b''.join(base.packed.STATIC.pack(bytes.fromhex(q),1) for q in sorted(high)))
    base.publish_bytes(DEFAULT/'claims.bin',b'');atomic(DEFAULT/'membership.json',members)
    registry=base.registry();registry.update(shared_records=base.ref(DEFAULT/'components.bin'),seed_claims=base.ref(DEFAULT/'claims.bin'))
    report=dict(parent,record_count=len(high),shared_records_ref=registry['shared_records'],seed_claims_ref=registry['seed_claims'],
        optional_stratum_included=True,root_only_value_branch=False,
        stratum_counts={'boundary25':8,'optional26-32':4},optional_cohort_ref=base.ref(COHORT/'optional/report.json'),
        descriptive_strata_reported_separately=True,independent_clusters_claimed=False,
        parent_reservations_ref=base.ref(base.DEFAULT/'certificate.json'),
        source_inputs=dict(parent['source_inputs'],**common.pins([__file__,COHORT/'optional/report.json',
            *[base.DEFAULT/name for name in ('components.bin','claims.bin','membership.json','report.json','registry.json','certificate.json','catalog.json')]])))
    atomic(DEFAULT/'report.json',report);registry['graph']=base.ref(DEFAULT/'report.json');atomic(DEFAULT/'registry.json',registry)
    certificate=dict(schema=SCHEMA+'-certificate',passed=True,
        files=common.pins([DEFAULT/name for name in ('components.bin','claims.bin','membership.json','report.json','registry.json')]))
    atomic(DEFAULT/'certificate.json',certificate)
    atomic(DEFAULT/'catalog.json',dict(schema=SCHEMA+'-catalog',certificate_ref=base.ref(DEFAULT/'certificate.json'),passed=True))
    return report


def verify(full=True):
    certificate=read(DEFAULT/'certificate.json');need(certificate['passed'] is True,'Extended reservation certificate failed')
    common.current(certificate['files']);report=read(DEFAULT/'report.json')
    if full:common.current(report['source_inputs'])
    return report
def membership():verify(False);return read(DEFAULT/'membership.json')
def registry():verify(False);return read(DEFAULT/'registry.json')
