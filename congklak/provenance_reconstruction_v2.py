"""Versioned role correction over unchanged certified raw provenance V1.

Historical admitted training is ancestral exposure.  It protects future fit and
arena data, but does not independently prohibit the expressly reused512 roots.
Genuine heldout/evaluation/arena/forensic exclusions remain optimizer blockers.
"""
from collections import Counter
import copy
from pathlib import Path
from . import config, neurodiag_common as common
from . import provenance_reconstruction as old, provenance_job as journal
from .valuepilot_reservations import Writer

ROOT=common.ROOT
DEFAULT=old.DEFAULT/'certified-v2'
SCHEMA='source-witnessed-provenance-role-correction-v2'
SOURCES=('congklak/provenance_reconstruction_v2.py','congklak/provenance_reconstruction.py',
         'congklak/provenance_job.py','congklak/value_pilot_data.py','congklak/ordinary_value_pilot.py')
need,read,sha=common.require,common.read,common.sha


def training_group(marker):
    w=marker['witness'];path=w.get('path');pointer=w.get('pointer','')
    if marker['reason']!='previous value diagnostic cache':return None
    if path=='build/value-pilot-rows.json' and pointer.startswith('/groups/train/'):
        return 'prior admitted value pilot train group'
    if path=='build/ordinary-value-rows.json' and any(pointer.startswith('/groups/'+g+'/') for g in ('high','low')):
        return 'prior admitted ordinary value high/low training group'
    return None


def corrected(original):
    need(original['schema']==old.SCHEMA+'-report' and original['models_and_policy_fixed'] is True and
         original['production_admission'] is False,'V1 provenance scope differs')
    report=copy.deepcopy(original);changes=[]
    for index,marker in enumerate(report['marker_ledger']):
        group=training_group(marker)
        if group is None:continue
        before=copy.deepcopy(marker)
        marker.update(optimizer_blocked=False,ancestral_training=True,
            reason='historical admitted training exposure; exact512 reuse permitted, no pool expansion')
        changes.append(dict(marker_index=index,before=before,after=copy.deepcopy(marker),classification_basis=group,
            exemption_scope='Only this training-cache marker; every other optimizer/fit/arena marker remains enforced.'))
    for component in report['components'].values():
        entries=[report['marker_ledger'][i] for i in component['marker_indices']]
        for key in ('optimizer_blocked','fit_blocked','arena_blocked','ancestral_training','acquisition_exposed'):
            component[key]=any(m[key] for m in entries)
    blocked=[]
    for row in report['candidate_pools']['support_reuse']:
        component=report['components'][row['new_domain_component']]
        reasons=[]
        if component['optimizer_blocked']:reasons.append('whole component has genuine optimizer exclusion')
        if component['opaque_legacy_unions']:reasons.append('opaque legacy union remains conservatively unsplittable')
        row.update(eligible=not reasons,exclusion_reasons=reasons,ancestral_training=component['ancestral_training'])
        if reasons:
            entries=[report['marker_ledger'][i] for i in component['marker_indices'] if report['marker_ledger'][i]['optimizer_blocked']]
            blocked.append(dict(state_key=row['state_key'],family=row['family'],component=row['new_domain_component'],
                                reasons=dict(sorted(Counter(m['reason'] for m in entries).items()))))
    for role,rows in report['candidate_pools'].items():
        prior=report['availability'][role]
        prior['eligible_roots']=sum(r['eligible'] for r in rows)
        prior['eligible_components']=len({r['new_domain_component'] for r in rows if r['eligible']})
        for cell in old.metadata.CELLS:
            selected=[r for r in rows if r['cell']==cell and r['eligible']]
            prior['cells'][cell]=dict(roots=len(selected),components=len({r['new_domain_component'] for r in selected}))
    report.update(schema=SCHEMA+'-report',base_graph_schema=original['schema'],
        raw_incidence_and_component_partition_changed=False,role_corrections=changes,
        genuine_support_conflicts=blocked,support_training_admission=not blocked,
        support_roots_with_genuine_exclusion=len(blocked),support_roots_exactly_reused=512,
        larger_training_pool_authorized=False,
        prior_known_fit_arena_forensic_protections_preserved=True,
        historical_training_only_optimizer_ban_removed=True,
        training_authorized=False,production_admission=False)
    return report


def _inputs():
    base=journal.DEFAULT
    return common.pins([base/'identity.json',base/'catalog.json',base/'report.json',*SOURCES,*old.MODELS])


def completed(path=DEFAULT):
    path=Path(path).resolve();certificate=read(path/'certificate.json')
    need(certificate['schema']==SCHEMA+'-certificate' and certificate['passed'] is True,
         'Corrected provenance certificate scope differs')
    common.current(certificate['inputs'])
    need(sha(path/'report.json')==certificate['report_sha256'],'Corrected provenance report changed')
    original=journal.completed()['report'];report=read(path/'report.json')
    need(report==corrected(original),'Corrected provenance role semantics differ')
    need(certificate['inputs']==_inputs(),'Corrected provenance source closure differs')
    return report


def run(path=DEFAULT):
    path=Path(path).resolve();need(path.is_relative_to(old.DEFAULT),'Corrected provenance output namespace differs')
    if (path/'certificate.json').exists():return completed(path)
    original=journal.completed()['report'];report=corrected(original);inputs=_inputs()
    path.mkdir(parents=True,exist_ok=True)
    with Writer(path):
        common.publish(path/'report.json',report)
        common.publish(path/'certificate.json',dict(schema=SCHEMA+'-certificate',passed=True,
            inputs=inputs,report_sha256=sha(path/'report.json'),raw_incidence_changed=False,
            optimizer_steps=0,new_games=0,searches=0,model_forwards=0))
    return completed(path)
