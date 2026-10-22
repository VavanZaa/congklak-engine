"""Seal independently audited engineering evidence; leave training on HOLD."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,ownership_implementation_common as stage
from congklak import ownership_implementation_acceptance as acceptance
from congklak import prospective_reservations_v2 as reservations

def main():
    if acceptance.PATH.exists():acceptance.main();return
    auditpath=stage.FOLDER/'audit/audit.json';audit=stage.read(auditpath)
    stage.need(audit['passed'] and audit['independent_of_implementation'] and not audit['unresolved_findings'],
        'Independent audit must finish before closure')
    selection=reservations.completed()['selection']
    stage.need(selection['metadata_reservations_complete'],'Joint metadata reservations not complete')
    cards=[dict(title='Certified saved-source provenance',state='complete',
        result='354 units; 550,846 saved moves; 661,153 actual origin/state incidences. No invented Cartesian edges.',
        evidence='build/ownership-implementation-v1/provenance/certified-v2/certificate.json',
        improvement='Known saved sources certified; missing earlier engineering footprints remain an admission limitation.'),
      dict(title='Complete native tracing',state='complete',
        result='32 author checks and independent physical coverage/conformance/refusal checks passed.',
        evidence='build/ownership-implementation-v1/native/author-v2/result.json',
        improvement='Reduce trace finalization and ownership-check costs before admitting 25 ms production strength tests.'),
      dict(title='Live handoff and interruption recovery',state='complete',
        result='9 fixed and 2 timed fixture games, 159 decisions; 5 actual worker interruption boundaries passed.',
        evidence='build/ownership-implementation-v1/audit/audit.json',
        improvement='V2 launcher fixes completed-run routing. Unclean leases stay fully charged; no committed search repeats.'),
      dict(title='Joint conditional metadata reservations',state='hold',
        result='128 new fit roots, 32-case subset, 24 arena origins jointly reserved in 152 known-disjoint components; unchanged 512 support retained.',
        evidence='build/ownership-implementation-v1/reservations/production-v2/certificate.json',
        improvement='Training/evaluation admission held: 320 support roots conflict with protection; missing earlier traces and timing debt remain.')]
    events=sorted((stage.FOLDER/'events').glob('*.json'))
    event=dict(schema='ownership-implementation-progress-v1',state='complete',
        message='Implementation and conditional metadata reservations audited. Training and production evaluation remain on HOLD.',
        cards=cards,next_operation='Reduce traced decision costs and resolve the support/provenance admission protocol before labels, training or production evaluation.',
        acceptance_schema=acceptance.SCHEMA,baseline_sha256=stage.sha(stage.FOLDER/'baseline.json'),
        models_sha256=stage.common.pins(['data/quality-train-v1/model.json','data/value-training-v1/training/model.json',
            'data/training-stage5-deeper-h8-f127-seed2718/model.json']),models_fixed=True,policy_fixed=True,
        training_counters=stage.TRAINING_ZERO,execution_ready=False,training_authorized=False)
    eventpath=(events[-1] if events and stage.read(events[-1])==event else stage.FOLDER/'events'/f'{len(events):06d}.json')
    stage.atomic(eventpath,event)
    files=[p for p in stage.FOLDER.rglob('*') if p.is_file() and p!=acceptance.PATH and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files+=list(acceptance.SOURCES)+['tools/finalize-ownership-implementation.py']
    for directory,patterns in [('congklak',('ownership*.py','provenance*.py','prospective*.py','engineering_provenance*.py','native_ownership*.py','quality_ownership*.py')),
        ('tools',('*ownership*.py','*provenance*.py')),('tests',('*ownership*.py','*provenance*.py','*prospective*.py')),
        ('docs',('*OWNERSHIP*',))]:
        for pattern in patterns:files.extend((ROOT/directory).glob(pattern))
    files.extend((ROOT/'native/ownership_v1').rglob('*'))
    files=[p for p in files if not isinstance(p,Path) or p.is_file()]
    inputs=dict(stage.baseline()['inputs']);new=stage.common.pins(files)
    stage.need(all(k not in inputs or inputs[k]==v for k,v in new.items()),'Closure changed accepted history')
    inputs.update(new)
    live={k:v for k,v in new.items() if k.endswith(('.py','.cpp','.h','.hpp','.html','.md','.dll')) or
        Path(k).name in ('catalog.json','certificate.json','report.json','selection.json','identity.json','audit.json')}
    live.update(stage.common.pins([eventpath,stage.FOLDER/'baseline.json']))
    live.update(event['models_sha256'])
    live.update(audit['source_inputs'])
    for method in acceptance.METHODS:
        live.update({e['path']:e['sha256'] for e in audit['methods'][method]['evidence']})
    body=dict(schema=acceptance.SCHEMA,passed=True,inputs=inputs,live_inputs=live,prior_files_preserved=59154,
        independent_audit=auditpath.relative_to(ROOT).as_posix(),progress_event=eventpath.relative_to(ROOT).as_posix(),
        progress_event_sha256=stage.sha(eventpath),execution_ready=False,training_authorized=False,evaluation_admission=False,
        models_fixed=True,policy_fixed=True,training_counters=stage.TRAINING_ZERO,
        candidate_quarantined=True,default_engine='H_v1',conditional_metadata_only=True,
        metadata_counts=dict(unchanged_support=512,fit=128,fit_case_subset=32,arena=24,new_components=152),
        admission_holds=['320 support roots with genuine high-domain protection conflicts',
            'unrecorded earlier engineering footprints; no complete historical absence claim',
            'traced whole-decision costs exceed 25 ms on tested positions'],
        additional_engineering_counters=dict(native_owner_attempts=106,metadata_candidate_draws=selection['counter_draws'],
            metadata_rule_action_probes=selection['native_action_probes']) |
            {'independent_audit_'+k:v for k,v in audit.get('engineering_counters',{}).items()})
    acceptance.validate(body)
    stage.atomic(acceptance.PATH,body)
    print(config.canonical(dict(passed=True,inputs=len(inputs),training_counters=stage.TRAINING_ZERO,
        metadata_fit_roots=128,metadata_arena_origins=24,support_roots_on_hold=320,
        execution_ready=False,training_authorized=False)),flush=True)

if __name__=='__main__':main()
