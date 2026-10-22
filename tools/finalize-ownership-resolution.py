"""Publish the scoped, independently audited metadata resolution closure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config, ownership_resolution_common as s, ownership_resolution_contract as c
from congklak import ownership_resolution_job as job, ownership_resolution_progress as progress


def main():
    baseline=s.read(s.FOLDER/'baseline.json');result=job.completed();report=result['report'];epoch=report['epoch']
    audit=s.read(s.FOLDER/'audit/audit.json')
    s.need(audit['passed'] is True and audit['unresolved_findings']==[] and
        all(audit['methods'][k]['passed'] is True for k in c.METHODS),'Independent resolution acceptance incomplete')
    ready=epoch['prerequisites_ready']
    cards=[dict(title='Support conflicts',state='complete',
        result='320 roots quarantined through their complete shared component; all original512 records and roles preserved.',
        improvement=f"New ordinary-only support: {epoch['ordinary_support_roots']} roots, exact four-cell probabilities. Old balanced recipe remains unavailable.",
        evidence='build/ownership-resolution-v1/production/catalog.json'),
        dict(title='Historical provenance',state='complete',
        result='Known recorded reconstruction and692 actual current query episodes bound to saved evidence.',
        improvement='Broad ancestral unknown, lost captures, partial tails and inconsistent old smoke counts remain explicit. Old results stay excluded.',
        evidence='build/ownership-resolution-v1/production/report.json'),
        dict(title='Future ordinary-development prerequisites',state='complete' if ready else 'hold',
        result=f"Ready: {ready}. Fit128 / case32 / arena24 checked against full refreshed known graph and population constraints.",
        improvement='Conditional on inherited fixed models and recorded-graph separation. A new explicit run recipe and complete future tracing are required.',
        evidence='build/ownership-resolution-v1/production/report.json'),
        dict(title='Independent audit and recovery',state='complete',
        result='Nine audit groups and all four actual metadata interruption boundaries passed.',
        improvement='Immutable prefixes and orphan results reused; no model, policy, optimizer, label or production-game changes.',
        evidence='build/ownership-resolution-v1/audit/audit.json'),
        dict(title='Current engine strength',state='hold',
        result='Functional research analyzer; both-baseline superiority and release strength remain unestablished.',
        improvement='H_v1 remains default. Legacy candidate quarantine and descriptive-only historical scores remain.',
        evidence='docs/OWNERSHIP_RESOLUTION_AUDIT.md')]
    progress.publish('complete' if ready else 'hold',
        'Scoped ownership resolution accepted; ordinary-development prerequisites ready.' if ready else
        'Metadata resolution audited; prospective prerequisite shortage remains on HOLD.',cards,
        'Define the bounded ordinary-value labeling/training recipe and consume this new epoch; keep broad strength claims on HOLD.' if ready else
        'Resolve the named prospective quota or overlap shortage before starting any new experiment.')
    previous=s.read(sorted((s.FOLDER/'events').glob('*.json'))[-1])
    event=dict(previous,acceptance_schema=s.SCHEMA+'-acceptance',baseline_sha256=s.sha(s.FOLDER/'baseline.json'),
        models=s.common.pins(s.MODELS))
    finalpath=s.FOLDER/'events'/f'{len(list((s.FOLDER/"events").glob("*.json"))):06d}.json';s.atomic(finalpath,event)
    inputs=dict(baseline['inputs'])
    files=[p for p in s.FOLDER.rglob('*') if p.is_file() and not p.name.endswith('.pending') and p.name!='acceptance.json']
    inputs.update(s.common.pins(files));inputs.update(audit['source_inputs']);inputs.update(s.common.pins(s.MODELS))
    live=dict(audit['source_inputs'])
    for method in audit['methods'].values():live.update({e['path']:e['sha256'] for e in method['evidence']})
    live.update(s.common.pins([__file__,progress.__file__,progress.HTML,s.FOLDER/'baseline.json',s.PRIOR,
        s.FOLDER/'contract.json',s.FOLDER/'plan.json',s.FOLDER/'production/report.json',
        s.FOLDER/'production/catalog.json',s.FOLDER/'audit/audit.json',finalpath,*s.MODELS]))
    inputs.update(live)
    body=dict(schema=s.SCHEMA+'-acceptance',passed=True,previous_files_preserved=baseline['inputs_count'],
        inputs=inputs,live_inputs=live,models_fixed=True,policy_fixed=True,activity=s.ZERO,
        training_authorized=False,production_evaluation_admitted=False,
        historical_provenance_complete=False,historically_unseen=False,inherited_unknown_background=True,
        contract=c.definition(),epoch=epoch,production_report_sha256=s.sha(s.FOLDER/'production/report.json'),
        progress_event=s.relative(finalpath),progress_event_sha256=s.sha(finalpath))
    s.need(progress.valid_closure(body,finalpath),'Candidate observer closure refused; no acceptance published')
    s.need(all(inputs.get(k)==v for k,v in baseline['inputs'].items()),'Final manifest overwrote a preserved baseline digest')
    # One full pass over the combined preserved/new manifest, immediately before publication.
    s.common.current(inputs);s.atomic(s.FOLDER/'acceptance.json',body)
    s.need(progress.snapshot()['accepted'] is True,'Real completed observer refused published evidence')
    print(config.canonical(dict(passed=True,prerequisites_ready=ready,
        previous_files_preserved=baseline['inputs_count'],ordinary_support_roots=epoch['ordinary_support_roots'],
        training_authorized=False,historical_provenance_complete=False,
        acceptance_sha256=s.sha(s.FOLDER/'acceptance.json'))),flush=True)


if __name__=='__main__':main()
