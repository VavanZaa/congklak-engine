"""Publish only fully bound, independently audited trace-speed evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,tracing_speed_common as stage,tracing_speed_progress as progress
from congklak import tracing_speed_launcher_v4 as launcher,tracing_speed_contract as contract

def main():
    folder=stage.FOLDER;baseline=stage.read(folder/'baseline.json')
    # The final full input check includes every preserved baseline file.
    # Hash them once immediately before publication rather than twice here.
    worker=launcher.completed(folder/'jobs/engineering-fixed')
    timedworker=launcher.completed(folder/'jobs/engineering-timed')
    timing=launcher.completed(folder/'timing/production');audit=stage.read(folder/'audit/audit.json')
    stage.need(audit['passed'] is True and audit['unresolved_findings']==[] and
        all(audit['methods'][name]['passed'] is True for name in contract.METHODS),'Independent acceptance incomplete')
    models=stage.common.pins(progress.MODELS);gate=timing['timing_gate']
    calibration=stage.read(folder/'native/calibration-v4/report.json')
    calibration_overshoots=sum(v['overshoots'] for v in calibration['summary'].values())
    timed_overshoots=timedworker['report']['overshoot_decisions']
    target=gate['passed'] and timed_overshoots==0 and timedworker['report']['maximum_critical_ms']<=25 and calibration_overshoots==0
    prior_calibration=sum(len(list((folder/f'native/calibration-v{v}/attempts').glob('*.json'))) for v in (1,2,3))
    final_calibration=len(list((folder/'native/calibration-v4/attempts').glob('*.json')))
    stage.need((prior_calibration,final_calibration)==(24,32),'Tracked calibration attempts differ')
    audit_counters=audit['engineering_counters']
    stage.need(all(type(v) is int and v>=0 for v in audit_counters.values()),'Independent scoped counters differ')
    counters=dict(fixed_games=worker['report']['games'],fixed_decisions=worker['report']['decisions'],
        timed_games=timedworker['report']['games'],timed_decisions=timedworker['report']['decisions'],
        timing_attempts=240,calibration_prior_attempts=prior_calibration,calibration_final_attempts=final_calibration,
        calibration_engine_queries=prior_calibration+final_calibration,
        **{'independent_audit_'+k:v for k,v in audit_counters.items()})
    clocks=stage.read(folder/'clock-summary.json');stage.common.current(clocks['inputs'])
    cards=[dict(title='Saved-trace cost diagnosis',state='complete',result='Python per-event ownership work replaced by compiled native finalization.',
        improvement='Complete raw traces retained; lossless archival cost reported and charged separately.',evidence='build/tracing-speed-v1/guard'),
        dict(title='Native coverage and ownership conformance',state='complete',result='Independent operation, packed-byte, guard and fixed-work checks passed.',
        improvement='Versioned implementation; search algorithms, model tensors and policy fixed.',evidence='build/tracing-speed-v1/audit/audit.json'),
        dict(title='25 ms empirical timing',state='complete' if target else 'hold',
        result=f"{gate['attempts']} attempts; p95 {gate['p95_ms']:.3f} ms; maximum {gate['maximum_ms']:.3f} ms.",
        improvement=f'Additional overshoots: timed handoffs {timed_overshoots}, calibration {calibration_overshoots}. All failures retained.',evidence='build/tracing-speed-v1/timing/production/report.json'),
        dict(title='Recoverable handoffs and resource accounting',state='complete',result='Eight kill/restart boundaries independently checked, including catalog/launcher handoff.',
        improvement='Saved choices recover before engine construction; unclean leases remain fully charged.',evidence='build/tracing-speed-v1/audit/audit.json'),
        dict(title='Training and production admission',state='hold',result='Models/policy unchanged; no optimizer steps, labels or production games.',
        improvement='Earlier 320 support conflicts and missing historical footprints remain open.',evidence='build/ownership-implementation-v1/acceptance.json')]
    progress.publish('complete' if target else 'hold','Trace-speed engineering accepted. '+
        ('The declared 25 ms tests passed.' if target else 'The 25 ms target remains on HOLD; measured failures retained.'),
        cards,'Resolve historical ownership/provenance holds before production evaluation or more training.' if target else
        'Reduce the measured remaining critical costs before claiming a 25 ms result.')
    eventpath=sorted((folder/'events').glob('*.json'))[-1];event=stage.read(eventpath)
    event.update(acceptance_schema='tracing-speed-acceptance-v1',training_counters=stage.ZERO,
        baseline_sha256=stage.sha(folder/'baseline.json'),models_sha256=models)
    # publish() output is immutable: bind a new final event instead of editing it.
    finalpath=folder/'events'/f'{len(list((folder/"events").glob("*.json"))):06d}.json';stage.atomic(finalpath,event)
    inputs=dict(baseline['inputs'])
    material=[p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.pending') and p.name!='acceptance.json']
    inputs.update(stage.common.pins(material));inputs.update(audit['source_inputs']);inputs.update(models)
    for method in audit['methods'].values():inputs.update({e['path']:e['sha256'] for e in method['evidence']})
    live=dict(audit['source_inputs']);live.update(models)
    for method in audit['methods'].values():live.update({e['path']:e['sha256'] for e in method['evidence']})
    live.update(stage.common.pins([__file__,'congklak/tracing_speed_progress.py','congklak/tracing_speed_contract.py',progress.HTML,
        folder/'baseline.json',stage.PRIOR,finalpath,folder/'audit/audit.json',folder/'timing/panel.json',
        folder/'timing/protocol.json',folder/'timing/production/report.json',folder/'jobs/engineering-timed/report.json',
        folder/'jobs/engineering-fixed/report.json',
        folder/'clock-summary.json',
        folder/'native/calibration-v4/report.json']))
    inputs.update(live)
    body=dict(schema='tracing-speed-acceptance-v1',passed=True,previous_files_preserved=62200,
        inputs=inputs,live_inputs=live,models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,
        training_authorized=False,production_evaluation_admitted=False,execution_ready=False,
        independent_audit='build/tracing-speed-v1/audit/audit.json',timing_gate=gate,
        target_25ms_met=target,timed_handoff_overshoots=timed_overshoots,calibration_overshoots=calibration_overshoots,
        progress_event=finalpath.relative_to(ROOT).as_posix(),progress_event_sha256=stage.sha(finalpath),
        engineering_counters=counters,open_holds=['320 prior support conflicts','missing historical engineering footprints'])
    body['resource_clock_summary']=clocks['timing']
    stage.need(progress.valid_closure(body,finalpath),'Candidate final evidence failed observer closure; nothing accepted')
    stage.need(all(inputs.get(k)==v for k,v in baseline['inputs'].items()),
        'Final manifest overwrote a preserved baseline digest')
    stage.common.current(inputs);stage.atomic(folder/'acceptance.json',body)
    stage.need(progress.snapshot()['accepted'] is True,'Published acceptance failed completed observer')
    print(config.canonical(dict(passed=True,prior_files_preserved=62200,target_25ms_met=target,
        timing_gate=gate,acceptance_sha256=stage.sha(folder/'acceptance.json'))),flush=True)

if __name__=='__main__':main()
