"""Separate critical, encoding/flush and full contained resource clocks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,tracing_speed_common as stage,tracing_speed_resources as resources

def main():
    report=resources.status();stage.need(report['pending'] is None,'Wait for all contained leases to settle')
    folder=stage.FOLDER;timingpath=folder/'timing/production/report.json';timing=stage.read(timingpath)
    reserves={e['lease']['token']:e['lease'] for e in report['events'] if e['kind']=='reserve'}
    finishes=[e for e in report['events'] if e['kind']=='finish']
    stage.need(len(finishes)==len(reserves) and not any(e['unknown_overrun'] for e in finishes),'Unknown resource overrun')
    by_output={}
    for event in finishes:
        lease=reserves[event['token']];name=Path(lease['worker_output']).relative_to(ROOT).as_posix()
        row=by_output.setdefault(name,dict(leases=0,contained_elapsed_seconds=0.,charged_seconds=0.,unclean_charged_seconds=0.))
        row['leases']+=1;row['contained_elapsed_seconds']+=event['elapsed_ns']/1e9;row['charged_seconds']+=event['charged_ns']/1e9
        if not event['clean']:row['unclean_charged_seconds']+=event['charged_ns']/1e9
    phase=by_output['build/tracing-speed-v1/timing/production']
    inputs=stage.common.pins([__file__,'congklak/tracing_speed_resources.py',timingpath,
        resources.DEFAULT/'identity.json',resources.DEFAULT/'head.json',*sorted((resources.DEFAULT/'events').glob('*.json'))])
    body=dict(schema='trace-speed-clock-summary-v1',inputs=inputs,
        timing=dict(attempts=240,critical_sum_ms=sum(r['critical_ms'] for r in timing['rows']),
            critical_p95_ms=timing['gate']['p95_ms'],critical_maximum_ms=timing['gate']['maximum_ms'],
            encoding_and_mapped_flush_sum_ms=sum(r['archival_ms'] for r in timing['rows']),**phase),
        all_contained=dict(leases=len(finishes),charged_seconds=report['charged_ns']/1e9,
            remaining_seconds=report['remaining_ns']/1e9,limit_seconds=report['identity']['protocol']['cumulative_ns']/1e9,
            contained_elapsed_seconds=sum(e['elapsed_ns'] for e in finishes)/1e9,
            unclean_charged_seconds=sum(e['charged_ns'] for e in finishes if not e['clean'])/1e9),
        by_output=by_output,models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,
        interpretation='Critical decisions include reset, search, complete raw capture/hash/physical validation, ownership checks and claim staging. Encoding/mapped-flush sums are a subset of off-clock work. Full contained elapsed includes setup, JSON/fsync/journal publication and repeated persisted-evidence verification. Supervisor and independent read-only audit time is additional. The25ms result applies to the declared critical decision clock; no end-to-end or worst-case latency guarantee.')
    stage.atomic(folder/'clock-summary.json',body)
    print(config.canonical(dict(timing=body['timing'],all_contained=body['all_contained'])),flush=True)

if __name__=='__main__':main()
