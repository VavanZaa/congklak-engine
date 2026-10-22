"""Profile immutable saved traces, without engines, model forwards or new play."""
import hashlib
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,native_ownership_trace as codec,ownership_fast_guard as fast
from congklak import ownership_handoff_worker as old


def clock(function):
    start=time.perf_counter_ns();value=function();return value,(time.perf_counter_ns()-start)/1e6


def main():
    spec_path=fast.ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json'
    spec=fast.read(spec_path);graph_path=fast.ROOT/spec['guard']['graph_path']
    graph,graph_ms=clock(lambda:fast.read(graph_path))
    (records,flags,forbidden,allowed,shared),compile_ms=clock(lambda:fast._static(spec['guard'],graph))
    origins=sorted({row['origin'] for row in spec['schedule']})
    loaded=fast.LoadedGuard({'records_sha256':hashlib.sha256(records).hexdigest()},
        {'origin_ids':{v:i+1 for i,v in enumerate(origins)},'guard_stamp':'saved-profile-only'},
        records,flags,graph['quotient_components'],allowed,forbidden)
    files=['fixed-00-control.json','fixed-00-candidate.json','fixednodes-H_v1.json','fixednodes-H8_relay_CPU.json','timed-H_v1.json']
    inputs=fast.common.pins(['tools/profile-ownership-guard.py','congklak/ownership_fast_guard.py',
        'congklak/ownership_handoff_worker.py','congklak/native_ownership_trace.py',
        spec_path.relative_to(fast.ROOT).as_posix(),graph_path.relative_to(fast.ROOT).as_posix()])
    results=[];origin=origins[0]
    # Warm the old graph only: each trace footprint compilation remains cold.
    old._graph(spec['guard']['graph_path'],spec['inputs'][spec['guard']['graph_path']])
    for name in files:
        path=fast.ROOT/'build/ownership-implementation-v1/native/author-v2'/name
        answer=fast.read(path);inputs[path.relative_to(fast.ROOT).as_posix()]=fast.sha(path)
        raw,decode_ms=clock(lambda:codec.trace_bytes(answer['trace']))
        old._FOOTPRINT_CACHE.clear()
        checked,old_ms=clock(lambda:old.check_ownership(answer,origin,spec,{}))
        reference,literal_ms=clock(lambda:fast.reference_raw(raw,loaded,origin,{}))
        old_owned={row['quotient']:row for row in checked['footprint']['rows'] if row['owned']}
        new_owned=fast.owned_rows(reference['owned_records'],reference['event_count'])
        fast.need(set(old_owned)=={r['quotient'] for r in new_owned},'Literal high footprint differs from accepted compiler')
        for row in new_owned:
            prior=old_owned[row['quotient']];state=prior['state']
            raw_state=bytes(state['pits'][0]+[state['stores'][0]]+state['pits'][1]+[state['stores'][1],state['controller']]).hex()
            fast.need(prior['first_event']==row['first_event'] and raw_state==row['raw_state_key'],
                      'Saved literal first state/event differs')
        fast.need(checked['passed']==reference['passed'],'Saved literal guard result differs')
        row=dict(source=path.relative_to(fast.ROOT).as_posix(),source_sha256=fast.sha(path),
            event_count=reference['event_count'],raw_bytes=len(raw),owned_unique_count=reference['owned_unique_count'],
            raw_sha256=reference['raw_sha256'],owned_sha256=reference['owned_sha256'],
            decompress_ms=decode_ms,old_warm_graph_cold_footprint_guard_ms=old_ms,
            pure_literal_raw_guard_ms=literal_ms,parity=True)
        results.append(row);print(config.canonical(row),flush=True)
    report=dict(schema='saved-ownership-guard-profile-v1',inputs=inputs,results=results,
        graph_load_ms=graph_ms,packed_compile_ms=compile_ms,static_records=len(flags),static_bytes=len(records),
        graph_shared_source_keys=len(shared),graph_upper_keys=sum(sum(q[:14])>=75 for q in flags),
        observation='Saved-trace readers are off-clock conformance references; native timing is measured separately.',
        claims='No speed claim for the native path is derived from this Python profile.',
        model_forwards=0,searches=0,new_games=0,new_positions=0,models_fixed=True,policy_fixed=True,
        training_admission=False,evaluation_admission=False,production_admission=False)
    output=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else fast.DEFAULT/'profile-v2.json'
    fast.need(output.is_relative_to(fast.DEFAULT),'Profile output outside new guard namespace')
    fast._publish(output,(config.canonical(report)+'\n').encode())
    print(config.canonical({'report':str(output),'sha256':fast.sha(output)}),flush=True)


if __name__=='__main__':main()
