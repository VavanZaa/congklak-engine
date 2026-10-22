"""Read-only profiling on already archived traces, before new searches."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,time,hashlib
from congklak import native_ownership_trace as old

OUT=ROOT/'build/tracing-speed-v1/native'
OUT.mkdir(parents=True,exist_ok=True)
results=[];inputs={}
for name in old.ENGINES:
 path=ROOT/('build/ownership-implementation-v1/native/author-v2/timed-'+name+'.json')
 envelope=old.read(path);trace=envelope['trace'];inputs[str(path.relative_to(ROOT)).replace('\\','/')]=old.sha(path)
 row={'engine':name,'events':trace['event_count'],'raw_bytes':trace['byte_count'],'compressed_bytes':trace['compressed_bytes']}
 for key,fn in [('decode_hash_ms',lambda:old.trace_bytes(trace)),('trace_partition_ms',lambda:old.validate_trace(trace)),
                ('physical_envelope_ms',lambda:old.validate_decision_coverage(envelope)),('footprint_ms',lambda:old.compile_footprint(trace))]:
  tick=time.perf_counter_ns();value=fn();row[key]=(time.perf_counter_ns()-tick)/1e6
  if key=='footprint_ms':row['distinct_quotients']=len(value['rows']);row['distinct_high_quotients']=sum(x['owned'] for x in value['rows'])
 results.append(row)
inputs['congklak/native_ownership_trace.py']=old.sha(ROOT/'congklak/native_ownership_trace.py')
inputs['tools/profile-existing-traces-v1.py']=old.sha(__file__)
body={'schema':'existing-trace-cost-profile-v1','new_searches':0,'new_games':0,'optimizer_steps':0,'new_teacher_labels':0,
 'results':results,'inputs':inputs,'interpretation':'one read-only timing pass; no strength result'}
path=OUT/'profile-existing-v1.json'
assert not path.exists();path.write_text(json.dumps(body,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(body),flush=True)
