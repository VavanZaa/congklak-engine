"""Replay immutable packed traces through the new native finalizer, no searches."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ctypes as C,hashlib,json,time,argparse
from congklak import native_trace_fast_v1 as fast,native_ownership_trace as old
from congklak.native import Native

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='build/tracing-speed-v1/native/saved-native-v1');parser.add_argument('--repeats',type=int,default=5)
 args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
 expected=ROOT/'build/tracing-speed-v1/audit/saved-trace-expectations/expectations.json';oracle=old.read(expected)
 identity=fast.source_identity();inputs={str(expected.relative_to(ROOT)).replace('\\','/'):old.sha(expected),
  'congklak/native_trace_fast_v1.py':old.sha(ROOT/'congklak/native_trace_fast_v1.py'),
  'tools/profile-native-finalization-v1.py':old.sha(__file__)}
 (out/'intent.json').write_text(json.dumps(dict(schema='saved-byte-replay-intent-v1',identity=identity,inputs=inputs,new_searches=0),sort_keys=True))
 n=Native(path=fast.DLL);old._attach(n);fast.attach_api(n);lib=n.lib
 guard=lib.cc_fast_guard_create(None,0,1);assert guard
 rows=[]
 try:
  for fixture in oracle['traces']:
   path=ROOT/fixture['source'];assert old.sha(path)==fixture['source_sha256'];envelope=old.read(path);trace=envelope['trace'];raw=old.trace_bytes(trace)
   inputs[fixture['source']]=old.sha(path);buffer=C.create_string_buffer(raw);counts=(C.c_uint64*64)()
   for name,value in fixture['operation_counts'].items():counts[old.OPERATIONS[name]]=value
   for trial in range(args.repeats):
    cursor=f'{len(rows):03d}';record=dict(source=fixture['source'],trial=trial,expected_complete=fixture['expected_complete'])
    (out/(cursor+'-intent.json')).write_text(json.dumps(record,sort_keys=True))
    tick=time.perf_counter_ns();assert lib.cc_fast_profile_load(buffer,len(raw)//24)==1
    record['capture_replay_ms']=(time.perf_counter_ns()-tick)/1e6
    assert lib.cc_fast_profile_expected(trace['attempted_events'],int(trace['overflow']),int(trace['invalid']),counts)==1
    tick=time.perf_counter_ns();cert=json.loads(lib.cc_fast_finalize(guard,1).decode())
    ownedraw=C.string_at(lib.cc_ownership_trace_data(),cert['event_count']*24)
    owned=C.string_at(lib.cc_fast_owned_data(),cert['owned_count']*40) if cert['owned_count'] else b''
    violations=C.string_at(lib.cc_fast_violation_data(),cert['violations_count']*16) if cert['violations_count'] else b''
    assert hashlib.sha256(ownedraw).hexdigest()==cert['event_sha256'] and hashlib.sha256(owned).hexdigest()==cert['owned_sha256']
    assert hashlib.sha256(violations).hexdigest()==cert['violations_sha256']
    record['finalize_copy_hash_ms']=(time.perf_counter_ns()-tick)/1e6
    assert ownedraw==raw and cert['event_sha256']==fixture['raw_sha256']
    assert cert['owned_keys_sha256']==fixture['high_key_bytes_sha256']
    keys=[fast.OWNED.unpack_from(owned,i)[0].hex() for i in range(0,len(owned),40)]
    assert keys==fixture['high_unique_keys']
    assert cert['valid']==fixture['expected_complete'] and cert['guard_passed']==fixture['expected_complete']
    if cert['valid']:
     assert cert['operation_counts']==list(counts)
     assert cert['scope_mask']==sum(1<<k for k in fixture['scopes'])
    record.update(passed=True,certificate=cert)
    (out/(cursor+'-result.json')).write_text(json.dumps(record,sort_keys=True))
    rows.append(record);lib.cc_fast_guard_discard_stage(guard);lib.cc_ownership_trace_end()
 finally:lib.cc_fast_guard_destroy(guard)
 final=dict(schema='saved-byte-native-finalization-profile-v1',passed=True,new_searches=0,new_games=0,optimizer_steps=0,
  repetitions=args.repeats,inputs=inputs,results=rows,identity=identity)
 (out/'result.json').write_text(json.dumps(final,sort_keys=True))
 print(json.dumps(dict(passed=True,traces=len(oracle['traces']),replays=len(rows),maximum_finalize_ms=max(x['finalize_copy_hash_ms'] for x in rows))))
if __name__=='__main__':main()
