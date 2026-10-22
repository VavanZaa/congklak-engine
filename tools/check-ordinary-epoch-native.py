"""Saved-byte tests and bounded allocation only; no engine or model queries."""
import argparse,ctypes as C,json,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import ordinary_epoch_native as native,ordinary_epoch_common as s
from congklak.ordinary_epoch_resources import private_commit


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(s.FOLDER/'native/author-pure-checks-v3.json'))
 args=parser.parse_args();output=Path(args.output).resolve()
 s.need(output.is_relative_to(s.FOLDER/'native'),'Author receipt is outside the new native namespace')
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
  ('tests.test_ordinary_epoch_native','tests.test_ordinary_epoch_execute'))
 started=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite)
 s.need(result.wasSuccessful(),'Pure native/executor acceptance tests failed')
 lib=C.CDLL(str(native.DLL));k=C.WinDLL('kernel32');k.GetCurrentProcess.restype=C.c_void_p
 handle=k.GetCurrentProcess();samples={'before_maximum_workspace':private_commit(handle)}
 lib.cc_fast_guard_create.argtypes=[C.c_void_p,C.c_uint64,C.c_uint32];lib.cc_fast_guard_create.restype=C.c_void_p
 lib.cc_fast_guard_destroy.argtypes=[C.c_void_p]
 lib.cc_ownership_trace_begin.argtypes=[C.c_uint64,C.c_ubyte];lib.cc_ownership_trace_begin.restype=C.c_int
 lib.cc_ownership_controls_created.restype=C.c_uint64
 guard=lib.cc_fast_guard_create(None,0,345);s.need(guard,'Maximum claim-table allocation failed')
 try:
  s.need(lib.cc_ownership_trace_begin(16000000,1)==1,'Maximum teacher trace allocation failed')
  samples['maximum_native_workspace']=private_commit(handle)
  # Immutable snapshot and packed durable claims bytecopies. This deliberately
  # does not claim to exercise every Python allocator/JSON/raw-audit worst case.
  buffers=[bytearray(count) for count in (16000000*24,4000000*40,8000000*19)]
  samples['workspace_plus_maximum_snapshot_claim_copies']=private_commit(handle)
  del buffers
  controls=int(lib.cc_ownership_controls_created());s.need(controls==0,'Pure checks accidentally created engine controls')
 finally:lib.cc_ownership_trace_end();lib.cc_fast_guard_destroy(guard)
 report=dict(schema=s.SCHEMA+'-native-author-pure-checks',passed=True,tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),elapsed_ms=(time.perf_counter()-started)*1000,
  native_controls_created=controls,new_engine_queries=0,new_labels=0,new_model_forwards=0,
  saved_traces=13,saved_events=1130701,maximum_workspace_private_bytes=samples,
  private_cap_bytes=4*1024**3,capacity_policy=dict(raw_teacher=16000000,raw_arena=2000000,
   unique_owned_teacher=4000000,global_claims=8000000),
  memory_scope='Maximum preallocated native workspace plus maximum immutable raw/owned/claims bytecopies measured; live archive/reference-validation peak remains independently contained, no universal full-cap success promise',
  maximum_archive_or_validation_memory_guaranteed=False,resource_cap_failure='HOLD; no reduced targets, dropped footprints, limit increases or retry',
  source_inputs=s.common.pins([__file__,ROOT/'tests/test_ordinary_epoch_native.py',ROOT/'tests/test_ordinary_epoch_execute.py',
   ROOT/'congklak/ordinary_epoch_execute.py',*[(ROOT/p) for p in native.input_pins()]]))
 s.atomic(output,report)
 print(json.dumps(dict(passed=True,tests=result.testsRun,controls_created=controls,private_bytes=samples)))


if __name__=='__main__':main()
