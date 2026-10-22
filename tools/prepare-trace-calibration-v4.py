"""Prepare the finite calibration contract without creating engine providers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,argparse
from congklak import native_trace_fast_v2 as fast,native_ownership_trace as old
from congklak import tracing_speed_common as stage,tracing_speed_launcher_v4 as launcher,ownership_fast_guard as guards

parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=6);args=parser.parse_args()
OUT=ROOT/'build/tracing-speed-v1/native/calibration-v4'
SPEC=ROOT/f'build/tracing-speed-v1/native/calibration-spec-v{args.version}.json'
oldspec=old.read(ROOT/'build/ownership-implementation-v1/handoff/engineering-spec.json')
oldfolder=ROOT/'build/ownership-implementation-v1/native/author-v2'
roots={key:old.read(oldfolder/f'fixed-{key}-control.json')['trace']['binding']['root_state'] for key in ('00','01','08')}
schedule=[]
for engine in old.ENGINES:
 keys=('00',) if engine in ('H_v1','H8_relay_CPU') else ('00','01','08')
 for key in keys:
  path=oldfolder/(f'fixednodes-{engine}.json' if engine in ('H_v1','H8_relay_CPU') else f'fixed-{key}-{engine}.json')
  schedule.append(dict(index=len(schedule),mode='fixed128',engine=engine,root=key,state=roots[key],origin='engineering-consumed-139',
   oracle=path.relative_to(ROOT).as_posix(),oracle_sha256=old.sha(path)))
for repetition in range(2):
 for key in ('00','01','08'):
  for engine in old.ENGINES:
   schedule.append(dict(index=len(schedule),mode='timed',engine=engine,root=key,state=roots[key],origin='engineering-consumed-139',repetition=repetition))
assert len(schedule)==32
profile=ROOT/'build/tracing-speed-v1/native/saved-native-v3/result.json'
policy=dict(schema='trace-finalization-debt-v1',reserve_ms=12.0,minimum_search_ms=10,maximum_outer_ms=25,
 same_all_engines=True,calibrated=True,calibration_kind='saved-byte profile; prospective bounded live confirmation',
 calibration_source=profile.relative_to(ROOT).as_posix(),calibration_source_sha256=old.sha(profile),
 no_worst_case_guarantee=True,overshoots_retained=True,maximum_events=2000000)
sources=['tools/prepare-trace-calibration-v4.py','congklak/trace_fast_calibration_v4.py','congklak/native_trace_fast_v2.py',
 'congklak/trace_fast_adapter_v2.py','congklak/trace_puct_witness_v1.py',
 'native/puct_witness_v1.cpp','tools/build-trace-witness-v1.cmd','tools/build-trace-witness-v1.ps1',
 'build/tracing-speed-v1/native/witness-build-v1.json','build/tracing-speed-v1/native/puct_witness_v1.dll','congklak/trace_fast_comparison_v1.py',
 'build/tracing-speed-v1/native/semantic-comparator-v1.json','build/tracing-speed-v1/native/comparator-review-v1/result.json',
 'tools/audit-puct-deadline-coverage-v3.py','build/tracing-speed-v1/native/deadline-coverage-v3/result.json',
 'build/tracing-speed-v1/native/adapter-projection-v2.json','build/tracing-speed-v1/native/build.json',
 'build/tracing-speed-v1/native/congklak_trace_fast_v1.dll','build/tracing-speed-v1/native/saved-native-v3/result.json',
 *launcher.SOURCES,*launcher.resources.SOURCES,'congklak/tracing_speed_common.py','congklak/ownership_fast_guard.py',
 'congklak/value_run_arena.py','tests/reference.py','congklak/timing_v2.py',*fast.build_record()['inputs']]
inputs=dict(oldspec['inputs']);inputs.update(stage.common.pins(sources))
inputs.update(stage.common.pins(['build/tracing-speed-v1/native/calibration-v1/attempts/000000.json',
 'build/tracing-speed-v1/native/memory-telemetry-repair-v1.json',
 *[f'build/tracing-speed-v1/native/calibration-v2/attempts/{i:06d}.json' for i in range(3)],
 *[f'build/tracing-speed-v1/native/calibration-v3/attempts/{i:06d}.json' for i in range(20)]]))
for row in schedule:
 if 'oracle' in row:inputs[row['oracle']]=row['oracle_sha256']
spec=dict(schema='fast-native-bounded-calibration-v1',kind='calibration',engineering_fixture=True,
 models_fixed=True,policy_fixed=True,training_counters=stage.ZERO,inputs=inputs,guard=oldspec['guard'],
 launcher_identity=launcher.identity(),schedule=schedule,search_policy=policy,capacity_events=2000000,
 scope='Consumed immutable roots only; fixed-work conformance plus bounded timing calibration; no strength or freshness claim',
 prior_actual_engineering_searches=24,prior_search_quality_failure=False,
 explicit_fixed_retest_after_reader_repair=True,total_calibration_engineering_search_budget=56,
 final_policy_will_not_change_from_panel_outcomes=True,failed_attempts='Archive raw+prefix and HOLD without rerun or filtering')
spec['guard_refs']=guards.prepare(spec,ROOT/f'build/tracing-speed-v1/guard/calibration-v{args.version}')
stage.atomic(SPEC,spec)
print(json.dumps(dict(prepared=True,spec=SPEC.relative_to(ROOT).as_posix(),spec_sha256=old.sha(SPEC),attempts=len(schedule),
 fixed=8,timed=24,search_policy=policy,output=OUT.relative_to(ROOT).as_posix())),flush=True)
