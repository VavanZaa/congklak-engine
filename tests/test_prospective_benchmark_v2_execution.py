"""Pure implementation conformance: no native loader, model, process or query."""
import ctypes,hashlib,struct,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from congklak import value_run_arena as oldliteral
from congklak import prospective_benchmark_v2_literal as literal
from congklak import prospective_benchmark_v2_guard as guard
from congklak import prospective_benchmark_v2_engineering as engineering
from congklak import prospective_benchmark_v2_execute as execute
from congklak import prospective_benchmark_v2_native as native
from congklak import prospective_benchmark_v2_timing as timing
from congklak import prospective_benchmark_v2_extension as data
ROOT=Path(__file__).resolve().parents[1]
def test_cached_literal_same_exact_rules_and_live_seal():
 rules=literal.LiteralRules(literal.sha(literal.REFERENCE));ref=oldliteral._literal()
 for meta in data.membership()['arena']:
  p=meta['state'];normal=rules.normalize(p)
  assert normal==oldliteral.normalize(p,reference=ref)
  for move in normal['legal']:assert rules.transition(p,move)==oldliteral.transition(p,move,reference=ref)
 original=rules._reference.normalize;rules._reference.normalize=lambda p:None
 with pytest.raises(ValueError,match='mutated'):rules.normalize(data.membership()['arena'][0]['state'])
 rules._reference.normalize=original;assert rules.close()['module_constructions']==1
 with pytest.raises(ValueError,match='seal'):rules.normalize(data.membership()['arena'][0]['state'])
 forged=object.__new__(literal.LiteralRules)
 with pytest.raises(ValueError):forged.verify()
def _loaded():
 records=(data.DEFAULT/'components.bin').read_bytes();registry=data.registry();cid=registry['origin_components'][data.OWNER]
 return SimpleNamespace(graph_lookup=guard.Lookup(records,registry['component_ordinals']),allowed={cid},forbidden=set(),guard_stamp='pure-test',
  owner_id=lambda name:registry['origins'].index(name)+1,claims_blob=lambda claimed:b'')
def test_strict_pure_guard_cases_include_default_deny_and_live_conflict():
 loaded=_loaded()
 from congklak.native import flatten
 from congklak.prospective_benchmark_v2_adapter import OPERATIONS
 from congklak.neurodiag_data import quotient
 for row in engineering.specification():
  p=row['state'];raw=b''.join(guard.EVENT.pack(bytes(flatten(p)),p['controller'],OPERATIONS[name],4,0,-1,-1) for name in ('played_input','played_output'))
  claims={quotient(p):data.SHARED} if row['seed_other_owner'] else {}
  proof=guard.reference_raw(raw,loaded,data.OWNER,claims)
  flags=[flags for index,flags,other in guard.VIOLATION.iter_unpack(proof['violation_records'])]
  assert flags==([] if row['expected_flags']==0 else [row['expected_flags']])
  assert proof['event_count']==2 and proof['passed'] is (row['expected_flags']==0)
def test_projection_changes_only_stricter_static_default():
 for name in ('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp','trace.hpp'):
  assert (ROOT/'native/guarded_value_pilot_v2'/name).read_bytes()==(ROOT/'native/prospective_benchmark_v2'/name).read_bytes()
 before=(ROOT/'native/guarded_value_pilot_v2/trace.cpp').read_bytes();after=(ROOT/'native/prospective_benchmark_v2/trace.cpp').read_bytes()
 fragment=b'return i!=statics.end()&&i->key==key?i->value:0;'
 assert before.count(fragment)==1 and after==before.replace(fragment,fragment[:-2]+b'2;')
def test_source_identity_and_schedule_cannot_construct_native_or_process():
 def forbidden(*a,**k):raise AssertionError('Pure read constructed native/process')
 with patch.object(ctypes,'WinDLL',forbidden),patch.object(ctypes,'CDLL',forbidden),patch.object(subprocess,'Popen',forbidden):
  identity=native.source_identity();assert identity['unknown_high_default_deny'] is True
  assert {k:len(execute.schedule(k)) for k in execute.OPERATIONS}==dict(engineering=72,endgame=128,arena=96)
  arena=execute.schedule('arena');assert sum(r['meta']['benchmark_stratum']=='boundary25' for r in arena)==64
  assert sum(r['meta']['benchmark_stratum']=='optional26-32' for r in arena)==32
  assert all(r['engine'] in ('control','candidate','H_v1','H8_relay_CPU') for r in execute.schedule('engineering'))
def test_fullclock_is_wall_authoritative_and_does_not_change_gc_policy():
 import gc,time
 before=(gc.isenabled(),gc.get_threshold(),list(gc.callbacks));observer=timing.GcObserver()
 start=time.perf_counter_ns();clock=observer.begin('pure',start);clock.mark('native');clock.mark('literal');clock.mark('cpu');result=clock.finish()
 assert result['critical_ms']>=result['phase_end_ms']['cpu']>=result['phase_end_ms']['literal']>=0
 assert result['gc_policy_changed'] is False and result['thread_cpu_ms']>=0 and result['process_cpu_ms']>=0
 observer.close();assert before==(gc.isenabled(),gc.get_threshold(),list(gc.callbacks))
