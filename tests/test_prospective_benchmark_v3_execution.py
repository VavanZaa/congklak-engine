"""The exact live PackedClaims path, without a native/model/process call."""
import base64,ctypes,hashlib,subprocess
from unittest.mock import patch
import pytest
from congklak import prospective_benchmark_v2_guard as previous
from congklak import prospective_benchmark_v3_guard as guard
from congklak import prospective_benchmark_v3_engineering as engineering
from congklak import prospective_benchmark_v2_extension as data
from congklak.prospective_benchmark_v2_adapter import OPERATIONS
from congklak.native import flatten
def loaded():
 registry=data.registry();records=(data.DEFAULT/'components.bin').read_bytes();stamp='pure-live-PackedClaims-regression'
 return guard.LoadedGuard(refs=dict(guard_stamp=stamp),manifest=dict(guard_stamp=stamp,origin_ids={o:i+1 for i,o in enumerate(registry['origins'])}),
  records=records,graph_lookup=guard.Lookup(records,registry['component_ordinals']),allowed={registry['origin_components'][data.OWNER]},forbidden=set(),expected_spec=None,admission_batch=None)
def raw(row):
 p=row['state'];return b''.join(guard.EVENT.pack(bytes(flatten(p)),p['controller'],OPERATIONS[name],4,0,-1,-1) for name in ('played_input','played_output'))
@pytest.mark.parametrize('case_index',range(5))
def test_all_five_cases_accept_actual_live_packed_claims(case_index):
 state=loaded();row=engineering.specification()[case_index];claims=engineering._claims(state,row)
 assert type(claims) is guard.PackedClaims
 with pytest.raises(TypeError,match='not iterable'):previous.reference_raw(raw(row),state,data.OWNER,claims)
 proof=guard.reference_raw(raw(row),state,data.OWNER,claims)
 flags=[flag for index,flag,other in guard.VIOLATION.iter_unpack(proof['violation_records'])]
 assert flags==([] if row['expected_flags']==0 else [row['expected_flags']])
 assert proof['passed'] is (row['expected_flags']==0) and proof['event_count']==2
def test_actual_engineering_receipt_validator_packed_claims_path_is_pure():
 state=loaded();rows=[]
 def forbidden(*a,**k):raise AssertionError('Pure receipt constructed native/model/process')
 with patch.object(ctypes,'CDLL',forbidden),patch.object(ctypes,'WinDLL',forbidden),patch.object(subprocess,'Popen',forbidden):
  for row in engineering.specification():
   source=raw(row);proof=guard.reference_raw(source,state,data.OWNER,engineering._claims(state,row))
   certificate=dict(event_sha256=proof['raw_sha256'],owned_sha256=proof['owned_sha256'],violations_sha256=proof['violation_sha256'],
    valid=True,guard_passed=proof['passed'],event_count=2,attempted_events=2,overflow=False,invalid=False)
   rows.append(dict(case=row,certificate=certificate,raw=base64.b64encode(source).decode(),owned=base64.b64encode(proof['owned_records']).decode(),violations=base64.b64encode(proof['violation_records']).decode()))
  receipt=dict(schema='prospective-benchmark-v3-guard-conformance',specification=engineering.specification(),model_queries=0,searches=0,
   lifecycle_before=dict(created=0,destroyed=0),lifecycle_after=dict(created=0,destroyed=0),rows=rows)
  assert engineering.validate(receipt,state) is True
def test_same_finite_scientific_schedules():
 from congklak import prospective_benchmark_v2_execute as old,prospective_benchmark_v3_execute as current
 for operation in old.OPERATIONS:assert current.schedule(operation)==old.schedule(operation)
