"""Use the actual retained native diagnostic bytes, never a fictional marker mock."""
import ast,base64,copy,ctypes,hashlib,inspect,json,subprocess,uuid
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import pytest
from congklak import prospective_benchmark_v3_engineering as previous
from congklak import prospective_benchmark_v4_engineering as engineering,prospective_benchmark_v4_guard as guard
from congklak import prospective_benchmark_v2_extension as data
ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases'
def fixture():
 files=sorted(CASES.glob('*.json'));assert len(files)==5
 rows=[json.loads(p.read_text()) for p in files];registry=data.registry();records=(data.DEFAULT/'components.bin').read_bytes();stamp='actual-four-event-fixture'
 static_hash=hashlib.sha256(b''.join(q+b'\0'*4 for q,ordinal in guard.STATIC.iter_unpack(records))).hexdigest()
 loaded=guard.LoadedGuard(refs=dict(guard_stamp=stamp,records_sha256=static_hash),manifest=dict(guard_stamp=stamp,origin_ids={o:i+1 for i,o in enumerate(registry['origins'])}),
  records=records,graph_lookup=guard.Lookup(records,registry['component_ordinals']),allowed={registry['origin_components'][data.OWNER]},forbidden=set(),expected_spec=None,admission_batch=None)
 return files,rows,loaded
def receipt(rows,schema='prospective-benchmark-v4'):
 return dict(schema=schema+'-guard-conformance',specification=engineering.specification(),model_queries=0,searches=0,
  rows=rows,lifecycle_before=dict(created=2,destroyed=0),lifecycle_after=dict(created=2,destroyed=0),
  reused_evidence=engineering.reused_evidence(),fresh_native_guard_traces=0,reused_native_guard_traces=5,
  source_identity=json.loads((ROOT/engineering.REUSED_PROVIDER_IDENTITY_REF['path']).read_text())['provider_identity'])
def test_actual_five_complete_native_cases_old_failure_new_validator_passes():
 files,rows,loaded=fixture();before={str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size,p.stat().st_mtime_ns) for p in files}
 def forbidden(*a,**k):raise AssertionError('Read-only diagnostic fixture constructed native/model/process')
 with patch.object(ctypes,'CDLL',forbidden),patch.object(ctypes,'WinDLL',forbidden),patch.object(subprocess,'Popen',forbidden):
  with pytest.raises(ValueError,match='raw fixture anchors'):previous.validate(receipt(rows,'prospective-benchmark-v3'),loaded)
  assert engineering.validate(receipt(rows),loaded) is True
  for row in rows:
   events=list(guard.EVENT.iter_unpack(base64.b64decode(row['raw'],validate=True)))
   assert [e[2] for e in events]==[1,31,1,32]
   assert row['certificate']['event_count']==row['certificate']['attempted_events']==4
   assert row['certificate']['operation_counts'][1]==2
   assert row['certificate']['played_input_index']==1 and row['certificate']['played_output_index']==3
   assert type(engineering._claims(loaded,row['case'])) is guard.PackedClaims
 assert before=={str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size,p.stat().st_mtime_ns) for p in files}
def test_decode_events_cannot_be_filtered_even_with_resealed_short_hashes():
 files,rows,loaded=fixture();altered=copy.deepcopy(rows)
 for row in altered:
  raw=base64.b64decode(row['raw']);short=raw[24:48]+raw[72:96]
  row['raw']=base64.b64encode(short).decode();row['certificate']['event_count']=row['certificate']['attempted_events']=2
  row['certificate']['event_sha256']=hashlib.sha256(short).hexdigest()
 with pytest.raises(ValueError,match='raw fixture anchors'):engineering.validate(receipt(altered),loaded)
def test_diagnostic_receipt_reuses_actual_decode_hooks_without_native_calls():
 files,rows,loaded=fixture()
 def forbidden(*a,**k):raise AssertionError('Reused receipt performed another native call')
 class NoNative:
  def __getattr__(self,name):raise AssertionError('Reused receipt accessed native '+name)
 identity=json.loads((ROOT/engineering.REUSED_PROVIDER_IDENTITY_REF['path']).read_text())['provider_identity']
 provider=SimpleNamespace(identity=identity,native=NoNative(),lifecycle=lambda:dict(created=2,destroyed=0))
 output=ROOT/'build/prospective-benchmark-v4/fixtures/reused-reader'/uuid.uuid4().hex/'receipt.json';milestones=[]
 with patch.object(ctypes,'CDLL',forbidden),patch.object(ctypes,'WinDLL',forbidden),patch.object(subprocess,'Popen',forbidden),patch.object(guard.LoadedGuard,'attach',forbidden):
  engineering.run(provider,loaded,output,lambda point,**fields:milestones.append(point))
 result=json.loads(output.read_text());assert result['rows']==rows and result['fresh_native_guard_traces']==0 and result['reused_native_guard_traces']==5
 assert milestones==['diagnostics_before_reused_evidence_validation','diagnostics_after_pure_validation','diagnostics_after_receipt']
def test_scientific_schedules_unchanged():
 from congklak import prospective_benchmark_v3_execute as old,prospective_benchmark_v4_execute as current
 for operation in old.OPERATIONS:assert current.schedule(operation)==old.schedule(operation)
