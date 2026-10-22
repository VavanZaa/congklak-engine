"""Pure v5 receipts and actual retained evidence; never load a DLL or query."""
import ast,base64,copy,ctypes,hashlib,json,subprocess
from pathlib import Path
from unittest.mock import patch
import pytest
from congklak import config
from congklak import prospective_benchmark_v5_adapter as adapter,prospective_benchmark_v5_core as core
from congklak import prospective_benchmark_v5_execute as execute,prospective_benchmark_v4_execute as previous
from congklak import prospective_benchmark_v5_engineering as engineering,prospective_benchmark_v5_guard as guard
from congklak import prospective_benchmark_v5_native as native,prospective_benchmark_v5_common as stage
from congklak import prospective_benchmark_v2_extension as data
ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/'build/prospective-benchmark-v3/jobs/engineering/diagnostics/cases'
ARENA=ROOT/'build/prospective-benchmark-v4/jobs/arena'
def forbidden(*a,**k):raise AssertionError('Pure conformance attempted native/model/process work')
@pytest.fixture(autouse=True)
def no_native_or_process():
 with patch.object(ctypes,'CDLL',forbidden),patch.object(ctypes,'WinDLL',forbidden),patch.object(subprocess,'Popen',forbidden):yield
def initial(stem):
 item=dict(tt_mb=16,capacity=131072,leased=False,epoch=0,cleared_entries=0,empty_entries=0,control_generation=0,start_nodes=0,start_deposits=0)
 if stem=='relay':item.update(cache_capacity=8192,cleared_cache=0,empty_cache=0)
 return item
def preallocation():
 life=dict(created=2,destroyed=0)
 return dict(before=life,after=life,elapsed_ns=1,model_queries=0,searches=0,empty_finalize_calls=0,guard_generation_changes=0,
  workspace_preallocation={s:initial(s) for s in ('h','relay')}),life
def reset_receipt(engine):
 before={s:initial(s) for s in ('h','relay')};after=copy.deepcopy(before)
 selected={'H_v1':'h','H8_relay_CPU':'relay'}.get(engine)
 if selected:
  after[selected].update(epoch=1,cleared_entries=131072,empty_entries=131072,control_generation=5)
  if selected=='relay':after[selected].update(cleared_cache=8192,empty_cache=8192)
 return dict(coverage=dict(workspace_reset=dict(before=before,after=after)),lifecycle=dict(after=dict(created=5)))
def test_constructor_preallocations_are_both_fresh_and_search_free():
 receipt,life=preallocation();assert core.validate_preallocation(receipt,life)
 for mutation in ('missing','searched','stale','wrong_capacity','leased'):
  altered=copy.deepcopy(receipt)
  if mutation=='missing':del altered['workspace_preallocation']['relay']
  elif mutation=='searched':altered['searches']=1
  elif mutation=='stale':altered['workspace_preallocation']['h']['epoch']=1
  elif mutation=='wrong_capacity':altered['workspace_preallocation']['relay']['cache_capacity']=4096
  else:altered['workspace_preallocation']['h']['leased']=True
  with pytest.raises(ValueError):core.validate_preallocation(altered,life)
@pytest.mark.parametrize('engine',['control','candidate','H_v1','H8_relay_CPU'])
def test_every_arm_validates_pool_epoch_and_full_reset(engine):
 assert adapter.validate_workspace_reset(reset_receipt(engine),engine)
@pytest.mark.parametrize('field,value',[('epoch',0),('cleared_entries',131071),('empty_entries',131071),('start_nodes',1),('start_deposits',1),('control_generation',4),('leased',True),('cleared_cache',8191),('empty_cache',8191)])
def test_selected_relay_rejects_stale_or_missing_resets(field,value):
 answer=reset_receipt('H8_relay_CPU');answer['coverage']['workspace_reset']['after']['relay'][field]=value
 with pytest.raises(ValueError):adapter.validate_workspace_reset(answer,'H8_relay_CPU')
def test_puct_cannot_change_either_baseline_pool():
 answer=reset_receipt('candidate');answer['coverage']['workspace_reset']['after']['h']['epoch']=1
 with pytest.raises(ValueError):adapter.validate_workspace_reset(answer,'candidate')
def diagnostic_fixture():
 files=sorted(CASES.glob('*.json'));assert len(files)==5
 rows=[json.loads(p.read_text()) for p in files];registry=data.registry();records=(data.DEFAULT/'components.bin').read_bytes();stamp='actual-four-event-v5-validator-fixture'
 static_hash=hashlib.sha256(b''.join(q+b'\0'*4 for q,ordinal in guard.STATIC.iter_unpack(records))).hexdigest()
 loaded=guard.LoadedGuard(refs=dict(guard_stamp=stamp,records_sha256=static_hash),manifest=dict(guard_stamp=stamp,origin_ids={o:i+1 for i,o in enumerate(registry['origins'])}),
  records=records,graph_lookup=guard.Lookup(records,registry['component_ordinals']),allowed={registry['origin_components'][data.OWNER]},forbidden=set(),expected_spec=None,admission_batch=None)
 # This is a pure validator envelope around retained v3 bytes, not a claim of v5 native execution.
 live=dict(generation=1601,claims_count=335,active_stage=0)
 receipt=dict(schema=stage.SCHEMA+'-guard-conformance',specification=engineering.specification(),model_queries=0,searches=0,rows=rows,
  lifecycle_before=dict(created=2,destroyed=0),lifecycle_after=dict(created=2,destroyed=0),fresh_native_guard_traces=5,reused_native_guard_traces=0,live_guard_before=live,live_guard_after=copy.deepcopy(live))
 return files,receipt,loaded
def test_actual_five_case_complete_bytes_validate_without_clearing_live_prefix():
 files,receipt,loaded=diagnostic_fixture();before={str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in files}
 assert engineering.validate(receipt,loaded)
 for row in receipt['rows']:
  assert [e[2] for e in guard.EVENT.iter_unpack(base64.b64decode(row['raw']))]==[1,31,1,32]
  assert type(engineering._claims(loaded,row['case'])) is guard.PackedClaims
 assert before=={str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in files}
 altered=copy.deepcopy(receipt);altered['live_guard_after']['generation']=0
 with pytest.raises(ValueError,match='known live prefix'):engineering.validate(altered,loaded)
 altered=copy.deepcopy(receipt);altered['fresh_native_guard_traces']=0;altered['reused_native_guard_traces']=5
 with pytest.raises(ValueError,match='known live prefix'):engineering.validate(altered,loaded)
def test_actual_decode_events_and_live_conflict_cannot_be_omitted():
 files,receipt,loaded=diagnostic_fixture();altered=copy.deepcopy(receipt)
 raw=base64.b64decode(altered['rows'][0]['raw']);altered['rows'][0]['raw']=base64.b64encode(raw[24:48]+raw[72:]).decode()
 with pytest.raises(ValueError,match='anchors'):engineering.validate(altered,loaded)
 altered=copy.deepcopy(receipt);altered['rows'][4]['violations']=''
 with pytest.raises(ValueError):engineering.validate(altered,loaded)
def test_actual_archived_history_compacts_only_memory_and_retains_refs_and_clocks():
 ownership={};events=[]
 for path in sorted((ARENA/'events').glob('*.json')):
  event=json.loads(path.read_text());events.append((path,event))
  if event['kind']=='ownership_check':ownership[event['decision_id']]=(path,event)
 selected=[];engines=set();overs=0
 for path in sorted((ARENA/'archives').glob('*.json')):
  body=json.loads(path.read_text())
  if body['engine'] not in engines or body['clock']['critical_ms']>25:
   engines.add(body['engine']);selected.append((path,body));overs+=body['clock']['critical_ms']>25
 assert len(engines)==4 and overs==4
 for path,body in selected:
  event_path,event=ownership[body['decision_id']];before=(path.read_bytes(),path.stat().st_mtime_ns,event_path.read_bytes(),event_path.stat().st_mtime_ns)
  pending=dict(body=body,result=event['result'],delta=b'not-retained',archive_ref=stage.ref(path),ownership_ref=stage.ref(event_path),annex_delta_ref=None)
  old=previous._thin_result(pending);new=execute._thin_result(pending)
  assert set(new['body'])=={'schema','identity_sha256','decision_id','query_kind','engine','origin','clock'}
  assert old['body']['clock']==new['body']['clock'] and old['result']['passed']==new['result']['passed']
  assert all(new[k]==pending[k] for k in ('archive_ref','ownership_ref','annex_delta_ref')) and 'delta' not in new
  assert before==(path.read_bytes(),path.stat().st_mtime_ns,event_path.read_bytes(),event_path.stat().st_mtime_ns)
 for path,event in events:
  compact=execute._thin_event(event)
  if event['kind'] in ('ownership_check','provider_epoch','component_annex'):
   assert compact==dict(kind=event['kind'],decision_id=event.get('decision_id'),validated_event_sha256=config.digest(event))
  else:assert compact is event
def test_source_admission_schedules_and_known_claim_seed_are_exact():
 assert native.source_identity()['unknown_high_default_deny'] is True
 required={f'congklak/prospective_benchmark_v5_{n}.py' for n in ('common','guard','execute','runtime','recovery','cli','worker','engineering','native','core','adapter')}
 assert required<=set(stage.REQUIRED_SOURCES)
 for operation in previous.OPERATIONS:assert execute.schedule(operation)==previous.schedule(operation)
 seed=execute._seed_for(stage.FOLDER/'jobs/engineering','engineering',False)
 assert seed['generation']==1601 and seed['known_high_claims']==335 and seed['old_claims_not_cleared'] and seed['operation']=='same_known_joint_benchmark_unit'
 assert (ROOT/seed['claims_ref']['path']).read_bytes()==(ARENA/'claims.bin').read_bytes()
 tree=ast.parse(Path(execute.__file__).read_text());replay=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_replay')
 text=ast.get_source_segment(Path(execute.__file__).read_text(),replay)
 assert text.index('_archive_check(')<text.index('results.append(_thin_result(pending))')<text.index('events[index]=_thin_event(event)')
