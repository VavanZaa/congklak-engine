"""Source/prefix/mutation audit with saved bytes only. Never admit old failures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ast,copy,json,hashlib,re,time,base64,zlib
from congklak import trace_puct_witness_v1 as w,trace_fast_adapter_v2 as reader,native_ownership_trace as old

OUT=ROOT/'build/tracing-speed-v1/native/deadline-coverage-v1';OUT.mkdir(exist_ok=False)
helper=w.NativeWitness();sources=['congklak/quality_ownership_v1.py','congklak/puct_cpu_inference.py',
 'native/ownership_v2/neural_puct_cpu.cpp','native/ownership_v2/neural_value_sidecar.cpp',
 'congklak/trace_puct_witness_v1.py','congklak/trace_fast_adapter_v2.py','congklak/native_trace_fast_v2.py',
 'tools/audit-puct-deadline-coverage-v1.py']
inputs={p:old.sha(ROOT/p) for p in sources};polls=[]
for name in sources[:2]:
 text=(ROOT/name).read_text();tree=ast.parse(text)
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_poll':
   polls.append(dict(source=name,line=node.lineno,source_text=text.splitlines()[node.lineno-1].strip()))
for name in sources[2:4]:
 for i,line in enumerate((ROOT/name).read_text().splitlines(),1):
  if 'c.poll()' in line:polls.append(dict(source=name,line=i,source_text=line.strip()))

def prepare(archive):
 answer=copy.deepcopy(archive);trace=answer['trace'];raw=old.trace_bytes(trace)
 witness=helper.compute(raw);assert witness==w.pure(raw,helper.identity['dll_sha256'])
 gap=trace['operation_counts']['model_done']+trace['operation_counts']['sidecar_done']-trace['operation_counts']['puct_expand']
 trace['binding']['provider_identity']['puct_witness']=helper.identity
 trace['fast_certificate']=dict(trace.get('fast_certificate',{}),puct_order_witness=witness if gap==1 else None)
 answer['coverage']=dict(native_operation_counters=trace['operation_counts'],native_event_count=trace['event_count'])
 return answer

rows=[]
for folder in ('calibration-v1','calibration-v2','calibration-v3'):
 for path in sorted((ROOT/'build/tracing-speed-v1/native'/folder/'attempts').glob('*.json')):
  saved=old.read(path);answer=prepare(saved['answer']);summary=reader.validate_decision_coverage(answer)
  rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=old.sha(path),
   diagnostic_only=True,historical_status_unchanged=True,coverage=summary))

edge=old.read(ROOT/'build/tracing-speed-v1/native/calibration-v3/attempts/000019.json')['answer'];draft=prepare(edge)
assert reader.validate_decision_coverage(draft)['timed_completed_evaluation_not_expanded']==1
report=draft['decision']['report'];ops=draft['trace']['operation_counts'];trace=draft['trace']
cases=[]
def refuse(name,change):
 mutated=copy.deepcopy(draft);change(mutated)
 try:reader.validate_decision_coverage(mutated)
 except (ValueError,KeyError,TypeError):cases.append(dict(name=name,refused=True));return
 raise AssertionError('Invalid coverage admitted: '+name)
for stop in ('simulations','node_budget','cancelled','memory_budget'):
 refuse('stop_'+stop,lambda x,s=stop:x['decision']['report'].__setitem__('stop_reason',s))
refuse('forest2',lambda x:x['decision']['report'].__setitem__('forest_size',2))
refuse('not_root_expanded',lambda x:x['decision']['report'].__setitem__('root_expanded',False))
refuse('not_usable',lambda x:x['decision']['report'].__setitem__('usable_decision',False))
refuse('no_completed_backup',lambda x:x['decision']['report'].__setitem__('completed_simulations',0))
refuse('unlimited_budget',lambda x:x['decision']['report']['timing'].__setitem__('requested_time_ms',0))
refuse('missing2',lambda x:x['decision']['report']['work'].__setitem__('neural_evaluations',105))
refuse('forward_report_mismatch',lambda x:x['decision']['report']['work'].__setitem__('neural_calls',105))
refuse('physical_forward_mismatch',lambda x:x['evaluator']['delta'].__setitem__('native_actual_evals',105))
refuse('physical_feature_mismatch',lambda x:x['evaluator']['delta'].__setitem__('feature_simulations',793))
def order_change(x,key,value):x['trace']['fast_certificate']['puct_order_witness'][key]=value
for key,value in [('last_done_index',8700),('last_expand_index',8781),('last_expand_index',None),
 ('preceding_node_index',8781),('model_done',105),('puct_expand',102),('puct_nodes',106),
 ('event_sha256','0'*64),('validator_sha256','0'*64),('complete_input',False)]:
 refuse('witness_'+key+'_'+str(value),lambda x,k=key,v=value:order_change(x,k,v))
refuse('node_actor_mismatch',lambda x:x['trace']['fast_certificate']['puct_order_witness']['preceding_node_state'].__setitem__('controller',0))
raw=old.trace_bytes(edge['trace']);native=helper.compute(raw);reordered=bytearray(raw)
a=native['last_done_index']*24;b=native['last_expand_index']*24
reordered[a:a+24],reordered[b:b+24]=reordered[b:b+24],reordered[a:a+24]
for name,altered in [('done_before_last_expansion',bytes(reordered)),
 ('bad_mass',bytes([raw[0]+1])+raw[1:]),('bad_reserved',raw[:19]+b'\x01'+raw[20:])]:
 if name!='done_before_last_expansion':
  try:helper.compute(altered)
  except ValueError:cases.append(dict(name=name,refused=True));continue
  raise AssertionError(name)
 witness=helper.compute(altered);assert witness==w.pure(altered,helper.identity['dll_sha256'])
 mutated=copy.deepcopy(draft);compressed=zlib.compress(altered,1);mutated['trace'].update(
  data=base64.b64encode(compressed).decode(),compressed_bytes=len(compressed),event_sha256=hashlib.sha256(altered).hexdigest())
 mutated['trace']['fast_certificate']['puct_order_witness']=witness
 try:reader.validate_decision_coverage(mutated)
 except ValueError:cases.append(dict(name=name,refused=True))
 else:raise AssertionError(name)
result=dict(schema='pure-puct-deadline-boundary-coverage-audit-v1',passed=True,new_searches=0,
 helper_identity=helper.identity,inputs=inputs,deadline_poll_sites=polls,retained_attempts=rows,negative_cases=cases,
 real_saved_edge_witness=native,historical_failed_attempt_admitted=False,
 source_cases=[
  'Poll before evaluation: no completed forward/expansion; exact raw prefix remains guarded',
  'Native move/feature interruption: at most one unmatched complete input, existing strict feature/move partitions retained',
  'Native/Python post-forward poll: completed physical forward charged through stopped-batch per_position, one witnessed unexpanded leaf',
  'Forest post-charge poll: validated completed batch accounted before expansion, same bounded witnessed single-tree case',
  'After expansion/before next selection: gap0 and all existing strict equalities apply',
  'Fixed simulations, node/memory budget and cancellation receive no new missing-expansion allowance'],
 scope='Diagnostics over preserved source/raw bytes; no old result or failed attempt is reclassified')
(OUT/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2),encoding='utf-8')
print(json.dumps(dict(passed=True,retained=len(rows),negative_cases=len(cases),poll_sites=len(polls),new_searches=0)))
