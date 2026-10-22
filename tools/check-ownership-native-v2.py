"""Bounded author engineering checks; no training, labels or production games."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from contextlib import ExitStack
import copy
import hashlib
import json
import time
from congklak import native_ownership_trace as trace,quality_search as old_search
from congklak.native import Native,flatten
from congklak.neural_relay import NeuralNative
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.value_sidecar_v2 import SidecarEvaluator

OUT=ROOT/'build/ownership-implementation-v1/native/author-v2'
OUT.mkdir(parents=True,exist_ok=True)
def publish(name,body):
 p=OUT/name;raw=(json.dumps(body,sort_keys=True,separators=(',',':'))+'\n').encode()
 if p.exists():raise RuntimeError('Retain prior author artifacts; choose a new version')
 p.write_bytes(raw)
def freeze():
 files=[Path(__file__),ROOT/'congklak/native_ownership_trace.py',ROOT/'congklak/quality_ownership_v1.py',
  ROOT/'tools/prepare-ownership-native-v1.py',ROOT/'tools/build-ownership-native-v1.cmd',ROOT/'tools/build-ownership-native-v1.ps1',
  *list((ROOT/'native/ownership_v1').glob('*')),trace.BUILD,trace.DLL,
  trace.DIRECTORY/'source-projection.json',trace.DIRECTORY/'python-projection.json']
 pins={}
 for p in files:
  if p.is_file():
   target=OUT/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():raise RuntimeError('Existing author source witness')
   target.write_bytes(p.read_bytes());pins[str(p.relative_to(ROOT)).replace('\\','/')]=trace.sha(p)
 publish('source-witness.json',pins)
 return pins
def physical(report):
 r={k:v for k,v in report.items() if k not in ('timing','work')}
 r['work']={k:v for k,v in report['work'].items() if k!='inference_ms'}
 return r
def baseline_view(r,name):
 return {'position':r['position'],'choice_depth':r['choice_depth'],'best_moves':r['best_moves'],
  'score':r['depth_result']['value_player0'] if name=='H_v1' else r['value_units'],
  'Q':[(x['pit'],x['depth_result']['value_player0'] if name=='H_v1' else x['value_units']) for x in r['candidates']],
  'pv':r['pv']}
sourcepins=freeze();rows=json.load(open(ROOT/'data/value-training-v1/cache/fit-check.json'))['rows']
selected={}
for row in rows:selected.setdefault(row['cell'],row)
states=[r['state'] for _,r in sorted(selected.items())]
low={'pits':[[1,2,3,0,1,2,3],[3,2,1,0,3,2,1]],'stores':[37,37],'controller':0}
assert sum(map(sum,low['pits']))==24
checks=[];counters={'fixed_PUCT_searches':0,'fixed_PUCT_backups':0,'fixed_depth_baseline_searches':0,
 'fixed_node_baseline_searches':0,'timed_searches':0,'failed_capacity_search_attempts':0,
 'optimizer_steps':0,'optimizer_draws':0,'new_games':0,'new_teacher_labels':0}
with ExitStack() as stack:
 p=trace.open_providers(stack,capacity=2000000)
 originals={'control':stack.enter_context(CpuBatchEvaluator(trace.CONTROL,cache_capacity=0,backend='avx2')),
  'candidate':stack.enter_context(SidecarEvaluator(trace.CANDIDATE,cache_capacity=0,backend='avx2'))}
 baselines={'H_v1':Native(),'H8_relay_CPU':stack.enter_context(NeuralNative(trace.RELAY,backend='avx2',mode='value-none'))}
 for index,state in enumerate(states+[low]):
  results={}
  for name in ('control','candidate'):
   identity=f'author-fixed-{index}-{name}'
   answer=p.fixed_decision(name,state,decision_id=identity);trace.validate_decision(answer,name,state,identity,p.identity)
   counters['fixed_PUCT_searches']+=1;counters['fixed_PUCT_backups']+=answer['decision']['completed_simulations']
   old=originals[name];old.begin_decision()
   reference=old_search.search(state,old,simulations=128,time_ms=0,max_nodes=2048,c_puct=.25,root_noise=None,memory_mb=4096)
   counters['fixed_PUCT_searches']+=1;counters['fixed_PUCT_backups']+=reference['completed_simulations']
   assert physical(answer['decision']['report'])==physical(reference),'Frozen PUCT treatment changed'
   publish(f'fixed-{index:02}-{name}.json',answer);results[name]=answer
   checks.append({'method':'frozen_PUCT_fixed128_conformance','state_index':index,'engine':name,
    'events':answer['trace']['event_count'],'whole_ms':answer['decision']['wall_ms'],'passed':True})
  if index==len(states):
   assert physical(results['control']['decision']['report'])==physical(results['candidate']['decision']['report'])
   assert results['candidate']['trace']['operation_counts']['sidecar_branch']==0
   checks.append({'method':'shared24_exact_control_candidate_fixed_work','passed':True})
  if index<2:
   for name,obj in baselines.items():
    identity=f'author-depth-{index}-{name}'
    answer=p.fixed_depth_decision(name,state,depth=4,decision_id=identity)
    trace.validate_decision(answer,name,state,identity,p.identity)
    if name=='H_v1':old=obj.analyze(flatten(state),actor=state['controller'],total=98,max_depth=4,time_ms=0,node_budget=0,memory_mb=4096,tt_mb=16,use_pvs=True,keep_history=False)
    else:old=obj.neural_analyze(state,max_depth=4,time_ms=0,node_budget=0,memory_mb=4096,tt_mb=16,use_pvs=True)
    counters['fixed_depth_baseline_searches']+=2
    assert baseline_view(answer['decision']['report'],name)==baseline_view(old['report'],name),'Frozen baseline scores/PV changed'
    publish(f'depth-{index:02}-{name}.json',answer)
    checks.append({'method':'frozen_baseline_depth4_Q_PV_conformance','state_index':index,'engine':name,
     'events':answer['trace']['event_count'],'old_neural_evaluations':old['report'].get('nn_evaluations'),
     'new_neural_evaluations':answer['decision']['report'].get('nn_evaluations'),'passed':True})
 for name in trace.ENGINES:
  identity='author-timed-'+name
  answer=p.timed_decision(name,states[0],ms=25,decision_id=identity)
  trace.validate_decision(answer,name,states[0],identity,p.identity)
  counters['timed_searches']+=1;publish('timed-'+name+'.json',answer)
  checks.append({'method':'whole_clock_trace_scope_physical_partition','engine':name,'events':answer['trace']['event_count'],
   'whole_ms':answer['decision']['wall_ms'],'overshoot_ms':answer['decision']['overshoot_ms'],'passed':True})
 for name in ('H_v1','H8_relay_CPU'):
  identity='author-fixednodes-'+name;answer=p.fixed_decision(name,states[0],decision_id=identity)
  trace.validate_decision(answer,name,states[0],identity,p.identity)
  counters['fixed_node_baseline_searches']+=1
  assert answer['decision']['nodes']<=128 and answer['work_budget']=={'kind':'decision_nodes','value':128}
  publish('fixednodes-'+name+'.json',answer)
  checks.append({'method':'bounded_distinct_baseline_node_units','engine':name,'passed':True})
 reset=p.reset_all(reason='handoff',decision_id='author-reset')
 assert reset['actual_controls_replaced']==2
 publish('reset.json',reset);checks.append({'method':'actual_both_inference_control_generation_reset','passed':True})
 # Mutate/reseal a trace partition without modifying its separate native coverage.
 bad=copy.deepcopy(results['control']);raw=trace.trace_bytes(bad['trace']);event=next(trace.iter_events(bad['trace']))
 bad['trace']['event_count']-=1;bad['trace']['attempted_events']-=1;bad['trace']['byte_count']-=24
 bad['trace']['operation_counts'][event['operation']]-=1
 import base64,zlib
 payload=raw[24:];compressed=zlib.compress(payload,1)
 bad['trace']['data']=base64.b64encode(compressed).decode();bad['trace']['compressed_bytes']=len(compressed)
 bad['trace']['event_sha256']=hashlib.sha256(payload).hexdigest()
 try:trace.validate_decision_coverage(bad)
 except ValueError:pass
 else:raise AssertionError('Dropped/resealed trace admitted')
 checks.append({'method':'drop_reseal_counter_partition_refused','passed':True})
with ExitStack() as stack:
 p=trace.open_providers(stack,capacity=8)
 try:p.fixed_decision('control',states[0],decision_id='author-capacity-failure')
 except trace.TraceHold as exc:
  counters['failed_capacity_search_attempts']+=1
  assert exc.trace['overflow'] and exc.trace['attempted_events']>exc.trace['event_count']
  publish('capacity-failure.json',{'error':str(exc),'trace':exc.trace})
 else:raise AssertionError('Trace exhaustion admitted')
 checks.append({'method':'capacity_exhaustion_preserved_HOLD','passed':True})
publish('result.json',{'schema':'ownership-native-author-checks-v1','passed':True,'checks':checks,'counters':counters,
 'inputs':sourcepins,'models_fixed':True,'policy_fixed':True,'production_admission':False})
print(json.dumps({'passed':True,'checks':len(checks),'counters':counters,'output':str(OUT/'result.json')}),flush=True)
