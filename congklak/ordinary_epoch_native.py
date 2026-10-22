"""Per-instance epoch exports with unchanged search and complete native traces.

The old provider is never patched globally. Each instance owns private Python
adapter modules, a source-bound candidate identity and the new capacity-only DLL.
Importing this module, reading its identity and validating archives load no DLL.
"""
from contextlib import ExitStack
from functools import lru_cache
import base64,hashlib,importlib.util,json,sys,time,uuid
from pathlib import Path
from . import config,ordinary_epoch_common as s,native_trace_fast_v2 as frozen

ROOT=s.ROOT
DIRECTORY=s.FOLDER/'native'
DLL=DIRECTORY/'congklak_ordinary_epoch_v1.dll'
BUILD=DIRECTORY/'build.json'
CORE=ROOT/'congklak/ordinary_epoch_fast_core.py'
ADAPTER=ROOT/'congklak/ordinary_epoch_trace_adapter.py'
FROZEN_CANDIDATE=ROOT/'data/value-training-v1/training/model.json'
FROZEN_SHA=frozen.MODEL_SHA['candidate']
POLICY=dict(schema='trace-finalization-debt-v1',reserve_ms=12.,minimum_search_ms=10,
 maximum_outer_ms=25,same_all_engines=True,calibrated=True)
TEACHER=dict(engine='H_v1',depth=8,node_budget=1000000,capacity_events=16000000,
 time_ms=0,tt_mb=16,use_pvs=True,memory_mb=4096)
SOURCES=('congklak/ordinary_epoch_native.py','congklak/ordinary_epoch_fast_core.py',
 'congklak/ordinary_epoch_trace_adapter.py','congklak/ordinary_epoch_guard.py',
 'congklak/ordinary_epoch_sidecar.py','congklak/ordinary_epoch_common.py')
need,read,sha=s.need,s.read,s.sha
TraceHold=frozen.TraceHold


def _same(a,b):return config.canonical(a)==config.canonical(b)

def _load(path,stem):
 name='congklak._ordinary_epoch_'+stem+'_'+uuid.uuid4().hex
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 spec.loader.exec_module(module);return module


def sidecar_model(path):
 """Loader selection is explicit; no accepted loader admission is relaxed."""
 path=Path(path).resolve()
 if path==FROZEN_CANDIDATE:
  need(sha(path)==FROZEN_SHA,'Frozen teacher-only candidate changed')
  from .value_sidecar_v2 import Model
 else:
  need(path.is_relative_to(s.DATA),'Epoch candidate lies outside the new data namespace')
  from .ordinary_epoch_sidecar import Model
 return Model(path)


def candidate_binding(path=None):
 path=FROZEN_CANDIDATE if path is None else Path(path).resolve()
 if path==FROZEN_CANDIDATE:
  need(sha(path)==FROZEN_SHA,'Frozen teacher-only candidate changed')
  return dict(mode='frozen-teacher-only',path=s.relative(path),sha256=FROZEN_SHA)
 need(path.is_relative_to(s.DATA),'Candidate path outside epoch namespace')
 body=read(path);m=body['model']
 need(set(body)=={'model','sha256'} and config.digest(m)==body['sha256'] and
  m['schema']=='congklak-ordinary-epoch-sidecar-v1' and m['bindings']['base_file_sha256']==frozen.MODEL_SHA['control'],
  'Candidate export schema/base/content differs')
 ref=m['bindings']['input_manifest'];manifest=Path(ref['path']).resolve()
 need(manifest.is_relative_to(s.DATA) and sha(manifest)==ref['sha256'],'Candidate epoch manifest differs')
 return dict(mode='ordinary-epoch-export',path=s.relative(path),sha256=sha(path),
  export_sha256=body['sha256'],input_manifest=ref,provenance=m['provenance'],
  epoch_id=m['bindings']['epoch_id'],plan_sha256=m['bindings']['plan_sha256'])


def _check_binding(binding):
 need(type(binding) is dict and _same(binding,candidate_binding(ROOT/binding['path'])),
  'Candidate instance binding changed')
 return binding


def input_pins():
 record=read(BUILD);paths=[*SOURCES,CORE,ADAPTER,BUILD,DLL,
  DIRECTORY/'source-projection.json',DIRECTORY/'adapter-projection-v2.json',DIRECTORY/'guard-projection-v3.json',
  ROOT/'tests/test_ordinary_epoch_native.py',ROOT/'tests/test_ordinary_epoch_execute.py',ROOT/'tools/check-ordinary-epoch-native.py',
  ROOT/'tools/check-ordinary-epoch-guard-batch.py']
 paths.extend(ROOT/p for p in record['inputs'])
 # The unchanged order witness is an executable validation dependency.
 from . import trace_puct_witness_v1 as witness
 proof=witness.build_identity();paths.extend([ROOT/'congklak/trace_puct_witness_v1.py',witness.DLL,witness.BUILD])
 paths.extend(ROOT/p for p in proof['sources'])
 return s.common.pins(paths)


def source_identity(binding=None):
 binding=candidate_binding() if binding is None else _check_binding(binding)
 record=read(BUILD)
 need(record['dll_sha256']==sha(DLL) and record['maximum_events']==16000000 and record['maximum_unique_owned']==4000000 and
  record['maximum_claims']==8000000 and record['claim_slots']==16777216,'Epoch native capacity/build differs')
 for name,h in record['inputs'].items():need(sha(ROOT/name)==h,'Epoch native source changed: '+name)
 projection=read(DIRECTORY/'source-projection.json')
 for name,entry in projection['copies'].items():
  a,b=ROOT/entry['original'],ROOT/entry['generated']
  need(sha(a)==entry['original_sha256'] and sha(b)==entry['generated_sha256'],'Epoch source projection changed')
  if name not in ('trace.cpp','trace.hpp'):need(a.read_bytes()==b.read_bytes(),'Search/model arithmetic changed')
 base=frozen.source_identity();models=dict(base['model_sha256']);models['candidate']=binding['sha256']
 return dict(base,model_sha256=models,candidate_binding=binding,module_sha256=sha(__file__),
  dll_sha256=sha(DLL),build_sha256=sha(BUILD),native_sources=record['inputs'],
  projected_adapter_sha256=sha(ADAPTER),projected_fast_core_sha256=sha(CORE),
  source_projection_sha256=sha(DIRECTORY/'source-projection.json'),
  adapter_projection_sha256=sha(DIRECTORY/'adapter-projection-v2.json'),
  maximum_events=16000000,maximum_unique_owned=4000000,maximum_claims=8000000,arena_maximum_events=2000000,
  frozen_policy=True,epoch_scope='ordinary-only inherited-background development',
  production_admission=False)


def _configured(binding):
 return _configured_cached(config.canonical(_check_binding(binding)))

@lru_cache(maxsize=8)
def _configured_cached(binding_json):
 binding=json.loads(binding_json);core=_load(CORE,'core');reader=_load(ADAPTER,'reader')
 core.TraceHold=TraceHold
 models=dict(frozen.MODEL_SHA);models['candidate']=binding['sha256']
 for module in (reader,):module.MODEL_SHA=dict(models);module.CANDIDATE=ROOT/binding['path']
 core.reader=reader
 core.source_identity=lambda:source_identity(binding)
 original=core._private_adapter
 def private(owner):
  module=original(owner);module.CANDIDATE=ROOT/binding['path'];module.MODEL_SHA=dict(models)
  return module
 core._private_adapter=private
 return core


class Providers:
 def __init__(self,stack,capacity=2000000,binding=None,policy=None):
  need(isinstance(stack,ExitStack),'Use ExitStack for model/control lifetime')
  need(binding is not None,'Every engine instance needs an explicit frozen teacher or epoch export binding')
  self.binding=_check_binding(binding)
  self.core=_configured(self.binding)
  self.provider=self.core.open_providers(stack,capacity,dict(POLICY if policy is None else policy))
  self.native=self.provider.native;self.identity=self.provider.identity
  need(_same(self.identity,source_identity(self.binding)),'Native instance identity differs')
 def attach_guard(self,compiled):return self.provider.attach_guard(compiled)
 def reset_all(self,reason='handoff',decision_id=None):return self.provider.reset_all(reason,decision_id)
 def decide(self,*args,**kwargs):return self.provider.decide(*args,**kwargs)
 def lifecycle(self):return self.provider.lifecycle()
 def teacher(self,state,decision_id,origin,attempt_sink=None):
  """One full-root H8 call; every cap produces retained evidence and HOLD."""
  p=self.provider;started=time.perf_counter();p._sink_path=Path(attempt_sink).resolve() if attempt_sink else None
  p._last_trace=None;p._attempt_begun=False;p._providers.last_failed_envelope=None;transient=None
  try:
   need(p.capacity==TEACHER['capacity_events'] and p.guard is not None,'Teacher capacity/guard absent')
   p.owner_id=p.compiled_guard.owner_id(origin)
   status=json.loads(p.native.lib.cc_fast_guard_status(p.guard.handle).decode())
   need(not status['active_stage'],'Prior query claims are not committed')
   reset=p.reset_all('every_teacher_query',decision_id)
   answer=p._providers._decision('H_v1',state,0,8,decision_id,baseline_depth=8,baseline_nodes=1000000)
   transient=p._transient(answer,'H_v1',state,decision_id,None)
   transient['both_seat_reset']=reset
   transient['timing']=dict(outer_elapsed_ms=(time.perf_counter()-started)*1000,requested_outer_ms=None,
    search_window_ms=0,finalization_ms=p._last_trace['_finalization_ms'],policy=TEACHER,archive_outside_clock=True)
   critical=dict(decision=answer['decision'],transition=answer['transition'],coverage=answer['coverage'],
    lifecycle=answer['lifecycle'],trace_sha256=transient['certificate']['event_sha256'],
    certificate=transient['certificate'],both_seat_reset=reset,binding=answer['trace']['binding'])
   transient['critical_sha256']=config.digest(critical)
   return transient
  except BaseException as exc:
   if p._last_trace is None and p._attempt_begun:
    try:p._snapshot(p.native,p.identity,'H_v1',decision_id,state,p.capacity)
    except BaseException:pass
   transient=p._transient(p._providers.last_failed_envelope,'H_v1',state,decision_id,str(exc))
   transient['timing']=dict(outer_elapsed_ms=(time.perf_counter()-started)*1000,failure=True,policy=TEACHER)
   raise self.core.TraceHold(str(exc),transient) from exc
  finally:p.native.lib.cc_ownership_trace_end()

 def probe(self,name,state,decision_id,origin,kind='prediction',attempt_sink=None):
  """One traced scalar prediction or encoder probe, with no search tree."""
  need(name in ('control','candidate') and kind in ('prediction','features'),'Unknown direct probe')
  p=self.provider;started=time.perf_counter();p.direct_probe=True
  p._sink_path=Path(attempt_sink).resolve() if attempt_sink else None
  p._last_trace=None;p._attempt_begun=False;context=None
  try:
   need(p.guard is not None,'Probe guard absent');p.owner_id=p.compiled_guard.owner_id(origin)
   status=json.loads(p.native.lib.cc_fast_guard_status(p.guard.handle).decode())
   need(not status['active_stage'],'Prior query claims are not committed')
   reset=p.reset_all('every_direct_probe',decision_id);life=p.lifecycle()
   need(p.native.lib.cc_ownership_trace_begin(p.capacity,4),'Cannot begin complete probe trace')
   p.native.trace_mark(state,'played_input');obj=p._providers.engines[name][1]
   before=obj.snapshot_counters();obj.begin_decision()
   value=obj.feature_vector(state) if kind=='features' else obj.evaluate_many([state])[0]
   after=obj.snapshot_counters()
   p.native.trace_mark(state,'played_output')
   trace=p._snapshot(p.native,p.identity,name,decision_id,state,p.capacity)
   self.core._native_summary(trace,p.identity,decision_id)
   answer=dict(probe=dict(schema='ordinary-epoch-direct-probe-v1',kind=kind,engine=name,state=state,
    result=value,counters_before=before,counters_after=after),trace=trace,
    lifecycle=dict(before=life,after=p.lifecycle()))
   _probe_coverage(answer)
   context=p._transient(answer,name,state,decision_id,None);context['both_seat_reset']=reset
   context['timing']=dict(outer_elapsed_ms=(time.perf_counter()-started)*1000,requested_outer_ms=None,
    finalization_ms=p._last_trace['_finalization_ms'],archive_outside_clock=True)
   context['critical_sha256']=config.digest(dict(probe=answer['probe'],lifecycle=answer['lifecycle'],
    certificate=context['certificate'],binding=trace['binding'],both_seat_reset=reset))
   return context
  except BaseException as exc:
   if p._last_trace is None and p._attempt_begun:
    try:p._snapshot(p.native,p.identity,name,decision_id,state,p.capacity)
    except BaseException:pass
   context=p._transient(None,name,state,decision_id,str(exc))
   context['timing']=dict(outer_elapsed_ms=(time.perf_counter()-started)*1000,failure=True)
   raise self.core.TraceHold(str(exc),context) from exc
  finally:p.native.lib.cc_ownership_trace_end();p.direct_probe=False


def open_providers(stack,capacity=2000000,binding=None,policy=None):return Providers(stack,capacity,binding,policy)
def archive_decision(transient):
 from . import ordinary_epoch_fast_core as codec
 return codec.archive_decision(transient)
archive_attempt=archive_decision
def validate_decision(answer,engine,state,decision_id,identity=None):
 bound=answer['trace']['binding']['provider_identity']
 need(identity is None or _same(bound,identity),'Saved candidate/provider identity differs')
 core=_configured(bound['candidate_binding'])
 need(_same(bound,source_identity(bound['candidate_binding'])),'Saved native identity no longer source-bound')
 return core.validate_decision(answer,engine,state,decision_id,bound)

def _probe_coverage(answer):
 probe=answer['probe'];trace=answer['trace'];cert=trace['fast_certificate'];ops=trace['operation_counts']
 need(cert['valid'] is True and cert['guard_passed'] is True and not cert['overflow'] and not cert['invalid'],
  'Direct probe trace or ownership is incomplete')
 need(ops['played_input']==ops['played_output']==1 and
  not any(ops[x] for x in ('puct_node','puct_select','puct_expand','alpha_node','relay_node','alpha_tt_probe','relay_tt_probe')),
  'Direct probe contains search work or missing anchors')
 before,after=probe['counters_before'],probe['counters_after'];value=probe['result']
 terminal=not sum(map(sum,probe['state']['pits']))
 expected=0 if probe['kind']=='features' or terminal else 1
 need(ops['model_done']+ops['sidecar_done']==after['native_actual_evals']-before['native_actual_evals']==expected,
  'Direct probe physical model completion differs')
 need(ops['feature_move_in']==ops['feature_move_out']==value['feature_simulations'] and
  ops['move_in']==ops['move_out']==ops['feature_move_in'],'Direct probe feature endpoints differ')
 if probe['kind']=='features':need(len(value['features'])==127 and ops['feature_input']==1,'Feature probe width/input differs')
 else:
  need(len(value['policy_logits'])==7 and value['native_actual_evals']==expected,'Prediction probe policy/forward differs')
  if probe['engine']=='control':need(ops['sidecar_eval']==ops['sidecar_done']==0,'Control probe consulted sidecar')
 life=answer['lifecycle']
 need(life['after']['created']-life['before']['created']==1 and
  life['after']['destroyed']-life['before']['destroyed']==1,'Probe inference control was not cold')
 return True

def validate_probe(answer,engine,state,decision_id,identity=None):
 from . import ordinary_epoch_trace_adapter as codec
 bound=answer['trace']['binding']['provider_identity'];binding=answer['trace']['binding']
 need(identity is None or _same(bound,identity),'Probe provider identity differs')
 need(_same(bound,source_identity(bound['candidate_binding'])) and binding['engine']==engine and
  binding['decision_id']==decision_id and _same(binding['root_state'],state),'Probe source/root/cursor differs')
 summary=codec.validate_trace(answer['trace'],bound,decision_id);_probe_coverage(answer)
 for key in ('played_inputs','played_outputs'):
  need(len(summary[key])==1 and _same(summary[key][0]['state'],state) and summary[key][0]['action']==-1,
   'Probe input/output raw anchor differs')
 cert=answer['fast_certificate'];need(_same(cert,answer['trace']['fast_certificate']) and
  answer.get('fast_error') is None and not answer.get('archival_errors'),'Probe archive incomplete')
 raw=trace_bytes(answer['trace']);owned=base64.b64decode(answer['fast_owned_high'],validate=True)
 need(hashlib.sha256(raw).hexdigest()==cert['event_sha256'] and hashlib.sha256(owned).hexdigest()==cert['owned_sha256'],
  'Probe captured bytes differ')
 _validate_owned(answer['trace'],owned)
 reset=answer['both_seat_reset']
 need(reset['decision_id']==decision_id and reset['source_identity_sha256']==config.digest(bound) and
  reset['actual_controls_replaced']==2,'Probe both-seat reset differs')
 expected=config.digest(dict(probe=answer['probe'],lifecycle=answer['lifecycle'],certificate=cert,binding=binding,both_seat_reset=reset))
 need(answer['fast_critical_sha256']==expected,'Probe critical certificate differs')
 return dict(passed=True,event_count=cert['event_count'],owned_count=cert['owned_count'])

def _validate_owned(trace,owned):
 """Independent complete raw scan; a resealed compact map cannot omit states."""
 from .ordinary_epoch_trace_adapter import PACK
 from .ordinary_epoch_guard import raw_key,OWNED
 expected={}
 for index,(cells,actor,*_) in enumerate(PACK.iter_unpack(trace_bytes(trace))):
  if 98-cells[7]-cells[15]>=25:
   q=raw_key(cells,actor);expected.setdefault(q,OWNED.pack(q,index,cells,actor))
 need(owned==b''.join(v for _,v in sorted(expected.items())),'Probe full owned footprint differs')
 return True

def teacher_certificate(answer,state):
 """No capped/partial action may enter the one finite-choice H8 target."""
 from .value_run_arena import normalize
 normal=normalize(state);need(_same(normal['state'],state) and not normal['terminal'],'Teacher root is not normalized nonterminal')
 decision=answer['decision'];report=decision['report'];legal=normal['legal'];actor=state['controller']
 need(decision['stop_reason']=='max_depth' and report is not None and report['choice_depth']==8 and
  report['completed_iteration'] is True and report['proof_status']=='finite_horizon_result' and
  report['evaluator']=='H_v1' and report['depth_result']['evaluator']=='H_v1' and
  report['depth_result']['choice_depth']==8 and report['depth_result']['bound']=='exact' and
  report['depth_result']['scope']=='finite_choice_horizon' and report['position']==state and
  report['input_controller']==report['normalized_controller']==actor and report['score_player']==actor and
  report['legal_actions']==legal and report['terminal'] is False,'Required H8 root target is incomplete or changed')
 need(report['policy']==report['depth_result']['policy']=='terminal-first/no-oracle/full-root-window-v2' and
  report['search_options']['tt_mb']==16 and report['search_options']['use_pvs'] is True,'Teacher recurrence/configuration differs')
 q={}
 for item in report['candidates']:
  depth=item['depth_result'];pit=item['pit'];value=depth['value_player0']
  need(type(pit) is int and pit not in q and pit in legal and type(value) is int and
   item['analysis_complete'] is True and item['actor']==actor and depth['choice_depth']==8 and
   depth['bound']=='exact' and depth['evaluator']=='H_v1' and depth['scope']=='finite_choice_horizon' and
   depth['policy']==report['policy'] and depth['score_player']==actor and
   depth['value_actor']==value*(1 if actor==0 else -1),'A root action has incomplete/mixed H8 Q')
  q[pit]=value
 need(set(q)==set(legal),'Teacher omits a legal action')
 target=(max if actor==0 else min)(q.values());best=[pit for pit in legal if q[pit]==target]
 need(report['depth_result']['value_player0']==target and report['best_moves']==best,'Teacher minimax value/ties differ')
 return dict(requested_depth=8,completed_depth=8,all_legal_Q_complete=True,action_q=[dict(pit=p,value_player0=q[p]) for p in legal],
  best_moves=best,target_player0=target,score_perspective='fixed player0',input_controller=actor,normalized_controller=actor,
  stop_reason=decision['stop_reason'],nodes=decision['nodes'],events=answer['trace']['event_count'],
  resource_envelope=dict(historical_memory_mb=512,memory_mb=4096,node_cap=1000000,trace_event_cap=16000000,
   unique_owned_cap=4000000,global_claim_cap=8000000),target_is_solved_truth=False)
def trace_bytes(trace):
 from .ordinary_epoch_trace_adapter import trace_bytes as decode
 return decode(trace)
def iter_trace_states(trace):
 from .ordinary_epoch_trace_adapter import iter_trace_states as iterate
 yield from iterate(trace)
def read_attempt_sink(path):
 from .ordinary_epoch_fast_core import read_attempt_sink as decode
 return decode(path)
