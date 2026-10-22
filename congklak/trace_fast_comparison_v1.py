"""Exact fixed-work semantics, excluding only ten enumerated telemetry paths."""
from . import native_ownership_trace as codec

TELEMETRY_PATHS=(
 ('engine_elapsed_ms',),('overshoot_ms',),('requested_ms',),('wall_ms',),
 ('report','elapsed_ms'),('report','peak_process_bytes'),
 ('report','timing','elapsed_ms'),('report','timing','overshoot_ms'),
 ('report','timing','requested_time_ms'),('report','work','inference_ms'))

def definition():
 return dict(schema='exact-fixed-work-semantic-comparator-v1',
  ignored_telemetry_paths=['/'+ '/'.join(p) for p in TELEMETRY_PATHS],
  remaining_fields='All remaining fields and values are compared exactly, including extra fields',
  full_lossless_raw_events='Byte equality required, including controller/op/scope/action/depth',
  configuration_preserved=['TT allocation','whole_decision_includes_inference','nodes','visits','root Q values','PV','policy','stop status','work counts'],
  telemetry_retained_in_archives=True,new_searches=0)

def semantic_decision(value,path=()):
 if isinstance(value,list):return [semantic_decision(x,path+(str(i),)) for i,x in enumerate(value)]
 if isinstance(value,dict):return {k:semantic_decision(v,path+(k,)) for k,v in value.items() if path+(k,) not in TELEMETRY_PATHS}
 return value

def conform(actual,expected):
 return codec.trace_bytes(actual['trace'])==codec.trace_bytes(expected['trace']) and \
  semantic_decision(actual['decision'])==semantic_decision(expected['decision'])
