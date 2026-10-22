"""Exhaustive pure field review and negative mutations of saved fixed oracles."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import copy,json,hashlib,base64,zlib
from congklak import native_ownership_trace as codec,trace_fast_comparison_v1 as cmp

OUT=ROOT/'build/tracing-speed-v1/native/comparator-review-v1';OUT.mkdir(exist_ok=False)
paths=[f'fixednodes-{name}.json' for name in ('H_v1','H8_relay_CPU')]
paths += [f'fixed-{key}-{name}.json' for name in ('control','candidate') for key in ('00','01','08')]
inputs={};rows=[];mutations=[]
def walk(value,path=()):
 if isinstance(value,dict):
  for key,v in value.items():yield from walk(v,path+(key,))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from walk(v,path+(str(i),))
 else:yield path,value
def mutate(body,path):
 target=body
 for key in path[:-1]:target=target[int(key)] if isinstance(target,list) else target[key]
 key=int(path[-1]) if isinstance(target,list) else path[-1];old=target[key]
 target[key]=(not old) if isinstance(old,bool) else old+1 if isinstance(old,(int,float)) else str(old)+'-changed'
for name in paths:
 path=ROOT/'build/ownership-implementation-v1/native/author-v2'/name;oracle=codec.read(path);inputs[path.relative_to(ROOT).as_posix()]=codec.sha(path)
 telemetry=[('/'+'/'.join(p),v) for p,v in walk(oracle['decision']) if any(k.endswith('_ms') or k=='peak_process_bytes' for k in p)]
 rows.append(dict(source=name,telemetry=telemetry))
 for p,v in walk(oracle['decision']):
  # Exercise EVERY scalar, not just selected convenient root fields.
  changed=copy.deepcopy(oracle);mutate(changed['decision'],p)
  expected=p in cmp.TELEMETRY_PATHS
  assert cmp.conform(changed,oracle)==expected,('/'.join(p),expected)
  mutations.append(dict(source=name,path='/'+'/'.join(p),expected_equal=expected))
 changed=copy.deepcopy(oracle);raw=bytearray(codec.trace_bytes(oracle['trace']));raw[16]^=1
 encoded=zlib.compress(raw,1);changed['trace'].update(data=base64.b64encode(encoded).decode(),compressed_bytes=len(encoded),event_sha256=hashlib.sha256(raw).hexdigest())
 assert not cmp.conform(changed,oracle);mutations.append(dict(source=name,path='raw/controller-resealed',expected_equal=False))
retained=[]
for directory in ('calibration-v1','calibration-v2'):
 for path in sorted((ROOT/'build/tracing-speed-v1/native'/directory/'attempts').glob('*.json')):
  saved=codec.read(path);oracle=codec.read(ROOT/saved['scheduled']['oracle']);assert cmp.conform(saved['answer'],oracle)
  retained.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=codec.sha(path),raw_exact=True,semantic_exact=True))
schema=ROOT/'build/tracing-speed-v1/native/semantic-comparator-v1.json'
schema.write_text(json.dumps(cmp.definition(),sort_keys=True,indent=2),encoding='utf-8')
inputs.update({p:codec.sha(ROOT/p) for p in ('congklak/trace_fast_comparison_v1.py','tools/audit-trace-comparator-v1.py')})
result=dict(schema='pure-exhaustive-fixed-work-comparison-review-v1',passed=True,new_searches=0,inputs=inputs,
 oracle_count=8,enumeration=rows,mutation_count=len(mutations),mutations=mutations,retained_attempts=retained,
 comparator=cmp.definition())
(OUT/'result.json').write_text(json.dumps(result,sort_keys=True),encoding='utf-8')
print(json.dumps(dict(passed=True,oracles=8,scalar_mutations=len(mutations),retained_actual_attempts=len(retained),new_searches=0)))
