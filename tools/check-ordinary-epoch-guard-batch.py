"""Finite metadata-only batch equivalence: no native controls or queries."""
import argparse,copy,hashlib,json,struct,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,ordinary_epoch_common as s,ordinary_epoch_guard as g
from congklak.ordinary_epoch_query_contract import derive_registry,membership


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--version',default='v2',choices=('v2','v3'))
 args=parser.parse_args()
 started=time.perf_counter();registry,graph=derive_registry(plan_sha256='pure-batch-source-equivalence')
 folder=s.FOLDER/('native/guard-batch-pure-'+args.version);folder.mkdir(parents=True,exist_ok=False)
 graph_path=folder/'graph.json';graph_path.write_text(config.canonical(graph)+'\n')
 token=folder/'common-token.txt';token.write_bytes(b'fixed-common-input')
 origins=[r['origin'] for r in membership()['fit']]
 def spec(origin):
  return dict(guard=dict(graph_path=s.relative(graph_path),allowed_components=[registry['origin_components'][origin]],
   forbidden_quotients=[],epoch_owned_component_only=True,owner_origin=origin),
   schedule=[dict(origin=o) for o in registry['origins']],inputs=dict(inputs))
 inputs=s.common.pins([graph_path,token,s.JOINT]);batch=g.AdmissionBatch(spec(origins[0]))
 rows=sorted((bytes.fromhex(q),cid) for q,cid in graph['quotient_components'].items() if sum(bytes.fromhex(q)[:14])>=25)
 base=b''.join(g.STATIC.pack(q,g.UNPERMITTED_COMPONENT) for q,_ in rows)
 indices={}
 for i,(_,cid) in enumerate(rows):indices.setdefault(cid,[]).append(i)
 loop=time.perf_counter();record_bytes=0
 for i,origin in enumerate(origins):
  guard=spec(origin)['guard'];actual=batch.static(guard);expected=bytearray(base)
  for index in indices.get(guard['allowed_components'][0],[]):struct.pack_into('<I',expected,index*19+15,0)
  s.need(actual[0]==expected,'Per-origin full sorted record equality failed')
  if i in (0,len(origins)-1):
   legacy=g._static(guard,graph)
   s.need(all(a==b for a,b in zip(actual,legacy)),'Full legacy records/flags/shared metadata differed')
  record_bytes+=len(expected)
  del actual,expected
 batch.verify_current();loop_ms=(time.perf_counter()-loop)*1000
 # One normal persisted guard exercises the actual pure loaded-record reader.
 output=g.DEFAULT/('batch-equivalence-pure-'+args.version);refs=g.prepare(spec(origins[0]),output,batch=batch)
 loaded=g.load_pure(refs,spec(origins[0]),batch=batch)
 s.need(loaded.admission_batch is batch and loaded.graph_lookup==graph['quotient_components'],'Loaded graph map differs')
 refusals=[]
 def refuses(label,fn):
  try:fn()
  except (ValueError,TypeError,AttributeError):refusals.append(label);return
  raise ValueError('Expected refusal: '+label)
 bad=copy.deepcopy(spec(origins[0]));bad['guard']['allowed_components']=[registry['origin_components'][origins[1]]]
 refuses('wrong-origin-component',lambda:batch.static(bad['guard']))
 bad=copy.deepcopy(spec(origins[0]));bad['schedule'].pop()
 refuses('changed-origin-registry',lambda:batch.assert_spec(bad))
 refuses('mutable-rows',lambda:setattr(batch,'rows',()))
 refuses('mutable-input-map',lambda:batch.inputs.__setitem__('x','y'))
 forged=object.__new__(g.AdmissionBatch)
 for name,value in vars(batch).items():object.__setattr__(forged,name,value)
 refuses('constructor-bypass-static',lambda:forged.static(spec(origins[0])['guard']))
 refuses('constructor-bypass-load',lambda:g.load_pure(refs,spec(origins[0]),batch=forged))
 original_rows=batch.rows;object.__setattr__(batch,'rows',())
 refuses('altered-real-snapshot',lambda:batch.static(spec(origins[0])['guard']))
 object.__setattr__(batch,'rows',original_rows);batch.assert_sealed(require_closed=True)
 original=token.read_bytes();token.write_bytes(b'changed-common-input')
 refuses('changed-common-at-closure',batch.verify_current)
 refuses('changed-common-cold-restart',lambda:g.AdmissionBatch(spec(origins[0])))
 token.write_bytes(original);batch.verify_current()
 altered=dict(refs,records_sha256='0'*64)
 refuses('changed-origin-record-binding',lambda:g.load_pure(altered,spec(origins[0]),batch=batch))
 report=dict(schema=s.SCHEMA+'-guard-batch-pure-checks',passed=True,origins=len(origins),
  all_origin_record_bytes_checked=record_bytes,shared_graph_parse_count=1,
  full_legacy_origin_comparisons=2,all_high_quotients=len(rows),upper_quotients=sum(sum(q[:14])>=75 for q,_ in rows),
  finite128_origin_check_ms=loop_ms,total_elapsed_ms=(time.perf_counter()-started)*1000,
  refusal_cases=refusals,retained_origin_loaded_objects=1,native_controls=0,new_queries=0,
  source_inputs=s.common.pins([__file__,s.ROOT/'congklak/ordinary_epoch_guard.py',
   s.ROOT/'congklak/ordinary_epoch_execute.py',s.ROOT/'congklak/ordinary_epoch_query_contract.py',s.JOINT]),
  metadata_inputs=s.common.pins([graph_path,token]),guard_refs=refs)
 s.atomic(folder/'report.json',report);print(config.canonical(report))


if __name__=='__main__':main()
