"""Stateless ordering proof over owned bytes; engine/search DLL stays unchanged."""
import ctypes as C
import hashlib,json,struct
from pathlib import Path
from . import native_ownership_trace as codec
ROOT=codec.ROOT
BUILD=ROOT/'build/tracing-speed-v1/native/witness-build-v1.json'
DLL=ROOT/'build/tracing-speed-v1/native/puct_witness_v1.dll'
MISSING=2**64-1

def build_identity():
 r=codec.read(BUILD)
 codec.need(r['schema']=='immutable-packed24-order-witness-build-v1' and r['dll_sha256']==codec.sha(DLL) and
  r['engine_dll_changed'] is False and r['search_calls']==0 and
  r['flags']==['/std:c++20','/EHsc','/O2','/fp:strict','/W4','/LD'],'Witness helper identity differs')
 for path,h in r['inputs'].items():codec.need(codec.sha(ROOT/path)==h,'Witness helper source differs')
 return dict(build_sha256=codec.sha(BUILD),dll_sha256=r['dll_sha256'],sources=r['inputs'],
  python_module_sha256=codec.sha(__file__))

def _result(values,states,raw,validator):
 def state(blob):return dict(ring_hex=blob[:16].hex(),controller=blob[16])
 return dict(schema='immutable-packed24-puct-order-witness-v1',event_count=values[0],
  model_done=values[1],sidecar_done=values[2],puct_expand=values[3],puct_nodes=values[7],
  last_done_index=None if values[4]==MISSING else values[4],
  last_expand_index=None if values[5]==MISSING else values[5],
  preceding_node_index=None if values[6]==MISSING else values[6],
  last_done_state=None if values[4]==MISSING else state(states[:17]),
  preceding_node_state=None if values[6]==MISSING else state(states[17:]),
  event_sha256=hashlib.sha256(raw).hexdigest(),validator_sha256=validator,complete_input=True)

class NativeWitness:
 def __init__(self):
  self.identity=build_identity();self.lib=C.CDLL(str(DLL))
  self.lib.cc_puct_witness_abi.argtypes=[];self.lib.cc_puct_witness_abi.restype=C.c_char_p
  codec.need(self.lib.cc_puct_witness_abi()==b'immutable-packed24/puct-order-witness-v1','Witness ABI differs')
  self.lib.cc_puct_witness.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64),C.POINTER(C.c_uint8)]
  self.lib.cc_puct_witness.restype=C.c_int
 def compute(self,raw):
  codec.need(type(raw) is bytes and len(raw)%24==0,'Witness requires immutable complete packed bytes')
  values=(C.c_uint64*8)();states=(C.c_uint8*34)()
  codec.need(self.lib.cc_puct_witness(C.c_char_p(raw),len(raw)//24,values,states)==1,'Witness raw validity failed: HOLD')
  return _result(list(values),bytes(states),raw,self.identity['dll_sha256'])

def pure(raw,validator):
 codec.need(type(raw) is bytes and len(raw)%24==0 and 1<=len(raw)//24<=2000000,'Witness raw geometry differs')
 values=[0]*8;values[4:7]=[MISSING]*3;states=bytes(34);node=MISSING;node_state=bytes(17)
 for i,(cells,actor,op,scope,reserved,action,depth) in enumerate(codec.PACK.iter_unpack(raw)):
  codec.need(sum(cells)==98 and actor in (0,1) and 1<=op<=37 and 1<=scope<=4 and reserved==0 and action>=-1 and depth>=-1,'Invalid witness raw event')
  if op==28:node=i;node_state=cells+bytes([actor]);values[7]+=1
  if op==30:values[3]+=1;values[5]=i
  if op in (35,36):values[1 if op==35 else 2]+=1;values[4]=i;values[6]=node;states=cells+bytes([actor])+node_state
 values[0]=len(raw)//24
 return _result(values,states,raw,validator)

def pending_allowed(report,ops,trace):
 completed=ops['model_done']+ops['sidecar_done']
 codec.need(report['work']['neural_evaluations']==completed,
  'PUCT report/physical completed-forward accounting differs')
 gap=completed-ops['puct_expand']
 if gap==0:return 0
 w=trace['fast_certificate'].get('puct_order_witness')
 codec.need(gap==1 and report['forest_size']==1 and report['stop_reason']=='time_budget' and
  report['root_expanded'] is True and report['usable_decision'] is True and report['completed_simulations']>0 and
  report['timing']['requested_time_ms']>0 and report['work']['neural_calls']-ops['puct_expand']==1 and
  ops['model_done']+ops['sidecar_done']==report['work']['neural_evaluations'] and
  isinstance(w,dict) and w['complete_input'] is True and w['event_count']==trace['event_count'] and
  w['validator_sha256']==trace['binding']['provider_identity']['puct_witness']['dll_sha256'] and
  w['event_sha256']==trace['event_sha256'] and w['model_done']==ops['model_done'] and
  w['sidecar_done']==ops['sidecar_done'] and w['puct_expand']==ops['puct_expand'] and
  w['puct_nodes']==ops['puct_node'] and type(w['last_done_index']) is int and
  type(w['last_expand_index']) is int and type(w['preceding_node_index']) is int and
  0<=w['last_expand_index']<w['last_done_index']<trace['event_count'] and
  0<=w['preceding_node_index']<w['last_done_index'] and w['last_done_state']==w['preceding_node_state'],
  'PUCT expansion footprint differs: no bounded final completed-evaluation witness')
 return 1
