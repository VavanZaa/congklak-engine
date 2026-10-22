"""One constructor-verified literal rules module per finite worker session."""
import hashlib,importlib.util,sys,types,uuid,weakref
from pathlib import Path
from . import config,value_run_arena as frozen

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'tests/reference.py'
_SEALS=weakref.WeakKeyDictionary()

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class LiteralRules:
    """Cache code only; every state/legal/transition check still runs in-clock."""
    def __setattr__(self,name,value):
        need(not getattr(self,'_sealed',False),'Literal rules session is immutable')
        object.__setattr__(self,name,value)
    def __init__(self,expected_sha256):
        need(type(expected_sha256) is str and sha(REFERENCE)==expected_sha256,'Literal source differs from the reviewed binding')
        name='congklak._benchmark_v2_literal_'+uuid.uuid4().hex
        spec=importlib.util.spec_from_file_location(name,REFERENCE)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
        functions=tuple(sorted((k,v) for k,v in vars(module).items() if isinstance(v,(types.FunctionType,type)) and getattr(v,'__module__',None)==name))
        need({'Position','normalize','legal_actions','apply'}<={k for k,v in functions},'Literal interpreter API incomplete')
        self._reference=module;self._functions=functions;self._sha=expected_sha256;self._closed=False;self._sealed=True
        _SEALS[self]=(module,functions,expected_sha256,False,frozen.normalize,frozen.transition)
        self.verify()
    def verify(self,closing=False):
        seal=_SEALS.get(self)
        need(type(self) is LiteralRules and seal is not None and self._reference is seal[0] and
             self._functions is seal[1] and self._sha==seal[2] and self._closed is seal[3] and not self._closed,
             'Literal rules require a live constructor-issued source seal')
        need(all(vars(self._reference).get(k) is v for k,v in self._functions) and
             frozen.normalize is seal[4] and frozen.transition is seal[5], 'Literal rules module/functions mutated')
        if closing:need(sha(REFERENCE)==self._sha,'Literal source changed before worker closure')
        return True
    def normalize(self,state):
        self.verify();return frozen.normalize(state,reference=self._reference)
    def transition(self,state,action):
        self.verify();return frozen.transition(state,action,reference=self._reference)
    def close(self):
        self.verify(closing=True);object.__setattr__(self,'_closed',True)
        seal=_SEALS[self];_SEALS[self]=(*seal[:3],True,*seal[4:])
        return dict(schema='prospective-benchmark-v2-literal-source-closure',reference_sha256=self._sha,
                    module_constructions=1,source_and_function_identity_verified=True,model_queries=0)
