"""Immutable identities for existing-data regret diagnostics."""
from .divergence_study_common import ROOT,CONTROL,CANDIDATE,require,read,sha,pins,current,publish,rawkey
from . import config
BASELINE=ROOT/'build/regret-baseline.json'
SCOPE=ROOT/'docs/REGRET_SCOPE.md'
def same(a,b):return config.canonical(a)==config.canonical(b)
def baseline(verify_all=False):
    body=read(BASELINE)
    require(body['schema']=='regret-prior-immutability-v1' and body['passed'] is True and type(body['inputs_count']) is int and body['inputs_count']==len(body['inputs'])==5252,'Regret baseline differs')
    if verify_all:current(body['inputs'])
    return body
