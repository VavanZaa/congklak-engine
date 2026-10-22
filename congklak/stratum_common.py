"""Immutable identities for the within-stratum-and-actor weighting study."""
from .divergence_study_common import ROOT,CONTROL,CANDIDATE,require,read,sha,pins,current,publish,rawkey
from . import config
BASELINE=ROOT/'build/stratum-baseline.json'
SCOPE=ROOT/'docs/STRATUM_SCOPE.md'
def same(a,b):return config.canonical(a)==config.canonical(b)
def baseline(verify_all=False):
    body=read(BASELINE)
    require(body['schema']=='stratum-prior-immutability-v1' and body['passed'] is True and type(body['inputs_count']) is int and body['inputs_count']==len(body['inputs'])==4546,'Stratum baseline differs')
    if verify_all:current(body['inputs'])
    return body
