"""Immutable identities for existing-data neurodiag diagnostics."""
from .divergence_study_common import ROOT,CONTROL,CANDIDATE,require,read,sha,pins,current,publish,rawkey
from . import config
BASELINE=ROOT/'build/neurodiag-baseline.json'
SCOPE=ROOT/'docs/NEURODIAG_SCOPE.md'
def same(a,b):return config.canonical(a)==config.canonical(b)
def baseline(verify_all=False):
    body=read(BASELINE)
    require(body['schema']=='neurodiag-prior-immutability-v1' and body['passed'] is True and type(body['inputs_count']) is int and body['inputs_count']==len(body['inputs'])==5377,'Neurodiag baseline differs')
    if verify_all:current(body['inputs'])
    return body
