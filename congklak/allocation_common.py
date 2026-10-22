"""Shared immutable identities for the allocation and separate weighting study."""
from .divergence_study_common import ROOT,CONTROL,CANDIDATE,require,read,sha,pins,current,publish,rawkey
from . import config

BASELINE=ROOT/'build/allocation-baseline.json'
SCOPE=ROOT/'docs/ALLOCATION_SCOPE.md'


def same(a,b):return config.canonical(a)==config.canonical(b)


def baseline(verify_all=False):
    body=read(BASELINE)
    require(body['schema']=='allocation-prior-immutability-v1' and body['passed'] is True and
            type(body['inputs_count']) is int and body['inputs_count']==len(body['inputs'])==4255,
            'Allocation baseline differs')
    if verify_all:current(body['inputs'])
    return body
