"""Independent preserved-history facts for the tracing-speed implementation."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'build/tracing-speed-v1/audit'
PRIOR=ROOT/'build/ownership-implementation-v1/acceptance.json'
MODELS={'data/quality-train-v1/model.json':'bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558',
 'data/value-training-v1/training/model.json':'c2a9a25c9cc5d3b70c664863aace9db3c15e02baab022d7a4229ed6dfb91b319',
 'data/training-stage5-deeper-h8-f127-seed2718/model.json':'4796c375398a155fb5edfcfdf1be2442734b7ff1563a32a81d7a131a74103b6b'}
def need(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def publish(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);need(not path.exists(),'Independent output already exists')
    with path.open('xb') as stream:stream.write((json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
def facts():
    prior=read(PRIOR);need(prior['schema']=='ownership-implementation-acceptance-v1' and prior['passed'] is True and
        prior['training_authorized'] is False and prior['execution_ready'] is False and len(prior['inputs'])==62199,'Prior accepted scope differs')
    inputs=dict(prior['inputs']);inputs[PRIOR.relative_to(ROOT).as_posix()]=sha(PRIOR)
    for name,digest in inputs.items():need(sha(ROOT/name)==digest,'Accepted history changed: '+name)
    for name,digest in MODELS.items():need(sha(ROOT/name)==digest,'Frozen model changed: '+name)
    control=read(ROOT/'data/quality-train-v1/model.json');candidate=read(ROOT/'data/value-training-v1/training/model.json')
    need(candidate['model']['base_export']==control,'Base model/policy export differs')
    need('torch' not in sys.modules,'Facts imported Torch')
    sources={p:sha(ROOT/p) for p in [*MODELS,PRIOR.relative_to(ROOT).as_posix(),'tests/tracing_speed_independent_audit.py','tests/reference.py']}
    publish(OUT/'facts.json',dict(schema='tracing-speed-independent-facts-v1',passed=True,prior_files_preserved=len(inputs),
        prior_acceptance_sha256=sha(PRIOR),source_inputs=sources,models_fixed=True,policy_fixed=True,
        searches=0,model_forwards=0,new_games=0,optimizer_steps=0,optimizer_draws=0,new_teacher_labels=0))
    print(json.dumps(dict(passed=True,prior_files_preserved=len(inputs),models_fixed=True,policy_fixed=True)),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['facts']);args=parser.parse_args();facts()
