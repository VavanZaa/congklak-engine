"""Isolated completed-path proof for the prospective Q-stability study."""
from __future__ import annotations
from contextlib import ExitStack
import argparse
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in the Q-stability completed reader')
        return None


sys.meta_path.insert(0,NoTorch())


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def name(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def durable(path):
    return path.is_file() and path.suffix!='.pyc' and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)
def files(path):return {p.resolve() for p in Path(path).rglob('*') if durable(p)} if Path(path).exists() else set()
def snapshot(paths):
    return {name(p):dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)}
def forbidden(*a,**k):raise AssertionError('Completed Q-stability path attempted new search/inference/training')


def closed(spec):
    from congklak import native,quality_search,puct_cpu_inference,ordinary_value_pilot,value_learning
    from congklak.neural_relay import NeuralNative,Model
    module=importlib.import_module('congklak.qstability_weighting')
    directories=[Path(p).resolve() for p in spec['watch_directories']]
    watched=set().union(*(files(p) for p in directories))
    watched.update((ROOT/p).resolve() for p in spec.get('extra_files',[]))
    if spec.get('watch_baseline'):watched.update((ROOT/p).resolve() for p in read(ROOT/'build/qstability-baseline.json')['inputs'])
    before=snapshot(watched);inventory={name(p) for root in directories for p in files(root)};results=[]
    with ExitStack() as stack:
        # Saved-report Native normalization/apply may repeat. New teacher search,
        # inference and optimizer entry points may not run on completed data.
        stack.enter_context(mock.patch.object(native.Native,'analyze',forbidden))
        stack.enter_context(mock.patch.object(native.Control,'__init__',forbidden))
        stack.enter_context(mock.patch.object(module.history,'execute',forbidden))
        for cls in (NeuralNative,puct_cpu_inference.CpuBatchEvaluator):stack.enter_context(mock.patch.object(cls,'__init__',forbidden))
        for cls,attrs in ((NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),
                          (Model,('reference',)),(puct_cpu_inference.CpuBatchEvaluator,('evaluate_many','feature_vector'))):
            for attr in attrs:
                if hasattr(cls,attr):stack.enter_context(mock.patch.object(cls,attr,forbidden))
        for attr in ('search','forest','_select','_backup','_expand'):stack.enter_context(mock.patch.object(quality_search,attr,forbidden))
        for owner in (ordinary_value_pilot,ordinary_value_pilot.cache,ordinary_value_pilot.joint):stack.enter_context(mock.patch.object(owner,'train',forbidden))
        stack.enter_context(mock.patch.object(value_learning,'runtime',forbidden))
        for call in spec['calls']:
            result=getattr(module,call['function'])(*call.get('args',[]),**call.get('kwargs',{}))
            results.append(dict(function=call['function'],returned=type(result).__name__))
        for call in spec.get('clis',[]):
            previous=sys.argv
            try:
                sys.argv=[str(ROOT/'tools/validate-qstability.py'),*call['argv']]
                try:runpy.run_path(sys.argv[0],run_name='__main__')
                except SystemExit as exc:assert exc.code in (None,0),str(exc.code)
            finally:sys.argv=previous
    assert before==snapshot(watched),'Durable bytes/size/mtime changed'
    assert inventory=={name(p) for root in directories for p in files(root)},'Durable inventory changed'
    assert 'torch' not in sys.modules
    return dict(schema='qstability-independent-completed-no-work-v1',passed=True,before=before,after=snapshot(watched),
        all_bytes_sizes_mtimes_unchanged=True,durable_inventory=sorted(inventory),calls=results,clis_completed=len(spec.get('clis',[])),
        new_teacher_searches=0,new_PUCT_searches=0,new_model_forwards=0,new_training_updates=0,new_games=0,new_attempt_commits=0,
        saved_transition_validation_allowed=True,zero_total_Native_calls_claim=False,torch_imported=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('spec');args=parser.parse_args();path=Path(args.spec).resolve()
    spec=read(path);proof=closed(spec);proof['inputs']={name(path):sha(path),name(__file__):sha(__file__),**{p:sha(ROOT/p) for p in spec.get('input_names',[])}}
    output=Path(spec['output']);assert not output.exists(),'Immutable worker proof exists'
    output.write_text(json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(event='qstability_independent_worker',passed=True)),flush=True)


if __name__=='__main__':main()
