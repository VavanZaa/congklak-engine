"""Prove certified completed access performs no neural or feature work."""
from contextlib import ExitStack
import hashlib
import importlib.abc
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in completed diagnostics')
        return None


def signature(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    stat=path.stat();return dict(sha256=digest,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    sys.meta_path.insert(0,NoTorch())
    from congklak import config,neurodiag as p,neurodiag_v2 as reader,neurodiag_data as data,neurodiag_metrics as m
    from congklak import native,quality_search,ordinary_value_pilot as pilot,puct_cpu_inference as cpu,value_learning,features_relay
    from congklak.neural_relay import NeuralNative,Model
    import numpy as np
    common=p.common;frozen=common.baseline(verify_all=True)
    names=set(frozen['inputs'])|set(common.read(p.OUTPUT)['inputs'])
    names.update(path.relative_to(ROOT).as_posix() for path in (p.OUTPUT,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,p.INPUTS,
        ROOT/'build/neurodiag-audit.json',ROOT/'build/neurodiag-production.json',ROOT/'build/neurodiag-input-preparation.json'))
    names.update(reader.SOURCES);names.add(reader.FREEZE.relative_to(ROOT).as_posix())
    def durable_files():
        return {path.relative_to(ROOT).as_posix() for path in p.DEFAULT.rglob('*') if path.is_file()
            and not path.name.endswith('-shm') and (not path.name.endswith('-wal') or path.stat().st_size>0)}
    durable_before=durable_files();names.update(durable_before)
    def record():return {name:signature(ROOT/name) for name in sorted(names)}
    before=record();events=[]
    def forbidden(*a,**k):raise AssertionError('Certified reopen attempted model/feature/numeric/search/training work')
    trapped=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(p.phase6_store.Work,'__init__',forbidden));trapped.append('phase6_store.Work.__init__')
        stack.enter_context(patch.object(p,'run',forbidden));trapped.append('neurodiag.run')
        for cls in (native.Native,native.Control,NeuralNative,cpu.CpuBatchEvaluator,Model):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for obj,attrs in (
            (native.Native,('analyze','normalize','apply','validate')),
            (NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),(Model,('reference',)),
            (cpu.CpuBatchEvaluator,('evaluate_many','feature_vector')),
            (quality_search,('search','forest','_select','_backup','_expand')),
            (pilot,('train','runtime')),(pilot.cache,('train',)),(pilot.joint,('train',)),(value_learning,('runtime',)),
            (features_relay,('encode','_native')),
            (m,('load_models','prepare_models','observe','forward_arrays','head_diagnostics','target_diagnostics','scalar_forward',
                'literal_features','validate_features','numpy_runtime','runtime_metadata','summarize_cached','projection_diagnostic','feature_geometry')),
            (data,('resolve_inputs','cache_features','source_item')),(np.linalg,('svd',))):
            for name in attrs:
                if hasattr(obj,name):
                    stack.enter_context(patch.object(obj,name,forbidden));trapped.append(getattr(obj,'__name__',type(obj).__name__)+'.'+name)
        manifest=reader.run(on_event=events.append)
        common.require(common.same(manifest,reader.raw_complete()['manifest']),'Completed manifest changed')
        assessed=reader.assess();verified=reader.verify()
        common.require(common.same(assessed,verified) and common.same(verified,common.read(p.OUTPUT)),'Completed report changed')
    common.require([event['event'] for event in events]==['neurodiag_already_complete'],'Unexpected completed events')
    after=record()
    common.require(before==after and durable_before==durable_files() and 'torch' not in sys.modules,'Completed durable evidence changed')
    common.publish(ROOT/'build/neurodiag-rerun.json',dict(schema='neurodiag-completed-rerun-v1',passed=True,before=before,after=after,
        all_bytes_sizes_mtimes_unchanged=True,producer_durable_file_inventory_unchanged=True,old_inputs_preserved=len(frozen['inputs']),
        closed_paths_verified=['neurodiag_v2.run','neurodiag_v2.raw_complete','neurodiag_v2.assess','neurodiag_v2.verify'],events=events,trapped_calls=trapped,
        certified_completed_writer_lock_acquisitions=0,versioned_reader=2,
        new_observation_commits=0,new_batch_commits=0,native_calls=0,new_training_updates=0,new_teacher_searches=0,new_PUCT_searches=0,
        new_model_forwards=0,new_derivatives=0,new_SVDs=0,new_summary_calls=0,new_feature_encodings=0,new_games=0,torch_imported=False,
        pure_semantic_validation_replay_allowed=True,
        excluded_ephemeral_files='SQLite -shm and absent/empty -wal only; nonempty WAL and writer locks retained',
        sources=common.pins([__file__,'congklak/neurodiag.py','congklak/neurodiag_v2.py','tools/diagnose-neural-v2.py',reader.FREEZE,
            'congklak/neurodiag_data.py','congklak/neurodiag_metrics.py','congklak/neurodiag_common.py'])))
    print(config.canonical(dict(event='neurodiag_completed_no_new_work_verified',passed=True,old_inputs=len(frozen['inputs']))),flush=True)


if __name__=='__main__':main()
