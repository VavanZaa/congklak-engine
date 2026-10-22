"""Prove completed regret analysis adds no cases or engine/training activity."""
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
            raise AssertionError('Torch is forbidden in regret completed reopening')
        return None
sys.meta_path.insert(0,NoTorch())
from congklak import config,regret_common as common,regret_drift as drift
from congklak import native,quality_search,ordinary_value_pilot as pilot,puct_cpu_inference as cpu,value_learning
from congklak.neural_relay import NeuralNative,Model


def signature(path):
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    stat=path.stat();return dict(sha256=digest,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    frozen=common.baseline(verify_all=True)
    names=set(frozen['inputs'])|set(common.read(drift.OUTPUT)['inputs'])|{p.relative_to(ROOT).as_posix() for p in (drift.OUTPUT,drift.PANEL,drift.PLAN,drift.IMPLEMENTATION)}
    def durable_files():
        return {p.relative_to(ROOT).as_posix() for p in drift.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith('-shm') and (not p.name.endswith('-wal') or p.stat().st_size>0)}
    durable_before=durable_files();names.update(durable_before)
    def record():return {name:signature(ROOT/name) for name in sorted(names)}
    before=record();events=[]
    def forbidden(*a,**k):raise AssertionError('Closed regret reopening attempted runtime or training work')
    with ExitStack() as stack:
        for cls in (native.Native,native.Control,NeuralNative,cpu.CpuBatchEvaluator):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for owner in (pilot.cache,pilot.joint):stack.enter_context(patch.object(owner,'train',forbidden))
        stack.enter_context(patch.object(value_learning,'runtime',forbidden))
        for obj,attrs in ((native.Native,('analyze','normalize','apply','validate')),(NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),(Model,('reference',)),(cpu.CpuBatchEvaluator,('evaluate_many','feature_vector')),(quality_search,('search','forest','_select','_backup','_expand')),(pilot,('train','runtime'))):
            for name in attrs:
                if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        manifest=drift.run(on_event=events.append)
        assert common.same(manifest,drift.load_rows()[1])
        assert common.same(drift.assess(),drift.verify()) and common.same(drift.verify(),common.read(drift.OUTPUT))
    assert [e['event'] for e in events]==['regret_drift_complete'],events
    after=record();assert before==after and durable_before==durable_files() and 'torch' not in sys.modules
    common.publish(ROOT/'build/regret-rerun.json',dict(schema='regret-completed-rerun-v1',passed=True,before=before,after=after,all_bytes_sizes_mtimes_unchanged=True,producer_durable_file_inventory_unchanged=True,old_inputs_preserved=len(frozen['inputs']),closed_paths_verified=3,events=events,new_case_commits=0,native_calls=0,new_training_updates=0,new_teacher_searches=0,new_PUCT_searches=0,new_model_forwards=0,new_games=0,torch_imported=False,pure_semantic_validation_replay_allowed=True,excluded_ephemeral_files='SQLite -shm and absent/empty -wal only; nonempty WAL and writer locks retained',sources=common.pins(['tools/verify-regret-rerun.py','congklak/regret_drift.py','congklak/regret_forensics.py','congklak/regret_common.py'])))
    print(config.canonical(dict(event='regret_completed_no_new_work_verified',passed=True,old_inputs=len(frozen['inputs']))),flush=True)


if __name__=='__main__':main()
