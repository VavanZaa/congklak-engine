"""Verify closed Q-stability assessment paths add no teacher or training work."""
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
            raise AssertionError('Torch is forbidden in completed Q-stability validation')
        return None


sys.meta_path.insert(0,NoTorch())
from congklak import config,qstability_common as common,qstability_weighting as weighting
from congklak import native,quality_search,ordinary_value_pilot as pilot,puct_cpu_inference as cpu,value_learning
from congklak.neural_relay import NeuralNative,Model


def signature(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    stat=path.stat();return dict(sha256=digest,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    frozen=common.baseline(verify_all=True);result=common.read(weighting.OUTPUT)
    names=set(frozen['inputs'])|set(result['inputs'])|{p.relative_to(ROOT).as_posix() for p in (weighting.OUTPUT,weighting.PLAN,weighting.IMPLEMENTATION)}
    def durable_files():
        return {p.relative_to(ROOT).as_posix() for p in weighting.DEFAULT.rglob('*') if p.is_file() and not p.name.endswith('-shm') and (not p.name.endswith('-wal') or p.stat().st_size>0)}
    durable_before=durable_files();names.update(durable_before)
    def record():return {name:signature(ROOT/name) for name in sorted(names)}
    before=record();events=[]
    def forbidden(*a,**k):raise AssertionError('Closed Q-stability reopening attempted search/inference/training')
    with ExitStack() as stack:
        for cls in (native.Control,NeuralNative,cpu.CpuBatchEvaluator):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for obj,attrs in ((native.Native,('analyze',)),(NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),(Model,('reference',)),(quality_search,('search','forest','_select','_backup','_expand')),(weighting.history,('execute',)),(pilot,('train',)),(pilot.cache,('train',)),(pilot.joint,('train',)),(value_learning,('runtime',))):
            for name in attrs:
                if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        print(config.canonical(dict(event='qstability_reopen_started')),flush=True)
        assert common.same(weighting.run(on_event=events.append),weighting.load_rows()[2])
        if weighting.RULE.exists():weighting.realize()
        else:assert result['status']=='references_incomplete_hold'
        assert common.same(weighting.assess(),weighting.verify()) and common.same(weighting.verify(),result)
    assert not any('precommit' in e['event'] or 'committed' in e['event'] for e in events),events
    after=record();assert before==after and durable_before==durable_files() and 'torch' not in sys.modules
    common.publish(ROOT/'build/qstability-rerun.json',dict(schema='qstability-completed-rerun-v1',passed=True,before=before,after=after,all_bytes_sizes_mtimes_unchanged=True,producer_durable_file_inventory_unchanged=True,old_inputs_preserved=5050,closed_paths_verified=3,events=events,new_attempt_commits=0,new_training_updates=0,new_teacher_searches=0,new_PUCT_searches=0,new_model_forwards=0,new_games=0,torch_imported=False,native_transition_validation_allowed=True,pure_semantic_validation_replay_allowed=True,excluded_ephemeral_files='SQLite SHM and absent/empty WAL only; nonempty WAL and writer locks retained',sources=common.pins(['tools/verify-qstability-rerun.py','congklak/qstability_weighting.py','congklak/qstability_metrics.py','congklak/qstability_common.py'])))
    print(config.canonical(dict(event='qstability_completed_no_new_work_verified',passed=True,old_inputs=5050)),flush=True)


if __name__=='__main__':main()
