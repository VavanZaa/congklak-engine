"""Verify the completed within-cell reference and assessment paths add no work."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,stratum_common as common,stratum_weighting as weighting
from congklak import quality_search,ordinary_value_pilot as pilot,puct_cpu_inference as cpu
from congklak.native import Native
from congklak.neural_relay import NeuralNative,Model


def signature(path):
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    stat=path.stat();return dict(sha256=digest,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    frozen=common.baseline(verify_all=True)
    names=set(frozen['inputs'])|set(common.read(weighting.OUTPUT)['inputs'])|{p.relative_to(ROOT).as_posix() for p in (weighting.OUTPUT,weighting.PLAN,weighting.IMPLEMENTATION,weighting.RULE)}
    def durable_files():
        return {p.relative_to(ROOT).as_posix() for folder in (weighting.DEFAULT,) for p in folder.rglob('*')
                if p.is_file() and not p.name.endswith('-shm') and (not p.name.endswith('-wal') or p.stat().st_size>0)}
    durable_before=durable_files();names.update(durable_before)
    def record():return {name:signature(ROOT/name) for name in sorted(names)}
    before=record()
    def forbidden(*a,**k):raise AssertionError('Closed stratum reopening attempted new experimental work')
    with ExitStack() as stack:
        for obj,name in ((Native,'analyze'),(NeuralNative,'analyze'),(Model,'reference'),(quality_search,'search'),(quality_search,'forest'),
                         (cpu,'CpuBatchEvaluator'),(weighting.history,'execute'),(pilot,'train'),(pilot,'runtime')):
            if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        print(config.canonical(dict(event='stratum_reopen_teacher_started')),flush=True)
        assert common.same(weighting.run(),weighting.load_rows()[2])
        print(config.canonical(dict(event='stratum_reopen_teacher_verified')),flush=True)
        assessment=weighting.assess();verified=weighting.verify()
        assert common.same(assessment,verified) and common.same(verified,common.read(weighting.OUTPUT))
        print(config.canonical(dict(event='stratum_reopen_assessment_verified')),flush=True)
    after=record();assert before==after and durable_before==durable_files() and 'torch' not in sys.modules
    common.publish(ROOT/'build/stratum-rerun.json',dict(schema='stratum-completed-rerun-v1',passed=True,before=before,after=after,
        all_bytes_sizes_mtimes_unchanged=True,producer_durable_file_inventory_unchanged=True,old_inputs_preserved=len(frozen['inputs']),
        closed_paths_verified=2,new_training_updates=0,new_teacher_searches=0,new_PUCT_searches=0,new_inference_runtime_activations=0,new_games=0,torch_imported=False,
        excluded_ephemeral_files='SQLite -shm and absent/empty -wal only; nonempty WAL and writer locks retained',
        sources=common.pins(['tools/verify-stratum-rerun.py','congklak/stratum_weighting.py','congklak/stratum_metrics.py','congklak/stratum_common.py'])))
    print(config.canonical(dict(event='stratum_completed_no_work_verified',passed=True,old_inputs=len(frozen['inputs']))),flush=True)


if __name__=='__main__':main()
