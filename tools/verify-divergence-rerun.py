"""Prove completed search/label/comparison reopening performs no new work."""
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from congklak import config,divergence_study_common as common,divergence_search as search,divergence_teacher as teacher,divergence_comparison as comparison
from congklak import quality_search,ordinary_value_pilot as pilot
from congklak.native import Native
from congklak.neural_relay import NeuralNative


def signature(path):
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    stat=path.stat();return dict(sha256=digest,size=stat.st_size,mtime_ns=stat.st_mtime_ns)


def main():
    frozen=common.baseline(verify_all=True)
    names=set(frozen['inputs'])|set(common.read(comparison.OUTPUT)['inputs'])|{comparison.OUTPUT.relative_to(ROOT).as_posix()}
    def durable_files():
        return {p.relative_to(ROOT).as_posix() for folder in (search.DEFAULT,teacher.DEFAULT) for p in folder.rglob('*')
                if p.is_file() and not p.name.endswith('-shm') and (not p.name.endswith('-wal') or p.stat().st_size>0)}
    durable_before=durable_files();names.update(durable_before)
    def record():return {name:signature(ROOT/name) for name in sorted(names)}
    before=record()
    def forbidden(*a,**k):raise AssertionError('Completed divergence reopening attempted new work')
    with ExitStack() as stack:
        for obj,name in ((Native,'analyze'),(NeuralNative,'analyze'),(quality_search,'search'),(quality_search,'forest'),(teacher,'execute'),(pilot,'train'),(pilot,'runtime')):
            if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        assert search.run()==search.verify()==common.read(search.OUTPUT)
        assert teacher.run()==teacher.load_rows()[2]==common.read(teacher.OUTPUT)
        assert comparison.run()==comparison.verify()==common.read(comparison.OUTPUT)
    after=record();assert before==after and durable_before==durable_files() and 'torch' not in sys.modules
    body=dict(schema='divergence-completed-rerun-v1',passed=True,before=before,after=after,
        all_bytes_sizes_mtimes_unchanged=True,old_inputs_preserved=len(frozen['inputs']),reports_rebuilt_and_verified=3,
        new_training_updates=0,new_teacher_searches=0,new_PUCT_searches=0,new_games=0,torch_imported=False,
        producer_durable_file_inventory_unchanged=True,excluded_ephemeral_files='SQLite -shm and absent/empty -wal only; nonempty WAL retained; writer.lock and all other producer files included',
        sources=common.pins(['tools/verify-divergence-rerun.py','congklak/divergence_search.py','congklak/divergence_teacher.py','congklak/divergence_comparison.py']))
    common.publish(ROOT/'build/divergence-rerun.json',body)
    print(config.canonical(dict(event='divergence_completed_no_work_verified',passed=True,old_inputs=len(frozen['inputs']))),flush=True)


if __name__=='__main__':main()
