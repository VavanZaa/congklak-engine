"""No-Torch isolated reexecution of the immutable saved-data review."""
from __future__ import annotations
import importlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,quality_search,ordinary_value_pilot as learning,ordinary_value_benchmark as benchmark,ordinary_value_followup as followup
from congklak.native import Native
from congklak.neural_relay import NeuralNative
from data_pilot_independent_audit import sha,load,save,pins


def signature(path):
    path=Path(path);return dict(sha256=sha(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Review attempted Torch import')


def main():
    specpath=Path(sys.argv[1]);spec=load(specpath);baseline=load(ROOT/'build/ordinary-value-review-baseline.json')
    assert len(baseline['inputs'])==3779 and 'torch' not in sys.modules
    for name,digest in baseline['inputs'].items():assert sha(ROOT/name)==digest,name
    finder=NoTorch();sys.meta_path.insert(0,finder)
    watched=[ROOT/name for name in spec['reports']];before={str(path.relative_to(ROOT)).replace('\\','/'):signature(path) for path in watched};results=[]
    def prohibited(*args,**kwargs):raise AssertionError('Saved-data review attempted new search, training or old producer publication')
    with mock.patch.object(Native,'analyze',prohibited),mock.patch.object(NeuralNative,'analyze',prohibited),\
         mock.patch.object(quality_search,'search',prohibited),mock.patch.object(quality_search,'forest',prohibited),\
         mock.patch.object(learning,'runtime',prohibited),mock.patch.object(learning,'prepare',prohibited),\
         mock.patch.object(learning,'run_labels',prohibited),mock.patch.object(learning,'train',prohibited),\
         mock.patch.object(learning,'inspect_run',prohibited),mock.patch.object(benchmark,'run',prohibited),\
         mock.patch.object(benchmark,'decide',prohibited),mock.patch.object(benchmark,'puct_search',prohibited),\
         mock.patch.object(followup,'run',prohibited):
        for item in spec['modules']:
            module=importlib.import_module(item['module']);original=load(ROOT/item['report']);rebuilt=module.build()
            assert config.canonical(rebuilt)==config.canonical(original)
            module.validate_report(original);module.run(ROOT/item['report']);module.verify(ROOT/item['report'])
            argv=sys.argv;sys.argv=[str(ROOT/item['cli']),*item.get('verify_args',['verify'])]
            try:
                try:runpy.run_path(str(ROOT/item['cli']),run_name='__main__')
                except SystemExit as exc:assert exc.code in (None,0)
            finally:sys.argv=argv
            fixture=Path(spec['fixture_directory'])/(item['module'].split('.')[-1]+'.json');assert not fixture.exists()
            module.run(fixture);assert load(fixture)==original
            first=signature(fixture);module.run(fixture);module.verify(fixture);assert signature(fixture)==first
            results.append(dict(module=item['module'],completed_API_CLI_rebuild_equal=True,
                missing_output_rebuilt_in_new_fixture=True,fixture=str(fixture.relative_to(ROOT)).replace('\\','/')))
    after={str(path.relative_to(ROOT)).replace('\\','/'):signature(path) for path in watched};assert before==after
    for name,digest in baseline['inputs'].items():assert sha(ROOT/name)==digest,name
    assert 'torch' not in sys.modules
    save(spec['output'],dict(schema='ordinary-value-review-independent-no-work-v1',passed=True,
        before=before,after=after,modules=results,all_bytes_sizes_mtimes_unchanged=True,
        old3779_inputs_current=True,new_teacher_PUCT_searches=0,new_training_steps=0,new_strength_or_selfplay_games=0,
        old_producer_publications=0,torch_imported=False,literal_CPU_replay_and_forward_validation_allowed=True,
        recovery='A missing deterministic report rebuilt from immutable evidence in a new fixture; no incremental journal or actual process kill claimed',
        inputs=pins([Path(__file__),specpath,ROOT/'build/ordinary-value-review-baseline.json']+[ROOT/item['cli'] for item in spec['modules']])) )
    print(config.canonical(dict(event='ordinary_value_review_independent_worker',passed=True,modules=len(results))),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
