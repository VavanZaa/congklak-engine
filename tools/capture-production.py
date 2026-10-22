"""Run the four saved-data cases through the public CLI with runtime traps."""
from contextlib import ExitStack
import importlib.abc
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch is forbidden in saved-data production')
        return None


def main():
    sys.meta_path.insert(0,NoTorch())
    from congklak import config,capture_common as common,capture_drift as drift
    from congklak import native,quality_search,puct_cpu_inference as cpu,ordinary_value_pilot as pilot,value_learning
    from congklak.neural_relay import NeuralNative,Model
    assert not drift.DEFAULT.exists(),'Four-case launch already exists; use the recoverable public CLI to continue it'
    def forbidden(*a,**k):raise AssertionError('Capture production attempted runtime/search/inference/training')
    with ExitStack() as stack:
        for cls in (native.Native,native.Control,NeuralNative,cpu.CpuBatchEvaluator):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for obj,attrs in ((native.Native,('analyze','normalize','apply','validate')),(NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),(Model,('reference',)),(cpu.CpuBatchEvaluator,('evaluate_many','feature_vector')),(quality_search,('search','forest','_select','_backup','_expand')),(pilot,('train',)),(pilot.cache,('train',)),(pilot.joint,('train',)),(value_learning,('runtime',))):
            for name in attrs:
                if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        sys.argv=[str(ROOT/'tools/investigate-capture-drift.py'),'all']
        try:runpy.run_path(sys.argv[0],run_name='__main__')
        except SystemExit as exc:assert exc.code in (None,0),exc.code
        result=drift.verify()
    assert result['cases']==4 and 'torch' not in sys.modules
    common.publish(ROOT/'build/capture-production.json',dict(schema='capture-guarded-production-v1',passed=True,cases=4,new_case_commits=4,native_calls=0,new_model_forwards=0,new_teacher_searches=0,new_PUCT_searches=0,new_training_updates=0,new_games=0,torch_imported=False,models_fixed=True,policy_fixed=True,public_CLI_used=True,inputs=common.pins(['tools/capture-production.py','tools/investigate-capture-drift.py','build/capture-drift-implementation.json','build/capture-audit/preflight.json','build/capture-drift-result.json'])))
    print(config.canonical(dict(event='capture_guarded_production_complete',passed=True,cases=4)),flush=True)


if __name__=='__main__':main()
