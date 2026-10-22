"""Package the frozen eligible panel under no-forward/no-search/no-training traps."""
from contextlib import ExitStack
import importlib.abc
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch is forbidden in neural diagnostics')
        return None


def main():
    sys.meta_path.insert(0,NoTorch())
    from congklak import config,neurodiag as p,neurodiag_data as data,neurodiag_metrics as m
    from congklak import native,quality_search,puct_cpu_inference as cpu,ordinary_value_pilot as pilot,value_learning,features_relay
    from congklak.neural_relay import NeuralNative,Model
    require=p.require;require(not p.INPUTS.exists(),'Input preparation already exists; use validated existing inputs')
    p.common.baseline(verify_all=True);before=p.common.pins([p.common.CONTROL,p.common.CANDIDATE]);calls=[]
    def forbidden(*a,**k):raise AssertionError('Input preparation attempted model/search/training work')
    original=m.literal_features
    def literal(state):calls.append(p.common.rawkey(state));return original(state)
    with ExitStack() as stack:
        for cls in (native.Native,native.Control,NeuralNative,cpu.CpuBatchEvaluator):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for obj,attrs in ((native.Native,('analyze','normalize','apply','validate')),(NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),
            (Model,('reference',)),(cpu.CpuBatchEvaluator,('evaluate_many','feature_vector')),(quality_search,('search','forest','_select','_backup','_expand')),
            (pilot,('train','runtime')),(pilot.cache,('train',)),(pilot.joint,('train',)),(value_learning,('runtime',)),
            (features_relay,('encode','_native')),(m,('load_models','prepare_models','observe','forward_arrays','head_diagnostics','target_diagnostics','scalar_forward','summarize_cached','projection_diagnostic','feature_geometry'))):
            for name in attrs:
                if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        stack.enter_context(patch.object(m,'literal_features',literal));pack=p.prepare_inputs()
    require(len(calls)==len(set(calls))==28 and set(calls)=={r['state_key'] for r in p.read(p.PANEL)['paired']},'Literal feature scope differs')
    require(before==p.common.pins([p.common.CONTROL,p.common.CANDIDATE]) and 'torch' not in sys.modules,'Fixed model/Torch contract differs')
    result=dict(schema='neurodiag-guarded-input-preparation-v1',passed=True,models_fixed=True,policy_fixed=True,
        counts=pack['counts'],literal_encoder_state_keys=calls,model_forwards=0,native_calls=0,new_teacher_searches=0,new_PUCT_searches=0,
        new_games=0,optimizer_updates=0,torch_imported=False,
        inputs=p.common.pins([__file__,p.INPUTS,p.PANEL,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,'congklak/neurodiag.py','congklak/neurodiag_data.py','congklak/neurodiag_metrics.py',p.common.CONTROL,p.common.CANDIDATE]))
    p.publish(ROOT/'build/neurodiag-input-preparation.json',result)
    print(config.canonical(dict(event='neurodiag_inputs_guarded',passed=True,counts=pack['counts'])),flush=True)


if __name__=='__main__':main()
