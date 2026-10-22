"""The sole full frozen-panel observation pass, with explicit runtime counters."""
from contextlib import ExitStack
import importlib.abc
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Torch is forbidden in frozen diagnostics')
        return None


def main():
    sys.meta_path.insert(0,NoTorch())
    from congklak import config,neurodiag as p,neurodiag_data as data,neurodiag_metrics as m
    from congklak import native,quality_search,puct_cpu_inference as cpu,ordinary_value_pilot as pilot,value_learning,features_relay
    from congklak.neural_relay import NeuralNative,Model
    require=p.require;require(not p.DEFAULT.exists(),'Full collection already exists; resume its public collect command')
    pack=data.load_inputs();n=len(pack['units']);before=p.common.pins([p.common.CONTROL,p.common.CANDIDATE])
    seen=[];counts=dict(observe_batches=0,shared_trunk_forwards=0,value_head_evaluations=0,target_diagnostics=0,summary_calls=0)
    originals={name:getattr(m,name) for name in ('observe','forward_arrays','head_diagnostics','target_diagnostics','summarize_cached')}
    def observe(units,models):
        require(not(set(units)&set(seen)),'Production repeated an observed unit');seen.extend(sorted(units));counts['observe_batches']+=1
        return originals['observe'](units,models)
    def counted(name,key):
        def call(*a,**k):counts[key]+=1;return originals[name](*a,**k)
        return call
    def forbidden(*a,**k):raise AssertionError('Frozen production attempted Native/search/training or fresh feature work')
    with ExitStack() as stack:
        for cls in (native.Native,native.Control,NeuralNative,cpu.CpuBatchEvaluator,Model):stack.enter_context(patch.object(cls,'__init__',forbidden))
        for obj,attrs in ((native.Native,('analyze','normalize','apply','validate')),(NeuralNative,('analyze','evaluate','measure_inference','neural_analyze')),
            (Model,('reference',)),(cpu.CpuBatchEvaluator,('evaluate_many','feature_vector')),(quality_search,('search','forest','_select','_backup','_expand')),
            (pilot,('train','runtime')),(pilot.cache,('train',)),(pilot.joint,('train',)),(value_learning,('runtime',)),
            (features_relay,('encode','_native')),(m,('literal_features',)),(data,('resolve_inputs','cache_features','source_item'))):
            for name in attrs:
                if hasattr(obj,name):stack.enter_context(patch.object(obj,name,forbidden))
        stack.enter_context(patch.object(m,'observe',observe))
        for name,key in (('forward_arrays','shared_trunk_forwards'),('head_diagnostics','value_head_evaluations'),('target_diagnostics','target_diagnostics'),('summarize_cached','summary_calls')):
            stack.enter_context(patch.object(m,name,counted(name,key)))
        sys.argv=[str(ROOT/'tools/diagnose-neural.py'),'collect']
        try:runpy.run_path(sys.argv[0],run_name='__main__')
        except SystemExit as exc:require(exc.code in (None,0),'Public collector failed')
        state=p.raw_complete()
    require(seen==sorted(pack['units']) and counts==dict(observe_batches=(n+15)//16,shared_trunk_forwards=n,value_head_evaluations=2*n,target_diagnostics=2*pack['counts']['target_records'],summary_calls=1),'Actual forward counters differ')
    require(before==p.common.pins([p.common.CONTROL,p.common.CANDIDATE]) and 'torch' not in sys.modules,'Fixed model contract differs')
    result=dict(schema='neurodiag-guarded-full-production-v1',passed=True,models_fixed=True,policy_fixed=True,public_CLI_used=True,
        unit_count=n,target_records=pack['counts']['target_records'],batch_count=state['manifest']['batch_count'],observed_unit_ids=seen,
        actual_calls=counts,native_calls=0,new_teacher_searches=0,new_PUCT_searches=0,new_games=0,optimizer_updates=0,torch_imported=False,
        fixture_work_included=False,engineering_fixture_forwards_recorded_separately=True,
        inputs=p.common.pins([__file__,'tools/diagnose-neural.py','build/neurodiag-input-preparation.json',p.INPUTS,p.PLAN,p.IMPLEMENTATION,p.SOURCE_FREEZE,
            'build/neurodiag-audit/preflight.json',p.DEFAULT/'work.sqlite3',p.DEFAULT/'manifest.json',p.DEFAULT/'summary.json',p.common.CONTROL,p.common.CANDIDATE]))
    p.publish(ROOT/'build/neurodiag-production.json',result)
    print(config.canonical(dict(event='neurodiag_full_collection_guarded',passed=True,units=n,actual_calls=counts)),flush=True)


if __name__=='__main__':main()
