"""Isolated deterministic divergence replay and completed no-work checks."""
from __future__ import annotations
from contextlib import ExitStack
import importlib.abc,json,math,runpy,sys
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from congklak import config,quality_search,divergence_study_common as common
from congklak.native import Native
from congklak.neural_relay import NeuralNative,Model
from congklak.puct_cpu_inference import CpuBatchEvaluator
from congklak.timing_v2 import PerformanceCore
from data_pilot_independent_audit import sha,load,save,pins,norm,position
from selfplay_pilot_independent_audit import literal_puct
from stage5_deeper_audit_acceptance import value_and_policy


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='torch' or fullname.startswith('torch.'):raise AssertionError('Divergence audit attempted Torch')


def signature(path):
    path=Path(path);return dict(sha256=sha(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)


def files(path):
    return [p for p in Path(path).rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))]


def close(a,b):assert abs(a-b)<=1e-10,(a,b)


def literal_cells(spec):
    from congklak import divergence_search as p
    roots,cells,manifest=p.load_cells(spec['run'],spec.get('fixture',False),spec.get('variant','normal'))
    modelpaths={'control':common.CONTROL,'candidate':common.CANDIDATE};references=[];rootparity=[]
    # Both backends must be numerically identical to the recorded execution.
    with ExitStack() as stack:
        cpu=stack.enter_context(PerformanceCore(manifest['identity']['cpu_declared']));cpu.verify()
        evaluators={name:stack.enter_context(CpuBatchEvaluator(path,cache_capacity=0,max_batch=1,backend='auto')) for name,path in modelpaths.items()}
        for root in roots:
            for name,path in modelpaths.items():
                native=evaluators[name].evaluate_many([root['state']])[0];value,units,logits=value_and_policy(norm(root['state']),load(path)['model']);close(value,native['value_player0'])
                for a,b in zip(logits,native['policy_logits']):close(a,b)
                rootparity.append(dict(state_key=root['state_key'],profile=name,literal_value=value,native_value=native['value_player0'],policy_max_abs_error=max(abs(a-b) for a,b in zip(logits,native['policy_logits']))))
        for cell in cells:
            unit=cell['unit'];root=roots[unit['root_index']];assert cell['status']=='complete';ev=evaluators[unit['profile']];ev.reset_cache()
            def predict(state):
                answer=ev.evaluate_many([state])[0];return answer['value_player0'],answer['policy_logits']
            import random
            tree,nodes=literal_puct(root['state'],unit['simulations'],random.Random(0),predict,c_puct=.25)
            raw=cell['report'];n=sum(tree['visits']);assert n==unit['simulations']==raw['completed_simulations'];assert nodes==raw['nodes'] and tree['visits']==raw['visits']
            legal=list(__import__('reference').legal_actions(tree['q']));chosen=min(legal,key=lambda a:(-tree['visits'][a],a));assert chosen==raw['chosen']
            means=[98*tree['sums'][a]/tree['visits'][a] if tree['visits'][a] else None for a in range(7)]
            for a in range(7):
                close(tree['priors'][a],raw['root_priors'][a]);close(tree['visits'][a]/n,raw['policy'][a])
                if means[a] is None:assert raw['action_mean_value_player0'][a] is None
                else:close(means[a],raw['action_mean_value_player0'][a])
            close(98*tree['bootstrap'],raw['unvisited_root_q_player0']);close(98*sum(tree['sums'])/n,raw['estimated_value_player0'])
            references.append(dict(unit=unit,nodes=nodes,visits=tree['visits'],chosen=chosen,root_priors=tree['priors'],action_mean_value_player0=means,estimated_value_player0=98*sum(tree['sums'])/n))
    return dict(schema='divergence-independent-literal-fixedwork-v1',passed=True,cells=len(cells),root_simulations=sum(r['unit']['simulations'] for r in references),references=references,root_model_parity=rootparity,search_selection_backup_and_seed_transitions_independent=True,leaf_evaluator='Same frozen native CPU numerical backend; root predictions independently checked with literal relay encoder and dense network',new_teacher_searches=0,new_strength_games=0,training_steps=0,torch_imported=False)


def closed(spec):
    from congklak import divergence_search as search,divergence_teacher as teacher,divergence_comparison as comparison
    path=Path(spec['run']);kind=spec['producer'];fixture=spec.get('fixture',False);variant=spec.get('variant','normal')
    watched=files(path)+[ROOT/name for name in spec.get('extra_files',[])]
    if spec.get('watch_baseline'):watched +=[ROOT/name for name in common.baseline()['inputs']]
    watched=list(dict.fromkeys(watched));before={str(p.relative_to(ROOT)).replace('\\','/'):signature(p) for p in watched}
    def prohibited(*args,**kwargs):raise AssertionError('Completed divergence reopening attempted new search or runtime work')
    with ExitStack() as stack:
        for owner,name in ((Native,'analyze'),(NeuralNative,'analyze'),(quality_search,'search'),(quality_search,'forest'),(teacher,'execute')):stack.enter_context(mock.patch.object(owner,name,prohibited))
        if kind=='search':
            stack.enter_context(mock.patch.object(search,'CpuBatchEvaluator',prohibited));stack.enter_context(mock.patch.object(search,'PerformanceCore',prohibited))
            expected=search.verify(path,fixture,variant);actual=search.run(path,fixture,variant);assert actual==expected;search.verify(path,fixture,variant)
            cli=ROOT/'tools/probe-divergence-search.py';argv=[str(cli),'verify','--path',str(path),'--variant',variant]+(['--fixture']if fixture else [])
        elif kind=='teacher':
            expected=teacher.load_rows(path)[2];actual=teacher.run(path,fixture=fixture);assert actual==expected;teacher.load_rows(path)
            cli=ROOT/'tools/label-divergence-actions.py';argv=[str(cli),'verify','--run',str(path)]+(['--fixture']if fixture else [])
        else:
            stack.enter_context(mock.patch.object(search,'CpuBatchEvaluator',prohibited));stack.enter_context(mock.patch.object(search,'PerformanceCore',prohibited))
            expected=comparison.verify();assert comparison.run()==expected;comparison.verify()
            cli=ROOT/'tools/compare-divergence.py';argv=[str(cli),'verify']
        original=sys.argv;sys.argv=argv
        try:
            try:runpy.run_path(str(cli),run_name='__main__')
            except SystemExit as exc:assert exc.code in (0,None)
        finally:sys.argv=original
    after={str(p.relative_to(ROOT)).replace('\\','/'):signature(p) for p in watched};assert before==after
    return dict(schema='divergence-independent-completed-no-work-v1',passed=True,producer=kind,before=before,after=after,all_bytes_sizes_mtimes_unchanged=True,API_CLI_verified=True,new_teacher_searches=0,new_PUCT_searches=0,new_inference_runtime_activations=0,new_strength_games=0,training_steps=0,torch_imported=False)


def main():
    specpath=Path(sys.argv[1]);spec=load(specpath);sys.meta_path.insert(0,NoTorch());assert 'torch' not in sys.modules
    result=literal_cells(spec) if spec['kind']=='literal_cells' else closed(spec)
    assert 'torch' not in sys.modules
    result['inputs']=pins([Path(__file__),specpath,common.BASELINE,common.PANEL,ROOT/'congklak/divergence_search.py',ROOT/'congklak/divergence_teacher.py',ROOT/'congklak/divergence_comparison.py',ROOT/'tests/selfplay_pilot_independent_audit.py',ROOT/'tests/reference.py',ROOT/'tests/stage5_deeper_audit_acceptance.py']+[ROOT/name for name in spec.get('input_names',[])])
    save(spec['output'],result);print(config.canonical(dict(event='divergence_independent_worker_complete',kind=spec['kind'],passed=True)),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
